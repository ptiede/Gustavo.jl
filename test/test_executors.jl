# ── The executor seam (M7) ────────────────────────────────────────────────────
#
# WHO runs the tasks is a per-run backend choice (`ExecutionConfig(executor)`,
# ambient `with_executor`); WHAT runs — chunking, budget admission, fold order —
# is fixed by the callers. So the contract under test is: θ and outputs are
# BIT-identical across executors, the admission semantics (ordering, clamping,
# in-flight cap) hold on both, nested spawn/fetch cannot deadlock, and errors
# surface comparably.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")
using Dagger      # activates GustavoDaggerExt (the DaggerExecutor backend)
using Gustavo.Executors
using Gustavo.Executors: exec_spawn, exec_fetch, exec_foreach

@testset "Executor seam" begin
    @testset "nested spawn/fetch on both backends" begin
        for ex in (ThreadsExecutor(), DaggerExecutor())
            # Parents that block on their children declare `blocking = true` —
            # under Dagger that spawns them at zero occupancy, so nested
            # fan-outs cannot starve even at 1–2 threads.
            r = with_executor(ex) do
                outer = [
                    exec_spawn(
                        () -> sum(exec_fetch(exec_spawn(() -> i + j)) for j in 1:4);
                        blocking = true,
                    )
                        for i in 1:8
                ]
                sum(exec_fetch.(outer))
            end
            @test r == sum(i + j for i in 1:8, j in 1:4)
            # The ambient executor survives into spawned tasks (Dagger tasks
            # re-bind it — scheduler tasks don't inherit scoped values).
            inner_ex = with_executor(() -> exec_fetch(exec_spawn(current_executor)), ex)
            @test inner_ex == ex
        end
    end

    @testset "exec_foreach chunking is executor-independent" begin
        for ex in (ThreadsExecutor(), DaggerExecutor())
            hits = zeros(Int, 23)
            with_executor(() -> exec_foreach(i -> hits[i] += 1, 1:23; ntasks = 4), ex)
            @test all(==(1), hits)
            with_executor(() -> exec_foreach(i -> hits[i] += 1, Int[]; ntasks = 4), ex)
            @test all(==(1), hits)
        end
    end

    @testset "errors rethrow the ORIGINAL exception under both" begin
        for ex in (ThreadsExecutor(), DaggerExecutor())
            err = try
                with_executor(() -> exec_fetch(exec_spawn(() -> error("boom"))), ex)
                nothing
            catch e
                e
            end
            # Threads wraps in TaskFailedException; Dagger's wrapper is
            # unwrapped by exec_fetch — both expose the ErrorException.
            inner = err isa TaskFailedException ? err.task.exception : err
            @test inner isa ErrorException && inner.msg == "boom"
        end
    end

    @testset "_scheduled_map: Dagger backend matches Threads semantics" begin
        for ex in (ThreadsExecutor(), DaggerExecutor())
            res, peak = FP._scheduled_map(
                x -> x * 10, 1:8, [10, 3, 3, 3, 1, 2, 2, 2], 12;
                max_tasks = 4, executor = ex,
            )
            @test res == [10, 20, 30, 40, 50, 60, 70, 80]   # items order
            @test 1 <= peak <= 4                            # in-flight cap held
            # Over-budget item is clamped so it still runs.
            res2, _ = FP._scheduled_map(
                x -> x + 1, 1:3, [100, 1, 1], 10; max_tasks = 4, executor = ex,
            )
            @test res2 == [2, 3, 4]
            # Worker errors propagate.
            @test_throws Exception FP._scheduled_map(
                x -> x == 2 ? error("boom") : x, 1:3, [1, 1, 1], 10;
                max_tasks = 2, executor = ex,
            )
        end
    end

    @testset "full pipeline: θ and output bit-identical across executors" begin
        uvset, _ = _build_fringe_uvset(; nscans = 2)
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
        mk(ex) = CalibrationPipeline(
            FringeFit(model = FringeModel(dispersion = false, sbd = false)),
            BandpassEstimator(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 2, executor = ex),
        )
        sol_t, out_t = fitcalibrate(mk(ThreadsExecutor()), uvset; reduce = [AverageFrequency(nout = 1)])
        sol_d, out_d = fitcalibrate(mk(DaggerExecutor()), uvset; reduce = [AverageFrequency(nout = 1)])
        @test sol_d.θ == sol_t.θ
        @test Gustavo.stage_names(sol_d) == Gustavo.stage_names(sol_t)
        for (k, leaf) in UVP.branches(out_t)
            ld = UVP.branches(out_d)[k]
            @test isequal(parent(leaf[:vis]), parent(ld[:vis]))
            @test isequal(parent(leaf[:weights]), parent(ld[:weights]))
        end

        # Standalone calibrate matches across executors too.
        c_t = calibrate(sol_t, uvset; executor = ThreadsExecutor())
        c_d = calibrate(sol_t, uvset; executor = DaggerExecutor())
        for (k, leaf) in UVP.branches(c_t)
            @test isequal(parent(leaf[:vis]), parent(UVP.branches(c_d)[k][:vis]))
        end
    end

    @testset "the driver-facing stream defaults to the ambient executor" begin
        uvset, _ = _build_fringe_uvset()
        # The bare default follows the ambient/process executor (Dagger unless
        # the suite runs under GUSTAVO_TEST_EXECUTOR=threads).
        st_d = with_executor(() -> FP.scan_stream(uvset), DaggerExecutor())
        @test st_d.executor isa DaggerExecutor
        @test FP.scan_stream(uvset).executor == current_executor()
        st_t = with_executor(() -> FP.scan_stream(uvset), ThreadsExecutor())
        @test st_t.executor isa ThreadsExecutor
        # Same search results either way (the QA drivers' path).
        grp_d = FP.materialize_cube(st_d, st_d.groups[1])
        grp_t = with_executor(
            () -> FP.materialize_cube(st_t, st_t.groups[1]), ThreadsExecutor(),
        )
        @test isequal(grp_d.Vg, grp_t.Vg) && isequal(grp_d.Wg, grp_t.Wg)
        r_d = FP.search_scan(st_d, grp_d, FP.FringeSearch(); ngroups = 1)
        r_t = with_executor(
            () -> FP.search_scan(st_t, grp_t, FP.FringeSearch(); ngroups = 1),
            ThreadsExecutor(),
        )
        @test all(r_d.det .=== r_t.det)
        @test r_d.max_snr == r_t.max_snr
    end
end
