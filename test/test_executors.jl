# ── The executor seam ─────────────────────────────────────────────────────────
#
# A run parallelizes at two independently-selected levels (see `ExecutionConfig`):
# the OUTER across-scan group scheduler (`ThreadsExecutor`/`DaggerExecutor`,
# driving the budget-admission pass runner) and the INNER within-scan fan-out
# (an OhMyThreads scheduler). The contract under test: θ and outputs are
# bit-identical across BOTH choices, the group admission semantics hold on both
# outer backends, and a failed group task surfaces its exception.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")
using Dagger      # activates GustavoDaggerExt (the DaggerExecutor backend)
using Gustavo.Executors: ThreadsExecutor, DaggerExecutor
using Gustavo: DynamicScheduler, SerialScheduler

# An outer executor with no `_scheduled_map`/`_spawn` backend, for the fallback.
struct UnbackedExecutor end

@testset "Executor seam" begin
    @testset "group admission: Dagger matches Threads" begin
        for ex in (ThreadsExecutor(), DaggerExecutor())
            res, peak = ST._scheduled_map(
                x -> x * 10, 1:8, [10, 3, 3, 3, 1, 2, 2, 2], 12;
                max_tasks = 4, executor = ex,
            )
            @test res == [10, 20, 30, 40, 50, 60, 70, 80]   # items order
            @test 1 <= peak <= 4                            # in-flight cap held
            # Over-budget item is clamped so it still runs.
            res2, _ = ST._scheduled_map(
                x -> x + 1, 1:3, [100, 1, 1], 10; max_tasks = 4, executor = ex,
            )
            @test res2 == [2, 3, 4]
            # Empty input.
            res0, peak0 = ST._scheduled_map(identity, Int[], Float64[], 10; max_tasks = 2, executor = ex)
            @test isempty(res0) && peak0 == 0
            # A failed group task rethrows its own exception (Dagger's wrappers
            # are peeled by exec_fetch, so both expose the ErrorException).
            err = try
                ST._scheduled_map(
                    x -> x == 2 ? error("boom") : x, 1:3, [1, 1, 1], 10;
                    max_tasks = 2, executor = ex,
                )
                nothing
            catch e
                e
            end
            inner = err isa TaskFailedException ? err.task.exception : err
            @test inner isa ErrorException && inner.msg == "boom"
        end
    end

    @testset "an unbacked outer executor fails fast" begin
        # No `_scheduled_map(::UnbackedExecutor, …)` method — a custom backend
        # must add one (the seam is open by dispatch, not by subtyping).
        @test_throws MethodError ST._scheduled_map(
            identity, 1:3, [1, 1, 1], 10; max_tasks = 2, executor = UnbackedExecutor(),
        )
        # The `_spawn` fallback names the missing backend; the Dagger method wins
        # when the extension is loaded.
        @test_throws "has no spawn backend loaded" Gustavo.Executors._spawn(
            UnbackedExecutor(), () -> 1, false,
        )
        @test Gustavo.Executors._spawn(DaggerExecutor(), () -> 1, false) isa Dagger.DTask
    end

    @testset "full pipeline: θ and output bit-identical across OUTER executors" begin
        uvset, _ = _build_fringe_uvset(; nscans = 2)
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
        mk(ex) = CalibrationPipeline(
            FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))),
            BandpassEstimator(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 2, outer_executor = ex),
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

        # Standalone calibrate matches across outer executors too.
        c_t = calibrate(sol_t, uvset; outer_executor = ThreadsExecutor())
        c_d = calibrate(sol_t, uvset; outer_executor = DaggerExecutor())
        for (k, leaf) in UVP.branches(c_t)
            @test isequal(parent(leaf[:vis]), parent(UVP.branches(c_d)[k][:vis]))
        end
    end

    @testset "full pipeline: θ bit-identical across INNER executors" begin
        uvset, _ = _build_fringe_uvset(; nscans = 2)
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
        mk(inner) = CalibrationPipeline(
            FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))),
            BandpassEstimator(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 2, inner_executor = inner),
        )
        # Serial vs multi-chunk within-scan fan-out: the per-block folds are
        # order-fixed by the data layout, so θ is bit-identical.
        sol_ser, _ = fitcalibrate(mk(SerialScheduler()), uvset)
        sol_dyn, _ = fitcalibrate(mk(DynamicScheduler(; nchunks = 4)), uvset)
        @test sol_ser.θ == sol_dyn.θ
    end

    @testset "the driver-facing stream carries both executors" begin
        uvset, _ = _build_fringe_uvset()
        st = FP.scan_stream(
            uvset;
            outer_executor = DaggerExecutor(), inner_executor = SerialScheduler(),
        )
        @test st.outer_executor isa DaggerExecutor
        @test st.inner_executor isa SerialScheduler
        # The bare default follows the process-wide outer default.
        st0 = FP.scan_stream(uvset)
        @test st0.outer_executor == Gustavo.Executors.DEFAULT_EXECUTOR[]
        # Same search results whichever inner executor runs the fan-out.
        stack, _ = FP.materialize_cube(st, st.groups[1])
        stack0, _ = FP.materialize_cube(st0, st0.groups[1])
        @test isequal(stack[:vis], stack0[:vis]) && isequal(stack[:weights], stack0[:weights])
        r  = FP.search_scan(stack,  st.geom,  FP.FringeSearch(); ngroups = 1)
        r0 = FP.search_scan(stack0, st0.geom, FP.FringeSearch(); ngroups = 1)
        # Same detections whichever inner executor runs the fan-out, per layer
        # (subsumes any derived aggregate like max SNR).
        @test all(all(r[k] .=== r0[k]) for k in keys(r))
    end
end
