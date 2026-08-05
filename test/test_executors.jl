# ── The executor seam ─────────────────────────────────────────────────────────
#
# A run parallelizes at two independently-selected levels (see `ExecutionConfig`):
# the OUTER across-scan group scheduler driving the pass runner, and the INNER
# within-scan fan-out — each an OhMyThreads `Scheduler`. The contract under
# test: groups are dispatched heaviest-first and run at the outer scheduler's
# own task count, θ and outputs are bit-identical across BOTH choices, and a
# failed group task surfaces its exception.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")
using Gustavo: DynamicScheduler, StaticScheduler, GreedyScheduler, SerialScheduler

# An outer executor with no `_scheduled_map` method, for the fallback.
struct UnbackedExecutor end

# The same scheduler kind at a chosen task count — the schedulers own their own
# concurrency now, so a test that needs a specific one constructs it.
_cap(::DynamicScheduler, n) = DynamicScheduler(; nchunks = n)
_cap(::StaticScheduler, n) = StaticScheduler(; nchunks = n)
_cap(::GreedyScheduler, n) = GreedyScheduler(; ntasks = n)

@testset "Executor seam" begin
    @testset "group dispatch across outer schedulers" begin
        for ex in (DynamicScheduler(), StaticScheduler(), GreedyScheduler())
            res = ST._scheduled_map(
                x -> x * 10, 1:8, [10, 3, 3, 3, 1, 2, 2, 2]; executor = _cap(ex, 4),
            )
            @test res == [10, 20, 30, 40, 50, 60, 70, 80]   # items order
            # Heaviest item first: with a single task, dispatch order IS run order.
            seen = Int[]
            ST._scheduled_map(
                x -> push!(seen, x), 1:4, [1, 9, 3, 5]; executor = _cap(ex, 1),
            )
            @test seen == [2, 4, 3, 1]
            # The task cap holds — it is what bounds resident scan groups.
            live = Threads.Atomic{Int}(0)
            peak = Threads.Atomic{Int}(0)
            ST._scheduled_map(1:16, collect(16:-1:1); executor = _cap(ex, 3)) do x
                Threads.atomic_max!(peak, Threads.atomic_add!(live, 1) + 1)
                sleep(0.02)
                Threads.atomic_sub!(live, 1)
                x
            end
            @test peak[] <= 3
            # Empty input.
            @test isempty(
                ST._scheduled_map(identity, Int[], Float64[]; executor = _cap(ex, 2)),
            )
            # A failed group task rethrows its own exception.
            err = try
                ST._scheduled_map(
                    x -> x == 2 ? error("boom") : x, 1:3, [1, 1, 1]; executor = _cap(ex, 1),
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
            identity, 1:3, [1, 1, 1]; executor = UnbackedExecutor(),
        )
        # Likewise for the memory gate: a scheduler whose task count cannot be
        # read cannot be checked against the budget.
        @test_throws MethodError ST.max_tasks(UnbackedExecutor())
    end

    @testset "full pipeline: θ and output bit-identical across OUTER executors" begin
        uvset, _ = _build_fringe_uvset(; nscans = 2)
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
        mk(ex) = CalibrationPipeline(
            FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))),
            Bandpass(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(outer_executor = ex),
        )
        sol_t, out_t = fitcalibrate(mk(DynamicScheduler()), uvset; reduce = [AverageFrequency(nout = 1)])
        sol_d, out_d = fitcalibrate(mk(GreedyScheduler()), uvset; reduce = [AverageFrequency(nout = 1)])
        @test parent(gains(sol_d)) == parent(gains(sol_t))
        @test Gustavo.stage_names(sol_d) == Gustavo.stage_names(sol_t)
        for (k, leaf) in UVP.branches(out_t)
            ld = UVP.branches(out_d)[k]
            @test isequal(parent(leaf[:vis]), parent(ld[:vis]))
            @test isequal(parent(leaf[:weights]), parent(ld[:weights]))
        end

        # Standalone calibrate matches across outer executors too.
        c_t = calibrate(sol_t, uvset; exec = ExecutionConfig(outer_executor = DynamicScheduler()))
        c_d = calibrate(sol_t, uvset; exec = ExecutionConfig(outer_executor = GreedyScheduler()))
        for (k, leaf) in UVP.branches(c_t)
            @test isequal(parent(leaf[:vis]), parent(UVP.branches(c_d)[k][:vis]))
        end
    end

    @testset "full pipeline: θ bit-identical across INNER executors" begin
        uvset, _ = _build_fringe_uvset(; nscans = 2)
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
        mk(inner) = CalibrationPipeline(
            FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))),
            Bandpass(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(inner_executor = inner),
        )
        # Serial vs multi-chunk within-scan fan-out: the per-block folds are
        # order-fixed by the data layout, so θ is bit-identical.
        sol_ser, _ = fitcalibrate(mk(SerialScheduler()), uvset)
        sol_dyn, _ = fitcalibrate(mk(DynamicScheduler(; nchunks = 4)), uvset)
        @test parent(gains(sol_ser)) == parent(gains(sol_dyn))
    end

    @testset "the driver-facing stream carries both executors" begin
        uvset, _ = _build_fringe_uvset()
        st = FP.scan_stream(
            uvset;
            exec = ExecutionConfig(
                outer_executor = GreedyScheduler(), inner_executor = SerialScheduler(),
            ),
        )
        @test FP.outer_executor(st) isa GreedyScheduler
        @test FP.inner_executor(st) isa SerialScheduler
        # The bare default runs scan groups serially.
        st0 = FP.scan_stream(uvset)
        @test FP.outer_executor(st0) isa SerialScheduler
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
