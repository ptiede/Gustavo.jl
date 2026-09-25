# ── The executor seam ─────────────────────────────────────────────────────────
#
# A run parallelizes at two independently-selected levels (see `ExecutionConfig`):
# the OUTER across-scan group scheduler behind `each_group`, and the INNER
# within-scan fan-out — each an OhMyThreads `Scheduler`. The contract under
# test: groups are dispatched heaviest-first and run at the outer scheduler's
# own task count, θ and outputs are bit-identical across BOTH choices, and a
# failed group task surfaces its exception.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")
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
            res = Gustavo._scheduled_map(
                x -> x * 10, 1:8, [10, 3, 3, 3, 1, 2, 2, 2]; executor = _cap(ex, 4),
            )
            @test res == [10, 20, 30, 40, 50, 60, 70, 80]   # items order
            # Heaviest item first: with a single task, dispatch order IS run order.
            seen = Int[]
            Gustavo._scheduled_map(
                x -> push!(seen, x), 1:4, [1, 9, 3, 5]; executor = _cap(ex, 1),
            )
            @test seen == [2, 4, 3, 1]
            # The task cap holds — it is what bounds resident scan groups.
            live = Threads.Atomic{Int}(0)
            peak = Threads.Atomic{Int}(0)
            Gustavo._scheduled_map(1:16, collect(16:-1:1); executor = _cap(ex, 3)) do x
                Threads.atomic_max!(peak, Threads.atomic_add!(live, 1) + 1)
                sleep(0.02)
                Threads.atomic_sub!(live, 1)
                x
            end
            @test peak[] <= 3
            # Empty input.
            @test isempty(
                Gustavo._scheduled_map(identity, Int[], Float64[]; executor = _cap(ex, 2)),
            )
            # A failed group task rethrows its own exception.
            err = try
                Gustavo._scheduled_map(
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
        @test_throws MethodError Gustavo._scheduled_map(
            identity, 1:3, [1, 1, 1]; executor = UnbackedExecutor(),
        )
        # Likewise for the memory gate: a scheduler whose task count cannot be
        # read cannot be checked against the budget.
        @test_throws MethodError Gustavo.max_tasks(UnbackedExecutor())
    end

    @testset "full pipeline: θ and output bit-identical across OUTER executors" begin
        ps, _ = _build_fringe_ps(; nscans = 2)
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, options = FP.AdhocOptions(; snr_floor = 0.0))
        run(ex) = fit(
            BaselineFringeFit() |> Bandpass() |> AdhocPhase(adhoc), ps;
            exec = ExecutionConfig(outer_executor = ex), gauge = PinAntenna(1),
        )
        sol_t, sol_d = run(DynamicScheduler()), run(GreedyScheduler())
        @test parent(gains(sol_d)) == parent(gains(sol_t))
        @test keys(sol_d) == keys(sol_t)
        out_t = calibrate(sol_t, ps; exec = ExecutionConfig(outer_executor = DynamicScheduler()))
        out_d = calibrate(sol_d, ps; exec = ExecutionConfig(outer_executor = GreedyScheduler()))
        for k in keys(ps)
            @test isequal(parent(out_t[k][:visibility]), parent(out_d[k][:visibility]))
            @test isequal(parent(out_t[k][:weight]), parent(out_d[k][:weight]))
        end
    end

    @testset "full pipeline: θ bit-identical across INNER executors" begin
        ps, _ = _build_fringe_ps(; nscans = 2)
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, options = FP.AdhocOptions(; snr_floor = 0.0))
        run(inner) = fit(
            BaselineFringeFit() |> Bandpass() |> AdhocPhase(adhoc), ps;
            exec = ExecutionConfig(inner_executor = inner), gauge = PinAntenna(1),
        )
        # Serial vs multi-chunk within-scan fan-out: the per-block folds are
        # order-fixed by the data layout, so θ is bit-identical.
        sol_ser = run(SerialScheduler())
        sol_dyn = run(DynamicScheduler(; nchunks = 4))
        @test parent(gains(sol_ser)) == parent(gains(sol_dyn))
    end
end
