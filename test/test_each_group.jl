# ── Scan groups: what `each_group` hands a step ──────────────────────────────
#
# Grouping by scan, reading into memory, the corrections before a step,
# scheduling and progress, the memory budget, and `fit`'s entry points.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# A solve step that records what each group looks like and solves nothing.
struct GroupProbe{F} <: Gustavo.SolveStep
    f::F
end
GroupProbe() = GroupProbe(group -> group)
Gustavo.provides(::GroupProbe) = :probe
Gustavo.solve(s::GroupProbe, ctx) = (; seen = each_group(s.f, ctx))

_probe(pipeline, data; kw...) =
    stage_info(fit(pipeline, data; gauge = PinAntenna(1), kw...), :probe).seen

_halve_weights(ms) = Gustavo._with_layers(ms; weight = DimensionalData.modify(w -> w ./ 2, ms[:weight]))

@testset "each_group" begin
    ps, _ = _build_fringe_ps(; nscans = 3, nspw = 2)
    by_scan = DimensionalData.groupby(ps, XRadio.ByScan())

    @testset "one in-memory group per scan, every window, in scan order" begin
        seen = _probe(GroupProbe(), ps)
        @test length(seen) == length(by_scan) == 3
        for (group, (key, lazy)) in zip(seen, by_scan)
            @test group isa XRadio.ProcessingSet
            @test collect(keys(group)) == collect(keys(lazy))
            @test all(ms -> parent(ms[:visibility]) isa Array, values(group))
            @test all(ms -> only(unique(ms[:scan_name])) == key.scan, values(group))
            for (name, ms) in pairs(group)
                @test isequal(parent(ms[:visibility]), parent(read(lazy[name])[:visibility]))
            end
        end
    end

    @testset "a step sees the corrections before it, and only those" begin
        w(group) = sum(ms -> sum(parent(ms[:weight])), values(group))
        plain = _probe(GroupProbe(w), ps)
        halved = _probe(Any[_halve_weights, GroupProbe(w)], ps)
        @test halved ≈ plain ./ 2
        # A correction after the step does not reach it; the solution records it.
        sol = fit(Any[GroupProbe(w), _halve_weights], ps; gauge = PinAntenna(1))
        @test stage_info(sol, :probe).seen == plain
        @test sol.sequence[2] === _halve_weights
        @test recorded_transforms(sol) == Any[_halve_weights]
    end

    @testset "a correction must return a Measurement Set" begin
        @test_throws "must return a MeasurementSet" _probe(Any[ms -> 1, GroupProbe()], ps)
    end

    @testset "a Measurement Set is fit as a processing set of one" begin
        ms = first(ps)
        seen = _probe(GroupProbe(), ms)
        @test length(seen) == length(unique(ms[:scan_name]))
        @test all(g -> length(g) == 1, seen)
    end

    @testset "the reads are the same under every scheduler" begin
        v(group) = [parent(ms[:visibility]) for ms in values(group)]
        base = _probe(GroupProbe(v), ps)
        for exec in (
                ExecutionConfig(outer_executor = GreedyScheduler(; ntasks = 2)),
                ExecutionConfig(inner_executor = SerialScheduler()),
                ExecutionConfig(outer_executor = DynamicScheduler(; nchunks = 3)),
            )
            @test isequal(_probe(GroupProbe(v), ps; exec), base)
        end
    end

    @testset "progress and timing" begin
        events = Tuple{Symbol, Int, Int}[]
        cb = (stage, done, total) -> push!(events, (stage, done, total))
        sol = fit(GroupProbe(), ps; gauge = PinAntenna(1), exec = ExecutionConfig(progress = cb))
        @test events[1] == (:probe, 0, 3)
        @test sort([e[2] for e in events[2:end]]) == 1:3
        timing = stage_info(sol, :probe).timing
        @test length(timing[:decode]) == 3 && all(>=(0), timing[:decode])
        @test sol.info.nscan == 3
    end

    @testset "a failing group surfaces its own error" begin
        boom = GroupProbe(g -> error("boom"))
        err = try
            fit(boom, ps; gauge = PinAntenna(1), exec = ExecutionConfig(outer_executor = GreedyScheduler(; ntasks = 2)))
            nothing
        catch e
            e
        end
        root = err isa TaskFailedException ? err.task.exception : err
        @test root isa ErrorException && root.msg == "boom"
    end

    @testset "the memory budget gates the outer scheduler" begin
        # The schedulers are used as configured, so a budget that cannot hold the
        # groups they would keep resident is an error before any data is read.
        @test_throws "memory budget" fit(
            GroupProbe(), ps; gauge = PinAntenna(1),
            exec = ExecutionConfig(mem_budget = 1.0, outer_executor = GreedyScheduler(; ntasks = 4)),
        )
        # One group at a time is the floor: no task count would make an oversized
        # group fit, so a serial run goes ahead however tight the budget.
        @test length(_probe(GroupProbe(), ps; exec = ExecutionConfig(mem_budget = 1.0))) == 3
        @test Gustavo._group_charge(first(values(by_scan))) ==
            round(Int, 2.5 * sum(ms -> length(ms[:visibility]) * (8 + 4 + 1), values(first(values(by_scan)))))

        @test Gustavo.max_tasks(SerialScheduler()) == 1
        @test Gustavo.max_tasks(GreedyScheduler(; ntasks = 3)) == 3
        @test Gustavo.max_tasks(DynamicScheduler(; nchunks = 5)) == 5
        # A chunksize-configured scheduler's count follows the data, so the bound
        # falls back to the thread count.
        @test Gustavo.max_tasks(DynamicScheduler(; chunksize = 2)) == Threads.nthreads()
    end

    @testset "what a pipeline may hold" begin
        @test_throws "not Int64" fit((GroupProbe(), 3), ps; gauge = PinAntenna(1))
        @test_throws "no gauge given" fit(GroupProbe(), ps)
        @test_throws "holds no solve step" fit((AutocorrelationNormalization(),), ps; gauge = PinAntenna(1))
    end
end
