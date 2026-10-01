# ── Scan groups: what `each_group` hands a step ──────────────────────────────
#
# Grouping by scan, reading into memory, the corrections before a step,
# scheduling and progress, and `fit`'s entry points.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# A solve step that records what each group looks like and solves nothing.
struct GroupProbe{F} <: Gustavo.SolveStep
    f::F
end
GroupProbe() = GroupProbe(group -> group)
Gustavo.provides(::GroupProbe) = :probe
Gustavo.solve(s::GroupProbe, ctx) = (; seen = each_group(s.f, ctx))

_probe(pipeline, data; kw...) =
    fit(pipeline, data; kw...).steps[:probe].seen

_halve_weights(ms) = (parent(ms[:weight]) ./= 2; ms)

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
        # A correction after the step does not reach it; the solution's
        # provenance records it.
        sol = fit(Any[GroupProbe(w), _halve_weights], ps)
        @test sol.steps[:probe].seen == plain
        @test occursin("_halve_weights", sol.provenance.pipeline)
        @test first(findfirst("GroupProbe", sol.provenance.pipeline)) < first(findfirst("_halve_weights", sol.provenance.pipeline))
    end

    @testset "a correction must return the Measurement Set it is handed" begin
        @test_throws "returned a Int64" _probe(Any[ms -> 1, GroupProbe()], ps)
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
        sol = fit(GroupProbe(), ps; exec = ExecutionConfig(progress = cb))
        @test events[1] == (:probe, 0, 3)
        @test sort([e[2] for e in events[2:end]]) == 1:3
        timing = sol.steps[:probe].timing
        @test length(timing[:decode]) == 3 && all(>=(0), timing[:decode])
        @test sol.info.nscan == 3
    end

    @testset "a failing group surfaces its own error" begin
        boom = GroupProbe(g -> error("boom"))
        err = try
            fit(boom, ps; exec = ExecutionConfig(outer_executor = GreedyScheduler(; ntasks = 2)))
            nothing
        catch e
            e
        end
        root = err isa TaskFailedException ? err.task.exception : err
        @test root isa ErrorException && root.msg == "boom"
    end

    @testset "group sizes and task bounds" begin
        @test Gustavo._group_bytes(first(values(by_scan))) ==
            sum(ms -> length(ms[:visibility]) * (8 + 4 + 1), values(first(values(by_scan))))

        @test Gustavo.max_tasks(SerialScheduler()) == 1
        @test Gustavo.max_tasks(GreedyScheduler(; ntasks = 3)) == 3
        @test Gustavo.max_tasks(DynamicScheduler(; nchunks = 5)) == 5
        # A chunksize-configured scheduler's count follows the data, so the bound
        # falls back to the thread count.
        @test Gustavo.max_tasks(DynamicScheduler(; chunksize = 2)) == Threads.nthreads()
    end

    @testset "what a pipeline may hold" begin
        @test_throws "not Int64" fit((GroupProbe(), 3), ps)
        @test_throws "holds no solve step" fit((AutocorrelationNormalization(),), ps)
    end
end
