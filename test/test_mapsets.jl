# ── mapsets: a function over units of data, each read into memory ────────────

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "mapsets" begin
    ps, _ = _build_fringe_ps(; nscans = 3, nspw = 2)
    by_scan = DimensionalData.groupby(ps, XRadio.ByScan())

    @testset "a ProcessingSet's units are its Measurement Sets, in order" begin
        out = mapsets(ms -> (ms isa XRadio.MeasurementSet, size(ms[:visibility])), ps)
        @test out == [(true, size(ms[:visibility])) for ms in values(ps)]
    end

    @testset "a groupby result's units are its groups, in order" begin
        out = mapsets(g -> (g isa XRadio.ProcessingSet, collect(keys(g))), by_scan)
        @test out == [(true, collect(keys(g))) for g in values(by_scan)]
    end

    @testset "units hold arrays of their own; the source is unchanged" begin
        before = deepcopy(ps)
        out = mapsets(by_scan) do g
            for ms in values(g)
                parent(ms[:visibility]) .*= 2
                parent(ms[:weight]) ./= 4
                parent(ms[:flag]) .= true
            end
            return [parent(ms[:visibility]) for ms in values(g)]
        end
        for ms in values(ps), arrays in out, A in arrays
            @test A !== parent(ms[:visibility])
        end
        for (k, ms) in pairs(ps), layer in (:visibility, :weight, :flag)
            @test parent(ms[layer]) == parent(before[k][layer])
        end
    end

    @testset "a fit inside the body equals a fit on the group" begin
        step = BaselineFringeFit(; gauge = PinAntenna(1))
        θ(sol) = [collect(c.params) for c in sol[:fringe].components]
        inside = mapsets(g -> θ(fit(step, g)), by_scan)
        @test inside == [θ(fit(step, g)) for g in values(by_scan)]
    end

    @testset "concurrent units: same results, progress per unit" begin
        calls = Tuple{Symbol, Int, Int}[]
        exec = ExecutionConfig(;
            outer_executor = GreedyScheduler(; ntasks = 2),
            progress = (stage, done, total) -> push!(calls, (stage, done, total)),
        )
        f(g) = sum(ms -> sum(parent(ms[:weight])), values(g))
        @test mapsets(f, by_scan; exec) == mapsets(f, by_scan)
        @test first(calls) == (:mapsets, 0, 3)
        @test all(c -> c[1] === :mapsets && c[3] == 3, calls)
        @test sort(getindex.(calls, 2)) == 0:3
    end

    @testset "an error in the body propagates" begin
        @test_throws "boom" mapsets(_ -> error("boom"), ps)
    end
end
