# ── mapsets: a function over units of data, each read into memory ────────────

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "mapsets" begin
    ps, _ = _build_fringe_ps(; nscans = 3, nspw = 2)
    by_scan = DimensionalData.groupby(ps, XRadio.ByScan())

    @testset "a ProcessingSet's units are its Measurement Sets, keyed by name" begin
        out = mapsets(ms -> (ms isa XRadio.MeasurementSet, size(ms[:visibility])), ps)
        @test out isa OrderedDict
        @test collect(keys(out)) == collect(keys(ps))
        @test all(out[k] == (true, size(ms[:visibility])) for (k, ms) in pairs(ps))
        averaged = mapsets(ms -> XRadio.average(ms, XRadio.ByScan()), ps)
        @test collect(keys(XRadio.ProcessingSet(averaged, copy(DimensionalData.metadata(ps))))) ==
            collect(keys(ps))
    end

    @testset "a groupby result's units are its groups, keyed by label" begin
        out = mapsets(g -> (g isa XRadio.ProcessingSet, collect(keys(g))), by_scan)
        @test collect(keys(out)) == collect(keys(by_scan))
        @test all(out[k] == (true, collect(keys(g))) for (k, g) in pairs(by_scan))
    end

    @testset "any ordered keyed collection of units" begin
        picked = OrderedDict(k => by_scan[k] for k in collect(keys(by_scan))[[3, 1]])
        out = mapsets(g -> collect(keys(g)), picked)
        @test collect(keys(out)) == collect(keys(picked))
        @test all(out[k] == collect(keys(g)) for (k, g) in pairs(picked))
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
        for ms in values(ps), arrays in values(out), A in arrays
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
        @test all(inside[k] == θ(fit(step, g)) for (k, g) in pairs(by_scan))
    end

    @testset "per-unit solutions save and load as a collection" begin
        step = BaselineFringeFit(; gauge = PinAntenna(1))
        sols = mapsets(g -> fit(step, g), by_scan)
        path = tempname()
        try
            @test save_solution(path, sols) == path
            back = load_solution(path)
            @test back isa OrderedDict
            @test collect(keys(back)) == collect(keys(sols))
            for (k, sol) in pairs(sols)
                @test back[k].components == sol.components
                @test all(f -> getfield(back[k].geom, f) == getfield(sol.geom, f), fieldnames(typeof(sol.geom)))
                @test collect(keys(back[k].steps)) == collect(keys(sol.steps))
                @test back[k].steps[:fringe].flagged_scan == sol.steps[:fringe].flagged_scan
                @test back[k].provenance == sol.provenance
            end
            @test_throws "exists" save_solution(path, sols)
        finally
            rm(path; force = true, recursive = true)
        end
        unsaveable = OrderedDict((x -> x) => first(values(sols)))
        @test_throws "cannot save the key" save_solution(path, unsaveable)
        @test !ispath(path)
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

@testset "write! fills a template from concurrent units" begin
    ps, _ = _build_fringe_ps(; nscans = 3, nspw = 2)
    schemas = [UVData.GUSTAVO_VISIBILITY_SCHEMA]
    data_layers = (:visibility, :flag, :weight, :uvw)
    # One unit per integration: every unit is a time slice of each member.
    per_time(ms) = collect(eachindex(lookup(ms, Ti)))
    halves(ms) = (eachindex(lookup(ms, Ti)) .- 1) .÷ 6

    path = joinpath(mktempdir(), "filled.ps.zarr")
    write(path, ps; data = false, chunks = (; time = 1), schemas)
    out = open(XRadio.ProcessingSet, path; mode = "r+")
    exec = ExecutionConfig(; outer_executor = GreedyScheduler(; ntasks = 4))
    mapsets(DimensionalData.groupby(ps, per_time); exec) do g
        write!(out, g)
        return nothing
    end
    back = read(XRadio.ProcessingSet, path)
    for (name, ms) in pairs(ps), layer in data_layers
        @test parent(back[name][layer]) == parent(ms[layer])
    end
    @test isempty(XRadio.check(back; schemas))

    unit = first(values(DimensionalData.groupby(ps, halves)))
    coarse = joinpath(mktempdir(), "coarse.ps.zarr")
    write(coarse, ps; data = false, chunks = (; time = 4), schemas)
    @test_throws "cover part of a stored chunk of" write!(
        open(XRadio.ProcessingSet, coarse; mode = "r+"), unit
    )
    @test_throws "open the store with `mode = \"r+\"`" write!(open(XRadio.ProcessingSet, path), unit)

    name, ms = first(pairs(unit))
    renamed = XRadio.ProcessingSet(OrderedDict(:elsewhere => ms))
    @test_throws "the store holds no Measurement Set `elsewhere`" write!(out, renamed)
    extra = copy(Gustavo.materialize(ms))
    extra[:visibility_model] = extra[:visibility]
    @test_throws "has no layer `visibility_model`" write!(out, XRadio.ProcessingSet(OrderedDict(name => extra)))
end
