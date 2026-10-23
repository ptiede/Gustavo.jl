# Records calibration outputs as Zarr stores that do not depend on Gustavo's
# types, so a later revision of the solver can be compared against them.
#
#     julia --project=test/references test/references/record.jl [synthetic] [bt164a]
#
# Synthetic cases are built from seeded `StableRNG`s; each writes its input to
# `testdata/references/<case>.ps.zarr` and its outputs to
# `testdata/references/<case>.zarr` (both ignored by git). BT164A reads the store
# `convert_bt164a` writes and records beside its data. Every store carries the
# commit, the chain, and the changes expected to move its results
# (`expected_changes`).

include(joinpath(@__DIR__, "chain.jl"))
include(joinpath(@__DIR__, "..", "synthetic_ps.jl"))

using DimensionalData: DimArray, AbstractDimArray, dims, lookup, name as dimname
using Zarr
using Dates: now

# Changes since this recording that are expected to move its results.
const EXPECTED_CHANGES = String[]

function provenance()
    git(args...) = readchomp(Cmd(`git $args`; dir = REPO))
    return Dict{String, Any}(
        "commit" => git("rev-parse", "HEAD"),
        "branch" => git("rev-parse", "--abbrev-ref", "HEAD"),
        "modified_files" => split(git("status", "--porcelain", "--untracked-files=no"), '\n'; keepempty = false),
        "recorded" => string(now()),
        "julia_version" => string(VERSION),
        "gustavo_version" => string(pkgversion(Gustavo)),
        "nthreads" => Threads.nthreads(),
        "chain" => CHAIN_TEXT,
        "expected_changes" => EXPECTED_CHANGES,
    )
end

# Zarr stores dimensions slowest-varying first, the reverse of a Julia array's.
function write_array!(g, key, A::AbstractArray, dimnames; attrs = Dict{String, Any}())
    length(dimnames) == ndims(A) || throw(DimensionMismatch("$key: $(ndims(A)) dimensions, names $dimnames"))
    data = storable(collect(A))
    attrs = merge(attrs, Dict{String, Any}("_ARRAY_DIMENSIONS" => reverse(collect(String, dimnames))))
    z = zcreate(eltype(data), g, String(key), size(data)...; attrs, chunks = max.(size(data), 1))
    z[ntuple(_ -> Colon(), ndims(data))...] = data
    return z
end

storable(A::AbstractArray{Symbol}) = String.(A)
storable(A::AbstractArray{<:AbstractString}) = String.(A)
storable(A::AbstractArray{<:Real}) = A
storable(A::AbstractArray{<:Complex}) = A
storable(A::AbstractArray) = string.(A)

# A labeled array as a group: `value` with its dimensions' names, and each
# dimension's lookup as an array of that name.
function write_labeled!(g, key, A::AbstractDimArray; attrs = Dict{String, Any}())
    lg = zgroup(g, String(key); attrs)
    names = [string(dimname(d)) for d in dims(A)]
    write_array!(lg, "value", parent(A), names)
    for (d, n) in zip(dims(A), names)
        write_array!(lg, n, collect(lookup(d)), [n])
    end
    return lg
end

# Scalars go into the group's attributes (anything else that is not an array
# as its `repr`); labeled arrays become labeled groups, other arrays arrays
# with generic dimension names; nested NamedTuples become subgroups. Wall-clock
# timings are omitted: they are not results.
const UNTIMED = (:timing, :t_pass)

function write_info!(parent, key, info)
    scalars = Dict{String, Any}()
    for (k, v) in pairs(info)
        (k in UNTIMED || v isa Union{AbstractArray, NamedTuple}) && continue
        scalars[String(k)] = v isa Union{Real, AbstractString, Nothing} ? v :
            v isa Symbol ? String(v) : repr(v)
    end
    g = zgroup(parent, String(key); attrs = scalars)
    for (k, v) in pairs(info)
        k in UNTIMED && continue
        if v isa AbstractDimArray
            write_labeled!(g, k, v)
        elseif v isa AbstractArray
            isempty(v) || write_array!(g, k, v, ["$(k)_dim_$i" for i in 1:ndims(v)])
        elseif v isa NamedTuple
            write_info!(g, k, v)
        end
    end
    return g
end

# Each component's parameters under its path (`phase/bandpass/…`), with the
# component it fits in the group's attributes.
function write_components!(g, sol)
    for c in sol.components
        pg = foldl((h, k) -> haskey(h.groups, String(k)) ? h.groups[String(k)] : zgroup(h, String(k)), c.path[1:(end - 1)]; init = g)
        write_labeled!(pg, last(c.path), c.params; attrs = Dict{String, Any}("component" => repr(c.component)))
    end
    return g
end

const GAIN_DIMS = ["frequency", "time", "antenna", "feed"]

function write_geometry!(g, sol)
    geom = sol.geom
    ag = zgroup(g, "axes"; attrs = Dict{String, Any}("t0" => geom.t0, "f0" => geom.f0))
    write_array!(ag, "frequency", geom.channel_freqs, ["frequency"]; attrs = Dict{String, Any}("units" => "Hz"))
    write_array!(ag, "time", geom.times, ["time"]; attrs = Dict{String, Any}("units" => "s"))
    write_array!(ag, "scan_of_time", geom.scan_of_time, ["time"])
    write_array!(ag, "spw_of_chan", geom.spw_of_chan, ["frequency"])
    write_array!(ag, "antenna", geom.antennas, ["antenna"])
    write_array!(ag, "feed", 1:geom.nfeed, ["feed"])
    isempty(geom.scan_names) || write_array!(ag, "scan_names", geom.scan_names, ["scan"])
    isempty(geom.spw_names) || write_array!(ag, "spw_names", geom.spw_names, ["spw"])
    return ag
end

"""
    write_solutions(path, sols; attrs)

Write the solutions `sols` (a NamedTuple in run order) as a Zarr store:
the shared geometry under `axes`, the gains of every solution composed under
`gains`, and per solution (`steps/<i>_<name>`) its own gains, its components'
parameters and its diagnostics. Gains are `V_corr = V / (g_a * conj(g_b))` in
the MSv4 phase sense.
"""
function write_solutions(path, sols::NamedTuple; attrs = Dict{String, Any}())
    ispath(path) && rm(path; recursive = true)
    attrs = merge(
        provenance(), attrs, Dict{String, Any}(
            "format" => "Gustavo reference solution",
            "gain_convention" => "V_corr = V / (g_a * conj(g_b)), MSv4 phase sense",
            "steps" => String.(keys(sols)),
        ),
    )
    root = zgroup(path; attrs)
    write_geometry!(root, first(sols))
    write_array!(root, "gains", parent(composed_gains(sols)), GAIN_DIMS)
    sg = zgroup(root, "steps")
    for (i, (name, sol)) in enumerate(pairs(sols))
        g = zgroup(sg, "$(i)_$(name)"; attrs = Dict{String, Any}("spec" => sol.spec))
        write_array!(g, "gains", parent(gains(sol)), GAIN_DIMS)
        write_components!(zgroup(g, "components"), sol)
        write_info!(g, "info", sol.info)
    end
    return path
end

function write_truth!(root, truth)
    g = zgroup(root, "truth"; attrs = Dict{String, Any}(String(k) => v for (k, v) in pairs(truth) if v isa Real))
    for (k, v) in pairs(truth)
        v isa AbstractArray || continue
        v isa AbstractArray{<:Tuple} && (v = reduce(hcat, collect.(v)))
        write_array!(g, k, v, ["$(k)_dim_$i" for i in 1:ndims(v)])
    end
    return g
end

function record_synthetic()
    mkpath(SYNTHETIC_DIR)
    for case in SYNTHETIC_CASES
        @info "Recording $(case.name)"
        ps, truth = _build_fringe_ps(; case.fixture...)
        input = synthetic_input(case.name)
        write(input, ps; mode = "w")
        path = write_solutions(
            synthetic_output(case.name), replay(open(ProcessingSet, input)); attrs = Dict{String, Any}(
                "input" => basename(input),
                "fixture" => repr(case.fixture),
                "rng" => "StableRNG",
            ),
        )
        write_truth!(zopen(path, "w"), truth)
    end
    return nothing
end

function record_bt164a()
    isdir(BT164A_STORE) || error("record_bt164a: $(BT164A_STORE) does not exist; `convert_bt164a()` writes it")
    return write_solutions(
        BT164A_REFERENCE, replay(bt164a_scans()); attrs = Dict{String, Any}(
            "input" => basename(BT164A_STORE),
            "scans" => BT164A_SCANS,
        ),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    targets = isempty(ARGS) ? ["synthetic", "bt164a"] : ARGS
    for t in targets
        t == "synthetic" ? record_synthetic() :
            t == "bt164a" ? @info("Wrote $(record_bt164a())") :
            error("unknown target $t; expected synthetic or bt164a")
    end
end
