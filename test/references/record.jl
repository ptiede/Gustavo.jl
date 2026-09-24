# Records calibration outputs as Zarr stores that do not depend on Gustavo's
# types, so a later revision of the solver can be compared against them.
#
#     julia --project=test/references test/references/record.jl [synthetic] [bt164a]
#
# Synthetic cases are built from seeded `StableRNG`s, so the same seed gives
# the same data on any Julia version; each writes its input to
# `testdata/references/<case>.idifits` and its outputs to
# `testdata/references/<case>.zarr` (both ignored by git). BT164A writes beside
# its data. Every store carries the commit, the settings, the seed, and the
# changes expected to move its results (`expected_changes`).

using Gustavo
using Gustavo: Calibration, Fring
using DimensionalData: DimArray, dims, lookup, name as dimname
using FITSFiles
using Zarr
using Dates: now
using StableRNGs: StableRNG

include(joinpath(@__DIR__, "..", "synthetic_uvset.jl"))

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const SYNTHETIC_DIR = joinpath(REPO, "testdata", "references")
const BT164A_DIR = expanduser("~/Research/M87Monitor/BT164/BT164A")
const BT164A_QBAND = joinpath(BT164A_DIR, "VLBA_BT164A_bt164aQband_BIN0_SRC0_0_260108T230453.idifits")

const EXPECTED_CHANGES = [
    "Weights: fitsidi2msv4 stores radiometer weights scaled by the quantization " *
        "efficiencies eta_a*eta_b and divided by the autocorrelations, where " *
        "load_fitsidi divides by the autocorrelations only; SNRs and weighted " *
        "averages move, noiseless gains should not.",
    "Flags: MSv4 carries FLAG as its own layer; a FITS-IDI FLAG row on a parallel " *
        "hand also flags the cross hands that use that autocorrelation.",
    "Feed invariance: refine fits, stationize and adhoc phasing stop splitting " *
        "products into same-feed and feed-linking sets; results change only on " *
        "baselines whose stations order their receptors differently.",
    "Precision: fitsidi2msv4 keeps the FLUX column's element type.",
    "Station order: compare by station name, not by position.",
]

# The standard four-step chain, used for BT164A and the default synthetic case.
default_chain() = FringeFit() |> DispersionSBDFit() |> Bandpass() |> TemporalSmoother()

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

# Scalars go into the group's attributes (anything else that is not an array
# as its `repr`); arrays become arrays with generic dimension names; nested
# NamedTuples become subgroups. Wall-clock timings are omitted: they are not
# results.
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
        v isa AbstractArray && write_array!(g, k, v, ["$(k)_dim_$i" for i in 1:ndims(v)])
        v isa NamedTuple && write_info!(g, k, v)
    end
    return g
end

function write_parameters!(g, tree::NamedTuple)
    for (k, v) in pairs(tree)
        write_parameters!(v isa NamedTuple ? zgroup(g, String(k)) : g, v, k)
    end
    return g
end
write_parameters!(g, tree::NamedTuple, _) = write_parameters!(g, tree)

function write_parameters!(g, leaf::DimArray, key)
    lg = zgroup(g, String(key))
    names = [string(dimname(d)) for d in dims(leaf)]
    write_array!(lg, "value", parent(leaf), names)
    for (d, n) in zip(dims(leaf), names)
        write_array!(lg, n, collect(lookup(d)), [n])
    end
    return lg
end

const GAIN_DIMS = ["frequency", "time", "station", "feed"]

function write_geometry!(g, sol)
    geom = sol.geom
    ag = zgroup(g, "axes"; attrs = Dict{String, Any}("t0" => geom.t0, "f0" => geom.f0))
    write_array!(ag, "frequency", geom.channel_freqs, ["frequency"]; attrs = Dict{String, Any}("units" => "Hz"))
    write_array!(ag, "time", geom.times, ["time"]; attrs = Dict{String, Any}("units" => "s"))
    write_array!(ag, "scan_of_time", geom.scan_of_time, ["time"])
    write_array!(ag, "spw_of_chan", geom.spw_of_chan, ["frequency"])
    write_array!(ag, "station", sol.info.ant_names, ["station"])
    write_array!(ag, "feed", 1:2, ["feed"])
    isempty(geom.scan_names) || write_array!(ag, "scan_names", geom.scan_names, ["scan"])
    isempty(geom.spw_names) || write_array!(ag, "spw_names", geom.spw_names, ["spw"])
    return ag
end

"""
    write_solution(path, sol; attrs)

Write `sol` as a Zarr store: the geometry under `axes`, the gains of every
step composed under `gains`, and per step (in run order, `steps/<i>_<name>`)
its own gains, its labeled parameters and its diagnostics. Gains are
`V_corr = V / (g_a * conj(g_b))` in the MSv4 phase sense.
"""
function write_solution(path, sol; attrs = Dict{String, Any}())
    ispath(path) && rm(path; recursive = true)
    attrs = merge(
        provenance(), attrs, Dict{String, Any}(
            "format" => "Gustavo reference solution",
            "gain_convention" => "V_corr = V / (g_a * conj(g_b)), MSv4 phase sense",
            "steps" => String.(keys(sol)),
        ),
    )
    root = zgroup(path; attrs)
    write_geometry!(root, sol)
    write_array!(root, "gains", parent(Calibration.gains(sol)), GAIN_DIMS)
    write_info!(root, "info", sol.info)
    sg = zgroup(root, "steps")
    for i in eachindex(sol)
        step = sol[i]
        g = zgroup(sg, "$(i)_$(only(keys(step)))")
        write_array!(g, "gains", parent(Calibration.gains(step)), GAIN_DIMS)
        write_array!(g, "theta", step.steps[1].θ, ["theta"])
        write_parameters!(zgroup(g, "parameters"), Calibration.parameters(step))
        write_info!(g, "info", step.steps[1].info)
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

# Each case is a fixture from `synthetic_uvset.jl` and a pipeline taken from the
# test suite. The fixture is written to FITS-IDI and read back, so the solve
# runs on the same bytes a later reader will see.
const SYNTHETIC_CASES = [
    (
        name = "chain_two_scans",
        fixture = (; seed = 1234, nant = 4, nspw = 2, nchan = 8, ntime = 12, nscans = 2, noise = 0.05),
        pipeline = default_chain,
        pipeline_text = "FringeFit() |> DispersionSBDFit() |> Bandpass() |> TemporalSmoother()",
    ),
    (
        name = "phase_bandpass",
        bandpass_seed = 0xBA9D,
        fixture = (;
            seed = 1234, nant = 4, nspw = 2, nchan = 8,
            bandpass = let rng = StableRNG(0xBA9D), bp = zeros(4, 2, 16)
                for a in 2:4, f in 1:2
                    off = rand(rng) - 0.5
                    for c in 1:16
                        bp[a, f, c] = off + sin(2π * c / 16 + a + f)
                    end
                end
                bp
            end,
        ),
        pipeline = () -> FringeFit() |> Bandpass() |>
            TemporalSmoother(Fring.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        pipeline_text = "FringeFit() |> Bandpass() |> " *
            "TemporalSmoother(SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0))",
    ),
    (
        name = "dispersion",
        fixture = (; seed = 1234, nant = 4, nspw = 4, nchan = 8, ntime = 12, noise = 0.02, dtec = [0.0, 0.3, -0.2, 0.5]),
        pipeline = default_chain,
        pipeline_text = "FringeFit() |> DispersionSBDFit() |> Bandpass() |> TemporalSmoother()",
    ),
    (
        name = "four_scans_gapped",
        fixture = (; seed = 1234, nant = 5, nspw = 2, nchan = 8, ntime = 10, nscans = 4, scan_gap = 600.0, noise = 0.05),
        pipeline = default_chain,
        pipeline_text = "FringeFit() |> DispersionSBDFit() |> Bandpass() |> TemporalSmoother()",
    ),
]

function record_synthetic(dir = SYNTHETIC_DIR)
    mkpath(dir)
    for case in SYNTHETIC_CASES
        @info "Recording $(case.name)"
        uvset, truth = _build_fringe_uvset(; case.fixture...)
        input = joinpath(dir, "$(case.name).idifits")
        write_fitsidi(input, uvset)
        sol = fit(case.pipeline(), load_fitsidi(input; lazy = false))
        path = write_solution(
            joinpath(dir, "$(case.name).zarr"), sol; attrs = Dict{String, Any}(
                "input" => basename(input),
                "reader" => "load_fitsidi(input; lazy = false)",
                "pipeline" => case.pipeline_text,
                "fixture" => repr(case.fixture),
                "seed" => case.fixture.seed,
                "rng" => "StableRNG",
                "bandpass_seed" => get(case, :bandpass_seed, nothing),
            ),
        )
        write_truth!(zopen(path, "w"), truth)
    end
    return nothing
end

# One cycle of the schedule: 3C273, M87, 3C279, M87.
const BT164A_SCANS = 1:4

function record_bt164a(
        path = BT164A_QBAND; scans = BT164A_SCANS,
        ntasks = max(1, Sys.CPU_THREADS - 4),
    )
    isfile(path) || error("record_bt164a: $path does not exist")
    uvset = load_fitsidi(path; lazy = true, scans)
    exec = ExecutionConfig(inner_executor = DynamicScheduler(; ntasks))
    sol = fit(default_chain(), uvset; exec)
    out = joinpath(dirname(path), "reference_Qband_scans_$(first(scans))-$(last(scans)).zarr")
    return write_solution(
        out, sol; attrs = Dict{String, Any}(
            "input" => basename(path),
            "reader" => "load_fitsidi(input; lazy = true, scans = $(scans))",
            "inner_ntasks" => ntasks,
            "pipeline" => "FringeFit() |> DispersionSBDFit() |> Bandpass() |> TemporalSmoother()",
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
