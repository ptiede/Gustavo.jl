# Compares the current solver against the stores `record.jl` wrote.
#
#     julia --project=test/references test/references/compare.jl [bt164a]
#
# Each synthetic case's recorded FITS-IDI input is converted with
# `fitsidi2msv4` (to `testdata/references/<case>.ps.zarr`, ignored by git), so
# both runs read the same bytes. The recorded chain is replayed one step at a
# time on the whole processing set, each step fit on data the steps before it
# corrected, all with the first station as the reference. The recorded
# `refine` step has no current counterpart and is skipped; the recorded
# composed gains include it.
#
# Gains are compared on the recorded grid with stations matched by name, as
# the wrapped phase difference (rad) and the log-amplitude difference over
# samples where both gains are finite. Parameters are compared per component
# path and station. Detections are matched by (scan, stations, feeds).

using Gustavo
using XRadio
using FITSFiles
using Zarr
using DimensionalData
using DimensionalData.Lookups: Categorical, Unordered
using LinearAlgebra: BLAS
using Printf: @sprintf

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const SYNTHETIC_DIR = joinpath(REPO, "testdata", "references")

# `weight` is the weight the fixture wrote: `2 / noise^2`, or `1e3` without
# noise. Without a `WEIGHTYP` card or autocorrelations, `fitsidi2msv4` reads the
# file's weights as validity fractions and stores radiometer weights, a uniform
# rescaling; each case is also run with the fixture's weight restored.
const CASES = [
    (name = "chain_two_scans", weight = 2 / 0.05^2),
    (name = "phase_bandpass", weight = 1.0e3),
    (name = "dispersion", weight = 2 / 0.02^2),
    (name = "four_scans_gapped", weight = 2 / 0.05^2),
]

# ── Reading a recorded store ─────────────────────────────────────────────────

function zpath(g, path)
    for k in split(path, '/')
        g = haskey(g.groups, k) ? g.groups[k] : g.arrays[k]
    end
    return g
end
zhas(g, k) = haskey(g.groups, k) || haskey(g.arrays, k)
zread(g, path) = (a = zpath(g, path); a[ntuple(_ -> Colon(), ndims(a))...])

function recorded_gains(g, axes)
    return DimArray(
        zread(g, "gains"),
        (
            Frequency(axes.frequency), Ti(axes.time),
            AntennaName(Categorical(axes.station; order = Unordered())), Feed(1:2),
        ),
    )
end

function recorded_axes(root)
    a = zpath(root, "axes")
    return (;
        frequency = zread(a, "frequency"), time = zread(a, "time"),
        station = String.(zread(a, "station")),
    )
end

# Every `value` array under a step's `parameters/`, keyed by its path.
function recorded_parameters(g, prefix = ())
    out = Dict{Tuple, Any}()
    for (k, sub) in g.groups
        if zhas(sub, "value")
            out[(prefix..., Symbol(k))] = (;
                value = zread(sub, "value"),
                station = String.(zread(sub, "Ant")),
            )
        else
            merge!(out, recorded_parameters(sub, (prefix..., Symbol(k))))
        end
    end
    return out
end

# The recorded detections name stations by their position in `axes/station`.
function recorded_detections(info, stations)
    zhas(info, "det_snr") || return nothing
    names = ("det_scan", "det_ant_a", "det_ant_b", "det_feed_a", "det_feed_b",
        "det_snr", "det_delay", "det_rate", "det_detected")
    d = NamedTuple{Symbol.(names)}(Tuple(zread(info, n) for n in names))
    return merge(d, (; det_ant_a = stations[d.det_ant_a], det_ant_b = stations[d.det_ant_b]))
end

# ── Differences ──────────────────────────────────────────────────────────────

wrap(x) = rem2pi(x, RoundNearest)

# Recorded and current gains on the recorded grid, stations matched by name.
# A station only one side has is listed, with whether its recorded gains are
# all identity (a station without data).
function gain_difference(recorded, current)
    shared = intersect(lookup(recorded, AntennaName), lookup(current, AntennaName))
    recorded_only = [
        (name, all(isone, recorded[AntennaName = At(name)]))
            for name in setdiff(lookup(recorded, AntennaName), shared)
    ]
    current_only = setdiff(lookup(current, AntennaName), shared)
    recorded = recorded[AntennaName = At(shared)]
    current = current[AntennaName = At(shared)]
    size(current) == size(recorded) || throw(DimensionMismatch("gain grids differ: $(size(current)) vs $(size(recorded))"))
    ok = isfinite.(parent(recorded)) .& isfinite.(parent(current))
    ratio = parent(current)[ok] ./ parent(recorded)[ok]
    return (;
        phase = isempty(ratio) ? NaN : maximum(abs ∘ angle, ratio),
        logamp = isempty(ratio) ? NaN : maximum(r -> abs(log(abs(r))), ratio),
        nonfinite = count(!, ok), recorded_only, current_only,
    )
end

# The recorded value's station axis: the one whose length is the station count.
function _station_axis(value, nstation)
    d = findall(==(nstation), size(value))
    length(d) == 1 || error("cannot identify the station axis of a recorded parameter of size $(size(value))")
    return only(d)
end

function parameter_differences(recorded, sol::CalibrationSolution)
    rows = NamedTuple[]
    for c in sol.components
        path = c.path
        haskey(recorded, path) || (push!(rows, (; path, note = "not recorded")); continue)
        r = recorded[path]
        keep = findall(in(lookup(c.params, AntennaName)), r.station)
        cur = c.params[AntennaName = At(r.station[keep])]
        value = selectdim(r.value, _station_axis(r.value, length(r.station)), keep)
        size(parent(cur)) == size(value) || (push!(rows, (; path, note = "shape $(size(parent(cur))) vs recorded $(size(value))")); continue)
        d = parent(cur) .- value
        isphase = first(path) === :phase && c.component.term isa ConstantTerm
        isphase && (d = wrap.(d))
        push!(rows, (; path, maxabs = maximum(abs, d; init = 0.0), scale = maximum(abs, value; init = 0.0)))
    end
    for path in setdiff(keys(recorded), [c.path for c in sol.components])
        push!(rows, (; path, note = "recorded only"))
    end
    return rows
end

detection_key(d, i) = (Int(d.det_scan[i]), String(d.det_ant_a[i]), String(d.det_ant_b[i]), Int(d.det_feed_a[i]), Int(d.det_feed_b[i]))

function detection_difference(recorded, info)
    isnothing(recorded) && return nothing
    rec = Dict(detection_key(recorded, i) => i for i in eachindex(recorded.det_snr))
    cur = Dict(detection_key(info, i) => i for i in eachindex(info.det_snr))
    both = intersect(keys(rec), keys(cur))
    flips = [k for k in both if recorded.det_detected[rec[k]] != info.det_detected[cur[k]]]
    snr_ratio = [info.det_snr[cur[k]] / recorded.det_snr[rec[k]] for k in both]
    ddelay = [abs(info.det_delay[cur[k]] - recorded.det_delay[rec[k]]) for k in both if recorded.det_detected[rec[k]]]
    drate = [abs(info.det_rate[cur[k]] - recorded.det_rate[rec[k]]) for k in both if recorded.det_detected[rec[k]]]
    return (;
        recorded = count(recorded.det_detected), current = count(info.det_detected),
        cells = (length(rec), length(cur)), only_recorded = length(setdiff(keys(rec), keys(cur))),
        only_current = length(setdiff(keys(cur), keys(rec))), flips,
        snr_ratio = isempty(snr_ratio) ? (NaN, NaN) : extrema(snr_ratio),
        delay = maximum(ddelay; init = 0.0), rate = maximum(drate; init = 0.0),
    )
end

# ── Replaying a case ─────────────────────────────────────────────────────────

function convert_case(name; dir = SYNTHETIC_DIR)
    dest = joinpath(dir, "$name.ps.zarr")
    fitsidi2msv4(joinpath(dir, "$name.idifits"), dest; mode = "w", release_date = "2000-01-01")
    return open(ProcessingSet, dest)
end

function with_weight(ps, w)
    ps = Gustavo.materialize(ps)
    for ms in values(ps)
        ms[:weight] .= w
    end
    return ps
end

# Each step is fit on data the steps before it corrected in place, so one copy
# of the set is held.
function replay(ps, gauge)
    ps = Gustavo.materialize(ps)
    fringe = fit(BaselineFringeFit(; gauge), ps)
    calibrate!(fringe, ps; flag_bad = false, apply_flags = false)
    bandpass = fit(Bandpass(; gauge), ps)
    calibrate!(bandpass, ps; flag_bad = false, apply_flags = false)
    adhoc = fit(AdhocPhase(; gauge), ps)
    return (; fringe, bandpass, adhoc)
end

function compare_run(root, sols)
    ax = recorded_axes(root)
    steps = zpath(root, "steps")
    recorded = Dict(split(k, '_'; limit = 2)[2] => g for (k, g) in steps.groups)
    out = Dict{Symbol, Any}()
    for (name, sol) in pairs(sols)
        g = recorded[String(name)]
        out[name] = (;
            gains = gain_difference(recorded_gains(g, ax), gains(sol)),
            parameters = parameter_differences(recorded_parameters(zpath(g, "parameters")), sol),
        )
    end
    out[:detections] = detection_difference(
        recorded_detections(zpath(recorded["fringe"], "info"), ax.station), sols.fringe.steps[:fringe],
    )
    counts = ("n_solved", "n_flat", "n_declined", "n_nodata")
    recorded_info = zpath(recorded["bandpass"], "info").attrs
    current_info = sols.bandpass.steps[:bandpass]
    out[:bandpass_status] = [(k, recorded_info[k], getproperty(current_info, Symbol(k))) for k in counts]
    composed = gains(sols.fringe) .* gains(sols.bandpass) .* gains(sols.adhoc)
    out[:composed] = gain_difference(recorded_gains(root, ax), composed)
    if haskey(recorded, "refine")
        refine = recorded_gains(recorded["refine"], ax)
        out[:refine] = gain_difference(refine, one.(composed))
        shared = intersect(lookup(composed, AntennaName), lookup(refine, AntennaName))
        refined = composed[AntennaName = At(shared)] .* refine[AntennaName = At(shared)]
        out[:composed_with_recorded_refine] = gain_difference(recorded_gains(root, ax), refined)
    end
    return out
end

function compare_case(case)
    root = zopen(joinpath(SYNTHETIC_DIR, "$(case.name).zarr"))
    gauge = PinAntenna(first(recorded_axes(root).station))
    ps = convert_case(case.name)
    return (;
        converted = compare_run(root, replay(ps, gauge)),
        fixture_weight = compare_run(root, replay(with_weight(ps, case.weight), gauge)),
    )
end

# BT164A is converted once, beside its data, by `convert_bt164a`; the replay
# reads the scans the reference was recorded on.
const BT164A_DIR = expanduser("~/Research/M87Monitor/BT164/BT164A")
const BT164A_QBAND = joinpath(BT164A_DIR, "VLBA_BT164A_bt164aQband_BIN0_SRC0_0_260108T230453.idifits")
const BT164A_STORE = joinpath(BT164A_DIR, "VLBA_BT164A_bt164aQband.ps.zarr")
const BT164A_REFERENCE = joinpath(BT164A_DIR, "reference_Qband_scans_1-4.zarr")
const BT164A_SCANS = string.(1:4)

convert_bt164a() = fitsidi2msv4(BT164A_QBAND, BT164A_STORE; mode = "w-", release_date = "2000-01-01")

function compare_bt164a()
    root = zopen(BT164A_REFERENCE)
    gauge = PinAntenna(first(recorded_axes(root).station))
    ps = filter(ms -> all(in(BT164A_SCANS), ms[:scan_name]), open(ProcessingSet, BT164A_STORE))
    isempty(ps) && error("no Measurement Set of $(BT164A_STORE) holds scans $(BT164A_SCANS)")
    return compare_run(root, replay(ps, gauge))
end

# ── Report ───────────────────────────────────────────────────────────────────

fmt(x::Real) = @sprintf("%.2e", x)
fmt(x) = string(x)

function report(io, name, run)
    println(io, "  ", name)
    for k in (:fringe, :bandpass, :adhoc, :composed, :refine, :composed_with_recorded_refine)
        haskey(run, k) || continue
        g = k in (:fringe, :bandpass, :adhoc) ? run[k].gains : run[k]
        println(io, "    ", rpad(k, 30), "gains: phase ", fmt(g.phase), " rad, logamp ", fmt(g.logamp),
            g.nonfinite > 0 ? ", $(g.nonfinite) non-finite" : "",
            isempty(g.recorded_only) ? "" : ", recorded only $(g.recorded_only)",
            isempty(g.current_only) ? "" : ", current only $(g.current_only)")
        k in (:fringe, :bandpass, :adhoc) || continue
        for r in run[k].parameters
            println(io, "      ", rpad(join(r.path, '.'), 26),
                haskey(r, :note) ? r.note : "max |Δ| $(fmt(r.maxabs)) (max |recorded| $(fmt(r.scale)))")
        end
    end
    println(io, "    bandpass status (recorded, current): ",
        join(("$k $r/$c" for (k, r, c) in run[:bandpass_status]), ", "))
    d = run[:detections]
    println(io, "    detections: recorded $(d.recorded), current $(d.current) of $(d.cells) cells; ",
        "flips $(length(d.flips)); SNR ratio $(fmt.(d.snr_ratio)); |Δdelay| $(fmt(d.delay)) s, |Δrate| $(fmt(d.rate)) Hz")
    return nothing
end

function main(io = stdout)
    BLAS.set_num_threads(1)
    results = Dict(case.name => compare_case(case) for case in CASES)
    for case in CASES
        println(io, case.name)
        report(io, "as converted", results[case.name].converted)
        report(io, "fixture weight $(case.weight)", results[case.name].fixture_weight)
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "bt164a" in ARGS
        BLAS.set_num_threads(1)
        result = @time compare_bt164a()
        report(stdout, "BT164A Q band, scans $(join(BT164A_SCANS, ", "))", result)
    else
        main()
    end
end
