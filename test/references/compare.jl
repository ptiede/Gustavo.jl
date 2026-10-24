# Compares the current solver against the stores `record.jl` wrote.
#
#     julia --project=test/references test/references/compare.jl [synthetic] [bt164a]
#
# The chain in `chain.jl` is replayed on the recorded input: each synthetic
# case's `testdata/references/<case>.ps.zarr`, and BT164A's store.
#
# Gains are compared on the recorded grid with antennas matched by name, as the
# wrapped phase difference (rad) and the log-amplitude difference over samples
# where both gains are finite. Component parameters are compared per path,
# antennas matched by name and every other dimension by its lookup. Fringe
# detections are matched by (scan, antennas, feeds).

include(joinpath(@__DIR__, "chain.jl"))

using Zarr
using DimensionalData
using DimensionalData.Lookups: Categorical, Unordered
using Printf: @sprintf

# ── Reading a recorded store ─────────────────────────────────────────────────

function zpath(g, path)
    for k in split(path, '/')
        g = haskey(g.groups, k) ? g.groups[k] : g.arrays[k]
    end
    return g
end
zhas(g, k) = haskey(g.groups, k) || haskey(g.arrays, k)
zread(a) = a[ntuple(_ -> Colon(), ndims(a))...]
zread(g, path) = zread(zpath(g, path))
# Zarr lists dimensions slowest-varying first, the reverse of a Julia array's.
dimnames(a) = reverse(String.(a.attrs["_ARRAY_DIMENSIONS"]))

function recorded_axes(root)
    a = zpath(root, "axes")
    return (; frequency = zread(a, "frequency"), time = zread(a, "time"), antenna = String.(zread(a, "antenna")), feed = zread(a, "feed"))
end

recorded_gains(g, ax) = DimArray(
    zread(g, "gains"),
    (Frequency(ax.frequency), Ti(ax.time), AntennaName(Categorical(ax.antenna; order = Unordered())), Feed(ax.feed)),
)

# Every labeled `value` under a solution's `components/`, keyed by its path: the
# values, their dimension names and each dimension's lookup.
function recorded_components(g, prefix = ())
    out = Dict{Tuple, Any}()
    for (k, sub) in g.groups
        path = (prefix..., Symbol(k))
        if zhas(sub, "value")
            v = zpath(sub, "value")
            names = dimnames(v)
            out[path] = (; value = zread(v), names, lookups = Dict(n => zread(sub, n) for n in names))
        else
            merge!(out, recorded_components(sub, path))
        end
    end
    return out
end

const DETECTION_FIELDS = (:det_scan, :det_ant_a, :det_ant_b, :det_feed_a, :det_feed_b, :det_snr, :det_delay, :det_rate, :det_detected)

function recorded_detections(info)
    zhas(info, "det_snr") || return nothing
    return NamedTuple{DETECTION_FIELDS}(Tuple(zread(info, String(k)) for k in DETECTION_FIELDS))
end

# ── Differences ──────────────────────────────────────────────────────────────

wrap(x) = rem2pi(x, RoundNearest)

# Recorded and current gains on the recorded grid, antennas matched by name. An
# antenna only one side has is listed, with whether its recorded gains are all
# identity (an antenna without data).
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

# One component's current parameters against the recorded ones: antennas
# matched by name, every other dimension required to have the recorded lookup.
function component_difference(r, c)
    names = [string(DimensionalData.name(d)) for d in dims(c.params)]
    names == r.names || return (; note = "dimensions $names vs recorded $(r.names)")
    cur = c.params
    rec = r.value
    if "AntennaName" in names
        recorded_antennas = String.(r.lookups["AntennaName"])
        keep = findall(in(lookup(cur, AntennaName)), recorded_antennas)
        cur = cur[AntennaName = At(recorded_antennas[keep])]
        rec = selectdim(rec, findfirst(==("AntennaName"), names), keep)
    end
    for (d, n) in zip(dims(cur), names)
        n == "AntennaName" && continue
        isequal(collect(lookup(d)), r.lookups[n]) || return (; note = "$n lookup differs from the recorded one")
    end
    d = parent(cur) .- rec
    first(c.path) === :phase && c.component.term isa ConstantTerm && (d = wrap.(d))
    return (; maxabs = maximum(abs, d; init = 0.0), scale = maximum(abs, rec; init = 0.0))
end

function component_differences(recorded, sol)
    rows = NamedTuple[]
    for c in sol.components
        push!(rows, merge((; c.path), haskey(recorded, c.path) ? component_difference(recorded[c.path], c) : (; note = "not recorded")))
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

const BANDPASS_COUNTS = ("n_solved", "n_flat", "n_nodata")

function compare_run(root, sols)
    ax = recorded_axes(root)
    recorded = Dict(split(k, '_'; limit = 2)[2] => g for (k, g) in zpath(root, "steps").groups)
    out = Dict{Symbol, Any}()
    for (name, sol) in pairs(sols)
        g = recorded[String(name)]
        out[name] = (;
            gains = gain_difference(recorded_gains(g, ax), gains(sol)),
            components = component_differences(recorded_components(zpath(g, "components")), sol),
        )
    end
    out[:detections] = detection_difference(recorded_detections(zpath(recorded["fringe"], "info")), sols.fringe.info)
    recorded_info = zpath(recorded["bandpass"], "info").attrs
    out[:bandpass_status] = [(k, recorded_info[k], getproperty(sols.bandpass.info, Symbol(k))) for k in BANDPASS_COUNTS]
    out[:composed] = gain_difference(recorded_gains(root, ax), composed_gains(sols))
    return out
end

compare_synthetic(case) = compare_run(zopen(synthetic_output(case.name)), replay(open(ProcessingSet, synthetic_input(case.name))))

compare_bt164a() = compare_run(zopen(BT164A_REFERENCE), replay(bt164a_scans()))

# ── Report ───────────────────────────────────────────────────────────────────

fmt(x::Real) = @sprintf("%.2e", x)
fmt(x) = string(x)

function report(io, name, run)
    println(io, name)
    for k in (:fringe, :bandpass, :adhoc, :composed)
        g = k === :composed ? run[k] : run[k].gains
        println(io, "  ", rpad(k, 10), "gains: phase ", fmt(g.phase), " rad, logamp ", fmt(g.logamp),
            g.nonfinite > 0 ? ", $(g.nonfinite) non-finite" : "",
            isempty(g.recorded_only) ? "" : ", recorded only $(g.recorded_only)",
            isempty(g.current_only) ? "" : ", current only $(g.current_only)")
        k === :composed && continue
        for r in run[k].components
            println(io, "    ", rpad(join(r.path, '.'), 24),
                haskey(r, :note) ? r.note : "max |Δ| $(fmt(r.maxabs)) (max |recorded| $(fmt(r.scale)))")
        end
    end
    println(io, "  bandpass status (recorded/current): ", join(("$k $r/$c" for (k, r, c) in run[:bandpass_status]), ", "))
    d = run[:detections]
    isnothing(d) && return nothing
    println(io, "  detections: recorded $(d.recorded), current $(d.current) of $(d.cells) cells; ",
        "flips $(length(d.flips)); SNR ratio $(fmt.(d.snr_ratio)); |Δdelay| $(fmt(d.delay)) s, |Δrate| $(fmt(d.rate)) Hz")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    targets = isempty(ARGS) ? ["synthetic", "bt164a"] : ARGS
    for t in targets
        if t == "synthetic"
            foreach(case -> report(stdout, case.name, compare_synthetic(case)), SYNTHETIC_CASES)
        elseif t == "bt164a"
            report(stdout, "BT164A Q band, scans $(join(BT164A_SCANS, ", "))", @time compare_bt164a())
        else
            error("unknown target $t; expected synthetic or bt164a")
        end
    end
end
