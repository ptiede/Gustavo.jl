# ── CalibrationSolution: container, geometry, apply, serialization ───────────
#
# A `CalibrationSolution` bundles a solved `StationGainModel`, its
# `ParameterLayout`, the `DataGeometry` it was solved over, and the flattened
# parameter vector θ, plus a free-form `info` NamedTuple of diagnostics. It is
# the hand-off object between a solver (e.g. `Gustavo.Fringe.solve_fringes`) and
# the data: `apply_calibration(uvset, sol)` divides every leaf's visibilities by
# the model's per-antenna gains, and `save_solution`/`load_solution` round-trip
# it through the `Serialization` stdlib.

using Serialization: serialize, deserialize
using Statistics: mean
using DimensionalData: lookup, Ti
using ..UVData: Frequency, Pol, Baseline

"""
    CalibrationSolution(model, layout, geom, θ, info)

A solved station gain model. `model::StationGainModel` is the shared model,
`layout::ParameterLayout` its flattened parameter plan over `geom::DataGeometry`,
`θ` the solved parameter vector (`length(θ) == layout.nθ`), and `info` a
NamedTuple of solver diagnostics (per-scan SNR, χ, residuals, …).
"""
struct CalibrationSolution{M <: StationGainModel, L <: ParameterLayout, G <: DataGeometry}
    model::M
    layout::L
    geom::G
    θ::Vector{Float64}
    info::NamedTuple
end

function CalibrationSolution(
        model::StationGainModel, layout::ParameterLayout, geom::DataGeometry,
        θ::AbstractVector, info::NamedTuple = NamedTuple(),
    )
    length(θ) == layout.nθ ||
        error("CalibrationSolution: θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)")
    return CalibrationSolution(model, layout, geom, Float64.(collect(θ)), info)
end

# ── Geometry from a UVSet ────────────────────────────────────────────────────

"""
    build_geometry(uvset; f0 = nothing, t0 = nothing) -> DataGeometry

Build the union `DataGeometry` spanning every leaf of `uvset`: the sorted unique
channel frequencies (Hz) of all bands, the sorted unique integration times
(hours) of all scans, with `spw_of_chan` / `scan_of_time` dense-ranked from the
leaves' `spw_name` / `scan_name`. `f0` defaults to the mean channel frequency,
`t0` to the first time. Assumes a single consistent antenna table across the
set (used only to size the solve elsewhere).
"""
function build_geometry(uvset::UVSet; f0 = nothing, t0 = nothing)
    # Collect (freq, spw_name) and (time, scan_name) observations from all leaves.
    freq_spw = Dict{Float64, String}()
    time_scan = Dict{Float64, String}()
    for (_, leaf) in UVData.branches(uvset)
        info = UVData.metadata(leaf)
        fs = channel_freqs(info.freq_setup)
        for f in fs
            fk = Float64(f)
            prev = get(freq_spw, fk, nothing)
            (prev === nothing || prev == info.spw_name) ||
                error(
                "build_geometry: channel frequency $fk Hz appears in conflicting spectral " *
                    "windows '$prev' and '$(info.spw_name)' — a single concatenated channel axis " *
                    "cannot dense-rank it to one spw. Partition the set so each frequency belongs " *
                    "to one spw, or rename the spws consistently."
            )
            freq_spw[fk] = info.spw_name
        end
        ts = lookup(leaf[:vis], Ti)
        for t in ts
            tk = Float64(t)
            prev = get(time_scan, tk, nothing)
            (prev === nothing || prev == info.scan_name) ||
                error(
                "build_geometry: time $tk h appears in conflicting scans '$prev' and " *
                    "'$(info.scan_name)' — a single concatenated time axis cannot dense-rank it to " *
                    "one scan. Check for overlapping scan windows or inconsistent scan names."
            )
            time_scan[tk] = info.scan_name
        end
    end

    freqs = sort!(collect(keys(freq_spw)))
    times = sort!(collect(keys(time_scan)))

    spw_labels = [freq_spw[f] for f in freqs]
    scan_labels = [time_scan[t] for t in times]

    spw_of_chan, _ = _dense_rank_labels(spw_labels)
    scan_of_time, _ = _dense_rank_labels(scan_labels)

    spw_names = _unique_in_order(spw_labels)
    scan_names = _unique_in_order(scan_labels)

    f0v = isnothing(f0) ? (isempty(freqs) ? 0.0 : mean(freqs)) : Float64(f0)
    t0v = isnothing(t0) ? (isempty(times) ? 0.0 : first(times)) : Float64(t0)

    return DataGeometry(;
        times = times,
        channel_freqs = freqs,
        scan_of_time = scan_of_time,
        spw_of_chan = spw_of_chan,
        t0 = t0v,
        f0 = f0v,
        scan_names = scan_names,
        spw_names = spw_names,
    )
end

# Dense-rank string labels to 1..k preserving first-appearance order.
function _dense_rank_labels(labels::AbstractVector{<:AbstractString})
    seen = Dict{String, Int}()
    out = Vector{Int}(undef, length(labels))
    next = 0
    for i in eachindex(labels)
        id = get(seen, labels[i], 0)
        if id == 0
            next += 1
            seen[labels[i]] = next
            id = next
        end
        out[i] = id
    end
    return out, next
end

function _unique_in_order(labels::AbstractVector{<:AbstractString})
    seen = Set{String}()
    out = String[]
    for l in labels
        if !(l in seen)
            push!(seen, l)
            push!(out, l)
        end
    end
    return out
end

"""
    leaf_window(geom, leaf) -> (chan_idx::Vector{Int}, ti_idx::Vector{Int})

Indices into `geom.channel_freqs` / `geom.times` of the channels and times the
leaf carries, matched by value (frequency by `isapprox` rtol 1e-9, time by atol
1e-9 h). Errors if any leaf sample has no match in the geometry.
"""
function leaf_window(geom::DataGeometry, leaf)
    fs = lookup(leaf[:vis], Frequency)
    ts = lookup(leaf[:vis], Ti)
    chan_idx = Vector{Int}(undef, length(fs))
    for (i, f) in enumerate(fs)
        j = findfirst(g -> isapprox(g, Float64(f); rtol = 1.0e-9), geom.channel_freqs)
        isnothing(j) && error("leaf_window: channel frequency $f not found in geometry")
        chan_idx[i] = j
    end
    ti_idx = Vector{Int}(undef, length(ts))
    for (i, t) in enumerate(ts)
        j = findfirst(g -> isapprox(g, Float64(t); atol = 1.0e-9), geom.times)
        isnothing(j) && error("leaf_window: time $t not found in geometry")
        ti_idx[i] = j
    end
    return chan_idx, ti_idx
end

# ── Apply ────────────────────────────────────────────────────────────────────

"""
    apply_calibration(uvset::UVSet, sol::CalibrationSolution) -> UVSet

Divide every leaf's visibilities by the solution's per-antenna gains. For a
baseline `(a, b)` and correlation product `p` with feeds `(fa, fb)`:

    V_corr = V / (g_a[fa] · conj(g_b[fb])),    W_corr = W · |g_a · g_b|²

Samples where either gain magnitude underflows are flagged (weight 0, vis NaN).
"""
function UVData.apply_calibration(uvset::UVSet, sol::CalibrationSolution)
    ev = GainEvaluator(sol.model, sol.layout)
    return UVData.apply(uvset) do leaf, info, root
        # Correction reads vis + weights; the output flag is re-derived from the
        # corrected weights downstream, so skip the redundant on-disk flag layer.
        leaf = materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
        ci, ti = leaf_window(sol.geom, leaf)
        g = evaluate_gains(ev, sol.θ, ci, ti)      # (nchan_leaf, nti_leaf, nant, 2)
        bl_pairs = UVData.baselines(leaf).pairs
        pols = pol_products(leaf)
        vis_corr, w_corr = _apply_gain_kernel(leaf[:vis], leaf[:weights], g, bl_pairs, pols)
        return with_visibilities(leaf, vis_corr, w_corr)
    end
end

# Divide visibilities by complex antenna gains `g[c, ti, ant, feed]`. The vis /
# weight DimArrays are (Frequency, Ti, Baseline, Pol).
function _apply_gain_kernel(
        vis_p::AbstractArray, w_p::AbstractArray,
        g::AbstractArray{<:Complex, 4},
        bl_pairs, pols,
    )
    V = parent(vis_p)
    W = parent(w_p)
    Vc = copy(V)
    Wc = copy(W)
    nchan, nti, nbl, npol = size(V)
    geps = 1.0e-12
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            for tt in 1:nti, c in 1:nchan
                w = W[c, tt, bi, p]
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                denom = ga * conj(gb)
                if abs(ga) < geps || abs(gb) < geps || !isfinite(denom)
                    Wc[c, tt, bi, p] = zero(eltype(Wc))
                    Vc[c, tt, bi, p] = convert(eltype(Vc), NaN)
                    continue
                end
                Vc[c, tt, bi, p] = V[c, tt, bi, p] / denom
                Wc[c, tt, bi, p] = w * abs2(ga * gb)
            end
        end
    end
    return Vc, Wc
end

# ── Serialization ────────────────────────────────────────────────────────────

"""
    save_solution(path, sol::CalibrationSolution)

Serialize `sol` to `path` via the `Serialization` stdlib inside a versioned
wrapper NamedTuple `(; version, model, layout, geom, θ, info)`.
"""
function save_solution(path::AbstractString, sol::CalibrationSolution)
    wrapper = (; version = 1, sol.model, sol.layout, sol.geom, sol.θ, sol.info)
    serialize(path, wrapper)
    return path
end

"""
    load_solution(path) -> CalibrationSolution

Inverse of [`save_solution`](@ref).
"""
function load_solution(path::AbstractString)
    w = deserialize(path)
    w.version == 1 || error("load_solution: unsupported version $(w.version)")
    return CalibrationSolution(w.model, w.layout, w.geom, w.θ, w.info)
end

"""
    save_solution_hdf5(path, sol::CalibrationSolution; gains = true, time_block = 1024)

Write `sol` to an HDF5 caltable readable from any language (Python/h5py, CASA, …),
NOT just Julia. Provided by `GustavoHDF5Ext` — load `HDF5` to enable it.

Layout:
- `gain/real`, `gain/imag` — the evaluated complex antenna gains on the
  `(channel, time, antenna, feed)` grid (`Float32`, chunked + gzip). Written in
  blocks of `time_block` integrations so the full ~GB cube is never resident. Omit
  with `gains = false` to write only the compact parametric form.
- `axes/*` — `channel_freq_hz`, `time`, `scan_of_time`, `spw_of_chan`, `f0`, `t0`.
- `info/*` — the solver diagnostics (per-scan SNR/χ/ncomp, counts).
- root attributes — format/version, units, and the gain convention
  `V_corr = V / (g_a · conj(g_b))`, `weight ×= |g_a g_b|²`.
- `julia/blob` — the `Serialization` bytes of `(model, layout, geom, θ, info)` so
  Julia can round-trip the solution losslessly (external readers ignore it).
"""
function save_solution_hdf5 end

"""
    load_solution_hdf5(path) -> CalibrationSolution

Reconstruct a `CalibrationSolution` from an HDF5 file written by
[`save_solution_hdf5`](@ref) (via its `julia/blob`). Provided by `GustavoHDF5Ext`.
"""
function load_solution_hdf5 end
