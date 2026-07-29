# ── Fringe diagnostics ───────────────────────────────────────────────────────
#
# Report-style (non-plotting) diagnostics for a fringe `CalibrationSolution`,
# plus the pure gain-extraction helpers the Makie plot stubs consume. Everything
# here works off the solved model/θ/geometry and the solver's `info` NamedTuple
# (per-scan max SNR / χ / component count), so it is Makie-free and unit-testable
# without loading a plotting backend. The plot entry points themselves
# (`plot_fringe_spectrum`, `plot_fringe_phases`, `plot_fringe_snr`) are stubs in
# `Fringe.jl`, implemented by `GustavoMakieExt`.

"""
    fringe_snr_table(sol::CalibrationSolution) -> Vector{NamedTuple}

Per-scan fringe-fit summary rows `(scan, max_snr, chi, ncomp, pfa)` pulled from
the solver's `info` (`scan_max_snr` / `scan_chi` / `scan_ncomp` / `scan_ncells`,
as populated by [`fit`](@ref)). `pfa` is the scan's false-alarm
probability [`fringe_pfa`](@ref): the chance that pure noise, searched over the
scan's full delay×rate×baseline×product space, would produce a peak of at least
`max_snr` — `pfa ≪ 1` marks a secure detection, `pfa` near 1 a likely FALSE
fringe (`NaN` when the solution predates `scan_ncells`). Returns an empty vector
if the solution carries no per-scan diagnostics.
"""
function fringe_snr_table(sol::CalibrationSolution)
    info = sol.info
    (haskey(info, :scan_max_snr) && haskey(info, :scan_chi) && haskey(info, :scan_ncomp)) || return NamedTuple[]
    snr = info.scan_max_snr
    chi = info.scan_chi
    ncomp = info.scan_ncomp
    ncells = get(info, :scan_ncells, Float64[])
    n = min(length(snr), length(chi), length(ncomp))
    return [
        (;
                scan = s, max_snr = Float64(snr[s]), chi = Float64(chi[s]), ncomp = Int(ncomp[s]),
                pfa = s <= length(ncells) ? fringe_pfa(snr[s], ncells[s]) : NaN,
            )
            for s in 1:n
    ]
end

"""
    print_fringe_snr_table(rows; io = stdout)

Pretty-print the rows from [`fringe_snr_table`](@ref).
"""
function print_fringe_snr_table(rows; io = stdout)
    isempty(rows) && return println(io, "No per-scan fringe diagnostics available")
    println(io)
    println(io, "Fringe per-scan summary")
    println(io, "scan   max_snr      chi   ncomp        pfa")
    for r in rows
        println(
            io,
            lpad(string(r.scan), 4), "   ",
            lpad(_fmt(r.max_snr), 7), "   ",
            lpad(_fmt(r.chi), 6), "   ",
            lpad(string(r.ncomp), 5), "   ",
            lpad(_fmt_pfa(get(r, :pfa, NaN)), 8),
        )
    end
    return nothing
end

_fmt(x::Real) = isfinite(x) ? string(round(x; digits = 3)) : "NaN"

# PFA formatting: probabilities span many decades, so switch to scientific
# notation below 10⁻³ instead of rounding to "0.0".
_fmt_pfa(x::Real) = !isfinite(x) ? "NaN" :
    (x > 0 && x < 1.0e-3 ? @sprintf("%.1e", x) : string(round(x; digits = 3)))

"""
    fringe_solution_summary(sol::CalibrationSolution) -> String

One-line summary of a fringe solution: antenna/scan/parameter counts and the
median per-scan max SNR.
"""
function fringe_solution_summary(sol::CalibrationSolution)
    rows = fringe_snr_table(sol)
    nant = get(sol.info, :nant, sol.layout.nant)
    nscan = get(sol.info, :nscan, length(rows))
    snrs = [r.max_snr for r in rows if isfinite(r.max_snr)]
    medsnr = isempty(snrs) ? NaN : median(snrs)
    return string(
        "FringeSolution: ", nant, " antennas, ", nscan, " scans, ",
        sol.layout.nθ, " parameters; median scan max-SNR = ", _fmt(medsnr),
    )
end

"""
    fringe_scan_groups(uvset, sol::CalibrationSolution) -> Vector{NamedTuple}

Per scan-group metadata in solver order: `(; scan_index, source, scan, max_snr)`.
Lazy — reads only leaf metadata (no visibilities), so it is cheap on a streamed
`UVSet`. Use it to choose which scans to inspect with [`baseline_fringe_data`](@ref)
/ `plot_baseline_fringes`, e.g. the highest-SNR scan of each source:

    g = fringe_scan_groups(uvset, sol)
    best = argmax(r -> r.max_snr, filter(r -> r.source == "M87", g))
"""
function fringe_scan_groups(uvset::UVSet, sol::CalibrationSolution)
    specs = scan_stream(uvset; geom = sol.geom, ntasks = 1).groups
    snr = get(sol.info, :scan_max_snr, Float64[])
    out = NamedTuple[]
    for (gi, g) in enumerate(specs)
        push!(
            out, (;
                scan_index = gi, source = g.source, scan = g.scan,
                max_snr = gi <= length(snr) ? Float64(snr[gi]) : NaN,
            ),
        )
    end
    return out
end

# ── Gain extractors (pure; consumed by the Makie plot stubs and tests) ─────────

"""
    fringe_gain_spectrum(sol::CalibrationSolution; ti = 1) -> (freqs, gains)

Evaluate the solution's complex antenna gains at time index `ti` over the full
geometry. Returns `(channel_freqs::Vector, gains::Array{Complex,3})` with `gains`
shaped `(nchan, nant, 2)` (last axis = feed). The fringe model is phase-only, so
`angle.(gains)` is the diagnostic of interest (delay slope + constant + rate/adhoc
offset at this time).
"""
function fringe_gain_spectrum(sol::CalibrationSolution; ti::Integer = 1)
    ev = GainEvaluator(sol.model, sol.layout)
    ntime = sol.layout.ntime
    1 <= ti <= ntime || error("fringe_gain_spectrum: ti=$ti out of range 1:$ntime")
    # Window to the single requested time — evaluating the full (nchan × ntime)
    # cube just to slice one column is O(ntime) wasted work (and memory) on long
    # tracks.
    g = evaluate_gains(ev, sol.θ, 1:(sol.layout.nchan), ti:ti)   # (nchan, 1, nant, 2)
    return sol.geom.channel_freqs, g[:, 1, :, :]
end

"""
    fringe_bandpass_spectrum(sol::CalibrationSolution) -> (freqs, gains)

Like [`fringe_gain_spectrum`](@ref) but evaluates ONLY the phase bandpass
component — the per-scan delay slope (`2π·τ·(f−f0)`), constant, rate and R–L
terms are zeroed — so `angle.(gains)` is the RESIDUAL instrumental passband
ripple with the (large, station-dependent) delay wrap removed. This is the
readable bandpass diagnostic: without it, a station with a big group delay shows a
`2π·τ·(f−f0)` sawtooth that wraps many times across the band and buries the ripple.
Returns `(channel_freqs, gains::(nchan, nant, 2))`. The bandpass is time-invariant,
so no time index is needed. Errors if the model carries no bandpass component.
"""
function fringe_bandpass_spectrum(sol::CalibrationSolution)
    layout = sol.layout
    bp_i = findfirst(_is_bandpass, sol.model.phase)
    bp_i === nothing &&
        error("fringe_bandpass_spectrum: model has no phase bandpass component")
    # θ with every parameter zeroed EXCEPT the bandpass component's own
    # contiguous range, so `evaluate_gains` returns the bandpass-only gain (all
    # other terms → unit gain).
    θbp = fill!(similar(sol.θ), 0)
    rng = component_ranges(layout)[bp_i]
    θbp[rng] = sol.θ[rng]
    ev = GainEvaluator(sol.model, sol.layout)
    g = evaluate_gains(ev, θbp, 1:(layout.nchan), 1:1)   # time-invariant → any ti
    return sol.geom.channel_freqs, g[:, 1, :, :]
end

"""
    fringe_gain_time_series(sol::CalibrationSolution; ci = 1) -> (times, gains)

Evaluate the solution's complex antenna gains at channel index `ci` over the full
geometry. Returns `(times::Vector, gains::Array{Complex,3})` with `gains` shaped
`(ntime, nant, 2)`. `angle.(gains)` vs time shows the rate + adhoc phase evolution
per (station, feed).
"""
function fringe_gain_time_series(sol::CalibrationSolution; ci::Integer = 1)
    ev = GainEvaluator(sol.model, sol.layout)
    nchan = sol.layout.nchan
    1 <= ci <= nchan || error("fringe_gain_time_series: ci=$ci out of range 1:$nchan")
    # Window to the single requested channel (see fringe_gain_spectrum).
    g = evaluate_gains(ev, sol.θ, ci:ci, 1:(sol.layout.ntime))   # (1, ntime, nant, 2)
    return sol.geom.times, g[1, :, :, :]
end

"""
    fringe_station_solutions(sol::CalibrationSolution) -> Vector{NamedTuple}

Decode the stationized per-scan delay/rate/constant-phase parameters straight out
of `sol.θ` into a per-`(scan, station, feed)` table — no data read, no re-search.
One row per (scan-group index `scan`, 1-based `station`, `feed ∈ {1, 2}`):

- `delay_ns`  — station group delay (ns): the per-scan feed-common delay plus, on
  feed 2, the R–L delay the model fit (whether `:global` — the same offset every
  scan — or `:perscan`; see `_fringe_model`'s `rl_delay`).
- `rate_mHz`  — station fringe rate (mHz).
- `phase_deg` — station constant phase (deg): per-scan feed-common phase plus, on
  feed 2, the global R–L phase.

Summed from every stage-B component (`fringe_stage_components` — the delay/rate/
constant terms, EXCLUDING the per-AP adhoc, the per-channel bandpass, dTEC and SBD),
so it tracks the model automatically. Values are gauge-fixed to the solve's
reference pin; a within-scan difference against the SAME feed of a reference station
is gauge-invariant (the reported `delay_rel`/`rate_rel`).

The table is DENSE: a row is emitted for every layout slot, so a (station, feed,
scan) the solve never constrained reads back as the identity 0 (indistinguishable
here from the reference's genuine gauge-zero). Mask it with the detection/flag info
(`suspect_fringes` / `info.flagged_ant`/`flagged_scan`, and single-feed stations
via which feeds ever appear in `info.det_pol`) — this accessor deliberately stays a
pure θ-decode and does not consult the detections.

Scan index matches the scan-group ordering used by [`fringe_snr_table`](@ref) and
`info.det_scan` (the per-scan time segmentation is the scan-group partition).
"""
function fringe_station_solutions(sol::CalibrationSolution)
    layout = sol.layout
    θ = sol.θ
    nant = layout.nant
    comps = fringe_stage_components(sol.model, layout)     # (plan, kind ∈ :delay/:rate/:phase)
    refplan = _perscan_delay_plan(sol.model, layout)
    refplan === nothing &&
        error("fringe_station_solutions: model has no per-scan (feed-common) delay component")
    nscan = size(refplan.off1, 3)                          # PerScan ⇒ ntseg == #scan groups
    # First time index landing in each scan segment — used to look up every plan's
    # own segment id for this scan (a `GlobalTime` R–L plan maps them all to 1, a
    # `PerScan` one to the scan itself, so the same lookup handles both bases).
    t0 = zeros(Int, nscan)
    for ti in eachindex(refplan.tseg_id)
        k = refplan.tseg_id[ti]
        (1 <= k <= nscan && t0[k] == 0) && (t0[k] = ti)
    end
    out = NamedTuple[]
    for k in 1:nscan
        ti = t0[k]
        ti == 0 && continue
        for a in 1:nant, f in 1:2
            d = 0.0; r = 0.0; p = 0.0
            hd = false; hr = false; hp = false
            for (plan, kind) in comps
                col = plan.off1[a, f, plan.tseg_id[ti], 1]  # fseg 1: stage-B terms are GlobalFrequency
                col == 0 && continue
                v = θ[col]
                if kind === :delay
                    d += v; hd = true
                elseif kind === :rate
                    r += v; hr = true
                else
                    p += v; hp = true
                end
            end
            push!(
                out, (;
                    scan = k, station = a, feed = f,
                    delay_ns = hd ? d * 1e9 : NaN,       # τ (s) → ns
                    rate_mHz = hr ? r * 1e3 : NaN,       # ṙ (Hz) → mHz
                    phase_deg = hp ? rad2deg(p) : NaN,   # φ (rad) → deg
                ),
            )
        end
    end
    return out
end

# ── Per-baseline before/after data (the fringe-fit quality check) ──────────────

"""
    BaselineFringeData

Per-baseline coherent visibility averages for ONE scan, before and after applying
a fringe `CalibrationSolution`. Produced by [`baseline_fringe_data`](@ref) and
consumed by `plot_baseline_fringes`.

Fields: `source`/`scan`/`scan_index`/`max_snr` identify the scan; `bl_pairs` and
`pol_products` label the baseline and correlation axes; `freqs` (Hz, all bands
stacked) and `times` (h) the data axes. The four data arrays are weighted coherent
means (vector averages, `NaN` where a cell has no unflagged data):

- `spec_before`/`spec_after` — `(nchan, nbl, npol)`, averaged over time. `angle`
  vs frequency shows the group-delay slope (flat after a good fit); `abs` shows
  the band-averaged coherence.
- `tser_before`/`tser_after` — `(ntime, nbl, npol)`, averaged over frequency.
  `angle` vs time shows the fringe-rate slope (flat after a good fit).

Wide-band multi-group data (e.g. the four VGOS 3/5/6/10 GHz band groups) also
carries the per-BAND-GROUP split, so diagnostics can be viewed one band group at
a time (`plot_baseline_fringes(data; band = k)`):

- `band_groups` — channel ranges of each band group ([`fringe_band_groups`](@ref)
  of `freqs`; a single full range on contiguous data).
- `tser_band_before`/`tser_band_after` — `(ntime, nbl, npol, ngroups)`, the time
  series averaged over ONLY that band group's channels.
"""
struct BaselineFringeData
    source::String
    scan::String
    scan_index::Int
    max_snr::Float64
    bl_pairs::Vector{Tuple{Int, Int}}
    ant_names::Vector{String}        # station codes, indexed by antenna number
    pol_products::Vector{String}
    freqs::Vector{Float64}
    times::Vector{Float64}
    spec_before::Array{ComplexF64, 3}
    spec_after::Array{ComplexF64, 3}
    tser_before::Array{ComplexF64, 3}
    tser_after::Array{ComplexF64, 3}
    band_groups::Vector{UnitRange{Int}}
    tser_band_before::Array{ComplexF64, 4}
    tser_band_after::Array{ComplexF64, 4}
end

# Backwards-compatible constructor (no band-group split): one group spanning all
# channels, band time series = the full-band ones.
function BaselineFringeData(
        source, scan, scan_index, max_snr, bl_pairs, ant_names, pol_products,
        freqs, times, spec_before, spec_after, tser_before, tser_after,
    )
    nti, nbl, npol = size(tser_before)
    return BaselineFringeData(
        source, scan, scan_index, max_snr, bl_pairs, ant_names, pol_products,
        freqs, times, spec_before, spec_after, tser_before, tser_after,
        [1:length(freqs)],
        reshape(copy(tser_before), nti, nbl, npol, 1),
        reshape(copy(tser_after), nti, nbl, npol, 1),
    )
end

# The scan stream a diagnostic materializes through. The transform chain is the
# one RECORDED on `sol` by default, so diagnostics see exactly the data the
# solve saw (the "pass `weight_scale` again or this map's SNR won't match the
# solve" trap is gone); the legacy explicit kwargs (`precal`/`flag_channels`/
# `weight_scale`) override it when any is given, in the solver's application
# order (precal division, weight scale, channel mask); `transforms` overrides
# everything with an explicit chain — `transforms = ()` inspects the RAW data.
function _diag_stream(
        uvset::UVSet, sol::CalibrationSolution;
        precal = nothing, flag_channels = nothing, weight_scale = nothing,
        transforms = nothing,
    )
    tfs = if transforms !== nothing
        collect(Any, transforms)
    elseif precal !== nothing || flag_channels !== nothing || weight_scale !== nothing
        t = Any[]
        precal === nothing || push!(t, ApplySolution(precal))
        weight_scale === nothing || push!(t, StationWeightScale(weight_scale))
        flag_channels === nothing || push!(t, FlagChannels(BitVector(flag_channels)))
        t
    else
        sol.transforms
    end
    any(t -> t === missing, tfs) && error(
        "diagnostics: the solution records a transform that did not survive " *
            "serialization — pass the chain explicitly (precal/flag_channels/weight_scale)."
    )
    return scan_stream(uvset; geom = sol.geom, transforms = tfs, ntasks = 1)
end

# Scan group with the largest detection SNR (the most informative to inspect),
# falling back to the first group when no per-scan SNR is recorded.
function _max_snr_scan(sol::CalibrationSolution, ngroups::Integer)
    haskey(sol.info, :scan_max_snr) || return 1
    snr = sol.info.scan_max_snr
    (isempty(snr) || all(!isfinite, snr)) && return 1
    best = argmax(i -> (isfinite(snr[i]) ? snr[i] : -Inf), 1:min(length(snr), ngroups))
    return best
end

# Divide a weighted sum by its weight, leaving NaN where there was no data.
function _coherent_mean!(sum::Array{ComplexF64}, w::Array{Float64})
    @inbounds for i in eachindex(sum, w)
        sum[i] = w[i] > 0 ? sum[i] / w[i] : ComplexF64(NaN, NaN)
    end
    return sum
end

"""
    baseline_fringe_data(uvset, sol; scan_index = nothing) -> BaselineFringeData

Materialize one scan of `uvset` and compute, per baseline and correlation product,
the weighted coherent visibility average vs frequency and vs time, BEFORE and AFTER
dividing out the fringe solution `sol`. This is the per-baseline before/after check:
a good fit flattens the phase slopes (delay in frequency, rate in time) and lifts
the coherent amplitude.

`scan_index` selects which `(source, scan)` group (in the same order
the solve used); the default is the highest-SNR scan. The "after"
visibility is `V / (g_a · conj(g_b))` with gains evaluated from `sol` exactly as the
solver applies them — no second disk read of the full set, just this one scan.
When the solve used a `precal` (e.g. `phasecal_solution`), pass the same one here
so both BEFORE and AFTER are pre-calibrated the way the solver saw the data; the
same goes for `flag_channels` (e.g. `tone_channel_mask` — flagged channels are
zero-weighted, dropping out of the plotted averages exactly as they dropped out
of the solve) and `weight_scale` (the per-station weight correction — see
[`station_weight_scale`](@ref)).
"""
function baseline_fringe_data(
        uvset::UVSet, sol::CalibrationSolution;
        scan_index::Union{Integer, Nothing} = nothing,
        precal::Union{Nothing, CalibrationSolution} = nothing,
        flag_channels = nothing,
        weight_scale = nothing,
        transforms = nothing,
    )
    stream = _diag_stream(uvset, sol; precal, flag_channels, weight_scale, transforms)
    groups = stream.groups
    isempty(groups) && error("baseline_fringe_data: uvset has no scan groups")
    gi = scan_index === nothing ? _max_snr_scan(sol, length(groups)) : Int(scan_index)
    (1 <= gi <= length(groups)) || error("baseline_fringe_data: scan_index $gi out of range 1:$(length(groups))")

    info = UVData.metadata(last(first(groups[gi].leaves)))   # source/scan from the lazy leaf
    stack, win = materialize_cube(stream, groups[gi])
    ev = GainEvaluator(sol.model, sol.layout)
    g = evaluate_gains(ev, sol.θ, win.chan_idx, win.ti_idx)   # (nchan, nti, nant, 2)
    fg = frequencies(stack)
    Vg = stack[:vis]
    Wg = stack[:weights]
    nchan, nti, nbl, npol = size(Vg)

    # Band-group split of the stacked frequency axis (per-channel group id).
    bgs = fringe_band_groups(fg)
    ngrp = length(bgs)
    gid = Vector{Int}(undef, nchan)
    for (k, r) in enumerate(bgs), c in r
        gid[c] = k
    end

    sb = zeros(ComplexF64, nchan, nbl, npol); swb = zeros(Float64, nchan, nbl, npol)
    sa = zeros(ComplexF64, nchan, nbl, npol); swa = zeros(Float64, nchan, nbl, npol)
    tb = zeros(ComplexF64, nti, nbl, npol); twb = zeros(Float64, nti, nbl, npol)
    ta = zeros(ComplexF64, nti, nbl, npol); twa = zeros(Float64, nti, nbl, npol)
    tbb = zeros(ComplexF64, nti, nbl, npol, ngrp); twbb = zeros(Float64, nti, nbl, npol, ngrp)
    tab = zeros(ComplexF64, nti, nbl, npol, ngrp); twab = zeros(Float64, nti, nbl, npol, ngrp)

    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pol_products(stack)[p])
        for bi in 1:nbl
            a, b = UVData.baselines(stack).pairs[bi]
            a == b && continue                          # skip autocorrelations
            for ti in 1:nti, c in 1:nchan
                w = Wg[c, ti, bi, p]
                v = Vg[c, ti, bi, p]
                (w > 0 && isfinite(w) && isfinite(v)) || continue
                k = gid[c]
                sb[c, bi, p] += w * v; swb[c, bi, p] += w
                tb[ti, bi, p] += w * v; twb[ti, bi, p] += w
                tbb[ti, bi, p, k] += w * v; twbb[ti, bi, p, k] += w
                ga = g[c, ti, a, fa]; gb = g[c, ti, b, fb]
                denom = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(denom)) || continue
                vc = v / denom
                isfinite(vc) || continue
                # inverse-variance weight of the CORRECTED datum (Var(V/g) =
                # 1/(w·|g|²)) — matches apply_calibration's reweighting.
                wd = w * abs2(denom)
                sa[c, bi, p] += wd * vc; swa[c, bi, p] += wd
                ta[ti, bi, p] += wd * vc; twa[ti, bi, p] += wd
                tab[ti, bi, p, k] += wd * vc; twab[ti, bi, p, k] += wd
            end
        end
    end

    msnr = (haskey(sol.info, :scan_max_snr) && gi <= length(sol.info.scan_max_snr)) ?
        Float64(sol.info.scan_max_snr[gi]) : NaN
    return BaselineFringeData(
        info.source_name, info.scan_name, gi, msnr,
        copy(UVData.baselines(stack).pairs), String.(collect(info.antennas.name)), copy(pol_products(stack)),
        copy(fg), copy(timestamps(stack)),
        _coherent_mean!(sb, swb), _coherent_mean!(sa, swa),
        _coherent_mean!(tb, twb), _coherent_mean!(ta, twa),
        bgs,
        _coherent_mean!(tbb, twbb), _coherent_mean!(tab, twab),
    )
end

"""
    baseline_pol_index(data, pol) -> Int

Resolve a correlation-product selector (`Integer` index, or `String`/`Symbol`
label like `"PP"`) against `data.pol_products`. With `pol = :parallel` (the
default used by the plots) returns the first parallel-hand product.
"""
baseline_pol_index(data::BaselineFringeData, pol) = _pol_index(data.pol_products, pol)

"""
    fringe_band_groups(freqs; gap_factor = 4.0) -> Vector{UnitRange{Int}}

Group the contiguous sub-band blocks of a channel-frequency axis into BAND
GROUPS. The inter-block gaps are split into "within-group" vs "between-group"
scales at the largest ratio jump in their sorted values (must exceed
`gap_factor`); when the gaps carry no such two-scale structure the axis is one
group (a single far-flung pair is arbitrated against the block widths instead).
On VGOS this recovers the four widely-separated 3/5/6/10 GHz groups (each
holding several 32 MHz sub-bands); on a contiguous axis (e.g. VLBA) it returns
one full-range group. Channel ranges index the stacked frequency axis.
"""
function fringe_band_groups(freqs::AbstractVector{<:Real}; gap_factor::Real = 4.0)
    blocks = _band_ranges(freqs)
    length(blocks) <= 1 && return blocks
    gaps = [Float64(freqs[first(blocks[i + 1])] - freqs[last(blocks[i])]) for i in 1:(length(blocks) - 1)]
    thr = Inf
    if length(gaps) == 1
        # No gap statistics: a lone pair of blocks splits when the gap dwarfs the
        # blocks themselves.
        wmed = median([Float64(freqs[last(r)] - freqs[first(r)]) for r in blocks])
        gaps[1] > gap_factor * wmed && (thr = gap_factor * wmed)
    else
        s = sort(gaps)
        best = 0.0
        for i in 1:(length(s) - 1)
            s[i] > 0 || continue
            r = s[i + 1] / s[i]
            if r > best
                best = r
                thr = 0.5 * (s[i] + s[i + 1])
            end
        end
        best > gap_factor || (thr = Inf)
    end
    groups = UnitRange{Int}[]
    lo = first(blocks[1])
    for i in 1:(length(blocks) - 1)
        if gaps[i] > thr
            push!(groups, lo:last(blocks[i]))
            lo = first(blocks[i + 1])
        end
    end
    push!(groups, lo:last(blocks[end]))
    return groups
end

# Contiguous band ranges of a channel-frequency axis: split where the step
# jumps by more than 3× the median spacing (the VGOS sub-band gaps).
function _band_ranges(freqs::AbstractVector{<:Real})
    n = length(freqs)
    n == 0 && return UnitRange{Int}[]
    n == 1 && return [1:1]
    dfs = abs.(diff(Float64.(freqs)))
    step = median(dfs)
    ranges = UnitRange{Int}[]
    lo = 1
    for i in 1:(n - 1)
        if dfs[i] > 3 * step
            push!(ranges, lo:i)
            lo = i + 1
        end
    end
    push!(ranges, lo:n)
    return ranges
end

"""
    fringe_band_stats(data::BaselineFringeData; pol = :parallel)
        -> Vector{@NamedTuple{f_lo, f_hi, nchan, eta_before, eta_after}}

Per-band coherence summary of one scan's [`baseline_fringe_data`](@ref):
for each contiguous sub-band, the within-band coherence `|Σ_c z_c| / Σ_c |z_c|`
of the per-channel time-averaged visibilities, pooled over cross baselines —
BEFORE and AFTER the fringe solution. A band whose `eta_after` lags its
neighbours localises residual frequency structure (RFI, station passband
defect) to that band.
"""
function fringe_band_stats(data::BaselineFringeData; pol = :parallel)
    p = _pol_index(data.pol_products, pol)
    out = @NamedTuple{f_lo::Float64, f_hi::Float64, nchan::Int, eta_before::Float64, eta_after::Float64}[]
    for r in _band_ranges(data.freqs)
        stats = map((data.spec_before, data.spec_after)) do spec
            num = 0.0
            den = 0.0
            for (bi, (a, b)) in enumerate(data.bl_pairs)
                a == b && continue
                acc = zero(ComplexF64)
                s = 0.0
                for c in r
                    z = spec[c, bi, p]
                    (isfinite(real(z)) && isfinite(imag(z))) || continue
                    acc += z
                    s += abs(z)
                end
                num += abs(acc)
                den += s
            end
            den > 0 ? num / den : NaN
        end
        push!(
            out, (
                f_lo = data.freqs[first(r)], f_hi = data.freqs[last(r)],
                nchan = length(r), eta_before = stats[1], eta_after = stats[2],
            )
        )
    end
    return out
end

# Selector resolution shared by `baseline_pol_index` and `fringe_search_map`.
function _pol_index(pol_products::AbstractVector{<:AbstractString}, pol)
    return if pol === :parallel
        idx = findfirst(p -> (fp = correlation_feed_pair(p); fp[1] == fp[2]), pol_products)
        idx === nothing ? 1 : idx
    elseif pol isa Integer
        Int(pol)
    else
        idx = findfirst(==(String(pol)), pol_products)
        idx === nothing ? error("pol $(pol) not in $(pol_products)") : idx
    end
end

# Coherence-weighted group delay (s) of one baseline's spectrum `z` over `freqs`
# from the per-channel phase increment: τ = ⟨angle(z[c+1] z[c]*)⟩ / (2π Δf). Uses
# the wrapped increment (no unwrap needed) and weights by |z|; skips flagged cells
# and the sub-band-boundary jumps (Δf ≫ in-band spacing) where the increment wraps.
function _baseline_delay(z::AbstractVector, freqs::AbstractVector)
    df = Float64[]
    @inbounds for c in 1:(length(freqs) - 1)
        d = freqs[c + 1] - freqs[c]
        d > 0 && push!(df, d)
    end
    isempty(df) && return NaN
    dfmed = median(df)
    num = 0.0; den = 0.0
    @inbounds for c in 1:(length(z) - 1)
        z1 = z[c]; z2 = z[c + 1]
        (isfinite(z1) && isfinite(z2) && abs(z1) > 0 && abs(z2) > 0) || continue
        d = freqs[c + 1] - freqs[c]
        (d > 0 && d <= 3 * dfmed) || continue         # skip sub-band boundary jumps
        w = abs(z1) * abs(z2)
        num += w * angle(z2 * conj(z1))
        den += w * 2π * d
    end
    return den > 0 ? num / den : NaN
end

"""
    delay_closure(data::BaselineFringeData; pol = :parallel) -> NamedTuple

Triangle delay-closure check, the consistency test a station-based delay solution
must pass. For every closed triangle `(a,b,c)` it forms `τ_ab + τ_bc − τ_ac` from
the per-baseline group delays fitted to the coherent spectra:

- `closure_before` — from the raw data. A property of the *data*: real station-
  based delays cancel around a triangle, so these are ≈ 0 (up to noise). Large
  values would mean the data itself is non-closing (not something a fit can fix).
- `closure_after` — from the corrected data. Must stay ≈ 0 (a correct station-based
  solution cannot create closure errors).
- `resid_delay` — the per-baseline residual group delay after correction; a correct
  delay solution drives these to ≈ 0 on every baseline.

Returns `(; pol, triangles, closure_before, closure_after, resid_delay, bl_pairs)`.
A station-structure or sign mistake shows up as nonzero `resid_delay` (and, if it
breaks closure, nonzero `closure_after`).
"""
function delay_closure(data::BaselineFringeData; pol = :parallel)
    p = baseline_pol_index(data, pol)
    nbl = length(data.bl_pairs)
    τb = fill(NaN, nbl); τa = fill(NaN, nbl)
    for bi in 1:nbl
        a, b = data.bl_pairs[bi]
        a == b && continue
        τb[bi] = _baseline_delay(view(data.spec_before, :, bi, p), data.freqs)
        τa[bi] = _baseline_delay(view(data.spec_after, :, bi, p), data.freqs)
    end
    blindex = Dict(data.bl_pairs[bi] => bi for bi in 1:nbl)
    ants = sort(unique(Iterators.flatten(data.bl_pairs)))
    tris = NTuple{3, Int}[]; cb = Float64[]; ca = Float64[]
    for a in ants, b in ants, c in ants
        (a < b < c) || continue
        (haskey(blindex, (a, b)) && haskey(blindex, (b, c)) && haskey(blindex, (a, c))) || continue
        ab, bc, ac = blindex[(a, b)], blindex[(b, c)], blindex[(a, c)]
        (isfinite(τb[ab]) && isfinite(τb[bc]) && isfinite(τb[ac])) || continue
        push!(tris, (a, b, c))
        push!(cb, τb[ab] + τb[bc] - τb[ac])
        push!(ca, τa[ab] + τa[bc] - τa[ac])
    end
    return (;
        pol = data.pol_products[p], triangles = tris, closure_before = cb,
        closure_after = ca, data_delay = τb, resid_delay = τa, bl_pairs = copy(data.bl_pairs),
    )
end

# ── Delay–rate search map (the false-fringe check) ─────────────────────────────

"""
    BaselineFringeMap

The delay–rate search map of ONE (baseline, correlation product) of one scan,
with its scan/baseline labels — [`fringe_search_map`](@ref) output, consumed by
`plot_fringe_search`. `source`/`scan`/`scan_index` identify the scan; `bl_pair`
(antenna indices into `ant_names`) and `pol` the searched block; `map` is the
[`FringeSearchMap`](@ref) (axes, SNR surface, refined detection, `ncells`,
`pfa`).
"""
struct BaselineFringeMap
    source::String
    scan::String
    scan_index::Int
    bl_pair::Tuple{Int, Int}
    ant_names::Vector{String}
    pol::String
    map::FringeSearchMap
end

# Resolve a baseline selector against `bl_pairs`: an Integer index, an antenna-
# index pair, or a station-code pair (order-insensitive). `nothing` → caller
# picks a default.
function _baseline_index(bl_pairs, ant_names, sel)
    if sel isa Integer
        (1 <= sel <= length(bl_pairs)) || error("baseline index $sel out of range 1:$(length(bl_pairs))")
        return Int(sel)
    end
    a, b = if sel isa Tuple{<:Integer, <:Integer}
        Int(sel[1]), Int(sel[2])
    elseif sel isa Tuple && length(sel) == 2
        ia = findfirst(==(String(sel[1])), ant_names)
        ib = findfirst(==(String(sel[2])), ant_names)
        (ia === nothing || ib === nothing) &&
            error("baseline $(sel): station not in $(ant_names)")
        ia, ib
    else
        error("baseline selector must be an Integer index, an (a, b) antenna-index tuple, or a station-code tuple")
    end
    bi = findfirst(pr -> pr == (a, b) || pr == (b, a), bl_pairs)
    bi === nothing && error("baseline ($a, $b) not present in this scan")
    return bi
end

"""
    fringe_search_map(uvset, sol; scan_index = nothing, baseline = nothing,
                      pol = :parallel, search = nothing) -> BaselineFringeMap

Recompute the delay–rate matched-filter surface (the HOPS-style fringe plot data,
and THE false-fringe check) for one baseline of one scan of `uvset` — exactly the
search the fringe pass ran, but keeping the whole windowed `|D|` plane in
SNR units instead of only the peak. A real fringe is a single sharp peak far
above the sidelobe forest (`pfa ≪ 1`); a false fringe barely clears it.

- `scan_index` — which `(source, scan)` group, in solver order (see
  [`fringe_scan_groups`](@ref)); default the highest-SNR scan.
- `baseline` — an Integer index into the scan's baseline table, an antenna-index
  pair `(1, 3)`, or a station-code pair `("AA", "LM")` (order-insensitive).
  Default: the baseline with the strongest detection on this scan.
- `pol` — correlation-product selector as [`baseline_pol_index`](@ref)
  (default `:parallel`).
- `search` — `FringeSearch` options; defaults to the ones the solve used
  (recorded in `sol.info`).
- `precal` — when the solve used one (e.g. `phasecal_solution`), pass the same
  solution so the map is computed on the data the solver actually searched; the
  same goes for `flag_channels` (e.g. `tone_channel_mask`) and `weight_scale`
  (the per-station weight correction — see [`station_weight_scale`](@ref)),
  without which this map's SNR/PFA would not match the solve's.

Materializes only the one scan. Returns a [`BaselineFringeMap`](@ref).
"""
function fringe_search_map(
        uvset::UVSet, sol::CalibrationSolution;
        scan_index::Union{Integer, Nothing} = nothing,
        baseline = nothing, pol = :parallel,
        search::Union{FringeSearch, Nothing} = nothing,
        precal::Union{Nothing, CalibrationSolution} = nothing,
        flag_channels = nothing,
        weight_scale = nothing,
        transforms = nothing,
    )
    stream = _diag_stream(uvset, sol; precal, flag_channels, weight_scale, transforms)
    groups = stream.groups
    isempty(groups) && error("fringe_search_map: uvset has no scan groups")
    gi = scan_index === nothing ? _max_snr_scan(sol, length(groups)) : Int(scan_index)
    (1 <= gi <= length(groups)) || error("fringe_search_map: scan_index $gi out of range 1:$(length(groups))")

    info = UVData.metadata(last(first(groups[gi].leaves)))
    ant_names = String.(collect(info.antennas.name))
    stack, win = materialize_cube(stream, groups[gi])
    Vg = stack[:vis]
    Wg = stack[:weights]
    fg = frequencies(stack)
    opts = search === nothing ? get(sol.info, :search, FringeSearch()) : search
    p = _pol_index(pol_products(stack), pol)
    times = timestamps(stack) .* 3600.0
    f0 = sol.geom.f0
    t0 = sol.geom.t0 * 3600.0

    bi = if baseline === nothing
        # Default to the strongest detection at this product — the same search the
        # solver ran, sharing one workspace/axes across baselines.
        ax = _search_axes(fg, times, opts)
        ws = FringeWorkspace()
        best = 0
        bestsnr = -Inf
        for k in eachindex(UVData.baselines(stack).pairs)
            a, b = UVData.baselines(stack).pairs[k]
            a == b && continue
            d = _baseline_fringe_search(
                view(Vg, :, :, k, p), view(Wg, :, :, k, p),
                fg, times, f0, t0, ax, ws, opts,
            )
            d.snr > bestsnr && (bestsnr = d.snr; best = k)
        end
        best == 0 && error("fringe_search_map: scan has no cross baselines")
        best
    else
        k = _baseline_index(UVData.baselines(stack).pairs, ant_names, baseline)
        a, b = UVData.baselines(stack).pairs[k]
        a == b && error("fringe_search_map: ($a, $b) is an autocorrelation")
        k
    end

    m = baseline_fringe_map(
        view(Vg, :, :, bi, p), view(Wg, :, :, bi, p),
        fg, times, f0, t0; opts = opts,
    )
    return BaselineFringeMap(
        info.source_name, info.scan_name, gi, UVData.baselines(stack).pairs[bi], ant_names,
        pol_products(stack)[p], m,
    )
end

"""
    fringe_station_flags(sol::CalibrationSolution) -> Vector{NamedTuple}

The (station, scan) pairs the stage-B solve left UNCONSTRAINED — no strong
detection on any of the station's baselines survived the closure screen and
the robust rejection (the EHT-HOPS flag criterion). These stations carry
identity gains for those scans, and [`apply_calibration`](@ref) zero-weights
their baselines there (`apply_flags = true`). Rows
`(; scan, scan_name, ant, station)`; empty when every participating station
was constrained (or the solution predates flag recording).
"""
function fringe_station_flags(sol::CalibrationSolution)
    info = sol.info
    haskey(info, :flagged_ant) || return NamedTuple[]
    names = get(info, :ant_names, String[])
    sta(i) = i <= length(names) ? String(names[i]) : string("ant", i)
    scname(s) = s <= length(sol.geom.scan_names) ? String(sol.geom.scan_names[s]) : string(s)
    rows = [
        (;
                scan = Int(info.flagged_scan[i]), scan_name = scname(Int(info.flagged_scan[i])),
                ant = Int(info.flagged_ant[i]), station = sta(Int(info.flagged_ant[i])),
            )
            for i in eachindex(info.flagged_ant)
    ]
    sort!(rows; by = r -> (r.scan, r.ant))
    return rows
end

"""
    suspect_fringes(sol::CalibrationSolution; pfa_max = 1.0e-4) -> Vector{NamedTuple}

Screen the fringe solution for possible FALSE fringes: every valid detection the
stage-B solve consumed (recorded per baseline during the search pass) whose
per-baseline false-alarm probability exceeds `pfa_max`. Rows
`(; scan, a, b, sta_a, sta_b, pol, snr, pfa)`, most-suspect (largest `pfa`)
first; empty when every detection is secure (or the solution predates detection
recording). Needs NO data read — inspect a flagged row with

    m = fringe_search_map(uvset, sol; scan_index = r.scan, baseline = (r.a, r.b), pol = r.pol)
    plot_fringe_search(m)
"""
function suspect_fringes(sol::CalibrationSolution; pfa_max::Real = 1.0e-4)
    info = sol.info
    haskey(info, :det_pfa) || return NamedTuple[]
    names = get(info, :ant_names, String[])
    sta(i) = i <= length(names) ? String(names[i]) : string("ant", i)
    rows = [
        (;
                scan = Int(info.det_scan[i]), a = Int(info.det_ant_a[i]), b = Int(info.det_ant_b[i]),
                sta_a = sta(Int(info.det_ant_a[i])), sta_b = sta(Int(info.det_ant_b[i])),
                pol = String(info.det_pol[i]), snr = Float64(info.det_snr[i]), pfa = Float64(info.det_pfa[i]),
            )
            for i in eachindex(info.det_pfa) if info.det_pfa[i] > pfa_max
    ]
    sort!(rows; by = r -> r.pfa, rev = true)
    return rows
end

"""
    print_solve_timing(sol::CalibrationSolution; io = stdout, top = 5)

Profiling summary of a fringe solve from the timers the solve
records in `sol.info`: wall time per stage, the decode vs. search vs. adhoc vs.
reduce split summed over scans (task-seconds — with N concurrent group tasks the
wall share is up to N× smaller), and the `top` slowest scans. Prints a notice
when the solution predates the timers.
"""
function print_solve_timing(sol::CalibrationSolution; io = stdout, top::Integer = 5)
    info = sol.info
    haskey(info, :t_search_pass) || return println(io, "No solve timing recorded in this solution")
    sums(k) = haskey(info, k) ? sum(getproperty(info, k)) : 0.0
    dec1 = sums(:scan_t_decode)
    srch = sums(:scan_t_search)
    dec2 = sums(:scan_t_decode2)
    adh = sums(:scan_t_adhoc)
    red = sums(:scan_t_reduce)
    println(io)
    println(io, "Fringe solve timing (peak ", get(info, :ntasks_used, "?"), " concurrent groups × ", get(info, :inner_tasks, "?"), " inner tasks)")
    println(io, @sprintf("  search pass   %8.1f s wall   (Σ decode %8.1f s, Σ search %8.1f s)", info.t_search_pass, dec1, srch))
    println(io, @sprintf("  bandpass      %8.1f s wall", info.t_bandpass_stage))
    if haskey(info, :scan_t_reduce)
        println(io, @sprintf("  adhoc+reduce  %8.1f s wall   (Σ decode %8.1f s, Σ adhoc %8.1f s, Σ reduce %8.1f s)", info.t_adhoc_pass, dec2, adh, red))
    else
        println(io, @sprintf("  adhoc pass    %8.1f s wall   (Σ decode %8.1f s, Σ adhoc %8.1f s)", info.t_adhoc_pass, dec2, adh))
    end
    tot = [
        info.scan_t_decode[i] + info.scan_t_search[i] + info.scan_t_decode2[i] + info.scan_t_adhoc[i] +
            (haskey(info, :scan_t_reduce) ? info.scan_t_reduce[i] : 0.0) for i in eachindex(info.scan_t_decode)
    ]
    ord = sortperm(tot; rev = true)
    println(io, "  slowest scans (decode₁/search/decode₂/adhoc s):")
    for i in ord[1:min(Int(top), length(ord))]
        println(
            io, @sprintf(
                "    scan %3d  %6.1f = %.1f/%.1f/%.1f/%.1f", i, tot[i],
                info.scan_t_decode[i], info.scan_t_search[i], info.scan_t_decode2[i], info.scan_t_adhoc[i],
            )
        )
    end
    return nothing
end

"""
    print_delay_closure(c; io = stdout)

Summarize [`delay_closure`](@ref): RMS/max triangle closure (data and residual)
and the RMS/max residual per-baseline delay, all in ns.
"""
function print_delay_closure(c; io = stdout)
    rms(v) = (u = filter(isfinite, v); isempty(u) ? NaN : sqrt(sum(abs2, u) / length(u)))
    mx(v) = (u = abs.(filter(isfinite, v)); isempty(u) ? NaN : maximum(u))
    ns(x) = 1.0e9 * x
    println(io, "Delay closure [", c.pol, "], ", length(c.triangles), " triangles:")
    println(io, @sprintf("  data     closure rms = %9.4f ns  max = %9.4f ns", ns(rms(c.closure_before)), ns(mx(c.closure_before))))
    println(io, @sprintf("  residual closure rms = %9.4f ns  max = %9.4f ns", ns(rms(c.closure_after)), ns(mx(c.closure_after))))
    println(io, @sprintf("  residual delay   rms = %9.4f ns  max = %9.4f ns", ns(rms(c.resid_delay)), ns(mx(c.resid_delay))))
    return nothing
end
