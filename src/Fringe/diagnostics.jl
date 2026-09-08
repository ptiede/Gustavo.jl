# ── Fringe diagnostics ───────────────────────────────────────────────────────
#
# Report-style (non-plotting) diagnostics for a fringe `CalibrationSolution`,
# plus the pure gain-extraction helpers the Makie plot stubs consume. Everything
# here works off the solved model/θ/geometry and the solver's `info` NamedTuple
# (per-scan max SNR / component count), so it is Makie-free and unit-testable
# without loading a plotting backend. The plot entry points themselves
# (`plot_fringe_spectrum`, `plot_fringe_phases`, `plot_fringe_snr`) are stubs in
# `Fringe.jl`, implemented by `GustavoMakieExt`.

# The `:fringe` step's own `StepSolution`, or `nothing` — every diagnostic
# below degrades gracefully (empty/NaN) rather than erroring when the
# solution carries no fringe stage (hand-built, or predates this design).
_fringe_step(sol::CalibrationSolution) = begin
    i = findfirst(s -> s.name === :fringe, sol.steps)
    i === nothing ? nothing : sol.steps[i]
end

"""
    fringe_snr_table(sol::CalibrationSolution) -> Vector{NamedTuple}

Per-scan fringe-fit summary rows `(scan, max_snr, ncomp, pfa)`, all pulled
from the fringe step's own diagnostics (`stage_info(sol, :fringe)`). `ncomp` is a
solve-wide scalar (the same value on every row, not literally per-scan). `pfa` is the scan's false-alarm probability [`fringe_pfa`](@ref):
the chance that pure noise, searched over the scan's full
delay×rate×baseline×product space, would produce a peak of at least `max_snr`
— `pfa ≪ 1` marks a secure detection, `pfa` near 1 a likely FALSE fringe (`NaN`
when the solution predates `scan_ncells`). Returns an empty vector if the
solution carries no `:fringe` stage, or that stage no per-scan diagnostics.
"""
function fringe_snr_table(sol::CalibrationSolution)
    step = _fringe_step(sol)
    step === nothing && return NamedTuple[]
    info = step.info
    (haskey(info, :scan_snr) && haskey(info, :ncomp)) || return NamedTuple[]
    snr = info.scan_snr
    ncomp = Int(info.ncomp)
    ncells = get(info, :scan_ncells, Float64[])
    return [
        (;
            scan = s, max_snr = Float64(snr[s]), ncomp = ncomp,
            pfa = s <= length(ncells) ? fringe_pfa(snr[s], ncells[s]) : NaN,
        )
            for s in eachindex(snr)
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
    println(io, "scan   max_snr   ncomp        pfa")
    for r in rows
        println(
            io,
            lpad(string(r.scan), 4), "   ",
            lpad(_fmt(r.max_snr), 7), "   ",
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
    nant = get(sol.info, :nant, sol.steps[1].layout.nant)
    nscan = get(sol.info, :nscan, length(rows))
    snrs = [r.max_snr for r in rows if isfinite(r.max_snr)]
    medsnr = isempty(snrs) ? NaN : median(snrs)
    nθ = sum(s.layout.nθ for s in sol.steps)
    return string(
        "FringeSolution: ", nant, " antennas, ", nscan, " scans, ",
        nθ, " parameters; median scan max-SNR = ", _fmt(medsnr),
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
    specs = scan_stream(uvset; geom = sol.geom).groups
    step = _fringe_step(sol)
    snr = step === nothing ? Float64[] : get(step.info, :scan_snr, Float64[])
    return [
        (;
            scan_index = gi, source = g.source, scan = g.scan,
            max_snr = gi <= length(snr) ? Float64(snr[gi]) : NaN,
        )
            for (gi, g) in enumerate(specs)
    ]
end

"""
    fringe_station_solutions(sol::CalibrationSolution) -> Vector{NamedTuple}

Decode the stationized per-scan delay/rate/constant-phase parameters from
the fringe step's θ into a per-`(scan, station, feed)` table; no data is
read. One row per (scan-group index `scan`, 1-based `station`,
`feed ∈ {1, 2}`):

- `delay_ns`  — station group delay (ns): the feed-common delay plus, on
  feed 2, the fitted inter-feed offset.
- `rate_mHz`  — station fringe rate (mHz).
- `phase_deg` — station constant phase (deg), at the epoch the scan's rate
  column is referenced to (the scan's own mean time), so it is comparable
  across scans only through a difference taken within one scan.

Summed from every stage-B component the fringe stage owns (delay/rate/
constant terms — not adhoc, bandpass, dTEC, or SBD) plus, if a
`DispersionSBDFit` step ran, its private per-scan delay-refinement column.
Values are gauge-fixed to the solve's reference pin; a within-scan
difference against the same feed of a reference station is gauge-invariant
(the reported `delay_rel`/`rate_rel`).

The table is dense: a (station, feed, scan) the solve never constrained
reads back as the identity 0, indistinguishable from a genuine gauge-zero.
Mask it with the detection/flag info (`suspect_fringes`,
`info.flagged_ant`/`flagged_scan`); this accessor is a pure θ-decode and
does not consult the detections.

Scan index matches the scan-group ordering of [`fringe_snr_table`](@ref) and
`info.det_scan`.
"""
function fringe_station_solutions(sol::CalibrationSolution)
    fringe_step = sol[:fringe].steps[1]
    model, layout, θ = fringe_step.model, fringe_step.layout, fringe_step.θ
    nant = layout.nant
    # The fringe step's own model carries only its own components — a later
    # step (e.g. `DispersionSBDFit`) compiling a component sharing a stage-B
    # signature by design (see `fringe_stage_components`) lives in a separate
    # step's own model and never appears here, so no restriction is needed.
    comps = fringe_stage_components(model, layout)   # (plan, kind ∈ :delay/:rate/:phase)
    refplan = _perscan_delay_plan(model, layout)
    refplan === nothing &&
        error("fringe_station_solutions: model has no per-scan (feed-common) delay component")
    # `DispersionSBDFit`'s own delay-refinement column, if that step ran — a
    # Separate step's own `(model, layout, θ)`, not a positional trick against
    # a merged model, so its own plain `_perscan_delay_plan` finds it directly
    # (its own model has only one such component).
    refine_i = findfirst(s -> s.name === :refine, sol.steps)
    delay_refine_plan = refine_i === nothing ? nothing :
        _perscan_delay_plan(sol.steps[refine_i].model, sol.steps[refine_i].layout)
    delay_refine_θ = refine_i === nothing ? nothing : sol.steps[refine_i].θ
    nscan = refplan.shape[4]                                # PerScan ⇒ ntseg == #scan groups
    # First time index landing in each scan segment — used to look up every plan's
    # own segment id for this scan (a `GlobalTime` inter-feed plan maps them all to 1, a
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
                node = _feed_node(plan.tying, f)            # fseg 1: stage-B terms are GlobalFrequency
                node == 0 && continue
                v = _component_leaf(plan, θ)[1, node, 1, plan.tseg_id[ti], a]
                if kind === :delay
                    d += v; hd = true
                elseif kind === :rate
                    r += v; hr = true
                else
                    p += v; hp = true
                end
            end
            if delay_refine_plan !== nothing
                node = _feed_node(delay_refine_plan.tying, f)
                if node != 0
                    d += _component_leaf(delay_refine_plan, delay_refine_θ)[1, node, 1, delay_refine_plan.tseg_id[ti], a]
                    hd = true
                end
            end
            push!(
                out, (;
                    scan = k, station = a, feed = f,
                    delay_ns = hd ? d * 1.0e9 : NaN,       # τ (s) → ns
                    rate_mHz = hr ? r * 1.0e3 : NaN,       # ṙ (Hz) → mHz
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

Per-baseline coherent visibility averages for one scan, before and after applying
a fringe `CalibrationSolution`. Produced by [`baseline_fringe_data`](@ref) and
consumed by `plot_baseline_fringes`.

Fields: `source`/`scan`/`scan_index`/`max_snr` identify the scan; `bl_pairs` and
`pol_products` label the baseline and correlation axes; `freqs` (Hz, all spws
stacked) and `times` (h) the data axes. The four data arrays are weighted coherent
means (vector averages, `NaN` where a cell has no unflagged data):

- `spec_before`/`spec_after` — `(nchan, nbl, npol)`, averaged over time. `angle`
  vs frequency shows the group-delay slope (flat after a good fit); `abs` shows
  the group-averaged coherence.
- `tser_before`/`tser_after` — `(ntime, nbl, npol)`, averaged over frequency.
  `angle` vs time shows the fringe-rate slope (flat after a good fit).

Wide-band multi-group data (e.g. the four VGOS 3/5/6/10 GHz frequency groups) also
carries the per-frequency-group split, so diagnostics can be viewed one frequency
group at a time (`plot_baseline_fringes(data; freqgroup = k)`):

- `freq_groups` — channel ranges of each frequency group ([`fringe_freq_groups`](@ref)
  of `freqs`; a single full range on contiguous data).
- `tser_freqgroup_before`/`tser_freqgroup_after` — `(ntime, nbl, npol, ngroups)`, the time
  series averaged over only that frequency group's channels.
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
    freq_groups::Vector{UnitRange{Int}}
    tser_freqgroup_before::Array{ComplexF64, 4}
    tser_freqgroup_after::Array{ComplexF64, 4}
    # Thermal 1σ on each coherent mean above, in visibility units — the radial
    # width of the complex sample, from which a phase error is σ/|V|. `NaN`
    # where the mean has no contributing data, or where the caller did not
    # supply weights.
    spec_sigma_before::Array{Float64, 3}
    spec_sigma_after::Array{Float64, 3}
    tser_sigma_before::Array{Float64, 3}
    tser_sigma_after::Array{Float64, 3}
    tser_freqgroup_sigma_before::Array{Float64, 4}
    tser_freqgroup_sigma_after::Array{Float64, 4}
end

# 1σ on a weighted coherent mean: with `w = 1/σ_vis²` per sample, the mean
# `Σwv / Σw` has variance `1/Σw`. `_coherent_mean!` leaves its weight-sum
# argument untouched, so this reads the same accumulator the mean divided by.
_mean_sigma(wsum::AbstractArray) = map(x -> x > 0 ? 1 / sqrt(x) : NaN, wsum)

# The 1σ error bar on a plotted view of a complex sample, given the sample `z`
# and its radial width `σ`: `abs` sees σ itself, `angle` sees the angle σ
# subtends at radius |z| (the same small-angle form `phase_series_with_noise`
# uses). Once σ reaches |z| the phase is unconstrained, so the bar saturates at
# π rather than reporting a misleadingly finite width.
_plotted_sigma(::typeof(abs), z, σ) = σ
_plotted_sigma(::typeof(angle), z, σ) = abs(z) > 0 ? min(σ / abs(z), Float64(π)) : Float64(π)

# Backwards-compatible constructor (no frequency-group split): one group spanning
# all channels, per-group time series = the full-span ones.
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
        # No weights were supplied, so the thermal widths are unknown; plots
        # draw no error bars for a NaN σ.
        fill(NaN, size(spec_before)), fill(NaN, size(spec_after)),
        fill(NaN, size(tser_before)), fill(NaN, size(tser_after)),
        fill(NaN, nti, nbl, npol, 1), fill(NaN, nti, nbl, npol, 1),
    )
end

# The scan stream a diagnostic materializes through. The transform chain is the
# one recorded on `sol` by default, so diagnostics see exactly the data the
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
    # Diagnostics inspect one group at a time, so the OUTER level stays serial; the
    # inner level still fans out across the group's leaves and baselines.
    return scan_stream(
        uvset; geom = sol.geom, transforms = tfs,
        exec = ExecutionConfig(
            outer_executor = SerialScheduler(), inner_executor = DynamicScheduler(),
        ),
    )
end

# Scan group with the largest detection SNR (the most informative to inspect),
# falling back to the first group when no per-scan SNR is recorded.
function _max_snr_scan(sol::CalibrationSolution, ngroups::Integer)
    step = _fringe_step(sol)
    step === nothing && return 1
    snr = get(step.info, :scan_snr, Float64[])
    (isempty(snr) || all(!isfinite, snr)) && return 1
    return argmax(i -> (isfinite(snr[i]) ? snr[i] : -Inf), 1:min(length(snr), ngroups))
end

# Divide a weighted sum by its weight, leaving NaN where there was no data.
function _coherent_mean!(sum::Array{ComplexF64}, w::Array{Float64})
    @inbounds for i in eachindex(sum, w)
        sum[i] = w[i] > 0 ? sum[i] / w[i] : ComplexF64(NaN, NaN)
    end
    return sum
end

# Per-cell accumulation for `baseline_fringe_data`, in its own method so it
# specializes on the cube's concrete array types — inlined in the caller it runs
# untyped, at ~1 µs per cell over ~10^8 cells. One task per baseline: every
# accumulator is indexed by `bi`, so the tasks write to disjoint slices and the
# fan-out needs no reduction.
function _accumulate_baseline_fringes!(
        acc, Vg, Wg, g, gid, bl_pairs, feeds,
        nchan::Int, nti::Int, nbl::Int, npol::Int, executor,
    )
    tforeach(1:nbl; scheduler = executor) do bi
        a, b = bl_pairs[bi]
        a == b && return                                # skip autocorrelations
        for p in 1:npol
            fa, fb = feeds[p]
            @inbounds for ti in 1:nti, c in 1:nchan
                w = Wg[c, ti, bi, p]
                v = Vg[c, ti, bi, p]
                (w > 0 && isfinite(w) && isfinite(v)) || continue
                k = gid[c]
                acc.sb[c, bi, p] += w * v; acc.swb[c, bi, p] += w
                acc.tb[ti, bi, p] += w * v; acc.twb[ti, bi, p] += w
                acc.tbb[ti, bi, p, k] += w * v; acc.twbb[ti, bi, p, k] += w
                ga = g[c, ti, a, fa]; gb = g[c, ti, b, fb]
                denom = ga * conj(gb)
                # Squared magnitudes: the guard is a threshold test, and `abs` on a
                # complex number costs a `hypot` per cell.
                (abs2(ga) > 1.0e-24 && abs2(gb) > 1.0e-24 && isfinite(denom)) || continue
                vc = v / denom
                isfinite(vc) || continue
                # inverse-variance weight of the CORRECTED datum (Var(V/g) =
                # 1/(w·|g|²)) — matches apply_calibration's reweighting.
                wd = w * abs2(denom)
                acc.sa[c, bi, p] += wd * vc; acc.swa[c, bi, p] += wd
                acc.ta[ti, bi, p] += wd * vc; acc.twa[ti, bi, p] += wd
                acc.tab[ti, bi, p, k] += wd * vc; acc.twab[ti, bi, p, k] += wd
            end
        end
    end
    return nothing
end

"""
    baseline_fringe_data(uvset, sol; scan_index = nothing) -> BaselineFringeData

Materialize one scan of `uvset` and compute, per baseline and correlation product,
the weighted coherent visibility average vs frequency and vs time, before and after
dividing out the fringe solution `sol`. This is the per-baseline before/after check:
a good fit flattens the phase slopes (delay in frequency, rate in time) and lifts
the coherent amplitude.

`scan_index` selects which `(source, scan)` group (in the same order
the solve used); the default is the highest-SNR scan. The "after"
visibility is `V / (g_a · conj(g_b))` with gains evaluated from `sol` exactly as the
solver applies them — no second disk read of the full set, just this one scan.
When the solve used a `precal` (e.g. `phasecal_solution`), pass the same one here
so both before and after are pre-calibrated the way the solver saw the data; the
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
    executor = inner_executor(stream)                   # within-group fan-out
    stack, win = materialize_cube(stream, groups[gi])
    g = _composed_gains(sol, win.chan_idx, win.ti_idx)   # (nchan, nti, nant, 2)
    fg = frequencies(stack)
    Vg = stack[:vis]
    Wg = stack[:weights]
    nchan, nti, nbl, npol = size(Vg)

    # Frequency-group split of the stacked frequency axis (per-channel group id).
    bgs = fringe_freq_groups(fg)
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

    _accumulate_baseline_fringes!(
        (; sb, swb, sa, swa, tb, twb, ta, twa, tbb, twbb, tab, twab),
        Vg, Wg, g, gid, UVData.baselines(stack).pairs,
        [correlation_feed_pair(pol_products(stack)[p]) for p in 1:npol],
        nchan, nti, nbl, npol, executor,
    )

    fstep = _fringe_step(sol)
    fsnr = fstep === nothing ? Float64[] : get(fstep.info, :scan_snr, Float64[])
    msnr = gi <= length(fsnr) ? Float64(fsnr[gi]) : NaN
    # Before the means overwrite their numerators: the weight sums are the
    # accumulated inverse variances, so σ = 1/√Σw on every coherent mean.
    σ_sb = _mean_sigma(swb); σ_sa = _mean_sigma(swa)
    σ_tb = _mean_sigma(twb); σ_ta = _mean_sigma(twa)
    σ_tbb = _mean_sigma(twbb); σ_tab = _mean_sigma(twab)
    return BaselineFringeData(
        info.source_name, info.scan_name, gi, msnr,
        copy(UVData.baselines(stack).pairs), String.(collect(info.antennas.name)), copy(pol_products(stack)),
        copy(fg), copy(timestamps(stack)),
        _coherent_mean!(sb, swb), _coherent_mean!(sa, swa),
        _coherent_mean!(tb, twb), _coherent_mean!(ta, twa),
        bgs,
        _coherent_mean!(tbb, twbb), _coherent_mean!(tab, twab),
        σ_sb, σ_sa, σ_tb, σ_ta, σ_tbb, σ_tab,
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
    fringe_freq_group_stats(data::BaselineFringeData; pol = :parallel)
        -> Vector{@NamedTuple{f_lo, f_hi, nchan, eta_before, eta_after}}

Per-frequency-group coherence summary of one scan's [`baseline_fringe_data`](@ref):
for each contiguous frequency group, the within-group coherence `|Σ_c z_c| / Σ_c |z_c|`
of the per-channel time-averaged visibilities, pooled over cross baselines —
before and after the fringe solution. A frequency group whose `eta_after` lags its
neighbours localises residual frequency structure (RFI, station passband
defect) to that group.
"""
function fringe_freq_group_stats(data::BaselineFringeData; pol = :parallel)
    p = _pol_index(data.pol_products, pol)
    out = @NamedTuple{f_lo::Float64, f_hi::Float64, nchan::Int, eta_before::Float64, eta_after::Float64}[]
    for r in _freq_group_ranges(data.freqs)
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

The delay–rate search map of one (baseline, correlation product) of one scan,
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
and the false-fringe check) for one baseline of one scan of `uvset` — exactly the
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
        C = eltype(Vg)
        ax = _search_axes(fg, times, opts, C)
        ws = FringeWorkspace(C)
        family_cells = _search_cells(ax, opts)
        best = 0
        bestsnr = -Inf
        for k in eachindex(UVData.baselines(stack).pairs)
            a, b = UVData.baselines(stack).pairs[k]
            a == b && continue
            d = _baseline_fringe_search(
                view(Vg, :, :, k, p), view(Wg, :, :, k, p),
                fg, times, f0, t0, ax, ws, opts, family_cells,
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

The (station, scan) pairs the stage-B solve left UNCONSTRAINED — no ACCEPTED
detection (`pfa <= Stationization.pfa_max`) on any of the station's baselines,
so nothing put it in a fringe group (the EHT-HOPS flag criterion). A measured
but rejected baseline does not rescue it: such a row constrains the fit without
fixing a fringe location. These stations carry
identity gains for those scans, and [`apply_calibration`](@ref Gustavo.UVData.apply_calibration) zero-weights
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

Screen the solution's ACCEPTED detections against a false-alarm probability of
`pfa_max`: the rows the stage-B solve treated as real fringes whose PFA exceeds
the threshold given here. The search pass records every measured cell, accepted
or not, so this reads the `detected` ones only — a rejected cell is not a suspect
fringe, it is a non-detection.

At the default this returns the empty set by construction, since acceptance is a
PFA test at the solve's own `Stationization.pfa_max`. It earns its keep when
passed something STRICTER than the solve used: those are the accepted detections
that would flip under a tighter threshold, i.e. the marginal ones worth eyeballing.

Rows `(; scan, a, b, sta_a, sta_b, pol, snr, pfa)`, most-suspect (largest `pfa`)
first. Needs no data read — inspect a flagged row with

    m = fringe_search_map(uvset, sol; scan_index = r.scan, baseline = (r.a, r.b), pol = r.pol)
    plot_fringe_search(m)
"""
function suspect_fringes(sol::CalibrationSolution; pfa_max::Real = 1.0e-4)
    step = _fringe_step(sol)
    step === nothing && return NamedTuple[]
    info = step.info
    haskey(info, :det_pfa) || return NamedTuple[]
    names = get(sol.info, :ant_names, String[])
    sta(i) = i <= length(names) ? String(names[i]) : string("ant", i)
    # A solution written before the table recorded rejected cells holds detections
    # only, so every row of one counts as accepted.
    detected = get(info, :det_detected, nothing)
    accepted(i) = detected === nothing || detected[i]
    rows = [
        (;
            scan = Int(info.det_scan[i]), a = Int(info.det_ant_a[i]), b = Int(info.det_ant_b[i]),
            sta_a = sta(Int(info.det_ant_a[i])), sta_b = sta(Int(info.det_ant_b[i])),
            pol = String(info.det_pol[i]), snr = Float64(info.det_snr[i]), pfa = Float64(info.det_pfa[i]),
        )
            for i in eachindex(info.det_pfa) if accepted(i) && info.det_pfa[i] > pfa_max
    ]
    sort!(rows; by = r -> r.pfa, rev = true)
    return rows
end

"""
    print_solve_timing(sol::CalibrationSolution; io = stdout, top = 5)

Profiling summary of a solve, GENERIC over every step (built-in or
third-party): one line per step that published timing (`stage_info(sol,
name).t_pass`, the pass's total wall time, and `.timing`, a `Scan`-indexed
`DimStack` of `decode`/`work`/`reduce` task-seconds — with N concurrent group
tasks the wall share is up to N× smaller), then the `top` slowest scans of
whichever step spent the most per-scan time. Prints a notice when the
solution carries no step timing at all.
"""
function print_solve_timing(sol::CalibrationSolution; io = stdout, top::Integer = 5)
    timed = [s for s in sol.steps if haskey(s.info, :t_pass)]
    isempty(timed) && return println(io, "No solve timing recorded in this solution")
    println(io)
    println(
        io, "Solve timing (peak ", get(sol.info, :ntasks_used, "?"), " concurrent groups × ",
        get(sol.info, :inner_tasks, "?"), " inner tasks)",
    )
    for s in timed
        line = @sprintf("  %-10s %8.1f s wall", String(s.name), s.info.t_pass)
        if haskey(s.info, :timing)
            t = s.info.timing
            line *= @sprintf(
                "   (Σ decode %8.1f s, Σ work %8.1f s, Σ reduce %8.1f s)",
                sum(t.decode), sum(t.work), sum(t.reduce),
            )
        end
        println(io, line)
    end
    heaviest = argmax(s -> haskey(s.info, :timing) ? sum(s.info.timing.work) : -Inf, timed)
    haskey(heaviest.info, :timing) || return nothing
    t = heaviest.info.timing
    tot = t.decode .+ t.work .+ t.reduce
    ord = sortperm(tot; rev = true)
    println(io, "  slowest scans (", heaviest.name, ", decode/work/reduce s):")
    for i in ord[1:min(Int(top), length(ord))]
        println(io, @sprintf("    scan %3d  %6.1f = %.1f/%.1f/%.1f", i, tot[i], t.decode[i], t.work[i], t.reduce[i]))
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
