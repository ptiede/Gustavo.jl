# ── Per-scan dTEC/SBD refinement over ScanDataViews (the carved-out service) ──
#
# The per-scan (Δτ, dTEC) band-phasor refinement and the per-band-group SBD
# refinement, operating on a `ScanDataView` (descended verbatim from the
# monolithic solver's concat-cube variants, deleted at M5). The θ slots they
# write are OWNED by the FringeFit step; other
# stages invoke them through [`refine_scan!`](@ref) — the bandpass stage
# dispersion/SBD-corrects each calibrator scan BEFORE accumulating (scans with
# different ionospheres would otherwise decohere the frozen per-channel curve),
# and the temporal-smoother stage re-refines every scan on the bandpass-
# corrected residual. The pure fit/stationize back halves
# (`_dispersion_fit_stationize!`, `_sbd_fit_stationize!`, `_fit_band_dispersion`,
# `_fit_chunk_delay`, `_accumulate_leaf_band_phasor!`, `_accumulate_leaf_chunks!`)
# live at the foot of this file.

# The refine "service" bundle a FringeFit publishes for later stages: the plans
# of the θ components it owns (per-scan delay, dTEC, SBD delay + constant), the
# co-located dTEC ties, and the polish contract — whether scans another stage
# already refined should be POLISHED in a narrow window instead of re-run with
# the full grid (`reuse_bandpass`), and the polish dTEC half-window (TECU,
# `polish_dtec`). `nothing` plans disable the matching refinement.
struct RefineService
    disp_plan::Any
    ps_delay_plan::Any
    sbd_plans::Any
    ties::Any
    reuse_bandpass::Bool
    polish_dtec::Float64
end

"""
    refine_scan!(θ, v::ScanDataView, geom, ev, rf::RefineService, ref_ant, nant;
                 inner = 1, polish = false) -> nrej

Refine one scan's FringeFit-owned per-scan θ columns on the current residual:
the joint (Δτ, dTEC) fit ([`refine_scan_dispersion!`](@ref)) followed by the
per-band-group SBD fit ([`refine_scan_sbd!`](@ref), on the dispersion-corrected
residual). Writes only this scan's θ columns — disjoint per scan, so concurrent
groups may refine in parallel. `polish = true` runs the dispersion fit in the
narrow window around the already-fit values (`rf.polish_dtec` TECU, the
monolith's `reuse_bandpass_refine` reuse path — cheap, and keeps the increment
a full skip would drop); SBD is cheap and always full-refines. Returns the
number of detections the robust dispersion station solve excised.
"""
function refine_scan!(
        θ, v::ScanDataView, geom::DataGeometry, ev, rf::RefineService, ref_ant, nant;
        inner::Integer = 1, polish::Bool = false,
    )
    nrej = polish ?
        refine_scan_dispersion!(
            θ, v, geom, ev, rf.ps_delay_plan, rf.disp_plan, ref_ant, nant;
            inner = inner, ties = rf.ties,
            tau_max = _DTEC_POLISH_TAU, dtec_max = rf.polish_dtec,
        ) :
        refine_scan_dispersion!(
            θ, v, geom, ev, rf.ps_delay_plan, rf.disp_plan, ref_ant, nant;
            inner = inner, ties = rf.ties,
        )
    rf.sbd_plans === nothing ||
        refine_scan_sbd!(θ, v, geom, ev, rf.sbd_plans, ref_ant, nant; inner = inner)
    return nrej
end

# The view's grp-local channel ranges per spectral window (the concat cube's
# per-band blocks, recovered from the global channel indices).
function _spw_blocks(geom::DataGeometry, chan_idx)
    soc = geom.spw_of_chan
    blocks = UnitRange{Int}[]
    lo = 1
    for c in 2:length(chan_idx)
        if soc[chan_idx[c]] != soc[chan_idx[c - 1]]
            push!(blocks, lo:(c - 1))
            lo = c
        end
    end
    push!(blocks, lo:length(chan_idx))
    return blocks
end

"""
    refine_scan_dispersion!(θ, v::ScanDataView, geom, ev, delay_plan, disp_plan,
                            ref_ant, nant; opts, snr_min, tau_max, dtec_max,
                            inner, ties) -> nrej

Joint per-scan (Δτ, dTEC) refinement of one scan view: per-spw band phasors →
per-baseline joint fits → two station solves accumulating into the per-scan
delay and dispersion θ columns. The view's per-spw channel blocks stand in for
the band leaves (the concat-cube path). Returns the number of detections the
robust station solves excised; a no-op (0) when `disp_plan === nothing` or
fewer than 4 bands are present (1/ν is unconstrainable).
"""
function refine_scan_dispersion!(
        θ, v::ScanDataView, geom::DataGeometry, ev, delay_plan, disp_plan, ref_ant, nant;
        opts::Stationization = Stationization(reject_iters = 0), snr_min::Real = 8.0,
        tau_max::Real = 2.0e-8, dtec_max::Real = 45.0, inner::Integer = 1,
        ties = nothing,
    )
    disp_plan === nothing && return 0
    ci = v.chan_idx
    ti = v.ti_idx
    blocks = _spw_blocks(geom, ci)
    length(blocks) >= 4 || return 0              # < 4 bands can't constrain 1/ν
    V = v.vis
    W = v.weights
    fg = v.freqs
    bl_pairs = v.bl_pairs
    pols = v.pol_products
    feeds = [correlation_feed_pair(p) for p in pols]
    nbl = length(bl_pairs)
    npol = length(pols)
    nlf = length(blocks)
    z = zeros(ComplexF64, nbl, npol, nlf)
    w = zeros(Float64, nbl, npol, nlf)
    fb = [sum(@view fg[r]) / length(r) for r in blocks]
    exec_foreach(1:nlf; ntasks = clamp(Int(inner), 1, nlf)) do li
        r = blocks[li]
        g = evaluate_gains(ev, θ, ci[r], ti)
        _accumulate_leaf_band_phasor!(
            view(z, :, :, li), view(w, :, :, li),
            view(V, r, :, :, :), view(W, r, :, :, :), g,
            bl_pairs, pols,
        )
    end
    return _dispersion_fit_stationize!(
        θ, z, w, fb, bl_pairs, pols, feeds, first(ti), geom,
        delay_plan, disp_plan, ref_ant, opts,
        Float64(snr_min), Float64(tau_max), Float64(dtec_max), ties,
    )
end

"""
    refine_scan_sbd!(θ, v::ScanDataView, geom, ev, sbd, ref_ant, nant;
                     nchunk = 4, snr_min = 8.0, tau_max = 6.0e-8, inner = 1) -> nrej

Per-scan per-band-group SBD refinement of one scan view (fourfit's single-band
delay): each spw block's sub-band chunk phasors → per-(baseline, group) exact
matched-filter slope fits → guarded station solves accumulating into the
per-scan `Delay × FrequencyBands` column and its companion constant. Run AFTER
the dispersion refinement so the within-band slopes it fits are
dispersion-corrected. A no-op (0) when `sbd === nothing`.
"""
function refine_scan_sbd!(
        θ, v::ScanDataView, geom::DataGeometry, ev, sbd, ref_ant, nant;
        nchunk::Integer = 4, snr_min::Real = 8.0, tau_max::Real = 6.0e-8, inner::Integer = 1,
    )
    sbd === nothing && return 0
    ci = v.chan_idx
    ti = v.ti_idx
    blocks = _spw_blocks(geom, ci)
    V = v.vis
    W = v.weights
    fg = v.freqs
    bl_pairs = v.bl_pairs
    pols = v.pol_products
    feeds = [correlation_feed_pair(p) for p in pols]
    nbl = length(bl_pairs)
    npol = length(pols)
    nlf = length(blocks)
    ntot = nlf * Int(nchunk)
    z = zeros(ComplexF64, nbl, npol, ntot)
    w = zeros(Float64, nbl, npol, ntot)
    chunkf = zeros(Float64, ntot)
    chunkgrp = zeros(Int, ntot)
    exec_foreach(1:nlf; ntasks = clamp(Int(inner), 1, nlf)) do li
        r = blocks[li]
        g = evaluate_gains(ev, θ, ci[r], ti)
        fs = fg[r]
        nc = length(fs)
        bgrp = findfirst(rr -> ci[first(r)] in rr, sbd.bands)
        bgrp === nothing && return
        edges = round.(Int, range(0, nc; length = Int(nchunk) + 1))
        coc = Vector{Int}(undef, nc)
        for k in 1:Int(nchunk)
            klo, khi = edges[k] + 1, edges[k + 1]
            khi >= klo || continue
            kk = (li - 1) * Int(nchunk) + k
            coc[klo:khi] .= kk
            chunkf[kk] = sum(@view fs[klo:khi]) / (khi - klo + 1)
            chunkgrp[kk] = bgrp
        end
        _accumulate_leaf_chunks!(
            view(z, :, :, ((li - 1) * Int(nchunk) + 1):(li * Int(nchunk))),
            view(w, :, :, ((li - 1) * Int(nchunk) + 1):(li * Int(nchunk))),
            view(V, r, :, :, :), view(W, r, :, :, :), g,
            bl_pairs, pols,
            coc .- (li - 1) * Int(nchunk),
        )
    end
    return _sbd_fit_stationize!(
        θ, z, w, chunkf, chunkgrp, bl_pairs, pols, feeds,
        first(ti), geom, sbd, ref_ant, nant;
        snr_min = Float64(snr_min), tau_max = Float64(tau_max),
    )
end

"""
    adhoc_scan!(θ, v::ScanDataView, geom, ev, adhoc_plan, adhoc, ref_ant, nant;
                shared_feeds = false, inner = 1, excl = nothing, psI = nothing) -> θ

The per-integration atmospheric-phase (adhoc) solve of one scan view: accumulate
the per-(baseline, product, AP) inverse-variance residual `V/g` (weight
`w·|g|²` — the same reweighting `apply_calibration` applies; raw `w` would
up-weight exactly the channels the amp bandpass marked low-|g|), solve the
globally-closing per-AP station phase through the pluggable `adhoc` smoother
([`solve_adhoc_phasing`](@ref)), and write this scan's `PerIntegration` θ slots
(disjoint per scan — concurrent groups may solve in parallel). `excl` drops
co-located (intra-site) baselines from the per-AP solve; `psI` (linear-feed
data) collapses the four products to one pseudo-Stokes-I row per (baseline, AP)
using the field-rotation coefficients.
"""
function adhoc_scan!(
        θ, v::ScanDataView, geom::DataGeometry, ev, adhoc_plan, adhoc, ref_ant, nant;
        shared_feeds::Bool = false, inner::Integer = 1, excl = nothing, psI = nothing,
    )
    bl_pairs = collect(v.bl_pairs)
    pols = String.(v.pol_products)
    tg = Float64.(v.times)
    ci = v.chan_idx
    g_ti = v.ti_idx
    nbl = length(bl_pairs)
    npol = length(pols)
    nap = length(tg)
    # Per-band accumulation (the view's per-spw channel blocks stand in for the
    # band leaves) fanned out over `inner` tasks — this loop (gain evaluation +
    # residual sum over every visibility) dominates the adhoc pass on many-band
    # data. Each BLOCK gets its own partial and the partials fold in block
    # order, so the float association is fixed by the data layout alone — the
    # result is bit-deterministic at any `inner`/ntasks (the monolith's
    # chunk-local partials were not: chunk boundaries moved with `inner`).
    blocks = _spw_blocks(geom, ci)
    nblk = length(blocks)
    parts = Vector{Tuple{Array{ComplexF64, 3}, Array{Float64, 3}}}(undef, nblk)
    exec_foreach(1:nblk; ntasks = clamp(Int(inner), 1, nblk)) do li
        r = blocks[li]
        rl = zeros(ComplexF64, nbl, npol, nap)
        wl = zeros(Float64, nbl, npol, nap)
        g = evaluate_gains(ev, θ, ci[r], g_ti)           # (nchan_block, nti, nant, 2)
        _accumulate_leaf_rbar!(
            rl, wl, view(v.vis, r, :, :, :), view(v.weights, r, :, :, :),
            g, bl_pairs, pols,
        )
        parts[li] = (rl, wl)
    end
    rbar = zeros(ComplexF64, nbl, npol, nap)
    wbar = zeros(Float64, nbl, npol, nap)
    for (rl, wl) in parts
        rbar .+= rl
        wbar .+= wl
    end
    # Drop co-located (intra-site) baselines from the per-AP solve — see
    # `_colocated_pair_set`. Zero weight ⇒ `_adhoc_ap_rows` skips the rows.
    if excl !== nothing
        for bi in eachindex(bl_pairs)
            bl_pairs[bi] in excl || continue
            fill!(view(rbar, bi, :, :), zero(ComplexF64))
            fill!(view(wbar, bi, :, :), 0.0)
        end
    end
    # Linear-feed data: collapse the four products to one pseudo-Stokes-I row
    # per (baseline, AP) using the field-rotation coefficients — see
    # `_pseudo_stokes_collapse!` (rotation-robust; single products can null).
    if psI !== nothing
        m0 = DimensionalData.metadata(getfield(v, :data))
        jds = [psI.base_jd + Float64(t) / 24.0 for t in tg]
        ψ = _field_rotation_angles(m0.antennas, m0.ra, m0.dec, jds)
        _pseudo_stokes_collapse!(rbar, wbar, bl_pairs, pols, ψ)
    end
    # `tg` is in hours; pass SECONDS so the adhoc's `:auto` window (T_AP / T_coh) is
    # in physical units. Detrend uses only the mean, so the scaling is otherwise inert.
    as = solve_adhoc_phasing(rbar, wbar, bl_pairs, pols, nant, tg .* 3600.0; ref_ant = ref_ant, smoother = adhoc, shared_feeds = shared_feeds)
    for (ap, gti) in enumerate(g_ti)
        tseg = adhoc_plan.tseg_id[gti]
        for ant in 1:nant, feed in 1:2
            val = as.phase[ant, feed, ap]
            isfinite(val) || continue
            off = adhoc_plan.off1[ant, feed, tseg, 1]
            off == 0 && continue
            θ[off] = val
        end
    end
    return θ
end

# ── Pure fit kernels (relocated verbatim from the deleted monolith) ──────────

# Accumulate one band leaf's per-AP residual into `rbar`/`wbar` as the
# inverse-variance mean of the CORRECTED data: `V/den` has variance 1/(w·|den|²)
# (Var(V) = 1/w), so its weight is `w·|den|²` — the same reweighting
# `apply_calibration` applies. Accumulating with the RAW `w` instead is only
# correct for |den| = 1 (phase-only gains); once the log-amp bandpass is in θ
# it UP-weights exactly the channels the amp solution marked low-|g| —
# amplitude-inflated noise dominating the per-AP phasor (on VR2505 this tripled
# K2's adhoc track noise, since its low band's |g| dip encodes its own phase
# scramble). Function barrier: `V`/`W` from `parent(leaf[...])` are
# type-unstable at the call site.
function _accumulate_leaf_rbar!(rbar, wbar, V, W, g, bl_pairs, pols)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            for tt in 1:nti, c in 1:nchan
                w = W[c, tt, bi, p]
                (w > 0 && isfinite(w)) || continue
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, tt, bi, p] / den
                isfinite(v) || continue
                wd = w * abs2(den)
                rbar[bi, p, tt] += wd * v
                wbar[bi, p, tt] += wd
            end
        end
    end
    return rbar, wbar
end

# ── Per-scan (Δτ, dTEC) band-phasor refinement ────────────────────────────────
#
# The FFT search + stationization solve a per-scan LINEAR delay; the ionosphere
# adds a dispersive phase K·dTEC·(1/f0 − 1/f) whose linear-in-ν part the delay
# absorbs (biasing it by hundreds of ps on VGOS) and whose curvature survives as
# cross-band structure that no GLOBAL per-channel bandpass can track scan-to-scan.
# This stage measures both self-consistently from each scan's own residual —
# fourfit's ionospheric search, done as a per-baseline (Δτ, dTEC) grid fit over
# the scan's band phasors, then stationized through `solve_station_systems!`
# (closure screen + robust rejection included) into the per-scan delay and
# dispersion θ columns. Runs inside pass 2 on the already-materialized leaves, so
# it costs no extra read; the adhoc solve then sees dispersion-corrected
# residuals. Feed-common (ionosphere is non-birefringent to first order); cross
# hands are skipped like the rate solve. It ALSO runs inside the bandpass stage
# on each accumulated calibrator scan (see `_solve_bandpass_stage!`), so the
# frozen phase bandpass is solved from dispersion-corrected residuals; each
# scan's pass-2 refinement then measures the residual dTEC against that CLEAN
# curve (θ increments compose — for the accumulated scans pass 2 is a small
# polish on top of the bandpass-stage fit).

# Collapse one band leaf to one residual phasor per (baseline, product):
# `z[bi, p] = Σ w·|den|²·(V/den)`, `w[bi, p] = Σ w·|den|²` over the leaf's
# channels × APs — the inverse-variance mean of the corrected data (see
# `_accumulate_leaf_rbar!` for why the |den|² reweighting is required once
# amp gains live in θ). Function barrier (V/W type-unstable at the call site).
function _accumulate_leaf_band_phasor!(z, w, V, W, g, bl_pairs, pols)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        fa == fb || continue                    # parallel hands only
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            acc = zero(ComplexF64)
            wsum = 0.0
            for tt in 1:nti, c in 1:nchan
                ww = W[c, tt, bi, p]
                (ww > 0 && isfinite(ww)) || continue
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, tt, bi, p] / den
                isfinite(v) || continue
                wd = ww * abs2(den)
                acc += wd * v
                wsum += wd
            end
            z[bi, p] = acc
            w[bi, p] = wsum
        end
    end
    return nothing
end

# Joint (Δτ, dTEC) fit of one baseline's band phasors: coarse-to-fine grid
# maximization of |Σ_b z_b·cis(−2πτ(f_b−f0) − K·dt·(1/f0−1/f_b))| — the exact
# matched filter over the two smooth terms (phases are wrapped, so no linear
# fit applies). SNR is the debiased coherent amplitude over √Var, Var(Σz) = Σw
# for inverse-variance weights.
function _fit_band_dispersion(
        fbs::Vector{Float64}, zs::Vector{ComplexF64}, ws::Vector{Float64}, f0::Float64;
        tau_max::Float64 = 2.0e-8, dtec_max::Float64 = 45.0,
    )
    nb = length(fbs)
    xdisp = [Calibration.DISPERSION_K * (1.0 / f0 - 1.0 / fbs[b]) for b in 1:nb]
    xtau = [2π * (fbs[b] - f0) for b in 1:nb]
    best_a = -1.0
    best_d = 0.0
    best_t = 0.0
    y = Vector{ComplexF64}(undef, nb)
    function sweep(dts, τs)
        for dt in dts
            @inbounds for b in 1:nb
                y[b] = zs[b] * cis(-dt * xdisp[b])
            end
            for τ in τs
                acc = zero(ComplexF64)
                @inbounds for b in 1:nb
                    acc += y[b] * cis(-τ * xtau[b])
                end
                a = abs(acc)
                if a > best_a
                    best_a = a
                    best_d = dt
                    best_t = τ
                end
            end
        end
        return nothing
    end
    sweep(-dtec_max:0.25:dtec_max, -tau_max:2.0e-11:tau_max)
    sweep((best_d - 0.3):0.01:(best_d + 0.3), (best_t - 3.0e-11):1.0e-12:(best_t + 3.0e-11))
    # Parabolic sub-grid polish (one axis at a time): at high SNR the CRB is far
    # below the fine-grid step, and leftover quantization would read as a
    # significant residual to the downstream station solve.
    value(dt, τ) = abs(sum(zs[b] * cis(-dt * xdisp[b] - τ * xtau[b]) for b in 1:nb))
    for _ in 1:2
        for (step, isdt) in ((0.01, true), (1.0e-12, false))
            d0 = best_d
            t0 = best_t
            am = isdt ? value(d0 - step, t0) : value(d0, t0 - step)
            ap = isdt ? value(d0 + step, t0) : value(d0, t0 + step)
            den = am - 2 * best_a + ap
            den < 0 || continue
            δ = 0.5 * step * (am - ap) / den
            abs(δ) <= step || continue
            if isdt
                best_d = d0 + δ
            else
                best_t = t0 + δ
            end
            best_a = value(best_d, best_t)
        end
    end
    var = sum(ws)
    snr = var > 0 ? sqrt(max(best_a^2 - var, 0.0) / var) : 0.0
    return (tau = best_t, dtec = best_d, amp = best_a, snr = snr)
end

# Narrow delay window for the pass-2 POLISH of a scan the bandpass stage already
# fit (`reuse_bandpass_refine`): the residual on top of the stage's fit is small,
# so a tight window keeps the fine-grid resolution/accuracy while shrinking the
# dominant coarse sweep (its cost scales with window extent). The dTEC half-width
# is the user-tunable `bandpass_polish_dtec` (a too-tight window CLIPS noisy weak-
# scan residuals → the coherence regression a full skip caused; default ±20 TECU
# covers the worst observed). The delay window stays fixed here — well within the
# band-comb ambiguity clamp (`_band_delay_halfwindow`).
const _DTEC_POLISH_TAU = 8.0e-9      # s

# Refine one materialized scan group: band phasors → per-baseline (Δτ, dTEC) →
# two station solves accumulating into the per-scan delay and dispersion columns
# (disjoint per scan, so pass-2 groups can refine concurrently). Returns the
# number of detections the robust station solves excised.

function _dispersion_fit_stationize!(
        θ, z, w, fb, bl_pairs, pols, feeds, ti0, geom,
        delay_plan, disp_plan, ref_ant, opts, snr_min, tau_max, dtec_max, ties = nothing,
    )
    nbl, npol, nlf = size(z)
    Dτ = fill(_INVALID_DETECTION, nbl, npol)
    Dd = fill(_INVALID_DETECTION, nbl, npol)
    for p in 1:npol
        feeds[p][1] == feeds[p][2] || continue
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            rows_f = Float64[]
            rows_z = ComplexF64[]
            rows_w = Float64[]
            for li in 1:nlf
                w[bi, p, li] > 0 || continue
                push!(rows_f, fb[li])
                push!(rows_z, z[bi, p, li])
                push!(rows_w, w[bi, p, li])
            end
            length(rows_f) >= 4 || continue
            fit = _fit_band_dispersion(
                rows_f, rows_z, rows_w, geom.f0;
                tau_max = _band_delay_halfwindow(rows_f, tau_max), dtec_max = dtec_max,
            )
            fit.snr >= snr_min || continue
            Dτ[bi, p] = FringeDetection(fit.tau, 0.0, 0.0, fit.amp, fit.snr, true)
            Dd[bi, p] = FringeDetection(fit.dtec, 0.0, 0.0, fit.amp, fit.snr, true)
        end
    end

    if haskey(ENV, "GUSTAVO_DTEC_DEBUG")
        for p in 1:npol, bi in 1:nbl
            Dτ[bi, p].valid || continue
            println(
                "  dtec-fit bl=", bl_pairs[bi], " p=", p,
                " Δτ=", round(Dτ[bi, p].delay * 1.0e12; digits = 1), "ps",
                " dtec=", round(Dd[bi, p].delay; digits = 3),
                " snr=", round(Dτ[bi, p].snr; digits = 1),
            )
        end
    end
    nrej = 0
    tied = ties !== nothing && any(ties[i] != i for i in eachindex(ties))
    for (D, plan, tie) in ((Dτ, delay_plan, false), (Dd, disp_plan, tied))
        plan === nothing && continue
        any(d -> d.valid, D) || continue
        pairs_s = bl_pairs
        Ds = D
        if tie
            # Solve on representative nodes; a baseline whose endpoints collapse
            # to the same representative (the co-located pair itself) carries no
            # differential-TEC information and is dropped.
            pairs_s = [(ties[a], ties[b]) for (a, b) in bl_pairs]
            Ds = copy(D)
            for bi in eachindex(pairs_s)
                if pairs_s[bi][1] == pairs_s[bi][2]
                    for p in axes(Ds, 2)
                        Ds[bi, p] = _INVALID_DETECTION
                    end
                end
            end
            any(d -> d.valid, Ds) || continue
        end
        _, _, nr, _ = solve_station_systems!(
            θ, (StationScanDetections(Ds, pairs_s, feeds, ti0),), ((plan, :delay),);
            ref_ant = tie ? ties[ref_ant] : ref_ant, opts = opts,
        )
        nrej += nr
        if tie
            # Members inherit the representative's solved value (assignment, not
            # increment — both start equal, so totals stay equal).
            ts = plan.tseg_id[ti0]
            for a in eachindex(ties)
                ties[a] == a && continue
                for f in 1:2
                    offm = plan.off1[a, f, ts, 1]
                    offr = plan.off1[ties[a], f, ts, 1]
                    (offm == 0 || offr == 0) && continue
                    θ[offm] = θ[offr]
                end
            end
        end
    end
    haskey(ENV, "GUSTAVO_DTEC_DEBUG") && println("  dtec-refine rejected: ", nrej)
    return nrej
end

# Half the band-comb delay ambiguity: band phasors sampled on band centers with
# an (approximate) common spacing grid `g` cannot distinguish τ from τ + k/g, so
# the (Δτ, dTEC) fit must search a window with a UNIQUE branch — otherwise
# baselines tie-break the exact degeneracy to different branches, the
# measurements break closure, and the robust station solve excises them instead
# of fixing the delay. Uses the same folded-Euclid grid the MBD search uses.
function _band_delay_halfwindow(fbs::Vector{Float64}, tau_max::Float64)
    length(fbs) < 3 && return tau_max
    fs = sort(fbs)
    steps = diff(fs)
    g = _approx_gcd(steps, 0.05 * minimum(steps))
    g > 0 || return tau_max
    return min(tau_max, 0.499 / g)
end

# ── Per-scan band-group SBD refinement (fourfit's single-band delay) ──────────
#
# The wideband (MBD) delay and dTEC are constrained by CROSS-band structure;
# the WITHIN-band phase slope is nearly orthogonal to both and instrumentally
# real: a station's per-band signal path can move relative to its phase-cal
# tones between scans (VR2505's YJ drifts by ~30 ns in the 3 GHz group), which
# no time-invariant per-channel bandpass can represent. This stage measures the
# residual within-band slope per (baseline, band group) from sub-band CHUNK
# phasors (exact matched filter over one delay about the group's centre),
# stationizes each group's slopes, and accumulates into the per-scan
# `Delay × FrequencyBands` column plus its companion constant (net correction
# 2πτ(ν − νg): zero at the group centre, so the cross-band solution the MBD +
# dTEC terms own is untouched). SNR-gated — quiet stations contribute nothing.

# Accumulate one channel-block's inverse-variance chunk phasors:
# `z[bi, p, chunk_of_chan[c]] += w·|den|²·(V/den)` (parallel hands only).
# Function barrier (V/W type-unstable at the call site).
function _accumulate_leaf_chunks!(z, w, V, W, g, bl_pairs, pols, chunk_of_chan)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        fa == fb || continue
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            for tt in 1:nti, c in 1:nchan
                ww = W[c, tt, bi, p]
                (ww > 0 && isfinite(ww)) || continue
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, tt, bi, p] / den
                isfinite(v) || continue
                wd = ww * abs2(den)
                k = chunk_of_chan[c]
                z[bi, p, k] += wd * v
                w[bi, p, k] += wd
            end
        end
    end
    return nothing
end

# Exact matched filter for ONE delay over chunk phasors about centre `fc`:
# argmax_τ |Σ_k z_k·cis(−2πτ(f_k − fc))|, coarse→fine sweep + parabolic polish.
function _fit_chunk_delay(
        fs::Vector{Float64}, zs::Vector{ComplexF64}, ws::Vector{Float64}, fc::Float64;
        tau_max::Float64,
    )
    value(τ) = abs(sum(zs[k] * cis(-2π * τ * (fs[k] - fc)) for k in eachindex(fs)))
    best_a = -1.0
    best_t = 0.0
    coarse = tau_max / 400
    for τ in (-tau_max):coarse:tau_max
        a = value(τ)
        a > best_a && (best_a = a; best_t = τ)
    end
    fine = coarse / 50
    for τ in (best_t - coarse):fine:(best_t + coarse)
        a = value(τ)
        a > best_a && (best_a = a; best_t = τ)
    end
    am = value(best_t - fine)
    ap = value(best_t + fine)
    den = am - 2 * best_a + ap
    if den < 0
        δ = 0.5 * fine * (am - ap) / den
        if abs(δ) <= fine
            best_t += δ
            best_a = value(best_t)
        end
    end
    var = sum(ws)
    snr = var > 0 ? sqrt(max(best_a^2 - var, 0.0) / var) : 0.0
    return (tau = best_t, amp = best_a, snr = snr)
end

# Shared back half: per-(baseline, group) fits over the accumulated chunk
# phasors — the within-group SLOPE (exact matched filter) plus the slope-
# corrected group phasor's PHASE — then one feed-common station solve of each
# per group. Both are needed: the wideband delay's decomposition against the
# per-group slopes is ambiguous (a common-mode slope shift leaves per-group
# constants of 2πΔτ(f0 − νg) behind that no single per-scan constant can
# absorb), and real instruments carry genuine per-band phase offsets. θ gets
# the per-scan per-group delay and its constant: net correction
# 2πτ(f − νg) + φg, referenced to the group centre. `chunkf`/`chunkgrp` label
# each accumulated chunk with its centre frequency and band-group id.
function _sbd_fit_stationize!(
        θ, z, w, chunkf, chunkgrp, bl_pairs, pols, feeds, ti0, geom, sbd, ref_ant, nant;
        opts::Stationization = Stationization(reject_iters = 0),
        snr_min::Float64 = 8.0, tau_max::Float64 = 6.0e-8,
    )
    nbl, npol, _ = size(z)
    ngrp = length(sbd.bands)
    nrej = 0
    # ── Tier 1: per-group fits + within-group SLOPE guard ───────────────────
    # The per-baseline SNR gate cannot catch a BIASED fit: on a group whose
    # chunk phasors are internally decoherent (real per-channel bandpass
    # structure, not a delay — VGOS band 2 is the worst), the matched filter
    # happily returns a confident wrong slope, and applying it bends every
    # baseline of a previously-better group (the VR2505 full run: +SBD
    # degraded g1/g2 η_f on 1803+784 and 4C39.25 and band-2 on 3C454.3 while
    # genuinely fixing 0607-157 and YJ). So: keep a group's stationized slope
    # only when it RAISES the group's within-band coherence on the very chunk
    # phasors it was fit from. The group CONSTANT is judged separately — it
    # cancels inside the within-group coherence (|Σ z·e^{-iφ}| = |Σ z|), so
    # this tier is structurally blind to it. A failing slope is ZEROED and the
    # group phase refit at τ = 0 (NOT dropped with the group: φg carries the
    # cross-band alignment, and dropping it collapsed 1803+784's cross-band η
    # in the first, group-dropping version of this guard).
    gsols = NamedTuple[]
    for gidx in 1:ngrp
        ks = findall(==(gidx), chunkgrp)
        length(ks) >= 3 || continue
        fc = sum(chunkf[ks]) / length(ks)       # FIXED group centre (matches the θ write)
        rows_τ = _ObsRow[]
        # per accepted baseline row: (a, b, φ at fitted slope, φ at τ = 0, weight)
        φrows = Tuple{Int, Int, Float64, Float64, Float64}[]
        for p in 1:npol
            feeds[p][1] == feeds[p][2] || continue
            for bi in 1:nbl
                a, b = bl_pairs[bi]
                a == b && continue
                fs = Float64[]
                zs = ComplexF64[]
                ws = Float64[]
                for k in ks
                    w[bi, p, k] > 0 || continue
                    push!(fs, chunkf[k])
                    push!(zs, z[bi, p, k])
                    push!(ws, w[bi, p, k])
                end
                length(fs) >= 3 || continue
                fit = _fit_chunk_delay(fs, zs, ws, fc; tau_max = _band_delay_halfwindow(fs, tau_max))
                fit.snr >= snr_min || continue
                φs = angle(sum(zs[k] * cis(-2π * fit.tau * (fs[k] - fc)) for k in eachindex(fs)))
                φ0 = angle(sum(zs))
                # feed-common node (SharedFeeds model): both hands constrain feed 1.
                push!(rows_τ, _ObsRow(a, b, 1, 1, fit.tau, fit.snr^2, 0))
                push!(φrows, (a, b, φs, φ0, fit.snr^2))
            end
        end
        isempty(rows_τ) && continue
        τv, _, covτ, _ = _solve_observable_robust(rows_τ, nant, ref_ant, opts; use_chi = false, rewrap = 0)
        num0 = 0.0
        num1 = 0.0
        for p in 1:npol
            feeds[p][1] == feeds[p][2] || continue
            for bi in 1:nbl
                a, b = bl_pairs[bi]
                a == b && continue
                τab = (covτ[a, 1] ? τv[a, 1] : 0.0) - (covτ[b, 1] ? τv[b, 1] : 0.0)
                acc0 = zero(ComplexF64)
                acc1 = zero(ComplexF64)
                for k in ks
                    w[bi, p, k] > 0 || continue
                    zk = z[bi, p, k]
                    acc0 += zk
                    acc1 += zk * cis(-2π * τab * (chunkf[k] - fc))
                end
                num0 += abs(acc0)
                num1 += abs(acc1)
            end
        end
        slope_ok = num1 > num0
        if !slope_ok
            nrej += 1
            fill!(τv, 0.0)
        end
        rows_φ = [_ObsRow(r[1], r[2], 1, 1, slope_ok ? r[3] : r[4], r[5], 0) for r in φrows]
        φv, _, covφ, _ = _solve_observable_robust(rows_φ, nant, ref_ant, opts; use_chi = false, rewrap = 2)
        push!(gsols, (; gidx, ks, fc, τv, covτ, φv, covφ))
    end
    isempty(gsols) && return nrej
    # ── Tier 2: scan-level JOINT guard on the CROSS-band statistic ──────────
    # The group constants' whole job is aligning the groups' band phasors, so
    # judge them (together with the surviving slopes) on the coherent sum over
    # ALL groups per baseline — the ηx-flavoured statistic — and apply the
    # scan's SBD solution all-or-nothing.
    num0 = 0.0
    num1 = 0.0
    for p in 1:npol
        feeds[p][1] == feeds[p][2] || continue
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            acc0 = zero(ComplexF64)
            acc1 = zero(ComplexF64)
            for gs in gsols
                τab = (gs.covτ[a, 1] ? gs.τv[a, 1] : 0.0) - (gs.covτ[b, 1] ? gs.τv[b, 1] : 0.0)
                φab = (gs.covφ[a, 1] ? gs.φv[a, 1] : 0.0) - (gs.covφ[b, 1] ? gs.φv[b, 1] : 0.0)
                for k in gs.ks
                    w[bi, p, k] > 0 || continue
                    zk = z[bi, p, k]
                    acc0 += zk
                    acc1 += zk * cis(-(2π * τab * (chunkf[k] - gs.fc) + φab))
                end
            end
            num0 += abs(acc0)
            num1 += abs(acc1)
        end
    end
    num1 > num0 || return nrej + length(gsols)
    ts = sbd.dplan.tseg_id[ti0]
    for gs in gsols
        for a in 1:nant
            τa = (gs.covτ[a, 1] && isfinite(gs.τv[a, 1])) ? gs.τv[a, 1] : 0.0
            φa = (gs.covφ[a, 1] && isfinite(gs.φv[a, 1])) ? gs.φv[a, 1] : 0.0
            (τa == 0.0 && φa == 0.0) && continue
            offd = sbd.dplan.off1[a, 1, ts, gs.gidx]
            offd == 0 && continue
            θ[offd] += τa
            offc = sbd.cplan.off1[a, 1, ts, gs.gidx]
            offc == 0 && continue
            θ[offc] += φa - 2π * τa * (gs.fc - geom.f0)
        end
    end
    return nrej
end

# Leaf-group variant (pass 2): each band leaf is split into `nchunk` contiguous
# channel chunks; their phasors feed `_sbd_fit_stationize!`.
