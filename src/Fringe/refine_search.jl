# ── Per-baseline matched filters for dispersion (dTEC) and SBD refinement ────
#
# The per-baseline measurement half of `DispersionSBDFit`: a joint (Δτ, dTEC)
# grid fit over a baseline's per-spw band phasors, and a single-delay fit over
# a band group's sub-band chunk phasors (fourfit's SBD). Both mirror the
# wideband search's matched-filter principle — maximize coherent sum magnitude
# over a delay-like parameter — at coarser (per-band/per-chunk, not
# per-channel) resolution, since only a handful of bands/chunks are available
# per baseline. `refine_stage.jl`'s `refine_scan_dispersion!`/`refine_scan_sbd!`
# assemble the band/chunk phasors these kernels fit from and station-solve the
# resulting per-baseline measurements.

# ── Per-scan (Δτ, dTEC) band-phasor fit ───────────────────────────────────────
#
# The FFT search + stationization solve a per-scan LINEAR delay; the ionosphere
# adds a dispersive phase K·dTEC·(1/f0 − 1/f) whose linear-in-ν part the delay
# absorbs (biasing it by hundreds of ps on VGOS) and whose curvature survives as
# cross-band structure that no GLOBAL per-channel bandpass can track scan-to-scan.
# This measures both self-consistently from each scan's own residual — fourfit's
# ionospheric search, done as a per-baseline (Δτ, dTEC) grid fit over the scan's
# band phasors. Feed-common (ionosphere is non-birefringent to first order);
# cross hands are skipped like the rate solve.

# Collapse one band leaf to one residual phasor per (baseline, product):
# `z[bi, p] = Σ w·V`, `w[bi, p] = Σ w` over the leaf's channels × APs — the
# inverse-variance mean of the data, already gain-corrected (and reweighted by
# |gain|², matching `apply_calibration`) through the pipeline's transform
# chain before this kernel ever sees it.
function _accumulate_leaf_band_phasor!(z, w, V, W, bl_pairs, pols)
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
                v = V[c, tt, bi, p]
                isfinite(v) || continue
                acc += ww * v
                wsum += ww
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

# Half the band-comb delay ambiguity: band/chunk phasors sampled on centers with
# an (approximate) common spacing grid `g` cannot distinguish τ from τ + k/g, so
# a delay fit must search a window with a UNIQUE branch — otherwise baselines
# tie-break the exact degeneracy to different branches, the measurements break
# closure, and the robust station solve excises them instead of fixing the
# delay. Uses the same folded-Euclid grid the MBD search uses. Shared by the
# dispersion and SBD fits below.
function _band_delay_halfwindow(fbs::Vector{Float64}, tau_max::Float64)
    length(fbs) < 3 && return tau_max
    fs = sort(fbs)
    steps = diff(fs)
    g = _approx_gcd(steps, 0.05 * minimum(steps))
    g > 0 || return tau_max
    return min(tau_max, 0.499 / g)
end

# ── Per-scan band-group SBD fit (fourfit's single-band delay) ────────────────
#
# The wideband (MBD) delay and dTEC are constrained by CROSS-band structure;
# the WITHIN-band phase slope is nearly orthogonal to both and instrumentally
# real: a station's per-band signal path can move relative to its phase-cal
# tones between scans (VR2505's YJ drifts by ~30 ns in the 3 GHz group), which
# no time-invariant per-channel bandpass can represent. This measures the
# residual within-band slope per (baseline, band group) from sub-band CHUNK
# phasors (exact matched filter over one delay about the group's centre).

# Accumulate one channel-block's inverse-variance chunk phasors:
# `z[bi, p, chunk_of_chan[c]] += w·V` (parallel hands only), off data already
# gain-corrected through the pipeline's transform chain.
function _accumulate_leaf_chunks!(z, w, V, W, bl_pairs, pols, chunk_of_chan)
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
                v = V[c, tt, bi, p]
                isfinite(v) || continue
                k = chunk_of_chan[c]
                z[bi, p, k] += ww * v
                w[bi, p, k] += ww
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
