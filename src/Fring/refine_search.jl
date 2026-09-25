# ── Per-baseline matched filters for dispersion (dTEC) and SBD refinement ────
#
# The per-baseline measurement half of `DispersionSBDFit`: a joint (Δτ, dTEC)
# grid fit over a baseline's per-spw band phasors, and a single-delay fit over a
# band group's sub-band chunk phasors (fourfit's SBD). Both maximize coherent
# sum magnitude over a delay-like parameter, as the wideband search does, but at
# per-band/per-chunk resolution, only a handful being available per baseline.

# ── Per-scan (Δτ, dTEC) band-phasor fit ─────────────────────────────────────
#
# The FFT search and stationization solve a per-scan linear delay. The
# ionosphere adds a dispersive phase K·dTEC·(1/f0 − 1/f) whose linear-in-ν part
# the delay absorbs, biasing it by hundreds of ps on VGOS, and whose curvature
# survives as cross-band structure no global per-channel bandpass can track
# scan to scan. This measures both self-consistently from each scan's own
# residual, as a per-baseline (Δτ, dTEC) grid fit over the scan's band phasors —
# fourfit's ionospheric search. Feed-common, the ionosphere being
# non-birefringent to first order; cross hands are skipped as in the rate solve.

# Collapse one band (a Measurement Set) to one residual phasor per (group
# baseline, feed pair): `phasor_sum[row, f] = Σ w·V`, `weight_sum[row, f] = Σ w`
# over the band's channels × APs, parallel hands only. Their ratio is the
# inverse-variance mean of the data, which the pipeline's corrections have
# already gain-corrected and reweighted by |gain|².
function _accumulate_band_phasor!(phasor_sum, weight_sum, V, W, F, win::GeometryWindow, blrow, feedrow)
    UVData.check_layer_axes(V, W, F)
    for k in eachindex(win.feed_order)
        fa, fb = win.feed_order[k]
        fa == fb || continue
        for bi in eachindex(win.stations)
            a, b = win.stations[bi]
            a == b && continue
            Vp, Wp, Fp = _member_planes(V, W, F, bi, win.feeds[k, bi])
            acc = zero(ComplexF64)
            wsum = 0.0
            for t in axes(Vp, 2), c in axes(Vp, 1)
                w, v = Wp[c, t], Vp[c, t]
                _usable(Fp[c, t], w, v) || continue
                acc += w * v
                wsum += w
            end
            phasor_sum[blrow[bi], feedrow[k]] = acc
            weight_sum[blrow[bi], feedrow[k]] = wsum
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
    xdisp = [Calibration.DISPERSION_K * (1.0 / f0 - 1.0 / fbs[b]) for b in eachindex(fbs)]
    xtau = [2π * (fbs[b] - f0) for b in eachindex(fbs)]
    best_a = -1.0
    best_d = 0.0
    best_t = 0.0
    y = Vector{ComplexF64}(undef, nb)
    function sweep(dts, τs)
        for dt in dts
            for b in eachindex(y, zs, xdisp)
                y[b] = zs[b] * cis(-dt * xdisp[b])
            end
            for τ in τs
                acc = zero(ComplexF64)
                for b in eachindex(y, xtau)
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
    value(dt, τ) = abs(sum(zs[b] * cis(-dt * xdisp[b] - τ * xtau[b]) for b in eachindex(zs, xdisp, xtau)))
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
    # Independent trials this SNR competes against: the coarse (dTEC, τ) sweep.
    # The refinement passes re-examine the same lobe, so they add no trials.
    ncells = length(-dtec_max:0.25:dtec_max) * length(-tau_max:2.0e-11:tau_max)
    return (tau = best_t, dtec = best_d, amp = best_a, snr = snr, ncells = float(ncells))
end

# Half the band-comb delay ambiguity: band/chunk phasors sampled on centers with
# an (approximate) common spacing grid `g` cannot distinguish τ from τ + k/g, so
# a delay fit must search a window with a unique branch — otherwise baselines
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
# The wideband (MBD) delay and dTEC are constrained by cross-band structure;
# the within-band phase slope is nearly orthogonal to both and instrumentally
# real: a station's per-band signal path can move by tens of nanoseconds
# relative to its phase-cal tones between scans, which no time-invariant
# per-channel bandpass can represent. This measures the residual within-band
# slope per (baseline, band group) from sub-band chunk phasors, an exact
# matched filter over one delay about the group's centre.

# Accumulate one band's inverse-variance chunk phasors:
# `phasor_sum[row, f, chunk_of_chan[c]] += w·V` and `weight_sum[…] += w`
# (parallel hands only), off data the pipeline's corrections have already
# gain-corrected.
function _accumulate_chunks!(phasor_sum, weight_sum, V, W, F, win::GeometryWindow, blrow, feedrow, chunk_of_chan)
    UVData.check_layer_axes(V, W, F)
    for k in eachindex(win.feed_order)
        fa, fb = win.feed_order[k]
        fa == fb || continue
        for bi in eachindex(win.stations)
            a, b = win.stations[bi]
            a == b && continue
            Vp, Wp, Fp = _member_planes(V, W, F, bi, win.feeds[k, bi])
            row, f = blrow[bi], feedrow[k]
            for t in axes(Vp, 2), c in axes(Vp, 1)
                w, v = Wp[c, t], Vp[c, t]
                _usable(Fp[c, t], w, v) || continue
                j = chunk_of_chan[c]
                phasor_sum[row, f, j] += w * v
                weight_sum[row, f, j] += w
            end
        end
    end
    return nothing
end

# Exact matched filter for one delay over chunk phasors about centre `fc`:
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
    # Independent trials this SNR competes against: the coarse τ sweep. The fine
    # pass re-examines the same lobe, so it adds no trials.
    ncells = length((-tau_max):coarse:tau_max)
    return (tau = best_t, amp = best_a, snr = snr, ncells = float(ncells))
end
