# ── Globally-closing adhoc phasing ───────────────────────────────────────────
#
# After the per-scan fringe solution (delay/rate/constant phase per station,
# feed) there remains a fast, time-variable phase per station — atmospheric
# turbulence — that a per-scan constant cannot track. EHT-HOPS estimates it
# relative to a reference antenna; here we solve it GLOBALLY: per accumulation
# period (AP) we fit station phases from ALL baselines via the same feed-aware
# incidence WLS used by `stationize_scan`, so the solution closes by construction
# and stays well-determined at low SNR even without a dominant anchor (no ALMA).
#
# Per scan: (1) the caller supplies residual baseline visibilities already
# divided by the fringe solution and coherently frequency-averaged to one complex
# number per (baseline, product, AP); (2) per AP, solve the per-(station, feed)
# phase + one source cross-hand phase χ over its connected components (cross-hand
# rows add connectivity and absorb residual field-rotation drift at AP
# resolution, since field rotation is deliberately NOT pre-corrected); (3) unwrap
# each (station, feed) track across APs; (4) smooth (Savitzky–Golay) or penalize
# (dense first-difference — no SparseArrays); (5) detrend per (station, feed):
# remove the per-scan weighted mean and slope so the adhoc track does not alias
# the Stage-B constant phase and rate.

"""
    AdhocPhasing(; mode, window, order, smoothness, snr_floor, phase_rewrap_iters, detrend)

Options for [`solve_adhoc_phasing`](@ref).

- `mode`  : `:smooth` (Savitzky–Golay, default), `:penalized` (dense
  first-difference regularization with strength `smoothness`), or `:none` (raw
  per-AP solves).
- `window`, `order` : Savitzky–Golay window (APs) and polynomial order.
- `smoothness`      : `λ` for `:penalized`.
- `snr_floor`       : per-AP per-baseline coherent-SNR floor; weaker rows drop.
- `phase_rewrap_iters` : re-wrap iterations in each per-AP solve.
- `detrend`         : remove the per-(station, feed) mean + slope per scan.
"""
Base.@kwdef struct AdhocPhasing
    mode::Symbol = :smooth
    window::Int = 11
    order::Int = 2
    smoothness::Float64 = 1.0
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    detrend::Bool = true
end

"""
    AdhocSolution

Per-(station, feed) adhoc phase track for one scan: `phase` is `(nant, 2, nap)`
(rad, `NaN` where unsolved), `chi` the per-AP source cross-hand phase `(nap,)`,
`times` the AP epochs, `covered` the solved cells.
"""
struct AdhocSolution
    phase::Array{Float64, 3}
    chi::Vector{Float64}
    times::Vector{Float64}
    covered::BitArray{3}
end

"""
    solve_adhoc_phasing(rbar, wbar, bl_pairs, pol_products, nant, times; ref_ant, opts) -> AdhocSolution

Solve globally-closing adhoc phases from coherently frequency-averaged residual
baseline visibilities. `rbar[baseline, product, ap]` is `Σ_chan w·V_residual`
(complex) and `wbar[baseline, product, ap]` is `Σ_chan w` for each AP, so the
coherent SNR² is `|rbar|²/wbar`. `times` are the AP epochs (any units; used only
for detrending). `ref_ant` sets the per-AP gauge (its adhoc phase is held at 0).
"""
function solve_adhoc_phasing(
        rbar::AbstractArray{<:Complex, 3}, wbar::AbstractArray{<:Real, 3},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString},
        nant::Integer, times::AbstractVector;
        ref_ant::Integer = 1,
        opts::AdhocPhasing = AdhocPhasing(),
    )
    nbl, npol, nap = size(rbar)
    size(wbar) == size(rbar) || error("rbar and wbar must have the same shape")
    nbl == length(bl_pairs) || error("rbar has $nbl baselines; bl_pairs has $(length(bl_pairs))")
    npol == length(pol_products) || error("rbar has $npol products; pol_products has $(length(pol_products))")
    nap == length(times) || error("rbar has $nap APs; times has $(length(times))")
    feeds = [correlation_feed_pair(p) for p in pol_products]

    phase = fill(NaN, nant, 2, nap)
    chi = fill(NaN, nap)
    covered = falses(nant, 2, nap)
    # Track per-(station,feed) coherent weight for smoothing / detrend.
    track_w = zeros(nant, 2, nap)

    for ap in 1:nap
        rows = _ObsRow[]
        for bi in 1:nbl, p in 1:npol
            a, b = bl_pairs[bi]
            a == b && continue
            r = rbar[bi, p, ap]
            w = wbar[bi, p, ap]
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            snr2 = abs2(r) / w
            snr2 >= opts.snr_floor^2 || continue
            fa, fb = feeds[p]
            push!(rows, _ObsRow(a, b, fa, fb, angle(r), snr2, _chi_sign(fa, fb)))
            track_w[a, fa, ap] += snr2
            track_w[b, fb, ap] += snr2
        end
        ph, c, cov, _ = _solve_observable(rows, nant, ref_ant; use_chi = true, rewrap = opts.phase_rewrap_iters)
        phase[:, :, ap] .= ph
        chi[ap] = c
        covered[:, :, ap] .= cov
    end

    # Restitch the per-AP gauge when the reference antenna drops out (K3). Each
    # per-AP solve pins `ref_ant`; in APs where `ref_ant` has no data the solve
    # falls back to a different anchor node, so that AP's whole solution is offset
    # by an arbitrary (non-2π) constant — which would otherwise inject a spurious
    # common-mode jump into every station's track. Re-reference those APs to the
    # trusted frame from neighbouring ref-present APs via the overlapping stations.
    _restitch_refant_gauge!(phase, covered, track_w, ref_ant)

    # Unwrap each (station, feed) track across APs (per-AP solves share the ref
    # gauge, so a track is continuous up to ±2π steps the unwrap removes).
    for a in 1:nant, f in 1:2
        any(isfinite, @view phase[a, f, :]) || continue
        phase[a, f, :] .= unwrap_phase_track(phase[a, f, :]; weights = track_w[a, f, :])
    end

    # Smooth / penalize.
    if opts.mode === :smooth
        for a in 1:nant, f in 1:2
            any(isfinite, @view phase[a, f, :]) || continue
            phase[a, f, :] .= savitzky_golay_smooth(phase[a, f, :], track_w[a, f, :]; window = opts.window, order = opts.order)
        end
    elseif opts.mode === :penalized
        for a in 1:nant, f in 1:2
            any(isfinite, @view phase[a, f, :]) || continue
            phase[a, f, :] .= _penalized_smooth(phase[a, f, :], track_w[a, f, :], opts.smoothness)
        end
    elseif opts.mode !== :none
        error("AdhocPhasing mode must be :smooth, :penalized or :none; got $(opts.mode)")
    end

    # Detrend per (station, feed): remove mean + slope over the scan so adhoc does
    # not alias the Stage-B constant phase / rate.
    if opts.detrend
        for a in 1:nant, f in 1:2
            _detrend_track!(@view(phase[a, f, :]), times, @view(track_w[a, f, :]))
        end
    end

    return AdhocSolution(phase, chi, collect(float.(times)), covered)
end

# Restitch per-AP gauges so the reference frame is consistent across APs even
# when `ref_ant` drops out (K3). When `ref_ant` is solved in an AP, that AP's
# per-AP solve already pins it (frame = ref_ant phase 0) and we trust it,
# refreshing the running anchor from this AP's solved cells (so real drift
# propagates). When `ref_ant` is ABSENT, the AP's solve anchored on a different
# node, so it carries an arbitrary global offset δ; we estimate δ as the
# weighted circular mean over the cells common to this AP and the anchor, and
# subtract it from every solved cell of the AP. This is a no-op when `ref_ant` is
# present in every AP (so it never perturbs the well-anchored case), and it only
# removes a single global per-AP constant — per-(station, feed) means and slopes
# are still handled later by `_detrend_track!`. Leading APs with no trusted
# anchor yet are left untouched (best effort). A multi-component AP keeps one
# global δ dominated by the largest overlap; per-island offsets remain a
# fundamental gauge freedom (documented in `stationize_scan`).
function _restitch_refant_gauge!(phase, covered, track_w, ref_ant::Integer)
    nant, _, nap = size(phase)
    anchor = fill(NaN, nant, 2)
    have_anchor = false
    for ap in 1:nap
        ref_present = covered[ref_ant, 1, ap] || covered[ref_ant, 2, ap]
        if !ref_present && have_anchor
            # Register PER FEED. With cross hands the two feeds share a component
            # but carry two gauge freedoms (the overall phase pin and the feed-2
            # EVPA pin); when ref_ant drops out both fall back to a different
            # antenna, shifting each feed by its own constant. The per-feed
            # convention (each feed gauged relative to ref_ant's feed) matches the
            # rest of the adhoc solve, so a separate δ per feed restores it.
            for f in 1:2
                num_s = 0.0
                num_c = 0.0
                wsum = 0.0
                for a in 1:nant
                    (covered[a, f, ap] && isfinite(anchor[a, f]) && isfinite(phase[a, f, ap])) || continue
                    d = phase[a, f, ap] - anchor[a, f]
                    wk = (isfinite(track_w[a, f, ap]) && track_w[a, f, ap] > 0) ? track_w[a, f, ap] : 1.0
                    num_s += wk * sin(d)
                    num_c += wk * cos(d)
                    wsum += wk
                end
                if wsum > 0 && (num_s != 0 || num_c != 0)
                    δ = atan(num_s, num_c)
                    for a in 1:nant
                        covered[a, f, ap] && (phase[a, f, ap] -= δ)
                    end
                end
            end
        end
        # Refresh the anchor from this AP's (now registered) solved cells.
        if ref_present || have_anchor
            for a in 1:nant, f in 1:2
                (covered[a, f, ap] && isfinite(phase[a, f, ap])) && (anchor[a, f] = phase[a, f, ap])
            end
            have_anchor = true
        end
    end
    return phase
end

# Dense first-difference penalized smoother (no SparseArrays — the system is
# tridiagonal, solved densely): minimize Σ w_k(φ_k − y_k)² + λ Σ(φ_{k+1}−φ_k)².
# Non-finite samples get zero data weight (interpolated by the penalty).
function _penalized_smooth(y::AbstractVector, w::AbstractVector, λ::Real)
    n = length(y)
    n == 0 && return collect(float.(y))
    M = zeros(Float64, n, n)
    rhs = zeros(Float64, n)
    for k in 1:n
        wk = (isfinite(y[k]) && isfinite(w[k]) && w[k] > 0) ? float(w[k]) : 0.0
        M[k, k] += wk
        rhs[k] += wk * (wk > 0 ? float(y[k]) : 0.0)
    end
    for k in 1:(n - 1)                                  # λ (φ_{k+1} − φ_k)²
        M[k, k] += λ
        M[k + 1, k + 1] += λ
        M[k, k + 1] -= λ
        M[k + 1, k] -= λ
    end
    # Guard against an all-unconstrained component (no data, λ = 0).
    all(iszero, M) && return collect(float.(y))
    return M \ rhs
end

# Remove the weighted mean and linear slope of a track over `times`.
function _detrend_track!(track::AbstractVector, times::AbstractVector, w::AbstractVector)
    idx = [i for i in eachindex(track) if isfinite(track[i])]
    length(idx) >= 1 || return track
    ws = [(isfinite(w[i]) && w[i] > 0) ? float(w[i]) : 1.0 for i in idx]
    t = Float64.(times[idx])
    tbar = sum(ws .* t) / sum(ws)
    tc = t .- tbar
    if length(idx) >= 2 && sum(ws .* tc .^ 2) > 0
        A = hcat(ones(length(idx)), tc)
        coef = weighted_least_squares(A, Float64.(track[idx]), ws)
        for i in idx
            track[i] -= coef[1] + coef[2] * (times[i] - tbar)
        end
    else                                                # single point: remove mean only
        m = sum(ws .* track[idx]) / sum(ws)
        for i in idx
            track[i] -= m
        end
    end
    return track
end
