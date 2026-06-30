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
# (dense first-difference — no SparseArrays); (5) demean per (station, feed):
# remove the per-scan weighted mean so the adhoc track does not alias the Stage-B
# constant phase (the slope/residual rate is kept so adhoc can flatten it).

"""
    AdhocPhasing(; mode, window, order, smoothness, snr_floor, phase_rewrap_iters, detrend)

Options for [`solve_adhoc_phasing`](@ref).

- `mode`  : `:smooth` (Savitzky–Golay, default), `:penalized` (dense
  first-difference regularization with strength `smoothness`), or `:none` (raw
  per-AP solves).
- `window`, `order` : Savitzky–Golay window (APs) and polynomial order. `window`
  may be `:auto` (default) — the EHT-HOPS per-station optimal window (Blackburn
  et al. 2019, Eqs 21–22): an SNR-adaptive integration time that balances thermal
  phase noise against atmospheric drift, from the assumed `coherence_time` and
  `structure_exponent`; see [`_savgol_window_dof`](@ref). Pass an integer to fix it.
- `coherence_time` : assumed atmospheric coherence time `T_coh` (SECONDS) — the time
  for phase to drift 1 rad. Band/weather-dependent (EHT-HOPS: ~18 s at 3.5 mm); set
  it for your observation. Only used when `window = :auto`.
- `structure_exponent` : phase structure-function exponent `α` (5/3 = 3D Kolmogorov,
  2/3 = 2D). Only used when `window = :auto`.
- `smoothness`      : `λ` for `:penalized`.
- `snr_floor`       : per-AP per-baseline coherent-SNR floor; weaker rows drop.
- `phase_rewrap_iters` : re-wrap iterations in each per-AP solve.
- `detrend`         : remove the per-(station, feed) weighted MEAN per scan (breaks
  the constant-phase gauge vs the Stage-B `ConstantTerm`); the slope/rate is kept so
  adhoc can flatten residual fringe rate left by an imperfect per-scan `Rate`.
"""
Base.@kwdef struct AdhocPhasing
    mode::Symbol = :smooth
    window::Union{Int, Symbol} = :auto
    order::Int = 2
    coherence_time::Float64 = 10.0
    structure_exponent::Float64 = 5 / 3
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
coherent SNR² is `|rbar|²/wbar`. `times` are the AP epochs in SECONDS — their
spacing sets `T_AP` for the `:auto` smoothing window (otherwise unused, the
detrend removes only the mean). `ref_ant` sets the per-AP gauge (its adhoc phase
is held at 0).

`shared_feeds` (default `false`) solves ONE feed-common station phase per AP
(both feeds collapsed to a single node, χ retained) instead of an independent
track per feed. The residual atmospheric phase is non-birefringent, so a shared
adhoc both denoises it (all four products constrain one node) and contributes
exactly zero R–L (RL/RR) phase — leaving the instrumental R–L offset to the
global Stage-B feed term. Pass `true` when the model's adhoc component is
`SharedFeeds`.
"""
# Data-driven noise variance of one (baseline, product) coherent track
# `V̄_ap = rbar/wbar`, from the robust scatter of its AP-to-AP differences. The
# source/atmosphere vary slowly AP-to-AP while noise is independent, so successive
# differences isolate the noise. For complex-Gaussian noise, `median(|ΔV̄|²) =
# 2 ln2 · σ²` (Δ of two APs has twice the variance, and the median of an
# exponential is `ln2 ×` its mean), so `σ² = median(|ΔV̄|²) / (2 ln2)`. Returns
# `NaN` when fewer than 4 differences are available (caller falls back).
function _track_noise2(rbar, wbar, bi::Int, p::Int, nap::Int)
    d2 = Float64[]
    prev = ComplexF64(NaN, NaN)
    @inbounds for ap in 1:nap
        w = wbar[bi, p, ap]
        v = w > 0 ? rbar[bi, p, ap] / w : ComplexF64(NaN, NaN)
        (isfinite(v) && isfinite(prev)) && push!(d2, abs2(v - prev))
        prev = v
    end
    length(d2) >= 4 || return NaN
    return median(d2) / (2 * log(2))
end

"""
    _savgol_window_dof(rho2, m_coh, alpha, order) -> Int

EHT-HOPS optimal Savitzky–Golay window in APs (Blackburn et al. 2019, Eqs 21–22).
The effective integration time per degree of freedom `T_dof` balances thermal phase
noise (`∝ 1/(ρ²·T)`) against residual atmospheric drift from the power-law structure
function `D_φ(t) = (t/T_coh)^α`:

    T_dof = [ (1+α)(2+α)·T_coh^α / (2^(−α)·α·(2+α−2^α)·ρ²) ]^(1/(α+1)),

and the order-`d` SG window is `N = max(d+1, 1 + 2⌊(d+1)·T_dof/(2·T_AP)⌋)`. Higher
SNR ⇒ shorter window (track the atmosphere); lower SNR ⇒ longer (average down
thermal noise). `rho2` is the station's per-AP coherent SNR², `m_coh = T_coh/T_AP`
the coherence time in APs, `alpha` the structure exponent. Returns the odd SG window
(≥ `order + 1`). (The paper's round-robin leave-one-out over channels is not
reproduced here.)
"""
function _savgol_window_dof(rho2::Real, m_coh::Real, alpha::Real, order::Integer)
    (isfinite(rho2) && rho2 > 0 && m_coh > 0) || return order + 1
    a = float(alpha)
    coef = (1 + a) * (2 + a) / (2.0^(-a) * a * (2 + a - 2.0^a))
    m_dof = (coef * float(m_coh)^a / float(rho2))^(1 / (a + 1))   # T_dof / T_AP, in APs
    n = max(order + 1, 1 + 2 * floor(Int, (order + 1) * m_dof / 2))
    return iseven(n) ? n + 1 : n
end

function solve_adhoc_phasing(
        rbar::AbstractArray{<:Complex, 3}, wbar::AbstractArray{<:Real, 3},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString},
        nant::Integer, times::AbstractVector;
        ref_ant::Integer = 1,
        opts::AdhocPhasing = AdhocPhasing(),
        shared_feeds::Bool = false,
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

    # Per-(baseline, product) noise of the coherent track, estimated data-driven
    # from its AP-to-AP scatter — see `_track_noise2`. The per-AP coherent SNR² is
    # then |V̄_ap|² / noise², which is SCALE-INVARIANT in the WEIGHT column: the
    # naive `|rbar|²/wbar` is only a true SNR when WEIGHT is calibrated inverse-
    # variance, and on raw correlator output (uncalibrated/uniform weights, common
    # in FITS-IDI) it is mis-scaled by an arbitrary factor, silently dropping every
    # row at any fixed `snr_floor` and killing the whole adhoc stage. For calibrated
    # weights `noise² → 1/wbar`, so this reduces to the old `|rbar|²/wbar` exactly.
    noise2 = [_track_noise2(rbar, wbar, bi, p, nap) for bi in 1:nbl, p in 1:npol]

    for ap in 1:nap
        rows = _ObsRow[]
        for bi in 1:nbl, p in 1:npol
            a, b = bl_pairs[bi]
            a == b && continue
            r = rbar[bi, p, ap]
            w = wbar[bi, p, ap]
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            n2 = noise2[bi, p]
            snr2 = isfinite(n2) && n2 > 0 ? abs2(r / w) / n2 : abs2(r) / w   # fall back if unestimable
            snr2 >= opts.snr_floor^2 || continue
            fa, fb = feeds[p]
            cs = _chi_sign(fa, fb)
            # With `shared_feeds` the residual phase is feed-COMMON (the atmosphere is
            # non-birefringent), so both feeds map to ONE station node (feed 1); the
            # cross-hand χ sign is kept (χ still absorbs the per-AP source cross-hand
            # phase). Collapsing here makes all four products constrain the same
            # node difference φ_a − φ_b (±χ), so the solve has no feed-2 node and thus
            # contributes ZERO R–L phase — R–L is left to the global instrumental term.
            na = shared_feeds ? 1 : fa
            nb = shared_feeds ? 1 : fb
            push!(rows, _ObsRow(a, b, na, nb, angle(r), snr2, cs))
            track_w[a, na, ap] += snr2
            track_w[b, nb, ap] += snr2
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

    # Smooth / penalize. `window = :auto` sets a PER-STATION Savitzky–Golay window
    # from the EHT-HOPS `T_dof` (Eqs 21–22): an SNR-adaptive integration time scaled
    # by the assumed coherence time. `T_AP` is the AP spacing (`times` in SECONDS);
    # the per-AP coherent SNR² is the station's mean `track_w` (Σ baseline SNR²).
    if opts.mode === :smooth
        auto = opts.window === :auto
        m_coh = if auto
            dts = filter(>(0), diff(sort(Float64.(collect(times)))))
            t_ap = isempty(dts) ? 1.0 : median(dts)
            opts.coherence_time / t_ap
        else
            0.0
        end
        for a in 1:nant, f in 1:2
            any(isfinite, @view phase[a, f, :]) || continue
            win = if auto
                tw = [track_w[a, f, ap] for ap in 1:nap if track_w[a, f, ap] > 0]
                rho2 = isempty(tw) ? 0.0 : sum(tw) / length(tw)
                _savgol_window_dof(rho2, m_coh, opts.structure_exponent, opts.order)
            else
                Int(opts.window)
            end
            phase[a, f, :] .= savitzky_golay_smooth(phase[a, f, :], track_w[a, f, :]; window = win, order = opts.order)
        end
    elseif opts.mode === :penalized
        for a in 1:nant, f in 1:2
            any(isfinite, @view phase[a, f, :]) || continue
            phase[a, f, :] .= _penalized_smooth(phase[a, f, :], track_w[a, f, :], opts.smoothness)
        end
    elseif opts.mode !== :none
        error("AdhocPhasing mode must be :smooth, :penalized or :none; got $(opts.mode)")
    end

    # Demean per (station, feed): remove the per-scan mean so adhoc does not alias
    # the Stage-B constant phase. The slope (residual rate) is intentionally kept —
    # see `_detrend_track!`.
    if opts.detrend
        for a in 1:nant, f in 1:2
            _detrend_track!(@view(phase[a, f, :]), @view(track_w[a, f, :]))
        end
    end

    # `shared_feeds`: the solve placed the feed-common track on feed 1 only (feed 2
    # nodes were never touched). Replicate it onto feed 2 so the per-(station, feed)
    # write into θ's SharedFeeds adhoc column is identical for both feeds (and so the
    # returned solution is a valid `(nant, 2, nap)` array). R–L from the adhoc is then
    # exactly 0 by construction.
    if shared_feeds
        phase[:, 2, :] .= @view phase[:, 1, :]
        covered[:, 2, :] .= @view covered[:, 1, :]
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
function _detrend_track!(track::AbstractVector, w::AbstractVector)
    idx = [i for i in eachindex(track) if isfinite(track[i])]
    length(idx) >= 1 || return track
    ws = [(isfinite(w[i]) && w[i] > 0) ? float(w[i]) : 1.0 for i in idx]
    # Remove the weighted MEAN only — NOT the slope. The constant-phase degeneracy
    # between the per-AP adhoc term and the Stage-B `ConstantTerm` is real and worth
    # breaking (zero-mean adhoc ⇒ the constant is owned by Stage-B). The *slope*,
    # however, is the residual fringe RATE left when Stage-B's per-scan `Rate` is
    # imperfect (e.g. within-scan atmospheric drift); removing it would force that
    # error to survive uncorrected. Letting adhoc keep the slope lets it flatten
    # residual rates — the whole point of a per-AP phase track. (`times` is unused
    # the whole point of a per-AP phase track.)
    m = sum(ws .* Float64.(track[idx])) / sum(ws)
    for i in idx
        track[i] -= m
    end
    return track
end
