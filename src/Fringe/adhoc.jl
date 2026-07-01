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
# each (station, feed) track across APs; (4) smooth (Savitzky–Golay), penalize
# (dense first-difference — no SparseArrays), or GP-smooth (Ornstein–Uhlenbeck /
# Matérn-1/2 Kalman + RTS — see statespace.jl); (5) demean per (station, feed):
# remove the per-scan weighted mean so the adhoc track does not alias the Stage-B
# constant phase (the slope/residual rate is kept so adhoc can flatten it).

# ── Adhoc smoothers: a pluggable type-based interface ─────────────────────────
#
# After the per-AP global solve (below) each (station, feed) phase track is
# smoothed. WHICH smoother — and how it is parameterized — is a strategy TYPE, one
# per method, mirroring the `AbstractBandpassSmoother` pattern in `pipeline.jl`.
# Adding a method is "define a `<: AbstractAdhocSmoother` struct + one method",
# nothing else. The informal interface a smoother participates in:
#
#   - `apply_adhoc!(sm, phase, chi, track_w, times; ref_ant, nant, ap_rows)` —
#     the single dispatch point; mutates `phase` (and `chi` for the joint solve)
#     in place. Per-track smoothers subtype `PerTrackAdhocSmoother` and instead
#     implement the per-track hook `smooth_track(sm, track, w, times)`; the shared
#     `apply_adhoc!` loops over the (station, feed) tracks for them.
#   - `_adhoc_coherence_time(sm)` — the assumed atmospheric `T_coh` (seconds); used
#     by the per-AP warm-start staleness. Default `10.0`; smoothers with a coherence
#     time override it.
#   - `_wants_ap_rows(sm)` — whether the per-AP SNR-gated observation rows must be
#     cached for the smoother to reuse (only the joint solve). Default `false`.
#   - `_requires_shared_feeds(sm)` — whether the smoother needs `shared_feeds`
#     (only the joint solve, whose state is feed-common). Default `false`.
#
# The shared per-AP-solve options (`snr_floor`, `phase_rewrap_iters`, `detrend`)
# are fields on EVERY smoother, so a smoother value fully specifies the adhoc stage.
abstract type AbstractAdhocSmoother end

# A smoother that acts INDEPENDENTLY on each (station, feed) phase track: it
# implements `smooth_track(sm, track, w, times) -> ŷ` and inherits the shared
# `apply_adhoc!` loop below.
abstract type PerTrackAdhocSmoother <: AbstractAdhocSmoother end

"""
    SavitzkyGolaySmoother(; window, order, coherence_time, structure_exponent,
                          snr_floor, phase_rewrap_iters, detrend)

Savitzky–Golay per-track smoother (the default). `window`/`order` are the SG window
(APs) and polynomial order; `window` may be `:auto` (default) — the EHT-HOPS
per-station optimal window (Blackburn et al. 2019, Eqs 21–22): an SNR-adaptive
integration time balancing thermal phase noise against atmospheric drift, from the
assumed `coherence_time` and `structure_exponent`; see [`_savgol_window_dof`](@ref).
Pass an integer to fix it.

- `coherence_time` : assumed atmospheric coherence time `T_coh` (SECONDS) — the time
  for phase to drift 1 rad. Band/weather-dependent (EHT-HOPS: ~18 s at 3.5 mm); set
  it for your observation. Used by the `:auto` window and the warm-start staleness.
- `structure_exponent` : phase structure-function exponent `α` (5/3 = 3D Kolmogorov,
  2/3 = 2D). Only used when `window = :auto`.

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/`detrend`.
"""
Base.@kwdef struct SavitzkyGolaySmoother <: PerTrackAdhocSmoother
    window::Union{Int, Symbol} = :auto
    order::Int = 2
    coherence_time::Float64 = 10.0
    structure_exponent::Float64 = 5 / 3
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    detrend::Bool = true
end

"""
    PenalizedSmoother(; smoothness, snr_floor, phase_rewrap_iters, detrend)

Dense first-difference (random-walk) penalized per-track smoother: minimizes
`Σ w_k(φ_k − y_k)² + smoothness·Σ(φ_{k+1}−φ_k)²`. No SparseArrays (the system is
tridiagonal, solved densely). Larger `smoothness` ⇒ stiffer track.

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/`detrend`.
"""
Base.@kwdef struct PenalizedSmoother <: PerTrackAdhocSmoother
    smoothness::Float64 = 1.0
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    detrend::Bool = true
end

"""
    OUSmoother(; coherence_time, fit_hypers, snr_floor, phase_rewrap_iters, detrend)

Per-station Ornstein–Uhlenbeck / Matérn-1/2 Gaussian-process per-track smoother via
an exact Kalman filter + RTS smoother (see [`smooth_ou_track`](@ref)), with
`coherence_time` as the τ seed. When `fit_hypers` (default `true`) the per-station
OU `(τ, σ²)` are fit by maximum Kalman marginal likelihood; when `false`,
`τ = coherence_time` and `σ²` is seeded from the track scatter (no optimization).

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/`detrend`.
"""
Base.@kwdef struct OUSmoother <: PerTrackAdhocSmoother
    coherence_time::Float64 = 10.0
    fit_hypers::Bool = true
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    detrend::Bool = true
end

"""
    JointOUSmoother(; coherence_time, fit_hypers, snr_floor, phase_rewrap_iters, detrend)

The paper-faithful JOINT solve — one multivariate OU Kalman over all station phases
observing baseline phase differences directly, closing and denoising together (better
at low SNR than per-track-solve-then-smooth). Seeded and rewrapped from the per-AP
solve; see [`_solve_gp_joint!`](@ref). Requires `shared_feeds` (its state is
feed-common). `coherence_time`/`fit_hypers` seed the per-station OU dynamics as in
[`OUSmoother`](@ref).

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/`detrend`.
"""
Base.@kwdef struct JointOUSmoother <: AbstractAdhocSmoother
    coherence_time::Float64 = 10.0
    fit_hypers::Bool = true
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    detrend::Bool = true
end

"""
    NoSmoothing(; snr_floor, phase_rewrap_iters, detrend)

No smoothing: the raw per-AP global solve only (unwrap + optional detrend still run).

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/`detrend`.
"""
Base.@kwdef struct NoSmoothing <: AbstractAdhocSmoother
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    detrend::Bool = true
end

# ── Smoother interface traits ─────────────────────────────────────────────────
# The assumed atmospheric coherence time (seconds) — used by the per-AP warm-start
# staleness. Default for smoothers that carry no coherence time (matches the old
# `coherence_time` field default).
_adhoc_coherence_time(::AbstractAdhocSmoother) = 10.0
_adhoc_coherence_time(sm::SavitzkyGolaySmoother) = sm.coherence_time
_adhoc_coherence_time(sm::OUSmoother) = sm.coherence_time
_adhoc_coherence_time(sm::JointOUSmoother) = sm.coherence_time

# Whether the per-AP SNR-gated observation rows must be cached for the smoother.
_wants_ap_rows(::AbstractAdhocSmoother) = false
_wants_ap_rows(::JointOUSmoother) = true

# Whether the smoother requires `shared_feeds` (feed-common state).
_requires_shared_feeds(::AbstractAdhocSmoother) = false
_requires_shared_feeds(::JointOUSmoother) = true

# ── apply_adhoc!: the single smoothing dispatch point ─────────────────────────
# Mutates the per-(station, feed) phase track array `phase` (and `chi` for the joint
# solve) in place. `ap_rows` is the per-AP SNR-gated observation-row cache (populated
# only when `_wants_ap_rows`); other smoothers ignore it.

# Per-track smoothers: loop the (station, feed) tracks and apply the per-track hook.
function apply_adhoc!(sm::PerTrackAdhocSmoother, phase, chi, track_w, times; ref_ant, nant, ap_rows)
    for a in 1:nant, f in 1:2
        any(isfinite, @view phase[a, f, :]) || continue
        phase[a, f, :] .= smooth_track(sm, phase[a, f, :], @view(track_w[a, f, :]), times)
    end
    return phase
end

# No smoothing — the per-AP solve stands as is.
apply_adhoc!(::NoSmoothing, phase, chi, track_w, times; ref_ant, nant, ap_rows) = phase

# Joint state-space solve: one multivariate OU Kalman over all station phases
# observing baseline differences directly, seeded/rewrapped from the per-AP solve.
function apply_adhoc!(sm::JointOUSmoother, phase, chi, track_w, times; ref_ant, nant, ap_rows)
    _solve_gp_joint!(phase, chi, track_w, ap_rows, nant, times, ref_ant, sm)
    return phase
end

# ── smooth_track: per-(station, feed) smoothing hook ──────────────────────────
# `track` is one (station, feed) phase track (unwrapped, radians), `w` its per-AP
# coherent weight, `times` the AP epochs (seconds). Returns the smoothed track.

# Savitzky–Golay. `window = :auto` sets a PER-STATION window from the EHT-HOPS
# `T_dof` (Eqs 21–22): an SNR-adaptive integration time scaled by the assumed
# coherence time. `T_AP` is the AP spacing; the per-AP coherent SNR² is the track's
# mean `w` (Σ baseline SNR²).
function smooth_track(sm::SavitzkyGolaySmoother, track, w, times)
    win = if sm.window === :auto
        0 < sm.structure_exponent < 2 || error(
            "SavitzkyGolaySmoother: structure_exponent must be in (0, 2) for window = :auto " *
                "(got $(sm.structure_exponent); 5/3 = 3D Kolmogorov, 2/3 = 2D). Outside this " *
                "range the T_dof denominator (2 + α − 2^α) is non-positive.",
        )
        dts = filter(>(0), diff(sort(Float64.(collect(times)))))
        t_ap = isempty(dts) ? 1.0 : median(dts)
        m_coh = sm.coherence_time / t_ap
        tw = [w[ap] for ap in eachindex(w) if w[ap] > 0]
        rho2 = isempty(tw) ? 0.0 : sum(tw) / length(tw)
        _savgol_window_dof(rho2, m_coh, sm.structure_exponent, sm.order)
    else
        Int(sm.window)
    end
    return savitzky_golay_smooth(track, w; window = win, order = sm.order)
end

# Dense first-difference penalized smoother.
smooth_track(sm::PenalizedSmoother, track, w, times) = _penalized_smooth(track, w, sm.smoothness)

# Ornstein–Uhlenbeck (Matérn-1/2) Gaussian-process smoother: an exact Kalman filter
# + RTS smoother with measurement variance 1/w and the process model set by the OU
# (τ, σ²) — fit per station by maximum marginal likelihood when `fit_hypers`.
# `w == 0` APs are missing (predicted through). Run on the mean-subtracted track (OU
# reverts to 0), then restore the mean; the downstream detrend removes it again anyway.
function smooth_track(sm::OUSmoother, track, w, times)
    τ_lo, τ_hi = _ou_tau_bounds(times)
    m, yc, τ, σ2 = _track_ou_hypers(track, w, times; τ0 = sm.coherence_time, τ_lo = τ_lo, τ_hi = τ_hi, fit = sm.fit_hypers)
    return smooth_ou_track(yc, w, times; τ = τ, σ2 = σ2) .+ m
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
    solve_adhoc_phasing(rbar, wbar, bl_pairs, pol_products, nant, times; ref_ant, smoother) -> AdhocSolution

Solve globally-closing adhoc phases from coherently frequency-averaged residual
baseline visibilities. `rbar[baseline, product, ap]` is `Σ_chan w·V_residual`
(complex) and `wbar[baseline, product, ap]` is `Σ_chan w` for each AP, so the
coherent SNR² is `|rbar|²/wbar`. `times` are the AP epochs in SECONDS — their
spacing sets `T_AP` for the `:auto` smoothing window (otherwise unused, the
detrend removes only the mean). `ref_ant` sets the per-AP gauge (its adhoc phase
is held at 0).

`smoother` is an [`AbstractAdhocSmoother`](@ref) selecting how each per-(station,
feed) track is smoothed after the per-AP solve — [`SavitzkyGolaySmoother`](@ref)
(default), [`PenalizedSmoother`](@ref), [`OUSmoother`](@ref),
[`JointOUSmoother`](@ref), or [`NoSmoothing`](@ref) — and carries the shared per-AP
options (`snr_floor`, `phase_rewrap_iters`, `detrend`).

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
    # T_dof is real only for a physical structure exponent 0 < α < 2 (the denominator
    # `2 + a − 2^a` vanishes at α = 2 and goes negative above). Fall back to the
    # minimal window rather than producing Inf/NaN — `solve_adhoc_phasing` validates
    # α up front, so this only guards direct callers.
    denom = 2 + a - 2.0^a
    (0 < a < 2 && denom > 0) || return order + 1
    coef = (1 + a) * (2 + a) / (2.0^(-a) * a * denom)
    m_dof = (coef * float(m_coh)^a / float(rho2))^(1 / (a + 1))   # T_dof / T_AP, in APs
    isfinite(m_dof) || return order + 1
    n = max(order + 1, 1 + 2 * floor(Int, (order + 1) * m_dof / 2))
    return iseven(n) ? n + 1 : n
end

# Build the WLS observation rows for one AP from the coherent residuals, with the
# same data-driven SNR gate and feed→node collapse the per-AP solve uses. Shared
# by the per-AP solve and the joint (`:gp_joint`) solve so both see identical rows.
function _adhoc_ap_rows(rbar, wbar, ap::Integer, bl_pairs, feeds, noise2, snr_floor2::Real, shared_feeds::Bool)
    rows = _ObsRow[]
    nbl = length(bl_pairs)
    @inbounds for bi in 1:nbl, p in eachindex(feeds)
        a, b = bl_pairs[bi]
        a == b && continue
        r = rbar[bi, p, ap]
        w = wbar[bi, p, ap]
        (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
        n2 = noise2[bi, p]
        snr2 = isfinite(n2) && n2 > 0 ? abs2(r / w) / n2 : abs2(r) / w   # fall back if unestimable
        snr2 >= snr_floor2 || continue
        fa, fb = feeds[p]
        cs = _chi_sign(fa, fb)
        na = shared_feeds ? 1 : fa
        nb = shared_feeds ? 1 : fb
        push!(rows, _ObsRow(a, b, na, nb, angle(r), snr2, cs))
    end
    return rows
end

# Joint (paper-faithful) adhoc solve: one MULTIVARIATE OU Kalman filter + RTS
# smoother over the whole station-phase vector, observing the baseline phase
# DIFFERENCES directly with a per-station temporal OU prior — closing and denoising
# in a single recursive estimator (better at low SNR than per-AP-solve-then-smooth,
# where a single AP is poorly conditioned). Mutates `phase[:, 1, :]` (and `chi`)
# in place; the caller's detrend + shared-feeds replicate run afterwards.
#
# State = `nant` station phases + one augmented, temporally-independent source
# cross-hand phase χ (τ = 0 ⇒ a = 0). The unobservable common mode is pinned near 0
# by the OU prior (the paper needs no reference station); we re-gauge to `ref_ant`
# afterwards for pipeline consistency. Requires `shared_feeds` (the fringe adhoc
# term is SharedFeeds).
function _solve_gp_joint!(
        phase, chi, track_w, ap_rows, nant::Integer, times, ref_ant::Integer, sm::JointOUSmoother,
    )
    nap = length(times)
    tsec = Float64.(collect(times))
    nx = nant + 1                       # station phases + augmented χ
    χidx = nant + 1

    # Seed (ref-gauged, unwrapped) per-station track and χ from the per-AP solve.
    θseed = [phase[i, 1, ap] for i in 1:nant, ap in 1:nap]
    χseed = copy(chi)

    # Per-station centering mean (weighted): the OU prior reverts to 0, so a nonzero-
    # mean track would be biased. Removed again by the downstream detrend (so its
    # exact value is inert to the output — it only de-biases the solve).
    mθ = [_weighted_mean_finite(view(θseed, i, :), view(track_w, i, 1, :)) for i in 1:nant]
    mχ = _weighted_mean_finite(χseed, ones(nap))

    # Per-station OU hypers from the (centered) seed track. The reference track is
    # structurally 0 (per-AP gauge) so it carries no variance info — give it and any
    # degenerate/uncovered station the ensemble-median dynamics, so the JOINT solve
    # treats every station as a real OU process (none pinned during the solve).
    τ_lo, τ_hi = _ou_tau_bounds(tsec)
    τv = fill(float(sm.coherence_time), nant)
    σ2v = fill(1.0e-2, nant)
    welldet = falses(nant)
    for i in 1:nant
        i == ref_ant && continue
        any(track_w[i, 1, ap] > 0 for ap in 1:nap) || continue
        _, _, τv[i], σ2v[i] = _track_ou_hypers(
            view(θseed, i, :), view(track_w, i, 1, :), tsec;
            τ0 = sm.coherence_time, τ_lo = τ_lo, τ_hi = τ_hi, fit = sm.fit_hypers,
        )
        welldet[i] = true
    end
    σ2_med = let g = [σ2v[i] for i in 1:nant if welldet[i]]
        isempty(g) ? 1.0e-2 : median(g)
    end
    τ_med = let g = [τv[i] for i in 1:nant if welldet[i]]
        isempty(g) ? float(sm.coherence_time) : median(g)
    end
    for i in 1:nant
        welldet[i] || (τv[i] = τ_med; σ2v[i] = σ2_med)
    end
    τall = vcat(τv, 0.0)                 # χ dim: diffuse, temporally independent
    σ2all = vcat(σ2v, (2π)^2)

    # Fixed per-AP observation geometry: dense H (+1 at a, −1 at b, cs at χ) plus the
    # raw wrapped phase and measurement variance (1/snr²) per row. The row's station
    # indices and cross-hand sign live entirely in H, so the rewrap step reads the
    # model and the centering back out via H·x (no separate a/b/χ index tables).
    Hs = Vector{Matrix{Float64}}(undef, nap)
    raws = Vector{Vector{Float64}}(undef, nap)
    rs = Vector{Vector{Float64}}(undef, nap)
    for ap in 1:nap
        rows = ap_rows[ap]
        m = length(rows)
        H = zeros(m, nx)
        raw = Vector{Float64}(undef, m)
        r_ = Vector{Float64}(undef, m)
        for (j, row) in enumerate(rows)
            H[j, row.a] += 1.0
            H[j, row.b] -= 1.0
            H[j, χidx] += row.chisign
            raw[j] = row.val
            r_[j] = 1.0 / row.w
        end
        Hs[ap] = H
        raws[ap] = raw
        rs[ap] = r_
    end

    # Current full (uncentered) estimate, seeded from the per-AP solve.
    θf = [isfinite(θseed[i, ap]) ? θseed[i, ap] : mθ[i] for i in 1:nant, ap in 1:nap]
    χf = [isfinite(χseed[ap]) ? χseed[ap] : mχ for ap in 1:nap]
    mfull = vcat(mθ, mχ)                 # nx-vector of centering means (H·mfull per row)

    # Iterated Kalman: rewrap each raw observation toward the current joint model,
    # then re-run the forward/backward smoother (relinearizing the ±2π branch).
    for _ in 1:max(sm.phase_rewrap_iters, 1)
        ys = Vector{Vector{Float64}}(undef, nap)
        for ap in 1:nap
            H = Hs[ap]
            raw = raws[ap]
            xfull = vcat(view(θf, :, ap), χf[ap])         # [station phases; χ] for this AP
            yv = Vector{Float64}(undef, length(raw))
            for j in eachindex(raw)
                hrow = view(H, j, :)
                model = dot(hrow, xfull)                  # φ_a − φ_b + cs·χ
                yunw = raw[j] + 2π * round((model - raw[j]) / (2π))
                yv[j] = yunw - dot(hrow, mfull)           # centered observation
            end
            ys[ap] = yv
        end
        xf, Pf, xp, Pp, avecs, _ = kalman_ou_mv_filter(Hs, ys, rs, tsec; τ = τall, σ2 = σ2all)
        xs, _ = rts_smooth_mv(xf, Pf, xp, Pp, avecs)
        for ap in 1:nap
            for i in 1:nant
                θf[i, ap] = xs[ap][i] + mθ[i]
            end
            χf[ap] = xs[ap][χidx] + mχ
        end
    end

    # Re-gauge to the reference (its phase → 0 per AP), matching the per-AP path.
    for ap in 1:nap
        θref = θf[ref_ant, ap]
        isfinite(θref) || continue
        for i in 1:nant
            θf[i, ap] -= θref
        end
    end

    # Write back the joint track only for stations covered somewhere in the scan
    # (internal gaps stay interpolated by the filter, matching :gp); leave never-seen
    # stations NaN so they are not fabricated into θ. Feed 2 stays NaN → the caller's
    # shared-feeds replicate copies feed 1 onto it.
    for i in 1:nant
        has = any(track_w[i, 1, ap] > 0 for ap in 1:nap)
        for ap in 1:nap
            phase[i, 1, ap] = has ? θf[i, ap] : NaN
        end
    end
    chi .= χf
    return phase
end

function solve_adhoc_phasing(
        rbar::AbstractArray{<:Complex, 3}, wbar::AbstractArray{<:Real, 3},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString},
        nant::Integer, times::AbstractVector;
        ref_ant::Integer = 1,
        smoother::AbstractAdhocSmoother = SavitzkyGolaySmoother(),
        shared_feeds::Bool = false,
    )
    nbl, npol, nap = size(rbar)
    size(wbar) == size(rbar) || error("rbar and wbar must have the same shape")
    nbl == length(bl_pairs) || error("rbar has $nbl baselines; bl_pairs has $(length(bl_pairs))")
    npol == length(pol_products) || error("rbar has $npol products; pol_products has $(length(pol_products))")
    nap == length(times) || error("rbar has $nap APs; times has $(length(times))")
    _requires_shared_feeds(smoother) && !shared_feeds && error(
        "$(typeof(smoother)) requires shared_feeds (the fringe adhoc term is " *
            "SharedFeeds); use OUSmoother for an independent per-feed track.",
    )
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

    # Carry solved node phases forward as a temporal warm-start for the next AP's
    # 2π-branch selection (continuity ⇒ no per-AP branch flips on weakly-constrained
    # stations). The seed is a REFERENCE-GAUGED snapshot: it is refreshed only from
    # APs where `ref_ant` has data (so every stored cell is in the ref=0 gauge and
    # the snapshot is mutually consistent) and applied only when the CURRENT AP also
    # has `ref_ant` (so the per-AP spanning-tree seed it partially overrides is ref=0
    # too). Skipping the seed on ref-dropout APs avoids mixing a ref=0 seed with a
    # differently-anchored tree seed, which would mis-pick the 2π branch on a
    # seeded↔tree boundary edge — the arbitrary non-2π offset such APs carry (K3).
    prev_phase = fill(NaN, nant, 2)   # NaN for cells no ref-present AP has covered yet
    prev_age = zeros(Int, nant, 2)    # APs since a cell was last refreshed (staleness)
    prev_chi = NaN
    # Trust a seed only while the atmosphere cannot yet have drifted past ±π (beyond
    # which the old branch is no longer a safe guide): ~π of drift takes a few
    # coherence times. `times` is in seconds; `T_AP` is its median spacing.
    dts0 = filter(>(0), diff(sort(Float64.(collect(times)))))
    t_ap0 = isempty(dts0) ? 1.0 : median(dts0)
    max_stale = max(10, ceil(Int, 3 * _adhoc_coherence_time(smoother) / t_ap0))
    # Cache each AP's SNR-gated rows when the joint solve will reuse them, so
    # `_solve_gp_joint!` need not rebuild the identical row set a second time.
    save_rows = _wants_ap_rows(smoother)
    ap_rows = save_rows ? Vector{Vector{_ObsRow}}(undef, nap) : Vector{Vector{_ObsRow}}()
    for ap in 1:nap
        # With `shared_feeds` the residual phase is feed-COMMON (the atmosphere is
        # non-birefringent), so both feeds map to ONE station node (feed 1); the
        # cross-hand χ sign is kept (χ still absorbs the per-AP source cross-hand
        # phase). Collapsing makes all four products constrain the same node
        # difference φ_a − φ_b (±χ), so the solve has no feed-2 node and thus
        # contributes ZERO R–L phase — R–L is left to the global instrumental term.
        rows = _adhoc_ap_rows(rbar, wbar, ap, bl_pairs, feeds, noise2, smoother.snr_floor^2, shared_feeds)
        save_rows && (ap_rows[ap] = rows)
        for row in rows
            track_w[row.a, row.fa, ap] += row.w
            track_w[row.b, row.fb, ap] += row.w
        end
        # `ref_ant` has data this AP iff some observation touches it (⇒ the solve is
        # anchored at ref=0). Seed only then, and only with fresh, ref-gauged cells.
        ref_here = any(r -> r.a == ref_ant || r.b == ref_ant, rows)
        seed = nothing
        if ref_here
            seed = fill(NaN, nant, 2)
            for a in 1:nant, f in 1:2
                (isfinite(prev_phase[a, f]) && prev_age[a, f] <= max_stale) &&
                    (seed[a, f] = prev_phase[a, f])
            end
        end
        ph, c, cov, _ = _solve_observable(
            rows, nant, ref_ant; use_chi = true, rewrap = smoother.phase_rewrap_iters,
            seed_phase = seed, seed_chi = ref_here ? prev_chi : NaN,
        )
        phase[:, :, ap] .= ph
        chi[ap] = c
        covered[:, :, ap] .= cov
        # Refresh the ref-gauged snapshot ONLY from ref-present APs (keep the last
        # known value for a station absent this AP, so a brief dropout does not reset
        # the branch); age every cell and zero the ones refreshed here.
        prev_age .+= 1
        if ref_here
            for a in 1:nant, f in 1:2
                if cov[a, f] && isfinite(ph[a, f])
                    prev_phase[a, f] = ph[a, f]
                    prev_age[a, f] = 0
                end
            end
            isfinite(c) && (prev_chi = c)
        end
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

    # Smooth: dispatch on the smoother type. Per-track smoothers loop the (station,
    # feed) tracks (`window = :auto` sets a PER-STATION Savitzky–Golay window from the
    # EHT-HOPS `T_dof`, Eqs 21–22); the joint solve runs one multivariate OU Kalman
    # over all station phases; `NoSmoothing` is a no-op. See `apply_adhoc!`.
    apply_adhoc!(smoother, phase, chi, track_w, times; ref_ant = ref_ant, nant = nant, ap_rows = ap_rows)

    # Demean per (station, feed): remove the per-scan mean so adhoc does not alias
    # the Stage-B constant phase. The slope (residual rate) is intentionally kept —
    # see `_detrend_track!`.
    if smoother.detrend
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
