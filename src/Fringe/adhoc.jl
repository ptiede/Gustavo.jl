# ── Globally-closing adhoc phasing ───────────────────────────────────────────
#
# After the per-scan fringe solution (delay/rate/constant phase per station,
# feed) there remains a fast, time-variable phase per station — atmospheric
# turbulence — that a per-scan constant cannot track. EHT-HOPS estimates it
# relative to a reference antenna; here we solve it GLOBALLY: per accumulation
# period (AP) we fit station phases from ALL baselines via the same feed-aware
# incidence WLS used by the station solve, so the solution closes by construction
# and stays well-determined at low SNR even without a dominant anchor (no ALMA).
#
# The observation model for baseline (a, b), correlation product p with feeds
# (fa, fb), at accumulation period t is
#
#     y_{ab,p}(t) = φ_{na}(t) − φ_{nb}(t) + x_{ab,p} + ε,
#
# where `n· = _feed_node(tying, f·)` maps a feed onto its parameter node (so the
# component's feed tying sets whether a station carries one phase track or two),
# and `x_{ab,p}` is the observed source visibility phase — held constant over the
# scan. Making the source term free PER (baseline, product) is what keeps the
# model independent of the polarization basis: the source EVPA, the D-terms, and
# the source's own CLOSURE PHASE all land in `x` rather than biasing the station
# tracks. One cross-hand phase shared by every baseline would be the rank-one
# restriction `x_{ab,p} = ±c`, which cannot represent closure phase at all.
#
# `x` is degenerate with a per-station constant (`φ_a → φ_a + c_a`, `x_{ab,p} →
# x_{ab,p} − (c_a − c_b)`); the per-scan demean in step (5) fixes that gauge. A
# per-station SLOPE is not degenerate with a constant `x`, so the residual-rate
# information the demean deliberately keeps survives untouched.
#
# Per scan: (1) the caller supplies residual baseline visibilities already
# divided by the fringe solution and coherently frequency-averaged to one complex
# number per (baseline, product, AP); (2) alternate — solve the per-AP node
# phases over the graph's connected components with the current `x` removed, then
# re-estimate each `x_{ab,p}` as the weighted circular mean of its residual over
# APs; (3) unwrap each track across APs; (4) smooth (Savitzky–Golay), penalize
# (dense first-difference — no SparseArrays), or GP-smooth (Ornstein–Uhlenbeck /
# Matérn-1/2 Kalman + RTS — see statespace.jl); (5) demean per track: remove the
# per-scan weighted mean so the adhoc track does not alias the Stage-B constant
# phase (the slope/residual rate is kept so adhoc can flatten it).
#
# Steps (1)–(3) are identical for every smoother, so each smoother sees the same
# source-corrected rows and the same seed track; a smoother only ever decides
# step (4).

# ── Adhoc smoothers: a pluggable type-based interface ─────────────────────────
#
# After the per-AP global solve (below) each (station, feed) phase track is
# smoothed. WHICH smoother — and how it is parameterized — is a strategy TYPE, one
# per method (unlike the amp-bandpass smoothers in `bandpass_stage.jl`, which are
# `WLSEstimator`-based callables: most adhoc smoothers are NOT WLS problems —
# `OUSmoother`/`JointOUSmoother` are Kalman/RTS filters, `NoSmoothing` is a no-op —
# so a shared struct-dispatch interface is the right fit here instead).
# Adding a method is "define a `<: AbstractAdhocSmoother` struct + one method",
# nothing else. The informal interface a smoother participates in:
#
#   - `apply_adhoc!(sm, phase, track_w, times; anchor, nant, ap_rows)` — the
#     single dispatch point; mutates `phase` in place. `ap_rows` holds the
#     SNR-gated observation rows per AP, source-corrected, identical for every
#     smoother. Per-track smoothers subtype `PerTrackAdhocSmoother` and instead
#     implement the per-track hook `smooth_track(sm, track, w, times)`; the shared
#     `apply_adhoc!` loops over the (station, node) tracks for them.
#   - `_adhoc_coherence_time(sm)` — the assumed atmospheric `T_coh` (seconds); used
#     by the per-AP warm-start staleness. Default `10.0`; smoothers with a coherence
#     time override it.
#   - `_requires_single_node(sm)` — whether the smoother needs one phase node per
#     station (only the joint solve, whose Kalman state is one dimension per
#     station). Default `false`.
#
# The shared solve options (`snr_floor`, `phase_rewrap_iters`, `source_iters`,
# `source_tol`, `detrend`, `complex_iters`) are fields on EVERY smoother, so a
# smoother value fully specifies the adhoc stage. `snr_floor` gates the SEED
# pass only — the phase-extraction solve whose job is the global 2π branch;
# `complex_iters` Gauss–Newton passes then re-fit the tracks against the
# complex residuals themselves, every AP entering ungated at its exact
# first-order information (`complex_iters = 0` keeps the seed as the answer).
abstract type AbstractAdhocSmoother end

# A smoother that acts INDEPENDENTLY on each (station, feed) phase track: it
# implements `smooth_track(sm, track, w, times) -> ŷ` and inherits the shared
# `apply_adhoc!` loop below.
abstract type PerTrackAdhocSmoother <: AbstractAdhocSmoother end

"""
    SavitzkyGolaySmoother(; window, order, coherence_time, structure_exponent,
                          snr_floor, phase_rewrap_iters, source_iters,
                          source_tol, detrend)

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

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/
`source_iters`/`source_tol`/`detrend`/`complex_iters`.
"""
Base.@kwdef struct SavitzkyGolaySmoother <: PerTrackAdhocSmoother
    window::Union{Int, Symbol} = :auto
    order::Int = 2
    coherence_time::Float64 = 10.0
    structure_exponent::Float64 = 5 / 3
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    source_iters::Int = 10
    source_tol::Float64 = 1.0e-6
    detrend::Bool = true
    complex_iters::Int = 2
end

"""
    PenalizedSmoother(; smoothness, snr_floor, phase_rewrap_iters, source_iters,
                      source_tol, detrend)

Dense first-difference (random-walk) penalized per-track smoother: minimizes
`Σ w_k(φ_k − y_k)² + smoothness·Σ(φ_{k+1}−φ_k)²`. No SparseArrays (the system is
tridiagonal, solved densely). Larger `smoothness` ⇒ stiffer track.

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/
`source_iters`/`source_tol`/`detrend`/`complex_iters`.
"""
Base.@kwdef struct PenalizedSmoother <: PerTrackAdhocSmoother
    smoothness::Float64 = 1.0
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    source_iters::Int = 10
    source_tol::Float64 = 1.0e-6
    detrend::Bool = true
    complex_iters::Int = 2
end

"""
    OUSmoother(; coherence_time, fit_hypers, snr_floor, phase_rewrap_iters,
               source_iters, source_tol, detrend)

Per-station Ornstein–Uhlenbeck / Matérn-1/2 Gaussian-process per-track smoother via
an exact Kalman filter + RTS smoother (see [`smooth_ou_track`](@ref)), with
`coherence_time` as the τ seed. When `fit_hypers` (default `true`) the per-station
OU `(τ, σ²)` are fit by maximum Kalman marginal likelihood; when `false`,
`τ = coherence_time` and `σ²` is seeded from the track scatter (no optimization).

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/
`source_iters`/`source_tol`/`detrend`/`complex_iters`.
"""
Base.@kwdef struct OUSmoother <: PerTrackAdhocSmoother
    coherence_time::Float64 = 10.0
    fit_hypers::Bool = true
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    source_iters::Int = 10
    source_tol::Float64 = 1.0e-6
    detrend::Bool = true
    complex_iters::Int = 2
end

"""
    JointOUSmoother(; coherence_time, fit_hypers, snr_floor, phase_rewrap_iters,
                    source_iters, source_tol, detrend)

The paper-faithful JOINT solve — one multivariate OU Kalman over all station phases
observing baseline phase differences directly, closing and denoising together (better
at low SNR than per-track-solve-then-smooth). Seeded and rewrapped from the per-AP
solve; see [`_solve_gp_joint!`](@ref). Requires one phase node per station (e.g. a
`SharedFeeds` adhoc component), since its state carries one dimension per station.
`coherence_time`/`fit_hypers` seed the per-station OU dynamics as in
[`OUSmoother`](@ref).

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/
`source_iters`/`source_tol`/`detrend`/`complex_iters`.
"""
Base.@kwdef struct JointOUSmoother <: AbstractAdhocSmoother
    coherence_time::Float64 = 10.0
    fit_hypers::Bool = true
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    source_iters::Int = 10
    source_tol::Float64 = 1.0e-6
    detrend::Bool = true
    complex_iters::Int = 2
end

"""
    NoSmoothing(; snr_floor, phase_rewrap_iters, source_iters, source_tol, detrend)

No smoothing: the raw per-AP global solve only (unwrap + optional detrend still run).

See `AbstractAdhocSmoother` for the shared `snr_floor`/`phase_rewrap_iters`/
`source_iters`/`source_tol`/`detrend`/`complex_iters`.
"""
Base.@kwdef struct NoSmoothing <: AbstractAdhocSmoother
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    source_iters::Int = 10
    source_tol::Float64 = 1.0e-6
    detrend::Bool = true
    complex_iters::Int = 2
end

# ── Smoother interface traits ─────────────────────────────────────────────────
# The assumed atmospheric coherence time (seconds) — used by the per-AP warm-start
# staleness. Default for smoothers that carry no coherence time (matches the old
# `coherence_time` field default).
_adhoc_coherence_time(::AbstractAdhocSmoother) = 10.0
_adhoc_coherence_time(sm::SavitzkyGolaySmoother) = sm.coherence_time
_adhoc_coherence_time(sm::OUSmoother) = sm.coherence_time
_adhoc_coherence_time(sm::JointOUSmoother) = sm.coherence_time

# Whether the smoother needs ONE phase node per station. The joint solve's Kalman
# state is one dimension per station, so it cannot represent two independent feed
# tracks.
_requires_single_node(::AbstractAdhocSmoother) = false
_requires_single_node(::JointOUSmoother) = true

# ── apply_adhoc!: the single smoothing dispatch point ─────────────────────────
# Mutates the per-(station, node) phase track array `phase` in place. `ap_rows` is
# the per-AP SNR-gated observation rows with the source term already removed, so
# every smoother sees the same observations; per-track smoothers ignore it.

# Per-track smoothers: loop the (station, node) tracks and apply the per-track hook.
function apply_adhoc!(sm::PerTrackAdhocSmoother, phase, track_w, times; anchor, nant, ap_rows)
    for a in 1:nant, f in 1:2
        any(isfinite, @view phase[a, f, :]) || continue
        phase[a, f, :] .= smooth_track(sm, phase[a, f, :], @view(track_w[a, f, :]), times)
    end
    return phase
end

# No smoothing — the per-AP solve stands as is.
apply_adhoc!(::NoSmoothing, phase, track_w, times; anchor, nant, ap_rows) = phase

# Joint state-space solve: one multivariate OU Kalman over all station phases
# observing baseline differences directly, seeded/rewrapped from the per-AP solve.
function apply_adhoc!(sm::JointOUSmoother, phase, track_w, times; anchor, nant, ap_rows)
    _solve_gp_joint!(phase, track_w, ap_rows, nant, times, anchor, sm)
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
        dts = filter(>(0), diff(sort(times)))
        t_ap = isempty(dts) ? 1.0 : float(median(dts))
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
    solve_adhoc_phasing(rbar, wbar, bl_pairs, pol_products, nant, times;
                        gauge, smoother, tying) -> DimStack

Returns a `DimStack` whose `:phase` layer (`Ant × Feed × Ti`, `Ti` carrying the AP
epochs) is the per-(station, feed) adhoc phase in radians, `NaN` where unsolved,
`:covered` marks the solved cells, and `:source` (`Baseline × Pol`) is the fitted
per-(baseline, product) source visibility phase, `NaN` where unidentifiable.

Solve globally-closing adhoc phases from coherently frequency-averaged residual
baseline visibilities under the model

    y[baseline, product, ap] = φ_na(ap) − φ_nb(ap) + x[baseline, product],

with the station phases `φ` on parameter NODES and one free source term `x` per
(baseline, product), constant over the scan. Because the source term is free per
baseline it carries the source's EVPA, D-terms and CLOSURE PHASE, so the station
tracks are unbiased by source structure and the solve needs no knowledge of the
polarization basis.

`rbar[baseline, product, ap]` is `Σ_chan w·V_residual` (complex) and
`wbar[baseline, product, ap]` is `Σ_chan w` for each AP, so the coherent SNR² is
`|rbar|²/wbar`. `times` are the AP epochs in SECONDS — their spacing sets `T_AP`
for the `:auto` smoothing window (otherwise unused, the detrend removes only the
mean). `gauge` sets the per-AP convention; a `PinAntenna` holds its reference's
adhoc phase at 0, a `ZeroSumPhase` centers each AP on zero mean.

`tying` is the adhoc component's [`AbstractFeedTying`](@ref); `_feed_node` maps
each feed onto the node it constrains. `PerFeed()` (the default, matching
[`GainComponent`](@ref)) solves an independent track per feed; `SharedFeeds()`
solves ONE feed-common station phase that all four correlation products constrain
and that therefore contributes exactly zero inter-feed phase. `ReferenceRelative` is rejected: its partner feed reads two
parameter blocks, which a single-node-per-row solve cannot represent.

Under `PerFeed`, cross-hand rows join the two feed blocks into one connected
component carrying a single gauge freedom, pinned at the reference's feed-1 node.
Every other node — the reference's own feed 2 included — is measured against it, so
the reference's inter-feed phase stays in the solution instead of being pinned away.

`smoother` is an [`AbstractAdhocSmoother`](@ref) selecting how each per-(station,
node) track is smoothed after the solve — [`SavitzkyGolaySmoother`](@ref)
(default), [`PenalizedSmoother`](@ref), [`OUSmoother`](@ref),
[`JointOUSmoother`](@ref), or [`NoSmoothing`](@ref) — and carries the shared solve
options (`snr_floor`, `phase_rewrap_iters`, `source_iters`, `source_tol`,
`detrend`). The `(φ, x)` blocks are fit by alternating minimization, stopping once
no source term moves by more than `source_tol` radians or after `source_iters`
passes; `source_iters = 1` fixes `x = 0`, reducing the model to a pure
station-difference solve.
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

# Circular (complex-phasor) solve of one AP's rows — the classic adhoc-phasing
# iteration z_a ← Σ_b w·e^{iφ_ab}·z_b, gauged at the anchor. The linear
# phases-as-values WLS has 2π-branch local minima when rows sit near ±π: a
# cold-started AP can converge ~150° off a strong row, and the warm-start chain
# then LOCKS that branch and relaxes toward truth across the whole scan — a
# fake smooth ±π-scale arc in the station track (VR2505 WN band 2: +230° over
# a 59-s scan whose rows close to ±20°). The phasor iteration is circular, so
# it has no branch structure; it is used only to SEED the linear solve at APs
# with no usable warm-start snapshot. Every row is a pure node difference (its
# source term is already removed), so all of them drive the iteration.
function _circular_ap_seed(rows::Vector{_ObsRow}, nant::Integer, anchor::Integer)
    isempty(rows) && return nothing
    z = ones(ComplexF64, nant, 2)
    present = falses(nant, 2)
    for r in rows
        present[r.a, r.na] = true
        present[r.b, r.nb] = true
    end
    any(present) || return nothing
    for _ in 1:50
        acc = zeros(ComplexF64, nant, 2)
        for r in rows
            R = cis(r.val)
            acc[r.a, r.na] += r.w * R * z[r.b, r.nb]
            acc[r.b, r.nb] += r.w * conj(R) * z[r.a, r.na]
        end
        for i in eachindex(z)
            present[i] || continue
            a = abs(acc[i])
            a > 0 && (z[i] = acc[i] / a)
        end
    end
    # One common gauge for the whole seed (anchor when present). A fully
    # consistent seed is safe on anchor-dropout APs too: the rewrap uses
    # prediction DIFFERENCES, so a common gauge offset cancels (unlike the
    # mixed-gauge partial snapshots the K3 restitch logic guards against).
    g = present[anchor, 1] ? conj(z[anchor, 1]) / abs(z[anchor, 1]) : one(ComplexF64)
    ph = fill(NaN, nant, 2)
    for f in 1:2, a in 1:nant
        present[a, f] && (ph[a, f] = angle(z[a, f] * g))
    end
    return ph
end

# Build the WLS observation rows for one AP from the coherent residuals, with the
# data-driven SNR gate and the component's feed→node map. Every correlation
# product contributes: a cross-hand row's extra phase is carried by that
# (baseline, product)'s own free source term `src`, so no product needs the
# polarization basis to be known.
function _adhoc_ap_rows(rbar, wbar, ap::Integer, bl_pairs, feeds, noise2, snr_floor2::Real, tying)
    rows = _ObsRow[]
    nbl = length(bl_pairs)
    @inbounds for bi in 1:nbl, p in eachindex(feeds)
        a, b = bl_pairs[bi]
        a == b && continue
        fa, fb = feeds[p]
        na = _feed_node(tying, fa)
        nb = _feed_node(tying, fb)
        # A feed the component does not parameterize (node 0, e.g. `SingleFeed`)
        # has no column for this row to constrain.
        (na == 0 || nb == 0) && continue
        r = rbar[bi, p, ap]
        w = wbar[bi, p, ap]
        (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
        n2 = noise2[bi, p]
        snr2 = isfinite(n2) && n2 > 0 ? abs2(r / w) / n2 : abs2(r) / w   # fall back if unestimable
        snr2 >= snr_floor2 || continue
        push!(rows, _ObsRow(a, b, na, nb, angle(r), snr2, (p - 1) * nbl + bi))
    end
    return rows
end

# Weighted circular mean of each (baseline, product) source term from its
# residual `y − (φ_na − φ_nb)` over the APs where the row survived the gate and
# both nodes solved. `raw_rows` must carry the UNCORRECTED `val`, since `x` is
# defined relative to the observation itself.
function _update_source_terms!(x, raw_rows, phase, nap::Integer)
    acc = zeros(ComplexF64, length(x))
    for ap in 1:nap
        for row in raw_rows[ap]
            row.src == 0 && continue
            pa = phase[row.a, row.na, ap]
            pb = phase[row.b, row.nb, ap]
            (isfinite(pa) && isfinite(pb)) || continue
            acc[row.src] += row.w * cis(row.val - (pa - pb))
        end
    end
    moved = zero(eltype(x))
    for i in eachindex(x, acc)
        abs(acc[i]) > 0 || continue
        xi = angle(acc[i])
        moved = max(moved, abs(rem2pi(xi - x[i], RoundNearest)))
        x[i] = xi
    end
    return moved
end

# The rows of one AP with their source term removed, ready for the node solve.
# Rows whose source term is unidentifiable (`keep[src]` false) are dropped: a
# (baseline, product) seen at a single AP is absorbed EXACTLY by its own source
# term, so it constrains no node phase and only inflates the apparent coverage.
function _source_corrected_rows(rows::Vector{_ObsRow}, x, keep)
    out = _ObsRow[]
    sizehint!(out, length(rows))
    for r in rows
        (r.src == 0 || keep[r.src]) || continue
        v = r.src == 0 ? r.val : r.val - x[r.src]
        push!(out, _ObsRow(r.a, r.b, r.na, r.nb, v, r.w, r.src))
    end
    return out
end

# Joint (paper-faithful) adhoc solve: one MULTIVARIATE OU Kalman filter + RTS
# smoother over the whole station-phase vector, observing the baseline phase
# DIFFERENCES directly with a per-station temporal OU prior — closing and denoising
# in a single recursive estimator (better at low SNR than per-AP-solve-then-smooth,
# where a single AP is poorly conditioned). Mutates `phase[:, 1, :]` in place; the
# caller's detrend and node→feed expansion run afterwards.
#
# State = `nant` station phases. `ap_rows` arrive with each row's source term
# already removed, so a row is a pure node difference and the filter needs no
# augmented source dimension. The unobservable common mode is pinned near 0 by the
# OU prior (the paper needs no reference station); we re-gauge to the anchor
# afterwards for pipeline consistency. Requires one node per station.
function _solve_gp_joint!(
        phase, track_w, ap_rows, nant::Integer, times, anchor::Integer, sm::JointOUSmoother,
    )
    nap = length(times)
    # Compute type flows from the data, not from the smoother's field types.
    T = float(promote_type(eltype(phase), eltype(track_w), eltype(times)))

    # Seed (ref-gauged, unwrapped) per-station track from the per-AP solve.
    θseed = [T(phase[i, 1, ap]) for i in 1:nant, ap in 1:nap]

    # Per-station centering mean (weighted): the OU prior reverts to 0, so a nonzero-
    # mean track would be biased. Removed again by the downstream detrend (so its
    # exact value is inert to the output — it only de-biases the solve).
    mθ = [_weighted_mean_finite(view(θseed, i, :), view(track_w, i, 1, :)) for i in 1:nant]

    # A station with no gated row anywhere in the scan has an all-zero column in every
    # H below, so the filter never updates it: it holds the OU prior mean for the whole
    # solve and the re-gauge turns that into minus the anchor's common mode — the same
    # information-free track for every such station. They are excluded from the hyper
    # fit and from the write-back.
    seen = [any(>(0), view(track_w, i, 1, :)) for i in 1:nant]

    # Per-station OU hypers from the (centered) seed track. The reference track is
    # structurally 0 (per-AP gauge) so it carries no variance info — give it and any
    # degenerate/uncovered station the ensemble-median dynamics, so the JOINT solve
    # treats every station as a real OU process (none pinned during the solve).
    τ_lo, τ_hi = _ou_tau_bounds(times)
    τv = fill(T(sm.coherence_time), nant)
    σ2v = fill(T(1.0e-2), nant)
    welldet = [seen[i] && i != anchor for i in 1:nant]
    for i in 1:nant
        welldet[i] || continue
        _, _, τv[i], σ2v[i] = _track_ou_hypers(
            view(θseed, i, :), view(track_w, i, 1, :), times;
            τ0 = sm.coherence_time, τ_lo = τ_lo, τ_hi = τ_hi, fit = sm.fit_hypers,
        )
    end
    σ2_med = let g = [σ2v[i] for i in 1:nant if welldet[i]]
        isempty(g) ? T(1.0e-2) : median(g)
    end
    τ_med = let g = [τv[i] for i in 1:nant if welldet[i]]
        isempty(g) ? T(sm.coherence_time) : median(g)
    end
    for i in 1:nant
        welldet[i] || (τv[i] = τ_med; σ2v[i] = σ2_med)
    end
    τall = τv
    σ2all = σ2v

    # Per-row measurement variance (1/snr²). The observation GEOMETRY needs no
    # buffer: `ap_rows` goes to the filter as-is, which reads `a`/`b` from each row.
    # `T[...]`, not `[...]`: `T` is a runtime value, so an AP with no rows would
    # otherwise collect to `Vector{Any}` while a populated one gives `Vector{T}`,
    # widening `rs` to `Vector{Vector}` — and the filter promotes its working type
    # from `eltype(eltype(rs))`.
    rs = [T[inv(T(row.w)) for row in ap_rows[ap]] for ap in 1:nap]
    ys = [Vector{T}(undef, length(ap_rows[ap])) for ap in 1:nap]

    # Current full (uncentered) estimate, seeded from the per-AP solve.
    θf = [isfinite(θseed[i, ap]) ? θseed[i, ap] : mθ[i] for i in 1:nant, ap in 1:nap]
    twoπ = 2 * T(π)

    # Iterated Kalman: rewrap each raw observation toward the current joint model,
    # then re-run the forward/backward smoother (relinearizing the ±2π branch).
    for _ in 1:max(sm.phase_rewrap_iters, 1)
        for ap in 1:nap
            rows = ap_rows[ap]
            y = ys[ap]
            for j in eachindex(rows, y)
                row = rows[j]
                model = θf[row.a, ap] - θf[row.b, ap]
                centre = mθ[row.a] - mθ[row.b]
                raw = T(row.val)
                y[j] = raw + twoπ * round((model - raw) / twoπ) - centre
            end
        end
        xf, Pf, xp, Pp, avecs, _ = kalman_ou_mv_filter(ap_rows, ys, rs, times; τ = τall, σ2 = σ2all)
        xs, _ = rts_smooth_mv(xf, Pf, xp, Pp, avecs)
        for ap in 1:nap
            for i in 1:nant
                θf[i, ap] = xs[i, ap] + mθ[i]
            end
        end
    end

    # Re-gauge to the anchor (its phase → 0 per AP), matching the per-AP path.
    for ap in 1:nap
        θref = θf[anchor, ap]
        isfinite(θref) || continue
        for i in 1:nant
            θf[i, ap] -= θref
        end
    end

    # Write back the joint track; internal gaps stay interpolated by the filter. An
    # unseen station keeps the per-AP solve's NaN, so it is never fabricated into θ
    # (`adhoc_scan!` gates its θ write on `isfinite`, not on `covered`).
    for i in 1:nant
        seen[i] || continue
        phase[i, 1, :] .= view(θf, i, :)
    end
    return phase
end

# One full per-AP sweep: solve every AP's station phases from `ap_rows` (with
# the anchor-gauged warm-start snapshot carrying 2π-branch continuity across
# APs), restitch anchor-dropout APs, and unwrap each (station, node) track.
# Shared by the seed alternation passes and the complex-domain refinement
# passes, which differ only in how their rows were built.
function _solve_ap_sweep!(
        phase, covered, track_w, ap_rows, nant::Integer, anchor::Integer,
        rewrap::Integer, max_stale::Integer,
    )
    nap = length(ap_rows)
    fill!(phase, convert(eltype(phase), NaN))
    fill!(covered, false)
    fill!(track_w, zero(eltype(track_w)))
    prev_phase = fill(convert(eltype(phase), NaN), nant, 2)  # cells no anchor-present AP has covered yet
    prev_age = zeros(Int, nant, 2)    # APs since a cell was last refreshed (staleness)
    for ap in 1:nap
        rows = ap_rows[ap]
        for row in rows
            track_w[row.a, row.na, ap] += row.w
            track_w[row.b, row.nb, ap] += row.w
        end
        # The anchor has data this AP iff some observation touches it (⇒ the solve
        # is pinned at anchor=0). Seed only then, and only with fresh cells in the
        # anchor gauge.
        ref_here = any(r -> r.a == anchor || r.b == anchor, rows)
        seed = nothing
        if ref_here
            seed = fill(NaN, nant, 2)
            for a in 1:nant, n in 1:2
                (isfinite(prev_phase[a, n]) && prev_age[a, n] <= max_stale) &&
                    (seed[a, n] = prev_phase[a, n])
            end
        end
        # No usable warm-start snapshot (first AP, all cells stale, or an
        # anchor-dropout AP): seed from the circular phasor solve instead of
        # trusting the tree-initialized linear solve's 2π branch — see
        # `_circular_ap_seed`.
        if seed === nothing || !any(isfinite, seed)
            seed = _circular_ap_seed(rows, nant, anchor)
        end
        ph, cov, _, _ = _solve_observable(
            rows, nant, PinAntenna(anchor); rewrap = rewrap,
            seed_phase = seed,
        )
        phase[:, :, ap] .= ph
        covered[:, :, ap] .= cov
        # Refresh the anchor-gauged snapshot ONLY from anchor-present APs (keep the
        # last known value for a station absent this AP, so a brief dropout does not
        # reset the branch); age every cell and zero the ones refreshed here.
        prev_age .+= 1
        if ref_here
            for a in 1:nant, n in 1:2
                if cov[a, n] && isfinite(ph[a, n])
                    prev_phase[a, n] = ph[a, n]
                    prev_age[a, n] = 0
                end
            end
        end
    end

    # Restitch the per-AP gauge when the anchor drops out (K3). Each per-AP solve
    # pins the anchor; in APs where it has no data the solve falls back to a
    # different pin node, so that AP's whole solution is offset by an arbitrary
    # (non-2π) constant — which would otherwise inject a spurious common-mode jump
    # into every station's track. Re-reference those APs to the trusted frame from
    # neighbouring anchor-present APs via the overlapping stations.
    _restitch_refant_gauge!(phase, covered, track_w, anchor)

    # Unwrap each (station, node) track across APs (per-AP solves share the ref
    # gauge, so a track is continuous up to ±2π steps the unwrap removes).
    for a in 1:nant, n in 1:2
        any(isfinite, @view phase[a, n, :]) || continue
        phase[a, n, :] .= unwrap_phase_track(phase[a, n, :]; weights = track_w[a, n, :])
    end
    return phase
end

# Per-(baseline, product) complex source term for the Gauss–Newton refinement:
# the inverse-variance mean of the model-derotated per-AP visibilities over the
# WHOLE scan, so its SNR is the track's rather than one AP's, and its |s̄|² is
# the signal power the linearized rows are weighted by. Also returns how many
# APs informed each term (its identifiability count).
function _complex_source_means(rbar, wbar, phase, bl_pairs, feeds, tying)
    nbl, npol, nap = size(rbar)
    sbar = zeros(ComplexF64, nbl, npol)
    nrm = zeros(Float64, nbl, npol)
    napu = zeros(Int, nbl, npol)
    for bi in 1:nbl, p in 1:npol
        a, b = bl_pairs[bi]
        a == b && continue
        na = _feed_node(tying, feeds[p][1])
        nb = _feed_node(tying, feeds[p][2])
        (na == 0 || nb == 0) && continue
        for ap in 1:nap
            w = wbar[bi, p, ap]
            r = rbar[bi, p, ap]
            (isfinite(r) && isfinite(w) && w > 0) || continue
            dphi = phase[a, na, ap] - phase[b, nb, ap]
            isfinite(dphi) || continue
            sbar[bi, p] += r * cis(-dphi)     # r = Σ w·V ⇒ this is Σ w·V·e^{-iΔφ̂}
            nrm[bi, p] += Float64(w)
            napu[bi, p] += 1
        end
        nrm[bi, p] > 0 && (sbar[bi, p] /= nrm[bi, p])
    end
    return sbar, napu
end

# Linearized (Gauss–Newton) rows for one AP, in the complex domain. Around the
# current tracks, `V̄·conj(s̄)e^{-iΔφ̂} ≈ |s̄|²(1 + i(Δφ − Δφ̂)) + n·conj(s̄)`, so
#
#     val  = Δφ̂ + Im(V̄·conj(s̄)e^{-iΔφ̂}) / |s̄|²
#     info = 2|s̄|² / σ²        (σ² the complex noise power of V̄, data-driven)
#
# is a LINEAR measurement of Δφ with exactly Gaussian noise — valid at any
# per-AP SNR, so every AP with data and a track enters, ungated: unlike a
# per-AP extracted phase, whose information collapses nonlinearly below
# SNR ≈ 1, the low-SNR APs here simply carry their honest (small) weight.
# `src = 0`: the source term is already divided out through `conj(s̄)`.
function _linearized_ap_rows(rbar, wbar, ap::Integer, bl_pairs, feeds, noise2, tying, phase, sbar)
    rows = _ObsRow[]
    nbl = length(bl_pairs)
    @inbounds for bi in 1:nbl, p in eachindex(feeds)
        a, b = bl_pairs[bi]
        a == b && continue
        na = _feed_node(tying, feeds[p][1])
        nb = _feed_node(tying, feeds[p][2])
        (na == 0 || nb == 0) && continue
        w = wbar[bi, p, ap]
        r = rbar[bi, p, ap]
        (isfinite(r) && isfinite(w) && w > 0) || continue
        dphi = phase[a, na, ap] - phase[b, nb, ap]
        isfinite(dphi) || continue
        s = sbar[bi, p]
        s2 = abs2(s)
        (isfinite(s2) && s2 > 0) || continue
        z = imag((r / w) * conj(s) * cis(-dphi)) / s2
        isfinite(z) || continue
        n2 = noise2[bi, p]
        # Fall back to the WEIGHT column's noise claim when the track is too
        # short to estimate its own (mirrors `_adhoc_ap_rows`'s fallback).
        wrow = isfinite(n2) && n2 > 0 ? 2 * s2 / n2 : s2 * Float64(w)
        push!(rows, _ObsRow(a, b, na, nb, dphi + z, wrow, 0))
    end
    return rows
end

function solve_adhoc_phasing(
        rbar::AbstractArray{<:Complex, 3}, wbar::AbstractArray{<:Real, 3},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString},
        nant::Integer, times::AbstractVector;
        gauge::AbstractGauge = PinAntenna(1),
        smoother::AbstractAdhocSmoother = SavitzkyGolaySmoother(),
        tying::AbstractFeedTying = PerFeed(),
    )
    nbl, npol, nap = size(rbar)
    size(wbar) == size(rbar) || error("rbar and wbar must have the same shape")
    nbl == length(bl_pairs) || error("rbar has $nbl baselines; bl_pairs has $(length(bl_pairs))")
    npol == length(pol_products) || error("rbar has $npol products; pol_products has $(length(pol_products))")
    nap == length(times) || error("rbar has $nap APs; times has $(length(times))")
    tying isa ReferenceRelative && error(
        "solve_adhoc_phasing cannot use a ReferenceRelative adhoc component: the " *
            "partner feed reads TWO parameter blocks, which the single-node-per-row " *
            "solve cannot represent. Use SharedFeeds or PerFeed.",
    )
    _requires_single_node(smoother) && _feed_node(tying, 1) != _feed_node(tying, 2) && error(
        "$(typeof(smoother)) requires one phase node per station (its Kalman state " *
            "is one dimension per station), but the adhoc component ties feeds as " *
            "$(typeof(tying)). Use OUSmoother for an independent per-feed track.",
    )
    feeds = [correlation_feed_pair(p) for p in pol_products]

    # Solved on the (station, NODE) graph; expanded back onto the feed axis at the end.
    phase = fill(convert(eltype(wbar), NaN), nant, 2, nap)
    covered = falses(nant, 2, nap)
    # Track per-(station,node) coherent weight for smoothing / detrend.
    track_w = zeros(eltype(wbar), nant, 2, nap)

    # Per-(baseline, product) noise of the coherent track, estimated data-driven
    # from its AP-to-AP scatter — see `_track_noise2`. The per-AP coherent SNR² is
    # then |V̄_ap|² / noise², which is SCALE-INVARIANT in the WEIGHT column: the
    # naive `|rbar|²/wbar` is only a true SNR when WEIGHT is calibrated inverse-
    # variance, and on raw correlator output (uncalibrated/uniform weights, common
    # in FITS-IDI) it is mis-scaled by an arbitrary factor, silently dropping every
    # row at any fixed `snr_floor` and killing the whole adhoc stage. For calibrated
    # weights `noise² → 1/wbar`, so this reduces to the old `|rbar|²/wbar` exactly.
    noise2 = [_track_noise2(rbar, wbar, bi, p, nap) for bi in 1:nbl, p in 1:npol]

    # Effective per-scan ANCHOR station: the gauge's preferred station when it observes in this scan,
    # else the best-covered station (largest total gated row weight). Everything
    # gauge-related below — the per-AP pin, the warm-start seed condition, the
    # gauge restitch, and the joint solve's re-gauge — keys on the anchor being
    # PRESENT. Keying on the literal reference disabled ALL of it on scans that
    # never see the reference (common in multi-subarray tracks: VR2505's
    # 0607-157 scan has no GS): the warm start never armed, so the K3 per-AP 2π
    # branch flips returned on weakly-constrained stations, and the per-AP pin
    # (lowest covered node) could hop with coverage flicker with the restitcher
    # never engaging. The anchor choice is inert to the applied correction (a
    # per-AP common mode cancels on every baseline); it exists so the
    # per-station tracks are temporally consistent — i.e. smoothable.
    # The SNR gate does not depend on the source terms, so every AP's gated rows are
    # built once and reused by every alternation pass (and by the joint smoother).
    raw_rows = [
        _adhoc_ap_rows(rbar, wbar, ap, bl_pairs, feeds, noise2, smoother.snr_floor^2, tying)
            for ap in 1:nap
    ]

    # Source terms, one per (baseline, product), indexed by each row's `src`.
    # `source_iters == 1` never updates them, so the model reduces exactly to a pure
    # station-difference solve with no source term.
    x = zeros(Float64, nbl * npol)
    fit_source = smoother.source_iters >= 2
    keep = trues(nbl * npol)
    if fit_source
        naps_src = zeros(Int, nbl * npol)
        for ap in 1:nap, row in raw_rows[ap]
            row.src == 0 || (naps_src[row.src] += 1)
        end
        # A (baseline, product) seen at a single AP is absorbed EXACTLY by its own
        # source term: it constrains no node phase, and admitting it would only
        # inflate the apparent coverage. Two APs is the identifiability threshold,
        # not a tuning choice.
        keep .= naps_src .>= 2
        # Seed each source term from its own observations. The gauge the per-track
        # demean imposes leaves each node track with ~zero scan mean, so a row's
        # scan-mean IS its source term to first order. Seeding from 0 instead would
        # make the first solve fit rows a full source phase away from their model,
        # which for a source phase near ±π locks the wrong 2π branch — one the
        # later passes inherit through the warm start and cannot leave.
        _update_source_terms!(x, raw_rows, zeros(eltype(phase), nant, 2, nap), nap)
    end
    ap_rows = [_source_corrected_rows(raw_rows[ap], x, keep) for ap in 1:nap]

    # Effective per-scan ANCHOR station: the gauge's preferred station when it observes in this scan,
    # else the best-covered station (largest total gated row weight). Everything
    # gauge-related below — the per-AP pin, the warm-start seed condition, the
    # gauge restitch, and the joint solve's re-gauge — keys on the anchor being
    # PRESENT. Keying on the literal reference disabled ALL of it on scans that
    # never see the reference (common in multi-subarray tracks: VR2505's
    # 0607-157 scan has no GS): the warm start never armed, so the K3 per-AP 2π
    # branch flips returned on weakly-constrained stations, and the per-AP pin
    # (lowest covered node) could hop with coverage flicker with the restitcher
    # never engaging. The anchor choice is inert to the applied correction (a
    # per-AP common mode cancels on every baseline); it exists so the
    # per-station tracks are temporally consistent — i.e. smoothable.
    anchor = let wtot = zeros(nant)
        for ap in 1:nap, row in ap_rows[ap]
            wtot[row.a] += row.w
            wtot[row.b] += row.w
        end
        # A ranked gauge walks its references before falling back to the
        # best-observed station, so a dropout costs the next choice, not an
        # arbitrary hop.
        cand = gauge_station_order(gauge, nant)
        j = findfirst(a -> 1 <= a <= nant && wtot[a] > 0, cand)
        j !== nothing ? Int(cand[j]) : (all(iszero, wtot) ? 1 : argmax(wtot))
    end

    # Carry solved node phases forward as a temporal warm-start for the next AP's
    # 2π-branch selection (continuity ⇒ no per-AP branch flips on weakly-constrained
    # stations). The seed is an ANCHOR-GAUGED snapshot: it is refreshed only from
    # APs where the anchor has data (so every stored cell is in the anchor=0 gauge
    # and the snapshot is mutually consistent) and applied only when the CURRENT AP
    # also has the anchor (so the per-AP spanning-tree seed it partially overrides
    # is anchor=0 too). Skipping the seed on anchor-dropout APs avoids mixing an
    # anchor=0 seed with a differently-anchored tree seed, which would mis-pick the
    # 2π branch on a seeded↔tree boundary edge — the arbitrary non-2π offset such
    # APs carry (K3). The snapshot is rebuilt from scratch by each alternation pass,
    # so a pass never inherits a branch chosen against a stale set of source terms.
    #
    # Trust a seed only while the atmosphere cannot yet have drifted past ±π (beyond
    # which the old branch is no longer a safe guide): ~π of drift takes a few
    # coherence times. `times` is in seconds; `T_AP` is its median spacing.
    dts0 = filter(>(0), diff(sort(times)))
    t_ap0 = isempty(dts0) ? 1.0 : float(median(dts0))
    max_stale = max(10, ceil(Int, 3 * _adhoc_coherence_time(smoother) / t_ap0))
    # Alternating minimization of the joint (node phases, source terms) least
    # squares: each pass re-solves every AP with the current source terms removed,
    # then re-estimates each source term from the residual it leaves. Both blocks
    # are conditionally linear, so the alternation descends a convex quadratic.
    # With `source_iters == 1` the source terms are never fitted and stay at 0, so
    # the model reduces to a pure station-difference solve.
    for iter in 1:max(smoother.source_iters, 1)
        _solve_ap_sweep!(
            phase, covered, track_w, ap_rows, nant, anchor,
            smoother.phase_rewrap_iters, max_stale,
        )
        (fit_source && iter < smoother.source_iters) || break
        moved = _update_source_terms!(x, raw_rows, phase, nap)
        for ap in 1:nap
            ap_rows[ap] = _source_corrected_rows(raw_rows[ap], x, keep)
        end
        moved <= smoother.source_tol && break
    end

    # Smooth: dispatch on the smoother type. Per-track smoothers loop the (station,
    # node) tracks (`window = :auto` sets a PER-STATION Savitzky–Golay window from the
    # EHT-HOPS `T_dof`, Eqs 21–22); the joint solve runs one multivariate OU Kalman
    # over all station phases (re-gauged to the anchor, matching the per-AP path);
    # `NoSmoothing` is a no-op. See `apply_adhoc!`.
    apply_adhoc!(smoother, phase, track_w, times; anchor = anchor, nant = nant, ap_rows = ap_rows)

    # Gauss–Newton refinement in the COMPLEX domain. Everything above is the
    # SEED: the phase-extraction solve's spanning-tree unwrap and warm starts
    # settle the global 2π branch, which no local linearization can, and the
    # SNR gate is confined to that seeding role. Each pass here re-derives the
    # per-(baseline, product) complex source terms from the whole scan,
    # linearizes every AP's residual around the current tracks
    # (`_linearized_ap_rows`), and re-solves and re-smooths on those rows —
    # every AP entering at its exact first-order information, ungated. With
    # the smoothing pass inside, each iteration is an extended-Kalman/RTS
    # step and the loop is Gauss–Newton on the MAP objective of the complex
    # data; the innovations start on the seed's branch, so the linearization
    # stays inside ±π by construction.
    sbar_ref = nothing
    nap_ref = nothing
    for _ in 1:max(smoother.complex_iters, 0)
        sbar, napu = _complex_source_means(rbar, wbar, phase, bl_pairs, feeds, tying)
        sbar_ref = sbar
        nap_ref = napu
        ref_rows = [
            _linearized_ap_rows(rbar, wbar, ap, bl_pairs, feeds, noise2, tying, phase, sbar)
                for ap in 1:nap
        ]
        _solve_ap_sweep!(
            phase, covered, track_w, ref_rows, nant, anchor,
            smoother.phase_rewrap_iters, max_stale,
        )
        apply_adhoc!(smoother, phase, track_w, times; anchor = anchor, nant = nant, ap_rows = ref_rows)
    end

    # Demean per track: remove the per-scan mean so adhoc does not alias the Stage-B
    # constant phase — this also fixes the (per-station constant ↔ source term)
    # gauge. The slope (residual rate) is intentionally kept — see `_detrend_track!`.
    if smoother.detrend
        for a in 1:nant, n in 1:2
            _detrend_track!(@view(phase[a, n, :]), @view(track_w[a, n, :]))
        end
    end

    # Impose the gauge LAST. Detrending removes each track's temporal mean, which
    # offsets every AP by the same constant, so applying the gauge after it leaves
    # the per-AP convention exact and the detrended tracks zero-mean up to one
    # global constant — and that constant is a per-AP common mode, which cancels on
    # every baseline. The two conventions are compatible only up to that constant;
    # this order is what makes the gauge the exact one.
    _apply_ap_gauge!(phase, covered, gauge, 2)

    # Expand the (station, NODE) solution onto the feed axis the caller indexes:
    # feeds sharing a node get identical tracks (so a `SharedFeeds` adhoc contributes
    # exactly zero inter-feed phase), and a feed the component does not parameterize
    # (node 0) stays NaN/uncovered.
    phase_out = fill(convert(eltype(wbar), NaN), nant, 2, nap)
    covered_out = falses(nant, 2, nap)
    for f in 1:2
        n = _feed_node(tying, f)
        n == 0 && continue
        phase_out[:, f, :] .= @view phase[:, n, :]
        covered_out[:, f, :] .= @view covered[:, n, :]
    end

    # The refinement's complex source means supersede the seed's phase-only
    # alternation estimates: same per-(baseline, product) constant, measured
    # against the final tracks over every usable AP. Two APs stays the
    # identifiability threshold.
    if sbar_ref !== nothing
        for p in 1:npol, bi in 1:nbl
            k = (p - 1) * nbl + bi
            if nap_ref[bi, p] >= 2 && abs2(sbar_ref[bi, p]) > 0
                x[k] = angle(sbar_ref[bi, p])
                keep[k] = true
            else
                keep[k] = false
            end
        end
    end

    # Source terms as measured, `NaN` where too poorly sampled to identify.
    source = [keep[(p - 1) * nbl + bi] ? x[(p - 1) * nbl + bi] : NaN for bi in 1:nbl, p in 1:npol]

    tdim = Ti(float.(times))
    axs = (Ant(1:nant), Feed(1:2), tdim)
    return DimensionalData.DimStack(
        (
            phase = DimArray(phase_out, axs),
            covered = DimArray(covered_out, axs),
            source = DimArray(source, (Baseline(1:nbl), Pol(1:npol))),
        )
    )
end

# ── Per-scan pipeline entry ───────────────────────────────────────────────────

# Accumulate one band leaf's per-AP residual into `rbar`/`wbar` as the
# inverse-variance mean of the data, already gain-corrected (and reweighted by
# |gain|², matching `apply_calibration`) through the pipeline's transform chain
# before this kernel ever sees it.
function _accumulate_leaf_rbar!(rbar, wbar, V, W)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        for bi in 1:nbl
            for tt in 1:nti, c in 1:nchan
                w = W[c, tt, bi, p]
                (w > 0 && isfinite(w)) || continue
                v = V[c, tt, bi, p]
                isfinite(v) || continue
                rbar[bi, p, tt] += w * v
                wbar[bi, p, tt] += w
            end
        end
    end
    return rbar, wbar
end

"""
    adhoc_scan!(θ, stack, win::GeometryWindow, adhoc_plan, adhoc, gauge, nant;
                executor = DynamicScheduler()) -> θ

The per-integration atmospheric-phase (adhoc) solve of one scan window — the
"caller" the module docstring above refers to. On data already gain-corrected
through the pipeline's transform chain: accumulate the per-(baseline, product,
AP) inverse-variance residual, solve the globally-closing per-AP station phase
through the pluggable `adhoc` smoother ([`solve_adhoc_phasing`](@ref)), and
write this scan's `PerIntegration` θ slots
(disjoint per scan — concurrent groups may solve in parallel). The feed tying
comes from `adhoc_plan`, so the number of phase nodes per station is the
model's choice and needs no separate argument.
"""
function adhoc_scan!(
        θ, stack::AbstractDimStack, win::GeometryWindow, adhoc_plan, adhoc, gauge, nant;
        executor = DynamicScheduler(),
    )
    geom = win.geom
    bl_pairs = collect(UVData.baselines(stack).pairs)
    pols = String.(pol_products(stack))
    tg = Float64.(timestamps(stack))
    ci = win.chan_idx
    g_ti = win.ti_idx
    nbl = length(bl_pairs)
    npol = length(pols)
    nap = length(tg)
    # Per-band accumulation (the window's per-spw channel blocks stand in for the
    # band leaves) fanned out over the inner `executor` — this loop (the residual
    # sum over every visibility) dominates the adhoc pass on many-band data. Each
    # BLOCK owns the trailing slice `li` of the partial buffers and the slices
    # fold in block order, so the float association is fixed by the data layout
    # alone — the result is bit-deterministic at any chunking.
    blocks = _spw_blocks(geom, ci)
    nblk = length(blocks)
    rparts = zeros(ComplexF64, nbl, npol, nap, nblk)
    wparts = zeros(Float64, nbl, npol, nap, nblk)
    tforeach(1:nblk; scheduler = executor) do li
        r = blocks[li]
        _accumulate_leaf_rbar!(
            view(rparts, :, :, :, li), view(wparts, :, :, :, li),
            view(stack[:vis], r, :, :, :), view(stack[:weights], r, :, :, :),
        )
    end
    rbar = zeros(ComplexF64, nbl, npol, nap)
    wbar = zeros(Float64, nbl, npol, nap)
    for li in 1:nblk
        rbar .+= view(rparts, :, :, :, li)
        wbar .+= view(wparts, :, :, :, li)
    end
    # `tg` is in hours; pass SECONDS so the adhoc's `:auto` window (T_AP / T_coh) is
    # in physical units. Detrend uses only the mean, so the scaling is otherwise inert.
    as = solve_adhoc_phasing(
        rbar, wbar, bl_pairs, pols, nant, tg .* 3600.0;
        gauge = gauge, smoother = adhoc, tying = adhoc_plan.tying,
    )
    adhoc_leaf = _component_leaf(adhoc_plan, θ)
    for (ap, gti) in enumerate(g_ti)
        tseg = adhoc_plan.tseg_id[gti]
        for ant in 1:nant, feed in 1:2
            val = as.phase[ant, feed, ap]
            isfinite(val) || continue
            node = _feed_node(adhoc_plan.tying, feed)
            node == 0 && continue
            adhoc_leaf[1, node, 1, tseg, ant] = val
        end
    end
    return θ
end

# Restitch per-AP gauges so the reference frame is consistent across APs even
# when the anchor drops out (K3). When the anchor is solved in an AP, that AP's
# per-AP solve already pins it (frame = anchor phase 0) and we trust it,
# refreshing the running anchor from this AP's solved cells (so real drift
# propagates). When the anchor is ABSENT, the AP's solve anchored on a different
# node, so it carries an arbitrary global offset δ; we estimate δ as the
# weighted circular mean over the cells common to this AP and the anchor, and
# subtract it from every solved cell of the AP. This is a no-op when the anchor is
# present in every AP (so it never perturbs the well-anchored case), and it only
# removes a single global per-AP constant — per-(station, feed) means and slopes
# are still handled later by `_detrend_track!`. Leading APs with no trusted
# anchor yet are left untouched (best effort). A multi-component AP keeps one
# global δ dominated by the largest overlap; per-island offsets remain a
# fundamental gauge freedom (one additive freedom per connected component).
# Put each AP on the gauge's own convention, over a station set that does NOT move
# between APs.
#
# The per-AP common mode is unobservable — it cancels on every baseline — so this
# changes how the tracks read, never the applied correction. That is exactly why
# the set must be fixed: a sum taken over whatever stations happen to be covered
# shifts frame whenever coverage flickers, putting steps into every track for a
# quantity that carries no information. Summing over the stations covered in EVERY
# AP keeps one frame for the whole scan.
#
# A pinned gauge needs nothing here: the per-AP solves already pin the anchor and
# `_restitch_refant_gauge!` has carried that frame across the APs where it drops out.
_apply_ap_gauge!(phase, covered, ::AbstractGauge, nnode::Integer) = phase

function _apply_ap_gauge!(phase, covered, gauge::ZeroSumPhase, nnode::Integer)
    nant, _, nap = size(phase)
    # ONE constant per AP, across BOTH feed nodes. Cross-hand rows join the two
    # feeds into a single connected component carrying a single additive freedom,
    # so a separate constant per feed would invent a second one and shift every
    # cross-hand difference `φ_{a,1} − φ_{b,2}` by the gap between them — breaking
    # the reconstruction the gauge must leave untouched. Subtracting the same
    # constant from every cell cancels in every baseline difference, parallel and
    # cross alike.
    #
    # The summed cells are those covered in EVERY AP: a sum over whatever happens
    # to be covered moves frame with coverage, putting steps into every track for a
    # quantity that carries no information.
    #
    # `weights` describes the station-solve constraint row, whose nodes are not
    # these cells, so the per-AP frame is unweighted.
    cells = [
        (a, n) for a in 1:nant for n in 1:nnode
            if all(covered[a, n, ap] for ap in 1:nap) &&
            (gauge.antennas === nothing || a in gauge.antennas)
    ]
    isempty(cells) && return phase
    for ap in 1:nap
        tot = zero(eltype(phase))
        cnt = 0
        for (a, n) in cells
            v = phase[a, n, ap]
            isfinite(v) || continue
            tot += v
            cnt += 1
        end
        cnt == 0 && continue
        d = tot / cnt
        for a in 1:nant, n in 1:nnode
            isfinite(phase[a, n, ap]) && (phase[a, n, ap] -= d)
        end
    end
    return phase
end

function _restitch_refant_gauge!(phase, covered, track_w, ref_station::Integer)
    nant, _, nap = size(phase)
    anchor = fill(NaN, nant, 2)
    have_anchor = false
    for ap in 1:nap
        ref_present = covered[ref_station, 1, ap] || covered[ref_station, 2, ap]
        if !ref_present && have_anchor
            # Register PER FEED. With cross hands the two feeds share a component
            # but carry two gauge freedoms (the overall phase pin and the feed-2
            # EVPA pin); when the anchor drops out both fall back to a different
            # antenna, shifting each feed by its own constant. The per-feed
            # convention (each feed gauged relative to the anchor's feed) matches the
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

# First-difference penalized smoother, via the generic penalized WLS solve
# with `A = I`, the penalty stacked as `√λ · D` (`D` the 1st-difference
# operator): minimize Σ w_k(φ_k − y_k)² + λ Σ(φ_{k+1}−φ_k)². Non-finite samples
# get zero data weight (interpolated by the penalty).
function _penalized_smooth(y::AbstractVector, w::AbstractVector, λ::Real)
    n = length(y)
    n == 0 && return collect(float.(y))
    wk = [(isfinite(y[k]) && isfinite(w[k]) && w[k] > 0) ? float(w[k]) : 0.0 for k in 1:n]
    yk = [wk[k] > 0 ? float(y[k]) : 0.0 for k in 1:n]
    # Guard against an all-unconstrained component (no data, λ = 0).
    all(iszero, wk) && iszero(λ) && return collect(float.(y))
    D = zeros(n - 1, n)
    for k in 1:(n - 1)
        D[k, k] = 1.0; D[k, k + 1] = -1.0
    end
    R = sqrt(λ) .* D
    return weighted_regularized_least_squares(Matrix(1.0I, n, n), yk, wk, R)
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
