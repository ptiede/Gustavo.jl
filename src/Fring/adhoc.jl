# ── Globally-closing adhoc phasing ───────────────────────────────────────────
#
# After the per-scan fringe solution there remains a fast, time-variable phase
# per station — atmospheric turbulence — that a per-scan constant cannot track.
# It is solved globally rather than against a reference antenna: per
# accumulation period (AP), station phases are fit from all baselines with the
# same feed-aware incidence WLS as the station solve, so the solution closes by
# construction and stays well-determined at low SNR with no dominant anchor.
#
# For baseline (a, b), correlation product p with feeds (fa, fb), at AP t:
#
#     y_{ab,p}(t) = φ_{na}(t) − φ_{nb}(t) + x_{ab,p} + ε,
#
# where `n· = _feed_node(tying, f·)` maps a feed onto its parameter node and
# `x_{ab,p}` is the source visibility phase, held constant over the scan.
#
# `x` must stay free per (baseline, product): that is what keeps the model
# independent of the polarization basis, since the source EVPA, the D-terms and
# the source's own closure phase then land in `x` rather than biasing the
# station tracks. One cross-hand phase shared by every baseline is the rank-one
# restriction `x_{ab,p} = ±c`, which cannot represent closure phase at all.
#
# `x` is degenerate with a per-station constant (`φ_a → φ_a + c_a`, `x_{ab,p} →
# x_{ab,p} − (c_a − c_b)`), a gauge the per-scan demean in step (5) fixes. A
# per-station slope is not degenerate with a constant `x`, so the residual-rate
# information survives the demean.
#
# Per scan: (1) the caller supplies residual baseline visibilities, already
# divided by the fringe solution and coherently frequency-averaged to one
# complex number per (baseline, product, AP); (2) alternate between solving the
# per-AP node phases over the graph's connected components and re-estimating
# each `x_{ab,p}` as the weighted circular mean of its residual; (3) unwrap each
# track across APs; (4) smooth — Savitzky–Golay, dense first-difference penalty,
# or GP (statespace.jl); (5) demean per track, so the adhoc track does not alias
# the Stage-B constant phase. Steps (1)–(3) are identical for every smoother,
# which only ever decides step (4).

# ── Adhoc smoothers: a pluggable type-based interface ────────────────────────
#
# After the per-AP global solve each (station, feed) phase track is smoothed.
# The smoother is a strategy type, one per method, rather than the
# `WLSEstimator`-based callables the amp-bandpass smoothers use: most adhoc
# smoothers are not WLS problems (`OUSmoother`/`JointOUSmoother` are Kalman/RTS
# filters, `NoSmoothing` is a no-op).
#
# Internal traits, with defaults in the traits section below:
# `_adhoc_coherence_time(sm)` is the assumed atmospheric `T_coh` in seconds,
# used by the per-AP warm-start staleness check; `_requires_single_node(sm)` is
# whether the smoother needs one phase node per station, which only the joint
# solve does.

"""
    AbstractAdhocSmoother

How the [`AdhocPhase`](@ref Gustavo.AdhocPhase) step smooths
the per-AP station phase tracks the adhoc solve produces
(`solve_adhoc_phasing`). Concretely `SavitzkyGolaySmoother` (the default),
`PenalizedSmoother`, `OUSmoother`, `JointOUSmoother`, or `NoSmoothing`.

# Implementing a smoother

Define a struct and:

    Gustavo.Fring.apply_adhoc!(sm::MySmoother, phase, track_w; obs, anchor)

the single dispatch point; mutates `phase` in place. `phase` and `track_w` are
`DimArray`s over `(Ant, FeedNode, Ti)`: each (station, feed node) phase track and
its per-AP coherent weight, `Ti` carrying the AP epochs in seconds. `obs` holds the
SNR-gated, source-corrected observations: `obs.val`, `obs.w` and the gate
`obs.mask` over `(StationPair, FeedPair, Ti)`, and `obs.nodes`, each cell's
`((a, na), (b, nb))` — the station index and feed node at either end; `anchor`
is the station the per-AP solves are pinned to. The default
`apply_adhoc!` smooths each (station, node) track independently through the
per-track hook, so a smoother that acts track by track implements only

    Gustavo.Fring.smooth_track!(sm::MySmoother, track, w) -> track

which overwrites `track` with its smoothed values; `lookup(track, Ti)` gives
its times.

A smoother declares which adhoc components it can solve with

    Gustavo.Fring.can_fit(sm::MySmoother, tc::Calibration.GainComponent, geom) -> Bool

checked at model-compile time. The default is the capability of the solve
machinery — per-AP constant phase over the global band, `SharedFeeds` or
`PerFeed`; `JointOUSmoother` restricts to `SharedFeeds` (its Kalman state is
one dimension per station).

Every smoother carries its solve options in an `options::AdhocOptions` field;
see [`AdhocOptions`](@ref).
"""
abstract type AbstractAdhocSmoother end

"""
    AdhocOptions(; snr_floor = 1.0, phase_rewrap_iters = 3, source_iters = 10,
                 source_tol = 1.0e-6, detrend = true, complex_iters = 2,
                 eltype = nothing)

The solve options every [`AbstractAdhocSmoother`](@ref) carries in its
`options` field, e.g. `SavitzkyGolaySmoother(; window = 7, options =
AdhocOptions(; snr_floor = 0.0))`.

`snr_floor` gates the seed pass only (the phase-extraction solve that fixes
the global 2π branch); `complex_iters` Gauss–Newton passes then re-fit the
tracks against the complex residuals with every AP entering ungated
(`complex_iters = 0` keeps the seed). `phase_rewrap_iters` bounds the
rewrap passes of each solve. `source_iters` alternations between the station
tracks and the per-(baseline, product) source phases run until the source
phases move less than `source_tol`; `source_iters = 1` fits no source term.

`detrend` removes each track's per-scan weighted mean, so the adhoc component
carries per-AP phase structure only and the per-scan constant stays with the
fringe stage's own constant term. The residual-rate slope is kept. Disable
only when the model has no other per-scan constant to alias against.

The solve works in the real type of the summed visibilities (`Float32` for an
MSv4 store); `eltype`, a real floating-point type, names another.
"""
Base.@kwdef struct AdhocOptions
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    source_iters::Int = 10
    source_tol::Float64 = 1.0e-6
    detrend::Bool = true
    complex_iters::Int = 2
    eltype::Union{Nothing, DataType} = nothing
    function AdhocOptions(snr_floor, phase_rewrap_iters, source_iters, source_tol, detrend, complex_iters, eltype)
        isnothing(eltype) || eltype <: AbstractFloat || throw(
            ArgumentError("`eltype` must be a real floating-point type or `nothing`, got $eltype"),
        )
        return new(snr_floor, phase_rewrap_iters, source_iters, source_tol, detrend, complex_iters, eltype)
    end
end

"""
    SavitzkyGolaySmoother(; window, order, coherence_time, structure_exponent,
                          options)

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

`options` is an [`AdhocOptions`](@ref).
"""
Base.@kwdef struct SavitzkyGolaySmoother <: AbstractAdhocSmoother
    window::Union{Int, Symbol} = :auto
    order::Int = 2
    coherence_time::Float64 = 10.0
    structure_exponent::Float64 = 5 / 3
    options::AdhocOptions = AdhocOptions()
end

"""
    PenalizedSmoother(; smoothness, options)

Dense first-difference (random-walk) penalized per-track smoother: minimizes
`Σ w_k(φ_k − y_k)² + smoothness·Σ(φ_{k+1}−φ_k)²`. No SparseArrays (the system is
tridiagonal, solved densely). Larger `smoothness` ⇒ stiffer track.

`options` is an [`AdhocOptions`](@ref).
"""
Base.@kwdef struct PenalizedSmoother <: AbstractAdhocSmoother
    smoothness::Float64 = 1.0
    options::AdhocOptions = AdhocOptions()
end

"""
    OUSmoother(; coherence_time, fit_hypers, options)

Per-station Ornstein–Uhlenbeck / Matérn-1/2 Gaussian-process per-track smoother via
an exact Kalman filter + RTS smoother (see [`smooth_ou_track`](@ref)), with
`coherence_time` as the τ seed. When `fit_hypers` (default `true`) the per-station
OU `(τ, σ²)` are fit by maximum Kalman marginal likelihood; when `false`,
`τ = coherence_time` and `σ²` is seeded from the track scatter (no optimization).

`options` is an [`AdhocOptions`](@ref).
"""
Base.@kwdef struct OUSmoother <: AbstractAdhocSmoother
    coherence_time::Float64 = 10.0
    fit_hypers::Bool = true
    options::AdhocOptions = AdhocOptions()
end

"""
    JointOUSmoother(; coherence_time, fit_hypers, options)

The paper-faithful JOINT solve — one multivariate OU Kalman over all station phases
observing baseline phase differences directly, closing and denoising together (better
at low SNR than per-track-solve-then-smooth). Seeded and rewrapped from the per-AP
solve; see `_solve_gp_joint!`. Requires one phase node per station (e.g. a
`SharedFeeds` adhoc component), since its state carries one dimension per station.
`coherence_time`/`fit_hypers` seed the per-station OU dynamics as in
[`OUSmoother`](@ref).

`options` is an [`AdhocOptions`](@ref).
"""
Base.@kwdef struct JointOUSmoother <: AbstractAdhocSmoother
    coherence_time::Float64 = 10.0
    fit_hypers::Bool = true
    options::AdhocOptions = AdhocOptions()
end

"""
    NoSmoothing(; options)

No smoothing: the raw per-AP global solve only (unwrap + optional detrend still run).

`options` is an [`AdhocOptions`](@ref).
"""
Base.@kwdef struct NoSmoothing <: AbstractAdhocSmoother
    options::AdhocOptions = AdhocOptions()
end

# ── Smoother interface traits ─────────────────────────────────────────────────
# The assumed atmospheric coherence time (seconds) — used by the per-AP warm-start
# staleness. Smoothers that carry no coherence time use the same default as those
# that do.
_adhoc_coherence_time(::AbstractAdhocSmoother) = 10.0
_adhoc_coherence_time(sm::SavitzkyGolaySmoother) = sm.coherence_time
_adhoc_coherence_time(sm::OUSmoother) = sm.coherence_time
_adhoc_coherence_time(sm::JointOUSmoother) = sm.coherence_time

# Whether the smoother needs one phase node per station. The joint solve's Kalman
# state is one dimension per station, so it cannot represent two independent feed
# tracks.
_requires_single_node(::AbstractAdhocSmoother) = false
_requires_single_node(::JointOUSmoother) = true

"""
    default_adhoc_terms(; feed = SharedFeeds()) -> GainModel

The default [`AdhocPhase`](@ref Gustavo.AdhocPhase) step model: one per-AP constant phase
over the global band — a `phase.adhoc` component with `Ti = PerIntegration()`.
`feed` is its feed tying: `SharedFeeds()` (the default) solves one track per
station — residual atmospheric phase is non-birefringent, and a feed-common
track contributes ZERO inter-feed phase, where a `PerFeed()` solve lets per-AP
noise differ between feeds and so injects spurious cross-hand scatter.
`PerFeed()` fits each feed's own track when the per-feed structure is real.

The adhoc stage solves exactly one phase component and no logamp.
"""
default_adhoc_terms(; feed::AbstractFeedTying = SharedFeeds()) =
    GainModel(phase = (; adhoc = GainComponent(ConstantTerm(); Ti = PerIntegration(), Feed = feed)))

# Capability declarations for the compile-time `can_fit`/`validate_model` seam
# (capability.jl; the step drives the checks in
# `model_components(::AdhocPhase, spec)`).
#
# What the solve machinery addresses: one constant per (feed node, AP) —
# `adhoc_scan!` writes leaf slot (param 1, node, freq segment 1, time segment,
# ant), and `solve_adhoc_phasing`'s single-node-per-row systems can tie feeds
# (`SharedFeeds`) or solve them independently (`PerFeed`) but cannot represent
# a `SingleFeed` scope.
_fits_adhoc_track(tc) =
    tc.term isa ConstantTerm && tc.Ti isa PerIntegration &&
    tc.Frequency isa GlobalFrequency && (tc.Feed isa PerFeed || tc.Feed isa SharedFeeds)

can_fit(::AbstractAdhocSmoother, tc, geom) = _fits_adhoc_track(tc)
# The joint solve needs one phase node per station (`_requires_single_node`).
can_fit(::JointOUSmoother, tc, geom) = _fits_adhoc_track(tc) && tc.Feed isa SharedFeeds

# The structural contract of the adhoc pass: exactly one per-integration phase
# component (the stage runs one globally-closing phase solve and writes one θ
# block), nothing in logamp.
function validate_model(sm::AbstractAdhocSmoother, model)
    ph = Calibration._flatten_components(model.phase)
    la = Calibration._flatten_components(model.logamp)
    length(ph) == 1 || throw(
        ArgumentError(
            "the adhoc stage solves exactly one per-integration phase component; the " *
                "model's phase group holds $(length(ph)). `default_adhoc_terms()` is the " *
                "standard model.",
        ),
    )
    isempty(la) || throw(
        ArgumentError(
            "the adhoc stage solves phase only; the model's logamp group must be empty, " *
                "got $(length(la)) component(s).",
        ),
    )
    return nothing
end

# ── apply_adhoc!: the single smoothing dispatch point ─────────────────────────
# Mutates the per-(station, node) phase track array `phase` in place. `obs` is
# the SNR-gated observations with the source term already removed, so every
# smoother sees the same observations; per-track smoothers ignore it.

# Default: loop the (station, node) tracks and apply the per-track hook.
function apply_adhoc!(sm::AbstractAdhocSmoother, phase, track_w; obs, anchor)
    for a in axes(phase, Ant), f in axes(phase, FeedNode)
        track = view(phase, a, f, :)
        # skip tracks that are NaN
        any(isfinite, track) || continue
        smooth_track!(sm, track, view(track_w, a, f, :))
    end
    return phase
end

# No smoothing — the per-AP solve stands as is.
apply_adhoc!(::NoSmoothing, phase, track_w; obs, anchor) = phase

# Joint state-space solve: one multivariate OU Kalman over all station phases
# observing baseline differences directly, seeded/rewrapped from the per-AP solve.
function apply_adhoc!(sm::JointOUSmoother, phase, track_w; obs, anchor)
    _solve_gp_joint!(phase, track_w, obs, anchor, sm)
    return phase
end

# ── smooth_track!: per-(station, feed) smoothing hook ─────────────────────────
# `track` is one (station, feed node) phase track (unwrapped, radians) over
# `Ti`, overwritten with its smoothed values; `w` is its per-AP coherent weight.

_track_times(track) = parent(lookup(track, Ti))

# Savitzky–Golay. `window = :auto` sets a per-station window from the EHT-HOPS
# `T_dof` (Eqs 21–22): an SNR-adaptive integration time scaled by the assumed
# coherence time. `T_AP` is the AP spacing; the per-AP coherent SNR² is the track's
# mean `w` (Σ baseline SNR²).
function smooth_track!(sm::SavitzkyGolaySmoother, track, w)
    win = if sm.window === :auto
        0 < sm.structure_exponent < 2 || error(
            "SavitzkyGolaySmoother: structure_exponent must be in (0, 2) for window = :auto " *
                "(got $(sm.structure_exponent); 5/3 = 3D Kolmogorov, 2/3 = 2D). Outside this " *
                "range the T_dof denominator (2 + α − 2^α) is non-positive.",
        )
        dts = filter(>(0), diff(sort(_track_times(track))))
        t_ap = isempty(dts) ? 1.0 : float(median(dts))
        m_coh = sm.coherence_time / t_ap
        tw = [w[ap] for ap in eachindex(w) if w[ap] > 0]
        rho2 = isempty(tw) ? 0.0 : sum(tw) / length(tw)
        _savgol_window_dof(rho2, m_coh, sm.structure_exponent, sm.order)
    else
        Int(sm.window)
    end
    return track .= savitzky_golay_smooth(track, w; window = win, order = sm.order)
end

# Dense first-difference penalized smoother.
smooth_track!(sm::PenalizedSmoother, track, w) = track .= _penalized_smooth(track, w, sm.smoothness)

# Ornstein–Uhlenbeck (Matérn-1/2) Gaussian-process smoother: an exact Kalman filter
# + RTS smoother with measurement variance 1/w and the process model set by the OU
# (τ, σ²) — fit per station by maximum marginal likelihood when `fit_hypers`.
# `w == 0` APs are missing (predicted through). Run on the mean-subtracted track (OU
# reverts to 0), then restore the mean; the downstream detrend removes it again anyway.
function smooth_track!(sm::OUSmoother, track, w)
    times = _track_times(track)
    τ_lo, τ_hi = _ou_tau_bounds(times)
    m, yc, τ, σ2 = _track_ou_hypers(track, w, times; τ0 = sm.coherence_time, τ_lo = τ_lo, τ_hi = τ_hi, fit = sm.fit_hypers)
    return track .= smooth_ou_track(yc, w, times; τ = τ, σ2 = σ2) .+ m
end

# Data-driven noise variance of one (baseline, product) coherent track
# `V̄ = r/w`, `r` and `w` its sums along time or frequency, from the robust
# scatter of successive differences. The source/atmosphere vary slowly from one
# sample to the next while noise is independent, so successive differences
# isolate the noise. For complex-Gaussian noise, `median(|ΔV̄|²) = 2 ln2 · σ²`
# (Δ of two samples has twice the variance, and the median of an exponential is
# `ln2 ×` its mean), so `σ² = median(|ΔV̄|²) / (2 ln2)`. Returns `NaN` when
# fewer than 4 differences are available (caller falls back).
function _track_noise2(r::AbstractVector, w::AbstractVector)
    C = eltype(r)
    T = real(C)
    d2 = T[]
    prev = C(NaN, NaN)
    for k in eachindex(r, w)
        wk = w[k]
        v = wk > 0 ? r[k] / wk : C(NaN, NaN)
        (isfinite(v) && isfinite(prev)) && push!(d2, abs2(v - prev))
        prev = v
    end
    length(d2) >= 4 || return T(NaN)
    return median(d2) / (2 * log(T(2)))
end

# `_track_noise2` of each cell's track along `along`, over `rbar`'s other dimensions.
function _cell_noise2(rbar, wbar, along)
    cells = DimensionalData.otherdims(rbar, along)
    n2 = [_track_noise2(view(rbar, I...), view(wbar, I...)) for I in DimensionalData.DimIndices(cells)]
    return DimArray(parent(n2), cells)
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
    # minimal window rather than producing Inf/NaN — `smooth_track!` validates α
    # first, so this only guards direct callers.
    denom = 2 + a - 2.0^a
    (0 < a < 2 && denom > 0) || return order + 1
    coef = (1 + a) * (2 + a) / (2.0^(-a) * a * denom)
    m_dof = (coef * float(m_coh)^a / float(rho2))^(1 / (a + 1))   # T_dof / T_AP, in APs
    isfinite(m_dof) || return order + 1
    n = max(order + 1, 1 + 2 * floor(Int, (order + 1) * m_dof / 2))
    return iseven(n) ? n + 1 : n
end

# Circular (complex-phasor) solve of one AP's system, z_a ← Σ_b w·e^{iφ_ab}·z_b,
# gauged at the anchor. It seeds the linear solve at APs with no usable
# warm-start snapshot and must not be replaced by a cold linear solve: the
# phases-as-values WLS has 2π-branch local minima when observations sit near
# ±π, and a cold-started AP can converge ~150° off a strong one, after which the
# warm-start chain locks that branch and relaxes toward truth across the scan,
# leaving a smooth ±π-scale arc in the station track. The phasor iteration is
# circular and so has no branch structure. Every gated cell is a pure node
# difference, its source term already removed, so all of them drive the iteration.
function _circular_ap_seed(val, w, mask, nodes, nant::Integer, anchor::Integer)
    T = eltype(val)
    cells = [I for I in eachindex(val, w, mask, nodes) if mask[I]]
    isempty(cells) && return nothing
    z = ones(Complex{T}, nant, 2)
    present = falses(nant, 2)
    for I in cells
        (a, na), (b, nb) = nodes[I]
        present[a, na] = true
        present[b, nb] = true
    end
    for _ in 1:50
        acc = zeros(Complex{T}, nant, 2)
        for I in cells
            (a, na), (b, nb) = nodes[I]
            R = cis(val[I])
            acc[a, na] += w[I] * R * z[b, nb]
            acc[b, nb] += w[I] * conj(R) * z[a, na]
        end
        for i in eachindex(z)
            present[i] || continue
            a = abs(acc[i])
            a > 0 && (z[i] = acc[i] / a)
        end
    end
    # One common gauge for the whole seed, the anchor when present. A fully
    # consistent seed is safe on anchor-dropout APs too: the rewrap uses
    # prediction differences, so a common gauge offset cancels.
    g = present[anchor, 1] ? conj(z[anchor, 1]) / abs(z[anchor, 1]) : one(Complex{T})
    ph = fill(T(NaN), nant, 2)
    for f in axes(ph, 2), a in axes(ph, 1)
        present[a, f] && (ph[a, f] = angle(z[a, f] * g))
    end
    return ph
end

# Each (station pair, feed pair) cell's two ends: `((a, na), (b, nb))`, the
# station's index in `stations` and the phase node of its feed under `tying`
# (0 where the component has none).
function _cell_nodes(rbar, stations, tying)
    slot = Dict(n => i for (i, n) in pairs(stations))
    station(n) = get(slot, n) do
        throw(ArgumentError("station `$n` of a station pair is not among the stations " * join(stations, ", ")))
    end
    sps = DimensionalData.dims(rbar, StationPair)
    fps = DimensionalData.dims(rbar, FeedPair)
    ends = [
        ((station(sa), _feed_node(tying, fa)), (station(sb), _feed_node(tying, fb)))
            for (sa, sb) in lookup(sps), (fa, fb) in lookup(fps)
    ]
    return DimArray(ends, (sps, fps))
end

# Whether a cell constrains two phase nodes: not an autocorrelation, and both
# feeds parameterized by the component.
_solvable(((a, na), (b, nb))) = a != b && na != 0 && nb != 0

# The seed pass's observations over `rbar`'s `(StationPair, FeedPair, Ti)`: each
# cell's phase `angle(r)`, weighted by its coherent SNR² and gated at
# `snr_floor2`. Every correlation product contributes: a cross-hand cell's extra
# phase is carried by its own free source term, so no product needs the
# polarization basis to be known.
function _adhoc_obs(rbar, wbar, nodes, noise2, snr_floor2::Real)
    T = real(eltype(rbar))
    val = similar(rbar, T)
    w = similar(rbar, T)
    mask = similar(rbar, Bool)
    for I in DimensionalData.DimIndices(rbar)
        cell = DimensionalData.otherdims(I, Ti)
        r, wr = rbar[I], wbar[I]
        n2 = noise2[cell]
        snr2 = isfinite(n2) && n2 > 0 ? abs2(r / wr) / n2 : abs2(r) / wr   # fall back if unestimable
        ok = _solvable(nodes[cell]) && isfinite(r) && abs(r) > 0 && isfinite(wr) && wr > 0 &&
            snr2 >= snr_floor2
        val[I] = angle(r)
        w[I] = ok ? snr2 : zero(T)
        mask[I] = ok
    end
    return (; val, w, mask, nodes)
end

# Weighted circular mean of each cell's source term from its residual
# `y − (φ_na − φ_nb)` over the APs where the cell passed the gate and both nodes
# solved. `obs` must carry the uncorrected phases, since `x` is defined relative
# to the observation itself. Returns the largest move.
function _update_source_terms!(x, obs, phase)
    (; val, w, mask, nodes) = obs
    acc = zeros(Complex{eltype(x)}, DimensionalData.dims(x))
    for ap in axes(val, Ti), I in DimensionalData.DimIndices(nodes)
        c = (I..., Ti(ap))
        mask[c...] || continue
        (a, na), (b, nb) = nodes[I...]
        pa = phase[a, na, ap]
        pb = phase[b, nb, ap]
        (isfinite(pa) && isfinite(pb)) || continue
        acc[I...] += w[c...] * cis(val[c...] - (pa - pb))
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

# The observations with each cell's source term `x` removed, gated to the cells
# whose source term is identifiable (`keep`): a (baseline, product) seen at a
# single AP is absorbed exactly by its own source term, so it constrains no
# node phase and only inflates the apparent coverage.
_source_corrected(obs, x, keep) = (;
    val = DimensionalData.broadcast_dims(-, obs.val, x),
    obs.w,
    mask = DimensionalData.broadcast_dims(&, obs.mask, keep),
    obs.nodes,
)

# Joint adhoc solve: one multivariate OU Kalman filter + RTS smoother over the
# whole station-phase vector, observing the baseline phase differences directly
# under a per-station temporal OU prior, so closure and denoising happen in one
# recursive estimator. This conditions better at low SNR than solving each AP
# and smoothing afterwards. Mutates `phase[:, 1, :]` in place; the caller's
# detrend and node→feed expansion run afterwards.
#
# State = `nant` station phases. `obs` arrives with each cell's source term
# already removed, so a cell is a pure node difference and the filter needs no
# augmented source dimension. The OU prior pins the unobservable common mode
# near 0; it is re-gauged to the anchor afterwards for pipeline consistency.
# Requires one node per station.
function _solve_gp_joint!(
        phase, track_w, obs, anchor::Integer, sm::JointOUSmoother,
    )
    nant = size(phase, Ant)
    times = parent(lookup(phase, Ti))
    # Compute type flows from the data, not from the smoother's field types.
    T = float(promote_type(eltype(phase), eltype(track_w), eltype(times)))

    # Seed (ref-gauged, unwrapped) per-station track from the per-AP solve.
    θseed = [T(phase[i, 1, ap]) for i in axes(phase, 1), ap in axes(phase, 3)]

    # Per-station centering mean (weighted): the OU prior reverts to 0, so a nonzero-
    # mean track would be biased. Removed again by the downstream detrend (so its
    # exact value is inert to the output — it only de-biases the solve).
    mθ = [_weighted_mean_finite(view(θseed, i, :), view(track_w, i, 1, :)) for i in axes(θseed, 1)]

    # A station with no gated row anywhere in the scan has an all-zero column in every
    # H below, so the filter never updates it: it holds the OU prior mean for the whole
    # solve and the re-gauge turns that into minus the anchor's common mode — the same
    # information-free track for every such station. They are excluded from the hyper
    # fit and from the write-back.
    seen = [any(>(0), view(track_w, i, 1, :)) for i in axes(track_w, 1)]

    # Per-station OU hypers from the (centered) seed track. The reference track is
    # structurally 0 (per-AP gauge) so it carries no variance info — give it and any
    # degenerate/uncovered station the ensemble-median dynamics, so the joint solve
    # treats every station as a real OU process (none pinned during the solve).
    τ_lo, τ_hi = _ou_tau_bounds(times)
    τv = fill(T(sm.coherence_time), nant)
    σ2v = fill(T(1.0e-2), nant)
    welldet = [seen[i] && i != anchor for i in eachindex(seen)]
    for i in eachindex(welldet)
        welldet[i] || continue
        _, _, τv[i], σ2v[i] = _track_ou_hypers(
            view(θseed, i, :), view(track_w, i, 1, :), times;
            τ0 = sm.coherence_time, τ_lo = τ_lo, τ_hi = τ_hi, fit = sm.fit_hypers,
        )
    end
    σ2_med = let g = [σ2v[i] for i in eachindex(σ2v, welldet) if welldet[i]]
        isempty(g) ? T(1.0e-2) : median(g)
    end
    τ_med = let g = [τv[i] for i in eachindex(τv, welldet) if welldet[i]]
        isempty(g) ? T(sm.coherence_time) : median(g)
    end
    for i in eachindex(τv, σ2v, welldet)
        welldet[i] || (τv[i] = τ_med; σ2v[i] = σ2_med)
    end

    # Per-cell measurement variance (1/snr²); 0 marks a gated-out cell, which the
    # filter skips. With one node per station, a cell's state pair is its stations.
    ends = map((((a, _), (b, _)),) -> (a, b), obs.nodes)
    rs = [
        map((m, wi) -> m ? inv(T(wi)) : zero(T), view(obs.mask, Ti(ap)), view(obs.w, Ti(ap)))
            for ap in axes(obs.w, Ti)
    ]
    ys = [similar(r) for r in rs]

    # Current full (uncentered) estimate, seeded from the per-AP solve.
    θf = [isfinite(θseed[i, ap]) ? θseed[i, ap] : mθ[i] for i in axes(θseed, 1), ap in axes(θseed, 2)]
    twoπ = 2 * T(π)

    # Iterated Kalman: rewrap each raw observation toward the current joint model,
    # then re-run the forward/backward smoother (relinearizing the ±2π branch).
    for _ in 1:max(sm.options.phase_rewrap_iters, 1)
        for ap in eachindex(ys)
            y = ys[ap]
            v = view(obs.val, Ti(ap))
            for j in eachindex(ends, y, v)
                a, b = ends[j]
                model = θf[a, ap] - θf[b, ap]
                centre = mθ[a] - mθ[b]
                raw = T(v[j])
                y[j] = raw + twoπ * round((model - raw) / twoπ) - centre
            end
        end
        xf, Pf, xp, Pp, avecs, _ = kalman_ou_mv_filter(ends, ys, rs, times; τ = τv, σ2 = σ2v)
        xs, _ = rts_smooth_mv(xf, Pf, xp, Pp, avecs)
        for ap in axes(θf, 2)
            for i in axes(θf, 1)
                θf[i, ap] = xs[i, ap] + mθ[i]
            end
        end
    end

    # Re-gauge to the anchor (its phase → 0 per AP), matching the per-AP path.
    for ap in axes(θf, 2)
        θref = θf[anchor, ap]
        isfinite(θref) || continue
        for i in axes(θf, 1)
            θf[i, ap] -= θref
        end
    end

    # Write back the joint track; internal gaps stay interpolated by the filter. An
    # unseen station keeps the per-AP solve's NaN, so it is never fabricated into θ
    # (`adhoc_scan!` gates its θ write on `isfinite`, not on `covered`).
    for i in eachindex(seen)
        seen[i] || continue
        phase[i, 1, :] .= view(θf, i, :)
    end
    return phase
end

# One full per-AP sweep: solve every AP's station phases from `obs` (with the
# anchor-gauged warm-start snapshot carrying 2π-branch continuity across APs),
# restitch anchor-dropout APs, and unwrap each (station, node) track. Shared by
# the seed alternation passes and the complex-domain refinement passes, which
# differ only in how their observations were built.
function _solve_ap_sweep!(
        phase, covered, track_w, obs, nant::Integer, anchor::Integer,
        rewrap::Integer, max_stale::Integer,
    )
    (; val, w, mask, nodes) = obs
    fill!(phase, convert(eltype(phase), NaN))
    fill!(covered, false)
    fill!(track_w, zero(eltype(track_w)))
    prev_phase = fill(convert(eltype(phase), NaN), nant, 2)  # cells no anchor-present AP has covered yet
    prev_age = zeros(Int, nant, 2)    # APs since a cell was last refreshed (staleness)
    for ap in axes(val, Ti)
        v, wk, mk = view(val, Ti(ap)), view(w, Ti(ap)), view(mask, Ti(ap))
        # The anchor has data this AP iff some observation touches it (⇒ the solve
        # is pinned at anchor=0). Seed only then, and only with fresh cells in the
        # anchor gauge.
        ref_here = false
        for I in eachindex(wk, mk, nodes)
            mk[I] || continue
            (a, na), (b, nb) = nodes[I]
            track_w[a, na, ap] += wk[I]
            track_w[b, nb, ap] += wk[I]
            ref_here |= a == anchor || b == anchor
        end
        seed = nothing
        if ref_here
            seed = fill(convert(eltype(phase), NaN), nant, 2)
            for a in axes(seed, 1), n in axes(seed, 2)
                (isfinite(prev_phase[a, n]) && prev_age[a, n] <= max_stale) &&
                    (seed[a, n] = prev_phase[a, n])
            end
        end
        # No usable warm-start snapshot (first AP, all cells stale, or an
        # anchor-dropout AP): seed from the circular phasor solve instead of
        # trusting the tree-initialized linear solve's 2π branch — see
        # `_circular_ap_seed`.
        if seed === nothing || !any(isfinite, seed)
            seed = _circular_ap_seed(v, wk, mk, nodes, nant, anchor)
        end
        ph, cov = view(phase, :, :, ap), view(covered, :, :, ap)
        _solve_observable!(ph, cov, v, wk, mk, nodes, PinAntenna(anchor); rewrap, seed_phase = seed)
        # Refresh the anchor-gauged snapshot only from anchor-present APs (keep the
        # last known value for a station absent this AP, so a brief dropout does not
        # reset the branch); age every cell and zero the ones refreshed here.
        prev_age .+= 1
        if ref_here
            for a in axes(ph, 1), n in axes(ph, 2)
                if cov[a, n] && isfinite(ph[a, n])
                    prev_phase[a, n] = ph[a, n]
                    prev_age[a, n] = 0
                end
            end
        end
    end

    # Restitch the per-AP gauge when the anchor drops out. Each per-AP solve pins
    # the anchor; in APs where it has no data the solve falls back to a different
    # pin node, so that AP's whole solution is offset by an arbitrary, non-2π
    # constant, which would otherwise inject a spurious common-mode jump into
    # every station's track.
    _restitch_refant_gauge!(phase, covered, track_w, anchor)

    # Unwrap each (station, node) track across APs (per-AP solves share the ref
    # gauge, so a track is continuous up to ±2π steps the unwrap removes).
    for a in axes(phase, 1), n in axes(phase, 2)
        any(isfinite, @view phase[a, n, :]) || continue
        phase[a, n, :] .= unwrap_phase_track(phase[a, n, :]; weights = track_w[a, n, :])
    end
    return phase
end

# Per-(baseline, product) complex source term for the Gauss–Newton refinement:
# the inverse-variance mean of the model-derotated per-AP visibilities over the
# whole scan, so its SNR is the track's rather than one AP's, and its |s̄|² is
# the signal power the linearized observations are weighted by. Also returns
# how many APs informed each term (its identifiability count).
function _complex_source_means(rbar, wbar, phase, nodes)
    cells = DimensionalData.dims(nodes)
    sbar = zeros(eltype(rbar), cells)
    nrm = zeros(real(eltype(rbar)), cells)
    napu = zeros(Int, cells)
    for I in DimensionalData.DimIndices(nodes)
        _solvable(nodes[I...]) || continue
        (a, na), (b, nb) = nodes[I...]
        for ap in axes(rbar, Ti)
            w = wbar[I..., Ti(ap)]
            r = rbar[I..., Ti(ap)]
            (isfinite(r) && isfinite(w) && w > 0) || continue
            dphi = phase[a, na, ap] - phase[b, nb, ap]
            isfinite(dphi) || continue
            sbar[I...] += r * cis(-dphi)     # r = Σ w·V ⇒ this is Σ w·V·e^{-iΔφ̂}
            nrm[I...] += w
            napu[I...] += 1
        end
        nrm[I...] > 0 && (sbar[I...] /= nrm[I...])
    end
    return sbar, napu
end

# Linearized (Gauss–Newton) observations in the complex domain. Around the
# current tracks, `V̄·conj(s̄)e^{-iΔφ̂} ≈ |s̄|²(1 + i(Δφ − Δφ̂)) + n·conj(s̄)`, so
#
#     val  = Δφ̂ + Im(V̄·conj(s̄)e^{-iΔφ̂}) / |s̄|²
#     info = 2|s̄|² / σ²        (σ² the complex noise power of V̄, data-driven)
#
# is a linear measurement of Δφ with Gaussian noise at any per-AP SNR. Every AP
# with data and a track therefore enters ungated, carrying its honest weight;
# a per-AP extracted phase would instead collapse nonlinearly below SNR ≈ 1.
# The source term is already divided out through `conj(s̄)`.
function _linearized_obs(rbar, wbar, nodes, noise2, phase, sbar)
    T = real(eltype(rbar))
    val = fill!(similar(rbar, T), T(NaN))
    w = fill!(similar(rbar, T), zero(T))
    mask = fill!(similar(rbar, Bool), false)
    for ap in axes(rbar, Ti), I in DimensionalData.DimIndices(nodes)
        _solvable(nodes[I...]) || continue
        (a, na), (b, nb) = nodes[I...]
        c = (I..., Ti(ap))
        wr = wbar[c...]
        r = rbar[c...]
        (isfinite(r) && isfinite(wr) && wr > 0) || continue
        dphi = phase[a, na, ap] - phase[b, nb, ap]
        isfinite(dphi) || continue
        s = sbar[I...]
        s2 = abs2(s)
        (isfinite(s2) && s2 > 0) || continue
        z = imag((r / wr) * conj(s) * cis(-dphi)) / s2
        isfinite(z) || continue
        n2 = noise2[I...]
        val[c...] = dphi + z
        # Fall back to the weight column's noise claim when the track is too
        # short to estimate its own (mirrors `_adhoc_obs`'s fallback).
        w[c...] = isfinite(n2) && n2 > 0 ? 2 * s2 / n2 : s2 * wr
        mask[c...] = true
    end
    return (; val, w, mask, nodes)
end

"""
    solve_adhoc_phasing(rbar, wbar, stations; gauge, smoother, tying) -> DimStack

Solve globally-closing adhoc phases from coherently frequency-averaged
residual baseline visibilities under the model

    y[station pair, feed pair, ap] = φ_na(ap) − φ_nb(ap) + x[station pair, feed pair],

with the station phases `φ` on parameter nodes and one free source term `x`
per (station pair, feed pair), constant over the scan. The free source term
carries the source's EVPA, D-terms, and closure phase, so the station tracks
are unbiased by source structure.

`rbar` is `Σ_chan w·V_residual` and `wbar` is `Σ_chan w`, both `DimArray`s
over `StationPair`, `FeedPair` and `Ti` in any storage order, with the same
lookups: station pairs labeled by station names, feed pairs by feed-index
pairs, and `Ti` by the AP epochs in seconds. The coherent SNR² is
`|rbar|²/wbar`. `stations` names the stations; each one's position in it is
its station index.

Returns a `DimStack`: `:phase` (`Ant(stations) × Feed × Ti`) is the
per-(station, feed) adhoc phase in radians, `NaN` where unsolved;
`:covered` marks the solved cells; `:source` (`StationPair × FeedPair`) is
the fitted source phase, `NaN` where unidentifiable.

`gauge` sets the per-AP convention: `PinAntenna` holds its reference's phase
at 0, `ZeroSumPhase` centers each AP on zero mean.

`tying` is the adhoc component's [`AbstractFeedTying`](@ref). `PerFeed()`
(the default) solves an independent track per feed; `SharedFeeds()` solves
one feed-common phase, constrained by all four correlation products and
contributing zero inter-feed phase. Under `PerFeed`, cross-hand rows join the two feed blocks
into one connected component with a single gauge freedom, pinned at the
reference's feed-1 node, so the reference's inter-feed phase stays in the
solution.

`smoother` is an [`AbstractAdhocSmoother`](@ref) selecting how each track is
smoothed after the solve; its [`AdhocOptions`](@ref) set the solve. The
`(φ, x)` blocks are fit by alternating minimization until no source term moves
by more than `options.source_tol` radians or `options.source_iters` passes;
`source_iters = 1` fixes `x = 0`.
"""
function solve_adhoc_phasing(
        rbar::DimensionalData.AbstractDimArray{<:Complex, 3},
        wbar::DimensionalData.AbstractDimArray{<:Real, 3},
        stations::AbstractVector;
        gauge::AbstractGauge = PinAntenna(1),
        smoother::AbstractAdhocSmoother = SavitzkyGolaySmoother(),
        tying::AbstractFeedTying = PerFeed(),
    )
    cell_dims = (StationPair, FeedPair, Ti)
    all(d -> DimensionalData.hasdim(rbar, d), cell_dims) || throw(
        ArgumentError(
            "`rbar` must be over StationPair, FeedPair and Ti; got " *
                join(DimensionalData.name(DimensionalData.dims(rbar)), ", "),
        ),
    )
    DimensionalData.comparedims(
        DimensionalData.dims(wbar, DimensionalData.dims(rbar)), DimensionalData.dims(rbar); val = true,
    )
    # Positions along `Ti` and in `stations` index the solve's 1-based arrays.
    Base.require_one_based_indexing(rbar, wbar, stations)
    allunique(stations) || throw(
        ArgumentError(
            "station names must be unique; repeated: " *
                join((n for n in unique(stations) if count(==(n), stations) > 1), ", "),
        ),
    )
    _requires_single_node(smoother) && _feed_node(tying, 1) != _feed_node(tying, 2) && error(
        "$(typeof(smoother)) requires one phase node per station (its Kalman state " *
            "is one dimension per station), but the adhoc component ties feeds as " *
            "$(typeof(tying)). Use OUSmoother for an independent per-feed track.",
    )
    T = something(smoother.options.eltype, float(real(eltype(rbar))))
    r = Complex{T}.(permutedims(rbar, cell_dims))
    w = T.(permutedims(wbar, cell_dims))
    return _solve_adhoc_phasing(r, w, _cell_nodes(r, stations, tying), stations, gauge, smoother, tying)
end

# Behind a function barrier: the working type comes from a runtime option.
# `rbar`/`wbar` are stored `(StationPair, FeedPair, Ti)`, as is every derived array.
function _solve_adhoc_phasing(rbar, wbar, nodes, stations, gauge, smoother, tying)
    T = real(eltype(rbar))
    nant = length(stations)
    nap = size(rbar, Ti)
    times = parent(lookup(rbar, Ti))

    # Solved on the (station, feed node) graph; expanded back onto the feed axis at the end.
    node_axes = (Ant(stations), FeedNode(1:2), DimensionalData.dims(rbar, Ti))
    phase = fill(T(NaN), node_axes)
    covered = DimArray(falses(nant, 2, nap), node_axes)
    # Track per-(station, feed node) coherent weight for smoothing / detrend.
    track_w = zeros(T, node_axes)

    # Per-(baseline, product) noise of the coherent track, estimated data-driven
    # from its AP-to-AP scatter — see `_track_noise2`. The per-AP coherent SNR² is
    # then |V̄_ap|² / noise², which is scale-invariant in the weight column: the
    # naive `|rbar|²/wbar` is only a true SNR when weight is calibrated inverse-
    # variance, and on raw correlator output (uncalibrated/uniform weights, common
    # in FITS-IDI) it is mis-scaled by an arbitrary factor, silently dropping every
    # row at any fixed `snr_floor` and killing the whole adhoc stage. For calibrated
    # weights `noise² → 1/wbar`, so this reduces to `|rbar|²/wbar` exactly.
    noise2 = _cell_noise2(rbar, wbar, Ti)

    # The SNR gate does not depend on the source terms, so the gated observations
    # are built once and reused by every alternation pass (and by the joint smoother).
    raw = _adhoc_obs(rbar, wbar, nodes, noise2, smoother.options.snr_floor^2)

    # Source terms, one per (station pair, feed pair). `source_iters == 1` never
    # updates them, so the model reduces exactly to a pure station-difference
    # solve with no source term.
    x = zeros(T, DimensionalData.dims(nodes))
    keep = fill!(similar(nodes, Bool), true)
    fit_source = smoother.options.source_iters >= 2
    if fit_source
        # A (baseline, product) seen at a single AP is absorbed exactly by its own
        # source term: it constrains no node phase, and admitting it would only
        # inflate the apparent coverage. Two APs is the identifiability threshold,
        # not a tuning choice.
        keep .= dropdims(sum(raw.mask; dims = Ti); dims = Ti) .>= 2
        # Seed each source term from its own observations. The gauge the per-track
        # demean imposes leaves each node track with ~zero scan mean, so a row's
        # scan-mean is its source term to first order. Seeding from 0 instead would
        # make the first solve fit rows a full source phase away from their model,
        # which for a source phase near ±π locks the wrong 2π branch — one the
        # later passes inherit through the warm start and cannot leave.
        _update_source_terms!(x, raw, zeros(T, nant, 2, nap))
    end
    obs = _source_corrected(raw, x, keep)

    # Effective per-scan anchor station: the gauge's preferred station when it
    # observes in this scan, else the best-covered station by total gated
    # weight. Everything gauge-related below — the per-AP pin, the warm-start
    # seed condition, the gauge restitch, the joint solve's re-gauge — keys on
    # the anchor being present, so it must not key on the literal reference:
    # a scan that never sees the reference would disable all of it, which is
    # common in multi-subarray tracks. The anchor choice is inert to the applied
    # correction, a per-AP common mode cancelling on every baseline; it exists so
    # the per-station tracks are temporally consistent, and so smoothable.
    anchor = let wtot = zeros(T, nant)
        for ap in axes(obs.w, Ti)
            wk, mk = view(obs.w, Ti(ap)), view(obs.mask, Ti(ap))
            for I in eachindex(wk, mk, nodes)
                mk[I] || continue
                (a, _), (b, _) = nodes[I]
                wtot[a] += wk[I]
                wtot[b] += wk[I]
            end
        end
        # A ranked gauge walks its references before falling back to the
        # best-observed station, so a dropout costs the next choice, not an
        # arbitrary hop.
        cand = gauge_station_order(gauge, nant)
        j = findfirst(a -> 1 <= a <= nant && wtot[a] > 0, cand)
        j !== nothing ? Int(cand[j]) : (all(iszero, wtot) ? 1 : argmax(wtot))
    end

    # Carry solved node phases forward as a temporal warm-start for the next AP's
    # 2π-branch selection, so weakly-constrained stations do not flip branch per
    # AP. The seed is an anchor-gauged snapshot: refreshed only from APs where
    # the anchor has data, so every stored cell is in the anchor = 0 gauge, and
    # applied only when the current AP also has the anchor, so the per-AP
    # spanning-tree seed it partially overrides is anchor = 0 too. Mixing an
    # anchor = 0 seed with a differently-anchored tree seed would mis-pick the 2π
    # branch on a seeded↔tree boundary edge. The snapshot is rebuilt from scratch
    # by each alternation pass, so a pass never inherits a branch chosen against
    # a stale set of source terms.
    #
    # A seed is trusted only while the atmosphere cannot yet have drifted past
    # ±π, beyond which the old branch is no longer a safe guide; that takes a few
    # coherence times. `times` is in seconds and `T_AP` is its median spacing.
    dts0 = filter(>(0), diff(sort(times)))
    t_ap0 = isempty(dts0) ? 1.0 : float(median(dts0))
    max_stale = max(10, ceil(Int, 3 * _adhoc_coherence_time(smoother) / t_ap0))
    # Alternating minimization of the joint (node phases, source terms) least
    # squares: each pass re-solves every AP with the current source terms removed,
    # then re-estimates each source term from the residual it leaves. Both blocks
    # are conditionally linear, so the alternation descends a convex quadratic.
    # With `source_iters == 1` the source terms are never fitted and stay at 0, so
    # the model reduces to a pure station-difference solve.
    for iter in 1:max(smoother.options.source_iters, 1)
        _solve_ap_sweep!(
            phase, covered, track_w, obs, nant, anchor,
            smoother.options.phase_rewrap_iters, max_stale,
        )
        (fit_source && iter < smoother.options.source_iters) || break
        moved = _update_source_terms!(x, raw, phase)
        obs = _source_corrected(raw, x, keep)
        moved <= smoother.options.source_tol && break
    end

    # Smooth: dispatch on the smoother type. Per-track smoothers loop the (station,
    # node) tracks (`window = :auto` sets a per-station Savitzky–Golay window from the
    # EHT-HOPS `T_dof`, Eqs 21–22); the joint solve runs one multivariate OU Kalman
    # over all station phases (re-gauged to the anchor, matching the per-AP path);
    # `NoSmoothing` is a no-op. See `apply_adhoc!`.
    apply_adhoc!(smoother, phase, track_w; obs, anchor)

    # Gauss–Newton refinement in the complex domain. Everything above is the
    # seed: the phase-extraction solve's spanning-tree unwrap and warm starts
    # settle the global 2π branch, which no local linearization can, and the SNR
    # gate is confined to that seeding role. Each pass here re-derives the
    # per-(baseline, product) complex source terms from the whole scan,
    # linearizes every AP's residual around the current tracks
    # (`_linearized_obs`), and re-solves and re-smooths on those observations,
    # every AP entering at its exact first-order information, ungated. With the
    # smoothing pass inside, each iteration is an extended-Kalman/RTS step and
    # the loop is Gauss–Newton on the MAP objective of the complex data; the
    # innovations start on the seed's branch, so the linearization stays inside
    # ±π by construction.
    sbar_ref = nothing
    nap_ref = nothing
    for _ in 1:max(smoother.options.complex_iters, 0)
        sbar, napu = _complex_source_means(rbar, wbar, phase, nodes)
        sbar_ref = sbar
        nap_ref = napu
        lin = _linearized_obs(rbar, wbar, nodes, noise2, phase, sbar)
        _solve_ap_sweep!(
            phase, covered, track_w, lin, nant, anchor,
            smoother.options.phase_rewrap_iters, max_stale,
        )
        apply_adhoc!(smoother, phase, track_w; obs = lin, anchor)
    end

    # Demean per track: remove the per-scan mean so adhoc does not alias the Stage-B
    # constant phase — this also fixes the (per-station constant ↔ source term)
    # gauge. The slope (residual rate) is intentionally kept — see `_detrend_track!`.
    if smoother.options.detrend
        for a in axes(phase, 1), n in axes(phase, 2)
            _detrend_track!(@view(phase[a, n, :]), @view(track_w[a, n, :]))
        end
    end

    # Impose the gauge last. Detrending removes each track's temporal mean, which
    # offsets every AP by the same constant, so applying the gauge after it leaves
    # the per-AP convention exact and the detrended tracks zero-mean up to one
    # global constant — and that constant is a per-AP common mode, which cancels on
    # every baseline. The two conventions are compatible only up to that constant;
    # this order is what makes the gauge the exact one.
    _apply_ap_gauge!(phase, covered, gauge, 2)

    # Expand the (station, feed node) solution onto the feed axis the caller
    # indexes: feeds sharing a node get identical tracks (so a `SharedFeeds` adhoc
    # contributes exactly zero inter-feed phase), and a feed the component does not
    # parameterize (node 0) stays NaN/uncovered.
    feed_axes = (Ant(stations), Feed(1:2), DimensionalData.dims(rbar, Ti))
    phase_out = fill(T(NaN), feed_axes)
    covered_out = DimArray(falses(nant, 2, nap), feed_axes)
    for f in lookup(phase_out, Feed)
        n = _feed_node(tying, f)
        n == 0 && continue
        view(phase_out, Feed(At(f))) .= view(phase, FeedNode(At(n)))
        view(covered_out, Feed(At(f))) .= view(covered, FeedNode(At(n)))
    end

    # The refinement's complex source means supersede the seed's phase-only
    # alternation estimates: same per-(baseline, product) constant, measured
    # against the final tracks over every usable AP. Two APs stays the
    # identifiability threshold.
    if sbar_ref !== nothing
        for I in eachindex(x, keep, sbar_ref, nap_ref)
            keep[I] = nap_ref[I] >= 2 && abs2(sbar_ref[I]) > 0
            keep[I] && (x[I] = angle(sbar_ref[I]))
        end
    end

    # Source terms as measured, `NaN` where too poorly sampled to identify.
    source = map((xi, k) -> k ? xi : T(NaN), x, keep)

    return DimensionalData.DimStack((phase = phase_out, covered = covered_out, source = source))
end

# ── Per-scan pipeline entry ───────────────────────────────────────────────────

"""
    adhoc_scan!(θ, group::XRadio.ProcessingSet, geom::DataGeometry, adhoc_plan,
                adhoc, gauge; executor = DynamicScheduler()) -> θ

The per-integration atmospheric-phase (adhoc) solve of one scan group — the
"caller" the module docstring above refers to. On data already gain-corrected
by the pipeline's corrections: sum the per-(station pair, feed pair, AP)
inverse-variance residual over every band (`weighted_sums`), solve the
globally-closing per-AP station phase through the pluggable `adhoc` smoother
(`solve_adhoc_phasing`), and write this scan's `PerIntegration` θ slots
(disjoint per scan — concurrent groups may solve in parallel). The feed tying
comes from `adhoc_plan`, so the number of phase nodes per station is the
model's choice and needs no separate argument.
"""
function adhoc_scan!(
        θ, group::XRadio.ProcessingSet, geom::DataGeometry, adhoc_plan, adhoc, gauge;
        executor = DynamicScheduler(),
    )
    (; rbar, wbar, ti) = _ap_sums(group, geom; executor)
    as = solve_adhoc_phasing(rbar, wbar, geom.stations; gauge, smoother = adhoc, tying = adhoc_plan.tying)
    adhoc_leaf = _component_leaf(adhoc_plan, θ)
    for gti in ti
        tseg = adhoc_plan.tseg_id[gti]
        at_t = view(as.phase, Ti(At(geom.times[gti])))
        for (ant, name) in pairs(geom.stations), feed in lookup(at_t, Feed)
            val = at_t[Ant(At(name)), Feed(At(feed))]
            isfinite(val) || continue
            node = _feed_node(adhoc_plan.tying, feed)
            node == 0 && continue
            adhoc_leaf[1, node, 1, tseg, ant] = val
        end
    end
    return θ
end

# Put each AP on the gauge's own convention, over a station set that does not
# move between APs. The per-AP common mode is unobservable — it cancels on every
# baseline — so this changes how the tracks read, never the applied correction.
# That is why the set must be fixed: a sum taken over whatever stations happen
# to be covered shifts frame whenever coverage flickers, putting steps into
# every track for a quantity that carries no information. Summing over the
# stations covered in every AP keeps one frame for the whole scan.
#
# A pinned gauge needs nothing here: the per-AP solves already pin the anchor,
# and `_restitch_refant_gauge!` carries that frame across the APs where it
# drops out.
_apply_ap_gauge!(phase, covered, ::AbstractGauge, nnode::Integer) = phase

function _apply_ap_gauge!(phase, covered, gauge::ZeroSumPhase, nnode::Integer)
    nant, _, nap = size(phase)
    # One constant per AP, across both feed nodes. Cross-hand rows join the two
    # feeds into a single connected component carrying a single additive freedom,
    # so a separate constant per feed would invent a second one and shift every
    # cross-hand difference `φ_{a,1} − φ_{b,2}` by the gap between them, breaking
    # the reconstruction the gauge must leave untouched. Subtracting the same
    # constant from every cell cancels in every baseline difference, parallel and
    # cross alike.
    #
    # The summed cells are those covered in every AP: a sum over whatever happens
    # to be covered moves frame with coverage, putting steps into every track for
    # a quantity that carries no information.
    #
    # `weights` describes the station-solve constraint row, whose nodes are not
    # these cells, so the per-AP frame is unweighted.
    cells = [
        (a, n) for a in axes(covered, 1) for n in 1:nnode
            if all(covered[a, n, ap] for ap in axes(covered, 3)) &&
            (gauge.antennas === nothing || a in gauge.antennas)
    ]
    isempty(cells) && return phase
    for ap in axes(phase, 3)
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
        for a in axes(phase, 1), n in 1:nnode
            isfinite(phase[a, n, ap]) && (phase[a, n, ap] -= d)
        end
    end
    return phase
end

# Re-reference anchor-absent APs to the trusted frame. Where the anchor is
# solved, that AP's own solve already pins it and the running anchor is
# refreshed from it, so real drift propagates. Where it is absent, the AP
# carries an arbitrary global offset δ, estimated as the weighted circular mean
# over the cells common to this AP and the anchor and subtracted from every
# solved cell. This is a no-op when the anchor is present in every AP, and it
# removes only a single global per-AP constant: per-(station, feed) means and
# slopes are left to `_detrend_track!`. Leading APs with no trusted anchor yet
# are left untouched, and a multi-component AP keeps one global δ dominated by
# the largest overlap, per-island offsets being a gauge freedom.
function _restitch_refant_gauge!(phase, covered, track_w, ref_station::Integer)
    T = eltype(phase)
    anchor = fill(T(NaN), size(phase, 1), size(phase, 2))
    have_anchor = false
    for ap in axes(phase, 3)
        ref_present = covered[ref_station, 1, ap] || covered[ref_station, 2, ap]
        if !ref_present && have_anchor
            # Register per feed. With cross hands the two feeds share a component
            # but carry two gauge freedoms (the overall phase pin and the feed-2
            # EVPA pin); when the anchor drops out both fall back to a different
            # antenna, shifting each feed by its own constant. The per-feed
            # convention (each feed gauged relative to the anchor's feed) matches the
            # rest of the adhoc solve, so a separate δ per feed restores it.
            for f in axes(phase, 2)
                num_s = zero(T)
                num_c = zero(T)
                wsum = zero(T)
                for a in axes(phase, 1)
                    (covered[a, f, ap] && isfinite(anchor[a, f]) && isfinite(phase[a, f, ap])) || continue
                    d = phase[a, f, ap] - anchor[a, f]
                    wk = (isfinite(track_w[a, f, ap]) && track_w[a, f, ap] > 0) ? T(track_w[a, f, ap]) : one(T)
                    num_s += wk * sin(d)
                    num_c += wk * cos(d)
                    wsum += wk
                end
                if wsum > 0 && (num_s != 0 || num_c != 0)
                    δ = atan(num_s, num_c)
                    for a in axes(phase, 1)
                        covered[a, f, ap] && (phase[a, f, ap] -= δ)
                    end
                end
            end
        end
        # Refresh the anchor from this AP's (now registered) solved cells.
        if ref_present || have_anchor
            for a in axes(phase, 1), f in axes(phase, 2)
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
    # The solve is a dense 1-based matrix problem, so the track is read 1-based.
    Base.require_one_based_indexing(y, w)
    T = float(eltype(y))
    n = length(y)
    n == 0 && return collect(T, y)
    wk = [(isfinite(y[k]) && isfinite(w[k]) && w[k] > 0) ? T(w[k]) : zero(T) for k in 1:n]
    yk = [wk[k] > 0 ? T(y[k]) : zero(T) for k in 1:n]
    # Guard against an all-unconstrained component (no data, λ = 0).
    all(iszero, wk) && iszero(λ) && return collect(T, y)
    D = zeros(T, n - 1, n)
    for k in axes(D, 1)
        D[k, k] = one(T); D[k, k + 1] = -one(T)
    end
    R = sqrt(T(λ)) .* D
    return weighted_regularized_least_squares(Matrix{T}(I, n, n), yk, wk, R)
end

# Remove the weighted mean of a track. Only the mean: the constant is degenerate
# with the fringe step's per-scan `ConstantTerm`, which owns it, while the slope
# is the residual fringe rate the fringe step's `Rate` left, which the adhoc
# track exists to correct.
function _detrend_track!(track::AbstractVector, w::AbstractVector)
    T = eltype(track)
    idx = [i for i in eachindex(track) if isfinite(track[i])]
    length(idx) >= 1 || return track
    ws = [(isfinite(w[i]) && w[i] > 0) ? T(w[i]) : one(T) for i in idx]
    m = sum(ws .* track[idx]) / sum(ws)
    for i in idx
        track[i] -= m
    end
    return track
end
