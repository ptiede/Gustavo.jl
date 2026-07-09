# ── Prior / regularizer layer (Stage 2, Milestone 7) ─────────────────────────
#
# The forward-model MAP objective is `f(θ) = Σ_leaf loglik + logprior`. This file
# adds the `logprior` term: a per-COMPONENT regularizer, attached to a model
# component BY NAME (like the gauge/scaling), evaluated ONCE by the driver (not
# per leaf) and added to the distributed likelihood. The gradient is ANALYTIC
# (the priors here are Gaussian, so grad = −Q·x in O(n)); it is added to the
# Enzyme likelihood gradient in `logdensity_and_gradient`.
#
# v1 (OU-first): the stochastic-time `OUPrior` on a station's per-integration
# adhoc-phase track. `IIDGaussianPrior` and `SmoothnessPrior` (2nd-difference along
# frequency) are the follow-ups.

import ..Fringe
using ..Fringe: ou_step

# ── Prior interface ───────────────────────────────────────────────────────────

"A per-component regularizer contributing a `logprior` term to the MAP objective."
abstract type AbstractPrior end

"No prior on a component (the default — the objective stays pure likelihood)."
struct NoPrior <: AbstractPrior end

"""
    OUPrior(; τ, σ2)

Stochastic-time Ornstein–Uhlenbeck (Matérn-1/2) Gaussian-Markov prior on a
station's per-integration phase track: `K(Δt) = σ2·exp(−|Δt|/τ)`, `τ` the
coherence time (SECONDS), `σ2` the stationary variance (rad²). The prior sees the
RAW phase increments (no wrapping — periodicity lives in the `cis` forward map),
and contributes `−½ xᵀQ x` with a tridiagonal precision `Q` built in O(n) from
[`ou_step`](@ref) — no dense covariance, no `SparseArrays` (Reactant-safe). The
track runs along the component's TIME-segment axis (one OU chain per
`(frequency segment, feed block, antenna)`); irregular Δt (scan gaps) are handled
naturally. Hypers are fixed in v1 (fit them from a warm-start track with
[`fit_ou_hypers`](@ref) later).
"""
struct OUPrior{T} <: AbstractPrior
    τ::T
    σ2::T
end
OUPrior(; τ::Real, σ2::Real) = OUPrior(promote(τ, σ2)...)

"""
    ComponentPriors(; name = prior, …)

Attach priors to model components by NAME (the Lux-style component keys, e.g.
`adhoc = OUPrior(τ = 30, σ2 = 1.0)`). Components absent from the set get
[`NoPrior`](@ref). The empty default `ComponentPriors()` is a no-op (pure
likelihood).
"""
struct ComponentPriors{NT <: NamedTuple}
    priors::NT
end
ComponentPriors(; kw...) = ComponentPriors((; kw...))

Base.isempty(cp::ComponentPriors) = isempty(cp.priors)

# ── OU precision + quadratic form (the numeric core, also the tested surface) ──

# Tridiagonal OU precision `Q` (as its diagonal `d` and first off-diagonal `e`)
# for a track sampled at `times` (same unit as τ), plus `logdet_extra =
# log σ2 + Σ_k log q_k` (so `logdet(Σ) = logdet_extra`, since a Markov chain's
# covariance determinant is the product of its conditional variances). The AR(1)
# with a_k = exp(−|Δt_k|/τ), q_k = σ2(1−a_k²), stationary variance σ2 realizes
# EXACTLY `K(Δt) = σ2·exp(−|Δt|/τ)`.
function _ou_precision(τ::T, σ2::T, times::AbstractVector{<:Real}) where {T}
    n = length(times)
    d = zeros(T, n)
    e = zeros(T, max(n - 1, 0))
    ld = log(σ2)
    n == 0 && return d, e, zero(T)
    d[1] += one(T) / σ2
    @inbounds for k in 2:n
        a, q = ou_step(τ, σ2, times[k] - times[k - 1])
        d[k] += one(T) / q
        d[k - 1] += a * a / q
        e[k - 1] += -a / q
        ld += log(q)
    end
    return d, e, ld
end

# Energy `½ xᵀQ x` for the tridiagonal `Q = (d, e)`, ACCUMULATING the logprior
# gradient `−Q x` into `g` (in place). `Q x` is matrix-free: `(Qx)_k = d_k x_k +
# e_{k-1} x_{k-1} + e_k x_{k+1}`.
function _ou_energy_and_grad!(g, d, e, x)
    n = length(x)
    E = zero(eltype(x))
    @inbounds for k in 1:n
        qxk = d[k] * x[k]
        k > 1 && (qxk += e[k - 1] * x[k - 1])
        k < n && (qxk += e[k] * x[k + 1])
        E += x[k] * qxk / 2
        g[k] += -qxk
    end
    return E
end

"""
    ou_logprior(x, times; τ, σ2) -> Real

The OU (Matérn-1/2) Gaussian log-density of a track `x` sampled at `times` (same
time unit as `τ`): `−½ xᵀQ x − ½(n·log2π + logdet Σ)`. Equal to
`logpdf(MvNormal(0, Σ), x)` with `Σ[i,j] = σ2·exp(−|tᵢ−tⱼ|/τ)`, computed in O(n).
"""
function ou_logprior(x::AbstractVector{T}, times::AbstractVector; τ::Real, σ2::Real) where {T}
    d, e, ld = _ou_precision(T(τ), T(σ2), times)
    g = zeros(T, length(x))
    E = _ou_energy_and_grad!(g, d, e, x)
    return -E - (length(x) * log(T(2π)) + ld) / 2
end

"""
    ou_logprior_grad(x, times; τ, σ2) -> Vector

Gradient `∂/∂x [ou_logprior] = −Q x` of [`ou_logprior`](@ref), in O(n).
"""
function ou_logprior_grad(x::AbstractVector{T}, times::AbstractVector; τ::Real, σ2::Real) where {T}
    d, e, _ = _ou_precision(T(τ), T(σ2), times)
    g = zeros(T, length(x))
    _ou_energy_and_grad!(g, d, e, x)
    return g
end

# ── Applying a prior to a component's parameter block ─────────────────────────

# Per time-segment representative time (SECONDS) for a component: the mean geometry
# time (hours → ×3600) of the samples mapped to each segment. One value per
# `ntseg`, so the OU chain's Δt matches the physical AP spacing (and scan gaps).
function _segment_times_sec(comp::SiteComponent, geom)
    n = comp.ntseg
    sums = zeros(eltype(geom.times), n)
    cnt = zeros(Int, n)
    @inbounds for (ti, s) in enumerate(comp.tseg_id)
        sums[s] += geom.times[ti]
        cnt[s] += 1
    end
    return [cnt[s] > 0 ? 3600.0 * sums[s] / cnt[s] : 0.0 for s in 1:n]
end

# NoPrior: contributes nothing.
_apply_prior!(::NoPrior, ::SiteComponent, A, gA, geom) = 0.0

# OUPrior: an independent OU chain along the time-segment axis for every
# (frequency segment, feed block, antenna) slice of the component's block
# `A[1, ntseg, nfseg, nfb, nant]`. Accumulates the logprior gradient into `gA`.
function _apply_prior!(pr::OUPrior, comp::SiteComponent, A, gA, geom)
    n = comp.ntseg
    n == 0 && return 0.0
    st = _segment_times_sec(comp, geom)
    d, e, ld = _ou_precision(pr.τ, pr.σ2, st)
    cst = 0.5 * (n * log(2π) + ld)
    lp = 0.0
    na = size(A, 5)
    @inbounds for la in 1:na, fb in 1:comp.nfb, fs in 1:comp.nfseg
        x = @view A[1, :, fs, fb, la]
        gx = @view gA[1, :, fs, fb, la]
        E = _ou_energy_and_grad!(gx, d, e, x)
        lp += -E - cst
    end
    return lp
end

# ── Whole-model logprior + gradient (added once by the driver) ────────────────

# Apply the component priors over one group's component tuple, accumulating the
# structured gradient into the parallel grad arrays; returns the group's logprior.
function _accum_group_prior!(cp::ComponentPriors, syms::Tuple, comps::Tuple, arrs::Tuple, garrs::Tuple, geom)
    lp = 0.0
    for i in eachindex(comps)
        pr = get(cp.priors, _sym(syms[i]), NoPrior())
        pr isa NoPrior && continue
        lp += _apply_prior!(pr, comps[i], arrs[i], garrs[i], geom)
    end
    return lp
end

"""
    logprior_and_grad!(g, cp::ComponentPriors, plan, geom, p) -> Real

Total `logprior` over all components `cp` attaches a prior to, ACCUMULATING its
gradient into the (zero-initialized) `ComponentVector` `g` (sharing `p`'s layout).
The empty `ComponentPriors()` returns 0 and leaves `g` untouched.
"""
function logprior_and_grad!(g, cp::ComponentPriors, plan::GainPlan, geom, p)
    isempty(cp) && return 0.0
    lp = 0.0
    for gi in eachindex(plan.groups)
        gp = plan.groups[gi]
        parr, aarr = group_arrays(plan, p, gi)
        gpar, gaar = group_arrays(plan, g, gi)
        lp += _accum_group_prior!(cp, gp.phase_syms, gp.phase, parr, gpar, geom)
        lp += _accum_group_prior!(cp, gp.logamp_syms, gp.logamp, aarr, gaar, geom)
    end
    return lp
end

"""
    logprior(cp::ComponentPriors, plan, geom, p) -> Real

Total `logprior` (value only) — see [`logprior_and_grad!`](@ref).
"""
function logprior(cp::ComponentPriors, plan::GainPlan, geom, p)
    isempty(cp) && return zero(float(eltype(p)))
    g = zero(p)
    return logprior_and_grad!(g, cp, plan, geom, p)
end
