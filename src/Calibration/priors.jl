# ── Priors on a component's parameters ──────────────────────────────────────
#
# A prior is the Gaussian belief about a component's parameters before the
# data. It sits on the `GainComponent` (`prior =`), never on the term, and
# relates values within one parameter block only: the component's segments are
# the breaks. A solver declares which priors it supports through `can_fit`.

"""
    AbstractPrior

A Gaussian prior on the parameters of a [`GainComponent`](@ref):
[`IIDPrior`](@ref), [`RandomWalkPrior`](@ref) or [`OUPrior`](@ref). A prior
relates values within one parameter block, never across blocks, so the
component's segments are its breaks.
"""
abstract type AbstractPrior end

"""
    IIDPrior(σ)

Every parameter of the component independently `N(0, σ²)`, with `σ` in the
parameter's own units.
"""
struct IIDPrior{T <: Real} <: AbstractPrior
    σ::T
    function IIDPrior{T}(σ) where {T}
        _check_positive("IIDPrior σ", σ)
        return new{T}(σ)
    end
end
IIDPrior(σ::Real) = IIDPrior{typeof(σ)}(σ)

"""
    RandomWalkPrior(dim; order = 1, σ)

A random walk of the given `order` along dimension `dim` (`Frequency` or
`Ti`): the `order`-th difference between neighboring values is independently
`N(0, σ²)`, with `σ` in the parameter's own units. The walk's starting level
(and, for `order > 1`, its first `order - 1` differences) is left free, so the
prior is improper in those directions. Order 2 makes the posterior mean a
Whittaker smoother with `λ = 1/σ²` against inverse-variance weights.

The component's term must carry a value per coordinate along `dim`
(`Bandpass` along `Frequency`).
"""
struct RandomWalkPrior{D, T <: Real} <: AbstractPrior
    order::Int
    σ::T
    function RandomWalkPrior{D, T}(order, σ) where {D, T}
        _check_prior_axis(RandomWalkPrior, D)
        order >= 1 || throw(ArgumentError("RandomWalkPrior order must be ≥ 1, got $order"))
        _check_positive("RandomWalkPrior σ", σ)
        return new{D, T}(order, σ)
    end
end
RandomWalkPrior(dim; order::Integer = 1, σ::Real) = RandomWalkPrior{dim, typeof(σ)}(order, σ)

"""
    OUPrior(dim; scale, σ)

A stationary Ornstein–Uhlenbeck process along dimension `dim` (`Frequency` or
`Ti`) about a free level per block: the deviations from the level are jointly
Gaussian with covariance `σ² exp(-|x - x′| / scale)` between coordinates `x`
and `x′`. `σ` is in the parameter's own units and `scale` in the coordinate's
(Hz or s).

Each of `scale` and `σ` is either a positive number, held fixed, or a
hyperprior — any density implementing DensityInterface's `logdensityof`, such
as a Distributions.jl distribution. A solver estimates the hyperparameters by
type-II MAP: it maximizes the marginal likelihood of the data, with the
parameter values integrated out, times the hyperprior density over
`(log scale, log σ)`.

The component's term must carry a value per coordinate along `dim`
(`Bandpass` along `Frequency`).
"""
struct OUPrior{D, S, G} <: AbstractPrior
    scale::S
    σ::G
    function OUPrior{D, S, G}(scale, σ) where {D, S, G}
        _check_prior_axis(OUPrior, D)
        _check_hyper("OUPrior scale", scale)
        _check_hyper("OUPrior σ", σ)
        return new{D, S, G}(scale, σ)
    end
end
function OUPrior(dim; scale, σ)
    s, g = _promote_fixed(scale, σ)
    return OUPrior{dim, typeof(s), typeof(g)}(s, g)
end

_promote_fixed(a::Real, b::Real) = promote(a, b)
_promote_fixed(a, b) = (a, b)

_check_hyper(what, x::Real) = _check_positive(what, x)
_check_hyper(what, x) =
    DensityInterface.DensityKind(x) === DensityInterface.HasDensity() || throw(
    ArgumentError(
        "$what must be a positive number or a density implementing " *
            "DensityInterface.logdensityof, got $(typeof(x))",
    )
)

"""
    is_fixed_hyper(x) -> Bool

Whether a prior hyperparameter is a fixed value (a number) rather than a
hyperprior to be estimated.
"""
is_fixed_hyper(::Real) = true
is_fixed_hyper(_) = false

_check_positive(what, x) =
    (isfinite(x) && x > 0) || throw(ArgumentError("$what must be positive and finite, got $x"))

_check_prior_axis(P, D) = D in TERM_AXES || throw(
    ArgumentError(
        "$(nameof(P)) axis must be one of $(map(DimensionalData.name, TERM_AXES)), got $D"
    )
)

"""
    prior_axis(prior) -> Union{Type, Nothing}

The dimension a prior correlates values along, or `nothing` for a prior
without one ([`IIDPrior`](@ref)).
"""
prior_axis(::IIDPrior) = nothing
prior_axis(::RandomWalkPrior{D}) where {D} = D
prior_axis(::OUPrior{D}) where {D} = D

# A correlated prior needs values to correlate within a block: the term must
# carry a value per coordinate along the prior's axis.
function _check_component_prior(e)
    isnothing(e.prior) && return nothing
    D = prior_axis(e.prior)
    (isnothing(D) || value_axis(e.term) === D) || throw(
        ArgumentError(
            "$(component_label(e)): $(nameof(typeof(e.prior))) correlates values along " *
                "$(DimensionalData.name(D)), but $(nameof(typeof(e.term))) has one value per " *
                "block along it. A prior never correlates across blocks; use a term with a " *
                "value per $(DimensionalData.name(D)) coordinate (e.g. Bandpass() along Frequency).",
        )
    )
    return nothing
end

_call_string(p::RandomWalkPrior{D}) where {D} =
    "RandomWalkPrior($(DimensionalData.name(D)); order = $(p.order), σ = $(repr(p.σ)))"
_call_string(p::OUPrior{D}) where {D} =
    "OUPrior($(DimensionalData.name(D)); scale = $(repr(p.scale)), σ = $(repr(p.σ)))"
