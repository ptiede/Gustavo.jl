# ── Priors on a component's parameters ──────────────────────────────────────
#
# A prior is the Gaussian belief about a component's parameters before the
# data. It sits on the `GainComponent` (`prior =`), never on the term. A
# correlated prior takes its axis from the component (`resolve_prior`). A solver
# declares which priors it supports through `can_fit`.

"""
    AbstractPrior

A Gaussian prior on the parameters of a [`GainComponent`](@ref):
[`IIDPrior`](@ref), [`RandomWalkPrior`](@ref) or [`OUPrior`](@ref). A
correlated prior relates values along the one axis the component segments
([`resolve_prior`](@ref)).
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
    RandomWalkPrior(; order = 1, σ)

A random walk of the given `order` `m` along the component's axis (see
[`resolve_prior`](@ref)): the `(m − 1)`-times integrated Brownian motion in the
axis coordinate. Its `(m − 1)`-th derivative changes by `N(0, σ²·Δx)` over a
step `Δx`, so `σ²` is in the parameter's units squared per `x^(2m − 1)`: per
second or per Hz for order 1, per Hz³ along frequency for order 2. The prior
holds for any spacing of the values and means the same at any segment
resolution. The walk's starting value (and, for `m > 1`, its first `m − 1`
derivatives) is left free, so the prior is improper in those directions.
Order 2 gives a cubic smoothing spline.
"""
struct RandomWalkPrior{T <: Real} <: AbstractPrior
    order::Int
    σ::T
    function RandomWalkPrior{T}(order, σ) where {T}
        order >= 1 || throw(ArgumentError("RandomWalkPrior order must be ≥ 1, got $order"))
        _check_positive("RandomWalkPrior σ", σ)
        return new{T}(order, σ)
    end
end
RandomWalkPrior(; order::Integer = 1, σ::Real) = RandomWalkPrior{typeof(σ)}(order, σ)

"""
    OUPrior(; scale, σ)

A stationary Ornstein–Uhlenbeck process along the component's axis (see
[`resolve_prior`](@ref)) about a free level per block: the deviations from the
level are jointly Gaussian with covariance `σ² exp(-|x - x′| / scale)` between
coordinates `x` and `x′`. `σ` is in the parameter's own units and `scale` in
the coordinate's (Hz or s).

Each of `scale` and `σ` is either a positive number, held fixed, or a
hyperprior — any density implementing DensityInterface's `logdensityof`, such
as a Distributions.jl distribution. A solver estimates the hyperparameters by
type-II MAP: it maximizes the marginal likelihood of the data, with the
parameter values integrated out, times the hyperprior density over
`(log scale, log σ)`.
"""
struct OUPrior{S, G} <: AbstractPrior
    scale::S
    σ::G
    function OUPrior{S, G}(scale, σ) where {S, G}
        _check_hyper("OUPrior scale", scale)
        _check_hyper("OUPrior σ", σ)
        return new{S, G}(scale, σ)
    end
end
function OUPrior(; scale, σ)
    s, g = _promote_fixed(scale, σ)
    return OUPrior{typeof(s), typeof(g)}(s, g)
end

const _CorrelatedPrior = Union{RandomWalkPrior, OUPrior}

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

# The axes a component segments: those whose segmentation is not `Global*`.
_segmented_axes(e) = (
    (e.Ti isa GlobalTime ? () : (:Ti,))...,
    (e.Frequency isa GlobalFrequency ? () : (:Frequency,))...,
)

"""
    resolve_prior(e::GainComponent) -> Union{Nothing, IIDPrior, NamedTuple}

The prior of `e` with each correlated prior keyed by the axis it relates
values along: `nothing`, an [`IIDPrior`](@ref) (which needs no axis), or a
`NamedTuple` such as `(Frequency = RandomWalkPrior(…),)`.

A bare [`RandomWalkPrior`](@ref) or [`OUPrior`](@ref) takes the one axis whose
segmentation is not `Global*`. A component segmenting both axes must key its
prior, `prior = (Ti = …, Frequency = …)` (either key may be left out). A
correlated prior on an axis the component does not segment relates nothing and
throws, as does a bare one on a component segmenting both.
"""
resolve_prior(e) = _resolve_prior(e, e.prior)
_resolve_prior(e, ::Nothing) = nothing
_resolve_prior(e, p::IIDPrior) = p
function _resolve_prior(e, p::_CorrelatedPrior)
    ax = _segmented_axes(e)
    isempty(ax) && throw(_relates_nothing(e, "segments neither Ti nor Frequency"))
    length(ax) == 1 || throw(
        ArgumentError(
            "$(component_label(e)): $(nameof(typeof(p))) is ambiguous, the component " *
                "segments both Ti and Frequency; key it by axis, e.g. " *
                "`prior = (Frequency = $(_call_string(p)),)`.",
        ),
    )
    return NamedTuple{ax}((p,))
end
function _resolve_prior(e, nt::NamedTuple)
    isempty(nt) && throw(
        ArgumentError("$(component_label(e)): a keyed prior needs at least one axis; use `prior = nothing`."),
    )
    ax = _segmented_axes(e)
    for (k, p) in pairs(nt)
        k in (:Ti, :Frequency) || throw(
            ArgumentError("$(component_label(e)): prior key $k is not an axis; use Ti or Frequency."),
        )
        p isa _CorrelatedPrior || throw(
            ArgumentError(
                "$(component_label(e)): the prior keyed $k is a $(typeof(p)); a keyed prior " *
                    "must be a RandomWalkPrior or OUPrior (pass an IIDPrior unkeyed).",
            ),
        )
        k in ax || throw(_relates_nothing(e, "does not segment $k"))
    end
    return nt
end
_relates_nothing(e, why) = ArgumentError(
    "$(component_label(e)): the correlated prior relates nothing, the component $why. " *
        "Segment the axis the prior should run along (e.g. `Frequency = ChannelBlocks(1)`).",
)

_call_string(p::RandomWalkPrior) = "RandomWalkPrior(; order = $(p.order), σ = $(repr(p.σ)))"
_call_string(p::OUPrior) = "OUPrior(; scale = $(repr(p.scale)), σ = $(repr(p.σ)))"
_call_string(nt::NamedTuple) =
    "(" * join(("$k = $(_call_string(p))" for (k, p) in pairs(nt)), ", ") * (length(nt) == 1 ? ",)" : ")")
