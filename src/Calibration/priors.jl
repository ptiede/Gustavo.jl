# ── Priors on a component's parameters ──────────────────────────────────────
#
# A prior is the Gaussian belief about a component's parameters before the
# data. It sits on the `GainComponent` (`prior =`), never on the term. A
# correlated prior takes its axis from the component (`resolve_prior`). A solver
# declares which priors it supports through `can_fit`.

"""
    AbstractPrior

A Gaussian prior on the parameters of a [`GainComponent`](@ref):
[`IIDPrior`](@ref), [`RandomWalkPrior`](@ref), [`OUPrior`](@ref) or
[`CARMAPrior`](@ref). A
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
    RandomWalkPrior(; order = 1, σ, init = nothing)

A random walk of the given `order` `m` along the component's axis (see
[`resolve_prior`](@ref)): the `(m − 1)`-times integrated Brownian motion in the
axis coordinate. Its `(m − 1)`-th derivative changes by `N(0, σ²·Δx)` over a
step `Δx`, so `σ²` is in the parameter's units squared per `x^(2m − 1)`: per
second or per Hz for order 1, per Hz³ along frequency for order 2. The prior
holds for any spacing of the values and means the same at any segment
resolution. Order 2 gives a cubic smoothing spline.

`init` is the prior on the initial value and its first `m − 1` derivatives,
`(f, f′, …, f^(m−1))` in units of the axis coordinate, relative to the track's
level: a `Distributions.MvNormal` of length `m`, or a `Normal` for `m = 1`.
With `init = nothing` these are left free (a flat start), so the prior is
improper in those directions and leaves the track's level free; with an `init`
the walk is proper, and a level beside it is estimated as for an
[`OUPrior`](@ref). A solver that needs a proper prior requires an `init`.

`σ` is either a positive number, held fixed, or a hyperprior (any density
implementing DensityInterface's `logdensityof`), estimated by type-II MAP as
for [`OUPrior`](@ref): the marginal likelihood of the data (restricted, with
a flat start or a level), with the values integrated out, times the hyperprior
density over `log σ`. The hyperprior must be proper.
"""
struct RandomWalkPrior{G, S} <: AbstractPrior
    order::Int
    σ::G
    init::S
    function RandomWalkPrior{G, S}(order, σ, init) where {G, S}
        order >= 1 || throw(ArgumentError("RandomWalkPrior order must be ≥ 1, got $order"))
        _check_hyper("RandomWalkPrior σ", σ)
        _check_init(order, init)
        return new{G, S}(order, σ, init)
    end
end
RandomWalkPrior(; order::Integer = 1, σ, init = nothing) = RandomWalkPrior{typeof(σ), typeof(init)}(order, σ, init)

_check_init(order, ::Nothing) = nothing
_check_init(order, ::Normal) = order == 1 || throw(
    ArgumentError("a RandomWalkPrior of order $order needs an MvNormal init of length $order, got a Normal"),
)
_check_init(order, init::AbstractMvNormal) = length(init) == order || throw(
    ArgumentError("a RandomWalkPrior of order $order needs an init of length $order, got length $(length(init))"),
)
_check_init(order, init) = throw(
    ArgumentError("a RandomWalkPrior init must be `nothing`, a Normal or an MvNormal, got $(typeof(init))"),
)

# The mean vector and covariance matrix of a `RandomWalkPrior`'s proper init.
_init_moments(init::Normal) = [mean(init)], fill(var(init), 1, 1)
_init_moments(init::AbstractMvNormal) = collect(mean(init)), Matrix(cov(init))

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

"""
    RealRoot(; scale)

A real root `−1/scale` of a [`CARMAPrior`](@ref)'s autoregressive or
moving-average polynomial. Alone in the autoregressive part it is an
Ornstein–Uhlenbeck decay over `scale`. `scale` is in the coordinate's units (Hz
along frequency): a positive number, held fixed, or a hyperprior.
"""
struct RealRoot{S}
    scale::S
    function RealRoot{S}(scale) where {S}
        _check_hyper("RealRoot scale", scale)
        return new{S}(scale)
    end
end
RealRoot(; scale) = RealRoot{typeof(scale)}(scale)

"""
    ComplexRoots(; period, scale)

The complex pair of roots `−1/scale ± 2πi/period` of a [`CARMAPrior`](@ref)'s
autoregressive or moving-average polynomial. In the autoregressive part it adds
a quasi-periodic term to the covariance, `∝ e^(−|Δ|/scale)·cos(2πΔ/period + ϑ)`:
a ripple of `period` whose phase stays coherent over about `scale`, and a
fixed sinusoid with Gaussian amplitudes as `scale → ∞`. Both are in the
coordinate's units, each a positive number or a hyperprior.
"""
struct ComplexRoots{P, S}
    period::P
    scale::S
    function ComplexRoots{P, S}(period, scale) where {P, S}
        _check_hyper("ComplexRoots period", period)
        _check_hyper("ComplexRoots scale", scale)
        return new{P, S}(period, scale)
    end
end
function ComplexRoots(; period, scale)
    p, s = _promote_fixed(period, scale)
    return ComplexRoots{typeof(p), typeof(s)}(p, s)
end

const _RootFactor = Union{RealRoot, ComplexRoots}

_root_count(::RealRoot) = 1
_root_count(::ComplexRoots) = 2
_root_count(::Tuple{}) = 0
_root_count(factors::Tuple) = _root_count(first(factors)) + _root_count(Base.tail(factors))

"""
    CARMAPrior(; ar, ma = (), σ)

A stationary continuous-time autoregressive moving-average process, CARMA(p, q)
(Kelly et al. 2014), along the component's axis (see [`resolve_prior`](@ref))
about a free level per block:

    a(D) x = b(D) ε,   a(s) = Π (s − r),   b(s) = Π (1 − s/r),

with `D` the derivative along the coordinate, `ε` white noise, `a` over the
roots of `ar` and `b` over those of `ma`. Each is a tuple of
[`RealRoot`](@ref) and [`ComplexRoots`](@ref) factors; `ar` holds `p ≥ 1`
roots and `ma` `q < p`. Every root has a negative real part, so the process is
stationary, and `σ` is its stationary standard deviation, in the parameter's
own units. The power spectrum is `∝ |b(2πiφ)|²/|a(2πiφ)|²`: the
autoregressive roots set the correlation scales and any quasi-periodic peaks,
and the moving-average roots raise the spectrum's tail, so that the process is
`p − q − 1` times differentiable. `ar = (RealRoot(; scale),)` is the
[`OUPrior`](@ref) with that `scale`.

Every `scale`, `period` and `σ` is a positive number, held fixed, or a
hyperprior estimated by type-II MAP as for [`OUPrior`](@ref), over their logs.
An estimated `scale` is confined to between one sample spacing `Δ` and the
span of the longest piece `W`, and a `period` to `[2Δ, W]`: the samples do not
measure them beyond.
"""
struct CARMAPrior{A <: Tuple, M <: Tuple, G} <: AbstractPrior
    ar::A
    ma::M
    σ::G
    function CARMAPrior{A, M, G}(ar, ma, σ) where {A, M, G}
        all(f -> f isa _RootFactor, ar) && all(f -> f isa _RootFactor, ma) || throw(
            ArgumentError("CARMAPrior ar and ma must hold RealRoot and ComplexRoots factors, got $(typeof(ar)) and $(typeof(ma))"),
        )
        p, q = _root_count(ar), _root_count(ma)
        p >= 1 || throw(ArgumentError("CARMAPrior needs at least one autoregressive root"))
        q < p || throw(
            ArgumentError("CARMAPrior needs fewer moving-average roots than autoregressive ones, got q = $q and p = $p"),
        )
        _check_hyper("CARMAPrior σ", σ)
        return new{A, M, G}(ar, ma, σ)
    end
end
function CARMAPrior(; ar, ma = (), σ)
    a, m = Tuple(ar), Tuple(ma)
    return CARMAPrior{typeof(a), typeof(m), typeof(σ)}(a, m, σ)
end

Base.:(==)(a::IIDPrior, b::IIDPrior) = a.σ == b.σ
Base.hash(p::IIDPrior, h::UInt) = hash(p.σ, hash(:IIDPrior, h))
Base.:(==)(a::RandomWalkPrior, b::RandomWalkPrior) = a.order == b.order && a.σ == b.σ && a.init == b.init
Base.hash(p::RandomWalkPrior, h::UInt) = hash(p.init, hash(p.σ, hash(p.order, hash(:RandomWalkPrior, h))))
Base.:(==)(a::OUPrior, b::OUPrior) = a.scale == b.scale && a.σ == b.σ
Base.hash(p::OUPrior, h::UInt) = hash(p.σ, hash(p.scale, hash(:OUPrior, h)))
Base.:(==)(a::RealRoot, b::RealRoot) = a.scale == b.scale
Base.hash(f::RealRoot, h::UInt) = hash(f.scale, hash(:RealRoot, h))
Base.:(==)(a::ComplexRoots, b::ComplexRoots) = a.period == b.period && a.scale == b.scale
Base.hash(f::ComplexRoots, h::UInt) = hash(f.scale, hash(f.period, hash(:ComplexRoots, h)))
Base.:(==)(a::CARMAPrior, b::CARMAPrior) = a.ar == b.ar && a.ma == b.ma && a.σ == b.σ
Base.hash(p::CARMAPrior, h::UInt) = hash(p.σ, hash(p.ma, hash(p.ar, hash(:CARMAPrior, h))))

Base.show(io::IO, f::RealRoot) = print(io, "RealRoot(; scale = ", repr(f.scale), ")")
Base.show(io::IO, f::ComplexRoots) = print(io, "ComplexRoots(; period = ", repr(f.period), ", scale = ", repr(f.scale), ")")

const _CorrelatedPrior = Union{RandomWalkPrior, OUPrior, CARMAPrior}

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
                    "must be a RandomWalkPrior, OUPrior or CARMAPrior (pass an IIDPrior unkeyed).",
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

_call_string(p::RandomWalkPrior) =
    "RandomWalkPrior(; order = $(p.order), σ = $(repr(p.σ))" * (isnothing(p.init) ? "" : ", init = $(repr(p.init))") * ")"
_call_string(p::OUPrior) = "OUPrior(; scale = $(repr(p.scale)), σ = $(repr(p.σ)))"
_call_string(p::CARMAPrior) = "CARMAPrior(; ar = $(repr(p.ar)), ma = $(repr(p.ma)), σ = $(repr(p.σ)))"
_call_string(nt::NamedTuple) =
    "(" * join(("$k = $(_call_string(p))" for (k, p) in pairs(nt)), ", ") * (length(nt) == 1 ? ",)" : ")")
