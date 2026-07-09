# ── Reparameterization: gauge fixing + parameter scaling (Stage 2, Milestone 5b)
#
# The raw parameter vector `p` is badly conditioned for a gradient optimizer: a
# delay's gradient is ~1e9× a phase's (phase = 2π·τ·(f−f0)), and the array has
# unobservable GAUGE directions (a common per-segment phase/amp shift leaves the
# objective invariant). Both are handled by a REPARAMETERIZATION `y ↔ p`:
#
#   • GAUGE fixing removes a set of parameters (e.g. the reference antenna's
#     phase/amp blocks) from the free vector — they are held at 0, killing the
#     flat directions.
#   • SCALING maps `p_free = scale .* y` so the optimizer works in O(1) units.
#
# The optimizer minimizes over the reduced, scaled `y` (a `ReparamPosterior`); the
# gradient chain-rules to `∂/∂yᵢ = scaleᵢ · (∂/∂p)[freeᵢ]`. Both the gauge and the
# scaling are pluggable and user-specifiable.

# ── Gauge ─────────────────────────────────────────────────────────────────────

"How the unobservable gauge is fixed (which parameters are held constant)."
abstract type AbstractGauge end

"No gauge fixing — all parameters free (the objective's flat directions remain)."
struct NoGauge <: AbstractGauge end

"""
    ReferenceAntenna(ant; phase = true, amp = true, components = :all)

Pin reference antenna `ant`'s gain to unity by holding its phase blocks (and, if
`amp`, its log-amplitude blocks) fixed at 0 — the standard fringe gauge. Pinning
amplitude too is correct for a profiled/relative-amplitude solve (the overall
scale is unobservable); use `amp = false` with a fixed source that constrains it.
`components` selects which named components to pin (`:all` or a tuple of names).
"""
struct ReferenceAntenna <: AbstractGauge
    ant::Int
    phase::Bool
    amp::Bool
    components::Any
end
ReferenceAntenna(ant::Integer; phase::Bool = true, amp::Bool = true, components = :all) =
    ReferenceAntenna(Int(ant), phase, amp, components)

"""
    FixParams(indices)

Hold an explicit set of flat parameter `indices` (into the `ComponentVector`)
fixed at 0. The escape hatch for a fully custom gauge.
"""
struct FixParams <: AbstractGauge
    indices::Vector{Int}
end

_want(components, name) = components === :all || name in components

# Flat indices of `p` that the gauge holds fixed.
_fixed_indices(::NoGauge, plan) = Int[]
_fixed_indices(g::FixParams, plan) = g.indices
function _fixed_indices(g::ReferenceAntenna, plan)
    mask = similar(plan.template, Bool)
    fill!(mask, false)
    for gp in plan.groups
        la = findfirst(==(g.ant), gp.ants)
        la === nothing && continue
        pg = _sub(mask, gp.groupval)
        g.phase && for (comp, v) in zip(gp.phase, gp.phase_syms)
            _want(g.components, _sym(v)) && (getproperty(pg, _sym(v))[:, :, :, :, la] .= true)
        end
        g.amp && for (comp, v) in zip(gp.logamp, gp.logamp_syms)
            _want(g.components, _sym(v)) && (getproperty(pg, _sym(v))[:, :, :, :, la] .= true)
        end
    end
    return findall(flatten(mask))
end

# ── Scaling ───────────────────────────────────────────────────────────────────

"How each parameter is scaled so the optimizer sees O(1) variables."
abstract type AbstractScaling end

"No scaling (unit scale on every parameter)."
struct NoScaling <: AbstractScaling end

"""
    AutoScale()

Per-component characteristic scale from the geometry via [`param_scale`](@ref):
a delay is scaled by `1/(2π·Δf)`, a rate by `1/(2π·Δt)`, etc., so a unit change in
the scaled variable is ~1 radian. The default.
"""
struct AutoScale <: AbstractScaling end

"""
    CustomScale(scales::NamedTuple)

User scales per named component, e.g. `CustomScale((clock = 1e-9, lo = 1e3))`;
components absent from `scales` keep the `AutoScale` value.
"""
struct CustomScale{NT <: NamedTuple} <: AbstractScaling
    scales::NT
end

"""
    param_scale(term::AbstractGainTerm, comp::SiteComponent) -> Real

The characteristic magnitude of `term`'s parameter — the optimizer's scaled
variable is `p / param_scale`. The generic default is the Jacobi scale
`1 / max|design column|` from `basis_columns` over the component's coordinate, so
it is correct for ANY term (delay → `1/(2π·max|f−f0|)`, constant/bandpass → 1)
with no per-term method; override it for a custom scaling.
"""
function param_scale(t::AbstractGainTerm, comp::SiteComponent)
    coord = coord_kind(t) == COORD_TIME ? comp.xt : comp.xf
    m = maximum(abs, basis_columns(t, coord); init = 0.0)
    return m > 0 ? inv(m) : 1.0
end

# Per-free-index scale vector.
_scale_vector(::NoScaling, plan) = ones(length(plan.template))
function _scale_vector(scaling::AbstractScaling, plan)
    sv = similar(plan.template, Float64)
    fill!(sv, 1.0)
    for gp in plan.groups
        pg = _sub(sv, gp.groupval)
        for (comp, v) in zip((gp.phase..., gp.logamp...), (gp.phase_syms..., gp.logamp_syms...))
            getproperty(pg, _sym(v)) .= _component_scale(scaling, _sym(v), comp)
        end
    end
    return flatten(sv)
end
_component_scale(::AutoScale, name, comp) = param_scale(comp.term, comp)
_component_scale(s::CustomScale, name, comp) =
    haskey(s.scales, name) ? Float64(getproperty(s.scales, name)) : param_scale(comp.term, comp)

# ── Reparam + reduced posterior ───────────────────────────────────────────────

"""
    Reparam(free, scale, p0)

The reparameterization: `free` are the flat indices of `p` the optimizer varies,
`scale[i]` the scale of `free[i]`, and `p0` a full parameter `ComponentVector`
holding the FIXED (gauge) values in place. `to_full(rp, y)` maps the reduced
scaled vector back to a full `p`.
"""
struct Reparam{CV}
    free::Vector{Int}
    scale::Vector{Float64}
    p0::CV
end

nfree(rp::Reparam) = length(rp.free)

"""
    build_reparam(plan, gauge, scaling, p0) -> Reparam

Construct the reparameterization from a `gauge` and a `scaling` over `plan`'s
layout, taking free-parameter initial values from the warm-start `p0` (fixed
positions are zeroed).
"""
function build_reparam(plan, gauge::AbstractGauge, scaling::AbstractScaling, p0)
    fixed = _fixed_indices(gauge, plan)
    fixedset = Set(fixed)
    free = [i for i in 1:length(plan.template) if !(i in fixedset)]
    scale = _scale_vector(scaling, plan)[free]
    base = copy(p0)
    d = flatten(base)
    @inbounds for i in fixed
        d[i] = 0.0
    end
    return Reparam(free, scale, base)
end

# y (reduced, scaled) → full p ComponentVector (fixed positions from p0).
function to_full(rp::Reparam, y)
    full = copy(rp.p0)
    d = flatten(full)
    @inbounds for i in eachindex(rp.free)
        d[rp.free[i]] = rp.scale[i] * y[i]
    end
    return full
end

# full p → y (reduced, scaled) — the warm-start init.
to_free(rp::Reparam, p) = flatten(p)[rp.free] ./ rp.scale

"""
    ReparamPosterior(post, reparam)

The `FringePosterior` viewed over the reduced, scaled variable `y`; implements
`LogDensityProblems` order-1 so any optimizer minimizes the well-conditioned,
gauge-fixed problem. The gradient chain-rules `∂/∂yᵢ = scaleᵢ·(∂/∂p)[freeᵢ]`.
"""
struct ReparamPosterior{P, R <: Reparam}
    post::P
    reparam::R
end

LogDensityProblems.dimension(rp::ReparamPosterior) = nfree(rp.reparam)
LogDensityProblems.capabilities(::Type{<:ReparamPosterior}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(rp::ReparamPosterior, y::AbstractVector)
    return LogDensityProblems.logdensity(rp.post, flatten(to_full(rp.reparam, y)))
end
function LogDensityProblems.logdensity_and_gradient(rp::ReparamPosterior, y::AbstractVector)
    v, gp = LogDensityProblems.logdensity_and_gradient(rp.post, flatten(to_full(rp.reparam, y)))
    gy = rp.reparam.scale .* gp[rp.reparam.free]
    return v, gy
end
