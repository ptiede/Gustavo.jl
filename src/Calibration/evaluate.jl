# ── Pure forward evaluation ──────────────────────────────────────────────────
#
# `GainEvaluator` pairs a `StationGainModel` with its `ParameterLayout`. The map
# θ → gains is pure: no mutation of θ, no global state, allocation only of the
# output array, and fully type-stable (verified by `@inferred` in the tests).
# This is the surface a future Comrade/Reactant global solver will trace; the
# WLS solvers may mutate their own scratch, but they go through this same map to
# predict visibilities.

struct GainEvaluator{M <: StationGainModel, L <: ParameterLayout}
    model::M
    layout::L
end

GainEvaluator(model::StationGainModel, geom::DataGeometry; nant::Integer) =
    GainEvaluator(model, plan_parameters(model, nant, geom))

nparameters(ev::GainEvaluator) = ev.layout.nθ

# Sum a tuple of TiedComponents' contributions for one (ant, feed, ti, c) cell.
# Recursion over the (compile-time) tuple keeps each term's type concrete, so
# `term_eval` dispatches statically — the key to type stability.
@inline _sum_components(::Tuple{}, plans, k, θ, ant, feed, ti, c) = zero(eltype(θ))
@inline function _sum_components(comps::Tuple, plans, k, θ, ant, feed, ti, c)
    tc = comps[1]
    @inbounds plan = plans[k]
    v = _component_value(term(tc), plan, θ, ant, feed, ti, c)
    return v + _sum_components(Base.tail(comps), plans, k + 1, θ, ant, feed, ti, c)
end

@inline function _component_value(t::AbstractGainTerm, plan::ComponentPlan, θ, ant, feed, ti, c)
    @inbounds ts = plan.tseg_id[ti]
    @inbounds fs = plan.fseg_id[c]
    @inbounds o1 = plan.off1[ant, feed, ts, fs]
    val = zero(eltype(θ))
    x = _cell_coordinates(t, plan, ti, c)
    @inbounds shapes = param_shapes(t, plan.nchan_seg[fs])
    shift = _channel_shift(t, plan, c, shapes)
    if o1 != 0
        val += term_eval(t, _block_params(shapes, θ, o1 + shift), x)
    end
    @inbounds o2 = plan.off2[ant, feed, ts, fs]
    if o2 != 0
        val += term_eval(t, _block_params(shapes, θ, o2 + shift), x)
    end
    return val
end

# The coordinates a term declares through `term_axes`, under those names. The
# declaration is a compile-time constant for a concrete term, so the selection
# folds away and each `term_eval` sees concretely-typed scalars.
@inline _cell_coordinates(t::AbstractGainTerm, plan::ComponentPlan, ti, c) =
    NamedTuple{term_axes(t)}(_axis_values(term_axes(t), plan, ti, c))

@inline _axis_values(::Tuple{}, plan, ti, c) = ()
@inline function _axis_values(names::Tuple, plan, ti, c)
    v = first(names) === :Frequency ? (@inbounds plan.xf[c]) : (@inbounds plan.xt[ti])
    return (v, _axis_values(Base.tail(names), plan, ti, c)...)
end

# Blocks of a `params_per_channel` term hold one parameter group per channel of
# the frequency segment; this cell reads the group for its own channel.
@inline function _channel_shift(t::AbstractGainTerm, plan::ComponentPlan, c, shapes)
    params_per_channel(t) || return 0
    @inbounds cl = plan.clocal[c]
    return (cl - 1) * _group_size(values(shapes))
end

@inline _group_size(::Tuple{}) = 0
@inline _group_size(shapes::Tuple) = prod(first(shapes)) + _group_size(Base.tail(shapes))

# Address the parameters of one block: the declared names, in declaration order,
# starting at `off` in θ. The layout reserved exactly `nparams_per_block` entries
# there, so the block is in bounds by construction — this is the one place that
# knows it, which is why the terms themselves need no bounds assertion.
@inline _block_params(shapes::NamedTuple{names}, θ, off) where {names} =
    NamedTuple{names}(_shaped_params(values(shapes), θ, off))

@inline _shaped_params(::Tuple{}, θ, off) = ()
@inline function _shaped_params(shapes::Tuple, θ, off)
    p = _shaped_param(first(shapes), θ, off)
    return (p, _shaped_params(Base.tail(shapes), θ, off + prod(first(shapes)))...)
end

@inline _shaped_param(::Tuple{}, θ, off) = @inbounds θ[off]
@inline _shaped_param(shape::Tuple{Integer}, θ, off) =
    @inbounds view(θ, off:(off + shape[1] - 1))

"""
    evaluate_gains(ev::GainEvaluator, θ) -> Array{Complex,4}

Pure forward map θ → complex antenna gains of shape `(nchan, ntime, nant, 2)`,
where the last axis is the feed (1, 2). `gain = exp(Σ logamp) · cis(Σ phase)`.
Element type follows `eltype(θ)` (so AD / Reactant tracing flows through).
"""
function evaluate_gains(ev::GainEvaluator, θ::AbstractVector)
    lay = ev.layout
    length(θ) == lay.nθ ||
        error("evaluate_gains: θ has length $(length(θ)), expected $(lay.nθ)")
    T = float(eltype(θ))
    gains = Array{Complex{T}}(undef, lay.nchan, lay.ntime, lay.nant, 2)
    phase_c = phase_components(ev.model)
    logamp_c = logamp_components(ev.model)
    nphase = lay.nphase
    plans = lay.plans
    @inbounds for feed in 1:2, ant in 1:lay.nant, ti in 1:lay.ntime, c in 1:lay.nchan
        phase = _sum_components(phase_c, plans, 1, θ, ant, feed, ti, c)
        logamp = _sum_components(logamp_c, plans, nphase + 1, θ, ant, feed, ti, c)
        gains[c, ti, ant, feed] = exp(logamp) * cis(phase)
    end
    return gains
end

"""
    evaluate_gains(ev::GainEvaluator, θ, chan_idx, ti_idx) -> Array{Complex,4}

Windowed pure forward map: gains of shape `(length(chan_idx), length(ti_idx),
nant, 2)` at the given GLOBAL channel and time indices (into `ev`'s geometry).
Use this to evaluate the gains a single UVSet leaf needs without materializing
the full `(nchan_total, ntime_total, …)` array.
"""
function evaluate_gains(
        ev::GainEvaluator, θ::AbstractVector,
        chan_idx::AbstractVector{<:Integer}, ti_idx::AbstractVector{<:Integer},
    )
    lay = ev.layout
    length(θ) == lay.nθ ||
        error("evaluate_gains: θ has length $(length(θ)), expected $(lay.nθ)")
    T = float(eltype(θ))
    gains = Array{Complex{T}}(undef, length(chan_idx), length(ti_idx), lay.nant, 2)
    phase_c = phase_components(ev.model)
    logamp_c = logamp_components(ev.model)
    nphase = lay.nphase
    plans = lay.plans
    @inbounds for feed in 1:2, ant in 1:lay.nant
        for (tii, ti) in enumerate(ti_idx), (ci, c) in enumerate(chan_idx)
            phase = _sum_components(phase_c, plans, 1, θ, ant, feed, ti, c)
            logamp = _sum_components(logamp_c, plans, nphase + 1, θ, ant, feed, ti, c)
            gains[ci, tii, ant, feed] = exp(logamp) * cis(phase)
        end
    end
    return gains
end

"""
    predict_visibilities(gains, coh, bl_a, bl_b, feed_a, feed_b) -> Array{Complex,4}

Pure visibility prediction `V̂[c,ti,bi,p] = g_a · coh · conj(g_b)` from antenna
gains and source coherencies.

- `gains`  : `(nchan, ntime, nant, 2)` from `evaluate_gains`.
- `coh`    : `(nbl, 2, 2)` source coherency per baseline (feed_a, feed_b indexed).
- `bl_a`, `bl_b` : antenna indices of each baseline (length `nbl`).
- `feed_a`, `feed_b` : feed index (1/2) of antenna A and B for each of the
  `npol` correlation products (from `correlation_feed_pair.(pol_products)`).
"""
function predict_visibilities(
        gains::AbstractArray{<:Complex, 4}, coh::AbstractArray{<:Complex, 3},
        bl_a::AbstractVector{<:Integer}, bl_b::AbstractVector{<:Integer},
        feed_a::AbstractVector{<:Integer}, feed_b::AbstractVector{<:Integer}
    )
    nchan, ntime, _, _ = size(gains)
    nbl = length(bl_a)
    length(bl_b) == nbl || error("bl_a and bl_b must have equal length")
    npol = length(feed_a)
    length(feed_b) == npol || error("feed_a and feed_b must have equal length")
    V = Array{eltype(gains)}(undef, nchan, ntime, nbl, npol)
    @inbounds for p in 1:npol
        fa = feed_a[p]
        fb = feed_b[p]
        for bi in 1:nbl
            a = bl_a[bi]
            b = bl_b[bi]
            s = coh[bi, fa, fb]
            for ti in 1:ntime, c in 1:nchan
                V[c, ti, bi, p] = gains[c, ti, a, fa] * s * conj(gains[c, ti, b, fb])
            end
        end
    end
    return V
end
