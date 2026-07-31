# ── Pure forward evaluation ──────────────────────────────────────────────────
#
# `GainEvaluator` pairs a `StationGainModel` with its `ParameterLayout`. The map
# θ → gains is pure: no mutation of θ, no global state, allocation only of the
# output array, and fully type-stable (verified by `@inferred` in the tests).
# This is the surface a future Comrade/Reactant global solver will trace; the
# WLS solvers may mutate their own scratch, but they go through this same map to
# predict visibilities.
#
# The map recurses the layout's `plantree` (a `NamedTuple` mirroring the model,
# so names are compile-time constants) and, at each component, slices its shaped
# leaf straight out of θ by the plan's `range`/`shape`, then reads the parameter
# block its `(feed, ti, c)` selects — no offset tables, no Dicts, no closures on
# the hot path. `θ` is a plain vector; `component_vector` wraps it for named
# inspection, but the map does not need that view.

struct GainEvaluator{M <: StationGainModel, L <: ParameterLayout}
    model::M
    layout::L
end

GainEvaluator(model::StationGainModel, geom::DataGeometry; nant::Integer) =
    GainEvaluator(model, plan_parameters(model, nant, geom))

nparameters(ev::GainEvaluator) = ev.layout.nθ

# Sum a group's (phase or log-amp) component contributions for one
# (ant, feed, ti, c) cell. Recursion over the group's nodes — the concretely
# typed value tuple of the `NamedTuple` — keeps each plan's type known, so
# `term_eval` dispatches statically and the walk unrolls.
@inline _sum_group(ptree::NamedTuple, θ, ant, feed, ti, c) =
    _sum_nodes(values(ptree), θ, ant, feed, ti, c)
@inline _sum_nodes(::Tuple{}, θ, ant, feed, ti, c) = zero(eltype(θ))
@inline function _sum_nodes(nodes::Tuple, θ, ant, feed, ti, c)
    here = _node_value(first(nodes), θ, ant, feed, ti, c)
    return here + _sum_nodes(Base.tail(nodes), θ, ant, feed, ti, c)
end

# A nested group (e.g. `sbd`) recurses; a leaf component evaluates its term.
@inline _node_value(sub::NamedTuple, θ, ant, feed, ti, c) =
    _sum_group(sub, θ, ant, feed, ti, c)
@inline function _node_value(plan::ComponentPlan, θ, ant, feed, ti, c)
    t = plan.term
    @inbounds ts = plan.tseg_id[ti]
    @inbounds fs = plan.fseg_id[c]
    x = _cell_coordinates(t, plan, ti, c)
    @inbounds shapes = param_shapes(t, plan.nchan_seg[fs])
    leaf = _component_leaf(plan, θ)
    val = zero(eltype(θ))
    node = _feed_node(plan.tying, feed)
    if node != 0
        val += term_eval(t, _block_params(shapes, _leaf_block(leaf, node, fs, ts, ant)), x)
    end
    # `ReferenceRelative`'s partner also reads its relative block (node 2).
    node2 = _feed_node2(plan.tying, feed)
    if node2 != 0
        val += term_eval(t, _block_params(shapes, _leaf_block(leaf, node2, fs, ts, ant)), x)
    end
    return val
end

# The parameter run of one block: the 1-D view over the leaf's `:param` axis at
# `(node, fs, ts, ant)`. The layout reserved exactly `nparams_per_block` entries
# there, so the block is in bounds by construction.
@inline _leaf_block(leaf, node, fs, ts, ant) = @inbounds view(leaf, :, node, fs, ts, ant)

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

# Name the parameters of one block: the declared names, in declaration order,
# over the block view (1-based). The block is in bounds by construction — this is
# the one place that knows it, which is why the terms need no bounds assertion.
@inline _block_params(shapes::NamedTuple{names}, block) where {names} =
    NamedTuple{names}(_shaped_params(values(shapes), block, 1))

@inline _shaped_params(::Tuple{}, block, off) = ()
@inline function _shaped_params(shapes::Tuple, block, off)
    p = _shaped_param(first(shapes), block, off)
    return (p, _shaped_params(Base.tail(shapes), block, off + prod(first(shapes)))...)
end

@inline _shaped_param(::Tuple{}, block, off) = @inbounds block[off]
@inline _shaped_param(shape::Tuple{Integer}, block, off) =
    @inbounds view(block, off:(off + shape[1] - 1))

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
    pp = lay.plantree.phase
    lp = lay.plantree.logamp
    @inbounds for feed in 1:2, ant in 1:lay.nant, ti in 1:lay.ntime, c in 1:lay.nchan
        phase = _sum_group(pp, θ, ant, feed, ti, c)
        logamp = _sum_group(lp, θ, ant, feed, ti, c)
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
    pp = lay.plantree.phase
    lp = lay.plantree.logamp
    @inbounds for feed in 1:2, ant in 1:lay.nant
        for (tii, ti) in enumerate(ti_idx), (ci, c) in enumerate(chan_idx)
            phase = _sum_group(pp, θ, ant, feed, ti, c)
            logamp = _sum_group(lp, θ, ant, feed, ti, c)
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
