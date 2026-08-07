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
    evaluate_gains(ev::GainEvaluator, θ, solve_geom::DataGeometry, target::DataGeometry;
                   chan_idx = …, ti_idx = …, time_span = nothing)

Forward map onto a FOREIGN grid: gains of shape `(length(chan_idx),
length(ti_idx), nant, 2)` for the samples of `target` selected by
`chan_idx`/`ti_idx`, evaluated from a θ laid out over `solve_geom` (the geometry
`ev.layout` was planned on).

Each target sample is placed in the solve segment it BELONGS to — matched by
scan and spw label, or by the segmentation's own bin formula evaluated with the
solve's parameters — so a solution applies to data sampled differently from the
grid it was fit on: a segmentation coarser than the data has a segment covering
every sample, which is exactly the statement that the gain is constant across
it. A sample with no such segment is an error, naming the segmentation that
could not place it; that is the "coarser is fine, finer is not" contract.

`time_span[k]` is the interval the `k`-th selected sample integrates over (see
`PartitionInfo.time_span`); where the segmentation bins a coordinate, a sample
whose span crosses a bin boundary is rejected rather than assigned by its centre.

Coordinates (a `Delay`'s `f − f0`, a `Polynomial`'s scaled offset) are read
pointwise from the solution's own resolved state, so they stay in the basis θ
was fit in.
"""
function evaluate_gains(
        ev::GainEvaluator, θ::AbstractVector, solve_geom::DataGeometry, target::DataGeometry;
        chan_idx::AbstractVector{<:Integer} = Base.OneTo(nchannels(target)),
        ti_idx::AbstractVector{<:Integer} = Base.OneTo(ntimes(target)),
        time_span = nothing,
    )
    lay = ev.layout
    length(θ) == lay.nθ ||
        error("evaluate_gains: θ has length $(length(θ)), expected $(lay.nθ)")
    pp = _resolve_tree(lay.plantree.phase, solve_geom, target, chan_idx, ti_idx, time_span)
    lp = _resolve_tree(lay.plantree.logamp, solve_geom, target, chan_idx, ti_idx, time_span)
    T = float(eltype(θ))
    gains = Array{Complex{T}}(undef, length(chan_idx), length(ti_idx), lay.nant, 2)
    @inbounds for feed in 1:2, ant in 1:lay.nant
        for ti in axes(gains, 2), c in axes(gains, 1)
            phase = _sum_group(pp, θ, ant, feed, ti, c)
            logamp = _sum_group(lp, θ, ant, feed, ti, c)
            gains[c, ti, ant, feed] = exp(logamp) * cis(phase)
        end
    end
    return gains
end

# Rebuild each plan's grid-indexed tables for the target samples asked for. The
# result is a `ComponentPlan` like any other — tables sized to those samples
# rather than to the solve grid, but still holding SOLVE-side segment ids, which
# is what makes the θ leaf and the coordinate state index correctly — so the
# forward map above is the same walk. Nothing is cached: the tables are
# recomputed per call, O(nchan + ntime) lookups.
_resolve_tree(nt::NamedTuple, solve, target, chan_idx, ti_idx, tspan) =
    map(v -> _resolve_node(v, solve, target, chan_idx, ti_idx, tspan), nt)
_resolve_node(nt::NamedTuple, solve, target, chan_idx, ti_idx, tspan) =
    _resolve_tree(nt, solve, target, chan_idx, ti_idx, tspan)

function _resolve_node(plan::ComponentPlan, solve, target, chan_idx, ti_idx, tspan)
    t = plan.term
    ax = term_axes(t)
    fseg = freq_segment_ids(plan.fseg, solve, target; chan_idx)
    tseg = time_segment_ids(plan.tseg, solve, target; ti_idx, time_span = tspan)
    xf = :Frequency in ax ?
        [freq_coordinate(t, target.channel_freqs[c], plan.fstate, fseg[k]) for (k, c) in enumerate(chan_idx)] :
        zeros(Float64, length(chan_idx))
    xt = :Ti in ax ?
        [time_coordinate(t, target.times[i], plan.tstate, tseg[k]) for (k, i) in enumerate(ti_idx)] :
        zeros(Float64, length(ti_idx))
    return ComponentPlan(
        t, plan.tseg, plan.fseg, tseg, fseg, xf, xt, plan.nchan_seg, plan.tying,
        plan.range, plan.shape, plan.fstate, plan.tstate,
    )
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
