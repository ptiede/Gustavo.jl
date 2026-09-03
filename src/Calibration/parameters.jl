# ── Parameter layout ─────────────────────────────────────────────────────────
#
# `plan_parameters` resolves one shared `StationGainModel`, replicated across
# `nant` antennas over a `DataGeometry`, into the structures a solve addresses:
#
#   * a `ComponentVector` `template` — θ named and shaped by component. Each
#     component's block space (parameters × feed-node × freq-segment ×
#     time-segment × antenna) is a shaped leaf under the component's name,
#     `θ.phase.<name>` / `θ.logamp.<name>`, so `component_vector` can wrap a solved
#     θ for named, shaped inspection.
#   * `plans` — the flat, depth-first list of `ComponentPlan`s (phase components
#     first, then log-amplitude); and `plantree`, the same plans nested under the
#     model's names. Each plan carries its term, segmentation metadata, and the
#     contiguous θ `range`/`shape` of its leaf, so the forward map and the solve
#     writers slice the block for a `(feed, ti, c)` cell directly out of θ.
#
# A component's leaf occupies exactly its `range` (column-major over parameters,
# feed-node, freq-segment, time-segment, antenna), so `reshape(view(θ, range),
# shape)` is the same array `component_vector` exposes under the component's name.
# The map recurses the `plantree` (names are compile-time constants) and slices θ
# by `range`, so it needs no offset tables, Dicts, or closures and stays
# type-stable and AD/Reactant-traceable.

"""
    ComponentPlan

The term, resolved segmentation, and θ location for one gain component. `range`
is the component's contiguous span in θ and `shape` its column-major shaped leaf
`(param, feed-node, freq-seg, time-seg, ant)`, so `reshape(view(θ, range),
shape)` recovers the leaf. The forward map reads `tseg_id[ti]`/`fseg_id[c]` to
place a cell in its segment, `xf`/`xt` for the term's coordinates, and
`nchan_seg` for the term's per-segment arity, then addresses the leaf block at
`(node, fs, ts, ant)` with `node = _feed_node(tying, feed)`. A block holds the
parameters `param_shapes(term, nchan_seg[fs])` declares, in declaration order.

Those four tables are indexed by position on the SOLVE GRID. `tseg`/`fseg` are
the segmentations they resolve, so the identical tables can be rebuilt for data
sampled anywhere by placing each foreign sample in its segment
([`time_segment_ids`](@ref) / [`freq_segment_ids`](@ref)) — the basis for
applying a solution to a different time or channel sampling. `fseg` is stored
[`materialize`](@ref)d, so a data-dependent segmentation never reaches a plan. `fstate`/`tstate`
are the term's own coordinate constants, resolved once against the solve
geometry ([`freq_coord_state`](@ref)), `nothing` on an axis the term does not
declare.
"""
struct ComponentPlan{
        T <: AbstractGainTerm, Ty <: AbstractFeedTying,
        TS <: AbstractTimeSegmentation, FS <: AbstractFrequencySegmentation, TC, FC,
    }
    term::T
    tseg::TS                    # the time segmentation `tseg_id` resolves
    fseg::FS                    # the frequency segmentation `fseg_id` resolves
    tseg_id::Vector{Int}        # length ntime  → time-segment id
    fseg_id::Vector{Int}        # length nchan  → freq-segment id
    xf::Vector{Float64}         # length nchan  → frequency coordinate
    xt::Vector{Float64}         # length ntime  → time coordinate
    nchan_seg::Vector{Int}      # length nfseg  → channels in the frequency segment
    tying::Ty
    range::UnitRange{Int}       # the component's contiguous θ span
    shape::NTuple{5, Int}       # (param, feed-node, freq-seg, time-seg, ant)
    fstate::FC                  # the term's resolved frequency-coordinate constants
    tstate::TC                  # …and its time-coordinate constants
end

"""
    ParameterLayout

The resolved plan for a solve: total length `nθ`, the grid dims, the number of
phase components `nphase`, the flat `plans` list and its `plantree` (the same
plans nested under the model's names), the named/shaped `template`
`ComponentVector`, and `axes` — a tree mirroring the model that records each
leaf's dimension sizes and the physical axis each dimension carries (`:Ant`,
`:Feed`, `:node`, `:Ti`, `:Frequency`, `:param`), for labelling a wrapped θ.
"""
struct ParameterLayout{PT, CV, AX}
    nθ::Int
    nant::Int
    ntime::Int
    nchan::Int
    nphase::Int
    plans::Vector{ComponentPlan}
    plantree::PT
    template::CV
    axes::AX
end

# The shaped leaf of a component in θ: `reshape(view(θ, range), shape)`, sharing
# data with θ. Writing into it writes through to θ. `shape` is a fixed rank-5
# tuple, so the reshape is type-stable.
@inline _component_leaf(plan::ComponentPlan, θ::AbstractVector) =
    reshape(view(θ, plan.range), plan.shape)

# The absolute 1-based θ index of a block's first parameter at
# `(node, fs, ts, ant)` — for the column-space station solver, which works in
# flat θ positions rather than shaped leaves.
@inline _block_index(plan::ComponentPlan, node::Int, fs::Int, ts::Int, ant::Int) =
    first(plan.range) + LinearIndices(plan.shape)[1, node, fs, ts, ant] - 1

# ── Per-component layout ─────────────────────────────────────────────────────
#
# One resolution of an `GainComponent` over a geometry: the segment ids and
# coordinates the `ComponentPlan` needs, plus the shaped-leaf description the
# template and range build from. The leaf `shape`/`roles` describe the block run
# as a fixed-rank column-major array: fastest to slowest over parameters,
# feed-node, frequency segment, time segment, then antenna, size-1 axes kept so
# every component reshapes to the same five axes and any consumer addresses it
# the same way. A term whose block length varies across frequency segments has no
# rectangular leaf and is rejected here.
function _component_layout(e::GainComponent, nant::Int, geom::DataGeometry)
    t = e.term
    tseg_id, ntseg = time_segment_ids(e.Ti, geom)
    # A data-dependent segmentation (`BandGroups`) resolves to its concrete form
    # here, so ids, plans, and provenance only ever hold materialized ones.
    fseg = materialize(e.Frequency, geom)
    fseg_id, nfseg = freq_segment_ids(fseg, geom)
    fseg_groups = segment_groups(fseg_id, nfseg)

    # Build only the axes the term declares; the rest stay zero. Calling a
    # builder solely for a declared axis means a term that declares one but is
    # missing its builder errors loudly (no silent zero-coordinate fallback) —
    # the key extensibility guard.
    axes = term_axes(t)
    all(in(TERM_AXES), axes) || throw(
        ArgumentError(
            "$(typeof(t)) declares unknown coordinate axes $(setdiff(axes, TERM_AXES)); " *
                "term_axes must be drawn from $TERM_AXES"
        )
    )
    fstate = :Frequency in axes ? freq_coord_state(t, geom, fseg_id, nfseg) : nothing
    tstate = :Ti in axes ? time_coord_state(t, geom, tseg_id, ntseg) : nothing
    xf = :Frequency in axes ?
        [freq_coordinate(t, geom.channel_freqs[c], fstate, fseg_id[c]) for c in eachindex(fseg_id)] :
        zeros(float(eltype(geom.channel_freqs)), nchannels(geom))
    xt = :Ti in axes ?
        [time_coordinate(t, geom.times[i], tstate, tseg_id[i]) for i in eachindex(tseg_id)] :
        zeros(float(eltype(geom.times)), ntimes(geom))

    # Channels per freq segment, and the block length each implies (only terms
    # whose arity comes from the data vary with it).
    nchan_seg = [length(grp) for grp in fseg_groups]
    blocklen = [nparams_per_block(t, n) for n in nchan_seg]

    nfeed = nfeed_blocks(e.Feed)
    bl = first(blocklen)                         # nchan_seg has one entry per segment (nfseg ≥ 1)
    all(==(bl), blocklen) || throw(
        ArgumentError(
            "$(typeof(t)): block length varies across frequency segments (blocklen = " *
                "$blocklen); the named-leaf layout requires one rectangular leaf per component."
        )
    )
    shape, roles = _leaf_shape(bl, nfeed, nfseg, ntseg, nant, e.Feed)

    return (;
        tseg = e.Ti, fseg,
        tseg_id, fseg_id, xf, xt, nchan_seg, tying = e.Feed, shape, roles, fstate, tstate,
    )
end

# Column-major leaf shape (fastest → slowest: parameters, feed-node, freq
# segment, time segment, antenna) and the physical role of each axis. Every leaf
# keeps all five axes, size-1 ones included, so the shape is a fixed-rank reshape
# of the component's θ block run: solve writers and the forward map address any
# component the same way, and each name retains its own complete
# `(param, node, Frequency, Ti, Ant)` role set for labelling a wrapped θ.
function _leaf_shape(bl, nfeed, nfseg, ntseg, nant, tying)
    node_role = tying isa PerFeed ? :Feed : :node
    return (bl, nfeed, nfseg, ntseg, nant), (:param, node_role, :Frequency, :Ti, :Ant)
end

# ── Named trees ──────────────────────────────────────────────────────────────
#
# Walk the model's named component tree so the template, plans, and axes nest
# exactly where the model does. The plans are built into both a flat depth-first
# list (`flat`) and the mirrored tree (returned), sharing a θ cursor (`next`) so a
# component's `range` matches the position of its leaf in the `ComponentVector`.
_template_tree(nt::NamedTuple, nant::Int, geom::DataGeometry) =
    map(v -> _template_node(v, nant, geom), nt)
_template_node(e::GainComponent, nant::Int, geom::DataGeometry) =
    zeros(_component_layout(e, nant, geom).shape...)
_template_node(nt::NamedTuple, nant::Int, geom::DataGeometry) = _template_tree(nt, nant, geom)

function _plans_tree(nt::NamedTuple, nant::Int, geom::DataGeometry, flat::Vector, next::Base.RefValue{Int})
    return NamedTuple{keys(nt)}(map(v -> _plans_node(v, nant, geom, flat, next), values(nt)))
end
function _plans_node(e::GainComponent, nant::Int, geom::DataGeometry, flat::Vector, next::Base.RefValue{Int})
    cl = _component_layout(e, nant, geom)
    dof = prod(cl.shape)
    range = next[]:(next[] + dof - 1)
    next[] += dof
    plan = ComponentPlan(
        e.term, cl.tseg, cl.fseg, cl.tseg_id, cl.fseg_id, cl.xf, cl.xt, cl.nchan_seg,
        cl.tying, range, cl.shape, cl.fstate, cl.tstate,
    )
    push!(flat, plan)
    return plan
end
_plans_node(nt::NamedTuple, nant::Int, geom::DataGeometry, flat::Vector, next::Base.RefValue{Int}) =
    _plans_tree(nt, nant, geom, flat, next)

_axes_tree(nt::NamedTuple, nant::Int, geom::DataGeometry) =
    map(v -> _axes_node(v, nant, geom), nt)
function _axes_node(e::GainComponent, nant::Int, geom::DataGeometry)
    cl = _component_layout(e, nant, geom)
    return (; dims = cl.shape, roles = cl.roles)
end
_axes_node(nt::NamedTuple, nant::Int, geom::DataGeometry) = _axes_tree(nt, nant, geom)

"""
    plan_parameters(model::StationGainModel, nant, geom::DataGeometry; require_nonempty = true) -> ParameterLayout

Resolve `model` (shared across `nant` antennas) over `geom` into a
`ParameterLayout`. The returned `nθ` is the length of the parameter vector that
`evaluate_gains` consumes.

`require_nonempty` runs [`validate_station_gain_model`](@ref) first, rejecting
a `model` with neither phase nor log-amplitude components — the right default
for a model meant to be solved. Pass `require_nonempty = false` when an empty
model is a legitimate, expected state (e.g. one pipeline step's own model,
which may legitimately compile no components while a sibling step's does);
the returned layout then simply has `nθ == 0`.
"""
function plan_parameters(model::StationGainModel, nant::Integer, geom::DataGeometry; require_nonempty::Bool = true)
    require_nonempty && validate_station_gain_model(model)
    nant = Int(nant)
    flat = ComponentPlan[]
    next = Ref(1)
    ptree = _plans_tree(model.phase, nant, geom, flat, next)
    nphase = length(flat)
    ltree = _plans_tree(model.logamp, nant, geom, flat, next)
    nθ = next[] - 1

    template = ComponentVector(
        phase = _template_tree(model.phase, nant, geom),
        logamp = _template_tree(model.logamp, nant, geom),
    )
    axes = (
        phase = _axes_tree(model.phase, nant, geom),
        logamp = _axes_tree(model.logamp, nant, geom),
    )
    length(template) == nθ || error(
        "plan_parameters: template length $(length(template)) disagrees with nθ = $nθ; " *
            "the named leaves and the component ranges must span the same θ."
    )
    return ParameterLayout(
        nθ, nant, ntimes(geom), nchannels(geom), nphase,
        flat, (; phase = ptree, logamp = ltree), template, axes,
    )
end

"""
    component_vector(layout::ParameterLayout, θ::AbstractVector) -> ComponentVector

Wrap a flat θ (length `layout.nθ`) as the layout's named, shaped
`ComponentVector`, so `cv.phase.<name>` / `cv.logamp.<name>` view each
component's block as a labelled array. The wrap shares data with `θ` (no copy);
use it for inspection.
"""
function component_vector(layout::ParameterLayout, θ::AbstractVector)
    length(θ) == layout.nθ || throw(
        DimensionMismatch(
            "component_vector: θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)"
        )
    )
    return ComponentArray(θ, getaxes(layout.template))
end
