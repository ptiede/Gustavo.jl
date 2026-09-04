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

Those four tables are indexed by position on the solve GRID. `tseg`/`fseg` are
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

# ── Station-signature canonicalization ───────────────────────────────────────
#
# A heterogeneous model (per-station trees differing in components or
# resolution) canonicalizes before layout: per component name, stations are
# grouped by identical `(term, Ti, Frequency, Feed)` signature, and each group
# gets its own rectangular leaf whose `:Ant` axis spans just that group's
# stations. θ is ragged across groups; segment lookup is per (station, time)
# through each group's own plan tables. A name whose single signature covers
# every station stays a plain `ComponentPlan` — the grouped machinery has zero
# footprint until a model actually differs across stations.

# One merged name's signature groups, before layout: the distinct components
# and, per component, the global station indices carrying it (in antenna
# order, so the grouping is deterministic).
struct _ComponentGroups
    comps::Vector{Any}             # GainComponents, one per signature group
    stations::Vector{Vector{Int}}  # global station indices per group
end

# Merge per-station component trees into one canonical name tree. `vals[i]` is
# the subtree station `members[i]` carries at `path`; keys appear in
# first-seen station order. A key whose values are all components groups by
# signature; all-`NamedTuple` values recurse; a mix is a structural conflict
# and errors — there is no honest layout for a name that is a leaf at one
# station and a subtree at another.
function _merge_station_trees(vals::Vector{Any}, members::Vector{Int}, nant::Int, path)
    ks = Symbol[]
    for v in vals, k in keys(v)
        k in ks || push!(ks, k)
    end
    out = Pair{Symbol, Any}[]
    for k in ks
        idx = [i for i in eachindex(vals) if haskey(vals[i], k)]
        sub = Any[vals[i][k] for i in idx]
        stn = members[idx]
        node = if all(x -> x isa GainComponent, sub)
            comps = Any[]
            groups = Vector{Int}[]
            for (x, a) in zip(sub, stn)
                gi = findfirst(==(x), comps)
                if gi === nothing
                    push!(comps, x)
                    push!(groups, [a])
                else
                    push!(groups[gi], a)
                end
            end
            length(comps) == 1 && length(stn) == nant ? comps[1] :
                _ComponentGroups(comps, groups)
        elseif all(x -> x isa NamedTuple, sub)
            _merge_station_trees(sub, stn, nant, (path..., k))
        else
            throw(
                ArgumentError(
                    "station trees disagree structurally at " *
                        "`$(join((path..., k), '.'))`: a component at some stations, " *
                        "a named subtree at others.",
                ),
            )
        end
        push!(out, k => node)
    end
    return NamedTuple(out)
end

"""
    GroupedComponentPlan

The plantree node of one station-heterogeneous component name: one
[`ComponentPlan`](@ref) per signature group under `groups` (keys `g1, g2, …`,
in first-station order), each plan's `:Ant` axis spanning just its group.
`stations[i]` holds group `i`'s global station indices; `group_of`/`local_of`
map a global station index to its `(group, within-group)` position (`0` when
the station carries this component in no group, contributing nothing). The
forward map routes each station through its own group's plan, so segment
lookup is per (station, time).
"""
struct GroupedComponentPlan{P <: NamedTuple}
    groups::P
    stations::Vector{Vector{Int}}
    group_of::Vector{Int}
    local_of::Vector{Int}
end

_group_keys(n::Int) = ntuple(i -> Symbol(:g, i), n)

# ── Named trees ──────────────────────────────────────────────────────────────
#
# Walk the model's named component tree so the template, plans, and axes nest
# exactly where the model does. The plans are built into both a flat depth-first
# list (`flat`) and the mirrored tree (returned), sharing a θ cursor (`next`) so a
# component's `range` matches the position of its leaf in the `ComponentVector`.
# Leaves are `GainComponent`s, or `_ComponentGroups` where the canonicalizer
# found station heterogeneity.
_template_tree(nt::NamedTuple, nant::Int, geom::DataGeometry) =
    map(v -> _template_node(v, nant, geom), nt)
_template_node(e::GainComponent, nant::Int, geom::DataGeometry) =
    zeros(_component_layout(e, nant, geom).shape...)
_template_node(nt::NamedTuple, nant::Int, geom::DataGeometry) = _template_tree(nt, nant, geom)
_template_node(g::_ComponentGroups, nant::Int, geom::DataGeometry) =
    NamedTuple{_group_keys(length(g.comps))}(
    Tuple(
        _template_node(g.comps[i], length(g.stations[i]), geom)
            for i in eachindex(g.comps)
    )
)

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
function _plans_node(g::_ComponentGroups, nant::Int, geom::DataGeometry, flat::Vector, next::Base.RefValue{Int})
    plans = NamedTuple{_group_keys(length(g.comps))}(
        Tuple(
            _plans_node(g.comps[i], length(g.stations[i]), geom, flat, next)
                for i in eachindex(g.comps)
        )
    )
    group_of = zeros(Int, nant)
    local_of = zeros(Int, nant)
    for (gi, sts) in pairs(g.stations), (li, a) in pairs(sts)
        group_of[a] = gi
        local_of[a] = li
    end
    return GroupedComponentPlan(plans, g.stations, group_of, local_of)
end

_axes_tree(nt::NamedTuple, nant::Int, geom::DataGeometry) =
    map(v -> _axes_node(v, nant, geom), nt)
function _axes_node(e::GainComponent, nant::Int, geom::DataGeometry)
    cl = _component_layout(e, nant, geom)
    return (; dims = cl.shape, roles = cl.roles)
end
_axes_node(nt::NamedTuple, nant::Int, geom::DataGeometry) = _axes_tree(nt, nant, geom)
# A group leaf's axes node additionally records the global station indices its
# `:Ant` axis spans, so a wrapped θ can be labelled with the group's stations
# rather than the run's full antenna list.
_axes_node(g::_ComponentGroups, nant::Int, geom::DataGeometry) =
    NamedTuple{_group_keys(length(g.comps))}(
    Tuple(
        (; _axes_node(g.comps[i], length(g.stations[i]), geom)..., stations = g.stations[i])
            for i in eachindex(g.comps)
    )
)

"""
    plan_parameters(model::StationGainModel, nant, geom::DataGeometry; require_nonempty = true) -> ParameterLayout
    plan_parameters(model::AbstractGainModel, antennas, geom::DataGeometry; require_nonempty = true) -> ParameterLayout

Resolve `model` over `geom` into a `ParameterLayout`. The returned `nθ` is the
length of the parameter vector that `evaluate_gains` consumes.

The `nant::Integer` form lays out a station-uniform `StationGainModel` (empty
`stations` — the codes could not be resolved otherwise) replicated across
`nant` antennas. The `antennas` form (an `AntennaTable` or an iterable of
station codes) [`materialize`](@ref)s any `AbstractGainModel` against the
station set first and handles heterogeneity: per component name, stations
group by identical `(term, Ti, Frequency, Feed)` signature, a name with one
signature covering every station lays out exactly as the uniform form does,
and a heterogeneous name becomes a [`GroupedComponentPlan`](@ref) with one
ragged leaf per group (`θ.phase.<name>.g1`, `.g2`, …). Iterate the groups with
[`station_blocks`](@ref).

`require_nonempty` runs `validate_station_gain_model` first, rejecting
a `model` with neither phase nor log-amplitude components — the right default
for a model meant to be solved. Pass `require_nonempty = false` when an empty
model is a legitimate, expected state (e.g. one pipeline step's own model,
which may legitimately compile no components while a sibling step's does);
the returned layout then simply has `nθ == 0`.
"""
function plan_parameters(model::StationGainModel, nant::Integer, geom::DataGeometry; require_nonempty::Bool = true)
    require_nonempty && validate_station_gain_model(model)
    isempty(model.stations) || throw(
        ArgumentError(
            "plan_parameters with an antenna COUNT cannot resolve the model's " *
                "`stations` codes; pass the antenna table (or the station codes) instead.",
        ),
    )
    return _plan_layout(model.phase, model.logamp, Int(nant), geom)
end

function plan_parameters(model::AbstractGainModel, antennas, geom::DataGeometry; require_nonempty::Bool = true)
    names = _station_names(antennas)
    mat = materialize(model, names, geom)
    require_nonempty && validate_station_gain_model(mat)
    nant = length(names)
    isempty(mat.stations) &&
        return plan_parameters(mat, nant, geom; require_nonempty = false)
    trees = [station_components(mat, n) for n in names]
    members = collect(1:nant)
    ptree = _merge_station_trees(Any[t.phase for t in trees], members, nant, (:phase,))
    ltree = _merge_station_trees(Any[t.logamp for t in trees], members, nant, (:logamp,))
    return _plan_layout(ptree, ltree, nant, geom)
end

function _plan_layout(ptree_spec::NamedTuple, ltree_spec::NamedTuple, nant::Int, geom::DataGeometry)
    flat = ComponentPlan[]
    next = Ref(1)
    ptree = _plans_tree(ptree_spec, nant, geom, flat, next)
    nphase = length(flat)
    ltree = _plans_tree(ltree_spec, nant, geom, flat, next)
    nθ = next[] - 1

    template = ComponentVector(
        phase = _template_tree(ptree_spec, nant, geom),
        logamp = _template_tree(ltree_spec, nant, geom),
    )
    axes = (
        phase = _axes_tree(ptree_spec, nant, geom),
        logamp = _axes_tree(ltree_spec, nant, geom),
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

# ── Station blocks: the public iteration surface of a (possibly ragged) leaf ──

"""
    station_blocks(layout::ParameterLayout, θ, path::Symbol...) -> Vector

The station blocks of one component name, the unit a heterogeneity-aware
solver loops over. `path` descends the layout's plantree starting at `:phase`
or `:logamp` (e.g. `station_blocks(layout, θ, :phase, :bandpass)`); each
returned block is `(; stations, θ, plan)`:

- `stations` — the global station indices this block's `:Ant` axis spans, in
  axis order;
- `θ` — the block's shaped leaf, a view into the given `θ` with axes
  `(param, node, Frequency, Ti, Ant)`;
- `plan` — the block's [`ComponentPlan`](@ref), carrying the group's own
  segment-id tables and coordinates (segment lookup is per station through its
  block's plan).

A station-uniform component yields exactly one block spanning every station.

Gauge note: the blocks of one name share a single physical degeneracy (one
common offset across all stations carrying the component), not one per block.
A solver must place its gauge constraint once across the union of the blocks'
`stations` — constraining each block separately over-constrains the solve, and
`stations` holds global indices precisely so a run-wide gauge
([`AbstractGauge`](@ref)) resolves against them unchanged.
"""
function station_blocks(layout::ParameterLayout, θ::AbstractVector, path::Symbol...)
    length(θ) == layout.nθ || throw(
        DimensionMismatch(
            "station_blocks: θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)"
        )
    )
    node = layout.plantree
    for s in path
        children = node isa GroupedComponentPlan ? node.groups : node
        avail = children isa NamedTuple ? join(keys(children), ", ") :
            "none — the path already names a component"
        (children isa NamedTuple && haskey(children, s)) || throw(
            ArgumentError(
                "station_blocks: no component at path $(join(path, '.')); " *
                    "available here: $avail.",
            ),
        )
        node = children[s]
    end
    return _station_blocks(node, layout.nant, θ, path)
end

_station_blocks(plan::ComponentPlan, nant::Int, θ, path) =
    [(; stations = collect(1:nant), θ = _component_leaf(plan, θ), plan)]
_station_blocks(g::GroupedComponentPlan, nant::Int, θ, path) = [
    (; stations = g.stations[i], θ = _component_leaf(p, θ), plan = p)
        for (i, p) in enumerate(values(g.groups))
]
function _station_blocks(nt::NamedTuple, nant::Int, θ, path)
    ks = join(keys(nt), ", ")
    throw(
        ArgumentError(
            "station_blocks: path $(join(path, '.')) names a component subtree " *
                "(keys: $ks); descend to a component.",
        ),
    )
end

# ── Station-uniformity enforcement ───────────────────────────────────────────

# The heterogeneous leaves of a materialized model over `nant` stations:
# `(path, groups)` pairs for every component name with more than one signature
# or with stations that lack it. Empty exactly when every consumer may treat
# the model as uniform.
function _station_heterogeneity(mat::StationGainModel, names)
    nant = length(names)
    isempty(mat.stations) && return Tuple{Tuple, _ComponentGroups}[]
    trees = [station_components(mat, n) for n in names]
    members = collect(1:nant)
    het = Tuple{Tuple, _ComponentGroups}[]
    _hetero_leaves!(het, _merge_station_trees(Any[t.phase for t in trees], members, nant, (:phase,)), (:phase,))
    _hetero_leaves!(het, _merge_station_trees(Any[t.logamp for t in trees], members, nant, (:logamp,)), (:logamp,))
    return het
end

function _hetero_leaves!(out, node::NamedTuple, path)
    for k in keys(node)
        _hetero_leaves!(out, node[k], (path..., k))
    end
    return out
end
_hetero_leaves!(out, ::GainComponent, path) = out
_hetero_leaves!(out, g::_ComponentGroups, path) = push!(out, (path, g))

"""
    require_station_uniform(model::StationGainModel, antennas, who) -> model

Assert that the materialized `model` assigns every station the same component
signatures, throwing otherwise with the differing components, their
signatures, and the stations carrying each. `who` names the consumer in the
error — the framework calls this for a solve step that has not opted into
heterogeneity (`supports_station_heterogeneity`).
"""
function require_station_uniform(model::StationGainModel, antennas, who::AbstractString)
    names = _station_names(antennas)
    het = _station_heterogeneity(model, names)
    isempty(het) && return model
    lines = String[]
    for (path, g) in het
        covered = reduce(vcat, g.stations)
        sigs = join(
            (
                "[" * join(names[g.stations[i]], ", ") * "]: " * component_label(g.comps[i])
                    for i in eachindex(g.comps)
            ), "; ",
        )
        absent = setdiff(1:length(names), covered)
        isempty(absent) || (sigs *= "; absent at [" * join(names[absent], ", ") * "]")
        push!(
            lines,
            "  $(join(path, '.')): $(length(g.comps)) signature(s) — $sigs",
        )
    end
    throw(
        ArgumentError(
            "$who supports only station-uniform gain models (it does not declare " *
                "`supports_station_heterogeneity`), but the compiled model differs " *
                "across stations:\n" * join(lines, "\n"),
        ),
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
