# ── Parameter layout ─────────────────────────────────────────────────────────
#
# `plan_parameters` flattens one shared `StationGainModel`, replicated across
# `nant` antennas over a `DataGeometry`, into a single parameter vector θ and the
# structures needed to address it. Two views of the same contiguous layout are
# built together and kept consistent:
#
#   * a `ComponentVector` `template` — θ named and shaped by component. The block
#     space of each component (parameters × feed-node × freq-segment × time-segment
#     × antenna) is a leaf array under the component's name, `θ.phase.<name>` /
#     `θ.logamp.<name>`, so a solved θ can be wrapped for named, shaped inspection.
#   * the per-component `off1`/`off2` integer tables the hot forward loop reads.
#
# Both derive from the same depth-first block assignment, so a component's named
# leaf occupies exactly the θ range its `off1`/`off2` entries point into (checked
# when the layout is built). The forward loop stays on integer offsets — no Dicts,
# Strings, or closures in the structures it reads — so it is type-stable and
# Reactant-traceable.

"""
    ComponentPlan

Resolved index tables for one gain component (one entry per phase component, then
per log-amplitude component). `off1`/`off2` give the 1-based start index in θ of
the primary and secondary parameter block for `(ant, feed, time_segment,
freq_segment)`; `0` means "no contribution". (`off2` is non-zero only for the
partner feed under `ReferenceRelative`, where the value is reference + relative.)
A block holds the parameters `param_shapes(term, nchan_seg[freq_segment])`
declares, laid out in the order the names are declared in.
"""
struct ComponentPlan
    axes::Vector{Symbol}        # the coordinate axes the term reads
    tseg_id::Vector{Int}        # length ntime  → time-segment id
    fseg_id::Vector{Int}        # length nchan  → freq-segment id
    xf::Vector{Float64}         # length nchan  → frequency coordinate
    xt::Vector{Float64}         # length ntime  → time coordinate
    nchan_seg::Vector{Int}      # length nfseg  → channels in the frequency segment
    off1::Array{Int, 4}         # (ant, feed, ntseg, nfseg)
    off2::Array{Int, 4}
end

"""
    ParameterLayout

The flattened parameter plan for a solve: total length `nθ`, the grid dims, one
`ComponentPlan` per component (phase components first, then log-amplitude), the
named/shaped `template` `ComponentVector`, and `axes` — a tree mirroring the
model that records each component leaf's dimension sizes and the physical axis
each dimension carries (`:Ant`, `:Feed`, `:Ti`, `:Frequency`, `:param`), for
labelling a wrapped θ.
"""
struct ParameterLayout{CV, AX}
    nθ::Int
    nant::Int
    ntime::Int
    nchan::Int
    nphase::Int
    plans::Vector{ComponentPlan}
    template::CV
    axes::AX
end

# ── Per-component layout ─────────────────────────────────────────────────────
#
# One resolution of a `TiedComponent` over a geometry: the segment ids and
# coordinates the `ComponentPlan` needs, plus the block bookkeeping the θ leaf
# and the offset tables both derive from. The leaf `shape`/`roles` describe the
# contiguous block run as a column-major array: fastest to slowest the block runs
# over parameters, feed-node, frequency segment, time segment, then antenna, and
# a size-1 axis among these is dropped. A component whose block length varies
# across frequency segments has no rectangular shape, so its leaf is a flat vector
# (`ragged`).
function _component_layout(tc::TiedComponent, nant::Int, geom::DataGeometry)
    t = term(tc)
    tseg_id, ntseg = time_segment_ids(time_segmentation(tc), geom)
    fseg_id, nfseg = freq_segment_ids(freq_segmentation(tc), geom)
    fseg_groups = segment_groups(fseg_id, nfseg)
    tseg_groups = segment_groups(tseg_id, ntseg)

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
    xf = :Frequency in axes ? freq_coordinate(t, geom.channel_freqs, fseg_groups, geom.f0) :
        zeros(Float64, nchannels(geom))
    xt = :Ti in axes ? time_coordinate(t, geom.times, tseg_groups, geom.t0) :
        zeros(Float64, ntimes(geom))

    # Channels per freq segment, and the block length each implies (only terms
    # whose arity comes from the data vary with it).
    nchan_seg = [length(grp) for grp in fseg_groups]
    blocklen = [nparams_per_block(t, n) for n in nchan_seg]

    nfeed = nfeed_blocks(tc.tying)
    dof = nant * ntseg * nfeed * sum(blocklen)
    bl = first(blocklen)                         # nchan_seg has one entry per segment (nfseg ≥ 1)
    ragged = !all(==(bl), blocklen)
    shape, roles = _leaf_shape(dof, ragged, bl, nfeed, nfseg, ntseg, nant, tc.tying)

    return (;
        axes, tseg_id, fseg_id, xf, xt, nchan_seg,
        blocklen, ntseg, nfseg, tying = tc.tying, dof, shape, roles,
    )
end

# Column-major leaf shape (fastest → slowest: parameters, feed-node, freq
# segment, time segment, antenna) with size-1 axes dropped, and the physical
# role of each retained axis. A ragged or empty component has no rectangular
# shape, so its leaf is a flat vector.
function _leaf_shape(dof, ragged, bl, nfeed, nfseg, ntseg, nant, tying)
    (ragged || dof == 0) && return (dof,), (:flat,)
    node_role = tying isa PerFeed ? :Feed : :node
    dims = ((bl, :param), (nfeed, node_role), (nfseg, :Frequency), (ntseg, :Ti), (nant, :Ant))
    kept = filter(d -> first(d) != 1, dims)
    isempty(kept) && return (dof,), (:scalar,)     # every axis size 1 (dof == 1)
    return Tuple(first(d) for d in kept), Tuple(last(d) for d in kept)
end

# Build a `ComponentPlan` and return it together with the updated θ cursor.
function _plan_component(tc::TiedComponent, nant::Int, geom::DataGeometry, next::Int)
    cl = _component_layout(tc, nant, geom)
    off1 = zeros(Int, nant, 2, cl.ntseg, cl.nfseg)
    off2 = zeros(Int, nant, 2, cl.ntseg, cl.nfseg)
    for ant in 1:nant, ts in 1:cl.ntseg, fs in 1:cl.nfseg
        bl = cl.blocklen[fs]
        bl == 0 && continue
        next = _assign_blocks!(off1, off2, cl.tying, ant, ts, fs, bl, next)
    end

    plan = ComponentPlan(
        collect(Symbol, cl.axes), cl.tseg_id, cl.fseg_id, cl.xf, cl.xt, cl.nchan_seg, off1, off2
    )
    return plan, next
end

function _assign_blocks!(off1, off2, ::PerFeed, ant, ts, fs, bl, next)
    off1[ant, 1, ts, fs] = next; next += bl
    off1[ant, 2, ts, fs] = next; next += bl
    return next
end

function _assign_blocks!(off1, off2, ::SharedFeeds, ant, ts, fs, bl, next)
    off1[ant, 1, ts, fs] = next
    off1[ant, 2, ts, fs] = next
    next += bl
    return next
end

function _assign_blocks!(off1, off2, tying::FeedComponent, ant, ts, fs, bl, next)
    off1[ant, tying.feed, ts, fs] = next
    next += bl
    return next
end

function _assign_blocks!(off1, off2, tying::ReferenceRelative, ant, ts, fs, bl, next)
    rf = tying.reference_feed
    partner = 3 - rf
    ref = next; next += bl
    rel = next; next += bl
    off1[ant, rf, ts, fs] = ref
    off1[ant, partner, ts, fs] = ref
    off2[ant, partner, ts, fs] = rel
    return next
end

# ── Named/shaped template ────────────────────────────────────────────────────
#
# Walk the model's named component tree (not the flattened list) so the template
# nests exactly where the model does; its depth-first leaf order equals the
# flat-list order the plans and offsets use.
_template_tree(nt::NamedTuple, nant::Int, geom::DataGeometry) =
    map(v -> _template_node(v, nant, geom), nt)
_template_node(tc::TiedComponent, nant::Int, geom::DataGeometry) =
    zeros(_component_layout(tc, nant, geom).shape...)
_template_node(nt::NamedTuple, nant::Int, geom::DataGeometry) = _template_tree(nt, nant, geom)

_axes_tree(nt::NamedTuple, nant::Int, geom::DataGeometry) =
    map(v -> _axes_node(v, nant, geom), nt)
function _axes_node(tc::TiedComponent, nant::Int, geom::DataGeometry)
    cl = _component_layout(tc, nant, geom)
    return (; dims = cl.shape, roles = cl.roles)
end
_axes_node(nt::NamedTuple, nant::Int, geom::DataGeometry) = _axes_tree(nt, nant, geom)

"""
    plan_parameters(model::StationGainModel, nant, geom::DataGeometry) -> ParameterLayout

Flatten `model` (shared across `nant` antennas) over `geom` into a
`ParameterLayout`. The returned `nθ` is the length of the parameter vector that
`evaluate_gains` consumes.
"""
function plan_parameters(model::StationGainModel, nant::Integer, geom::DataGeometry)
    validate_station_gain_model(model)
    nant = Int(nant)
    plans = ComponentPlan[]
    next = 1
    for tc in phase_components(model)
        plan, next = _plan_component(tc, nant, geom, next)
        push!(plans, plan)
    end
    nphase = length(plans)
    for tc in logamp_components(model)
        plan, next = _plan_component(tc, nant, geom, next)
        push!(plans, plan)
    end
    nθ = next - 1

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
            "the named leaves and the offset tables must span the same θ."
    )
    return ParameterLayout(nθ, nant, ntimes(geom), nchannels(geom), nphase, plans, template, axes)
end

"""
    component_vector(layout::ParameterLayout, θ::AbstractVector) -> ComponentVector

Wrap a flat θ (length `layout.nθ`) as the layout's named, shaped
`ComponentVector`, so `cv.phase.<name>` / `cv.logamp.<name>` view each
component's block as a labelled array. The wrap shares data with `θ` (no copy);
use it for inspection, not on the solve/AD hot path.
"""
function component_vector(layout::ParameterLayout, θ::AbstractVector)
    length(θ) == layout.nθ || throw(
        DimensionMismatch(
            "component_vector: θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)"
        )
    )
    return ComponentArray(θ, getaxes(layout.template))
end
