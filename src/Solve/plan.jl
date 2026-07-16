# ── Gain plan + per-site parameter container (Stage 2, Milestone 2) ───────────
#
# The forward map θ → gains is re-expressed for the global solver in TWO pieces:
#
#   • `GainPlan`   — the AD-INACTIVE constant tables (segment ids, coordinate
#                    axes, feed-block routing), with antennas GROUPED BY IDENTICAL
#                    MODEL STRUCTURE so each group evaluates under one concrete
#                    compile-time `StationGainModel` type. Built once by
#                    `plan_gains`; never differentiated.
#   • `GainParams` — the differentiable `ComponentVector` of per-site 2D-array
#                    parameter blocks. A `ComponentVector` is simultaneously the
#                    flat ℝⁿ vector the optimizer / LogDensityProblems consumes
#                    AND a named per-group / per-component structure of arrays.
#
# This replaces `Calibration`'s flat-θ + `off1/off2` offset tables
# (`parameters.jl`/`evaluate.jl`): the same `term_eval` scalar map and the same
# segment/coordinate builders are reused, but a term's parameters now live in a
# named per-group array `A[nparam, ntseg, nfseg, nfeedblock, nant_in_group]`
# indexed directly, rather than through a global offset. Grouping keeps the tuple
# recursion over components (`_sum_site`) `@inferred` even for heterogeneous
# arrays (a few distinct per-site structures) without one specialization per
# antenna — see the milestone tests.

# ── Per-component constant tables ─────────────────────────────────────────────

"""
    SiteComponent

The AD-inactive tables one gain component contributes to the forward map, plus
the shape of its parameter block. Mirrors `Calibration.ComponentPlan` but
addresses a per-group array (feed → feed-block routing) instead of a flat offset.

- `tseg_id`/`fseg_id` : segment id per time sample / global channel.
- `xf`/`xt`/`clocal`  : the coordinate scalars `term_eval` reads.
- `fb1`/`fb2`         : `feed (1,2) → feed-block index` (0 = no contribution).
  `fb2` is nonzero only for the partner feed under `ReferenceRelative`
  (partner = reference block `fb1` + relative block `fb2`).
- `nparam`/`ntseg`/`nfseg`/`nfb` : parameter-array dims (the first four axes of
  the component's per-group array; the fifth is the antenna axis). `nparam` is
  the MAX block length over frequency segments (only ragged `PerChannel` over
  uneven spws pads; padded slots are never read).
"""
struct SiteComponent{T <: AbstractGainTerm}
    term::T
    tseg_id::Vector{Int}
    fseg_id::Vector{Int}
    xf::Vector{Float64}
    xt::Vector{Float64}
    clocal::Vector{Int}
    fb1::NTuple{2, Int}
    fb2::NTuple{2, Int}
    nparam::Int
    ntseg::Int
    nfseg::Int
    nfb::Int
end

# feed (1,2) → (primary, secondary) feed-block index, matching the flat
# `_assign_blocks!` conventions in Calibration/parameters.jl exactly.
_feedblock_tables(::PerFeed) = ((1, 2), (0, 0))
_feedblock_tables(::SharedFeeds) = ((1, 1), (0, 0))
_feedblock_tables(t::FeedComponent) = t.feed == 1 ? ((1, 0), (0, 0)) : ((0, 1), (0, 0))
function _feedblock_tables(t::ReferenceRelative)
    rf = t.reference_feed
    partner = 3 - rf
    fb1 = (1, 1)                              # both feeds read the reference block
    fb2 = partner == 2 ? (0, 2) : (2, 0)     # only the partner feed reads the relative block
    return fb1, fb2
end

# Build a SiteComponent from a TiedComponent over a geometry. Reuses the exact
# segmentation / coordinate builders `Calibration.plan_parameters` uses.
function _site_component(tc::TiedComponent, geom::DataGeometry)
    t = term(tc)
    tseg_id, ntseg = time_segment_ids(time_segmentation(tc), geom)
    fseg_id, nfseg = freq_segment_ids(freq_segmentation(tc), geom)
    fseg_groups = segment_groups(fseg_id, nfseg)
    tseg_groups = segment_groups(tseg_id, ntseg)

    ck = coord_kind(t)
    xf = ck == COORD_FREQ ?
        freq_coordinate(t, geom.channel_freqs, fseg_groups, geom.f0) :
        zeros(Float64, length(geom.channel_freqs))
    xt = ck == COORD_TIME ?
        time_coordinate(t, geom.times, tseg_groups, geom.t0) :
        zeros(Float64, length(geom.times))

    nchan = length(geom.channel_freqs)
    clocal = Vector{Int}(undef, nchan)
    counter = zeros(Int, nfseg)
    @inbounds for c in 1:nchan
        f = fseg_id[c]
        counter[f] += 1
        clocal[c] = counter[f]
    end

    blocklens = [nparams_per_block(t, length(grp)) for grp in fseg_groups]
    nparam = isempty(blocklens) ? 0 : maximum(blocklens)
    fb1, fb2 = _feedblock_tables(tc.tying)
    nfb = nfeed_blocks(tc.tying)
    return SiteComponent(t, tseg_id, fseg_id, xf, xt, clocal, fb1, fb2, nparam, ntseg, nfseg, nfb)
end

# ── Per-group plan ────────────────────────────────────────────────────────────

"""
    GroupPlan

One antenna group (all sharing an identical `StationGainModel`). `groupval` is
`Val(:g<k>)` (the ComponentVector field key for this group's parameters);
`ants` the group's global antenna indices; `phase`/`logamp` the tuple of
[`SiteComponent`](@ref)s (parallel to `model.phase`/`model.logamp`); `phase_syms`
/`logamp_syms` the `Val(:phase_j)` / `Val(:logamp_j)` keys used to pull each
component's array out of the group's parameters type-stably.
"""
struct GroupPlan{GV, M <: StationGainModel, P <: Tuple, A <: Tuple, PS <: Tuple, AS <: Tuple}
    groupval::GV
    model::M
    ants::Vector{Int}
    phase::P
    logamp::A
    phase_syms::PS
    logamp_syms::AS
end

"""
    GainPlan

The AD-inactive forward-map plan: grid dims (`nant`/`ntime`/`nchan`), the tuple
of [`GroupPlan`](@ref)s (heterogeneous — one per distinct model structure), and a
zero `template` ComponentVector defining the parameter axes.
"""
struct GainPlan{G <: Tuple, CV}
    nant::Int
    ntime::Int
    nchan::Int
    groups::G
    template::CV
end

nparameters(plan::GainPlan) = length(plan.template)

"""
    plan_gains(model, nant, geom::DataGeometry) -> GainPlan

Plan the whole-array forward map. `model` is an [`ArrayGainModel`](@ref) (or a
bare `StationGainModel`, treated as a homogeneous array). Antennas `1:nant` are
grouped by identical model structure; each group gets its `SiteComponent` tables
and a per-group block of parameter arrays in the returned plan's `template`.
"""
plan_gains(model::StationGainModel, nant::Integer, geom::DataGeometry) =
    plan_gains(ArrayGainModel(model), nant, geom)

function plan_gains(am::ArrayGainModel, nant::Integer, geom::DataGeometry)
    nant = Int(nant)
    models, groupants = _group_by_model(am, nant)
    ngroup = length(models)

    groups = Vector{Any}(undef, ngroup)
    group_templates = Vector{Pair{Symbol, Any}}(undef, ngroup)
    for g in 1:ngroup
        m = models[g]
        validate_station_gain_model(m)
        ants = groupants[g]
        na = length(ants)
        phase = map(tc -> _site_component(tc, geom), phase_components(m))
        logamp = map(tc -> _site_component(tc, geom), logamp_components(m))
        # Component keys in the parameter ComponentVector come from the model's
        # component names (`p.<group>.<name>`), so the parameters are
        # self-documenting; unique across phase + logamp (checked at model build).
        pnames = phase_component_names(m)
        anames = logamp_component_names(m)
        phase_syms = map(Val, pnames)
        logamp_syms = map(Val, anames)
        groupsym = Symbol(:g, g)

        # This group's parameter block: one named array per component,
        # A[nparam, ntseg, nfseg, nfeedblock, nant_in_group].
        pairs = Pair{Symbol, Array{Float64, 5}}[]
        for (j, c) in enumerate(phase)
            push!(pairs, pnames[j] => zeros(Float64, c.nparam, c.ntseg, c.nfseg, c.nfb, na))
        end
        for (j, c) in enumerate(logamp)
            push!(pairs, anames[j] => zeros(Float64, c.nparam, c.ntseg, c.nfseg, c.nfb, na))
        end
        group_templates[g] = groupsym => (; pairs...)
        groups[g] = GroupPlan(Val(groupsym), m, ants, phase, logamp, phase_syms, logamp_syms)
    end

    template = ComponentVector((; (group_templates...,)...))
    return GainPlan(nant, length(geom.times), length(geom.channel_freqs), (groups...,), template)
end

# ── Parameter container helpers ──────────────────────────────────────────────

"""
    zero_params(plan::GainPlan) -> ComponentVector

A fresh zero-filled parameter `ComponentVector` matching `plan`'s layout. Fill
it (e.g. from a warm start) then optimize.
"""
zero_params(plan::GainPlan) = zero(plan.template)

"""
    flatten(p::ComponentVector) -> Vector

The underlying flat ℝⁿ parameter vector (what the optimizer sees). Shares
storage with `p`; `copy` it if you need an independent vector.
"""
flatten(p::ComponentVector) = getdata(p)

"""
    unflatten(plan::GainPlan, x::AbstractVector) -> ComponentVector

Wrap a flat ℝⁿ vector back into the named per-site parameter structure for
`plan`. Inverse of [`flatten`](@ref).
"""
unflatten(plan::GainPlan, x::AbstractVector) = ComponentVector(x, getaxes(plan.template))

"""
    group_arrays(plan::GainPlan, p, g::Integer) -> (phase_arrays, logamp_arrays)

The tuple of per-component parameter arrays (phase, then log-amplitude) for group
`g` of `plan`, viewed out of `p`. Each array is
`A[nparam, ntseg, nfseg, nfeedblock, nant_in_group]`. Handy for warm-start
seeding and diagnostics.
"""
function group_arrays(plan::GainPlan, p, g::Integer)
    gp = plan.groups[g]
    pg = _sub(p, gp.groupval)
    return _pull(pg, gp.phase_syms), _pull(pg, gp.logamp_syms)
end

# ── Grouped forward evaluation ────────────────────────────────────────────────

@inline _sym(::Val{S}) where {S} = S
@inline _sub(p, ::Val{S}) where {S} = getproperty(p, S)
@inline _pull(pg, syms::Tuple) = map(v -> getproperty(pg, _sym(v)), syms)

# Linear index of A[1, ts, fs, fb, la] in the column-major component array (the
# parameter axis is fastest), so `ParamBlock(A, off)` indexes the block.
@inline _block_offset(c::SiteComponent, ts, fs, fb, la) =
    1 + c.nparam * ((ts - 1) + c.ntseg * ((fs - 1) + c.nfseg * ((fb - 1) + c.nfb * (la - 1))))

# One component's phase / log-amp contribution for cell (feed, la, ti, c).
@inline function _site_value(c::SiteComponent, A, feed, la, ti, c_idx)
    @inbounds ts = c.tseg_id[ti]
    @inbounds fs = c.fseg_id[c_idx]
    @inbounds xf = c.xf[c_idx]
    @inbounds xt = c.xt[ti]
    @inbounds cl = c.clocal[c_idx]
    val = zero(eltype(A))
    @inbounds fb1 = c.fb1[feed]
    if fb1 != 0
        off = _block_offset(c, ts, fs, fb1, la)
        val += _term_contribution(c.term, A, off, xf, xt, cl)
    end
    @inbounds fb2 = c.fb2[feed]
    if fb2 != 0
        off = _block_offset(c, ts, fs, fb2, la)
        val += _term_contribution(c.term, A, off, xf, xt, cl)
    end
    return val
end

# Sum a tuple of components (with their parallel arrays) for one cell. Tuple
# recursion keeps every term's type concrete → `term_eval` dispatches statically.
@inline _sum_site(::Tuple{}, ::Tuple{}, acc, feed, la, ti, c) = acc
@inline function _sum_site(comps::Tuple, arrs::Tuple, acc, feed, la, ti, c)
    v = _site_value(comps[1], arrs[1], feed, la, ti, c)
    return _sum_site(Base.tail(comps), Base.tail(arrs), acc + v, feed, la, ti, c)
end

function _eval_group!(gains, gp::GroupPlan, pg, chan_idx, ti_idx, ::Type{T}) where {T}
    parr = _pull(pg, gp.phase_syms)
    aarr = _pull(pg, gp.logamp_syms)
    ants = gp.ants
    z = zero(T)
    # NB: the loop body is deliberately NOT wrapped in a blanket `@inbounds` — the
    # framework array reads inside `_site_value` carry their own `@inbounds`, but a
    # term's `p[k]` stays bounds-checked (a custom term over-indexing its block
    # errors instead of silently reading the next block). Elidable with
    # `--check-bounds=no` in production.
    for feed in 1:2
        for (la, ant) in enumerate(ants)
            for (tii, ti) in enumerate(ti_idx), (ci, c) in enumerate(chan_idx)
                phase = _sum_site(gp.phase, parr, z, feed, la, ti, c)
                logamp = _sum_site(gp.logamp, aarr, z, feed, la, ti, c)
                @inbounds gains[ci, tii, ant, feed] = exp(logamp) * cis(phase)
            end
        end
    end
    return gains
end

@inline _eval_groups!(gains, ::Tuple{}, p, ci, ti, ::Type{T}) where {T} = gains
@inline function _eval_groups!(gains, groups::Tuple, p, ci, ti, ::Type{T}) where {T}
    gp = groups[1]
    pg = _sub(p, gp.groupval)
    _eval_group!(gains, gp, pg, ci, ti, T)
    return _eval_groups!(gains, Base.tail(groups), p, ci, ti, T)
end

"""
    evaluate_gains(plan::GainPlan, p, chan_idx, ti_idx) -> Array{Complex,4}

Grouped pure forward map `p → gains` of shape `(length(chan_idx),
length(ti_idx), nant, 2)` (last axis = feed) at the given GLOBAL channel/time
indices. `gain = exp(Σ logamp)·cis(Σ phase)`; element type follows `eltype(p)`
so AD tracing flows through. Numerically identical to the flat-θ
`GainEvaluator` path for the same model and parameter values.
"""
function evaluate_gains(
        plan::GainPlan, p,
        chan_idx::AbstractVector{<:Integer}, ti_idx::AbstractVector{<:Integer},
    )
    T = float(eltype(p))
    gains = Array{Complex{T}}(undef, length(chan_idx), length(ti_idx), plan.nant, 2)
    _eval_groups!(gains, plan.groups, p, chan_idx, ti_idx, T)
    return gains
end

"""
    evaluate_gains(plan::GainPlan, p) -> Array{Complex,4}

Full-grid grouped forward map: gains of shape `(nchan, ntime, nant, 2)`.
"""
evaluate_gains(plan::GainPlan, p) =
    evaluate_gains(plan, p, Base.OneTo(plan.nchan), Base.OneTo(plan.ntime))
