# ── Parameter layout ─────────────────────────────────────────────────────────
#
# `plan_parameters` flattens one shared `StationGainModel`, replicated across
# `nant` antennas over a `DataGeometry`, into a single parameter vector θ and the
# integer index tables needed to evaluate it. Everything here is plain integer
# arrays — no Dicts, Strings, or closures in the structures the hot evaluation
# loop reads — so the forward map stays type-stable and Reactant-traceable.

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
    clocal::Vector{Int}         # length nchan  → local channel index within its fseg
    nchan_seg::Vector{Int}      # length nfseg  → channels in the frequency segment
    off1::Array{Int, 4}         # (ant, feed, ntseg, nfseg)
    off2::Array{Int, 4}
end

"""
    ParameterLayout

The flattened parameter plan for a solve: total length `nθ`, the grid dims, and
one `ComponentPlan` per component (phase components first, then log-amplitude).
"""
struct ParameterLayout
    nθ::Int
    nant::Int
    ntime::Int
    nchan::Int
    nphase::Int
    plans::Vector{ComponentPlan}
end

# Build a ComponentPlan and return it together with the updated θ cursor.
function _plan_component(tc::TiedComponent, nant::Int, geom::DataGeometry, next::Int)
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

    # Local channel index within each frequency segment (walk in channel order).
    nchan = nchannels(geom)
    clocal = Vector{Int}(undef, nchan)
    counter = zeros(Int, nfseg)
    @inbounds for c in 1:nchan
        f = fseg_id[c]
        counter[f] += 1
        clocal[c] = counter[f]
    end
    # Channels per freq segment, and the block length each implies (only terms
    # whose arity comes from the data vary with it).
    nchan_seg = [length(grp) for grp in fseg_groups]
    blocklen = [nparams_per_block(t, n) for n in nchan_seg]

    off1 = zeros(Int, nant, 2, ntseg, nfseg)
    off2 = zeros(Int, nant, 2, ntseg, nfseg)
    tying = tc.tying
    for ant in 1:nant, ts in 1:ntseg, fs in 1:nfseg
        bl = blocklen[fs]
        bl == 0 && continue
        next = _assign_blocks!(off1, off2, tying, ant, ts, fs, bl, next)
    end

    plan = ComponentPlan(
        collect(Symbol, axes), tseg_id, fseg_id, xf, xt, clocal, nchan_seg, off1, off2
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
    return ParameterLayout(next - 1, nant, ntimes(geom), nchannels(geom), nphase, plans)
end
