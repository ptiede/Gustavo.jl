# ── Bandpass stage: per-segment station phase/log-amp over scan windows ──────
#
# The `Bandpass` step reads every scan (accumulate → return the scan's
# contribution) and folds the contributions in group order, which keeps the
# result deterministic at any concurrency.
# What is fit is the step's model tree (see [`default_bandpass_model`](@ref)):
# per observable, a shape component with a value per `ChannelBlocks` segment,
# optionally beside a level component at spectral-window resolution or coarser.
# A prior on the shape relates its values within each spectral window. How it is
# solved is `JointSmoother`, which runs
# [`solve_joint_bandpass!`](@ref), fitting the complex visibilities against an
# explicit per-scan source term, with the priors entering inside the gain update.
#
# This file holds the step's side: model checks, accumulation, outcome codes and
# the report. The joint solve's data model is in bandpass_data.jl, its priors,
# hyperparameters and branches in bandpass_priors.jl, and the solve itself in
# bandpass_solve.jl. The prior fits live in prior_fits.jl.

"""
    default_bandpass_model(; freq = ChannelBlocks(1)) -> GainModel

The default [`Bandpass`](@ref Gustavo.Bandpass) step model: a time-stable, per-feed
`ConstantTerm` per frequency segment for each observable — a `phase.bandpass`
and a `logamp.bandpass` component with no prior. `freq` (a `ChannelBlocks`)
sets the segments; `ChannelBlocks(1)` is a free value per channel.

Either group of a `Bandpass` model may be empty. Fit one observable only by
keeping just that group, e.g.

    Bandpass(model = GainModel(; phase = default_bandpass_model().phase))

fits the phase bandpass alone. Uniform across every antenna.

A prior on a component (`GainComponent(...; prior)`) relates its values within
each spectral window. A group may add a level component — a `ConstantTerm` per
spectral window or coarser, with no prior — beside a shape under a zero-mean
`OUPrior`:

    shape = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1),
                          Feed = PerFeed(), prior = OUPrior(; scale = 50e6, σ = 0.1))
    level = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = PerSpectralWindow(),
                          Feed = PerFeed())
    GainModel(; phase = (; level, shape), logamp = (; level, shape))
"""
function default_bandpass_model(; freq::ChannelBlocks = ChannelBlocks(1))
    bandpass = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = freq, Feed = PerFeed())
    return GainModel(phase = (; bandpass), logamp = (; bandpass))
end

# A bandpass track is one value per frequency segment, per feed, held over a
# stretch of time, under a prior along frequency. The scans of a time segment
# are solved together as one stretch, so any segmentation whose
# segments span scans fits: `GlobalTime` is the whole track, `InstrumentScans`
# breaks it at named epochs, `TimeBlocks` at a fixed cadence. `PerScan` and
# `PerIntegration` do not — a segment holding one scan leaves the band mean of
# the gain degenerate with that scan's own source coherence.
_fits_bandpass_track(tc) =
    tc.term isa ConstantTerm &&
    tc.Ti isa Union{GlobalTime, InstrumentScans, TimeBlocks} &&
    tc.Feed isa PerFeed &&
    _is_prior_along(resolve_prior(tc), :Frequency)

# The structure every bandpass model must have: at least one component overall,
# and per observable group either one component or a shape and its level. In a
# pair, the shape is the component on `ChannelBlocks` and carries a proper
# zero-mean prior (an `OUPrior`, or a `RandomWalkPrior` with an `init`); the
# level is the other, on the same time segmentation, with no prior.
function _validate_bandpass_groups(model)
    np = length(Calibration._flatten_components(model.phase))
    na = length(Calibration._flatten_components(model.logamp))
    np + na >= 1 || throw(
        ArgumentError(
            "Bandpass model fits nothing: the model compiles no components, so the step " *
                "would accumulate every scan and write nowhere. Add a phase and/or logamp " *
                "component — `default_bandpass_model()` is the standard model.",
        ),
    )
    for name in (:phase, :logamp)
        comps = Calibration._flatten_components(getproperty(model, name))
        length(comps) <= 2 || throw(
            ArgumentError(
                "Bandpass model group `$name` holds $(length(comps)) components; the bandpass " *
                    "solve fits one track set per observable, so each group holds a shape and " *
                    "at most one level.",
            ),
        )
        length(comps) == 2 && _validate_level_pair(name, comps...)
    end
    return nothing
end

function _validate_level_pair(name, a, b)
    labels = "$(component_label(a)) and $(component_label(b))"
    count(c -> c.Frequency isa ChannelBlocks, (a, b)) == 1 || throw(
        ArgumentError(
            "Bandpass model group `$name` holds two components, so one is the shape " *
                "(`Frequency = ChannelBlocks(k)`) and the other its level, on spectral " *
                "windows or coarser. Got $labels.",
        ),
    )
    shape, level = a.Frequency isa ChannelBlocks ? (a, b) : (b, a)
    isnothing(level.prior) || throw(
        ArgumentError(
            "Bandpass model group `$name`: the level $(component_label(level)) carries a " *
                "prior. A level is flat and unknown; put the prior on the shape.",
        ),
    )
    level.Ti == shape.Ti || throw(
        ArgumentError(
            "Bandpass model group `$name`: the level and the shape must share one time " *
                "segmentation. Got $(repr(level.Ti)) (level) vs $(repr(shape.Ti)) (shape).",
        ),
    )
    _proper_prior(_prior_along(resolve_prior(shape), :Frequency)) || throw(
        ArgumentError(
            "Bandpass model group `$name`: the shape $(component_label(shape)) beside a " *
                "level needs a zero-mean OUPrior or a RandomWalkPrior with an `init` to " *
                "separate the two. Without a prior the shape absorbs the level, and a " *
                "random walk without an init leaves its own level free.",
        ),
    )
    return nothing
end

# The paths to a bandpass observable's shape and level components in a step
# layout's plantree, each `nothing` when absent. `_validate_bandpass_groups`
# allows at most two components per group, and the shape is the one on
# `ChannelBlocks`.
function _bandpass_paths(plantree, group::Symbol)
    paths = _leaf_paths(plantree[group], (group,))
    isempty(paths) && return (; shape = nothing, level = nothing)
    length(paths) == 1 && return (; shape = only(paths), level = nothing)
    a, b = paths
    return _is_shape_node(_plan_node(plantree, a)) ? (; shape = a, level = b) : (; shape = b, level = a)
end


_plan_node(tree, path) = foldl((node, k) -> node[k], path; init = tree)

_is_shape_node(plan::ComponentPlan) = plan.fseg isa ChannelBlocks
_is_shape_node(g::Calibration.GroupedComponentPlan) = _is_shape_node(first(values(g.groups)))

# The station blocks `(; antennas, θ, plan)` of an observable's (`:phase` or
# `:logamp`) shape or level component, empty when the model has none. The paths
# are resolved by name: `layout.plans` holds one entry per station-signature
# group, so its positions do not name the components once a model differs
# across stations.
_bandpass_blocks(setup, θ, group::Symbol) = _path_blocks(setup, θ, setup.paths[group].shape)
_bandpass_level_blocks(setup, θ, group::Symbol) = _path_blocks(setup, θ, setup.paths[group].level)

_path_blocks(setup, θ, ::Nothing) = NamedTuple[]
_path_blocks(setup, θ, path) = antenna_blocks(setup.layout, θ, path...)

"""
    accumulate_bandpass(group::XRadio.ProcessingSet, geom::DataGeometry, antenna_pairs, feeds;
                        executor = SerialScheduler()) -> (rl, wl)

One scan group's inverse-variance sums over time per (station pair, feed pair,
channel), `rl = Σ w·v` and `wl = Σ w`, on data already gain-corrected. Both are `DimArray`s over `AntennaPair(antenna_pairs)`,
`FeedPair(feeds)` and `Frequency(geom.channel_freqs)` — labels shared by every
group of a solve, so the sums of different scans line up — in the element types
of the data. Autocorrelations never contribute; a cross pair or feed pair of the
group that the labels do not hold is an error.
"""
function accumulate_bandpass(
        group::XRadio.ProcessingSet, geom::DataGeometry, antenna_pairs, feeds;
        executor = SerialScheduler(),
    )
    isempty(group) && throw(ArgumentError("the scan group holds no Measurement Sets"))
    parts = tmap(ms -> _member_bandpass_sums(ms, geom), _members_by_frequency(group); scheduler = executor)
    ax = (_antenna_pair_dim(antenna_pairs), FeedPair(feeds), Frequency(geom.channel_freqs))
    rl = zeros(mapreduce(p -> eltype(p.wv), promote_type, parts), ax)
    wl = zeros(mapreduce(p -> eltype(p.ws), promote_type, parts), ax)
    for p in parts
        chans = Frequency(At(geom.channel_freqs[p.chan]))
        _add_by_label!(rl, wl, p.wv, p.ws, p.antennas, p.feeds, chans; autos = false)
    end
    return rl, wl
end

function _member_bandpass_sums(ms::XRadio.MeasurementSet, geom::DataGeometry)
    wv, ws = weighted_sums(_member_layers(ms)...; dims = Ti)
    chan = Calibration._channel_indices(geom, XRadio.frequencies(ms))
    return (; wv, ws, antennas = _member_antenna_pairs(ms), feeds = feed_pairs(ms), chan)
end

# Write one station's feed-`f` levels, each through `gauge`, and return them;
# `nothing` without a level.
_write_level!(::Nothing, f, ts, gauge) = nothing
function _write_level!(lw, f, ts, gauge)
    L = view(lw.values, f, :)
    node = _feed_node(lw.tying, f)
    node == 0 && return L
    for (j, Lj) in pairs(L)
        isfinite(Lj) && (lw.leaf[1, node, j, ts, lw.ai] = gauge(Lj))
    end
    return L
end

# One frequency segment's coherent residual `(r, w)`: the sums of the
# accumulators over the channels it holds. `abs2(r) / w` is the SNR² of its
# phase and log-amplitude, since the weights are inverse variances.
function _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
    r = zero(eltype(rbar_bp))
    w = zero(eltype(wbar_bp))
    for gc in chans
        cell = (AntennaPair(bi), FeedPair(p), Frequency(gc))
        rc = rbar_bp[cell...]
        wc = wbar_bp[cell...]
        (isfinite(rc) && isfinite(wc) && wc > 0) || continue
        r += rc
        w += wc
    end
    return r, w
end

# ── Track fitting and outcome codes ─────────────────────────────────────────

"""
    TrackStatus

The outcome of fitting one piece (one spectral window) of a bandpass track, as
the bandpass report records it: `TrackNoData` (no usable channel; unit gain),
`TrackSolved` (fit, with frequency structure), `TrackFlat` (fit, but its range
is below `JointSmoother`'s `flat_span`) or `TrackDeclined` (its data do not
determine its phase's 2π branches; unit gain, its correlations flagged). θ alone
cannot tell these apart: an unfitted or declined piece reads back as unit gain
and a flat one as a constant.
"""
@enum TrackStatus::Int8 TrackNoData TrackSolved TrackFlat TrackDeclined

# The outcome code for one fitted piece: nothing estimated, a constant (a range
# below `flat_span`), or a real shape.
function _band_track_status(fitted, flat_span)
    obs = [v for v in fitted if isfinite(v)]
    isempty(obs) && return TrackNoData
    return (maximum(obs) - minimum(obs)) < flat_span ? TrackFlat : TrackSolved
end

# A shape plan's frequency segments as channel groups `fsegs`, each segment's
# mean channel frequency `x` (the coordinate a prior runs along), and its
# `pieces`: the segments of each spectral window, which a prior relates and
# never crosses, with `piece_of` each segment's piece and `piece_chans` each
# piece's channels. A segment is the unit solved for, so it must lie within one
# spectral window; one that straddles a boundary is an error.
function _shape_segments(plan, geom::DataGeometry)
    fsegs = segment_groups(plan.fseg_id, length(plan.nchan_seg))
    spw = map(fsegs) do chans
        s = geom.spw_of_chan[first(chans)]
        all(c -> geom.spw_of_chan[c] == s, chans) || throw(
            ArgumentError(
                "bandpass: a frequency segment straddles a spectral-window boundary; the " *
                    "shape's segmentation must refine the spectral windows.",
            ),
        )
        s
    end
    bands = sort!(unique(spw))
    pieces = [findall(==(b), spw) for b in bands]
    piece_of = [searchsortedfirst(bands, s) for s in spw]
    piece_chans = [reduce(vcat, fsegs[p]) for p in pieces]
    x = [sum(view(geom.channel_freqs, chans)) / length(chans) for chans in fsegs]
    return (; fsegs, x, pieces, piece_of, piece_chans)
end

# The level segment of each piece of `seg` under a level plan, and the number of
# level segments. A level segment holds whole spectral windows, since a piece
# shares one level.
function _piece_levels(level_plan, seg)
    level = map(seg.piece_chans) do chans
        l = level_plan.fseg_id[first(chans)]
        all(c -> level_plan.fseg_id[c] == l, chans) || throw(
            ArgumentError(
                "bandpass: a level segment splits a spectral window; the level's " *
                    "segmentation must hold whole spectral windows.",
            ),
        )
        l
    end
    return level, length(level_plan.nchan_seg)
end


# A station block's outcome per piece over `(AntennaName, Feed, Frequency, Ti)`:
# its stations, every feed, each spectral window of its shape over the extent of
# its channels, and each of its time segments over the extent of its samples.
function _block_status_array(block, geom::DataGeometry)
    (; plan) = block
    ax = (
        _antenna_dim(geom.antennas[block.antennas]), Feed(1:geom.nfeed),
        Frequency(_segment_lookup(geom.channel_freqs, _shape_segments(plan, geom).piece_chans)),
        Ti(_time_segment_lookup(plan, geom)),
    )
    return fill(TrackNoData, ax)
end

# A station block's phase slip probability per piece, labeled as
# `_block_status_array`; `NaN` until a fit under a prior writes it.
_block_slip_array(block, geom::DataGeometry) = fill(NaN, dims(_block_status_array(block, geom)))

# A station block's resolved prior per (station, feed, time segment), over
# `(AntennaName, Feed, Ti)` labeled as `_block_status_array`; each slot starts at
# its station's own prior.
function _block_prior_array(block, geom::DataGeometry)
    (; plan) = block
    ax = (_antenna_dim(geom.antennas[block.antennas]), Feed(1:geom.nfeed), Ti(_time_segment_lookup(plan, geom)))
    arr = DimArray(Array{Union{Nothing, AbstractPrior}}(undef, map(length, ax)), ax)
    for a in axes(arr, 1)
        arr[a, :, :] .= Ref(_prior_along(plan.priors[a], :Frequency))
    end
    return arr
end

# One entry per station block, keyed `g1, g2, …` as a `CalibrationSolution` keys a
# station-heterogeneous component's leaves; a single block is its own array.
_block_leaves(arrays) = isempty(arrays) ? nothing :
    length(arrays) == 1 ? only(arrays) :
    NamedTuple{Calibration._group_keys(length(arrays))}(Tuple(arrays))

_status_arrays(::Nothing) = ()
_status_arrays(st::NamedTuple) = values(st)
_status_arrays(st) = (st,)

"""
    _bandpass_track_report(phase_status, amp_status; phase_priors = nothing,
                          amp_priors = nothing, phase_slip = nothing) -> NamedTuple

Summarize a bandpass solve's per-piece outcomes into the record the
[`Bandpass`](@ref Gustavo.Bandpass) step publishes. `phase_status`/`amp_status`
hold one [`TrackStatus`](@ref) per (station, feed, frequency segment, time segment)
— either may be `nothing` when that half was not fit — as `DimArray`s over
`(AntennaName, Feed, Frequency, Ti)`: the stations, every feed, each frequency
segment of the component over the extent of its channels and each time segment
over the extent of its samples. `phase_priors`/`amp_priors` hold the prior each
(station, feed, time segment) track was fit under, its hyperparameters resolved,
over `(AntennaName, Feed, Ti)`. `phase_slip` holds, over the axes of
`phase_status`, the posterior probability that each phase piece fit under a
prior carries a 2π slip (`NaN` for a piece without a prior or data); see
[`JointSmoother`](@ref). Where the model gives stations different segmentations,
[`JointSmoother`](@ref) passes one array per station block, keyed `g1, g2, …`
as the solution's components name the station blocks, each over its own
stations and segments.

Returns the arrays as `phase_status`/`amp_status`/`phase_priors`/`amp_priors`/`phase_slip`
alongside the counts `n_solved`/`n_flat`/`n_nodata`/`n_declined` summed over both
observables. `flat` and `nodata` are the two ways a piece can occupy a slot
without measuring anything, and they are what the counts exist to expose: θ
itself records an unfitted piece as unit gain and a starved one as a constant,
neither distinguishable there from a genuinely flat response.
"""
function _bandpass_track_report(phase_status, amp_status; phase_priors = nothing, amp_priors = nothing, phase_slip = nothing)
    counts = zeros(Int, length(instances(TrackStatus)))
    for st in (phase_status, amp_status), a in _status_arrays(st), c in a
        counts[Int(c) + 1] += 1
    end
    # Concrete arrays throughout — the record is serialized with the solution, and
    # an observable that was not fit is an empty status rather than a missing field.
    empty_status = Array{TrackStatus, 4}(undef, 0, 0, 0, 0)
    empty_priors = Array{Union{Nothing, AbstractPrior}, 3}(undef, 0, 0, 0)
    return (;
        phase_status = something(phase_status, empty_status),
        amp_status = something(amp_status, empty_status),
        phase_priors = something(phase_priors, empty_priors),
        amp_priors = something(amp_priors, empty_priors),
        phase_slip = something(phase_slip, Array{Float64, 4}(undef, 0, 0, 0, 0)),
        n_nodata = counts[1], n_solved = counts[2], n_flat = counts[3], n_declined = counts[4],
    )
end

# Warn when a large share of the tracks measured nothing. Silence here would leave
# a bandpass that is mostly placeholder looking exactly like one that is mostly
# measured — the caller cannot tell from θ, which is why this is a warning and not
# only a record.
function _warn_degenerate_bandpass(report, warn_fraction)
    total = report.n_nodata + report.n_solved + report.n_flat + report.n_declined
    total > 0 || return nothing
    degenerate = report.n_flat + report.n_nodata + report.n_declined
    frac = degenerate / total
    frac > warn_fraction || return nothing
    @warn """
    Bandpass: $(round(100 * frac; digits = 1))% of (antenna, feed, frequency segment) pieces \
    carry no measured frequency shape — $(report.n_flat) fit flat, $(report.n_nodata) \
    with no usable data, $(report.n_declined) declined (of $total). They are unit gain or a constant in the solution, not a measured \
    response. Check per-segment SNR and the detection coverage of the bandpass scans.
    """ n_solved = report.n_solved
    return nothing
end

# Warn when pieces were declined, naming their stations: the solution holds no
# bandpass for them.
function _warn_declined(report, max_slip_probability)
    report.n_declined > 0 || return nothing
    names = sort!(unique(
        String(n) for a in _status_arrays(report.phase_status)
            for (n, row) in zip(lookup(a, AntennaName), eachslice(parent(a); dims = 1)) if TrackDeclined in row
    ))
    @warn """
    Bandpass: declined $(report.n_declined) (antenna, feed, frequency segment) pieces of \
    $(join(names, ", ")): their phase's posterior probability of a 2π slip exceeds \
    max_slip_probability = $max_slip_probability, so their data do not determine the phase \
    branch from channel to channel. They are unit gain, their correlations were left out of \
    the solve, and `calibrate!(…; apply_flags = true)` flags them. See the report's \
    `phase_status`.
    """
    return nothing
end

# Warn when phase pieces fit under a prior may carry a 2π slip: the data do not
# determine their branches, so the fitted track is one of several the posterior
# allows.
function _warn_uncertain_branches(report, max_slip_probability)
    slips = [p for a in _status_arrays(report.phase_slip) for p in a if isfinite(p)]
    nslip = count(>=(max_slip_probability), slips)
    nslip > 0 || return nothing
    @warn """
    Bandpass: $nslip of $(length(slips)) phase pieces fit under a prior have a posterior \
    probability of at least $max_slip_probability (largest $(round(maximum(slips); sigdigits = 3))) \
    that their fitted track carries a 2π slip: their data do not determine the phase \
    branch from channel to channel. See the report's `phase_slip`.
    """
    return nothing
end

"""
    _antenna_time_segments(blocks, results, nant) -> Matrix{Int}

Each station's own time segment for each scan: `tseg[ant, i]` is the segment
`results[i]` falls in under the segmentation `blocks` gives that station, and `0`
for a station no block covers — such a station carries no bandpass parameter and
stays at unit gain. A scan lies wholly inside one segment of any segmentation the
smoothers accept, so its first sample (`res.ti`) names the segment.

A station-uniform model has one block spanning every station, so every row is the
same.
"""
function _antenna_time_segments(blocks, results, nant)
    tseg = zeros(Int, Base.OneTo(nant), axes(results, 1))
    for b in blocks, a in b.antennas
        for (i, res) in pairs(results)
            tseg[a, i] = b.plan.tseg_id[res.ti]
        end
    end
    return tseg
end

# The `results` indices of each independently solvable scan group, given a
# per-station time-segment table.
#
# Two scans share a parameter only through a station that is in the same one of
# its own time segments in both, so the coupling graph has one node per
# (station, segment) and one edge per pair of stations a scan holds together, and
# the joint solve runs once per connected component. With one segmentation for the whole
# array the components are exactly the array-wide time segments. With one station
# broken mid-track and the rest constant, the constant stations bridge the break
# and the whole track is one component.
function _joint_scan_groups(tseg)
    # `connected_components` numbers its nodes densely from 1, which is what
    # `LinearIndices` hands back for a (station, segment) pair.
    ntseg = maximum(tseg; init = zero(eltype(tseg)))
    nodes = LinearIndices((axes(tseg, 1), Base.OneTo(ntseg)))
    edges = Tuple{Int, Int}[]
    anchor = zeros(eltype(nodes), axes(tseg, 2))
    for si in axes(tseg, 2)
        prev = zero(eltype(nodes))
        for a in axes(tseg, 1)
            ts = tseg[a, si]
            iszero(ts) && continue
            n = nodes[a, ts]
            iszero(prev) ? (anchor[si] = n) : push!(edges, (prev, n))
            prev = n
        end
        # A scan holding a single station still has to name a component: the
        # self-edge marks the node visited without joining it to anything.
        iszero(anchor[si]) || push!(edges, (anchor[si], anchor[si]))
    end
    compid, ncomp, _ = connected_components(length(nodes), edges)
    groups = [Int[] for _ in 1:ncomp]
    for si in axes(tseg, 2)
        iszero(anchor[si]) || push!(groups[compid[anchor[si]]], si)
    end
    return filter!(!isempty, groups)
end

"""
    JointSmoother(; max_iterations = 50, tolerance = 0.01, max_slip_probability = 0.05,
                  flat_span = 0.01, warn_fraction = 0.25, eltype = nothing)

Fit the station bandpass against the actual complex visibilities, with each
component's prior acting inside the solve rather than as a fit applied to it
afterwards: the gains maximize their posterior, with the per-scan source term
profiled out and type-II MAP hyperparameters ([`solve_joint_bandpass!`](@ref)).

Because the per-scan source term absorbs a baseline's own structure, this suits a
resolved or polarized calibrator. It solves one complex gain per
(station, feed, frequency segment), so a model with both a phase and a
log-amplitude shape must give them one frequency and one time segmentation per
station; their priors and levels may differ. A model with only one of them
holds the other at zero: unit amplitude, or zero phase.

It accepts components of the form `GainComponent(ConstantTerm(); Ti = <GlobalTime,
InstrumentScans or TimeBlocks>, Frequency = <any segmentation>, Feed = PerFeed(),
prior = <nothing, or a RandomWalkPrior or OUPrior along Frequency>)`: a time
segmentation whose segments each span several scans, solved one segment at a
time. Stations may have different models.

Iterations stop once no gain moves by more than `tolerance` of its standard
error from the data and the hyperparameters given as densities are within
`tolerance` of their standard errors of the type-II MAP, or once a step lowers
the negative log posterior by less than `tolerance²/2` (a plateau, where the
posterior is flat along the gains still moving), and warn if
`max_iterations` iterations pass first: an unconverged solve still depends on
its starting point.

A phase prior depends on each segment's 2π branch, which the data leave open, so
every iteration chooses the branches of each phase piece as well, by a Gaussian-sum
Kalman filter over branch histories, and fits the hyperparameters on those
branches (see the [fitting-under-priors](@ref fitting-under-priors) page). A piece whose
posterior probability of a 2π slip at the start exceeds `max_slip_probability`
is declined: its data do not determine its branches, so it is left at unit
gain, its correlations are left out of the solve, and the step reports it
`TrackDeclined` and warns, naming its stations; `calibrate!` flags its
channels on its station's baselines under `apply_flags`. The report's
`phase_slip` gives each piece's posterior probability of a 2π slip, and the
solve warns when any fitted piece's reaches `max_slip_probability`. A fitted piece whose range is below
`flat_span` (radians or nepers) is reported flat rather than as a measured shape,
and the solve warns when more than `warn_fraction` of its pieces are flat or
without data.

The visibilities are reduced onto the frequency segments, and the gains and
source terms held, in the real type of the summed visibilities (`Float32` for
an MSv4 store); `eltype`, a real floating-point type, names another. The
linearized steps are solved in `Float64`, the type the sparse Cholesky
factorization supports.
"""
struct JointSmoother{E}
    max_iterations::Int
    tolerance::Float64
    max_slip_probability::Float64
    flat_span::Float64
    warn_fraction::Float64
    eltype::E
end
JointSmoother(;
    max_iterations::Integer = 50, tolerance::Real = 0.01, max_slip_probability::Real = 0.05,
    flat_span::Real = 0.01, warn_fraction::Real = 0.25, eltype::Union{Nothing, Type{<:AbstractFloat}} = nothing,
) = JointSmoother{_eltype_parameter(eltype)}(max_iterations, tolerance, max_slip_probability, flat_span, warn_fraction, eltype)

_eltype_parameter(::Nothing) = Nothing
_eltype_parameter(::Type{T}) where {T} = Type{T}

can_fit(::JointSmoother, tc, geom) = _fits_bandpass_track(tc)

function validate_model(::JointSmoother, model)
    _validate_bandpass_groups(model)
    ph = _shape_component(model.phase)
    la = _shape_component(model.logamp)
    (isnothing(ph) || isnothing(la)) && return nothing
    ph.Frequency == la.Frequency || throw(
        ArgumentError(
            "JointSmoother requires the phase and logamp shapes to share one frequency " *
                "segmentation — it solves one COMPLEX gain per (antenna, feed, segment). Got " *
                "$(repr(ph.Frequency)) (phase) vs $(repr(la.Frequency)) (logamp).",
        ),
    )
    ph.Ti == la.Ti || throw(
        ArgumentError(
            "JointSmoother requires the phase and logamp components to share one time " *
                "segmentation — one complex gain per (antenna, feed, segment) is solved over " *
                "one stretch of time, not two. Got $(repr(ph.Ti)) (phase) vs " *
                "$(repr(la.Ti)) (logamp).",
        ),
    )
    return nothing
end

# A group's shape component: its only component, or the one on `ChannelBlocks`
# beside a level; `nothing` for an empty group.
function _shape_component(tree)
    comps = Calibration._flatten_components(tree)
    isempty(comps) && return nothing
    length(comps) == 1 && return only(comps)
    return only(c for c in comps if c.Frequency isa ChannelBlocks)
end

function _solve_bandpass!(sm::JointSmoother, θ, results, setup; gauge::AbstractGauge)
    geom = setup.geom
    phase_blocks = _bandpass_blocks(setup, θ, :phase)
    amp_blocks = _bandpass_blocks(setup, θ, :logamp)
    # One complex gain per (station, feed, segment) means one time segmentation
    # for both observables — `validate_model` holds each station's two plans to
    # it — so either side's table is the whole solve's.
    tseg = _antenna_time_segments(_gain_blocks(phase_blocks, amp_blocks), results, length(geom.antennas))
    arrays(f, blocks) = isempty(blocks) ? nothing : [f(b, geom) for b in blocks]
    phase_status = arrays(_block_status_array, phase_blocks)
    amp_status = arrays(_block_status_array, amp_blocks)
    phase_priors = arrays(_block_prior_array, phase_blocks)
    amp_priors = arrays(_block_prior_array, amp_blocks)
    phase_slip = arrays(_block_slip_array, phase_blocks)
    for idx in _joint_scan_groups(tseg)
        fits = solve_joint_bandpass!(
            θ, results[idx], geom, phase_blocks, amp_blocks;
            phase_level_blocks = _bandpass_level_blocks(setup, θ, :phase),
            amp_level_blocks = _bandpass_level_blocks(setup, θ, :logamp),
            gauge, sm.max_iterations, sm.tolerance, sm.flat_span, sm.eltype,
            sm.max_slip_probability, tseg = view(tseg, :, idx),
        )
        _record_track_fits!(phase_status, phase_priors, phase_slip, fits.phase)
        _record_track_fits!(amp_status, amp_priors, nothing, fits.amp)
    end
    leaves(arrays) = isnothing(arrays) ? nothing : _block_leaves(arrays)
    report = _bandpass_track_report(
        leaves(phase_status), leaves(amp_status);
        phase_priors = leaves(phase_priors), amp_priors = leaves(amp_priors), phase_slip = leaves(phase_slip),
    )
    _warn_degenerate_bandpass(report, sm.warn_fraction)
    _warn_declined(report, sm.max_slip_probability)
    _warn_uncertain_branches(report, sm.max_slip_probability)
    return report
end
