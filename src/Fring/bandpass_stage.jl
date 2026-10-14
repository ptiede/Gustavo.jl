# ── Bandpass stage: per-segment station phase/log-amp over scan windows ──────
#
# The `Bandpass` step reads every scan (accumulate → return the scan's
# contribution) and folds the contributions in group order, which keeps the
# result deterministic at any concurrency.
# What is fit is the step's model tree (see [`default_bandpass_terms`](@ref)):
# per observable, a shape component with a value per `ChannelBlocks` segment,
# optionally beside a level component at spectral-window resolution or coarser.
# A prior on the shape relates its values within each spectral window. How it is
# solved is `JointSmoother`, which runs
# [`solve_joint_bandpass!`](@ref), fitting the complex visibilities against an
# explicit per-scan source term, with the priors entering inside the gain update.
#
# The prior fits live in prior_fits.jl.

"""
    default_bandpass_terms(; freq = ChannelBlocks(1)) -> GainModel

The default [`Bandpass`](@ref Gustavo.Bandpass) step model: a time-stable, per-feed
`ConstantTerm` per frequency segment for each observable — a `phase.bandpass`
and a `logamp.bandpass` component with no prior. `freq` (a `ChannelBlocks`)
sets the segments; `ChannelBlocks(1)` is a free value per channel.

Either group of a `Bandpass` model may be empty. Fit one observable only by
keeping just that group, e.g.

    Bandpass(model = GainModel(; phase = default_bandpass_terms().phase))

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
function default_bandpass_terms(; freq::ChannelBlocks = ChannelBlocks(1))
    bandpass = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = freq, Feed = PerFeed())
    return GainModel(phase = (; bandpass), logamp = (; bandpass))
end

# A bandpass track is one value per frequency segment, per feed, held over a
# stretch of time, under a prior along frequency. The scans of a time segment
# are solved together as one stretch, so any segmentation whose
# segments span scans fits: `GlobalTime` is the whole track, `InstrumentScans`
# breaks it at named epochs, `TimeBlocks` at a fixed cadence. `PerScan` and
# `PerIntegration` do not — a segment holding one scan leaves the band mean of
# the gain degenerate with that scan's own source coherence, which needs the
# deviation-from-template scheme the smoother does not implement.
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
                "component — `default_bandpass_terms()` is the standard model.",
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

# The station blocks `(; stations, θ, plan)` of an observable's (`:phase` or
# `:logamp`) shape or level component, empty when the model has none. The paths
# are resolved by name: `layout.plans` holds one entry per station-signature
# group, so its positions do not name the components once a model differs
# across stations.
_bandpass_blocks(setup, θ, group::Symbol) = _path_blocks(setup, θ, setup.paths[group].shape)
_bandpass_level_blocks(setup, θ, group::Symbol) = _path_blocks(setup, θ, setup.paths[group].level)

_path_blocks(setup, θ, ::Nothing) = NamedTuple[]
_path_blocks(setup, θ, path) = station_blocks(setup.layout, θ, path...)

"""
    accumulate_bandpass(group::XRadio.ProcessingSet, geom::DataGeometry, station_pairs, feeds;
                        executor = SerialScheduler()) -> (rl, wl)

One scan group's inverse-variance sums over time per (station pair, feed pair,
channel), `rl = Σ w·v` and `wl = Σ w`, on data already gain-corrected. Both are `DimArray`s over `AntennaPair(station_pairs)`,
`FeedPair(feeds)` and `Frequency(geom.channel_freqs)` — labels shared by every
group of a solve, so the sums of different scans line up — in the element types
of the data. Autocorrelations never contribute; a cross pair or feed pair of the
group that the labels do not hold is an error. Every smoother receives these
sums as they are.
"""
function accumulate_bandpass(
        group::XRadio.ProcessingSet, geom::DataGeometry, station_pairs, feeds;
        executor = SerialScheduler(),
    )
    isempty(group) && throw(ArgumentError("the scan group holds no Measurement Sets"))
    parts = tmap(ms -> _member_bandpass_sums(ms, geom), _members_by_frequency(group); scheduler = executor)
    ax = (_station_pair_dim(station_pairs), FeedPair(feeds), Frequency(geom.channel_freqs))
    rl = zeros(mapreduce(p -> eltype(p.wv), promote_type, parts), ax)
    wl = zeros(mapreduce(p -> eltype(p.ws), promote_type, parts), ax)
    for p in parts
        chans = Frequency(At(geom.channel_freqs[p.chan]))
        _add_by_label!(rl, wl, p.wv, p.ws, p.stations, p.feeds, chans; autos = false)
    end
    return rl, wl
end

function _member_bandpass_sums(ms::XRadio.MeasurementSet, geom::DataGeometry)
    wv, ws = weighted_sums(_member_layers(ms)...; dims = Ti)
    chan = Calibration._channel_indices(geom, XRadio.frequencies(ms))
    return (; wv, ws, stations = _member_station_pairs(ms), feeds = feed_pairs(ms), chan)
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
`TrackSolved` (fit, with frequency structure) or `TrackFlat` (fit, but its range
is below the smoother's `flat_span`). θ alone cannot tell these apart: an unfitted
piece reads back as unit gain and a flat one as a constant.
"""
@enum TrackStatus::Int8 TrackNoData TrackSolved TrackFlat

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

# The level id of each of `seg`'s segments.
_segment_levels(level, seg) = [level[p] for p in seg.piece_of]

# A station block's outcome per piece over `(AntennaName, Feed, Frequency, Ti)`:
# its stations, every feed, each spectral window of its shape over the extent of
# its channels, and each of its time segments over the extent of its samples.
function _block_status_array(block, geom::DataGeometry)
    (; plan) = block
    ax = (
        _station_dim(geom.stations[block.stations]), Feed(1:geom.nfeed),
        Frequency(_segment_lookup(geom.channel_freqs, _shape_segments(plan, geom).piece_chans)),
        Ti(_time_segment_lookup(plan, geom, plan.shape[4])),
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
    ax = (_station_dim(geom.stations[block.stations]), Feed(1:geom.nfeed), Ti(_time_segment_lookup(plan, geom, plan.shape[4])))
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
alongside the counts `n_solved`/`n_flat`/`n_nodata` summed over both
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
        n_nodata = counts[1], n_solved = counts[2], n_flat = counts[3],
    )
end

# Warn when a large share of the tracks measured nothing. Silence here would leave
# a bandpass that is mostly placeholder looking exactly like one that is mostly
# measured — the caller cannot tell from θ, which is why this is a warning and not
# only a record.
function _warn_degenerate_bandpass(report, warn_fraction)
    total = report.n_nodata + report.n_solved + report.n_flat
    total > 0 || return nothing
    degenerate = report.n_flat + report.n_nodata
    frac = degenerate / total
    frac > warn_fraction || return nothing
    @warn """
    Bandpass: $(round(100 * frac; digits = 1))% of (station, feed, frequency segment) pieces \
    carry no measured frequency shape — $(report.n_flat) fit flat, $(report.n_nodata) \
    with no usable data (of $total). They are unit gain or a constant in the solution, not a measured \
    response. Check per-segment SNR and the detection coverage of the bandpass scans.
    """ n_solved = report.n_solved
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
    _station_time_segments(blocks, results, nant) -> Matrix{Int}

Each station's own time segment for each scan: `tseg[ant, i]` is the segment
`results[i]` falls in under the segmentation `blocks` gives that station, and `0`
for a station no block covers — such a station carries no bandpass parameter and
stays at unit gain. A scan lies wholly inside one segment of any segmentation the
smoothers accept, so its first sample (`res.ti`) names the segment.

A station-uniform model has one block spanning every station, so every row is the
same.
"""
function _station_time_segments(blocks, results, nant)
    tseg = zeros(Int, Base.OneTo(nant), axes(results, 1))
    for b in blocks, a in b.stations
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
# the ALS runs once per connected component. With one segmentation for the whole
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

# ── Joint complex bandpass + per-scan source coherence ───────────────────────
#
# `solve_joint_bandpass!` fits the complex visibilities against
#
#     V_ab(ν) | scan  ≈  g_a(ν) · S_{scan,ab,pol} · conj(g_b(ν)),
#
# one frequency-flat complex `S` per (scan, baseline, polarization product), so
# a resolved or polarized source's per-baseline structure is absorbed into `S`
# rather than biasing the station bandpass. `S` given the gains is closed-form
# (`_source_coherence!`), and the gains take Gauss–Newton steps of the MAP
# objective with `S` profiled out (`_map_step`).
#
# The data are `(Scan, AntennaPair, FeedPair, Frequency)` arrays over the
# refinement cells every station's segmentation is a union of. The gains are one
# `(AntennaName, Feed, Frequency, Ti)` array per phase station block, over the block's
# own stations, frequency segments and time segments; a station's block and
# position come from `_block_locations`. Station and feed
# indices come from the `AntennaPair`/`FeedPair` labels (`_cell_nodes`).

# One scan's per-(baseline, pol, segment) coherent residual, written directly
# into `rview`/`wview`, a (AntennaPair, FeedPair, Frequency) slice of the multi-scan
# accumulator, with no intermediate allocation.
#
# These must not be SNR-gated. The joint solve consumes them as complex
# residuals under inverse-variance weights, and that accumulation is unbiased at
# any SNR: a weak cell contributes its information at its honest weight and
# costs variance, never validity. A gate would preferentially delete the weak cells pairing
# different feeds, the only rows tying one feed's gain block to another's and so
# the only measurement of the bandpass between feeds, leaving those blocks at
# their initialization. Outliers
# belong to a flagging step upstream of the solve, not to a cell gate.
function _reduce_scan_segments!(rview, wview, sc, segs)
    for p in axes(rview, FeedPair), bi in axes(rview, AntennaPair)
        for (fs, chans) in enumerate(segs)
            rc, wc = _segment_residual(sc.rl, sc.wl, bi, p, chans)
            keep = isfinite(rc) && isfinite(wc) && wc > 0
            rview[bi, p, fs] = keep ? rc : zero(rc)
            wview[bi, p, fs] = keep ? wc : zero(wc)
        end
    end
    return nothing
end

# Every scan's (AntennaPair, FeedPair, segment) residual, stacked over an added Scan
# axis — not summed across scans, since the per-scan source coherence needs each
# scan's own coherent visibility, in the scan accumulators' element types or
# in the real type `eltype`.
function _reduce_all_scans(scans, segs, cells::Frequency; eltype = nothing)
    nscan = length(scans)
    rl = first(scans).rl
    nbl, npol = size(rl, AntennaPair), size(rl, FeedPair)
    nseg = length(segs)
    T = something(eltype, real(Base.eltype(first(scans).wl)))
    C = isnothing(eltype) ? Base.eltype(rl) : complex(eltype)
    d = (Scan(1:nscan), dims(rl, AntennaPair), dims(rl, FeedPair), cells)
    rseg = DimensionalData.DimArray(zeros(C, nscan, nbl, npol, nseg), d)
    wseg = DimensionalData.DimArray(zeros(T, nscan, nbl, npol, nseg), d)
    for (si, sc) in enumerate(scans)
        _reduce_scan_segments!(view(rseg, si, :, :, :), view(wseg, si, :, :, :), sc, segs)
    end
    return rseg, wseg
end

# Each station's block and its position on that block's `AntennaName` axis; `(0, 0)`
# for a station no block covers.
function _block_locations(blocks, nant)
    loc = fill((0, 0), nant)
    for (k, b) in pairs(blocks), (ai, a) in pairs(b.stations)
        loc[a] = (k, ai)
    end
    return loc
end

# The blocks a joint solve holds its complex gains on: the phase blocks, or the
# amplitude blocks when the model has no phase. Where both exist they match.
_gain_blocks(phase_blocks, amp_blocks) = isempty(phase_blocks) ? amp_blocks : phase_blocks

# One gain block's solved state over `(AntennaName, Feed, Frequency, Ti)` — its
# stations, every feed, its frequency segments and its time segments: the
# complex gains `g`; their phases `φ` on the branches the solve chose; the slots
# whose phase and whose log-amplitude the solve gave a value (`phase_valid`,
# `amp_valid`), which differ where a prior fills one observable over a slot
# without data and the other has no prior; and the gauge pins (`pinned`).

function _block_gains(block, geom::DataGeometry, C::Type)
    ax = (
        _station_dim(geom.stations[block.stations]), Feed(1:geom.nfeed),
        Frequency(_frequency_segment_lookup(block.plan, geom)),
        Ti(_time_segment_lookup(block.plan, geom)),
    )
    return DimStack((;
        g = ones(C, ax), φ = zeros(real(C), ax),
        phase_valid = zeros(Bool, ax), amp_valid = zeros(Bool, ax), pinned = zeros(Bool, ax),
    ))
end

"""
    _station_freq_segments(blocks, nant) -> (fseg, cells)

The frequency cells a joint solve reduces its data onto — the coarsest
channel partition every block's own segmentation is a union of cells of — and
`fseg[a, k]` is the segment station `a` carries over cell `k`, `0` for a station
no block covers.

Stations that share one frequency segmentation give the identity refinement, so
`cells` is that segmentation's own channel groups and every station's cell maps
to itself.
"""
function _station_freq_segments(blocks, nant)
    cell_ids, ncell = common_refinement([b.plan.fseg_id for b in blocks])
    cells = segment_groups(cell_ids, ncell)
    fseg = zeros(Int, Base.OneTo(nant), Base.OneTo(ncell))
    for b in blocks, a in b.stations
        for (k, chans) in pairs(cells)
            # A cell lies wholly inside one segment of every block's
            # segmentation, so its first channel names them all.
            fseg[a, k] = b.plan.fseg_id[first(chans)]
        end
    end
    return fseg, cells
end

# Dense ids of the gain slots, one array per block in its gain array's shape,
# numbered consecutively across the blocks.
function _gain_slot_ids(gains)
    lasts = cumsum(map(st -> length(st.g), gains))
    return map(gains, lasts) do st, l
        reshape((l - length(st.g) + 1):l, size(st.g))
    end
end

# The phase-gauge graph of a joint bandpass solve: one node per gain slot —
# (station, feed, that station's own frequency segment, its own time segment) —
# one edge per (scan, station pair, feed pair, refinement cell) correlation,
# joining its two ends in the segments they are in for that scan and that cell.
# Node degree stands in for the row weight the gauge scores elsewhere; the graph
# is built from the correlations that exist, before any per-channel gating, so a
# node's degree is the observation count available to anchor it.
#
# The node set is the gauge condition. An edge names the two gain slots one
# correlation reads, and a shared phase cancels out of `g_a·S·conj(g_b)` only
# when both ends of every internal edge carry it, so the unobservable phases are
# one constant per connected component. No coarser node set states that, and a
# station whose frequency segments are finer than the modes the array leaves free
# would be over-constrained by any pin that fixed its whole track.
#
# `ids` numbers the slots (`_gain_slot_ids`); `compid[n]` is `0` for a slot no
# correlation touches.
function _joint_bandpass_graph(data, layout, ids)
    (; ends) = data
    (; loc, tseg, fseg) = layout
    nslots = sum(length, ids; init = 0)
    edges = Tuple{Int, Int}[]
    deg = zeros(Int, nslots)
    for si in axes(tseg, 2), p in axes(ends, FeedPair), bi in axes(ends, AntennaPair)
        _solvable(ends[bi, p]) || continue
        (a, fa), (b, fb) = ends[bi, p]
        # A station no block covers (segment 0) carries no bandpass parameter, so
        # the correlations touching it constrain nothing.
        ta, tb = tseg[a, si], tseg[b, si]
        (iszero(ta) || iszero(tb)) && continue
        (ka, ia), (kb, ib) = loc[a], loc[b]
        for cell in axes(fseg, 2)
            na = ids[ka][ia, fa, fseg[a, cell], ta]
            nb = ids[kb][ib, fb, fseg[b, cell], tb]
            push!(edges, (na, nb))
            deg[na] += 1
            deg[nb] += 1
        end
    end
    compid, ncomp, _ = connected_components(nslots, edges)
    return compid, ncomp, deg
end

"""
    _joint_bandpass_gauge!(gains, data, layout, blocks, gauge) -> Union{Nothing, NamedTuple}

Impose `gauge` on the phase of a joint bandpass solve: one
[`GaugeFreedom`](@ref) per connected component of the joint-bandpass graph,
and the constraints `C·φ = d` that [`gauge_constraints`](@ref) gives over
every gain slot's phase `φ`, numbered by `_gain_slot_ids`. When every row
holds a single slot at zero, as `PinAntenna`'s do, those slots are marked in
the `pinned` layer of `gains`, held at zero phase, and the result is
`nothing`. Any other gauge is imposed exactly in every Gauss–Newton step
(`_map_step`), and this returns the constraints as `(; rows, d)`,
`rows[j]` the `(slot, coefficient)` pairs of row `j`. Either way a constrained
slot's amplitude is solved like any other slot's.

Only the phase is a gauge freedom. Multiplying the gains of a set of nodes by a
shared `c` sends `g_a·S·conj(g_b)` to `|c|²·g_a·S·conj(g_b)` on every
correlation internal to that set: the phase of `c` cancels between the two
conjugated factors. The sets on which it cancels everywhere are exactly the
components above, so the phase carries one unobservable constant per component
and needs one constraint there — no more. A station-uniform segmentation splits
the graph along the array-wide (frequency segment, time segment) cells, so the
free direction is a phase spectrum common to every station, and a pin per cell
zeroes the reference station's whole track; a model in which one station
breaks mid-track keeps the whole track in one component, because the stations
that hold one gain over it bridge the broken station's two segments, and the
relative phase across that break is then measured rather than gauged away.

The per-station priors along frequency are not invariant to that common
spectrum, so the gauge changes the solution and is imposed during the fit
rather than by shifting a fitted result.

The same bridging on the frequency axis makes a pin PARTIAL: where a station
holds one gain across cells the others split, those cells lie in one component,
and its single pin fixes ONE segment of the pinned station's track while the
rest of that track is fitted.

Besides that common phase, each (station, feed, time segment) track has a free
band-constant phase and log-amplitude: `S` is per scan and baseline but
frequency-flat, so it absorbs a band-constant factor on any one station, and
a prior is profiled over that constant (see `_add_track_prior!`). Pinning `|g|` as well
would assert the reference antenna has a flat amplitude bandpass, discarding
structure that is identifiable (mean-removing `log|V_ab| = la_a + la_b + ls_ab`
over the band eliminates `ls` and leaves the full-rank signless-Laplacian
system) and biasing every other station through the inconsistency.
"""
function _joint_bandpass_gauge!(gains, data, layout, blocks, gauge)
    T = real(eltype(first(gains).g))
    ids = _gain_slot_ids(gains)
    compid, ncomp, deg = _joint_bandpass_graph(data, layout, ids)
    slot = [(k, I) for (k, idk) in pairs(ids) for I in CartesianIndices(idk)]
    freedoms = map(1:ncomp) do c
        cn = findall(==(c), compid)
        GaugeFreedom(;
            nodes = cn, station = [blocks[slot[n][1]].stations[slot[n][2][1]] for n in cn],
            feed = [slot[n][2][2] for n in cn], scan = zeros(Int, length(cn)),
            component = [blocks[slot[n][1]].plan.path for n in cn], observable = fill(:phase, length(cn)),
            direction = ones(T, length(cn)), weight = deg[cn],
        )
    end
    C, d = _gauge_system(gauge, GaugeFreedoms{T}(freedoms, length(compid)))
    Cs = sparse(C)
    rows = [Tuple{Int, T}[] for _ in axes(Cs, 1)]
    for n in axes(Cs, 2), i in nzrange(Cs, n)
        iszero(nonzeros(Cs)[i]) || push!(rows[rowvals(Cs)[i]], (n, nonzeros(Cs)[i]))
    end
    if all(j -> length(rows[j]) == 1 && iszero(d[j]), eachindex(rows, d))
        for row in rows
            k, I = slot[first(only(row))]
            gains[k].pinned[I] = true
        end
        return nothing
    end
    return (; rows, d)
end

# The source coherence that minimizes the χ² for the given slot gains `g`: for
# each (scan, station pair, feed pair) the weighted least-squares solution of
# `Σ_cell w·|r/w − g_a·S·conj(g_b)|²` over the single complex `S`.
function _source_coherence!(S, g, data, jl)
    (; r, w) = data
    (; edge, slot_pairs) = jl
    num = zeros(ComplexF64, size(S))
    den = zeros(size(S))
    for I in CartesianIndices(edge)
        e = edge[I]
        iszero(e) && continue
        wc = w[I]
        wc > 0 || continue
        na, nb = slot_pairs[e]
        u = g[na] * conj(g[nb])
        J = CartesianIndex(I[1], I[2], I[3])
        num[J] += conj(u) * r[I]
        den[J] += wc * abs2(u)
    end
    for J in eachindex(S, num, den)
        S[J] = den[J] > 0 ? num[J] / den[J] : zero(eltype(S))
    end
    return S
end

# The χ² of the visibilities against `g_a·S·conj(g_b)`.
function _chi2(g, S, data, jl)
    (; r, w) = data
    (; edge, slot_pairs) = jl
    out = 0.0
    for I in CartesianIndices(edge)
        e = edge[I]
        iszero(e) && continue
        wc = w[I]
        wc > 0 || continue
        na, nb = slot_pairs[e]
        out += abs2(r[I] - wc * g[na] * S[I[1], I[2], I[3]] * conj(g[nb])) / wc
    end
    return out
end

# The χ² to second order in each slot's log-gain about `g`. A correlation's model
# `m = g_a·S·conj(g_b)` moves to `m·(1 + δℓ_a + δℓ_b + i(δφ_a − δφ_b))`, so with
# `u = r/(w·m) − 1` and weight `q = w·|m|²`, `Re u` measures `δℓ_a + δℓ_b` and
# `Im u` measures `δφ_a − δφ_b`, and the two observables decouple. Returns each
# correlation pair's weight `q`, each slot's `c = Σ q·u` (with `conj(u)` at a
# second end), whose real and imaginary parts are the log-amplitude and phase
# gradients, and its total weight `W`: `1 + c/W` is the slot's own gain
# estimate given every other gain and `S`, relative to its current one.
function _linearize(g, S, data, jl, nslots)
    (; r, w) = data
    (; edge, slot_pairs) = jl
    q = zeros(length(slot_pairs))
    c = zeros(ComplexF64, nslots)
    W = zeros(nslots)
    for I in CartesianIndices(edge)
        e = edge[I]
        iszero(e) && continue
        wc = w[I]
        wc > 0 || continue
        na, nb = slot_pairs[e]
        m = g[na] * S[I[1], I[2], I[3]] * conj(g[nb])
        qc = wc * abs2(m)
        qc > 0 || continue
        u = r[I] / (wc * m) - 1
        q[e] += qc
        c[na] += qc * u
        c[nb] += qc * conj(u)
        W[na] += qc
        W[nb] += qc
    end
    return (; q, c, W)
end

# The curvature of the χ² in one observable's slot values: `Σ_e q_e·(x_a + s·x_b)²`
# with `s = +1` for log-amplitude sums and `−1` for phase differences.
function _data_curvature(q, slot_pairs, s, nslots)
    Is, Js, Vs = Int[], Int[], Float64[]
    for (e, (na, nb)) in pairs(slot_pairs)
        qe = q[e]
        qe > 0 || continue
        append!(Is, (na, nb, na, nb))
        append!(Js, (na, nb, nb, na))
        append!(Vs, (qe, qe, s * qe, s * qe))
    end
    return sparse(Is, Js, Vs, nslots, nslots)
end

# ── Track priors as sparse quadratics ────────────────────────────────────────
#
# A prior's energy over a track is a sum of state-space terms: each piece (one
# spectral window) is a chain of states along frequency, `s_k = A_k·s_{k−1} + η_k`,
# `η_k ~ N(0, Q_k)`, from the prior's start where it is proper, whose first entry
# is the track's value less its level. Written over the states themselves the
# energy is a sum of small quadratic terms, so a track's prior, its latent states
# and its levels enter one sparse system with the data, exactly and without a
# start variance standing in for a flat start.
#
# The per-scan source term is frequency-flat, so it absorbs any constant added to
# a (station, feed, time segment) track: the data cannot measure a track's band
# constant. The MAP of everything else is then the same as under the prior
# profiled over that constant, so a proper prior acts on its track less a free
# constant, as if beside a level. A flat-start walk is unchanged by a constant.

# A quadratic `½uᵀKu − hᵀu + c` over a growing set of unknowns `u`, each with
# the frequency `key` by which they are ordered for factoring.
mutable struct _Quadratic
    Is::Vector{Int}
    Js::Vector{Int}
    Vs::Vector{Float64}
    h::Vector{Float64}
    c::Float64
    key::Vector{Float64}
end
_Quadratic(key) = _Quadratic(Int[], Int[], Float64[], zeros(length(key)), 0.0, collect(Float64, key))
_unknown!(Q::_Quadratic, key) = (push!(Q.h, 0.0); push!(Q.key, key); length(Q.h))
_matrix(Q::_Quadratic) = sparse(Q.Is, Q.Js, Q.Vs, length(Q.h), length(Q.h))
Base.copy(Q::_Quadratic) = _Quadratic(copy(Q.Is), copy(Q.Js), copy(Q.Vs), copy(Q.h), Q.c, Q.key)

# A linear form `Σ a·u[j] + b`: its terms `j => a` and its constant `b`.
const _Form = Tuple{Vector{Pair{Int, Float64}}, Float64}

# `Σ_j c_j·forms_j`.
function _combine(forms, coefs)
    terms = Pair{Int, Float64}[]
    b = 0.0
    for (f, a) in zip(forms, coefs)
        iszero(a) && continue
        append!(terms, (j => a * v) for (j, v) in f[1])
        b += a * f[2]
    end
    return (terms, b)
end

# Add `½ρᵀWρ` for the forms `ρ`.
function _add_term!(Q::_Quadratic, ρ, W)
    cols = unique(first(t) for f in ρ for t in f[1])
    G = zeros(length(ρ), length(cols))
    r = zeros(length(ρ))
    for (i, f) in pairs(ρ)
        for (j, a) in f[1]
            G[i, findfirst(==(j), cols)] += a
        end
        r[i] = -f[2]
    end
    W = (W + W') / 2
    GW = G' * W
    K = GW * G
    for a in eachindex(cols), b in eachindex(cols)
        push!(Q.Is, cols[a])
        push!(Q.Js, cols[b])
        push!(Q.Vs, K[a, b])
    end
    h = GW * r
    for (a, j) in pairs(cols)
        Q.h[j] += h[a]
    end
    Q.c += dot(r, W * r) / 2
    return Q
end

# The coordinates a prior's state-space model runs on over one piece's
# frequencies `x`, and their unit `h` in `x`.
function _piece_coordinates(::RandomWalkPrior, x)
    _, u, h = _random_walk_problem(x, ones(length(x)), x)
    return u, h
end
_piece_coordinates(::OUPrior, x) = float.(x), 1.0

# The state-space model of a resolved prior on coordinates of unit `h`, in
# element type `T`.
_piece_model(prior::RandomWalkPrior, h, T = Float64) = _random_walk_model(prior.order, prior.σ, prior.init, h, T)
_piece_model(prior::OUPrior, h, T = Float64) = OUModel(T(prior.scale), T(prior.σ)^2)
_statedim(prior::RandomWalkPrior) = prior.order
_statedim(::OUPrior) = 1

# The samples a flat-start walk needs to have its state determined; one for a
# proper prior.
_samples_needed(prior::RandomWalkPrior) = _proper_prior(prior) ? 1 : prior.order
_samples_needed(::OUPrior) = 1

# The level group of track `t`'s piece `j`: its level segment beside a level,
# one group for the whole track under a proper prior (the constant the source term
# absorbs), and `nothing` under a flat-start walk, which no constant changes.
_level_group(t, prior, j) = !isnothing(t.level) ? t.level[j] : _proper_prior(prior) ? 1 : nothing
_level_groups(t, prior) = !isnothing(t.level) ? t.level : _proper_prior(prior) ? ones(Int, length(t.pieces)) : nothing

# The state chain of each piece of track `t` its prior acts on, over the unknowns
# of `Q`: per sample its state less the piece's level, as linear forms whose
# first is the slot (`slot(n)` gives its unknown, or `0` and its fixed value) and
# the rest new latent unknowns, with the coordinates its prior runs on. Returns
# the chains and the unknown of each level group.
function _track_chains!(Q::_Quadratic, t, slot)
    levels = Dict{Int, Int}()
    chains = NamedTuple[]
    for (j, p) in pairs(t.pieces)
        t.active[j] || continue
        group = _level_group(t, t.prior, j)
        L = isnothing(group) ? 0 : get!(() -> _unknown!(Q, Inf), levels, group)
        states = map(p) do k
            j1, v1 = slot(t.slots[k])
            value = (iszero(j1) ? Pair{Int, Float64}[] : [j1 => 1.0], iszero(j1) ? v1 : 0.0)
            iszero(L) || push!(value[1], L => -1.0)
            _Form[value; [([_unknown!(Q, t.x[k]) => 1.0], 0.0) for _ in 2:_statedim(t.prior)]]
        end
        u, h = _piece_coordinates(t.prior, t.x[p])
        push!(chains, (; u, h, states))
    end
    return chains, levels
end

# Add the energy of `chains` under the resolved `prior` to `Q`: a term per
# transition and, where the prior's start is proper, one for the first state.
function _add_prior_terms!(Q::_Quadratic, chains, prior)
    for ch in chains
        model = _piece_model(prior, ch.h)
        start = initial(model, Float64)
        if !isnothing(start)
            μ0, P0 = start
            _add_term!(Q, [(f[1], f[2] - μ0[i]) for (i, f) in pairs(ch.states[1])], inv(Matrix(P0)))
        end
        for i in 2:length(ch.states)
            A, Qk = transition(model, ch.u[i] - ch.u[i - 1], Float64)
            ρ = [_combine([ch.states[i][r]; ch.states[i - 1]], [1.0; -A[r, :]]) for r in axes(A, 1)]
            _add_term!(Q, ρ, inv(Matrix(Qk)))
        end
    end
    return Q
end

# The log density of `chains` under the resolved `prior`, in expectation over
# the states' posterior with the moments `mom` (`_chain_moments`), or its
# normalization alone without them; in element type `T`.
_expected_log_prior(prior, chains, mom, T) = sum(
    (_expected_chain_log_prior(_piece_model(prior, ch.h, T), ch.u, isnothing(mom) ? nothing : mom[c], T) for (c, ch) in pairs(chains));
    init = zero(T),
)

function _expected_chain_log_prior(model, u, mom, T)
    D = statedim(model)
    out = zero(T)
    start = initial(model, T)
    if !isnothing(start)
        μ0, P0 = start
        out -= logdet(2π * P0) / 2
        if !isnothing(mom)
            m, M = SVector{D}(mom.start[1]), SMatrix{D, D}(mom.start[2])
            out -= tr(P0 \ (M - μ0 * m' - m * μ0' + μ0 * μ0')) / 2
        end
    end
    for i in 2:length(u)
        A, Q = transition(model, u[i] - u[i - 1], T)
        out -= logdet(2π * Q) / 2
        isnothing(mom) && continue
        G = hcat(SMatrix{D, D, T}(I), -A)
        out -= tr(Q \ (G * SMatrix{2D, 2D}(mom.pairs[i - 1]) * G')) / 2
    end
    return out
end

# The posterior moments of each chain's states, from the mean `z` and covariance
# `cov(i, j)` of the unknowns: the first state's mean and second moment, and the
# second moment of each consecutive pair `(s_i, s_{i−1})`.
function _chain_moments(chains, z, cov)
    mean_of(f) = sum((a * z[j] for (j, a) in f[1]); init = 0.0) + f[2]
    second(fs) = [
        sum((a * b * cov(i, j) for (i, a) in f[1] for (j, b) in g[1]); init = 0.0) + mean_of(f) * mean_of(g)
            for f in fs, g in fs
    ]
    return map(chains) do ch
        (;
            start = (mean_of.(ch.states[1]), second(ch.states[1])),
            pairs = [second([ch.states[i]; ch.states[i - 1]]) for i in 2:length(ch.states)],
        )
    end
end

# Track `t`'s prior energy at the slot values `x`, minimized over its latent
# states and levels, and those levels by group.
function _track_energy(t, prior, x)
    Q = _Quadratic(Float64[])
    chains, levels = _track_chains!(Q, t, n -> (0, x[n]))
    _add_prior_terms!(Q, chains, prior)
    isempty(Q.h) && return Q.c, Dict{Int, Float64}()
    u = cholesky(Symmetric(_matrix(Q))) \ Q.h
    return Q.c - dot(Q.h, u) / 2, Dict(g => u[j] for (g, j) in levels)
end

"""
    _posterior_system(H, b, x, free, tracks, key; gauge = nothing)

The MAP problem of one observable's slot values `x` with the χ² to second order
(data curvature `H` and gradient `b` at `x`, over every slot) and the exact prior
energy of each of `tracks`, over the `free` slots (the other slots keep their
values), the tracks' latent states and their levels: the data's quadratic and
each track's state chains (`_track_chains!`), the priors added by
`_assembled`. `gauge`, when given, holds constraint rows `(slot, coefficient)` and
right-hand sides `d`; each row is solved for its free slot of largest
coefficient, `x = T·u + t0` over the unknowns, so that `u`, the unknowns left,
are unconstrained and the system positive definite. `key` gives each slot's
frequency, by which the unknowns are ordered for factoring: the system is then
banded, coupling stations only at one frequency and frequencies only along a
track.
"""
function _posterior_system(H, b, x, free, tracks, key; gauge = nothing)
    idx = zeros(Int, length(x))
    nf = 0
    for n in eachindex(free)
        free[n] && (idx[n] = (nf += 1))
    end
    data = _Quadratic(key[free])
    Hf = H[free, free]
    Is, Js, Vs = findnz(Hf)
    append!(data.Is, Is)
    append!(data.Js, Js)
    append!(data.Vs, Vs)
    xf = Float64.(x[free])
    data.h .= b[free] .+ Hf * xf
    data.c = dot(xf, Hf * xf) / 2 + dot(b[free], xf)
    slot(n) = free[n] ? (idx[n], 0.0) : (0, Float64(x[n]))
    built = [_track_chains!(data, t, slot) for t in tracks]
    return (;
        data, chains = first.(built), levels = last.(built), x = Float64.(x), free, xf,
        gauge = _gauge_map(gauge, x, free, idx, length(data.h)),
    )
end

# Each gauge row solved for its free slot of largest coefficient, over the `n`
# unknowns of a system whose first are the free slots (numbered `idx`):
# `x = T·u + t0`, `keep` the unknowns `u` stands for, and `Tt = Tᵀ`. The rows hold
# disjoint slots, one per connected component.
function _gauge_map(gauge, x, free, idx, n)
    isnothing(gauge) && return nothing
    row_of = zeros(Int, n)
    for (r, row) in pairs(gauge.rows)
        cands = [(m, c) for (m, c) in row if free[m]]
        isempty(cands) && continue
        row_of[idx[first(cands[argmax([abs(c) for (_, c) in cands])])]] = r
    end
    keep = findall(iszero, row_of)
    col = zeros(Int, n)
    col[keep] = eachindex(keep)
    Is, Js, Vs = collect(keep), col[keep], ones(length(keep))
    t0 = zeros(n)
    for e in findall(!iszero, row_of)
        row = gauge.rows[row_of[e]]
        ce = only(c for (m, c) in row if free[m] && idx[m] == e)
        t0[e] = (gauge.d[row_of[e]] - sum((c * x[m] for (m, c) in row if !free[m]); init = 0.0)) / ce
        for (m, c) in row
            (free[m] && idx[m] != e) || continue
            push!(Is, e)
            push!(Js, col[idx[m]])
            push!(Vs, -c / ce)
        end
    end
    T = sparse(Is, Js, Vs, n, length(keep))
    return (; T, t0, keep, Tt = sparse(T'))
end

# The system's quadratic with each track's energy under `priors` added.
function _assembled(sys, priors)
    Q = copy(sys.data)
    for (chains, prior) in zip(sys.chains, priors)
        _add_prior_terms!(Q, chains, prior)
    end
    return Q
end

# `Q` plus `λ/2·‖x′ − x‖²` over the free slots, over the unknowns the gauge
# leaves: `(K, h, c, key)` of `½uᵀKu − hᵀu + c`.
function _reduced(sys, Q, λ)
    nf = length(sys.xf)
    K = _matrix(Q) + spdiagm([fill(Float64(λ), nf); zeros(length(Q.h) - nf)])
    h = copy(Q.h)
    h[1:nf] .+= λ .* sys.xf
    c = Q.c + λ * dot(sys.xf, sys.xf) / 2
    isnothing(sys.gauge) && return K, h, c, Q.key
    (; T, t0, keep) = sys.gauge
    return T' * K * T, T' * (h - K * t0), c + dot(t0, K * t0) / 2 - dot(h, t0), Q.key[keep]
end

# The unknowns `u` in the system's own.
_unreduced(sys, u) = isnothing(sys.gauge) ? u : sys.gauge.T * u + sys.gauge.t0

_factor(K, key; check = true) = cholesky(Symmetric(K); perm = sortperm(key), check)

"""
    _map_step(sys, priors; λ = 0) -> (x′, levels, model)

One Levenberg–Marquardt step of the MAP objective of the system `sys`
(`_posterior_system`) with each track's resolved prior from `priors`: the
minimum of its quadratic plus `λ/2·‖x′ − x‖²` over the free slots. Returns the
new slot values, each track's level values by group, and the undamped
quadratic's value there, which is the prior energy alone at `x′ = x`.
"""
function _map_step(sys, priors; λ = 0)
    Q = _assembled(sys, priors)
    K, h, _, key = _reduced(sys, Q, λ)
    z = _unreduced(sys, _factor(K, key) \ h)
    x′ = copy(sys.x)
    x′[sys.free] .= z[eachindex(sys.xf)]
    return x′, [Dict(g => z[j] for (g, j) in lv) for lv in sys.levels], dot(z, _matrix(Q) * z) / 2 - dot(Q.h, z) + Q.c
end

# ── Hyperparameters and branches ─────────────────────────────────────────────

"""
    _selected_inverse(F) -> (i, j) -> K⁻¹[i, j]

The entries of `K⁻¹` on the sparsity pattern of its Cholesky factor `F`, by the
recursion of Takahashi, Fagan & Chen (1973) from the last column back. The
pattern holds every pair of unknowns that share a term of the energy, so these
are the posterior covariances the expected energy needs. Looking up a pair
outside it throws.
"""
function _selected_inverse(F)
    L = sparse(F.L)
    rv, Lv = rowvals(L), nonzeros(L)
    n = size(L, 1)
    Zv = zeros(length(Lv))
    acc = zeros(n)
    pos = zeros(Int, n)
    mark = zeros(Int, n)
    for j in reverse(axes(L, 2))
        rng = nzrange(L, j)
        below = rng[2:end]
        for kk in below
            mark[rv[kk]] = j
            pos[rv[kk]] = kk
            acc[rv[kk]] = 0.0
        end
        # `Σ_k L_kj·Z_ik` for each row `i` below the diagonal, from the stored
        # lower triangle of the columns already done.
        for kk in below
            k = rv[kk]
            for zz in nzrange(L, k)
                i = rv[zz]
                mark[i] == j || continue
                acc[i] += Lv[kk] * Zv[zz]
                i != k && (acc[k] += Lv[pos[i]] * Zv[zz])
            end
        end
        d = Lv[first(rng)]
        s = 0.0
        for kk in below
            Zv[kk] = -acc[rv[kk]] / d
            s += Lv[kk] * Zv[kk]
        end
        Zv[first(rng)] = (1 / d - s) / d
    end
    Z = SparseMatrixCSC(n, n, copy(L.colptr), copy(rv), Zv)
    ip = invperm(F.p)
    return function (a, b)
        col, row = minmax(ip[a], ip[b])
        rng = nzrange(Z, col)
        k = searchsortedfirst(view(rowvals(Z), rng), row)
        k <= length(rng) && rowvals(Z)[rng[k]] == row ||
            error("posterior covariance ($a, $b) is not on the factor's sparsity pattern")
        return nonzeros(Z)[rng[k]]
    end
end

# The covariance of a system's own unknowns, from the covariance `cov` of the
# unknowns `u` its gauge leaves.
_covariance(cov, ::Nothing) = cov
function _covariance(cov, gauge)
    Tt = gauge.Tt
    rv, v = rowvals(Tt), nonzeros(Tt)
    return (a, b) -> sum((v[p] * v[q] * cov(rv[p], rv[q]) for p in nzrange(Tt, a) for q in nzrange(Tt, b)); init = 0.0)
end

# The hyperparameters `prior` gives as densities, by name; `prior` with those at
# the values `vals`; and their hyperprior's log density over their logs.
_hyper_names(::Nothing) = ()
_hyper_names(p::RandomWalkPrior) = is_fixed_hyper(p.σ) ? () : (:σ,)
_hyper_names(p::OUPrior) = Tuple(s for s in (:scale, :σ) if !is_fixed_hyper(getfield(p, s)))
_with_hypers(p::RandomWalkPrior, vals) = RandomWalkPrior(; p.order, σ = get(vals, :σ, p.σ), p.init)
_with_hypers(p::OUPrior, vals) = OUPrior(; scale = get(vals, :scale, p.scale), σ = get(vals, :σ, p.σ))
_log_hyperprior(prior, vals) = sum(logdensityof(getfield(prior, k), v) + log(v) for (k, v) in pairs(vals))

# The inverse of a symmetric curvature, its eigenvalues taken in magnitude.
function _inverse_curvature(C)
    E = eigen(Symmetric(C))
    return E.vectors * Diagonal(inv.(abs.(E.values))) * E.vectors'
end

"""
    _fit_hypers!(priors, curvature, sys, tracks, λ; tolerance, max_iterations) -> Bool

Type-II MAP of the hyperparameters `tracks`' priors give as densities, written
into `priors` (each track's resolved prior, the current values on entry): the
maximum over their logs of the hyperprior density times the Laplace
approximation of the marginal likelihood of the system `sys`
(`_posterior_system`, damped by `λ`). That quadratic is Gaussian in its
unknowns, so the marginal is exact for it:

    log p(data | θ) = −min_u f_θ(u) − ½·log det ∇²f_θ + Σ_k log N-normalization of each prior term,

the levels and flat starts integrated under their flat priors. Its gradient in
a track's hyperparameters is that of the track's prior log density in
expectation over the posterior of its states (Fisher's identity), whose moments
need only the covariances `_selected_inverse` gives.

The search is quasi-Newton with one secant (BFGS) block per track, started from
the curvature of that expected log density and kept in `curvature` from call to
call, each step backtracked until the log posterior is no lower. It stops once
the Newton step is within `tolerance` of its standard error, `gᵀH⁻¹g ≤
tolerance²`, by every track's curvature measured along a step of its own, and
returns whether it did within `max_iterations` steps.
"""
function _fit_hypers!(priors, curvature, sys, tracks, λ; tolerance, max_iterations)
    est = [i for i in eachindex(tracks) if !isempty(_hyper_names(tracks[i].prior))]
    isempty(est) && return true
    names = [_hyper_names(tracks[i].prior) for i in est]
    vals(k, lθ) = NamedTuple{names[k]}(Tuple(exp.(lθ)))
    θ = [log.([getfield(priors[i], s) for s in names[k]]) for (k, i) in pairs(est)]
    function evaluate(θ)
        trial = copy(priors)
        for (k, i) in pairs(est)
            trial[i] = _with_hypers(tracks[i].prior, vals(k, θ[k]))
        end
        K, h, c, key = _reduced(sys, _assembled(sys, trial), λ)
        F = _factor(K, key; check = false)
        # Hyperparameters so extreme that the posterior's curvature is lost to
        # rounding are no candidate.
        issuccess(F) || return (; lp = -Inf, F, u = h, trial)
        u = F \ h
        lp = dot(h, u) / 2 - c - logdet(F) / 2
        lp += sum(_expected_log_prior(p, ch, nothing, Float64) for (p, ch) in zip(trial, sys.chains); init = 0.0)
        lp += sum(_log_hyperprior(tracks[i].prior, vals(k, θ[k])) for (k, i) in pairs(est))
        return (; lp, F, u, trial)
    end
    # Each track's expected log prior density and hyperprior, as a function of
    # its log hyperparameters.
    function expected(at)
        cov = _covariance(_selected_inverse(at.F), sys.gauge)
        z = _unreduced(sys, at.u)
        moms = [_chain_moments(sys.chains[i], z, cov) for i in est]
        return [
            lθ -> _expected_log_prior(_with_hypers(tracks[i].prior, vals(k, lθ)), sys.chains[i], moms[k], eltype(lθ)) +
                _log_hyperprior(tracks[i].prior, vals(k, lθ))
                for (k, i) in pairs(est)
        ]
    end
    at = evaluate(θ)
    isfinite(at.lp) || throw(PosDefException(0))
    fs = expected(at)
    for (k, i) in pairs(est)
        isnothing(curvature[i]) &&
            (curvature[i] = (; inverse = _inverse_curvature(-ForwardDiff.hessian(fs[k], θ[k])), measured = false))
    end
    g = [ForwardDiff.gradient(fs[k], θ[k]) for k in eachindex(est)]
    converged = false
    for _ in 1:max_iterations
        Hinv = [curvature[i].inverse for i in est]
        if sum(k -> dot(g[k], Hinv[k] * g[k]), eachindex(est)) <= tolerance^2 && all(i -> curvature[i].measured, est)
            converged = true
            break
        end
        d = [Hinv[k] * g[k] for k in eachindex(est)]
        α = 1.0
        next = nothing
        while true
            θ1 = [θ[k] .+ α .* d[k] for k in eachindex(est)]
            all(k -> _within_rounding(θ1[k], θ[k]), eachindex(est)) && break
            trial = evaluate(θ1)
            # The log posterior is known to its rounding.
            trial.lp >= at.lp - 16 * eps(abs(at.lp)) && (next = (θ1, trial); break)
            α /= 2
        end
        # No step changes the hyperparameters: they are at the maximum to rounding.
        isnothing(next) && (converged = true; break)
        θ1, at = next
        fs = expected(at)
        g1 = [ForwardDiff.gradient(fs[k], θ1[k]) for k in eachindex(est)]
        for (k, i) in pairs(est)
            s, y = θ1[k] - θ[k], g[k] - g1[k]
            sy = dot(s, y)
            sy > 0 || continue
            V = I - y * s' / sy
            curvature[i] = (; inverse = V' * curvature[i].inverse * V + s * s' / sy, measured = true)
        end
        θ, g = θ1, g1
    end
    priors[est] .= at.trial[est]
    return converged
end

# Each slot's own estimate of one observable given every other gain and `S`, from
# its relative gain estimate `1 + c/W` (`_linearize`), and its weight; `NaN` and
# zero for a slot without data. With the phase held at zero the least-squares
# log-amplitude is that of `Re(1 + c/W)`.
function _slot_estimates(x, lin, observable; phase_held = false)
    y = fill(NaN, length(x))
    wy = zeros(length(x))
    for n in eachindex(x, lin.W)
        lin.W[n] > 0 || continue
        ρ = 1 + lin.c[n] / lin.W[n]
        v = observable === :phase ? angle(ρ) : phase_held ? (real(ρ) > 0 ? log(real(ρ)) : NaN) : log(abs(ρ))
        isfinite(v) || continue
        y[n] = x[n] + v
        wy[n] = lin.W[n] * abs2(ρ)
    end
    return y, wy
end

# Track `t`'s prior with its hyperparameters estimated from the slot estimates
# `y` (weights `wy`) as independent measurements of the track alone: type-II MAP
# over its pieces, any constant the source term absorbs integrated out, and for
# a phase over its pieces' 2π branches. The joint solve starts from these.
function _seed_prior(t, y, wy; phase::Bool)
    ys = [y[t.slots[p]] for p in t.pieces]
    ws = [wy[t.slots[p]] for p in t.pieces]
    xs = [float.(t.x[p]) for p in t.pieces]
    return _estimate_hypers(t.prior, ys, ws, xs; level = _level_groups(t, t.prior), branches = phase)
end

# Put phase track `t` on the branches `_branch_filter` finds most probable for its
# slot estimates `y` under the resolved `prior`: each free slot of `φ` moves by the
# multiple of 2π its estimate moves, which leaves its gain unchanged. A piece's
# branches are first moved together by the multiple of 2π nearest its estimates,
# and the pieces sharing a level group onto one branch. Writes each piece's
# `_slip_probability` into `slip` and returns the per-slot shifts (radians).
function _choose_branches!(φ, t, prior, y, wy, free; slip = nothing)
    groups = _level_groups(t, prior)
    ys = [y[t.slots[p]] for p in t.pieces]
    ws = [wy[t.slots[p]] for p in t.pieces]
    lifts = map(eachindex(t.pieces)) do j
        p = t.pieces[j]
        model, θ, r, u = _filter_inputs(prior, ys[j], ws[j], float.(t.x[p]))
        any(isfinite, θ) || return θ
        bf = _branch_filter(model, θ, r, u; level = !isnothing(groups))
        isnothing(slip) || (slip[j] = _slip_probability(model, bf.histories, r, u; level = !isnothing(groups)))
        lift = first(bf.histories).lift
        return lift .+ 2π * round(_weighted_mean_finite(θ .- lift, ws[j]) / 2π)
    end
    isnothing(groups) || _align_branches!(lifts, ws, groups)
    shift = zeros(length(φ))
    for (j, p) in pairs(t.pieces), (i, k) in pairs(p)
        n = t.slots[k]
        (free[n] && isfinite(lifts[j][i])) || continue
        shift[n] = 2π * round((lifts[j][i] - ys[j][i]) / 2π)
        φ[n] += shift[n]
    end
    return shift
end

# Move each gauge row's right-hand side by the branch shifts of its slots, so
# the constraint still holds on the gains after a change of branch.
function _shift_gauge!(gauge, shift)
    for (j, row) in pairs(gauge.rows)
        gauge.d[j] += sum((c * shift[n] for (n, c) in row); init = 0.0)
    end
    return gauge
end

# ── The solve ────────────────────────────────────────────────────────────────

# The fixed structure of the joint solve: `edge[si, bi, p, cell]` numbers the
# `(slot, slot)` pair a correlation joins, `0` for none, and `slot_pairs` lists
# them, the slots numbered by `ids` (`_gain_slot_ids`).
function _joint_gain_layout(data, layout, ids)
    (; r, ends) = data
    (; loc, tseg, fseg) = layout
    edge = zeros(Int32, size(r))
    index = Dict{Tuple{Int, Int}, Int32}()
    slot_pairs = Tuple{Int, Int}[]
    for cell in axes(r, Frequency), p in axes(r, FeedPair), bi in axes(r, AntennaPair)
        _solvable(ends[bi, p]) || continue
        (a, fa), (b, fb) = ends[bi, p]
        (ka, ia), (kb, ib) = loc[a], loc[b]
        for si in axes(r, Scan)
            ta, tb = tseg[a, si], tseg[b, si]
            (iszero(ta) || iszero(tb)) && continue
            key = (ids[ka][ia, fa, fseg[a, cell], ta], ids[kb][ib, fb, fseg[b, cell], tb])
            edge[si, bi, p, cell] = get!(index, key) do
                push!(slot_pairs, key)
                Int32(length(slot_pairs))
            end
        end
    end
    return (; edge, slot_pairs)
end

# Every (station, feed, time segment) track of one observable: its slots in
# frequency-segment order and its station's coordinates, pieces, prior and
# levels (`fits[a]`), with `active` marking the pieces its prior acts on — those
# with data or fixed values at enough slots (`known`) to determine its states.
function _bandpass_tracks(fits, observable, ids, loc, present, nfeed, known)
    tracks = NamedTuple[]
    for a in eachindex(fits, loc)
        (isnothing(fits[a]) || isnothing(fits[a][observable])) && continue
        f = fits[a][observable]
        k, ai = loc[a]
        for feed in 1:nfeed, ts in present[a]
            slots = vec(ids[k][ai, feed, :, ts])
            active = [!isnothing(f.prior) && count(n -> known[n], slots[p]) >= _samples_needed(f.prior) for p in f.pieces]
            push!(tracks, (; a, k, ai, feed, ts, slots, f.x, f.pieces, f.prior, f.level, f.nlevel, active))
        end
    end
    return tracks
end

# Whether `x′` differs from `x` by no more than `x`'s own rounding.
_within_rounding(x′, x) = all(i -> abs(x′[i] - x[i]) <= eps(abs(x[i])), eachindex(x′, x))

# The largest change of any slot's log-gain, `(Δℓ, Δφ)` with the phase change
# wrapped, in standard errors of its weight `W` from the data, and that slot.
function _largest_weighted_change(Δℓ, Δφ, W)
    worst, at = 0.0, 0
    for n in eachindex(Δℓ, Δφ, W)
        W[n] > 0 || continue
        c = sqrt(W[n]) * abs(complex(Δℓ[n], rem2pi(Δφ[n], RoundNearest)))
        c > worst && ((worst, at) = (c, n))
    end
    return worst, at
end

# `" (station, feed f, frequency segment s, time segment t)"` for gain `I` of block `k`.
function _slot_description(geom::DataGeometry, loc, k, I)
    ai, f, fs, ts = Tuple(I)
    ant = findfirst(==((k, ai)), loc)
    return " ($(geom.stations[ant]), feed $f, frequency segment $fs, time segment $ts)"
end

# A joint level writer's values at time segment `ts`.
_level_at(::Nothing, ts) = nothing
_level_at(lw, ts) = merge(lw, (; values = view(lw.values, :, :, ts)))

# Gauge-fix each (station, feed, time segment) track — zero band-mean
# log-amplitude, circular-mean reference phase — and write it into the station
# block that carries it. Both gauges
# are over the frequency track of one time segment, so each of a station's
# segments is normalized on its own. Each observable is written, and referenced,
# over the slots it has a value in; the rest are left unwritten, holding whatever
# θ already had, which is unit gain, as is a station no block covers. With a level, the reference comes off the
# level and the shape is written as fit.
#
# `gains[k]` holds the stations of phase block `k` and amplitude block `k`, in
# the blocks' own order; `level_writers[a]` holds station `a`'s
# `(; phase, amp)` level writers (see `_joint_level_writer`), each `nothing` without
# a level.
#
# The band means are removed at θ's precision, which may exceed the gains': the
# gauge is a property of θ.
function _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout, level_writers)
    for (k, gk) in pairs(gains)
        pb, ab = get(phase_blocks, k, nothing), get(amp_blocks, k, nothing)
        block = something(pb, ab)
        T = eltype(block.θ)
        (; g, φ, phase_valid, amp_valid) = gk
        node(b, f) = isnothing(b) ? 0 : _feed_node(b.plan.tying, f)
        for (ai, a) in pairs(block.stations), f in axes(g, Feed), ts in layout.present[a]
            pnode, anode = node(pb, f), node(ab, f)
            lw = map(l -> _level_at(l, ts), level_writers[a])
            pv = view(phase_valid, ai, f, :, ts)
            if !iszero(pnode) && any(pv)
                phase = fs -> T(φ[ai, f, fs, ts])
                mphase = angle(sum(cis(phase(fs)) for fs in axes(g, Frequency) if pv[fs]))
                Lp = _write_level!(lw.phase, f, ts, Lj -> rem2pi(T(Lj) - mphase, RoundNearest))
                for fs in axes(g, Frequency)
                    pv[fs] || continue
                    pb.θ[1, pnode, fs, ts, ai] = isnothing(Lp) ?
                        rem2pi(phase(fs) - mphase, RoundNearest) : phase(fs) - T(Lp[lw.phase.seg[fs]])
                end
            end
            av = view(amp_valid, ai, f, :, ts)
            if !iszero(anode) && any(av)
                logamp = fs -> log(T(abs(g[ai, f, fs, ts])))
                mlogamp = sum(logamp(fs) for fs in axes(g, Frequency) if av[fs]) / count(av)
                La = _write_level!(lw.amp, f, ts, Lj -> T(Lj) - mlogamp)
                for fs in axes(g, Frequency)
                    av[fs] || continue
                    ab.θ[1, anode, fs, ts, ai] = isnothing(La) ?
                        logamp(fs) - mlogamp : logamp(fs) - T(La[lw.amp.seg[fs]])
                end
            end
        end
    end
    return nothing
end

"""
    JointSmoother(; max_iterations = 200, tolerance = 0.01, max_slip_probability = 0.05,
                  flat_span = 0.01, warn_fraction = 0.25, eltype = nothing)

Fit the station bandpass against the actual complex visibilities, with each
component's prior acting inside the solve rather than as a fit applied to it
afterwards: the gains maximize their posterior, with the per-scan source term
profiled out and type-II MAP hyperparameters ([`solve_joint_bandpass!`](@ref)).

Because the per-scan source term absorbs a baseline's own structure, this suits a
resolved or polarized calibrator, which a closure solve that assumes the source
term cancels would bias. It solves one complex gain per
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
`tolerance` of their standard errors of the type-II MAP, and warn if
`max_iterations` iterations pass first: an unconverged solve still depends on
its starting point.

A phase prior depends on each segment's 2π branch, which the data leave open, so
every iteration chooses the branches of each phase piece as well, by a Gaussian-sum
Kalman filter over branch histories, and fits the hyperparameters on those
branches (see the [fitting-under-priors](@ref fitting-under-priors) page). The report's `phase_slip`
gives each piece's posterior probability of a 2π slip, and the solve warns when
any piece's reaches `max_slip_probability`. A fitted piece whose range is below
`flat_span` (radians or nepers) is reported flat rather than as a measured shape,
and the solve warns when more than `warn_fraction` of its pieces are flat or
without data.

The visibilities are reduced onto the frequency segments, and the gains and
source terms held, in the real type of the summed visibilities (`Float32` for
an MSv4 store); `eltype`, a real floating-point type, names another. The
linearized steps are solved in `Float64`, the type the sparse Cholesky
factorization supports.
"""
struct JointSmoother
    max_iterations::Int
    tolerance::Float64
    max_slip_probability::Float64
    flat_span::Float64
    warn_fraction::Float64
    eltype::Union{Nothing, DataType}
end
JointSmoother(;
    max_iterations::Integer = 200, tolerance::Real = 0.01, max_slip_probability::Real = 0.05,
    flat_span::Real = 0.01, warn_fraction::Real = 0.25, eltype::Union{Nothing, Type{<:AbstractFloat}} = nothing,
) = JointSmoother(max_iterations, tolerance, max_slip_probability, flat_span, warn_fraction, eltype)

can_fit(::JointSmoother, tc, geom) = _fits_bandpass_track(tc)

function validate_model(::JointSmoother, model)
    _validate_bandpass_groups(model)
    ph = _shape_component(model.phase)
    la = _shape_component(model.logamp)
    (isnothing(ph) || isnothing(la)) && return nothing
    ph.Frequency == la.Frequency || throw(
        ArgumentError(
            "JointSmoother requires the phase and logamp shapes to share one frequency " *
                "segmentation — it solves one COMPLEX gain per (station, feed, segment). Got " *
                "$(repr(ph.Frequency)) (phase) vs $(repr(la.Frequency)) (logamp).",
        ),
    )
    ph.Ti == la.Ti || throw(
        ArgumentError(
            "JointSmoother requires the phase and logamp components to share one time " *
                "segmentation — one complex gain per (station, feed, segment) is solved over " *
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
    tseg = _station_time_segments(_gain_blocks(phase_blocks, amp_blocks), results, length(geom.stations))
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
            tseg = view(tseg, :, idx),
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
    _warn_uncertain_branches(report, sm.max_slip_probability)
    return report
end

"""
    solve_joint_bandpass!(θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
                          phase_level_blocks = [], amp_level_blocks = [],
                          gauge, max_iterations = 200, tolerance = 0.01,
                          flat_span = 0.01, eltype = nothing, tseg = nothing) -> (; phase, amp)

Jointly solve the per-(station, feed, frequency segment) complex bandpass gain
and a per-scan, per-station-pair, per-feed-pair constant source coherence (see
the module comment above `_reduce_scan_segments!` for the model), then
gauge-fix each (station, feed, time segment) track and write the result into the
station blocks of the two observables (`_write_joint_bandpass!`).

`phase_blocks` and `amp_blocks` are `_bandpass_blocks(setup, θ, :phase)`
and `(…, :logamp)` — station blocks over the SAME `θ` this call is handed, since
each block's `θ` is a view into it — and `phase_level_blocks`/`amp_level_blocks`
the matching `_bandpass_level_blocks`, empty without a level. A
station-uniform model gives one block per component spanning every station;
where the model differs across stations, each block carries its own stations,
feed tying, segments and prior, and a station no block covers is left at unit
gain. Each track is fit under its own block's prior, which relates the values
within each spectral window, with the prior's hyperparameters re-estimated every
iteration.
A station's two shapes must hold the same stations on the same frequency and
time segmentations, which `validate_model(::JointSmoother, model)` enforces at
model-compile time; the throw here guards direct callers. Either list may be
empty, which holds that observable at zero: unit amplitude, or zero phase and
no gauge.

`geom` supplies the stations the `AntennaPair` labels name (θ's stations, in
θ's order) and the channel frequencies the priors are fit along.

Scaling a set of stations by one phase leaves every visibility internal to that
set unchanged, so the gauge fixes one constant per such set. The sets are the
connected components of the (station, feed, frequency segment, time segment)
graph the correlations span, and `gauge` gives one constraint per component
([`_joint_bandpass_gauge!`](@ref)): a pin holds its node at zero, and any other
constraint is imposed exactly in every step. Where stations' frequency
segmentations differ, a pin can hold part of a track; under a phase prior the
rest of the track is then fit with the pinned segment held at zero.

`scans` is the per-scan `(rl, wl)` accumulator pairs from
`accumulate_bandpass` — not summed across scans, since the source term
needs each scan's own coherent visibility.

`tseg`, when given, is the `(station, scan)` time-segment table
([`_station_time_segments`](@ref)): station `a`'s gain is solved separately for
each distinct `tseg[a, :]` value, in that station's own segment numbering, and a
station whose entry is `0` is left out of the solve. Every station shares one
segment when it is omitted. `scans` must hold the scans `tseg`'s columns
describe, in the same order. The phase gauge acts on the graph these scans
span, one constraint per connected component, so a break at one station is
measured against the stations that hold one gain across it rather than gauged
away. Solved phases are comparable only WITHIN a component: where that graph
splits — disjoint sub-arrays, or every station segmented at the same epoch —
each piece carries its own arbitrary constant, and a per-station change read
across the split is that constant plus the change.

The solve maximizes the posterior of the gains: half the χ² of the visibilities,
with `S` at its closed-form optimum for the gains, plus each track's prior energy
with its levels profiled out. `S` absorbs any constant on a (station, feed, time
segment) track, which the data therefore cannot measure, so a proper prior is
profiled over that constant as well. Each iteration:

1. linearizes the χ² in every slot's log-amplitude and phase about the current
   gains, with `S` held;
2. puts each phase spectral window on its most probable 2π branches, from every
   slot's own estimate given all the other gains and `S` (`_choose_branches!`);
3. fits the hyperparameters given as densities, all tracks together, by type-II
   MAP on the marginal likelihood of the linearized posterior (`_fit_hypers!`),
   as closely as the last step moved the gains;
4. takes one Levenberg–Marquardt step of the posterior, solved exactly as one
   sparse system over the slots, the priors' latent states and levels, with the
   gauge solved for one slot per constraint (`_map_step`), and accepts it once
   the posterior with `S` re-solved is no lower.

The solve starts from every slot's own estimate given unit gains elsewhere, and
the hyperparameters from type-II MAP on each track alone. Convergence is judged
on the largest change the undamped step makes to any (station, feed, segment)
gain's log, in standard errors of that gain from the data, so a gain the data
barely constrain counts for as little as its precision. The solve stops once
that is below `tolerance` with the hyperparameters fit to `tolerance` of their
standard errors, or once no damping lowers the posterior, with a warning if
that step would have moved a gain by `tolerance` or more: the posterior is
then flat or not convex there, typically because a prior too stiff for the data
drives an amplitude toward zero, where the mode does not exist. docs/src/priors.md
("The bandpass posterior") derives each step. The scans handed to one call are a connected piece of
the coupling graph, so this is one criterion over one coupled problem: nodes
that share no data are solved by separate calls rather than being averaged into
a common tolerance.

Returns, per observable, one `(; track, prior, status, slip)` per (station,
feed, time segment) track: the track, the prior it was fit under with its
hyperparameters resolved, the [`TrackStatus`](@ref) of each spectral window,
and for a phase track each spectral window's posterior probability of a 2π slip
from the last iteration (`nothing` for an amplitude track). The data are reduced
and the gains and source terms held in the accumulators' types, or in the real
type `eltype`; the steps are solved in `Float64`.
"""
function solve_joint_bandpass!(
        θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
        phase_level_blocks = NamedTuple[], amp_level_blocks = NamedTuple[],
        gauge::AbstractGauge, max_iterations::Integer = 200, tolerance::Real = 0.01,
        flat_span::Real = 0.01, eltype = nothing,
        tseg::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
    )
    isempty(scans) && return (; phase = [], amp = [])
    nant = length(geom.stations)
    # Both observables carry one complex gain per (station, feed, segment), so
    # their blocks must hold the same stations on the same segmentations
    # (`validate_model(::JointSmoother, model)` holds each station's tree to it;
    # the throw guards direct callers).
    _same_segmentation(pb, ab) =
        pb.stations == ab.stations && pb.plan.fseg_id == ab.plan.fseg_id && pb.plan.tseg_id == ab.plan.tseg_id
    isempty(phase_blocks) || isempty(amp_blocks) ||
        length(amp_blocks) == length(phase_blocks) && all(splat(_same_segmentation), zip(phase_blocks, amp_blocks)) || throw(
        ArgumentError(
            "solve_joint_bandpass!: the phase and logamp components give the stations " *
                "different segmentations — the solve carries one COMPLEX gain per " *
                "(station, feed, segment), not independent phase/log-amp tracks.",
        ),
    )

    # The data are reduced onto the refinement of every block's frequency
    # segmentation, and each station's gain is held in its own segments — one
    # gain over however many refinement cells that segment spans.
    blocks = _gain_blocks(phase_blocks, amp_blocks)
    fseg, cells = _station_freq_segments(blocks, nant)
    r, w = _reduce_all_scans(scans, cells, Frequency(_segment_lookup(geom.channel_freqs, cells)); eltype)
    ends = _cell_nodes(r, geom.stations, PerFeed())
    S = zeros(Base.eltype(r), (Scan(axes(r, Scan)), dims(r, AntennaPair), dims(r, FeedPair)))
    data = DimStack((; r, w, S, ends))

    # The time segments each station is actually solved for here, in its own
    # segmentation's numbering — the numbering of its block's `Ti` axis.
    tsg = isnothing(tseg) ? ones(Int, nant, length(scans)) : tseg
    loc = _block_locations(blocks, nant)
    layout = (;
        loc, tseg = tsg, fseg,
        present = [sort!(filter!(!iszero, unique(view(tsg, a, :)))) for a in axes(tsg, 1)],
    )
    gains = [_block_gains(b, geom, Base.eltype(r)) for b in blocks]
    # The gauge fixes phases; a solve that holds every phase at zero has none to fix.
    gauge_state = isempty(phase_blocks) ? nothing : _joint_bandpass_gauge!(gains, data, layout, blocks, gauge)
    gauge_rows = isnothing(gauge_state) ? nothing : (; gauge_state.rows, d = Float64.(gauge_state.d))

    segs = [_shape_segments(b.plan, geom) for b in blocks]
    phase_level = _station_levels(phase_level_blocks, segs, loc, geom, θ, real(Base.eltype(r)))
    amp_level = _station_levels(amp_level_blocks, segs, loc, geom, θ, real(Base.eltype(r)))
    fits = map(1:nant) do a
        k, ai = loc[a]
        iszero(k) && return nothing
        seg = segs[k]
        track(blocks, level) = isempty(blocks) ? nothing : (;
            seg.x, seg.pieces, prior = _prior_along(blocks[k].plan.priors[ai], :Frequency),
            level = isnothing(level) ? nothing : level.level, nlevel = isnothing(level) ? 0 : level.nlevel,
        )
        return (; phase = track(phase_blocks, phase_level[a]), amp = track(amp_blocks, amp_level[a]))
    end

    ids = _gain_slot_ids(gains)
    jl = _joint_gain_layout(data, layout, ids)
    slot = [(k, I) for (k, idk) in pairs(ids) for I in CartesianIndices(idk)]
    nslots = length(slot)
    pinned = [gains[k].pinned[I] for (k, I) in slot]
    hasdata = falses(nslots)
    for I in CartesianIndices(jl.edge)
        e = jl.edge[I]
        (iszero(e) || !(w[I] > 0)) && continue
        foreach(n -> hasdata[n] = true, jl.slot_pairs[e])
    end
    any(hasdata) || throw(ArgumentError("solve_joint_bandpass!: no gain slot has data to fit."))
    fitted = (; phase = !isempty(phase_blocks), amp = !isempty(amp_blocks))
    known = (; phase = hasdata .| pinned, amp = hasdata)
    tracks = map((:phase, :amp)) do obs
        fitted[obs] ? _bandpass_tracks(fits, obs, ids, loc, layout.present, geom.nfeed, known[obs]) : NamedTuple[]
    end
    tracks = (; phase = tracks[1], amp = tracks[2])
    # A slot moves where it has data, or where a prior acting on its track fills it.
    function movable(obs)
        out = fitted[obs] .& hasdata
        for t in tracks[obs], (j, p) in pairs(t.pieces)
            t.active[j] && (out[t.slots[p]] .= fitted[obs])
        end
        return out
    end
    free = (; phase = movable(:phase) .& .!pinned, amp = movable(:amp))
    filled = map(obs -> movable(obs) .& .!hasdata, (; phase = :phase, amp = :amp))
    priors = map(ts -> Any[t.prior for t in ts], tracks)
    acting = map(ts -> findall(t -> any(t.active), ts), tracks)
    slips = [fill(NaN, length(t.pieces)) for t in tracks.phase]
    energy(obs, x) = sum((first(_track_energy(tracks[obs][i], priors[obs][i], x)) for i in acting[obs]); init = 0.0)

    # Float64 whatever the data's type: CHOLMOD factors only Float64.
    ℓ, φ = zeros(nslots), zeros(nslots)
    gains_of(ℓ, φ) = exp.(complex.(ℓ, φ))
    _source_coherence!(S, gains_of(ℓ, φ), data, jl)
    # Start every slot at its own estimate given unit gains elsewhere, a nonlinear
    # step the linearized iterations then refine.
    lin = _linearize(gains_of(ℓ, φ), S, data, jl, nslots)
    for (obs, x) in ((:phase, φ), (:amp, ℓ))
        fitted[obs] || continue
        y, _ = _slot_estimates(x, lin, obs; phase_held = !fitted.phase)
        for n in eachindex(x)
            free[obs][n] && isfinite(y[n]) && (x[n] = y[n])
        end
    end
    _source_coherence!(S, gains_of(ℓ, φ), data, jl)

    objective(ℓ, φ, S) = _chi2(gains_of(ℓ, φ), S, data, jl) / 2 + energy(:phase, φ) + energy(:amp, ℓ)
    key = zeros(nslots)
    for ts in tracks, t in ts
        key[t.slots] .= t.x
    end
    curvature = map(ts -> Vector{Any}(nothing, length(ts)), tracks)
    change, worst, stalled, converged, hypers_fit = Inf, 0, false, false, true
    μ, ν, near = 1.0e-3, 2.0, false
    for iteration in 1:max_iterations
        lin = _linearize(gains_of(ℓ, φ), S, data, jl, nslots)
        if fitted.phase
            y, wy = _slot_estimates(φ, lin, :phase)
            for i in acting.phase
                t = tracks.phase[i]
                isone(iteration) && (priors.phase[i] = _seed_prior(t, y, wy; phase = true))
                shift = _choose_branches!(φ, t, priors.phase[i], y, wy, free.phase; slip = slips[i])
                isnothing(gauge_rows) || _shift_gauge!(gauge_rows, shift)
            end
        end
        if fitted.amp && isone(iteration)
            y, wy = _slot_estimates(ℓ, lin, :amp; phase_held = !fitted.phase)
            for i in acting.amp
                priors.amp[i] = _seed_prior(tracks.amp[i], y, wy; phase = false)
            end
        end
        # Levenberg–Marquardt: each observable's step is damped by `μ` times the
        # largest data curvature of its slots, a trust region in log-gain, where
        # the linearization's error is measured, and `μ` follows how well the
        # quadratic model predicted the objective's actual decrease (Madsen,
        # Nielsen & Tingleff 2004, §3.2). `μ` stays at least the curvature's own
        # precision, so a direction neither the data nor a prior determine takes
        # no step, and a slot with no curvature above that precision, from its data
        # or a prior, does not move.
        H = (; phase = _data_curvature(lin.q, jl.slot_pairs, -1, nslots), amp = _data_curvature(lin.q, jl.slot_pairs, 1, nslots))
        b = (; phase = imag.(lin.c), amp = real.(lin.c))
        scale = map(h -> maximum(diag(h); init = 0.0), H)
        systems = map((; phase = :phase, amp = :amp)) do obs
            fitted[obs] || return nothing
            d = diag(H[obs])
            moving = free[obs] .& ((d .> eps() * scale[obs]) .| filled[obs])
            return _posterior_system(
                H[obs], b[obs], obs === :phase ? φ : ℓ, moving, tracks[obs][acting[obs]], key;
                gauge = obs === :phase ? gauge_rows : nothing,
            )
        end
        # The hyperparameters' optimum moves with the linearization, so they are
        # fit only as closely as the gains last moved, and to `tolerance` once the
        # gains have converged.
        hypers_fit = all((:phase, :amp)) do obs
            fitted[obs] || return true
            p = priors[obs][acting[obs]]
            ok = _fit_hypers!(
                p, view(curvature[obs], acting[obs]), systems[obs], tracks[obs][acting[obs]], eps() * scale[obs];
                tolerance = max(tolerance, change), max_iterations,
            )
            priors[obs][acting[obs]] = p
            return ok
        end
        function propose(μ)
            steps = map((:phase, :amp)) do obs
                fitted[obs] || return (obs === :phase ? φ : ℓ), 0.0
                x′, _, model = _map_step(systems[obs], priors[obs][acting[obs]]; λ = μ * scale[obs])
                return x′, model
            end
            return (; phase = steps[1][1], amp = steps[2][1]), steps[1][2] + steps[2][2]
        end
        # A damped step understates the change; converged once the undamped step
        # is within `tolerance` too.
        if near
            x1, _ = propose(eps())
            change, worst = _largest_weighted_change(x1.amp .- ℓ, x1.phase .- φ, lin.W)
            change < tolerance && hypers_fit && (converged = true; break)
        end
        O0 = objective(ℓ, φ, S)
        E0 = energy(:phase, φ) + energy(:amp, ℓ)
        accepted = false
        while isfinite(μ)
            x1, predicted = propose(μ)
            change, worst = _largest_weighted_change(x1.amp .- ℓ, x1.phase .- φ, lin.W)
            _within_rounding(x1.amp, ℓ) && _within_rounding(x1.phase, φ) && break
            S1 = _source_coherence!(similar(S), gains_of(x1.amp, x1.phase), data, jl)
            O1 = objective(x1.amp, x1.phase, S1)
            # The objective is known to its rounding.
            if O1 <= O0 + 16 * eps(abs(O0))
                ρ = (O0 - O1) / (E0 - predicted)
                ℓ, φ, S = x1.amp, x1.phase, S1
                μ = max(μ * max(1 / 3, 1 - (2ρ - 1)^3), eps())
                ν = 2.0
                accepted = true
                break
            end
            μ *= ν
            ν *= 2
        end
        # A step lowers the objective by about half its squared change in
        # standard errors, so one too small for that to exceed the objective's
        # rounding cannot be verified, and ending there is not a stall.
        if !accepted
            stalled = change >= tolerance && change^2 / 2 > 16 * eps(abs(O0))
            converged = !stalled
            break
        end
        near = change < tolerance
    end
    location = iszero(worst) ? "" : _slot_description(geom, loc, slot[worst]...)
    stalled && @warn "the joint bandpass solve stopped at a step that could not lower its MAP " *
        "objective at any damping; that step would have moved a gain by $change of its standard " *
        "error$location. The objective is flat or not convex near this point, often because a prior " *
        "is too stiff for the data, which can drive a gain's amplitude toward zero."
    # Where a slot's phase is held more than π/2 from its own data — by a phase
    # prior or the gauge — the amplitude that fits it best is zero, and without an
    # amplitude prior the posterior has no mode: its log-amplitude runs off.
    if fitted.amp && (stalled || converged)
        lin = _linearize(gains_of(ℓ, φ), S, data, jl, nslots)
        unpriored = free.amp .& .!filled.amp
        for t in tracks.amp, (j, p) in pairs(t.pieces)
            t.active[j] && (unpriored[t.slots[p]] .= false)
        end
        modeless = [n for n in findall(unpriored) if lin.W[n] > 0 && real(1 + lin.c[n] / lin.W[n]) <= 0]
        isempty(modeless) || @warn "the joint bandpass posterior has no mode for the amplitude of " *
            "$(length(modeless)) gain slots, e.g.$(_slot_description(geom, loc, slot[first(modeless)]...)): " *
            "their phase is held more than π/2 from their data, by a phase prior or the gauge, so the " *
            "amplitude that fits them best is zero. Give the amplitude a prior."
    end
    hypers_fit || @warn "the joint bandpass hyperparameter fit did not converge in $max_iterations " *
        "steps at the last iteration; raise `JointSmoother(; max_iterations)`"
    stalled || converged || @warn "the joint bandpass solve did not converge in " *
        "$max_iterations iterations: the last step moved a gain by $change of its standard error$location, " *
        "against a tolerance of $tolerance; raise `JointSmoother(; max_iterations)`"

    for (n, (k, I)) in pairs(slot)
        gk = gains[k]
        gk.g[I] = exp(complex(ℓ[n], φ[n]))
        gk.φ[I] = φ[n]
        gk.phase_valid[I] = free.phase[n] || (fitted.phase && pinned[n])
        gk.amp_valid[I] = free.amp[n]
    end
    levels = [(; phase = _level_values(phase_level[a]), amp = _level_values(amp_level[a])) for a in 1:nant]
    fits = map((:phase, :amp)) do obs
        x = obs === :phase ? φ : ℓ
        valid = obs === :phase ? free.phase .| (fitted.phase .& pinned) : free.amp
        map(eachindex(tracks[obs])) do i
            t = tracks[obs][i]
            status = map(t.pieces) do p
                all(pinned[t.slots[p]]) && obs === :phase && return TrackSolved
                return _band_track_status([valid[n] ? x[n] : NaN for n in t.slots[p]], flat_span)
            end
            if !isnothing(t.level) && any(t.active)
                _, L = _track_energy(t, priors[obs][i], x)
                for (g, v) in L
                    levels[t.a][obs][t.feed, g, t.ts] = v
                end
            end
            return (; track = t, prior = priors[obs][i], status, slip = obs === :phase ? slips[i] : nothing)
        end
    end
    level_writers = [
        (; phase = _joint_level_writer(phase_level[a], levels[a].phase), amp = _joint_level_writer(amp_level[a], levels[a].amp))
            for a in 1:nant
    ]
    _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout, level_writers)
    return (; phase = fits[1], amp = fits[2])
end

# Record a joint solve's track fits (`solve_joint_bandpass!`) into one array per
# station block of their status (`_block_status_array`), prior
# (`_block_prior_array`) and slip probability (`_block_slip_array`), each
# `nothing` to skip. A solve over part of a track's time writes only its own
# time segments.
function _record_track_fits!(status, priors, slip, fits)
    for f in fits
        t = f.track
        isnothing(status) || (status[t.k][t.ai, t.feed, :, t.ts] = f.status)
        isnothing(priors) || (priors[t.k][t.ai, t.feed, t.ts] = f.prior)
        isnothing(slip) || isnothing(f.slip) || (slip[t.k][t.ai, t.feed, :, t.ts] = f.slip)
    end
    return nothing
end

# Each station's level for one observable, `nothing` where it has none: its
# piece → level map and level count against the station's shape segments, its
# level block's leaf, feed tying and position, the level of each shape segment,
# and its time-segment count.
function _station_levels(level_blocks, segs, loc, geom::DataGeometry, θ, T)
    out = Vector{Any}(nothing, length(loc))
    for lb in level_blocks, (li, a) in pairs(lb.stations)
        k, _ = loc[a]
        iszero(k) && continue
        level, nlevel = _piece_levels(lb.plan, segs[k])
        out[a] = (;
            level, nlevel, leaf = lb.θ, tying = lb.plan.tying, ai = li,
            seg = _segment_levels(level, segs[k]), nts = lb.plan.shape[4], geom.nfeed, T,
        )
    end
    return out
end

_level_values(::Nothing) = nothing
_level_values(l) = fill(l.T(NaN), l.nfeed, l.nlevel, l.nts)

_joint_level_writer(::Nothing, values) = nothing
_joint_level_writer(l, values) = (; l.leaf, l.tying, l.seg, values, l.ai)
