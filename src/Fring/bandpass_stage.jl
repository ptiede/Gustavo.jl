# ── Bandpass stage: per-segment station phase/log-amp over scan windows ──────
#
# The `Bandpass` solver reads every scan (accumulate → return the scan's
# contribution) and folds the contributions in group order, which keeps the
# result deterministic at any concurrency.
# What is fit is the solver's model tree: per observable, any components whose
# time segments are whole scans, each under a prior. How it is solved is
# `JointSmoother`, which runs [`solve_joint_bandpass!`](@ref), fitting the
# complex visibilities against a per-scan source term profiled out, with the
# priors entering inside the solve.
#
# This file holds the solver's side: model checks, accumulation, outcome codes and
# the report. The data reduction is in bandpass_data.jl, the station gains in
# station_gains.jl, the joint solve in joint_gains.jl over the priors of
# bandpass_priors.jl and the profiled MAP of bandpass_map.jl, and the bandpass
# entry point in bandpass_solve.jl. The prior fits live in prior_fits.jl.

# A bandpass component's time segments are whole scans (`PerScan`) or span
# several (`GlobalTime`, `InstrumentScans`, `TimeBlocks`), and the scans a
# segment spans are solved together; and it carries a prior: an `IIDPrior`, or
# a state-space prior along frequency whose start is proper.
_fits_bandpass_track(tc) = tc.Ti isa Union{PerScan, GlobalTime, InstrumentScans, TimeBlocks} && _bandpass_prior(tc)

function _bandpass_prior(tc)
    tc.prior isa IIDPrior && return true
    p = resolve_prior(tc)
    return p isa NamedTuple && keys(p) == (:Frequency,) && _proper_prior(p.Frequency)
end

# The structure every bandpass model must have: at least one component
# overall, and per observable group at most one component with a prior along
# frequency, whose chains the report records.
function _validate_bandpass_groups(model)
    np = length(Calibration._flatten_components(model.phase))
    na = length(Calibration._flatten_components(model.logamp))
    np + na >= 1 || throw(
        ArgumentError(
            "Bandpass model fits nothing: the model compiles no components, so the solver " *
                "would accumulate every scan and write nowhere. Add a phase and/or logamp component.",
        ),
    )
    for name in (:phase, :logamp)
        comps = Calibration._flatten_components(getproperty(model, name))
        chained = [c for c in comps if !(c.prior isa IIDPrior)]
        length(chained) <= 1 || throw(
            ArgumentError(
                "Bandpass model group `$name` holds $(length(chained)) components with a prior along " *
                    "frequency, " * join(map(component_label, chained), ", ") * "; at most one takes one.",
            ),
        )
    end
    return nothing
end

# The paths to a bandpass observable's components in a layout's plantree:
# `fitted`, every component; `chained`, the one with a prior along frequency,
# `nothing` for none.
function _bandpass_paths(plantree, group::Symbol)
    fitted = _leaf_paths(plantree[group], (group,))
    chained = findfirst(p -> any(!(q isa IIDPrior) for q in _plan_priors(_plan_node(plantree, p))), fitted)
    return (; fitted, chained = isnothing(chained) ? nothing : fitted[chained])
end

_plan_node(tree, path) = foldl((node, k) -> node[k], path; init = tree)

_plan_priors(plan::ComponentPlan) = plan.priors
_plan_priors(g::Calibration.GroupedComponentPlan) = reduce(vcat, [p.priors for p in values(g.groups)])

# Each of an observable's (`:phase` or `:logamp`) fitted components as its
# station blocks `(; antennas, θ, plan)`. The paths are resolved by name:
# `layout.plans` holds one entry per station-signature group, so its positions
# do not name the components once a model differs across stations.
_bandpass_components(setup, θ, group::Symbol) = [antenna_blocks(setup.layout, θ, path...) for path in setup.paths[group].fitted]

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
is below `JointSmoother`'s `flat_span`). θ alone cannot tell these apart: an
unfitted piece reads back as unit gain and a flat one as a constant.
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

# A station block's resolved prior per (station, feed, time segment), over
# `(AntennaName, Feed, Ti)` labeled as `_block_status_array`; each entry starts at
# its station's own prior.
function _block_prior_array(block, geom::DataGeometry)
    (; plan) = block
    ax = (_antenna_dim(geom.antennas[block.antennas]), Feed(1:geom.nfeed), Ti(_time_segment_lookup(plan, geom)))
    arr = DimArray(Array{Union{Nothing, AbstractPrior}}(undef, map(length, ax)), ax)
    for a in axes(arr, 1)
        p = plan.priors[a]
        arr[a, :, :] .= Ref(p isa NamedTuple ? _prior_along(p, :Frequency) : p)
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
                          amp_priors = nothing) -> NamedTuple

Summarize a bandpass solve's per-piece outcomes into the record the
[`Bandpass`](@ref Gustavo.Bandpass) solver publishes. `phase_status`/`amp_status`
hold one [`TrackStatus`](@ref) per (station, feed, frequency segment, time segment)
— either may be `nothing` when that half was not fit — as `DimArray`s over
`(AntennaName, Feed, Frequency, Ti)`: the stations, every feed, each frequency
segment of the component over the extent of its channels and each time segment
over the extent of its samples. `phase_priors`/`amp_priors` hold the prior each
(station, feed, time segment) track was fit under, its hyperparameters resolved,
over `(AntennaName, Feed, Ti)`. Where the model gives stations different segmentations,
[`JointSmoother`](@ref) passes one array per station block, keyed `g1, g2, …`
as the solution's components name the station blocks, each over its own
stations and segments.

Returns the arrays as `phase_status`/`amp_status`/`phase_priors`/`amp_priors`
alongside the counts `n_solved`/`n_flat`/`n_nodata` summed over both
observables. `flat` and `nodata` are the two ways a piece can take up θ entries
without measuring anything, and they are what the counts exist to expose: θ
itself records an unfitted piece as unit gain and a starved one as a constant,
neither distinguishable there from a genuinely flat response.
"""
function _bandpass_track_report(phase_status, amp_status; phase_priors = nothing, amp_priors = nothing)
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
    Bandpass: $(round(100 * frac; digits = 1))% of (antenna, feed, frequency segment) pieces \
    carry no measured frequency shape — $(report.n_flat) fit flat, $(report.n_nodata) \
    with no usable data (of $total). They are unit gain or a constant in the solution, not a measured \
    response. Check per-segment SNR and the detection coverage of the bandpass scans.
    """ n_solved = report.n_solved
    return nothing
end

# Each (component, station) pair's own time segment for each scan:
# `tseg[(k − 1)·nant + a, i]` is the segment `results[i]` falls in under the
# segmentation component `k` gives station `a`, and `0` where the component does
# not reach the station. A scan lies wholly inside one segment of any
# segmentation the smoother accepts, so its first sample (`res.ti`) names the
# segment.
function _component_time_segments(components, results, nant)
    tseg = zeros(Int, Base.OneTo(nant * length(components)), axes(results, 1))
    for (k, c) in pairs(components), b in c, a in b.antennas, (i, res) in pairs(results)
        tseg[(k - 1) * nant + a, i] = b.plan.tseg_id[res.ti]
    end
    return tseg
end

# The `results` indices of each independently solvable scan group, given a
# time-segment table with one row per (component, station) pair
# (`_component_time_segments`).
#
# Two scans share a parameter only through a row that is in the same one of its
# own time segments in both, so the coupling graph has one node per (row,
# segment) and one edge per pair of rows a scan holds together, and the joint
# solve runs once per connected component. With one segmentation for the whole
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
    JointSmoother(; max_iterations = 200, max_hyper_iterations = 100, tolerance = 0.01,
                  flat_span = 0.01, warn_fraction = 0.25, eltype = nothing,
                  linsolve = KrylovJL_CG(; linesearch = true), hyper_optimizer = LBFGSB(m = 20))

Fit the station bandpass against the actual complex visibilities, with each
component's prior acting inside the solve rather than as a fit applied to it
afterwards: the components' parameters maximize their posterior with the
per-scan source term profiled out, and the hyperparameters given as densities
are at their type-II MAP, maximizing their Laplace evidence
([`solve_joint_bandpass!`](@ref), [`solve_joint_gains!`](@ref)).

Because the per-scan source term absorbs a baseline's own structure, this suits
a resolved or polarized calibrator. A constant on any (station, feed) over a
scan is absorbed by it, so the data measure the bandpass up to that constant,
which the priors set.

It accepts components of any term, frequency segmentation and feed tying whose
time segments are whole scans (`PerScan`, `GlobalTime`, `InstrumentScans`,
`TimeBlocks`), each under a prior: an `IIDPrior(σ)` in the parameter's own units, or a
state-space prior along `Frequency` whose start is proper (an `OUPrior`, a
`CARMAPrior`, or a `RandomWalkPrior` with an `init`). At most one component per
observable takes a prior along frequency; for example a shape beside a level
and a delay,

```julia
shape = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1),
                      Feed = PerFeed(), prior = OUPrior(; scale = 50e6, σ = 0.1))
level = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = PerSpectralWindow(),
                      Feed = PerFeed(), prior = IIDPrior(1.0))
delay = GainComponent(Delay(); Ti = GlobalTime(), Feed = PerFeed(), prior = IIDPrior(1e-9))
GainModel(; phase = (; level, delay, shape), logamp = (; shape))
```

Set a station's own prior with [`with_antenna`](@ref). The scans of one time
segment are solved together; stations may have different models.

A component with `Ti = PerScan()` holds values that change from scan to scan,
such as a phase level and a delay per group of channels,

```julia
groups = FreqGroups([1:1, 2:13, 14:25, 26:32])
level = GainComponent(ConstantTerm(); Ti = PerScan(), Frequency = groups, Feed = SharedFeeds(), prior = IIDPrior(1.0))
delay = GainComponent(Delay(); Ti = PerScan(), Frequency = groups, Feed = SharedFeeds(), prior = IIDPrior(1e-10))
model = with_antenna(model, "NN"; phase = (; bandpass = shape, level, delay))
```

Its mean over the scans of a time segment trades against the components that
span them, and a change common to every station of a scan is not seen by the
data; the priors set both. With `N` scans, an `IIDPrior` holds the per-scan
values near a mean of zero once `N/σ²` outweighs the other components' prior
curvature. The gauge acts on the components that span scans, so a scan without
the reference needs none of its own. The start puts each scan's station-gain
phases on the 2π branch nearest their mean over the scans, so the data should
be fringe-corrected first: at every channel, each scan's phase must lie within
half a turn of that mean.

Each MAP solve stops once the undamped step would move no station gain by more
than `tolerance` of its standard error, and warns if `max_iterations` steps pass
first. The hyperparameters are fit by `hyper_optimizer`, an
[Optimization.jl](https://docs.sciml.ai/Optimization/stable/) optimizer that
takes a gradient and bounds (L-BFGS-B by default), over their logs within the
support the samples measure; the fit stops once an iteration moves no station
gain by more than `tolerance` of its standard error, from one converged MAP to
the next, or the optimizer stops, with a warning if `max_hyper_iterations`
iterations pass first. Each step is a
linear system over every parameter, solved matrix-free by `linsolve`, a Krylov
method of [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/stable/) (e.g.
`KrylovJL_CG`), preconditioned by a sparse Cholesky factorization. Its system
is the Hessian of the posterior, which is not positive definite far from the
mode, so a conjugate-gradient method should stop at nonpositive curvature
(`linesearch = true`).

A phase prior depends on each segment's 2π branch, which the data leave open, so
the solve chooses the branches of each phase chain once, by a Gaussian-sum
Kalman filter over branch histories, before the MAP solve (see the
[fitting-under-priors](@ref fitting-under-priors) page). A chain's spectral
window whose station-gain values span less than `flat_span` (radians or nepers)
is reported flat rather than as a measured shape, and the solve warns when
more than `warn_fraction` of the windows are flat or without data.

The visibilities are reduced onto the common refinement of the components'
frequency segments in the real type of the summed visibilities (`Float32` for
an MSv4 store); `eltype`, a real floating-point type, names another. The steps
are solved in `Float64`, the type the sparse Cholesky factorization supports.
"""
struct JointSmoother{E, L, O}
    max_iterations::Int
    max_hyper_iterations::Int
    tolerance::Float64
    flat_span::Float64
    warn_fraction::Float64
    eltype::E
    linsolve::L
    hyper_optimizer::O
end
JointSmoother(;
    max_iterations::Integer = 200, max_hyper_iterations::Integer = 100, tolerance::Real = 0.01,
    flat_span::Real = 0.01, warn_fraction::Real = 0.25,
    eltype::Union{Nothing, Type{<:AbstractFloat}} = nothing, linsolve = LinearSolve.KrylovJL_CG(; linesearch = true),
    hyper_optimizer = LBFGSB(m = 20),
) = JointSmoother{_eltype_parameter(eltype), typeof(linsolve), typeof(hyper_optimizer)}(
    max_iterations, max_hyper_iterations, tolerance, flat_span, warn_fraction, eltype, linsolve, hyper_optimizer,
)

_eltype_parameter(::Nothing) = Nothing
_eltype_parameter(::Type{T}) where {T} = Type{T}

can_fit(::JointSmoother, tc, geom) = _fits_bandpass_track(tc)

validate_model(::JointSmoother, model) = _validate_bandpass_groups(model)

function _solve_bandpass!(sm::JointSmoother, θ, results, setup; gauge::AbstractGauge, executor = SerialScheduler())
    geom = setup.geom
    components = (; phase = _bandpass_components(setup, θ, :phase), logamp = _bandpass_components(setup, θ, :logamp))
    chained = map(p -> isnothing(p.chained) ? NamedTuple[] : antenna_blocks(setup.layout, θ, p.chained...), setup.paths)
    tseg = _component_time_segments(vcat(components.phase, components.logamp), results, length(geom.antennas))
    arrays(f, blocks) = isempty(blocks) ? nothing : [f(b, geom) for b in blocks]
    leaves(arrays) = isnothing(arrays) ? nothing : _block_leaves(arrays)
    status = map(blocks -> arrays(_block_status_array, blocks), chained)
    priors = map(blocks -> arrays(_block_prior_array, blocks), chained)
    for idx in _joint_scan_groups(tseg)
        fits = solve_joint_bandpass!(
            results[idx], geom, components;
            gauge, sm.max_iterations, sm.max_hyper_iterations, sm.tolerance, sm.flat_span, sm.eltype,
            sm.linsolve, sm.hyper_optimizer, executor,
        )
        _record_track_fits!(status.phase, priors.phase, fits.phase, fits.status.phase, setup.paths.phase.chained, geom, chained.phase)
        _record_track_fits!(status.logamp, priors.logamp, fits.amp, fits.status.logamp, setup.paths.logamp.chained, geom, chained.logamp)
    end
    report = _bandpass_track_report(
        leaves(status.phase), leaves(status.logamp);
        phase_priors = leaves(priors.phase), amp_priors = leaves(priors.logamp),
    )
    _warn_degenerate_bandpass(report, sm.warn_fraction)
    return report
end
