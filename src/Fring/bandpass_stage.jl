# ── Bandpass stage: per-segment station phase/log-amp over scan windows ──────
#
# The `Bandpass` step reads every scan (accumulate → return the scan's
# contribution) and folds the contributions in group order, which keeps the
# result deterministic at any concurrency.
# What is fit is the step's model tree (see [`default_bandpass_terms`](@ref)):
# per observable, a shape component with a value per `ChannelBlocks` segment,
# optionally beside a level component at spectral-window resolution or coarser.
# A prior on the shape relates its values within each spectral window. How it is
# solved is pluggable through `AbstractBandpassSmoother`, in two tiers.
# `PerTrackSmoother` sums every scan's residual into one accumulator, runs the
# per-segment closure solves — which assume a baseline's source term cancels —
# and fits each resulting (station, feed) track under its component's prior.
# `JointSmoother` runs [`solve_joint_bandpass!`](@ref) instead, fitting the
# complex visibilities against an explicit per-scan source term, for when that
# assumption fails, with the priors entering inside the gain update.
#
# The graph/solve helpers (`_solve_observable!`, `_node`) live
# in adhoc.jl; the prior fits live in prior_fits.jl.

"""
    default_bandpass_terms(; freq = ChannelBlocks(1)) -> GainModel

The default [`Bandpass`](@ref Gustavo.Bandpass) step model: a time-stable, per-feed
`ConstantTerm` per frequency segment for each observable — a `phase.bandpass`
and a `logamp.bandpass` component with no prior. `freq` (a `ChannelBlocks`)
sets the segments; `ChannelBlocks(1)` is a free value per channel.

Either group of a `Bandpass` model may be empty. Fit one observable only by
keeping just that group, e.g.

    Bandpass(model = GainModel(; phase = default_bandpass_terms().phase),
             smoother = PerTrackSmoother())

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

"""
    AbstractBandpassSmoother

How the [`Bandpass`](@ref Gustavo.Bandpass) step turns the accumulated
per-channel residual into station bandpass tracks, under each component's
prior. Concretely [`PerTrackSmoother`](@ref), which solves the per-segment closures
and fits each track, or [`JointSmoother`](@ref), which fits the complex
visibilities against an explicit per-scan source term.

# Implementing a smoother

Define:

    Gustavo.Fring.can_fit(sm::MySmoother, tc::Calibration.GainComponent, geom) -> Bool
    Gustavo.Fring.solve_bandpass!(sm::MySmoother, θ, results, setup; gauge) -> report

[`can_fit`](@ref) declares which model components the smoother can solve; it
defaults to `false`, so an undeclared model is rejected at compile time
rather than leaving θ blocks unsolved. Both shipped smoothers accept
`GainComponent(ConstantTerm(); Ti = <GlobalTime, InstrumentScans or
TimeBlocks>, Frequency = <any segmentation>, Feed = PerFeed(), prior = <nothing,
or a RandomWalkPrior or OUPrior along Frequency>)` and nothing else — a
time segmentation whose segments each span several scans, solved one segment at
a time.

`solve_bandpass!` writes into `θ`'s bandpass blocks. `results` is the
per-scan `(; rl, wl, ti, source)` list in group-index order: `rl` and `wl` are
`accumulate_bandpass`'s sums, `ti` the scan's first sample on the
solve's time axis (hence which time segment it falls in). `setup` is
`(; station_pairs, feeds, layout, geom, paths)`, built once per
solve: `station_pairs` and `feeds` label the sums' `AntennaPair` and `FeedPair`
axes, and `geom` is the solve's `DataGeometry`, whose `stations` are θ's
stations. Reach each observable's parameters through
[`bandpass_blocks`](@ref)`(setup, θ, :phase)` / `(…, :logamp)`, and its level
through [`bandpass_level_blocks`](@ref bandpass_blocks), rather than the paths directly.
`report` is published on the step's solution record and
should say which tracks were measured (see [`bandpass_track_report`](@ref));
θ alone cannot distinguish a measured flat response from an unfitted one.
Return `nothing` to report nothing.

One optional hook:

    Gustavo.Fring.validate_model(sm::MySmoother, model)

`validate_model` receives the whole `(; phase, logamp)` tree at compile time for requirements
`can_fit` cannot express per component. A new method replaces the default,
so it must re-establish the base checks, available as
[`validate_bandpass_groups`](@ref).
"""
abstract type AbstractBandpassSmoother end

# A bandpass track is one value per frequency segment, per feed, held over a
# stretch of time, under a prior along frequency. Both smoothers pool the scans
# of a time segment and solve that stretch as a unit, so any segmentation whose
# segments span scans fits: `GlobalTime` is the whole track, `InstrumentScans`
# breaks it at named epochs, `TimeBlocks` at a fixed cadence. `PerScan` and
# `PerIntegration` do not — a segment holding one scan leaves the band mean of
# the gain degenerate with that scan's own source coherence, which needs the
# deviation-from-template scheme neither smoother implements.
_fits_bandpass_track(tc) =
    tc.term isa ConstantTerm &&
    tc.Ti isa Union{GlobalTime, InstrumentScans, TimeBlocks} &&
    tc.Feed isa PerFeed &&
    _is_prior_along(resolve_prior(tc), :Frequency)

"""
    validate_bandpass_groups(model)

The structural requirement the [`Bandpass`](@ref Gustavo.Bandpass) step
places on every smoother's model: at least one component overall, and per
observable group either one component or a shape and its level. In a pair, the
shape is the component on `ChannelBlocks` and carries a proper zero-mean prior
(an `OUPrior`, or a `RandomWalkPrior` with an `init`);
the level is the other, on the same time segmentation, with no prior. The
default `validate_model` for [`AbstractBandpassSmoother`](@ref), and the base a
smoother's own method must re-establish.
"""
function validate_bandpass_groups(model)
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

validate_model(sm::AbstractBandpassSmoother, model) = validate_bandpass_groups(model)

# The paths to a bandpass observable's shape and level components in a step
# layout's plantree, each `nothing` when absent. `validate_bandpass_groups`
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

"""
    bandpass_blocks(setup, θ, group::Symbol) -> Vector
    bandpass_level_blocks(setup, θ, group::Symbol) -> Vector

The station blocks of the bandpass's `:phase` or `:logamp` shape component, or
of its level component — [`Calibration.station_blocks`](@ref) resolved against
`setup`'s recorded paths — empty when the model compiles no such component for
that observable. Each block is `(; stations, θ, plan)`, and a station-uniform
model yields exactly one block spanning every station.

The paths are resolved by NAME through the layout's component tree: the flat
`layout.plans` list holds one entry per signature group, so its positions do not
name the components once a model differs across stations.
"""
bandpass_blocks(setup, θ, group::Symbol) = _path_blocks(setup, θ, setup.paths[group].shape)
bandpass_level_blocks(setup, θ, group::Symbol) = _path_blocks(setup, θ, setup.paths[group].level)

_path_blocks(setup, θ, ::Nothing) = NamedTuple[]
_path_blocks(setup, θ, path) = station_blocks(setup.layout, θ, path...)

function solve_bandpass! end
solve_bandpass!(sm::AbstractBandpassSmoother, θ, results, setup; gauge) =
    error(
    "$(typeof(sm)) does not implement the bandpass smoother interface: define " *
        "Gustavo.Fring.solve_bandpass!(::$(typeof(sm)), θ, results, setup; gauge)."
)

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

# Free per-segment closure seed for the phase bandpass: the globally-closing
# per-feed phase solved independently in each frequency segment, plus each
# (station, feed, segment)'s Fisher weight — the summed weight of the gated rows
# touching it, which is the diagonal of the segment's normal matrix and so the
# per-segment precision the prior fit weights the track by. Both are over
# `(AntennaName(stations), Feed, segments)`, in the sums' real type; the seed solves each
# feed as its own node.
function _seed_phase_tracks(
        rbar_bp, wbar_bp, stations, fsegs, segments::Frequency;
        gauge::AbstractGauge, snr_floor::Real = 1.0,
        component::Tuple{Vararg{Symbol}} = (),
    )
    T = real(eltype(rbar_bp))
    nodes = _cell_nodes(rbar_bp, stations, PerFeed())
    val = zeros(T, dims(nodes))
    wt = zeros(T, dims(nodes))
    mask = fill!(similar(nodes, Bool), false)
    nfeed = maximum(maximum, lookup(rbar_bp, FeedPair))
    tracks = (_station_dim(stations), Feed(1:nfeed), segments)
    phase = fill!(zeros(T, tracks), T(NaN))
    prec = zeros(T, tracks)
    solved = falses(length(stations), nfeed)
    for (fs, chans) in enumerate(fsegs)
        fill!(mask, false)
        for bi in axes(nodes, AntennaPair), p in axes(nodes, FeedPair)
            c = (AntennaPair(bi), FeedPair(p))
            _solvable(nodes[c...]) || continue
            (a, fa), (b, fb) = nodes[c...]
            r, w = _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            snr2 = abs2(r) / w
            snr2 >= snr_floor^2 || continue
            val[c...] = angle(r)
            wt[c...] = snr2
            mask[c...] = true
            prec[a, fa, fs] += snr2
            prec[b, fb, fs] += snr2
        end
        _solve_observable!(view(phase, Frequency(fs)), solved, val, wt, mask, nodes, gauge; rewrap = 0, component)
    end
    return phase, prec
end

# Gauge and write a solved phase bandpass `(AntennaName, Feed, Frequency)` over the
# plan's frequency segments: each (station, feed) track is referenced to its
# circular-mean phase over segments, so the bandpass applies zero net phase.
# `phase`'s `AntennaName` axis is θ's stations, in θ's order. With a `level`
# (`_level_writer`), `phase` holds level plus shape; the reference comes off the
# level and the shape is written as fit.
function _write_phase_bandpass!(θ, plan, phase, ts::Integer = 1; level = nothing)
    leaf = _component_leaf(plan, θ)
    for a in axes(phase, AntennaName), f in axes(phase, Feed)
        node = _feed_node(plan.tying, f)
        node == 0 && continue
        track = view(phase, AntennaName(a), Feed(f))
        acc = sum(v -> isfinite(v) ? cis(v) : zero(complex(v)), track)
        abs(acc) > 0 || continue
        m = angle(acc)
        L = _write_level!(_station_level(level, a), f, ts, Lj -> rem2pi(Lj - m, RoundNearest))
        for fs in axes(track, Frequency)
            v = track[fs]
            isfinite(v) || continue
            leaf[1, node, fs, ts, a] = isnothing(L) ? rem2pi(v - m, RoundNearest) : v - L[level.seg[fs]]
        end
    end
    return θ
end

# A level component's writer: its leaf and feed tying, the level segment of each
# shape segment `seg`, and the fitted `values` over `(AntennaName, Feed, level segment)`.
_level_writer(::Nothing, θ, seg, values) = nothing
_level_writer(plan, θ, seg, values) = (; leaf = _component_leaf(plan, θ), tying = plan.tying, seg, values)

# One station's view of a level writer: `values` over `(Feed, level segment)`,
# written at position `ai` of the leaf's `AntennaName` axis.
_station_level(::Nothing, a) = nothing
_station_level(level, a) = (; level.leaf, level.tying, level.seg, values = view(level.values, a, :, :), ai = a)

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


# Narrow-spike guard: additive contamination (pcal tones, RFI) violates the
# multiplicative gain model — a contaminated channel shows excess amplitude,
# the fit hands it |g| > 1, and applying that gain would then UP-weight it
# (w → w·|g|²), amplifying exactly the channels that should be distrusted.
# Genuine passband structure is smooth or negative (roll-off), so narrow
# positive log-amp outliers vs the per-(station, feed, piece) robust scale are
# excised (left unapplied, |g| = 1) instead of trusted. `spike_sigma = 0`
# disables the guard. `pieces` holds the channel groups the scale is taken over.
function _spike_guard!(la, pieces, spike_sigma::Real)
    spike_sigma > 0 || return la
    T = eltype(la)
    for sidx in pieces
        for a in axes(la, AntennaName), f in axes(la, Feed)
            track = view(la, AntennaName(a), Feed(f))
            v = [track[s] for s in sidx if isfinite(track[s])]
            length(v) >= 8 || continue
            med = median(v)
            cut = T(spike_sigma) * max(T(1.4826) * median(abs.(v .- med)), T(0.02))
            for s in sidx
                isfinite(track[s]) && track[s] - med > cut && (track[s] = T(NaN))
            end
        end
    end
    return la
end

# Gauge and write a solved log-amp bandpass `(AntennaName, Feed, Frequency)` over the
# plan's frequency segments: zero band-mean per (station, feed), so the bandpass
# applies unit net amplitude. `la`'s `AntennaName` axis is θ's stations, in θ's order.
# With a `level`, `la` holds level plus shape; the mean comes off the level and
# the shape is written as fit.
function _write_amp_bandpass!(θ, plan, la, max_logamp::Real, ts::Integer = 1; level = nothing)
    leaf = _component_leaf(plan, θ)
    # The band mean is removed at θ's precision, which may exceed the track's:
    # the gauge is a property of θ.
    T = eltype(leaf)
    for a in axes(la, AntennaName), f in axes(la, Feed)
        node = _feed_node(plan.tying, f)
        node == 0 && continue
        track = view(la, AntennaName(a), Feed(f))
        n = count(isfinite, track)
        n == 0 && continue
        m = sum(v -> isfinite(v) ? T(v) : zero(T), track) / n
        L = _write_level!(_station_level(level, a), f, ts, Lj -> T(Lj) - m)
        for fs in axes(track, Frequency)
            v = T(track[fs])
            isfinite(v) || continue
            # Leave implausibly-large corrections unapplied (|g| = 1). A prior that
            # interpolates gaps self-regularizes, but an unconstrained fit can hand a
            # low-SNR band-edge segment that barely clears the gate a huge log-amp;
            # applying it would up-weight that segment's noise, since
            # applying a gain scales weights by |g|². The bound is generous
            # (|g| ≤ 10) so real passband roll-off/structure passes unchanged — only
            # pathological noise blow-ups are gated.
            gated = abs(v - m) > max_logamp
            leaf[1, node, fs, ts, a] = if isnothing(L)
                gated ? zero(v) : v - m
            else
                Lv = T(L[level.seg[fs]])
                gated ? m - Lv : v - Lv
            end
        end
    end
    return θ
end

# Free per-segment closure seed for the log-amp bandpass: the sum closure
# `log|V̄_ab| = la_a + la_b` solved independently in each frequency segment, plus
# each (station, feed, segment)'s summed gate weight as its precision, both over
# `(AntennaName(stations), Feed, segments)` in the sums' real type. Segments with no
# gated observation are left `NaN` for a prior to estimate — or not.
function _seed_amp_tracks(
        rbar_bp, wbar_bp, stations, fsegs, segments::Frequency;
        snr_floor::Real = 1.0, ridge::Real = 1.0e-6,
    )
    T = real(eltype(rbar_bp))
    nodes = _cell_nodes(rbar_bp, stations, PerFeed())
    val = zeros(T, dims(nodes))
    wt = zeros(T, dims(nodes))
    mask = fill!(similar(nodes, Bool), false)
    tracks = (_station_dim(stations), Feed(1:maximum(maximum, lookup(rbar_bp, FeedPair))), segments)
    la = fill!(zeros(T, tracks), T(NaN))
    prec = zeros(T, tracks)
    for (fs, chans) in enumerate(fsegs)
        fill!(mask, false)
        for bi in axes(nodes, AntennaPair), p in axes(nodes, FeedPair)
            c = (AntennaPair(bi), FeedPair(p))
            _solvable(nodes[c...]) || continue
            (a, fa), (b, fb) = nodes[c...]
            r, w = _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            snr2 = abs2(r) / w
            snr2 >= snr_floor^2 || continue
            amp = abs(r / w); amp > 0 || continue
            val[c...] = log(amp)
            wt[c...] = snr2
            mask[c...] = true
            prec[a, fa, fs] += snr2
            prec[b, fb, fs] += snr2
        end
        _solve_log_amp!(view(la, Frequency(fs)), val, wt, mask, nodes, ridge)
    end
    return la, prec
end

# Solve the sum closure over the masked cells of `val` into `la` `(nant, nfeed)`,
# leaving nodes no cell touches untouched. A component whose graph is bipartite
# determines its nodes only up to an alternating offset; `ridge` picks the
# smallest solution.
function _solve_log_amp!(la, val, w, mask, nodes, ridge::Real)
    T = eltype(val)
    nant = size(la, 1)
    cells = [I for I in eachindex(val, w, mask, nodes) if mask[I]]
    isempty(cells) && return la
    edges = [_edge(nodes[I], nant) for I in cells]
    touched = sort!(unique!([n for e in edges for n in e]))
    column = zeros(Int, length(la))
    column[touched] .= eachindex(touched)
    ncell = length(cells)
    A = zeros(T, ncell + length(touched), length(touched))
    for (i, (u, v)) in enumerate(edges)
        A[i, column[u]] += one(T)
        A[i, column[v]] += one(T)
    end
    for j in eachindex(touched)
        A[ncell + j, j] = one(T)
    end
    b = [T[val[I] for I in cells]; zeros(T, length(touched))]
    wts = [T[w[I] for I in cells]; fill(T(ridge), length(touched))]
    x = FactoredWLS(A, wts)(b)
    for (j, n) in pairs(touched)
        la[(n - 1) % nant + 1, (n - 1) ÷ nant + 1] = x[j]
    end
    return la
end

# ── Per-track bandpass smoothing (seed closure solve → per-track prior fit) ────

const _BP_SPIKE_SIGMA = 5.0
const _BP_MAX_LOGAMP = log(10.0)

# Outcome of fitting one piece of a (station, feed) bandpass track, reported by
# `bandpass_track_report` so a caller can tell a measurement from a placeholder.
# `θ` carries no such distinction: an unfitted track reads back as unit gain and a
# starved one as a constant, both indistinguishable from a real flat response.
const _BP_TRACK_NODATA = Int8(0)      # no usable channel; left at unit gain
const _BP_TRACK_SOLVED = Int8(1)      # fit, with frequency structure
const _BP_TRACK_FLAT = Int8(2)        # fit, but constant to within `_BP_FLAT_SPAN`
const _BP_TRACK_DECLINED = Int8(3)    # phase branch undetermined; not fit

const _BP_TRACK_LABELS = ("nodata", "solved", "flat", "declined")

# A fitted track this flat carries no shape: reported as `_BP_TRACK_FLAT` rather
# than silently passed off as a measured response. In radians for a phase track and
# nepers for a log-amplitude one — both are the observable's own natural unit and a
# hundredth of it is far below any real passband feature.
const _BP_FLAT_SPAN = 0.01

# Above this fraction of coin-flip steps (`phase_unwrap_ambiguity`) a phase track's
# 2π branch is not determined by the data, and the smooth trend a prior then
# fits through the unwrap's random walk is an artifact of the walk. Such a track is
# declined rather than fit: unit gain is honest about knowing nothing, an invented
# multi-radian ramp is not.
const _BP_MAX_UNWRAP_AMBIGUITY = 0.25

# The largest log-gain step a linearized joint bandpass update takes; the
# first-order model of `|g·e^δ − ĝ|²` is poor much beyond it.
const _BP_MAX_LINEAR_STEP = 0.5

# Fraction of a solve's tracks that may come back flat or declined before the
# bandpass as a whole is worth a warning.
const _BP_DEGENERATE_WARN_FRACTION = 0.25

"""
    PerTrackSmoother(; eltype = nothing)

Solve the bandpass one (station, feed) track at a time: the per-segment closure
graph solve seeds real phase and log-amp tracks, then each track is fit under
its component's prior, one spectral window at a time, with the level of a level
component estimated alongside.

Phase tracks are unwrapped along frequency within each spectral window before
the fit, so a prior sees a continuous branch rather than a sawtooth. A prior
fills segments the closure solve had no data for; without one they stay
unapplied.

The scans of a time segment are pooled before the solve, each weighted by the
conjugate of its own band-averaged visibility per (baseline, feed pair), so
every scan contributes in proportion to its signal power. The pooled sums are held
in `eltype`, a real floating-point type; `nothing` keeps the data's.

The closure assumes a baseline's source term cancels out of the per-segment
phase-difference / log-amp-sum, so it is not appropriate for a resolved or
polarized calibrator (see [`JointSmoother`](@ref)).
"""
struct PerTrackSmoother{E} <: AbstractBandpassSmoother
    eltype::E
    function PerTrackSmoother(eltype)
        isnothing(eltype) || eltype isa Type{<:AbstractFloat} || throw(
            ArgumentError("`eltype` must be a real floating-point type or `nothing`, got $eltype"),
        )
        return new{typeof(eltype)}(eltype)
    end
end
PerTrackSmoother(; eltype = nothing) = PerTrackSmoother(eltype)

can_fit(::PerTrackSmoother, tc, geom) = _fits_bandpass_track(tc)

# The outcome code for one fitted piece: nothing estimated, a constant, or a
# real shape.
function _band_track_status(fitted)
    obs = [v for v in fitted if isfinite(v)]
    isempty(obs) && return _BP_TRACK_NODATA
    return (maximum(obs) - minimum(obs)) < _BP_FLAT_SPAN ? _BP_TRACK_FLAT : _BP_TRACK_SOLVED
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

# Fit one (station, feed, time segment) track in place under `prior`, one piece
# per spectral window: segment values `track`, weights `w`, segment frequencies
# `x`, and `pieces` the segment groups. The prior's hyperparameters are resolved
# once over all of the track's pieces. With `level` (each piece's level segment,
# of `nlevel`), the track is a level per level segment plus a zero-mean shape
# under the prior, and `track` receives their sum. Returns the resolved prior —
# `prior` itself when the track carries no data — and the levels, `nothing`
# without `level`.
#
# `unwrap` re-references each piece of a phase track to a continuous branch along
# frequency first, because the prior fits a real track and the ±π branch cuts of
# a raw phase solve would otherwise read as genuine structure; the pieces sharing
# a level are then put on one branch. Under a prior, a piece whose branch the
# data do not determine (`phase_unwrap_ambiguity` past `_BP_MAX_UNWRAP_AMBIGUITY`)
# is dropped instead; see that constant. Without one the segments are fit
# independently and the branch does not affect the fit.
#
# `status` receives one `_BP_TRACK_*` code per piece.
#
# `weights`, when given, receives each segment's weight in the final fit, zero
# where it carried none.
function _fit_track!(
        track, w, x, pieces, prior;
        level = nothing, nlevel::Integer = 0, unwrap::Bool = false, status = nothing,
        weights = nothing,
    )
    T = eltype(track)
    isnothing(weights) || fill!(weights, zero(T))
    ys = [T[track[c] for c in p] for p in pieces]
    ws = [T[w[c] for c in p] for p in pieces]
    xs = [float.(x[p]) for p in pieces]
    declined = falses(length(pieces))
    if unwrap
        for j in eachindex(ys, ws)
            if !isnothing(prior) && phase_unwrap_ambiguity(ys[j]; weights = ws[j]) > _BP_MAX_UNWRAP_AMBIGUITY
                declined[j] = true
                fill!(ys[j], T(NaN))
            else
                ys[j] = unwrap_phase_track(ys[j]; weights = ws[j])
            end
        end
        isnothing(level) || _align_branches!(ys, ws, level)
    end
    fill!(track, T(NaN))
    levels = isnothing(level) ? nothing : fill(T(NaN), nlevel)
    if !any(j -> any(k -> _shape_usable(ys[j][k], ws[j][k]), eachindex(ys[j], ws[j])), eachindex(ys, ws))
        isnothing(status) || (status .= ifelse.(declined, _BP_TRACK_DECLINED, _BP_TRACK_NODATA))
        return prior, levels
    end
    resolved = _estimate_hypers(prior, ys, ws, xs; level)
    if !isnothing(weights)
        for (j, p) in pairs(pieces), (i, c) in pairs(p)
            _shape_usable(ys[j][i], ws[j][i]) && (weights[c] = ws[j][i])
        end
    end
    isnothing(level) || (levels .= _estimate_levels(resolved, ys, ws, xs, level, nlevel))
    for (j, p) in pairs(pieces)
        L = isnothing(level) ? zero(T) : levels[level[j]]
        fitted = view(track, p)
        ys[j] .-= L
        _estimate_map!(fitted, resolved, ys[j], ws[j], xs[j])
        fitted .+= L
        isnothing(status) || (status[j] = declined[j] ? _BP_TRACK_DECLINED : _band_track_status(fitted))
    end
    return resolved, levels
end

# Shift each unwrapped phase piece by the multiple of 2π that brings its weighted
# mean nearest that of the first piece with data in its level group, so a level
# shared across spectral windows sees one branch.
function _align_branches!(ys, ws, level)
    ref = Dict{Int, eltype(eltype(ys))}()
    for j in eachindex(ys, ws, level)
        any(k -> _shape_usable(ys[j][k], ws[j][k]), eachindex(ys[j], ws[j])) || continue
        m = _weighted_mean_finite(ys[j], ws[j])
        r = get!(ref, level[j], m)
        ys[j] .-= 2π * round((m - r) / 2π)
    end
    return ys
end

# Fit every (station, feed) track of `tracks` in place (see `_fit_track!`), each
# segment weighted by its seed precision. `tracks` and `prec` are over
# `(AntennaName, Feed, Frequency)`, and `station_priors[a]` is station `a`'s resolved
# prior. `status`, when given, is over `(AntennaName, Feed, Frequency)` with one entry per
# piece; `priors` over `(AntennaName, Feed)`, receiving each track's resolved prior; and
# `levels`, with `level`, over `(AntennaName, Feed, level segment)`.
function _fit_tracks!(
        tracks, prec, x, pieces, station_priors;
        unwrap::Bool, level = nothing, levels = nothing, status = nothing, priors = nothing,
    )
    nlevel = isnothing(levels) ? 0 : size(levels, 3)
    for a in axes(tracks, AntennaName), f in axes(tracks, Feed)
        st = isnothing(status) ? nothing : view(status, AntennaName(a), Feed(f))
        resolved, L = _fit_track!(
            view(tracks, AntennaName(a), Feed(f)), view(prec, AntennaName(a), Feed(f)), x, pieces,
            _prior_along(station_priors[a], :Frequency);
            level, nlevel, unwrap, status = st,
        )
        isnothing(priors) || (priors[a, f] = resolved)
        isnothing(L) || (levels[a, f, :] .= L)
    end
    return tracks
end

# One observable's per-piece outcome array over `(AntennaName, Feed, Frequency, Ti)`:
# `stations`, every feed, each spectral window of `seg` over the extent of its
# channels, and each of `plan`'s `nts` time segments over the extent of its
# samples.
function _track_status_array(stations, geom::DataGeometry, plan, seg, nts::Integer)
    ax = (
        _station_dim(stations), Feed(1:geom.nfeed),
        Frequency(_segment_lookup(geom.channel_freqs, seg.piece_chans)),
        Ti(_time_segment_lookup(plan, geom, nts)),
    )
    return fill!(zeros(Int8, ax), _BP_TRACK_NODATA)
end

# One observable's resolved prior per (station, feed, time segment), over
# `(AntennaName, Feed, Ti)` labeled as `_track_status_array`; each slot starts at its
# station's own prior.
function _track_prior_array(stations, geom::DataGeometry, plan, nts::Integer)
    ax = (_station_dim(stations), Feed(1:geom.nfeed), Ti(_time_segment_lookup(plan, geom, nts)))
    arr = DimArray(Array{Union{Nothing, AbstractPrior}}(undef, map(length, ax)), ax)
    for a in axes(arr, 1)
        arr[a, :, :] .= Ref(_prior_along(plan.priors[a], :Frequency))
    end
    return arr
end

# A station block's outcome and prior arrays: its own stations and time segments.
_block_status_array(block, geom::DataGeometry) = _track_status_array(
    geom.stations[block.stations], geom, block.plan, _shape_segments(block.plan, geom), block.plan.shape[4],
)
_block_prior_array(block, geom::DataGeometry) =
    _track_prior_array(geom.stations[block.stations], geom, block.plan, block.plan.shape[4])

# One entry per station block, keyed `g1, g2, …` as a `CalibrationSolution` keys a
# station-heterogeneous component's leaves; a single block is its own array.
_block_leaves(arrays) = isempty(arrays) ? nothing :
    length(arrays) == 1 ? only(arrays) :
    NamedTuple{Calibration._group_keys(length(arrays))}(Tuple(arrays))

_status_arrays(::Nothing) = ()
_status_arrays(st::NamedTuple) = values(st)
_status_arrays(st) = (st,)

"""
    bandpass_track_report(phase_status, amp_status; phase_priors = nothing,
                          amp_priors = nothing) -> NamedTuple

Summarize a bandpass solve's per-piece outcomes into the record the
[`Bandpass`](@ref Gustavo.Bandpass) step publishes. `phase_status`/`amp_status`
hold one `_BP_TRACK_*` code per (station, feed, frequency segment, time segment)
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
alongside `track_labels` (the code → name mapping, so a reader needs no constant
from this module) and the counts `n_solved`/`n_flat`/`n_declined`/`n_nodata`
summed over both observables. `flat` and `declined` are the two ways a piece
can occupy a slot without measuring anything, and they are what the counts
exist to expose: θ itself records an unfitted piece as unit gain and a starved
one as a constant, neither distinguishable there from a genuinely flat response.
"""
function bandpass_track_report(phase_status, amp_status; phase_priors = nothing, amp_priors = nothing)
    counts = zeros(Int, length(_BP_TRACK_LABELS))
    for st in (phase_status, amp_status), a in _status_arrays(st), c in a
        counts[Int(c) + 1] += 1
    end
    # Concrete arrays throughout — the record is serialized with the solution, and
    # an observable that was not fit is an empty status rather than a missing field.
    empty_status = Array{Int8, 4}(undef, 0, 0, 0, 0)
    empty_priors = Array{Union{Nothing, AbstractPrior}, 3}(undef, 0, 0, 0)
    return (;
        phase_status = something(phase_status, empty_status),
        amp_status = something(amp_status, empty_status),
        phase_priors = something(phase_priors, empty_priors),
        amp_priors = something(amp_priors, empty_priors),
        track_labels = collect(String, _BP_TRACK_LABELS),
        n_nodata = counts[1], n_solved = counts[2],
        n_flat = counts[3], n_declined = counts[4],
    )
end

# Warn when a large share of the tracks measured nothing. Silence here would leave
# a bandpass that is mostly placeholder looking exactly like one that is mostly
# measured — the caller cannot tell from θ, which is why this is a warning and not
# only a record.
function _warn_degenerate_bandpass(report)
    total = report.n_nodata + report.n_solved + report.n_flat + report.n_declined
    total > 0 || return nothing
    degenerate = report.n_flat + report.n_declined + report.n_nodata
    frac = degenerate / total
    frac > _BP_DEGENERATE_WARN_FRACTION || return nothing
    @warn """
    Bandpass: $(round(100 * frac; digits = 1))% of (station, feed, frequency segment) pieces \
    carry no measured frequency shape — $(report.n_flat) fit flat, $(report.n_declined) \
    declined for an undetermined phase branch, $(report.n_nodata) with no usable data \
    (of $total). They are unit gain or a constant in the solution, not a measured \
    response. Check per-segment SNR and the detection coverage of the bandpass scans.
    """ n_solved = report.n_solved
    return nothing
end

"""
    time_segment_scans(plan, results) -> Vector{Vector{Int}}

The `results` indices belonging to each of `plan`'s time segments, in segment
order. A scan lies wholly inside one segment of any segmentation the smoothers
accept, so its first sample (`res.ti`) names the segment. Segments the pass
never visited come back empty.
"""
function time_segment_scans(plan, results)
    nts = isempty(plan.tseg_id) ? 1 : maximum(plan.tseg_id)
    groups = [Int[] for _ in 1:nts]
    for (i, res) in pairs(results)
        push!(groups[plan.tseg_id[res.ti]], i)
    end
    return groups
end

"""
    _station_time_segments(blocks, results, nant) -> Matrix{Int}

Each station's own time segment for each scan: `tseg[ant, i]` is the segment
`results[i]` falls in under the segmentation `blocks` gives that station, and `0`
for a station no block covers — such a station carries no bandpass parameter and
stays at unit gain. A scan lies wholly inside one segment of any segmentation the
smoothers accept, so its first sample (`res.ti`) names the segment.

A station-uniform model has one block spanning every station, so every row is the
same and the table reduces to [`time_segment_scans`](@ref)' single grouping.
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

# Sum the per-scan residual accumulators of `idx` into one pooled pair. A scan's
# source term on a baseline is its own, so each scan's sums are multiplied by
# `c = conj(Ŝ)/max|Ŝ|`, with `Ŝ` its band-averaged visibility per (baseline,
# feed pair) and the maximum over the pooled scans, and its weights by `|c|²`.
# Unaligned scans would partly cancel; weighting by `|Ŝ|` makes each cell's
# pooled `|r|²/w` the sum of the scans' own, so a weak scan adds its information
# instead of diluting a strong one; and the per-row normalization leaves the
# pooled amplitude `|g_a·g_b|` times a source amplitude, the band-averaged gains
# cancelling, so the amplitude closure is not coupled across stations through
# them. The factor is flat in frequency, so the bandpass shape and the phase
# steps between spectral windows are kept.
function _pool_scans(results, idx, T)
    rbar = zeros(complex(T), dims(results[first(idx)].rl))
    wbar = zeros(T, dims(results[first(idx)].wl))
    Ŝs = [_band_average(results[i].rl, results[i].wl) for i in idx]
    peak = map((xs...) -> maximum(x -> isfinite(x) ? abs(x) : zero(real(x)), xs), Ŝs...)
    for (i, Ŝ) in zip(idx, Ŝs)
        (; rl, wl) = results[i]
        c = map((s, m) -> m > 0 ? conj(s) / m : zero(s), Ŝ, peak)
        rbar .+= DimensionalData.broadcast_dims(*, rl, c)
        wbar .+= DimensionalData.broadcast_dims(*, wl, abs2.(c))
    end
    return rbar, wbar
end

_band_average(rl, wl) = map(
    (r, w) -> w > 0 ? r / w : zero(r),
    dropdims(sum(rl; dims = Frequency); dims = Frequency),
    dropdims(sum(wl; dims = Frequency); dims = Frequency),
)

function solve_bandpass!(sm::PerTrackSmoother, θ, results, setup; gauge::AbstractGauge)
    T = something(sm.eltype, mapreduce(r -> eltype(r.wl), promote_type, results))
    # The two observables are solved independently here, so each partitions the
    # scans by its own time segmentation — a phase bandpass that breaks mid-track
    # can sit beside an amplitude one held over the whole of it.
    phase = _per_track_observable!(θ, results, setup, :phase, T; gauge)
    amp = _per_track_observable!(θ, results, setup, :logamp, T; gauge)
    report = bandpass_track_report(
        phase.status, amp.status; phase_priors = phase.priors, amp_priors = amp.priors,
    )
    _warn_degenerate_bandpass(report)
    return report
end

# Seed, fit and write one observable of a `PerTrackSmoother` solve, returning its
# outcome and prior arrays (`nothing` when the model has no such component).
function _per_track_observable!(θ, results, setup, group::Symbol, T; gauge)
    blocks = bandpass_blocks(setup, θ, group)
    isempty(blocks) && return (; status = nothing, priors = nothing)
    geom = setup.geom
    plan = only(blocks).plan
    level_blocks = bandpass_level_blocks(setup, θ, group)
    level_plan = isempty(level_blocks) ? nothing : only(level_blocks).plan
    seg = _shape_segments(plan, geom)
    level, nlevel = isnothing(level_plan) ? (nothing, 0) : _piece_levels(level_plan, seg)
    segments = Frequency(_frequency_segment_lookup(plan, geom))
    groups = time_segment_scans(plan, results)
    status = _track_status_array(geom.stations, geom, plan, seg, length(groups))
    priors = _track_prior_array(geom.stations, geom, plan, length(groups))
    for (ts, idx) in pairs(groups)
        isempty(idx) && continue
        rbar, wbar = _pool_scans(results, idx, T)
        tracks, prec = group === :phase ?
            _seed_phase_tracks(rbar, wbar, geom.stations, seg.fsegs, segments; gauge, component = plan.path) :
            _seed_amp_tracks(rbar, wbar, geom.stations, seg.fsegs, segments)
        group === :logamp && _spike_guard!(tracks, seg.pieces, _BP_SPIKE_SIGMA)
        levels = isnothing(level) ? nothing : fill(eltype(tracks)(NaN), length(geom.stations), geom.nfeed, nlevel)
        _fit_tracks!(
            tracks, prec, seg.x, seg.pieces, plan.priors;
            unwrap = group === :phase, level, levels,
            status = view(status, Ti(ts)), priors = view(priors, Ti(ts)),
        )
        writer = _level_writer(level_plan, θ, isnothing(level) ? nothing : _segment_levels(level, seg), levels)
        group === :phase ?
            _write_phase_bandpass!(θ, plan, tracks, ts; level = writer) :
            _write_amp_bandpass!(θ, plan, tracks, _BP_MAX_LOGAMP, ts; level = writer)
    end
    return (; status, priors)
end

# ── Joint complex bandpass + per-scan source coherence (ALS) ─────────────────
#
# The per-segment closure solves assume a baseline's source term cancels out of
# the per-segment phase-difference/log-amp-sum closure, which holds only for an
# unresolved, unpolarized source. `solve_joint_bandpass!` instead fits the
# complex visibilities against
#
#     V_ab(ν) | scan  ≈  g_a(ν) · S_{scan,ab,pol} · conj(g_b(ν)),
#
# one frequency-flat complex `S` per (scan, baseline, polarization product), so
# a resolved or polarized source's per-baseline structure is absorbed into `S`
# rather than biasing the station bandpass. `g` is bilinear with `S`, so this
# alternates a closed-form per-(scan, baseline, pol) solve of `S` given the
# current `g` ([`_update_source_coherence!`](@ref)) with a Gauss-Seidel
# per-(station, feed) solve of `g` given `S` and every other station's current
# gain ([`_update_station_gains!`](@ref)), one gain per frequency segment, to
# convergence.
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
# costs variance, never validity. The closure tier's gate is justified there
# because that tier extracts a per-segment phase, which is meaningless below the
# noise. Applied here it would preferentially delete the weak cells pairing
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
# axis — not summed across scans (unlike the closure tier's fold), since the per-scan source coherence needs each
# scan's own coherent visibility. Element types follow the scan accumulators'
# own, not a hardcoded precision.
function _reduce_all_scans(scans, segs, cells::Frequency)
    nscan = length(scans)
    rl = first(scans).rl
    nbl, npol = size(rl, AntennaPair), size(rl, FeedPair)
    nseg = length(segs)
    C = eltype(rl)
    T = real(eltype(first(scans).wl))
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

# One phase block's solve state over `(AntennaName, Feed, Frequency, Ti)` — its
# stations, every feed, its frequency segments and its time segments: the
# complex gains `g`; their unwrapped phase tracks `φ`, carried across sweeps so
# the prior fit never sees a 2π branch cut; the slots a sweep has solved
# (`touched`) and the gauge pins (`pinned`).
# The blocks a joint solve holds its complex gains on: the phase blocks, or the
# amplitude blocks when the model has no phase. Where both exist they match.
_gain_blocks(phase_blocks, amp_blocks) = isempty(phase_blocks) ? amp_blocks : phase_blocks

# A station's track prior, `nothing` for a track without one or an observable the
# model holds at zero; and whether either of its tracks has one.
_prior(track) = isnothing(track) ? nothing : track.prior
_has_prior(fit) = !isnothing(_prior(fit.phase)) || !isnothing(_prior(fit.amp))

function _block_gains(block, geom::DataGeometry, C::Type)
    ax = (
        _station_dim(geom.stations[block.stations]), Feed(1:geom.nfeed),
        Frequency(_frequency_segment_lookup(block.plan, geom)),
        Ti(_time_segment_lookup(block.plan, geom)),
    )
    return DimStack((;
        g = ones(C, ax), φ = zeros(real(C), ax),
        touched = zeros(Bool, ax), pinned = zeros(Bool, ax),
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

# Every (station pair, feed pair) cell touching (station, feed): the cell, the
# station and feed at its other end, and whether (station, feed) is the cell's
# first end — built once, reused by every ALS iteration's gain update.
function _joint_bandpass_touching(cells, nant, nfeed)
    touching = [Tuple{Int, Int, Int, Int, Bool}[] for _ in 1:nant, _ in 1:nfeed]
    for p in axes(cells, FeedPair), bi in axes(cells, AntennaPair)
        _solvable(cells[bi, p]) || continue
        (a, fa), (b, fb) = cells[bi, p]
        push!(touching[a, fa], (bi, p, b, fb, true))
        push!(touching[b, fb], (bi, p, a, fa, false))
    end
    return touching
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
`nothing`. Any other gauge is imposed in every iteration's joint step
(`_joint_gain_step!`), and this returns the constraints as `(; rows, d)`,
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
`_move_band_level!` moves those levels into `S`. Pinning `|g|` as well
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

# The fit of one track system to `v ./ sys.w` (no observation where the
# weight is zero) under its resolved prior, levels and pieces.
function _track_response(sys, v)
    T = eltype(v)
    y = T[w > 0 ? vi / w : T(NaN) for (vi, w) in zip(v, sys.w)]
    _fit_track!(y, sys.w, sys.x, sys.pieces, sys.prior; sys.level, sys.nlevel)
    return y
end

# Closed-form per-(scan, station pair, feed pair) solve of the source coherence
# `S` given the current station gains: the weighted-least-squares minimizer of
# `Σ_cell w·|r/w − g_a·S·conj(g_b)|²` over the single complex unknown `S`.
#
# `r`/`w` are reduced onto the refinement grid the two stations have in common,
# while the gains are held in each station's own frequency segments, so each end
# is read through its own `fseg` row.
function _update_source_coherence!(data, gains, layout)
    (; r, w, S, ends) = data
    (; loc, tseg, fseg) = layout
    T = real(eltype(S))
    for p in axes(r, FeedPair), bi in axes(r, AntennaPair)
        _solvable(ends[bi, p]) || continue
        (a, fa), (b, fb) = ends[bi, p]
        (ka, ia), (kb, ib) = loc[a], loc[b]
        for si in axes(r, Scan)
            # Each station is in its own time segment for this scan; a station no
            # block covers (segment 0) has no gain to fit, so the pairs that
            # touch it carry no source term either.
            ta, tb = tseg[a, si], tseg[b, si]
            (iszero(ta) || iszero(tb)) && continue
            ga, gb = gains[ka].g, gains[kb].g
            numer = zero(eltype(S))
            denom = zero(T)
            for cell in axes(r, Frequency)
                wc = w[Scan(si), AntennaPair(bi), FeedPair(p), Frequency(cell)]
                wc > 0 || continue
                u = ga[ia, fa, fseg[a, cell], ta] * conj(gb[ib, fb, fseg[b, cell], tb])
                abs2(u) > 0 || continue
                numer += conj(u) * r[Scan(si), AntennaPair(bi), FeedPair(p), Frequency(cell)]
                denom += wc * abs2(u)
            end
            S[Scan(si), AntennaPair(bi), FeedPair(p)] = denom > 0 ? numer / denom : zero(eltype(S))
        end
    end
    return nothing
end

# One Gauss-Seidel sweep over every (station, feed, time segment): closed-form
# per-segment solve of its complex gain given the current source coherence `S`
# and every other station's current gain, which is immediately visible to later
# antennas in the same sweep — Gauss-Seidel, not Jacobi — with each observable's
# component prior acting on the resulting track rather than as a post-hoc smooth.
#
# Solving each segment independently is the no-prior case; the prior enters
# where that independence is dropped. A segment's residual is `denom·|g − ĝ|²`
# about its unconstrained estimate `ĝ`, and the real tracks are fit to it to
# first order in the log-gain. Without a prior the track lands on `ĝ`, so the
# first sweep, and every sweep of a track without a prior, linearizes about
# `ĝ` itself (weight `denom·|ĝ|²`); under a prior the track does not, so with
# `relinearize` the residual is linearized about the current gain `g`
# (observation `log g + (ĝ/g − 1)`, weight `denom·|g|²`), whose fixed point is
# the MAP of the complex residual rather than of its expansion about `ĝ`. Each
# sweep re-estimates the priors' hyperparameters, and any levels, from its own
# linearized tracks, so iterated to convergence this is MAP estimation under the
# two priors with type-II MAP hyperparameters.
#
# `φ` carries each (station, feed)'s unwrapped phase track across sweeps. It is
# unwrapped once, when `seed` (the first sweep) initializes it, and thereafter
# advanced by wrapped increments about its own current value, so no global
# unwrap is needed inside the loop and the 2π branch cannot flip between
# iterations. `fits[a]` holds station `a`'s phase and amplitude
# `(; x, pieces, prior, level, nlevel)`, `nothing` for an observable held at
# zero, and `levels[a]` its fitted levels, each
# over `(Feed, level segment, Ti)` or `nothing`. The status and prior arrays,
# when given, hold one array per block.
#
# `systems`, when given, receives every phase and log-amplitude track's fit,
# for `_joint_gain_step!`.
#
# Returns the number of slots the sweep updated and the `(station, feed, time
# segment)` of each phase track with a piece declined at unwrapping.
function _update_station_gains!(
        gains, data, layout; fits, levels, seed::Bool, relinearize::Bool = false, systems = nothing,
        phase_status = nothing, amp_status = nothing, phase_priors = nothing, amp_priors = nothing,
    )
    (; r, w, S) = data
    (; loc, touching, tseg, fseg, present) = layout
    C = eltype(first(gains).g)
    T = real(C)
    nfsmax = maximum(st -> size(st, Frequency), gains)
    num = Vector{C}(undef, nfsmax)
    den = Vector{T}(undef, nfsmax)
    ĝ = Vector{C}(undef, nfsmax)
    wf = Vector{T}(undef, nfsmax)
    la = Vector{T}(undef, nfsmax)
    φ̃ = Vector{T}(undef, nfsmax)
    wg = Vector{T}(undef, nfsmax)
    wa = Vector{T}(undef, nfsmax)
    isnothing(systems) || (empty!(systems.phase); empty!(systems.amp))
    nupdated = 0
    declined = Tuple{Int, Int, Int}[]
    for feed in axes(touching, 2), ant in eachindex(loc)
        k, ai = loc[ant]
        iszero(k) && continue
        entries = touching[ant, feed]
        isempty(entries) && continue
        (; g, φ, touched, pinned) = gains[k]
        nfs = size(g, Frequency)
        fit = fits[ant]
        for ts in present[ant]
            # The gauge fixes this node's phase in the frequency segments it pins —
            # every one of them where the array leaves this track no free structure,
            # a single one where a coarser station ties the band into one mode. The
            # amplitude is solved like any other node's (see `_joint_bandpass_gauge!`).
            pins = view(pinned, ai, feed, :, ts)
            for buf in (num, den, ĝ, wf, la, φ̃, wg, wa)
                resize!(buf, nfs)
            end
            fill!(num, zero(C))
            fill!(den, zero(T))
            for cell in axes(r, Frequency)
                sa = fseg[ant, cell]
                for (bi, p, o, fo, first_end) in entries
                    ko, io = loc[o]
                    # A station no block covers has no gain, so its cells constrain nothing.
                    iszero(ko) && continue
                    go = gains[ko].g
                    so = fseg[o, cell]
                    for si in axes(r, Scan)
                        # Only the scans this node's own segment covers constrain it.
                        tseg[ant, si] == ts || continue
                        wc = w[Scan(si), AntennaPair(bi), FeedPair(p), Frequency(cell)]
                        wc > 0 || continue
                        to = tseg[o, si]
                        iszero(to) && continue
                        s = S[Scan(si), AntennaPair(bi), FeedPair(p)]
                        # `V ≈ g_first·S·conj(g_second)`; the second end fits `conj(V)`.
                        coeff = first_end ? s * conj(go[io, fo, so, to]) : conj(go[io, fo, so, to] * s)
                        abs2(coeff) > 0 || continue
                        rc = r[Scan(si), AntennaPair(bi), FeedPair(p), Frequency(cell)]
                        num[sa] += conj(coeff) * (first_end ? rc : conj(rc))
                        den[sa] += wc * abs2(coeff)
                    end
                end
            end
            linear = relinearize && _has_prior(fit)
            for fs in 1:nfs
                gh = den[fs] > 0 ? num[fs] / den[fs] : zero(C)
                ĝ[fs] = gh
                gc = g[ai, feed, fs, ts]
                wf[fs] = den[fs] > 0 && abs(linear ? gc : gh) > 0 ? den[fs] * abs2(linear ? gc : gh) : zero(T)
                if wf[fs] > 0 && linear
                    # `den·|g − ĝ|²` to first order in the log-gain about the current `g`.
                    # Far from `ĝ` the weight is raised and the step shortened by the
                    # same factor, which leaves the gradient, and so the fixed point,
                    # unchanged.
                    ρ = gh / gc - 1
                    κ = max(one(T), abs(ρ) / _BP_MAX_LINEAR_STEP)
                    wf[fs] *= κ
                    la[fs] = log(abs(gc)) + real(ρ) / κ
                    φ̃[fs] = φ[ai, feed, fs, ts] + imag(ρ) / κ
                elseif wf[fs] > 0
                    # With the phase held at zero the least-squares amplitude is `Re ĝ`.
                    la[fs] = isnothing(fit.phase) ? (real(gh) > 0 ? log(real(gh)) : T(NaN)) : log(abs(gh))
                    # The wrapped increment about this track's current value keeps the
                    # candidate on the same 2π branch as the iterate it refines.
                    φ̃[fs] = φ[ai, feed, fs, ts] +
                        rem2pi(angle(gh) - φ[ai, feed, fs, ts], RoundNearest)
                else
                    la[fs] = T(NaN)
                    φ̃[fs] = T(NaN)
                end
            end
            # The status of the last sweep is the status of the solve: each sweep
            # overwrites the previous one's codes for this node.
            if isnothing(fit.amp)
                replace!(v -> isfinite(v) ? zero(T) : v, la)
            else
                ast = isnothing(amp_status) ? nothing : view(amp_status[k], ai, feed, :, ts)
                amp_prior, amp_levels = _fit_track!(
                    la, wf, fit.amp.x, fit.amp.pieces, fit.amp.prior;
                    fit.amp.level, fit.amp.nlevel, status = ast, weights = wa,
                )
                isnothing(systems) || push!(
                    systems.amp,
                    (; slots = systems.ids[k][ai, feed, 1:nfs, ts], w = wa[1:nfs], fit.amp.x, fit.amp.pieces, prior = amp_prior, fit.amp.level, fit.amp.nlevel),
                )
                isnothing(amp_priors) || (amp_priors[k][ai, feed, ts] = amp_prior)
                isnothing(amp_levels) || (levels[ant].amp[feed, :, ts] .= amp_levels)
            end
            pst = isnothing(fit.phase) ? nothing :
                isnothing(phase_status) ? similar(fit.phase.pieces, Int8) : view(phase_status[k], ai, feed, :, ts)
            if isnothing(fit.phase)
                replace!(v -> isfinite(v) ? zero(T) : v, φ̃)
            elseif all(pins)
                # A wholly pinned track is known rather than fitted: report it as such
                # instead of leaving it at NODATA.
                fill!(pst, _BP_TRACK_SOLVED)
                fill!(φ̃, zero(T))
                isnothing(levels[ant].phase) || (levels[ant].phase[feed, :, ts] .= zero(T))
            else
                phase_prior, phase_levels = _fit_track!(
                    φ̃, wf, fit.phase.x, fit.phase.pieces, fit.phase.prior;
                    fit.phase.level, fit.phase.nlevel, unwrap = seed, status = pst,
                    weights = wg,
                )
                isnothing(systems) || push!(
                    systems.phase,
                    (; slots = systems.ids[k][ai, feed, 1:nfs, ts], w = wg[1:nfs], fit.phase.x, fit.phase.pieces, prior = phase_prior, fit.phase.level, fit.phase.nlevel),
                )
                any(==(_BP_TRACK_DECLINED), pst) && push!(declined, (ant, feed, ts))
                isnothing(phase_priors) || (phase_priors[k][ai, feed, ts] = phase_prior)
                isnothing(phase_levels) || (levels[ant].phase[feed, :, ts] .= phase_levels)
                # A partial pin holds its own segments and leaves the rest fitted.
                # With segments fit independently that is the constrained fit itself;
                # `solve_joint_bandpass!` rejects a prior that relates them rather
                # than pass off a joint fit with one segment overwritten.
                for fs in 1:nfs
                    pins[fs] && (φ̃[fs] = zero(T))
                end
            end
            for fs in 1:nfs
                (isfinite(la[fs]) && isfinite(φ̃[fs])) || continue
                g[ai, feed, fs, ts] = exp(C(la[fs], φ̃[fs]))
                φ[ai, feed, fs, ts] = φ̃[fs]
                touched[ai, feed, fs, ts] = true
                nupdated += 1
            end
            _move_band_level!(gains[k], levels[ant], S, ai, feed, ts, entries, view(tseg, ant, :))
        end
    end
    return nupdated, declined
end

# Move one (station, feed, time segment) track's band-mean log-amplitude and
# phase into the source coherence `S` of every correlation it enters, which
# leaves every model visibility unchanged: the data leave that level free, and
# under a prior the sweep would otherwise drift along it. Moving it as soon as
# the track is fit keeps the gains and `S` the later tracks of the sweep are
# fit against consistent. `entries` are the track's `touching` entries and
# `tseg_a` its station's time segment per scan. A pinned track keeps its
# phase; levels move with their track.
function _move_band_level!(gk, lv, S, ai, f, ts, entries, tseg_a)
    (; g, φ, touched, pinned) = gk
    T = real(eltype(g))
    valid = view(touched, ai, f, :, ts)
    n = count(valid)
    iszero(n) && return gk
    mla = sum(log(abs(g[ai, f, fs, ts])) for fs in axes(g, Frequency) if valid[fs]) / n
    mφ = any(view(pinned, ai, f, :, ts)) ? zero(T) :
        sum(φ[ai, f, fs, ts] for fs in axes(g, Frequency) if valid[fs]) / n
    shift = exp(-complex(mla, mφ))
    for fs in axes(g, Frequency)
        valid[fs] || continue
        g[ai, f, fs, ts] *= shift
        φ[ai, f, fs, ts] -= mφ
    end
    isnothing(lv.amp) || (view(lv.amp, f, :, ts) .-= mla)
    isnothing(lv.phase) || (view(lv.phase, f, :, ts) .-= mφ)
    # `V ≈ g_first·S·conj(g_second)`.
    for (bi, p, _, _, first_end) in entries, si in axes(S, Scan)
        tseg_a[si] == ts || continue
        S[Scan(si), AntennaPair(bi), FeedPair(p)] *= exp(complex(mla, first_end ? mφ : -mφ))
    end
    return gk
end

# The fixed structure of the joint gain step (`_joint_gain_step!`): the slot
# numbering `ids` (`_gain_slot_ids`), the `(slot, slot)` pairs a correlation
# joins, `edge[si, bi, p, cell]` the pair of that correlation (`0` for none), and,
# for a gauge other than a pin, the slots of each constraint row's connected
# component.
function _joint_gain_layout(gains, data, layout, gauge_state)
    (; r, ends) = data
    (; loc, tseg, fseg) = layout
    ids = _gain_slot_ids(gains)
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
    row_slots = if isnothing(gauge_state)
        nothing
    else
        compid, ncomp, _ = _joint_bandpass_graph(data, layout, ids)
        members = [Int[] for _ in 1:ncomp]
        for n in eachindex(compid)
            iszero(compid[n]) || push!(members[compid[n]], n)
        end
        [isempty(row) ? Int[] : members[compid[first(first(row))]] for row in gauge_state.rows]
    end
    return (; ids, edge, slot_pairs, row_slots)
end

# One Gauss–Newton step on every gain slot at once, given `S`. To first order a
# correlation's model `m = g_a·S·conj(g_b)` moves to
# `m·(1 + δℓ_a + δℓ_b + i(δφ_a − δφ_b))` for log-amplitude `ℓ` and phase `φ`,
# so with `u = r/(w·m) − 1` and weight `q = w·|m|²` the step solves two weighted
# least-squares problems over the slots: `Im u` measures `δφ_a − δφ_b` and
# `Re u` measures `δℓ_a + δℓ_b`. Solving every slot together moves the
# directions a station-by-station sweep is slow along — those only weakly
# measured correlations constrain, such as the phase between feeds — as fast as
# any other. Without a prior the normal equations are solved directly and a χ²
# backtracking search keeps the step a descent step; under one, `systems` holds
# the sweep's per-track fits and the MAP step, under the gauge's constraints, is
# solved by `_joint_prior_solve`. Pinned slots keep their phase; any other gauge
# is restored after the step by shifting each component's phase along its free
# direction, which a full step under a prior leaves unchanged and which is
# exact without one. Returns each slot's information, the diagonal of the
# normal matrix: the inverse variance of its log-gain from the data alone.
function _joint_gain_step!(gains, data, jl, gauge_state, systems)
    (; r, w, S) = data
    (; ids, edge, slot_pairs, row_slots, fitted) = jl
    T = real(eltype(S))
    nslots = sum(length, ids)
    gv = zeros(complex(T), nslots)
    φv = zeros(T, nslots)
    phase_free = falses(nslots)
    amp_free = falses(nslots)
    for (k, idk) in pairs(ids), I in eachindex(idk)
        gk = gains[k]
        n = idk[I]
        gv[n], φv[n] = gk.g[I], gk.φ[I]
        amp_free[n] = fitted.amp && gk.touched[I]
        phase_free[n] = fitted.phase && gk.touched[I] && !gk.pinned[I]
    end

    function chi2(g)
        out = zero(Float64)
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

    q = zeros(Float64, length(slot_pairs))
    bφ = zeros(Float64, nslots)
    bℓ = zeros(Float64, nslots)
    for I in CartesianIndices(edge)
        e = edge[I]
        iszero(e) && continue
        wc = w[I]
        wc > 0 || continue
        na, nb = slot_pairs[e]
        m = gv[na] * S[I[1], I[2], I[3]] * conj(gv[nb])
        qc = wc * abs2(m)
        qc > 0 || continue
        u = r[I] / (wc * m) - 1
        q[e] += qc
        bφ[na] += qc * imag(u)
        bφ[nb] -= qc * imag(u)
        bℓ[na] += qc * real(u)
        bℓ[nb] += qc * real(u)
    end
    priors = !isnothing(systems) && any(sys -> !isnothing(sys.prior), Iterators.flatten((systems.phase, systems.amp)))
    step = one(T)
    if priors
        gauge = isnothing(gauge_state) ? nothing : (; gauge_state.rows, gauge_state.d, row_slots)
        δφ = _joint_prior_solve(q, slot_pairs, bφ, φv, phase_free, -1, systems.phase; gauge)
        δℓ = _joint_prior_solve(q, slot_pairs, bℓ, log.(abs.(gv)), amp_free, 1, systems.amp)
        step = T(min(1, _BP_MAX_LINEAR_STEP / max(maximum(abs, δφ), maximum(abs, δℓ))))
    else
        δφ = _joint_normal_solve(q, slot_pairs, bφ, phase_free, -1)
        δℓ = _joint_normal_solve(q, slot_pairs, bℓ, amp_free, 1)
        χ0 = chi2(gv)
        trial = similar(gv)
        step = zero(T)
        for t in (1, 1 // 2, 1 // 4, 1 // 8, 1 // 16)
            @. trial = gv * exp(complex(T(t) * δℓ, T(t) * δφ))
            if chi2(trial) < χ0
                step = T(t)
                break
            end
        end
    end

    shift = zeros(T, nslots)
    if !isnothing(gauge_state)
        for (j, row) in pairs(gauge_state.rows)
            isempty(row) && continue
            res = sum(c * (φv[n] + step * δφ[n]) for (n, c) in row) - gauge_state.d[j]
            α = -res / sum(last, row)
            for n in row_slots[j]
                shift[n] += α
            end
        end
    end
    for (k, idk) in pairs(ids), I in eachindex(idk)
        gk = gains[k]
        n = idk[I]
        gk.touched[I] || continue
        Δφ = step * δφ[n] + shift[n]
        gk.g[I] *= exp(complex(step * δℓ[n], Δφ))
        gk.φ[I] += Δφ
    end
    information = zeros(nslots)
    for (e, (na, nb)) in pairs(slot_pairs)
        information[na] += q[e]
        information[nb] += q[e]
    end
    return information
end

# Solve `Σ_e q_e·(x_a + s·x_b)²`'s normal equations `H·δ = b` over the `free`
# slots, `s = −1` for phase differences and `+1` for log-amplitude sums. A
# ridge of `1e-9` of each slot's own weight fixes the phase's free common
# direction per component and a bipartite component's alternating one without
# moving any measured direction.
function _joint_normal_solve(q, slot_pairs, b, free, s)
    pos = cumsum(free)
    nfree = last(pos)
    Is, Js, Vs = Int[], Int[], Float64[]
    for (e, (na, nb)) in pairs(slot_pairs)
        qe = q[e]
        qe > 0 || continue
        # A fixed end contributes only to the free end's own weight.
        for n in (na, nb)
            free[n] && (push!(Is, pos[n]); push!(Js, pos[n]); push!(Vs, qe))
        end
        if free[na] && free[nb]
            append!(Is, (pos[na], pos[nb]))
            append!(Js, (pos[nb], pos[na]))
            append!(Vs, (s * qe, s * qe))
        end
    end
    H = sparse(Is, Js, Vs, nfree, nfree)
    d = diag(H)
    keep = d .> 0
    δ = zeros(eltype(b), length(b))
    any(keep) || return δ
    Hk = H[keep, keep] + Diagonal(1.0e-9 .* d[keep])
    x = cholesky(Symmetric(Hk)) \ b[free][keep]
    idx = findall(free)[keep]
    δ[idx] .= x
    return δ
end

# The MAP step of one observable under the tracks' priors: `x` minimizes
# `½(x − x₀)ᵀH(x − x₀) − bᵀ(x − x₀) + E(x)` over the free slots, with `H` the
# normal matrix of `_joint_normal_solve` and `E` the tracks' prior energy with
# their levels profiled out (`_track_energy`), and the step is `x − x₀`. A slot
# its track's fit does not weight is held fixed and left out of `E`, which
# marginalizes it. Tracks without a prior are fit by their own data alone.
#
# Conjugate gradients, preconditioned by `M = W + P` with `W` each slot's
# weight in its track's fit and `P` the prior precision, so applying `M⁻¹` is
# one fit of every track (`_track_response`, affine in its observations), and
# `P·p` is the change of `E`'s gradient. `gauge`, when given, holds constraint
# rows `C·x = d` over the slots (`rows`, `d`) and the slots of each row's
# connected component (`row_slots`), one row per component. The start is moved
# onto the constraints along each component's common shift `z`, and every search
# direction is projected onto `C·p = 0` along the same shifts,
# `Π = I − z·(C·z)⁻¹·C`, which is CG on the problem reduced to that subspace.
function _joint_prior_solve(q, slot_pairs, b, x0, free, s, systems; gauge = nothing)
    nslots = length(b)
    weight = zeros(nslots)
    for (e, (na, nb)) in pairs(slot_pairs)
        weight[na] += q[e]
        weight[nb] += q[e]
    end
    tracks = NamedTuple[]
    for sys in systems
        isnothing(sys.prior) && continue
        wt = [free[n] ? Float64(wi) : 0.0 for (n, wi) in zip(sys.slots, sys.w)]
        weight[sys.slots] .= wt
        push!(tracks, merge(sys, (; w = wt)))
    end
    free = free .& (weight .> 0)
    inprior = falses(nslots)
    foreach(t -> inprior[t.slots] .= true, tracks)
    response0 = [_track_response(t, zeros(length(t.slots))) for t in tracks]

    function apply_H(p)
        out = zeros(nslots)
        for (e, (na, nb)) in pairs(slot_pairs)
            v = q[e] * (p[na] + s * p[nb])
            out[na] += v
            out[nb] += s * v
        end
        return out .* free
    end
    # `M⁻¹·v`, with `affine` adding the priors' mean term.
    function apply_Minv(v; affine = false)
        out = zeros(nslots)
        for n in eachindex(out)
            free[n] && !inprior[n] && (out[n] = v[n] / weight[n])
        end
        for (t, f0) in zip(tracks, response0)
            f = _track_response(t, v[t.slots])
            affine || (f .-= f0)
            for (i, n) in pairs(t.slots)
                free[n] && (out[n] = f[i])
            end
        end
        return out
    end

    function prior_gradient(v)
        out = zeros(nslots)
        for t in tracks
            y = [free[n] ? Float64(v[n]) : NaN for n in t.slots]
            g = zeros(length(y))
            _track_energy(t, y; gradient = g)
            for (i, n) in pairs(t.slots)
                free[n] && (out[n] += g[i])
            end
        end
        return out
    end
    gradient0 = prior_gradient(zeros(nslots))
    apply_A(p) = (apply_H(p) .+ prior_gradient(p) .- gradient0) .* free

    # Each constraint row's index `j`, free slots `(slot, coefficient)` and
    # component's free slots, and `C·z` over them.
    rows = isnothing(gauge) ? NamedTuple[] : map(eachindex(gauge.rows)) do j
        (; j, c = [(n, cn) for (n, cn) in gauge.rows[j] if free[n]], z = filter(n -> free[n], gauge.row_slots[j]))
    end
    filter!(row -> !isempty(row.c), rows)
    allunique(Iterators.flatten(row.z for row in rows)) ||
        throw(ArgumentError("the joint bandpass gauge holds more than one constraint on a connected component"))
    cz = [sum(last, row.c) for row in rows]
    function project!(v)
        for (row, czj) in zip(rows, cz)
            γ = sum(cn * v[n] for (n, cn) in row.c) / czj
            v[row.z] .-= γ
        end
        return v
    end
    function project_adjoint!(v)
        for (row, czj) in zip(rows, cz)
            γ = sum(n -> v[n], row.z) / czj
            for (n, cn) in row.c
                v[n] -= cn * γ
            end
        end
        return v
    end

    x = apply_Minv(weight .* x0 .+ b; affine = true)
    x[.!free] .= x0[.!free]
    for (row, czj) in zip(rows, cz)
        res = sum(cn * x[n] for (n, cn) in gauge.rows[row.j]) - gauge.d[row.j]
        x[row.z] .-= res / czj
    end
    r =(b .- apply_H(x .- x0) .- prior_gradient(x)) .* free
    ry = project_adjoint!(copy(r))
    z = apply_Minv(ry)
    py = copy(z)
    rz = dot(ry, z)
    target = 1.0e-20 * rz
    for _ in 1:max(50, 2 * count(free))
        rz <= target && break
        p = project!(copy(py))
        Ap = apply_A(p)
        α = rz / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        ry = project_adjoint!(copy(r))
        z = apply_Minv(ry)
        rz, rz_prev = dot(ry, z), rz
        py .= z .+ (rz / rz_prev) .* py
    end
    return (x .- x0) .* free
end

# The MAP objective of a joint bandpass solve: half the χ² of the visibilities
# against `g_a·S·conj(g_b)`, the scale its track fits weight their data on, plus
# each track's prior energy (`_prior_energy`) under the priors `systems` records,
# one spectral window at a time.
function _joint_objective(gains, data, layout, systems)
    (; r, w, S, ends) = data
    (; loc, tseg, fseg) = layout
    χ2 = 0.0
    for p in axes(r, FeedPair), bi in axes(r, AntennaPair)
        _solvable(ends[bi, p]) || continue
        (a, fa), (b, fb) = ends[bi, p]
        (ka, ia), (kb, ib) = loc[a], loc[b]
        ga, gb = gains[ka].g, gains[kb].g
        for si in axes(r, Scan)
            ta, tb = tseg[a, si], tseg[b, si]
            (iszero(ta) || iszero(tb)) && continue
            s = S[Scan(si), AntennaPair(bi), FeedPair(p)]
            for cell in axes(r, Frequency)
                wc = w[Scan(si), AntennaPair(bi), FeedPair(p), Frequency(cell)]
                wc > 0 || continue
                m = ga[ia, fa, fseg[a, cell], ta] * s * conj(gb[ib, fb, fseg[b, cell], tb])
                χ2 += abs2(r[Scan(si), AntennaPair(bi), FeedPair(p), Frequency(cell)] - wc * m) / wc
            end
        end
    end
    slot = [(k, I) for (k, idk) in pairs(systems.ids) for I in eachindex(idk)]
    energy = 0.0
    for (tracks, value) in ((systems.phase, gk -> gk.φ), (systems.amp, gk -> log.(abs.(gk.g))))
        values = map(value, gains)
        for sys in tracks
            isnothing(sys.prior) && continue
            y = map(sys.slots) do n
                k, I = slot[n]
                gains[k].touched[I] ? Float64(values[k][I]) : NaN
            end
            energy += _track_energy(sys, y)
        end
    end
    return χ2 / 2 + energy
end

# The prior energy of one track system's values `y`, `NaN` where unobserved,
# with each of its levels at the value that minimizes it: the prior acts on the
# track less its level, and a level is flat a priori, so this is the energy
# with the levels profiled out. `gradient`, when given, receives its gradient
# in `y`, which by the levels' optimality is the gradient at those levels.
function _track_energy(sys, y; gradient = nothing)
    (; prior, pieces, x, level) = sys
    pg = isnothing(gradient) && isnothing(level) ? nothing : [zeros(length(p)) for p in pieces]
    L = zeros(length(pieces))
    if !isnothing(level)
        # The energy is quadratic in a piece's level, with slope `−1ᵀ∇E` and
        # curvature `1ᵀ(∇E(y + 1) − ∇E(y))`.
        slope, curve = zeros(sys.nlevel), zeros(sys.nlevel)
        g1 = similar(first(pg), 0)
        for (j, p) in pairs(pieces)
            _prior_energy(prior, y[p], x[p]; gradient = pg[j])
            resize!(g1, length(p))
            _prior_energy(prior, y[p] .+ 1, x[p]; gradient = g1)
            slope[level[j]] += sum(pg[j])
            curve[level[j]] += sum(g1) - sum(pg[j])
        end
        for j in eachindex(pieces)
            c = curve[level[j]]
            L[j] = c > 0 ? slope[level[j]] / c : 0.0
        end
    end
    E = 0.0
    for (j, p) in pairs(pieces)
        E += _prior_energy(prior, y[p] .- L[j], x[p]; gradient = isnothing(pg) ? nothing : pg[j])
    end
    if !isnothing(gradient)
        fill!(gradient, 0)
        for (j, p) in pairs(pieces)
            gradient[p] .= pg[j]
        end
    end
    return E
end

# Hold an iteration of the joint bandpass solve to decreasing the MAP objective
# (`_joint_objective`, under the hyperparameters this iteration resolved): from
# the gains it started at (`previous`, `previous_φ`) the step to the current
# gains is halved, the source coherence re-solved each time, until the
# objective is no higher than its starting value up to the gains' rounding, down to a 64th
# of the step. A linear gauge constraint both ends satisfy holds all along the
# step. Returns `false`, with the starting gains restored, when no such step
# decreases the objective.
function _backtrack_joint!(gains, previous, previous_φ, data, layout, systems)
    J = _joint_objective(gains, data, layout, systems)
    current = [(copy(gk.g), copy(gk.φ)) for gk in gains]
    current_S = copy(data.S)
    function place!(t)
        for (gk, g0, φ0, (g1, φ1)) in zip(gains, previous, previous_φ, current)
            @. gk.φ = φ0 + t * (φ1 - φ0)
            @. gk.g = ifelse(iszero(g0) | iszero(g1), g1, abs(g0)^(1 - t) * abs(g1)^t * cis(gk.φ))
        end
        _update_source_coherence!(data, gains, layout)
        return gains
    end
    place!(0)
    J0 = _joint_objective(gains, data, layout, systems)
    # The gains are held at their own precision, which sets the objective's.
    bound = J0 + 16 * eps(real(eltype(first(gains).g))) * abs(J0)
    J <= bound && (place!(1); copyto!(data.S, current_S); return true)
    for t in (1 // 2, 1 // 4, 1 // 8, 1 // 16, 1 // 32, 1 // 64)
        place!(t)
        _joint_objective(gains, data, layout, systems) <= bound && return true
    end
    place!(0)
    return false
end

# The largest change of any gain's log from `previous`, the gains a sweep
# started from, in standard errors of its `information` (slots numbered by
# `ids`), and the `(k, I)` of that gain. A change within the gains' own
# precision is none; a gain that reaches or leaves zero moves infinitely far
# unless the data hold no information on it.
function _largest_weighted_change(gains, previous, ids, information)
    worst, at = 0.0, nothing
    for (k, gk) in pairs(gains), I in eachindex(gk.g, previous[k])
        wn = information[ids[k][I]]
        wn > 0 || continue
        new, old = gk.g[I], previous[k][I]
        Δ = iszero(new) || iszero(old) ? (new == old ? 0.0 : Inf) : abs(complex(log(abs(new / old)), angle(new / old)))
        Δ > 16 * eps(real(eltype(gk.g))) || continue
        change = sqrt(wn) * Δ
        change > worst && ((worst, at) = (change, (k, CartesianIndices(gk.g)[I])))
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
# log-amplitude, circular-mean reference phase, matching the closure tier's
# convention — and write it into the station block that carries it. Both gauges
# are over the frequency track of one time segment, so each of a station's
# segments is normalized on its own. A (station, feed, segment) that `touched`
# nowhere is left unwritten, holding whatever θ already had, which is unit gain;
# so is a station no block covers. With a level, the reference comes off the
# level and the shape is written as fit, as in the closure tier's writers.
#
# `gains[k]` holds the stations of phase block `k` and amplitude block `k`, in
# the blocks' own order; `level_writers[a]` holds station `a`'s
# `(; phase, amp)` level writers (see `_station_level`), each `nothing` without
# a level.
#
# The band means are removed at θ's precision, which may exceed the gains': the
# gauge is a property of θ.
function _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout, level_writers; max_logamp)
    for (k, gk) in pairs(gains)
        pb, ab = get(phase_blocks, k, nothing), get(amp_blocks, k, nothing)
        block = something(pb, ab)
        T = eltype(block.θ)
        (; g, φ, touched) = gk
        node(b, f) = isnothing(b) ? 0 : _feed_node(b.plan.tying, f)
        for (ai, a) in pairs(block.stations), f in axes(g, Feed), ts in layout.present[a]
            valid = view(touched, ai, f, :, ts)
            any(valid) || continue
            pnode, anode = node(pb, f), node(ab, f)
            phase(fs) = T(φ[ai, f, fs, ts])
            logamp(fs) = log(T(abs(g[ai, f, fs, ts])))
            mphase = angle(sum(cis(phase(fs)) for fs in axes(g, Frequency) if valid[fs]))
            mlogamp = sum(logamp(fs) for fs in axes(g, Frequency) if valid[fs]) / count(valid)
            lw = map(l -> _level_at(l, ts), level_writers[a])
            Lp = _write_level!(lw.phase, f, ts, Lj -> rem2pi(T(Lj) - mphase, RoundNearest))
            La = _write_level!(lw.amp, f, ts, Lj -> T(Lj) - mlogamp)
            for fs in axes(g, Frequency)
                valid[fs] || continue
                if !iszero(pnode)
                    pb.θ[1, pnode, fs, ts, ai] = isnothing(Lp) ?
                        rem2pi(phase(fs) - mphase, RoundNearest) : phase(fs) - T(Lp[lw.phase.seg[fs]])
                end
                iszero(anode) && continue
                la = logamp(fs) - mlogamp
                gated = abs(la) > max_logamp
                ab.θ[1, anode, fs, ts, ai] = if isnothing(La)
                    gated ? zero(la) : la
                else
                    Lv = T(La[lw.amp.seg[fs]])
                    gated ? mlogamp - Lv : logamp(fs) - Lv
                end
            end
        end
    end
    return nothing
end

"""
    JointSmoother(; max_iterations = 200, tolerance = 0.01)

Fit the station bandpass against the actual complex visibilities, with each
component's prior acting inside the solve rather than as a fit applied to it
afterwards. The alternating complex-gain / per-scan source-coherence scheme is
[`solve_joint_bandpass!`](@ref)'s; a prior changes the per-(station, feed) gain
update, which fits the whole track under the prior instead of solving each
segment on its own, and re-estimates the prior's hyperparameters, and any
levels, every sweep. Iterated to convergence that is MAP estimation under the
two priors.

Because the per-scan source term absorbs a baseline's own structure, this suits a
resolved or polarized calibrator, where [`PerTrackSmoother`](@ref)'s closure
assumption would bias the bandpass. It solves one complex gain per
(station, feed, frequency segment), so a model with both a phase and a
log-amplitude shape must give them one frequency and one time segmentation per
station; their priors and levels may differ. A model with only one of them
holds the other at zero: unit amplitude, or zero phase.

Sweeps stop once no gain moves by more than `tolerance` of its standard error
from the data, and warn if `max_iterations` sweeps pass first: an unconverged
solve still depends on its starting point.
"""
struct JointSmoother <: AbstractBandpassSmoother
    max_iterations::Int
    tolerance::Float64
end
JointSmoother(; max_iterations::Integer = 200, tolerance::Real = 0.01) =
    JointSmoother(Int(max_iterations), Float64(tolerance))

can_fit(::JointSmoother, tc, geom) = _fits_bandpass_track(tc)

function validate_model(::JointSmoother, model)
    validate_bandpass_groups(model)
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
                "$(repr(la.Ti)) (logamp). Use `smoother = PerTrackSmoother()` to give " *
                "the two observables different time resolutions.",
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

function solve_bandpass!(sm::JointSmoother, θ, results, setup; gauge::AbstractGauge)
    geom = setup.geom
    phase_blocks = bandpass_blocks(setup, θ, :phase)
    amp_blocks = bandpass_blocks(setup, θ, :logamp)
    # One complex gain per (station, feed, segment) means one time segmentation
    # for both observables — `validate_model` holds each station's two plans to
    # it — so either side's table is the whole solve's.
    tseg = _station_time_segments(_gain_blocks(phase_blocks, amp_blocks), results, length(geom.stations))
    arrays(f, blocks) = isempty(blocks) ? nothing : [f(b, geom) for b in blocks]
    phase_status = arrays(_block_status_array, phase_blocks)
    amp_status = arrays(_block_status_array, amp_blocks)
    phase_priors = arrays(_block_prior_array, phase_blocks)
    amp_priors = arrays(_block_prior_array, amp_blocks)
    for idx in _joint_scan_groups(tseg)
        solve_joint_bandpass!(
            θ, results[idx], geom, phase_blocks, amp_blocks;
            phase_level_blocks = bandpass_level_blocks(setup, θ, :phase),
            amp_level_blocks = bandpass_level_blocks(setup, θ, :logamp),
            gauge, max_iterations = sm.max_iterations, tolerance = sm.tolerance,
            tseg = view(tseg, :, idx), phase_status, amp_status, phase_priors, amp_priors,
        )
    end
    leaves(arrays) = isnothing(arrays) ? nothing : _block_leaves(arrays)
    report = bandpass_track_report(
        leaves(phase_status), leaves(amp_status);
        phase_priors = leaves(phase_priors), amp_priors = leaves(amp_priors),
    )
    _warn_degenerate_bandpass(report)
    return report
end

"""
    solve_joint_bandpass!(θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
                          phase_level_blocks = [], amp_level_blocks = [],
                          gauge, max_iterations = 200, tolerance = 0.01,
                          max_logamp = log(10.0), phase_status = nothing,
                          amp_status = nothing, phase_priors = nothing,
                          amp_priors = nothing, tseg = nothing)

Jointly solve the per-(station, feed, frequency segment) complex bandpass gain
and a per-scan, per-station-pair, per-feed-pair constant source coherence (see
the module comment above `_reduce_scan_segments!` for the model and the
alternating scheme), then gauge-fix each (station, feed, time segment) track and
write the result into the station blocks of the two observables
(`_write_joint_bandpass!`).

`phase_blocks` and `amp_blocks` are [`bandpass_blocks`](@ref)`(setup, θ, :phase)`
and `(…, :logamp)` — station blocks over the SAME `θ` this call is handed, since
each block's `θ` is a view into it — and `phase_level_blocks`/`amp_level_blocks`
the matching [`bandpass_level_blocks`](@ref bandpass_blocks), empty without a level. A
station-uniform model gives one block per component spanning every station;
where the model differs across stations, each block carries its own stations,
feed tying, segments and prior, and a station no block covers is left at unit
gain. Each track is fit under its own block's prior, one spectral window at a
time, with the prior's hyperparameters and any levels re-estimated every sweep.
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
constraint is imposed in every iteration's joint step. Where stations' frequency
segmentations differ, a pin can hold part of a track; a phase prior, which
relates the track's segments, then throws rather than fit around it.

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

Convergence is judged on the largest per-iteration change of any
(station, feed, segment) gain's log, in standard errors of that gain from the
data (the inverse square root of its weight in the joint step), so a gain the
data barely constrain counts for as little as its precision. The solve stops
once that is below `tolerance`, or once an iteration under a prior cannot lower
the objective yet moved no gain by `tolerance`. The scans handed to one call are
a connected piece of the coupling graph, so this is one criterion over one
coupled problem: nodes that share no data are solved by separate calls rather
than being averaged into a common tolerance.

A station-by-station sweep moves slowly along any direction that only weakly
measured correlations constrain, such as the phase between feeds, which only
the correlations pairing different feeds see. Each sweep is therefore followed
by one Gauss–Newton step on every gain slot at once (`_joint_gain_step!`), which
moves those directions as fast as any other and imposes the gauge; the two
steps share their fixed point, the minimum of the objective.

Under a prior that objective is the posterior: half the χ² of the visibilities
plus each track's prior energy at the hyperparameters the sweep resolves, with
any level profiled out (the prior acts on the track less its level). The sweep then linearizes each slot about its current gain,
whose fixed point is that posterior's mode, and every iteration is held to
lowering it by backtracking along its step. An iteration that cannot lower it
stops the solve, with a warning if it moved a gain by `tolerance` or more: the
objective is then flat or not convex there, typically because a prior too stiff
for the data drives an amplitude toward zero, where the mode does not exist.

`phase_status`/`amp_status`, when given, hold one `(AntennaName, Feed, Frequency, Ti)`
array per block of `phase_blocks`/`amp_blocks` ([`bandpass_track_report`](@ref)),
receiving each spectral window's `_BP_TRACK_*` outcome code from the final
sweep, and `phase_priors`/`amp_priors` one `(AntennaName, Feed, Ti)` array per block
receiving each track's resolved prior; a call solving part of a track writes
only its own time segments.
"""
function solve_joint_bandpass!(
        θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
        phase_level_blocks = NamedTuple[], amp_level_blocks = NamedTuple[],
        gauge::AbstractGauge, max_iterations::Integer = 200, tolerance::Real = 0.01,
        max_logamp::Real = _BP_MAX_LOGAMP,
        phase_status = nothing, amp_status = nothing,
        phase_priors = nothing, amp_priors = nothing,
        tseg::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
    )
    isempty(scans) && return θ
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
    r, w = _reduce_all_scans(scans, cells, Frequency(_segment_lookup(geom.channel_freqs, cells)))
    ends = _cell_nodes(r, geom.stations, PerFeed())
    S = zeros(eltype(r), (Scan(axes(r, Scan)), dims(r, AntennaPair), dims(r, FeedPair)))
    data = DimStack((; r, w, S, ends))

    # The time segments each station is actually solved for here, in its own
    # segmentation's numbering — the numbering of its block's `Ti` axis.
    tsg = isnothing(tseg) ? ones(Int, nant, length(scans)) : tseg
    loc = _block_locations(blocks, nant)
    layout = (;
        loc, tseg = tsg, fseg,
        present = [sort!(filter!(!iszero, unique(view(tsg, a, :)))) for a in axes(tsg, 1)],
        touching = _joint_bandpass_touching(ends, nant, geom.nfeed),
    )
    gains = [_block_gains(b, geom, eltype(r)) for b in blocks]
    # The gauge fixes phases; a solve that holds every phase at zero has none to fix.
    gauge_state = isempty(phase_blocks) ? nothing : _joint_bandpass_gauge!(gains, data, layout, blocks, gauge)

    segs = [_shape_segments(b.plan, geom) for b in blocks]
    phase_level = _station_levels(phase_level_blocks, segs, loc, geom, θ, real(eltype(r)))
    amp_level = _station_levels(amp_level_blocks, segs, loc, geom, θ, real(eltype(r)))
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
    levels = [(; phase = _level_values(phase_level[a]), amp = _level_values(amp_level[a])) for a in 1:nant]
    _reject_partial_pins(gains, fits, loc, geom)
    joint = merge(
        _joint_gain_layout(gains, data, layout, gauge_state),
        (; fitted = (; phase = !isempty(phase_blocks), amp = !isempty(amp_blocks))),
    )
    # Under a prior the sweep linearizes about the current gains, whose fixed point
    # is the MAP, and each iteration is held to decreasing the MAP objective.
    guarded = any(f -> !isnothing(f) && _has_prior(f), fits)
    systems = (; joint.ids, phase = NamedTuple[], amp = NamedTuple[])

    _update_source_coherence!(data, gains, layout)
    change, worst = Inf, nothing
    previous = [copy(gk.g) for gk in gains]
    previous_φ = [copy(gk.φ) for gk in gains]
    stalled = false
    for iter in 1:max_iterations
        foreach((dst, gk) -> copyto!(dst, gk.g), previous, gains)
        foreach((dst, gk) -> copyto!(dst, gk.φ), previous_φ, gains)
        # The joint step imposes the gauge, so the sweep fits free of it.
        nupdated, declined = _update_station_gains!(
            gains, data, layout;
            fits, levels, seed = iter == 1, relinearize = guarded && iter > 1, systems,
            phase_status, amp_status, phase_priors, amp_priors,
        )
        iszero(nupdated) && _throw_unchanged_sweep(declined, geom)
        information = _joint_gain_step!(gains, data, joint, gauge_state, systems)
        _update_source_coherence!(data, gains, layout)
        change, worst = _largest_weighted_change(gains, previous, joint.ids, information)
        # A sweep the safeguard rejects has converged if it moved no gain by `tolerance`.
        if guarded && iter > 1 && !_backtrack_joint!(gains, previous, previous_φ, data, layout, systems)
            stalled = change >= tolerance
            break
        end
        guarded && ((change, worst) = _largest_weighted_change(gains, previous, joint.ids, information))
        change < tolerance && break
    end
    location = isnothing(worst) ? "" : _slot_description(geom, loc, worst...)
    stalled && @warn "the joint bandpass solve stopped after a sweep that could not lower its MAP " *
        "objective even at a 64th of its step; that sweep moved a gain by $change of its standard " *
        "error$location. The objective is flat or not convex near this point, often because a prior " *
        "is too stiff for the data, which can drive a gain's amplitude toward zero."
    stalled || change < tolerance || @warn "the joint bandpass solve did not converge in " *
        "$max_iterations sweeps: the last sweep moved a gain by $change of its standard error$location, " *
        "against a tolerance of $tolerance; raise `JointSmoother(; max_iterations)`"

    level_writers = [
        (; phase = _joint_level_writer(phase_level[a], levels[a].phase), amp = _joint_level_writer(amp_level[a], levels[a].amp))
            for a in 1:nant
    ]
    _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout, level_writers; max_logamp)
    return θ
end

function _throw_unchanged_sweep(declined, geom::DataGeometry)
    head = "solve_joint_bandpass!: the joint bandpass changed no gain in a sweep. "
    isempty(declined) && throw(ArgumentError(head * "No gain slot had data to fit."))
    named = join(("$(geom.stations[a]) (feed $f, time segment $ts)" for (a, f, ts) in first(declined, 3)), ", ")
    more = length(declined) > 3 ? ", …" : ""
    throw(
        ArgumentError(
            head * "$(length(declined)) phase tracks were declined at unwrapping: $named$more. " *
                "Their phase changes between adjacent frequency segments are too scattered to " *
                "unwrap, which a phase prior requires; without a phase prior the segments are " *
                "fit independently and are not declined.",
        ),
    )
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

# A pin that holds part of a track leaves the rest fitted, which is the
# constrained fit only when the track's segments are fit independently. A phase
# prior relates them, so a partly pinned track under one is rejected rather than
# fit with one segment overwritten.
function _reject_partial_pins(gains, fits, loc, geom::DataGeometry)
    for a in eachindex(loc, fits)
        k, ai = loc[a]
        (iszero(k) || isnothing(_prior(fits[a].phase))) && continue
        st = gains[k]
        for ts in axes(st, Ti), feed in axes(st, Feed)
            track = view(st.pinned, ai, feed, :, ts)
            np, nfs = count(track), length(track)
            (iszero(np) || np == nfs) && continue
            throw(
                ArgumentError(
                    "solve_joint_bandpass!: the phase gauge pins $np of the $nfs frequency " *
                        "segments of station $(geom.stations[a]) (feed $feed, time segment " *
                        "$ts), because the stations' frequency segmentations tie those " *
                        "channels into fewer independent common-phase modes than this " *
                        "station has segments. Its phase prior relates the track's segments, " *
                        "so holding part of it at zero is not the constrained fit the prior " *
                        "asks for. Drop the phase prior, or give the stations frequency " *
                        "segmentations that nest.",
                ),
            )
        end
    end
    return nothing
end
