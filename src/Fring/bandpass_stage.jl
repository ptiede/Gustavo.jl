# ── Bandpass stage: per-channel station phase/log-amp over scan windows ──────
#
# The `Bandpass` step reads every scan (accumulate → return the scan's
# contribution) and folds the contributions in group order, which keeps the
# result deterministic at any concurrency.
# What is fit is the step's model tree (see [`default_bandpass_terms`](@ref)): a
# `Bandpass()` component per observable, whose frequency segments are the pieces
# its prior is fit over. How it is solved is pluggable through
# `AbstractBandpassSmoother`, in two tiers. `PerTrackSmoother` sums every scan's
# residual into one accumulator, runs the per-channel closure solves — which
# assume a baseline's source term cancels — and fits each resulting
# (station, feed) track under its component's prior. `JointSmoother` runs
# [`solve_joint_bandpass!`](@ref) instead, fitting the complex visibilities
# against an explicit per-scan source term, for when that assumption fails, with
# the priors entering inside the gain update.
#
# The graph/solve helpers (`_solve_observable!`, `_cell_noise2`, `_node`) live
# in stationize.jl/adhoc.jl; the prior fits live in prior_fits.jl.

"""
    default_bandpass_terms(; freq = PerSpectralWindow()) -> GainModel

The default [`Bandpass`](@ref Gustavo.Bandpass) step model: a time-stable, per-feed
value per channel for each observable — a `phase.bandpass` and a
`logamp.bandpass` component, each `Calibration.Bandpass()` with no prior. `freq`
(an `AbstractFrequencySegmentation`) sets the frequency segments, which a prior
added to either component is fit over independently.

Either group of a `Bandpass` model may be empty. Fit one observable only by
keeping just that group, e.g.

    Bandpass(model = GainModel(; phase = default_bandpass_terms().phase),
             smoother = PerTrackSmoother())

fits the phase bandpass alone. Uniform across every antenna. A prior on a
component (`GainComponent(...; prior)`) states how its values relate within a
segment.
"""
function default_bandpass_terms(; freq::AbstractFrequencySegmentation = PerSpectralWindow())
    bandpass = GainComponent(Calibration.Bandpass(); Ti = GlobalTime(), Frequency = freq, Feed = PerFeed())
    return GainModel(phase = (; bandpass), logamp = (; bandpass))
end

"""
    AbstractBandpassSmoother

How the [`Bandpass`](@ref Gustavo.Bandpass) step turns the accumulated
per-channel residual into station bandpass tracks, under each component's
prior. Concretely [`PerTrackSmoother`](@ref), which solves the per-channel closures
and fits each track, or [`JointSmoother`](@ref), which fits the complex
visibilities against an explicit per-scan source term.

# Implementing a smoother

Define:

    Gustavo.Fring.can_fit(sm::MySmoother, tc::Calibration.GainComponent, geom) -> Bool
    Gustavo.Fring.solve_bandpass!(sm::MySmoother, θ, results, setup; gauge) -> report

[`can_fit`](@ref) declares which model components the smoother can solve; it
defaults to `false`, so an undeclared model is rejected at compile time
rather than leaving θ blocks unsolved. Both shipped smoothers accept
`GainComponent(Calibration.Bandpass(); Ti = <GlobalTime, InstrumentScans or
TimeBlocks>, Frequency = <any segmentation>, Feed = PerFeed(), prior = <nothing,
RandomWalkPrior(Frequency; …) or OUPrior(Frequency; …)>)` and nothing else — a
time segmentation whose segments each span several scans, solved one segment at
a time.

`solve_bandpass!` writes into `θ`'s bandpass blocks. `results` is the
per-scan `(; rl, wl, ti, source)` list in group-index order: `rl` and `wl` are
`accumulate_bandpass`'s sums, `ti` the scan's first sample on the
solve's time axis (hence which time segment it falls in). `setup` is
`(; station_pairs, feeds, layout, geom, bp_path, amp_path)`, built once per
solve: `station_pairs` and `feeds` label the sums' `StationPair` and `FeedPair`
axes, and `geom` is the solve's `DataGeometry`, whose `stations` are θ's
stations. Reach each observable's parameters through
[`bandpass_blocks`](@ref)`(setup, θ, :phase)` / `(…, :logamp)` rather than the
paths directly. `report` is published on the step's solution record and
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

# The `false` default of `can_fit` (capability.jl) makes an undeclared smoother
# reject loudly instead of accepting silently.
can_fit(::AbstractBandpassSmoother, tc, geom) = false

# A bandpass track is one value per channel, per feed, held over a stretch of
# time, under a prior along frequency. Both smoothers pool the scans of a time segment and
# solve that stretch as a unit, so any segmentation whose segments span scans
# fits: `GlobalTime` is the whole track, `InstrumentScans` breaks it at named
# epochs, `TimeBlocks` at a fixed cadence. `PerScan` and `PerIntegration` do not
# — a segment holding one scan leaves the band mean of the gain degenerate with
# that scan's own source coherence, which needs the deviation-from-template
# scheme neither smoother implements.
_fits_bandpass_track(tc) =
    tc.term isa Calibration.Bandpass &&
    tc.Ti isa Union{GlobalTime, InstrumentScans, TimeBlocks} &&
    tc.Feed isa PerFeed &&
    tc.prior isa Union{Nothing, RandomWalkPrior{Frequency}, OUPrior{Frequency}}

# A bandpass plan's frequency segments as channel groups: the pieces its prior is
# fit over, each independently.
_bandpass_pieces(plan) = segment_groups(plan.fseg_id, length(plan.nchan_seg))

# Where channel `c` sits in a `Bandpass` leaf: its position within its segment,
# and the segment.
_channel_slot(plan, c) = (plan.xf[c], plan.fseg_id[c])

"""
    validate_bandpass_groups(model)

The structural requirement the [`Bandpass`](@ref Gustavo.Bandpass) step
places on every smoother's model: at least one component overall, and at
most one per group (the step hands `solve_bandpass!` one plan per
observable). The default `validate_model` for
[`AbstractBandpassSmoother`](@ref), and the base a smoother's own method
must re-establish.
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
    for (name, n) in ((:phase, np), (:logamp, na))
        n <= 1 || throw(
            ArgumentError(
                "Bandpass model group `$name` holds $n components; the bandpass solve fits " *
                    "one track set per observable, so each group holds at most one.",
            ),
        )
    end
    return nothing
end

validate_model(sm::AbstractBandpassSmoother, model) = validate_bandpass_groups(model)

# The path to a bandpass observable's component in a step layout's plantree,
# `nothing` when that group compiles none. `validate_bandpass_groups` caps each
# group at one component, so every level of the descent carries exactly one key
# and the path is unambiguous however the user nested the names. The plantree's
# type is a compile-time constant, so accumulating the path as a tuple keeps the
# descent inferrable and the splat into `station_blocks` type-stable.
_bandpass_path(plantree, group::Symbol) = _component_path(plantree[group], (group,))

# Descend a plantree subtree to its single component; a node that is not a named
# subtree is that component.
_component_path(_, path::Tuple{Vararg{Symbol}}) = path
function _component_path(nt::NamedTuple, path::Tuple{Vararg{Symbol}})
    isempty(nt) && return nothing
    k = only(keys(nt))
    return _component_path(nt[k], (path..., k))
end

"""
    bandpass_blocks(setup, θ, group::Symbol) -> Vector

The station blocks of the bandpass's `:phase` or `:logamp` observable —
[`Calibration.station_blocks`](@ref) resolved against `setup`'s recorded path —
empty when the model compiles no component for that observable. Each block is
`(; stations, θ, plan)`, and a station-uniform model yields exactly one block
spanning every station.

The path is resolved by NAME through the layout's component tree: the flat
`layout.plans` list holds one entry per signature group, so its positions do not
name the two observables once a model differs across stations.
"""
function bandpass_blocks(setup, θ, group::Symbol)
    path = group === :phase ? setup.bp_path : setup.amp_path
    path === nothing && return NamedTuple[]
    return station_blocks(setup.layout, θ, path...)
end

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
channel), `rl = Σ w·v` and `wl = Σ w`, on data already gain-corrected by the
pipeline's corrections. Both are `DimArray`s over `StationPair(station_pairs)`,
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
    rl = zeros(promote_type((eltype(p.wv) for p in parts)...), ax)
    wl = zeros(promote_type((eltype(p.ws) for p in parts)...), ax)
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
# `(Ant(stations), Feed, segments)`, in the sums' real type; the seed solves each
# feed as its own node.
function _seed_phase_tracks(
        rbar_bp, wbar_bp, stations, fsegs, segments::Frequency;
        gauge::AbstractGauge = PinAntenna(1), snr_floor::Real = 1.0,
    )
    noise2 = _cell_noise2(rbar_bp, wbar_bp, Frequency)
    T = real(eltype(rbar_bp))
    nodes = _cell_nodes(rbar_bp, stations, PerFeed())
    val = zeros(T, dims(nodes))
    wt = zeros(T, dims(nodes))
    mask = fill!(similar(nodes, Bool), false)
    tracks = (_station_dim(stations), Feed(1:2), segments)
    phase = fill!(zeros(T, tracks), T(NaN))
    prec = zeros(T, tracks)
    solved = falses(length(stations), 2)
    for (fs, chans) in enumerate(fsegs)
        fill!(mask, false)
        for bi in axes(nodes, StationPair), p in axes(nodes, FeedPair)
            c = (StationPair(bi), FeedPair(p))
            _solvable(nodes[c...]) || continue
            (a, fa), (b, fb) = nodes[c...]
            r, w, w2 = _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            snr2 = _segment_snr2(r, w, w2, noise2[c...])
            snr2 >= snr_floor^2 || continue
            val[c...] = angle(r)
            wt[c...] = snr2
            mask[c...] = true
            prec[a, fa, fs] += snr2
            prec[b, fb, fs] += snr2
        end
        _solve_observable!(view(phase, Frequency(fs)), solved, val, wt, mask, nodes, gauge; rewrap = 0)
    end
    return phase, prec
end

# Gauge and write a solved phase bandpass `(Ant, Feed, Frequency)` over channels:
# each (station, feed) track is referenced to its circular-mean phase over the band,
# so the bandpass carries shape only and applies zero net phase. `phase`'s `Ant`
# axis is θ's stations, in θ's order.
function _write_phase_bandpass!(θ, plan, phase, ts::Integer = 1)
    leaf = _component_leaf(plan, θ)
    for a in axes(phase, Ant), f in axes(phase, Feed)
        node = _feed_node(plan.tying, f)
        node == 0 && continue
        track = view(phase, Ant(a), Feed(f))
        acc = sum(v -> isfinite(v) ? cis(v) : zero(complex(v)), track)
        abs(acc) > 0 || continue
        m = angle(acc)
        for c in axes(track, Frequency)
            v = track[c]
            isfinite(v) || continue
            j, fs = _channel_slot(plan, c)
            leaf[j, node, fs, ts, a] = rem2pi(v - m, RoundNearest)
        end
    end
    return θ
end

# One frequency segment's coherent residual `(r, w, w2)`: the sums of the
# accumulators over the channels it holds, plus `w2 = Σ wᶜ²`, which converts a
# per-channel noise variance into the variance of this segment's normalized
# value `r/w` — `n2 · w2 / w²`, or `n2/k` for `k` equally-weighted channels.
# Scaling the noise any other way would make a wide block look worse than its
# channels, and the SNR gate would reject the very observations grouping exists
# to strengthen.
#
# A one-channel segment leaves all three quantities at that channel's own.
function _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
    r = zero(eltype(rbar_bp))
    w = zero(eltype(wbar_bp))
    w2 = zero(eltype(wbar_bp))
    for gc in chans
        cell = (StationPair(bi), FeedPair(p), Frequency(gc))
        rc = rbar_bp[cell...]
        wc = wbar_bp[cell...]
        (isfinite(rc) && isfinite(wc) && wc > 0) || continue
        r += rc
        w += wc
        w2 += wc^2
    end
    return r, w, w2
end

# Segment SNR² under the per-channel noise estimate `n2` (`NaN` when the track
# had too few channels to estimate one — then fall back to the weight itself).
_segment_snr2(r, w, w2, n2) =
    isfinite(n2) && n2 > 0 ? abs2(r / w) * w^2 / (n2 * w2) : abs2(r) / w


# Narrow-spike guard: additive contamination (pcal tones, RFI) violates the
# multiplicative gain model — a contaminated channel shows excess amplitude,
# the fit hands it |g| > 1, and `apply_calibration` would then UP-weight it
# (w → w·|g|²), amplifying exactly the channels that should be distrusted.
# Genuine passband structure is smooth or negative (roll-off), so narrow
# positive log-amp outliers vs the per-(station, feed, piece) robust scale are
# excised (left unapplied, |g| = 1) instead of trusted. `spike_sigma = 0`
# disables the guard. `pieces` holds the channel groups the scale is taken over.
function _spike_guard!(la, pieces, spike_sigma::Real)
    spike_sigma > 0 || return la
    T = eltype(la)
    for sidx in pieces
        for a in axes(la, Ant), f in axes(la, Feed)
            track = view(la, Ant(a), Feed(f))
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

# Gauge and write a solved log-amp bandpass `(Ant, Feed, Frequency)` over channels: zero
# band-mean per (station, feed), so the bandpass carries shape only and applies
# unit net amplitude. `la`'s `Ant` axis is θ's stations, in θ's order.
function _write_amp_bandpass!(θ, plan, la, max_logamp::Real, ts::Integer = 1)
    leaf = _component_leaf(plan, θ)
    # The band mean is removed at θ's precision, which may exceed the track's:
    # the gauge is a property of θ.
    T = eltype(leaf)
    for a in axes(la, Ant), f in axes(la, Feed)
        node = _feed_node(plan.tying, f)
        node == 0 && continue
        track = view(la, Ant(a), Feed(f))
        n = count(isfinite, track)
        n == 0 && continue
        m = sum(v -> isfinite(v) ? T(v) : zero(T), track) / n
        for c in axes(track, Frequency)
            v = T(track[c])
            isfinite(v) || continue
            # Leave implausibly-large corrections unapplied (|g| = 1). A prior that
            # interpolates gaps self-regularizes, but an unconstrained fit can hand a
            # low-SNR band-edge channel that barely clears the gate a huge log-amp;
            # applying it would up-weight that channel's noise, since
            # `apply_calibration` scales weights by |g|². The bound is generous
            # (|g| ≤ 10) so real passband roll-off/structure passes unchanged — only
            # pathological noise blow-ups are gated.
            j, fs = _channel_slot(plan, c)
            leaf[j, node, fs, ts, a] = abs(v - m) > max_logamp ? zero(v) : v - m
        end
    end
    return θ
end

# Free per-segment closure seed for the log-amp bandpass: the sum closure
# `log|V̄_ab| = la_a + la_b` solved independently in each frequency segment, plus
# each (station, feed, segment)'s summed gate weight as its precision, both over
# `(Ant(stations), Feed, segments)` in the sums' real type. Segments with no
# gated observation are left `NaN` for a prior to estimate — or not.
function _seed_amp_tracks(
        rbar_bp, wbar_bp, stations, fsegs, segments::Frequency;
        snr_floor::Real = 1.0, ridge::Real = 1.0e-6,
    )
    noise2 = _cell_noise2(rbar_bp, wbar_bp, Frequency)
    T = real(eltype(rbar_bp))
    nodes = _cell_nodes(rbar_bp, stations, PerFeed())
    val = zeros(T, dims(nodes))
    wt = zeros(T, dims(nodes))
    mask = fill!(similar(nodes, Bool), false)
    tracks = (_station_dim(stations), Feed(1:2), segments)
    la = fill!(zeros(T, tracks), T(NaN))
    prec = zeros(T, tracks)
    for (fs, chans) in enumerate(fsegs)
        fill!(mask, false)
        for bi in axes(nodes, StationPair), p in axes(nodes, FeedPair)
            c = (StationPair(bi), FeedPair(p))
            _solvable(nodes[c...]) || continue
            (a, fa), (b, fb) = nodes[c...]
            r, w, w2 = _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            snr2 = _segment_snr2(r, w, w2, noise2[c...])
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

# Solve the sum closure over the masked cells of `val` into `la` `(nant, 2)`,
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
    column = zeros(Int, 2nant)
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
        la[(n - 1) % nant + 1, n > nant ? 2 : 1] = x[j]
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

# Fraction of a solve's tracks that may come back flat or declined before the
# bandpass as a whole is worth a warning.
const _BP_DEGENERATE_WARN_FRACTION = 0.25

"""
    PerTrackSmoother(; eltype = nothing)

Solve the bandpass one (station, feed) track at a time: the per-channel closure
graph solve seeds real phase and log-amp tracks, then each track is fit under
its component's prior, one frequency segment of the component at a time.

Phase tracks are unwrapped along frequency within each segment before the fit,
so a prior sees a continuous branch rather than a sawtooth. A prior fills
channels the closure solve had no data for; without one they stay unapplied.

The scans of a time segment are pooled before the solve, each first rotated by
its own band-averaged phase per (baseline, feed pair). The pooled sums are held
in `eltype`, a real floating-point type; `nothing` keeps the data's.

The closure assumes a baseline's source term cancels out of the per-channel
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

# Fit one (station, feed, time segment) track in place under `prior`, one piece
# per frequency segment of its component: channel values `track`, weights `w`,
# channel frequencies `x`, and `pieces` the channel groups. The prior's
# hyperparameters are resolved once over all of the track's pieces, and each
# piece keeps its own level. Returns the resolved prior — `prior` itself when the
# track carries no data.
#
# `unwrap` re-references each piece of a phase track to a continuous branch along
# frequency first, because the prior fits a real track and the ±π branch cuts of
# a raw phase solve would otherwise read as genuine structure. A piece whose
# branch the data do not determine (`phase_unwrap_ambiguity` past
# `_BP_MAX_UNWRAP_AMBIGUITY`) is dropped instead; see that constant.
#
# `status` receives one `_BP_TRACK_*` code per piece.
function _fit_track!(track, w, x, pieces, prior; unwrap::Bool = false, status = nothing)
    T = eltype(track)
    ys = [T[track[c] for c in p] for p in pieces]
    ws = [T[w[c] for c in p] for p in pieces]
    xs = [float.(x[p]) for p in pieces]
    declined = falses(length(pieces))
    if unwrap
        for j in eachindex(ys, ws)
            if phase_unwrap_ambiguity(ys[j]; weights = ws[j]) > _BP_MAX_UNWRAP_AMBIGUITY
                declined[j] = true
                fill!(ys[j], T(NaN))
            else
                ys[j] = unwrap_phase_track(ys[j]; weights = ws[j])
            end
        end
    end
    fill!(track, T(NaN))
    if !any(j -> any(k -> _shape_usable(ys[j][k], ws[j][k]), eachindex(ys[j], ws[j])), eachindex(ys, ws))
        isnothing(status) || (status .= ifelse.(declined, _BP_TRACK_DECLINED, _BP_TRACK_NODATA))
        return prior
    end
    resolved = _estimate_hypers(prior, ys, ws, xs)
    for (j, p) in pairs(pieces)
        fitted = _estimate_map(resolved, ys[j], ws[j], xs[j])
        for (i, c) in pairs(p)
            track[c] = fitted[i]
        end
        isnothing(status) || (status[j] = declined[j] ? _BP_TRACK_DECLINED : _band_track_status(fitted))
    end
    return resolved
end

# Fit every (station, feed) track of `tracks` in place (see `_fit_track!`), each
# channel weighted by its seed precision. `tracks` and `prec` are over
# `(Ant, Feed, Frequency)`; `status`, when given, is over `(Ant, Feed, Frequency)`
# with one entry per piece, and `priors` over `(Ant, Feed)`, receiving each
# track's resolved prior.
function _fit_tracks!(tracks, prec, x, pieces, prior; unwrap::Bool, status = nothing, priors = nothing)
    for a in axes(tracks, Ant), f in axes(tracks, Feed)
        st = isnothing(status) ? nothing : view(status, Ant(a), Feed(f))
        resolved = _fit_track!(
            view(tracks, Ant(a), Feed(f)), view(prec, Ant(a), Feed(f)), x, pieces, prior;
            unwrap, status = st,
        )
        isnothing(priors) || (priors[a, f] = resolved)
    end
    return tracks
end

# One observable's per-piece outcome array over `(Ant, Feed, Frequency, Ti)`:
# `stations`, the two feeds, each frequency segment of `plan` over the extent of
# its channels, and each of its `nts` time segments over the extent of its
# samples.
function _track_status_array(stations, geom::DataGeometry, plan, nts::Integer)
    ax = (
        _station_dim(stations), Feed(1:2),
        Frequency(_frequency_segment_lookup(plan, geom, length(plan.nchan_seg))),
        Ti(_time_segment_lookup(plan, geom, nts)),
    )
    return fill!(zeros(Int8, ax), _BP_TRACK_NODATA)
end

# One observable's resolved prior per (station, feed, time segment), over
# `(Ant, Feed, Ti)` labeled as `_track_status_array`; each slot starts at the
# component's own prior.
function _track_prior_array(stations, geom::DataGeometry, plan, nts::Integer)
    ax = (_station_dim(stations), Feed(1:2), Ti(_time_segment_lookup(plan, geom, nts)))
    return fill!(DimArray(Array{Union{Nothing, AbstractPrior}}(undef, map(length, ax)), ax), plan.prior)
end

# A station block's outcome and prior arrays: its own stations and time segments.
_block_status_array(block, geom::DataGeometry) =
    _track_status_array(geom.stations[block.stations], geom, block.plan, block.plan.shape[4])
_block_prior_array(block, geom::DataGeometry) =
    _track_prior_array(geom.stations[block.stations], geom, block.plan, block.plan.shape[4])

# One entry per station block, keyed `g1, g2, …` as `parameters(sol)` keys a
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
`(Ant, Feed, Frequency, Ti)`: the stations, the two feeds, each frequency
segment of the component over the extent of its channels and each time segment
over the extent of its samples. `phase_priors`/`amp_priors` hold the prior each
(station, feed, time segment) track was fit under, its hyperparameters resolved,
over `(Ant, Feed, Ti)`. Where the model gives stations different segmentations,
[`JointSmoother`](@ref) passes one array per station block, keyed `g1, g2, …`
as [`parameters`](@ref Gustavo.Calibration.parameters) keys the blocks' leaves,
each over its own stations and segments.

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
# source phase on a baseline is its own, so each scan is first rotated by the
# conjugate of its band-averaged phase per (baseline, feed pair); unaligned scans
# would partly cancel. The rotation is flat in frequency, so the bandpass shape
# and the phase steps between spectral windows are kept.
function _pool_scans(results, idx, T)
    rbar = zeros(complex(T), dims(results[first(idx)].rl))
    wbar = zeros(T, dims(results[first(idx)].wl))
    for i in idx
        rl = results[i].rl
        rbar .+= DimensionalData.broadcast_dims(*, rl, _scan_alignment(rl))
        wbar .+= results[i].wl
    end
    return rbar, wbar
end

_scan_alignment(rl) = map(
    z -> iszero(z) ? one(z) : conj(z) / abs(z),
    dropdims(sum(rl; dims = Frequency); dims = Frequency),
)

function solve_bandpass!(sm::PerTrackSmoother, θ, results, setup; gauge::AbstractGauge)
    geom = setup.geom
    T = something(sm.eltype, mapreduce(r -> eltype(r.wl), promote_type, results))
    x = geom.channel_freqs
    channels = [c:c for c in eachindex(x)]
    cells = Frequency(x)
    phase_status = amp_status = phase_priors = amp_priors = nothing
    # The two observables are solved independently here, so each partitions the
    # scans by its own time segmentation — a phase bandpass that breaks mid-track
    # can sit beside an amplitude one held over the whole of it.
    bp_blocks = bandpass_blocks(setup, θ, :phase)
    amp_blocks = bandpass_blocks(setup, θ, :logamp)
    if !isempty(bp_blocks)
        plan = only(bp_blocks).plan
        pieces = _bandpass_pieces(plan)
        groups = time_segment_scans(plan, results)
        phase_status = _track_status_array(geom.stations, geom, plan, length(groups))
        phase_priors = _track_prior_array(geom.stations, geom, plan, length(groups))
        for (ts, idx) in pairs(groups)
            isempty(idx) && continue
            rbar, wbar = _pool_scans(results, idx, T)
            phase, prec = _seed_phase_tracks(rbar, wbar, geom.stations, channels, cells; gauge)
            _fit_tracks!(
                phase, prec, x, pieces, plan.prior;
                unwrap = true, status = view(phase_status, Ti(ts)), priors = view(phase_priors, Ti(ts)),
            )
            _write_phase_bandpass!(θ, plan, phase, ts)
        end
    end
    if !isempty(amp_blocks)
        plan = only(amp_blocks).plan
        pieces = _bandpass_pieces(plan)
        groups = time_segment_scans(plan, results)
        amp_status = _track_status_array(geom.stations, geom, plan, length(groups))
        amp_priors = _track_prior_array(geom.stations, geom, plan, length(groups))
        for (ts, idx) in pairs(groups)
            isempty(idx) && continue
            rbar, wbar = _pool_scans(results, idx, T)
            la, prec = _seed_amp_tracks(rbar, wbar, geom.stations, channels, cells)
            _spike_guard!(la, pieces, _BP_SPIKE_SIGMA)
            _fit_tracks!(
                la, prec, x, pieces, plan.prior;
                unwrap = false, status = view(amp_status, Ti(ts)), priors = view(amp_priors, Ti(ts)),
            )
            _write_amp_bandpass!(θ, plan, la, _BP_MAX_LOGAMP, ts)
        end
    end
    report = bandpass_track_report(phase_status, amp_status; phase_priors, amp_priors)
    _warn_degenerate_bandpass(report)
    return report
end

# ── Joint complex bandpass + per-scan source coherence (ALS) ─────────────────
#
# The per-channel closure solves assume a baseline's source term cancels out of
# the per-channel phase-difference/log-amp-sum closure, which holds only for an
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
# gain ([`_update_station_gains!`](@ref)), one gain per channel, to convergence.
#
# The data are `(Scan, StationPair, FeedPair, Frequency)` arrays over channels.
# The gains are one `(Ant, Feed, Frequency, Ti)` array per phase station block,
# over the block's own stations, the channels and the block's time segments; a
# station's block and position come from `_block_locations`. Station and feed
# indices come from the `StationPair`/`FeedPair` labels (`_cell_nodes`).

# One scan's per-(baseline, pol, segment) coherent residual, written directly
# into `rview`/`wview`, a (StationPair, FeedPair, Frequency) slice of the multi-scan
# accumulator, with no intermediate allocation.
#
# These must not be SNR-gated. The joint solve consumes them as complex
# residuals under inverse-variance weights, and that accumulation is unbiased at
# any SNR: a weak cell contributes its information at its honest weight and
# costs variance, never validity. The closure tier's gate is justified there
# because that tier extracts a per-segment phase, which is meaningless below the
# noise. Applied here it would preferentially delete the cross-hand cells, the
# only rows tying the feed-2 gain block to feed-1 and so the only measurement of
# the relative R–L bandpass, leaving that block at its initialization. Outliers
# belong to a flagging step upstream of the solve, not to a cell gate.
function _reduce_scan_segments!(rview, wview, sc, segs)
    for p in axes(rview, FeedPair), bi in axes(rview, StationPair)
        for (fs, chans) in enumerate(segs)
            rc, wc, _ = _segment_residual(sc.rl, sc.wl, bi, p, chans)
            keep = isfinite(rc) && isfinite(wc) && wc > 0
            rview[bi, p, fs] = keep ? rc : zero(rc)
            wview[bi, p, fs] = keep ? wc : zero(wc)
        end
    end
    return nothing
end

# Every scan's (StationPair, FeedPair, segment) residual, stacked over an added Scan
# axis — not summed across scans (unlike the closure tier's fold), since the per-scan source coherence needs each
# scan's own coherent visibility. Element types follow the scan accumulators'
# own, not a hardcoded precision.
function _reduce_all_scans(scans, segs, cells::Frequency)
    nscan = length(scans)
    rl = first(scans).rl
    nbl, npol = size(rl, StationPair), size(rl, FeedPair)
    nseg = length(segs)
    C = eltype(rl)
    T = real(eltype(first(scans).wl))
    d = (Scan(1:nscan), dims(rl, StationPair), dims(rl, FeedPair), cells)
    rseg = DimensionalData.DimArray(zeros(C, nscan, nbl, npol, nseg), d)
    wseg = DimensionalData.DimArray(zeros(T, nscan, nbl, npol, nseg), d)
    for (si, sc) in enumerate(scans)
        _reduce_scan_segments!(view(rseg, si, :, :, :), view(wseg, si, :, :, :), sc, segs)
    end
    return rseg, wseg
end

# Each station's block and its position on that block's `Ant` axis; `(0, 0)`
# for a station no block covers.
function _block_locations(blocks, nant)
    loc = fill((0, 0), nant)
    for (k, b) in pairs(blocks), (ai, a) in pairs(b.stations)
        loc[a] = (k, ai)
    end
    return loc
end

# One phase block's solve state over `(Ant, Feed, Frequency, Ti)` — its
# stations, the two feeds, the channels and its time segments: the complex gains
# `g`; their unwrapped phase tracks `φ`, carried across sweeps so the prior fit
# never sees a 2π branch cut; the slots a sweep has solved (`touched`) and the
# gauge pins (`pinned`).
function _block_gains(block, geom::DataGeometry, C::Type)
    ax = (
        _station_dim(geom.stations[block.stations]), Feed(1:2), Frequency(geom.channel_freqs),
        Ti(_time_segment_lookup(block.plan, geom)),
    )
    return DimStack((;
        g = ones(C, ax), φ = zeros(real(C), ax),
        touched = zeros(Bool, ax), pinned = zeros(Bool, ax),
    ))
end

# Every (station pair, feed pair) cell touching (station, feed): the cell, the
# station and feed at its other end, and whether (station, feed) is the cell's
# first end — built once, reused by every ALS iteration's gain update.
function _joint_bandpass_touching(cells, nant)
    touching = [Tuple{Int, Int, Int, Int, Bool}[] for _ in 1:nant, _ in 1:2]
    for p in axes(cells, FeedPair), bi in axes(cells, StationPair)
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
    next = 0
    ids = map(gains) do st
        n = size(st.g)
        idk = reshape((next + 1):(next + prod(n)), n)
        next += prod(n)
        idk
    end
    return ids
end

# The phase-gauge graph of a joint bandpass solve: one node per gain slot —
# (station, feed, channel, that station's own time segment) — one edge per
# (scan, station pair, feed pair, channel) correlation, joining its two ends in
# the time segments they are in for that scan.
# Node degree stands in for the row weight the gauge scores elsewhere; the graph
# is built from the correlations that exist, before any per-channel gating, so a
# node's degree is the observation count available to anchor it.
#
# The node set is the gauge condition. An edge names the two gain slots one
# correlation reads, and a shared phase cancels out of `g_a·S·conj(g_b)` only
# when both ends of every internal edge carry it, so the unobservable phases are
# one constant per connected component. No coarser node set states that.
#
# `ids` numbers the slots (`_gain_slot_ids`); `compid[n]` is `0` for a slot no
# correlation touches.
function _joint_bandpass_graph(data, layout, ids)
    (; ends) = data
    (; loc, tseg, fseg) = layout
    nslots = sum(length, ids; init = 0)
    edges = Tuple{Int, Int}[]
    deg = zeros(Int, nslots)
    for si in axes(tseg, 2), p in axes(ends, FeedPair), bi in axes(ends, StationPair)
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
    _joint_bandpass_pins!(gains, data, layout, blocks, gauge) -> gains

Mark one reference slot per connected component of the joint-bandpass graph in
the `pinned` layer of `gains` — mirrors `_solve_observable!`'s pin selection in
`stationize.jl`. The pinned slot's phase is held at zero; its amplitude is
solved like any other slot's.

Only the phase is a gauge freedom. Multiplying the gains of a set of nodes by a
shared `c` sends `g_a·S·conj(g_b)` to `|c|²·g_a·S·conj(g_b)` on every
correlation internal to that set: the phase of `c` cancels between the two
conjugated factors. The sets on which it cancels everywhere are exactly the
components above, so the phase carries one unobservable constant per component
and needs one constraint there — no more. A station-uniform segmentation splits
the graph along the array-wide (frequency segment, time segment) cells and
recovers one pin per cell, which together zero the reference station's whole
track; a model in which one station breaks mid-track keeps the whole track in
one component, because the stations that hold one gain over it bridge the
broken station's two segments, and the relative phase across that break is then
measured rather than gauged away.

Every gain is per channel, so the graph splits per channel. The edges are the
correlations that exist, the same at every channel, so every channel's
component holds the same (station, feed, time segment) nodes with the same
degrees and the gauge pins the same node at each: a track is pinned at every
channel or at none, and a prior never fits a track that is partly held.

The magnitude does not cancel, and `S` is frequency-flat, so it can only absorb
`|c|²` when `|c|` is constant across the band — leaving exactly one free
amplitude parameter overall, which the zero-band-mean gauge in
`_write_joint_bandpass!` removes. Pinning `|g|` as well
would assert the reference antenna has a flat amplitude bandpass, discarding
structure that is identifiable (mean-removing `log|V_ab| = la_a + la_b + ls_ab`
over the band eliminates `ls` and leaves the full-rank signless-Laplacian
system) and biasing every other station through the inconsistency.
"""
function _joint_bandpass_pins!(gains, data, layout, blocks, gauge)
    ids = _gain_slot_ids(gains)
    compid, ncomp, deg = _joint_bandpass_graph(data, layout, ids)
    slot = [(k, I) for (k, idk) in pairs(ids) for I in CartesianIndices(idk)]
    station_of(n) = blocks[slot[n][1]].stations[slot[n][2][1]]
    feed_of(n) = slot[n][2][2]
    for c in 1:ncomp
        k, I = slot[gauge_anchor(gauge, findall(==(c), compid), deg, station_of, feed_of)]
        gains[k].pinned[I] = true
    end
    return gains
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
    for p in axes(r, FeedPair), bi in axes(r, StationPair)
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
                wc = w[Scan(si), StationPair(bi), FeedPair(p), Frequency(cell)]
                wc > 0 || continue
                u = ga[ia, fa, fseg[a, cell], ta] * conj(gb[ib, fb, fseg[b, cell], tb])
                abs2(u) > 0 || continue
                numer += conj(u) * r[Scan(si), StationPair(bi), FeedPair(p), Frequency(cell)]
                denom += wc * abs2(u)
            end
            S[Scan(si), StationPair(bi), FeedPair(p)] = denom > 0 ? numer / denom : zero(eltype(S))
        end
    end
    return nothing
end

# One Gauss-Seidel sweep over every (station, feed, time segment): closed-form
# per-channel solve of its complex gain given the current source coherence `S`
# and every other station's current gain, which is immediately visible to later
# antennas in the same sweep — Gauss-Seidel, not Jacobi — with each observable's
# component prior acting on the resulting track rather than as a post-hoc smooth.
#
# Solving each channel independently is the no-prior case; the prior enters
# where that independence is dropped. Around the unconstrained per-channel
# estimate `ĝ` the residual linearizes as `Σ denom·|ĝ|²·(δlogamp² + δphase²)`, so
# `denom·|ĝ|²` is the Fisher weight both real tracks are fit under. Each sweep
# re-estimates the priors' hyperparameters from its own linearized tracks, so
# iterated to convergence this is MAP estimation under the two priors with
# type-II MAP hyperparameters.
#
# `φ` carries each (station, feed)'s unwrapped phase track across sweeps. It is
# unwrapped once, when `seed` (the first sweep) initializes it, and thereafter
# advanced by wrapped increments about its own current value, so no global
# unwrap is needed inside the loop and the 2π branch cannot flip between
# iterations. `fits[k]` holds block `k`'s phase and amplitude `(; pieces, prior)`.
# The status and prior arrays, when given, hold one array per block. Returns the
# largest relative gain change, for the caller's convergence check.
function _update_station_gains!(
        gains, data, layout; fits, seed::Bool,
        phase_status = nothing, amp_status = nothing, phase_priors = nothing, amp_priors = nothing,
    )
    (; r, w, S) = data
    (; loc, touching, tseg, fseg, present, x) = layout
    C = eltype(first(gains).g)
    T = real(C)
    maxrel = zero(T)
    nfsmax = maximum(st -> size(st, Frequency), gains)
    num = Vector{C}(undef, nfsmax)
    den = Vector{T}(undef, nfsmax)
    ĝ = Vector{C}(undef, nfsmax)
    wf = Vector{T}(undef, nfsmax)
    la = Vector{T}(undef, nfsmax)
    φ̃ = Vector{T}(undef, nfsmax)
    for feed in 1:2, ant in eachindex(loc)
        k, ai = loc[ant]
        iszero(k) && continue
        entries = touching[ant, feed]
        isempty(entries) && continue
        (; g, φ, touched, pinned) = gains[k]
        nfs = size(g, Frequency)
        for ts in present[ant]
            # The gauge fixes this node's phase at every channel or at none. The
            # amplitude is solved like any other node's (see `_joint_bandpass_pins!`).
            pins = view(pinned, ai, feed, :, ts)
            for buf in (num, den, ĝ, wf, la, φ̃)
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
                        wc = w[Scan(si), StationPair(bi), FeedPair(p), Frequency(cell)]
                        wc > 0 || continue
                        to = tseg[o, si]
                        iszero(to) && continue
                        s = S[Scan(si), StationPair(bi), FeedPair(p)]
                        # `V ≈ g_first·S·conj(g_second)`; the second end fits `conj(V)`.
                        coeff = first_end ? s * conj(go[io, fo, so, to]) : conj(go[io, fo, so, to] * s)
                        abs2(coeff) > 0 || continue
                        rc = r[Scan(si), StationPair(bi), FeedPair(p), Frequency(cell)]
                        num[sa] += conj(coeff) * (first_end ? rc : conj(rc))
                        den[sa] += wc * abs2(coeff)
                    end
                end
            end
            for fs in 1:nfs
                gh = den[fs] > 0 ? num[fs] / den[fs] : zero(C)
                ĝ[fs] = gh
                wf[fs] = (den[fs] > 0 && abs(gh) > 0) ? den[fs] * abs2(gh) : zero(T)
                if wf[fs] > 0
                    la[fs] = log(abs(gh))
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
            ast = isnothing(amp_status) ? nothing : view(amp_status[k], ai, feed, :, ts)
            pst = isnothing(phase_status) ? nothing : view(phase_status[k], ai, feed, :, ts)
            fit = fits[k]
            amp_prior = _fit_track!(la, wf, x, fit.amp.pieces, fit.amp.prior; status = ast)
            isnothing(amp_priors) || (amp_priors[k][ai, feed, ts] = amp_prior)
            if any(pins)
                # A pinned track is known rather than fitted: report it as such
                # instead of leaving it at NODATA.
                isnothing(pst) || fill!(pst, _BP_TRACK_SOLVED)
                fill!(φ̃, zero(T))
            else
                phase_prior = _fit_track!(φ̃, wf, x, fit.phase.pieces, fit.phase.prior; unwrap = seed, status = pst)
                isnothing(phase_priors) || (phase_priors[k][ai, feed, ts] = phase_prior)
            end
            la_new, φ_new = la, φ̃
            for fs in 1:nfs
                (isfinite(la_new[fs]) && isfinite(φ_new[fs])) || continue
                gold = g[ai, feed, fs, ts]
                gnew = exp(C(la_new[fs], φ_new[fs]))
                g[ai, feed, fs, ts] = gnew
                φ[ai, feed, fs, ts] = φ_new[fs]
                touched[ai, feed, fs, ts] = true
                maxrel = max(maxrel, abs(gnew - gold) / max(abs(gold), abs(gnew), eps(T)))
            end
        end
    end
    return maxrel
end

# Gauge-fix each (station, feed, time segment) track — zero band-mean
# log-amplitude, circular-mean reference phase, matching the closure tier's
# convention — and write each channel into the slot its own component's leaf
# gives it. Both gauges are over the channels of one time segment, so each of a
# station's time segments is normalized on its own. A (station, feed, segment) that `touched`
# nowhere is left unwritten, holding whatever θ already had, which is unit gain;
# so is a station no block covers.
#
# `gains[k]` holds the stations of phase block `k` and amplitude block `k`, in
# the blocks' own order.
#
# The band means are removed at θ's precision, which may exceed the gains': the
# gauge is a property of θ.
function _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout; max_logamp)
    for (k, (pb, ab)) in enumerate(zip(phase_blocks, amp_blocks))
        T = eltype(pb.θ)
        (; g, touched) = gains[k]
        for (ai, a) in pairs(pb.stations), f in axes(g, Feed), ts in layout.present[a]
            valid = view(touched, ai, f, :, ts)
            any(valid) || continue
            pnode, anode = _feed_node(pb.plan.tying, f), _feed_node(ab.plan.tying, f)
            phase(fs) = T(angle(g[ai, f, fs, ts]))
            logamp(fs) = log(T(abs(g[ai, f, fs, ts])))
            mphase = angle(sum(cis(phase(fs)) for fs in axes(g, Frequency) if valid[fs]))
            mlogamp = sum(logamp(fs) for fs in axes(g, Frequency) if valid[fs]) / count(valid)
            for c in axes(g, Frequency)
                valid[c] || continue
                jp, sp = _channel_slot(pb.plan, c)
                ja, sa = _channel_slot(ab.plan, c)
                iszero(pnode) || (pb.θ[jp, pnode, sp, ts, ai] = rem2pi(phase(c) - mphase, RoundNearest))
                la = logamp(c) - mlogamp
                iszero(anode) || (ab.θ[ja, anode, sa, ts, ai] = abs(la) > max_logamp ? zero(la) : la)
            end
        end
    end
    return nothing
end

"""
    JointSmoother(; max_iterations = 8, tolerance = 1.0e-6)

Fit the station bandpass against the actual complex visibilities, with each
component's prior acting inside the solve rather than as a fit applied to it
afterwards. The alternating complex-gain / per-scan source-coherence scheme is
[`solve_joint_bandpass!`](@ref)'s; a prior changes the per-(station, feed) gain
update, which fits the whole track under the prior instead of solving each
channel on its own, and re-estimates the prior's hyperparameters every sweep.
Iterated to convergence that is MAP estimation under the two priors.

Because the per-scan source term absorbs a baseline's own structure, this suits a
resolved or polarized calibrator, where [`PerTrackSmoother`](@ref)'s closure
assumption would bias the bandpass. It solves one complex gain per
(station, feed, channel), so it requires a model with both a phase and a
log-amplitude component on one time segmentation per station; their frequency
segments, and so their priors' breaks, may differ.
"""
struct JointSmoother <: AbstractBandpassSmoother
    max_iterations::Int
    tolerance::Float64
end
JointSmoother(; max_iterations::Integer = 8, tolerance::Real = 1.0e-6) =
    JointSmoother(Int(max_iterations), Float64(tolerance))

can_fit(::JointSmoother, tc, geom) = _fits_bandpass_track(tc)

function validate_model(::JointSmoother, model)
    validate_bandpass_groups(model)
    ph = Calibration._flatten_components(model.phase)
    la = Calibration._flatten_components(model.logamp)
    length(ph) == 1 && length(la) == 1 || throw(
        ArgumentError(
            "JointSmoother requires both a phase and a logamp component — it solves one " *
                "COMPLEX gain per (station, feed, channel), not independent phase/log-amp " *
                "tracks. Use `smoother = PerTrackSmoother()` for a phase-only or " *
                "amplitude-only bandpass.",
        ),
    )
    only(ph).Ti == only(la).Ti || throw(
        ArgumentError(
            "JointSmoother requires the phase and logamp components to share one time " *
                "segmentation — one complex gain per (station, feed, channel) is solved over " *
                "one stretch of time, not two. Got $(repr(only(ph).Ti)) (phase) vs " *
                "$(repr(only(la).Ti)) (logamp). Use `smoother = PerTrackSmoother()` to give " *
                "the two observables different time resolutions.",
        ),
    )
    return nothing
end

function solve_bandpass!(sm::JointSmoother, θ, results, setup; gauge::AbstractGauge)
    geom = setup.geom
    phase_blocks = bandpass_blocks(setup, θ, :phase)
    amp_blocks = bandpass_blocks(setup, θ, :logamp)
    # One complex gain per (station, feed, segment) means one time segmentation
    # for both observables — `validate_model` holds each station's two plans to
    # it — so the phase side's table is the whole solve's.
    tseg = _station_time_segments(phase_blocks, results, length(geom.stations))
    phase_status = [_block_status_array(b, geom) for b in phase_blocks]
    amp_status = [_block_status_array(b, geom) for b in amp_blocks]
    phase_priors = [_block_prior_array(b, geom) for b in phase_blocks]
    amp_priors = [_block_prior_array(b, geom) for b in amp_blocks]
    for idx in _joint_scan_groups(tseg)
        solve_joint_bandpass!(
            θ, results[idx], geom, phase_blocks, amp_blocks;
            gauge, max_iterations = sm.max_iterations, tolerance = sm.tolerance,
            tseg = view(tseg, :, idx), phase_status, amp_status, phase_priors, amp_priors,
        )
    end
    report = bandpass_track_report(
        _block_leaves(phase_status), _block_leaves(amp_status);
        phase_priors = _block_leaves(phase_priors), amp_priors = _block_leaves(amp_priors),
    )
    _warn_degenerate_bandpass(report)
    return report
end

"""
    solve_joint_bandpass!(θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
                          gauge = PinAntenna(1), max_iterations = 8, tolerance = 1.0e-6,
                          max_logamp = log(10.0), phase_status = nothing,
                          amp_status = nothing, phase_priors = nothing,
                          amp_priors = nothing, tseg = nothing)

Jointly solve the per-(station, feed, channel) complex bandpass gain and a
per-scan, per-station-pair, per-feed-pair constant source coherence (see the
module comment above `_reduce_scan_segments!` for the model and the alternating
scheme), then gauge-fix each (station, feed, time segment) track and write the
result into the station blocks of the two observables
(`_write_joint_bandpass!`).

`phase_blocks` and `amp_blocks` are [`bandpass_blocks`](@ref)`(setup, θ, :phase)`
and `(…, :logamp)` — station blocks over the SAME `θ` this call is handed, since
each block's `θ` is a view into it. A station-uniform model gives one block per
observable spanning every station; where the model differs across stations, each
block carries its own stations, feed tying, segments and prior, and a station
no block covers is left at unit gain. Each track is fit under its own block's
prior, one frequency segment of that block at a time, with the prior's
hyperparameters re-estimated every sweep. A station's two components must hold
the same stations on the same time segmentation, which
`validate_model(::JointSmoother, model)` enforces at model-compile time; the
throw here guards direct callers.

`geom` supplies the stations the `StationPair` labels name (θ's stations, in
θ's order) and the channel frequencies the priors are fit along.

Scaling a set of stations by one phase leaves every visibility internal to that
set unchanged, so the gauge fixes one constant per such set. The sets are the
connected components of the (station, feed, channel, time segment) graph the
correlations span, and one node of each is pinned
([`_joint_bandpass_pins!`](@ref)). Each channel is its own component, and the
gauge pins the same node at every channel, so the pinned track is held whole.

`scans` is the per-scan `(rl, wl)` accumulator pairs from
`accumulate_bandpass` — not summed across scans, since the source term
needs each scan's own coherent visibility.

`tseg`, when given, is the `(station, scan)` time-segment table
([`_station_time_segments`](@ref)): station `a`'s gain is solved separately for
each distinct `tseg[a, :]` value, in that station's own segment numbering, and a
station whose entry is `0` is left out of the solve. Every station shares one
segment when it is omitted. `scans` must hold the scans `tseg`'s columns
describe, in the same order. The phase gauge acts on the graph these scans
span, pinning one node per connected component, so a break at one station is
measured against the stations that hold one gain across it rather than gauged
away. Solved phases are comparable only WITHIN a component: where that graph
splits — disjoint sub-arrays, or every station segmented at the same epoch —
each piece carries its own arbitrary constant, and a per-station change read
across the split is that constant plus the change.

Convergence is judged on the largest relative per-iteration gain change over
every (station, feed, channel) node solved here, not a tracked χ² (which would
need a per-channel power accumulator this stage does not keep). The scans handed
to one call are a connected piece of the coupling graph, so this is one
criterion over one coupled problem: nodes that share no data are solved by
separate calls rather than being averaged into a common tolerance.

`phase_status`/`amp_status`, when given, hold one `(Ant, Feed, Frequency, Ti)`
array per block of `phase_blocks`/`amp_blocks` ([`bandpass_track_report`](@ref)),
receiving each piece's `_BP_TRACK_*` outcome code from the final sweep, and
`phase_priors`/`amp_priors` one `(Ant, Feed, Ti)` array per block receiving each
track's resolved prior; a call solving part of a track writes only its own time
segments.
"""
function solve_joint_bandpass!(
        θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
        gauge::AbstractGauge = PinAntenna(1), max_iterations::Integer = 8, tolerance::Real = 1.0e-6,
        max_logamp::Real = _BP_MAX_LOGAMP,
        phase_status = nothing, amp_status = nothing,
        phase_priors = nothing, amp_priors = nothing,
        tseg::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
    )
    isempty(scans) && return θ
    nant = length(geom.stations)
    # Both observables carry one complex gain per (station, feed, channel), so
    # their blocks must hold the same stations on the same time segmentation
    # (`validate_model(::JointSmoother, model)` holds each station's tree to it;
    # the throw guards direct callers).
    _same_stations(pb, ab) = pb.stations == ab.stations && pb.plan.tseg_id == ab.plan.tseg_id
    length(amp_blocks) == length(phase_blocks) && all(splat(_same_stations), zip(phase_blocks, amp_blocks)) || throw(
        ArgumentError(
            "solve_joint_bandpass!: the phase and logamp components give the stations " *
                "different station blocks or time segmentations — the solve carries one " *
                "COMPLEX gain per (station, feed, channel), not independent phase/log-amp tracks.",
        ),
    )

    # Every gain is per channel, so the data are reduced per channel and each
    # station a block covers reads channel `c` at slot `c`.
    nchan = length(geom.channel_freqs)
    loc = _block_locations(phase_blocks, nant)
    fseg = [iszero(first(loc[a])) ? 0 : c for a in 1:nant, c in 1:nchan]
    r, w = _reduce_all_scans(scans, [c:c for c in 1:nchan], Frequency(geom.channel_freqs))
    ends = _cell_nodes(r, geom.stations, PerFeed())
    S = zeros(eltype(r), (Scan(axes(r, Scan)), dims(r, StationPair), dims(r, FeedPair)))
    data = DimStack((; r, w, S, ends))

    # The time segments each station is actually solved for here, in its own
    # segmentation's numbering — the numbering of its block's `Ti` axis.
    tsg = isnothing(tseg) ? ones(Int, nant, length(scans)) : tseg
    layout = (;
        loc, tseg = tsg, fseg, x = geom.channel_freqs,
        present = [sort!(filter!(!iszero, unique(view(tsg, a, :)))) for a in axes(tsg, 1)],
        touching = _joint_bandpass_touching(ends, nant),
    )
    fits = [
        (;
                phase = (; pieces = _bandpass_pieces(pb.plan), prior = pb.plan.prior),
                amp = (; pieces = _bandpass_pieces(ab.plan), prior = ab.plan.prior),
            ) for (pb, ab) in zip(phase_blocks, amp_blocks)
    ]
    gains = [_block_gains(b, geom, eltype(r)) for b in phase_blocks]
    _joint_bandpass_pins!(gains, data, layout, phase_blocks, gauge)

    _update_source_coherence!(data, gains, layout)
    for iter in 1:max_iterations
        maxrel = _update_station_gains!(
            gains, data, layout;
            fits, seed = iter == 1, phase_status, amp_status, phase_priors, amp_priors,
        )
        _update_source_coherence!(data, gains, layout)
        maxrel < tolerance && break
    end

    _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout; max_logamp)
    return θ
end
