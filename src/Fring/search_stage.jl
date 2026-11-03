# ── Fringe fitting: the default fringe model + the matched filter ──────────
#
# - `default_fringe_model` — what a fringe closure solves by default: a
#   `(; phase)` tree of named components, each declaring its own feed scope
#   through its tying.
# - What the matched filter (the per-baseline search and the station solve)
#   can fit and requires of a model.
# - The closure's stage-B component filter, steering, and the flag table
#   recorded on the solution.

"""
    default_fringe_model(; rel_time = PerScan()) -> GainModel

The default model of a [`FringeClosure`](@ref Gustavo.FringeClosure) — the
standard VLBI fringe model, phase components only (each key names its
component; the order is the parameter order):

1. `mbd` — per-scan wideband delay, feed-common.
2. `rel_delay` — inter-feed delay offsets, `Ti = rel_time`,
   `Feed = ExceptFeed(1)`: each other feed's instrumental group-delay offset
   from feed 1. With one feed per station it has no parameters.
3. `rate` — per-scan rate, feed-common. There is no feed-specific rate: the
   inter-feed rate is negligible (EHT-HOPS convention). A genuine offset is
   added with
   `merge(default_fringe_model(); phase = (; rel_rate = GainComponent(Rate(); Ti = PerScan(), Feed = ExceptFeed(1))))`.

`rel_time` is the inter-feed delay's time segmentation: `PerScan()` (the
default) fits an offset per scan, so its scan-to-scan scatter is an
instrument-stability diagnostic and no column couples scans; `GlobalTime()`
fits one offset per station for the whole track (the EHT-HOPS / rPICARD
assumption) — bright scans pin it and weak scans inherit it, at the cost of
coupling every scan into one system, solved after every scan is searched.

The closure fits no constant phase. A station's per-scan phase trades against
the source's phase, so the data do not determine it without a source model;
it stays in the data for imaging or self-calibration. The later solves do not
need it: [`Bandpass`](@ref Gustavo.Bandpass) aligns each scan's band-averaged
phase per baseline and feed pair, and [`AdhocPhase`](@ref Gustavo.AdhocPhase)
removes each track's scan mean. The inter-feed delay is fit because a delay
decoheres across the band, so leaving it in costs signal.

Add or replace a component with [`merge`](@ref Base.merge(::GainModel)).
"""
default_fringe_model(; rel_time::AbstractTimeSegmentation = PerScan()) = GainModel(
    phase = (
        mbd = GainComponent(Delay(); Ti = PerScan(), Feed = SharedFeeds()),
        rel_delay = GainComponent(Delay(); Ti = rel_time, Feed = ExceptFeed(1)),
        rate = GainComponent(Rate(); Ti = PerScan(), Feed = SharedFeeds()),
    ),
)

# ── Model validation ─────────────────────────────────────────────────────────

# Reject compiled component sets a findfirst router cannot address uniquely,
# and exact duplicates (indistinguishable θ blocks are degenerate columns).
function _validate_fringe_components(comps::Tuple)
    for i in eachindex(comps), j in (i + 1):length(comps)
        _same_component_signature(comps[i], comps[j]) && throw(
            ArgumentError(
                "the fringe model has two identical components " *
                    "($(component_label(comps[i]))) — the routers and solvers cannot " *
                    "distinguish their θ blocks. Remove the duplicate.",
            ),
        )
    end
    _at_most_one(_is_perscan_delay, comps, "the per-scan feed-common delay signature")
    return comps
end

function _at_most_one(pred, comps::Tuple, what::String)
    n = count(pred, comps)
    n <= 1 || throw(
        ArgumentError(
            "the fringe model has $n components matching $what — the plan router " *
                "routes to the first and the rest would never be fit. Remove the " *
                "duplicate(s).",
        ),
    )
    return nothing
end

_same_component_signature(a::GainComponent, b::GainComponent) =
    typeof(a.term) === typeof(b.term) &&
    a.Ti == b.Ti && a.Frequency == b.Frequency && a.Feed == b.Feed

# What the matched filter does with a compiled component's θ block:
#
#   :delay / :rate          — stage B's station system of that kind writes it,
#                             from the per-baseline search observable of the
#                             same name.
#   nothing                 — the matched filter does not touch it.
#
# This is the matched filter's vocabulary, hence private: a global
# least-squares fringe fitter fits θ through `evaluate_gains` directly.
#
# The stage-B kinds cover the single-parameter, band-wide terms whose observable
# the search measures. Everything else must map to `nothing`, so that it is
# unfittable by the matched filter rather than approximated: `_solve_kind_cols!`
# writes one θ column per (station, feed, time) node — the block's first
# parameter at the first frequency segment — so a multi-parameter term such as a
# polynomial would have its trailing parameters left at zero. A constant phase,
# a `Dispersion` and a frequency-segmented term fall through to `nothing` on the
# checks below, so `can_fit` rejects them.
#
# The time axis is not decided here: whether a segmentation leaves a scan with
# more than one θ column depends on the scan lengths, so `can_fit` answers it
# against the geometry.
function matched_kind(tc)
    # The per-baseline search measures one delay and rate across the whole
    # band, so only a component spanning it can receive that estimate.
    tc.Frequency isa GlobalFrequency || return nothing
    term = tc.term
    term isa Delay && return :delay
    term isa Rate && return :rate
    return nothing
end

# Stage-B engine components `(plan, kind)` — the delay and rate terms the
# search + stationization solve, as declared by `matched_kind`, over every
# phase component of `model`, the fringe closure's own model, so no other
# solve's component can collide with a stage-B signature.
function fringe_stage_components(model, layout)
    comps = Tuple{ComponentPlan, Symbol}[]
    for (i, tc) in enumerate(phase_components(model))
        kind = matched_kind(tc)
        kind in (:delay, :rate) || continue
        push!(comps, (layout.plans[i], kind))
    end
    return comps
end

# ── The matched filter's capability ──────────────────────────────────────────

# Whether `seg` resolves more than one time segment inside any one scan.
function splits_a_scan(seg, geom)
    scans, _ = time_segment_ids(PerScan(), geom)
    ids, _ = time_segment_ids(seg, geom)
    first_of = Dict{Int, Int}()
    for i in eachindex(ids, scans)
        get!(first_of, scans[i], ids[i]) == ids[i] || return true
    end
    return false
end

# What the matched filter requires of a `(; phase, logamp)` model tree. Each
# absent item costs the closure its own output silently rather than crashing:
# `solve_antenna_systems!` skips a kind with no components, dropping every
# scan's search estimate for that observable.
function validate_fringe_model(model)
    name = "FringeClosure"
    isempty(_flatten_components(model.logamp)) || throw(
        ArgumentError(
            "$name fits phase components only; the fringe model's `logamp` group " *
                "must be empty.",
        ),
    )
    comps = _validate_fringe_components(_flatten_components(model.phase))
    kinds = map(matched_kind, comps)
    for (kind, what) in (
            (:delay, "a delay component (the per-baseline delay search has nowhere to go)"),
            (:rate, "a rate component (the per-baseline rate search has nowhere to go)"),
        )
        kind in kinds || throw(
            ArgumentError(
                "$name requires $what. Add one to the fringe model's `phase` " *
                    "tree — see `default_fringe_model`.",
            ),
        )
    end
    return nothing
end

# ── Closure flags and steering ───────────────────────────────────────────────

# EHT-HOPS-style station flags: a station's feed that participates in a scan
# (has products there) but is left unconstrained by the surviving stage-B rows
# keeps identity gains — record it as (station name, feed, geometry scan id) so
# `calibrate` flags its products instead of passing raw phases through as if
# they had been corrected. `covered` holds (station name, feed, position in
# `dets`).
function unconstrained_flags(dets, covered, geom::DataGeometry)
    Base.require_one_based_indexing(dets)
    flags = Tuple{String, Int, Int}[]
    for gi in eachindex(dets)
        scanid = geom.scan_of_time[_scan_ti(dets[gi])]
        feeds = _scan_feeds(dets[gi])
        present = unique(
            st for (a, b) in _scan_pairs(dets[gi]) if a != b
                for (fa, fb) in feeds for st in ((a, fa), (b, fb))
        )
        for (name, f) in present
            (name, f, gi) in covered || push!(flags, (name, f, scanid))
        end
    end
    return flags
end

"""
    scan_antenna_terms(model, layout, θ, ti, antennas) -> (delay, rate)

Per-`(antenna, feed)` group delay (s) and fringe rate (Hz) at time index `ti`,
summed over the stage-B components of `model`, as `DimArray`s over
`AntennaName(antennas) × Feed`; `antennas` names the layout's stations in order. This is the decode
[`fringe_antenna_solutions`](@ref) reports, for one scan and without a finished
solution. Entries are `NaN` where no component constrains that node.
"""
function scan_antenna_terms(model, layout, θ, ti::Integer, antennas)
    length(antennas) == layout.nant || throw(
        DimensionMismatch("the layout covers $(layout.nant) antennas but $(length(antennas)) are named")
    )
    comps = fringe_stage_components(model, layout)
    delay = fill(NaN, AntennaName(collect(antennas)), Feed(1:layout.nfeed))
    rate = similar(delay)
    fill!(rate, NaN)
    for a in axes(delay, 1), f in axes(delay, 2)
        d = 0.0; r = 0.0; hd = false; hr = false
        for (plan, kind) in comps
            node = _feed_node(plan.tying, f)        # fseg 1: stage-B terms are GlobalFrequency
            node == 0 && continue
            v = _component_leaf(plan, θ)[1, node, 1, plan.tseg_id[ti], a]
            if kind === :delay
                d += v; hd = true
            elseif kind === :rate
                r += v; hr = true
            end
        end
        hd && (delay[a, f] = d)
        hr && (rate[a, f] = r)
    end
    return (delay, rate)
end

"""
    steer_scan(group, geom, res, f0, t0, sta_delay, sta_rate; cells, executor = SerialScheduler())

Re-measure every cell of `res`, the `search_scan` result for scan group
`group`, at the delay and rate the station solution predicts for it
(`τ_{a,fa} − τ_{b,fb}`, and the same difference in rate) rather than at a
blind search peak. `sta_delay` and `sta_rate` are `DimArray`s over
`AntennaName × Feed`, as [`scan_antenna_terms`](@ref) returns them; the result
is labeled like `res`.

This recovers a fringe too weak to survive a blind search: the trial count
collapses from the search plane's ~1e4 cells to the `cells` covering the
prediction's uncertainty, so the SNR needed to clear a given false-alarm
probability drops by roughly 1.5σ.

The steered SNR is comparable to the blind one because both are formed
against the same noise (`σ = |D_blind|/snr` at the blind peak). Cells with
no usable data, or whose two stations are not both solved this scan, come
back `NaN`.

The cells are measured on `executor`, each into its own entries of the result.
"""
steer_scan(group::XRadio.ProcessingSet, geom::DataGeometry, res, f0::Real, t0::Real, sta_delay, sta_rate; kw...) =
    steer_scan(_GroupCells(group, geom), res, f0, t0, sta_delay, sta_rate; kw...)

function steer_scan(
        gc::_GroupCells, res, f0::Real, t0::Real,
        sta_delay::DimensionalData.AbstractDimArray, sta_rate::DimensionalData.AbstractDimArray;
        cells::Real = 9.0, executor = SerialScheduler(),
    )
    lookup(res, AntennaPair) == gc.antenna_pairs && lookup(res, FeedPair) == gc.feeds || throw(
        DimensionMismatch("the detections do not label the scan group's cells"),
    )
    C = eltype(first(first(gc.layers)))
    T = real(C)
    nan() = fill(T(NaN), DimensionalData.dims(res))
    sdelay, srate, sphase, samp, ssnr, spfa = nan(), nan(), nan(), nan(), nan(), nan()
    workspace = TaskLocalValue{FringeWorkspace{C, T}}(() -> FringeWorkspace(C))
    antenna(x, a, f) = x[AntennaName(At(a)), Feed(At(f))]
    # Each cell writes only its own entries of the result arrays.
    tforeach(vec(CartesianIndices((eachindex(gc.antenna_pairs), eachindex(gc.feeds)))); scheduler = executor) do I
        j, q = Tuple(I)
        cell = (AntennaPair(j), FeedPair(q))
        res[:valid][cell] || return
        snr0 = res[:snr][cell]
        snr0 > 0 || return
        a, b = gc.antenna_pairs[j]
        fa, fb = gc.feeds[q]
        dpred = antenna(sta_delay, a, fa) - antenna(sta_delay, b, fb)
        rpred = antenna(sta_rate, a, fa) - antenna(sta_rate, b, fb)
        (isfinite(dpred) && isfinite(rpred)) || return
        X, Wm = _gather_cell!(workspace[], gc, j, q)
        # σ of the blind pass, recovered from its own reported SNR.
        σ = abs(_exact_matched_filter(X, gc.freqs, gc.times, f0, t0, res[:delay][cell], res[:rate][cell])) / snr0
        σ > 0 || return
        D = _exact_matched_filter(X, gc.freqs, gc.times, f0, t0, dpred, rpred)
        Wsum = sum(Wm)
        sdelay[cell] = dpred
        srate[cell] = rpred
        sphase[cell] = angle(D)
        samp[cell] = Wsum > 0 ? abs(D) / Wsum : NaN
        ssnr[cell] = abs(D) / σ
        spfa[cell] = fringe_pfa(ssnr[cell], cells)
    end
    return (delay = sdelay, rate = srate, phase = sphase, amp = samp, snr = ssnr, pfa = spfa)
end

# The flag block for the solution `info` (plain parallel vectors): the
# stage-B-unconstrained (station name, feed, scan) triples.
function flag_table(antenna_flags)
    return (;
        flagged_ant = String[f[1] for f in antenna_flags],
        flagged_feed = Int[f[2] for f in antenna_flags],
        flagged_scan = Int[f[3] for f in antenna_flags],
    )
end
