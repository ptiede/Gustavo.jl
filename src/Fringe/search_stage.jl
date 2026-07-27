# ── FringeFit stage: typed model selection + the matched-filter estimator ─────
#
# The composable pipeline's fringe stage in three parts:
#
# - `FringeModel` — WHAT is solved: the gauge pin (`ref_ant`), the delay/rate
#   segmentations, the typed R–L selection (`CrossFeed`), and the dispersion/SBD
#   toggles. Purely declarative; `fringe_phase_components` compiles it into the
#   gain-model components (ordering identical to the legacy `_fringe_model`, so
#   the compiled layout matches the frozen oracle's fringe block exactly).
# - `MatchedFilter <: AbstractFringeEstimator` — HOW it is estimated: today's
#   stage A (per-baseline delay/rate matched-filter search + closure-screened
#   station WLS). The search and `Stationization` live HERE, not on the step —
#   an alternative estimator (e.g. a Schwab–Cotton-style global LS) plugs in
#   with no vestigial search/stationization options.
# - The stage machinery the runner drives through the streaming layer:
#   residual cubes for `rounds > 1`, R–L fit-on-subset masking, the stage-B
#   component filter, and the detection/flag tables recorded on the solution.

"""
    CrossFeed(; delay = GlobalTime(), rate = nothing, fit_on = AllScans())

Typed R–L (feed-2 − feed-1) model selection for [`FringeModel`](@ref):

- `delay` — the per-station R–L delay's time basis: `GlobalTime()` (one
  instrumental offset per station for the whole track — bright polarized scans
  pin it, weak scans inherit it; the robust default) or `PerScan()` (fit per
  scan — its scatter is an instrument-stability diagnostic, at the cost that a
  scan with no cross-hand detection leaves feed 2 untied).
- `rate` — `nothing` keeps the R–L rate tied ≡ 0 (the EHT-HOPS convention: a
  per-feed rate would lever hours of `t − t0` into arbitrary R–L jumps).
  Passing a segmentation (`GlobalTime()`) OPTS INTO a solvable per-feed rate;
  cross-hand rows are then included in the rate solve.
- `fit_on` — which scans' CROSS-HAND rows feed the solve (fit-on-subset /
  apply-everywhere: fit the R–L offset from a few bright polarized scans, apply
  it track-wide). Parallel-hand rows are unaffected.
"""
struct CrossFeed{TD <: AbstractTimeSegmentation, TR <: Union{Nothing, AbstractTimeSegmentation}, S <: AbstractScanSelection}
    delay::TD
    rate::TR
    fit_on::S
    function CrossFeed(delay::TD, rate::TR, fit_on::S) where {TD, TR, S}
        delay isa Union{GlobalTime, PerScan} ||
            error("CrossFeed: delay must be GlobalTime() or PerScan() (got $(typeof(delay)))")
        rate isa Union{Nothing, GlobalTime, PerScan} ||
            error("CrossFeed: rate must be nothing (tied ≡ 0), GlobalTime() or PerScan() (got $(typeof(rate)))")
        return new{TD, TR, S}(delay, rate, fit_on)
    end
end
CrossFeed(; delay = GlobalTime(), rate = nothing, fit_on = AllScans()) =
    CrossFeed(delay, rate, fit_on)

"""
    FringeModel(; ref_ant = 1, delay = PerScan(), rate = PerScan(), cross_feed = CrossFeed(),
                sbd = :auto)

WHAT the fringe stage solves of the INSTRUMENT — the model specification of a
`FringeFit` step. Propagation through the ionosphere is a separate model, given
to the step alongside this one ([`DispersionModel`](@ref)).

- `ref_ant` — the gauge pin: a 1-based antenna index or a station code
  (`"PT"`). Part of the MODEL (it changes what is solved), not of the
  execution configuration.
- `delay`, `rate` — feed-common segmentations (default `PerScan()`; the
  matched-filter estimator currently supports only `PerScan()`).
- `cross_feed` — the typed feed-2 − feed-1 selection ([`CrossFeed`](@ref)).
- `sbd` — per-scan per-band-group single-band delay (fourfit SBD): `:auto` (on
  when the frequency axis has ≥ 2 band groups), `true`, `false`. Instrumental,
  not propagation: a per-band-group delay offset, so it belongs here.
"""
Base.@kwdef struct FringeModel
    ref_ant::Union{Integer, AbstractString, Symbol} = 1
    delay::AbstractTimeSegmentation = PerScan()
    rate::AbstractTimeSegmentation = PerScan()
    cross_feed::CrossFeed = CrossFeed()
    sbd::Union{Bool, Symbol} = :auto
end

"""
    DispersionModel(; require_band_separation = true, tie_colocated = true)

The differential-ionosphere (dTEC) term: a per-scan, feed-common phase ∝ 1/ν.
Give it to a [`FringeFit`](@ref) alongside its [`FringeModel`](@ref), or pass
`dispersion = nothing` for a fit that models no ionosphere at all.

- `require_band_separation` — solve the term only when the band layout can
  actually separate 1/ν from a linear delay: several sub-bands over a wide
  fractional bandwidth (VGOS 3–10.7 GHz qualifies; a single contiguous band
  cannot constrain the curvature and the term would just soak up delay). Set
  `false` to solve it regardless.
- `tie_colocated` — tie co-located stations (< 1 km apart) to one dTEC. They
  see the same ionosphere, so a differential TEC between them is pure solve
  error.

Estimating dispersion is NOT separable from estimating delay: over a finite
band the two are near-degenerate, so the fringe estimator fits Δτ and dTEC
jointly. The separation here is of the model, not of the solve.
"""
Base.@kwdef struct DispersionModel
    require_band_separation::Bool = true
    tie_colocated::Bool = true
end

"""
    MatchedFilter(; search = FringeSearch(), closure = Stationization(), rounds = 1)

HOW the fringe stage is estimated (an [`AbstractFringeEstimator`](@ref)):
today's stage A — a per-baseline delay/rate matched-filter `search` on every
scan group, then ONE global closure-screened station WLS (`closure`) that ties
the feeds and solves any track-global columns. `rounds` re-runs the search on
the residual (each round divides out the current solution and accumulates the
leftover) — an iteration knob of THIS estimator.

When `search.pfa_max` is finite and `closure` is left at its default, the
stationization's fixed SNR floor is dropped (`snr_min = 0`): the PFA gate IS
the acceptance decision. A custom `closure` is used as given.
"""
Base.@kwdef struct MatchedFilter <: AbstractFringeEstimator
    search::FringeSearch = FringeSearch()
    closure::Stationization = Stationization()
    rounds::Int = 1
end

# ── Model compilation ─────────────────────────────────────────────────────────

"""
    fringe_phase_components(fm::FringeModel, dm, geom::DataGeometry) -> Tuple

The fringe stage's gain-model phase components compiled from the instrument
model `fm` and the propagation model `dm` (a [`DispersionModel`](@ref), or
`nothing` for no ionosphere term), in the LEGACY `_fringe_model` order (per-scan
constant, global R–L constant, delay, R–L delay, rate, [opt-in R–L rate,]
[dispersion,] [SBD delay + constant]) — so the compiled layout's fringe block
matches the frozen oracle's exactly whenever the optional R–L rate is off.
"""
function fringe_phase_components(fm::FringeModel, dm, geom::DataGeometry)
    fm.delay isa PerScan ||
        error("FringeModel: the matched-filter stage currently supports delay = PerScan() only (got $(typeof(fm.delay)))")
    fm.rate isa PerScan ||
        error("FringeModel: the matched-filter stage currently supports rate = PerScan() only (got $(typeof(fm.rate)))")
    fm.sbd in (:auto, true, false) ||
        error("FringeModel: sbd must be :auto, true or false (got $(fm.sbd))")
    rlrate = fm.cross_feed.rate === nothing ? () :
        (TiedComponent(GainComponent(Rate(), fm.cross_feed.rate, GlobalFrequency()), FeedComponent(2)),)
    disp = _dispersion_enabled(dm, geom) ?
        (TiedComponent(GainComponent(Dispersion(), PerScan(), GlobalFrequency()), SharedFeeds()),) : ()
    bands = _sbd_bands(fm.sbd, geom)
    sbd = bands === nothing ? () : (
        TiedComponent(GainComponent(Delay(), PerScan(), FrequencyBands(bands)), SharedFeeds()),
        TiedComponent(GainComponent(ConstantTerm(), PerScan(), FrequencyBands(bands)), SharedFeeds()),
    )
    return (
        TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), SharedFeeds()),
        TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), GlobalFrequency()), FeedComponent(2)),
        TiedComponent(GainComponent(Delay(), fm.delay, GlobalFrequency()), SharedFeeds()),
        TiedComponent(GainComponent(Delay(), fm.cross_feed.delay, GlobalFrequency()), FeedComponent(2)),
        TiedComponent(GainComponent(Rate(), fm.rate, GlobalFrequency()), SharedFeeds()),
        rlrate...,
        disp...,
        sbd...,
    )
end

# Stage-B engine components `(plan, kind)` — the delay/rate/const terms the
# search + stationization solve. Structural routing: excludes the
# per-integration adhoc, the per-channel
# bandpass, the dispersion term, and the SBD band-group terms.
function fringe_stage_components(model, layout)
    comps = Tuple{ComponentPlan, Symbol}[]
    for (i, tc) in enumerate(model.phase)
        tc.component.time isa PerIntegration && continue
        tc.component.term isa PerChannel && continue
        tc.component.term isa Dispersion && continue
        tc.component.freq isa FrequencyBands && continue
        term = tc.component.term
        kind = term isa Delay ? :delay : term isa Rate ? :rate : :phase
        push!(comps, (layout.plans[i], kind))
    end
    return comps
end

# The effective Stationization for a MatchedFilter run: with the PFA gate
# active and the DEFAULT closure, the search's `valid` IS the acceptance
# decision — drop the fixed SNR floor (the legacy solver's rule). A customized
# closure is honored as given. An opt-in R–L rate needs the cross-hand rows in
# the rate system, so `cross_hand_rate` is forced on.
function resolve_closure(est::MatchedFilter, rl_rate_on::Bool)
    c = est.closure
    if isfinite(est.search.pfa_max) && c == Stationization()
        c = Stationization(snr_min = 0.0)
    end
    if rl_rate_on && !c.cross_hand_rate
        c = Stationization(c.snr_min, true, c.phase_rewrap_iters, c.reject_sigma, c.reject_iters)
    end
    return c
end

# ── Stage machinery over the streaming layer ─────────────────────────────────

"""
    residual_vis(ev::GainEvaluator, θ, v::ScanDataView) -> Array{<:Complex, 4}

The scan's residual cube: `v.vis` divided by the current θ gains evaluated on
the view's (global chan, global ti) window — the search input for residual
re-search rounds. Cells with a degenerate gain become NaN (excluded by the
search's weight handling).
"""
function residual_vis(ev::GainEvaluator, θ::AbstractVector, v::ScanDataView)
    g = evaluate_gains(ev, θ, v.chan_idx, v.ti_idx)      # (nchan, nti, nant, 2)
    V = v.vis
    bl_pairs = v.bl_pairs
    pols = v.pol_products
    nchan, nti, nbl, npol = size(V)
    out = similar(V)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            for tt in 1:nti, c in 1:nchan
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                denom = ga * conj(gb)
                if abs(ga) < 1.0e-12 || abs(gb) < 1.0e-12 || !isfinite(denom)
                    out[c, tt, bi, p] = ComplexF64(NaN, NaN)
                else
                    out[c, tt, bi, p] = V[c, tt, bi, p] / denom
                end
            end
        end
    end
    return out
end

# R–L fit-on-subset: invalidate the CROSS-HAND detections of every scan the
# `CrossFeed.fit_on` selection does NOT pick, so only the selected scans' cross-hand
# rows feed the station solve — the solved time-global R–L components still
# apply to every scan. Parallel-hand rows are untouched. `dets` is the per-scan
# `StationScanDetections` vector (mutated); `snr` supplies per-scan SNRs for
# selections that need them.
function mask_unselected_cross_hands!(dets, fit_on::AbstractScanSelection, groups, snr)
    fit_on isa AllScans && return dets
    recs = [
        (; index = s.index, source = s.source, scan = s.scan, snr = Float64(snr[s.index]))
            for s in groups
    ]
    sel = Set(select_scans(fit_on, recs))
    for (gi, d) in enumerate(dets)
        gi in sel && continue
        for p in eachindex(d.feeds)
            fa, fb = d.feeds[p]
            fa == fb && continue
            for bi in axes(d.det, 1)
                d.det[bi, p] = FringeDetection(0.0, 0.0, 0.0, 0.0, 0.0, false)
            end
        end
    end
    return dets
end

# EHT-HOPS-style station flags: a station that PARTICIPATES in a scan (has
# baselines there) but is left UNCONSTRAINED by the surviving stage-B rows
# keeps identity gains — record it as (station, geometry scan id) so
# `apply_calibration` zero-weights its baselines instead of passing raw phases
# through at full weight.
function unconstrained_flags(dets, covered, geom::DataGeometry)
    flags = Tuple{Int, Int}[]
    for gi in eachindex(dets)
        scanid = geom.scan_of_time[dets[gi].ti]
        stations = Set{Int}()
        for (a, b) in dets[gi].bl_pairs
            a == b && continue
            push!(stations, a); push!(stations, b)
        end
        for st in stations
            (st, gi) in covered || push!(flags, (st, scanid))
        end
    end
    return flags
end

# Flatten per-scan detection rows into parallel plain vectors for the solution
# `info` — HDF5-representable and cheap to filter (`suspect_fringes`).
function detection_table(scan_dets)
    n = sum(length, scan_dets; init = 0)
    det_scan = Vector{Int}(undef, n); det_ant_a = Vector{Int}(undef, n)
    det_ant_b = Vector{Int}(undef, n); det_pol = Vector{String}(undef, n)
    det_snr = Vector{Float64}(undef, n); det_pfa = Vector{Float64}(undef, n)
    i = 0
    for (gi, rows) in enumerate(scan_dets), r in rows
        i += 1
        det_scan[i] = gi; det_ant_a[i] = r.a; det_ant_b[i] = r.b
        det_pol[i] = r.pol; det_snr[i] = r.snr; det_pfa[i] = r.pfa
    end
    return (; det_scan, det_ant_a, det_ant_b, det_pol, det_snr, det_pfa)
end

# The flag block for the solution `info` (plain parallel vectors,
# HDF5-representable): stage-B-unconstrained (station, scan) pairs and any
# intra-site baselines excluded for crosstalk.
function flag_table(station_flags, excl)
    pairs_ab = excl === nothing ? Tuple{Int, Int}[] : sort!([p for p in excl if p[1] < p[2]])
    return (;
        flagged_ant = Int[f[1] for f in station_flags],
        flagged_scan = Int[f[2] for f in station_flags],
        excluded_ant_a = Int[p[1] for p in pairs_ab],
        excluded_ant_b = Int[p[2] for p in pairs_ab],
    )
end
