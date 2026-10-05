# ── Built-in solve steps ─────────────────────────────────────────────────────
#
# The fringe-fitting solves as `SolveStep`s: each declares its model
# components and implements `solve`, reading the data through `each_group`.

"""
    BaselineFringeFit(; gauge, model = default_fringe_terms(), search = FringeSearch(),
                      closure = Stationization(), rounds = 1, steer_cells = 9.0)

The fringe-fitting stage: a per-baseline delay/rate matched-filter `search` on
every scan group, then the closure-screened station WLS (`closure`) that ties
the feeds. What is solved is `model`, a phase-only [`GainModel`](@ref):
per-scan constant/delay/rate and the inter-feed offsets; see
[`Fring.default_fringe_terms`](@ref) for the default. `gauge` (required, an
[`AbstractGauge`](@ref)) fixes the station systems' undetermined values, with
its full constraints; a [`ByComponent`](@ref) names components of `model`.

With one round and an all-per-scan model, each scan's station systems solve as
the scan is searched. A track-global column or `rounds > 1` instead pools every
scan's detections into one solve after the pass. `rounds` re-runs the search
on the residual of the current solution, reading the data once per round.

The models it fits: a `Delay`, `Rate` or `ConstantTerm` spanning the whole band
(`Frequency = GlobalFrequency()`), under any feed scope, with a time
segmentation no finer than a scan. One search per scan group measures one
delay, rate and phase per scan, so a segmentation that splits a scan (a
`TimeBlocks` shorter than the scans, `PerIntegration`) asks for θ columns the
search has no measurement to fill and is rejected when the step compiles the
model. A segmentation coarser than a scan is fitted: one column shared by
several scans is written by all of them. The model has phase components only.

The `search` measures every baseline and gates nothing; `closure.pfa_max` is
the one detection threshold, deciding which measurements are real fringes and
so which stations are calibrated (see [`Fring.Stationization`](@ref)).

After the station solve closes, every cell is re-measured at the delay and
rate the solution predicts ([`Fring.steer_scan`](@ref)), ungated: a station in
the fringe group has its baselines measured at the known fringe location to
arbitrarily low SNR. `steer_cells` sizes the trial count of the recorded
`pfa_steer` (a significance a caller may read, not a threshold);
`steer_cells = 0` skips steering. Steering runs only where each scan solves on
its own; under a pooled solve the steered columns are `NaN`.

The search measures one delay across the whole band, so this step fits neither
ionospheric dispersion (`Dispersion`) nor a per-band-group delay (a
`FreqGroups`-segmented `Delay`), and no shipped step does.
"""
Base.@kwdef struct BaselineFringeFit{M <: GainModel, C <: Fring.Stationization, R <: Real, G <: AbstractGauge} <: SolveStep
    model::M = Fring.default_fringe_terms()
    search::Fring.FringeSearch = Fring.FringeSearch()
    closure::C = Fring.Stationization()
    rounds::Int = 1
    steer_cells::R = 9.0
    gauge::G = _missing_gauge(BaselineFringeFit)
    function BaselineFringeFit(model::M, search, closure::C, rounds, steer_cells::R, gauge::G) where {M, C, R, G}
        _check_step_gauge(gauge, model)
        return new{M, C, R, G}(model, search, closure, rounds, steer_cells, gauge)
    end
end
provides(::BaselineFringeFit) = :fringe

"""
    Bandpass(; gauge, model = default_bandpass_terms(), smoother = JointSmoother())

The bandpass stage: the time-global phase / log-amplitude station bandpass,
solved from the residual of whichever earlier steps have already applied
their gains, over every scan it is given. What is fit
is `model`, a [`GainModel`](@ref) — see [`Fring.default_bandpass_terms`](@ref) for the
default and the component form the smoothers accept; each component's prior
states how its values relate within a spectral window. How it is solved
lives on `smoother`, a pluggable [`Fring.AbstractBandpassSmoother`](@ref); by
default [`Fring.JointSmoother`](@ref), which fits the complex visibilities
against an explicit per-scan source coherence and so does not assume the
calibrator is unresolved and unpolarized. `gauge` (required, an
[`AbstractGauge`](@ref)) references the bandpass phase: the smoothers pin its
[`gauge_anchor`](@ref) node, and `PerTrackSmoother`'s seed applies its full
constraints. It solves one complex gain per
(station, feed, frequency segment), so it needs both halves of the
model — a phase-only or amplitude-only model must name
[`Fring.PerTrackSmoother`](@ref) instead, which runs the per-segment closure
solves and then fits each track. The model is self-contained, so placing
`Bandpass` before or after `BaselineFringeFit` is equally legal.

The bandpass is fit on data the fringe solution has already corrected, for
example on calibrator scans held in memory:

    fr = fit(BaselineFringeFit(; gauge), calibrator_scans)
    bp = fit(Bandpass(; gauge), calibrate(fr, calibrator_scans; flag_bad = false, apply_flags = false))
"""
Base.@kwdef struct Bandpass{M <: GainModel, S <: Fring.AbstractBandpassSmoother, G <: AbstractGauge} <: SolveStep
    model::M = Fring.default_bandpass_terms()
    smoother::S = Fring.JointSmoother()
    gauge::G = _missing_gauge(Bandpass)
    function Bandpass(model::M, smoother::S, gauge::G) where {M, S, G}
        _check_step_gauge(gauge, model)
        return new{M, S, G}(model, smoother, gauge)
    end
end
provides(::Bandpass) = :bandpass

"""
    AdhocPhase(; gauge, model = default_adhoc_terms(), smoother = PerTrackAdhocSmoother())
    AdhocPhase(smoother; gauge)

The per-integration atmospheric-phase stage (adhoc phasing): solves the
globally-closing per-AP station phase on the fringe/bandpass residual. What is
fit is `model`, a [`GainModel`](@ref) holding the single adhoc component and
its prior along time — see [`Fring.default_adhoc_terms`](@ref) for the default
(feed-common, OU prior) form. How the tracks are fit under that prior lives on
`smoother`, a pluggable [`Fring.AbstractAdhocSmoother`](@ref)
([`Fring.PerTrackAdhocSmoother`](@ref) or [`Fring.JointKalmanSmoother`](@ref), which
requires the feed-common (`SharedFeeds`) model). The one-argument form takes
the smoother and keeps the default model. `gauge` (required, an
[`AbstractGauge`](@ref)) picks each scan's anchor station, the first of its
[`gauge_station_order`](@ref) the scan observes; a `ZeroSumPhase` also
references every AP to the stations covered throughout the scan.

The step's info holds `nscans` and `priors`: the prior each (station, feed)
track was fit under, its hyperparameters resolved, over `(AntennaName, Feed, Ti)` with
one `Ti` value per scan, its first AP epoch.
"""
Base.@kwdef struct AdhocPhase{M <: GainModel, S <: Fring.AbstractAdhocSmoother, G <: AbstractGauge} <: SolveStep
    model::M = Fring.default_adhoc_terms()
    smoother::S = Fring.PerTrackAdhocSmoother()
    gauge::G = _missing_gauge(AdhocPhase)
    function AdhocPhase(model::M, smoother::S, gauge::G) where {M, S, G}
        _check_step_gauge(gauge, model)
        return new{M, S, G}(model, smoother, gauge)
    end
end
AdhocPhase(smoother::Fring.AbstractAdhocSmoother; gauge = _missing_gauge(AdhocPhase)) = AdhocPhase(; smoother, gauge)
provides(::AdhocPhase) = :adhoc

step_gauge(s::Union{BaselineFringeFit, Bandpass, AdhocPhase}) = s.gauge

_missing_gauge(T) = throw(
    ArgumentError(
        "$(nameof(T)) needs a `gauge`: `PinAntenna(\"A1\")` (a reference station, or a ranked " *
            "list), `ZeroSumPhase()`, or `ByComponent((; component = gauge, …); default = gauge)` " *
            "to choose by component.",
    ),
)

# Every component path of `model` without its `:phase`/`:logamp` root, over the
# base tree and each station entry.
function _model_component_names(model::GainModel)
    names = Tuple[]
    for (_, tree) in _station_variants(model)
        append!(names, Calibration._leaf_paths(tree.phase), Calibration._leaf_paths(tree.logamp))
    end
    return unique!(names)
end

_check_step_gauge(gauge::AbstractGauge, model::GainModel) =
    check_gauge_components(gauge, _model_component_names(model))

# ── Model components (compiled in step order into one GainModel) ───────

# A step's model compiles without a geometry when a caller only wants the
# components (`model_components(step, nothing)`). `can_fit` is handed the
# `nothing` rather than a substitute, so a capability that genuinely needs the
# geometry fails there instead of being answered from a default.
_spec_geom(spec) = isnothing(spec) ? nothing : spec.geom

model_components(s::BaselineFringeFit, spec) = _vet_step_model(
    s, s.model,
    "It fits a `Delay`, `Rate` or `ConstantTerm` on `Frequency = GlobalFrequency()` " *
        "with a time segmentation no finer than the data's scans: the search measures " *
        "one value per scan, so a segmentation that splits a scan (`PerIntegration`, " *
        "or `TimeBlocks` shorter than a scan) leaves columns unmeasured. See " *
        "`BaselineFringeFit`. No shipped step fits dispersion (`Dispersion`) or a " *
        "per-band-group delay (a `FreqGroups`-segmented `Delay`).",
    spec,
)

# Vet a step's model argument against its solver's declared capability, at
# compile time, before any data is read: no component the solver cannot fit
# (`can_fit`; its θ block would stay at zero and the solution would look
# fitted), then the solver's own whole-tree requirements (`validate_model`).
# Both checks run once per distinct station tree — the base and each
# `stations` entry's effective pair — with the can_fit error naming the station
# whose entry carries the component. `accepted` finishes that error with the
# component form the solver family does fit. Returns `m`; the runner
# resolves it against the antenna table.
function _vet_step_model(solver, m::GainModel, accepted, spec = nothing)
    for (station, tree) in _station_variants(m)
        at = station === nothing ? "" : " (station $(repr(station)) entry)"
        for tc in Calibration._flatten_components(tree)
            Fring.can_fit(solver, tc, _spec_geom(spec)) || throw(
                ArgumentError(
                    "$(nameof(typeof(solver))) cannot fit the component " *
                        "$(Calibration.component_label(tc))$at; its parameters would " *
                        "never be solved. " * accepted,
                ),
            )
        end
        Fring.validate_model(solver, tree)
    end
    return m
end

# The distinct station trees a model assigns, as `station => (; phase, logamp)`
# pairs (`nothing` for the base).
function _station_variants(m::Calibration.GainModel)
    vars = Pair{Any, Any}[nothing => (; phase = m.phase, logamp = m.logamp)]
    for k in keys(m.stations)
        push!(vars, k => Calibration.station_components(m, k))
    end
    return vars
end

model_components(s::Bandpass, spec) = _vet_step_model(
    s.smoother, s.model,
    "Both shipped bandpass smoothers fit `GainComponent(ConstantTerm(); " *
        "Ti = <GlobalTime, InstrumentScans or TimeBlocks>, Frequency = <any " *
        "segmentation>, Feed = PerFeed(), prior = <nothing, or a RandomWalkPrior or " *
        "OUPrior along Frequency>)` — a time segmentation whose segments each span " *
        "several scans. A segment holding one scan (`PerScan`, `PerIntegration`) is " *
        "not fitted: the band mean of its gain is degenerate with that scan's source " *
        "coherence. See `default_bandpass_terms`.",
    spec,
)

# `Bandpass` delegates its solve to `smoother`, so the capability question and
# the name in the rejection both belong to the smoother rather than the step.
# `JointSmoother` writes its solve as a loop over station blocks; the closure
# path does not, and its θ blocks for the differing stations would stay at zero.
supports_station_heterogeneity(s::Bandpass) = supports_station_heterogeneity(s.smoother)
supports_station_heterogeneity(::Fring.AbstractBandpassSmoother) = false
supports_station_heterogeneity(::Fring.JointSmoother) = true

heterogeneity_rejector(s::Bandpass) =
    "Bandpass(smoother = $(nameof(typeof(s.smoother)))(...)), whose smoother solves " *
    "one rectangular gain table for every station — `Bandpass(smoother = JointSmoother())` " *
    "solves a per-station model"

model_components(s::AdhocPhase, spec) = _vet_step_model(
    s.smoother, s.model,
    "The adhoc smoothers fit `GainComponent(ConstantTerm(); Ti = PerIntegration(), " *
        "Frequency = GlobalFrequency(), Feed = SharedFeeds() or PerFeed(), prior = " *
        "<nothing, or a RandomWalkPrior or OUPrior along Ti>)` " *
        "(`JointKalmanSmoother`: `SharedFeeds()` and an `OUPrior` only) — see `default_adhoc_terms`.",
    spec,
)

# ── BaselineFringeFit: per-scan search + closure-screened station solve ──────

Fring.can_fit(::BaselineFringeFit, tc, geom) = Fring.fringe_can_fit(tc, geom)
Fring.validate_model(::BaselineFringeFit, model) = Fring.validate_fringe_model(model)

# One round and an all-per-scan model make the station systems block-diagonal
# per scan, so each scan's WLS closes as the scan is searched. A cross-scan
# time segmentation (a `GlobalTime` inter-feed offset shares a column across
# scans) or `rounds > 1` (the re-search reads the whole pass's residual) pools
# every scan's detections into one solve instead.
function _scan_local_solve(s::BaselineFringeFit)
    s.rounds <= 1 || return false
    m = s.model
    trees = (m.phase, (e.phase for e in values(m.stations) if haskey(e, :phase))...)
    return all(
        t -> all(Calibration.component_is_per_scan, Calibration._flatten_components(t)),
        trees,
    )
end

# Solve into `ctx.θ` the station systems `dets` closes, reporting the components
# written and the (station, feed, geometry scan id) triples left unconstrained. `dets`
# is one scan's detections where the systems are block-diagonal, every scan's
# where they couple — the two paths differ only in that argument.
function _station_solve!(s::BaselineFringeFit, ctx::SolveContext, stageB, dets)
    ncomp, covered = Fring.solve_station_systems!(
        ctx.θ, dets, stageB, ctx.geom.stations; gauge = ctx.gauge, opts = s.closure,
    )
    return ncomp, Fring.unconstrained_flags(dets, covered, ctx.geom)
end

# `scan_names`, `scan_snr`, `scan_ncells` and the detection table are pure
# logging (`fringe_snr_table` / `fringe_detections` read them off the step's
# `info`), built from the final round only. `det_scan` indexes `scan_names`.
_fringe_report(results, ctx) = (;
    scan_names = _group_scan_names(ctx.groups),
    scan_snr = Float64[r.max_snr for r in results],
    scan_ncells = Float64[r.ncells for r in results],
    Fring.detection_table(Vector{Fring.DetectionRow}[r.rows for r in results])...,
)

# Each step's per-group kernel is `_solve_group(step, ctx, setup, …)`,
# with `setup = _group_setup(step, ctx)` computed once before the data are read.

# The stage-B components. A model whose rate components disagree on any
# constant-phase epoch is rejected here, before any data is read (see
# `scan_phase_epoch`).
function _group_setup(s::BaselineFringeFit, ctx::SolveContext)
    stageB = Fring.fringe_stage_components(ctx.model, ctx.layout)
    Fring.validate_scan_epochs(stageB, length(ctx.geom.times))
    return stageB
end

function solve(s::BaselineFringeFit, ctx::SolveContext)
    stageB = _group_setup(s, ctx)
    if _scan_local_solve(s)
        results = each_group(ctx) do group
            _solve_group(s, ctx, stageB, group; round = 1, local_solve = true)
        end
        flags = reduce(append!, (r.flags for r in results); init = Tuple{String, Int, Int}[])::Vector{Tuple{String, Int, Int}}
        ncomp = sum((r.ncomp for r in results); init = 0)::Int
        return (; ncomp, Fring.flag_table(flags)..., _fringe_report(results, ctx)..., s.search)
    end
    local results, ncomp, flags
    for round in 1:max(s.rounds, 1)
        results = each_group(ctx) do group
            _solve_group(s, ctx, stageB, group; round, local_solve = false)
        end
        ncomp, flags = _station_solve!(s, ctx, stageB, [r.det for r in results])::Tuple{Int, Vector{Tuple{String, Int, Int}}}
    end
    return (; ncomp, Fring.flag_table(flags)..., _fringe_report(results, ctx)..., s.search)
end

# One scan group's search. Round 1 searches the data; later rounds search the
# residual of the current θ.
function _solve_group(
        s::BaselineFringeFit, ctx::SolveContext, stageB, group; round::Int, local_solve::Bool,
    )
    round > 1 && (group = Fring.residual_group(ctx.layout, ctx.θ, group, ctx.geom))
    gc = Fring._GroupCells(group, ctx.geom)
    ti = Calibration._time_index(ctx.geom, first(gc.times))
    # Reference the detection phases to the epoch this scan's constant phase
    # columns are the phase at (`scan_phase_epoch`), not to the track epoch
    # `search_scan` defaults to for a standalone caller. The station solve reads
    # each phase as a constant, so any gap between the two epochs pours that
    # row's rate uncertainty into the constant — and the inter-feed offset,
    # which the model gives no rate of its own, has nothing to absorb it with.
    # A model with no rate column pins no epoch; the scan's own mean time is
    # then the natural place to measure a constant.
    epoch = Fring.scan_phase_epoch(ctx.model, ctx.layout, ti)
    isnothing(epoch) && (epoch = sum(gc.times) / length(gc.times))
    res = Fring.search_scan(
        gc, ctx.geom, s.search;
        executor = inner_executor(ctx.exec), t0 = epoch,
    )
    feeds = gc.feeds
    antenna_pairs = gc.antenna_pairs
    # The scan's frequency/time lever arms travel with its detections: they set
    # the CRB uncertainty of a delay and a rate, which is what puts the station
    # solve's residuals in units of σ (see `Stationization`).
    det = Fring._with_ti(
        res, ti; epoch,
        freq_rms = Fring._rms_spread(gc.freqs), time_rms = Fring._rms_spread(gc.times),
    )

    # Per-scan search log for the solution diagnostics: every measured cell, its
    # family-wise PFA, and whether that PFA accepts it as a real fringe. Recording
    # the rejected cells too is what makes the near-threshold population visible;
    # `detected` is the column that separates them. The search cube is transient
    # (consumed by the station solve), so these are read off it here; `ncells` is
    # the family the recorded `pfa` was computed over.
    ncells = Fring._family_cells(gc, s.search)
    pfa_max = s.closure.pfa_max
    ncomp, flags = 0, Tuple{String, Int, Int}[]
    # Steering needs θ for this scan, so it can only run where the station solve
    # closes here; a pooled solve has no station parameters until every group
    # has been read and the cube is long gone.
    steer = nothing
    if local_solve
        # Block-diagonal model: this scan's station systems close from its own
        # detections, so its θ columns are complete before this returns. The
        # slots are disjoint per scan, so concurrent groups write without
        # contention.
        ncomp, flags = _station_solve!(s, ctx, stageB, (det,))
        if s.steer_cells > 0
            sd, sr = Fring.scan_station_terms(ctx.model, ctx.layout, ctx.θ, ti, ctx.geom.stations)
            # θ is dense: a station this scan never constrained reads back as an
            # identity 0, indistinguishable from a solved zero delay. Steering to
            # it would invent a prediction and manufacture detections, so the
            # solve's own unconstrained list is what makes those nodes unusable.
            for (name, f, _) in flags
                sd[AntennaName(At(name)), Feed(At(f))] = NaN
                sr[AntennaName(At(name)), Feed(At(f))] = NaN
            end
            steer = Fring.steer_scan(
                # The same epoch the search above referenced: `sr` is a rate
                # about it, as is the model's own Rate component.
                gc, res, ctx.geom.f0, epoch, sd, sr;
                cells = s.steer_cells,
            )
        end
    end
    _st(field, cell) = isnothing(steer) ? NaN : steer[field][cell]
    rows = Fring.DetectionRow[]
    for p in eachindex(feeds), j in eachindex(antenna_pairs)
        cell = (AntennaPair(j), FeedPair(p))
        res.valid[cell] || continue
        push!(
            rows, (;
                a = antenna_pairs[j][1], b = antenna_pairs[j][2], feeds = feeds[p],
                snr = res.snr[cell], pfa = res.pfa[cell],
                delay = res.delay[cell], rate = res.rate[cell],
                phase = res.phase[cell],
                detected = res.pfa[cell] <= pfa_max,
                snr_steer = _st(:snr, cell), pfa_steer = _st(:pfa, cell),
                delay_steer = _st(:delay, cell), rate_steer = _st(:rate, cell),
                # Measured at the station solution's delay and rate rather than found
                # blind. There is no threshold here: `pfa_max` decides fringe-group
                # membership on the blind pass, and once a station is in that group
                # its baselines are measured at the known fringe location to
                # arbitrarily low SNR. `pfa_steer` records the significance of what
                # was measured; it does not gate it.
                steered = res.pfa[cell] > pfa_max && isfinite(_st(:snr, cell)),
            ),
        )
    end
    max_snr = isempty(rows) ? 0.0 : maximum((r.snr for r in rows if r.detected); init = 0.0)
    return (; det, ncomp, flags, max_snr, ncells, rows)
end

# ── Bandpass: accumulate per scan → per-channel/joint solves ─────────────────

function _group_setup(::Bandpass, ctx::SolveContext)
    layout = ctx.layout
    # The accumulators' labels, shared by every scan group: each stored cross
    # pair and feed pair of the set.
    members = [ms for group in values(ctx.groups) for ms in values(group)]
    station_pairs, feeds = Fring._cross_cell_labels(
        map(Fring._member_station_pairs, members), map(feed_pairs, members), ctx.geom,
    )
    # `ctx.layout` holds only this step's own components. The two observables are
    # located by NAME through the layout's component tree: the flat `plans` list
    # carries one entry per station-signature group, so its positions stop naming
    # them as soon as a model differs across stations.
    return (;
        station_pairs, feeds, layout, geom = ctx.geom,
        paths = (;
            phase = Fring._bandpass_paths(layout.plantree, :phase),
            logamp = Fring._bandpass_paths(layout.plantree, :logamp),
        ),
    )
end

function _solve_group(s::Bandpass, ctx::SolveContext, setup, group)
    rl, wl = Fring.accumulate_bandpass(
        group, ctx.geom, setup.station_pairs, setup.feeds; executor = inner_executor(ctx.exec),
    )
    # `ti` locates this scan on the solve's global time axis, which is how a
    # time-segmented bandpass tells which segment the scan belongs to. A scan
    # lies within one segment of any segmentation coarser than a scan, so its
    # first sample names the segment.
    t0 = minimum(minimum(XRadio.times(ms)) for ms in values(group))
    sources = unique(source_name(ms) for ms in values(group))
    length(sources) == 1 || throw(
        ArgumentError("a scan group observes several sources: $(join(sources, ", "))")
    )
    return (; rl, wl, ti = Calibration._time_index(ctx.geom, t0), source = only(sources))
end

function solve(s::Bandpass, ctx::SolveContext)
    setup = _group_setup(s, ctx)
    results = each_group(group -> _solve_group(s, ctx, setup, group), ctx)
    isempty(results) && return (; nscans = 0)     # no scans → bandpass stays 0
    report = Fring.solve_bandpass!(s.smoother, ctx.θ, results, setup; gauge = ctx.gauge)
    # The smoother's own per-track record travels with the step's info, so a
    # consumer can tell a measured bandpass track from a placeholder without
    # re-deriving it from the gains (where the two look identical).
    return (;
        nscans = length(results),
        sources = unique(String[r.source for r in results]),
        (report === nothing ? (;) : report)...,
    )
end

# ── AdhocPhase: per-scan per-AP phase solve ──────────────────────────────────

_group_setup(::AdhocPhase, ctx::SolveContext) = Fring._adhoc_plan(ctx.model, ctx.layout)

_solve_group(s::AdhocPhase, ctx::SolveContext, adhoc_plan, group) = Fring.adhoc_scan!(
    ctx.θ, group, ctx.geom, adhoc_plan, s.smoother, ctx.gauge;
    executor = inner_executor(ctx.exec),
)

function solve(s::AdhocPhase, ctx::SolveContext)
    adhoc_plan = _group_setup(s, ctx)
    results = each_group(group -> _solve_group(s, ctx, adhoc_plan, group), ctx)
    isempty(results) && return (; nscans = 0)
    sort!(results; by = r -> only(lookup(r, Ti)))
    return (; nscans = length(results), priors = cat(results...; dims = Ti))
end
