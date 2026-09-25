# ── Built-in solve steps ─────────────────────────────────────────────────────
#
# The stages of the fringe pipeline as `SolveStep`s: each declares its model
# components and implements `solve`, reading the data through `each_group`.

"""
    BaselineFringeFit(; model = default_fringe_terms(), search = FringeSearch(),
                      closure = Stationization(), rounds = 1, steer_cells = 9.0)

The fringe-fitting stage: a per-baseline delay/rate matched-filter `search` on
every scan group, then the closure-screened station WLS (`closure`) that ties
the feeds. What is solved is `model`, a phase-only [`GainModel`](@ref):
per-scan constant/delay/rate and the inter-feed offsets; see
[`Fring.default_fringe_terms`](@ref) for the default. The gauge is run-wide,
an argument of [`fit`](@ref).

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
Base.@kwdef struct BaselineFringeFit{M <: GainModel} <: SolveStep
    model::M = Fring.default_fringe_terms()
    search::Fring.FringeSearch = Fring.FringeSearch()
    closure::Fring.Stationization = Fring.Stationization()
    rounds::Int = 1
    steer_cells::Float64 = 9.0
end
provides(::BaselineFringeFit) = :fringe

"""
    Bandpass(; model = default_bandpass_terms(), smoother = JointSmoother())

The bandpass stage: the time-global phase / log-amplitude station bandpass,
solved from the residual of whichever earlier steps have already applied
their gains, over every scan it is given. What is fit
is `model`, a [`GainModel`](@ref) — see [`Fring.default_bandpass_terms`](@ref) for the
default and the component form the smoothers accept. How it is solved lives on
`smoother`, a pluggable [`Fring.AbstractBandpassSmoother`](@ref) carrying one
shape spec per observable; by default [`Fring.JointSmoother`](@ref), which
fits the complex visibilities against an explicit per-scan source coherence and
so does not assume the calibrator is unresolved and unpolarized. It solves one
complex gain per (station, feed, segment), so it needs both halves of the
model — a phase-only or amplitude-only model must name
[`Fring.PerTrackSmoother`](@ref) instead, which runs the per-channel closure
solves and then fits each track. The model is self-contained, so placing
`Bandpass` before or after `BaselineFringeFit` is equally legal.

To fit the bandpass on calibrator scans only, fit it on those scans and carry
the solution into the full-data fit as a correction:

    fr = fit(BaselineFringeFit(), data; gauge)
    bp = fit(ApplySolution(fr) |> Bandpass(), calibrator_scans; gauge)
    sol = fit(ApplySolution(fr) |> ApplySolution(bp) |> AdhocPhase(), data; gauge)
"""
Base.@kwdef struct Bandpass{M <: GainModel, S <: Fring.AbstractBandpassSmoother} <: SolveStep
    model::M = Fring.default_bandpass_terms()
    smoother::S = Fring.JointSmoother()
end
provides(::Bandpass) = :bandpass

"""
    AdhocPhase(; model = default_adhoc_terms(), smoother = SavitzkyGolaySmoother())
    AdhocPhase(smoother)

The per-integration atmospheric-phase stage (adhoc phasing): solves the
globally-closing per-AP station phase on the fringe/bandpass residual. What is
fit is `model`, a [`GainModel`](@ref) holding the single adhoc component —
see [`Fring.default_adhoc_terms`](@ref) for the default (feed-common) form and
its `feed` tying knob. How the solved tracks are smoothed lives on `smoother`,
a pluggable [`Fring.AbstractAdhocSmoother`](@ref) (`SavitzkyGolaySmoother`,
`JointOUSmoother`, `OUSmoother`, `PenalizedSmoother`, …); `JointOUSmoother`
requires the feed-common (`SharedFeeds`) model. The one-argument form takes the
smoother and keeps the default model.
"""
Base.@kwdef struct AdhocPhase{M <: GainModel, S <: Fring.AbstractAdhocSmoother} <: SolveStep
    model::M = Fring.default_adhoc_terms()
    smoother::S = Fring.SavitzkyGolaySmoother()
end
AdhocPhase(smoother::Fring.AbstractAdhocSmoother) = AdhocPhase(; smoother)
provides(::AdhocPhase) = :adhoc

# ── Model components (compiled in step order into one GainModel) ───────

# A step's model compiles without a geometry when a caller only wants the
# components (`model_components(step, nothing)`). `can_fit` is handed the
# `nothing` rather than a substitute, so a capability that genuinely needs the
# geometry fails there instead of being answered from a default.
_spec_geom(spec) = spec === nothing ? nothing : spec.geom

model_components(s::BaselineFringeFit, spec) = _vet_step_model(
    s, s.model,
    "See `BaselineFringeFit` for the models it fits — a capability " *
        "can depend on the data's own sampling, so a term this step fits " *
        "elsewhere may still be unfittable here. No shipped step fits dispersion " *
        "(`Dispersion`) or a per-band-group delay (a `FreqGroups`-segmented `Delay`).",
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
# materializes it against the antenna table.
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
        "segmentation>, Feed = PerFeed())` — a time segmentation whose segments " *
        "each span several scans. See `default_bandpass_terms`.",
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
        "Frequency = GlobalFrequency(), Feed = SharedFeeds() or PerFeed())` " *
        "(`JointOUSmoother`: `SharedFeeds()` only) — see `default_adhoc_terms`.",
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
# written and the (station, geometry scan id) pairs left unconstrained. `dets`
# is one scan's detections where the systems are block-diagonal, every scan's
# where they couple — the two paths differ only in that argument.
function _station_solve!(s::BaselineFringeFit, ctx::SolveContext, stageB, dets)
    ncomp, covered = Fring.solve_station_systems!(
        ctx.θ, dets, stageB; gauge = ctx.gauge, opts = s.closure,
    )
    return ncomp, Fring.unconstrained_flags(dets, covered, ctx.geom)
end

# `scan_snr`, `scan_ncells` and the detection table are pure logging
# (`fringe_snr_table` / `suspect_fringes` read them off the step's `info`),
# built from the final round only.
_fringe_report(results) = (;
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
        results = each_group(ctx) do stack, win
            _solve_group(s, ctx, stageB, stack, win; round = 1, local_solve = true)
        end
        flags = reduce(append!, (r.flags for r in results); init = Tuple{Int, Int}[])
        ncomp = sum((r.ncomp for r in results); init = 0)
        return (; ncomp, Fring.flag_table(flags)..., _fringe_report(results)...)
    end
    local results, ncomp, flags
    for round in 1:max(s.rounds, 1)
        results = each_group(ctx) do stack, win
            _solve_group(s, ctx, stageB, stack, win; round, local_solve = false)
        end
        ncomp, flags = _station_solve!(s, ctx, stageB, [r.det for r in results])
    end
    return (; ncomp, Fring.flag_table(flags)..., _fringe_report(results)...)
end

# One scan group's search. Round 1 searches the data; later rounds search the
# residual of the current θ.
function _solve_group(
        s::BaselineFringeFit, ctx::SolveContext, stageB, stack, win::GeometryWindow;
        round::Int = 1, local_solve::Bool = _scan_local_solve(s),
    )
    Vsearch = round > 1 ? Fring.residual_vis(ctx.layout, ctx.θ, stack, win) : stack[:vis]
    # Reference the detection phases to the epoch this scan's constant phase
    # columns are the phase at (`scan_phase_epoch`), not to the track epoch
    # `search_scan` defaults to for a standalone caller. The station solve reads
    # each phase as a constant, so any gap between the two epochs pours that
    # row's rate uncertainty into the constant — and the inter-feed offset,
    # which the model gives no rate of its own, has nothing to absorb it with.
    # A model with no rate column pins no epoch; the scan's own mean time is
    # then the natural place to measure a constant.
    epoch = Fring.scan_phase_epoch(ctx.model, ctx.layout, first(win.ti_idx))
    if epoch === nothing
        ts = @view ctx.geom.times[win.ti_idx]
        epoch = sum(ts) / length(ts)
    end
    res = Fring.search_scan(
        stack, ctx.geom, s.search;
        Vsearch, ngroups = length(ctx.groups), executor = inner_executor(ctx.exec),
        t0 = epoch,
    )
    feeds = feed_pairs(stack)
    # `res` covers only the surviving (cross) baselines; take its own pair list.
    bl_pairs = collect(UVData.DimensionalData.lookup(res, UVData.BaselineID))
    # The scan's frequency/time lever arms travel with its detections: they set
    # the CRB uncertainty of a delay and a rate, which is what puts the station
    # solve's residuals in units of σ (see `Stationization`).
    det = Fring._with_ti(
        res, first(win.ti_idx); epoch,
        freq_rms = Fring._rms_spread(frequencies(stack)),
        time_rms = Fring._rms_spread(timestamps(stack)),
    )

    # Per-scan search log for the solution diagnostics: every measured cell, its
    # family-wise PFA, and whether that PFA accepts it as a real fringe. Recording
    # the rejected cells too is what makes the near-threshold population visible;
    # `detected` is the column that separates them. The search cube is transient
    # (consumed by the station solve), so these are read off it here; `cells1` is
    # the only piece not already in the cube.
    cells1 = Fring._search_cells(frequencies(stack), timestamps(stack), s.search)
    ncells = cells1 * max(length(bl_pairs) * length(feeds), 1)
    pfa_max = s.closure.pfa_max
    ncomp, flags = 0, Tuple{Int, Int}[]
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
            ti = first(win.ti_idx)
            sd, sr = Fring.scan_station_terms(ctx.model, ctx.layout, ctx.θ, ti)
            # θ is dense: a station this scan never constrained reads back as an
            # identity 0, indistinguishable from a solved zero delay. Steering to
            # it would invent a prediction and manufacture detections, so the
            # solve's own unconstrained list is what makes those nodes unusable.
            for (a, _) in flags
                (1 <= a <= size(sd, 1)) || continue
                sd[a, :] .= NaN
                sr[a, :] .= NaN
            end
            steer = Fring.steer_scan(
                # The same epoch the search above referenced: `sr` is a rate
                # about it, as is the model's own Rate component.
                stack, res, bl_pairs, feeds, ctx.geom.f0,
                epoch, sd, sr;
                cells = s.steer_cells,
            )
        end
    end
    _st(field, j, p) = steer === nothing ? NaN : steer[field][j, p]
    rows = [
        (;
            a = bl_pairs[j][1], b = bl_pairs[j][2], pol = feeds[p],
            snr = res.snr[j, p], pfa = res.pfa[j, p],
            delay = res.delay[j, p], rate = res.rate[j, p],
            phase = res.phase[j, p],
            detected = res.pfa[j, p] <= pfa_max,
            snr_steer = _st(:snr, j, p), pfa_steer = _st(:pfa, j, p),
            delay_steer = _st(:delay, j, p), rate_steer = _st(:rate, j, p),
            # Measured at the station solution's delay and rate rather than found
            # blind. There is no threshold here: `pfa_max` decides fringe-group
            # membership on the blind pass, and once a station is in that group
            # its baselines are measured at the known fringe location to
            # arbitrarily low SNR. `pfa_steer` records the significance of what
            # was measured; it does not gate it.
            steered = res.pfa[j, p] > pfa_max && isfinite(_st(:snr, j, p)),
        )
            for p in eachindex(feeds) for j in eachindex(bl_pairs) if res.valid[j, p]
    ]
    max_snr = isempty(rows) ? 0.0 : maximum((r.snr for r in rows if r.detected); init = 0.0)
    local_solve && return (; ncomp, flags, max_snr, ncells, rows)
    return (; det, max_snr, ncells, rows)
end

# ── Bandpass: accumulate per scan → per-channel/joint solves ─────────────────

function _group_setup(::Bandpass, ctx::SolveContext)
    layout = ctx.layout
    # The accumulators' labels, shared by every scan group: each stored cross
    # pair and feed pair of the set.
    members = [ms for group in values(ctx.groups) for ms in values(group)]
    cross = [p for ms in members for p in Fring._member_station_pairs(ms) if p[1] != p[2]]
    stations, bl_pairs = Fring._station_pairs(cross, ctx.geom)
    feeds = sort!(unique!([f for ms in members for f in feed_pairs(ms)]))
    # `ctx.layout` holds only this step's own components. The two observables are
    # located by NAME through the layout's component tree: the flat `plans` list
    # carries one entry per station-signature group, so its positions stop naming
    # them as soon as a model differs across stations.
    return (;
        stations, feeds, bl_pairs, nant = ctx.nant, layout,
        bp_path = Fring._bandpass_path(layout.plantree, :phase),
        amp_path = Fring._bandpass_path(layout.plantree, :logamp),
        channel_freqs = ctx.geom.channel_freqs, spw_of_chan = ctx.geom.spw_of_chan,
    )
end

function _solve_group(s::Bandpass, ctx::SolveContext, setup, group)
    rl, wl = Fring.accumulate_bandpass(
        group, ctx.geom, setup.stations, setup.feeds; executor = inner_executor(ctx.exec),
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

function _solve_group(s::AdhocPhase, ctx::SolveContext, adhoc_plan, group)
    Fring.adhoc_scan!(
        ctx.θ, group, ctx.geom, adhoc_plan, s.smoother, ctx.gauge, ctx.nant;
        executor = inner_executor(ctx.exec),
    )
    return nothing
end

function solve(s::AdhocPhase, ctx::SolveContext)
    adhoc_plan = _group_setup(s, ctx)
    results = each_group(group -> _solve_group(s, ctx, adhoc_plan, group), ctx)
    return (; nscans = length(results))
end
