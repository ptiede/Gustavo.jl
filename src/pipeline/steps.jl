# ── Built-in solve steps of the composable pipeline ──────────────────────────
#
# The stages of the fringe pipeline as SolveSteps. Each declares model
# components / dependencies through the protocol hooks and implements the
# executor-driven visitor contract (start_pass!/process_scan!/finish_pass! —
# the runner in verbs.jl drives them). Every pipeline runs on this engine.

"""
    BaselineFringeFit(; model = default_fringe_terms(), estimator = MatchedFilter())

The fringe-fitting stage. What is solved is `model`, a phase-only
[`GainModel`](@ref): per-scan constant/delay/rate and the inter-feed offsets;
see [`Fring.default_fringe_terms`](@ref) for the default. The gauge pin, `gauge`,
is run-wide — an argument of [`fit`](@ref).
How it is solved lives on `estimator`, a pluggable
[`AbstractFringeEstimator`](@ref); by default [`MatchedFilter`](@ref)
(per-baseline delay/rate search + closure-screened station WLS).

Ionospheric dispersion (dTEC) and single-band delay (SBD) are not part of this
step — add a [`DispersionSBDFit`](@ref) step after it to fit them on the
fringe-corrected residual.
"""
Base.@kwdef struct BaselineFringeFit{M <: GainModel, E <: Fring.AbstractFringeEstimator} <: SolveStep
    model::M = Fring.default_fringe_terms()
    estimator::E = Fring.MatchedFilter()
end
provides(::BaselineFringeFit) = :fringe
required_grouping(::BaselineFringeFit) = :scan_complete
# WHERE θ is written decides the scope, and the estimator answers for its own
# configuration (`Fring.scan_local_solve`): the default `MatchedFilter` with
# one round and an all-per-scan model solves each scan's station systems
# inside `process_scan!`, so the pass is scan-local; any cross-scan coupling —
# a residual re-search round, a `GlobalTime`-tied inter-feed column, an
# estimator that pools every scan's detections — forces `:global`.
fusable_grouping(s::BaselineFringeFit) =
    Fring.scan_local_solve(s.estimator, s.model) ? :scan : :global

"""
    DispersionSBDFit(; dispersion = DispersionModel(), sbd = SingleBandDelay())

The ionospheric-dispersion (dTEC) and single-band-delay (SBD) refinement
stage: a per-scan joint (Δτ, dTEC) fit ([`DispersionModel`](@ref)) and a
per-band-group delay fit ([`SingleBandDelay`](@ref)), on the fringe-corrected
residual — place a [`BaselineFringeFit`](@ref) step earlier in the pipeline. Set
either field to `nothing` to disable that term.

The Δτ half of the joint fit lands in a private per-scan delay column, not in
`BaselineFringeFit`'s own wideband delay: gains compose multiplicatively, so this
step's delay column times `BaselineFringeFit`'s is the same total correction as
incrementing one shared column would be, without either step writing into the
other's θ block.
"""
Base.@kwdef struct DispersionSBDFit{D, S} <: SolveStep
    dispersion::D = Fring.DispersionModel()
    sbd::S = Fring.SingleBandDelay()
end
provides(::DispersionSBDFit) = :refine
required_grouping(::DispersionSBDFit) = :scan_complete
# Both fits are per-scan and complete inside `process_scan!`; `finish_pass!`
# only reports what was compiled.
fusable_grouping(::DispersionSBDFit) = :scan

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
required_grouping(::Bandpass) = :scan_complete
# `process_scan!` writes no θ at all — it returns accumulators that
# `solve_bandpass!` closes into one system over every scan. A time-global
# bandpass is not scan-local under any configuration.
fusable_grouping(::Bandpass) = :global

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
required_grouping(::AdhocPhase) = :scan_complete
# `adhoc_scan!` fits the scan's per-AP track from that scan's stack alone and
# writes its θ before returning; the slots are disjoint per scan.
fusable_grouping(::AdhocPhase) = :scan

# ── Model components (compiled in step order into one GainModel) ───────

# A step's model compiles without a geometry when a caller only wants the
# components (`model_components(step, nothing)`). `can_fit` is handed the
# `nothing` rather than a substitute, so a capability that genuinely needs the
# geometry fails there instead of being answered from a default.
_spec_geom(spec) = spec === nothing ? nothing : spec.geom

model_components(s::BaselineFringeFit, spec) = _vet_step_model(
    s.estimator, s.model,
    "See `$(nameof(typeof(s.estimator)))` for the models it fits — a capability " *
        "can depend on the data's own sampling, so a term this estimator fits " *
        "elsewhere may still be unfittable here. Dispersion (`Dispersion`) and " *
        "single-band delay (a `FreqGroups`-segmented `Delay`) are fit by a " *
        "`DispersionSBDFit` step, not by the fringe step.",
    spec,
)

# The dispersion/SBD components: a private per-scan delay-refinement column
# (shares the fringe stage's own wideband-delay signature by design, but lives
# in this step's own separate model/θ) plus the dTEC column, both compiled only
# when the DispersionModel/geometry combination
# enables dTEC; and the SBD delay + companion constant, compiled only when the
# frequency axis has ≥ 2 band groups. Either half is dropped entirely by
# setting the matching field to `nothing`.
function model_components(s::DispersionSBDFit, spec)
    dispc = s.dispersion === nothing ? nothing : model_components(s.dispersion, spec)
    sbdc = s.sbd === nothing ? nothing : model_components(s.sbd, spec)
    phase = merge(
        dispc === nothing ? (;) : (;
                delay_refine = GainComponent(Delay(); Ti = PerScan(), Feed = SharedFeeds()),
                dtec = dispc,
            ),
        sbdc === nothing ? (;) : (; sbd = sbdc),
    )
    return GainModel(; phase)
end

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

# ── BaselineFringeFit visitor (stage A: per-scan search + station solve) ─────
#
# The station solve runs where `scan_local_solve` says it can: per scan inside
# `estimate_scan!` when the systems are block-diagonal (each scan's θ is
# complete before its `process_scan!` returns — what a `:scan` declaration
# promises), or once over every scan's detections in `finish_estimate!` when a
# cross-scan column or a re-search round couples them.

function start_pass!(s::BaselineFringeFit, ctx::SolveContext)
    ctx.scratch[:fringe_round] = get(ctx.scratch, :fringe_round, 0) + 1
    # `ctx.model` holds only BaselineFringeFit's own components (each step solves on its
    # own private model/θ, never a merged one), so no restriction is needed: a
    # later step's component sharing a stage-B signature by design
    # (`DispersionSBDFit`'s private delay-refinement column vs. this model's own
    # wideband delay) lives in a separate model and never appears here.
    stageB = Fring.fringe_stage_components(ctx.model, ctx.layout)
    # A model whose rate components disagree on any constant-phase epoch is
    # rejected here, before the pass reads any data (see `scan_phase_epoch`).
    Fring.validate_scan_epochs(stageB, length(ctx.stream.geom.times))
    ctx.scratch[:fringe_setup] = (; stageB)
    return nothing
end

process_scan!(s::BaselineFringeFit, ctx::SolveContext, stack, win::GeometryWindow) =
    Fring.estimate_scan!(s.estimator, ctx, s, stack, win)

finish_pass!(s::BaselineFringeFit, ctx::SolveContext) = Fring.finish_estimate!(s.estimator, ctx, s)

# ── MatchedFilter: the per-baseline search + closure-screened station WLS ─────
#
# Defined qualified on the `Fring` generics, not as bare `estimate_scan!`,
# which would mint a second function here and leave the seam's fallback in place.

# The struct itself, not a flattened copy: `fringe_search_map` replays it to
# reproduce the solve's search for diagnostics.
Fring.estimator_info(est::Fring.MatchedFilter) = (; search = est.search)

# Solve into `ctx.θ` the station systems `dets` closes, reporting the components
# written and the (station, geometry scan id) pairs left unconstrained. `dets`
# is one scan's detections where the systems are block-diagonal, every scan's
# where they couple — the two paths differ only in that argument.
function _station_solve!(est::Fring.MatchedFilter, ctx::SolveContext, dets)
    ncomp, covered = Fring.solve_station_systems!(
        ctx.θ, dets, ctx.scratch[:fringe_setup].stageB;
        gauge = ctx.gauge, opts = est.closure,
    )
    return ncomp, Fring.unconstrained_flags(dets, covered, ctx.geom)
end

# The pass diagnostics both solve paths report. `scan_snr`, `scan_ncells` and
# the detection table are pure logging (`Fring.diagnostics.jl`'s `fringe_snr_table` /
# `suspect_fringes` read them off that same `info`), built on the final round
# only, via the `scan_values` primitive every step's per-scan diagnostics use.
_fringe_report(results, ngroups) = (;
    scan_snr = scan_values(res -> res.r.max_snr, results, ngroups; default = 0.0),
    scan_ncells = scan_values(res -> res.r.ncells, results, ngroups; default = 0.0),
    Fring.detection_table(
        scan_values(res -> res.r.rows, results, ngroups; default = Fring.DetectionRow[]),
    )...,
)

function Fring.estimate_scan!(
        est::Fring.MatchedFilter, ctx::SolveContext, s::BaselineFringeFit,
        stack, win::GeometryWindow,
    )
    round = ctx.scratch[:fringe_round]::Int
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
        ts = @view ctx.stream.geom.times[win.ti_idx]
        epoch = sum(ts) / length(ts)
    end
    res = Fring.search_scan(
        stack, ctx.stream.geom, est.search;
        Vsearch, ngroups = length(ctx.stream.groups), executor = inner_executor(ctx.stream),
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
    cells1 = Fring._search_cells(frequencies(stack), timestamps(stack), est.search)
    ncells = cells1 * max(length(bl_pairs) * length(feeds), 1)
    pfa_max = est.closure.pfa_max
    local_solve = Fring.scan_local_solve(est, s.model)
    ncomp, flags = 0, Tuple{Int, Int}[]
    # Steering needs θ for this scan, so it can only run where the station solve
    # closes here (`scan_local_solve`); a pooled solve has no station parameters
    # until every group has been read and the cube is long gone.
    steer = nothing
    if local_solve
        # Block-diagonal model: this scan's station systems close from its own
        # detections, so its θ columns are complete before this returns. The
        # slots are disjoint per scan, so concurrent groups write without
        # contention.
        ncomp, flags = _station_solve!(est, ctx, (det,))
        if est.steer_cells > 0
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
                stack, res, bl_pairs, feeds, ctx.stream.geom.f0,
                epoch, sd, sr;
                cells = est.steer_cells,
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

function Fring.finish_estimate!(est::Fring.MatchedFilter, ctx::SolveContext, s::BaselineFringeFit)
    results = ctx.scratch[:pass_results]
    ngroups = length(ctx.stream.groups)
    if Fring.scan_local_solve(est, s.model)
        # θ was written scan by scan in `estimate_scan!`; what remains is the
        # aggregation the pooled solve would report: block-diagonal systems are
        # disjoint, so the component counts sum and the flags concatenate in
        # scan order.
        ctx.scratch[:fringe_flags] = reduce(
            append!,
            scan_values(res -> res.r.flags, results, ngroups; default = Tuple{Int, Int}[]);
            init = Tuple{Int, Int}[],
        )
        ncomp = sum(scan_values(res -> res.r.ncomp, results, ngroups; default = 0))
        return (; ncomp, _fringe_report(results, ngroups)...)
    end
    # `dets` is solve-ESSENTIAL (it feeds the closure-screened WLS immediately
    # below) — the pass covers every group every round, so a fresh build each
    # round is exact.
    dets = Vector{Any}(undef, ngroups)
    for res in results
        dets[res.index] = res.r.det
    end
    ncomp, flags = _station_solve!(est, ctx, dets)
    ctx.scratch[:fringe_flags] = flags
    # Another round re-searches the residual.
    ctx.scratch[:fringe_round]::Int < max(est.rounds, 1) &&
        return (; repeat_pass = true, ncomp)
    return (; ncomp, _fringe_report(results, ngroups)...)
end

# ── DispersionSBDFit visitor (per-scan joint (Δτ, dTEC) fit + SBD fit) ────────

# Co-located stations see the same ionosphere, so a differential TEC between
# them is pure solve error — but only a model that solves dTEC has any to tie,
# and only the caller knows the separation that counts as co-located here.
_dtec_ties(::Nothing, antennas) = nothing
_dtec_ties(dm::DispersionModel, antennas) =
    dm.colocated_sep === nothing ? nothing :
    UVData._colocated_ties(antennas; max_sep = dm.colocated_sep)

function start_pass!(s::DispersionSBDFit, ctx::SolveContext)
    disp_plan = Calibration._dispersion_plan(ctx.model, ctx.layout)
    # `ctx.model` holds only this step's own components: the
    # per-scan delay-refinement column — sharing BaselineFringeFit's wideband-delay
    # Signature by design — is the only `_is_perscan_delay` match here, so
    # the plain `findfirst` router (`_perscan_delay_plan`) finds it directly;
    # `nothing` when dispersion is disabled (no such component was compiled).
    delay_plan = disp_plan === nothing ? nothing : Fring._perscan_delay_plan(ctx.model, ctx.layout)
    ctx.scratch[:disp_sbd_setup] = (;
        delay_plan, disp_plan,
        sbd_plans = Fring._sbd_plans(ctx.model, ctx.layout),
        ties = _dtec_ties(s.dispersion, ctx.antennas),
    )
    return nothing
end

function process_scan!(s::DispersionSBDFit, ctx::SolveContext, stack, win::GeometryWindow)
    setup = ctx.scratch[:disp_sbd_setup]
    Fring.refine_scan_dispersion!(
        ctx.θ, stack, win, setup.delay_plan, setup.disp_plan, ctx.gauge, ctx.nant;
        executor = inner_executor(ctx.stream), ties = setup.ties,
    )
    Fring.refine_scan_sbd!(
        ctx.θ, stack, win, setup.sbd_plans, ctx.gauge, ctx.nant;
        executor = inner_executor(ctx.stream),
    )
    return nothing
end

function finish_pass!(s::DispersionSBDFit, ctx::SolveContext)
    results = ctx.scratch[:pass_results]
    setup = ctx.scratch[:disp_sbd_setup]
    # This step's own compiled model alone says whether dTEC/SBD were fit —
    # published here (not derived later from the model by a name-hardcoded
    # lookup) so a solution-level consumer needs no knowledge of this step's
    # name to ask "was dispersion/SBD applied?".
    return (;
        nscans = length(results),
        dispersion_applied = setup.disp_plan !== nothing, sbd_applied = setup.sbd_plans !== nothing,
    )
end

# ── Bandpass visitor (accumulate per scan → per-channel/joint solves) ────────

function start_pass!(s::Bandpass, ctx::SolveContext)
    layout = ctx.layout
    nant = ctx.nant
    # The global baseline table of the accumulation: every cross pair.
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    blidx = Dict(bl_pairs[i] => i for i in eachindex(bl_pairs))
    # `ctx.layout` holds only this step's own components. The two observables are
    # located by NAME through the layout's component tree: the flat `plans` list
    # carries one entry per station-signature group, so its positions stop naming
    # them as soon as a model differs across stations.
    ctx.scratch[:bp_setup] = (;
        bl_pairs, blidx, nant, layout,
        bp_path = Fring._bandpass_path(layout.plantree, :phase),
        amp_path = Fring._bandpass_path(layout.plantree, :logamp),
        channel_freqs = ctx.geom.channel_freqs, spw_of_chan = ctx.geom.spw_of_chan,
    )
    return nothing
end

function process_scan!(s::Bandpass, ctx::SolveContext, stack, win::GeometryWindow)
    setup = ctx.scratch[:bp_setup]
    # `stack` arrives already fringe/dispersion/SBD-corrected through the
    # pipeline's transform chain (every earlier step's finished solution) —
    # this step just accumulates the residual, no correction of its own.
    feeds = feed_pairs(stack)
    nchan = length(setup.channel_freqs)
    rl, wl = Fring.bandpass_accumulators(length(setup.bl_pairs), length(feeds), nchan)
    Fring.accumulate_bandpass!(
        rl, wl, setup.blidx, stack, win; derotate = Fring.bandpass_derotate(s.smoother),
    )
    # `ti` locates this scan on the solve's global time axis, which is how a
    # time-segmented bandpass tells which segment the scan belongs to. A scan
    # lies within one segment of any segmentation coarser than a scan, so its
    # first sample names the segment.
    return (; rl, wl, feeds, ti = first(win.ti_idx), source = source_name(stack))
end

function finish_pass!(s::Bandpass, ctx::SolveContext)
    setup = ctx.scratch[:bp_setup]
    results = ctx.scratch[:pass_results]
    isempty(results) && return (; nscans = 0)     # no scans → bandpass stays 0
    report = Fring.solve_bandpass!(
        s.smoother, ctx.θ, [res.r for res in results], setup; gauge = ctx.gauge,
    )
    scans = Int[res.index for res in results]
    # The smoother's own per-track record travels with the step's info, so a
    # consumer can tell a measured bandpass track from a placeholder without
    # re-deriving it from the gains (where the two look identical).
    return (;
        nscans = length(scans), scans,
        sources = unique(String[res.r.source for res in results]),
        (report === nothing ? (;) : report)...,
    )
end

# ── AdhocPhase visitor (refine + per-scan adhoc solve; final pass) ────────────

function start_pass!(s::AdhocPhase, ctx::SolveContext)
    ctx.scratch[:adhoc_setup] = (;
        adhoc_plan = Fring._adhoc_plan(ctx.model, ctx.layout),
    )
    return nothing
end

function process_scan!(s::AdhocPhase, ctx::SolveContext, stack, win::GeometryWindow)
    setup = ctx.scratch[:adhoc_setup]
    # `stack` arrives already fringe/dispersion/SBD/bandpass-corrected through
    # the pipeline's transform chain — the per-AP phases fit that residual
    # directly, no correction of its own.
    Fring.adhoc_scan!(
        ctx.θ, stack, win, setup.adhoc_plan, s.smoother, ctx.gauge, ctx.nant;
        executor = inner_executor(ctx.stream),
    )
    return nothing
end

function finish_pass!(s::AdhocPhase, ctx::SolveContext)
    results = ctx.scratch[:pass_results]
    return (; nscans = length(results))
end
