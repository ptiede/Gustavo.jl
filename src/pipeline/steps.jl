# ── Built-in solve steps of the composable pipeline ──────────────────────────
#
# The stages of the fringe pipeline as SolveSteps. Each declares model
# components / dependencies through the protocol hooks and implements the
# executor-driven visitor contract (start_pass!/process_scan!/finish_pass! —
# the runner in verbs.jl drives them). Every pipeline runs on this engine.

"""
    FringeFit(; model = FringeModel(), estimator = MatchedFilter())

The fringe-fitting stage. WHAT is solved is `model` ([`FringeModel`](@ref)):
the ordered phase-term list — per-scan constant/delay/rate, the inter-feed offsets
(the gauge pin, `ref_ant`, is run-wide — see [`CalibrationPipeline`](@ref)).
HOW it is solved lives on `estimator`, a pluggable
[`AbstractFringeEstimator`](@ref); by default [`MatchedFilter`](@ref)
(per-baseline delay/rate search + closure-screened station WLS).

Ionospheric dispersion (dTEC) and single-band delay (SBD) are NOT part of this
step — add a [`DispersionSBDFit`](@ref) step after it to fit them on the
fringe-corrected residual.
"""
Base.@kwdef struct FringeFit{M <: Fringe.FringeModel, E <: Fringe.AbstractFringeEstimator} <: SolveStep
    model::M = Fringe.FringeModel()
    estimator::E = Fringe.MatchedFilter()
end
provides(::FringeFit) = :fringe
required_grouping(::FringeFit) = :scan_complete
# The station solve runs in `finish_estimate!`, over the detections of EVERY
# scan at once (`solve_station_systems!`): a scan's θ does not exist until the
# pass ends, whatever the model's time segmentation says, so the pass is not
# scan-local even when the columns it solves are block-diagonal.
fusable_grouping(::FringeFit) = :global
# NOTE: no `fit_selection` method — the fringe pass streams EVERY scan (the
# default `AllScans`).

"""
    DispersionSBDFit(; dispersion = DispersionModel(), sbd = SingleBandDelay())

The ionospheric-dispersion (dTEC) and single-band-delay (SBD) refinement
stage: a per-scan joint (Δτ, dTEC) fit ([`DispersionModel`](@ref)) and a
per-band-group delay fit ([`SingleBandDelay`](@ref)), on the fringe-corrected
residual — place a [`FringeFit`](@ref) step earlier in the pipeline. Set
either field to `nothing` to disable that term.

The Δτ half of the joint fit lands in a PRIVATE per-scan delay column, not in
`FringeFit`'s own wideband delay: gains compose multiplicatively, so this
step's delay column times `FringeFit`'s is the same total correction as
incrementing one shared column would be, without either step writing into the
other's θ block.
"""
Base.@kwdef struct DispersionSBDFit{D, S} <: SolveStep
    dispersion::D = Fringe.DispersionModel()
    sbd::S = Fringe.SingleBandDelay()
end
provides(::DispersionSBDFit) = :refine
required_grouping(::DispersionSBDFit) = :scan_complete
# Both fits are per-scan and complete inside `process_scan!`; `finish_pass!`
# only reports what was compiled.
fusable_grouping(::DispersionSBDFit) = :scan

"""
    Bandpass(; model = BandpassModel(), estimator = SplitWLS())

The bandpass stage: the time-global phase / log-amplitude station bandpass,
solved from the residual of whichever earlier steps have already applied
their gains, over every scan (fit-on-subset / apply-everywhere: pre-filter the
`UVSet` before fitting if only a scan subset should contribute). WHAT is fit
is `model` ([`Fringe.BandpassModel`](@ref)): phase/amp/frequency-segmentation/
amp-shape-smoother. HOW it is solved lives on `estimator`, a pluggable
[`Fringe.AbstractBandpassEstimator`](@ref); by default
[`Fringe.SplitWLS`](@ref) (independent phase/log-amp closures) — or
[`Fringe.JointALS`](@ref), a joint complex-gain + per-scan source-coherence
solve for a resolved or polarized calibrator. The model is self-contained, so
placing `Bandpass` before or after `FringeFit` is equally legal.
"""
Base.@kwdef struct Bandpass{M <: Fringe.BandpassModel, E <: Fringe.AbstractBandpassEstimator} <: SolveStep
    model::M = Fringe.BandpassModel()
    estimator::E = Fringe.SplitWLS()
end
provides(::Bandpass) = :bandpass
required_grouping(::Bandpass) = :scan_complete
# `process_scan!` writes no θ at all — it returns accumulators that
# `solve_bandpass!` closes into one system over every scan. A time-global
# bandpass is not scan-local under any configuration.
fusable_grouping(::Bandpass) = :global
# No fit_selection override — every scan feeds the pass (protocol.jl's default AllScans).

"""
    TemporalSmoother(smoother = SavitzkyGolaySmoother())

The per-integration atmospheric-phase stage (adhoc phasing): solves the
globally-closing per-AP station phase on the fringe/bandpass residual, through
the pluggable [`AbstractAdhocSmoother`](@ref) (`SavitzkyGolaySmoother`,
`JointOUSmoother`, `OUSmoother`, `PenalizedSmoother`, …).
"""
Base.@kwdef struct TemporalSmoother <: SolveStep
    smoother::Fringe.AbstractAdhocSmoother = Fringe.SavitzkyGolaySmoother()
end
provides(::TemporalSmoother) = :adhoc
required_grouping(::TemporalSmoother) = :scan_complete
# `adhoc_scan!` fits the scan's per-AP track from that scan's stack alone and
# writes its θ before returning; the slots are disjoint per scan.
fusable_grouping(::TemporalSmoother) = :scan

# Solve steps run through the pipeline verbs, never the sequential
# `run_step` chain (they need the shared compiled model + streaming passes).
run_step(s::SolveStep, ctx::CalibrationContext) = error(
    "$(nameof(typeof(s))) is a SolveStep — run it through `fit`/`fitcalibrate`, " *
        "not step-by-step."
)

# ── Model components (compiled in step order into ONE StationGainModel) ───────

# The estimator vets the model here, at compile time, before any data is read.
# Both directions: no term the estimator cannot fit (its θ block would stay at
# zero and the solution would look fitted), and no missing term the estimator
# assumes exists (its own estimate of that quantity would be discarded). The
# check covers only THIS step's contributions — the adhoc and bandpass
# components come from steps that solve them themselves.
function model_components(s::FringeFit, spec)
    tree = Fringe.fringe_phase_components(s.model, spec.geom)
    comps = Calibration._flatten_components(tree)
    for tc in comps
        Fringe.can_fit(s.estimator, tc) || throw(
            ArgumentError(
                "$(nameof(typeof(s.estimator))) cannot fit the model term compiling to " *
                    "$(Calibration.component_label(tc)); its parameters would never be " *
                    "solved. Remove the term, or use an estimator that declares " *
                    "`Gustavo.Fringe.can_fit` for it.",
            ),
        )
    end
    Fringe.validate_model(s.estimator, comps)
    return (; phase = tree, logamp = (;))
end

# The dispersion/SBD components: a private per-scan delay-refinement column
# (shares the fringe stage's own wideband-delay SIGNATURE by design, but lives
# in this step's own SEPARATE model/θ) plus the dTEC column, both compiled only
# when the DispersionModel/geometry combination
# enables dTEC; and the SBD delay + companion constant, compiled only when the
# frequency axis has ≥ 2 band groups. Either half is dropped entirely by
# setting the matching field to `nothing`.
function model_components(s::DispersionSBDFit, spec)
    dispc = s.dispersion === nothing ? nothing : model_components(s.dispersion, spec.geom)
    sbdc = s.sbd === nothing ? nothing : model_components(s.sbd, spec.geom)
    phase = merge(
        dispc === nothing ? (;) : (;
            delay_refine = TiedComponent(Delay(), PerScan(), GlobalFrequency(), SharedFeeds()),
            dtec = dispc,
        ),
        sbdc === nothing ? (;) : (; sbd = sbdc),
    )
    return (; phase, logamp = (;))
end

# The bandpass components: phase and log-amp, per feed, time-stable, resolved in
# frequency by `s.freq` (the legacy `_fringe_model` placement — after the fringe
# terms).
function model_components(s::Bandpass, spec)
    Fringe.validate_bandpass(s.estimator, s.model)
    bpc = TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), s.model.freq), PerFeed())
    return (; phase = s.model.phase ? (bandpass = bpc,) : (;), logamp = s.model.amp ? (bandpass = bpc,) : (;))
end

# The per-integration adhoc phase: per-AP, feed-common, solved per scan by the
# temporal-smoother pass (the legacy `_fringe_model` placement — last).
model_components(s::TemporalSmoother, spec) = (;
    phase = (adhoc = TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), SharedFeeds()),),
    logamp = (;),
)

# ── FringeFit visitor (stage A: per-scan search → one global station solve) ───

function start_pass!(s::FringeFit, ctx::SolveContext)
    ctx.scratch[:fringe_round] = get(ctx.scratch, :fringe_round, 0) + 1
    return nothing
end

process_scan!(s::FringeFit, ctx::SolveContext, stack, win::GeometryWindow) =
    Fringe.estimate_scan!(s.estimator, ctx, s, stack, win)

finish_pass!(s::FringeFit, ctx::SolveContext) = Fringe.finish_estimate!(s.estimator, ctx, s)

# ── MatchedFilter: the per-baseline search + closure-screened station WLS ─────
#
# Defined qualified on the `Fringe` generics, not as bare `estimate_scan!`,
# which would mint a second function here and leave the seam's fallback in place.

Fringe.estimator_info(est::Fringe.MatchedFilter) = (; search = est.search)

function Fringe.estimate_scan!(
        est::Fringe.MatchedFilter, ctx::SolveContext, s::FringeFit,
        stack, win::GeometryWindow,
    )
    round = ctx.scratch[:fringe_round]::Int
    Vsearch = round > 1 ? Fringe.residual_vis(ctx.ev, ctx.θ, stack, win) : stack[:vis]
    res = Fringe.search_scan(
        stack, ctx.stream.geom, est.search;
        Vsearch, ngroups = length(ctx.stream.groups), executor = inner_executor(ctx.stream),
    )
    pols = pol_products(stack)
    # `res` covers only the surviving (cross) baselines; take its own pair list.
    bl_pairs = collect(UVData.DimensionalData.lookup(res, UVData.Baseline))
    # The scan's frequency/time lever arms travel with its detections: they set
    # the CRB uncertainty of a delay and a rate, which is what puts the station
    # solve's residuals in units of σ (see `Stationization`).
    det = Fringe._with_ti(
        res, first(win.ti_idx);
        freq_rms = Fringe._rms_spread(frequencies(stack)),
        time_rms = Fringe._rms_spread(timestamps(stack) .* 3600.0),
    )

    # Per-scan detection log for the solution diagnostics: the detection table
    # (with per-baseline PFA), its max SNR, and the scan's effective cell count.
    # The search cube is transient (consumed by the station solve), so these are
    # read off it here; `cells1` is the only piece not already in the cube.
    cells1 = Fringe._search_cells(frequencies(stack), timestamps(stack) .* 3600.0, est.search)
    ncells = cells1 * max(length(bl_pairs) * length(pols), 1)
    rows = [
        (; a = bl_pairs[j][1], b = bl_pairs[j][2], pol = pols[p],
           snr = res.snr[j, p], pfa = Fringe.fringe_pfa(res.snr[j, p], cells1))
            for p in eachindex(pols) for j in eachindex(bl_pairs) if res.valid[j, p]
    ]
    max_snr = isempty(rows) ? 0.0 : maximum(r.snr for r in rows)
    return (; det, max_snr, ncells, rows)
end

function Fringe.finish_estimate!(est::Fringe.MatchedFilter, ctx::SolveContext, s::FringeFit)
    results = ctx.scratch[:pass_results]
    ngroups = length(ctx.stream.groups)
    # `dets` is SOLVE-ESSENTIAL (it feeds the closure-screened WLS immediately
    # below) — the pass covers every group every round, so a fresh build each
    # round is exact.
    dets = Vector{Any}(undef, ngroups)
    for res in results
        dets[res.index] = res.r.det
    end
    scan_snr = scan_values(res -> res.r.max_snr, results, ngroups; default = 0.0)
    # `ctx.model` holds only FringeFit's own components (each step solves on
    # its own private model/θ, never a merged one — CHUNK-069), so no
    # restriction is needed: a later step's component sharing a stage-B
    # signature by design (`DispersionSBDFit`'s private delay-refinement
    # column vs. this model's own wideband delay) lives in a SEPARATE model
    # and never appears here.
    stageB = Fringe.fringe_stage_components(ctx.model, ctx.layout)
    opts = Fringe.resolve_closure(est)
    ncomp, covered = Fringe.solve_station_systems!(
        ctx.θ, dets, stageB; ref_ant = ctx.ref_ant, opts = opts,
    )
    ctx.scratch[:fringe_flags] = Fringe.unconstrained_flags(dets, covered, ctx.geom)
    round = ctx.scratch[:fringe_round]::Int
    # Another round re-searches the residual.
    round < max(est.rounds, 1) && return (; repeat_pass = true, ncomp)
    # `scan_snr` is published for a LATER step's non-data input (e.g. a
    # `ScanWhere` selection reading it off this step's `StepSolution.info` —
    # see `_scan_snr`). `scan_ncells` and the detection table are pure logging
    # (`Fringe.diagnostics.jl`'s `fringe_snr_table`/`suspect_fringes` read them
    # off this same `StepSolution.info`) — built only now, on the final round,
    # via the same `scan_values` primitive every step's per-scan diagnostics use.
    return (;
        ncomp, scan_snr,
        scan_ncells = scan_values(res -> res.r.ncells, results, ngroups; default = 0.0),
        Fringe.detection_table(
            scan_values(res -> res.r.rows, results, ngroups; default = Fringe.DetectionRow[]),
        )...,
    )
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
    # `ctx.model` holds only this step's own components (CHUNK-069): the
    # per-scan delay-refinement column — sharing FringeFit's wideband-delay
    # SIGNATURE by design — is the only `_is_perscan_delay` match here, so
    # the plain `findfirst` router (`_perscan_delay_plan`) finds it directly;
    # `nothing` when dispersion is disabled (no such component was compiled).
    delay_plan = disp_plan === nothing ? nothing : Fringe._perscan_delay_plan(ctx.model, ctx.layout)
    ctx.scratch[:disp_sbd_setup] = (;
        delay_plan, disp_plan,
        sbd_plans = Fringe._sbd_plans(ctx.model, ctx.layout),
        ties = _dtec_ties(s.dispersion, ctx.antennas),
    )
    return nothing
end

function process_scan!(s::DispersionSBDFit, ctx::SolveContext, stack, win::GeometryWindow)
    setup = ctx.scratch[:disp_sbd_setup]
    Fringe.refine_scan_dispersion!(
        ctx.θ, stack, win, setup.delay_plan, setup.disp_plan, ctx.ref_ant, ctx.nant;
        executor = inner_executor(ctx.stream), ties = setup.ties,
    )
    Fringe.refine_scan_sbd!(
        ctx.θ, stack, win, setup.sbd_plans, ctx.ref_ant, ctx.nant;
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
    # The GLOBAL baseline table of the accumulation: every cross pair.
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    blidx = Dict(bl_pairs[i] => i for i in eachindex(bl_pairs))
    # This step's own model names its components (`model_components` above), so
    # the plans come straight off the named tree — the same `phase`/`amp` flags
    # decide both what was compiled and what is fetched.
    ctx.scratch[:bp_setup] = (;
        bl_pairs, blidx, nant,
        bp_plan = s.model.phase ? layout.plantree.phase.bandpass : nothing,
        amp_plan = s.model.amp ? layout.plantree.logamp.bandpass : nothing,
        channel_freqs = ctx.geom.channel_freqs, spw_of_chan = ctx.geom.spw_of_chan,
    )
    return nothing
end

function process_scan!(s::Bandpass, ctx::SolveContext, stack, win::GeometryWindow)
    setup = ctx.scratch[:bp_setup]
    # `stack` arrives already fringe/dispersion/SBD-corrected through the
    # pipeline's transform chain (every earlier step's finished solution) —
    # this step just accumulates the residual, no correction of its own.
    pols = String.(pol_products(stack))
    nchan = length(setup.channel_freqs)
    rl, wl = Fringe.bandpass_accumulators(length(setup.bl_pairs), length(pols), nchan)
    Fringe.accumulate_bandpass!(
        rl, wl, setup.blidx, stack, win; derotate = Fringe.bandpass_derotate(s.estimator),
    )
    return (; rl, wl, pols, source = source_name(stack))
end

function finish_pass!(s::Bandpass, ctx::SolveContext)
    setup = ctx.scratch[:bp_setup]
    results = ctx.scratch[:pass_results]
    isempty(results) && return (; nscans = 0)     # no scans → bandpass stays 0
    Fringe.solve_bandpass!(
        s.estimator, ctx.θ, [res.r for res in results], setup, s.model; ref_ant = ctx.ref_ant,
    )
    scans = Int[res.index for res in results]
    return (;
        nscans = length(scans), scans,
        sources = unique(String[res.r.source for res in results]),
    )
end

# ── TemporalSmoother visitor (refine + per-scan adhoc solve; final pass) ──────

function start_pass!(s::TemporalSmoother, ctx::SolveContext)
    ctx.scratch[:adhoc_setup] = (;
        adhoc_plan = Fringe._adhoc_plan(ctx.model, ctx.layout),
    )
    return nothing
end

function process_scan!(s::TemporalSmoother, ctx::SolveContext, stack, win::GeometryWindow)
    setup = ctx.scratch[:adhoc_setup]
    # `stack` arrives already fringe/dispersion/SBD/bandpass-corrected through
    # the pipeline's transform chain — the per-AP phases fit that residual
    # directly, no correction of its own.
    Fringe.adhoc_scan!(
        ctx.θ, stack, win, setup.adhoc_plan, s.smoother, ctx.ref_ant, ctx.nant;
        executor = inner_executor(ctx.stream),
    )
    return nothing
end

function finish_pass!(s::TemporalSmoother, ctx::SolveContext)
    results = ctx.scratch[:pass_results]
    return (; nscans = length(results))
end
