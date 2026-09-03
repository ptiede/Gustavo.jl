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
(the gauge pin, `gauge`, is run-wide — see [`CalibrationPipeline`](@ref)).
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
# WHERE θ is written decides the scope, and the estimator answers for its own
# configuration (`Fringe.scan_local_solve`): the default `MatchedFilter` with
# one round and an all-per-scan term list solves each scan's station systems
# inside `process_scan!`, so the pass is scan-local; any cross-scan coupling —
# a residual re-search round, a `GlobalTime`-tied inter-feed column, an
# estimator that pools every scan's detections — forces `:global`.
fusable_grouping(s::FringeFit) =
    Fringe.scan_local_solve(s.estimator, s.model) ? :scan : :global
# Consulted only inside a fused run — exactly the scan-local configuration,
# whose `estimate_scan!` return carries the scan's own unconstrained flags.
scan_flags(s::FringeFit, r) = r.flags
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
    Bandpass(; model = default_bandpass_terms(), smoother = JointSmoother())

The bandpass stage: the time-global phase / log-amplitude station bandpass,
solved from the residual of whichever earlier steps have already applied
their gains, over every scan (fit-on-subset / apply-everywhere: pre-filter the
`UVSet` before fitting if only a scan subset should contribute). WHAT is fit
is `model`: a `(; phase, logamp)` tree of named `Calibration.GainComponent`s
(or a `StationGainModel`) — see [`Fringe.default_bandpass_terms`](@ref) for the
default and the component form the smoothers accept. HOW it is solved lives on
`smoother`, a pluggable [`Fringe.AbstractBandpassSmoother`](@ref) carrying one
shape spec per observable; by default [`Fringe.JointSmoother`](@ref), which
fits the complex visibilities against an explicit per-scan source coherence and
so does not assume the calibrator is unresolved and unpolarized. It solves one
complex gain per (station, feed, segment), so it needs both halves of the
model — a phase-only or amplitude-only model must name
[`Fringe.PerTrackSmoother`](@ref) instead, which runs the per-channel closure
solves and then fits each track. The model is self-contained, so placing
`Bandpass` before or after `FringeFit` is equally legal.
"""
Base.@kwdef struct Bandpass{M, S <: Fringe.AbstractBandpassSmoother} <: SolveStep
    model::M = Fringe.default_bandpass_terms()
    smoother::S = Fringe.JointSmoother()
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
                delay_refine = GainComponent(Delay(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds()),
                dtec = dispc,
            ),
        sbdc === nothing ? (;) : (; sbd = sbdc),
    )
    return (; phase, logamp = (;))
end

# A step's model argument, lifted to the `(; phase, logamp)` tree
# `model_components` returns: a `StationGainModel` contributes its two groups; a
# bare NamedTuple tree may omit either group but no other key is legal — the
# likely mistake is writing a component name at the top level.
_step_model_tree(m::Calibration.StationGainModel) = (; phase = m.phase, logamp = m.logamp)
function _step_model_tree(m::NamedTuple)
    unknown = setdiff(keys(m), (:phase, :logamp))
    isempty(unknown) || throw(
        ArgumentError(
            "step model tree has unexpected key(s) $(Tuple(unknown)): a model is " *
                "`(; phase, logamp)` — component names nest INSIDE the groups, e.g. " *
                "`(; phase = (; bandpass = GainComponent(...)))`.",
        ),
    )
    return (;
        phase = Calibration._named_components(get(m, :phase, (;))),
        logamp = Calibration._named_components(get(m, :logamp, (;))),
    )
end
_step_model_tree(m) = throw(
    ArgumentError(
        "a step model must be a `StationGainModel` or a `(; phase, logamp)` NamedTuple " *
            "tree of named `GainComponent`s, got $(typeof(m)).",
    ),
)

# The bandpass components come straight from the step's model argument; the
# smoother vets them here, at compile time, before any data is read — the same
# two-sided check as `FringeFit`'s: no component the smoother cannot fit (its θ
# block would stay at zero and the solution would look fitted), and the
# smoother's own whole-tree requirements (`validate_model`).
function model_components(s::Bandpass, spec)
    tree = _step_model_tree(s.model)
    for tc in Calibration._flatten_components(tree)
        Fringe.can_fit(s.smoother, tc) || throw(
            ArgumentError(
                "$(nameof(typeof(s.smoother))) cannot fit the bandpass component " *
                    "$(Calibration.component_label(tc)); its parameters would never be " *
                    "solved. Both shipped smoothers fit " *
                    "`GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = <any " *
                    "segmentation>, Feed = PerFeed())` — see `default_bandpass_terms`.",
            ),
        )
    end
    Fringe.validate_model(s.smoother, tree)
    return tree
end

# The per-integration adhoc phase: per-AP, feed-common, solved per scan by the
# temporal-smoother pass (the legacy `_fringe_model` placement — last).
model_components(s::TemporalSmoother, spec) = (;
    phase = (adhoc = GainComponent(ConstantTerm(); Ti = PerIntegration(), Frequency = GlobalFrequency(), Feed = SharedFeeds()),),
    logamp = (;),
)

# ── FringeFit visitor (stage A: per-scan search + station solve) ─────────────
#
# The station solve runs where `scan_local_solve` says it can: per scan inside
# `estimate_scan!` when the systems are block-diagonal (each scan's θ is
# complete before its `process_scan!` returns — what a `:scan` declaration
# promises), or once over every scan's detections in `finish_estimate!` when a
# cross-scan column or a re-search round couples them.

function start_pass!(s::FringeFit, ctx::SolveContext)
    ctx.scratch[:fringe_round] = get(ctx.scratch, :fringe_round, 0) + 1
    # `ctx.model` holds only FringeFit's own components (each step solves on its
    # own private model/θ, never a merged one), so no restriction is needed: a
    # later step's component sharing a stage-B signature by design
    # (`DispersionSBDFit`'s private delay-refinement column vs. this model's own
    # wideband delay) lives in a SEPARATE model and never appears here.
    ctx.scratch[:fringe_setup] = (;
        stageB = Fringe.fringe_stage_components(ctx.model, ctx.layout),
    )
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

# Solve into `ctx.θ` the station systems `dets` closes, reporting the components
# written and the (station, geometry scan id) pairs left unconstrained. `dets`
# is one scan's detections where the systems are block-diagonal, every scan's
# where they couple — the two paths differ only in that argument.
function _station_solve!(est::Fringe.MatchedFilter, ctx::SolveContext, dets)
    ncomp, covered = Fringe.solve_station_systems!(
        ctx.θ, dets, ctx.scratch[:fringe_setup].stageB;
        gauge = ctx.gauge, opts = est.closure,
    )
    return ncomp, Fringe.unconstrained_flags(dets, covered, ctx.geom)
end

# The pass diagnostics both solve paths report. `scan_snr` is a LATER step's
# non-data input (e.g. a `ScanWhere` selection reading it off this step's
# `StepSolution.info` — see `_scan_snr`); `scan_ncells` and the detection table
# are pure logging (`Fringe.diagnostics.jl`'s `fringe_snr_table` /
# `suspect_fringes` read them off that same `info`), built on the final round
# only, via the `scan_values` primitive every step's per-scan diagnostics use.
_fringe_report(results, ngroups) = (;
    scan_snr = scan_values(res -> res.r.max_snr, results, ngroups; default = 0.0),
    scan_ncells = scan_values(res -> res.r.ncells, results, ngroups; default = 0.0),
    Fringe.detection_table(
        scan_values(res -> res.r.rows, results, ngroups; default = Fringe.DetectionRow[]),
    )...,
)

function Fringe.estimate_scan!(
        est::Fringe.MatchedFilter, ctx::SolveContext, s::FringeFit,
        stack, win::GeometryWindow,
    )
    round = ctx.scratch[:fringe_round]::Int
    Vsearch = round > 1 ? Fringe.residual_vis(ctx.ev, ctx.θ, stack, win) : stack[:vis]
    # Reference the detection phases to the epoch this scan's constant phase
    # columns are the phase AT (`scan_phase_epoch`), not to the track epoch
    # `search_scan` defaults to for a standalone caller. The station solve reads
    # each phase as a constant, so any gap between the two epochs pours that
    # row's rate uncertainty into the constant — and the inter-feed offset,
    # which the model gives no rate of its own, has nothing to absorb it with.
    # A model with no rate column pins no epoch; the scan's own mean time is
    # then the natural place to measure a constant.
    epoch = Fringe.scan_phase_epoch(ctx.model, ctx.layout, first(win.ti_idx))
    if epoch === nothing
        ts = @view ctx.stream.geom.times[win.ti_idx]
        epoch = sum(ts) / length(ts)
    end
    res = Fringe.search_scan(
        stack, ctx.stream.geom, est.search;
        Vsearch, ngroups = length(ctx.stream.groups), executor = inner_executor(ctx.stream),
        t0 = epoch * 3600.0,
    )
    pols = pol_products(stack)
    # `res` covers only the surviving (cross) baselines; take its own pair list.
    bl_pairs = collect(UVData.DimensionalData.lookup(res, UVData.Baseline))
    # The scan's frequency/time lever arms travel with its detections: they set
    # the CRB uncertainty of a delay and a rate, which is what puts the station
    # solve's residuals in units of σ (see `Stationization`).
    det = Fringe._with_ti(
        res, first(win.ti_idx); epoch,
        freq_rms = Fringe._rms_spread(frequencies(stack)),
        time_rms = Fringe._rms_spread(timestamps(stack) .* 3600.0),
    )

    # Per-scan search log for the solution diagnostics: every MEASURED cell, its
    # family-wise PFA, and whether that PFA accepts it as a real fringe. Recording
    # the rejected cells too is what makes the near-threshold population visible;
    # `detected` is the column that separates them. The search cube is transient
    # (consumed by the station solve), so these are read off it here; `cells1` is
    # the only piece not already in the cube.
    cells1 = Fringe._search_cells(frequencies(stack), timestamps(stack) .* 3600.0, est.search)
    ncells = cells1 * max(length(bl_pairs) * length(pols), 1)
    pfa_max = est.closure.pfa_max
    local_solve = Fringe.scan_local_solve(est, s.model)
    ncomp, flags = 0, Tuple{Int, Int}[]
    # Steering needs θ for THIS scan, so it can only run where the station solve
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
            sd, sr = Fringe.scan_station_terms(ctx.model, ctx.layout, ctx.θ, ti)
            # θ is dense: a station this scan never constrained reads back as an
            # identity 0, indistinguishable from a solved zero delay. Steering to
            # it would invent a prediction and manufacture detections, so the
            # solve's own unconstrained list is what makes those nodes unusable.
            for (a, _) in flags
                (1 <= a <= size(sd, 1)) || continue
                sd[a, :] .= NaN
                sr[a, :] .= NaN
            end
            steer = Fringe.steer_scan(
                # The SAME epoch the search above referenced: `sr` is a rate
                # about it, as is the model's own Rate component.
                stack, res, bl_pairs, pols, ctx.stream.geom.f0,
                epoch * 3600.0, sd, sr;
                cells = est.steer_cells,
            )
        end
    end
    _st(field, j, p) = steer === nothing ? NaN : steer[field][j, p]
    rows = [
        (;
            a = bl_pairs[j][1], b = bl_pairs[j][2], pol = pols[p],
            snr = res.snr[j, p], pfa = res.pfa[j, p],
            delay = res.delay[j, p], rate = res.rate[j, p],
            phase = res.phase[j, p],
            detected = res.pfa[j, p] <= pfa_max,
            snr_steer = _st(:snr, j, p), pfa_steer = _st(:pfa, j, p),
            delay_steer = _st(:delay, j, p), rate_steer = _st(:rate, j, p),
            # Measured at the station solution's delay and rate rather than found
            # blind. There is NO threshold here: `pfa_max` decides fringe-group
            # membership on the blind pass, and once a station is in that group
            # its baselines are measured at the known fringe location to
            # arbitrarily low SNR. `pfa_steer` records the significance of what
            # was measured; it does not gate it.
            steered = res.pfa[j, p] > pfa_max && isfinite(_st(:snr, j, p)),
        )
            for p in eachindex(pols) for j in eachindex(bl_pairs) if res.valid[j, p]
    ]
    max_snr = isempty(rows) ? 0.0 : maximum((r.snr for r in rows if r.detected); init = 0.0)
    local_solve && return (; ncomp, flags, max_snr, ncells, rows)
    return (; det, max_snr, ncells, rows)
end

function Fringe.finish_estimate!(est::Fringe.MatchedFilter, ctx::SolveContext, s::FringeFit)
    results = ctx.scratch[:pass_results]
    ngroups = length(ctx.stream.groups)
    if Fringe.scan_local_solve(est, s.model)
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
    # `dets` is SOLVE-ESSENTIAL (it feeds the closure-screened WLS immediately
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
        ctx.θ, stack, win, setup.delay_plan, setup.disp_plan, ctx.gauge, ctx.nant;
        executor = inner_executor(ctx.stream), ties = setup.ties,
    )
    Fringe.refine_scan_sbd!(
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
    # The GLOBAL baseline table of the accumulation: every cross pair.
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    blidx = Dict(bl_pairs[i] => i for i in eachindex(bl_pairs))
    # `ctx.layout` holds only this step's own components, in phase-then-logamp
    # order, and `validate_bandpass_groups` capped each group at one — so the
    # plan list positions ARE the two observables, whatever the user named them.
    plans = layout.plans
    ctx.scratch[:bp_setup] = (;
        bl_pairs, blidx, nant,
        bp_plan = layout.nphase == 1 ? plans[1] : nothing,
        amp_plan = length(plans) == layout.nphase + 1 ? plans[end] : nothing,
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
        rl, wl, setup.blidx, stack, win; derotate = Fringe.bandpass_derotate(s.smoother),
    )
    return (; rl, wl, pols, source = source_name(stack))
end

function finish_pass!(s::Bandpass, ctx::SolveContext)
    setup = ctx.scratch[:bp_setup]
    results = ctx.scratch[:pass_results]
    isempty(results) && return (; nscans = 0)     # no scans → bandpass stays 0
    report = Fringe.solve_bandpass!(
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
        ctx.θ, stack, win, setup.adhoc_plan, s.smoother, ctx.gauge, ctx.nant;
        executor = inner_executor(ctx.stream),
    )
    return nothing
end

function finish_pass!(s::TemporalSmoother, ctx::SolveContext)
    results = ctx.scratch[:pass_results]
    return (; nscans = length(results))
end
