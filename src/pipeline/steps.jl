# ── Built-in solve steps of the composable pipeline ──────────────────────────
#
# The stages of the fringe pipeline as SolveSteps. Each declares model
# components / dependencies through the protocol hooks and implements the
# executor-driven visitor contract (start_pass!/process_scan!/finish_pass! —
# the runner in verbs.jl drives them). Every pipeline runs on this engine.

"""
    FringeFit(; model = FringeModel(), estimator = MatchedFilter())

The fringe-fitting stage. WHAT is solved is `model` ([`FringeModel`](@ref)):
the ordered phase-term list — per-scan constant/delay/rate, the R–L offsets
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
# NOTE: no `fit_selection` method — the fringe pass streams EVERY scan (the
# default `AllScans`); the estimator's `cross_hand_fit_on` masks cross-hand
# ROWS inside its solve, it does not restrict which scans are read.

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

"""
    BandpassEstimator(; phase = true, amp = true,
                      freq = ChannelBlocks(1),
                      amp_model = PenalizedBandpass(1.0),
                      select = AllScans())

The bandpass stage: the time-global phase / log-amplitude station bandpass
solved from the residual of whichever earlier steps have already applied
their gains, over the scans `select` picks (fit-on-subset / apply-everywhere:
a bandpass fit from a few bright calibrator scans still applies to the whole
track). `amp_model` is the amplitude-shape estimator ([`PenalizedBandpass`](@ref) /
[`PolynomialBandpass`](@ref) / [`FreeBandpass`](@ref)).

`freq` sets how finely the bandpass is resolved in frequency: the default
[`ChannelBlocks`](@ref)`(1)` is one free value per channel, and
`ChannelBlocks(k)` ties each `k` consecutive channels of a spw to one value —
fewer parameters, and each fit from `k` channels' worth of signal, for tracks
where the per-channel SNR will not support a free bandpass.

`select` accepts any [`AbstractScanSelection`](@ref) (`AllScans`, `SourceScans`,
`ScanIndices`, `ScanWhere`, or a custom one); the default runs over every scan.
The model is self-contained, so placing `BandpassEstimator` before or after
`FringeFit` is equally legal.
"""
Base.@kwdef struct BandpassEstimator <: SolveStep
    phase::Bool = true
    amp::Bool = true
    freq::ChannelBlocks = ChannelBlocks(1)
    amp_model::Fringe.AbstractBandpassSmoother = Fringe.PenalizedBandpass(1.0)
    select::Fringe.AbstractScanSelection = Fringe.AllScans()
end
provides(::BandpassEstimator) = :bandpass
required_grouping(::BandpassEstimator) = :scan_complete
# The pass streams the user's selection PLUS the coverage top-up: stations the
# selected scans never observe would get no bandpass (g = 1), so each such
# station's best scan is added (any source — safe for the SHAPE, see
# `Fringe.CoverageTopup`).
fit_selection(s::BandpassEstimator, prior_solutions) = Fringe.CoverageTopup(s.select)

"""
    TemporalSmoother(smoother = SavitzkyGolaySmoother(); pseudo_stokes = :auto)

The per-integration atmospheric-phase stage (adhoc phasing): solves the
globally-closing per-AP station phase on the fringe/bandpass residual, through
the pluggable [`AbstractAdhocSmoother`](@ref) (`SavitzkyGolaySmoother`,
`JointOUSmoother`, `OUSmoother`, `PenalizedSmoother`, …).

`pseudo_stokes` (`:auto`/`true`/`false`) collapses the four correlation
products to one pseudo-Stokes-I row per (baseline, AP) for the per-AP solve —
`:auto` enables it when the feeds are linear (X/Y).
"""
Base.@kwdef struct TemporalSmoother <: SolveStep
    smoother::Fringe.AbstractAdhocSmoother = Fringe.SavitzkyGolaySmoother()
    pseudo_stokes::Union{Bool, Symbol} = :auto
end
TemporalSmoother(smoother::Fringe.AbstractAdhocSmoother; pseudo_stokes = :auto) =
    TemporalSmoother(smoother, pseudo_stokes)
provides(::TemporalSmoother) = :adhoc
required_grouping(::TemporalSmoother) = :scan_complete

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
function model_components(s::BandpassEstimator, spec)
    bpc = TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), s.freq), PerFeed())
    return (; phase = s.phase ? (bandpass = bpc,) : (;), logamp = s.amp ? (bandpass = bpc,) : (;))
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
        Vsearch, ngroups = length(ctx.stream.groups), executor = ctx.stream.inner_executor,
    )
    pols = pol_products(stack)
    # `res` covers only the surviving (cross) baselines; take its own pair list.
    bl_pairs = collect(UVData.DimensionalData.lookup(res, UVData.Baseline))
    det = Fringe._with_ti(res, first(win.ti_idx))

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
    # `dets`/`scan_snr` are SOLVE-ESSENTIAL (drive `mask_unselected_cross_hands!`
    # and the closure-screened WLS immediately below) — `AllScans()` covers
    # every group every round, so a fresh build each round is exact, not just
    # an approximation of the old `get!`-persisted array.
    dets = Vector{Any}(undef, ngroups)
    for res in results
        dets[res.index] = res.r.det
    end
    scan_snr = scan_values(res -> res.r.max_snr, results, ngroups; default = 0.0)
    Fringe.mask_unselected_cross_hands!(dets, est.cross_hand_fit_on, ctx.stream.groups, scan_snr)
    # `ctx.model` holds only FringeFit's own components (each step solves on
    # its own private model/θ, never a merged one — CHUNK-069), so no
    # restriction is needed: a later step's component sharing a stage-B
    # signature by design (`DispersionSBDFit`'s private delay-refinement
    # column vs. this model's own wideband delay) lives in a SEPARATE model
    # and never appears here.
    stageB = Fringe.fringe_stage_components(ctx.model, ctx.layout)
    # An opted-in R–L rate is a feed-specific Rate component in THIS step's own
    # model (other steps contribute no rate terms) — cross-hand rows must then
    # join the rate system.
    rl_rate_on = Fringe._has_feed_rate(ctx.model.phase)
    opts = Fringe.resolve_closure(est, rl_rate_on)
    chi, ncomp, nrej, covered = Fringe.solve_station_systems!(
        ctx.θ, dets, stageB; ref_ant = ctx.ref_ant, opts = opts,
    )
    ctx.scratch[:fringe_flags] = Fringe.unconstrained_flags(dets, covered, ctx.geom)
    round = ctx.scratch[:fringe_round]::Int
    # Another round re-searches the residual.
    round < max(est.rounds, 1) && return (; repeat_pass = true, chi, ncomp, rejected = nrej)
    # `scan_snr` is published for a LATER step's non-data input (e.g.
    # `BandpassEstimator`'s SNR-aware `fit_selection` reads it off this step's
    # `StepSolution.info` — see `_scan_snr`). `scan_ncells` and the detection
    # table are pure logging (`Fringe.diagnostics.jl`'s
    # `fringe_snr_table`/`suspect_fringes` read them off this same
    # `StepSolution.info`) — built only now, on the final round, via the same
    # `scan_values` primitive every step's per-scan diagnostics use.
    return (;
        chi, ncomp, rejected = nrej, scan_snr,
        scan_ncells = scan_values(res -> res.r.ncells, results, ngroups; default = 0.0),
        Fringe.detection_table(
            scan_values(res -> res.r.rows, results, ngroups; default = Fringe.DetectionRow[]),
        )...,
    )
end

# ── DispersionSBDFit visitor (per-scan joint (Δτ, dTEC) fit + SBD fit) ────────

# Co-located stations see the same ionosphere, so a differential TEC between
# them is pure solve error — but only a model that solves dTEC has any to tie.
_dtec_ties(::Nothing, antennas) = nothing
_dtec_ties(dm::DispersionModel, antennas) =
    dm.tie_colocated ? UVData._colocated_ties(antennas) : nothing

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
    nrej = Fringe.refine_scan_dispersion!(
        ctx.θ, stack, win, setup.delay_plan, setup.disp_plan, ctx.ref_ant, ctx.nant;
        executor = ctx.stream.inner_executor, ties = setup.ties,
    )
    nrej += Fringe.refine_scan_sbd!(
        ctx.θ, stack, win, setup.sbd_plans, ctx.ref_ant, ctx.nant;
        executor = ctx.stream.inner_executor,
    )
    return (; nrej)
end

function finish_pass!(s::DispersionSBDFit, ctx::SolveContext)
    results = ctx.scratch[:pass_results]
    nrej = sum(res.r.nrej for res in results; init = 0)
    setup = ctx.scratch[:disp_sbd_setup]
    # This step's own compiled model alone says whether dTEC/SBD were fit —
    # published here (not derived later from the model by a name-hardcoded
    # lookup) so a solution-level consumer needs no knowledge of this step's
    # name to ask "was dispersion/SBD applied?".
    return (;
        nscans = length(results), dtec_rejected = nrej,
        dispersion_applied = setup.disp_plan !== nothing, sbd_applied = setup.sbd_plans !== nothing,
    )
end

# ── BandpassEstimator visitor (accumulate per scan → per-channel solves) ─────

function start_pass!(s::BandpassEstimator, ctx::SolveContext)
    model = ctx.model
    layout = ctx.layout
    nant = ctx.nant
    excl = ctx.scratch[:excl]
    # The GLOBAL baseline table of the accumulation (all cross pairs); co-located
    # (intra-site) pairs are excluded from `blidx` up front — their non-closing
    # crosstalk would otherwise pull the ~snr²-weighted per-channel solves.
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    blidx = Dict(
        bl_pairs[i] => i for i in eachindex(bl_pairs)
            if excl === nothing || bl_pairs[i] ∉ excl
    )
    ctx.scratch[:bp_setup] = (;
        bl_pairs, blidx,
        bp_plan = s.phase ? Fringe._bandpass_plan(model, layout) : nothing,
        amp_plan = s.amp ? Fringe._amp_bandpass_plan(model, layout) : nothing,
    )
    return nothing
end

function process_scan!(s::BandpassEstimator, ctx::SolveContext, stack, win::GeometryWindow)
    setup = ctx.scratch[:bp_setup]
    # `stack` arrives already fringe/dispersion/SBD-corrected through the
    # pipeline's transform chain (every earlier step's finished solution) —
    # this step just accumulates the residual, no correction of its own.
    pols = String.(pol_products(stack))
    nchan = length(ctx.geom.channel_freqs)
    rl, wl = Fringe.bandpass_accumulators(length(setup.bl_pairs), length(pols), nchan)
    Fringe.accumulate_bandpass!(rl, wl, setup.blidx, stack, win)
    return (; rl, wl, pols, source = source_name(stack))
end

function finish_pass!(s::BandpassEstimator, ctx::SolveContext)
    setup = ctx.scratch[:bp_setup]
    results = ctx.scratch[:pass_results]
    isempty(results) && return (; nscans = 0)     # selection empty → bandpass stays 0
    pols = results[1].r.pols
    nchan = length(ctx.geom.channel_freqs)
    nbl = length(setup.bl_pairs)
    rbar, wbar = Fringe.bandpass_accumulators(nbl, length(pols), nchan)
    for res in results                            # group-index order: the fold is
        rbar .+= res.r.rl                         # deterministic at ANY concurrency
        wbar .+= res.r.wl
    end
    setup.bp_plan === nothing || Fringe.solve_phase_bandpass!(
        ctx.θ, rbar, wbar, setup.bl_pairs, pols, ctx.nant, setup.bp_plan;
        ref_ant = ctx.ref_ant,
    )
    setup.amp_plan === nothing || Fringe.solve_amp_bandpass!(
        ctx.θ, rbar, wbar, setup.bl_pairs, pols, ctx.nant, setup.amp_plan,
        ctx.geom.channel_freqs;
        spw_of_chan = ctx.geom.spw_of_chan, smoother = s.amp_model,
    )
    scans = Int[res.index for res in results]
    return (;
        nscans = length(scans), scans,
        sources = unique(String[res.r.source for res in results]),
    )
end

# ── TemporalSmoother visitor (refine + per-scan adhoc solve; final pass) ──────

function start_pass!(s::TemporalSmoother, ctx::SolveContext)
    uv = ctx.stream.uvset
    first_leaf = first(values(UVData.branches(uv)))
    ctx.scratch[:adhoc_setup] = (;
        adhoc_plan = Fringe._adhoc_plan(ctx.model, ctx.layout),
        shared = Fringe._adhoc_shared(ctx.model),
        psI = Fringe._pseudo_stokes_config(
            s.pseudo_stokes, first_leaf, UVData.DimensionalData.metadata(uv),
        ),
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
        shared_feeds = setup.shared, executor = ctx.stream.inner_executor,
        excl = ctx.scratch[:excl], psI = setup.psI,
    )
    return nothing
end

function finish_pass!(s::TemporalSmoother, ctx::SolveContext)
    results = ctx.scratch[:pass_results]
    return (; nscans = length(results))
end
