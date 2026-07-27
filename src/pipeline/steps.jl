# ── Built-in solve steps of the composable pipeline ──────────────────────────
#
# The three stages of the fringe pipeline as SolveSteps. Each declares model
# components / dependencies through the protocol hooks and implements the
# executor-driven visitor contract (start_pass!/process_scan!/finish_pass! —
# the runner in verbs.jl drives them). Every pipeline runs on this engine.

"""
    FringeFit(; model = FringeModel(), estimator = MatchedFilter(),
              reuse_bandpass_refine = true, polish_dtec = 20.0)

The fringe-fitting stage: solves the model components declared by
[`FringeModel`](@ref) (per-scan delay/rate/phase, the R–L selection, optional
dispersion/SBD) with a pluggable [`AbstractFringeEstimator`](@ref) — by default
[`MatchedFilter`](@ref) (per-baseline delay/rate search + closure-screened
station WLS). WHAT is solved lives on `model` (including the `ref_ant` gauge
pin); HOW on `estimator` (including its `search`, `Stationization`, and
residual `rounds`).

The dTEC/SBD θ slots are owned by this step; `reuse_bandpass_refine` /
`polish_dtec` control how later stages reuse or polish its per-scan
refinements (see the legacy solver's kwargs of the same names).
"""
Base.@kwdef struct FringeFit{E <: Fringe.AbstractFringeEstimator} <: SolveStep
    model::Fringe.FringeModel = Fringe.FringeModel()
    estimator::E = Fringe.MatchedFilter()
    reuse_bandpass_refine::Bool = true
    polish_dtec::Float64 = 20.0
end
provides(::FringeFit) = :fringe
required_grouping(::FringeFit) = :scan_complete
# NOTE: no `fit_selection` method — the fringe pass streams EVERY scan (the
# default `AllScans`); `model.cross_feed.fit_on` masks cross-hand ROWS inside
# the estimator's solve, it does not restrict which scans are read.

"""
    BandpassEstimator(; phase = true, amp = true,
                      amp_model = PenalizedBandpass(1.0),
                      select = BrightestCalibrator())

The bandpass stage: per-channel phase / log-amplitude station bandpass solved
from the fringe-corrected residual of the scans `select` picks
(fit-on-subset / apply-everywhere: the time-global bandpass fit from a few
bright calibrator scans applies to the whole track). `amp_model` is the
amplitude-shape estimator ([`PenalizedBandpass`](@ref) /
[`PolynomialBandpass`](@ref) / [`FreeBandpass`](@ref)).

`select` accepts any [`AbstractScanSelection`](@ref); the default
[`BrightestCalibrator`](@ref) reproduces the legacy total-SNR calibrator pick
(`BrightestCalibrator(max_scans = k)` caps the accumulation to the k
highest-SNR scans).
"""
Base.@kwdef struct BandpassEstimator <: SolveStep
    phase::Bool = true
    amp::Bool = true
    amp_model::Fringe.AbstractBandpassSmoother = Fringe.PenalizedBandpass(1.0)
    select::Fringe.AbstractScanSelection = Fringe.BrightestCalibrator()
end
provides(::BandpassEstimator) = :bandpass
requires(::BandpassEstimator) = (:fringe,)
required_grouping(::BandpassEstimator) = :scan_complete
# The pass streams the user's selection PLUS the coverage top-up: stations the
# selected scans never observe would get no bandpass (g = 1), so each such
# station's best scan is added (any source — safe for the SHAPE, see
# `Fringe.CoverageTopup`).
fit_selection(s::BandpassEstimator) = Fringe.CoverageTopup(s.select)

"""
    TemporalSmoother(smoother = SavitzkyGolaySmoother(); pseudo_stokes = :auto)

The per-integration atmospheric-phase stage (adhoc phasing): solves the
globally-closing per-AP station phase on the fringe/bandpass residual, through
the pluggable [`AbstractAdhocSmoother`](@ref) (`SavitzkyGolaySmoother`,
`JointOUSmoother`, `OUSmoother`, `PenalizedSmoother`, …). Its pass also
re-refines each scan's FringeFit-owned dTEC/SBD columns on the
bandpass-corrected residual before the per-AP solve (scans the bandpass stage
already fit are polished in a narrow window — see `FringeFit`'s
`reuse_bandpass_refine`/`polish_dtec`).

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
requires(::TemporalSmoother) = (:fringe,)
required_grouping(::TemporalSmoother) = :scan_complete

# Solve steps run through the pipeline verbs, never the sequential
# `run_step` chain (they need the shared compiled model + streaming passes).
run_step(s::SolveStep, ctx::CalibrationContext) = error(
    "$(nameof(typeof(s))) is a SolveStep — run it through `fit`/`fitcalibrate`, " *
        "not step-by-step."
)

# ── Model components (compiled in step order into ONE StationGainModel) ───────

model_components(s::FringeFit, spec) =
    (; phase = Fringe.fringe_phase_components(s.model, spec.geom), logamp = ())

# The per-channel bandpass components: phase and log-amp, per feed, time-stable
# (the legacy `_fringe_model` placement — after the fringe terms).
function model_components(s::BandpassEstimator, spec)
    bpc = TiedComponent(GainComponent(PerChannel(), GlobalTime(), GlobalFrequency()), PerFeed())
    return (; phase = s.phase ? (bpc,) : (), logamp = s.amp ? (bpc,) : ())
end

# The per-integration adhoc phase: per-AP, feed-common, solved per scan by the
# temporal-smoother pass (the legacy `_fringe_model` placement — last).
model_components(s::TemporalSmoother, spec) = (;
    phase = (TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), SharedFeeds()),),
    logamp = (),
)

# ── FringeFit visitor (stage A: per-scan search → one global station solve) ───

function start_pass!(s::FringeFit, ctx::SolveContext)
    ctx.scratch[:fringe_round] = get(ctx.scratch, :fringe_round, 0) + 1
    return nothing
end

process_scan!(s::FringeFit, ctx::SolveContext, v::Fringe.ScanDataView) =
    Fringe.estimate_scan!(s.estimator, ctx, s, v)

function finish_pass!(s::FringeFit, ctx::SolveContext)
    info = Fringe.finish_estimate!(s.estimator, ctx, s)
    # The refine service describes the per-scan θ columns THIS STEP owns, so it
    # is the step's to publish and every estimator gets it. A pass that repeats
    # has not finished writing those columns yet.
    get(info, :repeat_pass, false) && return info
    fm = s.model
    ctx.scratch[:refine] = Fringe.RefineService(
        Fringe._dispersion_plan(ctx.model, ctx.layout),
        Fringe._perscan_delay_plan(ctx.model, ctx.layout),
        Fringe._sbd_plans(ctx.model, ctx.layout),
        fm.dtec_tie_colocated ? Fringe._colocated_ties(ctx.antennas) : nothing,
        s.reuse_bandpass_refine, s.polish_dtec,
    )
    return info
end

# ── MatchedFilter: the per-baseline search + closure-screened station WLS ─────
#
# Defined qualified on the `Fringe` generics, not as bare `estimate_scan!`,
# which would mint a second function here and leave the seam's fallback in place.

Fringe.estimator_info(est::Fringe.MatchedFilter) = (; search = est.search)

function Fringe.estimate_scan!(
        est::Fringe.MatchedFilter, ctx::SolveContext, s::FringeFit, v::Fringe.ScanDataView,
    )
    round = ctx.scratch[:fringe_round]::Int
    Vsearch = round > 1 ? Fringe.residual_vis(ctx.ev, ctx.θ, v) : v.vis
    res = Fringe.search_scan(ctx.stream, v, est.search; Vsearch = Vsearch)
    feeds = [correlation_feed_pair(p) for p in v.pol_products]
    det = Fringe.StationScanDetections(res.det, v.bl_pairs, feeds, first(v.ti_idx))
    return (; det, max_snr = res.max_snr, ncells = res.ncells, rows = res.rows)
end

function Fringe.finish_estimate!(est::Fringe.MatchedFilter, ctx::SolveContext, s::FringeFit)
    fm = s.model
    results = ctx.scratch[:pass_results]
    ngroups = length(ctx.stream.groups)
    scan_snr = get!(() -> zeros(ngroups), ctx.scratch, :scan_snr)::Vector{Float64}
    scan_ncells = get!(() -> zeros(ngroups), ctx.scratch, :scan_ncells)::Vector{Float64}
    scan_dets = get!(() -> [Fringe.DetectionRow[] for _ in 1:ngroups], ctx.scratch, :scan_dets)
    scan_t_decode = get!(() -> zeros(ngroups), ctx.scratch, :scan_t_decode)::Vector{Float64}
    scan_t_search = get!(() -> zeros(ngroups), ctx.scratch, :scan_t_search)::Vector{Float64}
    dets = Vector{Any}(undef, ngroups)
    for res in results
        gi = res.index
        r = res.r
        dets[gi] = r.det
        scan_snr[gi] = r.max_snr
        scan_ncells[gi] = r.ncells
        scan_dets[gi] = r.rows
        scan_t_decode[gi] += res.decode
        scan_t_search[gi] += res.work
    end
    Fringe.mask_unselected_cross_hands!(dets, fm.cross_feed.fit_on, ctx.stream.groups, scan_snr)
    stageB = Fringe.fringe_stage_components(ctx.model, ctx.layout)
    opts = Fringe.resolve_closure(est, fm.cross_feed.rate !== nothing)
    chi, ncomp, nrej, covered = Fringe.solve_station_systems!(
        ctx.θ, dets, stageB; ref_ant = ctx.ref_ant, opts = opts,
    )
    ctx.scratch[:fringe_flags] = Fringe.unconstrained_flags(dets, covered, ctx.geom)
    round = ctx.scratch[:fringe_round]::Int
    # Another round re-searches the residual; the step holds back its refine
    # service until the last one.
    round < max(est.rounds, 1) && return (; repeat_pass = true, chi, ncomp, rejected = nrej)
    return (; chi, ncomp, rejected = nrej)
end

# ── BandpassEstimator visitor (refine + accumulate per scan → per-channel solves) ──

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

function process_scan!(s::BandpassEstimator, ctx::SolveContext, v::Fringe.ScanDataView)
    setup = ctx.scratch[:bp_setup]
    # Dispersion/SBD-correct THIS scan before accumulating (FringeFit's refine
    # service): scans with different ionospheres would otherwise decohere the
    # frozen per-channel curve. Writes only this scan's θ columns — disjoint
    # under the concurrent group tasks.
    rf = get(ctx.scratch, :refine, nothing)::Union{Nothing, Fringe.RefineService}
    nrej = rf === nothing ? 0 :
        Fringe.refine_scan!(
        ctx.θ, v, ctx.geom, ctx.ev, rf, ctx.ref_ant, ctx.nant;
        inner = ctx.stream.inner,
    )
    pols = String.(v.pol_products)
    nchan = length(ctx.geom.channel_freqs)
    rl, wl = Fringe.bandpass_accumulators(length(setup.bl_pairs), length(pols), nchan)
    Fringe.accumulate_bandpass!(rl, wl, setup.blidx, ctx.ev, ctx.θ, v)
    return (; rl, wl, nrej, pols, source = v.source, t0 = first(v.ti_idx))
end

function finish_pass!(s::BandpassEstimator, ctx::SolveContext)
    setup = ctx.scratch[:bp_setup]
    results = ctx.scratch[:pass_results]
    isempty(results) && return (; nscans = 0)     # selection empty → bandpass stays 0
    pols = results[1].r.pols
    nchan = length(ctx.geom.channel_freqs)
    nbl = length(setup.bl_pairs)
    rbar, wbar = Fringe.bandpass_accumulators(nbl, length(pols), nchan)
    nrej = 0
    for res in results                            # group-index order: the fold is
        rbar .+= res.r.rl                         # deterministic at ANY concurrency
        wbar .+= res.r.wl
        nrej += res.r.nrej
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
    # The scans whose per-scan dTEC/SBD θ columns were refined here — the
    # temporal-smoother stage polishes these in a narrow window instead of
    # re-fitting (`reuse_bandpass_refine`). `refined_t0` records them by the
    # scan's first global time index (the identity a `process_scan!` can read
    # off its view), `refined_scans` by stream group index (diagnostics).
    ctx.scratch[:refined_scans] = Set(scans)
    ctx.scratch[:refined_t0] = Set(Int[res.r.t0 for res in results])
    ctx.scratch[:bp_dtec_rejected] = nrej
    return (;
        nscans = length(scans), scans,
        sources = unique(String[res.r.source for res in results]),
        dtec_rejected = nrej,
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

function process_scan!(s::TemporalSmoother, ctx::SolveContext, v::Fringe.ScanDataView)
    setup = ctx.scratch[:adhoc_setup]
    # Per-scan (Δτ, dTEC) + SBD refinement BEFORE the adhoc solve, so the per-AP
    # phases fit dispersion-corrected residuals (FringeFit's refine service —
    # writes only this scan's θ columns, disjoint under the concurrent group
    # tasks). Scans the bandpass stage already fit are POLISHED in a narrow
    # window (cheap) rather than re-run with the full grid search — keeps the
    # weak-source increment a full skip would drop.
    rf = get(ctx.scratch, :refine, nothing)::Union{Nothing, Fringe.RefineService}
    nrej = 0
    if rf !== nothing
        polish = rf.reuse_bandpass &&
            first(v.ti_idx) in get(() -> Set{Int}(), ctx.scratch, :refined_t0)
        nrej = Fringe.refine_scan!(
            ctx.θ, v, ctx.geom, ctx.ev, rf, ctx.ref_ant, ctx.nant;
            inner = ctx.stream.inner, polish = polish,
        )
    end
    Fringe.adhoc_scan!(
        ctx.θ, v, ctx.geom, ctx.ev, setup.adhoc_plan, s.smoother, ctx.ref_ant, ctx.nant;
        shared_feeds = setup.shared, inner = ctx.stream.inner,
        excl = ctx.scratch[:excl], psI = setup.psI,
    )
    return (; nrej)
end

function finish_pass!(s::TemporalSmoother, ctx::SolveContext)
    results = ctx.scratch[:pass_results]
    ngroups = length(ctx.stream.groups)
    scan_t_decode2 = get!(() -> zeros(ngroups), ctx.scratch, :scan_t_decode2)::Vector{Float64}
    scan_t_adhoc = get!(() -> zeros(ngroups), ctx.scratch, :scan_t_adhoc)::Vector{Float64}
    nrej = 0
    for res in results
        scan_t_decode2[res.index] = res.decode
        scan_t_adhoc[res.index] = res.work
        nrej += res.r.nrej
    end
    ctx.scratch[:adhoc_dtec_rejected] = nrej
    return (; nscans = length(results), dtec_rejected = nrej)
end
