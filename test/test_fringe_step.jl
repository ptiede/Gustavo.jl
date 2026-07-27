# ── The FringeFit step (stage A on the composable engine) ─────────────────────
#
# A FringeFit-only pipeline solves the matched-filter stage standalone. The M3
# bit-parity gates against the frozen monolith ran before its deletion; the
# standing invariant kept here is that LATER STAGES NEVER MOVE THE FRINGE
# BLOCKS: a fringe-only fit's θ blocks are bit-identical to the same blocks of
# a fuller pipeline (dispersion/SBD off, so no later stage refines the compared
# slots). Plus the step's capabilities: the opt-in cross-feed rate solve,
# fit-on-subset cross-hand masking, and transforms on the streaming path.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "FringeFit step (new engine)" begin
    @testset "fringe blocks invariant under later stages" begin
        uvset, _ = _build_fringe_uvset()
        solm = fit(
            FringeFit(model = FringeModel(ref_ant = 1, sbd = false), dispersion = nothing) |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        sol = fit(FringeFit(model = FringeModel(sbd = false), dispersion = nothing), uvset)
        @test length(sol.model.phase) == 5
        rn = CAL.component_ranges(sol.layout)
        rm = CAL.component_ranges(solm.layout)
        for i in 1:5
            @test sol.θ[rn[i]] == solm.θ[rm[i]]        # bit-identical
        end
        @test stage_names(sol) == [:fringe]
        @test sol.info.nscan == solm.info.nscan
        @test sol.info.scan_max_snr == solm.info.scan_max_snr
        @test sol.info.scan_ncells == solm.info.scan_ncells
        @test sol.info.det_snr == solm.info.det_snr
        @test sol.info.det_pfa == solm.info.det_pfa

        # ref_ant as a station code resolves identically.
        sol_code = fit(FringeFit(model = FringeModel(ref_ant = "A1", sbd = false), dispersion = nothing), uvset)
        @test sol_code.θ == sol.θ

        # The snapshot machinery works on a single-stage solution.
        @test CAL.stage_solution(sol[:fringe]).θ == sol.θ

        # A fringe-only solution applies cleanly.
        corr = UVP.apply_calibration(uvset, sol)
        @test corr isa UVP.UVSet
    end

    @testset "rounds > 1: fringe blocks invariant under later stages" begin
        uvset, _ = _build_fringe_uvset()
        solm = fit(
            FringeFit(
                model = FringeModel(ref_ant = 1, sbd = false),
                dispersion = nothing,
                estimator = MatchedFilter(rounds = 2),
            ) |> TemporalSmoother(FP.SavitzkyGolaySmoother(window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        sol = fit(
            FringeFit(
                model = FringeModel(sbd = false),
                dispersion = nothing,
                estimator = MatchedFilter(rounds = 2),
            ), uvset,
        )
        rn = CAL.component_ranges(sol.layout)
        rm = CAL.component_ranges(solm.layout)
        for i in 1:5
            @test sol.θ[rn[i]] == solm.θ[rm[i]]
        end
    end

    @testset "opt-in cross-feed rate (CrossFeed.rate)" begin
        inj = [0.0, 2.0e-4, -1.0e-4, 5.0e-5]
        uvset, _ = _build_fringe_uvset(rl_rate = inj)
        mk(rate) = FringeModel(sbd = false, cross_feed = CrossFeed(rate = rate))

        sol = fit(FringeFit(model = mk(Gustavo.GlobalTime()), dispersion = nothing), uvset)
        @test length(sol.model.phase) == 6
        plan = sol.layout.plans[6]
        solved = [sol.θ[plan.off1[a, 2, 1, 1]] for a in 1:4]
        @test solved ≈ inj .- inj[1] atol = 1.0e-7

        # Null case: no injected feed-rate offset → solved offsets ≈ 0.
        uv0, _ = _build_fringe_uvset()
        sol0 = fit(FringeFit(model = mk(Gustavo.GlobalTime()), dispersion = nothing), uv0)
        plan0 = sol0.layout.plans[6]
        @test maximum(abs, [sol0.θ[plan0.off1[a, 2, 1, 1]] for a in 1:4]) < 1.0e-7

        # Default (rate = nothing): the component does not exist — tied ≡ 0.
        sold = fit(FringeFit(model = FringeModel(sbd = false), dispersion = nothing), uvset)
        @test length(sold.model.phase) == 5
    end

    @testset "CrossFeed.fit_on masks cross-hand rows" begin
        uvset, _ = _build_fringe_uvset()      # per-feed delay/phi ⇒ real R–L offset
        base = fit(FringeFit(model = FringeModel(sbd = false), dispersion = nothing), uvset)
        # The global feed-2 delay offset (component 4) is solved.
        @test any(!iszero, base.θ[CAL.component_ranges(base.layout)[4]])

        # Selecting the (only) scan is a no-op: bit-identical to AllScans.
        same = fit(
            FringeFit(
                model = FringeModel(
                    sbd = false,
                    cross_feed = CrossFeed(fit_on = Gustavo.ScanIndices(1)),
                ),
                dispersion = nothing,
            ), uvset,
        )
        @test same.θ == base.θ

        # Deselecting every scan removes the feed-tying cross rows: the solve
        # still succeeds (feed 2 gets its own gauge pin; the QQ parallel rows
        # keep its per-station columns constrained) and the feed-common delay
        # is unperturbed beyond solver precision on this noiseless synthetic.
        masked = fit(
            FringeFit(
                model = FringeModel(
                    sbd = false,
                    cross_feed = CrossFeed(fit_on = Gustavo.ScanIndices(10_000)),
                ),
                dispersion = nothing,
            ), uvset,
        )
        @test all(isfinite, masked.θ)
        @test masked.θ[CAL.component_ranges(masked.layout)[3]] ≈
            base.θ[CAL.component_ranges(base.layout)[3]] atol = 1.0e-12

        # The masker's direct contract: with an empty selection every
        # CROSS-HAND detection is invalidated, parallel hands untouched.
        geom = CAL.build_geometry(uvset)
        st = FP.scan_stream(uvset; geom = geom, workspace = FP.FringeWorkspace)
        grp = FP.materialize_cube(st, st.groups[1])
        res = FP.search_scan(st, grp, FP.FringeSearch())
        feeds = [CAL.correlation_feed_pair(p) for p in grp.pol_products]
        d = FP.StationScanDetections(copy(res.det), grp.bl_pairs, feeds, first(grp.g_ti))
        FP.mask_unselected_cross_hands!([d], Gustavo.ScanIndices(10_000), st.groups, [NaN])
        for p in eachindex(feeds), bi in axes(d.det, 1)
            fa, fb = feeds[p]
            if fa != fb
                @test !d.det[bi, p].valid
            else
                @test d.det[bi, p] === res.det[bi, p]
            end
        end
    end

    @testset "transforms on the new path (incl. CalFunction)" begin
        uvset, _ = _build_fringe_uvset()
        ws = [1.0, 0.5, 1.0, 2.0]
        # Weight scale: the search is invariant (snr from the |D|² plane), so
        # the fringe θ matches the untransformed solve bit-for-bit.
        sol_ws = fit(StationWeightScale(ws) |> FringeFit(model = FringeModel(sbd = false), dispersion = nothing), uvset)
        sol = fit(FringeFit(model = FringeModel(sbd = false), dispersion = nothing), uvset)
        @test sol_ws.θ == sol.θ
        @test length(sol_ws.transforms) == 1 && sol_ws.transforms[1] isa StationWeightScale

        # CalFunction runs on the new path (it errors only when bridging), and
        # is recorded + replayed by calibrate: zeroing one baseline's weights
        # zero-weights it in the calibrated output.
        touched = Threads.Atomic{Int}(0)
        kill12 = CalFunction() do v
            Threads.atomic_add!(touched, 1)
            for (bi, (a, b)) in enumerate(v.bl_pairs)
                minmax(a, b) == (1, 2) && (v.weights[:, :, bi, :] .= 0)
            end
        end
        sol_cf, out = fitcalibrate(kill12 |> FringeFit(model = FringeModel(sbd = false), dispersion = nothing), uvset)
        @test touched[] > 0
        @test sol_cf.transforms[1] isa CalFunction
        for (_, leaf) in DimensionalData.branches(out)
            W = parent(leaf[:weights])
            for (bi, (a, b)) in enumerate(UVP.baselines(leaf).pairs)
                minmax(a, b) == (1, 2) && @test all(W[:, :, bi, :] .== 0)
            end
        end

        # fit + calibrate ≡ fitcalibrate on the new path.
        out2 = calibrate(sol_cf, uvset)
        for (k, leaf) in DimensionalData.branches(out)
            V = parent(leaf[:vis])
            V2 = parent(DimensionalData.branches(out2)[k][:vis])
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x == y, zip(V, V2))
        end
    end

    @testset "model validation + full-pipeline option coverage" begin
        uvset, _ = _build_fringe_uvset()
        @test_throws ErrorException fit(
            FringeFit(model = FringeModel(delay = Gustavo.GlobalTime())), uvset)
        @test_throws ErrorException CrossFeed(delay = Gustavo.PerIntegration())
        @test_throws ErrorException CrossFeed(rate = Gustavo.PerIntegration())
        # `dispersion` is no longer a FringeModel field — it is the step's own
        # propagation model, so the instrument model rejects the keyword outright
        # rather than validating a value domain.
        @test_throws MethodError FringeModel(dispersion = :maybe)
        @test :dispersion ∉ fieldnames(FringeModel)
        @test :dispersion ∈ fieldnames(typeof(FringeFit()))

        # The options the legacy bridge used to reject (custom Stationization,
        # the R–L rate opt-in, fit_on subsetting, arbitrary CalFunction
        # transforms) run in FULL pipelines now — every pipeline is new-engine.
        sol_full = fit(
            CalibrationPipeline(
                CalFunction(v -> nothing),
                FringeFit(
                    model = FringeModel(cross_feed = CrossFeed(
                        rate = Gustavo.GlobalTime(), fit_on = Gustavo.ScanIndices(1))),
                    estimator = MatchedFilter(closure = FP.Stationization(snr_min = 3.0)),
                ),
                BandpassEstimator(), TemporalSmoother();
                exec = ExecutionConfig(ntasks = 1),
            ),
            uvset,
        )
        @test Gustavo.stage_names(sol_full) == [:fringe, :bandpass, :adhoc]
        # BandpassEstimator without TemporalSmoother still solves a :bandpass
        # stage (F |> B — no final pass).
        sol_fb = fit(CalibrationPipeline(FringeFit(), BandpassEstimator()), uvset)
        @test any(r -> r.name === :bandpass, sol_fb.stages)
    end
end

# ── The estimator seam, exercised from outside the package ───────────────────
#
# `AbstractFringeEstimator` is a supported extension point, so the proof is an
# estimator defined HERE — not in `src/` — driven all the way through `fit`.

# Delegates both hooks to a MatchedFilter it wraps, counting the calls. Anything
# it gets wrong shows up as a θ difference against the same fit run directly.
struct _ProbeEstimator{E <: FP.AbstractFringeEstimator} <: FP.AbstractFringeEstimator
    inner::E
    scans::Base.RefValue{Int}
    passes::Base.RefValue{Int}
end
_ProbeEstimator(inner) = _ProbeEstimator(inner, Ref(0), Ref(0))

function FP.estimate_scan!(e::_ProbeEstimator, ctx, step, view)
    e.scans[] += 1
    return FP.estimate_scan!(e.inner, ctx, step, view)
end
function FP.finish_estimate!(e::_ProbeEstimator, ctx, step)
    e.passes[] += 1
    return FP.finish_estimate!(e.inner, ctx, step)
end

# Implements neither hook: must fail loudly rather than solve nothing.
struct _SilentEstimator <: FP.AbstractFringeEstimator end

# The independence probe: implements the interface and NOTHING else. It writes no
# θ and publishes none of the matched filter's diagnostic scratch tables, so it
# fails if any of them is secretly required to assemble a solution.
struct _NullEstimator <: FP.AbstractFringeEstimator
    scans::Base.RefValue{Int}
end
_NullEstimator() = _NullEstimator(Ref(0))
function FP.estimate_scan!(e::_NullEstimator, ctx, step, view)
    e.scans[] += 1
    return (; max_snr = NaN)
end
FP.finish_estimate!(::_NullEstimator, ctx, step) = (; chi = 0.0, ncomp = 0, rejected = 0)

@testset "fringe estimator seam" begin
    uvset, _ = _build_fringe_uvset()
    model = FringeModel(ref_ant = 1)

    @testset "an out-of-package estimator drives the whole pipeline" begin
        probe = _ProbeEstimator(MatchedFilter())
        sol = fit(
            FringeFit(; model, estimator = probe) |> BandpassEstimator() |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        ref = fit(
            FringeFit(; model) |> BandpassEstimator() |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        # Bit-identical, not approximate: the seam must not perturb the solve.
        @test sol.θ == ref.θ
        @test stage_names(sol) == stage_names(ref)
        @test probe.scans[] == sol.info.nscan
        @test probe.passes[] == 1
    end

    @testset "the step's refine service reaches an out-of-package estimator" begin
        # Downstream stages depend on it, so a third-party estimator must get it
        # without publishing it itself.
        probe = _ProbeEstimator(MatchedFilter())
        sol = fit(
            FringeFit(; model = FringeModel(ref_ant = 1, sbd = true), estimator = probe) |>
                BandpassEstimator(),
            uvset,
        )
        @test any(r -> r.name === :bandpass, sol.stages)
    end

    @testset "an estimator publishing no diagnostics still yields a solution" begin
        null = _NullEstimator()
        sol = fit(FringeFit(; model, estimator = null), uvset)
        @test null.scans[] == sol.info.nscan
        @test all(iszero, sol.θ)                 # it solved nothing, by construction
        @test isempty(sol.info.det_snr)          # no detections reported
        @test isempty(sol.info.flagged_ant)
        @test all(isnan, sol.info.scan_max_snr) || all(iszero, sol.info.scan_max_snr)
        # `search` is MatchedFilter provenance, so this solution carries none.
        @test !haskey(sol.info, :search)
        @test haskey(fit(FringeFit(; model), uvset).info, :search)
    end

    @testset "an estimator implementing neither hook errors by name" begin
        @test_throws "does not implement the fringe estimator interface" fit(
            FringeFit(; model, estimator = _SilentEstimator()), uvset,
        )
        @test_throws "estimate_scan!" fit(
            FringeFit(; model, estimator = _SilentEstimator()), uvset,
        )
    end

    @testset "the estimator is carried as a type parameter, not an abstract field" begin
        # Removing the old `::MatchedFilter` assertion would otherwise put a
        # dynamic dispatch in the per-scan path.
        @test isconcretetype(fieldtype(typeof(FringeFit()), :estimator))
        @test fieldtype(typeof(FringeFit(estimator = _SilentEstimator())), :estimator) ===
            _SilentEstimator
    end
end

# ── Dispersion as a model of its own ─────────────────────────────────────────
#
# The ionosphere is specified separately from the instrument, but ESTIMATED
# jointly with the delay it is degenerate with — these assert both halves.
@testset "dispersion is a separate model" begin
    # A VGOS-like layout: four sub-bands over a wide fractional bandwidth, which
    # is what lets 1/ν be separated from a linear delay at all.
    uvset, _ = _build_fringe_uvset(
        nbands = 4, nchan = 8, dtec = [0.0, 3.0, -2.0, 1.5],
        band_origins = [3.0e9, 5.0e9, 8.0e9, 1.03e10], feed_common = true,
    )
    mf = FP.MatchedFilter(search = FP.FringeSearch(algorithm = FP.FullGrid()))

    @testset "the instrument model no longer carries the ionosphere" begin
        @test :dispersion ∉ fieldnames(FringeModel)
        @test :dtec_tie_colocated ∉ fieldnames(FringeModel)
        @test :sbd ∈ fieldnames(FringeModel)      # instrumental: stays put
        @test fieldnames(DispersionModel) == (:require_band_separation, :tie_colocated)
    end

    @testset "the propagation model is Calibration's, not Fringe's" begin
        # An ionosphere is modelled without loading the fringe-fitting module:
        # the spec sits beside the `Dispersion` term it configures, and only the
        # joint (Δτ, dTEC) estimator stays in `Fringe`.
        @test parentmodule(DispersionModel) === Gustavo.Calibration
        @test which(CAL._dispersion_enabled, Tuple{Nothing, CAL.DataGeometry}).module ===
            Gustavo.Calibration
        # Re-exported, so `FP.DispersionModel` and a bare `using Gustavo` name
        # the SAME type rather than a shadowing second one.
        @test FP.DispersionModel === CAL.DispersionModel === Gustavo.DispersionModel
        @test !isdefined(FP, :_dispersion_plan)
        @test FP._dispersion_enabled === CAL._dispersion_enabled
        # Co-location is array geometry with two unrelated consumers — the dTEC
        # tie and the intra-site baseline exclusion — so it belongs to UVData.
        @test parentmodule(UVP._colocated_ties) === Gustavo.UVData
        @test parentmodule(UVP._colocated_pair_set) === Gustavo.UVData
        @test !isdefined(FP, :_colocated_ties)
    end

    @testset "the step decides whether an ionosphere is modelled at all" begin
        # Whether the term is SOLVED end to end is `info.dispersion_applied`, which
        # additionally needs a stage that refines it (see the dispersion testset in
        # test_pipeline.jl). What this asserts is the model structure the step built.
        on = fit(FringeFit(model = FringeModel(ref_ant = 1), estimator = mf), uvset)
        off = fit(FringeFit(model = FringeModel(ref_ant = 1), dispersion = nothing,
                estimator = mf), uvset)
        @test CAL._dispersion_plan(on.model, on.layout) !== nothing
        @test CAL._dispersion_plan(off.model, off.layout) === nothing
        @test any(tc -> tc.component.term isa CAL.Dispersion, on.model.phase)
        @test !any(tc -> tc.component.term isa CAL.Dispersion, off.model.phase)
        # No dTEC term means no dTEC columns in θ at all.
        @test length(off.θ) < length(on.θ)
    end

    @testset "require_band_separation gates on the band layout" begin
        # A single contiguous band cannot constrain the 1/ν curvature.
        narrow, _ = _build_fringe_uvset(nbands = 1, nchan = 8)
        geom_n = CAL.build_geometry(narrow)
        geom_w = CAL.build_geometry(uvset)
        @test !CAL._dispersion_enabled(DispersionModel(), geom_n)
        @test CAL._dispersion_enabled(DispersionModel(), geom_w)
        # Forcing it on solves the term regardless of what the layout supports.
        @test CAL._dispersion_enabled(DispersionModel(require_band_separation = false), geom_n)
        @test !CAL._dispersion_enabled(nothing, geom_n)
    end

    @testset "tie_colocated reaches the refine service through the step" begin
        # The tie is the dispersion model's, but RefineService is the step's to
        # publish, so it must arrive without the estimator knowing about it.
        ants = Gustavo.UVData.metadata(
            first(values(Gustavo.UVData.branches(uvset)))).antennas
        @test Gustavo._dtec_ties(DispersionModel(tie_colocated = true), ants) !== nothing
        @test Gustavo._dtec_ties(DispersionModel(tie_colocated = false), ants) === nothing
        @test Gustavo._dtec_ties(nothing, ants) === nothing
    end

    @testset "delay and dTEC are still estimated jointly" begin
        # The separation is of the specification only: one RefineService carries
        # both plans, because over a finite band the two are near-degenerate.
        sol = fit(FringeFit(model = FringeModel(ref_ant = 1), estimator = mf), uvset)
        rf = FP.RefineService(
            CAL._dispersion_plan(sol.model, sol.layout),
            FP._perscan_delay_plan(sol.model, sol.layout),
            FP._sbd_plans(sol.model, sol.layout), nothing, true, 20.0,
        )
        @test rf.disp_plan !== nothing
        @test rf.ps_delay_plan !== nothing
    end
end
