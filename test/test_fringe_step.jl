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
            FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))) |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        sol = fit(FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))), uvset)
        fr, frm = CAL._step(sol, :fringe), CAL._step(solm, :fringe)
        @test length(fr.model.phase) == 5
        rn = CAL.component_ranges(fr.layout)
        rm = CAL.component_ranges(frm.layout)
        for i in 1:5
            @test fr.θ[rn[i]] == frm.θ[rm[i]]        # bit-identical
        end
        @test stage_names(sol) == [:fringe]
        @test sol.info.nscan == solm.info.nscan
        @test fr.info.scan_snr == frm.info.scan_snr
        @test fr.info.scan_ncells == frm.info.scan_ncells
        @test fr.info.det_snr == frm.info.det_snr
        @test fr.info.det_pfa == frm.info.det_pfa

        # ref_ant as a station code resolves identically.
        sol_code = fit(
            FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))),
            uvset; ref_ant = "A1",
        )
        @test CAL._step(sol_code, :fringe).θ == fr.θ

        # The snapshot machinery works on a single-stage solution.
        @test CAL._step(stage_solution(sol, :fringe), :fringe).θ == fr.θ

        # A fringe-only solution applies cleanly.
        corr = UVP.apply_calibration(uvset, sol)
        @test corr isa UVP.UVSet
    end

    @testset "rounds > 1: fringe blocks invariant under later stages" begin
        uvset, _ = _build_fringe_uvset()
        solm = fit(
            FringeFit(
                model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false)),
                estimator = MatchedFilter(rounds = 2),
            ) |> TemporalSmoother(FP.SavitzkyGolaySmoother(window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        sol = fit(
            FringeFit(
                model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false)),
                estimator = MatchedFilter(rounds = 2),
            ), uvset,
        )
        fr, frm = CAL._step(sol, :fringe), CAL._step(solm, :fringe)
        rn = CAL.component_ranges(fr.layout)
        rm = CAL.component_ranges(frm.layout)
        for i in 1:5
            @test fr.θ[rn[i]] == frm.θ[rm[i]]
        end
    end

    @testset "opt-in cross-feed rate (a feed-2 Rate list element)" begin
        inj = [0.0, 2.0e-4, -1.0e-4, 5.0e-5]
        uvset, _ = _build_fringe_uvset(rl_rate = inj)
        # A solvable R–L rate is ADDED to the term list — a feed-specific Rate
        # component; the estimator detects it structurally and includes the
        # cross-hand rows in the rate system.
        rl_terms = (;
            _fringe_terms(dispersion = false, sbd = false)...,
            rl_rate = CAL.TiedComponent(CAL.Rate(), CAL.GlobalTime(), CAL.GlobalFrequency(), CAL.FeedComponent(2)),
        )

        sol = fit(FringeFit(model = FringeModel(terms = rl_terms)), uvset)
        fr = CAL._step(sol, :fringe)
        @test length(CAL.phase_components(fr.model)) == 6
        plan = fr.layout.plans[6]
        solved = [fr.θ[plan_off1(plan)[a, 2, 1, 1]] for a in 1:4]
        @test solved ≈ inj .- inj[1] atol = 1.0e-7

        # Null case: no injected feed-rate offset → solved offsets ≈ 0.
        uv0, _ = _build_fringe_uvset()
        sol0 = fit(FringeFit(model = FringeModel(terms = rl_terms)), uv0)
        fr0 = CAL._step(sol0, :fringe)
        plan0 = fr0.layout.plans[6]
        @test maximum(abs, [fr0.θ[plan_off1(plan0)[a, 2, 1, 1]] for a in 1:4]) < 1.0e-7

        # No feed-specific Rate element: the component does not exist — the
        # R–L rate is tied ≡ 0.
        sold = fit(FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))), uvset)
        @test length(CAL._step(sold, :fringe).model.phase) == 5
    end

    @testset "cross_hand_fit_on masks cross-hand rows" begin
        uvset, _ = _build_fringe_uvset()      # per-feed delay/phi ⇒ real R–L offset
        base = CAL._step(
            fit(FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))), uvset),
            :fringe,
        )
        # The global feed-2 delay offset (component 4) is solved.
        @test any(!iszero, base.θ[CAL.component_ranges(base.layout)[4]])

        # Selecting the (only) scan is a no-op: bit-identical to AllScans.
        same = CAL._step(
            fit(
                FringeFit(
                    model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false)),
                    estimator = MatchedFilter(cross_hand_fit_on = Gustavo.ScanIndices(1)),
                ), uvset,
            ), :fringe,
        )
        @test same.θ == base.θ

        # Deselecting every scan removes the feed-tying cross rows: the solve
        # still succeeds (feed 2 gets its own gauge pin; the QQ parallel rows
        # keep its per-station columns constrained) and the feed-common delay
        # is unperturbed beyond solver precision on this noiseless synthetic.
        masked = CAL._step(
            fit(
                FringeFit(
                    model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false)),
                    estimator = MatchedFilter(cross_hand_fit_on = Gustavo.ScanIndices(10_000)),
                ), uvset,
            ), :fringe,
        )
        @test all(isfinite, masked.θ)
        @test masked.θ[CAL.component_ranges(masked.layout)[3]] ≈
            base.θ[CAL.component_ranges(base.layout)[3]] atol = 1.0e-12

        # The masker's direct contract: with an empty selection every
        # CROSS-HAND detection is invalidated, parallel hands untouched.
        geom = CAL.build_geometry(uvset)
        st = FP.scan_stream(uvset; geom = geom)
        stack, win = FP.materialize_cube(st, st.groups[1])
        res = FP.search_scan(stack, st.geom, FP.FringeSearch())
        feeds = [CAL.correlation_feed_pair(p) for p in pol_products(stack)]
        d = FP.StationScanDetections(
            copy(res), baselines(stack).pairs, feeds, first(win.ti_idx),
        )
        FP.mask_unselected_cross_hands!([d], Gustavo.ScanIndices(10_000), st.groups, [NaN])
        for p in eachindex(feeds), bi in axes(d.det, 1)
            fa, fb = feeds[p]
            if fa != fb
                @test !d.det[bi, p].valid
            else
                @test d.det[bi, p] === res[bi, p]
            end
        end
    end

    @testset "transforms on the new path (incl. CalFunction)" begin
        uvset, _ = _build_fringe_uvset()
        ws = [1.0, 0.5, 1.0, 2.0]
        # Weight scale: the search is invariant (snr from the |D|² plane), so
        # the fringe θ matches the untransformed solve bit-for-bit.
        sol_ws = fit(
            StationWeightScale(ws) |>
                FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))),
            uvset,
        )
        sol = fit(FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))), uvset)
        @test CAL._step(sol_ws, :fringe).θ == CAL._step(sol, :fringe).θ
        @test length(sol_ws.transforms) == 1 && sol_ws.transforms[1] isa StationWeightScale

        # CalFunction runs on the new path (it errors only when bridging), and
        # is recorded + replayed by calibrate: zeroing one baseline's weights
        # zero-weights it in the calibrated output.
        touched = Threads.Atomic{Int}(0)
        kill12 = CalFunction() do stack, win
            Threads.atomic_add!(touched, 1)
            for (bi, (a, b)) in enumerate(baselines(stack).pairs)
                minmax(a, b) == (1, 2) && (stack[:weights][Baseline = bi] .= 0)
            end
        end
        sol_cf, out = fitcalibrate(
            kill12 |> FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))),
            uvset,
        )
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
        # The model is the term list alone (the gauge pin is run-wide, on
        # CalibrationPipeline) — no per-effect fields or keywords survive on
        # FringeModel or FringeFit.
        @test_throws MethodError FringeModel(dispersion = :maybe)
        @test_throws MethodError FringeModel(sbd = false)
        @test fieldnames(FringeModel) == (:terms,)
        @test :dispersion ∉ fieldnames(typeof(FringeFit()))

        # The options the legacy bridge used to reject (custom Stationization,
        # the R–L rate opt-in, fit_on subsetting, arbitrary CalFunction
        # transforms) run in FULL pipelines now — every pipeline is new-engine.
        sol_full = fit(
            CalibrationPipeline(
                CalFunction((stack, win) -> nothing),
                FringeFit(
                    model = FringeModel(terms = (;
                        default_fringe_terms()...,
                        rl_rate = CAL.TiedComponent(CAL.Rate(), CAL.GlobalTime(), CAL.GlobalFrequency(), CAL.FeedComponent(2)),
                    )),
                    estimator = MatchedFilter(
                        closure = FP.Stationization(snr_min = 3.0),
                        cross_hand_fit_on = Gustavo.ScanIndices(1),
                    ),
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
        @test any(r -> r.name === :bandpass, sol_fb.steps)
    end

    @testset "term-list compilation: order, gating, duplicate rejection" begin
        uvset, _ = _build_fringe_uvset()   # 2 band groups; narrow fractional bandwidth
        geom = CAL.build_geometry(uvset)
        sig(tc) = (
            typeof(tc.component.term), typeof(tc.component.time),
            typeof(tc.component.freq), typeof(tc.tying),
        )

        # The default list compiles IN LIST ORDER to the standard sequence — 5
        # elements → 5 components; dispersion/SBD are DispersionSBDFit's, not
        # FringeModel's, so they never appear here regardless of geometry.
        comps = FP.fringe_phase_components(FringeModel(), geom)
        @test collect(map(sig, CAL._flatten_components(comps))) == [
            (CAL.ConstantTerm, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
            (CAL.ConstantTerm, CAL.GlobalTime, CAL.GlobalFrequency, CAL.FeedComponent),
            (CAL.Delay, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
            (CAL.Delay, CAL.GlobalTime, CAL.GlobalFrequency, CAL.FeedComponent),
            (CAL.Rate, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
        ]

        # DispersionSBDFit compiles its own private delay-refinement column +
        # dTEC (gate closed on this narrow-fractional-bandwidth geometry) + SBD
        # delay/constant pair (2 band groups here, so SBD's gate is open).
        dscomps = Gustavo.model_components(DispersionSBDFit(), (; geom, antennas = nothing))
        @test collect(map(sig, CAL._flatten_components(dscomps.phase))) == [
            (CAL.Delay, CAL.PerScan, CAL.FrequencyBands, CAL.SharedFeeds),
            (CAL.ConstantTerm, CAL.PerScan, CAL.FrequencyBands, CAL.SharedFeeds),
        ]
        @test isempty(dscomps.logamp)

        # Geometry-gated elements emit nothing when unconstrainable.
        @test CAL.model_components(DispersionModel(), geom) === nothing
        @test CAL.model_components(
            DispersionModel(require_band_separation = false), geom) isa CAL.TiedComponent
        narrow, _ = _build_fringe_uvset(nbands = 1)
        @test CAL.model_components(SingleBandDelay(), CAL.build_geometry(narrow)) === nothing
        # A bare TiedComponent compiles to itself.
        tc = CAL.TiedComponent(CAL.Rate(), CAL.GlobalTime(), CAL.GlobalFrequency(), CAL.FeedComponent(2))
        @test CAL.model_components(tc, geom) === tc

        # Exact duplicate components are rejected by message.
        dup = (;
            default_fringe_terms()...,
            mbd2 = CAL.TiedComponent(CAL.Delay(), CAL.PerScan(), CAL.GlobalFrequency(), CAL.SharedFeeds()),
        )
        @test_throws "two identical components" FP.fringe_phase_components(
            FringeModel(terms = dup), geom)

        # A second component matching a findfirst router's signature — without
        # being an exact duplicate — is rejected naming the signature.
        collide = (;
            default_fringe_terms()...,
            mbd_tb = CAL.TiedComponent(CAL.Delay(), CAL.TimeBlocks(1.0), CAL.GlobalFrequency(), CAL.SharedFeeds()),
        )
        @test_throws "per-scan feed-common delay signature" FP.fringe_phase_components(
            FringeModel(terms = collide), geom)

        # DispersionModel/SingleBandDelay elements are rejected outright —
        # dispersion/SBD are DispersionSBDFit's, not FringeModel's.
        @test_throws "DispersionSBDFit" FringeModel(
            terms = (; default_fringe_terms()..., dtec2 = DispersionModel(tie_colocated = false)))
        @test_throws "DispersionSBDFit" FringeModel(
            terms = (; default_fringe_terms()..., sbd2 = SingleBandDelay()))
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

function FP.estimate_scan!(e::_ProbeEstimator, ctx, step, stack, win)
    e.scans[] += 1
    return FP.estimate_scan!(e.inner, ctx, step, stack, win)
end
function FP.finish_estimate!(e::_ProbeEstimator, ctx, step)
    e.passes[] += 1
    return FP.finish_estimate!(e.inner, ctx, step)
end
# A wrapper fits exactly what it wraps, so both capability hooks forward too.
FP.can_fit(e::_ProbeEstimator, tc) = FP.can_fit(e.inner, tc)
FP.validate_model(e::_ProbeEstimator, comps) = FP.validate_model(e.inner, comps)

# Implements neither solve hook: must fail loudly rather than solve nothing.
# Claims the whole model so the failure is the missing hook, not the capability
# check that runs before it.
struct _SilentEstimator <: FP.AbstractFringeEstimator end
FP.can_fit(::_SilentEstimator, tc) = true

# The independence probe: implements the interface and NOTHING else. It writes no
# θ and publishes none of the matched filter's diagnostic scratch tables, so it
# fails if any of them is secretly required to assemble a solution.
struct _NullEstimator <: FP.AbstractFringeEstimator
    scans::Base.RefValue{Int}
end
_NullEstimator() = _NullEstimator(Ref(0))
function FP.estimate_scan!(e::_NullEstimator, ctx, step, stack, win)
    e.scans[] += 1
    return (; max_snr = NaN)
end
FP.finish_estimate!(::_NullEstimator, ctx, step) = (; chi = 0.0, ncomp = 0, rejected = 0)
FP.can_fit(::_NullEstimator, tc) = true

# Declares no capability at all — the default. Every model term is unclaimed.
struct _UnclaimingEstimator <: FP.AbstractFringeEstimator end
FP.estimate_scan!(::_UnclaimingEstimator, ctx, step, stack, win) = (; max_snr = NaN)
FP.finish_estimate!(::_UnclaimingEstimator, ctx, step) = (; chi = 0.0, ncomp = 0, rejected = 0)

@testset "fringe estimator seam" begin
    uvset, _ = _build_fringe_uvset()
    model = FringeModel()

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
        @test stage_names(sol) == stage_names(ref)
        @test all(a.θ == b.θ for (a, b) in zip(sol.steps, ref.steps))
        @test probe.scans[] == sol.info.nscan
        @test probe.passes[] == 1
    end

    @testset "the step's refine service reaches an out-of-package estimator" begin
        # Downstream stages depend on it, so a third-party estimator must get it
        # without publishing it itself.
        probe = _ProbeEstimator(MatchedFilter())
        sol = fit(
            FringeFit(; model = FringeModel(), estimator = probe) |>
                BandpassEstimator(),
            uvset,
        )
        @test any(r -> r.name === :bandpass, sol.steps)
    end

    @testset "an estimator publishing no diagnostics still yields a solution" begin
        null = _NullEstimator()
        sol = fit(FringeFit(; model, estimator = null), uvset)
        fringe = CAL._step(sol, :fringe)
        @test null.scans[] == sol.info.nscan
        @test all(iszero, fringe.θ)     # it solved nothing, by construction
        @test !haskey(fringe.info, :det_snr)     # no detections published at all
        @test isempty(sol.info.flagged_ant)
        @test !haskey(fringe.info, :scan_snr)
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

    @testset "an estimator that declares no capability is rejected, not run" begin
        # `can_fit`'s default is `false` and the STEP drives the loop, so the
        # estimator that never thought about capability fails at model-compile
        # time instead of returning a solution full of unwritten θ.
        @test_throws "cannot fit the model term" fit(
            FringeFit(; model, estimator = _UnclaimingEstimator()), uvset,
        )
        @test_throws "_UnclaimingEstimator" fit(
            FringeFit(; model, estimator = _UnclaimingEstimator()), uvset,
        )
    end

    @testset "a term the estimator cannot fit is rejected by name" begin
        # A polynomial-in-frequency phase is a legitimate gain term that the
        # matched filter has no observable for: its θ block would stay at zero
        # while the solution looked fitted.
        terms = (; _fringe_terms()..., poly = CAL.TiedComponent(
            CAL.PolynomialFreq(2), CAL.PerScan(), CAL.GlobalFrequency(), CAL.SharedFeeds()))
        @test_throws "MatchedFilter cannot fit the model term" fit(
            FringeFit(model = FringeModel(terms = terms)), uvset,
        )
    end

    @testset "a model missing a term the estimator requires is rejected by name" begin
        # The kind is missing outright: nothing to write the rate search into.
        norate = filter(
            t -> !(t isa CAL.TiedComponent && t.component.term isa CAL.Rate),
            _fringe_terms(),
        )
        @test_throws "requires a rate component" fit(
            FringeFit(model = FringeModel(terms = norate)), uvset,
        )

        # The kind is PRESENT and the router signature is not: the R–L delay is
        # still a `:delay`, so only a signature-level check catches a wideband
        # delay tied across the whole track.
        globaldelay = map(_fringe_terms()) do t
            t isa CAL.TiedComponent && FP._is_perscan_delay(t) ?
                CAL.TiedComponent(t.component.term, CAL.GlobalTime(), t.component.freq, t.tying) : t
        end
        @test_throws "requires a per-scan feed-common wideband delay" fit(
            FringeFit(model = FringeModel(terms = globaldelay)), uvset,
        )
    end

    @testset "the matched filter's kind vocabulary stays private" begin
        # `matched_kind` answers "what does THIS estimator do with this
        # component" — one estimator's vocabulary, so retiring the estimator
        # must not be a public API removal.
        @test !(:matched_kind in names(Gustavo.Fringe))
        @test !(:matched_kind in names(Gustavo))
        @test :can_fit in names(Gustavo.Fringe)
        @test :validate_model in names(Gustavo.Fringe)
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

    @testset "the ionosphere is DispersionSBDFit's own field, not FringeModel's" begin
        @test :dispersion ∉ fieldnames(FringeModel)
        @test :dtec_tie_colocated ∉ fieldnames(FringeModel)
        # Instrumental terms stay in the instrument model's default list;
        # dispersion/SBD are DispersionSBDFit's own fields, not term-list
        # elements — putting either type IN the term list is rejected outright.
        @test !any(t -> t isa SingleBandDelay, FringeModel().terms)
        @test !any(t -> t isa DispersionModel, FringeModel().terms)
        @test fieldnames(DispersionModel) == (:require_band_separation, :tie_colocated)
        @test fieldnames(DispersionSBDFit) == (:dispersion, :sbd)
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
        # Whether the term is SOLVED end to end is `stage_info(sol,
        # :refine).dispersion_applied` (see the dispersion testset in
        # test_pipeline.jl). What this asserts
        # is the model structure DispersionSBDFit's presence/field builds.
        ff = FringeFit(model = FringeModel(), estimator = mf)
        on = fit(ff |> DispersionSBDFit(), uvset)
        off = fit(ff |> DispersionSBDFit(dispersion = nothing), uvset)
        on_ref, off_ref = CAL._step(on, :refine), CAL._step(off, :refine)
        @test CAL._dispersion_plan(on_ref.model, on_ref.layout) !== nothing
        @test CAL._dispersion_plan(off_ref.model, off_ref.layout) === nothing
        @test any(tc -> tc.component.term isa CAL.Dispersion, CAL.phase_components(on_ref.model))
        @test !any(tc -> tc.component.term isa CAL.Dispersion, CAL.phase_components(off_ref.model))
        # No dTEC term means no dTEC column in θ at all — but `off` still has
        # SBD's columns (untouched by the `dispersion` field), so `on` has
        # exactly one more (the private delay-refinement column that only
        # accompanies dTEC).
        @test length(on_ref.θ) - length(off_ref.θ) == length(CAL._dispersion_plan(on_ref.model, on_ref.layout).range) +
            length(FP._perscan_delay_plan(on_ref.model, on_ref.layout).range)
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

    @testset "tie_colocated reaches DispersionSBDFit's own start_pass! through the step" begin
        # The tie is the dispersion model's, but DispersionSBDFit is the step
        # that reads it (`_dtec_ties(s.dispersion, ctx.antennas)`), so it must
        # arrive without the estimator knowing about it.
        ants = Gustavo.UVData.metadata(
            first(values(Gustavo.UVData.branches(uvset)))).antennas
        @test Gustavo._dtec_ties(DispersionModel(tie_colocated = true), ants) !== nothing
        @test Gustavo._dtec_ties(DispersionModel(tie_colocated = false), ants) === nothing
        @test Gustavo._dtec_ties(nothing, ants) === nothing
    end

    @testset "delay and dTEC are still estimated jointly" begin
        # The separation is of the specification only: DispersionSBDFit's own
        # process_scan! fits both plans in one joint (Δτ, dTEC) grid search,
        # because over a finite band the two are near-degenerate — confirmed by
        # both plans existing (and being solved, not left at zero) once the step
        # runs.
        ff = FringeFit(model = FringeModel(), estimator = mf)
        sol = fit(ff |> DispersionSBDFit(), uvset)
        refine = CAL._step(sol, :refine)
        @test CAL._dispersion_plan(refine.model, refine.layout) !== nothing
        @test FP._perscan_delay_plan(refine.model, refine.layout) !== nothing
        @test stage_info(sol, :refine).dispersion_applied
    end
end
