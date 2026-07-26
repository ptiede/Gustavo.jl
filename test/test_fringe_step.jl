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
            FringeFit(model = FringeModel(ref_ant = 1, dispersion = false, sbd = false)) |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        sol = fit(FringeFit(model = FringeModel(dispersion = false, sbd = false)), uvset)
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
        sol_code = fit(FringeFit(model = FringeModel(ref_ant = "A1", dispersion = false, sbd = false)), uvset)
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
                model = FringeModel(ref_ant = 1, dispersion = false, sbd = false),
                estimator = MatchedFilter(rounds = 2),
            ) |> TemporalSmoother(FP.SavitzkyGolaySmoother(window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        sol = fit(
            FringeFit(
                model = FringeModel(dispersion = false, sbd = false),
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
        mk(rate) = FringeModel(dispersion = false, sbd = false, cross_feed = CrossFeed(rate = rate))

        sol = fit(FringeFit(model = mk(Gustavo.GlobalTime())), uvset)
        @test length(sol.model.phase) == 6
        plan = sol.layout.plans[6]
        solved = [sol.θ[plan.off1[a, 2, 1, 1]] for a in 1:4]
        @test solved ≈ inj .- inj[1] atol = 1.0e-7

        # Null case: no injected feed-rate offset → solved offsets ≈ 0.
        uv0, _ = _build_fringe_uvset()
        sol0 = fit(FringeFit(model = mk(Gustavo.GlobalTime())), uv0)
        plan0 = sol0.layout.plans[6]
        @test maximum(abs, [sol0.θ[plan0.off1[a, 2, 1, 1]] for a in 1:4]) < 1.0e-7

        # Default (rate = nothing): the component does not exist — tied ≡ 0.
        sold = fit(FringeFit(model = FringeModel(dispersion = false, sbd = false)), uvset)
        @test length(sold.model.phase) == 5
    end

    @testset "CrossFeed.fit_on masks cross-hand rows" begin
        uvset, _ = _build_fringe_uvset()      # per-feed delay/phi ⇒ real R–L offset
        base = fit(FringeFit(model = FringeModel(dispersion = false, sbd = false)), uvset)
        # The global feed-2 delay offset (component 4) is solved.
        @test any(!iszero, base.θ[CAL.component_ranges(base.layout)[4]])

        # Selecting the (only) scan is a no-op: bit-identical to AllScans.
        same = fit(
            FringeFit(
                model = FringeModel(
                    dispersion = false, sbd = false,
                    cross_feed = CrossFeed(fit_on = Gustavo.ScanIndices(1)),
                ),
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
                    dispersion = false, sbd = false,
                    cross_feed = CrossFeed(fit_on = Gustavo.ScanIndices(10_000)),
                ),
            ), uvset,
        )
        @test all(isfinite, masked.θ)
        @test masked.θ[CAL.component_ranges(masked.layout)[3]] ≈
            base.θ[CAL.component_ranges(base.layout)[3]] atol = 1.0e-12

        # The masker's direct contract: with an empty selection every
        # CROSS-HAND detection is invalidated, parallel hands untouched.
        geom = CAL.build_geometry(uvset)
        st = FP.scan_stream(uvset; geom = geom)
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
        sol_ws = fit(StationWeightScale(ws) |> FringeFit(model = FringeModel(dispersion = false, sbd = false)), uvset)
        sol = fit(FringeFit(model = FringeModel(dispersion = false, sbd = false)), uvset)
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
        sol_cf, out = fitcalibrate(kill12 |> FringeFit(model = FringeModel(dispersion = false, sbd = false)), uvset)
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
        @test_throws ErrorException fit(
            FringeFit(model = FringeModel(dispersion = :maybe)), uvset)

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
