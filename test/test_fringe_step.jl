# ── The BaselineFringeFit step (stage A on the composable engine) ─────────────
#
# A BaselineFringeFit-only pipeline solves the matched-filter stage standalone. The M3
# bit-parity gates against the frozen monolith ran before its deletion; the
# standing invariant kept here is that LATER STAGES NEVER MOVE THE FRINGE
# BLOCKS: a fringe-only fit's θ blocks are bit-identical to the same blocks of
# a fuller pipeline (dispersion/SBD off, so no later stage refines the compared
# slots). Plus the step's capabilities: the opt-in cross-feed rate solve,
# fit-on-subset cross-hand masking, and transforms on the streaming path.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "BaselineFringeFit step (new engine)" begin
    @testset "fringe blocks invariant under later stages" begin
        uvset, _ = _build_fringe_uvset()
        solm = fit(
            BaselineFringeFit() |>
                AdhocPhase(FP.SavitzkyGolaySmoother(window = 7, order = 2, options = FP.AdhocOptions(; snr_floor = 0.0))),
            uvset,
            gauge = PinAntenna(1),
        )
        sol = fit(BaselineFringeFit(), uvset; gauge = PinAntenna(1))
        fr, frm = sol[:fringe].steps[1], solm[:fringe].steps[1]
        @test length(fr.model.phase) == 4
        rn = [p.range for p in fr.layout.plans]
        rm = [p.range for p in frm.layout.plans]
        for i in 1:4
            @test fr.θ[rn[i]] == frm.θ[rm[i]]        # bit-identical
        end
        @test keys(sol) == [:fringe]
        @test sol.info.nscan == solm.info.nscan
        @test fr.info.scan_snr == frm.info.scan_snr
        @test fr.info.scan_ncells == frm.info.scan_ncells
        @test fr.info.det_snr == frm.info.det_snr
        @test fr.info.det_pfa == frm.info.det_pfa

        # A gauge naming a station code resolves identically.
        sol_code = fit(
            BaselineFringeFit(),
            uvset; gauge = PinAntenna("A1"),
        )
        @test sol_code[:fringe].steps[1].θ == fr.θ

        # Step selection works on a single-step solution.
        @test sol[1:1][:fringe].steps[1].θ == fr.θ

        # A fringe-only solution applies cleanly.
        corr = UVP.apply_calibration(uvset, sol)
        @test corr isa UVP.UVSet
    end

    @testset "a scan the reference sits out still calibrates" begin
        # The reference antenna is in the table but observes no baseline. The
        # stations that DID observe are constrained by their own closure, so none
        # of them is flagged and their data is calibrated rather than blanked.
        uvset, _ = _build_fringe_uvset(nant = 4, omit_station = 4)
        model = default_fringe_terms()

        sol = fit(BaselineFringeFit(; model), uvset; gauge = PinAntenna("A4"))
        @test isempty(sol.info.flagged_ant)
        @test isempty(FP.fringe_station_flags(sol))
        @test UVP.apply_calibration(uvset, sol) isa UVP.UVSet

        # The flags do not depend on which station holds the gauge.
        present = fit(BaselineFringeFit(; model), uvset; gauge = PinAntenna("A1"))
        @test present.info.flagged_ant == sol.info.flagged_ant
        @test present.info.flagged_scan == sol.info.flagged_scan
    end

    @testset "rounds > 1: fringe blocks invariant under later stages" begin
        uvset, _ = _build_fringe_uvset()
        solm = fit(
            BaselineFringeFit(
                model = default_fringe_terms(),
                rounds = 2,
            ) |> AdhocPhase(FP.SavitzkyGolaySmoother(window = 7, order = 2, options = FP.AdhocOptions(; snr_floor = 0.0))),
            uvset,
            gauge = PinAntenna(1),
        )
        sol = fit(
            BaselineFringeFit(
                model = default_fringe_terms(),
                rounds = 2,
            ), uvset,
            gauge = PinAntenna(1),
        )
        fr, frm = sol[:fringe].steps[1], solm[:fringe].steps[1]
        rn = [p.range for p in fr.layout.plans]
        rm = [p.range for p in frm.layout.plans]
        for i in 1:4
            @test fr.θ[rn[i]] == frm.θ[rm[i]]
        end
    end

    @testset "opt-in cross-feed rate (a feed-2 Rate list element)" begin
        inj = [0.0, 2.0e-4, -1.0e-4, 5.0e-5]
        uvset, _ = _build_fringe_uvset(rel_rate = inj)
        # A solvable inter-feed rate is ADDED to the term list — a feed-specific Rate
        # component; the estimator detects it structurally and includes the
        # cross-hand rows in the rate system.
        rel_terms = merge(
            default_fringe_terms();
            phase = (; rel_rate = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Feed = CAL.SingleFeed(2))),
        )

        sol = fit(BaselineFringeFit(model = rel_terms), uvset; gauge = PinAntenna(1))
        fr = sol[:fringe].steps[1]
        @test length(CAL.phase_components(fr.model)) == 5
        plan = fr.layout.plans[5]
        solved = [fr.θ[plan_off1(plan)[a, 2, 1, 1]] for a in 1:4]
        @test solved ≈ inj .- inj[1] atol = 1.0e-7

        # Null case: no injected feed-rate offset → solved offsets ≈ 0.
        uv0, _ = _build_fringe_uvset()
        sol0 = fit(BaselineFringeFit(model = rel_terms), uv0; gauge = PinAntenna(1))
        fr0 = sol0[:fringe].steps[1]
        plan0 = fr0.layout.plans[5]
        @test maximum(abs, [fr0.θ[plan_off1(plan0)[a, 2, 1, 1]] for a in 1:4]) < 1.0e-7

        # No feed-specific Rate element: the component does not exist — the
        # The inter-feed rate is tied ≡ 0.
        sold = fit(BaselineFringeFit(), uvset; gauge = PinAntenna(1))
        @test length(sold[:fringe].steps[1].model.phase) == 4
    end

    @testset "a constant phase is referenced to its own scan" begin
        # A rate is measured only to within its own uncertainty, so a constant
        # phase quoted a lever arm Δt from the data carries 2π·σ_ṙ·Δt of it.
        # A feed-COMMON constant hides that — the station rate solved from the
        # same rows moves with it — so the probe is a feed-2 constant, which the
        # model gives no rate of its own: it keeps the whole lever arm. Referred
        # to a track-wide epoch instead of its own scan's, hours of lever arm
        # randomize it outright. Noise is what makes this visible; with exact
        # rates there is no uncertainty to lever.
        #
        # Not part of `default_fringe_terms` (see there for why the inter-feed
        # phase offset is deliberately absent) — added here precisely because it
        # is the column most sensitive to the epoch.
        nscans = 4
        model = merge(
            default_fringe_terms();
            phase = (; rel_phase = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Feed = CAL.SingleFeed(2))),
        )
        uvset, truth = _build_fringe_uvset(; nscans, scan_gap = 2.0, noise = 0.5, seed = 21)
        sol = fit(BaselineFringeFit(; model), uvset; gauge = PinAntenna(1))
        rel = CAL.parameters(sol[:fringe, :phase, :rel_phase])
        want = truth.phi[:, 2] .- truth.phi[:, 1]
        for a in eachindex(want), s in 1:nscans
            got = only(rel[1, :, 1, s, a])
            @test abs(rem2pi(got - want[a], RoundNearest)) < 0.2
        end
    end

    @testset "transforms on the new path (incl. CalFunction)" begin
        uvset, _ = _build_fringe_uvset()
        ws = [1.0, 0.5, 1.0, 2.0]
        # Weight scale: the search is invariant (snr from the |D|² plane), so
        # the fringe θ matches the untransformed solve bit-for-bit.
        sol_ws = fit(
            StationWeightScale(ws) |>
                BaselineFringeFit(),
            uvset,
            gauge = PinAntenna(1),
        )
        sol = fit(BaselineFringeFit(), uvset; gauge = PinAntenna(1))
        @test sol_ws[:fringe].steps[1].θ == sol[:fringe].steps[1].θ
        @test only(recorded_transforms(sol_ws)) isa StationWeightScale

        # CalFunction runs on the new path (it errors only when bridging), and
        # is recorded + replayed by calibrate: flagging one baseline flags it in
        # the calibrated output.
        touched = Threads.Atomic{Int}(0)
        kill12 = CalFunction() do stack, win
            Threads.atomic_add!(touched, 1)
            for (bi, (a, b)) in enumerate(baselines(stack).pairs)
                if minmax(a, b) == (1, 2)
                    stack[:flags][BaselineID = bi] .= true
                end
            end
        end
        sol_cf = fit(kill12 |> BaselineFringeFit(), uvset; gauge = PinAntenna(1))
        @test touched[] > 0
        @test only(recorded_transforms(sol_cf)) isa CalFunction
        out = calibrate(sol_cf, uvset)
        for (_, leaf) in DimensionalData.branches(out)
            F = parent(leaf[:flags])
            for (bi, (a, b)) in enumerate(UVP.baselines(leaf).pairs)
                minmax(a, b) == (1, 2) && @test all(F[:, :, bi, :])
            end
        end
    end

    @testset "model validation + full-pipeline option coverage" begin
        uvset, _ = _build_fringe_uvset()
        # The model is the component tree alone (the gauge pin is run-wide, an
        # argument of `fit`) — no per-effect fields or keywords on BaselineFringeFit.
        @test fieldnames(typeof(BaselineFringeFit())) ==
            (:model, :search, :closure, :rounds, :steer_cells)

        # The options the legacy bridge used to reject (custom Stationization,
        # the inter-feed rate opt-in, arbitrary CalFunction transforms) run in
        # FULL pipelines now — every pipeline is new-engine.
        sol_full = fit(
            [CalFunction((stack, win) -> nothing),
                BaselineFringeFit(
                    model = merge(
                        default_fringe_terms();
                        phase = (; rel_rate = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Feed = CAL.SingleFeed(2))),
                    ),
                    closure = FP.Stationization(pfa_max = 1.0e-2),
                ),
                Bandpass(), AdhocPhase()],
            uvset,
            exec = ExecutionConfig(),
            gauge = PinAntenna(1),
        )
        @test keys(sol_full) == [:fringe, :bandpass, :adhoc]
        # Bandpass without AdhocPhase still solves a :bandpass
        # stage (F |> B — no final pass).
        sol_fb = fit([BaselineFringeFit(), Bandpass()], uvset; gauge = PinAntenna(1))
        @test any(r -> r.name === :bandpass, sol_fb.steps)
    end

    @testset "model compilation: order, gating, duplicate rejection" begin
        uvset, _ = _build_fringe_uvset()   # 2 band groups; narrow fractional bandwidth
        geom = CAL.build_geometry(uvset)
        fringe_phase(model) =
            Gustavo.model_components(BaselineFringeFit(; model), (; geom, antennas = nothing)).phase
        sig(tc) = (
            typeof(tc.term), typeof(tc.Ti),
            typeof(tc.Frequency), typeof(tc.Feed),
        )

        # The default model compiles in order to the standard sequence, with no
        # inter-feed CONSTANT (see `default_fringe_terms`).
        comps = fringe_phase(default_fringe_terms())
        @test collect(map(sig, CAL._flatten_components(comps))) == [
            (CAL.ConstantTerm, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
            (CAL.Delay, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
            (CAL.Delay, CAL.PerScan, CAL.GlobalFrequency, CAL.SingleFeed),
            (CAL.Rate, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
        ]

        # `rel_time` moves the inter-feed delay onto a track-global column.
        gcomps = fringe_phase(default_fringe_terms(rel_time = CAL.GlobalTime()))
        @test collect(map(sig, CAL._flatten_components(gcomps))) == [
            (CAL.ConstantTerm, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
            (CAL.Delay, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
            (CAL.Delay, CAL.GlobalTime, CAL.GlobalFrequency, CAL.SingleFeed),
            (CAL.Rate, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
        ]

        # A bare GainComponent compiles to itself.
        tc = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SingleFeed(2))
        @test CAL.model_components(tc, (; geom, antennas = nothing)) === tc

        # Exact duplicate components are rejected by message.
        dup = merge(
            default_fringe_terms();
            phase = (; mbd2 = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds())),
        )
        @test_throws "two identical components" fringe_phase(dup)

        # A second component matching a findfirst router's signature — without
        # being an exact duplicate — is rejected naming the signature.
        collide = merge(
            default_fringe_terms();
            phase = (; mbd_tb = CAL.GainComponent(CAL.Delay(); Ti = CAL.TimeBlocks(1.0e6), Feed = CAL.SharedFeeds())),
        )
        @test_throws "per-scan feed-common delay signature" fringe_phase(collide)

        # Dispersion and a per-band-group delay are rejected: no shipped step
        # fits either.
        withphase(; kw...) = merge(default_fringe_terms(); phase = NamedTuple(kw))
        @test_throws "No shipped step fits dispersion" fringe_phase(
            withphase(dtec2 = CAL.GainComponent(CAL.Dispersion(); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds())),
        )
        @test_throws "No shipped step fits dispersion" fringe_phase(
            withphase(sbd2 = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Frequency = CAL.FreqGroups([1:1, 2:length(geom.channel_freqs)]), Feed = CAL.SharedFeeds())),
        )

        # The fringe step fits phase only.
        @test_throws "`logamp` group must be empty" fringe_phase(
            merge(default_fringe_terms(); logamp = (; a = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds()))),
        )
    end
end

# ── What the fringe step can fit ─────────────────────────────────────────────

@testset "fringe step capability" begin
    uvset, _ = _build_fringe_uvset()
    model = default_fringe_terms()

    @testset "the solution records the search configuration" begin
        sol = fit(BaselineFringeFit(; model), uvset; gauge = PinAntenna(1))
        @test sol.info.search == FP.FringeSearch()
        @test stage_info(sol, :fringe).flagged_ant == sol.info.flagged_ant
    end

    @testset "a term the step cannot fit is rejected by name" begin
        # A polynomial-in-frequency phase is a legitimate gain term that the
        # matched filter has no observable for: its θ block would stay at zero
        # while the solution looked fitted.
        poly = CAL.GainComponent(CAL.PolynomialFreq(2); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds())
        @test_throws "BaselineFringeFit cannot fit the component" fit(
            BaselineFringeFit(model = merge(default_fringe_terms(); phase = (; poly))), uvset,
            gauge = PinAntenna(1),
        )
    end

    @testset "a model missing a term the step requires is rejected by name" begin
        # The kind is missing outright: nothing to write the rate search into.
        norate = GainModel(; phase = Base.structdiff(default_fringe_terms().phase, (; rate = nothing)))
        @test_throws "requires a rate component" fit(BaselineFringeFit(model = norate), uvset; gauge = PinAntenna(1))

        # The kind is PRESENT and the router signature is not: the inter-feed delay is
        # still a `:delay`, so only a signature-level check catches a wideband
        # delay tied across the whole track.
        globaldelay = map(default_fringe_terms().phase) do t
            FP._is_perscan_delay(t) ?
                CAL.GainComponent(t.term; Ti = CAL.GlobalTime(), Frequency = t.Frequency, Feed = t.Feed) : t
        end
        @test_throws "requires a per-scan feed-common wideband delay" fit(
            BaselineFringeFit(model = GainModel(; phase = globaldelay)), uvset,
            gauge = PinAntenna(1),
        )
    end

    @testset "a time segmentation finer than a scan is rejected, not half-solved" begin
        # One search per scan group measures one delay, rate and phase per scan.
        # A segmentation splitting a scan asks for columns nothing writes: the
        # scan's first segment would be solved and the rest left at identity
        # gain while `calibrate` places each sample in its own segment's column.
        # `_build_fringe_uvset` gives one 330 s scan, so 150 s blocks split it.
        subscan = map(
            t -> CAL.GainComponent(t.term; Ti = CAL.TimeBlocks(150.0), Frequency = t.Frequency, Feed = t.Feed),
            default_fringe_terms().phase,
        )
        @test_throws "BaselineFringeFit cannot fit the component" fit(
            BaselineFringeFit(model = GainModel(; phase = subscan)), uvset,
            gauge = PinAntenna(1),
        )
        @test_throws "the data's own sampling" fit(
            BaselineFringeFit(model = GainModel(; phase = subscan)), uvset,
            gauge = PinAntenna(1),
        )

        geom = CAL.build_geometry(uvset)
        mbd(Ti) = CAL.GainComponent(CAL.Delay(); Ti, Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds())

        # The same boundary expressed as an instrument scan edge, and the
        # per-integration limit the classifier used to special-case.
        @test !FP.can_fit(BaselineFringeFit(), mbd(CAL.InstrumentScans([geom.times[1] + 150.0])), geom)
        @test !FP.can_fit(BaselineFringeFit(), mbd(CAL.PerIntegration()), geom)

        # Coarser than a scan is fitted: one column several scans share is
        # written by all of them. The rule is the segmentation against the
        # geometry, not the segmentation alone — 3600 s blocks do not split a
        # 330 s scan.
        @test FP.can_fit(BaselineFringeFit(), mbd(CAL.PerScan()), geom)
        @test FP.can_fit(BaselineFringeFit(), mbd(CAL.GlobalTime()), geom)
        @test FP.can_fit(BaselineFringeFit(), mbd(CAL.TimeBlocks(3600.0)), geom)
    end

    @testset "the matched filter's kind vocabulary stays private" begin
        # `matched_kind` answers "what does the matched filter do with this
        # component", an implementation detail of the fringe step.
        @test !(:matched_kind in names(Gustavo.Fring))
        @test !(:matched_kind in names(Gustavo))
        @test :can_fit in names(Gustavo.Fring)
        @test :validate_model in names(Gustavo.Fring)
    end

end
