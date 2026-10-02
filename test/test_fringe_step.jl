# ── The BaselineFringeFit step (stage A on the composable engine) ─────────────
#
# A BaselineFringeFit fit solves the matched-filter stage standalone. Tested
# here: the step's gauges and capabilities, the opt-in cross-feed rate solve,
# fit-on-subset cross-hand masking, and corrections applied before the step.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "station gauge: every scan pinned, undetermined columns throw" begin
    ps, _ = _build_fringe_ps(; nant = 4, nscans = 3, noise = 0.3, eltype = ComplexF64)
    at_ref(sol, key) = only(c for c in sol[:fringe].components if last(c.path) === key).params[AntennaName(At("A1"))]

    # A track-global inter-feed delay couples the scans without moving the gauge.
    glob = BaselineFringeFit(; model = FP.default_fringe_terms(; rel_time = CAL.GlobalTime()), gauge = PinAntenna(1))
    @test all(iszero, at_ref(fit(glob, ps), :mbd))

    # Without cross hands the reference's feed-2 delay offset is itself a gauge.
    psp, _ = _build_fringe_ps(; nant = 4, nscans = 2, noise = 0.3, eltype = ComplexF64, polarizations = ["RR", "LL"])
    @test all(iszero, at_ref(fit(BaselineFringeFit(; gauge = PinAntenna(1)), psp), :rel_delay))

    # Two feed-2 delays that every detection sees only as their sum.
    twice = merge(
        FP.default_fringe_terms();
        phase = (; rel_delay_global = CAL.GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Feed = CAL.SingleFeed(2))),
    )
    @test_throws "delay station system is not determined" fit(BaselineFringeFit(; model = twice, gauge = PinAntenna(1)), ps)
end

# Station 1's delays read `value` rather than zero.
struct DelayOffset{T} <: CAL.AbstractGauge
    value::T
end
function CAL.gauge_constraint(g::DelayOffset, f::CAL.GaugeFreedom)
    row, _ = CAL.gauge_constraint(PinAntenna(1), f)
    return row, all(==(:delay), f.observable) ? g.value : zero(g.value)
end

@testset "custom gauges drive the station solve" begin
    ps, _ = _build_fringe_ps(; nant = 4, nscans = 3, noise = 0.3, eltype = ComplexF64)
    params(sol, key) = only(c for c in sol[:fringe].components if last(c.path) === key).params
    at(sol, key, name) = params(sol, key)[AntennaName(At(name))]
    # Every station's value less station `i`'s: invariant under the gauge.
    rel(sol, key, i) = (p = parent(params(sol, key)); p .- p[:, :, :, :, i:i])
    pinned = fit(BaselineFringeFit(; gauge = PinAntenna(1)), ps)

    by_path = fit(BaselineFringeFit(; gauge = ByComponent((; rate = PinAntenna("A2")); default = PinAntenna("A1"))), ps)
    @test all(iszero, at(by_path, :rate, "A2"))
    @test all(iszero, at(by_path, :mbd, "A1"))
    @test rel(by_path, :rate, 2) ≈ rel(pinned, :rate, 2)
    @test params(by_path, :mbd) == params(pinned, :mbd)
    @test_throws "no component rat in the step's model; the components are atmos, mbd, rel_delay, rate" BaselineFringeFit(;
        gauge = ByComponent((; rat = PinAntenna(2)); default = PinAntenna(1)),
    )

    offset = fit(BaselineFringeFit(; gauge = DelayOffset(2.0e-9)), ps)
    @test all(≈(2.0e-9; atol = 1.0e-18), at(offset, :mbd, "A1"))
    @test rel(offset, :mbd, 1) ≈ rel(pinned, :mbd, 1) atol = 1.0e-18
    @test params(offset, :rate) == params(pinned, :rate)
end

@testset "each step applies its own gauge" begin
    ps, _ = _build_fringe_ps(; nant = 4, nscans = 2, noise = 0.3, eltype = ComplexF64)
    fr, ad = _fit_chain((BaselineFringeFit(; gauge = PinAntenna("A1")), AdhocPhase(; gauge = PinAntenna("A2"))), ps)
    at(c, name) = c.params[AntennaName(At(name))]
    mbd = only(c for c in fr[:fringe].components if last(c.path) === :mbd)
    adhoc = only(ad[:adhoc].components)
    @test all(iszero, at(mbd, "A1"))
    @test all(iszero, at(adhoc, "A2"))
    @test !all(iszero, at(adhoc, "A1"))
    @test occursin("PinAntenna{String}(\"A1\")", fr.provenance.pipeline)
    @test occursin("PinAntenna{String}(\"A2\")", ad.provenance.pipeline)

    by_name = ByComponent((; adhoc = PinAntenna("A2")); default = PinAntenna("A1"))
    _, ad2 = _fit_chain((BaselineFringeFit(; gauge = PinAntenna("A1")), AdhocPhase(; gauge = by_name)), ps)
    @test only(ad2[:adhoc].components).params == adhoc.params
end

@testset "BaselineFringeFit step (new engine)" begin
    @testset "a fringe-only solution" begin
        ps, _ = _build_fringe_ps()
        sol = fit(BaselineFringeFit(; gauge = PinAntenna(1)), ps)
        fr = sol[:fringe].components
        @test length(fr) == 4
        @test collect(keys(sol.steps)) == [:fringe]

        # `scan_ncells` is the false-alarm family each recorded `pfa` was computed over.
        solr = fit(BaselineFringeFit(; gauge = PinAntenna(1)), _build_fringe_ps(; nscans = 3, noise = 0.5)[1])
        fr3 = solr.steps[:fringe]
        @test fr3.det_pfa ≈ FP.fringe_pfa.(fr3.det_snr, fr3.scan_ncells[fr3.det_scan])

        # A gauge naming a station code resolves identically.
        sol_code = fit(BaselineFringeFit(; gauge = PinAntenna("A1")), ps)
        @test sol_code[:fringe].components == fr

        # Step selection works on a single-step solution.
        @test filter(c -> c.step === :fringe, sol)[:fringe].components == fr

        # A fringe-only solution applies cleanly.
        @test calibrate(sol, ps) isa XRadio.ProcessingSet
    end

    @testset "a scan the reference sits out still calibrates" begin
        # The reference antenna is in the table but observes no baseline. The
        # stations that DID observe are constrained by their own closure, so none
        # of them is flagged and their data is calibrated rather than blanked.
        ps, _ = _build_fringe_ps(nant = 4, omit_station = 4)
        model = default_fringe_terms()

        sol = fit(BaselineFringeFit(; model, gauge = PinAntenna("A4")), ps)
        @test isempty(sol.steps[:fringe].flagged_ant)
        @test isempty(FP.fringe_station_flags(sol))
        @test calibrate(sol, ps) isa XRadio.ProcessingSet

        # The flags do not depend on which station holds the gauge.
        present = fit(BaselineFringeFit(; model, gauge = PinAntenna("A1")), ps)
        @test present.steps[:fringe].flagged_ant == sol.steps[:fringe].flagged_ant
        @test present.steps[:fringe].flagged_scan == sol.steps[:fringe].flagged_scan
    end

    @testset "opt-in cross-feed rate (a feed-2 Rate list element)" begin
        inj = [0.0, 2.0e-4, -1.0e-4, 5.0e-5]
        ps, _ = _build_fringe_ps(rel_rate = inj)
        # A solvable inter-feed rate is ADDED to the term list — a feed-specific Rate
        # component; the estimator detects it structurally and includes the
        # cross-hand rows in the rate system.
        rel_terms = merge(
            default_fringe_terms();
            phase = (; rel_rate = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Feed = CAL.SingleFeed(2))),
        )

        sol = fit(BaselineFringeFit(; model = rel_terms, gauge = PinAntenna(1)), ps)
        @test count(c -> first(c.path) === :phase, sol[:fringe].components) == 5
        solved = only(sol[:fringe, :phase, :rel_rate].components).params
        @test vec(parent(solved)) ≈ inj .- inj[1] atol = 1.0e-7

        # Null case: no injected feed-rate offset → solved offsets ≈ 0.
        ps0, _ = _build_fringe_ps()
        sol0 = fit(BaselineFringeFit(; model = rel_terms, gauge = PinAntenna(1)), ps0)
        @test maximum(abs, only(sol0[:fringe, :phase, :rel_rate].components).params) < 1.0e-7

        # No feed-specific Rate element: the component does not exist — the
        # The inter-feed rate is tied ≡ 0.
        sold = fit(BaselineFringeFit(; gauge = PinAntenna(1)), ps)
        @test length(sold[:fringe].components) == 4
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
        ps, truth = _build_fringe_ps(; nscans, scan_gap = 2.0, noise = 0.5, seed = 21)
        sol = fit(BaselineFringeFit(; model, gauge = PinAntenna(1)), ps)
        rel = only(sol[:fringe, :phase, :rel_phase].components).params
        want = truth.phi[:, 2] .- truth.phi[:, 1]
        for a in eachindex(want), s in 1:nscans
            got = only(rel[1, :, 1, s, a])
            @test abs(rem2pi(got - want[a], RoundNearest)) < 0.2
        end
    end

    @testset "corrections before the step, including a function" begin
        ps, _ = _build_fringe_ps()
        ws = DimArray([1.0, 0.5, 1.0, 2.0], AntennaName(["A1", "A2", "A3", "A4"]))
        # Weight scale: on noiseless data every baseline's delay and rate are
        # exact, so re-weighting baselines leaves the fringe θ where it was.
        scaled = Gustavo.materialize(ps)
        foreach(ms -> scale_weights!(ms, ws), values(scaled))
        sol_ws = fit(BaselineFringeFit(; gauge = PinAntenna(1)), scaled)
        sol = fit(BaselineFringeFit(; gauge = PinAntenna(1)), ps)
        @test all(
            isapprox(parent(a.params), parent(b.params); rtol = 1.0e-4)
                for (a, b) in zip(sol_ws[:fringe].components, sol[:fringe].components)
        )

        # Flagging one baseline before the step keeps it flagged in the
        # calibrated output.
        is12(a, b) = Set((a, b)) == Set(("A1", "A2"))
        function kill12(ms)
            for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
                is12(a, b) && (view(ms[:flag], BaselineID(bi)) .= true)
            end
            return ms
        end
        killed = Gustavo.materialize(ps)
        foreach(kill12, values(killed))
        sol_cf = fit(BaselineFringeFit(; gauge = PinAntenna(1)), killed)
        calibrated = calibrate(sol_cf, killed)
        for ms in values(calibrated), (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
            is12(a, b) && @test all(view(ms[:flag], BaselineID(bi)))
        end
    end

    @testset "model validation + option coverage ahead of later steps" begin
        ps, _ = _build_fringe_ps()
        # The model is the component tree alone, and the gauge a field of its own —
        # no per-effect fields or keywords on BaselineFringeFit.
        @test fieldnames(typeof(BaselineFringeFit(; gauge = PinAntenna(1)))) ==
            (:model, :search, :closure, :rounds, :steer_cells, :gauge)

        # A custom Stationization and the inter-feed rate opt-in run ahead of
        # the bandpass and adhoc steps.
        gauge = PinAntenna(1)
        sols_full = _fit_chain(
            (
                BaselineFringeFit(;
                    model = merge(
                        default_fringe_terms();
                        phase = (; rel_rate = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Feed = CAL.SingleFeed(2))),
                    ),
                    closure = FP.Stationization(pfa_max = 1.0e-2),
                    gauge,
                ),
                Bandpass(; gauge), AdhocPhase(; gauge),
            ),
            ps; exec = ExecutionConfig(),
        )
        @test [only(keys(s.steps)) for s in sols_full] == [:fringe, :bandpass, :adhoc]
        # Bandpass after the fringe step alone still solves a :bandpass stage.
        _, sol_fb = _fit_chain((BaselineFringeFit(; gauge), Bandpass(; gauge)), ps)
        @test haskey(sol_fb, :bandpass)

        # Any real `steer_cells`, and a Float32 station solve, agreeing with the default.
        @test BaselineFringeFit(; steer_cells = 9, gauge).steer_cells === 9
        sol64 = fit(BaselineFringeFit(; gauge = PinAntenna(1)), ps)
        sol32 = fit(BaselineFringeFit(; closure = FP.Stationization(eltype = Float32), steer_cells = 9, gauge = PinAntenna(1)), ps)
        for (c32, c64) in zip(sol32[:fringe].components, sol64[:fringe].components)
            @test c32.params ≈ c64.params atol = 1.0e-6
        end
        @test sol32.steps[:fringe].flagged_ant == sol64.steps[:fringe].flagged_ant
    end

    @testset "model compilation: order, gating, duplicate rejection" begin
        ps, _ = _build_fringe_ps()   # 2 band groups; narrow fractional bandwidth
        geom = CAL.DataGeometry(ps)
        fringe_phase(model) =
            Gustavo.model_components(BaselineFringeFit(; model, gauge = PinAntenna(1)), (; geom)).phase
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
        @test CAL.model_components(tc, (; geom)) === tc

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
    ps, _ = _build_fringe_ps()
    model = default_fringe_terms()

    @testset "the solution records the search configuration" begin
        sol = fit(BaselineFringeFit(; model, gauge = PinAntenna(1)), ps)
        @test sol.steps[:fringe].search == FP.FringeSearch()
        @test haskey(sol.steps[:fringe], :flagged_ant) && haskey(sol.steps[:fringe], :flagged_scan)
    end

    @testset "a term the step cannot fit is rejected by name" begin
        # A polynomial-in-frequency phase is a legitimate gain term that the
        # matched filter has no observable for: its θ block would stay at zero
        # while the solution looked fitted.
        poly = CAL.GainComponent(CAL.PolynomialFreq(2); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds())
        @test_throws "BaselineFringeFit cannot fit the component" fit(
            BaselineFringeFit(; model = merge(default_fringe_terms(); phase = (; poly)), gauge = PinAntenna(1)), ps,
        )
    end

    @testset "a model missing a term the step requires is rejected by name" begin
        # The kind is missing outright: nothing to write the rate search into.
        norate = GainModel(; phase = Base.structdiff(default_fringe_terms().phase, (; rate = nothing)))
        @test_throws "requires a rate component" fit(BaselineFringeFit(; model = norate, gauge = PinAntenna(1)), ps)

        # The kind is PRESENT and the router signature is not: the inter-feed delay is
        # still a `:delay`, so only a signature-level check catches a wideband
        # delay tied across the whole track.
        globaldelay = map(default_fringe_terms().phase) do t
            FP._is_perscan_delay(t) ?
                CAL.GainComponent(t.term; Ti = CAL.GlobalTime(), Frequency = t.Frequency, Feed = t.Feed) : t
        end
        @test_throws "requires a per-scan feed-common wideband delay" fit(
            BaselineFringeFit(; model = GainModel(; phase = globaldelay), gauge = PinAntenna(1)), ps,
        )
    end

    @testset "a time segmentation finer than a scan is rejected, not half-solved" begin
        # One search per scan group measures one delay, rate and phase per scan.
        # A segmentation splitting a scan asks for columns nothing writes: the
        # scan's first segment would be solved and the rest left at identity
        # gain while `calibrate` places each sample in its own segment's column.
        # `_build_fringe_ps` gives one 330 s scan, so 150 s blocks split it.
        subscan = map(
            t -> CAL.GainComponent(t.term; Ti = CAL.TimeBlocks(150.0), Frequency = t.Frequency, Feed = t.Feed),
            default_fringe_terms().phase,
        )
        @test_throws "BaselineFringeFit cannot fit the component" fit(
            BaselineFringeFit(; model = GainModel(; phase = subscan), gauge = PinAntenna(1)), ps,
        )
        @test_throws "the data's own sampling" fit(
            BaselineFringeFit(; model = GainModel(; phase = subscan), gauge = PinAntenna(1)), ps,
        )

        geom = CAL.DataGeometry(ps)
        mbd(Ti) = CAL.GainComponent(CAL.Delay(); Ti, Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds())

        # The same boundary expressed as an instrument scan edge, and the
        # per-integration limit the classifier used to special-case.
        @test !FP.can_fit(BaselineFringeFit(; gauge = PinAntenna(1)), mbd(CAL.InstrumentScans([geom.times[1] + 150.0])), geom)
        @test !FP.can_fit(BaselineFringeFit(; gauge = PinAntenna(1)), mbd(CAL.PerIntegration()), geom)

        # Coarser than a scan is fitted: one column several scans share is
        # written by all of them. The rule is the segmentation against the
        # geometry, not the segmentation alone — 3600 s blocks do not split a
        # 330 s scan.
        @test FP.can_fit(BaselineFringeFit(; gauge = PinAntenna(1)), mbd(CAL.PerScan()), geom)
        @test FP.can_fit(BaselineFringeFit(; gauge = PinAntenna(1)), mbd(CAL.GlobalTime()), geom)
        @test FP.can_fit(BaselineFringeFit(; gauge = PinAntenna(1)), mbd(CAL.TimeBlocks(3600.0)), geom)
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
