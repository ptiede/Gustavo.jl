# ── The fringe fit: a search of every scan, then the closure ─────────────────
#
# Tested here: the closure's gauges and capabilities, the opt-in cross-feed
# rate solve, fit-on-subset cross-hand masking, and corrections applied before
# the fit.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "antenna gauge: every scan pinned, undetermined columns throw" begin
    ps, _ = _build_fringe_ps(; nant = 4, nscans = 3, noise = 0.3, eltype = ComplexF64)
    at_ref(sol, key) = only(c for c in sol.components if last(c.path) === key).params[AntennaName(At("A1"))]
    # The fit, and whether it warned that a fringe group's feeds are not linked.
    function fit_unlinked(closure, data)
        logs, sol = Test.collect_test_logs(() -> _fringe_fit(data, closure))
        return sol, any(l -> occursin("offset between its feeds is not measured", string(l.message)), logs)
    end

    # A track-global inter-feed delay couples the scans without moving the gauge.
    glob = FringeClosure(; model = FP.default_fringe_model(; rel_time = CAL.GlobalTime()), gauge = PinAntenna(1))
    sol, warned = fit_unlinked(glob, ps)
    @test all(iszero, at_ref(sol, :mbd))
    @test !warned

    # Without detections relating different feeds the reference's feed-2 delay
    # offset is a gauge too, and the fit says so.
    psp, _ = _build_fringe_ps(; nant = 4, nscans = 2, noise = 0.3, eltype = ComplexF64, polarizations = ["RR", "LL"])
    sol, warned = fit_unlinked(FringeClosure(; gauge = PinAntenna(1)), psp)
    @test all(iszero, at_ref(sol, :rel_delay))
    @test warned

    # Likewise for a station whose feed order differs from the reference's: the
    # free offset is pinned whichever feed index carries it.
    for order in (["R", "L"], ["L", "R"])
        pso, truth = _build_fringe_ps(;
            nant = 4, noise = 0.3, eltype = ComplexF64, polarizations = ["RR", "LL"],
            receptor_order = Dict("A2" => order),
        )
        sol, warned = fit_unlinked(FringeClosure(; gauge = PinAntenna(1)), pso)
        d = parent(FP.fringe_antenna_solutions(sol).delay)[1, :, :]
        @test maximum(abs, (d .- d[1:1, :]) .- (truth.delay .- truth.delay[1:1, :])) < 5.0e-11
        @test warned
    end

    # Two feed-2 delays that every detection sees only as their sum.
    twice = merge(
        FP.default_fringe_model();
        phase = (; rel_delay_global = CAL.GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Feed = CAL.SingleFeed(2))),
    )
    @test_throws "the system does not determine every unknown" _fringe_fit(ps, FringeClosure(; model = twice, gauge = PinAntenna(1)))
end

@testset "a rate needs two times per scan" begin
    ps, _ = _build_fringe_ps(; nant = 4, nscans = 2, ntime = 1, noise = 0.3, eltype = ComplexF64)
    @test_throws "has one time, and a rate needs at least two" _fringe_fit(ps, FringeClosure(; gauge = PinAntenna(1)))
end

# Station 1's delays read `value` rather than zero.
struct DelayOffset{T} <: CAL.AbstractGauge
    value::T
end
function CAL.gauge_constraint(g::DelayOffset, f::CAL.GaugeFreedom)
    row, _ = CAL.gauge_constraint(PinAntenna(1), f)
    return row, all(==(:delay), f.observable) ? g.value : zero(g.value)
end

@testset "custom gauges drive the antenna solve" begin
    ps, _ = _build_fringe_ps(; nant = 4, nscans = 3, noise = 0.3, eltype = ComplexF64)
    params(sol, key) = only(c for c in sol.components if last(c.path) === key).params
    at(sol, key, name) = params(sol, key)[AntennaName(At(name))]
    # Every station's value less station `i`'s: invariant under the gauge.
    rel(sol, key, i) = (p = parent(params(sol, key)); p .- p[:, :, :, :, i:i])
    pinned = _fringe_fit(ps, FringeClosure(; gauge = PinAntenna(1)))

    by_path = _fringe_fit(ps, FringeClosure(; gauge = ByComponent((; rate = PinAntenna("A2")); default = PinAntenna("A1"))))
    @test all(iszero, at(by_path, :rate, "A2"))
    @test all(iszero, at(by_path, :mbd, "A1"))
    @test rel(by_path, :rate, 2) ≈ rel(pinned, :rate, 2)
    @test params(by_path, :mbd) == params(pinned, :mbd)
    @test_throws "no component rat in the gauged model; the components are mbd, rel_delay, rate" FringeClosure(;
        gauge = ByComponent((; rat = PinAntenna(2)); default = PinAntenna(1)),
    )

    offset = _fringe_fit(ps, FringeClosure(; gauge = DelayOffset(2.0e-9)))
    @test all(≈(2.0e-9; atol = 1.0e-18), at(offset, :mbd, "A1"))
    @test rel(offset, :mbd, 1) ≈ rel(pinned, :mbd, 1) atol = 1.0e-18
    @test params(offset, :rate) == params(pinned, :rate)
end

@testset "each fit applies its own gauge" begin
    ps, _ = _build_fringe_ps(; nant = 4, nscans = 2, noise = 0.3, eltype = ComplexF64)
    fr, ad = _fit_chain((FringeClosure(; gauge = PinAntenna("A1")), AdhocPhase(; gauge = PinAntenna("A2"))), ps)
    at(c, name) = c.params[AntennaName(At(name))]
    mbd = only(c for c in fr.components if last(c.path) === :mbd)
    adhoc = only(ad.components)
    @test all(iszero, at(mbd, "A1"))
    @test all(iszero, at(adhoc, "A2"))
    @test !all(iszero, at(adhoc, "A1"))
    @test occursin("PinAntenna{String}(\"A1\")", fr.spec)
    @test occursin("PinAntenna{String}(\"A2\")", ad.spec)

    by_name = ByComponent((; adhoc = PinAntenna("A2")); default = PinAntenna("A1"))
    _, ad2 = _fit_chain((FringeClosure(; gauge = PinAntenna("A1")), AdhocPhase(; gauge = by_name)), ps)
    @test only(ad2.components).params == adhoc.params
end

@testset "fringe fit" begin
    @testset "a fringe-only solution" begin
        ps, _ = _build_fringe_ps()
        sol = _fringe_fit(ps, FringeClosure(; gauge = PinAntenna(1)))
        fr = sol.components
        @test length(fr) == 3
        @test sol.name === :fringe

        # `ncells` is the false-alarm family each recorded `pfa` was computed over.
        dets = fit(FringeSearch(), _build_fringe_ps(; nscans = 3, noise = 0.5)[1])
        @test length(dets) == 3
        for d in dets
            v = parent(d.valid)
            @test parent(d.pfa)[v] ≈ FP.fringe_pfa.(parent(d.snr)[v], metadata(d).ncells)
        end

        # A gauge naming a station code resolves identically.
        sol_code = _fringe_fit(ps, FringeClosure(; gauge = PinAntenna("A1")))
        @test sol_code.components == fr

        # Path selection and filtering keep every component of a fringe-only solution.
        @test filter(c -> first(c.path) === :phase, sol)[:phase].components == fr

        # A fringe-only solution applies cleanly.
        @test calibrate(sol, ps) isa XRadio.ProcessingSet
    end

    @testset "a scan the reference sits out still calibrates" begin
        # The reference antenna is in the table but observes no baseline. The
        # stations that DID observe are constrained by their own closure, so none
        # of them is flagged and their data is calibrated rather than blanked.
        ps, _ = _build_fringe_ps(nant = 4, omit_antenna = 4)
        model = default_fringe_model()

        sol = _fringe_fit(ps, FringeClosure(; model, gauge = PinAntenna("A4")))
        @test isempty(sol.info.flagged_ant)
        @test !any(FP.fringe_antenna_flags(sol))
        @test calibrate(sol, ps) isa XRadio.ProcessingSet

        # The flags do not depend on which station holds the gauge.
        present = _fringe_fit(ps, FringeClosure(; model, gauge = PinAntenna("A1")))
        @test present.info.flagged_ant == sol.info.flagged_ant
        @test present.info.flagged_scan == sol.info.flagged_scan
    end

    @testset "opt-in cross-feed rate (a feed-2 Rate list element)" begin
        inj = [0.0, 2.0e-4, -1.0e-4, 5.0e-5]
        ps, _ = _build_fringe_ps(rel_rate = inj)
        # A solvable inter-feed rate is ADDED to the term list — a feed-specific Rate
        # component; the estimator detects it structurally and includes the
        # cross-hand rows in the rate system.
        rel_terms = merge(
            default_fringe_model();
            phase = (; rel_rate = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Feed = CAL.SingleFeed(2))),
        )

        sol = _fringe_fit(ps, FringeClosure(; model = rel_terms, gauge = PinAntenna(1)))
        @test count(c -> first(c.path) === :phase, sol.components) == 4
        solved = only(sol[:phase, :rel_rate].components).params
        @test vec(parent(solved)) ≈ inj .- inj[1] atol = 1.0e-7

        # Null case: no injected feed-rate offset → solved offsets ≈ 0.
        ps0, _ = _build_fringe_ps()
        sol0 = _fringe_fit(ps0, FringeClosure(; model = rel_terms, gauge = PinAntenna(1)))
        @test maximum(abs, only(sol0[:phase, :rel_rate].components).params) < 1.0e-7

        # No feed-specific Rate element: the component does not exist — the
        # The inter-feed rate is tied ≡ 0.
        sold = _fringe_fit(ps, FringeClosure(; gauge = PinAntenna(1)))
        @test length(sold.components) == 3
    end

    @testset "corrections before the fit, including a function" begin
        ps, _ = _build_fringe_ps()
        ws = Dict("A1" => 1.0, "A2" => 0.5, "A3" => 1.0, "A4" => 2.0)
        # Weight scale: on noiseless data every baseline's delay and rate are
        # exact, so re-weighting baselines leaves the fringe θ where it was.
        scaled = Gustavo.materialize(ps)
        foreach(ms -> scale_weights!(ms, ws), values(scaled))
        sol_ws = _fringe_fit(scaled, FringeClosure(; gauge = PinAntenna(1)))
        sol = _fringe_fit(ps, FringeClosure(; gauge = PinAntenna(1)))
        @test all(
            isapprox(parent(a.params), parent(b.params); rtol = 1.0e-4)
                for (a, b) in zip(sol_ws.components, sol.components)
        )

        # Flagging one baseline before the fit keeps it flagged in the
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
        sol_cf = _fringe_fit(killed, FringeClosure(; gauge = PinAntenna(1)))
        calibrated = calibrate(sol_cf, killed)
        for ms in values(calibrated), (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
            is12(a, b) && @test all(view(ms[:flag], BaselineID(bi)))
        end
    end

    @testset "a solve uses samples whose input flags are cleared" begin
        ps, _ = _build_fringe_ps(; noise = 0.3)
        closure = FringeClosure(; gauge = PinAntenna(1))
        params(sol) = [parent(c.params) for c in sol.components]
        reference = params(_fringe_fit(ps, closure))
        data = Gustavo.materialize(ps)
        function set_flags!(value)
            for ms in values(data), (bi, pair) in pairs(collect(XRadio.baselines(ms)))
                pair == ("A1", "A2") || continue
                view(ms[:flag], BaselineID(bi), Ti(1:(length(XRadio.times(ms)) ÷ 2))) .= value
            end
            return data
        end
        flagged = params(_fringe_fit(set_flags!(true), closure))
        @test !all(isapprox.(flagged, reference; rtol = 1.0e-6))
        @test all(isapprox.(params(_fringe_fit(set_flags!(false), closure)), reference; rtol = 1.0e-10))
    end

    @testset "model validation + option coverage ahead of later fits" begin
        ps, _ = _build_fringe_ps()
        # The model is the component tree alone, and the gauge a field of its own —
        # no per-effect fields or keywords on FringeClosure.
        @test fieldnames(typeof(FringeClosure(; gauge = PinAntenna(1)))) == (:model, :closure, :gauge)

        # A custom Stationization and the inter-feed rate opt-in run ahead of
        # the bandpass and adhoc solvers.
        gauge = PinAntenna(1)
        sols_full = _fit_chain(
            (
                FringeClosure(;
                    model = merge(
                        default_fringe_model();
                        phase = (; rel_rate = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Feed = CAL.SingleFeed(2))),
                    ),
                    closure = FP.Stationization(pfa_max = 1.0e-2),
                    gauge,
                ),
                Bandpass(; gauge), AdhocPhase(; gauge),
            ),
            ps; exec = ExecutionConfig(),
        )
        @test [s.name for s in sols_full] == [:fringe, :bandpass, :adhoc]
        # Bandpass after the fringe fit alone still solves a :bandpass stage.
        _, sol_fb = _fit_chain((FringeClosure(; gauge), Bandpass(; gauge)), ps)
        @test sol_fb.name === :bandpass && !isempty(sol_fb.components)

        # Any real `steer_cells`, and a Float32 station solve, agreeing with the default.
        @test FringeSearch(; steer_cells = 9).steer_cells === 9.0
        sol64 = _fringe_fit(ps, FringeClosure(; gauge = PinAntenna(1)))
        sol32 = _fringe_fit(ps, FringeClosure(; closure = FP.Stationization(eltype = Float32), gauge = PinAntenna(1)))
        for (c32, c64) in zip(sol32.components, sol64.components)
            @test c32.params ≈ c64.params atol = 1.0e-6
        end
        @test sol32.info.flagged_ant == sol64.info.flagged_ant
    end

    @testset "model compilation: order, gating, duplicate rejection" begin
        ps, _ = _build_fringe_ps()   # 2 band groups; narrow fractional bandwidth
        geom = CAL.DataGeometry(ps)
        fringe_phase(model) =
            Gustavo._vetted_model(FringeClosure(; model, gauge = PinAntenna(1)), geom).phase
        sig(tc) = (
            typeof(tc.term), typeof(tc.Ti),
            typeof(tc.Frequency), typeof(tc.Feed),
        )

        # The default model compiles in order to the standard sequence, with no
        # constant phase (see `default_fringe_model`).
        comps = fringe_phase(default_fringe_model())
        @test collect(map(sig, CAL._flatten_components(comps))) == [
            (CAL.Delay, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
            (CAL.Delay, CAL.PerScan, CAL.GlobalFrequency, CAL.ExceptFeed),
            (CAL.Rate, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
        ]

        # `rel_time` moves the inter-feed delay onto a track-global column.
        gcomps = fringe_phase(default_fringe_model(rel_time = CAL.GlobalTime()))
        @test collect(map(sig, CAL._flatten_components(gcomps))) == [
            (CAL.Delay, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
            (CAL.Delay, CAL.GlobalTime, CAL.GlobalFrequency, CAL.ExceptFeed),
            (CAL.Rate, CAL.PerScan, CAL.GlobalFrequency, CAL.SharedFeeds),
        ]

        # Exact duplicate components are rejected by message.
        dup = merge(
            default_fringe_model();
            phase = (; mbd2 = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds())),
        )
        @test_throws "two identical components" fringe_phase(dup)

        # A second component matching a findfirst router's signature — without
        # being an exact duplicate — is rejected naming the signature.
        collide = merge(
            default_fringe_model();
            phase = (; mbd_tb = CAL.GainComponent(CAL.Delay(); Ti = CAL.TimeBlocks(1.0e6), Feed = CAL.SharedFeeds())),
        )
        @test_throws "per-scan feed-common delay signature" fringe_phase(collide)

        withphase(; kw...) = merge(default_fringe_model(); phase = NamedTuple(kw))

        # A constant phase is not fitted: the data do not determine it.
        @test_throws "It fits a `Delay` or `Rate`" fringe_phase(
            withphase(atmos = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds())),
        )

        # Dispersion and a per-band-group delay are rejected: the search
        # measures one delay across the band.
        @test_throws "It fits no dispersion" fringe_phase(
            withphase(dtec2 = CAL.GainComponent(CAL.Dispersion(); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds())),
        )
        @test_throws "It fits no dispersion" fringe_phase(
            withphase(sbd2 = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Frequency = CAL.FreqGroups([1:1, 2:length(geom.channel_freqs)]), Feed = CAL.SharedFeeds())),
        )

        # The fringe closure fits phase only.
        @test_throws "`logamp` group must be empty" fringe_phase(
            merge(default_fringe_model(); logamp = (; a = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds()))),
        )
    end
end

# ── What the fringe closure can fit ──────────────────────────────────────────

@testset "fringe closure capability" begin
    ps, _ = _build_fringe_ps()
    model = default_fringe_model()

    @testset "the detections record the search configuration" begin
        @test metadata(only(fit(FringeSearch(), ps))).search == FringeSearch()
        sol = _fringe_fit(ps, FringeClosure(; model, gauge = PinAntenna(1)))
        @test haskey(sol.info, :flagged_ant) && haskey(sol.info, :flagged_scan)
    end

    @testset "a term the fringe closure cannot fit is rejected by name" begin
        # A polynomial-in-frequency phase is a legitimate gain term that the
        # matched filter has no observable for: its θ block would stay at zero
        # while the solution looked fitted.
        poly = CAL.GainComponent(CAL.PolynomialFreq(2); Ti = CAL.PerScan(), Feed = CAL.SharedFeeds())
        @test_throws "FringeClosure cannot fit the component" _fringe_fit(
            ps, FringeClosure(; model = merge(default_fringe_model(); phase = (; poly)), gauge = PinAntenna(1)),
        )
    end

    @testset "a model missing a term the fringe closure requires is rejected by name" begin
        # The kind is missing outright: nothing to write the rate search into.
        norate = GainModel(; phase = Base.structdiff(default_fringe_model().phase, (; rate = nothing)))
        @test_throws "requires a rate component" _fringe_fit(ps, FringeClosure(; model = norate, gauge = PinAntenna(1)))

    end

    @testset "per-scan antenna solutions need a per-scan feed-common delay" begin
        globaldelay = map(default_fringe_model().phase) do t
            FP._is_perscan_delay(t) ?
                CAL.GainComponent(t.term; Ti = CAL.GlobalTime(), Frequency = t.Frequency, Feed = t.Feed) : t
        end
        sol = _fringe_fit(ps, FringeClosure(; model = GainModel(; phase = globaldelay), gauge = PinAntenna(1)))
        @test_throws "no per-scan (feed-common) delay component" FP.fringe_antenna_solutions(sol)
    end

    @testset "a time segmentation finer than a scan is rejected, not half-solved" begin
        # One search per scan group measures one delay, rate and phase per scan.
        # A segmentation splitting a scan asks for columns nothing writes: the
        # scan's first segment would be solved and the rest left at identity
        # gain while `calibrate` places each sample in its own segment's column.
        # `_build_fringe_ps` gives one 330 s scan, so 150 s blocks split it.
        subscan = map(
            t -> CAL.GainComponent(t.term; Ti = CAL.TimeBlocks(150.0), Frequency = t.Frequency, Feed = t.Feed),
            default_fringe_model().phase,
        )
        @test_throws "FringeClosure cannot fit the component" _fringe_fit(
            ps, FringeClosure(; model = GainModel(; phase = subscan), gauge = PinAntenna(1)),
        )
        @test_throws "a segmentation that splits a scan" _fringe_fit(
            ps, FringeClosure(; model = GainModel(; phase = subscan), gauge = PinAntenna(1)),
        )

        geom = CAL.DataGeometry(ps)
        mbd(Ti) = CAL.GainComponent(CAL.Delay(); Ti, Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds())

        # The same boundary expressed as an instrument scan edge, and the
        # per-integration limit the classifier used to special-case.
        @test !FP.can_fit(FringeClosure(; gauge = PinAntenna(1)), mbd(CAL.InstrumentScans([geom.times[1] + 150.0])), geom)
        @test !FP.can_fit(FringeClosure(; gauge = PinAntenna(1)), mbd(CAL.PerIntegration()), geom)

        # Coarser than a scan is fitted: one column several scans share is
        # written by all of them. The rule is the segmentation against the
        # geometry, not the segmentation alone — 3600 s blocks do not split a
        # 330 s scan.
        @test FP.can_fit(FringeClosure(; gauge = PinAntenna(1)), mbd(CAL.PerScan()), geom)
        @test FP.can_fit(FringeClosure(; gauge = PinAntenna(1)), mbd(CAL.GlobalTime()), geom)
        @test FP.can_fit(FringeClosure(; gauge = PinAntenna(1)), mbd(CAL.TimeBlocks(3600.0)), geom)
    end

    @testset "the matched filter's kind vocabulary stays private" begin
        # `matched_kind` answers "what does the matched filter do with this
        # component", an implementation detail of the fringe fit.
        @test !(:matched_kind in names(Gustavo.Fring))
        @test !(:matched_kind in names(Gustavo))
        @test :can_fit in names(Gustavo.Fring)
        @test :validate_model in names(Gustavo.Fring)
    end

end

@testset "antennas with one or three receptors" begin
    gauge = PinAntenna(1)
    for receptors in (("R",), ("R", "L", "X"))
        n = length(receptors)
        products = [a * b for a in receptors for b in receptors]
        ps, truth = _build_fringe_ps(; polarizations = products, receptors, noise = 0.3, eltype = ComplexF64)
        fr = _fringe_fit(ps, FringeClosure(; gauge))
        st = FP.fringe_antenna_solutions(fr)
        @test lookup(st, Gustavo.Feed) == 1:n
        @test !any(FP.fringe_antenna_flags(fr))
        d = parent(st.delay)[1, :, :]
        @test maximum(abs, (d .- d[1:1, :]) .- (truth.delay .- truth.delay[1:1, :])) < 5.0e-11

        # The bandpass flattens every product's spectrum.
        bp = 0.5 .* randn(MersenneTwister(0x268), 4, n, 2 * 8)
        bp[1, :, :] .= 0
        ps, _ = _build_fringe_ps(; polarizations = products, receptors, noise = 0.3, eltype = ComplexF64, bandpass = bp)
        fr = _fringe_fit(ps, FringeClosure(; gauge))
        bp_sol = fit(Bandpass(; gauge), _precal(fr, ps))
        @test lookup(only(bp_sol[:phase, :bandpass].components).params, Gustavo.Feed) == 1:n
        out = _calibrate_chain([fr, bp_sol], ps; flag_bad = false, apply_flags = false)
        coh = [
            abs(sum(z)) / sum(abs, z)
                for ms in values(out) for z in eachslice(dropdims(sum(parent(ms[:visibility]); dims = 4); dims = 4); dims = (1, 3))
        ]
        @test minimum(coh) > 0.97
    end
end

@testset "a prior along time ties a component across scans" begin
    ps, truth = _build_fringe_ps(; nscans = 5, noise = 1.0, eltype = ComplexF64)
    dets = fit(FringeSearch(), ps)
    level = GainComponent(Delay(); Ti = GlobalTime(), Feed = ExceptFeed(1))
    drifting(prior) = merge(
        default_fringe_model();
        phase = (; rel_delay = level, rel_drift = GainComponent(Delay(); Ti = PerScan(), Feed = ExceptFeed(1), prior)),
    )
    ou = drifting(OUPrior(; scale = 3600.0, σ = 3.0e-11))
    @test !Gustavo._closes_per_scan(ou)
    rel(sol) = (d = parent(FP.fringe_antenna_solutions(sol).delay); d[:, :, 2] .- d[:, :, 1])
    err(sol) = (r = rel(sol) .- (truth.delay[:, 2] .- truth.delay[:, 1])'; r .- r[:, 1:1])
    rms(x) = sqrt(sum(abs2, x) / length(x))

    # The prior pools the scans: the inter-feed delays scatter less about the
    # truth than scan-by-scan ones do.
    per_scan = fit(FringeClosure(; gauge = PinAntenna(1)), dets)
    pooled = fit(FringeClosure(; model = ou, gauge = PinAntenna(1)), dets)
    @test rms(err(pooled)) < rms(err(per_scan)) / 2

    # CARMA with one real root is the OU process.
    carma = drifting(CARMAPrior(; ar = (RealRoot(; scale = 3600.0),), σ = 3.0e-11))
    @test rel(fit(FringeClosure(; model = carma, gauge = PinAntenna(1)), dets)) ≈ rel(pooled) rtol = 1.0e-8

    # A flat walk has no mean of its own, so it stands without a level.
    walk = merge(
        default_fringe_model();
        phase = (; rel_delay = GainComponent(Delay(); Ti = PerScan(), Feed = ExceptFeed(1), prior = RandomWalkPrior(; order = 1, σ = 1.0e-13))),
    )
    @test rms(err(fit(FringeClosure(; model = walk, gauge = PinAntenna(1)), dets))) < rms(err(per_scan)) / 2
end

@testset "a feed the detections leave weak takes the prior's value" begin
    ps, truth = _build_fringe_ps(; nscans = 5, noise = 1.0, eltype = ComplexF64)
    data = Gustavo.materialize(ps)
    # Scan 3's products on A3's feed 2 keep their data at a millionth of the weight.
    for ms in values(data)
        only(unique(ms[:scan_name])) == "3" || continue
        feeds = feed_pairs(ms)
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms))), p in axes(feeds, 1)
            fa, fb = feeds[p, bi]
            ((a == "A3" && fa == 2) || (b == "A3" && fb == 2)) || continue
            view(ms[:weight], Polarization(p), BaselineID(bi)) .*= 1.0e-6
        end
    end
    dets = fit(FringeSearch(), data)
    model = merge(
        default_fringe_model();
        phase = (;
            rel_delay = GainComponent(Delay(); Ti = GlobalTime(), Feed = ExceptFeed(1)),
            rel_drift = GainComponent(Delay(); Ti = PerScan(), Feed = ExceptFeed(1), prior = OUPrior(; scale = 3600.0, σ = 3.0e-11)),
        ),
    )
    flagged(sol) = Set(zip(sol.info.flagged_ant, sol.info.flagged_feed, sol.info.flagged_scan))
    @test ("A3", 2, 3) in flagged(fit(FringeClosure(; gauge = PinAntenna(1)), dets))
    sol = fit(FringeClosure(; model, gauge = PinAntenna(1)), dets)
    @test !(("A3", 2, 3) in flagged(sol))
    d = parent(FP.fringe_antenna_solutions(sol).delay)
    @test d[3, 3, 2] - d[3, 3, 1] ≈ truth.delay[3, 2] - truth.delay[3, 1] atol = 1.0e-10
end

@testset "the gauge fixes only what the data and the prior leave free" begin
    # Without products relating the feeds, one inter-feed level per track is
    # free; the drift under its prior is not, so its values do not depend on
    # the gauge's reference.
    ps, _ = _build_fringe_ps(; nscans = 4, noise = 1.0, eltype = ComplexF64, polarizations = ["RR", "LL"])
    dets = fit(FringeSearch(), ps)
    model = merge(
        default_fringe_model();
        phase = (;
            rel_delay = GainComponent(Delay(); Ti = GlobalTime(), Feed = ExceptFeed(1)),
            rel_drift = GainComponent(Delay(); Ti = PerScan(), Feed = ExceptFeed(1), prior = OUPrior(; scale = 3600.0, σ = 3.0e-11)),
        ),
    )
    rel(sol) = (d = parent(FP.fringe_antenna_solutions(sol).delay); d[:, :, 2] .- d[:, :, 1])
    a = rel(fit(FringeClosure(; model, gauge = PinAntenna(1)), dets))
    b = rel(fit(FringeClosure(; model, gauge = PinAntenna(2)), dets))
    @test a .- a[:, 1:1] ≈ b .- b[:, 1:1] atol = 1.0e-15
    drift(gauge) = parent(only(fit(FringeClosure(; model, gauge), dets)[:phase, :rel_drift].components).params)
    @test drift(PinAntenna(1)) ≈ drift(PinAntenna(2)) atol = 1.0e-15
end

# Detections of a fringe search with every cross-hand cell replaced by the
# delay a solution predicts, at a tenth of the parallel hands' SNR, those on
# `leaky`'s feed 1 offset by `offset`, as a leaked fringe would sit.
function _with_cross_hands(dets, sol; leaky, offset)
    st = FP.fringe_antenna_solutions(sol).delay
    return map(dets) do d
        d = deepcopy(d)
        scan = metadata(d).scan
        τ(a, f) = st[Gustavo.Scan(At(scan)), AntennaName(At(a)), Feed(At(f))]
        for (a, b) in lookup(d, AntennaPair), (fa, fb) in lookup(d, FeedPair)
            fa == fb && continue
            cell = (AntennaPair(At((a, b))), FeedPair(At((fa, fb))))
            leaked = (a == leaky && fa == 1) || (b == leaky && fb == 1)
            d.delay[cell...] = τ(a, fa) - τ(b, fb) + (leaked ? (a == leaky ? offset : -offset) : 0.0)
            d.snr[cell...] = minimum(d.snr[FeedPair(At((fa, fa)))]) / 10
            d.pfa[cell...] = 1.0e-30
            d.valid[cell...] = true
        end
        d
    end
end

@testset "OutlierMixture separates leaked cross-hand fringes" begin
    ps, _ = _build_fringe_ps(; nscans = 4, noise = 1.0, eltype = ComplexF64)
    dets = fit(FringeSearch(), ps)
    reference = fit(FringeClosure(; gauge = PinAntenna(1)), dets)
    clean = _with_cross_hands(dets, reference; leaky = "A3", offset = 0.0)
    leaky = _with_cross_hands(dets, reference; leaky = "A3", offset = 7.0e-9)
    rel(sol) = (d = parent(FP.fringe_antenna_solutions(sol).delay); d[:, :, 2] .- d[:, :, 1])
    lsq = Stationization(; loss = FP.LeastSquares())
    truth = rel(fit(FringeClosure(; gauge = PinAntenna(1), closure = lsq), clean))

    # Least squares spreads the leaked fringes over the inter-feed delays.
    @test maximum(abs, rel(fit(FringeClosure(; gauge = PinAntenna(1), closure = lsq), leaky)) .- truth) > 1.0e-10

    mixture = Stationization(; loss = OutlierMixture())
    sol = @test_logs (:warn, r"A3 feed 1, different-feed detections: 12 of 12 unexplained, median offset (6\.9|7\.0)\d* ns") match_mode = :any fit(
        FringeClosure(; gauge = PinAntenna(1), closure = mixture), leaky,
    )
    @test maximum(abs, rel(sol) .- truth) < 3.0e-11
    @test sol.info.mixture.delay.different_feed.outlier_fraction ≈ 12 / 48 atol = 0.02
    @test sol.info.em_iterations.delay < mixture.max_em_iterations
end

@testset "a station whose detections are all unexplained is unflagged and uncorrected" begin
    ps, _ = _build_fringe_ps(; nscans = 4, noise = 1.0, eltype = ComplexF64)
    dets = fit(FringeSearch(), ps)
    # In scan 2, A4's accepted detections sit 6 ns off on either side, and
    # only its rejected ones agree with the other stations.
    d = dets[2]
    for (a, b) in lookup(d, AntennaPair), f in lookup(d, FeedPair)
        "A4" in (a, b) || continue
        other = a == "A4" ? b : a
        off = Dict("A1" => 6.0e-9, "A2" => -6.0e-9, "A3" => 0.0)[other] * (f[1] == 2 ? -1 : 1)
        cell = (AntennaPair(At((a, b))), FeedPair(At(f)))
        d.delay[cell...] += a == "A4" ? off : -off
        other == "A3" && (d.pfa[cell...] = 1.0)
    end
    sol = fit(FringeClosure(; gauge = PinAntenna(1), closure = Stationization(; loss = OutlierMixture())), dets)
    @test isempty(sol.info.flagged_ant)
    delay = FP.fringe_antenna_solutions(sol).delay
    @test all(iszero, delay[Gustavo.Scan(At(metadata(d).scan)), AntennaName(At("A4"))])
    @test !any(iszero, delay[Gustavo.Scan(At(metadata(d).scan)), AntennaName(At("A2"))])
end

@testset "EM estimates a drift prior's hyperparameters per station" begin
    ps, _ = _build_fringe_ps(; nscans = 8, noise = 1.0, eltype = ComplexF64, scan_gap = 1800.0)
    dets = fit(FringeSearch(), ps)
    # A2's feed-2 delays drift by 0.1 ns per scan.
    for (k, d) in pairs(dets), (a, b) in lookup(d, AntennaPair), (fa, fb) in lookup(d, FeedPair)
        shift = (a == "A2" && fa == 2 ? 1 : 0) - (b == "A2" && fb == 2 ? 1 : 0)
        shift == 0 || (d.delay[AntennaPair(At((a, b))), FeedPair(At((fa, fb)))] += shift * 1.0e-10 * k)
    end
    model = merge(
        default_fringe_model();
        phase = (;
            rel_delay = GainComponent(Delay(); Ti = GlobalTime(), Feed = ExceptFeed(1)),
            rel_drift = GainComponent(Delay(); Ti = PerScan(), Feed = ExceptFeed(1),
                prior = OUPrior(; scale = LogNormal(log(3600.0), 1.0), σ = LogNormal(log(3.0e-11), 2.0))),
        ),
    )
    sol = fit(FringeClosure(; model, gauge = PinAntenna(1)), dets)
    priors = sol.info.priors.rel_drift
    @test dims(priors) == (AntennaName(["A1", "A2", "A3", "A4"]), Gustavo.FeedNode([1]))
    σ = [p.σ for p in parent(priors)[:, 1]]
    @test σ[2] > 5 * maximum(σ[[1, 3, 4]])
    @test sol.info.em_iterations.delay < Stationization().max_em_iterations

    path = tempname()
    save_solution(path, sol)
    @test load_solution(path).info.priors.rel_drift == priors
end
