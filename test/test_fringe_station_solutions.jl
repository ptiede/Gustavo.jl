# `fringe_station_solutions` (θ → per-(scan,station,feed) delay/rate/phase decode)
# and the `rel_time` model option. Reuses `_build_fringe_ps` and the
# CAL/FP aliases from test_pipeline.jl (included earlier in runtests.jl).

@testset "rel_time model option + fringe_station_solutions" begin

    @testset "rel_time switches the inter-feed delay's time basis" begin
        mg = _full_fringe_model(rel_time = CAL.GlobalTime())
        mp = _full_fringe_model(rel_time = CAL.PerScan())
        @test length(mg.phase) == length(mp.phase)
        # Exactly the one `ExceptFeed`-tied offset differs between the models,
        # and only in its time-segmentation type. (There is no inter-feed
        # CONSTANT — see `default_fringe_terms`.)
        diff = findall(
            i -> typeof(mg.phase[i].Ti) != typeof(mp.phase[i].Ti),
            eachindex(mg.phase)
        )
        @test length(diff) == 1
        for i in diff
            @test mg.phase[i].Feed isa CAL.ExceptFeed
            @test mg.phase[i].Ti isa CAL.GlobalTime
            @test mp.phase[i].Ti isa CAL.PerScan
        end
        # The offset is the delay.
        @test sort([nameof(typeof(mg.phase[i].term)) for i in diff]) == [:Delay]

        # Per-scan is the default: the inter-feed offsets carry no cross-scan column.
        md = _full_fringe_model()
        @test all(
            tc -> tc.Ti isa CAL.PerScan,
            filter(tc -> tc.Feed isa CAL.ExceptFeed, collect(md.phase)),
        )
    end

    @testset "θ decode: units + feed-2 = shared + inter-feed offset" begin
        ps, _ = _build_fringe_ps(nant = 4)
        geom = CAL.DataGeometry(ps)
        model = _full_fringe_model()
        layout = CAL.plan_parameters(model, 4, geom)

        # The two GlobalFrequency delay plans: the feed-common per-scan one (has a
        # feed-1 column) and the inter-feed one (feed-2 only). Identified structurally.
        dplans = [
            p for (p, k) in FP.fringe_stage_components(model, layout)
                if k === :delay
        ]
        shared = only(filter(p -> plan_off1(p)[2, 1, 1, 1] != 0, dplans))
        rel = only(filter(p -> plan_off1(p)[2, 1, 1, 1] == 0 && plan_off1(p)[2, 2, 1, 1] != 0, dplans))

        θ = zeros(layout.nθ)
        θ[plan_off1(shared)[2, 1, 1, 1]] = 2.0e-9      # station 2 per-scan (feed-common) delay: 2 ns
        θ[plan_off1(rel)[2, 2, 1, 1]] = 0.5e-9      # station 2 inter-feed delay: +0.5 ns on feed 2
        sol = CAL.CalibrationSolution(model, layout, geom, θ, (; nscan = 1); name = :fringe)

        st = FP.fringe_station_solutions(sol)
        @test st isa DimStack
        @test keys(st) == (:delay, :rate)
        @test size(st) == (1, 4, 2)               # dense: nscan(1) × nant(4) × 2 feeds
        @test collect(lookup(st, AntennaName)) == geom.stations
        f(a, fd) = st.delay[Gustavo.Scan(1), AntennaName(a), Feed(fd)]
        @test f(2, 1) ≈ 2.0e-9                     # feed 1: shared only
        @test f(2, 2) ≈ 2.5e-9                     # feed 2: shared + inter-feed offset
        @test f(1, 1) ≈ 0.0                        # untouched station stays identity-0
        @test f(3, 2) ≈ 0.0

        # Rate and phase route through their own kinds.
        rplan = only(
            p for (p, k) in FP.fringe_stage_components(model, layout)
                if k === :rate
        )
        θ2 = zeros(layout.nθ)
        θ2[plan_off1(rplan)[3, 1, 1, 1]] = 1.0e-3       # 1 mHz
        sol2 = CAL.CalibrationSolution(model, layout, geom, θ2, (; nscan = 1); name = :fringe)
        st2 = FP.fringe_station_solutions(sol2)
        @test st2.rate[Gustavo.Scan(1), AntennaName(3), Feed(1)] ≈ 1.0e-3
    end

    @testset "end-to-end: recovers injected per-feed delays from a solve" begin
        ps, truth = _build_fringe_ps(nant = 4)
        gauge = PinAntenna(1)
        sol = fit(BaselineFringeFit(; gauge), ps)
        st = FP.fringe_station_solutions(sol)
        @test size(st) == (sol.info.nscan, length(sol.geom.stations), 2)
        @test collect(lookup(st, Gustavo.Scan)) == sol.geom.scan_names
        val(a, fd) = st.delay[Gustavo.Scan(1), AntennaName(a), Feed(fd)]

        # Gauge: reference station (1) is pinned to 0 on both feeds.
        @test val(1, 1) ≈ 0.0 atol = 1.0e-12
        # Recovery: absolute (ref-gauged) delay ≈ the injected per-feed delay.
        # feed 1 ≈ shared per-scan delay; feed 2 ≈ that + the inter-feed delay = truth[a,2].
        for a in 2:4, fd in 1:2
            @test val(a, fd) ≈ truth.delay[a, fd] atol = 0.05e-9
        end
    end
end
