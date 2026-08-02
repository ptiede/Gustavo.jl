# `fringe_station_solutions` (θ → per-(scan,station,feed) delay/rate/phase decode)
# and the `rl_delay = :global | :perscan` model option. Reuses `_build_fringe_uvset`
# and the CAL/FP/UVP aliases from test_pipeline.jl (included earlier in runtests.jl).

@testset "rl_delay model option + fringe_station_solutions" begin

    @testset "rl_delay switches only the R–L delay's time basis" begin
        mg = FP._fringe_model(rl_delay = :global)
        mp = FP._fringe_model(rl_delay = :perscan)
        @test length(mg.phase) == length(mp.phase)
        # Exactly one phase component differs between the two models, and only in its
        # time-segmentation type.
        diff = findall(i -> typeof(mg.phase[i].component.time) != typeof(mp.phase[i].component.time),
                       eachindex(mg.phase))
        @test length(diff) == 1
        i = only(diff)
        @test mg.phase[i].component.term isa FP.Delay          # it IS the R–L delay
        # Its basis matches a known-basis sibling: PerScan like phase[1] (per-scan const)
        # under :perscan, GlobalTime like phase[2] (global R–L const) under :global.
        @test typeof(mp.phase[i].component.time) == typeof(mp.phase[1].component.time)
        @test typeof(mg.phase[i].component.time) == typeof(mg.phase[2].component.time)
        @test typeof(mp.phase[i].component.time) != typeof(mg.phase[i].component.time)

        @test_throws ErrorException FP._fringe_model(rl_delay = :bogus)
    end

    @testset "θ decode: units + feed-2 = shared + R–L" begin
        uvset, _ = _build_fringe_uvset(nant = 4)
        geom = CAL.build_geometry(uvset)
        model = FP._fringe_model(rl_delay = :perscan)
        layout = CAL.plan_parameters(model, 4, geom)

        # The two GlobalFrequency delay plans: the feed-common per-scan one (has a
        # feed-1 column) and the R–L one (feed-2 only). Identified structurally.
        dplans = [
            p for (p, k) in FP.fringe_stage_components(model, layout)
                if k === :delay
        ]
        shared = only(filter(p -> plan_off1(p)[2, 1, 1, 1] != 0, dplans))
        rl = only(filter(p -> plan_off1(p)[2, 1, 1, 1] == 0 && plan_off1(p)[2, 2, 1, 1] != 0, dplans))

        θ = zeros(layout.nθ)
        θ[plan_off1(shared)[2, 1, 1, 1]] = 2.0e-9      # station 2 per-scan (feed-common) delay: 2 ns
        θ[plan_off1(rl)[2, 2, 1, 1]]     = 0.5e-9      # station 2 R–L delay: +0.5 ns on feed 2
        sol = CAL.CalibrationSolution(model, layout, geom, θ, (; nscan = 1); name = :fringe)

        rows = FP.fringe_station_solutions(sol)
        @test rows isa Vector{<:NamedTuple}
        @test length(rows) == 4 * 2               # dense: nscan(1) × nant(4) × 2 feeds
        f(st, fd) = only(filter(r -> r.scan == 1 && r.station == st && r.feed == fd, rows)).delay_ns
        @test f(2, 1) ≈ 2.0                        # feed 1: shared only, s → ns
        @test f(2, 2) ≈ 2.5                        # feed 2: shared + R–L
        @test f(1, 1) ≈ 0.0                        # untouched station stays identity-0
        @test f(3, 2) ≈ 0.0

        # Rate (Hz → mHz) and phase (rad → deg) route through their own kinds.
        rplan = only(
            p for (p, k) in FP.fringe_stage_components(model, layout)
                if k === :rate
        )
        θ2 = zeros(layout.nθ)
        θ2[plan_off1(rplan)[3, 1, 1, 1]] = 1.0e-3       # 1 mHz
        sol2 = CAL.CalibrationSolution(model, layout, geom, θ2, (; nscan = 1); name = :fringe)
        rows2 = FP.fringe_station_solutions(sol2)
        @test only(filter(r -> r.station == 3 && r.feed == 1, rows2)).rate_mHz ≈ 1.0
    end

    @testset "end-to-end: recovers injected per-feed delays from a solve" begin
        uvset, truth = _build_fringe_uvset(nant = 4)
        # Per-scan R–L delay: the default list with the feed-2 Delay element's
        # time basis switched from GlobalTime to PerScan.
        rl_perscan = map(
            t -> t isa CAL.TiedComponent && t.component.term isa CAL.Delay &&
                t.tying isa CAL.FeedComponent ?
                CAL.TiedComponent(CAL.Delay(), CAL.PerScan(), CAL.GlobalFrequency(), CAL.FeedComponent(2)) : t,
            default_fringe_terms(),
        )
        sol = fit(
            FringeFit(model = FringeModel(terms = rl_perscan)) |>
                BandpassEstimator() |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        rows = FP.fringe_station_solutions(sol)
        @test length(rows) == sol.info.nscan * sol.steps[1].layout.nant * 2
        val(st, fd) = only(filter(r -> r.scan == 1 && r.station == st && r.feed == fd, rows)).delay_ns

        # Gauge: reference station (1) is pinned to 0 on both feeds.
        @test val(1, 1) ≈ 0.0 atol = 1.0e-3
        # Recovery: absolute (ref-gauged) delay ≈ the injected per-feed delay (ns).
        # feed 1 ≈ shared per-scan delay; feed 2 ≈ that + the R–L delay = truth[a,2].
        for a in 2:4, fd in 1:2
            @test val(a, fd) ≈ truth.delay[a, fd] * 1.0e9 atol = 0.05
        end
    end
end
