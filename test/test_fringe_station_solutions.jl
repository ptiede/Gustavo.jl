# `fringe_station_solutions` (θ → per-(scan,station,feed) delay/rate/phase decode)
# and the `rel_time` model option. Reuses `_build_fringe_uvset` and the
# CAL/FP/UVP aliases from test_pipeline.jl (included earlier in runtests.jl).

@testset "rel_time model option + fringe_station_solutions" begin

    @testset "rel_time switches both inter-feed offsets' time basis" begin
        mg = FP._fringe_model(rel_time = CAL.GlobalTime())
        mp = FP._fringe_model(rel_time = CAL.PerScan())
        @test length(mg.phase) == length(mp.phase)
        # Exactly the two `FeedComponent`-tied offsets differ between the models,
        # and only in their time-segmentation type.
        diff = findall(i -> typeof(mg.phase[i].component.time) != typeof(mp.phase[i].component.time),
                       eachindex(mg.phase))
        @test length(diff) == 2
        for i in diff
            @test mg.phase[i].tying isa CAL.FeedComponent
            @test mg.phase[i].component.time isa CAL.GlobalTime
            @test mp.phase[i].component.time isa CAL.PerScan
        end
        # The two offsets are the constant phase and the delay.
        @test sort([nameof(typeof(mg.phase[i].component.term)) for i in diff]) ==
            [:ConstantTerm, :Delay]

        # Per-scan is the default: the inter-feed offsets carry no cross-scan column.
        md = FP._fringe_model()
        @test all(
            tc -> tc.component.time isa CAL.PerScan,
            filter(tc -> tc.tying isa CAL.FeedComponent, collect(md.phase)),
        )
    end

    @testset "θ decode: units + feed-2 = shared + inter-feed offset" begin
        uvset, _ = _build_fringe_uvset(nant = 4)
        geom = CAL.build_geometry(uvset)
        model = FP._fringe_model()
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
        θ[plan_off1(rel)[2, 2, 1, 1]]    = 0.5e-9      # station 2 inter-feed delay: +0.5 ns on feed 2
        sol = CAL.CalibrationSolution(model, layout, geom, θ, (; nscan = 1); name = :fringe)

        rows = FP.fringe_station_solutions(sol)
        @test rows isa Vector{<:NamedTuple}
        @test length(rows) == 4 * 2               # dense: nscan(1) × nant(4) × 2 feeds
        f(st, fd) = only(filter(r -> r.scan == 1 && r.station == st && r.feed == fd, rows)).delay_ns
        @test f(2, 1) ≈ 2.0                        # feed 1: shared only, s → ns
        @test f(2, 2) ≈ 2.5                        # feed 2: shared + inter-feed offset
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
        sol = fit(
            FringeFit(model = FringeModel(terms = default_fringe_terms())) |>
                Bandpass() |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        rows = FP.fringe_station_solutions(sol)
        @test length(rows) == sol.info.nscan * sol.steps[1].layout.nant * 2
        val(st, fd) = only(filter(r -> r.scan == 1 && r.station == st && r.feed == fd, rows)).delay_ns

        # Gauge: reference station (1) is pinned to 0 on both feeds.
        @test val(1, 1) ≈ 0.0 atol = 1.0e-3
        # Recovery: absolute (ref-gauged) delay ≈ the injected per-feed delay (ns).
        # feed 1 ≈ shared per-scan delay; feed 2 ≈ that + the inter-feed delay = truth[a,2].
        for a in 2:4, fd in 1:2
            @test val(a, fd) ≈ truth.delay[a, fd] * 1.0e9 atol = 0.05
        end
    end
end
