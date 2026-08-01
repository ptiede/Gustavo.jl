# Per-station weight correction (`station_weight_scale` / the `weight_scale`
# option): a NOISE-ESTIMATE fix for correlator weights that are miscalibrated on
# particular stations. Applied at every scan materialization, so it must reach
# the search pass, the leaf (pass 2 / export) path and the diagnostics — and it
# must NOT touch the visibilities. Reuses `_build_fringe_uvset` + the FP/CAL/UVP
# aliases from test_pipeline.jl (included earlier in runtests.jl).

@testset "Per-station weight scale" begin
    uvset, truth = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)
    ant_names = String.(collect(UVP.metadata(first(values(UVP.branches(uvset)))).antennas.name))
    nant = length(ant_names)

    @testset "station_weight_scale maps codes to station indices" begin
        # A2 and A3 affected; "ZZ" is not in the data and must be ignored.
        ws = FP.station_weight_scale(uvset, Dict("A2" => 0.5, "A3" => 0.25, "ZZ" => 7.0))
        @test ws == [1.0, 0.5, 0.25, 1.0]
        @test FP.station_weight_scale(ant_names, Dict("A2" => 0.5)) == [1.0, 0.5, 1.0, 1.0]
        @test FP.station_weight_scale(ant_names, Dict(); default = 2.0) == fill(2.0, nant)
        @test FP.station_weight_scale(ant_names, Dict(:A1 => 0.5))[1] == 0.5     # Symbol keys
        # Zero/negative factors are rejected at transform construction.
        @test_throws ErrorException FP.StationWeightScale([1.0, 0.0, 1.0, 1.0])
        @test_throws ErrorException FP.StationWeightScale([1.0, -1.0, 1.0, 1.0])
    end

    # The correction factorizes per station, so a baseline with ONE affected
    # station gets s and a baseline between two affected stations gets s² — the
    # 2×/4× pattern of a per-station correlator weight bug.
    ws = FP.station_weight_scale(uvset, Dict("A2" => 0.5, "A3" => 0.5))
    st0 = FP.scan_stream(uvset; geom = geom)
    stw = FP.scan_stream(uvset; geom = geom, transforms = (FP.StationWeightScale(ws),))

    @testset "scan-group path scales weights, not visibilities" begin
        plain, _ = FP.materialize_cube(st0, st0.groups[1])
        fixed, _ = FP.materialize_cube(stw, stw.groups[1])
        @test fixed[:vis] == plain[:vis]                             # visibilities untouched
        @test baselines(fixed).pairs == baselines(plain).pairs
        for (bi, (a, b)) in enumerate(baselines(fixed).pairs)
            s = ws[a] * ws[b]
            @test all(fixed[:weights][:, :, bi, :] .≈ Float32(s) .* plain[:weights][:, :, bi, :])
            @test s ≈ ((a, b) == (2, 3) ? 0.25 : (a in (2, 3) || b in (2, 3)) ? 0.5 : 1.0)
        end
    end

    @testset "leaf (output / export) path scales the same way" begin
        plain = FP.materialize_leaves(st0, st0.groups[1])
        fixed = FP.materialize_leaves(stw, stw.groups[1])
        for ((_, lp), (_, lf)) in zip(plain, fixed)
            pairs = collect(UVP.baselines(lf).pairs)
            @test parent(lf[:vis]) == parent(lp[:vis])
            for (bi, (a, b)) in enumerate(pairs)
                @test all(parent(lf[:weights])[:, :, bi, :] .≈
                          Float32(ws[a] * ws[b]) .* parent(lp[:weights])[:, :, bi, :])
            end
        end
    end

    @testset "search is invariant; exported weights carry the fix" begin
        # The search's SNR is data-driven (noise estimated from the |D|² plane,
        # not from Σw), so a PER-BASELINE weight rescale cancels exactly: the
        # detections — SNR, delay, rate — are bit-identical. What the fix moves
        # is the RELATIVE inter-baseline weighting of the stages that accumulate
        # ACROSS baselines (stage B / bandpass / adhoc) and the exported weights.
        chain0 = FringeFit(model = FringeModel(ref_ant = 1)) |> BandpassEstimator() |>
            TemporalSmoother(FP.NoSmoothing())
        base = fitcalibrate(chain0, uvset)
        fixd = fitcalibrate(FP.StationWeightScale(ws) |> chain0, uvset)
        bfr, ffr = CAL._step(base[1], :fringe), CAL._step(fixd[1], :fringe)
        @test ffr.info.scan_snr == bfr.info.scan_snr
        @test ffr.info.det_snr == bfr.info.det_snr
        # …so the solution only shifts at the level of the re-weighted stages.
        @test all(
            isapprox(s1.θ, s2.θ; atol = 1.0e-3) for (s1, s2) in zip(fixd[1].steps, base[1].steps)
        )

        for (k, leaf) in UVP.branches(fixd[2])
            pairs = collect(UVP.baselines(leaf).pairs)
            wb = parent(UVP.branches(base[2])[k][:weights])
            wf = parent(leaf[:weights])
            for (bi, (a, b)) in enumerate(pairs)
                @test all(wf[:, :, bi, :] .≈ Float32(ws[a] * ws[b]) .* wb[:, :, bi, :])
            end
        end
    end

    @testset "diagnostics replay the solve's recorded transforms" begin
        sol = fit(
            FP.StationWeightScale(ws) |> FringeFit(model = FringeModel(ref_ant = 1)) |>
                BandpassEstimator() |> TemporalSmoother(FP.NoSmoothing()),
            uvset,
        )
        # The old footgun (forgetting to re-pass weight_scale to a diagnostic)
        # is dead: with NO kwargs the diagnostics materialize through
        # `sol.transforms` — the recorded StationWeightScale — so the explicit
        # kwarg and the default now see IDENTICAL data.
        @test length(sol.transforms) == 1 && sol.transforms[1] isa FP.StationWeightScale
        m = FP.fringe_search_map(uvset, sol; weight_scale = ws)
        m0 = FP.fringe_search_map(uvset, sol)
        @test m.map.detection.snr ≈ m0.map.detection.snr
        d = FP.baseline_fringe_data(uvset, sol; weight_scale = ws)
        d0 = FP.baseline_fringe_data(uvset, sol)
        @test isequal(d.spec_after, d0.spec_after)
        @test isequal(d.spec_before, d0.spec_before)
    end
end
