# Modular calibration-pipeline tests: the pipeline-level surface
# (`CalibrationPipeline`, `fitcalibrate`, reduce steps, defaults).
# Reuses `_build_fringe_uvset` and the CAL/FP/UVP aliases from test_pipeline.jl
# (included earlier in runtests.jl).

# A throwaway reduce step proving the ReduceStep extension point: subtype + one
# `prepare_reducer` method (identity transform that records it ran).
struct _ProbeReduce <: Gustavo.ReduceStep
    seen::Base.RefValue{Bool}
end
Gustavo.prepare_reducer(s::_ProbeReduce, ctx::Gustavo.CalibrationContext) =
    ((uv -> (s.seen[] = true; uv)), ctx)

@testset "Calibration pipeline" begin
    @testset "fitcalibrate defaults (solve + corrected output)" begin
        uvset, _ = _build_fringe_uvset()
        pipe = CalibrationPipeline([FringeFit(), BandpassEstimator(), TemporalSmoother()])
        sol, out = fitcalibrate(pipe, uvset)
        @test sol isa CAL.CalibrationSolution
        @test out !== nothing                   # the full pipeline sets corrected output
        # The fused output tail does not perturb the solve: θ ≡ the fit-only θ.
        @test sol.θ == fit(pipe, uvset).θ
        @test sol.info.nant == 4
        @test isempty(sol.postcal)              # no a-priori step
    end

    @testset "pipeline ReduceSteps fuse into the streaming pass" begin
        uvset, _ = _build_fringe_uvset(nbands = 3, nchan = 4)
        chain = [FringeFit(), BandpassEstimator(), TemporalSmoother()]
        pipe = CalibrationPipeline(vcat(
            chain, [AverageFrequency(nout = 1), CombineSpw(), AverageTime(seconds = 1.0e6)],
        ))
        sol, out = fitcalibrate(pipe, uvset)
        # Equivalent hand-written reducer chain (same order) on the two-pass
        # corrected set.
        reducer = uv -> UVP.time_bin_average(UVP.combine_spw(UVP.frequency_average(uv; nout = 1)), 1.0e6)
        out_ref = reducer(Gustavo.apply_calibration(uvset, sol))
        @test Set(keys(DimensionalData.branches(out))) ==
            Set(keys(DimensionalData.branches(out_ref)))
        for (k, leaf) in DimensionalData.branches(out_ref)
            Vr = parent(leaf[:vis])
            Vf = parent(DimensionalData.branches(out)[k][:vis])
            @test size(Vf) == size(Vr)
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
        end
    end

    @testset "fitcalibrate reduce kwarg ≡ pipeline ReduceStep" begin
        uvset, _ = _build_fringe_uvset()
        chain = FringeFit() |> BandpassEstimator() |> TemporalSmoother()
        _, out_kw = fitcalibrate(chain, uvset; reduce = [AverageFrequency(nout = 1)])
        _, out_pl = fitcalibrate(
            CalibrationPipeline([
                FringeFit(), BandpassEstimator(), TemporalSmoother(), AverageFrequency(nout = 1),
            ]),
            uvset,
        )
        for (k, leaf) in DimensionalData.branches(out_kw)
            Vr = parent(leaf[:vis])
            Vf = parent(DimensionalData.branches(out_pl)[k][:vis])
            @test size(Vf) == size(Vr)
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
        end
    end

    @testset "extensibility: custom ReduceStep runs in the pipeline" begin
        uvset, _ = _build_fringe_uvset()
        seen = Ref(false)
        sol, _ = fitcalibrate(
            CalibrationPipeline([
                FringeFit(), BandpassEstimator(), TemporalSmoother(), _ProbeReduce(seen),
            ]),
            uvset,
        )
        @test seen[]
        @test sol isa CAL.CalibrationSolution

        # Arbitrary run_step-based action steps no longer thread through a
        # pipeline (solve steps share one compiled model + streaming passes).
        struct_probe = Gustavo.DataTransformStep(CalFunction(v -> nothing))
        @test_throws ErrorException Gustavo.run_step(struct_probe, Gustavo.CalibrationContext())
    end

    @testset "band edges" begin
        uvset, _ = _build_fringe_uvset(nbands = 2, nchan = 8)   # fraction 0.2 → 1 edge chan

        flagged = UVP.flag_band_edges(uvset; mode = :flag_fraction, fraction = 0.2)
        for (k, leaf) in DimensionalData.branches(flagged)
            W = parent(leaf[:weights])
            W0 = parent(DimensionalData.branches(uvset)[k][:weights])
            @test all(W[1, :, :, :] .== 0)
            @test all(W[end, :, :, :] .== 0)
            @test W[2:(end - 1), :, :, :] == W0[2:(end - 1), :, :, :]
        end

        trimmed = UVP.flag_band_edges(uvset; mode = :trim, fraction = 0.2)
        for (_, leaf) in DimensionalData.branches(trimmed)
            @test size(parent(leaf[:vis]), 1) == 6
            @test length(channel_freqs(DimensionalData.metadata(leaf).freq_setup)) == 6
        end

        @test_throws ErrorException UVP.flag_band_edges(uvset; mode = :bogus, fraction = 0.1)

        # As a fused reduce step on the corrected output.
        _, out = fitcalibrate(
            CalibrationPipeline(
                [
                    FringeFit(), BandpassEstimator(), TemporalSmoother(),
                    FlagBandEdges(mode = :flag_fraction, fraction = 0.2),
                ]
            ),
            uvset,
        )
        for (_, leaf) in DimensionalData.branches(out)
            W = parent(leaf[:weights])
            @test all(W[1, :, :, :] .== 0)
            @test all(W[end, :, :, :] .== 0)
        end
    end

    @testset "ref_ant by station code" begin
        uvset, _ = _build_fringe_uvset()    # antennas named A1..A4
        @test Gustavo._resolve_ref_ant(3, uvset) == 3
        @test Gustavo._resolve_ref_ant("A2", uvset) == 2
        @test Gustavo._resolve_ref_ant(:A4, uvset) == 4
        @test_throws ErrorException Gustavo._resolve_ref_ant("ZZ", uvset)
        # End-to-end (new engine): code "A1" resolves to index 1 → identical solve.
        by_code = fit(FringeFit(model = FringeModel(ref_ant = "A1")), uvset)
        by_idx = fit(FringeFit(model = FringeModel(ref_ant = 1)), uvset)
        @test by_code.θ ≈ by_idx.θ
    end

    @testset "defaults" begin
        f = FringeFit()
        @test f.model.ref_ant == 1
        @test f.model.delay isa Gustavo.PerScan && f.model.rate isa Gustavo.PerScan
        @test f.model.cross_feed.delay isa Gustavo.GlobalTime
        @test f.model.cross_feed.rate === nothing          # tied ≡ 0
        @test f.model.cross_feed.fit_on isa AllScans
        @test f.model.dispersion == :auto && f.model.sbd == :auto
        @test f.estimator isa MatchedFilter
        @test f.estimator.search == FP.FringeSearch()
        @test f.estimator.closure == FP.Stationization()
        @test f.estimator.rounds == 1
        @test f.reuse_bandpass_refine && f.polish_dtec == 20.0

        b = BandpassEstimator()
        @test b.phase && b.amp
        @test b.amp_model == FP.PenalizedBandpass(1.0)
        @test b.select == BrightestCalibrator()

        t = TemporalSmoother()
        @test t.smoother == FP.SavitzkyGolaySmoother()

        e = ExecutionConfig()
        @test e.mem_fraction == 0.6 && e.mem_budget === nothing && e.exclude_colocated

        # AprioriAmplitude carries a pre-built band_cals (loading is the caller's job).
        bc = Dict(1 => :dummy)
        ap = AprioriAmplitude(bc; min_elevation_deg = 10.0)
        @test ap.band_cals === bc
        @test ap.min_elevation_deg == 10.0
        @test ap.on_missing_station == :warn
    end
end
