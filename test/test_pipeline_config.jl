# Modular calibration-pipeline tests (the `CalibrationPipeline` / `calibrate`
# refactor). Reuses `_build_fringe_uvset` and the CAL/FP/UVP aliases from
# test_pipeline.jl (included earlier in runtests.jl).

# A throwaway step proving the extension point: subtype + one `run_step` method.
struct _ProbeStep <: Gustavo.CalibrationStep
    seen::Base.RefValue{Bool}
end
Gustavo.run_step(s::_ProbeStep, ctx::Gustavo.CalibrationContext) = (s.seen[] = true; ctx)

@testset "Calibration pipeline" begin
    @testset "calibrate ≡ solve_fringes (defaults preserve behavior)" begin
        uvset, _ = _build_fringe_uvset()
        ref = FP.solve_fringes(uvset)
        res = calibrate(uvset, CalibrationPipeline([FringeFit()]))
        @test res.solution isa CAL.CalibrationSolution
        @test res.solution.θ ≈ ref.θ
        @test res.solution.info.nant == ref.info.nant
        @test res.solution.info.nscan == ref.info.nscan
        @test res.output !== nothing            # FringeFit always sets corrected output
        @test res.band_cals === nothing         # no a-priori step
    end

    @testset "FringeFit(reduce=…) fuses into the streaming pass" begin
        uvset, _ = _build_fringe_uvset(nbands = 3, nchan = 4)
        pipe = CalibrationPipeline(
            [
                FringeFit(reduce = [AverageFrequency(nout = 1), CombineSpw(), AverageTime(seconds = 1.0e6)]),
            ]
        )
        res = calibrate(uvset, pipe)
        # Equivalent hand-written reducer chain (same order).
        reducer = uv -> UVP.time_bin_average(UVP.combine_spw(UVP.frequency_average(uv; nout = 1)), 1.0e6)
        _, out_ref = FP.solve_and_reduce_fringes(uvset; postprocess = reducer)
        @test Set(keys(DimensionalData.branches(res.output))) ==
            Set(keys(DimensionalData.branches(out_ref)))
        for (k, leaf) in DimensionalData.branches(out_ref)
            Vr = parent(leaf[:vis])
            Vf = parent(DimensionalData.branches(res.output)[k][:vis])
            @test size(Vf) == size(Vr)
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
        end
    end

    @testset "standalone ReduceStep runs eagerly after a fit" begin
        uvset, _ = _build_fringe_uvset()
        res = calibrate(uvset, CalibrationPipeline([FringeFit(), AverageFrequency(nout = 1)]))
        _, corr = FP.solve_and_reduce_fringes(uvset)        # corrected full set
        ref = UVP.frequency_average(corr; nout = 1)
        for (k, leaf) in DimensionalData.branches(ref)
            Vr = parent(leaf[:vis])
            Vf = parent(DimensionalData.branches(res.output)[k][:vis])
            @test size(Vf) == size(Vr)
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
        end
    end

    @testset "extensibility: custom step is run and threads context" begin
        uvset, _ = _build_fringe_uvset()
        seen = Ref(false)
        res = calibrate(uvset, CalibrationPipeline([_ProbeStep(seen), FringeFit()]))
        @test seen[]
        @test res.solution isa CAL.CalibrationSolution
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
        res = calibrate(
            uvset, CalibrationPipeline(
                [
                    FringeFit(reduce = [FlagBandEdges(mode = :flag_fraction, fraction = 0.2)]),
                ]
            )
        )
        for (_, leaf) in DimensionalData.branches(res.output)
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
        # End-to-end: code "A1" resolves to index 1 → identical solve.
        by_code = calibrate(uvset, CalibrationPipeline([FringeFit(ref_ant = "A1")]))
        by_idx = calibrate(uvset, CalibrationPipeline([FringeFit(ref_ant = 1)]))
        @test by_code.solution.θ ≈ by_idx.solution.θ
    end

    @testset "defaults" begin
        f = FringeFit()
        @test f.ref_ant == 1 && f.rounds == 1
        @test f.search == FP.FringeSearch()
        @test f.adhoc == FP.SavitzkyGolaySmoother()
        @test f.bandpass.phase && f.bandpass.amp && f.bandpass.source === nothing
        @test f.mem_fraction == 0.6 && f.mem_budget === nothing   # deterministic-budget defaults
        @test isempty(f.reduce)

        # AprioriAmplitude carries a pre-built band_cals (loading is the caller's job).
        bc = Dict(1 => :dummy)
        ap = AprioriAmplitude(bc; min_elevation_deg = 10.0)
        @test ap.band_cals === bc
        @test ap.min_elevation_deg == 10.0
        @test ap.on_missing_station == :warn
    end
end
