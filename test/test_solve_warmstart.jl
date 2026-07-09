# Gustavo.Solve — FFT + stationization warm-start (Stage 2, Milestone 5c).
#
# The forward-model objective is non-convex in delay: once a station delay wraps
# the phase across the band (τ > 1/BW), a zero start locks onto a matched-filter
# sidelobe and LBFGS cannot escape. `fft_warmstart` seeds the delay/phase basin
# from the KEPT FFT search + closure stationization; from that seed LBFGS polishes
# to machine precision. This test injects LARGE (wrapping) delays so the warm-start
# is the difference between a good fit and a stuck one.
#
# Reuses `_build_fringe_uvset` (test_pipeline.jl) and `_forward_uvset`
# (test_solve_fit.jl), so it is included after both in runtests.jl.

using Gustavo.Solve
using Gustavo.Solve: fringe_solve, fft_warmstart, seed_from_stationization!,
    plan_gains, zero_params, evaluate_gains, point_source_coherency,
    PointSource, fringe_objective
import Gustavo.UVData as UV
using Gustavo.Calibration:
    build_geometry, leaf_window, predict_visibilities, correlation_feed_pair,
    StationGainModel, GainComponent, TiedComponent, Delay, ConstantTerm, PerChannel,
    PerScan, GlobalTime, GlobalFrequency, PerSpectralWindow, PerFeed
using Gustavo.Fringe: FringeSearch, Stationization, stationize_scan
using Enzyme, Optimization, OptimizationOptimJL     # enable Enzyme + LBFGS extensions
using Random: MersenneTwister
using Test

@testset "Solve M5c: FFT warm-start seeds the delay basin" begin
    # A single wide contiguous band → clean 2-D FFT search; 32 channels × 2 MHz =
    # 64 MHz bandwidth, so one phase cycle at τ = 1/BW ≈ 15.6 ns.
    uvset0, _ = _build_fringe_uvset(nant = 4, nbands = 1, nchan = 32, ntime = 8, feed_common = false)
    geom = build_geometry(uvset0)
    model = StationGainModel(
        phase = (
            clock = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            fringe = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), PerFeed()),
        ),
        logamp = (
            bandpass = TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
    )
    plan = plan_gains(model, 4, geom)

    # Injected truth (refant = 1 pinned to 0): delays large enough to wrap the
    # phase several times across the band, plus arbitrary constant phases. The
    # amplitude bandpass is left at 0 (the seed leaves it 0; LBFGS confirms it).
    rng = MersenneTwister(20260708)
    p_true = zero_params(plan)
    for a in 2:4, f in 1:2
        p_true.g1.clock[1, :, 1, f, a] .= (rand(rng) - 0.5) * 2.0e-7   # ±100 ns (wraps ~6×)
        p_true.g1.fringe[1, :, 1, f, a] .= (rand(rng) - 0.5) * 2π      # any constant phase
    end

    uvset = _forward_uvset(uvset0, plan, p_true, geom)   # visibilities = forward model at p_true

    obj_zero = fringe_objective(plan, zero_params(plan), uvset, geom; source = PointSource(1.0))

    @testset "warm-start alone lands in the right basin" begin
        p0 = fft_warmstart(plan, uvset, geom; ref_ant = 1)
        @test p0 isa typeof(zero_params(plan))
        # The reference antenna is pinned to 0 by the stationization gauge (up to
        # the constrained-QR numerical dust — far below the injected ±100 ns / ±π).
        @test all(x -> abs(x) < 1.0e-9, p0.g1.clock[:, :, :, :, 1])
        @test all(x -> abs(x) < 1.0e-9, p0.g1.fringe[:, :, :, :, 1])
        # Non-reference stations were seeded (not all left at 0).
        @test any(!iszero, p0.g1.clock[:, :, :, :, 2:4])
        obj_fft = fringe_objective(plan, p0, uvset, geom; source = PointSource(1.0))
        # The FFT seed's objective is enormously better than the sidelobe-locked
        # zero start — it resolved the wrapping delays.
        @test obj_fft > obj_zero + 10
    end

    @testset "fringe_solve(warmstart = :fft) reaches machine precision" begin
        sol = fringe_solve(
            uvset, model;
            optimizer = LBFGS(), source = PointSource(1.0),
            warmstart = :fft, maxiters = 2000,
        )
        @test sol.info.final_objective > -1.0e-8
        @test flatten(sol.p) ≈ flatten(p_true) atol = 1.0e-6
    end

    @testset ":auto is an alias for :fft" begin
        sol = fringe_solve(
            uvset, model;
            optimizer = LBFGS(), source = PointSource(1.0),
            warmstart = :auto, maxiters = 2000,
        )
        @test sol.info.final_objective > -1.0e-8
    end
end
