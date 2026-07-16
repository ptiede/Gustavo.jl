# Gustavo.Solve — per-site model overrides (Stage 2, Milestone 6).
#
# An `ArrayGainModel` fits most of the array under one default `StationGainModel`
# but lets individual antennas carry a DIFFERENT structure (an extra term the rest
# of the array lacks). `plan_gains` groups antennas by identical model structure,
# so the override lives in its own group with its own parameter block; the solve,
# gauge, scaling, warm-start and gradient must all handle the multi-group plan.
# This test injects a noiseless forward model where antenna 2 additionally carries
# a fringe RATE the others do not (a stand-in for the plan's motivating dTEC case),
# and checks the override recovers its extra parameter while the homogeneous subset
# is unaffected.
#
# Reuses `_build_fringe_uvset` (test_pipeline.jl) and `_forward_uvset`
# (test_solve_fit.jl), so it is included after both in runtests.jl.

using Gustavo.Solve
using Gustavo.Solve: fringe_solve, plan_gains, zero_params, evaluate_gains,
    point_source_coherency, PointSource, ArrayGainModel, site_model
import Gustavo.UVData as UV
using Gustavo.Calibration:
    build_geometry, leaf_window, predict_visibilities, correlation_feed_pair,
    StationGainModel, GainComponent, TiedComponent, Delay, Rate, ConstantTerm,
    PerChannel, PerScan, GlobalTime, GlobalFrequency, PerSpectralWindow, PerFeed
using Enzyme, Optimization, OptimizationOptimJL     # enable Enzyme + LBFGS extensions
using Random: MersenneTwister
using Test

@testset "Solve M6: per-site model override solves independently" begin
    uvset0, _ = _build_fringe_uvset(nant = 4, nbands = 2, nchan = 4, ntime = 6, feed_common = true)
    geom = build_geometry(uvset0)

    default = StationGainModel(
        phase = (
            clock = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            fringe = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), PerFeed()),
        ),
        logamp = (
            bandpass = TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
    )
    # Antenna 2's override: the default plus an extra per-scan fringe RATE.
    override = StationGainModel(
        phase = (
            clock = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            fringe = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), PerFeed()),
            drift = TiedComponent(GainComponent(Rate(), PerScan(), GlobalFrequency()), PerFeed()),
        ),
        logamp = (
            bandpass = TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
    )
    am = ArrayGainModel(default, Dict(2 => override))
    @test site_model(am, 2) === override
    @test site_model(am, 1) === default

    plan = plan_gains(am, 4, geom)
    # Two groups: default (ants 1,3,4 — refant first, so it is g1) and the
    # override (ant 2, g2). The grouping is deterministic in antenna order.
    @test length(plan.groups) == 2
    @test plan.groups[1].ants == [1, 3, 4]
    @test plan.groups[2].ants == [2]
    # Only the override group carries the drift (Rate) block.
    @test :drift in propertynames(zero_params(plan).g2)
    @test !(:drift in propertynames(zero_params(plan).g1))

    # Injected truth (refant = 1 pinned to 0). Small main-lobe values so a zero
    # warm-start recovers; the override's rate is small enough to stay < π over
    # the 150 s scan (2π·1e-3·150 ≈ 0.9 rad).
    rng = MersenneTwister(20260708)
    p_true = zero_params(plan)
    for (la, a) in enumerate(plan.groups[1].ants), f in 1:2   # default group (1,3,4)
        a == 1 && continue                                    # refant = 0
        p_true.g1.clock[1, :, 1, f, la] .= (rand(rng) - 0.5) * 1.0e-9
        p_true.g1.fringe[1, :, 1, f, la] .= (rand(rng) - 0.5) * 1.0
        p_true.g1.bandpass[:, 1, :, f, la] .= 0.1 .* randn(rng, size(p_true.g1.bandpass, 1), size(p_true.g1.bandpass, 3))
    end
    for f in 1:2                                              # override group (ant 2, la = 1)
        p_true.g2.clock[1, :, 1, f, 1] .= (rand(rng) - 0.5) * 1.0e-9
        p_true.g2.fringe[1, :, 1, f, 1] .= (rand(rng) - 0.5) * 1.0
        p_true.g2.drift[1, :, 1, f, 1] .= (rand(rng) - 0.5) * 2.0e-3    # ±1 mHz rate
        p_true.g2.bandpass[:, 1, :, f, 1] .= 0.1 .* randn(rng, size(p_true.g2.bandpass, 1), size(p_true.g2.bandpass, 3))
    end
    @test any(!iszero, p_true.g2.drift)                       # the override truly has a rate

    uvset = _forward_uvset(uvset0, plan, p_true, geom)         # visibilities = forward model at p_true

    @testset "heterogeneous solve recovers every group" begin
        sol = fringe_solve(
            uvset, am;
            optimizer = LBFGS(), source = PointSource(1.0), maxiters = 3000,
            warmstart = zero_params(plan),
        )
        @test sol.info.final_objective > -1.0e-9
        @test flatten(sol.p) ≈ flatten(p_true) atol = 1.0e-6
        # The override's extra rate is recovered (solved independently)…
        @test sol.p.g2.drift ≈ p_true.g2.drift atol = 1.0e-6
        # …and the homogeneous subset (default group) is unaffected.
        @test sol.p.g1.clock ≈ p_true.g1.clock atol = 1.0e-6
        @test sol.p.g1.fringe ≈ p_true.g1.fringe atol = 1.0e-6
    end

    @testset "FFT warm-start seeds the override group too" begin
        sol = fringe_solve(
            uvset, am;
            optimizer = LBFGS(), source = PointSource(1.0), maxiters = 3000,
            warmstart = :fft,
        )
        @test sol.info.final_objective > -1.0e-9
        @test flatten(sol.p) ≈ flatten(p_true) atol = 1.0e-6
    end
end
