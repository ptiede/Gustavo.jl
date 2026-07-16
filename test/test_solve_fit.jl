# Gustavo.Solve — fringe_solve driver: optimize the forward-model MAP objective
# and recover injected gains (Stage 2, Milestone 5).
#
# A UVSet whose visibilities are EXACTLY the forward model at a known p_true (via
# predict_visibilities, noiseless) is fit from a zero warm-start. The objective is
# maximized (→ 0) at the recovered gains, apply_calibration flattens the fringe,
# and the profiled-source path (no assumed flux/amp scale) reaches the same.
# Reuses _build_fringe_uvset from test_pipeline.jl for a well-formed UVSet.

using Gustavo.Solve
using Gustavo.Solve: fringe_solve, FringeSolution, fringe_objective, plan_gains,
    zero_params, evaluate_gains, point_source_coherency, PointSource, ProfiledPointSource
using Gustavo: SerialExecutor
import Gustavo.UVData as UV
import Gustavo.Calibration as CAL
using Gustavo.Calibration:
    build_geometry, leaf_window, predict_visibilities, correlation_feed_pair,
    StationGainModel, GainComponent, TiedComponent, Delay, ConstantTerm, PerChannel,
    PerScan, GlobalTime, GlobalFrequency, PerSpectralWindow, PerFeed, SharedFeeds
using Enzyme, Optimization, OptimizationOptimJL     # enable Enzyme + LBFGS extensions
using Random: MersenneTwister
using Test

# Rebuild `uvset0` with visibilities set to the forward model at `p_true`.
function _forward_uvset(uvset0, plan, p_true, geom)
    return UV.apply(uvset0) do leaf, info, root
        leaf = UV.materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
        ci, ti = leaf_window(geom, leaf)
        g = evaluate_gains(plan, p_true, ci, ti)
        bl = UV.baselines(leaf)
        bl_a = Int[p[1] for p in bl.pairs]
        bl_b = Int[p[2] for p in bl.pairs]
        feeds = correlation_feed_pair.(UV.pol_products(leaf))
        coh = point_source_coherency(length(bl_a); flux = 1.0)
        V = predict_visibilities(g, coh, bl_a, bl_b, Int[f[1] for f in feeds], Int[f[2] for f in feeds])
        return UV.with_visibilities(leaf, V, fill(1.0, size(V)))
    end
end

@testset "Solve M5: fringe_solve recovers injected gains" begin
    uvset0, _ = _build_fringe_uvset(nant = 3, nbands = 2, nchan = 4, ntime = 4, feed_common = true)
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
    plan = plan_gains(model, 3, geom)

    # Injected truth: small (main-lobe) delays / phases / log-amps; refant = 0.
    rng = MersenneTwister(20260711)
    p_true = zero_params(plan)
    p_true .= 0.0
    for a in 2:3, f in 1:2
        p_true.g1.clock[1, :, 1, f, a] .= (rand(rng) - 0.5) * 1.0e-9   # ±0.5 ns
        p_true.g1.fringe[1, :, 1, f, a] .= (rand(rng) - 0.5) * 1.0     # ±0.5 rad
        p_true.g1.bandpass[:, 1, :, f, a] .= 0.1 .* randn(rng, size(p_true.g1.bandpass, 1), size(p_true.g1.bandpass, 3))
    end

    uvset = _forward_uvset(uvset0, plan, p_true, geom)

    obj0 = fringe_objective(plan, zero_params(plan), uvset, geom; source = PointSource(1.0))

    # The injected truth has refant (antenna 1) = 0, matching the default
    # ReferenceAntenna gauge, so an exact fit recovers p_true uniquely.
    @testset "fixed point source recovers the injected gains" begin
        # Any Optimization.jl optimizer works — pass it directly. Default
        # AutoScale + ReferenceAntenna(1) gauge make it well-conditioned.
        sol = fringe_solve(
            uvset, model;
            optimizer = LBFGS(), source = PointSource(1.0), maxiters = 2000,
            warmstart = zero_params(plan),
        )
        @test sol isa FringeSolution
        # Machine-precision fit (was ~-0.1 before scaling+gauge).
        @test sol.info.final_objective > -1.0e-10
        @test sol.info.final_objective > obj0 + 1.0
        # Recovered parameters match the injected truth (unique up to the pinned gauge).
        @test flatten(sol.p) ≈ flatten(p_true) atol = 1.0e-6
        # apply_calibration runs and returns a calibrated UVSet.
        @test UV.apply_calibration(uvset, sol) isa UV.UVSet
    end

    @testset "profiled source (no assumed flux/amp scale) also recovers it" begin
        sol = fringe_solve(
            uvset, model;
            optimizer = LBFGS(), source = ProfiledPointSource(), maxiters = 2000,
            warmstart = zero_params(plan),
        )
        @test sol.info.final_objective > -1.0e-10
        @test flatten(sol.p) ≈ flatten(p_true) atol = 1.0e-6
        # The reference antenna's phase + amplitude blocks are pinned to 0 (gauge).
        @test all(iszero, sol.p.g1.clock[:, :, :, :, 1])
        @test all(iszero, sol.p.g1.fringe[:, :, :, :, 1])
        @test all(iszero, sol.p.g1.bandpass[:, :, :, :, 1])
    end
end
