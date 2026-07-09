# Gustavo.Solve — pluggable solve strategies + direct linear phase block steps.
#
# `fringe_solve` runs either pure gradient descent (`GradientDescent`, the default)
# or a block-coordinate scheme (`BlockCoordinate`) that alternates a gradient step
# (with named components frozen) with direct LINEAR block solves — the per-channel
# phase bandpass (`FreqStep`) and the per-AP adhoc (`TimeStep`), both the same
# closure solve on orthogonal axes (`LinearPhaseStep`). Tests: the default path is
# unchanged, `frozen` validates against real parameter-block names, and the hybrid
# reaches a correct calibration on noiseless data.
#
# Reuses `_build_fringe_uvset` (test_pipeline.jl) and `_forward_uvset`
# (test_solve_fit.jl), so it is included after both in runtests.jl.

using Gustavo.Solve
using Gustavo.Solve: fringe_solve, plan_gains, zero_params, evaluate_gains,
    point_source_coherency, PointSource,
    GradientDescent, BlockCoordinate, GradientStep, FreqStep, TimeStep, LinearPhaseStep,
    refine_phase_component!
import Gustavo.UVData as UV
using Gustavo.Calibration:
    build_geometry, leaf_window, predict_visibilities, correlation_feed_pair,
    StationGainModel, GainComponent, TiedComponent, Delay, ConstantTerm, PerChannel,
    PerScan, GlobalTime, GlobalFrequency, PerSpectralWindow, PerFeed
using Enzyme, Optimization, OptimizationOptimJL
using Random: MersenneTwister
using Test

@testset "Solve: pluggable strategies + linear block steps" begin
    uvset0, _ = _build_fringe_uvset(nant = 4, nbands = 2, nchan = 5, ntime = 5, feed_common = true)
    geom = build_geometry(uvset0)
    model = StationGainModel(
        phase = (
            clock = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            fringe = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), PerFeed()),
            bandpass = TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
        logamp = (),
    )
    plan = plan_gains(model, 4, geom)

    # Injected truth (refant = 1 = 0): small delays/phases + a per-channel phase
    # bandpass (the block the hybrid solves linearly).
    rng = MersenneTwister(20260709)
    p_true = zero_params(plan)
    for a in 2:4, f in 1:2
        p_true.g1.clock[1, :, 1, f, a] .= (rand(rng) - 0.5) * 1.0e-9
        p_true.g1.fringe[1, :, 1, f, a] .= (rand(rng) - 0.5) * 1.0
        p_true.g1.bandpass[:, 1, :, f, a] .= 0.3 .* randn(rng, size(p_true.g1.bandpass, 1), size(p_true.g1.bandpass, 3))
    end
    uvset = _forward_uvset(uvset0, plan, p_true, geom)

    @testset "GradientDescent strategy ≡ the default optimizer path" begin
        z = zero_params(plan)
        sol_a = fringe_solve(uvset, model; optimizer = LBFGS(), source = PointSource(1.0), warmstart = z, maxiters = 800)
        sol_b = fringe_solve(uvset, model; strategy = GradientDescent(LBFGS(); maxiters = 800), source = PointSource(1.0), warmstart = z)
        @test flatten(sol_a.p) ≈ flatten(sol_b.p) atol = 1.0e-8
        @test sol_a.info.final_objective ≈ sol_b.info.final_objective atol = 1.0e-8
    end

    @testset "block-coordinate hybrid reaches a correct calibration" begin
        strat = BlockCoordinate(
            GradientStep(LBFGS(); frozen = (:bandpass,), maxiters = 800),
            FreqStep(:bandpass);
            rounds = 5,
        )
        sol = fringe_solve(uvset, model; strategy = strat, source = PointSource(1.0), warmstart = zero_params(plan))
        # Noiseless data → the total gain is recovered (objective → 0), even though
        # the delay/bandpass split is a gauge. The linear bandpass step + LBFGS on
        # the nonlinear params fit it far below the raw objective.
        obj0 = Gustavo.Solve.fringe_objective(plan, zero_params(plan), uvset, geom; source = PointSource(1.0))
        @test sol.info.final_objective > -1.0e-4
        @test sol.info.final_objective > obj0 + 1.0
        @test sol.info.rounds == 5
        @test UV.apply_calibration(uvset, sol) isa UV.UVSet
    end

    @testset "frozen validates against actual parameter-block names" begin
        strat = BlockCoordinate(GradientStep(LBFGS(); frozen = (:nope,)), FreqStep(:bandpass))
        @test_throws ErrorException fringe_solve(
            uvset, model; strategy = strat, source = PointSource(1.0), warmstart = zero_params(plan),
        )
    end

    @testset "convenience steps build the right LinearPhaseStep" begin
        @test FreqStep(:bandpass).axis === :freq
        @test TimeStep(:adhoc).axis === :time
        @test TimeStep(:adhoc).shared_feeds
        @test !FreqStep(:bandpass).shared_feeds
        @test LinearPhaseStep(:x; axis = :time).component === :x
    end
end
