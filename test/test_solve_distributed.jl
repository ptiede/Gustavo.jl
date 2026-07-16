# Gustavo.Solve — distributed objective + gradient over the Graph substrate
# (Stage 2, Milestone 4).
#
# The total forward-model log-likelihood and its gradient are summed over the
# UVSet's partitions via `pmapreduce`. The milestone invariants: the result is
# executor-independent (SerialExecutor ≡ DaggerExecutor, bit-identical), the
# summed value/gradient equal a monolithic reference (ForwardDiff of the total),
# and the `FringePosterior` LogDensityProblems interface returns the same.
# Reuses `_build_fringe_uvset` from test_pipeline.jl.

using Gustavo.Solve
using Gustavo.Solve: fringe_objective, fringe_objective_and_grad, FringePosterior,
    plan_gains, zero_params, flatten, unflatten
using Gustavo: SerialExecutor, DaggerExecutor
import Gustavo.Calibration as CAL
using Gustavo.Calibration:
    build_geometry, StationGainModel, GainComponent, TiedComponent,
    Delay, ConstantTerm, PerChannel, PerScan, GlobalTime, GlobalFrequency,
    PerSpectralWindow, PerFeed, SharedFeeds
using Enzyme                    # enables leaf_value_and_grad
import ForwardDiff
import LogDensityProblems as LDP
using Random: MersenneTwister
using Test

@testset "Solve M4: distributed objective + gradient" begin
    uvset, _ = _build_fringe_uvset(nant = 4, nbands = 2, nchan = 6, ntime = 6)
    geom = build_geometry(uvset)

    model = StationGainModel(
        phase = (
            clock = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            fringe = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), SharedFeeds()),
        ),
        logamp = (
            bandpass = TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
    )
    plan = plan_gains(model, 4, geom)

    rng = MersenneTwister(20260710)
    p = zero_params(plan)
    p .= 0.03 .* randn(rng, length(p))

    vs, gs = fringe_objective_and_grad(plan, p, uvset, geom; executor = SerialExecutor())
    vd, gd = fringe_objective_and_grad(plan, p, uvset, geom; executor = DaggerExecutor())

    @testset "SerialExecutor ≡ DaggerExecutor (bit-identical)" begin
        @test vs == vd
        @test flatten(gs) == flatten(gd)
        @test gs isa typeof(p)                       # structured gradient
        # Force the per-leaf granularity path (target > #groups): the scan's band
        # leaves fan out to one compute task each — still bit-identical to Serial.
        vl, gl = fringe_objective_and_grad(plan, p, uvset, geom; executor = DaggerExecutor(target = 8))
        @test vs == vl
        @test flatten(gs) == flatten(gl)
    end

    @testset "value-only path + sum == monolithic" begin
        # Value-only path (no Enzyme) agrees with the value from the grad path.
        @test fringe_objective(plan, p, uvset, geom; executor = SerialExecutor()) ≈ vs
        @test fringe_objective(plan, p, uvset, geom; executor = DaggerExecutor()) ≈ vs
        # The total gradient equals ForwardDiff of the total objective (monolithic).
        g_fd = ForwardDiff.gradient(
            x -> fringe_objective(plan, unflatten(plan, x), uvset, geom), flatten(p),
        )
        g_enz = flatten(gs)
        @test all(@. abs(g_enz - g_fd) <= 1.0e-6 * abs(g_fd) + 1.0e-8)
    end

    @testset "FringePosterior LogDensityProblems (order 1)" begin
        post = FringePosterior(plan, uvset, geom)
        @test LDP.dimension(post) == length(p)
        @test LDP.capabilities(typeof(post)) == LDP.LogDensityOrder{1}()
        @test LDP.logdensity(post, flatten(p)) ≈ vs
        v, g = LDP.logdensity_and_gradient(post, flatten(p))
        @test v ≈ vs
        @test g ≈ flatten(gs)
        # Dagger-backed posterior agrees.
        postd = FringePosterior(plan, uvset, geom; executor = DaggerExecutor())
        vd2, gd2 = LDP.logdensity_and_gradient(postd, flatten(p))
        @test vd2 == v && gd2 == g
    end
end
