# Gustavo.Solve — per-leaf reverse-mode value + structured gradient
# (Stage 2, Milestone 3, GustavoEnzymeExt).
#
# `leaf_value_and_grad` Enzyme-differentiates the per-site coherency-matrix
# log-likelihood wrt the parameter ComponentVector. The gradient must (1) match
# ForwardDiff element-wise to 1e-6, (2) return the correct primal, and (3) be
# STRUCTURALLY SPARSE — exactly zero for time/frequency segments the leaf's
# window does not touch.

using Gustavo.Solve
using Gustavo.Solve: leaf_value_and_grad, leaf_loglik, flatten, unflatten, group_arrays
import Gustavo.Calibration as CAL
using Gustavo.Calibration:
    DataGeometry, StationGainModel, GainComponent, TiedComponent, FrequencyTerm,
    ConstantTerm, Delay, PerChannel, PerScan, GlobalTime, GlobalFrequency,
    PerSpectralWindow, PerFeed, SharedFeeds, correlation_feed_pair
using Enzyme                    # triggers GustavoEnzymeExt
import ForwardDiff
using Random: MersenneTwister
using Test

# ── A brand-new instrumental term, defined here in "user space" ──────────────
# Extensibility guarantee: a new gain model is a family choice + a name + ONE
# method — NO other Gustavo changes, NO Enzyme rule — and it flows through the
# forward map AND the reverse-mode gradient. `QuadraticDelay`: a curved/chromatic
# delay, phase = 2π·τ·(ν − ν0)². `FrequencyTerm` supplies coord_kind and the
# ν − ν0 coordinate (as `x`); `param_names` gives the count (1) and named access
# `p.τ` (exercising the named ParamBlock path through Enzyme).
struct QuadraticDelay <: FrequencyTerm end
CAL.param_names(::QuadraticDelay) = (:τ,)
CAL.term_eval(::QuadraticDelay, p, x) = 2π * p.τ * x^2

@testset "Solve M3: per-leaf Enzyme value + gradient" begin
    # Full geometry: 2 scans × 2 times, 2 spws × 2 channels.
    geom = DataGeometry(
        times = [0.0, 0.1, 1.0, 1.1],
        scan_of_time = [1, 1, 2, 2],
        channel_freqs = [1.0e9, 1.1e9, 2.0e9, 2.1e9],
        spw_of_chan = [1, 1, 2, 2],
    )
    nant = 3
    model = StationGainModel(
        phase = (
            TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), SharedFeeds()),
        ),
        logamp = (
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
    )
    plan = plan_gains(model, nant, geom)

    # Leaf window: scan 1 (times 1,2) × spw 1 (channels 1,2) only.
    bl_pairs = [(1, 2), (1, 3), (2, 3)]
    pols = ["PP", "PQ", "QP", "QQ"]
    feeds = correlation_feed_pair.(pols)
    rng = MersenneTwister(20260709)
    Vobs = randn(rng, ComplexF64, 2, 2, length(bl_pairs), length(pols))
    ctx = (;
        chan_idx = [1, 2], ti_idx = [1, 2],
        bl_a = Int[p[1] for p in bl_pairs], bl_b = Int[p[2] for p in bl_pairs],
        feed_a = Int[f[1] for f in feeds], feed_b = Int[f[2] for f in feeds],
        vis = Vobs, weights = fill(1.0, size(Vobs)),
    )
    S = point_source_coherency(length(bl_pairs); flux = 1.5)

    p = zero_params(plan)
    p .= 0.05 .* randn(rng, length(p))

    value, grad = leaf_value_and_grad(plan, p, ctx, S)

    @testset "primal + gradient vs ForwardDiff" begin
        @test value ≈ leaf_loglik(plan, p, ctx; S = S)
        g_enz = flatten(grad)
        g_fd = ForwardDiff.gradient(x -> leaf_loglik(plan, unflatten(plan, x), ctx; S = S), flatten(p))
        # Element-wise relative check (the delay gradient is ~1e10, the bandpass
        # ~10, so a single norm-based tolerance would mask the small components).
        @test all(@. abs(g_enz - g_fd) <= 1.0e-6 * abs(g_fd) + 1.0e-8)
        @test grad isa typeof(p)              # structured gradient shares p's layout
    end

    @testset "structural sparsity (untouched segments zero)" begin
        parr, aarr = group_arrays(plan, grad, 1)
        delayA = parr[1]     # Delay × PerScan: (1, ntseg=2, 1, nfb=2, nant=3)
        constA = parr[2]     # Const × PerScan (SharedFeeds): (1, 2, 1, 1, 3)
        bpA = aarr[1]        # PerChannel × PerSpw: (2, 1, nfseg=2, nfb=2, nant=3)
        # Scan 1 / spw 1 touched → nonzero; scan 2 / spw 2 untouched → exactly 0.
        @test maximum(abs, delayA[:, 1, :, :, :]) > 0
        @test all(iszero, delayA[:, 2, :, :, :])
        @test all(iszero, constA[:, 2, :, :, :])
        @test maximum(abs, bpA[:, :, 1, :, :]) > 0
        @test all(iszero, bpA[:, :, 2, :, :])
    end
end

@testset "Solve M3: custom term extensibility (forward + gradient)" begin
    # The user-defined QuadraticDelay above drives the full stack unchanged.
    geom = DataGeometry(
        times = [0.0, 0.1, 1.0, 1.1],
        scan_of_time = [1, 1, 2, 2],
        channel_freqs = [1.0e9, 1.1e9, 2.0e9, 2.1e9],
        spw_of_chan = [1, 1, 2, 2],
    )
    nant = 3
    model = StationGainModel(
        phase = (
            TiedComponent(GainComponent(QuadraticDelay(), PerScan(), GlobalFrequency()), PerFeed()),
        ),
        logamp = (
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
    )
    plan = plan_gains(model, nant, geom)

    bl_pairs = [(1, 2), (1, 3), (2, 3)]
    feeds = correlation_feed_pair.(["PP", "PQ", "QP", "QQ"])
    rng = MersenneTwister(11)
    Vobs = randn(rng, ComplexF64, 4, 4, length(bl_pairs), length(feeds))
    ctx = (;
        chan_idx = [1, 2, 3, 4], ti_idx = [1, 2, 3, 4],
        bl_a = Int[p[1] for p in bl_pairs], bl_b = Int[p[2] for p in bl_pairs],
        feed_a = Int[f[1] for f in feeds], feed_b = Int[f[2] for f in feeds],
        vis = Vobs, weights = fill(1.0, size(Vobs)),
    )
    S = point_source_coherency(length(bl_pairs); flux = 1.0)

    p = zero_params(plan)
    p .= 0.02 .* randn(rng, length(p))

    # Forward map + type stability are inherited by the new term with no changes.
    @test (@inferred evaluate_gains(plan, p)) isa Array{ComplexF64, 4}

    # And the Enzyme gradient flows through the custom term_eval — matching
    # ForwardDiff element-wise — with no per-term AD rule.
    value, grad = leaf_value_and_grad(plan, p, ctx, S)
    @test value ≈ leaf_loglik(plan, p, ctx; S = S)
    g_enz = flatten(grad)
    g_fd = ForwardDiff.gradient(x -> leaf_loglik(plan, unflatten(plan, x), ctx; S = S), flatten(p))
    @test all(@. abs(g_enz - g_fd) <= 1.0e-6 * abs(g_fd) + 1.0e-8)
end
