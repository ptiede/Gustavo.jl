# A gauge fixes only what the data cannot: every baseline's gain product
# g_a[fa]·conj(g_b[fb]) must not depend on which gauge a step uses, nor, beyond
# rounding, on the visibilities' precision.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# The largest wrapped phase and log-amplitude differences between the baseline
# gain products of two solutions, over samples where both are finite.
function _baseline_product_difference(sol1, sol2)
    g1, g2 = parent(gains(sol1)), parent(gains(sol2))
    axes(g1) == axes(g2) || throw(DimensionMismatch("gains $(axes(g1)) vs $(axes(g2))"))
    dphase, dlogamp = 0.0, 0.0
    ants = axes(g1, 3)
    for a in ants, b in ants, fa in axes(g1, 4), fb in axes(g1, 4), t in axes(g1, 2), c in axes(g1, 1)
        a < b || continue
        p1 = g1[c, t, a, fa] * conj(g1[c, t, b, fb])
        p2 = g2[c, t, a, fa] * conj(g2[c, t, b, fb])
        (isfinite(p1) && isfinite(p2)) || continue
        dphase = max(dphase, abs(angle(p1 * conj(p2))))
        dlogamp = max(dlogamp, abs(log(abs(p1)) - log(abs(p2))))
    end
    return dphase, dlogamp
end

# The fringe fit one scan at a time and pooled, then `Bandpass` and
# `AdhocPhase` (with each smoother) on data the first corrected.
function _fit_each_step(ps, gauge)
    fringe = fit(BaselineFringeFit(; gauge), ps)
    corrected = calibrate(fringe, ps; flag_bad = false, apply_flags = false)
    return (;
        fringe, pooled = fit(BaselineFringeFit(; rounds = 2, gauge), ps),
        bandpass = fit(Bandpass(; gauge), corrected), adhoc = fit(AdhocPhase(; gauge), corrected),
        adhoc_joint = fit(AdhocPhase(Gustavo.Fring.JointKalmanSmoother(); gauge), corrected),
    )
end

@testset "gauge and precision invariance" begin
    ps64, _ = _build_fringe_ps(; eltype = ComplexF64, nscans = 4, noise = 0.5)
    zs = _fit_each_step(ps64, ZeroSumPhase())

    @testset "baseline products do not depend on the gauge" begin
        pinned = _fit_each_step(ps64, PinAntenna([3, 1]))
        # The bandpass gauge enters only through a phase prior, which the default model lacks.
        for step in keys(zs)
            dphase, dlogamp = _baseline_product_difference(zs[step], pinned[step])
            @test dphase < 1.0e-6
            @test dlogamp < 1.0e-4
        end
    end

    @testset "Float32 visibilities agree with Float64" begin
        ps32, _ = _build_fringe_ps(; eltype = ComplexF32, nscans = 4, noise = 0.5)
        zs32 = _fit_each_step(ps32, ZeroSumPhase())
        for step in keys(zs)
            dphase, dlogamp = _baseline_product_difference(zs[step], zs32[step])
            @test dphase < 1.0e-3
            @test dlogamp < 1.0e-5
        end
    end
end
