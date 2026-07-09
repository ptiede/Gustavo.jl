# Gustavo.Solve — prior / regularizer layer, OU-first (Stage 2, Milestone 7).
#
# The MAP objective is Σ_leaf loglik + logprior. This tests the OU (Matérn-1/2)
# stochastic-time prior on a station's per-integration adhoc-phase track:
#   (1) ou_logprior / grad match a dense OU MvNormal and finite differences, and
#       the (likelihood + OU) MAP equals the RTS smoother mean (statespace.jl);
#   (2) ComponentPriors sums the per-track prior over a plan, with the correct
#       structured gradient, and the empty default is a no-op;
#   (3) FringePosterior adds the prior term + gradient to the distributed
#       likelihood; and
#   (4) end-to-end, MAP+OU recovers a smooth low-SNR adhoc track better than the
#       unregularized fit.
#
# Reuses _build_fringe_uvset (test_pipeline.jl); included after it in runtests.jl.

using Gustavo.Solve
using Gustavo.Solve: fringe_solve, plan_gains, zero_params, evaluate_gains,
    point_source_coherency, PointSource, FringePosterior,
    OUPrior, NoPrior, ComponentPriors, logprior, logprior_and_grad!,
    ou_logprior, ou_logprior_grad, flatten, unflatten
import Gustavo.UVData as UV
import Gustavo.Fringe as FR
using Gustavo.Calibration:
    build_geometry, leaf_window, predict_visibilities, correlation_feed_pair,
    StationGainModel, GainComponent, TiedComponent, ConstantTerm, Delay, PerChannel,
    PerScan, PerIntegration, GlobalTime, GlobalFrequency, PerSpectralWindow,
    PerFeed, SharedFeeds
import LogDensityProblems as LDP
using Enzyme, Optimization, OptimizationOptimJL
using LinearAlgebra: Diagonal, dot, logdet
using Random: MersenneTwister, randn
using Test

# Dense OU (Matérn-1/2) Gaussian log-density — the ground truth for ou_logprior.
function _dense_ou_logpdf(x, t; τ, σ2)
    n = length(x)
    S = [σ2 * exp(-abs(t[i] - t[j]) / τ) for i in 1:n, j in 1:n]
    return -0.5 * (dot(x, S \ x) + logdet(2π * S))
end

@testset "Solve M7: OU prior (stochastic-time regularizer)" begin

    @testset "ou_logprior math vs dense OU + RTS smoother" begin
        rng = MersenneTwister(7)
        t = cumsum(rand(rng, 8)) .* 3.0          # irregular sample times
        x = randn(rng, 8)
        τ, σ2 = 4.0, 0.6
        @test ou_logprior(x, t; τ = τ, σ2 = σ2) ≈ _dense_ou_logpdf(x, t; τ = τ, σ2 = σ2) atol = 1.0e-9
        # gradient vs central finite differences
        g = ou_logprior_grad(x, t; τ = τ, σ2 = σ2)
        h = 1.0e-6
        fd = map(eachindex(x)) do i
            xp = copy(x); xp[i] += h
            xm = copy(x); xm[i] -= h
            (ou_logprior(xp, t; τ = τ, σ2 = σ2) - ou_logprior(xm, t; τ = τ, σ2 = σ2)) / (2h)
        end
        @test g ≈ fd atol = 1.0e-6
        # (Gaussian likelihood + OU prior) MAP == RTS smoother posterior mean.
        w = fill(2.0, 8)
        y = randn(rng, 8)
        S = [σ2 * exp(-abs(t[i] - t[j]) / τ) for i in 1:8, j in 1:8]
        xhat_map = (Diagonal(w) + inv(S)) \ (w .* y)
        @test xhat_map ≈ FR.smooth_ou_track(y, w, t; τ = τ, σ2 = σ2) atol = 1.0e-8
    end

    # A per-integration adhoc-phase model over a single-scan UVSet.
    uvset0, _ = _build_fringe_uvset(nant = 4, nbands = 1, nchan = 6, ntime = 10, feed_common = true)
    geom = build_geometry(uvset0)
    adhoc_model = StationGainModel(
        phase = (
            adhoc = TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), SharedFeeds()),
        ),
        logamp = (),
    )
    plan = plan_gains(adhoc_model, 4, geom)
    tsec = geom.times .* 3600.0                  # AP times in seconds
    τ, σ2 = 200.0, 0.5
    cp = ComponentPriors(adhoc = OUPrior(τ = τ, σ2 = σ2))

    @testset "ComponentPriors sums per-track OU over the plan" begin
        rng = MersenneTwister(11)
        p = zero_params(plan)
        p.g1.adhoc .= 0.3 .* randn(rng, size(p.g1.adhoc))
        # logprior == Σ over (antenna) of the per-track OU density (SharedFeeds ⇒
        # one feed block; GlobalFrequency ⇒ one freq segment).
        expected = 0.0
        for la in 1:4
            expected += ou_logprior(p.g1.adhoc[1, :, 1, 1, la], tsec; τ = τ, σ2 = σ2)
        end
        @test logprior(cp, plan, geom, p) ≈ expected atol = 1.0e-9
        # structured gradient vs finite differences on the flat vector
        g = zero(p)
        logprior_and_grad!(g, cp, plan, geom, p)
        x0 = copy(flatten(p))
        h = 1.0e-6
        fd = map(eachindex(x0)) do i
            xp = copy(x0); xp[i] += h
            xm = copy(x0); xm[i] -= h
            (
                logprior(cp, plan, geom, unflatten(plan, xp)) -
                    logprior(cp, plan, geom, unflatten(plan, xm))
            ) / (2h)
        end
        @test flatten(g) ≈ fd atol = 1.0e-6
        # empty ComponentPriors is a no-op
        @test logprior(ComponentPriors(), plan, geom, p) == 0.0
        ge = zero(p)
        @test logprior_and_grad!(ge, ComponentPriors(), plan, geom, p) == 0.0
        @test all(iszero, flatten(ge))
    end

    @testset "FringePosterior adds the prior term + gradient" begin
        # Noiseless forward data so the likelihood is well-defined; we only check
        # that the prior is added consistently to value and gradient.
        rng = MersenneTwister(3)
        p_true = zero_params(plan)
        for la in 2:4
            p_true.g1.adhoc[1, :, 1, 1, la] .= 0.2 .* randn(rng, size(p_true.g1.adhoc, 2))
        end
        uvset = UV.apply(uvset0) do leaf, info, root
            leaf = UV.materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
            ci, ti = leaf_window(geom, leaf)
            g = evaluate_gains(plan, p_true, ci, ti)
            bl = UV.baselines(leaf)
            ba = Int[q[1] for q in bl.pairs]; bb = Int[q[2] for q in bl.pairs]
            fe = correlation_feed_pair.(UV.pol_products(leaf))
            V = predict_visibilities(
                g, point_source_coherency(length(ba); flux = 1.0),
                ba, bb, Int[f[1] for f in fe], Int[f[2] for f in fe]
            )
            return UV.with_visibilities(leaf, V, fill(1.0, size(V)))
        end
        post0 = FringePosterior(plan, uvset, geom; source = PointSource(1.0))
        postp = FringePosterior(plan, uvset, geom; source = PointSource(1.0), prior = cp)
        x = collect(flatten(p_true)) .+ 0.05
        p = unflatten(plan, x)
        @test LDP.logdensity(postp, x) ≈ LDP.logdensity(post0, x) + logprior(cp, plan, geom, p) atol = 1.0e-8
        v0, g0 = LDP.logdensity_and_gradient(post0, x)
        vp, gp = LDP.logdensity_and_gradient(postp, x)
        gpri = zero(p)
        lp = logprior_and_grad!(gpri, cp, plan, geom, p)
        @test vp ≈ v0 + lp atol = 1.0e-8
        @test gp ≈ g0 .+ flatten(gpri) atol = 1.0e-8
    end

    @testset "MAP+OU recovers a smooth low-SNR track" begin
        # Smooth injected truth (one slow cycle over the scan), refant = 0.
        ntime = length(geom.times)
        truth = zeros(4, ntime)
        for la in 2:4
            ph = (la - 2) * 0.7
            truth[la, :] .= 0.6 .* sin.(2π .* (0:(ntime - 1)) ./ (ntime - 1) .* 0.5 .+ ph)
        end
        p_true = zero_params(plan)
        for la in 1:4
            p_true.g1.adhoc[1, :, 1, 1, la] .= truth[la, :]
        end
        # Noisy forward data (single leaf, so the closure is called once — rng safe).
        rng = MersenneTwister(2026)
        σn = 0.6
        uvset = UV.apply(uvset0) do leaf, info, root
            leaf = UV.materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
            ci, ti = leaf_window(geom, leaf)
            g = evaluate_gains(plan, p_true, ci, ti)
            bl = UV.baselines(leaf)
            ba = Int[q[1] for q in bl.pairs]; bb = Int[q[2] for q in bl.pairs]
            fe = correlation_feed_pair.(UV.pol_products(leaf))
            V = predict_visibilities(
                g, point_source_coherency(length(ba); flux = 1.0),
                ba, bb, Int[f[1] for f in fe], Int[f[2] for f in fe]
            )
            V = V .+ σn .* (randn(rng, ComplexF64, size(V))) ./ sqrt(2)
            return UV.with_visibilities(leaf, V, fill(1.0 / σn^2, size(V)))
        end
        rmse(sol) = sqrt(sum(la -> sum(abs2, sol.p.g1.adhoc[1, :, 1, 1, la] .- truth[la, :]), 2:4) / (3 * ntime))
        sol0 = fringe_solve(
            uvset, adhoc_model; optimizer = LBFGS(), source = PointSource(1.0),
            warmstart = zero_params(plan), maxiters = 500
        )
        solp = fringe_solve(
            uvset, adhoc_model; optimizer = LBFGS(), source = PointSource(1.0),
            prior = cp, warmstart = zero_params(plan), maxiters = 500
        )
        # The OU prior denoises: the regularized track is closer to the smooth truth.
        @test rmse(solp) < rmse(sol0)
    end
end
