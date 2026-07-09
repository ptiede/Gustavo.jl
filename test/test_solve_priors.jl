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
    OUPrior, NoPrior, IIDGaussianPrior, BandpassARPrior, ComponentPriors,
    logprior, logprior_and_grad!, ou_logprior, ou_logprior_grad, flatten, unflatten
import Gustavo.UVData as UV
import Gustavo.Fringe as FR
using Gustavo.Calibration:
    build_geometry, leaf_window, predict_visibilities, correlation_feed_pair,
    DataGeometry, StationGainModel, GainComponent, TiedComponent, ConstantTerm, Delay, PerChannel,
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

@testset "Solve M7b: IID Gaussian + AR bandpass priors" begin
    # A per-channel (bandpass) log-amp block over a 12-channel single-segment axis.
    nchan = 12
    geom = DataGeometry(
        times = [0.0], channel_freqs = collect(range(2.3e11, 2.31e11, length = nchan)),
        scan_of_time = [1], spw_of_chan = ones(Int, nchan),
    )
    model = StationGainModel(
        phase = (fringe = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), PerFeed()),),
        logamp = (bandpass = TiedComponent(GainComponent(PerChannel(), GlobalTime(), GlobalFrequency()), PerFeed()),),
    )
    plan = plan_gains(model, 3, geom)
    rng = MersenneTwister(9)
    p = zero_params(plan)
    p.g1.bandpass .= 0.2 .* randn(rng, size(p.g1.bandpass))
    p.g1.fringe .= 0.3 .* randn(rng, size(p.g1.fringe))

    # 2nd-difference operator along the 12-channel axis (for the φ=[2,-1] check).
    D2 = zeros(nchan - 2, nchan)
    for i in 1:(nchan - 2)
        D2[i, i] = 1.0; D2[i, i + 1] = -2.0; D2[i, i + 2] = 1.0
    end

    @testset "IIDGaussianPrior: elementwise shrinkage" begin
        μ, σ = 0.05, 0.4
        cp = ComponentPriors(bandpass = IIDGaussianPrior(μ = μ, σ = σ))
        g = zero(p)
        lp = logprior_and_grad!(g, cp, plan, geom, p)
        # value == Σ over the block of the scalar Gaussian log-density
        blk = p.g1.bandpass
        expv = sum(b -> -0.5 * (b - μ)^2 / σ^2 - 0.5 * log(2π * σ^2), blk)
        @test lp ≈ expv atol = 1.0e-9
        # analytic gradient −(x−μ)/σ² on the block; zero elsewhere (fringe untouched)
        @test g.g1.bandpass ≈ .-(blk .- μ) ./ σ^2 atol = 1.0e-10
        @test all(iszero, g.g1.fringe)
        # only the named component is penalized
        @test logprior(ComponentPriors(fringe = IIDGaussianPrior(μ = 0.0, σ = 1.0)), plan, geom, p) !=
            logprior(cp, plan, geom, p)
    end

    # Analytic anchored-conditional AR(ord) log-density of one channel block.
    function ar_logprior_block(b, φ, σε, σ0)
        m = length(b); ord = length(φ)
        lp = 0.0
        for k in 1:min(ord, m)
            lp += -0.5 * b[k]^2 / σ0^2 - 0.5 * log(2π * σ0^2)
        end
        for k in (ord + 1):m
            r = b[k] - sum(φ[j] * b[k - j] for j in 1:ord)
            lp += -0.5 * r^2 / σε^2 - 0.5 * log(2π * σε^2)
        end
        return lp
    end

    @testset "BandpassARPrior: general AR(p) value + gradient" begin
        φ = [0.6, 0.2]                      # order-2 AR coefficients
        σε, σ0 = 0.3, 0.8
        cp = ComponentPriors(bandpass = BandpassARPrior(phi = φ, sigma_eps = σε, sigma0 = σ0))
        g = zero(p)
        lp = logprior_and_grad!(g, cp, plan, geom, p)
        # value == Σ over (feed, antenna) blocks of the analytic AR log-density
        expv = sum(fb -> sum(la -> ar_logprior_block(p.g1.bandpass[:, 1, 1, fb, la], φ, σε, σ0), 1:3), 1:2)
        @test lp ≈ expv atol = 1.0e-9
        @test all(iszero, g.g1.fringe)       # only the named component is penalized
        # gradient vs central finite differences on the flat vector
        x0 = copy(flatten(p)); h = 1.0e-6
        fd = map(eachindex(x0)) do i
            xp = copy(x0); xp[i] += h; xm = copy(x0); xm[i] -= h
            (logprior(cp, plan, geom, unflatten(plan, xp)) - logprior(cp, plan, geom, unflatten(plan, xm))) / (2h)
        end
        @test flatten(g) ≈ fd atol = 1.0e-6
    end

    @testset "phi = [2, -1] is the 2nd-difference (curvature) penalty" begin
        σε, σ0 = 0.5, 1.0e6                  # weak anchor: conditional part ≈ pure curvature
        cp = ComponentPriors(bandpass = BandpassARPrior(phi = [2.0, -1.0], sigma_eps = σε, sigma0 = σ0))
        lp = logprior(cp, plan, geom, p)
        # conditional residuals r_k = b_k − 2b_{k-1} + b_{k-2} == (D2·b); anchor negligible
        expv = 0.0
        for fb in 1:2, la in 1:3
            b = p.g1.bandpass[:, 1, 1, fb, la]
            expv += -0.5 / σε^2 * sum(abs2, D2 * b) - 0.5 * (nchan - 2) * log(2π * σε^2)
            expv += sum(k -> -0.5 * b[k]^2 / σ0^2 - 0.5 * log(2π * σ0^2), 1:2)
        end
        @test lp ≈ expv atol = 1.0e-6
    end

    @testset "proper: DC and slope are penalized (gauge-breaking)" begin
        cp = ComponentPriors(bandpass = BandpassARPrior(phi = [2.0, -1.0], sigma_eps = 0.5, sigma0 = 1.0))
        # A constant block and a linear-ramp block both cost logprior (the anchor
        # pins DC + slope), so the prior removes the delay/bandpass flat directions.
        pc = zero_params(plan); pc.g1.bandpass .= 0.7
        pr = zero_params(plan)
        for fb in 1:2, la in 1:3
            pr.g1.bandpass[:, 1, 1, fb, la] .= collect(1.0:nchan) .* 0.1
        end
        @test logprior(cp, plan, geom, pc) < 0          # constant is penalized (proper)
        @test logprior(cp, plan, geom, pr) < 0          # ramp is penalized (proper)
        # Adding a DC offset changes the log-density ⇒ the DC direction is not flat.
        pshift = copy(p); pshift.g1.bandpass .+= 0.5
        @test !isapprox(logprior(cp, plan, geom, p), logprior(cp, plan, geom, pshift); atol = 1.0e-6)
    end
end
