# Gustavo.Solve — per-leaf coherency-matrix WLS objective (Stage 2, Milestone 1).
#
# Tests the core `leaf_loglik_gains` directly on a hand-built context (no UVSet /
# θ machinery needed — that's exercised in later milestones): the log-likelihood
# is maximized (≈ 0) at the true gains, drops when perturbed, is additive over
# disjoint product sets, and gracefully handles missing polarization products
# (a parallel-only leaf contributes only its PP/QQ terms).

using Gustavo.Solve: leaf_loglik_gains, point_source_coherency
import Gustavo.Calibration as CAL
using Random: MersenneTwister

# Build a ctx NamedTuple (the AD-inactive per-leaf context) directly from arrays.
_ctx(bl_pairs, feeds, Vobs, W) = (;
    bl_a = Int[p[1] for p in bl_pairs], bl_b = Int[p[2] for p in bl_pairs],
    feed_a = Int[f[1] for f in feeds], feed_b = Int[f[2] for f in feeds],
    vis = Vobs, weights = W,
)

@testset "Solve M1: leaf coherency WLS objective" begin
    rng = MersenneTwister(20260708)
    nchan, nti, nant = 3, 2, 4
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    nbl = length(bl_pairs)
    pols = ["PP", "PQ", "QP", "QQ"]
    feeds = CAL.correlation_feed_pair.(pols)
    flux = 2.5
    S = point_source_coherency(nbl; flux = flux)

    # Random true antenna gains (nchan, nti, nant, 2).
    g = Array{ComplexF64}(undef, nchan, nti, nant, 2)
    for i in eachindex(g)
        g[i] = exp(0.15 * randn(rng)) * cis(2π * rand(rng))
    end

    # Observed vis = g_a · S · conj(g_b) for every product (so truth is exact).
    Vobs = zeros(ComplexF64, nchan, nti, nbl, length(pols))
    for (p, (fa, fb)) in enumerate(feeds), (bi, (a, b)) in enumerate(bl_pairs)
        for t in 1:nti, c in 1:nchan
            Vobs[c, t, bi, p] = g[c, t, a, fa] * S[bi, fa, fb] * conj(g[c, t, b, fb])
        end
    end
    W = fill(1.0, size(Vobs))
    ctx = _ctx(bl_pairs, feeds, Vobs, W)

    # (1) At the true gains the WLS log-likelihood is ~0 (its maximum).
    @test leaf_loglik_gains(g, ctx, S) ≈ 0 atol = 1e-9

    # (2) Perturbing any gain strictly decreases it.
    gp = copy(g)
    gp[1, 1, 1, 1] *= cis(0.3)
    gp[2, 1, 3, 2] *= 1.1
    @test leaf_loglik_gains(gp, ctx, S) < -1e-6

    # (3) Additivity over disjoint product sets + graceful missing pol:
    #     full objective == parallel-only (PP,QQ) + cross-only (PQ,QP).
    par = [1, 4]            # PP, QQ
    cross = [2, 3]          # PQ, QP
    ctx_par = _ctx(bl_pairs, feeds[par], Vobs[:, :, :, par], W[:, :, :, par])
    ctx_cross = _ctx(bl_pairs, feeds[cross], Vobs[:, :, :, cross], W[:, :, :, cross])
    @test leaf_loglik_gains(gp, ctx, S) ≈
        leaf_loglik_gains(gp, ctx_par, S) + leaf_loglik_gains(gp, ctx_cross, S)

    # (4) A parallel-only (PP/QQ) leaf is well-defined and ~0 at truth — it simply
    #     never touches the absent cross-hand products (no 4-slot padding).
    @test leaf_loglik_gains(g, ctx_par, S) ≈ 0 atol = 1e-9

    # (5) The parallel-only perturbed objective responds ONLY to feed-touching
    #     products: perturbing a Q-feed gain changes QQ (present) but the PP-only
    #     objective is unaffected.
    only_pp = [1]
    ctx_pp = _ctx(bl_pairs, feeds[only_pp], Vobs[:, :, :, only_pp], W[:, :, :, only_pp])
    gq = copy(g)
    gq[1, 1, 2, 2] *= cis(0.5)      # perturb a feed-2 (Q) gain
    @test leaf_loglik_gains(gq, ctx_pp, S) ≈ 0 atol = 1e-9        # PP untouched by Q
    @test leaf_loglik_gains(gq, ctx_par, S) < -1e-6              # QQ sees it

    # (6) Element type follows the gains (the AD-relevant property: the tracked
    #     gains are the widest type). A fully-Float32 leaf returns Float32.
    g32 = ComplexF32.(g)
    S32 = point_source_coherency(nbl; flux = flux, T = ComplexF32)
    ctx32 = _ctx(bl_pairs, feeds, ComplexF32.(Vobs), Float32.(W))
    @test leaf_loglik_gains(g32, ctx32, S32) isa Float32
end
