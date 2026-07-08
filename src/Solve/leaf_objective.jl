# ── Per-leaf coherency-matrix WLS objective (Stage 2, Milestone 1) ────────────
#
# The forward-model fringe fit's objective is a sum over `UVSet` partitions of a
# per-leaf Gaussian log-likelihood: for each PRESENT correlation product, predict
# the total visibility `V̂ = g_a · S · conj(g_b)` (diagonal-Jones) from the
# per-station gains `g` and the 2×2 source coherency `S`, and accumulate
# `-½ w |Vobs − V̂|²`. Missing polarization products are simply absent from the
# leaf's `Pol` axis, so the accumulation only ever touches present products — no
# fixed 4-slot padding.
#
# This milestone uses the EXISTING flat-θ `GainEvaluator`/`evaluate_gains`; the
# per-site 2D-array parameter container replaces θ in Milestone 2. The core
# `leaf_loglik_gains` is an allocation-free fused loop over
# (product, baseline, time, channel), so it is ready for Enzyme reverse-mode in
# Milestone 3 — every branch is on DATA (weight, finiteness), never on the
# differentiated gains.

"""
    build_leaf_ctx(geom, leaf) -> NamedTuple

Precompute the per-leaf, AD-INACTIVE context the objective needs: the geometry
window (`chan_idx`, `ti_idx`), the split baseline antenna indices (`bl_a`,
`bl_b`), the present products' feed pairs (`feed_a`, `feed_b` from
`correlation_feed_pair`), and the observed `vis`/`weights` arrays. Built once per
leaf and captured `Const` by the differentiated closure.
"""
function build_leaf_ctx(geom::DataGeometry, leaf)
    ci, ti = leaf_window(geom, leaf)
    bl = UVData.baselines(leaf).pairs
    bl_a = Int[Int(p[1]) for p in bl]
    bl_b = Int[Int(p[2]) for p in bl]
    pols = UVData.pol_products(leaf)
    feeds = correlation_feed_pair.(pols)
    feed_a = Int[f[1] for f in feeds]
    feed_b = Int[f[2] for f in feeds]
    return (;
        chan_idx = ci, ti_idx = ti,
        bl_a = bl_a, bl_b = bl_b, feed_a = feed_a, feed_b = feed_b,
        vis = parent(leaf[:vis]), weights = parent(leaf[:weights]),
    )
end

"""
    point_source_coherency(nbl; flux = 1.0, T = ComplexF64) -> Array{T,3}

The 2×2 source coherency per baseline for an unpolarized point source of `flux`:
parallel hands = `flux`, cross hands = 0, i.e. `S = flux·I` for every baseline.
Indexed `S[bi, fa, fb]`. This is the v1 default; a caller may instead pass a
fixed/known per-baseline coherency (polarized or resolved) of the same shape.
"""
function point_source_coherency(nbl::Integer; flux::Real = 1.0, T = ComplexF64)
    S = zeros(T, nbl, 2, 2)
    @inbounds for bi in 1:nbl
        S[bi, 1, 1] = flux
        S[bi, 2, 2] = flux
    end
    return S
end

"""
    leaf_loglik_gains(g, ctx, S) -> Real

Gaussian WLS log-likelihood (up to an additive constant) of one leaf given the
windowed antenna gains `g :: (nchan, nti, nant, 2)` and the source coherency
`S :: (nbl, 2, 2)`. Accumulates `-½ Σ w |Vobs − g_a·S·conj(g_b)|²` over the
leaf's PRESENT correlation products only. Allocation-free; branches only on data
(`w > 0`, `isfinite(Vobs)`), so it is Enzyme-ready.
"""
function leaf_loglik_gains(
        g::AbstractArray{<:Complex, 4}, ctx, S::AbstractArray{<:Complex, 3},
    )
    Vobs = ctx.vis
    W = ctx.weights
    nchan, nti, nbl, npol = size(Vobs)
    ll = zero(real(eltype(g)))
    @inbounds for p in 1:npol
        fa = ctx.feed_a[p]
        fb = ctx.feed_b[p]
        for bi in 1:nbl
            a = ctx.bl_a[bi]
            b = ctx.bl_b[bi]
            s = S[bi, fa, fb]
            for t in 1:nti, c in 1:nchan
                w = W[c, t, bi, p]
                vo = Vobs[c, t, bi, p]
                (w > 0 && isfinite(vo)) || continue
                v̂ = g[c, t, a, fa] * s * conj(g[c, t, b, fb])
                ll -= w * abs2(vo - v̂) / 2      # `/2` (not `0.5*`) preserves eltype

            end
        end
    end
    return ll
end

"""
    leaf_loglik(ev, θ, geom, leaf; S = nothing, flux = 1.0) -> Real

Per-leaf WLS log-likelihood through the (current flat-θ) `GainEvaluator`:
windowed `evaluate_gains` → [`leaf_loglik_gains`](@ref). `S` defaults to an
unpolarized point source of `flux`. (Milestone 1 seam; the per-site parameter
container replaces the `(ev, θ)` pair in Milestone 2.)
"""
function leaf_loglik(
        ev::GainEvaluator, θ::AbstractVector, geom::DataGeometry, leaf;
        S = nothing, flux::Real = 1.0,
    )
    ctx = build_leaf_ctx(geom, leaf)
    g = evaluate_gains(ev, θ, ctx.chan_idx, ctx.ti_idx)
    Smat = S === nothing ? point_source_coherency(length(ctx.bl_a); flux = flux) : S
    return leaf_loglik_gains(g, ctx, Smat)
end
