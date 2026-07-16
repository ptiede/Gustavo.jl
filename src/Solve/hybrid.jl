# ── Hybrid solve: direct linear per-slice phase block steps ──────────────────
#
# A per-CHANNEL phase bandpass and a per-AP adhoc phase are both LINEAR in their
# parameters given the other gains, and both are large and weakly coupled — so
# LBFGS crawls over them (150+ iters on real VLBA data). Solve each DIRECTLY:
# divide out the current gains, coherently average the residual over the OTHER
# axis, and stationize the residual phase per slice with the globally-closing WLS
# `solve_adhoc_phasing`. The bandpass and the adhoc are the SAME operation on
# orthogonal axes:
#
#   * `axis = :freq` — bandpass: average over TIME, one phase per (station, feed)
#     per CHANNEL.  (`solve_adhoc_phasing` with the channel axis as its "AP" axis.)
#   * `axis = :time` — adhoc: average over FREQUENCY, one phase per (station, feed)
#     per AP.  (`solve_adhoc_phasing`'s native use.)
#
# A block-coordinate driver alternates LBFGS (the nonlinear delay/rate, both blocks
# frozen) with these linear steps.

using ..Fringe: solve_adhoc_phasing, NoSmoothing, AbstractAdhocSmoother, _chi_sign, _node
using ..UVData: materialize_leaf, branches, baselines, pol_products
using ..Calibration: correlation_feed_pair
using LinearSolve: LinearProblem, KrylovJL_CG, FunctionOperator, init, solve!
import LinearAlgebra

# Accumulate one leaf's residual phasor per (baseline, product, SLICE), coherently
# over the averaged axis, after dividing out the current gains `g`. `axis = :freq`
# slices on the GLOBAL channel (`gci`), averaging over time; `axis = :time` slices
# on the GLOBAL AP (`gti`), averaging over frequency. The inverse-variance weight
# `w·|den|²` matches the adhoc's `_accumulate_leaf_rbar!`.
function _accumulate_leaf_phasor!(z, wsum, V, W, g, bl_a, bl_b, feed_a, feed_b, gci, gti, freqaxis::Bool)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa = feed_a[p]
        fb = feed_b[p]
        for bi in 1:nbl
            a = bl_a[bi]
            b = bl_b[bi]
            a == b && continue
            for t in 1:nti, c in 1:nchan
                ww = W[c, t, bi, p]
                (ww > 0 && isfinite(ww)) || continue
                ga = g[c, t, a, fa]
                gb = g[c, t, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, t, bi, p] / den
                isfinite(v) || continue
                wd = ww * abs2(den)
                s = freqaxis ? gci[c] : gti[t]
                z[bi, p, s] += wd * v
                wsum[bi, p, s] += wd
            end
        end
    end
    return z, wsum
end

# ADD a per-(antenna, feed, slice) phase increment into the named component's
# block(s). Feeds sharing a feed block are AVERAGED (so a SharedFeeds component is
# written once, not double-added; a PerFeed one gets each feed in its own block).
# `axis = :freq` writes `A[c_local, 1, fseg, fb, la]` (PerChannel × GlobalTime);
# `axis = :time` writes `A[1, tseg, 1, fb, la]` (scalar × PerIntegration × GlobalFreq).
function _add_phase_component!(p, plan::GainPlan, phase::Array{Float64, 3}, name::Symbol, freqaxis::Bool)
    nslice = freqaxis ? plan.nchan : plan.ntime
    for gi in eachindex(plan.groups)
        gp = plan.groups[gi]
        ci = findfirst(v -> _sym(v) === name, gp.phase_syms)
        ci === nothing && continue
        comp = gp.phase[ci]
        parr, _ = group_arrays(plan, p, gi)
        A = parr[ci]
        @inbounds for (la, ant) in enumerate(gp.ants), b in 1:comp.nfb
            for s in 1:nslice
                acc = 0.0
                n = 0
                for feed in 1:2
                    comp.fb1[feed] == b || continue
                    v = phase[ant, feed, s]
                    isfinite(v) || continue
                    acc += v
                    n += 1
                end
                n == 0 && continue
                val = acc / n
                if freqaxis
                    A[comp.clocal[s], 1, comp.fseg_id[s], b, la] += val
                else
                    A[1, comp.tseg_id[s], 1, b, la] += val
                end
            end
        end
    end
    return p
end

# ── Exact MAP block update: prior-coupled matrix-free CG ──────────────────────
#
# The one-shot `solve_adhoc_phasing` solves the DATA (ML) closure per slice and
# leaves the block's prior (BandpassARPrior across channels, OUPrior across APs)
# unused — fine at high SNR (ML≈MAP), lossy on weak channels/APs. The prior COUPLES
# slices, so the block stops separating per slice; the MAP block update is ONE
# coupled linear system (design: plan "Exact MAP block update" §):
#
#     (H_data + H_prior) x = b
#       H_data  = ⊕_slices Aᵀ diag(W_s) A     couples STATIONS within a slice
#       H_prior = ⊕_nodes  Q                  couples SLICES within a node
#
# solved matrix-free by CG. `x[node, slice]` is a per-(station[, feed]) phase
# increment; the refant node is pinned to 0. Per-feed, the CROSS hands tie the two
# feed gauges through a per-slice source cross-hand phase `χ` (see `_map_block_solve`);
# the shared-feeds adhoc uses parallel hands only. An outer Gauss–Newton loop rotates
# the accumulated residual phasor `z` in place (phase-only ⇒ weights `W` unchanged ⇒
# no data re-read), exact at any SNR; one iteration suffices at high SNR.

# Find the `SiteComponent` named `name` (phase components only) — its segmentation
# tables drive the prior precision. Structure is identical across antenna groups, so
# the first match is representative.
function _find_phase_component(plan::GainPlan, name::Symbol)
    for gp in plan.groups
        ci = findfirst(v -> _sym(v) === name, gp.phase_syms)
        ci === nothing || return gp.phase[ci]
    end
    return nothing
end

# H_data · X : per slice, the weighted station graph Laplacian of the edges.
# `na`/`nb` are per-edge node indices; `W[edge, slice]` the phase weights.
function _apply_data!(Y, X, na, nb, W)
    fill!(Y, 0.0)
    nedge, nslice = size(W)
    @inbounds for s in axes(W, 2), e in axes(W, 1)
        w = W[e, s]
        w == 0 && continue
        d = w * (X[na[e], s] - X[nb[e], s])
        Y[na[e], s] += d
        Y[nb[e], s] -= d
    end
    return Y
end

# RHS AᵀW·θ with θ = angle(Z): b[node, slice] = Σ_edge (±) W·angle(Z).
function _data_rhs!(B, Z, W, na, nb)
    fill!(B, 0.0)
    nedge, nslice = size(W)
    @inbounds for s in axes(W, 2), e in axes(W, 1)
        w = W[e, s]
        w == 0 && continue
        t = w * angle(Z[e, s])
        B[na[e], s] += t
        B[nb[e], s] -= t
    end
    return B
end

# H_prior · X: the (single-source-of-truth) prior precision `Q` applied along the
# slice axis for every node. `Q` is symmetric banded, given as blocks (one spw's
# channels for `:freq`, all APs for `:time`) each in band form `(qdiag, qsuper)`
# (`qdiag[k] = Q[k,k]`, `qsuper[o][k] = Q[k, k+o]`) with bandwidth `kd`. ADDS `Q·X`
# into `Y`. The SAME `(blocks, qbands, kd)` (from `_prior_bands`) feed both this
# matrix-free apply and the preconditioner factorization, so they cannot desync.
function _apply_banded_blocks!(Y, X, blocks, qbands, kd::Integer)
    @inbounds for a in axes(X, 1)
        for bi in eachindex(blocks)
            idx = blocks[bi]
            qdiag, qsuper = qbands[bi]
            m = length(idx)
            for k in eachindex(idx)
                acc = qdiag[k] * X[a, idx[k]]
                for o in 1:kd
                    k + o <= m && (acc += qsuper[o][k] * X[a, idx[k + o]])
                    k - o >= 1 && (acc += qsuper[o][k - o] * X[a, idx[k - o]])
                end
                Y[a, idx[k]] += acc
            end
        end
    end
    return Y
end

# A reusable MATRIX-FREE CG solver for the FIXED operator `applyM!(Y, X)` on
# `(nnode, nslice)` MATRICES — a thin reshape adapter over the flat solver below
# (used by the operator≡dense tests; the production path uses the flat form directly).
# Returns a closure `bmat -> Δmat`.
function _map_cg_solver(applyM!, nnode::Integer, nslice::Integer; abstol, reltol, maxiters)
    flatapply = (w, v) -> (applyM!(reshape(w, nnode, nslice), reshape(v, nnode, nslice)); w)
    flatsolve = _map_cg_solver_flat(flatapply, nnode * nslice; abstol = abstol, reltol = reltol, maxiters = maxiters)
    return bmat -> reshape(flatsolve(vec(bmat)), nnode, nslice)
end

# Flat-vector variant: `applyM!(w, v)` already acts on flat length-`N` vectors (the
# augmented `[vec(x); χ]` state of the per-feed MAP block). Same LinearSolve machinery,
# returns a closure `bvec -> Δvec` reusing one cache across the GN iterations. `Pl` is
# an optional left preconditioner (a `MapBlockPrecond`); without it the biharmonic AR
# prior + zero-weight flagged-channel runs make unpreconditioned CG crawl (κ ~ N⁴).
function _map_cg_solver_flat(applyM!, N::Integer; Pl = nothing, abstol, reltol, maxiters)
    opfun = function (w, v, u, p, t)
        applyM!(w, v)
        return w
    end
    op = FunctionOperator(
        opfun, zeros(N), zeros(N);
        islinear = true, isposdef = true, issymmetric = true,
    )
    prob = LinearProblem(op, zeros(N))
    cache = Pl === nothing ?
        init(prob, KrylovJL_CG(); abstol = abstol, reltol = reltol, maxiters = maxiters) :
        init(prob, KrylovJL_CG(); Pl = Pl, abstol = abstol, reltol = reltol, maxiters = maxiters)
    return function (bvec)
        cache.b = bvec
        sol = solve!(cache)
        return copy(sol.u)
    end
end

# ── Block-Jacobi preconditioner for the MAP block CG ──────────────────────────
#
# The coupled operator `M = data-Laplacian + prior Q + χ-ridge` is stiff: the AR
# prior [2,-1] is biharmonic (κ ~ N⁴ over a spw's channels) and flagged-channel runs
# leave long pure-prior stretches, so unpreconditioned CG needs 10³–10⁴ iterations.
# Preconditioning per node with `(D_node + Q)⁻¹` along the slice axis — the exact
# inverse of the diagonal-data + prior part, leaving only the (well-conditioned)
# station off-diagonal coupling for CG — plus a diagonal `1/(dχ + λχ)` for the χ
# nuisance, cuts that to ~10–100 iterations. `Q` is banded (bandwidth `kd` = AR order,
# or 1 for OU), so each block is a dense-banded LU (LAPACK `gbtrf!`/`gbtrs!`; NO
# SparseArrays — [[reactant-no-sparse-arrays]]).
struct MapBlockPrecond
    ABs::Matrix{Matrix{Float64}}                 # (node, block) → gbtrf!-factored band
    ipivs::Matrix{Vector{LinearAlgebra.BlasInt}} # (node, block) → gbtrf! pivots
    blocks::Vector{Vector{Int}}     # block → slice indices (a spw's channels / all APs)
    kd::Int
    nnode::Int
    nslice::Int
    nn::Int
    dχinv::Vector{Float64}          # 1/(dχ + λχ), length nslice (empty if no χ)
    refnodes::Vector{Int}
    buf::Vector{Float64}            # reused gather/solve scratch (≥ any block length)
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::MapBlockPrecond, x::AbstractVector)
    Xx = reshape(view(x, 1:P.nn), P.nnode, P.nslice)
    Yx = reshape(view(y, 1:P.nn), P.nnode, P.nslice)
    @inbounds for a in axes(Xx, 1)
        for bi in eachindex(P.blocks)
            idx = P.blocks[bi]
            rhs = view(P.buf, eachindex(idx))   # reused scratch (CG applies P serially)
            for t in eachindex(idx)
                rhs[t] = Xx[a, idx[t]]
            end
            LinearAlgebra.LAPACK.gbtrs!('N', P.kd, P.kd, length(idx), P.ABs[a, bi], P.ipivs[a, bi], rhs)
            for t in eachindex(idx)
                Yx[a, idx[t]] = rhs[t]
            end
        end
    end
    for r in P.refnodes
        @views Yx[r, :] .= 0.0
    end
    @inbounds for s in eachindex(P.dχinv)
        y[P.nn + s] = x[P.nn + s] * P.dχinv[s]
    end
    return y
end
LinearAlgebra.ldiv!(P::MapBlockPrecond, x::AbstractVector) = copyto!(x, LinearAlgebra.ldiv!(similar(x), P, x))
Base.eltype(::MapBlockPrecond) = Float64
Base.size(P::MapBlockPrecond) = (P.nn + length(P.dχinv), P.nn + length(P.dχinv))
Base.size(P::MapBlockPrecond, ::Integer) = P.nn + length(P.dχinv)

# Band representation `(qdiag, qsuper)` of the anchored-conditional AR(p) precision for
# one channel block of length `m` (`qdiag[k] = Q[k,k]`, `qsuper[o][k] = Q[k, k+o]`):
# `a0` on the first `p` diagonal entries, `aε·gᵀg` curvature (`g = [1, −φ…]`). Filled
# directly in band form (O(m·p²), no dense `m×m` build).
function _ar_bands(m::Integer, φ, a0, aε)
    p = length(φ)
    qdiag = zeros(m)
    qsuper = [zeros(max(m - o, 0)) for o in 1:p]
    c = vcat(1.0, [-φ[j] for j in 1:p])          # residual-row coefficients
    @inbounds for k in 1:min(p, m)
        qdiag[k] += a0
    end
    @inbounds for k in (p + 1):m
        for i1 in 0:p, i2 in 0:p
            r1 = k - i1
            r2 = k - i2
            v = aε * c[i1 + 1] * c[i2 + 1]
            if r1 == r2
                qdiag[r1] += v
            elseif r2 > r1
                qsuper[r2 - r1][r1] += v          # symmetric partner (i2,i1) fills r2<r1
            end
        end
    end
    return qdiag, qsuper
end

# The prior's banded precision `Q` as `(blocks, qbands, kd)` — the SINGLE source of
# truth for both the matrix-free apply (`_apply_banded_blocks!`) and the preconditioner
# factorization. One block per spw (AR across that spw's channels) for `:freq`, one
# block (OU tridiagonal over the APs) for `:time`. The axis↔prior-type coupling (AR is a
# frequency prior, OU a time prior) and the segmentation assumptions are validated here.
function _prior_bands(pr::BandpassARPrior, comp::SiteComponent, geom, freqaxis::Bool, nslice::Integer)
    freqaxis || error("BandpassARPrior is a frequency prior; use it with axis = :freq")
    length(comp.fseg_id) == nslice || error(
        "MAP block: BandpassARPrior expects one channel per slice (nslice = $nslice, " *
            "fseg_id length = $(length(comp.fseg_id))).",
    )
    groups = [Int[] for _ in 1:comp.nfseg]
    @inbounds for c in eachindex(comp.fseg_id)
        push!(groups[comp.fseg_id[c]], c)
    end
    a0 = 1.0 / pr.σ0^2
    aε = 1.0 / pr.σε^2
    qbands = [_ar_bands(length(g), pr.φ, a0, aε) for g in groups]
    return groups, qbands, length(pr.φ)
end
function _prior_bands(pr::OUPrior, comp::SiteComponent, geom, freqaxis::Bool, nslice::Integer)
    freqaxis && error("OUPrior is a time prior; use it with axis = :time")
    comp.ntseg == nslice || error(
        "MAP block: OUPrior exact path needs one time segment per slice (PerIntegration): " *
            "nslice = $nslice, ntseg = $(comp.ntseg).",
    )
    st = _segment_times_sec(comp, geom)
    d, e, _ = _ou_precision(pr.τ, pr.σ2, st)
    return [collect(1:nslice)], [(d, [e])], 1
end
_prior_bands(pr::AbstractPrior, ::SiteComponent, geom, freqaxis::Bool, nslice::Integer) =
    error(
    "MAP block: no exact-solve precision for prior $(typeof(pr)); use a " *
        "BandpassARPrior (freq) or OUPrior (time), or drop the prior for the one-shot path."
)

# Factor `(Diagonal(D_node) + Q)` per (node, block) into band storage; assemble the
# χ-diagonal inverse. `blocks`/`qbands`/`kd` are the prior bands (from `_prior_bands`);
# `Ddiag[node, s]` is the operator's data diagonal (Σ edge weights at that node/slice),
# `dχ[s]` the cross-hand weight sum.
function _build_map_precond(
        blocks, qbands, kd::Integer,
        nnode::Integer, nslice::Integer, Ddiag, dχ, λχ, refnodes, use_chi::Bool,
    )
    nblock = length(blocks)
    ABs = Matrix{Matrix{Float64}}(undef, nnode, nblock)
    ipivs = Matrix{Vector{LinearAlgebra.BlasInt}}(undef, nnode, nblock)
    # gbtrf! (general banded LU) band storage: kl = ku = kd, `A[i,j]` at
    # `AB[kl+ku+1+i-j, j]`; the leading `kl` rows are pivot fill space.
    kl = ku = kd
    for a in axes(ABs, 1), bi in axes(ABs, 2)
        idx = blocks[bi]
        qdiag, qsuper = qbands[bi]
        m = length(idx)
        AB = zeros(2kl + ku + 1, m)
        @inbounds for j in eachindex(idx)
            AB[kl + ku + 1, j] = qdiag[j] + Ddiag[a, idx[j]]     # diagonal
            for o in 1:kd
                j - o >= 1 && (AB[kl + ku + 1 - o, j] = qsuper[o][j - o])  # super A[j-o,j]
                j + o <= m && (AB[kl + ku + 1 + o, j] = qsuper[o][j])      # sub  A[j+o,j]
            end
        end
        ABf, ipiv = LinearAlgebra.LAPACK.gbtrf!(kl, ku, m, AB)
        ABs[a, bi] = ABf
        ipivs[a, bi] = ipiv
    end
    dχinv = use_chi ? [1.0 / (dχ[s] + λχ) for s in eachindex(dχ)] : Float64[]
    buf = Vector{Float64}(undef, isempty(blocks) ? 0 : maximum(length, blocks))
    return MapBlockPrecond(ABs, ipivs, blocks, kd, nnode, nslice, nnode * nslice, dχinv, collect(Int, refnodes), buf)
end

# The prior-coupled MAP block solve. Accumulated per-(baseline, product, global
# slice) residual phasor `z`/`wsum` (from `_accumulate_leaf_phasor!`) → a per-slice
# station[, feed] phase increment via matrix-free CG, returned as a
# `phase[nant, 2, nslice]` array (same shape `solve_adhoc_phasing` returns), so the
# `_add_phase_component!` write-back is unchanged.
#
# PER-FEED (`shared_feeds = false`, the bandpass): all four products are used — the
# parallel hands (PP/QQ) drive each feed's own nodes, and the CROSS hands (PQ/QP)
# tie the two feed gauges together through a per-slice source cross-hand phase `χ_s`
# (the row model is `angle(z) ≈ x[a,fa] − x[b,fb] + cs·χ_s`, `cs = _chi_sign(fa,fb)`).
# Without the cross hands the feed-1 and feed-2 node graphs are DISCONNECTED and the
# per-station R–L relative phase/delay is unconstrained; the cross-hand χ tie is what
# makes the per-feed solutions globally close (refant pinned in BOTH feeds; the
# remaining global R–L/EVPA gauge sits in `χ`, fixed downstream). `χ` is a pure
# nuisance (discarded), regularized by a tiny ridge so slices with no cross-hand
# data stay well-posed.
#
# SHARED-FEEDS (`shared_feeds = true`, the atmospheric adhoc): parallel hands ONLY —
# the atmosphere is non-birefringent, and cross hands carry field-rotation/D-term
# phase that would staircase the feed-common track (matches `solve_adhoc_phasing`).
function _map_block_solve(
        z, wsum, bl_a, bl_b, feed_a, feed_b, nant::Integer, nslice::Integer,
        prior::AbstractPrior, comp::SiteComponent, geom, freqaxis::Bool;
        ref_ant::Integer = 1, shared_feeds::Bool = false,
        map_maxgn::Integer = 5, map_tol::Real = 1.0e-8,
        cg_tol::Real = 1.0e-10, cg_maxit::Integer = 2000,
    )
    nnode = shared_feeds ? nant : 2 * nant

    # Edges: parallel hands always; cross hands only PER-FEED (they carry the χ tie).
    ena = Int[]
    enb = Int[]
    ecs = Int[]                       # χ sign per edge (0 parallel, ±1 cross)
    ebi = Int[]
    epi = Int[]
    for pidx in eachindex(feed_a, feed_b)
        fa = feed_a[pidx]
        fb = feed_b[pidx]
        cs = _chi_sign(fa, fb)
        shared_feeds && cs != 0 && continue      # drop cross hands for the adhoc
        for bi in eachindex(bl_a, bl_b)
            a = bl_a[bi]
            b = bl_b[bi]
            a == b && continue
            push!(ena, shared_feeds ? a : _node(a, fa, nant))
            push!(enb, shared_feeds ? b : _node(b, fb, nant))
            push!(ecs, cs)
            push!(ebi, bi)
            push!(epi, pidx)
        end
    end
    nedge = length(ena)

    # Dense per-edge phasor `Z` and fixed phase weight `W = |z|²/wsum` (≈ coherent
    # inverse-variance; matches the adhoc's |r|²/w). `Z` is rotated by the GN loop.
    Z = zeros(ComplexF64, nedge, nslice)
    W = zeros(Float64, nedge, nslice)
    @inbounds for s in axes(W, 2), e in axes(W, 1)
        zz = z[ebi[e], epi[e], s]
        ws = wsum[ebi[e], epi[e], s]
        Z[e, s] = zz
        W[e, s] = (ws > 0 && isfinite(ws) && isfinite(zz)) ? abs2(zz) / ws : 0.0
    end

    # A per-slice χ nuisance (source cross-hand phase) exists only PER-FEED with cross
    # hands present. A tiny ridge (relative to the typical weight) keeps χ well-posed
    # where cross-hand data is thin without biasing where it is not.
    use_chi = !shared_feeds && any(!=(0), ecs)
    nchi = use_chi ? nslice : 0
    nposw = count(>(0), W)
    wscale = nposw == 0 ? 1.0 : sum(w for w in W if w > 0) / nposw
    λχ = 1.0e-6 * wscale

    # The prior precision `Q` as banded blocks — the single source of truth shared by
    # the matrix-free apply below AND the preconditioner (so they cannot desync).
    blocks, qbands, kd = _prior_bands(prior, comp, geom, freqaxis, nslice)

    refnodes = shared_feeds ? (ref_ant,) : (ref_ant, nant + ref_ant)
    nn = nnode * nslice
    ntot = nn + nchi
    # Matrix-free apply on the FLAT state `u = [vec(x); χ]`: data closure GᵀWG (node
    # Laplacian + χ coupling) + prior Q on nodes + ridge on χ, refant nodes pinned.
    function applyM!(w, v)
        Xx = reshape(view(v, 1:nn), nnode, nslice)
        Yx = reshape(view(w, 1:nn), nnode, nslice)
        fill!(Yx, 0.0)
        Xχ = use_chi ? view(v, (nn + 1):ntot) : view(v, 1:0)
        Yχ = use_chi ? view(w, (nn + 1):ntot) : view(w, 1:0)
        use_chi && fill!(Yχ, 0.0)
        @inbounds for s in axes(W, 2), e in axes(W, 1)
            ww = W[e, s]
            ww == 0 && continue
            d = Xx[ena[e], s] - Xx[enb[e], s]
            (use_chi && ecs[e] != 0) && (d += ecs[e] * Xχ[s])
            d *= ww
            Yx[ena[e], s] += d
            Yx[enb[e], s] -= d
            (use_chi && ecs[e] != 0) && (Yχ[s] += ecs[e] * d)
        end
        _apply_banded_blocks!(Yx, Xx, blocks, qbands, kd)
        if use_chi
            @inbounds for s in eachindex(Yχ, Xχ)
                Yχ[s] += λχ * Xχ[s]
            end
        end
        for r in refnodes
            @views Yx[r, :] .= 0.0
        end
        return w
    end

    # Block-Jacobi preconditioner: the operator's data diagonal per node (Σ edge
    # weights) + cross-hand weight per slice feed `(D_node + Q)⁻¹` / `1/(dχ + λχ)`.
    Ddiag = zeros(Float64, nnode, nslice)
    dχ = zeros(Float64, nslice)
    @inbounds for s in axes(W, 2), e in axes(W, 1)
        ww = W[e, s]
        ww == 0 && continue
        Ddiag[ena[e], s] += ww
        Ddiag[enb[e], s] += ww
        (use_chi && ecs[e] != 0) && (dχ[s] += ww)
    end
    precond = _build_map_precond(
        blocks, qbands, kd, nnode, nslice, Ddiag, dχ, λχ, refnodes, use_chi,
    )
    solve_cg = _map_cg_solver_flat(applyM!, ntot; Pl = precond, abstol = cg_tol, reltol = cg_tol, maxiters = cg_maxit)

    u = zeros(Float64, ntot)
    x = reshape(view(u, 1:nn), nnode, nslice)
    χ = use_chi ? view(u, (nn + 1):ntot) : view(u, 1:0)
    b = zeros(Float64, ntot)
    bx = reshape(view(b, 1:nn), nnode, nslice)
    bχ = use_chi ? view(b, (nn + 1):ntot) : view(b, 1:0)
    QX = zeros(Float64, nnode, nslice)
    for _ in 1:map_maxgn
        # b = GᵀW·angle(Z) − [Q·x; λχ·χ]  (prior/ridge pull the estimate toward 0).
        fill!(b, 0.0)
        @inbounds for s in axes(W, 2), e in axes(W, 1)
            ww = W[e, s]
            ww == 0 && continue
            t = ww * angle(Z[e, s])
            bx[ena[e], s] += t
            bx[enb[e], s] -= t
            (use_chi && ecs[e] != 0) && (bχ[s] += ecs[e] * t)
        end
        fill!(QX, 0.0)
        _apply_banded_blocks!(QX, x, blocks, qbands, kd)
        @. bx -= QX
        use_chi && @. bχ -= λχ * χ
        for r in refnodes
            @views bx[r, :] .= 0.0
        end
        Δ = solve_cg(b)
        Δx = reshape(view(Δ, 1:nn), nnode, nslice)
        Δχ = use_chi ? view(Δ, (nn + 1):ntot) : view(Δ, 1:0)
        @. u += Δ
        # Rotate Z by −(Δx_a − Δx_b + cs·Δχ): the residual after this GN step.
        @inbounds for s in axes(W, 2), e in axes(W, 1)
            W[e, s] == 0 && continue
            r = Δx[ena[e], s] - Δx[enb[e], s]
            (use_chi && ecs[e] != 0) && (r += ecs[e] * Δχ[s])
            Z[e, s] *= cis(-r)
        end
        maximum(abs, Δ) < map_tol && break
    end

    phase = zeros(Float64, nant, 2, nslice)
    @inbounds for s in axes(phase, 3), a in axes(phase, 1)
        phase[a, 1, s] = x[a, s]
        phase[a, 2, s] = shared_feeds ? x[a, s] : x[nant + a, s]
    end
    return phase
end

"""
    refine_phase_component!(p, plan, uvset, geom; component, axis = :freq,
        ref_ant = 1, shared_feeds = false, prior = NoPrior(),
        smoother = NoSmoothing(detrend = false, phase_rewrap_iters = 0)) -> p

Solve a per-slice phase component DIRECTLY (in place, one linear step): divide out
`p`'s current gains, coherently average the residual over the axis ORTHOGONAL to
`axis`, and stationize the residual phase per slice with the globally-closing
[`solve_adhoc_phasing`](@ref), adding the result into `component`.

- `axis = :freq` — the per-channel phase BANDPASS (average over time).
- `axis = :time` — the per-AP ADHOC phase (average over frequency).

`shared_feeds = true` solves one feed-common phase per slice (for a `SharedFeeds`
component, e.g. the adhoc). Far faster and better-conditioned than LBFGS over the
~10³–10⁴ weakly-coupled params of these blocks.

With a non-trivial `prior` (a [`BandpassARPrior`](@ref) on a `:freq` component or an
[`OUPrior`](@ref) on a `:time` component), the per-slice ML closure is replaced by
the EXACT prior-coupled MAP block update ([`_map_block_solve`](@ref)) — a matrix-free
CG over the coupled `(node × slice)` system, outer Gauss–Newton for large residuals.
This regularizes weak channels/APs (the prior bites at low SNR). `prior = NoPrior()`
(the default) keeps the robust one-shot `solve_adhoc_phasing` path.
"""
function refine_phase_component!(
        p, plan::GainPlan, uvset, geom;
        component::Symbol, axis::Symbol = :freq,
        ref_ant::Integer = 1, shared_feeds::Bool = false,
        prior::AbstractPrior = NoPrior(),
        map_maxgn::Integer = 5, map_tol::Real = 1.0e-8,
        cg_tol::Real = 1.0e-10, cg_maxit::Integer = 2000,
        smoother::AbstractAdhocSmoother = NoSmoothing(detrend = false, phase_rewrap_iters = 0),
    )
    axis in (:freq, :time) || error("refine_phase_component!: axis must be :freq or :time (got :$axis)")
    freqaxis = axis === :freq
    nant = plan.nant
    nslice = freqaxis ? plan.nchan : plan.ntime

    l0 = materialize_leaf(first(values(branches(uvset))); layers = (:vis, :weights, :uvw))
    bl = baselines(l0).pairs
    bl_a = Int[Int(q[1]) for q in bl]
    bl_b = Int[Int(q[2]) for q in bl]
    pols = String.(pol_products(l0))
    feeds = correlation_feed_pair.(pols)
    feed_a = Int[f[1] for f in feeds]
    feed_b = Int[f[2] for f in feeds]
    nbl = length(bl)
    npol = length(pols)

    z = zeros(ComplexF64, nbl, npol, nslice)
    wsum = zeros(Float64, nbl, npol, nslice)
    for (_, leaf) in branches(uvset)
        leaf = materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
        ci, ti = leaf_window(geom, leaf)
        g = evaluate_gains(plan, p, ci, ti)
        _accumulate_leaf_phasor!(
            z, wsum, parent(leaf[:vis]), parent(leaf[:weights]), g,
            bl_a, bl_b, feed_a, feed_b, ci, ti, freqaxis,
        )
    end

    if prior isa NoPrior
        # The slice index stands in for `solve_adhoc_phasing`'s AP axis: unit "times"
        # for channels (freq), physical epochs (s) for APs (time).
        times = freqaxis ? collect(1.0:nslice) : (geom.times .* 3600.0)
        as = solve_adhoc_phasing(
            z, wsum, [(bl_a[i], bl_b[i]) for i in 1:nbl], pols, nant, times;
            ref_ant = ref_ant, smoother = smoother, shared_feeds = shared_feeds,
        )
        phase = as.phase
    else
        comp = _find_phase_component(plan, component)
        comp === nothing && error(
            "refine_phase_component!: no phase component named `$component` for the MAP block solve.",
        )
        phase = _map_block_solve(
            z, wsum, bl_a, bl_b, feed_a, feed_b, nant, nslice, prior, comp, geom, freqaxis;
            ref_ant = ref_ant, shared_feeds = shared_feeds,
            map_maxgn = map_maxgn, map_tol = map_tol, cg_tol = cg_tol, cg_maxit = cg_maxit,
        )
    end
    _add_phase_component!(p, plan, phase, component, freqaxis)
    return p
end

"""
    refine_phase_bandpass!(p, plan, uvset, geom; bandpass = :bandpass, kw...) -> p

The per-channel phase-bandpass case of [`refine_phase_component!`](@ref) (`axis =
:freq`).
"""
refine_phase_bandpass!(p, plan::GainPlan, uvset, geom; bandpass::Symbol = :bandpass, kw...) =
    refine_phase_component!(p, plan, uvset, geom; component = bandpass, axis = :freq, kw...)
