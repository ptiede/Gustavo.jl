# ── Estimating one parameter block under a component prior ──────────────────
#
# A block is one spectral window of one (station, feed, time segment) track: a
# real observable (unwrapped phase or log-amplitude) per frequency segment `y`,
# its inverse-variance weights `w`, and the segments' frequencies `x`. A segment
# with non-finite `y`, or `w` not positive and finite, carries no data. A prior
# relates values within a block, never across blocks.
#
# A block may share a level with other blocks: a separate component, flat and
# unknown, that the prior's zero mean makes identifiable. `level` then gives each
# block's level group, and `_estimate_levels` its GLS estimate.
#
# Fitting is in steps: `_estimate_hypers` fixes a prior's hyperparameters from
# whichever blocks the caller pools, `_estimate_levels` the levels under them, and
# `_estimate_map` fits one block, its level removed, under the resolved prior.

_shape_usable(yk, wk) = isfinite(yk) && isfinite(wk) && wk > 0

"""
    _estimate_hypers(prior, ys, ws, xs; level = nothing) -> prior

`prior` with every hyperparameter fixed, estimated from the blocks
`(ys[i], ws[i], xs[i])`. An `OUPrior` hyperparameter given as a hyperprior is
estimated by type-II MAP: zero-mean, or with the level of each group of
`level` (one group id per block) integrated out under a flat prior. One given
as a number is kept. Blocks without data are ignored; with none left the
hyperpriors cannot be resolved and this throws.
"""
_estimate_hypers(prior::Union{Nothing, RandomWalkPrior}, ys, ws, xs; level = nothing) = prior

function _estimate_hypers(prior::OUPrior, ys, ws, xs; level = nothing)
    is_fixed_hyper(prior.scale) && is_fixed_hyper(prior.σ) && return prior
    usable = [i for i in eachindex(ys, ws, xs) if any(k -> _shape_usable(ys[i][k], ws[i][k]), eachindex(ys[i], ws[i]))]
    isempty(usable) && throw(
        ArgumentError("cannot estimate the OUPrior hyperparameters: no block carries data"),
    )
    yus = [_masked(ys[i], ws[i]) for i in usable]
    wus = [ws[i] for i in usable]
    xus = [xs[i] for i in usable]
    τ_lo, τ_hi = _group_ou_tau_bounds(xus)
    τ, σ2 = _map_ou_hypers(
        yus, wus, xus, prior.scale, prior.σ;
        τ_lo, τ_hi, σ2_seed = _init_track_var(reduce(vcat, yus), reduce(vcat, wus)),
        levels = isnothing(level) ? nothing : level[usable],
    )
    return OUPrior(; scale = τ, σ = sqrt(σ2))
end

"""
    _estimate_levels(prior::OUPrior, ys, ws, xs, level, nlevel) -> Vector

The GLS estimate of each of `nlevel` levels, where block `i` is its level
`level[i]` plus a zero-mean process under `prior` (hyperparameters fixed): the
level maximizing the marginal likelihood of its blocks. `NaN` for a level no
block with data holds.
"""
function _estimate_levels(prior::OUPrior, ys, ws, xs, level, nlevel::Integer)
    T = promote_type((_block_eltype(y, w) for (y, w) in zip(ys, ws))...)
    b = zeros(T, nlevel)
    c = zeros(T, nlevel)
    for i in eachindex(ys, ws, xs, level)
        r = [_shape_usable(ys[i][k], ws[i][k]) ? inv(T(ws[i][k])) : T(Inf) for k in eachindex(ys[i], ws[i])]
        _, bi, ci = _ou_level_sums(_masked(ys[i], ws[i]), r, xs[i]; τ = prior.scale, σ2 = prior.σ^2)
        b[level[i]] += bi
        c[level[i]] += ci
    end
    return T[c[g] > 0 ? b[g] / c[g] : T(NaN) for g in eachindex(b, c)]
end

"""
    _estimate_map(prior, y, w, x) -> ŷ

The maximum a posteriori values of one block under `prior`, whose
hyperparameters must all be fixed (see [`_estimate_hypers`](@ref)). The block's
posterior is Gaussian, so this is also its posterior mean wherever the prior is
proper along every direction. `NaN` where the data and prior do not determine a
value; a block without data is all `NaN`.

- `nothing`: the measured values.
- `RandomWalkPrior` of order `m`: the `(m−1)`-times integrated Brownian
  motion along `x`, with `σ²` per unit of `x^(2m−1)` (see
  [`_random_walk_track`](@ref)). A block with fewer than `m` usable segments
  does not determine the walk and returns its measured values.
- `OUPrior`: a zero-mean OU process.
"""
function _estimate_map(::Nothing, y, w, x)
    T = _block_eltype(y, w)
    return T[_shape_usable(y[k], w[k]) ? y[k] : T(NaN) for k in eachindex(y, w)]
end

function _estimate_map(prior::RandomWalkPrior, y, w, x)
    count(k -> _shape_usable(y[k], w[k]), eachindex(y, w)) >= prior.order ||
        return _estimate_map(nothing, y, w, x)
    return _random_walk_track(y, w, x, prior.order, prior.σ)
end

function _estimate_map(prior::OUPrior, y, w, x)
    (is_fixed_hyper(prior.scale) && is_fixed_hyper(prior.σ)) || throw(
        ArgumentError("OUPrior hyperparameters must be fixed; resolve them with `_estimate_hypers` first"),
    )
    T = _block_eltype(y, w)
    any(k -> _shape_usable(y[k], w[k]), eachindex(y, w)) || return fill(T(NaN), length(y))
    return smooth_ou_track(_masked(y, w), w, x; τ = prior.scale, σ2 = prior.σ^2)
end

_block_eltype(y, w) = float(promote_type(eltype(y), eltype(w)))

# The block, `NaN` where it carries no data.
function _masked(y, w)
    T = _block_eltype(y, w)
    return T[_shape_usable(y[k], w[k]) ? y[k] : T(NaN) for k in eachindex(y, w)]
end

"""
    _random_walk_track(y, w, x, order, σ) -> ŷ

The MAP values of `y` under a random walk of order `m = order` along the
strictly monotone coordinates `x`: the `(m−1)`-times integrated Brownian motion
whose `(m−1)`-th derivative has increments `N(0, σ²·|Δx|)`. The walk's value
and first `m−1` derivatives at the start are free (a flat prior).

The state at each sample is `s = (f, f′, …, f^(m−1))`; over a step `Δ` it
evolves as `s′ = A s + η` with `A[i, j] = Δ^(j−i)/(j−i)!` and `η ~ N(0, Q)`,
`Q[i, j] = σ² Δ^(2m−1−i−j) / ((2m−1−i−j)(m−1−i)!(m−1−j)!)` (0-based). The MAP
minimizes `Σ w (y − f)² + Σ (s′ − A s)ᵀ Q⁻¹ (s′ − A s)` over every state, a
banded system solved by a banded Cholesky factorization. A segment without
data has weight 0 and the prior alone sets it. The system is positive definite
once `m` segments carry data, which the caller guarantees.

`x` is rescaled by its median spacing before the solve, so the states have
comparable magnitudes; the solve is in at least `Float64`.
"""
function _random_walk_track(y, w, x, order::Integer, σ::Real)
    Base.require_one_based_indexing(y, w, x)
    T = _block_eltype(y, w)
    S = promote_type(T, Float64)
    n = length(y)
    m = Int(order)
    Δx = S[abs(x[k] - x[k - 1]) for k in 2:n]
    (all(>(0), diff(x)) || all(<(0), diff(x))) || throw(
        ArgumentError("a random walk needs strictly monotone coordinates"),
    )
    h = isempty(Δx) ? one(S) : median(Δx)
    σ2 = S(σ)^2 * h^(2m - 1)
    N = fill!(BandedMatrix{S}(undef, (n * m, n * m), (2m - 1, 2m - 1)), zero(S))
    rhs = zeros(S, n * m)
    at(k, i) = (k - 1) * m + i + 1
    for k in eachindex(y, w)
        if _shape_usable(y[k], w[k])
            N[at(k, 0), at(k, 0)] += w[k]
            rhs[at(k, 0)] = S(w[k]) * S(y[k])
        end
    end
    for k in 2:n
        Δ = Δx[k - 1] / h
        A = S[j >= i ? Δ^(j - i) / factorial(j - i) : zero(S) for i in 0:(m - 1), j in 0:(m - 1)]
        Q = S[
            σ2 * Δ^(2m - 1 - i - j) / ((2m - 1 - i - j) * factorial(m - 1 - i) * factorial(m - 1 - j))
                for i in 0:(m - 1), j in 0:(m - 1)
        ]
        B = [-A Matrix{S}(I, m, m)]
        C = B' * (cholesky(Symmetric(Q)) \ B)
        idx = at(k - 1, 0):at(k, m - 1)
        for (b, jj) in pairs(idx), (a, ii) in pairs(idx)
            N[ii, jj] += C[a, b]
        end
    end
    s = cholesky(Symmetric(N)) \ rhs
    return copyto!(similar(y, T, n), s[at.(1:n, 0)])
end
