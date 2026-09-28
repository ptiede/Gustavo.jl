# ── Estimating one parameter block under a component prior ──────────────────
#
# A block is one frequency segment of one (station, feed, time segment) track: a
# real observable (unwrapped phase or log-amplitude) per channel `y`, its
# inverse-variance weights `w`, and the channel frequencies `x`. A channel with
# non-finite `y`, or `w` not positive and finite, carries no data. A prior
# relates values within a block, never across blocks.
#
# Fitting is two steps: `_estimate_hypers` fixes a prior's hyperparameters from
# whichever blocks the caller pools, and `_estimate_map` fits one block under the
# resolved prior.

"""
    _estimate_hypers(prior, ys, ws, xs) -> prior

`prior` with every hyperparameter fixed, estimated from the blocks
`(ys[i], ws[i], xs[i])`. An `OUPrior` hyperparameter given as a hyperprior is
estimated by type-II MAP, with each block centered on its own weighted mean; one
given as a number is kept. Blocks without data are ignored; with none left the
hyperpriors cannot be resolved and this throws.
"""
_estimate_hypers(prior::Union{Nothing, RandomWalkPrior}, ys, ws, xs) = prior

function _estimate_hypers(prior::OUPrior{D}, ys, ws, xs) where {D}
    is_fixed_hyper(prior.scale) && is_fixed_hyper(prior.σ) && return prior
    usable = [i for i in eachindex(ys, ws, xs) if any(k -> _shape_usable(ys[i][k], ws[i][k]), eachindex(ys[i], ws[i]))]
    isempty(usable) && throw(
        ArgumentError("cannot estimate the OUPrior hyperparameters: no block carries data"),
    )
    ycs = [_centered(ys[i], ws[i]) for i in usable]
    wus = [ws[i] for i in usable]
    xus = [xs[i] for i in usable]
    τ_lo, τ_hi = _group_ou_tau_bounds(xus)
    τ, σ2 = _map_ou_hypers(
        ycs, wus, xus, prior.scale, prior.σ;
        τ_lo, τ_hi, σ2_seed = _init_track_var(reduce(vcat, ycs), reduce(vcat, wus)),
    )
    return OUPrior(D; scale = τ, σ = sqrt(σ2))
end

"""
    _estimate_map(prior, y, w, x) -> ŷ

The maximum a posteriori values of one block under `prior`, whose
hyperparameters must all be fixed (see [`_estimate_hypers`](@ref)). The block's
posterior is Gaussian, so this is also its posterior mean wherever the prior is
proper along every direction. `NaN` where the data and prior do not determine a
value; a block without data is all `NaN`.

- `nothing`: the measured values.
- `RandomWalkPrior{Frequency}` of order `k`: the `k`-th difference between
  neighboring channels is `N(0, σ²)`. A block with fewer than `k` usable
  channels does not determine the walk and returns its measured values.
- `OUPrior{Frequency}`: an OU process about the block's weighted mean.
"""
function _estimate_map(::Nothing, y, w, x)
    T = _block_eltype(y, w)
    return T[_shape_usable(y[k], w[k]) ? y[k] : T(NaN) for k in eachindex(y, w)]
end

function _estimate_map(prior::RandomWalkPrior{Frequency}, y, w, x)
    count(k -> _shape_usable(y[k], w[k]), eachindex(y, w)) >= prior.order ||
        return _estimate_map(nothing, y, w, x)
    return _random_walk_track(y, w, prior.order, prior.σ)
end

function _estimate_map(prior::OUPrior{Frequency}, y, w, x)
    (is_fixed_hyper(prior.scale) && is_fixed_hyper(prior.σ)) || throw(
        ArgumentError("OUPrior hyperparameters must be fixed; resolve them with `_estimate_hypers` first"),
    )
    T = _block_eltype(y, w)
    any(k -> _shape_usable(y[k], w[k]), eachindex(y, w)) || return fill(T(NaN), length(y))
    m = _weighted_mean_finite(y, w)
    return smooth_ou_track(_centered(y, w), w, x; τ = prior.scale, σ2 = prior.σ^2) .+ m
end

_block_eltype(y, w) = float(promote_type(eltype(y), eltype(w)))

# The block about its weighted mean, `NaN` where it carries no data.
function _centered(y, w)
    T = _block_eltype(y, w)
    m = _weighted_mean_finite(y, w)
    return T[_shape_usable(y[k], w[k]) ? y[k] - m : T(NaN) for k in eachindex(y, w)]
end

# The random-walk MAP `(W + DₖᵀDₖ/σ²)ŷ = Wy`, `Dₖ` the `k`-th
# difference between neighboring channels: banded with bandwidth `k`, so a banded
# Cholesky. A channel without data has weight 0 and the prior alone sets it. The
# system is positive definite once `k` channels carry data, which the caller
# guarantees. Solved in at least `Float64`: a wide gap or a small `σ` puts the
# condition number past what `Float32` resolves.
function _random_walk_track(y, w, order::Integer, σ::Real)
    Base.require_one_based_indexing(y, w)
    T = _block_eltype(y, w)
    S = promote_type(T, Float64)
    n = length(y)
    λ = inv(S(σ)^2)
    c = S[(-1)^(order - j) * binomial(order, j) for j in 0:order]
    N = fill!(BandedMatrix{S}(undef, (n, n), (order, order)), zero(S))
    rhs = zeros(S, n)
    for k in eachindex(y, w)
        if _shape_usable(y[k], w[k])
            N[k, k] += w[k]
            rhs[k] = S(w[k]) * S(y[k])
        end
    end
    for r in 1:(n - order), a in 0:order, b in 0:order
        N[r + a, r + b] += λ * c[a + 1] * c[b + 1]
    end
    return copyto!(similar(y, T, n), cholesky(Symmetric(N)) \ rhs)
end
