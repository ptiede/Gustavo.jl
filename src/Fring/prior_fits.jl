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
`(ys[i], ws[i], xs[i])`. A hyperparameter given as a number is kept; one given
as a hyperprior is estimated by type-II MAP over the blocks together:

- `OUPrior`: zero-mean, or with the level of each group of `level` (one group
  id per block) integrated out under a flat prior.
- `RandomWalkPrior`: the restricted likelihood of each block
  ([`_random_walk_loglik`](@ref)). A walk leaves its own level free, so
  `level` must be `nothing`.

Blocks without data (for a walk of order `m`, with fewer than `m` usable
segments) are ignored; with none left the hyperpriors cannot be resolved and
this throws.
"""
_estimate_hypers(::Nothing, ys, ws, xs; level = nothing) = nothing

function _estimate_hypers(prior::RandomWalkPrior, ys, ws, xs; level = nothing)
    isnothing(level) || throw(ArgumentError("a RandomWalkPrior leaves its level free; it takes no level groups"))
    is_fixed_hyper(prior.σ) && return prior
    m = prior.order
    usable = [i for i in eachindex(ys, ws, xs) if count(k -> _shape_usable(ys[i][k], ws[i][k]), eachindex(ys[i], ws[i])) >= m]
    isempty(usable) && throw(
        ArgumentError("cannot estimate the RandomWalkPrior σ: no block has $m usable segments"),
    )
    T = _blocks_eltype(ys[usable], ws[usable])
    neglp = _RandomWalkNegLogPost(ys[usable], ws[usable], xs[usable], m, prior.σ)
    lσ, _ = _nelder_mead(neglp, T[log(_random_walk_σ_seed(neglp.ys, neglp.ws, neglp.xs, m))])
    return RandomWalkPrior(; order = m, σ = exp(only(lσ)))
end

# The negative log posterior of a random walk's `log σ` over the blocks
# `(ys[i], ws[i], xs[i])`: their restricted likelihoods, the hyperprior density
# of `σ`, and the Jacobian `log σ`.
struct _RandomWalkNegLogPost{Y, W, X, H}
    ys::Y
    ws::W
    xs::X
    order::Int
    hyper::H
end

function (f::_RandomWalkNegLogPost)(p)
    lσ = only(p)
    σ = exp(lσ)
    lp = logdensityof(f.hyper, σ) + lσ
    for i in eachindex(f.ys, f.ws, f.xs)
        lp += _random_walk_loglik(f.ys[i], f.ws[i], f.xs[i], f.order, σ)
    end
    return isfinite(lp) ? -lp : oftype(lp, Inf)
end

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
    T = _blocks_eltype(ys, ws)
    b = zeros(T, nlevel)
    c = zeros(T, nlevel)
    model = OUModel(prior.scale, prior.σ^2)
    for i in eachindex(ys, ws, xs, level)
        r = [_shape_usable(ys[i][k], ws[i][k]) ? inv(T(ws[i][k])) : T(Inf) for k in eachindex(ys[i], ws[i])]
        _, bi, ci = _level_sums(model, _masked(ys[i], ws[i]), r, xs[i])
        b[level[i]] += bi
        c[level[i]] += ci
    end
    return T[c[g] > 0 ? b[g] / c[g] : T(NaN) for g in eachindex(b, c)]
end

"""
    _estimate_map!(out, prior, y, w, x) -> out

Write into `out` the maximum a posteriori values of one block under `prior`,
whose hyperparameters must all be fixed (see [`_estimate_hypers`](@ref)). The
block's posterior is Gaussian, so this is also its posterior mean wherever the
prior is proper along every direction. `NaN` where the data and prior do not
determine a value; a block without data is all `NaN`. `out` may be `y`.

- `nothing`: the measured values.
- `RandomWalkPrior` of order `m`: the `(m−1)`-times integrated Brownian
  motion along `x`, with `σ²` per unit of `x^(2m−1)` (see
  [`_random_walk_track`](@ref)). A block with fewer than `m` usable segments
  does not determine the walk and keeps its measured values.
- `OUPrior`: a zero-mean OU process.
"""
function _estimate_map!(out, ::Nothing, y, w, x)
    T = eltype(out)
    return map!((yk, wk) -> _shape_usable(yk, wk) ? yk : T(NaN), out, y, w)
end

function _estimate_map!(out, prior::RandomWalkPrior, y, w, x)
    is_fixed_hyper(prior.σ) || throw(
        ArgumentError("RandomWalkPrior σ must be fixed; resolve it with `_estimate_hypers` first"),
    )
    count(k -> _shape_usable(y[k], w[k]), eachindex(y, w)) >= prior.order ||
        return _estimate_map!(out, nothing, y, w, x)
    return _random_walk_track!(out, y, w, x, prior.order, prior.σ)
end

function _estimate_map!(out, prior::OUPrior, y, w, x)
    (is_fixed_hyper(prior.scale) && is_fixed_hyper(prior.σ)) || throw(
        ArgumentError("OUPrior hyperparameters must be fixed; resolve them with `_estimate_hypers` first"),
    )
    any(k -> _shape_usable(y[k], w[k]), eachindex(y, w)) || return fill!(out, NaN)
    return copyto!(out, smooth_ou_track(_masked(y, w), w, x; τ = prior.scale, σ2 = prior.σ^2))
end

"""
    _estimate_map(prior, y, w, x) -> ŷ

[`_estimate_map!`](@ref) into a new vector.
"""
_estimate_map(prior, y, w, x) = _estimate_map!(similar(y, _block_eltype(y, w), length(y)), prior, y, w, x)

_block_eltype(y, w) = float(promote_type(eltype(y), eltype(w)))
_blocks_eltype(ys, ws) = mapreduce(i -> _block_eltype(ys[i], ws[i]), promote_type, eachindex(ys, ws))

# The block, `NaN` where it carries no data.
function _masked(y, w)
    T = _block_eltype(y, w)
    return T[_shape_usable(y[k], w[k]) ? y[k] : T(NaN) for k in eachindex(y, w)]
end

"""
    _random_walk_track(y, w, x, order, σ) -> ŷ

The MAP values of `y` under a random walk of order `m = order` along the
strictly monotone coordinates `x`: [`RandomWalkModel`](@ref)`{m}(σ²)`, the
`(m−1)`-times integrated Brownian motion whose `(m−1)`-th derivative has
increments `N(0, σ²·|Δx|)`, with a flat start. A segment without data is
interpolated by the prior. At least `m` segments must carry data.

`x` is rescaled by its median spacing first, so the states have comparable
magnitudes; the fit is in at least `Float64`.
"""
_random_walk_track(y, w, x, order::Integer, σ::Real) =
    _random_walk_track!(similar(y, _block_eltype(y, w), length(y)), y, w, x, order, σ)

# The model's type depends on the runtime `order`; the assertions give callers a
# concrete result type.
function _random_walk_track!(out, y, w, x, order::Integer, σ::Real)
    model, r, u, _ = _random_walk_problem(y, w, x, order, σ)
    return copyto!(out, smooth_track(model, _masked(y, w), r, u)::Vector{eltype(r)})
end

"""
    _random_walk_loglik(y, w, x, order, σ) -> ll

The restricted log likelihood of `y` under the random walk of
[`_random_walk_track`](@ref): the flat start integrated out under the Lebesgue
measure on the first segment's state `(f, f′, …, f^(m−1))`, derivatives in
units of `x`.
"""
function _random_walk_loglik(y, w, x, order::Integer, σ::Real)
    model, r, u, h = _random_walk_problem(y, w, x, order, σ)
    m = Int(order)
    # The filter's derivatives are per unit of `u = x/h`; `f^(j)` in `u` is `hʲ` times `f^(j)` in `x`.
    return _kalman_loglik(model, _masked(y, w), r, u)::eltype(r) - m * (m - 1) * log(h) / 2
end

# The model, measurement variances and coordinates `u = (x − x₁)/h` the random
# walk is filtered on, and `h`, the median spacing; `σ²` is scaled to match.
function _random_walk_problem(y, w, x, order::Integer, σ::Real)
    Base.require_one_based_indexing(y, w, x)
    S = promote_type(_block_eltype(y, w), Float64)
    dx = diff(x)
    (all(>(0), dx) || all(<(0), dx)) ||
        throw(ArgumentError("a random walk needs strictly monotone coordinates"))
    h = isempty(dx) ? one(S) : S(median(abs.(dx)))
    u = (S.(x) .- S(x[1])) ./ h
    r = S[_shape_usable(y[k], w[k]) ? inv(S(w[k])) : S(Inf) for k in eachindex(y, w)]
    return RandomWalkModel{Int(order)}(S(σ)^2 * h^(2order - 1)), r, u, h
end

# Seed for a random walk's σ: the m-th differences of consecutive usable
# segments, less their noise variance, against the walk's variance of an m-th
# difference at the median spacing `h`, `c_m σ² h^(2m−1)`. `c_m` is the order-2m
# cardinal B-spline at its center (1, 2/3, 11/20, …).
function _random_walk_σ_seed(ys, ws, xs, m::Integer)
    T = _blocks_eltype(ys, ws)
    c = sum(k -> (-1)^k * binomial(2m, k) * T(m - k)^(2m - 1), 0:m) / factorial(2m - 1)
    binom = [binomial(m, k) for k in 0:m]
    sig = noise = zero(T)
    nd = 0
    hs = T[]
    for (y, w, x) in zip(ys, ws, xs)
        o = [k for k in eachindex(y, w) if _shape_usable(y[k], w[k])]
        append!(hs, abs.(diff(T.(x[o]))))
        for j in firstindex(o):(lastindex(o) - m)
            idx = o[j:(j + m)]
            d = sum((-1)^k * binom[k + 1] * T(y[idx[k + 1]]) for k in 0:m)
            sig += d^2
            noise += sum(binom[k + 1]^2 / T(w[idx[k + 1]]) for k in 0:m)
            nd += 1
        end
    end
    h = isempty(hs) ? one(T) : median(hs)
    # With no excess over the noise, start well below the noise level.
    excess = nd > 0 ? max(sig - noise, noise / 100) / nd : one(T)
    return sqrt(excess / (c * h^(2m - 1)))
end
