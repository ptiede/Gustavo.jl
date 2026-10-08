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
    _proper_prior(prior) -> Bool

Whether `prior` is proper along every direction of a block, so that a level
beside it is identifiable: an `OUPrior`, or a `RandomWalkPrior` with an `init`.
"""
_proper_prior(::OUPrior) = true
_proper_prior(p::RandomWalkPrior) = !isnothing(p.init)
_proper_prior(_) = false

# A resolved prior is `nothing` or a NamedTuple keyed by the dimensions it runs along.
_is_prior_along(::Nothing, dim) = true
_is_prior_along(p::NamedTuple, dim) = keys(p) == (dim,)
_is_prior_along(_, dim) = false

_prior_along(::Nothing, dim) = nothing
_prior_along(p::NamedTuple, dim) = p[dim]

"""
    _estimate_hypers(prior, ys, ws, xs; level = nothing) -> prior

`prior` with every hyperparameter fixed, estimated from the blocks
`(ys[i], ws[i], xs[i])`. A hyperparameter given as a number is kept; one given
as a hyperprior is estimated by type-II MAP over the blocks together, on their
marginal likelihood: zero-mean, or with the level of each group of `level`
(one group id per block) integrated out under a flat prior. A walk without a
`init` leaves its own level free, so it takes no `level`; its likelihood is
the restricted one ([`_random_walk_loglik`](@ref)).

Blocks without data (for a walk without an `init` of order `m`, with fewer
than `m` usable segments) are ignored; with none left the hyperpriors cannot
be resolved and this throws.
"""
_estimate_hypers(::Nothing, ys, ws, xs; level = nothing) = nothing

function _estimate_hypers(prior::RandomWalkPrior, ys, ws, xs; level = nothing)
    _proper_prior(prior) || isnothing(level) || throw(
        ArgumentError("a RandomWalkPrior without an init leaves its level free; it takes no level groups"),
    )
    is_fixed_hyper(prior.σ) && return prior
    m = prior.order
    need = _proper_prior(prior) ? 1 : m
    usable = [i for i in eachindex(ys, ws, xs) if count(k -> _shape_usable(ys[i][k], ws[i][k]), eachindex(ys[i], ws[i])) >= need]
    isempty(usable) && throw(
        ArgumentError("cannot estimate the RandomWalkPrior σ: no block has $need usable segments"),
    )
    T = _blocks_eltype(ys[usable], ws[usable])
    problems = [_random_walk_problem(ys[i], ws[i], xs[i]) for i in usable]
    neglp = _RandomWalkNegLogPost(
        [_masked(ys[i], ws[i]) for i in usable], first.(problems), getindex.(problems, 2), last.(problems),
        m, prior.init, prior.σ, isnothing(level) ? nothing : level[usable],
    )
    lσ, _ = _nelder_mead(neglp, T[log(_random_walk_σ_seed(ys[usable], ws[usable], xs[usable], m))])
    return RandomWalkPrior(; order = m, σ = exp(only(lσ)), prior.init)
end

# The negative log posterior of a random walk's `log σ` over blocks already set
# up for the filter (masked values `ys`, variances `rs`, rescaled coordinates
# `us`, spacings `hs`): their pooled marginal likelihood (restricted, for a flat
# init or with `levels`), the hyperprior density of `σ`, and the Jacobian
# `log σ`.
struct _RandomWalkNegLogPost{Y, R, U, H, S, P, L}
    ys::Y
    rs::R
    us::U
    hs::H
    order::Int
    init::S
    hyper::P
    levels::L
end

function (f::_RandomWalkNegLogPost)(p)
    lσ = only(p)
    σ = exp(lσ)
    S = eltype(eltype(f.rs))
    models = [_random_walk_model(f.order, σ, f.init, h, S) for h in f.hs]
    lp = _pooled_loglik(models, f.ys, f.rs, f.us, f.levels)::S + logdensityof(f.hyper, σ) + lσ
    if isnothing(f.init)
        for h in f.hs
            lp -= _flat_start_jacobian(f.order, h)
        end
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
    _estimate_levels(prior, ys, ws, xs, level, nlevel) -> Vector

The GLS estimate of each of `nlevel` levels, where block `i` is its level
`level[i]` plus a zero-mean process under the proper `prior` (hyperparameters
fixed; see [`_proper_prior`](@ref)): the level maximizing the marginal
likelihood of its blocks. `NaN` for a level no block with data holds.
"""
function _estimate_levels(prior, ys, ws, xs, level, nlevel::Integer)
    _proper_prior(prior) || throw(
        ArgumentError("a level is not identifiable beside $(isnothing(prior) ? "no prior" : _call_string(prior))"),
    )
    T = _blocks_eltype(ys, ws)
    b = zeros(T, nlevel)
    c = zeros(T, nlevel)
    for i in eachindex(ys, ws, xs, level)
        model, ym, r, u = _filter_inputs(prior, ys[i], ws[i], xs[i])
        _, bi, ci = _level_sums(model, ym, r, u)::NTuple{3, eltype(r)}
        b[level[i]] += bi
        c[level[i]] += ci
    end
    return T[c[g] > 0 ? b[g] / c[g] : T(NaN) for g in eachindex(b, c)]
end

# One block as the filter takes it under a prior with fixed hyperparameters: the
# model, the values `NaN` where they carry no data, the measurement variances,
# and the coordinates.
function _filter_inputs(prior::OUPrior, y, w, x)
    T = float(promote_type(_block_eltype(y, w), typeof(prior.scale), typeof(prior.σ)))
    r = T[_shape_usable(y[k], w[k]) ? inv(T(w[k])) : T(Inf) for k in eachindex(y, w)]
    return OUModel{T}(prior.scale, prior.σ^2), _masked(y, w), r, x
end

function _filter_inputs(prior::RandomWalkPrior, y, w, x)
    r, u, h = _random_walk_problem(y, w, x)
    return _random_walk_model(prior.order, prior.σ, prior.init, h, eltype(r)), _masked(y, w), r, u
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
  motion along `x`, with `σ²` per unit of `x^(2m−1)`, from its `init` (see
  [`_random_walk_track`](@ref)). Without an init, a block with fewer than `m`
  usable segments does not determine the walk and keeps its measured values.
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
    nusable = count(k -> _shape_usable(y[k], w[k]), eachindex(y, w))
    _proper_prior(prior) && nusable == 0 && return fill!(out, NaN)
    nusable >= prior.order || _proper_prior(prior) || return _estimate_map!(out, nothing, y, w, x)
    return _random_walk_track!(out, y, w, x, prior.order, prior.σ; prior.init)
end

function _estimate_map!(out, prior::OUPrior, y, w, x)
    (is_fixed_hyper(prior.scale) && is_fixed_hyper(prior.σ)) || throw(
        ArgumentError("OUPrior hyperparameters must be fixed; resolve them with `_estimate_hypers` first"),
    )
    any(k -> _shape_usable(y[k], w[k]), eachindex(y, w)) || return fill!(out, NaN)
    return copyto!(out, smooth_ou_track(_masked(y, w), w, x; τ = prior.scale, σ2 = prior.σ^2))
end

"""
    _prior_energy(prior, y, x) -> E

`−log` of the resolved `prior`'s density at the track `y` over `x`, less a
normalization that depends only on `prior` and `x` (see `_exact_energy`).
Non-finite entries are unobserved; `nothing` contributes zero.
"""
_prior_energy(::Nothing, y, x) = 0.0

function _prior_energy(prior::RandomWalkPrior, y, x)
    _, u, h = _random_walk_problem(y, ones(length(y)), x)
    return _exact_energy(_random_walk_model(prior.order, prior.σ, prior.init, h, Float64), y, u)
end

_prior_energy(prior::OUPrior, y, x) =
    _exact_energy(OUModel(Float64(prior.scale), Float64(prior.σ)^2), y, Float64.(x))

# `½·Σ ν²/s` over the innovations of `model`'s filter with `y` observed exactly:
# the negative log density of `y` less its normalization, which the innovation
# variances `s` alone carry and which does not depend on `y`. A flat start's
# first `statedim` observations fix its state and are left out, so a random walk
# of order `m` does not see a polynomial of degree below `m`; with fewer
# observations than that the energy is zero.
function _exact_energy(model, y, u)
    D = statedim(model)
    start = initial(model, Float64)
    flat = isnothing(start)
    x = flat ? zero(SVector{D, Float64}) : SVector{D, Float64}(start[1])
    # A flat start as a variance far above any step's.
    P = flat ? SMatrix{D, D, Float64}(1.0e8 * I) : SMatrix{D, D, Float64}(start[2])
    E = 0.0
    seen = 0
    prev = 0
    for k in eachindex(y, u)
        isfinite(y[k]) || continue
        if prev > 0
            A, Q = transition(model, u[k] - u[prev], Float64)
            x = A * x
            P = A * P * A' + Q
        end
        ν = y[k] - x[1]
        s = P[1, 1]
        seen += 1
        (flat && seen <= D) || (E += ν^2 / (2s))
        K = P[:, 1] / s
        x = x + K * ν
        P = _symmetric(P - K * P[1, :]')
        prev = k
    end
    return E
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
    _random_walk_track(y, w, x, order, σ; init = nothing) -> ŷ

The MAP values of `y` under a random walk of order `m = order` along the
strictly monotone coordinates `x`: the `(m−1)`-times integrated Brownian
motion whose `(m−1)`-th derivative has increments `N(0, σ²·|Δx|)`
([`RandomWalkModel`](@ref)), from `init` (a `Normal` or `MvNormal` on
`(f, f′, …, f^(m−1))` at `x[1]`, in units of `x`) or, with `init = nothing`,
a flat start. A segment without data is interpolated by the prior. With a
flat start at least `m` segments must carry data.

`x` is rescaled by its median spacing first, so the states have comparable
magnitudes; the fit is in at least `Float64`.
"""
_random_walk_track(y, w, x, order::Integer, σ::Real; init = nothing) =
    _random_walk_track!(similar(y, _block_eltype(y, w), length(y)), y, w, x, order, σ; init)

# The model's type depends on the runtime `order`; the assertions give callers a
# concrete result type.
function _random_walk_track!(out, y, w, x, order::Integer, σ::Real; init = nothing)
    r, u, h = _random_walk_problem(y, w, x)
    model = _random_walk_model(order, σ, init, h, eltype(r))
    return copyto!(out, smooth_track(model, _masked(y, w), r, u)::Vector{eltype(r)})
end

"""
    _random_walk_loglik(y, w, x, order, σ; init = nothing) -> ll

The log marginal likelihood of `y` under the random walk of
[`_random_walk_track`](@ref). With a flat start it is the restricted one: the
start integrated out under the Lebesgue measure on the first segment's state
`(f, f′, …, f^(m−1))`, derivatives in units of `x`.
"""
function _random_walk_loglik(y, w, x, order::Integer, σ::Real; init = nothing)
    r, u, h = _random_walk_problem(y, w, x)
    model = _random_walk_model(order, σ, init, h, eltype(r))
    ll = _kalman_loglik(model, _masked(y, w), r, u)::eltype(r)
    return isnothing(init) ? ll - _flat_start_jacobian(order, h) : ll
end

# The filter's derivatives are per unit of `u = x/h`, and `f^(j)` in `u` is `hʲ`
# times `f^(j)` in `x`, so the Lebesgue measure on the start differs by `h^Σj`.
_flat_start_jacobian(order, h) = order * (order - 1) * log(h) / 2

# The measurement variances and the coordinates `u = (x − x₁)/h` a random walk is
# filtered on, and `h`, the median spacing.
function _random_walk_problem(y, w, x)
    Base.require_one_based_indexing(y, w, x)
    S = promote_type(_block_eltype(y, w), Float64)
    dx = diff(x)
    (all(>(0), dx) || all(<(0), dx)) ||
        throw(ArgumentError("a random walk needs strictly monotone coordinates"))
    h = isempty(dx) ? one(S) : S(median(abs.(dx)))
    u = (S.(x) .- S(x[1])) ./ h
    r = S[_shape_usable(y[k], w[k]) ? inv(S(w[k])) : S(Inf) for k in eachindex(y, w)]
    return r, u, h
end

# The walk's model on `u = (x − x₁)/h`: `σ²` scaled by `h^(2m−1)`, and the init's
# `j`-th derivative by `hʲ`.
function _random_walk_model(order::Integer, σ, init, h, ::Type{S}) where {S}
    σ2 = S(σ)^2 * h^(2order - 1)
    isnothing(init) && return RandomWalkModel{Int(order)}(σ2)
    μ, Σ = _init_moments(init)
    d = [h^j for j in 0:(order - 1)]
    return RandomWalkModel{Int(order)}(σ2, S.(d .* μ), S.(d .* Σ .* d'))
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
