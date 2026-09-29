# ── Linear-Gaussian state-space priors along a coordinate ───────────────────
#
# Every correlated prior on a track is a linear-Gaussian state-space model: a
# small state `s` per sample, of which the track value is the first entry,
# evolving over a step `Δ` in the coordinate as `s′ = A s + η`, `η ~ N(0, Q)`.
# One Kalman filter and RTS smoother fit any such model exactly in O(n), and the
# filter gives its marginal likelihood (restricted, for a flat start), from
# which hyperparameters are estimated. docs/src/priors.md derives both.
#
# - `OUModel`: Ornstein–Uhlenbeck (Matérn-1/2), K(Δ) = σ²·exp(-|Δ|/τ); one state,
#   started from its stationary distribution.
# - `RandomWalkModel{M}`: the (M−1)-times integrated Brownian motion; states
#   `(f, f′, …, f^(M−1))`, started flat.
#
# Särkkä & Solin, Applied Stochastic Differential Equations (2019).

# Exact discrete OU transition over a gap Δt for K(Δt)=σ²·exp(-|Δt|/τ): the
# state contracts by a=exp(-|Δt|/τ) toward the (zero) mean, with process
# variance q=σ²(1-a²) injected so the stationary variance stays σ².
@inline function ou_step(τ::Real, σ2::Real, Δt::Real)
    a = exp(-abs(Δt) / τ)
    return a, σ2 * (1 - a * a)
end

"""
    OUModel(τ, σ2)

The Ornstein–Uhlenbeck process with covariance `σ2·exp(-|Δ|/τ)` as a
one-state model, started from its stationary distribution `N(0, σ2)`.
"""
struct OUModel{T}
    τ::T
    σ2::T
end
OUModel(τ, σ2) = OUModel{float(promote_type(typeof(τ), typeof(σ2)))}(τ, σ2)

"""
    RandomWalkModel{M}(σ2)

The `(M−1)`-times integrated Brownian motion whose `(M−1)`-th derivative has
increments `N(0, σ2·|Δ|)`, with state `(f, f′, …, f^(M−1))`. The start is
flat: the first sample's state is unconstrained by the prior.
"""
struct RandomWalkModel{M, T}
    σ2::T
end
RandomWalkModel{M}(σ2) where {M} = RandomWalkModel{M, float(typeof(σ2))}(σ2)

statedim(::OUModel) = 1
statedim(::RandomWalkModel{M}) where {M} = M

_model_eltype(::OUModel{T}) where {T} = T
_model_eltype(::RandomWalkModel{M, T}) where {M, T} = T

# `(A, Q)` over a step `Δ` in the coordinate, in element type `T`.
function transition(m::OUModel, Δ, ::Type{T}) where {T}
    a, q = ou_step(T(m.τ), T(m.σ2), T(Δ))
    return SMatrix{1, 1, T}(a), SMatrix{1, 1, T}(q)
end

function transition(m::RandomWalkModel{M}, Δ, ::Type{T}) where {M, T}
    d = abs(T(Δ))
    σ2 = T(m.σ2)
    A = SMatrix{M, M, T}(ntuple(Val(M * M)) do l
        i, j = (l - 1) % M, (l - 1) ÷ M
        j >= i ? d^(j - i) / factorial(j - i) : zero(T)
    end)
    Q = SMatrix{M, M, T}(ntuple(Val(M * M)) do l
        i, j = (l - 1) % M, (l - 1) ÷ M
        p = 2M - 1 - i - j
        σ2 * d^p / (p * factorial(M - 1 - i) * factorial(M - 1 - j))
    end)
    return A, Q
end

# The starting mean and covariance, or `nothing` for a flat start.
initial(m::OUModel, ::Type{T}) where {T} = SVector{1, T}(0), SMatrix{1, 1, T}(m.σ2)
initial(::RandomWalkModel, ::Type) = nothing

_symmetric(P) = (P + P') / 2

"""
    kalman_filter(model, y, r, x) -> NamedTuple

Forward Kalman filter of the state-space `model` observed as `y_k = s_k[1] + ε_k`,
`ε_k ~ N(0, r_k)`, at the coordinates `x` (any spacing, either direction; only
the steps `x_k − x_{k−1}` enter). A sample with non-finite `y_k` or `r_k` not
positive and finite is missing: predicted through, not updated.

A flat start (`RandomWalkModel`) is filtered in information form until the
observed samples determine the state, then in covariance form; this is exact.
It throws if the samples never determine the state.

Fields of the result, per sample `k`:

- `μf`, `Pf`: filtered mean and covariance, for `k > ndiffuse`;
- `ηf`, `Λf`: filtered information vector and matrix, for `k ≤ ndiffuse`;
- `μp`, `Pp`: predicted mean and covariance, for `k > ndiffuse + 1`, and for
  `k = 1` with a proper start;
- `A`, `Q`: the transition into sample `k` (`k ≥ 2`);
- `v`, `S`: innovation and its variance (`NaN` and `0` where not observed or
  `k ≤ ndiffuse`);
- `loglik`: the log marginal likelihood of the observed samples, restricted
  (the flat start integrated out under the Lebesgue measure on the first
  state) for a flat start.
"""
function kalman_filter(model, y, r, x)
    Base.require_one_based_indexing(y, r, x)
    T = _filter_eltype(model, y, r, x)
    D = statedim(model)
    n = length(y)
    nanvec = SVector{D, T}(ntuple(_ -> T(NaN), D))
    nanmat = SMatrix{D, D, T}(ntuple(_ -> T(NaN), D * D))
    rec = (;
        μf = fill(nanvec, n), Pf = fill(nanmat, n), ηf = fill(nanvec, n), Λf = fill(nanmat, n),
        μp = fill(nanvec, n), Pp = fill(nanmat, n), A = fill(nanmat, n), Q = fill(nanmat, n),
        v = fill(T(NaN), n), S = zeros(T, n),
    )
    loglik, ndiffuse, _, _ = _kalman_forward(model, y, r, x, Val(D), T, Val(false), rec)
    return (; rec..., ndiffuse, loglik)
end

_filter_eltype(model, y, r, x) = float(promote_type(eltype(y), eltype(r), eltype(x), _model_eltype(model)))

# The forward recursion of `kalman_filter`, storing each step into `rec` unless
# it is `nothing`. With `Val(true)` it also filters a track of ones, missing
# where `y` is, under the same gains (proper start only), and accumulates
# `b = Σ vʸ·v¹/S` and `c = Σ (v¹)²/S`. Returns `(loglik, ndiffuse, b, c)`.
function _kalman_forward(model, y, r, x, ::Val{D}, ::Type{T}, ::Val{L}, rec) where {D, T, L}
    Vec = SVector{D, T}
    Mat = SMatrix{D, D, T, D * D}
    e1 = Vec(ntuple(i -> i == 1 ? one(T) : zero(T), Val(D)))
    start = initial(model, T)
    diffuse = isnothing(start)
    L && diffuse &&
        throw(ArgumentError("a level cannot be separated from a flat-start $(nameof(typeof(model)))"))
    ndiffuse = 0
    nobs = 0
    loglik = b = c1 = zero(T)
    μ, P = diffuse ? (zero(Vec), zero(Mat)) : start
    μ1 = zero(Vec)
    # Flat start: the running density of (y_1:k, s_k) is exp(c + ηᵀs − sᵀΛs/2).
    η, Λ, c = zero(Vec), zero(Mat), zero(T)
    for k in eachindex(y, r, x)
        if k > 1
            A, Q = transition(model, x[k] - x[k - 1], T)
            isnothing(rec) || (rec.A[k] = A; rec.Q[k] = Q)
            if diffuse
                # Predict in information form without Q⁻¹, so Q may be singular.
                Ai = inv(A)
                M = Ai' * Λ * Ai
                η̃ = Ai' * η
                F = I + Q * M
                c += dot(η̃, Q * (F' \ η̃)) / 2 - log(abs(det(A))) - log(det(F)) / 2
                Λ = _symmetric(F' \ M)
                η = F' \ η̃
            else
                μ = A * μ
                L && (μ1 = A * μ1)
                P = _symmetric(A * P * A' + Q)
            end
        end
        diffuse || isnothing(rec) || (rec.μp[k] = μ; rec.Pp[k] = P)
        yk, rk = y[k], r[k]
        if isfinite(yk) && isfinite(rk) && rk > 0
            if diffuse
                Λ += e1 * e1' / rk
                η += e1 * (yk / rk)
                c -= (log(2 * T(π) * rk) + yk * yk / rk) / 2
                nobs += 1
            else
                Sk = P[1, 1] + rk
                vk = yk - μ[1]
                K = P[:, 1] / Sk
                μ += K * vk
                if L
                    v1 = one(T) - μ1[1]
                    μ1 += K * v1
                    b += vk * v1 / Sk
                    c1 += v1 * v1 / Sk
                end
                IK = I - K * e1'
                P = _symmetric(IK * P * IK' + K * rk * K')
                isnothing(rec) || (rec.v[k] = vk; rec.S[k] = Sk)
                loglik -= (log(2 * T(π) * Sk) + vk * vk / Sk) / 2
            end
        end
        if diffuse && nobs >= D
            C = cholesky(Symmetric(Matrix(Λ)); check = false)
            if issuccess(C)
                P = _symmetric(inv(Λ))
                μ = P * η
                loglik = c + dot(η, μ) / 2 - logdet(C) / 2 + D * log(2 * T(π)) / 2
                diffuse = false
            end
        end
        if diffuse
            isnothing(rec) || (rec.ηf[k] = η; rec.Λf[k] = Λ)
            ndiffuse = k
        else
            isnothing(rec) || (rec.μf[k] = μ; rec.Pf[k] = P)
        end
    end
    diffuse && throw(
        ArgumentError(
            "the observed samples do not determine the flat-start state of $(nameof(typeof(model))) " *
                "($nobs usable samples, state dimension $D)",
        ),
    )
    return loglik, ndiffuse, b, c1
end

# The log marginal likelihood of `y` under `model` (restricted, for a flat
# start), without storing the per-sample record.
function _kalman_loglik(model, y, r, x)
    Base.require_one_based_indexing(y, r, x)
    T = _filter_eltype(model, y, r, x)
    return first(_kalman_forward(model, y, r, x, Val(statedim(model)), T, Val(false), nothing))
end

"""
    rts_smooth(kf) -> (μs, Ps)

Rauch–Tung–Striebel backward smoother paired with [`kalman_filter`](@ref): the
mean and covariance of every sample's state given the whole track. Samples
filtered in information form are smoothed through the backward conditional
`p(s_k | s_{k+1}, y_{1:k})`, written without `Q⁻¹`.
"""
function rts_smooth(kf)
    (; μf, Pf, ηf, Λf, μp, Pp, A, Q, ndiffuse) = kf
    n = length(μf)
    μs = copy(μf)
    Ps = copy(Pf)
    for k in (n - 1):-1:1
        Ak, Qk = A[k + 1], Q[k + 1]
        if k > ndiffuse
            G = (Pf[k] * Ak') / Pp[k + 1]
            μs[k] = μf[k] + G * (μs[k + 1] - μp[k + 1])
            Ps[k] = _symmetric(Pf[k] + G * (Ps[k + 1] - Pp[k + 1]) * G')
        else
            Ai = inv(Ak)
            G = Ai / (I + Qk * (Ai' * Λf[k] * Ai))
            μs[k] = G * (Qk * (Ai' * ηf[k]) + μs[k + 1])
            Ps[k] = _symmetric(G * Qk * Ai' + G * Ps[k + 1] * G')
        end
    end
    return μs, Ps
end

"""
    smooth_track(model, y, r, x) -> ŷ

The posterior mean of the track value `s_k[1]` at every sample under `model`,
from [`kalman_filter`](@ref) and [`rts_smooth`](@ref); samples without data are
interpolated.
"""
function smooth_track(model, y, r, x)
    μs, _ = rts_smooth(kalman_filter(model, y, r, x))
    return map(first, μs)
end

"""
    smooth_ou_track(y, w, times; τ, σ2) -> ŷ

[`smooth_track`](@ref) under `OUModel(τ, σ2)` of a zero-mean track `y` with
precision weights `w` (`w ≤ 0` or non-finite ⇒ missing), at the coordinates
`times` in `τ`'s unit.
"""
function smooth_ou_track(y, w, times; τ::Real, σ2::Real)
    T = float(promote_type(eltype(y), eltype(w), eltype(times), typeof(τ), typeof(σ2)))
    r = [(isfinite(wk) && wk > 0) ? inv(T(wk)) : T(Inf) for wk in w]
    return smooth_track(OUModel{T}(τ, σ2), y, r, times)
end

# Compact fixed-budget Nelder–Mead for low-dimensional unconstrained minimization
# — enough for the 2-parameter (log τ, log σ2) ML fit, avoiding an Optim dependency.
# Standard coefficients (reflect 1, expand 2, contract 1/2, shrink 1/2).
function _nelder_mead(f, x0::AbstractVector{<:Real}; step::Real = 0.5, iters::Integer = 200, tol::Real = 1.0e-6)
    # The working precision is the starting point's — `step`/`tol` are scale
    # hyperparameters and are converted into it rather than promoted against it.
    T = float(eltype(x0))
    n = length(x0)
    simplex = [collect(T, x0) for _ in 1:(n + 1)]
    for i in 1:n
        simplex[i + 1][i] += T(step)
    end
    fval = [f(s) for s in simplex]
    # Never demand more agreement than the working precision can express.
    rtol = max(T(tol), eps(T))
    for _ in 1:iters
        order = sortperm(fval)
        simplex = simplex[order]
        fval = fval[order]
        (abs(fval[end] - fval[1]) <= rtol * (abs(fval[1]) + rtol)) && break
        c = zeros(T, n)                               # centroid of all but worst
        for i in 1:n
            c .+= simplex[i]
        end
        c ./= n
        xw = simplex[end]
        xr = c .+ (c .- xw)                           # reflection
        fr = f(xr)
        if fr < fval[1]
            xe = c .+ 2 .* (c .- xw)                  # expansion
            fe = f(xe)
            if fe < fr
                simplex[end] = xe
                fval[end] = fe
            else
                simplex[end] = xr
                fval[end] = fr
            end
        elseif fr < fval[end - 1]
            simplex[end] = xr
            fval[end] = fr
        else
            xc = c .+ (xw .- c) ./ 2                  # contraction toward centroid
            fc = f(xc)
            if fc < fval[end]
                simplex[end] = xc
                fval[end] = fc
            else
                for i in 2:(n + 1)                    # shrink toward best
                    simplex[i] = simplex[1] .+ (simplex[i] .- simplex[1]) ./ 2
                    fval[i] = f(simplex[i])
                end
            end
        end
    end
    b = argmin(fval)
    return simplex[b], fval[b]
end

# Weighted mean of the finite entries of a track (weights ≤ 0 / non-finite skipped).
function _weighted_mean_finite(y, w)
    T = float(promote_type(eltype(y), eltype(w)))
    sw = zero(T)
    sy = zero(T)
    for k in eachindex(y, w)
        yk = y[k]
        wk = w[k]
        (isfinite(yk) && isfinite(wk) && wk > 0) || continue
        sw += wk
        sy += wk * yk
    end
    return sw > 0 ? sy / sw : zero(T)
end

# Seed for the OU stationary variance σ2: the weighted sample variance of the
# track minus its expected noise contribution (what's left is signal), floored to
# stay positive.
function _init_track_var(y, w)
    T = float(promote_type(eltype(y), eltype(w)))
    m = _weighted_mean_finite(y, w)
    sw = zero(T)
    s2 = zero(T)
    nobs = 0
    for k in eachindex(y, w)
        yk = y[k]
        wk = w[k]
        (isfinite(yk) && isfinite(wk) && wk > 0) || continue
        sw += wk
        s2 += wk * (yk - m)^2
        nobs += 1
    end
    nobs > 0 || return T(1.0e-4)
    var_y = s2 / sw
    # Expected noise contribution to the weight-averaged variance var_y = Σw(y-m)²/Σw
    # is Σ w_k·r_k / Σ w_k = Σ1 / Σw = nobs/sw (since r_k = 1/w_k). Subtracting
    # mean(1/w) instead would be correct only for uniform weights (by AM–HM it
    # over-subtracts, collapsing σ² to the floor for heteroscedastic weights).
    r_bar = nobs / sw
    return max(var_y - r_bar, T(1.0e-4))
end

# The Kalman filter of a proper-start `model` run over `y` and, with the same
# gains, over a track of ones. The filter is linear in its observations, so the
# innovations of `y - L` are `vʸ - L·v¹`. Returns `(ll, b, c)` over the observed
# samples: the zero-mean log likelihood of `y`, `b = Σ vʸ·v¹/S` and
# `c = Σ (v¹)²/S`. The GLS level of `y` is `b/c`.
function _level_sums(model, y, r, x)
    Base.require_one_based_indexing(y, r, x)
    T = _filter_eltype(model, y, r, x)
    ll, _, b, c = _kalman_forward(model, y, r, x, Val(statedim(model)), T, Val(true), nothing)
    return ll, b, c
end

# The log marginal likelihood of zero-mean OU tracks `ys`, or, with `levels`
# giving each track's level group, of tracks offset by one unknown level per
# group, integrated out under a flat prior (REML): per group
# `Σ ll + b²/2c − log(c)/2`, up to a constant.
function _ou_loglik(ys, rs, xs, levels; τ, σ2)
    model = OUModel(τ, σ2)
    isnothing(levels) && return sum(i -> _kalman_loglik(model, ys[i], rs[i], xs[i]), eachindex(ys, rs, xs))
    ngroup = maximum(levels)
    T = float(typeof(σ2))
    b = zeros(T, ngroup)
    c = zeros(T, ngroup)
    lp = zero(T)
    for i in eachindex(ys, rs, xs, levels)
        ll, bi, ci = _level_sums(model, ys[i], rs[i], xs[i])
        lp += ll
        b[levels[i]] += bi
        c[levels[i]] += ci
    end
    for g in eachindex(b, c)
        c[g] > 0 && (lp += b[g]^2 / (2 * c[g]) - log(c[g]) / 2)
    end
    return lp
end

# Type-II MAP of an `OUPrior`'s hyperparameters shared by a group of tracks:
# the pooled Kalman marginal likelihood (`_ou_loglik`, zero-mean or with each
# `levels` group's level integrated out) plus the hyperprior density over
# `(log τ, log σ)`, whose Jacobian adds `log τ + log σ`. `scale` and `σ` are each
# a fixed value or a density (`is_fixed_hyper`); only the densities are searched.
# `[τ_lo, τ_hi]` bounds the `τ` search and seeds it at its geometric mean.
function _map_ou_hypers(ys, ws, xs, scale, σ; τ_lo::Real, τ_hi::Real, σ2_seed::Real, levels = nothing)
    T = float(
        promote_type(
            typeof(τ_lo), typeof(τ_hi), typeof(σ2_seed),
            (eltype(y) for y in ys)..., (eltype(w) for w in ws)..., (eltype(x) for x in xs)...,
        ),
    )
    fixτ, fixσ = is_fixed_hyper(scale), is_fixed_hyper(σ)
    fixτ && fixσ && return T(scale), T(σ)^2
    rs = [[(isfinite(wk) && wk > 0) ? inv(T(wk)) : T(Inf) for wk in w] for w in ws]
    σ2_lo = T(1.0e-8)
    lτ_lo, lτ_hi = log(T(τ_lo)), log(T(τ_hi))
    lτ0 = fixτ ? log(T(scale)) : (lτ_lo + lτ_hi) / 2
    lσ20 = fixσ ? 2 * log(T(σ)) : log(max(T(σ2_seed), σ2_lo))
    unpack(p) = fixτ ? (lτ0, p[1]) : fixσ ? (p[1], lσ20) : (p[1], p[2])
    function neglp(p)
        lτ, lσ2 = unpack(p)
        τ = fixτ ? exp(lτ) : exp(clamp(lτ, lτ_lo, lτ_hi))
        σ2 = max(exp(lσ2), σ2_lo)
        lp = _ou_loglik(ys, rs, xs, levels; τ, σ2)
        fixτ || (lp += logdensityof(scale, τ) + log(τ))
        fixσ || (lp += logdensityof(σ, sqrt(σ2)) + log(σ2) / 2)
        val = isfinite(lp) ? -lp : T(Inf)
        # Outside the box τ saturates and the objective goes flat; the excursion
        # penalty drives the simplex back in.
        excursion = fixτ ? zero(T) : max(lτ_lo - lτ, zero(T)) + max(lτ - lτ_hi, zero(T))
        return val + 100 * excursion
    end
    x0 = fixτ ? T[lσ20] : fixσ ? T[clamp(lτ0, lτ_lo, lτ_hi)] : T[clamp(lτ0, lτ_lo, lτ_hi), lσ20]
    xbest, _ = _nelder_mead(neglp, x0)
    lτ, lσ2 = unpack(xbest)
    τ = fixτ ? exp(lτ) : exp(clamp(lτ, lτ_lo, lτ_hi))
    return τ, max(exp(lσ2), σ2_lo)
end

# OU correlation-scale search bounds read off the sample coordinate itself, so every
# caller fits hypers under the same prior: `τ_lo` is one median sample spacing
# (floored), `τ_hi` is 10× the observed span. The coordinate is time for the adhoc
# phase smoothers and frequency for the bandpass shape specs.
function _ou_tau_bounds(times)
    T = float(eltype(times))
    ts = sort!(collect(T, times))
    dts = filter(>(zero(T)), diff(ts))
    t_ap = isempty(dts) ? one(T) : T(median(dts))
    span = length(ts) > 1 ? (last(ts) - first(ts)) : t_ap
    τ_lo = max(t_ap, T(1.0e-3))
    τ_hi = max(10 * span, 10 * τ_lo)
    return τ_lo, τ_hi
end

# The same bounds for a group of tracks sharing one correlation scale: the
# tightest `τ_lo` and the widest `τ_hi` any member would impose alone.
#
# Each member's bounds come from its own coordinates, never from the concatenation:
# the scale describes structure inside one track, so the gaps between tracks — the
# jump from one spectral window to the next — carry no shape information and must
# not be mistaken for sample spacing or for span.
function _group_ou_tau_bounds(xs)
    T = float(promote_type((eltype(x) for x in xs)...))
    τ_lo = T(Inf)
    τ_hi = zero(T)
    for x in xs
        lo, hi = _ou_tau_bounds(x)
        τ_lo = min(τ_lo, T(lo))
        τ_hi = max(τ_hi, T(hi))
    end
    isfinite(τ_lo) || return T(1.0e-3), T(1.0e-2)
    return τ_lo, max(τ_hi, 10 * τ_lo)
end

# ── Joint state-space filter over station phases ────────────────────────────
#
# One Kalman filter over the stacked states of every station observes the
# baseline phase differences `ϕ_ab = θ_a − θ_b` directly, each station under its
# own state-space model, closing and denoising in a single recursive estimator
# rather than solving each AP's station phases independently and smoothing each
# track afterwards. This conditions better at low SNR, where a single AP is
# poorly determined and temporal structure resolves it. The joint transition is
# block diagonal, one block per station.

"""
    kalman_mv_filter(pairs, ys, rs, times, models) -> (xf, Pf, xp, Pp, As, loglik)

Forward Kalman filter over the stacked states of `models`, one state-space model
per station, each with a proper start. At step `k`, observation `j` is a
difference of two stations' values (the first entry of each station's state),

    ys[k][j] = θ[a] − θ[b] + ε,   ε ~ N(0, rs[k][j])

with `(a, b) = pairs[j]`, the same station pair at every step; `ys[k]` and
`rs[k]` share `pairs`' indices. An observation whose value is not finite or
whose variance is not positive is skipped, so a step observes any subset.

Rows are applied as sequential scalar updates (diagonal `R`, so this is exact
and avoids an `m×m` inverse); a row has two nonzero design entries, so an update
costs `O(n²)` for `n` stacked states.

Returns the filtered and predicted means as `n × nsteps` matrices, their
covariances and the transitions into each step as `n × n × nsteps` arrays, and
the joint log marginal likelihood — so step `k` is `view(xf, :, k)` /
`view(Pf, :, :, k)`. Station `i`'s value is state `1 + sum(statedim, models[1:i-1])`.
"""
function kalman_mv_filter(pairs, ys, rs, times, models)
    Base.require_one_based_indexing(times, ys, rs, models)
    T = float(
        promote_type(
            eltype(eltype(ys)), eltype(eltype(rs)), eltype(times), map(_model_eltype, models)...,
        ),
    )
    nsteps = length(times)
    nstation = length(models)
    dims = map(statedim, models)
    offsets = cumsum(dims) .- dims
    n = sum(dims)
    starts = [initial(m, T) for m in models]
    any(isnothing, starts) &&
        throw(ArgumentError("the joint filter needs a proper start for every station's model"))
    xf = zeros(T, n, nsteps)
    Pf = zeros(T, n, n, nsteps)
    xp = zeros(T, n, nsteps)
    Pp = zeros(T, n, n, nsteps)
    As = zeros(T, n, n, nsteps)
    loglik = zero(T)
    # Reused across every row of every step: the update touches no other temporary.
    Ph = Vector{T}(undef, n)
    K = Vector{T}(undef, n)
    Q = zeros(T, n, n)
    AP = zeros(T, n, n)
    block(i) = offsets[i] .+ (1:dims[i])
    for k in 1:nsteps
        x = view(xf, :, k)
        P = view(Pf, :, :, k)
        if k == 1
            for i in 1:nstation
                μ0, P0 = starts[i]
                x[block(i)] .= μ0
                P[block(i), block(i)] .= P0
            end
        else
            Δt = times[k] - times[k - 1]
            A = view(As, :, :, k)
            for i in 1:nstation
                Ai, Qi = transition(models[i], Δt, T)
                A[block(i), block(i)] .= Ai
                Q[block(i), block(i)] .= Qi
            end
            mul!(x, A, view(xf, :, k - 1))
            mul!(AP, A, view(Pf, :, :, k - 1))
            mul!(P, AP, A')
            P .+= Q
        end
        copyto!(view(xp, :, k), x)
        copyto!(view(Pp, :, :, k), P)

        yk = ys[k]
        rk = rs[k]
        for j in eachindex(pairs, yk, rk)
            yj = yk[j]
            rj = rk[j]
            (isfinite(yj) && isfinite(rj) && rj > 0) || continue
            sa, sb = pairs[j]
            (1 <= sa <= nstation && 1 <= sb <= nstation) || throw(
                ArgumentError(
                    "observation row references stations $sa/$sb outside 1:$nstation",
                ),
            )
            ia, ib = offsets[sa] + 1, offsets[sb] + 1
            # h has nonzeros only at ia and ib, so P·h is a difference of two columns
            # of P and hᵀv is two of its entries.
            for i in 1:n
                Ph[i] = P[i, ia] - P[i, ib]
            end
            s = Ph[ia] - Ph[ib] + rj
            innov = yj - (x[ia] - x[ib])
            s > 0 || continue
            invs = inv(s)
            for i in 1:n
                K[i] = Ph[i] * invs
                x[i] += K[i] * innov
            end
            # Joseph-form covariance update, in the rank-2 form it collapses to for a
            # scalar row: P ← P − K(Ph)ᵀ − (Ph)Kᵀ + s·KKᵀ. Algebraically this is
            # P − (Ph)(Ph)ᵀ/s, but the grouping keeps P symmetric and PSD, so it cannot
            # drift negative-definite and silently drop later rows (via the `s > 0`
            # gate) or break the RTS solve downstream.
            for jj in 1:n
                Kj = K[jj]
                Phj = Ph[jj]
                for i in 1:n
                    P[i, jj] -= K[i] * Phj + Ph[i] * Kj - s * K[i] * Kj
                end
            end
            loglik -= (log(2 * T(π) * s) + innov * innov * invs) / 2
        end
    end
    return xf, Pf, xp, Pp, As, loglik
end

"""
    rts_smooth_mv(xf, Pf, xp, Pp, As) -> (xs, Ps)

Rauch–Tung–Striebel backward smoother paired with [`kalman_mv_filter`](@ref),
in the same step-sliced layout: `xs` is `n × nsteps` and `Ps` is
`n × n × nsteps`.
"""
function rts_smooth_mv(xf, Pf, xp, Pp, As)
    nsteps = size(xf, 2)
    xs = copy(xf)
    Ps = copy(Pf)
    for k in (nsteps - 1):-1:1
        # A zero-gap step or an ill-conditioned common-mode direction can make the
        # predicted covariance singular; step `k` then keeps its filtered value.
        F = cholesky(Symmetric(Pp[:, :, k + 1]), check = false)
        issuccess(F) || continue
        G = (view(Pf, :, :, k) * view(As, :, :, k + 1)') / F
        dx = view(xs, :, k + 1) .- view(xp, :, k + 1)
        dP = view(Ps, :, :, k + 1) .- view(Pp, :, :, k + 1)
        mul!(view(xs, :, k), G, dx, true, true)
        mul!(view(Ps, :, :, k), G * dP, transpose(G), true, true)
    end
    return xs, Ps
end
