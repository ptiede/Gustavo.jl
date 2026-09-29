# ── Ornstein–Uhlenbeck (Matérn-1/2) state-space phase smoother ───────────────
#
# Models a station's residual atmospheric phase track as a Gaussian process
# with a Matérn-1/2 kernel K(Δt) = σ²·exp(-|Δt|/τ), τ the coherence time.
# Matérn-1/2 is an Ornstein–Uhlenbeck process, a first-order linear-Gaussian
# state-space model, so the GP posterior mean is exact in O(n) from a scalar
# Kalman filter plus RTS smoother — no dense covariance and no SparseArrays.
# The same filter's marginal likelihood is what the prior's hyperparameters
# are estimated from (prior_fits.jl); docs/src/priors.md derives both.
#
# Blackburn/Bouman et al., AJ (doi:10.3847/1538-3881/ae160f); for the OU
# state-space form of a Matérn-1/2 GP see Särkkä & Solin, Applied SDEs.

# Exact discrete OU transition over a time gap Δt for K(Δt)=σ²·exp(-|Δt|/τ): the
# state contracts by a=exp(-|Δt|/τ) toward the (zero) mean, with process variance
# q=σ²(1-a²) injected so the stationary variance stays σ². Irregular Δt (gaps) are
# handled naturally — a is just larger for wider gaps.
@inline function ou_step(τ::Real, σ2::Real, Δt::Real)
    a = exp(-abs(Δt) / τ)
    return a, σ2 * (1 - a * a)
end

"""
    kalman_ou_filter(y, r, times; τ, σ2) -> (μf, Pf, μp, Pp, avec, loglik)

Forward Kalman filter for a scalar Ornstein–Uhlenbeck (Matérn-1/2) state observed
as `y_k = θ_k + ε_k`, `ε_k ~ N(0, r_k)`. A sample with non-finite `y_k` or
non-finite/non-positive `r_k` is treated as MISSING (predict-only, no update).
The prior is the OU stationary distribution `θ_0 ~ N(0, σ2)`.

`times` holds the sample coordinates and `τ` the correlation scale in the same
unit — seconds along a time track, Hz along a frequency track; only their
differences enter. Irregular spacing and gaps are allowed.

Returns the filtered mean/variance `μf`/`Pf`, the one-step predicted mean/variance
`μp`/`Pp`, the per-step transition `avec` (`avec[k]` links k-1→k, `avec[1]=0`),
and the log marginal likelihood `Σ_k log N(v_k | 0, S_k)` over observed samples
(the quantity maximized to fit (τ, σ2)).
"""
function kalman_ou_filter(y, r, times; τ::Real, σ2::Real)
    Base.require_one_based_indexing(y, r, times)
    T = float(
        promote_type(eltype(y), eltype(r), eltype(times), typeof(τ), typeof(σ2)),
    )
    n = length(y)
    μf = zeros(T, n)
    Pf = zeros(T, n)
    μp = zeros(T, n)
    Pp = zeros(T, n)
    avec = zeros(T, n)
    loglik = zero(T)
    μprev = zero(T)
    Pprev = T(σ2)
    for k in eachindex(y, r, times)
        if k == 1
            a = zero(T)
            μpr = zero(T)
            Ppr = T(σ2)
        else
            a, q = ou_step(τ, σ2, times[k] - times[k - 1])
            μpr = a * μprev
            Ppr = a * a * Pprev + q
        end
        avec[k] = a
        μp[k] = μpr
        Pp[k] = Ppr
        yk = y[k]
        rk = r[k]
        if isfinite(yk) && isfinite(rk) && rk > 0
            S = Ppr + rk
            v = yk - μpr
            K = Ppr / S
            μc = μpr + K * v
            Pc = (1 - K) * Ppr
            loglik -= (log(2 * T(π) * S) + v * v / S) / 2
        else
            μc = μpr
            Pc = Ppr
        end
        μf[k] = μc
        Pf[k] = Pc
        μprev = μc
        Pprev = Pc
    end
    return μf, Pf, μp, Pp, avec, loglik
end

"""
    rts_smooth(μf, Pf, μp, Pp, avec) -> (μs, Ps)

Rauch–Tung–Striebel backward smoother paired with [`kalman_ou_filter`](@ref):
returns the smoothed mean/variance conditioned on the whole track. `Ps ≤ Pf`.
"""
function rts_smooth(μf, Pf, μp, Pp, avec)
    n = length(μf)
    μs = copy(μf)
    Ps = copy(Pf)
    for k in (n - 1):-1:1
        Ppk = Pp[k + 1]
        Ppk > 0 || continue
        C = Pf[k] * avec[k + 1] / Ppk
        μs[k] = μf[k] + C * (μs[k + 1] - μp[k + 1])
        Ps[k] = Pf[k] + C * C * (Ps[k + 1] - Ppk)
    end
    return μs, Ps
end

"""
    smooth_ou_track(y, w, times; τ, σ2) -> ŷ

OU Kalman filter + RTS smoother of a real track `y`, mean-subtracted (the process
reverts to zero) and, for a phase track, already unwrapped. `w` holds per-sample
precision weights (measurement variance `r = 1/w`; `w ≤ 0` or non-finite ⇒
missing) and `times` the sample coordinates, in `τ`'s unit — a phase track over
APs or a bandpass track over channel frequencies. Returns the smoothed track with
gaps interpolated (matching `_penalized_smooth`).
"""
function smooth_ou_track(y, w, times; τ::Real, σ2::Real)
    T = float(
        promote_type(eltype(y), eltype(w), eltype(times), typeof(τ), typeof(σ2)),
    )
    r = [(isfinite(wk) && wk > 0) ? inv(T(wk)) : T(Inf) for wk in w]
    μf, Pf, μp, Pp, avec, _ = kalman_ou_filter(y, r, times; τ = τ, σ2 = σ2)
    μs, _ = rts_smooth(μf, Pf, μp, Pp, avec)
    return μs
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

# The OU Kalman filter run over `y` and, with the same gains, over a vector of
# ones. The filter is linear in its observations, so the innovations of
# `y - L` are `vʸ - L·v¹`. Returns `(ll, b, c)` over the observed samples: the
# zero-mean log likelihood of `y`, `b = Σ vʸ·v¹/S` and `c = Σ (v¹)²/S`. The GLS
# level of `y` is `b/c`.
function _ou_level_sums(y, r, times; τ::Real, σ2::Real)
    Base.require_one_based_indexing(y, r, times)
    T = float(promote_type(eltype(y), eltype(r), eltype(times), typeof(τ), typeof(σ2)))
    ll = b = c = zero(T)
    μy = μ1 = zero(T)
    P = T(σ2)
    for k in eachindex(y, r, times)
        if k == 1
            μyp, μ1p, Pp = zero(T), zero(T), T(σ2)
        else
            a, q = ou_step(τ, σ2, times[k] - times[k - 1])
            μyp, μ1p, Pp = a * μy, a * μ1, a * a * P + q
        end
        if isfinite(y[k]) && isfinite(r[k]) && r[k] > 0
            S = Pp + r[k]
            K = Pp / S
            vy, v1 = y[k] - μyp, one(T) - μ1p
            ll -= (log(2 * T(π) * S) + vy * vy / S) / 2
            b += vy * v1 / S
            c += v1 * v1 / S
            μy, μ1, P = μyp + K * vy, μ1p + K * v1, (1 - K) * Pp
        else
            μy, μ1, P = μyp, μ1p, Pp
        end
    end
    return ll, b, c
end

# The log marginal likelihood of zero-mean OU tracks `ys`, or, with `levels`
# giving each track's level group, of tracks offset by one unknown level per
# group, integrated out under a flat prior (REML): per group
# `Σ ll + b²/2c − log(c)/2`, up to a constant.
function _ou_loglik(ys, rs, xs, levels; τ, σ2)
    isnothing(levels) && return sum(i -> kalman_ou_filter(ys[i], rs[i], xs[i]; τ, σ2)[6], eachindex(ys, rs, xs))
    ngroup = maximum(levels)
    T = float(typeof(σ2))
    b = zeros(T, ngroup)
    c = zeros(T, ngroup)
    lp = zero(T)
    for i in eachindex(ys, rs, xs, levels)
        ll, bi, ci = _ou_level_sums(ys[i], rs[i], xs[i]; τ, σ2)
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

# ── Multivariate OU state-space (joint station-phase solve) ─────────────────
#
# One multivariate OU Kalman filter over the whole station-phase vector observes
# the baseline phase differences `ϕ_ij = θ_i − θ_j` directly under the
# per-station OU temporal prior, closing and denoising in a single recursive
# estimator, rather than solving each AP's station phases independently and
# smoothing each track afterwards. This conditions better at low SNR, where a
# single AP is poorly determined and temporal structure resolves it. Each state
# dimension has its own OU `(τ_i, σ_i²)`; the transition is diagonal, so `A P Aᵀ`
# is `(a_i a_j)·P_ij`. A dimension with `τ_i ≤ 0` is treated as independent per
# step (`a = 0`) under a diffuse prior, having no temporal correlation to carry.

# Per-dimension exact OU transition, guarding the diffuse (`τ ≤ 0`) dimension.
@inline function _ou_ab(τ::Real, σ2::Real, Δt::Real)
    T = float(promote_type(typeof(τ), typeof(σ2), typeof(Δt)))
    (τ > 0 && Δt != 0) || return (τ > 0 ? one(T) : zero(T)), (τ > 0 ? zero(T) : T(σ2))
    a = exp(-abs(Δt) / τ)
    return T(a), T(σ2 * (1 - a * a))
end

"""
    kalman_ou_mv_filter(pairs, ys, rs, times; τ, σ2) -> (xf, Pf, xp, Pp, avecs, loglik)

Forward multivariate OU Kalman filter over closure observations. At step `k`,
observation `j` is a station-phase DIFFERENCE,

    ys[k][j] = x[a] − x[b] + ε,   ε ~ N(0, rs[k][j])

with `(a, b) = pairs[j]`, the same state pair at every step; `ys[k]` and
`rs[k]` share `pairs`' indices. An observation whose value is not finite or
whose variance is not positive is skipped, so a step observes any subset.

Rows are applied as sequential scalar updates (diagonal `R`, so this is exact and
avoids an `m×m` inverse). A row has two nonzero design entries whatever `n` is, so
an update costs `O(n²)` — the covariance rank-2 update — where a dense design row
would cost `O(n³)`.

`τ`/`σ2` are per-dimension OU parameters (`τ[i] ≤ 0` ⇒ diffuse, temporally
independent dimension). The prior is `x_0 ~ N(0, Diagonal(σ2))`.

Returns the filtered and predicted means as `n × nsteps` matrices, their covariances
as `n × n × nsteps` arrays, the per-step diagonal transition `avecs` (`n × nsteps`),
and the joint log marginal likelihood — so step `k` is `view(xf, :, k)` /
`view(Pf, :, :, k)`. The element type is promoted from `ys`, `rs`, `τ`, `σ2` and
`times`.
"""
function kalman_ou_mv_filter(pairs, ys, rs, times; τ::AbstractVector, σ2::AbstractVector)
    # Steps and state dimensions are addressed as 1:nsteps / 1:n throughout.
    Base.require_one_based_indexing(τ, σ2, times, ys, rs)
    T = float(
        promote_type(
            eltype(eltype(ys)), eltype(eltype(rs)), eltype(τ), eltype(σ2), eltype(times),
        ),
    )
    nsteps = length(times)
    n = length(τ)
    length(σ2) == n ||
        throw(DimensionMismatch("τ and σ2 must have equal length: $n vs $(length(σ2))"))
    xf = zeros(T, n, nsteps)
    Pf = zeros(T, n, n, nsteps)
    xp = zeros(T, n, nsteps)
    Pp = zeros(T, n, n, nsteps)
    avecs = ones(T, n, nsteps)
    loglik = zero(T)
    # Reused across every row of every step: the update touches no other temporary.
    Ph = Vector{T}(undef, n)
    K = Vector{T}(undef, n)
    q = Vector{T}(undef, n)
    for k in 1:nsteps
        a = view(avecs, :, k)
        x = view(xf, :, k)
        P = view(Pf, :, :, k)
        if k == 1
            for i in 1:n
                P[i, i] = σ2[i]
            end
        else
            Δt = times[k] - times[k - 1]
            xprev = view(xf, :, k - 1)
            Pprev = view(Pf, :, :, k - 1)
            for i in 1:n
                a[i], q[i] = _ou_ab(τ[i], σ2[i], Δt)
            end
            for j in 1:n, i in 1:n
                P[i, j] = a[i] * a[j] * Pprev[i, j]
            end
            for i in 1:n
                P[i, i] += q[i]
                x[i] = a[i] * xprev[i]
            end
        end
        copyto!(view(xp, :, k), x)
        copyto!(view(Pp, :, :, k), P)

        yk = ys[k]
        rk = rs[k]
        for j in eachindex(pairs, yk, rk)
            yj = yk[j]
            rj = rk[j]
            (isfinite(yj) && isfinite(rj) && rj > 0) || continue
            ia, ib = pairs[j]
            (1 <= ia <= n && 1 <= ib <= n) || throw(
                ArgumentError(
                    "observation row references states $ia/$ib outside the state 1:$n",
                ),
            )
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
    return xf, Pf, xp, Pp, avecs, loglik
end

"""
    rts_smooth_mv(xf, Pf, xp, Pp, avecs) -> (xs, Ps)

Rauch–Tung–Striebel backward smoother paired with [`kalman_ou_mv_filter`](@ref)
(diagonal transition `Diagonal(view(avecs, :, k))`), in the same step-sliced layout:
`xs` is `n × nsteps` and `Ps` is `n × n × nsteps`.
"""
function rts_smooth_mv(xf, Pf, xp, Pp, avecs)
    nsteps = size(xf, 2)
    n = size(xf, 1)
    xs = copy(xf)
    Ps = copy(Pf)
    for k in (nsteps - 1):-1:1
        a = view(avecs, :, k + 1)
        # Guard the predicted covariance before inverting — mirror the scalar
        # `rts_smooth`'s `Ppk > 0 || continue`. A zero-gap step (Δt = 0 ⇒ q = 0) or
        # an ill-conditioned common-mode direction can make Pp[:, :, k+1] singular;
        # skip the smoothing update there (leaving step `k` at its filtered value).
        F = cholesky(Symmetric(Pp[:, :, k + 1]), check = false)
        issuccess(F) || continue
        # G = Pf[k]·Aᵀ·inv(Pp[k+1]); Aᵀ diagonal scales columns of Pf[k] by a.
        G = (view(Pf, :, :, k) .* reshape(a, 1, :)) / F
        dx = view(xs, :, k + 1) .- view(xp, :, k + 1)
        dP = view(Ps, :, :, k + 1) .- view(Pp, :, :, k + 1)
        mul!(view(xs, :, k), G, dx, true, true)
        mul!(view(Ps, :, :, k), G * dP, transpose(G), true, true)
    end
    return xs, Ps
end
