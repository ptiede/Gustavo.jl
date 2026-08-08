# ── Ornstein–Uhlenbeck (Matérn-1/2) state-space phase smoother ────────────────
#
# A physically-motivated alternative to the Savitzky–Golay / first-difference
# adhoc smoothers: model each station's residual atmospheric phase track as a
# Gaussian process with a Matérn-1/2 kernel K(Δt) = σ²·exp(-|Δt|/τ) (τ = coherence
# time). Matérn-1/2 is an Ornstein–Uhlenbeck process — a first-order (AR(1))
# linear-Gaussian state-space model — so the GP posterior mean is computed exactly
# and in O(n) by a scalar Kalman filter + RTS smoother, with NO dense covariance
# and NO SparseArrays (Reactant-safe). Unlike the first-difference (random-walk)
# penalty, OU is stationary and mean-reverting with a physical timescale, and its
# Kalman marginal likelihood lets us fit (τ, σ²) per station from the data.
#
# Reference: Blackburn/Bouman et al., AJ (doi:10.3847/1538-3881/ae160f); the OU
# state-space form of a Matérn-1/2 GP is standard (Särkkä & Solin, Applied SDEs).

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
    @inbounds for k in (n - 1):-1:1
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

"""
    fit_ou_hypers(y, w, times; τ0, σ2_0, τ_lo, τ_hi) -> (τ, σ2)

Maximum-likelihood point estimate of the OU coherence time `τ` and stationary
phase variance `σ2` by maximizing the Kalman marginal likelihood
([`kalman_ou_filter`](@ref)) over `(log τ, log σ2)`, seeded from `τ0`/`σ2_0`.
Falls back to the seeds when fewer than 5 samples are observed. `τ` is clamped to
`[τ_lo, τ_hi]` and `σ2` to a small positive floor.
"""
function fit_ou_hypers(y, w, times; τ0::Real, σ2_0::Real, τ_lo::Real, τ_hi::Real)
    T = float(
        promote_type(
            eltype(y), eltype(w), eltype(times),
            typeof(τ0), typeof(σ2_0), typeof(τ_lo), typeof(τ_hi),
        ),
    )
    nobs = count(k -> isfinite(y[k]) && isfinite(w[k]) && w[k] > 0, eachindex(y, w))
    nobs >= 5 || return T(τ0), T(σ2_0)
    r = [(isfinite(wk) && wk > 0) ? inv(T(wk)) : T(Inf) for wk in w]
    σ2_lo = T(1.0e-8)
    lτ_lo = log(T(τ_lo))
    lτ_hi = log(T(τ_hi))
    function negll(p)
        τ = exp(clamp(p[1], lτ_lo, lτ_hi))
        σ2 = max(exp(p[2]), σ2_lo)
        ll = kalman_ou_filter(y, r, times; τ = τ, σ2 = σ2)[6]
        val = isfinite(ll) ? -ll : T(Inf)
        # Soft barrier: outside the [lτ_lo, lτ_hi] box τ saturates (clamped), so the
        # objective would be flat there and Nelder–Mead could converge on a
        # non-optimal boundary. Penalize the excursion so the simplex is driven back
        # into the box toward the true constrained optimum.
        excursion = max(lτ_lo - p[1], zero(T)) + max(p[1] - lτ_hi, zero(T))
        return val + 100 * excursion
    end
    x0 = T[clamp(log(T(τ0)), lτ_lo, lτ_hi), log(max(T(σ2_0), σ2_lo))]
    xbest, _ = _nelder_mead(negll, x0)
    τ = exp(clamp(xbest[1], lτ_lo, lτ_hi))
    σ2 = max(exp(xbest[2]), σ2_lo)
    return τ, σ2
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

# Center a track and fit (or seed) its OU hypers, consistently for every caller.
# Returns `(m, yc, τ, σ2)`: the weighted mean `m`, the mean-subtracted track `yc`
# (OU reverts to 0), and the OU correlation scale / stationary variance — ML-fit
# by Kalman marginal likelihood when `fit`, else `(τ0, seed)`. `σ2_0` seeds the
# stationary variance; `nothing` takes the seed from the track's own scatter.
function _track_ou_hypers(
        trk, w, times; τ0::Real, τ_lo::Real, τ_hi::Real, fit::Bool,
        σ2_0::Union{Nothing, Real} = nothing,
    )
    T = float(promote_type(eltype(trk), eltype(w)))
    m = _weighted_mean_finite(trk, w)
    yc = [isfinite(x) ? T(x) - m : T(NaN) for x in trk]
    σ2_seed = σ2_0 === nothing ? _init_track_var(yc, w) : T(σ2_0)
    τ, σ2 = fit ? fit_ou_hypers(yc, w, times; τ0, σ2_0 = σ2_seed, τ_lo, τ_hi) : (T(τ0), σ2_seed)
    return m, yc, τ, σ2
end

# ── Multivariate OU state-space (joint station-phase solve) ───────────────────
#
# The joint (paper-faithful) solve: instead of solving each AP's station phases
# independently and then smoothing each station track, one MULTIVARIATE OU Kalman
# filter over the whole station-phase vector observes the baseline phase
# DIFFERENCES directly (`ϕ_ij = θ_i − θ_j`) with the per-station OU temporal prior,
# closing and denoising in a single recursive estimator (better at low SNR, where
# a single AP is poorly conditioned and temporal structure resolves it). Each state
# dimension has its own OU `(τ_i, σ_i²)`; the transition is diagonal, so `A P Aᵀ`
# is just `(a_i a_j)·P_ij`. A dimension with `τ_i ≤ 0` is treated as independent
# per step (`a = 0`) with a diffuse prior — a state with no temporal correlation
# to carry.

# Per-dimension exact OU transition, guarding the diffuse (`τ ≤ 0`) dimension.
@inline function _ou_ab(τ::Real, σ2::Real, Δt::Real)
    T = float(promote_type(typeof(τ), typeof(σ2), typeof(Δt)))
    (τ > 0 && Δt != 0) || return (τ > 0 ? one(T) : zero(T)), (τ > 0 ? zero(T) : T(σ2))
    a = exp(-abs(Δt) / τ)
    return T(a), T(σ2 * (1 - a * a))
end

"""
    kalman_ou_mv_filter(rows, ys, rs, times; τ, σ2) -> (xf, Pf, xp, Pp, avecs, loglik)

Forward multivariate OU Kalman filter over closure rows. At step `k`, row `j`
observes a station-phase DIFFERENCE,

    ys[k][j] = x[a] − x[b] + ε,   ε ~ N(0, rs[k][j])

with `(a, b)` read from `rows[k][j]` — an [`_ObsRow`](@ref), whose `val`/`w`/feed
fields are ignored here: the caller passes the processed observation and its
variance in `ys`/`rs`.

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
function kalman_ou_mv_filter(rows, ys, rs, times; τ::AbstractVector, σ2::AbstractVector)
    # Steps and state dimensions are addressed as 1:nsteps / 1:n throughout.
    Base.require_one_based_indexing(τ, σ2, times, rows, ys, rs)
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

        rowsk = rows[k]
        yk = ys[k]
        rk = rs[k]
        for j in eachindex(rowsk, yk, rk)
            yj = yk[j]
            rj = rk[j]
            (isfinite(yj) && isfinite(rj) && rj > 0) || continue
            row = rowsk[j]
            ia, ib = row.a, row.b
            (1 <= ia <= n && 1 <= ib <= n) || throw(
                ArgumentError(
                    "observation row references states $ia/$ib outside the state 1:$n",
                ),
            )
            # h has nonzeros only at ia and ib, so P·h is a difference of two COLUMNS
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
        # G = Pf[k]·Aᵀ·inv(Pp[k+1]); Aᵀ diagonal scales COLUMNS of Pf[k] by a.
        G = (view(Pf, :, :, k) .* reshape(a, 1, :)) / F
        dx = view(xs, :, k + 1) .- view(xp, :, k + 1)
        dP = view(Ps, :, :, k + 1) .- view(Pp, :, :, k + 1)
        mul!(view(xs, :, k), G, dx, true, true)
        mul!(view(Ps, :, :, k), G * dP, transpose(G), true, true)
    end
    return xs, Ps
end
