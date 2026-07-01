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
The prior is the OU stationary distribution `θ_0 ~ N(0, σ2)`; `times` are the
sample epochs (seconds — irregular spacing and gaps allowed).

Returns the filtered mean/variance `μf`/`Pf`, the one-step predicted mean/variance
`μp`/`Pp`, the per-step transition `avec` (`avec[k]` links k-1→k, `avec[1]=0`),
and the log marginal likelihood `Σ_k log N(v_k | 0, S_k)` over observed samples
(the quantity maximized to fit (τ, σ2)).
"""
function kalman_ou_filter(y, r, times; τ::Real, σ2::Real)
    n = length(y)
    μf = zeros(n)
    Pf = zeros(n)
    μp = zeros(n)
    Pp = zeros(n)
    avec = zeros(n)
    loglik = 0.0
    μprev = 0.0
    Pprev = float(σ2)
    @inbounds for k in 1:n
        if k == 1
            a = 0.0
            μpr = 0.0
            Ppr = float(σ2)
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
            loglik += -0.5 * (log(2π * S) + v * v / S)
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

OU Kalman filter + RTS smoother of a per-AP phase track `y` (radians, already
unwrapped and mean-subtracted) with per-sample precision weights `w` (measurement
variance `r = 1/w`; `w ≤ 0` or non-finite ⇒ missing). Returns the smoothed track
with gaps interpolated (matching `_penalized_smooth`).
"""
function smooth_ou_track(y, w, times; τ::Real, σ2::Real)
    n = length(y)
    r = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        wk = w[k]
        r[k] = (isfinite(wk) && wk > 0) ? 1.0 / wk : Inf
    end
    μf, Pf, μp, Pp, avec, _ = kalman_ou_filter(y, r, times; τ = τ, σ2 = σ2)
    μs, _ = rts_smooth(μf, Pf, μp, Pp, avec)
    return μs
end

# Compact fixed-budget Nelder–Mead for low-dimensional unconstrained minimization
# — enough for the 2-parameter (log τ, log σ2) ML fit, avoiding an Optim dependency.
# Standard coefficients (reflect 1, expand 2, contract 1/2, shrink 1/2).
function _nelder_mead(f, x0::Vector{Float64}; step::Real = 0.5, iters::Integer = 200, tol::Real = 1.0e-6)
    n = length(x0)
    simplex = [copy(x0) for _ in 1:(n + 1)]
    for i in 1:n
        simplex[i + 1][i] += step
    end
    fval = [f(s) for s in simplex]
    for _ in 1:iters
        order = sortperm(fval)
        simplex = simplex[order]
        fval = fval[order]
        (abs(fval[end] - fval[1]) <= tol * (abs(fval[1]) + tol)) && break
        c = zeros(n)                                  # centroid of all but worst
        for i in 1:n
            c .+= simplex[i]
        end
        c ./= n
        xw = simplex[end]
        xr = c .+ (c .- xw)                           # reflection
        fr = f(xr)
        if fr < fval[1]
            xe = c .+ 2.0 .* (c .- xw)                # expansion
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
            xc = c .+ 0.5 .* (xw .- c)                # contraction toward centroid
            fc = f(xc)
            if fc < fval[end]
                simplex[end] = xc
                fval[end] = fc
            else
                for i in 2:(n + 1)                    # shrink toward best
                    simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1])
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
    sw = 0.0
    sy = 0.0
    @inbounds for k in eachindex(y)
        yk = y[k]
        wk = w[k]
        (isfinite(yk) && isfinite(wk) && wk > 0) || continue
        sw += wk
        sy += wk * yk
    end
    return sw > 0 ? sy / sw : 0.0
end

# Seed for the OU stationary variance σ2: the weighted sample variance of the
# track minus its expected noise contribution (what's left is signal), floored to
# stay positive.
function _init_track_var(y, w)
    m = _weighted_mean_finite(y, w)
    sw = 0.0
    s2 = 0.0
    nobs = 0
    @inbounds for k in eachindex(y)
        yk = y[k]
        wk = w[k]
        (isfinite(yk) && isfinite(wk) && wk > 0) || continue
        sw += wk
        s2 += wk * (yk - m)^2
        nobs += 1
    end
    nobs > 0 || return 1.0e-4
    var_y = s2 / sw
    # Expected noise contribution to the weight-averaged variance var_y = Σw(y-m)²/Σw
    # is Σ w_k·r_k / Σ w_k = Σ1 / Σw = nobs/sw (since r_k = 1/w_k). Subtracting
    # mean(1/w) instead would be correct only for uniform weights (by AM–HM it
    # over-subtracts, collapsing σ² to the floor for heteroscedastic weights).
    r_bar = nobs / sw
    return max(var_y - r_bar, 1.0e-4)
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
    nobs = count(k -> isfinite(y[k]) && isfinite(w[k]) && w[k] > 0, eachindex(y))
    nobs >= 5 || return float(τ0), float(σ2_0)
    n = length(y)
    r = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        wk = w[k]
        r[k] = (isfinite(wk) && wk > 0) ? 1.0 / wk : Inf
    end
    σ2_lo = 1.0e-8
    lτ_lo = log(τ_lo)
    lτ_hi = log(τ_hi)
    function negll(p)
        τ = exp(clamp(p[1], lτ_lo, lτ_hi))
        σ2 = max(exp(p[2]), σ2_lo)
        ll = kalman_ou_filter(y, r, times; τ = τ, σ2 = σ2)[6]
        val = isfinite(ll) ? -ll : Inf
        # Soft barrier: outside the [lτ_lo, lτ_hi] box τ saturates (clamped), so the
        # objective would be flat there and Nelder–Mead could converge on a
        # non-optimal boundary. Penalize the excursion so the simplex is driven back
        # into the box toward the true constrained optimum.
        excursion = max(lτ_lo - p[1], 0.0) + max(p[1] - lτ_hi, 0.0)
        return val + 100.0 * excursion
    end
    x0 = [clamp(log(τ0), lτ_lo, lτ_hi), log(max(σ2_0, σ2_lo))]
    xbest, _ = _nelder_mead(negll, x0)
    τ = exp(clamp(xbest[1], lτ_lo, lτ_hi))
    σ2 = max(exp(xbest[2]), σ2_lo)
    return τ, σ2
end

# Physical OU coherence-time search bounds from the AP grid, shared by the `:gp` and
# `:gp_joint` adhoc modes so both fit hypers under the same prior: `τ_lo` is one AP
# spacing (floored), `τ_hi` is 10× the observed track span.
function _ou_tau_bounds(times)
    tsec = Float64.(collect(times))
    dts = filter(>(0), diff(sort(tsec)))
    t_ap = isempty(dts) ? 1.0 : median(dts)
    span = length(tsec) > 1 ? (maximum(tsec) - minimum(tsec)) : t_ap
    τ_lo = max(t_ap, 1.0e-3)
    τ_hi = max(10 * span, 10 * τ_lo)
    return τ_lo, τ_hi
end

# Center a phase track and fit (or seed) its OU hypers, consistently for both GP
# adhoc modes. Returns `(m, yc, τ, σ2)`: the weighted mean `m`, the mean-subtracted
# track `yc` (OU reverts to 0), and the OU coherence time / stationary variance —
# ML-fit by Kalman marginal likelihood when `fit`, else `(τ0, seed)`.
function _track_ou_hypers(trk, w, times; τ0::Real, τ_lo::Real, τ_hi::Real, fit::Bool)
    m = _weighted_mean_finite(trk, w)
    yc = [isfinite(x) ? x - m : NaN for x in trk]
    σ2_0 = _init_track_var(yc, w)
    τ, σ2 = fit ? fit_ou_hypers(yc, w, times; τ0 = τ0, σ2_0 = σ2_0, τ_lo = τ_lo, τ_hi = τ_hi) : (float(τ0), σ2_0)
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
# per step (`a = 0`) with a diffuse prior — used for the per-AP source cross-hand
# phase χ carried as an augmented, temporally-uncorrelated state.

# Per-dimension exact OU transition, guarding the diffuse (`τ ≤ 0`) dimension.
@inline function _ou_ab(τ::Real, σ2::Real, Δt::Real)
    (τ > 0 && Δt != 0) || return (τ > 0 ? 1.0 : 0.0), (τ > 0 ? 0.0 : σ2)
    a = exp(-abs(Δt) / τ)
    return a, σ2 * (1 - a * a)
end

"""
    kalman_ou_mv_filter(Hs, ys, rs, times; τ, σ2) -> (xf, Pf, xp, Pp, avecs, loglik)

Forward multivariate OU Kalman filter. At step `k` the observations are
`ys[k] = Hs[k]·x_k + ε`, `ε ~ N(0, Diagonal(rs[k]))`; rows are processed as
independent sequential scalar updates (diagonal `R`, so this is exact and avoids an
`m×m` inverse). `τ`/`σ2` are per-dimension OU parameters (`τ[i] ≤ 0` ⇒ diffuse,
temporally-independent dimension). The prior is `x_0 ~ N(0, Diagonal(σ2))`.
Returns filtered/predicted means and covariances, the per-step diagonal transition
`avecs`, and the joint log marginal likelihood.
"""
function kalman_ou_mv_filter(Hs, ys, rs, times; τ::AbstractVector, σ2::AbstractVector)
    T = length(times)
    n = length(τ)
    xf = [zeros(n) for _ in 1:T]
    Pf = [zeros(n, n) for _ in 1:T]
    xp = [zeros(n) for _ in 1:T]
    Pp = [zeros(n, n) for _ in 1:T]
    avecs = [ones(n) for _ in 1:T]
    loglik = 0.0
    xprev = zeros(n)
    Pprev = Matrix(Diagonal(float.(σ2)))
    for k in 1:T
        if k == 1
            a = ones(n)
            xpr = zeros(n)
            Ppr = Matrix(Diagonal(float.(σ2)))
        else
            Δt = times[k] - times[k - 1]
            a = Vector{Float64}(undef, n)
            Ppr = Matrix{Float64}(undef, n, n)
            q = Vector{Float64}(undef, n)
            @inbounds for i in 1:n
                a[i], q[i] = _ou_ab(τ[i], σ2[i], Δt)
            end
            @inbounds for j in 1:n, i in 1:n
                Ppr[i, j] = a[i] * a[j] * Pprev[i, j]
            end
            @inbounds for i in 1:n
                Ppr[i, i] += q[i]
            end
            xpr = a .* xprev
        end
        avecs[k] = a
        xp[k] = copy(xpr)
        Pp[k] = copy(Ppr)
        x = copy(xpr)
        P = Ppr
        H = Hs[k]
        y = ys[k]
        r = rs[k]
        @inbounds for jrow in eachindex(y)
            yj = y[jrow]
            rj = r[jrow]
            (isfinite(yj) && isfinite(rj) && rj > 0) || continue
            h = @view H[jrow, :]
            Ph = P * h                       # n-vector
            s = dot(h, Ph) + rj
            s > 0 || continue
            innov = yj - dot(h, x)
            invs = 1.0 / s
            K = Ph .* invs                   # Kalman gain (n-vector)
            @inbounds for i in 1:n
                x[i] += K[i] * innov
            end
            # Joseph-form covariance update P ← (I − K hᵀ) P (I − K hᵀ)ᵀ + r K Kᵀ:
            # algebraically equal to P − Ph Phᵀ/s but numerically PSD-stable, so P
            # cannot drift negative-definite and silently drop later rows (via the
            # `s > 0` gate) or break the RTS solve downstream.
            ImKh = I - K * transpose(h)
            P .= ImKh * P * transpose(ImKh) .+ (rj .* (K * transpose(K)))
            loglik += -0.5 * (log(2π * s) + innov * innov * invs)
        end
        xf[k] = x
        Pf[k] = P
        xprev = x
        Pprev = P
    end
    return xf, Pf, xp, Pp, avecs, loglik
end

"""
    rts_smooth_mv(xf, Pf, xp, Pp, avecs) -> (xs, Ps)

Rauch–Tung–Striebel backward smoother paired with [`kalman_ou_mv_filter`](@ref)
(diagonal transition `Diagonal(avecs[k])`).
"""
function rts_smooth_mv(xf, Pf, xp, Pp, avecs)
    T = length(xf)
    xs = deepcopy(xf)
    Ps = deepcopy(Pf)
    for k in (T - 1):-1:1
        a = avecs[k + 1]
        # Guard the predicted covariance before inverting — mirror the scalar
        # `rts_smooth`'s `Ppk > 0 || continue`. A zero-gap step (Δt = 0 ⇒ q = 0) or
        # an ill-conditioned common-mode direction can make Pp[k+1] singular; skip
        # the smoothing update there (leaving xs[k] = xf[k], Ps[k] = Pf[k]).
        F = cholesky(Symmetric(Pp[k + 1]), check = false)
        issuccess(F) || continue
        # G = Pf[k]·Aᵀ·inv(Pp[k+1]); Aᵀ diagonal scales COLUMNS of Pf[k] by a.
        PfA = Pf[k] .* reshape(a, 1, :)
        G = PfA / F
        xs[k] = xf[k] .+ G * (xs[k + 1] .- xp[k + 1])
        Ps[k] = Pf[k] .+ G * (Ps[k + 1] .- Pp[k + 1]) * transpose(G)
    end
    return xs, Ps
end
