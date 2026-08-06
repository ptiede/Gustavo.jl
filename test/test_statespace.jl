# Ornstein–Uhlenbeck (Matérn-1/2) state-space phase smoother primitives.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using LinearAlgebra
using Statistics: mean, var

const FRs = Gustavo.Fringe

# Dense Matérn-1/2 GP covariance K[i,j] = σ²·exp(-|tᵢ-tⱼ|/τ).
_matern12(times, τ, σ2) = [σ2 * exp(-abs(times[i] - times[j]) / τ) for i in eachindex(times), j in eachindex(times)]

@testset "OU step (Matérn-1/2 discretization)" begin
    a, q = FRs.ou_step(3.0, 0.7, 0.0)
    @test a ≈ 1.0
    @test q ≈ 0.0                                   # no gap ⇒ no process noise
    a, q = FRs.ou_step(3.0, 0.7, 1.5)
    @test a ≈ exp(-1.5 / 3.0)
    @test q ≈ 0.7 * (1 - a^2)                       # stationary variance preserved
    # Symmetric in the sign of the gap.
    @test FRs.ou_step(3.0, 0.7, -1.5) == FRs.ou_step(3.0, 0.7, 1.5)
end

@testset "Kalman filter loglik == dense GP loglik" begin
    rng = MersenneTwister(0x000A11CE)
    n = 14
    times = sort(cumsum(rand(rng, n)) .* 3.0)       # irregular spacing
    τ, σ2 = 4.0, 0.7
    r = fill(0.05, n)
    y = randn(rng, n) .* 0.5
    K = _matern12(times, τ, σ2)
    C = K + Diagonal(r)
    dense_ll = -0.5 * (logdet(C) + dot(y, C \ y) + n * log(2π))
    kal_ll = FRs.kalman_ou_filter(y, r, times; τ = τ, σ2 = σ2)[6]
    @test isapprox(kal_ll, dense_ll; atol = 1.0e-9)
end

@testset "RTS smoother == GP posterior mean; variance shrinks" begin
    rng = MersenneTwister(0x0B0B)
    n = 14
    times = sort(cumsum(rand(rng, n)) .* 3.0)
    τ, σ2 = 4.0, 0.7
    r = fill(0.05, n)
    y = randn(rng, n) .* 0.5
    K = _matern12(times, τ, σ2)
    C = K + Diagonal(r)
    gp_mean = K * (C \ y)                            # full-track GP posterior mean

    μf, Pf, μp, Pp, avec, _ = FRs.kalman_ou_filter(y, r, times; τ = τ, σ2 = σ2)
    μs, Ps = FRs.rts_smooth(μf, Pf, μp, Pp, avec)
    @test maximum(abs, μs .- gp_mean) < 1.0e-10
    @test all(Ps .<= Pf .+ 1.0e-12)                 # smoothing never increases variance
end

@testset "Missing observations are predicted through" begin
    rng = MersenneTwister(0x00C0FFEE)
    n = 20
    times = collect(0.0:(n - 1))
    τ, σ2 = 5.0, 1.0
    y = [sin(2π * k / n) for k in 1:n]
    r = fill(0.02, n)
    r[8:12] .= Inf                                  # a gap of missing samples
    yg = copy(y); yg[8:12] .= NaN
    kal = FRs.kalman_ou_filter(yg, r, times; τ = τ, σ2 = σ2)
    @test isfinite(kal[6])                          # loglik sums only observed samples
    μs = FRs.smooth_ou_track(yg, 1.0 ./ r, times; τ = τ, σ2 = σ2)
    @test all(isfinite, μs)                         # gap interpolated to finite values
    # Interpolated gap stays within the range of its finite neighbours.
    @test all(minimum(y) - 0.5 .<= μs .<= maximum(y) + 0.5)
end

@testset "smooth_ou_track denoises an OU-generated track" begin
    rng = MersenneTwister(0x5EED)
    N, dt = 300, 1.0
    times = collect(0:(N - 1)) .* dt
    τ, σ = 20.0, 1.0
    θ = zeros(N); θ[1] = randn(rng) * σ
    for k in 2:N
        a, q = FRs.ou_step(τ, σ^2, dt)
        θ[k] = a * θ[k - 1] + sqrt(q) * randn(rng)
    end
    rn = 0.25
    y = θ .+ sqrt(rn) .* randn(rng, N)
    w = fill(1.0 / rn, N)
    ŷ = FRs.smooth_ou_track(y .- mean(y), w, times; τ = τ, σ2 = σ^2) .+ mean(y)
    @test mean(abs2, ŷ .- θ) < mean(abs2, y .- θ)   # smoother beats the raw track
end

@testset "fit_ou_hypers recovers (τ, σ²) by ML" begin
    N, dt = 600, 1.0
    times = collect(0:(N - 1)) .* dt
    τ_true, σ_true = 20.0, 1.0
    τs = Float64[]; σ2s = Float64[]
    for trial in 1:8
        rng = MersenneTwister(0x1000 + trial)
        θ = zeros(N); θ[1] = randn(rng) * σ_true
        for k in 2:N
            a, q = FRs.ou_step(τ_true, σ_true^2, dt)
            θ[k] = a * θ[k - 1] + sqrt(q) * randn(rng)
        end
        rn = 0.01
        y = θ .+ sqrt(rn) .* randn(rng, N)
        w = fill(1.0 / rn, N)
        τh, σ2h = FRs.fit_ou_hypers(y .- mean(y), w, times; τ0 = 10.0, σ2_0 = 0.5, τ_lo = dt, τ_hi = 100 * N * dt)
        push!(τs, τh); push!(σ2s, σ2h)
    end
    @test 12.0 <= mean(τs) <= 32.0                  # ML τ near truth (finite-sample spread)
    @test 0.6 <= mean(σ2s) <= 1.5                   # ML σ² near truth
end

@testset "Scalar OU path: element type follows the data" begin
    N, dt = 200, 1.0f0
    times = collect(0.0f0:dt:(dt * (N - 1)))
    θ = Float32[0.8 * sin(2π * k / 60) for k in 1:N]
    y = θ .+ 0.1f0 .* Float32[sin(11k) for k in 1:N]
    w = fill(100.0f0, N)

    μf, Pf, μp, Pp, avec, ll = FRs.kalman_ou_filter(y, 1 ./ w, times; τ = 20.0f0, σ2 = 1.0f0)
    @test eltype(μf) === Float32
    @test eltype(avec) === Float32
    @test ll isa Float32
    @test eltype(FRs.rts_smooth(μf, Pf, μp, Pp, avec)[1]) === Float32
    @test eltype(FRs.smooth_ou_track(y, w, times; τ = 20.0f0, σ2 = 1.0f0)) === Float32
    @test FRs._init_track_var(y, w) isa Float32

    τh, σ2h = FRs.fit_ou_hypers(y, w, times; τ0 = 10.0f0, σ2_0 = 0.5f0, τ_lo = 1.0f0, τ_hi = 1.0f4)
    @test τh isa Float32
    @test σ2h isa Float32
    # The Float32 fit lands near the Float64 fit on the same data.
    τ64, σ264 = FRs.fit_ou_hypers(
        Float64.(y), Float64.(w), Float64.(times);
        τ0 = 10.0, σ2_0 = 0.5, τ_lo = 1.0, τ_hi = 1.0e4,
    )
    @test isapprox(τh, τ64; rtol = 0.05)
    @test isapprox(σ2h, σ264; rtol = 0.05)
end

@testset "fit_ou_hypers falls back on too-few samples" begin
    y = [0.1, NaN, 0.2, NaN]
    w = [1.0, 0.0, 1.0, 0.0]
    times = collect(0.0:3.0)
    τ, σ2 = FRs.fit_ou_hypers(y, w, times; τ0 = 7.0, σ2_0 = 0.3, τ_lo = 1.0, τ_hi = 100.0)
    @test τ == 7.0
    @test σ2 == 0.3
end

# Dense joint GP (independent OU per dim) log-likelihood and posterior mean under a
# per-timestep linear design `Hs[k]`, for validating the multivariate recursion.
function _dense_joint_gp(Hs, ys, rs, times, τ, σ2)
    T = length(times)
    n = length(τ)
    N = n * T
    Σ = zeros(N, N)
    for i in 1:n, s in 1:T, t in 1:T
        Σ[(i - 1) * T + s, (i - 1) * T + t] = σ2[i] * exp(-abs(times[s] - times[t]) / τ[i])
    end
    Drows = Vector{Float64}[]
    yvec = Float64[]
    Rdiag = Float64[]
    for k in 1:T, j in eachindex(ys[k])
        d = zeros(N)
        for i in 1:n
            d[(i - 1) * T + k] = Hs[k][j, i]
        end
        push!(Drows, d)
        push!(yvec, ys[k][j])
        push!(Rdiag, rs[k][j])
    end
    D = Matrix(reduce(vcat, transpose.(Drows)))
    C = D * Σ * D' + Diagonal(Rdiag)
    M = length(yvec)
    ll = -0.5 * (logdet(C) + dot(yvec, C \ yvec) + M * log(2π))
    meanZ = Σ * D' * (C \ yvec)
    Xgp = permutedims(reshape(meanZ, T, n))       # [dim, time]
    return ll, Xgp
end

# A closure row for the multivariate filter: `x[a] − x[b]`. Only the geometry
# fields are read there, so `val`/`w`/feeds are inert.
_row(a, b) = FRs._ObsRow(a, b, 1, 1, 0.0, 1.0)

# The dense design row the filter's `(a, b)` geometry stands for, for validating
# against `_dense_joint_gp`.
function _dense_rows(rows, n)
    H = zeros(length(rows), n)
    for (j, r) in enumerate(rows)
        H[j, r.a] += 1.0
        H[j, r.b] -= 1.0
    end
    return H
end

@testset "Multivariate OU Kalman ≡ dense joint GP" begin
    rng = MersenneTwister(0x515C)
    n, T = 3, 8
    τ = [3.0, 5.0, 8.0]
    σ2 = [0.6, 0.9, 0.4]
    times = sort(cumsum(rand(rng, T)) .* 2.0)
    # Differences only — as in the joint adhoc solve, the proper OU prior (not an
    # anchor row) is what leaves the common mode well-posed.
    rows = [_row(1, 2), _row(1, 3), _row(2, 3)]
    rowss = [rows for _ in 1:T]
    Hs = [_dense_rows(rows, n) for _ in 1:T]
    m = length(rows)
    ys = [randn(rng, m) .* 0.4 for _ in 1:T]
    rs = [fill(0.02, m) for _ in 1:T]

    xf, Pf, xp, Pp, avecs, kal_ll = FRs.kalman_ou_mv_filter(rowss, ys, rs, times; τ = τ, σ2 = σ2)
    xs, Ps = FRs.rts_smooth_mv(xf, Pf, xp, Pp, avecs)
    dense_ll, Xgp = _dense_joint_gp(Hs, ys, rs, times, τ, σ2)

    @test isapprox(kal_ll, dense_ll; atol = 1.0e-8)
    @test maximum(abs(xs[i, k] - Xgp[i, k]) for k in 1:T for i in 1:n) < 1.0e-8
end

@testset "Multivariate OU: diffuse (τ=0) dim is temporally independent" begin
    # A τ ≤ 0 dimension carries a diffuse per-step prior and no temporal coupling,
    # so the RTS pass has nothing to propagate back: the smoothed track equals the
    # filtered one exactly. Every dimension here is diffuse AND observed, so the
    # equality is a statement about the transition, not about an idle state.
    n, T = 3, 6
    τ = zeros(3)
    σ2 = [0.5, 0.5, (2π)^2]
    times = collect(0.0:(T - 1))
    rows = [_row(1, 2), _row(1, 3), _row(2, 3)]
    rowss = [rows for _ in 1:T]
    ys = [[0.1 * k, sin(k), cos(k)] for k in 1:T]
    rs = [fill(0.05, 3) for _ in 1:T]
    xf, Pf, xp, Pp, avecs, _ = FRs.kalman_ou_mv_filter(rowss, ys, rs, times; τ = τ, σ2 = σ2)
    xs, _ = FRs.rts_smooth_mv(xf, Pf, xp, Pp, avecs)
    @test all(avecs[i, k] == 0.0 for i in 1:n, k in 2:T)             # diffuse transition
    @test maximum(abs(xs[i, k] - xf[i, k]) for i in 1:n, k in 1:T) < 1.0e-12
    @test any(!iszero, xf)                                           # the states ARE observed
end

@testset "Multivariate OU: element type follows the data" begin
    n, T = 3, 5
    rows = [_row(1, 2), _row(1, 3), _row(2, 3)]
    rowss = [rows for _ in 1:T]
    ys = [Float32[0.1, -0.2, 0.3] for _ in 1:T]
    rs = [fill(0.02f0, 3) for _ in 1:T]
    times = collect(0.0f0:(T - 1))
    xf, Pf, xp, Pp, avecs, ll = FRs.kalman_ou_mv_filter(
        rowss, ys, rs, times; τ = Float32[3, 5, 8], σ2 = Float32[0.6, 0.9, 0.4],
    )
    @test eltype(xf) === Float32
    @test eltype(Pf) === Float32
    @test ll isa Float32
    xs, _ = FRs.rts_smooth_mv(xf, Pf, xp, Pp, avecs)
    @test eltype(xs) === Float32
end

@testset "Multivariate OU: a row outside the state is an error" begin
    rows = [_row(1, 4)]
    @test_throws "outside the state 1:3" FRs.kalman_ou_mv_filter(
        [rows], [[0.1]], [[0.02]], [0.0]; τ = [3.0, 5.0, 8.0], σ2 = [0.6, 0.9, 0.4],
    )
end
