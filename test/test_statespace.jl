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

@testset "Multivariate OU Kalman ≡ dense joint GP" begin
    rng = MersenneTwister(0x515C)
    n, T = 3, 8
    τ = [3.0, 5.0, 8.0]
    σ2 = [0.6, 0.9, 0.4]
    times = sort(cumsum(rand(rng, T)) .* 2.0)
    pairs = [(1, 2), (1, 3), (2, 3)]
    Hs = Vector{Matrix{Float64}}(undef, T)
    for k in 1:T
        rows = Vector{Float64}[]
        for (a, b) in pairs
            h = zeros(n); h[a] = 1.0; h[b] = -1.0
            push!(rows, h)
        end
        h = zeros(n); h[1] = 1.0; push!(rows, h)   # one anchor row keeps the GP well-posed
        Hs[k] = Matrix(reduce(vcat, transpose.(rows)))
    end
    m = size(Hs[1], 1)
    ys = [randn(rng, m) .* 0.4 for _ in 1:T]
    rs = [fill(0.02, m) for _ in 1:T]

    xf, Pf, xp, Pp, avecs, kal_ll = FRs.kalman_ou_mv_filter(Hs, ys, rs, times; τ = τ, σ2 = σ2)
    xs, Ps = FRs.rts_smooth_mv(xf, Pf, xp, Pp, avecs)
    dense_ll, Xgp = _dense_joint_gp(Hs, ys, rs, times, τ, σ2)

    @test isapprox(kal_ll, dense_ll; atol = 1.0e-8)
    @test maximum(abs(xs[k][i] - Xgp[i, k]) for k in 1:T for i in 1:n) < 1.0e-8
end

@testset "Multivariate OU: diffuse (τ=0) dim is temporally independent" begin
    # A τ=0 dimension carries a diffuse per-step prior and no temporal coupling —
    # its RTS-smoothed value equals its filtered value (nothing to propagate back).
    n, T = 2, 6
    τ = [4.0, 0.0]             # dim 2 diffuse (the augmented χ pattern)
    σ2 = [0.5, (2π)^2]
    times = collect(0.0:(T - 1))
    Hs = [Matrix{Float64}([1.0 0.0; 0.0 1.0]) for _ in 1:T]   # observe both dims directly
    ys = [[0.1 * k, sin(k)] for k in 1:T]
    rs = [fill(0.05, 2) for _ in 1:T]
    xf, Pf, xp, Pp, avecs, _ = FRs.kalman_ou_mv_filter(Hs, ys, rs, times; τ = τ, σ2 = σ2)
    xs, _ = FRs.rts_smooth_mv(xf, Pf, xp, Pp, avecs)
    @test all(avecs[k][2] == 0.0 for k in 2:T)                # diffuse transition
    @test maximum(abs(xs[k][2] - xf[k][2]) for k in 1:T) < 1.0e-10   # not smoothed across time
end
