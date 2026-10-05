# Fitting a bandpass track under a component's prior, piece by piece.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using LinearAlgebra
using Statistics: mean, median, std
using Distributions: LogNormal, Normal, MvNormal

const FRpf = Gustavo.Fring
const CALpf = Gustavo.Calibration
const Freq = Gustavo.Frequency

# The zero-mean OU posterior covariance pieces by dense linear algebra.
_ou_cov(x, τ, σ2) = [σ2 * exp(-abs(a - b) / τ) for a in x, b in x]
_dense_ou_map(y, w, x, τ, σ2) = (K = _ou_cov(x, τ, σ2); K * ((K + Diagonal(inv.(w))) \ y))

# The random-walk MAP by dense linear algebra, as the posterior mean of a
# Gaussian process: an integrated Brownian motion from the first coordinate plus
# a polynomial of degree `order − 1` whose coefficients have variance `κ`, which
# tends to the flat start as `κ` grows. Orders 1 and 2.
function _dense_random_walk(y, w, x, order, σ; κ = 1.0e8)
    u = (x .- x[1]) ./ (x[end] - x[1])
    L = x[end] - x[1]
    brownian(s, t) = σ^2 * L * min(s, t)
    integrated(s, t) = (m = min(s, t); σ^2 * L^3 * (m^3 / 3 + abs(t - s) * m^2 / 2))
    k = order == 1 ? brownian : integrated
    K = [κ * sum((a * b)^d for d in 0:(order - 1)) + k(a, b) for a in u, b in u]
    o = [i for i in eachindex(y) if isfinite(y[i]) && w[i] > 0]
    return K[:, o] * ((K[o, o] + Diagonal(inv.(w[o]))) \ y[o])
end

@testset "random-walk prior: integrated Brownian motion in the coordinate" begin
    rng = Random.Xoshiro(3)
    n = 40
    x = cumsum(1.0e6 .* (0.5 .+ rand(rng, n)))      # uneven spacing, Hz
    y = sin.((x .- x[1]) ./ 1.0e7) .+ 0.05 .* randn(rng, n)
    w = fill(400.0, n)
    y[10:14] .= NaN
    σ = Dict(1 => 1.0e-4, 2 => 1.0e-11)
    for order in 1:2
        @test FRpf._random_walk_track(y, w, x, order, σ[order]) ≈
            _dense_random_walk(y, w, x, order, σ[order]) rtol = 1.0e-5
    end

    # Order 1 exactly: independent steps `N(0, σ²Δx)`.
    D = diff(Matrix(1.0I, n, n); dims = 1)
    wk = ifelse.(isfinite.(y), w, 0.0)
    exact = (Diagonal(wk) + D' * Diagonal(inv.(σ[1]^2 .* diff(x))) * D) \ (wk .* ifelse.(isfinite.(y), y, 0.0))
    @test FRpf._random_walk_track(y, w, x, 1, σ[1]) ≈ exact rtol = 1.0e-10

    # The prior is defined along the coordinate, not by sample order: reversed
    # coordinates and a shifted origin give the same fit.
    @test FRpf._random_walk_track(reverse(y), reverse(w), reverse(x), 2, σ[2]) ≈
        reverse(FRpf._random_walk_track(y, w, x, 2, σ[2])) rtol = 1.0e-8
    @test FRpf._random_walk_track(y, w, x .+ 5.0e9, 2, σ[2]) ≈ FRpf._random_walk_track(y, w, x, 2, σ[2]) rtol = 1.0e-8
    @test_throws "strictly monotone coordinates" FRpf._random_walk_track(y, w, [x[2]; x[1]; x[3:end]], 1, σ[1])

    @test eltype(FRpf._random_walk_track(Float32.(y), Float32.(w), x, 2, σ[2])) === Float32
    @test FRpf._random_walk_track(Float32.(y), Float32.(w), x, 2, σ[2]) ≈
        FRpf._random_walk_track(y, w, x, 2, σ[2]) rtol = 1.0e-5
end

@testset "random-walk prior: restricted likelihood" begin
    rng = Random.Xoshiro(5)
    n = 30
    x = cumsum(1.0e6 .* (0.5 .+ rand(rng, n)))
    y = sin.((x .- x[1]) ./ 1.0e7) .+ 0.05 .* randn(rng, n)
    w = 300.0 .+ 200.0 .* rand(rng, n)
    y[7:9] .= NaN
    σ = Dict(1 => 1.0e-4, 2 => 1.0e-11)
    o = findall(isfinite, y)
    # Dense REML: the walk from a zero state at x[1], plus the flat starting state
    # as regression on `(x − x₁)ʲ/j!`, integrated out under a flat prior.
    for m in 1:2
        t = x[o] .- x[1]
        k(s, u) = m == 1 ? σ[m]^2 * min(s, u) : (v = min(s, u); σ[m]^2 * (v^3 / 3 + abs(u - s) * v^2 / 2))
        Σ = [k(a, b) for a in t, b in t] + Diagonal(inv.(w[o]))
        X = [tj^j / factorial(j) for tj in t, j in 0:(m - 1)]
        Σi = inv(Σ)
        XΣX = X' * Σi * X
        Π = Σi - Σi * X * (XΣX \ (X' * Σi))
        reml = -((length(o) - m) * log(2π) + logdet(Σ) + logdet(XΣX) + y[o]' * Π * y[o]) / 2
        @test FRpf._random_walk_loglik(y, w, x, m, σ[m]) ≈ reml rtol = 1.0e-8
    end

    # The filter needs `m` usable samples to determine the flat start.
    @test_throws "do not determine the flat-start state" FRpf.kalman_filter(
        FRpf.RandomWalkModel{2}(1.0), [NaN, 0.3, NaN], fill(0.01, 3), [0.0, 1.0, 2.0],
    )
end

# A walk of order `m` sampled at `x`, simulated from its exact transition, plus noise.
function _simulate_walk(rng, x, m, σ, noise)
    s = zeros(m)
    f = zeros(length(x))
    for k in eachindex(x)
        if k > 1
            A, Q = FRpf.transition(FRpf.RandomWalkModel{m}(σ^2), x[k] - x[k - 1], Float64)
            s = A * s + cholesky(Symmetric(Matrix(Q))).L * randn(rng, m)
        end
        f[k] = s[1]
    end
    return f .+ noise .* randn(rng, length(x))
end

@testset "random-walk σ by type-II MAP" begin
    rng = Random.Xoshiro(11)
    x = [cumsum(0.5 .+ rand(rng, 150)) for _ in 1:4]
    ws = [fill(1.0e4, 150) for _ in 1:4]
    for (m, σ) in ((1, 0.05), (2, 0.002))
        ys = [_simulate_walk(rng, xi, m, σ, 0.01) for xi in x]
        est = FRpf._estimate_hypers(CALpf.RandomWalkPrior(; order = m, σ = LogNormal(log(0.1), 3.0)), ys, ws, x)
        @test est isa CALpf.RandomWalkPrior && est.order == m && CALpf.is_fixed_hyper(est.σ)
        @test 0.7 < est.σ / σ < 1.4
        # A concentrated hyperprior pins σ whatever the data say.
        @test FRpf._estimate_hypers(CALpf.RandomWalkPrior(; order = m, σ = LogNormal(log(0.3), 0.001)), ys, ws, x).σ ≈ 0.3 rtol = 0.01
        # A block too short to inform the walk does not enter the estimate.
        short = fill(NaN, 150)
        short[5] = 0.1
        pooled = FRpf._estimate_hypers(CALpf.RandomWalkPrior(; order = 2, σ = LogNormal(log(0.1), 3.0)), [ys; [short]], [ws; ws[1:1]], [x; x[1:1]])
        @test pooled.σ ≈ FRpf._estimate_hypers(CALpf.RandomWalkPrior(; order = 2, σ = LogNormal(log(0.1), 3.0)), ys, ws, x).σ
    end

    y = _simulate_walk(rng, x[1], 2, 0.002, 0.01)
    hyper = CALpf.RandomWalkPrior(; order = 2, σ = LogNormal(log(0.1), 3.0))
    fixed = CALpf.RandomWalkPrior(; order = 2, σ = 0.01)
    @test FRpf._estimate_hypers(fixed, [y], [ws[1]], [x[1]]) === fixed
    @test FRpf._estimate_hypers(hyper, [Float32.(y)], [Float32.(ws[1])], [x[1]]).σ isa Float32
    @test_throws "takes no level groups" FRpf._estimate_hypers(hyper, [y], [ws[1]], [x[1]]; level = [1])
    @test_throws "no block has 2 usable segments" FRpf._estimate_hypers(hyper, [fill(NaN, 150)], [ws[1]], [x[1]])
    @test_throws "RandomWalkPrior σ must be fixed" FRpf._estimate_map(hyper, y, ws[1], x[1])
end

# A walk of order `m` from a Gaussian start `N(μ, Σ)` on `(f, f′, …)` at `x[1]`, as a
# Gaussian process: its mean and covariance at `x`.
function _walk_gp(x, m, σ, μ, Σ)
    t = x .- x[1]
    k0(s, u) = m == 1 ? σ^2 * min(s, u) : (v = min(s, u); σ^2 * (v^3 / 3 + abs(u - s) * v^2 / 2))
    X = [tj^j / factorial(j) for tj in t, j in 0:(m - 1)]
    return X * μ, X * Σ * X' + [k0(a, b) for a in t, b in t]
end

@testset "random-walk prior with an init" begin
    rng = Random.Xoshiro(21)
    n = 30
    x = cumsum(1.0e6 .* (0.5 .+ rand(rng, n)))
    y = 0.3 .+ sin.((x .- x[1]) ./ 1.0e7) .+ 0.05 .* randn(rng, n)
    y[6:8] .= NaN
    w = 300.0 .+ 200.0 .* rand(rng, n)
    o = findall(isfinite, y)
    cases = (
        (1, 1.0e-4, Normal(0.2, 0.5), [0.2], fill(0.25, 1, 1)),
        (2, 1.0e-11, MvNormal([0.2, 1.0e-8], Diagonal([0.25, 1.0e-14])), [0.2, 1.0e-8], Diagonal([0.25, 1.0e-14])),
    )
    for (m, σ, init, μ, Σ) in cases
        prior = CALpf.RandomWalkPrior(; order = m, σ, init)
        mvec, K = _walk_gp(x, m, σ, μ, Σ)
        C = K[o, o] + Diagonal(inv.(w[o]))
        d = y[o] - mvec[o]
        @test FRpf._estimate_map(prior, y, w, x) ≈ mvec + K[:, o] * (C \ d) rtol = 1.0e-6
        @test FRpf._random_walk_loglik(y, w, x, m, σ; init) ≈ -(logdet(C) + d' * (C \ d) + length(o) * log(2π)) / 2 rtol = 1.0e-8
        # The level beside the walk is its GLS estimate under the walk's covariance.
        L = only(FRpf._estimate_levels(prior, [y], [w], [x], [1], 1))
        @test L ≈ sum(C \ d) / sum(C \ ones(length(o))) rtol = 1.0e-6
        @test all(isnan, FRpf._estimate_map(prior, fill(NaN, n), w, x))
        # A σ hyperprior resolves with the level integrated out and keeps the init.
        est = FRpf._estimate_hypers(CALpf.RandomWalkPrior(; order = m, σ = LogNormal(log(σ), 1.0), init), [y], [w], [x]; level = [1])
        @test CALpf.is_fixed_hyper(est.σ) && est.init == init
    end

    flat = CALpf.RandomWalkPrior(; order = 2, σ = 1.0e-11)
    @test !FRpf._proper_prior(flat) && FRpf._proper_prior(CALpf.RandomWalkPrior(; σ = 0.1, init = Normal(0, 1)))
    @test_throws "a level is not identifiable" FRpf._estimate_levels(flat, [y], [w], [x], [1], 1)
    @test_throws "needs an init of length 2" CALpf.RandomWalkPrior(; order = 2, σ = 1.0, init = MvNormal(zeros(3), Diagonal(ones(3))))
    @test_throws "got a Normal" CALpf.RandomWalkPrior(; order = 2, σ = 1.0, init = Normal(0, 1))
    @test_throws "must be `nothing`, a Normal or an MvNormal" CALpf.RandomWalkPrior(; σ = 1.0, init = 0.3)
end

@testset "MAP of one block" begin
    rng = Random.Xoshiro(4)
    n = 16
    x = collect(range(1.0e9, 1.03e9; length = n))
    y = 0.3 .* sin.(range(0, 2; length = n)) .+ 0.02 .* randn(rng, n)
    w = fill(2500.0, n)
    rw = CALpf.RandomWalkPrior(; order = 2, σ = 1.0e-11)
    @test FRpf._estimate_map(rw, y, w, x) ≈ FRpf._random_walk_track(y, w, x, 2, 1.0e-11)

    # No prior returns the measured values, `NaN` where there are none.
    y2 = copy(y)
    y2[5] = NaN
    @test isequal(FRpf._estimate_map(nothing, y2, w, x), ifelse.(isfinite.(y2), y2, NaN))

    # Fewer usable channels than the walk's order: the measurement is kept.
    y3 = fill(NaN, n)
    y3[7] = 0.4
    @test isequal(FRpf._estimate_map(rw, y3, w, x), y3)
    @test all(isnan, FRpf._estimate_map(rw, fill(NaN, n), w, x))

    # OU is zero-mean: an offset is shrunk toward zero, not removed.
    ou = CALpf.OUPrior(; scale = 1.0e7, σ = 0.2)
    yl = y .+ 3.0
    @test FRpf._estimate_map(ou, yl, w, x) ≈ FRpf.smooth_ou_track(yl, w, x; τ = 1.0e7, σ2 = 0.04)
    @test FRpf._estimate_map(ou, yl, w, x) ≈ _dense_ou_map(yl, w, x, 1.0e7, 0.04)
    # Hyperparameters must be resolved first.
    @test_throws "OUPrior hyperparameters must be fixed" FRpf._estimate_map(
        CALpf.OUPrior(; scale = LogNormal(16.0, 1.0), σ = 0.2), y, w, x,
    )
end

@testset "fitting a block in place" begin
    rng = Random.Xoshiro(12)
    n = 30
    x = collect(range(1.0e9, 1.06e9; length = n))
    y = 0.2 .* sin.(range(0, 3; length = n)) .+ 0.02 .* randn(rng, n)
    y[4] = NaN
    w = fill(2500.0, n)
    for prior in (nothing, CALpf.RandomWalkPrior(; order = 2, σ = 1.0e-11), CALpf.OUPrior(; scale = 1.0e7, σ = 0.2))
        expected = FRpf._estimate_map(prior, y, w, x)
        # `out` may be the block itself, or a view into a larger array.
        yy = copy(y)
        @test isequal(FRpf._estimate_map!(yy, prior, yy, w, x), expected)
        big = fill(-1.0, 2n)
        FRpf._estimate_map!(view(big, 2:2:(2n)), prior, y, w, x)
        @test isequal(big[2:2:end], expected) && all(==(-1.0), big[1:2:end])
        @inferred FRpf._estimate_map(prior, y, w, x)
    end
    @inferred FRpf._random_walk_loglik(y, w, x, 2, 1.0e-11)
    @inferred FRpf._estimate_hypers(CALpf.RandomWalkPrior(; order = 2, σ = LogNormal(-25.0, 1.0)), [y], [w], [x])
    @inferred FRpf._estimate_hypers(CALpf.OUPrior(; scale = LogNormal(16.0, 1.0), σ = LogNormal(-1.0, 1.0)), [y], [w], [x])
    @inferred FRpf._estimate_levels(CALpf.OUPrior(; scale = 1.0e7, σ = 0.2), [y], [w], [x], [1], 1)
end

@testset "hyperparameters estimated over the blocks the caller pools" begin
    rng = Random.Xoshiro(6)
    xs = [collect(range(1.0e9 + 1.0e8 * (i - 1), 1.0e9 + 1.0e8 * (i - 1) + 6.3e7; length = 64)) for i in 1:3]
    ys = [0.2 .* sin.(x ./ 5.0e6) .+ 0.05 .* randn(rng, 64) .+ i for (i, x) in enumerate(xs)]
    ws = [fill(400.0, 64) for _ in 1:3]

    # Priors without hyperpriors come back unchanged.
    rw = CALpf.RandomWalkPrior(; order = 2, σ = 0.01)
    @test FRpf._estimate_hypers(rw, ys, ws, xs) === rw
    @test isnothing(FRpf._estimate_hypers(nothing, ys, ws, xs))
    fixed = CALpf.OUPrior(; scale = 1.0e7, σ = 0.2)
    @test FRpf._estimate_hypers(fixed, ys, ws, xs) === fixed

    # Hyperpriors resolve to one fixed (scale, σ) for all the blocks: the
    # type-II MAP over the blocks, each block's level integrated out.
    ou = CALpf.OUPrior(; scale = LogNormal(log(1.0e7), 1.0), σ = LogNormal(log(0.2), 1.0))
    level = [1, 2, 3]
    est = FRpf._estimate_hypers(ou, ys, ws, xs; level)
    @test CALpf.is_fixed_hyper(est.scale) && CALpf.is_fixed_hyper(est.σ)
    τ_lo, τ_hi = FRpf._group_ou_tau_bounds(xs)
    τ, σ2 = FRpf._map_ou_hypers(
        ys, ws, xs, ou.scale, ou.σ;
        τ_lo, τ_hi, σ2_seed = FRpf._init_track_var(reduce(vcat, ys), reduce(vcat, ws)), levels = level,
    )
    @test est.scale ≈ τ && est.σ ≈ sqrt(σ2)
    # Zero-mean, the offsets 1, 2, 3 read as signal and inflate σ.
    @test FRpf._estimate_hypers(ou, ys, ws, xs).σ > 2 * est.σ
    # The resolved prior follows the ripple rather than flattening it.
    L = FRpf._estimate_levels(est, ys, ws, xs, level, 3)
    @test L ≈ [1, 2, 3] atol = 0.15
    @test std(FRpf._estimate_map(est, ys[1] .- L[1], ws[1], xs[1])) > 0.1

    # A block without data does not enter the estimate; with no data at all
    # there is nothing to estimate from.
    empty = fill(NaN, 64)
    @test FRpf._estimate_hypers(ou, [ys; [empty]], [ws; [ws[1]]], [xs; [xs[1]]]; level = [level; 1]) == est
    @test_throws "no block carries data" FRpf._estimate_hypers(ou, [empty], ws[1:1], xs[1:1])
end

@testset "a level shared by blocks under a zero-mean OU" begin
    rng = Random.Xoshiro(8)
    τ, σ2 = 2.0e7, 0.03
    xs = [collect(range(1.0e9, 1.06e9; length = 12)), collect(range(1.2e9, 1.26e9; length = 9))]
    ws = [100.0 .+ 50.0 .* rand(rng, length(x)) for x in xs]
    ys = [0.7 .+ 0.2 .* sin.(x ./ 1.0e7) .+ randn(rng, length(x)) ./ sqrt.(w) for (x, w) in zip(xs, ws)]
    ys[2][4] = NaN

    # Dense GLS: the two blocks are independent, each covariance K + W⁻¹.
    usable(i) = isfinite.(ys[i])
    Σinv(i) = inv(_ou_cov(xs[i][usable(i)], τ, σ2) + Diagonal(inv.(ws[i][usable(i)])))
    b = sum(sum(Σinv(i) * ys[i][usable(i)]) for i in 1:2)
    c = sum(sum(Σinv(i)) for i in 1:2)
    prior = CALpf.OUPrior(; scale = τ, σ = sqrt(σ2))
    L = FRpf._estimate_levels(prior, ys, ws, xs, [1, 1], 1)
    @test only(L) ≈ b / c rtol = 1.0e-10
    # A level no block holds has no estimate.
    @test isnan(FRpf._estimate_levels(prior, ys, ws, xs, [1, 1], 2)[2])

    # REML: the likelihood with the level integrated out under a flat prior.
    rs = [inv.(w) for w in ws]
    reml = sum(1:2) do i
        u = usable(i)
        r = ys[i][u] .- only(L)
        S = _ou_cov(xs[i][u], τ, σ2) + Diagonal(inv.(ws[i][u]))
        -(logdet(S) + r' * (S \ r) + count(u) * log(2π)) / 2
    end - log(c) / 2
    @test FRpf._ou_loglik(ys, rs, xs, [1, 1]; τ, σ2) ≈ reml rtol = 1.0e-10

    # The shape given the level is the zero-mean MAP of the remainder.
    y1 = ys[1] .- only(L)
    @test FRpf._estimate_map(prior, y1, ws[1], xs[1]) ≈ _dense_ou_map(y1, ws[1], xs[1], τ, σ2) rtol = 1.0e-8
end
