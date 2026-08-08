# Per-observable frequency-shape specs (FreeShape / PolynomialShape /
# WhittakerShape / ARShape) and their per-track fit.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using LinearAlgebra
using Statistics: mean, median

const FRsh = Gustavo.Fringe

# A 32-segment spw at ~228 GHz with 2 MHz segments — the frequency coordinate
# `fit_track` is written against.
_spw_freqs(n = 32; f0 = 2.28e11, df = 2.0e6) = f0 .+ df .* (0:(n - 1))

# Dense Matérn-1/2 (OU) covariance over an arbitrary sample coordinate.
_ou_cov(x, ν, σ2) = [σ2 * exp(-abs(x[i] - x[j]) / ν) for i in eachindex(x), j in eachindex(x)]

@testset "FreeShape keeps the data and estimates no gap" begin
    x = _spw_freqs(5)
    y = [1.0, 2.0, NaN, 4.0, 5.0]
    w = [1.0, 1.0, 0.0, 0.0, 2.0]
    # Segment 3 has no value, segment 4 no weight — neither is estimable without
    # a shape assumption.
    @test isequal(FRsh.fit_track(FRsh.FreeShape(), y, w, x), [1.0, 2.0, NaN, NaN, 5.0])
end

@testset "PolynomialShape recovers a polynomial across gaps" begin
    x = _spw_freqs(32)
    u = (x .- (first(x) + last(x)) / 2) ./ ((last(x) - first(x)) / 2)
    ytrue = @. 0.3 - 0.7 * u + 0.2 * u^2 - 0.05 * u^3
    y = collect(ytrue)
    w = ones(length(x))
    for k in (5, 6, 7, 21)                                  # unobserved segments
        y[k] = NaN
        w[k] = 0.0
    end

    fit3 = FRsh.fit_track(FRsh.PolynomialShape(3), y, w, x)
    @test fit3 ≈ ytrue atol = 1.0e-6                        # gaps included

    # A degree the data cannot support is lowered to what it can: two usable
    # segments admit a line, and the fit passes through both.
    wsparse = zeros(length(x))
    wsparse[[3, 20]] .= 1.0
    ysparse = fill(NaN, length(x))
    ysparse[[3, 20]] .= (ytrue[3], ytrue[20])
    line = FRsh.fit_track(FRsh.PolynomialShape(4), ysparse, wsparse, x)
    @test all(isfinite, line)
    @test line[[3, 20]] ≈ ytrue[[3, 20]] atol = 1.0e-6
    @test maximum(abs, diff(diff(line))) < 1.0e-9

    # One usable segment identifies only a constant — never a slope extrapolated
    # across the band.
    wone = zeros(length(x))
    wone[7] = 1.0
    yone = fill(NaN, length(x))
    yone[7] = 0.25
    flat = FRsh.fit_track(FRsh.PolynomialShape(4), yone, wone, x)
    @test all(isfinite, flat)
    @test maximum(abs, diff(flat)) < 1.0e-9
    @test flat[7] ≈ 0.25 rtol = 1.0e-4

    # Below its own degree the fit is the weighted least-squares projection.
    obs = findall(>(0), w)
    B = [u[k]^j for k in obs, j in 0:1]
    ref = B * (B \ ytrue[obs])
    @test FRsh.fit_track(FRsh.PolynomialShape(1), y, w, x)[obs] ≈ ref rtol = 1.0e-4

    @test_throws "degree must be ≥ 1" FRsh.PolynomialShape(0)
end

@testset "WhittakerShape smooths, interpolates, and spans Free ↔ a line" begin
    rng = MersenneTwister(0x5A1AD)
    x = _spw_freqs(40)
    u = (x .- first(x)) ./ (last(x) - first(x))
    ytrue = 0.5 .* sinpi.(2 .* u)
    σ = 0.05
    y = ytrue .+ σ .* randn(rng, length(x))
    w = fill(inv(σ^2), length(x))
    for k in (11, 12, 13)
        y[k] = NaN
        w[k] = 0.0
    end
    obs = findall(>(0), w)

    smoothed = FRsh.fit_track(FRsh.WhittakerShape(1.0), y, w, x)
    @test all(isfinite, smoothed)                           # gaps interpolated
    @test mean(abs2, smoothed .- ytrue) < mean(abs2, y[obs] .- ytrue[obs])

    # λ = 0 leaves the data untouched; a huge λ leaves only the second-difference
    # null space, a straight line.
    @test isequal(
        FRsh.fit_track(FRsh.WhittakerShape(0.0), y, w, x),
        FRsh.fit_track(FRsh.FreeShape(), y, w, x),
    )
    stiff = FRsh.fit_track(FRsh.WhittakerShape(1.0e8), y, w, x)
    @test maximum(abs, diff(diff(stiff))) < 1.0e-4

    # Fewer than three segments span no second difference.
    @test isequal(
        FRsh.fit_track(FRsh.WhittakerShape(1.0), [0.5, NaN], [1.0, 0.0], x[1:2]),
        [0.5, NaN],
    )

    @test_throws "lambda must be ≥ 0" FRsh.WhittakerShape(-1.0)
end

@testset "ARShape is the OU posterior mean over the frequency coordinate" begin
    rng = MersenneTwister(0x0A5E)
    n = 24
    x = 2.28e11 .+ cumsum(rand(rng, n)) .* 2.0e6            # irregular segments
    ν = 8.0e6
    σ = 0.3
    r = 0.02^2
    y = 0.4 .+ σ .* randn(rng, n)
    w = fill(inv(r), n)

    fitted = FRsh.fit_track(FRsh.ARShape(ν; sigma = σ, fit_hypers = false), y, w, x)

    # Exactly the dense GP posterior mean about the (weighted) track mean.
    m = mean(y)
    C = _ou_cov(x, ν, σ^2) + r * I
    ref = m .+ _ou_cov(x, ν, σ^2) * (C \ (y .- m))
    @test maximum(abs, fitted .- ref) < 1.0e-10

    # Only coordinate DIFFERENCES enter, so rescaling `x` and `bandwidth`
    # together — Hz to MHz — is inert.
    rescaled = FRsh.fit_track(
        FRsh.ARShape(ν / 1.0e6; sigma = σ, fit_hypers = false), y, w, (x .- first(x)) ./ 1.0e6,
    )
    @test maximum(abs, rescaled .- fitted) < 1.0e-10
end

@testset "ARShape interpolates gaps and fits its hypers" begin
    rng = MersenneTwister(0xC0FFEE)
    n = 64
    x = _spw_freqs(n)
    ν = 12.0e6
    σ = 0.4
    # An OU draw along frequency: the process ARShape assumes.
    track = zeros(n)
    track[1] = σ * randn(rng)
    for k in 2:n
        a = exp(-abs(x[k] - x[k - 1]) / ν)
        track[k] = a * track[k - 1] + sqrt(σ^2 * (1 - a^2)) * randn(rng)
    end
    rnoise = 0.1^2
    y = track .+ sqrt(rnoise) .* randn(rng, n)
    w = fill(inv(rnoise), n)
    gaps = 25:32
    y[gaps] .= NaN
    w[gaps] .= 0.0

    filled = FRsh.fit_track(FRsh.ARShape(ν), y, w, x)
    @test all(isfinite, filled)
    @test mean(abs2, filled[gaps] .- track[gaps]) < σ^2      # better than the prior mean

    # Seeded a hundred bandwidths away and with a wildly loose prior variance,
    # the marginal-likelihood fit still beats holding those seeds fixed.
    seed = FRsh.ARShape(100 * ν; sigma = 10.0, fit_hypers = false)
    fitted = FRsh.ARShape(100 * ν; sigma = 10.0, fit_hypers = true)
    mse(spec) = mean(abs2, FRsh.fit_track(spec, y, w, x) .- track)
    @test mse(fitted) < mse(seed)

    @test_throws "bandwidth must be > 0" FRsh.ARShape(-1.0)
    @test_throws "sigma must be > 0" FRsh.ARShape(1.0e6; sigma = 0.0)
end

@testset "shape specs: the shared track contract" begin
    x = _spw_freqs(6)
    specs = (
        FRsh.FreeShape(), FRsh.PolynomialShape(2), FRsh.WhittakerShape(1.0),
        FRsh.ARShape(5.0e6),
    )
    for spec in specs
        # A track with no usable segment is estimable nowhere.
        empty = FRsh.fit_track(spec, fill(NaN, 6), zeros(6), x)
        @test length(empty) == 6
        @test all(isnan, empty)

        # Value, weight and coordinate are one per segment.
        @test_throws DimensionMismatch FRsh.fit_track(spec, ones(6), ones(5), x)
        @test_throws DimensionMismatch FRsh.fit_track(spec, ones(6), ones(6), x[1:5])
    end
end
