# Per-observable frequency-shape specs (FreeShape / PolynomialShape /
# WhittakerShape / ARShape) and their per-track fit.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using LinearAlgebra
using Statistics: mean, median, std

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

# A group of `nband` spws, each `nchan` segments wide, drawn from one shared OU
# process per band plus independent noise. Returns `(ys, ws, xs, truths)` in
# `fit_track_group` order, each band centred so a fit's own level is free.
function _spw_group(rng; nband = 8, nchan = 64, df = 5.0e5, span = 3.2e7, ν = 8.0e6, rms = 0.15, noise = 0.4)
    ys, ws, xs, truths = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]
    for b in 1:nband
        x = (b - 1) * span .+ df .* (0:(nchan - 1))
        a = exp(-df / ν)
        t = zeros(nchan)
        t[1] = randn(rng)
        for k in 2:nchan
            t[k] = a * t[k - 1] + sqrt(1 - a^2) * randn(rng)
        end
        t .*= rms / std(t)
        t .-= mean(t)
        push!(truths, t)
        push!(ys, t .+ noise .* randn(rng, nchan))
        push!(ws, fill(1 / noise^2, nchan))
        push!(xs, collect(x))
    end
    return ys, ws, xs, truths
end

@testset "fit_track_group: default is the per-track fit" begin
    rng = MersenneTwister(4242)
    ys, ws, xs, _ = _spw_group(rng; nband = 3, nchan = 16)
    # Every spec whose shape parameters are SUPPLIED estimates nothing across
    # members, so grouping them must not change a single fitted value.
    for spec in (FRsh.FreeShape(), FRsh.PolynomialShape(2), FRsh.WhittakerShape(1.0))
        grouped = FRsh.fit_track_group(spec, ys, ws, xs)
        for b in eachindex(ys)
            @test grouped[b] == FRsh.fit_track(spec, ys[b], ws[b], xs[b])
        end
    end
    # ARShape with the hypers held fixed has nothing to pool either.
    fixed = FRsh.ARShape(1.6e7; fit_hypers = false)
    for (b, g) in enumerate(FRsh.fit_track_group(fixed, ys, ws, xs))
        @test g == FRsh.fit_track(fixed, ys[b], ws[b], xs[b])
    end
end

@testset "fit_track_group: ARShape pools the hypers, not the levels" begin
    rng = MersenneTwister(90210)
    nband, nchan = 6, 48
    xs = [collect((b - 1) * 3.2e7 .+ 5.0e5 .* (0:(nchan - 1))) for b in 1:nband]
    ws = [fill(1.0e4, nchan) for _ in 1:nband]
    levels = [2.0 * b for b in 1:nband]
    ys = [fill(levels[b], nchan) .+ 0.01 .* randn(rng, nchan) for b in 1:nband]

    fitted = FRsh.fit_track_group(FRsh.ARShape(1.6e7), ys, ws, xs)
    # Each member keeps its own free level: a shared shape must not pull the bands
    # toward a common mean, which is what makes a real spw discontinuity
    # representable.
    for b in 1:nband
        @test mean(fitted[b]) ≈ levels[b] atol = 0.02
    end
end

@testset "ARShape: pooling beats per-spw fitting on starved windows" begin
    rng = MersenneTwister(31337)
    spec = FRsh.ARShape(1.6e7)
    eind = Float64[]
    epool = Float64[]
    flat_ind = Float64[]
    flat_pool = Float64[]
    for _ in 1:6
        ys, ws, xs, truths = _spw_group(rng; nband = 16, nchan = 64, noise = 0.4)
        indep = [FRsh.fit_track(spec, ys[b], ws[b], xs[b]) for b in eachindex(ys)]
        pooled = FRsh.fit_track_group(spec, ys, ws, xs)
        err(f, t) = sqrt(mean(abs2, (f .- mean(f)) .- t))
        push!(eind, mean(err(indep[b], truths[b]) for b in eachindex(ys)))
        push!(epool, mean(err(pooled[b], truths[b]) for b in eachindex(ys)))
        isflat(f) = (maximum(f) - minimum(f)) < 0.01
        push!(flat_ind, mean(isflat, indep))
        push!(flat_pool, mean(isflat, pooled))
    end
    # One window of 64 noisy channels frequently cannot separate the ripple from
    # the noise, and the per-window ML then returns that window's mean. Estimating
    # the pair from every window at once both recovers more of the truth and stops
    # the outcome flipping between neighbouring windows of identical quality.
    @test median(epool) < median(eind)
    @test median(flat_pool) < median(flat_ind)
end
