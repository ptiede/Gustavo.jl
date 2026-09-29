# Fitting a bandpass track under a component's prior, piece by piece.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using LinearAlgebra
using Statistics: mean, median, std
using Distributions: LogNormal

const FRpf = Gustavo.Fring
const CALpf = Gustavo.Calibration
const Freq = Gustavo.UVData.Frequency

# The random-walk MAP by dense linear algebra.
function _dense_random_walk(y, w, order, σ)
    n = length(y)
    D = Matrix(1.0I, n, n)
    for _ in 1:order
        D = diff(D; dims = 1)
    end
    usable = [isfinite(y[k]) && w[k] > 0 for k in eachindex(y)]
    W = Diagonal(ifelse.(usable, w, 0.0))
    return (W + D'D / σ^2) \ (W * ifelse.(usable, y, 0.0))
end

@testset "random-walk prior: banded MAP" begin
    rng = Random.Xoshiro(3)
    n = 40
    y = sin.(range(0, 3; length = n)) .+ 0.05 .* randn(rng, n)
    w = fill(400.0, n)
    y[10:14] .= NaN
    for order in 1:3
        @test FRpf._random_walk_track(y, w, order, 0.02) ≈ _dense_random_walk(y, w, order, 0.02) atol = 1.0e-12
    end
    @test eltype(FRpf._random_walk_track(Float32.(y), Float32.(w), 2, 0.02)) === Float32
    @test FRpf._random_walk_track(Float32.(y), Float32.(w), 2, 0.02) ≈ _dense_random_walk(y, w, 2, 0.02) rtol = 1.0e-5
end

@testset "MAP of one block" begin
    rng = Random.Xoshiro(4)
    n = 16
    x = collect(range(1.0e9, 1.03e9; length = n))
    y = 0.3 .* sin.(range(0, 2; length = n)) .+ 0.02 .* randn(rng, n)
    w = fill(2500.0, n)
    rw = CALpf.RandomWalkPrior(; order = 2, σ = 0.01)
    @test FRpf._estimate_map(rw, y, w, x) ≈ _dense_random_walk(y, w, 2, 0.01)

    # No prior returns the measured values, `NaN` where there are none.
    y2 = copy(y)
    y2[5] = NaN
    @test isequal(FRpf._estimate_map(nothing, y2, w, x), ifelse.(isfinite.(y2), y2, NaN))

    # Fewer usable channels than the walk's order: the measurement is kept.
    y3 = fill(NaN, n)
    y3[7] = 0.4
    @test isequal(FRpf._estimate_map(rw, y3, w, x), y3)
    @test all(isnan, FRpf._estimate_map(rw, fill(NaN, n), w, x))

    # OU is centered on the block's weighted mean.
    ou = CALpf.OUPrior(; scale = 1.0e7, σ = 0.2)
    yl = y .+ 3.0
    m = sum(yl .* w) / sum(w)
    @test FRpf._estimate_map(ou, yl, w, x) ≈ FRpf.smooth_ou_track(yl .- m, w, x; τ = 1.0e7, σ2 = 0.04) .+ m
    # Hyperparameters must be resolved first.
    @test_throws "OUPrior hyperparameters must be fixed" FRpf._estimate_map(
        CALpf.OUPrior(; scale = LogNormal(16.0, 1.0), σ = 0.2), y, w, x,
    )
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
    # type-II MAP over the blocks, each centered on its own mean.
    ou = CALpf.OUPrior(; scale = LogNormal(log(1.0e7), 1.0), σ = LogNormal(log(0.2), 1.0))
    est = FRpf._estimate_hypers(ou, ys, ws, xs)
    @test CALpf.is_fixed_hyper(est.scale) && CALpf.is_fixed_hyper(est.σ)
    ycs = [y .- sum(y .* w) / sum(w) for (y, w) in zip(ys, ws)]
    τ_lo, τ_hi = FRpf._group_ou_tau_bounds(xs)
    τ, σ2 = FRpf._map_ou_hypers(
        ycs, ws, xs, ou.scale, ou.σ; τ_lo, τ_hi, σ2_seed = FRpf._init_track_var(reduce(vcat, ycs), reduce(vcat, ws)),
    )
    @test est.scale ≈ τ && est.σ ≈ sqrt(σ2)
    # The resolved prior follows the ripple rather than flattening it.
    @test std(FRpf._estimate_map(est, ys[1], ws[1], xs[1])) > 0.1

    # A block without data does not enter the estimate; with no data at all
    # there is nothing to estimate from.
    empty = fill(NaN, 64)
    @test FRpf._estimate_hypers(ou, [ys; [empty]], [ws; [ws[1]]], [xs; [xs[1]]]) == est
    @test_throws "no block carries data" FRpf._estimate_hypers(ou, [empty], ws[1:1], xs[1:1])
end
