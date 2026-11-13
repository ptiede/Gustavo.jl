# Per-AP station phases of one scan (`solve_adhoc`).
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using Statistics: mean, std
using Gustavo.DimensionalData: At, DimArray, Ti, lookup
using Distributions: LogNormal, Normal

const FRa = Gustavo.Fring
const CALa = Gustavo.Calibration

all_bl_a(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]
const FULL_FEEDS = [(1, 1), (1, 2), (2, 1), (2, 2)]

# Weighted sums `r = Σ w·v` and `w = Σ w` over (AntennaPair, FeedPair, Ti) of
# `v = amp·flux[t]·exp(η[a, fa, t] + η[b, fb, t] + i(screen[a, fa, t] − screen[b, fb, t] + x[bl, p]))`
# plus complex noise of standard deviation `σ` per component, one visibility per
# cell with weight `1/σ²`. Station `i` is `"S$i"`; `drop(a, t)` leaves station `a`
# out of AP `t`.
function adhoc_sums(
        screen, feeds = FULL_FEEDS; x = nothing, amp = 1.0, σ = 0.01, rng = Xoshiro(1),
        times = 0.0:(size(screen, 3) - 1), drop = (a, t) -> false, T = Float64,
        η = zeros(size(screen)), flux = ones(size(screen, 3)),
    )
    nant, _, nap = size(screen)
    bl = all_bl_a(nant)
    names = ["S$i" for i in 1:nant]
    ax = (FRa._antenna_pair_dim([(names[a], names[b]) for (a, b) in bl]), FRa.FeedPair(feeds), Ti(collect(Float64, times)))
    r, w = zeros(Complex{T}, ax), zeros(T, ax)
    for t in 1:nap, (p, (fa, fb)) in pairs(feeds), (bi, (a, b)) in pairs(bl)
        (drop(a, t) || drop(b, t)) && continue
        v = amp * flux[t] * exp(η[a, fa, t] + η[b, fb, t]) * cis(screen[a, fa, t] - screen[b, fb, t] + (isnothing(x) ? 0.0 : x[bi, p])) +
            σ * complex(randn(rng), randn(rng))
        r[bi, p, t], w[bi, p, t] = v / σ^2, 1 / σ^2
    end
    return r, w, names
end

# A smooth screen per (station, feed): a random walk of step `step` radians per
# AP, the same on both feeds unless `perfeed`.
function walk_screen(rng, nant, nap; step = 0.1, perfeed = false)
    s = cumsum(step .* randn(rng, nant, 2, nap); dims = 3)
    perfeed || (s[:, 2, :] .= s[:, 1, :])
    return s
end

# The solved phases of feed `f` against the truth, with what the data leave free
# taken out: each AP's mean over stations and each station's mean over the scan.
function adhoc_error(sol, screen; f = 1)
    d = parent(sol.phase)[:, f, :] .- screen[:, f, :]
    d .-= mean(d; dims = 1)
    d .-= mean(d; dims = 2)
    return d
end

@testset "solve_adhoc" begin
    @testset "recovers the station phases up to the freedoms the data leave" begin
        rng = Xoshiro(11)
        nant, nap = 5, 40
        screen = walk_screen(rng, nant, nap)
        x = 2π .* rand(rng, length(all_bl_a(nant)), length(FULL_FEEDS))
        r, w, names = adhoc_sums(screen; x, σ = 0.05, rng)
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
        @test maximum(abs, adhoc_error(sol, screen)) < 0.03
        # A feed-common track: both feeds carry the same phase.
        @test parent(sol.phase)[:, 1, :] == parent(sol.phase)[:, 2, :]
    end

    @testset "the gauge sets each AP's common phase and nothing else" begin
        rng = Xoshiro(12)
        nant, nap = 4, 30
        screen = walk_screen(rng, nant, nap)
        r, w, names = adhoc_sums(screen; σ = 0.05, rng)
        zs = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
        pin = FRa.solve_adhoc(r, w, names; gauge = PinAntenna(2))
        @test maximum(abs, sum(parent(zs.phase)[:, 1, :]; dims = 1)) < 1.0e-8
        @test maximum(abs, parent(pin.phase)[2, :, :]) < 1.0e-8
        baselines(s) = [s.phase[a, 1, t] - s.phase[b, 1, t] for (a, b) in all_bl_a(nant), t in 1:nap]
        @test maximum(abs, baselines(zs) .- baselines(pin)) < 1.0e-6
    end

    @testset "the gauge moves a station under a nearly rigid prior" begin
        # The gauge moves every station by its AP's common shift, so it must
        # reach a station whose prior curvature exceeds the others' by ~1e14.
        rng = Xoshiro(17)
        nant, nap = 4, 200
        screen = walk_screen(rng, nant, nap)
        screen[1, :, :] .= 0.3
        r, w, names = adhoc_sums(screen; σ = 0.05, rng)
        rigid = GainComponent(ConstantTerm(); Ti = PerIntegration(), Feed = SharedFeeds(), prior = OUPrior(; scale = 7.5e3, σ = 5.0e-7))
        model = CALa.with_antenna(FRa.default_adhoc_model(; phase = OUPrior(; scale = 10.0, σ = 0.3)), "S1"; phase = (; adhoc = rigid))
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase(), model)
        @test maximum(abs, sum(parent(sol.phase)[:, 1, :]; dims = 1)) < 1.0e-8
        @test maximum(abs, adhoc_error(sol, screen)) < 0.05
    end

    @testset "a ranked reference falls to the next where the first is absent" begin
        rng = Xoshiro(13)
        nant, nap = 4, 30
        screen = walk_screen(rng, nant, nap)
        absent = 11:20
        r, w, names = adhoc_sums(screen; σ = 0.05, rng, drop = (a, t) -> a == 1 && t in absent)
        sol = FRa.solve_adhoc(r, w, names; gauge = PinAntenna([1, 2]))
        present = setdiff(1:nap, absent)
        @test maximum(abs, parent(sol.phase)[1, 1, present]) < 1.0e-8
        @test maximum(abs, parent(sol.phase)[2, 1, absent]) < 1.0e-8
        @test all(isnan, parent(sol.phase)[1, :, absent])
    end

    @testset "low SNR with source phases near ±π: no branch slips" begin
        rng = Xoshiro(14)
        nant, nap = 5, 120
        screen = walk_screen(rng, nant, nap; step = 0.05)
        nbl = length(all_bl_a(nant))
        x = [π - 0.05 * rand(rng) * (isodd(k) ? 1 : -1) for k in 1:nbl, _ in FULL_FEEDS]
        # SNR 1 per visibility component.
        r, w, names = adhoc_sums(screen; x, σ = 1.0, rng)
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
        d = adhoc_error(sol, screen)
        @test maximum(abs, d) < π / 2
        @test std(d) < 0.25
    end

    @testset "the prior smooths a smooth screen" begin
        rng = Xoshiro(15)
        nant, nap = 5, 80
        screen = walk_screen(rng, nant, nap; step = 0.02)
        r, w, names = adhoc_sums(screen; σ = 0.7, rng)
        smooth = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
        perap = FRa.solve_adhoc(
            r, w, names; gauge = ZeroSumPhase(), model = FRa.default_adhoc_model(; phase = IIDPrior(10.0)),
        )
        @test std(adhoc_error(smooth, screen)) < 0.6 * std(adhoc_error(perap, screen))
    end

    @testset "a station's phase rate survives" begin
        rng = Xoshiro(16)
        nant, nap = 4, 60
        rates = [0.0, 0.02, -0.015, 0.01]
        screen = zeros(nant, 2, nap)
        for a in 1:nant, t in 1:nap
            screen[a, :, t] .= rates[a] * (t - 1)
        end
        r, w, names = adhoc_sums(screen; σ = 0.05, rng)
        sol = FRa.solve_adhoc(r, w, names; gauge = PinAntenna(1))
        t = collect(0.0:(nap - 1))
        slope(y) = sum((t .- mean(t)) .* (y .- mean(y))) / sum(abs2, t .- mean(t))
        @test [slope(parent(sol.phase)[a, 1, :]) for a in 1:nant] ≈ rates .- rates[1] atol = 1.0e-3
    end

    @testset "each station's prior comes back resolved" begin
        rng = Xoshiro(17)
        nant, nap = 4, 40
        screen = walk_screen(rng, nant, nap)
        r, w, names = adhoc_sums(screen; σ = 0.1, rng, drop = (a, t) -> a == 4)
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
        @test all(p -> p isa OUPrior && p.scale isa Real && p.σ isa Real, parent(sol.phase_prior)[1:3, :])
        @test all(isnothing, parent(sol.phase_prior)[4, :])
        walk = RandomWalkPrior(; σ = LogNormal(log(0.1), 1.0), init = Normal(0.0, 1.0))
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase(), model = FRa.default_adhoc_model(; phase = walk))
        @test all(p -> p isa RandomWalkPrior && p.σ isa Real, parent(sol.phase_prior)[1:3, :])
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase(), model = FRa.default_adhoc_model(; phase = IIDPrior(1.0)))
        @test all(==(IIDPrior(1.0)), parent(sol.phase_prior)[1:3, :])
        @test all(isnothing, parent(sol.phase_prior)[4, :])
    end

    @testset "a scan of one AP" begin
        rng = Xoshiro(19)
        screen = walk_screen(rng, 4, 1)
        r, w, names = adhoc_sums(screen; σ = 0.05, rng)
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
        @test maximum(abs, adhoc_error(sol, screen)) < 0.1
    end

    @testset "autocorrelations alone are no data" begin
        ax = (FRa._antenna_pair_dim([("S1", "S1"), ("S1", "S2")]), FRa.FeedPair([(1, 1)]), Ti([0.0]))
        w = zeros(ax)
        w[1, 1, 1] = 1.0
        @test !FRa._has_cross_data(w)
        w[2, 1, 1] = 1.0
        @test FRa._has_cross_data(w)
    end

    @testset "the solve takes the sums' element type" begin
        rng = Xoshiro(18)
        screen = walk_screen(rng, 4, 30)
        r64, w64, names = adhoc_sums(screen; σ = 0.05, rng = Xoshiro(1))
        r32, w32, _ = adhoc_sums(screen; σ = 0.05, rng = Xoshiro(1), T = Float32)
        a = FRa.solve_adhoc(r64, w64, names; gauge = ZeroSumPhase())
        b = FRa.solve_adhoc(r32, w32, names; gauge = ZeroSumPhase())
        @test maximum(abs, parent(a.phase) .- parent(b.phase)) < 1.0e-4
    end

    @testset "sums are read by label, not position" begin
        rng = Xoshiro(19)
        screen = walk_screen(rng, 4, 30)
        r, w, names = adhoc_sums(screen; σ = 0.05, rng)
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
        bl, fp = reverse(1:size(r, 1)), [3, 1, 4, 2]
        shuffled = FRa.solve_adhoc(r[bl, fp, :], w[bl, fp, :], names; gauge = ZeroSumPhase())
        @test maximum(abs, parent(sol.phase) .- parent(shuffled.phase)) < 1.0e-8
    end

    @testset "PerFeed fits each feed's own track" begin
        rng = Xoshiro(20)
        nant, nap = 4, 40
        screen = walk_screen(rng, nant, nap; perfeed = true)
        r, w, names = adhoc_sums(screen; σ = 0.05, rng)
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase(), model = FRa.default_adhoc_model(; feed = PerFeed()))
        # The cross hands join the feeds, so one common phase per AP is free:
        # remove it once, over both feeds.
        d = parent(sol.phase) .- screen
        d .-= mean(d; dims = (1, 2))
        d .-= mean(d; dims = 3)
        @test std(d) < 0.03
    end

    @testset "a station with one receptor" begin
        rng = Xoshiro(21)
        nant, nap = 4, 30
        screen = walk_screen(rng, nant, nap)
        r, w, names = adhoc_sums(screen; σ = 0.05, rng)
        for (bi, (a, b)) in pairs(all_bl_a(nant)), (p, (fa, fb)) in pairs(FULL_FEEDS)
            (a == 3 && fa == 2) || (b == 3 && fb == 2) || continue
            r[bi, p, :] .= 0
            w[bi, p, :] .= 0
        end
        sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
        @test std(adhoc_error(sol, screen)) < 0.03
        @test parent(sol.phase)[3, 2, :] == parent(sol.phase)[3, 1, :]
    end
end

@testset "solve_adhoc: amplitudes" begin
    # Per-station log-amplitude tracks come back up to each station's constant,
    # which its `S` absorbs and its prior holds.
    rng = Xoshiro(21)
    nant, nap = 5, 60
    screen = walk_screen(rng, nant, nap)
    η = cumsum(0.01 .* randn(rng, nant, 2, nap); dims = 3)
    η[:, 2, :] .= η[:, 1, :]
    r, w, names = adhoc_sums(screen; η, σ = 0.02, rng)
    sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
    d = parent(sol.logamp)[:, 1, :] .- η[:, 1, :]
    d .-= mean(d; dims = 2)
    @test maximum(abs, d) < 0.02
    @test parent(sol.logamp)[:, 1, :] == parent(sol.logamp)[:, 2, :]
    @test all(p -> p isa OUPrior && p.scale isa Real && p.σ isa Real, sol.logamp_prior)
    @test maximum(abs, adhoc_error(sol, screen)) < 0.05

    # `S` is constant over the scan, so a change of the source's flux within it
    # lands on the stations: half its log on each.
    logF = 0.2 .* sin.(range(0, 2π; length = nap))
    r, w, names = adhoc_sums(screen; flux = exp.(logF), σ = 0.02, rng)
    sol = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase())
    ℓ = parent(sol.logamp)[:, 1, :]
    @test maximum(abs, (ℓ .- mean(ℓ; dims = 2)) .- (logF .- mean(logF))' ./ 2) < 0.02

    # A phase-only model leaves the amplitudes unsolved.
    phase_only = FRa.solve_adhoc(r, w, names; gauge = ZeroSumPhase(), model = FRa.default_adhoc_model(; logamp = nothing))
    @test all(isnan, phase_only.logamp)
    @test all(isnothing, phase_only.logamp_prior)

    # Two stations: their one baseline measures only the sum of their
    # log-amplitudes, so the priors alone split it.
    r2, w2, names2 = adhoc_sums(screen[1:2, :, :]; σ = 0.02, rng)
    @test_logs (:warn, r"no odd loop") match_mode = :any FRa.solve_adhoc(r2, w2, names2; gauge = PinAntenna(1))
end

@testset "the hyperparameter fit's gradient is its Laplace evidence's" begin
    rng = Xoshiro(31)
    nant, nap = 4, 30
    screen = walk_screen(rng, nant, nap)
    η = cumsum(0.02 .* randn(rng, nant, 2, nap); dims = 3)
    η[:, 2, :] .= η[:, 1, :]
    rbar, wbar, names = adhoc_sums(screen; η, σ = 0.05, rng)
    # The joint solve of this scan up to its hyperparameter fit, as `_solve_adhoc!` sets it up.
    times = collect(Float64, lookup(rbar, Ti))
    geom = FRa.DataGeometry(; times, channel_freqs = [0.0], antennas = names, nfeed = 2)
    model = FRa.default_adhoc_model()
    layout = FRa.plan_parameters(model, geom.antennas, geom)
    blocks = map(p -> FRa.antenna_blocks(layout, zeros(layout.nθ), p.path...), FRa._adhoc_plans(model, layout))
    r, w = map((rbar, wbar)) do x
        y = permutedims(x, (FRa.AntennaPair, FRa.FeedPair, Ti))
        DimArray(reshape(parent(y), 1, size(y)...), (FRa.Scan(1:1), FRa.DimensionalData.dims(y)...))
    end
    tolerance, linsolve = 1.0e-6, FRa.LinearSolve.KrylovJL_CG(; linesearch = true)
    (; ctx, priors, x) = FRa._joint_setup(
        (; phase = [blocks.phase], logamp = [blocks.logamp]), (; r, w), geom;
        cover = [(t, 1) for _ in 1:1, t in eachindex(times)], max_iterations = 200, tolerance, linsolve, executor = SerialScheduler(),
    )
    est = FRa._estimated(ctx)
    ψ = FRa._log_hyper_vector(priors, ctx, est)
    function evidence(ψ)
        FRa._set_log_hypers!(priors, ctx, est, ψ)
        solved = FRa._profiled_map!(x, ctx, priors; tolerance, max_iterations = 200, linsolve, exact = true)
        return FRa._hyper_objective(priors, x, ctx, solved, est; linsolve, abstol = tolerance / 10)
    end
    _, G, _ = evidence(ψ)
    h = 1.0e-3
    fd = map(eachindex(ψ)) do i
        e = h .* (eachindex(ψ) .== i)
        return (first(evidence(ψ .+ e)) - first(evidence(ψ .- e))) / 2h
    end
    @test all(i -> abs(G[i] - fd[i]) < 0.02 * max(abs(fd[i]), 1), eachindex(ψ))
end

@testset "Adhoc models" begin
    comp = CALa.GainComponent(CALa.ConstantTerm(); Ti = CALa.PerIntegration(), prior = FRa.default_adhoc_phase_prior())
    vet(model) = Gustavo.model_components(Adhoc(; model), nothing)
    @test vet(FRa.default_adhoc_model()) isa CALa.GainModel
    @test vet(FRa.default_adhoc_model(; phase = IIDPrior(1.0))) isa CALa.GainModel
    @test_throws "exactly one per-integration phase component" vet(CALa.GainModel(; phase = (; a = comp, b = comp)))
    @test vet(CALa.GainModel(; phase = (; adhoc = comp), logamp = (; adhoc = comp))) isa CALa.GainModel
    @test vet(FRa.default_adhoc_model(; logamp = nothing)) isa CALa.GainModel
    @test_throws "at most one per-integration logamp component" vet(CALa.GainModel(; phase = (; adhoc = comp), logamp = (; a = comp, b = comp)))
    unsmoothed = CALa.GainComponent(CALa.ConstantTerm(); Ti = CALa.PerIntegration(), prior = nothing)
    @test_throws "Adhoc cannot fit the component" vet(CALa.GainModel(; phase = (; adhoc = unsmoothed)))
    flat_walk = CALa.GainComponent(CALa.ConstantTerm(); Ti = CALa.PerIntegration(), prior = RandomWalkPrior(; σ = 0.1))
    @test_throws "Adhoc cannot fit the component" vet(CALa.GainModel(; phase = (; adhoc = flat_walk)))
end
