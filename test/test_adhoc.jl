# Phase 5 — globally-closing adhoc phasing.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using Statistics: mean, std

const FRa = Gustavo.Fringe
const CALa = Gustavo.Calibration

_cs_a(fa, fb) = fa == fb ? 0 : (fa < fb ? 1 : -1)
all_bl_a(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]

# Build residual baseline visibilities rbar[bl,pol,ap] from a per-(station,feed,
# ap) phase screen and a per-AP source phase χ. `amp` sets the coherent SNR.
function inject_screen(bl_pairs, pol_products, screen, χ; amp = 10.0, noise = 0.0, rng = nothing)
    nbl, npol = length(bl_pairs), length(pol_products)
    nap = size(screen, 3)
    feeds = [CALa.correlation_feed_pair(p) for p in pol_products]
    rbar = Array{ComplexF64}(undef, nbl, npol, nap)
    wbar = ones(nbl, npol, nap)
    for ap in 1:nap, bi in 1:nbl, p in 1:npol
        a, b = bl_pairs[bi]
        fa, fb = feeds[p]
        cs = _cs_a(fa, fb)
        model = screen[a, fa, ap] - screen[b, fb, ap] + cs * χ[ap]
        v = amp * cis(model)
        if noise > 0 && rng !== nothing
            v += noise * (randn(rng) + im * randn(rng)) / sqrt(2)
        end
        rbar[bi, p, ap] = v
    end
    return rbar, wbar
end

# Max |measured − model-from-solution| (mod 2π) over valid baselines/APs.
function adhoc_recon(rbar, sol, bl_pairs, pol_products)
    feeds = [CALa.correlation_feed_pair(p) for p in pol_products]
    m = 0.0
    for ap in axes(rbar, 3), bi in eachindex(bl_pairs), p in eachindex(pol_products)
        a, b = bl_pairs[bi]
        a == b && continue
        r = rbar[bi, p, ap]
        abs(r) > 0 || continue
        fa, fb = feeds[p]
        cs = _cs_a(fa, fb)
        (isfinite(sol.phase[a, fa, ap]) && isfinite(sol.phase[b, fb, ap])) || continue
        model = sol.phase[a, fa, ap] - sol.phase[b, fb, ap] + (cs == 0 ? 0.0 : cs * sol.chi[ap])
        m = max(m, abs(rem2pi(angle(r) - model, RoundNearest)))
    end
    return m
end

@testset "Adhoc: raw per-AP global solve closes" begin
    rng = MersenneTwister(0x0ADC)
    nant, nap = 5, 20
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    screen = 0.3 .* randn(rng, nant, 2, nap)
    χ = 0.2 .* randn(rng, nap)
    times = collect(0:(nap - 1)) .* 1.0

    rbar, wbar = inject_screen(bl, pols, screen, χ)
    sol = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.NoSmoothing(detrend = false))

    @test adhoc_recon(rbar, sol, bl, pols) < 1.0e-9
    # Reference station held at zero adhoc phase (the per-AP gauge).
    @test all(abs.(sol.phase[ref, 1, :]) .< 1.0e-9)
    @test all(abs.(sol.phase[ref, 2, :]) .< 1.0e-9)
    # Per-AP recovery up to the (ref, feed) gauge.
    for ap in 1:nap, a in 1:nant, f in 1:2
        truth = screen[a, f, ap] - screen[ref, f, ap]
        @test isapprox(rem2pi(sol.phase[a, f, ap] - truth, RoundNearest), 0.0; atol = 1.0e-8)
    end
end

@testset "Adhoc: detrend removes per-scan mean only (keeps slope/rate)" begin
    rng = MersenneTwister(0x0DE7)
    nant, nap = 4, 30
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    tc = times .- mean(times)
    # Screen = per-station constant + slope + a common wiggle (cancels in the
    # ref-relative solve, so the recovered track is exactly constant + slope).
    c0 = 0.5 .* randn(rng, nant, 2)
    c1 = 0.02 .* randn(rng, nant, 2)
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant, f in 1:2, ap in 1:nap
        screen[a, f, ap] = c0[a, f] + c1[a, f] * tc[ap] + 0.05 * sin(2π * ap / nap)
    end
    χ = zeros(nap)
    rbar, wbar = inject_screen(bl, pols, screen, χ)
    sol = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.NoSmoothing(detrend = true))

    # Detrend removes the per-station MEAN (breaks the constant-phase gauge vs the
    # Stage-B ConstantTerm) but KEEPS the slope — so adhoc can flatten a residual
    # fringe rate. The recovered slope must match the injected differential rate
    # c1[a] - c1[ref] (the common wiggle cancels in the ref-relative solve).
    for a in 1:nant, f in 1:2
        a == ref && continue
        tr = sol.phase[a, f, :]
        @test abs(mean(tr)) < 1.0e-8
        slope = sum(tc .* (tr .- mean(tr))) / sum(tc .^ 2)
        @test isapprox(slope, c1[a, f] - c1[ref, f]; atol = 1.0e-8)
    end
end

@testset "Adhoc: SNR gate is invariant to WEIGHT column scale" begin
    # Regression: the per-AP SNR gate used |rbar|²/wbar, a true SNR only for
    # calibrated inverse-variance weights. On raw correlator output (uniform/
    # uncalibrated WEIGHT) the absolute scale is arbitrary, so a fixed snr_floor
    # dropped every row and killed the whole adhoc stage. The data-driven gate must
    # be invariant to a global weight rescale (rbar and wbar both scale by k).
    rng = MersenneTwister(0xBEEF)
    nant, nap = 4, 40
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant, f in 1:2, ap in 1:nap
        screen[a, f, ap] = 0.3 * randn(rng) + 0.04 * sin(2π * ap / nap + a)
    end
    rbar, wbar = inject_screen(bl, pols, screen, zeros(nap); amp = 5.0, noise = 0.4, rng = rng)
    sm = FRa.NoSmoothing(detrend = false, snr_floor = 1.0)
    s1 = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = sm)
    k = 1.0e-6
    s2 = FRa.solve_adhoc_phasing(k .* rbar, k .* wbar, bl, pols, nant, times; ref_ant = ref, smoother = sm)

    @test count(isfinite, s1.phase) > 0                      # adhoc actually runs
    @test count(isfinite, s2.phase) == count(isfinite, s1.phase)   # scale doesn't change coverage
    for i in eachindex(s1.phase)
        (isfinite(s1.phase[i]) && isfinite(s2.phase[i])) || continue
        @test isapprox(s1.phase[i], s2.phase[i]; atol = 1.0e-9)
    end
end

@testset "Adhoc: smoothing reduces noise on a smooth screen" begin
    rng = MersenneTwister(0x5704)
    nant, nap = 5, 60
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    # Smooth (band-limited) screen per station/feed.
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant, f in 1:2
        ph = 2π * rand(rng)
        amp = 0.4 * rand(rng)
        for ap in 1:nap
            screen[a, f, ap] = amp * sin(2π * 2 * ap / nap + ph)
        end
    end
    χ = zeros(nap)
    rbar, wbar = inject_screen(bl, pols, screen, χ; amp = 4.0, noise = 1.5, rng = rng)

    truth(a, f) = screen[a, f, :] .- screen[ref, f, :]
    rms_to_truth(sol) = begin
        e = Float64[]
        for a in 1:nant, f in 1:2
            a == ref && continue
            d = sol.phase[a, f, :] .- truth(a, f)
            d .-= mean(d)                      # remove gauge constant
            append!(e, d)
        end
        sqrt(mean(abs2, e))
    end

    raw = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.NoSmoothing(detrend = false))
    sm = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.SavitzkyGolaySmoother(window = 11, order = 2, detrend = false))
    pen = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.PenalizedSmoother(smoothness = 20.0, detrend = false))
    gp = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.OUSmoother(coherence_time = 15.0, detrend = false))

    @test rms_to_truth(sm) < rms_to_truth(raw)
    @test rms_to_truth(pen) < rms_to_truth(raw)
    # The OU / Matérn-1/2 Gaussian-process smoother (with ML-fit hypers) also
    # denoises, and should be competitive with the first-difference penalty.
    @test rms_to_truth(gp) < rms_to_truth(raw)
    @test rms_to_truth(gp) < 1.5 * rms_to_truth(pen)
end

@testset "Adhoc :gp preserves the injected differential slope" begin
    # Like the detrend test but through the :gp smoother: a constant + slope
    # screen (no wiggle) must survive GP smoothing + detrend with its slope
    # matching the injected differential rate c1[a] - c1[ref].
    rng = MersenneTwister(0x6959)
    nant, nap = 4, 40
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    tc = times .- mean(times)
    c0 = 0.5 .* randn(rng, nant, 2)
    c1 = 0.02 .* randn(rng, nant, 2)
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant, f in 1:2, ap in 1:nap
        screen[a, f, ap] = c0[a, f] + c1[a, f] * tc[ap]
    end
    rbar, wbar = inject_screen(bl, pols, screen, zeros(nap); amp = 20.0)
    sol = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.OUSmoother(coherence_time = 30.0, detrend = true))

    for a in 1:nant, f in 1:2
        a == ref && continue
        tr = sol.phase[a, f, :]
        @test abs(mean(tr)) < 1.0e-6
        slope = sum(tc .* (tr .- mean(tr))) / sum(tc .^ 2)
        @test isapprox(slope, c1[a, f] - c1[ref, f]; atol = 5.0e-3)
    end
end

@testset "Adhoc :gp ref antenna pinned to zero" begin
    rng = MersenneTwister(0x7A17)
    nant, nap = 4, 30
    ref = 2
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant, f in 1:2, ap in 1:nap
        screen[a, f, ap] = 0.3 * sin(2π * ap / nap + a) + 0.2 * randn(rng)
    end
    rbar, wbar = inject_screen(bl, pols, screen, zeros(nap); amp = 8.0, noise = 0.5, rng = rng)
    sol = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.OUSmoother(detrend = false))
    @test all(abs.(sol.phase[ref, :, :]) .< 1.0e-8)   # reference gauge held at 0
end

# Non-birefringent (feed-common) smooth screen for the joint solve — matches the
# `shared_feeds` truth (feed 2 ≡ feed 1) so the recovered per-station track is well
# defined up to the ref gauge.
function shared_screen(rng, nant, nap; k = 2.0, amp = 0.4)
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant
        ph = 2π * rand(rng)
        a0 = amp * rand(rng)
        for ap in 1:nap
            screen[a, 1, ap] = a0 * sin(2π * k * ap / nap + ph)
            screen[a, 2, ap] = screen[a, 1, ap]
        end
    end
    return screen
end

@testset "Adhoc :gp_joint runs, pins ref, recovers χ" begin
    rng = MersenneTwister(0x00102547)
    nant, nap = 5, 50
    ref = 2
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    screen = shared_screen(rng, nant, nap)
    χtrue = 0.3 .* sin.(2π .* (1:nap) ./ nap)
    rbar, wbar = inject_screen(bl, pols, screen, χtrue; amp = 6.0, noise = 1.0, rng = rng)
    sol = FRa.solve_adhoc_phasing(
        rbar, wbar, bl, pols, nant, times; ref_ant = ref,
        smoother = FRa.JointOUSmoother(coherence_time = 15.0, detrend = false), shared_feeds = true,
    )
    @test all(abs.(filter(isfinite, sol.phase[ref, :, :])) .< 1.0e-8)     # ref pinned
    @test all(sol.phase[:, 1, :] .=== sol.phase[:, 2, :])                 # shared feeds replicated
    # χ is per-AP; recovered up to a constant (the χ ↔ common-mode gauge).
    a = sol.chi .- mean(sol.chi)
    b = χtrue .- mean(χtrue)
    @test sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b)) > 0.9
end

@testset "Adhoc :gp_joint denoises and closes" begin
    rng = MersenneTwister(0x9317)
    nant, nap = 5, 60
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    screen = shared_screen(rng, nant, nap; k = 1.5, amp = 0.5)
    rbar, wbar = inject_screen(bl, pols, screen, zeros(nap); amp = 3.0, noise = 2.0, rng = rng)

    truth(a, f) = screen[a, f, :] .- screen[ref, f, :]
    rms_to_truth(sol) = begin
        e = Float64[]
        for a in 1:nant, f in 1:2
            a == ref && continue
            d = sol.phase[a, f, :] .- truth(a, f)
            all(isfinite, d) || continue
            d .-= mean(d)
            append!(e, d)
        end
        sqrt(mean(abs2, e))
    end

    none = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.NoSmoothing(detrend = false), shared_feeds = true)
    gp = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.OUSmoother(coherence_time = 20.0, detrend = false), shared_feeds = true)
    gpj = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.JointOUSmoother(coherence_time = 20.0, detrend = false), shared_feeds = true)

    @test rms_to_truth(gpj) < rms_to_truth(none)          # joint solve denoises
    @test rms_to_truth(gpj) < 1.5 * rms_to_truth(gp)      # competitive with per-track
end

@testset "Adhoc :gp_joint requires shared_feeds" begin
    rng = MersenneTwister(0x5A17)
    nant, nap = 3, 10
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    screen = shared_screen(rng, nant, nap)
    rbar, wbar = inject_screen(bl, pols, screen, zeros(nap); amp = 8.0)
    @test_throws ErrorException FRa.solve_adhoc_phasing(
        rbar, wbar, bl, pols, nant, times;
        smoother = FRa.JointOUSmoother(), shared_feeds = false,
    )
end

@testset "Adhoc: low-SNR no-anchor solve is unbiased" begin
    # No dominant anchor station (all equal SNR), low per-baseline SNR. The
    # global solve over all baselines should be unbiased — averaging many noise
    # realizations recovers the gauged truth.
    nant, nap = 5, 4
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    rng = MersenneTwister(0xB1A5)
    screen = 0.4 .* randn(rng, nant, 2, nap)
    χ = zeros(nap)

    ntrial = 80
    acc = zeros(nant, 2, nap)
    cnt = zeros(nant, 2, nap)
    for _ in 1:ntrial
        rbar, wbar = inject_screen(bl, pols, screen, χ; amp = 2.0, noise = 1.0, rng = rng)
        sol = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = FRa.NoSmoothing(detrend = false, snr_floor = 0.0))
        for a in 1:nant, f in 1:2, ap in 1:nap
            isfinite(sol.phase[a, f, ap]) || continue
            acc[a, f, ap] += rem2pi(sol.phase[a, f, ap], RoundNearest)
            cnt[a, f, ap] += 1
        end
    end
    maxbias = 0.0
    for a in 1:nant, f in 1:2, ap in 1:nap
        a == ref && continue
        cnt[a, f, ap] > 0 || continue
        est = acc[a, f, ap] / cnt[a, f, ap]
        truth = screen[a, f, ap] - screen[ref, f, ap]
        maxbias = max(maxbias, abs(rem2pi(est - truth, RoundNearest)))
    end
    # Averaging 80 low-SNR trials: bias well below the single-trial scatter.
    @test maxbias < 0.1
end

@testset "Adhoc: ref-antenna dropout gauge restitch (K3)" begin
    # A time-CONSTANT screen so the only across-AP variation is the per-AP gauge.
    # The reference antenna drops out in a middle block of APs; without the K3
    # restitch those APs anchor on a different node, injecting a common-mode jump
    # into every station's track. With it, the recovered track (relative to the
    # injected truth) is the SAME constant in every AP — including the dropout.
    nant, nap = 5, 12
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    rng = MersenneTwister(0xC3C3)

    c = 0.6 .* randn(rng, nant, 2)                 # per-(station,feed) constant
    screen = repeat(reshape(c, nant, 2, 1), 1, 1, nap)
    χ = zeros(nap)
    rbar, wbar = inject_screen(bl, pols, screen, χ)

    # Drop every baseline touching ref_ant in APs 5..8.
    dropaps = 5:8
    for ap in dropaps, bi in eachindex(bl)
        (bl[bi][1] == ref || bl[bi][2] == ref) || continue
        rbar[bi, :, ap] .= 0.0 + 0.0im
        wbar[bi, :, ap] .= 0.0
    end

    sol = FRa.solve_adhoc_phasing(
        rbar, wbar, bl, pols, nant, times;
        ref_ant = ref, smoother = FRa.NoSmoothing(detrend = false),
    )

    # ref_ant is unsolved in the dropout APs but solved elsewhere.
    @test all(!sol.covered[ref, 1, ap] && !sol.covered[ref, 2, ap] for ap in dropaps)
    @test all(sol.covered[ref, 1, ap] for ap in 1:nap if !(ap in dropaps))

    # Non-ref stations are still solved in the dropout APs, and the recovered
    # phase relative to truth is the SAME constant across ALL APs (no jump).
    for a in 2:nant, f in 1:2
        truth = c[a, f] - c[ref, f]
        d = [rem2pi(sol.phase[a, f, ap] - truth, RoundNearest) for ap in 1:nap if isfinite(sol.phase[a, f, ap])]
        @test length(d) == nap                                    # solved every AP
        @test maximum(d) - minimum(d) < 1.0e-6                    # gauge consistent across dropout
    end
end

@testset "Adhoc: warm-start selects the rewrap branch (no per-AP flips)" begin
    # Regression for the LA-baseline bimodality: a weakly-constrained station's
    # per-AP solve can be BISTABLE — two rewrap fixed points a sub-2π distance
    # apart — and which one the spanning-tree seed lands on can flip AP-to-AP,
    # injecting a phantom phase jump the integer-2π unwrap and the smoother both
    # leave intact. The per-AP solve must therefore accept a temporal warm-start
    # (`seed_phase`, the previous AP's solution) that pins the branch.
    #
    # Construct a genuinely bistable solve: station 4 sees two weak edges (to 2 and
    # 3, both pinned ≈0 by strong edges to ref=1) whose WRAPPED phases disagree by
    # ~2π, so φ4 ≈ 0 and φ4 ≈ ±π are BOTH self-consistent rewrap fixed points.
    OR = FRa._ObsRow
    rows = OR[
        OR(1, 2, 1, 1, 0.0, 100.0, 0),
        OR(1, 3, 1, 1, 0.0, 100.0, 0),
        OR(2, 4, 1, 1, 3.0, 1.0, 0),
        OR(3, 4, 1, 1, -3.0, 1.0, 0),
    ]
    nant, ref = 4, 1
    solve4(seed) = begin
        sp = seed === nothing ? nothing : (M = fill(NaN, nant, 2); M[4, 1] = seed; M)
        ph, _, cov, _ = FRa._solve_observable(rows, nant, ref; use_chi = false, rewrap = 3, seed_phase = sp)
        @test cov[4, 1]
        ph[4, 1]
    end

    # Unseeded, the solve lands on the ±π branch (NOT zero).
    φ_none = solve4(nothing)
    @test abs(abs(φ_none) - π) < 1.0e-3
    # A warm-start in the basin of zero selects the zero branch; one near π selects π.
    @test abs(solve4(0.0)) < 1.0e-3
    @test abs(abs(solve4(Float64(π))) - π) < 1.0e-3
    # The two seeded branches differ by ~π — the seed genuinely controls the result.
    @test abs(rem2pi(solve4(Float64(π)) - solve4(0.0), RoundNearest)) > 1.0

    # The default `solve_adhoc_phasing` warm-starts internally, so a smooth screen
    # near the wrap cut is recovered as a CONTINUOUS track (no spurious sub-2π
    # jumps) on the weak station.
    rng = MersenneTwister(0xBADC0FFE)
    nant2, nap = 5, 60
    bl = all_bl_a(nant2)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    screen = 0.2 .* randn(rng, nant2, 2, nap)
    for ap in 1:nap                                  # station 5 hovers near +π
        screen[5, :, ap] .= (π - 0.1) .+ 0.1 .* sin(2π * ap / 15)
    end
    χ = zeros(nap)
    rbar, wbar = inject_screen(bl, pols, screen, χ; amp = 6.0, noise = 1.0, rng = rng)
    sol = FRa.solve_adhoc_phasing(
        rbar, wbar, bl, pols, nant2, times;
        ref_ant = 1, smoother = FRa.NoSmoothing(detrend = false),
    )
    tr = sol.phase[5, 1, :]
    jumps = [abs(rem2pi(tr[ap + 1] - tr[ap], RoundNearest)) for ap in 1:(nap - 1) if isfinite(tr[ap]) && isfinite(tr[ap + 1])]
    @test maximum(jumps) < 1.0                       # no ~π branch flip in the track
end

@testset "EHT-HOPS adhoc window (T_dof, Eqs 21-22)" begin
    # Savitzky–Golay window from the EHT-HOPS optimal integration time. SNR-adaptive
    # (higher per-AP SNR² → shorter window) and grows with the assumed coherence
    # time; odd and ≥ order+1.
    order = 2
    w_hi = FRa._savgol_window_dof(1.0e3, 15.0, 5 / 3, order)         # high SNR
    w_lo = FRa._savgol_window_dof(10.0, 15.0, 5 / 3, order)          # low  SNR
    w_lo_longcoh = FRa._savgol_window_dof(10.0, 60.0, 5 / 3, order)  # longer T_coh
    @test w_hi <= w_lo                                  # higher SNR → tighter window
    @test w_lo_longcoh >= w_lo                          # longer coherence → larger window
    for w in (w_hi, w_lo, w_lo_longcoh)
        @test isodd(w) && w >= order + 1
    end
    # Degenerate inputs fall back to the minimal window.
    @test FRa._savgol_window_dof(0.0, 15.0, 5 / 3, order) == order + 1
    @test FRa._savgol_window_dof(10.0, 0.0, 5 / 3, order) == order + 1
end

@testset "Adhoc smoother interface: every type dispatches" begin
    # The pluggable `AbstractAdhocSmoother` interface — each concrete smoother
    # constructs, subtypes the abstract type, and drives `solve_adhoc_phasing`
    # through `apply_adhoc!`. Mirrors the bandpass-smoother loop in test_pipeline.jl.
    rng = MersenneTwister(0x00FACADE)
    nant, nap = 4, 30
    ref = 1
    bl = all_bl_a(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    times = collect(0:(nap - 1)) .* 1.0
    screen = shared_screen(rng, nant, nap)         # feed-common (works for the joint solve too)
    rbar, wbar = inject_screen(bl, pols, screen, zeros(nap); amp = 6.0, noise = 0.5, rng = rng)

    # (smoother, shared_feeds) — the joint solve requires feed-common state.
    cases = [
        (FRa.SavitzkyGolaySmoother(window = 9, detrend = false), false),
        (FRa.PenalizedSmoother(smoothness = 10.0, detrend = false), false),
        (FRa.OUSmoother(coherence_time = 15.0, detrend = false), false),
        (FRa.NoSmoothing(detrend = false), false),
        (FRa.JointOUSmoother(coherence_time = 15.0, detrend = false), true),
    ]
    for (sm, shared) in cases
        @test sm isa FRa.AbstractAdhocSmoother
        sol = FRa.solve_adhoc_phasing(rbar, wbar, bl, pols, nant, times; ref_ant = ref, smoother = sm, shared_feeds = shared)
        @test sol isa FRa.AdhocSolution
        @test count(isfinite, sol.phase) > 0                     # the stage actually ran
        @test all(abs.(filter(isfinite, sol.phase[ref, :, :])) .< 1.0e-8)   # ref gauge held
    end
end
