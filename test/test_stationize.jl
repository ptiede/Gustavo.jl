# Phase 4 — per-feed stationization with closure.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using Statistics: mean

const FR = Gustavo.Fringe
const CALs = Gustavo.Calibration

_chisign(fa, fb) = fa == fb ? 0 : (fa < fb ? 1 : -1)

# Build a noiseless per-baseline detection matrix from injected per-(station,
# feed) delays/rates/phases and a scan source phase χ. Baselines are exact
# station differences (+χ on cross-hand phase rows).
function inject_detections(bl_pairs, pol_products, τ, ṙ, φ, χ; snr = 100.0)
    nbl, npol = length(bl_pairs), length(pol_products)
    feeds = [CALs.correlation_feed_pair(p) for p in pol_products]
    D = Matrix{FR.FringeDetection}(undef, nbl, npol)
    for bi in 1:nbl, p in 1:npol
        a, b = bl_pairs[bi]
        fa, fb = feeds[p]
        cs = _chisign(fa, fb)
        delay = τ[a, fa] - τ[b, fb]
        rate = ṙ[a, fa] - ṙ[b, fb]
        phase = rem2pi(φ[a, fa] - φ[b, fb] + cs * χ, RoundNearest)
        D[bi, p] = FR.FringeDetection(delay, rate, phase, 1.0, snr, true)
    end
    return D
end

# Max |measured − model-from-solution| over all valid, non-auto baselines.
function recon_residuals(D, sol, bl_pairs, pol_products)
    feeds = [CALs.correlation_feed_pair(p) for p in pol_products]
    rd = rr = rp = 0.0
    for bi in eachindex(bl_pairs), p in eachindex(pol_products)
        det = D[bi, p]
        det.valid || continue
        a, b = bl_pairs[bi]
        a == b && continue
        fa, fb = feeds[p]
        cs = _chisign(fa, fb)
        rd = max(rd, abs(det.delay - (sol.delay[a, fa] - sol.delay[b, fb])))
        # Rate uses parallel hands only by default, so feed gauges are independent
        # — cross-hand rate is not reconstructable; only check parallel hands.
        cs == 0 && (rr = max(rr, abs(det.rate - (sol.rate[a, fa] - sol.rate[b, fb]))))
        mph = sol.phase[a, fa] - sol.phase[b, fb] + (cs == 0 ? 0.0 : cs * sol.chi)
        rp = max(rp, abs(rem2pi(det.phase - mph, RoundNearest)))
    end
    return (delay = rd, rate = rr, phase = rp)
end

all_baselines(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]

@testset "Stationize: per-(station,feed) recovery up to gauge" begin
    rng = MersenneTwister(0x5712)
    nant = 5
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)
    φ = 0.3 .* randn(rng, nant, 2)         # small enough to avoid wraps
    χ = 0.6

    D = inject_detections(bl, pols, τ, ṙ, φ, χ)
    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = ref)

    # Cross hands tie the feeds: delay is one component, gauged at (ref, feed1).
    for a in 1:nant, f in 1:2
        @test isapprox(sol.delay[a, f], τ[a, f] - τ[ref, 1]; atol = 1.0e-18)
    end
    # Rate uses parallel hands only → feeds gauged independently per feed.
    for a in 1:nant, f in 1:2
        @test isapprox(sol.rate[a, f], ṙ[a, f] - ṙ[ref, f]; atol = 1.0e-12)
    end
    # Phase: EVPA gauge pins both ref feeds; χ shifts by φ[ref,1]−φ[ref,2].
    for a in 1:nant, f in 1:2
        @test isapprox(rem2pi(sol.phase[a, f] - (φ[a, f] - φ[ref, f]), RoundNearest), 0.0; atol = 1.0e-10)
    end
    @test isapprox(rem2pi(sol.chi - (χ + φ[ref, 1] - φ[ref, 2]), RoundNearest), 0.0; atol = 1.0e-10)

    # Solution reconstructs every product (closure of the data).
    r = recon_residuals(D, sol, bl, pols)
    @test r.delay < 1.0e-15
    @test r.rate < 1.0e-12
    @test r.phase < 1.0e-9
end

@testset "Stationize: R-L offset recovered from cross hands" begin
    rng = MersenneTwister(0x99)
    nant = 4
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    # Feed-2 = feed-1 + a per-station R-L offset (delay and phase).
    τ1 = 2.0e-9 .* randn(rng, nant)
    rl_delay = 5.0e-9 .* randn(rng, nant)
    τ = hcat(τ1, τ1 .+ rl_delay)
    ṙ = zeros(nant, 2)
    φ1 = 0.2 .* randn(rng, nant)
    rl_phase = 0.4 .* randn(rng, nant)
    φ = hcat(φ1, φ1 .+ rl_phase)
    χ = -0.3

    D = inject_detections(bl, pols, τ, ṙ, φ, χ)
    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = ref)

    # Delay is ONE component (both feeds share the ref-feed-1 gauge via cross-hand
    # delay), so the per-station R-L offset is recovered *absolutely*.
    for a in 1:nant
        recovered = sol.delay[a, 2] - sol.delay[a, 1]
        @test isapprox(recovered, rl_delay[a]; atol = 1.0e-15)
    end
    @test sol.ncomp == 1                                # cross hands merge feeds
end

@testset "Stationize: parallel-hand triangle closure ≈ 0" begin
    rng = MersenneTwister(0x04)
    nant = 5
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    D = inject_detections(bl, pols, 1.0e-9 .* randn(rng, nant, 2), zeros(nant, 2), 0.25 .* randn(rng, nant, 2), 0.5)
    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = 1)
    # Parallel-hand products (PP=1, QQ=4) close exactly on noiseless data.
    for prod in (1, 4), obs in (:delay, :phase)
        res = FR.station_closure_residuals(D, bl, pols, sol; observable = obs, product = prod)
        @test !isempty(res)
        @test maximum(abs, res) < 1.0e-9
    end
end

@testset "Stationize: phase re-wrap handles |Δφ| > π" begin
    rng = MersenneTwister(0x2025)
    nant = 5
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    # Per-feed phases large enough that a fraction of station differences exceed
    # ±π and wrap (realistic residual-phase scale after delay/rate removal).
    φ = 1.2 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, zeros(nant, 2), zeros(nant, 2), φ, 0.0)
    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = ref, opts = FR.Stationization(phase_rewrap_iters = 6))
    r = recon_residuals(D, sol, bl, pols)
    @test r.phase < 1.0e-8                              # re-wrap closes despite wrapping
end

@testset "Stationize: disconnected array (two islands)" begin
    rng = MersenneTwister(0x01)
    # Antennas 1-3 and 4-6 with NO inter-island baselines.
    nant = 6
    bl = vcat(all_baselines(3), [(a, b) for a in 4:6 for b in (a + 1):6])
    pols = ["PP", "PQ", "QP", "QQ"]
    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)
    φ = 0.3 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, τ, ṙ, φ, 0.4)
    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = 1)
    @test sol.ncomp == 2                                # one component per island (feeds tied within each)
    # Reconstruction is gauge-invariant → residuals close within each island.
    r = recon_residuals(D, sol, bl, pols)
    @test r.delay < 1.0e-15
    @test r.phase < 1.0e-9
end

@testset "Stationize: parallel-only fallback (cross hands undetected)" begin
    rng = MersenneTwister(0x07)
    nant = 5
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    τ = 1.0e-9 .* randn(rng, nant, 2)
    φ = 0.3 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, τ, zeros(nant, 2), φ, 0.5)
    # Knock out every cross-hand detection (low SNR).
    feeds = [CALs.correlation_feed_pair(p) for p in pols]
    for bi in eachindex(bl), p in eachindex(pols)
        if feeds[p][1] != feeds[p][2]
            d = D[bi, p]
            D[bi, p] = FR.FringeDetection(d.delay, d.rate, d.phase, d.amp, 0.0, false)
        end
    end
    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = ref)
    @test sol.ncomp == 2                                # feeds NOT tied without cross hands
    @test isnan(sol.chi)                                # no cross hands → no χ
    # Each feed gauged independently; delay relative to that feed's ref.
    for a in 1:nant, f in 1:2
        @test isapprox(sol.delay[a, f], τ[a, f] - τ[ref, f]; atol = 1.0e-15)
        @test isapprox(rem2pi(sol.phase[a, f] - (φ[a, f] - φ[ref, f]), RoundNearest), 0.0; atol = 1.0e-10)
    end
end

@testset "Stationize: snr_min drops low-SNR detections" begin
    nant = 4
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    τ = 1.0e-9 .* randn(MersenneTwister(0x33), nant, 2)
    D = inject_detections(bl, pols, τ, zeros(nant, 2), zeros(nant, 2), 0.0; snr = 5.0)
    # All detections at SNR 5: snr_min=2 solves, snr_min=6 drops everything.
    sol_lo = FR.stationize_scan(D, bl, pols, nant; ref_ant = 1, opts = FR.Stationization(snr_min = 2.0))
    @test any(sol_lo.covered)
    sol_hi = FR.stationize_scan(D, bl, pols, nant; ref_ant = 1, opts = FR.Stationization(snr_min = 6.0))
    @test !any(sol_hi.covered)
    @test all(isnan, sol_hi.delay)
end
