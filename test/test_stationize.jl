# Phase 4 — per-feed stationization with closure.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using Statistics: mean
using DimensionalData: metadata

const FR = Gustavo.Fringe
const CALs = Gustavo.Calibration

_chisign(fa, fb) = fa == fb ? 0 : (fa < fb ? 1 : -1)

# Build a noiseless per-baseline detection matrix from injected per-(station,
# feed) delays/rates/phases and a scan source phase χ. Baselines are exact
# station differences (+χ on cross-hand phase rows).
function inject_detections(bl_pairs, pol_products, τ, ṙ, φ, χ; snr = 100.0)
    nbl, npol = length(bl_pairs), length(pol_products)
    feeds = [CALs.correlation_feed_pair(p) for p in pol_products]
    D = Matrix{FR.Detection}(undef, nbl, npol)
    for bi in 1:nbl, p in 1:npol
        a, b = bl_pairs[bi]
        fa, fb = feeds[p]
        cs = _chisign(fa, fb)
        delay = τ[a, fa] - τ[b, fb]
        rate = ṙ[a, fa] - ṙ[b, fb]
        phase = rem2pi(φ[a, fa] - φ[b, fb] + cs * χ, RoundNearest)
        D[bi, p] = FR.Detection((delay, rate, phase, 1.0, snr, true))
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
        mph = sol.phase[a, fa] - sol.phase[b, fb] + (cs == 0 ? 0.0 : cs * metadata(sol)[:chi])
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

    @test sol isa Gustavo.DimensionalData.AbstractDimStack
    @test (:delay, :rate, :phase, :covered) ⊆ keys(sol)      # Ant × Feed layers
    @test haskey(metadata(sol), :chi) && haskey(metadata(sol), :ncomp)

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
    @test isapprox(rem2pi(metadata(sol)[:chi] - (χ + φ[ref, 1] - φ[ref, 2]), RoundNearest), 0.0; atol = 1.0e-10)

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
    @test metadata(sol)[:ncomp] == 1                                # cross hands merge feeds
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
    @test metadata(sol)[:ncomp] == 2                                # one component per island (feeds tied within each)
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
            D[bi, p] = FR.Detection((d.delay, d.rate, d.phase, d.amp, 0.0, false))
        end
    end
    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = ref)
    @test metadata(sol)[:ncomp] == 2                                # feeds NOT tied without cross hands
    @test isnan(metadata(sol)[:chi])                                # no cross hands → no χ
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

@testset "Stationize: spanning-tree re-wrap recovers |Δφ| > π (K1)" begin
    # A chain of high-SNR baselines (1-2-3-4-5) carries small per-step phase
    # increments that accumulate to a station-to-station difference exceeding ±π.
    # Redundant (lower-SNR) baselines therefore have TRUE phase differences > π,
    # whose measured values wrap into (−π, π]. The max-weight spanning-tree seed
    # propagates the unambiguous chain unwrapping so the redundant edges are
    # re-wrapped to the correct 2π branch; a fit seeded from the raw wrapped
    # observations can lock onto the wrong branch.
    nant = 5
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    feeds = [CALs.correlation_feed_pair(p) for p in pols]

    # Cumulative phase: φ[a] − φ[ref] grows past π along the chain.
    φ = zeros(nant, 2)
    for a in 1:nant
        φ[a, 1] = 1.2 * (a - 1)            # 0, 1.2, 2.4, 3.6, 4.8  (1↔5 diff = 4.8 > π)
        φ[a, 2] = 1.2 * (a - 1) + 0.4
    end
    χ = 0.3
    τ = zeros(nant, 2)
    ṙ = zeros(nant, 2)

    # Chain edges get the highest SNR so the spanning tree follows them.
    is_chain(a, b) = abs(a - b) == 1
    D = Matrix{FR.Detection}(undef, length(bl), length(pols))
    for (bi, (a, b)) in enumerate(bl), (p, (fa, fb)) in enumerate(feeds)
        cs = _chisign(fa, fb)
        phase = rem2pi(φ[a, fa] - φ[b, fb] + cs * χ, RoundNearest)
        snr = is_chain(a, b) ? 200.0 : 100.0
        D[bi, p] = FR.Detection((τ[a, fa] - τ[b, fb], ṙ[a, fa] - ṙ[b, fb], phase, 1.0, snr, true))
    end

    # At least one redundant baseline genuinely exceeds ±π in parallel hand.
    @test any(!is_chain(a, b) && abs(φ[a, 1] - φ[b, 1]) > π for (a, b) in bl)

    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = ref)

    # Model reproduces every measured phase (closure after correct unwrap).
    @test recon_residuals(D, sol, bl, pols).phase < 1.0e-6
    # The recovered absolute (unwrapped) station phases match the injected branch
    # — not a wrapped alias. Feed-1 is gauged at (ref, feed1).
    for a in 1:nant
        @test isapprox(sol.phase[a, 1] - sol.phase[ref, 1], φ[a, 1] - φ[ref, 1]; atol = 1.0e-6)
    end
end

@testset "solve_station_systems! ≡ stationize_scan (per-scan, model-driven)" begin
    # The model-driven column solver must reproduce the per-scan reference solve
    # byte-for-byte when the model is per-scan/per-feed (the block-diagonal case).
    rng = MersenneTwister(0x00C0FFEE)
    nant = 5
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    feeds = [CALs.correlation_feed_pair(p) for p in pols]

    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)
    φ = 0.3 .* randn(rng, nant, 2)
    χ = 0.7
    D = inject_detections(bl, pols, τ, ṙ, φ, χ; snr = 100.0)

    # Reference: the existing per-scan stationizer.
    ss = FR.stationize_scan(D, bl, pols, nant; ref_ant = ref)

    # Model-driven engine: const/delay/rate as PerScan × PerFeed over a one-scan
    # geometry, solved through the off1 columns the layout declares.
    geom = CALs.DataGeometry(; times = [0.0, 1.0, 2.0], channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9)
    model = CALs.StationGainModel(
        phase = (
            offset = CALs.TiedComponent(CALs.GainComponent(CALs.ConstantTerm(), CALs.PerScan(), CALs.GlobalFrequency()), CALs.PerFeed()),
            delay = CALs.TiedComponent(CALs.GainComponent(CALs.Delay(), CALs.PerScan(), CALs.GlobalFrequency()), CALs.PerFeed()),
            rate = CALs.TiedComponent(CALs.GainComponent(CALs.Rate(), CALs.PerScan(), CALs.GlobalFrequency()), CALs.PerFeed()),
        ),
    )
    layout = CALs.plan_parameters(model, nant, geom)
    cplan, dplan, rplan = layout.plans[1], layout.plans[2], layout.plans[3]

    θ = zeros(layout.nθ)
    scans = (FR.StationScanDetections(D, bl, feeds, 1),)
    chi, ncomp = FR.solve_station_systems!(
        θ, scans, ((cplan, :phase), (dplan, :delay), (rplan, :rate)); ref_ant = ref,
    )

    colval(plan, ant, feed) = (c = plan.off1[ant, feed, 1, 1]; c == 0 ? NaN : θ[c])
    for ant in 1:nant, feed in 1:2
        if isfinite(ss.delay[ant, feed])
            @test colval(dplan, ant, feed) ≈ ss.delay[ant, feed] atol = 1.0e-12
        end
        if isfinite(ss.rate[ant, feed])
            @test colval(rplan, ant, feed) ≈ ss.rate[ant, feed] atol = 1.0e-12
        end
        if isfinite(ss.phase[ant, feed])
            @test colval(cplan, ant, feed) ≈ ss.phase[ant, feed] atol = 1.0e-9
        end
    end
    @test ncomp == metadata(ss)[:ncomp]
    @test chi ≈ metadata(ss)[:chi] atol = 1.0e-9
end

@testset "Global R-L offset: stable across scans, weak scan inherits it" begin
    # Two scans share ONE stable R-L (feed-2 − feed-1) delay/phase offset per
    # station; the per-scan feed-common delays/phases differ. Scan 2 has NO
    # cross-hand detections (the weak case that splits into ncomp=2 per-scan). The
    # global FeedComponent(2) × GlobalTime offset, pinned by scan 1's cross hands,
    # must tie scan 2's feeds too.
    rng = MersenneTwister(0x5EED)
    nant = 4
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    feeds = [CALs.correlation_feed_pair(p) for p in pols]

    δ = 1.0e-9 .* randn(rng, nant)          # global R-L delay offset (feed2 − feed1)
    ε = 0.5 .* randn(rng, nant)             # global R-L phase offset
    Dc = [1.0e-9 .* randn(rng, nant), 1.0e-9 .* randn(rng, nant)]   # per-scan feed-common delay
    Φc = [0.3 .* randn(rng, nant), 0.3 .* randn(rng, nant)]          # per-scan feed-common phase
    χs = [0.6, -0.4]

    function scan_det(s; with_cross)
        D = Matrix{FR.Detection}(undef, length(bl), length(pols))
        for (bi, (a, b)) in enumerate(bl), (p, (fa, fb)) in enumerate(feeds)
            cs = _chisign(fa, fb)
            valid = with_cross || cs == 0
            τa = Dc[s][a] + (fa == 2 ? δ[a] : 0.0)
            τb = Dc[s][b] + (fb == 2 ? δ[b] : 0.0)
            φa = Φc[s][a] + (fa == 2 ? ε[a] : 0.0)
            φb = Φc[s][b] + (fb == 2 ? ε[b] : 0.0)
            phase = rem2pi(φa - φb + cs * χs[s], RoundNearest)
            D[bi, p] = FR.Detection((τa - τb, 0.0, phase, 1.0, 100.0, valid))
        end
        return D
    end
    D1 = scan_det(1; with_cross = true)
    D2 = scan_det(2; with_cross = false)     # weak scan: cross hands undetected

    geom = CALs.DataGeometry(;
        times = [0.0, 1.0, 2.0, 100.0, 101.0, 102.0],
        scan_of_time = [1, 1, 1, 2, 2, 2],
        channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9,
    )
    mkc(term, tseg, tying) = CALs.TiedComponent(CALs.GainComponent(term, tseg, CALs.GlobalFrequency()), tying)
    model = CALs.StationGainModel(
        phase = (
            atmos = mkc(CALs.ConstantTerm(), CALs.PerScan(), CALs.SharedFeeds()),
            rl_phase = mkc(CALs.ConstantTerm(), CALs.GlobalTime(), CALs.FeedComponent(2)),
            mbd = mkc(CALs.Delay(), CALs.PerScan(), CALs.SharedFeeds()),
            rl_delay = mkc(CALs.Delay(), CALs.GlobalTime(), CALs.FeedComponent(2)),
            rate = mkc(CALs.Rate(), CALs.PerScan(), CALs.PerFeed()),
        ),
    )
    layout = CALs.plan_parameters(model, nant, geom)
    cf_sf, cf_g, d_sf, d_g, _ = layout.plans

    θ = zeros(layout.nθ)
    scans = (FR.StationScanDetections(D1, bl, feeds, 1), FR.StationScanDetections(D2, bl, feeds, 4))
    comps = (
        (cf_sf, :phase), (cf_g, :phase), (d_sf, :delay), (d_g, :delay), (layout.plans[5], :rate),
    )
    chi, ncomp = FR.solve_station_systems!(θ, scans, comps; ref_ant = ref)

    # The global R-L delay offset is recovered absolutely (cross hands pin it).
    δrec = [d_g.off1[a, 2, 1, 1] == 0 ? NaN : θ[d_g.off1[a, 2, 1, 1]] for a in 1:nant]
    for a in 1:nant
        @test δrec[a] ≈ δ[a] atol = 1.0e-13
    end
    # The global R-L phase offset is recovered up to the EVPA gauge (ref pinned).
    εrec = [cf_g.off1[a, 2, 1, 1] == 0 ? NaN : θ[cf_g.off1[a, 2, 1, 1]] for a in 1:nant]
    for a in 1:nant
        @test (εrec[a] - εrec[ref]) ≈ (ε[a] - ε[ref]) atol = 1.0e-9
    end

    # Reconstruction: recovered feed values reproduce EVERY observed delay,
    # including scan 2's QQ rows whose feed-2 is tied only through the global δ.
    recov_delay(a, feed, ti) = begin
        seg = d_sf.tseg_id[ti]
        cc = d_sf.off1[a, feed, seg, 1]
        gg = d_g.off1[a, feed, 1, 1]
        (cc == 0 ? 0.0 : θ[cc]) + (gg == 0 ? 0.0 : θ[gg])
    end
    worst = 0.0
    for (s, (D, ti)) in enumerate(((D1, 1), (D2, 4)))
        for (bi, (a, b)) in enumerate(bl), (p, (fa, fb)) in enumerate(feeds)
            D[bi, p].valid || continue
            model_d = recov_delay(a, fa, ti) - recov_delay(b, fb, ti)
            worst = max(worst, abs(model_d - D[bi, p].delay))
        end
    end
    @test worst < 1.0e-13
    @test ncomp == 1                          # global offset ties everything into one component
end

@testset "Stationize: robust rejection excises a closure-breaking false fringe" begin
    # A co-located telescope pair (e.g. the Onsala twins) can carry a coherent
    # crosstalk/tone fringe: ONE high-SNR baseline detection whose delay/rate is
    # wildly closure-inconsistent with every other baseline. Without rejection
    # the SNR²-weighted solve drags both stations' values; with it (the default)
    # the poisoned rows are excised and the truth is recovered exactly.
    rng = MersenneTwister(0x0E0F)
    nant = 6
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    feeds = [CALs.correlation_feed_pair(p) for p in pols]
    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)
    φ = 0.3 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, τ, ṙ, φ, 0.5; snr = 30.0)
    # Realistic measurement noise (CRB-scale, ∝ 1/snr): noiseless detections
    # close EXACTLY, which makes every robust scale (MAD) zero and correctly
    # disables rejection — the screen needs a genuine noise floor to cut against.
    for bi in eachindex(bl), p in eachindex(pols)
        d = D[bi, p]
        D[bi, p] = FR.Detection((
            d.delay + 1.0e-11 * randn(rng), d.rate + 1.0e-5 * randn(rng),
            d.phase + 0.01 * randn(rng), d.amp, d.snr, true,
        ))
    end
    poisoned = findfirst(==((5, 6)), bl)      # the "twin" baseline
    for p in eachindex(pols)
        D[poisoned, p] = FR.Detection((-690.0e-9, 4.7e-3, 1.3, 1.0, 80.0, true))
    end

    sol = FR.stationize_scan(D, bl, pols, nant; ref_ant = ref)
    for a in 1:nant, f in 1:2
        @test isapprox(sol.delay[a, f], τ[a, f] - τ[ref, 1]; atol = 1.0e-10)
        @test isapprox(sol.rate[a, f], ṙ[a, f] - ṙ[ref, f]; atol = 1.0e-4)
    end

    # Rejection disabled: the false fringe drags stations 5/6 off truth.
    sol0 = FR.stationize_scan(
        D, bl, pols, nant; ref_ant = ref,
        opts = FR.Stationization(reject_sigma = 0.0),
    )
    @test abs(sol0.delay[5, 1] - (τ[5, 1] - τ[ref, 1])) > 1.0e-9

    # Same through the model-driven pipeline path (solve_station_systems!).
    geom = CALs.DataGeometry(; times = [0.0, 1.0], channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9)
    model = CALs.StationGainModel(
        phase = (
            offset = CALs.TiedComponent(CALs.GainComponent(CALs.ConstantTerm(), CALs.PerScan(), CALs.GlobalFrequency()), CALs.PerFeed()),
            delay = CALs.TiedComponent(CALs.GainComponent(CALs.Delay(), CALs.PerScan(), CALs.GlobalFrequency()), CALs.PerFeed()),
            rate = CALs.TiedComponent(CALs.GainComponent(CALs.Rate(), CALs.PerScan(), CALs.GlobalFrequency()), CALs.PerFeed()),
        ),
    )
    layout = CALs.plan_parameters(model, nant, geom)
    cplan, dplan, rplan = layout.plans[1], layout.plans[2], layout.plans[3]
    θ = zeros(layout.nθ)
    scans = (FR.StationScanDetections(D, bl, feeds, 1),)
    _, _, nrej = FR.solve_station_systems!(
        θ, scans, ((cplan, :phase), (dplan, :delay), (rplan, :rate)); ref_ant = ref,
    )
    @test nrej >= 4                           # ≥ the 4 poisoned products (closure screen)
    for ant in 1:nant, feed in 1:2
        c = dplan.off1[ant, feed, 1, 1]
        c == 0 && continue
        @test isapprox(θ[c], τ[ant, feed] - τ[ref, 1]; atol = 1.0e-10)
    end
end
