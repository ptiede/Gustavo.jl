# Phase 4 — per-feed stationization with closure.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using Statistics: mean

const FR = Gustavo.Fringe
const CALs = Gustavo.Calibration

# Sign with which an unresolved source's cross-hand phase enters a product with
# feeds (fa, fb): +1 on (1,2), −1 on (2,1), 0 on parallel hands.
_cross_sign(fa, fb) = fa == fb ? 0 : (fa < fb ? 1 : -1)

# Build a noiseless per-baseline detection matrix from injected per-(station,
# feed) delays/rates/phases and a source cross-hand phase χ. Baselines are exact
# station differences, with ±χ added to the cross-hand phase rows.
#
# χ enters the phase rows exactly as a rigid −χ shift of every feed-2 station
# phase, so it is not separable from the instrumental inter-feed offset — the
# solve absorbs it rather than estimating it. `absorbed_phase` is the station
# phase the solve can actually recover from data built this way.
function inject_detections(bl_pairs, pol_products, τ, ṙ, φ, χ; snr = 100.0)
    nbl, npol = length(bl_pairs), length(pol_products)
    feeds = [CALs.correlation_feed_pair(p) for p in pol_products]
    D = Matrix{FR.Detection{Float64}}(undef, nbl, npol)
    for bi in 1:nbl, p in 1:npol
        a, b = bl_pairs[bi]
        fa, fb = feeds[p]
        cs = _cross_sign(fa, fb)
        delay = τ[a, fa] - τ[b, fb]
        rate = ṙ[a, fa] - ṙ[b, fb]
        phase = rem2pi(φ[a, fa] - φ[b, fb] + cs * χ, RoundNearest)
        D[bi, p] = FR.Detection{Float64}((delay, rate, phase, 1.0, snr, true))
    end
    return D
end

# The per-(station, feed) phase a χ-carrying injection is estimable up to: feed 1
# untouched, feed 2 shifted by −χ.
absorbed_phase(φ, χ) = hcat(φ[:, 1], φ[:, 2] .- χ)

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
        cs = _cross_sign(fa, fb)
        rd = max(rd, abs(det.delay - (sol.delay[a, fa] - sol.delay[b, fb])))
        rr = max(rr, abs(det.rate - (sol.rate[a, fa] - sol.rate[b, fb])))
        mph = sol.phase[a, fa] - sol.phase[b, fb]
        rp = max(rp, abs(rem2pi(det.phase - mph, RoundNearest)))
    end
    return (delay = rd, rate = rr, phase = rp)
end

all_baselines(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]

# A representative scan geometry for the row weights: a 2 GHz spanned band and a
# 300 s scan, as RMS spreads (uniform coverage ⇒ width/√12). These set the CRB σ
# of a delay and a rate, so they fix what `loss_scale` means; the weighted
# least-squares solution itself is invariant to them, since every row of a scan
# shares the same factor.
const SCAN_SPREAD = (freq_rms = 2.0e9 / sqrt(12), time_rms = 300.0 / sqrt(12))

detstack(D, bl, pols; kw...) = FR.detection_stack(D, bl, pols; SCAN_SPREAD..., kw...)

# The station model these tests are written against: one column per (station,
# feed) for each of delay, rate and constant phase, over a single scan.
function perfeed_scan_layout(nant)
    geom = CALs.DataGeometry(;
        times = [0.0, 1.0, 2.0], channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9,
    )
    mk(term) = CALs.TiedComponent(
        CALs.GainComponent(term, CALs.PerScan(), CALs.GlobalFrequency()), CALs.PerFeed(),
    )
    model = CALs.StationGainModel(
        phase = (
            offset = mk(CALs.ConstantTerm()), delay = mk(CALs.Delay()), rate = mk(CALs.Rate()),
        ),
    )
    return CALs.plan_parameters(model, nant, geom)
end

# The (station, feed) cells at least one usable detection touches — the cells a
# solve can say anything about.
function touched_cells(D, bl, pols, nant, opts)
    feeds = [CALs.correlation_feed_pair(p) for p in pols]
    t = falses(nant, 2)
    for bi in eachindex(bl), p in eachindex(pols)
        det = D[bi, p]
        (det.valid && det.snr >= opts.snr_min) || continue
        a, b = bl[bi]
        a == b && continue
        fa, fb = feeds[p]
        t[a, fa] = true
        t[b, fb] = true
    end
    return t
end

# Solve one scan on that model and read the θ columns back as (nant, 2) matrices.
# An untouched cell reads `NaN`: its θ stays 0, which would otherwise be
# indistinguishable from a solved zero correction.
function stationize(
        D, bl, pols, nant; ref_ant = 1, opts = FR.Stationization(),
        spreads = SCAN_SPREAD,
    )
    layout = perfeed_scan_layout(nant)
    cplan, dplan, rplan = layout.plans[1], layout.plans[2], layout.plans[3]
    θ = zeros(layout.nθ)
    scans = (FR.detection_stack(D, bl, pols; ti = 1, spreads...),)
    ncomp, ref_covered = FR.solve_station_systems!(
        θ, scans, ((cplan, :phase), (dplan, :delay), (rplan, :rate));
        ref_ant = ref_ant, opts = opts,
    )
    touched = touched_cells(D, bl, pols, nant, opts)
    readcols(plan) = [
        let c = plan_off1(plan)[a, f, 1, 1]
            (c == 0 || !touched[a, f]) ? NaN : θ[c]
        end
            for a in 1:nant, f in 1:2
    ]
    return (;
        delay = readcols(dplan), rate = readcols(rplan), phase = readcols(cplan),
        covered = touched, ncomp, ref_covered,
    )
end

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
    sol = stationize(D, bl, pols, nant; ref_ant = ref)

    @test size(sol.delay) == size(sol.rate) == size(sol.phase) == (nant, 2)

    # Cross hands tie the feeds: delay is one component, gauged at (ref, feed1).
    for a in 1:nant, f in 1:2
        @test isapprox(sol.delay[a, f], τ[a, f] - τ[ref, 1]; atol = 1.0e-18)
    end
    # Every product's rate row enters the system, so cross hands tie the feeds
    # here exactly as they do for delay: one component, gauged at (ref, feed1).
    for a in 1:nant, f in 1:2
        @test isapprox(sol.rate[a, f], ṙ[a, f] - ṙ[ref, 1]; atol = 1.0e-12)
    end
    # Phase: cross hands merge the feeds into one component, gauged at (ref, feed1).
    # The source's χ is absorbed as a rigid −χ shift of every feed-2 phase, so what
    # comes back is `absorbed_phase`, not φ itself.
    ψ = absorbed_phase(φ, χ)
    for a in 1:nant, f in 1:2
        @test isapprox(rem2pi(sol.phase[a, f] - (ψ[a, f] - ψ[ref, 1]), RoundNearest), 0.0; atol = 1.0e-10)
    end

    # Solution reconstructs every product (closure of the data).
    r = recon_residuals(D, sol, bl, pols)
    @test r.delay < 1.0e-15
    @test r.rate < 1.0e-12
    @test r.phase < 1.0e-9
end

@testset "Stationize: source cross-hand phase is absorbed, not fitted" begin
    # The source's cross-hand phase enters the phase system as a rigid shift of the
    # feed-2 block, so it is not separable from the instrumental inter-feed offset.
    # Injecting it must therefore leave every fitted row prediction and every feed-1
    # phase untouched, moving only the feed-2 phases — all of them by the same
    # constant. A solve carrying station phases alone represents this data exactly.
    rng = MersenneTwister(0x2C41)
    nant = 5
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)
    φ = 0.3 .* randn(rng, nant, 2)
    χ = 0.8

    sol0 = stationize(inject_detections(bl, pols, τ, ṙ, φ, 0.0), bl, pols, nant; ref_ant = ref)
    Dχ = inject_detections(bl, pols, τ, ṙ, φ, χ)
    solχ = stationize(Dχ, bl, pols, nant; ref_ant = ref)

    # Feed 1 is untouched.
    for a in 1:nant
        @test isapprox(rem2pi(solχ.phase[a, 1] - sol0.phase[a, 1], RoundNearest), 0.0; atol = 1.0e-10)
    end
    # Feed 2 moves rigidly: every station shifts by the SAME constant.
    shifts = [rem2pi(solχ.phase[a, 2] - sol0.phase[a, 2], RoundNearest) for a in 1:nant]
    for a in 1:nant
        @test isapprox(rem2pi(shifts[a] - shifts[1], RoundNearest), 0.0; atol = 1.0e-10)
    end
    # And the fit still reproduces every measured row, cross hands included.
    @test recon_residuals(Dχ, solχ, bl, pols).phase < 1.0e-9
    # Delay and rate carry no cross-hand source term at all.
    @test maximum(abs, filter(isfinite, solχ.delay .- sol0.delay)) < 1.0e-18
end

@testset "Stationize: inter-feed offset recovered from cross hands" begin
    rng = MersenneTwister(0x99)
    nant = 4
    ref = 1
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    # Feed-2 = feed-1 + a per-station inter-feed offset (delay and phase).
    τ1 = 2.0e-9 .* randn(rng, nant)
    rel_delay = 5.0e-9 .* randn(rng, nant)
    τ = hcat(τ1, τ1 .+ rel_delay)
    ṙ = zeros(nant, 2)
    φ1 = 0.2 .* randn(rng, nant)
    rel_phase = 0.4 .* randn(rng, nant)
    φ = hcat(φ1, φ1 .+ rel_phase)
    χ = -0.3

    D = inject_detections(bl, pols, τ, ṙ, φ, χ)
    sol = stationize(D, bl, pols, nant; ref_ant = ref)

    # Delay is ONE component (both feeds share the ref-feed-1 gauge via cross-hand
    # delay), so the per-station inter-feed offset is recovered *absolutely*.
    for a in 1:nant
        recovered = sol.delay[a, 2] - sol.delay[a, 1]
        @test isapprox(recovered, rel_delay[a]; atol = 1.0e-15)
    end
    @test sol.ncomp == 1                                # cross hands merge feeds
end

@testset "Stationize: parallel-hand triangle closure ≈ 0" begin
    rng = MersenneTwister(0x04)
    nant = 5
    bl = all_baselines(nant)
    pols = ["PP", "PQ", "QP", "QQ"]
    D = inject_detections(bl, pols, 1.0e-9 .* randn(rng, nant, 2), zeros(nant, 2), 0.25 .* randn(rng, nant, 2), 0.5)
    sol = stationize(D, bl, pols, nant; ref_ant = 1)
    # Parallel-hand products (PP=1, QQ=4) close exactly on noiseless data.
    for prod in (1, 4), obs in (:delay, :phase)
        res = FR.station_closure_residuals(D, bl, pols; observable = obs, product = prod)
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
    sol = stationize(D, bl, pols, nant; ref_ant = ref, opts = FR.Stationization(phase_rewrap_iters = 6))
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
    sol = stationize(D, bl, pols, nant; ref_ant = 1)
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
            D[bi, p] = FR.Detection{Float64}((d.delay, d.rate, d.phase, d.amp, 0.0, false))
        end
    end
    sol = stationize(D, bl, pols, nant; ref_ant = ref)
    @test sol.ncomp == 2                                # feeds NOT tied without cross hands
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
    sol_lo = stationize(D, bl, pols, nant; ref_ant = 1, opts = FR.Stationization(snr_min = 2.0))
    @test any(sol_lo.covered)
    sol_hi = stationize(D, bl, pols, nant; ref_ant = 1, opts = FR.Stationization(snr_min = 6.0))
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
    D = Matrix{FR.Detection{Float64}}(undef, length(bl), length(pols))
    for (bi, (a, b)) in enumerate(bl), (p, (fa, fb)) in enumerate(feeds)
        cs = _cross_sign(fa, fb)
        phase = rem2pi(φ[a, fa] - φ[b, fb] + cs * χ, RoundNearest)
        snr = is_chain(a, b) ? 200.0 : 100.0
        D[bi, p] = FR.Detection{Float64}((τ[a, fa] - τ[b, fb], ṙ[a, fa] - ṙ[b, fb], phase, 1.0, snr, true))
    end

    # At least one redundant baseline genuinely exceeds ±π in parallel hand.
    @test any(!is_chain(a, b) && abs(φ[a, 1] - φ[b, 1]) > π for (a, b) in bl)

    sol = stationize(D, bl, pols, nant; ref_ant = ref)

    # Model reproduces every measured phase (closure after correct unwrap).
    @test recon_residuals(D, sol, bl, pols).phase < 1.0e-6
    # The recovered absolute (unwrapped) station phases match the injected branch
    # — not a wrapped alias. Feed-1 is gauged at (ref, feed1).
    for a in 1:nant
        @test isapprox(sol.phase[a, 1] - sol.phase[ref, 1], φ[a, 1] - φ[ref, 1]; atol = 1.0e-6)
    end
end

@testset "Track-global inter-feed offset: stable across scans, weak scan inherits it" begin
    # Two scans share ONE stable inter-feed (feed-2 − feed-1) delay/phase offset per
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

    δ = 1.0e-9 .* randn(rng, nant)          # track-global inter-feed delay offset (feed2 − feed1)
    ε = 0.5 .* randn(rng, nant)             # track-global inter-feed phase offset
    Dc = [1.0e-9 .* randn(rng, nant), 1.0e-9 .* randn(rng, nant)]   # per-scan feed-common delay
    Φc = [0.3 .* randn(rng, nant), 0.3 .* randn(rng, nant)]          # per-scan feed-common phase
    χs = [0.6, -0.4]

    function scan_det(s; with_cross)
        D = Matrix{FR.Detection{Float64}}(undef, length(bl), length(pols))
        for (bi, (a, b)) in enumerate(bl), (p, (fa, fb)) in enumerate(feeds)
            cs = _cross_sign(fa, fb)
            valid = with_cross || cs == 0
            τa = Dc[s][a] + (fa == 2 ? δ[a] : 0.0)
            τb = Dc[s][b] + (fb == 2 ? δ[b] : 0.0)
            φa = Φc[s][a] + (fa == 2 ? ε[a] : 0.0)
            φb = Φc[s][b] + (fb == 2 ? ε[b] : 0.0)
            phase = rem2pi(φa - φb + cs * χs[s], RoundNearest)
            D[bi, p] = FR.Detection{Float64}((τa - τb, 0.0, phase, 1.0, 100.0, valid))
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
            rel_phase = mkc(CALs.ConstantTerm(), CALs.GlobalTime(), CALs.FeedComponent(2)),
            mbd = mkc(CALs.Delay(), CALs.PerScan(), CALs.SharedFeeds()),
            rel_delay = mkc(CALs.Delay(), CALs.GlobalTime(), CALs.FeedComponent(2)),
            rate = mkc(CALs.Rate(), CALs.PerScan(), CALs.PerFeed()),
        ),
    )
    layout = CALs.plan_parameters(model, nant, geom)
    cf_sf, cf_g, d_sf, d_g, _ = layout.plans

    θ = zeros(layout.nθ)
    scans = (detstack(D1, bl, pols; ti = 1), detstack(D2, bl, pols; ti = 4))
    comps = (
        (cf_sf, :phase), (cf_g, :phase), (d_sf, :delay), (d_g, :delay), (layout.plans[5], :rate),
    )
    ncomp, = FR.solve_station_systems!(θ, scans, comps; ref_ant = ref)

    # The track-global inter-feed delay offset is recovered absolutely (cross hands pin it).
    δrec = [plan_off1(d_g)[a, 2, 1, 1] == 0 ? NaN : θ[plan_off1(d_g)[a, 2, 1, 1]] for a in 1:nant]
    for a in 1:nant
        @test δrec[a] ≈ δ[a] atol = 1.0e-13
    end
    # The track-global inter-feed phase offset is recovered up to one additive
    # constant — the reference's own offset plus the source's cross-hand phase,
    # which share this column — so only differences against the reference are
    # checked.
    εrec = [plan_off1(cf_g)[a, 2, 1, 1] == 0 ? NaN : θ[plan_off1(cf_g)[a, 2, 1, 1]] for a in 1:nant]
    for a in 1:nant
        @test (εrec[a] - εrec[ref]) ≈ (ε[a] - ε[ref]) atol = 1.0e-9
    end

    # Reconstruction: recovered feed values reproduce EVERY observed delay,
    # including scan 2's QQ rows whose feed-2 is tied only through the global δ.
    recov_delay(a, feed, ti) = begin
        seg = d_sf.tseg_id[ti]
        cc = plan_off1(d_sf)[a, feed, seg, 1]
        gg = plan_off1(d_g)[a, feed, 1, 1]
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

# Robust loss on the PHASE system, where `w = snr²` is a true inverse variance
# (σ_phase = 1/snr rad) so `z = resid·√w` is genuinely in units of σ. The delay
# and rate systems weight rows by the same `snr²` but their residuals are in
# seconds / Hz, so a dimensionless `loss_scale` does not normalize them — see
# the `Stationization` docstring.
@testset "Robust loss: IRLS at the noise-model scale" begin
    # One clean scan with a single grossly inconsistent phase row.
    function poisoned_scan(nant; offset = 2.0, snr = 100.0, seed = 0x33)
        rng = MersenneTwister(seed)
        bl = all_baselines(nant)
        pols = ["PP", "PQ", "QP", "QQ"]
        τ = 1.0e-9 .* randn(rng, nant, 2)
        φ = 0.3 .* randn(rng, nant, 2)
        D = inject_detections(bl, pols, τ, zeros(nant, 2), φ, 0.0; snr = snr)
        # Poison one parallel-hand row: closure-breaking, so it cannot be
        # absorbed by any station solution.
        d = D[2, 1]
        D[2, 1] = FR.Detection{Float64}(
            (d.delay, d.rate, rem2pi(d.phase + offset, RoundNearest), d.amp, d.snr, true),
        )
        return (; bl, pols, τ, φ, D)
    end
    # Cross hands merge the feeds into one component, so BOTH feeds are gauged
    # against the reference's feed-1 node.
    pherr(sol, φ, ref, nant) =
        maximum(abs(rem2pi(sol.phase[a, f] - (φ[a, f] - φ[ref, 1]), RoundNearest))
                    for a in 1:nant, f in 1:2)

    @testset "the identity element is plain weighted least squares" begin
        s = poisoned_scan(5)
        # IRLS's first pass always solves at the untouched noise-model weights,
        # so capping the iteration at zero isolates it — `LeastSquares` must
        # reproduce it BIT-for-bit, not approximately.
        ls = stationize(
            s.D, s.bl, s.pols, 5; ref_ant = 1,
            opts = FR.Stationization(loss = FR.LeastSquares()),
        )
        wls = stationize(
            s.D, s.bl, s.pols, 5; ref_ant = 1,
            opts = FR.Stationization(loss = FR.SoftL1(), irls_iters = 0),
        )
        @test ls.phase == wls.phase
        @test ls.delay == wls.delay
        @test ls.rate == wls.rate
        # And it is genuinely non-robust: the outlier drags the solution.
        @test pherr(ls, s.φ, 1, 5) > 0.1
    end

    @testset "SoftL1 recovers truth through an outlier" begin
        s = poisoned_scan(5)
        rob = stationize(
            s.D, s.bl, s.pols, 5; ref_ant = 1,
            opts = FR.Stationization(loss = FR.SoftL1()),
        )
        ls = stationize(
            s.D, s.bl, s.pols, 5; ref_ant = 1,
            opts = FR.Stationization(loss = FR.LeastSquares()),
        )
        @test pherr(rob, s.φ, 1, 5) < 0.1
        @test pherr(rob, s.φ, 1, 5) < 0.2 * pherr(ls, s.φ, 1, 5)
        # Every station keeps a solution — a downweighted row is still a row, so
        # the graph stays connected and no feed splits off.
        @test all(rob.covered)
        @test rob.ncomp == 1
    end

    @testset "Cauchy suppresses a far outlier harder than SoftL1" begin
        s = poisoned_scan(5)
        errs = map((FR.SoftL1(), FR.Huber(), FR.Cauchy())) do loss
            sol = stationize(
                s.D, s.bl, s.pols, 5; ref_ant = 1, opts = FR.Stationization(; loss),
            )
            pherr(sol, s.φ, 1, 5)
        end
        @test all(<(0.1), errs)
        @test errs[3] < errs[1]                       # Cauchy redescends; SoftL1 does not
    end

    @testset "the effective threshold does not depend on the system size" begin
        # The property the noise-model scale buys, and the one a residual-fitted
        # MAD scale lacks: a scale estimated from fitted residuals shrinks with
        # the degrees of freedom, so the same outlier is cut differently on a
        # 3-station scan than on an 8-station one. Here the same outlier, in the
        # same σ units, must be suppressed to the same standard in both.
        for nant in (3, 8)
            s = poisoned_scan(nant)
            rob = stationize(
                s.D, s.bl, s.pols, nant; ref_ant = 1,
                opts = FR.Stationization(loss = FR.SoftL1()),
            )
            ls = stationize(
                s.D, s.bl, s.pols, nant; ref_ant = 1,
                opts = FR.Stationization(loss = FR.LeastSquares()),
            )
            @test pherr(rob, s.φ, 1, nant) < 0.1
            @test pherr(rob, s.φ, 1, nant) < 0.5 * pherr(ls, s.φ, 1, nant)
        end
    end

    @testset "the delay system is robust too, not just the phase system" begin
        # `w = snr²` alone is an inverse variance ONLY for phase (σ_φ = 1/snr
        # rad). A delay residual is in seconds, so it takes the band's lever arm
        # to express it in σ — without that the loss silently never fires here,
        # which is the whole reason `freq_rms`/`time_rms` are threaded through.
        rng = MersenneTwister(0x51)
        nant, ref = 5, 1
        bl = all_baselines(nant)
        pols = ["PP", "PQ", "QP", "QQ"]
        τ = 1.0e-9 .* randn(rng, nant, 2)
        φ = 0.3 .* randn(rng, nant, 2)
        D = inject_detections(bl, pols, τ, zeros(nant, 2), φ, 0.0)
        d = D[3, 1]
        D[3, 1] = FR.Detection{Float64}((50.0e-9, d.rate, d.phase, d.amp, d.snr, true))

        derr(sol) = maximum(abs(sol.delay[a, f] - (τ[a, f] - τ[ref, 1]))
                                for a in 1:nant, f in 1:2)
        rob = stationize(D, bl, pols, nant; ref_ant = ref,
                         opts = FR.Stationization(loss = FR.SoftL1()))
        ls = stationize(D, bl, pols, nant; ref_ant = ref,
                        opts = FR.Stationization(loss = FR.LeastSquares()))
        @test derr(ls) > 1.0e-9                       # dragged by the 50 ns outlier
        @test derr(rob) < 1.0e-11                     # suppressed
        @test derr(rob) < 1.0e-3 * derr(ls)
    end

    @testset "a robust loss without the scan geometry is refused" begin
        # Fail fast rather than leave every delay row at full weight while
        # reporting a robust solve.
        s = poisoned_scan(4)
        @test_throws ArgumentError stationize(
            s.D, s.bl, s.pols, 4; ref_ant = 1, spreads = (;),
            opts = FR.Stationization(loss = FR.SoftL1()),
        )
        @test_throws "needs the scan's RMS frequency spread" stationize(
            s.D, s.bl, s.pols, 4; ref_ant = 1, spreads = (;),
            opts = FR.Stationization(loss = FR.SoftL1()),
        )
        @test_throws "RMS time spread" stationize(
            s.D, s.bl, s.pols, 4; ref_ant = 1, spreads = (freq_rms = SCAN_SPREAD.freq_rms,),
            opts = FR.Stationization(loss = FR.SoftL1()),
        )
        # `LeastSquares` needs no geometry: scaling every weight in a system by
        # a common factor leaves the solution invariant (to rounding — the QR
        # runs on differently scaled numbers, so this is not bit-identical).
        bare = stationize(
            s.D, s.bl, s.pols, 4; ref_ant = 1, spreads = (;),
            opts = FR.Stationization(loss = FR.LeastSquares()),
        ).delay
        scaled = stationize(
            s.D, s.bl, s.pols, 4; ref_ant = 1,
            opts = FR.Stationization(loss = FR.LeastSquares()),
        ).delay
        @test bare ≈ scaled atol = 1.0e-18
    end

    @testset "multi-scan: an outlier in one scan does not leak into another" begin
        # No test exercised multi-scan robustness before this one. Two scans
        # share one model; the poison sits in scan 2 only, so scan 1 must come
        # back exactly as if solved alone.
        nant, ref = 5, 1
        s1 = poisoned_scan(nant; offset = 0.0, seed = 0x41)     # clean
        s2 = poisoned_scan(nant; offset = 2.0, seed = 0x42)     # poisoned
        # `scan_of_time` is what gives `PerScan` two distinct segments, so the
        # two scans land in separate θ columns.
        geom = CALs.DataGeometry(;
            times = [0.0, 1.0, 100.0, 101.0], scan_of_time = [1, 1, 2, 2],
            channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9,
        )
        model = CALs.StationGainModel(
            phase = (
                offset = CALs.TiedComponent(
                    CALs.GainComponent(CALs.ConstantTerm(), CALs.PerScan(), CALs.GlobalFrequency()),
                    CALs.PerFeed(),
                ),
            ),
        )
        layout = CALs.plan_parameters(model, nant, geom)
        cplan = layout.plans[1]
        opts = FR.Stationization(loss = FR.SoftL1())

        θ = zeros(layout.nθ)
        FR.solve_station_systems!(
            θ, (detstack(s1.D, s1.bl, s1.pols; ti = 1),
                detstack(s2.D, s2.bl, s2.pols; ti = 3)),
            ((cplan, :phase),); ref_ant = ref, opts = opts,
        )
        θ1 = zeros(layout.nθ)
        FR.solve_station_systems!(
            θ1, (detstack(s1.D, s1.bl, s1.pols; ti = 1),),
            ((cplan, :phase),); ref_ant = ref, opts = opts,
        )
        # Scan 1's columns are untouched by scan 2's outlier: with a per-scan
        # model the systems are block-diagonal, and the loss keeps them that way
        # because it reweights rows rather than pooling a scale across them.
        for a in 1:nant, f in 1:2
            c = plan_off1(cplan)[a, f, 1, 1]
            c == 0 && continue
            @test θ[c] ≈ θ1[c] atol = 1.0e-12
        end
        # Scan 2 still recovers its own truth despite carrying the outlier.
        for a in 1:nant, f in 1:2
            c = plan_off1(cplan)[a, f, 2, 1]
            c == 0 && continue
            @test abs(rem2pi(θ[c] - (s2.φ[a, f] - s2.φ[ref, 1]), RoundNearest)) < 0.1
        end
    end
end
