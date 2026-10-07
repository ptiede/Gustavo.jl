# Phase 4 — per-feed stationization with closure.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using Statistics: mean

const FR = Gustavo.Fring
const CALs = Gustavo.Calibration

# Build a noiseless per-baseline detection matrix from injected per-(station,
# feed) delays/rates (and optionally phases). Baselines are exact station
# differences.
function inject_detections(bl_pairs, pol_products, τ, ṙ; φ = zero(τ), snr = 100.0, pfa = 0.0)
    nbl, npol = length(bl_pairs), length(pol_products)
    feeds = collect(pol_products)
    D = Matrix{FR.Detection{Float64}}(undef, nbl, npol)
    for bi in 1:nbl, p in 1:npol
        a, b = bl_pairs[bi]
        fa, fb = feeds[p]
        delay = τ[a, fa] - τ[b, fb]
        rate = ṙ[a, fa] - ṙ[b, fb]
        phase = rem2pi(φ[a, fa] - φ[b, fb], RoundNearest)
        D[bi, p] = FR.Detection{Float64}((delay, rate, phase, 1.0, snr, pfa, true))
    end
    return D
end

# Max |measured − model-from-solution| over all valid, non-auto baselines.
function recon_residuals(D, sol, bl_pairs, pol_products)
    feeds = collect(pol_products)
    rd = rr = 0.0
    for bi in eachindex(bl_pairs), p in eachindex(pol_products)
        det = D[bi, p]
        det.valid || continue
        a, b = bl_pairs[bi]
        a == b && continue
        fa, fb = feeds[p]
        rd = max(rd, abs(det.delay - (sol.delay[a, fa] - sol.delay[b, fb])))
        rr = max(rr, abs(det.rate - (sol.rate[a, fa] - sol.rate[b, fb])))
    end
    return (delay = rd, rate = rr)
end

all_baselines(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]

# A representative scan geometry for the row weights: a 2 GHz spanned band and a
# 300 s scan, as RMS spreads (uniform coverage ⇒ width/√12). These set the CRB σ
# of a delay and a rate, so they fix what `loss_scale` means; the weighted
# least-squares solution itself is invariant to them, since every row of a scan
# shares the same factor.
const SCAN_SPREAD = (freq_rms = 2.0e9 / sqrt(12), time_rms = 300.0 / sqrt(12))

# CRB σ of one delay / rate detection at `snr` on that geometry.
delay_sigma(snr) = inv(2π * SCAN_SPREAD.freq_rms * snr)
rate_sigma(snr) = inv(2π * SCAN_SPREAD.time_rms * snr)

# Station `i` of these fixtures is named "A$i"; `STATIONS` numbers them so.
const STATIONS = ["A$i" for i in 1:64]
named(bl) = [(STATIONS[a], STATIONS[b]) for (a, b) in bl]
detstack(D, bl, pols; kw...) = FR.detection_stack(D, named(bl), pols; SCAN_SPREAD..., kw...)
solve_named!(θ, scans, comps; kw...) = FR.solve_station_systems!(θ, scans, comps, STATIONS; kw...)

# The station model these tests are written against: one column per (station,
# feed) for each of delay and rate, over a single scan.
function perfeed_scan_layout(nant)
    geom = CALs.DataGeometry(; nfeed = 2,
        times = [0.0, 1.0, 2.0], channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9,
    )
    mk(term) = CALs.GainComponent(term; Ti = CALs.PerScan(), Frequency = CALs.GlobalFrequency(), Feed = CALs.PerFeed())
    model = CALs.GainModel(phase = (delay = mk(CALs.Delay()), rate = mk(CALs.Rate())))
    return CALs.plan_parameters(model, nant, geom)
end

# The (station, feed) cells at least one usable detection touches — the cells a
# solve can say anything about.
function touched_cells(D, bl, pols, nant, opts)
    feeds = collect(pols)
    t = falses(nant, 2)
    for bi in eachindex(bl), p in eachindex(pols)
        det = D[bi, p]
        (det.valid && det.pfa <= opts.pfa_max) || continue
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
        D, bl, pols, nant; gauge = PinAntenna(1), opts = FR.Stationization(),
        spreads = SCAN_SPREAD,
    )
    layout = perfeed_scan_layout(nant)
    dplan, rplan = layout.plans[1], layout.plans[2]
    θ = zeros(layout.nθ)
    scans = (FR.detection_stack(D, named(bl), pols; ti = 1, spreads...),)
    ncomp, ref_covered = solve_named!(
        θ, scans, ((dplan, :delay), (rplan, :rate)); gauge, opts,
    )
    touched = touched_cells(D, bl, pols, nant, opts)
    readcols(plan) = [
        let c = plan_off1(plan)[a, f, 1, 1]
            (c == 0 || !touched[a, f]) ? NaN : θ[c]
        end
            for a in 1:nant, f in 1:2
    ]
    return (;
        delay = readcols(dplan), rate = readcols(rplan),
        covered = touched, ncomp, ref_covered,
    )
end

@testset "Stationize: per-(station,feed) recovery up to gauge" begin
    rng = MersenneTwister(0x5712)
    nant = 5
    ref = 1
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)

    D = inject_detections(bl, pols, τ, ṙ)
    sol = stationize(D, bl, pols, nant; gauge = PinAntenna(ref))

    @test size(sol.delay) == size(sol.rate) == (nant, 2)

    # Products relating different feeds tie the feeds: one component, gauged at
    # (ref, feed1), for delay and rate alike.
    for a in 1:nant, f in 1:2
        @test isapprox(sol.delay[a, f], τ[a, f] - τ[ref, 1]; atol = 1.0e-18)
        @test isapprox(sol.rate[a, f], ṙ[a, f] - ṙ[ref, 1]; atol = 1.0e-12)
    end

    # Solution reconstructs every product (closure of the data).
    r = recon_residuals(D, sol, bl, pols)
    @test r.delay < 1.0e-15
    @test r.rate < 1.0e-12
end

@testset "Stationize: the solve runs in the options' element type" begin
    @test FR.Stationization() isa FR.Stationization{Float64, FR.SoftL1}
    o32 = FR.Stationization(eltype = Float32, pfa_max = 1.0e-2, loss = FR.Huber())
    @test o32 isa FR.Stationization{Float32, FR.Huber}
    @test o32.pfa_max === 1.0f-2
    @test o32 == FR.Stationization{Float32}(pfa_max = 1.0e-2, loss = FR.Huber())

    rng = MersenneTwister(0x5712)
    nant = 5
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    D = inject_detections(bl, pols, 1.0e-9 .* randn(rng, nant, 2), 1.0e-3 .* randn(rng, nant, 2))
    D32 = map(d -> FR.Detection{Float32}(Tuple(d)), D)
    s64 = stationize(D, bl, pols, nant)
    for d in (D, D32)
        s32 = stationize(d, bl, pols, nant; opts = FR.Stationization(eltype = Float32))
        @test s32.ref_covered == s64.ref_covered
        @test s32.delay ≈ s64.delay atol = 1.0e-15
        @test s32.rate ≈ s64.rate atol = 1.0e-9
    end

    layout = perfeed_scan_layout(nant)
    scans = (FR.detection_stack(D32, named(bl), pols; ti = 1, SCAN_SPREAD...),)
    slot = Dict(n => i for (i, n) in pairs(STATIONS))
    for (T, kind, plan) in ((Float32, :delay, layout.plans[1]), (Float64, :rate, layout.plans[2]))
        r = @inferred FR._solve_kind_cols!(
            zeros(layout.nθ), scans, [plan], slot, PinAntenna(1),
            FR.Stationization(eltype = T), Val(kind),
        )
        @test r[3] isa Dict{Int, T}
    end
end

@testset "Stationize: inter-feed offset recovered from products relating different feeds" begin
    rng = MersenneTwister(0x99)
    nant = 4
    ref = 1
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    # Feed-2 = feed-1 + a per-station inter-feed delay offset.
    τ1 = 2.0e-9 .* randn(rng, nant)
    rel_delay = 5.0e-9 .* randn(rng, nant)
    τ = hcat(τ1, τ1 .+ rel_delay)

    D = inject_detections(bl, pols, τ, zeros(nant, 2))
    sol = stationize(D, bl, pols, nant; gauge = PinAntenna(ref))

    # Delay is ONE component (both feeds share the ref-feed-1 gauge), so the
    # per-station inter-feed offset is recovered *absolutely*.
    for a in 1:nant
        recovered = sol.delay[a, 2] - sol.delay[a, 1]
        @test isapprox(recovered, rel_delay[a]; atol = 1.0e-15)
    end
    @test sol.ncomp == 1
end

@testset "Stationize: same-feed triangle closure ≈ 0" begin
    rng = MersenneTwister(0x04)
    nant = 5
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    D = inject_detections(bl, pols, 1.0e-9 .* randn(rng, nant, 2), zeros(nant, 2); φ = 0.25 .* randn(rng, nant, 2))
    # Same-feed products (11 = 1, 22 = 4) close exactly on noiseless data.
    for prod in (1, 4), obs in (:delay, :phase)
        res = FR.station_closure_residuals(detstack(D, bl, pols; ti = 1); observable = obs, feeds = pols[prod])
        @test !isempty(res)
        @test maximum(abs, res) < 1.0e-9
    end
end

@testset "Stationize: disconnected array (two islands)" begin
    rng = MersenneTwister(0x01)
    # Antennas 1-3 and 4-6 with NO inter-island baselines.
    nant = 6
    bl = vcat(all_baselines(3), [(a, b) for a in 4:6 for b in (a + 1):6])
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, τ, ṙ)
    sol = stationize(D, bl, pols, nant; gauge = PinAntenna(1))
    @test sol.ncomp == 2                                # one component per island (feeds tied within each)
    # Reconstruction is gauge-invariant → residuals close within each island.
    r = recon_residuals(D, sol, bl, pols)
    @test r.delay < 1.0e-15
    @test r.rate < 1.0e-12
end

@testset "Stationize: same-feed-only fallback (different-feed products undetected)" begin
    rng = MersenneTwister(0x07)
    nant = 5
    ref = 1
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    τ = 1.0e-9 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, τ, zeros(nant, 2))
    feeds = collect(pols)
    for bi in eachindex(bl), p in eachindex(pols)
        if feeds[p][1] != feeds[p][2]
            d = D[bi, p]
            D[bi, p] = FR.Detection{Float64}((d.delay, d.rate, d.phase, d.amp, 0.0, 1.0, false))
        end
    end
    sol = stationize(D, bl, pols, nant; gauge = PinAntenna(ref))
    @test sol.ncomp == 2                                # feeds NOT tied
    # Each feed gauged independently; delay relative to that feed's ref.
    for a in 1:nant, f in 1:2
        @test isapprox(sol.delay[a, f], τ[a, f] - τ[ref, f]; atol = 1.0e-15)
    end
end

@testset "Stationize: coverage does not depend on the reference antenna" begin
    # A scan the reference sits out. The remaining stations form a perfectly
    # well-determined system, and the solve must report them covered — coverage
    # asks whether a station is CONSTRAINED, not whether it is linked to the
    # reference, so an absent reference costs the scan nothing.
    nant = 5
    absent_ref = 5                                  # observes no baseline below
    bl = all_baselines(4)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rng = MersenneTwister(0x9c31)
    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, τ, ṙ)

    sol = stationize(D, bl, pols, nant; gauge = PinAntenna(absent_ref))
    @test sol.ref_covered == Set((STATIONS[a], f, 1) for a in 1:4 for f in 1:2)
    @test !any(c -> first(c) == STATIONS[absent_ref], sol.ref_covered)     # never fabricated
    # The solution is exact despite the missing reference: only its gauge is
    # arbitrary, and a gauge cancels on every baseline of its own component.
    r = recon_residuals(D, sol, bl, pols)
    @test r.delay < 1.0e-20
    @test r.rate < 1.0e-15

    # Pin-invariance: the same data referenced to a present station reports the
    # same coverage, so a reporting reference can be chosen after the solve.
    @test stationize(D, bl, pols, nant; gauge = PinAntenna(1)).ref_covered == sol.ref_covered
    @test stationize(D, bl, pols, nant; gauge = PinAntenna(3)).ref_covered == sol.ref_covered
end

@testset "Stationize: disconnected islands are each covered on their own gauge" begin
    # Two mutually unlinked subarrays in one scan. Each carries its own additive
    # zero, which is unobservable and cancels within the island, so both are
    # solved and both are covered — including the island holding no reference.
    nant = 5
    bl = [(1, 2), (1, 3), (2, 3), (4, 5)]
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rng = MersenneTwister(0x4a17)
    τ = 1.0e-9 .* randn(rng, nant, 2)
    ṙ = 1.0e-3 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, τ, ṙ)

    sol = stationize(D, bl, pols, nant; gauge = PinAntenna(1))
    @test sol.ref_covered == Set((STATIONS[a], f, 1) for a in 1:nant for f in 1:2)
    r = recon_residuals(D, sol, bl, pols)
    @test r.delay < 1.0e-20
    @test r.rate < 1.0e-15
end

@testset "Stationize: a reference-free component pins its best-observed node" begin
    # With no reference node to pin, the gauge falls to the node carrying the most
    # row weight, so the zero sits on the best-observed station rather than
    # wherever the node numbering happens to start.
    nant = 5
    absent_ref = 5
    bl = all_baselines(4)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rng = MersenneTwister(0x1d05)
    τ = 1.0e-9 .* randn(rng, nant, 2)
    D0 = inject_detections(bl, pols, τ, zeros(nant, 2); snr = 100.0)

    # The pin is imposed as a constraint row, so the gauge zero is zero to solver
    # precision against a ~ns signal, not bit-exactly.
    gauge_zero(m) = argmin(abs.(@view m[1:4, :]))

    # Uniform weights: the tie resolves to the lowest node, i.e. station 1 feed 1.
    uniform = stationize(D0, bl, pols, nant; gauge = PinAntenna(absent_ref))
    @test gauge_zero(uniform.delay) == CartesianIndex(1, 1)
    @test abs(uniform.delay[1, 1]) < 1.0e-15

    # Station 3 observed far more strongly: the pin moves to it.
    D = copy(D0)
    for bi in eachindex(bl), p in eachindex(pols)
        3 in bl[bi] || continue
        d = D[bi, p]
        D[bi, p] = FR.Detection{Float64}((d.delay, d.rate, d.phase, d.amp, 1.0e4, 0.0, true))
    end
    strong = stationize(D, bl, pols, nant; gauge = PinAntenna(absent_ref))
    @test gauge_zero(strong.delay) == CartesianIndex(3, 1)
    @test abs(strong.delay[3, 1]) < 1.0e-15
    # Regauging is all that changed: baseline differences are untouched.
    @test recon_residuals(D, strong, bl, pols).delay < 1.0e-20
end

@testset "Stationize: pfa_max decides which detections are real" begin
    nant = 4
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    τ = 1.0e-9 .* randn(MersenneTwister(0x33), nant, 2)
    D = inject_detections(bl, pols, τ, zeros(nant, 2); snr = 5.0, pfa = 1.0e-3)
    # Every detection at PFA 1e-3: a looser threshold accepts them and the solve
    # covers the array; a stricter one accepts nothing, so no station is
    # calibrated even though every row still entered the system.
    sol_lo = stationize(D, bl, pols, nant; gauge = PinAntenna(1), opts = FR.Stationization(pfa_max = 1.0e-2))
    @test any(sol_lo.covered)
    sol_hi = stationize(D, bl, pols, nant; gauge = PinAntenna(1), opts = FR.Stationization(pfa_max = 1.0e-4))
    @test !any(sol_hi.covered)
    @test all(isnan, sol_hi.delay)
end

@testset "Stationize: an unconstrained station's θ is identity" begin
    nant = 4
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rng = MersenneTwister(0x212)
    τ, ṙ = 1.0e-9 .* randn(rng, nant, 2), 1.0e-3 .* randn(rng, nant, 2)
    D = inject_detections(bl, pols, τ, ṙ)
    # Station 4's detections are all noise peaks: rows in the system, none accepted.
    for bi in eachindex(bl), p in eachindex(pols)
        4 in bl[bi] && (D[bi, p] = merge(D[bi, p], (; pfa = 1.0)))
    end
    layout = perfeed_scan_layout(nant)
    θ = zeros(layout.nθ)
    plans = ((layout.plans[1], :delay), (layout.plans[2], :rate))
    _, covered = solve_named!(θ, (FR.detection_stack(D, named(bl), pols; ti = 1, SCAN_SPREAD...),), plans; gauge = PinAntenna(1))
    @test covered == Set((STATIONS[a], f, 1) for a in 1:3 for f in 1:2)
    for (plan, _) in plans, f in 1:2
        @test θ[plan_off1(plan)[4, f, 1, 1]] == 0
        @test any(a -> θ[plan_off1(plan)[a, f, 1, 1]] != 0, 1:3)
    end
end

@testset "Stationize: a feed only rejected rows touch is unconstrained" begin
    nant, ref, lone = 4, 1, 4
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rng = MersenneTwister(0x04e1)
    τ = 1.0e-9 .* randn(rng, nant)
    δ = 1.0e-10 .* randn(rng, nant)
    ṙ = 1.0e-3 .* randn(rng, nant)
    D = Matrix{FR.Detection{Float64}}(undef, length(bl), length(pols))
    for (bi, (a, b)) in pairs(bl), (p, (fa, fb)) in pairs(pols)
        delay = τ[a] + (fa == 2) * δ[a] - τ[b] - (fb == 2) * δ[b]
        D[bi, p] = FR.Detection{Float64}((delay, ṙ[a] - ṙ[b], 0.0, 1.0, 100.0, 0.0, true))
        if (a == lone && fa == 2) || (b == lone && fb == 2)
            D[bi, p] = FR.Detection{Float64}((delay + 3.0e-7, 0.01, 2.0, 1.0, 4.0, 1.0, true))
        end
    end
    geom = CALs.DataGeometry(; nfeed = 2, times = [0.0, 1.0, 2.0], channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9)
    mk(term, tying) = CALs.GainComponent(term; Ti = CALs.PerScan(), Frequency = CALs.GlobalFrequency(), Feed = tying)
    model = CALs.GainModel(
        phase = (
            mbd = mk(CALs.Delay(), CALs.SharedFeeds()),
            rel_delay = mk(CALs.Delay(), CALs.SingleFeed(2)),
            rate = mk(CALs.Rate(), CALs.SharedFeeds()),
        ),
    )
    layout = CALs.plan_parameters(model, nant, geom)
    mbd, rel, rate = layout.plans
    θ = zeros(layout.nθ)
    comps = ((mbd, :delay), (rel, :delay), (rate, :rate))
    _, covered = solve_named!(θ, (detstack(D, bl, pols; ti = 1),), comps; gauge = PinAntenna(ref))

    col(plan, a, f) = θ[plan_off1(plan)[a, f, 1, 1]]
    @test covered == Set((STATIONS[a], f, 1) for a in 1:nant for f in 1:2 if (a, f) != (lone, 2))
    @test col(rel, lone, 2) == 0
    for a in 1:nant
        a == lone || @test col(rel, a, 2) ≈ δ[a] atol = 1.0e-12
    end
    @test col(mbd, lone, 1) ≈ τ[lone] - τ[ref] atol = 1.0e-12
    @test col(rate, lone, 1) ≈ ṙ[lone] - ṙ[ref] atol = 1.0e-9
end

@testset "Stationize: rejected rows constrain but never connect" begin
    nant = 4
    bl = all_baselines(nant)
    pols = [(1, 1), (2, 2)]
    τ = 1.0e-9 .* randn(MersenneTwister(0x51), nant, 2)
    # One weak baseline among strong ones: it must not extend coverage, and the
    # accepted detections must still solve exactly.
    D = inject_detections(bl, pols, τ, zeros(nant, 2); snr = 100.0)
    strong = stationize(D, bl, pols, nant; gauge = PinAntenna(1))
    Dw = copy(D)
    for p in eachindex(pols)
        d = Dw[1, p]
        Dw[1, p] = FR.Detection{Float64}((d.delay, d.rate, d.phase, d.amp, 3.0, 0.5, true))
    end
    weak = stationize(Dw, bl, pols, nant; gauge = PinAntenna(1))
    # Coverage is unchanged: the remaining strong baselines already reach every
    # station, and the weak row adds no connectivity of its own.
    @test weak.covered == strong.covered
    # And a weak row is ~1e6x downweighted, so it cannot move the fit measurably.
    @test maximum(abs, filter(isfinite, weak.delay .- strong.delay)) < 1.0e-15
end

@testset "Stationize: delay does not depend on feed labels" begin
    # Swapping one station's receptor order relabels its feeds in every product it
    # takes part in. The physical feeds are unchanged, so each must solve to the
    # same delay under either label.
    rng = MersenneTwister(0xfeed)
    nant, ref, swapped = 5, 1, 3
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    snr = 30.0
    D = inject_detections(bl, pols, 1.0e-9 .* randn(rng, nant, 2), 1.0e-3 .* randn(rng, nant, 2); snr)
    D = map(D) do d
        merge(d, (; delay = d.delay + 2 * delay_sigma(snr) * randn(rng), rate = d.rate + 2 * rate_sigma(snr) * randn(rng)))
    end

    relabel(a, f) = a == swapped ? 3 - f : f
    Dr = similar(D)
    for (bi, (a, b)) in pairs(bl), (p, (fa, fb)) in pairs(pols)
        Dr[bi, p] = D[bi, findfirst(==((relabel(a, fa), relabel(b, fb))), pols)]
    end

    sol = stationize(D, bl, pols, nant; gauge = PinAntenna(ref))
    solr = stationize(Dr, bl, pols, nant; gauge = PinAntenna(ref))
    @test recon_residuals(D, sol, bl, pols).delay > 1.0e-13          # noisy: not a trivial exact fit
    for a in 1:nant, f in 1:2
        @test solr.delay[a, f] ≈ sol.delay[a, relabel(a, f)] atol = 1.0e-20
        @test solr.rate[a, f] ≈ sol.rate[a, relabel(a, f)] atol = 1.0e-15
    end
    @test solr.ref_covered == sol.ref_covered
end

@testset "Track-global inter-feed offset: stable across scans, weak scan inherits it" begin
    # Two scans share ONE stable inter-feed (feed-2 − feed-1) delay offset per
    # station; the per-scan feed-common delays differ. Scan 2 has NO detections
    # relating different feeds (the weak case that splits into ncomp=2 per-scan).
    # The global SingleFeed(2) × GlobalTime offset, pinned by scan 1's
    # different-feed products, must tie scan 2's feeds too.
    rng = MersenneTwister(0x5EED)
    nant = 4
    ref = 1
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    feeds = collect(pols)

    δ = 1.0e-9 .* randn(rng, nant)          # track-global inter-feed delay offset (feed2 − feed1)
    Dc = [1.0e-9 .* randn(rng, nant), 1.0e-9 .* randn(rng, nant)]   # per-scan feed-common delay

    function scan_det(s; mixed_feeds)
        D = Matrix{FR.Detection{Float64}}(undef, length(bl), length(pols))
        for (bi, (a, b)) in enumerate(bl), (p, (fa, fb)) in enumerate(feeds)
            valid = mixed_feeds || fa == fb
            τa = Dc[s][a] + (fa == 2 ? δ[a] : 0.0)
            τb = Dc[s][b] + (fb == 2 ? δ[b] : 0.0)
            D[bi, p] = FR.Detection{Float64}((τa - τb, 0.0, 0.0, 1.0, 100.0, valid ? 0.0 : 1.0, valid))
        end
        return D
    end
    D1 = scan_det(1; mixed_feeds = true)
    D2 = scan_det(2; mixed_feeds = false)     # weak scan: different-feed products undetected

    geom = CALs.DataGeometry(; nfeed = 2,
        times = [0.0, 1.0, 2.0, 100.0, 101.0, 102.0],
        scan_of_time = [1, 1, 1, 2, 2, 2],
        channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9,
    )
    mkc(term, tseg, tying) = CALs.GainComponent(term; Ti = tseg, Frequency = CALs.GlobalFrequency(), Feed = tying)
    model = CALs.GainModel(
        phase = (
            mbd = mkc(CALs.Delay(), CALs.PerScan(), CALs.SharedFeeds()),
            rel_delay = mkc(CALs.Delay(), CALs.GlobalTime(), CALs.SingleFeed(2)),
            rate = mkc(CALs.Rate(), CALs.PerScan(), CALs.PerFeed()),
        ),
    )
    layout = CALs.plan_parameters(model, nant, geom)
    d_sf, d_g, rplan = layout.plans

    θ = zeros(layout.nθ)
    scans = (detstack(D1, bl, pols; ti = 1), detstack(D2, bl, pols; ti = 4))
    comps = ((d_sf, :delay), (d_g, :delay), (rplan, :rate))
    ncomp, = solve_named!(θ, scans, comps; gauge = PinAntenna(ref))

    # The track-global inter-feed delay offset is recovered absolutely.
    δrec = [plan_off1(d_g)[a, 2, 1, 1] == 0 ? NaN : θ[plan_off1(d_g)[a, 2, 1, 1]] for a in 1:nant]
    for a in 1:nant
        @test δrec[a] ≈ δ[a] atol = 1.0e-13
    end

    # Reconstruction: recovered feed values reproduce EVERY observed delay,
    # including scan 2's feed-2 rows, tied to feed 1 only through the global δ.
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
    # One gauged component per scan: the global offset ties each scan's feeds
    # but not the scans' levels, so the reference reads 0 in both.
    @test ncomp == 2
    for ti in (1, 4)
        @test θ[plan_off1(d_sf)[ref, 1, d_sf.tseg_id[ti], 1]] == 0
    end
end

# Robust loss on the delay system. Row weights are the CRB inverse variance on
# the scan geometry, so `z = resid·√w` is in units of σ and the outliers and
# tolerances below are stated in multiples of `delay_sigma`.
@testset "Robust loss: IRLS at the noise-model scale" begin
    # One clean scan with a single grossly inconsistent delay row, `offset` σ off.
    function poisoned_scan(nant; offset = 200.0, snr = 100.0, seed = 0x33)
        rng = MersenneTwister(seed)
        bl = all_baselines(nant)
        pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
        τ = 1.0e-9 .* randn(rng, nant, 2)
        D = inject_detections(bl, pols, τ, zeros(nant, 2); snr)
        # Poison one same-feed row: closure-breaking, so it cannot be absorbed
        # by any station solution.
        D[2, 1] = merge(D[2, 1], (; delay = D[2, 1].delay + offset * delay_sigma(snr)))
        return (; bl, pols, τ, D, σ = delay_sigma(snr))
    end
    # Products relating different feeds merge the feeds into one component, so
    # BOTH feeds are gauged against the reference's feed-1 node. Error in σ.
    derr(sol, s, ref, nant) =
        maximum(abs(sol.delay[a, f] - (s.τ[a, f] - s.τ[ref, 1])) for a in 1:nant, f in 1:2) / s.σ

    @testset "the identity element is plain weighted least squares" begin
        s = poisoned_scan(5)
        # IRLS's first pass always solves at the untouched noise-model weights,
        # so capping the iteration at zero isolates it — `LeastSquares` must
        # reproduce it BIT-for-bit, not approximately.
        ls = stationize(
            s.D, s.bl, s.pols, 5; gauge = PinAntenna(1),
            opts = FR.Stationization(loss = FR.LeastSquares()),
        )
        wls = stationize(
            s.D, s.bl, s.pols, 5; gauge = PinAntenna(1),
            opts = FR.Stationization(loss = FR.SoftL1(), irls_iters = 0),
        )
        @test ls.delay == wls.delay
        @test ls.rate == wls.rate
        # And it is genuinely non-robust: the outlier drags the solution.
        @test derr(ls, s, 1, 5) > 10
    end

    @testset "SoftL1 recovers truth through an outlier" begin
        s = poisoned_scan(5)
        rob = stationize(
            s.D, s.bl, s.pols, 5; gauge = PinAntenna(1),
            opts = FR.Stationization(loss = FR.SoftL1()),
        )
        ls = stationize(
            s.D, s.bl, s.pols, 5; gauge = PinAntenna(1),
            opts = FR.Stationization(loss = FR.LeastSquares()),
        )
        @test derr(rob, s, 1, 5) < 10
        @test derr(rob, s, 1, 5) < 0.2 * derr(ls, s, 1, 5)
        # Every station keeps a solution — a downweighted row is still a row, so
        # the graph stays connected and no feed splits off.
        @test all(rob.covered)
        @test rob.ncomp == 1
    end

    @testset "Cauchy suppresses a far outlier harder than SoftL1" begin
        s = poisoned_scan(5)
        errs = map((FR.SoftL1(), FR.Huber(), FR.Cauchy())) do loss
            sol = stationize(
                s.D, s.bl, s.pols, 5; gauge = PinAntenna(1), opts = FR.Stationization(; loss),
            )
            derr(sol, s, 1, 5)
        end
        @test all(<(10), errs)
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
                s.D, s.bl, s.pols, nant; gauge = PinAntenna(1),
                opts = FR.Stationization(loss = FR.SoftL1()),
            )
            ls = stationize(
                s.D, s.bl, s.pols, nant; gauge = PinAntenna(1),
                opts = FR.Stationization(loss = FR.LeastSquares()),
            )
            @test derr(rob, s, 1, nant) < 10
            @test derr(rob, s, 1, nant) < 0.5 * derr(ls, s, 1, nant)
        end
    end

    @testset "a 50 ns outlier is suppressed by orders of magnitude" begin
        rng = MersenneTwister(0x51)
        nant, ref = 5, 1
        bl = all_baselines(nant)
        pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
        τ = 1.0e-9 .* randn(rng, nant, 2)
        D = inject_detections(bl, pols, τ, zeros(nant, 2))
        D[3, 1] = merge(D[3, 1], (; delay = 50.0e-9))

        derr50(sol) = maximum(
            abs(sol.delay[a, f] - (τ[a, f] - τ[ref, 1]))
                for a in 1:nant, f in 1:2
        )
        rob = stationize(
            D, bl, pols, nant; gauge = PinAntenna(ref),
            opts = FR.Stationization(loss = FR.SoftL1())
        )
        ls = stationize(
            D, bl, pols, nant; gauge = PinAntenna(ref),
            opts = FR.Stationization(loss = FR.LeastSquares())
        )
        @test derr50(ls) > 1.0e-9
        @test derr50(rob) < 1.0e-11
        @test derr50(rob) < 1.0e-3 * derr50(ls)
    end

    @testset "a robust loss without the scan geometry is refused" begin
        # Fail fast rather than leave every delay row at full weight while
        # reporting a robust solve.
        s = poisoned_scan(4)
        @test_throws ArgumentError stationize(
            s.D, s.bl, s.pols, 4; gauge = PinAntenna(1), spreads = (;),
            opts = FR.Stationization(loss = FR.SoftL1()),
        )
        @test_throws "needs the scan's RMS frequency spread" stationize(
            s.D, s.bl, s.pols, 4; gauge = PinAntenna(1), spreads = (;),
            opts = FR.Stationization(loss = FR.SoftL1()),
        )
        @test_throws "RMS time spread" stationize(
            s.D, s.bl, s.pols, 4; gauge = PinAntenna(1), spreads = (freq_rms = SCAN_SPREAD.freq_rms,),
            opts = FR.Stationization(loss = FR.SoftL1()),
        )
        # `LeastSquares` needs no geometry: scaling every weight in a system by
        # a common factor leaves the solution invariant (to rounding — the QR
        # runs on differently scaled numbers, so this is not bit-identical).
        bare = stationize(
            s.D, s.bl, s.pols, 4; gauge = PinAntenna(1), spreads = (;),
            opts = FR.Stationization(loss = FR.LeastSquares()),
        ).delay
        scaled = stationize(
            s.D, s.bl, s.pols, 4; gauge = PinAntenna(1),
            opts = FR.Stationization(loss = FR.LeastSquares()),
        ).delay
        @test bare ≈ scaled atol = 1.0e-18
    end

    @testset "multi-scan: an outlier in one scan does not leak into another" begin
        # Two scans share one model; the poison sits in scan 2 only, so scan 1
        # must come back exactly as if solved alone.
        nant, ref = 5, 1
        s1 = poisoned_scan(nant; offset = 0.0, seed = 0x41)     # clean
        s2 = poisoned_scan(nant; seed = 0x42)                   # poisoned
        # `scan_of_time` is what gives `PerScan` two distinct segments, so the
        # two scans land in separate θ columns.
        geom = CALs.DataGeometry(; nfeed = 2,
            times = [0.0, 1.0, 100.0, 101.0], scan_of_time = [1, 1, 2, 2],
            channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9,
        )
        model = CALs.GainModel(
            phase = (
                delay = CALs.GainComponent(CALs.Delay(); Ti = CALs.PerScan(), Frequency = CALs.GlobalFrequency(), Feed = CALs.PerFeed()),
            ),
        )
        layout = CALs.plan_parameters(model, nant, geom)
        dplan = layout.plans[1]
        opts = FR.Stationization(loss = FR.SoftL1())

        θ = zeros(layout.nθ)
        solve_named!(
            θ, (
                detstack(s1.D, s1.bl, s1.pols; ti = 1),
                detstack(s2.D, s2.bl, s2.pols; ti = 3),
            ),
            ((dplan, :delay),); gauge = PinAntenna(ref), opts,
        )
        θ1 = zeros(layout.nθ)
        solve_named!(
            θ1, (detstack(s1.D, s1.bl, s1.pols; ti = 1),),
            ((dplan, :delay),); gauge = PinAntenna(ref), opts,
        )
        # Scan 1's columns are untouched by scan 2's outlier: with a per-scan
        # model the systems are block-diagonal, and the loss keeps them that way
        # because it reweights rows rather than pooling a scale across them.
        for a in 1:nant, f in 1:2
            c = plan_off1(dplan)[a, f, 1, 1]
            c == 0 && continue
            @test θ[c] ≈ θ1[c] atol = 1.0e-3 * s1.σ
        end
        # Scan 2 still recovers its own truth despite carrying the outlier.
        for a in 1:nant, f in 1:2
            c = plan_off1(dplan)[a, f, 2, 1]
            c == 0 && continue
            @test abs(θ[c] - (s2.τ[a, f] - s2.τ[ref, 1])) < 10 * s2.σ
        end
    end
end

# The engine contract behind `BaselineFringeFit`'s scan-local mode: on a model whose
# every column is per-scan, solving each scan's system alone accumulates the
# same solution as one pooled call over all scans. The graph aggregation is
# EXACT — connectivity is weight-independent, so component counts sum and the
# covered sets are equal — while θ agrees only to solver rounding: the pooled
# IRLS couples its stopping rule across the (independent) blocks, and the
# stacked QR rounds differently than the per-block ones.
@testset "Stationize: per-scan solves ≡ the pooled block-diagonal system" begin
    nant, ref = 5, 1
    geom = CALs.DataGeometry(; nfeed = 2,
        times = [0.0, 1.0, 100.0, 101.0, 200.0, 201.0], scan_of_time = [1, 1, 2, 2, 3, 3],
        channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9,
    )
    mk(term) = CALs.GainComponent(term; Ti = CALs.PerScan(), Frequency = CALs.GlobalFrequency(), Feed = CALs.PerFeed())
    model = CALs.GainModel(phase = (mbd = mk(CALs.Delay()), rate = mk(CALs.Rate())))
    layout = CALs.plan_parameters(model, nant, geom)
    comps = ((layout.plans[1], :delay), (layout.plans[2], :rate))
    bl = all_baselines(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    # Scan 2 carries a closure-breaking delay outlier, so the robust loss
    # genuinely iterates and the pooled stopping-rule coupling is exercised.
    function scan_D(seed; poison)
        rng = MersenneTwister(seed)
        D = inject_detections(bl, pols, 1.0e-9 .* randn(rng, nant, 2), 1.0e-12 .* randn(rng, nant, 2))
        poison && (D[2, 1] = merge(D[2, 1], (; delay = D[2, 1].delay + 50.0e-9)))
        return D
    end
    stacks = [detstack(scan_D(0x40 + i; poison = i == 2), bl, pols; ti = 2i - 1) for i in 1:3]
    opts = FR.Stationization(loss = FR.SoftL1())

    θp = zeros(layout.nθ)
    ncomp_p, cov_p = solve_named!(θp, Tuple(stacks), comps; gauge = PinAntenna(ref), opts)
    θs = zeros(layout.nθ)
    ncomp_s = 0
    cov_s = Set{Tuple{String, Int, Int}}()
    for (gi, st) in enumerate(stacks)
        nc, cov = solve_named!(θs, (st,), comps; gauge = PinAntenna(ref), opts)
        ncomp_s += nc
        union!(cov_s, Set((a, f, gi) for (a, f, _) in cov))
    end
    @test ncomp_s == ncomp_p
    @test cov_s == cov_p
    @test θs ≈ θp rtol = 1.0e-9
    @test maximum(abs, θs .- θp) < 1.0e-11
end

@testset "Stationize: systematic floor constructor defaults" begin
    o = FR.Stationization(systematic_delay = 3.0e-12, systematic_rate = 2.0e-3)
    @test o.systematic_delay == 3.0e-12
    @test o.systematic_rate == 2.0e-3
    @test FR.Stationization().systematic_delay == FR.Stationization().systematic_rate == 0
    @test FR.Stationization() == FR.Stationization()
end

# Delay and rate on a SharedFeeds model: one column per station, fed by every
# product whatever feeds it relates.
function sharedfeeds_scan_layout(nant)
    geom = CALs.DataGeometry(; nfeed = 2,
        times = [0.0, 1.0, 2.0], channel_freqs = [1.0e9], t0 = 0.0, f0 = 1.0e9,
    )
    mk(term) = CALs.GainComponent(term; Ti = CALs.PerScan(), Frequency = CALs.GlobalFrequency(), Feed = CALs.SharedFeeds())
    model = CALs.GainModel(phase = (delay = mk(CALs.Delay()), rate = mk(CALs.Rate())))
    return CALs.plan_parameters(model, nant, geom)
end

@testset "Stationize: a single-feed station solves on shared-feed delay and rate" begin
    rng = MersenneTwister(0x7a11)
    nant = 5
    single = 4                                    # single-feed station: feed 2 only
    bl = all_baselines(nant)
    pols = [(1, 1), (2, 2), (1, 2), (2, 1)]
    pfeeds = collect(pols)

    τ = repeat(randn(rng, nant) .* 1.0e-9, 1, 2)
    ṙ = repeat(randn(rng, nant) .* 1.0e-3, 1, 2)
    D = inject_detections(bl, pols, τ, ṙ)
    for bi in eachindex(bl), p in eachindex(pols)
        a, b = bl[bi]
        fa, fb = pfeeds[p]
        if (a == single && fa == 1) || (b == single && fb == 1)
            D[bi, p] = FR.Detection{Float64}((NaN, NaN, NaN, NaN, NaN, NaN, false))
        end
    end

    layout = sharedfeeds_scan_layout(nant)
    dplan, rplan = layout.plans
    θ = zeros(layout.nθ)
    solve_named!(θ, (detstack(D, bl, pols; ti = 1),), ((dplan, :delay), (rplan, :rate)); gauge = PinAntenna(1))

    col(plan, a) = plan_off1(plan)[a, 1, 1, 1]
    for a in 2:nant
        @test θ[col(dplan, a)] ≈ τ[a, 1] - τ[1, 1] atol = 1.0e-18
        @test θ[col(rplan, a)] ≈ ṙ[a, 1] - ṙ[1, 1] atol = 1.0e-12
    end
end
