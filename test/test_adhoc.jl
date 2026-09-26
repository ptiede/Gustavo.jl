# Phase 5 — globally-closing adhoc phasing.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using Statistics: mean, std
using Gustavo.DimensionalData: At, DimArray, Ti, dims, lookup
using OffsetArrays: OffsetArray

const FRa = Gustavo.Fring
const CALa = Gustavo.Calibration

all_bl_a(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]

# Build residual baseline visibilities rbar[bl,pol,ap] from a per-(station,feed,
# ap) phase screen and a per-(baseline,product) source visibility phase `x`
# (constant over the scan, as `solve_adhoc_phasing` models it). `amp` sets the
# coherent SNR.
function inject_screen(bl_pairs, pol_products, screen, x = nothing; amp = 10.0, noise = 0.0, rng = nothing)
    nbl, npol = length(bl_pairs), length(pol_products)
    nap = size(screen, 3)
    feeds = collect(pol_products)
    xs = x === nothing ? zeros(nbl, npol) : x
    rbar = Array{ComplexF64}(undef, nbl, npol, nap)
    wbar = ones(nbl, npol, nap)
    for ap in 1:nap, bi in 1:nbl, p in 1:npol
        a, b = bl_pairs[bi]
        fa, fb = feeds[p]
        model = screen[a, fa, ap] - screen[b, fb, ap] + xs[bi, p]
        v = amp * cis(model)
        if noise > 0 && rng !== nothing
            v += noise * (randn(rng) + im * randn(rng)) / sqrt(2)
        end
        rbar[bi, p, ap] = v
    end
    return rbar, wbar
end

# Labels positional sums `[baseline, product, ap]` for `solve_adhoc_phasing`:
# station `i` is named `"S$i"`, and `bl_pairs`, `pols` and `times` label the axes.
function label_sums(rbar, wbar, bl_pairs, pols, nant, times)
    names = ["S$i" for i in 1:nant]
    ax = (
        FRa._station_pair_dim([(names[a], names[b]) for (a, b) in bl_pairs]),
        FRa.FeedPair(collect(pols)), Ti(times),
    )
    return DimArray(rbar, ax), DimArray(wbar, ax), names
end

function solve_positional(rbar, wbar, bl_pairs, pols, nant, times; kw...)
    R, W, names = label_sums(rbar, wbar, bl_pairs, pols, nant, times)
    return FRa.solve_adhoc_phasing(R, W, names; kw...)
end

# Max |measured − model-from-solution| (mod 2π) over valid baselines/APs.
function adhoc_recon(rbar, sol, bl_pairs, pol_products)
    feeds = collect(pol_products)
    m = 0.0
    for ap in axes(rbar, 3), bi in eachindex(bl_pairs), p in eachindex(pol_products)
        a, b = bl_pairs[bi]
        a == b && continue
        r = rbar[bi, p, ap]
        abs(r) > 0 || continue
        fa, fb = feeds[p]
        (isfinite(sol.phase[a, fa, ap]) && isfinite(sol.phase[b, fb, ap])) || continue
        xhat = isfinite(sol.source[bi, p]) ? sol.source[bi, p] : 0.0
        model = sol.phase[a, fa, ap] - sol.phase[b, fb, ap] + xhat
        m = max(m, abs(rem2pi(angle(r) - model, RoundNearest)))
    end
    return m
end

@testset "Adhoc: ZeroSumPhase gauges each AP without moving the frame" begin
    rng = MersenneTwister(0x0ADC)
    nant, nap = 5, 20
    ref = 2
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    screen = 0.3 .* randn(rng, nant, 2, nap)
    times = collect(0:(nap - 1)) .* 1.0
    rbar, wbar = inject_screen(bl, pols, screen)

    zs = solve_positional(
        rbar, wbar, bl, pols, nant, times;
        gauge = ZeroSumPhase(), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)),
    )
    pin = solve_positional(
        rbar, wbar, bl, pols, nant, times;
        gauge = PinAntenna(ref), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)),
    )

    # The gauge is a per-AP common mode, which cancels on every baseline: both
    # conventions reconstruct the data identically.
    @test adhoc_recon(rbar, zs, bl, pols) < 1.0e-9
    @test adhoc_recon(rbar, pin, bl, pols) < 1.0e-9

    # ONE constant per AP across BOTH feeds: cross hands tie the feeds into a
    # single component with a single freedom, so the sum that vanishes is taken
    # over every covered (station, feed) cell, not per feed.
    for ap in 1:nap
        cells = [(a, f) for a in 1:nant for f in 1:2 if zs.covered[a, f, ap]]
        isempty(cells) && continue
        @test sum(zs.phase[a, f, ap] for (a, f) in cells) ≈ 0 atol = 1.0e-8
    end
    # No station is held at zero the way a pin holds its reference.
    @test !all(abs.(zs.phase[ref, 1, :]) .< 1.0e-9)
    @test all(abs.(pin.phase[ref, 1, :]) .< 1.0e-9)

    # The two differ by a per-AP constant and nothing more, and it is the SAME
    # constant on both feeds — the frame moved, the physics did not. A per-feed
    # constant would pass a per-feed check while breaking every cross-hand
    # difference, so the spread is taken across feeds together.
    for ap in 1:nap
        d = [
            zs.phase[a, f, ap] - pin.phase[a, f, ap]
                for a in 1:nant for f in 1:2
                if zs.covered[a, f, ap] && pin.covered[a, f, ap]
        ]
        length(d) < 2 && continue
        @test maximum(d) - minimum(d) ≈ 0 atol = 1.0e-8
    end
end

@testset "Adhoc: raw per-AP global solve closes" begin
    rng = MersenneTwister(0x0ADC)
    nant, nap = 5, 20
    ref = 1
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    screen = 0.3 .* randn(rng, nant, 2, nap)
    times = collect(0:(nap - 1)) .* 1.0

    rbar, wbar = inject_screen(bl, pols, screen)
    sol = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)))

    @test adhoc_recon(rbar, sol, bl, pols) < 1.0e-9
    # Cross-hand rows join the two feed blocks into ONE connected component, whose
    # single per-AP gauge freedom is pinned at the reference's feed-1 node. Every
    # other node — the reference's own feed 2 included — is measured against it, so
    # the reference's inter-feed phase stays in the solution instead of being
    # pinned away.
    @test all(abs.(sol.phase[ref, 1, :]) .< 1.0e-9)

    # The free source terms leave one further freedom, a constant per node
    # (`φ_a → φ_a + c_a`, `x_ab → x_ab − (c_a − c_b)`), so a track is recovered up
    # to its own constant — which is exactly what the demean then fixes.
    truth(a, f) = [screen[a, f, ap] - screen[ref, 1, ap] for ap in 1:nap]
    for a in 1:nant, f in 1:2
        d = sol.phase[a, f, :] .- truth(a, f)
        @test maximum(d) - minimum(d) < 1.0e-8
    end

    # With the demean on, the gauge is fixed and each track matches truth exactly,
    # to within the per-scan mean the demean removes by design.
    sd = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = true)))
    for a in 1:nant, f in 1:2
        t = truth(a, f)
        @test maximum(abs.(sd.phase[a, f, :] .- (t .- mean(t)))) < 1.0e-8
    end
end

@testset "Adhoc: a free source term absorbs the source visibility phase" begin
    # The model carries one source phase per (baseline, product), constant over the
    # scan. Its NON-closing part is the source's closure phase, which no station
    # term can represent — omit it and the per-AP solve absorbs it into the station
    # tracks. The absorption is time-varying (and so survives the demean) exactly
    # when coverage flickers, because that is what makes the per-AP solve matrix
    # vary from AP to AP.
    rng = MersenneTwister(0x50C1)
    nant, nap = 6, 30
    ref = 1
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = repeat(0.4 .* randn(rng, nant, 1, nap), 1, 2, 1)   # feed-common truth
    xtrue = 0.8 .* randn(rng, length(bl), length(pols))
    rbar, wbar = inject_screen(bl, pols, screen, xtrue; amp = 30.0)
    for ap in 1:nap, bi in eachindex(bl), p in eachindex(pols)   # coverage flicker
        rand(rng) < 0.35 || continue
        rbar[bi, p, ap] = 0.0 + 0.0im
        wbar[bi, p, ap] = 0.0
    end

    # Deviation of the recovered track from truth, after removing the per-station
    # constant the (φ, x) gauge leaves free. Zero iff the source term was absorbed.
    wobble(sol) = maximum(
        begin
            v = filter(isfinite, [sol.phase[a, 1, ap] - (screen[a, 1, ap] - screen[ref, 1, ap]) for ap in 1:nap])
            isempty(v) ? 0.0 : maximum(abs.(v .- (sum(v) / length(v))))
        end for a in 1:nant
    )
    opts = (; gauge = PinAntenna(ref), tying = CALa.SharedFeeds())
    # The negative control disables BOTH source-term mechanisms: the seed
    # alternation (`source_iters = 1`) and the complex-domain refinement
    # (`complex_iters = 0`), whose complex source means otherwise absorb the
    # same per-(baseline, product) constant.
    off = solve_positional(
        rbar, wbar, bl, pols, nant, times;
        smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false, snr_floor = 0.0, source_iters = 1, complex_iters = 0)), opts...,
    )
    on = solve_positional(
        rbar, wbar, bl, pols, nant, times;
        smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false, snr_floor = 0.0)), opts...,
    )
    # The refinement alone (seed alternation still disabled) also absorbs the
    # source phase: its complex source means play the same role.
    refined = solve_positional(
        rbar, wbar, bl, pols, nant, times;
        smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false, snr_floor = 0.0, source_iters = 1)), opts...,
    )
    @test wobble(off) > 0.3                       # unmodelled source phase corrupts the tracks
    @test wobble(on) < 1.0e-3                     # modelling it removes the corruption
    @test wobble(off) > 100 * wobble(on)          # by orders of magnitude, not marginally
    @test wobble(refined) < 0.05

    # `x` is defined only up to `x_ab → x_ab − (c_a − c_b)`; the closure triangle is
    # the gauge-invariant part, and it must match the injected source. A triangle
    # sums three terms, each converged to the smoother's `source_tol`.
    ix = Dict(bl[i] => i for i in eachindex(bl))
    worst = 0.0
    for a in 1:nant, b in (a + 1):nant, c in (b + 1):nant, p in eachindex(pols)
        tri(v) = v[ix[(a, b)], p] + v[ix[(b, c)], p] - v[ix[(a, c)], p]
        worst = max(worst, abs(rem2pi(tri(on.source) - tri(xtrue), RoundNearest)))
    end
    @test worst < 1.0e-5
end

@testset "Adhoc: a one-AP (baseline, product) is dropped, not fitted" begin
    # Its source term absorbs its single row exactly, so the row constrains no
    # station phase; admitting it would only inflate `covered`.
    rng = MersenneTwister(0x01AF)
    nant, nap = 4, 12
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = repeat(0.3 .* randn(rng, nant, 1, nap), 1, 2, 1)
    rbar, wbar = inject_screen(bl, pols, screen; amp = 20.0)
    wbar[1, 1, 2:end] .= 0.0                   # baseline 1, product 1 survives at ONE AP
    rbar[1, 1, 2:end] .= 0.0 + 0.0im
    sol = solve_positional(
        rbar, wbar, bl, pols, nant, times;
        gauge = PinAntenna(1), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false, snr_floor = 0.0)),
        tying = CALa.SharedFeeds(),
    )
    @test isnan(sol.source[1, 1])              # unidentifiable, so never fitted
    @test all(isfinite, sol.source[2:end, 1])  # its neighbours still are
end

@testset "Adhoc: detrend removes per-scan mean only (keeps slope/rate)" begin
    rng = MersenneTwister(0x0DE7)
    nant, nap = 4, 30
    ref = 1
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
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
    rbar, wbar = inject_screen(bl, pols, screen)
    sol = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = true)))

    # Detrend removes the per-station MEAN (breaks the constant-phase gauge vs the
    # Stage-B ConstantTerm) but KEEPS the slope — so adhoc can flatten a residual
    # fringe rate. The recovered slope must match the injected differential rate
    # c1[a] - c1[ref] (the common wiggle cancels in the ref-relative solve).
    for a in 1:nant, f in 1:2
        a == ref && continue
        tr = sol.phase[a, f, :]
        @test abs(mean(tr)) < 1.0e-8
        slope = sum(tc .* (tr .- mean(tr))) / sum(tc .^ 2)
        @test isapprox(slope, c1[a, f] - c1[ref, 1]; atol = 1.0e-8)
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
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant, f in 1:2, ap in 1:nap
        screen[a, f, ap] = 0.3 * randn(rng) + 0.04 * sin(2π * ap / nap + a)
    end
    rbar, wbar = inject_screen(bl, pols, screen; amp = 5.0, noise = 0.4, rng = rng)
    sm = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false, snr_floor = 1.0))
    s1 = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = sm)
    k = 1.0e-6
    s2 = solve_positional(k .* rbar, k .* wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = sm)

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
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
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
    rbar, wbar = inject_screen(bl, pols, screen; amp = 4.0, noise = 1.5, rng = rng)

    truth(a, f) = screen[a, f, :] .- screen[ref, 1, :]
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

    raw = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)))
    sm = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.SavitzkyGolaySmoother(window = 11, order = 2, options = FRa.AdhocOptions(; detrend = false)))
    pen = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.PenalizedSmoother(smoothness = 20.0, options = FRa.AdhocOptions(; detrend = false)))
    gp = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.OUSmoother(coherence_time = 15.0, options = FRa.AdhocOptions(; detrend = false)))

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
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    tc = times .- mean(times)
    c0 = 0.5 .* randn(rng, nant, 2)
    c1 = 0.02 .* randn(rng, nant, 2)
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant, f in 1:2, ap in 1:nap
        screen[a, f, ap] = c0[a, f] + c1[a, f] * tc[ap]
    end
    rbar, wbar = inject_screen(bl, pols, screen; amp = 20.0)
    sol = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.OUSmoother(coherence_time = 30.0, options = FRa.AdhocOptions(; detrend = true)))

    for a in 1:nant, f in 1:2
        a == ref && continue
        tr = sol.phase[a, f, :]
        @test abs(mean(tr)) < 1.0e-6
        slope = sum(tc .* (tr .- mean(tr))) / sum(tc .^ 2)
        @test isapprox(slope, c1[a, f] - c1[ref, 1]; atol = 5.0e-3)
    end
end

@testset "Adhoc :gp ref antenna pinned to zero" begin
    rng = MersenneTwister(0x7A17)
    nant, nap = 4, 30
    ref = 2
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = Array{Float64}(undef, nant, 2, nap)
    for a in 1:nant, f in 1:2, ap in 1:nap
        screen[a, f, ap] = 0.3 * sin(2π * ap / nap + a) + 0.2 * randn(rng)
    end
    rbar, wbar = inject_screen(bl, pols, screen; amp = 8.0, noise = 0.5, rng = rng)
    sol = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.OUSmoother(options = FRa.AdhocOptions(; detrend = false)))
    # The gauge pins the reference's feed-1 node; its feed 2 holds the reference's
    # own inter-feed phase, measured against that pin.
    @test all(abs.(sol.phase[ref, 1, :]) .< 1.0e-8)
end

# Non-birefringent (feed-common) smooth screen for the joint solve — matches the
# `SharedFeeds` truth (feed 2 ≡ feed 1) so the recovered per-station track is well
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

# Cross-hand rows enter the solve like any other product: whatever extra phase
# they carry — source EVPA, D-terms, the rotated Stokes combination a linear feed
# sees — is a per-(baseline, product) constant the model fits, so it never reaches
# the station tracks. That extra phase may be large and wildly baseline-dependent
# without disturbing the result; what the model does require is that it hold still
# over the scan.
@testset "Adhoc :gp_joint pins ref and absorbs a poisoned cross hand" begin
    rng = MersenneTwister(0x00102547)
    nant, nap = 5, 50
    ref = 2
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = shared_screen(rng, nant, nap)
    rbar, wbar = inject_screen(bl, pols, screen; amp = 6.0, noise = 1.0, rng = rng)
    # Cross hands carrying a strong, baseline-dependent phase near the wrap cut —
    # the linear-feed case. Constant over the scan, so the source term takes it.
    feeds = collect(pols)
    for p in eachindex(pols)
        fa, fb = feeds[p]
        fa == fb && continue
        for bi in eachindex(bl), ap in 1:nap
            a, b = bl[bi]
            rbar[bi, p, ap] = 20.0 * cis(screen[a, 1, ap] - screen[b, 1, ap] + π - 0.02 * bi)
        end
    end
    sol = solve_positional(
        rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref),
        smoother = FRa.JointOUSmoother(coherence_time = 15.0, options = FRa.AdhocOptions(; detrend = false)), tying = CALa.SharedFeeds(),
    )
    @test all(abs.(filter(isfinite, sol.phase[ref, :, :])) .< 1.0e-8)     # ref pinned
    @test all(sol.phase[:, 1, :] .=== sol.phase[:, 2, :])                 # both feeds share the node
    # Station tracks recover the injected screen (ref-relative, up to each track's
    # own gauge constant) undisturbed by the cross hands.
    dev = 0.0
    for a in 1:nant
        d = [sol.phase[a, 1, ap] - (screen[a, 1, ap] - screen[ref, 1, ap]) for ap in 1:nap]
        d .-= mean(d)
        dev = max(dev, maximum(abs, d))
    end
    @test dev < 0.35
end

@testset "Adhoc :gp_joint denoises and closes" begin
    rng = MersenneTwister(0x9317)
    nant, nap = 5, 60
    ref = 1
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = shared_screen(rng, nant, nap; k = 1.5, amp = 0.5)
    rbar, wbar = inject_screen(bl, pols, screen; amp = 3.0, noise = 2.0, rng = rng)

    truth(a, f) = screen[a, f, :] .- screen[ref, 1, :]
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

    none = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)), tying = CALa.SharedFeeds())
    gp = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.OUSmoother(coherence_time = 20.0, options = FRa.AdhocOptions(; detrend = false)), tying = CALa.SharedFeeds())
    gpj = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.JointOUSmoother(coherence_time = 20.0, options = FRa.AdhocOptions(; detrend = false)), tying = CALa.SharedFeeds())

    @test rms_to_truth(gpj) < rms_to_truth(none)          # joint solve denoises
    @test rms_to_truth(gpj) < 1.5 * rms_to_truth(gp)      # competitive with per-track
end

@testset "Adhoc :gp_joint requires one node per station" begin
    rng = MersenneTwister(0x5A17)
    nant, nap = 3, 10
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = shared_screen(rng, nant, nap)
    rbar, wbar = inject_screen(bl, pols, screen; amp = 8.0)
    @test_throws ErrorException solve_positional(
        rbar, wbar, bl, pols, nant, times;
        smoother = FRa.JointOUSmoother(), tying = CALa.PerFeed(),
    )
end

@testset "Adhoc: low-SNR no-anchor solve is unbiased" begin
    # No dominant anchor station (all equal SNR), low per-baseline SNR. The
    # global solve over all baselines should be unbiased — averaging many noise
    # realizations recovers the gauged truth.
    nant, nap = 5, 4
    ref = 1
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    rng = MersenneTwister(0xB1A5)
    screen = 0.4 .* randn(rng, nant, 2, nap)

    ntrial = 80
    acc = zeros(nant, 2, nap)
    cnt = zeros(nant, 2, nap)
    for _ in 1:ntrial
        rbar, wbar = inject_screen(bl, pols, screen; amp = 2.0, noise = 1.0, rng = rng)
        sol = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false, snr_floor = 0.0)))
        for a in 1:nant, f in 1:2, ap in 1:nap
            isfinite(sol.phase[a, f, ap]) || continue
            acc[a, f, ap] += rem2pi(sol.phase[a, f, ap], RoundNearest)
            cnt[a, f, ap] += 1
        end
    end
    maxbias = 0.0
    for a in 1:nant, f in 1:2
        a == ref && continue
        aps = [ap for ap in 1:nap if cnt[a, f, ap] > 0]
        isempty(aps) && continue
        d = [acc[a, f, ap] / cnt[a, f, ap] - (screen[a, f, ap] - screen[ref, 1, ap]) for ap in aps]
        d .-= mean(d)                      # the per-track gauge constant
        maxbias = max(maxbias, maximum(abs, d))
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
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    rng = MersenneTwister(0xC3C3)

    c = 0.6 .* randn(rng, nant, 2)                 # per-(station,feed) constant
    screen = repeat(reshape(c, nant, 2, 1), 1, 1, nap)
    rbar, wbar = inject_screen(bl, pols, screen)

    # Drop every baseline touching the reference in APs 5..8.
    dropaps = 5:8
    for ap in dropaps, bi in eachindex(bl)
        (bl[bi][1] == ref || bl[bi][2] == ref) || continue
        rbar[bi, :, ap] .= 0.0 + 0.0im
        wbar[bi, :, ap] .= 0.0
    end

    sol = solve_positional(
        rbar, wbar, bl, pols, nant, times;
        gauge = PinAntenna(ref), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)),
    )

    # The reference is unsolved in the dropout APs but solved elsewhere.
    @test all(!sol.covered[ref, 1, ap] && !sol.covered[ref, 2, ap] for ap in dropaps)
    @test all(sol.covered[ref, 1, ap] for ap in 1:nap if !(ap in dropaps))

    # Non-ref stations are still solved in the dropout APs, and the recovered
    # phase relative to truth is the SAME constant across ALL APs (no jump).
    for a in 2:nant, f in 1:2
        truth = c[a, f] - c[ref, 1]
        d = [rem2pi(sol.phase[a, f, ap] - truth, RoundNearest) for ap in 1:nap if isfinite(sol.phase[a, f, ap])]
        @test length(d) == nap                                    # solved every AP
        @test maximum(d) - minimum(d) < 1.0e-6                    # gauge consistent across dropout
    end

    # REF-ABSENT variant (multi-subarray track): the nominal reference NEVER
    # observes, and the best-covered station (the effective anchor) drops out in
    # a middle block. Keying the restitch on the literal reference left these
    # scans with NO gauge repair at all — the dropout APs pin on a different
    # node and every station's track jumps by an arbitrary constant (this is
    # what let the adhoc DEGRADE VR2505's GS-less 0607-157 scan).
    nant6 = nant + 1                                 # station 6 = the absent reference
    rbar2, wbar2 = inject_screen(bl, pols, screen)
    for ap in dropaps, bi in eachindex(bl)           # drop the anchor (station 1) instead
        (bl[bi][1] == 1 || bl[bi][2] == 1) || continue
        rbar2[bi, :, ap] .= 0.0 + 0.0im
        wbar2[bi, :, ap] .= 0.0
    end
    sol2 = solve_positional(
        rbar2, wbar2, bl, pols, nant6, times;
        gauge = PinAntenna(nant6), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)),
    )
    @test !any(sol2.covered[nant6, :, :])            # absent ref never fabricated
    for a in 2:nant, f in 1:2
        truth = c[a, f] - c[1, 1]
        d = [rem2pi(sol2.phase[a, f, ap] - truth, RoundNearest) for ap in 1:nap if isfinite(sol2.phase[a, f, ap])]
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
    ends = [(1, 2), (1, 3), (2, 4), (3, 4)]
    cells = (FRa._station_pair_dim([("S$a", "S$b") for (a, b) in ends]), FRa.FeedPair([(1, 1)]))
    nodes = DimArray([((a, 1), (b, 1)) for (a, b) in ends, _ in 1:1], cells)
    val = DimArray(reshape([0.0, 0.0, 3.0, -3.0], :, 1), cells)
    w = DimArray(reshape([100.0, 100.0, 1.0, 1.0], :, 1), cells)
    mask = DimArray(trues(4, 1), cells)
    nant, ref = 4, 1
    solve4(seed) = begin
        sp = seed === nothing ? nothing : (M = fill(NaN, nant, 2); M[4, 1] = seed; M)
        ph, cov = zeros(nant, 2), falses(nant, 2)
        FRa._solve_observable!(ph, cov, val, w, mask, nodes, PinAntenna(ref); rewrap = 3, seed_phase = sp)
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
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = 0.2 .* randn(rng, nant2, 2, nap)
    for ap in 1:nap                                  # station 5 hovers near +π
        screen[5, :, ap] .= (π - 0.1) .+ 0.1 .* sin(2π * ap / 15)
    end
    rbar, wbar = inject_screen(bl, pols, screen; amp = 6.0, noise = 1.0, rng = rng)
    sol = solve_positional(
        rbar, wbar, bl, pols, nant2, times;
        gauge = PinAntenna(1), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)),
    )
    tr = sol.phase[5, 1, :]
    jumps = [abs(rem2pi(tr[ap + 1] - tr[ap], RoundNearest)) for ap in 1:(nap - 1) if isfinite(tr[ap]) && isfinite(tr[ap + 1])]
    @test maximum(jumps) < 1.0                       # no ~π branch flip in the track

    # REF-ABSENT scan (multi-subarray track, e.g. VR2505's 0607-157 with no GS):
    # the reference antenna never observes, so the warm start must key on the
    # EFFECTIVE anchor (best-covered station) instead — keying on the literal
    # the literal reference disabled the warm start entirely here and the branch flips
    # returned on the weak station.
    nant3 = nant2 + 1                                # station 6 = the absent reference
    sol_noref = solve_positional(
        rbar, wbar, bl, pols, nant3, times;
        gauge = PinAntenna(nant3), smoother = FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)),
    )
    @test !any(sol_noref.covered[nant3, :, :])       # absent ref is never fabricated
    trn = sol_noref.phase[5, 1, :]
    @test count(isfinite, trn) == nap
    jumpsn = [abs(rem2pi(trn[ap + 1] - trn[ap], RoundNearest)) for ap in 1:(nap - 1) if isfinite(trn[ap]) && isfinite(trn[ap + 1])]
    @test maximum(jumpsn) < 1.0                      # warm start armed via the anchor
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

# A per-track smoother that replaces each track by its mean.
struct FlattenSmoother <: FRa.AbstractAdhocSmoother
    options::FRa.AdhocOptions
end
FRa.smooth_track!(::FlattenSmoother, track, w) = fill!(track, mean(filter(isfinite, track)))

@testset "Adhoc smoother interface: every type dispatches" begin
    # The pluggable `AbstractAdhocSmoother` interface — each concrete smoother
    # constructs, subtypes the abstract type, and drives `solve_adhoc_phasing`
    # through `apply_adhoc!`. Mirrors the bandpass-smoother loop in test_pipeline.jl.
    rng = MersenneTwister(0x00FACADE)
    nant, nap = 4, 30
    ref = 1
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 1.0
    screen = shared_screen(rng, nant, nap)         # feed-common (works for the joint solve too)
    rbar, wbar = inject_screen(bl, pols, screen; amp = 6.0, noise = 0.5, rng = rng)

    # (smoother, tying) — the joint solve needs one phase node per station.
    cases = [
        (FRa.SavitzkyGolaySmoother(window = 9, options = FRa.AdhocOptions(; detrend = false)), CALa.PerFeed()),
        (FRa.PenalizedSmoother(smoothness = 10.0, options = FRa.AdhocOptions(; detrend = false)), CALa.PerFeed()),
        (FRa.OUSmoother(coherence_time = 15.0, options = FRa.AdhocOptions(; detrend = false)), CALa.PerFeed()),
        (FRa.NoSmoothing(options = FRa.AdhocOptions(; detrend = false)), CALa.PerFeed()),
        (FRa.JointOUSmoother(coherence_time = 15.0, options = FRa.AdhocOptions(; detrend = false)), CALa.SharedFeeds()),
    ]
    for (sm, ty) in cases
        @test sm isa FRa.AbstractAdhocSmoother
        sol = solve_positional(rbar, wbar, bl, pols, nant, times; gauge = PinAntenna(ref), smoother = sm, tying = ty)
        @test sol isa Gustavo.DimensionalData.AbstractDimStack
        @test (:phase, :covered, :source) ⊆ keys(sol)               # DimStack layers
        @test size(sol.phase) == (nant, 2, length(times))        # Ant × Feed × Ti
        @test count(isfinite, sol.phase) > 0                     # the stage actually ran
        @test all(abs.(filter(isfinite, sol.phase[ref, 1, :])) .< 1.0e-8)   # ref gauge held
    end

    # A smoother that defines only the per-track hook overwrites each track.
    flat = solve_positional(
        rbar, wbar, bl, pols, nant, times;
        gauge = PinAntenna(ref), smoother = FlattenSmoother(FRa.AdhocOptions(; detrend = false)),
    )
    for a in 1:nant, f in 1:2
        tr = filter(isfinite, flat.phase[a, f, :])
        isempty(tr) || @test maximum(tr) - minimum(tr) < 1.0e-6
    end
end


# ── Complex-domain Gauss–Newton refinement ───────────────────────────────────
#
# The refinement's value case: a few strong baselines give the phase-extraction
# seed its 2π-branch spine, and the WEAK baselines — whose per-AP rows the
# seed's gate drops or mis-weights — contribute through the linearized complex
# rows at their exact first-order information. The refined tracks must beat the
# seed-only tracks where weak data dominates, and stay equivalent where the
# seed was already near-optimal.
@testset "Adhoc: complex-domain refinement exploits weak baselines" begin
    function screen_scenario(snrs; seed = 0x0ADC, nap = 240, nant = 6)
        rng = MersenneTwister(seed)
        bl = all_bl_a(nant)
        pols = [(1, 1), (2, 2)]
        times = collect(0:(nap - 1)) .* 1.0
        screen = zeros(nant, nap)
        for a in 2:nant, t in 2:nap
            screen[a, t] = screen[a, t - 1] + 0.08 * randn(rng)
        end
        rbar = zeros(ComplexF64, length(bl), 2, nap)
        wbar = zeros(length(bl), 2, nap)
        for (bi, (a, b)) in enumerate(bl)
            sigma = 1.0 / (sqrt(2) * snrs(a, b))
            w = 1 / sigma^2
            for p in 1:2, ap in 1:nap
                v = cis(screen[a, ap] - screen[b, ap]) + sigma * (randn(rng) + im * randn(rng))
                rbar[bi, p, ap] = w * v
                wbar[bi, p, ap] = w
            end
        end
        return rbar, wbar, bl, pols, times, screen
    end
    function track_rmse(sol, screen, nant, nap)
        tot = 0.0; n = 0
        for a in 2:nant
            v = filter(isfinite, [sol.phase[a, 1, ap] - screen[a, ap] for ap in 1:nap])
            isempty(v) && continue
            v .-= sum(v) / length(v)
            tot += sum(abs2, v); n += length(v)
        end
        return n == 0 ? NaN : sqrt(tot / n)
    end
    nant, nap = 6, 240
    rbar, wbar, bl, pols, times, screen =
        screen_scenario((a, b) -> (a == 2 || b == 2) ? 3.0 : 0.5)
    rms = map((0, 2)) do ci
        sol = solve_positional(
            rbar, wbar, bl, pols, nant, times;
            gauge = PinAntenna(1),
            smoother = FRa.JointOUSmoother(coherence_time = 30.0, options = FRa.AdhocOptions(; complex_iters = ci)),
            tying = CALa.SharedFeeds(),
        )
        track_rmse(sol, screen, nant, nap)
    end
    @test rms[2] < 0.85 * rms[1]     # weak baselines genuinely add information
    @test rms[2] < 0.15              # and the refined tracks are good in absolute terms

    # Where the seed is already near-optimal (uniform moderate SNR), refinement
    # must not degrade it beyond its slightly different smoothing balance.
    rbar, wbar, bl, pols, times, screen = screen_scenario((a, b) -> 1.5)
    rms = map((0, 2)) do ci
        sol = solve_positional(
            rbar, wbar, bl, pols, nant, times;
            gauge = PinAntenna(1),
            smoother = FRa.JointOUSmoother(coherence_time = 30.0, options = FRa.AdhocOptions(; complex_iters = ci)),
            tying = CALa.SharedFeeds(),
        )
        track_rmse(sol, screen, nant, nap)
    end
    @test rms[2] < 1.3 * rms[1]
    @test rms[2] < 0.15
end

@testset "Adhoc: the solve works in the data's element type" begin
    rng = MersenneTwister(0x0AD7)
    nant, nap = 5, 20
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    screen = 0.3 .* randn(rng, nant, 2, nap)
    times = collect(0:(nap - 1)) .* 1.0
    rbar, wbar = inject_screen(bl, pols, screen; noise = 0.5, rng)
    r32, w32 = ComplexF32.(rbar), Float32.(wbar)
    solve(r, w, sm) = solve_positional(r, w, bl, pols, nant, times; gauge = PinAntenna(1), smoother = sm)

    s32 = solve(r32, w32, FRa.SavitzkyGolaySmoother())
    @test eltype(s32.phase) == Float32
    @test eltype(s32.source) == Float32
    s64 = solve(r32, w32, FRa.SavitzkyGolaySmoother(; options = FRa.AdhocOptions(; eltype = Float64)))
    @test eltype(s64.phase) == Float64
    @test maximum(abs, filter(isfinite, s64.phase .- s32.phase)) < 1.0e-5
    @test eltype(solve(rbar, wbar, FRa.NoSmoothing()).phase) == Float64
    @test_throws "`eltype` must be a real floating-point type" FRa.AdhocOptions(; eltype = Int)

    R, W, names = label_sums(r32, w32, bl, pols, nant, times)
    nodes = FRa._cell_nodes(R, names, CALa.PerFeed())
    noise2 = @inferred FRa._cell_noise2(R, W, Ti)
    @test eltype(noise2) == Float32
    obs = @inferred FRa._adhoc_obs(R, W, nodes, noise2, 1.0)
    @test eltype(obs.val) == eltype(obs.w) == Float32
    phase = zeros(Float32, nant, 2, nap)
    sbar = ones(ComplexF32, dims(nodes))
    @test eltype((@inferred FRa._linearized_obs(R, W, nodes, noise2, phase, sbar)).val) == Float32
    slice(A) = view(A, Ti(1))
    ph, cov = zeros(Float32, nant, 2), falses(nant, 2)
    @test (@inferred FRa._solve_observable!(
        ph, cov, slice(obs.val), slice(obs.w), slice(obs.mask), nodes, PinAntenna(1); rewrap = 2,
    )) == 1
    @test all(isfinite, ph) && all(cov)
    seed = @inferred Union{Nothing, Matrix{Float32}} FRa._circular_ap_seed(
        slice(obs.val), slice(obs.w), slice(obs.mask), nodes, nant, 1,
    )
    @test seed isa Matrix{Float32}
end

@testset "Adhoc: sums are read by label" begin
    rng = MersenneTwister(0x1ABE)
    nant, nap = 5, 16
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    times = collect(0:(nap - 1)) .* 2.0
    rbar, wbar = inject_screen(bl, pols, 0.3 .* randn(rng, nant, 2, nap); noise = 1.0, rng)
    R, W, names = label_sums(rbar, wbar, bl, pols, nant, times)
    sm = FRa.OUSmoother()
    sol = FRa.solve_adhoc_phasing(R, W, names; smoother = sm)
    @test lookup(sol.phase, FRa.Ant) == names
    @test lookup(sol.phase, Ti) == times
    @test lookup(sol.source, FRa.StationPair) == lookup(R, FRa.StationPair)
    @test lookup(sol.source, FRa.FeedPair) == pols

    # Storage order is free, and a view along time solves only its APs.
    perm = FRa.solve_adhoc_phasing(
        permutedims(R, (Ti, FRa.FeedPair, FRa.StationPair)), permutedims(W, (FRa.FeedPair, Ti, FRa.StationPair)),
        names; smoother = sm,
    )
    @test isequal(perm.phase, sol.phase) && isequal(perm.source, sol.source)
    R2, W2, _ = label_sums(cat(rbar, rbar; dims = 3), cat(wbar, wbar; dims = 3), bl, pols, nant, vcat(times, times .+ 100))
    part = FRa.solve_adhoc_phasing(view(R2, Ti(1:nap)), view(W2, Ti(1:nap)), names; smoother = sm)
    @test isequal(parent(part.phase), parent(sol.phase))

    # Stations are matched by name: an extra, unobserved station is uncovered
    # and leaves the others untouched.
    wider = FRa.solve_adhoc_phasing(R, W, vcat(names, "X"); smoother = sm)
    @test !any(wider.covered[FRa.Ant(At("X"))])
    @test parent(wider.phase)[1:nant, :, :] ≈ parent(sol.phase) atol = 1.0e-12

    @test_throws "station `S5` of a station pair is not among the stations" FRa.solve_adhoc_phasing(
        R, W, names[1:4]; smoother = sm,
    )
    @test_throws "station names must be unique; repeated: S2" FRa.solve_adhoc_phasing(
        R, W, vcat(names, "S2"); smoother = sm,
    )
    shifted(A) = DimArray(
        OffsetArray(parent(A), 1, 0, 0),
        (FRa.StationPair(OffsetArray(parent(lookup(A, FRa.StationPair)), 1)), dims(A, FRa.FeedPair), dims(A, Ti)),
    )
    @test_throws "offset arrays are not supported" FRa.solve_adhoc_phasing(shifted(R), shifted(W), names; smoother = sm)
    @test_throws "must be over StationPair, FeedPair and Ti" FRa.solve_adhoc_phasing(
        DimArray(rbar, (FRa.BaselineID(1:length(bl)), FRa.Polarization(1:4), Ti(times))), W, names,
    )
end

@testset "Adhoc: the source alternation ignores per-station constants" begin
    rng = MersenneTwister(0x0176)
    nant = 5
    bl = all_bl_a(nant)
    pols = [(1, 1), (1, 2), (2, 1), (2, 2)]
    R, _, names = label_sums(zeros(ComplexF64, length(bl), 4, 2), ones(length(bl), 4, 2), bl, pols, nant, [0.0, 1.0])
    nodes = FRa._cell_nodes(R, names, CALa.PerFeed())
    x_prev = DimArray(randn(rng, length(bl), 4), dims(nodes))
    cell_w = map(_ -> 1.0 + rand(rng), x_prev)
    mask = map(_ -> true, x_prev)
    c = 0.1 .* randn(rng, nant, 2)
    shifted = map((x, ((a, na), (b, nb))) -> x + c[a, na] - c[b, nb], x_prev, nodes)
    @test FRa._source_move(shifted, x_prev, cell_w, mask, nodes, nant, 1) < 1.0e-12
    moved = copy(shifted)
    moved[3, 2] += 0.01
    @test 1.0e-3 < FRa._source_move(moved, x_prev, cell_w, mask, nodes, nant, 1) <= 0.01
end
