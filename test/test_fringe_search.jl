# Phase 3 — per-baseline FFT delay/rate fringe search.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test
using Random
using Statistics: mean

const FR = Gustavo.Fringe

# Build a noiseless fringe block V[chan,time] = A·exp(i[φ + 2πτ(f−f0) + 2πṙ(t−t0)]).
function inject_fringe(freqs, times, f0, t0; delay, rate, phase, amp = 1.0)
    V = Matrix{ComplexF64}(undef, length(freqs), length(times))
    for (ti, t) in enumerate(times), (ci, f) in enumerate(freqs)
        V[ci, ti] = amp * cis(phase + 2π * delay * (f - f0) + 2π * rate * (t - t0))
    end
    return V
end

@testset "Fringe search: noiseless delay/rate/phase recovery" begin
    nchan = 64
    Δf = 0.5e6                                  # 0.5 MHz channels
    freqs = 43.0e9 .+ (0:(nchan - 1)) .* Δf
    f0 = mean(freqs)
    nt = 30
    Δt = 1.0                                    # 1 s integrations
    times = (0:(nt - 1)) .* Δt
    t0 = mean(times)

    # Choose injected values well inside the (resolvable, unaliased) windows.
    τ_true = 12.0e-9                            # 12 ns
    ṙ_true = 8.0e-3                             # 8 mHz
    φ_true = 0.7

    V = inject_fringe(freqs, times, f0, t0; delay = τ_true, rate = ṙ_true, phase = φ_true)
    W = ones(size(V))
    det = FR.baseline_fringe_search(V, W, freqs, times, f0, t0)

    @test det.valid
    # Bin spacing sets the tolerance; quadratic interp pushes well below it.
    @test isapprox(det.delay, τ_true; atol = 1.0e-9)
    @test isapprox(det.rate, ṙ_true; atol = 5.0e-4)
    @test isapprox(rem2pi(det.phase - φ_true, RoundNearest), 0.0; atol = 1.0e-2)
    @test isapprox(det.amp, 1.0; rtol = 1.0e-2)
end

@testset "Fringe search: phase reference at (f0, t0)" begin
    nchan, nt = 48, 20
    Δf, Δt = 1.0e6, 2.0
    freqs = 86.0e9 .+ (0:(nchan - 1)) .* Δf
    times = (0:(nt - 1)) .* Δt
    # Reference at the band edge / first time (NOT the grid centre) to exercise
    # the phase rotation from grid origin to (f0, t0).
    f0, t0 = freqs[1], times[1]
    τ, ṙ, φ = -20.0e-9, -6.0e-3, -1.3
    V = inject_fringe(freqs, times, f0, t0; delay = τ, rate = ṙ, phase = φ)
    det = FR.baseline_fringe_search(V, ones(size(V)), freqs, times, f0, t0)
    @test isapprox(det.delay, τ; atol = 1.0e-9)
    @test isapprox(det.rate, ṙ; atol = 5.0e-4)
    @test isapprox(rem2pi(det.phase - φ, RoundNearest), 0.0; atol = 2.0e-2)
end

@testset "Fringe search: degenerate axes" begin
    # Single time → delay-only (rate must be 0).
    nchan = 64
    freqs = 43.0e9 .+ (0:(nchan - 1)) .* 0.5e6
    f0 = mean(freqs)
    Vmat = inject_fringe(freqs, [0.0], f0, 0.0; delay = 15.0e-9, rate = 0.0, phase = 0.4)
    det = FR.baseline_fringe_search(Vmat, ones(size(Vmat)), freqs, [0.0], f0, 0.0)
    @test det.rate == 0.0
    @test isapprox(det.delay, 15.0e-9; atol = 1.0e-9)
    @test isapprox(rem2pi(det.phase - 0.4, RoundNearest), 0.0; atol = 2.0e-2)

    # Single channel → rate-only (delay must be 0).
    nt = 40
    times = (0:(nt - 1)) .* 0.5
    t0 = mean(times)
    Vmat2 = inject_fringe([43.0e9], times, 43.0e9, t0; delay = 0.0, rate = 10.0e-3, phase = -0.2)
    det2 = FR.baseline_fringe_search(Vmat2, ones(size(Vmat2)), [43.0e9], times, 43.0e9, t0)
    @test det2.delay == 0.0
    @test isapprox(det2.rate, 10.0e-3; atol = 5.0e-4)

    # A vector that matches neither axis is a shape error, not a search failure.
    @test_throws DimensionMismatch FR.baseline_fringe_search(
        vec(Vmat2)[1:3], ones(3), [43.0e9], times, 43.0e9, t0)
    @test_throws "matches neither" FR.baseline_fringe_search(
        vec(Vmat2)[1:3], ones(3), [43.0e9], times, 43.0e9, t0)
end

@testset "Fringe search: shape validation" begin
    nchan, nt = 8, 4
    freqs = 43.0e9 .+ (0:(nchan - 1)) .* 0.5e6
    times = collect(0:(nt - 1)) .* 0.5
    V = ones(ComplexF64, nchan, nt)

    @test_throws DimensionMismatch FR.baseline_fringe_search(
        V, ones(nchan, nt - 1), freqs, times, mean(freqs), 0.0)
    @test_throws "same shape" FR.baseline_fringe_search(
        V, ones(nchan, nt - 1), freqs, times, mean(freqs), 0.0)
    @test_throws DimensionMismatch FR.baseline_fringe_search(
        V, ones(nchan, nt), freqs[1:(end - 1)], times, mean(freqs), 0.0)
    @test_throws "expected (length(freqs), length(times))" FR.baseline_fringe_map(
        V, ones(nchan, nt), freqs[1:(end - 1)], times, mean(freqs), 0.0)
end

@testset "Fringe search: multi-band gapped frequency axis" begin
    # Two 32-channel bands separated by a gap larger than a band → the search
    # must grid onto a common Δf and still recover the (wide-band) delay.
    Δf = 0.5e6
    band1 = 43.0e9 .+ (0:31) .* Δf
    band2 = 43.0e9 .+ 256 .* Δf .+ (0:31) .* Δf      # gap of ~224 channels
    freqs = vcat(band1, band2)
    f0 = mean(freqs)
    nt = 16
    times = (0:(nt - 1)) .* 1.0
    t0 = mean(times)
    τ, ṙ, φ = 5.0e-9, 4.0e-3, 0.9
    V = inject_fringe(freqs, times, f0, t0; delay = τ, rate = ṙ, phase = φ)
    det = FR.baseline_fringe_search(V, ones(size(V)), freqs, times, f0, t0)
    @test det.valid
    @test isapprox(det.delay, τ; atol = 1.0e-9)
    @test isapprox(det.rate, ṙ; atol = 5.0e-4)
end

@testset "Fringe search: flagged samples ignored, all-flagged invalid" begin
    nchan, nt = 64, 20
    freqs = 43.0e9 .+ (0:(nchan - 1)) .* 0.5e6
    f0 = mean(freqs)
    times = (0:(nt - 1)) .* 1.0
    t0 = mean(times)
    V = inject_fringe(freqs, times, f0, t0; delay = 9.0e-9, rate = 5.0e-3, phase = 0.1)
    W = ones(size(V))
    # Flag half the channels (and inject garbage there) — result must be unchanged.
    W[1:2:end, :] .= 0.0
    V[1:2:end, :] .= 1.0e6 .* cis(2.3)
    det = FR.baseline_fringe_search(V, W, freqs, times, f0, t0)
    @test det.valid
    @test isapprox(det.delay, 9.0e-9; atol = 1.5e-9)

    # All flagged → invalid, zeroed detection.
    det0 = FR.baseline_fringe_search(V, zeros(size(V)), freqs, times, f0, t0)
    @test !det0.valid
    @test det0.snr == 0.0
end

@testset "Fringe search: SNR scaling and threshold" begin
    rng = MersenneTwister(0xF1)
    nchan, nt = 128, 60
    freqs = 43.0e9 .+ (0:(nchan - 1)) .* 0.5e6
    f0 = mean(freqs)
    times = (0:(nt - 1)) .* 1.0
    t0 = mean(times)
    τ, ṙ = 7.0e-9, 3.0e-3

    # Per-sample noise σ; weights = 1/σ². Expected SNR ≈ A·√(N/σ²) = A·√(Σw).
    σ = 0.5
    nsamp = nchan * nt
    A = 1.0
    expected_snr = A * sqrt(nsamp / σ^2)

    snrs = Float64[]
    for trial in 1:8
        V = inject_fringe(freqs, times, f0, t0; delay = τ, rate = ṙ, phase = 0.0, amp = A)
        V .+= σ .* (randn(rng, size(V)) .+ im .* randn(rng, size(V))) ./ sqrt(2)
        W = fill(1 / σ^2, size(V))
        det = FR.baseline_fringe_search(V, W, freqs, times, f0, t0)
        push!(snrs, det.snr)
        @test det.valid
        @test det.pfa < 1.0e-4
        @test isapprox(det.delay, τ; atol = 2.0e-9)
    end
    # Measured SNR should track the matched-filter prediction within ~15%.
    @test isapprox(mean(snrs), expected_snr; rtol = 0.15)

    # Pure noise (no fringe): still MEASURED — the search reports the best peak it
    # found and leaves the verdict to `pfa`, which here is nowhere near a detection.
    Vn = σ .* (randn(rng, nchan, nt) .+ im .* randn(rng, nchan, nt)) ./ sqrt(2)
    detn = FR.baseline_fringe_search(Vn, fill(1 / σ^2, nchan, nt), freqs, times, f0, t0)
    @test detn.valid
    @test detn.pfa > 1.0e-4
end

@testset "Fringe search map + PFA" begin
    nchan, nt = 64, 30
    freqs = 43.0e9 .+ (0:(nchan - 1)) .* 0.5e6
    f0 = mean(freqs)
    times = (0:(nt - 1)) .* 1.0
    t0 = mean(times)
    τ, ṙ, φ = 12.0e-9, 8.0e-3, 0.7
    V = inject_fringe(freqs, times, f0, t0; delay = τ, rate = ṙ, phase = φ)
    W = ones(size(V))

    m = FR.baseline_fringe_map(V, W, freqs, times, f0, t0)
    @test m isa FR.FringeSearchMap
    @test size(m.snr) == (length(m.delays), length(m.rates))
    @test issorted(m.delays) && issorted(m.rates)
    opts = FR.FringeSearch()
    @test all(d -> opts.delay_window[1] <= d <= opts.delay_window[2], m.delays)
    @test all(r -> opts.rate_window[1] <= r <= opts.rate_window[2], m.rates)

    # The embedded detection is what the standalone search returns (≈ only
    # because the two calls plan separate FFTW MEASURE transforms).
    det = FR.baseline_fringe_search(V, W, freqs, times, f0, t0)
    @test isapprox(m.detection.delay, det.delay; rtol = 1.0e-10, atol = 1.0e-20)
    @test isapprox(m.detection.rate, det.rate; rtol = 1.0e-10, atol = 1.0e-15)
    @test isapprox(m.detection.snr, det.snr; rtol = 1.0e-10)
    @test m.detection.valid

    # The map's discrete peak sits at the injected (delay, rate) — within a grid
    # bin (quad refinement of the detection goes below the bin; the map does not).
    pk = argmax(m.snr)
    dbin = m.delays[2] - m.delays[1]
    rbin = m.rates[2] - m.rates[1]
    @test isapprox(m.delays[pk[1]], τ; atol = dbin)
    @test isapprox(m.rates[pk[2]], ṙ; atol = rbin)
    # ...and its height matches the refined detection SNR (the exact re-evaluated
    # peak is ≥ the discrete grid peak, but only marginally at oversample = 8).
    @test isapprox(maximum(m.snr), det.snr; rtol = 0.05)
    @test maximum(m.snr) <= det.snr * (1 + 1.0e-9)

    # A strong fringe is a secure detection.
    @test m.ncells >= 1
    @test m.pfa < 1.0e-10

    # Pure noise: the peak is consistent with the sidelobe forest (PFA not small).
    rng = MersenneTwister(0x000FA15E)
    Vn = (randn(rng, nchan, nt) .+ im .* randn(rng, nchan, nt)) ./ sqrt(2)
    mn = FR.baseline_fringe_map(Vn, W, freqs, times, f0, t0)
    @test mn.detection.valid                 # measured, not accepted
    @test mn.detection.pfa > 1.0e-3
    @test mn.pfa > 1.0e-3

    # All-flagged block → empty map, invalid detection.
    m0 = FR.baseline_fringe_map(V, zeros(size(V)), freqs, times, f0, t0)
    @test isempty(m0.delays) && isempty(m0.rates)
    @test !m0.detection.valid
    @test isnan(m0.pfa)

    # fringe_pfa: bounds, monotonicity, small-p linearization, edge cases.
    @test FR.fringe_pfa(0.0, 1.0e6) == 1.0
    @test FR.fringe_pfa(6.0, 1.0e6) < 1.0e-6
    @test FR.fringe_pfa(6.0, 1.0e6) > FR.fringe_pfa(7.0, 1.0e6)      # ↓ in snr
    @test FR.fringe_pfa(6.0, 1.0e8) > FR.fringe_pfa(6.0, 1.0e6)      # ↑ in ncells
    @test isapprox(FR.fringe_pfa(5.0, 1.0e3), 1.0e3 * exp(-25.0); rtol = 1.0e-6)
    @test FR.fringe_pfa(100.0, 1.0e12) == 0.0                        # underflow → secure
    @test isnan(FR.fringe_pfa(NaN, 10.0))
    @test 0.0 <= FR.fringe_pfa(2.0, 1.0e4) <= 1.0
end

@testset "Hierarchical MBD search (VGOS-style)" begin
    # 8 narrow bands (16 ch × 1 MHz) with origins every 100 MHz: the common-Δf
    # grid would be 716 bins for 128 real channels (>4× → mostly zeros), so
    # :auto picks the hierarchical path. MBD ambiguity A = 1/100 MHz = 10 ns.
    Δf = 1.0e6
    nfreqgroup, nchan_b, nt = 8, 16, 24
    freqs = Float64[]
    for b in 0:(nfreqgroup - 1)
        append!(freqs, 8.0e9 .+ b * 100.0e6 .+ (0:(nchan_b - 1)) .* Δf)
    end
    f0 = mean(freqs)
    times = (0:(nt - 1)) .* 1.0
    t0 = mean(times)

    auto = FR.FringeSearch()
    full = FR.FringeSearch(algorithm = FR.FullGrid())
    mbd = FR.FringeSearch(algorithm = FR.HierarchicalMBD())
    @test FR._search_axes(freqs, times, auto, ComplexF64).mbd !== nothing     # auto → hierarchical
    @test FR._search_axes(freqs, times, full, ComplexF64).mbd === nothing

    # Delay spanning MANY ambiguities (137.3 ns ≈ 8.8 × A) — the arbitration must
    # unfold it; plus a rate and phase.
    τ, ṙ, φ = 137.3e-9, 6.0e-3, -0.9
    V = inject_fringe(freqs, times, f0, t0; delay = τ, rate = ṙ, phase = φ, amp = 0.7)
    W = ones(size(V))

    dm = FR.baseline_fringe_search(V, W, freqs, times, f0, t0; opts = mbd)
    df = FR.baseline_fringe_search(V, W, freqs, times, f0, t0; opts = full)
    da = FR.baseline_fringe_search(V, W, freqs, times, f0, t0; opts = auto)
    @test da.delay == dm.delay                                    # auto took the mbd path
    for d in (dm, df)
        @test d.valid
        @test isapprox(d.delay, τ; atol = 0.5e-9)
        @test isapprox(d.rate, ṙ; atol = 5.0e-4)
        @test isapprox(rem2pi(d.phase - φ, RoundNearest), 0.0; atol = 2.0e-2)
        @test isapprox(d.amp, 0.7; rtol = 2.0e-2)
    end
    # The two algorithms agree with each other (same matched filter; the exact-
    # filter polish makes the mbd delay slightly MORE accurate than the FFT quad).
    @test isapprox(dm.delay, df.delay; atol = 0.5e-9)
    @test isapprox(dm.rate, df.rate; atol = 2.0e-4)

    # With thermal noise the hierarchical path still finds and unfolds the
    # fringe, and the two paths' data-driven SNRs agree (the noise-dominated
    # regime is where the SNR convention is defined).
    rng = MersenneTwister(0x000BEEF5)
    σ = 0.5
    Vn = V .+ σ .* (randn(rng, size(V)) .+ im .* randn(rng, size(V))) ./ sqrt(2)
    Wn = fill(1 / σ^2, size(V))
    dn = FR.baseline_fringe_search(Vn, Wn, freqs, times, f0, t0; opts = mbd)
    dnf = FR.baseline_fringe_search(Vn, Wn, freqs, times, f0, t0; opts = full)
    @test dn.valid
    @test isapprox(dn.delay, τ; atol = 1.0e-9)
    @test isapprox(dn.rate, ṙ; atol = 5.0e-4)
    @test isapprox(dn.snr, dnf.snr; rtol = 0.15)

    # Flagged channels (kill two whole bands + garbage) — result unchanged.
    Vf = copy(V); Wf = copy(W)
    Wf[1:(2 * nchan_b), :] .= 0.0
    Vf[1:(2 * nchan_b), :] .= 1.0e6 .* cis(1.1)
    dflag = FR.baseline_fringe_search(Vf, Wf, freqs, times, f0, t0; opts = mbd)
    @test dflag.valid
    @test isapprox(dflag.delay, τ; atol = 0.5e-9)

    # All flagged → invalid.
    d0 = FR.baseline_fringe_search(V, zeros(size(V)), freqs, times, f0, t0; opts = mbd)
    @test !d0.valid

    # Explicit :mbd on a CONTIGUOUS band falls back to the full path (single
    # block → no hierarchy), with identical results by construction.
    fc = 43.0e9 .+ (0:63) .* 0.5e6
    @test FR._search_axes(fc, times, mbd, ComplexF64).mbd === nothing
    Vc = inject_fringe(fc, times, mean(fc), t0; delay = 9.0e-9, rate = 3.0e-3, phase = 0.2)
    d1 = FR.baseline_fringe_search(Vc, ones(size(Vc)), fc, times, mean(fc), t0; opts = mbd)
    d2 = FR.baseline_fringe_search(Vc, ones(size(Vc)), fc, times, mean(fc), t0; opts = full)
    @test d1.delay == d2.delay && d1.snr == d2.snr

    # Degenerate time axis (one AP): delay-only search through the mbd path.
    V1 = inject_fringe(freqs, [0.0], f0, 0.0; delay = 40.0e-9, rate = 0.0, phase = 0.3)
    dd = FR.baseline_fringe_search(V1, ones(size(V1)), freqs, [0.0], f0, 0.0; opts = mbd)
    @test dd.rate == 0.0
    @test isapprox(dd.delay, 40.0e-9; atol = 0.5e-9)

    # VGOS-like MIXED origin spacings (64/96/192 MHz, as in real VR2505 data):
    # the common divisor (32 MHz) is not reachable by halving any single spacing
    # — the approximate-GCD grid fit must find it, and the ambiguity follows it.
    fv = Float64[]
    for off in (0.0, 192.0e6, 288.0e6, 352.0e6, 416.0e6)   # diffs 192/96/64/64
        append!(fv, 3.0e9 .+ off .+ (0:15) .* Δf)
    end
    axv = FR._search_axes(fv, times, mbd, ComplexF64)
    @test axv.mbd !== nothing
    @test axv.mbd.bc_step ≈ 32.0e6 rtol = 1.0e-9            # true GCD, not median/2^k
    @test axv.mbd.ambig ≈ 1 / 32.0e6 rtol = 1.0e-9
    f0v = mean(fv)
    τv = 55.0e-9                                            # ≫ A = 31.25 ns
    Vv = inject_fringe(fv, times, f0v, t0; delay = τv, rate = 2.0e-3, phase = 0.5)
    dv = FR.baseline_fringe_search(Vv, ones(size(Vv)), fv, times, f0v, t0; opts = mbd)
    dvf = FR.baseline_fringe_search(Vv, ones(size(Vv)), fv, times, f0v, t0; opts = full)
    @test dv.valid
    @test isapprox(dv.delay, τv; atol = 0.5e-9)
    @test isapprox(dv.delay, dvf.delay; atol = 0.5e-9)

    # Map diagnostic on multi-band data: plane from the full grid, detection from
    # the (auto → mbd) search. On NOISY data (where the data-driven noise
    # estimate is defined) the map peak tracks the detection SNR; the noiseless
    # limit is degenerate (both "noise" estimates measure different sidelobe
    # floors), so compare there.
    m = FR.baseline_fringe_map(V, W, freqs, times, f0, t0; opts = auto)
    @test m.detection.delay == da.delay                            # detection = solver's search
    mn2 = FR.baseline_fringe_map(Vn, Wn, freqs, times, f0, t0; opts = auto)
    @test isapprox(mn2.detection.delay, dn.delay; atol = 1.0e-12)
    @test isapprox(maximum(mn2.snr), mn2.detection.snr; rtol = 0.15)
end

@testset "Fringe search: ComplexF32 runs natively (not silently upcast)" begin
    nchan, nt = 64, 30
    freqs = 43.0e9 .+ (0:(nchan - 1)) .* 0.5e6
    f0 = mean(freqs)
    times = (0:(nt - 1)) .* 1.0
    t0 = mean(times)
    τ_true, ṙ_true, φ_true = 12.0e-9, 8.0e-3, 0.7

    V64 = inject_fringe(freqs, times, f0, t0; delay = τ_true, rate = ṙ_true, phase = φ_true)
    V32 = ComplexF32.(V64)
    W32 = ones(Float32, size(V32))

    # The search grid, FFT plan, and workspace buffers must be native ComplexF32
    # — not the pre-CHUNK-084 behavior of silently upcasting into ComplexF64 —
    # since that's the whole point of flowing the compute type from `V`'s eltype.
    ax = FR._search_axes(freqs, times, FR.FringeSearch(), ComplexF32)
    @test eltype(ax.delays) === Float32
    @test eltype(ax.rates) === Float32

    ws = FR.FringeWorkspace(ComplexF32)
    det = FR.baseline_fringe_search(V32, W32, freqs, times, f0, t0; workspace = ws)
    @test eltype(ws.G) === ComplexF32
    @test eltype(ws.D) === ComplexF32
    @test eltype(ws.dwin) === Float32
    @test det isa FR.Detection{Float32}

    @test det.valid
    @test isapprox(det.delay, τ_true; atol = 5.0f-9)
    @test isapprox(det.rate, ṙ_true; atol = 5.0f-4)
    @test isapprox(rem2pi(det.phase - φ_true, RoundNearest), 0.0; atol = 2.0f-2)
    @test isapprox(det.amp, 1.0f0; rtol = 2.0f-2)

    # The hierarchical (MBD) path also stays native ComplexF32 in its workspace.
    Δf = 1.0e6
    nfreqgroup, nchan_b = 8, 16
    freqs_m = Float64[]
    for b in 0:(nfreqgroup - 1)
        append!(freqs_m, 8.0e9 .+ b * 100.0e6 .+ (0:(nchan_b - 1)) .* Δf)
    end
    f0m = mean(freqs_m)
    Vm64 = inject_fringe(freqs_m, times, f0m, t0; delay = 137.3e-9, rate = ṙ_true, phase = φ_true, amp = 0.7)
    Vm32 = ComplexF32.(Vm64)
    Wm32 = ones(Float32, size(Vm32))
    mbd = FR.FringeSearch(algorithm = FR.HierarchicalMBD())
    wsm = FR.FringeWorkspace(ComplexF32)
    detm = FR.baseline_fringe_search(Vm32, Wm32, freqs_m, times, f0m, t0; opts = mbd, workspace = wsm)
    @test detm isa FR.Detection{Float32}
    @test wsm.mbd isa FR._MBDWorkspace{ComplexF32}
    @test eltype(wsm.mbd.Gb) === ComplexF32
    @test detm.valid
    @test isapprox(detm.delay, 137.3e-9; atol = 5.0f-9)
end

# An algorithm defined OUTSIDE the package — the only thing that proves
# `AbstractSearchAlgorithm` is a real extension point rather than a declared one.
struct _ProbeFullGrid <: FR.AbstractSearchAlgorithm end
FR._mbd_axes(::_ProbeFullGrid, freqs, fax, tax, rates, opts, ::Type{C}) where {C} = nothing

# Subtypes the seam but implements nothing.
struct _ProbeUnimplemented <: FR.AbstractSearchAlgorithm end

@testset "Search algorithm seam" begin
    # A VGOS-style axis: 8 narrow bands spread over a wide span, so :auto
    # resolves to the hierarchical path and the two built-ins differ.
    Δf = 1.0e6
    freqs = Float64[]
    for b in 0:7
        append!(freqs, 8.0e9 .+ b * 100.0e6 .+ (0:15) .* Δf)
    end
    times = (0:23) .* 1.0
    fax = FR._uniform_axis(freqs)

    @test FR._resolve_algorithm(:auto, freqs, fax) isa FR.HierarchicalMBD
    # An explicit algorithm passes through untouched, including a foreign one.
    for alg in (FR.FullGrid(), FR.HierarchicalMBD(), _ProbeFullGrid())
        @test FR._resolve_algorithm(alg, freqs, fax) === alg
    end
    # A contiguous axis has nothing to decompose, so :auto stays on the full grid.
    contig = collect(8.0e9 .+ (0:127) .* Δf)
    @test FR._resolve_algorithm(:auto, contig, FR._uniform_axis(contig)) isa FR.FullGrid

    # The foreign algorithm reaches the search and selects the full-grid path.
    probe = FR.FringeSearch(algorithm = _ProbeFullGrid())
    @test FR._search_axes(freqs, times, probe, ComplexF64).mbd === nothing
    f0, t0 = mean(freqs), mean(times)
    V = inject_fringe(freqs, times, f0, t0; delay = 13.7e-9, rate = 6.0e-3, phase = -0.9)
    W = ones(Float64, size(V))
    d_probe = FR.baseline_fringe_search(V, W, freqs, times, f0, t0; opts = probe)
    d_full = FR.baseline_fringe_search(
        V, W, freqs, times, f0, t0; opts = FR.FringeSearch(algorithm = FR.FullGrid()))
    @test d_probe.delay == d_full.delay
    @test d_probe.snr == d_full.snr

    # No silent fallback: an algorithm with no `_mbd_axes` method is an error,
    # not a quiet switch to a different search.
    @test_throws "defines no `Gustavo.Fringe._mbd_axes` method" FR._search_axes(
        freqs, times, FR.FringeSearch(algorithm = _ProbeUnimplemented()), ComplexF64)

    # The sentinel is the ONLY Symbol accepted; the retired :full/:mbd names
    # fail loudly rather than being silently reinterpreted.
    for bogus in (:full, :mbd, :bogus)
        opts = FR.FringeSearch(algorithm = bogus)
        @test_throws ArgumentError FR._search_axes(freqs, times, opts, ComplexF64)
        @test_throws "algorithm must be :auto or an AbstractSearchAlgorithm" FR._search_axes(
            freqs, times, opts, ComplexF64)
    end
end
