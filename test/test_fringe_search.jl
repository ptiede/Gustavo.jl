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
        det = FR.baseline_fringe_search(V, W, freqs, times, f0, t0; opts = FR.FringeSearch(snr_min = 6.0))
        push!(snrs, det.snr)
        @test det.valid
        @test isapprox(det.delay, τ; atol = 2.0e-9)
    end
    # Measured SNR should track the matched-filter prediction within ~15%.
    @test isapprox(mean(snrs), expected_snr; rtol = 0.15)

    # Pure noise (no fringe) should usually fall below a high threshold.
    Vn = σ .* (randn(rng, nchan, nt) .+ im .* randn(rng, nchan, nt)) ./ sqrt(2)
    detn = FR.baseline_fringe_search(
        Vn, fill(1 / σ^2, nchan, nt), freqs, times, f0, t0;
        opts = FR.FringeSearch(snr_min = 7.0)
    )
    @test !detn.valid
end
