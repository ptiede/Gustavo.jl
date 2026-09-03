# ── coherence_report: the debiased estimator is unbiased at low SNR ───────────
#
# The debiased η pools unbiased per-bin signal POWERS `(|Σ w·V|² − 2Σw)/Σw` and
# takes a single square root on the pooled ratio. The properties fixed here:
# a perfectly coherent baseline reads η ≈ 1 at EVERY averaging interval no
# matter how weak (a per-bin clipped amplitude draws a dip-and-recover curve at
# bin SNR ≲ 1); pure-noise baselines contribute zero in expectation to the
# pooled aggregate instead of dragging it down (their own trace reads NaN when
# their measured power is not positive); and genuine decoherence is still
# measured at its analytic value.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

# A fixture leaf overwritten with controlled signal and noise: unit signal with
# a per-cell complex SNR of `snr_cell`, a linear phase ramp of `phase_ramp`
# radians across the scan on baseline 1 only, and `noise_only_bls` carrying no
# signal at all. Weights are inverse variances per real component times
# `weight_scale` (1 = honest; ≠1 exercises the measured-noise-scale debias).
function _coherence_fixture_report(;
        snr_cell, phase_ramp = 0.0, noise_only_bls = Int[], weight_scale = 1.0, seed = 7,
    )
    UV = Gustavo.UVData
    uvset, _ = _build_fringe_uvset(nant = 3, nspw = 1, nchan = 64, ntime = 128)
    leaf = first(values(UV.branches(uvset)))
    V = parent(leaf[:vis]); W = parent(leaf[:weights])   # (chan, ti, bl, pol)
    rng = MersenneTwister(seed)
    nchan, nti, nbl, npol = size(V)
    sigma = 1.0 / (sqrt(2) * snr_cell)
    w = 1 / sigma^2
    for bl in 1:nbl, p in 1:npol, ti in 1:nti, c in 1:nchan
        amp = bl in noise_only_bls ? 0.0 : 1.0
        ph = bl == 1 ? phase_ramp * (ti - 1) / (nti - 1) : 0.0
        V[c, ti, bl, p] = amp * cis(ph) + sigma * (randn(rng) + im * randn(rng))
        W[c, ti, bl, p] = w * weight_scale
    end
    return UV.coherence_report(uvset; debias = true, marginalize = false)
end

@testset "coherence_report: debiased η is unbiased at low SNR" begin
    # Weak but perfectly coherent: flat ≈1 at every interval, both axes.
    rep = _coherence_fixture_report(snr_cell = 0.3)
    @test all(η -> 0.95 <= η <= 1.0, rep.time.eta)
    @test all(η -> 0.95 <= η <= 1.0, rep.freq.eta)

    # One real baseline pooled with two pure-noise ones: the aggregate stays
    # signal-driven, the strong baseline's own trace stays ≈1, and a noise
    # baseline's denominator fails the 3σ power-detection guard — NaN, never a
    # fabricated (clamped) η.
    rep = _coherence_fixture_report(snr_cell = 0.3, noise_only_bls = [2, 3])
    @test all(η -> 0.9 <= η <= 1.0, rep.time.eta)
    @test rep.time.eta_baseline[end, 1] > 0.9
    @test isnan(rep.time.eta_baseline[end, 2])
    @test isnan(rep.time.eta_baseline[end, 3])

    # A dishonest WEIGHT column (30% optimistic in variance) must not corrupt
    # η: the absolute noise scale is measured from adjacent-sample differences,
    # the weights only set relative cell weighting.
    rep = _coherence_fixture_report(snr_cell = 0.3, weight_scale = 1.3)
    @test all(η -> 0.95 <= η <= 1.0, rep.time.eta)
    rep = _coherence_fixture_report(snr_cell = 0.3, weight_scale = 0.7)
    @test all(η -> 0.95 <= η <= 1.0, rep.time.eta)

    # Genuine decoherence at high SNR: a π phase ramp across the scan on
    # baseline 1 gives full-scan η = |sinc(1/2)| = 2/π there; the flat
    # baselines stay at 1.
    rep = _coherence_fixture_report(snr_cell = 20.0, phase_ramp = Float64(π))
    @test isapprox(rep.time.eta_baseline[end, 1], 2 / π; atol = 0.02)
    @test rep.time.eta_baseline[end, 2] > 0.99
    @test rep.time.eta_baseline[end, 3] > 0.99

    # The raw (debias = false) estimator is untouched: bounded by 1 and
    # noise-suppressed below it at coarse averaging on weak data.
    uvset, _ = _build_fringe_uvset(nant = 3, nspw = 1, nchan = 16, ntime = 32)
    raw = Gustavo.UVData.coherence_report(uvset; debias = false, marginalize = false)
    @test all(η -> 0.0 <= η <= 1.0, filter(isfinite, raw.time.eta))
end
