# Phase 1 — unified Calibration framework core.
# Standalone-runnable (`julia --project=. test/test_calibration.jl`) and included
# from runtests.jl.

using Gustavo
using Test
using LinearAlgebra

const CAL = Gustavo.Calibration

@testset "Calibration segmentation ids" begin
    # 5 times across 2 scans; 6 channels across 2 spws.
    geom = CAL.DataGeometry(;
        times = [0.0, 0.1, 0.2, 1.0, 1.1],
        scan_of_time = [7, 7, 7, 9, 9],          # arbitrary labels → dense-ranked
        channel_freqs = collect(1.0:6.0) .* 1.0e9,
        spw_of_chan = [2, 2, 2, 5, 5, 5],
        t0 = 0.0, f0 = 3.5e9,
    )

    @test CAL.time_segment_ids(CAL.GlobalTime(), geom) == (fill(1, 5), 1)
    @test CAL.time_segment_ids(CAL.PerIntegration(), geom) == (collect(1:5), 5)
    @test CAL.time_segment_ids(CAL.PerScan(), geom) == ([1, 1, 1, 2, 2], 2)

    # TimeBlocks of 0.5 h: t=0,0.1,0.2 → block 0; t=1.0,1.1 → block 2 → dense-rank 1,2.
    ids, n = CAL.time_segment_ids(CAL.TimeBlocks(0.5), geom)
    @test ids == [1, 1, 1, 2, 2] && n == 2

    # InstrumentScans boundary at 0.5 h splits the same way.
    @test CAL.time_segment_ids(CAL.InstrumentScans([0.5]), geom) == ([1, 1, 1, 2, 2], 2)

    @test CAL.freq_segment_ids(CAL.GlobalFrequency(), geom) == (fill(1, 6), 1)
    @test CAL.freq_segment_ids(CAL.PerSpectralWindow(), geom) == ([1, 1, 1, 2, 2, 2], 2)
    # ChannelBlocks(2) within each spw: spw1 → [1,1,2], spw2 → [3,3,4].
    @test CAL.freq_segment_ids(CAL.ChannelBlocks(2), geom) == ([1, 1, 2, 3, 3, 4], 4)
    # ChannelBlocks never straddles a spw: block_size 4 still splits at the spw edge.
    @test CAL.freq_segment_ids(CAL.ChannelBlocks(4), geom) == ([1, 1, 1, 2, 2, 2], 2)

    @test CAL.segment_groups([1, 1, 2, 3, 3, 4], 4) == [[1, 2], [3], [4, 5], [6]]
end

@testset "Calibration terms: nparams, basis, eval" begin
    @test CAL.nparams_per_block(CAL.ConstantTerm(), 7) == 1
    @test CAL.nparams_per_block(CAL.Delay(), 7) == 1
    @test CAL.nparams_per_block(CAL.Rate(), 7) == 1
    @test CAL.nparams_per_block(CAL.PolynomialFreq(3), 7) == 3
    @test CAL.nparams_per_block(CAL.PolynomialTime(2), 7) == 2
    @test CAL.nparams_per_block(CAL.PerChannel(), 7) == 7

    # basis_columns shapes
    x = collect(-1.0:0.5:1.0)              # 5 samples
    @test size(CAL.basis_columns(CAL.ConstantTerm(), x)) == (5, 1)
    @test CAL.basis_columns(CAL.ConstantTerm(), x) == reshape(ones(5), :, 1)
    @test CAL.basis_columns(CAL.Delay(), x) ≈ reshape(2π .* x, :, 1)
    @test size(CAL.basis_columns(CAL.PolynomialFreq(3), x)) == (5, 3)
    @test CAL.basis_columns(CAL.PolynomialFreq(2), x)[:, 1] ≈ x
    @test CAL.basis_columns(CAL.PolynomialFreq(2), x)[:, 2] ≈ x .^ 2
    @test CAL.basis_columns(CAL.PerChannel(), x) == Matrix(I, 5, 5)

    # scalar term_eval primitives: p = ParamBlock(θ, off) (params indexed from 1),
    # plus the term's coordinate (freq/time) or the local channel index.
    θ = [0.3, 1.5, -0.7]
    @test CAL.term_eval(CAL.ConstantTerm(), CAL.ParamBlock(θ, 2)) == 1.5
    @test CAL.term_eval(CAL.Delay(), CAL.ParamBlock(θ, 1), 4.0) ≈ 2π * 0.3 * 4.0
    @test CAL.term_eval(CAL.Rate(), CAL.ParamBlock(θ, 1), 5.0) ≈ 2π * 0.3 * 5.0
    @test CAL.term_eval(CAL.PerChannel(), CAL.ParamBlock(θ, 1), 3) == θ[3]
    @test CAL.term_eval(CAL.PolynomialFreq(3), CAL.ParamBlock(θ, 1), 2.0) ≈ 0.3 * 2 + 1.5 * 4 + (-0.7) * 8
    # The family adapter forwards the right coordinate to each term, given the
    # block's backing store + offset (it builds the ParamBlock internally).
    @test CAL._term_contribution(CAL.Delay(), θ, 1, 4.0, 0.0, 1) ≈ 2π * 0.3 * 4.0
    @test CAL._term_contribution(CAL.Rate(), θ, 1, 0.0, 5.0, 1) ≈ 2π * 0.3 * 5.0
    @test CAL._term_contribution(CAL.PerChannel(), θ, 1, 0.0, 0.0, 3) == θ[3]
end

@testset "Calibration: named parameters + bounds-checked block" begin
    # A term that names its parameters: names are the single source of truth —
    # the per-block count derives from them and they're read by name.
    @eval CAL begin
        struct _TwoParamTerm <: FrequencyTerm end
        param_names(::_TwoParamTerm) = (:a, :b)
        term_eval(::_TwoParamTerm, p, x) = p.a + p.b * x
    end
    t = CAL._TwoParamTerm()
    @test CAL.param_names(t) == (:a, :b)
    @test CAL.nparams_per_block(t, 4) == 2                 # derived from names

    θ = [10.0, 3.0]
    p = CAL.ParamBlock{(:a, :b)}(θ, 1)
    @test p.a == 10.0 && p.b == 3.0                        # named access
    @test p[1] == 10.0 && p[2] == 3.0                      # positional still works
    @test_throws BoundsError p[3]                          # over-index errors (was silent)
    @test CAL.term_eval(t, p, 2.0) ≈ 10.0 + 3.0 * 2.0

    # An unnamed single-parameter term defaults to one parameter, positional.
    @test CAL.param_names(CAL.Delay()) == ()
    @test CAL.nparams_per_block(CAL.Delay(), 4) == 1
end

@testset "Calibration feed tying offset algebra" begin
    geom = CAL.DataGeometry(; times = [0.0, 1.0], channel_freqs = [1.0e9, 2.0e9])
    # One ConstantTerm, GlobalTime × GlobalFrequency, 2 antennas.
    mk(tying) = CAL.StationGainModel(
        phase = (CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), tying),),
    )

    lay_pf = CAL.plan_parameters(mk(CAL.PerFeed()), 2, geom)
    @test lay_pf.nθ == 4                                    # 2 ant × 2 feeds
    p = lay_pf.plans[1]
    @test p.off1[1, 1, 1, 1] != p.off1[1, 2, 1, 1]          # feeds independent
    @test all(p.off2 .== 0)

    lay_sf = CAL.plan_parameters(mk(CAL.SharedFeeds()), 2, geom)
    @test lay_sf.nθ == 2                                    # 2 ant, shared across feeds
    q = lay_sf.plans[1]
    @test q.off1[1, 1, 1, 1] == q.off1[1, 2, 1, 1]          # feeds share a block
    @test all(q.off2 .== 0)

    lay_rr = CAL.plan_parameters(mk(CAL.ReferenceRelative(1)), 2, geom)
    @test lay_rr.nθ == 4                                    # ref + relative per ant
    r = lay_rr.plans[1]
    @test r.off1[1, 1, 1, 1] == r.off1[1, 2, 1, 1]          # both feeds reference the ref block
    @test r.off2[1, 1, 1, 1] == 0                           # reference feed has no relative
    @test r.off2[1, 2, 1, 1] != 0                           # partner feed adds a relative block
end

@testset "Calibration evaluate_gains: correctness, purity, inference" begin
    nant = 3
    freqs = collect(2.28e11:1.0e8:(2.28e11 + 5.0e8))         # 6 channels
    f0 = sum(freqs) / length(freqs)
    times = [0.0, 0.5, 1.0]
    geom = CAL.DataGeometry(; times, channel_freqs = freqs, t0 = 0.0, f0)

    # Pure per-feed delay model: phase = 2π τ (f − f0), one τ per (ant, feed).
    model = CAL.StationGainModel(
        phase = (CAL.TiedComponent(CAL.GainComponent(CAL.Delay(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
    )
    ev = CAL.GainEvaluator(model, geom; nant)
    @test CAL.nparameters(ev) == nant * 2

    τ = 1.0e-9 .* collect(1:(nant * 2))                    # distinct delays in ns
    g = CAL.evaluate_gains(ev, τ)
    @test size(g) == (length(freqs), length(times), nant, 2)

    # Hand-check a couple of cells: |g| == 1 (no amplitude), phase = 2π τ (f−f0).
    p = ev.layout.plans[1]
    for ant in 1:nant, feed in 1:2, ci in eachindex(freqs)
        off = p.off1[ant, feed, 1, 1]
        expected = cis(2π * τ[off] * (freqs[ci] - f0))
        @test g[ci, 1, ant, feed] ≈ expected
        @test g[ci, 2, ant, feed] ≈ expected            # time-invariant (GlobalTime)
    end

    # Purity: θ untouched, repeated evaluation bit-identical, no aliasing.
    τ_copy = copy(τ)
    g2 = CAL.evaluate_gains(ev, τ)
    @test τ == τ_copy
    @test g == g2
    @test g !== g2

    # Type stability of the forward map.
    @inferred CAL.evaluate_gains(ev, τ)

    # Rate term: phase grows linearly in time, flat in frequency.
    rate_model = CAL.StationGainModel(
        phase = (CAL.TiedComponent(CAL.GainComponent(CAL.Rate(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.SharedFeeds()),),
    )
    evr = CAL.GainEvaluator(rate_model, geom; nant)
    ṙ = 1.0e-3 .* collect(1:nant)                          # mHz-scale rates
    gr = CAL.evaluate_gains(evr, ṙ)
    pr = evr.layout.plans[1]
    for ant in 1:nant, ti in eachindex(times)
        off = pr.off1[ant, 1, 1, 1]
        expected = cis(2π * ṙ[off] * (times[ti] - 0.0) * 3600.0)
        @test gr[1, ti, ant, 1] ≈ expected
        @test gr[1, ti, ant, 2] ≈ expected               # shared across feeds
    end
end

@testset "Calibration predict_visibilities closes" begin
    nant = 3
    geom = CAL.DataGeometry(; times = [0.0], channel_freqs = [2.28e11, 2.281e11], t0 = 0.0)
    model = CAL.StationGainModel(
        phase = (CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
        logamp = (CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
    )
    ev = CAL.GainEvaluator(model, geom; nant)
    θ = randn(CAL.nparameters(ev))
    g = CAL.evaluate_gains(ev, θ)

    bl_pairs = [(1, 2), (1, 3), (2, 3)]
    bl_a = first.(bl_pairs)
    bl_b = last.(bl_pairs)
    # Parallel + cross hands: PP, PQ, QP, QQ → feed pairs.
    pol_products = ["PP", "PQ", "QP", "QQ"]
    feed_a = [CAL.correlation_feed_pair(p)[1] for p in pol_products]
    feed_b = [CAL.correlation_feed_pair(p)[2] for p in pol_products]

    coh = zeros(ComplexF64, length(bl_pairs), 2, 2)
    for bi in eachindex(bl_pairs)
        coh[bi, :, :] .= ComplexF64[1.0 0.1; 0.1 0.9]
    end
    V = CAL.predict_visibilities(g, coh, bl_a, bl_b, feed_a, feed_b)
    @test size(V) == (2, 1, 3, 4)

    # Phase closure of the gain factors on triangle (1,2,3) for the PP product:
    # arg(V12) + arg(V23) − arg(V13), with a point source the source phase is 0,
    # so the closure phase is exactly 0.
    pp = 1
    for c in 1:2
        cphase = angle(V[c, 1, 1, pp]) + angle(V[c, 1, 3, pp]) - angle(V[c, 1, 2, pp])
        # source coherency PP is real positive → contributes 0 to closure.
        @test isapprox(rem2pi(cphase, RoundNearest), 0.0; atol = 1.0e-10)
    end
end

# ── Audit-driven regression tests ────────────────────────────────────────────

@testset "Calibration parallel_hand_indices errors when PP/QQ missing" begin
    @test CAL.parallel_hand_indices(["PP", "PQ", "QP", "QQ"]) == (1, 4)
    # Regression for the operator-precedence bug: a missing PP must error, not
    # silently return (nothing, idx).
    @test_throws ErrorException CAL.parallel_hand_indices(["RL", "QQ"])
    @test_throws ErrorException CAL.parallel_hand_indices(["PP", "RL"])
    @test_throws ErrorException CAL.parallel_hand_indices(["RR", "LL"])
end

@testset "Calibration savitzky_golay_smooth: isolated finite sample" begin
    # Regression: a window containing one finite sample (ord = 0) must not crash.
    y = [NaN, 1.5, NaN]
    out = CAL.savitzky_golay_smooth(y, [0.0, 2.0, 0.0]; window = 7, order = 2)
    @test out[2] ≈ 1.5
    @test all(isfinite, out)                      # gaps interpolated from the one sample
    # A fully-finite smooth track is preserved.
    t = sin.(range(0, 2π; length = 30))
    sm = CAL.savitzky_golay_smooth(t; window = 7, order = 2)
    @test maximum(abs.(sm .- t)) < 0.1
end

@testset "Calibration: misdeclared term errors loudly (N3)" begin
    geom = CAL.DataGeometry(; times = [0.0], channel_freqs = [1.0e9, 2.0e9])

    # (1) NEW guard: a term that subtypes AbstractGainTerm directly (forgot to
    #     pick a family) has no coord_kind → errors at plan time, never guesses.
    @eval CAL begin
        struct _AuditNoFamilyTerm <: AbstractGainTerm end
    end
    model1 = CAL.StationGainModel(
        phase = (CAL.TiedComponent(CAL.GainComponent(CAL._AuditNoFamilyTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
    )
    @test_throws MethodError CAL.plan_parameters(model1, 1, geom)

    # (2) A term that claims to read frequency but defines no freq_coordinate must
    #     also error, not silently evaluate with xf = 0. (Frequency terms get a
    #     ν − f0 default; this one bypasses it by subtyping the root + declaring
    #     coord_kind manually — so no matching coordinate builder exists.)
    @eval CAL begin
        struct _AuditBadFreqTerm <: AbstractGainTerm end
        coord_kind(::_AuditBadFreqTerm) = COORD_FREQ
        # NOTE: deliberately no freq_coordinate method.
    end
    model2 = CAL.StationGainModel(
        phase = (CAL.TiedComponent(CAL.GainComponent(CAL._AuditBadFreqTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
    )
    @test_throws MethodError CAL.plan_parameters(model2, 1, geom)
end
