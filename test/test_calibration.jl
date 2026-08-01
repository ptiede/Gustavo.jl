# Phase 1 — unified Calibration framework core.
# Standalone-runnable (`julia --project=test test/test_calibration.jl`) and
# included from runtests.jl.

using Gustavo
using Test
using LinearAlgebra
using DimensionalData: DimArray, Dim, lookup, Ti, name, dims
using Statistics: mean
import OffsetArrays

const UVD = Gustavo.UVData

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

@testset "Calibration terms: names, nparams, eval" begin
    # A term names its parameters and shapes them; the count is derived, so it
    # cannot disagree with the names.
    @test CAL.param_shapes(CAL.Delay(), 7) == (delay = (),)
    @test CAL.param_shapes(CAL.PolynomialFreq(3), 7) == (coeffs = (3,),)

    # One `Polynomial` term; the axis it reads is a type parameter, and the
    # convenience constructors are functions returning it.
    @test CAL.PolynomialFreq(3) isa CAL.Polynomial{:Frequency}
    @test CAL.PolynomialTime(3) isa CAL.Polynomial{:Ti}
    @test CAL.term_axes(CAL.PolynomialFreq(3)) == (:Frequency,)
    @test CAL.term_axes(CAL.PolynomialTime(3)) == (:Ti,)
    @test CAL.term_label(CAL.PolynomialFreq(3)) == "polyf3"
    @test CAL.term_label(CAL.PolynomialTime(2)) == "polyt2"
    @test_throws ArgumentError CAL.PolynomialFreq(0)
    @test_throws "axis must be one of" CAL.Polynomial{:nope}(2)

    # A term is handed the axes it declares, under the dimension names, and a
    # channel index is never one of them.
    @test CAL.term_axes(CAL.Delay()) == (:Frequency,)
    @test CAL.term_axes(CAL.Rate()) == (:Ti,)
    @test CAL.term_axes(CAL.ConstantTerm()) == ()

    # A block's size never depends on how many channels its segment holds: how
    # finely a term varies in frequency is said by the segmentation.
    @test CAL.nparams_per_block(CAL.ConstantTerm(), 7) == 1
    @test CAL.nparams_per_block(CAL.Delay(), 7) == 1
    @test CAL.nparams_per_block(CAL.Rate(), 7) == 1
    @test CAL.nparams_per_block(CAL.PolynomialFreq(3), 7) == 3
    @test CAL.nparams_per_block(CAL.PolynomialTime(2), 7) == 2

    # scalar term_eval primitives: a term sees its own named parameters and the
    # coordinates `term_axes` declares, under those names.
    @test CAL.term_eval(CAL.ConstantTerm(), (offset = 1.5,), NamedTuple()) == 1.5
    @test CAL.term_eval(CAL.Delay(), (delay = 0.3,), (Frequency = 4.0,)) ≈ 2π * 0.3 * 4.0
    @test CAL.term_eval(CAL.Rate(), (rate = 0.3,), (Ti = 5.0,)) ≈ 2π * 0.3 * 5.0
    @test CAL.term_eval(CAL.Dispersion(), (dtec = 0.3,), (Frequency = 4.0,)) ≈ 0.3 * 4.0
    @test CAL.term_eval(
        CAL.PolynomialFreq(3), (coeffs = [0.3, 1.5, -0.7],), (Frequency = 2.0,)
    ) ≈ 0.3 * 2 + 1.5 * 4 + (-0.7) * 8
    @test CAL.term_eval(
        CAL.PolynomialTime(3), (coeffs = [0.3, 1.5, -0.7],), (Ti = 2.0,)
    ) ≈ 0.3 * 2 + 1.5 * 4 + (-0.7) * 8

    # The named parameters of a block are addressed over the block's own view.
    θ = [0.0, 0.3, 1.5, -0.7]
    @test CAL._block_params(CAL.param_shapes(CAL.Delay(), 1), view(θ, 2:2)) == (delay = 0.3,)
    @test CAL._block_params(CAL.param_shapes(CAL.PolynomialFreq(2), 1), view(θ, 3:4)).coeffs ==
        [1.5, -0.7]
    # Declaration order, and each name gets exactly the size it declared.
    shapes = (a = (), b = (2,), c = ())
    p = CAL._block_params(shapes, θ)
    @test p.a == 0.0 && p.b == [0.3, 1.5] && p.c == -0.7

    # `basis_columns` is gone: WLS solvers build their own systems.
    @test !isdefined(CAL, :basis_columns)
end

@testset "Calibration feed tying offset algebra" begin
    geom = CAL.DataGeometry(; times = [0.0, 1.0], channel_freqs = [1.0e9, 2.0e9])
    # One ConstantTerm, GlobalTime × GlobalFrequency, 2 antennas.
    mk(tying) = CAL.StationGainModel(
        phase = (c = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), tying),),
    )

    lay_pf = CAL.plan_parameters(mk(CAL.PerFeed()), 2, geom)
    @test lay_pf.nθ == 4                                    # 2 ant × 2 feeds
    p = lay_pf.plans[1]
    @test plan_off1(p)[1, 1, 1, 1] != plan_off1(p)[1, 2, 1, 1]          # feeds independent
    @test all(plan_off2(p) .== 0)

    lay_sf = CAL.plan_parameters(mk(CAL.SharedFeeds()), 2, geom)
    @test lay_sf.nθ == 2                                    # 2 ant, shared across feeds
    q = lay_sf.plans[1]
    @test plan_off1(q)[1, 1, 1, 1] == plan_off1(q)[1, 2, 1, 1]          # feeds share a block
    @test all(plan_off2(q) .== 0)

    lay_rr = CAL.plan_parameters(mk(CAL.ReferenceRelative(1)), 2, geom)
    @test lay_rr.nθ == 4                                    # ref + relative per ant
    r = lay_rr.plans[1]
    @test plan_off1(r)[1, 1, 1, 1] == plan_off1(r)[1, 2, 1, 1]          # both feeds reference the ref block
    @test plan_off2(r)[1, 1, 1, 1] == 0                           # reference feed has no relative
    @test plan_off2(r)[1, 2, 1, 1] != 0                           # partner feed adds a relative block
end

@testset "Calibration ComponentVector template" begin
    freqs = [1.0e9, 2.0e9, 3.0e9]                    # 3 channels
    geom = CAL.DataGeometry(; times = [0.0, 1.0], channel_freqs = freqs)
    nant = 2
    model = CAL.StationGainModel(
        phase = (
            a = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),
            bp = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.ChannelBlocks(1)), CAL.SharedFeeds()),
            grp = (                                  # one element compiling to a nested subtree
                d = CAL.TiedComponent(CAL.GainComponent(CAL.Delay(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.SharedFeeds()),
                c = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.SharedFeeds()),
            ),
        ),
    )
    layout = CAL.plan_parameters(model, nant, geom)
    θ = Float64.(1:layout.nθ)
    cv = CAL.component_vector(layout, θ)

    # The named/shaped view spans exactly the flat θ (one source of truth) and
    # nests where the model does.
    @test length(layout.template) == layout.nθ
    @test propertynames(cv) == (:phase, :logamp)
    @test propertynames(cv.phase) == (:a, :bp, :grp)
    @test propertynames(cv.phase.grp) == (:d, :c)

    # Each leaf is the full-rank shape its tying × segmentation imply
    # (param, feed-node, freq-seg, time-seg, ant), size-1 axes kept.
    @test size(cv.phase.a) == (1, 2, 1, 1, nant)                 # PerFeed: two feed nodes
    @test size(cv.phase.bp) == (1, 1, length(freqs), 1, nant)    # ChannelBlocks(1): one freq-seg per channel
    @test size(cv.phase.grp.d) == (1, 1, 1, 1, nant)             # a nested part is a plain leaf
    @test size(cv.phase.grp.c) == (1, 1, 1, 1, nant)

    # The stored axis roles name each dimension.
    @test layout.axes.phase.a.roles == (:param, :Feed, :Frequency, :Ti, :Ant)
    @test layout.axes.phase.bp.roles == (:param, :node, :Frequency, :Ti, :Ant)
    @test layout.axes.phase.grp.d.roles == (:param, :node, :Frequency, :Ti, :Ant)

    # A named leaf is exactly the component's θ block (matches component_ranges:
    # phase components depth-first — a, bp, grp.d, grp.c).
    rng = CAL.component_ranges(layout)
    @test vec(cv.phase.a) == θ[rng[1]]
    @test vec(cv.phase.bp) == θ[rng[2]]
    @test vec(cv.phase.grp.d) == θ[rng[3]]
    @test vec(cv.phase.grp.c) == θ[rng[4]]

    # The wrap shares data (no copy), and the forward map reads it identically to
    # the flat vector.
    ev = CAL.GainEvaluator(model, layout)
    @test CAL.evaluate_gains(ev, cv) == CAL.evaluate_gains(ev, θ)
    cv[1] = -99.0
    @test θ[1] == -99.0
end

@testset "@comp: a component leaf as a labelled DimArray" begin
    freqs = [1.0e9, 2.0e9, 3.0e9, 4.0e9]
    times = [0.0, 1.0, 2.0, 3.0]
    geom = CAL.DataGeometry(;
        times, channel_freqs = freqs,
        scan_of_time = [1, 1, 2, 2], spw_of_chan = [1, 1, 1, 1], t0 = 0.0, f0 = 2.5e9,
    )
    nant = 3
    ants = ["PT", "LM", "AA"]
    model = CAL.StationGainModel(
        phase = (
            atmos = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.PerScan(), CAL.GlobalFrequency()), CAL.PerFeed()),
            bp = CAL.TiedComponent(CAL.GainComponent(CAL.PolynomialFreq(2), CAL.GlobalTime(), CAL.ChannelBlocks(2)), CAL.SharedFeeds()),
            rl = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.ReferenceRelative(1)),
        ),
        logamp = (
            amp = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.PerScan(), CAL.GlobalFrequency()), CAL.SharedFeeds()),
        ),
    )
    layout = CAL.plan_parameters(model, nant, geom)
    θ = Float64.(1:layout.nθ)
    sol = CAL.CalibrationSolution(model, layout, geom, θ, (; ant_names = ants))

    # A leaf's DimArray is shaped and rolled exactly as the layout stored it.
    a = @comp sol.θ.phase.atmos
    @test a isa DimArray
    @test size(a) == layout.axes.phase.atmos.dims
    @test name.(dims(a)) == layout.axes.phase.atmos.roles
    @test name(a) == :atmos

    # PerFeed carries a physical Feed axis; the tied tyings carry a positional
    # node axis (ReferenceRelative: reference + relative, two nodes).
    @test name(dims(a, 2)) == :Feed
    @test name(dims((@comp sol.θ.phase.bp), 2)) == :node
    @test size((@comp sol.θ.phase.rl), 2) == 2

    # Segment axes carry a representative physical coordinate per segment: a
    # frequency segment's centre, a time segment's mean epoch.
    @test lookup(a, UVD.Frequency) == [mean(freqs)]              # GlobalFrequency: one centre
    @test lookup(a, Ti) == [mean(times[1:2]), mean(times[3:4])]  # PerScan: per-scan mean epoch
    @test lookup((@comp sol.θ.phase.bp), UVD.Frequency) ==
        [mean(freqs[1:2]), mean(freqs[3:4])]                     # ChannelBlocks(2): block centres

    # Antennas take the solution's station names; feed is 1:2.
    @test lookup(a, UVD.Ant) == ants
    @test lookup(a, UVD.Feed) == 1:2

    # logamp descends the same way.
    @test (@comp sol.θ.logamp.amp) isa DimArray

    # The leaf is a view onto θ (no copy): its data is the component's block, and
    # writing through it mirrors into θ.
    rng = CAL.component_ranges(layout)
    @test vec(parent(a)) == θ[rng[1]]
    a[1, 1, 1, 1, 1] = -7.0
    @test sol.steps[1].θ[first(rng[1])] == -7.0

    # No ant_names in info → the antenna axis falls back to 1:nant.
    soln = CAL.CalibrationSolution(model, layout, geom, θ, (; nant))
    @test lookup((@comp soln.θ.phase.atmos), UVD.Ant) == 1:nant

    # A multi-emit wrapper's nested subtree is reached leaf by leaf, the shape
    # SingleBandDelay compiles to (`sbd.delay` / `sbd.constant`).
    gb = CAL.DataGeometry(;
        times = [0.0, 1.0], channel_freqs = [1.0e9, 1.1e9, 5.0e9, 5.1e9],
        scan_of_time = [1, 1], spw_of_chan = [1, 1, 2, 2], t0 = 0.0, f0 = 3.0e9,
    )
    sbd = CAL.model_components(SingleBandDelay(), gb)
    msbd = CAL.StationGainModel(phase = (sbd = sbd,))
    lsbd = CAL.plan_parameters(msbd, 2, gb)
    ssbd = CAL.CalibrationSolution(msbd, lsbd, gb, Float64.(1:lsbd.nθ), (;))
    dl = @comp ssbd.θ.phase.sbd.delay
    @test dl isa DimArray
    @test name(dl) == :delay
    @test lookup(dl, UVD.Frequency) == [mean([1.0e9, 1.1e9]), mean([5.0e9, 5.1e9])]
    @test vec(parent(dl)) == ssbd.steps[1].θ[lsbd.plantree.phase.sbd.delay.range]

    # A path that stops at a group, or omits the `.θ` marker, is rejected.
    @test_throws ArgumentError (@comp ssbd.θ.phase.sbd)
    @test_throws "names a component group" (@comp ssbd.θ.phase.sbd)
    @test_throws LoadError @eval @comp sol.phase.atmos
end

@testset "Calibration evaluate_gains: correctness, purity, inference" begin
    nant = 3
    freqs = collect(2.28e11:1.0e8:(2.28e11 + 5.0e8))         # 6 channels
    f0 = sum(freqs) / length(freqs)
    times = [0.0, 0.5, 1.0]
    geom = CAL.DataGeometry(; times, channel_freqs = freqs, t0 = 0.0, f0)

    # Pure per-feed delay model: phase = 2π τ (f − f0), one τ per (ant, feed).
    model = CAL.StationGainModel(
        phase = (delay = CAL.TiedComponent(CAL.GainComponent(CAL.Delay(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
    )
    ev = CAL.GainEvaluator(model, geom; nant)
    @test CAL.nparameters(ev) == nant * 2

    τ = 1.0e-9 .* collect(1:(nant * 2))                    # distinct delays in ns
    g = CAL.evaluate_gains(ev, τ)
    @test size(g) == (length(freqs), length(times), nant, 2)

    # Hand-check a couple of cells: |g| == 1 (no amplitude), phase = 2π τ (f−f0).
    p = ev.layout.plans[1]
    for ant in 1:nant, feed in 1:2, ci in eachindex(freqs)
        off = plan_off1(p)[ant, feed, 1, 1]
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
        phase = (rate = CAL.TiedComponent(CAL.GainComponent(CAL.Rate(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.SharedFeeds()),),
    )
    evr = CAL.GainEvaluator(rate_model, geom; nant)
    ṙ = 1.0e-3 .* collect(1:nant)                          # mHz-scale rates
    gr = CAL.evaluate_gains(evr, ṙ)
    pr = evr.layout.plans[1]
    for ant in 1:nant, ti in eachindex(times)
        off = plan_off1(pr)[ant, 1, 1, 1]
        expected = cis(2π * ṙ[off] * (times[ti] - 0.0) * 3600.0)
        @test gr[1, ti, ant, 1] ≈ expected
        @test gr[1, ti, ant, 2] ≈ expected               # shared across feeds
    end
end

@testset "Calibration predict_visibilities closes" begin
    nant = 3
    geom = CAL.DataGeometry(; times = [0.0], channel_freqs = [2.28e11, 2.281e11], t0 = 0.0)
    model = CAL.StationGainModel(
        phase = (offset = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
        logamp = (offset = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
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

@testset "Calibration: misdeclared coordinate term errors loudly (N3)" begin
    # A new term that declares the :Frequency axis but defines no
    # freq_coordinate must error at plan time, not silently evaluate at x = 0.
    @eval CAL begin
        struct _AuditBadFreqTerm <: AbstractGainTerm end
        term_axes(::_AuditBadFreqTerm) = (:Frequency,)
        param_shapes(::_AuditBadFreqTerm, n) = (scale = (),)
        # NOTE: deliberately no freq_coordinate method.
    end
    geom = CAL.DataGeometry(; times = [0.0], channel_freqs = [1.0e9, 2.0e9])
    model = CAL.StationGainModel(
        phase = (bad = CAL.TiedComponent(CAL.GainComponent(CAL._AuditBadFreqTerm(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),),
    )
    @test_throws MethodError CAL.plan_parameters(model, 1, geom)
end

@testset "CalibrationSolution θ keeps its array type" begin
    nant = 3
    freqs = collect(2.28e11:1.0e8:(2.28e11 + 5.0e8))
    times = [0.0, 0.5, 1.0]
    geom = CAL.DataGeometry(;
        times, channel_freqs = freqs, t0 = 0.0, f0 = sum(freqs) / length(freqs),
    )
    # A delay term plus a per-channel bandpass, so `step_solution` has
    # something to extract.
    model = CAL.StationGainModel(
        phase = (
            delay = CAL.TiedComponent(CAL.GainComponent(CAL.Delay(), CAL.GlobalTime(), CAL.GlobalFrequency()), CAL.PerFeed()),
            bandpass = CAL.TiedComponent(CAL.GainComponent(CAL.ConstantTerm(), CAL.GlobalTime(), CAL.ChannelBlocks(1)), CAL.SharedFeeds()),
        ),
    )
    layout = CAL.plan_parameters(model, nant, geom)
    θv = collect(1:(layout.nθ)) ./ 1.0e10

    solv = CAL.CalibrationSolution(model, layout, geom, θv, (; nant))
    @test solv.steps[1].θ isa Vector{Float64}

    @testset "a DimArray θ survives construction and every derived path" begin
        # Split into the delay-only :fringe step and the bandpass-only
        # :bandpass step so the step-scoped accessors (`stage_solution`,
        # `step_solution`, `component_gains`) have real steps to address.
        model_fr = CAL.StationGainModel(phase = (delay = model.phase.delay,))
        model_bp = CAL.StationGainModel(phase = (bandpass = model.phase.bandpass,))
        layout_fr = CAL.plan_parameters(model_fr, nant, geom)
        layout_bp = CAL.plan_parameters(model_bp, nant, geom)
        θv_fr = θv[1:(layout_fr.nθ)]
        θv_bp = θv[(layout_fr.nθ + 1):end]
        two_step(θfr, θbp) = CAL.CalibrationSolution(
            [CAL.StepSolution(:fringe, model_fr, layout_fr, θfr),
                CAL.StepSolution(:bandpass, model_bp, layout_bp, θbp)],
            geom, (; nant),
        )
        solv2 = two_step(θv_fr, θv_bp)

        θd_fr = DimArray(copy(θv_fr), Dim{:param}(1:(layout_fr.nθ)))
        θd_bp = DimArray(copy(θv_bp), Dim{:param}(1:(layout_bp.nθ)))
        sold2 = two_step(θd_fr, θd_bp)
        @test sold2.steps[1].θ isa DimArray
        @test sold2.steps[1].θ == θd_fr
        # Same numbers as the Vector-backed solution, not merely close.
        ev = CAL.GainEvaluator(model_fr, layout_fr)
        @test CAL.evaluate_gains(ev, sold2.steps[1].θ) == CAL.evaluate_gains(ev, solv2.steps[1].θ)
        @test CAL.component_gains(sold2, :fringe, 1) == CAL.component_gains(solv2, :fringe, 1)
        # A stage snapshot is index-matched to θ, so it propagates the array type.
        @test CAL.stage_solution(sold2, :fringe).steps[1].θ isa DimArray
        @test CAL.stage_solution(sold2, :fringe).steps[1].θ == CAL.stage_solution(solv2, :fringe).steps[1].θ
        # `step_solution` returns the named step's own solution — its θ shares
        # no parameter identity with the merged original, only the element type.
        @test CAL.step_solution(sold2, :bandpass).steps[1].θ == CAL.step_solution(solv2, :bandpass).steps[1].θ
    end

    @testset "gains(sol) labels the forward map for inspection" begin
        ev = CAL.GainEvaluator(model, layout)
        g = gains(solv)
        @test g isa DimArray
        @test size(g) == (length(freqs), length(times), nant, 2)
        # The same numbers `evaluate_gains` / `apply_calibration` use.
        @test parent(g) == CAL.evaluate_gains(ev, solv.steps[1].θ)
        # Axes carry the geometry, so a user can index by physical coordinate.
        @test lookup(g, UVD.Frequency) == freqs
        @test lookup(g, Ti) == times
        @test lookup(g, UVD.Ant) == 1:nant           # no ant_names in info → 1:nant
        @test lookup(g, UVD.Feed) == 1:2
        # amp/phase recover from the complex gain, no separate accessor needed.
        @test abs.(g) == abs.(CAL.evaluate_gains(ev, solv.steps[1].θ))
        @test angle.(g) == angle.(CAL.evaluate_gains(ev, solv.steps[1].θ))
        # Indifferent to θ's array type.
        θd = DimArray(copy(θv), Dim{:param}(1:(layout.nθ)))
        @test gains(CAL.CalibrationSolution(model, layout, geom, θd, (; nant))) == g
    end

    @testset "the 1-based contract is enforced, not assumed" begin
        # A component's `range` holds absolute positions and the term kernels read
        # its block under `@inbounds`, so a shifted-axes θ must be refused here
        # rather than read out of bounds later.
        θoff = OffsetArrays.OffsetArray(copy(θv), 0:(layout.nθ - 1))
        @test_throws ArgumentError CAL.CalibrationSolution(model, layout, geom, θoff, (;))
        @test_throws DimensionMismatch CAL.CalibrationSolution(
            model, layout, geom, θv[1:(end - 1)], (;))
        @test_throws "θ has length" CAL.CalibrationSolution(model, layout, geom, θv[1:(end - 1)], (;))
    end

    @testset "the element type is carried, not coerced to Float64" begin
        @test CAL.CalibrationSolution(model, layout, geom, Float32.(θv), (;)).steps[1].θ isa Vector{Float32}
    end

    @testset "the solution copies θ rather than aliasing it" begin
        # The fused output tail builds a solution per scan group from the run's
        # live θ while sibling groups are still writing their own slots.
        θmut = copy(θv)
        s = CAL.CalibrationSolution(model, layout, geom, θmut, (;))
        θmut[1] = -999.0
        @test s.steps[1].θ[1] == θv[1]
    end
end

@testset "argument validation is typed" begin
    # Constructor and argument validation throws `ArgumentError` or
    # `DimensionMismatch`; bare `error` is reserved for algorithmic failure, so a
    # caller can tell "you passed me nonsense" from "the solve did not converge".
    @testset "segmentation constructors" begin
        @test_throws ArgumentError CAL.TimeBlocks(0.0)
        @test_throws "duration_hr must be positive" CAL.TimeBlocks(-1.0)
        @test_throws ArgumentError CAL.ChannelBlocks(0)
        @test_throws "block_size must be at least 1" CAL.ChannelBlocks(-2)
        @test_throws ArgumentError CAL.FrequencyBands(UnitRange{Int}[])
        @test_throws "at least one range" CAL.FrequencyBands(UnitRange{Int}[])
        @test_throws "must start at channel 1" CAL.FrequencyBands([2:4])
        @test_throws "contiguous and ascending" CAL.FrequencyBands([1:4, 6:8])
    end

    @testset "geometry axis lengths" begin
        @test_throws DimensionMismatch CAL.DataGeometry(;
            times = [0.0, 1.0], channel_freqs = [1.0e9], scan_of_time = [1])
        @test_throws "scan_of_time length" CAL.DataGeometry(;
            times = [0.0, 1.0], channel_freqs = [1.0e9], scan_of_time = [1])
        @test_throws DimensionMismatch CAL.DataGeometry(;
            times = [0.0], channel_freqs = [1.0e9, 2.0e9], spw_of_chan = [1])
        @test_throws "spw_of_chan length" CAL.DataGeometry(;
            times = [0.0], channel_freqs = [1.0e9, 2.0e9], spw_of_chan = [1])

        # `FrequencyBands` is structurally valid but must also cover the geometry.
        geom = CAL.DataGeometry(; times = [0.0], channel_freqs = collect(1.0:6.0) .* 1.0e9)
        @test_throws DimensionMismatch CAL.freq_segment_ids(CAL.FrequencyBands([1:4]), geom)
        @test_throws "geometry has 6" CAL.freq_segment_ids(CAL.FrequencyBands([1:4]), geom)
    end

    @testset "feed tying and empty models" begin
        @test_throws ArgumentError CAL.ReferenceRelative(3)
        @test_throws "reference_feed must be 1 or 2" CAL.ReferenceRelative(0)
        @test_throws ArgumentError CAL.FeedComponent(3)
        @test_throws "feed must be 1 or 2" CAL.FeedComponent(0)
        @test_throws ArgumentError CAL.validate_station_gain_model(CAL.StationGainModel())
        @test_throws "neither phase nor log-amplitude" CAL.validate_station_gain_model(
            CAL.StationGainModel())
    end
end
