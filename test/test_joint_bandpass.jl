# ── Bandpass(smoother = JointSmoother()): solve_joint_bandpass! ──────────────
#
# The joint solve fits an explicit per-scan source coherence, so the synthetic
# visibilities here are post-multiplied by an independent random complex factor per
# (scan, baseline, polarization) — a stand-in for a resolved/polarized
# calibrator's per-baseline structure — and the tests check that
# `solve_joint_bandpass!` still recovers the injected station bandpass.
#
# A single station's ABSOLUTE per-channel bandpass is never observable from
# baseline data alone (only differences between stations are), so the recovered
# bandpass is relative to the gauge's reference — truth is gauged the same way
# before comparing.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# A bandpass solution's `obs` (`:phase` or `:logamp`) bandpass parameters over
# `(param, Feed, Frequency, Ti, AntennaName)`.
_jb_leaf(sol, obs) = parent(only(sol[obs, :bandpass].components).params)

# Each observable's fitted components of layout `l`, their parameters in `θ`,
# in the form `solve_joint_bandpass!` takes.
function _jb_components(l, θ)
    blocks(group) = [CAL.antenna_blocks(l, θ, path...) for path in FP._bandpass_paths(l.plantree, group).fitted]
    return (; phase = blocks(:phase), logamp = blocks(:logamp))
end

# Priors far wider than any value here, so that their pull on the noise-free
# fits below is under the tolerances: one per value, and one along frequency,
# whose chains the solve reports.
const _JB_WIDE = CAL.IIDPrior(1.0e5)
const _JB_WIDE_CHAIN = (; Frequency = CAL.OUPrior(; scale = 1.0e6, σ = 1.0e5))

# Multiply each (scan, baseline, product) of `ps` by `amp·cis(phase)`, indexed
# `[scan, baseline, product]` in each Measurement Set's own baseline and product
# order: per-scan source structure the station gains cannot absorb.
function _scale_sources!(ps, amp, phase)
    for ms in values(ps)
        s = parse(Int, only(unique(ms[:scan_name])))
        V = DimensionalData.modify(Array, ms[:visibility])
        for p in axes(amp, 3), bi in axes(amp, 2)
            view(V, BaselineID(bi), Polarization(p)) .*= ComplexF32(amp[s, bi, p] * cis(phase[s, bi, p]))
        end
        ms[:visibility] = V
    end
    return ps
end

@testset "Bandpass(smoother = JointSmoother()): recovers the bandpass under per-baseline/pol source structure" begin
    nant, nspw, nchan, ntime, nscans = 4, 1, 8, 6, 6
    rng = MersenneTwister(4242)
    nglob = nspw * nchan
    bp_true = 0.4 .* randn(rng, nant, 2, nglob)
    abp_true = 0.15 .* randn(rng, nant, 2, nglob)
    ps, truth = _build_fringe_ps(;
        nant, nspw, nchan, ntime, nscans, bandpass = bp_true, amp_bandpass = abp_true,
        seed = 99,
    )

    nbl = length(truth.bl_pairs)
    npol = length(truth.polarizations)
    src_amp = 0.4 .+ 1.6 .* rand(rng, nscans, nbl, npol)
    src_phase = (2 .* rand(rng, nscans, nbl, npol) .- 1) .* pi
    _scale_sources!(ps, src_amp, src_phase)

    fm = default_fringe_model()
    # `ref_ant` indexes the truth arrays below; `gauge` is what the solve takes.
    ref_ant = 1
    gauge = PinAntenna(ref_ant)     # the gauge every test here passes to `fit`
    _, sol = _fit_chain((FringeClosure(; model = fm, gauge), Bandpass(; model = _bandpass_model(), gauge)), ps; exec = ExecutionConfig())
    pleaf, aleaf = _jb_leaf(sol, :phase), _jb_leaf(sol, :logamp)

    # The fringe fit's delays take up each station's phase slope across the band,
    # so the phase is compared after removing a straight line in channel.
    function wrapped_err(x, y)
        d = rem2pi.(x .- y, RoundNearest)
        A = [ones(length(d)) eachindex(d)]
        return maximum(abs, d .- A * (A \ d))
    end
    # PHASE is recovered relative to the reference's feed 1, whose phase the gauge
    # pins at every channel, so truth is gauged the same way before comparing,
    # then to its own circular mean.
    gauge_phase_rel(x, xref) = rem2pi.((x .- xref) .- angle(sum(cis, x .- xref)), RoundNearest)
    # AMPLITUDE is NOT referenced to any antenna: the gauge pins only phase, so
    # each station's own passband shape is identifiable and only its band mean
    # is unobservable; both sides are compared with it removed.
    gauge_amp(x) = x .- sum(x) / length(x)

    max_phase_err = 0.0
    max_amp_err = 0.0
    for a in 1:nant, f in 1:2
        bj = Float64[pleaf[1, f, c, 1, a] for c in 1:nglob]
        aj = Float64[aleaf[1, f, c, 1, a] for c in 1:nglob]
        max_phase_err = max(max_phase_err, wrapped_err(bj, gauge_phase_rel(bp_true[a, f, :], bp_true[ref_ant, 1, :])))
        max_amp_err = max(max_amp_err, maximum(abs, gauge_amp(aj) .- gauge_amp(abp_true[a, f, :])))
    end
    # log|V| = la_a + la_b has no baseline term to absorb the injected
    # per-baseline/pol factor; the explicit source coherence absorbs it.
    @test max_phase_err < 5.0e-3
    @test max_amp_err < 1.0e-3
end

@testset "JointSmoother: the component priors act INSIDE the ALS" begin
    # The gain update fits each (station, feed) track under its component's prior
    # instead of solving every channel independently. A weak IID prior on either
    # observable is nearly that independent per-channel solve; a stiff
    # second-order walk must instead leave only its null space — a straight line
    # in frequency.
    nant, nspw, nchan, ntime, nscans = 4, 1, 8, 4, 3
    nglob = nspw * nchan
    rng = MersenneTwister(0x101A)
    bp_true = 0.4 .* randn(rng, nant, 2, nglob)
    abp_true = 0.15 .* randn(rng, nant, 2, nglob)
    ps, _ = _build_fringe_ps(;
        nant, nspw, nchan, ntime, nscans,
        bandpass = bp_true, amp_bandpass = abp_true, seed = 21,
    )
    fm = default_fringe_model()
    bpc(prior) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior)
    runbp(sm; prior = IIDPrior(10.0)) = last(
        _fit_chain(
            (
                FringeClosure(; model = fm, gauge = PinAntenna(1)),
                Bandpass(; model = CAL.GainModel(; phase = (; bandpass = bpc(prior)), logamp = (; bandpass = bpc(prior))), smoother = sm, gauge = PinAntenna(1)),
            ),
            ps; exec = ExecutionConfig(),
        )
    )

    s_free = runbp(FP.JointSmoother())
    # The data reduced in another type: the same gains to that type's precision.
    s32, s64 = runbp(FP.JointSmoother(; eltype = Float32)), runbp(FP.JointSmoother(; eltype = Float64))
    for leaf in (:phase, :logamp)
        @test _jb_leaf(s32, leaf) ≈ _jb_leaf(s64, leaf) rtol = 1.0e-4
    end
    @test_logs (:warn, r"did not converge in 2 iterations") match_mode = :any runbp(FP.JointSmoother(max_iterations = 2))

    # σ = 1e-6 rad per channel (2 MHz) as rad/Hz^(3/2), from a weak start: 10 rad,
    # and 10 rad over the 16 MHz band.
    stiff = CAL.RandomWalkPrior(;
        order = 2, σ = 1.0e-6 * sqrt(3 / (2 * 2.0e6^3)), init = MvNormal(zeros(2), Diagonal([10.0^2, (10.0 / 16.0e6)^2])),
    )
    # A prior the gains cannot follow exactly must still converge.
    logs, s_stiff = Test.collect_test_logs(() -> runbp(FP.JointSmoother(); prior = stiff))
    @test !any(l -> occursin("did not converge", string(l.message)), logs)
    pleaf, aleaf = _jb_leaf(s_stiff, :phase), _jb_leaf(s_stiff, :logamp)
    nflat = 0
    for a in 1:nant, f in 1:2
        ph = Float64[pleaf[1, f, c, 1, a] for c in 1:nglob]
        la = Float64[aleaf[1, f, c, 1, a] for c in 1:nglob]
        (all(isfinite, ph) && all(isfinite, la)) || continue
        nflat += 1
        @test maximum(abs, diff(diff(CAL.unwrap_phase_track(ph)))) < 1.0e-4
        @test maximum(abs, diff(diff(la))) < 1.0e-4
    end
    @test nflat == 2 * nant
    # ...and the unconstrained solve is genuinely rougher, so the flatness above
    # is the prior acting rather than a featureless track.
    apleaf = _jb_leaf(s_free, :logamp)
    rough = maximum(
        maximum(abs, diff(diff(Float64[apleaf[1, f, c, 1, a] for c in 1:nglob])))
            for a in 1:nant, f in 1:2
    )
    @test rough > 0.1
end

@testset "JointSmoother: reference-antenna amplitude is gauged, not pinned" begin
    # Only the per-channel PHASE is a gauge freedom: the phase of a factor common
    # to every station at one channel cancels between `g_a` and `conj(g_b)`, while
    # its MAGNITUDE does not and the frequency-flat `S` cannot absorb it. So the
    # reference node's phase is pinned at every channel and its amplitude is
    # solved like any other station's. Pinning the amplitude too would discard the
    # reference antenna's own passband — identifiable structure — and bias every
    # other station through the resulting inconsistency.
    nant, nspw, nchan, ntime, nscans = 4, 1, 8, 4, 3
    nglob = nspw * nchan
    rng = MersenneTwister(0x9A17)
    bp_true = 0.4 .* randn(rng, nant, 2, nglob)
    abp_true = 0.25 .* randn(rng, nant, 2, nglob)
    ps, truth = _build_fringe_ps(;
        nant, nspw, nchan, ntime, nscans,
        bandpass = bp_true, amp_bandpass = abp_true, seed = 33,
    )
    # Per-(scan, baseline, pol) structure, which the source coherence absorbs.
    nbl = length(truth.bl_pairs); npol = length(truth.polarizations)
    samp = 0.4 .+ 1.6 .* rand(rng, nscans, nbl, npol)
    sph = (2 .* rand(rng, nscans, nbl, npol) .- 1) .* pi
    _scale_sources!(ps, samp, sph)

    fm = default_fringe_model()
    runbp(sm) = last(
        _fit_chain(
            (FringeClosure(; model = fm, gauge = PinAntenna(1)), Bandpass(; model = _bandpass_model(), smoother = sm, gauge = PinAntenna(1))),
            ps; exec = ExecutionConfig(),
        )
    )
    s_joint = runbp(FP.JointSmoother())

    la(s, a, f) = (
        L = _jb_leaf(s, :logamp);
        Float64[L[1, f, c, 1, a] for c in 1:nglob]
    )
    gauge(x) = x .- sum(x) / length(x)
    # Under the `PinAntenna(1)` gauge the fits above use, feed 1 is the pinned
    # node. Its amplitude bandpass must come back, not flat.
    ref_true = gauge(abp_true[1, 1, :])
    ref_got = gauge(la(s_joint, 1, 1))
    @test std(ref_got) > 0.5 * std(ref_true)
    @test maximum(abs, ref_got .- ref_true) < 0.1

    # ...and with the reference free to carry its own shape, every station's is
    # recovered.
    amprms(s) = sqrt(
        sum(sum(abs2, gauge(la(s, a, f)) .- gauge(abp_true[a, f, :])) for a in 1:nant, f in 1:2) /
            (2 * nant * nglob),
    )
    @test amprms(s_joint) < 1.0e-3
end

# ── The per-station time-segment axis inside one solve ───────────────────────
#
# The components carry a time-segment axis and the scans that couple through a
# shared (station, segment) parameter are solved together, so a station whose
# bandpass breaks mid-track can sit beside stations held over the whole of it.
# These fixtures drive `solve_joint_bandpass!` directly on synthetic
# accumulators — `rl = w·g_a·S·conj(g_b)`, exactly the model the solve inverts —
# because the segment table is what is under test, not the accumulation.

# A four-scan, one-time-sample-per-scan geometry of `nant` stations `A1, A2, …`
# whose second half is a separate instrument segment.
function _seg_geometry(nchan; nant = 4)
    return CAL.DataGeometry(;
        nfeed = 2,
        times = collect(0.0:3.0), scan_of_time = collect(1:4),
        channel_freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6),
        spw_of_chan = ones(Int, nchan),
        scan_names = ["No00$i" for i in 1:4], spw_names = ["A"],
        antennas = ["A$i" for i in 1:nant],
    )
end

# Per-scan accumulators for `g[ant, feed, segment, channel]` observed through
# `S[scan, baseline, pol]`, with `tseg[ant, scan]` naming each station's segment,
# labeled by `geom`'s station names and channel frequencies.
function _joint_scan_accumulators(g, S, tseg, bl_pairs, feeds, geom)
    names = geom.antennas
    return map(axes(S, 1)) do si
        ax = (
            FP._antenna_pair_dim([(names[a], names[b]) for (a, b) in bl_pairs]),
            FP.FeedPair(feeds), Frequency(geom.channel_freqs),
        )
        rl, wl = zeros(ComplexF64, ax), zeros(ax)
        for (bi, (a, b)) in pairs(bl_pairs), p in eachindex(feeds)
            fa, fb = feeds[p]
            for c in axes(rl, Frequency)
                v = g[a, fa, tseg[a, si], c] * S[si, bi, p] *
                    conj(g[b, fb, tseg[b, si], c])
                wl[bi, p, c] = 1.0
                rl[bi, p, c] = v
            end
        end
        return (; rl, wl, ti = si)
    end
end

# A phase track less its circular mean, and a track less its mean: the
# constants over the scans that the source term absorbs.
_jb_cdemean(v) = rem2pi.(v .- angle(sum(cis, v)), RoundNearest)
_jb_demean(v) = v .- sum(v) / length(v)

@testset "JointSmoother: a window without data solves nothing" begin
    nant, nchan = 4, 8
    geom = CAL.DataGeometry(;
        nfeed = 2,
        times = collect(0.0:3.0), scan_of_time = collect(1:4),
        channel_freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6),
        spw_of_chan = repeat([1, 2]; inner = nchan ÷ 2),
        scan_names = ["No00$i" for i in 1:4], spw_names = ["A", "B"],
        antennas = ["A$i" for i in 1:nant],
    )
    comp = GainComponent(
        ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(),
        prior = CAL.OUPrior(; scale = 4.0e6, σ = 1.0),
    )
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; bandpass = comp), logamp = (; bandpass = comp)), geom.antennas, geom)
    rng = MersenneTwister(61)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    g = exp.(complex.(0.1 .* randn(rng, nant, 2, 1, nchan), 0.5 .* randn(rng, nant, 2, 1, nchan)))
    S = [cis(2π * rand(rng)) for _ in 1:4, _ in bl_pairs, _ in 1:2]
    results = _joint_scan_accumulators(g, S, fill(1, nant, 4), bl_pairs, [(1, 1), (2, 2)], geom)
    # Window B is flagged throughout, so its station gains have no data.
    for r in results
        r.wl[:, :, 5:8] .= 0
        r.rl[:, :, 5:8] .= 0
    end
    θ = zeros(l.nθ)
    comps = _jb_components(l, θ)
    pb = only(comps.phase)
    status = [FP._block_status_array(b, geom) for b in pb]
    fits = FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1), max_iterations = 100, tolerance = 1.0e-10)
    FP._record_track_fits!(status, nothing, fits.phase, fits.status.phase, (:phase, :bandpass), geom, pb)
    for f in 1:2
        @test only(status)[2, f, 1, 1] == FP.TrackSolved
        @test only(status)[2, f, 2, 1] == FP.TrackNoData
        # The reference's phase is zero where it has data; without data the
        # parameters sit at their prior's mean.
        @test maximum(abs, only(pb).θ[1, f, 1:4, 1, 1]) < 1.0e-12
        @test maximum(abs, only(pb).θ[1, f, 5:8, 1, :]) < 1.0e-12
    end
end

@testset "JointSmoother: cross-hands without weight leave each feed its own reference" begin
    nant, nchan = 4, 6
    geom = _seg_geometry(nchan)
    comp = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = _JB_WIDE)
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; bandpass = comp), logamp = (; bandpass = comp)), geom.antennas, geom)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    rng = MersenneTwister(19)
    g = exp.(complex.(0.1 .* randn(rng, nant, 2, 1, nchan), 0.5 .* randn(rng, nant, 2, 1, nchan)))
    S = [(0.5 + rand(rng)) * cis(2π * rand(rng)) for _ in 1:4, _ in bl_pairs, _ in 1:4]
    results = _joint_scan_accumulators(g, S, fill(1, nant, 4), bl_pairs, [(1, 1), (1, 2), (2, 1), (2, 2)], geom)
    for r in results
        r.wl[:, 2:3, :] .= 0
        r.rl[:, 2:3, :] .= 0
    end
    θ = zeros(l.nθ)
    comps = _jb_components(l, θ)
    FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1), tolerance = 1.0e-10)
    φ = only(only(comps.phase)).θ
    # Nothing relates the feeds, so each has its own free phase per channel,
    # and the reference's is zero on both.
    @test maximum(abs, φ[1, :, :, 1, 1]) < 1.0e-12
    for a in 2:nant, f in 1:2
        @test _jb_cdemean(φ[1, f, :, 1, a]) ≈ _jb_cdemean(angle.(g[a, f, 1, :]) .- angle.(g[1, f, 1, :])) atol = 1.0e-8
    end
end

@testset "JointSmoother: each antenna's own time segments in one solve" begin
    nant, nchan = 4, 6
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan)
    bp(ti) = GainComponent(ConstantTerm(); Ti = ti, Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = _JB_WIDE_CHAIN)
    breakmodel(ti) = CAL.GainModel(;
        phase = (; bandpass = bp(ti)), logamp = (; bandpass = bp(ti)),
    )
    layout(ti) = CAL.plan_parameters(breakmodel(ti), anames, geom)

    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    pol_products = [(1, 1), (2, 2)]
    feeds = collect(pol_products)

    rng = MersenneTwister(20260908)
    # Station 1 breaks across the boundary; every other station holds one gain
    # over the whole track.
    gtrue = ones(ComplexF64, nant, 2, 2, nchan)
    for a in 1:nant, f in 1:2, c in 1:nchan
        gt = exp(complex(0.2 * randn(rng), 0.6 * randn(rng)))
        gtrue[a, f, 1, c] = gt
        gtrue[a, f, 2, c] = a == 1 ? exp(complex(0.2 * randn(rng), 0.6 * randn(rng))) : gt
    end
    Strue = [
        (0.5 + rand(rng)) * cis(2pi * rand(rng))
            for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(pol_products)
    ]

    @testset "a uniform table reproduces the per-segment partition" begin
        l = layout(InstrumentScans([1.5]))
        results = _joint_scan_accumulators(
            gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom,
        )
        tseg = FP._component_time_segments(_jb_components(l, zeros(l.nθ)).phase, results, nant)
        # One block spans every station, so every row is that block's own table.
        @test all(tseg[a, :] == [1, 1, 2, 2] for a in 1:nant)
        @test FP._joint_scan_groups(tseg) == [[1, 2], [3, 4]]
    end

    @testset "a heterogeneous model writes through each antenna's own block" begin
        # Station 1 alone carries the break; the rest hold one gain over the
        # track, so the two signature groups become two blocks with different
        # `:AntennaName` axes and different time segmentations, and the solve has to
        # place each station in the block that holds it.
        het_model = CAL.GainModel(;
            phase = (; bandpass = bp(CAL.GlobalTime())),
            logamp = (; bandpass = bp(CAL.GlobalTime())),
            antennas = (
                A1 = (;
                    phase = (; bandpass = bp(InstrumentScans([1.5]))),
                    logamp = (; bandpass = bp(InstrumentScans([1.5]))),
                ),
            ),
        )
        lh = CAL.plan_parameters(het_model, anames, geom)
        θ = zeros(lh.nθ)
        comps = _jb_components(lh, θ)
        phase_blocks, amp_blocks = only(comps.phase), only(comps.logamp)
        @test [b.antennas for b in phase_blocks] == [[1], [2, 3, 4]]
        # The parameter saving is real: the stations that hold one gain carry one
        # time segment, not the union's two.
        @test phase_blocks[1].plan.shape[4] == 2
        @test phase_blocks[2].plan.shape[4] == 1

        tsg = fill(1, nant, 4)
        tsg[1, :] = [1, 1, 2, 2]
        results = _joint_scan_accumulators(gtrue, Strue, tsg, bl_pairs, feeds, geom)
        @test FP._component_time_segments(comps.phase, results, nant) == tsg
        @test FP._joint_scan_groups(tsg) == [[1, 2, 3, 4]]

        # Each block's status array spans its own stations and time segments,
        # labeled as its θ leaf is.
        phase_status = [FP._block_status_array(b, geom) for b in phase_blocks]
        amp_status = [FP._block_status_array(b, geom) for b in amp_blocks]
        for (st, b) in zip(phase_status, phase_blocks)
            @test lookup(st, FP.AntennaName) == geom.antennas[b.antennas]
            @test size(st, Ti) == b.plan.shape[4]
            ref = CAL._time_segment_lookup(b.plan, geom)
            @test collect(lookup(st, Ti)) == collect(ref)
            @test val(DimensionalData.Lookups.span(lookup(st, Ti))) == val(DimensionalData.Lookups.span(ref))
            @test all(==(FP.TrackNoData), st)
        end

        fits = FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(2), max_iterations = 200, tolerance = 1.0e-13)
        FP._record_track_fits!(phase_status, nothing, fits.phase, fits.status.phase, (:phase, :bandpass), geom, phase_blocks)
        FP._record_track_fits!(amp_status, nothing, fits.amp, fits.status.logamp, (:logamp, :bandpass), geom, amp_blocks)

        # Each (station, feed, time segment) track is compared up to its constant.
        want_phase(a, f, ts) = _jb_cdemean(
            angle.(gtrue[a, f, ts, :]) .- angle.(gtrue[2, f, 1, :]),
        )
        want_amp(a, f, ts) = _jb_demean(log.(abs.(gtrue[a, f, ts, :])))

        for f in 1:2
            # The broken station is alone on its block's `:AntennaName` axis, and both of
            # its segments land in that block.
            for ts in 1:2
                @test _jb_cdemean(phase_blocks[1].θ[1, f, :, ts, 1]) ≈ want_phase(1, f, ts) atol = 1.0e-8
                @test _jb_demean(amp_blocks[1].θ[1, f, :, ts, 1]) ≈ want_amp(1, f, ts) atol = 1.0e-8
            end
            # …and each of the others lands at ITS block-local position, which is
            # not its global station index.
            for (ai, a) in pairs(phase_blocks[2].antennas)
                @test _jb_cdemean(phase_blocks[2].θ[1, f, :, 1, ai]) ≈ want_phase(a, f, 1) atol = 1.0e-8
                @test _jb_demean(amp_blocks[2].θ[1, f, :, 1, ai]) ≈ want_amp(a, f, 1) atol = 1.0e-8
            end
            @test phase_status[1][FP.AntennaName(At("A1")), Feed(f), Ti(2)] == [FP.TrackSolved]
        end
        # The report keys the blocks as `parameters` keys their leaves, and counts
        # only the tracks the blocks have: station A1's two segments and one
        # segment of each of the others.
        rep = FP._bandpass_track_report(FP._block_leaves(phase_status), FP._block_leaves(amp_status))
        @test keys(rep.phase_status) == keys(rep.amp_status) == (:g1, :g2)
        @test rep.phase_status.g2 === phase_status[2]
        @test rep.n_nodata + rep.n_solved + rep.n_flat == 2 * 2 * (2 + (nant - 1))
    end
end

# ── The phase gauge across per-station time segments ─────────────────────────
#
# The data leave one phase free per connected group of station gains, and the
# stations that hold one gain over the whole track join a broken station's
# segments into one group, so the gauge fixes one phase for both. The relative
# phase across such a break is then measured, not gauged away — even when the
# broken station is itself the reference.

@testset "JointSmoother: the phase gauge spans per-antenna time segments" begin
    nant, nchan = 4, 6
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan)
    bp = GainComponent(
        ConstantTerm(); Ti = InstrumentScans([1.5]),
        Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = _JB_WIDE,
    )
    # Every station breaks at 1.5 s…
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; bandpass = bp), logamp = (; bandpass = bp)), anames, geom)
    # …or only A1 does, the rest holding one gain over the track.
    held = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = _JB_WIDE)
    het_model = CAL.GainModel(;
        phase = (; bandpass = held), logamp = (; bandpass = held),
        antennas = (A1 = (; phase = (; bandpass = bp), logamp = (; bandpass = bp)),),
    )
    lh = CAL.plan_parameters(het_model, anames, geom)

    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    pol_products = [(1, 1), (2, 2)]
    feeds = collect(pol_products)

    rng = MersenneTwister(20260908)
    # Station 1 breaks across the boundary; every other station holds one gain
    # over the whole track.
    gtrue = ones(ComplexF64, nant, 2, 2, nchan)
    for a in 1:nant, f in 1:2, c in 1:nchan
        gt = exp(complex(0.2 * randn(rng), 0.6 * randn(rng)))
        gtrue[a, f, 1, c] = gt
        gtrue[a, f, 2, c] = a == 1 ? exp(complex(0.2 * randn(rng), 0.6 * randn(rng))) : gt
    end
    Strue = [
        (0.5 + rand(rng)) * cis(2pi * rand(rng))
            for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(pol_products)
    ]

    het = fill(1, nant, 4)
    het[1, :] = [1, 1, 2, 2]
    uniform = repeat([1 1 2 2], nant)

    @testset "one reference phase per connected group of station gains" begin
        results = _joint_scan_accumulators(gtrue, Strue, het, bl_pairs, feeds, geom)
        # The constant stations bridge the break, so each (feed, channel) is one
        # group and the reference is zero in its first time segment only — its
        # second is free to carry the break it actually has.
        θ = zeros(lh.nθ)
        broken = first(only(_jb_components(lh, θ).phase))
        FP.solve_joint_bandpass!(results, geom, _jb_components(lh, θ); gauge = PinAntenna(1), tolerance = 1.0e-10)
        @test maximum(abs, broken.θ[1, :, :, 1, 1]) < 1.0e-12
        @test maximum(abs, broken.θ[1, :, :, 2, 1]) > 1.0e-3

        # A segmentation every station shares has nothing to bridge the epochs:
        # each (feed, time segment, channel) is its own group with its own
        # reference phase.
        θu = zeros(l.nθ)
        uniform_blocks = only(only(_jb_components(l, θu).phase))
        FP.solve_joint_bandpass!(
            _joint_scan_accumulators(gtrue, Strue, uniform, bl_pairs, feeds, geom), geom, _jb_components(l, θu);
            gauge = PinAntenna(1), tolerance = 1.0e-10,
        )
        @test maximum(abs, uniform_blocks.θ[1, :, :, :, 1]) < 1.0e-12
    end

    @testset "the reference antenna's own break is measured, not gauged away" begin
        θ = zeros(lh.nθ)
        comps = _jb_components(lh, θ)
        results = _joint_scan_accumulators(gtrue, Strue, het, bl_pairs, feeds, geom)
        FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1), max_iterations = 200, tolerance = 1.0e-13)

        broken, rest = only(comps.phase)
        # Everything is measured against the reference node — station 1's FIRST
        # segment — including station 1's second segment, each track up to its
        # constant.
        want_phase(a, f, ts) = _jb_cdemean(
            angle.(gtrue[a, f, ts, :]) .- angle.(gtrue[1, f, 1, :]),
        )
        for f in 1:2
            @test _jb_cdemean(broken.θ[1, f, :, 2, 1]) ≈ want_phase(1, f, 2) atol = 1.0e-8
            @test !isapprox(broken.θ[1, f, :, 2, 1], broken.θ[1, f, :, 1, 1]; atol = 1.0e-3)
            # The stations that span the break carry one parameter each, and it
            # is the truth both halves share — the break at station 1 leaks into
            # neither half.
            for (ai, a) in pairs(rest.antennas)
                @test _jb_cdemean(rest.θ[1, f, :, 1, ai]) ≈ want_phase(a, f, 1) atol = 1.0e-8
            end
        end
    end
end

# ── Per-station frequency segments ───────────────────────────────────────────
#
# Each station's gain is held in its own frequency segments, and the data are
# reduced onto the cells every station's segmentation is a union of.

@testset "JointSmoother: each antenna's own frequency segments" begin
    nant, nchan = 4, 6
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan)
    bpf(fs; prior = _JB_WIDE) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = fs, Feed = PerFeed(), prior)
    hetmodel(a1, rest; prior = _JB_WIDE) = CAL.GainModel(;
        phase = (; bandpass = bpf(rest; prior)), logamp = (; bandpass = bpf(rest)),
        antennas = (
            A1 = (; phase = (; bandpass = bpf(a1; prior)), logamp = (; bandpass = bpf(a1))),
        ),
    )
    # Each cell's segment under each block's segmentation.
    segment_of(block, cells) = [block.plan.fseg_id[first(c)] for c in cells]

    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    pol_products = [(1, 1), (2, 2)]
    feeds = collect(pol_products)

    @testset "segmentations that cut at different channels refine each other" begin
        # Station 1 in blocks of two channels, the rest in blocks of three: no
        # cut of one lands on every cut of the other, so the refinement is
        # strictly finer than either.
        l = CAL.plan_parameters(hetmodel(ChannelBlocks(2), ChannelBlocks(3)), anames, geom)
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        pb, ab = only(comps.phase), only(comps.logamp)
        @test [b.antennas for b in pb] == [[1], [2, 3, 4]]
        cells = FP._frequency_cells(comps, nchan)
        @test cells == [[1, 2], [3], [4], [5, 6]]
        @test segment_of(pb[1], cells) == [1, 2, 2, 3]
        @test segment_of(pb[2], cells) == [1, 1, 2, 2]

        # Neither station's segments nest inside the other's, so the whole band
        # is one connected group per feed: the gauge has ONE phase to fix.
        rng = MersenneTwister(20260908)
        # Each station's truth is constant over its OWN segments — two channels
        # for station 1, three for the rest.
        gtrue = ones(ComplexF64, nant, 2, 1, nchan)
        for a in 1:nant, f in 1:2
            for chans in (a == 1 ? [1:2, 3:4, 5:6] : [1:3, 4:6])
                gtrue[a, f, 1, chans] .= exp(complex(0.2 * randn(rng), 0.6 * randn(rng)))
            end
        end
        Strue = [
            (0.5 + rand(rng)) * cis(2pi * rand(rng))
                for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(pol_products)
        ]
        results = _joint_scan_accumulators(
            gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom,
        )
        # The misaligned refinement couples the two segmentations through cells
        # neither owns alone; the recovery is exact.
        FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1), max_iterations = 1000, tolerance = 1.0e-13)

        # The first channel of each of a station's own segments stands for it.
        reps(a) = a == 1 ? [1, 3, 5] : [1, 4]
        for f in 1:2, (bi, block) in pairs(pb)
            for (ai, a) in pairs(block.antennas)
                truth = [gtrue[a, f, 1, c] for c in reps(a)]
                @test _jb_cdemean(block.θ[1, f, :, 1, ai]) ≈ _jb_cdemean(angle.(truth)) atol = 1.0e-10
                @test _jb_demean(ab[bi].θ[1, f, :, 1, ai]) ≈ _jb_demean(log.(abs.(truth))) atol = 1.0e-10
            end
        end

        # A phase prior along frequency fits each station's track on its own
        # segments.
        rw = CAL.RandomWalkPrior(; σ = 0.1, init = Normal(0.0, 1.0))
        lp = CAL.plan_parameters(hetmodel(ChannelBlocks(2), ChannelBlocks(3); prior = rw), anames, geom)
        θ2 = zeros(lp.nθ)
        FP.solve_joint_bandpass!(results, geom, _jb_components(lp, θ2); gauge = PinAntenna(1))
        @test all(isfinite, θ2)
    end

    @testset "one segmentation for every antenna is the identity refinement" begin
        l = CAL.plan_parameters(
            CAL.GainModel(;
                phase = (; bandpass = bpf(ChannelBlocks(2))),
                logamp = (; bandpass = bpf(ChannelBlocks(2))),
            ), anames, geom,
        )
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        block = only(only(comps.phase))
        plan = block.plan
        cells = FP._frequency_cells(comps, nchan)
        # The cells ARE that segmentation's own channel groups and every station
        # maps each cell to itself, so the solve is the pre-refinement one.
        @test cells == CAL.segment_groups(plan.fseg_id, length(plan.nchan_seg))
        @test segment_of(block, cells) == collect(eachindex(cells))
        # Nothing ties one cell to another, so every cell is its own group and
        # the reference station's phase is zero in all of them.
        rng = MersenneTwister(23)
        g = exp.(complex.(0.1 .* randn(rng, nant, 2, 1, nchan), 0.5 .* randn(rng, nant, 2, 1, nchan)))
        S = [(0.5 + rand(rng)) * cis(2π * rand(rng)) for _ in 1:4, _ in bl_pairs, _ in feeds]
        results = _joint_scan_accumulators(g, S, fill(1, nant, 4), bl_pairs, feeds, geom)
        FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1), tolerance = 1.0e-10)
        @test maximum(abs, block.θ[1, :, :, 1, 1]) < 1.0e-12
    end

    @testset "an antenna holding one gain over cells the others split" begin
        # Station 1 carries ONE bandpass over the whole band while the rest
        # carry two, so its single segment pools both refinement cells and ties
        # them into one group per feed.
        l = CAL.plan_parameters(hetmodel(CAL.GlobalFrequency(), ChannelBlocks(3)), anames, geom)
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        phase_blocks, amp_blocks = only(comps.phase), only(comps.logamp)
        cells = FP._frequency_cells(comps, nchan)
        @test cells == [[1, 2, 3], [4, 5, 6]]
        @test segment_of(phase_blocks[1], cells) == [1, 1]
        @test segment_of(phase_blocks[2], cells) == [1, 2]
        rng = MersenneTwister(20260908)
        # Each station's truth is constant over its OWN segments: station 1 over
        # the whole band, the rest over each half.
        gtrue = ones(ComplexF64, nant, 2, 1, nchan)
        for a in 1:nant, f in 1:2
            for chans in (a == 1 ? [1:6] : [1:3, 4:6])
                gt = exp(complex(0.2 * randn(rng), 0.6 * randn(rng)))
                gtrue[a, f, 1, chans] .= gt
            end
        end
        Strue = [
            (0.5 + rand(rng)) * cis(2pi * rand(rng))
                for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(pol_products)
        ]
        results = _joint_scan_accumulators(
            gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom,
        )
        FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1), max_iterations = 200, tolerance = 1.0e-13)

        # The first channel of each cell stands for the segment holding it.
        rep = [1, 4]
        for f in 1:2
            # The station with one segment has nothing to vary against: its
            # phase is the reference's zero.
            @test maximum(abs, phase_blocks[1].θ[1, f, :, 1, 1]) < 1.0e-12
            # The rest recover their two segments from data pooled over three
            # channels each.
            for (ai, a) in pairs(phase_blocks[2].antennas)
                @test _jb_cdemean(phase_blocks[2].θ[1, f, :, 1, ai]) ≈
                    _jb_cdemean([angle(gtrue[a, f, 1, c]) for c in rep]) atol = 1.0e-8
                @test _jb_demean(amp_blocks[2].θ[1, f, :, 1, ai]) ≈
                    _jb_demean([log(abs(gtrue[a, f, 1, c])) for c in rep]) atol = 1.0e-8
            end
        end
        # Referencing a station whose segmentation is FINER than the group fixes
        # the same one phase per feed, so every track comes out the same up to
        # its constant.
        θ2 = zeros(l.nθ)
        comps2 = _jb_components(l, θ2)
        FP.solve_joint_bandpass!(results, geom, comps2; gauge = PinAntenna(2), max_iterations = 200, tolerance = 1.0e-13)
        for (b2, b1) in zip(only(comps2.phase), phase_blocks), ai in eachindex(b1.antennas), f in 1:2
            @test _jb_cdemean(b2.θ[1, f, :, 1, ai]) ≈ _jb_cdemean(b1.θ[1, f, :, 1, ai]) atol = 1.0e-8
        end
    end

    @testset "hyperparameters are resolved and recorded under $(nameof(typeof(spec)))" for spec in (
            CAL.OUPrior(; scale = LogNormal(log(2.0e6), 1.0), σ = LogNormal(log(0.2), 1.0)),
            CAL.CARMAPrior(;
                ar = (CAL.ComplexRoots(; period = LogNormal(log(8.0e6), 1.0), scale = LogNormal(log(4.0e6), 1.0)),),
                ma = (CAL.RealRoot(; scale = 1.0e6),), σ = LogNormal(log(0.2), 1.0),
            ),
        )
        l = CAL.plan_parameters(
            CAL.GainModel(;
                phase = (; bandpass = bpf(ChannelBlocks(1))), logamp = (; bandpass = bpf(ChannelBlocks(1); prior = spec)),
            ), anames, geom,
        )
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        ab = only(comps.logamp)
        rng = MersenneTwister(5)
        gtrue = [exp(complex(0.1 * randn(rng), 0.3 * randn(rng))) for _ in 1:nant, _ in 1:2, _ in 1:1, _ in 1:nchan]
        Strue = [(0.5 + rand(rng)) * cis(2pi * rand(rng)) for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(feeds)]
        results = _joint_scan_accumulators(gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom)
        amp_priors = [FP._block_prior_array(b, geom) for b in ab]
        @test all(p -> p === spec, only(amp_priors))
        fits = FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1))
        FP._record_track_fits!(nothing, amp_priors, fits.amp, fits.status.logamp, (:logamp, :bandpass), geom, ab)
        @test all(p -> p isa typeof(spec).name.wrapper && FP._hyper_count(p) == 0, only(amp_priors))
        rep = FP._bandpass_track_report(nothing, nothing; amp_priors = only(amp_priors))
        @test rep.amp_priors === only(amp_priors)

        θ1 = zeros(l.nθ)
        @test_logs (:warn, r"stopped at 1 EM iterations") match_mode = :any FP.solve_joint_bandpass!(
            results, geom, _jb_components(l, θ1); gauge = PinAntenna(1), max_em_iterations = 1,
        )
    end
end

# ── A level per spectral window beside a zero-mean shape ─────────────────────

@testset "JointSmoother: levels per spectral window plus an $name shape" for (name, shape_prior) in (
        ("OU", CAL.OUPrior(; scale = 3.0e6, σ = 1.0)),
        ("CARMA", CAL.CARMAPrior(; ar = (CAL.ComplexRoots(; period = 1.2e7, scale = 3.0e6),), ma = (CAL.RealRoot(; scale = 1.0e6),), σ = 1.0)),
    )
    nant, nchan = 4, 12
    anames = ["A$i" for i in 1:nant]
    geom = CAL.DataGeometry(;
        nfeed = 2,
        times = collect(0.0:3.0), scan_of_time = collect(1:4),
        channel_freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6),
        spw_of_chan = repeat([1, 2]; inner = nchan ÷ 2),
        scan_names = ["No00$i" for i in 1:4], spw_names = ["A", "B"],
        antennas = anames,
    )
    comp(fs; prior) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = fs, Feed = PerFeed(), prior)
    obs = (; level = comp(CAL.PerSpectralWindow(); prior = IIDPrior(10.0)), shape = comp(ChannelBlocks(1); prior = shape_prior))
    l = CAL.plan_parameters(CAL.GainModel(; phase = obs, logamp = obs), anames, geom)
    θ = zeros(l.nθ)
    comps = _jb_components(l, θ)
    plb, pb = comps.phase
    alb, ab = comps.logamp

    # A step between the windows, and a small smooth ripple within each.
    rng = MersenneTwister(77)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (2, 2)]
    spw = geom.spw_of_chan
    ripple = 0.03 .* sin.((1:nchan) ./ 2)
    gtrue = ones(ComplexF64, nant, 2, 1, nchan)
    for a in 1:nant, f in 1:2
        step_la, step_φ = 0.3 * randn(rng, 2), 0.8 * randn(rng, 2)
        gtrue[a, f, 1, :] .= exp.(complex.(step_la[spw] .+ ripple, step_φ[spw] .+ ripple))
    end
    Strue = [(0.5 + rand(rng)) * cis(2pi * rand(rng)) for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(feeds)]
    results = _joint_scan_accumulators(gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom)
    FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1))

    level_leaf(b) = only(b).θ
    shape_leaf(b) = only(b).θ
    for a in 2:nant, f in 1:2
        # Level plus shape is the gain, each track up to its constant.
        la = [level_leaf(alb)[1, f, spw[c], 1, a] + shape_leaf(ab)[1, f, c, 1, a] for c in 1:nchan]
        φ = [level_leaf(plb)[1, f, spw[c], 1, a] + shape_leaf(pb)[1, f, c, 1, a] for c in 1:nchan]
        @test _jb_demean(la) ≈ _jb_demean(log.(abs.(gtrue[a, f, 1, :]))) atol = 1.0e-2
        want = angle.(gtrue[a, f, 1, :]) .- angle.(gtrue[1, f, 1, :])
        @test maximum(abs, rem2pi.(_jb_cdemean(φ) .- _jb_cdemean(want), RoundNearest)) < 1.0e-2
        # The level carries the step; the shape stays small.
        @test maximum(abs, shape_leaf(ab)[1, f, :, 1, a]) < 0.1
        Lstep = level_leaf(alb)[1, f, 2, 1, a] - level_leaf(alb)[1, f, 1, 1, a]
        tstep = log(abs(gtrue[a, f, 1, end])) - log(abs(gtrue[a, f, 1, 1]))
        @test Lstep ≈ tstep atol = 0.05
    end
end

# ── A delay beside a zero-mean shape ─────────────────────────────────────────

@testset "JointSmoother: a delay beside an OU shape takes up each feed's delay" begin
    nant, nchan = 4, 64
    anames = ["A$i" for i in 1:nant]
    geom = CAL.DataGeometry(;
        nfeed = 2,
        times = collect(0.0:3.0), scan_of_time = collect(1:4),
        channel_freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6),
        spw_of_chan = repeat([1, 2]; inner = nchan ÷ 2),
        scan_names = ["No00$i" for i in 1:4], spw_names = ["A", "B"],
        antennas = anames,
    )
    ou = CAL.OUPrior(; scale = LogNormal(log(8.0e6), 1.0), σ = LogNormal(log(0.1), 1.5))
    shape = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = ou)
    delay = GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Feed = PerFeed(), prior = IIDPrior(1.0e-6))
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; delay, shape), logamp = (; shape)), anames, geom)
    θ = zeros(l.nθ)
    comps = _jb_components(l, θ)
    db, pb = comps.phase
    @test length(db) == 1

    # Delays of a few ns per feed, several radians across the band, and a small
    # ripple; the cross-hands relate the feeds.
    rng = MersenneTwister(78)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (1, 2), (2, 1), (2, 2)]
    f0 = geom.f0
    τtrue = 4.0e-9 .* randn(rng, nant, 2)
    ripple = 0.03 .* sin.((1:nchan) ./ 3)
    gtrue = ones(ComplexF64, nant, 2, 1, nchan)
    for a in 1:nant, f in 1:2
        gtrue[a, f, 1, :] .= cis.(2π .* τtrue[a, f] .* (geom.channel_freqs .- f0) .+ ripple)
    end
    Strue = [(0.5 + rand(rng)) * cis(2pi * rand(rng)) for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(feeds)]
    results = _joint_scan_accumulators(gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom)
    FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1))

    τ = only(db).θ
    shape_φ = only(pb).θ
    for a in 1:nant, f in 1:2
        # Delays are measured against the reference's feed 1, whose phase is
        # zero.
        @test τ[1, f, 1, 1, a] ≈ τtrue[a, f] - τtrue[1, 1] atol = 0.05e-9
        φ = shape_φ[1, f, :, 1, a] .+ 2π .* τ[1, f, 1, 1, a] .* (geom.channel_freqs .- f0)
        want = angle.(gtrue[a, f, 1, :]) .- angle.(gtrue[1, 1, 1, :])
        @test maximum(abs, rem2pi.(_jb_cdemean(φ) .- _jb_cdemean(want), RoundNearest)) < 1.0e-2
        # The shape keeps only the ripple.
        @test maximum(abs, shape_φ[1, f, :, 1, a]) < 0.2
    end
end

@testset "JointSmoother: a station's prior on its mean draws it toward zero" begin
    nant, nchan = 4, 64
    anames = ["A$i" for i in 1:nant]
    geom = CAL.DataGeometry(;
        nfeed = 2,
        times = collect(0.0:3.0), scan_of_time = collect(1:4),
        channel_freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6),
        spw_of_chan = repeat([1, 2]; inner = nchan ÷ 2),
        scan_names = ["No00$i" for i in 1:4], spw_names = ["A", "B"],
        antennas = anames,
    )
    ou = CAL.OUPrior(; scale = LogNormal(log(8.0e6), 1.0), σ = LogNormal(log(0.1), 1.5))
    shape = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = ou)
    level(prior = IIDPrior(10.0)) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.PerSpectralWindow(), Feed = PerFeed(), prior)
    delay(prior = IIDPrior(1.0e-6)) = GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Feed = PerFeed(), prior)

    rng = MersenneTwister(79)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (1, 2), (2, 1), (2, 2)]
    f0 = geom.f0
    spw = geom.spw_of_chan
    τtrue = 4.0e-9 .* randn(rng, nant, 2)
    steps = 0.5 .* randn(rng, nant, 2, 2)
    ripple = 0.03 .* sin.((1:nchan) ./ 3)
    gtrue = ones(ComplexF64, nant, 2, 1, nchan)
    for a in 1:nant, f in 1:2
        gtrue[a, f, 1, :] .= cis.(2π .* τtrue[a, f] .* (geom.channel_freqs .- f0) .+ steps[a, f, spw] .+ ripple)
    end
    Strue = [(0.5 + rand(rng)) * cis(2pi * rand(rng)) for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(feeds)]
    results = _joint_scan_accumulators(gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom)

    # Each station's level per window and delay per feed, from the blocks that hold it.
    station_value(blocks, a, f, g) = only(b.θ[1, f, g, 1, findfirst(==(a), b.antennas)] for b in blocks if a in b.antennas)
    function solve(model)
        l = CAL.plan_parameters(model, anames, geom)
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        plb, db = CAL.antenna_blocks(l, θ, :phase, :level), CAL.antenna_blocks(l, θ, :phase, :delay)
        FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1))
        return (;
            τ = [station_value(db, a, f, 1) for a in 1:nant, f in 1:2],
            L = [station_value(plb, a, f, g) for a in 1:nant, f in 1:2, g in 1:2],
        )
    end
    flat = solve(CAL.GainModel(; phase = (; level = level(), delay = delay(), shape), logamp = (; shape)))
    @test flat.τ ≈ τtrue .- τtrue[1, 1] atol = 0.05e-9

    # A prior wider still leaves the solution.
    wide = solve(CAL.GainModel(; phase = (; level = level(CAL.IIDPrior(100.0)), delay = delay(CAL.IIDPrior(1.0e-5)), shape), logamp = (; shape)))
    @test maximum(abs, wide.τ .- flat.τ) < 5.0e-12
    @test maximum(abs, (wide.L .- wide.L[:, :, 1:1]) .- (flat.L .- flat.L[:, :, 1:1])) < 1.0e-3

    # A tight prior on one station holds its own mean at zero.
    base = CAL.GainModel(; phase = (; level = level(), delay = delay(), shape), logamp = (; shape))
    tight = solve(with_antenna(base, "A3"; phase = (; level = level(CAL.IIDPrior(1.0e-3)), delay = delay(CAL.IIDPrior(1.0e-12)), shape)))
    @test maximum(abs, tight.τ[3, :]) < 1.0e-3 * maximum(abs, flat.τ[3, :])
    @test maximum(abs, tight.L[3, :, :]) < 1.0e-3 * maximum(abs, flat.L[3, :, :])
end

# The same-feed products do not see a delay added to the second feed of every
# station, so only the weak different-feed products measure it, whatever the
# feeds' basis.
@testset "JointSmoother: a delay common to every station's second feed" begin
    nant, nspw, nchan, chan_bw, spw_sep, ref_freq = 5, 4, 32, 4.0e6, 1.28e8, 43.0e9
    freqs = [ref_freq + (b - 1) * spw_sep + (c - 1) * chan_bw for b in 1:nspw for c in 1:nchan]
    f0 = sum(freqs) / length(freqs)
    rng = MersenneTwister(4242)
    τ = 8.0e-9 .+ 0.5e-9 .* randn(rng, nant)
    bandpass = zeros(nant, 2, length(freqs))
    for a in 1:nant, f in 1:2
        bandpass[a, f, :] .= 0.3 .* sin.(2π .* (freqs .- f0) ./ (60 * chan_bw) .+ 2π * rand(rng))
    end
    for a in 1:nant
        bandpass[a, 2, :] .+= 2π .* τ[a] .* (freqs .- f0)
    end
    ou = CAL.OUPrior(; scale = LogNormal(log(4 * chan_bw), 1.0), σ = LogNormal(log(0.1), 1.5))
    shape = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = ou)
    delay = GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Feed = PerFeed(), prior = IIDPrior(1.0e-6))
    solver = Bandpass(; model = CAL.GainModel(; phase = (; delay, shape)), gauge = PinAntenna(1))
    build(; kw...) = first(
        _build_fringe_ps(;
            nant, nspw, nchan, chan_bw, spw_sep, ref_freq, nscans = 4, noise = 0.25, antenna_gains = false,
            bandpass, kw...,
        )
    )
    # The largest error of any station's phase between its feeds, about its band mean.
    function rl_error(sol)
        g = gains(sol)
        @test collect(lookup(g, Frequency)) ≈ freqs
        worst = 0.0
        for name in lookup(g, AntennaName)
            a = parse(Int, name[2:end])
            z = g[Ti = 1, AntennaName = At(name), Feed = At(2)] ./ g[Ti = 1, AntennaName = At(name), Feed = At(1)]
            e = angle.(z) .- (bandpass[a, 2, :] .- bandpass[a, 1, :])
            worst = max(worst, maximum(abs, rem2pi.(e .- angle(sum(cis, e)), RoundNearest)))
        end
        return worst
    end
    # A threaded BLAS adds its own warning to every fit.
    blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @testset "$basis feeds" for (basis, kw) in (
            ("circular", (;)),
            ("linear", (; receptors = ("X", "Y"), polarizations = ["XX", "XY", "YX", "YY"])),
            ("mixed", (; receptor_order = Dict("A3" => ["X", "Y"], "A4" => ["X", "Y"]))),
        )
        sol = @test_logs fit(solver, build(; different_feed_amplitude = 0.03, kw...))
        @test rl_error(sol) < 0.2
    end
    @test_logs (:warn, r"found no delay of feed 2") match_mode = :any fit(solver, build(; different_feed_amplitude = 0.0))
    BLAS.set_num_threads(blas)
end

# ── The gauge after the solve ────────────────────────────────────────────────
#
# The free direction of the joint solve is a phase spectrum common to every
# station. The priors fix the solution along it, and the gauge then moves the
# phases along it at least cost in prior energy, so no gauge changes what the
# data measure.

@testset "JointSmoother: the gauge moves only what the data leave free" begin
    nant, nchan = 6, 16
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan; nant)
    comp(; prior = _JB_WIDE) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (2, 2)]

    # Smooth phase bandpasses, except the reference station's, which has
    # strong channel-to-channel structure.
    rng = MersenneTwister(286)
    ν = range(0, 1; length = nchan)
    gtrue = ones(ComplexF64, nant, 2, 1, nchan)
    for a in 1:nant, f in 1:2
        φ = a == 1 ? 0.8 .* randn(rng, nchan) : 0.5 .* sin.(2π .* (ν .+ rand(rng)))
        gtrue[a, f, 1, :] .= exp.(complex.(0.1 .* randn(rng, nchan), φ))
    end
    Strue = [(0.5 + rand(rng)) * cis(2pi * rand(rng)) for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(feeds)]
    results = _joint_scan_accumulators(gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom)

    layout(prior = _JB_WIDE, amp_prior = _JB_WIDE) = CAL.plan_parameters(
        CAL.GainModel(; phase = (; bandpass = comp(; prior)), logamp = (; bandpass = comp(; prior = amp_prior))), anames, geom,
    )
    # Converged within `max_iterations` iterations: no warning.
    function solve(gauge; prior = _JB_WIDE, amp_prior = _JB_WIDE, data = results, max_iterations = 200)
        l = layout(prior, amp_prior)
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        @test_logs FP.solve_joint_bandpass!(data, geom, comps; gauge, max_iterations, tolerance = 1.0e-10)
        φ = only(only(comps.phase)).θ
        return [φ[1, f, c, 1, a] for a in 1:nant, f in 1:2, c in 1:nchan]
    end
    # The largest and rms phase error of every baseline product against truth,
    # each referenced to its own circular band mean, the constant the source
    # coherence absorbs.
    function product_error(φ)
        worst, ss, n = 0.0, 0.0, 0
        for (a, b) in bl_pairs, f in 1:2
            e = [φ[a, f, c] - φ[b, f, c] - angle(gtrue[a, f, 1, c] * conj(gtrue[b, f, 1, c])) for c in 1:nchan]
            e = rem2pi.(e .- angle(sum(cis, e)), RoundNearest)
            worst = max(worst, maximum(abs, e))
            ss += sum(abs2, e)
            n += nchan
        end
        return worst, sqrt(ss / n)
    end
    # The largest station sum of any (feed, channel).
    antenna_sum(φ) = maximum(abs, sum(φ; dims = 1))

    # A second-order walk of `σ` rad per channel (1 MHz), as rad/Hz^(3/2), from
    # a weak start: 10 rad, and 10 rad over the 16 MHz band.
    rw2(σ) = CAL.RandomWalkPrior(;
        order = 2, σ = σ * sqrt(3 / (2 * 1.0e6^3)), init = MvNormal(zeros(2), Diagonal([10.0^2, (10.0 / 16.0e6)^2])),
    )

    @testset "under a wide prior the gauge leaves the products unchanged" begin
        zs, pin = solve(ZeroSumPhase()), solve(PinAntenna(1))
        @test antenna_sum(zs) < 1.0e-8
        @test product_error(zs)[1] < 1.0e-8
        @test product_error(pin)[1] < 1.0e-8
    end

    @testset "under a prior along frequency the gauge leaves the products unchanged" begin
        prior = rw2(1.0)
        zs, pin = solve(ZeroSumPhase(); prior), solve(PinAntenna(1); prior)
        @test antenna_sum(zs) < 1.0e-8
        @test maximum(abs, pin[1, :, :]) < 1.0e-12
        # Every baseline product is the same under either gauge.
        products(φ) = [φ[a, f, c] - φ[b, f, c] for (a, b) in bl_pairs, f in 1:2, c in 1:nchan]
        @test maximum(abs, rem2pi.(products(zs) .- products(pin), RoundNearest)) < 1.0e-8

        # `ByComponent` maps the bandpass component to its choice.
        bc(choice, default) = ByComponent((; bandpass = choice); default)
        @test solve(bc(ZeroSumPhase(), PinAntenna(1)); prior) == zs
        @test solve(bc(PinAntenna(1), ZeroSumPhase()); prior) == pin
    end

    @testset "a stiff prior converges under either gauge" for σ in (0.05, 0.01)
        # A phase this stiff cannot follow the reference's structure.
        zs = solve(ZeroSumPhase(); prior = rw2(σ), amp_prior = rw2(0.05), max_iterations = 80)
        solve(PinAntenna(1); prior = rw2(σ), amp_prior = rw2(0.05), max_iterations = 80)
        @test antenna_sum(zs) < 1.0e-8
    end

    @testset "a station gain without data takes the constraint through its priors" begin
        gap = deepcopy(results)
        for r in gap, (bi, (a, b)) in pairs(bl_pairs)
            3 in (a, b) || continue
            r.wl[bi, :, 5] .= 0
            r.rl[bi, :, 5] .= 0
        end
        zs = solve(ZeroSumPhase(); prior = rw2(0.05), amp_prior = rw2(0.05), data = gap)
        @test all(isfinite, zs[3, :, 5])
        # The sum runs over the station gains with data.
        @test antenna_sum(zs[:, :, setdiff(1:nchan, 5)]) < 1.0e-8
        @test antenna_sum(zs[[1, 2, 4, 5, 6], :, 5:5]) < 1.0e-8
    end
end

@testset "JointSmoother: weak cross-hands converge in a few iterations" begin
    nant, nchan = 5, 12
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan; nant)
    comp = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = _JB_WIDE)
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; bandpass = comp), logamp = (; bandpass = comp)), anames, geom)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rng = MersenneTwister(313)
    gtrue = exp.(complex.(0.1 .* randn(rng, nant, 2, 1, nchan), 0.6 .* randn(rng, nant, 2, 1, nchan)))
    # The cross-hand source terms carry ~1e-3 of the parallel hands' information;
    # a step on every gain at once resolves the phase between feeds they measure.
    # At unit weights they are far below the noise those weights state, so the
    # start's delay search between the feeds detects nothing.
    undetected = (:warn, r"found no delay of feed 2")
    S = [(p in (2, 3) ? 0.03 : 1.0) * (0.5 + rand(rng)) * cis(2π * rand(rng)) for _ in 1:4, _ in bl_pairs, p in eachindex(feeds)]
    results = _joint_scan_accumulators(gtrue, S, fill(1, nant, 4), bl_pairs, feeds, geom)
    # Every station's phase between feeds, referenced to its circular band mean.
    function rl_error(φ)
        worst = 0.0
        for a in 1:nant
            e = [φ[a, 1, c] - φ[a, 2, c] - angle(gtrue[a, 1, 1, c] * conj(gtrue[a, 2, 1, c])) for c in 1:nchan]
            worst = max(worst, maximum(abs, rem2pi.(e .- angle(sum(cis, e)), RoundNearest)))
        end
        return worst
    end
    for gauge in (PinAntenna(1), ZeroSumPhase())
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        @test_logs undetected FP.solve_joint_bandpass!(results, geom, comps; gauge, max_iterations = 30, tolerance = 1.0e-10)
        φ = only(only(comps.phase)).θ
        @test rl_error([φ[1, f, c, 1, a] for a in 1:nant, f in 1:2, c in 1:nchan]) < 1.0e-8
    end

    # Under a prior the step is the MAP step. The source term re-solved between
    # steps makes the last digits come linearly here.
    rw2 = CAL.RandomWalkPrior(;
        order = 2, σ = 0.3 * sqrt(3 / (2 * 1.0e6^3)), init = MvNormal(zeros(2), Diagonal([10.0^2, (10.0 / 12.0e6)^2])),
    )
    ou = CAL.OUPrior(; scale = 3.0e6, σ = 0.5)
    cp = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = rw2)
    leveled = (;
        level = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.PerSpectralWindow(), Feed = PerFeed(), prior = IIDPrior(10.0)),
        shape = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = ou),
    )
    @testset "$name, $gauge" for (name, obs) in (("RW2", (; bandpass = cp)), ("level + OU", leveled)), gauge in (PinAntenna(1), ZeroSumPhase())
        lp = CAL.plan_parameters(CAL.GainModel(; phase = obs, logamp = obs), anames, geom)
        θ = zeros(lp.nθ)
        @test_logs undetected FP.solve_joint_bandpass!(
            results, geom, _jb_components(lp, θ); gauge, max_iterations = 60, tolerance = 1.0e-10,
        )
        # Another Krylov method of LinearSolve.jl solves the same steps.
        θm = zeros(lp.nθ)
        FP.solve_joint_bandpass!(
            results, geom, _jb_components(lp, θm);
            gauge, max_iterations = 60, tolerance = 1.0e-10, linsolve = FP.LinearSolve.KrylovJL_MINRES(),
        )
        @test all(isfinite, θ)
        @test θm ≈ θ rtol = 1.0e-6
    end
end

# ── Stopping in standard errors ──────────────────────────────────────────────
#
# A station whose data carry almost no information moves by large fractions of
# a radian between iterations long after the solution has settled within its
# standard error; the default tolerance stops on the change in standard errors.

@testset "JointSmoother: the default tolerance stops within a fraction of a standard error" begin
    nant, nchan = 5, 48
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan; nant)
    comp = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = IIDPrior(10.0))
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; bandpass = comp), logamp = (; bandpass = comp)), anames, geom)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (2, 2)]
    rng = MersenneTwister(308)
    gtrue = exp.(complex.(0.1 .* randn(rng, nant, 2, 1, nchan), 0.6 .* randn(rng, nant, 2, 1, nchan)))
    S = [(0.5 + rand(rng)) * cis(2π * rand(rng)) for _ in 1:4, _ in bl_pairs, _ in feeds]
    results = _joint_scan_accumulators(gtrue, S, fill(1, nant, 4), bl_pairs, feeds, geom)
    # Weight 1/σ² per real component; station A5 carries 2e-6 of the rest's.
    for res in results, (bi, (a, b)) in pairs(bl_pairs)
        w = nant in (a, b) ? 1.0e-4 : 50.0
        res.wl[bi, :, :] .= w
        res.rl[bi, :, :] .= w .* res.rl[bi, :, :] .+ sqrt(w) .* complex.(randn(rng, 2, nchan), randn(rng, 2, nchan))
    end
    function solve(; kw...)
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1), kw...)
        return only(only(comps.phase)).θ[1, :, :, 1, :]
    end
    φ_default = @test_logs solve(max_iterations = 60)
    φ_tight = solve(max_iterations = 5000, tolerance = 1.0e-8)
    # Phase standard error of a well-measured station's gain: 1/√(Σ w|m|²) over
    # its 4 baselines and 4 scans, with |g| ≈ 1 and |S| ≥ 0.5.
    σφ = 1 / sqrt(4 * 4 * 50.0 * 0.25)
    # Each (feed, station) track is compared up to its constant, which the
    # data do not measure.
    d = rem2pi.(φ_default[:, :, 1:(nant - 1)] .- φ_tight[:, :, 1:(nant - 1)], RoundNearest)
    d = rem2pi.(d .- angle.(sum(cis.(d); dims = 2)), RoundNearest)
    @test maximum(abs, d) < 0.1σφ

    @test_logs (:warn, r"did not converge in 2 iterations: the last step moved a gain by .* of its standard error") match_mode = :any solve(max_iterations = 2)
end

# ── Band-referenced products ──────────────────────────────────────────────────

@testset "JointSmoother: gauges agree on band-referenced products" begin
    bl_pairs(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (2, 2)]
    bpf(fs; prior = _JB_WIDE) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = fs, Feed = PerFeed(), prior)
    hetmodel(a1, rest; prior = _JB_WIDE) = CAL.GainModel(;
        phase = (; bandpass = bpf(rest; prior)), logamp = (; bandpass = bpf(rest)),
        antennas = (A1 = (; phase = (; bandpass = bpf(a1; prior)), logamp = (; bandpass = bpf(a1))),),
    )
    # Station 1 holds one gain per `n1` channels and the rest one per `nrest`.
    function synthetic(n1, nrest; nant, nchan, spread, seed)
        rng = MersenneTwister(seed)
        gtrue = ones(ComplexF64, nant, 2, 1, nchan)
        for a in 1:nant, f in 1:2
            n = a == 1 ? n1 : nrest
            for c0 in 1:n:nchan
                gtrue[a, f, 1, c0:(c0 + n - 1)] .= exp(complex(0.2 * randn(rng), spread * randn(rng)))
            end
        end
        S = [(0.5 + rand(rng)) * cis(2π * rand(rng)) for _ in 1:4, _ in bl_pairs(nant), _ in feeds]
        geom = _seg_geometry(nchan; nant)
        return gtrue, geom, _joint_scan_accumulators(gtrue, S, fill(1, nant, 4), bl_pairs(nant), feeds, geom)
    end
    function solve(n1, nrest, gauge, geom, results; prior = _JB_WIDE, max_iterations = 2000)
        nant = length(geom.antennas)
        l = CAL.plan_parameters(hetmodel(ChannelBlocks(n1), ChannelBlocks(nrest); prior), geom.antennas, geom)
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        pb = only(comps.phase)
        FP.solve_joint_bandpass!(results, geom, comps; gauge, max_iterations, tolerance = 1.0e-12)
        φ = zeros(nant, 2, length(geom.channel_freqs))
        for blk in pb, (ai, a) in pairs(blk.antennas), f in 1:2, c in axes(φ, 3)
            φ[a, f, c] = blk.θ[1, f, blk.plan.fseg_id[c], 1, ai]
        end
        return φ
    end
    # The largest phase error of every baseline product against truth, each
    # referenced to its own circular band mean, the constant `S` absorbs.
    function product_error(φ, gtrue)
        worst = 0.0
        for (a, b) in bl_pairs(size(φ, 1)), f in 1:2
            e = [φ[a, f, c] - φ[b, f, c] - angle(gtrue[a, f, 1, c] * conj(gtrue[b, f, 1, c])) for c in axes(φ, 3)]
            e = rem2pi.(e .- angle(sum(cis, e)), RoundNearest)
            worst = max(worst, maximum(abs, e))
        end
        return worst
    end

    @testset "antenna 1 on $n1-channel blocks, the rest on $nrest" for (n1, nrest) in ((1, 2), (2, 1))
        gtrue, geom, results = synthetic(n1, nrest; nant = 4, nchan = 4, spread = 0.6, seed = 302)
        for gauge in (PinAntenna(1), PinAntenna(2), ZeroSumPhase())
            @test product_error(solve(n1, nrest, gauge, geom, results), gtrue) < 1.0e-8
        end
    end

    @testset "per-channel structure a stiff prior cannot follow" begin
        gtrue, geom, results = synthetic(1, 1; nant = 6, nchan = 16, spread = 2.0, seed = 302)
        @test product_error(solve(1, 1, ZeroSumPhase(), geom, results), gtrue) < 1.0e-8
        # A phase prior far too stiff for that structure leaves the phases a
        # smooth track the data pull on.
        prior = CAL.RandomWalkPrior(;
            order = 2, σ = sqrt(3 / (2 * 1.0e6^3)), init = MvNormal(zeros(2), Diagonal([10.0^2, (10.0 / 16.0e6)^2])),
        )
        φp = solve(1, 1, ZeroSumPhase(), geom, results; prior, max_iterations = 100)
        @test all(isfinite, φp)
        @test product_error(φp, gtrue) > 1
    end
end

# ── One observable ──────────────────────────────────────────────────────────

@testset "JointSmoother: a station whose baselines carry noise alone leaves the others as if it were flagged" begin
    nant, nspw, nchan, ntime, nscans = 4, 2, 16, 6, 3
    rng = MersenneTwister(0xDEC1)
    bp_true = cumsum(0.1 .* randn(rng, nant, 2, nspw * nchan); dims = 3)
    ps, _ = _build_fringe_ps(; nant, nspw, nchan, ntime, nscans, antenna_gains = false, bandpass = bp_true, noise = 0.3, seed = 31)
    # A4's baselines carry noise alone.
    for ms in values(ps)
        V = DimensionalData.modify(Array, ms[:visibility])
        for (bi, pair) in pairs(XRadio.baselines(ms))
            "A4" in pair || continue
            v = view(V, BaselineID(bi))
            v .= ComplexF32.(0.3 .* complex.(randn(rng, size(v)), randn(rng, size(v))) ./ sqrt(2))
        end
        ms[:visibility] = V
    end
    unit = sqrt(3 / (2 * 2.0e6^3))
    # A weak start: 10 rad, and 10 rad over a 32 MHz spectral window.
    init = MvNormal(zeros(2), Diagonal([10.0^2, (10.0 / 32.0e6)^2]))
    rw2(σ) = CAL.RandomWalkPrior(; order = 2, σ = LogNormal(log(σ * unit), 1.0), init)
    bpc(prior) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior)
    solver = Bandpass(;
        model = CAL.GainModel(; phase = (; bandpass = bpc(rw2(0.05))), logamp = (; bandpass = bpc(rw2(0.02)))), gauge = PinAntenna(1),
    )
    sol = fit(solver, ps)

    flagged = Gustavo.materialize(ps)
    for ms in values(flagged), (bi, pair) in pairs(XRadio.baselines(ms))
        "A4" in pair && (view(ms[:flag], BaselineID(bi)) .= true)
    end
    ref = @test_logs (:warn, r"carry no measured frequency shape") match_mode = :any fit(solver, flagged)
    st = sol.info.phase_status
    others = setdiff(axes(st, 1), findfirst(==("A4"), lookup(st, AntennaName)))
    # Each (station, feed) track is compared up to its constant.
    centered(x) = x .- mean(x; dims = 3)
    for obs in (:phase, :logamp)
        @test maximum(abs, centered(_jb_leaf(sol, obs)[:, :, :, :, others]) .- centered(_jb_leaf(ref, obs)[:, :, :, :, others])) < 1.0e-3
    end
end

@testset "JointSmoother: a phase-only or amplitude-only model holds the other at zero" begin
    nant, nchan, ntime, nscans = 4, 8, 6, 6
    rng = MersenneTwister(0x0515)
    gauge = PinAntenna(1)
    fm = default_fringe_model()
    terms = _bandpass_model()
    # Where the data hold no bandpass in the other observable, a one-sided model
    # gives the same answer as the two-sided one.
    function compare(obs, data_kw)
        ps, _ = _build_fringe_ps(; nant, nspw = 1, nchan, ntime, nscans, data_kw...)
        one_sided = GainModel(; (obs => getproperty(terms, obs),)...)
        _, sol1 = _fit_chain((FringeClosure(; model = fm, gauge), Bandpass(; model = one_sided, gauge)), ps)
        _, sol2 = _fit_chain((FringeClosure(; model = fm, gauge), Bandpass(; model = terms, gauge)), ps)
        centered(x) = x .- mean(x; dims = 3)
        return sol1, maximum(abs, centered(_jb_leaf(sol1, obs)) .- centered(_jb_leaf(sol2, obs)))
    end

    sol, Δ = compare(:phase, (; bandpass = 0.4 .* randn(rng, nant, 2, nchan), seed = 7))
    @test Δ < 1.0e-4
    @test isempty(sol.info.amp_status)

    abp_true = 0.15 .* randn(rng, nant, 2, nchan)
    # A gain ten times too strong at one channel is written as fit.
    abp_true[2, 1, 4] = 3.0
    sol, Δ = compare(:logamp, (; amp_bandpass = abp_true, seed = 8))
    @test Δ < 1.0e-4
    @test isempty(sol.info.phase_status)
    leaf = _jb_leaf(sol, :logamp)
    @test all(abs(_jb_demean(leaf[1, f, :, 1, a])[c] - _jb_demean(abp_true[a, f, :])[c]) < 1.0e-3 for a in 1:nant, f in 1:2, c in 1:nchan)
end

# ── A log-amplitude bandpass the data cannot divide ─────────────────────────

@testset "JointSmoother: baselines without an odd loop leave the amplitude split free" begin
    nchan = 8
    feeds = [(1, 1), (2, 2)]
    comp(prior) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior)
    function solve(nant, bl_pairs; amp_prior = _JB_WIDE)
        geom = _seg_geometry(nchan; nant)
        l = CAL.plan_parameters(CAL.GainModel(; phase = (; bandpass = comp(_JB_WIDE)), logamp = (; bandpass = comp(amp_prior))), geom.antennas, geom)
        rng = MersenneTwister(304)
        gtrue = exp.(complex.(0.3 .* randn(rng, nant, 2, 1, nchan), 0.5 .* randn(rng, nant, 2, 1, nchan)))
        S = [(0.5 + rand(rng)) * cis(2π * rand(rng)) for _ in 1:4, _ in bl_pairs, _ in feeds]
        results = _joint_scan_accumulators(gtrue, S, fill(1, nant, 4), bl_pairs, feeds, geom)
        θ = zeros(l.nθ)
        comps = _jb_components(l, θ)
        FP.solve_joint_bandpass!(results, geom, comps; gauge = PinAntenna(1), tolerance = 1.0e-10)
        return only(only(comps.logamp)).θ[1, :, :, 1, :], log.(abs.(gtrue[:, :, 1, :]))   # (feed, channel, station), (station, feed, channel)
    end
    undetermined = (:warn, r"no odd loop")
    centered(v) = v .- sum(v) / length(v)

    # Two stations: each channel's amplitude sum is measured, its split is not;
    # equal priors split it evenly, and the solve keeps the measured sum.
    la, ltrue = @test_logs undetermined match_mode = :any solve(2, [(1, 2)])
    # The split lies along a direction only the wide prior measures, which the
    # solve resolves only to its tolerance in that direction's standard error.
    @test la[:, :, 1] ≈ la[:, :, 2] atol = 1.0e-6
    for f in 1:2
        @test centered(la[f, :, 1] .+ la[f, :, 2]) ≈ centered(ltrue[1, f, :] .+ ltrue[2, f, :]) atol = 1.0e-8
    end
    # A chain has no odd loop either; a triangle does.
    @test_logs undetermined match_mode = :any solve(3, [(1, 2), (2, 3)])
    @test_logs solve(3, [(1, 2), (1, 3), (2, 3)])
    # Whatever the amplitude prior, it is what sets the split.
    @test_logs undetermined match_mode = :any solve(2, [(1, 2)]; amp_prior = CAL.OUPrior(; scale = 5.0e6, σ = 0.3))
end

# ── The profiled objective's derivatives and the step's preconditioner ───────

# A toy joint problem over `nant` stations of one feed and `C` channels, every
# baseline in `nscan` scans, each (scan, baseline) with its own source term:
# the cells, the station-gain pairs, the true gains and the number of gains.
function _toy_bandpass_cells(rng; nscan = 2, nant = 3, C = 6, noise = 0.3)
    bl = [(a, b) for a in 1:nant for b in (a + 1):nant]
    gain(a, c) = (a - 1) * C + c
    edge = zeros(Int32, nscan, length(bl), 1, C)
    gain_pairs = Tuple{Int, Int}[]
    for c in 1:C, (k, (a, b)) in pairs(bl)
        push!(gain_pairs, (gain(a, c), gain(b, c)))
        edge[:, k, 1, c] .= length(gain_pairs)
    end
    gtrue = exp.(complex.(0.2 .* randn(rng, nant * C), randn(rng, nant * C)))
    w = 1 .+ rand(rng, size(edge)...)
    r = zeros(ComplexF64, size(edge))
    for I in CartesianIndices(edge)
        na, nb = gain_pairs[edge[I]]
        S = (1 + I[1]) * cis(I[2])
        r[I] = w[I] * (gtrue[na] * S * conj(gtrue[nb]) + noise * complex(randn(rng), randn(rng)) / sqrt(w[I]))
    end
    return FP.BandpassCells(r, w, edge, gain_pairs; nchunks = 2), gain_pairs, gtrue, nant * C
end

# The Hessian of `f` at `x` by central differences.
function _fd_hessian(f, x; h = 1.0e-4)
    n = length(x)
    e(i) = h .* (1:n .== i)
    return [(f(x + e(i) + e(j)) - f(x + e(i) - e(j)) - f(x - e(i) + e(j)) + f(x - e(i) - e(j))) / (4h^2) for i in 1:n, j in 1:n]
end

@testset "Bandpass: the profiled χ²'s gradient and Hessian" begin
    rng = MersenneTwister(11)
    ex = FP.SerialScheduler()
    cells, gain_pairs, g0, n = _toy_bandpass_cells(rng)
    ℓ0 = real.(log.(g0)) .+ 0.05 .* randn(rng, n)
    φ0 = angle.(g0) .+ 0.1 .* randn(rng, n)
    g = exp.(complex.(ℓ0, φ0))
    S = FP._source!(zeros(ComplexF64, length(cells.ranges)), g, cells, ex)
    lin = FP._linearize(g, S, cells, n, length(gain_pairs), ex)
    H, pos = FP._curvature_pattern(gain_pairs, n)
    unit(i) = Float64.((1:n) .== i)
    for (σ, obs) in ((-1, :phase), (1, :logamp))
        f(x) = FP._profiled_chi2(σ < 0 ? exp.(complex.(ℓ0, x)) : exp.(complex.(x, φ0)), cells, ex) / 2
        x0 = σ < 0 ? φ0 : ℓ0
        held = Matrix(FP._data_curvature!(copy(H), pos, lin.q, σ))
        exact = held + reduce(hcat, [FP._profile_product!(zeros(n), unit(i), g, S, cells, σ, ex) for i in 1:n])
        fd = _fd_hessian(f, x0)
        @test maximum(abs, exact - fd) < 1.0e-5 * maximum(abs, fd)
        # The gradient is the held-S gradient at the optimal S.
        b = σ < 0 ? imag.(lin.c) : real.(lin.c)
        fdg = [(f(x0 + 1.0e-6 * unit(i)) - f(x0 - 1.0e-6 * unit(i))) / 2.0e-6 for i in 1:n]
        @test -b ≈ fdg rtol = 1.0e-6
    end

    # Both observables at once, with the cross block between them.
    fj(v) = FP._profiled_chi2(exp.(complex.(v[(n + 1):end], v[1:n])), cells, ex) / 2
    held = cat(Matrix(FP._data_curvature!(copy(H), pos, lin.q, -1)), Matrix(FP._data_curvature!(copy(H), pos, lin.q, 1)); dims = (1, 2))
    joint = reduce(hcat, [vcat(FP._joint_profile_product!(zeros(n), zeros(n), e[1:n], e[(n + 1):end], g, S, cells, ex)...) for e in (Float64.((1:2n) .== i) for i in 1:2n)])
    fd = _fd_hessian(fj, [φ0; ℓ0])
    @test maximum(abs, held + joint - fd) < 1.0e-5 * maximum(abs, fd)
    @test maximum(abs, fd[1:n, (n + 1):end]) > 1.0e-3 * maximum(abs, fd)

    # Where the model fits the data, the exact Hessian is the Gauss–Newton one:
    # the held-S curvature less its Schur complement in S.
    cells, gain_pairs, g0, n = _toy_bandpass_cells(rng; noise = 0.0)
    S = FP._source!(zeros(ComplexF64, length(cells.ranges)), g0, cells, ex)
    for σ in (-1, 1)
        x = randn(rng, n)
        @test FP._profile_product!(zeros(n), x, g0, S, cells, σ, ex) ≈ FP._gauss_newton_product!(zeros(n), x, g0, S, cells, σ, ex)
    end
    # The passes do not depend on how the J's are chunked.
    one_chunk = FP.BandpassCells(cells.r, cells.w, cells.e, cells.na, cells.nb, cells.ranges, [1:length(cells.ranges)])
    x = randn(rng, n)
    @test FP._profile_product!(zeros(n), x, g0, S, cells, 1, ex) ≈ FP._profile_product!(zeros(n), x, g0, S, one_chunk, 1, ex)
end

@testset "Bandpass: the balancing preconditioner" begin
    rng = MersenneTwister(12)
    n, m = 30, 3
    B = FP.sparse(randn(rng, n, n) .* (rand(rng, n, n) .< 0.15))
    K = B * B' + 4I
    A = Matrix(K) - (x -> x * x')(randn(rng, n, 2) ./ sqrt(n))
    Z = FP.sparse(repeat(1:m; inner = 4), 1:(4m), ones(4m), m, n)'
    U = A * Z
    F = FP._factor(K, collect(1.0:n))
    P = FP.ProfilePreconditioner(F, FP.sparse(Z), U)
    E = Matrix(Z' * U)
    Q = I - Z * (E \ U')
    Minv = Q * inv(Matrix(K)) * Q' + Z * (E \ Z')
    x = randn(rng, n)
    @test ldiv!(zeros(n), P, x) ≈ Minv * x
    # It is A's inverse on the coarse space.
    @test ldiv!(zeros(n), P, A * Z[:, 1]) ≈ Z[:, 1]
    cov = FP._covariance(P, K)
    pattern = [(i, j) for j in 1:n for i in FP.rowvals(K)[FP.nzrange(K, j)]]
    @test maximum(((a, b),) -> abs(cov(a, b) - Minv[a, b]), pattern) < 1.0e-10
end

# A synthetic set whose stations `moving` add `dev[a][s, c]` to their phase on
# scan `s`, both feeds, beside a static bandpass for every station, under
# per-baseline source structure.
function _scan_term_ps(dev; nant = 4, nchan = 32, nscans = 8, seed = 7)
    rng = MersenneTwister(seed)
    static = 0.3 .* randn(rng, nant, 2, nchan)
    bp = zeros(nant, 2, nchan, nscans)
    for s in 1:nscans
        bp[:, :, :, s] .= static
        for (a, d) in dev
            bp[a, :, :, s] .+= reshape(d[s, :], 1, nchan)
        end
    end
    ps, truth = _build_fringe_ps(; nant, nspw = 1, nchan, ntime = 6, nscans, antenna_gains = false, bandpass = bp, noise = 0.3)
    nbl, npol = length(truth.bl_pairs), length(truth.polarizations)
    _scale_sources!(ps, 0.5 .+ rand(rng, nscans, nbl, npol), 2π .* rand(rng, nscans, nbl, npol))
    return ps, truth
end

# The rms phase about the band mean of each scan's averaged spectrum, parallel
# hands, on the baselines of station `a` and on the rest.
function _scan_spectrum_rms(sol, ps, truth, a)
    c = calibrate(sol, ps; flag_bad = false, apply_flags = false)
    on, off = Float64[], Float64[]
    for ms in values(c)
        V = parent(ms[:visibility])
        for bl in axes(V, 3), p in (1, 4)
            v = vec(sum(V[p, :, bl, :]; dims = 2))
            push!(a in truth.bl_pairs[bl] ? on : off, sqrt(mean(abs2, angle.(v ./ (sum(v) / length(v))))))
        end
    end
    return mean(on), mean(off)
end

# Each scan's phase of station `a`, feed 1, over the channels, with the
# per-channel mean over scans and each scan's mean over channels removed.
function _double_centered(X)
    return X .- mean(X; dims = 2) .- mean(X; dims = 1) .+ mean(X)
end
function _scan_phases(sol, ps, a, nscans)
    geom = CAL.DataGeometry(ps)
    G = parent(gains(sol))
    first_time(s) = findfirst(==(findfirst(==(string(s)), geom.scan_names)), geom.scan_of_time)
    return reduce(vcat, [angle.(G[:, first_time(s), a, 1] ./ G[:, first_time(1), a, 1])' for s in 1:nscans])
end

# Zero the weights of every baseline of station `a` on scan `scan`.
function _drop_station!(ps, truth, a, scan)
    for ms in values(ps)
        parse(Int, only(unique(ms[:scan_name]))) == scan || continue
        W = DimensionalData.modify(Array, ms[:weight])
        for (bi, pair) in pairs(truth.bl_pairs)
            a in pair && (view(W, BaselineID(bi)) .= 0)
        end
        ms[:weight] = W
    end
    return ps
end

@testset "JointSmoother: per-scan terms beside the shape" begin
    nchan, nscans = 32, 8
    ou = OUPrior(; scale = LogNormal(log(8.0e6), 1.0), σ = LogNormal(log(0.1), 1.5))
    shape = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = (; Frequency = ou))
    base = GainModel(; phase = (; bandpass = shape), logamp = (; bandpass = shape))
    gauge = PinAntenna(1)

    @testset "a level and a delay per group of channels on one station" begin
        groups = [1:1, 2:13, 14:25, 26:32]
        rng = MersenneTwister(11)
        dev = zeros(nscans, nchan)
        for s in 1:nscans, g in groups
            dev[s, g] .= 0.2 * randn(rng) .+ 0.03 * randn(rng) .* (g .- mean(g))
        end
        ps, truth = _scan_term_ps(Dict(2 => dev); nchan, nscans)
        fg = FreqGroups(groups)
        model = with_antenna(
            base, "A2"; phase = (;
                bandpass = shape,
                level = GainComponent(ConstantTerm(); Ti = PerScan(), Frequency = fg, Feed = SharedFeeds(), prior = IIDPrior(1.0)),
                delay = GainComponent(Delay(); Ti = PerScan(), Frequency = fg, Feed = SharedFeeds(), prior = IIDPrior(1.0e-8)),
            ),
        )
        without = _scan_spectrum_rms(fit(Bandpass(; model = base, gauge), ps), ps, truth, 2)
        sol = fit(Bandpass(; model, gauge), ps)
        with = _scan_spectrum_rms(sol, ps, truth, 2)
        # The A2 baselines come down to the noise the others show.
        @test without[1] > 3 * without[2]
        @test with[1] < 1.2 * with[2]
        err = _double_centered(_scan_phases(sol, ps, 2, nscans)) .- _double_centered(dev .- dev[1:1, :])
        @test sqrt(mean(abs2, err)) < 0.05 * sqrt(mean(abs2, _double_centered(dev)))
    end

    @testset "a level per group of channels and feed" begin
        groups = [1:16, 17:32]
        rng = MersenneTwister(13)
        dev = zeros(nscans, nchan)
        for s in 1:nscans, g in groups
            dev[s, g] .= 0.3 * randn(rng)
        end
        ps, truth = _scan_term_ps(Dict(2 => dev); nchan, nscans, seed = 9)
        level = GainComponent(ConstantTerm(); Ti = PerScan(), Frequency = FreqGroups(groups), Feed = PerFeed(), prior = IIDPrior(1.0))
        sol = fit(Bandpass(; model = with_antenna(base, "A2"; phase = (; bandpass = shape, level)), gauge), ps)
        err = _double_centered(_scan_phases(sol, ps, 2, nscans)) .- _double_centered(dev .- dev[1:1, :])
        @test sqrt(mean(abs2, err)) < 0.05 * sqrt(mean(abs2, _double_centered(dev)))
    end

    @testset "a band-wide delay per scan on two stations" begin
        rng = MersenneTwister(12)
        freqs = 230.0e9 .+ (0:(nchan - 1)) .* 2.0e6
        ramp = 2π .* (freqs .- sum(freqs) / nchan)
        dev = Dict(a => reduce(vcat, [(1.0e-9 * randn(rng) .* ramp)' for _ in 1:nscans]) for a in (2, 3))
        ps, truth = _scan_term_ps(dev; nchan, nscans, seed = 8)
        delay = GainComponent(Delay(); Ti = PerScan(), Feed = SharedFeeds(), prior = IIDPrior(1.0e-8))
        model = with_antenna(with_antenna(base, "A2"; phase = (; bandpass = shape, delay)), "A3"; phase = (; bandpass = shape, delay))
        sol = fit(Bandpass(; model, gauge), ps)
        for a in (2, 3)
            err = _double_centered(_scan_phases(sol, ps, a, nscans)) .- _double_centered(dev[a] .- dev[a][1:1, :])
            @test sqrt(mean(abs2, err)) < 0.05 * sqrt(mean(abs2, _double_centered(dev[a])))
        end
    end

    @testset "a band-wide delay per scan on every station" begin
        nant = 4
        rng = MersenneTwister(14)
        freqs = 230.0e9 .+ (0:(nchan - 1)) .* 2.0e6
        ramp = 2π .* (freqs .- sum(freqs) / nchan)
        dev = Dict(a => reduce(vcat, [(1.0e-9 * randn(rng) .* ramp)' for _ in 1:nscans]) for a in 1:nant)
        ps, truth = _scan_term_ps(dev; nant, nchan, nscans, seed = 10)
        delay = GainComponent(Delay(); Ti = PerScan(), Feed = SharedFeeds(), prior = IIDPrior(1.0e-8))
        model = GainModel(; phase = (; bandpass = shape, delay), logamp = (; bandpass = shape))
        # A delay common to every station of a scan is not seen by the data,
        # so the phases are compared relative to station 1 on the scans it
        # observes.
        function check(sol, scans)
            rel(a) = (_scan_phases(sol, ps, a, nscans) .- _scan_phases(sol, ps, 1, nscans))[scans, :]
            for a in 2:nant
                truth_rel = ((dev[a] .- dev[a][1:1, :]) .- (dev[1] .- dev[1][1:1, :]))[scans, :]
                err = _double_centered(rel(a)) .- _double_centered(truth_rel)
                @test sqrt(mean(abs2, err)) < 0.05 * sqrt(mean(abs2, _double_centered(truth_rel)))
            end
        end
        for gauge in (PinAntenna(1), ZeroSumPhase())
            check(fit(Bandpass(; model, gauge), ps), 1:nscans)
        end
        # The gauge acts on the shape, which spans the scans, so a scan
        # without the reference needs no reference of its own.
        _drop_station!(ps, truth, 1, 3)
        for gauge in (PinAntenna(1), ZeroSumPhase())
            check(fit(Bandpass(; model, gauge), ps), setdiff(1:nscans, 3))
        end
    end

    @testset "a per-scan delay alone" begin
        ps, _ = _scan_term_ps(Dict{Int, Matrix{Float64}}(); nant = 3, nchan = 8, nscans = 2)
        delay = GainComponent(Delay(); Ti = PerScan(), Feed = SharedFeeds(), prior = IIDPrior(1.0e-8))
        sol = fit(Bandpass(; model = GainModel(; phase = (; delay), logamp = (; bandpass = shape)), gauge), ps)
        @test all(isfinite, parent(gains(sol)))
    end
end

@testset "_align_component_phases! removes each free component's rotation" begin
    nant, nap = 3, 10
    rng = MersenneTwister(7)
    truth = cumsum(0.2 .* randn(rng, nant, nap); dims = 2)
    gain(a, t) = (t - 1) * nant + a
    φ, compid = zeros(nant * nap), zeros(Int, nant * nap)
    for t in 1:nap
        rot = 2π * rand(rng)
        for a in 1:nant
            φ[gain(a, t)] = truth[a, t] + rot
            compid[gain(a, t)] = t
        end
    end
    tracks = [(; gains = [gain(a, t) for t in 1:nap]) for a in 1:nant]
    free = trues(length(φ))
    free[gain(1, 5)] = false
    held = [φ[gain(a, 5)] for a in 1:nant]
    FP._align_component_phases!(φ, compid, free, ones(length(φ)), tracks)
    @test [φ[gain(a, 5)] for a in 1:nant] == held
    for t in setdiff(2:nap, 5)
        @test abs(angle(sum(cis(φ[gain(a, t - 1)] - φ[gain(a, t)]) for a in 1:nant))) < 1.0e-12
    end
end

@testset "solve_joint_gains! along Ti: a phase prior under ZeroSumPhase" begin
    nant, nap = 4, 30
    names = ["A$i" for i in 1:nant]
    geom = CAL.DataGeometry(;
        nfeed = 2, times = collect(0.0:(nap - 1)), scan_of_time = ones(Int, nap),
        channel_freqs = [230.0e9], scan_names = ["No0001"], spw_names = ["A"], antennas = names,
    )
    comp = GainComponent(ConstantTerm(); Ti = CAL.PerIntegration(), Feed = SharedFeeds(), prior = FP.default_adhoc_prior())
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; adhoc = comp)), names, geom)
    θ = zeros(l.nθ)
    pb = CAL.antenna_blocks(l, θ, :phase, :adhoc)
    rng = MersenneTwister(20261107)
    # Steps of order a radian per AP: the tracks wander over several turns.
    φ = cumsum(0.8 .* randn(rng, nant, nap); dims = 2)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (1, 2), (2, 1), (2, 2)]
    S = [(1 + rand(rng)) * cis(2π * rand(rng)) for _ in bl_pairs, _ in feeds]
    ax = (FP.Scan(1:1), FP._antenna_pair_dim([(names[a], names[b]) for (a, b) in bl_pairs]), FP.FeedPair(feeds), Ti(1:nap))
    # Weighted sums `Σ w·v` at weight 10⁴ per (baseline, product, AP).
    w = fill(1.0e4, ax)
    r = zeros(ComplexF64, ax)
    for t in 1:nap, p in eachindex(feeds), (bi, (a, b)) in pairs(bl_pairs)
        r[1, bi, p, t] = w[1, bi, p, t] * cis(φ[a, t]) * S[bi, p] * cis(-φ[b, t])
    end
    FP.solve_joint_gains!(
        (; phase = [pb], logamp = []), (; r, w), geom;
        cover = [(t, 1) for _ in 1:1, t in 1:nap], gauge = ZeroSumPhase(), max_iterations = 200, tolerance = 1.0e-8,
    )
    est = [only(pb).θ[1, 1, 1, t, a] for a in 1:nant, t in 1:nap]
    # The data fix each baseline's phase up to a constant the source term takes.
    for a in 2:nant
        d = (est[a, :] .- est[1, :]) .- (φ[a, :] .- φ[1, :])
        @test maximum(abs, rem2pi.(d .- angle(sum(cis, d)), RoundNearest)) < 1.0e-3
    end
end
