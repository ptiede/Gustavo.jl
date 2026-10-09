# ── Bandpass(smoother = JointSmoother()): solve_joint_bandpass! ──────────────
#
# `PerTrackSmoother` (the default, closure-based path)
# assumes a baseline's source term cancels out of the per-channel
# phase-difference/log-amp-sum closure — true only for an unresolved,
# unpolarized calibrator. This exercises the alternative: an unresolved point
# source cannot distinguish the two paths, so the synthetic visibilities here
# are post-multiplied by an independent random complex factor per
# (scan, baseline, polarization) — a stand-in for a resolved/polarized
# calibrator's per-baseline structure — and the test checks that
# `solve_joint_bandpass!` still recovers the injected station bandpass.
#
# A single station's ABSOLUTE per-channel bandpass is never observable from
# baseline data alone (only differences between stations are), so both paths'
# recovered bandpass is relative to the gauge's reference — truth is gauged the same way
# before comparing. The injected factor is frequency-FLAT (matching "constant
# per scan"), which the closure path's per-AP de-rotation heuristic already
# absorbs incidentally for PHASE (it isn't targeted at this, but a band-flat
# offset is exactly what de-rotating each AP to its own band-average removes),
# so the phase gate here only asks that the joint solve be no worse. The
# injected AMPLITUDE has no such accidental cover — `log|V| = la_a + la_b` has
# no baseline term at all — so that is where the joint solve's structural
# advantage is unambiguous.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# The bandpass step's `obs` (`:phase` or `:logamp`) bandpass parameters over
# `(param, Feed, Frequency, Ti, AntennaName)`.
_jb_leaf(sol, obs) = parent(only(sol[:bandpass, obs, :bandpass].components).params)

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

    fm = default_fringe_terms()
    # `ref_ant` indexes the truth arrays below; `gauge` is what the solve takes.
    ref_ant = 1
    gauge = PinAntenna(ref_ant)     # the gauge every test here passes to `fit`
    _, sol_closure = _fit_chain(
        (BaselineFringeFit(; model = fm, gauge), Bandpass(; smoother = FP.PerTrackSmoother(), gauge)),
        ps; exec = ExecutionConfig(),
    )
    # This synthetic truth is i.i.d. RANDOM per channel (a deliberately hard,
    # uncorrelated-neighbor case for the ALS to track) — noticeably slower to
    # converge than JointSmoother's own defaults, hence the raised iteration
    # count/tolerance.
    _, sol_joint = _fit_chain(
        (
            BaselineFringeFit(; model = fm, gauge),
            Bandpass(; smoother = FP.JointSmoother(), gauge),
        ),
        ps; exec = ExecutionConfig(),
    )

    bp_leaves(sol) = (_jb_leaf(sol, :phase), _jb_leaf(sol, :logamp))
    pleaf_c, aleaf_c = bp_leaves(sol_closure)
    pleaf_j, aleaf_j = bp_leaves(sol_joint)

    wrapped_err(x, y) = maximum(abs, rem2pi.(x .- y, RoundNearest))
    # PHASE is recovered relative to the reference's feed 1 — the joint tier pins that
    # node's phase at every channel, and the closure tier references its solve to
    # it — so truth is gauged the same way before comparing, then to its own
    # circular mean.
    gauge_phase_rel(x, xref) = rem2pi.((x .- xref) .- angle(sum(cis, x .- xref)), RoundNearest)
    # AMPLITUDE is NOT referenced to any antenna: the closure tier's SUM incidence
    # is full rank and the joint tier pins only phase, so each station's own
    # passband shape is identifiable and only its band mean is unobservable.
    # Subtracting the reference's track here would inject a spurious error into
    # both arms.
    gauge_amp(x) = x .- sum(x) / length(x)

    max_phase_err_closure = 0.0
    max_phase_err_joint = 0.0
    max_amp_err_closure = 0.0
    max_amp_err_joint = 0.0
    for a in 1:nant, f in 1:2
        btrue = gauge_phase_rel(bp_true[a, f, :], bp_true[ref_ant, 1, :])
        abtrue = gauge_amp(abp_true[a, f, :])
        bc = Float64[pleaf_c[1, f, c, 1, a] for c in 1:nglob]
        bj = Float64[pleaf_j[1, f, c, 1, a] for c in 1:nglob]
        ac = Float64[aleaf_c[1, f, c, 1, a] for c in 1:nglob]
        aj = Float64[aleaf_j[1, f, c, 1, a] for c in 1:nglob]
        max_phase_err_closure = max(max_phase_err_closure, wrapped_err(bc, btrue))
        max_phase_err_joint = max(max_phase_err_joint, wrapped_err(bj, btrue))
        max_amp_err_closure = max(max_amp_err_closure, maximum(abs, ac .- abtrue))
        max_amp_err_joint = max(max_amp_err_joint, maximum(abs, aj .- abtrue))
    end

    # Phase: the joint solve must not be meaningfully worse than the closure
    # (both recover essentially the same thing here — see the header note).
    @test max_phase_err_joint < max_phase_err_closure + 0.02
    # Amplitude: log|V| = la_a + la_b has no baseline term to absorb the
    # injected per-baseline/pol factor, so the closure solve is measurably
    # biased by it; the joint solve's explicit source coherence absorbs it.
    @test max_amp_err_joint < 0.3
    @test max_amp_err_joint < 0.7 * max_amp_err_closure
end

@testset "JointSmoother: the component priors act INSIDE the ALS" begin
    # The gain update fits each (station, feed) track under its component's prior
    # instead of solving every channel independently. No prior on either
    # observable IS that independent per-channel solve; a stiff second-order walk
    # must instead leave only its null space — a straight line in frequency.
    nant, nspw, nchan, ntime, nscans = 4, 1, 8, 4, 3
    nglob = nspw * nchan
    rng = MersenneTwister(0x101A)
    bp_true = 0.4 .* randn(rng, nant, 2, nglob)
    abp_true = 0.15 .* randn(rng, nant, 2, nglob)
    ps, _ = _build_fringe_ps(;
        nant, nspw, nchan, ntime, nscans,
        bandpass = bp_true, amp_bandpass = abp_true, seed = 21,
    )
    fm = default_fringe_terms()
    bpc(prior) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior)
    runbp(sm; prior = nothing) = last(
        _fit_chain(
            (
                BaselineFringeFit(; model = fm, gauge = PinAntenna(1)),
                Bandpass(; model = CAL.GainModel(; phase = (; bandpass = bpc(prior)), logamp = (; bandpass = bpc(prior))), smoother = sm, gauge = PinAntenna(1)),
            ),
            ps; exec = ExecutionConfig(),
        )
    )

    s_free = runbp(FP.JointSmoother())
    @test_logs (:warn, r"did not converge in 2 sweeps") match_mode = :any runbp(FP.JointSmoother(max_iterations = 2))

    # σ = 1e-6 rad per channel (2 MHz) as rad/Hz^(3/2).
    stiff = CAL.RandomWalkPrior(; order = 2, σ = 1.0e-6 * sqrt(3 / (2 * 2.0e6^3)))
    # A prior the gains cannot follow exactly must still converge: the band
    # level that trades with the source coherence is held fixed in the sweep.
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
    # Per-(scan, baseline, pol) structure: what the joint tier exists to absorb,
    # and what biases the closure.
    nbl = length(truth.bl_pairs); npol = length(truth.polarizations)
    samp = 0.4 .+ 1.6 .* rand(rng, nscans, nbl, npol)
    sph = (2 .* rand(rng, nscans, nbl, npol) .- 1) .* pi
    _scale_sources!(ps, samp, sph)

    fm = default_fringe_terms()
    runbp(sm) = last(
        _fit_chain(
            (BaselineFringeFit(; model = fm, gauge = PinAntenna(1)), Bandpass(; smoother = sm, gauge = PinAntenna(1))),
            ps; exec = ExecutionConfig(),
        )
    )
    s_joint = runbp(FP.JointSmoother())
    s_closure = runbp(FP.PerTrackSmoother())

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

    # ...and with the reference free to carry its own shape, the joint solve's
    # amplitude beats the closure it is meant to improve on under this structure.
    amprms(s) = sqrt(
        sum(sum(abs2, gauge(la(s, a, f)) .- gauge(abp_true[a, f, :])) for a in 1:nant, f in 1:2) /
            (2 * nant * nglob),
    )
    @test amprms(s_joint) < amprms(s_closure)
end

# ── The per-station time-segment axis inside one ALS ─────────────────────────
#
# The gain arrays carry a time-segment axis and the scans that couple through a
# shared (station, segment) node are solved in one alternating run, so a station
# whose bandpass breaks mid-track can sit beside stations held over the whole of
# it. These fixtures drive `solve_joint_bandpass!` directly on synthetic
# accumulators — `rl = w·g_a·S·conj(g_b)`, exactly the model the ALS inverts —
# because the segment table is what is under test, not the accumulation.

# A four-scan, one-time-sample-per-scan geometry of `nant` stations `A1, A2, …`
# whose second half is a separate instrument segment.
function _seg_geometry(nchan; nant = 4)
    return CAL.DataGeometry(; nfeed = 2,
        times = collect(0.0:3.0), scan_of_time = collect(1:4),
        channel_freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6),
        spw_of_chan = ones(Int, nchan),
        scan_names = ["No00$i" for i in 1:4], spw_names = ["A"],
        stations = ["A$i" for i in 1:nant],
    )
end

# Per-scan accumulators for `g[ant, feed, segment, channel]` observed through
# `S[scan, baseline, pol]`, with `tseg[ant, scan]` naming each station's segment,
# labeled by `geom`'s station names and channel frequencies.
function _joint_scan_accumulators(g, S, tseg, bl_pairs, feeds, geom)
    names = geom.stations
    return map(axes(S, 1)) do si
        ax = (
            FP._station_pair_dim([(names[a], names[b]) for (a, b) in bl_pairs]),
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

# The pinned gain slots of a joint solve's phase gauge, as `(block, (ant, feed,
# channel, time segment))` positions; empty for a gauge imposed in the sweep.
function _joint_pins(geom, blocks, results, tseg, gauge)
    nant = length(geom.stations)
    fseg, _ = FP._station_freq_segments(blocks, nant)
    data = (; ends = FP._cell_nodes(first(results).rl, geom.stations, PerFeed()))
    layout = (; loc = FP._block_locations(blocks, nant), tseg, fseg)
    gains = [FP._block_gains(b, geom, ComplexF64) for b in blocks]
    FP._joint_bandpass_gauge!(gains, data, layout, blocks, gauge)
    return Set((k, Tuple(I)) for (k, st) in pairs(gains) for I in findall(parent(st.pinned)))
end

@testset "JointSmoother: each station's own time segments in one ALS" begin
    nant, nchan = 4, 6
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan)
    bp(ti) = GainComponent(ConstantTerm(); Ti = ti, Frequency = ChannelBlocks(1), Feed = PerFeed())
    breakmodel(ti) = CAL.GainModel(;
        phase = (; bandpass = bp(ti)), logamp = (; bandpass = bp(ti)),
    )
    layout(ti) = CAL.plan_parameters(breakmodel(ti), anames, geom)
    setup(l) = (;
        layout = l,
        paths = (;
            phase = FP._bandpass_paths(l.plantree, :phase),
            logamp = FP._bandpass_paths(l.plantree, :logamp),
        ),
    )

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
        plan = only(FP.bandpass_blocks(setup(l), zeros(l.nθ), :phase)).plan
        results = _joint_scan_accumulators(
            gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom,
        )
        blocks = FP.bandpass_blocks(setup(l), zeros(l.nθ), :phase)
        tseg = FP._station_time_segments(blocks, results, nant)
        # One block spans every station, so every row is that block's own table.
        @test all(tseg[a, :] == [1, 1, 2, 2] for a in 1:nant)
        @test FP._joint_scan_groups(tseg) ==
            filter(!isempty, FP.time_segment_scans(plan, results))

        # …and the merged driver's θ is the per-segment loop's, exactly.
        θ_merged = zeros(l.nθ)
        merged_blocks = FP.bandpass_blocks(setup(l), θ_merged, :phase)
        for idx in FP._joint_scan_groups(tseg)
            FP.solve_joint_bandpass!(
                θ_merged, results[idx], geom,
                merged_blocks, merged_blocks;
                gauge = PinAntenna(2), max_iterations = 200, tolerance = 1.0e-12,
                tseg = view(tseg, :, idx),
            )
        end
        θ_loop = zeros(l.nθ)
        loop_blocks = FP.bandpass_blocks(setup(l), θ_loop, :phase)
        for (ts, idx) in pairs(FP.time_segment_scans(plan, results))
            isempty(idx) && continue
            FP.solve_joint_bandpass!(
                θ_loop, results[idx], geom,
                loop_blocks, loop_blocks;
                gauge = PinAntenna(2), max_iterations = 200, tolerance = 1.0e-12,
                tseg = fill(ts, nant, length(idx)),
            )
        end
        @test θ_merged == θ_loop
    end

    @testset "one station breaks and the rest span the track" begin
        l = layout(InstrumentScans([1.5]))
        s = setup(l)
        θ = zeros(l.nθ)
        phase_blocks = FP.bandpass_blocks(s, θ, :phase)
        amp_blocks = FP.bandpass_blocks(s, θ, :logamp)
        phase_plan = only(phase_blocks).plan
        amp_plan = only(amp_blocks).plan
        # Station 1 alone is solved per half; the others carry one segment, so
        # their scans bridge the break and the whole track is one ALS.
        tseg = fill(1, nant, 4)
        tseg[1, :] = [1, 1, 2, 2]
        results = _joint_scan_accumulators(gtrue, Strue, tseg, bl_pairs, feeds, geom)
        @test FP._joint_scan_groups(tseg) == [[1, 2, 3, 4]]

        phase_status = [FP._block_status_array(b, geom) for b in phase_blocks]
        amp_status = [FP._block_status_array(b, geom) for b in amp_blocks]
        FP.solve_joint_bandpass!(
            θ, results, geom, phase_blocks, amp_blocks;
            gauge = PinAntenna(2), max_iterations = 200, tolerance = 1.0e-13,
            tseg, phase_status, amp_status,
        )

        pleaf = CAL._component_leaf(phase_plan, θ)
        aleaf = CAL._component_leaf(amp_plan, θ)
        demean(v) = v .- sum(v) / length(v)
        cdemean(v) = rem2pi.(v .- angle(sum(cis, v)), RoundNearest)
        # Phase is recovered relative to the pinned station's track, then to its
        # own circular band mean; amplitude carries no reference, only the
        # arithmetic band mean.
        want_phase(a, f, ts) = cdemean(
            angle.(gtrue[a, f, ts, :]) .- angle.(gtrue[2, f, 1, :]),
        )
        want_amp(a, f, ts) = demean(log.(abs.(gtrue[a, f, ts, :])))

        for f in 1:2
            # The broken station's two halves are each recovered, and they differ.
            for ts in 1:2
                @test pleaf[1, f, :, ts, 1] ≈ want_phase(1, f, ts) atol = 1.0e-8
                @test aleaf[1, f, :, ts, 1] ≈ want_amp(1, f, ts) atol = 1.0e-8
            end
            @test !isapprox(pleaf[1, f, :, 1, 1], pleaf[1, f, :, 2, 1]; atol = 1.0e-3)
            # Every other station holds ONE segment: its second slot is never
            # solved, so θ still carries the zero it started at.
            for a in 2:nant
                @test pleaf[1, f, :, 1, a] ≈ want_phase(a, f, 1) atol = 1.0e-8
                @test aleaf[1, f, :, 1, a] ≈ want_amp(a, f, 1) atol = 1.0e-8
                @test all(iszero, pleaf[1, f, :, 2, a])
                @test all(iszero, aleaf[1, f, :, 2, a])
                # …and the status array leaves that slot at its initial code.
                @test only(phase_status)[a, f, 1, 2] == FP._BP_TRACK_NODATA
            end
            @test only(phase_status)[1, f, 1, 2] == FP._BP_TRACK_SOLVED
        end
    end

    @testset "a heterogeneous model writes through each station's own block" begin
        # Station 1 alone carries the break; the rest hold one gain over the
        # track, so the two signature groups become two blocks with different
        # `:AntennaName` axes and different time segmentations, and the solve has to
        # place each station in the block that holds it.
        het_model = CAL.GainModel(;
            phase = (; bandpass = bp(CAL.GlobalTime())),
            logamp = (; bandpass = bp(CAL.GlobalTime())),
            stations = (
                A1 = (;
                    phase = (; bandpass = bp(InstrumentScans([1.5]))),
                    logamp = (; bandpass = bp(InstrumentScans([1.5]))),
                ),
            ),
        )
        lh = CAL.plan_parameters(het_model, anames, geom)
        sh = setup(lh)
        θ = zeros(lh.nθ)
        phase_blocks = FP.bandpass_blocks(sh, θ, :phase)
        amp_blocks = FP.bandpass_blocks(sh, θ, :logamp)
        @test [b.stations for b in phase_blocks] == [[1], [2, 3, 4]]
        # The parameter saving is real: the stations that hold one gain carry one
        # time segment, not the union's two.
        @test phase_blocks[1].plan.shape[4] == 2
        @test phase_blocks[2].plan.shape[4] == 1

        tsg = fill(1, nant, 4)
        tsg[1, :] = [1, 1, 2, 2]
        results = _joint_scan_accumulators(gtrue, Strue, tsg, bl_pairs, feeds, geom)
        @test FP._station_time_segments(phase_blocks, results, nant) == tsg
        @test FP._joint_scan_groups(tsg) == [[1, 2, 3, 4]]

        # Each block's status array spans its own stations and time segments,
        # labeled as its θ leaf is.
        phase_status = [FP._block_status_array(b, geom) for b in phase_blocks]
        amp_status = [FP._block_status_array(b, geom) for b in amp_blocks]
        for (st, b) in zip(phase_status, phase_blocks)
            @test lookup(st, FP.AntennaName) == geom.stations[b.stations]
            @test size(st, Ti) == b.plan.shape[4]
            ref = CAL._time_segment_lookup(b.plan, geom)
            @test collect(lookup(st, Ti)) == collect(ref)
            @test val(DimensionalData.Lookups.span(lookup(st, Ti))) == val(DimensionalData.Lookups.span(ref))
            @test all(==(FP._BP_TRACK_NODATA), st)
        end

        FP.solve_joint_bandpass!(
            θ, results, geom, phase_blocks, amp_blocks;
            gauge = PinAntenna(2), max_iterations = 200, tolerance = 1.0e-13,
            tseg = tsg, phase_status, amp_status,
        )

        demean(v) = v .- sum(v) / length(v)
        cdemean(v) = rem2pi.(v .- angle(sum(cis, v)), RoundNearest)
        want_phase(a, f, ts) = cdemean(
            angle.(gtrue[a, f, ts, :]) .- angle.(gtrue[2, f, 1, :]),
        )
        want_amp(a, f, ts) = demean(log.(abs.(gtrue[a, f, ts, :])))

        for f in 1:2
            # The broken station is alone on its block's `:AntennaName` axis, and both of
            # its segments land in that block.
            for ts in 1:2
                @test phase_blocks[1].θ[1, f, :, ts, 1] ≈ want_phase(1, f, ts) atol = 1.0e-8
                @test amp_blocks[1].θ[1, f, :, ts, 1] ≈ want_amp(1, f, ts) atol = 1.0e-8
            end
            # …and each of the others lands at ITS block-local position, which is
            # not its global station index.
            for (ai, a) in pairs(phase_blocks[2].stations)
                @test phase_blocks[2].θ[1, f, :, 1, ai] ≈ want_phase(a, f, 1) atol = 1.0e-8
                @test amp_blocks[2].θ[1, f, :, 1, ai] ≈ want_amp(a, f, 1) atol = 1.0e-8
            end
            @test phase_status[1][FP.AntennaName(At("A1")), Feed(f), Ti(2)] == [FP._BP_TRACK_SOLVED]
        end
        # The report keys the blocks as `parameters` keys their leaves, and counts
        # only the tracks the blocks have: station A1's two segments and one
        # segment of each of the others.
        rep = FP.bandpass_track_report(FP._block_leaves(phase_status), FP._block_leaves(amp_status))
        @test keys(rep.phase_status) == keys(rep.amp_status) == (:g1, :g2)
        @test rep.phase_status.g2 === phase_status[2]
        @test rep.n_nodata + rep.n_solved + rep.n_flat + rep.n_declined == 2 * 2 * (2 + (nant - 1))
    end
end

# ── The phase gauge across per-station time segments ─────────────────────────
#
# The gauge graph's nodes are (station, feed, that station's own time segment)
# and its edges are the correlations, so the stations that hold one gain over the
# whole track bridge a broken station's segments and one pin covers both. The
# relative phase across such a break is then measured, not gauged away — even
# when the broken station is itself the reference.

@testset "JointSmoother: the phase gauge spans per-station time segments" begin
    nant, nchan = 4, 6
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan)
    bp = GainComponent(
        ConstantTerm(); Ti = InstrumentScans([1.5]),
        Frequency = ChannelBlocks(1), Feed = PerFeed(),
    )
    model = CAL.GainModel(; phase = (; bandpass = bp), logamp = (; bandpass = bp))
    l = CAL.plan_parameters(model, anames, geom)
    s = (;
        layout = l,
        paths = (;
            phase = FP._bandpass_paths(l.plantree, :phase),
            logamp = FP._bandpass_paths(l.plantree, :logamp),
        ),
    )

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

    @testset "one pin per connected component of the promoted graph" begin
        # Every gain is per channel, so the graph splits along the channels.
        blocks = FP.bandpass_blocks(s, zeros(l.nθ), :phase)
        results = _joint_scan_accumulators(gtrue, Strue, het, bl_pairs, feeds, geom)
        pins = _joint_pins(geom, blocks, results, het, PinAntenna(1))
        # The constant stations bridge the break, so each (feed, channel) is one
        # component and the reference is pinned in its first time segment only —
        # its second is free to carry the break it actually has.
        @test pins == Set((1, (1, f, c, 1)) for f in 1:2 for c in 1:nchan)

        # A segmentation every station shares has nothing to bridge the epochs:
        # each (feed, time segment, channel) is its own component and carries its
        # own pin.
        upins = _joint_pins(geom, blocks, results, uniform, PinAntenna(1))
        @test length(upins) == 2 * 2 * nchan
    end

    @testset "the reference station's own break is measured, not gauged away" begin
        θ = zeros(l.nθ)
        phase_blocks = FP.bandpass_blocks(s, θ, :phase)
        amp_blocks = FP.bandpass_blocks(s, θ, :logamp)
        results = _joint_scan_accumulators(gtrue, Strue, het, bl_pairs, feeds, geom)
        FP.solve_joint_bandpass!(
            θ, results, geom, phase_blocks, amp_blocks;
            gauge = PinAntenna(1), max_iterations = 200, tolerance = 1.0e-13,
            tseg = het,
        )

        pleaf = CAL._component_leaf(only(phase_blocks).plan, θ)
        cdemean(v) = rem2pi.(v .- angle(sum(cis, v)), RoundNearest)
        # Everything is measured against the pinned node — station 1's FIRST
        # segment — including station 1's second segment.
        want_phase(a, f, ts) = cdemean(
            angle.(gtrue[a, f, ts, :]) .- angle.(gtrue[1, f, 1, :]),
        )
        for f in 1:2
            @test pleaf[1, f, :, 2, 1] ≈ want_phase(1, f, 2) atol = 1.0e-8
            @test !isapprox(pleaf[1, f, :, 2, 1], pleaf[1, f, :, 1, 1]; atol = 1.0e-3)
            # The stations that span the break carry one parameter each, and it
            # is the truth both halves share — the break at station 1 leaks into
            # neither half.
            for a in 2:nant
                @test pleaf[1, f, :, 1, a] ≈ want_phase(a, f, 1) atol = 1.0e-8
            end
        end
    end
end

# ── Per-station frequency segments ───────────────────────────────────────────
#
# Each station's gain is held in its own frequency segments, and the data are
# reduced onto the cells every station's segmentation is a union of.

@testset "JointSmoother: each station's own frequency segments" begin
    nant, nchan = 4, 6
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan)
    bpf(fs; prior = nothing) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = fs, Feed = PerFeed(), prior)
    hetmodel(a1, rest; prior = nothing) = CAL.GainModel(;
        phase = (; bandpass = bpf(rest; prior)), logamp = (; bandpass = bpf(rest)),
        stations = (
            A1 = (; phase = (; bandpass = bpf(a1; prior)), logamp = (; bandpass = bpf(a1))),
        ),
    )
    setup(l) = (;
        layout = l,
        paths = (;
            phase = FP._bandpass_paths(l.plantree, :phase),
            logamp = FP._bandpass_paths(l.plantree, :logamp),
        ),
    )

    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    pol_products = [(1, 1), (2, 2)]
    feeds = collect(pol_products)

    @testset "segmentations that cut at different channels refine each other" begin
        # Station 1 in blocks of two channels, the rest in blocks of three: no
        # cut of one lands on every cut of the other, so the refinement is
        # strictly finer than either.
        l = CAL.plan_parameters(hetmodel(ChannelBlocks(2), ChannelBlocks(3)), anames, geom)
        blocks = FP.bandpass_blocks(setup(l), zeros(l.nθ), :phase)
        @test [b.stations for b in blocks] == [[1], [2, 3, 4]]
        fseg, cells = FP._station_freq_segments(blocks, nant)
        @test cells == [[1, 2], [3], [4], [5, 6]]
        @test fseg[1, :] == [1, 2, 2, 3]
        @test all(fseg[a, :] == [1, 1, 2, 2] for a in 2:nant)
        @test length(cells) > maximum(fseg[1, :])
        @test length(cells) > maximum(fseg[2, :])

        # Neither station's segments nest inside the other's, so the whole band
        # is one connected component per feed: the gauge has ONE constant to fix
        # and the pin holds one segment of station 1's three-segment track. The
        # other two are free to carry the structure they really have.
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
        @test _joint_pins(geom, blocks, results, fill(1, nant, 4), PinAntenna(1)) ==
            Set([(1, (1, 1, 1, 1)), (1, (1, 2, 1, 1))])

        θ = zeros(l.nθ)
        pb = FP.bandpass_blocks(setup(l), θ, :phase)
        ab = FP.bandpass_blocks(setup(l), θ, :logamp)
        # The misaligned refinement couples the two segmentations through cells
        # neither owns alone, which the alternating sweep works through slowly:
        # the recovery is exact, but 200 sweeps only reach 3e-8 of it.
        FP.solve_joint_bandpass!(
            θ, results, geom, pb, ab; gauge = PinAntenna(1),
            max_iterations = 1000, tolerance = 1.0e-13,
        )

        demean(v) = v .- sum(v) / length(v)
        cdemean(v) = rem2pi.(v .- angle(sum(cis, v)), RoundNearest)
        # The first channel of each of a station's own segments stands for it.
        reps(a) = a == 1 ? [1, 3, 5] : [1, 4]
        for f in 1:2, (bi, block) in pairs(pb)
            for (ai, a) in pairs(block.stations)
                truth = [gtrue[a, f, 1, c] for c in reps(a)]
                @test block.θ[1, f, :, 1, ai] ≈ cdemean(angle.(truth)) atol = 1.0e-10
                @test ab[bi].θ[1, f, :, 1, ai] ≈ demean(log.(abs.(truth))) atol = 1.0e-10
            end
        end

        # The two observables must give each station the same segments.
        lu = CAL.plan_parameters(hetmodel(ChannelBlocks(3), ChannelBlocks(3)), anames, geom)
        θu = zeros(lu.nθ)
        @test_throws "different segmentations" FP.solve_joint_bandpass!(
            θ, results, geom, pb, FP.bandpass_blocks(setup(lu), θu, :logamp); gauge = PinAntenna(1),
        )

        # A phase prior relates a track's segments, so it cannot express the
        # partial pin: two of station 1's three segments would be fitted and the
        # third overwritten with zero, which is not the fit the prior asks for.
        lp = CAL.plan_parameters(
            hetmodel(ChannelBlocks(2), ChannelBlocks(3); prior = CAL.RandomWalkPrior(; σ = 0.1)), anames, geom,
        )
        θ2 = zeros(lp.nθ)
        @test_throws "Drop the phase prior" FP.solve_joint_bandpass!(
            θ2, results, geom,
            FP.bandpass_blocks(setup(lp), θ2, :phase),
            FP.bandpass_blocks(setup(lp), θ2, :logamp);
            gauge = PinAntenna(1),
        )
    end

    @testset "one segmentation for every station is the identity refinement" begin
        l = CAL.plan_parameters(
            CAL.GainModel(;
                phase = (; bandpass = bpf(ChannelBlocks(2))),
                logamp = (; bandpass = bpf(ChannelBlocks(2))),
            ), anames, geom,
        )
        blocks = FP.bandpass_blocks(setup(l), zeros(l.nθ), :phase)
        plan = only(blocks).plan
        fseg, cells = FP._station_freq_segments(blocks, nant)
        # The cells ARE that segmentation's own channel groups and every station
        # maps each cell to itself, so the solve is the pre-refinement one.
        @test cells == CAL.segment_groups(plan.fseg_id, length(plan.nchan_seg))
        @test all(fseg[a, :] == collect(eachindex(cells)) for a in 1:nant)
        # Nothing ties one cell to another, so every cell is its own component
        # and the reference station is pinned in all of them — the whole-track
        # zeroing a station-uniform model has always had.
        results = _joint_scan_accumulators(
            ones(ComplexF64, nant, 2, 1, nchan), ones(ComplexF64, 4, length(bl_pairs), length(feeds)),
            fill(1, nant, 4), bl_pairs, feeds, geom,
        )
        @test _joint_pins(geom, blocks, results, fill(1, nant, 4), PinAntenna(1)) ==
            Set((1, (1, f, k, 1)) for f in 1:2 for k in eachindex(cells))
    end

    @testset "a station holding one gain over cells the others split" begin
        # Station 1 carries ONE bandpass over the whole band while the rest
        # carry two, so its single segment pools both refinement cells. Pinning
        # it gauges the solve exactly: its gain is constant in frequency, so the
        # phase the pin removes from every station is a single constant.
        l = CAL.plan_parameters(hetmodel(CAL.GlobalFrequency(), ChannelBlocks(3)), anames, geom)
        s = setup(l)
        θ = zeros(l.nθ)
        phase_blocks = FP.bandpass_blocks(s, θ, :phase)
        amp_blocks = FP.bandpass_blocks(s, θ, :logamp)
        fseg, cells = FP._station_freq_segments(phase_blocks, nant)
        @test cells == [[1, 2, 3], [4, 5, 6]]
        @test fseg[1, :] == [1, 1]
        @test all(fseg[a, :] == [1, 2] for a in 2:nant)
        # Station 1 ties the two cells into one component, so the gauge has one
        # constant to fix per feed however it picks the node to fix it at.
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
        @test _joint_pins(geom, phase_blocks, results, fill(1, nant, 4), PinAntenna(1)) ==
            Set([(1, (1, 1, 1, 1)), (1, (1, 2, 1, 1))])
        FP.solve_joint_bandpass!(
            θ, results, geom, phase_blocks, amp_blocks;
            gauge = PinAntenna(1), max_iterations = 200, tolerance = 1.0e-13,
        )

        demean(v) = v .- sum(v) / length(v)
        cdemean(v) = rem2pi.(v .- angle(sum(cis, v)), RoundNearest)
        # The first channel of each cell stands for the segment holding it.
        rep = [1, 4]
        for f in 1:2
            # The station with one segment has nothing to vary against: its own
            # band mean IS its single value, so both observables gauge to zero.
            @test all(iszero, phase_blocks[1].θ[1, f, :, 1, 1])
            @test all(iszero, amp_blocks[1].θ[1, f, :, 1, 1])
            # The rest recover their two segments from data pooled over three
            # channels each, measured against the pinned station.
            for (ai, a) in pairs(phase_blocks[2].stations)
                @test phase_blocks[2].θ[1, f, :, 1, ai] ≈
                    cdemean([angle(gtrue[a, f, 1, c]) for c in rep]) atol = 1.0e-8
                @test amp_blocks[2].θ[1, f, :, 1, ai] ≈
                    demean([log(abs(gtrue[a, f, 1, c])) for c in rep]) atol = 1.0e-8
            end
        end
        # Pinning a station whose segmentation is FINER than the free modes is a
        # partial pin, not an over-constraint: one of station 2's two segments is
        # held and the other is fitted. The gauge constant it removes is common to
        # the whole component and each track is written band-demeaned, so θ comes
        # out the same as pinning station 1.
        θ2 = zeros(l.nθ)
        pb2 = FP.bandpass_blocks(s, θ2, :phase)
        FP.solve_joint_bandpass!(
            θ2, results, geom, pb2,
            FP.bandpass_blocks(s, θ2, :logamp); gauge = PinAntenna(2),
            max_iterations = 200, tolerance = 1.0e-13,
        )
        for (bi, block) in pairs(pb2)
            @test block.θ ≈ phase_blocks[bi].θ atol = 1.0e-8
        end
    end

    @testset "hyperparameters are resolved every sweep and recorded" begin
        ou = CAL.OUPrior(; scale = LogNormal(log(2.0e6), 1.0), σ = LogNormal(log(0.2), 1.0))
        l = CAL.plan_parameters(
            CAL.GainModel(;
                phase = (; bandpass = bpf(ChannelBlocks(1))), logamp = (; bandpass = bpf(ChannelBlocks(1); prior = ou)),
            ), anames, geom,
        )
        θ = zeros(l.nθ)
        pb, ab = FP.bandpass_blocks(setup(l), θ, :phase), FP.bandpass_blocks(setup(l), θ, :logamp)
        rng = MersenneTwister(5)
        gtrue = [exp(complex(0.1 * randn(rng), 0.3 * randn(rng))) for _ in 1:nant, _ in 1:2, _ in 1:1, _ in 1:nchan]
        Strue = [(0.5 + rand(rng)) * cis(2pi * rand(rng)) for _ in 1:4, _ in eachindex(bl_pairs), _ in eachindex(feeds)]
        results = _joint_scan_accumulators(gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, geom)
        amp_priors = [FP._block_prior_array(b, geom) for b in ab]
        @test all(p -> p === ou, only(amp_priors))
        FP.solve_joint_bandpass!(θ, results, geom, pb, ab; gauge = PinAntenna(1), amp_priors)
        @test all(p -> CAL.is_fixed_hyper(p.scale) && CAL.is_fixed_hyper(p.σ), only(amp_priors))
        rep = FP.bandpass_track_report(nothing, nothing; amp_priors = only(amp_priors))
        @test rep.amp_priors === only(amp_priors)
    end
end

# ── A level per spectral window beside a zero-mean shape ─────────────────────

@testset "JointSmoother: levels per spectral window plus an OU shape" begin
    nant, nchan = 4, 12
    anames = ["A$i" for i in 1:nant]
    geom = CAL.DataGeometry(; nfeed = 2,
        times = collect(0.0:3.0), scan_of_time = collect(1:4),
        channel_freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6),
        spw_of_chan = repeat([1, 2]; inner = nchan ÷ 2),
        scan_names = ["No00$i" for i in 1:4], spw_names = ["A", "B"],
        stations = anames,
    )
    ou = CAL.OUPrior(; scale = 3.0e6, σ = 1.0)
    comp(fs; prior = nothing) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = fs, Feed = PerFeed(), prior)
    obs = (; level = comp(CAL.PerSpectralWindow()), shape = comp(ChannelBlocks(1); prior = ou))
    l = CAL.plan_parameters(CAL.GainModel(; phase = obs, logamp = obs), anames, geom)
    s = (;
        layout = l,
        paths = (; phase = FP._bandpass_paths(l.plantree, :phase), logamp = FP._bandpass_paths(l.plantree, :logamp)),
    )
    θ = zeros(l.nθ)
    pb, ab = FP.bandpass_blocks(s, θ, :phase), FP.bandpass_blocks(s, θ, :logamp)
    plb, alb = FP.bandpass_level_blocks(s, θ, :phase), FP.bandpass_level_blocks(s, θ, :logamp)

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
    FP.solve_joint_bandpass!(
        θ, results, geom, pb, ab; phase_level_blocks = plb, amp_level_blocks = alb,
        gauge = PinAntenna(1),
    )

    level_leaf(b) = only(b).θ
    shape_leaf(b) = only(b).θ
    demean(v) = v .- sum(v) / length(v)
    cdemean(v) = rem2pi.(v .- angle(sum(cis, v)), RoundNearest)
    for a in 2:nant, f in 1:2
        # Level plus shape is the gain, gauged over the band.
        la = [level_leaf(alb)[1, f, spw[c], 1, a] + shape_leaf(ab)[1, f, c, 1, a] for c in 1:nchan]
        φ = [level_leaf(plb)[1, f, spw[c], 1, a] + shape_leaf(pb)[1, f, c, 1, a] for c in 1:nchan]
        @test la ≈ demean(log.(abs.(gtrue[a, f, 1, :]))) atol = 1.0e-2
        want = angle.(gtrue[a, f, 1, :]) .- angle.(gtrue[1, f, 1, :])
        @test maximum(abs, rem2pi.(φ .- cdemean(want), RoundNearest)) < 1.0e-2
        # The level carries the step; the shape stays small.
        @test maximum(abs, shape_leaf(ab)[1, f, :, 1, a]) < 0.1
        Lstep = level_leaf(alb)[1, f, 2, 1, a] - level_leaf(alb)[1, f, 1, 1, a]
        tstep = log(abs(gtrue[a, f, 1, end])) - log(abs(gtrue[a, f, 1, 1]))
        @test Lstep ≈ tstep atol = 0.05
    end
end

# ── A gauge imposed inside the solve ─────────────────────────────────────────
#
# The free direction of the joint solve is a phase spectrum common to every
# station, and a phase prior along frequency is not invariant to it: under a
# pin, the reference's bandpass lands, sign-flipped, in every other station's
# track. `ZeroSumPhase` is imposed as a constraint on every joint step.

@testset "JointSmoother: a zero-sum gauge is a constraint on the solve" begin
    nant, nchan = 6, 16
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan; nant)
    comp(; prior = nothing) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior)
    setup(l) = (;
        layout = l,
        paths = (; phase = FP._bandpass_paths(l.plantree, :phase), logamp = FP._bandpass_paths(l.plantree, :logamp)),
    )
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

    layout(prior = nothing, amp_prior = nothing) = CAL.plan_parameters(
        CAL.GainModel(; phase = (; bandpass = comp(; prior)), logamp = (; bandpass = comp(; prior = amp_prior))), anames, geom,
    )
    # Converged within `max_iterations` sweeps: no warning.
    function solve(gauge; prior = nothing, amp_prior = nothing, data = results, max_iterations = 200)
        l = layout(prior, amp_prior)
        θ = zeros(l.nθ)
        pb, ab = FP.bandpass_blocks(setup(l), θ, :phase), FP.bandpass_blocks(setup(l), θ, :logamp)
        @test_logs FP.solve_joint_bandpass!(θ, data, geom, pb, ab; gauge, max_iterations, tolerance = 1.0e-10)
        φ = only(pb).θ
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
    # Each written track is referenced to its own circular band mean, so the
    # zero station mean of every cell reads as one constant across the band.
    station_mean_spread(φ) = maximum(f -> (m = vec(sum(φ[:, f, :]; dims = 1)) ./ nant; maximum(m) - minimum(m)), 1:2)

    # A second-order walk of `σ` rad per channel (1 MHz), as rad/Hz^(3/2).
    rw2(σ) = CAL.RandomWalkPrior(; order = 2, σ = σ * sqrt(3 / (2 * 1.0e6^3)))

    @testset "without a prior the gauge leaves the products unchanged" begin
        zs, pin = solve(ZeroSumPhase()), solve(PinAntenna(1))
        @test station_mean_spread(zs) < 1.0e-8
        @test product_error(zs)[1] < 1.0e-8
        @test product_error(pin)[1] < 1.0e-8
    end

    @testset "under a prior the zero sum spreads the reference's structure" begin
        prior = rw2(1.0)
        zs, pin = solve(ZeroSumPhase(); prior), solve(PinAntenna(1); prior)
        @test station_mean_spread(zs) < 1.0e-8
        # Pinned, every track carries the reference's bandpass and its prior
        # smooths it away: rms error 0.20 rad, worst 0.87, against 0.10 and 0.41.
        worst_zs, rms_zs = product_error(zs)
        worst_pin, rms_pin = product_error(pin)
        @test rms_zs < 0.6 * rms_pin
        @test worst_zs < 0.6 * worst_pin

        # `ByComponent` maps the bandpass component to its choice.
        bc(choice, default) = ByComponent((; bandpass = choice); default)
        @test solve(bc(ZeroSumPhase(), PinAntenna(1)); prior) == zs
        @test solve(bc(PinAntenna(1), ZeroSumPhase()); prior) == pin
    end

    @testset "a stiff prior converges in about a pin's sweep count" for σ in (0.05, 0.01)
        # A phase this stiff cannot follow the reference's structure, and where a
        # slot is left more than π/2 off, a free amplitude's MAP is zero; the
        # amplitude prior keeps the MAP finite.
        zs = solve(ZeroSumPhase(); prior = rw2(σ), amp_prior = rw2(0.05), max_iterations = 80)
        solve(PinAntenna(1); prior = rw2(σ), amp_prior = rw2(0.05), max_iterations = 80)
        @test station_mean_spread(zs) < 1.0e-8

        # Without it the MAP does not exist, and the solve says so.
        l = layout(rw2(σ), nothing)
        θ = zeros(l.nθ)
        pb, ab = FP.bandpass_blocks(setup(l), θ, :phase), FP.bandpass_blocks(setup(l), θ, :logamp)
        @test_logs (:warn, r"did not converge|could not lower") match_mode = :any FP.solve_joint_bandpass!(
            θ, results, geom, pb, ab; gauge = PinAntenna(1), max_iterations = 80, tolerance = 1.0e-10,
        )
    end

    @testset "a slot without data takes the constraint through its priors" begin
        gap = deepcopy(results)
        for r in gap, (bi, (a, b)) in pairs(bl_pairs)
            3 in (a, b) || continue
            r.wl[bi, :, 5] .= 0
            r.rl[bi, :, 5] .= 0
        end
        zs = solve(ZeroSumPhase(); prior = rw2(0.05), amp_prior = rw2(0.05), data = gap)
        @test all(isfinite, zs[3, :, 5])
        @test station_mean_spread(zs) < 1.0e-8
    end

    @testset "only a pin marks pinned slots" begin
        blocks = FP.bandpass_blocks(setup(layout()), zeros(layout().nθ), :phase)
        @test isempty(_joint_pins(geom, blocks, results, fill(1, nant, 4), ZeroSumPhase()))
        @test length(_joint_pins(geom, blocks, results, fill(1, nant, 4), PinAntenna(1))) == 2 * nchan
    end

    @test Bandpass().gauge === ZeroSumPhase()
end

@testset "JointSmoother: weak cross-hands converge in a few sweeps" begin
    nant, nchan = 5, 12
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan; nant)
    comp = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed())
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; bandpass = comp), logamp = (; bandpass = comp)), anames, geom)
    setup = (; layout = l, paths = (; phase = FP._bandpass_paths(l.plantree, :phase), logamp = FP._bandpass_paths(l.plantree, :logamp)))
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (1, 2), (2, 1), (2, 2)]
    rng = MersenneTwister(313)
    gtrue = exp.(complex.(0.1 .* randn(rng, nant, 2, 1, nchan), 0.6 .* randn(rng, nant, 2, 1, nchan)))
    # The cross-hand source terms carry ~1e-3 of the parallel hands' information,
    # which a station-by-station sweep alone takes thousands of sweeps to resolve.
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
        pb, ab = FP.bandpass_blocks(setup, θ, :phase), FP.bandpass_blocks(setup, θ, :logamp)
        @test_logs FP.solve_joint_bandpass!(θ, results, geom, pb, ab; gauge, max_iterations = 30, tolerance = 1.0e-10)
        φ = only(pb).θ
        @test rl_error([φ[1, f, c, 1, a] for a in 1:nant, f in 1:2, c in 1:nchan]) < 1.0e-8
    end

    # Under a prior the step is the MAP step under the gauge, with any level
    # profiled out, and converges as fast; a sweep alone takes tens of thousands
    # of sweeps here.
    rw2 = CAL.RandomWalkPrior(; order = 2, σ = 0.3 * sqrt(3 / (2 * 1.0e6^3)))
    ou = CAL.OUPrior(; scale = 3.0e6, σ = 0.5)
    cp = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = rw2)
    leveled = (;
        level = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.PerSpectralWindow(), Feed = PerFeed()),
        shape = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior = ou),
    )
    @testset "$name, $gauge" for (name, obs) in (("RW2", (; bandpass = cp)), ("level + OU", leveled)), gauge in (PinAntenna(1), ZeroSumPhase())
        lp = CAL.plan_parameters(CAL.GainModel(; phase = obs, logamp = obs), anames, geom)
        setup_p = (; layout = lp, paths = (; phase = FP._bandpass_paths(lp.plantree, :phase), logamp = FP._bandpass_paths(lp.plantree, :logamp)))
        θ = zeros(lp.nθ)
        pb, ab = FP.bandpass_blocks(setup_p, θ, :phase), FP.bandpass_blocks(setup_p, θ, :logamp)
        plb, alb = FP.bandpass_level_blocks(setup_p, θ, :phase), FP.bandpass_level_blocks(setup_p, θ, :logamp)
        @test_logs FP.solve_joint_bandpass!(
            θ, results, geom, pb, ab; phase_level_blocks = plb, amp_level_blocks = alb,
            gauge, max_iterations = 30, tolerance = 1.0e-10,
        )
    end
end

# ── Stopping in standard errors ──────────────────────────────────────────────
#
# A station whose data carry almost no information moves by large fractions of
# a radian between sweeps long after the solution has settled within its
# standard error; the default tolerance stops on the change in standard errors.

@testset "JointSmoother: the default tolerance stops within a fraction of a standard error" begin
    nant, nchan = 5, 48
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan; nant)
    comp = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed())
    l = CAL.plan_parameters(CAL.GainModel(; phase = (; bandpass = comp), logamp = (; bandpass = comp)), anames, geom)
    setup = (; layout = l, paths = (; phase = FP._bandpass_paths(l.plantree, :phase), logamp = FP._bandpass_paths(l.plantree, :logamp)))
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
        pb, ab = FP.bandpass_blocks(setup, θ, :phase), FP.bandpass_blocks(setup, θ, :logamp)
        FP.solve_joint_bandpass!(θ, results, geom, pb, ab; gauge = PinAntenna(1), kw...)
        return only(pb).θ[1, :, :, 1, :]
    end
    # A relative-change criterion needs over 160 sweeps here.
    φ_default = @test_logs solve(max_iterations = 60)
    φ_tight = solve(max_iterations = 5000, tolerance = 1.0e-8)
    # Phase standard error of a well-measured station's slot: 1/√(Σ w|m|²) over
    # its 4 baselines and 4 scans, with |g| ≈ 1 and |S| ≥ 0.5.
    σφ = 1 / sqrt(4 * 4 * 50.0 * 0.25)
    @test maximum(abs, rem2pi.(φ_default[:, :, 1:(nant - 1)] .- φ_tight[:, :, 1:(nant - 1)], RoundNearest)) < 0.1σφ

    @test_logs (:warn, r"did not converge in 2 sweeps: the last sweep moved a gain by .* of its standard error \(A\d, feed \d, frequency segment \d+, time segment 1\)") solve(max_iterations = 2)
end

# ── Band-referenced products and declined tracks ─────────────────────────────

@testset "JointSmoother: gauges agree on band-referenced products" begin
    bl_pairs(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]
    feeds = [(1, 1), (2, 2)]
    bpf(fs; prior = nothing) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = fs, Feed = PerFeed(), prior)
    hetmodel(a1, rest; prior = nothing) = CAL.GainModel(;
        phase = (; bandpass = bpf(rest; prior)), logamp = (; bandpass = bpf(rest)),
        stations = (A1 = (; phase = (; bandpass = bpf(a1; prior)), logamp = (; bandpass = bpf(a1))),),
    )
    setup(l) = (;
        layout = l,
        paths = (; phase = FP._bandpass_paths(l.plantree, :phase), logamp = FP._bandpass_paths(l.plantree, :logamp)),
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
    function solve(n1, nrest, gauge, geom, results; prior = nothing)
        nant = length(geom.stations)
        l = CAL.plan_parameters(hetmodel(ChannelBlocks(n1), ChannelBlocks(nrest); prior), geom.stations, geom)
        θ = zeros(l.nθ)
        pb, ab = FP.bandpass_blocks(setup(l), θ, :phase), FP.bandpass_blocks(setup(l), θ, :logamp)
        FP.solve_joint_bandpass!(θ, results, geom, pb, ab; gauge, max_iterations = 2000, tolerance = 1.0e-12)
        φ = zeros(nant, 2, length(geom.channel_freqs))
        for blk in pb, (ai, a) in pairs(blk.stations), f in 1:2, c in axes(φ, 3)
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

    @testset "station 1 on $n1-channel blocks, the rest on $nrest" for (n1, nrest) in ((1, 2), (2, 1))
        gtrue, geom, results = synthetic(n1, nrest; nant = 4, nchan = 4, spread = 0.6, seed = 302)
        for gauge in (PinAntenna(1), PinAntenna(2), ZeroSumPhase())
            @test product_error(solve(n1, nrest, gauge, geom, results), gtrue) < 1.0e-8
        end
    end

    @testset "per-channel structure past unwrapping" begin
        gtrue, geom, results = synthetic(1, 1; nant = 6, nchan = 16, spread = 2.0, seed = 302)
        @test product_error(solve(1, 1, ZeroSumPhase(), geom, results), gtrue) < 1.0e-8
        # Under a prior every track is declined at unwrapping, so no gain changes.
        prior = CAL.RandomWalkPrior(; order = 2, σ = sqrt(3 / (2 * 1.0e6^3)))
        @test_throws "changed no gain" solve(1, 1, ZeroSumPhase(), geom, results; prior)
    end
end

# ── One observable ──────────────────────────────────────────────────────────

@testset "JointSmoother: a phase-only or amplitude-only model holds the other at zero" begin
    nant, nchan, ntime, nscans = 4, 8, 6, 6
    rng = MersenneTwister(0x0515)
    gauge = PinAntenna(1)
    fm = default_fringe_terms()
    terms = default_bandpass_terms()
    # Where the data hold no bandpass in the other observable, a one-sided model
    # gives the same answer as the two-sided one.
    function compare(obs, data_kw)
        ps, _ = _build_fringe_ps(; nant, nspw = 1, nchan, ntime, nscans, data_kw...)
        one_sided = GainModel(; (obs => getproperty(terms, obs),)...)
        _, sol1 = _fit_chain((BaselineFringeFit(; model = fm, gauge), Bandpass(; model = one_sided, gauge)), ps)
        _, sol2 = _fit_chain((BaselineFringeFit(; model = fm, gauge), Bandpass(; model = terms, gauge)), ps)
        return sol1, maximum(abs, _jb_leaf(sol1, obs) .- _jb_leaf(sol2, obs))
    end

    sol, Δ = compare(:phase, (; bandpass = 0.4 .* randn(rng, nant, 2, nchan), seed = 7))
    @test Δ < 1.0e-4
    @test isempty(sol.steps[:bandpass].amp_status)

    abp_true = 0.15 .* randn(rng, nant, 2, nchan)
    sol, Δ = compare(:logamp, (; amp_bandpass = abp_true, seed = 8))
    @test Δ < 1.0e-4
    @test isempty(sol.steps[:bandpass].phase_status)
    leaf = _jb_leaf(sol, :logamp)
    @test all(abs(leaf[1, f, c, 1, a] - (abp_true[a, f, c] - sum(abp_true[a, f, :]) / nchan)) < 1.0e-3 for a in 1:nant, f in 1:2, c in 1:nchan)
end
