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

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "Bandpass(smoother = JointSmoother()): recovers the bandpass under per-baseline/pol source structure" begin
    nant, nspw, nchan, ntime, nscans = 4, 1, 8, 6, 6
    rng = MersenneTwister(4242)
    nglob = nspw * nchan
    bp_true = 0.4 .* randn(rng, nant, 2, nglob)
    abp_true = 0.15 .* randn(rng, nant, 2, nglob)
    uvset, truth = _build_fringe_uvset(;
        nant, nspw, nchan, ntime, nscans, bandpass = bp_true, amp_bandpass = abp_true,
        seed = 99,
    )

    nbl = length(truth.bl_pairs)
    npol = length(truth.pol_labels)
    src_amp = 0.4 .+ 1.6 .* rand(rng, nscans, nbl, npol)
    src_phase = (2 .* rand(rng, nscans, nbl, npol) .- 1) .* pi
    for (_, leaf) in DimensionalData.branches(uvset)
        s = parse(Int, DimensionalData.metadata(leaf).scan_name)
        V = leaf[:vis]
        for p in 1:npol, bi in 1:nbl
            V[:, :, bi, p] .*= ComplexF32(src_amp[s, bi, p] * cis(src_phase[s, bi, p]))
        end
    end

    fm = default_fringe_terms()
    # `ref_ant` indexes the truth arrays below; `gauge` is what the solve takes.
    ref_ant = 1
    gauge = PinAntenna(ref_ant)     # Bandpass's/CalibrationPipeline's default
    sol_closure = fit(
        CalibrationPipeline(
            BaselineFringeFit(model = fm), Bandpass(smoother = FP.PerTrackSmoother());
            exec = ExecutionConfig(),
            gauge = PinAntenna(1),
        ),
        uvset,
    )
    # This synthetic truth is i.i.d. RANDOM per channel (a deliberately hard,
    # uncorrelated-neighbor case for the ALS to track) — noticeably slower to
    # converge than JointSmoother's own defaults, hence the raised iteration
    # count/tolerance.
    sol_joint = fit(
        CalibrationPipeline(
            BaselineFringeFit(model = fm),
            Bandpass(smoother = FP.JointSmoother(max_iterations = 60, tolerance = 1.0e-10));
            exec = ExecutionConfig(),
            gauge = PinAntenna(1),
        ),
        uvset,
    )

    function bp_leaves(sol)
        bstep = sol[:bandpass].steps[1]
        pplan = bstep.layout.plantree.phase.bandpass
        aplan = bstep.layout.plantree.logamp.bandpass
        return CAL._component_leaf(pplan, bstep.θ), CAL._component_leaf(aplan, bstep.θ)
    end
    pleaf_c, aleaf_c = bp_leaves(sol_closure)
    pleaf_j, aleaf_j = bp_leaves(sol_joint)

    wrapped_err(x, y) = maximum(abs, rem2pi.(x .- y, RoundNearest))
    # PHASE is recovered relative to the reference's feed 1 — the joint tier pins that
    # node's phase at every segment, and the closure tier references its solve to
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

@testset "JointSmoother: the shape specs act as priors INSIDE the ALS" begin
    # The gain update fits each (station, feed) track under its spec instead of
    # solving every frequency segment independently. `FreeShape` on both
    # observables IS that independent per-segment solve; a stiff roughness penalty
    # must instead leave only the second-difference null space — a straight line
    # in frequency.
    nant, nspw, nchan, ntime, nscans = 4, 1, 8, 4, 3
    nglob = nspw * nchan
    rng = MersenneTwister(0x101A)
    bp_true = 0.4 .* randn(rng, nant, 2, nglob)
    abp_true = 0.15 .* randn(rng, nant, 2, nglob)
    uvset, _ = _build_fringe_uvset(;
        nant, nspw, nchan, ntime, nscans,
        bandpass = bp_true, amp_bandpass = abp_true, seed = 21,
    )
    fm = default_fringe_terms()
    runbp(sm) = fit(
        CalibrationPipeline(BaselineFringeFit(model = fm), Bandpass(smoother = sm); exec = ExecutionConfig(), gauge = PinAntenna(1)),
        uvset,
    )[:bandpass].steps[1]

    s_free = runbp(FP.JointSmoother(max_iterations = 40, tolerance = 1.0e-10))

    stiff = FP.WhittakerShape(1.0e8)
    s_stiff = runbp(FP.JointSmoother(phase = stiff, amp = stiff, max_iterations = 40))
    pleaf = CAL._component_leaf(s_stiff.layout.plantree.phase.bandpass, s_stiff.θ)
    aleaf = CAL._component_leaf(s_stiff.layout.plantree.logamp.bandpass, s_stiff.θ)
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
    apleaf = CAL._component_leaf(s_free.layout.plantree.logamp.bandpass, s_free.θ)
    rough = maximum(
        maximum(abs, diff(diff(Float64[apleaf[1, f, c, 1, a] for c in 1:nglob])))
            for a in 1:nant, f in 1:2
    )
    @test rough > 0.1
end

@testset "JointSmoother: reference-antenna amplitude is gauged, not pinned" begin
    # Only the per-segment PHASE is a gauge freedom: the phase of a factor common
    # to every station at one segment cancels between `g_a` and `conj(g_b)`, while
    # its MAGNITUDE does not and the frequency-flat `S` cannot absorb it. So the
    # reference node's phase is pinned at every segment and its amplitude is
    # solved like any other station's. Pinning the amplitude too would discard the
    # reference antenna's own passband — identifiable structure — and bias every
    # other station through the resulting inconsistency.
    nant, nspw, nchan, ntime, nscans = 4, 1, 8, 4, 3
    nglob = nspw * nchan
    rng = MersenneTwister(0x9A17)
    bp_true = 0.4 .* randn(rng, nant, 2, nglob)
    abp_true = 0.25 .* randn(rng, nant, 2, nglob)
    uvset, truth = _build_fringe_uvset(;
        nant, nspw, nchan, ntime, nscans,
        bandpass = bp_true, amp_bandpass = abp_true, seed = 33,
    )
    # Per-(scan, baseline, pol) structure: what the joint tier exists to absorb,
    # and what biases the closure.
    nbl = length(truth.bl_pairs); npol = length(truth.pol_labels)
    samp = 0.4 .+ 1.6 .* rand(rng, nscans, nbl, npol)
    sph = (2 .* rand(rng, nscans, nbl, npol) .- 1) .* pi
    for (_, leaf) in DimensionalData.branches(uvset)
        sc = parse(Int, DimensionalData.metadata(leaf).scan_name)
        V = leaf[:vis]
        for p in 1:npol, bi in 1:nbl
            V[:, :, bi, p] .*= ComplexF32(samp[sc, bi, p] * cis(sph[sc, bi, p]))
        end
    end

    fm = default_fringe_terms()
    runbp(sm) = fit(
        CalibrationPipeline(BaselineFringeFit(model = fm), Bandpass(smoother = sm); exec = ExecutionConfig(), gauge = PinAntenna(1)),
        uvset,
    )[:bandpass].steps[1]
    s_joint = runbp(FP.JointSmoother(max_iterations = 60, tolerance = 1.0e-12))
    s_closure = runbp(FP.PerTrackSmoother(amp = FP.FreeShape()))

    la(s, a, f) = (
        L = CAL._component_leaf(s.layout.plantree.logamp.bandpass, s.θ);
        Float64[L[1, f, c, 1, a] for c in 1:nglob]
    )
    gauge(x) = x .- sum(x) / length(x)
    # gauge = PinAntenna(1), feed 1 is the pinned node (Bandpass's/CalibrationPipeline's
    # default reference). Its amplitude bandpass must come back, not flat.
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

# A four-scan, one-time-sample-per-scan geometry whose second half is a separate
# instrument segment.
function _seg_geometry(nchan)
    return CAL.DataGeometry(;
        times = collect(0.0:3.0), scan_of_time = collect(1:4),
        channel_freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6),
        spw_of_chan = ones(Int, nchan),
        scan_names = ["No00$i" for i in 1:4], spw_names = ["A"],
    )
end

# Per-scan accumulators for `g[ant, feed, segment, channel]` observed through
# `S[scan, baseline, pol]`, with `tseg[ant, scan]` naming each station's segment.
function _joint_scan_accumulators(g, S, tseg, bl_pairs, feeds, nchan)
    nbl, npol = length(bl_pairs), length(feeds)
    return map(axes(S, 1)) do si
        rl, wl = FP.bandpass_accumulators(nbl, npol, nchan)
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
        bp_path = FP._bandpass_path(l.plantree, :phase),
        amp_path = FP._bandpass_path(l.plantree, :logamp),
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
            gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, nchan,
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
                θ_merged, results[idx], bl_pairs, pol_products, nant,
                merged_blocks, merged_blocks;
                gauge = PinAntenna(2), max_iterations = 40, tolerance = 1.0e-12,
                tseg = view(tseg, :, idx),
            )
        end
        θ_loop = zeros(l.nθ)
        loop_blocks = FP.bandpass_blocks(setup(l), θ_loop, :phase)
        for (ts, idx) in pairs(FP.time_segment_scans(plan, results))
            isempty(idx) && continue
            FP.solve_joint_bandpass!(
                θ_loop, results[idx], bl_pairs, pol_products, nant,
                loop_blocks, loop_blocks;
                gauge = PinAntenna(2), max_iterations = 40, tolerance = 1.0e-12,
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
        results = _joint_scan_accumulators(gtrue, Strue, tseg, bl_pairs, feeds, nchan)
        @test FP._joint_scan_groups(tseg) == [[1, 2, 3, 4]]

        phase_status = fill(FP._BP_TRACK_NODATA, nant, 2, 1, 2)
        amp_status = fill(FP._BP_TRACK_NODATA, nant, 2, 1, 2)
        FP.solve_joint_bandpass!(
            θ, results, bl_pairs, pol_products, nant, phase_blocks, amp_blocks;
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
                @test phase_status[a, f, 1, 2] == FP._BP_TRACK_NODATA
            end
            @test phase_status[1, f, 1, 2] == FP._BP_TRACK_SOLVED
        end
    end

    @testset "a heterogeneous model writes through each station's own block" begin
        # Station 1 alone carries the break; the rest hold one gain over the
        # track, so the two signature groups become two blocks with different
        # `:Ant` axes and different time segmentations, and the solve has to
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
        results = _joint_scan_accumulators(gtrue, Strue, tsg, bl_pairs, feeds, nchan)
        @test FP._station_time_segments(phase_blocks, results, nant) == tsg
        @test FP._joint_scan_groups(tsg) == [[1, 2, 3, 4]]

        # The status grid spans the union of the two segmentations, and the cells
        # the one-segment stations do not have are absent, not unmeasured.
        phase_status = FP._joint_status_array(phase_blocks, nant, 1)
        amp_status = FP._joint_status_array(amp_blocks, nant, 1)
        @test size(phase_status) == (nant, 2, 1, 2)
        @test all(==(FP._BP_TRACK_NODATA), view(phase_status, :, :, :, 1))
        @test all(==(FP._BP_TRACK_NODATA), view(phase_status, 1, :, :, 2))
        @test all(==(FP._BP_TRACK_NA), view(phase_status, 2:nant, :, :, 2))

        FP.solve_joint_bandpass!(
            θ, results, bl_pairs, pol_products, nant, phase_blocks, amp_blocks;
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
            # The broken station is alone on its block's `:Ant` axis, and both of
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
            @test phase_status[1, f, 1, 2] == FP._BP_TRACK_SOLVED
        end
        # No solve writes an absent cell, so the code survives the whole run…
        @test all(==(FP._BP_TRACK_NA), view(phase_status, 2:nant, :, :, 2))
        @test all(==(FP._BP_TRACK_NA), view(amp_status, 2:nant, :, :, 2))
        # …and it is reported apart from the four outcomes a real track can have,
        # so it does not dilute the degenerate-track fraction.
        rep = FP.bandpass_track_report(phase_status, amp_status, [1])
        @test rep.track_labels[FP._BP_TRACK_NA + 1] == "na"
        @test rep.n_na == 2 * 2 * (nant - 1)
        @test rep.n_nodata + rep.n_solved + rep.n_flat + rep.n_declined == 2 * 2 * (nant + 1)
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
        bp_path = FP._bandpass_path(l.plantree, :phase),
        amp_path = FP._bandpass_path(l.plantree, :logamp),
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
        # Every station shares `ChannelBlocks(1)` here, so each channel is its own
        # frequency segment and the graph splits along them.
        fseg = repeat(collect(1:nchan)', nant)
        nodes, pins = FP._joint_bandpass_pins(
            bl_pairs, feeds, nant, het, fseg, nchan, PinAntenna(1),
        )
        # The constant stations bridge the break, so each (feed, channel) is one
        # component and the reference is pinned in its first time segment only —
        # its second is free to carry the break it actually has.
        @test pins == Set(nodes[1, f, 1, c] for f in 1:2 for c in 1:nchan)

        # A segmentation every station shares has nothing to bridge the epochs:
        # each (feed, time segment, channel) is its own component and carries its
        # own pin.
        _, upins = FP._joint_bandpass_pins(
            bl_pairs, feeds, nant, uniform, fseg, nchan, PinAntenna(1),
        )
        @test length(upins) == 2 * 2 * nchan
    end

    @testset "the reference station's own break is measured, not gauged away" begin
        θ = zeros(l.nθ)
        phase_blocks = FP.bandpass_blocks(s, θ, :phase)
        amp_blocks = FP.bandpass_blocks(s, θ, :logamp)
        results = _joint_scan_accumulators(gtrue, Strue, het, bl_pairs, feeds, nchan)
        FP.solve_joint_bandpass!(
            θ, results, bl_pairs, pol_products, nant, phase_blocks, amp_blocks;
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

# ── Per-station frequency segmentation ───────────────────────────────────────
#
# Two stations with different `Frequency` segmentations share no segment grid, so
# the accumulators are reduced onto the COMMON REFINEMENT of the blocks'
# segmentations and each station's gain is solved on its own segments — one gain
# pooled over however many refinement cells a segment spans.

@testset "JointSmoother: each station's own frequency segments" begin
    nant, nchan = 4, 6
    anames = ["A$i" for i in 1:nant]
    geom = _seg_geometry(nchan)
    bpf(fs) = GainComponent(ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = fs, Feed = PerFeed())
    hetmodel(a1, rest) = CAL.GainModel(;
        phase = (; bandpass = bpf(rest)), logamp = (; bandpass = bpf(rest)),
        stations = (
            A1 = (; phase = (; bandpass = bpf(a1)), logamp = (; bandpass = bpf(a1))),
        ),
    )
    setup(l) = (;
        layout = l,
        bp_path = FP._bandpass_path(l.plantree, :phase),
        amp_path = FP._bandpass_path(l.plantree, :logamp),
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
        nfsmax = maximum(fseg)
        nodes, pins = FP._joint_bandpass_pins(
            bl_pairs, feeds, nant, fill(1, nant, 4), fseg, nfsmax, PinAntenna(1),
        )
        @test pins == Set([nodes[1, 1, 1, 1], nodes[1, 2, 1, 1]])

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
            gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, nchan,
        )

        θ = zeros(l.nθ)
        pb = FP.bandpass_blocks(setup(l), θ, :phase)
        ab = FP.bandpass_blocks(setup(l), θ, :logamp)
        # The misaligned refinement couples the two segmentations through cells
        # neither owns alone, which the alternating sweep works through slowly:
        # the recovery is exact, but 200 sweeps only reach 3e-8 of it.
        FP.solve_joint_bandpass!(
            θ, results, bl_pairs, pol_products, nant, pb, ab; gauge = PinAntenna(1),
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

        # A spec that fits a band's segments jointly cannot express the partial
        # pin: two of station 1's three segments would be fitted and the third
        # overwritten with zero, which is not the fit the spec asks for.
        θ2 = zeros(l.nθ)
        @test_throws "Use `FreeShape` for the phase" FP.solve_joint_bandpass!(
            θ2, results, bl_pairs, pol_products, nant,
            FP.bandpass_blocks(setup(l), θ2, :phase),
            FP.bandpass_blocks(setup(l), θ2, :logamp);
            gauge = PinAntenna(1), phase_spec = FP.PolynomialShape(1),
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
        nodes, pins = FP._joint_bandpass_pins(
            bl_pairs, feeds, nant, fill(1, nant, 4), fseg, length(cells), PinAntenna(1),
        )
        @test pins == Set(nodes[1, f, 1, k] for f in 1:2 for k in eachindex(cells))
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
        nodes, pins = FP._joint_bandpass_pins(
            bl_pairs, feeds, nant, fill(1, nant, 4), fseg, 2, PinAntenna(1),
        )
        @test pins == Set([nodes[1, 1, 1, 1], nodes[1, 2, 1, 1]])

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
            gtrue, Strue, fill(1, nant, 4), bl_pairs, feeds, nchan,
        )
        FP.solve_joint_bandpass!(
            θ, results, bl_pairs, pol_products, nant, phase_blocks, amp_blocks;
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
            θ2, results, bl_pairs, pol_products, nant, pb2,
            FP.bandpass_blocks(s, θ2, :logamp); gauge = PinAntenna(2),
            max_iterations = 200, tolerance = 1.0e-13,
        )
        for (bi, block) in pairs(pb2)
            @test block.θ ≈ phase_blocks[bi].θ atol = 1.0e-8
        end
    end
end
