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

    fm = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))
    # `ref_ant` indexes the truth arrays below; `gauge` is what the solve takes.
    ref_ant = 1
    gauge = PinAntenna(ref_ant)     # Bandpass's/CalibrationPipeline's default
    sol_closure = fit(
        CalibrationPipeline(
            FringeFit(model = fm), Bandpass(smoother = FP.PerTrackSmoother());
            exec = ExecutionConfig(),
        ),
        uvset,
    )
    # This synthetic truth is i.i.d. RANDOM per channel (a deliberately hard,
    # uncorrelated-neighbor case for the ALS to track) — noticeably slower to
    # converge than JointSmoother's own defaults, hence the raised iteration
    # count/tolerance.
    sol_joint = fit(
        CalibrationPipeline(
            FringeFit(model = fm),
            Bandpass(smoother = FP.JointSmoother(max_iterations = 60, tolerance = 1.0e-10));
            exec = ExecutionConfig(),
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
    fm = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))
    runbp(sm) = fit(
        CalibrationPipeline(FringeFit(model = fm), Bandpass(smoother = sm); exec = ExecutionConfig()),
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

    fm = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))
    runbp(sm) = fit(
        CalibrationPipeline(FringeFit(model = fm), Bandpass(smoother = sm); exec = ExecutionConfig()),
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
