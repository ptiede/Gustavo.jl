# ── Bandpass(estimator = JointALS()): solve_joint_bandpass! ──────────────────
#
# solve_phase_bandpass!/solve_amp_bandpass! (the default, closure-based path)
# assume a baseline's source term cancels out of the per-channel
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
# recovered bandpass is relative to `ref_ant`'s — truth is gauged the same way
# before comparing. The injected factor is frequency-FLAT (matching "constant
# per scan"), which the closure path's per-AP de-rotation heuristic already
# absorbs incidentally for PHASE (it isn't targeted at this, but a band-flat
# offset is exactly what de-rotating each AP to its own band-average removes),
# so the phase gate here only asks that the joint solve be no worse. The
# injected AMPLITUDE has no such accidental cover — `log|V| = la_a + la_b` has
# no baseline term at all — so that is where the joint solve's structural
# advantage is unambiguous.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "Bandpass(estimator = JointALS()): recovers the bandpass under per-baseline/pol source structure" begin
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
    ref_ant = 1     # Bandpass's/CalibrationPipeline's default
    sol_closure = fit(
        CalibrationPipeline(FringeFit(model = fm), Bandpass(); exec = ExecutionConfig(ntasks = 1)),
        uvset,
    )
    # This synthetic truth is i.i.d. RANDOM per channel (a deliberately hard,
    # uncorrelated-neighbor case for the ALS to track) — noticeably slower to
    # converge than JointALS's own defaults, hence the raised iteration
    # count/tolerance.
    sol_joint = fit(
        CalibrationPipeline(
            FringeFit(model = fm),
            Bandpass(estimator = JointALS(max_iterations = 60, tolerance = 1.0e-10));
            exec = ExecutionConfig(ntasks = 1),
        ),
        uvset,
    )

    function bp_leaves(sol)
        bstep = CAL._step(sol, :bandpass)
        pplan = FP._bandpass_plan(bstep.model, bstep.layout)
        aplan = FP._amp_bandpass_plan(bstep.model, bstep.layout)
        return CAL._component_leaf(pplan, bstep.θ), CAL._component_leaf(aplan, bstep.θ)
    end
    pleaf_c, aleaf_c = bp_leaves(sol_closure)
    pleaf_j, aleaf_j = bp_leaves(sol_joint)

    wrapped_err(x, y) = maximum(abs, rem2pi.(x .- y, RoundNearest))
    # Truth gauged RELATIVE TO ref_ant's feed 1 (both solvers' recovered
    # bandpass is relative to that one reference track), then referenced to
    # its own circular mean / zero mean — matching solve_phase_bandpass!'s /
    # solve_amp_bandpass!'s final per-(station, feed) gauge convention.
    gauge_phase_rel(x, xref) = rem2pi.((x .- xref) .- angle(sum(cis, x .- xref)), RoundNearest)
    gauge_amp_rel(x, xref) = (x .- xref) .- sum(x .- xref) / length(x)

    max_phase_err_closure = 0.0
    max_phase_err_joint = 0.0
    max_amp_err_closure = 0.0
    max_amp_err_joint = 0.0
    for a in 1:nant, f in 1:2
        btrue = gauge_phase_rel(bp_true[a, f, :], bp_true[ref_ant, 1, :])
        abtrue = gauge_amp_rel(abp_true[a, f, :], abp_true[ref_ant, 1, :])
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
