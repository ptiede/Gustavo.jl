# ── Bandpass step ─────────────────────────────────────────────────────────────
#
# The bandpass stage on the composable engine. (The M4 parity gates against the
# frozen monolith ran before its deletion.) Standing guarantees:
# - FringeFit |> Bandpass's fringe and per-channel bandpass blocks are
#   invariant under appending a TemporalSmoother stage (later stages never move
#   earlier blocks), and θ is bit-deterministic across group concurrency (the
#   per-scan accumulator contributions fold in group-index order).
# - The refine kernels (dTEC, SBD) are inner-invariant and recover the
#   injected dTEC standalone on a scan view.
# - `step_solution` extracts a portable bandpass-only solution and
#   `ApplySolution` applies it same-set (index-aligned) and cross-set
#   (station-name-mapped, channel-layout-validated, time-constant only).

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

# One component's θ block from a step's own layout. `i` indexes
# `step.layout.plans` (phase components first, then log-amplitude).
_blk(step, i) = step.θ[CAL.component_ranges(step.layout)[i]]

# The bandpass step's own components, reached by the names its model gives them.
_bp_phase_plan(step) = step.layout.plantree.phase.bandpass
_bp_amp_plan(step) = step.layout.plantree.logamp.bandpass
_bp_phase(step) = step.θ[_bp_phase_plan(step).range]
_bp_amp(step) = step.θ[_bp_amp_plan(step).range]

@testset "Bandpass step (new engine)" begin
    nant, nspw, nchan = 4, 2, 8
    rng = MersenneTwister(11)
    nglob = nspw * nchan
    bp_true = 0.5 .* randn(rng, nant, 2, nglob)
    abp_true = 0.2 .* randn(rng, nant, 2, nglob)
    uvset, _ = _build_fringe_uvset(;
        nant, nspw, nchan, bandpass = bp_true, amp_bandpass = abp_true,
    )
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    fm = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))

    # The fuller-pipeline reference (adhoc is solved AFTER the bandpass, so its
    # presence must not move the fringe/bandpass blocks; dispersion/sbd are OFF
    # so no later stage refines the compared slots).
    sol_o = fit(
        CalibrationPipeline(
            FringeFit(model = fm), Bandpass(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(),
        ),
        uvset,
    )
    sol_n = fit(
        CalibrationPipeline(
            FringeFit(model = fm), Bandpass();
            exec = ExecutionConfig(),
        ),
        uvset,
    )

    @testset "θ blocks invariant under the appended smoother stage" begin
        # Stage-B fringe blocks: bit-identical (components 1..5 in both models).
        fn = CAL._step(sol_n, :fringe); fo = CAL._step(sol_o, :fringe)
        for i in 1:5
            @test _blk(fn, i) == _blk(fo, i)
        end
        # Per-channel phase + log-amp bandpass: rtol 1e-12 (fold association).
        bn = CAL._step(sol_n, :bandpass); bo = CAL._step(sol_o, :bandpass)
        @test isapprox(_bp_phase(bn), _bp_phase(bo); rtol = 1.0e-12, atol = 1.0e-12)
        @test any(!=(0), _bp_phase(bn))
        @test isapprox(_bp_amp(bn), _bp_amp(bo); rtol = 1.0e-12, atol = 1.0e-12)
        @test any(!=(0), _bp_amp(bn))
        # Stage provenance: the bandpass stage's own model carries only the
        # bandpass component (nothing merged in from the fringe stage), so it is
        # the sole entry of each group and spans that group's whole θ block.
        @test stage_names(sol_n) == [:fringe, :bandpass]
        @test length(CAL.phase_components(bn.model)) == 1 && length(CAL.logamp_components(bn.model)) == 1
        @test _bp_phase(bn) == _blk(bn, 1) && _bp_amp(bn) == _blk(bn, bn.layout.nphase + 1)
        @test stage_info(sol_n, :bandpass).nscans == length(FP.scan_stream(uvset).groups)
        @test stage_info(sol_n, :bandpass).t_pass > 0
        @test :refine ∉ stage_names(sol_n)     # no DispersionSBDFit step in this pipeline at all
    end

    @testset "new-engine fold is deterministic across ntasks" begin
        sol_n4 = fit(
            CalibrationPipeline(
                FringeFit(model = fm), Bandpass();
                exec = ExecutionConfig(),
            ),
            uvset,
        )
        @test all(a.θ == b.θ for (a, b) in zip(sol_n4.steps, sol_n.steps))
    end

    @testset "steps compose in any declared order" begin
        # No step vetoes its position at construction time — every SolveStep
        # runs in whatever order the pipeline declares. DispersionSBDFit and
        # TemporalSmoother still solve for a fringe-corrected residual, but
        # that is now an assumption of their own solve kernel, not a checked
        # precondition: placed ahead of FringeFit, they fit the UNCORRECTED
        # residual instead and complete without error — a quietly worse fit,
        # not a construction-time rejection.
        solds = fit(CalibrationPipeline(DispersionSBDFit(), FringeFit(model = fm)), uvset)
        @test solds isa CAL.CalibrationSolution
        solts = fit(CalibrationPipeline(TemporalSmoother(), FringeFit(model = fm)), uvset)
        @test solts isa CAL.CalibrationSolution
        # Bandpass's model is self-contained regardless of position,
        # so bandpass-before-fringe was always legal and stays so.
        solbf = fit(CalibrationPipeline(Bandpass(), FringeFit(model = fm)), uvset)
        @test solbf isa CAL.CalibrationSolution
    end

    @testset "CoverageTopup selection" begin
        recs = [
            (; index = 1, source = "C", scan = "1", snr = 50.0, stations = Set([1, 2, 3])),
            (; index = 2, source = "X", scan = "2", snr = 10.0, stations = Set([3, 4])),
            (; index = 3, source = "X", scan = "3", snr = 20.0, stations = Set([2, 4])),
            (; index = 4, source = "Y", scan = "4", snr = 5.0, stations = Set([5])),
        ]
        # Station 4 is missing from scan 1 → its best scan (index 3, snr 20)
        # tops up; station 5 needs scan 4 too.
        @test FP.select_scans(FP.CoverageTopup(FP.ScanIndices(1)), recs) == [1, 3, 4]
        # A selection that already covers everything is unchanged.
        @test FP.select_scans(FP.CoverageTopup(FP.AllScans()), recs) == [1, 2, 3, 4]
        # Records without a stations field pass through untouched.
        recs2 = [(; index = 1, source = "C", scan = "1", snr = 1.0)]
        @test FP.select_scans(FP.CoverageTopup(FP.AllScans()), recs2) == [1]
    end

    @testset "grouped bandpass: ChannelBlocks(k) ties channels within a block" begin
        # How finely the bandpass varies in frequency is said by the SEGMENTATION,
        # so `ChannelBlocks(4)` gives one solved value per 4 consecutive channels
        # of a spw — the channels of a block share a θ parameter, and the
        # component is that many times smaller.
        k = 4
        sol_g = fit(
            CalibrationPipeline(
                FringeFit(model = fm), Bandpass(model = BandpassModel(freq = CAL.ChannelBlocks(k)));
                exec = ExecutionConfig(),
            ),
            uvset,
        )
        bg = CAL._step(sol_g, :bandpass); bn = CAL._step(sol_n, :bandpass)
        @test length(_bp_phase(bg)) * k == length(_bp_phase(bn))
        @test any(!=(0), _bp_phase(bg))

        # The tie is structural: every channel of a block addresses one θ slot,
        # so the evaluated bandpass gain is constant across the block.
        plan = _bp_phase_plan(bg)
        @test plan.fseg_id == repeat(1:(nglob ÷ k), inner = k)
        _, gbp = FP.fringe_bandpass_spectrum(sol_g)
        for a in 1:nant, f in 1:2, b in 1:(nglob ÷ k)
            cs = ((b - 1) * k + 1):(b * k)
            @test all(≈(gbp[first(cs), a, f]), gbp[cs, a, f])
        end
        # ...but the band still has shape: the blocks differ from each other.
        @test !all(≈(gbp[1, 2, 1]), gbp[:, 2, 1])
    end

    @testset "step_solution extraction" begin
        bps = step_solution(sol_n, :bandpass)
        bp = only(bps.steps)
        @test length(bp.model.phase) == 1 && length(bp.model.logamp) == 1
        @test keys(bp.layout.plantree.phase) == (:bandpass,)
        bn = CAL._step(sol_n, :bandpass)
        @test _bp_phase(bp) == _bp_phase(bn)
        @test _bp_amp(bp) == _bp_amp(bn)
        @test collect(bps.info.ant_names) == collect(sol_n.info.ant_names)
        # A solution with no bandpass STEP at all refuses extraction.
        sol_f = fit(FringeFit(model = fm), uvset)
        @test_throws ArgumentError step_solution(sol_f, :bandpass)
        @test_throws "no stage :bandpass" step_solution(sol_f, :bandpass)
    end

    @testset "validate_bandpass rejects a model the estimator cannot solve" begin
        # Both halves off compiles no component at all, so the step would
        # accumulate every scan and write nowhere — rejected at compile time,
        # before any data is read.
        @test_throws ArgumentError fit(Bandpass(model = BandpassModel(phase = false, amp = false)), uvset)
        @test_throws "BandpassModel fits nothing" fit(
            Bandpass(model = BandpassModel(phase = false, amp = false)), uvset,
        )
        # JointALS is stricter: one complex gain per (station, feed, segment)
        # needs both halves, not just one.
        @test_throws "JointALS requires" fit(
            Bandpass(model = BandpassModel(amp = false), estimator = JointALS()), uvset,
        )
    end

    @testset "portable ApplySolution: same-set + cross-set by station name" begin
        bps = step_solution(sol_n, :bandpass)
        bp = only(bps.steps)
        ev = CAL.GainEvaluator(bp.model, bp.layout)

        # Same-set (identical geometry): index-aligned division.
        st0 = FP.scan_stream(uvset)
        stack0, win0 = FP.materialize_cube(st0, st0.groups[1])
        stt = FP.scan_stream(uvset; transforms = (FP.ApplySolution(bps),))
        stackt, _ = FP.materialize_cube(stt, stt.groups[1])
        g = CAL.evaluate_gains(ev, bp.θ, win0.chan_idx, win0.ti_idx)
        Vm = copy(parent(stack0[:vis])); Wm = copy(parent(stack0[:weights]))
        for p in axes(Vm, 4), (bi, (a, b)) in enumerate(baselines(stack0).pairs)
            fa, fb = CAL.correlation_feed_pair(pol_products(stack0)[p])
            for t in axes(Vm, 2), c in axes(Vm, 1)
                den = g[c, t, a, fa] * conj(g[c, t, b, fb])
                (isfinite(den) && abs2(den) > 0) || continue
                Vm[c, t, bi, p] /= den
                Wm[c, t, bi, p] *= abs2(den)
            end
        end
        @test isequal(parent(stackt[:vis]), Vm) && isequal(parent(stackt[:weights]), Wm)

        # Cross-set: a different track (fewer times) with a station subset —
        # the time-constant bandpass ports, stations matched by name.
        uvsub, _ = _build_fringe_uvset(; nant = 3, nspw, nchan, ntime = 5)
        sts = FP.scan_stream(uvsub; transforms = (FP.ApplySolution(bps),))
        @test CAL.build_geometry(uvsub).times != bps.geom.times
        stackx, _ = FP.materialize_cube(sts, sts.groups[1])
        st0s = FP.scan_stream(uvsub)
        stack0s, win0s = FP.materialize_cube(st0s, st0s.groups[1])
        gx = CAL.evaluate_gains(ev, bp.θ, win0s.chan_idx, 1:1)   # A1..A3 ≡ solution rows 1..3
        Vx = copy(parent(stack0s[:vis])); Wx = copy(parent(stack0s[:weights]))
        for p in axes(Vx, 4), (bi, (a, b)) in enumerate(baselines(stack0s).pairs)
            fa, fb = CAL.correlation_feed_pair(pol_products(stack0s)[p])
            for t in axes(Vx, 2), c in axes(Vx, 1)
                den = gx[c, 1, a, fa] * conj(gx[c, 1, b, fb])
                (isfinite(den) && abs2(den) > 0) || continue
                Vx[c, t, bi, p] /= den
                Wx[c, t, bi, p] *= abs2(den)
            end
        end
        @test isequal(parent(stackx[:vis]), Vx) && isequal(parent(stackx[:weights]), Wx)

        # A station the solution never saw keeps identity gains (with a warning).
        uvbig, _ = _build_fringe_uvset(; nant = 5, nspw, nchan, ntime = 5)
        stb = FP.scan_stream(uvbig; transforms = (FP.ApplySolution(bps),))
        @test_logs (:warn, r"A5") match_mode = :any FP.materialize_cube(stb, stb.groups[1])

        # Guard rails: channel-layout mismatch and non-time-constant solutions
        # are rejected at STREAM CONSTRUCTION (fail-fast, before any read).
        uvnc, _ = _build_fringe_uvset(; nant = 3, nspw, nchan = 4, ntime = 5)
        @test_throws ErrorException FP.scan_stream(uvnc; transforms = (FP.ApplySolution(bps),))
        @test_throws ErrorException FP.scan_stream(uvsub; transforms = (FP.ApplySolution(sol_n),))
    end

    @testset "refine kernels: standalone on a scan view (determinism + recovery)" begin
        # VGOS-like dispersive layout (8 sub-bands, wide fractional bandwidth).
        # (Originally gated bit-for-bit against the monolith's concat-cube
        # variants; those died with the monolith at M5 — these ARE the kernels
        # now, so the gates are inner-invariance and truth recovery.)
        dtec_true = [0.0, 3.0, -5.0, 1.5]
        uvd, _ = _build_fringe_uvset(
            nant = 4, nspw = 8, nchan = 8, ref_freq = 3.0e9, spw_sep = 0.5e9,
            dtec = dtec_true, seed = 77, feed_common = true,
        )
        geom = CAL.build_geometry(uvd)
        @test CAL._dispersion_enabled(CAL.DispersionModel(), geom)
        model = FP._fringe_model(
            dispersion = true, sbd_freq_groups = FP.fringe_freq_groups(geom.channel_freqs),
        )
        layout = CAL.plan_parameters(model, 4, geom)
        disp_plan = CAL._dispersion_plan(model, layout)
        ps_delay = FP._perscan_delay_plan(model, layout)
        sbd = FP._sbd_plans(model, layout)
        @test disp_plan !== nothing && ps_delay !== nothing && sbd !== nothing

        st = FP.scan_stream(uvd; geom = geom)
        stack, win = FP.materialize_cube(st, st.groups[1]; executor = SerialScheduler())

        θn = zeros(layout.nθ)
        θ4 = zeros(layout.nθ)
        # `stack` is raw (no transform chain) — the kernel now assumes
        # already-corrected data, and with no prior gains to divide out here
        # that's exactly the raw visibilities.
        nn = FP.refine_scan_dispersion!(θn, stack, win, ps_delay, disp_plan, 1, 4; executor = SerialScheduler())
        n4 = FP.refine_scan_dispersion!(θ4, stack, win, ps_delay, disp_plan, 1, 4; executor = DynamicScheduler(; nchunks = 4))
        # Per-block accumulation ⇒ bit-identical at any inner fan-out.
        @test nn == n4
        @test θn == θ4
        @test any(!=(0), θn)
        # The dispersion column recovers the injected differential dTEC.
        for a in 2:4
            off = plan_off1(disp_plan)[a, 1, 1, 1]
            @test isapprox(θn[off], dtec_true[a] - dtec_true[1]; atol = 0.05)
        end
        FP.refine_scan_sbd!(θn, stack, win, sbd, 1, 4; executor = SerialScheduler())
        FP.refine_scan_sbd!(θ4, stack, win, sbd, 1, 4; executor = DynamicScheduler(; nchunks = 4))
        @test θn == θ4
    end
end
