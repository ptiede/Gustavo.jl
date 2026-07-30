# ── BandpassEstimator step ────────────────────────────────────────────────────
#
# The bandpass stage on the composable engine. (The M4 parity gates against the
# frozen monolith ran before its deletion.) Standing guarantees:
# - FringeFit |> BandpassEstimator's fringe and per-channel bandpass blocks are
#   invariant under appending a TemporalSmoother stage (later stages never move
#   earlier blocks), and θ is bit-deterministic across group concurrency (the
#   per-scan accumulator contributions fold in group-index order).
# - The refine kernels (dTEC, SBD) are inner-invariant and recover the
#   injected dTEC standalone on a scan view.
# - `bandpass_solution` extracts a portable bandpass-only solution and
#   `ApplySolution` applies it same-set (index-aligned) and cross-set
#   (station-name-mapped, channel-layout-validated, time-constant only).

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

# One component's θ block. `i` indexes `layout.plans` (phase components first,
# then log-amplitude).
_blk(sol, i) = sol.θ[CAL.component_ranges(sol.layout)[i]]
_pc_phase_idx(sol) = findfirst(CAL._is_bandpass, CAL.phase_components(sol.model))
_pc_amp_idx(sol) = findfirst(CAL._is_bandpass, CAL.logamp_components(sol.model))

@testset "BandpassEstimator step (new engine)" begin
    nant, nbands, nchan = 4, 2, 8
    rng = MersenneTwister(11)
    nglob = nbands * nchan
    bp_true = 0.5 .* randn(rng, nant, 2, nglob)
    abp_true = 0.2 .* randn(rng, nant, 2, nglob)
    uvset, _ = _build_fringe_uvset(;
        nant, nbands, nchan, bandpass = bp_true, amp_bandpass = abp_true,
    )
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    fm = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))

    # The fuller-pipeline reference (adhoc is solved AFTER the bandpass, so its
    # presence must not move the fringe/bandpass blocks; dispersion/sbd are OFF
    # so no later stage refines the compared slots).
    sol_o = fit(
        CalibrationPipeline(
            FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 1),
        ),
        uvset,
    )
    sol_n = fit(
        CalibrationPipeline(
            FringeFit(model = fm), BandpassEstimator();
            exec = ExecutionConfig(ntasks = 1),
        ),
        uvset,
    )

    @testset "θ blocks invariant under the appended smoother stage" begin
        # Stage-B fringe blocks: bit-identical (components 1..5 in both models).
        for i in 1:5
            @test _blk(sol_n, i) == _blk(sol_o, i)
        end
        # Per-channel phase + log-amp bandpass: rtol 1e-12 (fold association).
        ipn = _pc_phase_idx(sol_n); ipo = _pc_phase_idx(sol_o)
        @test isapprox(_blk(sol_n, ipn), _blk(sol_o, ipo); rtol = 1.0e-12, atol = 1.0e-12)
        @test any(!=(0), _blk(sol_n, ipn))
        jan = _pc_amp_idx(sol_n); jao = _pc_amp_idx(sol_o)
        @test isapprox(
            _blk(sol_n, sol_n.layout.nphase + jan), _blk(sol_o, sol_o.layout.nphase + jao);
            rtol = 1.0e-12, atol = 1.0e-12,
        )
        @test any(!=(0), _blk(sol_n, sol_n.layout.nphase + jan))
        # Stage provenance: the bandpass stage owns exactly the bandpass blocks.
        @test [r.name for r in sol_n.stages] == [:fringe, :bandpass]
        bprec = sol_n.stages[2]
        @test bprec.phase_comps == [ipn] && bprec.logamp_comps == [jan]
        @test stage_info(sol_n[:bandpass]).nscans == length(FP.scan_stream(uvset).groups)
        @test sol_n.info.t_bandpass_stage > 0
        @test sol_n.info.dispersion_applied == false && sol_n.info.sbd_applied == false
    end

    @testset "new-engine fold is deterministic across ntasks" begin
        sol_n4 = fit(
            CalibrationPipeline(
                FringeFit(model = fm), BandpassEstimator();
                exec = ExecutionConfig(ntasks = 4),
            ),
            uvset,
        )
        @test sol_n4.θ == sol_n.θ
    end

    @testset "max_scans = 1 subset: blocks invariant under the smoother stage" begin
        sol_oc = fit(
            CalibrationPipeline(
                FringeFit(model = fm),
                BandpassEstimator(select = FP.BrightestCalibrator(max_scans = 1)),
                TemporalSmoother(adhoc);
                exec = ExecutionConfig(ntasks = 1),
            ),
            uvset,
        )
        sol_nc = fit(
            CalibrationPipeline(
                FringeFit(model = fm),
                BandpassEstimator(select = FP.BrightestCalibrator(max_scans = 1));
                exec = ExecutionConfig(ntasks = 1),
            ),
            uvset,
        )
        ipn = _pc_phase_idx(sol_nc); ipo = _pc_phase_idx(sol_oc)
        @test _blk(sol_nc, ipn) == _blk(sol_oc, ipo)
        jan = _pc_amp_idx(sol_nc); jao = _pc_amp_idx(sol_oc)
        @test _blk(sol_nc, sol_nc.layout.nphase + jan) == _blk(sol_oc, sol_oc.layout.nphase + jao)
        @test stage_info(sol_nc[:bandpass]).nscans == 1
    end

    @testset "step order honors requires/provides" begin
        @test_throws ArgumentError fit(
            CalibrationPipeline(BandpassEstimator(), FringeFit(model = fm)), uvset)
        @test_throws "BandpassEstimator requires :fringe" fit(
            CalibrationPipeline(BandpassEstimator(), FringeFit(model = fm)), uvset)
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
                FringeFit(model = fm), BandpassEstimator(freq = CAL.ChannelBlocks(k));
                exec = ExecutionConfig(ntasks = 1),
            ),
            uvset,
        )
        ipg = _pc_phase_idx(sol_g)
        @test length(_blk(sol_g, ipg)) * k == length(_blk(sol_n, _pc_phase_idx(sol_n)))
        @test any(!=(0), _blk(sol_g, ipg))

        # The tie is structural: every channel of a block addresses one θ slot,
        # so the evaluated bandpass gain is constant across the block.
        plan = FP._bandpass_plan(sol_g.model, sol_g.layout)
        @test plan.fseg_id == repeat(1:(nglob ÷ k), inner = k)
        _, gbp = FP.fringe_bandpass_spectrum(sol_g)
        for a in 1:nant, f in 1:2, b in 1:(nglob ÷ k)
            cs = ((b - 1) * k + 1):(b * k)
            @test all(≈(gbp[first(cs), a, f]), gbp[cs, a, f])
        end
        # ...but the band still has shape: the blocks differ from each other.
        @test !all(≈(gbp[1, 2, 1]), gbp[:, 2, 1])
    end

    @testset "bandpass_solution extraction" begin
        bps = bandpass_solution(sol_n)
        @test length(bps.model.phase) == 1 && length(bps.model.logamp) == 1
        @test CAL._is_bandpass(bps.model.phase[1])
        ipn = _pc_phase_idx(sol_n)
        jan = _pc_amp_idx(sol_n)
        @test _blk(bps, 1) == _blk(sol_n, ipn)
        @test _blk(bps, bps.layout.nphase + 1) == _blk(sol_n, sol_n.layout.nphase + jan)
        @test collect(bps.info.ant_names) == collect(sol_n.info.ant_names)
        # A solution with no bandpass component refuses extraction.
        sol_f = fit(FringeFit(model = fm), uvset)
        @test_throws ArgumentError bandpass_solution(sol_f)
        @test_throws "carries no bandpass component" bandpass_solution(sol_f)
    end

    @testset "portable ApplySolution: same-set + cross-set by station name" begin
        bps = bandpass_solution(sol_n)
        ev = CAL.GainEvaluator(bps.model, bps.layout)

        # Same-set (identical geometry): index-aligned division.
        st0 = FP.scan_stream(uvset)
        stack0, win0 = FP.materialize_cube(st0, st0.groups[1])
        stt = FP.scan_stream(uvset; transforms = (FP.ApplySolution(bps),))
        stackt, _ = FP.materialize_cube(stt, stt.groups[1])
        g = CAL.evaluate_gains(ev, bps.θ, win0.chan_idx, win0.ti_idx)
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
        uvsub, _ = _build_fringe_uvset(; nant = 3, nbands, nchan, ntime = 5)
        sts = FP.scan_stream(uvsub; transforms = (FP.ApplySolution(bps),))
        @test CAL.build_geometry(uvsub).times != bps.geom.times
        stackx, _ = FP.materialize_cube(sts, sts.groups[1])
        st0s = FP.scan_stream(uvsub)
        stack0s, win0s = FP.materialize_cube(st0s, st0s.groups[1])
        gx = CAL.evaluate_gains(ev, bps.θ, win0s.chan_idx, 1:1)   # A1..A3 ≡ solution rows 1..3
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
        uvbig, _ = _build_fringe_uvset(; nant = 5, nbands, nchan, ntime = 5)
        stb = FP.scan_stream(uvbig; transforms = (FP.ApplySolution(bps),))
        @test_logs (:warn, r"A5") match_mode = :any FP.materialize_cube(stb, stb.groups[1])

        # Guard rails: channel-layout mismatch and non-time-constant solutions
        # are rejected at STREAM CONSTRUCTION (fail-fast, before any read).
        uvnc, _ = _build_fringe_uvset(; nant = 3, nbands, nchan = 4, ntime = 5)
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
            nant = 4, nbands = 8, nchan = 8, ref_freq = 3.0e9, band_sep = 0.5e9,
            dtec = dtec_true, seed = 77, feed_common = true,
        )
        geom = CAL.build_geometry(uvd)
        @test CAL._dispersion_enabled(CAL.DispersionModel(), geom)
        model = FP._fringe_model(
            dispersion = true, sbd_bands = FP.fringe_band_groups(geom.channel_freqs),
        )
        layout = CAL.plan_parameters(model, 4, geom)
        ev = CAL.GainEvaluator(model, layout)
        disp_plan = CAL._dispersion_plan(model, layout)
        ps_delay = FP._perscan_delay_plan(model, layout)
        sbd = FP._sbd_plans(model, layout)
        @test disp_plan !== nothing && ps_delay !== nothing && sbd !== nothing

        st = FP.scan_stream(uvd; geom = geom)
        stack, win = FP.materialize_cube(st, st.groups[1]; executor = SerialScheduler())

        θn = zeros(layout.nθ)
        θ4 = zeros(layout.nθ)
        nn = FP.refine_scan_dispersion!(θn, stack, win, ev, ps_delay, disp_plan, 1, 4; executor = SerialScheduler())
        n4 = FP.refine_scan_dispersion!(θ4, stack, win, ev, ps_delay, disp_plan, 1, 4; executor = DynamicScheduler(; nchunks = 4))
        # Per-block accumulation ⇒ bit-identical at any inner fan-out.
        @test nn == n4
        @test θn == θ4
        @test any(!=(0), θn)
        # The dispersion column recovers the injected differential dTEC.
        for a in 2:4
            off = disp_plan.off1[a, 1, 1, 1]
            @test isapprox(θn[off], dtec_true[a] - dtec_true[1]; atol = 0.05)
        end
        FP.refine_scan_sbd!(θn, stack, win, ev, sbd, 1, 4; executor = SerialScheduler())
        FP.refine_scan_sbd!(θ4, stack, win, ev, sbd, 1, 4; executor = DynamicScheduler(; nchunks = 4))
        @test θn == θ4
    end
end
