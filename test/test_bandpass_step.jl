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
# - selection (`sol[:bandpass]`) extracts a portable bandpass-only solution and
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
        # Stage-B fringe blocks: bit-identical (components 1..4 in both models).
        fn = sol_n[:fringe].steps[1]; fo = sol_o[:fringe].steps[1]
        for i in 1:4
            @test _blk(fn, i) == _blk(fo, i)
        end
        # Per-channel phase + log-amp bandpass: rtol 1e-12 (fold association).
        bn = sol_n[:bandpass].steps[1]; bo = sol_o[:bandpass].steps[1]
        @test isapprox(_bp_phase(bn), _bp_phase(bo); rtol = 1.0e-12, atol = 1.0e-12)
        @test any(!=(0), _bp_phase(bn))
        @test isapprox(_bp_amp(bn), _bp_amp(bo); rtol = 1.0e-12, atol = 1.0e-12)
        @test any(!=(0), _bp_amp(bn))
        # Stage provenance: the bandpass stage's own model carries only the
        # bandpass component (nothing merged in from the fringe stage), so it is
        # the sole entry of each group and spans that group's whole θ block.
        @test keys(sol_n) == [:fringe, :bandpass]
        @test length(CAL.phase_components(bn.model)) == 1 && length(CAL.logamp_components(bn.model)) == 1
        @test _bp_phase(bn) == _blk(bn, 1) && _bp_amp(bn) == _blk(bn, bn.layout.nphase + 1)
        @test stage_info(sol_n, :bandpass).nscans == length(FP.scan_stream(uvset).groups)
        @test stage_info(sol_n, :bandpass).t_pass > 0
        @test :refine ∉ keys(sol_n)     # no DispersionSBDFit step in this pipeline at all
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
                FringeFit(model = fm), Bandpass(model = default_bandpass_terms(freq = CAL.ChannelBlocks(k)));
                exec = ExecutionConfig(),
            ),
            uvset,
        )
        bg = sol_g[:bandpass].steps[1]; bn = sol_n[:bandpass].steps[1]
        @test length(_bp_phase(bg)) * k == length(_bp_phase(bn))
        @test any(!=(0), _bp_phase(bg))

        # The tie is structural: every channel of a block addresses one θ slot,
        # so the evaluated bandpass gain is constant across the block.
        plan = _bp_phase_plan(bg)
        @test plan.fseg_id == repeat(1:(nglob ÷ k), inner = k)
        gbp = parent(gains(sol_g[:bandpass, :phase, :bandpass]; Ti = 1))
        for a in 1:nant, f in 1:2, b in 1:(nglob ÷ k)
            cs = ((b - 1) * k + 1):(b * k)
            @test all(≈(gbp[first(cs), a, f]), gbp[cs, a, f])
        end
        # ...but the band still has shape: the blocks differ from each other.
        @test !all(≈(gbp[1, 2, 1]), gbp[:, 2, 1])
    end

    @testset "step-selection extraction" begin
        bps = sol_n[:bandpass]
        bp = only(bps.steps)
        @test length(bp.model.phase) == 1 && length(bp.model.logamp) == 1
        @test keys(bp.layout.plantree.phase) == (:bandpass,)
        bn = sol_n[:bandpass].steps[1]
        @test _bp_phase(bp) == _bp_phase(bn)
        @test _bp_amp(bp) == _bp_amp(bn)
        @test collect(bps.info.ant_names) == collect(sol_n.info.ant_names)
        # A solution with no bandpass STEP at all refuses extraction.
        sol_f = fit(FringeFit(model = fm), uvset)
        @test_throws ArgumentError sol_f[:bandpass]
        @test_throws "no stage :bandpass" sol_f[:bandpass]
    end

    @testset "compile-time model vetting (can_fit / validate_model)" begin
        pertrack = FP.PerTrackSmoother()
        phase_only = (; phase = default_bandpass_terms().phase)
        # An empty tree compiles no component at all, so the step would
        # accumulate every scan and write nowhere — rejected at compile time,
        # before any data is read.
        @test_throws ArgumentError fit(Bandpass(model = (;), smoother = pertrack), uvset)
        @test_throws "fits nothing" fit(Bandpass(model = (;), smoother = pertrack), uvset)
        # JointSmoother is stricter: one complex gain per (station, feed, segment)
        # needs both observables, not just one — so it rejects a model
        # PerTrackSmoother would happily solve.
        @test_throws "JointSmoother requires" fit(Bandpass(model = phase_only), uvset)
        @test fit(Bandpass(model = phase_only, smoother = pertrack), uvset) isa
            CAL.CalibrationSolution
        # ...and its two components must share one frequency segmentation.
        mixed = (;
            phase = (; bandpass = FP._bandpass_component(CAL.ChannelBlocks(1))),
            logamp = (; bandpass = FP._bandpass_component(CAL.ChannelBlocks(2))),
        )
        @test_throws "share one frequency segmentation" model_components(
            Bandpass(model = mixed), nothing,
        )
        # A component the smoothers' θ writes cannot address (here: a Delay
        # term) is rejected by `can_fit`, naming the component.
        delay_model = (;
            phase = (;
                bandpass = CAL.GainComponent(
                    CAL.Delay(); Ti = CAL.GlobalTime(),
                    Frequency = CAL.ChannelBlocks(1), Feed = CAL.PerFeed(),
                ),
            ),
        )
        @test_throws "cannot fit the component" model_components(
            Bandpass(model = delay_model), nothing,
        )
        # One track set per observable: a second component in a group is
        # structurally unsolvable, whatever its form.
        doubled = (;
            phase = (;
                bandpass = FP._bandpass_component(CAL.ChannelBlocks(1)),
                ripple = FP._bandpass_component(CAL.ChannelBlocks(4)),
            ),
        )
        @test_throws "at most one" model_components(Bandpass(model = doubled), nothing)
        # The tree's only legal top-level keys are the two groups — a component
        # name written at the top level is the likely mistake.
        @test_throws "unexpected key" model_components(
            Bandpass(model = (; bandpass = FP._bandpass_component(CAL.ChannelBlocks(1)))),
            nothing,
        )
        # Anything that is neither a tree nor a StationGainModel is rejected
        # with the expected forms named.
        @test_throws "must be a `StationGainModel`" model_components(
            Bandpass(model = CAL.ChannelBlocks(1)), nothing,
        )
        # A StationGainModel is accepted verbatim as the model argument.
        sgm = CAL.StationGainModel(; default_bandpass_terms()...)
        @test model_components(Bandpass(model = sgm), nothing) ==
            model_components(Bandpass(), nothing)

        # A `stations` entry is vetted like the base, with the can_fit error
        # naming the station whose entry carries the unfittable component.
        bad_entry = CAL.StationGainModel(;
            default_bandpass_terms()...,
            stations = (
                A1 = (;
                    phase = (;
                        bandpass = CAL.GainComponent(
                            CAL.Delay(); Ti = CAL.GlobalTime(),
                            Frequency = CAL.ChannelBlocks(1), Feed = CAL.PerFeed(),
                        ),
                    ),
                ),
            ),
        )
        @test_throws "station :A1 entry" model_components(
            Bandpass(model = bad_entry), nothing,
        )

        # A heterogeneous model that vets cleanly still cannot reach a step
        # that has not opted in (`supports_station_heterogeneity` defaults to
        # false): the runner rejects it at compile time, naming the differing
        # component and its per-station signatures.
        @test !Gustavo.supports_station_heterogeneity(Bandpass())
        het = CAL.StationGainModel(;
            default_bandpass_terms()...,
            stations = (
                A1 = (;
                    phase = (; bandpass = FP._bandpass_component(CAL.ChannelBlocks(4))),
                    logamp = (; bandpass = FP._bandpass_component(CAL.ChannelBlocks(4))),
                ),
            ),
        )
        @test model_components(Bandpass(model = het), nothing) isa CAL.StationGainModel
        @test_throws "station-uniform" fit(Bandpass(model = het), uvset)
        @test_throws "phase.bandpass" fit(Bandpass(model = het), uvset)
    end

    @testset "PerTrackSmoother: a shape on both observables" begin
        # θ slot for one (station, feed, GLOBAL channel) of a bandpass component.
        bpθ(θ, plan, a, f, gc) =
            (off = plan_off1(plan)[a, f, 1, plan.fseg_id[gc]]; off == 0 ? NaN : θ[off])

        ffb = FringeFit(model = fm)
        sol_free = fit(
            ffb |> Bandpass(smoother = FP.PerTrackSmoother(phase = FP.FreeShape(), amp = FP.FreeShape())),
            uvset,
        )
        sfree = sol_free[:bandpass].steps[1]

        # A stiff roughness penalty on the PHASE leaves only the second-difference
        # null space — a straight line in frequency, per spw. Phase carried no shape
        # hook at all before, so this is the capability the spec pair adds.
        sol_stiff = fit(
            ffb |> Bandpass(smoother = FP.PerTrackSmoother(phase = FP.WhittakerShape(1.0e8))),
            uvset,
        )
        sstiff = sol_stiff[:bandpass].steps[1]
        pplan = _bp_phase_plan(sstiff)
        for a in 1:nant, f in 1:2, s in 1:nspw
            gcs = ((s - 1) * nchan + 1):(s * nchan)
            trk = [bpθ(sstiff.θ, pplan, a, f, gc) for gc in gcs]
            all(isfinite, trk) || continue
            @test maximum(abs, diff(diff(CAL.unwrap_phase_track(trk)))) < 1.0e-3
        end
        # ...and the free fit is genuinely rougher, so the flatness above is the
        # spec acting rather than a featureless track.
        pplan_f = _bp_phase_plan(sfree)
        rough = Float64[]
        for a in 1:nant, f in 1:2, s in 1:nspw
            gcs = ((s - 1) * nchan + 1):(s * nchan)
            trk = [bpθ(sfree.θ, pplan_f, a, f, gc) for gc in gcs]
            all(isfinite, trk) || continue
            push!(rough, maximum(abs, diff(diff(CAL.unwrap_phase_track(trk)))))
        end
        @test !isempty(rough)
        @test maximum(rough) > 0.1

        # The zero band-mean log-amp gauge is applied AFTER the shape fit, so a spec
        # that rewrites every segment (this solve's default `WhittakerShape(1.0)`
        # amp) still leaves the bandpass SHAPE only.
        aplan = _bp_amp_plan(sstiff)
        for a in 1:nant, f in 1:2
            la = [bpθ(sstiff.θ, aplan, a, f, gc) for gc in 1:nglob]
            @test all(isfinite, la)
            @test abs(sum(la) / length(la)) < 1.0e-8
        end
    end

    @testset "gate → spike guard → shape: a contaminated channel is excised, then estimated" begin
        # The synthetic visibilities carry no thermal noise, so an unconstrained
        # per-segment solve reproduces the injected bandpass exactly and nothing
        # about rejection would be exercised. The case that does exercise it is a
        # channel the multiplicative gain model does NOT describe: one narrowband
        # contaminant, on the baselines of a single station. It shows as EXCESS
        # amplitude, so the Fisher weight of the very segment carrying it is the
        # LARGEST on the track — no Gaussian prior can outvote it, and rejection
        # is the narrow-spike guard's job, not the spec's. What the spec then
        # supplies is the estimate of the excised channel.
        nspwc, nchanc, ntimec, nscansc = 1, 32, 4, 2
        nglobc = nspwc * nchanc
        chan_bw = 2.0e6
        rngc = MersenneTwister(0x000A51DE)
        # Injected truth: an OU (AR(1)) draw along frequency per (station, feed) —
        # the process `ARShape` assumes, correlated over 6 channels.
        nu = 6 * chan_bw
        sigma_bp = 0.15
        a1 = exp(-chan_bw / nu)
        bpc = zeros(nant, 2, nglobc)
        abpc = zeros(nant, 2, nglobc)
        for trk in (bpc, abpc), a in 1:nant, f in 1:2
            trk[a, f, 1] = sigma_bp * randn(rngc)
            for c in 2:nglobc
                trk[a, f, c] = a1 * trk[a, f, c - 1] + sqrt(sigma_bp^2 * (1 - a1^2)) * randn(rngc)
            end
        end
        uvc, truthc = _build_fringe_uvset(;
            nant, nspw = nspwc, nchan = nchanc, ntime = ntimec, nscans = nscansc,
            chan_bw, bandpass = bpc, amp_bandpass = abpc, seed = 7,
        )
        bad_chan, bad_ant = 17, 2
        for (_, leaf) in DimensionalData.branches(uvc)
            for (bi, (a, b)) in enumerate(truthc.bl_pairs)
                (a == bad_ant || b == bad_ant) || continue
                leaf[:vis][bad_chan, :, bi, :] .*= 8.0f0
            end
        end

        fmc = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))
        runc(sm) = fit(
            CalibrationPipeline(
                FringeFit(model = fmc), Bandpass(smoother = sm); exec = ExecutionConfig(),
            ), uvc,
        )[:bandpass].steps[1]
        track(s, a, f) = (
            L = CAL._component_leaf(s.layout.plantree.logamp.bandpass, s.θ);
            Float64[L[1, f, c, 1, a] for c in 1:nglobc]
        )
        gauge(x) = x .- sum(x) / length(x)
        cleanc = [c for c in 1:nglobc if abs(c - bad_chan) > 2]
        rms_clean(v, a, f) =
            sqrt(sum(abs2, gauge(v)[cleanc] .- gauge(abpc[a, f, :])[cleanc]) / length(cleanc))

        s_free = runc(FP.PerTrackSmoother(amp = FP.FreeShape()))
        s_ar = runc(FP.PerTrackSmoother(amp = FP.ARShape(nu)))
        s_wh = runc(FP.PerTrackSmoother(amp = FP.WhittakerShape(1.0)))

        # The guard excises the contaminated segment; `FreeShape` estimates
        # nothing it has no datum for, so that slot stays UNAPPLIED (log-amp 0)
        # while every uncontaminated channel is recovered exactly.
        vfree = track(s_free, bad_ant, 1)
        @test vfree[bad_chan] == 0.0
        @test rms_clean(vfree, bad_ant, 1) < 1.0e-2
        for a in (1, 3, 4)                       # stations the contaminant never touched
            @test rms_clean(track(s_free, a, 1), a, 1) < 1.0e-6
        end

        # A spec that estimates gaps fills that slot from the in-band shape, and
        # lands far closer to the truth than leaving it unapplied did.
        var = track(s_ar, bad_ant, 1)
        err_at(v) = abs(gauge(v)[bad_chan] - gauge(abpc[bad_ant, 1, :])[bad_chan])
        @test var[bad_chan] != 0.0
        @test err_at(var) < 0.5 * err_at(vfree)

        # The AR prior is the correctly-specified one for this track (the truth IS
        # an OU draw along frequency), so on the uncontaminated channels it costs
        # less than the shape-agnostic roughness penalty.
        ar_rms = mean(rms_clean(track(s_ar, a, f), a, f) for a in 1:nant, f in 1:2)
        wh_rms = mean(rms_clean(track(s_wh, a, f), a, f) for a in 1:nant, f in 1:2)
        @test ar_rms < 0.85 * wh_rms
    end

    @testset "portable ApplySolution: same-set + cross-set by station name" begin
        bps = sol_n[:bandpass]
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

        # A time-VARYING solution ports the same way: `sol_n`'s fringe terms are
        # per scan, and `uvsub` is the same scan sampled over fewer APs, so every
        # target sample places in the scan it belongs to.
        stn = FP.scan_stream(uvsub; transforms = (FP.ApplySolution(sol_n),))
        @test FP.materialize_cube(stn, stn.groups[1]) isa Tuple

        # Guard rails: a bandpass cuts the channel-INDEX axis, so a set indexing
        # different channels has no segment to place against…
        uvnc, _ = _build_fringe_uvset(; nant = 3, nspw, nchan = 4, ntime = 5)
        stnc = FP.scan_stream(uvnc; transforms = (FP.ApplySolution(bps),))
        @test_throws "identical channel layout" FP.materialize_cube(stnc, stnc.groups[1])
        # …and a scan the solve never saw is refused, not served by a neighbour.
        uv2, _ = _build_fringe_uvset(; nant = 3, nspw, nchan, ntime = 5, nscans = 2)
        st2 = FP.scan_stream(uv2; transforms = (FP.ApplySolution(sol_n),))
        @test_throws "is not in the solution" [
            FP.materialize_cube(st2, g) for g in st2.groups
        ]
        # Station identity is what construction checks, and a name is all of it.
        @test_throws "shares no station with this set" FP.scan_stream(
            uvsub;
            transforms = (
                FP.ApplySolution(
                    CAL.CalibrationSolution(
                        bps.steps, bps.geom, merge(bps.info, (; ant_names = ["QQ", "RR"])),
                    ),
                ),
            ),
        )
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
        model = _full_fringe_model(
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
        nn = FP.refine_scan_dispersion!(θn, stack, win, ps_delay, disp_plan, PinAntenna(1), 4; executor = SerialScheduler())
        n4 = FP.refine_scan_dispersion!(θ4, stack, win, ps_delay, disp_plan, PinAntenna(1), 4; executor = DynamicScheduler(; nchunks = 4))
        # Per-block accumulation ⇒ bit-identical at any inner fan-out.
        @test nn == n4
        @test θn == θ4
        @test any(!=(0), θn)
        # The dispersion column recovers the injected differential dTEC.
        for a in 2:4
            off = plan_off1(disp_plan)[a, 1, 1, 1]
            @test isapprox(θn[off], dtec_true[a] - dtec_true[1]; atol = 0.05)
        end
        FP.refine_scan_sbd!(θn, stack, win, sbd, PinAntenna(1), 4; executor = SerialScheduler())
        FP.refine_scan_sbd!(θ4, stack, win, sbd, PinAntenna(1), 4; executor = DynamicScheduler(; nchunks = 4))
        @test θn == θ4
    end
end

# ── Per-track outcome reporting and the undetermined-branch gate ──────────────
#
# θ records an unfitted bandpass track as unit gain and a starved one as a
# constant, neither distinguishable there from a genuinely flat response. These
# cover the record that makes the difference visible, and the gate that keeps an
# undetermined phase branch from being reported as a measured ramp.

@testset "Bandpass: per-track outcome codes" begin
    rng = MersenneTwister(24601)
    nband, nchan = 4, 32
    seg_spw = repeat(1:nband; inner = nchan)
    seg_freq = collect(range(8.6e10, 8.6e10 + 1.28e8; length = nband * nchan))
    spec = FP.ARShape(1.6e7)

    # A clean, structured track: every band solved.
    smooth = 0.3 .* sin.(range(0, 6π; length = nband * nchan))
    w = fill(1.0e4, nband * nchan)
    st = fill(Int8(-1), nband)
    FP._fit_track_bands(spec, smooth, w, seg_spw, seg_freq; unwrap = true, status = st)
    @test all(==(FP._BP_TRACK_SOLVED), st)

    # A band with no usable segment estimates nothing and says so.
    gappy = copy(smooth)
    wg = copy(w)
    gappy[1:nchan] .= NaN
    wg[1:nchan] .= 0
    st = fill(Int8(-1), nband)
    out = FP._fit_track_bands(spec, gappy, wg, seg_spw, seg_freq; unwrap = true, status = st)
    @test st[1] == FP._BP_TRACK_NODATA
    @test all(isnan, out[1:nchan])

    # A constant band is fit, but carries no shape — reported as flat rather than
    # passed off as a measured response.
    flat = fill(0.2, nband * nchan)
    st = fill(Int8(-1), nband)
    FP._fit_track_bands(spec, flat, w, seg_spw, seg_freq; unwrap = true, status = st)
    @test all(==(FP._BP_TRACK_FLAT), st)

    # Phase noise past a radian leaves the 2π branch undetermined: the band is
    # declined, not fit, so nothing is written for it and no invented trend can
    # reach θ. The amplitude path is never unwrapped and so is never declined.
    noisy = 2.0 .* randn(rng, nband * nchan)
    st = fill(Int8(-1), nband)
    out = FP._fit_track_bands(spec, noisy, w, seg_spw, seg_freq; unwrap = true, status = st)
    @test all(==(FP._BP_TRACK_DECLINED), st)
    @test all(isnan, out)
    st = fill(Int8(-1), nband)
    FP._fit_track_bands(spec, noisy, w, seg_spw, seg_freq; unwrap = false, status = st)
    @test !any(==(FP._BP_TRACK_DECLINED), st)
end

@testset "bandpass_track_report: counts and the unfitted observable" begin
    ph = Int8[FP._BP_TRACK_SOLVED FP._BP_TRACK_FLAT; FP._BP_TRACK_NODATA FP._BP_TRACK_DECLINED]
    phase_status = reshape(ph, 2, 2, 1)
    rep = FP.bandpass_track_report(phase_status, nothing, [3])
    @test (rep.n_solved, rep.n_flat, rep.n_nodata, rep.n_declined) == (1, 1, 1, 1)
    @test rep.band_ids == [3]
    @test rep.track_labels[FP._BP_TRACK_SOLVED + 1] == "solved"
    # An observable that was not fit is an EMPTY status, not a missing one: the
    # record is serialized with the solution and every field must carry a value.
    @test rep.amp_status isa AbstractArray && isempty(rep.amp_status)
    @test rep.phase_status === phase_status
end

@testset "Bandpass: the solve publishes its per-track record" begin
    nant, nspw, nchan, ntime, nscans = 4, 2, 8, 6, 4
    rng = MersenneTwister(777)
    nglob = nspw * nchan
    uvset, _ = _build_fringe_uvset(;
        nant, nspw, nchan, ntime, nscans,
        bandpass = 0.3 .* randn(rng, nant, 2, nglob),
        amp_bandpass = 0.1 .* randn(rng, nant, 2, nglob),
        seed = 5,
    )
    sol = fit(
        CalibrationPipeline(
            FringeFit(model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))),
            Bandpass();
            exec = ExecutionConfig(),
        ),
        uvset,
    )
    info = stage_info(sol, :bandpass)
    @test size(info.phase_status) == (nant, 2, nspw)
    @test size(info.amp_status) == (nant, 2, nspw)
    @test info.band_ids == collect(1:nspw)
    total = info.n_solved + info.n_flat + info.n_declined + info.n_nodata
    @test total == 2 * nant * 2 * nspw
    # High-SNR synthetic data with real injected structure: the tracks are
    # measured, not placeholders.
    @test info.n_solved > total ÷ 2
    @test info.n_declined == 0
end
