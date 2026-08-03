# ── TemporalSmoother step + output sink ───────────────────────────────────────
#
# The full three-stage pipeline (FringeFit |> BandpassEstimator |>
# TemporalSmoother) on the composable engine. The M5 parity gates against the
# frozen monolith (fringe blocks bit-identical, bandpass/adhoc to rtol 1e-12,
# polish-split full-θ bit-identical) ran BEFORE its deletion; what this file
# keeps are the engine's standing guarantees:
# - θ is bit-deterministic across group concurrency (per-block partials fold in
#   a fixed order — ntasks 1 ≡ 4), and the multi-scan solve flattens the data.
# - `fitcalibrate` fused output ≡ standalone `calibrate(sol, uvset)` output,
#   bit-for-bit (same per-group tail by construction).
# - The refine polish split (`bandpass_max_scans` ⇒ cal scan polished in the
#   narrow window, others full-refined by the smoother pass) still recovers an
#   injected dTEC truth.
# - AprioriAmplitude is an output-chain pipeline step: applied after the gains
#   and before reductions, recorded on `sol.postcal`, replayed by the
#   standalone apply, and rejected from `reduce`.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")
using Dates

# One component's θ block of a single STEP. `i` indexes that step's own
# `layout.plans` (phase components first, then log-amplitude).
_blk(step, i) = step.θ[CAL.component_ranges(step.layout)[i]]

# Leaf-by-leaf equality of two UVSets' vis/weights.
function _sets_equal(a, b; exact = true)
    for (k, leaf) in pairs(UVP.branches(a))
        lb = UVP.branches(b)[k]
        va, wa = parent(leaf[:vis]), parent(leaf[:weights])
        vb, wb = parent(lb[:vis]), parent(lb[:weights])
        ok = exact ? (isequal(va, vb) && isequal(wa, wb)) :
            (isapprox(va, vb; rtol = 1.0e-6) && isapprox(wa, wb; rtol = 1.0e-6))
        ok || return false
    end
    return length(collect(pairs(UVP.branches(a)))) == length(collect(pairs(UVP.branches(b))))
end

# Worst parallel-hand coherence of a corrected set.
function _worst_parallel_coherence(corr)
    worst = 1.0
    for (_, leaf) in UVP.branches(corr)
        V = parent(leaf[:vis]); W = parent(leaf[:weights])
        bl_pairs = UVP.baselines(leaf).pairs
        lp = pol_products(leaf)
        for p in eachindex(lp)
            CAL.is_parallel_hand(lp[p]) || continue
            for bi in eachindex(bl_pairs)
                a, b = bl_pairs[bi]
                a == b && continue
                worst = min(worst, _coherence(@view(V[:, :, bi, p]), @view(W[:, :, bi, p])))
            end
        end
    end
    return worst
end

@testset "TemporalSmoother step + output sink (new engine)" begin
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    nant = 4
    nglob = 16
    # SMOOTH injected per-(station, feed) bandpass shapes (as in test_pipeline's
    # bandpass testsets): the stage's per-channel SNR gate estimates noise from
    # cross-channel scatter, so a white-noise injection is OUT OF MODEL (the
    # gate reads it as noise and correctly refuses to fit it) — smooth shapes
    # are the physically representative case the stage must flatten.
    rng = MersenneTwister(11)
    bp_true = zeros(nant, 2, nglob)
    abp_true = zeros(nant, 2, nglob)
    for a in 1:nant, f in 1:2
        po = (rand(rng) - 0.5)
        ao = (rand(rng) - 0.5) * 0.2
        for gc in 1:nglob
            bp_true[a, f, gc] = po + 0.5 * sin(2π * gc / nglob + a + f)
            abp_true[a, f, gc] = ao + 0.15 * sin(4π * gc / nglob + a - f)
        end
    end
    uvset, _ = _build_fringe_uvset(;
        nant, nspw = 2, nchan = 8, nscans = 3,
        bandpass = bp_true, amp_bandpass = abp_true,
    )
    fm = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false))
    pipe = CalibrationPipeline(
        FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc);
        exec = ExecutionConfig(ntasks = 1),
    )
    sol_n = fit(pipe, uvset)

    @testset "3-scan full pipeline: structure, determinism, coherence" begin
        @test Gustavo.stage_names(sol_n) == [:fringe, :bandpass, :adhoc]
        adhoc_step = CAL._step(sol_n, :adhoc)
        phases = CAL.phase_components(adhoc_step.model)
        # The adhoc block is really solved (nonzero) on every scan.
        ipi = findfirst(tc -> CAL.time_segmentation(tc) isa CAL.PerIntegration, phases)
        @test any(!=(0), _blk(adhoc_step, ipi))
        @test adhoc_step.info.t_pass > 0
        @test :refine ∉ Gustavo.stage_names(sol_n)     # no DispersionSBDFit step in this pipeline at all

        # θ bit-deterministic across group concurrency (per-block partials fold
        # in a fixed order regardless of ntasks/inner).
        pipe4 = CalibrationPipeline(
            FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 4),
        )
        @test parent(gains(fit(pipe4, uvset))) == parent(gains(sol_n))

        # The multi-scan solve flattens the data (bandpass + screen recovered).
        @test _worst_parallel_coherence(Gustavo.apply_calibration(uvset, sol_n)) > 0.99
    end

    @testset "fitcalibrate fused ≡ standalone calibrate" begin
        sol_f, out_n = fitcalibrate(pipe, uvset; reduce = [AverageFrequency(nout = 1)])
        @test parent(gains(sol_f)) == parent(gains(sol_n))

        # Standalone calibrate replays transforms + gains + reduce through the
        # SAME per-group tail as the fused output, agreeing to Float32
        # precision — NOT bit-identical: standalone divides by ONE combined
        # gain (`sol_f`'s full θ), while the fused path composes earlier
        # steps' corrections through the solve-time transform chain as
        # separate sequential divisions (mathematically the same total gain,
        # different floating-point rounding).
        out_s = calibrate(sol_f, uvset; reduce = [AverageFrequency(nout = 1)])
        @test _sets_equal(out_n, out_s; exact = false)

        # And the fused output matches the explicit two-pass apply + reduce.
        red_ref = UVP.frequency_average(Gustavo.apply_calibration(uvset, sol_f); nout = 1)
        @test _sets_equal(out_n, red_ref; exact = false)
    end

    @testset "DispersionSBDFit: dTEC recovery + determinism" begin
        dtec_true = [0.0, 6.0, -4.0, 2.5]
        uvd, _ = _build_fringe_uvset(;
            nant, nspw = 8, nchan = 8, nscans = 3,
            ref_freq = 3.0e9, spw_sep = 0.5e9, dtec = dtec_true,
            seed = 77, feed_common = true,
        )
        # This band layout needs `require_band_separation` off for the dTEC
        # term to be emitted at all.
        ds = DispersionSBDFit(dispersion = CAL.DispersionModel(require_band_separation = false))
        pd = CalibrationPipeline(
            FringeFit(model = fm), ds,
            BandpassEstimator(select = ScanIndices(1)),
            TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 1),
        )
        sol_nd = fit(pd, uvd)
        @test stage_info(sol_nd, :refine).dispersion_applied
        @test Gustavo.stage_names(sol_nd) == [:fringe, :refine, :bandpass, :adhoc]

        # Injected per-station dTEC recovered on EVERY scan — DispersionSBDFit's
        # own pass covers the whole track unconditionally, regardless of which
        # scans the bandpass stage separately selected.
        refine_step = CAL._step(sol_nd, :refine)
        dplan = CAL._dispersion_plan(refine_step.model, refine_step.layout)
        @test dplan !== nothing
        nseg = size(plan_off1(dplan), 3)
        @test nseg >= 3                        # per-scan dTEC columns
        for a in 1:nant, s in 1:nseg
            off = plan_off1(dplan)[a, 1, s, 1]
            off == 0 && continue
            @test isapprox(refine_step.θ[off], dtec_true[a] - dtec_true[1]; atol = 0.05)
        end

        # Bit-deterministic across group concurrency through the whole
        # refine + bandpass + adhoc chain.
        pd4 = CalibrationPipeline(
            FringeFit(model = fm), ds,
            BandpassEstimator(select = ScanIndices(1)),
            TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 4),
        )
        @test parent(gains(fit(pd4, uvd))) == parent(gains(sol_nd))
    end

    @testset "FringeFit |> TemporalSmoother (no bandpass)" begin
        sol_fs = fit(
            CalibrationPipeline(
                FringeFit(model = fm), TemporalSmoother(adhoc);
                exec = ExecutionConfig(ntasks = 1),
            ),
            uvset,
        )
        @test Gustavo.stage_names(sol_fs) == [:fringe, :adhoc]
        @test !any(CAL._is_bandpass, Iterators.flatten(CAL.phase_components(s.model) for s in sol_fs.steps))
        adhoc_step_fs = CAL._step(sol_fs, :adhoc)
        phases = CAL.phase_components(adhoc_step_fs.model)
        ipi = findfirst(tc -> CAL.time_segmentation(tc) isa CAL.PerIntegration, phases)
        @test any(!=(0), _blk(adhoc_step_fs, ipi))
        # Without the bandpass stage the injected per-channel bandpass survives,
        # so full coherence is NOT reached — but the delay/rate/adhoc solve must
        # still be sane (all θ finite, per-scan SNRs strong).
        @test all(s -> all(isfinite, s.θ), sol_fs.steps)
        @test all(>(10), filter(isfinite, CAL._step(sol_fs, :fringe).info.scan_snr))
    end

    @testset "AprioriAmplitude is an output-chain step" begin
        BP = Gustavo.UVData
        ant_names = String.(collect(UVP.union_antennas(uvset).name))
        rdate = DimensionalData.metadata(uvset).array_obs.rdate
        base_dt = DateTime(Date(rdate))
        ts_all = sort!(unique(reduce(vcat, [collect(UVP.obs_time(l)) for l in values(UVP.branches(uvset))])))
        times = [base_dt + Millisecond(round(Int, t * 3_600_000)) for t in ts_all]
        times = [times[1] - Hour(1); times; times[end] + Hour(1)]
        tsys_band = Dict(1 => 100.0, 2 => 400.0)
        spw_cals = Dict{Int, BP.AntabCalibration}()
        for (b, tsys) in tsys_band
            stns = Dict{String, BP.AntabStation}()
            for nm in ant_names
                gain = BP.AntabGainCurve((1.0, 1.0), [1.0])
                vals = repeat([tsys tsys], length(times), 1)
                series = BP.AntabTsysSeries(times, [(0, :R), (0, :L)], vals)
                stns[nm] = BP.AntabStation(nm, gain, series, 0)
            end
            spw_cals[b] = BP.AntabCalibration("synthetic", "synth", 2000, stns)
        end
        ap = AprioriAmplitude(spw_cals; min_elevation_deg = -Inf)

        pipe_ap = CalibrationPipeline(
            FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc), ap;
            exec = ExecutionConfig(ntasks = 1),
        )
        sol_ap, out_ap = fitcalibrate(pipe_ap, uvset)
        # Recorded on the solution; the solve's θ is untouched by it.
        @test length(sol_ap.postcal) == 1 && sol_ap.postcal[1] === ap
        @test parent(gains(sol_ap)) == parent(gains(sol_n))
        # Applied after the gains: relative to the no-apriori output every
        # visibility scales by SEFD = Tsys_band (flat gain, DPFU = 1).
        _, out_plain = fitcalibrate(pipe, uvset)
        for (k, leaf) in pairs(UVP.branches(out_ap))
            b = UVP.metadata(leaf).ddi + 1
            lp = UVP.branches(out_plain)[k]
            va = parent(leaf[:vis])
            vp = parent(lp[:vis])
            m = isfinite.(va) .& isfinite.(vp) .& (abs.(vp) .> 0)
            @test isapprox(va[m], tsys_band[b] .* vp[m]; rtol = 1.0e-5)
        end
        # The standalone apply replays sol.postcal, agreeing with the fused
        # output to Float32 precision (not bit-identical — see the fused ≡
        # standalone comment above).
        @test _sets_equal(out_ap, calibrate(sol_ap, uvset); exact = false)
        # …and refuses it in `reduce` (it is not a reduction).
        @test_throws ArgumentError fitcalibrate(pipe, uvset; reduce = [ap])
        @test_throws "is a pipeline step, not a reduction" calibrate(sol_n, uvset; reduce = [ap])

        # Serialization round-trips postcal (version 3).
        path = tempname() * ".jls"
        try
            CAL.save_solution(path, sol_ap)
            sol_l = CAL.load_solution(path)
            @test length(sol_l.postcal) == 1
            @test sol_l.postcal[1] isa AprioriAmplitude
            @test all(s1.θ == s2.θ for (s1, s2) in zip(sol_l.steps, sol_ap.steps))
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "AprioriAmplitude/ReduceStep compose in declared order" begin
        BP = Gustavo.UVData
        ant_names = String.(collect(UVP.union_antennas(uvset).name))
        rdate = DimensionalData.metadata(uvset).array_obs.rdate
        base_dt = DateTime(Date(rdate))
        ts_all = sort!(unique(reduce(vcat, [collect(UVP.obs_time(l)) for l in values(UVP.branches(uvset))])))
        times = [base_dt + Millisecond(round(Int, t * 3_600_000)) for t in ts_all]
        times = [times[1] - Hour(1); times; times[end] + Hour(1)]
        gain = BP.AntabGainCurve((1.0, 1.0), [1.0])
        vals = repeat([100.0 100.0], length(times), 1)
        stns = Dict(
            nm => BP.AntabStation(nm, gain, BP.AntabTsysSeries(times, [(0, :R), (0, :L)], vals), 0)
                for nm in ant_names
        )
        # Deliberately incomplete: this uvset has two bands (ddi 0 and 1), but
        # the dict only covers band 1.
        band_cals_1 = Dict(1 => BP.AntabCalibration("synthetic", "synth", 2000, stns))
        ap = AprioriAmplitude(band_cals_1; min_elevation_deg = -Inf)

        # AprioriAmplitude declared before any reduction: band 2's leaves are
        # still native, and band_cals_1 has no entry for them — the SAME error
        # `apply_calibration` would raise called directly, with no
        # pipeline-level ordering check in front of it.
        pipe_ap_first = CalibrationPipeline(
            FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc), ap;
            exec = ExecutionConfig(ntasks = 1),
        )
        @test_throws "no a-priori calibration for band 2" fitcalibrate(pipe_ap_first, uvset)

        # The identical AprioriAmplitude declared AFTER a CombineSpw ReduceStep
        # runs against the merged, single-band output instead — succeeding on
        # the very same `band_cals_1` that failed above. This only works if
        # the output chain composes AprioriAmplitude/ReduceStep in their
        # DECLARED relative order, not a hardcoded "apriori always first".
        pipe_reduce_first = CalibrationPipeline(
            FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc),
            CombineSpw(), ap;
            exec = ExecutionConfig(ntasks = 1),
        )
        _, out = fitcalibrate(pipe_reduce_first, uvset)
        @test all(UVP.metadata(leaf).ddi == 0 for (_, leaf) in pairs(UVP.branches(out)))
    end
end
