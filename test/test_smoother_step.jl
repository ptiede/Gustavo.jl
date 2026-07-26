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

# One component's θ block. `i` indexes `layout.plans` (phase components first,
# then log-amplitude).
_blk(sol, i) = sol.θ[CAL.component_ranges(sol.layout)[i]]

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
        nant, nbands = 2, nchan = 8, nscans = 3,
        bandpass = bp_true, amp_bandpass = abp_true,
    )
    fm = FringeModel(dispersion = false, sbd = false)
    pipe = CalibrationPipeline(
        FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc);
        exec = ExecutionConfig(ntasks = 1),
    )
    sol_n = fit(pipe, uvset)

    @testset "3-scan full pipeline: structure, determinism, coherence" begin
        @test Gustavo.stage_names(sol_n) == [:fringe, :bandpass, :adhoc]
        phases = collect(sol_n.model.phase)
        # The adhoc block is really solved (nonzero) on every scan.
        ipi = findfirst(tc -> CAL.time_segmentation(tc) isa CAL.PerIntegration, phases)
        @test any(!=(0), _blk(sol_n, ipi))
        @test sol_n.info.t_adhoc_pass > 0
        @test sol_n.info.dispersion_applied == false

        # θ bit-deterministic across group concurrency (per-block partials fold
        # in a fixed order regardless of ntasks/inner).
        pipe4 = CalibrationPipeline(
            FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 4),
        )
        @test fit(pipe4, uvset).θ == sol_n.θ

        # The multi-scan solve flattens the data (bandpass + screen recovered).
        @test _worst_parallel_coherence(Gustavo.apply_calibration(uvset, sol_n)) > 0.99
    end

    @testset "fitcalibrate fused ≡ standalone calibrate" begin
        sol_f, out_n = fitcalibrate(pipe, uvset; reduce = [AverageFrequency(nout = 1)])
        @test sol_f.θ == sol_n.θ

        # Standalone calibrate replays transforms + gains + reduce through the
        # SAME per-group tail — bit-identical to the fused output.
        out_s = calibrate(sol_f, uvset; reduce = [AverageFrequency(nout = 1)])
        @test _sets_equal(out_n, out_s)

        # And the fused output matches the explicit two-pass apply + reduce.
        red_ref = UVP.frequency_average(Gustavo.apply_calibration(uvset, sol_f); nout = 1)
        @test _sets_equal(out_n, red_ref; exact = false)
    end

    @testset "polish split: dTEC recovery + determinism" begin
        dtec_true = [0.0, 6.0, -4.0, 2.5]
        uvd, _ = _build_fringe_uvset(;
            nant, nbands = 8, nchan = 8, nscans = 3,
            ref_freq = 3.0e9, band_sep = 0.5e9, dtec = dtec_true,
            seed = 77, feed_common = true,
        )
        # bandpass capped to the single best calibrator scan: that scan's
        # dTEC/SBD columns take the narrow polish window in the smoother pass
        # (`reuse_bandpass_refine`), the other scans the full grid.
        pd = CalibrationPipeline(
            FringeFit(model = FringeModel(dispersion = true, sbd = :auto)),
            BandpassEstimator(select = BrightestCalibrator(max_scans = 1)),
            TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 1),
        )
        sol_nd = fit(pd, uvd)
        @test sol_nd.info.dispersion_applied

        # Injected per-station dTEC recovered on EVERY scan through the split.
        dplan = FP._dispersion_plan(sol_nd.model, sol_nd.layout)
        @test dplan !== nothing
        nseg = size(dplan.off1, 3)
        @test nseg >= 3                        # per-scan dTEC columns
        for a in 1:nant, s in 1:nseg
            off = dplan.off1[a, 1, s, 1]
            off == 0 && continue
            @test isapprox(sol_nd.θ[off], dtec_true[a] - dtec_true[1]; atol = 0.05)
        end

        # Bit-deterministic across group concurrency through the whole
        # bandpass-refine + polish + adhoc chain.
        pd4 = CalibrationPipeline(
            FringeFit(model = FringeModel(dispersion = true, sbd = :auto)),
            BandpassEstimator(select = BrightestCalibrator(max_scans = 1)),
            TemporalSmoother(adhoc);
            exec = ExecutionConfig(ntasks = 4),
        )
        @test fit(pd4, uvd).θ == sol_nd.θ
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
        phases = collect(sol_fs.model.phase)
        @test !any(tc -> CAL.term(tc) isa CAL.PerChannel, phases)
        ipi = findfirst(tc -> CAL.time_segmentation(tc) isa CAL.PerIntegration, phases)
        @test any(!=(0), _blk(sol_fs, ipi))
        # Without the bandpass stage the injected per-channel bandpass survives,
        # so full coherence is NOT reached — but the delay/rate/adhoc solve must
        # still be sane (all θ finite, per-scan SNRs strong).
        @test all(isfinite, sol_fs.θ)
        @test all(>(10), filter(isfinite, sol_fs.info.scan_max_snr))
    end

    @testset "AprioriAmplitude is an output-chain step" begin
        BP = Gustavo.Bandpass
        ant_names = String.(collect(UVP.union_antennas(uvset).name))
        rdate = DimensionalData.metadata(uvset).array_obs.rdate
        base_dt = DateTime(Date(rdate))
        ts_all = sort!(unique(reduce(vcat, [collect(UVP.obs_time(l)) for l in values(UVP.branches(uvset))])))
        times = [base_dt + Millisecond(round(Int, t * 3_600_000)) for t in ts_all]
        times = [times[1] - Hour(1); times; times[end] + Hour(1)]
        tsys_band = Dict(1 => 100.0, 2 => 400.0)
        band_cals = Dict{Int, BP.AntabCalibration}()
        for (b, tsys) in tsys_band
            stns = Dict{String, BP.AntabStation}()
            for nm in ant_names
                gain = BP.AntabGainCurve((1.0, 1.0), [1.0])
                vals = repeat([tsys tsys], length(times), 1)
                series = BP.AntabTsysSeries(times, [(0, :R), (0, :L)], vals)
                stns[nm] = BP.AntabStation(nm, gain, series, 0)
            end
            band_cals[b] = BP.AntabCalibration("synthetic", "synth", 2000, stns)
        end
        ap = AprioriAmplitude(band_cals; min_elevation_deg = -Inf)

        pipe_ap = CalibrationPipeline(
            FringeFit(model = fm), BandpassEstimator(), TemporalSmoother(adhoc), ap;
            exec = ExecutionConfig(ntasks = 1),
        )
        sol_ap, out_ap = fitcalibrate(pipe_ap, uvset)
        # Recorded on the solution; the solve's θ is untouched by it.
        @test length(sol_ap.postcal) == 1 && sol_ap.postcal[1] === ap
        @test sol_ap.θ == sol_n.θ
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
        # The standalone apply replays sol.postcal — identical to the fused out.
        @test _sets_equal(out_ap, calibrate(sol_ap, uvset))
        # …and refuses it in `reduce` (it is not a reduction).
        @test_throws ErrorException fitcalibrate(pipe, uvset; reduce = [ap])
        @test_throws ErrorException calibrate(sol_n, uvset; reduce = [ap])

        # Serialization round-trips postcal (version 3).
        path = tempname() * ".jls"
        try
            CAL.save_solution(path, sol_ap)
            sol_l = CAL.load_solution(path)
            @test length(sol_l.postcal) == 1
            @test sol_l.postcal[1] isa AprioriAmplitude
            @test sol_l.θ == sol_ap.θ
        finally
            isfile(path) && rm(path)
        end
    end
end
