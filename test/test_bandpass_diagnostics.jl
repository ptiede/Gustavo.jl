# Bandpass diagnostics on synthetic sets with known bandpasses: the normalized
# residuals, the per-station step F test, and the χ² of corrected data.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "bandpass diagnostics" begin
    FR = Gustavo.Fring
    spectra(ps) = FR.baseline_spectra(XRadio.average(ps, XRadio.ByScan()))

    @testset "normalized_residuals" begin
        nchan, τ0 = 32, 3.0e-9
        freqs = 230.0e9 .+ (0:(nchan - 1)) .* 2.0e6
        f0 = sum(freqs) / nchan
        bp = zeros(4, 2, nchan)
        bp[2, :, :] .= reshape(2π .* τ0 .* (freqs .- f0), 1, nchan)
        ps, _ = _build_fringe_ps(; nspw = 1, nchan, nscans = 2, noise = 0.2, antenna_gains = false, bandpass = bp, seed = 5)
        parent(ps[:synth_fringe_1_1][:flag])[:, 3, 1, :] .= true
        s = spectra(ps)

        res = FR.normalized_residuals(s)
        @test keys(res) == (:r, :var, :scan_name)
        @test dims(res.r) == dims(s.visibility)
        @test res.scan_name == s.scan_name
        # Each cut's weighted mean along the axis is 1.
        cut_means(res, w, D) = [
            let r = res.r[cell...], wc = w[cell...], ok = isfinite.(r)
                sum(wc[ok] .* r[ok]) / sum(wc[ok])
            end
                for cell in DimensionalData.DimIndices(DimensionalData.otherdims(res.r, D)) if any(isfinite, res.r[cell...])
        ]
        @test all(≈(1), cut_means(res, s.weight, Frequency))
        @test all(>(0), filter(isfinite, res.var))
        # The delay on A2 stays in the residual; the other baselines are flat.
        @test maximum(abs ∘ angle, filter(isfinite, res.r[AntennaPair = At(("A1", "A2"))])) > 0.4
        @test maximum(abs ∘ angle, filter(isfinite, res.r[AntennaPair = At(("A1", "A3"))])) < 0.1
        # One baseline's channel 3 is flagged in scan 1, on all four feed pairs.
        @test s.scan_name[Ti = 1] == "1"
        @test count(isnan, res.r[Ti = 1, Frequency = 3]) == 4
        @test count(isnan, res.var[Ti = 1, Frequency = 3]) == 4
        @test !any(isnan, res.r[Ti = 2])
        # A processing set gives the same as its spectra.
        @test isequal(FR.normalized_residuals(XRadio.average(ps, XRadio.ByScan())).r, res.r)

        # Along time, on one Measurement Set's integrations.
        ms = ps[:synth_fringe_2_1]
        along = FR.normalized_residuals(ms; dims = Ti)
        layout = FR.baseline_spectra(ms)
        @test dims(along.r) == dims(layout.visibility)
        @test all(≈(1), cut_means(along, layout.weight, Ti))
    end

    @testset "scan_intervals" begin
        ps, truth = _build_fringe_ps(; nscans = 3, ntime = 12)
        iv = FR.scan_intervals(ps)
        @test iv isa Dict{String, Tuple{Float64, Float64}}
        @test iv == Dict(string(s) => (truth.t0_sec + (s - 1) * 420.0, truth.t0_sec + (s - 1) * 420.0 + 330.0) for s in 1:3)
    end

    @testset "step_tests" begin
        nant, nchan, nscans = 4, 32, 8
        rng = StableRNG(7)
        static = 0.3 .* randn(rng, nant, 2, nchan)
        jump = 0.3 .* randn(rng, nchan)
        bp_flat = repeat(static, 1, 1, 1, nscans)
        bp_step = copy(bp_flat)
        bp_step[3, :, :, 5:end] .+= reshape(jump, 1, nchan)
        build(bp) = first(
            _build_fringe_ps(; nant, nspw = 1, nchan, ntime = 12, nscans, noise = 1.0, antenna_gains = false, bandpass = bp, seed = 11)
        )

        ps = build(bp_step)
        iv = FR.scan_intervals(ps)
        res = FR.normalized_residuals(spectra(ps))
        rows = FR.step_tests(res, iv, "A3")
        # The first and last gaps leave one scan on a side, too few to split.
        @test length(rows) == nscans - 3
        @test keys(first(rows)) == (:time, :gap, :F, :chi2_split, :dof_split, :nbaselines)
        @test all(r -> r.gap ≈ 90.0, rows)
        @test all(r -> r.nbaselines == 3 * 2, rows)
        @test issorted(rows; by = r -> r.time)
        k = findfirst(r -> r.time ≈ (iv["4"][2] + iv["5"][1]) / 2, rows)
        @test !isnothing(k)
        @test argmax(r -> r.F, rows) === rows[k]
        @test rows[k].F > 8
        @test rows[k].F > 2 * maximum(r.F for r in rows if r !== rows[k])
        @test rows[k].dof_split == 3 * 2 * nchan * 2 * (nscans - 2)
        # Only one of A1's three baselines carries A3's step.
        @test rows[k].F > FR.step_tests(res, iv, "A1")[k].F

        flat = FR.normalized_residuals(spectra(build(bp_flat)))
        for station in ("A1", "A2", "A3", "A4")
            rows = FR.step_tests(flat, iv, station)
            @test length(rows) == nscans - 3
            @test all(r -> 0.7 < r.F < 1.4, rows)
            @test all(r -> 0.8 < r.chi2_split / r.dof_split < 1.2, rows)
        end
        @test isempty(FR.step_tests(res, iv, "A9"))
    end

    @testset "tss and chi2" begin
        nspw, nchan, ntime, nscans = 2, 16, 12, 4
        ps, _ = _build_fringe_ps(; nspw, nchan, ntime, nscans, noise = 1.0, antenna_gains = false, seed = 3)
        s = spectra(ps)
        parallel = s[FeedPair = At([(1, 1), (2, 2)])]
        everything = (Frequency, Ti, AntennaPair, FeedPair)

        q = FR.tss(parallel)
        @test keys(q) == (:tss, :dof)
        @test dims(q.tss) == DimensionalData.otherdims(parallel.visibility, Frequency)
        @test all(==(2 * (nspw * nchan - 1)), q.dof)
        @test sum(q.tss) / sum(q.dof) ≈ 1 atol = 0.15
        per = sum(q.tss; dims = Ti) ./ sum(q.dof; dims = Ti)
        @test all(x -> 0.6 < x < 1.5, per)
        @test FR.tss(XRadio.average(ps, XRadio.ByScan())) == FR.tss(s)

        # Per integration, across the band.
        ap = FR.tss(ps)
        @test sum(ap.tss) / sum(ap.dof) ≈ 1 atol = 0.15
        # Along time, on one Measurement Set: each channel about its mean over the scan.
        ms = ps[:synth_fringe_1_1]
        along = FR.tss(ms; dims = Ti)
        @test dims(along.tss) == DimensionalData.otherdims(FR.baseline_spectra(ms).visibility, Ti)
        @test sum(along.tss) / sum(along.dof) ≈ 1 atol = 0.15

        # Against its own band means as the model, the χ² is the total sum of
        # squares with the means' 2 degrees of freedom per cut not charged.
        means = zeros(ComplexF64, DimensionalData.otherdims(parallel.visibility, Frequency))
        for cell in DimensionalData.DimIndices(means)
            v, wc = parallel.visibility[cell...], parallel.weight[cell...]
            ok = .!parallel.flag[cell...] .& (wc .> 0)
            means[cell...] = sum(wc[ok] .* v[ok]) / sum(wc[ok])
        end
        c = FR.chi2(parallel, means; dims = everything)
        @test keys(c) == (:chi2, :dof)
        @test c.chi2[] ≈ sum(q.tss)
        @test c.dof[] == sum(q.dof) + 2 * length(means)

        # A model off by a constant raises the χ².
        @test FR.chi2(parallel, 1.1 .* means; dims = everything).chi2[] > c.chi2[]

        # Weights off by a constant factor, then rescaled to inverse variances.
        off, _ = _build_fringe_ps(; nspw, nchan, ntime, nscans, noise = 1.0, antenna_gains = false, seed = 3, weight = 1.0)
        t = FR.tss(spectra(off)[FeedPair = At([(1, 1), (2, 2)])])
        @test sum(t.tss) / sum(t.dof) ≈ 0.5 atol = 0.08
        foreach(ms -> scale_weights!(ms, weight_scales(ms)), values(off))
        t = FR.tss(spectra(off)[FeedPair = At([(1, 1), (2, 2)])])
        @test sum(t.tss) / sum(t.dof) ≈ 1 atol = 0.15

        # A residual bandpass raises it.
        bp = 0.2 .* randn(StableRNG(2), 4, 2, nspw * nchan)
        bad = spectra(first(_build_fringe_ps(; nspw, nchan, ntime, nscans, noise = 1.0, antenna_gains = false, seed = 3, bandpass = bp)))
        b = FR.tss(bad[FeedPair = At([(1, 1), (2, 2)])])
        @test sum(b.tss) / sum(b.dof) > 2
    end
end
