# Per-station weight correction (`scale_weights!`): a NOISE-ESTIMATE fix
# for correlator weights that are miscalibrated on particular stations. It must
# leave the fringe search alone, move the solve only through the stages that
# weight baselines against each other, and carry into the calibrated weights.
# Reuses `_build_fringe_ps` + the FP/CAL/UVP aliases from test_pipeline.jl
# (included earlier in runtests.jl).

@testset "Per-station weight scale" begin
    ps, _ = _build_fringe_ps()
    ws = DimArray([1.0, 0.5, 0.5, 1.0], AntennaName(["A1", "A2", "A3", "A4"]))
    scale(name) = ws[AntennaName(At(name))]

    @testset "search is invariant; calibrated weights carry the fix" begin
        # The search's SNR is data-driven (noise estimated from the |D|² plane,
        # not from Σw), so a PER-BASELINE weight rescale cancels exactly: the
        # detections — SNR, delay, rate — are bit-identical. What the fix moves
        # is the RELATIVE inter-baseline weighting of the stages that accumulate
        # ACROSS baselines (stage B / bandpass / adhoc) and the calibrated weights.
        gauge = PinAntenna(1)
        steps = (
            BaselineFringeFit(; gauge), Bandpass(; gauge),
            AdhocPhase(; model = default_adhoc_terms(; prior = nothing), gauge),
        )
        base = let sols = _fit_chain(steps, ps)
            (sols, _calibrate_chain(sols, ps))
        end
        scaled = Gustavo.materialize(ps)
        foreach(ms -> scale_weights!(ms, ws), values(scaled))
        fixd = let sols = _fit_chain(steps, scaled)
            (sols, _calibrate_chain(sols, scaled))
        end
        bfr, ffr = base[1][1].steps[:fringe], fixd[1][1].steps[:fringe]
        @test ffr.scan_snr == bfr.scan_snr
        @test ffr.det_snr == bfr.det_snr
        # …so the solution only shifts at the level of the re-weighted stages.
        @test all(
            isapprox(parent(c1.params), parent(c2.params); atol = 1.0e-3)
                for (s1, s2) in zip(fixd[1], base[1]) for (c1, c2) in zip(s1.components, s2.components)
        )

        # The correction factorizes per station, so a baseline with ONE affected
        # station gets s and a baseline between two affected stations gets s² —
        # the 2×/4× pattern of a per-station correlator weight bug.
        for (k, ms) in pairs(fixd[2])
            for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
                a == b && continue
                s = scale(a) * scale(b)
                @test s ≈ (Set((a, b)) == Set(("A2", "A3")) ? 0.25 : ("A2" in (a, b) || "A3" in (a, b)) ? 0.5 : 1.0)
                wb = view(base[2][k][:weight], BaselineID(bi))
                wf = view(ms[:weight], BaselineID(bi))
                @test all(wf .≈ Float32(s) .* wb)
            end
        end
    end
end
