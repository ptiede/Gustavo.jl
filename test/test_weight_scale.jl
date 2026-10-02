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

    @testset "search SNR follows the weights; calibrated weights carry the fix" begin
        # The search's SNR is |D|/√Σw, so rescaling a baseline's weights by s
        # scales its SNR by √s and leaves its peak where it was. The solve moves
        # only through the relative weighting of the stages that combine
        # baselines (station solve, bandpass, adhoc).
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
        for i in eachindex(bfr.det_snr, ffr.det_snr)
            s = scale(bfr.det_ant_a[i]) * scale(bfr.det_ant_b[i])
            @test ffr.det_snr[i] ≈ sqrt(s) * bfr.det_snr[i] rtol = 1.0e-4
            @test ffr.det_delay[i] ≈ bfr.det_delay[i]
        end
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
