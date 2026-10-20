# Per-station weight correction (`scale_weights!`): a NOISE-ESTIMATE fix
# for correlator weights that are miscalibrated on particular stations. It must
# leave the fringe search alone, move the solve only through the stages that
# weight baselines against each other, and carry into the calibrated weights.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "Per-antenna weight scale" begin
    ps, _ = _build_fringe_ps()
    ws = Dict("A1" => 1.0, "A2" => 0.5, "A3" => 0.5, "A4" => 1.0)
    scale(name) = ws[name]

    @testset "search SNR follows the weights; calibrated weights carry the fix" begin
        # The search's SNR is |D|/√Σw, so rescaling a baseline's weights by s
        # scales its SNR by √s and leaves its peak where it was. The solve moves
        # only through the relative weighting of the stages that combine
        # baselines (station solve, bandpass, adhoc).
        gauge = PinAntenna(1)
        steps = (
            BaselineFringeFit(; gauge), Bandpass(; gauge),
            AdhocPhase(; model = default_adhoc_model(; prior = nothing), gauge),
        )
        base = let sols = _fit_chain(steps, ps)
            (sols, _calibrate_chain(sols, ps))
        end
        scaled = Gustavo.materialize(ps)
        foreach(ms -> scale_weights!(ms, ws), values(scaled))
        fixd = let sols = _fit_chain(steps, scaled)
            (sols, _calibrate_chain(sols, scaled))
        end
        bfr, ffr = base[1][1].info, fixd[1][1].info
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

@testset "Weight scales from consecutive time samples" begin
    # A constant signal (no antenna gains) under noise whose per-component
    # variance is noise²/2; 64 channels × 39 consecutive pairs per cell.
    noise = 0.5
    build(weight) = Gustavo.materialize(
        first(values(first(_build_fringe_ps(; noise, weight, antenna_gains = false, nspw = 1, nchan = 64, ntime = 40))))
    )
    ms = build(2 / noise^2)
    s = weight_scales(ms)
    bytes = @allocated weight_scales(ms)
    @test bytes < 4 * _layer_bytes(ms)
    @test dims(s) == (dims(ms[:weight], Polarization), dims(ms[:weight], BaselineID))
    @test all(x -> isapprox(x, 1; rtol = 0.15), s)
    @test mean(s) ≈ 1 rtol = 0.03

    # Weights claiming 4× less noise than the data carry.
    ms = build(8 / noise^2)
    s = weight_scales(ms)
    @test mean(s) ≈ 0.25 rtol = 0.03
    w = copy(ms[:weight])
    scale_weights!(ms, s)
    @test all(
        ms[:weight][Polarization(At(p)), BaselineID(At(b))] == w[Polarization(At(p)), BaselineID(At(b))] .* s[Polarization(At(p)), BaselineID(At(b))]
            for p in lookup(s, Polarization), b in lookup(s, BaselineID)
    )
    @test mean(weight_scales(ms)) ≈ 1 rtol = 0.03

    # A fully flagged cell has no estimate and holds no data to scale; one whose
    # unflagged samples are never consecutive has data and no estimate.
    ms = build(2 / noise^2)
    flag = ms[:flag]
    view(flag, BaselineID(1), Polarization(1)) .= true
    single = view(flag, BaselineID(2), Polarization(2))
    single .= true
    view(single, Ti(2)) .= false
    a, b = XRadio.baselines(ms)[2]
    pol = lookup(flag, Polarization)[2]
    s = @test_logs (:warn, "weight_scales: no consecutive unflagged time samples to estimate the weight scale of $pol on $(a)–$(b); their factor is 1") weight_scales(ms)
    @test s[Polarization(1), BaselineID(1)] == 1
    @test s[Polarization(2), BaselineID(2)] == 1
end
