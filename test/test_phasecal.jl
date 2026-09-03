# Phase-cal (injected tone) calibration: multitone fit, solution packing, the
# precal hook in the fringe solve, and the tone-channel mask. Reuses
# `_build_fringe_uvset` + `_coherence` and the FP/CAL/UVP aliases from
# test_pipeline.jl (included earlier in runtests.jl).

@testset "Phase-cal (tone) calibration" begin
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)
    first_leaf = first(values(UVP.branches(uvset)))
    ants = UVP.metadata(first_leaf).antennas
    nant = length(ants)
    ant_names = String.(collect(ants.name))
    nchan = length(geom.channel_freqs)
    spw_of_chan = geom.spw_of_chan
    nspw = maximum(spw_of_chan)
    f0 = geom.f0

    # Known instrumental per-(station, feed, spw) delay + phase, station 1 clean.
    rng = MersenneTwister(0x9CA1)
    τ_inst = zeros(nant, 2, nspw)
    φ_inst = zeros(nant, 2, nspw)
    for a in 2:nant, f in 1:2, s in 1:nspw
        τ_inst[a, f, s] = (rand(rng) - 0.5) * 40.0e-9      # ±20 ns
        φ_inst[a, f, s] = (rand(rng) - 0.5) * 4.0          # ±2 rad
    end
    inst_gain(a, f, c) = cis(φ_inst[a, f, spw_of_chan[c]] + 2π * τ_inst[a, f, spw_of_chan[c]] * (geom.channel_freqs[c] - f0))

    # Synthetic tone table: 3 tones per spw (4 MHz comb → 125 ns ambiguity, well
    # above the injected delays), 2 epochs per station across the track. The
    # measured tone phasor IS the instrumental response at the tone frequency.
    ntone = 3
    tone_ν = Matrix{Float64}(undef, ntone, nspw)
    for s in 1:nspw
        cs = findall(==(s), spw_of_chan)
        flo, fhi = extrema(geom.channel_freqs[cs])
        tone_ν[:, s] = range(flo + 1.0e6, fhi - 1.0e6; length = ntone)
    end
    t1, t2 = extrema(geom.times)
    epochs = [t1 + 0.25 * (t2 - t1), t1 + 0.75 * (t2 - t1)]
    rows = [(a, e) for a in 1:nant for e in epochs]
    nrow = length(rows)
    pfreq = Array{Float64, 4}(undef, ntone, nspw, 2, nrow)
    ptone = Array{ComplexF64, 4}(undef, ntone, nspw, 2, nrow)
    for (r, (a, _)) in enumerate(rows), f in 1:2, s in 1:nspw, tn in 1:ntone
        ν = tone_ν[tn, s]
        pfreq[tn, s, f, r] = ν
        ptone[tn, s, f, r] = cis(φ_inst[a, f, s] + 2π * τ_inst[a, f, s] * (ν - f0))
    end
    pcal = FP.PhaseCalTable(
        [ant_names[a] for (a, _) in rows], [e for (_, e) in rows],
        fill(0.01, nrow), fill(NaN, nrow), pfreq, ptone,
    )

    @testset "multitone fit recovers the injected gains" begin
        sol = FP.phasecal_solution(pcal, uvset; sign = 1)
        @test sol isa CAL.CalibrationSolution
        @test sol.info.nblocks == nant * 2 * sol.info.nscan * nspw
        @test sol.info.nmissing == 0
        step = sol.steps[1]
        ev = CAL.GainEvaluator(step.model, step.layout)
        g = CAL.evaluate_gains(ev, step.θ, 1:nchan, 1:1)
        for a in 1:nant, f in 1:2, c in 1:nchan
            @test isapprox(g[c, 1, a, f], inst_gain(a, f, c); atol = 1.0e-8)
        end

        # Outlier tone rejection: corrupt one tone of one block hard.
        pbad = deepcopy(pcal)
        pbad.tone[2, 1, 1, 1] *= cis(2.7)
        pbad.tone[2, 1, 1, 2] *= cis(2.7)                    # both epochs of ant 1
        solb = FP.phasecal_solution(pbad, uvset; sign = 1)
        stepb = solb.steps[1]
        gb = CAL.evaluate_gains(CAL.GainEvaluator(stepb.model, stepb.layout), stepb.θ, 1:nchan, 1:1)
        for c in findall(==(1), spw_of_chan)
            @test isapprox(gb[c, 1, 1, 1], inst_gain(1, 1, c); atol = 1.0e-6)
        end

        # A station absent from the table gets identity gains.
        keep = findall(r -> pcal.station[r] != ant_names[2], 1:nrow)
        psub = FP.PhaseCalTable(
            pcal.station[keep], pcal.time[keep], pcal.interval[keep], pcal.cable[keep],
            pcal.freq[:, :, :, keep], pcal.tone[:, :, :, keep],
        )
        sols = FP.phasecal_solution(psub, uvset; sign = 1)
        step = sols.steps[1]
        gs = CAL.evaluate_gains(CAL.GainEvaluator(step.model, step.layout), step.θ, 1:nchan, 1:1)
        @test all(gs[c, 1, 2, f] ≈ 1.0 for c in 1:nchan, f in 1:2)
    end

    @testset "apply/precal removes injected instrumental phases" begin
        # Corrupt a copy of the uvset with the instrumental gains (V ← V·g_a·g_b*).
        corrupt = deepcopy(uvset)
        for (_, leaf) in UVP.branches(corrupt)
            win = CAL.leaf_window(geom, leaf)
            ci, ti = win.chan_idx, win.ti_idx
            V = parent(leaf[:vis])
            bl_pairs = UVP.baselines(leaf).pairs
            lp = pol_products(leaf)
            for p in eachindex(lp)
                fa, fb = CAL.correlation_feed_pair(lp[p])
                for (bi, (a, b)) in enumerate(bl_pairs), (tt, _) in enumerate(ti), (cc, c) in enumerate(ci)
                    V[cc, tt, bi, p] *= inst_gain(a, fa, c) * conj(inst_gain(b, fb, c))
                end
            end
        end

        sol = FP.phasecal_solution(pcal, uvset; sign = 1)
        fixed = Gustavo.UVData.apply_calibration(corrupt, sol)
        l0 = first(values(UVP.branches(uvset)))
        lf = first(values(UVP.branches(fixed)))
        @test isapprox(parent(lf[:vis]), parent(l0[:vis]); rtol = 1.0e-5)

        # The precal hook: fringe solve on the corrupted set with precal divides
        # the instrumental gains out in-stream — the corrected+reduced output is
        # as coherent as a clean-data solve. The INPUT set must come through
        # untouched (regression: pass 2 once divided eager leaves in place).
        lc = first(values(UVP.branches(corrupt)))
        snapshot = copy(parent(lc[:vis]))
        solf, output = fitcalibrate(
            FP.ApplySolution(sol) |> FringeFit(model = FringeModel()) |>
                Bandpass() |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
            corrupt,
        )
        @test solf.info.precal_applied
        @test parent(lc[:vis]) == snapshot                 # caller's data unmutated
        worst = 1.0
        for (_, leaf) in UVP.branches(output)
            V = parent(leaf[:vis])
            W = parent(leaf[:weights])
            bl_pairs = UVP.baselines(leaf).pairs
            for p in 1:size(V, 4), (bi, (a, b)) in enumerate(bl_pairs)
                a == b && continue
                worst = min(worst, _coherence(@view(V[:, :, bi, p]), @view(W[:, :, bi, p])))
            end
        end
        @test worst > 0.99

        # The WRONG sign doubles the corruption instead of removing it.
        wrong = FP.phasecal_solution(pcal, uvset; sign = -1)
        worse = Gustavo.UVData.apply_calibration(corrupt, wrong)
        lw = first(values(UVP.branches(worse)))
        @test !isapprox(parent(lw[:vis]), parent(l0[:vis]); rtol = 1.0e-2)

        # The solution is per scan × per spw, so it ports to a set that samples
        # the same scan and spws differently — 6 channels per band instead of 8.
        other, _ = _build_fringe_uvset(nchan = 6)
        @test fit(FP.ApplySolution(sol) |> FringeFit(), other) isa CAL.CalibrationSolution
        # A scan it never saw is refused, fail-fast at stream construction.
        twoscan, _ = _build_fringe_uvset(nscans = 2)
        @test_throws "is not in the solution" fit(FP.ApplySolution(sol) |> FringeFit(), twoscan)

        # Diagnostics see the pre-calibrated data when the same precal is passed:
        # the "before" spectra of the corrupted set + precal equal the clean
        # set's RAW spectra (`transforms = ()` suppresses the recorded-chain
        # replay — solf records the precal, and the no-kwarg default replays it).
        d_clean = FP.baseline_fringe_data(uvset, solf; transforms = ())
        d_pcal = FP.baseline_fringe_data(corrupt, solf; precal = sol)
        d_replay = FP.baseline_fringe_data(corrupt, solf)   # replays sol.transforms
        @test isequal(d_replay.spec_before, d_pcal.spec_before)
        finite_close(x, y) = all(
            !isfinite(x[i]) || !isfinite(y[i]) || isapprox(x[i], y[i]; rtol = 1.0e-4, atol = 1.0e-10)
                for i in eachindex(x, y)
        )
        @test finite_close(d_clean.spec_before, d_pcal.spec_before)
        m = FP.fringe_search_map(corrupt, solf; precal = sol)
        @test m isa FP.BaselineFringeMap
    end

    @testset "tone_channel_mask + flag_channels" begin
        mask = FP.tone_channel_mask(pcal, uvset)
        @test length(mask) == nchan
        @test count(mask) >= nspw                       # ≥ 1 flagged channel per tone comb
        @test count(mask) <= ntone * nspw * 3
        # Every tone frequency lands on a flagged channel.
        for s in 1:nspw, tn in 1:ntone
            c = argmin(abs.(geom.channel_freqs .- tone_ν[tn, s]))
            @test mask[c]
        end

        # Solve with flagging: runs, and the output zero-weights those channels.
        _, out2 = fitcalibrate(
            FP.FlagChannels(mask) |> FringeFit(model = FringeModel()) |>
                Bandpass() |>
                TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
            uvset,
        )
        lo = first(values(UVP.branches(out2)))
        ci = CAL.leaf_window(geom, lo).chan_idx
        W = parent(lo[:weights])
        for (cc, c) in enumerate(ci)
            mask[c] && @test all(w -> w <= 0 || !isfinite(w), @view(W[cc, :, :, :]))
        end
    end
end
