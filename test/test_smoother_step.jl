# ── AdhocPhase step + calibrate ───────────────────────────────────────────────
#
# The full three-stage pipeline (BaselineFringeFit |> Bandpass |>
# AdhocPhase) on a ProcessingSet:
# - θ is bit-deterministic across runs, and the multi-scan solve flattens the
#   data.
# - A pipeline is the same as its steps fit one at a time, chained through
#   `ApplySolution`, and never writes the caller's data.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# One component's θ block of a single STEP. `i` indexes that step's own
# `layout.plans` (phase components first, then log-amplitude).
_blk(step, i) = step.θ[[p.range for p in step.layout.plans][i]]

# Worst parallel-hand coherence over the cross baselines of a corrected set.
function _worst_parallel_coherence(corr)
    worst = 1.0
    for ms in values(corr)
        feeds = UVP.feed_pairs(ms)
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms))), p in axes(feeds, 1)
            a == b && continue
            fa, fb = feeds[p, bi]
            fa == fb || continue
            V = UVP._cell_plane(ms[:visibility], bi, p)
            W = UVP._cell_plane(ms[:weight], bi, p)
            worst = min(worst, _coherence(V, W))
        end
    end
    return worst
end

@testset "AdhocPhase step + output sink (new engine)" begin
    adhoc = FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0))
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
    ps, _ = _build_fringe_ps(;
        nant, nspw = 2, nchan = 8, nscans = 3,
        bandpass = bp_true, amp_bandpass = abp_true,
    )
    fm = default_fringe_terms()
    pipe = [BaselineFringeFit(model = fm), Bandpass(), AdhocPhase(adhoc)]
    sol_n = fit(pipe, ps; exec = ExecutionConfig(), gauge = PinAntenna(1))

    @testset "3-scan full pipeline: structure, determinism, coherence" begin
        @test keys(sol_n) == [:fringe, :bandpass, :adhoc]
        adhoc_step = sol_n[:adhoc].steps[1]
        phases = CAL.phase_components(adhoc_step.model)
        # The adhoc block is really solved (nonzero) on every scan.
        ipi = findfirst(tc -> tc.Ti isa CAL.PerIntegration, phases)
        @test any(!=(0), _blk(adhoc_step, ipi))
        @test adhoc_step.info.t_pass > 0

        # θ is bit-deterministic across runs.
        pipe4 = [BaselineFringeFit(model = fm), Bandpass(), AdhocPhase(adhoc)]
        @test parent(gains(fit(pipe4, ps; exec = ExecutionConfig(), gauge = PinAntenna(1)))) == parent(gains(sol_n))

        # The multi-scan solve flattens the data (bandpass + screen recovered).
        @test _worst_parallel_coherence(calibrate(sol_n, ps)) > 0.99
    end

    @testset "BaselineFringeFit |> AdhocPhase (no bandpass)" begin
        sol_fs = fit(
            [BaselineFringeFit(model = fm), AdhocPhase(adhoc)],
            ps,
            exec = ExecutionConfig(),
            gauge = PinAntenna(1),
        )
        @test keys(sol_fs) == [:fringe, :adhoc]
        @test !any(s -> haskey(s.layout.plantree.phase, :bandpass), sol_fs.steps)
        adhoc_step_fs = sol_fs[:adhoc].steps[1]
        phases = CAL.phase_components(adhoc_step_fs.model)
        ipi = findfirst(tc -> tc.Ti isa CAL.PerIntegration, phases)
        @test any(!=(0), _blk(adhoc_step_fs, ipi))
        # Without the bandpass stage the injected per-channel bandpass survives,
        # so full coherence is NOT reached — but the delay/rate/adhoc solve must
        # still be sane (all θ finite, per-scan SNRs strong).
        @test all(s -> all(isfinite, s.θ), sol_fs.steps)
        @test all(>(10), filter(isfinite, sol_fs[:fringe].steps[1].info.scan_snr))
    end

    @testset "a pipeline ≡ its steps fit separately" begin
        # The oracle is the composition a caller can write by hand: separate
        # `fit` calls of one step each, chained through `ApplySolution`.
        bp = Bandpass()
        pre = ApplySolution(sol_n[:fringe])

        sol_pipe = fit(pre |> bp |> AdhocPhase(adhoc), ps; gauge = PinAntenna(1))
        @test keys(sol_pipe) == [:bandpass, :adhoc]
        # Neither step is vacuous.
        @test sol_pipe[:bandpass].steps[1].layout.nθ > 0
        @test any(!=(0), sol_pipe[:bandpass].steps[1].θ)
        @test any(!=(0), sol_pipe[:adhoc].steps[1].θ)

        sol_a = fit(pre |> bp, ps; gauge = PinAntenna(1))
        bandpass_tf = ApplySolution(sol_a[:bandpass])
        sol_b = fit(pre |> bandpass_tf |> AdhocPhase(adhoc), ps; gauge = PinAntenna(1))
        @test sol_pipe[:bandpass].steps[1].θ == sol_a[:bandpass].steps[1].θ
        @test sol_pipe[:adhoc].steps[1].θ == sol_b[:adhoc].steps[1].θ

        sol_3 = fit(BaselineFringeFit() |> bp |> AdhocPhase(adhoc), ps; gauge = PinAntenna(1))
        @test keys(sol_3) == [:fringe, :bandpass, :adhoc]
        sol_f1 = fit(BaselineFringeFit(), ps; gauge = PinAntenna(1))
        pre_f = ApplySolution(sol_f1[:fringe])
        sol_b1 = fit(pre_f |> bp, ps; gauge = PinAntenna(1))
        pre_b = ApplySolution(sol_b1[:bandpass])
        sol_a1 = fit(pre_f |> pre_b |> AdhocPhase(adhoc), ps; gauge = PinAntenna(1))
        @test sol_3[:fringe].steps[1].θ == sol_f1[:fringe].steps[1].θ
        @test sol_3[:bandpass].steps[1].θ == sol_b1[:bandpass].steps[1].θ
        @test sol_3[:adhoc].steps[1].θ == sol_a1[:adhoc].steps[1].θ
        # The fringe step reports the same flags and diagnostics either way.
        @test sol_3.info.flagged_ant == sol_f1.info.flagged_ant
        @test sol_3.info.flagged_scan == sol_f1.info.flagged_scan
        @test sol_3[:fringe].steps[1].info.scan_snr == sol_f1[:fringe].steps[1].info.scan_snr

        # With no correction in front of a step, reading may hand back an
        # in-memory set's own arrays; the caller's data is never written.
        snap = Dict(
            k => (copy(parent(ms[:visibility])), copy(parent(ms[:weight])))
                for (k, ms) in pairs(ps)
        )
        fit(bp |> AdhocPhase(adhoc), ps; gauge = PinAntenna(1))
        @test all(
            isequal(snap[k][1], parent(ms[:visibility])) && isequal(snap[k][2], parent(ms[:weight]))
                for (k, ms) in pairs(ps)
        )
    end
end
