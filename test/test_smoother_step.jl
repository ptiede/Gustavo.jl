# ── AdhocPhase step + calibrate ───────────────────────────────────────────────
#
# The full three-stage pipeline (BaselineFringeFit |> Bandpass |>
# AdhocPhase) on a ProcessingSet:
# - θ is bit-deterministic across runs, and the multi-scan solve flattens the
#   data.
# - A pipeline is the same as its steps fit one at a time, chained through
#   `ApplySolution`, and never writes the caller's data.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# Every parameter a step solved, in component order.
_step_θ(sol, step) = vcat((vec(parent(c.params)) for c in sol[step].components)...)

# The parameters of a step's first per-integration phase component.
_per_integration_params(sol, step) =
    first(c for c in sol[step].components if first(c.path) === :phase && c.component.Ti isa CAL.PerIntegration).params

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
    gauge = PinAntenna(1)
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
    pipe = [BaselineFringeFit(; model = fm, gauge), Bandpass(; gauge), AdhocPhase(adhoc; gauge)]
    sol_n = fit(pipe, ps; exec = ExecutionConfig())

    @testset "3-scan full pipeline: structure, determinism, coherence" begin
        @test collect(keys(sol_n.steps)) == [:fringe, :bandpass, :adhoc]
        # The adhoc block is really solved (nonzero) on every scan.
        @test any(!=(0), _per_integration_params(sol_n, :adhoc))
        @test sol_n.steps[:adhoc].t_pass > 0

        # θ is bit-deterministic across runs.
        pipe4 = [BaselineFringeFit(; model = fm, gauge), Bandpass(; gauge), AdhocPhase(adhoc; gauge)]
        @test parent(gains(fit(pipe4, ps; exec = ExecutionConfig()))) == parent(gains(sol_n))

        # The multi-scan solve flattens the data (bandpass + screen recovered).
        @test _worst_parallel_coherence(calibrate(sol_n, ps)) > 0.99
    end

    @testset "BaselineFringeFit |> AdhocPhase (no bandpass)" begin
        sol_fs = fit(
            [BaselineFringeFit(; model = fm, gauge), AdhocPhase(adhoc; gauge)],
            ps,
            exec = ExecutionConfig(),
        )
        @test collect(keys(sol_fs.steps)) == [:fringe, :adhoc]
        @test !any(c -> c.path[2] === :bandpass, sol_fs.components)
        @test any(!=(0), _per_integration_params(sol_fs, :adhoc))
        # Without the bandpass stage the injected per-channel bandpass survives,
        # so full coherence is NOT reached — but the delay/rate/adhoc solve must
        # still be sane (all θ finite, per-scan SNRs strong).
        @test all(c -> all(isfinite, c.params), sol_fs.components)
        @test all(>(10), filter(isfinite, sol_fs.steps[:fringe].scan_snr))
    end

    @testset "a pipeline ≡ its steps fit separately" begin
        # The oracle is the composition a caller can write by hand: separate
        # `fit` calls of one step each, chained through `ApplySolution`.
        bp = Bandpass(; gauge)
        adhoc_step = AdhocPhase(adhoc; gauge)
        pre = ApplySolution(sol_n[:fringe])

        sol_pipe = fit(pre |> bp |> adhoc_step, ps)
        @test collect(keys(sol_pipe.steps)) == [:bandpass, :adhoc]
        # Neither step is vacuous.
        @test !isempty(_step_θ(sol_pipe, :bandpass))
        @test any(!=(0), _step_θ(sol_pipe, :bandpass))
        @test any(!=(0), _step_θ(sol_pipe, :adhoc))

        sol_a = fit(pre |> bp, ps)
        bandpass_tf = ApplySolution(sol_a[:bandpass])
        sol_b = fit(pre |> bandpass_tf |> adhoc_step, ps)
        @test sol_pipe[:bandpass].components == sol_a[:bandpass].components
        @test sol_pipe[:adhoc].components == sol_b[:adhoc].components

        sol_3 = fit(BaselineFringeFit(; gauge) |> bp |> adhoc_step, ps)
        @test collect(keys(sol_3.steps)) == [:fringe, :bandpass, :adhoc]
        sol_f1 = fit(BaselineFringeFit(; gauge), ps)
        pre_f = ApplySolution(sol_f1[:fringe])
        sol_b1 = fit(pre_f |> bp, ps)
        pre_b = ApplySolution(sol_b1[:bandpass])
        sol_a1 = fit(pre_f |> pre_b |> adhoc_step, ps)
        @test sol_3[:fringe].components == sol_f1[:fringe].components
        @test sol_3[:bandpass].components == sol_b1[:bandpass].components
        @test sol_3[:adhoc].components == sol_a1[:adhoc].components
        # The fringe step reports the same flags and diagnostics either way.
        @test sol_3.steps[:fringe].flagged_ant == sol_f1.steps[:fringe].flagged_ant
        @test sol_3.steps[:fringe].flagged_scan == sol_f1.steps[:fringe].flagged_scan
        @test sol_3.steps[:fringe].scan_snr == sol_f1.steps[:fringe].scan_snr

        # With no correction in front of a step, reading may hand back an
        # in-memory set's own arrays; the caller's data is never written.
        snap = Dict(
            k => (copy(parent(ms[:visibility])), copy(parent(ms[:weight])))
                for (k, ms) in pairs(ps)
        )
        fit(bp |> adhoc_step, ps)
        @test all(
            isequal(snap[k][1], parent(ms[:visibility])) && isequal(snap[k][2], parent(ms[:weight]))
                for (k, ms) in pairs(ps)
        )
    end
end
