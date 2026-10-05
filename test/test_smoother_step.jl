# ── AdhocPhase step + calibrate ───────────────────────────────────────────────
#
# BaselineFringeFit, Bandpass and AdhocPhase fit in turn on a ProcessingSet,
# each on the data the earlier solutions corrected:
# - θ is bit-deterministic across runs, and the multi-scan solve flattens the
#   data.
# - `fit` never writes the caller's data.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# The parameters of a step's first per-integration phase component.
_per_integration_params(sol, step) =
    first(c for c in sol[step].components if first(c.path) === :phase && c.component.Ti isa CAL.PerIntegration).params

# Worst parallel-hand coherence over the cross baselines of a corrected set.
function _worst_parallel_coherence(corr)
    worst = 1.0
    for ms in values(corr)
        feeds = Gustavo.feed_pairs(ms)
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms))), p in axes(feeds, 1)
            a == b && continue
            fa, fb = feeds[p, bi]
            fa == fb || continue
            V = Gustavo._cell_plane(ms[:visibility], bi, p)
            W = Gustavo._cell_plane(ms[:weight], bi, p)
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
    steps = (BaselineFringeFit(; model = fm, gauge), Bandpass(; gauge), AdhocPhase(adhoc; gauge))
    sols_n = _fit_chain(steps, ps; exec = ExecutionConfig())
    ad_n = last(sols_n)

    @testset "3-scan full chain: structure, determinism, coherence" begin
        @test [only(keys(s.steps)) for s in sols_n] == [:fringe, :bandpass, :adhoc]
        # The adhoc block is really solved (nonzero) on every scan.
        @test any(!=(0), _per_integration_params(ad_n, :adhoc))
        @test ad_n.steps[:adhoc].t_pass > 0

        # θ is bit-deterministic across runs.
        sols4 = _fit_chain(steps, ps; exec = ExecutionConfig())
        for (s4, s) in zip(sols4, sols_n)
            @test parent(gains(s4)) == parent(gains(s))
        end

        # The multi-scan solve flattens the data (bandpass + screen recovered).
        @test _worst_parallel_coherence(_calibrate_chain(sols_n, ps)) > 0.99
    end

    @testset "BaselineFringeFit then AdhocPhase (no bandpass)" begin
        fr_fs, ad_fs = _fit_chain(
            (BaselineFringeFit(; model = fm, gauge), AdhocPhase(adhoc; gauge)), ps;
            exec = ExecutionConfig(),
        )
        components = vcat(fr_fs.components, ad_fs.components)
        @test !any(c -> c.path[2] === :bandpass, components)
        @test any(!=(0), _per_integration_params(ad_fs, :adhoc))
        # Without the bandpass stage the injected per-channel bandpass survives,
        # so full coherence is NOT reached — but the delay/rate/adhoc solve must
        # still be sane (all θ finite, per-scan SNRs strong).
        @test all(c -> all(isfinite, c.params), components)
        @test all(>(10), filter(isfinite, fr_fs.steps[:fringe].scan_snr))
    end

    @testset "fit never writes the caller's data" begin
        # Reading may hand back an in-memory set's own arrays.
        snap = Dict(
            k => (copy(parent(ms[:visibility])), copy(parent(ms[:weight])))
                for (k, ms) in pairs(ps)
        )
        fit(Bandpass(; gauge), ps)
        fit(AdhocPhase(adhoc; gauge), ps)
        @test all(
            isequal(snap[k][1], parent(ms[:visibility])) && isequal(snap[k][2], parent(ms[:weight]))
                for (k, ms) in pairs(ps)
        )
    end
end
