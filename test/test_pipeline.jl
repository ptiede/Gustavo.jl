# End-to-end fringe-fitter pipeline test. Builds a small synthetic multi-band
# ProcessingSet with KNOWN injected per-(station, feed) delay/rate/constant
# phase plus a per-(station, feed, AP) atmospheric phase screen, then verifies
# that `fit` → `calibrate` flattens the residual baseline phases (the
# coherence test) and that save/load round-trips.

# Shared usings/aliases (CAL/FP) and the `_coherence` metric.
include("pipeline_helpers.jl")
@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# Coherence of every cross-baseline product of a corrected set whose feed pair
# satisfies `keep`.
function _product_coherences(corr; keep = Returns(true))
    out = Float64[]
    for ms in values(corr)
        feeds = Gustavo.feed_pairs(ms)
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms))), p in axes(feeds, 1)
            (a == b || !keep(feeds[p, bi])) && continue
            V = Gustavo._cell_plane(ms[:visibility], bi, p)
            W = Gustavo._cell_plane(ms[:weight], bi, p)
            push!(out, _coherence(V, W))
        end
    end
    return out
end

_parallel_hand(fp) = fp[1] == fp[2]

# The weighted time average of a corrected set, per (channel, baseline,
# product), its channels running over every spectral window in frequency
# order. Every member must carry the same baselines and products.
function _time_averaged_spectra(corr)
    members = sort!(collect(values(corr)); by = ms -> first(XRadio.frequencies(ms)))
    bls = collect(XRadio.baselines(first(members)))
    feeds = Gustavo.feed_pairs(first(members))
    spectra = map(members) do ms
        @assert collect(XRadio.baselines(ms)) == bls && Gustavo.feed_pairs(ms) == feeds
        S = fill(complex(NaN), length(XRadio.frequencies(ms)), length(bls), size(feeds, 1))
        for bi in eachindex(bls), p in axes(feeds, 1)
            V = Gustavo._cell_plane(ms[:visibility], bi, p)
            W = Gustavo._cell_plane(ms[:weight], bi, p)
            F = Gustavo._cell_plane(ms[:flag], bi, p)
            for c in axes(V, 1)
                num, den = zero(ComplexF64), 0.0
                for t in axes(V, 2)
                    (F[c, t] || !isfinite(V[c, t]) || !(W[c, t] > 0)) && continue
                    num += W[c, t] * V[c, t]
                    den += W[c, t]
                end
                den > 0 && (S[c, bi, p] = num / den)
            end
        end
        S
    end
    return (; spec = reduce(vcat, spectra), bls, feeds = feeds[:, 1])
end

# The band coherence |Σ V| / Σ |V| of each cross baseline's time-averaged
# spectrum on feed pair `feeds`, one value per scan and baseline. Each scan
# keeps its own station phases, so spectra of different scans are not summed.
function _band_coherence(ps, feeds)
    R = Float64[]
    for g in values(XRadio.groupby(ps, XRadio.ByScan()))
        spec = _time_averaged_spectra(g)
        p = findfirst(==(feeds), spec.feeds)
        for (bi, (a, b)) in pairs(spec.bls)
            a == b && continue
            z = filter(isfinite, spec.spec[:, bi, p])
            isempty(z) || push!(R, abs(sum(z)) / sum(abs.(z)))
        end
    end
    return R
end

@testset "Fringe pipeline end-to-end" begin
    ps, _truth = _build_fringe_ps()

    # Adhoc smoother window 7 (< the 12-AP scan) tracks the screen; snr_floor 0
    # keeps every well-determined AP in this high-SNR synthetic.
    gauge = PinAntenna(1)
    sol = _combined(
        _fit_chain(
            (
                BaselineFringeFit(; gauge), Bandpass(; gauge),
                AdhocPhase(FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0)); gauge),
            ), ps,
        )
    )
    @test sol isa CAL.CalibrationSolution
    @test collect(keys(sol.steps)) == [:fringe, :bandpass, :adhoc]
    @test all(haskey(sol, k) for k in keys(sol.steps))

    corr = calibrate(sol, ps)

    @testset "parallel-hand coherence ≈ 1" begin
        cohs = _product_coherences(corr; keep = _parallel_hand)
        @test !isempty(cohs)
        @test all(>(0.99), cohs)
        @info "worst parallel-hand coherence" minimum(cohs)
    end

    @testset "cross-hand coherence ≈ 1" begin
        # Source is unpolarized here, so cross hands should also flatten.
        cohs = _product_coherences(corr; keep = !_parallel_hand)
        @test !isempty(cohs)
        @test all(>(0.99), cohs)
    end

    @testset "save / load round-trip" begin
        path = tempname()
        CAL.save_solution(path, sol)
        sol2 = CAL.load_solution(path)
        @test sol2.geom.times == sol.geom.times
        @test sol2.geom.channel_freqs == sol.geom.channel_freqs
        @test sol2.geom.channel_widths == sol.geom.channel_widths
        @test sol2.components == sol.components
        @test collect(keys(sol2.steps)) == collect(keys(sol.steps))
        @test sol2.provenance == sol.provenance

        corr2 = calibrate(sol2, ps)
        for (k, ms) in pairs(corr)
            @test isequal(parent(corr2[k][:visibility]), parent(ms[:visibility]))
        end
        rm(path; force = true, recursive = true)
    end

    @testset "eager input is left unmodified" begin
        # An in-memory set is the caller's own data, so `calibrate` corrects a
        # copy. The synthetic input carries no NaNs, so `==` is exact.
        snap = Dict(
            k => (copy(parent(ms[:visibility])), copy(parent(ms[:weight])), copy(parent(ms[:flag])))
                for (k, ms) in pairs(ps)
        )
        calibrate(sol, ps)
        for (k, ms) in pairs(ps)
            @test parent(ms[:visibility]) == snap[k][1]
            @test parent(ms[:weight]) == snap[k][2]
            @test parent(ms[:flag]) == snap[k][3]
        end
    end

    @testset "standard interfaces: show" begin
        # `show` gives each type its own summary line instead of a raw dump.
        @test occursin("CalibrationSolution", sprint(show, sol))
        @test occursin("CalibrationSolution", sprint(show, MIME"text/plain"(), sol))
        @test occursin("SolvedComponent", sprint(show, first(sol[:fringe].components)))
    end
end

@testset "Rate & adhoc are feed-tied (SharedFeeds regression)" begin
    # The fringe model MUST tie the per-scan RATE and the per-AP ADHOC phase across
    # feeds (`SharedFeeds`). Both are feed-common physics: the fringe rate is shared
    # by the two feeds and the residual atmospheric screen is non-birefringent. A
    # revert to `PerFeed` lets a spurious inter-feed rate (rate₂ − rate₁) / adhoc phase
    # float on noise and — multiplied by the whole-track Rate lever arm
    # `2π·rate·(t − t0_global)`, hours long — inject large, arbitrary scan-to-scan
    # cross-hand phase jumps (see the `default_fringe_terms` and
    # `default_adhoc_terms` docstrings for the tying rationale).
    #
    # The end-to-end coherence test injects FEED-COMMON truth, so a `PerFeed` revert
    # would still recover it at high SNR and pass — it does NOT guard this decision.
    # Assert the tying structurally (the model) AND that the layout realises it (both
    # feeds share one θ column per (station, time-seg), so the solved inter-feed
    # rate is ≡ 0).
    model = _full_fringe_model()
    phase = CAL.phase_components(model)

    rate_i = findfirst(tc -> tc.term isa CAL.Rate, phase)
    @test rate_i !== nothing
    @test phase[rate_i].Feed isa CAL.SharedFeeds
    @test !(phase[rate_i].Feed isa CAL.PerFeed)

    adhoc_i = findfirst(tc -> tc.Ti isa CAL.PerIntegration, phase)
    @test adhoc_i !== nothing
    @test phase[adhoc_i].Feed isa CAL.SharedFeeds
    @test !(phase[adhoc_i].Feed isa CAL.PerFeed)

    # Layout: `SharedFeeds` folds both feeds to ONE node (θ column), `PerFeed` keeps
    # two distinct ones. So feed-1 and feed-2 share every column iff the tie holds —
    # a `PerFeed` revert breaks this on any solved (station, seg).
    ps, _ = _build_fringe_ps()
    geom = CAL.DataGeometry(ps)
    nant = length(geom.stations)
    layout = CAL.plan_parameters(model, nant, geom)

    for (ci, label) in ((rate_i, "rate"), (adhoc_i, "adhoc"))
        off1 = plan_off1(layout.plans[ci])                 # (ant, feed, ntseg, nfseg)
        @test off1[:, 1, :, :] == off1[:, 2, :, :]   # both feeds → same θ columns
        @test any(!=(0), off1[:, 1, :, :])           # ...and the plan is non-trivial
    end

    # The step's default carries the same tie: the AdhocPhase step's compiled
    # component is the feed-common adhoc form.
    t = Gustavo.model_components(AdhocPhase(; gauge = PinAntenna(1)), nothing)
    @test t.phase.adhoc.Feed isa CAL.SharedFeeds
    @test isempty(t.logamp)
end

@testset "AdhocPhase model surface: vetting + per-feed adhoc" begin
    sm = FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0))
    # One-argument form: the smoother, with the default model.
    gauge = PinAntenna(1)
    @test AdhocPhase(sm; gauge).model == default_adhoc_terms()
    @test AdhocPhase(sm; gauge).smoother === sm

    # Compile-time vetting. The joint smoother cannot solve a per-feed model
    # (its Kalman state is one node per station)...
    pf = default_adhoc_terms(feed = CAL.PerFeed())
    @test_throws "JointKalmanSmoother cannot fit" Gustavo.model_components(
        AdhocPhase(; model = pf, smoother = FP.JointKalmanSmoother(), gauge), nothing,
    )
    # ...no adhoc smoother can address a single-feed tying...
    @test_throws "cannot fit the component" Gustavo.model_components(
        AdhocPhase(; model = default_adhoc_terms(feed = CAL.SingleFeed(2)), gauge),
        nothing,
    )
    # ...and the stage solves exactly one phase component, nothing in logamp.
    two = GainModel(; phase = (; a = pf.phase.adhoc, b = pf.phase.adhoc))
    @test_throws "is not identifiable" Gustavo.model_components(
        AdhocPhase(; model = two, gauge), nothing,
    )
    # ...the joint smoother needs an OU prior...
    @test_throws "JointKalmanSmoother cannot fit" Gustavo.model_components(
        AdhocPhase(; model = default_adhoc_terms(; prior = RandomWalkPrior(; σ = 0.1)), smoother = FP.JointKalmanSmoother(), gauge),
        nothing,
    )
    # ...and the prior is a random walk or OU process along time.
    @test_throws "cannot fit the component" Gustavo.model_components(
        AdhocPhase(; model = default_adhoc_terms(; prior = IIDPrior(0.1)), gauge), nothing,
    )
    la = GainModel(; phase = pf.phase, logamp = pf.phase)
    @test_throws "phase only" Gustavo.model_components(
        AdhocPhase(; model = la, gauge), nothing,
    )

    # End-to-end per-feed adhoc: the layout realizes two nodes per station, both
    # feed tracks are solved, and (the screen being feed-common) the corrected
    # parallel hands still flatten.
    ps, _ = _build_fringe_ps()
    gauge = PinAntenna(1)
    sol = _combined(_fit_chain((BaselineFringeFit(; gauge), AdhocPhase(; model = pf, smoother = sm, gauge)), ps))
    adhoc_c = only(sol[:adhoc, :phase, :adhoc].components)
    @test adhoc_c.component.Feed isa CAL.PerFeed
    leaf = parent(adhoc_c.params)     # (param, node, fseg, tseg, ant)
    @test any(!=(0), @view leaf[1, 1, 1, :, :])
    @test any(!=(0), @view leaf[1, 2, 1, :, :])
    @test all(>(0.99), _product_coherences(calibrate(sol, ps); keep = _parallel_hand))
end

@testset "calibrate ≡ member-by-member calibrate" begin
    # `calibrate` corrects one Measurement Set at a time, and each member's
    # gains depend only on its own (disjoint) θ slots, so calibrating a member
    # on its own gives the same result as calibrating the whole set.
    ps, _ = _build_fringe_ps()
    adhoc = FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0))
    gauge = PinAntenna(1)
    sol_ref = _combined(_fit_chain((BaselineFringeFit(; gauge), Bandpass(; gauge), AdhocPhase(adhoc; gauge)), ps))
    out_fused = calibrate(sol_ref, ps)
    @test collect(keys(out_fused)) == collect(keys(ps))
    same(x, y) = (isnan(x) && isnan(y)) || isapprox(x, y; rtol = 1.0e-5)
    for (k, ms) in pairs(ps)
        alone = only(values(calibrate(sol_ref, XRadio.ProcessingSet(OrderedDict(k => ms)))))
        Vr = parent(alone[:visibility]); Wr = parent(alone[:weight])
        Vf = parent(out_fused[k][:visibility]); Wf = parent(out_fused[k][:weight])
        @test size(Vf) == size(Vr)
        @test all(splat(same), zip(Vr, Vf))
        @test all(splat(same), zip(Wr, Wf))
    end

    # `post` runs on each corrected member: the same as applying it to every
    # member of the corrected set.
    halve(ms) = (parent(ms[:weight]) ./= 2; ms)
    out_post = calibrate(sol_ref, ps; post = halve)
    for (k, ms) in pairs(out_fused)
        @test parent(out_post[k][:weight]) == parent(ms[:weight]) ./ 2
        @test isequal(parent(out_post[k][:visibility]), parent(ms[:visibility]))
    end
end

@testset "Residual accumulation is a plain weighted mean of pre-corrected data" begin
    # `calibrate!` divides out the gains, including the |g|² weight
    # reweighting (Var(V/g) = 1/(w·|g|²)), before `weighted_sums` sees the
    # data. Feeding it already-reweighted (V, W) must give back exactly that
    # inverse-variance mean, with no further correction applied.
    nchan, nti, nbl, npol = 2, 1, 1, 1
    axs = (Frequency([2.2e10, 2.2e10 + 1.0e6]), Ti([0.0]), BaselineID(1:nbl), Polarization(["PP"]))
    V = DimArray(zeros(ComplexF32, nchan, nti, nbl, npol), axs)
    W = DimArray(zeros(Float32, nchan, nti, nbl, npol), axs)
    V[1, 1, 1, 1] = 1.0 + 0.0im          # channel 1: unit gain ⇒ unchanged, weight 1
    V[2, 1, 1, 1] = 10.0im               # channel 2: pre-corrected by |g|² = 0.01,
    W[1, 1, 1, 1] = 1.0                  # so its weight is reweighted to 0.0001
    W[2, 1, 1, 1] = 0.0001
    F = DimArray(falses(nchan, nti, nbl, npol), axs)
    rbar(V, W, F) = map(only, FP.weighted_sums(V, W, F; dims = Frequency))

    r, w = rbar(V, W, F)
    @test w ≈ 1.0001
    @test r / w ≈ (1.0 + 0.001im) / 1.0001

    # Same data in a different memory layout is the same measurement: the kernel
    # locates every axis by name, so only the dimensions decide what is read.
    perm = (Polarization, BaselineID, Ti, Frequency)
    Vp, Wp, Fp = permutedims(V, perm), permutedims(W, perm), permutedims(F, perm)
    @test rbar(Vp, Wp, Fp) == (r, w)
    @test rbar(V, Wp, F) == (r, w)
    # ...and so is a view into larger layers.
    big(A) = cat(A, A; dims = 3)
    sub(A) = view(DimArray(big(parent(A)), (dims(A)[1:2]..., BaselineID(1:2), dims(A, Polarization))), BaselineID(1:1))
    @test rbar(sub(V), sub(W), sub(F)) == (r, w)

    # A flagged channel contributes nothing even though its weight is positive.
    F[2, 1, 1, 1] = true
    r, w = rbar(V, W, F)
    @test w ≈ 1.0
    @test r ≈ 1.0 + 0.0im

    # The per-member kernel sits behind a function barrier and infers.
    out = DimensionalData.otherdims(V, Frequency)
    @inferred FP._weighted_sums!(zeros(ComplexF32, out), zeros(Float32, out), V, W, F, Frequency)
end

@testset "Fringe pipeline: rounds > 1 accumulates (no corruption)" begin
    # Each round accumulates onto the solution of the rounds before it, so
    # coherence stays high on every round.
    ps, _ = _build_fringe_ps()
    adhoc = FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0))
    for r in (1, 2, 3)
        gauge = PinAntenna(1)
        sol = _combined(_fit_chain((BaselineFringeFit(; rounds = r, gauge), Bandpass(; gauge), AdhocPhase(adhoc; gauge)), ps))
        @test minimum(_product_coherences(calibrate(sol, ps); keep = _parallel_hand)) > 0.99
    end
end

@testset "Fringe false-alarm family is one scan's searches" begin
    # Noise puts every pfa strictly inside (0, 1), where the family size moves it.
    ps, _ = _build_fringe_ps(; nscans = 2, noise = 10.0)
    step = BaselineFringeFit(; gauge = PinAntenna(1))
    both = fit(step, ps).steps[:fringe]
    first_scan = first(values(XRadio.groupby(ps, XRadio.ByScan())))
    alone = fit(step, first_scan).steps[:fringe]
    rows = both.det_scan .== 1
    @test count(rows) == length(alone.det_pfa) > 0
    @test any(p -> 0 < p < 1, alone.det_pfa)
    @test 0 < count(alone.det_detected) < length(alone.det_detected)
    @test both.det_snr[rows] == alone.det_snr
    @test both.det_pfa[rows] == alone.det_pfa
    @test both.det_detected[rows] == alone.det_detected
end

@testset "Phase bandpass: per-channel phase recovered" begin
    # Inject a smooth per-(station, feed, channel) phase bandpass (ref ant 1 = 0)
    # on top of the usual delay/rate/phase/screen. The bandpass stage should
    # flatten the per-channel phase; with it OFF the bandpass survives uncorrected.
    nant, nspw, nchan = 4, 2, 8
    nchg = nspw * nchan
    rng = MersenneTwister(0xBA9D)
    bp = zeros(nant, 2, nchg)
    for a in 2:nant, f in 1:2
        off = (rand(rng) - 0.5)
        for gc in 1:nchg
            bp[a, f, gc] = off + 1.0 * sin(2π * gc / nchg + a + f)   # smooth shape + offset
        end
    end
    ps, _ = _build_fringe_ps(; nant = nant, nspw = nspw, nchan = nchan, bandpass = bp)
    adhoc = FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0))
    gauge = PinAntenna(1)
    ff = BaselineFringeFit(; gauge)
    sol_on = _combined(_fit_chain((ff, Bandpass(; gauge), AdhocPhase(adhoc; gauge)), ps))
    sol_off = _combined(
        _fit_chain(
            (
                ff,
                Bandpass(; model = GainModel(; logamp = default_bandpass_terms().logamp), smoother = FP.PerTrackSmoother(), gauge),
                AdhocPhase(adhoc; gauge),
            ), ps,
        )
    )

    don = _time_averaged_spectra(calibrate(sol_on, ps))
    doff = _time_averaged_spectra(calibrate(sol_off, ps))
    p = findfirst(==((1, 1)), don.feeds)
    # Per-channel phase coherence per cross baseline: R = |Σ_c V̄_c| / Σ_c |V̄_c|
    # (1 ⇒ flat per-channel phase). Mean over baselines.
    function freq_coh(spec)
        rs = Float64[]
        for bi in eachindex(don.bls)
            a, b = don.bls[bi]
            a == b && continue
            z = filter(isfinite, spec[:, bi, p])
            isempty(z) && continue
            push!(rs, abs(sum(z)) / sum(abs.(z)))
        end
        return sum(rs) / length(rs)
    end
    R_on = freq_coh(don.spec)
    R_off = freq_coh(doff.spec)
    @test R_on > R_off                  # the bandpass stage flattens per-channel phase
    @test R_on > 0.97                   # nearly flat after the bandpass
    @test R_off < 0.95                  # bandpass survives without the stage

    # The bandpass is station-based, so the corrected data still close around
    # every triangle, with or without it.
    mx(c) = maximum(abs, filter(isfinite, c); init = 0.0)
    τscale = mx(FP.baseline_delays(FP.baseline_spectra(XRadio.average(ps, XRadio.ByScan()))))
    @test τscale > 0
    @test mx(FP.delay_closure(FP.baseline_spectra(XRadio.average(calibrate(sol_on, ps), XRadio.ByScan())))) < 1.0e-3 * τscale
    @test mx(FP.delay_closure(FP.baseline_spectra(XRadio.average(calibrate(sol_off, ps), XRadio.ByScan())))) < 1.0e-3 * τscale
end

@testset "Bandpass on scan-averaged data applies to the full data" begin
    nant, nspw, nchan = 4, 2, 8
    nchg = nspw * nchan
    rng = MersenneTwister(0xA7E2)
    bp = zeros(nant, 2, nchg)
    for a in 2:nant, f in 1:2, gc in 1:nchg
        bp[a, f, gc] = 0.8 * sin(2π * gc / nchg + a + f)
    end
    ps, _ = _build_fringe_ps(; nant, nspw, nchan, nscans = 3, bandpass = bp)
    gauge = PinAntenna(1)
    ff = BaselineFringeFit(; gauge)

    averaged = mapsets(XRadio.groupby(ps, XRadio.ByScan())) do g
        calibrate!(fit(ff, g), g; flag_bad = false, apply_flags = false)
        return XRadio.average(g, XRadio.ByScan())
    end
    avg = XRadio.ProcessingSet(averaged)
    @test all(ms -> length(XRadio.times(ms)) == 1, values(avg))
    bp_avg = fit(Bandpass(; gauge), avg)

    fr = fit(ff, ps)
    bp_full = fit(Bandpass(; gauge), _precal(fr, ps))
    θ(sol) = reduce(vcat, [vec(collect(c.params)) for c in sol.components])
    @test θ(bp_avg) ≈ θ(bp_full) rtol = 1.0e-4

    out = _precal(fr, ps)
    calibrate!(bp_avg, out)
    R = _band_coherence(out, (1, 1))
    @test minimum(R) > 0.97
end

@testset "Bandpass on channel-averaged data applies to the full channels" begin
    nant, nspw, nchan = 4, 2, 8
    nchg = nspw * nchan
    bp = zeros(nant, 2, nchg)
    for a in 2:nant, f in 1:2, gc in 1:nchg
        bp[a, f, gc] = 0.8 * sin(2π * gc / nchg + a + f)
    end
    ps, _ = _build_fringe_ps(; nant, nspw, nchan, nscans = 3, bandpass = bp)
    gauge = PinAntenna(1)
    ff = BaselineFringeFit(; gauge)

    averaged = mapsets(XRadio.groupby(ps, XRadio.ByScan())) do g
        calibrate!(fit(ff, g), g; flag_bad = false, apply_flags = false)
        return XRadio.average(g, XRadio.ByScan(), XRadio.ChannelBins(2))
    end
    avg = XRadio.ProcessingSet(averaged)
    @test all(ms -> length(XRadio.frequencies(ms)) == nchan ÷ 2, values(avg))
    bp_avg = fit(Bandpass(; gauge), avg)

    out = _precal(fit(ff, ps), ps)
    calibrate!(bp_avg, out)
    R = _band_coherence(out, (1, 1))
    @test minimum(R) > 0.95
end

@testset "Phase bandpass: reference inter-feed shape solved" begin
    # Cross-hand rows tie the two feed blocks at every frequency segment, so the
    # reference's relative inter-feed phase bandpass is SOLVED across the band
    # rather than zeroed — otherwise it survives uncorrected in every cross-hand
    # visibility. Only the one band-constant inter-feed offset is conventional,
    # and the circular-mean referencing removes it.
    nant, nspw, nchan = 4, 2, 8
    nchg = nspw * nchan
    rng = MersenneTwister(0xEC9A)
    bp = zeros(nant, 2, nchg)
    # Reference (ant 1): feed 1 flat (the true per-channel gauge), feed 2 a smooth
    # NONZERO shape — the relative inter-feed phase bandpass under test. The shape is
    # orthogonalized against per-band constant + slope: those components are
    # legitimately (and feed-COMMONLY) absorbed by the per-band SBD delay/const
    # and inter-feed delay stages, so leaving them in would make the earlier solves fight
    # an unmodeled feed-2-only per-band ramp instead of exercising the band shape
    # this test is about.
    for b in 1:nspw
        cs = ((b - 1) * nchan + 1):(b * nchan)
        s = [0.6 * sin(2π * k / nchan + 0.9 * b) for k in 1:nchan]   # full period per band
        X = hcat(ones(nchan), collect(Float64, 1:nchan))
        bp[1, 2, cs] .= s .- X * (X \ s)                             # ⊥ {const, slope} per band
    end
    for a in 2:nant, f in 1:2
        off = (rand(rng) - 0.5)
        for gc in 1:nchg
            bp[a, f, gc] = off + 1.0 * sin(2π * gc / nchg + a + f)
        end
    end
    ps, _ = _build_fringe_ps(; nant = nant, nspw = nspw, nchan = nchan, bandpass = bp)
    adhoc = FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0))
    gauge = PinAntenna(1)
    ff = BaselineFringeFit(; gauge)
    sol_on = _combined(_fit_chain((ff, Bandpass(; gauge), AdhocPhase(adhoc; gauge)), ps))
    sol_off = _combined(
        _fit_chain(
            (
                ff,
                Bandpass(; model = GainModel(; logamp = default_bandpass_terms().logamp), smoother = FP.PerTrackSmoother(), gauge),
                AdhocPhase(adhoc; gauge),
            ), ps,
        )
    )

    don = _time_averaged_spectra(calibrate(sol_on, ps))
    doff = _time_averaged_spectra(calibrate(sol_off, ps))
    # Per-channel phase coherence R = |Σ_c V̄_c| / Σ_c |V̄_c| over a product set.
    # The source is unpolarized, so a correct solve leaves the corrected CROSS
    # spectra flat; a SIGN error in the feed-2 block doubles the ref inter-feed
    # ripple instead of removing it, so the cross coherence is also a sign check.
    function freq_coh(spec, ps)
        rs = Float64[]
        for bi in eachindex(don.bls), p in ps
            a, b = don.bls[bi]
            a == b && continue
            z = filter(isfinite, spec[:, bi, p])
            isempty(z) && continue
            push!(rs, abs(sum(z)) / sum(abs.(z)))
        end
        return sum(rs) / length(rs)
    end
    cross_ps = findall(fp -> fp[1] != fp[2], don.feeds)
    R_on = freq_coh(don.spec, cross_ps)
    R_off = freq_coh(doff.spec, cross_ps)
    @test R_on > R_off
    @test R_on > 0.97                   # ref inter-feed shape corrected in the cross hands
    @test R_off < 0.95                  # ...and survives with the stage off

    # θ recovery: the (ref, feed 2) bandpass slots carry the injected shape up
    # to per-band constant + slope components (the parts the per-band SBD and inter-feed
    # delay terms legitimately absorb). Remove the per-band best-fit constant +
    # slope from the difference; the residual shape must match.
    bp_on = only(CAL._applied(sol_on[:bandpass, :phase, :bandpass]).groups)
    plan = only(bp_on.layout.plans)
    @test plan_off1(plan)[1, 2, 1, 1] != 0
    rec = [bp_on.θ[plan_off1(plan)[1, 2, 1, plan.fseg_id[gc]]] for gc in 1:nchg]
    worst = 0.0
    for b in 1:nspw
        cs = ((b - 1) * nchan + 1):(b * nchan)
        d = [rem2pi(rec[gc] - bp[1, 2, gc], RoundNearest) for gc in cs]
        X = hcat(ones(nchan), collect(Float64, 1:nchan))
        worst = max(worst, maximum(abs.(d .- X * (X \ d))))
    end
    @test worst < 0.15

    # Feed-1 products are invariant under the feed-2 common-mode re-gauge.
    @test freq_coh(don.spec, (findfirst(==((1, 1)), don.feeds),)) > 0.97
end

@testset "Amplitude bandpass: component priors (random walk / OU / none)" begin
    # Inject a per-BAND log-amp roll-off (the filterbank passband, deep toward each
    # band's high-channel edge) shared by all stations, plus a small per-station
    # ripple. THEN kill one interior channel per band (zero its weight on every
    # baseline). Both priors must flatten the band AND estimate the killed channels
    # from the in-spw shape; with no prior the killed channels stay untouched
    # (|g| = 1).
    nant, nspw, nchan = 4, 2, 8
    nchg = nspw * nchan
    rng = MersenneTwister(0x5A11)
    locof(gc) = (gc - 1) % nchan + 1                                # local channel within its band
    rolloff(gc) = -0.5 * ((locof(gc) - 1) / (nchan - 1))^2          # per-band: 0 → −0.5 toward high edge
    abp = zeros(nant, 2, nchg)
    for a in 1:nant, f in 1:2
        dev = a == 1 ? 0.0 : (rand(rng) - 0.5) * 0.3               # per-station offset (gauge removes it)
        for gc in 1:nchg
            wig = a == 1 ? 0.0 : 0.05 * sin(2π * gc / nchg + a + f) # small per-station ripple
            abp[a, f, gc] = rolloff(gc) + dev + wig
        end
    end
    ps, _ = _build_fringe_ps(; nant = nant, nspw = nspw, nchan = nchan, amp_bandpass = abp)

    # Kill local channel 7 in every band (globals 7 and 15): flag it on all
    # baselines so the per-channel solve has NO data there.
    dead_local = 7
    dead_globals = [(b - 1) * nchan + dead_local for b in 1:nspw]
    for ms in values(ps)
        flag = DimensionalData.modify(copy, ms[:flag])
        view(flag, Frequency(dead_local)) .= true
        ms[:flag] = flag
    end

    adhoc = FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0))
    larec(θ, plan, a, f, gc) = CAL._component_leaf(plan, θ)[1, f, plan.fseg_id[gc], 1, a]
    amp_model(prior) = GainModel(;
        phase = default_bandpass_terms().phase,
        logamp = (;
            bandpass = CAL.GainComponent(
                CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.ChannelBlocks(1), Feed = CAL.PerFeed(), prior,
            ),
        ),
    )
    function amp_ripple(spec, don, p)
        rs = Float64[]
        for bi in eachindex(don.bls)
            a, b = don.bls[bi]
            a == b && continue
            m = abs.(filter(isfinite, spec[:, bi, p]))
            (isempty(m) || minimum(m) <= 0) && continue
            push!(rs, maximum(m) / minimum(m))
        end
        isempty(rs) ? NaN : sum(rs) / length(rs)
    end

    gauge = PinAntenna(1)
    ff = BaselineFringeFit(; gauge)
    sol_off = _combined(
        _fit_chain(
            (
                ff,
                Bandpass(; model = GainModel(; phase = default_bandpass_terms().phase), smoother = FP.PerTrackSmoother(), gauge),
                AdhocPhase(adhoc; gauge),
            ), ps,
        )
    )
    doff = _time_averaged_spectra(calibrate(sol_off, ps))
    poff = findfirst(==((1, 1)), doff.feeds)
    @test amp_ripple(doff.spec, doff, poff) > 1.3    # roll-off ripple without the stage

    # A prior flattens the band AND fills the killed channels onto the in-spw curve
    # (≈ the mean of the live neighbours, well away from log-amp 0).
    priors = (
        # σ = 0.02 per channel (2 MHz) as 1/Hz^(3/2).
        CAL.RandomWalkPrior(; order = 2, σ = 0.02 * sqrt(3 / (2 * 2.0e6^3))),
        CAL.OUPrior(; scale = LogNormal(log(1.6e7), 1.0), σ = LogNormal(log(0.2), 1.0)),
    )
    for prior in priors
        sol = _combined(
            _fit_chain(
                (ff, Bandpass(; model = amp_model(prior), smoother = FP.PerTrackSmoother(), gauge), AdhocPhase(adhoc; gauge)),
                ps,
            )
        )
        don = _time_averaged_spectra(calibrate(sol, ps))
        p = findfirst(==((1, 1)), don.feeds)
        @test amp_ripple(don.spec, don, p) < 1.08
        bp = only(CAL._applied(sol[:bandpass, :logamp, :bandpass]).groups)
        plan = only(bp.layout.plans)
        for dg in dead_globals, a in 2:nant, f in 1:2
            nbr = 0.5 * (larec(bp.θ, plan, a, f, dg - 1) + larec(bp.θ, plan, a, f, dg + 1))
            @test isfinite(larec(bp.θ, plan, a, f, dg))
            @test abs(larec(bp.θ, plan, a, f, dg) - nbr) < 0.1    # estimated, on the smooth curve
            @test abs(nbr) > 0.12                                 # ...curve far from |g|=1 (meaningful)
        end
    end

    # With no prior the killed channels are NOT estimated — their θ slot is
    # untouched (log-amp 0 ⇒ |g| = 1), the contrast that motivates the priors.
    solf = _combined(_fit_chain((ff, Bandpass(; smoother = FP.PerTrackSmoother(), gauge), AdhocPhase(adhoc; gauge)), ps))
    bpf = only(CAL._applied(solf[:bandpass, :logamp, :bandpass]).groups)
    planf = only(bpf.layout.plans)
    for dg in dead_globals, a in 2:nant, f in 1:2
        @test larec(bpf.θ, planf, a, f, dg) == 0.0
    end
end

@testset "Default adhoc step flattens end-to-end" begin
    ps, _ = _build_fringe_ps()
    gauge = PinAntenna(1)
    sol = _combined(_fit_chain((BaselineFringeFit(; gauge), Bandpass(; gauge), AdhocPhase(; gauge)), ps))
    @test minimum(_product_coherences(calibrate(sol, ps); keep = _parallel_hand)) > 0.99
end

@testset "Fringe pipeline via hierarchical MBD search" begin
    # Narrow bands widely separated (4 × 8 ch, origins every 150 MHz): the
    # common-Δf grid is ≈ 7× the real channel count, so the group search
    # auto-selects the hierarchical SBD→MBD path — verify, then check the
    # end-to-end solve flattens the data exactly like the full path does.
    ps, _ = _build_fringe_ps(nspw = 4, nchan = 8, spw_sep = 1.5e8)
    geom = CAL.DataGeometry(ps)
    ax = FP._search_axes(geom.channel_freqs, geom.times, FP.FringeSearch(), ComplexF64)
    @test ax.mbd !== nothing

    gauge = PinAntenna(1)
    sol = _combined(
        _fit_chain(
            (
                BaselineFringeFit(; gauge), Bandpass(; gauge),
                AdhocPhase(FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0)); gauge),
            ), ps,
        )
    )
    @test all(>(10), filter(isfinite, sol.steps[:fringe].scan_snr))
    det = FP.fringe_detections(sol)
    @test all(det.pfa[det.detected] .< 1.0e-10)             # all detections secure
    @test minimum(_product_coherences(calibrate(sol, ps))) > 0.99
end

@testset "Threaded per-baseline search ≡ serial, and stage timers" begin
    ps, _ = _build_fringe_ps()
    geom = CAL.DataGeometry(ps)
    det1 = FP.search_scan(ps, geom, FP.FringeSearch(); executor = SerialScheduler())
    det4 = FP.search_scan(ps, geom, FP.FringeSearch(); executor = DynamicScheduler(; nchunks = 4))
    @test all(k -> isequal(parent(det1[k]), parent(det4[k])), keys(det1))

    # Stage timers land in the solution info and print; the progress callback
    # fires per completed scan of each pass (plus a done=0 pass announcement).
    events = Tuple{Symbol, Int, Int}[]
    gauge = PinAntenna(1)
    sol = _combined(
        _fit_chain(
            (
                BaselineFringeFit(; gauge), Bandpass(; gauge),
                AdhocPhase(FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0)); gauge),
            ), ps;
            exec = ExecutionConfig(progress = (st, d, t) -> push!(events, (st, d, t))),
        )
    )
    ngroups = length(XRadio.groupby(ps, XRadio.ByScan()))
    # Stages report under each step's `provides` name (:fringe/:bandpass/:adhoc).
    for st in (:fringe, :adhoc)
        ev = [(d, t) for (s2, d, t) in events if s2 == st]
        @test !isempty(ev)
        total = ev[1][2]
        st === :fringe && @test total == ngroups
        @test sort([d for (d, _) in ev]) == collect(0:total)   # announcement + every scan
        @test all(t == total for (_, t) in ev)
    end
    @test any(e -> e[1] === :bandpass, events)                 # bandpass enabled by default
    inf = sol.info
    fringe_step, bandpass_step, adhoc_step = sol.steps[:fringe], sol.steps[:bandpass], sol.steps[:adhoc]
    @test fringe_step.t_pass > 0 && adhoc_step.t_pass > 0 && bandpass_step.t_pass >= 0
    @test inf.ntasks_used >= 1 && inf.inner_tasks >= 1
    # Every step publishes the SAME generic per-scan timing shape — no
    # per-step-name code in the runner, so a third-party step gets this too.
    for step in (fringe_step, bandpass_step, adhoc_step)
        t = step.timing
        @test collect(lookup(t, Gustavo.Scan)) == sol.geom.scan_names
        @test all(>=(0), t.decode) && all(>=(0), t.work)
    end
    @test sum(fringe_step.timing.work) > 0

    # The bandpass/adhoc chain still produces a working solve on a
    # single-scan set (the stage-B/bandpass/adhoc chain is intact).
    ev2 = Tuple{Symbol, Int, Int}[]
    gauge = PinAntenna(1)
    solc = _combined(
        _fit_chain(
            (
                BaselineFringeFit(; gauge), Bandpass(; gauge),
                AdhocPhase(FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0)); gauge),
            ), ps;
            exec = ExecutionConfig(progress = (st, d, t) -> push!(ev2, (st, d, t))),
        )
    )
    @test solc isa CAL.CalibrationSolution
    bp2 = [(d, t) for (s2, d, t) in ev2 if s2 === :bandpass]
    @test !isempty(bp2) && bp2[1][2] == 1                     # one scan in this set
    l2 = first(values(calibrate(solc, ps)))
    b2 = findfirst(pr -> pr[1] != pr[2], collect(XRadio.baselines(l2)))
    @test _coherence(Gustavo._cell_plane(l2[:visibility], b2, 1), Gustavo._cell_plane(l2[:weight], b2, 1)) > 0.99
end

@testset "Largest-first group map" begin
    # Results come back in index order regardless of completion order.
    res = Gustavo._scheduled_map(x -> x * 10, 1:8, [10, 3, 3, 3, 1, 2, 2, 2])
    @test res == [10, 20, 30, 40, 50, 60, 70, 80]
    @test res isa Vector{Int}   # `work`'s return type, not `Any`

    # Empty input and worker-error propagation.
    @test isempty(Gustavo._scheduled_map(identity, Int[], Float64[]))
    @test_throws Exception Gustavo._scheduled_map(
        x -> x == 2 ? error("boom") : x, 1:3, [1, 1, 1],
    )

    # A size per item is required.
    @test_throws "items and sizes must match" Gustavo._scheduled_map(
        identity, 1:3, [1, 1],
    )
end

@testset "EHT-HOPS station flags: unconstrained station flagged" begin
    # Station 4 participates (baselines with valid weights) but carries NO
    # fringe — pure weak noise — so after the closure-screened global solve no
    # strong detection constrains it: the EHT-HOPS flag criterion. Its gains
    # stay identity, and `calibrate` must flag its baselines instead of passing
    # the uncalibrated data through unmarked.
    ps, _ = _build_fringe_ps(nant = 4, nspw = 2, nchan = 8, ntime = 12, feed_common = true)
    rng = MersenneTwister(0xF1A6)
    has4(a, b) = "A4" in (a, b)
    for ms in values(ps)
        vis = DimensionalData.modify(copy, ms[:visibility])
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
            has4(a, b) || continue
            plane = view(vis, BaselineID(bi))
            plane .= 0.01 .* complex.(randn(rng, size(plane)), randn(rng, size(plane)))
        end
        ms[:visibility] = vis
    end
    gauge = PinAntenna(1)
    ff = BaselineFringeFit(;
        model = default_fringe_terms(),
        search = FP.FringeSearch(algorithm = FP.FullGrid()),
        gauge,
    )
    # Each scan's station systems solve as it is searched.
    @test Gustavo._scan_local_solve(ff)
    sol = _combined(_fit_chain((ff, AdhocPhase(; gauge)), ps))
    flags = FP.fringe_station_flags(sol)
    @test any(flags)
    @test all(flags[AntennaName = At("A4")])          # only station 4 unconstrained
    @test !any(flags[AntennaName = At(["A1", "A2", "A3"])])

    # An unconstrained row is flagged, not blanked: it holds real data that the
    # solve left uncalibrated, so its visibilities and weights survive and
    # clearing the flag gives it back.
    corr = calibrate(sol, ps)
    for ms in values(corr)
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
            a == b && continue
            F = view(ms[:flag], BaselineID(bi))
            W = view(ms[:weight], BaselineID(bi))
            if has4(a, b)
                @test all(F)
                @test all(>(0), W)
            else
                @test !any(F)
                @test any(>(0), W)
            end
        end
    end
    # Opting out leaves the (identity-gain) row unflagged.
    l0 = first(values(calibrate(sol, ps; apply_flags = false)))
    bi4 = findfirst(((a, b),) -> a != b && has4(a, b), collect(XRadio.baselines(l0)))
    @test !any(view(l0[:flag], BaselineID(bi4)))
    @test any(>(0), view(l0[:weight], BaselineID(bi4)))
end
