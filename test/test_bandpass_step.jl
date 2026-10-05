# ── Bandpass step ─────────────────────────────────────────────────────────────
#
# The bandpass stage on the composable engine. (The M4 parity gates against the
# frozen monolith ran before its deletion.) Standing guarantees:
# - The bandpass step fit after BaselineFringeFit solves only its own
#   per-channel blocks, and θ is bit-deterministic across group concurrency (the
#   per-scan accumulator contributions fold in group-index order).
# - The refine kernels (dTEC, SBD) are inner-invariant and recover the
#   injected dTEC standalone on a scan view.
# - selection (`sol[:bandpass]`) extracts a portable bandpass-only solution and
#   `calibrate!` applies it same-set (index-aligned) and cross-set
#   (station-name-mapped, channel-layout-validated, time-constant only).

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

_bpc(freq; prior = nothing, Ti = CAL.GlobalTime()) =
    CAL.GainComponent(CAL.ConstantTerm(); Ti, Frequency = freq, Feed = CAL.PerFeed(), prior)
_bpmodel(; phase = nothing, amp = nothing, freq = CAL.ChannelBlocks(1)) = GainModel(;
    phase = (; bandpass = _bpc(freq; prior = phase)), logamp = (; bandpass = _bpc(freq; prior = amp)),
)

# The parameters of the one component of `sol` under `path` (`:bandpass, :phase, :bandpass`).
_bp_params(sol, path...) = only(sol[path...].components).params

# The bandpass step's own components, reached by the names its model gives them.
_bp_phase(sol) = _bp_params(sol, :bandpass, :phase, :bandpass)
_bp_amp(sol) = _bp_params(sol, :bandpass, :logamp, :bandpass)

# The components of one grouped (station-specific) component as blocks: each
# group's stations, by index into the geometry's, and its parameters.
_bp_station_blocks(sol, path...) = [
    (; stations = [findfirst(==(s), sol.geom.stations) for s in lookup(c.params, AntennaName)], θ = parent(c.params))
        for c in sol[path...].components
]

# A bandpass leaf's values over `(Frequency, Feed, Ti, AntennaName)`, its unit `:param`
# axis dropped.
_by_channel(p) = permutedims(dropdims(parent(p); dims = 1), (2, 1, 3, 4))

@testset "Bandpass step (new engine)" begin
    nant, nspw, nchan = 4, 2, 8
    rng = MersenneTwister(11)
    nglob = nspw * nchan
    bp_true = 0.5 .* randn(rng, nant, 2, nglob)
    abp_true = 0.2 .* randn(rng, nant, 2, nglob)
    ps, _ = _build_fringe_ps(;
        nant, nspw, nchan, bandpass = bp_true, amp_bandpass = abp_true,
    )
    adhoc = FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0))
    fm = default_fringe_terms()

    fr_n, bp_n = _fit_chain(
        (BaselineFringeFit(; model = fm, gauge = PinAntenna(1)), Bandpass(; gauge = PinAntenna(1))),
        ps; exec = ExecutionConfig(),
    )

    @testset "the bandpass step solves only its own components" begin
        @test length(fr_n.components) == 4
        @test any(!=(0), _bp_phase(bp_n))
        @test any(!=(0), _bp_amp(bp_n))
        @test collect(keys(bp_n.steps)) == [:bandpass]
        @test [c.path for c in bp_n.components] == [(:phase, :bandpass), (:logamp, :bandpass)]
        @test all(c -> c.step === :bandpass, bp_n.components)
        @test bp_n.steps[:bandpass].nscans == length(XRadio.groupby(ps, XRadio.ByScan()))
        @test bp_n.steps[:bandpass].t_pass > 0
    end

    @testset "new-engine fold is deterministic across ntasks" begin
        sols4 = _fit_chain(
            (BaselineFringeFit(; model = fm, gauge = PinAntenna(1)), Bandpass(; gauge = PinAntenna(1))),
            ps; exec = ExecutionConfig(),
        )
        @test [c for s in sols4 for c in s.components] == vcat(fr_n.components, bp_n.components)
    end

    @testset "steps run in any order" begin
        # AdhocPhase solves for a fringe-corrected residual, but that is an
        # assumption of its own solve kernel, not a checked precondition: fit
        # before BaselineFringeFit, it fits the UNCORRECTED residual instead and
        # completes without error — a quietly worse fit, not a rejection.
        solts = _fit_chain((AdhocPhase(; gauge = PinAntenna(1)), BaselineFringeFit(; model = fm, gauge = PinAntenna(1))), ps)
        @test all(s -> s isa CAL.CalibrationSolution, solts)
        # Bandpass's model is self-contained, so bandpass-before-fringe is legal.
        solbf = _fit_chain((Bandpass(; gauge = PinAntenna(1)), BaselineFringeFit(; model = fm, gauge = PinAntenna(1))), ps)
        @test all(s -> s isa CAL.CalibrationSolution, solbf)
    end

    @testset "the leaf holds one frequency segment per channel" begin
        @test size(_bp_phase(bp_n)) == (1, 2, nglob, 1, nant)
        @test collect(lookup(_bp_phase(bp_n), AntennaName)) == bp_n.geom.stations
    end

    @testset "step-selection extraction" begin
        bps = bp_n[:bandpass]
        @test [c.path for c in bps.components] == [(:phase, :bandpass), (:logamp, :bandpass)]
        @test collect(keys(bps.steps)) == [:bandpass]
        @test _bp_phase(bps) == _bp_phase(bp_n)
        @test _bp_amp(bps) == _bp_amp(bp_n)
        @test bps.geom.stations == bp_n.geom.stations
        # A solution with no bandpass STEP at all refuses extraction.
        sol_f = fit(BaselineFringeFit(; model = fm, gauge = PinAntenna(1)), ps)
        @test_throws ArgumentError sol_f[:bandpass]
        @test_throws "holds no component under bandpass" sol_f[:bandpass]
    end

    @testset "gate → spike guard → shape: a contaminated channel is excised, then estimated" begin
        # The synthetic visibilities carry no thermal noise, so an unconstrained
        # per-segment solve reproduces the injected bandpass exactly and nothing
        # about rejection would be exercised. The case that does exercise it is a
        # channel the multiplicative gain model does NOT describe: one narrowband
        # contaminant, on the baselines of a single station. It shows as EXCESS
        # amplitude, so the Fisher weight of the very segment carrying it is the
        # LARGEST on the track — no Gaussian prior can outvote it, and rejection
        # is the narrow-spike guard's job, not the prior's. What the prior then
        # supplies is the estimate of the excised channel.
        nspwc, nchanc, ntimec, nscansc = 1, 32, 4, 2
        nglobc = nspwc * nchanc
        chan_bw = 2.0e6
        rngc = MersenneTwister(0x000A51DE)
        # Injected truth: an OU (AR(1)) draw along frequency per (station, feed) —
        # the process `OUPrior` assumes, correlated over 6 channels.
        nu = 6 * chan_bw
        sigma_bp = 0.15
        a1 = exp(-chan_bw / nu)
        bpc = zeros(nant, 2, nglobc)
        abpc = zeros(nant, 2, nglobc)
        for trk in (bpc, abpc), a in 1:nant, f in 1:2
            trk[a, f, 1] = sigma_bp * randn(rngc)
            for c in 2:nglobc
                trk[a, f, c] = a1 * trk[a, f, c - 1] + sqrt(sigma_bp^2 * (1 - a1^2)) * randn(rngc)
            end
        end
        uvc, truthc = _build_fringe_ps(;
            nant, nspw = nspwc, nchan = nchanc, ntime = ntimec, nscans = nscansc,
            chan_bw, bandpass = bpc, amp_bandpass = abpc, seed = 7, eltype = ComplexF64,
        )
        bad_chan, bad_ant = 17, 2
        for ms in values(uvc)
            V = DimensionalData.modify(Array, ms[:visibility])
            for (bi, (a, b)) in pairs(truthc.bl_pairs)
                (a == bad_ant || b == bad_ant) || continue
                view(V, Frequency(bad_chan), BaselineID(bi)) .*= 8
            end
            ms[:visibility] = V
        end

        fmc = default_fringe_terms()
        runc(amp) = last(
            _fit_chain(
                (
                    BaselineFringeFit(; model = fmc, gauge = PinAntenna(1)), AdhocPhase(adhoc; gauge = PinAntenna(1)),
                    Bandpass(; model = _bpmodel(; amp), smoother = FP.PerTrackSmoother(), gauge = PinAntenna(1)),
                ), uvc; exec = ExecutionConfig(),
            )
        )
        track(s, a, f) = (
            L = parent(_bp_amp(s));
            Float64[L[1, f, c, 1, a] for c in 1:nglobc]
        )
        gauge(x) = x .- sum(x) / length(x)
        cleanc = [c for c in 1:nglobc if abs(c - bad_chan) > 2]
        rms_clean(v, a, f) =
            sqrt(sum(abs2, gauge(v)[cleanc] .- gauge(abpc[a, f, :])[cleanc]) / length(cleanc))

        s_free = runc(nothing)
        s_ar = runc(CAL.OUPrior(; scale = LogNormal(log(nu), 1.0), σ = LogNormal(log(sigma_bp), 1.0)))
        # σ = 0.02 rad per channel (2 MHz) as rad/Hz^(3/2).
        s_wh = runc(CAL.RandomWalkPrior(; order = 2, σ = 0.02 * sqrt(3 / (2 * 2.0e6^3))))

        # The guard excises the contaminated channel; no prior estimates
        # nothing it has no datum for, so that slot stays UNAPPLIED (log-amp 0)
        # while every uncontaminated channel is recovered exactly.
        vfree = track(s_free, bad_ant, 1)
        @test vfree[bad_chan] == 0.0
        @test rms_clean(vfree, bad_ant, 1) < 1.0e-2
        for a in (1, 3, 4)                       # stations the contaminant never touched
            @test rms_clean(track(s_free, a, 1), a, 1) < 1.0e-6
        end

        # A prior fills that slot from the in-band shape, closer to the truth than
        # leaving it unapplied, and about as close as the OU conditional mean of
        # the true neighbors under the true parameters allows.
        var = track(s_ar, bad_ant, 1)
        err_at(v) = abs(gauge(v)[bad_chan] - gauge(abpc[bad_ant, 1, :])[bad_chan])
        oracle = a1 * (abpc[bad_ant, 1, bad_chan - 1] + abpc[bad_ant, 1, bad_chan + 1]) / (1 + a1^2)
        @test var[bad_chan] != 0.0
        @test err_at(var) < err_at(vfree)
        @test err_at(var) < 1.25 * abs(oracle - abpc[bad_ant, 1, bad_chan])

        # The OU prior is the correctly-specified one for this track (the truth IS
        # an OU draw along frequency), so on the uncontaminated channels it costs
        # less than the second-order random walk.
        ar_rms = mean(rms_clean(track(s_ar, a, f), a, f) for a in 1:nant, f in 1:2)
        wh_rms = mean(rms_clean(track(s_wh, a, f), a, f) for a in 1:nant, f in 1:2)
        @test ar_rms < 0.85 * wh_rms
    end

    @testset "portable calibrate!: same-set + cross-set by station name" begin
        bps = bp_n[:bandpass]
        correct(sol, ps) = collect(values(calibrate!(sol, deepcopy(ps); flag_bad = false, apply_flags = false)))
        # `ms` with the gains `g[c, t, station, feed]` divided out, stations by
        # position in the solution's antenna names.
        function divided(ms, g; tconst = false)
            V = DimensionalData.modify(Array, ms[:visibility])
            W = DimensionalData.modify(Array, ms[:weight])
            slot = Dict(n => i for (i, n) in pairs(bps.geom.stations))
            feeds = Gustavo.feed_pairs(ms)
            for (bi, (na, nb)) in pairs(collect(XRadio.baselines(ms))), p in axes(feeds, 1)
                a, b = slot[String(na)], slot[String(nb)]
                fa, fb = feeds[p, bi]
                Vp = Gustavo._cell_plane(V, bi, p); Wp = Gustavo._cell_plane(W, bi, p)
                for t in axes(Vp, 2), c in axes(Vp, 1)
                    tg = tconst ? 1 : t
                    den = g[c, tg, a, fa] * conj(g[c, tg, b, fb])
                    (isfinite(den) && abs2(den) > 0) || continue
                    Vp[c, t] /= den
                    Wp[c, t] *= abs2(den)
                end
            end
            return V, W
        end

        # Same set (identical geometry): index-aligned division.
        for (ms, out) in zip(values(ps), correct(bps, ps))
            win = CAL.GeometryWindow(bps.geom, ms)
            V, W = divided(ms, parent(gains(bps, win)))
            @test isequal(parent(out[:visibility]), parent(V)) && isequal(parent(out[:weight]), parent(W))
        end

        # Cross-set: a different track (fewer times) with a station subset —
        # the time-constant bandpass ports, stations matched by name.
        uvsub, _ = _build_fringe_ps(; nant = 3, nspw, nchan, ntime = 5)
        @test CAL.DataGeometry(uvsub).times != bps.geom.times
        for (ms, out) in zip(values(uvsub), correct(bps, uvsub))
            chan = CAL.GeometryWindow(bps.geom, ms).chan_idx
            V, W = divided(ms, parent(gains(bps, CAL.GeometryWindow(bps.geom, chan, 1:1))); tconst = true)
            @test isequal(parent(out[:visibility]), parent(V)) && isequal(parent(out[:weight]), parent(W))
        end

        # A station the solution never saw is named in a warning.
        uvbig, _ = _build_fringe_ps(; nant = 5, nspw, nchan, ntime = 5)
        @test_logs (:warn, r"A5") match_mode = :any correct(bps, uvbig)

        # A time-VARYING solution ports the same way: `fr_n`'s fringe terms are
        # per scan, and `uvsub` is the same scan sampled over fewer APs, so every
        # target sample places in the scan it belongs to.
        @test all(ms -> ms isa XRadio.MeasurementSet, correct(fr_n, uvsub))

        # A bandpass segments the solve's channels: a set holding some of them
        # takes their segments, matched by frequency…
        uvnc, _ = _build_fringe_ps(; nant = 3, nspw, nchan = 4, ntime = 5)
        for (ms, out) in zip(values(uvnc), correct(bps, uvnc))
            chan = CAL.GeometryWindow(bps.geom, ms).chan_idx
            V, W = divided(ms, parent(gains(bps, CAL.GeometryWindow(bps.geom, chan, 1:1))); tconst = true)
            @test isequal(parent(out[:visibility]), parent(V)) && isequal(parent(out[:weight]), parent(W))
        end
        # …while wider channels on the same centers, or channels between the
        # solve's, cross a solve channel's edge…
        uvwide, _ = _build_fringe_ps(; nant = 3, nspw, nchan = 4, ntime = 5, chan_bw = 4.0e6)
        @test_throws "extends beyond the solution's" correct(bps, uvwide)
        uvoff, _ = _build_fringe_ps(; nant = 3, nspw, nchan, ntime = 5, ref_freq = 230.0e9 + 1.0e6)
        @test_throws "extends beyond the solution's" correct(bps, uvoff)
        # …and a scan the solve never saw is refused, not served by a neighbour.
        uv2, _ = _build_fringe_ps(; nant = 3, nspw, nchan, ntime = 5, nscans = 2)
        @test_throws "is not in the solution" correct(fr_n, uv2)
        # Stations are matched by name; a solution sharing none is refused.
        g = bps.geom
        others = ["Q$i" for i in eachindex(g.stations)]
        strangers = CAL.CalibrationSolution(
            CAL.DataGeometry(
                g.times, g.scan_of_time, g.channel_freqs, g.spw_of_chan, g.channel_widths, g.t0, g.f0,
                g.scan_names, g.spw_names, others,
            ),
            [
                CAL.SolvedComponent(c.step, c.path, c.component, DimArray(parent(c.params), (dims(c.params)[1:4]..., AntennaName(others))))
                    for c in bps.components
            ],
            bps.steps, bps.info,
        )
        @test_throws "shares no station with this data" correct(strangers, uvsub)
    end

end

@testset "PerTrackSmoother: a prior on both observables, broken at its segments" begin
    nant, nspw, nchan = 4, 2, 8
    rng = MersenneTwister(11)
    nglob = nspw * nchan
    ps, _ = _build_fringe_ps(;
        nant, nspw, nchan, station_gains = false,
        bandpass = 0.5 .* randn(rng, nant, 2, nglob), amp_bandpass = 0.2 .* randn(rng, nant, 2, nglob),
    )
    # One (feed, station) phase track over the solve's channels.
    phase_track(sol, a, f) = _by_channel(_bp_phase(sol))[:, f, 1, a]
    curvature(trk) = maximum(abs, diff(diff(CAL.unwrap_phase_track(trk))))
    run(model) = fit(Bandpass(; model, smoother = FP.PerTrackSmoother(), gauge = PinAntenna(1)), ps)

    # A stiff second-order walk on the PHASE leaves only its null space — a
    # straight line in frequency within each spectral window, each with its own
    # level and slope.
    # σ per channel converted to rad/Hz^(3/2): an order-2 walk's second difference
    # over an even step h has variance (2/3)σ²h³.
    per_channel(σ) = CAL.RandomWalkPrior(; order = 2, σ = σ * sqrt(3 / (2 * 2.0e6^3)))
    stiff = per_channel(1.0e-6)
    sol_stiff = run(_bpmodel(; phase = stiff, amp = per_channel(0.01)))
    sol_free = run(_bpmodel())
    spws = [((s - 1) * nchan + 1):(s * nchan) for s in 1:nspw]
    for a in 1:nant, f in 1:2, cs in spws
        @test curvature(phase_track(sol_stiff, a, f)[cs]) < 1.0e-3
    end
    # ...the free fit is genuinely rougher, so the flatness above is the prior
    # acting rather than a featureless track...
    @test maximum(curvature(phase_track(sol_free, a, f)[cs]) for a in 2:nant, f in 1:2, cs in spws) > 0.1
    # ...and the spectral windows are separate pieces: across the boundary the
    # track is not one line.
    @test maximum(curvature(phase_track(sol_stiff, a, f)) for a in 2:nant, f in 1:2) > 0.1
    # A shape segment is the unit solved for, so one straddling two windows is refused.
    @test_throws "straddles a spectral-window boundary" run(_bpmodel(; freq = CAL.GlobalFrequency()))

    # Stations differing only in prior share the layout; each fits under its own.
    mixed = with_station(_bpmodel(), "A3"; phase = (; bandpass = _bpc(CAL.ChannelBlocks(1); prior = stiff)))
    sol_mixed = run(mixed)
    @test size(_bp_phase(sol_mixed)) == size(_bp_phase(sol_free))
    pri = sol_mixed.steps[:bandpass].phase_priors
    @test all(==(stiff), pri[3, :, :]) && all(isnothing, pri[[1, 2, 4], :, :])
    for f in 1:2, cs in spws
        @test curvature(phase_track(sol_mixed, 3, f)[cs]) < 1.0e-3
    end
    @test maximum(curvature(phase_track(sol_mixed, a, f)[cs]) for a in (2, 4), f in 1:2, cs in spws) > 0.1

    # The zero band-mean log-amp gauge is applied AFTER the prior fit, so a prior
    # that rewrites every channel still leaves the bandpass SHAPE only.
    amp = _by_channel(_bp_amp(sol_stiff))
    for a in 1:nant, f in 1:2
        @test abs(mean(amp[:, f, 1, a])) < 4 * eps(eltype(amp))
    end
end

@testset "Bandpass step: model vetting" begin
    nant, nspw, nchan = 4, 2, 8
    rng = MersenneTwister(11)
    nglob = nspw * nchan
    bp_true = 0.5 .* randn(rng, nant, 2, nglob)
    abp_true = 0.2 .* randn(rng, nant, 2, nglob)
    ps, _ = _build_fringe_ps(;
        nant, nspw, nchan, bandpass = bp_true, amp_bandpass = abp_true,
    )

    @testset "compile-time model vetting (can_fit / validate_model)" begin
        pertrack = FP.PerTrackSmoother()
        phase_only = GainModel(; phase = default_bandpass_terms().phase)
        # An empty tree compiles no component at all, so the step would
        # accumulate every scan and write nowhere — rejected at compile time,
        # before any data is read.
        @test_throws ArgumentError fit(Bandpass(; model = GainModel(), smoother = pertrack, gauge = PinAntenna(1)), ps)
        @test_throws "fits nothing" fit(Bandpass(; model = GainModel(), smoother = pertrack, gauge = PinAntenna(1)), ps)
        # JointSmoother is stricter: one complex gain per (station, feed, segment)
        # needs both observables, not just one — so it rejects a model
        # PerTrackSmoother would happily solve.
        @test_throws "JointSmoother requires" fit(Bandpass(; model = phase_only, gauge = PinAntenna(1)), ps)
        @test fit(Bandpass(; model = phase_only, smoother = pertrack, gauge = PinAntenna(1)), ps) isa
            CAL.CalibrationSolution
        # Its two shapes share one frequency segmentation: one complex gain per segment.
        mixed = GainModel(;
            phase = (; bandpass = _bpc(CAL.ChannelBlocks(1))),
            logamp = (; bandpass = _bpc(CAL.ChannelBlocks(2))),
        )
        @test_throws "share one frequency segmentation" model_components(Bandpass(; model = mixed, gauge = PinAntenna(1)), nothing)
        @test model_components(Bandpass(; model = mixed, smoother = pertrack, gauge = PinAntenna(1)), nothing) isa CAL.GainModel
        # A prior the smoothers do not fit is rejected by `can_fit`.
        @test_throws "cannot fit the component" model_components(
            Bandpass(; model = GainModel(; phase = (; bandpass = _bpc(CAL.ChannelBlocks(1); prior = CAL.IIDPrior(0.1)))), gauge = PinAntenna(1)),
            nothing,
        )
        # A component the smoothers' θ writes cannot address (here: a Delay
        # term) is rejected by `can_fit`, naming the component.
        delay_model = GainModel(;
            phase = (;
                bandpass = CAL.GainComponent(
                    CAL.Delay(); Ti = CAL.GlobalTime(),
                    Frequency = CAL.ChannelBlocks(1), Feed = CAL.PerFeed(),
                ),
            ),
        )
        @test_throws "cannot fit the component" model_components(
            Bandpass(; model = delay_model, gauge = PinAntenna(1)), nothing,
        )
        # One track set per observable: a shape, and at most one level beside it.
        ou = CAL.OUPrior(; scale = 1.0e7, σ = 0.1)
        shape = _bpc(CAL.ChannelBlocks(1); prior = ou)
        level = _bpc(CAL.PerSpectralWindow())
        pair(; kw...) = model_components(Bandpass(; model = GainModel(; phase = (; kw...)), smoother = pertrack, gauge = PinAntenna(1)), nothing)
        @test pair(; level, shape) isa CAL.GainModel
        @test_throws "a shape and at most one level" pair(; shape, level, whole = _bpc(CAL.GlobalFrequency()))
        @test_throws "one is the shape" pair(; a = shape, b = _bpc(CAL.ChannelBlocks(2)))
        @test_throws "one is the shape" pair(; a = level, b = _bpc(CAL.GlobalFrequency()))
        @test_throws "the level GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = PerSpectralWindow()" *
            ", Feed = PerFeed(), prior = OUPrior(; scale = 1.0e7, σ = 0.1)) carries a prior" pair(;
            shape, level = _bpc(CAL.PerSpectralWindow(); prior = ou),
        )
        @test_throws "share one time segmentation" pair(; shape, level = _bpc(CAL.PerSpectralWindow(); Ti = CAL.TimeBlocks(10.0)))
        # Without a proper zero-mean prior the shape absorbs the level.
        @test_throws "needs a zero-mean OUPrior" pair(; level, shape = _bpc(CAL.ChannelBlocks(1)))
        @test_throws "needs a zero-mean OUPrior" pair(;
            level, shape = _bpc(CAL.ChannelBlocks(1); prior = CAL.RandomWalkPrior(; σ = 0.1)),
        )
        @test pair(;
            level, shape = _bpc(CAL.ChannelBlocks(1); prior = CAL.RandomWalkPrior(; σ = 0.1, init = Normal(0.0, 0.3))),
        ) isa CAL.GainModel
        # A `stations` entry is vetted like the base, with the can_fit error
        # naming the station whose entry carries the unfittable component.
        bad_entry = with_station(
            default_bandpass_terms(), "A1";
            phase = (;
                bandpass = CAL.GainComponent(
                    CAL.Delay(); Ti = CAL.GlobalTime(),
                    Frequency = CAL.ChannelBlocks(1), Feed = CAL.PerFeed(),
                ),
            ),
        )
        @test_throws "station :A1 entry" model_components(
            Bandpass(; model = bad_entry, gauge = PinAntenna(1)), nothing,
        )

        # A heterogeneous model reaches the solver only where the smoother
        # solves per station. `Bandpass` forwards the question to it: the joint
        # path loops over the layout's station blocks and takes the model, the
        # closure path solves one rectangular gain table and does not, and the
        # runner rejects it there, naming the differing component and its
        # per-station signatures.
        @test supports_station_heterogeneity(Bandpass(; smoother = FP.JointSmoother(), gauge = PinAntenna(1)))
        @test !supports_station_heterogeneity(Bandpass(; smoother = pertrack, gauge = PinAntenna(1)))
        het = with_station(
            default_bandpass_terms(), "A1";
            phase = (; bandpass = _bpc(CAL.ChannelBlocks(2))),
            logamp = (; bandpass = _bpc(CAL.ChannelBlocks(2))),
        )
        @test model_components(Bandpass(; model = het, gauge = PinAntenna(1)), nothing) isa CAL.GainModel
        @test fit(Bandpass(; model = het, gauge = PinAntenna(1)), ps) isa CAL.CalibrationSolution
        @test_throws "station-uniform" fit(Bandpass(; model = het, smoother = pertrack, gauge = PinAntenna(1)), ps)
        @test_throws "phase.bandpass" fit(Bandpass(; model = het, smoother = pertrack, gauge = PinAntenna(1)), ps)
    end

    @testset "observables are located by name, not plan-list position" begin
        anames = ["A1", "A2", "A3"]
        geom = CAL.DataGeometry(;
            times = [0.0, 1.0], scan_of_time = [1, 1],
            channel_freqs = collect(1.0e9 .+ (0:3) .* 1.0e6), spw_of_chan = ones(Int, 4),
            scan_names = ["No001"], spw_names = ["A"],
        )
        _setup(layout) = (;
            layout,
            paths = (;
                phase = FP._bandpass_paths(layout.plantree, :phase),
                logamp = FP._bandpass_paths(layout.plantree, :logamp),
            ),
        )

        lu = CAL.plan_parameters(default_bandpass_terms(), anames, geom)
        @test FP._bandpass_paths(lu.plantree, :phase) == (; shape = (:phase, :bandpass), level = nothing)
        @test only(FP.bandpass_blocks(_setup(lu), zeros(lu.nθ), :phase)).stations == 1:3
        @test isempty(FP.bandpass_level_blocks(_setup(lu), zeros(lu.nθ), :phase))

        # The path is a NAME descent, so it reaches a component the user nested
        # under names of their own.
        nested = CAL.GainModel(
            phase = (; inst = (; bp = _bpc(CAL.ChannelBlocks(1)))),
        )
        @test FP._bandpass_paths(CAL.plan_parameters(nested, anames, geom).plantree, :phase).shape ==
            (:phase, :inst, :bp)

        # The shape is the component on `ChannelBlocks`, whatever the names and
        # their order.
        ou = CAL.OUPrior(; scale = 1.0e7, σ = 0.1)
        leveled = CAL.GainModel(
            phase = (; spw = _bpc(CAL.PerSpectralWindow()), chan = (; c = _bpc(CAL.ChannelBlocks(1); prior = ou))),
        )
        ll = CAL.plan_parameters(leveled, anames, geom)
        @test FP._bandpass_paths(ll.plantree, :phase) == (; shape = (:phase, :chan, :c), level = (:phase, :spw))
        @test only(FP.bandpass_level_blocks(_setup(ll), zeros(ll.nθ), :phase)).plan.fseg isa CAL.PerSpectralWindow

        # A model differing across stations puts one plan per signature group on
        # the flat `plans` list, so its positions no longer name the two
        # observables — the blocks still resolve, one per signature group.
        mh = with_station(
            default_bandpass_terms(), "A1";
            phase = (; bandpass = _bpc(CAL.ChannelBlocks(2))),
            logamp = (; bandpass = _bpc(CAL.ChannelBlocks(2))),
        )
        lh = CAL.plan_parameters(mh, anames, geom)
        @test lh.nphase == 2 && length(lh.plans) == 4
        for group in (:phase, :logamp)
            @test [b.stations for b in FP.bandpass_blocks(_setup(lh), zeros(lh.nθ), group)] ==
                [[1], [2, 3]]
        end

        # An observable the model omits resolves to no path and no blocks.
        lp = CAL.plan_parameters(
            CAL.GainModel(; phase = default_bandpass_terms().phase), anames, geom,
        )
        @test FP._bandpass_paths(lp.plantree, :logamp) == (; shape = nothing, level = nothing)
        @test isempty(FP.bandpass_blocks(_setup(lp), zeros(lp.nθ), :logamp))
    end

end

# ── Per-track outcome reporting and the undetermined-branch gate ──────────────
#
# θ records an unfitted bandpass track as unit gain and a starved one as a
# constant, neither distinguishable there from a genuinely flat response. These
# cover the record that makes the difference visible, and the gate that keeps an
# undetermined phase branch from being reported as a measured ramp.

@testset "Bandpass: per-piece outcome codes" begin
    rng = MersenneTwister(24601)
    nband, nchan = 4, 32
    pieces = [((b - 1) * nchan + 1):(b * nchan) for b in 1:nband]
    x = collect(range(8.6e10, 8.6e10 + 1.28e8; length = nband * nchan))
    prior = CAL.OUPrior(; scale = LogNormal(log(1.6e7), 1.0), σ = LogNormal(log(0.3), 1.0))
    fit!(track, w; unwrap) = (st = fill(Int8(-1), nband); FP._fit_track!(track, w, x, pieces, prior; unwrap, status = st); st)

    # A clean, structured track: every piece solved.
    smooth = 0.3 .* sin.(range(0, 6π; length = nband * nchan))
    w = fill(1.0e4, nband * nchan)
    @test all(==(FP._BP_TRACK_SOLVED), fit!(copy(smooth), w; unwrap = true))

    # A piece with no usable channel estimates nothing and says so.
    gappy = copy(smooth)
    wg = copy(w)
    gappy[1:nchan] .= NaN
    wg[1:nchan] .= 0
    st = fit!(gappy, wg; unwrap = true)
    @test st[1] == FP._BP_TRACK_NODATA
    @test all(isnan, gappy[1:nchan])

    # A constant piece is fit, but carries no shape — reported as flat rather
    # than passed off as a measured response.
    @test all(==(FP._BP_TRACK_FLAT), fit!(fill(0.2, nband * nchan), w; unwrap = true))

    # Phase noise past a radian leaves the 2π branch undetermined: the piece is
    # declined, not fit, so nothing is written for it and no invented trend can
    # reach θ. The amplitude path is never unwrapped and so is never declined.
    noisy = 2.0 .* randn(rng, nband * nchan)
    out = copy(noisy)
    @test all(==(FP._BP_TRACK_DECLINED), fit!(out, w; unwrap = true))
    @test all(isnan, out)
    @test !any(==(FP._BP_TRACK_DECLINED), fit!(copy(noisy), w; unwrap = false))

    # The resolved prior is returned: hyperpriors become the fitted values.
    resolved, levels = FP._fit_track!(copy(smooth), w, x, pieces, prior)
    @test CAL.is_fixed_hyper(resolved.scale) && CAL.is_fixed_hyper(resolved.σ)
    @test isnothing(levels)
    # A track with no data keeps the prior it was given.
    @test FP._fit_track!(fill(NaN, nband * nchan), w, x, pieces, prior) === (prior, nothing)
end

@testset "Bandpass: a track as levels plus a zero-mean shape" begin
    rng = MersenneTwister(31)
    nband, nchan = 4, 32
    pieces = [((b - 1) * nchan + 1):(b * nchan) for b in 1:nband]
    x = collect(range(8.6e10, 8.6e10 + 1.28e8; length = nband * nchan))
    prior = CAL.OUPrior(; scale = 2.0e6, σ = 0.1)
    # Pieces 1–2 share one level, 3–4 another.
    level = [1, 1, 2, 2]
    truth = [0.8, -0.5]
    shape = 0.1 .* sin.(x ./ 1.5e6)
    w = fill(1.0e4, nband * nchan)
    y = [truth[level[b]] for b in 1:nband for _ in 1:nchan] .+ shape .+ randn(rng, nband * nchan) ./ 100

    track = copy(y)
    _, L = FP._fit_track!(track, w, x, pieces, prior; level, nlevel = 2)
    @test L ≈ truth atol = 0.05
    # The track receives their sum.
    @test sqrt(mean(abs2, track .- ([truth[level[b]] for b in 1:nband for _ in 1:nchan] .+ shape))) < 0.03

    # Phase pieces sharing a level are put on one 2π branch before the level is fit.
    wrapped = copy(y)
    wrapped[pieces[2]] .+= 2π
    tp = copy(wrapped)
    _, Lp = FP._fit_track!(tp, w, x, pieces, prior; level, nlevel = 2, unwrap = true)
    @test Lp ≈ truth atol = 0.05
    @test sqrt(mean(abs2, tp .- track)) < 1.0e-6
end

@testset "bandpass_track_report: counts and the unfitted observable" begin
    ph = Int8[FP._BP_TRACK_SOLVED FP._BP_TRACK_FLAT; FP._BP_TRACK_NODATA FP._BP_TRACK_DECLINED]
    phase_status = reshape(ph, 2, 2, 1)
    rep = FP.bandpass_track_report(phase_status, nothing)
    @test (rep.n_solved, rep.n_flat, rep.n_nodata, rep.n_declined) == (1, 1, 1, 1)
    @test !haskey(rep, :band_ids)
    @test rep.track_labels[FP._BP_TRACK_SOLVED + 1] == "solved"
    # An observable that was not fit is an EMPTY status, not a missing one: the
    # record is serialized with the solution and every field must carry a value.
    @test rep.amp_status isa AbstractArray && isempty(rep.amp_status)
    @test rep.phase_status === phase_status
    @test rep.phase_priors isa AbstractArray && isempty(rep.phase_priors)
    # Per-block arrays are counted together.
    blocks = FP.bandpass_track_report((g1 = phase_status, g2 = phase_status), nothing)
    @test (blocks.n_solved, blocks.n_flat, blocks.n_nodata, blocks.n_declined) == (2, 2, 2, 2)
end

@testset "PerTrackSmoother seeds are labeled and read the sums by name" begin
    rng = MersenneTwister(179)
    stations = ["A1", "A2", "A3", "A4"]
    pairs_ = [(stations[a], stations[b]) for a in 1:4 for b in (a + 1):4]
    feeds = [(1, 1), (1, 2), (2, 1), (2, 2)]
    freqs = collect(2.3e11 .+ (0:11) .* 1.0e6)
    ax = (FP._station_pair_dim(pairs_), FP.FeedPair(feeds), Frequency(freqs))
    φ = 0.4 .* randn(rng, 4, 2, 12)
    la = 0.1 .* randn(rng, 4, 2, 12)
    rl = zeros(ComplexF32, ax)
    wl = zeros(Float32, ax)
    for (bi, (a, b)) in enumerate([(a, b) for a in 1:4 for b in (a + 1):4]), (p, (fa, fb)) in enumerate(feeds), c in 1:12
        w = 1.0f0 + rand(rng, Float32)
        wl[bi, p, c] = w
        rl[bi, p, c] = w * exp(la[a, fa, c] + la[b, fb, c] + im * (φ[a, fa, c] - φ[b, fb, c]))
    end
    fsegs = [[c] for c in 1:12]
    segs = dims(DimArray(freqs, Frequency(freqs)), Frequency)

    phase, pprec = @inferred FP._seed_phase_tracks(rl, wl, stations, fsegs, segs; gauge = PinAntenna(1))
    amp, aprec = @inferred FP._seed_amp_tracks(rl, wl, stations, fsegs, segs)
    for A in (phase, pprec, amp, aprec)
        @test eltype(A) == Float32
        @test lookup(A, FP.AntennaName) == stations
        @test lookup(A, Frequency) == freqs
    end
    @test maximum(abs, (phase .- (φ .- φ[1:1, :, :]))[FP.Feed(1)]) < 1.0e-4
    @test maximum(abs, amp .- la) < 1.0e-4

    # Storage order is not an input: the same sums stored permuted seed the same tracks.
    rp, wp = permutedims(rl, (Frequency, FP.FeedPair, FP.AntennaPair)), permutedims(wl, (Frequency, FP.FeedPair, FP.AntennaPair))
    @test isequal(FP._seed_phase_tracks(rp, wp, stations, fsegs, segs; gauge = PinAntenna(1)), (phase, pprec))
    @test isequal(FP._seed_amp_tracks(rp, wp, stations, fsegs, segs), (amp, aprec))
end

@testset "Bandpass: the solve publishes its per-track record" begin
    nant, nspw, nchan, ntime, nscans = 4, 2, 8, 6, 4
    rng = MersenneTwister(777)
    nglob = nspw * nchan
    ps, _ = _build_fringe_ps(;
        nant, nspw, nchan, ntime, nscans,
        bandpass = 0.3 .* randn(rng, nant, 2, nglob),
        amp_bandpass = 0.1 .* randn(rng, nant, 2, nglob),
        seed = 5,
    )
    _, sol = _fit_chain((BaselineFringeFit(; gauge = PinAntenna(1)), Bandpass(; gauge = PinAntenna(1))), ps)
    info = sol.steps[:bandpass]
    # (AntennaName, Feed, frequency segment, time segment): the default model's segments
    # are the spectral windows, and a time-stable bandpass is one time segment.
    @test size(info.phase_status) == (nant, 2, nspw, 1)
    @test size(info.amp_status) == (nant, 2, nspw, 1)
    @test dims(info.phase_status) == dims(info.amp_status)
    total = info.n_solved + info.n_flat + info.n_declined + info.n_nodata
    @test total == 2 * nant * 2 * nspw
    # High-SNR synthetic data with real injected structure: the tracks are
    # measured, not placeholders.
    @test info.n_solved > total ÷ 2
    @test info.n_declined == 0
    # No prior to resolve: every track records the component's own.
    @test size(info.phase_priors) == (nant, 2, 1)
    @test all(isnothing, info.phase_priors) && all(isnothing, info.amp_priors)
end

# ── Break segmentation: a bandpass that changes at named epochs ───────────────
#
# Both smoothers pool the scans of a time segment and solve that stretch as a
# unit, so a segmentation whose segments each span several scans (`InstrumentScans`,
# `TimeBlocks`) is solved one segment at a time. The gauge already demeans every
# written track over the band, so each segment's band mean is identically zero and
# the G3 invariant holds across the break for free.

@testset "bandpass break segmentation" begin
    nant, nchan, nscans = 4, 8, 6
    ps, _ = _build_fringe_ps(; nant, nspw = 1, nchan, nscans, ntime = 8, seed = 7)
    geom = CAL.DataGeometry(ps)
    t, sot = geom.times, geom.scan_of_time
    # Three scans on each side of the break.
    boundary = (maximum(t[sot .== 3]) + minimum(t[sot .== 4])) / 2
    seg = InstrumentScans([boundary])

    # A station phase bandpass present only in the second half.
    Δ = [0.5 * sin(4π * (c - 1) / nchan + a) for a in 1:nant, c in 1:nchan]
    broken = deepcopy(ps)
    for ms in values(broken)
        only(XRadio.scans(ms)) in ("4", "5", "6") || continue
        slot = Dict(n => i for (i, n) in pairs(geom.stations))
        pairs_ = [(slot[String(a)], slot[String(b)]) for (a, b) in XRadio.baselines(ms)]
        vis = copy(parent(ms[:visibility]))
        V = DimArray(vis, dims(ms[:visibility]))
        for (bi, (a, b)) in enumerate(pairs_), p in axes(V, XRadio.Polarization)
            plane = Gustavo._cell_plane(V, bi, p)
            for c in axes(plane, 1)
                plane[c, :] .*= cis(Δ[a, c] - Δ[b, c])
            end
        end
        ms[:visibility] = rebuild(ms[:visibility], vis)
    end

    bp(ti) = GainComponent(ConstantTerm(); Ti = ti, Frequency = ChannelBlocks(1), Feed = PerFeed())
    model(ti) = GainModel(phase = (; bandpass = bp(ti)), logamp = (; bandpass = bp(ti)))

    @testset "θ carries one block per time segment" begin
        sol = fit(Bandpass(; model = model(seg), gauge = PinAntenna(1)), broken)
        leaf = _by_channel(_bp_phase(sol))
        @test size(leaf, 3) == 2                       # (Frequency, Feed, Ti, AntennaName)
        # Each segment is solved from its own scans, so the halves disagree —
        # they would be one block under `GlobalTime`.
        @test any(!iszero, leaf[:, 1, 1, :])
        @test any(!iszero, leaf[:, 1, 2, :])
        @test !isapprox(leaf[:, 1, 1, :], leaf[:, 1, 2, :]; atol = 1.0e-3)
    end

    @testset "a break tracks the data a time-global fit cannot" begin
        # Per-scan spread of the corrected phase about the track median — the
        # same diagnostic the M87 evolution check uses. A time-global template
        # must average the two halves; a break follows both.
        # The injected shape has two cycles across the band, so it is orthogonal
        # to a constant and to a delay ramp. Removing both from each scan's track
        # leaves only what a bandpass owns — otherwise the fixture's own per-scan
        # delays, which no bandpass model here removes, dominate the comparison.
        function detrend(v)
            x = collect(1.0:length(v)); x .-= sum(x) / length(x)
            y = v .- sum(v) / length(v)
            return y .- (sum(x .* y) / sum(abs2, x)) .* x
        end
        function scan_spread(sol)
            corr = calibrate(sol, broken)
            tracks = [
                detrend(angle.(vec(sum(Gustavo._cell_plane(ms[:visibility], 1, 1); dims = 2))))
                    for ms in values(corr)
            ]
            med = [median([tr[c] for tr in tracks]) for c in 1:nchan]
            return sqrt(mean(abs2, reduce(vcat, [tr .- med for tr in tracks])))
        end
        s_glob = scan_spread(fit(Bandpass(; model = model(GlobalTime()), gauge = PinAntenna(1)), broken))
        s_brk = scan_spread(fit(Bandpass(; model = model(seg), gauge = PinAntenna(1)), broken))
        @test s_brk < 0.5 * s_glob
    end

    @testset "the per-track record is labeled like θ" begin
        sol = fit(Bandpass(; model = model(seg), smoother = FP.PerTrackSmoother(), gauge = PinAntenna(1)), broken)
        st = sol.steps[:bandpass].phase_status
        leaf = _bp_phase(sol)
        @test lookup(st, Ti) == lookup(leaf, Ti)
        @test DimensionalData.intervalbounds(st, Ti) == DimensionalData.intervalbounds(leaf, Ti)
        @test length(lookup(st, Ti)) == 2
        @test lookup(st, AntennaName) == geom.stations
        @test only(DimensionalData.intervalbounds(st, Frequency)) == extrema(geom.channel_freqs)
        (_, first_end), (second_start, _) = DimensionalData.intervalbounds(st, Ti)
        @test st[Ti(Contains(second_start))] == st[Ti(2)]
        @test_throws "No interval contains" st[Ti(Contains((first_end + second_start) / 2))]
    end

    @testset "the joint record is labeled like θ, one array per block" begin
        # The first station breaks at the boundary and the rest hold one gain
        # over the track, so the model compiles two station blocks.
        st1 = Symbol(first(geom.stations))
        het = GainModel(;
            phase = (; bandpass = bp(GlobalTime())), logamp = (; bandpass = bp(GlobalTime())),
            stations = NamedTuple{(st1,)}(((; phase = (; bandpass = bp(seg)), logamp = (; bandpass = bp(seg))),)),
        )
        sol = fit(Bandpass(; model = het, smoother = FP.JointSmoother(), gauge = PinAntenna(1)), broken)
        info = sol.steps[:bandpass]
        for obs in (:phase, :logamp)
            leaves = NamedTuple(last(c.path) => c.params for c in sol[:bandpass, obs, :bandpass].components)
            st = getproperty(info, obs === :phase ? :phase_status : :amp_status)
            @test keys(st) == keys(leaves) == (:g1, :g2)
            for k in keys(st)
                @test collect(lookup(st[k], AntennaName)) == collect(lookup(leaves[k], AntennaName))
                @test lookup(st[k], Ti) == lookup(leaves[k], Ti)
                @test DimensionalData.intervalbounds(st[k], Ti) == DimensionalData.intervalbounds(leaves[k], Ti)
            end
            @test [size(st[k], Ti) for k in keys(st)] == [2, 1]
        end
        @test info.n_nodata + info.n_solved + info.n_flat + info.n_declined == 2 * 2 * (2 + nant - 1)
        @test !haskey(info, :n_na)
    end

    @testset "G3: the band mean is time-invariant across the break" begin
        sol = fit(Bandpass(; model = model(seg), gauge = PinAntenna(1)), broken)
        leaf = _by_channel(_bp_phase(sol))
        aleaf = _by_channel(_bp_amp(sol))
        for a in axes(leaf, 4), f in axes(leaf, 2), ts in axes(leaf, 3)
            ph = leaf[:, f, ts, a]
            any(!iszero, ph) || continue
            @test abs(angle(sum(cis, ph))) < 1.0e-8    # zero circular band mean
            @test abs(mean(aleaf[:, f, ts, a])) < 1.0e-8
        end
    end

    @testset "a time-global model is untouched by the change" begin
        # The break path must reduce exactly to the old one-segment solve.
        a = fit(Bandpass(; model = model(GlobalTime()), gauge = PinAntenna(1)), broken)
        b = fit(Bandpass(; model = default_bandpass_terms(), gauge = PinAntenna(1)), broken)
        @test _bp_phase(a) == _bp_phase(b)
    end

    @testset "per-scan resolution is still refused" begin
        @test_throws "cannot fit the component" fit(Bandpass(; model = model(PerScan()), gauge = PinAntenna(1)), broken)
        @test_throws "cannot fit the component" fit(
            Bandpass(; model = model(PerScan()), smoother = FP.PerTrackSmoother(), gauge = PinAntenna(1)), broken,
        )
        @test_throws "degenerate with that scan's source coherence" fit(Bandpass(; model = model(PerScan()), gauge = PinAntenna(1)), broken)
        @test_throws "GlobalTime, InstrumentScans or TimeBlocks" fit(Bandpass(; model = model(PerScan()), gauge = PinAntenna(1)), broken)
    end

    @testset "JointSmoother holds both observables to one time segmentation" begin
        mixed = GainModel(phase = (; bandpass = bp(seg)), logamp = (; bandpass = bp(GlobalTime())))
        @test_throws "share one time segmentation" fit(Bandpass(; model = mixed, gauge = PinAntenna(1)), broken)
        # PerTrackSmoother solves the two independently, so it allows the split —
        # a phase bandpass that breaks beside an amplitude one held all track.
        @test fit(
            Bandpass(; model = mixed, smoother = FP.PerTrackSmoother(), gauge = PinAntenna(1)), broken,
        ) isa CAL.CalibrationSolution
    end
end

# ── A per-station bandpass time segmentation, end to end ─────────────────────
#
# `JointSmoother` solves its gains as a loop over the layout's station blocks,
# so `Bandpass` forwards `supports_station_heterogeneity` to it and a model that
# gives one station its own time segmentation reaches the solver. The saving is
# the point: the nine stations that hold one bandpass over the track carry one
# time segment of θ, not two, and the gauge measures the tenth station's break
# against them rather than gauging it away.

@testset "Bandpass(smoother = JointSmoother()): a per-station time segmentation" begin
    nant, nspw, nchan, ntime, nscans = 10, 1, 8, 6, 2
    nglob = nspw * nchan

    rng = MersenneTwister(20260908)
    bp_true = zeros(nant, 2, nglob, nscans)
    for a in 1:nant, f in 1:2
        base = 0.5 .* randn(rng, nglob)
        bp_true[a, f, :, 1] .= base
        bp_true[a, f, :, 2] .= base
    end
    # Only station 1 changes across the gap.
    brk = [0.6 .* randn(rng, nglob) for _ in 1:2]
    for f in 1:2
        bp_true[1, f, :, 2] .+= brk[f]
    end

    # `station_gains = false` leaves the bandpass as the only station gain, so
    # the solved θ is comparable to the injected track channel by channel. With
    # a fringe stage in front, the delay-like part of each station's bandpass
    # would already have been absorbed into its delay and the comparison would
    # be against something the fixture does not state.
    # Parallel hands only: a cross-hand correlation joins the two feeds' nodes in
    # the gauge graph, which makes the feeds one component carrying one constant
    # between them, and every feed-2 track is then referenced to feed 1. That is
    # correct — with cross-hands the inter-feed phase is measured — but it is not
    # what this test is about.
    uvset, _ = _build_fringe_ps(;
        nant, nspw, nchan, ntime, nscans, bandpass = bp_true, station_gains = false,
        polarizations = ["XX", "YY"],
    )
    # The segmentation boundary sits inside the inter-scan gap: after the last
    # AP of scan 1 and before the first of scan 2.
    ts = CAL.DataGeometry(uvset).times
    t_break = (ts[ntime] + ts[ntime + 1]) / 2

    bpc(ti) = GainComponent(
        ConstantTerm(); Ti = ti, Frequency = ChannelBlocks(1), Feed = PerFeed(),
    )
    het = GainModel(;
        phase = (; bandpass = bpc(GlobalTime())),
        logamp = (; bandpass = bpc(GlobalTime())),
        stations = (
            A1 = (;
                phase = (; bandpass = bpc(InstrumentScans([t_break]))),
                logamp = (; bandpass = bpc(InstrumentScans([t_break]))),
            ),
        ),
    )

    @testset "the capability is declared by the smoother, not the step" begin
        @test supports_station_heterogeneity(Bandpass(; smoother = FP.JointSmoother(), gauge = PinAntenna(1)))
        @test !supports_station_heterogeneity(Bandpass(; smoother = FP.PerTrackSmoother(), gauge = PinAntenna(1)))
        # The closure path solves one rectangular gain table for every station,
        # so the rejection names it and the smoother that would take the model.
        @test_throws "Bandpass(smoother = JointSmoother())" fit(
            Bandpass(; model = het, smoother = FP.PerTrackSmoother(), gauge = PinAntenna(1)), uvset,
        )
    end

    # The alternating solve converges slowly on this model: a sweep change of
    # 2e-6 still leaves 1e-5 rad, so it runs to Float32's resolution instead.
    sol = fit(
        Bandpass(;
            model = het,
            smoother = FP.JointSmoother(max_iterations = 400, tolerance = 1.0e-12),
            gauge = PinAntenna(1),
        ),
        uvset,
    )
    pb = _bp_station_blocks(sol, :bandpass, :phase, :bandpass)

    @testset "the uniform stations carry one time segment, not two" begin
        @test [b.stations for b in pb] == [[1], collect(2:nant)]
        # A block's θ is (param, feed node, frequency segment, time segment, station).
        @test size(pb[1].θ, 4) == 2
        @test size(pb[2].θ, 4) == 1
    end

    @testset "the break is measured against the stations that span it" begin
        cdemean(v) = rem2pi.(v .- angle(sum(cis, v)), RoundNearest)
        for f in 1:2
            # Station 1 is the gauge reference and the broken station both, so
            # the pin holds its FIRST segment and its second carries the break.
            @test all(iszero, pb[1].θ[1, f, :, 1, 1])
            @test pb[1].θ[1, f, :, 2, 1] ≈ cdemean(brk[f]) atol = 1.0e-6
            # The other nine hold one bandpass across the gap, and it is the one
            # they really have — station 1's break leaks into none of them.
            for (ai, a) in pairs(pb[2].stations)
                want = cdemean(bp_true[a, f, :, 1] .- bp_true[1, f, :, 1])
                @test pb[2].θ[1, f, :, 1, ai] ≈ want atol = 1.0e-6
            end
        end
    end

    @testset "a heterogeneous solution round-trips through save/load" begin
        mktempdir() do dir
            path = joinpath(dir, "het.zarr")
            save_solution(path, sol)
            back = load_solution(path)
            @test back.components == sol.components
            qb = _bp_station_blocks(back, :bandpass, :phase, :bandpass)
            @test [b.stations for b in qb] == [b.stations for b in pb]
            @test [size(b.θ) for b in qb] == [size(b.θ) for b in pb]
            for (x, y) in zip(qb, pb)
                @test x.θ == y.θ
            end
        end
    end
end
