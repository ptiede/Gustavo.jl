# ── Bandpass step ─────────────────────────────────────────────────────────────
#
# - The bandpass step fit after BaselineFringeFit solves only its own
#   per-channel blocks, and θ is bit-deterministic across group concurrency (the
#   per-scan accumulator contributions fold in group-index order).
# - A bandpass solution is portable: `calibrate!` applies it same-set
#   (index-aligned) and cross-set (station-name-mapped, channel-layout-validated,
#   time-constant only).

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

_bpc(freq; prior = nothing, Ti = CAL.GlobalTime()) =
    CAL.GainComponent(CAL.ConstantTerm(); Ti, Frequency = freq, Feed = CAL.PerFeed(), prior)
_bpmodel(; phase = nothing, amp = nothing, freq = CAL.ChannelBlocks(1)) = GainModel(;
    phase = (; bandpass = _bpc(freq; prior = phase)), logamp = (; bandpass = _bpc(freq; prior = amp)),
)

# The parameters of the one component of `sol` under `path` (`:phase, :bandpass`).
_bp_params(sol, path...) = only(sol[path...].components).params

# The bandpass step's own components, reached by the names its model gives them.
_bp_phase(sol) = _bp_params(sol, :phase, :bandpass)
_bp_amp(sol) = _bp_params(sol, :logamp, :bandpass)

# The components of one grouped (station-specific) component as blocks: each
# group's stations, by index into the geometry's, and its parameters.
_bp_station_blocks(sol, path...) = [
    (; stations = [findfirst(==(s), sol.geom.stations) for s in lookup(c.params, AntennaName)], θ = parent(c.params))
        for c in sol[path...].components
]

# A bandpass leaf's values over `(Frequency, Feed, Ti, AntennaName)`, its unit `:param`
# axis dropped.
_by_channel(p) = permutedims(dropdims(parent(p); dims = 1), (2, 1, 3, 4))

@testset "Bandpass step" begin
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
        @test length(fr_n.components) == 3
        @test any(!=(0), _bp_phase(bp_n))
        @test any(!=(0), _bp_amp(bp_n))
        @test bp_n.name === :bandpass
        @test [c.path for c in bp_n.components] == [(:phase, :bandpass), (:logamp, :bandpass)]
        @test bp_n.info.nscans == length(XRadio.groupby(ps, XRadio.ByScan()))
        @test bp_n.info.t_pass > 0
    end

    @testset "the fold is deterministic across ntasks" begin
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

    @testset "path selection" begin
        bpp = bp_n[:phase, :bandpass]
        @test [c.path for c in bpp.components] == [(:phase, :bandpass)]
        @test bpp.name === :bandpass
        @test _bp_phase(bpp) == _bp_phase(bp_n)
        @test bpp.geom.stations == bp_n.geom.stations
        # A solution with no bandpass component refuses the selection.
        sol_f = fit(BaselineFringeFit(; model = fm, gauge = PinAntenna(1)), ps)
        @test_throws ArgumentError sol_f[:phase, :bandpass]
        @test_throws "holds no component under phase.bandpass" sol_f[:phase, :bandpass]
    end

    @testset "portable calibrate!: same-set + cross-set by station name" begin
        bps = bp_n
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
                g.scan_names, g.spw_names, others, g.nfeed,
            ),
            bps.name,
            [
                CAL.ComponentFit(c.path, c.component, DimArray(parent(c.params), (dims(c.params)[1:4]..., AntennaName(others))))
                    for c in bps.components
            ],
            bps.info; step = bps.step,
        )
        @test_throws "shares no station with this data" correct(strangers, uvsub)
    end

end

@testset "Bandpass: a prior on both observables, broken at its segments" begin
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
    run(model) = fit(Bandpass(; model, gauge = PinAntenna(1)), ps)

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
    pri = sol_mixed.info.phase_priors
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
        phase_only = GainModel(; phase = default_bandpass_terms().phase)
        # An empty tree compiles no component at all, so the step would
        # accumulate every scan and write nowhere — rejected at compile time,
        # before any data is read.
        @test_throws ArgumentError fit(Bandpass(; model = GainModel(), gauge = PinAntenna(1)), ps)
        @test_throws "fits nothing" fit(Bandpass(; model = GainModel(), gauge = PinAntenna(1)), ps)
        @test fit(Bandpass(; model = phase_only, gauge = PinAntenna(1)), ps) isa CAL.CalibrationSolution
        # Its two shapes share one frequency segmentation: one complex gain per segment.
        mixed = GainModel(;
            phase = (; bandpass = _bpc(CAL.ChannelBlocks(1))),
            logamp = (; bandpass = _bpc(CAL.ChannelBlocks(2))),
        )
        @test_throws "share one frequency segmentation" model_components(Bandpass(; model = mixed, gauge = PinAntenna(1)), nothing)
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
        pair(; kw...) = model_components(Bandpass(; model = GainModel(; phase = (; kw...)), gauge = PinAntenna(1)), nothing)
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

        # The joint solve loops over the layout's station blocks, so it takes a
        # model that differs across stations.
        @test supports_station_heterogeneity(Bandpass(; gauge = PinAntenna(1)))
        het = with_station(
            default_bandpass_terms(), "A1";
            phase = (; bandpass = _bpc(CAL.ChannelBlocks(2))),
            logamp = (; bandpass = _bpc(CAL.ChannelBlocks(2))),
        )
        @test model_components(Bandpass(; model = het, gauge = PinAntenna(1)), nothing) isa CAL.GainModel
        @test fit(Bandpass(; model = het, gauge = PinAntenna(1)), ps) isa CAL.CalibrationSolution
    end

    @testset "observables are located by name, not plan-list position" begin
        anames = ["A1", "A2", "A3"]
        geom = CAL.DataGeometry(; nfeed = 2,
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
        @test only(FP._bandpass_blocks(_setup(lu), zeros(lu.nθ), :phase)).stations == 1:3
        @test isempty(FP._bandpass_level_blocks(_setup(lu), zeros(lu.nθ), :phase))

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
        @test only(FP._bandpass_level_blocks(_setup(ll), zeros(ll.nθ), :phase)).plan.fseg isa CAL.PerSpectralWindow

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
            @test [b.stations for b in FP._bandpass_blocks(_setup(lh), zeros(lh.nθ), group)] ==
                [[1], [2, 3]]
        end

        # An observable the model omits resolves to no path and no blocks.
        lp = CAL.plan_parameters(
            CAL.GainModel(; phase = default_bandpass_terms().phase), anames, geom,
        )
        @test FP._bandpass_paths(lp.plantree, :logamp) == (; shape = nothing, level = nothing)
        @test isempty(FP._bandpass_blocks(_setup(lp), zeros(lp.nθ), :logamp))
    end

end

# ── Per-track outcome reporting and phase branches ───────────────────────────
#
# θ records an unfitted bandpass track as unit gain and a starved one as a
# constant, neither distinguishable there from a genuinely flat response. These
# cover the record that makes the difference visible, and the choice of a phase
# track's 2π branches under a prior.

# One track fit the way the joint solve fits each of its tracks: the prior
# seeded from the observations `y` (weights `w`), for a phase the branches
# chosen, the hyperparameters fit to the marginal likelihood, and the MAP step,
# which for this quadratic likelihood is the MAP.
function _fit_one_track(y, w, x, pieces, prior; level = nothing, nlevel = 0, phase = false, flat_span = 0.01)
    n = length(y)
    known = isfinite.(y) .& (w .> 0)
    fits = [(; phase = (; x, pieces, prior, level, nlevel))]
    t = only(FP._bandpass_tracks(fits, :phase, [reshape(1:n, 1, 1, n, 1)], [(1, 1)], [[1]], 1, known))
    yk, wk = ifelse.(known, y, NaN), ifelse.(known, w, 0.0)
    x0 = ifelse.(known, y, 0.0)
    priors = Any[any(t.active) ? FP._seed_prior(t, yk, wk; phase) : prior]
    slip = fill(NaN, length(pieces))
    phase && any(t.active) && FP._choose_branches!(x0, t, only(priors), yk, wk, known; slip)
    free = copy(known)
    for (j, p) in pairs(pieces)
        t.active[j] && (free[p] .= true)
    end
    sys = FP._posterior_system(FP.spdiagm(wk), zeros(n), x0, free, any(t.active) ? [t] : [], x)
    any(t.active) && FP._fit_hypers!(priors, Any[nothing], sys, [t], 0.0; tolerance = 1.0e-6, max_iterations = 100)
    x1, levels, _ = FP._map_step(sys, any(t.active) ? priors : [])
    values = ifelse.(free, x1, NaN)
    status = [FP._band_track_status(values[p], flat_span) for p in pieces]
    return (; values, status, prior = only(priors), levels = any(t.active) ? only(levels) : Dict{Int, Float64}(), slip)
end

@testset "Bandpass: a branch choice that raises the prior energy is undone" begin
    nband, nchan = 2, 64
    n = nband * nchan
    pieces = [((b - 1) * nchan + 1):(b * nchan) for b in 1:nband]
    x = collect(range(8.6e10, 8.6e10 + 6.4e7; length = n))
    prior = CAL.RandomWalkPrior(; order = 2, σ = sqrt(3 / (2 * (x[2] - x[1])^3)))
    known = trues(n)
    fits = [(; phase = (; x, pieces, prior, level = nothing, nlevel = 0))]
    t = only(FP._bandpass_tracks(fits, :phase, [reshape(1:n, 1, 1, n, 1)], [(1, 1)], [[1]], 1, known))
    tl = FP._track_layout(t)
    energy(φ) = first(FP._track_energy(tl, prior, φ))
    rng = MersenneTwister(3)

    # Low SNR: the most probable branches of the noisy estimates put kinks in a
    # smooth track.
    smooth = cumsum(cumsum(0.7 .* randn(rng, n))) ./ 10
    w = ones(n)
    y = smooth .+ rem2pi.(randn(rng, n), RoundNearest)
    raw = copy(smooth)
    @test any(!iszero, FP._choose_branches!(raw, t, prior, y, w, known))
    @test energy(raw) > energy(smooth)
    φ = copy(smooth)
    @test all(iszero, FP._choose_descending_branches!(φ, tl, t, prior, y, w, known))
    @test φ == smooth

    # High SNR: unwrapping a steep track lowers its energy and is kept.
    steep = 0.5 .* (1:n) .+ 0.01 .* randn(rng, n)
    wrapped = rem2pi.(steep, RoundNearest)
    φ = copy(wrapped)
    shift = FP._choose_descending_branches!(φ, tl, t, prior, wrapped, fill(1.0e4, n), known)
    @test any(!iszero, shift)
    @test energy(φ) < energy(wrapped)
end

@testset "Bandpass: per-piece outcome codes" begin
    nband, nchan = 4, 32
    pieces = [((b - 1) * nchan + 1):(b * nchan) for b in 1:nband]
    x = collect(range(8.6e10, 8.6e10 + 1.28e8; length = nband * nchan))
    prior = CAL.OUPrior(; scale = LogNormal(log(1.6e7), 1.0), σ = LogNormal(log(0.3), 1.0))
    fit(y, w; kw...) = _fit_one_track(y, w, x, pieces, prior; phase = true, kw...)

    # A clean, structured track: every piece solved, its branches certain.
    smooth = 0.3 .* sin.(range(0, 6π; length = nband * nchan))
    w = fill(1.0e4, nband * nchan)
    f = fit(smooth, w)
    @test all(==(FP.TrackSolved), f.status)
    @test all(<(1.0e-6), f.slip)
    # The resolved prior: hyperpriors become the fitted values.
    @test CAL.is_fixed_hyper(f.prior.scale) && CAL.is_fixed_hyper(f.prior.σ)

    # A piece with no usable channel estimates nothing and says so.
    gappy = copy(smooth)
    wg = copy(w)
    gappy[1:nchan] .= NaN
    wg[1:nchan] .= 0
    f = fit(gappy, wg)
    @test f.status[1] == FP.TrackNoData
    @test all(isnan, f.values[1:nchan])
    @test isnan(f.slip[1])

    # A constant piece is fit, but carries no shape — reported as flat rather
    # than passed off as a measured response.
    @test all(==(FP.TrackFlat), fit(fill(0.2, nband * nchan), w).status)
    # What counts as flat is the smoother's `flat_span`.
    @test all(==(FP.TrackFlat), fit(smooth, w; flat_span = 1.0).status)

    # A track with no data keeps the prior it was given.
    @test fit(fill(NaN, nband * nchan), w).prior === prior
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

    f = _fit_one_track(y, w, x, pieces, prior; level, nlevel = 2)
    @test [f.levels[1], f.levels[2]] ≈ truth atol = 0.05
    # The track is their sum.
    @test sqrt(mean(abs2, f.values .- ([truth[level[b]] for b in 1:nband for _ in 1:nchan] .+ shape))) < 0.03

    # Phase pieces sharing a level are put on one 2π branch before the level is fit.
    wrapped = copy(y)
    wrapped[pieces[2]] .+= 2π
    fp = _fit_one_track(wrapped, w, x, pieces, prior; level, nlevel = 2, phase = true)
    @test [fp.levels[1], fp.levels[2]] ≈ truth atol = 0.05
    @test sqrt(mean(abs2, fp.values .- f.values)) < 1.0e-6
end

@testset "Bandpass: a phase track's branches at low SNR" begin
    # A smooth track under a second-order random walk with 0.9 rad of phase noise
    # per channel: stepping from channel to channel slips 2π on most tracks, while
    # choosing every channel's branch under the prior recovers them.
    rng = MersenneTwister(1017)
    n, σn, sq = 64, 0.9, 0.02
    x = collect(1.0:n)
    prior = CAL.RandomWalkPrior(; order = 2, σ = sq / sqrt(2 / 3))
    w = fill(1 / σn^2, n)
    function slipped(track, truth)
        e = rem2pi.(track .- truth, RoundNearest)
        return maximum(abs, rem2pi.(e .- angle(sum(cis, e)), RoundNearest)) > π / 2
    end
    nwalk = nfilter = 0
    for _ in 1:20
        truth = cumsum(cumsum(sq .* randn(rng, n))) .+ 0.4 * (2rand(rng) - 1) .* x
        θ = rem2pi.(truth .+ σn .* randn(rng, n), RoundNearest)
        nwalk += slipped(FP._estimate_map(prior, CAL.unwrap_phase_track(θ; weights = w), w, x), truth)
        nfilter += slipped(_fit_one_track(θ, w, x, [1:n], prior; phase = true).values, truth)
    end
    @test nwalk >= 8
    @test nfilter <= 1

    # With one branch history the summed likelihood is that history's own, as the
    # filter computes it on the lifted track, so the hyperparameters estimated over
    # every branch match those of the lifted track.
    truth = cumsum(cumsum(sq .* randn(rng, n))) .+ 0.3 .* x
    wclean = fill(1.0e4, n)
    θ = rem2pi.(truth .+ randn(rng, n) ./ 100, RoundNearest)
    model, ym, r, u = FP._filter_inputs(prior, θ, wclean, x)
    bf = FP._branch_filter(model, ym, r, u)
    @test length(bf.histories) == 1
    @test bf.loglik ≈ bf.best ≈ FP._kalman_loglik(model, only(bf.histories).lift, r, u)
    @test maximum(abs, diff(only(bf.histories).lift .- truth)) < 0.1
    hyper = CAL.RandomWalkPrior(; order = 2, σ = LogNormal(log(0.02), 1.0))
    @test FP._estimate_hypers(hyper, [θ], [wclean], [x]; branches = true).σ ≈
        FP._estimate_hypers(hyper, [only(bf.histories).lift], [wclean], [x]).σ rtol = 1.0e-6
end

@testset "Bandpass: hyperparameters from the Laplace marginal" begin
    rng = MersenneTwister(2022)

    # The selected inverse is the inverse on the factor's pattern.
    A = FP.sparse(randn(rng, 40, 40) .* (rand(rng, 40, 40) .< 0.1))
    A = A * A' + 5I
    F = FP._factor(A, rand(rng, 40))
    Σ = FP._selected_inverse(F)
    L = FP.sparse(F.L)
    pattern = [(F.p[i], F.p[j]) for j in axes(L, 2) for i in FP.rowvals(L)[FP.nzrange(L, j)]]
    @test maximum(((a, b),) -> abs(Σ(a, b) - inv(Matrix(A))[a, b]), pattern) < 1.0e-12
    outside = first((a, b) for a in 1:40, b in 1:40 if !((a, b) in pattern || (b, a) in pattern))
    @test_throws "not on the factor's sparsity pattern" Σ(outside...)

    # A gauge row solved for one slot is the constrained minimum.
    n = 6
    M = randn(rng, n, n)
    H = FP.sparse(M * M' + I)
    b = randn(rng, n)
    x = randn(rng, n)
    free = trues(n)
    free[2] = false
    gauge = (; rows = [[(1, 0.5), (2, 1.0), (4, 2.0), (6, -1.0)]], d = [0.3])
    sys = FP._posterior_system(H, b, x, free, [], collect(1.0:n); gauge)
    x1, _, _ = FP._map_step(sys, [])
    f = findall(free)
    C = [0.5 0 2.0 0 -1.0]
    K = [Matrix(H[f, f]) C'; C 0]
    ref = (K \ [b[f] .+ H[f, f] * x[f]; 0.3 - x[2]])[1:length(f)]
    @test x1[f] ≈ ref rtol = 1.0e-10
    @test x1[2] == x[2]
    @test dot([0.5, 1.0, 2.0, -1.0], x1[[1, 2, 4, 6]]) ≈ 0.3

    # One track whose slots are measured independently: the Laplace marginal is
    # the Kalman one, and its maximum is type-II MAP on the track.
    m = 64
    xf = collect(range(1.0e9, 1.0e9 + 63.0e6; length = m))
    y = 0.3 .* sin.(range(0, 3π; length = m)) .+ randn(rng, m) ./ 20
    w = fill(400.0, m)
    function marginal(prior, resolved)
        fits = [(; phase = (; x = xf, pieces = [1:m], prior, level = nothing, nlevel = 0))]
        t = only(FP._bandpass_tracks(fits, :phase, [reshape(1:m, 1, 1, m, 1)], [(1, 1)], [[1]], 1, trues(m)))
        sys = FP._posterior_system(FP.spdiagm(w), w .* y, zeros(m), trues(m), [t], xf)
        K, h, c = FP._reduce!(sys, FP._assemble!(sys.layout, [resolved]), 0.0)
        G = FP._factor(K, sys.key)
        lp = dot(h, G \ h) / 2 - c - logdet(G) / 2 + FP._expected_log_prior(resolved, only(sys.chains), nothing, Float64)
        return lp, t, sys
    end
    rw = CAL.RandomWalkPrior(; order = 2, σ = LogNormal(log(1.0e-12), 2.0))
    walk(σ) = CAL.RandomWalkPrior(; order = 2, σ)
    kalman(σ) = FP._random_walk_loglik(y, w, xf, 2, σ)
    @test first(marginal(rw, walk(2.0e-11))) - first(marginal(rw, walk(1.0e-11))) ≈ kalman(2.0e-11) - kalman(1.0e-11) rtol = 1.0e-8
    _, t, sys = marginal(rw, walk(1.0e-11))
    priors = Any[walk(1.0e-11)]
    @test FP._fit_hypers!(priors, Any[nothing], sys, [t], 0.0; tolerance = 1.0e-4, max_iterations = 50).converged
    @test only(priors).σ ≈ FP._estimate_hypers(rw, [y], [w], [xf]).σ rtol = 1.0e-3

    # A carried-over curvature far too flat for a start 10⁻⁴ below the maximum:
    # steps of at most a factor of 10 reach the same maximum.
    σ̂ = only(priors).σ
    priors = Any[walk(1.0e-4 * σ̂)]
    stale = Any[(; inverse = fill(1.0e6, 1, 1), measured = true)]
    fit = FP._fit_hypers!(priors, stale, sys, [t], 0.0; tolerance = 1.0e-4, max_iterations = 50)
    @test fit.converged
    @test fit.steps >= 4
    @test only(priors).σ ≈ σ̂ rtol = 1.0e-3

    # An OU prior, both hyperparameters, its band constant integrated out.
    ou = CAL.OUPrior(; scale = LogNormal(log(1.0e7), 1.0), σ = LogNormal(log(0.3), 1.0))
    _, t, sys = marginal(ou, CAL.OUPrior(; scale = 1.0e7, σ = 0.3))
    priors = Any[CAL.OUPrior(; scale = 1.0e7, σ = 0.3)]
    @test FP._fit_hypers!(priors, Any[nothing], sys, [t], 0.0; tolerance = 1.0e-4, max_iterations = 50).converged
    ref = FP._estimate_hypers(ou, [y], [w], [xf]; level = [1])
    @test only(priors).scale ≈ ref.scale rtol = 1.0e-2
    @test only(priors).σ ≈ ref.σ rtol = 1.0e-2
end

@testset "_bandpass_track_report: counts and the unfitted observable" begin
    ph = [FP.TrackSolved FP.TrackFlat; FP.TrackNoData FP.TrackNoData]
    phase_status = reshape(ph, 2, 2, 1)
    rep = FP._bandpass_track_report(phase_status, nothing)
    @test (rep.n_solved, rep.n_flat, rep.n_nodata) == (1, 1, 2)
    @test !haskey(rep, :band_ids)
    # An observable that was not fit is an EMPTY status, not a missing one: the
    # record is serialized with the solution and every field must carry a value.
    @test rep.amp_status isa AbstractArray && isempty(rep.amp_status)
    @test rep.phase_status === phase_status
    @test rep.phase_priors isa AbstractArray && isempty(rep.phase_priors)
    @test rep.phase_slip isa AbstractArray && isempty(rep.phase_slip)
    # Per-block arrays are counted together.
    blocks = FP._bandpass_track_report((g1 = phase_status, g2 = phase_status), nothing)
    @test (blocks.n_solved, blocks.n_flat, blocks.n_nodata) == (2, 2, 4)
    # Three of four pieces measured nothing; the smoother's `warn_fraction` decides
    # whether that warns.
    @test_logs (:warn, r"75.0% of") FP._warn_degenerate_bandpass(rep, 0.5)
    @test_logs FP._warn_degenerate_bandpass(rep, 0.8)
    # A phase piece whose slip probability reaches the smoother's threshold warns.
    slipped = FP._bandpass_track_report(phase_status, nothing; phase_slip = reshape([0.01 0.2; NaN NaN], 2, 2, 1))
    @test_logs (:warn, r"1 of 2 phase pieces") FP._warn_uncertain_branches(slipped, 0.05)
    @test_logs FP._warn_uncertain_branches(slipped, 0.5)
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
    info = sol.info
    # (AntennaName, Feed, frequency segment, time segment): the default model's segments
    # are the spectral windows, and a time-stable bandpass is one time segment.
    @test size(info.phase_status) == (nant, 2, nspw, 1)
    @test size(info.amp_status) == (nant, 2, nspw, 1)
    @test dims(info.phase_status) == dims(info.amp_status)
    total = info.n_solved + info.n_flat + info.n_nodata
    @test total == 2 * nant * 2 * nspw
    # High-SNR synthetic data with real injected structure: the tracks are
    # measured, not placeholders.
    @test info.n_solved > total ÷ 2
    # No phase prior, so no branch was chosen and no slip probability recorded.
    @test dims(info.phase_slip) == dims(info.phase_status)
    @test all(isnan, info.phase_slip)
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

    @testset "the record is labeled like θ" begin
        sol = fit(Bandpass(; model = model(seg), gauge = PinAntenna(1)), broken)
        st = sol.info.phase_status
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
        info = sol.info
        for obs in (:phase, :logamp)
            leaves = NamedTuple(last(c.path) => c.params for c in sol[obs, :bandpass].components)
            st = getproperty(info, obs === :phase ? :phase_status : :amp_status)
            @test keys(st) == keys(leaves) == (:g1, :g2)
            for k in keys(st)
                @test collect(lookup(st[k], AntennaName)) == collect(lookup(leaves[k], AntennaName))
                @test lookup(st[k], Ti) == lookup(leaves[k], Ti)
                @test DimensionalData.intervalbounds(st[k], Ti) == DimensionalData.intervalbounds(leaves[k], Ti)
            end
            @test [size(st[k], Ti) for k in keys(st)] == [2, 1]
        end
        @test info.n_nodata + info.n_solved + info.n_flat == 2 * 2 * (2 + nant - 1)
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
        @test_throws "degenerate with that scan's source coherence" fit(Bandpass(; model = model(PerScan()), gauge = PinAntenna(1)), broken)
        @test_throws "GlobalTime, InstrumentScans or TimeBlocks" fit(Bandpass(; model = model(PerScan()), gauge = PinAntenna(1)), broken)
    end

    @testset "JointSmoother holds both observables to one time segmentation" begin
        mixed = GainModel(phase = (; bandpass = bp(seg)), logamp = (; bandpass = bp(GlobalTime())))
        @test_throws "share one time segmentation" fit(Bandpass(; model = mixed, gauge = PinAntenna(1)), broken)
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

    # The solve converges slowly on this model, so it runs far below the default
    # tolerance to resolve the break to 1e-6 rad.
    sol = fit(
        Bandpass(;
            model = het,
            smoother = FP.JointSmoother(max_iterations = 400, tolerance = 1.0e-6),
            gauge = PinAntenna(1),
        ),
        uvset,
    )
    pb = _bp_station_blocks(sol, :phase, :bandpass)

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
            @test maximum(abs, pb[1].θ[1, f, :, 2, 1] .- cdemean(brk[f])) < 1.0e-6
            # The other nine hold one bandpass across the gap, and it is the one
            # they really have — station 1's break leaks into none of them.
            for (ai, a) in pairs(pb[2].stations)
                want = cdemean(bp_true[a, f, :, 1] .- bp_true[1, f, :, 1])
                @test maximum(abs, pb[2].θ[1, f, :, 1, ai] .- want) < 1.0e-6
            end
        end
    end

    @testset "a heterogeneous solution round-trips through save/load" begin
        mktempdir() do dir
            path = joinpath(dir, "het.zarr")
            save_solution(path, sol)
            back = load_solution(path)
            @test back.components == sol.components
            qb = _bp_station_blocks(back, :phase, :bandpass)
            @test [b.stations for b in qb] == [b.stations for b in pb]
            @test [size(b.θ) for b in qb] == [size(b.θ) for b in pb]
            for (x, y) in zip(qb, pb)
                @test x.θ == y.θ
            end
        end
    end
end
