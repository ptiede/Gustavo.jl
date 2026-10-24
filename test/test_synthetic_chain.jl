# ── The calibration chain on a synthetic track ───────────────────────────────
#
# Per-scan station delays, rates and phases, a phase wander within each scan, a
# drifting inter-feed delay and a time-stable bandpass, calibrated by a
# feed-common fringe fit, the bandpass, a second fringe search with the track
# closure, and ad hoc phasing. What is left is constant over time and frequency
# within each scan and product.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

# `ps` with each station's `gain(a, f, s, ti, t, ν)` applied, `t` from the
# scan's first sample.
function _synth_corrupt!(ps, gain)
    for ms in values(ps)
        s = parse(Int, only(unique(ms[:scan_name])))
        station(n) = parse(Int, n[2:end])
        a1, a2 = station.(ms[:baseline_antenna1_name]), station.(ms[:baseline_antenna2_name])
        feeds = Gustavo.feed_pairs(ms)
        t = collect(lookup(ms, XRadio.Ti))
        t .-= first(t)
        ν = collect(XRadio.frequencies(ms))
        V = copy(parent(ms[:visibility]))
        for bl in axes(V, 3), p in axes(V, 1), ti in axes(V, 4), c in axes(V, 2)
            fa, fb = feeds[p, bl]
            V[p, c, bl, ti] *= gain(a1[bl], fa, s, ti, t[ti], ν[c]) * conj(gain(a2[bl], fb, s, ti, t[ti], ν[c]))
        end
        ms[:visibility] = rebuild(ms[:visibility], V)
    end
    return ps
end

# Per (scan, station pair, feed pair, spectral window): the visibilities summed
# over time and channels, and their coherence.
function _synth_sums(data)
    out = Dict{Any, @NamedTuple{v::ComplexF64, coherence::Float64}}()
    for ms in values(data)
        s = only(unique(ms[:scan_name]))
        feeds = Gustavo.feed_pairs(ms)
        a1, a2 = ms[:baseline_antenna1_name], ms[:baseline_antenna2_name]
        V = parent(ms[:visibility])
        f = first(XRadio.frequencies(ms))
        for bl in axes(V, 3), p in axes(V, 1)
            a1[bl] == a2[bl] && continue
            v = view(V, p, :, bl, :)
            out[(s, a1[bl], a2[bl], feeds[p, bl], f)] = (; v = sum(v), coherence = abs(sum(v)) / sum(abs, v))
        end
    end
    return out
end

# The phase (degrees) of each (scan, window) ratio of product `num` to `den`
# against its mean over the group `group(station pair, window)` names.
function _synth_variation(sums, num, den, group)
    ratios = Dict{Any, Vector{ComplexF64}}()
    for (k, x) in sums
        k[4] == num || continue
        push!(get!(ratios, group((k[2], k[3]), k[5]), ComplexF64[]), x.v * conj(sums[(k[1], k[2], k[3], den, k[5])].v))
    end
    return reduce(vcat, [rad2deg.(angle.(r .* conj(sum(r ./ abs.(r))))) for r in values(ratios)])
end

_rms(x) = sqrt(sum(abs2, x) / length(x))

@testset "the calibration chain flattens a synthetic track" begin
    nant, nscans, nspw, nchan = 5, 6, 4, 32
    rng = StableRNG(7)
    nglob = nspw * nchan
    bandpass = [0.4 * (a + f) / nant + 0.5 * sin(2π * gc / nglob + a + f) for a in 1:nant, f in 1:2, gc in 1:nglob]
    amp_bandpass = [0.05 * (a - f) + 0.1 * sin(4π * gc / nglob + a - f) for a in 1:nant, f in 1:2, gc in 1:nglob]
    ps, truth = _build_fringe_ps(;
        nant, nscans, nspw, nchan, chan_bw = 4.0e6, spw_sep = 1.28e8, noise = 0.25,
        different_feed_amplitude = 0.1, antenna_gains = false, bandpass, amp_bandpass, eltype = ComplexF64,
    )
    delay = (rand(rng, nant, nscans) .- 0.5) .* 2.0e-9
    rl_level = (rand(rng, nant) .- 0.5) .* 1.0e-9
    rl_drift = (rand(rng, nant) .- 0.5) .* 1.0e-11
    rate = (rand(rng, nant, nscans) .- 0.5) .* 2.0e-3
    phase = (rand(rng, nant, nscans) .- 0.5) .* 2π
    rl_phase = (rand(rng, nant) .- 0.5) .* 2π
    τ(a, f, s) = delay[a, s] + (f == 2 ? rl_level[a] + rl_drift[a] * (s - 1) : 0.0)
    gain(a, f, s, ti, t, ν) = cis(
        phase[a, s] + (f == 2 ? rl_phase[a] : 0.0) + 2π * τ(a, f, s) * (ν - truth.f0) +
            2π * rate[a, s] * t + 0.3 * sin(0.4 * ti + a + s)
    )
    _synth_corrupt!(ps, gain)

    gauge = PinAntenna(1)
    search = FringeSearch(; rate_window = (-0.01, 0.01))
    feed_common = GainModel(;
        phase = (;
            mbd = GainComponent(Delay(); Ti = PerScan(), Feed = SharedFeeds()),
            rate = GainComponent(Rate(); Ti = PerScan(), Feed = SharedFeeds()),
        ),
    )
    track_model = merge(
        default_fringe_model();
        phase = (;
            rel_delay = GainComponent(Delay(); Ti = GlobalTime(), Feed = ExceptFeed(1)),
            rel_drift = GainComponent(Delay(); Ti = PerScan(), Feed = ExceptFeed(1), prior = OUPrior(; scale = 3 * 3600.0, σ = 3.0e-11)),
        ),
    )
    first_pass = _precal(fit(FringeClosure(; model = feed_common, gauge), fit(search, ps)), ps)
    bandpassed = _precal(fit(Bandpass(; gauge), first_pass), first_pass)
    closure = FringeClosure(; model = track_model, gauge, closure = Stationization(; loss = OutlierMixture()))
    track = fit(closure, fit(search, bandpassed))
    tracked = _precal(track, bandpassed)
    flat = _precal(fit(AdhocPhase(; gauge), tracked), tracked)
    sums = _synth_sums(flat)

    @test isempty(track.info.flagged_ant)
    # Every same-feed product is flat over its scan and window.
    @test minimum(x -> x.coherence, (x for (k, x) in sums if k[4][1] == k[4][2])) > 0.99
    # The inter-feed delay drift is recovered, per station, against scan 1.
    rl = parent(FP.fringe_antenna_solutions(track).delay)
    drift_error = [(rl[s, a, 2] - rl[s, a, 1]) - (rl[1, a, 2] - rl[1, a, 1]) - rl_drift[a] * (s - 1) for s in 1:nscans, a in 1:nant]
    @test maximum(abs, drift_error) < 4.0e-12
    # The stations' relative inter-feed phases neither vary from scan to scan
    # nor across the band.
    @test maximum(abs, _synth_variation(sums, (2, 2), (1, 1), (pair, f) -> (pair, f))) < 1.5
    @test maximum(abs, _synth_variation(sums, (2, 2), (1, 1), (pair, f) -> pair)) < 1.5
    # Nor do the different-feed products, to within their noise.
    for fp in ((1, 2), (2, 1)), group in ((pair, f) -> (pair, f), (pair, f) -> pair)
        @test _rms(_synth_variation(sums, fp, (1, 1), group)) < 2.5
    end
    # The amplitude bandpass is removed: each window's amplitude matches its
    # product's mean over the band.
    spread = Float64[]
    for k in unique(k[1:4] for k in keys(sums) if k[4][1] == k[4][2])
        amps = [abs(x.v) for (j, x) in sums if j[1:4] == k]
        append!(spread, amps ./ (sum(amps) / length(amps)) .- 1)
    end
    @test maximum(abs, spread) < 0.02
end
