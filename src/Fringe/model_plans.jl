# ── Fringe model construction + structural plan lookups ──────────────────────
#
# The canonical full fringe StationGainModel (`_fringe_model` — used by the
# legacy kwarg wrappers' tests and standalone stage work; the pipeline verbs
# compile the same components step by step), and the structural plan routers
# that locate each stage's θ components by TERM TYPE (dispersion, per-scan
# delay, SBD, bandpass, adhoc) — never by hardcoded index. Relocated verbatim
# from the deleted monolith (`pipeline.jl`).

# The shared fringe model. Phase, over the global frequency band:
#   - feed-COMMON per-scan constant + delay (`SharedFeeds`): the atmosphere/clock
#     terms that vary scan-to-scan and are the same for both polarization feeds;
#   - a GLOBAL R–L offset CONSTANT (`GlobalTime × FeedComponent(2)`): the
#     instrumental feed-2−feed-1 phase offset, stable across the whole observation
#     (EHT-HOPS / rPICARD assumption). Solved once from all scans' cross hands so
#     the bright polarized scans pin it and weak scans inherit it (feeds that
#     would otherwise split into `ncomp = 2` are tied);
#   - an R–L offset DELAY whose time basis is the `rl_delay` option: `:global`
#     (`GlobalTime × FeedComponent(2)`, the default — one instrumental offset for
#     the track, tying weak scans through a shared column exactly like the R–L
#     constant) or `:perscan` (`PerScan × FeedComponent(2)` — fit per scan so its
#     scatter is an instrument-stability diagnostic). Under `:perscan` a scan whose
#     cross hands don't detect no longer ties feed 2 through a shared column, so its
#     feed-2 component can split (`ncomp = 2`) and take an arbitrary per-scan pin;
#     that is harmless to the feed-relative delay (differenced against the SAME feed
#     of the reference, gauge-invariant) but leaves feed 2 meaningful only on scans
#     that actually detect cross hands;
#   - per-scan rate (`SharedFeeds`): the fringe rate is common to both feeds, so it
#     is tied across them — exactly like the per-scan constant/delay. Solving it
#     `PerFeed` instead lets a spurious R–L rate (`rate₂ − rate₁`) float on noise;
#     since the Rate phase is `2π·rate·(t − t0_global)` with `t0` the WHOLE-TRACK
#     reference, that per-feed noise is multiplied by a per-scan lever arm of hours,
#     injecting a large, arbitrary, scan-to-scan R–L (RL/RR) phase jump. R–L rate is
#     negligible (EHT-HOPS), so tie it; a genuine offset would be a `GlobalTime ×
#     FeedComponent(2)` rate term (the analog of the R–L constant/delay), not PerFeed;
#   - a per-AP adhoc-phase constant (`SharedFeeds`): residual atmospheric phase is
#     non-birefringent (common to both feeds), so it is solved feed-common — which
#     denoises it and, crucially, contributes ZERO R–L phase. A `PerFeed` adhoc lets
#     per-AP solve noise differ between feeds and so injects spurious R–L (RL/RR)
#     scatter on top of the stable global instrumental R–L offset.
# Log-amplitude empty. `solve_station_systems!` reads the off1 columns this model
# declares — so the global-vs-per-scan split is a model choice, not solver code.
function _fringe_model(; dispersion::Bool = false, sbd_bands = nothing, rl_delay::Symbol = :global)
    rl_delay in (:global, :perscan) ||
        error("rl_delay must be :global or :perscan (got $rl_delay)")
    # The R–L (feed-2−feed-1) delay's time basis: `GlobalTime` fits ONE offset per
    # station for the whole track (bright polarized scans pin it, weak scans inherit
    # it — the robust default); `PerScan` fits it per scan, so its scan-to-scan
    # scatter is a direct instrument-stability diagnostic, at the cost that a scan
    # with no cross-hand detection leaves feed 2 untied (see the comment above).
    rl_time = rl_delay === :perscan ? PerScan() : GlobalTime()
    # Ionospheric dispersion: per-scan, feed-common differential TEC (TECU),
    # phase = K·θ·(1/f0 − 1/f). NOT solved by the FFT search — measured by the
    # per-scan (Δτ, dTEC) band-phasor refinement (`_refine_scan_dispersion!`),
    # which jointly updates the per-scan delay (the two covary over any finite
    # band). `Dispersion` is its routing signature, like `PerChannel` for the
    # bandpass. Only included when the band layout can constrain it.
    disp = dispersion ?
        (TiedComponent(GainComponent(Dispersion(), PerScan(), GlobalFrequency()), SharedFeeds()),) : ()
    # Per-scan per-band-group single-band delay (fourfit's SBD): a station's
    # per-band signal path can move relative to its phase-cal tones between
    # scans (~30 ns on VR2505's YJ), which neither the wideband delay (one slope
    # across all groups) nor the time-invariant per-channel bandpass can track.
    # Measured from WITHIN-band chunk slopes by `_refine_scan_sbd!` — nearly
    # orthogonal to the cross-band observables that set the MBD delay and dTEC.
    # The Delay coordinate is (f − f0) with the GLOBAL f0, so correcting a group
    # slope about the group's own centre νg needs the companion per-group
    # constant −2πτ(νg − f0): net phase 2πτ(f − νg), zero at the group centre —
    # the cross-band solution is untouched. `FrequencyBands` is the routing
    # signature (excluded from stage-B).
    sbd = sbd_bands === nothing ? () : (
            TiedComponent(GainComponent(Delay(), PerScan(), FrequencyBands(sbd_bands)), SharedFeeds()),
            TiedComponent(GainComponent(ConstantTerm(), PerScan(), FrequencyBands(sbd_bands)), SharedFeeds()),
        )
    return StationGainModel(
        phase = (
            TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), SharedFeeds()),
            TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), GlobalFrequency()), FeedComponent(2)),
            TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), SharedFeeds()),
            TiedComponent(GainComponent(Delay(), rl_time, GlobalFrequency()), FeedComponent(2)),
            TiedComponent(GainComponent(Rate(), PerScan(), GlobalFrequency()), SharedFeeds()),
            disp...,
            sbd...,
            # Phase bandpass: per-channel, stable across the observation (HOPS-style),
            # per feed. Captures the residual nonlinear-in-frequency instrumental phase
            # that the per-scan (linear) delay cannot represent. Solved by a dedicated
            # frequency-stationization stage (`_solve_phase_bandpass!`), not by the
            # delay/rate search — `PerChannel` is its signature, used to route it.
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), GlobalFrequency()), PerFeed()),
            TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), SharedFeeds()),
        ),
        # Amplitude bandpass: per-channel, time-stable, per-feed log-amplitude — the
        # per-station instrumental amplitude shape (filterbank passband), FLATTENED
        # from the calibrator. Solved by the bandpass stage via a pluggable
        # `AbstractBandpassSmoother` (`solve_amp_bandpass!`); the SNR gate drops
        # no-signal channels, which the smoother then estimates (or, for `FreeBandpass`,
        # leaves at gain 1). The absolute level stays the a-priori amplitude cal's job.
        logamp = (
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), GlobalFrequency()), PerFeed()),
        ),
    )
end

# Stage-B engine components `(plan, kind)` — the delay/rate/const terms solved by
# the fringe search + stationization. EXCLUDES the per-integration adhoc term
# (`PerIntegration`, solved per-AP) and the phase bandpass (`PerChannel`, solved
# per-channel). `kind` is derived from the term type, so the routing is structural
# (by term/segmentation), not by hardcoded indices.

function _dispersion_plan(model, layout)
    i = findfirst(tc -> tc.component.term isa Dispersion, model.phase)
    return i === nothing ? nothing : layout.plans[i]
end

# Whether this geometry gets a dTEC term. No model, no term. With one, the
# `require_band_separation` gate asks whether the band layout can separate 1/ν
# from a linear delay: several sub-bands over a wide fractional bandwidth (VGOS
# 3–10.7 GHz qualifies; a single contiguous band cannot constrain the curvature
# and the term would just soak up delay).
_dispersion_enabled(::Nothing, ::DataGeometry) = false
function _dispersion_enabled(dm::DispersionModel, geom::DataGeometry)
    dm.require_band_separation || return true
    nb = length(unique(geom.spw_of_chan))
    fmin, fmax = extrema(geom.channel_freqs)
    return nb >= 4 && fmax / fmin > 1.3
end

function _perscan_delay_plan(model, layout)
    i = findfirst(
        tc -> tc.component.term isa Delay && !(tc.component.time isa GlobalTime) &&
            tc.component.freq isa GlobalFrequency,   # NOT the per-band-group SBD delay
        model.phase,
    )
    return i === nothing ? nothing : layout.plans[i]
end

# The SBD components' plans `(dplan, cplan, bands)` (per-scan per-band-group
# delay + companion constant), or `nothing` when the model carries none.
function _sbd_plans(model, layout)
    i = findfirst(
        tc -> tc.component.term isa Delay && tc.component.freq isa FrequencyBands,
        model.phase,
    )
    i === nothing && return nothing
    j = findfirst(
        tc -> tc.component.term isa ConstantTerm && tc.component.freq isa FrequencyBands,
        model.phase,
    )
    j === nothing && error("SBD delay component present without its companion constant")
    return (dplan = layout.plans[i], cplan = layout.plans[j], bands = model.phase[i].component.freq.ranges)
end

# Resolve the `sbd` option (`:auto | true | false`) into the band-group channel
# ranges, or `nothing` when disabled/unconstrainable. A single band group is
# fully degenerate with the stage-B wideband delay, so the term needs ≥ 2.
function _sbd_bands(sbd, geom::DataGeometry)
    sbd === false && return nothing
    sbd === true || sbd === :auto || error("sbd must be :auto, true or false (got $sbd)")
    bands = fringe_band_groups(geom.channel_freqs)
    return length(bands) >= 2 ? bands : nothing
end

# ant → representative-station map for co-located groups (separation below
# `max_sep` meters, e.g. the Onsala twins at ~75 m). Used to TIE the per-scan
# dTEC solve: co-located stations see the same ionosphere, so a differential
# TEC between them is pure solve error (VR2505 gave OE−OW = −2.4 TECU untied).
function _colocated_ties(antennas; max_sep::Real = 1000.0)
    n = length(antennas)
    xyz = antennas.station_xyz
    ties = collect(1:n)
    # Guard: missing/degenerate positions (synthetic tables often carry zeros)
    # would tie the whole array into one node; require a real VLBI-scale array
    # before trusting the positions at all.
    maxd2 = 0.0
    for j in 2:n, i in 1:(j - 1)
        maxd2 = max(maxd2, sum(abs2, Float64.(xyz[j]) .- Float64.(xyz[i])))
    end
    maxd2 > (10.0e3)^2 || return ties
    # NOTE the explicit nesting: in a comma-nested `for j, i` a `break` exits
    # BOTH levels — the old form stopped after the FIRST co-located pair in
    # the array (on VR2505 it tied Onsala OE-OW and silently never examined
    # the Wettzell twins, so WN-WS kept polluting the adhoc/bandpass solves
    # and the dTEC tie).
    for j in 2:n
        for i in 1:(j - 1)
            d2 = sum(abs2, Float64.(xyz[j]) .- Float64.(xyz[i]))
            if d2 <= float(max_sep)^2
                ties[j] = ties[i]
                break
            end
        end
    end
    return ties
end

# Baseline pairs joining co-located stations (same `_colocated_ties` group, e.g.
# the Onsala OE-OW and Wettzell WN-WS twins), both orders. These intra-site
# baselines carry enormous SNR but non-closing (crosstalk) frequency/time
# structure, so any solve that pools ALL baselines with ~snr² weights — the
# per-AP adhoc phasing and the phase/amp bandpass — is pulled to fit the
# crosstalk instead of the sky, splitting it into the two twins' gains and
# decohering their SKY baselines (on VR2505: OE-OW band-avg coherence 0.95 →
# 0.34 through the bandpass stages; WN's long baselines 1.00 → 0.6-0.7 through
# adhoc). Stage B is protected by the stationize closure screen; these stages
# pool raw residuals and need the exclusion up front. Empty when the array has
# no co-located pair (or positions are untrustworthy — see `_colocated_ties`).
function _colocated_pair_set(antennas; max_sep::Real = 1000.0)
    t = _colocated_ties(antennas; max_sep = max_sep)
    excl = Set{Tuple{Int, Int}}()
    for j in 2:length(t), i in 1:(j - 1)
        t[i] == t[j] || continue
        push!(excl, (i, j))
        push!(excl, (j, i))
    end
    return excl
end

# Index of the adhoc component (the per-integration phase term) within `model.phase`.
_adhoc_idx(model) = findfirst(tc -> tc.component.time isa PerIntegration, model.phase)

# The adhoc component's plan (the per-integration phase term).
_adhoc_plan(model, layout) = layout.plans[_adhoc_idx(model)]

# Whether the adhoc (per-integration) phase component is feed-common (`SharedFeeds`),
# so `solve_adhoc_phasing` solves one feed-common track and contributes zero R–L.
_adhoc_shared(model) = model.phase[_adhoc_idx(model)].tying isa SharedFeeds

# The phase-bandpass component's plan (the per-channel phase term), or `nothing`
# if the model carries no bandpass component.
function _bandpass_plan(model, layout)
    i = findfirst(tc -> tc.component.term isa PerChannel, model.phase)
    return i === nothing ? nothing : layout.plans[i]
end

# The amplitude-bandpass component's plan (the per-channel log-amp term), or
# `nothing`. Log-amp plans follow the phase plans in `layout.plans` (offset
# `nphase`), so index the `PerChannel` position within `model.logamp`.
function _amp_bandpass_plan(model, layout)
    j = findfirst(tc -> tc.component.term isa PerChannel, model.logamp)
    return j === nothing ? nothing : layout.plans[layout.nphase + j]
end

