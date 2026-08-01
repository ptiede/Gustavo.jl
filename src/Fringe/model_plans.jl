# ── Fringe model construction + structural plan lookups ──────────────────────
#
# The canonical full fringe StationGainModel (`_fringe_model` — used by the
# legacy kwarg wrappers' tests and standalone stage work; the pipeline verbs
# compile the same components step by step), and the structural plan routers
# that locate each stage's θ components by TERM TYPE (per-scan delay, SBD,
# bandpass, adhoc) — never by hardcoded index. The dispersion term's router is
# `Calibration._dispersion_plan`, beside the model that configures it.

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
# Log-amplitude empty. `solve_station_systems!` reads the θ columns this model
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
    # band). `Dispersion` is its routing signature, like `ChannelBlocks` for the
    # bandpass. Only included when the band layout can constrain it.
    disp = dispersion ?
        (dtec = TiedComponent(GainComponent(Dispersion(), PerScan(), GlobalFrequency()), SharedFeeds()),) : (;)
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
    sbd = sbd_bands === nothing ? (;) : (
            sbd = (
                delay = TiedComponent(GainComponent(Delay(), PerScan(), FrequencyBands(sbd_bands)), SharedFeeds()),
                constant = TiedComponent(GainComponent(ConstantTerm(), PerScan(), FrequencyBands(sbd_bands)), SharedFeeds()),
            ),
        )
    return StationGainModel(
        phase = merge(
            (
                atmos = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), SharedFeeds()),
                rl_phase = TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), GlobalFrequency()), FeedComponent(2)),
                mbd = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), SharedFeeds()),
                rl_delay = TiedComponent(GainComponent(Delay(), rl_time, GlobalFrequency()), FeedComponent(2)),
                rate = TiedComponent(GainComponent(Rate(), PerScan(), GlobalFrequency()), SharedFeeds()),
            ),
            disp, sbd,
            (
                # Phase bandpass: one free offset per channel, stable across the
                # observation (HOPS-style), per feed. Captures the residual
                # nonlinear-in-frequency instrumental phase that the per-scan (linear) delay
                # cannot represent. Solved by a dedicated frequency-stationization stage
                # (`solve_phase_bandpass!`), not by the delay/rate search — the
                # `ChannelBlocks` segmentation is its routing signature.
                bandpass = TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), ChannelBlocks(1)), PerFeed()),
                adhoc = TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), SharedFeeds()),
            ),
        ),
        # Amplitude bandpass: per-channel, time-stable, per-feed log-amplitude — the
        # per-station instrumental amplitude shape (filterbank passband), FLATTENED
        # from the calibrator. Solved by the bandpass stage via a pluggable
        # `AbstractBandpassSmoother` (`solve_amp_bandpass!`); the SNR gate drops
        # no-signal channels, which the smoother then estimates (or, for `FreeBandpass`,
        # leaves at gain 1). The absolute level stays the a-priori amplitude cal's job.
        logamp = (
            bandpass = TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), ChannelBlocks(1)), PerFeed()),
        ),
    )
end

# ── Router signatures ────────────────────────────────────────────────────────
#
# Each plan router locates a θ block by `findfirst` over a compiled component's
# (term, time segmentation, frequency segmentation, tying) types. The predicate
# is the SIGNATURE: it is what a model must contain for the router to find
# anything, so `validate_model` checks the same predicates the routers use
# rather than a separate description of them. A term-level check is too coarse —
# `Delay × GlobalTime` is a `Delay` and still leaves `_perscan_delay_plan`
# empty-handed.

# The feed-common wideband delay: NOT the per-band-group SBD delay
# (`FrequencyBands`), NOT a feed-specific R–L delay (`FeedComponent`). Shared by
# `FringeFit`'s own wideband delay (`mbd`) and `DispersionSBDFit`'s private
# per-scan delay-refinement column — but each lives in its OWN step's private
# model, so `findfirst` over either step's own component list finds the right
# one without ambiguity; no positional trick is needed to tell them apart.
_is_perscan_delay(tc) =
    tc.component.term isa Delay && !(tc.component.time isa GlobalTime) &&
    tc.component.freq isa GlobalFrequency && tc.tying isa SharedFeeds

# The per-band-group single-band delay and its companion constant, which are
# fit together by `refine_scan_sbd!`.
_is_sbd_delay(tc) = tc.component.term isa Delay && tc.component.freq isa FrequencyBands
_is_sbd_constant(tc) =
    tc.component.term isa ConstantTerm && tc.component.freq isa FrequencyBands

# The step's own feed-common wideband delay — `FringeFit`'s `mbd` when called
# on the fringe step's own `(model, layout)`, or `DispersionSBDFit`'s private
# delay-refinement column when called on the refine step's own — either way
# the ONLY `_is_perscan_delay`-signatured component in that step's private
# model.
function _perscan_delay_plan(model, layout)
    i = findfirst(_is_perscan_delay, phase_components(model))
    return i === nothing ? nothing : layout.plans[i]
end

# The SBD components' plans `(dplan, cplan, bands)` (per-scan per-band-group
# delay + companion constant), or `nothing` when the model carries none.
function _sbd_plans(model, layout)
    pcs = phase_components(model)
    i = findfirst(_is_sbd_delay, pcs)
    i === nothing && return nothing
    j = findfirst(_is_sbd_constant, pcs)
    j === nothing && error("SBD delay component present without its companion constant")
    return (dplan = layout.plans[i], cplan = layout.plans[j], bands = pcs[i].component.freq.ranges)
end

# Index of the adhoc component (the per-integration phase term) in the flat
# phase-component order (= `layout.plans` order).
_adhoc_idx(model) = findfirst(tc -> tc.component.time isa PerIntegration, phase_components(model))

# The adhoc component's plan (the per-integration phase term).
_adhoc_plan(model, layout) = layout.plans[_adhoc_idx(model)]

# Whether the adhoc (per-integration) phase component is feed-common (`SharedFeeds`),
# so `solve_adhoc_phasing` solves one feed-common track and contributes zero R–L.
_adhoc_shared(model) = phase_components(model)[_adhoc_idx(model)].tying isa SharedFeeds

# The phase-bandpass component's plan, or `nothing` if the model carries no
# bandpass component. `Calibration._is_bandpass` is the signature.
function _bandpass_plan(model, layout)
    i = findfirst(_is_bandpass, phase_components(model))
    return i === nothing ? nothing : layout.plans[i]
end

# The amplitude-bandpass component's plan, or `nothing`. Log-amp plans follow
# the phase plans in `layout.plans` (offset `nphase`), so index the bandpass
# position within `model.logamp`.
function _amp_bandpass_plan(model, layout)
    j = findfirst(_is_bandpass, logamp_components(model))
    return j === nothing ? nothing : layout.plans[layout.nphase + j]
end

