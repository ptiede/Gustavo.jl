# ── Fringe model construction + structural plan lookups ──────────────────────
#
# The canonical full fringe StationGainModel (`_fringe_model` — used by the
# legacy kwarg wrappers' tests and standalone stage work; the pipeline verbs
# compile the same components step by step), and the structural plan routers
# that locate each stage's θ components by TERM TYPE (per-scan delay, SBD,
# adhoc) — never by hardcoded index. The dispersion term's router is
# `Calibration._dispersion_plan`, beside the model that configures it.
#
# A step whose own model names the component it needs reaches it directly
# through `layout.plantree` instead; a router is only for a component a stage
# must recognize in a model it did not name itself.

# The shared fringe model. Phase, over the global frequency band:
#   - feed-COMMON per-scan constant + delay (`SharedFeeds`): the atmosphere/clock
#     terms that vary scan-to-scan and are the same for both polarization feeds;
#   - an inter-feed offset CONSTANT and an inter-feed offset DELAY
#     (`rel_time × FeedComponent(2)`): the instrumental feed-2−feed-1 phase and
#     group-delay offsets. Their shared time basis is the `rel_time` option.
#     `PerScan()` (the default) fits them per scan, so their scan-to-scan scatter
#     is an instrument-stability diagnostic and no column couples scans; a scan
#     whose cross hands don't detect then leaves feed 2 untied, so its feed-2
#     component can split (`ncomp = 2`) and take an arbitrary per-scan pin — that
#     is harmless to the feed-relative delay (differenced against the SAME feed of
#     the reference, gauge-invariant) but leaves feed 2 meaningful only on scans
#     that actually detect cross hands. `GlobalTime()` fits one offset per station
#     for the whole track (EHT-HOPS / rPICARD assumption), so bright polarized
#     scans pin it and weak scans inherit it through the shared column. Either way
#     the offset absorbs the source's cross-hand phase, which shares its column, so
#     it is estimable only up to that constant;
#   - per-scan rate (`SharedFeeds`): the fringe rate is common to both feeds, so it
#     is tied across them — exactly like the per-scan constant/delay. Solving it
#     `PerFeed` instead lets a spurious inter-feed rate (`rate₂ − rate₁`) float on
#     noise; since the Rate phase is `2π·rate·(t − t0_global)` with `t0` the
#     WHOLE-TRACK reference, that per-feed noise is multiplied by a per-scan lever
#     arm of hours, injecting a large, arbitrary, scan-to-scan cross-hand phase
#     jump. The inter-feed rate is negligible (EHT-HOPS), so tie it; a genuine
#     offset would be a `GlobalTime × FeedComponent(2)` rate term (the analog of
#     the inter-feed constant/delay), not PerFeed;
#   - a per-AP adhoc-phase constant (`SharedFeeds`): residual atmospheric phase is
#     non-birefringent (common to both feeds), so it is solved feed-common — which
#     denoises it and, crucially, contributes ZERO inter-feed phase. A `PerFeed`
#     adhoc lets per-AP solve noise differ between feeds and so injects spurious
#     cross-hand scatter on top of the instrumental inter-feed offset.
# Log-amplitude empty. `solve_station_systems!` reads the θ columns this model
# declares — so the global-vs-per-scan split is a model choice, not solver code.
function _fringe_model(;
        dispersion::Bool = false, sbd_freq_groups = nothing,
        rel_time::AbstractTimeSegmentation = PerScan(),
    )
    # Ionospheric dispersion: per-scan, feed-common differential TEC (TECU),
    # phase = K·θ·(1/f0 − 1/f). NOT solved by the FFT search — measured by the
    # per-scan (Δτ, dTEC) band-phasor refinement (`_refine_scan_dispersion!`),
    # which jointly updates the per-scan delay (the two covary over any finite
    # band). `Dispersion` is its routing signature. Only included when the band
    # layout can constrain it.
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
    # the cross-band solution is untouched. `FreqGroups` is the routing
    # signature (excluded from stage-B).
    sbd = sbd_freq_groups === nothing ? (;) : (
            sbd = (
                delay = TiedComponent(GainComponent(Delay(), PerScan(), FreqGroups(sbd_freq_groups)), SharedFeeds()),
                constant = TiedComponent(GainComponent(ConstantTerm(), PerScan(), FreqGroups(sbd_freq_groups)), SharedFeeds()),
            ),
        )
    return StationGainModel(
        phase = merge(
            (
                atmos = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), SharedFeeds()),
                rel_phase = TiedComponent(GainComponent(ConstantTerm(), rel_time, GlobalFrequency()), FeedComponent(2)),
                mbd = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), SharedFeeds()),
                rel_delay = TiedComponent(GainComponent(Delay(), rel_time, GlobalFrequency()), FeedComponent(2)),
                rate = TiedComponent(GainComponent(Rate(), PerScan(), GlobalFrequency()), SharedFeeds()),
            ),
            disp, sbd,
            (
                # Phase bandpass: one free offset per channel, stable across the
                # observation (HOPS-style), per feed. Captures the residual
                # nonlinear-in-frequency instrumental phase that the per-scan (linear) delay
                # cannot represent. Solved by a dedicated frequency-stationization stage
                # (`solve_phase_bandpass!`), not by the delay/rate search.
                bandpass = TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), ChannelBlocks(1)), PerFeed()),
                adhoc = TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), SharedFeeds()),
            ),
        ),
        # Amplitude bandpass: per-channel, time-stable, per-feed log-amplitude — the
        # per-station instrumental amplitude shape (filterbank passband), FLATTENED
        # from the calibrator. Solved by the bandpass stage via a pluggable
        # `WLSEstimator`-based smoother (`solve_amp_bandpass!`); the SNR gate drops
        # no-signal channels, which the smoother then estimates (or, for
        # `free_bandpass`, leaves at gain 1). The absolute level stays the a-priori
        # amplitude cal's job.
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
# (`FreqGroups`), NOT a feed-specific inter-feed delay (`FeedComponent`). Shared by
# `FringeFit`'s own wideband delay (`mbd`) and `DispersionSBDFit`'s private
# per-scan delay-refinement column — but each lives in its OWN step's private
# model, so `findfirst` over either step's own component list finds the right
# one without ambiguity; no positional trick is needed to tell them apart.
_is_perscan_delay(tc) =
    tc.component.term isa Delay && !(tc.component.time isa GlobalTime) &&
    tc.component.freq isa GlobalFrequency && tc.tying isa SharedFeeds

# The per-band-group single-band delay and its companion constant, which are
# fit together by `refine_scan_sbd!`.
_is_sbd_delay(tc) = tc.component.term isa Delay && tc.component.freq isa FreqGroups
_is_sbd_constant(tc) =
    tc.component.term isa ConstantTerm && tc.component.freq isa FreqGroups

# The step's own feed-common wideband delay — `FringeFit`'s `mbd` when called
# on the fringe step's own `(model, layout)`, or `DispersionSBDFit`'s private
# delay-refinement column when called on the refine step's own — either way
# the ONLY `_is_perscan_delay`-signatured component in that step's private
# model.
function _perscan_delay_plan(model, layout)
    i = findfirst(_is_perscan_delay, phase_components(model))
    return i === nothing ? nothing : layout.plans[i]
end

# The SBD components' plans `(dplan, cplan, freqgroups)` (per-scan per-band-group
# delay + companion constant), or `nothing` when the model carries none.
function _sbd_plans(model, layout)
    pcs = phase_components(model)
    i = findfirst(_is_sbd_delay, pcs)
    i === nothing && return nothing
    j = findfirst(_is_sbd_constant, pcs)
    j === nothing && error("SBD delay component present without its companion constant")
    return (dplan = layout.plans[i], cplan = layout.plans[j], freqgroups = pcs[i].component.freq.ranges)
end

# Index of the adhoc component (the per-integration phase term) in the flat
# phase-component order (= `layout.plans` order).
_adhoc_idx(model) = findfirst(tc -> tc.component.time isa PerIntegration, phase_components(model))

# The adhoc component's plan (the per-integration phase term).
_adhoc_plan(model, layout) = layout.plans[_adhoc_idx(model)]
