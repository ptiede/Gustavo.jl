# ── Bandpass model layer (ported onto the unified Calibration framework) ──────
#
# The old type tower — PerChannelBandpassModel / FlatBandpassModel /
# DelayBandpassModel / PolynomialBandpassModel, the *Segmentation types,
# SegmentedBandpassModel / CompositeBandpassModel / BandpassSpec /
# FeedBandpassModel / StationBandpassModel — is GONE. Calibration's unified
# vocabulary (`AbstractGainTerm`, the segmentation types, `GainComponent`,
# `TiedComponent`, `StationGainModel`) is now the single model layer shared by
# both `Bandpass` and `Fringe`.
#
# What remains here:
#   1. Thin BUILDERS that keep the familiar bandpass construction syntax while
#      producing Calibration types. A bandpass station model is a
#      `StationGainModel` whose components are tied with `SharedFeeds` (the
#      reference/common part, applied to both feeds) plus `FeedComponent(partner)`
#      (the relative deviation, applied to the partner feed only). This exactly
#      reproduces the old asymmetric reference/relative feed structure.
#   2. The RESOLVED per-feed view (`ResolvedBandpassModel`) the solver consumes,
#      plus `resolve_bandpass_model` which recovers reference_feed / reference /
#      relative component lists from a `StationGainModel`.
#
# A "spec" is now simply a `Tuple` of `GainComponent`s. The solver's design-column
# / projection math is unchanged (byte-for-byte); only the dispatch types and
# field accessors moved from the old tower onto Calibration types.

using ..Calibration:
    AbstractGainTerm, ConstantTerm, Delay, PolynomialFreq, PerChannel,
    AbstractTimeSegmentation, GlobalTime, PerScan,
    AbstractFrequencySegmentation, GlobalFrequency, PerSpectralWindow, ChannelBlocks,
    GainComponent, TiedComponent, StationGainModel,
    AbstractFeedTying, PerFeed, SharedFeeds, FeedComponent,
    phase_components, logamp_components, validate_station_gain_model,
    is_per_scan, component_is_per_scan, station_model_summary, component_label

# ── Term builders (old atomic-model names → Calibration terms) ─────────────────
PerChannelBandpassModel() = PerChannel()
FlatBandpassModel() = ConstantTerm()
DelayBandpassModel() = Delay()
PolynomialBandpassModel(degree::Integer) = PolynomialFreq(degree)

# ── Segmentation builders (old names → Calibration segmentation) ───────────────
GlobalTimeSegmentation() = GlobalTime()
PerScanTimeSegmentation() = PerScan()
GlobalFrequencySegmentation() = GlobalFrequency()
BlockFrequencySegmentation(block_size::Integer) = ChannelBlocks(block_size)

# ── Component / spec builders ─────────────────────────────────────────────────
# A single segmented component is a `GainComponent`; a composite or a spec is a
# `Tuple` of `GainComponent`s.
SegmentedBandpassModel(term::AbstractGainTerm, time::AbstractTimeSegmentation, freq::AbstractFrequencySegmentation) =
    GainComponent(term, time, freq)

function CompositeBandpassModel(components::GainComponent...)
    isempty(components) && error("CompositeBandpassModel requires at least one component")
    return components
end

# `BandpassSpec` normalizes either a single component or a tuple of components
# into the canonical tuple-of-components "spec" form.
BandpassSpec(component::GainComponent) = (component,)
BandpassSpec(components::Tuple) = components

# Walk a spec's components (a spec is already a tuple of `GainComponent`s).
spec_components(components::Tuple) = components
spec_components(component::GainComponent) = (component,)

# ── FeedBandpassModel: a per-feed (phase, amplitude) carrier ──────────────────
# Kept as a struct so `feed isa FeedBandpassModel` and the keyword constructor
# both work. `phase`/`amplitude` are tuples of `GainComponent`s.
struct FeedBandpassModel
    phase::Tuple
    amplitude::Tuple
end
function FeedBandpassModel(; phase, amplitude)
    return FeedBandpassModel(_as_component_tuple(phase), _as_component_tuple(amplitude))
end
_as_component_tuple(c::GainComponent) = (c,)
_as_component_tuple(t::Tuple) = t

phase_is_per_scan(fm::FeedBandpassModel) = any(component_is_per_scan, fm.phase)
amplitude_is_per_scan(fm::FeedBandpassModel) = any(component_is_per_scan, fm.amplitude)

# ── StationBandpassModel builder → StationGainModel ───────────────────────────
# The reference feed model is applied to BOTH feeds (`SharedFeeds`); the relative
# (partner) model is applied to the partner feed only (`FeedComponent(partner)`).
"""
    StationBandpassModel(; reference_feed, reference, relative) -> StationGainModel

Build a unified `StationGainModel` for a bandpass solve. `reference` and
`relative` are `FeedBandpassModel`s. The reference model is shared by both feeds;
the relative model is the partner-feed-only deviation (partner = reference +
relative), so the two feeds may carry different term/segmentation structure.
"""
function StationBandpassModel(; reference_feed::Integer, reference::FeedBandpassModel, relative::FeedBandpassModel)
    rf = reference_feed in (1, 2) ? Int(reference_feed) : error("reference_feed must be 1 or 2")
    pf = 3 - rf
    phase = (
        map(c -> TiedComponent(c, SharedFeeds()), reference.phase)...,
        map(c -> TiedComponent(c, FeedComponent(pf)), relative.phase)...,
    )
    logamp = (
        map(c -> TiedComponent(c, SharedFeeds()), reference.amplitude)...,
        map(c -> TiedComponent(c, FeedComponent(pf)), relative.amplitude)...,
    )
    return validate_station_gain_model(StationGainModel(phase = phase, logamp = logamp))
end

"The default bandpass station model: per-channel phase & amplitude, global time, global frequency."
function default_bandpass_station_model()
    pc = GainComponent(PerChannel(), GlobalTime(), GlobalFrequency())
    return StationGainModel(
        phase = (TiedComponent(pc, SharedFeeds()), TiedComponent(pc, FeedComponent(2))),
        logamp = (TiedComponent(pc, SharedFeeds()), TiedComponent(pc, FeedComponent(2))),
    )
end

partner_feed_index(reference_feed::Integer) = 3 - reference_feed
reference_feed_label(reference_feed::Integer) = string(reference_feed)

# ── Resolved per-feed view (what the solver consumes) ─────────────────────────
"""
    ResolvedFeedModel(phase, amplitude)

Resolved per-feed component lists (`phase`/`amplitude` are tuples of
`GainComponent`). For the bandpass solver, `reference` holds the common
(`SharedFeeds`) components and `relative` holds the partner-only
(`FeedComponent`) deviation components.
"""
struct ResolvedFeedModel{P <: Tuple, A <: Tuple}
    phase::P
    amplitude::A
end

struct ResolvedBandpassModel{R <: ResolvedFeedModel, S <: ResolvedFeedModel}
    reference_feed::Int
    reference::R
    relative::S
end

"""
    resolve_bandpass_model(model::StationGainModel) -> ResolvedBandpassModel

Recover the reference_feed and the reference (shared) / relative (partner-only)
component lists from a unified `StationGainModel`. Only `SharedFeeds` and
`FeedComponent` tyings are supported (the reference/relative bandpass form);
`PerFeed`/`ReferenceRelative` tyings (the fringe per-feed form) error here.
"""
function resolve_bandpass_model(model::StationGainModel)
    shared_phase = GainComponent[]
    partner_phase = GainComponent[]
    shared_amp = GainComponent[]
    partner_amp = GainComponent[]
    feed_targets = Set{Int}()

    for tc in phase_components(model)
        _classify_tied!(shared_phase, partner_phase, feed_targets, tc)
    end
    for tc in logamp_components(model)
        _classify_tied!(shared_amp, partner_amp, feed_targets, tc)
    end

    length(feed_targets) <= 1 ||
        error("resolve_bandpass_model: both feeds carry a FeedComponent deviation ($(sort!(collect(feed_targets)))); not a reference/relative bandpass model")
    partner = isempty(feed_targets) ? 2 : first(feed_targets)
    reference_feed = 3 - partner

    reference = ResolvedFeedModel(Tuple(shared_phase), Tuple(shared_amp))
    relative = ResolvedFeedModel(Tuple(partner_phase), Tuple(partner_amp))
    return ResolvedBandpassModel(reference_feed, reference, relative)
end

function _classify_tied!(shared, partner, feed_targets, tc::TiedComponent)
    ty = tc.tying
    if ty isa SharedFeeds
        push!(shared, tc.component)
    elseif ty isa FeedComponent
        push!(partner, tc.component)
        push!(feed_targets, ty.feed)
    else
        error("resolve_bandpass_model: tying $(typeof(ty)) is not supported by the bandpass solver; use SharedFeeds + FeedComponent (e.g. via StationBandpassModel)")
    end
    return nothing
end

# ── Per-feed per-scan predicates on the resolved model ────────────────────────
_phase_per_scan(fm::ResolvedFeedModel) = any(component_is_per_scan, fm.phase)
_amp_per_scan(fm::ResolvedFeedModel) = any(component_is_per_scan, fm.amplitude)

phase_is_per_scan(m::ResolvedBandpassModel) = _phase_per_scan(m.reference) || _phase_per_scan(m.relative)
amplitude_is_per_scan(m::ResolvedBandpassModel) = _amp_per_scan(m.reference) || _amp_per_scan(m.relative)

function phase_is_per_scan(m::ResolvedBandpassModel, feed::Integer)
    feed == m.reference_feed && return _phase_per_scan(m.reference)
    feed == partner_feed_index(m.reference_feed) && return _phase_per_scan(m.reference) || _phase_per_scan(m.relative)
    return error("feed must be 1 or 2")
end

function amplitude_is_per_scan(m::ResolvedBandpassModel, feed::Integer)
    feed == m.reference_feed && return _amp_per_scan(m.reference)
    feed == partner_feed_index(m.reference_feed) && return _amp_per_scan(m.reference) || _amp_per_scan(m.relative)
    return error("feed must be 1 or 2")
end
