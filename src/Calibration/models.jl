# ── Gain components and station models ───────────────────────────────────────
#
# Composition hierarchy (replaces the old StationBandpassModel / FeedBandpassModel
# / BandpassSpec / SegmentedBandpassModel tower):
#
#   StationGainModel
#     .phase  :: Tuple of TiedComponent   →  Σ phase contributions
#     .logamp :: Tuple of TiedComponent   →  Σ log-amplitude contributions
#       gain(t, f) = exp(Σ logamp) · cis(Σ phase)
#
#   TiedComponent
#     .component :: GainComponent(term, time_seg, freq_seg)
#     .tying     :: how the two feeds share (or don't share) this component
#
# One `StationGainModel` is shared by all antennas in a solve (each antenna gets
# its own parameter values, not its own structure). Heterogeneous per-station
# model *structure* is a deliberate non-goal of this first cut.

"""
    GainComponent(term, time, freq)

A single gain term replicated over a time segmentation and a frequency
segmentation. One parameter block is allocated per (time segment, frequency
segment).
"""
struct GainComponent{T <: AbstractGainTerm, TS <: AbstractTimeSegmentation, FS <: AbstractFrequencySegmentation}
    term::T
    time::TS
    freq::FS
end

# ── Feed tying ───────────────────────────────────────────────────────────────

abstract type AbstractFeedTying end

"Independent parameters per feed — feed 1 and feed 2 solved separately."
struct PerFeed <: AbstractFeedTying end

"One shared parameter block used by both feeds."
struct SharedFeeds <: AbstractFeedTying end

"""
    ReferenceRelative(reference_feed)

The `reference_feed` reads a reference block; the partner feed reads the
reference block PLUS a relative-deviation block (partner = reference + relative).
This reproduces the old absolute/relative bandpass feed pattern when both feeds
share the *same* term and segmentation. For an asymmetric bandpass model (the
two feeds use different terms/segmentations) use a `SharedFeeds` component for
the common part plus a `FeedComponent` for the partner-only deviation.
"""
struct ReferenceRelative <: AbstractFeedTying
    reference_feed::Int
    function ReferenceRelative(reference_feed::Integer)
        reference_feed in (1, 2) || error("ReferenceRelative reference_feed must be 1 or 2")
        return new(Int(reference_feed))
    end
end

"""
    FeedComponent(feed)

The component applies to one `feed` (1 or 2) only — the other feed gets no
contribution from it. Allocates a single parameter block, assigned to `feed`.
This is the primitive that expresses a partner-feed-only deviation, so the old
asymmetric reference/relative bandpass model decomposes as a `SharedFeeds`
common part plus a `FeedComponent(partner)` deviation.
"""
struct FeedComponent <: AbstractFeedTying
    feed::Int
    function FeedComponent(feed::Integer)
        feed in (1, 2) || error("FeedComponent feed must be 1 or 2")
        return new(Int(feed))
    end
end

"""
    TiedComponent(component, tying)

A `GainComponent` together with its across-feed tying. `TiedComponent(component)`
defaults to `PerFeed()`.
"""
struct TiedComponent{C <: GainComponent, T <: AbstractFeedTying}
    component::C
    tying::T
end
TiedComponent(component::GainComponent) = TiedComponent(component, PerFeed())

term(tc::TiedComponent) = tc.component.term
time_segmentation(tc::TiedComponent) = tc.component.time
freq_segmentation(tc::TiedComponent) = tc.component.freq

# Number of distinct feed-blocks this tying allocates per (ant, tseg, fseg).
#   PerFeed           → 2 (feed1, feed2)
#   SharedFeeds       → 1 (shared)
#   ReferenceRelative → 2 (reference, relative)
nfeed_blocks(::PerFeed) = 2
nfeed_blocks(::SharedFeeds) = 1
nfeed_blocks(::ReferenceRelative) = 2
nfeed_blocks(::FeedComponent) = 1

"""
    StationGainModel(; phase = (), logamp = ())

The gain model for a station: a tuple of phase `TiedComponent`s and a tuple of
log-amplitude `TiedComponent`s. `gain = exp(Σ logamp) · cis(Σ phase)`.
"""
struct StationGainModel{P <: Tuple, A <: Tuple}
    phase::P
    logamp::A
end
StationGainModel(; phase = (), logamp = ()) =
    StationGainModel(_as_tied_tuple(phase), _as_tied_tuple(logamp))

_as_tied_tuple(t::Tuple) = map(_as_tied, t)
_as_tied_tuple(c) = (_as_tied(c),)
_as_tied(tc::TiedComponent) = tc
_as_tied(c::GainComponent) = TiedComponent(c)

phase_components(m::StationGainModel) = m.phase
logamp_components(m::StationGainModel) = m.logamp

# ── Per-component / per-model time-segmentation queries ──────────────────────
# Used by solvers to route global-time vs per-scan components.
is_per_scan(::AbstractTimeSegmentation) = false
is_per_scan(::PerScan) = true
is_per_scan(::PerIntegration) = true
component_is_per_scan(tc::TiedComponent) = is_per_scan(time_segmentation(tc))
component_is_per_scan(c::GainComponent) = is_per_scan(c.time)

phase_is_per_scan(m::StationGainModel) = any(component_is_per_scan, m.phase)
amplitude_is_per_scan(m::StationGainModel) = any(component_is_per_scan, m.logamp)

# ── Validation ───────────────────────────────────────────────────────────────
# A frequency-only / time-only term must not be paired with a segmentation that
# makes it degenerate-free; the linear-algebra layer tolerates redundancy, so
# validation here is light — mainly catching empty models.
function validate_station_gain_model(m::StationGainModel)
    (isempty(m.phase) && isempty(m.logamp)) &&
        error("StationGainModel has neither phase nor log-amplitude components")
    return m
end

# ── Summaries ────────────────────────────────────────────────────────────────
tying_label(::PerFeed) = "perfeed"
tying_label(::SharedFeeds) = "shared"
tying_label(t::ReferenceRelative) = "refrel$(t.reference_feed)"
tying_label(t::FeedComponent) = "feed$(t.feed)"

function component_label(tc::TiedComponent)
    return string(
        term_label(term(tc)), "×",
        time_segmentation_label(time_segmentation(tc)), "×",
        frequency_segmentation_label(freq_segmentation(tc)), "[",
        tying_label(tc.tying), "]",
    )
end

time_segmentation_label(::GlobalTime) = "global"
time_segmentation_label(::PerScan) = "perscan"
time_segmentation_label(::PerIntegration) = "perint"
time_segmentation_label(s::TimeBlocks) = "blocks$(s.duration_hr)h"
time_segmentation_label(::InstrumentScans) = "instrscans"

frequency_segmentation_label(::GlobalFrequency) = "global"
frequency_segmentation_label(::PerSpectralWindow) = "perspw"
frequency_segmentation_label(s::ChannelBlocks) = "chblocks$(s.block_size)"

function station_model_summary(name, m::StationGainModel)
    ph = isempty(m.phase) ? "—" : join(component_label.(m.phase), " + ")
    am = isempty(m.logamp) ? "—" : join(component_label.(m.logamp), " + ")
    return string(name, "  phase(", ph, ")  logamp(", am, ")")
end
