# ── Gain components and station models ───────────────────────────────────────
#
# Composition hierarchy:
#
#   StationGainModel
#     .phase  :: NamedTuple, name → TiedComponent   →  Σ phase contributions
#     .logamp :: NamedTuple, name → TiedComponent   →  Σ log-amplitude contributions
#       gain(t, f) = exp(Σ logamp) · cis(Σ phase)
#
# Each component carries a user-specified name (the NamedTuple key), unique
# within its group. A value that is itself a NamedTuple is a named subtree — one
# list element that compiled to several components (e.g. a single-band delay's
# delay + companion constant), namespaced under the element's key. `θ` addresses
# the parameters the same way: `θ.phase.<name>` / `θ.logamp.<name>`. The layout
# and the forward map consume the depth-first flat view `phase_components` /
# `logamp_components`, whose order is the list order.
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
        reference_feed in (1, 2) || throw(
            ArgumentError(
                "ReferenceRelative reference_feed must be 1 or 2, got $reference_feed"
            )
        )
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
        feed in (1, 2) ||
            throw(ArgumentError("FeedComponent feed must be 1 or 2, got $feed"))
        return new(Int(feed))
    end
end

"""
    TiedComponent(component, tying)
    TiedComponent(term, time, freq, tying = PerFeed())

A `GainComponent` together with its across-feed tying. `TiedComponent(component)`
defaults to `PerFeed()`; the four-argument form builds the `GainComponent`
inline — `TiedComponent(Delay(), PerScan(), GlobalFrequency(), SharedFeeds())`.
"""
struct TiedComponent{C <: GainComponent, T <: AbstractFeedTying}
    component::C
    tying::T
end
TiedComponent(component::GainComponent) = TiedComponent(component, PerFeed())
TiedComponent(
    term::AbstractGainTerm, time::AbstractTimeSegmentation,
    freq::AbstractFrequencySegmentation, tying::AbstractFeedTying = PerFeed(),
) = TiedComponent(GainComponent(term, time, freq), tying)

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

# The feed-node (column of a component's leaf `:Feed`/`:node` axis) a feed reads
# its PRIMARY block from, or 0 when the tying carries no block for that feed:
# `PerFeed` keeps the two feeds as distinct nodes; `SharedFeeds` folds them to
# one; `FeedComponent(k)` keeps only feed k; `ReferenceRelative`'s primary block
# is the shared reference (node 1) for both feeds.
_feed_node(::PerFeed, feed::Integer) = feed
_feed_node(::SharedFeeds, feed::Integer) = 1
_feed_node(::ReferenceRelative, feed::Integer) = 1
_feed_node(t::FeedComponent, feed::Integer) = feed == t.feed ? 1 : 0

# The SECONDARY block a feed also reads, or 0 for none. Only `ReferenceRelative`
# has one: the partner feed (non-reference) adds its own relative block (node 2)
# on top of the shared reference block.
_feed_node2(::AbstractFeedTying, feed::Integer) = 0
_feed_node2(t::ReferenceRelative, feed::Integer) = feed == 3 - t.reference_feed ? 2 : 0

"""
    StationGainModel(; phase = (;), logamp = (;))

The gain model for a station: a `NamedTuple` of named phase `TiedComponent`s and
a `NamedTuple` of named log-amplitude `TiedComponent`s.
`gain = exp(Σ logamp) · cis(Σ phase)`. The keys are the component names, unique
within each group; a value may itself be a `NamedTuple` — a named subtree for one
element that compiled to several components. A `GainComponent` value is coerced to
a `PerFeed` `TiedComponent`.
"""
struct StationGainModel{P <: NamedTuple, A <: NamedTuple}
    phase::P
    logamp::A
end
StationGainModel(; phase = (;), logamp = (;)) =
    StationGainModel(_named_components(phase), _named_components(logamp))

_named_components(nt::NamedTuple) = map(_as_tied, nt)
_named_components(::Tuple{}) = (;)
_named_components(t::Tuple) = throw(
    ArgumentError(
        "StationGainModel components must be named: pass a NamedTuple " *
            "(e.g. `phase = (delay = TiedComponent(...),)`), not a bare tuple.",
    ),
)
_as_tied(tc::TiedComponent) = tc
_as_tied(c::GainComponent) = TiedComponent(c)
_as_tied(nt::NamedTuple) = map(_as_tied, nt)   # a named subtree

# Depth-first flat tuple of the `TiedComponent`s in a named component tree, names
# dropped — the order the layout and forward map consume. Type-stable: the tree
# shape lives in the NamedTuple type, so the recursion specializes and each
# `_flatten_one` dispatch resolves at compile time.
_flatten_components(nt::NamedTuple) = _flatten_vals(values(nt))
_flatten_vals(::Tuple{}) = ()
_flatten_vals(t::Tuple) = (_flatten_one(first(t))..., _flatten_vals(Base.tail(t))...)
_flatten_one(tc::TiedComponent) = (tc,)
_flatten_one(nt::NamedTuple) = _flatten_components(nt)

phase_components(m::StationGainModel) = _flatten_components(m.phase)
logamp_components(m::StationGainModel) = _flatten_components(m.logamp)

# Merge two named component groups, failing loudly on a name clash — `merge`
# alone would silently drop the earlier component. Names are unique WITHIN a
# group (phase or log-amp), so this guards the assembly of a model from several
# steps' contributions.
function _merge_components(a::NamedTuple, b::NamedTuple)
    dup = intersect(keys(a), keys(b))
    isempty(dup) || throw(
        ArgumentError(
            "duplicate component name(s) $(collect(dup)): each component in a " *
                "group must be uniquely named.",
        ),
    )
    return merge(a, b)
end

# ── Model-list elements ──────────────────────────────────────────────────────

"""
    model_components(element, geom::DataGeometry) -> TiedComponent | NamedTuple | Nothing

Compile one model-list element for the data geometry `geom`. The result is named
by the element's key in the term list, so an element returns only its own
internal structure:

- a single `TiedComponent` — the element's list key names it (`θ.phase.<key>`);
- a `NamedTuple` of `TiedComponent`s — one element that compiles to several
  components, nested under its list key (`θ.phase.<key>.<part>`);
- `nothing` — the geometry cannot constrain the element, so it contributes no
  component (and no key).

A bare `TiedComponent` compiles to itself; wrapper elements (e.g.
[`DispersionModel`](@ref)) consult the geometry. The same generic applied to a
pipeline step returns the step's `(; phase, logamp)` named component trees —
steps and list elements compose through one mechanism.
"""
function model_components end

model_components(tc::TiedComponent, ::DataGeometry) = tc

# ── Per-component / per-model time-segmentation queries ──────────────────────
# Used by solvers to route global-time vs per-scan components.
is_per_scan(::AbstractTimeSegmentation) = false
is_per_scan(::PerScan) = true
is_per_scan(::PerIntegration) = true
component_is_per_scan(tc::TiedComponent) = is_per_scan(time_segmentation(tc))
component_is_per_scan(c::GainComponent) = is_per_scan(c.time)

phase_is_per_scan(m::StationGainModel) = any(component_is_per_scan, phase_components(m))
amplitude_is_per_scan(m::StationGainModel) = any(component_is_per_scan, logamp_components(m))

# ── Validation ───────────────────────────────────────────────────────────────
# A frequency-only / time-only term must not be paired with a segmentation that
# makes it degenerate-free; the linear-algebra layer tolerates redundancy, so
# validation here is light — mainly catching empty models.
function validate_station_gain_model(m::StationGainModel)
    (isempty(phase_components(m)) && isempty(logamp_components(m))) && throw(
        ArgumentError("StationGainModel has neither phase nor log-amplitude components")
    )
    return m
end

# ── Routing signatures ───────────────────────────────────────────────────────

# The bandpass component's routing signature: a time-invariant offset resolved
# in frequency by `ChannelBlocks` — one free value per block of channels, the
# station's instrumental frequency response. `ChannelBlocks(1)` is the classic
# free per-channel bandpass; a larger block ties channels together.
#
# The frequency segmentation IS the signature. The R–L offset is the same term
# and the same time segmentation over `GlobalFrequency`, and the searched
# per-scan constant differs only in time — so a term-level test cannot tell
# them apart, and only this one says "resolved in frequency".
_is_bandpass(tc) =
    tc.component.term isa ConstantTerm && tc.component.time isa GlobalTime &&
    tc.component.freq isa ChannelBlocks

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
frequency_segmentation_label(s::FrequencyBands) = "bands$(length(s.ranges))"

function station_model_summary(name, m::StationGainModel)
    p, a = phase_components(m), logamp_components(m)
    ph = isempty(p) ? "—" : join(component_label.(p), " + ")
    am = isempty(a) ? "—" : join(component_label.(a), " + ")
    return string(name, "  phase(", ph, ")  logamp(", am, ")")
end

function Base.show(io::IO, m::StationGainModel)
    p, a = phase_components(m), logamp_components(m)
    ph = isempty(p) ? "—" : join(component_label.(p), " + ")
    am = isempty(a) ? "—" : join(component_label.(a), " + ")
    return print(io, "StationGainModel(phase: ", ph, ", logamp: ", am, ")")
end
