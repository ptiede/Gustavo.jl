# ── Gain components and station models ──────────────────────────────────────
#
# Composition hierarchy:
#
#   StationGainModel
#     .phase  :: NamedTuple, name → GainComponent   →  Σ phase contributions
#     .logamp :: NamedTuple, name → GainComponent   →  Σ log-amplitude contributions
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
# One `StationGainModel` is shared by all antennas in a solve: each antenna gets
# its own parameter values, not its own structure.

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
the common part plus a `SingleFeed` for the partner-only deviation.
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
    SingleFeed(feed)

The component applies to one `feed` (1 or 2) only — the other feed gets no
contribution from it. Allocates a single parameter block, assigned to `feed`.
This is the primitive that expresses a partner-feed-only deviation, so the old
asymmetric reference/relative bandpass model decomposes as a `SharedFeeds`
common part plus a `SingleFeed(partner)` deviation.
"""
struct SingleFeed <: AbstractFeedTying
    feed::Int
    function SingleFeed(feed::Integer)
        feed in (1, 2) ||
            throw(ArgumentError("SingleFeed feed must be 1 or 2, got $feed"))
        return new(Int(feed))
    end
end

# ── GainComponent ────────────────────────────────────────────────────────────

"""
    GainComponent(term; Ti, Frequency, Feed = PerFeed())

One contribution to a station's gain: a gain `term` replicated over a `Ti`
(time) segmentation and a `Frequency` segmentation, with `Feed` saying how the
two feeds share it. The keywords are the dimension names the block's axes
carry. One parameter block is allocated per (time segment, frequency
segment, feed block).

```julia
GainComponent(Delay(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds())
```

This keyword form is the one public spelling; [`component_label`](@ref) and
error messages print it back verbatim.
"""
struct GainComponent{
        T <: AbstractGainTerm, TS <: AbstractTimeSegmentation,
        FS <: AbstractFrequencySegmentation, F <: AbstractFeedTying,
    }
    term::T
    Ti::TS
    Frequency::FS
    Feed::F
end
GainComponent(term; Ti, Frequency, Feed = PerFeed()) = GainComponent(term, Ti, Frequency, Feed)

# Number of distinct feed-blocks this tying allocates per (ant, tseg, fseg).
#   PerFeed           → 2 (feed1, feed2)
#   SharedFeeds       → 1 (shared)
#   ReferenceRelative → 2 (reference, relative)
nfeed_blocks(::PerFeed) = 2
nfeed_blocks(::SharedFeeds) = 1
nfeed_blocks(::ReferenceRelative) = 2
nfeed_blocks(::SingleFeed) = 1

# The feed-node (column of a component's leaf `:Feed`/`:node` axis) a feed reads
# its PRIMARY block from, or 0 when the tying carries no block for that feed:
# `PerFeed` keeps the two feeds as distinct nodes; `SharedFeeds` folds them to
# one; `SingleFeed(k)` keeps only feed k; `ReferenceRelative`'s primary block
# is the shared reference (node 1) for both feeds.
_feed_node(::PerFeed, feed::Integer) = feed
_feed_node(::SharedFeeds, feed::Integer) = 1
_feed_node(::ReferenceRelative, feed::Integer) = 1
_feed_node(t::SingleFeed, feed::Integer) = feed == t.feed ? 1 : 0

# The SECONDARY block a feed also reads, or 0 for none. Only `ReferenceRelative`
# has one: the partner feed (non-reference) adds its own relative block (node 2)
# on top of the shared reference block.
_feed_node2(::AbstractFeedTying, feed::Integer) = 0
_feed_node2(t::ReferenceRelative, feed::Integer) = feed == 3 - t.reference_feed ? 2 : 0

"""
    StationGainModel(; phase = (;), logamp = (;))

The gain model for a station: a `NamedTuple` of named phase [`GainComponent`](@ref)s
and a `NamedTuple` of named log-amplitude `GainComponent`s.
`gain = exp(Σ logamp) · cis(Σ phase)`. The keys are the component names, unique
within each group; a value may itself be a `NamedTuple` — a named subtree for
one element that compiled to several components.
"""
struct StationGainModel{P <: NamedTuple, A <: NamedTuple}
    phase::P
    logamp::A
end
StationGainModel(; phase = (;), logamp = (;)) =
    StationGainModel(_named_components(phase), _named_components(logamp))

_named_components(nt::NamedTuple) = map(_as_component, nt)
_named_components(::Tuple{}) = (;)
_named_components(t::Tuple) = throw(
    ArgumentError(
        "StationGainModel components must be named: pass a NamedTuple " *
            "(e.g. `phase = (delay = GainComponent(...),)`), not a bare tuple.",
    ),
)
_as_component(e::GainComponent) = e
_as_component(nt::NamedTuple) = map(_as_component, nt)   # a named subtree

# Depth-first flat tuple of the `GainComponent`s in a named component tree, names
# dropped — the order the layout and forward map consume. Type-stable: the tree
# shape lives in the NamedTuple type, so the recursion specializes and each
# `_flatten_one` dispatch resolves at compile time.
_flatten_components(nt::NamedTuple) = _flatten_vals(values(nt))
_flatten_vals(::Tuple{}) = ()
_flatten_vals(t::Tuple) = (_flatten_one(first(t))..., _flatten_vals(Base.tail(t))...)
_flatten_one(e::GainComponent) = (e,)
_flatten_one(nt::NamedTuple) = _flatten_components(nt)

phase_components(m::StationGainModel) = _flatten_components(m.phase)
logamp_components(m::StationGainModel) = _flatten_components(m.logamp)

# ── Model-list elements ──────────────────────────────────────────────────────

"""
    model_components(element, geom::DataGeometry) -> GainComponent | NamedTuple | Nothing

Compile one model-list element for the data geometry `geom`. The result is named
by the element's key in the term list, so an element returns only its own
internal structure:

- a single [`GainComponent`](@ref) — the element's list key names it (`θ.phase.<key>`);
- a `NamedTuple` of `GainComponent`s — one element that compiles to several
  components, nested under its list key (`θ.phase.<key>.<part>`);
- `nothing` — the geometry cannot constrain the element, so it contributes no
  component (and no key).

A bare `GainComponent` compiles to itself; wrapper elements (e.g.
[`DispersionModel`](@ref)) consult the geometry. The same generic applied to a
pipeline step returns the step's `(; phase, logamp)` named component trees —
steps and list elements compose through one mechanism.
"""
function model_components end

model_components(e::GainComponent, ::DataGeometry) = e

# ── Per-component / per-model time-segmentation queries ─────────────────────────
# Used by solvers to route global-time vs per-scan components.
is_per_scan(::AbstractTimeSegmentation) = false
is_per_scan(::PerScan) = true
is_per_scan(::PerIntegration) = true
component_is_per_scan(e::GainComponent) = is_per_scan(e.Ti)

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

# ── Equality ─────────────────────────────────────────────────────────────────
# Field-wise value equality. The default struct `==` compares `Vector`-holding
# fields (e.g. a `FreqGroups` segmentation) by identity, so two independently
# built but identical models would compare unequal without these.
Base.:(==)(a::GainComponent, b::GainComponent) =
    a.term == b.term && a.Ti == b.Ti && a.Frequency == b.Frequency && a.Feed == b.Feed
Base.hash(e::GainComponent, h::UInt) =
    hash(e.Feed, hash(e.Frequency, hash(e.Ti, hash(e.term, hash(:GainComponent, h)))))

Base.:(==)(a::StationGainModel, b::StationGainModel) =
    a.phase == b.phase && a.logamp == b.logamp
Base.hash(m::StationGainModel, h::UInt) =
    hash(m.logamp, hash(m.phase, hash(:StationGainModel, h)))

# ── Summaries ────────────────────────────────────────────────────────────────

# The constructor call that builds `x`, as a pasteable string: the unqualified
# type name applied to `repr`s of the field values. Types whose public
# constructor is not `TypeName(fields...)` override.
_call_string(x) = string(
    nameof(typeof(x)), "(",
    join((repr(getfield(x, i)) for i in 1:nfields(x)), ", "), ")",
)
_call_string(t::Polynomial{:Frequency}) = "PolynomialFreq($(t.degree))"
_call_string(t::Polynomial{:Ti}) = "PolynomialTime($(t.degree))"

"""
    component_label(e::GainComponent) -> String

The [`GainComponent`](@ref) constructor call as a string, e.g.
`GainComponent(Delay(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds())`.
Used in errors and summaries so the printed form is one the user can paste back.
"""
function component_label(e::GainComponent)
    return string(
        "GainComponent(", _call_string(e.term),
        "; Ti = ", _call_string(e.Ti),
        ", Frequency = ", _call_string(e.Frequency),
        ", Feed = ", _call_string(e.Feed), ")",
    )
end

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
