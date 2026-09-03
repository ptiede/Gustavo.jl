# ── Gain components and station models ──────────────────────────────────────
#
# Composition hierarchy:
#
#   StationGainModel
#     .phase  :: NamedTuple, name → GainComponent   →  Σ phase contributions
#     .logamp :: NamedTuple, name → GainComponent   →  Σ log-amplitude contributions
#       gain(t, f) = exp(Σ logamp) · cis(Σ phase)
#     .stations :: NamedTuple, station → replacement `(; phase, logamp)` entry
#
# Each component carries a user-specified name (the NamedTuple key), unique
# within its group. A value that is itself a NamedTuple is a named subtree — one
# list element that compiled to several components (e.g. a single-band delay's
# delay + companion constant), namespaced under the element's key. `θ` addresses
# the parameters the same way: `θ.phase.<name>` / `θ.logamp.<name>`. The layout
# and the forward map consume the depth-first flat view `phase_components` /
# `logamp_components`, whose order is the list order.
#
# The model structure is shared by all antennas in a solve — each antenna gets
# its own parameter values, not its own structure — except where a `stations`
# entry replaces a whole group for one station (see `station_components`).

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
    AbstractGainModel

A station gain model: the assignment of a `(; phase, logamp)` pair of named
[`GainComponent`](@ref) trees to every station of an observation.
`gain = exp(Σ logamp) · cis(Σ phase)` per station.

The one required method is the station seam,
[`station_components`](@ref)`(model, station) -> (; phase, logamp)` — the trees
the named station solves. [`StationGainModel`](@ref) is the shipped
implementation; a rule-based model (e.g. "every station whose code starts with
a given prefix gets a smoother bandpass") is a subtype implementing that one
method. Before a solve, a model is resolved against the observation's antenna
table by [`materialize`](@ref)`(model, antennas, geom)`, which records the
per-station trees the solve actually uses.
"""
abstract type AbstractGainModel end

"""
    StationGainModel(; phase = (;), logamp = (;), stations = (;))

The gain model for a station: a `NamedTuple` of named phase [`GainComponent`](@ref)s
and a `NamedTuple` of named log-amplitude `GainComponent`s.
`gain = exp(Σ logamp) · cis(Σ phase)`. The keys are the component names, unique
within each group; a value may itself be a `NamedTuple` — a named subtree for
one element that compiled to several components.

`stations` maps a station code to a replacement entry used verbatim for that
station: a `NamedTuple` whose `phase` and/or `logamp` tree REPLACES the
corresponding base group wholesale — the model never merges within a group. A
group the entry omits is inherited from the base. Replacement is whole-group
because components within a group interact (they sum and share degeneracies)
while the two groups do not. An entry key other than `phase`/`logamp` errors at
construction — the likely mistake is writing a component name at the top level.

```julia
StationGainModel(
    phase = (bandpass = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1)),),
    stations = (AA = (; phase = (bandpass = GainComponent(PolynomialFreq(3); Ti = GlobalTime(), Frequency = GlobalFrequency()),)),),
)
```

Station codes resolve against the observation's antenna table when the model is
[`materialize`](@ref)d for a solve; an unknown code errors there, naming the
known stations. Equality is order-insensitive in `stations`.
"""
struct StationGainModel{P <: NamedTuple, A <: NamedTuple, S <: NamedTuple} <: AbstractGainModel
    phase::P
    logamp::A
    stations::S
end
StationGainModel(; phase = (;), logamp = (;), stations = (;)) =
    StationGainModel(
    _named_components(phase), _named_components(logamp), _station_entries(stations)
)

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

_station_entries(nt::NamedTuple) = map(_station_entry, nt)
_station_entries(::Tuple{}) = (;)
function _station_entry(e::NamedTuple)
    unknown = setdiff(keys(e), (:phase, :logamp))
    isempty(unknown) || throw(
        ArgumentError(
            "station entry has unexpected key(s) $(Tuple(unknown)): an entry replaces " *
                "whole groups — `(; phase = ..., logamp = ...)`, either may be omitted — " *
                "and component names nest INSIDE the groups, e.g. " *
                "`(; phase = (; bandpass = GainComponent(...)))`.",
        ),
    )
    return map(_named_components, e)
end
_station_entry(x) = throw(
    ArgumentError(
        "a station entry must be a `(; phase, logamp)` NamedTuple (either key may be " *
            "omitted), got $(typeof(x)).",
    ),
)

"""
    station_components(model::AbstractGainModel, station) -> (; phase, logamp)

The named component trees `station` (a station code, `String` or `Symbol`)
solves under `model`. This is the extension seam of [`AbstractGainModel`](@ref):
a model subtype implements this one method and the layout machinery does the
rest. Solves consult it only through [`materialize`](@ref), so what a solve
records and what this returns coincide.

For a [`StationGainModel`](@ref): the base `phase`/`logamp` trees, except where
a `stations` entry replaces a whole group for this station.
"""
function station_components end

function station_components(m::StationGainModel, station)
    e = get(m.stations, Symbol(station), nothing)
    e === nothing && return (; phase = m.phase, logamp = m.logamp)
    return (;
        phase = haskey(e, :phase) ? e.phase : m.phase,
        logamp = haskey(e, :logamp) ? e.logamp : m.logamp,
    )
end

"""
    as_gain_model(m) -> AbstractGainModel

Lift a model specification to an [`AbstractGainModel`](@ref): an
`AbstractGainModel` passes through; a bare `(; phase, logamp)` `NamedTuple`
tree (either key may be omitted, no other key is legal) lifts to a uniform
[`StationGainModel`](@ref). Anything else errors. This is how the pipeline
accepts a step's `model_components` return and a step's own `model` argument in
either form.
"""
as_gain_model(m::AbstractGainModel) = m
function as_gain_model(m::NamedTuple)
    unknown = setdiff(keys(m), (:phase, :logamp))
    isempty(unknown) || throw(
        ArgumentError(
            "step model tree has unexpected key(s) $(Tuple(unknown)): a model is " *
                "`(; phase, logamp)` — component names nest INSIDE the groups, e.g. " *
                "`(; phase = (; bandpass = GainComponent(...)))`.",
        ),
    )
    return StationGainModel(phase = get(m, :phase, (;)), logamp = get(m, :logamp, (;)))
end
as_gain_model(m) = throw(
    ArgumentError(
        "a step model must be a `StationGainModel` or a `(; phase, logamp)` NamedTuple " *
            "tree of named `GainComponent`s, got $(typeof(m)).",
    ),
)

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
    model_components(element, spec) -> GainComponent | NamedTuple | Nothing

Compile one model-list element for an observation. `spec = (; geom, antennas)`
carries the `DataGeometry` and the antenna table — the same spec a pipeline
step's `model_components` receives, so steps and list elements compose through
one mechanism (the same generic applied to a step returns the step's whole
model). The result is named by the element's key in the term list, so an
element returns only its own internal structure:

- a single [`GainComponent`](@ref) — the element's list key names it (`θ.phase.<key>`);
- a `NamedTuple` of `GainComponent`s — one element that compiles to several
  components, nested under its list key (`θ.phase.<key>.<part>`);
- `nothing` — the geometry cannot constrain the element, so it contributes no
  component (and no key).

A bare `GainComponent` compiles to itself; wrapper elements (e.g.
[`DispersionModel`](@ref)) consult `spec.geom`.
"""
function model_components end

model_components(e::GainComponent, spec) = e

# ── Per-component / per-model time-segmentation queries ─────────────────────────
# Used by solvers to route global-time vs per-scan components.
is_per_scan(::AbstractTimeSegmentation) = false
is_per_scan(::PerScan) = true
is_per_scan(::PerIntegration) = true
component_is_per_scan(e::GainComponent) = is_per_scan(e.Ti)

# The per-scan queries answer for the whole model: the base groups plus every
# station entry's replacement groups.
phase_is_per_scan(m::StationGainModel) =
    any(component_is_per_scan, phase_components(m)) ||
    any(e -> haskey(e, :phase) && any(component_is_per_scan, _flatten_components(e.phase)), values(m.stations))
amplitude_is_per_scan(m::StationGainModel) =
    any(component_is_per_scan, logamp_components(m)) ||
    any(e -> haskey(e, :logamp) && any(component_is_per_scan, _flatten_components(e.logamp)), values(m.stations))

# ── Validation ───────────────────────────────────────────────────────────────
# A frequency-only / time-only term must not be paired with a segmentation that
# makes it degenerate-free; the linear-algebra layer tolerates redundancy, so
# validation here is light — mainly catching empty models.
function validate_station_gain_model(m::StationGainModel)
    isempty(phase_components(m)) && isempty(logamp_components(m)) &&
        all(e -> all(isempty ∘ _flatten_components, values(e)), values(m.stations)) &&
        throw(
        ArgumentError("StationGainModel has neither phase nor log-amplitude components")
    )
    return m
end

# ── Equality ─────────────────────────────────────────────────────────────────
# Field-wise value equality. The default struct `==` compares `Vector`-holding
# fields (e.g. a `FreqGroups` segmentation) by identity, so two independently
# built but identical models would compare unequal without these. `stations` is
# order-insensitive: it is a station → entry map, and no consumer reads its
# entry order.
Base.:(==)(a::GainComponent, b::GainComponent) =
    a.term == b.term && a.Ti == b.Ti && a.Frequency == b.Frequency && a.Feed == b.Feed
Base.hash(e::GainComponent, h::UInt) =
    hash(e.Feed, hash(e.Frequency, hash(e.Ti, hash(e.term, hash(:GainComponent, h)))))

Base.:(==)(a::StationGainModel, b::StationGainModel) =
    a.phase == b.phase && a.logamp == b.logamp && _stations_equal(a.stations, b.stations)

function _stations_equal(a::NamedTuple, b::NamedTuple)
    length(a) == length(b) || return false
    for k in keys(a)
        haskey(b, k) || return false
        a[k] == b[k] || return false
    end
    return true
end

function _stations_hash(s::NamedTuple, h::UInt)
    for k in sort(collect(keys(s)))
        h = hash(s[k], hash(k, h))
    end
    return h
end

Base.hash(m::StationGainModel, h::UInt) =
    _stations_hash(m.stations, hash(m.logamp, hash(m.phase, hash(:StationGainModel, h))))

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
    return string(name, "  phase(", ph, ")  logamp(", am, ")", _stations_suffix(m))
end

_stations_suffix(m::StationGainModel) =
    isempty(m.stations) ? "" :
    string("  stations(", join(string.(keys(m.stations)), ", "), ")")

function Base.show(io::IO, m::StationGainModel)
    p, a = phase_components(m), logamp_components(m)
    ph = isempty(p) ? "—" : join(component_label.(p), " + ")
    am = isempty(a) ? "—" : join(component_label.(a), " + ")
    return print(io, "StationGainModel(phase: ", ph, ", logamp: ", am, _stations_suffix(m), ")")
end

# ── Materialization against an observation ───────────────────────────────────
#
# `materialize(model, antennas, geom)` resolves a gain model into the concrete
# `StationGainModel` a solve uses and a solution records: station codes checked
# against the antenna table, every component's data-dependent frequency
# segmentation resolved (`materialize(seg, geom)`), and `stations` entries that
# turn out identical to the base dropped. The result is idempotent under
# re-materialization, and is what provenance stores — a rule-based model
# round-trips as the per-station trees it actually solved.

materialize(e::GainComponent, geom::DataGeometry) =
    GainComponent(e.term, e.Ti, materialize(e.Frequency, geom), e.Feed)
materialize(nt::NamedTuple, geom::DataGeometry) = map(v -> materialize(v, geom), nt)

# Station codes of an antenna table or an iterable of codes, as `String`s.
_station_names(antennas::UVData.AntennaTable) = String.(collect(antennas.name))
_station_names(names) = String.(collect(names))
_station_names(n::Integer) = throw(
    ArgumentError(
        "an antenna COUNT cannot resolve station codes; pass the antenna table " *
            "or the station codes themselves.",
    ),
)

function _validate_station_keys(stations::NamedTuple, names)
    for k in keys(stations)
        String(k) in names || throw(
            ArgumentError(
                "unknown station $(repr(k)) in `stations`; the antenna table has: " *
                    join(names, ", ") * ".",
            ),
        )
    end
    return nothing
end

"""
    materialize(model::AbstractGainModel, antennas, geom::DataGeometry) -> StationGainModel

Resolve `model` against an observation: `antennas` (an `AntennaTable` or an
iterable of station codes) fixes the station set, and `geom` resolves every
component's data-dependent frequency segmentation to its concrete form. The
result holds the per-station trees the solve actually uses — `stations` entries
only for stations whose trees differ from the base — and is what a solution
records as provenance. A `stations` key not in the antenna table errors, naming
the known stations.
"""
function materialize(m::StationGainModel, antennas, geom::DataGeometry)
    names = _station_names(antennas)
    _validate_station_keys(m.stations, names)
    phase = materialize(m.phase, geom)
    logamp = materialize(m.logamp, geom)
    ents = Pair{Symbol, Any}[]
    for k in keys(m.stations)
        e = map(nt -> materialize(nt, geom), m.stations[k])
        full = (;
            phase = haskey(e, :phase) ? e.phase : phase,
            logamp = haskey(e, :logamp) ? e.logamp : logamp,
        )
        (full.phase == phase && full.logamp == logamp) || push!(ents, k => full)
    end
    return StationGainModel(phase, logamp, NamedTuple(ents))
end

# A `station_components` return with either group omitted, completed to the
# full `(; phase, logamp)` pair — the same tolerance the model-tree lift
# gives — with any other key rejected.
function _full_tree(t::NamedTuple)
    unknown = setdiff(keys(t), (:phase, :logamp))
    isempty(unknown) || throw(
        ArgumentError(
            "station_components must return a `(; phase, logamp)` NamedTuple " *
                "(either key may be omitted); got unexpected key(s) $(Tuple(unknown)).",
        ),
    )
    return (; phase = get(t, :phase, (;)), logamp = get(t, :logamp, (;)))
end

# A model subtype is resolved through its station seam: one tree per station,
# with the most common tree as the base (ties to first appearance) so a
# mostly-uniform model records compactly.
function materialize(model::AbstractGainModel, antennas, geom::DataGeometry)
    names = _station_names(antennas)
    trees = [
        map(nt -> materialize(nt, geom), _full_tree(station_components(model, n)))
            for n in names
    ]
    counts = [count(==(t), trees) for t in trees]
    base = trees[findmax(counts)[2]]
    ents = Pair{Symbol, Any}[
        Symbol(names[i]) => trees[i] for i in eachindex(names) if trees[i] != base
    ]
    return StationGainModel(base.phase, base.logamp, NamedTuple(ents))
end
