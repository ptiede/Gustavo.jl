# Gauge conventions for station-based solves.
#
# A station solve determines node values only up to one additive constant per
# connected component of the (station, feed) graph: every observation is a
# DIFFERENCE of two nodes, so the design matrix has a null vector per component.
# A gauge supplies the missing constraint row.
#
# The constraint is what makes the system full rank; it does not change any
# gauge-invariant quantity (baseline differences, closure phases, the applied
# calibration). It DOES fix what per-station values are reported, and — where a
# quantity is compared across scans, as the R–L phase is — which comparisons are
# meaningful.

"""
    AbstractGauge

How a station solve fixes the arbitrary additive constant on each connected
component of the (station, feed) graph.

Implementations provide [`gauge_anchor`](@ref) and [`gauge_row!`](@ref).
"""
abstract type AbstractGauge end

"""
    PinAntenna(refs)

Set one node to zero per connected component: the first entry of `refs` present
in that component (feed 1 preferred over feed 2), else the component's
best-observed node.

`refs` is a station index, a station code, or a ranked collection of either. A
ranked list matters when the leading choice is absent or unconstrained on some
scans — the gauge then falls to the next entry instead of to an arbitrary node.

Values reported under this gauge are differences against the pinned node, so
they are comparable across scans only where the SAME node was pinned.
"""
struct PinAntenna{R} <: AbstractGauge
    refs::R
end
PinAntenna(refs...) = PinAntenna(collect(refs))

"""
    ZeroSumPhase(; antennas = nothing, weights = nothing)

Constrain the weighted sum of each component's nodes to zero rather than pinning
one of them.

No single node carries the gauge, so the solve does not change character when one
station drops out — the failure mode `PinAntenna` has when its reference is
absent. `antennas` restricts the sum to a fixed set of station indices (those
present in the component); `weights` gives per-node weights. Both `nothing` sums
uniformly over every node in the component.

A sum taken over whatever stations happen to be present shifts when that set
changes, so a bare `ZeroSumPhase()` is no more comparable ACROSS scans than a
pin is. Pass `antennas` to hold the summed set fixed when cross-scan comparison
matters.
"""
struct ZeroSumPhase{A, W} <: AbstractGauge
    antennas::A
    weights::W
end
ZeroSumPhase(; antennas = nothing, weights = nothing) = ZeroSumPhase(antennas, weights)

"""
    gauge_anchor(g::AbstractGauge, comp_nodes, nodew, station_of, feed_of) -> Int

A representative node of `comp_nodes`, used to seed phase unwrapping.

This is a numerical starting point, not the gauge itself: unwrapping propagates
relative phases outward from a real node, which a zero-sum constraint does not
supply. Every gauge must name one.

`station_of(n)` and `feed_of(n)` map a node index to its station index and feed
(feed 2 is the second feed; anything else counts as feed 1 / shared). Passing
them keeps a gauge independent of how the caller lays out its node vector — the
per-scan station system and the tagged system that spans scans differ.
"""
function gauge_anchor end

"""
    gauge_row!(row, g::AbstractGauge, comp_nodes, nodew, station_of, feed_of) -> nothing

Write this component's constraint into `row` (a view over one row of the
constraint matrix, indexed by node). `row` arrives zeroed, and its element type
is the one the solve runs in — write through `eltype(row)` rather than assuming
a concrete float.
"""
function gauge_row! end

# The fallback for a component holding no listed reference: its best-OBSERVED
# node, i.e. the one carrying the most total row weight, ties broken by lowest
# node index. Such a component's gauge is arbitrary by construction — there is no
# reference to express it against — so the only properties that matter are
# determinism and stability, and anchoring on the best-observed node is what buys
# the second: a structurally-chosen node (the lowest index, say) hops as soon as a
# marginal station's coverage flickers between solves, moving the whole
# component's zero with it.
function _best_gauge_node(comp_nodes, nodew)
    best = first(comp_nodes)
    for n in comp_nodes
        (nodew[n] > nodew[best] || (nodew[n] == nodew[best] && n < best)) && (best = n)
    end
    return best
end

# Iterate the references without materializing a container: a `Tuple` built from a
# runtime-length vector has no statically known length, which costs an allocation
# and a dynamic dispatch on a function the station solve calls per component.
_gauge_refs(r::Integer) = (Int(r),)
_gauge_refs(r::AbstractVector{<:Integer}) = r
_gauge_refs(r) = error(
    "gauge references must be resolved to station indices before use; got $(typeof(r)). " *
        "Call `resolve_gauge(gauge, ant_names)` first.",
)

function gauge_anchor(g::PinAntenna, comp_nodes, nodew, station_of, feed_of)
    # Feed 1 (or a feed-shared node) before feed 2, so a station's reported
    # values stay referenced to the same feed wherever both are present.
    for a in _gauge_refs(g.refs)
        for n in comp_nodes
            station_of(n) == a && feed_of(n) != 2 && return n
        end
        for n in comp_nodes
            station_of(n) == a && return n
        end
    end
    return _best_gauge_node(comp_nodes, nodew)
end

function gauge_row!(row, g::PinAntenna, comp_nodes, nodew, station_of, feed_of)
    row[gauge_anchor(g, comp_nodes, nodew, station_of, feed_of)] = one(eltype(row))
    return nothing
end

gauge_anchor(::ZeroSumPhase, comp_nodes, nodew, station_of, feed_of) =
    _best_gauge_node(comp_nodes, nodew)

function gauge_row!(row, g::ZeroSumPhase, comp_nodes, nodew, station_of, feed_of)
    sel = if g.antennas === nothing
        comp_nodes
    else
        want = Set(Int(a) for a in g.antennas)
        [n for n in comp_nodes if station_of(n) in want]
    end
    # A restricted set that misses this component entirely would leave the row
    # empty and the system rank-deficient; sum over the whole component instead.
    isempty(sel) && (sel = comp_nodes)
    T = eltype(row)
    if g.weights === nothing
        s = one(T) / length(sel)
        for n in sel
            row[n] = s
        end
    else
        tot = sum(g.weights[n] for n in sel)
        tot > zero(tot) || error(
            "ZeroSumPhase: weights sum to $tot over this component's nodes $(collect(sel)); " *
                "the gauge row would be empty and the solve rank-deficient.",
        )
        for n in sel
            row[n] = g.weights[n] / tot
        end
    end
    return nothing
end

"""
    regauge!(x, g::AbstractGauge) -> x

Shift a per-station vector to `g`'s convention in place, skipping non-finite
entries. Used where a solve leaves its own arbitrary common mode — the adhoc
phase filter, whose prior pins the common mode near 0 rather than at a station —
and the result must land on the same convention as the rest of the pipeline.

Leaves `x` untouched when the gauge has nothing finite to reference, so an AP
with no usable station keeps whatever the solve produced instead of acquiring a
fabricated zero.
"""
function regauge!(x, g::PinAntenna)
    for a in _gauge_refs(g.refs)
        checkbounds(Bool, x, a) || continue
        isfinite(x[a]) || continue
        ref = x[a]
        for i in eachindex(x)
            isfinite(x[i]) && (x[i] -= ref)
        end
        return x
    end
    return x
end

function regauge!(x, g::ZeroSumPhase)
    idx = g.antennas === nothing ? eachindex(x) :
        [i for i in g.antennas if checkbounds(Bool, x, i)]
    tot = zero(eltype(x))
    n = 0
    for i in idx
        isfinite(x[i]) || continue
        tot += x[i]
        n += 1
    end
    n == 0 && return x
    ref = tot / n
    for i in eachindex(x)
        isfinite(x[i]) && (x[i] -= ref)
    end
    return x
end

"""
    resolve_gauge(g::AbstractGauge, ant_names) -> AbstractGauge

Return `g` with any station codes replaced by 1-based station indices, matched
against `ant_names`. Errors on a code the antenna table does not carry.
"""
resolve_gauge(g::AbstractGauge, ant_names) = g

function resolve_gauge(g::PinAntenna, ant_names)
    refs = g.refs isa Union{Integer, AbstractString, Symbol} ? (g.refs,) : g.refs
    out = Int[]
    for r in refs
        if r isa Integer
            push!(out, Int(r))
        else
            i = findfirst(==(String(r)), ant_names)
            i === nothing && error(
                "PinAntenna: station code \"$r\" not in antenna table $(ant_names).",
            )
            push!(out, i)
        end
    end
    isempty(out) && error("PinAntenna: no references given.")
    return PinAntenna(out)
end

function resolve_gauge(g::ZeroSumPhase, ant_names)
    g.antennas === nothing && return g
    out = Int[]
    for a in g.antennas
        if a isa Integer
            push!(out, Int(a))
        else
            i = findfirst(==(String(a)), ant_names)
            i === nothing && error(
                "ZeroSumPhase: station code \"$a\" not in antenna table $(ant_names).",
            )
            push!(out, i)
        end
    end
    return ZeroSumPhase(out, g.weights)
end

"""
    gauge_primary(g::AbstractGauge) -> Union{Int, Nothing}

The station index a caller may substitute when stations are merged into
representatives, or `nothing` when the gauge names no single station.
"""
gauge_primary(g::PinAntenna) = first(_gauge_refs(g.refs))
gauge_primary(::AbstractGauge) = nothing

"""
    remap_gauge(g::AbstractGauge, map) -> AbstractGauge

Rewrite the gauge's station indices through `map`, which sends a station index to
its representative. Used where feeds or stations are tied into groups before the
solve.
"""
remap_gauge(g::PinAntenna, map) = PinAntenna([map[a] for a in _gauge_refs(g.refs)])
remap_gauge(g::ZeroSumPhase, map) =
    ZeroSumPhase(g.antennas === nothing ? nothing : unique(map[a] for a in g.antennas), g.weights)

"""
    gauge_station_order(g::AbstractGauge, nant) -> AbstractVector{Int}

The station indices this gauge prefers as a numerical anchor, best first.

Empty when the gauge expresses no preference (a summed constraint names no
station), leaving the caller to pick on its own criterion — typically the
best-observed station.
"""
gauge_station_order(g::PinAntenna, nant) = collect(_gauge_refs(g.refs))
gauge_station_order(::AbstractGauge, nant) = Int[]
