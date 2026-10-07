# Gauge conventions for station-based solves.
#
# A station solve determines node values only up to one additive constant per
# connected component of the (station, feed) graph: every observation is a
# difference of two nodes, so the design matrix has a null vector per component.
# A gauge supplies the missing constraint row.
#
# The constraint is what makes the system full rank; it does not change any
# gauge-invariant quantity (baseline differences, closure phases, the applied
# calibration). It does fix what per-station values are reported, and — where a
# quantity is compared across scans, as the R–L phase is — which comparisons are
# meaningful.

"""
    AbstractGauge

How a station solve fixes the values its data leave undetermined: one
constraint per [`GaugeFreedom`](@ref).

A gauge implements [`gauge_constraint`](@ref)`(g, freedom) -> (row, value)`, one
constraint per freedom, or, for a rule that ties freedoms together,
[`gauge_constraints`](@ref)`(g, freedoms) -> (C, d)` over all freedoms of a
system. [`gauge_anchor`](@ref) and [`gauge_station_order`](@ref) have defaults.
A gauge that names stations also implements [`resolve_gauge`](@ref) and
[`remap_gauge`](@ref).
"""
abstract type AbstractGauge end

"""
    PinAntenna(refs)

Set one node to zero per connected component: the first entry of `refs` present
in that component (its lowest feed), else the component's
best-observed node.

`refs` is a station index, a station code, or a ranked collection of either. A
ranked list matters when the leading choice is absent or unconstrained on some
scans — the gauge then falls to the next entry instead of to an arbitrary node.

Values reported under this gauge are differences against the pinned node, so
they are comparable across scans only where the same node was pinned.
"""
struct PinAntenna{R} <: AbstractGauge
    refs::R
end
PinAntenna(refs...) = PinAntenna(collect(refs))

"""
    ZeroSumPhase()

Constrain the mean over every node of each connected component to zero rather
than pinning one of them.

No single node carries the gauge, so the solve does not change character when one
station drops out — the failure mode `PinAntenna` has when its reference is
absent. The mean is over whatever stations a component holds, so it shifts when
that set changes: values under `ZeroSumPhase` are no more comparable ACROSS
scans than under a pin.
"""
struct ZeroSumPhase <: AbstractGauge end

"""
    GaugeFreedom(; nodes, station, feed, scan, component, observable, direction, weight)

One gauge freedom of a solve: moving the values of `nodes` together along
`direction` changes no row the data must hold, so the solve needs one constraint
to fix it. `nodes` index the system's unknowns; every other field runs parallel
to `nodes` and describes each one:

- `station`: station index;
- `feed`: feed index from 1, or 0 for an unknown every feed shares;
- `scan`: scan index, or 0 for an unknown spanning scans or a solve without scans;
- `component`: the model component's path, e.g. `(:phase, :mbd)`, or `()` where
  the solve names none (a nuisance offset, a seed solve);
- `observable`: `:delay`, `:rate` or `:phase`;
- `direction`: the shift's coefficient, in the solve's element type;
- `weight`: the total weight of the rows touching the node.
"""
struct GaugeFreedom{N, S, F, SC, C, O, D, W}
    nodes::N
    station::S
    feed::F
    scan::SC
    component::C
    observable::O
    direction::D
    weight::W
end
GaugeFreedom(; nodes, station, feed, scan, component, observable, direction, weight) =
    GaugeFreedom(nodes, station, feed, scan, component, observable, direction, weight)

"""
    GaugeFreedoms{T}(freedoms, nnodes)

The gauge freedoms of one system of `nnodes` unknowns solved in element type `T`,
as handed to [`gauge_constraints`](@ref). Indexes like `freedoms`.
"""
struct GaugeFreedoms{T, F, V <: AbstractVector{F}} <: AbstractVector{F}
    freedoms::V
    nnodes::Int
end
GaugeFreedoms{T}(freedoms::AbstractVector{F}, nnodes::Integer) where {T, F} =
    GaugeFreedoms{T, F, typeof(freedoms)}(freedoms, nnodes)
Base.size(fs::GaugeFreedoms) = size(fs.freedoms)
Base.getindex(fs::GaugeFreedoms, i::Int) = fs.freedoms[i]

"""
    gauge_constraints(g::AbstractGauge, freedoms::GaugeFreedoms{T}) -> (C, d)

The constraints `C * x == d` that fix every freedom of one system: `C` is
`length(freedoms) × freedoms.nnodes` and `d` has one entry per row. The rows
together must determine each freedom; a gauge that ties freedoms (the same
value in consecutive scans, say) overrides this method.

The default stacks [`gauge_constraint`](@ref) over the freedoms, in type `T`.
"""
function gauge_constraints(g::AbstractGauge, fs::GaugeFreedoms{T}) where {T}
    C = zeros(T, length(fs), fs.nnodes)
    d = zeros(T, length(fs))
    for j in eachindex(fs)
        f = fs[j]
        row, value = gauge_constraint(g, f)
        for i in eachindex(f.nodes, row)
            C[j, f.nodes[i]] = row[i]
        end
        d[j] = value
    end
    return C, d
end

"""
    gauge_constraint(g::AbstractGauge, f::GaugeFreedom) -> (row, value)

One constraint `sum(row .* x[f.nodes]) == value` fixing the freedom `f`. `row`
runs parallel to `f.nodes`, and must not be orthogonal to `f.direction`.
"""
function gauge_constraint end

"""
    gauge_anchor(g::AbstractGauge, f::GaugeFreedom) -> Int

A node of `f`, used to seed phase unwrapping and, where a solver pins a single
node per freedom, as that pin.

This is a numerical starting point, not the gauge itself: unwrapping propagates
relative phases outward from a real node, which a zero-sum constraint does not
supply. The default takes the first station of [`gauge_station_order`](@ref)
present in `f` (its node of the lowest feed index), else the node with the most
row weight.
"""
function gauge_anchor(g::AbstractGauge, f::GaugeFreedom)
    # The lowest feed, so a station's reported values stay referenced to the
    # same feed wherever it is present.
    for a in gauge_station_order(g)
        best = 0
        for i in eachindex(f.nodes, f.station, f.feed)
            f.station[i] == a && (best == 0 || f.feed[i] < f.feed[best]) && (best = i)
        end
        best == 0 || return f.nodes[best]
    end
    return _best_gauge_node(f)
end

# The fallback for a freedom holding no listed reference: its best-observed
# node, i.e. the one carrying the most total row weight, ties broken by lowest
# node index. Such a freedom's gauge is arbitrary by construction — there is no
# reference to express it against — so the only properties that matter are
# determinism and stability, and anchoring on the best-observed node is what buys
# the second: a structurally-chosen node (the lowest index, say) hops as soon as a
# marginal station's coverage flickers between solves, moving the whole
# component's zero with it.
function _best_gauge_node(f::GaugeFreedom)
    best = firstindex(f.nodes)
    for i in eachindex(f.nodes, f.weight)
        (f.weight[i] > f.weight[best] || (f.weight[i] == f.weight[best] && f.nodes[i] < f.nodes[best])) &&
            (best = i)
    end
    return f.nodes[best]
end

# `gauge_constraints` with its shape checked: one row per freedom over every node.
function _gauge_system(g::AbstractGauge, fs::GaugeFreedoms)
    C, d = gauge_constraints(g, fs)
    (size(C) == (length(fs), fs.nnodes) && length(d) == length(fs)) || throw(
        DimensionMismatch(
            "gauge_constraints for $(typeof(g)) must return one row per freedom over every node, " *
                "C of size $((length(fs), fs.nnodes)) and d of length $(length(fs)); " *
                "got $(size(C)) and $(length(d))",
        ),
    )
    return C, d
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

function gauge_constraint(g::PinAntenna, f::GaugeFreedom)
    T = eltype(f.direction)
    a = gauge_anchor(g, f)
    return T[n == a ? one(T) : zero(T) for n in f.nodes], zero(T)
end

function gauge_constraint(g::ZeroSumPhase, f::GaugeFreedom)
    T = eltype(f.direction)
    return fill(one(T) / length(f.nodes), length(f.nodes)), zero(T)
end

"""
    resolve_gauge(g::AbstractGauge, ant_names) -> AbstractGauge

Return `g` with any station codes replaced by 1-based station indices, matched
against `ant_names`. Errors on a code the antenna table does not carry.
"""
resolve_gauge(g::AbstractGauge, ant_names) = g

function resolve_gauge(g::PinAntenna, ant_names)
    refs = g.refs isa Union{Integer, AbstractString, Symbol} ? (g.refs,) : g.refs
    out = Int[_antenna_index(r, ant_names, "PinAntenna") for r in refs]
    isempty(out) && error("PinAntenna: no references given.")
    return PinAntenna(out)
end

_antenna_index(r::Integer, ant_names, who) = Int(r)
function _antenna_index(r, ant_names, who)
    i = findfirst(==(String(r)), ant_names)
    isnothing(i) && error("$who: station code \"$r\" not in antenna table $(ant_names).")
    return i
end

"""
    remap_gauge(g::AbstractGauge, map) -> AbstractGauge

Rewrite the gauge's station indices through `map`, which sends a station index to
its representative. Used where feeds or stations are tied into groups before the
solve.
"""
remap_gauge(g::PinAntenna, map) = PinAntenna([map[a] for a in _gauge_refs(g.refs)])
remap_gauge(g::ZeroSumPhase, map) = g

"""
    gauge_station_order(g::AbstractGauge) -> collection of Int

The station indices this gauge prefers as a numerical anchor, best first.

Empty by default: a gauge that names no station (a summed constraint) leaves the
caller to pick on its own criterion, typically the best-observed station.
"""
gauge_station_order(g::PinAntenna) = _gauge_refs(g.refs)
gauge_station_order(::AbstractGauge) = ()

"""
    ByComponent(choices::NamedTuple; default::AbstractGauge)

Use a different gauge for some model components: `choices` maps component
names to gauges, and every other freedom uses `default`. The names are the
step model's own, as a solution lists its components (`fringe.phase.rate` is
`rate`); a nested model takes nested choices, and a gauge given for a subtree
covers every component under it.

    BaselineFringeFit(; gauge = ByComponent((; rate = PinAntenna("A2")); default = PinAntenna("A1")))

A step rejects, when constructed, a name its model does not have. A freedom
spanning components with different choices is an error.
"""
struct ByComponent{C <: NamedTuple, D <: AbstractGauge} <: AbstractGauge
    choices::C
    default::D
end
ByComponent(choices::NamedTuple; default::AbstractGauge) = ByComponent(choices, default)

# The gauge for a component path `(:phase, names...)`.
function _component_choice(g::ByComponent, path::Tuple)
    node = g.choices
    for k in Base.tail(path)
        haskey(node, k) || return g.default
        node = node[k]
        node isa AbstractGauge && return node
    end
    return g.default
end
_component_choice(g::ByComponent, ::Tuple{}) = g.default

function _component_choice(g::ByComponent, f::GaugeFreedom)
    c = _component_choice(g, first(f.component))
    for p in f.component
        _component_choice(g, p) === c || throw(
            ArgumentError(
                "ByComponent: one gauge freedom spans components with different gauges " *
                    "($(join(unique(f.component), ", "))); give them the same choice.",
            ),
        )
    end
    return c
end

function gauge_constraints(g::ByComponent, fs::GaugeFreedoms{T}) where {T}
    C = zeros(T, length(fs), fs.nnodes)
    d = zeros(T, length(fs))
    choice = [_component_choice(g, f) for f in fs]
    for c in unique(choice)
        idx = findall(x -> x === c, choice)
        C[idx, :], d[idx] = _gauge_system(c, GaugeFreedoms{T}(fs.freedoms[idx], fs.nnodes))
    end
    return C, d
end

gauge_anchor(g::ByComponent, f::GaugeFreedom) = gauge_anchor(_component_choice(g, f), f)

# The gauge `g` applies to the component at `path`; a solver whose freedoms all
# belong to one component reads that gauge's station order or type through it.
_gauge_for(g::AbstractGauge, path::Tuple) = g
_gauge_for(g::ByComponent, path::Tuple) = _component_choice(g, path)

_map_choices(fn, nt::NamedTuple) = map(v -> v isa NamedTuple ? _map_choices(fn, v) : fn(v), nt)
resolve_gauge(g::ByComponent, ant_names) =
    ByComponent(_map_choices(c -> resolve_gauge(c, ant_names), g.choices), resolve_gauge(g.default, ant_names))
remap_gauge(g::ByComponent, map) =
    ByComponent(_map_choices(c -> remap_gauge(c, map), g.choices), remap_gauge(g.default, map))

"""
    check_gauge_components(g::AbstractGauge, names)

Throw if `g` names a component absent from `names`, the component paths
without their `:phase`/`:logamp` root (e.g. `(:rate,)`) of the model of the
step `g` belongs to. The default checks nothing.
"""
check_gauge_components(::AbstractGauge, names) = nothing

function check_gauge_components(g::ByComponent, names)
    for key in _leaf_paths(g.choices)
        any(n -> length(n) >= length(key) && n[1:length(key)] == key, names) || throw(
            ArgumentError(
                "ByComponent: no component $(join(key, '.')) in the step's model; " *
                    "the components are $(join((join(n, '.') for n in names), ", ")).",
            ),
        )
    end
    check_gauge_components(g.default, names)
    return nothing
end
