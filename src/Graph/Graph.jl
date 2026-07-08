# ── Gustavo.Graph — distributed map/reduce substrate over a processing set ────
#
# A Julia analog of casangi's GraphVIPER (astroviper's parallel engine): map a
# user node-function over the partitions of a `UVSet` (an xradio-style flat
# processing set), then optionally reduce the per-partition results. The same
# API runs under two executors:
#
#   * `SerialExecutor` — a plain sequential walk (bit-identical to `apply` /
#                        `mapleaves` for a per-leaf transform); the reference.
#   * `DaggerExecutor` — Dagger.jl owns the parallelism (a task DAG that runs on
#                        threads in one process and scales out to distributed
#                        workers unchanged). This is the general parallel path.
#
# Dagger is THE parallelization engine for Gustavo — there is no bespoke thread
# scheduler here. The crux GraphVIPER borrows and we reproduce is the **shared
# load layer**: sibling-band leaves of one scan share on-disk bytes, so each
# `(source, scan)` group is materialized ONCE (`materialize_group`, which the
# FITS-IDI reader turns into a single sequential span read) inside one Dagger
# task, and the group's leaves are mapped from that one in-memory result —
# instead of re-opening/decoding the file per leaf.

module Graph

using ..UVData
using ..UVData: UVSet, materialize_group, materialize_leaf
using DimensionalData
using DimensionalData: AbstractDimTree, TreeDict, metadata, branches, rebuild
using OrderedCollections: OrderedDict
import Dagger

export ParallelCoords, MapSpec, ReduceSpec
export AbstractExecutor, SerialExecutor, DaggerExecutor
export pmap, pmapreduce, load_groups

# The layers a node function's data is materialized with by default. `:flag` is
# derived from `weights <= 0` on rebuild, so it is dropped — it costs as much to
# read as vis+weights combined and carries nothing the weights don't.
const DEFAULT_LAYERS = (:vis, :weights, :uvw)

# ── Parallel coordinates ─────────────────────────────────────────────────────
#
# Declares how the processing set is chunked. The natural VLBI unit is the
# partition (one leaf = one source×scan×spw). GraphVIPER also chunks within a
# partition along (time, baseline, frequency, polarization); that finer
# sub-selection is a planned extension — v1 chunks at partition granularity.

"""
    ParallelCoords(; over = (:partition,))

Declares the axes the processing set is chunked over for a `pmap`/`pmapreduce`.
Currently only `:partition` (one chunk per `UVSet` leaf) is supported; finer
within-leaf axes (`:time`, `:baseline_id`, `:frequency`, `:polarization`, à la
GraphVIPER) are a planned extension and error for now.
"""
struct ParallelCoords{N}
    over::NTuple{N, Symbol}
end
ParallelCoords(; over = (:partition,)) = ParallelCoords(_as_over_tuple(over))

_as_over_tuple(o::ParallelCoords) = o.over
_as_over_tuple(o::Symbol) = (o,)
_as_over_tuple(o::Tuple) = o
_as_over_tuple(o) = Tuple(o)

function _check_supported(pc::ParallelCoords)
    pc.over == (:partition,) || throw(ArgumentError(
        "Graph: only `over = (:partition,)` is supported in this version " *
            "(got $(pc.over)); finer within-leaf chunking is a planned extension.",
    ))
    return pc
end

# ── Map / reduce specs ───────────────────────────────────────────────────────

"""
    MapSpec(node_fn; params = nothing)

A node computation: `node_fn` is applied to each partition's materialized data.
`node_fn` is arity-polymorphic (like `apply`'s callable):

- `node_fn(data)` — just the materialized leaf `DimTree`.
- `node_fn(data, coords)` — `coords` is a `NamedTuple` `(; key, info, root)`
  (`key`::Symbol partition key, `info`::PartitionInfo, `root`::UVMetadata).
- `node_fn(data, coords, params)` — plus the user payload `params`.

Return a `DimTree` to have `pmap` rebuild a `UVSet` (tree-out, like `apply`);
return anything else for a dict-out result (like `mapleaves`).
"""
struct MapSpec{F, P}
    node_fn::F
    params::P
end
MapSpec(node_fn; params = nothing) = MapSpec(node_fn, params)

"""
    ReduceSpec(op; init = nothing, tree = true)

How to combine per-partition results: `op(a, b)` is a binary combiner. `tree =
true` folds pairwise (log-depth binary tree, lower peak memory); `tree = false`
is a single-node left fold. `init`, if given, seeds the fold (and is the result
for an empty processing set).
"""
struct ReduceSpec{O, I}
    op::O
    init::I
    tree::Bool
end
ReduceSpec(op; init = nothing, tree = true) = ReduceSpec(op, init, tree)

# ── Executors ────────────────────────────────────────────────────────────────

abstract type AbstractExecutor end

"""
    SerialExecutor()

Sequential executor. `pmap`/`pmapreduce` under it are bit-identical to a plain
`apply`/`mapleaves`-style walk; use it as the reference and for debugging.
"""
struct SerialExecutor <: AbstractExecutor end

"""
    DaggerExecutor()

Dagger.jl executor — the general parallel path. Each load-group becomes a
`Dagger.@spawn` task (which materializes the group once, then maps its leaves),
and Dagger's scheduler runs them across threads in one process or across
distributed workers, unchanged. Results are reassembled in deterministic
partition order, so the output equals the `SerialExecutor`'s.
"""
struct DaggerExecutor <: AbstractExecutor end

# Run `work` over each element of `items`, returning a Vector of results in the
# SAME order as `items`. Executors override this. The default is serial.
_run(::SerialExecutor, work, items) = map(work, items)
# Dagger owns the parallelism: one task per load-group, then gather. The shared
# single-read-per-group happens inside `work` (via `materialize_group`).
function _run(::DaggerExecutor, work, items)
    tasks = [Dagger.@spawn work(item) for item in items]
    return map(fetch, tasks)
end

# ── Load-group discovery (the shared load layer) ─────────────────────────────

"""
    load_groups(uvset::UVSet) -> Vector{Vector{Pair{Symbol, <:AbstractDimTree}}}

Group the leaves of `uvset` into shared-load units: sibling-band leaves of one
`(source, scan)` share on-disk bytes and are materialized together in a single
read. Each group is a Vector of `key => leaf` pairs; groups are in first-seen
partition order, and leaf order within a group is preserved.
"""
function load_groups(uvset::UVSet)
    groups = OrderedDict{Tuple{String, String}, Vector{Pair{Symbol, Any}}}()
    for (k, leaf) in branches(uvset)
        info = metadata(leaf)
        gkey = (String(info.source_name), String(info.scan_name))
        push!(get!(() -> Pair{Symbol, Any}[], groups, gkey), k => leaf)
    end
    return collect(values(groups))
end

# Invoke a node function against `(data, coords, params)` using whichever arity
# it supports (mirrors UVData's `_call_leaf_fn`).
@inline function _call_node(f, data, coords, params)
    if applicable(f, data)
        return f(data)
    elseif applicable(f, data, coords)
        return f(data, coords)
    else
        return f(data, coords, params)
    end
end

# Map `spec` over every partition and return a Dict{key => result}. The shared
# load layer materializes each group once; the node function then runs per leaf.
function _map_to_resultmap(
        spec::MapSpec, uvset::UVSet, exec::AbstractExecutor, layers,
    )
    root = metadata(uvset)
    groups = load_groups(uvset)
    work = function (group)
        leaves = [leaf for (_, leaf) in group]
        mats = materialize_group(leaves; layers = layers)
        out = Vector{Pair{Symbol, Any}}(undef, length(group))
        @inbounds for i in eachindex(group)
            key = group[i].first
            data = mats[i]
            coords = (; key = key, info = metadata(data), root = root)
            out[i] = key => _call_node(spec.node_fn, data, coords, spec.params)
        end
        return out
    end
    grouped = _run(exec, work, groups)
    resultmap = Dict{Symbol, Any}()
    for gr in grouped, kv in gr
        resultmap[kv.first] = kv.second
    end
    return resultmap
end

# ── pmap ─────────────────────────────────────────────────────────────────────

"""
    pmap(node_fn, uvset::UVSet; over = (:partition,), params = nothing,
         executor = SerialExecutor(), layers = $(DEFAULT_LAYERS)) -> UVSet | OrderedDict

Map `node_fn` over the partitions of `uvset` (see [`MapSpec`](@ref) for the node
signature). Each `(source, scan)` group of sibling-band leaves is materialized
once (the shared load layer). If `node_fn` returns a `DimTree`, the result is a
`UVSet` with the same partition keys (tree-out, like `apply`); otherwise it is
an `OrderedDict{Symbol}` keyed by partition (like `mapleaves`). Partition order
is preserved regardless of executor.
"""
function pmap(
        node_fn, uvset::UVSet;
        over = (:partition,), params = nothing,
        executor::AbstractExecutor = SerialExecutor(),
        layers = DEFAULT_LAYERS,
    )
    _check_supported(over isa ParallelCoords ? over : ParallelCoords(; over = over))
    resultmap = _map_to_resultmap(MapSpec(node_fn, params), uvset, executor, layers)
    return _assemble(uvset, resultmap)
end

# Build the tree-out UVSet or dict-out OrderedDict from a key=>result map,
# iterating the ORIGINAL branch order so the output is deterministic.
function _assemble(uvset::UVSet, resultmap::Dict{Symbol, Any})
    src = branches(uvset)
    if isempty(src)
        return rebuild(uvset; branches = TreeDict())
    end
    firstval = resultmap[first(keys(src))]
    if firstval isa AbstractDimTree
        new_branches = TreeDict()
        sizehint!(new_branches, length(src))
        for k in keys(src)
            new_branches[k] = resultmap[k]
        end
        return rebuild(uvset; branches = new_branches)
    else
        out = OrderedDict{Symbol, Any}()
        for k in keys(src)
            out[k] = resultmap[k]
        end
        return out
    end
end

# ── pmapreduce ───────────────────────────────────────────────────────────────

"""
    pmapreduce(node_fn, op, uvset::UVSet; over = (:partition,), params = nothing,
               executor = SerialExecutor(), init = nothing, tree = true,
               layers = $(DEFAULT_LAYERS))

Map `node_fn` over the partitions (as [`pmap`](@ref)) and reduce the
per-partition results with `op`. Results are reduced in deterministic partition
order, so the value is executor-independent. `tree = true` reduces pairwise
(binary tree); `tree = false` is a single-node left fold. `init` seeds the fold
and is returned for an empty processing set.
"""
function pmapreduce(
        node_fn, op, uvset::UVSet;
        over = (:partition,), params = nothing,
        executor::AbstractExecutor = SerialExecutor(),
        init = nothing, tree = true,
        layers = DEFAULT_LAYERS,
    )
    _check_supported(over isa ParallelCoords ? over : ParallelCoords(; over = over))
    resultmap = _map_to_resultmap(MapSpec(node_fn, params), uvset, executor, layers)
    # Flatten in deterministic partition order.
    src = branches(uvset)
    results = Any[resultmap[k] for k in keys(src)]
    return _reduce_results(ReduceSpec(op, init, tree), results)
end

function _reduce_results(spec::ReduceSpec, results::AbstractVector)
    if isempty(results)
        spec.init === nothing && throw(ArgumentError(
            "pmapreduce: empty processing set and no `init` to return.",
        ))
        return spec.init
    end
    combined = spec.tree ? _tree_reduce(spec.op, results) : foldl(spec.op, results)
    return spec.init === nothing ? combined : spec.op(spec.init, combined)
end

# Pairwise (binary-tree) reduce: combine adjacent pairs each pass. Log-depth,
# lower peak memory than a linear fold when `op` accumulates large objects.
function _tree_reduce(op, xs::AbstractVector)
    length(xs) == 1 && return xs[1]
    buf = collect(xs)
    while length(buf) > 1
        m = length(buf)
        nxt = Vector{Any}(undef, cld(m, 2))
        @inbounds for i in 1:2:m
            nxt[cld(i, 2)] = i + 1 <= m ? op(buf[i], buf[i + 1]) : buf[i]
        end
        buf = nxt
    end
    return buf[1]
end

end # module Graph
