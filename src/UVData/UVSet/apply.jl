# Tree-walking primitives over `UVSet` leaves. All three accept a callable
# whose arity matches one of:
#   f(leaf)
#   f(leaf, info)                   # info = metadata(leaf)::PartitionInfo
#   f(leaf, info, root_meta)        # root_meta = metadata(uvset)::UVMetadata
# `apply` is tree-out (returns UVSet); `mapleaves` collects per-leaf results
# into an OrderedDict; `flatmap` vcat's per-leaf results into a flat Vector.

# Internal: invoke `f` against a leaf using whichever supported arity it
# accepts. Falls back to the strict 3-arg signature for backward compat.
@inline function _call_leaf_fn(f, leaf, info, root)
    if applicable(f, leaf)
        return f(leaf)
    elseif applicable(f, leaf, info)
        return f(leaf, info)
    else
        return f(leaf, info, root)
    end
end

"""
    apply(f, uvset::UVSet) -> UVSet

Walk the leaves of `uvset` and rebuild a tree with each leaf replaced by
`f(...)`. The callable can take 1, 2, or 3 arguments:

- `f(leaf)` — just the per-leaf `DimTree`.
- `f(leaf, info)` — `info::PartitionInfo` is `metadata(leaf)`.
- `f(leaf, info, root_meta)` — `root_meta::UVMetadata` is `metadata(uvset)`.

Whichever signature `f` defines, it must return a `DimTree` (e.g. via
`with_visibilities`) so the resulting `UVSet` is well-formed. Tree shape
is preserved. Built-in reducers (`TimeAverage`) and the bandpass /
apriori applicators implement this signature.

For per-leaf computations that don't return a `DimTree`, use
[`mapleaves`](@ref) (collect into `OrderedDict`) or [`flatmap`](@ref)
(`vcat` into a flat `Vector`).
"""
function apply(f, uvset::UVSet)
    root_meta = DimensionalData.metadata(uvset)
    src = DimensionalData.branches(uvset)
    new_branches = DimensionalData.TreeDict()
    sizehint!(new_branches, length(src))
    for (k, leaf) in src
        new_branches[k] = _call_leaf_fn(f, leaf, DimensionalData.metadata(leaf), root_meta)
    end
    return DimensionalData.rebuild(uvset; branches = new_branches)
end

"""
    mapleaves(f, uvset::UVSet) -> OrderedDict{Symbol, T}

Walk the leaves and collect `f(leaf)` (or `f(leaf, info)` / `f(leaf, info,
root_meta)` per the same arity rules as [`apply`](@ref)) into a fresh
`OrderedDict` keyed by partition key. Iteration order matches
`branches(uvset)`. The leaf is left untouched; this is the right verb
for diagnostics that compute per-leaf summaries.

The element type is fixed to `typeof(f(first_leaf))`, so the result is
type-stable as long as `f` has a stable return type. Subsequent leaves
must produce values convertible to that type (mixed-return-type
callables should map to a common supertype themselves).

```julia
stats = mapleaves(uvset) do leaf
    info = metadata(leaf)
    (; scan = info.scan_name, n = length(obs_time(leaf)))
end
```
"""
function mapleaves(f, uvset::UVSet)
    root_meta = DimensionalData.metadata(uvset)
    src = DimensionalData.branches(uvset)
    iter = pairs(src)
    state = iterate(iter)
    state === nothing && return OrderedDict{Symbol, Any}()
    pair1, st = state
    k1, leaf1 = pair1
    v1 = _call_leaf_fn(f, leaf1, DimensionalData.metadata(leaf1), root_meta)
    return _mapleaves_collect(f, iter, st, root_meta, k1, v1, length(src))
end

# Hot loop with the value type `T` fixed as a parameter, so the
# `OrderedDict{Symbol, T}` allocation, `setindex!`, and the closure
# return are all concrete.
function _mapleaves_collect(f::F, iter, state, root_meta, k1::Symbol, v1::T, n::Integer) where {F, T}
    out = OrderedDict{Symbol, T}()
    sizehint!(out, n)
    out[k1] = v1
    next = iterate(iter, state)
    while next !== nothing
        pair, state = next
        k, leaf = pair
        v = _call_leaf_fn(f, leaf, DimensionalData.metadata(leaf), root_meta)::T
        out[k] = v
        next = iterate(iter, state)
    end
    return out
end

"""
    flatmap(f, uvset::UVSet) -> Vector{T}

Walk the leaves and concatenate `f(...)` results into a flat `Vector`.
`f` must return an `AbstractVector` (or any iterable supported by
`append!`). Same arity-overload rules as [`apply`](@ref).

The result element type `T` is fixed to `eltype(f(first_leaf))`, so the
output is type-stable when `f`'s per-leaf vector type is consistent.

```julia
rows = flatmap(uvset) do leaf
    info = metadata(leaf)
    bls = baselines(leaf)
    [(; scan = info.scan_name, baseline = lbl) for lbl in bls.labels]
end
```
"""
function flatmap(f, uvset::UVSet)
    root_meta = DimensionalData.metadata(uvset)
    src = DimensionalData.branches(uvset)
    iter = values(src)
    state = iterate(iter)
    state === nothing && return Any[]
    leaf1, st = state
    first_piece = _call_leaf_fn(f, leaf1, DimensionalData.metadata(leaf1), root_meta)
    first_piece isa AbstractVector || throw(ArgumentError(
        "flatmap: callable must return an `AbstractVector` per leaf (got " *
            "$(typeof(first_piece))). Use `mapleaves` if you want per-leaf " *
            "results collected into an OrderedDict, or `apply` if you want " *
            "to rebuild a `UVSet` with new leaf data.",
    ))
    return _flatmap_collect(f, iter, st, root_meta, first_piece, length(src))
end

# Hot loop with the element type `T` fixed as a parameter on the first
# piece. Concrete `Vector{T}` + `append!` keeps the body type-stable.
function _flatmap_collect(
        f::F, iter, state, root_meta,
        first_piece::AbstractVector{T}, n::Integer,
    ) where {F, T}
    out = Vector{T}()
    sizehint!(out, n * length(first_piece))
    append!(out, first_piece)
    next = iterate(iter, state)
    while next !== nothing
        leaf, state = next
        piece = _call_leaf_fn(f, leaf, DimensionalData.metadata(leaf), root_meta)
        append!(out, piece)
        next = iterate(iter, state)
    end
    return out
end
