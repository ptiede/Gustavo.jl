# ── Streaming dataset API (format-neutral) ───────────────────────────────────
#
# Large visibility files (FITS-IDI from DiFX can be tens of GB) do not fit in
# memory. The reader builds an ordinary `UVSet` whose per-(scan, spw) leaves
# carry *lazy*, disk-backed `vis`/`weights` layers; a solver materializes
# one leaf at a time (a single scan×band chunk does fit). `apply` and the
# selectors work unchanged on a lazy `UVSet`.
#
# `src/` stays format-neutral: laziness is detected through the overridable
# `_layer_is_lazy` trait (the FITS extension adds a method for its chunk array),
# and materialization is just `Array(x)`, which triggers the disk read on any
# `DiskArrays.AbstractDiskArray` without `src/` depending on DiskArrays.

# Laziness trait. Plain in-memory arrays are eager; an I/O extension overrides
# this for its disk-backed array type.
_layer_is_lazy(::AbstractArray) = false

"""
    is_lazy(leaf) -> Bool

True if the leaf's `vis` layer is disk-backed (not yet materialized).
"""
is_lazy(leaf::DimensionalData.AbstractDimTree) = _layer_is_lazy(parent(leaf[:vis]))

"""
    is_lazy(uvset::UVSet) -> Bool

True if any leaf of `uvset` is lazy.
"""
is_lazy(uvset::UVSet) = any(is_lazy, values(DimensionalData.branches(uvset)))

# Materialize a single layer's backing array (no-op if already a dense Array).
_materialize_layer(x::Array) = x
_materialize_layer(x::AbstractArray) = Array(x)

"""
    materialize_leaf(leaf) -> DimTree

Return a leaf with its data layers read into dense in-memory arrays. A no-op
(returns `leaf`) if the leaf is already eager. This is the natural unit a solver
pulls into memory: one scan × one spw.
"""
function materialize_leaf(leaf::DimensionalData.AbstractDimTree)
    is_lazy(leaf) || return leaf
    vis = leaf[:vis]
    w = leaf[:weights]
    uvw = leaf[:uvw]
    vis_m = DimArray(_materialize_layer(parent(vis)), dims(vis))
    w_m = DimArray(_materialize_layer(parent(w)), dims(w))
    uvw_m = DimArray(_materialize_layer(parent(uvw)), dims(uvw))
    return _build_leaf(vis_m, w_m, uvw_m; partition_info = metadata(leaf))
end

"""
    materialize_group(leaves; executor = SerialScheduler()) -> Vector

Materialize a group of lazy leaves. A reader extension can mark its backing array
bulk-capable (`_bulk_backend`) and provide `_materialize_group_bulk` to read the
group's shared on-disk bytes once — e.g. the FITS-IDI reader reads a scan's whole
contiguous row span in a single sequential read and extracts every spw's
visibilities from it, instead of per-leaf, per-row `seek`+`read`s. Falls back to
per-leaf [`materialize_leaf`](@ref) when no bulk path applies (including for
already-eager leaves).

`executor` is the OhMyThreads scheduler the bulk path decodes each leaf under
(decode — byte-swap, complex repack, pol permute — is CPU-bound and fans out
over baseline columns). It defaults to `SerialScheduler()`; a streaming solve
passes its within-scan scheduler, so decode reuses the same fan-out budget as
the rest of the group's work.
"""
function materialize_group(leaves; executor = SerialScheduler())
    if !isempty(leaves) && _bulk_backend(parent(first(leaves)[:vis])) !== nothing
        bulk = _materialize_group_bulk(leaves, executor)
        bulk === nothing || return bulk
    end
    return [materialize_leaf(l) for l in leaves]
end

# Bulk-read hooks: an I/O extension overrides `_bulk_backend` for its disk-backed
# array type (returning a non-`nothing` marker) and provides a method of
# `_materialize_group_bulk(leaves, executor)`. Defaults keep `src/` format-neutral
# and make the bulk path inert when no extension is loaded.
_bulk_backend(::AbstractArray) = nothing
function _materialize_group_bulk end

"""
    materialize_group_into!(dests, leaves; executor = SerialScheduler()) -> Bool

Decode a group of sibling-spw lazy `leaves` DIRECTLY into caller-provided
destination arrays — no per-spw intermediate dense copy. `dests[i]` is a
`(vis_dest, weights_dest)` pair the i-th leaf decodes into; each destination's
axis-1 length must equal that leaf's channel count and axes 2-4 its
`(ti, baseline, pol)`. Views into a larger concatenated cube are the intended use
(the fringe search builds its stacked-frequency cube this way, avoiding the
materialize-then-copy round trip and one full in-RAM data copy).

Returns `true` if a bulk backend handled it; `false` otherwise (the caller then
falls back to [`materialize_group`](@ref) + an explicit copy). Format-neutral: the
actual one-read decode lives in the I/O extension's `_materialize_group_bulk_into!`.
`executor` is the decode scheduler, as for [`materialize_group`](@ref).
"""
function materialize_group_into!(dests, leaves; executor = SerialScheduler())
    if !isempty(leaves) && _bulk_backend(parent(first(leaves)[:vis])) !== nothing
        _materialize_group_bulk_into!(dests, leaves, executor) && return true
    end
    return false
end
# Method provided by the I/O extension (mirrors `_materialize_group_bulk`); the
# `_bulk_backend` guard above ensures we only dispatch when a backend is loaded.
function _materialize_group_bulk_into! end

"""
    materialize(uvset::UVSet) -> UVSet

Eagerly materialize every leaf of `uvset`. Use only when the whole dataset fits
in memory; otherwise materialize leaf-by-leaf inside a traversal.
"""
function materialize(uvset::UVSet)
    is_lazy(uvset) || return uvset
    branches = DimensionalData.TreeDict(
        k => materialize_leaf(v) for (k, v) in DimensionalData.branches(uvset)
    )
    return DimensionalData.rebuild(uvset; branches = branches)
end
