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
    materialize_group(leaves) -> Vector

Materialize a group of lazy leaves. A reader extension can mark its backing array
bulk-capable (`_bulk_backend`) and provide `_materialize_group_bulk` to read the
group's shared on-disk bytes ONCE — e.g. the FITS-IDI reader reads a scan's whole
contiguous row span in a single sequential read and extracts every band's
visibilities from it, instead of per-leaf, per-row `seek`+`read`s. Falls back to
per-leaf [`materialize_leaf`](@ref) when no bulk path applies (including for
already-eager leaves).
"""
function materialize_group(leaves)
    if !isempty(leaves) && _bulk_backend(parent(first(leaves)[:vis])) !== nothing
        bulk = _materialize_group_bulk(leaves)
        bulk === nothing || return bulk
    end
    return [materialize_leaf(l) for l in leaves]
end

# Bulk-read hooks: an I/O extension overrides `_bulk_backend` for its disk-backed
# array type (returning a non-`nothing` marker) and provides a method of
# `_materialize_group_bulk(leaves)`. Defaults keep `src/` format-neutral
# and make the bulk path inert when no extension is loaded.
_bulk_backend(::AbstractArray) = nothing
function _materialize_group_bulk end

"""
    materialize_group_into!(dests, leaves) -> Bool

Decode a group of sibling-band lazy `leaves` DIRECTLY into caller-provided
destination arrays — no per-band intermediate dense copy. `dests[i]` is a
`(vis_dest, weights_dest)` pair the i-th leaf decodes into; each destination's
axis-1 length must equal that leaf's channel count and axes 2-4 its
`(ti, baseline, pol)`. Views into a larger concatenated cube are the intended use
(the fringe search builds its stacked-frequency cube this way, avoiding the
materialize-then-copy round trip and one full in-RAM data copy).

Returns `true` if a bulk backend handled it; `false` otherwise (the caller then
falls back to [`materialize_group`](@ref) + an explicit copy). Format-neutral: the
actual one-read decode lives in the I/O extension's `_materialize_group_bulk_into!`.
"""
function materialize_group_into!(dests, leaves)
    if !isempty(leaves) && _bulk_backend(parent(first(leaves)[:vis])) !== nothing
        _materialize_group_bulk_into!(dests, leaves) && return true
    end
    return false
end
# Method provided by the I/O extension (mirrors `_materialize_group_bulk`); the
# `_bulk_backend` guard above ensures we only dispatch when a backend is loaded.
function _materialize_group_bulk_into! end

# Decode concurrency for the bulk reader: how many tasks a single leaf's
# vis/weights fill spreads its baseline columns over. Decode (byte-swap + complex
# repack + pol permute) is CPU-bound, so threading it uses the cores the solve's
# memory cap otherwise leaves idle. Default 1 = sequential (unchanged behavior for
# tests and any non-solver caller); the fringe solve raises it around its passes.
const _DECODE_NTASKS = Ref(1)

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
