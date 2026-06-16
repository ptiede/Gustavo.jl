# ── Streaming dataset API (format-neutral) ───────────────────────────────────
#
# Large visibility files (FITS-IDI from DiFX can be tens of GB) do not fit in
# memory. The reader builds an ordinary `UVSet` whose per-(scan, spw) leaves
# carry *lazy*, disk-backed `vis`/`weights`/`flag` layers; a solver materializes
# one leaf at a time (a single scan×band chunk does fit). `apply` and the
# selectors work unchanged on a lazy `UVSet`.
#
# `src/` stays format-neutral: laziness is detected through the overridable
# `_layer_is_lazy` trait (the FITS extension adds a method for its chunk array),
# and materialization is just `Array(x)`, which triggers the disk read on any
# `DiskArrays.AbstractDiskArray` without `src/` depending on DiskArrays.

"""
    AbstractUVDataset

Handle to an on-disk visibility dataset that can produce a (possibly lazy)
`UVSet`. Concrete subtypes (e.g. the FITS-IDI reader's `FITSIDIDataset`) live in
I/O extensions.
"""
abstract type AbstractUVDataset end

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
    materialize_leaf(leaf; layers = (:vis, :weights, :uvw, :flag)) -> DimTree

Return a leaf with its data layers read into dense in-memory arrays. A no-op
(returns `leaf`) if the leaf is already eager. This is the natural unit a solver
pulls into memory: one scan × one spw.

`layers` selects which layers are read from the (lazy) backing store. `vis`,
`weights`, and `uvw` are always materialized — they are required to rebuild the
leaf and `uvw` is already eager/tiny in the streaming readers, so it is free.
The one expensive, *optional* layer is `flag`: on a FITS-IDI leaf the flag layer
costs roughly as much to read as `vis`+`weights` combined, yet it carries no
information the weights don't already — the reader zeroes the weight of every
flagged cell (FLAG table included), so `_derive_flag(weights)` (`w <= 0`) is
bit-identical to the on-disk flag layer. Drop `:flag` from `layers` (the fringe
solve does) to skip that redundant read; the rebuilt leaf then derives its flag
from the materialized weights.
"""
function materialize_leaf(
        leaf::DimensionalData.AbstractDimTree;
        layers = (:vis, :weights, :uvw, :flag),
    )
    is_lazy(leaf) || return leaf
    vis = leaf[:vis]
    w = leaf[:weights]
    uvw = leaf[:uvw]
    vis_m = DimArray(_materialize_layer(parent(vis)), dims(vis))
    w_m = DimArray(_materialize_layer(parent(w)), dims(w))
    uvw_m = DimArray(_materialize_layer(parent(uvw)), dims(uvw))
    flag_m = if :flag in layers
        flag = leaf[:flag]
        DimArray(_materialize_layer(parent(flag)), dims(flag))
    else
        nothing   # _build_leaf derives flag from weights (w <= 0)
    end
    return _build_leaf(vis_m, w_m, uvw_m, flag_m; partition_info = metadata(leaf))
end

"""
    materialize_group(leaves; layers = (:vis, :weights, :uvw, :flag)) -> Vector

Materialize a group of lazy leaves. A reader extension can mark its backing array
bulk-capable (`_bulk_backend`) and provide `_materialize_group_bulk` to read the
group's shared on-disk bytes ONCE — e.g. the FITS-IDI reader reads a scan's whole
contiguous row span in a single sequential read and extracts every band's
visibilities from it, instead of per-leaf, per-row `seek`+`read`s. Falls back to
per-leaf [`materialize_leaf`](@ref) when no bulk path applies (including for
already-eager leaves). `layers` is forwarded to both paths.
"""
function materialize_group(leaves; layers = (:vis, :weights, :uvw, :flag))
    if !isempty(leaves) && _bulk_backend(parent(first(leaves)[:vis])) !== nothing
        bulk = _materialize_group_bulk(leaves, layers)
        bulk === nothing || return bulk
    end
    return [materialize_leaf(l; layers = layers) for l in leaves]
end

# Bulk-read hooks: an I/O extension overrides `_bulk_backend` for its disk-backed
# array type (returning a non-`nothing` marker) and provides a method of
# `_materialize_group_bulk(leaves, layers)`. Defaults keep `src/` format-neutral
# and make the bulk path inert when no extension is loaded.
_bulk_backend(::AbstractArray) = nothing
function _materialize_group_bulk end

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
