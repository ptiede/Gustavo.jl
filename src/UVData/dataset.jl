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
    materialize_leaf(leaf) -> DimTree

Return a leaf with all data layers (`vis`/`weights`/`uvw`/`flag`) read into
dense in-memory arrays. A no-op (returns `leaf`) if the leaf is already eager.
This is the natural unit a solver pulls into memory: one scan × one spw.
"""
function materialize_leaf(leaf::DimensionalData.AbstractDimTree)
    is_lazy(leaf) || return leaf
    vis = leaf[:vis]
    w = leaf[:weights]
    uvw = leaf[:uvw]
    flag = leaf[:flag]
    vis_m = DimArray(_materialize_layer(parent(vis)), dims(vis))
    w_m = DimArray(_materialize_layer(parent(w)), dims(w))
    uvw_m = DimArray(_materialize_layer(parent(uvw)), dims(uvw))
    flag_m = DimArray(_materialize_layer(parent(flag)), dims(flag))
    return _build_leaf(vis_m, w_m, uvw_m, flag_m; partition_info = metadata(leaf))
end

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
