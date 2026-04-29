"""
    BaselineIndex

Pair-canonical baseline metadata. The identity is the antenna-index
tuple `(a, b)`; AIPS UVFITS's `bl = a*256 + b` packing lives only
inside the FITS extension and never leaks out here.

Per-baseline (each vector has one entry per unique baseline):
- `pairs`        — `(a, b)` antenna-index pairs (the canonical key set).
- `labels`       — `"AA-AX"` style strings (parallel to `pairs`).
- `ant1_names`   — antenna-1 (POLA-side) name (parallel to `pairs`).
- `ant2_names`   — antenna-2 (POLB-side) name (parallel to `pairs`).

Per-record (length `nrecord`, only meaningful in the legacy flat
layout where `(time, baseline)` are fused on the `Integration` axis;
not used by the partitioned `UVSet` storage):
- `pairs_per_record` — `(a, b)` pair per record.

Lookup:
- `lookup` — `Dict((a, b) => slot)` for O(1) pair-to-slot mapping.
"""
struct BaselineIndex{TPairs, TLookup, TLabels, TNames}
    pairs_per_record::TPairs   # per-record (a, b) tuples (legacy flat layout)
    pairs::TPairs              # unique deduped (a, b) pairs
    lookup::TLookup            # (a, b) → slot index in `pairs`
    labels::TLabels            # "AA-AX" string per baseline (parallel to pairs)
    ant1_names::TNames         # antenna-1 name per baseline (parallel to pairs)
    ant2_names::TNames         # antenna-2 name per baseline (parallel to pairs)
end

"""
    BaselineIndex(pairs_per_record, pairs; antenna_names = nothing) -> BaselineIndex

Build a `BaselineIndex` from per-record and deduped `(a, b)` antenna
index pairs. Derives `lookup`, `labels`, `ant1_names`, `ant2_names`
from `pairs` (resolving names from `antenna_names` when supplied,
falling back to `"ant<idx>"`).
"""
function BaselineIndex(
        pairs_per_record::AbstractVector{<:Tuple{Integer, Integer}},
        pairs::AbstractVector{<:Tuple{Integer, Integer}};
        antenna_names::Union{AbstractVector{<:AbstractString}, Nothing} = nothing,
    )
    _resolve_name(idx) = if antenna_names !== nothing && 1 <= idx <= length(antenna_names)
        string(antenna_names[idx])
    else
        string("ant", idx)
    end
    pairs_v = collect(pairs)
    ant1 = [_resolve_name(a) for (a, _) in pairs_v]
    ant2 = [_resolve_name(b) for (_, b) in pairs_v]
    labels = string.(ant1, "-", ant2)
    lookup = Dict{eltype(pairs_v), Int}(p => i for (i, p) in enumerate(pairs_v))
    return BaselineIndex(collect(pairs_per_record), pairs_v, lookup, labels, ant1, ant2)
end
