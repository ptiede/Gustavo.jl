"""
    BaselineIndex

Encapsulates baseline encoding field list. End-users should rely on convenience functions
(`nbaselines`, `antenna_names`, etc.) rather than accessing these fields
directly.

Per-baseline (each vector has one entry per unique baseline):
- `pairs`        — `(a_idx, b_idx)` antenna-index pairs (Int).
- `unique_codes` — sorted unique AIPS baseline codes (Float64, `256·a + b`).
- `labels`       — `"AA-AX"` style strings (parallel to `pairs`).
- `ant1_names`   — antenna-1 (POLA-side) name (parallel to `pairs`).
- `ant2_names`   — antenna-2 (POLB-side) name (parallel to `pairs`).

Lookup:
- `lookup` — `Dict(code => index)` for O(1) AIPS-code-to-index mapping.

Per-integration (length `nint`, only meaningful in the legacy flat layout
where `(time, baseline)` are fused on the `Integration` axis; not used by
the partitioned `UVSet` storage):
- `codes` — per-integration AIPS baseline code as Float64.
"""
struct BaselineIndex{TCodes, TPairs, TLookup, TLabels, TNames}
    codes::TCodes              # per-integration AIPS codes (legacy flat layout)
    pairs::TPairs              # unique (a_idx, b_idx) pairs
    lookup::TLookup            # code → index in pairs
    unique_codes::TCodes       # sorted unique codes (per-baseline)
    labels::TLabels            # "AA-AX" string per baseline (parallel to pairs)
    ant1_names::TNames         # antenna-1 name per baseline (parallel to pairs)
    ant2_names::TNames         # antenna-2 name per baseline (parallel to pairs)
end

# Back-compat constructor — derives `labels`/`ant1_names`/`ant2_names` from
# `pairs` when only the four legacy fields are passed. Used by the test
# fixtures that built `BaselineIndex` by hand before the parallel name
# vectors were introduced.
function BaselineIndex(
        codes, pairs, lookup, unique_codes;
        antenna_names::Union{AbstractVector{<:AbstractString}, Nothing} = nothing
    )
    _resolve_name(idx) = if antenna_names !== nothing && 1 <= idx <= length(antenna_names)
        string(antenna_names[idx])
    else
        string("ant", idx)
    end
    ant1 = [_resolve_name(a) for (a, _) in pairs]
    ant2 = [_resolve_name(b) for (_, b) in pairs]
    labels = string.(ant1, "-", ant2)
    return BaselineIndex(codes, pairs, lookup, unique_codes, labels, ant1, ant2)
end
