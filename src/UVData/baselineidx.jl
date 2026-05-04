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

# ── Lookup sugar ──────────────────────────────────────────────────────
#
# Resolve a baseline key to its slot index. Accepts:
#   `(a::Int, b::Int)`        — antenna-index pair (the canonical key);
#   `("AA", "AX")` or `["AA", "AX"]` — antenna-name pair, ordered;
#   `"AA-AX"`                 — the dash-joined label.
# Throws `KeyError` on miss; use [`Base.haskey`](@ref) to test.

"""
    baseline_index(bls::BaselineIndex, key) -> Int

Slot index into `bls.pairs` / `bls.labels` for the requested baseline.
Same key shapes as `Base.getindex(bls, key)`. Returns `0` when the
baseline is absent (use `haskey(bls, key)` for an explicit test).
"""
function baseline_index(bls::BaselineIndex, key::Tuple{Integer, Integer})
    return get(bls.lookup, (Int(key[1]), Int(key[2])), 0)
end
function baseline_index(bls::BaselineIndex, key::Tuple{<:AbstractString, <:AbstractString})
    a = findfirst(==(String(key[1])), bls.ant1_names)
    a === nothing && return 0
    # Walk forward from `a` so we match the (ant1, ant2) ordering.
    for i in eachindex(bls.pairs)
        if bls.ant1_names[i] == key[1] && bls.ant2_names[i] == key[2]
            return i
        end
    end
    return 0
end
function baseline_index(bls::BaselineIndex, key::AbstractVector{<:AbstractString})
    length(key) == 2 || throw(ArgumentError(
        "baseline_index: name-vector keys must have length 2 (got $(length(key)))",
    ))
    return baseline_index(bls, (String(key[1]), String(key[2])))
end
function baseline_index(bls::BaselineIndex, key::AbstractString)
    i = findfirst(==(String(key)), bls.labels)
    return i === nothing ? 0 : i
end

function Base.getindex(bls::BaselineIndex, key)
    i = baseline_index(bls, key)
    i == 0 && throw(KeyError(key))
    return i
end

Base.haskey(bls::BaselineIndex, key) = baseline_index(bls, key) > 0
Base.length(bls::BaselineIndex) = length(bls.pairs)
