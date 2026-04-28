"""
    sanitize_source(name::AbstractString) -> Symbol

Sanitize a source name into a valid Julia identifier `Symbol`. Non-identifier
chars are replaced with `_`; names starting with a digit are prefixed with
`M`. Examples: `"3C273"` → `:M3C273`, `"Sgr A*"` → `:Sgr_A_`,
`"NGC 4486"` → `:NGC_4486`. Used as the source-segment of a partition key.
"""
function sanitize_source(name::AbstractString)
    s = replace(strip(String(name)), r"[^A-Za-z0-9_]" => "_")
    isempty(s) && return :unknown
    return isdigit(first(s)) ? Symbol("M", s) : Symbol(s)
end

"""
    partition_key(source_key::Symbol, scan_idx::Integer) -> Symbol

Compose a leaf branch key as `:<source_key>_scan_<scan_idx>`.
"""
partition_key(source_key::Symbol, scan_idx::Integer) =
    Symbol(source_key, :_scan_, scan_idx)

"""
    scan_key(id::Integer) -> Symbol

Legacy scan key helper retained for backwards compatibility — note
that with multi-source UVSets the per-source `partition_key(src, id)` is
preferred. Returns `:scan_<id>`.
"""
scan_key(id::Integer) = Symbol("scan_", id)

# Preserve DimArray dim metadata when callers hand back a plain array result
# of the same shape (e.g. an externally-built vis cube). Already-DimArray
# inputs are passed through unchanged.
_rewrap_like(A::AbstractDimArray, ::AbstractDimArray) = A
_rewrap_like(A::AbstractDimArray, _) = A
_rewrap_like(A, ref::AbstractDimArray) =
    size(A) == size(ref) ? DimArray(A, dims(ref)) : A
_rewrap_like(A, _) = A

"""
    decode_baseline(bl::Integer) -> (a::Int, b::Int)

AIPS UVFITS BASELINE-column convention: pack `(a, b)` antenna indices as
`bl = a*256 + b`. Inverse on read.
"""
decode_baseline(bl::Integer) = (bl ÷ 256, bl % 256)

"""
    assign_scans(times, scans) -> Vector{Int}

Map each entry of `times` to the index of the half-open scan interval
`[scans[s].lower, scans[s].upper)` containing it; 0 if no scan matches.
"""
function assign_scans(times, scans)
    idx = zeros(Int, length(times))
    @inbounds for i in eachindex(times)
        for s in eachindex(scans)
            if scans[s].lower ≤ times[i] < scans[s].upper
                idx[i] = s
                break
            end
        end
    end
    return idx
end

"""
    pol_products(x) -> Vector{String}

Return the polarization-product labels (e.g. `["PP", "PQ", "QP", "QQ"]`)
read off the `Pol` dimension of `x`'s underlying visibility array. Works
on a `DimArray` (the lookup), a leaf `AbstractDimTree`, or a `UVSet`
(uses the first leaf — all leaves share the same Pol axis on read).
"""
pol_products(vis::AbstractDimArray) = collect(lookup(vis, Pol))
pol_products(leaf::AbstractDimTree) = pol_products(leaf[:vis])
function pol_products(uvset::UVSet)
    bs = DimensionalData.branches(uvset)
    isempty(bs) && error("pol_products: UVSet has no leaves")
    return pol_products(first(values(bs)))
end

