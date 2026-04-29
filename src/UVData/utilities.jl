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
    partition_key(source_key, scan_name; sub_scan_name = "") -> Symbol

Compose a leaf branch key. Default shape is `:<source_key>_scan_<scan_name>`.
When `sub_scan_name` is non-empty (sub-array / sub-scan disambiguation,
mirrors xradio's `SUB_SCAN_NUMBER` partitioning), the key becomes
`:<source_key>_scan_<scan_name>_<sub_scan_name>`.
"""
function partition_key(
        source_key::Symbol, scan_name::AbstractString;
        sub_scan_name::AbstractString = "",
    )
    return isempty(sub_scan_name) ?
        Symbol(source_key, :_scan_, scan_name) :
        Symbol(source_key, :_scan_, scan_name, :_, sub_scan_name)
end

# Back-compat: tests / fixtures historically called with an Integer scan id.
partition_key(source_key::Symbol, scan_idx::Integer; kwargs...) =
    partition_key(source_key, string(scan_idx); kwargs...)

"""
    scan_key(id) -> Symbol

Legacy scan-key helper retained for back-compat. Returns `:scan_<id>`.
"""
scan_key(id) = Symbol("scan_", id)

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
