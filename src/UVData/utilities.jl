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
    partition_key(info::PartitionInfo) -> Symbol

Compose the leaf branch key by walking `partition_axes(info)`.
Empty axis values are skipped; remaining values are joined with
underscores. Default key shape (xradio MSv4):
`:<source>_<spw_name>_scan_<scan_name>[_<sub_scan_name>]`.

Adding a new axis is a one-line change to `DEFAULT_PARTITION_AXES` (or
a method override of `partition_axes`); this function never needs to
change.
"""
partition_key(info::PartitionInfo) = partition_key(info, partition_axes(info))

function partition_key(info::PartitionInfo, axes)
    parts = String[]
    for ax in axes
        s = ax.value(info)
        isempty(s) || push!(parts, s)
    end
    return Symbol(join(parts, "_"))
end

"""
    partition_key(; source_key, scan_name, spw_name = "spw_0",
                    sub_scan_name = "") -> Symbol

Lightweight key-only helper for tests / fixtures that don't have a
real `PartitionInfo`. Mirrors `DEFAULT_PARTITION_AXES` shape; production
code goes through `partition_key(info)` exclusively.
"""
function partition_key(;
        source_key::Symbol,
        scan_name::AbstractString,
        spw_name::AbstractString = "spw_0",
        sub_scan_name::AbstractString = "",
    )
    parts = String[string(source_key)]
    isempty(spw_name) || push!(parts, spw_name)
    isempty(scan_name) || push!(parts, "scan_" * scan_name)
    isempty(sub_scan_name) || push!(parts, sub_scan_name)
    return Symbol(join(parts, "_"))
end

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
