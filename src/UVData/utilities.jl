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

# ── Polarization-by-name selectors ────────────────────────────────────
#
# Internal MSv4 canonical labels are "PP" / "PQ" / "QP" / "QQ" (feed-1
# vs feed-2 cross-product). EHT users habitually write "RR" / "RL" /
# "LR" / "LL" (or X/Y for ALMA pre-PolConvert); these are aliases for the
# same canonical pair on a circular- or linear-feed antenna. The map is
# unambiguous: feed letter 1 → P, feed letter 2 → Q.

_POL_FEED_LETTERS = ('P', 'Q', 'R', 'L', 'X', 'Y')

function _canonical_pol_label(label::AbstractString)
    s = uppercase(strip(String(label)))
    length(s) == 2 || throw(ArgumentError(
        "pol_index: expected a 2-character label like \"PP\" / \"RR\" / \"XY\", got \"$label\""
    ))
    return string(_canonical_feed(s[1]), _canonical_feed(s[2]))
end
_canonical_feed(c::Char) = c in ('P', 'R', 'X') ? 'P' :
    c in ('Q', 'L', 'Y') ? 'Q' :
    throw(ArgumentError("pol_index: unsupported feed letter '$c' (expected one of P/Q/R/L/X/Y)"))

# Convert a `(PolType, PolType)` pair to a canonical "PP"/"PQ"/"QP"/"QQ" label.
_canonical_pol_label(p::Tuple{<:PolTypes, <:PolTypes}) =
    string(_feed_char(p[1]), _feed_char(p[2]))
_feed_char(::Union{RPol, XPol}) = 'P'
_feed_char(::Union{LPol, YPol}) = 'Q'

"""
    pol_index(x, label) -> Int

Resolve a polarization-product `label` to its integer index along the
`Pol` axis of `x` (`x` may be a `DimArray`, a leaf `AbstractDimTree`, a
`UVSet`, or any concrete `Vector{String}` of pol products). Accepts:

- a string like `"PP"` / `"RR"` / `"LL"` / `"XY"` — the EHT shorthands
  (`R/L`, `X/Y`) are folded onto the canonical MSv4 `P/Q` pair before
  lookup, so e.g. `pol_index(leaf, "RR") == pol_index(leaf, "PP")`;
- a tuple `(RPol(), RPol())` of `PolarizedTypes` — same canonicalization
  via `nominal_basis` semantics.

Throws `KeyError` when the canonicalized label is absent from `x`.
"""
pol_index(x, label::AbstractString) = _pol_index_lookup(x, _canonical_pol_label(label))
pol_index(x, label::Tuple{<:PolTypes, <:PolTypes}) =
    _pol_index_lookup(x, _canonical_pol_label(label))

_pol_index_lookup(products::AbstractVector{<:AbstractString}, canon::AbstractString) =
    let i = findfirst(==(canon), products)
        i === nothing ? throw(KeyError(canon)) : i
    end
_pol_index_lookup(x, canon) = _pol_index_lookup(pol_products(x), canon)

"""
    pol_at(label) -> DimensionalData.At

DimensionalData selector for the canonical pol label, suitable for
indexing `Pol`-dimensioned arrays:

```julia
amp = abs.(stack[:vis][Pol = pol_at("RR")])
```

Equivalent to `At(canonical_label(label))` after folding the EHT
shorthands (`R/L`, `X/Y`) onto the MSv4 `P/Q` convention.
"""
pol_at(label::AbstractString) = At(_canonical_pol_label(label))
pol_at(label::Tuple{<:PolTypes, <:PolTypes}) = At(_canonical_pol_label(label))
