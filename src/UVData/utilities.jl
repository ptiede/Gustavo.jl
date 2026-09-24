"""
    check_layer_axes(reference, layers...)

Throw a `DimensionMismatch` unless every array in `layers` shares `reference`'s
axes.

The solver kernels walk parallel `:vis`, `:weights` and `:flags` planes with one
set of loop variables, so a layer whose axes differ would be read at the wrong
cells instead of being reported. Call this where the layers arrive as separate
arrays; a leaf built through `_build_leaf` is already checked.
"""
function check_layer_axes(reference, layers...)
    ref = axes(reference)
    for l in layers
        axes(l) == ref || throw(
            DimensionMismatch("layer has axes $(axes(l)); expected $(ref)"),
        )
    end
    return nothing
end

"""
    sanitize_source(name::AbstractString) -> Symbol

Sanitize a source name into a valid Julia identifier `Symbol`, always prefixed
with `src_` so the key is identifier-safe (digit-leading catalog names like
`3C273` are otherwise illegal identifiers) and never masquerades as a real
source name. Non-identifier chars are replaced with `_`. Examples:
`"3C273"` → `:src_3C273`, `"Sgr A*"` → `:src_Sgr_A_`,
`"NGC 4486"` → `:src_NGC_4486`. Used as the source-segment of a partition key.
"""
function sanitize_source(name::AbstractString)
    s = replace(strip(String(name)), r"[^A-Za-z0-9_]" => "_")
    isempty(s) && (s = "unknown")
    return Symbol("src_", s)
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

# Collect `xs` into a Vector whose element type is the tightest common supertype
# of what it actually holds — concrete whenever the entries share a type, however
# loosely the source container was typed. An empty `xs` has nothing to join and
# becomes `Vector{Any}`; `Vector{Union{}}` could hold no entry at all.
function _narrow_eltype(xs)
    isempty(xs) && return Vector{Any}(undef, 0)
    return collect(mapreduce(typeof, typejoin, xs), xs)
end

"""
    pol_products(x) -> Vector

The values of the `Polarization` lookup of `x`'s visibilities: the stored
product labels of a leaf or `UVSet` (`"PP"`, `"PQ"`, …), or the feed pairs of a
solver cube (see [`feed_pairs`](@ref)). A `UVSet` answers for its first leaf;
all leaves share one axis.
"""
pol_products(vis::AbstractDimArray) = collect(lookup(vis, Polarization))
pol_products(leaf::PartitionedData) = pol_products(leaf[:vis])
function pol_products(uvset::UVSet)
    bs = DimensionalData.branches(uvset)
    isempty(bs) && error("pol_products: UVSet has no leaves")
    return pol_products(first(values(bs)))
end

"""
    feed_pairs(x) -> Vector{Tuple{Int, Int}}

The `(feed_a, feed_b)` pair each product along the `Polarization` axis of `x`
relates, so that `V[a, b, p] = g_a[feed_a] · S · conj(g_b[feed_b])`. A solver
cube's lookup holds these pairs directly. A leaf or `UVSet` labels its products
`P` (feed 1) and `Q` (feed 2). A Measurement Set's labels are resolved through
each antenna's receptors by
[`feed_pairs(::XRadio.MeasurementSet)`](@ref feed_pairs(::XRadio.MeasurementSet)).
"""
feed_pairs(vis::AbstractDimArray) = _feed_pairs(lookup(vis, Polarization))
feed_pairs(x::Union{PartitionedData, UVSet}) = _feed_pairs(pol_products(x))

_feed_pairs(products::AbstractVector{<:Tuple{Integer, Integer}}) = collect(Tuple{Int, Int}, products)
_feed_pairs(products::AbstractVector{<:AbstractString}) = map(_stored_feed_pair, products)

function _stored_feed_pair(label::AbstractString)
    feed(c) = c == 'P' ? 1 : c == 'Q' ? 2 : throw(
        ArgumentError("a leaf labels its products P (feed 1) and Q (feed 2), got \"$label\"")
    )
    length(label) == 2 || throw(ArgumentError("a product label has two feeds, got \"$label\""))
    return (feed(label[1]), feed(label[2]))
end

"""
    frequencies(x) -> Vector{Float64}

Channel frequencies (Hz) off the `Frequency` lookup of `x`'s visibility array —
a `DimArray`, a leaf `AbstractDimTree`, or a layer selection off one. The raw
coordinate vector, not a lookup wrapper.
"""
frequencies(vis::AbstractDimArray) = parent(lookup(vis, Frequency))
frequencies(leaf::PartitionedData) = frequencies(leaf[:vis])

"""
    timestamps(x) -> Vector{Float64}

Integration times (seconds) off the `Ti` lookup of `x`'s visibility array — a
`DimArray`, a leaf `AbstractDimTree`, or a layer selection off one. The raw
coordinate vector, not a lookup wrapper.
"""
timestamps(vis::AbstractDimArray) = parent(lookup(vis, Ti))
timestamps(leaf::PartitionedData) = timestamps(leaf[:vis])

# ── Time axis ────────────────────────────────────────────────────────────────

"""
    JD_UNIX_EPOCH

Julian Day of 1970-01-01T00:00:00 UTC, the origin of the `Ti` axis.
"""
const JD_UNIX_EPOCH = 2440587.5

"""
    jd_to_unix(jd) -> Float64
    unix_to_jd(t) -> Float64

Convert between a Julian Day and the `Ti` axis' seconds since
[`JD_UNIX_EPOCH`](@ref).

A Julian Day near the present is ~2.46e6, where a `Float64` resolves only
~40 µs, so a caller holding the day and its fraction separately — as FITS-IDI
`DATE`/`TIME` and the AIPS `DATE` PTYPE pair both do — must subtract the epoch
from the integer part *before* adding the fraction to keep sub-microsecond
timestamps.
"""
jd_to_unix(jd::Real) = (Float64(jd) - JD_UNIX_EPOCH) * 86400.0
unix_to_jd(t::Real) = JD_UNIX_EPOCH + Float64(t) / 86400.0

# ── Selecting a product ───────────────────────────────────────────────

"""
    pol_index(x, pair::Tuple{Integer, Integer}) -> Int

The index along the `Polarization` axis of `x` of the product relating feed
`pair[1]` of the first antenna to feed `pair[2]` of the second (see
[`feed_pairs`](@ref)). Throws `KeyError` when `x` has no such product.
"""
pol_index(x, pair::Tuple{Integer, Integer}) = _pol_index_lookup(feed_pairs(x), pair)

function _pol_index_lookup(pairs::AbstractVector{<:Tuple{Integer, Integer}}, pair)
    i = findfirst(==(pair), pairs)
    i === nothing && throw(KeyError(pair))
    return i
end

"""
    pol_at(x, pair::Tuple{Integer, Integer}) -> DimensionalData.At

A selector for the product of `x` relating feed pair `pair`, for indexing
`Polarization`-dimensioned arrays:

```julia
amp = abs.(stack[:vis][Polarization = pol_at(stack, (1, 1))])
```
"""
pol_at(x, pair::Tuple{Integer, Integer}) = At(pol_products(x)[pol_index(x, pair)])
