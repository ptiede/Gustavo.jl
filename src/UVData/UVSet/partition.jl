# ── Partitions accessor (tab-completable wrapper) ────────────────────────────

"""
    Partitions(uvset::UVSet)
    partitions(uvset::UVSet) -> Partitions

Wrapper struct exposing each branch key as a tab-completable property.
`partitions(uvset).<TAB>` lists sanitized partition keys; access a leaf
by `partitions(uvset).M3C273_scan_1`. `uvset.M3C273_scan_1` works equally
well via the inherited `AbstractDimTree.getproperty`.
"""
struct Partitions{T <: UVSet}
    uvset::T
end

partitions(uvset::UVSet) = Partitions(uvset)

Base.propertynames(p::Partitions) =
    Tuple(collect(keys(DimensionalData.branches(getfield(p, :uvset)))))

function Base.getproperty(p::Partitions, k::Symbol)
    k === :uvset && return getfield(p, :uvset)
    return DimensionalData.branches(getfield(p, :uvset))[k]
end

Base.show(io::IO, ::MIME"text/plain", p::Partitions) =
    Base.show(io, MIME"text/plain"(), summary(getfield(p, :uvset)))


function _partition_info(;
        source_name::AbstractString, source_key::Symbol,
        scan_idx::Integer, ra::Real, dec::Real,
        baselines::BaselineIndex,
        record_order::Vector{Tuple{Int, Int}},
        date_param::AbstractMatrix,
        extra_columns::NamedTuple = NamedTuple(),
        spw_name::AbstractString = "spw_0",
        intent::AbstractString = "",
        ddi::Integer = 0,
        field_name::Union{Nothing, AbstractString} = nothing,
        partition_name::Union{Nothing, AbstractString} = nothing,
        basename::AbstractString = "uvfits",
    )
    field_n = field_name === nothing ? String(source_name) : String(field_name)
    scan_n = string("scan_", scan_idx)
    pname = partition_name === nothing ?
        string(basename, "_", ddi, "_", String(source_name), "_", scan_n) :
        String(partition_name)
    return (;
        source_name = String(source_name),
        source_key = source_key,
        field_name = field_n,
        scan_name = scan_n,
        scan_idx = Int(scan_idx),
        spw_name = String(spw_name),
        intent = String(intent),
        ra = Float64(ra),
        dec = Float64(dec),
        ddi = Int(ddi),
        partition_name = pname,
        baselines = baselines,
        record_order = record_order,
        date_param = date_param,
        extra_columns = extra_columns,
    )
end

function _show_partition(io::IO, leaf::DimensionalData.AbstractDimTree)
    info = DimensionalData.metadata(leaf)
    nti = length(obs_time(leaf))
    nbl = length(info.baselines.pairs)
    return print(io, "UVDataSet(scan=$(info.scan_idx), nti=$(nti), nbaselines=$(nbl))")
end
