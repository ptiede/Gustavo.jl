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


"""
    PartitionInfo

Per-leaf metadata for one MSv4-shaped partition. Mirrors xradio's
`MeasurementSetXdt.get_partition_info()` shape.

Holds source identification (`source_name`/`source_key`/`field_name`/`ra`/
`dec`), the spectral-window handle (`spw_name`/`ddi`), `intent`, the leaf's
`baselines::BaselineIndex`, the per-record bookkeeping carried in
`record_order`/`date_param`/`extra_columns`, the human-readable
`partition_name`, the leaf's own `freq_setup`, and the leaf's scan handles.

Scan model follows xradio's `ScanArray` (schema.py:779): the *primary*
scan label is cached here as `scan_name::String`, while the per-time
scan-name DimVector lives as the `:scan_name` data layer of the leaf
DimTree (alongside `vis`/`weights`/`uvw`/`flag`). `scan_intents` tracks
`scan_intents` from the MSv4 schema. `sub_scan_name` distinguishes
sub-arrays / sub-scans that would otherwise share `(source, scan)`
keys (default `""`); included in the partition key only when non-empty.

Use `PartitionInfo(; ...)` to construct and `update(info; ...)` to produce
a copy with field overrides.
"""
struct PartitionInfo{
        TBL <: BaselineIndex, TFS <: AbstractFrequencySetup,
        TDP, TEx <: NamedTuple,
    }
    source_name::String
    source_key::Symbol
    field_name::String
    scan_name::String
    scan_intents::Vector{String}
    sub_scan_name::String
    spw_name::String
    intent::String
    ra::Float64
    dec::Float64
    ddi::Int
    partition_name::String
    baselines::TBL
    record_order::Vector{Tuple{Int, Int}}
    date_param::TDP
    extra_columns::TEx
    freq_setup::TFS
end

function PartitionInfo(;
        source_name::AbstractString, source_key::Symbol,
        scan_name::AbstractString, ra::Real, dec::Real,
        baselines::BaselineIndex,
        record_order::Vector{Tuple{Int, Int}},
        date_param::AbstractMatrix,
        freq_setup::AbstractFrequencySetup,
        extra_columns::NamedTuple = NamedTuple(),
        scan_intents::AbstractVector{<:AbstractString} = String[],
        sub_scan_name::AbstractString = "",
        spw_name::AbstractString = "spw_0",
        intent::AbstractString = "",
        ddi::Integer = 0,
        field_name::Union{Nothing, AbstractString} = nothing,
        partition_name::Union{Nothing, AbstractString} = nothing,
        basename::AbstractString = "uvfits",
    )
    field_n = field_name === nothing ? String(source_name) : String(field_name)
    sub_n = String(sub_scan_name)
    pname_default = string(
        basename, "_", ddi, "_", String(source_name), "_scan_", String(scan_name),
        isempty(sub_n) ? "" : string("_", sub_n),
    )
    pname = partition_name === nothing ? pname_default : String(partition_name)
    return PartitionInfo(
        String(source_name), source_key, field_n, String(scan_name),
        String.(collect(scan_intents)), sub_n,
        String(spw_name), String(intent), Float64(ra), Float64(dec),
        Int(ddi), pname, baselines, record_order, date_param,
        extra_columns, freq_setup,
    )
end

"""
    update(info::PartitionInfo; kwargs...) -> PartitionInfo

Return a copy of `info` with the fields named by `kwargs` overridden.
Replaces the prior `merge(info, (; field = value))` NamedTuple pattern.
"""
function update(info::PartitionInfo; kwargs...)
    return PartitionInfo(
        get(kwargs, :source_name, info.source_name),
        get(kwargs, :source_key, info.source_key),
        get(kwargs, :field_name, info.field_name),
        get(kwargs, :scan_name, info.scan_name),
        get(kwargs, :scan_intents, info.scan_intents),
        get(kwargs, :sub_scan_name, info.sub_scan_name),
        get(kwargs, :spw_name, info.spw_name),
        get(kwargs, :intent, info.intent),
        get(kwargs, :ra, info.ra),
        get(kwargs, :dec, info.dec),
        get(kwargs, :ddi, info.ddi),
        get(kwargs, :partition_name, info.partition_name),
        get(kwargs, :baselines, info.baselines),
        get(kwargs, :record_order, info.record_order),
        get(kwargs, :date_param, info.date_param),
        get(kwargs, :extra_columns, info.extra_columns),
        get(kwargs, :freq_setup, info.freq_setup),
    )
end

function _show_partition(io::IO, leaf::DimensionalData.AbstractDimTree)
    info = DimensionalData.metadata(leaf)
    nti = length(obs_time(leaf))
    nbl = length(info.baselines.pairs)
    sub = isempty(info.sub_scan_name) ? "" : "/" * info.sub_scan_name
    return print(io, "UVDataSet(scan=$(info.scan_name)$(sub), nti=$(nti), nbaselines=$(nbl))")
end
