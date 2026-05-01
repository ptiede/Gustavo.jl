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
        TAnt <: AntennaTable, TBL <: BaselineIndex,
        TFS <: AbstractFrequencySetup, TEx <: NamedTuple,
    }
    source_name::String
    source_key::Symbol
    field_name::String
    scan_name::String
    scan_intents::Vector{String}
    sub_scan_name::String
    spw_name::String
    subarray_name::String
    intent::String
    ra::Float64
    dec::Float64
    ddi::Int
    partition_name::String
    antennas::TAnt
    baselines::TBL
    record_order::Vector{Tuple{Int, Int}}
    extra_columns::TEx
    freq_setup::TFS
end

function PartitionInfo(;
        source_name::AbstractString, source_key::Symbol,
        scan_name::AbstractString, ra::Real, dec::Real,
        antennas::AntennaTable,
        baselines::BaselineIndex,
        record_order::Vector{Tuple{Int, Int}},
        freq_setup::AbstractFrequencySetup,
        extra_columns::NamedTuple = NamedTuple(),
        scan_intents::AbstractVector{<:AbstractString} = String[],
        sub_scan_name::AbstractString = "",
        spw_name::AbstractString = "spw_0",
        subarray_name::AbstractString = "",
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
        String(spw_name), String(subarray_name), String(intent),
        Float64(ra), Float64(dec),
        Int(ddi), pname, antennas, baselines, record_order,
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
        get(kwargs, :subarray_name, info.subarray_name),
        get(kwargs, :intent, info.intent),
        get(kwargs, :ra, info.ra),
        get(kwargs, :dec, info.dec),
        get(kwargs, :ddi, info.ddi),
        get(kwargs, :partition_name, info.partition_name),
        get(kwargs, :antennas, info.antennas),
        get(kwargs, :baselines, info.baselines),
        get(kwargs, :record_order, info.record_order),
        get(kwargs, :extra_columns, info.extra_columns),
        get(kwargs, :freq_setup, info.freq_setup),
    )
end

"""
    PartitionAxis(name::Symbol, value::Function)

A single axis component of the leaf partition key. `value(info)` returns
its rendered string for a given `PartitionInfo`; an empty string causes
the axis to be omitted from the key.

Adding a new partition axis is a one-line change: append a
`PartitionAxis(:newaxis, info -> ...)` entry to
`DEFAULT_PARTITION_AXES`. No other code path needs to change — the
generic `partition_key(info)` walks the registered list.
"""
struct PartitionAxis
    name::Symbol
    value::Function
end

"""
    DEFAULT_PARTITION_AXES

Ordered tuple of axes contributing to the leaf key, in xradio MSv4
shape: `:source`, `:spw`, `:scan`, `:sub_scan`. Each renderer carries
its own prefix convention (e.g. `scan_<n>`); empty values are skipped.

Default key shape: `:<source>_<spw_name>_scan_<scan_name>[_<sub_scan>]`,
e.g. `:M3C273_spw_0_scan_1` or `:M3C273_spw_1_scan_1_A`.
"""
const DEFAULT_PARTITION_AXES = (
    PartitionAxis(:source, info -> string(info.source_key)),
    PartitionAxis(:spw, info -> info.spw_name),
    PartitionAxis(:subarray, info -> info.subarray_name),
    PartitionAxis(
        :scan,
        info -> isempty(info.scan_name) ? "" : "scan_" * info.scan_name,
    ),
    PartitionAxis(:sub_scan, info -> info.sub_scan_name),
)

"""
    partition_axes(info::PartitionInfo) -> Tuple{PartitionAxis, ...}

Hook to override the axis list for a custom `PartitionInfo` shape (e.g.
to add an `intent` segment in a downstream extension). Defaults to
`DEFAULT_PARTITION_AXES`.
"""
partition_axes(::PartitionInfo) = DEFAULT_PARTITION_AXES

function _show_partition(io::IO, leaf::DimensionalData.AbstractDimTree)
    info = DimensionalData.metadata(leaf)
    nti = length(obs_time(leaf))
    nbl = length(info.baselines.pairs)
    sub = isempty(info.sub_scan_name) ? "" : "/" * info.sub_scan_name
    return print(io, "UVDataSet(scan=$(info.scan_name)$(sub), nti=$(nti), nbaselines=$(nbl))")
end

# DimensionalData's `print_metadata_block` calls `isempty(metadata)` —
# `PartitionInfo` is never empty, so short-circuit before iterate is hit.
Base.isempty(::PartitionInfo) = false

function Base.show(io::IO, ::MIME"text/plain", info::PartitionInfo)
    nant = length(info.antennas)
    nbl = length(info.baselines.pairs)
    nrec = length(info.record_order)
    extras = isempty(info.extra_columns) ? "—" :
        join(string.(keys(info.extra_columns)), ", ")
    intents = isempty(info.scan_intents) ? "—" : join(info.scan_intents, ",")
    sub = isempty(info.sub_scan_name) ? "" : "  sub_scan=$(info.sub_scan_name)"
    println(io, "PartitionInfo $(info.partition_name)")
    println(io, "  source       = $(info.source_name)  (key=:$(info.source_key))  field=$(info.field_name)")
    println(io, "  ra/dec       = $(round(info.ra; digits = 6)) / $(round(info.dec; digits = 6)) (rad)")
    println(io, "  scan         = $(info.scan_name)$(sub)  intents=$(intents)")
    println(io, "  spw          = $(info.spw_name)  ddi=$(info.ddi)  subarray=$(isempty(info.subarray_name) ? "—" : info.subarray_name)")
    println(io, "  antennas     = $(nant)   baselines = $(nbl)   records = $(nrec)")
    println(io, "  freq_setup   = $(info.freq_setup.name) ($(length(channel_freqs(info.freq_setup))) chan)")
    return print(io, "  extras       = $(extras)")
end

Base.show(io::IO, info::PartitionInfo) =
    print(io, "PartitionInfo($(info.partition_name))")
