"""
    UVSet <: DimensionalData.AbstractDimTree

The in-memory form of a UVFITS file, as [`load_uvfits`](@ref) returns it: a
flat `OrderedDict` of MSv4-shaped partition leaves under `branches`, keyed by
sanitized `:<source>_<spw>_scan_<n>` Symbols (e.g. `:src_3C273_spw_0_scan_1`).
[`uvset_to_processingset`](@ref) converts it to the `XRadio.ProcessingSet` the
solver reads.

Each leaf is a plain `DimTree` carrying:

- `data`     : `:vis`, `:weights`, `:flags`, `:uvw` `DimArray` layers.
  `:flags` is `Bool` on the `:vis` axes and is `true` where the datum must
  not be used; `:weights` says how good it would have been. The two are
  independent, matching MSv4's `FLAG`/`WEIGHT` pair: a flagged cell may
  carry a positive weight, and a zero weight does not by itself flag.
- `metadata` : a [`PartitionInfo`](@ref).

Array-wide globals live on the root `metadata::UVMetadata`
(`array_obs::ObsArrayMetadata`).
"""
mutable struct UVSet{TM <: UVMetadata} <: DimensionalData.AbstractDimTree
    data::DimensionalData.DataDict
    dims::Tuple
    refdims::Tuple
    layerdims::DimensionalData.TupleDict
    layermetadata::DimensionalData.DataDict
    metadata::TM
    branches::DimensionalData.TreeDict
    tree::Union{Nothing, DimensionalData.AbstractDimTree}
end

# Convenience kwarg constructor mirroring DD's DimTree. `metadata` is
# required since UVSet's whole point is a typed root-level metadata bundle.
function UVSet(;
        metadata::UVMetadata,
        data = DimensionalData.DataDict(),
        dims = (),
        refdims = (),
        layerdims = DimensionalData.TupleDict(),
        layermetadata = DimensionalData.DataDict(),
        branches = DimensionalData.TreeDict(),
        tree = nothing,
    )
    return UVSet(data, dims, refdims, layerdims, layermetadata, metadata, branches, tree)
end

"""
    UVSet(flat::NamedTuple)

Construct a `UVSet` from a flat record-layout `NamedTuple` produced by
`_load_uvfits_flat`. The named tuple must carry the per-record arrays
(`vis`, `weights`, `uvw`, `obs_time`, `record_scan_name`,
`record_spw_index`, `baselines`, `date_param`, `extra_columns`), the
observation-level globals (`antennas`, `array_config`, `array_obs`,
`freq_setups`), and per-source fields (`source_name`, `ra`, `dec`).
"""
function UVSet(flat::NamedTuple)
    src_key = sanitize_source(flat.source_name)
    base_name = haskey(flat, :basename) ? flat.basename : "uvfits"

    # `flat.antenna_tables`: Vector{AntennaTable} indexed by subarray slot.
    # Single-subarray flat tuples may pass `antennas = <single>`; promote it
    # to a 1-element vector so downstream uniformly indexes by subarray.
    antenna_tables = if haskey(flat, :antenna_tables)
        flat.antenna_tables
    elseif haskey(flat, :antennas)
        [flat.antennas]
    else
        error("UVSet(flat): missing antenna_tables / antennas")
    end
    n_sub = length(antenna_tables)
    record_sub_idx = if haskey(flat, :record_subarray_index)
        Int.(flat.record_subarray_index)
    else
        ones(Int, length(flat.record_scan_name))
    end

    # Each unique (scan_name, spw_index, subarray_index) becomes one MSv4
    # partition leaf, in first-seen order. Mirrors xradio's MSv2→MSv4
    # partitioning rule: DDI is always a partition axis; sub-array
    # participation splits leaves further when it's non-uniform.
    seen = Set{Tuple{String, Int, Int}}()
    ordered = Tuple{String, Int, Int}[]
    grouped_int_inds = Dict{Tuple{String, Int, Int}, Vector{Int}}()
    for i in eachindex(flat.record_scan_name)
        triple = (
            String(flat.record_scan_name[i]),
            Int(flat.record_spw_index[i]),
            record_sub_idx[i],
        )
        if !(triple in seen)
            push!(ordered, triple)
            push!(seen, triple)
            grouped_int_inds[triple] = Int[]
        end
        push!(grouped_int_inds[triple], i)
    end
    isempty(ordered) && error("UVSet(flat): no records in input")

    branches = DimensionalData.TreeDict()
    for (lbl, spw, sub) in ordered
        leaf, info = _extract_scan_leaf(
            flat, grouped_int_inds[(lbl, spw, sub)], lbl, spw, sub, antenna_tables;
            source_key = src_key, basename = base_name, n_sub = n_sub,
        )
        key = partition_key(info)
        if haskey(branches, key)
            error("UVSet: duplicate partition key $(key)")
        end
        branches[key] = leaf
    end

    metadata = UVMetadata(flat.array_obs)
    return UVSet(; metadata = metadata, branches = branches)
end


# DD AbstractDimTree interface forwarding.
DimensionalData.data(u::UVSet) = getfield(u, :data)
DimensionalData.dims(u::UVSet) = getfield(u, :dims)
DimensionalData.refdims(u::UVSet) = getfield(u, :refdims)
DimensionalData.layerdims(u::UVSet) = getfield(u, :layerdims)
DimensionalData.layermetadata(u::UVSet) = getfield(u, :layermetadata)
DimensionalData.metadata(u::UVSet) = getfield(u, :metadata)
DimensionalData.branches(u::UVSet) = getfield(u, :branches)
DimensionalData.tree(u::UVSet) = getfield(u, :tree)
DimensionalData.basetypeof(::Type{<:UVSet}) = UVSet

function DimensionalData.rebuild(
        u::UVSet;
        data = DimensionalData.data(u),
        dims = DimensionalData.dims(u),
        refdims = DimensionalData.refdims(u),
        metadata = DimensionalData.metadata(u),
        layerdims = DimensionalData.layerdims(u),
        layermetadata = DimensionalData.layermetadata(u),
        tree = DimensionalData.tree(u),
        branches = DimensionalData.branches(u),
    )
    return UVSet(data, dims, refdims, layerdims, layermetadata, metadata, branches, tree)
end


# DD upstream defines `metadata(s::AbstractDimStack) = getfield(s, :metadata)`
# but no equivalent for `AbstractDimTree` — the generic fallback returns
# `NoMetadata()` even when the `metadata` field is populated. This is needed
# for our leaf `partition_info` to be read back.
# Mild piracy; remove when upstream lands the missing method.
DimensionalData.metadata(dt::DimensionalData.DimTree) = getfield(dt, :metadata)

# Time axis lookup. Values are Float64 seconds since `UVData.JD_UNIX_EPOCH`,
# matching MSv4's `time` coordinate in its default `unix` format. Magnitudes
# near the present are ~1.7e9, where a Float64 resolves ~0.24 µs; every rate
# and delay term works on `t − t0` about a segment-local origin, where the
# resolution is picoseconds.
function obs_time(part::DimensionalData.DimTree)
    return lookup(part[:vis], Ti)
end
