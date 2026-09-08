# The containers that carry a leaf's `PartitionInfo`: the leaf tree itself and
# any layer selection off it (`leaf[(:vis, :weights)]`), which keeps the
# metadata. Both answer the identity accessors defined throughout this file.
const PartitionedData = Union{DimensionalData.AbstractDimTree, DimensionalData.AbstractDimStack}

"""
    UVSet <: DimensionalData.AbstractDimTree

Top-level container mirroring xradio's MSv4 `ProcessingSet`: a flat
`OrderedDict` of MSv4-shaped partition leaves under `branches`, keyed by
sanitized `:<source>_scan_<n>` Symbols (e.g. `:src_3C273_scan_1`). Multi-source
data shows up as sibling flat partitions, never as a nested Source dim.

The struct subtypes `AbstractDimTree` so all DD machinery (selectors,
`length`/`iterate`, `setindex!`, `rebuild`, branch-name `getproperty`)
works without bespoke overloads. Per-leaf data lives on each branch
(a plain `DimTree`) carrying:

- `data`     : `:vis`, `:weights`, `:uvw` `DimArray` layers. A cell is
  flagged iff its weight is `≤ 0`, so no separate flag layer is stored.
- `metadata` : `PartitionInfo` struct (`source_name`, `source_key`,
  `field_name`, `scan_name`, `scan_intents`, `sub_scan_name`, `spw_name`,
  `intent`, `ra`, `dec`, `ddi`, `partition_name`, `baselines::BaselineIndex`,
  `record_order`, `date_param`, `extra_columns`, `freq_setup`).
- `branches` : empty for UVData read-in; reserved for future MSv4
  sub-tables (`antenna_xds`, `field_and_source_xds`, …).

Array-wide globals live on the root `metadata::UVMetadata` (`antennas`,
`array_config`, `array_obs::ObsArrayMetadata`). FITS primary-HDU cards
(write-back state) live in a FITS-extension-owned `WeakKeyDict`, not on
`UVMetadata`.

Discovery helpers: `summary(uvset)`, `sources(uvset)`, `scan_ids(uvset, src)`,
`partitions(uvset)` (tab-completable accessor), `select_source` /
`select_scan` / `select_partition`. Tree-walking transforms: `apply(f, uvset)`,
`leaves(uvset)`. Built-in reducers `TimeAverage`, `BandpassCorrection`
implement the `apply` callable signature.
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
The flat tuple may also carry `primary_cards` (consumed by the FITS
extension's `load_uvfits` for write-back; ignored here).
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

"""
    _propagate_extension_state!(new::UVSet, old::UVSet)

Hook for format-extension shadow state (e.g. FITS primary-HDU cards) to
follow a `UVSet` through `rebuild` / `select_*` / `merge_uvsets`. The
default is a no-op; the FITS extension overloads it to copy registered
primary cards from `old` to `new`.
"""
_propagate_extension_state!(new, old) = new

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
    new = UVSet(data, dims, refdims, layerdims, layermetadata, metadata, branches, tree)
    return _propagate_extension_state!(new, u)
end

"""
    nscans(uvset::UVSet) -> Int

Number of unique scan labels across leaves (the actual scan count).
Differs from `length(branches(uvset))` when sub-arrays or multi-source
partitions cause multiple leaves to share a `scan_name`.
"""
function nscans(uvset::UVSet)
    seen = Set{String}()
    for (_, leaf) in DimensionalData.branches(uvset)
        push!(seen, DimensionalData.metadata(leaf).scan_name)
    end
    return length(seen)
end

"""
    leaves(uvset::UVSet)

Iterator over `(partition_key::Symbol, leaf::DimTree)` pairs for every leaf
in the tree, in branch insertion order.
"""
leaves(uvset::UVSet) = pairs(DimensionalData.branches(uvset))

"""
    sources(uvset::UVSet) -> Vector{String}

Unique source names appearing across the leaves of `uvset`, in the order
they first appear (matches xradio's stable summary ordering).
"""
function sources(uvset::UVSet)
    seen = String[]
    for (_, leaf) in DimensionalData.branches(uvset)
        sn = DimensionalData.metadata(leaf).source_name
        sn in seen || push!(seen, sn)
    end
    return seen
end

"""
    scan_ids(uvset::UVSet, source::AbstractString) -> Vector{String}

Scan labels (primary `scan_name`s) belonging to `source`, in tree order.
"""
function scan_ids(uvset::UVSet, source::AbstractString)
    out = String[]
    for (_, leaf) in DimensionalData.branches(uvset)
        m = DimensionalData.metadata(leaf)
        m.source_name == source && push!(out, m.scan_name)
    end
    return out
end

"""
    Base.summary(uvset::UVSet)

Return a `Vector{NamedTuple}` with one row per leaf:
`(key, source_name, field_name, scan_name, scan_intents, sub_scan_name,
spw_name, intent, ra, dec, ddi, shape)`. Mirrors xradio's
`ProcessingSet._summary()` aggregation over leaves.
"""
function Base.summary(uvset::UVSet)
    rows = NamedTuple[]

    for (k, leaf) in DimensionalData.branches(uvset)
        m = DimensionalData.metadata(leaf)
        vis = leaf[:vis]
        push!(
            rows, (;
                key = k,
                source_name = m.source_name,
                field_name = m.field_name,
                scan_name = m.scan_name,
                scan_intents = m.scan_intents,
                sub_scan_name = m.sub_scan_name,
                spw_name = m.spw_name,
                intent = m.intent,
                ra = m.ra,
                dec = m.dec,
                ddi = m.ddi,
                shape = size(vis),
            )
        )
    end
    return rows
end

"""
    union_frequency_axis(uvset::UVSet) -> Vector{FrequencySetup}

Vector of `FrequencySetup`s spanning every leaf, deduplicated by `==`/`hash`,
in first-seen order. Mirrors xradio's `ProcessingSet.get_freq_axis()`.
Used by the FITS writer to assemble the union FQ table and by the future
per-SPW solver dispatch.
"""
function union_frequency_axis(uvset::UVSet)
    out = FrequencySetup[]
    seen = Set{FrequencySetup}()
    for (_, leaf) in DimensionalData.branches(uvset)
        fs = DimensionalData.metadata(leaf).freq_setup
        if !(fs in seen)
            push!(out, fs)
            push!(seen, fs)
        end
    end
    return out
end

"""
    freq_setup(uvset::UVSet) -> FrequencySetup

Single-SPW shorthand: returns the unique `FrequencySetup` if every leaf
shares one, otherwise throws `ArgumentError`. Multi-SPW callers should
use `union_frequency_axis(uvset)` or read each leaf's `freq_setup`
individually.
"""
function freq_setup(uvset::UVSet)
    setups = union_frequency_axis(uvset)
    n = length(setups)
    n == 1 && return setups[1]
    n == 0 && throw(ArgumentError("UVSet has no leaves; no frequency setup"))
    throw(
        ArgumentError(
            "UVSet has $(n) distinct frequency setups; " *
                "use union_frequency_axis(uvset) or freq_setup(leaf)",
        )
    )
end

function Base.show(io::IO, ::MIME"text/plain", uvset::UVSet)
    root = DimensionalData.metadata(uvset)
    bs = DimensionalData.branches(uvset)
    # Walk leaves for the antenna summary so multi-subarray UVSets render
    # the union (or surface a conflict). Falls back gracefully on empty sets.
    ants = isempty(bs) ? nothing : union_antennas(uvset)
    nms = ants === nothing ? String[] : ants.name
    arr_name = ants === nothing ? "" : array_name(ants)
    arr = root.array_obs
    setups = union_frequency_axis(uvset)
    chan_freqs = isempty(setups) ?
        Float64[] :
        reduce(vcat, channel_freqs(fs) for fs in setups)
    ref_freq_ghz = isempty(setups) ?
        NaN : round(ref_freq(first(setups)) / 1.0e9; digits = 3)
    flo = isempty(chan_freqs) ?
        NaN : round(minimum(chan_freqs) / 1.0e9; digits = 3)
    fhi = isempty(chan_freqs) ?
        NaN : round(maximum(chan_freqs) / 1.0e9; digits = 3)
    n_part = length(bs)
    src_list = sources(uvset)
    println(io, "UVSet")
    println(io, "  Telescope : $(arr.telescope)")
    println(io, "  Array     : $(arr_name) ($ref_freq_ghz GHz)")
    println(io, "  Sources   : $(length(src_list)) ($(join(src_list, ", ")))")
    println(io, "  Partitions: $(n_part)")
    println(io, "  Spectral  : $(length(setups)) setup(s), $(length(chan_freqs)) channels total: $(flo)–$(fhi) GHz")
    print(io, "  Antennas ($(length(nms))): $(join(nms, ", "))")
    return io
end

Base.show(io::IO, uvset::UVSet) = Base.show(io, MIME"text/plain"(), uvset)


"""
    scan_time_centers(uvset::UVSet) -> Vector{Float64}

Mid-point of each leaf's `Ti` axis, in branch insertion order. Replaces
the old root-level scans table indirection — mirrors xradio's per-partition
time coord.
"""
function scan_time_centers(uvset::UVSet)
    out = Float64[]
    for (_, leaf) in DimensionalData.branches(uvset)
        lo, hi = scan_window(leaf)
        push!(out, (lo + hi) / 2)
    end
    return out
end
spw_center_frequency(uvset::UVSet) = spw_center_frequency(freq_setup(uvset))
centered_channel_freqs(uvset::UVSet) = centered_channel_freqs(freq_setup(uvset))

function baseline_sites(uvset::UVSet, bl::Tuple{String, String})
    ants = union_antennas(uvset)
    a_idx = findfirst(==(bl[1]), ants.name)
    b_idx = findfirst(==(bl[2]), ants.name)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl")
    return a_idx, b_idx
end

function baseline_number(leaf::DimensionalData.AbstractDimTree, bl::Tuple{String, String})
    bi = findfirst(==(string(bl[1], "-", bl[2])), baselines(leaf).labels)
    isnothing(bi) && error("Baseline $bl not in partition")
    return bi
end

"""
    antennas(leaf::AbstractDimTree) -> AntennaTable

Per-leaf antenna table.
"""
antennas(leaf::PartitionedData) =
    DimensionalData.metadata(leaf).antennas

"""
    union_antennas(uvset::UVSet) -> AntennaTable

Walk leaves and union participating antennas by name. Errors if the
same antenna name has different metadata (mount, station_xyz,
nominal_basis, response, pol_angles) across leaves — a multi-track
observation that should be split via `select_*` and processed per-SPW.
"""
function union_antennas(uvset::UVSet)
    bs = DimensionalData.branches(uvset)
    isempty(bs) && error("union_antennas: UVSet has no leaves")
    leaves_v = collect(values(bs))
    template = DimensionalData.metadata(first(leaves_v)).antennas
    # Fast path: every leaf points to the same AntennaTable instance (the
    # common case for single-subarray observations).
    same_ref = all(DimensionalData.metadata(l).antennas === template for l in leaves_v)
    same_ref && return template
    seen = Set{String}()
    rows = eltype(getfield(template, :antennas))[]
    for leaf in leaves_v
        sa = getfield(DimensionalData.metadata(leaf).antennas, :antennas)
        for ant in sa
            if !(ant.name in seen)
                push!(rows, ant)
                push!(seen, ant.name)
            else
                # Re-find the existing row by name and verify equality.
                idx = findfirst(r -> r.name == ant.name, rows)
                rows[idx] == ant || error(
                    "union_antennas: antenna '$(ant.name)' has " *
                        "inconsistent metadata across leaves; split via " *
                        "select_* and process per-SPW.",
                )
            end
        end
    end
    return AntennaTable(
        StructArray(rows), array_xyz(template), array_name(template), extras(template),
    )
end

"""
    unify_antennas(uvset::UVSet) -> UVSet

Put every leaf on one antenna table — [`union_antennas`](@ref)'s — re-indexing
each leaf's `BaselineIndex` into it.

A leaf's `(a, b)` pairs index that leaf's OWN table, so a sub-array leaf listing
only the stations that observed it numbers them differently from a fuller leaf:
where six antennas observed, index 3 may be `KT` while the full array's index 3
is `GL`. Anything reading pairs against a single table — which is every solver,
since a solve has one station axis — would then attribute one station's data to
another. This makes the indices mean the same thing everywhere, so sets whose
leaves saw different sub-arrays can be solved together.

Antennas are matched by NAME (`union_antennas` refuses inconsistent metadata for
a shared name). Leaves already on the union table are returned untouched, so a
single-sub-array set costs nothing and stays lazy.
"""
function unify_antennas(uvset::UVSet)
    table = union_antennas(uvset)
    names = collect(String.(table.name))
    all(
        collect(String.(DimensionalData.metadata(l).antennas.name)) == names
            for l in values(DimensionalData.branches(uvset))
    ) && return uvset
    slot = Dict(n => i for (i, n) in pairs(names))
    return apply(uvset) do leaf, info, root
        local_names = collect(String.(info.antennas.name))
        local_names == names && return leaf
        m = [slot[n] for n in local_names]
        remap(ps) = [(m[a], m[b]) for (a, b) in ps]
        b = info.baselines
        # `record_order` holds (time, baseline-slot) pairs, and slot order is
        # preserved here, so it needs no remapping.
        newb = BaselineIndex(remap(b.pairs_per_record), remap(b.pairs); antenna_names = names)
        return DimensionalData.rebuild(
            leaf; metadata = update(info; antennas = table, baselines = newb),
        )
    end
end

"""
    union_pol_products(uvset::UVSet) -> Vector{String}

Pol product set shared across leaves. Errors if leaves disagree —
mirrors `union_antennas` for the polarization axis.
"""
function union_pol_products(uvset::UVSet)
    bs = DimensionalData.branches(uvset)
    isempty(bs) && error("union_pol_products: UVSet has no leaves")
    leaves_v = collect(values(bs))
    first_pp = pol_products(first(leaves_v))
    for leaf in leaves_v
        Set(pol_products(leaf)) == Set(first_pp) ||
            error(
            "union_pol_products: leaves have different pol product sets; " *
                "got $first_pp vs $(pol_products(leaf)).",
        )
    end
    return first_pp
end

antenna_names(uvset::UVSet) = union_antennas(uvset).name
nchannels(uvset::UVSet) = nchannels(freq_setup(uvset))
npols(uvset::UVSet) = length(pol_products(uvset))
nbaselines(uvset::UVSet) = length(
    unique(
        reduce(
            vcat,
            [collect(baselines(leaf).pairs) for (_, leaf) in DimensionalData.branches(uvset)];
            init = Tuple{Int, Int}[],
        )
    )
)
nintegrations(uvset::UVSet) = sum(
    length(record_order(leaf)) for (_, leaf) in DimensionalData.branches(uvset);
    init = 0,
)


# DD upstream defines `metadata(s::AbstractDimStack) = getfield(s, :metadata)`
# but no equivalent for `AbstractDimTree` — the generic fallback returns
# `NoMetadata()` even when the `metadata` field is populated. This is needed
# for our leaf `partition_info` to flow through `apply` / `rebuild`.
# Mild piracy; remove when upstream lands the missing method.
DimensionalData.metadata(dt::DimensionalData.DimTree) = getfield(dt, :metadata)


freq_setup(part::PartitionedData) = DimensionalData.metadata(part).freq_setup
baselines(part::PartitionedData) = DimensionalData.metadata(part).baselines
record_order(part::PartitionedData) = DimensionalData.metadata(part).record_order
extra_columns(part::PartitionedData) = DimensionalData.metadata(part).extra_columns

"""
    source_name(part) -> String

Name of the source the leaf (or a layer selection off it) observes.
"""
source_name(part::PartitionedData) = DimensionalData.metadata(part).source_name

# Each leaf maps to exactly one xradio MSv4 scan, so the scan label is a
# scalar field on `PartitionInfo`. `scan_name` and `primary_scan_name`
# return that String — twin accessors retained for callers that previously
# read the per-Ti vector form.
scan_name(part::PartitionedData) =
    DimensionalData.metadata(part).scan_name
primary_scan_name(part::PartitionedData) =
    DimensionalData.metadata(part).scan_name
scan_intents(part::PartitionedData) =
    DimensionalData.metadata(part).scan_intents
sub_scan_name(part::PartitionedData) =
    DimensionalData.metadata(part).sub_scan_name

"""
    scan_window(leaf) -> Tuple{Float64, Float64}

`(lower, upper)` time bounds for a leaf, derived from the extrema of its
`Ti` lookup. Empty leaves return `(NaN, NaN)`.
"""
function scan_window(part::DimensionalData.AbstractDimTree)
    t = obs_time(part)
    isempty(t) && return (NaN, NaN)
    return extrema(t)
end

"""
    participating_antennas(leaf) -> Vector{String}

Antenna names that appear in the leaf's baselines (sorted union of
`ant1_names ∪ ant2_names`). The root antenna table is the union/superset;
this accessor surfaces the actual sub-array participation per leaf.
"""
function participating_antennas(part::DimensionalData.AbstractDimTree)
    bls = baselines(part)
    return sort!(collect(Set{String}(vcat(bls.ant1_names, bls.ant2_names))))
end

# Time axis lookup. Values are Float64 fractional hours since RDATE 00:00 UTC
# (the AIPS RDATE card on the AN HDU). For a single-night track, magnitudes
# are bounded by ~24; multi-night tracks accumulate as
# 24·days_offset + hour_within_day.
function obs_time(part::DimensionalData.AbstractDimTree)
    return lookup(part[:vis], Ti)
end

"""
    rebuild_visibilities(part::AbstractDimTree, vis, weights) -> DimTree

Return a new leaf sharing `part`'s `uvw` layer and metadata, with
`vis`/`weights` swapped in. A cell is flagged iff its weight is `≤ 0`.
"""
function rebuild_visibilities(part::DimensionalData.AbstractDimTree, vis = part[:vis], weights = part[:weights], uvw = part[:uvw])
    vis_l = _rewrap_like(vis, part[:vis])
    w_l = _rewrap_like(weights, part[:weights])
    uvw_l = _rewrap_like(uvw, part[:uvw])
    return _build_leaf(
        vis_l, w_l, uvw_l;
        partition_info = DimensionalData.metadata(part),
    )
end
