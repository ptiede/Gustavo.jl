"""
    UVSet <: DimensionalData.AbstractDimTree

Top-level container mirroring xradio's MSv4 `ProcessingSet`: a flat
`OrderedDict` of MSv4-shaped partition leaves under `branches`, keyed by
sanitized `:<source>_scan_<n>` Symbols (e.g. `:M3C273_scan_1`). Multi-source
data shows up as sibling flat partitions, never as a nested Source dim.

The struct subtypes `AbstractDimTree` so all DD machinery (selectors,
`length`/`iterate`, `setindex!`, `rebuild`, branch-name `getproperty`)
works without bespoke overloads. Per-leaf data lives on each branch
(a plain `DimTree`) carrying:

- `data`     : `:vis`, `:weights`, `:uvw`, `:flag` `DimArray` layers.
- `metadata` : `PartitionInfo` struct (`source_name`, `source_key`,
  `field_name`, `scan_name`, `scan_intents`, `sub_scan_name`, `spw_name`,
  `intent`, `ra`, `dec`, `ddi`, `partition_name`, `baselines::BaselineIndex`,
  `record_order`, `date_param`, `extra_columns`, `freq_setup`).
- `branches` : empty for UVData read-in; reserved for future MSv4
  sub-tables (`antenna_xds`, `field_and_source_xds`, …).

Array-wide globals live on the root `metadata::UVMetadata` (`scans`,
`antennas`, `array_config`, `array_obs::ObsArrayMetadata`,
`primary_cards`).

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
`record_freqid`, `baselines`, `date_param`, `extra_columns`), the
observation-level globals (`antennas`, `array_config`, `array_obs`,
`freq_setups`, `primary_cards`), and per-source fields
(`source_name`, `ra`, `dec`).
"""
function UVSet(flat::NamedTuple)
    src_key = sanitize_source(flat.source_name)
    base_name = haskey(flat, :basename) ? flat.basename : "uvfits"

    # Each unique record_scan_name becomes one MSv4 partition leaf, in
    # first-seen order. Mirrors xradio's MSv2→MSv4 partitioning rule for
    # `SCAN_NUMBER` (partition_queries.py:36).
    scan_labels = String[]
    seen = Set{String}()
    for lbl in flat.record_scan_name
        s = String(lbl)
        if !(s in seen)
            push!(scan_labels, s)
            push!(seen, s)
        end
    end
    isempty(scan_labels) && error("UVSet(flat): no records in input")

    branches = DimensionalData.TreeDict()
    for lbl in scan_labels
        leaf, info = _extract_scan_leaf(flat, lbl; source_key = src_key, basename = base_name)
        key = partition_key(src_key, info.scan_name; sub_scan_name = info.sub_scan_name)
        # Collision guard. Phase 2.5 (multi-AN-extver read) will populate
        # `sub_scan_name` when the same `(source, scan)` would otherwise collide.
        if haskey(branches, key)
            error("UVSet: duplicate partition key $(key)")
        end
        branches[key] = leaf
    end

    metadata = UVMetadata(
        flat.antennas, flat.array_config,
        flat.array_obs, flat.primary_cards,
    )
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
    nant = length(root.antennas)
    nms = root.antennas.name
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
    n_part = length(DimensionalData.branches(uvset))
    src_list = sources(uvset)
    println(io, "UVSet")
    println(io, "  Telescope : $(arr.telescope)")
    println(io, "  Array     : $(array_name(root.antennas)) ($ref_freq_ghz GHz)")
    println(io, "  Sources   : $(length(src_list)) ($(join(src_list, ", ")))")
    println(io, "  Partitions: $(n_part)")
    println(io, "  Spectral  : $(length(setups)) setup(s), $(length(chan_freqs)) IFs total: $(flo)–$(fhi) GHz")
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
band_center_frequency(uvset::UVSet) = band_center_frequency(freq_setup(uvset))
centered_channel_freqs(uvset::UVSet) = centered_channel_freqs(freq_setup(uvset))

function baseline_sites(uvset::UVSet, bl::Tuple{String, String})
    ants = DimensionalData.metadata(uvset).antennas
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

antenna_names(uvset::UVSet) = DimensionalData.metadata(uvset).antennas.name
nchannels(uvset::UVSet) = nchannels(freq_setup(uvset))
npols(uvset::UVSet) = length(pol_products(uvset))
nbaselines(uvset::UVSet) = length(
    unique(
        reduce(
            vcat,
            [collect(baselines(leaf).unique_codes) for (_, leaf) in DimensionalData.branches(uvset)];
            init = Int[],
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


freq_setup(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).freq_setup
baselines(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).baselines
record_order(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).record_order
date_param(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).date_param
extra_columns(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).extra_columns

# xradio's `ScanArray` (schema.py:779): the per-Ti scan_name vector lives as
# the `:scan_name` data layer, while `primary_scan_name` returns the cached
# scalar handle (typical case: one scan per leaf).
scan_name(part::DimensionalData.AbstractDimTree) = part[:scan_name]
primary_scan_name(part::DimensionalData.AbstractDimTree) =
    DimensionalData.metadata(part).scan_name
scan_intents(part::DimensionalData.AbstractDimTree) =
    DimensionalData.metadata(part).scan_intents
sub_scan_name(part::DimensionalData.AbstractDimTree) =
    DimensionalData.metadata(part).sub_scan_name

"""
    scan_window(leaf) -> Tuple{Float64, Float64}

`(lower, upper)` time bounds for a leaf, derived from the extrema of its
`Ti` lookup. Empty leaves return `(NaN, NaN)`.
"""
function scan_window(part::DimensionalData.AbstractDimTree)
    t = obs_time(part)
    isempty(t) && return (NaN, NaN)
    return (Float64(minimum(t)), Float64(maximum(t)))
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

# Time axis lookup. Leaves use `Ti`.
function obs_time(part::DimensionalData.AbstractDimTree)
    vis = part[:vis]
    return hasdim(vis, Ti) ? lookup(vis, Ti) : lookup(vis, Integration)
end

# `weights ≤ 0` carries the FITS flag convention. We derive a Bool layer at
# construction so `data.flag` is always available without recomputation.
_derive_flag(w::AbstractDimArray) = DimArray(parent(w) .<= 0, dims(w))
_derive_flag(w::AbstractArray) = w .<= 0

"""
    with_visibilities(part::AbstractDimTree, vis, weights) -> DimTree

Return a new leaf sharing `part`'s `uvw` layer and metadata, with
`vis`/`weights`/`flag` swapped in. `flag` is re-derived from `weights`.
"""
function with_visibilities(part::DimensionalData.AbstractDimTree, vis, weights)
    vis_l = _rewrap_like(vis, part[:vis])
    w_l = _rewrap_like(weights, part[:weights])
    return _build_leaf(
        vis_l, w_l, part[:uvw];
        partition_info = DimensionalData.metadata(part),
    )
end
