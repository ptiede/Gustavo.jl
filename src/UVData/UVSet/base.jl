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
- `metadata` : `PartitionInfo` `NamedTuple` (`source_name`, `source_key`,
  `field_name`, `scan_name`, `scan_idx`, `spw_name`, `intent`, `ra`, `dec`,
  `ddi`, `partition_name`, `baselines::BaselineIndex`, `record_order`,
  `date_param`, `extra_columns`).
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
(`vis`, `weights`, `uvw`, `obs_time`, `scan_idx`, `baselines`, `date_param`,
`extra_columns`), the observation-level globals (`scans`, `antennas`,
`array_config`, `array_obs`, `primary_cards`), and per-source fields
(`source_name`, `ra`, `dec`).
"""
function UVSet(flat::NamedTuple)
    nscan = length(flat.scans)
    nscan > 0 || error("UVSet(flat): no scans in input")

    src_key = sanitize_source(flat.source_name)
    base_name = haskey(flat, :basename) ? flat.basename : "uvfits"

    branches = DimensionalData.TreeDict()
    for s in 1:nscan
        leaf, info = _extract_scan_leaf(flat, s; source_key = src_key, basename = base_name)
        key = partition_key(src_key, info.scan_idx)
        # Collision guard for sanitized-key collisions across sources (single-
        # source UVData load can't collide; future merge_uvsets validates).
        if haskey(branches, key)
            error("UVSet: duplicate partition key $(key)")
        end
        branches[key] = leaf
    end

    metadata = UVMetadata(
        flat.scans, flat.antennas, flat.array_config,
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

nscans(uvset::UVSet) = length(DimensionalData.branches(uvset))

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
    scan_ids(uvset::UVSet, source::AbstractString) -> Vector{Int}

Scan indices belonging to `source`, in tree order.
"""
function scan_ids(uvset::UVSet, source::AbstractString)
    out = Int[]
    for (_, leaf) in DimensionalData.branches(uvset)
        m = DimensionalData.metadata(leaf)
        m.source_name == source && push!(out, m.scan_idx)
    end
    return out
end

"""
    Base.summary(uvset::UVSet)

Return a `Vector{NamedTuple}` with one row per leaf:
`(key, source_name, field_name, scan_name, scan_idx, spw_name, intent,
 ra, dec, ddi, shape)`. Mirrors xradio's `ProcessingSet.summary()`.
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
                scan_idx = m.scan_idx,
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

function Base.show(io::IO, ::MIME"text/plain", uvset::UVSet)
    root = DimensionalData.metadata(uvset)
    nant = length(root.antennas)
    nms = root.antennas.name
    arr = root.array_obs
    chan_freqs = arr.freq_setup.channel_freqs
    ref_freq = round(arr.freq_setup.ref_freq / 1.0e9; digits = 3)
    flo = round(minimum(chan_freqs) / 1.0e9; digits = 3)
    fhi = round(maximum(chan_freqs) / 1.0e9; digits = 3)
    n_part = length(DimensionalData.branches(uvset))
    src_list = sources(uvset)
    println(io, "UVSet")
    println(io, "  Telescope : $(arr.telescope)")
    println(io, "  Array     : $(array_name(root.antennas)) ($ref_freq GHz)")
    println(io, "  Sources   : $(length(src_list)) ($(join(src_list, ", ")))")
    println(io, "  Partitions: $(n_part)")
    println(io, "  IFs ($(length(chan_freqs))): $(flo)–$(fhi) GHz")
    print(io, "  Antennas ($(length(nms))): $(join(nms, ", "))")
    return io
end

Base.show(io::IO, uvset::UVSet) = Base.show(io, MIME"text/plain"(), uvset)


scan_time_centers(uvset::UVSet) =
    [(scan.lower + scan.upper) / 2 for scan in DimensionalData.metadata(uvset).scans]
band_center_frequency(uvset::UVSet) = (
    first(DimensionalData.metadata(uvset).array_obs.freq_setup.channel_freqs) +
        last(DimensionalData.metadata(uvset).array_obs.freq_setup.channel_freqs)
) / 2
centered_channel_freqs(uvset::UVSet) =
    DimensionalData.metadata(uvset).array_obs.freq_setup.channel_freqs .- band_center_frequency(uvset)

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
nchannels(uvset::UVSet) = length(DimensionalData.metadata(uvset).array_obs.freq_setup.channel_freqs)
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


scan_idx(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).scan_idx
scan_id(part::DimensionalData.AbstractDimTree) = scan_idx(part)
baselines(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).baselines
record_order(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).record_order
date_param(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).date_param
extra_columns(part::DimensionalData.AbstractDimTree) = DimensionalData.metadata(part).extra_columns

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





