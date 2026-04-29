function _build_leaf(
        vis::AbstractDimArray, weights::AbstractDimArray,
        uvw::AbstractDimArray, flag::Union{Nothing, AbstractDimArray} = nothing;
        partition_info::PartitionInfo,
        scan_name::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    )
    flag_layer = flag === nothing ? _derive_flag(weights) : flag
    vis_dims = dims(vis)
    uvw_dims = dims(uvw)
    vis_names = map(DimensionalData.name, vis_dims)
    uvw_names = map(DimensionalData.name, uvw_dims)
    # ScanArray (xradio schema.py:779): per-Ti label, default to the cached
    # primary `scan_name` repeated along the leaf's time axis.
    ti_dim = vis_dims[findfirst(d -> DimensionalData.name(d) === :Ti, vis_dims)]
    nti = length(lookup(ti_dim))
    scan_name_vec = scan_name === nothing ?
        fill(partition_info.scan_name, nti) : String.(collect(scan_name))
    length(scan_name_vec) == nti ||
        error("scan_name length $(length(scan_name_vec)) does not match Ti axis length $(nti)")
    nm = DimensionalData.Lookups.NoMetadata()
    data_dict = DimensionalData.DataDict(
        :vis => parent(vis),
        :weights => parent(weights),
        :uvw => parent(uvw),
        :flag => parent(flag_layer),
        :scan_name => scan_name_vec,
    )
    layerdims = DimensionalData.TupleDict(
        :vis => vis_names,
        :weights => vis_names,
        :uvw => uvw_names,
        :flag => vis_names,
        :scan_name => (DimensionalData.name(ti_dim),),
    )
    layermetadata = DimensionalData.DataDict(
        :vis => nm, :weights => nm, :uvw => nm, :flag => nm,
        :scan_name => nm,
    )
    all_dims = (vis_dims..., DimensionalData.otherdims(uvw_dims, vis_dims)...)
    return DimensionalData.DimTree(;
        data = data_dict,
        dims = all_dims,
        layerdims = layerdims,
        layermetadata = layermetadata,
        metadata = partition_info,
    )
end

function _extract_scan_leaf(
        flat::NamedTuple, scan_label::AbstractString, spw_index::Integer;
        source_key::Symbol, basename::AbstractString,
    )
    int_inds = findall(
        i -> flat.record_scan_name[i] == scan_label &&
            Int(flat.record_spw_index[i]) == Int(spw_index),
        eachindex(flat.record_scan_name),
    )

    leaf_freq_setup = flat.freq_setups[Int(spw_index)]
    bl_codes_scan = flat.baselines.codes[int_inds]
    obs_times_per_int = flat.obs_time[int_inds]

    unique_codes_scan = sort(unique(bl_codes_scan))
    bl_lookup_scan = Dict(c => i for (i, c) in enumerate(unique_codes_scan))
    src_pair = Dict(c => p for (c, p) in zip(flat.baselines.unique_codes, flat.baselines.pairs))
    src_ant1 = Dict(c => n for (c, n) in zip(flat.baselines.unique_codes, flat.baselines.ant1_names))
    src_ant2 = Dict(c => n for (c, n) in zip(flat.baselines.unique_codes, flat.baselines.ant2_names))
    bl_pairs_scan = [src_pair[c] for c in unique_codes_scan]
    bl_ant1_scan = [src_ant1[c] for c in unique_codes_scan]
    bl_ant2_scan = [src_ant2[c] for c in unique_codes_scan]
    bl_labels_scan = string.(bl_ant1_scan, "-", bl_ant2_scan)
    baselines_scan = BaselineIndex(
        bl_codes_scan, bl_pairs_scan, bl_lookup_scan, unique_codes_scan,
        bl_labels_scan, bl_ant1_scan, bl_ant2_scan,
    )

    unique_times = sort(unique(obs_times_per_int))
    time_lookup = Dict(t => i for (i, t) in enumerate(unique_times))
    nti = length(unique_times)
    nbl = length(unique_codes_scan)
    vis_flat = parent(flat.vis)
    weights_flat = parent(flat.weights)
    uvw_flat = parent(flat.uvw)
    npol = size(vis_flat, 2)
    nchan = size(vis_flat, 3)

    vis_dense = fill(
        complex(eltype(real(eltype(vis_flat)))(NaN), eltype(real(eltype(vis_flat)))(NaN)),
        nti, nbl, npol, nchan,
    )
    weights_dense = zeros(eltype(weights_flat), nti, nbl, npol, nchan)
    uvw_dense = fill(eltype(uvw_flat)(NaN), nti, nbl, 3)

    record_order = Vector{Tuple{Int, Int}}(undef, length(int_inds))
    for (rec_i, int_i) in enumerate(int_inds)
        ti = time_lookup[obs_times_per_int[rec_i]]
        bi = bl_lookup_scan[bl_codes_scan[rec_i]]
        record_order[rec_i] = (ti, bi)
        vis_dense[ti, bi, :, :] .= vis_flat[int_i, :, :]
        weights_dense[ti, bi, :, :] .= weights_flat[int_i, :, :]
        uvw_dense[ti, bi, :] .= uvw_flat[int_i, :]
    end

    pol_labels = collect(lookup(flat.vis, Pol))
    vis_part = DimArray(
        vis_dense,
        (
            Ti(unique_times), Baseline(baselines_scan.labels),
            Pol(pol_labels), IF(channel_freqs(leaf_freq_setup)),
        ),
    )
    weights_part = DimArray(weights_dense, dims(vis_part))
    uvw_part = DimArray(
        uvw_dense,
        (Ti(unique_times), Baseline(baselines_scan.labels), UVW(["U", "V", "W"])),
    )

    date_param_part = flat.date_param[int_inds, :]
    extras_part = NamedTuple{keys(flat.extra_columns)}(
        ntuple(i -> flat.extra_columns[i][int_inds], length(flat.extra_columns))
    )

    info = PartitionInfo(;
        source_name = flat.source_name,
        source_key = source_key,
        scan_name = String(scan_label),
        ra = flat.ra, dec = flat.dec,
        baselines = baselines_scan,
        record_order = record_order,
        date_param = date_param_part,
        extra_columns = extras_part,
        freq_setup = leaf_freq_setup,
        spw_name = "spw_$(Int(spw_index) - 1)",
        ddi = Int(spw_index) - 1,
        basename = basename,
    )
    leaf = _build_leaf(vis_part, weights_part, uvw_part; partition_info = info)
    return leaf, info
end
