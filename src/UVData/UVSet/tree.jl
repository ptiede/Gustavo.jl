function _build_leaf(
        vis::AbstractDimArray, weights::AbstractDimArray,
        uvw::AbstractDimArray, flag::Union{Nothing, AbstractDimArray} = nothing;
        partition_info::PartitionInfo,
    )
    flag_layer = flag === nothing ? _derive_flag(weights) : flag
    vis_dims = dims(vis)
    uvw_dims = dims(uvw)
    vis_names = map(DimensionalData.name, vis_dims)
    uvw_names = map(DimensionalData.name, uvw_dims)
    nm = DimensionalData.Lookups.NoMetadata()
    data_dict = DimensionalData.DataDict(
        :vis => parent(vis),
        :weights => parent(weights),
        :uvw => parent(uvw),
        :flag => parent(flag_layer),
    )
    layerdims = DimensionalData.TupleDict(
        :vis => vis_names,
        :weights => vis_names,
        :uvw => uvw_names,
        :flag => vis_names,
    )
    layermetadata = DimensionalData.DataDict(
        :vis => nm, :weights => nm, :uvw => nm, :flag => nm,
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
        flat::NamedTuple, scan_label::AbstractString,
        spw_index::Integer, subarray_index::Integer,
        antenna_tables::AbstractVector, record_sub_idx::AbstractVector{<:Integer};
        source_key::Symbol, basename::AbstractString, n_sub::Integer = 1,
    )
    int_inds = findall(
        i -> flat.record_scan_name[i] == scan_label &&
            Int(flat.record_spw_index[i]) == Int(spw_index) &&
            record_sub_idx[i] == Int(subarray_index),
        eachindex(flat.record_scan_name),
    )

    leaf_freq_setup = flat.freq_setups[Int(spw_index)]
    leaf_antennas = antenna_tables[Int(subarray_index)]
    bl_pairs_per_record = flat.baselines.pairs_per_record[int_inds]
    obs_times_per_int = flat.obs_time[int_inds]

    bl_pairs_scan = sort(unique(bl_pairs_per_record))
    bl_lookup_scan = Dict(p => i for (i, p) in enumerate(bl_pairs_scan))
    src_ant1 = Dict(p => n for (p, n) in zip(flat.baselines.pairs, flat.baselines.ant1_names))
    src_ant2 = Dict(p => n for (p, n) in zip(flat.baselines.pairs, flat.baselines.ant2_names))
    bl_ant1_scan = [src_ant1[p] for p in bl_pairs_scan]
    bl_ant2_scan = [src_ant2[p] for p in bl_pairs_scan]
    bl_labels_scan = string.(bl_ant1_scan, "-", bl_ant2_scan)
    baselines_scan = BaselineIndex(
        bl_pairs_per_record, bl_pairs_scan, bl_lookup_scan,
        bl_labels_scan, bl_ant1_scan, bl_ant2_scan,
    )

    unique_times = sort(unique(obs_times_per_int))
    time_lookup = Dict(t => i for (i, t) in enumerate(unique_times))
    nti = length(unique_times)
    nbl = length(bl_pairs_scan)
    vis_flat = parent(flat.vis)
    weights_flat = parent(flat.weights)
    uvw_flat = parent(flat.uvw)
    npol = size(vis_flat, 2)
    nchan = size(vis_flat, 3)

    # Memory layout: frequency varies fastest, pol slowest. Order is
    # (Frequency, Ti, Baseline, Pol) for vis/weights/flag and
    # (UVW, Ti, Baseline) for uvw — matches xradio MSv4 frequency-fastest
    # convention and gives stride-1 channel access in bandpass loops.
    vis_dense = fill(
        complex(eltype(real(eltype(vis_flat)))(NaN), eltype(real(eltype(vis_flat)))(NaN)),
        nchan, nti, nbl, npol,
    )
    weights_dense = zeros(eltype(weights_flat), nchan, nti, nbl, npol)
    uvw_dense = fill(eltype(uvw_flat)(NaN), nti, nbl, 3)

    record_order = Vector{Tuple{Int, Int}}(undef, length(int_inds))
    for (rec_i, int_i) in enumerate(int_inds)
        ti = time_lookup[obs_times_per_int[rec_i]]
        bi = bl_lookup_scan[bl_pairs_per_record[rec_i]]
        record_order[rec_i] = (ti, bi)
        # flat layout is (Integration, Pol, Frequency); permute into
        # (Frequency, Ti, Baseline, Pol).
        for p in 1:npol, c in 1:nchan
            vis_dense[c, ti, bi, p] = vis_flat[int_i, p, c]
            weights_dense[c, ti, bi, p] = weights_flat[int_i, p, c]
        end
        # uvw layout: (Ti, Baseline, UVW) — easy slicing on Ti/Baseline.
        for k in 1:3
            uvw_dense[ti, bi, k] = uvw_flat[int_i, k]
        end
    end

    pol_labels = collect(lookup(flat.vis, Pol))
    vis_part = DimArray(
        vis_dense,
        (
            Frequency(channel_freqs(leaf_freq_setup)), Ti(unique_times),
            Baseline(baselines_scan.labels), Pol(pol_labels),
        ),
    )
    weights_part = DimArray(weights_dense, dims(vis_part))
    uvw_part = DimArray(
        uvw_dense,
        (Ti(unique_times), Baseline(baselines_scan.labels), UVW(["U", "V", "W"])),
    )

    extras_part = NamedTuple{keys(flat.extra_columns)}(
        ntuple(i -> flat.extra_columns[i][int_inds], length(flat.extra_columns))
    )

    # Single-subarray files keep an empty subarray_name (axis omitted from
    # the partition key); multi-subarray files render `sub_<n>`.
    sub_name = n_sub > 1 ? "sub_$(Int(subarray_index) - 1)" : ""
    info = PartitionInfo(;
        source_name = flat.source_name,
        source_key = source_key,
        scan_name = String(scan_label),
        ra = flat.ra, dec = flat.dec,
        antennas = leaf_antennas,
        baselines = baselines_scan,
        record_order = record_order,
        extra_columns = extras_part,
        freq_setup = leaf_freq_setup,
        spw_name = "spw_$(Int(spw_index) - 1)",
        subarray_name = sub_name,
        ddi = Int(spw_index) - 1,
        basename = basename,
    )
    leaf = _build_leaf(vis_part, weights_part, uvw_part; partition_info = info)
    return leaf, info
end
