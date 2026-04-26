function wrap_baseline_array(A, data::UVData, bl::Tuple{String,String}; kind::Symbol, obs_inds=nothing)
    freqs = data.metadata.channel_freqs
    metadata = Dict(
        :baseline => join(bl, "-"),
        :Sites => bl,
        :kind => kind,
        :pol_codes => data.metadata.pol_codes,
        :pol_labels => data.metadata.pol_labels,
        :channel_freqs => data.metadata.channel_freqs,
        :band_center_frequency => band_center_frequency(data),
    )

    if ndims(A) == 2
        error("wrap_baseline_array expects rank-3 baseline slices")
    elseif ndims(A) == 3 && !isnothing(obs_inds)
        metadata[:obs_indices] = obs_inds
        metadata[:scan_indices] = data.scan_idx[obs_inds]
        return DimArray(A, (
            Ti(data.obs_time[obs_inds]),
            Pol(collect(data.metadata.pol_labels[1:size(A, 2)])),
            IF(freqs),
        ); metadata=metadata)
    elseif ndims(A) == 3
        return DimArray(A, (
            Scan(scan_time_centers(data)),
            Pol(collect(data.metadata.pol_labels[1:size(A, 2)])),
            IF(freqs),
        ); metadata=metadata)
    elseif ndims(A) == 4
        error("wrap_baseline_array expects a baseline-selected slice, not the full UV cube")
    else
        error("Unsupported baseline slice rank: $(ndims(A))")
    end
end


"""
    baseline_visibilities(data::UVData, bl::Tuple{String,String})

Return visibilities for a single baseline as a `DimArray`.

- For raw `UVData` loaded from uvfits this returns dimensions `(Ti, Pol, IF)`.
- For scan-averaged `UVData` this returns dimensions `(Scan, Pol, IF)`.
"""
function baseline_visibilities(data::UVData, bl::Tuple{String, String})
    if ndims(data.vis) == 3
        bi = baseline_number(data, bl)
        bl_code = data.baselines.unique_codes[bi]
        obs_inds = findall(==(bl_code), data.baselines.codes)
        return wrap_baseline_array(@view(data.vis[obs_inds, :, :]), data, bl; kind = :vis, obs_inds = obs_inds)
    elseif ndims(data.vis) == 4
        bi = baseline_number(data, bl)
        return wrap_baseline_array(@view(data.vis[:, bi, :, :]), data, bl; kind = :vis)
    else
        error("Unsupported visibility rank: $(ndims(data.vis))")
    end
end

"""
    baseline_weights(data::UVData, bl::Tuple{String,String})

Return weights for a single baseline as a `DimArray`.
The dimensional layout matches `baseline_visibilities`.
"""
function baseline_weights(data::UVData, bl::Tuple{String, String})
    if ndims(data.weights) == 3
        bi = baseline_number(data, bl)
        bl_code = data.baselines.unique_codes[bi]
        obs_inds = findall(==(bl_code), data.baselines.codes)
        return wrap_baseline_array(@view(data.weights[obs_inds, :, :]), data, bl; kind = :weights, obs_inds = obs_inds)
    elseif ndims(data.weights) == 4
        bi = baseline_number(data, bl)
        return wrap_baseline_array(@view(data.weights[:, bi, :, :]), data, bl; kind = :weights)
    else
        error("Unsupported weight rank: $(ndims(data.weights))")
    end
end

Base.getindex(data::UVData, bl::Tuple{String,String}) = baseline_visibilities(data, bl)

"""
    wrap_gain_solutions(gains, data::UVData; pol_keys=1:2)

Wrap the solved gain cube in a `DimArray` so scans, sites, polarisations, and IFs
carry labeled dimensions for interactive inspection.
"""
function wrap_gain_solutions(gains, data::UVData; pol_keys=1:2)
    size(gains, 1) == length(data.scans) || error("Gain scan axis does not match UVData scans")
    size(gains, 2) == length(data.antennas) || error("Gain antenna axis does not match UVData antennas")
    size(gains, 3) == length(pol_keys) || error("pol_keys length must match gain polarisation axis")
    size(gains, 4) == length(data.metadata.channel_freqs) || error("Gain channel axis does not match UVData channels")

    return DimArray(gains, (
        Scan(scan_time_centers(data)),
        Ant(data.antennas.name),
        Pol(collect(pol_keys)),
        IF(data.metadata.channel_freqs),
    ); metadata=Dict(
        :band_center_frequency => band_center_frequency(data),
    ))
end

"""
    wrap_xy_correction(xy_correction, data::UVData, ref_ant; applies_to_pol, reference_pol)

Wrap the solved reference-site relative-feed correction in a `DimArray` with
scan and IF axes, plus metadata describing which site and polarisation it
applies to.
"""
function wrap_xy_correction(xy_correction, data::UVData, ref_ant; applies_to_pol, reference_pol)
    size(xy_correction, 1) == length(data.scans) || error("XY correction scan axis does not match UVData scans")
    size(xy_correction, 2) == length(data.metadata.channel_freqs) || error("XY correction IF axis does not match UVData channels")
    1 <= ref_ant <= length(data.antennas) || error("ref_ant index out of bounds")

    metadata = Dict(
        :SiteIndices => [ref_ant],
        :Sites => [data.antennas.name[ref_ant]],
        :applies_to_pol => applies_to_pol,
        :reference_pol => reference_pol,
        :channel_freqs => data.metadata.channel_freqs,
        :band_center_frequency => band_center_frequency(data),
    )

    return DimArray(
        xy_correction, (
            Scan(scan_time_centers(data)),
            IF(data.metadata.channel_freqs),
        ); metadata = metadata
    )
end
