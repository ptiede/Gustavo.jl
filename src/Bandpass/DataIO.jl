"""
    baseline_visibilities(data::UVData, bl::Tuple{String,String})

Return visibilities for a single baseline as a `DimArray`.

- For raw `UVData` loaded from uvfits this returns dimensions `(Integration, Pol, IF)`.
- For scan-averaged `UVData` this returns dimensions `(Scan, Pol, IF)`.
"""
baseline_visibilities(data::UVData, bl::Tuple{String, String}) =
    _baseline_slice(data.vis, data, bl, :vis)

"""
    baseline_weights(data::UVData, bl::Tuple{String,String})

Return weights for a single baseline as a `DimArray`.
The dimensional layout matches `baseline_visibilities`.
"""
baseline_weights(data::UVData, bl::Tuple{String, String}) =
    _baseline_slice(data.weights, data, bl, :weights)

# Pull a single baseline out of either the per-integration cube
# (Integration × Pol × IF) or the scan-averaged cube
# (Scan × Baseline × Pol × IF). Returned `DimArray` carries baseline
# metadata so downstream plotters and diagnostics can label sites.
function _baseline_slice(A::AbstractDimArray, data::UVData, bl::Tuple{String, String}, kind::Symbol)
    bi = baseline_number(data, bl)
    metadata = Dict(
        :baseline => join(bl, "-"),
        :Sites => bl,
        :kind => kind,
        :pol_codes => data.metadata.pol_codes,
        :pol_labels => data.metadata.pol_labels,
        :channel_freqs => data.metadata.channel_freqs,
        :band_center_frequency => band_center_frequency(data),
    )
    if hasdim(A, Baseline)
        return rebuild(@view(A[Baseline = bi]); metadata = metadata)
    end
    bl_code = data.baselines.unique_codes[bi]
    obs_inds = findall(==(bl_code), data.baselines.codes)
    metadata[:obs_indices] = obs_inds
    metadata[:scan_indices] = data.scan_idx[obs_inds]
    return rebuild(@view(A[Integration = obs_inds]); metadata = metadata)
end

Base.getindex(data::UVData, bl::Tuple{String, String}) = baseline_visibilities(data, bl)

"""
    getindex(data::UVData; kwargs...)

Slice the visibility, weights, and UVW arrays by named DimensionalData
dimensions. Selectors are forwarded to whichever fields contain that dim — e.g.
`Pol` and `IF` apply to `vis`/`weights`, `Integration` applies to all three,
and `Baseline`/`Scan` apply to scan-averaged datasets.

```julia
data[Pol = At("RR")]                        # NamedTuple of slices
data[Baseline = At("AA-AX"), IF = 1:4]      # scan-averaged path
```

Returns a `NamedTuple{(:vis, :weights, :uvw)}` of the sliced arrays.
"""
function Base.getindex(data::UVData; kwargs...)
    return (
        vis = _slice_dim(data.vis; kwargs...),
        weights = _slice_dim(data.weights; kwargs...),
        uvw = _slice_dim(data.uvw; kwargs...),
    )
end

function _slice_dim(A::AbstractDimArray; kwargs...)
    keys_keep = Tuple(k for k in keys(kwargs) if hasdim(A, name2dim(k)))
    isempty(keys_keep) && return A
    keep = NamedTuple{keys_keep}(values(kwargs))
    return getindex(A; keep...)
end

"""
    wrap_gain_solutions(gains, data::UVData; pol_keys=1:2)

Wrap the solved gain cube in a `DimArray` so scans, sites, polarisations, and IFs
carry labeled dimensions for interactive inspection.
"""
function wrap_gain_solutions(gains, data::UVData; pol_keys = 1:2)
    size(gains, 1) == length(data.scans) || error("Gain scan axis does not match UVData scans")
    size(gains, 2) == length(data.antennas) || error("Gain antenna axis does not match UVData antennas")
    size(gains, 3) == length(pol_keys) || error("pol_keys length must match gain polarisation axis")
    size(gains, 4) == length(data.metadata.channel_freqs) || error("Gain channel axis does not match UVData channels")

    return DimArray(
        gains, (
            Scan(scan_time_centers(data)),
            Ant(data.antennas.name),
            Pol(collect(pol_keys)),
            IF(data.metadata.channel_freqs),
        ); metadata = Dict(
            :band_center_frequency => band_center_frequency(data),
        )
    )
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
