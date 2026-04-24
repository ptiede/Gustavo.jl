DD.@dim IF "IF or Channel"
DD.@dim Pol "Polarization"
DD.@dim Scan DD.TimeDim "Telescope Scan"
DD.@dim Ant "Antenna or Site"

struct UVData{
        V <: AbstractArray{<:Complex},
        W <: AbstractArray{<:Real},
        TObs <: AbstractVector{<:Real},
        TScan <: AbstractVector{<:Integer},
        TBlCodes <: AbstractVector{<:Real},
        TBlPairs <: AbstractVector{<:Tuple{<:Integer, <:Integer}},
        TLookup <: AbstractDict{<:Real, <:Integer},
        TUniqueBls <: AbstractVector{<:Real},
        TAntNames <: AbstractVector{<:AbstractString},
        S,
        TPolCodes <: AbstractVector{<:Integer},
        TPolLabels <: AbstractVector{<:AbstractString},
        TFreqs <: AbstractVector{<:Real},
        TRawShape <: Tuple,
        TSqueeze <: AbstractVector{<:Integer},
    }
    vis::V
    weights::W
    obs_time::TObs
    scan_idx::TScan
    bl_codes::TBlCodes
    bl_pairs::TBlPairs
    bl_lookup::TLookup
    unique_bls::TUniqueBls
    ant_names::TAntNames
    sc::S
    pol_codes::TPolCodes
    pol_labels::TPolLabels
    channel_freqs::TFreqs
    raw_shape::TRawShape
    squeeze_dims::TSqueeze
end

function with_visibilities(data::UVData, vis, weights)
    return UVData(
        vis, weights, data.obs_time, data.scan_idx, data.bl_codes, data.bl_pairs, data.bl_lookup,
        data.unique_bls, data.ant_names, data.sc, data.pol_codes, data.pol_labels, data.channel_freqs,
        data.raw_shape, data.squeeze_dims
    )
end

scan_time_centers(data::UVData) = [(scan.lower + scan.upper) / 2 for scan in data.sc]
band_center_frequency(data::UVData) = (first(data.channel_freqs) + last(data.channel_freqs)) / 2
centered_channel_freqs(data::UVData) = data.channel_freqs .- band_center_frequency(data)

function baseline_sites(data::UVData, bl::Tuple{String, String})
    a_idx = findfirst(==(bl[1]), data.ant_names)
    b_idx = findfirst(==(bl[2]), data.ant_names)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl")
    return a_idx, b_idx
end

function baseline_number(data::UVData, bl::Tuple{String, String})
    a_idx, b_idx = baseline_sites(data, bl)
    bi = findfirst(==((a_idx, b_idx)), data.bl_pairs)
    isnothing(bi) && error("Baseline $bl not in data")
    return bi
end

function wrap_baseline_array(A, data::UVData, bl::Tuple{String, String}; kind::Symbol, obs_inds = nothing)
    centered_freqs = centered_channel_freqs(data)
    metadata = Dict(
        :baseline => join(bl, "-"),
        :Sites => collect(bl),
        :kind => kind,
        :pol_codes => collect(data.pol_codes),
        :pol_labels => collect(data.pol_labels),
        :channel_freqs => collect(data.channel_freqs),
        :band_center_frequency => band_center_frequency(data),
    )

    if ndims(A) == 2
        error("wrap_baseline_array expects rank-3 baseline slices")
    elseif ndims(A) == 3 && !isnothing(obs_inds)
        metadata[:obs_indices] = collect(obs_inds)
        metadata[:scan_indices] = data.scan_idx[obs_inds]
        return DimArray(
            A, (
                Ti(data.obs_time[obs_inds]),
                Pol(collect(data.pol_labels[1:size(A, 2)])),
                IF(centered_freqs),
            ); metadata = metadata
        )
    elseif ndims(A) == 3
        return DimArray(
            A, (
                Scan(scan_time_centers(data)),
                Pol(collect(data.pol_labels[1:size(A, 2)])),
                IF(centered_freqs),
            ); metadata = metadata
        )
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
        bl_code = data.unique_bls[bi]
        obs_inds = findall(==(bl_code), data.bl_codes)
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
        bl_code = data.unique_bls[bi]
        obs_inds = findall(==(bl_code), data.bl_codes)
        return wrap_baseline_array(@view(data.weights[obs_inds, :, :]), data, bl; kind = :weights, obs_inds = obs_inds)
    elseif ndims(data.weights) == 4
        bi = baseline_number(data, bl)
        return wrap_baseline_array(@view(data.weights[:, bi, :, :]), data, bl; kind = :weights)
    else
        error("Unsupported weight rank: $(ndims(data.weights))")
    end
end

Base.getindex(data::UVData, bl::Tuple{String, String}) = baseline_visibilities(data, bl)

function card_value(cards, key)
    prefix = rpad(key, 8)
    for card in cards
        s = string(card)
        startswith(s, prefix) || continue
        parts = split(s, "="; limit = 2)
        length(parts) == 2 || continue
        raw = strip(first(split(parts[2], "/"; limit = 2)))
        if startswith(raw, "'") && endswith(raw, "'")
            return strip(raw[2:(end - 1)])
        end
        try
            return parse(Int, raw)
        catch
        end
        try
            return parse(Float64, raw)
        catch
        end
        return raw
    end
    return nothing
end

function parse_stokes_axis(cards, npol)
    axis = nothing
    for i in 1:7
        ctype = card_value(cards, "CTYPE$i")
        ctype == "STOKES" || continue
        axis = i
        break
    end
    isnothing(axis) && return collect(1:npol), string.(1:npol)

    crval = something(card_value(cards, "CRVAL$axis"), 1.0)
    cdelt = something(card_value(cards, "CDELT$axis"), 1.0)
    crpix = something(card_value(cards, "CRPIX$axis"), 1.0)
    pol_codes = Int.(round.(crval .+ cdelt .* ((1:npol) .- crpix)))
    pol_labels = polarization_label.(pol_codes)
    return pol_codes, pol_labels
end

function load_uvfits(path)
    fid = FITSFiles.fits(path)
    dt = fid[1].data
    an = fid[2].data
    fq = fid[3].data
    nx = fid[4].data

    clean(s) = filter(c -> isascii(c) && isprint(c) && !isspace(c), string(s))
    ant_names = clean.(an.ANNAME)

    dim1 = findall(==(1), size(dt.data))
    raw_shape = size(dt.data)
    raw = dropdims(dt.data, dims = Tuple(dim1))

    vis = complex.(raw[:, 1, :, :], raw[:, 2, :, :])
    weights = Float64.(raw[:, 3, :, :])
    pol_codes, pol_labels = parse_stokes_axis(fid[1].cards, size(vis, 2))
    channel_freqs = vec(Float64.(getproperty(fq, Symbol("IF FREQ"))))
    length(channel_freqs) == size(vis, 3) || error("Frequency table does not match channel count")

    Ti = dt.DATE[:, 2] .* 24
    bl_codes = Float64.(dt.BASELINE)

    lower = (nx.TIME .- nx.var"TIME INTERVAL" ./ 2) .* 24
    upper = (nx.TIME .+ nx.var"TIME INTERVAL" ./ 2) .* 24
    sc = StructArray(lower = lower, upper = upper)

    scan_idx = assign_scans(Ti, sc)

    unique_bls = sort(unique(bl_codes))
    bl_lookup = Dict(bl => i for (i, bl) in enumerate(unique_bls))
    bl_pairs = decode_baseline.(unique_bls)

    return UVData(
        vis, weights, Ti, scan_idx, bl_codes, bl_pairs, bl_lookup, unique_bls,
        ant_names, sc, pol_codes, pol_labels, channel_freqs, raw_shape, dim1
    )
end

decode_baseline(bl) = (Int(round(bl)) ÷ 256, Int(round(bl)) % 256)

function assign_scans(Ti, sc)
    idx = zeros(Int, length(Ti))
    for i in eachindex(Ti)
        for s in eachindex(sc)
            if sc[s].lower ≤ Ti[i] < sc[s].upper
                idx[i] = s
                break
            end
        end
    end
    return idx
end

function scan_average(data::UVData)
    return with_visibilities(data, _scan_average_arrays(data.vis, data.weights, data)...)
end

function scan_average(vis, weights, data::UVData)
    return with_visibilities(data, _scan_average_arrays(vis, weights, data)...)
end

function _scan_average_arrays(vis, weights, data::UVData)
    nint, npol, nchan = size(vis)
    nscan = length(data.sc)
    nbl = length(data.bl_pairs)

    V = zeros(ComplexF64, nscan, nbl, npol, nchan)
    W = zeros(Float64, nscan, nbl, npol, nchan)

    for i in 1:nint
        s = data.scan_idx[i]
        s == 0 && continue
        bi = get(data.bl_lookup, data.bl_codes[i], 0)
        bi == 0 && continue
        for p in 1:npol, c in 1:nchan
            w = weights[i, p, c]
            v = vis[i, p, c]
            (w > 0 && isfinite(w) && isfinite(real(v))) || continue
            V[s, bi, p, c] += w * v
            W[s, bi, p, c] += w
        end
    end

    for k in eachindex(V)
        V[k] = W[k] > 0 ? V[k] / W[k] : NaN + NaN * im
    end

    return V, W
end

"""
    wrap_gain_solutions(gains, data::UVData; pol_keys=1:2)

Wrap the solved gain cube in a `DimArray` so scans, sites, polarisations, and IFs
carry labeled dimensions for interactive inspection.
"""
function wrap_gain_solutions(gains, data::UVData; pol_keys = 1:2)
    size(gains, 1) == length(data.sc) || error("Gain scan axis does not match UVData scans")
    size(gains, 2) == length(data.ant_names) || error("Gain antenna axis does not match UVData antennas")
    size(gains, 3) == length(pol_keys) || error("pol_keys length must match gain polarisation axis")
    size(gains, 4) == length(data.channel_freqs) || error("Gain channel axis does not match UVData channels")

    return DimArray(
        gains, (
            Scan(scan_time_centers(data)),
            Ant(data.ant_names),
            Pol(collect(pol_keys)),
            IF(centered_channel_freqs(data)),
        ); metadata = Dict(
            :channel_freqs => collect(data.channel_freqs),
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
    size(xy_correction, 1) == length(data.sc) || error("XY correction scan axis does not match UVData scans")
    size(xy_correction, 2) == length(data.channel_freqs) || error("XY correction IF axis does not match UVData channels")
    1 <= ref_ant <= length(data.ant_names) || error("ref_ant index out of bounds")

    metadata = Dict(
        :SiteIndices => [ref_ant],
        :Sites => [data.ant_names[ref_ant]],
        :applies_to_pol => applies_to_pol,
        :reference_pol => reference_pol,
        :channel_freqs => collect(data.channel_freqs),
        :band_center_frequency => band_center_frequency(data),
    )

    return DimArray(
        xy_correction, (
            Scan(scan_time_centers(data)),
            IF(centered_channel_freqs(data)),
        ); metadata = metadata
    )
end
