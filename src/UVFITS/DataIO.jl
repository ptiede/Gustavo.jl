stokes_feed_pair(code::Integer) = code == -1 ? (1, 1) :
    code == -2 ? (2, 2) :
    code == -3 ? (1, 2) :
    code == -4 ? (2, 1) :
    error("Unsupported Stokes code: $code")

feed_pair_label(feeds::Tuple{<:Integer,<:Integer}) = string(feeds[1], feeds[2])
polarization_label(code::Integer) = feed_pair_label(stokes_feed_pair(code))

"""
    UVData

Full per-integration UV dataset loaded from a UVFITS file.

`vis` shape: `[nint, npol, nchan]`. `uvw` shape: `[nint, 3]` in light-seconds.
`scans` is a StructArray with `.lower` / `.upper` fields (hours).
`_hdus` holds the full raw `Vector{HDU}` for round-trip write-back.
"""
struct UVData{
    V<:AbstractArray{<:Complex},
    W<:AbstractArray{<:Real},
    TUvw<:AbstractArray{<:Real},
    TObs<:AbstractVector{<:Real},
    TScan<:AbstractVector{<:Integer},
    TBlCodes<:AbstractVector{<:Real},
    TBlPairs<:AbstractVector{<:Tuple{<:Integer,<:Integer}},
    TLookup<:AbstractDict{<:Real,<:Integer},
    TUniqueBls<:AbstractVector{<:Real},
    S,
    TAnt,
    TMeta,
    TRawShape<:Tuple,
    TSqueeze<:AbstractVector{<:Integer},
    THdus,
}
    vis          ::V
    weights      ::W
    uvw          ::TUvw
    obs_time     ::TObs
    scan_idx     ::TScan
    bl_codes     ::TBlCodes
    bl_pairs     ::TBlPairs
    bl_lookup    ::TLookup
    unique_bls   ::TUniqueBls
    scans        ::S
    antennas     ::TAnt
    metadata     ::TMeta
    raw_shape    ::TRawShape
    squeeze_dims ::TSqueeze
    _hdus        ::THdus
end

function with_visibilities(data::UVData, vis, weights)
    return UVData(vis, weights, data.uvw, data.obs_time, data.scan_idx, data.bl_codes,
        data.bl_pairs, data.bl_lookup, data.unique_bls, data.scans, data.antennas,
        data.metadata, data.raw_shape, data.squeeze_dims, data._hdus)
end

scan_time_centers(data::UVData) = [(scan.lower + scan.upper) / 2 for scan in data.scans]
band_center_frequency(data::UVData) = (first(data.metadata.channel_freqs) + last(data.metadata.channel_freqs)) / 2
centered_channel_freqs(data::UVData) = data.metadata.channel_freqs .- band_center_frequency(data)

function baseline_sites(data::UVData, bl::Tuple{String,String})
    a_idx = findfirst(==(bl[1]), data.antennas.name)
    b_idx = findfirst(==(bl[2]), data.antennas.name)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl")
    return a_idx, b_idx
end

function baseline_number(data::UVData, bl::Tuple{String,String})
    a_idx, b_idx = baseline_sites(data, bl)
    bi = findfirst(==((a_idx, b_idx)), data.bl_pairs)
    isnothing(bi) && error("Baseline $bl not in data")
    return bi
end

function card_value(cards, key)
    prefix = rpad(key, 8)
    for card in cards
        s = string(card)
        startswith(s, prefix) || continue
        parts = split(s, "="; limit=2)
        length(parts) == 2 || continue
        raw = strip(first(split(parts[2], "/"; limit=2)))
        if startswith(raw, "'") && endswith(raw, "'")
            return strip(raw[2:end-1])
        end
        try return parse(Int, raw) catch end
        try return parse(Float64, raw) catch end
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

function _find_freq_axis(cards)
    for i in 1:9
        ctype = card_value(cards, "CTYPE$i")
        (ctype isa AbstractString && uppercase(strip(ctype)) == "FREQ") || continue
        return something(card_value(cards, "CRVAL$i"), 0.0)
    end
    return 0.0
end

function _build_antenna_table(an_hdu)
    cards = an_hdu.cards
    an    = an_hdu.data

    clean(s) = filter(c -> isascii(c) && isprint(c) && !isspace(c), string(s))

    nant = size(an.ANNAME, 1)
    names_raw  = collect(an.ANNAME)
    names      = clean.(names_raw)
    xyz_raw    = collect(an.STABXYZ)
    mount_raw  = collect(an.MNTSTA)
    axoff_raw  = hasproperty(an, :STAXOF)   ? collect(an.STAXOF)   : fill(0.0f0, nant)
    diam_raw   = hasproperty(an, :DIAMETER) ? collect(an.DIAMETER)  : fill(0.0f0, nant)
    poltya_raw = hasproperty(an, :POLTYA)   ? collect(an.POLTYA)    : fill("R",   nant)
    poltyb_raw = hasproperty(an, :POLTYB)   ? collect(an.POLTYB)    : fill("L",   nant)
    polaa_raw  = hasproperty(an, :POLAA)    ? collect(an.POLAA)     : fill(0.0f0, nant)
    polab_raw  = hasproperty(an, :POLAB)    ? collect(an.POLAB)     : fill(0.0f0, nant)

    station_xyz = [Float64.(xyz_raw[i, :]) for i in eachindex(names)]

    antennas = StructArray{Antenna}((
        name        = collect(names),
        station_xyz = station_xyz,
        mount_type  = collect(Int.(mount_raw)),
        axis_offset = collect(Float64.(axoff_raw)),
        diameter    = collect(Float64.(diam_raw)),
        feed_a      = collect(clean.(poltya_raw)),
        feed_b      = collect(clean.(poltyb_raw)),
        pola_angle  = collect(Float64.(polaa_raw)),
        polb_angle  = collect(Float64.(polab_raw)),
    ))

    arrayx = something(card_value(cards, "ARRAYX"),  0.0)
    arrayy = something(card_value(cards, "ARRAYY"),  0.0)
    arrayz = something(card_value(cards, "ARRAYZ"),  0.0)
    arrnam  = something(card_value(cards, "ARRNAM"),  "")
    freq    = something(card_value(cards, "FREQ"),    0.0)
    rdate   = something(card_value(cards, "RDATE"),   "")
    gstia0  = something(card_value(cards, "GSTIA0"),  0.0)
    degpdy  = something(card_value(cards, "DEGPDY"),  360.9856)
    ut1utc  = something(card_value(cards, "UT1UTC"),  0.0)
    timsys  = something(card_value(cards, "TIMSYS"),  "UTC")
    frame   = something(card_value(cards, "FRAME"),   "ITRF")
    xyzhand = something(card_value(cards, "XYZHAND"), "RIGHT")

    return AntennaTable(
        antennas,
        (Float64(arrayx), Float64(arrayy), Float64(arrayz)),
        string(arrnam),
        Float64(freq),
        string(rdate),
        Float64(gstia0),
        Float64(degpdy),
        Float64(ut1utc),
        string(timsys),
        string(frame),
        string(xyzhand),
    )
end

function _build_obs_metadata(primary_hdu, fq_hdu, nvis_pol, nvis_chan)
    cards  = primary_hdu.cards
    fq     = fq_hdu.data

    object    = something(card_value(cards, "OBJECT"),   "")
    telescope = something(card_value(cards, "TELESCOP"), "")
    observer  = something(card_value(cards, "OBSERVER"), "")
    date_obs  = something(card_value(cards, "DATE-OBS"), "")
    equinox   = something(card_value(cards, "EQUINOX"),  2000.0)
    bunit     = something(card_value(cards, "BUNIT"),    "UNCALIB")

    ra  = something(card_value(cards, "OBSRA"),  0.0)
    dec = something(card_value(cards, "OBSDEC"), 0.0)
    if ra == 0.0
        ra = Float64(something(_find_crval(cards, "RA"), 0.0))
    end
    if dec == 0.0
        dec = Float64(something(_find_crval(cards, "DEC"), 0.0))
    end

    ref_freq = Float64(_find_freq_axis(cards))

    if_freq  = vec(Float64.(collect(getproperty(fq, Symbol("IF FREQ")))))
    channel_freqs = ref_freq .+ if_freq

    chan_bw   = hasproperty(fq, Symbol("TOTAL BANDWIDTH")) ?
        vec(Float64.(collect(getproperty(fq, Symbol("TOTAL BANDWIDTH"))))) :
        fill(0.0, length(if_freq))
    ch_width  = hasproperty(fq, Symbol("CH WIDTH")) ?
        Float64(first(collect(getproperty(fq, Symbol("CH WIDTH"))))) : 0.0
    sidebands = hasproperty(fq, :SIDEBAND) ?
        vec(Int.(collect(fq.SIDEBAND))) :
        fill(1, length(if_freq))

    length(channel_freqs) == nvis_chan ||
        error("FQ table has $(length(channel_freqs)) IFs but vis has $nvis_chan channels")

    pol_codes, pol_labels = parse_stokes_axis(cards, nvis_pol)

    return ObsMetadata(
        string(object),
        string(telescope),
        string(observer),
        string(date_obs),
        Float64(equinox),
        string(bunit),
        Float64(ra),
        Float64(dec),
        ref_freq,
        channel_freqs,
        chan_bw,
        ch_width,
        sidebands,
        pol_codes,
        pol_labels,
    )
end

function _find_crval(cards, ctype_prefix)
    for i in 1:9
        ctype = card_value(cards, "CTYPE$i")
        (ctype isa AbstractString &&
            startswith(uppercase(strip(ctype)), uppercase(ctype_prefix))) || continue
        return card_value(cards, "CRVAL$i")
    end
    return nothing
end

"""
    load_uvfits(path)

Load a UVFITS file, returning a `UVData`.
"""
function load_uvfits(path)
    fid = FITSFiles.fits(path)
    dt  = read(fid[1]).data
    an_hdu = read(fid[2])
    fq_hdu = read(fid[3])
    nx  = read(fid[4]).data

    dim1      = findall(==(1), size(dt.data))
    raw_shape = size(dt.data)
    raw       = Array(dropdims(dt.data, dims=Tuple(dim1)))

    vis     = complex.(raw[:, 1, :, :], raw[:, 2, :, :])
    weights = Float64.(raw[:, 3, :, :])

    antennas = _build_antenna_table(an_hdu)
    metadata = _build_obs_metadata(fid[1], fq_hdu, size(vis, 2), size(vis, 3))

    Ti       = dt.DATE[:, 2] .* 24
    bl_codes = Float64.(dt.BASELINE)

    _col(nt, prefix) = getproperty(nt, first(filter(k -> startswith(string(k), prefix), propertynames(nt))))
    uvw = hcat(Float64.(_col(dt, "UU")), Float64.(_col(dt, "VV")), Float64.(_col(dt, "WW")))

    lower    = (nx.TIME .- nx.var"TIME INTERVAL" ./ 2) .* 24
    upper    = (nx.TIME .+ nx.var"TIME INTERVAL" ./ 2) .* 24
    scans    = StructArray(lower=lower, upper=upper)

    scan_idx    = assign_scans(Ti, scans)
    unique_bls  = sort(unique(bl_codes))
    bl_lookup   = Dict(bl => i for (i, bl) in enumerate(unique_bls))
    bl_pairs    = decode_baseline.(unique_bls)

    return UVData(vis, weights, uvw, Ti, scan_idx, bl_codes, bl_pairs, bl_lookup,
        unique_bls, scans, antennas, metadata, raw_shape, dim1, fid)
end

decode_baseline(bl) = (Int(round(bl)) ÷ 256, Int(round(bl)) % 256)

function assign_scans(Ti, scans)
    idx = zeros(Int, length(Ti))
    for i in eachindex(Ti)
        for s in eachindex(scans)
            if scans[s].lower ≤ Ti[i] < scans[s].upper
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
    nscan = length(data.scans)
    nbl   = length(data.bl_pairs)

    V = zeros(ComplexF64, nscan, nbl, npol, nchan)
    W = zeros(Float64, nscan, nbl, npol, nchan)

    for i in 1:nint
        s  = data.scan_idx[i]
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

function restore_uvfits_shape(raw, raw_shape, squeeze_dims)
    shape = ntuple(i -> i in squeeze_dims ? 1 : raw_shape[i], length(raw_shape))
    return Float32.(reshape(raw, shape))
end

function default_output_path(path)
    root, ext = splitext(path)
    return root * "+bandpass" * ext
end

"""
    write_uvfits(output_path, data::UVData)

Write a UVFITS file with the visibilities and weights from `data`, reusing the
original HDUs stored in `data._hdus` for all metadata.
"""
function write_uvfits(output_path, data::UVData)
    lazy   = data._hdus[1].data           # LazyStructuredData from original file
    pcount = lazy.format.param            # number of random-groups parameters (e.g. 7)
    etype  = lazy.format.type             # Float32 (BITPIX=-32)
    dlen   = lazy.format.leng             # data elements per group (e.g. 384 = 3×4×32)
    nint   = size(data.vis, 1)

    # Build corrected raw data in original FITS shape, then flatten per-group
    raw_corr = zeros(Float32, nint, 3, size(data.vis, 2), size(data.vis, 3))
    raw_corr[:, 1, :, :] = Float32.(real.(data.vis))
    raw_corr[:, 2, :, :] = Float32.(imag.(data.vis))
    raw_corr[:, 3, :, :] = Float32.(data.weights)
    raw_flat = reshape(
        restore_uvfits_shape(raw_corr, data.raw_shape, data.squeeze_dims),
        nint, :)   # [nint, dlen] in FITS column-major order

    # Copy original file byte-for-bit (preserves PCOUNT, PTYPE, parameter data, and
    # all other HDUs exactly). Then overwrite only the DATA bytes in-place.
    cp(lazy.filnam, output_path; force=true)

    param_bytes    = pcount * sizeof(etype)
    bytes_per_group = (pcount + dlen) * sizeof(etype)
    param_buf      = Vector{UInt8}(undef, param_bytes)

    open(output_path, "r+") do io
        seek(io, lazy.begpos)
        for i in 1:nint
            read!(io, param_buf)          # advance past parameter bytes (keep them)
            write(io, hton.(raw_flat[i, :]))
        end
    end

    return output_path
end
