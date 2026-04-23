struct UVData{V<:AbstractArray{<:Complex}, W<:AbstractArray{<:Real}}
    vis          ::V
    weights      ::W
    scan_idx     ::Vector{Int}
    bl_codes     ::Vector{Float64}
    bl_pairs     ::Vector{Tuple{Int,Int}}
    bl_lookup    ::Dict{Float64,Int}
    unique_bls   ::Vector{Float64}
    ant_names    ::Vector{String}
    sc           ::StructArray
    channel_freqs::Vector{Float64}
    raw_shape    ::Tuple
    squeeze_dims ::Vector{Int}
end

function with_visibilities(data::UVData, vis, weights)
    return UVData(vis, weights, data.scan_idx, data.bl_codes, data.bl_pairs, data.bl_lookup,
        data.unique_bls, data.ant_names, data.sc, data.channel_freqs,
        data.raw_shape, data.squeeze_dims)
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
    raw = dropdims(dt.data, dims=Tuple(dim1))

    vis = complex.(raw[:, 1, :, :], raw[:, 2, :, :])
    weights = Float64.(raw[:, 3, :, :])
    channel_freqs = vec(Float64.(getproperty(fq, Symbol("IF FREQ"))))
    length(channel_freqs) == size(vis, 3) || error("Frequency table does not match channel count")

    Ti = dt.DATE[:, 2] .* 24
    bl_codes = Float64.(dt.BASELINE)

    lower = (nx.TIME .- nx.var"TIME INTERVAL" ./ 2) .* 24
    upper = (nx.TIME .+ nx.var"TIME INTERVAL" ./ 2) .* 24
    sc = StructArray(lower=lower, upper=upper)

    scan_idx = assign_scans(Ti, sc)

    unique_bls = sort(unique(bl_codes))
    bl_lookup = Dict(bl => i for (i, bl) in enumerate(unique_bls))
    bl_pairs = decode_baseline.(unique_bls)

    return UVData(vis, weights, scan_idx, bl_codes, bl_pairs, bl_lookup, unique_bls,
        ant_names, sc, channel_freqs, raw_shape, dim1)
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
    wrap_gain_solutions(gains, data::UVData; feed_labels=["R/X", "L/Y"])

Wrap the solved gain cube in a `DimArray` so scans, antennas, feeds, and channels
carry labeled dimensions for interactive inspection.
"""
function wrap_gain_solutions(gains, data::UVData; feed_labels=["R/X", "L/Y"])
    size(gains, 1) == length(data.sc) || error("Gain scan axis does not match UVData scans")
    size(gains, 2) == length(data.ant_names) || error("Gain antenna axis does not match UVData antennas")
    size(gains, 3) == length(feed_labels) || error("feed_labels length must match gain feed axis")
    size(gains, 4) == length(data.channel_freqs) || error("Gain channel axis does not match UVData channels")

    return DimArray(gains, (
        Dim{:scan}(collect(1:length(data.sc))),
        Dim{:antenna}(data.ant_names),
        Dim{:feed}(collect(feed_labels)),
        Dim{:channel}(data.channel_freqs),
    ))
end
