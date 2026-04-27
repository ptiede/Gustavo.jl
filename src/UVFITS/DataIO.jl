stokes_feed_pair(code::Integer) =
    code == -1 ? (1, 1) :
    code == -2 ? (2, 2) :
    code == -3 ? (1, 2) :
    code == -4 ? (2, 1) :
    code == -5 ? (1, 1) :
    code == -6 ? (2, 2) :
    code == -7 ? (1, 2) :
    code == -8 ? (2, 1) :
    error("Unsupported Stokes code: $code")

function polarization_label(code::Integer)
    code == -1 && return "RR"
    code == -2 && return "LL"
    code == -3 && return "RL"
    code == -4 && return "LR"
    code == -5 && return "XX"
    code == -6 && return "YY"
    code == -7 && return "XY"
    code == -8 && return "YX"
    error("Unsupported Stokes code: $code")
end

feed_type(s::String) = uppercase(s) ∈ ("R", "L") ? :circular : :linear
same_feed_type(a::String, b::String) = feed_type(a) == feed_type(b)


using DimensionalData: @dim, TimeDim
@dim Scan TimeDim "Scan Number"
@dim Pol "Polarization"
@dim IF "Intermediate Frequency"
@dim Ant "Antenna"
@dim Baseline "Baseline"
@dim Integration TimeDim "Integration Number (both baseline and time)"
@dim UVW "UVW Coordinate"


"""
    BaselineIndex

Encapsulates the AIPS baseline encoding internals so they do not clutter
the `UVData` field list.  End-users should rely on convenience functions
(`nbaselines`, `antenna_names`, etc.) rather than accessing these fields
directly.

- `codes` — per-integration AIPS baseline code (256a + b) as Float64
- `pairs` — unique `(a_idx, b_idx)` antenna-index pairs, one per unique baseline
- `lookup` — `Dict(code => index)` for O(1) mapping from code to pair index
- `unique_codes` — sorted unique baseline codes (one per unique baseline)
"""
struct BaselineIndex{TCodes, TPairs, TLookup}
    codes::TCodes   # per-integration AIPS codes
    pairs::TPairs   # unique (a_idx, b_idx) pairs
    lookup::TLookup  # code → index in pairs
    unique_codes::TCodes   # sorted unique codes
end

"""
    UVData

Full per-integration UV dataset loaded from a UVFITS file.

`vis` shape: `[nint, npol, nchan]`. `uvw` shape: `[nint, 3]` in light-seconds.
`scans` is a StructArray with `.lower` / `.upper` fields (hours).
`primary_cards` holds the FITS header cards from the primary HDU for round-trip
write-back without keeping two copies of the visibility data in memory.
`extra_columns` is a `NamedTuple` of any per-integration PTYPE columns from the
primary HDU that are not part of the canonical UVFITS axes
(UU/VV/WW/BASELINE/DATE) — e.g. `INTTIM`, `FREQSEL`, `SOURCE`. Keys are the
exact PTYPE strings. They are preserved through `with_visibilities` so that
`apply_bandpass(...) → write_uvfits(...)` round-trips do not lose them.
"""
struct UVData{
        V <: AbstractArray{<:Complex},
        W <: AbstractArray{<:Real},
        TUvw <: AbstractArray{<:Real},
        TObs <: AbstractVector{<:Real},
        TScan <: AbstractVector{<:Integer},
        TBl,
        S,
        TAnt,
        TCfg,
        TMeta,
        TCards,
        TDateParam <: AbstractMatrix{<:Real},
        TExtras <: NamedTuple,
    }
    vis::V
    weights::W
    uvw::TUvw
    obs_time::TObs
    scan_idx::TScan
    baselines::TBl
    scans::S
    antennas::TAnt
    array_config::TCfg
    metadata::TMeta
    primary_cards::TCards
    date_param::TDateParam   # raw [nint, 2] DATE matrix from PTYPE5/PTYPE6 (col 1 = JD reference, col 2 = fractional day)
    extra_columns::TExtras
end

# Backward-compatible constructors. Tests and scripts that pre-date
# `date_param` / `extra_columns` get sensible defaults: `date_param`
# reconstructs the (zero, obs_time/24) layout that the writer used to
# synthesize, and `extra_columns` defaults to empty.
UVData(
    vis, weights, uvw, obs_time, scan_idx, baselines, scans, antennas,
    array_config, metadata, primary_cards
) =
    UVData(
    vis, weights, uvw, obs_time, scan_idx, baselines, scans, antennas,
    array_config, metadata, primary_cards, _default_date_param(obs_time), NamedTuple()
)

UVData(
    vis, weights, uvw, obs_time, scan_idx, baselines, scans, antennas,
    array_config, metadata, primary_cards, extra_columns::NamedTuple
) =
    UVData(
    vis, weights, uvw, obs_time, scan_idx, baselines, scans, antennas,
    array_config, metadata, primary_cards, _default_date_param(obs_time), extra_columns
)

_default_date_param(obs_time) =
    hcat(zeros(Float32, length(obs_time)), Float32.(obs_time ./ 24))

function with_visibilities(data::UVData, vis, weights)
    return UVData(
        _rewrap_like(vis, data.vis), _rewrap_like(weights, data.weights),
        data.uvw, data.obs_time, data.scan_idx, data.baselines,
        data.scans, data.antennas, data.array_config, data.metadata, data.primary_cards,
        data.date_param, data.extra_columns
    )
end

# Preserve DimArray dim metadata when callers hand back a plain array result
# of the same shape (e.g. an externally-built vis cube). Already-DimArray
# inputs are passed through unchanged.
_rewrap_like(A::AbstractDimArray, ::AbstractDimArray) = A
_rewrap_like(A::AbstractDimArray, _) = A
_rewrap_like(A, ref::AbstractDimArray) =
    size(A) == size(ref) ? DimArray(A, dims(ref)) : A
_rewrap_like(A, _) = A

scan_time_centers(data::UVData) = [(scan.lower + scan.upper) / 2 for scan in data.scans]
band_center_frequency(data::UVData) = (first(data.metadata.channel_freqs) + last(data.metadata.channel_freqs)) / 2
centered_channel_freqs(data::UVData) = data.metadata.channel_freqs .- band_center_frequency(data)

function baseline_sites(data::UVData, bl::Tuple{String, String})
    a_idx = findfirst(==(bl[1]), data.antennas.name)
    b_idx = findfirst(==(bl[2]), data.antennas.name)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl")
    return a_idx, b_idx
end

function baseline_number(data::UVData, bl::Tuple{String, String})
    a_idx, b_idx = baseline_sites(data, bl)
    bi = findfirst(==((a_idx, b_idx)), data.baselines.pairs)
    isnothing(bi) && error("Baseline $bl not in data")
    return bi
end

function card_value(cards, key)
    target = rstrip(key)
    for card in cards
        rstrip(string(card.key)) == target || continue
        v = card.value
        v isa AbstractString && return strip(v)
        return v
    end
    # Fall back to scanning the serialized card text for callers that pass a
    # bare `Vector{Card}` whose `.key`/`.value` accessors are not available.
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
        catch end
        try
            return parse(Float64, raw)
        catch end
        # FORTRAN double-precision exponent ('D' instead of 'E').
        try
            return parse(Float64, replace(raw, r"[dD]" => "E"))
        catch end
        return raw
    end
    return nothing
end

# Number of regular axes in this HDU (FITS NAXIS card). Used to bound axis
# scans like CTYPE1..CTYPE_NAXIS rather than guessing a magic ceiling.
_naxis(cards) = Int(something(card_value(cards, "NAXIS"), 0))

# Find the first axis index `i ∈ 1:NAXIS` whose CTYPE$i value satisfies `pred`.
# Returns `nothing` if no axis matches.
function _find_axis(cards, pred)
    for i in 1:_naxis(cards)
        ctype = card_value(cards, "CTYPE$i")
        ctype isa AbstractString || continue
        pred(uppercase(strip(ctype))) && return i
    end
    return nothing
end

function parse_stokes_axis(cards, npol)
    axis = _find_axis(cards, ==("STOKES"))
    isnothing(axis) && return collect(1:npol), string.(1:npol)

    crval = something(card_value(cards, "CRVAL$axis"), 1.0)
    cdelt = something(card_value(cards, "CDELT$axis"), 1.0)
    crpix = something(card_value(cards, "CRPIX$axis"), 1.0)
    pol_codes = Int.(round.(crval .+ cdelt .* ((1:npol) .- crpix)))
    pol_labels = polarization_label.(pol_codes)
    return pol_codes, pol_labels
end

function _find_freq_axis(cards)
    i = _find_axis(cards, ==("FREQ"))
    isnothing(i) && return 0.0
    return something(card_value(cards, "CRVAL$i"), 0.0)
end

const _AN_MANDATORY_COLS = Set(
    [
        :ANNAME, :STABXYZ, :ORBPARM, :NOSTA, :MNTSTA, :STAXOF,
        :POLTYA, :POLAA, :POLCALA, :POLTYB, :POLAB, :POLCALB,
    ]
)

# Collect AN-table columns that are *not* in the mandatory Memo-117 Table 10
# set. Mirrors `_collect_extra_columns` for the primary HDU.
function _collect_an_extras(an)
    pairs = Pair{Symbol, Any}[]
    for sym in propertynames(an)
        sym in _AN_MANDATORY_COLS && continue
        push!(pairs, sym => getproperty(an, sym))
    end
    return (; pairs...)
end

# POLCALA/POLCALB are stored per Memo 117 as `E(NOPCAL, NO_IF)` — i.e. one
# `(NOPCAL × NO_IF)` Float32 array per antenna. FITSFiles may surface that as
# a `Matrix` (nant, nvals_per_ant) or already as `Vector{Vector{Float32}}`;
# normalize to a per-antenna vector. When the column is absent (`NOPCAL == 0`)
# we return empty per-antenna vectors.
function _split_per_antenna_polcal(an, sym::Symbol, nant)
    hasproperty(an, sym) || return [Float32[] for _ in 1:nant]
    raw = getproperty(an, sym)
    if raw isa AbstractMatrix
        return [Float32.(view(raw, i, :)) for i in 1:nant]
    end
    return [Float32.(x) for x in collect(raw)]
end

function _build_antenna_table(an_hdu)
    cards = an_hdu.cards
    an = an_hdu.data

    clean(s) = filter(c -> isascii(c) && isprint(c) && !isspace(c), string(s))

    nant = size(an.ANNAME, 1)
    names_raw = collect(an.ANNAME)
    names = clean.(names_raw)
    xyz_raw = collect(an.STABXYZ)
    mount_raw = collect(an.MNTSTA)
    axoff_raw = hasproperty(an, :STAXOF) ? collect(an.STAXOF) : fill(0.0f0, nant)
    poltya_raw = hasproperty(an, :POLTYA) ? collect(an.POLTYA) : fill("R", nant)
    poltyb_raw = hasproperty(an, :POLTYB) ? collect(an.POLTYB) : fill("L", nant)
    polaa_raw = hasproperty(an, :POLAA) ? collect(an.POLAA) : fill(0.0f0, nant)
    polab_raw = hasproperty(an, :POLAB) ? collect(an.POLAB) : fill(0.0f0, nant)
    polcala = _split_per_antenna_polcal(an, :POLCALA, nant)
    polcalb = _split_per_antenna_polcal(an, :POLCALB, nant)

    # STABXYZ is 3D (Float64) per Table 10.
    station_xyz = [Float64.(xyz_raw[i, :]) for i in eachindex(names)]

    antennas = StructArray{Antenna}(
        (
            name = names,
            station_xyz = station_xyz,
            mount_type = Int32.(mount_raw),
            axis_offset = Float32.(axoff_raw),
            feed_a = clean.(poltya_raw),
            feed_b = clean.(poltyb_raw),
            pola_angle = Float32.(polaa_raw),
            polb_angle = Float32.(polab_raw),
            polcala = polcala,
            polcalb = polcalb,
        )
    )

    # ARRAYX/Y/Z are header E (per Table 1, "usually double precision"); keep
    # Float64 to preserve geocentric precision.
    arrayx = Float64(something(card_value(cards, "ARRAYX"), 0.0))
    arrayy = Float64(something(card_value(cards, "ARRAYY"), 0.0))
    arrayz = Float64(something(card_value(cards, "ARRAYZ"), 0.0))
    arrnam = string(something(card_value(cards, "ARRNAM"), ""))

    rdate = string(something(card_value(cards, "RDATE"), ""))
    gstia0 = Float32(something(card_value(cards, "GSTIA0"), 0.0))
    degpdy = Float32(something(card_value(cards, "DEGPDY"), 360.9856))
    ut1utc = Float32(something(card_value(cards, "UT1UTC"), 0.0))
    polarx = Float32(something(card_value(cards, "POLARX"), 0.0))
    polary = Float32(something(card_value(cards, "POLARY"), 0.0))
    datutc = Float32(something(card_value(cards, "DATUTC"), 0.0))
    timsys = string(something(card_value(cards, "TIMSYS"), "UTC"))
    frame = string(something(card_value(cards, "FRAME"), "ITRF"))
    xyzhand = string(something(card_value(cards, "XYZHAND"), "RIGHT"))
    poltype = string(something(card_value(cards, "POLTYPE"), ""))
    extver = Int32(something(card_value(cards, "EXTVER"), 1))
    numorb = Int32(something(card_value(cards, "NUMORB"), 0))
    no_if = Int32(something(card_value(cards, "NO_IF"), 1))
    nopcal = Int32(something(card_value(cards, "NOPCAL"), 0))
    freqid = Int32(something(card_value(cards, "FREQID"), 1))

    ant_table = AntennaTable(
        antennas,
        (arrayx, arrayy, arrayz),
        arrnam,
        _collect_an_extras(an),
    )
    arr_config = ArrayConfig(
        rdate, gstia0, degpdy, ut1utc, polarx, polary, datutc,
        timsys, frame, xyzhand, poltype,
        extver, numorb, no_if, nopcal, freqid,
    )
    return ant_table, arr_config
end

const _FQ_MANDATORY_COLS = Set(
    [
        :FRQSEL, Symbol("IF FREQ"), Symbol("CH WIDTH"),
        Symbol("TOTAL BANDWIDTH"), :SIDEBAND,
    ]
)

function _collect_fq_extras(fq)
    pairs = Pair{Symbol, Any}[]
    for sym in propertynames(fq)
        sym in _FQ_MANDATORY_COLS && continue
        push!(pairs, sym => getproperty(fq, sym))
    end
    return (; pairs...)
end

# Optional primary-HDU cards we want to surface in `metadata.extras` when
# present. Mandatory cards (Table 5 above the line) are mapped to direct
# fields below; the rest land here.
const _OBS_OPTIONAL_CARDS = ("OBSERVER", "DATE-MAP", "BSCALE", "BZERO", "ALTRPIX")

function _collect_obs_card_extras(cards)
    pairs = Pair{Symbol, Any}[]
    for key in _OBS_OPTIONAL_CARDS
        v = card_value(cards, key)
        v === nothing && continue
        push!(pairs, Symbol(replace(key, "-" => "_")) => v)
    end
    return (; pairs...)
end

function _build_frequency_setup(cards, fq, nvis_chan)
    # FREQ axis CRVAL — kept Float64 for arithmetic stability across the band.
    ref_freq = Float64(_find_freq_axis(cards))

    # FQ table mandatory columns. `IF FREQ` is D (Float64); `CH WIDTH` and
    # `TOTAL BANDWIDTH` are E (Float32); `SIDEBAND` is J (Int32).
    if_freqs = vec(Float64.(collect(getproperty(fq, Symbol("IF FREQ")))))
    channel_freqs = ref_freq .+ if_freqs

    total_bandwidths = vec(Float32.(collect(getproperty(fq, Symbol("TOTAL BANDWIDTH")))))
    ch_widths = vec(Float32.(collect(getproperty(fq, Symbol("CH WIDTH")))))
    sidebands = vec(Int32.(collect(fq.SIDEBAND)))
    freqid = hasproperty(fq, :FRQSEL) ? Int32(first(collect(fq.FRQSEL))) : Int32(1)

    length(channel_freqs) == nvis_chan ||
        error("FQ table has $(length(channel_freqs)) IFs but vis has $nvis_chan channels")

    return FrequencySetup(
        freqid, ref_freq, channel_freqs, if_freqs,
        ch_widths, total_bandwidths, sidebands,
        _collect_fq_extras(fq),
    )
end

function _build_obs_metadata(primary_hdu, fq_hdu, nvis_pol, nvis_chan)
    cards = primary_hdu.cards
    fq = fq_hdu.data

    # Mandatory primary-HDU cards (Memo 117 Table 5, above the line).
    object = string(something(card_value(cards, "OBJECT"), ""))
    telescope = string(something(card_value(cards, "TELESCOP"), ""))
    instrume = string(something(card_value(cards, "INSTRUME"), ""))
    date_obs = string(something(card_value(cards, "DATE-OBS"), ""))
    equinox = Float32(something(card_value(cards, "EQUINOX"), 2000.0))
    bunit = string(something(card_value(cards, "BUNIT"), "UNCALIB"))

    ra = Float64(something(card_value(cards, "OBSRA"), 0.0))
    dec = Float64(something(card_value(cards, "OBSDEC"), 0.0))
    if ra == 0.0
        ra = Float64(something(_find_crval(cards, "RA"), 0.0))
    end
    if dec == 0.0
        dec = Float64(something(_find_crval(cards, "DEC"), 0.0))
    end

    freq_setup = _build_frequency_setup(cards, fq, nvis_chan)
    pol_codes, pol_labels = parse_stokes_axis(cards, nvis_pol)

    return ObsMetadata(
        object, telescope, instrume, date_obs, equinox, bunit,
        ra, dec,
        freq_setup,
        pol_codes, pol_labels,
        _collect_obs_card_extras(cards),
    )
end

function _find_crval(cards, ctype_prefix)
    needle = uppercase(ctype_prefix)
    i = _find_axis(cards, c -> startswith(c, needle))
    isnothing(i) && return nothing
    return card_value(cards, "CRVAL$i")
end

"""
    load_uvfits(path)

Load a UVFITS file, returning a `UVData`.
"""
function load_uvfits(path)
    fid = FITSFiles.fits(path)
    primary_hdu = read(fid[1])
    dt = primary_hdu.data
    an_hdu = read(fid[2])
    fq_hdu = read(fid[3])
    nx = read(fid[4]).data

    dim1 = findall(==(1), size(dt.data))
    raw = Array(dropdims(dt.data, dims = Tuple(dim1)))

    # Visibilities are stored on disk as triples (real, imag, weight) all `1E`
    # (Float32) on the COMPLEX axis. We keep that precision in memory: `vis`
    # is ComplexF32 and `weights` is Float32. No silent widening on read; no
    # narrowing on write.
    vis_raw = complex.(raw[:, 1, :, :], raw[:, 2, :, :])
    weights_raw = raw[:, 3, :, :]

    antennas, array_config = _build_antenna_table(an_hdu)
    metadata = _build_obs_metadata(primary_hdu, fq_hdu, size(vis_raw, 2), size(vis_raw, 3))

    # `obs_time` is a *summed* JD in hours-since-day-start; widen the Float32
    # DATE column to Float64 here for arithmetic stability (subtractions of
    # similar-magnitude JDs lose digits in Float32). The raw 2-column DATE
    # matrix is preserved verbatim in `date_param` for round-tripping.
    obs_time = Float64.(dt.DATE[:, 2]) .* 24
    bl_codes = round.(Int, dt.BASELINE)

    # UU/VV/WW are random params `1E` per Memo 117 Table 4 → keep Float32.
    _col(nt, prefix) = getproperty(nt, first(filter(k -> startswith(string(k), prefix), propertynames(nt))))
    uvw_raw = hcat(_col(dt, "UU"), _col(dt, "VV"), _col(dt, "WW"))

    vis = _wrap_int_pol_if(vis_raw, obs_time, metadata)
    weights = _wrap_int_pol_if(weights_raw, obs_time, metadata)
    uvw = _wrap_uvw(uvw_raw, obs_time)


    extra_columns = _collect_extra_columns(dt, primary_hdu.cards)
    # Preserve the raw 2-column DATE matrix verbatim — UVFITS stores the JD
    # reference in column 1 (sometimes via PZERO, sometimes inline) and the
    # fractional day in column 2. We must round-trip it exactly to keep the
    # absolute date intact on write-back.
    date_param = Matrix(dt.DATE)

    lower = (nx.TIME .- nx.var"TIME INTERVAL" ./ 2) .* 24
    upper = (nx.TIME .+ nx.var"TIME INTERVAL" ./ 2) .* 24
    scans = StructArray(lower = lower, upper = upper)

    scan_idx = assign_scans(obs_time, scans)
    unique_codes = sort(unique(bl_codes))
    bl_lookup = Dict(bl => i for (i, bl) in enumerate(unique_codes))
    bl_pairs = decode_baseline.(unique_codes)
    baselines = BaselineIndex(bl_codes, bl_pairs, bl_lookup, unique_codes)

    return UVData(
        vis, weights, uvw, obs_time, scan_idx, baselines, scans, antennas, array_config,
        metadata, primary_hdu.cards, date_param, extra_columns
    )
end

# Wrap a per-integration array with `(Integration, Pol, IF)` dims using the
# obs metadata for label lookups. Pol carries `pol_labels` (e.g. "RR"); IF
# carries `channel_freqs` (Hz, Float64); Integration carries `obs_time` hours.
_wrap_int_pol_if(arr, obs_time, metadata) = DimArray(
    arr,
    (Integration(obs_time), Pol(metadata.pol_labels), IF(metadata.channel_freqs)),
)

_wrap_uvw(arr, obs_time) = DimArray(arr, (Integration(obs_time), UVW(["U", "V", "W"])))

# Wrap the 4-D scan-averaged cube with `(Scan, Baseline, Pol, IF)` dims.
_wrap_scan_avg(arr, data::UVData) = DimArray(
    arr,
    (
        Scan(scan_time_centers(data)),
        Baseline(_baseline_labels(data)),
        Pol(data.metadata.pol_labels),
        IF(data.metadata.channel_freqs),
    ),
)

_baseline_labels(data::UVData) =
    [string(data.antennas.name[a], "-", data.antennas.name[b]) for (a, b) in data.baselines.pairs]

# Capture per-integration PTYPE columns that are not part of the canonical
# UVFITS axes (UU/VV/WW/BASELINE/DATE). Iterates the parsed primary-HDU
# NamedTuple directly rather than scanning numbered cards. Returned as a
# NamedTuple keyed by the exact PTYPE string so write-back can rebuild the
# layout verbatim.
function _collect_extra_columns(dt, primary_cards)
    canonical_prefixes = Set(["UU", "VV", "WW", "BASELINE", "DATE"])
    pairs = Pair{Symbol, Any}[]
    for sym in propertynames(dt)
        sym === :data && continue
        prefix = uppercase(String(split(String(sym), "-")[1]))
        prefix in canonical_prefixes && continue
        push!(pairs, sym => getproperty(dt, sym))
    end
    return (; pairs...)
end

decode_baseline(bl::Integer) = (bl ÷ 256, bl % 256)

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
    V, W = _scan_average_arrays(data.vis, data.weights, data)
    return with_visibilities(data, _wrap_scan_avg(V, data), _wrap_scan_avg(W, data))
end

function scan_average(vis, weights, data::UVData)
    V, W = _scan_average_arrays(vis, weights, data)
    return with_visibilities(data, _wrap_scan_avg(V, data), _wrap_scan_avg(W, data))
end

function _scan_average_arrays(vis, weights, data::UVData)
    nint, npol, nchan = size(vis)
    nscan = length(data.scans)
    nbl = length(data.baselines.pairs)

    V = zeros(eltype(vis), nscan, nbl, npol, nchan)
    W = zeros(eltype(weights), nscan, nbl, npol, nchan)

    for i in 1:nint
        s = data.scan_idx[i]
        s == 0 && continue
        bi = get(data.baselines.lookup, data.baselines.codes[i], 0)
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

function default_output_path(path)
    root, ext = splitext(path)
    return root * "+bandpass" * ext
end

# Iterate `primary_cards` and return PTYPE entries as `(index, name)` pairs in
# index order. The cards collection is the authoritative source; we don't probe
# `PTYPE1..PTYPE_N` with a magic ceiling.
function _ptype_entries(primary_cards)
    entries = Tuple{Int, String}[]
    for card in primary_cards
        m = match(r"^PTYPE(\d+)$", strip(string(card.key)))
        m === nothing && continue
        idx = parse(Int, m.captures[1])
        push!(entries, (idx, rstrip(string(card.value))))
    end
    sort!(entries; by = first)
    return entries
end

# Build the random-groups primary-HDU NamedTuple with field names matching the
# PTYPE cards on disk (e.g. "UU---SIN", "VV---SIN", "DATE"). FITSFiles' Random
# writer looks up `data[Symbol(PTYPE_name)]`, so the NamedTuple keys must equal
# the rstripped PTYPE values verbatim. Canonical UVFITS axes are filled from
# UVData fields; any extra PTYPEs (INTTIM, FREQSEL, SOURCE, …) come from
# `data.extra_columns`.
function _build_primary_data(data::UVData, raw_data, date)
    # All five canonical random-group columns are 1E on disk (Memo 117
    # Table 4). UVData carries them at native Float32 already; just hand them
    # through without round-trip conversion.
    uvw_raw = parent(data.uvw)
    canonical = Dict{String, Any}(
        "UU" => uvw_raw[:, 1],
        "VV" => uvw_raw[:, 2],
        "WW" => uvw_raw[:, 3],
        "BASELINE" => Float32.(data.baselines.codes),
        "DATE" => date,
    )
    pairs = Pair{Symbol, Any}[]
    seen = Set{String}()
    for (idx, name) in _ptype_entries(data.primary_cards)
        name in seen && continue   # duplicate PTYPEs (e.g. two DATE fields) share one key
        push!(seen, name)
        prefix = uppercase(String(split(name, "-")[1]))
        col = if haskey(canonical, prefix)
            canonical[prefix]
        elseif haskey(canonical, uppercase(String(name)))
            canonical[uppercase(String(name))]
        elseif haskey(data.extra_columns, Symbol(name))
            getproperty(data.extra_columns, Symbol(name))
        else
            error(
                "write_uvfits: PTYPE$idx = \"$name\" has no mapped column " *
                    "(neither a canonical UVFITS axis nor in data.extra_columns)"
            )
        end
        push!(pairs, Symbol(name) => col)
    end
    if isempty(pairs)
        # No PTYPE cards: fall back to canonical UVFITS axis layout.
        for k in ("UU", "VV", "WW", "BASELINE", "DATE")
            push!(pairs, Symbol(k) => canonical[k])
        end
    end
    push!(pairs, :data => raw_data)
    return (; pairs...)
end

function _build_an_hdu(antennas::AntennaTable, cfg::ArrayConfig, ref_freq::Float64)
    nant = length(antennas)
    # Mandatory Table 10 columns. Eltypes already match TFORM (Float32 for E,
    # Float64 for D); pass through.
    base = (
        ANNAME = rpad.(antennas.name, 8),
        STABXYZ = [collect(antennas.station_xyz[i]) for i in 1:nant],
        NOSTA = Int32.(1:nant),
        MNTSTA = antennas.mount_type,
        STAXOF = antennas.axis_offset,
        POLTYA = String.(antennas.feed_a),
        POLAA = antennas.pola_angle,
        POLTYB = String.(antennas.feed_b),
        POLAB = antennas.polb_angle,
        POLCALA = antennas.polcala,
        POLCALB = antennas.polcalb,
    )
    # Append optional columns (DIAMETER, BEAMFWHM, …) from `extras` so the
    # AN HDU is round-trip stable.
    data = merge(base, antennas.extras)
    cards = [
        Card("EXTNAME", "AIPS AN"),
        Card("EXTVER", cfg.extver),
        Card("ARRAYX", Float64(antennas.array_xyz[1])),
        Card("ARRAYY", Float64(antennas.array_xyz[2])),
        Card("ARRAYZ", Float64(antennas.array_xyz[3])),
        Card("ARRNAM", antennas.array_name),
        Card("FREQ", ref_freq),
        Card("RDATE", cfg.rdate),
        Card("GSTIA0", cfg.gst_iat0),
        Card("DEGPDY", cfg.earth_rot_rate),
        Card("UT1UTC", cfg.ut1utc),
        Card("POLARX", cfg.polarx),
        Card("POLARY", cfg.polary),
        Card("DATUTC", cfg.datutc),
        Card("TIMSYS", cfg.time_sys),
        Card("FRAME", cfg.frame),
        Card("XYZHAND", cfg.xyzhand),
        Card("POLTYPE", cfg.poltype),
        Card("NUMORB", cfg.numorb),
        Card("NO_IF", cfg.no_if),
        Card("NOPCAL", cfg.nopcal),
        Card("FREQID", cfg.freqid),
    ]
    return HDU(Bintable, data, cards)
end

_build_fq_hdu(metadata::ObsMetadata) = _build_fq_hdu(metadata.freq_setup)

function _build_fq_hdu(fs::FrequencySetup)
    # Eltypes already match Memo 117 Table 22 TFORMs (D for IF FREQ, E for CH
    # WIDTH and TOTAL BANDWIDTH, J for SIDEBAND); pass through.
    base = (
        FRQSEL = Int32[fs.freqid],
        var"IF FREQ" = [fs.if_freqs],
        var"CH WIDTH" = [fs.ch_widths],
        var"TOTAL BANDWIDTH" = [fs.total_bandwidths],
        SIDEBAND = [fs.sidebands],
    )
    data = merge(base, fs.extras)
    cards = Card[Card("EXTNAME", "AIPS FQ"), Card("NO_IF", Int32(length(fs.if_freqs)))]
    return HDU(Bintable, data, cards)
end

function _build_nx_hdu(data::UVData)
    nscan = length(data.scans)
    # NX columns: TIME is 1D (Float64), TIME INTERVAL is 1E (Float32).
    time_center = [Float64(scan.lower + scan.upper) for scan in data.scans] ./ 48.0
    time_interval = [Float32(scan.upper - scan.lower) for scan in data.scans] ./ 24.0f0
    start_vis = zeros(Int32, nscan)
    end_vis = zeros(Int32, nscan)
    for s in 1:nscan
        idxs = findall(==(s), data.scan_idx)
        isempty(idxs) && continue
        start_vis[s] = Int32(first(idxs))
        end_vis[s] = Int32(last(idxs))
    end
    nt_data = (
        TIME = time_center,
        var"TIME INTERVAL" = time_interval,
        var"SOURCE ID" = fill(Int32(1), nscan),
        SUBARRAY = fill(Int32(1), nscan),
        var"FREQ ID" = fill(Int32(1), nscan),
        var"START VIS" = start_vis,
        var"END VIS" = end_vis,
    )
    return HDU(Bintable, nt_data, Card[Card("EXTNAME", "AIPS NX")])
end

"""
    write_uvfits(output_path, data::UVData)

Write a UVFITS file from `data`, reconstructing all four HDUs (primary Random
Groups, AN antenna table, FQ frequency table, NX index table) from the metadata
stored in the `UVData` struct.
"""
function write_uvfits(output_path, data::UVData)
    nint = size(data.vis, 1)
    npol = size(data.vis, 2)
    nchan = size(data.vis, 3)

    raw_data = zeros(Float32, nint, 3, npol, nchan, 1, 1, 1)
    # vis is ComplexF32 and weights is Float32 (matching Memo 117 1E disk
    # types), so these stores are zero-conversion. Unwrap the DimArrays via
    # `parent` so the broadcast feeds plain Arrays into the FITS writer.
    vis_raw = parent(data.vis)
    weights_raw = parent(data.weights)
    raw_data[:, 1, :, :, 1, 1, 1] .= real.(vis_raw)
    raw_data[:, 2, :, :, 1, 1, 1] .= imag.(vis_raw)
    raw_data[:, 3, :, :, 1, 1, 1] .= weights_raw

    primary_data = _build_primary_data(data, raw_data, data.date_param)
    primary_hdu = HDU(Random, primary_data, copy(data.primary_cards))
    an_hdu = _build_an_hdu(data.antennas, data.array_config, data.metadata.ref_freq)
    fq_hdu = _build_fq_hdu(data.metadata)
    nx_hdu = _build_nx_hdu(data)

    write(output_path, HDU[primary_hdu, an_hdu, fq_hdu, nx_hdu])
    return output_path
end

antenna_names(data::UVData) = data.antennas.name
nbaselines(data::UVData) = length(data.baselines.pairs)
nscans(data::UVData) = length(data.scans)
nchannels(data::UVData) = length(data.metadata.channel_freqs)
npols(data::UVData) = length(data.metadata.pol_codes)
nintegrations(data::UVData) = size(data.vis, 1)

function Base.show(io::IO, data::UVData)
    names = join(antenna_names(data), ", ")
    nant = length(data.antennas)
    flo = round(minimum(data.metadata.channel_freqs) / 1.0e9; digits = 3)
    fhi = round(maximum(data.metadata.channel_freqs) / 1.0e9; digits = 3)
    pols = join(data.metadata.pol_labels, ", ")
    println(io, "UVData")
    println(io, "  Array: $(data.antennas.array_name) ($nant antennas: $names)")
    println(io, "  Scans: $(nscans(data))  Integrations: $(nintegrations(data))  Baselines: $(nbaselines(data))")
    println(io, "  IFs ($(nchannels(data))): $(flo)–$(fhi) GHz")
    return print(io, "  Polarizations ($(npols(data))): $pols")
end
