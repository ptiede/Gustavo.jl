module GustavoFITSFilesExt

using FITSFiles
using FITSFiles: HDU, Random, Bintable, Card
using StructArrays
using LinearAlgebra: Diagonal
using DimensionalData
using DimensionalData: DimArray
using PolarizedTypes: CirBasis, LinBasis, XPol, YPol, RPol, LPol

import Gustavo.UVData
using Gustavo.UVData:
    UVSet, UVMetadata, ObsArrayMetadata, FrequencySetup,
    Antenna, AntennaTable, ArrayConfig, BaselineIndex,
    Mount, MountAltAz, MountEquatorial, MountNaismithR, MountNaismithL,
    Integration, Pol, IF, UVW,
    sources, parallactic_mount, elevation_mount, offset_mount,
    array_xyz, array_name, extras,
    assign_scans, decode_baseline


# AIPS POLTYA/POLTYB letter ↔ PolarizedTypes
function poltype(type)
    type == "R" && return RPol()
    type == "L" && return LPol()
    type == "X" && return XPol()
    type == "Y" && return YPol()
    error("Unsupported polarization type: $type")
end

poltype_letter(::RPol) = "R"
poltype_letter(::LPol) = "L"
poltype_letter(::XPol) = "X"
poltype_letter(::YPol) = "Y"


# AIPS Stokes code → generic correlation-product label.
# Codes -1..-4 (RR/LL/RL/LR) and -5..-8 (XX/YY/XY/YX) both map to the same
# (P, Q) feed-index pattern, so the generic labels are basis-agnostic:
#   (1,1) → "PP", (2,2) → "QQ", (1,2) → "PQ", (2,1) → "QP".
function aips_code_to_generic(code::Integer)
    code in (-1, -5) && return "PP"
    code in (-2, -6) && return "QQ"
    code in (-3, -7) && return "PQ"
    code in (-4, -8) && return "QP"
    error("Unsupported Stokes code: $code")
end

# Inverse: pick AIPS Stokes codes for the given generic labels, in either the
# circular (-1..-4) or linear (-5..-8) block. Mixed-feed arrays fall back to
# the circular block by convention.
function generic_to_aips_code(label::AbstractString, basis::Symbol)
    block = basis === :linear ? (-5, -6, -7, -8) : (-1, -2, -3, -4)
    label == "PP" && return block[1]
    label == "QQ" && return block[2]
    label == "PQ" && return block[3]
    label == "QP" && return block[4]
    error("Unsupported correlation label: $label")
end

# MSv4 canonical ordering: parallel-hands first, then cross-hands in PQ/QP
# order. For sub-sets we keep the canonical relative order.
const _MSV4_CANONICAL = ("PP", "PQ", "QP", "QQ")
function _msv4_order(labels::AbstractVector{<:AbstractString})
    out = String[]
    for canon in _MSV4_CANONICAL
        canon in labels && push!(out, canon)
    end
    Set(out) == Set(labels) ||
        error("Unexpected polarization labels for MSv4 ordering: $labels")
    return out
end


# AIPS MNTSTA ↔ Mount round-trip.
function mnt_codes_to_type(code, offset)
    code == 0 && return MountAltAz(offset)
    code == 1 && return MountEquatorial(offset)
    code == 2 && throw(ArgumentError("Orbital antennas are not supported yet"))
    code == 3 && throw(ArgumentError("X-Y mounts are not supported yet"))
    code == 4 && return MountNaismithR(offset)
    code == 5 && return MountNaismithL(offset)
    code == 6 && throw(ArgumentError("Aperture/phased array mounts are not supported yet"))
    error("Unsupported MNTSTA code: $code")
end

function mount_to_mntsta(m::Mount)::Int32
    par = parallactic_mount(m)
    el = elevation_mount(m)
    par == 1 && el == 0 && return Int32(0)   # alt-az
    par == 0 && el == 0 && return Int32(1)   # equatorial
    par == 1 && el == 1 && return Int32(4)   # Naismith R
    par == 1 && el == -1 && return Int32(5)  # Naismith L
    error("Mount $(m) has no AIPS MNTSTA mapping")
end





    # ── Card / HDU parsing helpers ──────────────────────────────────────────────

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

_naxis(cards) = Int(something(card_value(cards, "NAXIS"), 0))

function _find_axis(cards, pred)
    for i in 1:_naxis(cards)
        ctype = card_value(cards, "CTYPE$i")
        ctype isa AbstractString || continue
        pred(uppercase(strip(ctype))) && return i
    end
    return nothing
end

"""
    parse_stokes_axis(cards, npol) -> (aips_codes, aips_labels, msv4_labels, perm)

Parse AIPS STOKES axis. Returns the raw AIPS pol codes (e.g. `[-1,-2,-3,-4]`),
the corresponding generic labels in the original AIPS axis order, the same
labels permuted to MSv4 canonical order (`["PP","PQ","QP","QQ"]`), and the
permutation `perm` such that `aips_labels[perm] == msv4_labels`.
"""
function parse_stokes_axis(cards, npol)
    axis = _find_axis(cards, ==("STOKES"))
    if isnothing(axis)
        labels = string.(1:npol)
        return Int[], labels, labels, collect(1:npol)
    end

    crval = something(card_value(cards, "CRVAL$axis"), 1.0)
    cdelt = something(card_value(cards, "CDELT$axis"), 1.0)
    crpix = something(card_value(cards, "CRPIX$axis"), 1.0)
    aips_codes = Int.(round.(crval .+ cdelt .* ((1:npol) .- crpix)))
    aips_labels = aips_code_to_generic.(aips_codes)
    msv4_labels = _msv4_order(aips_labels)
    perm = [findfirst(==(lab), aips_labels) for lab in msv4_labels]
    any(isnothing, perm) && error("parse_stokes_axis: cannot reconcile AIPS labels $aips_labels with MSv4 order $msv4_labels")
    return aips_codes, aips_labels, msv4_labels, Vector{Int}(perm)
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

function _collect_an_extras(an)
    pairs = Pair{Symbol, Any}[]
    for sym in propertynames(an)
        sym in _AN_MANDATORY_COLS && continue
        push!(pairs, sym => getproperty(an, sym))
    end
    return (; pairs...)
end

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

    names = clean.(collect(an.ANNAME))
    nant = length(names)
    xyz_raw = collect(an.STABXYZ)
    mount_raw = collect(an.MNTSTA)
    staxof_raw = hasproperty(an, :STAXOF) ? collect(an.STAXOF) : fill(0.0f0, nant)
    mnts = mnt_codes_to_type.(mount_raw, staxof_raw)
    poltya_raw = hasproperty(an, :POLTYA) ? collect(an.POLTYA) : fill("R", nant)
    poltya = poltype.(poltya_raw)
    poltyb_raw = hasproperty(an, :POLTYB) ? collect(an.POLTYB) : fill("L", nant)
    poltyb = poltype.(poltyb_raw)
    polaa_raw = hasproperty(an, :POLAA) ? collect(an.POLAA) : fill(0.0f0, nant)
    polab_raw = hasproperty(an, :POLAB) ? collect(an.POLAB) : fill(0.0f0, nant)
    pol_angles = tuple.(Float32.(polaa_raw), Float32.(polab_raw))

    response = [Diagonal(ones(ComplexF32, 2)) for _ in 1:nant]
    station_xyz = [Float64.(xyz_raw[i, :]) for i in eachindex(names)]
    nominal_basis = tuple.(poltya, poltyb)

    antennas = [
        Antenna(;
            name = names[i],
            station_xyz = station_xyz[i],
            mount = mnts[i],
            nominal_basis = nominal_basis[i],
            response = response[i],
            pol_angles = pol_angles[i],
        )
        for i in 1:nant
    ]

    arrayx = Float64(something(card_value(cards, "ARRAYX"), 0.0))
    arrayy = Float64(something(card_value(cards, "ARRAYY"), 0.0))
    arrayz = Float64(something(card_value(cards, "ARRAYZ"), 0.0))
    arrnam = string(something(card_value(cards, "ARRNAM"), ""))

    ant_extras = (;
        POLCALA = _split_per_antenna_polcal(an, :POLCALA, nant),
        POLCALB = _split_per_antenna_polcal(an, :POLCALB, nant),
        _collect_an_extras(an)...,
    )

    ant_table = AntennaTable(StructArray(antennas), (arrayx, arrayy, arrayz), arrnam, ant_extras)


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
    poltype_card = string(something(card_value(cards, "POLTYPE"), ""))
    extver = Int32(something(card_value(cards, "EXTVER"), 1))
    numorb = Int32(something(card_value(cards, "NUMORB"), 0))
    no_if = Int32(something(card_value(cards, "NO_IF"), 1))
    nopcal = Int32(something(card_value(cards, "NOPCAL"), 0))
    freqid = Int32(something(card_value(cards, "FREQID"), 1))

    arr_config = ArrayConfig(
        rdate, gstia0, degpdy, ut1utc, polarx, polary, datutc,
        timsys, frame, xyzhand, poltype_card,
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
    ref_freq = Float64(_find_freq_axis(cards))

    if_freqs = vec(Float64.(collect(getproperty(fq, Symbol("IF FREQ")))))
    channel_freqs = ref_freq .+ if_freqs

    total_bandwidths = vec(Float32.(collect(getproperty(fq, Symbol("TOTAL BANDWIDTH")))))
    ch_widths = vec(Float32.(collect(getproperty(fq, Symbol("CH WIDTH")))))
    sidebands = vec(Int32.(collect(fq.SIDEBAND)))
    freqid = hasproperty(fq, :FRQSEL) ? Int32(first(collect(fq.FRQSEL))) : Int32(1)

    length(channel_freqs) == nvis_chan ||
        error("FQ table has $(length(channel_freqs)) IFs but vis has $nvis_chan channels")

    return FrequencySetup(;
        name = string("FRQSEL_", freqid),
        ref_freq,
        channel_freqs,
        ch_widths,
        total_bandwidths,
        sidebands,
        extras = _collect_fq_extras(fq),
    )
end

function _build_array_obs_metadata(primary_hdu, fq_hdu, nvis_chan)
    cards = primary_hdu.cards
    fq = fq_hdu.data

    telescope = string(something(card_value(cards, "TELESCOP"), ""))
    instrume = string(something(card_value(cards, "INSTRUME"), ""))
    date_obs = string(something(card_value(cards, "DATE-OBS"), ""))
    equinox = Float32(something(card_value(cards, "EQUINOX"), 2000.0))
    bunit = string(something(card_value(cards, "BUNIT"), "UNCALIB"))

    freq_setup = _build_frequency_setup(cards, fq, nvis_chan)

    return ObsArrayMetadata(;
        telescope, instrume, date_obs, equinox, bunit,
        freq_setup,
        extras = _collect_obs_card_extras(cards),
    )
end

function _build_source_info(primary_hdu)
    cards = primary_hdu.cards
    object = string(something(card_value(cards, "OBJECT"), ""))
    ra = Float64(something(card_value(cards, "OBSRA"), 0.0))
    dec = Float64(something(card_value(cards, "OBSDEC"), 0.0))
    if ra == 0.0
        ra = Float64(something(_find_crval(cards, "RA"), 0.0))
    end
    if dec == 0.0
        dec = Float64(something(_find_crval(cards, "DEC"), 0.0))
    end
    return (; source_name = object, ra = ra, dec = dec)
end

function _find_crval(cards, ctype_prefix)
    needle = uppercase(ctype_prefix)
    i = _find_axis(cards, c -> startswith(c, needle))
    isnothing(i) && return nothing
    return card_value(cards, "CRVAL$i")
end

# ── Read path ───────────────────────────────────────────────────────────────

UVData.load_uvfits(path) = UVSet(_load_uvfits_flat(path))

function _load_uvfits_flat(path)
    fid = FITSFiles.fits(path)
    primary_hdu = fid[1]
    dt = primary_hdu.data
    an_hdu = fid[2]
    fq_hdu = fid[3]
    nx = fid[4].data

    dim1 = findall(==(1), size(dt.data))
    raw = Array(dropdims(dt.data, dims = Tuple(dim1)))

    vis_raw = complex.(raw[:, 1, :, :], raw[:, 2, :, :])
    weights_raw = raw[:, 3, :, :]

    antennas, array_config = _build_antenna_table(an_hdu)
    array_obs = _build_array_obs_metadata(primary_hdu, fq_hdu, size(vis_raw, 3))
    src_info = _build_source_info(primary_hdu)

    aips_codes, aips_labels, msv4_labels, perm = parse_stokes_axis(primary_hdu.cards, size(vis_raw, 2))
    _check_stokes_vs_poltya(aips_codes, antennas)
    vis_raw = vis_raw[:, perm, :]
    weights_raw = weights_raw[:, perm, :]

    # AIPS UVFITS often stores DATE as two PTYPE columns (integer JD +
    # fractional day). FITSFiles returns these merged as a matrix with the
    # fractional part in column 2. When only one DATE column is present
    # (e.g. round-tripped synthetic fixtures), the value is a vector of
    # fractional days.
    date_raw = collect(dt.DATE)
    obs_time = if ndims(date_raw) == 2
        Float64.(date_raw[:, 2]) .* 24
    else
        Float64.(date_raw) .* 24
    end
    bl_codes = round.(Int, collect(dt.BASELINE))

    _col(nt, prefix) = getproperty(nt, first(filter(k -> startswith(string(k), prefix), propertynames(nt))))
    uvw_raw = hcat(_col(dt, "UU"), _col(dt, "VV"), _col(dt, "WW"))

    vis = _wrap_int_pol_if(vis_raw, obs_time, msv4_labels, array_obs.freq_setup.channel_freqs)
    weights = _wrap_int_pol_if(weights_raw, obs_time, msv4_labels, array_obs.freq_setup.channel_freqs)
    uvw = _wrap_uvw(uvw_raw, obs_time)

    extra_columns = _collect_extra_columns(dt, primary_hdu.cards)
    date_param = ndims(date_raw) == 2 ? Matrix(date_raw) : reshape(collect(date_raw), :, 1)

    lower = (nx.TIME .- nx.var"TIME INTERVAL" ./ 2) .* 24
    upper = (nx.TIME .+ nx.var"TIME INTERVAL" ./ 2) .* 24
    scans = StructArray(lower = lower, upper = upper)

    scan_idx = assign_scans(obs_time, scans)
    unique_codes = sort(unique(bl_codes))
    bl_lookup = Dict(bl => i for (i, bl) in enumerate(unique_codes))
    bl_pairs = decode_baseline.(unique_codes)
    baselines = BaselineIndex(
        bl_codes, bl_pairs, bl_lookup, unique_codes;
        antenna_names = antennas.name
    )

    basename = String(splitext(_basename_of_path(path))[1])

    return (;
        vis, weights, uvw, obs_time, scan_idx, baselines,
        date_param, extra_columns,
        scans, antennas, array_config, array_obs,
        pol_labels = msv4_labels,
        aips_pol_codes = aips_codes,
        source_name = src_info.source_name,
        ra = src_info.ra, dec = src_info.dec,
        primary_cards = primary_hdu.cards,
        basename = basename,
    )
end

_basename_of_path(path) = isempty(path) ? "uvfits" : Base.basename(String(path))

_wrap_int_pol_if(arr, obs_time, pol_labels, channel_freqs) = DimArray(
    arr,
    (Integration(obs_time), Pol(pol_labels), IF(channel_freqs)),
)

# Warn if the AIPS Stokes axis (circular vs linear block) doesn't match the
# antennas' nominal basis (POLTYA/POLTYB). For mixed arrays we just check the
# circular vs linear block — fine-grained per-antenna mismatches are the
# user's problem.
function _check_stokes_vs_poltya(aips_codes, antennas)
    isempty(aips_codes) && return nothing
    stokes_is_linear = all(c -> c <= -5, aips_codes)
    stokes_is_circular = all(c -> -4 <= c <= -1, aips_codes)
    feeds = collect(antennas.nominal_basis)
    poltype_is_linear(p::Tuple) = all(x -> x isa Union{XPol, YPol}, p)
    poltype_is_circular(p::Tuple) = all(x -> x isa Union{RPol, LPol}, p)
    feeds_linear = all(poltype_is_linear, feeds)
    feeds_circular = all(poltype_is_circular, feeds)
    if stokes_is_linear && !feeds_linear
        @warn "Stokes axis is linear (XX/YY/XY/YX) but POLTYA/POLTYB are not all linear"
    elseif stokes_is_circular && !feeds_circular
        @warn "Stokes axis is circular (RR/LL/RL/LR) but POLTYA/POLTYB are not all circular"
    end
    return nothing
end

_wrap_uvw(arr, obs_time) = DimArray(arr, (Integration(obs_time), UVW(["U", "V", "W"])))

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

# ── Write path ──────────────────────────────────────────────────────────────

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

function _build_primary_data(uvset::UVSet, uu, vv, ww, bl_codes, date_param, extra_cols, raw_data)
    canonical = Dict{String, Any}(
        "UU" => uu,
        "VV" => vv,
        "WW" => ww,
        "BASELINE" => Float32.(bl_codes),
        "DATE" => date_param,
    )
    pairs = Pair{Symbol, Any}[]
    seen = Set{String}()
    for (idx, name) in _ptype_entries(DimensionalData.metadata(uvset).primary_cards)
        name in seen && continue
        push!(seen, name)
        prefix = uppercase(String(split(name, "-")[1]))
        col = if haskey(canonical, prefix)
            canonical[prefix]
        elseif haskey(canonical, uppercase(String(name)))
            canonical[uppercase(String(name))]
        elseif haskey(extra_cols, Symbol(name))
            getproperty(extra_cols, Symbol(name))
        else
            error(
                "write_uvfits: PTYPE$idx = \"$name\" has no mapped column " *
                    "(neither a canonical UVData axis nor in extra_columns)"
            )
        end
        push!(pairs, Symbol(name) => col)
    end
    if isempty(pairs)
        for k in ("UU", "VV", "WW", "BASELINE", "DATE")
            push!(pairs, Symbol(k) => canonical[k])
        end
    end
    push!(pairs, :data => raw_data)
    return (; pairs...)
end

function _build_an_hdu(antennas::AntennaTable, cfg::ArrayConfig, ref_freq::Float64)
    nant = length(antennas)
    nb = collect(antennas.nominal_basis)
    pa = collect(antennas.pol_angles)
    mounts = collect(antennas.mount)
    xyz = collect(antennas.station_xyz)
    ext = extras(antennas)
    polcala = haskey(ext, :POLCALA) ? ext.POLCALA : [Float32[] for _ in 1:nant]
    polcalb = haskey(ext, :POLCALB) ? ext.POLCALB : [Float32[] for _ in 1:nant]
    extras_rest = NamedTuple{filter(s -> !(s in (:POLCALA, :POLCALB)), keys(ext))}(ext)

    base = (
        ANNAME = rpad.(antennas.name, 8),
        STABXYZ = [collect(xyz[i]) for i in 1:nant],
        NOSTA = Int32.(1:nant),
        MNTSTA = [mount_to_mntsta(m) for m in mounts],
        STAXOF = Float32[Float32(offset_mount(m)) for m in mounts],
        POLTYA = [poltype_letter(p[1]) for p in nb],
        POLAA = Float32[a[1] for a in pa],
        POLTYB = [poltype_letter(p[2]) for p in nb],
        POLAB = Float32[a[2] for a in pa],
        POLCALA = polcala,
        POLCALB = polcalb,
    )
    data = merge(base, extras_rest)
    arr_xyz = array_xyz(antennas)
    cards = [
        Card("EXTNAME", "AIPS AN"),
        Card("EXTVER", cfg.extver),
        Card("ARRAYX", Float64(arr_xyz[1])),
        Card("ARRAYY", Float64(arr_xyz[2])),
        Card("ARRAYZ", Float64(arr_xyz[3])),
        Card("ARRNAM", array_name(antennas)),
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

_build_fq_hdu(metadata::ObsArrayMetadata) = _build_fq_hdu(metadata.freq_setup)

function _parse_freqid_from_name(name)
    s = string(name)
    m = match(r"(\d+)\s*$", s)
    return m === nothing ? Int32(1) : Int32(parse(Int, m.captures[1]))
end

function _build_fq_hdu(fs::FrequencySetup)
    if_freqs = collect(fs.channel_freqs) .- fs.ref_freq
    freqid = _parse_freqid_from_name(fs.name)
    base = (
        FRQSEL = Int32[freqid],
        var"IF FREQ" = [if_freqs],
        var"CH WIDTH" = [fs.ch_widths],
        var"TOTAL BANDWIDTH" = [fs.total_bandwidths],
        SIDEBAND = [fs.sidebands],
    )
    data = merge(base, fs.extras)
    cards = Card[Card("EXTNAME", "AIPS FQ"), Card("NO_IF", Int32(length(if_freqs)))]
    return HDU(Bintable, data, cards)
end

function _build_nx_hdu(uvset::UVSet, record_starts::AbstractVector, record_ends::AbstractVector)
    scans = DimensionalData.metadata(uvset).scans
    nscan = length(scans)
    time_center = [Float64(scan.lower + scan.upper) for scan in scans] ./ 48.0
    time_interval = [Float32(scan.upper - scan.lower) for scan in scans] ./ 24.0f0
    nt_data = (
        TIME = time_center,
        var"TIME INTERVAL" = time_interval,
        var"SOURCE ID" = fill(Int32(1), nscan),
        SUBARRAY = fill(Int32(1), nscan),
        var"FREQ ID" = fill(Int32(1), nscan),
        var"START VIS" = Int32.(record_starts),
        var"END VIS" = Int32.(record_ends),
    )
    return HDU(Bintable, nt_data, Card[Card("EXTNAME", "AIPS NX")])
end

function UVData.write_uvfits(output_path, uvset::UVSet)
    src_list = sources(uvset)
    length(src_list) == 1 || error(
        "write_uvfits: UVData is single-source; got sources=$(src_list). " *
            "Use select_source(uvset, name) before writing."
    )

    branches_dict = DimensionalData.branches(uvset)
    isempty(branches_dict) && error("write_uvfits: UVSet has no partitions")

    leaf_list = collect(values(branches_dict))
    sids = [DimensionalData.metadata(l).scan_idx for l in leaf_list]
    issorted(sids) || error("write_uvfits: branches are not in scan_idx order; sids=$(sids)")

    root = DimensionalData.metadata(uvset)
    array_obs = root.array_obs
    msv4_labels = collect(UVData.pol_products(uvset))
    npol = length(msv4_labels)
    nchan = length(array_obs.freq_setup.channel_freqs)

    # Map MSv4 pol order (in-memory) back to whatever AIPS Stokes order the
    # primary_cards specify, so the on-disk layout matches the round-tripped
    # CRVAL/CDELT.
    aips_codes_disk, aips_labels_disk, _, _ = parse_stokes_axis(root.primary_cards, npol)
    pol_perm = if isempty(aips_codes_disk) || aips_labels_disk == msv4_labels
        collect(1:npol)
    else
        [findfirst(==(lab), msv4_labels) for lab in aips_labels_disk]
    end
    any(isnothing, pol_perm) && error("write_uvfits: cannot map MSv4 pols $msv4_labels to AIPS order $aips_labels_disk")

    nrec_total = sum(length(DimensionalData.metadata(l).record_order) for l in leaf_list)
    nrec_total > 0 || error("write_uvfits: UVSet has no records to write")

    fp = first(leaf_list)
    fp_info = DimensionalData.metadata(fp)
    uvw_eltype = eltype(parent(fp[:uvw]))
    date_eltype = eltype(fp_info.date_param)
    date_ncols = size(fp_info.date_param, 2)

    raw_data = zeros(Float32, nrec_total, 3, npol, nchan, 1, 1, 1)
    uu = Vector{uvw_eltype}(undef, nrec_total)
    vv = Vector{uvw_eltype}(undef, nrec_total)
    ww_ = Vector{uvw_eltype}(undef, nrec_total)
    bl_codes = Vector{Int}(undef, nrec_total)
    date_param_cat = Matrix{date_eltype}(undef, nrec_total, date_ncols)
    extras_keys = keys(fp_info.extra_columns)
    extras_eltypes = ntuple(i -> eltype(fp_info.extra_columns[i]), length(extras_keys))
    extras_bufs = ntuple(i -> Vector{extras_eltypes[i]}(undef, nrec_total), length(extras_keys))

    nscan = length(root.scans)
    record_starts = zeros(Int32, nscan)
    record_ends = zeros(Int32, nscan)

    rec_offset = 0
    for leaf in leaf_list
        info = DimensionalData.metadata(leaf)
        sid = info.scan_idx
        bls = info.baselines
        ro = info.record_order
        first_row_in_scan = rec_offset + 1
        _write_records_kernel!(
            raw_data, uu, vv, ww_, bl_codes, date_param_cat,
            parent(leaf[:vis]), parent(leaf[:weights]), parent(leaf[:uvw]),
            bls.unique_codes, ro, info.date_param, rec_offset, pol_perm,
        )
        for (rec_i, _) in enumerate(ro)
            row = rec_offset + rec_i
            for (i, _) in enumerate(extras_keys)
                extras_bufs[i][row] = info.extra_columns[i][rec_i]
            end
        end
        if !isempty(ro) && 1 <= sid <= nscan
            record_starts[sid] = Int32(first_row_in_scan)
            record_ends[sid] = Int32(rec_offset + length(ro))
        end
        rec_offset += length(ro)
    end

    extras_cat = NamedTuple{extras_keys}(extras_bufs)
    primary_data = _build_primary_data(uvset, uu, vv, ww_, bl_codes, date_param_cat, extras_cat, raw_data)
    primary_hdu = HDU(Random, primary_data, copy(root.primary_cards))
    an_hdu = _build_an_hdu(root.antennas, root.array_config, array_obs.freq_setup.ref_freq)
    fq_hdu = _build_fq_hdu(array_obs)
    nx_hdu = _build_nx_hdu(uvset, record_starts, record_ends)

    write(output_path, HDU[primary_hdu, an_hdu, fq_hdu, nx_hdu])
    return output_path
end

function _write_records_kernel!(
        raw_data::AbstractArray{Float32, 7},
        uu::AbstractVector{Tuvw},
        vv::AbstractVector{Tuvw},
        ww_::AbstractVector{Tuvw},
        bl_codes::AbstractVector{Int},
        date_param_cat::AbstractMatrix{Tdate},
        vis_dense::AbstractArray{Tvis, 4},
        w_dense::AbstractArray{Tw, 4},
        uvw_dense::AbstractArray{Tuvw, 3},
        unique_codes,
        record_order::AbstractVector{Tuple{Int, Int}},
        date_param::AbstractMatrix{Tdate},
        rec_offset::Integer,
        pol_perm::AbstractVector{Int},
    ) where {Tvis, Tw, Tuvw, Tdate}
    npol = length(pol_perm)
    nchan = size(vis_dense, 4)
    @inbounds for (rec_i, (ti, bi)) in enumerate(record_order)
        row = rec_offset + rec_i
        for pdisk in 1:npol
            pmem = pol_perm[pdisk]
            for c in 1:nchan
                v = vis_dense[ti, bi, pmem, c]
                raw_data[row, 1, pdisk, c, 1, 1, 1] = real(v)
                raw_data[row, 2, pdisk, c, 1, 1, 1] = imag(v)
                raw_data[row, 3, pdisk, c, 1, 1, 1] = w_dense[ti, bi, pmem, c]
            end
        end
        uu[row] = uvw_dense[ti, bi, 1]
        vv[row] = uvw_dense[ti, bi, 2]
        ww_[row] = uvw_dense[ti, bi, 3]
        bl_codes[row] = round(Int, unique_codes[bi])
        date_param_cat[row, :] .= @view date_param[rec_i, :]
    end
    return nothing
end

end # module
