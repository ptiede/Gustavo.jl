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
    UVSet, UVMetadata, ObsArrayMetadata, FrequencySetup, AbstractFrequencySetup,
    Antenna, AntennaTable, ArrayConfig, BaselineIndex,
    Mount, MountAltAz, MountEquatorial, MountNaismithR, MountNaismithL,
    Integration, Pol, IF, UVW,
    sources, parallactic_mount, elevation_mount, offset_mount,
    array_xyz, array_name, extras,
    decode_baseline,
    channel_freqs, ref_freq, ch_widths, total_bandwidths, sidebands, setup_name


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

    # TODO (Phase 1.5): support multi-row FQ tables. We currently consume
    # only the first FRQSEL and assume nvis_chan == NO_IF for that row.
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

function _build_array_obs_metadata(primary_hdu)
    cards = primary_hdu.cards

    telescope = string(something(card_value(cards, "TELESCOP"), ""))
    instrume = string(something(card_value(cards, "INSTRUME"), ""))
    date_obs = string(something(card_value(cards, "DATE-OBS"), ""))
    equinox = Float32(something(card_value(cards, "EQUINOX"), 2000.0))
    bunit = string(something(card_value(cards, "BUNIT"), "UNCALIB"))

    return ObsArrayMetadata(;
        telescope, instrume, date_obs, equinox, bunit,
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
    array_obs = _build_array_obs_metadata(primary_hdu)
    # Phase 1: a single freq setup per file. Phase 1.5 will read every FQ
    # row and use NX `FREQ ID` (or per-record FRQSEL) to bin records.
    freq_setups = FrequencySetup[_build_frequency_setup(primary_hdu.cards, fq_hdu.data, size(vis_raw, 3))]
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

    _col(nt, prefix) = collect(getproperty(nt, first(filter(k -> startswith(string(k), prefix), propertynames(nt)))))
    uvw_raw = hcat(_col(dt, "UU"), _col(dt, "VV"), _col(dt, "WW"))

    vis = _wrap_int_pol_if(vis_raw, obs_time, msv4_labels, channel_freqs(first(freq_setups)))
    weights = _wrap_int_pol_if(weights_raw, obs_time, msv4_labels, channel_freqs(first(freq_setups)))
    uvw = _wrap_uvw(uvw_raw, obs_time)

    extra_columns = _collect_extra_columns(dt, primary_hdu.cards)
    date_param = ndims(date_raw) == 2 ? Matrix(date_raw) : reshape(collect(date_raw), :, 1)

    # Materialize the NX columns up front: they come back as lazy
    # `DiskArrays`-backed broadcasts. Each NX row defines one MSv4 partition;
    # we bin records by half-open `[lower, upper)` time intervals into a
    # per-record scan label vector. AIPS NX has no SCAN_NUMBER column, so the
    # row index becomes the canonical scan label (string-cast for xradio
    # `ScanArray` shape).
    nx_time = Float64.(collect(nx.TIME))
    nx_dt = Float64.(collect(nx.var"TIME INTERVAL"))
    nx_lower = (nx_time .- nx_dt ./ 2) .* 24
    nx_upper = (nx_time .+ nx_dt ./ 2) .* 24

    # Bin each record into the NX row whose center is nearest. This is
    # robust to:
    #   - degenerate zero-width intervals (single-timestamp leaves where
    #     `lower == upper` collapse to one center).
    #   - Float32→Float64 round-off in the DATE PTYPE column (DATE * 24 can
    #     drift by ~1e-7, which would push a record outside a `[l, u]`
    #     check on the literal interval).
    # Records were pre-binned by NX row on the writer side, so the nearest-
    # center rule is unambiguous.
    nx_centers = (nx_lower .+ nx_upper) ./ 2
    record_scan_name = Vector{String}(undef, length(obs_time))
    if isempty(nx_centers)
        fill!(record_scan_name, "")
    else
        @inbounds for i in eachindex(obs_time)
            t = obs_time[i]
            best_s = 1
            best_d = abs(t - nx_centers[1])
            for s in 2:length(nx_centers)
                d = abs(t - nx_centers[s])
                if d < best_d
                    best_d = d
                    best_s = s
                end
            end
            record_scan_name[i] = string(best_s)
        end
    end
    # Drop records that fall outside any NX row; they cannot be assigned to a
    # leaf and would otherwise pollute the partition.
    valid = findall(!=(""), record_scan_name)
    if length(valid) != length(obs_time)
        obs_time = obs_time[valid]
        record_scan_name = record_scan_name[valid]
        bl_codes = bl_codes[valid]
        vis = vis[Integration = valid]
        weights = weights[Integration = valid]
        uvw = uvw[Integration = valid]
        date_param = date_param[valid, :]
        extra_columns = NamedTuple{keys(extra_columns)}(
            ntuple(i -> extra_columns[i][valid], length(extra_columns))
        )
    end

    unique_codes = sort(unique(bl_codes))
    bl_lookup = Dict(bl => i for (i, bl) in enumerate(unique_codes))
    bl_pairs = decode_baseline.(unique_codes)
    baselines = BaselineIndex(
        bl_codes, bl_pairs, bl_lookup, unique_codes;
        antenna_names = antennas.name
    )

    basename = String(splitext(_basename_of_path(path))[1])

    record_freqid = ones(Int32, length(obs_time))

    return (;
        vis, weights, uvw, obs_time, baselines,
        date_param, extra_columns,
        antennas, array_config, array_obs,
        freq_setups, record_freqid, record_scan_name,
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
        # Materialize: per-scan partition extraction indexes these columns
        # repeatedly; leaving them as lazy DiskArrays makes each `[int_inds]`
        # re-open the FITS file.
        push!(pairs, sym => collect(getproperty(dt, sym)))
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

function _parse_freqid_from_name(name)
    s = string(name)
    m = match(r"(\d+)\s*$", s)
    return m === nothing ? Int32(1) : Int32(parse(Int, m.captures[1]))
end

"""
    _collect_freq_setups(uvset) -> (Vector{FrequencySetup}, Dict{FrequencySetup,Int32})

Walk leaves and gather their unique `FrequencySetup`s in first-seen order
(via `union_frequency_axis`), then build a setup → FRQSEL lookup. The
FRQSEL value is parsed from the setup name when it ends in digits, else
the leaf's 1-based position in the union list. Setup names that share a
parsed FRQSEL are reassigned 1..N to keep FRQSEL unique on disk.
"""
function _collect_freq_setups(uvset::UVSet)
    setups = UVData.union_frequency_axis(uvset)
    parsed = [_parse_freqid_from_name(setup_name(fs)) for fs in setups]
    # If parsed FRQSELs collide, fall back to positional 1..N to guarantee uniqueness.
    freqids = length(unique(parsed)) == length(parsed) ? parsed : Int32.(1:length(setups))
    lookup = Dict{eltype(setups), Int32}()
    for (fs, fid) in zip(setups, freqids)
        lookup[fs] = fid
    end
    return setups, lookup
end

function _build_fq_hdu(setups::AbstractVector{<:FrequencySetup}, freqids::AbstractVector{<:Integer})
    isempty(setups) && error("_build_fq_hdu: empty setup list")
    nif = length(channel_freqs(first(setups)))
    for fs in setups
        length(channel_freqs(fs)) == nif ||
            error("_build_fq_hdu: ragged channel counts across setups not yet supported (Phase 1.5)")
    end
    if_freqs_per_row = [collect(channel_freqs(fs)) .- ref_freq(fs) for fs in setups]
    base = (
        FRQSEL = Int32.(freqids),
        var"IF FREQ" = if_freqs_per_row,
        var"CH WIDTH" = [collect(ch_widths(fs)) for fs in setups],
        var"TOTAL BANDWIDTH" = [collect(total_bandwidths(fs)) for fs in setups],
        SIDEBAND = [collect(sidebands(fs)) for fs in setups],
    )
    # Carry per-row extras only if every setup has the same extras keyset
    # (Phase 1.5 will need per-setup ragged handling here too).
    extras_keys = keys(first(setups).extras)
    if all(keys(fs.extras) == extras_keys for fs in setups)
        extras_per_row = NamedTuple{extras_keys}(
            ntuple(i -> [fs.extras[i] for fs in setups], length(extras_keys))
        )
        data = merge(base, extras_per_row)
    else
        data = base
    end
    cards = Card[Card("EXTNAME", "AIPS FQ"), Card("NO_IF", Int32(nif))]
    return HDU(Bintable, data, cards)
end

# Single-setup convenience kept for callers (and tests) that hand in one fs.
_build_fq_hdu(fs::FrequencySetup) =
    _build_fq_hdu([fs], Int32[_parse_freqid_from_name(fs.name)])

function _build_nx_hdu(
        scan_windows::AbstractVector{Tuple{Float64, Float64}},
        record_starts::AbstractVector, record_ends::AbstractVector,
        freqid_per_scan::AbstractVector{<:Integer},
        subarray_per_scan::AbstractVector{<:Integer} = fill(Int32(1), length(scan_windows)),
    )
    nscan = length(scan_windows)
    time_center = [Float64(lo + hi) / 48.0 for (lo, hi) in scan_windows]
    time_interval = [Float32(hi - lo) / 24.0f0 for (lo, hi) in scan_windows]
    nt_data = (
        TIME = time_center,
        var"TIME INTERVAL" = time_interval,
        var"SOURCE ID" = fill(Int32(1), nscan),
        SUBARRAY = Int32.(subarray_per_scan),
        var"FREQ ID" = Int32.(freqid_per_scan),
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

    # Phase 2: write one AN/FQ HDU and one SUBARRAY=1 column. Multi-subarray
    # write (different AN extvers) is Phase 2.5 — guard explicitly.
    sub_scans = unique(DimensionalData.metadata(l).sub_scan_name for (_, l) in branches_dict)
    sub_scans == [""] || error(
        "write_uvfits: multi-subarray write (sub_scan_name set) is not yet " *
            "supported (Phase 2.5). Got sub_scan_names = $(sub_scans)."
    )

    # Sort leaves by scan-window start so NX rows ascend in time, matching
    # AIPS convention. Within identical start times, fall back to scan_name.
    leaf_list = sort(
        collect(values(branches_dict));
        by = leaf -> (
            UVData.scan_window(leaf)[1],
            DimensionalData.metadata(leaf).scan_name,
        ),
    )

    root = DimensionalData.metadata(uvset)
    msv4_labels = collect(UVData.pol_products(uvset))
    npol = length(msv4_labels)
    setups, freqid_lookup = _collect_freq_setups(uvset)
    nchan = length(channel_freqs(first(setups)))

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

    nscan = length(leaf_list)
    record_starts = zeros(Int32, nscan)
    record_ends = zeros(Int32, nscan)
    freqid_per_scan = ones(Int32, nscan)
    scan_windows = Vector{Tuple{Float64, Float64}}(undef, nscan)

    rec_offset = 0
    for (sid, leaf) in enumerate(leaf_list)
        info = DimensionalData.metadata(leaf)
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
        scan_windows[sid] = UVData.scan_window(leaf)
        if !isempty(ro)
            record_starts[sid] = Int32(first_row_in_scan)
            record_ends[sid] = Int32(rec_offset + length(ro))
            freqid_per_scan[sid] = freqid_lookup[info.freq_setup]
        end
        rec_offset += length(ro)
    end

    extras_cat = NamedTuple{extras_keys}(extras_bufs)
    primary_data = _build_primary_data(uvset, uu, vv, ww_, bl_codes, date_param_cat, extras_cat, raw_data)
    primary_hdu = HDU(Random, primary_data, copy(root.primary_cards))
    # AN table reflects the array nominal — one ref_freq for the array, taken
    # from the first setup. Per-SPW reference frequencies live on each FQ row.
    an_hdu = _build_an_hdu(root.antennas, root.array_config, ref_freq(first(setups)))
    fq_hdu = _build_fq_hdu(setups, [freqid_lookup[fs] for fs in setups])
    nx_hdu = _build_nx_hdu(scan_windows, record_starts, record_ends, freqid_per_scan)

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
