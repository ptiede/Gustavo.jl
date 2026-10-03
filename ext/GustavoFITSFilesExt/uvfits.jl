using FITSFiles
using FITSFiles: HDU
using StructArrays
using Dates: Date, DateTime, datetime2julian
using DimensionalData
using DimensionalData: DimArray
using PolarizedTypes: XPol, YPol, RPol, LPol

import Gustavo.UVData
using Gustavo.UVData:
    UVSet, ObsArrayMetadata, FrequencySetup,
    Antenna, AntennaTable, BaselineIndex,
    MountAltAz, MountEquatorial, MountNasmythR, MountNasmythL,
    MountBWGR, MountBWGL, MountXY, MountOrbiting,
    Polarization, Frequency, UVW,
    extras,
    channel_freqs, ref_freq, ch_widths, total_bandwidths, sidebands, setup_name

const POLBASIS = Union{RPol, LPol, XPol, YPol}

# AIPS UVFITS BASELINE-column convention: pack `(a, b)` antenna indices
# as `bl = a*256 + b`. Caps the array at 255 antennas. Lives in the FITS
# extension only; format-neutral code in `src/` speaks `(a, b)` tuples.
_decode_aips_baseline(bl::Integer)::Tuple{Int, Int} = (bl ÷ 256, bl % 256)


# AIPS POLTYA/POLTYB letter → PolarizedTypes
function poltype(type)
    type == "R" && return RPol()
    type == "L" && return LPol()
    type == "X" && return XPol()
    type == "Y" && return YPol()
    error("Unsupported polarization type: $type")
end


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

# Product order written to a leaf; a subset keeps this relative order.
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


# AIPS MNTSTA codes 0–7, shared by the UVFITS AN and FITS-IDI ARRAY_GEOMETRY
# tables; 6 and 7 are the beam-waveguide mounts (Dodson & Rioja,
# arXiv:2210.13381), which pyuvdata codes differently. `offset` is the mount's
# axis offset in meters.
const _MNTSTA_MOUNTS = (
    MountAltAz, MountEquatorial, MountOrbiting, MountXY, MountNasmythR, MountNasmythL,
    MountBWGR, MountBWGL,
)

function mnt_codes_to_type(code, offset)
    0 <= code < length(_MNTSTA_MOUNTS) ||
        throw(ArgumentError("MNTSTA $code names no mount; AIPS defines codes 0–7"))
    return _MNTSTA_MOUNTS[code + 1](offset)
end

# The UVFITS AN table states one axis offset per antenna, along the station's x.
_uvfits_axis_offset(staxof) = (Float64(staxof), 0.0, 0.0)


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
    aips_codes::Vector{Int} = Int.(round.(crval .+ cdelt .* ((1:npol) .- crpix)))
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
    # Materialize eagerly via `collect`: the AN HDU column may come
    # back as a lazy `LazyFieldArray`, and downstream `hash`/`==` on
    # the antenna struct iterates the per-antenna POLCAL vectors —
    # which crashes on lazy 0-length chunked arrays. We avoid
    # `Matrix{T}(::LazyFieldArray)` because that path goes through
    # DiskArrays' broadcast machinery and also fails on empty fields.
    raw_mat = collect(raw)
    if raw_mat isa AbstractMatrix
        return [Float32.(raw_mat[i, :]) for i in 1:nant]
    end
    return [Float32.(collect(x)) for x in raw_mat]
end

function _build_antenna_table(an_hdu)
    cards = an_hdu.cards
    an = an_hdu.data

    clean(s) = filter(c -> isascii(c) && isprint(c) && !isspace(c), string(s))

    names = clean.(collect(an.ANNAME))
    nant = length(names)
    xyz_raw = collect(an.STABXYZ)
    mount_raw = collect(an.MNTSTA)
    staxof_raw::Vector{Float32} = hasproperty(an, :STAXOF) ? collect(an.STAXOF) : fill(0.0f0, nant)
    mnts = mnt_codes_to_type.(mount_raw, _uvfits_axis_offset.(staxof_raw))
    poltya_raw = hasproperty(an, :POLTYA) ? collect(an.POLTYA) : fill("R", nant)
    poltya::Vector{POLBASIS} = poltype.(poltya_raw)
    poltyb_raw = hasproperty(an, :POLTYB) ? collect(an.POLTYB) : fill("L", nant)
    poltyb::Vector{POLBASIS} = poltype.(poltyb_raw)
    polaa_raw::Vector{Float32} = hasproperty(an, :POLAA) ? collect(an.POLAA) : fill(0.0f0, nant)
    polab_raw::Vector{Float32} = hasproperty(an, :POLAB) ? collect(an.POLAB) : fill(0.0f0, nant)
    pol_angles::Vector{Tuple{Float32, Float32}} = tuple.(deg2rad.(polaa_raw), deg2rad.(polab_raw))

    # `STABXYZ` is measured from the ARRAYX/Y/Z array center.
    center = [Float64(something(card_value(cards, k), 0.0)) for k in ("ARRAYX", "ARRAYY", "ARRAYZ")]
    station_xyz::Vector{Vector{Float64}} = [center .+ Float64.(xyz_raw[i, :]) for i in eachindex(names)]
    nominal_basis::Vector{Tuple{POLBASIS, POLBASIS}} = tuple.(poltya, poltyb)

    antennas = [
        Antenna(;
            name = names[i],
            station_xyz = station_xyz[i],
            mount = mnts[i],
            nominal_basis = nominal_basis[i],
            pol_angles = pol_angles[i],
        )
            for i in 1:nant
    ]

    arrnam::String = string(something(card_value(cards, "ARRNAM"), ""))

    ant_extras = (;
        POLCALA = _split_per_antenna_polcal(an, :POLCALA, nant),
        POLCALB = _split_per_antenna_polcal(an, :POLCALB, nant),
        _collect_an_extras(an)...,
    )

    ant_table = AntennaTable(StructArray(antennas), arrnam, ant_extras)
    return ant_table
end

const _FQ_MANDATORY_COLS = Set(
    [
        :FRQSEL, Symbol("IF FREQ"), Symbol("CH WIDTH"),
        Symbol("TOTAL BANDWIDTH"), :SIDEBAND,
    ]
)

function _collect_fq_extras(fq, r::Integer)
    pairs = Pair{Symbol, Any}[]
    for sym in propertynames(fq)
        sym in _FQ_MANDATORY_COLS && continue
        col = getproperty(fq, sym)
        # FQ extras come as either a per-row vector or an (nrows, ncol) matrix.
        val = ndims(col) == 1 ? col[r] : vec(col[r, :])
        push!(pairs, sym => val)
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

"""
    _build_frequency_setups(cards, fq, nvis_chan)
        -> (Vector{FrequencySetup}, Vector{Int32})

Read every row of the FQ HDU. Each row becomes a `FrequencySetup` with
MSv4-flavored `setup_name = "spw_<r-1>"`; the on-disk AIPS FRQSEL is
preserved in `extras.frqsel`.

Returns the dense vector of setups plus the parallel `frqsels` vector
(per-row FRQSEL values). The setups vector is indexed 1..nrows; if the
on-disk FRQSEL column is sparse, callers must densify per-record SPW
indices via `frqsels`.

Errors when channel counts differ across rows (ragged setups deferred).
"""
function _build_frequency_setups(cards, fq, nvis_chan)
    ref_freq_v::Float64 = Float64(_find_freq_axis(cards))

    if_freqs_all::Matrix{Float64} = collect(getproperty(fq, Symbol("IF FREQ")))
    ch_widths_all::Matrix{Float64} = collect(getproperty(fq, Symbol("CH WIDTH")))
    total_bw_all::Matrix{Float64} = collect(getproperty(fq, Symbol("TOTAL BANDWIDTH")))
    sidebands_all::Matrix{Float64} = collect(getproperty(fq, :SIDEBAND))
    nrows::Int = ndims(if_freqs_all) == 1 ? 1 : size(if_freqs_all, 1)
    nif::Int = ndims(if_freqs_all) == 1 ? length(if_freqs_all) : size(if_freqs_all, 2)
    nif == nvis_chan ||
        error("FQ table reports $nif IFs but vis has $nvis_chan channels")

    frqsels::Vector{Int32} = if hasproperty(fq, :FRQSEL)
        round.(Int32, collect(getproperty(fq, :FRQSEL)))
    else
        round.(Int32, collect(1:nrows))
    end

    _row(M, r) = ndims(M) == 1 ? collect(M) : vec(M[r, :])

    setups = FrequencySetup[]
    for r in 1:nrows
        if_freqs_r = _row(if_freqs_all, r)
        ch_widths_r = _row(ch_widths_all, r)
        total_bw_r = _row(total_bw_all, r)
        sidebands_r = _row(sidebands_all, r)
        length(if_freqs_r) == nif ||
            error("FQ row $r has $(length(if_freqs_r)) IFs; expected $nif (ragged setups not yet supported)")

        extras = merge(_collect_fq_extras(fq, r), (; frqsel = frqsels[r]))
        push!(
            setups, FrequencySetup(;
                name = string("spw_", r - 1),
                ref_freq = ref_freq_v,
                channel_freqs = ref_freq_v .+ if_freqs_r,
                ch_widths = ch_widths_r,
                total_bandwidths = total_bw_r,
                sidebands = sidebands_r,
                extras,
            )
        )
    end
    return setups, frqsels
end

function _build_array_obs_metadata(primary_hdu, an_hdu = nothing)
    cards = primary_hdu.cards

    telescope::String = string(something(card_value(cards, "TELESCOP"), ""))
    instrume::String = string(something(card_value(cards, "INSTRUME"), ""))
    date_obs::String = string(something(card_value(cards, "DATE-OBS"), ""))
    equinox::Float32 = Float32(something(card_value(cards, "EQUINOX"), 2000.0f0))
    bunit::String = string(something(card_value(cards, "BUNIT"), "UNCALIB"))

    # Time-system / Earth-orientation / coord-frame fields live on the AIPS
    # AN HDU header (Memo 117 §4.1). When no AN HDU is supplied (synthetic
    # path), defaults kick in via the kwarg constructor.
    if an_hdu === nothing
        return ObsArrayMetadata(;
            telescope, instrume, date_obs, equinox, bunit,
            extras = _collect_obs_card_extras(cards),
        )
    end


    an_cards = an_hdu.cards

    rdate::String = string(something(card_value(an_cards, "RDATE"), ""))
    gst_iat0::Float32 = Float32(something(card_value(an_cards, "GSTIA0"), 0.0))
    earth_rot_rate::Float32 = Float32(something(card_value(an_cards, "DEGPDY"), 360.0))
    ut1utc::Float32 = Float32(something(card_value(an_cards, "UT1UTC"), 0.0))
    polarx::Float32 = Float32(something(card_value(an_cards, "POLARX"), 0.0))
    polary::Float32 = Float32(something(card_value(an_cards, "POLARY"), 0.0))
    datutc::Float32 = Float32(something(card_value(an_cards, "DATUTC"), 0.0))
    time_sys::String = string(something(card_value(an_cards, "TIMSYS"), "UTC"))
    frame::String = string(something(card_value(an_cards, "FRAME"), "ITRF"))
    xyzhand::String = string(something(card_value(an_cards, "XYZHAND"), "RIGHT"))
    poltype::String = string(something(card_value(an_cards, "POLTYPE"), ""))


    return ObsArrayMetadata(;
        telescope, instrume, date_obs, equinox, bunit,
        rdate, gst_iat0, earth_rot_rate, ut1utc, polarx, polary, datutc, time_sys,
        frame, xyzhand, poltype,
        extras = _collect_obs_card_extras(cards),
    )
end

function _build_source_info(primary_hdu)
    cards = primary_hdu.cards
    object::String = string(something(card_value(cards, "OBJECT"), ""))
    ra::Float64 = Float64(something(card_value(cards, "OBSRA"), 0.0))
    dec::Float64 = Float64(something(card_value(cards, "OBSDEC"), 0.0))
    if ra == 0.0
        ra = Float64(something(_find_crval(cards, "RA"), 0.0))
    end
    if dec == 0.0
        dec = Float64(something(_find_crval(cards, "DEC"), 0.0))
    end
    # AIPS UVFITS stores OBSRA/OBSDEC and the RA/DEC axis CRVALs in DEGREES
    # (Memo 117 §3.1.1); Gustavo's internal source coordinates are radians.
    return (; source_name = object, ra = deg2rad(ra), dec = deg2rad(dec))
end

function _find_crval(cards, ctype_prefix)
    needle = uppercase(ctype_prefix)
    i = _find_axis(cards, c -> startswith(c, needle))
    isnothing(i) && return nothing
    return card_value(cards, "CRVAL$i")
end


# ── Read path ───────────────────────────────────────────────────────────────

UVData.load_uvfits(path; element_type::Union{Nothing, Type} = nothing) =
    UVSet(_load_uvfits_flat(path; element_type))

# Slack on the NX window match. The DATE PTYPE's sub-day fraction is Float32,
# which resolves ~5 ms of a day, so a record's decoded epoch and the NX window
# it belongs to can disagree by that much after a round trip.
const _NX_TIME_TOL_SECONDS = 1.0e-2

function _assign_records_to_nx_rows_by_time(obs_time, nx_lower, nx_upper)
    nrec = length(obs_time)
    rows = zeros(Int, nrec)
    isempty(nx_lower) && return rows

    perm = sortperm(nx_lower)
    lower_s = nx_lower[perm]
    upper_s = nx_upper[perm]
    center_s = (lower_s .+ upper_s) ./ 2

    @inbounds for i in eachindex(obs_time)
        t = obs_time[i]
        pos = searchsortedlast(lower_s, t)

        assigned = 0
        if 1 <= pos <= length(lower_s)
            lo = lower_s[pos] - _NX_TIME_TOL_SECONDS
            hi = upper_s[pos] + _NX_TIME_TOL_SECONDS
            if lo <= t <= hi
                assigned = perm[pos]
            end
        end
        if assigned == 0 && pos + 1 <= length(lower_s)
            lo = lower_s[pos + 1] - _NX_TIME_TOL_SECONDS
            hi = upper_s[pos + 1] + _NX_TIME_TOL_SECONDS
            if lo <= t <= hi
                assigned = perm[pos + 1]
            end
        end

        if assigned == 0
            best_j = clamp(pos, 1, length(center_s))
            best_d = abs(t - center_s[best_j])
            for j in max(1, pos - 2):min(length(center_s), pos + 2)
                d = abs(t - center_s[j])
                if d < best_d
                    best_d = d
                    best_j = j
                end
            end
            assigned = perm[best_j]
        end

        rows[i] = assigned
    end
    return rows
end

function _assign_records_to_nx_rows(
        obs_time, nx_lower, nx_upper;
        nx_start_vis = nothing, nx_end_vis = nothing,
    )
    nrec = length(obs_time)
    if nx_start_vis !== nothing && nx_end_vis !== nothing && length(nx_start_vis) == length(nx_end_vis)
        rows = zeros(Int, nrec)
        complete = true
        @inbounds for s in eachindex(nx_start_vis)
            lo = Int(round(nx_start_vis[s]))
            hi = Int(round(nx_end_vis[s]))
            if !(1 <= lo <= hi <= nrec) || any(!=(0), @view(rows[lo:hi]))
                complete = false
                break
            end
            rows[lo:hi] .= s
        end
        complete && all(!=(0), rows) && return rows
    end
    return _assign_records_to_nx_rows_by_time(obs_time, nx_lower, nx_upper)
end

_is_lazy_random(data) = data isa FITSFiles.LazyStructuredData && data.hdu_type === FITSFiles.Random

function _load_uvfits_flat(path; element_type::Union{Nothing, Type} = nothing)
    fid = FITSFiles.fits(path)
    primary_hdu = fid[1]
    # Bypass FITSFiles' per-record Vector{Float32} allocation when the
    # primary HDU is lazy random-group data.
    primary_lazy = getfield(primary_hdu, :data)
    dt = _is_lazy_random(primary_lazy) ?
        _fast_random_read(primary_lazy) : primary_hdu.data
    # Collect every AN HDU (filtered by EXTNAME=AIPS AN) — multi-AN-extver
    # files carry one AN table per subarray.
    an_hdus = HDU[]
    fq_hdu = nothing
    nx = nothing
    for hdu in fid[2:end]
        ext = string(something(card_value(hdu.cards, "EXTNAME"), ""))
        if ext == "AIPS AN"
            push!(an_hdus, hdu)
        elseif ext == "AIPS FQ"
            fq_hdu = hdu
        elseif ext == "AIPS NX"
            nx = hdu.data
        end
    end
    isempty(an_hdus) && error("load_uvfits: no AIPS AN HDU found in $(path)")
    fq_hdu === nothing && error("load_uvfits: no AIPS FQ HDU found in $(path)")
    nx === nothing && error("load_uvfits: no AIPS NX HDU found in $(path)")
    an_hdu = first(an_hdus)

    dim1 = findall(==(1), size(dt.data))
    raw = dropdims(dt.data, dims = Tuple(dim1))
    element_type === nothing || element_type <: AbstractFloat || throw(
        ArgumentError(
            "load_uvfits: element_type must be a real float type, got $(element_type)",
        ),
    )
    T = element_type === nothing ? eltype(raw) : element_type

    # Read the visibilities verbatim. AIPS random-groups UVFITS shares Gustavo's
    # internal phase convention — both are the AIPS/CASA/casacore sense, which is
    # the conjugate of FITS-IDI's V = ⟨E_a1 · conj(E_a2)⟩ (AIPS Memo 114r §2.1;
    # casacore FitsIDItoMS.cc: "FITS-IDI convention is conjugate of AIPS and CASA
    # convention"). (u,v,w) and the 256·a1+a2 baseline codes agree too, so nothing
    # on this boundary is transformed.
    vis_raw::Array{Complex{T}, 3} = complex.(raw[:, 1, :, :], raw[:, 2, :, :])
    weights_raw::Array{T, 3} = raw[:, 3, :, :]

    antenna_tables = AntennaTable[_build_antenna_table(h) for h in an_hdus]
    antennas = first(antenna_tables)
    array_obs = _build_array_obs_metadata(primary_hdu, an_hdu)
    freq_setups, fq_frqsels = _build_frequency_setups(
        primary_hdu.cards, fq_hdu.data, size(vis_raw, 3)
    )
    src_info = _build_source_info(primary_hdu)

    aips_codes, aips_labels, msv4_labels, perm = parse_stokes_axis(primary_hdu.cards, size(vis_raw, 2))
    _check_stokes_vs_poltya(aips_codes, antennas)
    vis_raw = vis_raw[:, perm, :]
    weights_raw = weights_raw[:, perm, :]

    # AIPS UVFITS stores DATE as two PTYPE columns. Different writers use
    # different splits:
    #   * AIPS strict: col1 = integer JD (~2.46e6 for 2022 data), col2 =
    #     fractional day. Sum is full JD.
    #   * Gustavo / round-tripped: col1 = floor(days_since_RDATE) (~0..few),
    #     col2 = fractional remainder. Sum is days_since_RDATE.
    # We lift each column to Float64 *before* combining (Float32 ULP at JD
    # magnitude is ~0.25 days, which would collapse sub-second timestamps),
    # then detect whether the pair is a full JD or already RDATE-relative and
    # produce **seconds since `UVData.JD_UNIX_EPOCH`**, the `Ti` convention.
    # The two columns stay separate through the epoch subtraction: a Float64
    # resolves only ~40 µs at Julian-Day magnitude, so summing them first
    # would discard the split's whole purpose.
    rdate_jd = _rdate_jd_or_zero(array_obs.rdate)
    date_raw = collect(dt.DATE)
    date_hi, date_lo = if ndims(date_raw) == 2
        Float64.(@view date_raw[:, 1]), Float64.(@view date_raw[:, 2])
    else
        Float64.(date_raw), zeros(Float64, length(date_raw))
    end
    # Detect AIPS-strict full-JD (large) vs already-RDATE-relative (small).
    # Full JD has been > 2.4e6 since ~1858 (the MJD reference); plausible
    # days-since-RDATE is < ~1e3 even for decade-long campaigns. A value
    # in [1e3, 2.4e6) cannot come from either convention and indicates a
    # corrupted file or unsupported writer — error rather than guess.
    obs_time::Vector{Float64} = if isempty(date_hi)
        Float64[]
    else
        raw_max = maximum(date_hi .+ date_lo)
        if raw_max >= 2.4e6
            ((date_hi .- UVData.JD_UNIX_EPOCH) .+ date_lo) .* 86400.0
        elseif raw_max <= 1.0e3
            rdate_jd == 0.0 && error(
                "load_uvfits: DATE PTYPE values are days relative to RDATE " *
                    "(max(col1+col2) = $raw_max), but the AN HDU carries no " *
                    "usable RDATE, so the records have no absolute epoch."
            )
            UVData.jd_to_unix(rdate_jd) .+ (date_hi .+ date_lo) .* 86400.0
        else
            error(
                "load_uvfits: ambiguous DATE PTYPE values — max(col1+col2) = $raw_max " *
                    "is neither plausibly a full Julian Day (≥2.4e6) nor a " *
                    "days-since-RDATE offset (≤1e3). The file may be corrupted " *
                    "or written with an unsupported convention."
            )
        end
    end
    bl_codes::Vector{Int} = round.(Int, collect(dt.BASELINE))

    _col(nt, prefix) = collect(getproperty(nt, first(filter(k -> startswith(string(k), prefix), propertynames(nt)))))
    # (u,v,w) are read verbatim: every FITS flavour Gustavo reads shares one
    # baseline-coordinate convention (u,v,w in light-seconds, coord = r_a1 − r_a2;
    # AIPS Memo 117r §4.1.2 = Memo 114r §4.1.2 word for word). The FITS-IDI↔AIPS
    # difference is purely the visibility conjugation, which FITS-IDI carries and
    # UVFITS does not.
    uvw_raw = hcat(_col(dt, "UU"), _col(dt, "VV"), _col(dt, "WW"))

    cfq::Vector{Float64} = channel_freqs(first(freq_setups))
    dims = (Ti(obs_time), Polarization(msv4_labels), Frequency(cfq))

    vis, weights, uvw = _build_arrays(vis_raw, weights_raw, uvw_raw, dims)

    extra_columns = _collect_extra_columns(dt, primary_hdu.cards)

    # Materialize the NX columns up front: they come back as lazy
    # `DiskArrays`-backed broadcasts. Each NX row defines one MSv4 partition.
    # AIPS NX has no SCAN_NUMBER column, so the row index becomes the
    # canonical scan label (string-cast for xradio `ScanArray` shape).
    nx_time::Vector{Float64} = Float64.(collect(nx.TIME))
    nx_dt::Vector{Float64} = Float64.(collect(nx.var"TIME INTERVAL"))
    # NX TIME / TIME INTERVAL are days-since-RDATE; shift onto `obs_time`'s
    # absolute seconds.
    nx_rdate_unix = UVData.jd_to_unix(rdate_jd)
    nx_lower::Vector{Float64} = nx_rdate_unix .+ (nx_time .- nx_dt ./ 2) .* 86400.0
    nx_upper::Vector{Float64} = nx_rdate_unix .+ (nx_time .+ nx_dt ./ 2) .* 86400.0
    nx_freqid::Vector{Int32} = hasproperty(nx, Symbol("FREQ ID")) ?
        round.(Int32, collect(nx.var"FREQ ID")) : ones(Int32, length(nx_time))
    nx_subarray::Vector{Int32} = hasproperty(nx, :SUBARRAY) ?
        round.(Int32, collect(nx.SUBARRAY)) : ones(Int32, length(nx_time))
    nx_start_vis = hasproperty(nx, Symbol("START VIS")) ?
        collect(getproperty(nx, Symbol("START VIS"))) : nothing
    nx_end_vis = hasproperty(nx, Symbol("END VIS")) ?
        collect(getproperty(nx, Symbol("END VIS"))) : nothing

    # Prefer explicit NX record ranges when present: they're exact and avoid
    # an O(nrecord * nscan) nearest-center search on large files. Fall back to
    # time-based assignment for files that omit START/END VIS.
    record_nx_row = _assign_records_to_nx_rows(
        obs_time, nx_lower, nx_upper;
        nx_start_vis = nx_start_vis, nx_end_vis = nx_end_vis,
    )
    record_scan_name = [r == 0 ? "" : string(r) for r in record_nx_row]
    valid = findall(!=(""), record_scan_name)
    if length(valid) != length(obs_time)
        obs_time = obs_time[valid]
        record_scan_name = record_scan_name[valid]
        record_nx_row = record_nx_row[valid]
        bl_codes = bl_codes[valid]
        vis = vis[Ti = valid]
        weights = weights[Ti = valid]
        uvw = uvw[Ti = valid]
        extra_columns = NamedTuple{keys(extra_columns)}(
            ntuple(i -> extra_columns[i][valid], length(extra_columns))
        )
    end

    bl_pairs_per_record::Vector{Tuple{Int, Int}} = _decode_aips_baseline.(bl_codes)
    bl_pairs::Vector{Tuple{Int, Int}} = sort(unique(bl_pairs_per_record))
    baselines = BaselineIndex(
        bl_pairs_per_record, bl_pairs;
        antenna_names = antennas.name::Vector{String},
    )

    basename = String(splitext(_basename_of_path(path))[1])

    # Per-record SPW index: prefer a per-record PTYPE (FREQSEL / FREQID),
    # else propagate the per-NX-row FREQ ID column, else default to 1.
    # After densification, indices are 1..length(freq_setups).
    nfreq = length(freq_setups)
    record_spw_index::Vector{Int32} = if hasproperty(dt, :FREQSEL)
        round.(Int32, collect(getproperty(dt, :FREQSEL)))
    elseif hasproperty(dt, :FREQID)
        round.(Int32, collect(getproperty(dt, :FREQID)))
    elseif length(nx_freqid) > 0
        Int32[nx_freqid[r] for r in record_nx_row]
    else
        ones(Int32, length(obs_time))
    end
    if hasproperty(dt, :FREQSEL) || hasproperty(dt, :FREQID) || length(nx_freqid) > 0
        # Densify FRQSEL slots to 1..nrows index when the FQ FRQSELs are sparse.
        if fq_frqsels != 1:nfreq
            remap = Dict{Int32, Int32}()
            for (i, f) in enumerate(fq_frqsels)
                remap[f] = Int32(i)
            end
            record_spw_index = [
                get(remap, Int32(s)) do
                    error("record FRQSEL $s not present in FQ table $(fq_frqsels)")
                end for s in record_spw_index
            ]
        end
    end
    if nfreq > 1 && all(==(record_spw_index[1]), record_spw_index)
        @warn "load_uvfits: $nfreq FQ rows but every record reports the same SPW " *
            "index $(record_spw_index[1]); resulting UVSet will have one leaf per scan."
    end

    # Per-record subarray index. AIPS UVFITS rarely emits a per-record
    # SUBARRAY PTYPE; the NX SUBARRAY column is the canonical source.
    record_subarray_index::Vector{Int32} = if hasproperty(dt, :SUBARRAY)
        round.(Int32, collect(getproperty(dt, :SUBARRAY)))
    elseif length(nx_subarray) > 0
        Int32[nx_subarray[r] for r in record_nx_row]
    else
        ones(Int32, length(obs_time))
    end

    return (;
        vis, weights, uvw, obs_time, baselines,
        extra_columns,
        antenna_tables, array_obs,
        freq_setups, record_spw_index, record_scan_name,
        record_subarray_index,
        pol_labels = msv4_labels,
        aips_pol_codes = aips_codes,
        source_name = src_info.source_name,
        ra = src_info.ra, dec = src_info.dec,
        basename = basename,
    )
end

function _build_arrays(vis_raw, weights_raw, uvw_raw, dims)
    vis = DimArray(vis_raw, dims)
    weights = DimArray(weights_raw, dims)
    uvw = DimArray(uvw_raw, (dims[1], UVW(["U", "V", "W"])))
    return vis, weights, uvw
end

_basename_of_path(path) = isempty(path) ? "uvfits" : Base.basename(String(path))

# Bypass FITSFiles' per-record `Vector{Float32}` allocation by streaming
# the entire random-group data section into a single buffer. Returns a
# NamedTuple with the same key shape as `read(io, ::Type{Random}, …)` on
# the same file: one entry per unique PTYPE name (Vector for unique
# names, Matrix for duplicates) plus `:data` of shape `(N, format.shape…)`.
#
# The original FITSFiles path allocates one `Vector{Float32}` per record
# (~50 M allocs / 2 GiB on a 100k-record EHT file). The bulk read here
# is a single `read!` into a `Vector{T}` of length
# `N * (P + prod(shape))`, plus an in-place `bswap` pass.
# The two-method split is a function barrier: the body runs with `T` concrete.
function _fast_random_read(lazy::FITSFiles.LazyStructuredData)
    T = lazy.format.type
    T === Float32 || T === Float64 ||
        error("_fast_random_read: only Float32/Float64 random groups supported (got $T)")
    return _fast_random_read(T, lazy)
end

function _fast_random_read(::Type{T}, lazy::FITSFiles.LazyStructuredData) where {T <: AbstractFloat}
    fmt = lazy.format
    fields = lazy.fields::AbstractVector{<:FITSFiles.AbstractField}
    P = fmt.param::Int
    N = fmt.group::Int
    leng_data = prod(fmt.shape)::Int
    L = P + leng_data
    total = N * L
    buf = Vector{T}(undef, total)
    open(lazy.filnam) do io
        seek(io, lazy.begpos)
        read!(io, buf)
    end
    @inbounds @simd for i in eachindex(buf)
        buf[i] = ntoh(buf[i])
    end
    # View the buffer as `(L, N)`: column j holds the j-th record laid
    # out as `[PTYPE_1, …, PTYPE_P, data_1, …, data_leng_data]`.
    rec = reshape(buf, L, N)

    data_field = fields[end]
    # Permute (leng_data, N) view → (N, leng_data) materialised matrix
    # via blocked transpose (Base's `permutedims` handles cache locality
    # well on 2-D), then reshape to (N, shape...). Copying here decouples
    # the data block from the PTYPE block so `buf` can be freed.
    data_view = view(rec, (P + 1):L, :)
    data_perm = permutedims(data_view, (2, 1))
    data_block = reshape(data_perm, N, fmt.shape...)
    # Apply BZERO/BSCALE if present (canonical UVFITS files have unity).
    if !ismissing(data_field.zero) && !ismissing(data_field.scale)
        if data_field.zero != 0 || data_field.scale != 1
            @inbounds @simd for i in eachindex(data_block)
                data_block[i] = data_field.zero + data_field.scale * data_block[i]
            end
        end
    end

    # Group PTYPE columns by name (duplicate names → Matrix; unique → Vector).
    name_indices = Dict{String, Vector{Int}}()
    name_order = String[]
    for j in 1:P
        name = String(fields[j].name)
        if !haskey(name_indices, name)
            name_indices[name] = Int[]
            push!(name_order, name)
        end
        push!(name_indices[name], j)
    end
    pairs = Pair{Symbol, Any}[]
    for name in name_order
        ndx = name_indices[name]
        col = if length(ndx) == 1
            fld = fields[ndx[1]]
            v = Vector{T}(undef, N)
            @inbounds for j in 1:N
                v[j] = rec[ndx[1], j]
            end
            if !ismissing(fld.zero) && !ismissing(fld.scale) &&
                    (fld.zero != 0 || fld.scale != 1)
                # Scale in the promoted type, as FITSFiles does.
                fld.zero .+ fld.scale .* v
            else
                v
            end
        else
            m = Matrix{T}(undef, N, length(ndx))
            @inbounds for (k, fi) in enumerate(ndx)
                fld = fields[fi]
                for j in 1:N
                    m[j, k] = rec[fi, j]
                end
                if !ismissing(fld.zero) && !ismissing(fld.scale) &&
                        (fld.zero != 0 || fld.scale != 1)
                    for j in 1:N
                        m[j, k] = fld.zero + fld.scale * m[j, k]
                    end
                end
            end
            m
        end
        push!(pairs, Symbol(name) => col)
    end
    push!(pairs, :data => data_block)
    return (; pairs...)
end

# Parse `array_obs.rdate` (e.g. "2022-01-01") to a Julian Day at 0h UT.
# Anchors the AIPS NX table, whose times are days relative to RDATE.
# Returns 0.0 when the string is empty or unparseable.
function _rdate_jd_or_zero(rdate_str::AbstractString)
    isempty(rdate_str) && return 0.0
    return try
        datetime2julian(DateTime(Date(rdate_str)))
    catch
        0.0
    end
end

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

_wrap_uvw(arr, obs_time) = DimArray(arr, (Ti(obs_time), UVW(["U", "V", "W"])))

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
