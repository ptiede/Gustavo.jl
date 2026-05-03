module GustavoFITSFilesExt

using FITSFiles
using FITSFiles: HDU, Random, Bintable, Card
using StructArrays
using LinearAlgebra: Diagonal
using Dates: Date, DateTime, datetime2julian
using DimensionalData
using DimensionalData: DimArray
using PolarizedTypes: CirBasis, LinBasis, XPol, YPol, RPol, LPol

import Gustavo.UVData
using Gustavo.UVData:
    UVSet, UVMetadata, ObsArrayMetadata, FrequencySetup, AbstractFrequencySetup,
    Antenna, AntennaTable, BaselineIndex,
    Mount, MountAltAz, MountEquatorial, MountNaismithR, MountNaismithL,
    Integration, Pol, Frequency, UVW,
    sources, parallactic_mount, elevation_mount, offset_mount,
    array_xyz, array_name, extras,
    channel_freqs, ref_freq, ch_widths, total_bandwidths, sidebands, setup_name,
    nchannels

const POLBASIS = Union{RPol, LPol, XPol, YPol}

# AIPS UVFITS BASELINE-column convention: pack `(a, b)` antenna indices
# as `bl = a*256 + b`. Caps the array at 255 antennas. Lives in the FITS
# extension only; format-neutral code in `src/` speaks `(a, b)` tuples.
_decode_aips_baseline(bl::Integer)::Tuple{Int, Int} = (bl ÷ 256, bl % 256)
function _encode_aips_baseline(a::Integer, b::Integer)
    (1 <= a < 256 && 1 <= b < 256) || error(
        "AIPS UVFITS BASELINE column packs (a, b) as a*256 + b; " *
            "antenna index $((a, b)) exceeds the 255-antenna limit. " *
            "Use BLN_NUM-style encoding (not yet supported)."
    )
    return Int32(a * 256 + b)
end


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
    mnts = mnt_codes_to_type.(mount_raw, staxof_raw)
    poltya_raw = hasproperty(an, :POLTYA) ? collect(an.POLTYA) : fill("R", nant)
    poltya::Vector{POLBASIS} = poltype.(poltya_raw)
    poltyb_raw = hasproperty(an, :POLTYB) ? collect(an.POLTYB) : fill("L", nant)
    poltyb::Vector{POLBASIS} = poltype.(poltyb_raw)
    polaa_raw::Vector{Float32} = hasproperty(an, :POLAA) ? collect(an.POLAA) : fill(0.0f0, nant)
    polab_raw::Vector{Float32} = hasproperty(an, :POLAB) ? collect(an.POLAB) : fill(0.0f0, nant)
    pol_angles::Vector{Tuple{Float32, Float32}} = tuple.(Float32.(polaa_raw), Float32.(polab_raw))

    response::Vector{Diagonal{ComplexF32, Vector{ComplexF32}}} = [Diagonal(ones(ComplexF32, 2)) for _ in 1:nant]
    station_xyz::Vector{Vector{Float64}} = [Float64.(xyz_raw[i, :]) for i in eachindex(names)]
    nominal_basis::Vector{Tuple{POLBASIS, POLBASIS}} = tuple.(poltya, poltyb)

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

    arrayx::Float64 = Float64(something(card_value(cards, "ARRAYX"), 0.0))
    arrayy::Float64 = Float64(something(card_value(cards, "ARRAYY"), 0.0))
    arrayz::Float64 = Float64(something(card_value(cards, "ARRAYZ"), 0.0))
    arrnam::String = string(something(card_value(cards, "ARRNAM"), ""))

    ant_extras = (;
        POLCALA = _split_per_antenna_polcal(an, :POLCALA, nant),
        POLCALB = _split_per_antenna_polcal(an, :POLCALB, nant),
        _collect_an_extras(an)...,
    )

    ant_table = AntennaTable(StructArray(antennas), (arrayx, arrayy, arrayz), arrnam, ant_extras)
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
preserved in `extras.frqsel` so write paths can recover it.

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
    return (; source_name = object, ra = ra, dec = dec)
end

function _find_crval(cards, ctype_prefix)
    needle = uppercase(ctype_prefix)
    i = _find_axis(cards, c -> startswith(c, needle))
    isnothing(i) && return nothing
    return card_value(cards, "CRVAL$i")
end

# ── Primary-card stash ──────────────────────────────────────────────────────
#
# Primary-HDU cards are pure UVFITS write-back state (not format-neutral
# observation metadata), so they live here, in the FITS extension, rather
# than on `UVMetadata`. `WeakKeyDict` so the cards are GC'd when the UVSet
# is.
const _PRIMARY_CARDS = WeakKeyDict{UVSet, Vector{Card}}()

function UVData.primary_cards(uvset::UVSet)
    haskey(_PRIMARY_CARDS, uvset) || error(
        "primary_cards(uvset): no FITS primary-HDU cards registered. " *
            "Either load via `load_uvfits`, or call " *
            "`register_primary_cards!(uvset, cards)` before writing."
    )
    return _PRIMARY_CARDS[uvset]
end

UVData.register_primary_cards!(uvset::UVSet, cards::AbstractVector) =
    (_PRIMARY_CARDS[uvset] = Vector{Card}(cards); uvset)

# Hook so primary-HDU cards follow a UVSet through `rebuild` / `select_*`
# / `merge_uvsets`. Source-of-truth lives only here; format-neutral code
# in `src/` calls `_propagate_extension_state!` and gets a no-op when
# the FITS extension isn't loaded.
function UVData._propagate_extension_state!(new::UVSet, old::UVSet)
    haskey(_PRIMARY_CARDS, old) && (_PRIMARY_CARDS[new] = _PRIMARY_CARDS[old])
    return new
end

# ── Read path ───────────────────────────────────────────────────────────────

function UVData.load_uvfits(path)
    flat = _load_uvfits_flat(path)
    uvset = UVSet(flat)
    UVData.register_primary_cards!(uvset, flat.primary_cards)
    return uvset
end

const _NX_TIME_TOL_HOURS = 1.0e-6

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
            lo = lower_s[pos] - _NX_TIME_TOL_HOURS
            hi = upper_s[pos] + _NX_TIME_TOL_HOURS
            if lo <= t <= hi
                assigned = perm[pos]
            end
        end
        if assigned == 0 && pos + 1 <= length(lower_s)
            lo = lower_s[pos + 1] - _NX_TIME_TOL_HOURS
            hi = upper_s[pos + 1] + _NX_TIME_TOL_HOURS
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

function _load_uvfits_flat(path)
    fid = FITSFiles.fits(path)
    primary_hdu = fid[1]
    # Bypass FITSFiles' per-record Vector{Float32} allocation when the
    # primary HDU is a random-group LazyArray.
    primary_lazy = getfield(primary_hdu, :data)
    dt = primary_lazy isa FITSFiles.LazyArray ?
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
    raw::Array{Float32, 4} = dropdims(dt.data, dims = Tuple(dim1))

    vis_raw::Array{ComplexF32, 3} = complex.(raw[:, 1, :, :], raw[:, 2, :, :])
    weights_raw::Array{Float32, 3} = raw[:, 3, :, :]

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
    # We lift each column to Float64 *before* summing (Float32 ULP at JD
    # magnitude is ~0.25 days, which would collapse sub-second timestamps),
    # then detect whether the sum is full JD or already RDATE-relative and
    # produce **fractional hours since RDATE 00:00 UTC** ("hours since the
    # start of the track"). Magnitude is bounded by track length (typically
    # 0..~24 h) so float-tolerance comparisons in scan-matching code (e.g.
    # `_scan_index_for_leaf`'s `≈`) have plenty of headroom.
    rdate_jd = _rdate_jd_or_zero(array_obs.rdate)
    date_raw = collect(dt.DATE)
    raw_sum::Vector{Float64} = if ndims(date_raw) == 2
        Float64.(@view date_raw[:, 1]) .+ Float64.(@view date_raw[:, 2])
    else
        Float64.(date_raw)
    end
    # Detect AIPS-strict full-JD (large) vs already-RDATE-relative (small).
    # Full JD has been > 2.4e6 since ~1858 (the MJD reference); plausible
    # days-since-RDATE is < ~1e3 even for decade-long campaigns. A value
    # in [1e3, 2.4e6) cannot come from either convention and indicates a
    # corrupted file or unsupported writer — error rather than guess.
    days_since_rdate::Vector{Float64} = if isempty(raw_sum)
        raw_sum
    else
        raw_max = maximum(raw_sum)
        if raw_max >= 2.4e6
            raw_sum .- rdate_jd          # AIPS-strict full JD
        elseif raw_max <= 1.0e3
            raw_sum                       # Gustavo / RDATE-relative
        else
            error(
                "load_uvfits: ambiguous DATE PTYPE values — max(col1+col2) = $raw_max " *
                    "is neither plausibly a full Julian Day (≥2.4e6) nor a " *
                    "days-since-RDATE offset (≤1e3). The file may be corrupted " *
                    "or written with an unsupported convention."
            )
        end
    end
    obs_time::Vector{Float64} = days_since_rdate .* 24.0
    bl_codes::Vector{Int} = round.(Int, collect(dt.BASELINE))

    _col(nt, prefix) = collect(getproperty(nt, first(filter(k -> startswith(string(k), prefix), propertynames(nt)))))
    uvw_raw::Matrix{Float32} = hcat(_col(dt, "UU"), _col(dt, "VV"), _col(dt, "WW"))

    cfq::Vector{Float64} = channel_freqs(first(freq_setups))
    dims = (Integration(obs_time), Pol(msv4_labels), Frequency(cfq))

    vis, weights, uvw = _build_arrays(vis_raw, weights_raw, uvw_raw, dims)

    extra_columns = _collect_extra_columns(dt, primary_hdu.cards)

    # Materialize the NX columns up front: they come back as lazy
    # `DiskArrays`-backed broadcasts. Each NX row defines one MSv4 partition.
    # AIPS NX has no SCAN_NUMBER column, so the row index becomes the
    # canonical scan label (string-cast for xradio `ScanArray` shape).
    nx_time::Vector{Float64} = Float64.(collect(nx.TIME))
    nx_dt::Vector{Float64} = Float64.(collect(nx.var"TIME INTERVAL"))
    # NX TIME / TIME INTERVAL are days-since-RDATE; convert to hours
    # since RDATE 00:00 UTC to share `obs_time`'s scale.
    nx_lower::Vector{Float64} = (nx_time .- nx_dt ./ 2) .* 24.0
    nx_upper::Vector{Float64} = (nx_time .+ nx_dt ./ 2) .* 24.0
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
        vis = vis[Integration = valid]
        weights = weights[Integration = valid]
        uvw = uvw[Integration = valid]
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
        primary_cards = primary_hdu.cards,
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
# is a single `read!` into a `Vector{Float32}` of length
# `N * (P + prod(shape))`, plus an in-place `bswap` pass.
function _fast_random_read(lazy::FITSFiles.LazyArray)
    fmt = lazy.format
    fmt.type === Float32 ||
        error("_fast_random_read: only Float32 random groups supported (got $(fmt.type))")
    fields = lazy.fields::AbstractVector{<:FITSFiles.AbstractField}
    P = fmt.param::Int
    N = fmt.group::Int
    leng_data = prod(fmt.shape)::Int
    L = P + leng_data
    total = N * L
    buf = Vector{Float32}(undef, total)
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
            v = Vector{Float32}(undef, N)
            @inbounds for j in 1:N
                v[j] = rec[ndx[1], j]
            end
            if !ismissing(fld.zero) && !ismissing(fld.scale) &&
                    (fld.zero != 0 || fld.scale != 1)
                @inbounds @simd for j in 1:N
                    v[j] = fld.zero + fld.scale * v[j]
                end
            end
            v
        else
            m = Matrix{Float32}(undef, N, length(ndx))
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
# Used to subtract from AIPS-strict full-JD `DATE` PTYPE values when
# decoding to "days since RDATE". Returns 0.0 when the string is empty /
# unparseable so the subtraction is a no-op (Gustavo-written files
# already store days-since-RDATE in `DATE` and do not need the subtract).
function _rdate_jd_or_zero(rdate_str::AbstractString)
    isempty(rdate_str) && return 0.0
    return try
        datetime2julian(DateTime(Date(rdate_str)))
    catch
        0.0
    end
end

# Reconstruct AIPS DATE PTYPE columns from a leaf's `obs_time` (fractional
# hours since RDATE 00:00 UTC). Emits two columns shaped `(nrec_leaf, 2)`:
# column 1 is the integer-day part of `obs_time / 24`, column 2 is the
# sub-day fractional remainder. Their sum is days-since-RDATE; the
# integer-JD reference is on the AN HDU's RDATE card. On read the
# loader recognises this layout (sum is small, < 1e5) and uses it
# directly; AIPS-strict files (sum is full JD ≥ 1e5) get rdate_jd
# subtracted first.
function _build_date_param(obs_time_hr::AbstractVector{<:Real}, record_order)
    n = length(record_order)
    out = Matrix{Float32}(undef, n, 2)
    @inbounds for (rec_i, (ti, _)) in enumerate(record_order)
        days = obs_time_hr[ti] / 24.0
        intpart = floor(days)
        out[rec_i, 1] = Float32(intpart)
        out[rec_i, 2] = Float32(days - intpart)
    end
    return out
end

_wrap_int_pol_if(arr, obs_time, pol_labels, channel_freqs) = DimArray(
    arr,
    (Integration(obs_time), Pol(pol_labels), Frequency(channel_freqs)),
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

"""
    _strip_stale_shape_cards!(cards)

Remove all PTYPE/PSCAL/PZERO/PUNIT/NAXIS\$j cards plus NAXIS, PCOUNT,
GCOUNT from the copied primary header so `create_cards!` rebuilds them
cleanly from the freshly-derived format/fields. Preserving the originals
is unsafe because `create_cards!` reuses any existing `PTYPE\$j` card via
`popat!`, which carries forward stale duplicate `DATE` entries from the
input header.
"""
function _strip_stale_shape_cards!(cards)
    pat = r"^(PTYPE|PSCAL|PZERO|PUNIT|NAXIS)\d+$|^(NAXIS|PCOUNT|GCOUNT)$"
    filter!(c -> match(pat, strip(string(c.key))) === nothing, cards)
    return cards
end

"""
    _inject_duplicate_date!(hdu)

Mutate the primary HDU's cards to add a second `PTYPE\$P=DATE` entry
and bump `PCOUNT` accordingly, restoring the AIPS-convention
integer-JD + fractional-day PTYPE pair. The HDU's `data.DATE` value
must already be an N×2 matrix; on `Base.write`, FITSFiles' Random
field layout reads two `DATE` PTYPE cards from the header and writes
columns 1 and 2 of the matrix as the two parameter values.
"""
function _inject_duplicate_date!(hdu)
    cards = getfield(hdu, :cards)
    date_idxs = Int[]
    pcount_idx = 0
    naxis_val = 0
    for (i, c) in enumerate(cards)
        key = strip(string(c.key))
        if key == "PCOUNT"
            pcount_idx = i
        elseif key == "NAXIS"
            naxis_val = Int(c.value)
        else
            m = match(r"^PTYPE(\d+)$", key)
            if m !== nothing && rstrip(string(c.value)) == "DATE"
                push!(date_idxs, parse(Int, m.captures[1]))
            end
        end
    end
    isempty(date_idxs) && return hdu
    length(date_idxs) >= 2 && return hdu
    pcount_idx == 0 && return hdu
    new_pcount = Int(cards[pcount_idx].value) + 1
    cards[pcount_idx] = Card("PCOUNT", Int32(new_pcount))
    push!(cards, Card("PTYPE$(new_pcount)", "DATE"))
    return hdu
end

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
    # Emit one NamedTuple entry per unique PTYPE name. Duplicate AIPS
    # `DATE` PTYPEs (integer-JD + fractional-day pair) collapse to a
    # single :DATE key whose value is an N×2 matrix; the second PTYPE
    # card is re-injected into the header after HDU construction (see
    # `_inject_duplicate_date!`).
    pairs = Pair{Symbol, Any}[]
    seen = Set{String}()
    for (idx, name) in _ptype_entries(UVData.primary_cards(uvset))
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

function _build_an_hdu(
        antennas::AntennaTable, array_obs::ObsArrayMetadata, ref_freq::Float64;
        extver::Integer = 1, no_if::Integer = 1, freqid::Integer = 1,
    )
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
    # AIPS AN HDU header: time-system / Earth-orientation / coord-frame
    # fields source from ObsArrayMetadata. Pure-AIPS bookkeeping fields
    # (NUMORB, NO_IF, NOPCAL, FREQID, EXTVER) are reconstructed from the
    # input data on write.
    nopcal = isempty(polcala) ? 0 : max(length(polcala[1]), length(polcalb[1]))
    cards = [
        Card("EXTNAME", "AIPS AN"),
        Card("EXTVER", Int32(extver)),
        Card("ARRAYX", Float64(arr_xyz[1])),
        Card("ARRAYY", Float64(arr_xyz[2])),
        Card("ARRAYZ", Float64(arr_xyz[3])),
        Card("ARRNAM", array_name(antennas)),
        Card("FREQ", ref_freq),
        Card("RDATE", array_obs.rdate),
        Card("GSTIA0", array_obs.gst_iat0),
        Card("DEGPDY", array_obs.earth_rot_rate),
        Card("UT1UTC", array_obs.ut1utc),
        Card("POLARX", array_obs.polarx),
        Card("POLARY", array_obs.polary),
        Card("DATUTC", array_obs.datutc),
        Card("TIMSYS", array_obs.time_sys),
        Card("FRAME", array_obs.frame),
        Card("XYZHAND", array_obs.xyzhand),
        Card("POLTYPE", array_obs.poltype),
        Card("NUMORB", Int32(0)),
        Card("NO_IF", Int32(no_if)),
        Card("NOPCAL", Int32(nopcal)),
        Card("FREQID", Int32(freqid)),
    ]
    return HDU(Bintable, data, cards)
end

"""
    _collect_freq_setups(uvset) -> (Vector{FrequencySetup}, Dict{FrequencySetup,Int32})

Walk leaves and gather their unique `FrequencySetup`s in first-seen
order (via `union_frequency_axis`), then build a setup → FRQSEL
lookup. The FRQSEL value is recovered from `setup.extras.frqsel` when
present (preserved on read), else the setup's 1-based position. If the
recovered FRQSELs collide, fall back to positional 1..N to guarantee
uniqueness on disk.
"""
function _collect_freq_setups(uvset::UVSet)
    setups = UVData.union_frequency_axis(uvset)
    parsed = Int32[Int32(get(fs.extras, :frqsel, i)) for (i, fs) in enumerate(setups)]
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
    # Carry per-row extras only if every setup has the same extras keyset.
    # `:frqsel` is preserved on read for round-trip recovery; it's already
    # a top-level FRQSEL column here, so drop it from the extras merge.
    _drop_frqsel(ex) = NamedTuple{filter(!=(:frqsel), keys(ex))}(ex)
    extras_keys = keys(_drop_frqsel(first(setups).extras))
    if all(keys(_drop_frqsel(fs.extras)) == extras_keys for fs in setups)
        extras_per_row = NamedTuple{extras_keys}(
            ntuple(i -> [_drop_frqsel(fs.extras)[i] for fs in setups], length(extras_keys))
        )
        data = merge(base, extras_per_row)
    else
        data = base
    end
    cards = Card[Card("EXTNAME", "AIPS FQ"), Card("NO_IF", Int32(nif))]
    return HDU(Bintable, data, cards)
end

function _build_nx_hdu(
        scan_windows::AbstractVector{Tuple{Float64, Float64}},
        record_starts::AbstractVector, record_ends::AbstractVector,
        freqid_per_scan::AbstractVector{<:Integer},
        subarray_per_scan::AbstractVector{<:Integer} = fill(Int32(1), length(scan_windows)),
    )
    nscan = length(scan_windows)
    # scan_windows are fractional hours since RDATE 00:00 UTC; convert
    # to AIPS NX's days-since-RDATE convention.
    time_center = [(lo + hi) / 2 / 24.0 for (lo, hi) in scan_windows]
    time_interval = [Float32((hi - lo) / 24.0) for (lo, hi) in scan_windows]
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

    # UVFITS limitation: a single global STOKES axis on the primary HDU
    # forces the same pol product set across every record. UVSet supports
    # per-leaf pol products natively, but writing them out via UVFITS is
    # not possible without padding (lossy). Use MSv4 / xradio for that
    # round-trip case.
    UVData.union_pol_products(uvset)

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
    cards = UVData.primary_cards(uvset)
    aips_codes_disk, aips_labels_disk, _, _ = parse_stokes_axis(cards, npol)
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
    raw_data = zeros(Float32, nrec_total, 3, npol, nchan, 1, 1, 1)
    uu = Vector{uvw_eltype}(undef, nrec_total)
    vv = Vector{uvw_eltype}(undef, nrec_total)
    ww_ = Vector{uvw_eltype}(undef, nrec_total)
    bl_codes = Vector{Int}(undef, nrec_total)
    # AIPS DATE PTYPE: two columns whose sum is days-since-RDATE. We
    # split `obs_time / 24` (already RDATE-relative) into floor +
    # fractional parts so the Float32 storage carries sub-second
    # precision; the integer-JD reference lives on the AN HDU's RDATE
    # card.
    date_param_cat = Matrix{Float32}(undef, nrec_total, 2)
    extras_keys = keys(fp_info.extra_columns)
    extras_eltypes = ntuple(i -> eltype(fp_info.extra_columns[i]), length(extras_keys))
    extras_bufs = ntuple(i -> Vector{extras_eltypes[i]}(undef, nrec_total), length(extras_keys))

    # Dedup leaf antenna tables: each unique table becomes one AN HDU on
    # disk with a distinct EXTVER (1, 2, ...). Single-subarray observations
    # produce one entry; multi-subarray observations produce many.
    unique_antennas = AntennaTable[]
    extver_lookup = Dict{AntennaTable, Int32}()
    for (_, leaf) in branches_dict
        ants = DimensionalData.metadata(leaf).antennas
        if !haskey(extver_lookup, ants)
            push!(unique_antennas, ants)
            extver_lookup[ants] = Int32(length(unique_antennas))
        end
    end

    nscan = length(leaf_list)
    record_starts = zeros(Int32, nscan)
    record_ends = zeros(Int32, nscan)
    freqid_per_scan = ones(Int32, nscan)
    subarray_per_scan = ones(Int32, nscan)
    scan_windows = Vector{Tuple{Float64, Float64}}(undef, nscan)

    rec_offset = 0
    for (sid, leaf) in enumerate(leaf_list)
        info = DimensionalData.metadata(leaf)
        bls = info.baselines
        ro = info.record_order
        first_row_in_scan = rec_offset + 1
        # Re-encode pairs to AIPS BASELINE codes only at the FITS boundary.
        bl_aips_codes = [_encode_aips_baseline(a, b) for (a, b) in bls.pairs]
        # Reconstruct DATE PTYPE columns per-record from obs_time
        # (already in hours since RDATE 00:00 UTC).
        date_param_leaf = _build_date_param(UVData.obs_time(leaf), ro)
        _write_records_kernel!(
            raw_data, uu, vv, ww_, bl_codes, date_param_cat,
            parent(leaf[:vis]), parent(leaf[:weights]), parent(leaf[:uvw]),
            bl_aips_codes, ro, date_param_leaf, rec_offset, pol_perm,
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
            subarray_per_scan[sid] = extver_lookup[info.antennas]
        end
        rec_offset += length(ro)
    end

    extras_cat = NamedTuple{extras_keys}(extras_bufs)
    primary_data = _build_primary_data(uvset, uu, vv, ww_, bl_codes, date_param_cat, extras_cat, raw_data)
    primary_cards_out = _strip_stale_shape_cards!(copy(cards))
    primary_hdu = HDU(Random, primary_data, primary_cards_out)
    _inject_duplicate_date!(primary_hdu)
    # One AN HDU per unique antenna table. Per-SPW reference frequencies
    # live on each FQ row; the AN ref_freq is the array nominal taken from
    # the first setup.
    first_setup = first(setups)
    no_if_v = nchannels(first_setup)
    freqid_v = Int(get(first_setup.extras, :frqsel, Int32(1)))
    an_hdus = HDU[
        _build_an_hdu(
                ants, root.array_obs, ref_freq(first_setup);
                extver = extver_lookup[ants], no_if = no_if_v, freqid = freqid_v,
            ) for ants in unique_antennas
    ]
    fq_hdu = _build_fq_hdu(setups, [freqid_lookup[fs] for fs in setups])
    nx_hdu = _build_nx_hdu(
        scan_windows, record_starts, record_ends, freqid_per_scan, subarray_per_scan,
    )

    out_hdus = HDU[primary_hdu]
    append!(out_hdus, an_hdus)
    push!(out_hdus, fq_hdu)
    push!(out_hdus, nx_hdu)
    write(output_path, out_hdus)
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
        bl_aips_codes_local::AbstractVector{Int32},
        record_order::AbstractVector{Tuple{Int, Int}},
        date_param::AbstractMatrix{Tdate},
        rec_offset::Integer,
        pol_perm::AbstractVector{Int},
    ) where {Tvis, Tw, Tuvw, Tdate}
    # Leaf storage: (Frequency, Ti, Baseline, Pol) for vis/weights;
    # (Ti, Baseline, UVW) for uvw.
    npol = length(pol_perm)
    nchan = size(vis_dense, 1)
    @inbounds for (rec_i, (ti, bi)) in enumerate(record_order)
        row = rec_offset + rec_i
        for pdisk in 1:npol
            pmem = pol_perm[pdisk]
            for c in 1:nchan
                v = vis_dense[c, ti, bi, pmem]
                raw_data[row, 1, pdisk, c, 1, 1, 1] = real(v)
                raw_data[row, 2, pdisk, c, 1, 1, 1] = imag(v)
                raw_data[row, 3, pdisk, c, 1, 1, 1] = w_dense[c, ti, bi, pmem]
            end
        end
        uu[row] = uvw_dense[ti, bi, 1]
        vv[row] = uvw_dense[ti, bi, 2]
        ww_[row] = uvw_dense[ti, bi, 3]
        bl_codes[row] = Int(bl_aips_codes_local[bi])
        date_param_cat[row, :] .= @view date_param[rec_i, :]
    end
    return nothing
end

end # module
