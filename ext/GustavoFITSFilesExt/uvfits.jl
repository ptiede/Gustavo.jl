using FITSFiles
using FITSFiles: HDU
using Dates: Dates, Date, DateTime, datetime2julian
using OrderedCollections: OrderedDict
import XRadio

import Gustavo.UVData
using Gustavo.UVData:
    MountAltAz, MountEquatorial, MountNasmythR, MountNasmythL,
    MountBWGR, MountBWGL, MountXY, MountOrbiting, sanitize_source

# AIPS UVFITS BASELINE-column convention: pack `(a, b)` antenna indices
# as `bl = a*256 + b`. Caps the array at 255 antennas. Lives in the FITS
# extension only; format-neutral code in `src/` speaks `(a, b)` tuples.
_decode_aips_baseline(bl::Integer)::Tuple{Int, Int} = (bl ÷ 256, bl % 256)


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

const _AIPS_STOKES_LABELS = Dict(
    -1 => "RR", -2 => "LL", -3 => "RL", -4 => "LR",
    -5 => "XX", -6 => "YY", -7 => "XY", -8 => "YX",
)

# Product order of a Measurement Set; a subset keeps this relative order.
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
    parse_stokes_axis(cards, npol) -> (aips_codes, labels, perm)

Parse AIPS STOKES axis. Returns the raw AIPS pol codes (e.g. `[-1,-2,-3,-4]`),
the MSv4 labels of the products in canonical feed-pair order (`PP`, `PQ`, `QP`,
`QQ`, so `["RR", "RL", "LR", "LL"]` for a circular file), and the permutation
`perm` taking the file's axis order to that one.
"""
function parse_stokes_axis(cards, npol)
    axis = _find_axis(cards, ==("STOKES"))
    isnothing(axis) && throw(
        ArgumentError("load_uvfits: the primary HDU has no STOKES axis to label its products")
    )

    crval = something(card_value(cards, "CRVAL$axis"), 1.0)
    cdelt = something(card_value(cards, "CDELT$axis"), 1.0)
    crpix = something(card_value(cards, "CRPIX$axis"), 1.0)
    aips_codes::Vector{Int} = Int.(round.(crval .+ cdelt .* ((1:npol) .- crpix)))
    aips_labels = aips_code_to_generic.(aips_codes)
    msv4_labels = _msv4_order(aips_labels)
    perm = [findfirst(==(lab), aips_labels) for lab in msv4_labels]
    labels = [_AIPS_STOKES_LABELS[c] for c in aips_codes[perm]]
    return aips_codes, labels, Vector{Int}(perm)
end

function _find_freq_axis(cards)
    i = _find_axis(cards, ==("FREQ"))
    isnothing(i) && return 0.0
    return something(card_value(cards, "CRVAL$i"), 0.0)
end

_clean_name(s) = filter(c -> isascii(c) && isprint(c) && !isspace(c), string(s))

# One AN table, as the columns the antenna sub-dataset and the baseline names
# are built from.
function _read_antenna_table(an_hdu)
    cards = an_hdu.cards
    an = an_hdu.data
    names = _clean_name.(collect(an.ANNAME))
    nant = length(names)
    numbers = hasproperty(an, :NOSTA) ? Int.(collect(an.NOSTA)) : collect(1:nant)
    xyz = collect(an.STABXYZ)
    staxof = hasproperty(an, :STAXOF) ? collect(an.STAXOF) : zeros(nant)
    mounts = mnt_codes_to_type.(collect(an.MNTSTA), _uvfits_axis_offset.(staxof))
    poltya = hasproperty(an, :POLTYA) ? strip.(collect(an.POLTYA)) : fill("R", nant)
    poltyb = hasproperty(an, :POLTYB) ? strip.(collect(an.POLTYB)) : fill("L", nant)
    polaa = hasproperty(an, :POLAA) ? collect(an.POLAA) : zeros(nant)
    polab = hasproperty(an, :POLAB) ? collect(an.POLAB) : zeros(nant)
    for p in Iterators.flatten((poltya, poltyb))
        p in ("R", "L", "X", "Y") || throw(ArgumentError("Unsupported polarization type: $p"))
    end

    # `STABXYZ` is measured from the ARRAYX/Y/Z array center.
    center = [Float64(something(card_value(cards, k), 0.0)) for k in ("ARRAYX", "ARRAYY", "ARRAYZ")]
    positions = [center[c] + Float64(xyz[i, c]) for c in 1:3, i in 1:nant]
    return (;
        names, numbers, positions, mounts,
        polarization_type = [String(r == 1 ? poltya[i] : poltyb[i]) for r in 1:2, i in 1:nant],
        receptor_angle = [deg2rad(Float64(r == 1 ? polaa[i] : polab[i])) for r in 1:2, i in 1:nant],
        diameter = hasproperty(an, :DIAMETER) ? Float64.(collect(an.DIAMETER)) : nothing,
        array_name = string(something(card_value(cards, "ARRNAM"), "")),
        rdate = string(something(card_value(cards, "RDATE"), "")),
        earth_orientation = Dict{Symbol, Any}(
            :gst_iat0 => Float64(something(card_value(cards, "GSTIA0"), 0.0)),
            :earth_rot_rate => Float64(something(card_value(cards, "DEGPDY"), 360.0)),
            :ut1utc => Float64(something(card_value(cards, "UT1UTC"), 0.0)),
            :polarx => Float64(something(card_value(cards, "POLARX"), 0.0)),
            :polary => Float64(something(card_value(cards, "POLARY"), 0.0)),
            :datutc => Float64(something(card_value(cards, "DATUTC"), 0.0)),
            :xyzhand => string(something(card_value(cards, "XYZHAND"), "RIGHT")),
            :poltype => string(something(card_value(cards, "POLTYPE"), "")),
        ),
    )
end

function _antenna_dataset(tab)
    n = length(tab.names)
    ds = XRadio.subdataset(
        :antenna;
        antenna_name = tab.names,
        cartesian_pos_label = ["x", "y", "z"],
        receptor_label = ["pol_0", "pol_1"],
        antenna_position = tab.positions,
        station_name = tab.names,
        telescope_name = fill(tab.array_name, n),
        polarization_type = tab.polarization_type,
        antenna_receptor_angle = tab.receptor_angle,
        antenna_dish_diameter = tab.diameter,
        metadata = (; overall_telescope_name = tab.array_name, relocatable_antennas = false),
        array_metadata = (;
            antenna_position = (;
                frame = "ITRS", coordinate_system = "geocentric", origin_object_name = "earth",
            ),
        ),
    )
    return XRadio.set_mounts!(ds, tab.mounts)
end

# An FQ column as an `(nrows, nif)` matrix. A column of one value per row is a
# single IF.
function _fq_matrix(col)
    m = collect(col)
    return Float64.(m isa AbstractVector ? reshape(m, :, 1) : m)
end

"""
    _read_frequency_setups(cards, fq, nif) -> (setups, frqsels)

Read every row of the FQ HDU into the channel frequencies, widths, total
bandwidths and sidebands of one spectral window, together with the rows' AIPS
FRQSEL values. Errors when a row's IF count is not the data's.
"""
function _read_frequency_setups(cards, fq, nif)
    ref_freq = Float64(_find_freq_axis(cards))
    if_freqs = _fq_matrix(getproperty(fq, Symbol("IF FREQ")))
    widths = _fq_matrix(getproperty(fq, Symbol("CH WIDTH")))
    bandwidths = _fq_matrix(getproperty(fq, Symbol("TOTAL BANDWIDTH")))
    sidebands = _fq_matrix(getproperty(fq, :SIDEBAND))
    size(if_freqs, 2) == nif ||
        error("FQ table reports $(size(if_freqs, 2)) IFs but vis has $nif channels")
    nrows = size(if_freqs, 1)
    frqsels = hasproperty(fq, :FRQSEL) ?
        round.(Int32, collect(getproperty(fq, :FRQSEL))) : Int32.(1:nrows)
    setups = [
        (;
            ref_freq,
            channel_freqs = ref_freq .+ if_freqs[r, :],
            ch_width = widths[r, :],
            total_bandwidth = bandwidths[r, :],
            sideband = Int.(sidebands[r, :]),
        ) for r in 1:nrows
    ]
    return setups, frqsels
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

const _C_LIGHT = 299792458.0

function UVData.load_uvfits(path; element_type::Union{Nothing, Type} = nothing)
    isnothing(element_type) || element_type <: AbstractFloat || throw(
        ArgumentError(
            "load_uvfits: element_type must be a real float type, got $(element_type)",
        ),
    )
    return _read_uvfits(path, element_type)
end

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

# The data array as `(record, complex, stokes, IF)`. Each IF holds one channel,
# which is how the FQ table describes the band; a file with no IF axis has one.
function _record_cube(cards, data)
    function axis(name)
        i = _find_axis(cards, ==(name))
        isnothing(i) && throw(ArgumentError("load_uvfits: the primary HDU has no $name axis"))
        return i
    end
    complex_, stokes, freq = axis("COMPLEX"), axis("STOKES"), axis("FREQ")
    if_ = _find_axis(cards, ==("IF"))
    if isnothing(if_)
        data = reshape(data, size(data)..., 1)
        if_ = ndims(data)
    end
    size(data, complex_) == 3 || throw(
        ArgumentError(
            "load_uvfits: the COMPLEX axis has $(size(data, complex_)) entries; " *
                "a record states real, imaginary and weight"
        )
    )
    size(data, freq) == 1 || throw(
        ArgumentError(
            "load_uvfits: the FREQ axis has $(size(data, freq)) channels per IF; " *
                "only one channel per IF is supported"
        )
    )
    rest = setdiff(2:ndims(data), (complex_, stokes, freq, if_))
    all(d -> size(data, d) == 1, rest) || throw(
        ArgumentError("load_uvfits: the data array has more than one pixel along an axis other than COMPLEX, STOKES, FREQ and IF")
    )
    cube = permutedims(data, (1, complex_, stokes, if_, freq, rest...))
    return reshape(cube, size(cube, 1), size(cube, 2), size(cube, 3), size(cube, 4))
end

function _visibility_units(cards, path)
    u = card_value(cards, "BUNIT")
    if isnothing(u) || isempty(string(u))
        @warn "UVFITS primary HDU carries no BUNIT; writing `uncalib` as \
               `VISIBILITY.units`. Data calibrated elsewhere needs its units \
               stated in the file." file = path
        return "uncalib"
    end
    return uppercase(string(u)) == "JY" ? "Jy" : string(u)
end

function _read_uvfits(path, element_type)
    fid = FITSFiles.fits(path)
    primary_hdu = fid[1]
    cards = primary_hdu.cards
    # Bypass FITSFiles' per-record Vector{Float32} allocation when the
    # primary HDU is lazy random-group data.
    primary_lazy = getfield(primary_hdu, :data)
    dt = _is_lazy_random(primary_lazy) ?
        _fast_random_read(primary_lazy) : primary_hdu.data
    # Multi-AN-extver files carry one AN table per subarray.
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
    isnothing(fq_hdu) && error("load_uvfits: no AIPS FQ HDU found in $(path)")
    isnothing(nx) && error("load_uvfits: no AIPS NX HDU found in $(path)")

    raw = _record_cube(cards, dt.data)
    T = isnothing(element_type) ? eltype(raw) : element_type

    aips_codes, pol_labels, perm = parse_stokes_axis(cards, size(raw, 3))
    # Read the visibilities verbatim. AIPS random-groups UVFITS shares MSv4's
    # phase convention — both are the AIPS/CASA/casacore sense, which is the
    # conjugate of FITS-IDI's V = ⟨E_a1 · conj(E_a2)⟩ (AIPS Memo 114r §2.1;
    # casacore FitsIDItoMS.cc: "FITS-IDI convention is conjugate of AIPS and CASA
    # convention"). (u,v,w) and the 256·a1+a2 baseline codes agree too, so only
    # UVW's units change on this boundary.
    vis_raw = Complex{T}.(view(raw, :, 1, perm, :), view(raw, :, 2, perm, :))
    weights_raw = T.(view(raw, :, 3, perm, :))

    antenna_tables = [_read_antenna_table(h) for h in an_hdus]
    _check_stokes_vs_poltya(aips_codes, first(antenna_tables))
    freq_setups, fq_frqsels = _read_frequency_setups(cards, fq_hdu.data, size(raw, 4))
    src_info = _build_source_info(primary_hdu)

    # AIPS UVFITS stores DATE as two PTYPE columns. Different writers use
    # different splits:
    #   * AIPS strict: col1 = integer JD (~2.46e6 for 2022 data), col2 =
    #     fractional day. Sum is full JD.
    #   * RDATE-relative: col1 = floor(days_since_RDATE) (~0..few),
    #     col2 = fractional remainder. Sum is days_since_RDATE.
    # We lift each column to Float64 *before* combining (Float32 ULP at JD
    # magnitude is ~0.25 days, which would collapse sub-second timestamps),
    # then detect whether the pair is a full JD or already RDATE-relative and
    # produce **seconds since `UVData.JD_UNIX_EPOCH`**, the `Ti` convention.
    # The two columns stay separate through the epoch subtraction: a Float64
    # resolves only ~40 µs at Julian-Day magnitude, so summing them first
    # would discard the split's whole purpose.
    rdate_jd = _rdate_jd_or_zero(first(antenna_tables).rdate)
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
    bl_pairs::Vector{Tuple{Int, Int}} = _decode_aips_baseline.(round.(Int, collect(dt.BASELINE)))
    inttim = hasproperty(dt, :INTTIM) ? Float64.(collect(dt.INTTIM)) : nothing

    _col(nt, prefix) = collect(getproperty(nt, first(filter(k -> startswith(string(k), prefix), propertynames(nt)))))
    # (u,v,w) share one baseline-coordinate convention across every FITS
    # flavour (coord = r_a1 − r_a2; AIPS Memo 117r §4.1.2 = Memo 114r §4.1.2
    # word for word), stated in light-seconds where MSv4 states meters.
    uvw_raw = hcat(_col(dt, "UU"), _col(dt, "VV"), _col(dt, "WW"))

    # Materialize the NX columns up front: they come back as lazy
    # `DiskArrays`-backed broadcasts. Each NX row defines one MSv4 partition.
    # AIPS NX has no SCAN_NUMBER column, so the row index becomes the
    # canonical scan label.
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

    # Per-record SPW index: prefer a per-record PTYPE (FREQSEL / FREQID),
    # else propagate the per-NX-row FREQ ID column, else default to 1.
    # After densification, indices are 1..length(freq_setups).
    nfreq = length(freq_setups)
    record_spw_index::Vector{Int32} = if hasproperty(dt, :FREQSEL)
        round.(Int32, collect(getproperty(dt, :FREQSEL)))
    elseif hasproperty(dt, :FREQID)
        round.(Int32, collect(getproperty(dt, :FREQID)))
    elseif length(nx_freqid) > 0
        Int32[r == 0 ? Int32(1) : nx_freqid[r] for r in record_nx_row]
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
            "index $(record_spw_index[1]); the result has one Measurement Set per scan."
    end

    # Per-record subarray index. AIPS UVFITS rarely emits a per-record
    # SUBARRAY PTYPE; the NX SUBARRAY column is the canonical source.
    record_subarray_index::Vector{Int32} = if hasproperty(dt, :SUBARRAY)
        round.(Int32, collect(getproperty(dt, :SUBARRAY)))
    elseif length(nx_subarray) > 0
        Int32[r == 0 ? Int32(1) : nx_subarray[r] for r in record_nx_row]
    else
        ones(Int32, length(obs_time))
    end

    # One Measurement Set per (NX scan, spectral window, subarray), in the
    # order the records first name them. Records in no NX scan are dropped.
    groups = OrderedDict{Tuple{Int, Int, Int}, Vector{Int}}()
    for i in eachindex(obs_time)
        record_nx_row[i] == 0 && continue
        key = (record_nx_row[i], Int(record_spw_index[i]), Int(record_subarray_index[i]))
        push!(get!(Vector{Int}, groups, key), i)
    end
    isempty(groups) && error("load_uvfits: no records in $(path)")

    shared = (;
        vis_raw, weights_raw, uvw_raw, obs_time, bl_pairs, inttim, pol_labels,
        units = _visibility_units(cards, path),
        source = src_info,
        observation_info = Dict{Symbol, Any}(
            :observer => String[string(something(card_value(cards, "TELESCOP"), ""))],
            :release_date => "",
            :project_UID => "",
        ),
        processor_info = Dict{Symbol, Any}(
            :type => "CORRELATOR",
            :sub_type => string(something(card_value(cards, "INSTRUME"), "")),
        ),
    )
    source_key = string(sanitize_source(src_info.source_name))
    sets = OrderedDict{Symbol, XRadio.MeasurementSet}()
    for ((scan, spw, sub), records) in groups
        parts = [source_key, "spw_$(spw - 1)"]
        length(antenna_tables) > 1 && push!(parts, "sub_$(sub - 1)")
        push!(parts, "scan_$scan")
        key = Symbol(join(parts, "_"))
        haskey(sets, key) && error("load_uvfits: duplicate partition key $(key)")
        sets[key] = _measurement_set(
            shared, records, string(scan), spw - 1, freq_setups[spw], antenna_tables[sub],
        )
    end
    return XRadio.ProcessingSet(sets)
end

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
function _check_stokes_vs_poltya(aips_codes, tab)
    stokes_is_linear = all(c -> c <= -5, aips_codes)
    stokes_is_circular = all(c -> -4 <= c <= -1, aips_codes)
    feeds_linear = all(in(("X", "Y")), tab.polarization_type)
    feeds_circular = all(in(("R", "L")), tab.polarization_type)
    if stokes_is_linear && !feeds_linear
        @warn "Stokes axis is linear (XX/YY/XY/YX) but POLTYA/POLTYB are not all linear"
    elseif stokes_is_circular && !feeds_circular
        @warn "Stokes axis is circular (RR/LL/RL/LR) but POLTYA/POLTYB are not all circular"
    end
    return nothing
end

_quantity(v, units) = XRadio.Measure(
    Float64(v), Dict{Symbol, Any}(:units => units, :type => "quantity"),
)

# The nominal integration time of a Measurement Set: the largest of its records'
# INTTIM where the file states one, and otherwise the smallest positive spacing
# of its times.
function _integration_time(inttim, records, times)
    isnothing(inttim) || return maximum(inttim[records])
    length(times) < 2 && return 0.0
    return minimum(filter(>(0), diff(times)))
end

function _measurement_set(shared, records, scan_name, spw_id, setup, tab)
    times = sort!(unique(shared.obs_time[records]))
    pairs_ = sort!(unique(shared.bl_pairs[records]))
    time_of = Dict(t => k for (k, t) in pairs(times))
    base_of = Dict(p => b for (b, p) in pairs(pairs_))
    name_of = Dict(zip(tab.numbers, tab.names))
    antenna_name(a) = get(name_of, a) do
        throw(ArgumentError("load_uvfits: BASELINE names antenna $a, which the AN table does not number"))
    end

    vis_raw, weights_raw = shared.vis_raw, shared.weights_raw
    npol, nchan = size(vis_raw, 2), size(vis_raw, 3)
    C = eltype(vis_raw)
    shape = (npol, nchan, length(pairs_), length(times))
    vis = fill(C(NaN, NaN), shape)
    weight = zeros(real(C), shape)
    flag = trues(shape)
    # `Float64` whatever the science arrays are: a VLBI baseline is ~1e7 m and
    # the phase it fixes turns over in a wavelength of millimeters.
    uvw = fill(NaN, 3, length(pairs_), length(times))
    effective = isnothing(shared.inttim) ? nothing : zeros(length(pairs_), length(times))
    for r in records
        k = time_of[shared.obs_time[r]]
        b = base_of[shared.bl_pairs[r]]
        for axis in 1:3
            uvw[axis, b, k] = Float64(shared.uvw_raw[r, axis]) * _C_LIGHT
        end
        isnothing(effective) || (effective[b, k] = shared.inttim[r])
        for c in 1:nchan, p in 1:npol
            w = weights_raw[r, p, c]
            isfinite(w) || throw(
                ArgumentError("load_uvfits: record $r carries weight $w, which states neither a weight nor a flag")
            )
            vis[p, c, b, k] = vis_raw[r, p, c]
            # UVFITS records a flag as a negative weight; its magnitude is the
            # weight the sample would have had.
            weight[p, c, b, k] = abs(w)
            flag[p, c, b, k] = w < 0
        end
    end

    freq_attrs = (;
        spectral_window_name = "spw_$spw_id",
        spectral_window_intents = [""],
        reference_frequency = XRadio.Measure(
            setup.ref_freq,
            Dict{Symbol, Any}(:units => "Hz", :type => "spectral_coord", :observer => "icrs"),
        ),
        # MSv4 states one channel width for the window; the file states one per IF.
        channel_width = _quantity(first(setup.ch_width), "Hz"),
        # Neither has an MSv4 field; `GUSTAVO_VISIBILITY_SCHEMA` describes them.
        sideband = first(setup.sideband),
        total_bandwidth = _quantity(first(setup.total_bandwidth), "Hz"),
    )
    src = shared.source
    field_and_source = XRadio.subdataset(
        :field_and_source;
        field_name = [src.source_name],
        sky_dir_label = ["ra", "dec"],
        source_name = [src.source_name],
        field_phase_center_direction = reshape([src.ra, src.dec], 2, 1),
        array_metadata = (; field_phase_center_direction = (; frame = "icrs")),
    )
    nt = length(times)
    return XRadio.measurement_set(;
        time = times,
        frequency = setup.channel_freqs,
        polarization = shared.pol_labels,
        baseline_id = collect(1:length(pairs_)),
        baseline_antenna1_name = [antenna_name(a) for (a, _) in pairs_],
        baseline_antenna2_name = [antenna_name(b) for (_, b) in pairs_],
        uvw_label = ["u", "v", "w"],
        visibility = vis, flag, weight, uvw,
        effective_integration_time = effective,
        field_name = fill(src.source_name, nt),
        scan_name = fill(scan_name, nt),
        metadata = (;
            observation_info = shared.observation_info,
            processor_info = shared.processor_info,
            creator = Dict{Symbol, Any}(
                :software_name => "Gustavo.jl", :version => string(pkgversion(UVData)),
            ),
            # No MSv4 field; `GUSTAVO_VISIBILITY_SCHEMA` describes the block.
            earth_orientation = tab.earth_orientation,
        ),
        array_metadata = (;
            time = (; integration_time = _quantity(_integration_time(shared.inttim, records, times), "s")),
            frequency = freq_attrs,
            visibility = (; units = shared.units),
            uvw = (; frame = "icrs"),
            scan_name = (; scan_intents = String[]),
        ),
        subdatasets = (; antenna = _antenna_dataset(tab), field_and_source_base = field_and_source),
    )
end
