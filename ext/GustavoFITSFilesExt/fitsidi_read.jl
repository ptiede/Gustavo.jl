# FITS-IDI streaming/lazy reader (AIPS Memo 114).
#
# Parses every small header table (ARRAY_GEOMETRY, ANTENNA, SOURCE,
# FREQUENCY, PRIMARY) eagerly into Gustavo metadata types and leaves the
# `UV_DATA` FLUX matrix lazy: each per-(source, band, scan) leaf carries a
# disk-backed `IDIChunkArray` for `vis`/`weights`/`flag`, materialized only
# when `materialize_leaf` (or `Array`) is called. The index pass reads only
# the small per-row columns (DATE/TIME/BASELINE/SOURCE/FREQID/INTTIM/UVW);
# FLUX is never touched until a leaf is materialized, and then only the
# requested band's contiguous slice is read per (time, baseline) cell.
#
# Layout facts for the validation file (VLBA Q-band, DiFX), confirmed on
# disk and assumed generically via the table keywords:
#   MAXIS1 = COMPLEX(2)  re,im            (fastest)
#   MAXIS2 = STOKES(4)   AIPS -1..-4 = RR,LL,RL,LR
#   MAXIS3 = FREQ(NO_CHAN)
#   MAXIS4 = BAND(NO_BAND)                (slowest within FLUX)
# One band's data is a contiguous 2*NO_STKD*NO_CHAN slice at element offset
# (band-1)*nperband within the FLUX field.
#
# WEIGHT(NO_STKD*NO_BAND) carries one weight per (stokes, band) shared
# across all channels; the on-disk ordering is stokes-fastest then band,
# i.e. linear index = (band-1)*NO_STKD + stokes (verified empirically: each
# group of NO_STKD entries shows the parallel/cross-hand signature).

# DiskArrays is not a direct Gustavo dependency; reach it through FITSFiles,
# which `using DiskArrays` (its lazy field arrays subtype AbstractDiskArray).
const DiskArrays = FITSFiles.DiskArrays
using Statistics: median
using OhMyThreads: tforeach

# Tasks to spread one leaf's vis/weights decode over (bounded by the baseline
# count). Reads the solve-set knob in UVData; defaults to 1 (sequential).
_decode_ntasks(nbl::Int) = max(1, min(nbl, UVData._DECODE_NTASKS[]))

# ── Small helpers ────────────────────────────────────────────────────────────

# Strip FITS string padding (trailing spaces / NULs) from an `nA` column entry.
_idi_clean(s) = filter(c -> isascii(c) && isprint(c) && !isspace(c), string(s))

# A vector column read by FITSFiles comes back as either a `(nrows, leng)`
# matrix or, for `leng == 1`, a `(nrows,)` vector. `_idi_row(M, r)` returns row
# `r` as a vector regardless.
_idi_row(M::AbstractMatrix, r::Integer) = vec(M[r, :])
_idi_row(M::AbstractVector, r::Integer) = [M[r]]

# JD of RDATE 00:00 UTC, reusing the UVFITS convention. 0.0 when unparseable.
_idi_rdate_jd(rdate::AbstractString) = _rdate_jd_or_zero(rdate)

# ── STOKES axis → MSv4 Pol permutation ───────────────────────────────────────
#
# The FITS-IDI STOKES axis is defined by STK_1/NO_STKD (or CRVAL2/CDELT2 on
# the UV_DATA matrix), not by a CTYPE card. AIPS codes are
# `STK_1 + (k-1)*CDELT2` for k = 1..NO_STKD (e.g. -1,-2,-3,-4 → RR,LL,RL,LR →
# generic PP,QQ,PQ,QP). We map each to its MSv4 canonical Pol index so that
# `aips_labels[perm] == msv4_labels` (PP,PQ,QP,QQ).
function _idi_stokes_perm(cards)
    no_stkd = Int(something(card_value(cards, "NO_STKD"), 4))
    stk_1 = Int(something(card_value(cards, "STK_1"), -1))
    # CDELT2 on the matrix axis is the stride between successive codes; default
    # to -1 (the AIPS convention) when absent.
    cdelt = Int(round(Float64(something(card_value(cards, "CDELT2"), -1.0))))
    aips_codes = Int[stk_1 + (k - 1) * cdelt for k in 1:no_stkd]
    aips_labels = aips_code_to_generic.(aips_codes)
    msv4_labels = _msv4_order(aips_labels)
    perm = [findfirst(==(lab), aips_labels) for lab in msv4_labels]
    any(isnothing, perm) && error(
        "FITS-IDI STOKES axis: cannot reconcile AIPS labels $aips_labels " *
            "with MSv4 order $msv4_labels",
    )
    return aips_codes, aips_labels, msv4_labels, Vector{Int}(perm)
end

# ── Metadata builders ────────────────────────────────────────────────────────

# Build an `AntennaTable` from ARRAY_GEOMETRY (positions, mounts, offsets) and
# ANTENNA (feed bases / pol angles). Analogous to `_build_antenna_table` but
# sourcing the two FITS-IDI tables instead of a single AIPS-AN HDU.
function _build_idi_antenna_table(ag_hdu, an_hdu)
    ag_cards = ag_hdu.cards
    ag = ag_hdu.data
    an = an_hdu.data

    names = _idi_clean.(collect(ag.ANNAME))
    nant = length(names)

    xyz_raw = collect(ag.STABXYZ)               # (nant, 3)
    station_xyz = [Float64.(xyz_raw[i, :]) for i in 1:nant]

    mount_raw = collect(ag.MNTSTA)
    staxof_raw = hasproperty(ag, :STAXOF) ? collect(ag.STAXOF) : fill(0.0f0, nant)
    # STAXOF is `(nant, 3)`; the scalar feed offset is the first component.
    staxof_scalar = staxof_raw isa AbstractMatrix ?
        Float32.(staxof_raw[:, 1]) : Float32.(staxof_raw)
    mounts = mnt_codes_to_type.(mount_raw, staxof_scalar)

    poltya = poltype.(_idi_clean.(collect(an.POLTYA)))::Vector{<:POLBASIS}
    poltyb = poltype.(_idi_clean.(collect(an.POLTYB)))::Vector{<:POLBASIS}
    nominal_basis = tuple.(poltya, poltyb)

    # POLAA/POLAB are `(nant, NO_BAND)` (one angle per band); take the first
    # band's angle and convert from degrees to radians.
    polaa_raw = hasproperty(an, :POLAA) ? collect(an.POLAA) : fill(0.0f0, nant)
    polab_raw = hasproperty(an, :POLAB) ? collect(an.POLAB) : fill(0.0f0, nant)
    _first_band(M, i) = M isa AbstractMatrix ? Float32(M[i, 1]) : Float32(M[i])
    pol_angles = [
        (deg2rad(_first_band(polaa_raw, i)), deg2rad(_first_band(polab_raw, i)))
            for i in 1:nant
    ]

    response = [Diagonal(ones(ComplexF32, 2)) for _ in 1:nant]

    antennas = [
        Antenna(;
                name = names[i],
                station_xyz = station_xyz[i],
                mount = mounts[i],
                nominal_basis = nominal_basis[i],
                response = response[i],
                pol_angles = pol_angles[i],
            )
            for i in 1:nant
    ]

    arrayx = Float64(something(card_value(ag_cards, "ARRAYX"), 0.0))
    arrayy = Float64(something(card_value(ag_cards, "ARRAYY"), 0.0))
    arrayz = Float64(something(card_value(ag_cards, "ARRAYZ"), 0.0))
    arrnam = string(something(card_value(ag_cards, "ARRNAM"), ""))

    extras = (;
        DIAMETER = hasproperty(ag, :DIAMETER) ?
            Float32.(collect(ag.DIAMETER)) : fill(0.0f0, nant),
        NOSTA = round.(Int32, collect(ag.NOSTA)),
    )
    return AntennaTable(StructArray(antennas), (arrayx, arrayy, arrayz), arrnam, extras)
end

# One `FrequencySetup` per BAND. `channel_freqs = REF_FREQ + BANDFREQ[b] +
# (0:NO_CHAN-1)*CH_WIDTH[b]`.
function _build_idi_freq_setups(fq_hdu, ref_freq, no_chan, no_band)
    fq = fq_hdu.data
    bandfreq = collect(fq.BANDFREQ)        # (nfreqid, no_band)
    ch_width = collect(fq.CH_WIDTH)
    total_bw = collect(fq.TOTAL_BANDWIDTH)
    sideband = collect(fq.SIDEBAND)

    # Use the first (only) FREQID row.
    bf = _idi_row(bandfreq, 1)
    cw = _idi_row(ch_width, 1)
    tb = _idi_row(total_bw, 1)
    sb = _idi_row(sideband, 1)

    setups = FrequencySetup[]
    for b in 1:no_band
        chf = Float64(ref_freq) .+ Float64(bf[b]) .+ (0:(no_chan - 1)) .* Float64(cw[b])
        push!(
            setups, FrequencySetup(;
                name = "band_$(b)",
                ref_freq = Float64(ref_freq),
                channel_freqs = chf,
                ch_widths = fill(Float64(cw[b]), no_chan),
                total_bandwidths = fill(Float64(tb[b]), no_chan),
                sidebands = fill(Float64(sb[b]), no_chan),
                extras = (; bandfreq = Float64(bf[b]), band = b),
            )
        )
    end
    return setups
end

function _build_idi_array_obs(primary_hdu, ag_hdu)
    pc = primary_hdu.cards
    ac = ag_hdu.cards

    telescope = string(something(card_value(pc, "TELESCOP"), ""))
    instrume = string(something(card_value(pc, "INSTRUME"), ""))
    date_obs = string(something(card_value(pc, "DATE-OBS"), ""))
    equinox = Float32(something(card_value(pc, "EQUINOX"), 2000.0f0))
    bunit = string(something(card_value(pc, "BUNIT"), "UNCALIB"))

    rdate = string(something(card_value(ac, "RDATE"), ""))
    gst_iat0 = Float32(something(card_value(ac, "GSTIA0"), 0.0))
    earth_rot_rate = Float32(something(card_value(ac, "DEGPDY"), 360.0))
    ut1utc = Float32(something(card_value(ac, "UT1UTC"), 0.0))
    polarx = Float32(something(card_value(ac, "POLARX"), 0.0))
    polary = Float32(something(card_value(ac, "POLARY"), 0.0))
    datutc = Float32(something(card_value(ac, "IATUTC"), 0.0))
    frame = string(something(card_value(ac, "FRAME"), "GEOCENTRIC"))

    correlat = string(something(card_value(pc, "CORRELAT"), ""))
    return ObsArrayMetadata(;
        telescope, instrume, date_obs, equinox, bunit,
        rdate, gst_iat0, earth_rot_rate, ut1utc, polarx, polary, datutc,
        frame,
        extras = (; correlat = correlat),
    )
end

# SOURCE table → per-SOURCE_ID `(name, ra_rad, dec_rad)`. RAEPO/DECEPO are in
# degrees in FITS-IDI.
function _build_idi_sources(src_hdu)
    src = src_hdu.data
    ids = round.(Int, collect(src.SOURCE_ID))
    names = _idi_clean.(collect(src.SOURCE))
    raepo = Float64.(collect(src.RAEPO))
    decepo = Float64.(collect(src.DECEPO))
    out = Dict{Int, NamedTuple{(:name, :ra, :dec), Tuple{String, Float64, Float64}}}()
    for i in eachindex(ids)
        out[ids[i]] = (;
            name = names[i],
            ra = deg2rad(raepo[i]),
            dec = deg2rad(decepo[i]),
        )
    end
    return out
end

# ── FLAG table (AIPS Memo 114 §) ──────────────────────────────────────────────
#
# The FITS-IDI FLAG table records correlator/operator flags (RFI, bad channels,
# antennas dropping out, …) that are NOT reflected in the per-record WEIGHT
# column. Each row asserts that some (source, antenna(s), freqid, time range,
# band(s), channel range, polarization(s)) selection of UV_DATA cells is bad.
# We parse it eagerly into `FlagEntry`s (the table is tiny — a few thousand
# rows — vs the multi-GB FLUX matrix) and consult them lazily in the chunk
# array `readblock!`s, so building the UVSet never touches FLUX/WEIGHT.
#
# Columns (verified on the VLBA validation file; AIPS Memo 114):
#   SOURCE_ID(1J)  0 = all sources
#   ARRAY(1J)      (ignored — single array)
#   ANTS(2J)       one or two NOSTA antenna numbers; 0 = all/wildcard. One
#                  antenna flags every baseline touching it; two flag that
#                  baseline only.
#   FREQID(1J)     0 = all freq setups
#   TIMERANG(2E)   (start, end) in DAYS relative to RDATE. Multiply by 24 to
#                  get the reader's `t_hours` (hours since RDATE 00:00 UTC);
#                  TIMERANG is already RDATE-relative, so no jd0 term is added.
#   BANDS(NO_BAND J)  per-band flag, 1 = flag that band
#   CHANS(2J)      (lo, hi) 1-based channel range; (0,0) = all channels
#   PFLAGS(NO_STKD J) per-stokes flag in on-disk (RR/LL/RL/LR) order, 1 = flag
#   REASON(A), SEVERITY(1J)  (ignored)

# One parsed FLAG row. Antenna numbers are mapped NOSTA → global 1-based antenna
# index; `pflags` is permuted into MSv4 Pol order so it aligns with the leaf Pol
# axis. `bands` has length NO_BAND, `pflags` length NO_STKD.
struct FlagEntry
    source_id::Int            # 0 = all sources
    ant1::Int                 # global antenna index, 0 = wildcard
    ant2::Int                 # global antenna index, 0 = wildcard
    freqid::Int               # 0 = all freq setups
    t0::Float64               # hours since RDATE 00:00 UTC
    t1::Float64
    bands::Vector{Bool}       # per-band (length NO_BAND)
    chan_lo::Int              # 1-based channel lo (0 = all)
    chan_hi::Int              # 1-based channel hi (0 = all)
    pflags::Vector{Bool}      # per-pol, MSv4 order (length NO_STKD)
end

# True when this FLAG entry applies to band `band` (1-based). Out-of-range BANDS
# (shorter than NO_BAND) is treated as "flag" (defensive: a malformed/short
# BANDS column should not silently un-flag).
@inline _flag_band(e::FlagEntry, band::Int) =
    band <= length(e.bands) ? e.bands[band] : true

# True when global channel `ch` (1-based) is within this entry's channel range.
@inline function _flag_chan(e::FlagEntry, ch::Int)
    (e.chan_lo == 0 && e.chan_hi == 0) && return true
    lo = e.chan_lo == 0 ? 1 : e.chan_lo
    hi = e.chan_hi == 0 ? typemax(Int) : e.chan_hi
    return lo <= ch <= hi
end

# True when this entry's antenna selection matches a baseline (global antenna
# indices `a`, `b`). One antenna (other = 0) flags every baseline touching it;
# two antennas flag that baseline (either ordering); (0,0) flags all baselines.
@inline function _flag_baseline(e::FlagEntry, a::Int, b::Int)
    e.ant1 == 0 && e.ant2 == 0 && return true
    if e.ant2 == 0
        return e.ant1 == a || e.ant1 == b
    elseif e.ant1 == 0
        return e.ant2 == a || e.ant2 == b
    else
        return (e.ant1 == a && e.ant2 == b) || (e.ant1 == b && e.ant2 == a)
    end
end

# Parse the FLAG HDU into `FlagEntry`s. `nosta_to_idx` maps NOSTA → global
# antenna index; `perm[p]` is the on-disk stokes index for MSv4 pol `p`, so
# `pflags_msv4[p] = pflags_disk[perm[p]]` aligns PFLAGS with the leaf Pol axis.
function _build_idi_flags(flag_hdu, nosta_to_idx, perm, no_band, no_chan, no_stkd)
    flag_hdu === nothing && return FlagEntry[]
    d = flag_hdu.data
    # The FLAG table is small; read its columns eagerly. Missing columns fall
    # back to wildcards (older producers may omit some).
    n = Int(flag_hdu.data.format.shape[2])
    n == 0 && return FlagEntry[]

    _col(name) = hasproperty(d, name) ? collect(getproperty(d, name)) : nothing
    src = _col(:SOURCE_ID)
    ants = _col(:ANTS)
    fqid = _col(:FREQID)
    trang = _col(:TIMERANG)
    bands = _col(:BANDS)
    chans = _col(:CHANS)
    pflags = _col(:PFLAGS)

    _mapant(nosta) = nosta == 0 ? 0 : get(nosta_to_idx, Int(nosta), Int(nosta))

    out = Vector{FlagEntry}(undef, n)
    for r in 1:n
        sid = src === nothing ? 0 : Int(src isa AbstractMatrix ? src[r, 1] : src[r])
        ar = ants === nothing ? Int32[0, 0] : _idi_row(ants, r)
        a1 = _mapant(length(ar) >= 1 ? ar[1] : 0)
        a2 = _mapant(length(ar) >= 2 ? ar[2] : 0)
        fq = fqid === nothing ? 0 : Int(fqid isa AbstractMatrix ? fqid[r, 1] : fqid[r])

        if trang === nothing
            t0 = -Inf
            t1 = Inf
        else
            tr = _idi_row(trang, r)
            # TIMERANG is in DAYS relative to RDATE → hours.
            t0 = Float64(tr[1]) * 24.0
            t1 = Float64(length(tr) >= 2 ? tr[2] : tr[1]) * 24.0
            # (0,0) means "all times" in AIPS convention.
            (t0 == 0.0 && t1 == 0.0) && (t0 = -Inf; t1 = Inf)
        end

        bvec = if bands === nothing
            fill(true, no_band)
        else
            br = _idi_row(bands, r)
            Bool[(b <= length(br) ? br[b] != 0 : true) for b in 1:no_band]
        end

        if chans === nothing
            clo = 0
            chi = 0
        else
            cr = _idi_row(chans, r)
            clo = Int(cr[1])
            chi = Int(length(cr) >= 2 ? cr[2] : cr[1])
        end

        pvec_disk = if pflags === nothing
            fill(true, no_stkd)
        else
            pr = _idi_row(pflags, r)
            Bool[(s <= length(pr) ? pr[s] != 0 : true) for s in 1:no_stkd]
        end
        # Permute on-disk stokes order → MSv4 Pol order.
        pvec = Bool[pvec_disk[perm[p]] for p in 1:no_stkd]

        out[r] = FlagEntry(sid, a1, a2, fq, t0, t1, bvec, clo, chi, pvec)
    end
    return out
end

# Pre-filter the global flag list to those entries that can possibly touch a
# given leaf: matching source (or wildcard), flagging this band, and whose time
# range overlaps the scan's [t_lo, t_hi]. Done once at leaf construction so each
# `readblock!` iterates a short list. Channel/pol/baseline matching stays in the
# inner loop (cheap; the leaf has only one band).
function _filter_flags_for_leaf(entries, source_id, band, t_lo, t_hi)
    isempty(entries) && return FlagEntry[]
    out = FlagEntry[]
    for e in entries
        (e.source_id == 0 || e.source_id == source_id) || continue
        _flag_band(e, band) || continue
        # Time overlap (treat ±Inf endpoints as "all times").
        (e.t1 >= t_lo && e.t0 <= t_hi) || continue
        push!(out, e)
    end
    return out
end

# ── Lazy chunk array ─────────────────────────────────────────────────────────

"""
    IDIChunkArray{T, K} <: DiskArrays.AbstractDiskArray{T, 4}

Disk-backed `(Frequency, Ti, Baseline, Pol)` view of one band of one scan of
a FITS-IDI `UV_DATA` table. `kind::Val{:vis}` / `Val{:weights}` / `Val{:flag}`
selects the layer. No FLUX bytes are read until `readblock!` runs; each call
opens the file once (`open_lazy_source`), then per (time, baseline) cell seeks
to the cell's band slice, bulk-reads it in one `readbytes!`, byte-swaps it into
a reused scratch buffer, and closes in a `finally` — never per element.

Performance characteristics (measured on the 24 GB VLBA validation file, one
leaf = 256 chan × 240 ti × 21 bl × 4 pol ≈ 41 MB):
  * vis read ≈ 690 MB/s, full materialize (vis+weights+flag) ≈ 240 MB/s.
  * The disk read is NOT the bottleneck (~7% of profile); the byte-swap + copy
    loop was, so it uses a flat `Vector{Float32}` cube and direct linear
    indexing rather than `reshape(ntoh.(reinterpret(...)))`.

Contiguous-span fast path (considered, NOT implemented): with `SORT='T*'` the
rows of a (scan, band) are contiguous on disk, so one bulk read of the whole
row span followed by in-memory striding would replace the per-cell seeks. Not
done because (a) the per-cell seek path already hits ~690 MB/s — a large
fraction of this NVMe's ~2.5 GB/s sequential rate — at 41 MB/leaf, and (b) a
whole-span read pulls in the interleaved small columns / other bands too
(row stride 32880 B vs one band's 8192 B), reading ~4× the bytes; net win is
marginal here. Revisit if profiling on a slower/cold disk shows seek latency
dominating.

`row_of[ti, bl]` is the `UV_DATA` row backing cell `(ti, bl)`, or 0 when the
cell is missing (→ NaN vis / 0 weight / true flag).
"""
struct IDIChunkArray{T, K, TD, TFF, TW} <: DiskArrays.AbstractDiskArray{T, 4}
    data::TD                # LazyStructuredData (UV_DATA)
    flux_field::TFF         # BinaryField for FLUX
    weight_col::TW          # LazyFieldArray for WEIGHT (or nothing)
    begpos::Int             # data.begpos
    L::Int                  # row stride in bytes (format.shape[1])
    flux_M::Int             # FLUX byte offset within a row (first(slice)-1)
    band::Int               # 1-based band index
    no_stkd::Int            # NO_STKD (stokes axis length on disk)
    no_chan::Int            # NO_CHAN
    no_band::Int            # NO_BAND
    nperband::Int           # 2 * no_stkd * no_chan
    perm::Vector{Int}       # on-disk stokes idx → MSv4 Pol idx
    flux_scale::Bool        # whether FLUX has active TSCAL/TZERO
    row_of::Matrix{Int}     # (nti, nbl) UV_DATA row per cell (0 = missing)
    flags::Vector{FlagEntry}  # FLAG entries pre-filtered to this leaf
    bl_ants::Vector{Tuple{Int, Int}}  # global antenna pair per baseline column
    times::Vector{Float64}  # per-ti time (hours since RDATE) for TIMERANG match
    wfactor::Float32        # radiometer weight factor 2·Δν·η² for this band (0 ⇒ no scaling)
    inttim::Vector{Float32} # per-(global-row) INTTIM (s); empty when wfactor == 0
    # Per-integration autocorrelation normalization (correlation coefficients):
    auto_row::Matrix{Int}   # (nti, nant) UV_DATA row of the (a,a) record, 0 = absent
    normalize::Bool         # divide cross by √(A_a·A_b) per (chan, ti, feed); empty auto_row ⇒ no-op
    feed_pairs::Vector{Tuple{Int, Int}}  # per MSv4 pol → (feed_a, feed_b) ∈ {1,2}
    auto_stokes::NTuple{2, Int}          # on-disk stokes index of the (feed1,feed1) and (feed2,feed2) autocorr
    kind::K
end

function _idi_chunk(
        ::Type{T}, kind::K, data, flux_field, weight_col,
        band, no_stkd, no_chan, no_band, perm, flux_scale, row_of,
        flags, bl_ants, times, wfactor, inttim,
        auto_row, normalize, feed_pairs, auto_stokes,
    ) where {T, K}
    L = Int(data.format.shape[1])
    flux_M = first(flux_field.slice) - 1
    nperband = 2 * no_stkd * no_chan
    return IDIChunkArray{T, K, typeof(data), typeof(flux_field), typeof(weight_col)}(
        data, flux_field, weight_col, Int(data.begpos), L, flux_M,
        Int(band), Int(no_stkd), Int(no_chan), Int(no_band), nperband,
        Vector{Int}(perm), Bool(flux_scale), row_of,
        flags, bl_ants, Vector{Float64}(times),
        Float32(wfactor), Vector{Float32}(inttim),
        auto_row, Bool(normalize), Vector{Tuple{Int, Int}}(feed_pairs),
        (Int(auto_stokes[1]), Int(auto_stokes[2])), kind,
    )
end

# Decode the per-(chan, ti, feed) autocorrelation amplitude `Aspec[c, ti, ant, f]`
# = |autocorr(a,a) at the feed's parallel-hand stokes|, for every antenna present
# in this leaf's `auto_row`. `read_row!(cube, r)` fills `cube` (a length-`nperband`
# Float32 buffer) with UV_DATA row `r`'s band slice (from a span or from io).
# Used to form correlation coefficients: V_ab ← V_ab / √(A_a·A_b).
function _autocorr_spectra(a::IDIChunkArray, read_row!::F) where {F}
    nchan = a.no_chan
    nti, nant = size(a.auto_row)
    twostk = 2 * a.no_stkd
    Aspec = fill(NaN32, nchan, nti, nant, 2)
    cube = Vector{Float32}(undef, a.nperband)
    @inbounds for ant in 1:nant, ti in 1:nti
        r = a.auto_row[ti, ant]
        r == 0 && continue
        read_row!(cube, r)
        for f in 1:2
            base = (a.auto_stokes[f] - 1) * 2     # re/im offset of this feed's parallel-hand
            for c in 1:nchan
                o = (c - 1) * twostk + base
                Aspec[c, ti, ant, f] = abs(complex(cube[o + 1], cube[o + 2]))
            end
        end
    end
    return Aspec
end

# Autocorrelation spectra from a row-span buffer (bulk path) / from `io` (readblock).
_aspec_from_span(a::IDIChunkArray, span, rmin::Int, ::Type{D}) where {D} =
    _autocorr_spectra(a, (cube, r) -> _flux_from_span!(cube, span, a, r, rmin, D))
function _aspec_from_io(a::IDIChunkArray, io)
    rawbuf = Vector{UInt8}(undef, a.nperband * sizeof(a.flux_field.type))
    return _autocorr_spectra(a, (cube, r) -> _read_flux_cell!(cube, rawbuf, io, a, r))
end

# Form correlation coefficients in a dense vis block: `out[*, bl(a,b), pol] /=
# √(A_a·A_b)` with the feed-appropriate autocorr; NaN where an autocorr is
# missing/≤0 (the weights pass flags those). `r*` are the block's global index
# ranges (full-leaf ranges in the bulk path); `Aspec` is indexed by global (c, ti).
function _normalize_vis!(out, a::IDIChunkArray{T}, Aspec, rchan, rti, rbl, rpol) where {T}
    nan = T(complex(NaN32, NaN32))
    @inbounds for (bj, bl) in enumerate(rbl)
        ea, eb = a.bl_ants[bl]
        for (pj, p) in enumerate(rpol)
            fa, fb = a.feed_pairs[p]
            ok_feed = fa != 0 && fb != 0
            for (tj, ti) in enumerate(rti), (cj, c) in enumerate(rchan)
                d = ok_feed ? sqrt(Aspec[c, ti, ea, fa] * Aspec[c, ti, eb, fb]) : NaN32
                out[cj, tj, bj, pj] = (isfinite(d) && d > 0) ? out[cj, tj, bj, pj] / d : nan
            end
        end
    end
    return out
end

# Scale a dense weights block to track the correlation-coefficient normalization:
# noise ÷ √(A_a·A_b) ⇒ weight × (A_a·A_b). Cells with a missing/≤0 autocorr are
# flagged (weight ← 0), matching the NaN the vis pass writes there.
function _scale_weights!(out, a::IDIChunkArray{T}, Aspec, rchan, rti, rbl, rpol) where {T}
    @inbounds for (bj, bl) in enumerate(rbl)
        ea, eb = a.bl_ants[bl]
        for (pj, p) in enumerate(rpol)
            fa, fb = a.feed_pairs[p]
            ok_feed = fa != 0 && fb != 0
            for (tj, ti) in enumerate(rti), (cj, c) in enumerate(rchan)
                w = out[cj, tj, bj, pj]
                (w > 0 && isfinite(w)) || continue
                fac = ok_feed ? Aspec[c, ti, ea, fa] * Aspec[c, ti, eb, fb] : NaN32
                out[cj, tj, bj, pj] = (isfinite(fac) && fac > 0) ? w * fac : zero(T)
            end
        end
    end
    return out
end

# Per-cell radiometer weight scale: 2·Δν·τ·η² for UV_DATA row `r`, where the
# band-constant part `2·Δν·η²` is precomputed in `a.wfactor` and τ = INTTIM[r].
# Returns 1 when scaling is disabled (`wfactor == 0`, i.e. `weight_mode` was
# `:validity`, Δν was unavailable, or there was no INTTIM column).
@inline _wscale_phys(a::IDIChunkArray, r::Int) =
    a.wfactor == 0.0f0 ? 1.0f0 : a.wfactor * @inbounds(a.inttim[r])

Base.size(a::IDIChunkArray) =
    (a.no_chan, size(a.row_of, 1), size(a.row_of, 2), a.no_stkd)

DiskArrays.haschunks(::IDIChunkArray) = DiskArrays.Unchunked()

UVData._layer_is_lazy(::IDIChunkArray) = true

# Read band `b`'s contiguous `nperband` slice for UV_DATA row `i` into the
# reusable `cube` buffer (a plain `Vector{Float32}`), byte-swapping in place.
# `rawbuf` is a reusable `Vector{UInt8}` of `nperband*sizeof(dtype)` bytes.
#
# Indexing `cube` directly (a dense `Vector{Float32}`) is far faster than the
# previous `reshape(ntoh.(reinterpret(...)), …)` path, which materialized a
# lazy byte-swapped reinterpret wrapper and paid reinterpret-indexing cost on
# every one of the ~2048 element copies — that copy loop, not the disk read,
# dominated leaf materialization (≈68% of samples in a profile). Here we do one
# bulk `readbytes!` + one tight in-place `ntoh` loop into a plain array.
#
# On-disk per-band layout is column-major (2, no_stkd, no_chan): the flat index
# of (complex c∈{1,2}, stokes s, chan ch) is `(ch-1)*2*no_stkd + (s-1)*2 + c`.
@inline function _read_flux_cell!(
        cube::Vector{Float32}, rawbuf::Vector{UInt8}, io, a::IDIChunkArray, i::Integer,
    )
    dtype = a.flux_field.type
    nb = a.nperband
    off = a.begpos + a.L * (i - 1) + a.flux_M + sizeof(dtype) * ((a.band - 1) * nb)
    seek(io, off)
    readbytes!(io, rawbuf, nb * sizeof(dtype))
    _swap_into!(cube, rawbuf, dtype, nb, a.flux_field, a.flux_scale)
    return cube
end

# Byte-swap `nb` on-disk values of type `D` starting at pointer `p` into `cube`
# (`Float32`), optionally applying TSCAL/TZERO. Specialized per on-disk type so
# the hot loop has a concrete element type (DiFX Float32 fast path; Int16/Int32
# scaled paths for other producers). `p` may point into a per-row scratch buffer
# (`_swap_into!`) or directly into a multi-row span buffer (`_flux_from_span!`).
@inline function _swap_ptr!(cube, p::Ptr{D}, nb, field, scale::Bool) where {D}
    @inbounds if scale
        for k in 1:nb
            cube[k] = Float32(FITSFiles.scale_value(ntoh(unsafe_load(p, k)), field, true))
        end
    else
        for k in 1:nb
            cube[k] = Float32(ntoh(unsafe_load(p, k)))
        end
    end
    return cube
end

@inline _swap_into!(cube, rawbuf, ::Type{D}, nb, field, scale::Bool) where {D} =
    _swap_ptr!(cube, Ptr{D}(pointer(rawbuf)), nb, field, scale)

# Byte-swap `ns` on-disk WEIGHT values of type `D` from `wraw` into `wbuf`
# (`Float32`), optionally applying TSCAL/TZERO. Same flat-buffer + in-place
# `ntoh` loop as `_swap_into!` (the vis path); specialized per on-disk type so
# the hot loop has a concrete element type.
@inline function _swap_weights_ptr!(wbuf, p::Ptr{D}, ns, field, scale::Bool) where {D}
    @inbounds if scale
        for s in 1:ns
            wbuf[s] = Float32(FITSFiles.scale_value(ntoh(unsafe_load(p, s)), field, true))
        end
    else
        for s in 1:ns
            wbuf[s] = Float32(ntoh(unsafe_load(p, s)))
        end
    end
    return wbuf
end

@inline _swap_weights_into!(wbuf, wraw, ::Type{D}, ns, field, scale::Bool) where {D} =
    _swap_weights_ptr!(wbuf, Ptr{D}(pointer(wraw)), ns, field, scale)

# ── Bulk (one-read) materialization ──────────────────────────────────────────
#
# The reader's natural unit is a scan×band leaf, but the per-row `seek`+`read`
# pattern (one ~8 KB FLUX slice per (time, baseline) cell) is seek-bound: ~20k
# tiny random reads per scan group, ~110 MB/s on an NVMe that streams at GB/s.
# Since the file is `SORT='T*'`, a scan's UV_DATA rows are a CONTIGUOUS range and
# all of its bands share the SAME `row_of`, so `_materialize_group_bulk` reads the
# whole scan's row span ONCE (one big sequential read) and extracts every band's
# FLUX + WEIGHT from that buffer — one read per scan instead of (bands × layers ×
# cells) seeks.

# Cap on the transient span buffer; larger spans fall back to the (slow, seek-bound)
# per-leaf path. Sized so a real scan's whole row span fits the fast one-read path:
# VLBA Q-band scans span ~1 GB; VGOS VR2505 scans span up to ~3.34 GB. The solve's
# memory governor (`_group_peak_bytes` × `_bounded_ntasks`) bounds how many spans
# are resident at once (an over-budget group runs alone), so the cap can be generous;
# 6 GiB covers both files with margin and keeps every real scan on the fast path.
const _MAX_SPAN_BYTES = 6 * 1024 * 1024 * 1024   # 6 GiB

# Byte size above which the span read is split across concurrent handles. Below it,
# one sequential `readbytes!` already saturates the device; above it, multiple
# streams hide LUKS/dm-crypt latency (measured ~1 GB/s single-stream vs ~2-3 GB/s
# with 2-8 streams on the encrypted NVMe when a big scan runs alone — the scheduler
# runs over-budget scans alone, so there is no cross-group read parallelism then).
const _SPAN_STREAM_BYTES = 512 * 1024 * 1024     # 512 MiB
const _SPAN_MAX_STREAMS = 4

# Read the contiguous byte span covering UV_DATA rows [rmin, rmax] into one buffer
# (or `nothing` if it would exceed the cap). The file is time-sorted, so a scan's
# rows are contiguous. Small spans: one sequential read on `io`. Large spans: split
# the byte range across `nstream` concurrent handles (disjoint ranges → no data
# race) to recover device bandwidth the single buffered stream leaves on the table.
function _read_row_span(io, a::IDIChunkArray, rmin::Int, rmax::Int)
    nbytes = (rmax - rmin + 1) * a.L
    nbytes <= _MAX_SPAN_BYTES || return nothing
    span = Vector{UInt8}(undef, nbytes)
    base = a.begpos + a.L * (rmin - 1)
    nstream = nbytes >= _SPAN_STREAM_BYTES ?
        clamp(cld(nbytes, _SPAN_STREAM_BYTES), 1, _SPAN_MAX_STREAMS) : 1
    if nstream == 1
        seek(io, base)
        readbytes!(io, span, nbytes)
    else
        part = cld(nbytes, nstream)
        tforeach(1:nstream; ntasks = nstream) do s
            lo = (s - 1) * part
            len = min(part, nbytes - lo)
            len <= 0 && return
            sio = FITSFiles.open_lazy_source(a.data)
            try
                seek(sio, base + lo)
                GC.@preserve span unsafe_read(sio, pointer(span, lo + 1), len)
            finally
                close(sio)
            end
        end
    end
    return span
end

# Byte-swap row `r`'s FLUX band slice out of the span buffer (row `rmin` is the
# span's first row). Mirrors `_read_flux_cell!` but reads from memory, not `io`.
# `D` (the on-disk FLUX element type) is passed in as a CONCRETE type parameter so
# `Ptr{D}` and the `_swap_ptr!` dispatch are statically resolved. Reading it from
# `a.flux_field.type` here instead — as the code used to — is type-unstable
# (FITSFiles' `DataFormat.type::Type` is abstract), forcing one dynamic dispatch
# per row. The caller resolves `D` ONCE per leaf via the `_fill_*_dense!` barrier.
@inline function _flux_from_span!(cube, span::Vector{UInt8}, a::IDIChunkArray, r::Int, rmin::Int, ::Type{D}) where {D}
    nb = a.nperband
    off = a.L * (r - rmin) + a.flux_M + sizeof(D) * ((a.band - 1) * nb)
    return _swap_ptr!(cube, Ptr{D}(pointer(span, off + 1)), nb, a.flux_field, a.flux_scale)
end

# Byte-swap row `r`'s WEIGHT block out of the span buffer (mirror of
# `_read_weight_row!`).
@inline function _weights_from_span!(wbuf, span::Vector{UInt8}, a::IDIChunkArray, r::Int, rmin::Int, scale::Bool, ::Type{D}) where {D}
    if a.weight_col === nothing
        fill!(wbuf, 1.0f0)
        return wbuf
    end
    wf = a.weight_col.fields[1]
    ns = a.no_stkd
    off = a.L * (r - rmin) + (first(wf.slice) - 1) + sizeof(D) * ((a.band - 1) * ns)
    return _swap_weights_ptr!(wbuf, Ptr{D}(pointer(span, off + 1)), ns, wf, scale)
end

# Fill a dense (no_chan, nti, nbl, no_stkd) vis array (MSv4 pol order) from the
# span — the full-leaf equivalent of the vis `readblock!` loop.
# Outer method: resolve the abstract on-disk dtype `a.flux_field.type` to a
# concrete `Type{D}` ONCE, then dispatch into the typed kernel. Everything past
# the barrier is type-stable (no per-row dynamic dispatch on the FLUX dtype).
function _fill_vis_dense!(out, a::IDIChunkArray{T, Val{:vis}}, span, rmin::Int) where {T}
    return _fill_vis_dense!(out, a, span, rmin, a.flux_field.type)
end

@noinline function _fill_vis_dense!(out, a::IDIChunkArray{T, Val{:vis}}, span, rmin::Int, ::Type{D}) where {T, D}
    nbl = size(a.row_of, 2)
    # Thread over baseline columns: decode is CPU-bound (byte-swap + complex repack
    # + permute), so spreading columns across the cores the memory cap leaves idle
    # lifts materialize throughput. The per-baseline work is a NAMED function (not a
    # `do`-closure) so it specializes cleanly — a closure here boxed its captures
    # and halved single-thread speed. Nested under the solve's group-level `tmap`,
    # but Julia's scheduler caps live tasks at `nthreads`, so it composes.
    nt = _decode_ntasks(nbl)
    if nt == 1
        for bl in 1:nbl
            _decode_vis_bl!(out, a, span, rmin, D, bl)
        end
    else
        tforeach(bl -> _decode_vis_bl!(out, a, span, rmin, D, bl), 1:nbl; ntasks = nt)
    end
    if a.normalize
        Aspec = _aspec_from_span(a, span, rmin, D)
        _normalize_vis!(out, a, Aspec, 1:a.no_chan, 1:size(a.row_of, 1), 1:nbl, 1:a.no_stkd)
    end
    return out
end

# Decode one baseline column (all times) of a vis leaf from the span into `out`.
@inline function _decode_vis_bl!(out, a::IDIChunkArray{T, Val{:vis}}, span, rmin::Int, ::Type{D}, bl::Int) where {T, D}
    nchan = a.no_chan
    nti = size(a.row_of, 1)
    npol = a.no_stkd
    twostk = 2 * a.no_stkd
    nan = T(complex(NaN32, NaN32))
    cube = Vector{Float32}(undef, a.nperband)
    @inbounds for ti in 1:nti
        r = a.row_of[ti, bl]
        if r == 0
            for p in 1:npol, c in 1:nchan
                out[c, ti, bl, p] = nan
            end
            continue
        end
        _flux_from_span!(cube, span, a, r, rmin, D)
        for p in 1:npol
            s = a.perm[p]
            base = (s - 1) * 2
            for c in 1:nchan
                o = (c - 1) * twostk + base
                out[c, ti, bl, p] = T(complex(cube[o + 1], cube[o + 2]))
            end
        end
    end
    return nothing
end

# Fill a dense (no_chan, nti, nbl, no_stkd) weights array from the span — the
# full-leaf equivalent of the weights `readblock!` loop (FLAG-table aware).
# Barrier: resolve the abstract on-disk WEIGHT dtype once (Float32 placeholder when
# there is no WEIGHT column — `_weights_from_span!` fills 1.0 then), then run the
# type-stable, baseline-threaded kernel.
function _fill_weights_dense!(out, a::IDIChunkArray{T, Val{:weights}}, span, rmin::Int) where {T}
    D = a.weight_col === nothing ? Float32 : a.weight_col.fields[1].type
    return _fill_weights_dense!(out, a, span, rmin, D)
end

@noinline function _fill_weights_dense!(out, a::IDIChunkArray{T, Val{:weights}}, span, rmin::Int, ::Type{D}) where {T, D}
    nbl = size(a.row_of, 2)
    nt = _decode_ntasks(nbl)
    if nt == 1
        for bl in 1:nbl
            _decode_weights_bl!(out, a, span, rmin, D, bl)
        end
    else
        tforeach(bl -> _decode_weights_bl!(out, a, span, rmin, D, bl), 1:nbl; ntasks = nt)
    end
    if a.normalize
        Aspec = _aspec_from_span(a, span, rmin, a.flux_field.type)
        _scale_weights!(out, a, Aspec, 1:a.no_chan, 1:size(a.row_of, 1), 1:nbl, 1:a.no_stkd)
    end
    return out
end

# Decode one baseline column (all times) of a weights leaf from the span (FLAG-aware).
@inline function _decode_weights_bl!(out, a::IDIChunkArray{T, Val{:weights}}, span, rmin::Int, ::Type{D}, bl::Int) where {T, D}
    nchan = a.no_chan
    nti = size(a.row_of, 1)
    npol = a.no_stkd
    wscale = _weight_scale(a)
    have_flags = !isempty(a.flags)
    wbuf = Vector{Float32}(undef, a.no_stkd)
    ea, eb = a.bl_ants[bl]
    @inbounds for ti in 1:nti
        r = a.row_of[ti, bl]
        r != 0 && _weights_from_span!(wbuf, span, a, r, rmin, wscale, D)
        sc = r == 0 ? 1.0f0 : _wscale_phys(a, r)   # radiometer scale (1 if disabled)
        t = a.times[ti]
        for p in 1:npol
            wv = r == 0 ? 0.0f0 : wbuf[a.perm[p]]
            # Scale only valid (positive) weights; <=0 stays a flag sentinel.
            w = wv > 0 ? T(wv * sc) : T(wv)
            if have_flags && r != 0 && w > 0
                for c in 1:nchan
                    out[c, ti, bl, p] = _idi_cell_flagged(a, c, t, ea, eb, p) ? zero(T) : w
                end
            else
                for c in 1:nchan
                    out[c, ti, bl, p] = w
                end
            end
        end
    end
    return nothing
end

# Mark IDI-backed leaves as bulk-capable and provide the one-read group reader.
UVData._bulk_backend(a::IDIChunkArray) = a

function UVData._materialize_group_bulk(leaves, layers)
    isempty(leaves) && return nothing
    a1 = parent(first(leaves)[:vis])
    a1 isa IDIChunkArray || return nothing
    # All leaves must be sibling bands of one scan: same UV_DATA + same row map.
    for l in leaves
        av = parent(l[:vis])
        (
            av isa IDIChunkArray && av.data === a1.data && av.row_of === a1.row_of &&
                av.begpos == a1.begpos && av.L == a1.L
        ) || return nothing
    end
    rmin = typemax(Int)
    rmax = 0
    @inbounds for r in a1.row_of
        r == 0 && continue
        r < rmin && (rmin = r)
        r > rmax && (rmax = r)
    end
    # The autocorrelation rows (used as correlation-coefficient normalizers) must
    # also be inside the span, or `_aspec_from_span` reads past the buffer.
    @inbounds for r in a1.auto_row
        r == 0 && continue
        r < rmin && (rmin = r)
        r > rmax && (rmax = r)
    end
    rmax == 0 && return nothing                      # empty scan → let caller fall back
    io = FITSFiles.open_lazy_source(a1.data)
    local span
    try
        span = _read_row_span(io, a1, rmin, rmax)
    finally
        close(io)
    end
    span === nothing && return nothing               # span over the cap → fall back

    want_flag = :flag in layers
    return map(leaves) do l
        av = parent(l[:vis])
        aw = parent(l[:weights])
        nti, nbl = size(av.row_of)
        vis_dense = Array{eltype(av)}(undef, av.no_chan, nti, nbl, av.no_stkd)
        _fill_vis_dense!(vis_dense, av, span, rmin)
        w_dense = Array{Float32}(undef, aw.no_chan, nti, nbl, aw.no_stkd)
        _fill_weights_dense!(w_dense, aw, span, rmin)
        vis_da = DimArray(vis_dense, dims(l[:vis]))
        w_da = DimArray(w_dense, dims(l[:weights]))
        uvw_da = DimArray(UVData._materialize_layer(parent(l[:uvw])), dims(l[:uvw]))
        flag_da = want_flag ? DimArray(UVData._materialize_layer(parent(l[:flag])), dims(l[:flag])) : nothing
        UVData._build_leaf(vis_da, w_da, uvw_da, flag_da; partition_info = DimensionalData.metadata(l))
    end
end

# Direct-into-destination variant of `_materialize_group_bulk`: read the scan's
# shared row span ONCE, then decode each band's vis/weights straight into the
# caller's `dests[i] = (vis_dest, weights_dest)` (typically contiguous channel-block
# views of one stacked cube) — skipping the per-band intermediate dense arrays and
# the uvw/flag layers the fringe search never uses. The decode kernels write
# `out[c, ti, bl, p]` generically, so a SubArray destination works (axis-1 = the
# band's channel block, stride 1). Returns `false` to fall back if the leaves are
# not a single sibling-band scan span (then the caller uses materialize_group + copy).
function UVData._materialize_group_bulk_into!(dests, leaves, layers)
    isempty(leaves) && return false
    a1 = parent(first(leaves)[:vis])
    a1 isa IDIChunkArray || return false
    for l in leaves
        av = parent(l[:vis])
        (
            av isa IDIChunkArray && av.data === a1.data && av.row_of === a1.row_of &&
                av.begpos == a1.begpos && av.L == a1.L
        ) || return false
    end
    rmin = typemax(Int)
    rmax = 0
    @inbounds for r in a1.row_of
        r == 0 && continue
        r < rmin && (rmin = r)
        r > rmax && (rmax = r)
    end
    @inbounds for r in a1.auto_row          # autocorr normalizers must be in-span too
        r == 0 && continue
        r < rmin && (rmin = r)
        r > rmax && (rmax = r)
    end
    rmax == 0 && return false
    io = FITSFiles.open_lazy_source(a1.data)
    local span
    try
        span = _read_row_span(io, a1, rmin, rmax)
    finally
        close(io)
    end
    span === nothing && return false        # span over the cap → fall back
    for (i, l) in enumerate(leaves)
        av = parent(l[:vis])
        aw = parent(l[:weights])
        vis_dest, w_dest = dests[i]
        _fill_vis_dense!(vis_dest, av, span, rmin)
        _fill_weights_dense!(w_dest, aw, span, rmin)
    end
    return true
end

# Read this band's `no_stkd` WEIGHT entries for UV_DATA row `i` into `wbuf`
# (length `no_stkd`) in one seek+read. WEIGHT linear order is stokes-fastest
# then band, so this band's block starts at element (band-1)*no_stkd + 1 and is
# contiguous. `wbuf[s]` is the weight for on-disk stokes `s`.
#
# `wraw` is a reusable `Vector{UInt8}` scratch (≥ ns*sizeof(type) bytes). Like
# the vis path, this does one bulk `readbytes!` + one tight in-place `ntoh` loop
# instead of `read_bigendian`'s `ntoh.(reinterpret(type, read(...)))`, which
# allocated two vectors per row and paid reinterpret-indexing cost — a per-row
# cost on the WEIGHT read repeated for every (baseline, time) cell.
@inline function _read_weight_row!(
        wbuf::Vector{Float32}, wraw::Vector{UInt8}, io, a::IDIChunkArray, i::Integer, scale::Bool,
    )
    if a.weight_col === nothing
        fill!(wbuf, 1.0f0)
        return wbuf
    end
    wf = a.weight_col.fields[1]
    ns = a.no_stkd
    off = a.begpos + a.L * (i - 1) + (first(wf.slice) - 1) +
        sizeof(wf.type) * ((a.band - 1) * ns)
    seek(io, off)
    readbytes!(io, wraw, ns * sizeof(wf.type))
    _swap_weights_into!(wbuf, wraw, wf.type, ns, wf, scale)
    return wbuf
end

# Whether this leaf's WEIGHT column carries active TSCAL/TZERO (computed once per
# readblock, then passed into the per-row reader so the hot loop can skip the
# scaling branch entirely on the DiFX Float32 fast path).
@inline _weight_scale(a::IDIChunkArray) =
    a.weight_col === nothing ? false :
    FITSFiles._has_active_scaling(a.weight_col.fields[1], true)

# Raw-byte scratch buffer big enough for one band's WEIGHT block.
@inline _weight_rawbuf(a::IDIChunkArray) =
    a.weight_col === nothing ? Vector{UInt8}(undef, 0) :
    Vector{UInt8}(undef, a.no_stkd * sizeof(a.weight_col.fields[1].type))

# True when any FLAG entry on this leaf flags cell (global channel `ch`, time
# `t`, baseline column `bl` with global antennas (ea,eb), MSv4 pol `p`). The
# entry list is already pre-filtered to this leaf's (source, band, time-span),
# so this only checks the per-cell axes (time, channel, pol, baseline).
@inline function _idi_cell_flagged(a::IDIChunkArray, ch::Int, t::Float64, ea::Int, eb::Int, p::Int)
    @inbounds for e in a.flags
        (t >= e.t0 && t <= e.t1) || continue
        e.pflags[p] || continue
        _flag_chan(e, ch) || continue
        _flag_baseline(e, ea, eb) || continue
        return true
    end
    return false
end

function DiskArrays.readblock!(
        a::IDIChunkArray{T, Val{:vis}}, out,
        rchan::AbstractUnitRange, rti::AbstractUnitRange,
        rbl::AbstractUnitRange, rpol::AbstractUnitRange,
    ) where {T}
    io = FITSFiles.open_lazy_source(a.data)
    # Reused scratch: raw bytes + byte-swapped Float32 cube for one row's band
    # slice. Allocated once per readblock! (one materialization), not per row.
    cube = Vector{Float32}(undef, a.nperband)
    rawbuf = Vector{UInt8}(undef, a.nperband * sizeof(a.flux_field.type))
    twostk = 2 * a.no_stkd
    try
        nan = T(complex(NaN32, NaN32))
        @inbounds for (bj, bl) in enumerate(rbl), (tj, ti) in enumerate(rti)
            r = a.row_of[ti, bl]
            if r == 0
                for (pj, _) in enumerate(rpol), cj in eachindex(rchan)
                    out[cj, tj, bj, pj] = nan
                end
                continue
            end
            _read_flux_cell!(cube, rawbuf, io, a, r)   # flat (2, no_stkd, no_chan)
            for (pj, p) in enumerate(rpol)
                s = a.perm[p]                            # on-disk stokes for MSv4 pol p
                base = (s - 1) * 2                        # +1 → re, +2 → im for chan 1
                for (cj, c) in enumerate(rchan)
                    o = (c - 1) * twostk + base
                    out[cj, tj, bj, pj] = T(complex(cube[o + 1], cube[o + 2]))
                end
            end
        end
        if a.normalize
            _normalize_vis!(out, a, _aspec_from_io(a, io), rchan, rti, rbl, rpol)
        end
    finally
        close(io)
    end
    return out
end

function DiskArrays.readblock!(
        a::IDIChunkArray{T, Val{:weights}}, out,
        rchan::AbstractUnitRange, rti::AbstractUnitRange,
        rbl::AbstractUnitRange, rpol::AbstractUnitRange,
    ) where {T}
    io = FITSFiles.open_lazy_source(a.data)
    wbuf = Vector{Float32}(undef, a.no_stkd)
    wraw = _weight_rawbuf(a)
    wscale = _weight_scale(a)
    have_flags = !isempty(a.flags)
    try
        @inbounds for (bj, bl) in enumerate(rbl), (tj, ti) in enumerate(rti)
            r = a.row_of[ti, bl]
            r != 0 && _read_weight_row!(wbuf, wraw, io, a, r, wscale)
            sc = r == 0 ? 1.0f0 : _wscale_phys(a, r)   # radiometer scale (1 if disabled)
            ea, eb = a.bl_ants[bl]
            t = a.times[ti]
            for (pj, p) in enumerate(rpol)
                wv = r == 0 ? 0.0f0 : wbuf[a.perm[p]]
                # Scale only valid (positive) weights; <=0 stays a flag sentinel.
                w = wv > 0 ? T(wv * sc) : T(wv)
                # FLAG-table entries zero the weight (channel-dependent when
                # CHANS is set). OR'd with the existing weight<=0 path.
                if have_flags && r != 0 && w > 0
                    for (cj, c) in enumerate(rchan)
                        out[cj, tj, bj, pj] =
                            _idi_cell_flagged(a, c, t, ea, eb, p) ? zero(T) : w
                    end
                else
                    for cj in eachindex(rchan)
                        out[cj, tj, bj, pj] = w
                    end
                end
            end
        end
        if a.normalize
            _scale_weights!(out, a, _aspec_from_io(a, io), rchan, rti, rbl, rpol)
        end
    finally
        close(io)
    end
    return out
end

function DiskArrays.readblock!(
        a::IDIChunkArray{T, Val{:flag}}, out,
        rchan::AbstractUnitRange, rti::AbstractUnitRange,
        rbl::AbstractUnitRange, rpol::AbstractUnitRange,
    ) where {T}
    io = FITSFiles.open_lazy_source(a.data)
    wbuf = Vector{Float32}(undef, a.no_stkd)
    wraw = _weight_rawbuf(a)
    wscale = _weight_scale(a)
    have_flags = !isempty(a.flags)
    try
        @inbounds for (bj, bl) in enumerate(rbl), (tj, ti) in enumerate(rti)
            r = a.row_of[ti, bl]
            r != 0 && _read_weight_row!(wbuf, wraw, io, a, r, wscale)
            ea, eb = a.bl_ants[bl]
            t = a.times[ti]
            for (pj, p) in enumerate(rpol)
                # weight<=0 path (or missing row) flags the whole channel run;
                # otherwise consult the FLAG table per channel.
                wflag = r == 0 || wbuf[a.perm[p]] <= 0
                if wflag || !have_flags
                    for cj in eachindex(rchan)
                        out[cj, tj, bj, pj] = wflag
                    end
                else
                    for (cj, c) in enumerate(rchan)
                        out[cj, tj, bj, pj] = _idi_cell_flagged(a, c, t, ea, eb, p)
                    end
                end
            end
        end
    finally
        close(io)
    end
    return out
end

# ── Index pass ───────────────────────────────────────────────────────────────
#
# Reading the small per-row columns (DATE/TIME/BASELINE/SOURCE/FREQID/INTTIM/
# UVW) one column at a time through FITSFiles costs one full strided traversal
# of the 24 GB file *per column* (FITSFiles seeks once per row and reads a
# single scalar — see `Base.read(io, ::BinaryField, …)`), so nine columns ≈ 9 ×
# 35 s ≈ 5 min on the validation file. All these columns live in the first 48
# bytes of every 32880-byte row, so `_idi_read_small_columns` instead makes ONE
# pass, seeking to each row and reading only its leading prefix into a reused
# scratch buffer, then decoding every requested field from that buffer. That is
# ~13× faster (one pass, 48 B/row instead of nine passes of 8 B/row) and reads
# ~36 MB total instead of touching 24 GB nine times.

# Big-endian decode of one scalar `T` at byte offset `o` (1-based) in `buf`.
@inline function _be_scalar(::Type{T}, buf::Vector{UInt8}, o::Int) where {T}
    x = unsafe_load(Ptr{T}(pointer(buf, o)))
    return ntoh(x)
end

# Read the requested scalar columns of a FITS-IDI bintable in a single strided
# pass. `cols` is a vector of (name::Symbol, T::Type) pairs; each named field is
# located via `data.fields` (its `slice` gives the byte offset within a row).
# Returns a NamedTuple of dense column vectors. Only the leading prefix covering
# all requested fields is read per row.
function _idi_read_small_columns(data, cols)
    L = Int(data.format.shape[1])
    n = Int(data.format.shape[2])
    begpos = Int(data.begpos)

    # Resolve each requested column to (output_type, byte_offset, on_disk_type).
    field_by_name = Dict(Symbol(f.name) => f for f in data.fields)
    specs = map(cols) do (name, T)
        f = field_by_name[Symbol(name)]
        f.leng == 1 || error(
            "_idi_read_small_columns: $(name) is not scalar (leng=$(f.leng))",
        )
        (; out = T, off = Int(first(f.slice)), dtype = f.type, field = f)
    end
    prefix = maximum(s -> s.off - 1 + sizeof(s.dtype), specs; init = 0)

    outs = map(s -> Vector{s.out}(undef, n), specs)
    buf = Vector{UInt8}(undef, prefix)
    io = FITSFiles.open_lazy_source(data)
    try
        @inbounds for r in 1:n
            seek(io, begpos + L * (r - 1))
            readbytes!(io, buf, prefix)
            for (ci, s) in enumerate(specs)
                v = _decode_field(s.dtype, buf, s.off)
                v = FITSFiles.scale_value(v, s.field, true)
                outs[ci][r] = s.out(v)
            end
        end
    finally
        close(io)
    end
    return NamedTuple{Tuple(first.(cols))}(Tuple(outs))
end

# Decode one big-endian scalar of the on-disk type at 1-based byte offset `o`.
_decode_field(::Type{Float64}, buf, o) = _be_scalar(Float64, buf, o)
_decode_field(::Type{Float32}, buf, o) = _be_scalar(Float32, buf, o)
_decode_field(::Type{Int32}, buf, o) = _be_scalar(Int32, buf, o)
_decode_field(::Type{Int16}, buf, o) = _be_scalar(Int16, buf, o)
_decode_field(::Type{Int64}, buf, o) = _be_scalar(Int64, buf, o)

# Segment time-ordered rows into scans: a new scan starts when SOURCE_ID
# changes or the time gap exceeds `gap_hours`. Returns a Vector of (range,
# source_id) tuples (ranges index into the time-ordered row arrays).
function _idi_segment_scans(source_ids, t_hours, gap_hours)
    n = length(t_hours)
    segs = Tuple{UnitRange{Int}, Int}[]
    n == 0 && return segs
    start = 1
    for i in 2:n
        if source_ids[i] != source_ids[start] || (t_hours[i] - t_hours[i - 1]) > gap_hours
            push!(segs, (start:(i - 1), source_ids[start]))
            start = i
        end
    end
    push!(segs, (start:n, source_ids[start]))
    return segs
end

# ── Public entry point ───────────────────────────────────────────────────────

"""
    load_fitsidi(path; lazy=true, scans=:, bands=:,
                 weight_mode=:validity, weight_efficiency=1.0, drop_autocorr=true) -> UVSet

Load a FITS-IDI file into a (lazily streamed) `UVSet`.

`normalize_autocorr` (default `true`) divides each cross-correlation by the
per-(channel, integration, feed) autocorrelations, `V_ab ← V_ab/√(A_a·A_b)`,
forming true correlation coefficients (flattens the amplitude bandpass; weights
scale by `A_a·A_b`). This is what makes the data match the HOPS/rPICARD
"correlation coefficient" convention. The autocorrelations are then consumed
(the `a==a` baselines are dropped). A no-op on data with no autocorrelations.

`drop_autocorr` (default `true`) excludes autocorrelation baselines (antenna
`a == a`) from the visibility set even when not normalizing (total power, not
interferometric). `normalize_autocorr=true` implies the autocorrelations are
dropped from the output regardless. Pass both `false` to keep autocorrelations.

`weight_mode` controls how the on-disk `WEIGHT` column is interpreted:

  * `:validity` (default) — return the correlator weights verbatim. These are
    DiFX/FITS-IDI *validity* weights (fraction of the sample actually
    correlated, ≈1), NOT inverse variances, so `1/√weight` is not a noise.
  * `:radiometer` — convert to thermal-noise inverse variances in the data's
    (correlation-coefficient) units: `w → (w / weight_norm) · 2·Δν·τ·η²`, with
    channel width `Δν` from the FREQUENCY table (`CH_WIDTH`) and per-record
    accumulation time `τ` from the `INTTIM` column. `weight_efficiency` (η) is the
    correlator/quantization efficiency (a single scalar; it sets the absolute
    χ²/SNR scale, not the relative weighting). Tsys is not needed — it cancels
    in correlation-coefficient units. Non-positive weights (flag sentinels) are
    preserved; bands lacking a usable `CH_WIDTH`/`INTTIM` fall back to
    `:validity`.

    `weight_norm` is the nominal fully-valid value of the on-disk `WEIGHT` column.
    The radiometer formula treats `WEIGHT` as a ≈1 validity *fraction*, but some
    correlators write an unnormalized value (e.g. DiFX BT164 ≈ 3.4); leaving
    `weight_norm = 1.0` then overstates the inverse-variance by that factor. Set it
    to `median(positive WEIGHT)` so a fully-valid cell lands at the correct
    `2·Δν·τ·η²`. A successive-difference scatter test (`|ΔV|²/(σᵢ²+σⱼ²) ≈ 1`) on the
    output is the way to confirm the absolute scale.
"""
function UVData.load_fitsidi(
        path; lazy = true, scans = :, bands = :,
        weight_mode::Symbol = :validity, weight_efficiency::Real = 1.0,
        weight_norm::Real = 1.0,
        drop_autocorr::Bool = true, normalize_autocorr::Bool = true,
    )
    weight_mode in (:validity, :radiometer) || error(
        "load_fitsidi: weight_mode must be :validity (raw correlator validity " *
            "weights, as on disk) or :radiometer (convert to thermal-noise " *
            "inverse-variance via 2·Δν·τ·η²); got $(weight_mode).",
    )
    weight_norm > 0 || error("load_fitsidi: weight_norm must be positive, got $(weight_norm).")
    fid = FITSFiles.fits(path)

    # Locate HDUs by EXTNAME.
    primary_hdu = fid[1]
    ag_hdu = an_hdu = src_hdu = fq_hdu = uv_hdu = flag_hdu = nothing
    n_uv = 0
    for hdu in fid[2:end]
        ext = strip(string(something(card_value(hdu.cards, "EXTNAME"), "")))
        ext == "ARRAY_GEOMETRY" && (ag_hdu = hdu)
        ext == "ANTENNA" && (an_hdu = hdu)
        ext == "SOURCE" && (src_hdu = hdu)
        ext == "FREQUENCY" && (fq_hdu = hdu)
        ext == "FLAG" && (flag_hdu = hdu)
        if ext == "UV_DATA"
            uv_hdu = hdu
            n_uv += 1
        end
    end
    ag_hdu === nothing && error("load_fitsidi: no ARRAY_GEOMETRY HDU in $(path)")
    an_hdu === nothing && error("load_fitsidi: no ANTENNA HDU in $(path)")
    src_hdu === nothing && error("load_fitsidi: no SOURCE HDU in $(path)")
    fq_hdu === nothing && error("load_fitsidi: no FREQUENCY HDU in $(path)")
    uv_hdu === nothing && error("load_fitsidi: no UV_DATA HDU in $(path)")
    # Multiple UV_DATA HDUs would silently merge into the last one's geometry;
    # we do not support that (each can carry a different setup). Error clearly.
    n_uv > 1 && error(
        "load_fitsidi: $(n_uv) UV_DATA HDUs found in $(path); only a single " *
            "UV_DATA HDU is supported.",
    )

    uv_cards = uv_hdu.cards
    no_stkd = Int(something(card_value(uv_cards, "NO_STKD"), 4))
    no_band = Int(something(card_value(uv_cards, "NO_BAND"), 1))
    no_chan = Int(something(card_value(uv_cards, "NO_CHAN"), 1))
    ref_freq = Float64(something(card_value(uv_cards, "REF_FREQ"), 0.0))

    # Eager metadata.
    antennas = _build_idi_antenna_table(ag_hdu, an_hdu)
    nosta = extras(antennas).NOSTA::Vector{Int32}
    # The reader decodes BASELINE as `256*a + b`; that only round-trips for
    # NOSTA < 256. AIPS' alternate 2048-packing (`bl = 2048*a + b + 0.01*subarray`)
    # is a different convention and is not supported.
    let bad = filter(s -> Int(s) >= 256, nosta)
        isempty(bad) || error(
            "load_fitsidi: NOSTA values $(Int.(bad)) are >= 256; the 256*a+b " *
                "BASELINE packing assumed here cannot represent them (the AIPS " *
                "2048-packing variant is unsupported).",
        )
    end
    # NOSTA → 1-based antenna index (row in the antenna table).
    nosta_to_idx = Dict{Int, Int}(Int(nosta[i]) => i for i in eachindex(nosta))
    array_obs = _build_idi_array_obs(primary_hdu, ag_hdu)
    # Multi-setup (>1 FREQID) files carry several frequency configurations; the
    # reader only uses the first FREQID row. Warn rather than silently drop.
    let nfreqid = Int(fq_hdu.data.format.shape[2])
        nfreqid > 1 && @warn(
            "load_fitsidi: FREQUENCY table has $(nfreqid) FREQIDs; only the " *
                "first is used."
        )
    end
    freq_setups = _build_idi_freq_setups(fq_hdu, ref_freq, no_chan, no_band)
    src_table = _build_idi_sources(src_hdu)
    _, _, msv4_labels, perm = _idi_stokes_perm(uv_cards)

    # Autocorrelation-normalization maps (constant across scans/bands): per MSv4
    # pol → (feed_a, feed_b), and the on-disk stokes of the two parallel-hand
    # autocorr products (PP→feed1, QQ→feed2; 0 if that hand is absent). MSv4
    # canonical labels are P/Q.
    _feedchar(c) = c == 'P' ? 1 : c == 'Q' ? 2 : 0
    feed_pairs = Tuple{Int, Int}[(_feedchar(first(lab)), _feedchar(last(lab))) for lab in msv4_labels]
    _idx(lab) = (i = findfirst(==(lab), msv4_labels); i === nothing ? 0 : Int(perm[i]))
    auto_stokes = (_idx("PP"), _idx("QQ"))

    # Parse the FLAG table eagerly (small; the FLUX matrix is never touched).
    flag_entries = _build_idi_flags(flag_hdu, nosta_to_idx, perm, no_band, no_chan, no_stkd)

    # Index pass: small per-row columns only (no WEIGHT / FLUX). Read in ONE
    # strided pass over the row prefixes (see `_idi_read_small_columns`); doing
    # it column-by-column through FITSFiles would traverse the whole file once
    # per column (~5 min on the 24 GB validation file vs ~25 s here).
    data = uv_hdu.data
    small = _idi_read_small_columns(
        data, [
            (:DATE, Float64), (:TIME, Float64),
            (:BASELINE, Int32), (:SOURCE, Int32), (:INTTIM, Float32),
            (Symbol("UU---SIN"), Float32),
            (Symbol("VV---SIN"), Float32),
            (Symbol("WW---SIN"), Float32),
        ],
    )
    date = small.DATE
    tim = small.TIME
    bl_codes = Int.(small.BASELINE)
    source_ids = Int.(small.SOURCE)
    inttim = small.INTTIM
    uu = small[Symbol("UU---SIN")]
    vv = small[Symbol("VV---SIN")]
    ww = small[Symbol("WW---SIN")]

    # Time in fractional hours since RDATE 00:00 UTC.
    jd0 = _idi_rdate_jd(array_obs.rdate)
    t_hours = ((date .+ tim) .- jd0) .* 24.0

    # FLUX field descriptor (read once; shared by all chunk arrays).
    flux_col = data[:FLUX]
    flux_field = flux_col.fields[1]
    flux_scale = FITSFiles._has_active_scaling(flux_field, true)
    weight_col = data[:WEIGHT]

    # Scan segmentation. Data is SORT='T*' (time-ordered).
    med_int_hours = isempty(inttim) ? 0.0 : Float64(median(inttim)) / 3600.0
    gap_hours = 2 * med_int_hours
    gap_hours <= 0 && (gap_hours = Inf)   # single integration → one scan
    segments = _idi_segment_scans(source_ids, t_hours, gap_hours)

    # Optional scan / band restriction.
    band_sel = bands === Colon() ? (1:no_band) : collect(bands)
    scan_sel = scans === Colon() ? (1:length(segments)) : collect(scans)

    branches = DimensionalData.TreeDict()
    base_name = String(splitext(basename(String(path)))[1])
    eltype_vis = ComplexF32

    for (scan_i, seg) in enumerate(segments)
        scan_i in scan_sel || continue
        rng, sid = seg
        si = src_table[sid]
        source_key = UVData.sanitize_source(si.name)

        # Per-(ti, bl) row map shared across bands within the scan.
        seg_t = @view t_hours[rng]
        seg_bl = @view bl_codes[rng]
        unique_times = sort(unique(seg_t))
        time_lookup = Dict(t => i for (i, t) in enumerate(unique_times))

        # Autocorrelations are dropped from the output baseline set when either
        # dropping or normalizing (normalize consumes them as the √(A_a·A_b)
        # normalizer, then they are no longer science visibilities).
        excl_auto = drop_autocorr || normalize_autocorr
        bl_pairs = Tuple{Int, Int}[]
        seen_bl = Set{Tuple{Int, Int}}()
        for code in seg_bl
            a = code ÷ 256
            b = code % 256
            pa = get(nosta_to_idx, a, a)
            pb = get(nosta_to_idx, b, b)
            excl_auto && pa == pb && continue
            p = (pa, pb)
            if !(p in seen_bl)
                push!(bl_pairs, p)
                push!(seen_bl, p)
            end
        end
        sort!(bl_pairs)
        bl_lookup = Dict(p => i for (i, p) in enumerate(bl_pairs))
        nti = length(unique_times)
        nbl = length(bl_pairs)

        baselines = BaselineIndex(bl_pairs, bl_pairs; antenna_names = antennas.name)

        row_of = zeros(Int, nti, nbl)
        uvw_dense = fill(Float32(NaN), nti, nbl, 3)
        # `auto_row[ti, ant]` = UV_DATA row of the (ant,ant) autocorr record, kept
        # (when normalizing) so the decoder can form correlation coefficients even
        # though the autocorr baselines are absent from the output.
        nant_tot = length(antennas.name)
        auto_row = normalize_autocorr ? zeros(Int, nti, nant_tot) : zeros(Int, 0, 0)
        # Records for dropped (autocorr) baselines are skipped, so `record_order`
        # grows only over the kept cross-correlation records (in on-disk order).
        record_order = Tuple{Int, Int}[]
        for gi in rng
            ti = time_lookup[t_hours[gi]]
            code = bl_codes[gi]
            pa = get(nosta_to_idx, code ÷ 256, code ÷ 256)
            pb = get(nosta_to_idx, code % 256, code % 256)
            if pa == pb
                normalize_autocorr && 1 <= pa <= nant_tot && (auto_row[ti, pa] = gi)
                continue                              # autocorr: not a science baseline
            end
            bi = get(bl_lookup, (pa, pb), 0)
            bi == 0 && continue
            row_of[ti, bi] = gi
            push!(record_order, (ti, bi))
            uvw_dense[ti, bi, 1] = uu[gi]
            uvw_dense[ti, bi, 2] = vv[gi]
            uvw_dense[ti, bi, 3] = ww[gi]
        end
        do_normalize = normalize_autocorr && any(!iszero, auto_row)

        uvw_part = DimArray(
            uvw_dense,
            (Ti(unique_times), Baseline(baselines.labels), UVW(["U", "V", "W"])),
        )

        # Scan time span (hours), used to pre-filter FLAG entries per leaf.
        scan_t_lo = isempty(unique_times) ? -Inf : first(unique_times)
        scan_t_hi = isempty(unique_times) ? Inf : last(unique_times)
        no_flags = FlagEntry[]

        for band in band_sel
            fsetup = freq_setups[band]
            chf = channel_freqs(fsetup)
            vis_dims = (
                Frequency(chf), Ti(unique_times),
                Baseline(baselines.labels), Pol(msv4_labels),
            )

            # Radiometer weight factor for this band: 2·Δν·η² / weight_norm (per-row
            # τ = INTTIM is applied at decode). Disabled (0) for :validity mode, or
            # when the band channel width / INTTIM column is unavailable — then the
            # raw correlator validity weights pass through unchanged.
            #
            # `weight_norm` divides out the nominal fully-valid value of the on-disk
            # WEIGHT column so that a fully-valid cell ends up at the correct
            # `2·Δν·τ·η²`. The radiometer formula assumes WEIGHT is a ≈1 validity
            # FRACTION, but some correlators (e.g. this DiFX BT164 set, WEIGHT ≈ 3.4)
            # write an unnormalized value; multiplying it in verbatim then overstates
            # the inverse-variance by that factor (confirmed by a successive-difference
            # scatter test). Pass `weight_norm = median(on-disk WEIGHT)` (1.0 keeps the
            # old behaviour, correct only when WEIGHT really is ≈1).
            cws_b = ch_widths(fsetup)
            dnu_b = isempty(cws_b) ? 0.0 : abs(Float64(first(cws_b)))
            wfactor_b = (weight_mode === :radiometer && dnu_b > 0 && !isempty(inttim)) ?
                Float32(2 * dnu_b * weight_efficiency^2 / weight_norm) : 0.0f0
            inttim_b = wfactor_b == 0.0f0 ? Float32[] : inttim

            # Flags touching this leaf (source, band, time-span). The vis layer
            # is never flagged (its values stay as read); flagging is carried by
            # the weights (→0) and flag (→true) layers.
            leaf_flags = _filter_flags_for_leaf(
                flag_entries, sid, band, scan_t_lo, scan_t_hi,
            )

            vis_chunk = _idi_chunk(
                eltype_vis, Val(:vis), data, flux_field, weight_col,
                band, no_stkd, no_chan, no_band, perm, flux_scale, row_of,
                no_flags, bl_pairs, unique_times, wfactor_b, inttim_b,
                auto_row, do_normalize, feed_pairs, auto_stokes,
            )
            w_chunk = _idi_chunk(
                Float32, Val(:weights), data, flux_field, weight_col,
                band, no_stkd, no_chan, no_band, perm, flux_scale, row_of,
                leaf_flags, bl_pairs, unique_times, wfactor_b, inttim_b,
                auto_row, do_normalize, feed_pairs, auto_stokes,
            )
            flag_chunk = _idi_chunk(
                Bool, Val(:flag), data, flux_field, weight_col,
                band, no_stkd, no_chan, no_band, perm, flux_scale, row_of,
                leaf_flags, bl_pairs, unique_times, wfactor_b, inttim_b,
                auto_row, false, feed_pairs, auto_stokes,
            )

            vis_part = DimArray(vis_chunk, vis_dims)
            w_part = DimArray(w_chunk, vis_dims)
            flag_part = DimArray(flag_chunk, vis_dims)

            info = UVData.PartitionInfo(;
                source_name = si.name,
                source_key = source_key,
                scan_name = string(scan_i),
                ra = si.ra, dec = si.dec,
                antennas = antennas,
                baselines = baselines,
                record_order = record_order,
                freq_setup = fsetup,
                spw_name = "band_$(band)",
                ddi = band - 1,
                basename = base_name,
            )
            leaf = UVData._build_leaf(
                vis_part, w_part, uvw_part, flag_part;
                partition_info = info,
            )
            key = UVData.partition_key(info)
            haskey(branches, key) && error("load_fitsidi: duplicate partition key $(key)")
            branches[key] = leaf
        end
    end

    uvset = UVSet(; metadata = UVMetadata(array_obs), branches = branches)
    register_source_path!(uvset, path)
    lazy || return register_source_path!(UVData.materialize(uvset), path)
    return uvset
end
