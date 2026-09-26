# FITS-IDI writer (AIPS Memo 114) — Phase 2.
#
# Inverse of `fitsidi_read.jl`. Emits a stub PRIMARY HDU plus five binary
# tables: ARRAY_GEOMETRY, ANTENNA, FREQUENCY, SOURCE, and a time-ordered
# UV_DATA table, plus a FLAG table when any sample is flagged. The on-disk conventions mirror the reader exactly so that
# `load_fitsidi(write_fitsidi(path, uvset))` reproduces `uvset` leaf-for-leaf.
#
# Layout facts emitted (read back by `fitsidi_read.jl`):
#   * FLUX matrix element order (MAXIS1 fastest):
#       COMPLEX(2: re,im) × STOKES(NO_STKD) × FREQ(NO_CHAN) × BAND(NO_BAND)
#       × RA(1) × DEC(1). One band = contiguous 2*NO_STKD*NO_CHAN block at
#       element offset (band-1)*nperband. Float32 ('E').
#   * WEIGHT column = NO_STKD*NO_BAND floats, stokes-fastest then band:
#       linear index (band-1)*NO_STKD + stokes. One weight per (stokes, band),
#       shared across channels. Negative/zero = flagged.
#   * STOKES axis on disk is AIPS order RR,LL,RL,LR (STK_1 + (k-1)*CDELT2 with
#       STK_1 = -1, CDELT2 = -1). The MSv4-ordered leaf Polarization axis is mapped back
#       to disk order by inverting the reader's `perm`.
#   * BASELINE = 256*NOSTA[a] + NOSTA[b] (1-based NOSTA values).
#   * Time: DATE = jd0 (constant JD of RDATE midnight), TIME = the record's
#       offset from it in days, so DATE + TIME is the record's Julian Day.
#   * UV_DATA rows are time-ordered (SORT='T*'): sorted by (time, baseline).

# ── Inverse helpers ──────────────────────────────────────────────────────────

# MSv4 POLBASIS → AIPS POLTYA/POLTYB letter.
_idi_poltype_letter(::RPol) = "R"
_idi_poltype_letter(::LPol) = "L"
_idi_poltype_letter(::XPol) = "X"
_idi_poltype_letter(::YPol) = "Y"

# Per-(stokes, band) WEIGHT linear index, stokes-fastest then band (the
# reader's convention). `s` and `band` are 1-based.
_idi_weight_index(s::Integer, band::Integer, no_stkd::Integer) =
    (band - 1) * no_stkd + s

# ── Metadata table builders (inverse of the reader's `_build_idi_*`) ──────────

# ARRAY_GEOMETRY: ANNAME / STABXYZ / NOSTA / MNTSTA / STAXOF / DIAMETER plus
# the array-center / time-system / Earth-orientation header cards. Inverse of
# `_build_idi_antenna_table` + `_build_idi_array_obs`.
function _build_idi_array_geometry_hdu(
        antennas::AntennaTable, array_obs::ObsArrayMetadata, ref_freq::Float64, no_band::Integer,
    )
    nant = length(antennas)
    names = collect(antennas.name)
    xyz = collect(antennas.station_xyz)
    mounts = collect(antennas.mount)
    ext = extras(antennas)
    nosta = haskey(ext, :NOSTA) ? Int32.(collect(ext.NOSTA)) : Int32.(1:nant)
    diameter = haskey(ext, :DIAMETER) ? Float32.(collect(ext.DIAMETER)) : fill(0.0f0, nant)

    data = (
        ANNAME = rpad.(names, 8),
        STABXYZ = [Float64.(collect(xyz[i])) for i in 1:nant],
        NOSTA = nosta,
        MNTSTA = [mount_to_mntsta(m) for m in mounts],
        STAXOF = [Float32.(collect(XRadio.axis_offset(m))) for m in mounts],
        DIAMETER = diameter,
    )
    cards = Card[
        Card("EXTNAME", "ARRAY_GEOMETRY"),
        Card("EXTVER", Int32(1)),
        # `STABXYZ` is geocentric, so the array center is the geocenter.
        Card("ARRAYX", 0.0),
        Card("ARRAYY", 0.0),
        Card("ARRAYZ", 0.0),
        Card("ARRNAM", array_name(antennas)),
        Card("FRAME", array_obs.frame),
        Card("NUMORB", Int32(0)),
        Card("NO_BAND", Int32(no_band)),
        Card("FREQ", Float64(ref_freq)),
        Card("RDATE", array_obs.rdate),
        Card("GSTIA0", Float64(array_obs.gst_iat0)),
        Card("DEGPDY", Float64(array_obs.earth_rot_rate)),
        Card("UT1UTC", Float64(array_obs.ut1utc)),
        Card("IATUTC", Float64(array_obs.datutc)),
        Card("POLARX", Float64(array_obs.polarx)),
        Card("POLARY", Float64(array_obs.polary)),
    ]
    return HDU(Bintable, data, cards)
end

# ANTENNA: POLTYA/POLTYB (feed basis letters) + POLAA/POLAB (feed angles in
# degrees, one per band). Inverse of the POLTYA/POLAA reads in
# `_build_idi_antenna_table`.
function _build_idi_antenna_hdu(
        antennas::AntennaTable, ref_freq::Float64, no_band::Integer,
    )
    nant = length(antennas)
    nb = collect(antennas.nominal_basis)
    pa = collect(antennas.pol_angles)
    nosta = let ext = extras(antennas)
        haskey(ext, :NOSTA) ? Int32.(collect(ext.NOSTA)) : Int32.(1:nant)
    end

    # POLAA/POLAB are one angle per band; replicate the single stored feed
    # angle across all bands and convert radians → degrees (reader does
    # deg→rad). With a single band, emit a scalar column (Vector{Float32});
    # FITSFiles' bintable writer cannot serialize length-1 vector columns.
    polaa = if no_band == 1
        Float32[Float32(rad2deg(pa[i][1])) for i in 1:nant]
    else
        [fill(Float32(rad2deg(pa[i][1])), no_band) for i in 1:nant]
    end
    polab = if no_band == 1
        Float32[Float32(rad2deg(pa[i][2])) for i in 1:nant]
    else
        [fill(Float32(rad2deg(pa[i][2])), no_band) for i in 1:nant]
    end

    data = (
        ANNAME = rpad.(collect(antennas.name), 8),
        ANTENNA_NO = nosta,
        ARRAY = fill(Int32(1), nant),
        FREQID = fill(Int32(1), nant),
        NO_LEVELS = fill(Int32(2), nant),
        POLTYA = [_idi_poltype_letter(p[1]) for p in nb],
        POLAA = polaa,
        POLTYB = [_idi_poltype_letter(p[2]) for p in nb],
        POLAB = polab,
    )
    cards = Card[
        Card("EXTNAME", "ANTENNA"),
        Card("EXTVER", Int32(1)),
        Card("NO_BAND", Int32(no_band)),
        Card("NOPCAL", Int32(0)),
        Card("POLTYPE", "APPROX"),
    ]
    return HDU(Bintable, data, cards)
end

# FREQUENCY: one row (FREQID = 1) carrying per-band BANDFREQ / CH_WIDTH /
# TOTAL_BANDWIDTH / SIDEBAND. BANDFREQ[b] = channel_freqs(band b)[1] - REF_FREQ
# (inverse of the reader's `REF_FREQ + BANDFREQ[b] + (0:n-1)*CH_WIDTH[b]`).
function _build_idi_frequency_hdu(
        setups::AbstractVector{<:FrequencySetup}, ref_freq::Float64,
    )
    no_band = length(setups)
    bandfreq = Float64[Float64(first(channel_freqs(fs))) - ref_freq for fs in setups]
    ch_width = Float64[Float64(first(ch_widths(fs))) for fs in setups]
    total_bw = Float64[Float64(first(total_bandwidths(fs))) for fs in setups]
    sideband = Int32[round(Int32, first(sidebands(fs))) for fs in setups]

    # One FREQID row, each column a length-no_band vector. With a single band,
    # emit scalar columns (FITSFiles can't serialize length-1 vector columns;
    # the reader's `_idi_row` handles both the vector and scalar shapes).
    data = if no_band == 1
        (
            FREQID = Int32[1],
            BANDFREQ = Float64[bandfreq[1]],
            CH_WIDTH = Float64[ch_width[1]],
            TOTAL_BANDWIDTH = Float64[total_bw[1]],
            SIDEBAND = Int32[sideband[1]],
        )
    else
        (
            FREQID = Int32[1],
            BANDFREQ = [bandfreq],
            CH_WIDTH = [ch_width],
            TOTAL_BANDWIDTH = [total_bw],
            SIDEBAND = [sideband],
        )
    end
    cards = Card[Card("EXTNAME", "FREQUENCY"), Card("EXTVER", Int32(1)), Card("NO_BAND", Int32(no_band))]
    return HDU(Bintable, data, cards)
end

# SOURCE: SOURCE_ID / SOURCE / RAEPO / DECEPO (degrees). Inverse of
# `_build_idi_sources` (which does deg→rad).
function _build_idi_source_hdu(source_name::AbstractString, ra::Real, dec::Real)
    data = (
        SOURCE_ID = Int32[1],
        SOURCE = [rpad(String(source_name), 16)],
        QUAL = Int32[0],
        CALCODE = [rpad("", 4)],
        RAEPO = Float64[rad2deg(ra)],
        DECEPO = Float64[rad2deg(dec)],
        EQUINOX = [rpad("J2000", 8)],
    )
    cards = Card[Card("EXTNAME", "SOURCE"), Card("EXTVER", Int32(1))]
    return HDU(Bintable, data, cards)
end

# ── Public entry point ───────────────────────────────────────────────────────

# ── FLAG table (inverse of the reader's `_build_idi_flags`) ──────────────────
#
# The dense `:flags` layer is re-encoded as FLAG rows. For one (band, baseline,
# on-disk stokes) the flagged channels of an integration form a set of runs, and
# consecutive integrations carrying the identical set share a row — so a
# whole-scan flag on one baseline costs one row rather than `nti·nchan`.

# One FLAG row before serialization. `t0`/`t1` are DAYS relative to RDATE, the
# units TIMERANG is written in; `ants` are NOSTA numbers and `stokes` is an
# on-disk stokes slot.
struct IDIFlagRow
    ants::Tuple{Int, Int}
    t0::Float64
    t1::Float64
    band::Int
    chan_lo::Int
    chan_hi::Int
    stokes::Int
end

# Inclusive 1-based (lo, hi) channel runs where `col` is set.
function _flag_runs(col::AbstractVector{Bool})
    runs = Tuple{Int, Int}[]
    lo = 0
    for c in eachindex(col)
        if col[c]
            lo == 0 && (lo = c)
        elseif lo != 0
            push!(runs, (lo, c - 1))
            lo = 0
        end
    end
    lo != 0 && push!(runs, (lo, lastindex(col)))
    return runs
end

# Append this scan's FLAG rows. `band_flags[b]` is band `b`'s dense
# (Frequency, Ti, BaselineID, Polarization) flag layer; `disk_to_msv4[s]` is the MSv4 pol
# stored at on-disk stokes slot `s`.
#
# TIMERANG is padded by half an integration each side: a flag covers the
# integration rather than its centre, and the column is single precision, so a
# range written to the sample times themselves could round inside them. The pad
# stops half-way to the neighbouring integration, which `_flag_timerange` then
# checks survives the narrowing to `Float32`.
function _append_idi_flag_rows!(
        rows::Vector{IDIFlagRow}, band_flags, bls, nosta, ti_vals, jd0_unix, disk_to_msv4,
    )
    nti = length(ti_vals)
    nti == 0 && return rows
    issorted(ti_vals) || error(
        "write_fitsidi: the Ti axis must ascend to write a FLAG table; got $(ti_vals).",
    )
    pad = _idi_inttim(ti_vals) / 2
    for (band, F) in pairs(band_flags)
        any(F) || continue
        nchan = size(F, 1)
        col = Vector{Bool}(undef, nchan)
        for bi in axes(F, 3), s_disk in eachindex(disk_to_msv4)
            p = disk_to_msv4[s_disk]
            prev = Tuple{Int, Int}[]
            ti_start = 1
            for ti in 1:nti
                for c in 1:nchan
                    col[c] = F[c, ti, bi, p]
                end
                runs = _flag_runs(col)
                runs == prev && continue
                _push_idi_flag_rows!(
                    rows, prev, bls, nosta, bi, band, s_disk, ti_vals, ti_start, ti - 1, pad, jd0_unix,
                )
                prev = runs
                ti_start = ti
            end
            _push_idi_flag_rows!(
                rows, prev, bls, nosta, bi, band, s_disk, ti_vals, ti_start, nti, pad, jd0_unix,
            )
        end
    end
    return rows
end

# TIMERANG (days relative to RDATE) covering integrations `ti_lo:ti_hi`, checked
# against the single precision it is stored in: the stored interval must still
# contain every integration it covers and still exclude the neighbours.
function _flag_timerange(ti_vals, ti_lo::Int, ti_hi::Int, pad::Float64, jd0_unix::Float64)
    t0 = Float32((ti_vals[ti_lo] - pad - jd0_unix) / 86400.0)
    t1 = Float32((ti_vals[ti_hi] + pad - jd0_unix) / 86400.0)
    s0 = jd0_unix + Float64(t0) * 86400.0
    s1 = jd0_unix + Float64(t1) * 86400.0
    prev = ti_lo > 1 ? ti_vals[ti_lo - 1] : -Inf
    next = ti_hi < length(ti_vals) ? ti_vals[ti_hi + 1] : Inf
    (s0 <= ti_vals[ti_lo] && s1 >= ti_vals[ti_hi] && s0 > prev && s1 < next) || error(
        "write_fitsidi: cannot express a flag over integrations $(ti_lo):$(ti_hi) in " *
            "TIMERANG's single precision — the stored range $(s0) .. $(s1) s does not " *
            "isolate $(ti_vals[ti_lo]) .. $(ti_vals[ti_hi]) s from its neighbours.",
    )
    return Float64(t0), Float64(t1)
end

function _push_idi_flag_rows!(
        rows, runs, bls, nosta, bi::Int, band::Int, s_disk::Int,
        ti_vals, ti_lo::Int, ti_hi::Int, pad::Float64, jd0_unix::Float64,
    )
    (isempty(runs) || ti_lo > ti_hi) && return rows
    a, b = bls.pairs[bi]
    t0, t1 = _flag_timerange(ti_vals, ti_lo, ti_hi, pad, jd0_unix)
    for (lo, hi) in runs
        push!(rows, IDIFlagRow((nosta[a], nosta[b]), t0, t1, band, lo, hi, s_disk))
    end
    return rows
end

# Serialize collected FLAG rows. BANDS and PFLAGS select one band / one stokes
# each; with a single band or stokes they become scalar columns, since FITSFiles'
# bintable writer cannot serialize length-1 vector columns.
function _build_idi_flag_hdu(
        rows::Vector{IDIFlagRow}, no_band::Integer, no_stkd::Integer, no_chan::Integer,
    )
    n = length(rows)
    bands = no_band == 1 ? fill(Int32(1), n) :
        [Int32[i == r.band ? 1 : 0 for i in 1:no_band] for r in rows]
    pflags = no_stkd == 1 ? fill(Int32(1), n) :
        [Int32[i == r.stokes ? 1 : 0 for i in 1:no_stkd] for r in rows]
    data = (
        SOURCE_ID = fill(Int32(0), n),
        ARRAY = fill(Int32(1), n),
        ANTS = [Int32[r.ants[1], r.ants[2]] for r in rows],
        FREQID = fill(Int32(0), n),
        TIMERANG = [Float32[r.t0, r.t1] for r in rows],
        BANDS = bands,
        CHANS = [Int32[r.chan_lo, r.chan_hi] for r in rows],
        PFLAGS = pflags,
        REASON = [rpad("GUSTAVO", 24) for _ in rows],
        SEVERITY = fill(Int32(-1), n),
    )
    cards = Card[
        Card("EXTNAME", "FLAG"),
        Card("EXTVER", Int32(1)),
        Card("NO_STKD", Int32(no_stkd)),
        Card("NO_BAND", Int32(no_band)),
        Card("NO_CHAN", Int32(no_chan)),
    ]
    return HDU(Bintable, data, cards)
end

function UVData.write_fitsidi(output_path, uvset::UVSet)
    _assert_not_writing_to_source(output_path, uvset)
    src_list = sources(uvset)
    length(src_list) == 1 || error(
        "write_fitsidi: UVData is single-source; got sources=$(src_list). " *
            "Use select_source(uvset, name) before writing.",
    )

    branches_dict = DimensionalData.branches(uvset)
    isempty(branches_dict) && error("write_fitsidi: UVSet has no partitions")

    root = DimensionalData.metadata(uvset)
    array_obs = root.array_obs
    antennas = UVData.union_antennas(uvset)
    nant = length(antennas)
    nosta = let ext = extras(antennas)
        haskey(ext, :NOSTA) ? Int.(collect(ext.NOSTA)) : collect(1:nant)
    end

    # Frequency setups → bands, ordered by the `band` extra (or ddi). The
    # reader builds bands 1..no_band; we must serialize them in that order so
    # BANDFREQ[b] lines up with the per-band FLUX slice.
    setups = UVData.union_frequency_axis(uvset)
    band_of_setup = Dict{FrequencySetup, Int}()
    for fs in setups
        b = Int(get(fs.extras, :band, 0))
        band_of_setup[fs] = b
    end
    if all(b -> b > 0, values(band_of_setup))
        sort!(setups; by = fs -> band_of_setup[fs])
    end
    no_band = length(setups)
    band_index = Dict{FrequencySetup, Int}(fs => i for (i, fs) in enumerate(setups))

    no_chan = length(channel_freqs(first(setups)))
    for fs in setups
        length(channel_freqs(fs)) == no_chan ||
            error("write_fitsidi: ragged channel counts across bands not supported")
    end
    ref_freq_v = Float64(ref_freq(first(setups)))

    # Polarization axis (MSv4) shared across leaves → on-disk STOKES order (AIPS
    # RR,LL,RL,LR). The reader computes `perm` s.t. aips_labels[perm] ==
    # msv4_labels; serializing means writing MSv4 pol `msv4_to_disk[s_disk]`
    # into on-disk stokes slot `s_disk`. `disk_to_msv4[s] = perm[s]`.
    msv4_labels = collect(UVData.union_pol_products(uvset))
    no_stkd = length(msv4_labels)
    stk_1 = -1
    cdelt2 = -1
    aips_codes = Int[stk_1 + (k - 1) * cdelt2 for k in 1:no_stkd]
    aips_labels = aips_code_to_generic.(aips_codes)
    # disk_to_msv4[s] = MSv4 pol index stored at on-disk stokes slot `s`.
    disk_to_msv4 = [findfirst(==(lab), msv4_labels) for lab in aips_labels]
    any(isnothing, disk_to_msv4) && error(
        "write_fitsidi: cannot map MSv4 pols $msv4_labels to AIPS stokes labels $aips_labels",
    )
    disk_to_msv4 = Vector{Int}(disk_to_msv4)

    # jd0 = JD of RDATE midnight. DATE col is jd0 (constant), TIME the offset.
    jd0 = _rdate_jd_or_zero(array_obs.rdate)
    if jd0 == 0.0
        @warn(
            "write_fitsidi: `array_obs.rdate` is empty/unparseable; DATE/TIME " *
                "will be emitted relative to JD 0. The Gustavo reader still " *
                "round-trips (jd0 cancels), but external tools will reject it.",
        )
    end
    jd0_unix = UVData.jd_to_unix(jd0)

    # Group leaves by scan (the reader segments scans by time gap / SOURCE_ID;
    # one UV_DATA row per (time, baseline) carries every band's FLUX). Bands of
    # the same scan share Ti/BaselineID. Key on scan_name.
    scan_leaves = Dict{String, Vector{Any}}()
    scan_order = String[]
    for (_, leaf) in branches_dict
        sn = DimensionalData.metadata(leaf).scan_name
        if !haskey(scan_leaves, sn)
            scan_leaves[sn] = Any[]
            push!(scan_order, sn)
        end
        push!(scan_leaves[sn], leaf)
    end
    # Order scans by their time window so emitted rows ascend in time overall.
    sort!(scan_order; by = sn -> UVData.scan_window(first(scan_leaves[sn]))[1])

    # ── Assemble UV_DATA rows ────────────────────────────────────────────────
    nperband = 2 * no_stkd * no_chan
    flux_len = nperband * no_band
    weight_len = no_stkd * no_band

    # A FITS-IDI column carries its own TFORM, so each one is written at the
    # width of the layer it holds.
    leaf_vals = values(branches_dict)
    Tflux = _fits_float_type(
        "write_fitsidi",
        mapreduce(l -> real(eltype(parent(l[:vis]))), promote_type, leaf_vals),
    )
    Tweight = _fits_float_type(
        "write_fitsidi",
        mapreduce(l -> eltype(parent(l[:weights])), promote_type, leaf_vals),
    )
    Tuvw = _fits_float_type(
        "write_fitsidi",
        mapreduce(l -> eltype(parent(l[:uvw])), promote_type, leaf_vals),
    )

    flux_rows = Vector{Vector{Tflux}}()
    weight_rows = Vector{Vector{Tweight}}()
    date_col = Float64[]
    time_col = Float64[]
    bl_col = Int32[]
    source_col = Int32[]
    freqid_col = Int32[]
    inttim_col = Float32[]
    uu_col = Tuvw[]
    vv_col = Tuvw[]
    ww_col = Tuvw[]
    sort_keys = Tuple{Float64, Int32}[]   # (time_days, baseline) for SORT='T*'
    flag_rows = IDIFlagRow[]

    for sn in scan_order
        leaves = scan_leaves[sn]
        # Map each leaf to its band index; all leaves in a scan share Ti/BaselineID.
        ref_leaf = first(leaves)
        ref_meta = DimensionalData.metadata(ref_leaf)
        bls = ref_meta.baselines
        ti_vals = collect(UVData.obs_time(ref_leaf))
        nti = length(ti_vals)
        nbl = length(bls.pairs)

        # Per-band materialized layers, indexed by band number. The flag layer
        # is serialized separately, as FLAG rows.
        band_vis = Dict{Int, Any}()
        band_w = Dict{Int, Any}()
        band_f = Dict{Int, Any}()
        for leaf in leaves
            fs = DimensionalData.metadata(leaf).freq_setup
            b = band_index[fs]
            ml = UVData.materialize_leaf(leaf)
            band_vis[b] = parent(ml[:vis])      # (Frequency, Ti, BaselineID, Polarization)
            band_w[b] = parent(ml[:weights])
            band_f[b] = parent(ml[:flags])
        end
        _append_idi_flag_rows!(flag_rows, band_f, bls, nosta, ti_vals, jd0_unix, disk_to_msv4)
        # uvw from the reference leaf (shared across bands).
        uvw_dense = parent(UVData.materialize_leaf(ref_leaf)[:uvw])  # (Ti, BaselineID, UVW)

        # Baseline AIPS codes via NOSTA.
        bl_codes = Int32[Int32(256 * nosta[a] + nosta[b]) for (a, b) in bls.pairs]

        for ti in 1:nti, bi in 1:nbl
            # Skip cells that are entirely missing across every band (no record).
            # Presence is signalled by a non-NaN vis in band 1.
            present = false
            for b in 1:no_band
                v = band_vis[b][1, ti, bi, 1]
                if !(isnan(real(v)) && isnan(imag(v)))
                    present = true
                    break
                end
            end
            present || continue

            flux = Vector{Tflux}(undef, flux_len)
            wts = Vector{Tweight}(undef, weight_len)
            for b in 1:no_band
                vis_b = band_vis[b]
                w_b = band_w[b]
                base_off = (b - 1) * nperband
                for s_disk in 1:no_stkd
                    p = disk_to_msv4[s_disk]
                    # WEIGHT: collapse per-channel weights to a single value per
                    # (stokes, band) — the column has no channel axis. Use the
                    # first finite positive weight; a cell with none keeps its
                    # non-positive value, which is what the reader reads back.
                    wval = zero(Tweight)
                    found_w = false
                    for c in 1:no_chan
                        wc = Tweight(w_b[c, ti, bi, p])
                        if isfinite(wc) && wc > 0
                            wval = wc
                            found_w = true
                            break
                        end
                    end
                    if !found_w
                        # No usable weight on any channel — preserve the
                        # non-positive value the reader expects.
                        wval = Tweight(w_b[1, ti, bi, p])
                        (isfinite(wval) && wval <= 0) || (wval = -one(Tweight))
                    end
                    wts[_idi_weight_index(s_disk, b, no_stkd)] = wval
                    for c in 1:no_chan
                        v = vis_b[c, ti, bi, p]
                        off = base_off + (c - 1) * (2 * no_stkd) + (s_disk - 1) * 2
                        flux[off + 1] = Tflux(real(v))
                        # Conjugate on the way out: FITS-IDI stores
                        # V = ⟨E_a1 · conj(E_a2)⟩, the conjugate of Gustavo's
                        # internal MSv4/casacore sense (AIPS Memo 114r §2.1).
                        flux[off + 2] = Tflux(-imag(v))
                    end
                end
            end

            t_day = (ti_vals[ti] - jd0_unix) / 86400.0
            push!(flux_rows, flux)
            push!(weight_rows, wts)
            push!(date_col, jd0)
            push!(time_col, t_day)
            push!(bl_col, bl_codes[bi])
            push!(source_col, Int32(1))
            push!(freqid_col, Int32(1))
            # INTTIM: use the median spacing of the time axis if available.
            push!(inttim_col, Float32(_idi_inttim(ti_vals)))
            push!(uu_col, Tuvw(uvw_dense[ti, bi, 1]))
            push!(vv_col, Tuvw(uvw_dense[ti, bi, 2]))
            push!(ww_col, Tuvw(uvw_dense[ti, bi, 3]))
            push!(sort_keys, (t_day, bl_codes[bi]))
        end
    end

    isempty(flux_rows) && error("write_fitsidi: UVSet has no records to write")

    # TIMERANG is expressed in days relative to RDATE, so a FLAG table cannot be
    # written without one.
    (!isempty(flag_rows) && jd0 == 0.0) && error(
        "write_fitsidi: the set carries flags but `array_obs.rdate` is empty or " *
            "unparseable — the FLAG table's TIMERANG is measured from RDATE.",
    )

    # SORT='T*': order rows by (time, baseline).
    order = sortperm(sort_keys; by = identity)
    uv_data = (
        DATE = date_col[order],
        TIME = time_col[order],
        BASELINE = bl_col[order],
        SOURCE = source_col[order],
        FREQID = freqid_col[order],
        INTTIM = inttim_col[order],
        var"UU---SIN" = uu_col[order],
        var"VV---SIN" = vv_col[order],
        var"WW---SIN" = ww_col[order],
        FLUX = flux_rows[order],
        WEIGHT = weight_rows[order],
    )

    chan_bw = Float64(first(ch_widths(first(setups))))
    uv_cards = Card[
        Card("EXTNAME", "UV_DATA"),
        Card("EXTVER", Int32(1)),
        Card("NMATRIX", Int32(1)),
        Card("TMATX11", true),
        Card("MAXIS", Int32(6)),
        Card("MAXIS1", Int32(2)),
        Card("CTYPE1", "COMPLEX"),
        Card("CRVAL1", 1.0), Card("CDELT1", 1.0), Card("CRPIX1", 1.0),
        Card("MAXIS2", Int32(no_stkd)),
        Card("CTYPE2", "STOKES"),
        Card("CRVAL2", Float64(stk_1)), Card("CDELT2", Float64(cdelt2)), Card("CRPIX2", 1.0),
        Card("MAXIS3", Int32(no_chan)),
        Card("CTYPE3", "FREQ"),
        Card("CRVAL3", ref_freq_v), Card("CDELT3", chan_bw), Card("CRPIX3", 1.0),
        Card("MAXIS4", Int32(no_band)),
        Card("CTYPE4", "BAND"),
        Card("CRVAL4", 1.0), Card("CDELT4", 1.0), Card("CRPIX4", 1.0),
        Card("MAXIS5", Int32(1)),
        Card("CTYPE5", "RA"), Card("CRVAL5", rad2deg(DimensionalData.metadata(first(values(branches_dict))).ra)),
        Card("CDELT5", 1.0), Card("CRPIX5", 1.0),
        Card("MAXIS6", Int32(1)),
        Card("CTYPE6", "DEC"), Card("CRVAL6", rad2deg(DimensionalData.metadata(first(values(branches_dict))).dec)),
        Card("CDELT6", 1.0), Card("CRPIX6", 1.0),
        Card("TMATX11", true),
        Card("NO_STKD", Int32(no_stkd)),
        Card("STK_1", Int32(stk_1)),
        Card("NO_BAND", Int32(no_band)),
        Card("NO_CHAN", Int32(no_chan)),
        Card("REF_FREQ", ref_freq_v),
        Card("CHAN_BW", chan_bw),
        Card("NO_POL", Int32(no_stkd)),
        Card("RDATE", array_obs.rdate),
        Card("OBSCODE", string(get(array_obs.extras, :obscode, ""))),
        Card("SORT", "T*"),
    ]
    uv_hdu = HDU(Bintable, uv_data, uv_cards)

    # ── Primary stub + tables ────────────────────────────────────────────────
    primary_cards = Card[
        Card("SIMPLE", true),
        Card("BITPIX", Int32(8)),
        Card("NAXIS", Int32(0)),
        Card("EXTEND", true),
        Card("TELESCOP", array_obs.telescope),
        Card("INSTRUME", array_obs.instrume),
        Card("DATE-OBS", array_obs.date_obs),
        Card("EQUINOX", Float64(array_obs.equinox)),
        Card("BUNIT", array_obs.bunit),
        Card("CORRELAT", string(get(array_obs.extras, :correlat, ""))),
        Card("GROUPS", false),
    ]
    primary_hdu = HDU(missing, primary_cards)

    ag_hdu = _build_idi_array_geometry_hdu(antennas, array_obs, ref_freq_v, no_band)
    an_hdu = _build_idi_antenna_hdu(antennas, ref_freq_v, no_band)
    fq_hdu = _build_idi_frequency_hdu(setups, ref_freq_v)
    src_hdu = _build_idi_source_hdu(
        first(src_list),
        DimensionalData.metadata(first(values(branches_dict))).ra,
        DimensionalData.metadata(first(values(branches_dict))).dec,
    )

    out_hdus = HDU[primary_hdu, ag_hdu, an_hdu, fq_hdu, src_hdu, uv_hdu]
    isempty(flag_rows) ||
        push!(out_hdus, _build_idi_flag_hdu(flag_rows, no_band, no_stkd, no_chan))
    write(output_path, out_hdus)
    return output_path
end

# Representative integration time (seconds) from the leaf's Ti axis: the
# smallest positive spacing, falling back to 1 when a single integration.
function _idi_inttim(ti_vals::AbstractVector)
    length(ti_vals) < 2 && return 1.0
    diffs = diff(sort(collect(ti_vals)))
    pos = filter(>(0), diffs)
    isempty(pos) && return 1.0
    return minimum(pos)
end
