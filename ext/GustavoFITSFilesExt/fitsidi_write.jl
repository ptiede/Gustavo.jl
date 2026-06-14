# FITS-IDI writer (AIPS Memo 114) — Phase 2.
#
# Inverse of `fitsidi_read.jl`. Emits a stub PRIMARY HDU plus five binary
# tables: ARRAY_GEOMETRY, ANTENNA, FREQUENCY, SOURCE, and a time-ordered
# UV_DATA table. The on-disk conventions mirror the reader exactly so that
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
#       STK_1 = -1, CDELT2 = -1). The MSv4-ordered leaf Pol axis is mapped back
#       to disk order by inverting the reader's `perm`.
#   * BASELINE = 256*NOSTA[a] + NOSTA[b] (1-based NOSTA values).
#   * Time: DATE = jd0 (constant JD of RDATE midnight), TIME = t_hours/24
#       (fraction of day), so the reader's `24*((DATE+TIME) - jd0)` recovers Ti.
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
        STAXOF = Float32[Float32(offset_mount(m)) for m in mounts],
        DIAMETER = diameter,
    )
    arr_xyz = array_xyz(antennas)
    cards = Card[
        Card("EXTNAME", "ARRAY_GEOMETRY"),
        Card("EXTVER", Int32(1)),
        Card("ARRAYX", Float64(arr_xyz[1])),
        Card("ARRAYY", Float64(arr_xyz[2])),
        Card("ARRAYZ", Float64(arr_xyz[3])),
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

function UVData.write_fitsidi(output_path, uvset::UVSet)
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

    # Pol axis (MSv4) shared across leaves → on-disk STOKES order (AIPS
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

    # jd0 = JD of RDATE midnight. DATE col is jd0 (constant), TIME = t_hours/24.
    jd0 = _rdate_jd_or_zero(array_obs.rdate)
    if jd0 == 0.0
        @warn(
            "write_fitsidi: `array_obs.rdate` is empty/unparseable; DATE/TIME " *
                "will be emitted relative to JD 0. The Gustavo reader still " *
                "round-trips (jd0 cancels), but external tools will reject it.",
        )
    end

    # Group leaves by scan (the reader segments scans by time gap / SOURCE_ID;
    # one UV_DATA row per (time, baseline) carries every band's FLUX). Bands of
    # the same scan share Ti/Baseline. Key on scan_name.
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

    flux_rows = Vector{Vector{Float32}}()
    weight_rows = Vector{Vector{Float32}}()
    date_col = Float64[]
    time_col = Float64[]
    bl_col = Int32[]
    source_col = Int32[]
    freqid_col = Int32[]
    inttim_col = Float32[]
    uu_col = Float32[]
    vv_col = Float32[]
    ww_col = Float32[]
    sort_keys = Tuple{Float64, Int32}[]   # (time_days, baseline) for SORT='T*'

    for sn in scan_order
        leaves = scan_leaves[sn]
        # Map each leaf to its band index; all leaves in a scan share Ti/Baseline.
        ref_leaf = first(leaves)
        ref_meta = DimensionalData.metadata(ref_leaf)
        bls = ref_meta.baselines
        ti_vals = collect(UVData.obs_time(ref_leaf))
        nti = length(ti_vals)
        nbl = length(bls.pairs)

        # Per-band materialized layers, indexed by band number. Flagging is
        # carried by the WEIGHT sign on disk, so only vis/weights are needed.
        band_vis = Dict{Int, Any}()
        band_w = Dict{Int, Any}()
        for leaf in leaves
            fs = DimensionalData.metadata(leaf).freq_setup
            b = band_index[fs]
            ml = UVData.materialize_leaf(leaf)
            band_vis[b] = parent(ml[:vis])      # (Frequency, Ti, Baseline, Pol)
            band_w[b] = parent(ml[:weights])
        end
        # uvw from the reference leaf (shared across bands).
        uvw_dense = parent(UVData.materialize_leaf(ref_leaf)[:uvw])  # (Ti, Baseline, UVW)

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

            flux = Vector{Float32}(undef, flux_len)
            wts = Vector{Float32}(undef, weight_len)
            for b in 1:no_band
                vis_b = band_vis[b]
                w_b = band_w[b]
                base_off = (b - 1) * nperband
                for s_disk in 1:no_stkd
                    p = disk_to_msv4[s_disk]
                    # WEIGHT: collapse per-channel weights to a single value per
                    # (stokes, band). Use the first finite weight; if the cell is
                    # flagged across all channels emit a non-positive weight so
                    # the reader flags it.
                    wval = 0.0f0
                    found_w = false
                    for c in 1:no_chan
                        wc = Float32(w_b[c, ti, bi, p])
                        if isfinite(wc) && wc > 0
                            wval = wc
                            found_w = true
                            break
                        end
                    end
                    if !found_w
                        # All channels flagged/zero — preserve flagged state.
                        wval = Float32(w_b[1, ti, bi, p])
                        (isfinite(wval) && wval <= 0) || (wval = -1.0f0)
                    end
                    wts[_idi_weight_index(s_disk, b, no_stkd)] = wval
                    for c in 1:no_chan
                        v = vis_b[c, ti, bi, p]
                        off = base_off + (c - 1) * (2 * no_stkd) + (s_disk - 1) * 2
                        flux[off + 1] = Float32(real(v))
                        flux[off + 2] = Float32(imag(v))
                    end
                end
            end

            t_hours = ti_vals[ti]
            push!(flux_rows, flux)
            push!(weight_rows, wts)
            push!(date_col, jd0)
            push!(time_col, t_hours / 24.0)
            push!(bl_col, bl_codes[bi])
            push!(source_col, Int32(1))
            push!(freqid_col, Int32(1))
            # INTTIM: use the median spacing of the time axis if available.
            push!(inttim_col, Float32(_idi_inttim(ti_vals)))
            push!(uu_col, Float32(uvw_dense[ti, bi, 1]))
            push!(vv_col, Float32(uvw_dense[ti, bi, 2]))
            push!(ww_col, Float32(uvw_dense[ti, bi, 3]))
            push!(sort_keys, (t_hours / 24.0, bl_codes[bi]))
        end
    end

    isempty(flux_rows) && error("write_fitsidi: UVSet has no records to write")

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
    write(output_path, out_hdus)
    return output_path
end

# Representative integration time (seconds) from the leaf's Ti axis (hours):
# the smallest positive spacing, falling back to 1 when a single integration.
function _idi_inttim(ti_vals::AbstractVector)
    length(ti_vals) < 2 && return 1.0
    diffs = diff(sort(collect(ti_vals)))
    pos = filter(>(0), diffs)
    isempty(pos) && return 1.0
    return minimum(pos) * 3600.0
end
