# A-priori amplitude calibration from a FITS-IDI file's GAIN_CURVE (DPFU +
# elevation gain polynomial) and SYSTEM_TEMPERATURE (Tsys) tables. Produces one
# `UVData.AntabCalibration` per band, reusing Gustavo's existing SEFD/elevation
# apply machinery (`apply_calibration(uvset, band_cals)`). FITS-IDI specifics
# (table layout, NOSTA→name mapping, time convention) live here; the calibration
# math stays format-neutral in `src/UVData/apriori.jl`.

using Dates: Date, DateTime, Millisecond, year
import Gustavo.UVData
using Gustavo.UVData: AntabCalibration, AntabStation, AntabGainCurve, AntabTsysSeries

# Locate the first HDU whose EXTNAME (trimmed) equals `name`.
function _idi_find_hdu(fid, name)
    for hdu in fid
        j = findfirst(c -> c.key == "EXTNAME", hdu.cards)
        j === nothing && continue
        strip(string(hdu.cards[j].value)) == name && return hdu
    end
    return nothing
end

# A Tsys value is usable unless it is non-finite, non-positive, the AIPS `999`
# placeholder, or an out-of-range outlier (off-source / failed measurement).
@inline _tsys_ok(v, tmax) = isfinite(v) && v > 0 && !(998.5 < v < 999.5) && v <= tmax

function UVData.load_fitsidi_apriori(path; tsys_max::Real = 1.0e4)
    fid = FITSFiles.fits(path)

    ag = _idi_find_hdu(fid, "ARRAY_GEOMETRY")
    gc = _idi_find_hdu(fid, "GAIN_CURVE")
    st = _idi_find_hdu(fid, "SYSTEM_TEMPERATURE")
    ag === nothing && error("load_fitsidi_apriori: no ARRAY_GEOMETRY HDU in $(path)")
    gc === nothing && error("load_fitsidi_apriori: no GAIN_CURVE HDU in $(path)")
    st === nothing && error("load_fitsidi_apriori: no SYSTEM_TEMPERATURE HDU in $(path)")

    # NOSTA → cleaned station name (must match what `load_fitsidi` stores).
    nosta = round.(Int, collect(getproperty(ag.data, :NOSTA)))
    anames = _idi_clean.(collect(getproperty(ag.data, :ANNAME)))
    name_of = Dict(nosta[i] => anames[i] for i in eachindex(nosta))

    # Reference date + provenance. RDATE lives on a table header (try GC, then
    # SYSTEM_TEMPERATURE, then ARRAY_GEOMETRY).
    rdate_str = ""
    for h in (gc, st, ag)
        v = card_value(h.cards, "RDATE")
        v === nothing || (rdate_str = strip(string(v)); !isempty(rdate_str) && break)
    end
    isempty(rdate_str) && error("load_fitsidi_apriori: no RDATE keyword found in $(path)")
    rdate0 = DateTime(Date(rdate_str))
    yr = year(rdate0)
    label = String(splitext(basename(String(path)))[1])

    no_band = Int(something(card_value(gc.cards, "NO_BAND"), 1))
    no_tabs = Int(something(card_value(gc.cards, "NO_TABS"), 1))

    # ── GAIN_CURVE: per antenna, per band → DPFU (R, L) + elevation gain poly ──
    gtbl = gc.data
    gc_nosta = round.(Int, collect(getproperty(gtbl, :ANTENNA_NO)))
    sens1 = getproperty(gtbl, :SENS_1)            # (ngc, no_band)  DPFU feed R
    sens2 = getproperty(gtbl, :SENS_2)            # (ngc, no_band)  DPFU feed L
    gain1 = getproperty(gtbl, :GAIN_1)            # (ngc, no_tabs*no_band) feed R
    nterm1 = getproperty(gtbl, :NTERM_1)          # (ngc, no_band)
    # gain[ant, band] curve (elevation poly; user-confirmed elevation convention,
    # so GAIN coefficients are used directly as g = c1 + c2·El + c3·El² + …).
    gain_of = Dict{Tuple{Int, Int}, AntabGainCurve}()
    for r in eachindex(gc_nosta)
        ant = gc_nosta[r]
        for b in 1:no_band
            nt = max(1, Int(nterm1[r, b]))
            off = (b - 1) * no_tabs
            poly = Float64[gain1[r, off + k] for k in 1:nt]
            dpfu = (Float64(sens1[r, b]), Float64(sens2[r, b]))
            gain_of[(ant, b)] = AntabGainCurve(dpfu, poly)
        end
    end

    # ── SYSTEM_TEMPERATURE: per antenna time-series of (R, L) Tsys per band ─────
    stbl = st.data
    st_nosta = round.(Int, collect(getproperty(stbl, :ANTENNA_NO)))
    st_time = Float64.(collect(getproperty(stbl, :TIME)))   # days since RDATE 0h
    tsys1 = getproperty(stbl, :TSYS_1)            # (nrow, no_band) feed R
    tsys2 = getproperty(stbl, :TSYS_2)            # (nrow, no_band) feed L
    # Group SYSTEM_TEMPERATURE row indices by antenna.
    rows_of = Dict{Int, Vector{Int}}()
    for r in eachindex(st_nosta)
        push!(get!(rows_of, st_nosta[r], Int[]), r)
    end

    # ── Assemble one AntabCalibration per band ─────────────────────────────────
    band_cals = Dict{Int, AntabCalibration}()
    for b in 1:no_band
        stations = Dict{String, AntabStation}()
        for (ant, rows) in rows_of
            haskey(name_of, ant) || continue
            haskey(gain_of, (ant, b)) || continue
            nm = name_of[ant]
            times = DateTime[rdate0 + Millisecond(round(Int, st_time[r] * 86_400_000)) for r in rows]
            # Aggregate (per-feed, broadcast to all channels) Tsys with
            # other-feed fallback for missing/bad values.
            vals = Matrix{Float64}(undef, length(rows), 2)
            for (i, r) in enumerate(rows)
                tr = Float64(tsys1[r, b]); tl = Float64(tsys2[r, b])
                okr = _tsys_ok(tr, tsys_max); okl = _tsys_ok(tl, tsys_max)
                vals[i, 1] = okr ? tr : (okl ? tl : NaN)   # feed R (falls back to L)
                vals[i, 2] = okl ? tl : (okr ? tr : NaN)   # feed L (falls back to R)
            end
            tseries = AntabTsysSeries(times, [(0, :R), (0, :L)], vals)
            stations[nm] = AntabStation(nm, gain_of[(ant, b)], tseries, 0)
        end
        band_cals[b] = AntabCalibration(String(path), label, yr, stations)
    end

    return band_cals
end
