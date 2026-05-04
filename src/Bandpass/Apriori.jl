using AstroLib: ct2lst
using Dates

# ECEF (m) → geodetic (lat_rad, lon_rad, h_m). WGS84.
function _ecef_to_geodetic(xyz::AbstractVector{<:Real})
    a = 6378137.0                              # WGS84 semi-major axis (m)
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 1 - (b / a)^2
    ep2 = (a / b)^2 - 1
    x, y, z = Float64(xyz[1]), Float64(xyz[2]), Float64(xyz[3])
    p = sqrt(x^2 + y^2)
    th = atan(z * a, p * b)
    lon = atan(y, x)
    lat = atan(z + ep2 * b * sin(th)^3, p - e2 * a * cos(th)^3)
    N = a / sqrt(1 - e2 * sin(lat)^2)
    h = p / cos(lat) - N
    return lat, lon, h
end

# Source elevation (radians) for a given antenna ECEF position, source
# (RA, Dec) in radians, and Julian Date (UTC).
function _source_elevation(ecef::AbstractVector{<:Real}, ra::Real, dec::Real, jd::Real)
    lat, lon, _ = _ecef_to_geodetic(ecef)
    lon_deg = rad2deg(lon)
    lst_hr = ct2lst(lon_deg, jd)              # local sidereal time, hours
    ha = mod(deg2rad(lst_hr * 15.0) - ra + π, 2π) - π
    sin_alt = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(ha)
    return asin(clamp(sin_alt, -1.0, 1.0))
end

# Convert a leaf's `obs_time` axis (fractional hours since RDATE 0h UTC)
# back to absolute UTC `DateTime`s using the array's `rdate` string.
function _leaf_obs_datetimes(leaf, root_meta)
    obs = obs_time(leaf)
    rdate_str = root_meta.array_obs.rdate
    isempty(rdate_str) && error(
        "apply_calibration: root metadata has empty `rdate`; cannot resolve " *
            "ANTAB DOY+UT timestamps to the leaf's time axis.",
    )
    base = DateTime(Date(rdate_str))           # 0h UTC on RDATE
    return [base + Millisecond(round(Int, t * 3_600_000)) for t in obs]
end

# Convert a leaf's obs_time axis to a vector of Julian Dates (UTC).
function _leaf_obs_jds(leaf, root_meta)
    obs = obs_time(leaf)
    rdate_str = root_meta.array_obs.rdate
    isempty(rdate_str) && error(
        "apply_calibration: root metadata has empty `rdate`; cannot compute " *
            "antenna elevations.",
    )
    base_jd = datetime2julian(DateTime(Date(rdate_str)))
    return [base_jd + t / 24.0 for t in obs]
end

"""
    AprioriFluxGains

Per-leaf precomputed apriori-calibration result. Returned by
[`apriori_flux_gains`](@ref) for inspection (e.g. plotting SEFD curves)
without applying the correction. The `gains` field carries the real,
positive amplitude gains used by [`apply_calibration`](@ref); they are
indexed as `gains[c, ti, a, p]` where `p ∈ {1, 2}` is the feed slot
(P/Q in MSv4 labels, R/L in EHT convention).
"""
struct AprioriFluxGains
    gains::Array{Float64, 4}                  # (nchan, nti, nant, 2)
    sefd::Array{Float64, 4}                   # raw SEFD per (chan, ti, ant, feed)
    elevation_deg::Matrix{Float64}            # (nti, nant)
    antennas::Vector{String}
    missing_stations::Vector{String}
end

# Compute per-leaf SEFD and gain arrays.
#
# Tsys is looked up by **scan window**: each leaf's `scan_window(leaf)`
# defines a UTC range, and we average antab rows inside it (per
# antenna, channel, pol). This avoids contamination from non-target
# scans (slews, calibrators) that the processed antab interleaves in
# its time series. Elevation, and therefore the gain curve `g_E`,
# remains evaluated per integration.
function _build_apriori_gains(
        leaf, info, root_meta, antab::AntabCalibration;
        on_missing_station::Symbol = :warn,
    )
    on_missing_station in (:warn, :error, :ignore) || error(
        "apply_calibration: on_missing_station must be :warn, :error, or :ignore"
    )
    ant_table = info.antennas
    ant_names = ant_table.name
    ant_xyz = ant_table.station_xyz
    nant = length(ant_names)
    chan_freqs = collect(info.freq_setup.channel_freqs)
    nchan = length(chan_freqs)
    nti = length(obs_time(leaf))

    jds = _leaf_obs_jds(leaf, root_meta)

    # Convert the leaf's scan window (hours since RDATE 0h UTC) to absolute
    # UTC DateTimes for matching against antab timestamps.
    rdate_str = root_meta.array_obs.rdate
    isempty(rdate_str) && error(
        "apply_calibration: root metadata has empty `rdate`; cannot align " *
            "ANTAB scan windows to the leaf's time axis.",
    )
    base_dt = DateTime(Date(rdate_str))
    lo_h, hi_h = UVData.scan_window(leaf)
    (isfinite(lo_h) && isfinite(hi_h)) || error(
        "apply_calibration: leaf has no finite scan window",
    )
    t_lo = base_dt + Millisecond(round(Int, lo_h * 3_600_000))
    t_hi = base_dt + Millisecond(round(Int, hi_h * 3_600_000))

    elevation_deg = Matrix{Float64}(undef, nti, nant)
    sefd = Array{Float64}(undef, nchan, nti, nant, 2)
    gains = Array{Float64}(undef, nchan, nti, nant, 2)

    pol_syms = (:R, :L)
    missing_stations = String[]

    # PartitionInfo stores ra/dec verbatim from the FITS primary HDU
    # (CRVAL of the RA/DEC axes), which is degrees per the FITS standard.
    ra_rad = deg2rad(Float64(info.ra))
    dec_rad = deg2rad(Float64(info.dec))

    for (a, name) in pairs(ant_names)
        if !haskey(antab, name)
            push!(missing_stations, String(name))
            for ti in 1:nti
                elevation_deg[ti, a] = NaN
                for c in 1:nchan, p in 1:2
                    sefd[c, ti, a, p] = NaN
                    gains[c, ti, a, p] = 1.0
                end
            end
            continue
        end
        st = antab[name]
        # Per-time elevation only depends on the antenna position, not channel/pol.
        for ti in 1:nti
            el_rad = _source_elevation(ant_xyz[a], ra_rad, dec_rad, jds[ti])
            elevation_deg[ti, a] = rad2deg(el_rad)
        end

        # Tsys is constant across the scan window; precompute once per
        # (channel, pol) for this antenna by averaging antab rows in the
        # leaf's scan window.
        scan_tsys = Matrix{Float64}(undef, nchan, 2)
        for p in 1:2, c in 1:nchan
            scan_tsys[c, p] = tsys_in_window(st, t_lo, t_hi, c, pol_syms[p])
        end

        for ti in 1:nti
            el = elevation_deg[ti, a]
            gE = elevation_gain(st.gain, el)
            for p in 1:2
                dpfu = st.gain.dpfu[p]
                for c in 1:nchan
                    tsys = scan_tsys[c, p]
                    if !(isfinite(tsys) && isfinite(gE) && isfinite(dpfu) && dpfu > 0 && gE > 0 && tsys > 0)
                        sefd[c, ti, a, p] = NaN
                        gains[c, ti, a, p] = NaN
                    else
                        s = tsys / (dpfu * gE)
                        sefd[c, ti, a, p] = s
                        gains[c, ti, a, p] = 1.0 / sqrt(s)
                    end
                end
            end
        end
    end

    if !isempty(missing_stations)
        if on_missing_station === :error
            error("apply_calibration: ANTAB has no record for stations $(missing_stations)")
        elseif on_missing_station === :warn
            @warn "apply_calibration: ANTAB has no record for stations; baselines involving them are left unchanged" stations=missing_stations track=antab.track_label
        end
    end

    return AprioriFluxGains(
        gains, sefd, elevation_deg, collect(String.(ant_names)), missing_stations,
    )
end

"""
    apriori_flux_gains(uvset::UVSet, antab::AntabCalibration; on_missing_station = :warn) -> Dict{Symbol, AprioriFluxGains}

Build per-leaf apriori-calibration gains without modifying `uvset`. Useful
for inspection: each value carries the real-valued amplitude gains, the
underlying SEFD array, and per-(integration, antenna) elevation samples.

`on_missing_station` controls behavior when the ANTAB lacks a station
present in the data: `:warn` (default) leaves those baselines unchanged
with a warning; `:error` raises; `:ignore` suppresses the message.
"""
function apriori_flux_gains(
        uvset::UVSet, antab::AntabCalibration;
        on_missing_station::Symbol = :warn,
    )
    out = Dict{Symbol, AprioriFluxGains}()
    root_meta = DimensionalData.metadata(uvset)
    for (k, leaf) in DimensionalData.branches(uvset)
        info = DimensionalData.metadata(leaf)
        out[k] = _build_apriori_gains(
            leaf, info, root_meta, antab;
            on_missing_station = on_missing_station,
        )
    end
    return out
end

"""
    apply_calibration(uvset::UVSet, antab::AntabCalibration; on_missing_station = :warn) -> UVSet

Apply a priori flux calibration to `uvset` from a parsed ANTAB. Per
antenna, channel, polarization, and integration time, computes
`SEFD = T_sys^eff / (DPFU * g_E(elevation))` and rescales each visibility
by `sqrt(SEFD_a * SEFD_b)` so the output amplitudes are in Jy.

This implementation mirrors the structure of [`apply_bandpass`](@ref):
visibilities are divided by `g_a * conj(g_b)` with `g_a = 1/sqrt(SEFD_a)`
(real, positive), and weights scale by `|g_a|^2 |g_b|^2`. Samples whose
ANTAB Tsys is missing or non-positive are flagged (weight set to 0).

The caller is responsible for matching the ANTAB to the right uvfits
track and band; pass `on_missing_station=:error` to refuse to silently
skip stations.
"""
function apply_calibration(
        uvset::UVSet, antab::AntabCalibration;
        on_missing_station::Symbol = :warn,
    )
    return UVData.apply(uvset) do leaf, info, root
        gains_pkg = _build_apriori_gains(
            leaf, info, root, antab; on_missing_station = on_missing_station,
        )
        bl_pairs = baselines(leaf).pairs
        vis_l = leaf[:vis]
        w_l = leaf[:weights]
        vis_corr, weights_corr = _apply_apriori_kernel(
            vis_l, w_l, gains_pkg.gains, bl_pairs, pol_products(leaf),
        )
        return with_visibilities(leaf, vis_corr, weights_corr)
    end
end

# Apply per-(channel, integration, antenna, feed) real-valued gains. NaN
# gains flag the sample (weight ← 0); non-NaN scaling matches the
# bandpass kernel convention so the two corrections compose cleanly.
function _apply_apriori_kernel(
        vis_p::AbstractArray, w_p::AbstractArray,
        gains::AbstractArray{Float64, 4},
        bl_pairs, pol_products,
    )
    vis_corr = copy(vis_p)
    weights_corr = copy(w_p)
    for ti in axes(vis_p, Ti), bi in axes(vis_p, Baseline)
        a, b = bl_pairs[bi]
        for p in axes(vis_p, Pol)
            fa, fb = correlation_feed_pair(pol_products[Int(p)])
            for c in axes(vis_p, Frequency)
                w = w_p[c, ti, bi, p]
                (w > 0 && isfinite(w)) || continue
                ga = gains[c, ti, a, fa]
                gb = gains[c, ti, b, fb]
                if !(isfinite(ga) && isfinite(gb))
                    weights_corr[c, ti, bi, p] = zero(w)
                    continue
                end
                vis_corr[c, ti, bi, p] /= ga * gb
                weights_corr[c, ti, bi, p] *= (ga * gb)^2
            end
        end
    end
    return vis_corr, weights_corr
end
