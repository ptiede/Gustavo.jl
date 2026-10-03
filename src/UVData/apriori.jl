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

"""
    AprioriFluxGains

A-priori calibration result for one block of data, returned by
[`apriori_gains`](@ref). The `gains` field carries the real, positive
amplitude gains, indexed as `gains[c, ti, a, p]` where `p ∈ {1, 2}` is the
feed slot (P/Q in MSv4 labels, R/L in EHT convention).
"""
struct AprioriFluxGains
    gains::Array{Float64, 4}                  # (nchan, nti, nant, 2)
    sefd::Array{Float64, 4}                   # raw SEFD per (chan, ti, ant, feed)
    elevation_deg::Matrix{Float64}            # (nti, nant)
    antennas::Vector{String}
    missing_stations::Vector{String}
end

"""
    apriori_gains(antab, antennas, chan_index, jds, t_lo, t_hi, ra, dec;
                  on_missing_station = :warn, min_elevation_deg = 0.0)
        -> AprioriFluxGains

SEFD-derived amplitude gains for one block of data, from explicit coordinates:
`antennas` is an [`AntennaTable`](@ref), `jds` the block's
integration epochs as Julian Dates (UTC), `[t_lo, t_hi]` the scan window whose
ANTAB rows are averaged for Tsys, and `ra`/`dec` the source position in radians.

`chan_index[c]` is the ANTAB channel number of the block's `c`-th channel. An
ANTAB numbers channels **within a spectral window** (ALMA's `'L1|R1' … 'L32|R32'`
declares 32 per-channel SEFDs for one band), so a block spanning several spws
must say which channel of which band each of its columns is; a single-spw block
passes `Base.OneTo(nchan)`. Stations whose ANTAB declares aggregate Tsys ignore
the index entirely.
"""
function apriori_gains(
        antab::AntabCalibration, ant_table::AntennaTable,
        chan_index::AbstractVector{<:Integer}, jds::AbstractVector{<:Real},
        t_lo::DateTime, t_hi::DateTime, ra::Real, dec::Real;
        on_missing_station::Symbol = :warn, min_elevation_deg::Real = 0.0,
    )
    on_missing_station in (:warn, :error, :ignore) || throw(
        ArgumentError(
            "on_missing_station must be :warn, :error, or :ignore (got :$(on_missing_station))"
        )
    )
    # The gain/SEFD cubes below are freshly allocated 1-based arrays indexed in
    # lockstep with these inputs.
    Base.require_one_based_indexing(chan_index, jds)

    ant_names = ant_table.name
    ant_xyz = ant_table.station_xyz
    nant = length(ant_names)
    nchan = length(chan_index)
    nti = length(jds)

    ra_rad = Float64(ra)
    dec_rad = Float64(dec)

    elevation_deg = Matrix{Float64}(undef, nti, nant)
    sefd = Array{Float64}(undef, nchan, nti, nant, 2)
    gains = Array{Float64}(undef, nchan, nti, nant, 2)

    pol_syms = (:R, :L)
    missing_stations = String[]

    for (a, name) in pairs(ant_names)
        if !haskey(antab, name)
            push!(missing_stations, String(name))
            for ti in axes(elevation_deg, 1)
                elevation_deg[ti, a] = NaN
                for c in axes(sefd, 1), p in axes(sefd, 4)
                    sefd[c, ti, a, p] = NaN
                    gains[c, ti, a, p] = 1.0
                end
            end
            continue
        end
        st = antab[name]
        # Per-time elevation only depends on the antenna position, not channel/pol.
        for ti in axes(elevation_deg, 1)
            el_rad = _source_elevation(ant_xyz[a], ra_rad, dec_rad, jds[ti])
            elevation_deg[ti, a] = rad2deg(el_rad)
        end

        # Tsys is constant across the scan window; precompute once per
        # (channel, pol) for this antenna by averaging antab rows in the
        # scan window.
        scan_tsys = Matrix{Float64}(undef, nchan, 2)
        for p in axes(scan_tsys, 2), c in axes(scan_tsys, 1)
            scan_tsys[c, p] = tsys_in_window(st, t_lo, t_hi, Int(chan_index[c]), pol_syms[p])
        end

        for ti in axes(elevation_deg, 1)
            el = elevation_deg[ti, a]
            gE = elevation_gain(st.gain, el)
            # Below the elevation cutoff (default: the horizon) the source is
            # not observable and the gain-curve polynomial extrapolates to tiny
            # / negative values, which makes SEFD = Tsys/(DPFU·gE) explode. Flag
            # those samples on apply rather than apply a blown-up gain.
            el_ok = isfinite(el) && el >= min_elevation_deg
            for p in axes(sefd, 4)
                dpfu = st.gain.dpfu[p]
                for c in axes(sefd, 1)
                    tsys = scan_tsys[c, p]
                    if !(el_ok && isfinite(tsys) && isfinite(gE) && isfinite(dpfu) && dpfu > 0 && gE > 0 && tsys > 0)
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
            throw(
                ArgumentError(
                    "ANTAB $(repr(antab.track_label)) has no record for stations " *
                        "$(missing_stations); baselines involving them would pass through " *
                        "uncalibrated. Pass `on_missing_station = :warn` or `:ignore` to " *
                        "allow that.",
                )
            )
        elseif on_missing_station === :warn
            @warn "apriori_gains: ANTAB has no record for stations; baselines involving them are left unchanged" stations = missing_stations track = antab.track_label
        end
    end

    return AprioriFluxGains(
        gains, sefd, elevation_deg, collect(String.(ant_names)), missing_stations,
    )
end

# Apply per-(channel, integration, antenna, feed) real-valued gains. A
# non-finite gain flags the sample; non-NaN scaling matches the bandpass kernel
# convention so the two corrections compose cleanly.
#
# A flagged sample keeps the visibility and weight it arrived with: no gain was
# applied to it, so there is nothing to record beyond the flag itself.
function _apply_apriori_kernel(
        vis_p::AbstractArray, w_p::AbstractArray, flags_p::AbstractArray,
        gains::AbstractArray{Float64, 4},
        bl_pairs, feeds,
    )
    check_layer_axes(vis_p, w_p, flags_p)
    vis_corr = copy(vis_p)
    weights_corr = copy(w_p)
    flags_corr = copy(flags_p)
    for ti in axes(vis_p, Ti), bi in axes(vis_p, BaselineID)
        a, b = bl_pairs[bi]
        # Autocorrelations (a == b) are total power, not interferometric
        # visibilities: `√(SEFD_a·SEFD_b)` flux-scaling is meaningless and blows
        # their amplitude up by the SEFD. Flag them so they are not used
        # downstream — the fringe solve already skips them.
        if a == b
            for p in axes(vis_p, Polarization), c in axes(vis_p, Frequency)
                flags_corr[Frequency(c), Ti(ti), BaselineID(bi), Polarization(p)] = true
            end
            continue
        end
        for p in axes(vis_p, Polarization)
            fa, fb = feeds[p]
            for c in axes(vis_p, Frequency)
                cell = (Frequency(c), Ti(ti), BaselineID(bi), Polarization(p))
                flags_p[cell] && continue
                w = w_p[cell]
                (w > 0 && isfinite(w)) || continue
                ga = gains[c, ti, a, fa]
                gb = gains[c, ti, b, fb]
                if !(isfinite(ga) && isfinite(gb))
                    flags_corr[cell] = true
                    continue
                end
                vis_corr[cell] /= ga * gb
                weights_corr[cell] *= (ga * gb)^2
            end
        end
    end
    return vis_corr, weights_corr, flags_corr
end
