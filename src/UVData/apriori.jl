"""
    apriori_calibrate!(ms::XRadio.MeasurementSet; tsys = ScanMean(), min_elevation = 0.0,
                       max_tsys = 1.0e4, tsys_placeholders = (999.0,),
                       quantization_efficiency = nothing) -> ms

Put the visibilities of `ms` in janskys, in place, from the system
temperatures and gain curves the Measurement Set records
(`XRadio.system_temperatures`, `XRadio.gaincurves`), whether they came from
FITS-IDI (`XRadio.fitsidi2msv4`) or ANTAB (`XRadio.read_antab!`).

Each receptor's system equivalent flux density is `T_sys / (DPFU · g)`, with
`g` its gain curve at the source's elevation and `DPFU` the curve's
sensitivity. A visibility on baseline `(a, b)` relating receptors `(fa, fb)` is
multiplied by `√(SEFD_a · SEFD_b)` and its weight divided by `SEFD_a · SEFD_b`,
so the weight stays `1/σ²` of the visibility it describes. Receptors come from
[`feed_pairs`](@ref).

The same visibility is divided by `η_a · η_b` and its weight multiplied by
`(η_a · η_b)²`, which corrects the loss of correlated amplitude to the
correlator's quantization. `η` is each antenna's quantization efficiency,
from the `digitizer_levels` variable of the antenna dataset: 2, 3 and 4 levels
give `2/π`, 0.8098 and 0.8825, the weak-signal efficiencies with optimal
thresholds (Thompson, Moran & Swenson, *Interferometry and Synthesis in Radio
Astronomy*, ch. 8). A Measurement Set without `digitizer_levels` needs
`quantization_efficiency`: one number for every antenna, or a `Dict` from
antenna name (string or symbol) to η holding every antenna. Where
`digitizer_levels` are recorded, `quantization_efficiency` supplies η only for
the antennas whose level count has none tabulated, and must be given for them:
one number for all of them, or a `Dict` naming no other antenna.

`tsys` places the system temperatures, which are sampled on their own clock, on
the visibilities' times: [`ScanMean`](@ref), [`LinearInTime`](@ref) or
[`NearestInTime`](@ref). A system temperature that is not positive, exceeds
`max_tsys` (kelvin), or equals one of `tsys_placeholders` (values written for
"not measured") counts as unmeasured before it is placed; the store keeps it.
Elevations come from the antenna positions, the field
phase center and the time; a source below `min_elevation` (radians) has no
usable gain.

A sample whose antennas lack a usable system temperature, gain curve or
elevation is flagged and keeps its visibility and weight. Autocorrelations are
flagged. The visibility `units` become `Jy`; a Measurement Set whose
visibilities are already in janskys is an error, as is one that records no
system temperatures or no gain curves.
"""
function apriori_calibrate!(
        ms::XRadio.MeasurementSet; tsys = ScanMean(), min_elevation::Real = 0.0,
        max_tsys::Real = 1.0e4, tsys_placeholders = (999.0,), quantization_efficiency = nothing,
    )
    vis = ms[:visibility]
    units = get(metadata(vis), :units, nothing)
    units == "Jy" && throw(ArgumentError(
        "the visibilities of spectral window `$(XRadio.spectralwindow(ms))` are already in Jy"
    ))
    T = XRadio.system_temperatures(ms)
    isempty(T) && throw(ArgumentError(
        "spectral window `$(XRadio.spectralwindow(ms))` records no system temperatures; " *
            "`XRadio.read_antab!` reads them from an ANTAB file"
    ))
    curves = XRadio.gaincurves(ms)
    isempty(curves[:curve]) && throw(ArgumentError(
        "spectral window `$(XRadio.spectralwindow(ms))` records no gain curves; " *
            "`XRadio.read_antab!` reads them from an ANTAB file"
    ))

    names = collect(lookup(XRadio.polarization_types(ms), XRadio.AntennaName))
    η = _quantization_efficiencies(ms, names, quantization_efficiency)
    plausible(v) = isfinite(v) && 0 < v <= max_tsys && !(v in tsys_placeholders)
    scale = _sefd_scale(ms, T, curves, tsys, Float64(min_elevation), plausible)
    feeds = feed_pairs(ms)
    V, W, F = _storage_order(vis), _storage_order(ms[:weight]), _storage_order(ms[:flag])
    for (bi, (a, b)) in pairs(_antenna_pairs(ms))
        if a == b
            view(F, :, :, bi, :) .= true
            continue
        end
        for t in axes(V, 4), p in axes(V, 1)
            fa, fb = feeds[p, bi]
            for c in axes(V, 2)
                s = scale[fa, a, t, _channel(scale, c)] * scale[fb, b, t, _channel(scale, c)] /
                    (η[a] * η[b])
                if isfinite(s) && s > 0
                    V[p, c, bi, t] *= s
                    W[p, c, bi, t] /= s^2
                else
                    F[p, c, bi, t] = true
                end
            end
        end
    end
    metadata(vis)[:units] = "Jy"
    return ms
end

_channel(scale, c) = size(scale, 4) == 1 ? 1 : c

# Weak-signal quantization efficiency with optimal thresholds, by digitizer
# level count (Thompson, Moran & Swenson, ch. 8).
const _QUANTIZATION_EFFICIENCY = Dict(2 => 2 / π, 3 => 0.8098, 4 => 0.8825)

function _quantization_efficiencies(ms, names, given)
    xds = branches(ms)[:antenna]
    window = XRadio.spectralwindow(ms)
    if !haskey(xds, :digitizer_levels)
        isnothing(given) && throw(ArgumentError(
            "spectral window `$window` records no `digitizer_levels`; pass " *
                "`quantization_efficiency` (e.g. `2/π` for 2-level sampling) or record " *
                "`digitizer_levels` in the antenna dataset"
        ))
        return _given_efficiencies(given, names)
    end
    levels = xds[:digitizer_levels]
    level(a) = levels[XRadio.AntennaName(At(a))]
    untabulated = filter(a -> !haskey(_QUANTIZATION_EFFICIENCY, level(a)), names)
    if isnothing(given)
        isempty(untabulated) || throw(ArgumentError(
            "antennas $(join(untabulated, ", ")) have $(join(map(level, untabulated), ", ")) " *
                "digitizer levels, for which no quantization efficiency is tabulated (2, 3 " *
                "and 4 are); pass `quantization_efficiency` for them"
        ))
    else
        isempty(untabulated) && throw(ArgumentError(
            "spectral window `$window` records `digitizer_levels` with a tabulated quantization " *
                "efficiency for every antenna; `quantization_efficiency` cannot also be given"
        ))
        _check_untabulated_only(given, untabulated)
    end
    supplied = isnothing(given) ? Dict{String, Float64}() :
        Dict(zip(untabulated, _given_efficiencies(given, untabulated)))
    return [a in untabulated ? supplied[a] : _QUANTIZATION_EFFICIENCY[level(a)] for a in names]
end

_check_untabulated_only(::Real, untabulated) = nothing
function _check_untabulated_only(η::AbstractDict, untabulated)
    extra = setdiff(String.(keys(η)), untabulated)
    isempty(extra) || throw(ArgumentError(
        "`quantization_efficiency` gives antennas $(join(extra, ", ")), whose " *
            "`digitizer_levels` fix it; give it only for $(join(untabulated, ", "))"
    ))
    return nothing
end

_given_efficiencies(η::Real, names) = fill(η, length(names))
function _given_efficiencies(η::AbstractDict, names)
    byname = Dict(String(k) => v for (k, v) in η)
    return map(names) do a
        haskey(byname, a) || throw(ArgumentError("`quantization_efficiency` has no value for antenna $a"))
        byname[a]
    end
end

# √SEFD over (receptor, antenna, time, channel) in the antenna dataset's order,
# with a channel axis of length one unless the system temperatures resolve the
# data's channels. `NaN` where any ingredient is missing or unusable.
function _sefd_scale(ms, T, curves, rule, min_elevation, plausible)
    A, R = XRadio.AntennaName, XRadio.ReceptorLabel
    types = XRadio.polarization_types(ms)
    names, receptors = collect(lookup(types, A)), collect(lookup(types, R))
    times = XRadio.times(ms)
    scans = collect(ms[:scan_name])
    meta = metadata(lookup(ms[:visibility], Ti))
    half = haskey(meta, :integration_time) ? meta[:integration_time].value / 2 : 0.0

    tdim = only(d for d in dims(T) if d isa Union{XRadio.TimeSystemCal, Ti})
    hasdim(T, XRadio.FrequencySystemCal) && throw(ArgumentError(
        "the system temperatures are over `frequency_system_cal`, which MSv4 does not define"
    ))
    resolved = hasdim(T, XRadio.Frequency)
    resolved && collect(lookup(T, XRadio.Frequency)) != XRadio.frequencies(ms) && throw(ArgumentError(
        "the system temperatures are resolved over frequencies other than the visibilities'"
    ))
    nchan = resolved ? length(XRadio.frequencies(ms)) : 1
    rows = collect(lookup(tdim))
    elevation = _elevations(ms, names, times)

    scale = fill(NaN, length(receptors), length(names), length(times), nchan)
    for (i, a) in pairs(names), (j, r) in pairs(receptors)
        (a in lookup(T, A) && a in lookup(curves[:curve], A) && r in lookup(T, R)) || continue
        curve = curves[:curve][A(At(a)), R(At(r))]
        dpfu = curves[:sensitivity][A(At(a)), R(At(r))]
        series = T[A(At(a)), R(At(r))]
        for c in 1:nchan
            values = [plausible(v) ? v : NaN for v in (resolved ? series[XRadio.Frequency(c)] : series)]
            placed = place_tsys(rule, rows, values, times, scans, half)
            for t in eachindex(times)
                el = elevation[t, i]
                el >= min_elevation || continue
                g = XRadio.gain(curve, (; elevation = el, zenith_angle = π / 2 - el))
                XRadio.gain_quantity(curve) === :voltage && (g = g^2)
                sefd = placed[t] / (dpfu * g)
                isfinite(sefd) && sefd > 0 && (scale[j, i, t, c] = sqrt(sefd))
            end
        end
    end
    return scale
end

# Source elevation (radians) over (time, antenna), from each antenna's position,
# the phase center of the field observed at each time, and the time as UTC.
function _elevations(ms, names, times)
    meta = metadata(lookup(ms[:visibility], Ti))
    (get(meta, :scale, nothing), get(meta, :format, nothing)) == ("utc", "unix") || throw(ArgumentError(
        "elevations need UTC times, and this Measurement Set's are " *
            "$(get(meta, :scale, nothing)) $(get(meta, :format, nothing))"
    ))
    positions = XRadio.antenna_positions(ms)
    directions = XRadio.field_and_source(ms)[:field_phase_center_direction]
    frame = get(metadata(directions), :frame, nothing)
    frame in ("icrs", "fk5") || throw(ArgumentError(
        "elevations need an equatorial phase center, and this one's frame is $frame"
    ))
    fields = collect(ms[:field_name])
    sky = XRadio.SkyDirLabel
    out = Matrix{Float64}(undef, length(times), length(names))
    for (i, a) in pairs(names)
        xyz = collect(positions[XRadio.AntennaName(At(a))])
        for (t, time) in pairs(times)
            d = directions[XRadio.FieldName(At(fields[t]))]
            out[t, i] = _source_elevation(xyz, d[sky(At("ra"))], d[sky(At("dec"))], time / 86400 + 2440587.5)
        end
    end
    return out
end

"""
    TsysPlacement

How [`apriori_calibrate!`](@ref) places system temperatures, sampled on their
own clock, on the visibilities' times: [`ScanMean`](@ref),
[`LinearInTime`](@ref) or [`NearestInTime`](@ref). A new rule is a subtype with
a method of

    Gustavo.UVData.place_tsys(rule, rows, values, times, scans, half) -> Vector

returning one temperature per visibility time (`NaN` for none), given the
temperature rows' times and values (`NaN` where unmeasured), the visibility
times and their scan names, and half the integration time.
"""
abstract type TsysPlacement end

"""
    ScanMean()

The mean of the system temperatures inside each scan's time span, so a
measurement taken while slewing or on another source between scans does not
reach the scan. A scan with none is `NaN`.
"""
struct ScanMean <: TsysPlacement end

"""
    LinearInTime()

Linear interpolation between the system temperature rows either side of each
visibility time; `NaN` outside the first and last rows.
"""
struct LinearInTime <: TsysPlacement end

"""
    NearestInTime()

The system temperature row nearest each visibility time.
"""
struct NearestInTime <: TsysPlacement end

function place_tsys(::ScanMean, rows, values, times, scans, half)
    out = fill(NaN, length(times))
    for s in unique(scans)
        idx = findall(==(s), scans)
        lo, hi = minimum(times[idx]) - half, maximum(times[idx]) + half
        inside = [v for (t, v) in zip(rows, values) if lo <= t <= hi && isfinite(v)]
        isempty(inside) || (out[idx] .= mean(inside))
    end
    return out
end

function place_tsys(::LinearInTime, rows, values, times, scans, half)
    keep = findall(isfinite, values)
    r, v = rows[keep], values[keep]
    return map(times) do t
        isempty(r) && return NaN
        (t < first(r) || t > last(r)) && return NaN
        hi = searchsortedfirst(r, t)
        r[hi] == t && return v[hi]
        lo = hi - 1
        return v[lo] + (t - r[lo]) / (r[hi] - r[lo]) * (v[hi] - v[lo])
    end
end

function place_tsys(::NearestInTime, rows, values, times, scans, half)
    keep = findall(isfinite, values)
    isempty(keep) && return fill(NaN, length(times))
    return [values[keep[argmin(abs.(rows[keep] .- t))]] for t in times]
end

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
