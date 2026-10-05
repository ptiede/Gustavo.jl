using FITSFiles: Bintable, Card, Random
using DimensionalData: DimensionalData, Ti, lookup
using Dates: unix2datetime
using XRadio: AbstractMount

# ── Encoders ────────────────────────────────────────────────────────────────

function _encode_aips_baseline(a::Integer, b::Integer)
    (1 <= a < 256 && 1 <= b < 256) || throw(
        ArgumentError(
            "write_uvfits: the BASELINE parameter packs antennas (a, b) as a*256 + b, " *
                "so it numbers at most 255 antennas; this baseline is $((a, b))"
        )
    )
    return 256 * a + b
end

const _AIPS_STOKES_CODES = Dict(label => code for (code, label) in _AIPS_STOKES_LABELS)

const _POLTYPE_LETTERS = Dict(v => k for (k, v) in UVData._POL_TYPES)

function _mount_to_mntsta(m::AbstractMount)
    for (code, make) in pairs(_MNTSTA_MOUNTS)
        m == make(XRadio.axis_offset(m)) && return Int32(code - 1)
    end
    throw(ArgumentError("write_uvfits: $m has no AIPS MNTSTA code"))
end

function _uvfits_staxof(m::AbstractMount)
    x, y, z = XRadio.axis_offset(m)
    iszero(y) && iszero(z) || throw(
        ArgumentError(
            "write_uvfits: $m has an axis offset off the station's x axis, which " *
                "the AN table's scalar STAXOF cannot state"
        )
    )
    return Float32(x)
end

# Random groups carry one float type for the data array and every parameter.
function _fits_float_type(T::Type{<:AbstractFloat})
    promote_type(T, Float32) === Float32 && return Float32
    promote_type(T, Float64) === Float64 && return Float64
    throw(
        ArgumentError(
            "write_uvfits: the set holds $T values, which FITS cannot store; its " *
                "floating-point forms are Float32 and Float64"
        )
    )
end

const _EARTH_ORIENTATION_DEFAULTS = (
    gst_iat0 = 0.0, earth_rot_rate = 360.0, ut1utc = 0.0, polarx = 0.0, polary = 0.0,
    datutc = 0.0, xyzhand = "RIGHT", poltype = "",
)

# ── Validation of the set's shape ───────────────────────────────────────────

# The set's scans in time order, each as its name and its windows in frequency
# order.
function _scans(ps)
    groups = OrderedDict{String, Vector{XRadio.MeasurementSet}}()
    for ms in ps
        push!(get!(Vector{XRadio.MeasurementSet}, groups, only(XRadio.scans(ms))), ms)
    end
    scans = [name => sort(sets; by = ms -> minimum(XRadio.frequencies(ms))) for (name, sets) in groups]
    sort!(scans; by = s -> minimum(XRadio.times(first(last(s)))))
    for ((name, sets), (next, later)) in zip(scans, Iterators.drop(scans, 1))
        maximum(XRadio.times(first(sets))) < minimum(XRadio.times(first(later))) || throw(
            ArgumentError(
                "write_uvfits: scans $name and $next overlap in time; a UVFITS file " *
                    "written here holds one subarray, and a set of several cannot be written as one"
            )
        )
    end
    return scans
end

_window_signature(fs) = (
    channel_freqs = collect(UVData.channel_freqs(fs)), ch_width = first(UVData.ch_widths(fs)),
    sideband = Int(first(UVData.sidebands(fs))), total_bandwidth = first(UVData.total_bandwidths(fs)),
    ref_freq = UVData.ref_freq(fs),
)

# The windows every scan shares, checking that each scan holds them on one time
# and baseline axis.
function _windows_of(scans)
    reference = nothing
    first_scan = first(first(scans))
    for (name, sets) in scans
        times, bls = XRadio.times(first(sets)), XRadio.baselines(first(sets))
        for ms in sets
            XRadio.times(ms) == times && XRadio.baselines(ms) == bls || throw(
                ArgumentError(
                    "write_uvfits: the windows of scan $name do not share their time " *
                        "and baseline axes, which a UVFITS record spans"
                )
            )
        end
        signature = [_window_signature(UVData.freq_setup(ms)) for ms in sets]
        allunique(s.channel_freqs for s in signature) || throw(
            ArgumentError(
                "write_uvfits: scan $name holds two Measurement Sets with the same " *
                    "channel frequencies; a set of several subarrays cannot be written as one"
            )
        )
        if isnothing(reference)
            reference = signature
        else
            same = length(signature) == length(reference) && all(
                s.channel_freqs == r.channel_freqs && s.ch_width == r.ch_width &&
                    s.sideband == r.sideband && s.total_bandwidth == r.total_bandwidth
                    for (s, r) in zip(signature, reference)
            )
            same || throw(
                ArgumentError(
                    "write_uvfits: the spectral windows of scan $name differ from those " *
                        "of scan $first_scan in number, channel frequencies, widths, " *
                        "sidebands or total bandwidths; a UVFITS file states one frequency setup"
                )
            )
        end
    end
    return reference
end

# The FQ row and the (FREQ, IF) placement of each (window, channel).
function _frequency_layout(windows)
    ref_freq = first(windows).ref_freq
    nchan = [length(w.channel_freqs) for w in windows]
    if length(windows) == 1
        w = only(windows)
        n = only(nchan)
        return (;
            ref_freq, nfreq = 1, nif = n, place = (_, c) -> (1, c),
            if_freq = w.channel_freqs .- ref_freq, ch_width = fill(w.ch_width, n),
            total_bandwidth = fill(w.total_bandwidth, n), sideband = fill(w.sideband, n),
        )
    end
    allequal(nchan) || throw(
        ArgumentError(
            "write_uvfits: the spectral windows hold $(nchan) channels; the FREQ axis " *
                "gives every IF the same number"
        )
    )
    if first(nchan) == 1
        allequal((w.ch_width, w.sideband, w.total_bandwidth) for w in windows) || throw(
            ArgumentError(
                "write_uvfits: one-channel windows are written as the IFs of one window, " *
                    "which states one channel width, sideband and total bandwidth; " *
                    "these windows differ in them"
            )
        )
    else
        for w in windows
            spacing = diff(w.channel_freqs)
            all(s -> isapprox(s, w.ch_width; rtol = 1.0e-9), spacing) || throw(
                ArgumentError(
                    "write_uvfits: the window starting at $(first(w.channel_freqs)) Hz " *
                        "has channels not evenly spaced by its channel width " *
                        "$(w.ch_width) Hz, which the FREQ axis requires"
                )
            )
        end
    end
    return (;
        ref_freq, nfreq = first(nchan), nif = length(windows), place = (w, c) -> (c, w),
        if_freq = [first(w.channel_freqs) - ref_freq for w in windows],
        ch_width = [w.ch_width for w in windows],
        total_bandwidth = [w.total_bandwidth for w in windows],
        sideband = [w.sideband for w in windows],
    )
end

# The products' AIPS Stokes codes in on-disk order (descending).
function _stokes_codes(ps)
    labels = XRadio.polarizations(first(ps))
    for ms in ps
        Set(XRadio.polarizations(ms)) == Set(labels) || throw(
            ArgumentError(
                "write_uvfits: the Measurement Sets hold polarizations $(labels) and " *
                    "$(XRadio.polarizations(ms)); a UVFITS file states one STOKES axis"
            )
        )
    end
    codes = map(labels) do label
        get(_AIPS_STOKES_CODES, label) do
            throw(ArgumentError("write_uvfits: polarization $label has no AIPS Stokes code"))
        end
    end
    sort!(codes; rev = true)
    contiguous = all(==(-1), diff(codes)) &&
        (all(c -> -4 <= c <= -1, codes) || all(c -> -8 <= c <= -5, codes))
    contiguous || throw(
        ArgumentError(
            "write_uvfits: polarizations $(labels) are AIPS Stokes codes $(codes), " *
                "which are not a contiguous run of one basis that a STOKES axis can state"
        )
    )
    return codes
end

function _antenna_union(ps)
    rows = UVData.Antenna[]
    diameters = Union{Nothing, Float64}[]
    for ms in ps
        tab = UVData._antenna_table(ms)
        diameter = get(UVData.extras(tab), :DIAMETER, nothing)
        for (i, ant) in pairs(getfield(tab, :antennas))
            j = findfirst(r -> r.name == ant.name, rows)
            if isnothing(j)
                push!(rows, ant)
                push!(diameters, isnothing(diameter) ? nothing : Float64(diameter[i]))
            else
                isequal(rows[j], ant) || throw(
                    ArgumentError(
                        "write_uvfits: antenna $(ant.name) has different positions, mounts " *
                            "or receptors in different Measurement Sets; a UVFITS file " *
                            "written here holds one antenna table"
                    )
                )
            end
        end
    end
    return rows, any(isnothing, diameters) ? nothing : Float64.(diameters),
        UVData.array_name(UVData._antenna_table(first(ps)))
end

function _only_value(ps, f, what)
    values = unique(f(ms) for ms in ps)
    length(values) == 1 || throw(
        ArgumentError("write_uvfits: the Measurement Sets state different $what: $(values)")
    )
    return only(values)
end

function _earth_orientation(ps)
    eo = _only_value(ps, ms -> get(DimensionalData.metadata(ms), :earth_orientation, nothing), "Earth orientation")
    if isnothing(eo)
        @warn "write_uvfits: the Measurement Sets state no `earth_orientation`; the AN \
               table's GSTIA0, DEGPDY, UT1UTC, POLARX, POLARY, DATUTC, XYZHAND and \
               POLTYPE are written as defaults" defaults = _EARTH_ORIENTATION_DEFAULTS
        return _EARTH_ORIENTATION_DEFAULTS
    end
    return NamedTuple{keys(_EARTH_ORIENTATION_DEFAULTS)}(Tuple(eo[k] for k in keys(_EARTH_ORIENTATION_DEFAULTS)))
end

function _bunit(units)
    isnothing(units) && return nothing
    units == "Jy" && return "JY"
    units == "uncalib" && return "UNCALIB"
    return units
end

# ── Records ─────────────────────────────────────────────────────────────────

const _VIS_ORDER = (XRadio.Polarization, XRadio.Frequency, XRadio.BaselineID, Ti)

_dense(A, order) = Array(permutedims(A, order))

function _window_arrays(ms, codes)
    labels = XRadio.polarizations(ms)
    return (;
        vis = _dense(XRadio.correlated(ms), _VIS_ORDER),
        weight = _dense(XRadio.weights(ms), _VIS_ORDER),
        flag = _dense(XRadio.flags(ms), _VIS_ORDER),
        pol = [findfirst(==(_AIPS_STOKES_LABELS[c]), labels) for c in codes],
    )
end

function _integration_times(ms)
    haskey(ms, :effective_integration_time) &&
        return _dense(ms[:effective_integration_time], (XRadio.BaselineID, Ti))
    meta = DimensionalData.metadata(lookup(ms, Ti))
    haskey(meta, :integration_time) || throw(
        ArgumentError(
            "write_uvfits: a Measurement Set states neither `effective_integration_time` " *
                "nor the time axis' `integration_time`, so INTTIM has no value"
        )
    )
    nominal = XRadio.value(meta[:integration_time])
    return fill(nominal, length(XRadio.baselines(ms)), length(XRadio.times(ms)))
end

# One FQ row; a single IF is a scalar column.
_fq_column(v) = length(v) == 1 ? v : [v]

_is_gap(A, k, b) = all(
    A.flag[p, c, b, k] && iszero(A.weight[p, c, b, k]) for p in A.pol, c in axes(A.vis, 2)
)

function UVData.write_uvfits(path, ps::XRadio.ProcessingSet; overwrite::Bool = false)
    ispath(path) && !overwrite && throw(
        ArgumentError("write_uvfits: $path exists; pass `overwrite = true` to replace it")
    )
    isempty(ps) && throw(ArgumentError("write_uvfits: the set holds no Measurement Sets"))
    srcs, flds = XRadio.sources(ps), XRadio.fields(ps)
    length(srcs) == 1 && length(flds) == 1 || throw(
        ArgumentError(
            "write_uvfits: the set holds sources $(srcs) and fields $(flds); a UVFITS " *
                "file holds one. Select one first, e.g. with `filter`."
        )
    )
    scans = _scans(ps)
    layout = _frequency_layout(_windows_of(scans))
    codes = _stokes_codes(ps)
    rows, diameters, arrnam = _antenna_union(ps)
    number = Dict(ant.name => n for (n, ant) in pairs(rows))
    eo = _earth_orientation(ps)
    units = _only_value(ps, ms -> get(DimensionalData.metadata(XRadio.correlated(ms)), :units, nothing), "visibility units")
    T = _fits_float_type(
        mapreduce(promote_type, ps) do ms
            promote_type(real(eltype(XRadio.correlated(ms))), eltype(XRadio.weights(ms)))
        end
    )

    written = map(scans) do (_, sets)
        arrays = [_window_arrays(ms, codes) for ms in sets]
        times = XRadio.times(first(sets))
        bls = eachindex(XRadio.baselines(first(sets)))
        keep = [(k, b) for k in eachindex(times) for b in bls if !all(A -> _is_gap(A, k, b), arrays)]
        (; sets, arrays, times, keep)
    end
    nrec = sum(s -> length(s.keep), written)
    nrec > 0 || throw(ArgumentError("write_uvfits: every sample of the set is a gap; no record to write"))

    t0 = minimum(s.times[first(first(s.keep))] for s in written if !isempty(s.keep))
    rdate = string(Date(unix2datetime(t0)))
    rdate_unix = UVData.jd_to_unix(_rdate_jd_or_zero(rdate))

    npol = length(codes)
    data = zeros(T, nrec, 3, npol, layout.nfreq, layout.nif, 1, 1)
    uu, vv, ww, baseline, inttim = (Vector{T}(undef, nrec) for _ in 1:5)
    date = Matrix{T}(undef, nrec, 2)
    nx = (; time = Float64[], interval = Float32[], start = Int32[], stop = Int32[])
    row = 0
    for s in written
        isempty(s.keep) && continue
        ref = first(s.sets)
        uvw = _dense(XRadio.uvw(ref), (XRadio.UVWLabel, XRadio.BaselineID, Ti))
        for ms in Iterators.drop(s.sets, 1)
            isequal(_dense(XRadio.uvw(ms), (XRadio.UVWLabel, XRadio.BaselineID, Ti)), uvw) || throw(
                ArgumentError(
                    "write_uvfits: spectral windows `$(XRadio.spectralwindow(ref))` and " *
                        "`$(XRadio.spectralwindow(ms))` of one scan state different uvw; a " *
                        "UVFITS record holds one"
                )
            )
        end
        integration = _integration_times(ref)
        names = XRadio.baselines(ref)
        start = row + 1
        for (k, b) in s.keep
            row += 1
            t = s.times[k]
            # The Julian Date of the day's 0h UT, and the fraction of that day.
            day = floor(UVData.JD_UNIX_EPOCH + t / 86400 - 0.5) + 0.5
            date[row, 1] = day
            date[row, 2] = t / 86400 - (day - UVData.JD_UNIX_EPOCH)
            uu[row], vv[row], ww[row] = (uvw[i, b, k] / _C_LIGHT for i in 1:3)
            a1, a2 = names[b]
            baseline[row] = _encode_aips_baseline(number[a1], number[a2])
            inttim[row] = integration[b, k]
            for (w, A) in pairs(s.arrays), c in axes(A.vis, 2), (p, q) in pairs(A.pol)
                f, i = layout.place(w, c)
                flagged, wt, v = A.flag[q, c, b, k], A.weight[q, c, b, k], A.vis[q, c, b, k]
                gap = flagged && iszero(wt)
                data[row, 1, p, f, i] = gap ? zero(T) : real(v)
                data[row, 2, p, f, i] = gap ? zero(T) : imag(v)
                data[row, 3, p, f, i] = flagged ? -abs(wt) : wt
            end
        end
        kept_times = [s.times[k] for (k, _) in s.keep]
        half = maximum(inttim[start:row]) / 2
        lo, hi = minimum(kept_times) - half, maximum(kept_times) + half
        push!(nx.time, ((lo + hi) / 2 - rdate_unix) / 86400)
        push!(nx.interval, (hi - lo) / 86400)
        push!(nx.start, start)
        push!(nx.stop, row)
    end

    first_ms = first(ps)
    fs = XRadio.field_and_source(first_ms)
    ra, dec = rad2deg.(collect(fs[:field_phase_center_direction])[:, 1])
    meta = DimensionalData.metadata(first_ms)
    observer = get(get(meta, :observation_info, Dict{Symbol, Any}()), :observer, String[])
    telescope = isempty(observer) || isempty(first(observer)) ? arrnam : string(first(observer))
    instrume = string(get(get(meta, :processor_info, Dict{Symbol, Any}()), :sub_type, ""))

    primary_cards = Card[
        Card("NAXIS", 7), Card("EXTEND", true), Card("BSCALE", 1.0), Card("BZERO", 0.0),
        Card("OBJECT", only(srcs)), Card("TELESCOP", telescope), Card("INSTRUME", instrume),
        Card("DATE-OBS", rdate),
    ]
    bunit = _bunit(units)
    isnothing(bunit) || push!(primary_cards, Card("BUNIT", bunit))
    axis_cards = (
        ("COMPLEX", 1.0, 1.0), ("STOKES", Float64(first(codes)), -1.0),
        ("FREQ", layout.ref_freq, first(layout.ch_width)), ("IF", 1.0, 1.0),
        ("RA", ra, 1.0), ("DEC", dec, 1.0),
    )
    push!(primary_cards, Card("EQUINOX", 2000.0), Card("EPOCH", 2000.0))
    for (n, (ctype, crval, cdelt)) in zip(2:7, axis_cards)
        push!(
            primary_cards, Card("CTYPE$n", ctype), Card("CRVAL$n", crval),
            Card("CDELT$n", cdelt), Card("CRPIX$n", 1.0), Card("CROTA$n", 0.0),
        )
    end
    push!(primary_cards, Card("OBSRA", ra), Card("OBSDEC", dec))
    params = (;
        var"UU---SIN" = uu, var"VV---SIN" = vv, var"WW---SIN" = ww,
        BASELINE = baseline, DATE = date, INTTIM = inttim, data,
    )
    primary = HDU(Random, params, primary_cards)

    nant = length(rows)
    an_data = (;
        ANNAME = [rpad(ant.name, 8) for ant in rows],
        STABXYZ = [Float64.(collect(ant.station_xyz)) for ant in rows],
        ORBPARM = [Float64[] for _ in 1:nant],
        NOSTA = Int32.(1:nant),
        MNTSTA = [_mount_to_mntsta(ant.mount) for ant in rows],
        STAXOF = [_uvfits_staxof(ant.mount) for ant in rows],
        POLTYA = [_POLTYPE_LETTERS[ant.nominal_basis[1]] for ant in rows],
        POLAA = [Float32(rad2deg(ant.pol_angles[1])) for ant in rows],
        POLTYB = [_POLTYPE_LETTERS[ant.nominal_basis[2]] for ant in rows],
        POLAB = [Float32(rad2deg(ant.pol_angles[2])) for ant in rows],
        POLCALA = [Float32[] for _ in 1:nant],
        POLCALB = [Float32[] for _ in 1:nant],
        (isnothing(diameters) ? (;) : (; DIAMETER = Float32.(diameters)))...,
    )
    an = HDU(
        Bintable, an_data, Card[
            Card("EXTNAME", "AIPS AN"), Card("EXTVER", Int32(1)),
            Card("ARRAYX", 0.0), Card("ARRAYY", 0.0), Card("ARRAYZ", 0.0),
            Card("ARRNAM", arrnam), Card("FREQ", layout.ref_freq), Card("RDATE", rdate),
            Card("GSTIA0", Float64(eo.gst_iat0)), Card("DEGPDY", Float64(eo.earth_rot_rate)),
            Card("UT1UTC", Float64(eo.ut1utc)), Card("POLARX", Float64(eo.polarx)),
            Card("POLARY", Float64(eo.polary)), Card("DATUTC", Float64(eo.datutc)),
            Card("TIMSYS", "UTC"), Card("FRAME", "ITRF"),
            Card("XYZHAND", string(eo.xyzhand)), Card("POLTYPE", string(eo.poltype)),
            Card("NUMORB", Int32(0)), Card("NO_IF", Int32(layout.nif)),
            Card("NOPCAL", Int32(0)), Card("FREQID", Int32(1)),
        ]
    )
    fq = HDU(
        Bintable, (;
            FRQSEL = Int32[1],
            var"IF FREQ" = _fq_column(Float64.(layout.if_freq)),
            var"CH WIDTH" = _fq_column(Float32.(layout.ch_width)),
            var"TOTAL BANDWIDTH" = _fq_column(Float32.(layout.total_bandwidth)),
            SIDEBAND = _fq_column(Int32.(layout.sideband)),
        ), Card[Card("EXTNAME", "AIPS FQ"), Card("NO_IF", Int32(layout.nif))]
    )
    nscan = length(nx.time)
    nx_hdu = HDU(
        Bintable, (;
            TIME = nx.time,
            var"TIME INTERVAL" = nx.interval,
            var"SOURCE ID" = ones(Int32, nscan),
            SUBARRAY = ones(Int32, nscan),
            var"FREQ ID" = ones(Int32, nscan),
            var"START VIS" = nx.start,
            var"END VIS" = nx.stop,
        ), Card[Card("EXTNAME", "AIPS NX"), Card("EXTVER", Int32(1))]
    )
    write(path, HDU[primary, an, fq, nx_hdu])
    return path
end
