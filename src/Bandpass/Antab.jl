using Dates

"""
    AntabGainCurve(dpfu, poly)

Per-station, per-feed gain calibration carried by an ANTAB `GAIN` record.

- `dpfu::NTuple{2, Float64}` — degrees-per-flux-unit (K/Jy) for the two feeds
  `(R/X, L/Y)`. When the file specifies a single value, both entries are set
  to that value.
- `poly::Vector{Float64}` — elevation-gain polynomial coefficients, evaluated
  as `g_E = poly[1] + poly[2] * El + poly[3] * El^2 + …` with `El` in
  degrees. `poly = [1.0]` indicates a flat curve.
"""
struct AntabGainCurve
    dpfu::NTuple{2, Float64}
    poly::Vector{Float64}
end

"""
    AntabTsysSeries(times, columns, values)

Time-series of `T_sys^eff` (or SEFD, for phased arrays) parsed from one
station's `TSYS` block.

- `times::Vector{DateTime}` — UTC timestamps reconstructed from the file's
  `DOY HH:MM:SS` (or ALMA's `DOY HH:MM.fffff`) records.
- `columns::Vector{Tuple{Int, Symbol}}` — for each value column, the
  channel index (1-based) and feed designator: `:R`, `:L`, or `:both`.
  Aggregate (one value per polarization, broadcast to all channels) is
  encoded as `channel = 0`.
- `values::Matrix{Float64}` — `(ntimes, ncolumns)`.
"""
struct AntabTsysSeries
    times::Vector{DateTime}
    columns::Vector{Tuple{Int, Symbol}}
    values::Matrix{Float64}
end

"""
    AntabStation(name, gain, tsys, nchannels)

Calibration record for one station.

`nchannels` is `0` when the `INDEX` declares aggregate (per-polarization,
broadcast to every channel) Tsys columns, otherwise it is the number of
distinct channel indices declared in `INDEX` (e.g. 32 for ALMA's
per-channel SEFDs).
"""
struct AntabStation
    name::String
    gain::AntabGainCurve
    tsys::AntabTsysSeries
    nchannels::Int
end

"""
    AntabCalibration(source_path, track_label, year, stations)

Top-level inspectable container returned by [`load_antab`](@ref). Holds the
station calibration records keyed by 2-character VEX station code (e.g.
`"AA"`, `"NN"`) plus provenance fields used downstream by
[`apply_calibration`](@ref) when aligning Tsys timestamps to a `UVSet`'s
`obs_time` axis.
"""
struct AntabCalibration
    source_path::String
    track_label::String
    year::Int
    stations::Dict{String, AntabStation}
end

stations(a::AntabCalibration) = a.stations
Base.haskey(a::AntabCalibration, name::AbstractString) = haskey(a.stations, String(name))
Base.getindex(a::AntabCalibration, name::AbstractString) = a.stations[String(name)]
Base.keys(a::AntabCalibration) = keys(a.stations)
Base.length(a::AntabCalibration) = length(a.stations)

# ─────────────────────────── Parsing ────────────────────────────────

# Year inference from filenames like "e22a26_b3_proc.AN" → 2022. We look
# for the first occurrence of `e<DD>` where DD is a 2-digit year suffix.
function _year_from_filename(path::AbstractString)
    m = match(r"e(\d{2})[a-z]\d", basename(path))
    m === nothing && return nothing
    yy = parse(Int, m.captures[1])
    return yy >= 70 ? 1900 + yy : 2000 + yy
end

function _track_label_from_filename(path::AbstractString)
    m = match(r"(e\d{2}[a-z]\d{2}(?:_b\d)?)", basename(path))
    return m === nothing ? splitext(basename(path))[1] : m.captures[1]
end

# Parse a single GAIN line: `GAIN <ANT> ELEV DPFU = ... POLY = ... /`.
function _parse_gain_line(line::AbstractString)
    body = strip(line)
    startswith(body, "GAIN") || error("_parse_gain_line: not a GAIN record: $line")
    body = replace(body, '/' => "")
    tokens = split(body)
    length(tokens) >= 2 || error("_parse_gain_line: malformed GAIN line: $line")
    ant = String(tokens[2])
    rest = join(tokens[3:end], " ")

    # Split on `DPFU` and `POLY` keywords (case-sensitive in real files).
    dpfu_match = match(r"DPFU\s*=\s*([^A-Za-z/]+?)(?=\s+POLY|\s*$)"i, rest)
    poly_match = match(r"POLY\s*=\s*([^A-Za-z/]+?)(?=\s*$)"i, rest)
    dpfu_match === nothing && error("_parse_gain_line: missing DPFU on $line")
    poly_match === nothing && error("_parse_gain_line: missing POLY on $line")

    dpfu_vals = _parse_float_list(dpfu_match.captures[1])
    poly_vals = _parse_float_list(poly_match.captures[1])

    dpfu_t = if length(dpfu_vals) == 1
        (dpfu_vals[1], dpfu_vals[1])
    elseif length(dpfu_vals) >= 2
        (dpfu_vals[1], dpfu_vals[2])
    else
        error("_parse_gain_line: empty DPFU on $line")
    end
    return ant, AntabGainCurve(dpfu_t, isempty(poly_vals) ? [1.0] : poly_vals)
end

function _parse_float_list(s::AbstractString)
    out = Float64[]
    for piece in split(s, [',', ' ', '\t']; keepempty = false)
        isempty(strip(piece)) && continue
        push!(out, parse(Float64, strip(piece)))
    end
    return out
end

# Parse the INDEX clause and return a Vector{Tuple{Int, Symbol}} where each
# entry is (channel, pol). channel == 0 means "broadcast across all
# channels" (the aggregate convention used by `'R1:32'`).
function _parse_index_clause(clause::AbstractString)
    body = strip(clause)
    out = Tuple{Int, Symbol}[]
    # Split on commas at the top level. Quoted items are e.g. 'R1:32' or
    # 'L1|R1' or 'R1'. We accept both straight and curly quotes, since
    # some files use ' but tooling may emit it differently.
    pieces = split(body, ',')
    for raw in pieces
        s = strip(raw)
        isempty(s) && continue
        s = strip(s, ['\'', '"'])
        s = strip(s)
        isempty(s) && continue
        push!(out, _parse_index_token(s))
    end
    return out
end

function _parse_index_token(tok::AbstractString)
    # 'R1' / 'L17'
    m1 = match(r"^([RLXY])(\d+)$"i, tok)
    if m1 !== nothing
        pol = _pol_from_letter(m1.captures[1][1])
        ch = parse(Int, m1.captures[2])
        return (ch, pol)
    end
    # 'R1:32' / 'L1:32' — aggregate column applied to all channels in range
    m2 = match(r"^([RLXY])\d+:\d+$"i, tok)
    if m2 !== nothing
        pol = _pol_from_letter(m2.captures[1][1])
        return (0, pol)
    end
    # 'L1|R1' (ALMA): channel 1, both polarizations share one value
    m3 = match(r"^([RLXY])(\d+)\s*\|\s*([RLXY])(\d+)$"i, tok)
    if m3 !== nothing
        ch1 = parse(Int, m3.captures[2])
        ch2 = parse(Int, m3.captures[4])
        ch1 == ch2 || error("_parse_index_token: paired channels must match in $tok")
        return (ch1, :both)
    end
    return error("_parse_index_token: unrecognized INDEX entry $tok")
end

function _pol_from_letter(c::Char)
    cu = uppercase(c)
    cu in ('R', 'X') && return :R
    cu in ('L', 'Y') && return :L
    return error("_pol_from_letter: unsupported feed '$c'")
end

# Convert "DOY HH:MM:SS.SS" or "DOY HH:MM.fffff" into a UTC DateTime.
# Sub-millisecond precision is rounded; rounding up at e.g. 59.9999 s
# correctly carries into the next minute via Millisecond addition.
function _parse_antab_time(year::Int, doy::Int, timestr::AbstractString)
    base = DateTime(Date(year, 1, 1) + Day(doy - 1))
    ncolons = count(==(':'), timestr)
    if ncolons == 2
        hh, mm, ssf = split(timestr, ':')
        h = parse(Int, hh)
        m = parse(Int, mm)
        sf = parse(Float64, ssf)
        ms_total = round(Int, ((h * 60 + m) * 60 + sf) * 1000)
        return base + Millisecond(ms_total)
    elseif ncolons == 1
        # ALMA: HH:MM.fffff (decimal minutes).
        hh, mmf = split(timestr, ':')
        h = parse(Int, hh)
        mtot = parse(Float64, mmf)
        ms_total = round(Int, (h * 60 + mtot) * 60_000)
        return base + Millisecond(ms_total)
    else
        return error("_parse_antab_time: unsupported time format \"$timestr\"")
    end
end

# Strip an inline `!`-introduced comment.
_strip_comment(line::AbstractString) =
    let i = findfirst('!', line)
        i === nothing ? line : line[1:prevind(line, i)]
    end

"""
    load_antab(path::AbstractString; year::Union{Nothing, Int} = nothing) -> AntabCalibration

Parse one EHT-style ANTAB file (e.g. `e22a26_b3_proc.AN`). Returns an
[`AntabCalibration`](@ref) keyed by 2-character VEX station code.

The `year` is inferred from the filename when omitted (`e22*` → 2022,
`e21*` → 2021, etc.). Pass `year=2022` (or similar) explicitly if the
filename does not match the convention.
"""
function load_antab(path::AbstractString; year::Union{Nothing, Int} = nothing)
    isfile(path) || error("load_antab: file does not exist: $path")
    yy = year === nothing ? _year_from_filename(path) : year
    yy === nothing && error(
        "load_antab: cannot infer observation year from filename \"$(basename(path))\"; " *
            "pass `year=YYYY` explicitly.",
    )

    raw_lines = readlines(path)
    # Strip `!` comments and keep raw layout so block detection (lone `/`) works.
    lines = String[String(rstrip(_strip_comment(l))) for l in raw_lines]

    gains = Dict{String, AntabGainCurve}()
    tsys_blocks = Dict{String, AntabTsysSeries}()
    nchans_map = Dict{String, Int}()

    i = 1
    n = length(lines)
    while i <= n
        line = strip(lines[i])
        if isempty(line)
            i += 1
            continue
        end
        upper = uppercase(line)
        if startswith(upper, "GAIN")
            ant, curve = _parse_gain_line(lines[i])
            gains[ant] = curve
            i += 1
        elseif startswith(upper, "TSYS")
            j_after, ant, ts, nch = _parse_tsys_block(lines, i, yy)
            tsys_blocks[ant] = ts
            nchans_map[ant] = nch
            i = j_after
        else
            i += 1
        end
    end

    stations_d = Dict{String, AntabStation}()
    for (ant, ts) in tsys_blocks
        haskey(gains, ant) || error("load_antab: TSYS block for $ant has no matching GAIN record in $path")
        stations_d[ant] = AntabStation(ant, gains[ant], ts, nchans_map[ant])
    end

    return AntabCalibration(
        abspath(path), _track_label_from_filename(path), yy, stations_d,
    )
end

# Parse a TSYS block starting at `lines[i]`. Returns
# `(next_index, antenna_name, AntabTsysSeries, nchannels_distinct)`.
#
# Two layouts in EHT-2022 ANTAB files:
#   (a) `TSYS <ANT> kw=val kw=val INDEX = ... /` on one line, then data, then `/`.
#   (b) `TSYS <ANT> kw=val kw=val` on one line, `INDEX = ... ` on the next,
#       then `/` on its own line, then data, then `/`.
# We collect header text (lines until the first standalone `/` or until the
# `/` that splits header from data appears inline), parse INDEX from that
# accumulated text, then read data rows until the closing `/`.
function _parse_tsys_block(lines::Vector{String}, i::Int, year_int::Int)
    header_io = IOBuffer()
    j = i
    n = length(lines)
    saw_inline_terminator = false
    # First line carries `TSYS <ANT> ...`. Capture antenna and start the
    # header buffer with the post-TSYS content.
    first_line = strip(lines[j])
    tokens = split(first_line)
    length(tokens) >= 2 || error("_parse_tsys_block: malformed TSYS line: $first_line")
    ant = String(tokens[2])
    header_first = join(tokens[3:end], " ")
    if occursin('/', header_first)
        # Inline `/` separates header from data on the same line.
        idx = findfirst('/', header_first)
        write(header_io, header_first[1:prevind(header_first, idx)])
        saw_inline_terminator = true
    else
        write(header_io, header_first)
    end
    j += 1

    # If the inline `/` was not present, append subsequent lines until we
    # hit one whose stripped contents is exactly `/` (block-terminator),
    # OR a line that contains `/` after the INDEX clause continues
    # (continuation).
    while !saw_inline_terminator && j <= n
        s = strip(lines[j])
        if isempty(s)
            j += 1
            continue
        end
        if s == "/"
            saw_inline_terminator = true
            j += 1
            break
        end
        if occursin('/', s)
            idx = findfirst('/', s)
            write(header_io, " ", s[1:prevind(s, idx)])
            saw_inline_terminator = true
            j += 1
            break
        end
        write(header_io, " ", s)
        j += 1
    end

    header_text = String(take!(header_io))
    # Locate INDEX clause within header_text (case-insensitive).
    m = match(r"INDEX\s*=\s*(.+)$"is, header_text)
    m === nothing && error("_parse_tsys_block: missing INDEX clause for $ant")
    index_clause = String(m.captures[1])
    columns = _parse_index_clause(index_clause)
    isempty(columns) && error("_parse_tsys_block: no columns parsed from INDEX for $ant")
    nch_distinct = length(unique(c[1] for c in columns if c[1] != 0))

    # Read data rows until the closing `/`.
    times = DateTime[]
    rows = Vector{Float64}[]
    while j <= n
        s = strip(lines[j])
        if isempty(s)
            j += 1
            continue
        end
        if s == "/" || startswith(s, "/")
            j += 1
            break
        end
        toks = split(s)
        # Expected: doy time v1 v2 ... vN, with N = length(columns).
        length(toks) >= 2 + length(columns) || error(
            "_parse_tsys_block: row for $ant has $(length(toks)) tokens; " *
                "expected ≥ 2 + $(length(columns)) on line: $s",
        )
        doy = parse(Int, toks[1])
        t = _parse_antab_time(year_int, doy, toks[2])
        vals = Float64[parse(Float64, toks[2 + k]) for k in 1:length(columns)]
        push!(times, t)
        push!(rows, vals)
        j += 1
    end

    if isempty(rows)
        values = Matrix{Float64}(undef, 0, length(columns))
    else
        values = Matrix{Float64}(undef, length(rows), length(columns))
        @inbounds for r in 1:length(rows), c in 1:length(columns)
            values[r, c] = rows[r][c]
        end
    end
    return j, ant, AntabTsysSeries(times, columns, values), nch_distinct
end

# ─────────────────────────── Lookup helpers ─────────────────────────

"""
    tsys_at(station::AntabStation, t::DateTime, channel::Int, pol::Symbol) -> Float64

Linear-in-time interpolation of the parsed Tsys series. `pol` is `:R` or
`:L`; aggregate INDEX rows (`'R1:32'`-style) are returned for either pol
when no per-pol breakdown exists. Per-channel ALMA blocks (`'L1|R1'`-style)
broadcast a single value to both pols. Returns `NaN` when the time is
outside the available samples or when no matching column exists.
"""
function tsys_at(station::AntabStation, t::DateTime, channel::Int, pol::Symbol)
    pol in (:R, :L) || error("tsys_at: pol must be :R or :L (got $pol)")
    col_idx = _select_column(station.tsys.columns, channel, pol)
    col_idx == 0 && return NaN
    return _interp_in_time(station.tsys.times, view(station.tsys.values, :, col_idx), t)
end

# Resolve a (channel, pol) request to an INDEX column index. Channel == 0
# never matches a request — callers always ask for a real channel; we do
# the broadcast lookup here.
function _select_column(columns::Vector{Tuple{Int, Symbol}}, channel::Int, pol::Symbol)
    # Prefer per-channel + matching pol.
    for (i, (ch, p)) in pairs(columns)
        ch == channel && p == pol && return i
    end
    # Per-channel + :both (ALMA shared-pol case).
    for (i, (ch, p)) in pairs(columns)
        ch == channel && p === :both && return i
    end
    # Aggregate column for the requested pol (channel encoded as 0).
    for (i, (ch, p)) in pairs(columns)
        ch == 0 && p == pol && return i
    end
    # Aggregate :both fallback (rare; keeps single-pol single-dish blocks safe).
    for (i, (ch, p)) in pairs(columns)
        ch == 0 && p === :both && return i
    end
    return 0
end

function _interp_in_time(times::Vector{DateTime}, values::AbstractVector{<:Real}, t::DateTime)
    n = length(times)
    n == 0 && return NaN
    n == 1 && return Float64(values[1])
    (t < times[1] || t > times[end]) && return NaN
    # Binary search for the bracketing pair.
    lo, hi = 1, n
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if times[mid] <= t
            lo = mid
        else
            hi = mid
        end
    end
    t0, t1 = times[lo], times[hi]
    if t0 == t1
        return Float64(values[lo])
    end
    span = (t1 - t0).value           # ms
    frac = (t - t0).value / span
    return Float64(values[lo]) + frac * (Float64(values[hi]) - Float64(values[lo]))
end

"""
    tsys_in_window(station::AntabStation, t_lo::DateTime, t_hi::DateTime,
                   channel::Int, pol::Symbol) -> Float64

Mean of the parsed Tsys series over a UTC window. Returns the simple
mean of all rows whose timestamp lies in `[t_lo, t_hi]` (inclusive)
with finite values for the requested column. Returns `NaN` when no
row falls in the window.

Use this lookup (rather than [`tsys_at`](@ref)'s linear time interpolation)
when calibrating per-scan: the processed-ANTAB time series mixes rows
from every scan in a track — including slews and other-source
calibrators — so interpolating across rows can pull a normal-scan Tsys
toward an adjacent unphysical value. Restricting to rows inside the
target scan's time window avoids that contamination by construction.
"""
function tsys_in_window(
        station::AntabStation, t_lo::DateTime, t_hi::DateTime,
        channel::Int, pol::Symbol,
    )
    pol in (:R, :L) || error("tsys_in_window: pol must be :R or :L (got $pol)")
    col = _select_column(station.tsys.columns, channel, pol)
    col == 0 && return NaN
    times = station.tsys.times
    n = length(times)
    n == 0 && return NaN
    acc = 0.0
    cnt = 0
    @inbounds for i in 1:n
        if t_lo <= times[i] <= t_hi
            v = station.tsys.values[i, col]
            isfinite(v) || continue
            acc += v
            cnt += 1
        end
    end
    return cnt == 0 ? NaN : acc / cnt
end

"""
    elevation_gain(curve::AntabGainCurve, elevation_deg::Real) -> Float64

Evaluate the elevation gain polynomial at `elevation_deg` (degrees).
"""
function elevation_gain(curve::AntabGainCurve, elevation_deg::Real)
    p = curve.poly
    isempty(p) && return 1.0
    El = Float64(elevation_deg)
    # Horner from highest order down.
    acc = 0.0
    @inbounds for k in length(p):-1:1
        acc = acc * El + p[k]
    end
    return acc
end

# ─────────────────────────── Display ─────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", a::AntabCalibration)
    println(io, "AntabCalibration(track=$(a.track_label), year=$(a.year), $(length(a.stations)) stations)")
    println(io, "  source: $(a.source_path)")
    if !isempty(a.stations)
        names = sort!(collect(keys(a.stations)))
        println(io, "  stations:")
        for nm in names
            s = a.stations[nm]
            println(io, "    ", _station_summary(s))
        end
    end
    return nothing
end
Base.show(io::IO, a::AntabCalibration) =
    print(io, "AntabCalibration($(a.track_label), $(length(a.stations)) stations)")

function Base.show(io::IO, ::MIME"text/plain", s::AntabStation)
    return print(io, _station_summary(s))
end
Base.show(io::IO, s::AntabStation) = print(io, "AntabStation($(s.name))")

function _station_summary(s::AntabStation)
    nt = length(s.tsys.times)
    chmode = s.nchannels == 0 ? "aggregate" : "$(s.nchannels) channels"
    trange = if nt == 0
        "no samples"
    else
        string(s.tsys.times[1], " → ", s.tsys.times[end])
    end
    poly_str = join(string.(s.gain.poly), ",")
    dpfu_str = string(s.gain.dpfu[1], "/", s.gain.dpfu[2])
    return "$(s.name): DPFU=$(dpfu_str)  POLY=[$(poly_str)]  TSYS=$(chmode)  $(nt) rows  $(trange)"
end
