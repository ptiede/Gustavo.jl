# ── Phase calibration from injected tones ────────────────────────────────────

"""
    PhaseCal(; time = PerScan(), delay_time = time)

Station phases measured by the injected phase-calibration tones that each
Measurement Set's `phase_calibration` sub-dataset records. Nothing is fit to the
visibilities. In each spectral window, a station's tones on one feed give a
straight line in phase across frequency — a phase at the window's center and a
delay — and that line, evaluated at every channel, is the instrumental phase
the solution divides out.

`time` sets how the tone measurements become gains:

- `PerScan()` averages, as complex numbers, the tone measurements whose
  intervals overlap a scan, each weighted by its overlap, and fits one line per
  scan.
- `PerIntegration()` fits a line to each tone measurement and interpolates its
  phase and delay linearly to every integration, holding the nearest
  measurement before the first and after the last.

`delay_time = GlobalTime()` instead fits one delay per station, feed and
spectral window for the whole track, from all of its measurements at once, and
measures only the phase per scan or integration along it. A delay from a few
tones in one measurement is noisy, while the instrumental delay is usually
stable over a track; a station with sparse tones then gains a steadier
correction.

Tones `Δν` apart fix a delay only modulo `1/Δν`. Each station, feed and
spectral window takes the delay within `1/(2Δν)` of its own track value — the
peak of the delay spectrum summed in power over all of its measurements — so
the delay stays on one branch across the track rather than jumping between
branches that fit the tones equally well.

A station a scan observes with no tone measurement overlapping it keeps unit
gain there, except that with `delay_time = GlobalTime()` it still carries its
track delay, with zero phase at the observation's reference frequency. Such
cells are listed in `sol.info` (`untoned_station`, `untoned_feed`,
`untoned_spw`, `untoned_scan`) and reported in a warning.
"""
Base.@kwdef struct PhaseCal{T <: Union{PerScan, PerIntegration}, D <: Union{GlobalTime, PerScan, PerIntegration}} <: AbstractSolver
    time::T = PerScan()
    delay_time::D = time
    function PhaseCal{T, D}(time, delay_time) where {T, D}
        delay_time isa GlobalTime || delay_time == time || throw(ArgumentError(
            "PhaseCal: delay_time must be GlobalTime() or the same as time ($time), got $delay_time"
        ))
        return new{T, D}(time, delay_time)
    end
end
PhaseCal(time, delay_time) = PhaseCal{typeof(time), typeof(delay_time)}(time, delay_time)
solution_name(::PhaseCal) = :phasecal

model_components(s::PhaseCal, spec) = GainModel(;
    phase = (;
        phasecal = GainComponent(ConstantTerm(); Ti = s.time, Frequency = PerSpectralWindow(), Feed = PerFeed()),
        phasecal_delay = GainComponent(Delay(); Ti = s.delay_time, Frequency = PerSpectralWindow(), Feed = PerFeed()),
    ),
)

# The station gain phase of a tone phase ψ, by the sign stated on the
# Conventions page.
_tone_gain_phase(ψ) = -ψ

function solve(s::PhaseCal, ctx::SolveContext)
    phase = only(Calibration.antenna_blocks(ctx.layout, ctx.θ, :phase, :phasecal))
    delay = only(Calibration.antenna_blocks(ctx.layout, ctx.θ, :phase, :phasecal_delay))
    windows = [_tone_window(ms, ctx.geom, phase.plan.fseg_id) for group in values(ctx.groups) for ms in values(group)]
    tracks = _tone_tracks(windows, s.delay_time isa GlobalTime)
    lines = s.time isa PerIntegration ? Dict(k => _track_lines(tr) for (k, tr) in tracks) : nothing
    column = Dict(a => i for (i, a) in pairs(phase.antennas))
    untoned = (; station = String[], feed = Int[], spw = String[], scan = String[])
    for w in windows, a in w.stations, p in 1:ctx.geom.nfeed
        key = (a, w.fseg, p)
        tr = get(tracks, key, nothing)
        overlapping = isnothing(tr) ? Int[] : _overlapping(tr, w.span)
        if isempty(overlapping)
            push!(untoned.station, ctx.geom.antennas[a]); push!(untoned.feed, p)
            push!(untoned.spw, w.spw); push!(untoned.scan, w.scan)
            continue
        end
        scan_line = s.time isa PerScan ? _scan_line(tr, overlapping, w.span) : nothing
        for ti in w.ti
            τ, φ = s.time isa PerScan ? scan_line : _interpolated_line(lines[key], ctx.geom.times[ti])
            delay.θ[1, p, w.fseg, delay.plan.tseg_id[ti], column[a]] = _tone_gain_phase(τ)
            phase.θ[1, p, w.fseg, phase.plan.tseg_id[ti], column[a]] = _tone_gain_phase(φ - 2π * τ * (tr.νc - ctx.geom.f0))
        end
    end
    isempty(untoned.station) || @warn "PhaseCal: $(length(untoned.station)) (station, feed, spectral window, scan) \
        cells have no tone measurement and keep unit gain, or only their track delay." stations = sort!(unique(untoned.station))
    return (;
        untoned_station = untoned.station, untoned_feed = untoned.feed, untoned_spw = untoned.spw,
        untoned_scan = untoned.scan,
        reference_station = [ctx.geom.antennas[k[1]] for k in keys(tracks)],
        reference_spw = [tracks[k].spw for k in keys(tracks)],
        reference_feed = [k[3] for k in keys(tracks)],
        reference_delay = [tracks[k].τref for k in keys(tracks)],
    )
end

const ToneMeasurement = @NamedTuple{station::Int, feed::Int, t::Float64, interval::Float64, z::Vector{ComplexF64}, ν::Vector{Float64}}

# One Measurement Set's tone measurements, read from its `phase_calibration`
# sub-dataset, with where the set sits in the solve: its frequency segment, its
# global time indices, its span in seconds and the stations its baselines name.
function _tone_window(ms, geom, fseg_id)
    win = GeometryWindow(geom, ms)
    t = XRadio.times(ms)
    ν = XRadio.frequencies(ms)
    half = XRadio.value(metadata(lookup(ms, Ti))[:integration_time]) / 2
    station = Dict(n => i for (i, n) in pairs(geom.antennas))
    measurements = ToneMeasurement[]
    if haskey(branches(ms), :phase_calibration)
        pc = read(branches(ms)[:phase_calibration])
        tones, freqs, interval = pc[:phase_cal], pc[:phase_cal_tone_frequency], pc[:phase_cal_interval]
        tpc = lookup(tones, XRadio.TimePhaseCal)
        receptors = eachindex(lookup(tones, XRadio.ReceptorLabel))
        for (ia, name) in pairs(lookup(tones, XRadio.AntennaName))
            a = get(station, String(name), nothing)
            isnothing(a) && continue
            for it in eachindex(tpc), p in receptors
                at = (XRadio.AntennaName(ia), XRadio.TimePhaseCal(it))
                cell = (at..., XRadio.ReceptorLabel(p))
                z, ν = parent(tones[cell]), parent(freqs[cell])
                keep = isfinite.(z) .& isfinite.(ν)
                any(keep) || continue
                push!(measurements, (; station = a, feed = p, t = tpc[it], interval = interval[at], z = z[keep], ν = ν[keep]))
            end
        end
    end
    return (;
        fseg = fseg_id[first(win.chan_idx)], spw = XRadio.spectralwindow(ms),
        scan = only(XRadio.scans(ms)), νc = sum(ν) / length(ν),
        ti = win.ti_idx, span = (first(t) - half, last(t) + half),
        stations = sort!(unique!(collect(Iterators.flatten(win.antennas)))), measurements,
    )
end

# Every (station, frequency segment, feed)'s tone measurements over the track,
# each once though neighbouring scans both list it, in time order, with the
# ambiguity period of its tone spacing, its reference delay and, with
# `global_delay`, its single delay.
function _tone_tracks(windows, global_delay)
    rows = Dict{NTuple{3, Int}, Dict{Float64, ToneMeasurement}}()
    window = Dict{NTuple{3, Int}, eltype(windows)}()
    for w in windows, m in w.measurements
        key = (m.station, w.fseg, m.feed)
        get!(Dict{Float64, ToneMeasurement}, rows, key)[m.t] = m
        window[key] = w
    end
    return Dict(key => _tone_track(window[key], sort!(collect(values(r)); by = m -> m.t), global_delay) for (key, r) in rows)
end

function _tone_track(w, measurements, global_delay)
    t = [m.t for m in measurements]
    interval = [m.interval for m in measurements]
    z = [m.z for m in measurements]
    ν = [m.ν for m in measurements]
    period = _ambiguity_period(ν)
    ngrid = _delay_grid_size(ν, period, w.spw)
    τref = _reference_delay(z, ν, w.νc, period, ngrid)
    τfix = global_delay ? _track_delay(z, ν, w.νc, τref, period) : nothing
    return (; w.spw, w.νc, t, interval, halfmax = maximum(interval) / 2, z, ν, period, ngrid, τref, τfix)
end

# The delay period over which the tones' phases repeat: 1/Δν for Δν the
# greatest common divisor of their spacings in whole Hz, or `Inf` for a single
# tone, which fixes no delay.
function _ambiguity_period(ν)
    f = sort!(unique!(round.(Int, reduce(vcat, ν))))
    length(f) < 2 && return Inf
    return 1 / gcd(diff(f))
end

# Delays searched across the ambiguity period: at least four per width `1/span`
# of the delay-spectrum peak of tones spanning `span`.
function _delay_grid_size(ν, period, spw)
    isfinite(period) || return 0
    f = reduce(vcat, ν)
    n = max(512, ceil(Int, 4 * period * (maximum(f) - minimum(f))))
    n <= 65536 || throw(ArgumentError(
        "PhaseCal: the tones of spectral window $spw repeat only every $period s; a delay search across that period needs $n points, over 65536"
    ))
    return n
end

_tone_sum(z, ν, νc, τ) = sum(z[k] * cis(-2π * τ * (ν[k] - νc)) for k in eachindex(z, ν))

# The track's delay: the peak, within one ambiguity period, of the delay spectrum
# summed in power over every measurement. With `R = Σ z z'` over the track's tone
# frequencies `f`, that power is `e' R e` for `e = cis(2πτ(f - νc))`.
function _reference_delay(z, ν, νc, period, n)
    isfinite(period) || return 0.0
    f = sort!(unique!(reduce(vcat, ν)))
    R = zeros(eltype(eltype(z)), length(f), length(f))
    for (zi, νi) in zip(z, ν)
        k = searchsortedfirst.(Ref(f), νi)
        for b in eachindex(k, zi), a in eachindex(k, zi)
            R[k[a], k[b]] += zi[a] * conj(zi[b])
        end
    end
    return _peak_delay(τ -> (e = cis.(2π * τ .* (f .- νc)); real(dot(e, R, e))), 0.0, period, n)
end

# The delay maximizing `power` within half a period of `center`: a grid of
# `n` delays across the period, then a parabola through the best and its
# neighbours.
function _peak_delay(power, center, period, n)
    step = period / n
    grid = center - period / 2 .+ (0:(n - 1)) .* step
    y = power.(grid)
    j = argmax(y)
    return grid[j] + Fring._quad_offset(power(grid[j] - step), y[j], power(grid[j] + step)) * step
end

# The weighted least-squares sums `(num, den)` of a line through the residual
# phases of tones `z` about delay `τ`, accumulated in one pass (West 1979);
# `num / (2π den)` is the delay step.
function _slope_moments(z, ν, νc, τ)
    c = _tone_sum(z, ν, νc, τ)
    W = x̄ = r̄ = num = den = 0.0
    for k in eachindex(z, ν)
        w = abs2(z[k])
        w > 0 || continue
        x = ν[k] - νc
        r = angle(z[k] * cis(-2π * τ * x) * conj(c))
        W += w
        dx = x - x̄
        x̄ += w / W * dx
        r̄ += w / W * (r - r̄)
        num += w * dx * (r - r̄)
        den += w * dx * (x - x̄)
    end
    return num, den
end

# The line through one set of tones of track `tr`: the delay on the track's
# branch, or the track's single delay where it has one, and the phase at the
# window's center along it.
function _tone_line(z, ν, tr)
    τ = _line_delay(z, ν, tr)
    return τ, angle(_tone_sum(z, ν, tr.νc, τ))
end

# The grid peak is refined by one least-squares step on the tones' residual
# phases, which is exact for tones on a line.
function _line_delay(z, ν, tr)
    isnothing(tr.τfix) || return tr.τfix
    isfinite(tr.period) || return 0.0
    τ = _peak_delay(τ -> abs2(_tone_sum(z, ν, tr.νc, τ)), tr.τref, tr.period, tr.ngrid)
    num, den = _slope_moments(z, ν, tr.νc, τ)
    return den > 0 ? τ + num / (2π * den) : τ
end

# A track's single delay: its reference delay refined by one least-squares step
# on every measurement's residual tone phases, each measurement keeping its own
# phase. Exact for tones on lines of one slope.
function _track_delay(z, ν, νc, τref, period)
    isfinite(period) || return 0.0
    num = den = 0.0
    for (zi, νi) in zip(z, ν)
        length(zi) < 2 && continue
        n, d = _slope_moments(zi, νi, νc, τref)
        num += n
        den += d
    end
    return den > 0 ? τref + num / (2π * den) : τref
end

# Seconds of measurement `i` falling inside `span`.
_overlap(tr, i, span) = max(0.0, min(tr.t[i] + tr.interval[i] / 2, span[2]) - max(tr.t[i] - tr.interval[i] / 2, span[1]))

# The measurements of a track that overlap `span`.
function _overlapping(tr, span)
    candidates = searchsortedfirst(tr.t, span[1] - tr.halfmax):searchsortedlast(tr.t, span[2] + tr.halfmax)
    return filter(i -> _overlap(tr, i, span) > 0, candidates)
end

# One line for a scan, through the tones averaged over the `overlapping`
# measurements, each weighted by its overlap with `span`.
function _scan_line(tr, overlapping, span)
    sums = Dict{Float64, eltype(eltype(tr.z))}()
    for i in overlapping
        w = _overlap(tr, i, span)
        for (zk, νk) in zip(tr.z[i], tr.ν[i])
            sums[νk] = get(sums, νk, zero(zk)) + w * zk
        end
    end
    return _tone_line(collect(values(sums)), collect(keys(sums)), tr)
end

# A line through each measurement of a track, with the phase unwrapped along
# time so that interpolation between measurements follows the shorter way round.
function _track_lines(tr)
    lines = [_tone_line(tr.z[i], tr.ν[i], tr) for i in eachindex(tr.z, tr.ν)]
    return (; tr.t, τ = first.(lines), φ = Fring.unwrap_phase_track(last.(lines)))
end

# A track's phase and delay at epoch `t`, linear between measurements and held
# beyond the first and last.
function _interpolated_line(line, t)
    i = searchsortedlast(line.t, t)
    i < 1 && return line.τ[1], line.φ[1]
    i >= length(line.t) && return line.τ[end], line.φ[end]
    w = (t - line.t[i]) / (line.t[i + 1] - line.t[i])
    return (1 - w) * line.τ[i] + w * line.τ[i + 1], (1 - w) * line.φ[i] + w * line.φ[i + 1]
end
