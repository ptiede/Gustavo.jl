# Synthetic multi-band ProcessingSet with known injected per-(station, feed)
# fringe parameters (delay/rate/phase + atmospheric screen, optional
# bandpass/dTEC), built on `XRadio.Testing`.
using Gustavo
using Random
using StableRNGs: StableRNG
using DimensionalData
using DimensionalData: dims, lookup
using Dates: Date, DateTime, datetime2unix
using OrderedCollections: OrderedDict
import XRadio
using XRadio: Testing

_bp_at(bp::AbstractArray{<:Any, 3}, a, f, gc, s) = bp[a, f, gc]
_bp_at(bp::AbstractArray{<:Any, 4}, a, f, gc, s) = bp[a, f, gc, s]

"""
    _build_fringe_ps(; kw...) -> (ps, truth)

One Measurement Set per (scan, spectral window), holding
`V = A · exp(i·[Δφ + 2π·Δτ·(f − f0) + 2π·Δṙ·(t − t0) + Δscreen + Δbandpass + Δdtec])`
on each baseline and product, with `Δx = x[a, fa] − x[b, fb]` and the feeds
`(fa, fb)` from [`Gustavo.feed_pairs`](@ref). `f0` is the mean channel
frequency across windows and `t0` the first sample time.

The random draws are made in a fixed order from `StableRNG(seed)`, so the same
keywords always give the same data.

# Keywords
- `nant`, `nspw`, `nchan`, `ntime`, `nscans`: the array and sampling sizes;
  samples are 30 s apart and scans start `scan_gap` seconds apart, by default
  two samples after the previous one ends.
- `noise`: σ per visibility sample, against a signal amplitude of 2.5.
- `weight`: every sample's weight; by default the inverse variance per real
  component, `2 / noise^2`, or `1e3` without noise.
- `polarizations`: the stored correlation labels.
- `ref_freq`, `chan_bw`, `spw_sep`, or `spw_origins` for explicit window starts.
- `bandpass`, `amp_bandpass`: per-channel phase (rad) and log-amplitude, sized
  `(nant, 2, nspw·nchan)`, or `(nant, 2, nspw·nchan, nscans)` to change between
  scans.
- `dtec`: per-station TEC (TECU), feed-common.
- `feed_common`: tie delay and phase across feeds.
- `station_rate`: per-station feed-common rate (Hz) replacing the draw;
  `rel_rate`: a feed-2 − feed-1 rate offset (Hz).
- `station_positions`: per-station xyz (m).
- `omit_station`: a station index in the antenna dataset but on no baseline.
- `receptor_order`: station name => its receptors, in place of `R, L`.
- `station_gains = false`: zero delay, rate, phase and screen, leaving the
  bandpass as the only station gain.
- `eltype`: the visibilities' element type.
"""
function _build_fringe_ps(;
        nant = 4, nspw = 2, nchan = 8, ntime = 12, nscans = 1,
        scan_gap = nothing, noise = nothing,
        polarizations = ["RR", "RL", "LR", "LL"],
        ref_freq = 230.0e9, chan_bw = 2.0e6, spw_sep = 1.0e8,
        seed = 1234,
        bandpass = nothing, amp_bandpass = nothing, dtec = nothing,
        feed_common = false, station_rate = nothing, rel_rate = nothing,
        spw_origins = nothing, station_positions = nothing, omit_station = nothing,
        receptor_order = Dict{String, Vector{String}}(),
        station_gains = true, eltype = ComplexF32,
        weight = isnothing(noise) ? 1.0e3 : 2 / noise^2,
    )
    rng = StableRNG(seed)
    names = ["A$i" for i in 1:nant]
    antenna_xds = Testing.antenna(
        names;
        positions = station_positions === nothing ?
            [1.0e4 * c * i for c in 1:3, i in 1:nant] :
            reduce(hcat, (Float64.(p) for p in station_positions)),
    )
    if !isempty(receptor_order)
        types = copy(parent(antenna_xds[:polarization_type]))
        for (n, r) in receptor_order
            types[:, findfirst(==(n), names)] = r
        end
        antenna_xds[:polarization_type] = rebuild(antenna_xds[:polarization_type], types)
    end
    drop = omit_station === nothing ? String[] : [names[omit_station]]

    integration = 30.0
    t_epoch = datetime2unix(DateTime(Date("2021-03-04")))
    times = collect(t_epoch .+ (0:(ntime - 1)) .* integration)
    scan_span = scan_gap === nothing ? (ntime + 2) * integration : Float64(scan_gap)
    starts = [
        spw_origins === nothing ? ref_freq + (b - 1) * spw_sep : Float64(spw_origins[b])
            for b in 1:nspw
    ]
    freqs = [s .+ (0:(nchan - 1)) .* chan_bw for s in starts]
    f0 = let all = sort!(unique!(reduce(vcat, freqs)))
        sum(all) / length(all)
    end
    t0 = first(times)

    delay = zeros(nant, 2)
    rate = zeros(nant, 2)
    phi = zeros(nant, 2)
    if station_gains
        for a in 2:nant
            rc = (rand(rng) - 0.5) * 2.0e-3
            for f in 1:2
                delay[a, f] = (rand(rng) - 0.5) * 2.0e-9
                rate[a, f] = rc
                phi[a, f] = (rand(rng) - 0.5) * 2.0
            end
        end
    end
    station_rate === nothing || (rate .= station_rate)
    rel_rate === nothing || (rate[:, 2] .= rate[:, 1] .+ rel_rate)
    if feed_common
        delay[:, 2] .= delay[:, 1]
        phi[:, 2] .= phi[:, 1]
    end

    # Feed-common, since the atmosphere is not birefringent: a per-station
    # offset plus a slow track, drawn afresh for each scan.
    screen = zeros(nant, 2, ntime, nscans)
    for s in 1:nscans, a in (station_gains ? (2:nant) : 1:0)
        base = (rand(rng) - 0.5) * 1.0
        for ti in 1:ntime
            screen[a, :, ti, s] .= base + 0.2 * sin(0.5 * ti + a)
        end
    end

    A0 = 2.5
    index = Dict(n => i for (i, n) in enumerate(names))
    sets = OrderedDict{Symbol, XRadio.MeasurementSet}()
    local feeds, bl_pairs
    for s in 1:nscans, b in 1:nspw
        ms = Testing.measurement_set(;
            antennas = names, drop, antenna_xds,
            times = times .+ (s - 1) * scan_span, frequencies = freqs[b],
            polarizations, integration_time = integration, channel_width = chan_bw,
            reference_frequency = ref_freq, spectral_window = "band_$b",
            scan = string(s), field = "SRC1", source = "SRC1", direction = (1.234, -0.56),
            eltype,
        )
        feeds = Gustavo.feed_pairs(ms)
        ant1 = [index[n] for n in ms[:baseline_antenna1_name]]
        ant2 = [index[n] for n in ms[:baseline_antenna2_name]]
        bl_pairs = collect(zip(ant1, ant2))
        tsec = collect(lookup(ms, XRadio.Ti))
        vis = similar(parent(ms[:visibility]))
        for p in axes(vis, 1), bl in axes(vis, 3), ti in axes(vis, 4), c in axes(vis, 2)
            a, bb = ant1[bl], ant2[bl]
            fa, fb = feeds[p, bl]
            f = freqs[b][c]
            gc = (b - 1) * nchan + c
            dbp = bandpass === nothing ? 0.0 :
                _bp_at(bandpass, a, fa, gc, s) - _bp_at(bandpass, bb, fb, gc, s)
            dla = amp_bandpass === nothing ? 0.0 :
                _bp_at(amp_bandpass, a, fa, gc, s) + _bp_at(amp_bandpass, bb, fb, gc, s)
            ddt = dtec === nothing ? 0.0 :
                Gustavo.Calibration.DISPERSION_K * (dtec[a] - dtec[bb]) * (1.0 / f0 - 1.0 / f)
            ph = (phi[a, fa] - phi[bb, fb]) +
                2π * (delay[a, fa] - delay[bb, fb]) * (f - f0) +
                2π * (rate[a, fa] - rate[bb, fb]) * (tsec[ti] - t0) +
                (screen[a, fa, ti, s] - screen[bb, fb, ti, s]) + dbp + ddt
            z = A0 * exp(dla) * cis(ph)
            noise === nothing || (z += noise * complex(randn(rng), randn(rng)) / sqrt(2))
            vis[p, c, bl, ti] = z
        end
        ms[:visibility] = rebuild(ms[:visibility], vis)
        ms[:weight] = rebuild(ms[:weight], fill!(similar(parent(ms[:weight])), weight))
        sets[Symbol("synth_fringe_", s, "_", b)] = ms
    end

    return XRadio.ProcessingSet(sets), (;
        delay, rate, phi,
        screen = nscans == 1 ? reshape(screen, nant, 2, ntime) : screen,
        bandpass, amp_bandpass, dtec, f0, t0_sec = t0, bl_pairs, polarizations, feeds,
    )
end

# A copy of `data` with `sol` divided out: a degenerate gain leaves the sample,
# and no flags are applied.
_precal(sol, data) = calibrate(sol, data; flag_bad = false, apply_flags = false)

# Each step fit in turn on `data`, with the solutions of the steps before it
# divided out as `_precal` divides them.
function _fit_chain(steps, data; kwargs...)
    sols = CalibrationSolution[]
    for st in steps
        isempty(sols) || (data = _precal(last(sols), data))
        push!(sols, fit(st, data; kwargs...))
    end
    return sols
end

# A copy of `data` with each solution of `sols` divided out in turn.
function _calibrate_chain(sols, data; kwargs...)
    out = calibrate(first(sols), data; kwargs...)
    foreach(s -> calibrate!(s, out; kwargs...), sols[2:end])
    return out
end

# One solution holding the components and diagnostics of every solution in `sols`.
_combined(sols) = CalibrationSolution(
    first(sols).geom, reduce(vcat, [s.components for s in sols]),
    merge((s.steps for s in sols)...), first(sols).info,
)
