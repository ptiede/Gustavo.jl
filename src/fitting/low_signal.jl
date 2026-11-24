"""
    signal_information(data) -> DimArray

The information the visibilities carry on each feed's gain, per integration
and relative to its largest value in the scan, over
`(Scan, AntennaName, Feed, Ti)`. `data` is a processing set in memory,
corrected by a per-integration gain solution such as an [`Adhoc`](@ref) fit.
The information of feed `f` of station `a` at time `t` is

    I_af(t) = Σ_b Σ w_ab(t) |S_ab|²,

summed over its baselines `b`, the products relating feed `f` of `a`, and
channels, with `w` the corrected weight (the inverse variance of each real
component) and `S_ab` the weighted mean of the corrected visibilities of that
baseline, product and spectral window over the scan. It stays near 1 while a
feed observes the source and falls where its signal is lost, as when the
station is off source. A feed without data at a time is `NaN`.

[`flag_low_signal!`](@ref) flags where it is low.
"""
function signal_information(data::XRadio.ProcessingSet)
    geom = Calibration.DataGeometry(data)
    members = collect(values(data))
    scans = unique(only(XRadio.scans(ms)) for ms in members)
    times = sort!(unique(t for ms in members for t in Float64.(collect(XRadio.times(ms)))))
    scan_i = Dict(s => i for (i, s) in pairs(scans))
    ant_i = Dict(a => i for (i, a) in pairs(geom.antennas))
    time_i = Dict(t => i for (i, t) in pairs(times))
    info = zeros(length(scans), length(geom.antennas), geom.nfeed, length(times))
    seen = falses(size(info))
    for ms in members
        ts = [time_i[t] for t in Float64.(collect(XRadio.times(ms)))]
        ends = [(ant_i[a], ant_i[b]) for (a, b) in Fring._member_antenna_pairs(ms)]
        _add_information!(info, seen, scan_i[only(XRadio.scans(ms))], Fring.Cubes(ms), ts, ends, feed_pairs(ms))
    end
    info[.!seen] .= NaN
    for s in axes(info, 1), n in axes(info, 2), f in axes(info, 3)
        peak = maximum(x -> isnan(x) ? -Inf : x, view(info, s, n, f, :))
        peak > 0 && (view(info, s, n, f, :) ./= peak)
    end
    dims = (Fring._scan_dim(scans), Fring._antenna_dim(geom.antennas), Feed(1:geom.nfeed), Ti(times))
    return DimensionalData.DimArray(info, dims)
end

"""
    flag_low_signal!(data, info; below = 0.01) -> data

Flag every product that relates a feed at the integrations where that feed's
[`signal_information`](@ref) `info` is below `below`, a fraction of its largest
value in the scan: at `below = 0.01` the feed's signal amplitude there is under
a tenth of its best. Integrations `info` does not cover, or where it is `NaN`,
are left as they are. `data` must be in memory.
"""
function flag_low_signal!(data::XRadio.ProcessingSet, info::AbstractDimArray; below::Real = 0.01)
    low = _low_samples(permutedims(info, (Scan, AntennaName, Feed, Ti)), below)
    for ms in values(data)
        ts = Float64.(collect(XRadio.times(ms)))
        _flag_low!(Fring.Cubes(ms), low, only(XRadio.scans(ms)), ts, Fring._member_antenna_pairs(ms), feed_pairs(ms))
    end
    return data
end

# The `(scan, station, feed, time)` of every entry of `info`, over
# `(Scan, AntennaName, Feed, Ti)`, below `below`.
function _low_samples(info, below)
    sc, an, fe, tt = lookup(info, Scan), lookup(info, AntennaName), lookup(info, Feed), lookup(info, Ti)
    return Set((sc[I[1]], an[I[2]], fe[I[3]], tt[I[4]]) for I in CartesianIndices(info) if info[I] < below)
end

# One Measurement Set's part of `signal_information` into scan `s` of `info`:
# `ts` maps its integrations to `info`'s times, `ends` each baseline to its two
# stations and `feeds[p, bl]` each product to the feeds it relates.
function _add_information!(info, seen, s, c::Fring.Cubes, ts, ends, feeds)
    (; V, W, F) = c
    usable(ch, ti, bl, p) = !F[ch, ti, bl, p] && W[ch, ti, bl, p] > 0 && isfinite(W[ch, ti, bl, p]) && isfinite(V[ch, ti, bl, p])
    for (bl, (a, b)) in pairs(ends), p in axes(V, 4)
        a == b && continue
        num, den = zero(ComplexF64), 0.0
        for ti in axes(V, 2), ch in axes(V, 1)
            usable(ch, ti, bl, p) || continue
            num += W[ch, ti, bl, p] * V[ch, ti, bl, p]
            den += W[ch, ti, bl, p]
        end
        den > 0 || continue
        power = abs2(num / den)
        fa, fb = feeds[p, bl]
        for (j, ti) in zip(ts, axes(V, 2)), ch in axes(V, 1)
            usable(ch, ti, bl, p) || continue
            for (n, f) in ((a, fa), (b, fb))
                info[s, n, f, j] += W[ch, ti, bl, p] * power
                seen[s, n, f, j] = true
            end
        end
    end
    return info
end

# Flags one Measurement Set's products at the `(scan, station, feed, time)` in
# `low`.
function _flag_low!(c::Fring.Cubes, low, scan, ts, ends, feeds)
    for (bl, (a, b)) in pairs(ends), p in axes(c.F, 4), (ti, t) in zip(axes(c.F, 2), ts)
        fa, fb = feeds[p, bl]
        ((scan, a, fa, t) in low || (scan, b, fb, t) in low) && (c.F[:, ti, bl, p] .= true)
    end
    return c
end
