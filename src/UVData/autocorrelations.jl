"""
    normalize_by_autocorrelations(ms::XRadio.MeasurementSet) -> MeasurementSet

A copy of `ms` whose cross-correlations are correlation coefficients. Each
visibility on baseline `(a, b)` relating feeds `(fa, fb)` is divided by
`√(A_a · A_b)`, where `A_a` is the amplitude of antenna `a`'s autocorrelation of
feed `fa` at the same channel and time, and its weight is multiplied by
`A_a · A_b`, so the weight stays `1/σ²` of the visibility it describes. Feeds
come from [`feed_pairs`](@ref), not from the product labels.

A sample whose two autocorrelations are not both present, unflagged, finite
and positive is flagged instead, and left undivided. The autocorrelation
baselines are then flagged: MSv4 keeps them, but they are no longer data.

A Measurement Set with no autocorrelation baselines is returned unchanged:
its visibilities are taken to be normalized already, or to need no
normalization.
"""
function normalize_by_autocorrelations(ms::XRadio.MeasurementSet)
    feeds = feed_pairs(ms)
    stations = baselines(ms).pairs
    autos = Dict(
        (a, first(feeds[p, bi])) => (Polarization(p), BaselineID(bi))
            for (bi, (a, b)) in pairs(stations) if a == b
            for p in axes(feeds, 1) if allequal(feeds[p, bi])
    )
    isempty(autos) && return ms
    vis = modify(Array, ms[:visibility])
    weight = modify(Array, ms[:weight])
    flag = modify(Array, ms[:flag])
    for (bi, (a, b)) in pairs(stations)
        a == b && continue
        for p in axes(feeds, 1)
            fa, fb = feeds[p, bi]
            _normalize_cell!(
                vis, weight, flag, (Polarization(p), BaselineID(bi)),
                get(autos, (a, fa), nothing), get(autos, (b, fb), nothing),
            )
        end
    end
    for (bi, (a, b)) in pairs(stations)
        a == b && (view(flag, BaselineID(bi)) .= true)
    end
    out = copy(ms)
    out[:visibility] = vis
    out[:weight] = weight
    out[:flag] = flag
    return out
end

function _normalize_cell!(vis, weight, flag, cell, auto_a, auto_b)
    f = view(flag, cell...)
    if auto_a === nothing || auto_b === nothing
        f .= true
        return
    end
    power = abs.(view(vis, auto_a...)) .* abs.(view(vis, auto_b...))
    usable = .!view(flag, auto_a...) .& .!view(flag, auto_b...) .&
        isfinite.(power) .& (power .> 0)
    v = view(vis, cell...)
    w = view(weight, cell...)
    v .= ifelse.(usable, v ./ sqrt.(power), v)
    w .= ifelse.(usable, w .* power, w)
    f .|= .!usable
    return
end
