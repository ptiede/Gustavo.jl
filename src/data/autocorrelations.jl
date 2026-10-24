"""
    normalize_by_autocorrelations!(ms::XRadio.MeasurementSet) -> ms

Make the cross-correlations of `ms` correlation coefficients, in place. Each
visibility on baseline `(a, b)` relating feeds `(fa, fb)` is divided by
`√(A_a · A_b)`, where `A_a` is the amplitude of antenna `a`'s autocorrelation of
feed `fa` at the same channel and time, and its weight is multiplied by
`A_a · A_b`, so the weight stays `1/σ²` of the visibility it describes. Feeds
come from [`feed_pairs`](@ref), not from the product labels.

A sample whose two autocorrelations are not both present, unflagged, finite
and positive is flagged instead, and left undivided. The autocorrelation
baselines are then flagged: MSv4 keeps them, but they are no longer data.

A Measurement Set with no autocorrelation baselines is left unchanged: its
visibilities are taken to be normalized already, or to need no normalization.
"""
function normalize_by_autocorrelations!(ms::XRadio.MeasurementSet)
    feeds = feed_pairs(ms)
    antennas = _antenna_pairs(ms)
    # Each autocorrelation as the (antenna, feed) it measures and its (product, baseline).
    autos = [
        (a, first(feeds[p, bi])) => (p, bi)
            for (bi, (a, b)) in pairs(antennas) if a == b
            for p in axes(feeds, 1) if allequal(feeds[p, bi])
    ]
    isempty(autos) && return ms
    slot = Dict(first(x) => k for (k, x) in pairs(autos))
    # Each (product, baseline) cell's two autocorrelations, as indices into
    # `autos` (0 for one the set lacks), or `(-1, -1)` for an autocorrelation.
    ends = [
        a == b ? (-1, -1) : (get(slot, (a, feeds[p, bi][1]), 0), get(slot, (b, feeds[p, bi][2]), 0))
            for p in axes(feeds, 1), (bi, (a, b)) in pairs(antennas)
    ]
    layers = map(_storage_order, (ms[:visibility], ms[:weight], ms[:flag]))
    _normalize!(layers..., last.(autos), ends)
    return ms
end

# The normalization on the layers in storage order `(Polarization, Frequency,
# BaselineID, Ti)`, behind a function barrier: each autocorrelation's scale
# `√|A|` once, `NaN` where it is unusable, then one pass over every sample in
# storage order.
function _normalize!(V, W, F, autos, ends)
    T = real(eltype(V))
    scale = similar(V, T, (axes(V, 2), axes(V, 4), eachindex(autos)))
    for (k, (p, bi)) in pairs(autos), t in axes(V, 4), c in axes(V, 2)
        A = abs(V[p, c, bi, t])
        scale[c, t, k] = !F[p, c, bi, t] && isfinite(A) && A > 0 ? sqrt(T(A)) : T(NaN)
    end
    for t in axes(V, 4), bi in axes(V, 3), c in axes(V, 2), p in axes(V, 1)
        ka, kb = ends[p, bi]
        s = ka > 0 && kb > 0 ? scale[c, t, ka] * scale[c, t, kb] : T(NaN)
        if isfinite(s)
            V[p, c, bi, t] /= s
            W[p, c, bi, t] *= s * s
        else
            F[p, c, bi, t] = true
        end
    end
    return nothing
end
