# ── Inverse-variance sums over named dimensions ──────────────────────────────

# The visibility, weight and flag layers of one member, for a kernel behind a
# function barrier (a Measurement Set's layers do not infer).
_member_layers(ms::XRadio.MeasurementSet) = (ms[:visibility], ms[:weight], ms[:flag])

@inline _usable(f, w, v) = !f && w > 0 && isfinite(w) && isfinite(v)

"""
    weighted_sums(V, W, F; dims) -> (wv, ws)

The inverse-variance sums `Σ w·v` and `Σ w` of visibility layer `V` over the
dimensions `dims`, taken over the usable cells: not flagged in `F`, weight
positive and finite, visibility finite. Both are `DimArray`s over
`otherdims(V, dims)`, with the element types of `w·v` and `w`.

`W` and `F` must have `V`'s dimensions and lookups, in any storage order.
"""
function weighted_sums(V, W, F; dims)
    vdims = DimensionalData.dims(V)
    for L in (W, F)
        ndims(L) == ndims(V) || throw(
            DimensionMismatch("a layer has $(ndims(L)) dimensions; the visibilities have $(ndims(V))"),
        )
        DimensionalData.comparedims(DimensionalData.dims(L, vdims), vdims; val = true)
    end
    out = DimensionalData.otherdims(V, dims)
    wv = zeros(typeof(zero(eltype(W)) * zero(eltype(V))), out)
    ws = zeros(eltype(W), out)
    return _weighted_sums!(wv, ws, V, W, F, dims)
end

# Behind a function barrier: a Measurement Set's layers do not infer. `dims` is
# usually a type, which Julia does not specialize on unless it is a parameter.
function _weighted_sums!(wv, ws, V, W, F, dims::D) where {D}
    for I in DimensionalData.DimIndices(V)
        w, v = W[I], V[I]
        _usable(F[I], w, v) || continue
        cell = DimensionalData.otherdims(I, dims)
        wv[cell] += w * v
        ws[cell] += w
    end
    return wv, ws
end

# ── A scan group's sums, by label ────────────────────────────────────────────

# The group's inverse-variance sums per (station pair, feed pair, AP) over every
# channel of every member. Members are added in frequency order, so the float
# association is fixed by the data alone. Station pairs follow `geom`'s station
# numbering; `ti` gives the APs as indices into `geom.times`.
function _ap_sums(group::XRadio.ProcessingSet, geom::DataGeometry; executor)
    isempty(group) && throw(ArgumentError("the scan group holds no Measurement Sets"))
    parts = tmap(ms -> _member_ap_sums(ms, geom), _members_by_frequency(group); scheduler = executor)

    stations = _station_pairs(reduce(vcat, (p.stations for p in parts)), geom)
    feeds = sort!(unique!(reduce(vcat, (vec(p.feeds) for p in parts))))
    ti = sort!(unique!(reduce(vcat, (p.ti for p in parts))))

    ax = (_station_pair_dim(stations), FeedPair(feeds), Ti(geom.times[ti]))
    rbar = zeros(promote_type((eltype(p.wv) for p in parts)...), ax)
    wbar = zeros(promote_type((eltype(p.ws) for p in parts)...), ax)
    for p in parts
        _add_by_label!(rbar, wbar, p.wv, p.ws, p.stations, p.feeds, Ti(At(geom.times[p.ti])))
    end
    return (; rbar, wbar, ti)
end

_members_by_frequency(group) = sort!(collect(values(group)); by = ms -> minimum(XRadio.frequencies(ms)))

function _member_ap_sums(ms::XRadio.MeasurementSet, geom::DataGeometry)
    wv, ws = weighted_sums(_member_layers(ms)...; dims = Frequency)
    ti = Calibration._time_indices(geom, XRadio.times(ms))
    return (; wv, ws, stations = _member_station_pairs(ms), feeds = feed_pairs(ms), ti)
end

"""
    _station_pairs(pairs, geom::DataGeometry) -> Vector

The distinct station pairs among `pairs` (antenna-name tuples), ordered by
`geom`'s station numbering. Refuses a pair stored in both orders, since the two
would conjugate each other.
"""
function _station_pairs(pairs, geom::DataGeometry)
    slot = Dict(n => i for (i, n) in Base.pairs(geom.stations))
    station(n) = get(slot, n) do
        throw(ArgumentError("baseline antenna `$n` is not among the geometry's stations, " * join(geom.stations, ", ")))
    end
    order = Dict((a, b) => (station(a), station(b)) for (a, b) in pairs)
    stations = sort!(collect(keys(order)); by = p -> order[p])
    stored = Set(stations)
    for (a, b) in stations
        a != b && (b, a) in stored && throw(
            ArgumentError("stations $a and $b are stored in both orders"),
        )
    end
    return stations
end

_station_pair_dim(stations) = StationPair(
    DimensionalData.Lookups.Categorical(stations; order = DimensionalData.Lookups.Unordered()),
)
_station_dim(stations) = Ant(
    DimensionalData.Lookups.Categorical(stations; order = DimensionalData.Lookups.Unordered()),
)

_member_station_pairs(ms::XRadio.MeasurementSet) = [(String(a), String(b)) for (a, b) in XRadio.baselines(ms)]

# A stored product's feed pair varies by baseline, so each (baseline, product)
# of a member finds its row by label. `along` selects the third axis.
function _add_by_label!(rbar, wbar, wv, ws, stations, feeds, along; autos::Bool = true)
    for bi in axes(feeds, 2), p in axes(feeds, 1)
        a, b = stations[bi]
        autos || a != b || continue
        sel = (StationPair(At(stations[bi])), FeedPair(At(feeds[p, bi])), along)
        cell = (BaselineID(bi), Polarization(p))
        view(rbar, sel...) .+= view(wv, cell...)
        view(wbar, sel...) .+= view(ws, cell...)
    end
    return rbar, wbar
end
