# ── Fringe search over one scan group ────────────────────────────────────────
#
# Given a scan group (a `ProcessingSet` of Measurement Sets sharing one time
# axis) and its `DataGeometry`, run the per-baseline kernels in `search.jl` over
# every (antenna pair, feed pair) cell.

# One recorded search row: the antenna names, feed pair, SNR, the
# family-wise false-alarm probability, and whether that PFA accepted it as a real
# fringe. Every measured cell gets a row, so `detected`, not the row's presence,
# is what marks a detection.
#
# `phase` (rad) is the measured phase at the epoch the search referenced, the
# scan's mean time. No station solve uses it.
const DetectionRow = @NamedTuple{
    a::String, b::String, feeds::Tuple{Int, Int}, snr::Float64, pfa::Float64,
    delay::Float64, rate::Float64, phase::Float64, detected::Bool,
    snr_steer::Float64, pfa_steer::Float64,
    delay_steer::Float64, rate_steer::Float64, steered::Bool,
}

# A scan group indexed by cell: each member's layers and, for every cross
# antenna pair `j` (antenna names) and feed pair `q` of the group, where that cell is stored in
# member `m` (`loc[m, j, q] = (bi, p)`, or `(0, 0)` when the member lacks it). A
# stored product's feed pair varies by baseline, so the cells are joined by
# label. Members are in frequency order and share one time axis; `freqs` is
# their channels concatenated in that order, the rows of a gathered plane.
struct _GroupCells{L, N, F, T, B, P, I}
    layers::L
    nchan::N
    freqs::F
    times::T
    antenna_pairs::B
    feeds::P
    loc::I
end

function _GroupCells(group::XRadio.ProcessingSet, geom::DataGeometry)
    isempty(group) && throw(ArgumentError("the scan group holds no Measurement Sets"))
    members = _members_by_frequency(group)
    times = collect(XRadio.times(first(members)))
    for ms in members
        XRadio.times(ms) == times || throw(
            DimensionMismatch("the members of a scan group must share one time axis"),
        )
    end
    member_pairs = map(_member_station_pairs, members)
    member_feeds = map(feed_pairs, members)
    stations, feeds = _cross_cell_labels(member_pairs, member_feeds, geom)
    row = Dict(p => j for (j, p) in pairs(stations))
    col = Dict(f => q for (q, f) in pairs(feeds))
    loc = fill((0, 0), length(members), length(stations), length(feeds))
    for (m, (ps, fs)) in enumerate(zip(member_pairs, member_feeds)), bi in axes(fs, 2)
        j = get(row, ps[bi], 0)
        j == 0 && continue
        for p in axes(fs, 1)
            loc[m, j, col[fs[p, bi]]] = (bi, p)
        end
    end
    return _GroupCells(
        map(_member_layers, members), [length(XRadio.frequencies(ms)) for ms in members],
        reduce(vcat, (collect(XRadio.frequencies(ms)) for ms in members)), times,
        stations, feeds, loc,
    )
end

# The cross antenna pairs of Measurement Sets with antenna pairs `member_pairs`
# and feed pairs `member_feeds`, in `geom`'s station order, and the sorted union
# of their feed pairs.
function _cross_cell_labels(member_pairs, member_feeds, geom::DataGeometry)
    cross = [p for ps in member_pairs for p in ps if p[1] != p[2]]
    feeds = sort!(unique!([f for fs in member_feeds for f in fs]))
    return _station_pairs(cross, geom), feeds
end

# The false-alarm family of a search of `gc` (see `search_scan`): the trial
# count of one search times the scan's `ncross×npol` searches.
function _family_cells(gc::_GroupCells, params::FringeSearch)
    nsearch = max(length(gc.antenna_pairs) * length(gc.feeds), 1)
    return _search_cells(gc.freqs, gc.times, params) * nsearch
end

# Cell `(j, q)`'s layers copied into `ws` frequency-fastest, one member's
# channels after another. A member that does not store the cell contributes
# flagged rows.
function _gather_cell!(ws::FringeWorkspace, gc::_GroupCells, j::Integer, q::Integer)
    bufs = _plane_buffers!(ws, first(gc.layers), (length(gc.freqs), length(gc.times)))
    offset = 0
    for m in eachindex(gc.layers, gc.nchan)
        rows = (offset + 1):(offset + gc.nchan[m])
        offset = last(rows)
        _copy_cell_rows!(bufs, gc.layers[m], gc.loc[m, j, q], rows)
    end
    return bufs
end

function _copy_cell_rows!(bufs, layers, (bi, p), rows)
    V, W, F = bufs
    if bi == 0
        V[rows, :] .= zero(eltype(V))
        W[rows, :] .= zero(eltype(W))
        F[rows, :] .= true
        return nothing
    end
    foreach(bufs, layers) do buf, L
        buf[rows, :] .= _cell_plane(L, bi, p)
    end
    return nothing
end

"""
    search_scan(group::XRadio.ProcessingSet, geom::DataGeometry, params::FringeSearch;
                executor = SerialScheduler(), t0 = geom.t0) -> DimStack

Fringe-search every cross (antenna pair, feed pair) cell of a scan
group, one Measurement Set per spectral window sharing one time axis. Each
cell's plane is gathered across the members in frequency order, so the delay
search spans every window. Autocorrelations are skipped. `geom` supplies the
reference frequency `f0`, the default phase epoch, and the order of the
antenna pairs. To search a residual, pass the group [`residual_group`](@ref)
returns.

Each cell's `pfa` is computed over the scan's family of `ncross×npol`
searches, which share one budget (Bonferroni), so a recorded `pfa` is directly
comparable to `Stationization.pfa_max` and does not depend on how many other
scans a solve holds.

`t0` (seconds) is the epoch the detection phases are referenced to
(delay/rate/SNR are epoch-invariant). Quoting a phase a lever arm from the
data costs it `2π·σ_rate·Δt`, so the scan midpoint is usually wanted. The
default is `geom`'s track epoch, which is the midpoint only for a single-scan
geometry.

Returns a `DimStack` over `AntennaPair × FeedPair` whose layers are the seven
[`Detection`](@ref) fields (`:delay`/`:rate`/`:phase`/`:amp`/`:snr`/`:pfa`/`:valid`),
so one cell reads back as a `Detection` `NamedTuple`. The `AntennaPair` lookup
holds each cross pair's antenna names, in `geom`'s station order, and the
`FeedPair` lookup the feed-index pairs. The layers' element type is the real
type of the visibilities. Results are bit-identical to the serial loop
regardless of the fan-out `executor`.
"""
search_scan(group::XRadio.ProcessingSet, geom::DataGeometry, params::FringeSearch; kw...) =
    search_scan(_GroupCells(group, geom), geom, params; kw...)

function search_scan(
        gc::_GroupCells, geom::DataGeometry, params::FringeSearch;
        executor = SerialScheduler(), t0::Real = geom.t0,
    )
    C = eltype(first(first(gc.layers)))
    T = real(C)
    dims = (_station_pair_dim(gc.antenna_pairs), FeedPair(gc.feeds))
    delay = zeros(T, dims...)
    rate = similar(delay)
    phase = similar(delay)
    amp = similar(delay)
    snr = similar(delay)
    # `zeros(Bool, …)`, not `falses`: adjacent bits of a BitArray share a word,
    # and the fan-out writes distinct cells concurrently.
    valid = zeros(Bool, dims...)
    pfa = similar(delay)
    scube = DimensionalData.DimStack((; delay, rate, phase, amp, snr, pfa, valid))

    family_cells = _family_cells(gc, params)
    ax = _search_axes(gc.freqs, gc.times, params, C)

    workspace = TaskLocalValue{FringeWorkspace{C}}(() -> FringeWorkspace(C))
    cells = vec(CartesianIndices(valid))
    tforeach(cells; scheduler = executor) do I
        j, q = Tuple(I)
        ws = workspace[]
        V, W, F = _gather_cell!(ws, gc, j, q)
        scube[AntennaPair(j), FeedPair(q)] = _baseline_fringe_search(
            V, W, F, gc.freqs, gc.times, geom.f0, Float64(t0), ax, ws, params, family_cells,
        )
    end
    _warn_edge_peaks(scube, params, ax)
    return scube
end
