# ── A scan group placed on the run's geometry ────────────────────────────────

"""
    GroupTables(group::XRadio.ProcessingSet, geom::DataGeometry)

The Measurement Sets of one scan group placed on `geom`: each member's
[`GeometryWindow`](@ref Calibration.GeometryWindow), and the group's own
baseline, feed-pair and time tables that a per-scan accumulator is indexed by.

`bl_pairs` are station pairs in `geom`'s station numbering, `feeds` the feed
pairs of the stored products, `ti` the group's time samples as indices into
`geom.times`; each is sorted. For member `m`, `blrow[m][bi]` is the group row
of its baseline `bi`, `feedrow[m][k]` the group index of its feed pair
`wins[m].feed_order[k]`, and `tpos[m][t]` the group position of its time `t`.

Members are ordered by their first channel in `geom`, so a sum over members
visits channels in increasing order. A baseline stored as `(a, b)` in one
member and `(b, a)` in another is refused: the two would conjugate each other.
"""
struct GroupTables
    members::Vector{XRadio.MeasurementSet}
    wins::Vector{GeometryWindow}
    bl_pairs::Vector{Tuple{Int, Int}}
    feeds::Vector{Tuple{Int, Int}}
    ti::Vector{Int}
    blrow::Vector{Vector{Int}}
    feedrow::Vector{Vector{Int}}
    tpos::Vector{Vector{Int}}
end

function GroupTables(group::XRadio.ProcessingSet, geom::DataGeometry)
    isempty(group) && throw(ArgumentError("the scan group holds no Measurement Sets"))
    members = XRadio.MeasurementSet[ms for ms in values(group)]
    wins = [GeometryWindow(geom, ms) for ms in members]
    order = sortperm(wins; by = w -> isempty(w.chan_idx) ? typemax(Int) : first(w.chan_idx))
    members, wins = members[order], wins[order]
    bl_pairs = sort!(unique!(reduce(vcat, (w.stations for w in wins))))
    for (a, b) in bl_pairs
        a < b && insorted((b, a), bl_pairs) && throw(
            ArgumentError(
                "stations $(geom.stations[a]) and $(geom.stations[b]) are stored in both " *
                    "orders within one scan group"
            )
        )
    end
    feeds = sort!(unique!(reduce(vcat, (w.feed_order for w in wins))))
    ti = sort!(unique!(reduce(vcat, (w.ti_idx for w in wins))))
    blrow = [[searchsortedfirst(bl_pairs, s) for s in w.stations] for w in wins]
    feedrow = [[searchsortedfirst(feeds, f) for f in w.feed_order] for w in wins]
    tpos = [[searchsortedfirst(ti, t) for t in w.ti_idx] for w in wins]
    return GroupTables(members, wins, bl_pairs, feeds, ti, blrow, feedrow, tpos)
end

# The visibility, weight and flag layers of one member, for a kernel behind a
# function barrier (a Measurement Set's layers do not infer).
_member_layers(ms::XRadio.MeasurementSet) = (ms[:visibility], ms[:weight], ms[:flag])

# The `(Frequency, Ti)` planes of baseline `bi`, product `p`. The kernels index
# them by the member's own positions, which the window's 1-based tables address.
function _member_planes(V, W, F, bi, p)
    Vp = UVData._cell_plane(V, bi, p)
    Wp = UVData._cell_plane(W, bi, p)
    Fp = UVData._cell_plane(F, bi, p)
    Base.require_one_based_indexing(Vp, Wp, Fp)
    return Vp, Wp, Fp
end

@inline _usable(f, w, v) = !f && w > 0 && isfinite(w) && isfinite(v)
