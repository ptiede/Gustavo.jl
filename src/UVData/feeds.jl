# Correlation-product label conventions. A generic product label is two feed
# letters, `"P"` (feed 1) and `"Q"` (feed 2), so `"PQ"` relates `V[a, b, p]` to
# antenna `a`'s feed 1 and antenna `b`'s feed 2. These read `pol_products` and
# `AntennaTable.nominal_basis`, so they belong beside the data model; the
# calibration solvers re-export them.

"""
    correlation_feed_pair(label::AbstractString) -> Tuple{Int, Int}

Map a generic correlation-product label `"PP"`/`"PQ"`/`"QP"`/`"QQ"` to the
`(feed_a_idx, feed_b_idx)` pair (each ∈ {1, 2}) that the visibility
`V[a, b, p]` relates to per-antenna gains via
`V[a, b, p] = g_a[feed_a] · S · conj(g_b[feed_b])`.
"""
function correlation_feed_pair(label::AbstractString)
    length(label) == 2 || error("correlation_feed_pair: expected 2-char label, got \"$label\"")
    a = _feed_index(label[1])
    b = _feed_index(label[2])
    return a, b
end
_feed_index(c::Char) = c == 'P' ? 1 : c == 'Q' ? 2 : error("Unsupported feed letter '$c' (expected 'P' or 'Q')")

"""
    is_parallel_hand(label::AbstractString) -> Bool

True for the parallel-hand correlations `"PP"` and `"QQ"`; false for
`"PQ"` and `"QP"`.
"""
is_parallel_hand(label::AbstractString) =
    length(label) == 2 && label[1] == label[2]

"""
    same_feed_label(a, b) -> Bool

True iff two `PolTypes` (RPol/LPol/XPol/YPol) carry the same feed label.
Used by `build_parallel_hand_mask` to determine, per-baseline, whether
the two antennas' feed-1 (resp. feed-2) form a parallel-hand correlation.
"""
same_feed_label(a::T, b::T) where {T <: PolTypes} = true
same_feed_label(::PolTypes, ::PolTypes) = false

"""
    parallel_hand_indices(pol_products) -> Tuple{Int, Int}

Indices of `"PP"` and `"QQ"` in `pol_products`. Errors if either is
missing.
"""
function parallel_hand_indices(pol_products)
    pp = findfirst(==("PP"), pol_products)
    qq = findfirst(==("QQ"), pol_products)
    (isnothing(pp) || isnothing(qq)) &&
        error("Parallel-hand products not found in pol_products=$(collect(pol_products))")
    return pp, qq
end

"""
    cross_hand_indices(pol_products) -> NamedTuple{(:pq, :qp), Tuple{Int, Int}} or nothing

Indices of the two cross-hand correlations `"PQ"` and `"QP"`, or `nothing`
if either is absent.
"""
function cross_hand_indices(pol_products)
    pq = findfirst(==("PQ"), pol_products)
    qp = findfirst(==("QP"), pol_products)
    !isnothing(pq) && !isnothing(qp) && return (; pq, qp)
    return nothing
end

# `build_parallel_hand_mask(antennas, bl_pairs)` answers, for each
# `(baseline, feed_index ∈ {1, 2})`, whether antenna A's feed-`feed_index`
# and antenna B's feed-`feed_index` carry the same nominal label (so the
# corresponding parallel-hand correlation on this baseline is meaningful).
# Mixed-feed arrays produce `false` for baselines crossing different
# nominal labels; uniform-feed arrays produce all `true`.
function build_parallel_hand_mask(antennas, bl_pairs)
    mask = falses(length(bl_pairs), 2)
    nb = antennas.nominal_basis  # Vector{NTuple{2, PolTypes}} via StructArray forwarding
    for (bi, (a, b)) in enumerate(bl_pairs)
        mask[bi, 1] = same_feed_label(nb[a][1], nb[b][1])
        mask[bi, 2] = same_feed_label(nb[a][2], nb[b][2])
    end
    return mask
end
