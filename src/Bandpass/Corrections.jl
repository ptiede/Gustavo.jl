"""
    BandpassCorrection(gains, sid_lookup)

Per-leaf reducer that divides visibilities by `g_a * conj(g_b)` and scales
weights by `abs2(g_a)*abs2(g_b)`. `sid_lookup` is a callable mapping a leaf
to its 1-based scan index in the `gains` array (built once by
`apply_bandpass(uvset, gains)`). Flagged samples (weight ≤ 0) pass through
unchanged.

```julia
corrected = apply_bandpass(uvset, gains)
```
"""
struct BandpassCorrection{G, L} <: UVData.AbstractPartitionReducer
    gains::G
    sid_lookup::L
end

(c::BandpassCorrection)(leaf::DimensionalData.AbstractDimTree, ::UVData.PartitionInfo, ::UVData.UVMetadata) =
    _apply_bandpass_partition(leaf, c.gains, c.sid_lookup, pol_products(leaf))

"""
    apply_bandpass(uvset::UVSet, gains)

Return a new `UVSet` with each leaf's visibilities divided by
`g_{fa}[a,c] * conj(g_{fb}[b,c])` and weights scaled by
`abs2(g_a) * abs2(g_b)`. `gains` may be:

- a `DimArray{(Scan,Ant,Pol,IF)}` (typically returned by
  `wrap_gain_solutions`) — leaves are matched to gain slots by `Scan`
  lookup against each leaf's scan-time center, or
- a plain `Array{T,4}` — the first axis is taken to enumerate leaves in the
  same order as `_to_bandpass_dataset` (sorted by `(scan_window.lower,
  sub_scan_name)`).
"""
function apply_bandpass(uvset::UVSet, gains)
    sid_lookup = _build_gains_sid_lookup(gains, uvset)
    return UVData.apply(BandpassCorrection(gains, sid_lookup), uvset)
end

# Map leaf → sid by matching leaf scan-time center to the gains Scan axis.
function _build_gains_sid_lookup(gains::DimensionalData.AbstractDimArray, uvset::UVSet)
    scan_lookup = collect(lookup(gains, Scan))
    centers = Dict{Float64, Int}()
    for (i, c) in enumerate(scan_lookup)
        centers[Float64(c)] = i
    end
    return leaf -> begin
        lo, hi = UVData.scan_window(leaf)
        (isfinite(lo) && isfinite(hi)) || return 0
        get(centers, (lo + hi) / 2, 0)
    end
end

# Plain Array: first axis aligned with leaves in `_to_bandpass_dataset` order.
function _build_gains_sid_lookup(gains::AbstractArray, uvset::UVSet)
    sorted_leaves = sort(
        collect(values(DimensionalData.branches(uvset)));
        by = leaf -> (
            UVData.scan_window(leaf)[1],
            DimensionalData.metadata(leaf).sub_scan_name,
        ),
    )
    sid_by_obj = IdDict{Any, Int}()
    for (i, leaf) in enumerate(sorted_leaves)
        sid_by_obj[leaf] = i
    end
    return leaf -> get(sid_by_obj, leaf, 0)
end

function _apply_bandpass_partition(
        leaf::DimensionalData.AbstractDimTree, gains, sid_lookup, pol_products,
    )
    sid = sid_lookup(leaf)
    sid <= 0 && return leaf

    bl_pairs = baselines(leaf).pairs
    vis_l = leaf[:vis]
    w_l = leaf[:weights]
    # Function barrier: dispatch through `_apply_bandpass_kernel!` so the
    # inner loops see concrete eltypes. `parent(leaf[:vis])` returns a
    # DimArray backed by `data(leaf)::OrderedDict{Symbol, Any}`, which
    # otherwise boxes every scalar in the hot loop.
    vis_corr, weights_corr = _apply_bandpass_kernel(
        vis_l, w_l, gains, bl_pairs, pol_products, sid,
    )
    return with_visibilities(leaf, vis_corr, weights_corr)
end

function _apply_bandpass_kernel(
        vis_p::AbstractArray,
        w_p::AbstractArray,
        gains, bl_pairs, pol_products, sid::Integer,
    )
    vis_corr = copy(vis_p)
    weights_corr = copy(w_p)

    for ti in axes(vis_p, Ti), bi in axes(vis_p, Baseline)
        a, b = bl_pairs[bi]
        for p in axes(vis_p, Pol)
            fa, fb = correlation_feed_pair(pol_products[Int(p)])
            for c in axes(vis_p, IF)
                w = w_p[ti, bi, p, c]
                (w > 0 && isfinite(w)) || continue
                ga = gains[sid, a, fa, c]
                gb = gains[sid, b, fb, c]
                vis_corr[ti, bi, p, c] /= ga * conj(gb)
                weights_corr[ti, bi, p, c] *= abs2(ga) * abs2(gb)
            end
        end
    end
    return vis_corr, weights_corr
end
