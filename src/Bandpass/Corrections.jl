"""
    BandpassCorrection(gains)

Per-leaf reducer that divides visibilities by `g_a * conj(g_b)` and scales
weights by `abs2(g_a)*abs2(g_b)`. Reads pol codes from the root
`UVMetadata.array_obs` passed in by `apply`. Flagged samples (weight ≤ 0)
pass through unchanged.

```julia
corrected = apply(BandpassCorrection(gains), uvset)
```

`apply_bandpass(uvset, gains)` is the named-call shortcut.
"""
struct BandpassCorrection{G} <: UVData.AbstractPartitionReducer
    gains::G
end

(c::BandpassCorrection)(leaf::DimensionalData.AbstractDimTree, ::NamedTuple, ::UVData.UVMetadata) =
    _apply_bandpass_partition(leaf, c.gains, pol_products(leaf))

"""
    apply_bandpass(uvset::UVSet, gains)

Return a new `UVSet` with each leaf's visibilities divided by
`g_{fa}[a,c] * conj(g_{fb}[b,c])` and weights scaled by
`abs2(g_a) * abs2(g_b)`.
"""
apply_bandpass(uvset::UVSet, gains) = UVData.apply(BandpassCorrection(gains), uvset)

function _apply_bandpass_partition(leaf::DimensionalData.AbstractDimTree, gains, pol_products)
    sid = scan_idx(leaf)
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
