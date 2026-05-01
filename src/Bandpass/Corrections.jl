"""
    apply_bandpass(uvset::UVSet, sols::AbstractDict) -> UVSet

Apply per-SPW bandpass solutions to `uvset`. `sols` is the Dict
returned by `solve_bandpass(uvset, ref_ant)`: keyed by `spw_name`,
each value is a `DimArray{Complex, 4}` with `(Scan, Ant, Pol, IF)`
dims.

Each leaf reads its own `info.spw_name`, looks up the matching gain
DimArray, and divides visibilities by `g_a * conj(g_b)` (scaling
weights by `abs2(g_a) * abs2(g_b)`). Errors loudly when a leaf's
SPW has no entry in `sols`.

Flagged samples (weight ≤ 0) pass through unchanged.

```julia
sols = solve_bandpass(uvset, ref_ant)
corrected = apply_bandpass(uvset, sols)
```
"""
function apply_bandpass(uvset::UVSet, sols::AbstractDict)
    return UVData.apply(uvset) do leaf, info, _root
        haskey(sols, info.spw_name) || error(
            "apply_bandpass: no gain solution for SPW '$(info.spw_name)'; " *
                "available SPWs = $(collect(keys(sols))).",
        )
        gains = sols[info.spw_name]
        sid = _scan_index_for_leaf(leaf, gains)
        sid <= 0 && return leaf

        bl_pairs = baselines(leaf).pairs
        vis_l = leaf[:vis]
        w_l = leaf[:weights]
        vis_corr, weights_corr = _apply_bandpass_kernel(
            vis_l, w_l, gains, bl_pairs, pol_products(leaf), sid,
        )
        return with_visibilities(leaf, vis_corr, weights_corr)
    end
end

# Look up a leaf's scan slot in the per-SPW gain DimArray's `Scan` axis
# by matching scan-time center. Returns 0 if the leaf has no usable
# scan window or no matching center (caller passes through unchanged).
function _scan_index_for_leaf(leaf, gains::DimensionalData.AbstractDimArray)
    scan_lookup = collect(lookup(gains, Ti))
    lo, hi = UVData.scan_window(leaf)
    (isfinite(lo) && isfinite(hi)) || return 0
    target = (lo + hi) / 2
    for (i, c) in enumerate(scan_lookup)
        c ≈ target && return i
    end
    return 0
end

function _apply_bandpass_kernel(
        vis_p::AbstractArray,
        w_p::AbstractArray,
        gains, bl_pairs, pol_products, sid::Integer,
    )
    # Layout: vis/weights are (Frequency, Ti, Baseline, Pol);
    # gains are (Frequency, Ti, Ant, Pol).
    vis_corr = copy(vis_p)
    weights_corr = copy(w_p)

    for ti in axes(vis_p, Ti), bi in axes(vis_p, Baseline)
        a, b = bl_pairs[bi]
        for p in axes(vis_p, Pol)
            fa, fb = correlation_feed_pair(pol_products[Int(p)])
            for c in axes(vis_p, Frequency)
                w = w_p[c, ti, bi, p]
                (w > 0 && isfinite(w)) || continue
                ga = gains[c, sid, a, fa]
                gb = gains[c, sid, b, fb]
                vis_corr[c, ti, bi, p] /= ga * conj(gb)
                weights_corr[c, ti, bi, p] *= abs2(ga) * abs2(gb)
            end
        end
    end
    return vis_corr, weights_corr
end
