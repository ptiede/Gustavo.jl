"""
    apply_bandpass(data::UVData, gains)

Return a new `UVData` with corrected visibilities. Each visibility is divided by
g_{fa}[a,c] * conj(g_{fb}[b,c]) where (fa, fb) is the feed pair for that
polarisation product. Flagged samples (weight ≤ 0) are passed through unchanged.
"""
function apply_bandpass(data::UVData, gains)
    nint, npol, nchan = size(data.vis)
    vis_corr     = copy(data.vis)
    weights_corr = copy(data.weights)

    for i in 1:nint
        s  = data.scan_idx[i]
        s == 0 && continue
        bi = get(data.bl_lookup, data.bl_codes[i], 0)
        bi == 0 && continue
        a, b = data.bl_pairs[bi]

        for p in 1:min(npol, 4), c in 1:nchan
            data.weights[i, p, c] > 0 || continue
            fa, fb = polarization_feeds(data, p)
            vis_corr[i, p, c]     /= gains[s, a, fa, c] * conj(gains[s, b, fb, c])
            weights_corr[i, p, c] *= abs2(gains[s, a, fa, c]) * abs2(gains[s, b, fb, c])
        end
    end

    return with_visibilities(data, vis_corr, weights_corr)
end
