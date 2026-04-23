"""
    apply_bandpass(data::UVData, gains)

Return a new `UVData` with corrected visibilities. Each visibility is divided by
g_{fa}[a,c] * conj(g_{fb}[b,c]) where (fa, fb) is the feed pair for that
polarisation product. Flagged samples (weight ≤ 0) are passed through unchanged.
"""
function apply_bandpass(data::UVData, gains)
    nint, npol, nchan = size(data.vis)
    vis_corr = copy(data.vis)
    weights_corr = copy(data.weights)

    for i in 1:nint
        s = data.scan_idx[i]
        s == 0 && continue
        bi = get(data.bl_lookup, data.bl_codes[i], 0)
        bi == 0 && continue
        a, b = data.bl_pairs[bi]

        for p in 1:min(npol, 4), c in 1:nchan
            data.weights[i, p, c] > 0 || continue
            fa, fb = polarization_feeds(data, p)
            vis_corr[i, p, c] /= gains[s, a, fa, c] * conj(gains[s, b, fb, c])
            weights_corr[i, p, c] *= abs2(gains[s, a, fa, c]) * abs2(gains[s, b, fb, c])
        end
    end

    return with_visibilities(data, vis_corr, weights_corr)
end

function restore_uvfits_shape(raw, raw_shape, squeeze_dims)
    shape = ntuple(i -> i in squeeze_dims ? 1 : raw_shape[i], length(raw_shape))
    return Float32.(reshape(raw, shape))
end

function default_output_path(path)
    root, ext = splitext(path)
    return root * "+bandpass" * ext
end

"""
    export_uvfits(data::UVData, output_path, vis_corr, weights_corr)

Write a uvfits copy of the file that produced `data` with the corrected visibilities
and weights stored in the primary random-groups payload.
"""
function export_uvfits(input_path, data::UVData, output_path, vis_corr, weights_corr)
    hdus = FITSFiles.fits(input_path)

    raw_corr = zeros(Float32, size(vis_corr, 1), 3, size(vis_corr, 2), size(vis_corr, 3))
    raw_corr[:, 1, :, :] = Float32.(real.(vis_corr))
    raw_corr[:, 2, :, :] = Float32.(imag.(vis_corr))
    raw_corr[:, 3, :, :] = Float32.(weights_corr)

    primary_data = merge(hdus[1].data, (data=restore_uvfits_shape(raw_corr, data.raw_shape, data.squeeze_dims),))
    hdus[1] = HDU(Random, primary_data, hdus[1].cards)

    write(output_path, hdus)
    return output_path
end
