# ── Inverse-variance sums over named dimensions ──────────────────────────────

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

# Behind a function barrier: a Measurement Set's layers do not infer.
function _weighted_sums!(wv, ws, V, W, F, dims)
    for I in DimensionalData.DimIndices(V)
        w, v = W[I], V[I]
        _usable(F[I], w, v) || continue
        cell = DimensionalData.otherdims(I, dims)
        wv[cell] += w * v
        ws[cell] += w
    end
    return wv, ws
end
