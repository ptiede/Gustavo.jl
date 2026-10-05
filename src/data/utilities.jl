"""
    check_layer_axes(reference, layers...)

Throw a `DimensionMismatch` unless every array in `layers` shares `reference`'s
axes.

The solver kernels walk parallel `:vis`, `:weights` and `:flags` planes with one
set of loop variables, so a layer whose axes differ would be read at the wrong
cells instead of being reported. Call this where the layers arrive as separate
arrays.
"""
function check_layer_axes(reference, layers...)
    ref = axes(reference)
    for l in layers
        axes(l) == ref || throw(
            DimensionMismatch("layer has axes $(axes(l)); expected $(ref)"),
        )
    end
    return nothing
end

"""
    feed_pairs(x) -> Vector{Tuple{Int, Int}}

The `(feed_a, feed_b)` pair each product along the `Polarization` axis of `x`
relates, so that `V[a, b, p] = g_a[feed_a] · S · conj(g_b[feed_b])`. A solver
cube's lookup holds these pairs directly. A Measurement Set's labels are
resolved through each antenna's receptors by
[`feed_pairs(::XRadio.MeasurementSet)`](@ref feed_pairs(::XRadio.MeasurementSet)).
"""
feed_pairs(vis::AbstractDimArray) = collect(Tuple{Int, Int}, lookup(vis, Polarization))

