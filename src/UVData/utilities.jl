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
    sanitize_source(name::AbstractString) -> Symbol

Sanitize a source name into a valid Julia identifier `Symbol`, always prefixed
with `src_` so the key is identifier-safe (digit-leading catalog names like
`3C273` are otherwise illegal identifiers) and never masquerades as a real
source name. Non-identifier chars are replaced with `_`. Examples:
`"3C273"` → `:src_3C273`, `"Sgr A*"` → `:src_Sgr_A_`,
`"NGC 4486"` → `:src_NGC_4486`. Used as the source segment of the Measurement
Set keys [`load_uvfits`](@ref) gives.
"""
function sanitize_source(name::AbstractString)
    s = replace(strip(String(name)), r"[^A-Za-z0-9_]" => "_")
    isempty(s) && (s = "unknown")
    return Symbol("src_", s)
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

"""
    frequencies(x) -> Vector{Float64}

Channel frequencies (Hz) off the `Frequency` lookup of the visibility array
`x`. The raw coordinate vector, not a lookup wrapper.
"""
frequencies(vis::AbstractDimArray) = parent(lookup(vis, Frequency))

# ── Time axis ────────────────────────────────────────────────────────────────

"""
    JD_UNIX_EPOCH

Julian Day of 1970-01-01T00:00:00 UTC, the origin of the `Ti` axis.
"""
const JD_UNIX_EPOCH = 2440587.5

"""
    jd_to_unix(jd) -> Float64

Convert a Julian Day to the `Ti` axis' seconds since [`JD_UNIX_EPOCH`](@ref).

A Julian Day near the present is ~2.46e6, where a `Float64` resolves only
~40 µs, so a caller holding the day and its fraction separately — as FITS-IDI
`DATE`/`TIME` and the AIPS `DATE` PTYPE pair both do — must subtract the epoch
from the integer part *before* adding the fraction to keep sub-microsecond
timestamps.
"""
jd_to_unix(jd::Real) = (Float64(jd) - JD_UNIX_EPOCH) * 86400.0
