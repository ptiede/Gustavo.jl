# ── Propagation model: the dispersive ionosphere ─────────────────────────────
#
# The specification of the dTEC term and the structural lookup of its θ
# columns. `Dispersion` (terms.jl) is the term this model configures; the
# estimator that fits it — jointly with the delay it is degenerate with — lives
# in `Gustavo.Fringe`.

"""
    DispersionModel(; require_band_separation = true, tie_colocated = true)

The differential-ionosphere (dTEC) term: a per-scan, feed-common phase ∝ 1/ν.
Give it to a `FringeFit` alongside its `FringeModel`, or pass
`dispersion = nothing` for a fit that models no ionosphere at all.

- `require_band_separation` — solve the term only when the band layout can
  actually separate 1/ν from a linear delay: several sub-bands over a wide
  fractional bandwidth (VGOS 3–10.7 GHz qualifies; a single contiguous band
  cannot constrain the curvature and the term would just soak up delay). Set
  `false` to solve it regardless.
- `tie_colocated` — tie co-located stations (< 1 km apart) to one dTEC. They
  see the same ionosphere, so a differential TEC between them is pure solve
  error.

Estimating dispersion is NOT separable from estimating delay: over a finite
band the two are near-degenerate, so the fringe estimator fits Δτ and dTEC
jointly. The separation here is of the model, not of the solve.
"""
Base.@kwdef struct DispersionModel
    require_band_separation::Bool = true
    tie_colocated::Bool = true
end

# Whether this geometry gets a dTEC term. No model, no term. With one, the
# `require_band_separation` gate asks whether the band layout can separate 1/ν
# from a linear delay: several sub-bands over a wide fractional bandwidth (VGOS
# 3–10.7 GHz qualifies; a single contiguous band cannot constrain the curvature
# and the term would just soak up delay).
_dispersion_enabled(::Nothing, ::DataGeometry) = false
function _dispersion_enabled(dm::DispersionModel, geom::DataGeometry)
    dm.require_band_separation || return true
    nb = length(unique(geom.spw_of_chan))
    fmin, fmax = extrema(geom.channel_freqs)
    return nb >= 4 && fmax / fmin > 1.3
end

# The dispersion component's plan, located by TERM TYPE rather than by index,
# or `nothing` when the model carries no dTEC term.
function _dispersion_plan(model, layout)
    i = findfirst(tc -> tc.component.term isa Dispersion, model.phase)
    return i === nothing ? nothing : layout.plans[i]
end
