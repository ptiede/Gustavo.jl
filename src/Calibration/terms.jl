# ── Atomic gain terms ────────────────────────────────────────────────────────
#
# A term is one physical contribution to a station's phase or log-amplitude
# response. It lives inside a `GainComponent` that pins it to one (time-segment,
# frequency-segment) block of parameters, so a term never sees the global
# parameter vector — `term_eval` hands it its own named parameters and the
# coordinates it asked for.
#
# Writing a term means: a `term_axes`, a `param_shapes`, a `term_eval`, a
# `term_label`, and a coordinate builder for each axis declared.

abstract type AbstractGainTerm end

"Constant offset: phase/log-amp = `offset`. (Per-block constant — a fringe phase, or a flat gain.)"
struct ConstantTerm <: AbstractGainTerm end

"Group delay: phase = 2π·`delay`·(f − f0), `delay` in seconds."
struct Delay <: AbstractGainTerm end

# rad·Hz per TECU (1 TECU = 1e16 el/m²): ionospheric phase = −K·TEC/f.
const DISPERSION_K = 8.4479e9

"""
Ionospheric dispersion: phase = K·`dtec`·(1/f0 − 1/f) with K = $DISPERSION_K rad·Hz/TECU,
so `dtec` is a differential TEC in TECU. Station-based and non-magnetic to first
order, so it ties feeds (`SharedFeeds`). Referenced to f0 — the 1/f0 offset
lands in the accompanying constant/phase term, keeping this term pure shape.
"""
struct Dispersion <: AbstractGainTerm end

"Fringe rate: phase = 2π·`rate`·(t − t0)·3600, `rate` in Hz (t in hours)."
struct Rate <: AbstractGainTerm end

"""
    Polynomial{axis}(degree)

Polynomial in a segment-scaled coordinate: Σ_{d=1}^{degree} `coeffs`[d] · x^d, where
`axis` is the coordinate axis it reads — `:Frequency` or `:Ti`. The axis is a type
parameter because it must be known statically. Use the constructor functions
[`PolynomialFreq`](@ref) / [`PolynomialTime`](@ref).
"""
struct Polynomial{A} <: AbstractGainTerm
    degree::Int
    function Polynomial{A}(degree) where {A}
        A in TERM_AXES ||
            throw(ArgumentError("Polynomial axis must be one of $TERM_AXES, got $(repr(A))"))
        degree >= 1 || throw(ArgumentError("Polynomial degree must be ≥ 1, got $degree"))
        return new{A}(Int(degree))
    end
end

"""
    PolynomialFreq(degree) -> Polynomial{:Frequency}

Polynomial in a segment-scaled frequency coordinate. Dispatch on `Polynomial`
(or `Polynomial{:Frequency}`); this is a function, not a type.
"""
PolynomialFreq(degree::Integer) = Polynomial{:Frequency}(degree)

"""
    PolynomialTime(degree) -> Polynomial{:Ti}

Polynomial in a segment-scaled time coordinate. Dispatch on `Polynomial` (or
`Polynomial{:Ti}`); this is a function, not a type.
"""
PolynomialTime(degree::Integer) = Polynomial{:Ti}(degree)

# ── Coordinate axes ──────────────────────────────────────────────────────────
#
# `term_axes(term)` names the coordinate axes a term reads: `:Frequency`, `:Ti`,
# both, or neither. `term_eval` receives exactly these as a `NamedTuple` — a
# term reading both writes `x.Frequency` and `x.Ti` — and the layout builds only
# the axes that are declared. The names are checked at plan time, so a typo
# fails there rather than evaluating against the wrong axis.
const TERM_AXES = (:Frequency, :Ti)

term_axes(::ConstantTerm) = ()
term_axes(::Delay) = (:Frequency,)
term_axes(::Dispersion) = (:Frequency,)
term_axes(::Rate) = (:Ti,)
term_axes(::Polynomial{C}) where {C} = (C,)

# ── Parameter names and shapes ───────────────────────────────────────────────
#
# `param_shapes(term, nchan_seg)` names the parameters of one block and gives
# each one a shape: `()` for a scalar, `(n,)` for a vector of length `n`.
# `nchan_seg` is the number of channels in the term's frequency segment, for the
# terms whose arity the data sets. How finely a term varies in frequency is said
# by its frequency SEGMENTATION, not by its parameter count: a free value per
# channel is `ConstantTerm` × `ChannelBlocks(1)`.
#
# `term_eval` receives these as a `NamedTuple`, so a term author writes `p.delay`
# and never a position into a block whose length they would have to know. Shapes
# are part of the type, not runtime data, so the `NamedTuple` is built statically
# and costs nothing.
#
# The names are LOCAL to one term's own block and are never merged with another
# term's: two terms may both name a parameter `:offset` without colliding. An
# aggregate view of θ must therefore namespace by component, never by parameter
# name.
param_shapes(::ConstantTerm, nchan_seg) = (offset = (),)
param_shapes(::Delay, nchan_seg) = (delay = (),)
param_shapes(::Dispersion, nchan_seg) = (dtec = (),)
param_shapes(::Rate, nchan_seg) = (rate = (),)
param_shapes(t::Polynomial, nchan_seg) = (coeffs = (t.degree,),)

"""
    nparams_per_block(term, nchan_seg) -> Int

Number of parameters one (time-segment, frequency-segment) block of `term`
occupies in θ. A term names its parameters rather than counting them, so a count
can never disagree with the names it is derived from.
"""
nparams_per_block(t::AbstractGainTerm, nchan_seg) =
    sum(prod, values(param_shapes(t, nchan_seg)))

# ── Coordinate builders ──────────────────────────────────────────────────────
#
# One builder per axis, because a term may read both: `freq_coordinate` gives the
# `x.Frequency` a term sees, one entry per channel, and `time_coordinate` the
# `x.Ti`, one entry per time sample. A term defines a builder for each axis it
# declares in `term_axes`.
#
# There is deliberately NO generic fallback for either. `plan_parameters` calls a
# builder for every axis a term declares, so a term that forgets one errors
# loudly instead of silently evaluating that axis at zero.

# Delay uses the physical offset (f − f0) in Hz so θ is a delay in seconds.
freq_coordinate(::Delay, channel_freqs, fseg_groups, f0) = Float64.(channel_freqs) .- f0

# Dispersion uses K·(1/f0 − 1/f) so θ is a differential TEC in TECU. The
# f0-referencing keeps it orthogonal to the constant term at f0 (not globally —
# the delay↔dTEC covariance over a finite band is physical; solvers fit them
# jointly).
freq_coordinate(::Dispersion, channel_freqs, fseg_groups, f0) =
    DISPERSION_K .* (1.0 / f0 .- 1.0 ./ Float64.(channel_freqs))

# Rate uses (t − t0) in seconds (t given in hours) so θ is a rate in Hz.
time_coordinate(::Rate, times, tseg_groups, t0) = (Float64.(times) .- t0) .* 3600.0

# A polynomial uses a per-segment centered/scaled coordinate in ~[-1, 1] so the
# basis is well conditioned; the same construction serves either axis, which is
# why one `Polynomial` term covers both.
freq_coordinate(::Polynomial{:Frequency}, channel_freqs, fseg_groups, f0) =
    _centered_scaled(channel_freqs, fseg_groups)
time_coordinate(::Polynomial{:Ti}, times, tseg_groups, t0) =
    _centered_scaled(times, tseg_groups)

# Center = segment mean; scale = max|v − center|.
function _centered_scaled(values, groups)
    x = zeros(Float64, length(values))
    for grp in groups
        isempty(grp) && continue
        vs = @view values[grp]
        center = sum(vs) / length(vs)
        scale = maximum(abs.(vs .- center))
        scale = scale > 0 ? scale : 1.0
        for i in grp
            x[i] = (values[i] - center) / scale
        end
    end
    return x
end

# ── Pure scalar evaluation ───────────────────────────────────────────────────
#
#     term_eval(term, p, x)
#
# is a term's contribution to phase / log-amplitude at one (channel, time) cell.
# `p` holds the term's own parameters, named and shaped as `param_shapes`
# declares — a scalar per `()` name, a vector view per `(n,)` name. `x` holds the
# coordinates `term_axes` declares, under those names: `x.Frequency`, `x.Ti`, or
# both. A term never sees the global parameter vector, nor the layout that
# addresses it, nor an index into either.
#
# This is the hot path of `evaluate_gains` — kept allocation-free and type-stable
# so the whole forward map is inferrable (and Reactant-traceable). Read `p` by
# field name; iterating a `NamedTuple` is not type-stable.
@inline term_eval(::ConstantTerm, p, x) = p.offset
@inline term_eval(::Delay, p, x) = 2π * p.delay * x.Frequency
@inline term_eval(::Dispersion, p, x) = p.dtec * x.Frequency
@inline term_eval(::Rate, p, x) = 2π * p.rate * x.Ti

# Σ_{d=1}^{degree} c_d · x^d, in the axis this polynomial declared. The basis
# starts at x¹: a constant belongs to an accompanying `ConstantTerm`, and
# including one here would be degenerate with it.
@inline function term_eval(::Polynomial{A}, p, x) where {A}
    xa = getproperty(x, A)
    return xa * evalpoly(xa, p.coeffs)
end

# Labels for diagnostics / summaries.
term_label(::ConstantTerm) = "const"
term_label(::Delay) = "delay"
term_label(::Dispersion) = "dtec"
term_label(::Rate) = "rate"
term_label(t::Polynomial{:Frequency}) = "polyf$(t.degree)"
term_label(t::Polynomial{:Ti}) = "polyt$(t.degree)"
