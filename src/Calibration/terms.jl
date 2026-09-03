# ── Atomic gain terms ────────────────────────────────────────────────────────
#
# A term is one physical contribution to a station's phase or log-amplitude
# response. It lives inside an `GainComponent` that pins it to one (time-segment,
# frequency-segment) block of parameters, so a term never sees the global
# parameter vector — `term_eval` hands it its own named parameters and the
# coordinates it asked for.
#
# Writing a term means: a `term_axes`, a `param_shapes`, a `term_eval`, and —
# for each axis declared — a coordinate builder plus the resolved state that
# builder reads. `term_label` is optional; it defaults to the type name and only
# needs an override for a more evocative diagnostic label.

"""
    AbstractGainTerm

One physical contribution to a station's phase or log-amplitude response — a
delay, a rate, a polynomial bandpass shape. A concrete term implements
[`term_axes`](@ref), [`param_shapes`](@ref), a coordinate builder
([`freq_coordinate`](@ref) / [`time_coordinate`](@ref)) and its resolved state
([`freq_coord_state`](@ref) / [`time_coord_state`](@ref)) for each axis
declared, and [`term_eval`](@ref); [`term_label`](@ref) is optional. See the
"Authoring a new gain term" documentation page for a worked example.

A term lives inside an [`GainComponent`](@ref), which pins it to one
(time-segment, frequency-segment) block of parameters, so a term itself never
sees the global parameter vector.
"""
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

"""
Fringe rate: phase = 2π·`rate`·(t − t0)·3600, `rate` in Hz (t in hours), with t0
the mean epoch of the term's OWN time segment. A companion constant term is
therefore the phase at the middle of each segment, not at a track-wide epoch —
see `time_coord_state(::Rate, …)` for why the distinction is not cosmetic.
"""
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
const TERM_AXES = (:Frequency, :Ti)

"""
    term_axes(term) -> Tuple

The coordinate axes `term` reads, drawn from `(:Frequency, :Ti)`, both, or
neither. [`term_eval`](@ref) receives exactly these as a `NamedTuple` — a term
reading both writes `x.Frequency` and `x.Ti` — and the layout builds only the
axes that are declared. The names are checked at plan time, so a typo fails
there rather than evaluating against the wrong axis.
"""
function term_axes end

term_axes(::ConstantTerm) = ()
term_axes(::Delay) = (:Frequency,)
term_axes(::Dispersion) = (:Frequency,)
term_axes(::Rate) = (:Ti,)
term_axes(::Polynomial{C}) where {C} = (C,)

# ── Parameter names and shapes ───────────────────────────────────────────────

"""
    param_shapes(term, nchan_seg) -> NamedTuple

The parameters of one (time-segment, frequency-segment) block of `term`,
named and shaped: `()` for a scalar, `(n,)` for a vector of length `n`.
`nchan_seg` is the number of channels in the term's frequency segment, for
terms whose arity the data sets. How finely a term varies in frequency is
said by its frequency SEGMENTATION, not by its parameter count: a free value
per channel is `ConstantTerm` paired with `ChannelBlocks(1)`.

[`term_eval`](@ref) receives these as a `NamedTuple`, so a term author writes
`p.delay` and never a position into a block whose length it would have to
know. Shapes are part of the type, not runtime data, so the `NamedTuple` is
built statically and costs nothing.

The names are LOCAL to one term's own block and are never merged with
another's: two terms may both name a parameter `:offset` without colliding.
An aggregate view of θ must therefore namespace by component, never by
parameter name.
"""
function param_shapes end

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

"""
    freq_coordinate(term, f, state, seg::Integer) -> Real

`term`'s `x.Frequency` at frequency `f` (Hz), lying in the SOLVE's frequency
segment `seg`. `state` is whatever [`freq_coord_state`](@ref) resolved for this
term — a reference frequency, a per-segment normalization, whatever the term's
coordinate needs. Required for any term declaring `:Frequency` in
[`term_axes`](@ref); there is deliberately no generic fallback —
`plan_parameters` calls this for every term that declares the axis, so a term
that forgets it errors loudly (a `MethodError`) instead of silently evaluating
that axis at zero.

Pointwise in `f`, so the same definition serves the solve grid and any
coordinate off it (see `evaluate_gains`).
"""
function freq_coordinate end

"""
    time_coordinate(term, t, state, seg::Integer) -> Real

`term`'s `x.Ti` at epoch `t` (hours), lying in the SOLVE's time segment `seg`,
against the state [`time_coord_state`](@ref) resolved. Required for any term
declaring `:Ti` in [`term_axes`](@ref); see [`freq_coordinate`](@ref) for why
there is no generic fallback.
"""
function time_coordinate end

"""
    freq_coord_state(term, geom::DataGeometry, fseg_id, nfseg)
    time_coord_state(term, geom::DataGeometry, tseg_id, ntseg)

The constants `term`'s coordinate reads, resolved against the solve geometry
once at plan time and stored on the [`ComponentPlan`](@ref) as `fstate`/`tstate`.
They are not θ parameters (nothing fits them) and not fields of the term (a term
is a user declaration, written before any geometry exists), so each term defines
its own state and a new term needing new constants widens nothing shared.

Required for any term declaring the corresponding axis in [`term_axes`](@ref),
with no generic fallback, for the same reason [`freq_coordinate`](@ref) has
none. An undeclared axis stores `nothing` and needs no method.
"""
function freq_coord_state end

@doc (@doc freq_coord_state)
function time_coord_state end

# Delay uses the physical offset (f − f0) in Hz so θ is a delay in seconds.
freq_coord_state(::Delay, geom::DataGeometry, fseg_id, nfseg) = geom.f0
freq_coordinate(::Delay, f, f0, seg::Integer) = f - f0

# Dispersion uses K·(1/f0 − 1/f) so θ is a differential TEC in TECU. The
# f0-referencing keeps it orthogonal to the constant term at f0 (not globally —
# the delay↔dTEC covariance over a finite band is physical; solvers fit them
# jointly).
freq_coord_state(::Dispersion, geom::DataGeometry, fseg_id, nfseg) = geom.f0
freq_coordinate(::Dispersion, f, f0, seg::Integer) = DISPERSION_K * (1.0 / f0 - 1.0 / f)

# Rate uses (t − t0) in seconds (t given in hours) so θ is a rate in Hz, with t0
# the SEGMENT's own mean epoch rather than a track-wide one.
#
# The origin is where the co-located constant phase lives: phase = φ + 2π·ṙ·(t −
# t0), so φ is the phase at t0. A rate carries an uncertainty σ_ṙ, and quoting
# the constant at an epoch a lever arm Δt away costs it 2π·σ_ṙ·Δt — for a
# milli-hertz-scale σ_ṙ that is a full turn within the hour, so a per-scan
# constant referenced to the far end of a multi-hour track is pure noise. Keeping
# each segment's origin inside the segment holds the lever arm to the segment's
# own extent, where σ_φ = 1/snr is the whole story.
time_coord_state(::Rate, geom::DataGeometry, tseg_id, ntseg) =
    _segment_center(geom.times, tseg_id, ntseg, geom.t0)
time_coordinate(::Rate, t, t0, seg::Integer) = @inbounds (t - t0[seg]) * 3600.0

"""
    PolyNorm(center, scale)

Per-segment centering and scaling of a [`Polynomial`](@ref) term's coordinate:
segment `s` reads `(x − center[s]) / scale[s]`. Fixed by the solve grid, so a
solution evaluated on foreign data keeps the basis it was fit in.
"""
struct PolyNorm
    center::Vector{Float64}
    scale::Vector{Float64}
end

freq_coord_state(::Polynomial{:Frequency}, geom::DataGeometry, fseg_id, nfseg) =
    _poly_norm(geom.channel_freqs, fseg_id, nfseg)
time_coord_state(::Polynomial{:Ti}, geom::DataGeometry, tseg_id, ntseg) =
    _poly_norm(geom.times, tseg_id, ntseg)

# A polynomial uses its segment's centered/scaled coordinate in ~[-1, 1] so the
# basis is well conditioned; the same construction serves either axis, which is
# why one `Polynomial` term covers both.
freq_coordinate(::Polynomial{:Frequency}, f, st, seg::Integer) =
    @inbounds (f - st.center[seg]) / st.scale[seg]
time_coordinate(::Polynomial{:Ti}, t, st, seg::Integer) =
    @inbounds (t - st.center[seg]) / st.scale[seg]

# Center each segment on the mean of its coordinates and scale by the widest
# excursion from it. A single-sample segment has zero spread; its scale is
# REPLACED by 1 rather than floored — the coordinate is then identically zero
# either way, and a floor in physical units would mean nothing shared between a
# frequency axis (Hz) and a time axis (hours).
function _poly_norm(coords::AbstractVector{<:Real}, ids::AbstractVector{<:Integer}, nseg::Integer)
    center = _segment_center(coords, ids, nseg, 0.0)
    scale = zeros(Float64, nseg)
    for i in eachindex(ids, coords)
        scale[ids[i]] = max(scale[ids[i]], abs(coords[i] - center[ids[i]]))
    end
    for s in eachindex(scale)
        scale[s] > 0 || (scale[s] = 1.0)
    end
    return PolyNorm(center, scale)
end

# Mean of each segment's coordinates. A segment the solve grid never populates
# has no mean of its own and takes `empty`, so the coordinate it hands a term
# stays finite.
function _segment_center(
        coords::AbstractVector{<:Real}, ids::AbstractVector{<:Integer},
        nseg::Integer, empty::Real,
    )
    sums = zeros(Float64, nseg)
    cnt = zeros(Int, nseg)
    for i in eachindex(ids, coords)
        sums[ids[i]] += coords[i]
        cnt[ids[i]] += 1
    end
    return [cnt[s] > 0 ? sums[s] / cnt[s] : Float64(empty) for s in 1:nseg]
end

# ── Pure scalar evaluation ───────────────────────────────────────────────────

"""
    term_eval(term, p, x)

`term`'s contribution to phase / log-amplitude at one (channel, time) cell.
`p` holds the term's own parameters, named and shaped as [`param_shapes`](@ref)
declares — a scalar per `()` name, a vector view per `(n,)` name. `x` holds
the coordinates [`term_axes`](@ref) declares, under those names: `x.Frequency`,
`x.Ti`, or both. A term never sees the global parameter vector, nor the
layout that addresses it, nor an index into either.

This is the hot path of `evaluate_gains` — kept allocation-free and
type-stable so the whole forward map is inferrable (and Reactant-traceable).
Read `p` and `x` by field name; iterating a `NamedTuple` is not type-stable.
"""
function term_eval end

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

"""
    term_label(term) -> String

A short diagnostic label for `term`, used in `show` and summaries. Defaults
to the type name; override only for a more evocative label than the type
itself provides.
"""
term_label(t::AbstractGainTerm) = string(nameof(typeof(t)))
term_label(::ConstantTerm) = "const"
term_label(::Delay) = "delay"
term_label(::Dispersion) = "dtec"
term_label(::Rate) = "rate"
term_label(t::Polynomial{:Frequency}) = "polyf$(t.degree)"
term_label(t::Polynomial{:Ti}) = "polyt$(t.degree)"
