# ── Atomic gain terms ────────────────────────────────────────────────────────
#
# Every term is LINEAR in its parameters θ in phase / log-amplitude space. That
# single property is what unifies the bandpass solver and the fringe fitter:
# both reduce to weighted least squares on the same `basis_columns`, while the
# pure forward map `term_value` (θ → phase contribution) is the Reactant-facing
# primitive a future Comrade global solver will trace through.
#
# A term lives inside a `GainComponent` that pins it to one (time-segment,
# frequency-segment) block of parameters. Terms therefore never see the global
# parameter vector — they are handed a coordinate scalar and a parameter view.

"""
    AbstractGainTerm

Base type for an atomic gain term (a phase- or log-amplitude contribution). A
term is deliberately the ONE extension point of the whole calibration/fringe
stack: define a new term and it works everywhere — the flat-θ `GainEvaluator`,
the per-site `GainPlan` forward map, AND the Enzyme/ForwardDiff gradient
(`Gustavo.Solve`) — with no solver changes and no AD rule, because the optimizer
differentiates straight through your `term_eval`.

Adding a term is two steps (qualify the methods, e.g. `import Gustavo.Calibration as C`):

1. Subtype a FAMILY (which fixes the coordinate the term reads — so you never
   declare that separately):
   - `ScalarTerm`    — no coordinate (a constant offset).
   - `FrequencyTerm` — a per-channel frequency coordinate (delay, dispersion, …).
   - `TimeTerm`      — a per-time coordinate (rate, …).
   - `ChannelTerm`   — one parameter per channel in the segment (a bandpass).

2. Define ONE method, `term_eval` — the pure, differentiable primitive. `p` is
   the block's parameters, read by position (`p[1]`, `p[2]`) or, if you declare
   `param_names`, by name (`p.τ`):
   - `C.term_eval(::T, p)`     for a `ScalarTerm`      → e.g. `p[1]`.
   - `C.term_eval(::T, p, x)`  for `FrequencyTerm`/`TimeTerm`, `x` the coordinate
      scalar (default `ν − ν0` / `t − t0`) → e.g. `2π * p[1] * x`.
   - `C.term_eval(::T, p, k)`  for a `ChannelTerm`, `k` the local channel index.

   Keep it linear in `p`, allocation-free, and branch only on integers/coordinates
   (never on `p`), so the forward map stays inferrable and AD-clean.

OPTIONAL overrides (sensible defaults provided, so most terms skip these):
- `C.param_names(::T) = (:τ, :dtec)` — name a multi-parameter term's parameters.
  This is the SINGLE source of truth: the per-block count is derived from it AND
  you read the parameters by name (`p.τ`), while positional `p[k]` becomes
  bounds-checked against it. A single unnamed parameter (the default) needs
  nothing — just use `p[1]`.
- `C.nparams_per_block(::T, nchan_seg)` — a VARIABLE parameter count not fixed by
  names (a polynomial degree, a per-channel count); defaults to
  `max(1, length(param_names))`.
- `C.freq_coordinate(::T, channel_freqs, fseg_groups, f0)` /
  `C.time_coordinate(::T, times, tseg_groups, t0)` — override the coordinate `x`
  (defaults: `ν − ν0` for `FrequencyTerm`, `t − t0` hours for `TimeTerm`).
- `C.basis_columns`, `C.term_label` — legacy WLS solvers / diagnostics only.

So a new curved delay `phase = 2π·τ·(ν − ν0)²` is just:
`struct QuadraticDelay <: FrequencyTerm end` +
`param_names(::QuadraticDelay) = (:τ,)` +
`term_eval(::QuadraticDelay, p, x) = 2π * p.τ * x^2`
(or drop `param_names` and write `p[1]`).

See `Delay`/`Rate`/`PerChannel`/`PolynomialFreq` for worked examples.
"""
abstract type AbstractGainTerm end

# Family supertypes: a term's family fixes which coordinate it reads (so a term
# never declares that separately) and supplies the default coordinate. Subtype
# one of these, NOT `AbstractGainTerm` directly (which has no `coord_kind` and so
# errors loudly at plan time — a guard against forgetting to classify a term).
abstract type ScalarTerm <: AbstractGainTerm end
abstract type FrequencyTerm <: AbstractGainTerm end
abstract type TimeTerm <: AbstractGainTerm end
abstract type ChannelTerm <: AbstractGainTerm end

"Constant offset: phase/log-amp = θ₁. (Per-block constant — a fringe phase, or a flat gain.)"
struct ConstantTerm <: ScalarTerm end

"Group delay: phase = 2π·θ₁·(f − f0), θ₁ in seconds."
struct Delay <: FrequencyTerm end

# rad·Hz per TECU (1 TECU = 1e16 el/m²): ionospheric phase = −K·TEC/f.
const DISPERSION_K = 8.4479e9

"""
Ionospheric dispersion: phase = K·θ₁·(1/f0 − 1/f) with K = $DISPERSION_K rad·Hz/TECU,
so θ₁ is a differential TEC in TECU. Station-based and non-magnetic to first
order, so it ties feeds (`SharedFeeds`). Referenced to f0 — the 1/f0 offset
lands in the accompanying constant/phase term, keeping this term pure shape.
"""
struct Dispersion <: FrequencyTerm end

"Fringe rate: phase = 2π·θ₁·(t − t0)·3600, θ₁ in Hz (t in hours)."
struct Rate <: TimeTerm end

"Polynomial in a segment-scaled frequency coordinate: Σ_{d=1}^{degree} θ_d · xf^d."
struct PolynomialFreq <: FrequencyTerm
    degree::Int
    function PolynomialFreq(degree::Integer)
        degree >= 1 || error("PolynomialFreq degree must be ≥ 1")
        return new(Int(degree))
    end
end

"Polynomial in a segment-scaled time coordinate: Σ_{d=1}^{degree} θ_d · xt^d."
struct PolynomialTime <: TimeTerm
    degree::Int
    function PolynomialTime(degree::Integer)
        degree >= 1 || error("PolynomialTime degree must be ≥ 1")
        return new(Int(degree))
    end
end

"One free parameter per channel within the frequency segment (the classic bandpass)."
struct PerChannel <: ChannelTerm end

# ── Coordinate kind (derived from the term family) ───────────────────────────
# Which precomputed coordinate axis a term reads. Defined on the family
# supertypes, so a term inherits it from its `<: FrequencyTerm` / … choice and
# never declares it. `AbstractGainTerm` itself has NO method → subtyping the root
# directly errors loudly at plan time.
@enum CoordKind COORD_NONE COORD_FREQ COORD_TIME COORD_PERCHANNEL

coord_kind(::ScalarTerm) = COORD_NONE
coord_kind(::FrequencyTerm) = COORD_FREQ
coord_kind(::TimeTerm) = COORD_TIME
coord_kind(::ChannelTerm) = COORD_PERCHANNEL

# ── Parameter names + count per block ────────────────────────────────────────
# A term may OPTIONALLY name its parameters: `param_names(::T) = (:τ, :dtec)`.
# When it does, the names are the SINGLE source of truth — the per-block count is
# derived from them (no separate `nparams_per_block` to keep in sync) and the
# term reads them by name (`p.τ`) inside `term_eval`. Terms that don't declare
# names use positional access (`p[1]`, `p[2]`) and default to one parameter, or
# override `nparams_per_block` for a variable count (polynomial degree, channels).
param_names(::AbstractGainTerm) = ()

# `nchan_seg` is the number of channels in the term's frequency segment.
nparams_per_block(t::AbstractGainTerm, nchan_seg) = max(1, length(param_names(t)))
nparams_per_block(t::PolynomialFreq, nchan_seg) = t.degree
nparams_per_block(t::PolynomialTime, nchan_seg) = t.degree
nparams_per_block(::PerChannel, nchan_seg) = nchan_seg

# ── Frequency-coordinate builders ────────────────────────────────────────────
# Build, for each global channel, the coordinate `x` a freq-dependent term reads.
# `fseg_groups` is the list of channel-index groups (one per frequency segment).
# DEFAULT (any `FrequencyTerm`): the physical offset (f − f0) in Hz, so θ is a
# delay in seconds — used by `Delay` and by simple custom terms. Terms needing a
# different coordinate (dispersion, polynomials) override the specific method.
freq_coordinate(::FrequencyTerm, channel_freqs, fseg_groups, f0) =
    Float64.(channel_freqs) .- f0

# Dispersion uses K·(1/f0 − 1/f) so θ is a differential TEC in TECU. The
# f0-referencing keeps it orthogonal to the constant term at f0 (not globally —
# the delay↔dTEC covariance over a finite band is physical; solvers fit them
# jointly).
function freq_coordinate(::Dispersion, channel_freqs, fseg_groups, f0)
    return DISPERSION_K .* (1.0 / f0 .- 1.0 ./ Float64.(channel_freqs))
end

# PolynomialFreq uses a per-segment centered/scaled coordinate in ~[-1, 1] so the
# basis is well conditioned. Center = segment mean; scale = max|f − center|.
function freq_coordinate(t::PolynomialFreq, channel_freqs, fseg_groups, f0)
    x = zeros(Float64, length(channel_freqs))
    for grp in fseg_groups
        isempty(grp) && continue
        fs = @view channel_freqs[grp]
        center = sum(fs) / length(fs)
        scale = maximum(abs.(fs .- center))
        scale = scale > 0 ? scale : 1.0
        @inbounds for c in grp
            x[c] = (channel_freqs[c] - center) / scale
        end
    end
    return x
end

# ── Time-coordinate builders ─────────────────────────────────────────────────
# DEFAULT (any `TimeTerm`): (t − t0) in hours. `Rate` overrides to seconds so its
# θ is a rate in Hz; polynomials override for a segment-scaled coordinate.
time_coordinate(::TimeTerm, times, tseg_groups, t0) = Float64.(times) .- t0

# Rate uses (t − t0) in seconds (t given in hours) so θ is a rate in Hz.
function time_coordinate(::Rate, times, tseg_groups, t0)
    return (Float64.(times) .- t0) .* 3600.0
end

function time_coordinate(t::PolynomialTime, times, tseg_groups, t0)
    x = zeros(Float64, length(times))
    for grp in tseg_groups
        isempty(grp) && continue
        ts = @view times[grp]
        center = sum(ts) / length(ts)
        scale = maximum(abs.(ts .- center))
        scale = scale > 0 ? scale : 1.0
        @inbounds for i in grp
            x[i] = (times[i] - center) / scale
        end
    end
    return x
end

# ── Parameter block view + pure evaluation ───────────────────────────────────
#
# `ParamBlock` is a lightweight, non-allocating handle on one block's parameters
# inside the flat parameter vector (flat-θ path) or the per-site component array
# (`GainPlan` path). It lets each term's `term_eval` read its parameters by
# position (`p[1]`, `p[2]`) OR — when the term declares `param_names` — by name
# (`p.τ`), with no offset arithmetic, while the evaluator hot loop stays
# allocation-free. `Names` (a compile-time tuple of Symbols, possibly empty) is
# the term's `param_names`; when non-empty, positional `p[k]` is bounds-checked
# against it (so `p[3]` on a 2-parameter block errors instead of silently reading
# the next block). Named access `p.τ` resolves to a compile-time-valid index.
struct ParamBlock{Names, A}
    data::A
    off::Int
end
ParamBlock{Names}(data::A, off::Integer) where {Names, A} = ParamBlock{Names, A}(data, Int(off))
ParamBlock(data, off::Integer) = ParamBlock{()}(data, off)

@inline function Base.getindex(p::ParamBlock{Names}, k::Integer) where {Names}
    @boundscheck (length(Names) == 0 || 1 <= k <= length(Names)) || throw(BoundsError(p, k))
    return @inbounds getfield(p, :data)[getfield(p, :off) + Int(k) - 1]
end
@inline function Base.getproperty(p::ParamBlock{Names}, s::Symbol) where {Names}
    (s === :data || s === :off) && return getfield(p, s)
    return p[_name_index(Names, s)]
end
@inline Base.eltype(p::ParamBlock) = eltype(getfield(p, :data))

# Compile-time index of a named parameter (constant-folds when `s` is a literal).
@inline _name_index(names::Tuple, s::Symbol) =
    first(names) === s ? 1 : 1 + _name_index(Base.tail(names), s)
@inline _name_index(::Tuple{}, s::Symbol) = throw(ArgumentError("no parameter named :$s"))

# `term_eval` — the pure, differentiable primitive each term defines. `p` is the
# block's parameters (indexed from 1); `x` the term's coordinate scalar (freq or
# time, per its family); `k` a `ChannelTerm`'s local channel index. These are the
# hot path of `evaluate_gains` — allocation-free, linear in `p`, and branch-free
# in `p`, so the whole forward map is inferrable (and AD- / Reactant-traceable).
@inline term_eval(::ConstantTerm, p) = p[1]
@inline term_eval(::Delay, p, x) = 2π * p[1] * x
@inline term_eval(::Dispersion, p, x) = p[1] * x
@inline term_eval(::Rate, p, x) = 2π * p[1] * x
@inline term_eval(::PerChannel, p, k) = p[k]

@inline function term_eval(t::PolynomialFreq, p, x)
    v = zero(eltype(p))
    xp = x
    for d in 1:t.degree
        v += p[d] * xp
        xp *= x
    end
    return v
end

@inline function term_eval(t::PolynomialTime, p, x)
    v = zero(eltype(p))
    xp = x
    for d in 1:t.degree
        v += p[d] * xp
        xp *= x
    end
    return v
end

# Framework adapter: the evaluator hot loop calls this uniformly on every term
# with the block's backing store `data` and 1-based `off`. It wraps them in a
# `ParamBlock` carrying the term's `param_names` (for named/bounds-checked access)
# and forwards to each family's `term_eval` the one coordinate that family reads
# (`xf`/`xt`) or the local channel index (`k`), keeping the loop term-agnostic.
@inline _paramblock(t, data, off) = ParamBlock{param_names(t)}(data, off)
@inline _term_contribution(t::ScalarTerm, data, off, xf, xt, k) = term_eval(t, _paramblock(t, data, off))
@inline _term_contribution(t::FrequencyTerm, data, off, xf, xt, k) = term_eval(t, _paramblock(t, data, off), xf)
@inline _term_contribution(t::TimeTerm, data, off, xf, xt, k) = term_eval(t, _paramblock(t, data, off), xt)
@inline _term_contribution(t::ChannelTerm, data, off, xf, xt, k) = term_eval(t, _paramblock(t, data, off), k)

# ── Linear design columns (for WLS solvers) ──────────────────────────────────
#
# `basis_columns(term, x)` returns the design columns of a single block as the
# columns of a matrix, given the term's coordinate vector `x` over the samples
# it varies on (frequency samples for freq terms / PerChannel, time samples for
# time terms, and an all-ones placeholder length for ConstantTerm). Each column
# is one parameter. This mirrors the per-block contribution of the old
# `model_basis_columns`; segmentation, demeaning, and rank-trimming are applied
# by the solver, not here.
basis_columns(::ConstantTerm, x::AbstractVector) = reshape(ones(Float64, length(x)), :, 1)
basis_columns(::Delay, x::AbstractVector) = reshape(2π .* Float64.(x), :, 1)
basis_columns(::Dispersion, x::AbstractVector) = reshape(Float64.(x), :, 1)
basis_columns(::Rate, x::AbstractVector) = reshape(2π .* Float64.(x), :, 1)

function basis_columns(t::PolynomialFreq, x::AbstractVector)
    A = Matrix{Float64}(undef, length(x), t.degree)
    @inbounds for d in 1:t.degree, i in eachindex(x)
        A[i, d] = Float64(x[i])^d
    end
    return A
end

function basis_columns(t::PolynomialTime, x::AbstractVector)
    A = Matrix{Float64}(undef, length(x), t.degree)
    @inbounds for d in 1:t.degree, i in eachindex(x)
        A[i, d] = Float64(x[i])^d
    end
    return A
end

basis_columns(::PerChannel, x::AbstractVector) = Matrix{Float64}(I, length(x), length(x))

# Labels for diagnostics / summaries.
term_label(::ConstantTerm) = "const"
term_label(::Delay) = "delay"
term_label(::Dispersion) = "dtec"
term_label(::Rate) = "rate"
term_label(t::PolynomialFreq) = "polyf$(t.degree)"
term_label(t::PolynomialTime) = "polyt$(t.degree)"
term_label(::PerChannel) = "perchan"
