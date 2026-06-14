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

abstract type AbstractGainTerm end

"Constant offset: phase/log-amp = θ₁. (Per-block constant — a fringe phase, or a flat gain.)"
struct ConstantTerm <: AbstractGainTerm end

"Group delay: phase = 2π·θ₁·(f − f0), θ₁ in seconds."
struct Delay <: AbstractGainTerm end

"Fringe rate: phase = 2π·θ₁·(t − t0)·3600, θ₁ in Hz (t in hours)."
struct Rate <: AbstractGainTerm end

"Polynomial in a segment-scaled frequency coordinate: Σ_{d=1}^{degree} θ_d · xf^d."
struct PolynomialFreq <: AbstractGainTerm
    degree::Int
    function PolynomialFreq(degree::Integer)
        degree >= 1 || error("PolynomialFreq degree must be ≥ 1")
        return new(Int(degree))
    end
end

"Polynomial in a segment-scaled time coordinate: Σ_{d=1}^{degree} θ_d · xt^d."
struct PolynomialTime <: AbstractGainTerm
    degree::Int
    function PolynomialTime(degree::Integer)
        degree >= 1 || error("PolynomialTime degree must be ≥ 1")
        return new(Int(degree))
    end
end

"One free parameter per channel within the frequency segment (the classic bandpass)."
struct PerChannel <: AbstractGainTerm end

# ── Coordinate kind ──────────────────────────────────────────────────────────
# Tells the layout builder which precomputed coordinate axis a term reads.
#   :none       — no coordinate (ConstantTerm)
#   :freq       — a per-channel frequency coordinate (Delay, PolynomialFreq)
#   :time       — a per-time coordinate (Rate, PolynomialTime)
#   :perchannel — indexes its parameter block by local channel position
@enum CoordKind COORD_NONE COORD_FREQ COORD_TIME COORD_PERCHANNEL

coord_kind(::ConstantTerm) = COORD_NONE
coord_kind(::Delay) = COORD_FREQ
coord_kind(::PolynomialFreq) = COORD_FREQ
coord_kind(::Rate) = COORD_TIME
coord_kind(::PolynomialTime) = COORD_TIME
coord_kind(::PerChannel) = COORD_PERCHANNEL

# ── Parameter count per (time-segment, frequency-segment) block ──────────────
# `nchan_seg` is the number of channels in the term's frequency segment; only
# `PerChannel` depends on it.
nparams_per_block(::ConstantTerm, nchan_seg) = 1
nparams_per_block(::Delay, nchan_seg) = 1
nparams_per_block(::Rate, nchan_seg) = 1
nparams_per_block(t::PolynomialFreq, nchan_seg) = t.degree
nparams_per_block(t::PolynomialTime, nchan_seg) = t.degree
nparams_per_block(::PerChannel, nchan_seg) = nchan_seg

# ── Frequency-coordinate builders ────────────────────────────────────────────
# Build, for each global channel, the coordinate a freq-dependent term reads.
# `fseg_groups` is the list of channel-index groups (one per frequency segment).

# Delay uses the physical offset (f − f0) in Hz so θ is a delay in seconds.
function freq_coordinate(::Delay, channel_freqs, fseg_groups, f0)
    return Float64.(channel_freqs) .- f0
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

# NOTE: there is deliberately NO generic `freq_coordinate(::AbstractGainTerm, …)`
# fallback. `plan_parameters` calls `freq_coordinate` only for terms whose
# `coord_kind` is `COORD_FREQ`, so a new frequency-dependent term that forgets to
# define this method errors loudly instead of silently evaluating with xf = 0.

# ── Time-coordinate builders ─────────────────────────────────────────────────
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

# NOTE: no generic `time_coordinate(::AbstractGainTerm, …)` fallback either, for
# the same reason — `plan_parameters` only calls it for `COORD_TIME` terms.

# ── Pure scalar evaluation ───────────────────────────────────────────────────
# `θ` is the full parameter vector; `off` the 1-based start of this block;
# `xf`/`xt` the precomputed coordinate scalars for this channel/time; `clocal`
# the channel's 1-based position within its frequency segment (for PerChannel).
# These are the hot path of `evaluate_gains` — kept allocation-free and
# type-stable so the whole forward map is inferrable (and Reactant-traceable).
@inline term_eval(::ConstantTerm, θ, off, xf, xt, clocal) = @inbounds θ[off]
@inline term_eval(::Delay, θ, off, xf, xt, clocal) = @inbounds 2π * θ[off] * xf
@inline term_eval(::Rate, θ, off, xf, xt, clocal) = @inbounds 2π * θ[off] * xt
@inline term_eval(::PerChannel, θ, off, xf, xt, clocal) = @inbounds θ[off + clocal - 1]

@inline function term_eval(t::PolynomialFreq, θ, off, xf, xt, clocal)
    v = zero(eltype(θ))
    p = xf
    @inbounds for d in 1:t.degree
        v += θ[off + d - 1] * p
        p *= xf
    end
    return v
end

@inline function term_eval(t::PolynomialTime, θ, off, xf, xt, clocal)
    v = zero(eltype(θ))
    p = xt
    @inbounds for d in 1:t.degree
        v += θ[off + d - 1] * p
        p *= xt
    end
    return v
end

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
term_label(::Rate) = "rate"
term_label(t::PolynomialFreq) = "polyf$(t.degree)"
term_label(t::PolynomialTime) = "polyt$(t.degree)"
term_label(::PerChannel) = "perchan"
