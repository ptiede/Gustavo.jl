# ── Frequency-shape specs for one real observable track ──────────────────────
#
# A shape spec is a small value type stating the frequency-shape assumption for
# ONE real observable track of one (station, feed, spw) — log-amplitude or
# unwrapped phase. Its per-track consumption is [`fit_track`](@ref): a
# penalized/projected weighted least-squares (or, for `ARShape`, a Gaussian-
# process posterior mean) of the track against that assumption.
#
# The specs are ordered by how much they assume: `FreeShape` constrains nothing
# and estimates nothing it has no data for; `PolynomialShape` restricts the track
# to a low-order basis; `WhittakerShape` penalizes roughness without assuming a
# shape; `ARShape` puts a stationary Ornstein–Uhlenbeck (Matérn-1/2) process
# prior on it, with a physical correlation bandwidth. The last three all estimate
# the segments they have no data for, from the ones they do.
#
# Adding a model is one struct plus one `fit_track` method.

abstract type AbstractShapeSpec end

# A segment carries usable data iff its value is finite and its weight is a
# positive, finite inverse variance.
_shape_usable(yk, wk) = isfinite(yk) && isfinite(wk) && wk > 0

# Ridge floor for the least-squares shape fits. The polynomial basis is built on
# a coordinate rescaled onto [-1, 1] and the Whittaker system carries a data
# term, so this only keeps a degenerate track (a coordinate repeated across
# segments, or a roughness null space no observation reaches) off a rank-
# deficient QR.
const _SHAPE_RIDGE = 1.0e-6

"""
    fit_track(spec::AbstractShapeSpec, y, w, x) -> ŷ

Fit one real observable track `y` — log-amplitude or unwrapped phase, one entry
per frequency segment — under the shape assumption `spec`. `w` holds the
per-segment inverse-variance weights (`w ≤ 0` or non-finite ⇒ that segment
carries no data, as does a non-finite `y`) and `x` the segment coordinates
(frequencies, in any unit as long as a spec's own scale parameters share it).
Irregular spacing and gaps are allowed.

Returns the fitted track, `NaN` at segments the spec cannot estimate. A track
with no usable segment at all is `NaN` throughout for every spec.
"""
function fit_track end

"""
    FreeShape()

No shape constraint: each frequency segment keeps its own estimate. Follows the
data wherever it has any, and estimates nothing where it has none — the
`lambda → 0` / `degree → ∞` limit of the other specs.
"""
struct FreeShape <: AbstractShapeSpec end

function fit_track(::FreeShape, y, w, x)
    Base.require_one_based_indexing(y, w, x)
    T = float(promote_type(eltype(y), eltype(w)))
    return T[_shape_usable(y[k], w[k]) ? y[k] : T(NaN) for k in eachindex(y, w, x)]
end

"""
    PolynomialShape(degree = 4)

Polynomial of `degree` in the track's frequency coordinate, fit by weighted least
squares and evaluated at every segment — so gaps are estimated by the fit.
Assumes the track is a smooth low-order curve across the band (passband
roll-off, gentle instrumental phase); a high degree can ring (Runge) at the
edges. The degree is lowered to one less than the number of usable segments
where the data cannot support it.
"""
struct PolynomialShape <: AbstractShapeSpec
    degree::Int
    function PolynomialShape(degree::Integer = 4)
        degree >= 1 || throw(
            ArgumentError("PolynomialShape: degree must be ≥ 1, got $degree"),
        )
        return new(degree)
    end
end

# The fit coordinate: `x` shifted and scaled onto [-1, 1], so the polynomial
# basis stays conditioned whatever the absolute frequency (Hz-scale coordinates
# would otherwise raise ~1e11 to the degree).
function _shape_coordinate(T::Type, x)
    lo, hi = extrema(x)
    c = (T(lo) + T(hi)) / 2
    s = (T(hi) - T(lo)) / 2
    s > 0 || return fill(zero(T), length(x))
    return T[(T(xk) - c) / s for xk in x]
end

function fit_track(spec::PolynomialShape, y, w, x)
    Base.require_one_based_indexing(y, w, x)
    T = float(promote_type(eltype(y), eltype(w), eltype(x)))
    obs = [k for k in eachindex(y, w, x) if _shape_usable(y[k], w[k])]
    isempty(obs) && return fill(T(NaN), length(y))
    xs = _shape_coordinate(T, x)
    # A degree of `nobs` or more is not identified by the data: the ridge alone
    # would pick among the fits interpolating it, extrapolating an arbitrary
    # slope across the unobserved segments. One usable segment supports only a
    # constant.
    deg = clamp(spec.degree, 0, length(obs) - 1)
    B = T[xs[k]^j for k in eachindex(xs), j in 0:deg]
    coef = weighted_regularized_least_squares(
        B[obs, :], T[y[k] for k in obs], T[w[k] for k in obs], fill(T(_SHAPE_RIDGE), deg + 1),
    )
    return B * coef
end

"""
    WhittakerShape(lambda = 1.0)

Roughness-penalized track (a Whittaker smoother): minimizes
`Σ w_k(ŷ_k − y_k)² + λ̄·Σ(ŷ_{k−1} − 2ŷ_k + ŷ_{k+1})²`, where `λ̄` is `lambda`
scaled by the median positive weight so the strength is data-relative. Makes NO
shape assumption — it follows real structure where the weights support it and
interpolates gaps smoothly where they do not. `lambda → 0` ⇒ [`FreeShape`](@ref);
large `lambda` ⇒ a straight line (the second-difference null space).

The penalty is on segment-to-segment differences, so it measures roughness per
SEGMENT, not per unit frequency: an irregular segmentation is penalized as if it
were uniform.
"""
struct WhittakerShape <: AbstractShapeSpec
    lambda::Float64
    function WhittakerShape(lambda::Real = 1.0)
        lambda >= 0 || throw(
            ArgumentError("WhittakerShape: lambda must be ≥ 0, got $lambda"),
        )
        return new(lambda)
    end
end

function fit_track(spec::WhittakerShape, y, w, x)
    Base.require_one_based_indexing(y, w, x)
    T = float(promote_type(eltype(y), eltype(w)))
    any(k -> _shape_usable(y[k], w[k]), eachindex(y, w, x)) ||
        return fill(T(NaN), length(y))
    # Fewer than three segments span no second difference, so there is no
    # roughness to penalize and the fit is the data itself.
    (spec.lambda > 0 && length(y) >= 3) || return fit_track(FreeShape(), y, w, x)
    return _whittaker_track(y, w, spec.lambda, _SHAPE_RIDGE)
end

# The Whittaker system: identity design, the roughness term stacked as `√λ̄·D`
# (`D` the 2nd-difference operator) and the ridge as `√ridge·I`. A segment with
# no data carries weight 0, so the penalty alone sets it — that is the
# interpolation. `λ` is scaled by the median positive weight to stay
# data-relative.
function _whittaker_track(y, w, lambda::Real, ridge::Real)
    T = float(promote_type(eltype(y), eltype(w)))
    n = length(y)
    pos = T[T(w[k]) for k in eachindex(y, w) if _shape_usable(y[k], w[k])]
    λ = T(lambda) * (isempty(pos) ? one(T) : T(median(pos)))
    # Allocated off `y`, so the fit stays in the caller's array type.
    wi = similar(y, T, n)
    yi = similar(y, T, n)
    for k in eachindex(y, w)
        ok = _shape_usable(y[k], w[k])
        wi[k] = ok ? T(w[k]) : zero(T)
        yi[k] = ok ? T(y[k]) : zero(T)
    end
    A = similar(y, T, (n, n))
    fill!(A, zero(T))
    # The penalty rows stacked in one operator: √ridge·I over √λ̄·D.
    R = similar(y, T, (2n - 2, n))
    fill!(R, zero(T))
    sr = sqrt(T(ridge))
    sλ = sqrt(λ)
    for k in 1:n
        A[k, k] = one(T)
        R[k, k] = sr
    end
    for i in 1:(n - 2)                                  # 2nd-difference rows [1, −2, 1]
        R[n + i, i] = sλ; R[n + i, i + 1] = -2 * sλ; R[n + i, i + 2] = sλ
    end
    return weighted_regularized_least_squares(A, yi, wi, R)
end

"""
    ARShape(bandwidth; sigma = :auto, fit_hypers = true)

Ornstein–Uhlenbeck (Matérn-1/2) process prior along frequency — an AR(1) track
with correlation function `σ²·exp(-|Δν|/bandwidth)`. The fit is the exact GP
posterior mean, computed in `O(nseg)` by the Kalman filter + RTS smoother of
[`smooth_ou_track`](@ref) with the segment frequencies as the sample coordinate,
so gaps are interpolated and irregular spacing costs nothing.

`bandwidth` is the correlation bandwidth in the units of `fit_track`'s `x`
(Hz for channel frequencies). `sigma` is the prior standard deviation of the
track about its mean; `:auto` seeds it from the track's own scatter in excess of
its noise. With `fit_hypers` (the default) both are then refit per track by
maximizing the Kalman marginal likelihood ([`fit_ou_hypers`](@ref)), with
`bandwidth` confined to between one segment spacing and ten times the fitted
span.

Unlike [`WhittakerShape`](@ref) this is stationary and mean-reverting with a
physical scale, and it measures smoothness per unit FREQUENCY rather than per
segment.
"""
struct ARShape <: AbstractShapeSpec
    bandwidth::Float64
    sigma::Union{Float64, Symbol}
    fit_hypers::Bool
    function ARShape(bandwidth::Real; sigma = :auto, fit_hypers::Bool = true)
        bandwidth > 0 || throw(
            ArgumentError(
                "ARShape: bandwidth must be > 0 (the correlation bandwidth, in the units " *
                    "of the frequency coordinate), got $bandwidth",
            ),
        )
        (sigma === :auto || (sigma isa Real && sigma > 0)) || throw(
            ArgumentError(
                "ARShape: sigma must be > 0 (the prior track standard deviation) or " *
                    ":auto to seed it from the track scatter, got $sigma",
            ),
        )
        return new(bandwidth, sigma === :auto ? :auto : Float64(sigma), fit_hypers)
    end
end

function fit_track(spec::ARShape, y, w, x)
    Base.require_one_based_indexing(y, w, x)
    T = float(promote_type(eltype(y), eltype(w), eltype(x)))
    any(k -> _shape_usable(y[k], w[k]), eachindex(y, w, x)) ||
        return fill(T(NaN), length(y))
    ν_lo, ν_hi = _ou_tau_bounds(x)
    # The OU prior reverts to zero, so the track is centred for the solve and its
    # weighted mean restored afterwards.
    m, yc, ν, σ2 = _track_ou_hypers(
        y, w, x;
        τ0 = spec.bandwidth, τ_lo = ν_lo, τ_hi = ν_hi, fit = spec.fit_hypers,
        σ2_0 = spec.sigma === :auto ? nothing : T(spec.sigma)^2,
    )
    return smooth_ou_track(yc, w, x; τ = ν, σ2 = σ2) .+ m
end

"""
    fit_track_group(spec::AbstractShapeSpec, ys, ws, xs) -> Vector

Fit a GROUP of tracks that share one shape assumption but not one level — the
per-spw pieces of one (station, feed) response. `ys`, `ws` and `xs` hold one
[`fit_track`](@ref) argument triple per member; the result holds one fitted track
per member, in the order given, `NaN` wherever `fit_track` would put it.

The members are independent in everything the shape does not tie together: each
keeps its own free level, so a genuine discontinuity between them is represented
exactly, never smoothed across.

The default fits each member on its own, which is exact for every spec whose shape
parameters are SUPPLIED rather than estimated ([`FreeShape`](@ref),
[`PolynomialShape`](@ref), [`WhittakerShape`](@ref)). [`ARShape`](@ref) overrides
it, because its correlation bandwidth and prior scatter are estimated from the
data and one member frequently cannot determine them.

A new spec needs no method here unless it estimates something from the data.
"""
function fit_track_group end

_fit_tracks_independently(spec::AbstractShapeSpec, ys, ws, xs) =
    [fit_track(spec, ys[i], ws[i], xs[i]) for i in eachindex(ys, ws, xs)]

fit_track_group(spec::AbstractShapeSpec, ys, ws, xs) = _fit_tracks_independently(spec, ys, ws, xs)

# One `(bandwidth, σ²)` for the whole group, then a per-member posterior under it.
#
# The correlation bandwidth is a property of the instrument's frequency response,
# not of the individual window it is measured in, so estimating it once from every
# window is both the physically right assumption and the numerically stable one: a
# single 64-channel window at low SNR routinely cannot separate real ripple from
# noise, and the per-window ML then drives σ² to its floor and returns that window's
# mean — a track that is flat because it was starved, indistinguishable from one
# that is flat because the instrument is.
function fit_track_group(spec::ARShape, ys, ws, xs)
    # With the hypers given rather than fitted there is nothing to pool: every
    # member is already fit under the same pair.
    spec.fit_hypers || return _fit_tracks_independently(spec, ys, ws, xs)
    T = float(
        promote_type(
            (eltype(y) for y in ys)..., (eltype(w) for w in ws)..., (eltype(x) for x in xs)...,
        ),
    )
    out = [fill(T(NaN), length(y)) for y in ys]
    usable = [
        i for i in eachindex(ys, ws, xs)
            if any(k -> _shape_usable(ys[i][k], ws[i][k]), eachindex(ys[i], ws[i], xs[i]))
    ]
    isempty(usable) && return out

    # The OU prior reverts to zero, so each member is centred on its OWN weighted
    # mean for the solve and that mean is restored afterwards — this is what leaves
    # the levels free while the shape is shared.
    ms = [_weighted_mean_finite(ys[i], ws[i]) for i in usable]
    ycs = [
        [_shape_usable(ys[i][k], ws[i][k]) ? T(ys[i][k]) - ms[j] : T(NaN) for k in eachindex(ys[i], ws[i])]
            for (j, i) in enumerate(usable)
    ]
    wus = [ws[i] for i in usable]
    xus = [xs[i] for i in usable]
    ν_lo, ν_hi = _group_ou_tau_bounds(xus)
    # The scatter seed is read off the centred members together, so it too is
    # informed by every window rather than by one.
    σ2_seed = spec.sigma === :auto ?
        _init_track_var(reduce(vcat, ycs), reduce(vcat, wus)) : T(spec.sigma)^2
    ν, σ2 = fit_ou_hypers_pooled(
        ycs, wus, xus; τ0 = spec.bandwidth, σ2_0 = σ2_seed, τ_lo = ν_lo, τ_hi = ν_hi,
    )
    for (j, i) in enumerate(usable)
        out[i] = smooth_ou_track(ycs[j], wus[j], xus[j]; τ = ν, σ2 = σ2) .+ ms[j]
    end
    return out
end
