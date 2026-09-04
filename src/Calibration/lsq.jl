function design_matrices(bl_pairs, nant)
    nbl = length(bl_pairs)
    A_amp = zeros(Float64, nbl, nant)
    A_phase = zeros(Float64, nbl, nant)
    for (i, (a, b)) in enumerate(bl_pairs)
        A_amp[i, a] = 1.0
        A_amp[i, b] = 1.0
        A_phase[i, a] = 1.0
        A_phase[i, b] = -1.0
    end
    return A_amp, A_phase
end

function weighted_phase_mean(phases, weights)
    sin_sum = sum(weights .* sin.(phases))
    cos_sum = sum(weights .* cos.(phases))
    return atan(sin_sum, cos_sum)
end

function weighted_complex_correction(samples, weights)
    isempty(samples) && return nothing
    sum(weights) > 0 || return nothing
    log_amp = sum(weights .* log.(abs.(samples))) / sum(weights)
    phase = weighted_phase_mean(angle.(samples), weights)
    return exp(log_amp) * cis(phase)
end

"""
    savitzky_golay_smooth(y, weights = nothing; window = 11, order = 2) -> Vector

Savitzky–Golay-style smoothing: at each index fit a degree-`order` polynomial by
(optionally weighted) least squares over a centered window of `window` samples
and evaluate it at the centre. Non-finite (`NaN`) samples are skipped in each
local fit (so the smoother also interpolates gaps), and the polynomial order is
reduced where a window has too few finite samples. Returns a new vector.
"""
function savitzky_golay_smooth(y::AbstractVector, weights = nothing; window::Integer = 11, order::Integer = 2)
    n = length(y)
    out = collect(float.(y))
    h = window ÷ 2
    for i in 1:n
        lo, hi = max(1, i - h), min(n, i + h)
        idx = [j for j in lo:hi if isfinite(y[j])]
        isempty(idx) && continue
        ord = min(order, length(idx) - 1)
        x = Float64.(idx .- i)                         # centred coordinate; centre is x = 0
        # Build the Vandermonde directly as a Matrix — `reduce(hcat, ...)` over a
        # single degree-0 column would return a Vector and crash the WLS solve
        # when a window has only one finite sample (ord == 0).
        A = Float64[x[r]^d for r in eachindex(x), d in 0:ord]
        b = Float64.(y[idx])
        w = weights === nothing ? ones(length(idx)) : Float64.(weights[idx])
        coef = weighted_least_squares(A, b, w)
        out[i] = coef[1]                               # value of the local polynomial at the centre
    end
    return out
end

"""
    connected_components(nnodes, edges) -> (compid, ncomp, touched)

Union–find connected components of an undirected graph on `nnodes` nodes given
`edges` (any iterable of `(u, v)` index pairs). Returns `compid::Vector{Int}`
(dense component id `1:ncomp` for each node that appears in an edge, `0` for
isolated/untouched nodes), the component count `ncomp`, and `touched::Vector{Bool}`
marking nodes that appear in at least one edge.

Used by the fringe stationization / adhoc-phasing solvers to gauge each
connected piece of the (station, feed) graph independently (so disconnected
array subgraphs each get their own reference pin).
"""
function connected_components(nnodes::Integer, edges)
    parent = collect(1:nnodes)
    function findroot(x)
        root = x
        while parent[root] != root
            root = parent[root]
        end
        while parent[x] != root           # path compression
            parent[x], x = root, parent[x]
        end
        return root
    end
    touched = falses(nnodes)
    for (u, v) in edges
        touched[u] = true
        touched[v] = true
        ru, rv = findroot(u), findroot(v)
        ru == rv || (parent[ru] = rv)
    end
    compid = zeros(Int, nnodes)
    label = Dict{Int, Int}()
    ncomp = 0
    for n in 1:nnodes
        touched[n] || continue
        r = findroot(n)
        id = get(label, r, 0)
        if id == 0
            ncomp += 1
            label[r] = ncomp
            id = ncomp
        end
        compid[n] = id
    end
    return compid, ncomp, touched
end

# Convention across Gustavo: a "weight" is always an *inverse variance*
# (precision, 1/σ²), matching both the Gaussian-likelihood derivation of
# weighted least squares and the AIPS UVData convention (Memo 117: visibility
# weights are in Jy⁻² = variance⁻¹).
#
# For y_i = A_i x + ε_i with independent ε_i ~ N(0, σ_i²) the MLE is
#   x* = (Aᵀ W A)⁻¹ Aᵀ W y,   W = diag(1/σ_i²).
# To solve this with QR we need S with SᵀS = W; since W is diagonal-positive,
# S = diag(√W_ii) = diag(1/σ_i), so we scale each row by √(weight) before
# handing to QR. **Callers pass inverse variance and never take the sqrt
# themselves.**
_row_scale(inv_variances) = sqrt.(inv_variances)

# Promote the WLS triple `(A, b, inv_variances)` to a single common element
# type. The design matrix `A` is built in Float64 by `design_matrices`, but
# `b` (log-amplitudes / phases) and `inv_variances` (UVData weights) come in
# at Float32 since AIPS Memo 117 stores weights as `1E`. LinearSolve's QR
# `ldiv!` rejects a Float64 factorization against a Float32 RHS, so we have
# to align eltypes up front.
function _promote_lsq(A, b, inv_variances)
    T = promote_type(eltype(A), eltype(b), real(eltype(inv_variances)))
    return convert(AbstractMatrix{T}, A), convert(AbstractVector{T}, b),
        convert(AbstractVector{T}, inv_variances)
end

function weighted_least_squares(A, b, inv_variances)
    A, b, inv_variances = _promote_lsq(A, b, inv_variances)
    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    return solve(LinearProblem(Aw, bw), QRFactorization()).u
end

"""
    weighted_regularized_least_squares(A, b, inv_variances, penalties)

Ridge-regularized WLS: solve `min_x ‖diag(√inv_variances)(Ax - b)‖² + ‖Rx‖²`
by stacking `R` onto the weighted design matrix (`Rx = 0` as extra
zero-target rows) and delegating to `weighted_least_squares`.

`penalties` is either a vector of per-column ridge weights `λᵢ ≥ 0` (the
diagonal case, giving `R = Diagonal(√λ)`) or an arbitrary penalty matrix `R`
with `size(R, 2) == size(A, 2)` — e.g. a (scaled) difference operator for
roughness penalties. A vector of all-nonpositive entries (no columns
penalized) short-circuits to the unregularized solve.
"""
function weighted_regularized_least_squares(A, b, inv_variances, penalties)
    isempty(penalties) && return weighted_least_squares(A, b, inv_variances)
    penalties isa AbstractVector && all(≤(0), penalties) && return weighted_least_squares(A, b, inv_variances)

    A, b, inv_variances = _promote_lsq(A, b, inv_variances)
    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    Areg = penalties isa AbstractVector ? Diagonal(sqrt.(penalties)) : penalties
    breg = zeros(eltype(Aw), size(Areg, 1))
    return solve(LinearProblem(vcat(Aw, Areg), vcat(bw, breg)), QRFactorization()).u
end

"""
    WLSEstimator(observation_model, penalty = nothing)

A weighted-least-squares estimator reified as a value: `observation_model`
is a callable that, given whatever domain-specific data a fit needs, builds
and returns `(A, b, inv_variances)`; `penalty` regularizes that system —
`nothing` (unregularized), a per-column ridge vector, an arbitrary penalty
matrix, or a function of `A` computing one of those (for penalties whose
shape depends on the system size, e.g. one ridge entry per column).

Calling `est(args...)` runs `observation_model(args...)` then dispatches to
`weighted_least_squares` or [`weighted_regularized_least_squares`](@ref)
according to `penalty`.
"""
struct WLSEstimator{O, P}
    observation_model::O
    penalty::P
end
WLSEstimator(observation_model) = WLSEstimator(observation_model, nothing)

function (est::WLSEstimator)(args...)
    A, b, inv_variances = est.observation_model(args...)
    est.penalty === nothing && return weighted_least_squares(A, b, inv_variances)
    penalty = est.penalty isa Function ? est.penalty(A) : est.penalty
    return weighted_regularized_least_squares(A, b, inv_variances, penalty)
end

function weighted_constrained_least_squares(A, b, inv_variances, C, d; constraint_weight = 1.0e6)
    isempty(C) && return weighted_least_squares(A, b, inv_variances)

    A, b, inv_variances = _promote_lsq(A, b, inv_variances)
    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    Acon = constraint_weight .* C
    bcon = constraint_weight .* d
    return solve(LinearProblem(vcat(Aw, Acon), vcat(bw, bcon)), QRFactorization()).u
end

# Inverse-variance weight of the increment between samples `k` and `k+1`: the
# precision of a difference of two independent estimates, `1/(1/w_k + 1/w_{k+1})`.
# A missing/non-positive endpoint weight makes the increment uninformative.
function _increment_weight(weights, k)
    weights === nothing && return 1.0
    wa, wb = weights[k], weights[k + 1]
    (isfinite(wa) && isfinite(wb) && wa > 0 && wb > 0) || return 0.0
    return wa * wb / (wa + wb)
end

# Where a track's step-to-step increments CENTER: the weighted circular mean of the
# wrapped increments between ADJACENT finite samples, which is also the track's
# dominant per-sample trend. Only strictly adjacent pairs contribute — across a gap
# the true increment is unknown modulo 2π, so a gap-spanning pair says nothing about
# either.
#
# `angle` returns the mean in (-π, π], the range a sampled phase can resolve at all:
# a trend steeper than π per sample is aliased in the data itself.
function _increment_center(phases, weights)
    T = float(eltype(phases))
    acc = zero(Complex{T})
    for k in firstindex(phases):(lastindex(phases) - 1)
        (isfinite(phases[k]) && isfinite(phases[k + 1])) || continue
        wk = _increment_weight(weights, k)
        wk > 0 || continue
        acc += wk * cis(T(phases[k + 1]) - T(phases[k]))
    end
    return iszero(acc) ? zero(T) : T(angle(acc))
end

"""
    unwrap_phase_track(phases; weights=nothing) -> Vector

Unwrap a phase track to remove ±2π discontinuities between adjacent finite
samples, moving samples only by whole multiples of 2π: no trend is
estimated, removed, or restored. The walk seeds from `argmax` of the finite
weights when `weights` is given, else the first finite phase.

A step is resolved only while the true increment plus its noise stays inside
±π, so a track carrying a steep trend (a group delay along frequency, a
fringe rate along time) is walked onto the wrong branch systematically —
fit the trend out first. Noise puts individual steps over the boundary at
random, accumulating 2π errors that a subsequent smooth fit reports as a
large trend; [`phase_unwrap_ambiguity`](@ref) detects that, so consult it
before trusting an unwrapped track.

The seed is an algorithmic anchor only; downstream gauge code should fix the
phase gauge itself rather than rely on the unwrap reference.
"""
function unwrap_phase_track(phases; weights = nothing)
    Base.require_one_based_indexing(phases)
    weights === nothing || Base.require_one_based_indexing(weights)
    unwrapped = copy(phases)
    n = length(unwrapped)
    n == 0 && return unwrapped

    finite = isfinite.(unwrapped)
    any(finite) || return unwrapped

    ref_idx = if weights === nothing
        findfirst(finite)
    else
        @assert length(weights) == n "weights length must match phases length"
        # Choose the highest-weight finite channel as the unwrap seed.
        best_w = -Inf
        best_i = findfirst(finite)
        @inbounds for i in 1:n
            (finite[i] && isfinite(weights[i])) || continue
            if weights[i] > best_w
                best_w = weights[i]
                best_i = i
            end
        end
        best_i
    end
    isnothing(ref_idx) && return unwrapped

    last = unwrapped[ref_idx]
    for i in (ref_idx + 1):n
        isfinite(unwrapped[i]) || continue
        unwrapped[i] += 2π * round((last - unwrapped[i]) / (2π))
        last = unwrapped[i]
    end

    last = unwrapped[ref_idx]
    for i in (ref_idx - 1):-1:1
        isfinite(unwrapped[i]) || continue
        unwrapped[i] += 2π * round((last - unwrapped[i]) / (2π))
        last = unwrapped[i]
    end

    return unwrapped
end

"""
    phase_unwrap_ambiguity(phases; weights=nothing) -> Float64

How far a phase track's step-to-step increments SCATTER: the fraction of adjacent
finite pairs whose wrapped increment lies more than π/2 from the increments' own
weighted circular mean. A pure measurement — the track is read, never modified.

Scatter is what decides whether [`unwrap_phase_track`](@ref)'s walk is meaningful.
0 means every step falls in a tight cluster and the branch it picks is determined;
as the fraction grows the walk degenerates into a random walk whose accumulated 2π
errors look, to any subsequent smooth fit, like a large genuine trend. Above
roughly a quarter a track should be treated as carrying no recoverable branch,
rather than as evidence of the trend that fit reports.

The circular mean is a location parameter here, nothing more: it is what makes this
a measure of scatter rather than of the trend the track happens to carry, so a
steadily-trending track reads near 0 however steep its trend. Steepness is a
separate failure of the walk (increments approaching ±π go onto the wrong branch
systematically) and this statistic does not report it — fit the trend out first if
a caller can carry one that large.

Returns 0 for a track with no adjacent finite pair to compare.
"""
function phase_unwrap_ambiguity(phases; weights = nothing)
    Base.require_one_based_indexing(phases)
    weights === nothing || Base.require_one_based_indexing(weights)
    center = _increment_center(phases, weights)
    nstep = 0
    namb = 0
    for k in firstindex(phases):(lastindex(phases) - 1)
        (isfinite(phases[k]) && isfinite(phases[k + 1])) || continue
        _increment_weight(weights, k) > 0 || continue
        nstep += 1
        abs(rem2pi(phases[k + 1] - phases[k] - center, RoundNearest)) > π / 2 && (namb += 1)
    end
    return nstep == 0 ? 0.0 : namb / nstep
end
