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
    for n in eachindex(touched)
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

# Across Gustavo a "weight" is an inverse variance 1/σ², as in AIPS Memo 117
# (visibility weights in Jy⁻²). QR solves the weighted problem on rows scaled
# by √weight; callers pass the inverse variance and never take the root.
_row_scale(inv_variances) = sqrt.(inv_variances)

# A solve works in the common type of its data (`A`, `b`, the weights), so a
# Float32 system stays Float32. The system is allocated in that type and filled,
# so a tuning quantity such as a penalty is converted on assignment and never
# sets the precision.
_lsq_eltype(A, b, inv_variances) = promote_type(eltype(A), eltype(b), real(eltype(inv_variances)))

# The row-weighted system `(diag(√w)·A, diag(√w)·b)` in `T`, followed by
# `nextra` rows the caller fills.
function _weighted_system(::Type{T}, A, b, inv_variances, nextra) where {T}
    Base.require_one_based_indexing(A, b, inv_variances)
    nr = size(A, 1)
    M = similar(A, T, nr + nextra, size(A, 2))
    y = similar(b, T, nr + nextra)
    sw = _row_scale(inv_variances)
    M[1:nr, :] .= A .* sw
    y[1:nr] .= b .* sw
    return M, y
end

function weighted_least_squares(A, b, inv_variances)
    M, y = _weighted_system(_lsq_eltype(A, b, inv_variances), A, b, inv_variances, 0)
    return solve(LinearProblem(M, y), QRFactorization()).u
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

    R = penalties isa AbstractVector ? Diagonal(sqrt.(penalties)) : penalties
    M, y = _weighted_system(_lsq_eltype(A, b, inv_variances), A, b, inv_variances, size(R, 1))
    rows = (size(A, 1) + 1):size(M, 1)
    M[rows, :] .= R
    y[rows] .= zero(eltype(y))
    return solve(LinearProblem(M, y), QRFactorization()).u
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

"""
    FactoredWLS(A, inv_variances)
    FactoredWLS{T}(A, inv_variances)

The weighted least-squares problem `min_x ‖diag(√inv_variances)(Ax − b)‖²`,
factored once for any number of right-hand sides: `F(b)` returns `x`, a
vector the next call overwrites. Throws if `A` does not determine every
unknown. `T` defaults to the common type of `A` and the weights.
"""
struct FactoredWLS{T, V <: AbstractVector{T}, C}
    sw::V
    cache::C
end

FactoredWLS(A, inv_variances) =
    FactoredWLS{promote_type(eltype(A), real(eltype(inv_variances)))}(A, inv_variances)

function FactoredWLS{T}(A, inv_variances) where {T}
    Base.require_one_based_indexing(A, inv_variances)
    length(inv_variances) == size(A, 1) ||
        throw(DimensionMismatch("$(length(inv_variances)) weights for $(size(A, 1)) rows"))
    sw = convert(AbstractVector{T}, _row_scale(inv_variances))
    cache = init(LinearProblem(A .* sw, zeros(T, size(A, 1))), QRFactorization())
    solve!(cache)  # factors
    _full_rank(cache.cacheval.R) || throw(ArgumentError("the system does not determine every unknown"))
    return FactoredWLS{T, typeof(sw), typeof(cache)}(sw, cache)
end

function (F::FactoredWLS)(b)
    F.cache.b = F.sw .* b
    return solve!(F.cache).u
end

function _full_rank(R)
    d = abs.(diag(R))
    return isempty(d) || minimum(d) > length(d) * eps(eltype(d)) * maximum(d)
end

"""
    ConstrainedWLS(A, inv_variances, C, d = nothing)
    ConstrainedWLS{T}(A, inv_variances, C, d = nothing)

The weighted least-squares problem `min_x ‖diag(√inv_variances)(Ax − b)‖²`
subject to `Cx = d` (`Cx = 0` when `d` is `nothing` or zero), factored once for
any number of right-hand sides: `F(b)` returns `x`.

The constraints hold exactly: with `Cᵀ = [Q₁ Q₂]R`, every feasible `x` is
`x₀ + Q₂z` for the particular solution `x₀ = Q₁R⁻ᵀd`, and `z` solves the
unconstrained problem in `A·Q₂` (a [`FactoredWLS`](@ref)) against `b − Ax₀`.
Throws if `C` has dependent rows or `A` does not determine `x` on the
constraints' null space. `T` defaults to the common type of `A`, the weights,
`C` and `d`.
"""
struct ConstrainedWLS{T, F <: FactoredWLS{T}, X}
    Z::Matrix{T}
    reduced::F
    x0::X       # particular solution of `Cx = d`, `nothing` for `d = 0`
    Ax0::X
end

ConstrainedWLS(A, inv_variances, C, d = nothing) = ConstrainedWLS{
    promote_type(eltype(A), real(eltype(inv_variances)), eltype(C), isnothing(d) ? Bool : eltype(d)),
}(A, inv_variances, C, d)

function ConstrainedWLS{T}(A, inv_variances, C, d = nothing) where {T}
    Base.require_one_based_indexing(A, C)
    n = size(A, 2)
    size(C, 2) == n || throw(DimensionMismatch("C has $(size(C, 2)) columns; A has $n"))
    p = size(C, 1)
    p <= n || throw(ArgumentError("$p constraints on $n unknowns"))
    isnothing(d) || length(d) == p || throw(DimensionMismatch("d has $(length(d)) entries; C has $p rows"))
    Fc = qr(Matrix{T}(transpose(C)))
    _full_rank(Fc.R) || throw(ArgumentError("the constraint rows are linearly dependent"))
    Q = Matrix{T}(Fc.Q * Matrix{T}(I, n, n))
    Z = Q[:, (p + 1):n]
    reduced = FactoredWLS{T}(A * Z, inv_variances)
    if isnothing(d) || iszero(d)
        return ConstrainedWLS{T, typeof(reduced), Nothing}(Z, reduced, nothing, nothing)
    end
    x0 = Q[:, 1:p] * (transpose(UpperTriangular(Fc.R)) \ Vector{T}(d))
    return ConstrainedWLS{T, typeof(reduced), Vector{T}}(Z, reduced, x0, A * x0)
end

(F::ConstrainedWLS{<:Any, <:Any, Nothing})(b) = F.Z * F.reduced(b)
(F::ConstrainedWLS{<:Any, <:Any, <:AbstractVector})(b) = F.x0 .+ F.Z * F.reduced(b .- F.Ax0)

# Inverse-variance weight of the increment between samples `k` and `k+1`: the
# precision of a difference of two independent estimates, `1/(1/w_k + 1/w_{k+1})`.
# A missing/non-positive endpoint weight makes the increment uninformative.
function _increment_weight(weights, k)
    weights === nothing && return 1.0
    wa, wb = weights[k], weights[k + 1]
    (isfinite(wa) && isfinite(wb) && wa > 0 && wb > 0) || return 0.0
    return wa * wb / (wa + wb)
end

# Where a track's step-to-step increments center: the weighted circular mean of the
# wrapped increments between adjacent finite samples, which is also the track's
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
        for i in eachindex(finite, weights)
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
    for i in (ref_idx + 1):lastindex(unwrapped)
        isfinite(unwrapped[i]) || continue
        unwrapped[i] += 2π * round((last - unwrapped[i]) / (2π))
        last = unwrapped[i]
    end

    last = unwrapped[ref_idx]
    for i in (ref_idx - 1):-1:firstindex(unwrapped)
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
