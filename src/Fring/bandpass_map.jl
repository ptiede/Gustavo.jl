# ── The profiled MAP solve ───────────────────────────────────────────────────
#
# The per-scan source term `S_J` of each (scan, station pair, feed pair) `J` is
# profiled out: at any gains it sits at its closed-form optimum, so the
# objective is a function of the gains alone, χ²/2 with `S = S*(g)` plus the
# prior energy. Its gradient is the held-`S` gradient at `S*`, since the
# optimum is stationary in `S`. Its Hessian is the held-`S` Gauss–Newton
# curvature `H`, which is sparse, plus a correction of rank at most three per `J`
# and observable (`_profile_product!`); where the model fits the data that
# correction is the Schur complement of `S`, `−Σ_J q qᵀ/q_J` over the cell
# weights `q_c = w|m_c|²` (`_gauss_newton_product!`). Phase and log-amplitude
# stay decoupled. The Hessian is dense across frequency, so each step is solved
# by preconditioned conjugate gradients, its products one pass over the cells.
#
# The Gauss–Newton form overstates the curvature where the data are weak: there
# `|S*|²` holds the noise as well as the signal, while the profiled χ² is nearly
# flat. Steps take the Gauss–Newton curvature while they are large, which keeps
# them in the basin they start in, and the exact Hessian once they are small.

"""
    BandpassCells(r, w, edge, gain_pairs; nchunks)

The correlations of a joint bandpass solve that carry data, grouped by their
(scan, station pair, feed pair) `J`: per cell the summed visibility `r`, weight
`w`, station-gain pair `e` (an index into `gain_pairs`) and its two station
gains `na`, `nb`; per `J` its cells `ranges[j]`; and the `J`s in `nchunks`
contiguous `chunks` of about equal size, the units the passes over the cells
are split into.
"""
struct BandpassCells{R <: AbstractVector, W <: AbstractVector, I <: AbstractVector{Int32}, U <: AbstractVector{UnitRange{Int}}}
    r::R
    w::W
    e::I
    na::I
    nb::I
    ranges::U
    chunks::Vector{UnitRange{Int}}
end

function BandpassCells(r, w, edge, gain_pairs; nchunks::Integer)
    rp, wp = parent(r), parent(w)
    rs, ws = eltype(rp)[], eltype(wp)[]
    es, na, nb = Int32[], Int32[], Int32[]
    ranges = UnitRange{Int}[]
    for J in CartesianIndices(ntuple(d -> axes(edge, d), 3))
        first_cell = length(es) + 1
        for c in axes(edge, 4)
            I = CartesianIndex(J, c)
            e = edge[I]
            (iszero(e) || !(wp[I] > 0)) && continue
            a, b = gain_pairs[e]
            push!(rs, rp[I]); push!(ws, wp[I]); push!(es, e); push!(na, a); push!(nb, b)
        end
        length(es) >= first_cell && push!(ranges, first_cell:length(es))
    end
    return BandpassCells(rs, ws, es, na, nb, ranges, _balanced_chunks(ranges, nchunks))
end

# `1:length(ranges)` in at most `n` contiguous chunks holding about equal numbers
# of cells.
function _balanced_chunks(ranges, n)
    total = isempty(ranges) ? 0 : last(last(ranges))
    chunks = UnitRange{Int}[]
    j0 = 1
    for i in 1:n
        j1 = i == n ? length(ranges) : searchsortedlast(ranges, total * i ÷ n; by = last)
        j1 >= j0 && push!(chunks, j0:j1)
        j0 = max(j0, j1 + 1)
    end
    return chunks
end

"""
    ChunkAccumulators(cells::BandpassCells, prototypes...)
    ChunkAccumulators(accs::ChunkAccumulators...)

Per-chunk accumulators for the passes over `cells`: for each of its chunks, an
array shaped like each of `prototypes`. A pass over the cells
(`_reduce_chunks!`) adds into one chunk's arrays at a time and sums them into
its outputs, so the arrays carry over from pass to pass instead of being made
for each. The second form joins the arrays of several, shared, not copied.
One pass at a time may use them.
"""
struct ChunkAccumulators{B <: Tuple}
    buffers::B
end

ChunkAccumulators(cells::BandpassCells, prototypes::AbstractArray...) =
    ChunkAccumulators(map(p -> [zero(p) for _ in cells.chunks], prototypes))
ChunkAccumulators(acc::ChunkAccumulators, accs::ChunkAccumulators...) =
    ChunkAccumulators(mapreduce(a -> a.buffers, (x, y) -> (x..., y...), accs; init = acc.buffers))

# `outs` set to the sum over the chunks of `cells` of what `f!(bufs, js)` adds
# into one chunk's arrays `bufs` of `acc`, zeroed first, from that chunk's `J`s
# `js`.
function _reduce_chunks!(f!, outs::NTuple{N, Any}, acc::ChunkAccumulators{<:NTuple{N, Any}}, cells::BandpassCells, executor) where {N}
    tforeach(eachindex(cells.chunks); scheduler = executor) do i
        f!(map(b -> fill!(b[i], zero(eltype(b[i]))), acc.buffers), cells.chunks[i])
    end
    map(_sum_into!, outs, acc.buffers)
    return outs
end

# `S` (one entry per `J` of `cells`) at its optimum for the station gains `g`.
function _source!(S, g, cells::BandpassCells, executor)
    (; r, w, na, nb) = cells
    tforeach(cells.chunks; scheduler = executor) do js
        for j in js
            num, den = zero(ComplexF64), 0.0
            for k in cells.ranges[j]
                u = g[na[k]] * conj(g[nb[k]])
                num += conj(u) * r[k]
                den += w[k] * abs2(u)
            end
            S[j] = den > 0 ? num / den : zero(ComplexF64)
        end
    end
    return S
end

# The χ² of the visibilities with `S` at its optimum for the station gains `g`.
function _profiled_chi2(g, cells::BandpassCells, executor)
    (; r, w, na, nb) = cells
    parts = tmap(cells.chunks; scheduler = executor) do js
        out = 0.0
        for j in js
            rng = cells.ranges[j]
            num, den = zero(ComplexF64), 0.0
            for k in rng
                u = g[na[k]] * conj(g[nb[k]])
                num += conj(u) * r[k]
                den += w[k] * abs2(u)
            end
            s = den > 0 ? num / den : zero(ComplexF64)
            for k in rng
                out += abs2(r[k] - w[k] * g[na[k]] * s * conj(g[nb[k]])) / w[k]
            end
        end
        out
    end
    return sum(parts)
end

# The χ² to second order in each station's log-gain about the gains `g`, with
# `S` (one entry per `J`) held. A correlation's model `m = g_a·S·conj(g_b)`
# moves to `m·(1 + δℓ_a + δℓ_b + i(δφ_a − δφ_b))`, so with `u = r/(w·m) − 1` and
# weight `q = w·|m|²`, `Re u` measures `δℓ_a + δℓ_b` and `Im u` measures
# `δφ_a − δφ_b`, and the two observables decouple. Returns each station-gain
# pair's weight `q`, each gain's `c = Σ q·u` (with `conj(u)` at a second end),
# whose real and imaginary parts are the log-amplitude and phase gradients, and
# its total weight `W`: `1 + c/W` is the gain's own estimate given every other
# gain and `S`, relative to its current one. The chunks add into `acc`, and the
# result into `lin`.
function _linearize(
        g, S, cells::BandpassCells, ngains, npairs, executor,
        acc = ChunkAccumulators(cells, zeros(npairs), zeros(ComplexF64, ngains), zeros(ngains)),
        lin = (; q = zeros(npairs), c = zeros(ComplexF64, ngains), W = zeros(ngains)),
    )
    (; r, w, e, na, nb) = cells
    _reduce_chunks!(values(lin), acc, cells, executor) do (q, c, W), js
        for j in js
            s = S[j]
            for k in cells.ranges[j]
                a, b = na[k], nb[k]
                m = g[a] * s * conj(g[b])
                qk = w[k] * abs2(m)
                qk > 0 || continue
                # `q·u` with `u = r/(w·m) − 1`.
                qu = conj(m) * r[k] - qk
                q[e[k]] += qk
                c[a] += qu
                c[b] += conj(qu)
                W[a] += qk
                W[b] += qk
            end
        end
    end
    return lin
end

# `ys = (Δ − L)·xs` over station-gain vectors for one observable (`σ = −1`
# phase, `+1` log-amplitude): the exact Hessian of the profiled χ²/2 less the
# held-`S` Gauss–Newton curvature `H` (`_data_curvature!`). Per `J`, with
# `z_c = conj(u_c)·r_c`, `p_c = w_c|u_c|²`, `D = Σ p_c` and `S = Σ z_c / D`,
# the phase Hessian is `diag(Re(S̄z_c)) − Re(z zᴴ)/D` and the log-amplitude
# Hessian `diag(2|S|²p_c − Re(S̄z_c)) − Re(z zᴴ)/D + 2(t pᵀ + p tᵀ)/D −
# 4|S|²ppᵀ/D` with `t = Re(S̄z)`, over each cell's `x_a + σ·x_b`; `H` is
# `diag(|S|²p_c)`.
function _profile_product!(ys, xs, g, S, cells::BandpassCells, σ, executor, acc = ChunkAccumulators(cells, ys))
    (; r, w, na, nb) = cells
    _reduce_chunks!((ys,), acc, cells, executor) do (y,), js
        for j in js
            rng = cells.ranges[j]
            s = S[j]
            s2 = abs2(s)
            D = T1 = T2 = T3 = 0.0
            for k in rng
                a, b = na[k], nb[k]
                u = g[a] * conj(g[b])
                z = conj(u) * r[k]
                p = w[k] * abs2(u)
                D += p
                xc = xs[a] + σ * xs[b]
                T1 += real(z) * xc
                T2 += imag(z) * xc
                T3 += p * xc
                δ = σ * (real(conj(s) * z) - s2 * p)
                y[a] -= δ * xc
                y[b] -= σ * δ * xc
            end
            D > 0 || continue
            Tt = real(s) * T1 + imag(s) * T2
            for k in rng
                a, b = na[k], nb[k]
                u = g[a] * conj(g[b])
                z = conj(u) * r[k]
                coef = -(real(z) * T1 + imag(z) * T2)
                if σ > 0
                    p = w[k] * abs2(u)
                    coef += 2 * (real(conj(s) * z) * T3 + p * Tt) - 4 * s2 * p * T3
                end
                coef /= D
                y[a] += coef
                y[b] += σ * coef
            end
        end
    end
    return ys
end

# `(Δ − L)·x` over both observables at once (`_profile_product!`), with the
# Hessian's cross block between them: in each cell's `ρ_c = ℓ_a + ℓ_b` and
# `θ_c = φ_a − φ_b`, `∂²/∂ρ_c∂θ_d` of the profiled χ²/2 is
# `−Im(z̄_c z_d)/D − δ_cd·Im(S̄z_c) + 2p_c·Im(S̄z_d)/D`, which vanishes where the
# model fits the data. Writes `yφ` and `yℓ` from the station-gain vectors `xφ`
# and `xℓ`.
function _joint_profile_product!(yφ, yℓ, xφ, xℓ, g, S, cells::BandpassCells, executor, acc = ChunkAccumulators(cells, yφ, yℓ))
    (; r, w, na, nb) = cells
    _reduce_chunks!((yφ, yℓ), acc, cells, executor) do (φy, ℓy), js
        for j in js
            rng = cells.ranges[j]
            s = S[j]
            s2 = abs2(s)
            D = Rθ = Iθ = Rρ = Iρ = Pρ = 0.0
            for k in rng
                a, b = na[k], nb[k]
                u = g[a] * conj(g[b])
                z = conj(u) * r[k]
                p = w[k] * abs2(u)
                D += p
                θ = xφ[a] - xφ[b]
                ρ = xℓ[a] + xℓ[b]
                Rθ += real(z) * θ
                Iθ += imag(z) * θ
                Rρ += real(z) * ρ
                Iρ += imag(z) * ρ
                Pρ += p * ρ
                t = real(conj(s) * z)
                c = -imag(conj(s) * z)
                yθ = (t - s2 * p) * θ + c * ρ
                yρ = (s2 * p - t) * ρ + c * θ
                φy[a] += yθ
                φy[b] -= yθ
                ℓy[a] += yρ
                ℓy[b] += yρ
            end
            D > 0 || continue
            Tρ = real(s) * Rρ + imag(s) * Iρ
            for k in rng
                a, b = na[k], nb[k]
                u = g[a] * conj(g[b])
                z = conj(u) * r[k]
                p = w[k] * abs2(u)
                t = real(conj(s) * z)
                v = imag(conj(s) * z)
                yθ = -(real(z) * Rθ + imag(z) * Iθ) - (imag(z) * Rρ - real(z) * Iρ) + 2 * v * Pρ
                yρ = -(real(z) * Rρ + imag(z) * Iρ) + 2 * (t * Pρ + p * Tρ) - 4 * s2 * p * Pρ -
                    (real(z) * Iθ - imag(z) * Rθ) + 2 * p * (real(s) * Iθ - imag(s) * Rθ)
                φy[a] += yθ / D
                φy[b] -= yθ / D
                ℓy[a] += yρ / D
                ℓy[b] += yρ / D
            end
        end
    end
    return yφ, yℓ
end

# `ys = −Σ_J q (qᵀ xs) / q_J`, the Gauss–Newton form of `_profile_product!`, with
# `q_c = w|u_c|²|S|²` and `q_J = Σ_c q_c`: the Schur complement of `S`, which is
# the profile's correction wherever the model fits the data.
function _gauss_newton_product!(ys, xs, g, S, cells::BandpassCells, σ, executor, acc = ChunkAccumulators(cells, ys))
    (; w, na, nb) = cells
    _reduce_chunks!((ys,), acc, cells, executor) do (y,), js
        for j in js
            rng = cells.ranges[j]
            s2 = abs2(S[j])
            qj = t = 0.0
            for k in rng
                q = w[k] * abs2(g[na[k]]) * abs2(g[nb[k]]) * s2
                qj += q
                t += q * (xs[na[k]] + σ * xs[nb[k]])
            end
            qj > 0 || continue
            t /= qj
            for k in rng
                qt = w[k] * abs2(g[na[k]]) * abs2(g[nb[k]]) * s2 * t
                y[na[k]] -= qt
                y[nb[k]] -= σ * qt
            end
        end
    end
    return ys
end

function _sum_into!(ys, parts)
    fill!(ys, zero(eltype(ys)))
    foreach(p -> ys .+= p, parts)
    return ys
end

# The profile of one observable at the current linearization: the station gains
# `g`, `S`, the cells, `σ`, the system's `J` from its unknowns `θ` to the station
# gains, station-gain-vector buffers, whether its curvature is the exact Hessian
# (`_profile_product!`) or the Gauss–Newton one (`_gauss_newton_product!`), the
# scheduler of its passes and their accumulators, one station-gain vector per
# chunk.
struct Profile{G <: AbstractVector, C, M, V <: AbstractVector, X, A}
    g::G
    S::G
    cells::C
    σ::Float64
    J::M
    xs::V
    ys::V
    exact::Bool
    executor::X
    acc::A
end

# The profile's correction (`_profile_product!` or `_gauss_newton_product!`)
# times `x` over the unknowns `θ`, the first entries of `x`, carried to the
# station gains by `J` and back by `Jᵀ`.
function _profile_apply!(y, p::Profile, x)
    mul!(p.xs, p.J, view(x, axes(p.J, 2)))
    (p.exact ? _profile_product! : _gauss_newton_product!)(p.ys, p.xs, p.g, p.S, p.cells, p.σ, p.executor, p.acc)
    mul!(view(y, axes(p.J, 2)), p.J', p.ys)
    return y
end

"""
    ProfiledCurvature(K, profile)

The Hessian of the profiled objective over a system's unknowns: `K`, the
held-`S` Gauss–Newton quadratic with the priors (and any damping), plus the
profile's correction on the parameters, which are the first unknowns
(`_profile_product!`). Applied matrix-free, one pass over the cells per
product; its entries are not stored, and indexing it throws. A later step's
profile replaces `profile` in place (`_renew!`).
"""
mutable struct ProfiledCurvature{M, P} <: AbstractMatrix{Float64}
    const K::M
    profile::P
    const buffer::Vector{Float64}
end
ProfiledCurvature(K, profile::Profile) = ProfiledCurvature(K, profile, zeros(size(profile.J, 2)))

Base.size(A::ProfiledCurvature) = size(A.K)
Base.size(A::ProfiledCurvature, d::Integer) = size(A.K, d)
Base.getindex(::ProfiledCurvature, ::Int, ::Int) = error("a ProfiledCurvature is applied matrix-free; its entries are not stored")

function LinearAlgebra.mul!(y::AbstractVector, A::ProfiledCurvature, x::AbstractVector)
    mul!(y, A.K, x)
    _profile_apply!(A.buffer, A.profile, x)
    view(y, eachindex(A.buffer)) .+= A.buffer
    return y
end

"""
    JointCurvature(Kφ, Kℓ, φ, ℓ)

The exact Hessian of the profiled objective over both observables' unknowns,
the phase's first: the held-`S` quadratics `Kφ` and `Kℓ` with their priors (and
any damping), plus the profile's correction with its cross block between the
observables (`_joint_profile_product!`) on the parameters of the profiles
`φ` and `ℓ`. Applied matrix-free, as [`ProfiledCurvature`](@ref).
"""
mutable struct JointCurvature{Mφ, Mℓ, P} <: AbstractMatrix{Float64}
    const Kφ::Mφ
    const Kℓ::Mℓ
    φ::P
    ℓ::P
end

Base.size(A::JointCurvature) = (n = size(A.Kφ, 1) + size(A.Kℓ, 1); (n, n))
Base.size(A::JointCurvature, d::Integer) = size(A)[d]
Base.getindex(::JointCurvature, ::Int, ::Int) = error("a JointCurvature is applied matrix-free; its entries are not stored")

function LinearAlgebra.mul!(y::AbstractVector, A::JointCurvature, x::AbstractVector)
    nφ = size(A.Kφ, 1)
    xφ, xℓ = view(x, 1:nφ), view(x, (nφ + 1):length(x))
    yφ, yℓ = view(y, 1:nφ), view(y, (nφ + 1):length(y))
    mul!(yφ, A.Kφ, xφ)
    mul!(yℓ, A.Kℓ, xℓ)
    _joint_apply!(yφ, yℓ, A.φ, A.ℓ, xφ, xℓ)
    return y
end

# Adds the joint profile correction (`_joint_profile_product!`) at the unknowns
# `xφ`, `xℓ` (the first entries of each, carried by each profile's `J`) to
# `yφ`, `yℓ`.
function _joint_apply!(yφ, yℓ, φ::Profile, ℓ::Profile, xφ, xℓ)
    mul!(φ.xs, φ.J, view(xφ, axes(φ.J, 2)))
    mul!(ℓ.xs, ℓ.J, view(xℓ, axes(ℓ.J, 2)))
    _joint_profile_product!(φ.ys, ℓ.ys, φ.xs, ℓ.xs, φ.g, φ.S, φ.cells, φ.executor, ChunkAccumulators(φ.acc, ℓ.acc))
    mul!(view(yφ, axes(φ.J, 2)), φ.J', φ.ys, true, true)
    mul!(view(yℓ, axes(ℓ.J, 2)), ℓ.J', ℓ.ys, true, true)
    return yφ, yℓ
end

# The two observables' preconditioners side by side, the phase's first `n` unknowns.
struct BlockPreconditioner{A, B}
    φ::A
    ℓ::B
    n::Int
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::BlockPreconditioner, x::AbstractVector)
    ldiv!(view(y, 1:P.n), P.φ, view(x, 1:P.n))
    ldiv!(view(y, (P.n + 1):length(y)), P.ℓ, view(x, (P.n + 1):length(x)))
    return y
end

"""
    ProfilePreconditioner(F, Z, U)

A balancing preconditioner for the profiled curvature `A`: the inverse of the
held-`S` quadratic `K` (its factor `F`) on the complement of the coarse space
`Z`, and `A` itself on `Z`,

    M⁻¹ = (I − Z E⁻¹ Uᵀ) K⁻¹ (I − U E⁻¹ Zᵀ) + Z E⁻¹ Zᵀ,   E = Zᵀ U,

with `U = A·Z`. `Z` holds the constant of each track whose prior holds it:
`S` absorbs those constants, so `A` sees them through the priors alone while
`K` sees them through every correlation of the track. `F` and `U` are read
where they are, so a refactorization of `F` and a new `U` in place carry over
once `E` is refactored (`_refactor_coarse!`).
"""
mutable struct ProfilePreconditioner{F, E}
    const F::F
    const Z::SparseMatrixCSC{Float64, Int}
    const U::Matrix{Float64}
    E::E
    const c::Vector{Float64}
    const t::Vector{Float64}
    const v::Vector{Float64}
    const w::Vector{Float64}
end

function ProfilePreconditioner(F, Z, U)
    n, m = size(U)
    E = cholesky!(Symmetric(Matrix(Z' * U)))
    return ProfilePreconditioner(F, Z, U, E, zeros(m), zeros(m), zeros(n), zeros(n))
end

_refactor_coarse!(P::ProfilePreconditioner) = (P.E = cholesky!(Symmetric(mul!(P.E.factors, P.Z', P.U))); P)

function LinearAlgebra.ldiv!(y::AbstractVector, P::ProfilePreconditioner, x::AbstractVector)
    c = ldiv!(P.E, mul!(P.c, P.Z', x))
    v = ldiv!(P.v, P.F, mul!(copyto!(P.w, x), P.U, c, -1, true))
    copyto!(y, v)
    t = ldiv!(P.E, mul!(P.t, P.U', v))
    t .= c .- t
    mul!(y, P.Z, t, true, true)
    return y
end

# The coarse space `Z` of `ProfilePreconditioner` for the system `sys`
# (`_coarse_basis`) and its image under the curvature, `(K − H)·Z + λ·D·Z`, the
# priors' curvature along it (`layout.prior`) plus the damping, `D` as
# `_reduce!` last wrote it.
# `S` absorbs the constants, so the data's curvature along them is zero. The
# image is in a buffer of `sys` the next call overwrites.
function _coarse_space!(sys, λ)
    (; layout, coarse, damping) = sys
    (; Z, U) = coarse
    K = layout.K
    fill!(U, 0.0)
    for g in axes(Z, 2), q in nzrange(Z, g)
        j, zj = rowvals(Z)[q], nonzeros(Z)[q]
        for p in nzrange(K, j)
            U[rowvals(K)[p], g] += zj * layout.prior[p]
        end
        j in eachindex(damping) && (U[j, g] += λ * damping[j] * zj)
    end
    return Z, U
end

# The reduction of a step's residual at which its Krylov solve stops.
const _FORCING = 0.1

"""
    _profiled_step(s, λ, linsolve; abstol, z0 = _current_unknowns(s.sys)) -> (; z, x′, decrease, iterations)

One damped Newton step of the profiled objective for the system `s` of one
observable (`_profiled_systems`): the minimum over its unknowns of the held-`S`
quadratic plus the profile's correction (exact or Gauss–Newton, as `profile`
holds), plus `λ/2·Σ d_j(θ′_j − θ_j)²` over the parameters, `d` their curvatures
(`_reduce!`). It is solved by
`linsolve`, a Krylov method of LinearSolve.jl, preconditioned by
`ProfilePreconditioner` over the constants of its tracks, from the current point
`z0`, by default the parameters with the latent unknowns at their optimum given
them (`_current_unknowns`). There the initial residual is the objective's
negative gradient (`_step_residual`), and the solve stops once the residual in
the preconditioned norm, which measures the error in the step's own standard
errors, falls below `abstol` or a tenth of its start (an inexact Newton step;
Eisenstat & Walker 1996). Returns the unknowns, the new gain values (in a
buffer of the system the next step overwrites), the decrease the undamped model
predicts and the Krylov iterations taken. The decrease is
`−gᵀδ − δᵀ(A − λ)δ/2` from the gradient `g` and the increment `δ`: the model's
own values are differences of terms of the order of the stiffest prior's
curvature, and lose the decrease to rounding.
"""
function _profiled_step(s, λ, linsolve; abstol, z0 = _current_unknowns(s.sys))
    (; sys) = s
    K, _, _ = _reduce!(sys, s.c, λ)
    F = _refactor!(sys, K)
    nθ = length(sys.θ)
    Z, U = _coarse_space!(sys, λ)
    z = copy(z0)
    residual = _step_residual(s, z)
    A, _, sol = _krylov_solve!(s.krylov, s.profile, residual, linsolve; abstol) do
        ProfiledCurvature(K, s.profile), ProfilePreconditioner(F, Z, U)
    end
    δ = sol.u
    z .+= δ
    decrease = _predicted_decrease(A, residual, δ, ((1:nθ, λ, sys.damping),))
    return (; z, x′ = _gains_at!(sys, z), decrease, iterations = sol.iters)
end

# `A·δ = r` solved by `linsolve`, preconditioned by `P`, to `abstol` or a tenth
# of the starting residual, from a LinearSolve cache kept in `state` with `A`
# and `P`, which `make()` gives on the first call: a later call renews them in
# place for its `profiles` (`_renew!`), with the factor and the coarse space
# they read already rewritten, so the Krylov workspace carries over. Returns
# `A`, `P` and the solution, whose `u` the next call overwrites.
function _krylov_solve!(make, state, profiles, r, linsolve; abstol)
    if isnothing(state[])
        A, P = make()
        cache = LinearSolve.init(LinearSolve.LinearProblem(A, r), linsolve; Pl = P, abstol, reltol = _FORCING, maxiters = 10 * length(r))
        state[] = (; A, P, cache)
    else
        (; A, P, cache) = state[]
        _renew!(A, P, profiles)
        cache.b = r
        cache.abstol = abstol
    end
    (; A, P, cache) = state[]
    sol = LinearSolve.solve!(cache)
    LinearSolve.SciMLBase.successful_retcode(sol.retcode) || error(
        "the profiled bandpass step did not converge: $(nameof(typeof(linsolve))) returned $(sol.retcode) " *
            "after $(sol.iters) iterations",
    )
    return A, P, sol
end

_renew!(A::ProfiledCurvature, P::ProfilePreconditioner, profile) = (A.profile = profile; _refactor_coarse!(P))
function _renew!(A::JointCurvature, P::BlockPreconditioner, (φ, ℓ))
    A.φ, A.ℓ = φ, ℓ
    _refactor_coarse!(P.φ)
    _refactor_coarse!(P.ℓ)
    return A
end

# The station-gain values of `sys` at the unknowns `z`, whose first are its `θ`,
# in a buffer of `sys` the next call overwrites.
_gains_at!(sys, z) = mul!(sys.gains, sys.J, view(z, eachindex(sys.θ)))

# The negative gradient of system `s`'s objective at the unknowns `z`, whose
# station gains are at their current values: the data's `b` on the gains less
# the priors' gradient (`_prior_gradient!`), in a buffer of `s.sys` the next
# call overwrites.
function _step_residual(s, z)
    r = _prior_gradient!(s.sys.residual, s.sys.layout, s.priors, z)
    r .= .-r
    view(r, eachindex(s.sys.bθ)) .+= s.sys.bθ
    return r
end

"""
    _joint_step(φ, ℓ, λ, linsolve; abstol, z0, state) -> (; z, x′, decrease, iterations)

`_profiled_step` over both observables at once, with the exact Hessian's cross
block between them (`JointCurvature`): `φ` and `ℓ` are the observables' systems
(`_profiled_systems`), `λ` their dampings and
`z0` their unknowns at the current point, as `(; phase, amp)`. Its LinearSolve
cache is kept in `state` (`_krylov_solve!`).
Away from the mode, where the model leaves residuals as large as the signal, the
cross block is as large as the diagonal ones.
"""
function _joint_step(φ, ℓ, λ, linsolve; abstol, z0 = map(s -> _current_unknowns(s.sys), (; phase = φ, amp = ℓ)), state)
    Kφ, _, _ = _reduce!(φ.sys, φ.c, λ.phase)
    Kℓ, _, _ = _reduce!(ℓ.sys, ℓ.c, λ.amp)
    Fφ, Fℓ = _refactor!(φ.sys, Kφ), _refactor!(ℓ.sys, Kℓ)
    Zφ, Uφ = _coarse_space!(φ.sys, λ.phase)
    Zℓ, Uℓ = _coarse_space!(ℓ.sys, λ.amp)
    nφ = size(Kφ, 1)
    z = [z0.phase; z0.amp]
    residual = [_step_residual(φ, z0.phase); _step_residual(ℓ, z0.amp)]
    A, _, sol = _krylov_solve!(state, (φ.profile, ℓ.profile), residual, linsolve; abstol) do
        P = BlockPreconditioner(ProfilePreconditioner(Fφ, Zφ, Uφ), ProfilePreconditioner(Fℓ, Zℓ, Uℓ), nφ)
        JointCurvature(Kφ, Kℓ, φ.profile, ℓ.profile), P
    end
    δ = sol.u
    z .+= δ
    zφ, zℓ = view(z, 1:nφ), view(z, (nφ + 1):length(z))
    decrease = _predicted_decrease(
        A, residual, δ, ((1:length(φ.sys.θ), λ.phase, φ.sys.damping), (nφ .+ (1:length(ℓ.sys.θ)), λ.amp, ℓ.sys.damping)),
    )
    return (;
        z = (; phase = collect(zφ), amp = collect(zℓ)), x′ = (; phase = _gains_at!(φ.sys, zφ), amp = _gains_at!(ℓ.sys, zℓ)),
        decrease, iterations = sol.iters,
    )
end

# The decrease of the quadratic with curvature `A`, less the damping `λ·d` on
# the unknowns `s` of each `(s, λ, d)` of `damped`, along the increment `δ` from
# a point where its negative gradient is `residual`.
function _predicted_decrease(A, residual, δ, damped)
    damping = sum(((s, λ, d),) -> λ * sum(k -> d[k] * δ[s[k]]^2, eachindex(s, d); init = 0.0), damped)
    return dot(residual, δ) - (dot(δ, mul!(similar(δ), A, δ)) - damping) / 2
end

# Each observable's MAP system at the station gains `x` (`(; phase, amp)`
# values): `(; sys, c, profile, priors)`, `priors` its tracks' resolved priors,
# or `nothing` where the observable is not fit, with the linearization `lin`,
# `S` re-solved. `ctx` is the solve's fixed structure (`solve_joint_gains!`):
# each observable's unknowns are its parameters `ctx.params[obs].θ`, carried to
# the station gains by its `J`. `exact` chooses the profile's curvature. The
# gains, `S`, the linearization and the profiles' vectors are buffers of `ctx`
# the next call overwrites.
function _profiled_systems(x, ctx, priors; exact)
    (; cells, fitted, tracks, params, buffers) = ctx
    g = _gains!(buffers.g, x)
    S = _source!(buffers.S, g, cells, ctx.executor)
    lin = _linearize(g, S, cells, ctx.ngains, ctx.npairs, ctx.executor, ctx.accumulators.linearize, buffers.lin)
    H = (; phase = _data_curvature!(ctx.Hφ, ctx.hpos, lin.q, -1), amp = _data_curvature!(ctx.Hℓ, ctx.hpos, lin.q, 1))
    b = (; phase = map!(imag, buffers.b.phase, lin.c), amp = map!(real, buffers.b.amp, lin.c))
    systems = map((; phase = :phase, amp = :amp)) do obs
        fitted[obs] || return nothing
        p = params[obs]
        sys = _cached_system!(ctx.systems, obs, H[obs], b[obs], p.J, p.θ, tracks[obs], p.key; p.ridge)
        c = _assemble!(sys.layout, priors[obs])
        xs, ys = buffers.profile[obs]
        profile = Profile(g, S, cells, obs === :phase ? -1.0 : 1.0, p.J, xs, ys, exact, ctx.executor, ctx.accumulators[obs])
        return (; sys, c, profile, priors = priors[obs], krylov = ctx.krylov[obs])
    end
    return systems, lin
end

# The station gains `exp(ℓ + iφ)` of the values `x` (`(; phase, amp)`), written into `g`.
_gains!(g, x) = (g .= exp.(complex.(x.amp, x.phase)))

# Writes each observable's parameters `θ` in `params` from the unknowns `z`,
# whose first they are.
function _set_parameters!(params, z)
    for obs in keys(params)
        isnothing(z[obs]) || (params[obs].θ .= view(z[obs], eachindex(params[obs].θ)))
    end
    return params
end

"""
    _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, μ = 1e-3, exact = false) -> NamedTuple

The MAP of the parameters `ctx.params` under the resolved `priors`, with the
station gains `x = J·θ` (`(; phase, amp)` values) kept with them, both updated
in place, by Levenberg–Marquardt on an objective over the parameters and the
tracks' latent unknowns: χ²/2 with `S` at its optimum plus the tracks' prior
energy at those unknowns (`_prior_energy`). The latent unknowns start at their
optimum given the parameters (`_current_unknowns`) and move
with each step, so the energies compared are sums of whitened innovations at
given states; re-minimizing over the states for each trial would perturb them by
the rounding of a solve as stiff as the stiffest prior. Each step is
`_profiled_step`, damped by `μ` times each parameter's curvature, with `μ`
following how well the step's model predicted the
objective's decrease (Madsen, Nielsen & Tingleff 2004, §3.2). The steps take the
Gauss–Newton curvature until one moves no gain by more than one standard error,
or by fewer than ten while the objective falls by more than 1.5 times the
model's prediction, and the exact Hessian from then on; with `exact`, a start
already in the mode's basin, from the first step. The solve stops once the
undamped step moves no gain by `tolerance` of its standard error from the data,
or once no damping lowers the objective. The undamped step is checked after a
step that moves no gain by `tolerance` or lowers the objective by less than
`tolerance²/2`.

Returns `(; converged, stalled, change, worst, iterations, krylov, μ, objective,
weights)`: the largest weighted change of the last step, undamped where no
damping lowered the objective, and its station gain, the steps and Krylov
iterations taken, the damping of the last accepted step, the final objective,
and each gain's data weight at the last linearization, the inverse square of
its standard error.
"""
function _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, μ = 1.0e-3, exact::Bool = false)
    (; cells, fitted) = ctx
    observables = (; phase = :phase, amp = :amp)
    energy(systems, z) = sum(obs -> isnothing(systems[obs]) ? 0.0 : _prior_energy(systems[obs].sys.layout, priors[obs], z[obs]), keys(observables))
    objective(systems, x, z) = _profiled_chi2(_gains!(ctx.buffers.trial, x), cells, ctx.executor) / 2 + energy(systems, z)
    abstol = tolerance / 10
    change, worst, stalled, converged, near = Inf, 0, false, false, false
    ν, iterations, krylov = 2.0, 0, 0
    O0, z = Inf, nothing
    μ_accepted = μ
    local lin
    for _ in 1:max_iterations
        iterations += 1
        systems, lin = _profiled_systems(x, ctx, priors; exact)
        if isnothing(z)
            z = map(s -> isnothing(s) ? nothing : _current_unknowns(s.sys), systems)
            O0 = objective(systems, x, z)
        end
        function propose(μ)
            if exact && all(fitted)
                step = _joint_step(systems.phase, systems.amp, (; phase = μ, amp = μ), linsolve; abstol, z0 = z, state = ctx.krylov.joint)
                krylov += step.iterations
                return step.x′, step.decrease, step.z
            end
            steps = map(keys(observables)) do obs
                s = systems[obs]
                isnothing(s) && return (x′ = x[obs], decrease = 0.0, iterations = 0, z = nothing)
                return _profiled_step(s, μ, linsolve; abstol, z0 = z[obs])
            end
            krylov += steps[1].iterations + steps[2].iterations
            return (; phase = steps[1].x′, amp = steps[2].x′), steps[1].decrease + steps[2].decrease, (; phase = steps[1].z, amp = steps[2].z)
        end
        if near
            x1, _, _ = propose(eps())
            change, worst = _largest_weighted_change(x1, x, lin.W)
            change < tolerance && (converged = true; break)
        end
        accepted, decrease = false, Inf
        while isfinite(μ)
            x1, predicted, z1 = propose(μ)
            change, worst = _largest_weighted_change(x1, x, lin.W)
            _within_rounding(x1.amp, x.amp) && _within_rounding(x1.phase, x.phase) && break
            O1 = objective(systems, x1, z1)
            # The objective is known to its rounding.
            if O1 <= O0 + 16 * eps(abs(O0))
                ρ = (O0 - O1) / predicted
                decrease = O0 - O1
                x.amp .= x1.amp
                x.phase .= x1.phase
                z, O0 = z1, O1
                _set_parameters!(ctx.params, z)
                # A step accepted at the objective's rounding can have `ρ ≤ 0`; it
                # doubles `μ`, as a rejection does.
                μ = max(μ * max(1 / 3, 1 - (2 * max(ρ, 0) - 1)^3), eps())
                μ_accepted = μ
                exact |= ρ > 1.5 && change < 10
                ν = 2.0
                accepted = true
                break
            end
            μ *= ν
            ν *= 2
        end
        # No damping lowers the objective. A step lowers it by about half its
        # squared change in standard errors, so an undamped step too small for
        # that to exceed the objective's rounding cannot be verified, and
        # ending there is not a stall.
        if !accepted
            x1, _, _ = propose(eps())
            change, worst = _largest_weighted_change(x1, x, lin.W)
            stalled = change >= tolerance && change^2 / 2 > 16 * eps(abs(O0))
            converged = !stalled
            break
        end
        near = change < tolerance || decrease < tolerance^2 / 2
        exact |= change < 1
    end
    return (; converged, stalled, change, worst, iterations, krylov, μ = μ_accepted, objective = O0, weights = copy(lin.W))
end

# Each station gain's summed data weight over the cells.
function _gain_weights(cells::BandpassCells, ngains)
    out = zeros(ngains)
    for (k, wk) in pairs(cells.w)
        out[cells.na[k]] += wk
        out[cells.nb[k]] += wk
    end
    return out
end

# The M-step for one track: `prior` (resolved from the `spec` that gives some
# hyperparameters as densities) with those hyperparameters at the maximum of the
# chains' expected log prior density under the state moments `mom`
# (`_chain_moments`) plus the hyperprior, and the largest change of a log
# hyperparameter. The hyperprior is restricted to the support the chains'
# pieces measure (`_log_hyper_box`). It gives the objective a curvature of at
# least about one per log hyperparameter squared, so agreement to `atol` nats
# places each log hyperparameter within about `√(2·atol)` of the maximum.
function _m_step(spec, prior, chains, mom)
    sums = _moment_sums(chains, mom)
    box = _track_box(spec, chains)
    function f(lθ)
        _in_box(lθ, box) || return convert(eltype(lθ), Inf)
        v = _expected_log_prior(_with_log_hypers(spec, lθ), sums, eltype(lθ)) + _log_hyperprior(spec, lθ)
        return isfinite(v) ? -v : oftype(v, Inf)
    end
    lθ0 = _log_hypers(spec, prior)
    lθ1, _ = _nelder_mead(f, _into_box(lθ0, box); atol = 1.0e-6)
    return _with_log_hypers(spec, lθ1), maximum(abs, lθ1 .- lθ0)
end

# The hyperparameters the tracks' priors give as densities, as `(observable,
# track)` pairs.
_estimated(ctx) = [(obs, k) for obs in (:phase, :amp) for k in eachindex(ctx.tracks[obs]) if _hyper_count(ctx.tracks[obs][k].prior) > 0]

_log_hyper_vector(priors, ctx, est) =
    reduce(vcat, [_log_hypers(ctx.tracks[obs][k].prior, priors[obs][k]) for (obs, k) in est]; init = Float64[])

# The support of `_log_hyper_vector` (`_track_box`), from the chains of the
# systems in `ctx`.
_log_hyper_boxes(ctx, est) =
    reduce(vcat, [_track_box(ctx.tracks[obs][k].prior, ctx.systems[obs].chains[k]) for (obs, k) in est]; init = NTuple{2, Float64}[])

"""
    _hyper_objective(priors, x, ctx, solved, est; linsolve, abstol) -> (L, ∇L, H)

The log posterior density `L` of the hyperparameters of `priors`, up to a
constant, by Laplace's method at the MAP `x` that `solved` (`_profiled_map!`)
reports, and its gradient over the log hyperparameters of the tracks `est`
(`_estimated`). `L = −O − N − ½ log det A + log p(ψ)`, with `O` the MAP
objective (`χ²/2` with `S` at its optimum plus the prior energy), `N` the
priors' log normalizer (`_prior_log_normalizer`), `A` the curvature of the
profiled objective in its Gauss–Newton form (`_joint_curvature!`), and `p(ψ)`
the hyperpriors over the log hyperparameters. Its gradient is in two parts.
With `L`'s other terms held at the MAP, the Fisher identity's: each track's
expected log prior density under the Gaussian posterior there, covariance
`A⁻¹`, plus its hyperprior, differentiated in its own log hyperparameters
(`_hyper_gradient`). Then the MAP moving with `ψ`, which changes `log det A`
through the cells' weights: with `Λ` the observables' summed `log det A`,
`dz*/dψ = −H⁻¹ ∂²O/∂z∂ψ` gives `½ ∂/∂ψ (aᵀ ∇z E)` with `a = H⁻¹ ∇z Λ`, `H` the
exact Hessian of the profiled objective (`_hessian_solve`, by `linsolve` to
`abstol`) and `E` the prior energy (`_logdet_gain_gradient`,
`_track_gradient_dot`). `H` holds per track of `est` the negative Hessian of its
expected log prior plus hyperprior (the M-step's objective) in its own log
hyperparameters, the complete data's information about them, at least the
evidence's own. A curvature that cannot be factored is an error.
"""
function _hyper_objective(priors, x, ctx, solved, est; linsolve, abstol)
    systems, _ = _profiled_systems(x, ctx, priors; exact = true)
    (; cells) = ctx
    parts = tmap(Any, [:phase, :amp]; scheduler = ctx.executor) do obs
        s = systems[obs]
        isnothing(s) && return nothing
        joint = _joint_curvature!(s.sys, s.profile, ctx.hpos)
        cov = joint.inverse()
        return (;
            z = _current_unknowns(s.sys), cov, joint.logdet, dq = joint.weight_gradient(cov),
            normalizer = _prior_log_normalizer(s.sys.layout, priors[obs]),
        )
    end
    curvatures = (; phase = parts[1], amp = parts[2])
    L = -solved.objective
    dΛdq = zeros(length(cells.w))
    for c in curvatures
        isnothing(c) && continue
        L -= c.normalizer + c.logdet / 2
        dΛdq .+= c.dq
    end
    dΛ = _logdet_gain_gradient(cells, systems, dΛdq)
    v = map((; phase = :phase, amp = :amp)) do obs
        s = systems[obs]
        isnothing(s) && return nothing
        return [s.sys.J' * dΛ[obs]; zeros(size(s.sys.layout.K, 1) - length(s.sys.θ))]
    end
    a = _hessian_solve(systems, v, ctx; linsolve, abstol)
    grads = map(est) do (obs, k)
        spec, s = ctx.tracks[obs][k].prior, systems[obs]
        chains, terms = s.sys.chains[k], s.sys.layout.terms[k]
        lθ = _log_hypers(spec, priors[obs][k])
        L += _log_hyperprior(spec, lθ)
        (; z, cov) = curvatures[obs]
        fisher, curvature = _hyper_gradient(spec, lθ, chains, _chain_moments(chains, z, cov, first(chains).dim))
        moving = gradient(l -> _track_gradient_dot(spec, l, terms, z, a[obs]), AutoForwardDiff(), lθ)
        return fisher .+ moving ./ 2, curvature
    end
    return L, reduce(vcat, first.(grads); init = Float64[]), last.(grads)
end

# `H⁻¹v` over each fitted observable's unknowns, `v` and the result as
# `(; phase, amp)`, with `H` the exact Hessian of the profiled objective at the
# systems' point, both observables' with its cross block between them where both
# are fit (`JointCurvature`): the step `_profiled_step` and `_joint_step` take
# undamped, with `v` for the residual, by `linsolve` to `abstol`.
function _hessian_solve(systems, v, ctx; linsolve, abstol)
    function prepare(s)
        K, _, _ = _reduce!(s.sys, s.c, 0.0)
        return K, _refactor!(s.sys, K), _coarse_space!(s.sys, 0.0)...
    end
    φ, ℓ = systems.phase, systems.amp
    if !isnothing(φ) && !isnothing(ℓ)
        Kφ, Fφ, Zφ, Uφ = prepare(φ)
        Kℓ, Fℓ, Zℓ, Uℓ = prepare(ℓ)
        nφ = size(Kφ, 1)
        _, _, sol = _krylov_solve!(ctx.krylov.joint, (φ.profile, ℓ.profile), [v.phase; v.amp], linsolve; abstol) do
            P = BlockPreconditioner(ProfilePreconditioner(Fφ, Zφ, Uφ), ProfilePreconditioner(Fℓ, Zℓ, Uℓ), nφ)
            JointCurvature(Kφ, Kℓ, φ.profile, ℓ.profile), P
        end
        return (; phase = sol.u[1:nφ], amp = sol.u[(nφ + 1):end])
    end
    solution = map((; phase = :phase, amp = :amp)) do obs
        s = systems[obs]
        isnothing(s) && return nothing
        K, F, Z, U = prepare(s)
        _, _, sol = _krylov_solve!(s.krylov, s.profile, v[obs], linsolve; abstol) do
            ProfiledCurvature(K, s.profile), ProfilePreconditioner(F, Z, U)
        end
        return copy(sol.u)
    end
    return solution
end

# `aᵀ ∇z E` for one track's chain `terms` (`_chain_terms`) under `spec` with the
# log hyperparameters `l`, at the unknowns `z`.
function _track_gradient_dot(spec, l, terms, z, a)
    T = eltype(l)
    return _with_piece_models(_with_log_hypers(spec, l), first(terms).dim, T) do model
        # A loop, as in `_assemble!`.
        out = zero(T)
        for t in terms
            out += _chain_gradient_dot(t, model(t.h), z, a, T)
        end
        out
    end
end

# The gradient of `Λ`, the observables' summed `log det A` (`_joint_curvature!`),
# over each observable's station-gain values, `(; phase, amp)`, from its
# gradient `dΛdq` over the cells' weights `q_c = w|u_c|²|S_J|²`,
# `u_c = g_a conj(g_b)`, with `S_J = N_J/P_J` at its optimum, `N_J = Σ conj(u_k) r_k`
# and `P_J = Σ w_k|u_k|²`: `∂ log q_c = 2∂(ℓ_a + ℓ_b) + 2∂ log|N_J| − 2∂ log P_J`.
function _logdet_gain_gradient(cells::BandpassCells, systems, dΛdq)
    (; g, S) = something(systems.phase, systems.amp).profile
    (; r, w, na, nb) = cells
    dφ, dℓ = zeros(length(g)), zeros(length(g))
    for (j, rng) in pairs(cells.ranges)
        N, P, GJ = zero(ComplexF64), 0.0, 0.0
        for k in rng
            u = g[na[k]] * conj(g[nb[k]])
            N += conj(u) * r[k]
            P += w[k] * abs2(u)
            q = w[k] * abs2(u * S[j])
            GJ += dΛdq[k] * q
            dℓ[na[k]] += 2 * dΛdq[k] * q
            dℓ[nb[k]] += 2 * dΛdq[k] * q
        end
        (abs2(N) > 0 && P > 0) || continue
        for k in rng
            u = g[na[k]] * conj(g[nb[k]])
            ρ = conj(N) * conj(u) * r[k] / abs2(N)
            p = w[k] * abs2(u) / P
            dℓ[na[k]] += 2GJ * (real(ρ) - 2p)
            dℓ[nb[k]] += 2GJ * (real(ρ) - 2p)
            dφ[na[k]] += 2GJ * imag(ρ)
            dφ[nb[k]] -= 2GJ * imag(ρ)
        end
    end
    return (; phase = dφ, amp = dℓ)
end

# The gradient over the log hyperparameters `lθ` of the hyperparameters `spec`
# gives as densities of the chains' expected log prior density under the state
# moments `mom` (`_chain_moments`), plus the hyperprior, and the negative of
# its curvature along each of them.
function _hyper_gradient(spec, lθ, chains, mom)
    sums = _moment_sums(chains, mom)
    f(l) = _expected_log_prior(_with_log_hypers(spec, l), sums, eltype(l)) + _log_hyperprior(spec, l)
    return gradient(f, AutoForwardDiff(), lθ), -hessian(f, AutoForwardDiff(), lθ)
end

# The Gauss–Newton curvature of the profiled objective over the unknowns of
# `sys`, `A = K − B D⁻¹ Bᵀ`: per `J` of the profile's cells, `D_J = Σ q_c` and
# `B_J = Jᵀ Σ q_c (e_a + σ e_b)` with `q_c = w|g_a S_J conj(g_b)|²`, the curvature
# with `S`'s log-amplitude (or phase) as an unknown, profiled out; a `J` with
# `D_J = 0` has no such unknown. With `B̃ = B D^{-½}` and `T = B̃ᵀ K⁻¹ B̃`,
# `log det A = log det K + log det(I − T)` and `A⁻¹ = K⁻¹ + W (I − T)⁻¹ Wᵀ`,
# `W = K⁻¹ B̃`. `T`'s eigenvalues lie in `[0, 1)`: those near 1 are the few
# directions along which `S` takes up the gains (a track's level over the band
# or scan), the rest are orders of magnitude smaller. `T` is taken at its
# largest eigenpairs `T ≈ U Θ Uᵀ` (`_top_eigenpairs`), the rest as zero:
# `log det A = log det K + Σ log(1 − θ)`, `A⁻¹ = K⁻¹ + V (I − Θ)⁻¹ Vᵀ` with
# `V = K⁻¹ B̃ U`, an error where `K` cannot be factored or a `θ` reaches 1.
# Returns `logdet`; `inverse()`, the entries of `A⁻¹` on `K`'s pattern
# (`SelectedInverse`), as a function of a pair of unknowns; and
# `weight_gradient(Σ)`, from those entries the gradient of
# `log det A = log det M − log det D`, `M = [K B; Bᵀ D]`, over the cells' `q_c`:
# `Σ_p (M⁻¹)_p ∂M_p/∂q_c − 1/D_J`, where `M⁻¹` is `A⁻¹` on `K`,
# `−V (I − Θ)⁻¹ Uᵀ/√D_J` on `B_J` and `(1 + (U Θ (I − Θ)⁻¹ Uᵀ)_JJ)/D_J` on `D_J`.
# `K`'s symbolic analysis, the selected inverse's plan and the cells of each
# station gain are kept in `sys.evidence` (`_evidence_layout`), with `hpos` the
# pattern of the data curvature, and the results read buffers the next call
# overwrites.
function _joint_curvature!(sys, profile, hpos)
    ev = sys.evidence[]
    isnothing(ev) && (ev = sys.evidence[] = _evidence_layout(sys, profile.cells))
    return _joint_curvature_at!(sys, profile, hpos, ev)
end

function _joint_curvature_at!(sys, profile, hpos, ev)
    (; g, S, cells, σ, executor) = profile
    (; q, D, w̃) = ev
    K = sys.layout.K
    fill!(D, 0.0)
    for (j, rng) in pairs(cells.ranges), k in rng
        q[k] = cells.w[k] * abs2(g[cells.na[k]] * S[j] * conj(g[cells.nb[k]]))
        D[j] += q[k]
    end
    for (j, rng) in pairs(cells.ranges), k in rng
        w̃[k] = D[j] > 0 ? q[k] / sqrt(D[j]) : 0.0
    end
    F0 = ev.factor[]
    F = ev.factor[] = isnothing(F0) ? cholesky(Symmetric(K); check = false) : cholesky!(F0, Symmetric(K); check = false)
    issuccess(F) || _not_positive_definite()
    θ, U = _top_eigenpairs(X -> _capacitance_product(sys, F, ev, cells, σ, X, executor), length(D))
    all(<(1), θ) || _not_positive_definite()
    gains = Ref{Matrix{Float64}}()
    function inverse()
        V = F \ _embed(sys, _lift(sys, ev, cells, σ, U, executor))
        Vs = V ./ sqrt.(1 .- θ)'
        Σ = ev.selected[]
        Σ = ev.selected[] = isnothing(Σ) ? SelectedInverse(F, K) : _selected_inverse!(Σ, F)
        Σv = nonzeros(Σ.Σ)
        Vt = permutedims(Vs)
        tforeach(axes(K, 2); scheduler = executor) do j
            for p in nzrange(K, j)
                Σv[p] += dot(view(Vt, :, rowvals(K)[p]), view(Vt, :, j))
            end
        end
        gains[] = permutedims(sys.J * view(V ./ (1 .- θ)', axes(sys.J, 2), :))
        return Σ
    end
    function weight_gradient(Σ)
        gH = sys.data_map' * nonzeros(Σ.Σ)
        JV = gains[]
        Ut = permutedims(U)
        level = vec(sum(abs2.(U) .* (θ ./ (1 .- θ))'; dims = 2))
        out = similar(q)
        tforeach(_blocks(length(D), executor); scheduler = executor) do js
            for j in js
                u = view(Ut, :, j)
                for k in cells.ranges[j]
                    h = hpos[cells.e[k]]
                    v = gH[h[1]] + gH[h[2]] + σ * (gH[h[3]] + gH[h[4]])
                    if D[j] > 0
                        v += level[j] / D[j] - 2 * (dot(view(JV, :, cells.na[k]), u) + σ * dot(view(JV, :, cells.nb[k]), u)) / sqrt(D[j])
                    end
                    out[k] = v
                end
            end
        end
        return out
    end
    return (; logdet = logdet(F) + sum(t -> log1p(-t), θ; init = 0.0), inverse, weight_gradient)
end

# The `r` largest eigenpairs of the positive semidefinite `T` of order `m`,
# given as its product `mul(X)` with a block `X`, by subspace iteration from a
# fixed Gaussian block (Halko, Martinsson & Tropp 2011), so the same `T` gives
# the same pairs: two products, then Rayleigh–Ritz on a third. The block doubles,
# from 64 columns, until its smallest Ritz value is below `tail`.
function _top_eigenpairs(mul, m; r0 = 64, tail = 1.0e-3)
    m == 0 && return Float64[], zeros(0, 0)
    r = min(r0, m)
    θ, U = _ritz_pairs(mul, m, r)
    while r < m && minimum(θ) >= tail
        r = min(2r, m)
        θ, U = _ritz_pairs(mul, m, r)
    end
    return θ, U
end

function _ritz_pairs(mul, m, r)
    X = randn!(Xoshiro(r), zeros(m, r))
    for _ in 1:2
        X = Matrix(qr!(mul(X)).Q)
    end
    e = eigen(Symmetric(X' * mul(X)))
    return max.(e.values, 0.0), X * e.vectors
end

# `T X = B̃ᵀ K⁻¹ B̃ X` for a block `X` over the `J` (`_joint_curvature!`), with `F` the
# factor of `K`.
_capacitance_product(sys, F, ev, cells, σ, X, executor) =
    _restrict(sys, ev, cells, σ, F \ _embed(sys, _lift(sys, ev, cells, σ, X, executor)), executor)

# `Jᵀ G̃ X`, `B̃ X` over the parameters, with `G̃` the station gains' weights in each
# `J`, `w̃_c (e_a + σ e_b)` for each of its cells: summed per station gain over
# the cells it enters (`ev.bygain`).
function _lift(sys, ev, cells, σ, X, executor)
    (; bygain, gainptr, cellj, w̃) = ev
    Xt = permutedims(X)
    Gt = zeros(size(X, 2), length(gainptr) - 1)
    tforeach(_blocks(length(gainptr) - 1, executor); scheduler = executor) do ns
        for n in ns, p in gainptr[n]:(gainptr[n + 1] - 1)
            k = abs(bygain[p])
            c = (bygain[p] > 0 ? 1.0 : σ) * w̃[k]
            view(Gt, :, n) .+= c .* view(Xt, :, cellj[k])
        end
    end
    return (Gt * sys.J)'
end

# `G̃ᵀ J Y[θ]` over the `J`, for `Y` over the unknowns.
function _restrict(sys, ev, cells, σ, Y, executor)
    Yg = permutedims(sys.J * view(Y, axes(sys.J, 2), :))
    out = zeros(size(Y, 2), length(cells.ranges))
    tforeach(_blocks(length(cells.ranges), executor); scheduler = executor) do js
        for j in js, k in cells.ranges[j]
            view(out, :, j) .+= ev.w̃[k] .* (view(Yg, :, cells.na[k]) .+ σ .* view(Yg, :, cells.nb[k]))
        end
    end
    return permutedims(out)
end

# Values over the parameters, the first unknowns, as values over every unknown.
_embed(sys, X) = [X; zeros(size(sys.layout.K, 1) - size(X, 1), size(X, 2))]

# `1:n` in contiguous blocks, one per task of `executor`.
_blocks(n, executor) = collect(Iterators.partition(1:n, max(1, cld(n, max_tasks(executor)))))

_not_positive_definite() = error(
    "the Gauss–Newton curvature of the gains and source terms is not positive definite at the MAP, " *
        "so the Laplace evidence is undefined there",
)

# The fixed structure of `_joint_curvature!` for `sys` over `cells`: per station
# gain the cells it enters (`bygain` from `gainptr[n]` on, a cell's index, negated
# where the gain is its second), each cell's `J`, buffers for the cells' `q_c`,
# `w̃_c = q_c/√D_J` and each `D_J`, and places for `K`'s factor and its selected
# inverse.
function _evidence_layout(sys, cells)
    ngains = size(sys.J, 1)
    uses = zeros(Int, ngains)
    foreach(k -> (uses[cells.na[k]] += 1; uses[cells.nb[k]] += 1), eachindex(cells.na))
    gainptr = cumsum([1; uses])
    bygain, slot = zeros(Int, gainptr[end] - 1), gainptr[1:ngains]
    for k in eachindex(cells.na), (n, k′) in ((cells.na[k], k), (cells.nb[k], -k))
        bygain[slot[n]] = k′
        slot[n] += 1
    end
    cellj = zeros(Int, length(cells.w))
    foreach(((j, rng),) -> (cellj[rng] .= j), pairs(cells.ranges))
    ncells = length(cells.w)
    return (;
        bygain, gainptr, cellj, q = zeros(ncells), w̃ = zeros(ncells), D = zeros(length(cells.ranges)),
        factor = Ref{Union{Nothing, CHOLMOD.Factor{Float64, Int}}}(nothing), selected = Ref{Any}(nothing),
    )
end

function _set_log_hypers!(priors, ctx, est, ψ)
    i = 0
    for (obs, k) in est
        spec = ctx.tracks[obs][k].prior
        n = _hyper_count(spec)
        priors[obs][k] = _with_log_hypers(spec, ψ[(i + 1):(i + n)])
        i += n
    end
    return priors
end

"""
    _fit_hypers!(priors, x, ctx, solved; tolerance, hyper_tolerance, max_iterations, max_hyper_iterations,
                 linsolve, optimizer, warmup = 5) -> (; solved, converged, evaluations, unconverged, remaining)

The hyperparameters the priors give as densities at their type-II MAP, written
into `priors` with the gains `x` at their MAP under them. The evidence can
have more than one maximum along a track's hyperparameters (a smooth track,
or one held nearly rigid at the sample spacing), so `warmup` EM maps
(`_em_step!`, each after the MAP at the last) first move each track toward
what its posterior moments support. From there `optimizer`, an
Optimization.jl optimizer that takes a gradient and bounds, maximizes the
Laplace log evidence (`_hyper_objective`) over the log hyperparameters within
their support (`_log_hyper_boxes`), each scaled by the square root of its
curvature there, so that one unit of each is about one standard error.

Each evaluation is the MAP of the gains (`_profiled_map!`, warm from the last,
with the exact Hessian) and the evidence and its gradient there. The fit stops
once the evidence a Newton step would still gain (`remaining`, in nats), per
track `½ gᵀH⁻¹g` over its gradient `g` projected onto the support and the
complete data's curvature `H`, is below `hyper_tolerance`; when the
optimizer stops on its own; or after `max_hyper_iterations` iterations, the
EM maps among them, where `converged` is false. `solved` is the MAP at the
starting hyperparameters, and the result's the MAP at the final ones;
`unconverged` counts the MAPs that stopped at `max_iterations`.
"""
function _fit_hypers!(
        priors, x, ctx, solved; tolerance, hyper_tolerance, max_iterations, max_hyper_iterations, linsolve, optimizer, warmup = 5,
    )
    est = _estimated(ctx)
    boxes = _log_hyper_boxes(ctx, est)
    lo, hi = first.(boxes), last.(boxes)
    nem = min(warmup, max_hyper_iterations)
    for i in 1:nem
        i == 1 || (solved = _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, solved.μ, exact = true))
        _em_step!(priors, x, ctx, solved, est)
    end
    ψ0 = _into_box(_log_hyper_vector(priors, ctx, est), boxes)
    mapped, at = (nem > 0 ? fill(NaN, length(ψ0)) : _log_hyper_vector(priors, ctx, est)), Float64[]
    L, G, H = -Inf, Float64[], Matrix{Float64}[]
    evaluations, unconverged = 0, 0
    function evaluate!(ψ)
        ψ == at && return (; L, G, H, settled = solved.converged)
        _set_log_hypers!(priors, ctx, est, ψ)
        if ψ != mapped
            solved = _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, solved.μ, exact = true)
            unconverged += !(solved.converged || solved.stalled)
            mapped = copy(ψ)
        end
        L, G, H = _hyper_objective(priors, x, ctx, solved, est; linsolve, abstol = tolerance / 10)
        at = copy(ψ)
        evaluations += 1
        return (; L, G, H, settled = solved.converged)
    end
    tracks = _track_ranges(ctx, est)
    converged, remaining = if nem < max_hyper_iterations
        _maximize_hypers!(optimizer, evaluate!, ψ0, lo, hi, tracks; tolerance = hyper_tolerance, maxiters = max_hyper_iterations - nem)
    else
        e = evaluate!(ψ0)
        false, _remaining(ψ0, e.G, e.H, lo, hi, tracks)
    end
    return (; solved, converged, evaluations, unconverged, remaining)
end

# The positions in `_log_hyper_vector` of each track's log hyperparameters.
function _track_ranges(ctx, est)
    counts = [_hyper_count(ctx.tracks[obs][k].prior) for (obs, k) in est]
    return [(e - n + 1):e for (e, n) in zip(cumsum(counts), counts)]
end

# Whether coordinate `i` of `ψ` is held at a bound by the gradient `G`.
_held(ψ, G, lo, hi, i) = (ψ[i] <= lo[i] && G[i] < 0) || (ψ[i] >= hi[i] && G[i] > 0)

# The evidence a Newton step would gain, `½ gᵀH⁻¹g` per track over its
# coordinates not held at a bound, with `H` the track's block of the curvature.
function _remaining(ψ, G, H, lo, hi, tracks)
    gain = 0.0
    for (h, r) in zip(H, tracks)
        free = [j for (j, i) in pairs(r) if !_held(ψ, G, lo, hi, i)]
        isempty(free) && continue
        g = G[r[free]]
        gain += dot(g, _positive_definite(h[free, free]) \ g) / 2
    end
    return gain
end

# `b` with its eigenvalues floored at `1e-3` of the largest and at `1e-6`.
function _positive_definite(b)
    e = eigen(Symmetric(b))
    λ = max.(e.values, max(1.0e-3 * maximum(abs, e.values; init = 0.0), 1.0e-6))
    return Symmetric(e.vectors * Diagonal(λ) * e.vectors')
end

# An Optimization.jl optimizer over the coordinates scaled by the square root of
# the curvature's diagonal at `ψ0`, halted once the evidence left to gain is
# below `tolerance`.
function _maximize_hypers!(optimizer, evaluate!, ψ0, lo, hi, tracks; tolerance, maxiters)
    e = evaluate!(ψ0)
    D = sqrt.(max.(reduce(vcat, diag.(e.H); init = Float64[]), 1.0))
    remaining = _remaining(ψ0, e.G, e.H, lo, hi, tracks)
    function current!(u)
        e = evaluate!(u ./ D)
        remaining = _remaining(u ./ D, e.G, e.H, lo, hi, tracks)
        remaining < tolerance && e.settled && throw(HyperparametersSettled())
        return e
    end
    converged = try
        current!(ψ0 .* D)
        objective(u, _) = -current!(u).L
        gradient!(out, u, _) = (out .= .-current!(u).G ./ D)
        problem = Optimization.OptimizationProblem(
            Optimization.OptimizationFunction(objective; grad = gradient!), ψ0 .* D; lb = lo .* D, ub = hi .* D,
        )
        result = Optimization.solve(problem, optimizer; maxiters)
        current!(result.u)
        Optimization.SciMLBase.successful_retcode(result.retcode)
    catch err
        err isa HyperparametersSettled || rethrow()
        true
    end
    return converged, remaining
end

# One EM map of the hyperparameters `est` (`_estimated`) at the MAP `x` that
# `solved` reports, written into `priors`: each track's state moments under the
# Gaussian posterior there, covariance `A⁻¹` (`_joint_curvature!`), and its
# hyperparameters at the maximum of their expected log prior plus hyperprior
# within their support (`_m_step`).
function _em_step!(priors, x, ctx, solved, est)
    systems, _ = _profiled_systems(x, ctx, priors; exact = true)
    parts = tmap(Any, [:phase, :amp]; scheduler = ctx.executor) do obs
        s = systems[obs]
        isnothing(s) && return nothing
        joint = _joint_curvature!(s.sys, s.profile, ctx.hpos)
        return (; z = _current_unknowns(s.sys), cov = joint.inverse())
    end
    posterior = (; phase = parts[1], amp = parts[2])
    for (obs, k) in est
        chains = systems[obs].sys.chains[k]
        (; z, cov) = posterior[obs]
        priors[obs][k], _ = _m_step(ctx.tracks[obs][k].prior, priors[obs][k], chains, _chain_moments(chains, z, cov, first(chains).dim))
    end
    return priors
end

# Thrown by the objective `_maximize_hypers!` hands an Optimization.jl optimizer,
# to end its run once the evidence left to gain is below the tolerance.
struct HyperparametersSettled <: Exception end

# `_posterior_system` of one observable, kept in `cache[obs]` and its values
# rewritten for the curvature `H`, the gradient `b` and the parameters `θ`
# (`_set_data!`): the pattern, the factor's symbolic analysis and the buffers
# carry over. `H` keeps one sparsity pattern throughout (`_curvature_pattern`).
function _cached_system!(cache, obs, H, b, J, θ, tracks, key; kw...)
    sys = get(cache, obs, nothing)
    isnothing(sys) && return cache[obs] = _posterior_system(H, b, J, θ, tracks, key; kw...)
    return _set_data!(sys, H, b, θ)
end
