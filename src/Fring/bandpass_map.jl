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
    BandpassCells(r, w, edge, slot_pairs; nchunks)

The correlations of a joint bandpass solve that carry data, grouped by their
(scan, station pair, feed pair) `J`: per cell the summed visibility `r`, weight
`w`, slot pair `e` (an index into `slot_pairs`) and its two slots `na`, `nb`;
per `J` its cells `ranges[j]`; and the `J`s in `nchunks` contiguous `chunks` of
about equal size, the units the passes over the cells are split into.
"""
struct BandpassCells{C, T}
    r::Vector{C}
    w::Vector{T}
    e::Vector{Int32}
    na::Vector{Int32}
    nb::Vector{Int32}
    ranges::Vector{UnitRange{Int}}
    chunks::Vector{UnitRange{Int}}
end

function BandpassCells(r, w, edge, slot_pairs; nchunks::Integer)
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
            a, b = slot_pairs[e]
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

# `S` (one entry per `J` of `cells`) at its optimum for the slot gains `g`.
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

# The χ² of the visibilities with `S` at its optimum for the slot gains `g`.
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

# The χ² to second order in each slot's log-gain about the gains `g`, with `S`
# (one entry per `J`) held. A correlation's model `m = g_a·S·conj(g_b)` moves to
# `m·(1 + δℓ_a + δℓ_b + i(δφ_a − δφ_b))`, so with `u = r/(w·m) − 1` and weight
# `q = w·|m|²`, `Re u` measures `δℓ_a + δℓ_b` and `Im u` measures `δφ_a − δφ_b`,
# and the two observables decouple. Returns each slot pair's weight `q`, each
# slot's `c = Σ q·u` (with `conj(u)` at a second end), whose real and imaginary
# parts are the log-amplitude and phase gradients, and its total weight `W`:
# `1 + c/W` is the slot's own gain estimate given every other gain and `S`,
# relative to its current one.
function _linearize(g, S, cells::BandpassCells, nslots, npairs, executor)
    (; r, w, e, na, nb) = cells
    parts = tmap(cells.chunks; scheduler = executor) do js
        q = zeros(npairs)
        c = zeros(ComplexF64, nslots)
        W = zeros(nslots)
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
        (; q, c, W)
    end
    return (; q = sum(p -> p.q, parts), c = sum(p -> p.c, parts), W = sum(p -> p.W, parts))
end

# `ys = (Δ − L)·xs` over slot vectors for one observable (`σ = −1` phase, `+1`
# log-amplitude): the exact Hessian of the profiled χ²/2 less the held-`S`
# Gauss–Newton curvature `H` (`_data_curvature!`). Per `J`, with
# `z_c = conj(u_c)·r_c`, `p_c = w_c|u_c|²`, `D = Σ p_c` and `S = Σ z_c / D`,
# the phase Hessian is `diag(Re(S̄z_c)) − Re(z zᴴ)/D` and the log-amplitude
# Hessian `diag(2|S|²p_c − Re(S̄z_c)) − Re(z zᴴ)/D + 2(t pᵀ + p tᵀ)/D −
# 4|S|²ppᵀ/D` with `t = Re(S̄z)`, over each cell's `x_a + σ·x_b`; `H` is
# `diag(|S|²p_c)`.
function _profile_product!(ys, xs, g, S, cells::BandpassCells, σ, executor)
    (; r, w, na, nb) = cells
    parts = tmap(cells.chunks; scheduler = executor) do js
        y = zeros(length(ys))
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
        y
    end
    return _sum_into!(ys, parts)
end

# `(Δ − L)·x` over both observables at once (`_profile_product!`), with the
# Hessian's cross block between them: in each cell's `ρ_c = ℓ_a + ℓ_b` and
# `θ_c = φ_a − φ_b`, `∂²/∂ρ_c∂θ_d` of the profiled χ²/2 is
# `−Im(z̄_c z_d)/D − δ_cd·Im(S̄z_c) + 2p_c·Im(S̄z_d)/D`, which vanishes where the
# model fits the data. Writes `yφ` and `yℓ` from the slot vectors `xφ` and `xℓ`.
function _joint_profile_product!(yφ, yℓ, xφ, xℓ, g, S, cells::BandpassCells, executor)
    (; r, w, na, nb) = cells
    parts = tmap(cells.chunks; scheduler = executor) do js
        φy, ℓy = zeros(length(yφ)), zeros(length(yℓ))
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
        (φy, ℓy)
    end
    fill!(yφ, 0.0)
    fill!(yℓ, 0.0)
    for (φy, ℓy) in parts
        yφ .+= φy
        yℓ .+= ℓy
    end
    return yφ, yℓ
end

# `ys = −Σ_J q (qᵀ xs) / q_J`, the Gauss–Newton form of `_profile_product!`, with
# `q_c = w|u_c|²|S|²` and `q_J = Σ_c q_c`: the Schur complement of `S`, which is
# the profile's correction wherever the model fits the data.
function _gauss_newton_product!(ys, xs, g, S, cells::BandpassCells, σ, executor)
    (; w, na, nb) = cells
    parts = tmap(cells.chunks; scheduler = executor) do js
        y = zeros(length(ys))
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
        y
    end
    return _sum_into!(ys, parts)
end

function _sum_into!(ys, parts)
    fill!(ys, 0.0)
    foreach(p -> ys .+= p, parts)
    return ys
end

# The profile of one observable at the current linearization: the slot gains
# `g`, `S`, the cells, `σ`, the system's free slots in unknown order (`slots[i]`
# is unknown `i`), slot-vector buffers, whether its curvature is the exact
# Hessian (`_profile_product!`) or the Gauss–Newton one
# (`_gauss_newton_product!`), and the scheduler of its passes.
struct Profile{C, X}
    g::Vector{ComplexF64}
    S::Vector{ComplexF64}
    cells::C
    σ::Float64
    slots::Vector{Int}
    xs::Vector{Float64}
    ys::Vector{Float64}
    exact::Bool
    executor::X
end
Profile(g, S, cells, σ, free, exact, executor) =
    Profile(g, S, cells, Float64(σ), findall(free), zeros(length(g)), zeros(length(g)), exact, executor)

# The profile's correction (`_profile_product!` or `_gauss_newton_product!`)
# times `x` over the free slots, the first entries of `x`.
function _profile_apply!(y, p::Profile, x)
    fill!(p.xs, 0.0)
    for (i, n) in pairs(p.slots)
        p.xs[n] = x[i]
    end
    (p.exact ? _profile_product! : _gauss_newton_product!)(p.ys, p.xs, p.g, p.S, p.cells, p.σ, p.executor)
    for (i, n) in pairs(p.slots)
        y[i] = p.ys[n]
    end
    return y
end

"""
    ProfiledCurvature(K, profile)

The Hessian of the profiled objective over a system's unknowns: `K`, the
held-`S` Gauss–Newton quadratic with the priors (and any damping), plus the
profile's correction on the free slots, which are the first unknowns
(`_profile_product!`). Applied matrix-free, one pass over the cells per
product; its entries are not stored, and indexing it throws.
"""
struct ProfiledCurvature{M, P} <: AbstractMatrix{Float64}
    K::M
    profile::P
    buffer::Vector{Float64}
end
ProfiledCurvature(K, profile::Profile) = ProfiledCurvature(K, profile, zeros(length(profile.slots)))

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
observables (`_joint_profile_product!`) on the free slots of the profiles `φ`
and `ℓ`. Applied matrix-free, as [`ProfiledCurvature`](@ref).
"""
struct JointCurvature{Mφ, Mℓ, P} <: AbstractMatrix{Float64}
    Kφ::Mφ
    Kℓ::Mℓ
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

# Adds the joint profile correction (`_joint_profile_product!`) at the free slots
# `xφ`, `xℓ` (the first entries of each) to `yφ`, `yℓ`.
function _joint_apply!(yφ, yℓ, φ::Profile, ℓ::Profile, xφ, xℓ)
    fill!(φ.xs, 0.0)
    fill!(ℓ.xs, 0.0)
    foreach(((i, n),) -> φ.xs[n] = xφ[i], pairs(φ.slots))
    foreach(((i, n),) -> ℓ.xs[n] = xℓ[i], pairs(ℓ.slots))
    _joint_profile_product!(φ.ys, ℓ.ys, φ.xs, ℓ.xs, φ.g, φ.S, φ.cells, φ.executor)
    foreach(((i, n),) -> yφ[i] += φ.ys[n], pairs(φ.slots))
    foreach(((i, n),) -> yℓ[i] += ℓ.ys[n], pairs(ℓ.slots))
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
`K` sees them through every correlation of the track.
"""
struct ProfilePreconditioner{F, E}
    F::F
    Z::SparseMatrixCSC{Float64, Int}
    U::Matrix{Float64}
    E::E
end

ProfilePreconditioner(F, Z, U) = ProfilePreconditioner(F, Z, U, cholesky(Symmetric(Matrix(Z' * U))))

function LinearAlgebra.ldiv!(y::AbstractVector, P::ProfilePreconditioner, x::AbstractVector)
    c = P.E \ (P.Z' * x)
    v = P.F \ (x - P.U * c)
    copyto!(y, v)
    mul!(y, P.Z, c - P.E \ (P.U' * v), true, true)
    return y
end

# `M⁻¹` of `P`, factoring the system `K`, as a function of a pair of unknowns
# `(i, j)` on `K`'s sparsity pattern (`_selected_inverse`): the posterior
# covariance the preconditioner approximates.
function _covariance(P::ProfilePreconditioner, K)
    selected = _selected_inverse(P.F, K)
    Einv = inv(P.E)
    G = (P.F \ P.U) * Einv
    Z = Matrix(P.Z)
    B = Einv + G' * P.U * Einv
    Zt, Gt, ZBt = permutedims(Z), permutedims(G), permutedims(Z * B)
    return function (i, j)
        zi, zj = view(Zt, :, i), view(Zt, :, j)
        return selected(i, j) - dot(zi, view(Gt, :, j)) - dot(view(Gt, :, i), zj) + dot(view(ZBt, :, i), zj)
    end
end

# The coarse space of `ProfilePreconditioner` for the system `sys` and its image
# under the curvature: one column per track that holds its constant by a level
# (`held`), one on its free slots, and `(K − H)·Z + λ·Z`, the priors' curvature
# along it plus the damping. `S` absorbs the constant, so the data's curvature
# along it is zero.
function _coarse_space(sys, tracks, λ)
    (; layout) = sys
    unknown = cumsum(sys.free)
    I, J = Int[], Int[]
    m = 0
    for t in tracks
        t.held > 0 || continue
        rows = [unknown[n] for n in t.slots if sys.free[n]]
        isempty(rows) && continue
        m += 1
        append!(I, rows)
        append!(J, fill(m, length(rows)))
    end
    n = size(layout.K, 1)
    Z = sparse(I, J, ones(length(I)), n, m)
    prior = SparseMatrixCSC(n, n, layout.K.colptr, rowvals(layout.K), nonzeros(layout.K) .- layout.base)
    U = Matrix(prior * Z)
    nf = length(sys.xf)
    U[1:nf, :] .+= λ .* Matrix(Z[1:nf, :])
    return Z, U
end

# The reduction of a step's residual at which its Krylov solve stops.
const _FORCING = 0.1

"""
    _profiled_step(sys, c, profile, tracks, λ, linsolve; abstol, z0 = nothing) -> (; z, x′, model, iterations, P, K)

One damped Newton step of the profiled objective for the system `sys`
(`_posterior_system`, assembled with constant `c`): the minimum over its
unknowns of the held-`S` quadratic plus the profile's correction (exact or
Gauss–Newton, as `profile` holds), plus `λ/2·‖x′ − x‖²` over the free slots.
It is solved by `linsolve`, a Krylov method of LinearSolve.jl, preconditioned by
`ProfilePreconditioner` over the constants of `tracks`, from `z0`, the unknowns
at the current slots (the last step's), or else from the preconditioner's own
solution. Started at the current point, the initial residual is the
objective's gradient, and the solve stops once the residual in the
preconditioned norm, which measures the error in the step's own standard
errors, falls below `abstol` or a tenth of its start (an inexact Newton step;
Eisenstat & Walker 1996). Returns the unknowns, the new slot values, the
undamped model's value there, the Krylov iterations taken, the
preconditioner, and the damped system it factors.
"""
function _profiled_step(sys, c, profile, tracks, λ, linsolve; abstol, z0 = nothing)
    K, h, _ = _reduce!(sys, c, λ)
    F = _refactor!(sys, K)
    nf = length(sys.xf)
    P = ProfilePreconditioner(F, _coarse_space(sys, tracks, λ)...)
    A = ProfiledCurvature(K, profile)
    rhs = copy(h)
    view(rhs, 1:nf) .+= _profile_apply!(zeros(nf), profile, sys.xf)
    z = isnothing(z0) ? ldiv!(similar(rhs), P, rhs) : copy(z0)
    residual = rhs - mul!(similar(rhs), A, z)
    sol = LinearSolve.solve(LinearSolve.LinearProblem(A, residual), linsolve; Pl = P, abstol, reltol = _FORCING, maxiters = 10 * size(K, 1))
    LinearSolve.SciMLBase.successful_retcode(sol.retcode) || error(
        "the profiled bandpass step did not converge: $(nameof(typeof(linsolve))) returned $(sol.retcode) " *
            "after $(sol.iters) iterations",
    )
    z .+= sol.u
    x′ = copy(sys.x)
    x′[sys.free] .= view(z, 1:nf)
    d = view(z, 1:nf) .- sys.xf
    (; layout) = sys
    model = dot(z, layout.K * z) / 2 - dot(layout.h, z) + c + dot(d, _profile_apply!(zeros(nf), profile, d)) / 2
    return (; z, x′, model, iterations = sol.iters, P, K)
end

"""
    _joint_step(φ, ℓ, tracks, λ, linsolve; abstol, z0 = nothing) -> (; z, x′, model, iterations)

`_profiled_step` over both observables at once, with the exact Hessian's cross
block between them (`JointCurvature`): `φ` and `ℓ` are the observables' systems
(`_profiled_systems`), `tracks` their acting tracks and `λ` their dampings, as
`(; phase, amp)`, and `z0` the unknowns at the current slots as `(; phase, amp)`.
Away from the mode, where the model leaves residuals as large as the signal, the
cross block is as large as the diagonal ones.
"""
function _joint_step(φ, ℓ, tracks, λ, linsolve; abstol, z0 = nothing)
    Kφ, hφ, _ = _reduce!(φ.sys, φ.c, λ.phase)
    Kℓ, hℓ, _ = _reduce!(ℓ.sys, ℓ.c, λ.amp)
    Pφ = ProfilePreconditioner(_refactor!(φ.sys, Kφ), _coarse_space(φ.sys, tracks.phase, λ.phase)...)
    Pℓ = ProfilePreconditioner(_refactor!(ℓ.sys, Kℓ), _coarse_space(ℓ.sys, tracks.amp, λ.amp)...)
    nφ = size(Kφ, 1)
    A = JointCurvature(Kφ, Kℓ, φ.profile, ℓ.profile)
    P = BlockPreconditioner(Pφ, Pℓ, nφ)
    rhs = [hφ; hℓ]
    _joint_apply!(view(rhs, 1:nφ), view(rhs, (nφ + 1):length(rhs)), φ.profile, ℓ.profile, φ.sys.xf, ℓ.sys.xf)
    z = isnothing(z0) ? ldiv!(similar(rhs), P, rhs) : [z0.phase; z0.amp]
    residual = rhs - mul!(similar(rhs), A, z)
    sol = LinearSolve.solve(LinearSolve.LinearProblem(A, residual), linsolve; Pl = P, abstol, reltol = _FORCING, maxiters = 10 * length(rhs))
    LinearSolve.SciMLBase.successful_retcode(sol.retcode) || error(
        "the profiled bandpass step did not converge: $(nameof(typeof(linsolve))) returned $(sol.retcode) " *
            "after $(sol.iters) iterations",
    )
    z .+= sol.u
    zφ, zℓ = view(z, 1:nφ), view(z, (nφ + 1):length(z))
    nfφ, nfℓ = length(φ.sys.xf), length(ℓ.sys.xf)
    dφ, dℓ = zφ[1:nfφ] .- φ.sys.xf, zℓ[1:nfℓ] .- ℓ.sys.xf
    cφ, cℓ = _joint_apply!(zeros(nfφ), zeros(nfℓ), φ.profile, ℓ.profile, dφ, dℓ)
    held(s, z) = dot(z, s.sys.layout.K * z) / 2 - dot(s.sys.layout.h, z) + s.c
    model = held(φ, zφ) + held(ℓ, zℓ) + (dot(dφ, cφ) + dot(dℓ, cℓ)) / 2
    x′(s, z) = (x = copy(s.sys.x); x[s.sys.free] .= view(z, eachindex(s.sys.xf)); x)
    return (; z = (; phase = collect(zφ), amp = collect(zℓ)), x′ = (; phase = x′(φ, zφ), amp = x′(ℓ, zℓ)), model, iterations = sol.iters)
end

# Each observable's MAP system at the gains `x` (`(; phase, amp)` slot values):
# `(; sys, c, profile, scale)`, `scale` its largest data curvature, or `nothing`
# where the observable is not fit, with the linearization `lin`, `S` re-solved.
# `ctx` is the solve's fixed structure (built in `solve_joint_bandpass!`), and
# `exact` chooses the profile's curvature.
function _profiled_systems(x, ctx, priors, S; exact)
    (; cells, fitted, free, filled, tracks, acting, key) = ctx
    g = exp.(complex.(x.amp, x.phase))
    _source!(S, g, cells, ctx.executor)
    lin = _linearize(g, S, cells, ctx.nslots, ctx.npairs, ctx.executor)
    H = (; phase = _data_curvature!(ctx.Hφ, ctx.hpos, lin.q, -1), amp = _data_curvature!(ctx.Hℓ, ctx.hpos, lin.q, 1))
    b = (; phase = imag.(lin.c), amp = real.(lin.c))
    systems = map((; phase = :phase, amp = :amp)) do obs
        fitted[obs] || return nothing
        d = diag(H[obs])
        scale = maximum(d; init = 0.0)
        moving = free[obs] .& ((d .> eps() * scale) .| filled[obs])
        sys = _cached_system!(ctx.systems, obs, H[obs], b[obs], x[obs], moving, tracks[obs][acting[obs]], key)
        c = _assemble!(sys.layout, priors[obs])
        profile = Profile(g, S, cells, obs === :phase ? -1 : 1, moving, exact, ctx.executor)
        return (; sys, c, profile, scale)
    end
    return systems, lin
end

"""
    _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, μ = 1e-3, exact = false) -> NamedTuple

The MAP of the gains `x` (`(; phase, amp)` slot values, updated in place) under
the resolved `priors`, by Levenberg–Marquardt on the profiled objective: χ²/2
with `S` at its optimum plus the acting tracks' prior energy. Each step is
`_profiled_step`, damped by `μ` times the largest data curvature of its
observable, with `μ` following how well the step's model predicted the
objective's decrease (Madsen, Nielsen & Tingleff 2004, §3.2). The steps take
the Gauss–Newton curvature until one moves no slot by more than one standard
error, or by fewer than ten while the objective falls by more than 1.5 times
the model's prediction, and the exact Hessian from then on; with `exact`, a
start already in the mode's basin, from the first step. The solve stops
once the undamped step moves no slot by `tolerance` of its standard error from
the data, once an accepted step lowers the objective by less than
`tolerance²/2` (a plateau), or once no damping lowers it.

Returns `(; converged, stalled, change, worst, iterations, krylov, μ)`: the
largest weighted change of the last step and its slot, the steps and Krylov
iterations taken, and the final damping.
"""
function _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, μ = 1.0e-3, exact::Bool = false)
    (; cells, fitted, tracks, acting, track_layouts) = ctx
    S = zeros(ComplexF64, length(cells.ranges))
    energy(obs, v) = sum((first(_track_energy(track_layouts[obs][i], priors[obs][j], v)) for (j, i) in pairs(acting[obs])); init = 0.0)
    objective(ℓ, φ) = _profiled_chi2(exp.(complex.(ℓ, φ)), cells, ctx.executor) / 2 + energy(:phase, φ) + energy(:amp, ℓ)
    abstol = tolerance / 10
    change, worst, stalled, converged, near = Inf, 0, false, false, false
    ν, iterations, krylov = 2.0, 0, 0
    O0 = objective(x.amp, x.phase)
    for _ in 1:max_iterations
        iterations += 1
        systems, lin = _profiled_systems(x, ctx, priors, S; exact)
        z0 = map(obs -> _warm_start(ctx, obs, systems[obs]), (; phase = :phase, amp = :amp))
        function propose(μ)
            if exact && all(fitted)
                λ = map(s -> μ * s.scale, systems)
                acting_tracks = map(obs -> tracks[obs][acting[obs]], (; phase = :phase, amp = :amp))
                step = _joint_step(
                    systems.phase, systems.amp, acting_tracks, λ, linsolve;
                    abstol, z0 = any(isnothing, z0) ? nothing : z0,
                )
                krylov += step.iterations
                return step.x′, step.model, step.z
            end
            steps = map((:phase, :amp)) do obs
                s = systems[obs]
                isnothing(s) && return (x′ = x[obs], model = 0.0, iterations = 0, z = nothing)
                return _profiled_step(s.sys, s.c, s.profile, tracks[obs][acting[obs]], μ * s.scale, linsolve; abstol, z0 = z0[obs])
            end
            krylov += steps[1].iterations + steps[2].iterations
            return (; phase = steps[1].x′, amp = steps[2].x′), steps[1].model + steps[2].model, (; phase = steps[1].z, amp = steps[2].z)
        end
        if near
            x1, _, _ = propose(eps())
            change, worst = _largest_weighted_change(x1.amp .- x.amp, x1.phase .- x.phase, lin.W)
            change < tolerance && (converged = true; break)
        end
        E0 = energy(:phase, x.phase) + energy(:amp, x.amp)
        accepted, decrease = false, Inf
        while isfinite(μ)
            x1, predicted, z1 = propose(μ)
            change, worst = _largest_weighted_change(x1.amp .- x.amp, x1.phase .- x.phase, lin.W)
            _within_rounding(x1.amp, x.amp) && _within_rounding(x1.phase, x.phase) && break
            O1 = objective(x1.amp, x1.phase)
            # The objective is known to its rounding.
            if O1 <= O0 + 16 * eps(abs(O0))
                ρ = (O0 - O1) / (E0 - predicted)
                decrease = O0 - O1
                x.amp .= x1.amp
                x.phase .= x1.phase
                foreach(obs -> _keep_warm!(ctx, obs, systems[obs], z1[obs]), (:phase, :amp))
                O0 = O1
                μ = max(μ * max(1 / 3, 1 - (2ρ - 1)^3), eps())
                exact |= ρ > 1.5 && change < 10
                ν = 2.0
                accepted = true
                break
            end
            μ *= ν
            ν *= 2
        end
        # A step lowers the objective by about half its squared change in
        # standard errors, so one too small for that to exceed the objective's
        # rounding cannot be verified, and ending there is not a stall.
        if !accepted
            stalled = change >= tolerance && change^2 / 2 > 16 * eps(abs(O0))
            converged = !stalled
            break
        end
        decrease < tolerance^2 / 2 && (converged = true; break)
        near = change < tolerance
        exact |= change < 1
    end
    return (; converged, stalled, change, worst, iterations, krylov, μ, objective = O0)
end

"""
    _em_hypers!(priors, x, ctx; linsolve, tolerance) -> Float64

One EM update of the hyperparameters the acting tracks' priors give as
densities, at the MAP gains `x`: the E-step takes the moments of each track's
state chain under the Gaussian posterior at the MAP, with the curvature
`ProfilePreconditioner` inverts (held-`S`, except on the track constants,
which keep their prior variance); the M-step
maximizes each track's expected log prior density plus its hyperprior over the
log hyperparameters. Writes the new hyperparameters into `priors` and returns
the largest change of a log hyperparameter.
"""
function _em_hypers!(priors, x, ctx; linsolve, tolerance)
    (; tracks, acting) = ctx
    S = zeros(ComplexF64, length(ctx.cells.ranges))
    systems, _ = _profiled_systems(x, ctx, priors, S; exact = true)
    Δ = 0.0
    for obs in (:phase, :amp)
        s = systems[obs]
        isnothing(s) && continue
        estimating = [k for (k, i) in pairs(acting[obs]) if _hyper_count(tracks[obs][i].prior) > 0]
        isempty(estimating) && continue
        step = _profiled_step(
            s.sys, s.c, s.profile, tracks[obs][acting[obs]], 0.0, linsolve;
            abstol = tolerance / 10, z0 = _warm_start(ctx, obs, s),
        )
        cov = _covariance(step.P, step.K)
        for k in estimating
            chains = s.sys.chains[k]
            mom = _chain_moments(chains, step.z, cov, first(chains).dim)
            priors[obs][k], δ = _m_step(tracks[obs][acting[obs][k]].prior, priors[obs][k], chains, mom)
            Δ = max(Δ, δ)
        end
    end
    return Δ
end

# Each slot's summed data weight over the cells.
function _slot_weights(cells::BandpassCells, nslots)
    out = zeros(nslots)
    for (k, wk) in pairs(cells.w)
        out[cells.na[k]] += wk
        out[cells.nb[k]] += wk
    end
    return out
end

# Track `t` with the constant `S` absorbs held: by a gauge pin on one of its
# slots; else by its first acting level group (`held`) under a proper prior,
# which the prior then sets; else by its slot of most data `weight`, marked in
# `pins`. `held` is `0` where no level group holds it.
function _hold_track_constant!(pins, t, gauge_pins, hasdata, weight)
    any(n -> gauge_pins[n], t.slots) && return merge(t, (; held = 0))
    j = findfirst(t.active)
    !isnothing(j) && _proper_prior(t.prior) && return merge(t, (; held = _level_group(t, t.prior, j)))
    measured = [n for n in t.slots if hasdata[n]]
    isempty(measured) || (pins[measured[argmax(weight[measured])]] = true)
    return merge(t, (; held = 0))
end

# The M-step for one track: `prior` (resolved from the `spec` that gives some
# hyperparameters as densities) with those hyperparameters at the maximum of the
# chains' expected log prior density under the state moments `mom`
# (`_chain_moments`) plus the hyperprior, and the largest change of a log
# hyperparameter.
function _m_step(spec, prior, chains, mom)
    sums = _moment_sums(chains, mom)
    function f(lθ)
        v = _expected_log_prior(_with_log_hypers(spec, lθ), sums, eltype(lθ)) + _log_hyperprior(spec, lθ)
        return isfinite(v) ? -v : oftype(v, Inf)
    end
    lθ0 = _log_hypers(spec, prior)
    lθ1, _ = _nelder_mead(f, lθ0; atol = 1.0e-10)
    return _with_log_hypers(spec, lθ1), maximum(abs, lθ1 .- lθ0)
end

# The hyperparameters the acting tracks' priors give as densities, as
# `(observable, position in acting)` pairs.
_estimated(ctx) = [(obs, k) for obs in (:phase, :amp) for (k, i) in pairs(ctx.acting[obs]) if _hyper_count(ctx.tracks[obs][i].prior) > 0]

_log_hyper_vector(priors, ctx, est) =
    reduce(vcat, [_log_hypers(ctx.tracks[obs][ctx.acting[obs][k]].prior, priors[obs][k]) for (obs, k) in est]; init = Float64[])

function _set_log_hypers!(priors, ctx, est, ψ)
    i = 0
    for (obs, k) in est
        spec = ctx.tracks[obs][ctx.acting[obs][k]].prior
        n = _hyper_count(spec)
        priors[obs][k] = _with_log_hypers(spec, ψ[(i + 1):(i + n)])
        i += n
    end
    return priors
end

"""
    _squarem!(priors, x, ctx, solved; tolerance, max_iterations, max_em_iterations, linsolve)
        -> (; solved, converged, evaluations)

The hyperparameters the priors give as densities by MAP-EM, accelerated by
SQUAREM (Varadhan & Roland 2008, scheme S3) over their logs, written into
`priors` with the gains `x` at their MAP under them. One EM map is the MAP of
the gains (`_profiled_map!`, warm from the last, with the exact Hessian)
followed by `_em_hypers!`.
Each cycle takes two EM maps from `ψ₀`, extrapolates along them by the step
`α = −‖r‖/‖v‖`, bounded by a length that grows after each step it bounds, and
takes one EM map from there. EM is slow where the data barely determine a
direction, so a plain EM step understates the distance to its fixed point; the
cycles stop once one moves no log hyperparameter by more than `tolerance`, or
after `max_em_iterations` EM maps. `solved` is the MAP at the starting
hyperparameters, and the result's the MAP at the final ones.
"""
function _squarem!(priors, x, ctx, solved; tolerance, max_iterations, max_em_iterations, linsolve)
    est = _estimated(ctx)
    evaluations = 0
    function em_map(ψ)
        _set_log_hypers!(priors, ctx, est, ψ)
        solved = _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, solved.μ, exact = true)
        _em_hypers!(priors, x, ctx; linsolve, tolerance)
        evaluations += 1
        return _log_hyper_vector(priors, ctx, est)
    end
    ψ0 = _log_hyper_vector(priors, ctx, est)
    # The MAP at ψ0 is current: the first map needs no new MAP.
    _em_hypers!(priors, x, ctx; linsolve, tolerance)
    evaluations += 1
    ψ1 = _log_hyper_vector(priors, ctx, est)
    αmax, converged = 1.0, false
    while evaluations + 2 <= max_em_iterations
        ψ2 = em_map(ψ1)
        r, v = ψ1 - ψ0, ψ2 - 2ψ1 + ψ0
        α = norm(v) > 0 ? -norm(r) / norm(v) : -1.0
        bounded = α < -αmax
        α = clamp(α, -αmax, -1.0)
        ψ′ = ψ0 - 2α * r + α^2 * v
        ψ3 = em_map(ψ′)
        bounded && (αmax *= 4)
        step = maximum(abs, ψ3 - ψ0; init = 0.0)
        ψ0 = ψ3
        if step < tolerance
            converged = true
            break
        end
        ψ1 = em_map(ψ0)
    end
    _set_log_hypers!(priors, ctx, est, ψ0)
    solved = _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, solved.μ, exact = true)
    return (; solved, converged, evaluations)
end

# `_posterior_system` of one observable, its structure kept in `cache[obs]` while
# the free slots are unchanged and its values rewritten for the curvature `H`,
# the gradient `b` and the slot values `x`: the prior terms' pattern and the
# factor's symbolic analysis carry over. `H` keeps one sparsity pattern
# throughout (`_curvature_pattern`).
function _cached_system!(cache, obs, H, b, x, free, tracks, key)
    entry = get(cache, obs, nothing)
    if isnothing(entry) || entry.sys.free != free
        sys = _posterior_system(H, b, x, free, tracks, key)
        Is, Js, _ = findnz(H[free, free])
        cache[obs] = (; sys, data = [_position(sys.layout.K, i, j) for (i, j) in zip(Is, Js)])
        return sys
    end
    sys = _update_system(entry.sys, entry.data, H, b, x)
    cache[obs] = (; sys, entry.data)
    return sys
end

function _update_system(sys, data, H, b, x)
    (; layout, free) = sys
    Hf = H[free, free]
    sys.x .= x
    sys.xf .= view(x, free)
    fill!(layout.base, 0.0)
    for (p, v) in zip(data, nonzeros(Hf))
        layout.base[p] += v
    end
    nf = length(sys.xf)
    Hx = Hf * sys.xf
    fill!(layout.h0, 0.0)
    view(layout.h0, 1:nf) .= view(b, free) .+ Hx
    c0 = dot(sys.xf, Hx) / 2 + dot(view(b, free), sys.xf)
    return merge(sys, (; layout = merge(layout, (; c0))))
end

# The unknowns of the last accepted step of observable `obs`, which hold the
# current slots, while its system `s` keeps their layout; `nothing` otherwise.
function _warm_start(ctx, obs, s)
    isnothing(s) && return nothing
    w = get(ctx.warm, obs, nothing)
    return isnothing(w) || w.free != s.sys.free ? nothing : w.z
end

_keep_warm!(ctx, obs, s, z) = isnothing(s) || isnothing(z) || (ctx.warm[obs] = (; free = copy(s.sys.free), z = copy(z)))
