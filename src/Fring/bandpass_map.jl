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
# gain and `S`, relative to its current one. The chunks add into `acc`.
function _linearize(
        g, S, cells::BandpassCells, ngains, npairs, executor,
        acc = ChunkAccumulators(cells, zeros(npairs), zeros(ComplexF64, ngains), zeros(ngains)),
    )
    (; r, w, e, na, nb) = cells
    lin = (; q = zeros(npairs), c = zeros(ComplexF64, ngains), W = zeros(ngains))
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
function Profile(g, S, cells, σ, J, exact, executor, acc = ChunkAccumulators(cells, fill!(similar(g, Float64), 0)))
    xs, ys = fill!(similar(g, Float64), 0), fill!(similar(g, Float64), 0)
    return Profile(g, S, cells, Float64(σ), J, xs, ys, exact, executor, acc)
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
product; its entries are not stored, and indexing it throws.
"""
struct ProfiledCurvature{M, P} <: AbstractMatrix{Float64}
    K::M
    profile::P
    buffer::Vector{Float64}
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
# under the curvature: one column per constant group of each track (its
# `constants`), one on its parameters in the pieces of that group, and
# `(K − H)·Z + λ·D·Z`, the priors' curvature along it plus the damping. `S`
# absorbs the constant, so the data's curvature along it is zero.
function _coarse_space(sys, tracks, λ)
    (; layout) = sys
    I, Js = Int[], Int[]
    m = 0
    for (t, chains) in zip(tracks, sys.chains), g in unique(t.constants)
        m += 1
        for (j, ch) in pairs(chains)
            t.constants[j] == g || continue
            append!(I, ch.gains)
            append!(Js, fill(m, length(ch.gains)))
        end
    end
    n = size(layout.K, 1)
    Z = sparse(I, Js, ones(length(I)), n, m)
    prior = SparseMatrixCSC(n, n, layout.K.colptr, rowvals(layout.K), nonzeros(layout.K) .- layout.base)
    U = Matrix(prior * Z)
    nθ = length(sys.θ)
    U[1:nθ, :] .+= λ .* _damping(sys) .* Matrix(Z[1:nθ, :])
    return Z, U
end

# The reduction of a step's residual at which its Krylov solve stops.
const _FORCING = 0.1

"""
    _profiled_step(s, tracks, λ, linsolve; abstol, z0 = _current_unknowns(s.sys)) -> (; z, x′, decrease, iterations, P, K)

One damped Newton step of the profiled objective for the system `s` of one
observable (`_profiled_systems`): the minimum over its unknowns of the held-`S`
quadratic plus the profile's correction (exact or Gauss–Newton, as `profile`
holds), plus `λ/2·Σ d_j(θ′_j − θ_j)²` over the parameters, `d` their curvatures
(`_damping`). It is solved by
`linsolve`, a Krylov method of LinearSolve.jl, preconditioned by
`ProfilePreconditioner` over the constants of `tracks`, from the current point
`z0`, by default the parameters with the latent unknowns at their optimum given
them (`_current_unknowns`). There the initial residual is the objective's
negative gradient (`_step_residual`), and the solve stops once the residual in
the preconditioned norm, which measures the error in the step's own standard
errors, falls below `abstol` or a tenth of its start (an inexact Newton step;
Eisenstat & Walker 1996). Returns the unknowns, the new gain values, the
decrease the undamped model predicts, the Krylov iterations taken, the
preconditioner, and the damped system it factors. The decrease is
`−gᵀδ − δᵀ(A − λ)δ/2` from the gradient `g` and the increment `δ`: the model's
own values are differences of terms of the order of the stiffest prior's
curvature, and lose the decrease to rounding.
"""
function _profiled_step(s, tracks, λ, linsolve; abstol, z0 = _current_unknowns(s.sys))
    (; sys) = s
    K, _, _ = _reduce!(sys, s.c, λ)
    F = _refactor!(sys, K)
    nθ = length(sys.θ)
    P = ProfilePreconditioner(F, _coarse_space(sys, tracks, λ)...)
    A = ProfiledCurvature(K, s.profile)
    z = copy(z0)
    residual = _step_residual(s, z)
    sol = LinearSolve.solve(LinearSolve.LinearProblem(A, residual), linsolve; Pl = P, abstol, reltol = _FORCING, maxiters = 10 * size(K, 1))
    LinearSolve.SciMLBase.successful_retcode(sol.retcode) || error(
        "the profiled bandpass step did not converge: $(nameof(typeof(linsolve))) returned $(sol.retcode) " *
            "after $(sol.iters) iterations",
    )
    δ = sol.u
    z .+= δ
    decrease = _predicted_decrease(A, residual, δ, (1:nθ,), (λ .* _damping(sys),))
    return (; z, x′ = _gains_at(sys, z), decrease, iterations = sol.iters, P, K)
end

# The station-gain values of `sys` at the unknowns `z`, whose first are its `θ`.
_gains_at(sys, z) = sys.J * view(z, eachindex(sys.θ))

# The negative gradient of system `s`'s objective at the unknowns `z`, whose
# station gains are at their current values: the data's `b` on the gains less
# the priors' gradient (`_prior_gradient`).
function _step_residual(s, z)
    r = -_prior_gradient(s.sys.layout, s.priors, z)
    view(r, eachindex(s.sys.bθ)) .+= s.sys.bθ
    return r
end

"""
    _joint_step(φ, ℓ, tracks, λ, linsolve; abstol, z0) -> (; z, x′, decrease, iterations)

`_profiled_step` over both observables at once, with the exact Hessian's cross
block between them (`JointCurvature`): `φ` and `ℓ` are the observables' systems
(`_profiled_systems`), `tracks` their tracks, `λ` their dampings and
`z0` their unknowns at the current point, as `(; phase, amp)`.
Away from the mode, where the model leaves residuals as large as the signal, the
cross block is as large as the diagonal ones.
"""
function _joint_step(φ, ℓ, tracks, λ, linsolve; abstol, z0 = map(s -> _current_unknowns(s.sys), (; phase = φ, amp = ℓ)))
    Kφ, _, _ = _reduce!(φ.sys, φ.c, λ.phase)
    Kℓ, _, _ = _reduce!(ℓ.sys, ℓ.c, λ.amp)
    Pφ = ProfilePreconditioner(_refactor!(φ.sys, Kφ), _coarse_space(φ.sys, tracks.phase, λ.phase)...)
    Pℓ = ProfilePreconditioner(_refactor!(ℓ.sys, Kℓ), _coarse_space(ℓ.sys, tracks.amp, λ.amp)...)
    nφ = size(Kφ, 1)
    A = JointCurvature(Kφ, Kℓ, φ.profile, ℓ.profile)
    P = BlockPreconditioner(Pφ, Pℓ, nφ)
    z = [z0.phase; z0.amp]
    residual = [_step_residual(φ, z0.phase); _step_residual(ℓ, z0.amp)]
    sol = LinearSolve.solve(LinearSolve.LinearProblem(A, residual), linsolve; Pl = P, abstol, reltol = _FORCING, maxiters = 10 * length(z))
    LinearSolve.SciMLBase.successful_retcode(sol.retcode) || error(
        "the profiled bandpass step did not converge: $(nameof(typeof(linsolve))) returned $(sol.retcode) " *
            "after $(sol.iters) iterations",
    )
    δ = sol.u
    z .+= δ
    zφ, zℓ = view(z, 1:nφ), view(z, (nφ + 1):length(z))
    decrease = _predicted_decrease(
        A, residual, δ, (1:length(φ.sys.θ), nφ .+ (1:length(ℓ.sys.θ))), (λ.phase .* _damping(φ.sys), λ.amp .* _damping(ℓ.sys)),
    )
    return (;
        z = (; phase = collect(zφ), amp = collect(zℓ)), x′ = (; phase = _gains_at(φ.sys, zφ), amp = _gains_at(ℓ.sys, zℓ)),
        decrease, iterations = sol.iters,
    )
end

# The decrease of the quadratic with curvature `A`, less the damping `λs[i]`
# (one weight per unknown) on the unknowns `ranges[i]`, along the increment `δ`
# from a point where its negative gradient is `residual`.
function _predicted_decrease(A, residual, δ, ranges, λs)
    damping = sum(((s, λ),) -> sum(λ .* abs2.(view(δ, s))), zip(ranges, λs))
    return dot(residual, δ) - (dot(δ, mul!(similar(δ), A, δ)) - damping) / 2
end

# Each observable's MAP system at the station gains `x` (`(; phase, amp)`
# values): `(; sys, c, profile, priors)`, `priors` its tracks' resolved priors,
# or `nothing` where the observable is not fit, with the linearization `lin`,
# `S` re-solved. `ctx` is the solve's fixed structure (`solve_joint_gains!`):
# each observable's unknowns are its parameters `ctx.params[obs].θ`, carried to
# the station gains by its `J`. `exact` chooses the profile's curvature.
function _profiled_systems(x, ctx, priors, S; exact)
    (; cells, fitted, tracks, params) = ctx
    g = exp.(complex.(x.amp, x.phase))
    _source!(S, g, cells, ctx.executor)
    lin = _linearize(g, S, cells, ctx.ngains, ctx.npairs, ctx.executor, ctx.accumulators.linearize)
    H = (; phase = _data_curvature!(ctx.Hφ, ctx.hpos, lin.q, -1), amp = _data_curvature!(ctx.Hℓ, ctx.hpos, lin.q, 1))
    b = (; phase = imag.(lin.c), amp = real.(lin.c))
    systems = map((; phase = :phase, amp = :amp)) do obs
        fitted[obs] || return nothing
        p = params[obs]
        sys = _cached_system!(ctx.systems, obs, H[obs], b[obs], p.J, p.θ, tracks[obs], p.key; p.ridge)
        c = _assemble!(sys.layout, priors[obs])
        profile = Profile(g, S, cells, obs === :phase ? -1 : 1, p.J, exact, ctx.executor, ctx.accumulators[obs])
        return (; sys, c, profile, priors = priors[obs])
    end
    return systems, lin
end

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
    (; cells, fitted, tracks) = ctx
    S = zeros(ComplexF64, length(cells.ranges))
    observables = (; phase = :phase, amp = :amp)
    energy(systems, z) = sum(obs -> isnothing(systems[obs]) ? 0.0 : _prior_energy(systems[obs].sys.layout, priors[obs], z[obs]), keys(observables))
    objective(systems, x, z) = _profiled_chi2(exp.(complex.(x.amp, x.phase)), cells, ctx.executor) / 2 + energy(systems, z)
    abstol = tolerance / 10
    change, worst, stalled, converged, near = Inf, 0, false, false, false
    ν, iterations, krylov = 2.0, 0, 0
    O0, z = Inf, nothing
    μ_accepted = μ
    local lin
    for _ in 1:max_iterations
        iterations += 1
        systems, lin = _profiled_systems(x, ctx, priors, S; exact)
        if isnothing(z)
            z = map(s -> isnothing(s) ? nothing : _current_unknowns(s.sys), systems)
            O0 = objective(systems, x, z)
        end
        function propose(μ)
            if exact && all(fitted)
                step = _joint_step(systems.phase, systems.amp, tracks, (; phase = μ, amp = μ), linsolve; abstol, z0 = z)
                krylov += step.iterations
                return step.x′, step.decrease, step.z
            end
            steps = map(keys(observables)) do obs
                s = systems[obs]
                isnothing(s) && return (x′ = x[obs], decrease = 0.0, iterations = 0, z = nothing)
                return _profiled_step(s, tracks[obs], μ, linsolve; abstol, z0 = z[obs])
            end
            krylov += steps[1].iterations + steps[2].iterations
            return (; phase = steps[1].x′, amp = steps[2].x′), steps[1].decrease + steps[2].decrease, (; phase = steps[1].z, amp = steps[2].z)
        end
        if near
            x1, _, _ = propose(eps())
            change, worst = _largest_weighted_change(x1.amp .- x.amp, x1.phase .- x.phase, lin.W)
            change < tolerance && (converged = true; break)
        end
        accepted, decrease = false, Inf
        while isfinite(μ)
            x1, predicted, z1 = propose(μ)
            change, worst = _largest_weighted_change(x1.amp .- x.amp, x1.phase .- x.phase, lin.W)
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
            change, worst = _largest_weighted_change(x1.amp .- x.amp, x1.phase .- x.phase, lin.W)
            stalled = change >= tolerance && change^2 / 2 > 16 * eps(abs(O0))
            converged = !stalled
            break
        end
        near = change < tolerance || decrease < tolerance^2 / 2
        exact |= change < 1
    end
    return (; converged, stalled, change, worst, iterations, krylov, μ = μ_accepted, objective = O0, weights = lin.W)
end

"""
    _em_hypers!(priors, x, ctx; linsolve, tolerance) -> Float64

One EM update of the hyperparameters the tracks' priors give as
densities, at the MAP gains `x`: the E-step takes the moments of each track's
state chain under the Gaussian posterior at the MAP, with the curvature
`ProfilePreconditioner` inverts (held-`S`, except on the track constants,
which keep their prior variance); the M-step
maximizes each track's expected log prior density plus its hyperprior over the
log hyperparameters. Writes the new hyperparameters into `priors` and returns
the largest change of a log hyperparameter.
"""
function _em_hypers!(priors, x, ctx; linsolve, tolerance)
    (; tracks) = ctx
    S = zeros(ComplexF64, length(ctx.cells.ranges))
    systems, _ = _profiled_systems(x, ctx, priors, S; exact = true)
    Δ = 0.0
    for obs in (:phase, :amp)
        s = systems[obs]
        isnothing(s) && continue
        estimating = [k for k in eachindex(tracks[obs]) if _hyper_count(tracks[obs][k].prior) > 0]
        isempty(estimating) && continue
        step = _profiled_step(s, tracks[obs], 0.0, linsolve; abstol = tolerance / 10)
        cov = _covariance(step.P, step.K)
        for k in estimating
            chains = s.sys.chains[k]
            mom = _chain_moments(chains, step.z, cov, first(chains).dim)
            priors[obs][k], δ = _m_step(tracks[obs][k].prior, priors[obs][k], chains, mom)
            Δ = max(Δ, δ)
        end
    end
    return Δ
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
    box = _log_hyper_box(spec, _piece_extent([ch.u for ch in chains], [ch.h for ch in chains])...)
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
    _squarem!(priors, x, ctx, solved; tolerance, max_iterations, max_em_iterations, linsolve)
        -> (; solved, converged, evaluations, unconverged, moved)

The hyperparameters the priors give as densities by MAP-EM, accelerated by
SQUAREM (Varadhan & Roland 2008, scheme S3) over their logs, written into
`priors` with the gains `x` at their MAP under them. One EM map is the MAP of
the gains (`_profiled_map!`, warm from the last, with the exact Hessian)
followed by `_em_hypers!`.
Each cycle takes two EM maps from `ψ₀`, extrapolates along them by the step
`α = −‖r‖/‖v‖`, bounded by a length that grows after each step it bounds, and
takes one EM map from there. EM is slow where the data barely determine a
direction, so a plain EM step understates the distance to its fixed point. The
cycles stop once one moves no gain by `tolerance` of its standard error, from
the MAP it starts at to the converged MAP at the extrapolated hyperparameters
(`moved`, the largest such change), or after `max_em_iterations` EM maps: the
hyperparameters the gains no longer depend on are left where EM has taken them.
`solved` is the MAP at the starting hyperparameters, and the result's the MAP at
the final ones; `unconverged` counts the EM maps whose MAP stopped at
`max_iterations`.
"""
function _squarem!(priors, x, ctx, solved; tolerance, max_iterations, max_em_iterations, linsolve)
    est = _estimated(ctx)
    evaluations, unconverged = 0, 0
    function em_map(ψ)
        _set_log_hypers!(priors, ctx, est, ψ)
        solved = _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, solved.μ, exact = true)
        unconverged += !(solved.converged || solved.stalled)
        _em_hypers!(priors, x, ctx; linsolve, tolerance)
        evaluations += 1
        return _log_hyper_vector(priors, ctx, est)
    end
    ψ0 = _log_hyper_vector(priors, ctx, est)
    # The MAP at ψ0 is current: the first map needs no new MAP.
    _em_hypers!(priors, x, ctx; linsolve, tolerance)
    evaluations += 1
    ψ1 = _log_hyper_vector(priors, ctx, est)
    αmax, converged, moved = 1.0, false, Inf
    while evaluations + 2 <= max_em_iterations
        start = map(copy, x)
        ψ2 = em_map(ψ1)
        r, v = ψ1 - ψ0, ψ2 - 2ψ1 + ψ0
        α = norm(v) > 0 ? -norm(r) / norm(v) : -1.0
        bounded = α < -αmax
        α = clamp(α, -αmax, -1.0)
        ψ′ = ψ0 - 2α * r + α^2 * v
        ψ3 = em_map(ψ′)
        bounded && (αmax *= 4)
        moved, _ = _largest_weighted_change(x.amp .- start.amp, x.phase .- start.phase, solved.weights)
        ψ0 = ψ3
        if moved < tolerance && solved.converged
            converged = true
            break
        end
        ψ1 = em_map(ψ0)
    end
    _set_log_hypers!(priors, ctx, est, ψ0)
    solved = _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve, solved.μ, exact = true)
    return (; solved, converged, evaluations, unconverged, moved)
end

# `_posterior_system` of one observable, its structure kept in `cache[obs]`
# and its values rewritten for the curvature `H`, the gradient `b` and the
# parameters `θ`: the prior terms' pattern and the factor's symbolic analysis
# carry over. `H` keeps one sparsity pattern throughout (`_curvature_pattern`),
# and so does `JᵀHJ`.
function _cached_system!(cache, obs, H, b, J, θ, tracks, key; kw...)
    entry = get(cache, obs, nothing)
    if isnothing(entry)
        sys = _posterior_system(H, b, J, θ, tracks, key; kw...)
        Is, Js, _ = findnz(J' * H * J)
        cache[obs] = (; sys, data = [_position(sys.layout.K, i, j) for (i, j) in zip(Is, Js)])
        return sys
    end
    sys = _update_system(entry.sys, entry.data, H, b, θ)
    cache[obs] = (; sys, entry.data)
    return sys
end

function _update_system(sys, data, H, b, θ)
    (; layout, J) = sys
    Hθ = J' * H * J
    length(data) == nnz(Hθ) || error("the data curvature over the parameters changed its sparsity pattern")
    sys.θ .= θ
    sys.bθ .= J' * b
    fill!(layout.base, 0.0)
    for (p, v) in zip(data, nonzeros(Hθ))
        layout.base[p] += v
    end
    nθ = length(sys.θ)
    Hx = Hθ * sys.θ
    fill!(layout.h0, 0.0)
    view(layout.h0, 1:nθ) .= sys.bθ .+ Hx
    c0 = dot(sys.θ, Hx) / 2 + dot(sys.bθ, sys.θ)
    return merge(sys, (; layout = merge(layout, (; c0))))
end
