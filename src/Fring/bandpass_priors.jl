# ── Track priors as sparse quadratics ────────────────────────────────────────
#
# A prior's energy over a track is a sum of state-space terms: each piece (one
# spectral window) is a chain of states along frequency, `s_k = A_k·s_{k−1} + η_k`,
# `η_k ~ N(0, Q_k)`, from the prior's start where it is proper, whose first entry
# is the track's value less its level. Written over the states themselves the
# energy is a sum of small quadratic terms, so a track's prior, its latent states
# and its levels enter one sparse system with the data, exactly and without a
# start variance standing in for a flat start.
#
# The per-scan source term is frequency-flat, so it absorbs any constant added to
# a (station, feed, time segment) track: the data cannot measure a track's band
# constant. The MAP of everything else is then the same as under the prior
# profiled over that constant, so a proper prior acts on its track less a free
# constant, as if beside a level. A flat-start walk is unchanged by a constant.

# A quadratic `½uᵀKu − hᵀu + c` over a growing set of unknowns `u`, each with
# the frequency `key` by which they are ordered for factoring.
mutable struct _Quadratic
    Is::Vector{Int}
    Js::Vector{Int}
    Vs::Vector{Float64}
    h::Vector{Float64}
    c::Float64
    key::Vector{Float64}
end
_Quadratic(key) = _Quadratic(Int[], Int[], Float64[], zeros(length(key)), 0.0, collect(Float64, key))
_unknown!(Q::_Quadratic, key) = (push!(Q.h, 0.0); push!(Q.key, key); length(Q.h))
_matrix(Q::_Quadratic) = sparse(Q.Is, Q.Js, Q.Vs, length(Q.h), length(Q.h))
Base.copy(Q::_Quadratic) = _Quadratic(copy(Q.Is), copy(Q.Js), copy(Q.Vs), copy(Q.h), Q.c, Q.key)

# A linear form `Σ a·u[j] + b`: its terms `j => a` and its constant `b`.
const _Form = Tuple{Vector{Pair{Int, Float64}}, Float64}

# `Σ_j c_j·forms_j`.
function _combine(forms, coefs)
    terms = Pair{Int, Float64}[]
    b = 0.0
    for (f, a) in zip(forms, coefs)
        iszero(a) && continue
        append!(terms, (j => a * v) for (j, v) in f[1])
        b += a * f[2]
    end
    return (terms, b)
end

# Add `½ρᵀWρ` for the forms `ρ`.
function _add_term!(Q::_Quadratic, ρ, W)
    cols = unique(first(t) for f in ρ for t in f[1])
    G = zeros(length(ρ), length(cols))
    r = zeros(length(ρ))
    for (i, f) in pairs(ρ)
        for (j, a) in f[1]
            G[i, findfirst(==(j), cols)] += a
        end
        r[i] = -f[2]
    end
    W = (W + W') / 2
    GW = G' * W
    K = GW * G
    for a in eachindex(cols), b in eachindex(cols)
        push!(Q.Is, cols[a])
        push!(Q.Js, cols[b])
        push!(Q.Vs, K[a, b])
    end
    h = GW * r
    for (a, j) in pairs(cols)
        Q.h[j] += h[a]
    end
    Q.c += dot(r, W * r) / 2
    return Q
end

# The coordinates a prior's state-space model runs on over one piece's
# frequencies `x`, and their unit `h` in `x`.
function _piece_coordinates(::RandomWalkPrior, x)
    _, u, h = _random_walk_problem(x, ones(length(x)), x)
    return u, h
end
_piece_coordinates(::OUPrior, x) = float.(x), 1.0

# The state-space model of a resolved prior on coordinates of unit `h`, in
# element type `T`.
_piece_model(prior::RandomWalkPrior, h, T = Float64) = _random_walk_model(prior.order, prior.σ, prior.init, h, T)
_piece_model(prior::OUPrior, h, T = Float64) = OUModel(T(prior.scale), T(prior.σ)^2)
_statedim(prior::RandomWalkPrior) = prior.order
_statedim(::OUPrior) = 1

# The samples a flat-start walk needs to have its state determined; one for a
# proper prior.
_samples_needed(prior::RandomWalkPrior) = _proper_prior(prior) ? 1 : prior.order
_samples_needed(::OUPrior) = 1

# The level group of track `t`'s piece `j`: its level segment beside a level,
# one group for the whole track under a proper prior (the constant the source term
# absorbs), and `nothing` under a flat-start walk, which no constant changes.
_level_group(t, prior, j) = !isnothing(t.level) ? t.level[j] : _proper_prior(prior) ? 1 : nothing
_level_groups(t, prior) = !isnothing(t.level) ? t.level : _proper_prior(prior) ? ones(Int, length(t.pieces)) : nothing

# The state chain of each piece of track `t` its prior acts on, over the unknowns
# of `Q`: per sample its state less the piece's level, as linear forms whose
# first is the slot (`slot(n)` gives its unknown, or `0` and its fixed value) and
# the rest new latent unknowns, with the coordinates its prior runs on. Returns
# the chains and the unknown of each level group.
function _track_chains!(Q::_Quadratic, t, slot)
    levels = Dict{Int, Int}()
    chains = NamedTuple[]
    for (j, p) in pairs(t.pieces)
        t.active[j] || continue
        group = _level_group(t, t.prior, j)
        L = isnothing(group) ? 0 : get!(() -> _unknown!(Q, Inf), levels, group)
        states = map(p) do k
            j1, v1 = slot(t.slots[k])
            value = (iszero(j1) ? Pair{Int, Float64}[] : [j1 => 1.0], iszero(j1) ? v1 : 0.0)
            iszero(L) || push!(value[1], L => -1.0)
            _Form[value; [([_unknown!(Q, t.x[k]) => 1.0], 0.0) for _ in 2:_statedim(t.prior)]]
        end
        u, h = _piece_coordinates(t.prior, t.x[p])
        push!(chains, (; u, h, states))
    end
    return chains, levels
end

# Add the energy of `chains` under the resolved `prior` to `Q`: a term per
# transition and, where the prior's start is proper, one for the first state.
function _add_prior_terms!(Q::_Quadratic, chains, prior)
    for ch in chains
        model = _piece_model(prior, ch.h)
        start = initial(model, Float64)
        if !isnothing(start)
            μ0, P0 = start
            _add_term!(Q, [(f[1], f[2] - μ0[i]) for (i, f) in pairs(ch.states[1])], inv(Matrix(P0)))
        end
        for i in 2:length(ch.states)
            A, Qk = transition(model, ch.u[i] - ch.u[i - 1], Float64)
            ρ = [_combine([ch.states[i][r]; ch.states[i - 1]], [1.0; -A[r, :]]) for r in axes(A, 1)]
            _add_term!(Q, ρ, inv(Matrix(Qk)))
        end
    end
    return Q
end

# The log density of `chains` under the resolved `prior`, in expectation over
# the states' posterior with the moments `mom` (`_chain_moments`), or its
# normalization alone without them; in element type `T`.
_expected_log_prior(prior, chains, mom, T) = sum(
    (_expected_chain_log_prior(_piece_model(prior, ch.h, T), ch.u, isnothing(mom) ? nothing : mom[c], T) for (c, ch) in pairs(chains));
    init = zero(T),
)

function _expected_chain_log_prior(model, u, mom, T)
    D = statedim(model)
    out = zero(T)
    start = initial(model, T)
    if !isnothing(start)
        μ0, P0 = start
        out -= logdet(2π * P0) / 2
        if !isnothing(mom)
            m, M = SVector{D}(mom.start[1]), SMatrix{D, D}(mom.start[2])
            out -= tr(P0 \ (M - μ0 * m' - m * μ0' + μ0 * μ0')) / 2
        end
    end
    for i in 2:length(u)
        A, Q = transition(model, u[i] - u[i - 1], T)
        out -= logdet(2π * Q) / 2
        isnothing(mom) && continue
        G = hcat(SMatrix{D, D, T}(I), -A)
        out -= tr(Q \ (G * SMatrix{2D, 2D}(mom.pairs[i - 1]) * G')) / 2
    end
    return out
end

# The posterior moments of each chain's states, from the mean `z` and covariance
# `cov(i, j)` of the unknowns: the first state's mean and second moment, and the
# second moment of each consecutive pair `(s_i, s_{i−1})`.
function _chain_moments(chains, z, cov)
    mean_of(f) = sum((a * z[j] for (j, a) in f[1]); init = 0.0) + f[2]
    second(fs) = [
        sum((a * b * cov(i, j) for (i, a) in f[1] for (j, b) in g[1]); init = 0.0) + mean_of(f) * mean_of(g)
            for f in fs, g in fs
    ]
    return map(chains) do ch
        (;
            start = (mean_of.(ch.states[1]), second(ch.states[1])),
            pairs = [second([ch.states[i]; ch.states[i - 1]]) for i in 2:length(ch.states)],
        )
    end
end

# Track `t`'s prior energy at the slot values `x`, minimized over its latent
# states and levels, and those levels by group.
function _track_energy(t, prior, x)
    Q = _Quadratic(Float64[])
    chains, levels = _track_chains!(Q, t, n -> (0, x[n]))
    _add_prior_terms!(Q, chains, prior)
    isempty(Q.h) && return Q.c, Dict{Int, Float64}()
    u = cholesky(Symmetric(_matrix(Q))) \ Q.h
    return Q.c - dot(Q.h, u) / 2, Dict(g => u[j] for (g, j) in levels)
end

"""
    _posterior_system(H, b, x, free, tracks, key; gauge = nothing)

The MAP problem of one observable's slot values `x` with the χ² to second order
(data curvature `H` and gradient `b` at `x`, over every slot) and the exact prior
energy of each of `tracks`, over the `free` slots (the other slots keep their
values), the tracks' latent states and their levels: the data's quadratic and
each track's state chains (`_track_chains!`), the priors added by
`_assembled`. `gauge`, when given, holds constraint rows `(slot, coefficient)` and
right-hand sides `d`; each row is solved for its free slot of largest
coefficient, `x = T·u + t0` over the unknowns, so that `u`, the unknowns left,
are unconstrained and the system positive definite. `key` gives each slot's
frequency, by which the unknowns are ordered for factoring: the system is then
banded, coupling stations only at one frequency and frequencies only along a
track.
"""
function _posterior_system(H, b, x, free, tracks, key; gauge = nothing)
    idx = zeros(Int, length(x))
    nf = 0
    for n in eachindex(free)
        free[n] && (idx[n] = (nf += 1))
    end
    data = _Quadratic(key[free])
    Hf = H[free, free]
    Is, Js, Vs = findnz(Hf)
    append!(data.Is, Is)
    append!(data.Js, Js)
    append!(data.Vs, Vs)
    xf = Float64.(x[free])
    data.h .= b[free] .+ Hf * xf
    data.c = dot(xf, Hf * xf) / 2 + dot(b[free], xf)
    slot(n) = free[n] ? (idx[n], 0.0) : (0, Float64(x[n]))
    built = [_track_chains!(data, t, slot) for t in tracks]
    return (;
        data, chains = first.(built), levels = last.(built), x = Float64.(x), free, xf,
        gauge = _gauge_map(gauge, x, free, idx, length(data.h)),
    )
end

# Each gauge row solved for its free slot of largest coefficient, over the `n`
# unknowns of a system whose first are the free slots (numbered `idx`):
# `x = T·u + t0`, `keep` the unknowns `u` stands for, and `Tt = Tᵀ`. The rows hold
# disjoint slots, one per connected component.
function _gauge_map(gauge, x, free, idx, n)
    isnothing(gauge) && return nothing
    row_of = zeros(Int, n)
    for (r, row) in pairs(gauge.rows)
        cands = [(m, c) for (m, c) in row if free[m]]
        isempty(cands) && continue
        row_of[idx[first(cands[argmax([abs(c) for (_, c) in cands])])]] = r
    end
    keep = findall(iszero, row_of)
    col = zeros(Int, n)
    col[keep] = eachindex(keep)
    Is, Js, Vs = collect(keep), col[keep], ones(length(keep))
    t0 = zeros(n)
    for e in findall(!iszero, row_of)
        row = gauge.rows[row_of[e]]
        ce = only(c for (m, c) in row if free[m] && idx[m] == e)
        t0[e] = (gauge.d[row_of[e]] - sum((c * x[m] for (m, c) in row if !free[m]); init = 0.0)) / ce
        for (m, c) in row
            (free[m] && idx[m] != e) || continue
            push!(Is, e)
            push!(Js, col[idx[m]])
            push!(Vs, -c / ce)
        end
    end
    T = sparse(Is, Js, Vs, n, length(keep))
    return (; T, t0, keep, Tt = sparse(T'))
end

# The system's quadratic with each track's energy under `priors` added.
function _assembled(sys, priors)
    Q = copy(sys.data)
    for (chains, prior) in zip(sys.chains, priors)
        _add_prior_terms!(Q, chains, prior)
    end
    return Q
end

# `Q` plus `λ/2·‖x′ − x‖²` over the free slots, over the unknowns the gauge
# leaves: `(K, h, c, key)` of `½uᵀKu − hᵀu + c`.
function _reduced(sys, Q, λ)
    nf = length(sys.xf)
    K = _matrix(Q) + spdiagm([fill(Float64(λ), nf); zeros(length(Q.h) - nf)])
    h = copy(Q.h)
    h[1:nf] .+= λ .* sys.xf
    c = Q.c + λ * dot(sys.xf, sys.xf) / 2
    isnothing(sys.gauge) && return K, h, c, Q.key
    (; T, t0, keep) = sys.gauge
    return T' * K * T, T' * (h - K * t0), c + dot(t0, K * t0) / 2 - dot(h, t0), Q.key[keep]
end

# The unknowns `u` in the system's own.
_unreduced(sys, u) = isnothing(sys.gauge) ? u : sys.gauge.T * u + sys.gauge.t0

_factor(K, key; check = true) = cholesky(Symmetric(K); perm = sortperm(key), check)

"""
    _map_step(sys, priors; λ = 0) -> (x′, levels, model)

One Levenberg–Marquardt step of the MAP objective of the system `sys`
(`_posterior_system`) with each track's resolved prior from `priors`: the
minimum of its quadratic plus `λ/2·‖x′ − x‖²` over the free slots. Returns the
new slot values, each track's level values by group, and the undamped
quadratic's value there, which is the prior energy alone at `x′ = x`.
"""
function _map_step(sys, priors; λ = 0)
    Q = _assembled(sys, priors)
    K, h, _, key = _reduced(sys, Q, λ)
    z = _unreduced(sys, _factor(K, key) \ h)
    x′ = copy(sys.x)
    x′[sys.free] .= z[eachindex(sys.xf)]
    return x′, [Dict(g => z[j] for (g, j) in lv) for lv in sys.levels], dot(z, _matrix(Q) * z) / 2 - dot(Q.h, z) + Q.c
end

# ── Hyperparameters and branches ─────────────────────────────────────────────

"""
    _selected_inverse(F) -> (i, j) -> K⁻¹[i, j]

The entries of `K⁻¹` on the sparsity pattern of its Cholesky factor `F`, by the
recursion of Takahashi, Fagan & Chen (1973) from the last column back. The
pattern holds every pair of unknowns that share a term of the energy, so these
are the posterior covariances the expected energy needs. Looking up a pair
outside it throws.
"""
function _selected_inverse(F)
    L = sparse(F.L)
    rv, Lv = rowvals(L), nonzeros(L)
    n = size(L, 1)
    Zv = zeros(length(Lv))
    acc = zeros(n)
    pos = zeros(Int, n)
    mark = zeros(Int, n)
    for j in reverse(axes(L, 2))
        rng = nzrange(L, j)
        below = rng[2:end]
        for kk in below
            mark[rv[kk]] = j
            pos[rv[kk]] = kk
            acc[rv[kk]] = 0.0
        end
        # `Σ_k L_kj·Z_ik` for each row `i` below the diagonal, from the stored
        # lower triangle of the columns already done.
        for kk in below
            k = rv[kk]
            for zz in nzrange(L, k)
                i = rv[zz]
                mark[i] == j || continue
                acc[i] += Lv[kk] * Zv[zz]
                i != k && (acc[k] += Lv[pos[i]] * Zv[zz])
            end
        end
        d = Lv[first(rng)]
        s = 0.0
        for kk in below
            Zv[kk] = -acc[rv[kk]] / d
            s += Lv[kk] * Zv[kk]
        end
        Zv[first(rng)] = (1 / d - s) / d
    end
    Z = SparseMatrixCSC(n, n, copy(L.colptr), copy(rv), Zv)
    ip = invperm(F.p)
    return function (a, b)
        col, row = minmax(ip[a], ip[b])
        rng = nzrange(Z, col)
        k = searchsortedfirst(view(rowvals(Z), rng), row)
        k <= length(rng) && rowvals(Z)[rng[k]] == row ||
            error("posterior covariance ($a, $b) is not on the factor's sparsity pattern")
        return nonzeros(Z)[rng[k]]
    end
end

# The covariance of a system's own unknowns, from the covariance `cov` of the
# unknowns `u` its gauge leaves.
_covariance(cov, ::Nothing) = cov
function _covariance(cov, gauge)
    Tt = gauge.Tt
    rv, v = rowvals(Tt), nonzeros(Tt)
    return (a, b) -> sum((v[p] * v[q] * cov(rv[p], rv[q]) for p in nzrange(Tt, a) for q in nzrange(Tt, b)); init = 0.0)
end

# The hyperparameters `prior` gives as densities, by name; `prior` with those at
# the values `vals`; and their hyperprior's log density over their logs.
_hyper_names(::Nothing) = ()
_hyper_names(p::RandomWalkPrior) = is_fixed_hyper(p.σ) ? () : (:σ,)
_hyper_names(p::OUPrior) = Tuple(s for s in (:scale, :σ) if !is_fixed_hyper(getfield(p, s)))
_with_hypers(p::RandomWalkPrior, vals) = RandomWalkPrior(; p.order, σ = get(vals, :σ, p.σ), p.init)
_with_hypers(p::OUPrior, vals) = OUPrior(; scale = get(vals, :scale, p.scale), σ = get(vals, :σ, p.σ))
_log_hyperprior(prior, vals) = sum(logdensityof(getfield(prior, k), v) + log(v) for (k, v) in pairs(vals))

# The inverse of a symmetric curvature, its eigenvalues taken in magnitude.
function _inverse_curvature(C)
    E = eigen(Symmetric(C))
    return E.vectors * Diagonal(inv.(abs.(E.values))) * E.vectors'
end

"""
    _fit_hypers!(priors, curvature, sys, tracks, λ; tolerance, max_iterations) -> Bool

Type-II MAP of the hyperparameters `tracks`' priors give as densities, written
into `priors` (each track's resolved prior, the current values on entry): the
maximum over their logs of the hyperprior density times the Laplace
approximation of the marginal likelihood of the system `sys`
(`_posterior_system`, damped by `λ`). That quadratic is Gaussian in its
unknowns, so the marginal is exact for it:

    log p(data | θ) = −min_u f_θ(u) − ½·log det ∇²f_θ + Σ_k log N-normalization of each prior term,

the levels and flat starts integrated under their flat priors. Its gradient in
a track's hyperparameters is that of the track's prior log density in
expectation over the posterior of its states (Fisher's identity), whose moments
need only the covariances `_selected_inverse` gives.

The search is quasi-Newton with one secant (BFGS) block per track, started from
the curvature of that expected log density and kept in `curvature` from call to
call, each step backtracked until the log posterior is no lower. It stops once
the Newton step is within `tolerance` of its standard error, `gᵀH⁻¹g ≤
tolerance²`, by every track's curvature measured along a step of its own, and
returns whether it did within `max_iterations` steps.
"""
function _fit_hypers!(priors, curvature, sys, tracks, λ; tolerance, max_iterations)
    est = [i for i in eachindex(tracks) if !isempty(_hyper_names(tracks[i].prior))]
    isempty(est) && return true
    names = [_hyper_names(tracks[i].prior) for i in est]
    vals(k, lθ) = NamedTuple{names[k]}(Tuple(exp.(lθ)))
    θ = [log.([getfield(priors[i], s) for s in names[k]]) for (k, i) in pairs(est)]
    function evaluate(θ)
        trial = copy(priors)
        for (k, i) in pairs(est)
            trial[i] = _with_hypers(tracks[i].prior, vals(k, θ[k]))
        end
        K, h, c, key = _reduced(sys, _assembled(sys, trial), λ)
        F = _factor(K, key; check = false)
        # Hyperparameters so extreme that the posterior's curvature is lost to
        # rounding are no candidate.
        issuccess(F) || return (; lp = -Inf, F, u = h, trial)
        u = F \ h
        lp = dot(h, u) / 2 - c - logdet(F) / 2
        lp += sum(_expected_log_prior(p, ch, nothing, Float64) for (p, ch) in zip(trial, sys.chains); init = 0.0)
        lp += sum(_log_hyperprior(tracks[i].prior, vals(k, θ[k])) for (k, i) in pairs(est))
        return (; lp, F, u, trial)
    end
    # Each track's expected log prior density and hyperprior, as a function of
    # its log hyperparameters.
    function expected(at)
        cov = _covariance(_selected_inverse(at.F), sys.gauge)
        z = _unreduced(sys, at.u)
        moms = [_chain_moments(sys.chains[i], z, cov) for i in est]
        return [
            lθ -> _expected_log_prior(_with_hypers(tracks[i].prior, vals(k, lθ)), sys.chains[i], moms[k], eltype(lθ)) +
                _log_hyperprior(tracks[i].prior, vals(k, lθ))
                for (k, i) in pairs(est)
        ]
    end
    at = evaluate(θ)
    isfinite(at.lp) || throw(PosDefException(0))
    fs = expected(at)
    for (k, i) in pairs(est)
        isnothing(curvature[i]) &&
            (curvature[i] = (; inverse = _inverse_curvature(-ForwardDiff.hessian(fs[k], θ[k])), measured = false))
    end
    g = [ForwardDiff.gradient(fs[k], θ[k]) for k in eachindex(est)]
    converged = false
    for _ in 1:max_iterations
        Hinv = [curvature[i].inverse for i in est]
        if sum(k -> dot(g[k], Hinv[k] * g[k]), eachindex(est)) <= tolerance^2 && all(i -> curvature[i].measured, est)
            converged = true
            break
        end
        d = [Hinv[k] * g[k] for k in eachindex(est)]
        α = 1.0
        next = nothing
        while true
            θ1 = [θ[k] .+ α .* d[k] for k in eachindex(est)]
            all(k -> _within_rounding(θ1[k], θ[k]), eachindex(est)) && break
            trial = evaluate(θ1)
            # The log posterior is known to its rounding.
            trial.lp >= at.lp - 16 * eps(abs(at.lp)) && (next = (θ1, trial); break)
            α /= 2
        end
        # No step changes the hyperparameters: they are at the maximum to rounding.
        isnothing(next) && (converged = true; break)
        θ1, at = next
        fs = expected(at)
        g1 = [ForwardDiff.gradient(fs[k], θ1[k]) for k in eachindex(est)]
        for (k, i) in pairs(est)
            s, y = θ1[k] - θ[k], g[k] - g1[k]
            sy = dot(s, y)
            sy > 0 || continue
            V = I - y * s' / sy
            curvature[i] = (; inverse = V' * curvature[i].inverse * V + s * s' / sy, measured = true)
        end
        θ, g = θ1, g1
    end
    priors[est] .= at.trial[est]
    return converged
end

# Each slot's own estimate of one observable given every other gain and `S`, from
# its relative gain estimate `1 + c/W` (`_linearize`), and its weight; `NaN` and
# zero for a slot without data. With the phase held at zero the least-squares
# log-amplitude is that of `Re(1 + c/W)`.
function _slot_estimates(x, lin, observable; phase_held = false)
    y = fill(NaN, length(x))
    wy = zeros(length(x))
    for n in eachindex(x, lin.W)
        lin.W[n] > 0 || continue
        ρ = 1 + lin.c[n] / lin.W[n]
        v = observable === :phase ? angle(ρ) : phase_held ? (real(ρ) > 0 ? log(real(ρ)) : NaN) : log(abs(ρ))
        isfinite(v) || continue
        y[n] = x[n] + v
        wy[n] = lin.W[n] * abs2(ρ)
    end
    return y, wy
end

# Track `t`'s prior with its hyperparameters estimated from the slot estimates
# `y` (weights `wy`) as independent measurements of the track alone: type-II MAP
# over its pieces, any constant the source term absorbs integrated out, and for
# a phase over its pieces' 2π branches. The joint solve starts from these.
function _seed_prior(t, y, wy; phase::Bool)
    ys = [y[t.slots[p]] for p in t.pieces]
    ws = [wy[t.slots[p]] for p in t.pieces]
    xs = [float.(t.x[p]) for p in t.pieces]
    return _estimate_hypers(t.prior, ys, ws, xs; level = _level_groups(t, t.prior), branches = phase)
end

# Put phase track `t` on the branches `_branch_filter` finds most probable for its
# slot estimates `y` under the resolved `prior`: each free slot of `φ` moves by the
# multiple of 2π its estimate moves, which leaves its gain unchanged. A piece's
# branches are first moved together by the multiple of 2π nearest its estimates,
# and the pieces sharing a level group onto one branch. Writes each piece's
# `_slip_probability` into `slip` and returns the per-slot shifts (radians).
function _choose_branches!(φ, t, prior, y, wy, free; slip = nothing)
    groups = _level_groups(t, prior)
    ys = [y[t.slots[p]] for p in t.pieces]
    ws = [wy[t.slots[p]] for p in t.pieces]
    lifts = map(eachindex(t.pieces)) do j
        p = t.pieces[j]
        model, θ, r, u = _filter_inputs(prior, ys[j], ws[j], float.(t.x[p]))
        any(isfinite, θ) || return θ
        bf = _branch_filter(model, θ, r, u; level = !isnothing(groups))
        isnothing(slip) || (slip[j] = _slip_probability(model, bf.histories, r, u; level = !isnothing(groups)))
        lift = first(bf.histories).lift
        return lift .+ 2π * round(_weighted_mean_finite(θ .- lift, ws[j]) / 2π)
    end
    isnothing(groups) || _align_branches!(lifts, ws, groups)
    shift = zeros(length(φ))
    for (j, p) in pairs(t.pieces), (i, k) in pairs(p)
        n = t.slots[k]
        (free[n] && isfinite(lifts[j][i])) || continue
        shift[n] = 2π * round((lifts[j][i] - ys[j][i]) / 2π)
        φ[n] += shift[n]
    end
    return shift
end

# Move each gauge row's right-hand side by the branch shifts of its slots, so
# the constraint still holds on the gains after a change of branch.
function _shift_gauge!(gauge, shift)
    for (j, row) in pairs(gauge.rows)
        gauge.d[j] += sum((c * shift[n] for (n, c) in row); init = 0.0)
    end
    return gauge
end
