# ── Chain priors as sparse quadratics ────────────────────────────────────────
#
# A state-space prior's energy over a chain is a sum of state-space terms: each
# piece (a spectral window, or a scan) is a chain of states along its axis,
# `s_k = A_k·s_{k−1} + η_k`, `η_k ~ N(0, Q_k)`, from the prior's start, whose
# first entry is the chain's parameter. Written over the states themselves the
# energy is a sum of small quadratic terms, so a chain's prior and its latent
# states enter one sparse system with the data.
#
# The per-scan source term absorbs a constant added to the station gains of one
# station and feed over a scan, so the data cannot measure a chain's constant
# over each of its `constants` groups (`_parameter_chains`): the prior sets it.

# A linear form `Σ a·u[j] + b`: its terms `j => a` and its constant `b`.
const _Form = Tuple{Vector{Pair{Int, Float64}}, Float64}

# A new unknown, ordered for factoring by the frequency `key`.
_unknown!(keys, key) = (push!(keys, key); length(keys))

# The index among `K`'s stored values of `K[i, j]`, which is on its pattern.
function _position(K, i, j)
    rng = nzrange(K, j)
    return rng[searchsortedfirst(view(rowvals(K), rng), i)]
end

# A prior term `½sᵀWs` over the state rows `s` of the forms `fs`: the entries
# `(row, unknown, coefficient)` of those rows, each row's constant, and the
# index among K's stored values of each pair of entries (`_locate!`).
function _prior_term(fs)
    entries = [(r, j, a) for (r, f) in pairs(fs) for (j, a) in f[1]]
    return (; entries, base = [f[2] for f in fs], pos = zeros(Int, length(entries), length(entries)))
end

function _locate!(term, K)
    for (a, e) in pairs(term.entries), (b, g) in pairs(term.entries)
        term.pos[a, b] = _position(K, e[2], g[2])
    end
    return term
end

# The prior terms of chain `ch` (`_track_chains!`): its start and one per
# transition, over the stacked states `(s_k, s_{k−1})`.
function _chain_terms(ch)
    return (;
        start = _prior_term(ch.states[1]),
        steps = [_prior_term([ch.states[k]; ch.states[k - 1]]) for k in 2:length(ch.states)],
        ch.u, ch.h, ch.dim,
    )
end

# Calls `f` on every prior term of `terms` (per track, per chain).
function _foreach_term(f, terms)
    for ts in terms, t in ts
        f(t.start)
        foreach(f, t.steps)
    end
    return nothing
end

# The quadratic `½uᵀKu − hᵀu + c` over `n` unknowns: the data terms `(Is, Js,
# Vs)` with gradient `h` and constant `c`, plus the prior terms of `chains` (per
# track, per chain), the diagonal of the first `nf` unknowns held for damping.
# The unknowns of `ridge` add `w·u²/2` each. Its pattern is fixed here;
# `_assemble!` writes its values for given priors.
function _quadratic_layout(chains, n; data = (Int[], Int[], Float64[]), h = zeros(n), c = 0.0, nf = 0, ridge = Pair{Int, Float64}[])
    terms = [[_chain_terms(ch) for ch in cs] for cs in chains]
    Is, Js = [data[1]; 1:nf; first.(ridge)], [data[2]; 1:nf; first.(ridge)]
    _foreach_term(terms) do term
        for e in term.entries, g in term.entries
            push!(Is, e[2])
            push!(Js, g[2])
        end
    end
    K = sparse(Is, Js, ones(length(Is)), n, n)
    fill!(nonzeros(K), 0.0)
    _foreach_term(term -> _locate!(term, K), terms)
    base = zeros(nnz(K))
    for (i, j, v) in zip(data...)
        base[_position(K, i, j)] += v
    end
    return (;
        K, base, h0 = h, c0 = Ref(c), h = similar(h), diag = [_position(K, j, j) for j in 1:nf], terms,
        ridge, ridge_at = [_position(K, j, j) for (j, _) in ridge], prior = zeros(nnz(K)), hprior = zeros(n),
        cprior = Ref(0.0), assembled = Ref{Any}(nothing),
    )
end

# Writes the quadratic under the resolved `priors` (one per track) into
# `lay.K`'s stored values and `lay.h`, and returns its constant. The priors'
# part of `K`'s values is kept apart in `lay.prior`, so it is read without the
# rounding of `K − base` where the data's curvature dwarfs it; it, the priors'
# part of `h` and their constant are rewritten only when `priors` differ from
# those they were last written for.
function _assemble!(lay, priors)
    Kv = nonzeros(lay.K)
    T = eltype(Kv)
    if !isequal(lay.assembled[], priors)
        Pv = fill!(lay.prior, zero(T))
        hp = fill!(lay.hprior, zero(T))
        for (p, (_, w)) in zip(lay.ridge_at, lay.ridge)
            Pv[p] += w
        end
        c = zero(T)
        for (ts, prior) in zip(lay.terms, priors)
            isempty(ts) && continue
            c += _with_piece_models(prior, first(ts).dim, T) do model
                # A loop: inference widens a sum over a generator here to Any when _fit_hypers! calls it.
                ct = zero(T)
                for t in ts
                    ct += _add_chain!(Pv, hp, t, model(t.h))
                end
                ct
            end
        end
        lay.cprior[] = c
        lay.assembled[] = copy(priors)
    end
    Kv .= lay.base .+ lay.prior
    lay.h .= lay.h0 .+ lay.hprior
    return lay.c0[] + lay.cprior[]
end

function _add_chain!(Kv, h, t, model)
    T = eltype(Kv)
    D = statedim(model)
    c = zero(T)
    start = initial(model, T)
    if !isnothing(start)
        μ0, P0 = start
        c += _add_term!(Kv, h, t.start, inv(P0), μ0)
    end
    step = nothing
    for (k, term) in pairs(t.steps)
        step = _transition(model, t.u[k + 1] - t.u[k], T, step)
        _, A, Q = step
        G = hcat(SMatrix{D, D, T}(I), -A)
        c += _add_term!(Kv, h, term, G' * (Q \ G), zero(SVector{2D, T}))
    end
    return c
end

# The prior energy of chain `t` (`_chain_terms`) under `model` at the unknowns
# `u`: half the squared whitened residual of its start and of each transition's
# innovation `s_k − A·s_{k−1}`. Each term is small where `Q` is, so no large
# terms cancel. A covariance singular to rounding gives no density: `Inf`.
function _chain_energy(t, model, u)
    T = eltype(u)
    D = statedim(model)
    E = zero(T)
    start = initial(model, T)
    if !isnothing(start)
        μ0, P0 = start
        E += _whitened_energy(_term_rows(t.start, u, Val(D)) - μ0, P0)
    end
    step = nothing
    for (k, term) in pairs(t.steps)
        step = _transition(model, t.u[k + 1] - t.u[k], T, step)
        _, A, Q = step
        s = _term_rows(term, u, Val(2D))
        E += _whitened_energy(s[SOneTo(D)] - A * s[SVector{D}((D + 1):(2D))], Q)
    end
    return E
end

# The state rows `s` of `term` at the unknowns `u`.
function _term_rows(term, u, ::Val{R}) where {R}
    s = SVector{R}(term.base)
    for (p, j, α) in term.entries
        s = Base.setindex(s, s[p] + α * u[j], p)
    end
    return s
end

# The log normalizer of `_chain_energy` for chain `t` under `model`: half the
# log determinant of `2π` times the covariance of its start and of each
# innovation, so that `exp(−E)` over the chain's states is a density. A
# covariance singular to rounding gives no density: `Inf`.
function _chain_log_normalizer(t, model)
    T = Float64
    out = zero(T)
    start = initial(model, T)
    isnothing(start) || (out += _log_det_2π(last(start)))
    step = nothing
    for k in eachindex(t.steps)
        step = _transition(model, t.u[k + 1] - t.u[k], T, step)
        out += _log_det_2π(last(step))
    end
    return out / 2
end

function _log_det_2π(Σ)
    C = cholesky(Symmetric(Σ); check = false)
    return issuccess(C) ? size(Σ, 1) * log(2π) + logdet(C) : Inf
end

# The log normalizer of the chains of `lay` under the resolved `priors` (one
# per track), summed over the chains (`_chain_log_normalizer`).
function _prior_log_normalizer(lay, priors)
    N = 0.0
    for (ts, prior) in zip(lay.terms, priors)
        isempty(ts) && continue
        N += _with_piece_models(prior, first(ts).dim, Float64) do model
            # A loop, as in `_assemble!`.
            out = 0.0
            for t in ts
                out += _chain_log_normalizer(t, model(t.h))
            end
            out
        end
    end
    return N
end

function _whitened_energy(e, Σ)
    C = cholesky(Symmetric(Σ); check = false)
    issuccess(C) || return oftype(float(eltype(e)), Inf)
    return sum(abs2, C.L \ e) / 2
end

# Adds `½sᵀWs` over `term`'s state rows `s`, less `μ`, to K's stored values
# `Kv` and to `h`, and returns its constant.
function _add_term!(Kv, h, term, W::SMatrix{R, R}, μ) where {R}
    W = (W + W') / 2
    r = SVector{R}(term.base) - μ
    Wr = W * r
    for (a, (p, j, α)) in pairs(term.entries)
        h[j] -= α * Wr[p]
        for (b, (q, _, β)) in pairs(term.entries)
            Kv[term.pos[a, b]] += α * β * W[p, q]
        end
    end
    return dot(r, Wr) / 2
end

# The coordinates a prior's state-space model runs on over one piece's
# frequencies `x`, and their unit `h` in `x`.
function _piece_coordinates(::Union{RandomWalkPrior, CARMAPrior}, x)
    _, u, h = _random_walk_problem(x, ones(length(x)), x)
    return u, h
end
_piece_coordinates(::OUPrior, x) = float.(x), 1.0

# The smallest median sample spacing and the largest span of the pieces with
# coordinates `us` of units `hs`, over those with two samples or more.
function _piece_extent(us, hs)
    Δ = minimum(median(abs.(diff(u))) * h for (u, h) in zip(us, hs) if length(u) >= 2)
    W = maximum(abs(last(u) - first(u)) * h for (u, h) in zip(us, hs) if length(u) >= 2)
    return Δ, W
end

# `f(model)` with `model(h)` the resolved `prior`'s state-space model of state
# dimension `dim` (a chain's, `Val(_statedim(prior))`) on coordinates of unit
# `h`, in element type `T`.
_with_piece_models(f, prior::RandomWalkPrior, dim::Val, ::Type{T}) where {T} = f(h -> _random_walk_model(dim, prior.σ, prior.init, h, T))
_with_piece_models(f, prior::OUPrior, ::Val, ::Type{T}) where {T} = f(_ -> OUModel(T(prior.scale), T(prior.σ)^2))
_with_piece_models(f, prior::CARMAPrior, dim::Val, ::Type{T}) where {T} = f(h -> _carma_model(dim, prior, h, T))
_statedim(prior::RandomWalkPrior) = prior.order
_statedim(::OUPrior) = 1
_statedim(prior::CARMAPrior) = Calibration._root_count(prior.ar)

# The resolved `prior`'s model on coordinates of unit `h`, where each root is `h`
# times its root in the axis coordinate.
function _carma_model(::Val{P}, prior::CARMAPrior, h, ::Type{T}) where {P, T}
    roots(f::RealRoot) = (Complex{T}(-h / f.scale),)
    roots(f::ComplexRoots) = (r = complex(-h / T(f.scale), 2π * h / T(f.period)); (r, conj(r)))
    ar = Complex{T}[r for f in prior.ar for r in roots(f)]
    ma = Complex{T}[r for f in prior.ma for r in roots(f)]
    return CARMAModel{P}(ar, ma, T(prior.σ)^2)
end

# The state chain of each piece of track `t`, over unknowns whose coordinates
# `keys` holds: per sample its state, as linear forms whose first is the
# sample's unknown `t.gains[k]` and the rest new latent unknowns, with the
# coordinates its prior runs on, its samples' unknowns and its state dimension
# as a type (`Val`).
_track_chains!(keys, t) = [_piece_chain!(keys, t, p) for p in t.pieces]

function _piece_chain!(keys, t, p)
    states = map(p) do k
        _Form[([t.gains[k] => 1.0], 0.0); [([_unknown!(keys, t.x[k]) => 1.0], 0.0) for _ in 2:_statedim(t.prior)]]
    end
    u, h = _piece_coordinates(t.prior, t.x[p])
    return (; u, h, states, gains = t.gains[p], dim = Val(_statedim(t.prior)))
end

# The log density of `chains` under the resolved `prior`, in expectation over the
# states' posterior, from their moments summed where the prior's terms are the
# same (`_moment_sums`); in element type `T`.
function _expected_log_prior(prior, sums, ::Type{T}) where {T}
    return _with_piece_models(prior, sums.dim, T) do model
        # A loop: inference widens a sum over a generator here to Any.
        out = zero(T)
        for g in sums.groups
            out += _expected_group_log_prior(model(g.h), g, T)
        end
        out
    end
end

function _expected_group_log_prior(model, g, ::Type{T}) where {T}
    D = statedim(model)
    out = zero(T)
    start = initial(model, T)
    if !isnothing(start)
        μ0, P0 = start
        n, m, M = g.start
        out -= _gaussian_energy(n, P0, M - μ0 * m' - m * μ0' + n * μ0 * μ0')
    end
    for (Δ, n, V) in g.steps
        A, Q = transition(model, Δ, T)
        G = hcat(SMatrix{D, D, T}(I), -A)
        out -= _gaussian_energy(n, Q, G * V * G')
    end
    return out
end

# `n` Gaussian samples' negative log density under covariance `Σ`, given their
# summed second moment `S` about the mean: `n·logdet(2πΣ)/2 + tr(Σ⁻¹S)/2`. A
# covariance that is singular to rounding, as a CARMA model's is where an MA root
# cancels an AR root, gives the samples no density: `Inf`.
function _gaussian_energy(n, Σ, S)
    T = float(eltype(Σ))
    C = cholesky(Symmetric(Σ); check = false)
    issuccess(C) || return T(Inf)
    return (n * (size(Σ, 1) * log(2T(π)) + logdet(C)) + tr(C \ S)) / 2
end

# The moments `mom` of `chains` (`_chain_moments`) summed where the prior's terms
# are the same, per coordinate unit `h`: the start's count, mean and second
# moment, and per step `Δ` (`_same_step`) the transitions' count and second
# moment. The expected log prior is linear in them, so a hyperparameter search
# evaluates one term per distinct step rather than one per sample.
_moment_sums(chains, mom) = _moment_sums(chains, mom, first(chains).dim)

function _moment_sums(chains, mom, dim::Val{D}) where {D}
    groups = map(unique(ch.h for ch in chains)) do h
        start = (0, zero(SVector{D, Float64}), zero(SMatrix{D, D, Float64}))
        steps = Tuple{Float64, Int, SMatrix{2D, 2D, Float64, 4 * D * D}}[]
        for (c, ch) in pairs(chains)
            ch.h == h || continue
            m, M = mom[c].start
            start = (start[1] + 1, start[2] + m, start[3] + M)
            for i in 2:length(ch.u)
                Δ = ch.u[i] - ch.u[i - 1]
                k = findfirst(st -> _same_step(st[1], Δ), steps)
                V = mom[c].pairs[i - 1]
                isnothing(k) ? push!(steps, (Δ, 1, V)) : (steps[k] = (steps[k][1], steps[k][2] + 1, steps[k][3] + V))
            end
        end
        (; h, start, steps)
    end
    return (; dim, groups)
end

# The posterior moments of each chain's states of dimension `D`, from the mean
# `z` and covariance `cov(i, j)` of the unknowns: the first state's mean and
# second moment, and the second moment of each consecutive pair `(s_i, s_{i−1})`.
function _chain_moments(chains, z, cov::C, ::Val{D}) where {C, D}
    return map(chains) do ch
        s1 = ch.states[1]
        (;
            start = (SVector{D}(ntuple(r -> _form_mean(s1[r], z), Val(D))), _second_moment(r -> s1[r], z, cov, Val(D))),
            pairs = [
                _second_moment(r -> r <= D ? ch.states[i][r] : ch.states[i - 1][r - D], z, cov, Val(2D))
                    for i in 2:length(ch.states)
            ],
        )
    end
end

function _form_mean(f, z)
    m = f[2]
    for (j, a) in f[1]
        m += a * z[j]
    end
    return m
end

function _form_covariance(f, g, cov)
    s = zero(f[2])
    for (i, a) in f[1], (j, b) in g[1]
        s += a * b * cov(i, j)
    end
    return s
end

# The second moment of the `R` forms `form(1), …, form(R)`.
function _second_moment(form::F, z, cov::C, ::Val{R}) where {F, C, R}
    return SMatrix{R, R}(
        ntuple(Val(R * R)) do l
            f, g = form((l - 1) % R + 1), form((l - 1) ÷ R + 1)
            _form_covariance(f, g, cov) + _form_mean(f, z) * _form_mean(g, z)
        end
    )
end

# The gradient of `_chain_energy` over the unknowns, added into `g`: each term's
# innovation, through `Q⁻¹`, carried back onto its rows. It matches the energy
# as evaluated to its rounding, unlike `K·u − h`, whose `Gᵀ·Q⁻¹·G` carries a
# rounding error of the order of the stiffest term's curvature.
function _chain_gradient!(g, t, model, u)
    T = eltype(u)
    D = statedim(model)
    start = initial(model, T)
    if !isnothing(start)
        μ0, P0 = start
        _add_rows!(g, t.start, cholesky(Symmetric(P0)) \ (_term_rows(t.start, u, Val(D)) - μ0))
    end
    step = nothing
    for (k, term) in pairs(t.steps)
        step = _transition(model, t.u[k + 1] - t.u[k], T, step)
        _, A, Q = step
        s = _term_rows(term, u, Val(2D))
        w = cholesky(Symmetric(Q)) \ (s[SOneTo(D)] - A * s[SVector{D}((D + 1):(2D))])
        _add_rows!(g, term, vcat(w, -A' * w))
    end
    return g
end

# `aᵀ` times the gradient of `_chain_energy` of chain `t` under `model`, in
# element type `T`, over the unknowns at `u` (`_chain_gradient!`, without its
# vector).
function _chain_gradient_dot(t, model, u, a, ::Type{T}) where {T}
    D = statedim(model)
    out = zero(T)
    start = initial(model, T)
    if !isnothing(start)
        μ0, P0 = start
        out += _rows_dot(t.start, a, cholesky(Symmetric(P0)) \ (_term_rows(t.start, u, Val(D)) - μ0))
    end
    step = nothing
    for (k, term) in pairs(t.steps)
        step = _transition(model, t.u[k + 1] - t.u[k], T, step)
        _, A, Q = step
        s = _term_rows(term, u, Val(2D))
        w = cholesky(Symmetric(Q)) \ (s[SOneTo(D)] - A * s[SVector{D}((D + 1):(2D))])
        out += _rows_dot(term, a, vcat(w, -A' * w))
    end
    return out
end

# `aᵀ` times the gradient `r` over `term`'s state rows carried onto the unknowns.
_rows_dot(term, a, r) = sum(((p, j, α),) -> α * a[j] * r[p], term.entries; init = zero(eltype(r)))

# Adds the gradient `r` over `term`'s state rows onto the unknowns in `g`.
function _add_rows!(g, term, r)
    for (p, j, α) in term.entries
        g[j] += α * r[p]
    end
    return g
end

# The gradient of `_prior_energy` at the unknowns `u`, written into `g`.
function _prior_gradient!(g, lay, priors, u)
    fill!(g, 0.0)
    for (j, w) in lay.ridge
        g[j] += w * u[j]
    end
    for (ts, prior) in zip(lay.terms, priors)
        isempty(ts) && continue
        _with_piece_models(prior, first(ts).dim, Float64) do model
            foreach(t -> _chain_gradient!(g, t, model(t.h), u), ts)
        end
    end
    return g
end

# The prior energy of the chains of `lay` under the resolved `priors` (one per
# track) at the unknowns `u`, summed over the chains (`_chain_energy`), and of
# its ridge.
function _prior_energy(lay, priors, u)
    E = sum(((j, w),) -> w * u[j]^2 / 2, lay.ridge; init = 0.0)
    for (ts, prior) in zip(lay.terms, priors)
        isempty(ts) && continue
        E += _with_piece_models(prior, first(ts).dim, Float64) do model
            # A loop, as in `_assemble!`.
            out = 0.0
            for t in ts
                out += _chain_energy(t, model(t.h), u)
            end
            out
        end
    end
    return E
end

"""
    _posterior_system(H, b, J, θ, tracks, key; ridge = [])

The MAP problem of one observable with the χ² to second order in its
station-gain values `x = J·θ` (data curvature `H` and gradient `b` there) and
the exact prior energy of each of `tracks`, over the parameters `θ` and the
tracks' latent states: the data's quadratic `JᵀHJ`, each track's state chains
(`_track_chains!`) and `w·θ_j²/2` for each pair `j => w` of `ridge`, the
priors added by `_assemble!`. `key` gives each parameter its coordinate along
its chain, by which the unknowns are ordered for factoring: the system is then
banded, coupling stations only at one frequency and frequencies only along a
chain. The data's quadratic keeps every product of stored entries of `Jᵀ`, `H`
and `J`, so the pattern does not depend on their values, and `_set_data!`
rewrites it for new values on `H`'s pattern. The system also holds the
coarse space of its tracks' constants (`_coarse_basis`) and buffers for the
steps on it.
"""
function _posterior_system(H, b, J, θ, tracks, key; ridge = Pair{Int, Float64}[])
    nθ = size(J, 2)
    keys = Float64.(key)
    chains = [_track_chains!(keys, t) for t in tracks]
    n = length(keys)
    Is, Js, qs, coefs = _product_terms(J, H)
    layout = _quadratic_layout(chains, n; data = (Is, Js, zeros(length(Is))), nf = nθ, ridge = collect(Pair{Int, Float64}, ridge))
    data_map = sparse([_position(layout.K, i, j) for (i, j) in zip(Is, Js)], qs, coefs, nnz(layout.K), nnz(H))
    Z = _coarse_basis(chains, tracks, n)
    m = size(Z, 2)
    sys = (;
        layout, chains, J, θ = Float64.(θ), bθ = zeros(nθ), data_map, damped = copy(layout.K), hλ = zeros(n), damping = zeros(nθ),
        residual = zeros(n), gains = zeros(size(J, 1)),
        coarse = (; Z, U = zeros(n, m)), key = keys,
        factor = Ref{Union{Nothing, CHOLMOD.Factor{Float64, Int}}}(nothing), evidence = Ref{Any}(nothing),
    )
    return _set_data!(sys, H, b, θ)
end

# The products `Jᵀ[i, k]·H[k, l]·J[l, j]` that make up `JᵀHJ`: per product its
# entry `(i, j)`, the index `q` of `H[k, l]` among `H`'s stored values, and
# `J[k, i]·J[l, j]`.
function _product_terms(J, H)
    rows = sparse(J')
    Is, Js, qs, coefs = Int[], Int[], Int[], Float64[]
    for l in axes(H, 2), q in nzrange(H, l)
        k = rowvals(H)[q]
        for a in nzrange(rows, k), b in nzrange(rows, l)
            push!(Is, rowvals(rows)[a])
            push!(Js, rowvals(rows)[b])
            push!(qs, q)
            push!(coefs, nonzeros(rows)[a] * nonzeros(rows)[b])
        end
    end
    return Is, Js, qs, coefs
end

# Writes into `sys` (`_posterior_system`) the data's quadratic for the
# curvature `H`, on the pattern `sys` was made with, the gradient `b` and the
# parameters `θ`.
function _set_data!(sys, H, b, θ)
    (; layout, J, data_map) = sys
    size(data_map, 2) == nnz(H) || error("the data curvature changed its sparsity pattern")
    K = layout.K
    sys.θ .= θ
    mul!(sys.bθ, J', b)
    mul!(layout.base, data_map, nonzeros(H))
    h0 = fill!(layout.h0, 0.0)
    for j in eachindex(sys.θ), q in nzrange(K, j)
        h0[rowvals(K)[q]] += layout.base[q] * sys.θ[j]
    end
    Hθ = view(h0, eachindex(sys.θ))
    layout.c0[] = dot(sys.θ, Hθ) / 2 + dot(sys.bθ, sys.θ)
    Hθ .+= sys.bθ
    return sys
end

# The coarse space of a system's chains (`_track_chains!` of `tracks`) over `n`
# unknowns: one column per constant group of each track (its `constants`), one
# on its parameters in the pieces of that group.
function _coarse_basis(chains, tracks, n)
    I, Js = Int[], Int[]
    m = 0
    for (t, cs) in zip(tracks, chains), g in unique(t.constants)
        m += 1
        for (j, ch) in pairs(cs)
            t.constants[j] == g || continue
            append!(I, ch.gains)
            append!(Js, fill(m, length(ch.gains)))
        end
    end
    return sparse(I, Js, ones(length(I)), n, m)
end

# The system's quadratic as last assembled (`_assemble!`, which returned `c`)
# plus `λ/2·Σ d_j(θ′_j − θ_j)²` over the unknowns in `θ`, `d` their damping
# weights (`sys.damping`, written here): `(K, h, c)` of `½uᵀKu − hᵀu + c`, in
# buffers of `sys` the next call overwrites.
function _reduce!(sys, c, λ)
    (; layout, damped, θ) = sys
    d = _damping!(sys)
    copyto!(nonzeros(damped), nonzeros(layout.K))
    for (p, dj) in zip(layout.diag, d)
        nonzeros(damped)[p] += λ * dj
    end
    h = copyto!(sys.hλ, layout.h)
    view(h, eachindex(θ)) .+= λ .* d .* θ
    c += λ * sum(j -> d[j] * θ[j]^2, eachindex(θ, d); init = 0.0) / 2
    return damped, h, c
end

# Writes into `sys.damping` the damping weight of each unknown in `sys.θ`: its
# curvature in the quadratic
# as last assembled, which makes the damping independent of the parameters' units
# (Marquardt 1963).
_damping!(sys) = map!(p -> nonzeros(sys.layout.K)[p], sys.damping, sys.layout.diag)

_factor(K, key; check = true) = cholesky(Symmetric(K); perm = sortperm(key), check)

# The unknowns of `sys` at its current point: `θ` at its values and the latent
# unknowns at the minimum of the quadratic as last assembled given them.
function _current_unknowns(sys)
    (; layout, θ, key) = sys
    nθ = length(θ)
    L = (nθ + 1):size(layout.K, 1)
    isempty(L) && return copy(θ)
    rhs = layout.h[L] .- layout.K[L, 1:nθ] * θ
    return [θ; _factor(layout.K[L, L], key[L]) \ rhs]
end

# Factors `K` into `sys`'s factorization, reusing its symbolic analysis.
function _refactor!(sys, K)
    F = sys.factor[]
    F = isnothing(F) ? _factor(K, sys.key) : cholesky!(F, Symmetric(K))
    sys.factor[] = F
    return F
end

# ── Hyperparameters and branches ─────────────────────────────────────────────

"""
    SelectedInverse(F, K)

The entries of `K⁻¹` on `K`'s sparsity pattern, from its Cholesky factor `F` by
selected inversion (SelectedInversion.jl; the recursion of Takahashi, Fagan &
Chen 1973), called as `Σ(i, j)`. The pattern holds every pair of unknowns that
share a term of the energy, so these are the posterior covariances the expected
energy needs. Looking up a pair outside it throws. `_selected_inverse!(Σ, F)`
rewrites them for a refactorization `F` with the same symbolic analysis.
"""
struct SelectedInverse{M <: SparseMatrixCSC, P <: AbstractVector{Int}}
    Σ::M
    plan::P
end

function SelectedInverse(F, K)
    (; Z, p) = SelectedInversion.selinv(F)
    n = size(K, 1)
    # Each stored value of `K` tagged with its index, carried to the factor's
    # permuted order, in which `Z` holds `K⁻¹[p, p]`.
    tags = SparseMatrixCSC(n, n, copy(K.colptr), copy(rowvals(K)), collect(1.0:nnz(K)))[p, p]
    plan = zeros(Int, nnz(K))
    src = _selinv_plan(Z, tags)
    for (t, k) in zip(nonzeros(tags), src)
        plan[Int(t)] = k
    end
    Σ = SelectedInverse(SparseMatrixCSC(n, n, copy(K.colptr), copy(rowvals(K)), zeros(nnz(K))), plan)
    return _fill_selected!(Σ, Z)
end

# The index in the stored values of the selected inverse `Z` of each stored
# entry of `B`, both in the factor's permuted order. A simplicial `Z` stores its
# lower triangle.
_selinv_plan(Z::SelectedInversion.SupernodalMatrix, B) = SelectedInversion.selinv_extract_setup(Z, B)
function _selinv_plan(Z::Symmetric, B)
    return [_position(Z.data, max(i, j), min(i, j)) for j in axes(B, 2) for i in view(rowvals(B), nzrange(B, j))]
end

_fill_selected!(Σ::SelectedInverse, Z::SelectedInversion.SupernodalMatrix) = (SelectedInversion.selinv_extract!(Σ.Σ, Z, Σ.plan); Σ)
_fill_selected!(Σ::SelectedInverse, Z::Symmetric) = (map!(k -> nonzeros(Z.data)[k], nonzeros(Σ.Σ), Σ.plan); Σ)

_selected_inverse!(Σ::SelectedInverse, F) = _fill_selected!(Σ, SelectedInversion.selinv(F).Z)

function (S::SelectedInverse)(a, b)
    (; Σ) = S
    rng = nzrange(Σ, b)
    k = searchsortedfirst(view(rowvals(Σ), rng), a)
    k <= length(rng) && rowvals(Σ)[rng[k]] == a ||
        error("posterior covariance ($a, $b) is not on the system's sparsity pattern")
    return nonzeros(Σ)[rng[k]]
end

# How many hyperparameters `prior` gives as densities, to be estimated.
_hyper_count(::Nothing) = 0
_hyper_count(p::RandomWalkPrior) = is_fixed_hyper(p.σ) ? 0 : 1
_hyper_count(p::OUPrior) = (is_fixed_hyper(p.scale) ? 0 : 1) + (is_fixed_hyper(p.σ) ? 0 : 1)
_hyper_count(p::CARMAPrior) = count(!is_fixed_hyper, _hyper_specs(p))

# A `CARMAPrior`'s hyperparameters in order: each root factor's (`period` before
# `scale`), the autoregressive factors' first, then `σ`.
_hyper_specs(f::RealRoot) = (f.scale,)
_hyper_specs(f::ComplexRoots) = (f.period, f.scale)
_hyper_specs(p::CARMAPrior) = (_factor_specs(p.ar)..., _factor_specs(p.ma)..., p.σ)
_factor_specs(::Tuple{}) = ()
_factor_specs(fs::Tuple) = (_hyper_specs(first(fs))..., _factor_specs(Base.tail(fs))...)

# The support of the hyperparameters `spec` gives as densities, over their logs,
# on pieces of sample spacing `Δ` and span `W` (`_piece_extent`): a `CARMAPrior`'s
# scales lie in `[Δ, W]` and its periods in `[2Δ, W]`, beyond which the samples
# do not measure them (a shorter period aliases to a longer one). An
# `OUPrior`'s scale lies in `[Δ, 10W]`, its seed's bounds (`_ou_tau_bounds`):
# beyond the span the process is a random walk whose diffusion `2σ²/τ` is all
# the samples fix.
_log_hyper_box(spec, Δ, W) = fill((-Inf, Inf), _hyper_count(spec))
function _log_hyper_box(spec::OUPrior, Δ, W)
    T = float(promote_type(typeof(Δ), typeof(W)))
    box = NTuple{2, T}[]
    is_fixed_hyper(spec.scale) || push!(box, (log(T(Δ)), log(10 * T(max(W, Δ)))))
    is_fixed_hyper(spec.σ) || push!(box, (T(-Inf), T(Inf)))
    return box
end

# The support of a track's hyperparameters over its `chains`: what the samples
# measure (`_log_hyper_box`) within the hyperpriors' support (`_hyperprior_box`).
# Chains of one sample each measure no scale, and leave only the hyperpriors'.
function _track_box(spec, chains)
    any(ch -> length(ch.u) >= 2, chains) || return _hyperprior_box(spec)
    samples = _log_hyper_box(spec, _piece_extent([ch.u for ch in chains], [ch.h for ch in chains])...)
    return [(max(a, c), min(b, d)) for ((a, b), (c, d)) in zip(samples, _hyperprior_box(spec))]
end

# The hyperpriors of the hyperparameters `spec` gives as densities, in
# `_log_hypers` order.
_hyperpriors(spec::RandomWalkPrior) = filter(!is_fixed_hyper, [spec.σ])
_hyperpriors(spec::OUPrior) = filter(!is_fixed_hyper, [spec.scale, spec.σ])
_hyperpriors(spec::CARMAPrior) = filter(!is_fixed_hyper, collect(_hyper_specs(spec)))

# The tail probability beyond which a hyperprior is taken to hold no mass, that
# of a normal beyond six standard deviations.
const _HYPER_TAIL = 1.0e-9

# Each hyperparameter's support over its log, between the quantiles of its
# hyperprior at `_HYPER_TAIL` and `1 − _HYPER_TAIL`. It keeps an optimizer's
# trial points where the solve can evaluate them: far beyond, a prior stiffer or
# looser than the data by more than the rounding leaves the systems singular.
_hyperprior_box(spec) = [(log(quantile(d, _HYPER_TAIL)), log(quantile(d, 1 - _HYPER_TAIL))) for d in _hyperpriors(spec)]
function _log_hyper_box(spec::CARMAPrior, Δ, W)
    T = float(promote_type(typeof(Δ), typeof(W)))
    box = NTuple{2, T}[]
    for f in (spec.ar..., spec.ma...)
        f isa ComplexRoots && !is_fixed_hyper(f.period) && push!(box, log.((2Δ, W)))
        is_fixed_hyper(f.scale) || push!(box, log.((Δ, W)))
    end
    is_fixed_hyper(spec.σ) || push!(box, (T(-Inf), T(Inf)))
    return box
end

_in_box(lθ, box) = all(((x, (lo, hi)),) -> lo <= x <= hi, zip(lθ, box))
_into_box(lθ, box) = [clamp(x, lo, hi) for (x, (lo, hi)) in zip(lθ, box)]

# `p` with its hyperparameters, in `_hyper_specs` order, replaced by the tuple
# `values`.
function _with_hyper_values(p::CARMAPrior, values)
    ar, rest = _factors_from(p.ar, values)
    ma, rest = _factors_from(p.ma, rest)
    return CARMAPrior(; ar, ma, σ = only(rest))
end

# The factors `fs` rebuilt from the leading `values`, and the values left.
_factors_from(::Tuple{}, values) = (), values
function _factors_from(fs::Tuple, values)
    f, values = first(fs) isa RealRoot ? (RealRoot(; scale = values[1]), Base.tail(values)) :
        (ComplexRoots(; period = values[1], scale = values[2]), Base.tail(Base.tail(values)))
    rest, values = _factors_from(Base.tail(fs), values)
    return (f, rest...), values
end

# One hyperparameter `spec`, a value or a density: its value, from the log
# value `lθ[i]` for a density, and the index of the next.
_take(spec::Real, lθ, i) = spec, i
_take(spec, lθ, i) = exp(lθ[i]), i + 1

# `prior` with its hyperparameters given as densities at the log values `lθ`,
# `scale` before `σ`.
_with_log_hypers(p::RandomWalkPrior, lθ) = RandomWalkPrior(; p.order, σ = first(_take(p.σ, lθ, 1)), p.init)
function _with_log_hypers(p::OUPrior, lθ)
    scale, i = _take(p.scale, lθ, 1)
    return OUPrior(; scale, σ = first(_take(p.σ, lθ, i)))
end
_with_log_hypers(p::CARMAPrior, lθ) = _with_hyper_values(p, _hyper_values(_hyper_specs(p), lθ, 1))

# The values of the hyperparameters `specs`, those given as densities from the
# log values `lθ` starting at `lθ[i]`.
_hyper_values(::Tuple{}, lθ, i) = ()
function _hyper_values(specs::Tuple, lθ, i)
    v, j = _take(first(specs), lθ, i)
    return (v, _hyper_values(Base.tail(specs), lθ, j)...)
end

# The log values in `p`, the resolved `spec`, of the hyperparameters `spec`
# gives as densities.
_log_hypers(spec::RandomWalkPrior, p) = [log(p.σ)]
_log_hypers(spec::OUPrior, p) = [v for (h, v) in ((spec.scale, log(p.scale)), (spec.σ, log(p.σ))) if !is_fixed_hyper(h)]
_log_hypers(spec::CARMAPrior, p) = [log(v) for (h, v) in zip(_hyper_specs(spec), _hyper_specs(p)) if !is_fixed_hyper(h)]

# The hyperprior's log density over the log hyperparameters `lθ`.
_hyperprior_term(spec::Real, lθ, i) = zero(eltype(lθ)), i
_hyperprior_term(spec, lθ, i) = logdensityof(spec, exp(lθ[i])) + lθ[i], i + 1
_log_hyperprior(p::RandomWalkPrior, lθ) = first(_hyperprior_term(p.σ, lθ, 1))
function _log_hyperprior(p::OUPrior, lθ)
    a, i = _hyperprior_term(p.scale, lθ, 1)
    return a + first(_hyperprior_term(p.σ, lθ, i))
end
function _log_hyperprior(p::CARMAPrior, lθ)
    out, i = zero(eltype(lθ)), 1
    for spec in _hyper_specs(p)
        a, i = _hyperprior_term(spec, lθ, i)
        out += a
    end
    return out
end

# Each station gain's own estimate of one observable given every other gain and
# `S`, from its relative gain estimate `1 + c/W` (`_linearize`), and its weight;
# `NaN` and zero for a gain without data. With the phase held at zero the
# least-squares log-amplitude is that of `Re(1 + c/W)`.
function _gain_estimates(x, lin, observable; phase_held = false)
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

# Track `t`'s prior with its hyperparameters estimated from the station-gain
# estimates `y` (weights `wy`) as independent measurements of the track alone:
# type-II MAP over its pieces, the constants the source term absorbs integrated
# out. A phase is first lifted onto the branches `_branch_filter` finds most
# probable at the hyperprior medians. The joint solve starts from these.
function _seed_prior(t, y, wy; phase::Bool)
    ys = [y[t.gains[p]] for p in t.pieces]
    ws = [wy[t.gains[p]] for p in t.pieces]
    xs = [float.(t.x[p]) for p in t.pieces]
    design = _mean_designs(t)
    if phase
        start = _hyper_start(t.prior)
        ys = map(eachindex(ys, ws, xs, design)) do j
            model, θ, r, u = _filter_inputs(start, ys[j], ws[j], xs[j])
            any(isfinite, θ) || return ys[j]
            return first(_branch_filter(model, θ, r, u; mean = design[j].rows).histories).lift
        end
    end
    return _estimate_hypers(t.prior, ys, ws, xs; mean = design)
end

# The design of track `t`'s constants per piece: piece `j` takes constant
# `t.constants[j]`, the one the source term absorbs over it.
_mean_designs(t) = [(; cols = SVector(t.constants[j]), rows = fill(SVector(1.0), length(p))) for (j, p) in pairs(t.pieces)]

# Put phase track `t` on the branches `_branch_filter` finds most probable for its
# station-gain estimates `y` under the resolved `prior`: each free gain of `φ`
# moves by the multiple of 2π that brings it nearest the track fitted to those
# branches, which leaves its value as a complex gain unchanged. The fitted
# track, not the gain's own estimate, sets the branch: a gain whose data barely
# constrain it can have an estimate π from the track while its value lies on the
# track. A piece's branches are first moved together by the multiple of 2π
# nearest its estimates, and the pieces sharing a mean onto consistent branches
# (`_align_branches!`). Each piece's branches are chosen with its mean's
# coefficients its own. Returns the per-gain shifts (radians).
function _choose_branches!(φ, t, prior, y, wy, free)
    design = _mean_designs(t)
    rows(j) = design[j].rows
    ys = [y[t.gains[p]] for p in t.pieces]
    ws = [wy[t.gains[p]] for p in t.pieces]
    inputs = map(eachindex(t.pieces)) do j
        model, θ, r, u = _filter_inputs(prior, ys[j], ws[j], float.(t.x[t.pieces[j]]))
        any(isfinite, θ) || return nothing
        bf = _branch_filter(model, θ, r, u; mean = rows(j))
        lift = first(bf.histories).lift
        return (; model, r, u, lift = lift .+ 2π * round(_weighted_mean_finite(θ .- lift, ws[j]) / 2π))
    end
    lifts = [isnothing(v) ? ys[j] : v.lift for (j, v) in pairs(inputs)]
    _align_branches!(lifts, ws, design)
    shift = zeros(length(φ))
    for (j, p) in pairs(t.pieces)
        isnothing(inputs[j]) && continue
        (; model, r, u) = inputs[j]
        fitted = _fitted_track(model, lifts[j], r, u; mean = rows(j))
        for (i, k) in pairs(p)
            n = t.gains[k]
            (free[n] && isfinite(fitted[i])) || continue
            shift[n] = 2π * round((fitted[i] - φ[n]) / 2π)
            φ[n] += shift[n]
        end
    end
    return shift
end

# The track `smooth_track` fits to `y` under `model`; with a `mean` (one row of
# regressors per sample), the track's estimated mean plus the smoothed
# remainder, the coefficients its samples leave free at zero.
function _fitted_track(model, y, r, x; mean = nothing)
    isnothing(mean) && return smooth_track(model, y, r, x)
    _, b, C = _mean_sums(model, y, r, x, mean)
    β = _mean_inverse(C, _mean_determined(C)) * b
    m = [dot(h, β) for h in mean]
    return m .+ smooth_track(model, y .- m, r, x)
end
