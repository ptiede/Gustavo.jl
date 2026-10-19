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
# `(row, unknown, coefficient)` of those rows, each row's constant and the fixed
# slot (or 0) whose value adds to it, and the index among K's stored values of
# each pair of entries (`_locate!`).
function _prior_term(fs, xref)
    entries = [(r, j, a) for (r, f) in pairs(fs) for (j, a) in f[1]]
    return (; entries, base = [f[2] for f in fs], xref, pos = zeros(Int, length(entries), length(entries)))
end

function _locate!(term, K)
    for (a, e) in pairs(term.entries), (b, g) in pairs(term.entries)
        term.pos[a, b] = _position(K, e[2], g[2])
    end
    return term
end

# The prior terms of chain `ch` (`_track_chains!`): its start, which counts where
# the prior's start is proper, and one per transition, over the stacked states
# `(s_k, s_{k−1})`. With `fixed`, the first row of each state takes its slot's
# value from `_fix_slots!`.
function _chain_terms(ch, fixed)
    xref(k) = [r == 1 && fixed ? ch.slots[k] : 0 for r in eachindex(ch.states[k])]
    return (;
        start = _prior_term(ch.states[1], xref(1)),
        steps = [_prior_term([ch.states[k]; ch.states[k - 1]], [xref(k); xref(k - 1)]) for k in 2:length(ch.states)],
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
# Its pattern is fixed here; `_assemble!` writes its values for given priors.
function _quadratic_layout(chains, n; data = (Int[], Int[], Float64[]), h = zeros(n), c = 0.0, nf = 0, fixed = false)
    terms = [[_chain_terms(ch, fixed) for ch in cs] for cs in chains]
    Is, Js = [data[1]; 1:nf], [data[2]; 1:nf]
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
    return (; K, base, h0 = h, c0 = c, h = similar(h), diag = [_position(K, j, j) for j in 1:nf], terms)
end

# Writes the slot values `x` into the rows of `lay`'s prior terms that take a
# fixed slot's value (`_chain_terms` with `fixed`).
function _fix_slots!(lay, x)
    _foreach_term(lay.terms) do term
        for (p, n) in pairs(term.xref)
            iszero(n) || (term.base[p] = x[n])
        end
    end
    return lay
end

# Writes the quadratic under the resolved `priors` (one per track) into
# `lay.K`'s stored values and `lay.h`, and returns its constant.
function _assemble!(lay, priors)
    Kv = nonzeros(lay.K)
    T = eltype(Kv)
    copyto!(Kv, lay.base)
    copyto!(lay.h, lay.h0)
    c = lay.c0
    for (ts, prior) in zip(lay.terms, priors)
        isempty(ts) && continue
        c += _with_piece_models(prior, first(ts).dim, T) do model
            # A loop: inference widens a sum over a generator here to Any when _fit_hypers! calls it.
            ct = zero(T)
            for t in ts
                ct += _add_chain!(Kv, lay.h, t, model(t.h))
            end
            ct
        end
    end
    return c
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
    for (k, term) in pairs(t.steps)
        A, Q = transition(model, t.u[k + 1] - t.u[k], T)
        G = hcat(SMatrix{D, D, T}(I), -A)
        c += _add_term!(Kv, h, term, G' * (Q \ G), zero(SVector{2D, T}))
    end
    return c
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
function _piece_coordinates(::RandomWalkPrior, x)
    _, u, h = _random_walk_problem(x, ones(length(x)), x)
    return u, h
end
_piece_coordinates(::OUPrior, x) = float.(x), 1.0

# `f(model)` with `model(h)` the resolved `prior`'s state-space model of state
# dimension `dim` (a chain's, `Val(_statedim(prior))`) on coordinates of unit
# `h`, in element type `T`.
_with_piece_models(f, prior::RandomWalkPrior, dim::Val, ::Type{T}) where {T} = f(h -> _random_walk_model(dim, prior.σ, prior.init, h, T))
_with_piece_models(f, prior::OUPrior, ::Val, ::Type{T}) where {T} = f(_ -> OUModel(T(prior.scale), T(prior.σ)^2))
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

# The state chain of each piece of track `t` its prior acts on, over unknowns
# whose frequencies `keys` holds: per sample its state less the piece's level, as
# linear forms whose first is the slot (`slot(n)` gives its unknown, or `0` and
# its fixed value) and the rest new latent unknowns, with the coordinates its
# prior runs on, its samples' slots and its state dimension as a type
# (`Val`). Returns the chains and the unknown of each level group.
function _track_chains!(keys, t, slot)
    levels = Dict{Int, Int}()
    chains = [_piece_chain!(keys, levels, t, slot, j, p) for (j, p) in pairs(t.pieces) if t.active[j]]
    return chains, levels
end

function _piece_chain!(keys, levels, t, slot, j, p)
    group = _level_group(t, t.prior, j)
    L = isnothing(group) ? 0 : get!(() -> _unknown!(keys, Inf), levels, group)
    states = map(p) do k
        j1, v1 = slot(t.slots[k])
        value = (iszero(j1) ? Pair{Int, Float64}[] : [j1 => 1.0], iszero(j1) ? v1 : 0.0)
        iszero(L) || push!(value[1], L => -1.0)
        _Form[value; [([_unknown!(keys, t.x[k]) => 1.0], 0.0) for _ in 2:_statedim(t.prior)]]
    end
    u, h = _piece_coordinates(t.prior, t.x[p])
    return (; u, h, states, slots = t.slots[p], dim = Val(_statedim(t.prior)))
end

# The log density of `chains` under the resolved `prior`, in expectation over
# the states' posterior with the moments `mom` (`_chain_moments`), or its
# normalization alone without them; in element type `T`.
function _expected_log_prior(prior, chains, mom, ::Type{T}) where {T}
    isempty(chains) && return zero(T)
    return _with_piece_models(prior, first(chains).dim, T) do model
        # A loop: inference widens a sum over a generator here to Any when _fit_hypers! calls it.
        out = zero(T)
        for (c, ch) in pairs(chains)
            out += _expected_chain_log_prior(model(ch.h), ch.u, isnothing(mom) ? nothing : mom[c], T)
        end
        out
    end
end

function _expected_chain_log_prior(model, u, mom, ::Type{T}) where {T}
    D = statedim(model)
    out = zero(T)
    start = initial(model, T)
    if !isnothing(start)
        μ0, P0 = start
        out -= logdet(2π * P0) / 2
        if !isnothing(mom)
            m, M = mom.start
            out -= tr(P0 \ (M - μ0 * m' - m * μ0' + μ0 * μ0')) / 2
        end
    end
    for i in 2:length(u)
        A, Q = transition(model, u[i] - u[i - 1], T)
        out -= logdet(2π * Q) / 2
        isnothing(mom) && continue
        G = hcat(SMatrix{D, D, T}(I), -A)
        out -= tr(Q \ (G * mom.pairs[i - 1] * G')) / 2
    end
    return out
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

# Track `t`'s prior over its latent states and levels with its slots fixed, for
# `_track_energy`: the quadratic's layout, the unknown of each level group, and
# the factorization of the prior it was last assembled under.
function _track_layout(t)
    keys = Float64[]
    chains, levels = _track_chains!(keys, t, _ -> (0, 0.0))
    layout = _quadratic_layout([chains], length(keys); fixed = true)
    return (; layout, levels, factor = Ref{Union{Nothing, CHOLMOD.Factor{Float64, Int}}}(nothing), prior = Ref{Any}(nothing))
end

# A track's prior energy at the slot values `x`, minimized over its latent states
# and levels, and those levels by group; `tl` is its `_track_layout`.
function _track_energy(tl, prior, x)
    (; layout) = tl
    c = _assemble!(_fix_slots!(layout, x), (prior,))
    isempty(layout.h) && return c, Dict{Int, Float64}()
    F = tl.factor[]
    if isnothing(F) || !isequal(tl.prior[], prior)
        F = isnothing(F) ? cholesky(Symmetric(layout.K)) : cholesky!(F, Symmetric(layout.K))
        tl.factor[] = F
        tl.prior[] = prior
    end
    u = F \ layout.h
    return c - dot(layout.h, u) / 2, Dict(g => u[j] for (g, j) in tl.levels)
end

"""
    _posterior_system(H, b, x, free, tracks, key; gauge = nothing)

The MAP problem of one observable's slot values `x` with the χ² to second order
(data curvature `H` and gradient `b` at `x`, over every slot) and the exact prior
energy of each of `tracks`, over the `free` slots (the other slots keep their
values), the tracks' latent states and their levels: the data's quadratic and
each track's state chains (`_track_chains!`), the priors added by
`_assemble!`. `gauge`, when given, holds constraint rows `(slot, coefficient)` and
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
    keys = key[free]
    Hf = H[free, free]
    xf = Float64.(x[free])
    Hx = Hf * xf
    slot(n) = free[n] ? (idx[n], 0.0) : (0, Float64(x[n]))
    built = [_track_chains!(keys, t, slot) for t in tracks]
    n = length(keys)
    Is, Js, Vs = findnz(Hf)
    stored = findall(!iszero, Vs)
    layout = _quadratic_layout(
        first.(built), n; data = (Is[stored], Js[stored], Vs[stored]), h = [b[free] .+ Hx; zeros(n - nf)],
        c = dot(xf, Hx) / 2 + dot(b[free], xf), nf,
    )
    g = _gauge_map(gauge, x, free, idx, n)
    damped = copy(layout.K)
    return (;
        layout, chains = first.(built), levels = last.(built), x = Float64.(x), free, xf, gauge = g,
        damped, hλ = zeros(n), reduction = isnothing(g) ? nothing : _reduction(damped, g),
        key = isnothing(g) ? keys : keys[g.keep], factors = Union{Nothing, CHOLMOD.Factor{Float64, Int}}[nothing, nothing, nothing],
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

# `TᵀKT` for the gauge map `g`, on the pattern `K`'s fixes, with the sparse map
# from `K`'s stored values to its own, and buffers for `_reduce!`.
function _reduction(K, g)
    unit(A) = SparseMatrixCSC(size(A)..., A.colptr, rowvals(A), ones(nnz(A)))
    Kr = unit(g.Tt) * unit(K) * unit(g.T)
    fill!(nonzeros(Kr), 0.0)
    rows, cols, vals = Int[], Int[], Float64[]
    rv, tv = rowvals(g.Tt), nonzeros(g.Tt)
    for j in axes(K, 2), s in nzrange(K, j), p in nzrange(g.Tt, rowvals(K)[s]), q in nzrange(g.Tt, j)
        push!(rows, _position(Kr, rv[p], rv[q]))
        push!(cols, s)
        push!(vals, tv[p] * tv[q])
    end
    return (; K = Kr, map = sparse(rows, cols, vals, nnz(Kr), nnz(K)), Kt0 = zeros(size(K, 1)), h = zeros(size(Kr, 1)))
end

# The system's quadratic as last assembled (`_assemble!`, which returned `c`)
# plus `λ/2·‖x′ − x‖²` over the free slots, over the unknowns the gauge leaves:
# `(K, h, c)` of `½uᵀKu − hᵀu + c`, in buffers of `sys` the next call overwrites.
function _reduce!(sys, c, λ)
    (; layout, damped, xf) = sys
    copyto!(nonzeros(damped), nonzeros(layout.K))
    for p in layout.diag
        nonzeros(damped)[p] += λ
    end
    h = copyto!(sys.hλ, layout.h)
    view(h, eachindex(xf)) .+= λ .* xf
    c += λ * dot(xf, xf) / 2
    isnothing(sys.gauge) && return damped, h, c
    r, t0 = sys.reduction, sys.gauge.t0
    mul!(nonzeros(r.K), r.map, nonzeros(damped))
    mul!(r.Kt0, damped, t0)
    c += dot(t0, r.Kt0) / 2 - dot(h, t0)
    r.Kt0 .= h .- r.Kt0
    mul!(r.h, sys.gauge.Tt, r.Kt0)
    return r.K, r.h, c
end

# The unknowns `u` in the system's own.
_unreduced(sys, u) = isnothing(sys.gauge) ? u : sys.gauge.T * u + sys.gauge.t0

_factor(K, key; check = true) = cholesky(Symmetric(K); perm = sortperm(key), check)

# Factors `K` into `sys`'s factorization `i`, reusing its symbolic analysis: 1
# and 2 for `_fit_hypers!`, 3 for `_map_step`.
function _refactor!(sys, i, K; check = true)
    F = sys.factors[i]
    F = isnothing(F) ? _factor(K, sys.key; check) : cholesky!(F, Symmetric(K); check)
    sys.factors[i] = F
    return F
end

"""
    _map_step(sys, priors; λ = 0) -> (x′, levels, model)

One Levenberg–Marquardt step of the MAP objective of the system `sys`
(`_posterior_system`) with each track's resolved prior from `priors`: the
minimum of its quadratic plus `λ/2·‖x′ − x‖²` over the free slots. Returns the
new slot values, each track's level values by group, and the undamped
quadratic's value there, which is the prior energy alone at `x′ = x`.
"""
function _map_step(sys, priors; λ = 0)
    c = _assemble!(sys.layout, priors)
    K, h, _ = _reduce!(sys, c, λ)
    z = _unreduced(sys, _refactor!(sys, 3, K) \ h)
    x′ = copy(sys.x)
    x′[sys.free] .= z[eachindex(sys.xf)]
    (; layout) = sys
    return x′, [Dict(g => z[j] for (g, j) in lv) for lv in sys.levels], dot(z, layout.K * z) / 2 - dot(layout.h, z) + c
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
function _covariance(cov::C, gauge) where {C}
    Tt = gauge.Tt
    rv, v = rowvals(Tt), nonzeros(Tt)
    return (a, b) -> sum((v[p] * v[q] * cov(rv[p], rv[q]) for p in nzrange(Tt, a) for q in nzrange(Tt, b)); init = 0.0)
end

# How many hyperparameters `prior` gives as densities, to be estimated.
_hyper_count(::Nothing) = 0
_hyper_count(p::RandomWalkPrior) = is_fixed_hyper(p.σ) ? 0 : 1
_hyper_count(p::OUPrior) = (is_fixed_hyper(p.scale) ? 0 : 1) + (is_fixed_hyper(p.σ) ? 0 : 1)

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

# The log values in `p`, the resolved `spec`, of the hyperparameters `spec`
# gives as densities.
_log_hypers(spec::RandomWalkPrior, p) = [log(p.σ)]
_log_hypers(spec::OUPrior, p) = [v for (h, v) in ((spec.scale, log(p.scale)), (spec.σ, log(p.σ))) if !is_fixed_hyper(h)]

# The hyperprior's log density over the log hyperparameters `lθ`.
_hyperprior_term(spec::Real, lθ, i) = zero(eltype(lθ)), i
_hyperprior_term(spec, lθ, i) = logdensityof(spec, exp(lθ[i])) + lθ[i], i + 1
_log_hyperprior(p::RandomWalkPrior, lθ) = first(_hyperprior_term(p.σ, lθ, 1))
function _log_hyperprior(p::OUPrior, lθ)
    a, i = _hyperprior_term(p.scale, lθ, 1)
    return a + first(_hyperprior_term(p.σ, lθ, i))
end

# Derivatives of `f` over the log hyperparameters `lθ` of `spec`, in one chunk
# whose size the type of `spec` fixes, so their types infer.
_hyper_chunk(spec) = ForwardDiff.Chunk{_hyper_count(spec)}()
_hyper_gradient(f, lθ, spec) = ForwardDiff.gradient(f, lθ, ForwardDiff.GradientConfig(f, lθ, _hyper_chunk(spec)))
_hyper_hessian(f, lθ, spec) = ForwardDiff.hessian(f, lθ, ForwardDiff.HessianConfig(f, lθ, _hyper_chunk(spec)))

# The inverse of a symmetric curvature, its eigenvalues taken in magnitude.
function _inverse_curvature(C)
    E = eigen(Symmetric(C))
    return E.vectors * Diagonal(inv.(abs.(E.values))) * E.vectors'
end

"""
    _fit_hypers!(priors, curvature, sys, tracks, λ; tolerance, max_iterations) -> (; converged, steps)

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
call, each track's step shortened to change no hyperparameter by more than a
factor of 10 and backtracked until the log posterior is no lower. It stops once
the Newton step is within `tolerance` of its standard error, `gᵀH⁻¹g ≤
tolerance²`, by every track's curvature measured along a step of its own, and
returns whether it did within `max_iterations` steps and how many steps it took.
"""
function _fit_hypers!(priors, curvature, sys, tracks, λ; tolerance, max_iterations)
    est = [i for i in eachindex(tracks) if _hyper_count(tracks[i].prior) > 0]
    isempty(est) && return (; converged = true, steps = 0)
    specs = [tracks[i].prior for i in est]
    θ = [_log_hypers(tracks[i].prior, priors[i]) for i in est]
    # Each evaluation factors into one of two factorizations, the accepted point's
    # kept while a trial point uses the other.
    function evaluate(θ, slot)
        trial = copy(priors)
        for (k, i) in pairs(est)
            trial[i] = _with_log_hypers(tracks[i].prior, θ[k])
        end
        K, h, c = _reduce!(sys, _assemble!(sys.layout, trial), λ)
        F = _refactor!(sys, slot, K; check = false)
        # Hyperparameters so extreme that the posterior's curvature is lost to
        # rounding are no candidate.
        issuccess(F) || return (; lp = -Inf, F, u = h, trial)
        u = F \ h
        lp = dot(h, u) / 2 - c - logdet(F) / 2
        lp += sum(_expected_log_prior(p, ch, nothing, Float64) for (p, ch) in zip(trial, sys.chains); init = 0.0)
        lp += sum(_log_hyperprior(tracks[i].prior, θ[k]) for (k, i) in pairs(est))
        return (; lp, F, u, trial)
    end
    # Each track's expected log prior density and hyperprior, as a function of
    # its log hyperparameters.
    function expected(at)
        cov = _covariance(_selected_inverse(at.F), sys.gauge)
        z = _unreduced(sys, at.u)
        moms = [_chain_moments(sys.chains[i], z, cov, first(sys.chains[i]).dim) for i in est]
        return [
            lθ -> _expected_log_prior(_with_log_hypers(tracks[i].prior, lθ), sys.chains[i], moms[k], eltype(lθ)) +
                _log_hyperprior(tracks[i].prior, lθ)
                for (k, i) in pairs(est)
        ]
    end
    cur = 1
    at = evaluate(θ, cur)
    isfinite(at.lp) || throw(PosDefException(0))
    fs = expected(at)
    for (k, i) in pairs(est)
        isnothing(curvature[i]) &&
            (curvature[i] = (; inverse = _inverse_curvature(-_hyper_hessian(fs[k], θ[k], specs[k])), measured = false))
    end
    g = map(_hyper_gradient, fs, θ, specs)
    converged, steps = false, 0
    for _ in 1:max_iterations
        Hinv = [curvature[i].inverse for i in est]
        if sum(dot.(g, Hinv .* g)) <= tolerance^2 && all(i -> curvature[i].measured, est)
            converged = true
            break
        end
        # At most a factor of 10 per step: the curvature carried over from the last
        # call may not fit this linearization, and a no-worse step can overshoot far.
        d = map(v -> v .* min(1, log(10) / maximum(abs, v)), Hinv .* g)
        α = 1.0
        next = nothing
        while true
            θ1 = θ .+ α .* d
            all(map(_within_rounding, θ1, θ)) && break
            trial = evaluate(θ1, 3 - cur)
            # The log posterior is known to its rounding.
            trial.lp >= at.lp - 16 * eps(abs(at.lp)) && (next = (θ1, trial); break)
            α /= 2
        end
        # No step changes the hyperparameters: they are at the maximum to rounding.
        isnothing(next) && (converged = true; break)
        θ1, at = next
        steps += 1
        cur = 3 - cur
        fs = expected(at)
        g1 = map(_hyper_gradient, fs, θ1, specs)
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
    return (; converged, steps)
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
# multiple of 2π that brings it nearest the track fitted to those branches, which
# leaves its gain unchanged. The fitted track, not the slot's own estimate, sets
# the branch: a slot whose data barely constrain it can have an estimate π from
# the track while its value lies on the track. A piece's branches are first moved
# together by the multiple of 2π nearest its estimates, and the pieces sharing a
# level group onto one branch. Writes each piece's `_slip_probability` into
# `slip` and returns the per-slot shifts (radians).
function _choose_branches!(φ, t, prior, y, wy, free; slip = nothing)
    groups = _level_groups(t, prior)
    level = !isnothing(groups)
    ys = [y[t.slots[p]] for p in t.pieces]
    ws = [wy[t.slots[p]] for p in t.pieces]
    inputs = map(eachindex(t.pieces)) do j
        model, θ, r, u = _filter_inputs(prior, ys[j], ws[j], float.(t.x[t.pieces[j]]))
        any(isfinite, θ) || return nothing
        bf = _branch_filter(model, θ, r, u; level)
        isnothing(slip) || (slip[j] = _slip_probability(model, bf.histories, r, u; level))
        lift = first(bf.histories).lift
        return (; model, r, u, lift = lift .+ 2π * round(_weighted_mean_finite(θ .- lift, ws[j]) / 2π))
    end
    lifts = [isnothing(v) ? ys[j] : v.lift for (j, v) in pairs(inputs)]
    level && _align_branches!(lifts, ws, groups)
    shift = zeros(length(φ))
    for (j, p) in pairs(t.pieces)
        isnothing(inputs[j]) && continue
        (; model, r, u) = inputs[j]
        fitted = _fitted_track(model, lifts[j], r, u; level)
        for (i, k) in pairs(p)
            n = t.slots[k]
            (free[n] && isfinite(fitted[i])) || continue
            shift[n] = 2π * round((fitted[i] - φ[n]) / 2π)
            φ[n] += shift[n]
        end
    end
    return shift
end

# The track `smooth_track` fits to `y` under `model`; with `level`, the track's
# estimated level plus the smoothed remainder.
function _fitted_track(model, y, r, x; level::Bool)
    level || return smooth_track(model, y, r, x)
    _, b, c = _level_sums(model, y, r, x)
    return b / c .+ smooth_track(model, y .- b / c, r, x)
end

# The slots of the phase pieces under a prior whose 2π branches the data do not
# determine: those whose slip probability under the track's resolved prior
# exceeds `max_slip`, with every piece sharing a level with one.
function _undetermined_slots(tracks, acting, priors, y, wy, max_slip)
    out = Set{Int}()
    for (j, i) in pairs(acting)
        t = tracks[i]
        slip = fill(NaN, length(t.pieces))
        _choose_branches!(zeros(length(y)), t, priors[j], y, wy, falses(length(y)); slip)
        undetermined = [p for p in eachindex(slip) if t.active[p] && slip[p] > max_slip]
        isnothing(t.level) || (undetermined = findall(in(t.level[undetermined]), t.level))
        foreach(p -> union!(out, t.slots[t.pieces[p]]), undetermined)
    end
    return out
end

# `_choose_branches!`, undone unless it leaves track `t`'s prior energy no
# higher (`tl` is its `_track_layout`). A shift leaves every gain unchanged, so
# this keeps the branch choice a descent move of the posterior.
function _choose_descending_branches!(φ, tl, t, prior, y, wy, free; slip = nothing)
    e0 = first(_track_energy(tl, prior, φ))
    before = φ[t.slots]
    shift = _choose_branches!(φ, t, prior, y, wy, free; slip)
    if any(!iszero, shift) && first(_track_energy(tl, prior, φ)) > e0
        φ[t.slots] .= before
        fill!(shift, 0.0)
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
