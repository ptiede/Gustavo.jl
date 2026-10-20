# ── Choosing a phase track's 2π branches under a state-space prior ──────────
#
# A phase measured from a complex gain is known modulo 2π. The likelihood does
# not depend on the branch, but a prior on the real-valued track does, so fitting
# a phase track under a prior means choosing each sample's branch as well. Lifted
# to the real line, each sample's likelihood is a sum of Gaussians, one at each
# θ_k + 2πm; filtering a state-space prior through those sums carries one Kalman
# filter per branch history, weighted by its marginal likelihood (a Gaussian-sum
# filter). docs/src/priors.md derives it.

# One branch history: the filtered state mean, the log likelihood of its lifted
# samples, with a level the running `Σ vʸ·v¹/S` (see `_kalman_forward`), and the
# log of the weight merged into it from histories that came to predict the same.
# The covariance, gains, and the track of ones do not depend on the samples'
# values, so the histories share them.
struct _BranchHistory{D, T}
    μ::SVector{D, T}
    ll::T
    b::T
    extra::T
end

# A history's log weight once `c = Σ (v¹)²/S` is accumulated: its likelihood with
# the level integrated out under a flat prior.
_branch_score(h, c) = h.ll + (c > 0 ? h.b^2 / (2c) : zero(c)) + h.extra

"""
    _branch_filter(model, θ, r, x; level = false) -> (; histories, loglik, best)

Choose the 2π branch of each sample of the phase track `θ` (radians, on any
branch), observed with variances `r` at the coordinates `x` under the state-space
`model`, by a Gaussian-sum Kalman filter over branch histories. A sample with
non-finite `θ_k` or `r_k` not positive and finite is missing. With `level`, the
track is a flat unknown level plus `model`'s zero-mean process (`model` must have
a proper start) and the level is integrated out.

`histories` holds every history carried to the end as `(; lift, weight)`, most
probable first: `lift` is `θ` on that history's branches (`NaN` where missing) and
`weight` its posterior probability among them. `loglik` is the log marginal
likelihood of the wrapped track, summed over those histories, and `best` the log
marginal likelihood of the first history's `lift` alone, as `_pooled_loglik`
computes it for a single track.

A history branches at a sample over every 2π branch within `nsig` predictive
standard deviations of its prediction, and always the nearest. A history more
than `prune` below the best in log weight is dropped and at most `maxhyp` are
carried. Two histories whose predictions agree to `merge_tol` of their standard
deviations predict the same from there on, so they are merged: weights summed,
the more probable one's branches kept. A flat start's first `statedim(model)`
usable samples, which fix its state, each take the branch nearest the previous
one. For samples on consecutive points of an evenly spaced grid that loses
nothing, since any other choice differs by a polynomial of degree below
`statedim(model)` with values in 2πℤ on the grid, which the flat start absorbs.
"""
function _branch_filter(
        model, θ, r, x; level::Bool = false,
        nsig::Real = 5, prune::Real = 25, maxhyp::Integer = 64, merge_tol::Real = 1.0e-3,
    )
    Base.require_one_based_indexing(θ, r, x)
    T = _filter_eltype(model, θ, r, x)
    D = statedim(model)
    Vec = SVector{D, T}
    e1 = Vec(ntuple(i -> i == 1 ? one(T) : zero(T), Val(D)))
    usable = [k for k in eachindex(θ, r, x) if isfinite(θ[k]) && isfinite(r[k]) && r[k] > 0]
    start = initial(model, T)
    flat = isnothing(start)
    base = fill(T(NaN), length(θ))
    nfixed = flat ? min(D, length(usable)) : 0
    for j in 1:nfixed
        k = usable[j]
        base[k] = j == 1 ? T(θ[k]) : base[usable[j - 1]] + rem2pi(T(θ[k]) - base[usable[j - 1]], RoundNearest)
    end
    if flat
        nfixed < D && return (; histories = [(; lift = base, weight = one(T))], loglik = zero(T), best = zero(T))
        kd = usable[D]
        kf = kalman_filter(model, view(base, 1:kd), view(r, 1:kd), view(x, 1:kd))
        hyps = [_BranchHistory(kf.μf[kd], T(kf.loglik), zero(T), zero(T))]
        P = kf.Pf[kd]
        k0 = kd + 1
    else
        hyps = [_BranchHistory(Vec(start[1]), zero(T), zero(T), zero(T))]
        P = SMatrix{D, D, T}(start[2])
        k0 = firstindex(θ)
    end
    μ1 = zero(Vec)
    c = zero(T)
    # `(parent, branch)` of each history after each filtered sample `steps[s]`.
    trail = Vector{Vector{NTuple{2, Int}}}()
    steps = Int[]
    children, links = _BranchHistory{D, T}[], NTuple{2, Int}[]
    workspace = (; scores = Float64[], order = Int[], kept = Int[], head = Dict{Int, Int}(), next = Int[])
    for k in k0:lastindex(θ)
        if k > firstindex(θ)
            A, Q = transition(model, x[k] - x[k - 1], T)
            for (i, h) in pairs(hyps)
                hyps[i] = _BranchHistory(A * h.μ, h.ll, h.b, h.extra)
            end
            μ1 = A * μ1
            P = _symmetric(A * P * A' + Q)
        end
        (isfinite(θ[k]) && isfinite(r[k]) && r[k] > 0) || continue
        θk, rk = T(θ[k]), T(r[k])
        S = P[1, 1] + rk
        S > 0 || _throw_precision_loss(model, "innovation variance", S, k)
        K = P[:, 1] / S
        v1 = one(T) - μ1[1]
        # Before a level is measured the prediction is flat, and the level absorbs
        # whichever branch the sample takes.
        proper = !level || c > 0
        sd = sqrt(level && c > 0 ? S + v1^2 / c : S)
        span = proper ? Int(min(ceil(nsig * sd / 2π), maxhyp)) : 0
        empty!(children)
        empty!(links)
        for (i, h) in pairs(hyps)
            center = h.μ[1] + (level && c > 0 ? h.b / c * v1 : zero(T))
            m0 = round(Int, (center - θk) / 2π)
            for m in (m0 - span):(m0 + span)
                y = θk + 2π * m
                (m == m0 || abs(y - center) <= nsig * sd) || continue
                v = y - h.μ[1]
                push!(children, _BranchHistory(h.μ + K * v, h.ll - (log(2π * S) + v^2 / S) / 2, level ? h.b + v * v1 / S : zero(T), h.extra))
                push!(links, (i, m))
            end
        end
        if level
            μ1 += K * v1
            c += v1^2 / S
        end
        IK = I - K * e1'
        P = _symmetric(IK * P * IK' + K * rk * K')
        kept = _prune_histories!(workspace, children, c, P; level, prune, maxhyp, merge_tol)
        hyps = children[kept]
        push!(trail, links[kept])
        push!(steps, k)
    end
    norm = level && c > 0 ? -log(c) / 2 : zero(T)
    scores = _branch_score.(hyps, c) .+ norm
    loglik = _logsumexp(scores)
    order = sortperm(scores; rev = true)
    histories = [(; lift = _branch_lift(base, θ, trail, steps, i), weight = exp(scores[i] - loglik)) for i in order]
    best = scores[first(order)] - hyps[first(order)].extra
    return (; histories, loglik, best)
end


# The indices of `children` to carry on, most probable first: within `prune` of
# the best, at most `maxhyp`, each history that predicts the same as a more
# probable one merged into the most probable such. Kept histories are binned by
# their first state in units of its merge tolerance, so a match lies in a
# history's own bin or a neighbor; `head` and `next` chain each bin's members.
function _prune_histories!(ws, children, c, P; level, prune, maxhyp, merge_tol)
    (; scores, order, kept, head, next) = ws
    resize!(scores, length(children))
    map!(h -> _branch_score(h, c), scores, children)
    sortperm!(resize!(order, length(children)), scores; rev = true)
    tol = merge_tol .* sqrt.(diag(P))
    empty!(kept)
    empty!(head)
    empty!(next)
    best = scores[first(order)]
    for i in order
        scores[i] < best - prune && break
        h = children[i]
        bin = floor(Int, h.μ[1] / tol[1])
        j = 0
        for b in (bin - 1):(bin + 1)
            m = get(head, b, 0)
            while !iszero(m)
                o = children[kept[m]]
                if (iszero(j) || m < j) && all(abs.(h.μ - o.μ) .<= tol) && (!level || abs(h.b - o.b) <= merge_tol * sqrt(c))
                    j = m
                end
                m = next[m]
            end
        end
        if iszero(j)
            if length(kept) < maxhyp
                push!(kept, i)
                push!(next, get(head, bin, 0))
                head[bin] = length(kept)
            end
        else
            g = children[kept[j]]
            children[kept[j]] = _BranchHistory(g.μ, g.ll, g.b, g.extra + log1p(exp(scores[i] - _branch_score(g, c))))
        end
    end
    return kept
end

# `base` with history `i`'s branch at every sample the filter chose one for.
function _branch_lift(base, θ, trail, steps, i)
    lift = copy(base)
    for s in reverse(eachindex(trail, steps))
        parent, m = trail[s][i]
        lift[steps[s]] = θ[steps[s]] + 2π * m
        i = parent
    end
    return lift
end

function _logsumexp(v)
    m = maximum(v)
    return m + log(sum(x -> exp(x - m), v))
end

# `_pooled_loglik` for phase tracks whose 2π branches are not known: each track
# on its most probable branches (the tracks sharing a level then on one branch),
# plus the log of the total weight of the histories `_branch_filter` carried
# relative to that one's. Without levels that is the tracks' marginal likelihood
# summed over their branches; with them each track's sum integrates its own
# level.
function _branch_pooled_loglik(models, θs, rs, xs, levels)
    lifts = similar(θs)
    extra = zero(eltype(eltype(θs)))
    for i in eachindex(θs, rs, xs)
        bf = _branch_filter(_track_model(models, i), θs[i], rs[i], xs[i]; level = !isnothing(levels))
        lifts[i] = first(bf.histories).lift
        extra += bf.loglik - bf.best
    end
    isnothing(levels) || _align_branches!(lifts, [inv.(r) for r in rs], levels)
    return _pooled_loglik(models, lifts, rs, xs, levels) + extra
end

# Shift each phase piece by the multiple of 2π that brings its weighted mean
# nearest that of the first piece with data in its level group, so a level shared
# across spectral windows sees one branch.
function _align_branches!(ys, ws, level)
    ref = Dict{Int, eltype(eltype(ys))}()
    for j in eachindex(ys, ws, level)
        any(k -> _shape_usable(ys[j][k], ws[j][k]), eachindex(ys[j], ws[j])) || continue
        m = _weighted_mean_finite(ys[j], ws[j])
        r = get!(ref, level[j], m)
        ys[j] .-= 2π * round((m - r) / 2π)
    end
    return ys
end

# The posterior probability, among the histories `_branch_filter` carried, that
# the fitted track differs from the most probable history's by more than π/2 at
# some sample once a common phase is removed: that the fit carries a 2π slip.
# The smoother is linear, so two histories' fits differ by the fit of their
# difference (its own level removed, with `level`).
function _slip_probability(model, histories, r, x; level::Bool)
    best = first(histories).lift
    p = zero(first(histories).weight)
    for h in Iterators.drop(histories, 1)
        Δ = h.lift .- best
        if level
            _, b, c = _level_sums(model, Δ, r, x)
            Δ .-= b / c
        end
        d = smooth_track(model, Δ, r, x)
        z = angle(sum(cis, d))
        any(v -> abs(rem2pi(v - z, RoundNearest)) > π / 2, d) && (p += h.weight)
    end
    return p
end
