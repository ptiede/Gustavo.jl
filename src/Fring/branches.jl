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
# samples, with a mean the running `Σ vʸ·vᴴ/S` (see `_kalman_forward`), and the
# log of the weight merged into it from histories that came to predict the same.
# The covariance, gains, and the filtered regressors do not depend on the
# samples' values, so the histories share them.
struct _BranchHistory{D, NH, T}
    μ::SVector{D, T}
    ll::T
    b::SVector{NH, T}
    extra::T
end

# A history's log weight given `Ci`, the inverse of `C = Σ vᴴ·vᴴᵀ/S` once the
# samples determine the mean: its likelihood with the mean integrated out under
# a flat prior.
_branch_score(h, Ci) = h.ll + dot(h.b, Ci * h.b) / 2 + h.extra

# Whether the samples so far determine every coefficient of the mean.
_mean_determined(C::SMatrix{0, 0}) = true
_mean_determined(C) = all(>(0), diag(C)) && det(C) > sqrt(eps(eltype(C))) * prod(diag(C))

# `C`'s inverse where it is determined, else its pseudo-inverse.
_mean_inverse(C::SMatrix{0, 0}, determined) = C
_mean_inverse(C, determined) = determined ? inv(C) : typeof(C)(pinv(Matrix(C)))

"""
    _branch_filter(model, θ, r, x; mean = nothing) -> (; histories, loglik, best)

Choose the 2π branch of each sample of the phase track `θ` (radians, on any
branch), observed with variances `r` at the coordinates `x` under the state-space
`model`, by a Gaussian-sum Kalman filter over branch histories. A sample with
non-finite `θ_k` or `r_k` not positive and finite is missing. With `mean`, one
row of regressors per sample (`SVector`s), the track is a linear model in them
with flat unknown coefficients plus `model`'s zero-mean process (`model` must
have a proper start), and the coefficients are integrated out; until the samples
determine them, each sample takes the branch nearest the track's prediction.

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
function _branch_filter(model, θ, r, x; mean = nothing, kw...)
    return _branch_filter(model, θ, r, x, mean, Val(_mean_width(mean)); kw...)
end

_mean_width(::Nothing) = 0
_mean_width(::AbstractVector{<:StaticVector{NH}}) where {NH} = NH

function _branch_filter(
        model, θ, r, x, mean, ::Val{NH};
        nsig::Real = 5, prune::Real = 25, maxhyp::Integer = 64, merge_tol::Real = 1.0e-3,
    ) where {NH}
    Base.require_one_based_indexing(θ, r, x)
    T = _filter_eltype(model, θ, r, x)
    D = statedim(model)
    Vec = SVector{D, T}
    e1 = Vec(ntuple(i -> i == 1 ? one(T) : zero(T), Val(D)))
    usable = [k for k in eachindex(θ, r, x) if isfinite(θ[k]) && isfinite(r[k]) && r[k] > 0]
    start = initial(model, T)
    flat = isnothing(start)
    flat && NH > 0 &&
        throw(ArgumentError("a mean cannot be separated from a flat-start $(nameof(typeof(model)))"))
    base = fill(T(NaN), length(θ))
    nfixed = flat ? min(D, length(usable)) : 0
    for j in 1:nfixed
        k = usable[j]
        base[k] = j == 1 ? T(θ[k]) : base[usable[j - 1]] + rem2pi(T(θ[k]) - base[usable[j - 1]], RoundNearest)
    end
    nob = zero(SVector{NH, T})
    if flat
        nfixed < D && return (; histories = [(; lift = base, weight = one(T))], loglik = zero(T), best = zero(T))
        kd = usable[D]
        kf = kalman_filter(model, view(base, 1:kd), view(r, 1:kd), view(x, 1:kd))
        hyps = [_BranchHistory(kf.μf[kd], T(kf.loglik), nob, zero(T))]
        P = kf.Pf[kd]
        k0 = kd + 1
    else
        hyps = [_BranchHistory(Vec(start[1]), zero(T), nob, zero(T))]
        P = SMatrix{D, D, T}(start[2])
        k0 = firstindex(θ)
    end
    μH = zero(SMatrix{D, NH, T})
    C = zero(SMatrix{NH, NH, T})
    determined = _mean_determined(C)
    Ci = _mean_inverse(C, determined)
    # `(parent, branch)` of each history after each filtered sample `steps[s]`.
    trail = Vector{Vector{NTuple{2, Int}}}()
    steps = Int[]
    children, links = _BranchHistory{D, NH, T}[], NTuple{2, Int}[]
    workspace = (; scores = Float64[], order = Int[], kept = Int[], head = Dict{Int, Int}(), next = Int[])
    step = nothing
    for k in k0:lastindex(θ)
        if k > firstindex(θ)
            step = _transition(model, x[k] - x[k - 1], T, step)
            _, A, Q = step
            for (i, h) in pairs(hyps)
                hyps[i] = _BranchHistory(A * h.μ, h.ll, h.b, h.extra)
            end
            μH = A * μH
            P = _symmetric(A * P * A' + Q)
        end
        (isfinite(θ[k]) && isfinite(r[k]) && r[k] > 0) || continue
        θk, rk = T(θ[k]), T(r[k])
        S = P[1, 1] + rk
        S > 0 || _throw_precision_loss(model, "innovation variance", S, k)
        K = P[:, 1] / S
        vH = NH > 0 ? SVector{NH, T}(mean[k]) - μH[1, :] : zero(SVector{NH, T})
        # Until the samples determine the mean the prediction is flat along what
        # they leave free, which absorbs whichever branch the sample takes.
        sd = sqrt(determined ? S + dot(vH, Ci * vH) : S)
        span = determined ? Int(min(ceil(nsig * sd / 2π), maxhyp)) : 0
        empty!(children)
        empty!(links)
        for (i, h) in pairs(hyps)
            center = h.μ[1] + dot(vH, Ci * h.b)
            m0 = round(Int, (center - θk) / 2π)
            for m in (m0 - span):(m0 + span)
                y = θk + 2π * m
                (m == m0 || abs(y - center) <= nsig * sd) || continue
                v = y - h.μ[1]
                push!(children, _BranchHistory(h.μ + K * v, h.ll - (log(2π * S) + v^2 / S) / 2, h.b + v * vH / S, h.extra))
                push!(links, (i, m))
            end
        end
        if NH > 0
            μH += K * vH'
            C += vH * vH' / S
            determined = _mean_determined(C)
            Ci = _mean_inverse(C, determined)
        end
        IK = I - K * e1'
        P = _symmetric(IK * P * IK' + K * rk * K')
        kept = _prune_histories!(workspace, children, Ci, merge_tol .* sqrt.(diag(C)), P; prune, maxhyp, merge_tol)
        hyps = children[kept]
        push!(trail, links[kept])
        push!(steps, k)
    end
    norm = NH > 0 && determined ? -logdet(C) / 2 : zero(T)
    scores = _branch_score.(hyps, Ref(Ci)) .+ norm
    loglik = _logsumexp(scores)
    order = sortperm(scores; rev = true)
    histories = [(; lift = _branch_lift(base, θ, trail, steps, i), weight = exp(scores[i] - loglik)) for i in order]
    best = scores[first(order)] - hyps[first(order)].extra
    return (; histories, loglik, best)
end


# The indices of `children` to carry on, most probable first: within `prune` of
# the best, at most `maxhyp`, each history that predicts the same as a more
# probable one, its state within `merge_tol` of its standard deviations and its
# mean sums within `btol`, merged into the most probable such. Kept histories
# are binned by their first state in units of its merge tolerance, so a match
# lies in a history's own bin or a neighbor; `head` and `next` chain each bin's
# members.
function _prune_histories!(ws, children, Ci, btol, P; prune, maxhyp, merge_tol)
    (; scores, order, kept, head, next) = ws
    resize!(scores, length(children))
    map!(h -> _branch_score(h, Ci), scores, children)
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
                if (iszero(j) || m < j) && all(abs.(h.μ - o.μ) .<= tol) && all(abs.(h.b - o.b) .<= btol)
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
            children[kept[j]] = _BranchHistory(g.μ, g.ll, g.b, g.extra + log1p(exp(scores[i] - _branch_score(g, Ci))))
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
# on its most probable branches (the tracks sharing a mean then aligned by
# `_align_branches!`), plus the log of the total weight of the histories
# `_branch_filter` carried relative to that one's. Without a mean that is the
# tracks' marginal likelihood summed over their branches; with one each track's
# sum integrates its own coefficients.
function _branch_pooled_loglik(models, θs, rs, xs, mean)
    design = isnothing(mean) || !(eltype(mean) <: Integer) ? mean : _level_design(mean, θs)
    lifts = similar(θs)
    extra = zero(eltype(eltype(θs)))
    for i in eachindex(θs, rs, xs)
        rows = isnothing(design) ? nothing : design[i].rows
        bf = _branch_filter(_track_model(models, i), θs[i], rs[i], xs[i]; mean = rows)
        lifts[i] = first(bf.histories).lift
        extra += bf.loglik - bf.best
    end
    isnothing(design) || _align_branches!(lifts, [inv.(r) for r in rs], design)
    return _pooled_loglik(models, lifts, rs, xs, design) + extra
end

# Shift each phase piece by the multiple of 2π that brings its weighted mean
# nearest the mean its design (`_level_design`) predicts there from the pieces
# before it, fit by weighted least squares, so that a mean shared across
# spectral windows sees one branch. A piece whose prediction the pieces before
# it do not determine keeps its branch.
function _align_branches!(ys, ws, design)
    T = eltype(eltype(ys))
    p = _mean_count(design)
    A = zeros(T, p, p)
    z = zeros(T, p)
    for j in eachindex(ys, ws, design)
        ok = [k for k in eachindex(ys[j], ws[j]) if _shape_usable(ys[j][k], ws[j][k])]
        isempty(ok) && continue
        (; cols, rows) = design[j]
        sw = sum(k -> ws[j][k], ok)
        G = zeros(T, p)
        G[cols] = sum(k -> ws[j][k] * rows[k], ok) / sw
        Ai = pinv(A)
        if norm(G - A * (Ai * G)) <= sqrt(eps(T)) * norm(G)
            m = sum(k -> ws[j][k] * ys[j][k], ok) / sw
            ys[j] .-= 2π * round((m - dot(G, Ai * z)) / 2π)
        end
        for k in ok
            h = rows[k]
            A[cols, cols] .+= ws[j][k] * h * h'
            z[cols] .+= ws[j][k] * ys[j][k] * h
        end
    end
    return ys
end
