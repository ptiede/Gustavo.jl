# ── Per-scan terms beside the joint bandpass ─────────────────────────────────
#
# A station's model may add components with `Ti = PerScan()` beside its
# bandpass shape, e.g. a phase level and a delay per group of channels that
# change from scan to scan. `JointSmoother` reaches the joint maximum by
# alternation: the shape is solved on the data with the per-scan terms divided
# out, then the per-scan terms scan by scan given the shape, each baseline's
# source term solved beside them, until no parameter moves by more than
# `tolerance` of its standard error. A per-scan term is evaluated at its scan's
# first sample and fit to the scan's averaged visibilities.
#
# The terms enter through the model layer. Every term's phase or log-amplitude
# is linear in its parameters, so a parameter's contribution at unit value (its
# basis, from its component's own `term_eval`) is its derivative, and the
# station gain on a scan is the shape's gain times exp(Σ x_k b_k).
#
# Two shifts leave every visibility unchanged. A phase or log-amplitude common
# to every channel of one station and scan is absorbed by each baseline's source
# term; as on the shape's tracks, one parameter that carries it is held at
# zero. The average over the scans of one shape segment is absorbed by the
# shape; each parameter is held to an information-weighted mean of zero over
# those scans, so the shape is the time-averaged bandpass.

const _MAX_SCAN_ROUNDS = 25

# One parameter of a per-scan component: its position in θ, station,
# observable, the scan (position in the solve's scan groups) it serves, 0 for
# none, and its `family`, the same parameter in every scan.
struct ScanParameter
    index::Int
    ant::Int
    phase::Bool
    scan::Int
    family::Int
end

# Every per-scan parameter and its basis over (channel, feed) at the first
# sample of the scan it serves; `nothing` without per-scan components.
function _scan_parameters(setup, θ, results)
    paths = vcat(setup.paths.phase.scan, setup.paths.logamp.scan)
    isempty(paths) && return nothing
    params = ScanParameter[]
    bases = Matrix{Float64}[]
    families = Dict{Any, Int}()
    unit = zero(θ)
    for path in paths, b in antenna_blocks(setup.layout, θ, path...)
        _add_scan_parameters!(params, bases, families, b.plan, size(b.θ), b.antennas, path, results, unit, setup.geom)
    end
    return (; params, bases)
end

function _add_scan_parameters!(params, bases, families, plan, leaf_size, antennas, path, results, unit, geom)
    lin = LinearIndices(leaf_size)
    scan_of = Dict{Int, Int}()
    for (i, res) in pairs(results)
        ts = plan.tseg_id[res.ti]
        haskey(scan_of, ts) && throw(
            ArgumentError(
                "scan $(geom.scan_names[geom.scan_of_time[res.ti]]) is split across scan groups; " *
                    "per-scan terms need one group per scan",
            ),
        )
        scan_of[ts] = i
    end
    nchan, nfeed = length(geom.channel_freqs), geom.nfeed
    for (ai, a) in pairs(antennas), I in CartesianIndices(leaf_size[1:4])
        p, node, fs, ts = Tuple(I)
        index = plan.range[lin[p, node, fs, ts, ai]]
        i = get(scan_of, ts, 0)
        B = zeros(nchan, nfeed)
        if i > 0
            unit[index] = 1
            for f in axes(B, 2), c in axes(B, 1)
                B[c, f] = Calibration._node_value(plan, unit, ai, f, results[i].ti, c)
            end
            unit[index] = 0
        end
        family = get!(families, (path, p, node, fs, a), length(families) + 1)
        push!(params, ScanParameter(index, a, first(path) === :phase, i, family))
        push!(bases, B)
    end
    return nothing
end

# Station `a`'s log-gain from the per-scan parameters `ks` at θ, over
# (channel, feed), added into `Z`.
function _add_scan_loggain!(Z, terms, ks, θ)
    for k in ks
        p = terms.params[k]
        x = θ[p.index]
        iszero(x) && continue
        s = p.phase ? x * im : complex(x)
        Bk = terms.bases[k]
        for f in axes(Bk, 2), c in axes(Bk, 1)
            Z[c, p.ant, f] += s * Bk[c, f]
        end
    end
    return Z
end

# The (pair, feed pair, channel) order of a scan's accumulator.
_cell_order(A) = PermutedDimsArray(parent(A), dimnum(A, (AntennaPair, FeedPair, Frequency)))

# A scan's accumulators corrected by the gains exp(Z) over (channel, station,
# feed): each visibility divided by its baseline's gain, its weight scaled by
# |gain|².
function _correct_scan(res, Z, slot)
    rl, wl = copy(res.rl), copy(res.wl)
    R, W = _cell_order(rl), _cell_order(wl)
    fps = lookup(dims(rl, FeedPair))
    for (j, (sa, sb)) in pairs(lookup(dims(rl, AntennaPair)))
        a, b = slot[sa], slot[sb]
        for (q, (fa, fb)) in pairs(fps), c in axes(R, 3)
            g = exp(Z[c, a, fa] + conj(Z[c, b, fb]))
            R[j, q, c] *= conj(g)
            W[j, q, c] *= abs2(g)
        end
    end
    return (; res..., rl, wl)
end

# Each scan's accumulators with the per-scan terms divided out.
function _remove_scan_terms(results, θ, terms, serving, geom)
    slot = Dict(n => i for (i, n) in pairs(geom.antennas))
    Z = zeros(ComplexF64, length(geom.channel_freqs), length(geom.antennas), geom.nfeed)
    return map(eachindex(results)) do i
        fill!(Z, 0)
        _correct_scan(results[i], _add_scan_loggain!(Z, terms, serving[i], θ), slot)
    end
end

# The cells of one scan that a per-scan fit acts on: each visibility `v`, its
# weight `w`, the shape's gain product `h`, its source term `src`, and, in
# compressed rows `ptr`, the fitted parameters `k` acting on it with the
# derivative `coef` of its log-model and the basis value `β` on the station's
# own end.
struct ScanCells
    v::Vector{ComplexF64}
    w::Vector{Float64}
    h::Vector{ComplexF64}
    src::Vector{Int}
    ptr::Vector{Int}
    k::Vector{Int}
    coef::Vector{ComplexF64}
    β::Vector{Float64}
    station::Vector{Int}
    feed::Vector{Int}
end

function _scan_cells(res, G0, terms, ks, slot)
    R, W = _cell_order(res.rl), _cell_order(res.wl)
    fps = lookup(dims(res.rl, FeedPair))
    cells = ScanCells(ComplexF64[], Float64[], ComplexF64[], Int[], [1], Int[], ComplexF64[], Float64[], Int[], Int[])
    stations = unique(terms.params[k].ant for k in ks)
    nsrc = 0
    anchored = false
    for (j, (sa, sb)) in pairs(lookup(dims(res.rl, AntennaPair)))
        a, b = slot[sa], slot[sb]
        (a in stations || b in stations) || continue
        for (q, (fa, fb)) in pairs(fps)
            src = 0
            for c in axes(R, 3)
                w = Float64(W[j, q, c])
                r = ComplexF64(R[j, q, c])
                (w > 0 && isfinite(w) && isfinite(r)) || continue
                iszero(src) && (src = nsrc += 1)
                anchored |= !(a in stations && b in stations)
                push!(cells.v, r / w)
                push!(cells.w, w)
                push!(cells.h, G0[c, a, fa] * conj(G0[c, b, fb]))
                push!(cells.src, src)
                for (local_k, k) in pairs(ks)
                    p = terms.params[k]
                    p.ant == a || p.ant == b || continue
                    f = p.ant == a ? fa : fb
                    β = terms.bases[k][c, f]
                    iszero(β) && continue
                    push!(cells.k, local_k)
                    push!(cells.coef, p.phase ? (p.ant == a ? im * β : -im * β) : complex(β))
                    push!(cells.β, β)
                    push!(cells.station, p.ant)
                    push!(cells.feed, f)
                end
                push!(cells.ptr, length(cells.k) + 1)
            end
        end
    end
    return cells, nsrc, anchored
end

# The log of cell `n`'s model gain over the shape's: Σ x_k coef_k over its
# fitted parameters (`active[k]`, 0 for a held one).
@inline function _cell_exponent(cells, n, x, active)
    z = zero(ComplexF64)
    for e in cells.ptr[n]:(cells.ptr[n + 1] - 1)
        j = active[cells.k[e]]
        iszero(j) || (z += x[j] * cells.coef[e])
    end
    return z
end

function _scan_chi2(cells, x, S, active)
    χ2 = 0.0
    for n in eachindex(cells.v)
        m = S[cells.src[n]] * cells.h[n] * exp(_cell_exponent(cells, n, x, active))
        χ2 += cells.w[n] * abs2(cells.v[n] - m)
    end
    return χ2
end

# The normal equations of the scan's χ² at (x, S): `A` and `g` over the fitted
# parameters, and per source term its 2×2 block `Ass`, its cross terms `Axs`
# with the parameters and its gradient `gs` (real and imaginary parts).
function _scan_normal!(A, g, Axs, Ass, gs, cells, x, S, active)
    fill!(A, 0); fill!(g, 0); fill!(Axs, 0); fill!(Ass, 0); fill!(gs, 0)
    for n in eachindex(cells.v)
        s, w = cells.src[n], cells.w[n]
        u = cells.h[n] * exp(_cell_exponent(cells, n, x, active))
        m = S[s] * u
        r = cells.v[n] - m
        dS = (u, im * u)
        for a in 1:2
            gs[a, s] -= w * real(conj(dS[a]) * r)
            for b in 1:2
                Ass[a, b, s] += w * real(conj(dS[a]) * dS[b])
            end
        end
        for e in cells.ptr[n]:(cells.ptr[n + 1] - 1)
            i = active[cells.k[e]]
            iszero(i) && continue
            di = cells.coef[e] * m
            g[i] -= w * real(conj(di) * r)
            for a in 1:2
                Axs[i, a, s] += w * real(conj(di) * dS[a])
            end
            for f in cells.ptr[n]:(cells.ptr[n + 1] - 1)
                j = active[cells.k[f]]
                iszero(j) || (A[i, j] += w * real(conj(di) * cells.coef[f] * m))
            end
        end
    end
    return A
end

# The Levenberg–Marquardt system with the source terms eliminated: each source
# term touches only its own baseline's cells, so its 2×2 block is solved out
# (a Schur complement), leaving `Ared`, `gred` over the per-scan parameters
# alone; undamped when `λ = 0`.
function _scan_reduce!(Ared, gred, T, A, g, Axs, Ass, gs, λ)
    Ared .= A
    for i in axes(A, 1)
        Ared[i, i] *= 1 + λ
    end
    gred .= g
    for s in axes(Ass, 3)
        Binv = _source_block_inverse(Ass, s, λ)
        C = view(Axs, :, :, s)
        mul!(T, C, Binv)
        mul!(Ared, T, C', -1, 1)
        mul!(gred, T, SVector(gs[1, s], gs[2, s]), -1, 1)
    end
    return Ared
end

# The inverse of source term `s`'s damped 2×2 block; zero for a term no cell
# constrains.
function _source_block_inverse(Ass, s, λ)
    B = SMatrix{2, 2}(Ass[1, 1, s] * (1 + λ), Ass[2, 1, s], Ass[1, 2, s], Ass[2, 2, s] * (1 + λ))
    return det(B) > 0 ? inv(B) : zero(B)
end

# The damped step `δx`, `δS`; false where the reduced system is not positive
# definite.
function _scan_step!(δx, δS, Ared, gred, T, A, g, Axs, Ass, gs, λ)
    _scan_reduce!(Ared, gred, T, A, g, Axs, Ass, gs, λ)
    F = cholesky!(Symmetric(Ared); check = false)
    issuccess(F) || return false
    ldiv!(δx, F, gred)
    δx .*= -1
    for s in axes(Ass, 3)
        c = SVector(dot(view(Axs, :, 1, s), δx), dot(view(Axs, :, 2, s), δx))
        d = -(_source_block_inverse(Ass, s, λ) * (SVector(gs[1, s], gs[2, s]) + c))
        δS[s] = complex(d[1], d[2])
    end
    return true
end

# Hold, among station `a`'s parameters `js` of one observable, enough to fix
# the combinations the source terms absorb: those whose action on the
# station's cells is constant on each of its feeds. Each feed pair has its own
# source term, so a per-feed constant is free. The parameters carrying the most
# of these combinations are held, preferring the families `held` names from
# earlier scans, which it then records.
function _hold_source_shifts!(hold, held, cells, terms, ks, js, a, phase)
    rows = Dict{Int, Int}()
    feed_of = Int[]
    entries = Tuple{Int, Int, Float64}[]
    for n in eachindex(cells.v), e in cells.ptr[n]:(cells.ptr[n + 1] - 1)
        cells.station[e] == a || continue
        col = findfirst(==(cells.k[e]), js)
        isnothing(col) && continue
        r = get!(rows, n) do
            push!(feed_of, cells.feed[e])
            length(rows) + 1
        end
        push!(entries, (r, col, cells.β[e]))
    end
    B = zeros(length(rows), length(js))
    for (r, col, β) in entries
        B[r, col] = β
    end
    feeds = unique(feed_of)
    E = [Float64(feed_of[r] == f) for r in eachindex(feed_of), f in feeds]
    N = nullspace(B - E * (E \ B))
    size(N, 2) == 0 && return hold
    score = [norm(view(N, col, :)) for col in eachindex(js)]
    family(col) = terms.params[ks[js[col]]].family
    previous = get(held, (a, phase), Int[])
    order = vcat(
        [col for col in eachindex(js) if family(col) in previous && score[col] >= 0.5 * maximum(score)],
        qr(permutedims(N), ColumnNorm()).p,
    )
    chosen = Int[]
    for col in order
        length(chosen) == size(N, 2) && break
        col in chosen && continue
        rank(N[vcat(chosen, col), :]) > length(chosen) && push!(chosen, col)
    end
    held[(a, phase)] = map(family, chosen)
    hold[js[chosen]] .= true
    return hold
end

# Fit the per-scan parameters `ks` serving one scan given the shape's gains
# `G0` (channel, station, feed), each (pair, feed pair) they act on taking a
# free source term, by Levenberg–Marquardt from θ's current values. Writes the
# values into θ and their information, the source terms marginalized, into
# `info`. `held` keeps, per station and observable, the families holding the
# shifts the source terms absorb.
function _fit_scan_terms!(θ, info, held, terms, ks, res, G0, geom; max_iterations)
    slot = Dict(n => i for (i, n) in pairs(geom.antennas))
    for k in ks
        info[terms.params[k].index] = 0
    end
    cells, nsrc, anchored = _scan_cells(res, G0, terms, ks, slot)
    isempty(cells.v) && return nothing
    anchored || throw(
        ArgumentError(
            "scan $(geom.scan_names[geom.scan_of_time[res.ti]]): every station observing it carries " *
                "per-scan terms, so a change common to all of them is free; leave at least one " *
                "station, such as the reference, without them",
        ),
    )
    stations = unique(terms.params[k].ant for k in ks)
    acting = falses(length(ks))
    acting[cells.k] .= true

    hold = falses(length(ks))
    for a in stations, phase in (true, false)
        js = [j for j in eachindex(ks) if acting[j] && terms.params[ks[j]].ant == a && terms.params[ks[j]].phase == phase]
        isempty(js) || _hold_source_shifts!(hold, held, cells, terms, ks, js, a, phase)
    end
    active = zeros(Int, length(ks))
    np = 0
    for j in eachindex(ks)
        (acting[j] && !hold[j]) && (active[j] = np += 1)
    end
    for j in eachindex(ks)
        hold[j] && (θ[terms.params[ks[j]].index] = 0)
    end
    x = zeros(np)
    for j in eachindex(ks)
        iszero(active[j]) || (x[active[j]] = θ[terms.params[ks[j]].index])
    end

    # Each source term at its optimum for x.
    S = zeros(ComplexF64, nsrc)
    den = zeros(nsrc)
    for n in eachindex(cells.v)
        u = cells.h[n] * exp(_cell_exponent(cells, n, x, active))
        S[cells.src[n]] += cells.w[n] * conj(u) * cells.v[n]
        den[cells.src[n]] += cells.w[n] * abs2(u)
    end
    S .= ifelse.(den .> 0, S ./ den, 0)

    A, g = zeros(np, np), zeros(np)
    Axs, Ass, gs = zeros(np, 2, nsrc), zeros(2, 2, nsrc), zeros(2, nsrc)
    δx, δS = zeros(np), zeros(ComplexF64, nsrc)
    xt, St = similar(x), similar(S)
    Ared, gred, T = zeros(np, np), zeros(np), zeros(np, 2)
    λ = 1.0e-3
    χ2 = _scan_chi2(cells, x, S, active)
    _scan_normal!(A, g, Axs, Ass, gs, cells, x, S, active)
    for _ in 1:max_iterations
        if !_scan_step!(δx, δS, Ared, gred, T, A, g, Axs, Ass, gs, λ)
            λ *= 4
            continue
        end
        xt .= x .+ δx
        St .= S .+ δS
        χ2t = _scan_chi2(cells, xt, St, active)
        if χ2t < χ2
            done = χ2 - χ2t < 1.0e-10 * χ2
            x .= xt
            S .= St
            χ2 = χ2t
            _scan_normal!(A, g, Axs, Ass, gs, cells, x, S, active)
            λ = max(λ / 3, 1.0e-12)
            done && break
        else
            λ *= 4
            λ > 1.0e12 && break
        end
    end
    _scan_reduce!(Ared, gred, T, A, g, Axs, Ass, gs, 0.0)
    for j in eachindex(ks)
        i = active[j]
        iszero(i) && continue
        θ[terms.params[ks[j]].index] = x[i]
        info[terms.params[ks[j]].index] = Ared[i, i]
    end
    return nothing
end

# Hold each per-scan parameter to an information-weighted mean of zero over
# the scans of its station's shape segment.
function _center_scan_terms!(θ, info, terms, tseg)
    families = Dict{Tuple{Int, Int}, Vector{Int}}()
    for p in terms.params
        iszero(p.scan) && continue
        push!(get!(families, (p.family, tseg[p.ant, p.scan]), Int[]), p.index)
    end
    for idx in values(families)
        s = sum(j -> info[j], idx)
        s > 0 || continue
        x̄ = sum(j -> info[j] * θ[j], idx) / s
        for j in idx
            info[j] > 0 && (θ[j] -= x̄)
        end
    end
    return θ
end

# Alternate the shape solve (`solve_shape`, on accumulators with the per-scan
# terms divided out) and the per-scan fit given the shape until no per-scan
# parameter moves by `tolerance` of its standard error; returns the last shape
# solve's report.
function _solve_with_scan_terms!(solve_shape, terms, θ, results, setup, tseg; tolerance, max_iterations)
    geom, layout = setup.geom, setup.layout
    idx = [p.index for p in terms.params]
    θ[idx] .= 0
    info = zero(θ)
    held = Dict{Tuple{Int, Bool}, Vector{Int}}()
    serving = [Int[] for _ in results]
    for (k, p) in pairs(terms.params)
        iszero(p.scan) || push!(serving[p.scan], k)
    end
    chans = collect(eachindex(geom.channel_freqs))
    for round in 1:_MAX_SCAN_ROUNDS
        report = solve_shape(_remove_scan_terms(results, θ, terms, serving, geom))
        previous = θ[idx]
        shape = copy(θ)
        shape[idx] .= 0
        for (i, res) in pairs(results)
            isempty(serving[i]) && continue
            G0 = view(evaluate_gains(layout, shape, chans, [res.ti]), :, 1, :, :)
            _fit_scan_terms!(θ, info, held, terms, serving[i], res, G0, geom; max_iterations)
        end
        _center_scan_terms!(θ, info, terms, tseg)
        change = maximum(j -> abs(θ[idx[j]] - previous[j]) * sqrt(info[idx[j]]), eachindex(idx); init = 0.0)
        @debug "per-scan terms" round change
        change < tolerance && return report
        round == _MAX_SCAN_ROUNDS && @warn "Bandpass: the per-scan terms did not settle within $_MAX_SCAN_ROUNDS rounds of the shape solve; the result depends on the starting point." change
        round == _MAX_SCAN_ROUNDS && return report
    end
    return
end
