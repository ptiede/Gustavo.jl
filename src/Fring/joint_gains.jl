# ── The joint solve over components' parameters ─────────────────────────────
#
# The unknowns are the parameters of the fitted components, each component
# under its own prior; the data reach them through the station gains
# (`StationGains`), `x = J·θ`. A state-space prior is a chain over its
# component's parameters along its axis, per station, feed node, segment of the
# other axis and parameter; an `IIDPrior` is a term `θ²/2σ²` per parameter.
# Every fitted component carries a prior whose curvature is positive in every
# direction, so the posterior has one mode. The source term `S` per (scan,
# station pair, feed pair) is profiled out as in the bandpass engine
# (bandpass_map.jl). The phase the data leave free on each connected group of
# station gains is fixed after the fit by the gauge (`_meet_phase_gauge!`).

"""
    solve_joint_gains!(components, data, geom::DataGeometry; cover, gauge, kw...) -> NamedTuple

Fit the parameters of the fitted `components` (`(; phase, logamp)`, each a
vector of components, each the station blocks [`antenna_blocks`](@ref)
returns) to the weighted sums `data = (; r, w)`: `r = Σ w·v` and `w = Σ w`
over `(Scan, AntennaPair, FeedPair, cell)`, entry `(si, c)` covering the time
and channel `cover[si, c] = (t, ch)` of the plans. Writes the parameters into
the blocks' `θ`.

Every station's prior on every component must be an `IIDPrior` or a
state-space prior along one axis whose start is proper (`OUPrior`,
`CARMAPrior`, or `RandomWalkPrior` with an `init`); a state-space prior must
act on parameters that each enter their station gains with coefficient one,
and at most one component per (station, observable) carries one along each
axis.

The model is `V = g_a·S·conj(g_b)` with `g = exp(ℓ + iφ)` the station gains and
`S` per scan, station pair and feed pair profiled out. The start estimates the
station gains alone (rounds of each gain's estimate given the others), puts
each phase chain on its most probable 2π branches (a sample that several
station gains share, as a per-scan component splits them, at their mean), and
solves for the
parameters nearest those estimates under the priors. The MAP is found by
Levenberg–Marquardt with the damping scaled by each parameter's curvature, and
hyperparameters given as densities at their type-II MAP, by `hyper_optimizer`
(an Optimization.jl optimizer that takes a gradient and bounds; L-BFGS-B by
default) on their Laplace evidence within their support (`_fit_hypers!`),
stopping once the evidence a Newton step would still gain is below
`tolerance²/2` or after `max_hyper_iterations` iterations. The gauge then
moves the phases along the freedoms the data leave, at least cost in prior
energy (`_meet_phase_gauge!`).

Returns `(; phase, amp, gains, converged, stalled)`: per observable one
`(; path, block, station, node, across, axis, windows, prior, pieces, values)`
per chain: its component's path, station block and position there, feed node
and segment of the other axis, its axis, the piece (spectral window or scan)
of each of its `pieces` of samples, its prior with the hyperparameters
resolved, and its samples' station-gain values; `gains`, the `StationGains`
`sg` and every station gain's final `phase` and `amp` values; and how the MAP
solve ended. A station-gain value is `NaN` where the gain has no data. A chain
whose station gains carry no data, and any other parameter that reads none, is
not solved and keeps its value.
"""
function solve_joint_gains!(
        components, data, geom::DataGeometry;
        cover, gauge::AbstractGauge, max_iterations::Integer = 200, max_hyper_iterations::Integer = 100,
        tolerance::Real = 0.01, linsolve = LinearSolve.KrylovJL_CG(; linesearch = true), hyper_optimizer = LBFGSB(m = 20),
        executor = SerialScheduler(),
    )
    (; ctx, priors, x, solved, sg, parts, hasdata, weight, observables) =
        _joint_setup(components, data, geom; cover, max_iterations, tolerance, linsolve, executor)
    (; cells, fitted, tracks, params) = ctx
    hyper_converged, hyper_remaining = true, 0.0
    if !isempty(_estimated(ctx))
        hyper = _fit_hypers!(priors, x, ctx, solved; tolerance, max_iterations, max_hyper_iterations, linsolve, optimizer = hyper_optimizer)
        solved, hyper_converged, hyper_remaining = hyper.solved, hyper.converged, hyper.remaining
    end
    (; converged, stalled, change) = solved
    stalled && @warn "solve_joint_gains! stopped at a step that could not lower its MAP objective at any " *
        "damping; that step would have moved a gain by $change of its standard error."
    stalled || converged || @warn "solve_joint_gains! did not converge in $max_iterations iterations: the " *
        "last step moved a gain by $change of its standard error, against a tolerance of $tolerance."
    hyper_converged || @warn "the hyperparameter fit of solve_joint_gains! stopped at $max_hyper_iterations " *
        "iterations with $hyper_remaining nats of log evidence left to gain, against $(tolerance^2 / 2)."

    fitted.amp && _warn_free_amplitudes(cells, sg, geom)

    fitted.phase && _meet_phase_gauge!(params.phase, tracks.phase, priors.phase, gauge, cells, sg, components.phase)
    for (obs, o) in pairs(observables)
        fitted[obs] && _write_params!(components[o], sg.params[o][parts[obs].kept], params[obs].θ)
    end
    x = map(p -> [hasdata[n] ? v : NaN for (n, v) in pairs(p.J * p.θ)], params)
    fits = map(parts, priors, x, (; phase = true, amp = false)) do part, priors, x, phase
        xs, _ = _with_merged(x, weight, part.merged; phase)
        map((t, prior) -> (; t.chain..., prior, t.pieces, values = xs[t.gains]), part.gain_tracks, priors)
    end
    return (; fits.phase, fits.amp, gains = (; sg, x.phase, x.amp), converged, stalled)
end

# The structure of `solve_joint_gains!` up to its hyperparameter fit: the
# station gains and cells, the chains, the start (each gain's estimate, the
# branches, the seeded priors and the parameters nearest the estimates under
# them), the solve's fixed structure `ctx` and buffers, and the MAP at the
# seeded priors, `x` its station-gain values and `solved` its report.
function _joint_setup(components, data, geom::DataGeometry; cover, max_iterations, tolerance, linsolve, executor)
    (; r, w) = data
    sg = StationGains(components, length(geom.antennas), geom.nfeed, cover)
    ngains = length(sg.at)
    edge, gain_pairs = _gain_edges(r, sg.index, geom.antennas)
    cells = BandpassCells(r, w, edge, gain_pairs; nchunks = max_tasks(executor))
    hasdata = falses(ngains)
    foreach(k -> hasdata[cells.na[k]] = hasdata[cells.nb[k]] = true, eachindex(cells.na))
    any(hasdata) || throw(ArgumentError("solve_joint_gains!: no station gain has data to fit."))
    observables = (; phase = :phase, amp = :logamp)
    parts = map(o -> _parameter_chains(components[o], sg, geom, o, hasdata), observables)
    fitted = map(p -> !isempty(p.kept), parts)
    tracks = map(p -> p.tracks, parts)
    params = map(observables, parts) do o, p
        kept = sg.params[o][p.kept]
        θ = Float64[v for v in _param_values(components[o], kept)]
        shared = [!(components[o][k][j].plan.tseg isa PerScan) for (k, j, _) in kept]
        (; J = sg.J[o][:, p.kept], θ, p.key, p.ridge, shared)
    end
    weight = _gain_weights(cells, ngains)

    φ, ℓ = zeros(ngains), zeros(ngains)
    npairs = length(gain_pairs)
    accumulators = (;
        linearize = ChunkAccumulators(cells, zeros(npairs), zeros(ComplexF64, ngains), zeros(ngains)),
        phase = ChunkAccumulators(cells, zeros(ngains)),
        amp = ChunkAccumulators(cells, zeros(ngains)),
    )
    buffers = (;
        g = zeros(ComplexF64, ngains), trial = zeros(ComplexF64, ngains), S = zeros(ComplexF64, length(cells.ranges)),
        lin = (; q = zeros(npairs), c = zeros(ComplexF64, ngains), W = zeros(ngains)),
        b = (; phase = zeros(ngains), amp = zeros(ngains)),
        profile = (; phase = (zeros(ngains), zeros(ngains)), amp = (zeros(ngains), zeros(ngains))),
    )
    gains_of(ℓ, φ) = _gains!(buffers.g, (; amp = ℓ, phase = φ))
    function linearized(ℓ, φ)
        g = gains_of(ℓ, φ)
        S = _source!(buffers.S, g, cells, executor)
        return _linearize(g, S, cells, ngains, npairs, executor, accumulators.linearize, buffers.lin)
    end
    # Each station gain starts at its own estimate given the others and `S`,
    # repeated with `S` re-solved until a round lowers χ² by less than 10⁻⁵ of
    # itself; every second round takes the mean of the last two, which keeps the
    # rounds from oscillating (Salvini & Wijnholds 2014). `S` absorbs a constant
    # on any track, so each round takes the mean out of each log-amplitude chain.
    χ2 = _profiled_chi2(gains_of(ℓ, φ), cells, executor)
    for round in 1:max_iterations
        lin = linearized(ℓ, φ)
        ℓ0, φ0 = copy(ℓ), copy(φ)
        for (obs, x, x0) in ((:phase, φ, φ0), (:amp, ℓ, ℓ0))
            fitted[obs] || continue
            y, _ = _gain_estimates(x0, lin, obs; phase_held = !fitted.phase)
            for n in eachindex(x, y)
                (hasdata[n] && isfinite(y[n])) || continue
                x[n] = iseven(round) ? (x0[n] + y[n]) / 2 : y[n]
            end
        end
        for t in parts.amp.gain_tracks
            moved = filter(n -> hasdata[n], t.members)
            isempty(moved) || (ℓ[moved] .-= sum(view(ℓ, moved)) / length(moved))
        end
        χ2, χ2_before = _profiled_chi2(gains_of(ℓ, φ), cells, executor), χ2
        χ2_before - χ2 < 1.0e-5 * χ2 && break
    end
    if fitted.phase
        _align_feed_delays!(φ, ℓ, cells, [n[2] for n in sg.at], [geom.channel_freqs[n[3][2]] for n in sg.at])
        compid, _, _ = connected_components(ngains, zip(cells.na, cells.nb))
        _align_component_phases!(φ, compid, hasdata, weight, [(; gains = t.members) for t in parts.phase.gain_tracks])
    end

    # Each chain's prior with its hyperparameters resolved, seeded by type-II
    # MAP on the chain alone from every station gain's own estimate, and each
    # phase chain put on the 2π branches most probable under it. A sample that
    # is the value of several station gains takes their mean, and each of
    # them the branch nearest the sample's.
    lin = linearized(ℓ, φ)
    estimates = (;
        phase = _gain_estimates(φ, lin, :phase),
        amp = _gain_estimates(ℓ, lin, :amp; phase_held = !fitted.phase),
    )
    samples = map(obs -> _with_merged(estimates[obs]..., parts[obs].merged; phase = obs === :phase), (; phase = :phase, amp = :amp))
    priors = map(observables) do o
        obs = o === :logamp ? :amp : :phase
        ys, ws = samples[obs]
        [_seed_prior(t, ys, ws; phase = obs === :phase) for t in parts[obs].gain_tracks]
    end
    if fitted.phase
        y, wy = samples.phase
        (; merged) = parts.phase
        φs, _ = _with_merged(φ, estimates.phase[2], merged; phase = true)
        free = [hasdata; [any(n -> hasdata[n], ns) for ns in merged]]
        for (t, prior) in zip(parts.phase.gain_tracks, priors.phase)
            _choose_branches!(φs, t, prior, y, wy, free)
        end
        copyto!(φ, view(φs, 1:ngains))
        for (v, ns) in pairs(merged), n in ns
            (hasdata[n] && isfinite(φs[ngains + v])) && (φ[n] += 2π * round((φs[ngains + v] - φ[n]) / 2π))
        end
    end
    for (obs, x̂) in ((:phase, φ), (:amp, ℓ))
        fitted[obs] || continue
        _, wy = estimates[obs]
        params[obs].θ .= _start_parameters(params[obs], tracks[obs], priors[obs], x̂, wy)
    end

    Hφ, hpos = _curvature_pattern(gain_pairs, ngains)
    ctx = (;
        cells, ngains, npairs, accumulators, fitted, tracks, Hφ, Hℓ = copy(Hφ), hpos, executor, systems = Dict{Symbol, Any}(), params, buffers,
        krylov = (; phase = Ref{Any}(nothing), amp = Ref{Any}(nothing), joint = Ref{Any}(nothing)),
    )
    x = map(p -> p.J * p.θ, params)
    solved = _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve)
    return (; ctx, priors, x, solved, sg, parts, hasdata, weight, observables)
end

# `y` with weights `w` over the station gains, followed by one entry per
# sample of `merged`: the weighted mean of its members' finite values, circular
# for a `phase`, with their summed weight (`NaN` and zero without any).
function _with_merged(y, w, merged; phase::Bool)
    ys, ws = collect(Float64, y), collect(Float64, w)
    for ns in merged
        ok = [n for n in ns if w[n] > 0 && isfinite(y[n])]
        W = sum(n -> w[n], ok; init = 0.0)
        m = iszero(W) ? NaN : sum(n -> w[n] * y[n], ok) / W
        (phase && !iszero(W)) && (m += angle(sum(n -> w[n] * cis(y[n] - m), ok)))
        push!(ys, m)
        push!(ws, W)
    end
    return ys, ws
end

# Warns of each connected group of station gains whose correlations hold no
# odd loop (two stations, a chain, an even ring): the data measure only the
# sums `ℓ_a + ℓ_b` there, so they leave the two sides' log-amplitudes free to
# shift oppositely, and only the priors fix the split.
function _warn_free_amplitudes(cells, sg, geom::DataGeometry)
    graph = Graphs.SimpleGraph(length(sg.at))
    foreach(k -> Graphs.add_edge!(graph, cells.na[k], cells.nb[k]), eachindex(cells.na))
    groups = [c for c in Graphs.connected_components(graph) if length(c) > 1 && Graphs.is_bipartite(first(Graphs.induced_subgraph(graph, c)))]
    isempty(groups) && return nothing
    names = sort!(unique(geom.antennas[sg.at[n][1]] for n in first(groups)))
    @warn "the data leave $(length(groups)) groups of station gains, e.g. those of antennas " *
        "$(join(names, ", ")), joined only by baselines with no odd loop (two antennas, a chain or " *
        "an even ring): they measure the sum of two sides' log-amplitudes but not how it divides " *
        "between them, and only the amplitude priors set the split."
    return nothing
end

# The pair of station gains each correlation of `r` (`(Scan, AntennaPair,
# FeedPair, cell)`) joins, numbered in `edge` (`0` for an autocorrelation or a
# station no component reaches) and listed in `gain_pairs`, from the station
# gains `index[a, p, si, c]`.
function _gain_edges(r, index, antennas)
    station = Dict(n => i for (i, n) in pairs(antennas))
    ends = [(station[a], station[b]) for (a, b) in lookup(DimensionalData.dims(r, AntennaPair))]
    feeds = lookup(DimensionalData.dims(r, FeedPair))
    edge = zeros(Int32, size(r))
    numbered = Dict{Tuple{Int, Int}, Int32}()
    gain_pairs = Tuple{Int, Int}[]
    for I in CartesianIndices(edge)
        si, bi, p, c = Tuple(I)
        (a, b), (fa, fb) = ends[bi], feeds[p]
        a == b && continue
        pair = (index[a, fa, si, c], index[b, fb, si, c])
        any(iszero, pair) && continue
        edge[I] = get!(numbered, pair) do
            push!(gain_pairs, pair)
            Int32(length(gain_pairs))
        end
    end
    return edge, gain_pairs
end

# One observable's chains and IID terms over the parameters of its
# `sg.params` that are solved, `kept`: those of a chain whose station gains
# carry data anywhere, and the others that read a station gain with data.
# Over the kept parameters, in their order:
# `tracks`, one per chain in the engine's form (`_track_chains!`), its samples
# the chain's unknowns and `constants` the group of each piece's constant the
# source term absorbs (one per chain along frequency, since the source term is
# frequency-flat; one per scan along time); `gain_tracks`, the same chains with
# their samples the station gains whose values they are, for the start, the
# seeding and the branch choice, and `members` those station gains in sample
# order; `merged`, the station gains of each sample that is the value of
# several, sample `ngains + v` taking `merged[v]`; `ridge`, `j => 1/σ²` for
# each parameter under an `IIDPrior`; and `key`, each unknown's coordinate
# along its chain (`Inf` off a chain), by which the system is ordered for
# factoring.
function _parameter_chains(components, sg::StationGains, geom, observable, hasdata)
    params, J = sg.params[observable], sg.J[observable]
    ngains = length(sg.at)
    ridge = Pair{Int, Float64}[]
    key = fill(Inf, length(params))
    groups = OrderedDict{NTuple{6, Int}, Vector{Int}}()
    axis_of = Dict{Tuple{Int, Symbol}, Int}()
    coordinates = Dict{Tuple{Int, Int, Symbol}, @NamedTuple{x::Vector{Float64}, piece::Vector{Int}}}()
    segments(k, j, axis) = get!(() -> _segment_coordinates(components[k][j].plan, geom, axis), coordinates, (k, j, axis))
    for (u, (k, j, i)) in pairs(params)
        b = components[k][j]
        q, node, fs, ts, ai = Tuple(CartesianIndices(b.θ)[i])
        prior = b.plan.priors[ai]
        prior isa IIDPrior && (push!(ridge, u => inv(Float64(prior.σ)^2)); continue)
        axis = _chain_axis(prior, b, ai, observable)
        owner = get!(axis_of, (b.antennas[ai], axis), k)
        owner == k || throw(
            ArgumentError(
                "solve_joint_gains!: station $(b.antennas[ai]) carries state-space priors along $axis on two " *
                    "$observable components, $(join(components[owner][1].plan.path, '.')) and " *
                    "$(join(b.plan.path, '.')); at most one component per station and axis takes one.",
            ),
        )
        along, across = axis === :Frequency ? (fs, ts) : (ts, fs)
        push!(get!(Vector{Int}, groups, (k, j, ai, node, across, q)), u)
        key[u] = segments(k, j, axis).x[along]
    end
    reads_data(u) = any(r -> hasdata[rowvals(J)[r]], nzrange(J, u))
    solved = falses(length(params))
    foreach(((u, _),) -> solved[u] = reads_data(u), ridge)
    for us in values(groups)
        any(reads_data, us) && (solved[us] .= true)
    end
    kept = findall(solved)
    position = zeros(Int, length(params))
    position[kept] = eachindex(kept)
    tracks = NamedTuple[]
    gain_tracks = NamedTuple[]
    merged = Vector{Int}[]
    for ((k, j, ai, node, across, _), us) in groups
        solved[first(us)] || continue
        b = components[k][j]
        prior = only(values(b.plan.priors[ai]))
        axis = only(keys(b.plan.priors[ai]))
        seg = segments(k, j, axis)
        along(u) = (I = CartesianIndices(b.θ)[params[u][3]]; axis === :Frequency ? I[3] : I[4])
        sort!(us; by = along)
        x = [seg.x[along(u)] for u in us]
        piece = [seg.piece[along(u)] for u in us]
        windows = unique(piece)
        pieces = [findall(==(p), piece) for p in windows]
        chain = (; b.plan.path, block = j, station = ai, node, across, axis, windows)
        constants = axis === :Frequency ? ones(Int, length(pieces)) : collect(eachindex(pieces))
        t = (; gains = position[us], x, pieces, constants, prior, chain)
        push!(tracks, t)
        values_of = [_value_gains(J, u, b, observable) for u in us]
        gains = map(values_of) do ns
            length(ns) == 1 && return only(ns)
            push!(merged, ns)
            return ngains + length(merged)
        end
        push!(gain_tracks, merge(t, (; gains, members = reduce(vcat, values_of))))
    end
    ridge = [position[u] => w for (u, w) in ridge if solved[u]]
    return (; tracks, gain_tracks, merged, ridge, key = key[kept], kept)
end

# The axis of a state-space prior on station `ai` of block `b`, which must be
# proper; a station with no prior, or one of another kind, is an error.
function _chain_axis(prior, b, ai, observable)
    name = "$observable component $(join(b.plan.path, '.')) of station $(b.antennas[ai])"
    isnothing(prior) && throw(
        ArgumentError(
            "solve_joint_gains!: the $name has no prior. Every fitted component takes a prior: an " *
                "`IIDPrior`, or a state-space prior along one axis whose start is proper.",
        ),
    )
    prior isa NamedTuple && length(prior) == 1 && _proper_prior(only(values(prior))) || throw(
        ArgumentError(
            "solve_joint_gains!: the $name has prior $prior. A fitted component takes an `IIDPrior` or " *
                "one state-space prior along one axis whose start is proper: an `OUPrior`, a `CARMAPrior`, " *
                "or a `RandomWalkPrior` with an `init`.",
        ),
    )
    return only(keys(prior))
end

# The station gains whose value parameter `u` is: its column of `J` must hold
# only ones.
function _value_gains(J, u, b, observable)
    rng = nzrange(J, u)
    all(isone, view(nonzeros(J), rng)) || throw(
        ArgumentError(
            "solve_joint_gains!: a state-space prior on the $observable component $(join(b.plan.path, '.')) " *
                "acts on parameters that do not each enter their station gains with coefficient one; its " *
                "chains are seeded and put on 2π branches from the station gains, which needs that.",
        ),
    )
    return rowvals(J)[rng]
end

# Each segment of `plan` along `axis`: its coordinate `x`, the mean frequency
# or time of its channels or times, and its `piece`, which a chain never
# crosses: its spectral window along frequency, its scan along time.
function _segment_coordinates(plan, geom::DataGeometry, axis::Symbol)
    if axis === :Frequency
        seg = _shape_segments(plan, geom)
        return (; seg.x, piece = seg.piece_of)
    end
    members = segment_groups(plan.tseg_id, maximum(plan.tseg_id; init = 0))
    x = [isempty(m) ? NaN : sum(view(geom.times, m)) / length(m) for m in members]
    piece = [isempty(m) ? 0 : geom.scan_of_time[first(m)] for m in members]
    return (; x, piece)
end

# The parameters nearest the station-gain estimates `x̂` (weights `w`, zero
# where an estimate is not finite) under the priors: the minimum of
# `½Σ w·(J·θ − x̂)²` plus the prior energy, one sparse solve.
function _start_parameters(p, tracks, priors, x̂, w)
    known = isfinite.(x̂) .& (w .> 0)
    wk = ifelse.(known, w, 0.0)
    sys = _posterior_system(spdiagm(wk), wk .* ifelse.(known, x̂, 0.0), p.J, zeros(length(p.θ)), tracks, p.key; p.ridge)
    K, h, _ = _reduce!(sys, _assemble!(sys.layout, priors), 0.0)
    return (_factor(K, sys.key) \ h)[eachindex(p.θ)]
end

"""
    _meet_phase_gauge!(p, tracks, priors, gauge, cells, sg, components)

Move the phase parameters `p.θ` onto `gauge` along the freedom the data leave
to the components that span scans (`p.shared`, every component but the
`PerScan` ones). Their part of a station gain is its shared value; station
gains with the same shared parameters share one. On each connected group `g`
of shared values joined by weighted correlations, a phase `c_g` added to every
one of them cancels from every correlation. The move `δ`, on the shared
parameters only, is the one of least prior energy at `θ + δ` among those that
change each shared value by its group's `c_g`, `J_s·δ = Σ_g c_g·1_g`, and meet
the gauge's constraints on the shared values, `C·J_s·(θ + δ) = d`
([`gauge_constraints`](@ref), one freedom per group). A group joined to a
station gain without a shared part has no such freedom and keeps its values.
The per-scan components and station gains without data are left to the
priors. The gauge's constraints must be reachable by such a move; a model that
cannot shift a group's phase is an error.
"""
function _meet_phase_gauge!(p, tracks, priors, gauge, cells, sg, components)
    ngains = length(sg.at)
    nθ = length(p.θ)
    gainw = _gain_weights(cells, ngains)
    # The distinct shared parts, numbered over the station gains with data.
    Js = dropzeros!(p.J * spdiagm(Float64.(p.shared)))
    rows = sparse(Js')
    value = zeros(Int, ngains)
    reps = Int[]
    numbered = Dict{Tuple{Vector{Int}, Vector{Float64}}, Int}()
    for n in 1:ngains
        rng = nzrange(rows, n)
        (gainw[n] > 0 && !isempty(rng)) || continue
        value[n] = get!(numbered, (rowvals(rows)[rng], nonzeros(rows)[rng])) do
            push!(reps, n)
            length(reps)
        end
    end
    nv = length(reps)
    edges = [(value[a], value[b]) for (a, b) in zip(cells.na, cells.nb) if !iszero(value[a]) && !iszero(value[b])]
    compid, ncomp, _ = connected_components(nv, edges)
    # A value correlated only with gains outside the shared parts is a component of its own.
    foreach(m -> iszero(compid[m]) && (compid[m] = ncomp += 1), 1:nv)
    blocked = falses(ncomp)
    for (a, b) in zip(cells.na, cells.nb)
        iszero(value[a]) == iszero(value[b]) && continue
        blocked[compid[max(value[a], value[b])]] = true
    end
    free = findall(!, blocked)
    isempty(free) && return p
    group = zeros(Int, ncomp)
    group[free] = eachindex(free)
    nodew = zeros(nv)
    foreach(n -> iszero(value[n]) || (nodew[value[n]] += gainw[n]), 1:ngains)
    members = [Int[] for _ in free]
    foreach(m -> iszero(group[compid[m]]) || push!(members[group[compid[m]]], m), 1:nv)
    paths = _station_paths(components, size(sg.index, 1))
    at = sg.at[reps]
    freedoms = map(members) do m
        Calibration._common_choice(gauge, unique(p for i in m for p in paths[at[i][1]]))
        GaugeFreedom(;
            nodes = m, antenna = [at[i][1] for i in m], feed = [at[i][2] for i in m], scan = zeros(Int, length(m)),
            component = [first(paths[at[i][1]]) for i in m], observable = fill(:phase, length(m)),
            direction = ones(length(m)), weight = nodew[m],
        )
    end
    C, d = _bandpass_gauge_system(gauge, GaugeFreedoms{Float64}(freedoms, nv))
    Jv = Js[reps, :]
    CJ = sparse(C) * Jv
    # The prior's quadratic over the parameters and the chains' latent states.
    sys = _posterior_system(spzeros(ngains, ngains), zeros(ngains), p.J, p.θ, tracks, p.key; p.ridge)
    _assemble!(sys.layout, priors)
    K = sys.layout.K
    u = _current_unknowns(sys)
    nu, nf = length(u), length(free)
    E = sparse([m for m in 1:nv if !iszero(group[compid[m]])], [group[compid[m]] for m in 1:nv if !iszero(group[compid[m]])], 1.0, nv, nf)
    held = findall(!, p.shared)
    A = [
        Jv spzeros(nv, nu - nθ) -E
        sparse(eachindex(held), held, 1.0, length(held), nu) spzeros(length(held), nf)
        CJ spzeros(size(CJ, 1), nu - nθ + nf)
    ]
    a = [zeros(nv + length(held)); d .- CJ * p.θ]
    # Scaled to unit curvature in each unknown and unit largest entry in each
    # constraint: the priors' curvatures differ by many orders of magnitude
    # with the parameters' units.
    D = [inv(sqrt(K[j, j])) for j in 1:nu]
    AD = A * blockdiag(spdiagm(D), sparse(1.0I, nf, nf))
    R = inv.(vec(maximum(abs, AD; dims = 2)))
    any(!isfinite, R) && error("solve_joint_gains!: a gauge constraint reads no phase parameter.")
    Ã = spdiagm(R) * AD
    nc = size(A, 1)
    KKT = [
        blockdiag(spdiagm(D) * K * spdiagm(D), spzeros(nf, nf)) Ã'
        Ã spzeros(nc, nc)
    ]
    rhs = [-(D .* (K * u .- sys.layout.h)); zeros(nf); R .* a]
    # The constraints outnumber the freedoms they leave whenever shared values
    # repeat a parameter combination, so `KKT` is singular. Its regularized
    # form is not, and refining against `KKT` converges to its solution when
    # the constraints are consistent (the proximal method of multipliers).
    γ = 1.0e-12
    F = ldlt(Symmetric(KKT + blockdiag(spzeros(nu, nu), γ * sparse(1.0I, nf, nf), -γ * sparse(1.0I, nc, nc))))
    sol = F \ rhs
    r = rhs .- KKT * sol
    for _ in 1:100
        trial = sol .+ F \ r
        rt = rhs .- KKT * trial
        maximum(abs, rt) < maximum(abs, r) || break
        sol, r = trial, rt
    end
    step = [D .* view(sol, 1:nu); view(sol, (nu + 1):(nu + nf))]
    RA, b = spdiagm(R) * A, R .* a
    tol = sqrt(eps()) * max(maximum(abs, b; init = 0.0), 1.0)
    _meet_constraints!(step, RA, b, γ, tol)
    miss = maximum(abs, RA * step .- b; init = 0.0)
    miss <= tol || error(
        "solve_joint_gains!: the gauge's constraints cannot be met by a phase shift the data do not see " *
            "(residual $miss): the components that span scans cannot add a constant phase to every station " *
            "gain of a connected group.",
    )
    p.θ .+= view(step, 1:nθ)
    return p
end

# Move `s` onto `M s = b`, when it misses by more than `tol`, by the least change
# in units that give each unknown a largest coefficient of 1 in `M`. A stiff
# prior leaves the KKT step short of the constraints by the rounding of its
# large multipliers, and a constraint may move a stiff unknown, so these units
# hold no prior curvature.
function _meet_constraints!(s, M, b, γ, tol)
    r = b .- M * s
    maximum(abs, r; init = 0.0) <= tol && return s
    S = [iszero(c) ? 0.0 : inv(c) for c in vec(maximum(abs, M; dims = 1))]
    B = M * spdiagm(S)
    n, m = size(B, 2), size(B, 1)
    P = ldlt(Symmetric([sparse(1.0I, n, n) B'; B -γ * sparse(1.0I, m, m)]))
    for _ in 1:10
        s .+= S .* view(P \ [zeros(n); r], 1:n)
        r = b .- M * s
        maximum(abs, r) <= tol && break
    end
    return s
end

# The paths of the `components` that reach each of `nant` stations, in order:
# the components a gauge applies to (`ByComponent`).
function _station_paths(components, nant)
    paths = [Any[] for _ in 1:nant]
    for c in components, b in c, a in b.antennas
        b.plan.path in paths[a] || push!(paths[a], b.plan.path)
    end
    return paths
end
