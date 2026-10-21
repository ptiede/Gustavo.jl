# ── The solve ────────────────────────────────────────────────────────────────

# The fixed structure of the joint solve: `edge[si, bi, p, cell]` numbers the
# `(slot, slot)` pair a correlation joins, `0` for none, and `slot_pairs` lists
# them, the slots numbered by `ids` (`_gain_slot_ids`).
function _joint_gain_layout(data, layout, ids)
    (; r, ends) = data
    (; loc, tseg, fseg) = layout
    edge = zeros(Int32, size(r))
    index = Dict{Tuple{Int, Int}, Int32}()
    slot_pairs = Tuple{Int, Int}[]
    for cell in axes(r, Frequency), p in axes(r, FeedPair), bi in axes(r, AntennaPair)
        _solvable(ends[bi, p]) || continue
        (a, fa), (b, fb) = ends[bi, p]
        (ka, ia), (kb, ib) = loc[a], loc[b]
        for si in axes(r, Scan)
            ta, tb = tseg[a, si], tseg[b, si]
            (iszero(ta) || iszero(tb)) && continue
            key = (ids[ka][ia, fa, fseg[a, cell], ta], ids[kb][ib, fb, fseg[b, cell], tb])
            edge[si, bi, p, cell] = get!(index, key) do
                push!(slot_pairs, key)
                Int32(length(slot_pairs))
            end
        end
    end
    return (; edge, slot_pairs)
end

# Every (station, feed, time segment) track of one observable: its slots in
# frequency-segment order and its station's coordinates, pieces, prior and
# levels (`fits[a]`), with `active` marking the pieces its prior acts on — those
# with data or fixed values at enough slots (`known`) to determine its states.
function _bandpass_tracks(fits, observable, ids, loc, present, nfeed, known)
    antennas = [a for a in eachindex(fits, loc) if !isnothing(fits[a]) && !isnothing(fits[a][observable])]
    return [_bandpass_track(fits[a][observable], a, loc[a], feed, ts, ids, known) for a in antennas for feed in 1:nfeed for ts in present[a]]
end

function _bandpass_track(f, a, (k, ai), feed, ts, ids, known)
    slots = vec(ids[k][ai, feed, :, ts])
    active = [!isnothing(f.prior) && count(n -> known[n], slots[p]) >= _samples_needed(f.prior) for p in f.pieces]
    return (; a, k, ai, feed, ts, slots, f.x, f.pieces, f.prior, f.level, f.nlevel, active)
end

# Whether `x′` differs from `x` by no more than `x`'s own rounding.
_within_rounding(x′, x) = all(i -> abs(x′[i] - x[i]) <= eps(abs(x[i])), eachindex(x′, x))

# The largest change of any slot's log-gain, `(Δℓ, Δφ)` with the phase change
# wrapped, in standard errors of its weight `W` from the data, and that slot.
function _largest_weighted_change(Δℓ, Δφ, W)
    worst, at = 0.0, 0
    for n in eachindex(Δℓ, Δφ, W)
        W[n] > 0 || continue
        c = sqrt(W[n]) * abs(complex(Δℓ[n], rem2pi(Δφ[n], RoundNearest)))
        c > worst && ((worst, at) = (c, n))
    end
    return worst, at
end

# `" (antenna, feed f, frequency segment s, time segment t)"` for gain `I` of block `k`.
function _slot_description(geom::DataGeometry, loc, k, I)
    ai, f, fs, ts = Tuple(I)
    ant = findfirst(==((k, ai)), loc)
    return " ($(geom.antennas[ant]), feed $f, frequency segment $fs, time segment $ts)"
end

# A joint level writer's values at time segment `ts`.
_level_at(::Nothing, ts) = nothing
_level_at(lw, ts) = merge(lw, (; values = view(lw.values, :, :, ts)))

# Gauge-fix each (station, feed, time segment) track — zero band-mean
# log-amplitude, circular-mean reference phase — and write it into the station
# block that carries it. Both gauges
# are over the frequency track of one time segment, so each of a station's
# segments is normalized on its own. Each observable is written, and referenced,
# over the slots it has a value in; the rest are left unwritten, holding whatever
# θ already had, which is unit gain, as is a station no block covers. With a level, the reference comes off the
# level and the shape is written as fit.
#
# `gains[k]` holds the stations of phase block `k` and amplitude block `k`, in
# the blocks' own order; `level_writers[a]` holds station `a`'s
# `(; phase, amp)` level writers (see `_joint_level_writer`), each `nothing` without
# a level.
#
# The band means are removed at θ's precision, which may exceed the gains': the
# gauge is a property of θ.
function _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout, level_writers)
    for (k, gk) in pairs(gains)
        pb, ab = get(phase_blocks, k, nothing), get(amp_blocks, k, nothing)
        block = something(pb, ab)
        T = eltype(block.θ)
        (; g, φ, phase_valid, amp_valid) = gk
        node(b, f) = isnothing(b) ? 0 : _feed_node(b.plan.tying, f)
        for (ai, a) in pairs(block.antennas), f in axes(g, Feed), ts in layout.present[a]
            pnode, anode = node(pb, f), node(ab, f)
            lw = map(l -> _level_at(l, ts), level_writers[a])
            pv = view(phase_valid, ai, f, :, ts)
            if !iszero(pnode) && any(pv)
                phase = fs -> T(φ[ai, f, fs, ts])
                mphase = angle(sum(cis(phase(fs)) for fs in axes(g, Frequency) if pv[fs]))
                Lp = _write_level!(lw.phase, f, ts, Lj -> rem2pi(T(Lj) - mphase, RoundNearest))
                for fs in axes(g, Frequency)
                    pv[fs] || continue
                    pb.θ[1, pnode, fs, ts, ai] = isnothing(Lp) ?
                        rem2pi(phase(fs) - mphase, RoundNearest) : phase(fs) - T(Lp[lw.phase.seg[fs]])
                end
            end
            av = view(amp_valid, ai, f, :, ts)
            if !iszero(anode) && any(av)
                logamp = fs -> log(T(abs(g[ai, f, fs, ts])))
                mlogamp = sum(logamp(fs) for fs in axes(g, Frequency) if av[fs]) / count(av)
                La = _write_level!(lw.amp, f, ts, Lj -> T(Lj) - mlogamp)
                for fs in axes(g, Frequency)
                    av[fs] || continue
                    ab.θ[1, anode, fs, ts, ai] = isnothing(La) ?
                        logamp(fs) - mlogamp : logamp(fs) - T(La[lw.amp.seg[fs]])
                end
            end
        end
    end
    return nothing
end

"""
    solve_joint_bandpass!(θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
                          phase_level_blocks = [], amp_level_blocks = [],
                          gauge, max_iterations = 200, tolerance = 0.01,
                          flat_span = 0.01, eltype = nothing, tseg = nothing) -> (; phase, amp)

Jointly solve the per-(station, feed, frequency segment) complex bandpass gain
and a per-scan, per-station-pair, per-feed-pair constant source coherence (see
the header comment of bandpass_data.jl for the model), then
gauge-fix each (station, feed, time segment) track and write the result into the
station blocks of the two observables (`_write_joint_bandpass!`).

`phase_blocks` and `amp_blocks` are `_bandpass_blocks(setup, θ, :phase)`
and `(…, :logamp)` — station blocks over the SAME `θ` this call is handed, since
each block's `θ` is a view into it — and `phase_level_blocks`/`amp_level_blocks`
the matching `_bandpass_level_blocks`, empty without a level. A
station-uniform model gives one block per component spanning every station;
where the model differs across stations, each block carries its own stations,
feed tying, segments and prior, and a station no block covers is left at unit
gain. Each track is fit under its own block's prior, which relates the values
within each spectral window, with the prior's hyperparameters re-estimated every
iteration.
A station's two shapes must hold the same stations on the same frequency and
time segmentations, which `validate_model(::JointSmoother, model)` enforces at
model-compile time. Either list may be
empty, which holds that observable at zero: unit amplitude, or zero phase and
no gauge.

`geom` supplies the stations the `AntennaPair` labels name (θ's stations, in
θ's order) and the channel frequencies the priors are fit along.

Scaling a set of stations by one phase leaves every visibility internal to that
set unchanged, so the gauge fixes one constant per such set. The sets are the
connected components of the (station, feed, frequency segment, time segment)
graph the correlations span, and `gauge` gives one constraint per component
([`_joint_bandpass_gauge!`](@ref)): a pin holds its node at zero, and any other
constraint is imposed exactly in every step. Where stations' frequency
segmentations differ, a pin can hold part of a track; under a phase prior the
rest of the track is then fit with the pinned segment held at zero.

`scans` is the per-scan `(rl, wl)` accumulator pairs from
`accumulate_bandpass` — not summed across scans, since the source term
needs each scan's own coherent visibility.

`tseg`, when given, is the `(antenna, scan)` time-segment table
([`_antenna_time_segments`](@ref)): station `a`'s gain is solved separately for
each distinct `tseg[a, :]` value, in that station's own segment numbering, and a
station whose entry is `0` is left out of the solve. Every station shares one
segment when it is omitted. `scans` must hold the scans `tseg`'s columns
describe, in the same order. The phase gauge acts on the graph these scans
span, one constraint per connected component, so a break at one station is
measured against the stations that hold one gain across it rather than gauged
away. Solved phases are comparable only WITHIN a component: where that graph
splits — disjoint sub-arrays, or every station segmented at the same epoch —
each piece carries its own arbitrary constant, and a per-station change read
across the split is that constant plus the change.

The solve maximizes the posterior of the gains: half the χ² of the visibilities,
with `S` at its closed-form optimum for the gains, plus each track's prior energy
with its levels profiled out. `S` absorbs any constant on a (station, feed, time
segment) track, which the data therefore cannot measure, so a proper prior is
profiled over that constant as well. Each iteration:

1. linearizes the χ² in every slot's log-amplitude and phase about the current
   gains, with `S` held;
2. moves phase slots by multiples of 2π: on the first iteration each slot to the
   branch nearest the track fitted to the most probable branches of every slot's
   own estimate given all the other gains and `S` (`_choose_branches!`); after
   that, each piece so that every difference along it lies in (−π, π], kept only
   where this lowers the prior energy (`_wrap_differences!`), so that, as the data
   are unchanged by 2π, every move lowers the posterior and none is undone;
3. fits the hyperparameters given as densities, all tracks together, by type-II
   MAP on the marginal likelihood of the linearized posterior (`_fit_hypers!`),
   as closely as the last step moved the gains; an observable whose last fit
   took no step is not refit until its gains have moved by that fit's
   tolerance, and is refit before the solve can stop as converged;
4. takes one Levenberg–Marquardt step of the posterior, solved exactly as one
   sparse system over the slots, the priors' latent states and levels, with the
   gauge solved for one slot per constraint (`_map_step`), and accepts it once
   the posterior with `S` re-solved is no lower.

The solve starts from every slot's own estimate given unit gains elsewhere, and
the hyperparameters from type-II MAP on each track alone. Convergence is judged
on the largest change the undamped step makes to any (station, feed, segment)
gain's log, in standard errors of that gain from the data, so a gain the data
barely constrain counts for as little as its precision. The solve stops once
that is below `tolerance` with the hyperparameters fit to `tolerance` of their
standard errors; on a plateau, once an accepted step lowers the negative log
posterior by less than `tolerance²/2`, the decrease of a step of `tolerance`
standard errors, with the hyperparameters fit; or once no damping lowers the
posterior, with a warning if that step would have moved a gain by `tolerance`
or more: the posterior is then flat or not convex there, typically because a
prior too stiff for the data drives an amplitude toward zero, where the mode
does not exist. docs/src/priors.md ("The bandpass posterior") derives each
step. The scans handed to one call are a connected piece of the coupling
graph, so this is one criterion over one coupled problem: nodes that share no
data are solved by separate calls rather than being averaged into a common
tolerance.

Returns, per observable, one `(; track, prior, status)` per (station, feed, time
segment) track: the track, the prior it was fit under with its hyperparameters
resolved, and the [`TrackStatus`](@ref) of each spectral window. The data are reduced
and the gains and source terms held in the accumulators' types, or in the real
type `eltype`; the steps are solved in `Float64`.
"""
function solve_joint_bandpass!(
        θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
        phase_level_blocks = NamedTuple[], amp_level_blocks = NamedTuple[],
        gauge::AbstractGauge, max_iterations::Integer = 200, tolerance::Real = 0.01,
        flat_span::Real = 0.01, eltype = nothing,
        tseg::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
    )
    isempty(scans) && return (; phase = [], amp = [])
    nant = length(geom.antennas)

    # The data are reduced onto the refinement of every block's frequency
    # segmentation, and each station's gain is held in its own segments — one
    # gain over however many refinement cells that segment spans.
    blocks = _gain_blocks(phase_blocks, amp_blocks)
    fseg, cells = _antenna_freq_segments(blocks, nant)
    r, w = _reduce_all_scans(scans, cells; eltype)
    ends = _cell_nodes(r, geom.antennas, PerFeed())
    S = zeros(Base.eltype(r), (Scan(axes(r, Scan)), dims(r, AntennaPair), dims(r, FeedPair)))
    data = DimStack((; r, w, S, ends))

    # The time segments each station is actually solved for here, in its own
    # segmentation's numbering — the numbering of its block's `Ti` axis.
    tsg = isnothing(tseg) ? ones(Int, nant, length(scans)) : tseg
    loc = _block_locations(blocks, nant)
    layout = (;
        loc, tseg = tsg, fseg,
        present = [sort!(filter!(!iszero, unique(view(tsg, a, :)))) for a in axes(tsg, 1)],
    )
    gains = [_block_gains(b, geom, Base.eltype(r)) for b in blocks]
    ids = _gain_slot_ids(gains)
    jl = _joint_gain_layout(data, layout, ids)
    # The gauge fixes phases; a solve that holds every phase at zero has none to fix.
    gauge_state = isempty(phase_blocks) ? nothing : _joint_bandpass_gauge!(gains, data, layout, blocks, gauge)
    gauge_rows = isnothing(gauge_state) ? nothing : (; gauge_state.rows, d = Float64.(gauge_state.d))

    segs = [_shape_segments(b.plan, geom) for b in blocks]
    phase_level = _antenna_levels(phase_level_blocks, segs, loc, geom, θ, real(Base.eltype(r)))
    amp_level = _antenna_levels(amp_level_blocks, segs, loc, geom, θ, real(Base.eltype(r)))
    fits = map(1:nant) do a
        k, ai = loc[a]
        iszero(k) && return nothing
        seg = segs[k]
        track(blocks, level) = isempty(blocks) ? nothing : (;
            seg.x, seg.pieces, prior = _prior_along(blocks[k].plan.priors[ai], :Frequency),
            level = isnothing(level) ? nothing : level.level, nlevel = isnothing(level) ? 0 : level.nlevel,
        )
        return (; phase = track(phase_blocks, phase_level[a]), amp = track(amp_blocks, amp_level[a]))
    end

    slot = [(k, I) for (k, idk) in pairs(ids) for I in CartesianIndices(idk)]
    nslots = length(slot)
    pinned = [gains[k].pinned[I] for (k, I) in slot]
    hasdata = falses(nslots)
    measured = falses(length(jl.slot_pairs))
    for I in CartesianIndices(jl.edge)
        e = jl.edge[I]
        (iszero(e) || !(w[I] > 0)) && continue
        measured[e] = true
        foreach(n -> hasdata[n] = true, jl.slot_pairs[e])
    end
    any(hasdata) || throw(ArgumentError("solve_joint_bandpass!: no gain slot has data to fit."))
    fitted = (; phase = !isempty(phase_blocks), amp = !isempty(amp_blocks))
    known = (; phase = hasdata .| pinned, amp = hasdata)
    tracks = map((; phase = :phase, amp = :amp)) do obs
        fitted[obs] ? _bandpass_tracks(fits, obs, ids, loc, layout.present, geom.nfeed, known[obs]) : NamedTuple[]
    end
    # A slot moves where it has data, or where a prior acting on its track fills it.
    function movable(obs)
        out = fitted[obs] .& hasdata
        for t in tracks[obs], (j, p) in pairs(t.pieces)
            t.active[j] && (out[t.slots[p]] .= fitted[obs])
        end
        return out
    end
    free = (; phase = movable(:phase) .& .!pinned, amp = movable(:amp))
    filled = map(obs -> movable(obs) .& .!hasdata, (; phase = :phase, amp = :amp))
    acting = map(ts -> findall(t -> any(t.active), ts), tracks)
    track_layouts = map(ts -> map(_track_layout, ts), tracks)

    # Float64 whatever the data's type: CHOLMOD factors only Float64.
    ℓ, φ = zeros(nslots), zeros(nslots)
    gains_of(ℓ, φ) = exp.(complex.(ℓ, φ))
    _source_coherence!(S, gains_of(ℓ, φ), data, jl)
    # Start every slot at its own estimate given unit gains elsewhere, a nonlinear
    # step the linearized iterations then refine.
    lin = _linearize(gains_of(ℓ, φ), S, data, jl, nslots)
    for (obs, x) in ((:phase, φ), (:amp, ℓ))
        fitted[obs] || continue
        y, _ = _slot_estimates(x, lin, obs; phase_held = !fitted.phase)
        for n in eachindex(x)
            free[obs][n] && isfinite(y[n]) && (x[n] = y[n])
        end
    end
    _source_coherence!(S, gains_of(ℓ, φ), data, jl)

    # Each acting track's prior with its hyperparameters resolved, seeded by
    # type-II MAP on the track alone from every slot's own estimate.
    lin = _linearize(gains_of(ℓ, φ), S, data, jl, nslots)
    estimates = (;
        phase = fitted.phase ? _slot_estimates(φ, lin, :phase) : nothing,
        amp = fitted.amp ? _slot_estimates(ℓ, lin, :amp; phase_held = !fitted.phase) : nothing,
    )
    priors = map((; phase = :phase, amp = :amp)) do obs
        isempty(acting[obs]) && return []
        y, wy = estimates[obs]
        return [_seed_prior(tracks[obs][i], y, wy; phase = obs === :phase) for i in acting[obs]]
    end
    energy(obs, x) = sum((first(_track_energy(track_layouts[obs][i], priors[obs][j], x)) for (j, i) in pairs(acting[obs])); init = 0.0)
    objective(ℓ, φ, S) = _chi2(gains_of(ℓ, φ), S, data, jl) / 2 + energy(:phase, φ) + energy(:amp, ℓ)
    key = zeros(nslots)
    for ts in tracks, t in ts
        key[t.slots] .= t.x
    end
    curvature = map(ts -> Vector{Union{Nothing, @NamedTuple{inverse::Matrix{Float64}, measured::Bool}}}(nothing, length(ts)), tracks)
    Hφ, hpos = _curvature_pattern(jl.slot_pairs, nslots)
    Hℓ = copy(Hφ)
    change, worst, stalled, converged, hypers_fit = Inf, 0, false, false, true
    μ, ν, near, verify = 1.0e-3, 2.0, false, false
    hypers_converged = Dict(:phase => true, :amp => true)
    # Per observable, the gains and tolerance of its last hyperparameter fit,
    # kept while that fit took no step.
    settled = Dict{Symbol, @NamedTuple{x::Vector{Float64}, tolerance::Float64}}()
    phase_energy(i, j) = x -> first(_track_energy(track_layouts[:phase][i], priors.phase[j], x))
    kept_back = Set{Tuple{Int, Int}}()
    for iteration in 1:max_iterations
        lin = _linearize(gains_of(ℓ, φ), S, data, jl, nslots)
        if fitted.phase
            y, wy = iteration == 1 ? _slot_estimates(φ, lin, :phase) : (nothing, nothing)
            for (j, i) in pairs(acting.phase)
                shift = if iteration == 1
                    _choose_branches!(φ, tracks.phase[i], priors.phase[j], y, wy, free.phase)
                else
                    order = priors.phase[j] isa RandomWalkPrior ? priors.phase[j].order : 1
                    moved, pieces = _wrap_differences!(φ, tracks.phase[i], order, free.phase, phase_energy(i, j))
                    union!(kept_back, ((i, p) for p in pieces))
                    moved
                end
                isnothing(gauge_rows) || _shift_gauge!(gauge_rows, shift)
                any(!iszero, shift) && delete!(settled, :phase)
            end
        end
        # Levenberg–Marquardt: each observable's step is damped by `μ` times the
        # largest data curvature of its slots, a trust region in log-gain, where
        # the linearization's error is measured, and `μ` follows how well the
        # quadratic model predicted the objective's actual decrease (Madsen,
        # Nielsen & Tingleff 2004, §3.2). `μ` stays at least the curvature's own
        # precision, so a direction neither the data nor a prior determine takes
        # no step, and a slot with no curvature above that precision, from its data
        # or a prior, does not move.
        current = (; phase = φ, amp = ℓ)
        H = (; phase = _data_curvature!(Hφ, hpos, lin.q, -1), amp = _data_curvature!(Hℓ, hpos, lin.q, 1))
        b = (; phase = imag.(lin.c), amp = real.(lin.c))
        scale = map(h -> maximum(diag(h); init = 0.0), H)
        systems = map((; phase = :phase, amp = :amp)) do obs
            fitted[obs] || return nothing
            d = diag(H[obs])
            moving = free[obs] .& ((d .> eps() * scale[obs]) .| filled[obs])
            return _posterior_system(
                H[obs], b[obs], current[obs], moving, tracks[obs][acting[obs]], key;
                gauge = obs === :phase ? gauge_rows : nothing,
            )
        end
        # The hyperparameters' optimum moves with the linearization, so they are
        # fit only as closely as the gains last moved, and to `tolerance` once the
        # gains have converged. A fit that took no step is not repeated until the
        # gains have moved by its tolerance since, or a stop needs it verified.
        hyper_tolerance = max(tolerance, change)
        hypers_fit = true
        for obs in (:phase, :amp)
            fitted[obs] || continue
            xo = current[obs]
            previous = get(settled, obs, nothing)
            if !(near || verify) && !isnothing(previous)
                Δx = xo .- previous.x
                drift, _ = obs === :phase ? _largest_weighted_change(zero(Δx), Δx, lin.W) : _largest_weighted_change(Δx, zero(Δx), lin.W)
                drift < previous.tolerance && (hypers_fit = false; continue)
            end
            hyperfit = _fit_hypers!(
                priors[obs], view(curvature[obs], acting[obs]), systems[obs], tracks[obs][acting[obs]], eps() * scale[obs];
                tolerance = hyper_tolerance, max_iterations,
            )
            hypers_converged[obs] = hyperfit.converged
            estimating = any(i -> _hyper_count(tracks[obs][i].prior) > 0, acting[obs])
            estimating && iszero(hyperfit.steps) ? (settled[obs] = (; x = copy(xo), tolerance = hyper_tolerance)) : delete!(settled, obs)
        end
        hypers_fit &= all(values(hypers_converged))
        verify = false
        function propose(μ)
            steps = map((:phase, :amp)) do obs
                fitted[obs] || return current[obs], 0.0
                x′, _, model = _map_step(systems[obs], priors[obs]; λ = μ * scale[obs])
                return x′, model
            end
            return (; phase = steps[1][1], amp = steps[2][1]), steps[1][2] + steps[2][2]
        end
        # A damped step understates the change; converged once the undamped step
        # is within `tolerance` too.
        if near
            x1, _ = propose(eps())
            change, worst = _largest_weighted_change(x1.amp .- ℓ, x1.phase .- φ, lin.W)
            change < tolerance && hypers_fit && (converged = true; break)
        end
        O0 = objective(ℓ, φ, S)
        E0 = energy(:phase, φ) + energy(:amp, ℓ)
        accepted, decrease = false, Inf
        while isfinite(μ)
            x1, predicted = propose(μ)
            change, worst = _largest_weighted_change(x1.amp .- ℓ, x1.phase .- φ, lin.W)
            _within_rounding(x1.amp, ℓ) && _within_rounding(x1.phase, φ) && break
            S1 = _source_coherence!(similar(S), gains_of(x1.amp, x1.phase), data, jl)
            O1 = objective(x1.amp, x1.phase, S1)
            # The objective is known to its rounding.
            if O1 <= O0 + 16 * eps(abs(O0))
                ρ = (O0 - O1) / (E0 - predicted)
                decrease = O0 - O1
                ℓ, φ, S = x1.amp, x1.phase, S1
                μ = max(μ * max(1 / 3, 1 - (2ρ - 1)^3), eps())
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
        # A plateau: the step lowered the posterior by less than a step of
        # `tolerance` standard errors would (`tolerance²/2`).
        if decrease < tolerance^2 / 2
            hypers_fit && (converged = true; break)
            verify = true
        end
        near = change < tolerance
    end
    isempty(kept_back) || @warn "the joint bandpass solve kept the current 2π branches of $(length(kept_back)) phase " *
        "pieces where moving every difference to its nearest wrap would not have lowered the prior energy"
    free_amp = fitted.amp ? _split_free_amplitudes!(ℓ, jl.slot_pairs[measured], free.amp, tracks.amp) : NamedTuple[]
    if !isempty(free_amp)
        g = first(free_amp)
        names = unique(geom.antennas[findfirst(==((slot[n][1], slot[n][2][1])), loc)] for n in g.slots)
        @warn "the joint bandpass data leave $(sum(x -> x.ndir, free_amp)) frequency-dependent " *
            "log-amplitude directions undetermined in $(length(free_amp)) groups of gain slots, e.g. " *
            "antennas $(join(sort!(names), ", "))$(_slot_description(geom, loc, slot[first(g.slots)]...)): " *
            "each group is joined only by baselines with no odd loop (two antennas, a chain or an even " *
            "ring), which measure the sum of two sides' log-amplitudes but not how it divides between " *
            "them, and the amplitude prior, if any, does not fix it. The solve splits each group evenly " *
            "(the minimum-norm solution)."
    end
    location = iszero(worst) ? "" : _slot_description(geom, loc, slot[worst]...)
    stalled && @warn "the joint bandpass solve stopped at a step that could not lower its MAP " *
        "objective at any damping; that step would have moved a gain by $change of its standard " *
        "error$location. The objective is flat or not convex near this point, often because a prior " *
        "is too stiff for the data, which can drive a gain's amplitude toward zero."
    # Where a slot's phase is held more than π/2 from its own data — by a phase
    # prior or the gauge — the amplitude that fits it best is zero, and without an
    # amplitude prior the posterior has no mode: its log-amplitude runs off.
    if fitted.amp && (stalled || converged)
        final = _linearize(gains_of(ℓ, φ), S, data, jl, nslots)
        unpriored = free.amp .& .!filled.amp
        for t in tracks.amp, (j, p) in pairs(t.pieces)
            t.active[j] && (unpriored[t.slots[p]] .= false)
        end
        modeless = [n for n in findall(unpriored) if final.W[n] > 0 && real(1 + final.c[n] / final.W[n]) <= 0]
        isempty(modeless) || @warn "the joint bandpass posterior has no mode for the amplitude of " *
            "$(length(modeless)) gain slots, e.g.$(_slot_description(geom, loc, slot[first(modeless)]...)): " *
            "their phase is held more than π/2 from their data, by a phase prior or the gauge, so the " *
            "amplitude that fits them best is zero. Give the amplitude a prior."
    end
    all(values(hypers_converged)) || @warn "the joint bandpass hyperparameter fit did not converge in $max_iterations " *
        "steps at its last fit; raise `JointSmoother(; max_iterations)`"
    stalled || converged || @warn "the joint bandpass solve did not converge in " *
        "$max_iterations iterations: the last step moved a gain by $change of its standard error$location, " *
        "against a tolerance of $tolerance; raise `JointSmoother(; max_iterations)`"

    for (n, (k, I)) in pairs(slot)
        gk = gains[k]
        gk.g[I] = exp(complex(ℓ[n], φ[n]))
        gk.φ[I] = φ[n]
        gk.phase_valid[I] = free.phase[n] || (fitted.phase && pinned[n])
        gk.amp_valid[I] = free.amp[n]
    end
    levels = [(; phase = _level_values(phase_level[a]), amp = _level_values(amp_level[a])) for a in 1:nant]
    solved = (; phase = φ, amp = ℓ)
    results = map((:phase, :amp)) do obs
        x = solved[obs]
        valid = obs === :phase ? free.phase .| (fitted.phase .& pinned) : free.amp
        prior_of = zeros(Int, length(tracks[obs]))
        prior_of[acting[obs]] = eachindex(acting[obs])
        map(eachindex(tracks[obs])) do i
            t = tracks[obs][i]
            j = prior_of[i]
            status = map(t.pieces) do p
                all(pinned[t.slots[p]]) && obs === :phase && return TrackSolved
                return _band_track_status([valid[n] ? x[n] : NaN for n in t.slots[p]], flat_span)
            end
            if !isnothing(t.level) && any(t.active)
                _, L = _track_energy(track_layouts[obs][i], priors[obs][j], x)
                for (g, v) in L
                    levels[t.a][obs][t.feed, g, t.ts] = v
                end
            end
            return (; track = t, prior = iszero(j) ? t.prior : priors[obs][j], status)
        end
    end
    level_writers = [
        (; phase = _joint_level_writer(phase_level[a], levels[a].phase), amp = _joint_level_writer(amp_level[a], levels[a].amp))
            for a in 1:nant
    ]
    _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout, level_writers)
    return (; phase = results[1], amp = results[2])
end

# The log-amplitude directions neither the data nor the priors determine,
# beyond each track's band constant (which the per-scan source term absorbs and
# the write-out gauges), with `ℓ` moved to the minimum-norm point along them.
# They are the shifts `δ` with `δ_a + δ_b = 0` on every measured baseline, the
# data measuring only `ℓ_a + ℓ_b`, and, on each piece a prior acts on, `δ` in the
# prior's null space (`_prior_projectors`). Only a connected group of measured
# slots with no odd loop (two stations, a chain, an even ring) is bipartite and
# can shift, its two sides oppositely, besides the slots only a prior fills.
# Returns, per cluster of slots that kept free directions, its slots and how many
# it kept.
function _split_free_amplitudes!(ℓ, data_pairs, movable, tracks)
    groups = @NamedTuple{slots::Vector{Int}, ndir::Int}[]
    measured = Graphs.SimpleGraph(length(ℓ))
    for (a, b) in data_pairs
        movable[a] && movable[b] && Graphs.add_edge!(measured, a, b)
    end
    free = movable .& iszero.(Graphs.degree(measured))
    any_bipartite = false
    for c in Graphs.connected_components(measured)
        length(c) > 1 && Graphs.is_bipartite(first(Graphs.induced_subgraph(measured, c))) || continue
        free[c] .= true
        any_bipartite = true
    end
    any_bipartite || return groups

    # Clusters: free slots linked by a measured baseline or a shared track.
    track_slots = [[s for s in t.slots if movable[s]] for t in tracks]
    track_of = Dict{Int, Int}()
    links = copy(measured)
    for (ti, ts) in pairs(track_slots)
        foreach(s -> track_of[s] = ti, ts)
        fs = filter(s -> free[s], ts)
        foreach(s -> s == first(fs) || Graphs.add_edge!(links, first(fs), s), fs)
    end
    projectors = Dict{Int, Vector{Tuple{Vector{Int}, Matrix{Float64}}}}()
    for c in Graphs.connected_components(links)
        free[first(c)] && any(s -> Graphs.degree(measured, s) > 0, c) || continue
        at = Dict(s => i for (i, s) in pairs(c))
        rows = Vector{Float64}[]
        for s in c, b in Graphs.neighbors(measured, s)
            b > s || continue
            row = zeros(length(c))
            row[at[s]] = row[at[b]] = 1
            push!(rows, row)
        end
        touched = unique(track_of[s] for s in c)
        for ti in touched, (slots, R) in get!(() -> _prior_projectors(tracks[ti]), projectors, ti), i in axes(R, 1)
            row = zeros(length(c))
            for (j, s) in pairs(slots)
                haskey(at, s) && (row[at[s]] = R[i, j])
            end
            any(!iszero, row) && push!(rows, row)
        end
        N = nullspace(permutedims(reduce(hcat, rows)))
        ext = unique([c; reduce(vcat, track_slots[touched])])
        pos = Dict(s => i for (i, s) in pairs(ext))
        M = zeros(length(ext), size(N, 2) + length(touched))
        M[axes(c, 1), axes(N, 2)] .= N
        for (i, ti) in pairs(touched), s in track_slots[ti]
            M[pos[s], size(N, 2) + i] = 1
        end
        F = svd(M)
        r = count(>(first(F.S) * length(ext) * eps()), F.S)
        Q = F.U[:, 1:r]
        ℓ[ext] .-= Q * (Q' * ℓ[ext])
        r > length(touched) && push!(groups, (; slots = sort(c), ndir = r - length(touched)))
    end
    return groups
end

# The constraints track `t`'s prior puts on a shift `δ` of its slots: per piece the
# prior acts on, or per level group for a proper prior, its slots and the
# projector `R = I − B·B⁺` off the null space `B` of the prior there, so that
# `R·δ = 0`. A proper prior is profiled over its level group's constant; a
# flat-start walk is unchanged by a polynomial of degree below its order.
function _prior_projectors(t)
    out = Tuple{Vector{Int}, Matrix{Float64}}[]
    active = [(j, p) for (j, p) in pairs(t.pieces) if t.active[j]]
    if _proper_prior(t.prior)
        for g in unique(_level_group(t, t.prior, j) for (j, _) in active)
            slots = reduce(vcat, [t.slots[p] for (j, p) in active if _level_group(t, t.prior, j) == g])
            push!(out, (slots, I - fill(1 / length(slots), length(slots), length(slots))))
        end
    else
        for (_, p) in active
            u, _ = _piece_coordinates(t.prior, t.x[p])
            v = u .- first(u)
            scale = maximum(abs, v)
            iszero(scale) || (v ./= scale)
            B = [x^k for x in v, k in 0:(_statedim(t.prior) - 1)]
            push!(out, (t.slots[p], I - B * pinv(B)))
        end
    end
    return out
end

# Record a joint solve's track fits (`solve_joint_bandpass!`) into one array per
# station block of their status (`_block_status_array`) and prior
# (`_block_prior_array`), each `nothing` to skip. A solve over part of a track's
# time writes only its own time segments.
function _record_track_fits!(status, priors, fits)
    for f in fits
        t = f.track
        isnothing(status) || (status[t.k][t.ai, t.feed, :, t.ts] = f.status)
        isnothing(priors) || (priors[t.k][t.ai, t.feed, t.ts] = f.prior)
    end
    return nothing
end

# Each station's level for one observable, `nothing` where it has none: its
# piece → level map and level count against the station's shape segments, its
# level block's leaf, feed tying and position, the level of each shape segment,
# and its time-segment count.
function _antenna_levels(level_blocks, segs, loc, geom::DataGeometry, θ, ::Type{T}) where {T}
    out = Vector{Any}(nothing, length(loc))
    for lb in level_blocks, (li, a) in pairs(lb.antennas)
        k, _ = loc[a]
        iszero(k) && continue
        level, nlevel = _piece_levels(lb.plan, segs[k])
        out[a] = (;
            level, nlevel, leaf = lb.θ, tying = lb.plan.tying, ai = li,
            seg = [level[p] for p in segs[k].piece_of], nts = lb.plan.shape.Ti, geom.nfeed, T,
        )
    end
    return out
end

_level_values(::Nothing) = nothing
_level_values(l) = fill(l.T(NaN), l.nfeed, l.nlevel, l.nts)

_joint_level_writer(::Nothing, values) = nothing
_joint_level_writer(l, values) = (; l.leaf, l.tying, l.seg, values, l.ai)
