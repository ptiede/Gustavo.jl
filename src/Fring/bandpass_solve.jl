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
# frequency-segment order and its station's coordinates, pieces, prior, levels
# and delays (`fits[a]`), with `active` marking the pieces its prior acts on —
# those with data or fixed values at enough slots (`known`) to determine its
# states. A delay's regressor `du` is each slot's delay coordinate over the
# largest of the track's, `dunit`; a track with fewer than two known slots on
# its active pieces does not determine a delay and takes none.
function _bandpass_tracks(fits, observable, ids, loc, present, nfeed, known)
    antennas = [a for a in eachindex(fits, loc) if !isnothing(fits[a]) && !isnothing(fits[a][observable])]
    return [_bandpass_track(fits[a][observable], a, loc[a], feed, ts, ids, known) for a in antennas for feed in 1:nfeed for ts in present[a]]
end

function _bandpass_track(f, a, (k, ai), feed, ts, ids, known)
    slots = vec(ids[k][ai, feed, :, ts])
    active = [!isnothing(f.prior) && count(n -> known[n], slots[p]) >= _samples_needed(f.prior) for p in f.pieces]
    nknown = sum((count(n -> known[n], slots[p]) for (j, p) in pairs(f.pieces) if active[j]); init = 0)
    delay = nknown >= 2 ? f.delay : nothing
    dunit = isnothing(f.delay) ? 1.0 : maximum(abs, f.dx)
    du = isnothing(f.delay) ? nothing : f.dx ./ dunit
    return (; a, k, ai, feed, ts, slots, f.x, f.pieces, f.prior, f.level, f.nlevel, delay, du, dunit, active)
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

# A joint level or delay writer's values at time segment `ts`.
_level_at(::Nothing, ts) = nothing
_level_at(lw, ts) = merge(lw, (; values = view(lw.values, :, :, ts)))

# The phase a delay writer's delay `τ` adds at shape segment `fs`, zero where
# the track took no delay.
_delay_phase(::Nothing, f, fs) = 0.0
function _delay_phase(dw, f, fs)
    τ = dw.values[f, dw.seg[fs]]
    return isfinite(τ) ? 2π * τ * dw.dx[fs] : 0.0
end

# Gauge-fix each (station, feed, time segment) track — zero band-mean
# log-amplitude, circular-mean reference phase — and write it into the station
# block that carries it. Both gauges
# are over the frequency track of one time segment, so each of a station's
# segments is normalized on its own. Each observable is written, and referenced,
# over the slots it has a value in; the rest are left unwritten, holding whatever
# θ already had, which is unit gain, as is a station no block covers. With a level, the reference comes off the
# level and the shape is written as fit. With a delay, the delay is written and
# its phase taken off the shape.
#
# `gains[k]` holds the stations of phase block `k` and amplitude block `k`, in
# the blocks' own order; `level_writers[a]` holds station `a`'s
# `(; phase, amp)` level writers (see `_joint_level_writer`), each `nothing` without
# a level, and `delay_writers[a]` its delay writer, `nothing` without a delay.
#
# The band means are removed at θ's precision, which may exceed the gains': the
# gauge is a property of θ.
function _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout, level_writers, delay_writers)
    for (k, gk) in pairs(gains)
        pb, ab = get(phase_blocks, k, nothing), get(amp_blocks, k, nothing)
        block = something(pb, ab)
        T = eltype(block.θ)
        (; g, φ, phase_valid, amp_valid) = gk
        node(b, f) = isnothing(b) ? 0 : _feed_node(b.plan.tying, f)
        for (ai, a) in pairs(block.antennas), f in axes(g, Feed), ts in layout.present[a]
            pnode, anode = node(pb, f), node(ab, f)
            lw = map(l -> _level_at(l, ts), level_writers[a])
            dw = _level_at(delay_writers[a], ts)
            pv = view(phase_valid, ai, f, :, ts)
            if !iszero(pnode) && any(pv)
                phase = fs -> T(φ[ai, f, fs, ts]) - T(_delay_phase(dw, f, fs))
                mphase = angle(sum(cis(phase(fs)) for fs in axes(g, Frequency) if pv[fs]))
                Lp = _write_level!(lw.phase, f, ts, Lj -> rem2pi(T(Lj) - mphase, RoundNearest))
                _write_level!(dw, f, ts, T)
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
                          phase_level_blocks = [], amp_level_blocks = [], delay_blocks = [],
                          gauge, max_iterations = 200, max_em_iterations = 100, tolerance = 0.01,
                          flat_span = 0.01, eltype = nothing, tseg = nothing,
                          linsolve = KrylovJL_CG(; linesearch = true),
                          executor = SerialScheduler()) -> (; phase, amp)

Jointly solve the per-(station, feed, frequency segment) complex bandpass gain
and a per-scan, per-station-pair, per-feed-pair constant source coherence (see
the header comment of bandpass_data.jl for the model), then
gauge-fix each (station, feed, time segment) track and write the result into the
station blocks of the two observables (`_write_joint_bandpass!`).

`phase_blocks` and `amp_blocks` are `_bandpass_blocks(setup, θ, :phase)`
and `(…, :logamp)` — station blocks over the SAME `θ` this call is handed, since
each block's `θ` is a view into it — `phase_level_blocks`/`amp_level_blocks`
the matching `_bandpass_level_blocks`, empty without a level, and
`delay_blocks` the phase's `_bandpass_delay_blocks`, empty without a delay. A
station-uniform model gives one block per component spanning every station;
where the model differs across stations, each block carries its own stations,
feed tying, segments and prior, and a station no block covers is left at unit
gain. Each track is fit under its own block's prior, which relates the values
within each spectral window.
A station's two shapes must hold the same stations on the same frequency and
time segmentations, which `validate_model(::JointSmoother, model)` enforces at
model-compile time. Either list may be
empty, which holds that observable at zero: unit amplitude, or zero phase and
no gauge.

`geom` supplies the stations the `AntennaPair` labels name (θ's stations, in
θ's order) and the channel frequencies the priors are fit along.

Scaling a set of stations by one phase leaves every visibility internal to that
set unchanged, so the data leave one phase free per such set: the connected
components of the (station, feed, frequency segment, time segment) graph the
correlations span ([`_joint_bandpass_gauge!`](@ref)). A pin holds the
reference's slot in every component at zero, so the priors act on the other
tracks relative to it. Under any other gauge the priors along frequency set
every such shift but those the tracks' means take up — a constant under a
level, a line under a delay — and only those are fixed, by a shift along them
after the solve; where no prior acts, that is one per component.

`scans` is the per-scan `(rl, wl)` accumulator pairs from
`accumulate_bandpass` — not summed across scans, since the source term
needs each scan's own coherent visibility.

`tseg`, when given, is the `(antenna, scan)` time-segment table
([`_antenna_time_segments`](@ref)): station `a`'s gain is solved separately for
each distinct `tseg[a, :]` value, in that station's own segment numbering, and a
station whose entry is `0` is left out of the solve. Every station shares one
segment when it is omitted. `scans` must hold the scans `tseg`'s columns
describe, in the same order. The phase gauge acts on the graph these scans
span, so a break at one station is
measured against the stations that hold one gain across it rather than gauged
away. Solved phases are comparable only WITHIN a component: where that graph
splits — disjoint sub-arrays, or every station segmented at the same epoch —
each piece carries its own arbitrary constant, and a per-station change read
across the split is that constant plus the change.

The solve maximizes the posterior of the gains: χ²/2 of the visibilities with
`S` at its closed-form optimum for the gains (profiled), plus each track's prior
energy. `S` absorbs a constant on any (station, feed, time segment) track, which
the data therefore cannot measure: a proper prior acting on the track holds it
at the value the prior prefers, and on any other track one slot is held. The
solve

1. starts every slot at its own estimate given every other gain and `S`,
   repeated with `S` re-solved (StEFCal-style rounds), then searches each feed
   after the first for one delay common to every station against the products
   pairing it with an earlier feed, which the same-feed products the rounds
   follow do not see, and applies it where it is detected
   (`_align_feed_delays!`);
2. seeds each prior's hyperparameters by type-II MAP on its track alone, and
   puts each phase track on the 2π branches most probable under it
   (`_choose_branches!`), once: the data are fringe-corrected, so phase steps
   along frequency are small;
3. finds the MAP of the gains under the current hyperparameters by
   Levenberg–Marquardt on the profiled objective (`_profiled_map!`), each step
   solved by `linsolve`, a Krylov method of LinearSolve.jl, matrix-free and
   preconditioned by a sparse factorization (`ProfilePreconditioner`);
4. fits the hyperparameters given as densities by MAP-EM, accelerated by
   SQUAREM (`_squarem!`): the E-step takes each track's state moments under the
   Gaussian posterior at the MAP, the M-step maximizes each track's expected
   log prior plus its hyperprior.

The MAP stops once the undamped step moves no gain by `tolerance` of its
standard error from the data, once an accepted step lowers the objective by less
than `tolerance²/2`, or once no damping lowers it, with a warning if that step
would have moved a gain by `tolerance` or more. EM stops once a SQUAREM cycle
moves no gain by `tolerance` of its standard error. Each warns when it runs out
of iterations (`max_iterations` per MAP, `max_em_iterations` EM maps); an
unconverged solve still depends on its starting point. The passes over the data
run on `executor`. docs/src/priors.md ("The bandpass posterior") derives each
step. The scans handed to one call are a connected piece of the coupling graph,
so this is one criterion over one coupled problem: nodes that share no data are
solved by separate calls rather than being averaged into a common tolerance.

Returns, per observable, one `(; track, prior, status)` per (station, feed, time
segment) track: the track, the prior it was fit under with its hyperparameters
resolved, and the [`TrackStatus`](@ref) of each spectral window. The data are reduced
and the gains and source terms held in the accumulators' types, or in the real
type `eltype`; the steps are solved in `Float64`.
"""
function solve_joint_bandpass!(
        θ, scans, geom::DataGeometry, phase_blocks, amp_blocks;
        phase_level_blocks = NamedTuple[], amp_level_blocks = NamedTuple[], delay_blocks = NamedTuple[],
        gauge::AbstractGauge, max_iterations::Integer = 200, max_em_iterations::Integer = 100,
        tolerance::Real = 0.01, flat_span::Real = 0.01, eltype = nothing,
        tseg::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
        linsolve = LinearSolve.KrylovJL_CG(; linesearch = true), executor = SerialScheduler(),
    )
    isempty(scans) && return (; phase = [], amp = [])
    nant = length(geom.antennas)

    # The data are reduced onto the refinement of every block's frequency
    # segmentation, and each station's gain is held in its own segments — one
    # gain over however many refinement cells that segment spans.
    blocks = _gain_blocks(phase_blocks, amp_blocks)
    fseg, refinement = _antenna_freq_segments(blocks, nant)
    r, w = _reduce_all_scans(scans, refinement; eltype)
    ends = _cell_nodes(r, geom.antennas, PerFeed())
    data = DimStack((; r, w, ends))

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

    segs = [_shape_segments(b.plan, geom) for b in blocks]
    phase_level = _antenna_levels(phase_level_blocks, segs, loc, geom, θ, real(Base.eltype(r)))
    amp_level = _antenna_levels(amp_level_blocks, segs, loc, geom, θ, real(Base.eltype(r)))
    delay = _antenna_levels(delay_blocks, segs, loc, geom, θ, real(Base.eltype(r)))
    fits = map(1:nant) do a
        k, ai = loc[a]
        iszero(k) && return nothing
        seg = segs[k]
        track(blocks, level, delay) = isempty(blocks) ? nothing : (;
            seg.x, seg.pieces, prior = _prior_along(blocks[k].plan.priors[ai], :Frequency),
            level = isnothing(level) ? nothing : level.level, nlevel = isnothing(level) ? 0 : level.nlevel,
            delay = isnothing(delay) ? nothing : delay.level, dx = isnothing(delay) ? nothing : delay.dx,
        )
        return (; phase = track(phase_blocks, phase_level[a], delay[a]), amp = track(amp_blocks, amp_level[a], nothing))
    end

    slot = [(k, I) for (k, idk) in pairs(ids) for I in CartesianIndices(idk)]
    nslots = length(slot)
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
    # The gauge fixes phases; a solve that holds every phase at zero has none to
    # fix. Its freedoms are the shifts the priors acting on the data do not see.
    gauge_state = fitted.phase ?
        _joint_bandpass_gauge!(
            gains, data, layout, blocks, gauge;
            tracks = _bandpass_tracks(fits, :phase, ids, loc, layout.present, geom.nfeed, hasdata),
        ) : nothing
    pinned = [gains[k].pinned[I] for (k, I) in slot]
    # A held slot is fixed, not measured: a track held throughout gives its prior
    # nothing to act on.
    known = (; phase = hasdata .& .!pinned, amp = hasdata)
    cells = BandpassCells(r, w, jl.edge, jl.slot_pairs; nchunks = Threads.nthreads())
    npairs = length(jl.slot_pairs)
    weight = _slot_weights(cells, nslots)
    track_pins = (; phase = falses(nslots), amp = falses(nslots))
    tracks = map((; phase = :phase, amp = :amp)) do obs
        fitted[obs] || return NamedTuple[]
        gauge_pins = obs === :phase ? pinned : falses(nslots)
        return [
            _hold_track_constant!(track_pins[obs], t, gauge_pins, hasdata, weight)
                for t in _bandpass_tracks(fits, obs, ids, loc, layout.present, geom.nfeed, known[obs])
        ]
    end
    # A slot moves where it has data, or where a prior acting on its track fills it.
    function movable(obs)
        out = fitted[obs] .& hasdata
        for t in tracks[obs], (j, p) in pairs(t.pieces)
            t.active[j] && (out[t.slots[p]] .= fitted[obs])
        end
        return out
    end
    free = (; phase = movable(:phase) .& .!pinned .& .!track_pins.phase, amp = movable(:amp) .& .!track_pins.amp)
    filled = map(obs -> movable(obs) .& .!hasdata, (; phase = :phase, amp = :amp))
    acting = map(ts -> findall(t -> any(t.active), ts), tracks)
    track_layouts = map(ts -> map(_track_layout, ts), tracks)

    # Float64 whatever the data's type: CHOLMOD factors only Float64.
    ℓ, φ = zeros(nslots), zeros(nslots)
    gains_of(ℓ, φ) = exp.(complex.(ℓ, φ))
    source = zeros(ComplexF64, length(cells.ranges))
    function linearized(ℓ, φ)
        g = gains_of(ℓ, φ)
        return _linearize(g, _source!(source, g, cells, executor), cells, nslots, npairs, executor)
    end
    # Start every slot at its own estimate given the other gains and `S`, repeated
    # with `S` re-solved until a round lowers χ² by less than 10⁻⁵ of itself: a
    # nonlinear start the MAP solve then refines. From unit gains, a delay
    # difference across the band decoheres `S` and with it a single round's
    # estimates. Every slot moves at once, so every second round takes the mean
    # of the last two, which keeps the rounds from oscillating (Salvini &
    # Wijnholds 2014). `S` absorbs a constant on any track, which nothing else
    # holds, so each round takes each amplitude track's band mean out.
    χ2 = _profiled_chi2(gains_of(ℓ, φ), cells, executor)
    for round in 1:max_iterations
        lin = linearized(ℓ, φ)
        ℓ0, φ0 = copy(ℓ), copy(φ)
        for (obs, x, x0) in ((:phase, φ, φ0), (:amp, ℓ, ℓ0))
            fitted[obs] || continue
            y, _ = _slot_estimates(x0, lin, obs; phase_held = !fitted.phase)
            for n in eachindex(x)
                (free[obs][n] && isfinite(y[n])) || continue
                x[n] = iseven(round) ? (x0[n] + y[n]) / 2 : y[n]
            end
        end
        if fitted.amp
            for t in tracks.amp
                moved = [n for n in t.slots if free.amp[n]]
                isempty(moved) || (ℓ[moved] .-= sum(view(ℓ, moved)) / length(moved))
            end
        end
        χ2, χ2_before = _profiled_chi2(gains_of(ℓ, φ), cells, executor), χ2
        χ2_before - χ2 < 1.0e-5 * χ2 && break
    end
    if fitted.phase && geom.nfeed > 1
        compid, _, _ = _joint_bandpass_graph(data, layout, ids)
        _align_feed_delays!(
            φ, ℓ, cells, [I[2] for (_, I) in slot], [segs[k].x[I[3]] for (k, I) in slot],
            compid, pinned, track_pins.phase, tracks.phase,
        )
    end

    # Each acting track's prior with its hyperparameters resolved, seeded by
    # type-II MAP on the track alone from every slot's own estimate, and each
    # phase track put on the 2π branches most probable under it. The data are
    # fringe-corrected, so phase steps along frequency are small and the
    # branches are chosen once.
    lin = linearized(ℓ, φ)
    estimates = (;
        phase = fitted.phase ? _slot_estimates(φ, lin, :phase) : nothing,
        amp = fitted.amp ? _slot_estimates(ℓ, lin, :amp; phase_held = !fitted.phase) : nothing,
    )
    priors = map((; phase = :phase, amp = :amp)) do obs
        isempty(acting[obs]) && return []
        y, wy = estimates[obs]
        return [_seed_prior(tracks[obs][i], y, wy; phase = obs === :phase) for i in acting[obs]]
    end
    if fitted.phase
        y, wy = estimates.phase
        for (j, i) in pairs(acting.phase)
            _choose_branches!(φ, tracks.phase[i], priors.phase[j], y, wy, free.phase)
        end
    end

    key = zeros(nslots)
    for ts in tracks, t in ts
        key[t.slots] .= t.x
    end
    Hφ, hpos = _curvature_pattern(jl.slot_pairs, nslots)
    ctx = (;
        cells, nslots, npairs, fitted, free, filled, tracks, acting, track_layouts, key, Hφ, Hℓ = copy(Hφ), hpos,
        executor, systems = Dict{Symbol, Any}(), warm = Dict{Symbol, Any}(),
    )
    x = (; phase = φ, amp = ℓ)
    solved = _profiled_map!(x, ctx, priors; tolerance, max_iterations, linsolve)
    em_converged, em_unconverged, em_moved = true, 0, 0.0
    if !isempty(_estimated(ctx))
        em_unconverged = !(solved.converged || solved.stalled)
        em = _squarem!(priors, x, ctx, solved; tolerance, max_iterations, max_em_iterations, linsolve)
        solved, em_converged, em_moved = em.solved, em.converged, em.moved
        em_unconverged += em.unconverged
    end
    (; converged, stalled, change, worst) = solved
    free_amp = fitted.amp ? _split_free_amplitudes!(ℓ, jl.slot_pairs[measured], free.amp .| track_pins.amp, tracks.amp) : NamedTuple[]
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
        final = linearized(ℓ, φ)
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
    em_converged || @warn "the joint bandpass hyperparameter fit stopped at $max_em_iterations EM " *
        "iterations with its last cycle still moving a gain by $em_moved of its standard error, against " *
        "a tolerance of $tolerance; raise `JointSmoother(; max_em_iterations)`"
    em_unconverged > 0 && @warn "$em_unconverged of the joint bandpass MAP solves the hyperparameter fit " *
        "made did not converge in $max_iterations iterations, so the hyperparameters were fit to " *
        "unconverged gains; raise `JointSmoother(; max_iterations)`"
    stalled || converged || @warn "the joint bandpass solve did not converge in " *
        "$max_iterations iterations: the last step moved a gain by $change of its standard error$location, " *
        "against a tolerance of $tolerance; raise `JointSmoother(; max_iterations)`"

    # The pins fix the freedoms inside the solve; the gauge asked for is a shift
    # along them, which neither the data nor the priors see.
    isnothing(gauge_state) || _meet_gauge!(φ, (; gauge_state.rows, d = Float64.(gauge_state.d)), gauge_state.freedoms)
    for (n, (k, I)) in pairs(slot)
        gk = gains[k]
        gk.g[I] = exp(complex(ℓ[n], φ[n]))
        gk.φ[I] = φ[n]
        gk.phase_valid[I] = free.phase[n] || track_pins.phase[n] || (fitted.phase && pinned[n])
        gk.amp_valid[I] = free.amp[n] || track_pins.amp[n]
    end
    levels = [(; phase = _level_values(phase_level[a]), amp = _level_values(amp_level[a])) for a in 1:nant]
    delays = [_level_values(delay[a]) for a in 1:nant]
    solved = (; phase = φ, amp = ℓ)
    results = map((:phase, :amp)) do obs
        x = solved[obs]
        valid = obs === :phase ? free.phase .| track_pins.phase .| (fitted.phase .& pinned) : free.amp .| track_pins.amp
        prior_of = zeros(Int, length(tracks[obs]))
        prior_of[acting[obs]] = eachindex(acting[obs])
        map(eachindex(tracks[obs])) do i
            t = tracks[obs][i]
            j = prior_of[i]
            status = map(t.pieces) do p
                all(pinned[t.slots[p]]) && obs === :phase && return TrackSolved
                return _band_track_status([valid[n] ? x[n] : NaN for n in t.slots[p]], flat_span)
            end
            if !(isnothing(t.level) && isnothing(t.delay)) && any(t.active)
                _, mean = _track_energy(track_layouts[obs][i], priors[obs][j], x)
                isnothing(t.level) || for (g, v) in mean.levels
                    levels[t.a][obs][t.feed, g, t.ts] = v
                end
                for (g, v) in mean.delays
                    delays[t.a][t.feed, g, t.ts] = v / (2π * t.dunit)
                end
            end
            # A level no mean estimate sets, a held one among them, is zero:
            # the track's values are all in its shape.
            if !isnothing(t.level)
                L = view(levels[t.a][obs], t.feed, :, t.ts)
                foreach(g -> isnan(L[g]) && (L[g] = 0), unique(t.level))
            end
            return (; track = t, prior = iszero(j) ? t.prior : priors[obs][j], status)
        end
    end
    level_writers = [
        (; phase = _joint_level_writer(phase_level[a], levels[a].phase), amp = _joint_level_writer(amp_level[a], levels[a].amp))
            for a in 1:nant
    ]
    delay_writers = [_joint_level_writer(delay[a], delays[a]) for a in 1:nant]
    _write_joint_bandpass!(phase_blocks, amp_blocks, gains, layout, level_writers, delay_writers)
    return (; phase = results[1], amp = results[2])
end

# The false-alarm probability under which `_align_feed_delays!` takes a feed's
# delay as detected.
const _FEED_DELAY_PFA = 1.0e-4

# Adding one phase spectrum `ρ(ν)` to feed `f` of every station leaves every
# same-feed product unchanged and turns each product pairing `f` with another
# feed by `±ρ`, so only those products measure it. For each feed `f` after the
# first, in turn, this searches one delay `τ`, `ρ = 2πτ(ν − ν̄)` over the slot
# frequencies `ν`, against the products pairing `f` with an earlier feed. With
# `S` profiled, a group `J` lowers the χ² by `|Σ_c e^{∓iρ(ν_c)} conj(u_c) r_c|²
# / Σ_c w_c|u_c|²`, `u = g_a·conj(g_b)`, which on noise alone is χ² with two
# degrees of freedom, since the weights are inverse variances per real
# component. The trial delays span `±1/(2δν)` in steps of `1/(4B)`, over the
# band `B` and the smallest spacing `δν` of the frequencies, so the best one is
# within `π/8` of the peak's phase at the band's edges. A peak whose
# false-alarm probability over those trials is below `_FEED_DELAY_PFA` is
# applied to `φ`, less a constant per component of the gauge graph (`component`)
# that keeps its `pinned` slot and a constant per track that keeps its `held`
# slot: the same-feed products and `S` see neither. Otherwise the feed's delay
# is left where it is, with a warning.
function _align_feed_delays!(φ, ℓ, cells::BandpassCells, feed, ν, component, pinned, held, tracks)
    for f in 2:maximum(feed)
        freqs = sort!(unique(ν[n] for n in eachindex(ν, feed) if feed[n] == f))
        length(freqs) > 1 || continue
        column = Dict(v => i for (i, v) in pairs(freqs))
        g = exp.(complex.(ℓ, φ))
        groups = Vector{ComplexF64}[]
        for rng in cells.ranges
            fa, fb = feed[cells.na[first(rng)]], feed[cells.nb[first(rng)]]
            ((fa == f && fb < f) || (fb == f && fa < f)) || continue
            y = zeros(ComplexF64, length(freqs))
            den = 0.0
            for k in rng
                a, b = cells.na[k], cells.nb[k]
                u = g[a] * conj(g[b])
                v = conj(u) * cells.r[k]
                fa == f ? (y[column[ν[a]]] += v) : (y[column[ν[b]]] += conj(v))
                den += cells.w[k] * abs2(u)
            end
            den > 0 && push!(groups, y ./ sqrt(den))
        end
        isempty(groups) && continue
        Y = reduce(hcat, groups)
        ν̄ = sum(freqs) / length(freqs)
        spacing = 1 / (4 * (last(freqs) - first(freqs)))
        n = ceil(Int, 1 / (2 * minimum(diff(freqs))) / spacing)
        τs = (-n:n) .* spacing
        power = zeros(length(τs))
        for trials in Iterators.partition(eachindex(τs), 256)
            power[trials] .= vec(sum(abs2, cis.(-2π .* τs[trials] .* (freqs .- ν̄)') * Y; dims = 2))
        end
        best = argmax(power)
        pfa = -expm1(length(τs) * log1p(-ccdf(Chisq(2 * size(Y, 2)), power[best])))
        if pfa > _FEED_DELAY_PFA
            @warn "the joint bandpass found no delay of feed $f common to every station in the " *
                "$(size(Y, 2)) groups of products pairing it with an earlier feed: the best of " *
                "$(length(τs)) trial delays has false-alarm probability $pfa. The solve keeps the " *
                "delay the data start with, and may stall short of the right one if that is far off."
            continue
        end
        shift = [feed[m] == f ? 2π * τs[best] * (ν[m] - ν̄) : 0.0 for m in eachindex(ν, feed)]
        keep = Dict(component[m] => shift[m] for m in eachindex(shift, pinned) if pinned[m] && !iszero(component[m]))
        for m in eachindex(shift, component)
            shift[m] -= get(keep, component[m], 0.0)
        end
        for t in tracks
            h = findfirst(m -> held[m], t.slots)
            isnothing(h) || (shift[t.slots] .-= shift[t.slots[h]])
        end
        φ .+= shift
    end
    return φ
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

# The constraints track `t`'s prior puts on a shift `δ` of its slots: per
# `_prior_null_bases` entry, its slots and the projector `R = I − B·B⁺` off its
# null space, so that `R·δ = 0`.
_prior_projectors(t) = [(slots, I - B * pinv(B)) for (slots, B) in _prior_null_bases(t)]

# The shifts of track `t`'s slots its prior does not see, as `(slots, B)` with
# the columns of `B` spanning them over `slots`: under a proper prior its mean
# over the active pieces (`_mean_designs`), which it is profiled over; under a
# flat-start walk a polynomial of degree below its order on each active piece.
function _prior_null_bases(t)
    active = [(j, p) for (j, p) in pairs(t.pieces) if t.active[j]]
    isempty(active) && return Tuple{Vector{Int}, Matrix{Float64}}[]
    if _proper_prior(t.prior)
        design = _mean_designs(t, t.prior)
        cols = sort!(unique(c for (j, _) in active for c in design[j].cols))
        slots = reduce(vcat, [t.slots[p] for (_, p) in active])
        B = zeros(length(slots), length(cols))
        r = 0
        for (j, p) in active, i in eachindex(p)
            r += 1
            for (c, h) in zip(design[j].cols, design[j].rows[i])
                B[r, searchsortedfirst(cols, c)] = h
            end
        end
        return [(slots, B)]
    end
    return map(active) do (_, p)
        u, _ = _piece_coordinates(t.prior, t.x[p])
        v = u .- first(u)
        scale = maximum(abs, v)
        iszero(scale) || (v ./= scale)
        (t.slots[p], [x^k for x in v, k in 0:(_statedim(t.prior) - 1)])
    end
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

# Each station's level for one observable, or its delay, `nothing` where it has
# none: its piece → segment map and segment count against the station's shape
# segments, its block's leaf, feed tying and position, the segment of each shape
# segment, its time-segment count, and each shape segment's mean frequency
# coordinate under the block's term (`dx`, the delay's `f − f0`).
function _antenna_levels(level_blocks, segs, loc, geom::DataGeometry, θ, ::Type{T}) where {T}
    out = Vector{Any}(nothing, length(loc))
    for lb in level_blocks, (li, a) in pairs(lb.antennas)
        k, _ = loc[a]
        iszero(k) && continue
        level, nlevel = _piece_levels(lb.plan, segs[k])
        out[a] = (;
            level, nlevel, leaf = lb.θ, tying = lb.plan.tying, ai = li,
            seg = [level[p] for p in segs[k].piece_of], nts = lb.plan.shape.Ti, geom.nfeed, T,
            dx = [sum(view(lb.plan.xf, chans)) / length(chans) for chans in segs[k].fsegs],
        )
    end
    return out
end

_level_values(::Nothing) = nothing
_level_values(l) = fill(l.T(NaN), l.nfeed, l.nlevel, l.nts)

_joint_level_writer(::Nothing, values) = nothing
_joint_level_writer(l, values) = (; l.leaf, l.tying, l.seg, values, l.ai, l.dx)
