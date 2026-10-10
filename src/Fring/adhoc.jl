# ── Globally-closing adhoc phasing ───────────────────────────────────────────
#
# After the per-scan fringe solution there remains a fast, time-variable phase
# per station — atmospheric turbulence — that a per-scan constant cannot track.
# It is solved globally rather than against a reference antenna: per
# accumulation period (AP), station phases are fit from all baselines with the
# same feed-aware incidence WLS as the station solve, so the solution closes by
# construction and stays well-determined at low SNR with no dominant anchor.
#
# For baseline (a, b), correlation product p with feeds (fa, fb), at AP t:
#
#     y_{ab,p}(t) = φ_{na}(t) − φ_{nb}(t) + x_{ab,p} + ε,
#
# where `n· = _feed_node(tying, f·)` maps a feed onto its parameter node and
# `x_{ab,p}` is the source visibility phase, held constant over the scan.
#
# `x` must stay free per (baseline, product): that is what keeps the model
# independent of the polarization basis, since the source EVPA, the D-terms and
# the source's own closure phase then land in `x` rather than biasing the
# station tracks. A source term shared by every baseline is a rank-one
# restriction, which cannot represent closure phase at all.
#
# `x` is degenerate with a per-station constant (`φ_a → φ_a + c_a`, `x_{ab,p} →
# x_{ab,p} − (c_a − c_b)`), a gauge the per-scan demean in step (5) fixes. A
# per-station slope is not degenerate with a constant `x`, so the residual-rate
# information survives the demean.
#
# Per scan: (1) the caller supplies residual baseline visibilities, already
# divided by the fringe solution and coherently frequency-averaged to one
# complex number per (baseline, product, AP); (2) alternate between solving the
# per-AP node phases over the graph's connected components and re-estimating
# each `x_{ab,p}` as the weighted circular mean of its residual; (3) unwrap each
# track across APs; (4) fit each track under the adhoc component's prior along
# time (statespace.jl, prior_fits.jl); (5) demean per track, so the adhoc track
# does not alias the Stage-B constant phase. Steps (1)–(3) are identical for
# every smoother, which only ever decides step (4).

# ── Adhoc smoothers: a pluggable type-based interface ────────────────────────
#
# After the per-AP global solve each (station, feed) phase track is fit under
# the adhoc component's prior along time. The smoother is a strategy type
# deciding how: track by track, or jointly over every station.
#
# Internal traits, with defaults in the traits section below:
# `_adhoc_coherence_time(sm)` is the assumed atmospheric `T_coh` in seconds,
# used by the per-AP warm-start staleness check; `_requires_single_node(sm)` is
# whether the smoother needs one phase node per station, which only the joint
# solve does.

"""
    AbstractAdhocSmoother

How the [`AdhocPhase`](@ref Gustavo.AdhocPhase) step fits the per-AP station
phase tracks the adhoc solve produces (`solve_adhoc_phasing`) under the adhoc
component's prior: [`PerTrackAdhocSmoother`](@ref) (the default) or
[`JointKalmanSmoother`](@ref).

# Implementing a smoother

Define a struct and:

    Gustavo.Fring.apply_adhoc!(sm::MySmoother, phase, track_w; obs, priors, resolved)

the single dispatch point; mutates `phase` in place. `phase` and `track_w` are
`DimArray`s over `(AntennaName, FeedNode, Ti)`: each (station, feed node) phase track and
its per-AP coherent weight, `Ti` carrying the AP epochs in seconds. Each AP of
`phase` is centered on the cells covered throughout the scan; the step's gauge
is applied after smoothing. `obs` holds the
SNR-gated, source-corrected observations: `obs.val`, `obs.w` and the gate
`obs.mask` over `(AntennaPair, FeedPair, Ti)`, and `obs.nodes`, each cell's
`((a, na), (b, nb))` — the station index and feed node at either end.
`priors[a]` is station `a`'s
prior along time: `nothing`, a [`RandomWalkPrior`](@ref) or an
[`OUPrior`](@ref). `resolved`, over `(AntennaName, FeedNode)`, receives the prior each
track was fit under, its hyperparameters fixed. The default `apply_adhoc!` fits
each (station, node) track that carries data independently through the
per-track hook, so a smoother that acts track by track implements only

    Gustavo.Fring.smooth_track!(sm::MySmoother, track, w, prior) -> resolved prior

which overwrites `track` with its fitted values; `lookup(track, Ti)` gives its
times.

A smoother declares which adhoc components it can solve with

    Gustavo.Fring.can_fit(sm::MySmoother, tc::Calibration.GainComponent, geom) -> Bool

checked at model-compile time. The default is the capability of the solve
machinery — per-AP constant phase over the global band, `SharedFeeds` or
`PerFeed`, with no prior or a `RandomWalkPrior` or `OUPrior` along `Ti`;
`JointKalmanSmoother` restricts to `SharedFeeds` (its Kalman state holds one
phase per station) and to a proper prior: an `OUPrior`, or a `RandomWalkPrior`
with an `init`.

Every smoother carries its solve options in an `options::AdhocOptions` field;
see [`AdhocOptions`](@ref).
"""
abstract type AbstractAdhocSmoother end

"""
    AdhocOptions(; snr_floor = 1.0, phase_rewrap_iters = 3, source_iters = 10,
                 source_tol = 1.0e-6, complex_iters = 2, eltype = nothing)

The solve options every [`AbstractAdhocSmoother`](@ref) carries in its
`options` field, e.g. `PerTrackAdhocSmoother(; options =
AdhocOptions(; snr_floor = 0.0))`.

`snr_floor` gates the seed pass only (the phase-extraction solve that fixes
the global 2π branch); `complex_iters` Gauss–Newton passes then re-fit the
tracks against the complex residuals with every AP entering ungated
(`complex_iters = 0` keeps the seed). `phase_rewrap_iters` bounds the
rewrap passes of each solve. `source_iters` alternations between the station
tracks and the per-(baseline, product) source phases run until no source
phase moves by more than `source_tol` radians, not counting a move that
per-station constants absorb; `source_iters = 1` fits no source term.

The solve works in the real type of the summed visibilities (`Float32` for an
MSv4 store); `eltype`, a real floating-point type, names another.
"""
Base.@kwdef struct AdhocOptions
    snr_floor::Float64 = 1.0
    phase_rewrap_iters::Int = 3
    source_iters::Int = 10
    source_tol::Float64 = 1.0e-6
    complex_iters::Int = 2
    eltype::Union{Nothing, DataType} = nothing
    function AdhocOptions(snr_floor, phase_rewrap_iters, source_iters, source_tol, complex_iters, eltype)
        isnothing(eltype) || eltype <: AbstractFloat || throw(
            ArgumentError("`eltype` must be a real floating-point type or `nothing`, got $eltype"),
        )
        return new(snr_floor, phase_rewrap_iters, source_iters, source_tol, complex_iters, eltype)
    end
end

"""
    PerTrackAdhocSmoother(; coherence_time = 10.0, options = AdhocOptions())

Fit each (station, feed node) phase track of a scan on its own, under the adhoc
component's prior along `Ti` (the default): its maximum a posteriori values.

- no prior: the per-AP solve stands as is.
- [`RandomWalkPrior`](@ref): a `σ` given as a hyperprior is resolved per
  (station, feed node, scan) by type-II MAP on the restricted likelihood; the
  track is then the random-walk MAP.
- [`OUPrior`](@ref): hyperparameters given as hyperpriors are resolved per
  (station, feed node, scan) by type-II MAP, with the track's level integrated
  out under a flat prior; the track is then the level plus the zero-mean OU
  MAP.

Hyperparameters and levels are independent between scans.

`coherence_time` is the assumed atmospheric coherence time in seconds, the time
for the phase to drift by one radian. It bounds how long a solved AP seeds the
2π branch of later ones; it does not enter any prior.

`options` is an [`AdhocOptions`](@ref).
"""
Base.@kwdef struct PerTrackAdhocSmoother <: AbstractAdhocSmoother
    coherence_time::Float64 = 10.0
    options::AdhocOptions = AdhocOptions()
end

"""
    JointKalmanSmoother(; coherence_time = 10.0, options = AdhocOptions())

Fit every station's phase track of a scan at once: one multivariate Kalman
filter and RTS smoother over the station phases, observing the baseline phase
differences directly under each station's prior along `Ti`: an
[`OUPrior`](@ref), or a [`RandomWalkPrior`](@ref) with an `init`, which holds
the common phase of all stations that the differences leave free. Closure
and denoising happen in one estimator, which conditions better at low SNR than
fitting each track after the per-AP solve. Seeded and rewrapped from the per-AP
solve; see `_solve_gp_joint!`.

A station's hyperparameters given as hyperpriors are resolved per scan by
type-II MAP on its per-AP track, with the track's level integrated out under a
flat prior. A station without a track of its own takes the median of the
resolved values of the stations whose prior has the same form (an `OUPrior`,
or a walk of the same order); a station whose hyperparameters are fixed keeps
them.

Requires one phase node per station (a `SharedFeeds` adhoc component), since
its state carries one dimension per station.

`coherence_time` is the assumed atmospheric coherence time in seconds. It
bounds how long a solved AP seeds the 2π branch of later ones; it does not
enter any prior.

`options` is an [`AdhocOptions`](@ref).
"""
Base.@kwdef struct JointKalmanSmoother <: AbstractAdhocSmoother
    coherence_time::Float64 = 10.0
    options::AdhocOptions = AdhocOptions()
end

# ── Smoother interface traits ─────────────────────────────────────────────────
# The assumed atmospheric coherence time (seconds) — used by the per-AP warm-start
# staleness. Smoothers that carry no coherence time use the same default as those
# that do.
_adhoc_coherence_time(::AbstractAdhocSmoother) = 10.0
_adhoc_coherence_time(sm::PerTrackAdhocSmoother) = sm.coherence_time
_adhoc_coherence_time(sm::JointKalmanSmoother) = sm.coherence_time

# Whether the smoother needs one phase node per station. The joint solve's Kalman
# state is one dimension per station, so it cannot represent two independent feed
# tracks.
_requires_single_node(::AbstractAdhocSmoother) = false
_requires_single_node(::JointKalmanSmoother) = true

"""
    default_adhoc_terms(; feed = SharedFeeds(), prior = default_adhoc_prior()) -> GainModel

The default [`AdhocPhase`](@ref Gustavo.AdhocPhase) step model: one per-AP constant phase
over the global band — a `phase.adhoc` component with `Ti = PerIntegration()`.
`feed` is its feed tying: `SharedFeeds()` (the default) solves one track per
station — residual atmospheric phase is non-birefringent, and a feed-common
track contributes zero inter-feed phase in any feed basis, where a `PerFeed()`
solve lets per-AP noise differ between feeds and so scatters the inter-feed
phase. `PerFeed()` fits each feed's own track when the per-feed structure is
real.

`prior` is the component's prior along `Ti`: by default an [`OUPrior`](@ref)
with weakly informative hyperpriors ([`default_adhoc_prior`](@ref)); `nothing`
leaves the per-AP solve unsmoothed.

The adhoc stage solves exactly one phase component and no logamp.
"""
default_adhoc_terms(; feed::AbstractFeedTying = SharedFeeds(), prior = default_adhoc_prior()) =
    GainModel(phase = (; adhoc = GainComponent(ConstantTerm(); Ti = PerIntegration(), Feed = feed, prior)))

"""
    default_adhoc_prior() -> OUPrior

The default prior of the adhoc phase along time: an [`OUPrior`](@ref) whose
`scale` has a `LogNormal` hyperprior with median 10 s and whose `σ` has one
with median 0.3 rad, each with log-standard deviation 1.
"""
default_adhoc_prior() = OUPrior(; scale = LogNormal(log(10.0), 1.0), σ = LogNormal(log(0.3), 1.0))

# Capability declarations for the compile-time `can_fit`/`validate_model` seam
# (capability.jl; the step drives the checks in
# `model_components(::AdhocPhase, spec)`).
#
# What the solve machinery addresses: one constant per (feed node, AP) —
# `adhoc_scan!` writes leaf slot (param 1, node, freq segment 1, time segment,
# ant), and `solve_adhoc_phasing`'s single-node-per-row systems can tie feeds
# (`SharedFeeds`) or solve them independently (`PerFeed`) but cannot represent
# a `SingleFeed` scope.
_fits_adhoc_track(tc) =
    tc.term isa ConstantTerm && tc.Ti isa PerIntegration &&
    tc.Frequency isa GlobalFrequency && (tc.Feed isa PerFeed || tc.Feed isa SharedFeeds) &&
    _is_prior_along(resolve_prior(tc), :Ti)

can_fit(::AbstractAdhocSmoother, tc, geom) = _fits_adhoc_track(tc)
# The joint solve needs one phase node per station (`_requires_single_node`) and
# a proper prior, which holds the common mode the baseline differences leave free.
can_fit(::JointKalmanSmoother, tc, geom) =
    _fits_adhoc_track(tc) && tc.Feed isa SharedFeeds && _proper_prior(_prior_along(resolve_prior(tc), :Ti))

# The structural contract of the adhoc pass: exactly one per-integration phase
# component (the stage runs one globally-closing phase solve and writes one θ
# block), nothing in logamp.
function validate_model(sm::AbstractAdhocSmoother, model)
    ph = Calibration._flatten_components(model.phase)
    la = Calibration._flatten_components(model.logamp)
    length(ph) == 1 || throw(
        ArgumentError(
            "the adhoc stage solves exactly one per-integration phase component; the " *
                "model's phase group holds $(length(ph)). A second (e.g. per-scan) " *
                "component is not identifiable: a constant on a station's track trades " *
                "against every baseline's free source term. `default_adhoc_terms()` is " *
                "the standard model.",
        ),
    )
    isempty(la) || throw(
        ArgumentError(
            "the adhoc stage solves phase only; the model's logamp group must be empty, " *
                "got $(length(la)) component(s).",
        ),
    )
    return nothing
end

# ── apply_adhoc!: the single smoothing dispatch point ─────────────────────────
# Mutates the per-(station, node) phase track array `phase` in place. `obs` is
# the SNR-gated observations with the source term already removed, so every
# smoother sees the same observations; per-track smoothers ignore it.

# Default: fit each (station, node) track that carries data through the per-track hook.
function apply_adhoc!(sm::AbstractAdhocSmoother, phase, track_w; obs, priors, resolved)
    for a in axes(phase, AntennaName), f in axes(phase, FeedNode)
        track = view(phase, a, f, :)
        w = view(track_w, a, f, :)
        any(k -> _shape_usable(track[k], w[k]), eachindex(track, w)) || continue
        resolved[a, f] = smooth_track!(sm, track, w, priors[a])
    end
    return phase
end

# Joint state-space solve: one Kalman filter over all station phases observing
# baseline differences directly, seeded/rewrapped from the per-AP solve.
function apply_adhoc!(sm::JointKalmanSmoother, phase, track_w; obs, priors, resolved)
    all(_proper_prior, priors) || throw(
        ArgumentError(
            "JointKalmanSmoother needs an OUPrior or a RandomWalkPrior with an `init` at " *
                "every station, got " *
                join(unique(map(p -> isnothing(p) ? "nothing" : _call_string(p), priors)), ", "),
        ),
    )
    _solve_gp_joint!(phase, track_w, obs, sm, priors, resolved)
    return phase
end

# ── smooth_track!: per-(station, feed) fitting hook ───────────────────────────
# `track` is one (station, feed node) phase track (unwrapped, radians) over
# `Ti`, overwritten with its fitted values; `w` is its per-AP coherent weight.
# Returns the prior the track was fit under.

_track_times(track) = parent(lookup(track, Ti))

smooth_track!(::PerTrackAdhocSmoother, track, w, ::Nothing) = nothing

# Under a proper prior the track's level is the per-scan constant the demean
# removes afterwards, so it is flat and unknown here: integrated out of the
# hyperparameter fit, then estimated by GLS and restored around the zero-mean
# MAP. A walk without an init leaves its level free already.
function smooth_track!(::PerTrackAdhocSmoother, track, w, prior::Union{OUPrior, RandomWalkPrior})
    x = _track_times(track)
    if !_proper_prior(prior)
        resolved = _estimate_hypers(prior, [track], [w], [x])
        _estimate_map!(track, resolved, track, w, x)
        return resolved
    end
    resolved = _estimate_hypers(prior, [track], [w], [x]; level = [1])
    level = only(_estimate_levels(resolved, [track], [w], [x], [1], 1))
    track .-= level
    _estimate_map!(track, resolved, track, w, x)
    track .+= level
    return resolved
end

# ── Node-graph solve (adhoc per-AP closures) ─────────────────────────────────

# Node index on the (station, feed node) graph: node `n`'s block is `(n - 1) * nant .+ (1:nant)`.
_node(ant::Integer, feed::Integer, nant::Integer) = (feed - 1) * nant + ant

# The two ends of a node-system edge on the (station, feed) graph. A cell of a
# `(AntennaPair, FeedPair)` system observes `φ(a, na) − φ(b, nb)`, where
# `nodes[cell] = ((a, na), (b, nb))` gives each end's station index and phase
# node — `_feed_node(tying, feed)` of the product's feeds, so the node equals the
# feed only under `PerFeed`.
_edge(((a, na), (b, nb)), nant::Integer) = (_node(a, na, nant), _node(b, nb, nant))

# Solve one observable's WLS system on the (station, feed) graph into `vals`
# and `cov`, each `(nant, nnode)`: the node values, `NaN` where unsolved, and which
# nodes were solved. `val`, `w`, `mask` and `nodes` share their `(AntennaPair,
# FeedPair)` axes; each cell with `mask` set is one observation `val` of its edge
# (see `_edge`) with weight `w`. The system is solved in `val`'s element type.
# `scan` labels the gauge freedoms, whose observable is `:phase` (see
# `GaugeFreedom`).
# Returns the number of connected components.
function _solve_observable!(
        vals, cov, val, w, mask, nodes, gauge::AbstractGauge;
        rewrap::Integer,
        seed_phase::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        scan::Integer = 0,
    )
    # `_node` numbers stations and feeds from 1.
    Base.require_one_based_indexing(vals, cov)
    isnothing(seed_phase) || Base.require_one_based_indexing(seed_phase)
    T = eltype(val)
    nant, nnode = size(vals)
    size(vals) == size(cov) ||
        throw(DimensionMismatch("vals and cov must have the same size; got $(size(vals)) and $(size(cov))"))
    fill!(vals, NaN)
    fill!(cov, false)
    nnodes = nnode * nant
    cells = [I for I in eachindex(val, w, mask, nodes) if mask[I]]
    isempty(cells) && return 0

    edges = [_edge(nodes[I], nant) for I in cells]
    b = T[val[I] for I in cells]
    wt = T[w[I] for I in cells]
    compid, ncomp, touched = connected_components(nnodes, edges)

    # `gauge` sets each component's offset (`_regauge!`). `anchors` names a real
    # node per component as well — phase unwrapping propagates outward from an
    # actual node, which a summed gauge row does not provide.
    nodew = _node_weights(nnodes, edges, wt)
    # Inverse of `_node`.
    station_of(n) = (n - 1) % nant + 1
    feed_of(n) = (n - 1) ÷ nant + 1
    freedoms = map(1:ncomp) do c
        cn = findall(==(c), compid)
        GaugeFreedom(;
            nodes = cn, station = station_of.(cn), feed = feed_of.(cn), scan = fill(Int(scan), length(cn)),
            component = fill((), length(cn)), observable = fill(:phase, length(cn)),
            direction = ones(T, length(cn)), weight = nodew[cn],
        )
    end
    anchors = [gauge_anchor(gauge, f) for f in freedoms]
    A = zeros(T, length(edges), nnodes)
    for (i, (u, v)) in enumerate(edges)
        A[i, u] += one(T)
        A[i, v] -= one(T)
    end
    # Only differences are observed, so each component is solved with its anchor
    # held at 0 (untouched nodes also stay 0) and `_regauge!` then applies `gauge`.
    free = [n for n in eachindex(touched) if touched[n] && n ∉ anchors]
    solve_free = FactoredWLS(A[:, free], wt)
    function solve_system(y)
        full = zeros(T, nnodes)
        full[free] .= solve_free(y)
        return full
    end

    # Phase: unwrap toward a spanning-tree seed rather than solving the raw
    # wrapped observations, whose WLS fit can lock onto the wrong 2π branch when
    # a station difference exceeds ±π. Every edge is a pure node difference
    # (the caller removes the source terms), so the tree may use any of them.
    if rewrap > 0
        xseed = _prim_seed(T, nnodes, zip(first.(edges), last.(edges), b, wt), anchors)
        # A `seed_phase` (the previous AP's solved node phases) overrides the
        # tree wherever it is finite. It only picks each observation's 2π
        # branch, and edge predictions are gauge-invariant, so a differently
        # anchored neighboring AP is a valid seed; it keeps a weakly constrained
        # station from flipping between branches from one AP to the next.
        if !isnothing(seed_phase)
            for ant in axes(seed_phase, 1), feed in axes(seed_phase, 2)
                v = seed_phase[ant, feed]
                isfinite(v) && (xseed[_node(ant, feed, nant)] = v)
            end
        end
        x, _ = _rewrap_solve(solve_system, A, b, xseed, rewrap)
    else
        x = solve_system(b)
    end
    _regauge!(x, GaugeFreedoms{T}(freedoms, nnodes), gauge)

    for ant in axes(vals, 1), feed in axes(vals, 2)
        n = _node(ant, feed, nant)
        if touched[n]
            vals[ant, feed] = x[n]
            cov[ant, feed] = true
        end
    end
    return ncomp
end

# Shift each freedom's nodes of `x` by a constant so that `x` meets its gauge
# constraint. A constant per component changes no edge difference; each
# freedom's `direction` is all ones here.
function _regauge!(x, freedoms::GaugeFreedoms, gauge::AbstractGauge)
    C, d = _gauge_system(gauge, freedoms)
    for (j, f) in pairs(freedoms)
        cn = f.nodes
        c = (sum(n -> C[j, n] * x[n], cn) - d[j]) / sum(n -> C[j, n], cn)
        for n in cn
            x[n] -= c
        end
    end
    return x
end

# Maximum-weight spanning-tree phase seed on `nnodes` nodes. Each tree edge
# `(u, v, y, w)` observes `φ[u] − φ[v] = y`, wrapped, with weight `w`. The tree
# grown from each anchor (Prim) attaches the heaviest edge leaving the visited
# set, so a node joined to the tree by a weak edge and a strong one takes its
# branch from the strong one. Anchors, and nodes no tree reaches, stay 0.
function _prim_seed(::Type{T}, nnodes::Integer, tree_edges, anchors) where {T}
    x = zeros(T, nnodes)
    adj = [Tuple{Int, T, T}[] for _ in 1:nnodes]
    for (u, v, y, w) in tree_edges
        push!(adj[u], (v, -y, w))
        push!(adj[v], (u, y, w))
    end
    visited = falses(nnodes)
    for p in anchors
        (1 <= p <= nnodes && !visited[p]) || continue
        visited[p] = true
        while true
            best_w = T(-Inf)
            best_u = best_v = 0
            best_add = zero(T)
            for u in eachindex(visited)
                visited[u] || continue
                for (v, add, w) in adj[u]
                    (!visited[v] && w > best_w) || continue
                    best_w, best_u, best_v, best_add = w, u, v, add
                end
            end
            best_v == 0 && break
            x[best_v] = x[best_u] + best_add
            visited[best_v] = true
        end
    end
    return x
end

# Solve `A x ≈ b` for phases `b` known modulo 2π: move each observation by whole
# turns to the branch nearest the model `A x`, starting from `xseed`, and
# re-solve, `iters + 1` times in all. Returns the solution and the unwrapped
# observations it fits.
function _rewrap_solve(solve, A, b, xseed, iters::Integer)
    twopi = 2 * eltype(b)(π)
    x = xseed
    bw = similar(b)
    for _ in 0:iters
        model = A * x
        @. bw = b + twopi * round((model - b) / twopi)
        x = solve(bw)
    end
    return x, bw
end

# Circular (complex-phasor) solve of one AP's system, z_a ← Σ_b w·e^{iφ_ab}·z_b,
# gauged at the anchor. It seeds the linear solve at APs with no usable
# warm-start snapshot and must not be replaced by a cold linear solve: the
# phases-as-values WLS has 2π-branch local minima when observations sit near
# ±π, and a cold-started AP can converge ~150° off a strong one, after which the
# warm-start chain locks that branch and relaxes toward truth across the scan,
# leaving a smooth ±π-scale arc in the station track. The phasor iteration is
# circular and so has no branch structure. Every gated cell is a pure node
# difference, its source term already removed, so all of them drive the iteration.
function _circular_ap_seed(val, w, mask, nodes, nant::Integer, nnode::Integer, anchor::Integer)
    T = eltype(val)
    cells = [I for I in eachindex(val, w, mask, nodes) if mask[I]]
    isempty(cells) && return nothing
    z = ones(Complex{T}, nant, nnode)
    present = falses(nant, nnode)
    for I in cells
        (a, na), (b, nb) = nodes[I]
        present[a, na] = true
        present[b, nb] = true
    end
    for _ in 1:50
        acc = zeros(Complex{T}, nant, nnode)
        for I in cells
            (a, na), (b, nb) = nodes[I]
            R = cis(val[I])
            acc[a, na] += w[I] * R * z[b, nb]
            acc[b, nb] += w[I] * conj(R) * z[a, na]
        end
        for i in eachindex(z)
            present[i] || continue
            a = abs(acc[i])
            a > 0 && (z[i] = acc[i] / a)
        end
    end
    # One common gauge for the whole seed, an anchor node when present. A fully
    # consistent seed is safe on anchor-dropout APs too: the rewrap uses
    # prediction differences, so a common gauge offset cancels.
    pin = findfirst(view(present, anchor, :))
    g = isnothing(pin) ? one(Complex{T}) : conj(z[anchor, pin]) / abs(z[anchor, pin])
    ph = fill(T(NaN), nant, nnode)
    for f in axes(ph, 2), a in axes(ph, 1)
        present[a, f] && (ph[a, f] = angle(z[a, f] * g))
    end
    return ph
end

# Each (station pair, feed pair) cell's two ends: `((a, na), (b, nb))`, the
# station's index in `stations` and the phase node of its feed under `tying`
# (0 where the component has none).
function _cell_nodes(rbar, stations, tying)
    slot = Dict(n => i for (i, n) in pairs(stations))
    station(n) = get(slot, n) do
        throw(ArgumentError("station `$n` of a station pair is not among the stations " * join(stations, ", ")))
    end
    sps = DimensionalData.dims(rbar, AntennaPair)
    fps = DimensionalData.dims(rbar, FeedPair)
    ends = [
        ((station(sa), _feed_node(tying, fa)), (station(sb), _feed_node(tying, fb)))
            for (sa, sb) in lookup(sps), (fa, fb) in lookup(fps)
    ]
    return DimArray(ends, (sps, fps))
end

# Whether a cell constrains two phase nodes: not an autocorrelation, and both
# feeds parameterized by the component.
_solvable(((a, na), (b, nb))) = a != b && na != 0 && nb != 0

# The seed pass's observations over `rbar`'s `(AntennaPair, FeedPair, Ti)`: each
# cell's phase `angle(r)`, weighted by its coherent SNR² and gated at
# `snr_floor2`. Every correlation product contributes: a product's own phase
# offset is carried by its free source term, so no product needs the
# polarization basis to be known.
function _adhoc_obs(rbar, wbar, nodes, snr_floor2::Real)
    T = real(eltype(rbar))
    val = similar(rbar, T)
    w = similar(rbar, T)
    mask = similar(rbar, Bool)
    for I in DimensionalData.DimIndices(rbar)
        cell = DimensionalData.otherdims(I, Ti)
        r, wr = rbar[I], wbar[I]
        snr2 = abs2(r) / wr
        ok = _solvable(nodes[cell]) && isfinite(r) && abs(r) > 0 && isfinite(wr) && wr > 0 &&
            snr2 >= snr_floor2
        val[I] = angle(r)
        w[I] = ok ? snr2 : zero(T)
        mask[I] = ok
    end
    return (; val, w, mask, nodes)
end

# Weighted circular mean of each cell's source term from its residual
# `y − (φ_na − φ_nb)` over the APs where the cell passed the gate and both nodes
# solved. `obs` must carry the uncorrected phases, since `x` is defined relative
# to the observation itself.
function _update_source_terms!(x, obs, phase)
    (; val, w, mask, nodes) = obs
    acc = zeros(Complex{eltype(x)}, DimensionalData.dims(x))
    for ap in axes(val, Ti), I in DimensionalData.DimIndices(nodes)
        c = (I..., Ti(ap))
        mask[c...] || continue
        (a, na), (b, nb) = nodes[I...]
        pa = phase[a, na, ap]
        pb = phase[b, nb, ap]
        (isfinite(pa) && isfinite(pb)) || continue
        acc[I...] += w[c...] * cis(val[c...] - (pa - pb))
    end
    for i in eachindex(x, acc)
        abs(acc[i]) > 0 && (x[i] = angle(acc[i]))
    end
    return x
end

# The source terms and the node tracks share a degeneracy: adding `c[a, na] −
# c[b, nb]` to every term and subtracting `c` from the tracks leaves the model
# unchanged. Returns the largest change from `x_prev` to `x` that no such `c`
# absorbs, fitting `c` by weighted least squares over the cells in `mask`.
function _source_move(x, x_prev, cell_w, mask, nodes, nant::Integer, nnode::Integer, anchor::Integer)
    T = eltype(x)
    dx = map((a, b) -> rem2pi(a - b, RoundNearest), x, x_prev)
    c = fill(T(NaN), nant, nnode)
    _solve_observable!(c, falses(nant, nnode), dx, cell_w, mask, nodes, PinAntenna(anchor); rewrap = 0)
    moved = zero(T)
    for I in eachindex(dx, mask, nodes)
        mask[I] || continue
        (a, na), (b, nb) = nodes[I]
        moved = max(moved, abs(dx[I] - (c[a, na] - c[b, nb])))
    end
    return moved
end

# The observations with each cell's source term `x` removed, gated to the cells
# whose source term is identifiable (`keep`): a (baseline, product) seen at a
# single AP is absorbed exactly by its own source term, so it constrains no
# node phase and only inflates the apparent coverage.
_source_corrected(obs, x, keep) = (;
    val = DimensionalData.broadcast_dims(-, obs.val, x),
    obs.w,
    mask = DimensionalData.broadcast_dims(&, obs.mask, keep),
    obs.nodes,
)

# Joint adhoc solve: one Kalman filter + RTS smoother over every station's
# state, observing the baseline phase differences directly under each station's
# proper temporal prior, so closure and denoising happen in one
# recursive estimator. This conditions better at low SNR than solving each AP
# and smoothing afterwards. Mutates `phase[:, 1, :]` in place and records each
# written station's resolved prior in `resolved[:, 1]`; the caller's demean and
# node→feed expansion run afterwards.
#
# State = each station's model state, its phase first. `obs` arrives with each
# cell's source term already removed, so a cell is a pure node difference and
# the filter needs no augmented source dimension. The proper priors pin the
# unobservable common mode near 0; the caller applies the gauge afterwards.
# Requires one node per station.
function _solve_gp_joint!(phase, track_w, obs, sm::JointKalmanSmoother, priors, resolved)
    nant = size(phase, AntennaName)
    times = parent(lookup(phase, Ti))
    # Compute type flows from the data, not from the smoother's field types.
    T = float(promote_type(eltype(phase), eltype(track_w), eltype(times)))

    # Seed (unwrapped) per-station track from the per-AP solve.
    θseed = [T(phase[i, 1, ap]) for i in axes(phase, 1), ap in axes(phase, 3)]

    # A station with no gated row anywhere in the scan has an all-zero column in every
    # H below, so the filter never updates it: it holds its prior mean for the whole
    # solve, an information-free track. Such stations are excluded from the hyper
    # fit and from the write-back. A track that is identically zero, the pinned
    # station's when no cell is covered throughout the scan, carries no variance
    # information either.
    seen = [any(>(0), view(track_w, i, 1, :)) for i in axes(track_w, 1)]
    welldet = [seen[i] && any(v -> isfinite(v) && !iszero(v), view(phase, i, 1, :)) for i in eachindex(seen)]
    any(welldet) || return phase
    track(i) = (view(θseed, i, :), collect(T, parent(view(track_w, i, 1, :))))

    # Per-station hyperparameters from the seed track, the level integrated out.
    # A station without a well-determined track takes the median of the others'
    # values, so every station keeps a proper prior during the solve.
    fitted = Vector{Any}(nothing, nant)
    for i in eachindex(welldet, priors)
        welldet[i] || continue
        y, w = track(i)
        fitted[i] = _estimate_hypers(priors[i], [y], [w], [times]; level = [1])
    end
    peers = [fitted[i] for i in eachindex(welldet) if welldet[i]]
    for i in eachindex(welldet, priors)
        welldet[i] || (fitted[i] = _peer_resolved(priors[i], peers, i))
    end
    models = [_state_model(fitted[i], T) for i in 1:nant]
    value_state = cumsum(map(statedim, models)) .- map(statedim, models) .+ 1

    # Per-station level: each prior is zero-mean, so each track is centered on
    # its GLS level under its own prior before the solve. The downstream demean
    # removes the level again.
    mθ = zeros(T, nant)
    for i in eachindex(seen)
        seen[i] || continue
        y, w = track(i)
        L = only(_estimate_levels(fitted[i], [y], [w], [times], [1], 1))
        isfinite(L) && (mθ[i] = L)
    end

    # Per-cell measurement variance (1/snr²); 0 marks a gated-out cell, which the
    # filter skips. With one node per station, a cell's state pair is its stations.
    ends = map((((a, _), (b, _)),) -> (a, b), obs.nodes)
    rs = [
        map((m, wi) -> m ? inv(T(wi)) : zero(T), view(obs.mask, Ti(ap)), view(obs.w, Ti(ap)))
            for ap in axes(obs.w, Ti)
    ]
    ys = [similar(r) for r in rs]

    # Current full (uncentered) estimate, seeded from the per-AP solve.
    θf = [isfinite(θseed[i, ap]) ? θseed[i, ap] : mθ[i] for i in axes(θseed, 1), ap in axes(θseed, 2)]
    twoπ = 2 * T(π)

    # Iterated Kalman: rewrap each raw observation toward the current joint model,
    # then re-run the forward/backward smoother (relinearizing the ±2π branch).
    for _ in 1:max(sm.options.phase_rewrap_iters, 1)
        for ap in eachindex(ys)
            y = ys[ap]
            v = view(obs.val, Ti(ap))
            for j in eachindex(ends, y, v)
                a, b = ends[j]
                model = θf[a, ap] - θf[b, ap]
                center = mθ[a] - mθ[b]
                raw = T(v[j])
                y[j] = raw + twoπ * round((model - raw) / twoπ) - center
            end
        end
        xf, Pf, xp, Pp, As, _ = kalman_mv_filter(ends, ys, rs, times, models)
        xs, _ = rts_smooth_mv(xf, Pf, xp, Pp, As)
        for ap in axes(θf, 2)
            for i in axes(θf, 1)
                θf[i, ap] = xs[value_state[i], ap] + mθ[i]
            end
        end
    end

    # Write back the joint track; internal gaps stay interpolated by the filter. An
    # unseen station keeps the per-AP solve's NaN, so it is never fabricated into θ
    # (`adhoc_scan!` gates its θ write on `isfinite`, not on `covered`).
    for i in eachindex(seen)
        seen[i] || continue
        phase[i, 1, :] .= view(θf, i, :)
        resolved[i, 1] = fitted[i]
    end
    return phase
end

# A station's prior as a state-space model along the scan's time coordinate.
_state_model(p::OUPrior, ::Type{T}) where {T} = OUModel{T}(p.scale, p.σ^2)
_state_model(p::RandomWalkPrior, ::Type{T}) where {T} = _random_walk_model(p.order, p.σ, p.init, one(T), T)

# The prior of a station without a well-determined track of its own: its fixed
# hyperparameters, and for each hyperprior the median of the resolved values of
# the `peers` whose prior has the same form.
_peer_resolved(p::OUPrior, peers, station) =
    OUPrior(; scale = _peer_median(p, :scale, peers, station), σ = _peer_median(p, :σ, peers, station))
_peer_resolved(p::RandomWalkPrior, peers, station) =
    RandomWalkPrior(; order = p.order, σ = _peer_median(p, :σ, peers, station), p.init)

_same_form(::OUPrior, q) = q isa OUPrior
_same_form(p::RandomWalkPrior, q) = q isa RandomWalkPrior && q.order == p.order

function _peer_median(p, name::Symbol, peers, station)
    v = getproperty(p, name)
    is_fixed_hyper(v) && return v
    vals = [getproperty(q, name) for q in peers if _same_form(p, q)]
    isempty(vals) && throw(
        ArgumentError(
            "JointKalmanSmoother: station $station has no track of its own to resolve the " *
                "`$name` hyperprior of $(_call_string(p)), and no other station has a prior " *
                "of the same form to take it from; give it a fixed value",
        ),
    )
    return median(vals)
end

# One full per-AP sweep: solve every AP's station phases from `obs` (with the
# anchor-gauged warm-start snapshot carrying 2π-branch continuity across APs),
# restitch anchor-dropout APs, and unwrap each (station, node) track. Shared by
# the seed alternation passes and the complex-domain refinement passes, which
# differ only in how their observations were built.
function _solve_ap_sweep!(
        phase, covered, track_w, obs, nant::Integer, anchor::Integer,
        rewrap::Integer, max_stale::Integer,
    )
    (; val, w, mask, nodes) = obs
    fill!(phase, convert(eltype(phase), NaN))
    fill!(covered, false)
    fill!(track_w, zero(eltype(track_w)))
    nnode = size(phase, 2)
    prev_phase = fill(convert(eltype(phase), NaN), nant, nnode)  # cells no anchor-present AP has covered yet
    prev_age = zeros(Int, nant, nnode)    # APs since a cell was last refreshed (staleness)
    for ap in axes(val, Ti)
        v, wk, mk = view(val, Ti(ap)), view(w, Ti(ap)), view(mask, Ti(ap))
        # The anchor has data this AP iff some observation touches it (⇒ the solve
        # is pinned at anchor=0). Seed only then, and only with fresh cells in the
        # anchor gauge.
        ref_here = false
        for I in eachindex(wk, mk, nodes)
            mk[I] || continue
            (a, na), (b, nb) = nodes[I]
            track_w[a, na, ap] += wk[I]
            track_w[b, nb, ap] += wk[I]
            ref_here |= a == anchor || b == anchor
        end
        seed = nothing
        if ref_here
            seed = fill(convert(eltype(phase), NaN), nant, nnode)
            for a in axes(seed, 1), n in axes(seed, 2)
                (isfinite(prev_phase[a, n]) && prev_age[a, n] <= max_stale) &&
                    (seed[a, n] = prev_phase[a, n])
            end
        end
        # No usable warm-start snapshot (first AP, all cells stale, or an
        # anchor-dropout AP): seed from the circular phasor solve instead of
        # trusting the tree-initialized linear solve's 2π branch — see
        # `_circular_ap_seed`.
        if seed === nothing || !any(isfinite, seed)
            seed = _circular_ap_seed(v, wk, mk, nodes, nant, nnode, anchor)
        end
        ph, cov = view(phase, :, :, ap), view(covered, :, :, ap)
        _solve_observable!(ph, cov, v, wk, mk, nodes, PinAntenna(anchor); rewrap, seed_phase = seed)
        # Refresh the anchor-gauged snapshot only from anchor-present APs (keep the
        # last known value for a station absent this AP, so a brief dropout does not
        # reset the branch); age every cell and zero the ones refreshed here.
        prev_age .+= 1
        if ref_here
            for a in axes(ph, 1), n in axes(ph, 2)
                if cov[a, n] && isfinite(ph[a, n])
                    prev_phase[a, n] = ph[a, n]
                    prev_age[a, n] = 0
                end
            end
        end
    end

    # Restitch the per-AP gauge when the anchor drops out. Each per-AP solve pins
    # the anchor; in APs where it has no data the solve falls back to a different
    # pin node, so that AP's whole solution is offset by an arbitrary, non-2π
    # constant, which would otherwise inject a spurious common-mode jump into
    # every station's track.
    _restitch_refant_gauge!(phase, covered, track_w, obs, anchor)

    # Unwrap each (station, node) track across APs (per-AP solves share the ref
    # gauge, so a track is continuous up to ±2π steps the unwrap removes).
    for a in axes(phase, 1), n in axes(phase, 2)
        any(isfinite, @view phase[a, n, :]) || continue
        phase[a, n, :] .= unwrap_phase_track(phase[a, n, :]; weights = track_w[a, n, :])
    end
    return phase
end

# Per-(baseline, product) complex source term for the Gauss–Newton refinement:
# the inverse-variance mean of the model-derotated per-AP visibilities over the
# whole scan, so its SNR is the track's rather than one AP's, and its |s̄|² is
# the signal power the linearized observations are weighted by. Also returns
# how many APs informed each term (its identifiability count).
function _complex_source_means(rbar, wbar, phase, nodes)
    cells = DimensionalData.dims(nodes)
    sbar = zeros(eltype(rbar), cells)
    nrm = zeros(real(eltype(rbar)), cells)
    napu = zeros(Int, cells)
    for I in DimensionalData.DimIndices(nodes)
        _solvable(nodes[I...]) || continue
        (a, na), (b, nb) = nodes[I...]
        for ap in axes(rbar, Ti)
            w = wbar[I..., Ti(ap)]
            r = rbar[I..., Ti(ap)]
            (isfinite(r) && isfinite(w) && w > 0) || continue
            dphi = phase[a, na, ap] - phase[b, nb, ap]
            isfinite(dphi) || continue
            sbar[I...] += r * cis(-dphi)     # r = Σ w·V ⇒ this is Σ w·V·e^{-iΔφ̂}
            nrm[I...] += w
            napu[I...] += 1
        end
        nrm[I...] > 0 && (sbar[I...] /= nrm[I...])
    end
    return sbar, napu
end

# Linearized (Gauss–Newton) observations in the complex domain. Around the
# current tracks, `V̄·conj(s̄)e^{-iΔφ̂} ≈ |s̄|²(1 + i(Δφ − Δφ̂)) + n·conj(s̄)`, so
#
#     val  = Δφ̂ + Im(V̄·conj(s̄)e^{-iΔφ̂}) / |s̄|²
#     info = |s̄|² · w̄          (w̄ = Σw, the inverse variance of V̄ per real component)
#
# is a linear measurement of Δφ with Gaussian noise at any per-AP SNR. Every AP
# with data and a track therefore enters ungated, carrying its honest weight;
# a per-AP extracted phase would instead collapse nonlinearly below SNR ≈ 1.
# The source term is already divided out through `conj(s̄)`.
function _linearized_obs(rbar, wbar, nodes, phase, sbar)
    T = real(eltype(rbar))
    val = fill!(similar(rbar, T), T(NaN))
    w = fill!(similar(rbar, T), zero(T))
    mask = fill!(similar(rbar, Bool), false)
    for ap in axes(rbar, Ti), I in DimensionalData.DimIndices(nodes)
        _solvable(nodes[I...]) || continue
        (a, na), (b, nb) = nodes[I...]
        c = (I..., Ti(ap))
        wr = wbar[c...]
        r = rbar[c...]
        (isfinite(r) && isfinite(wr) && wr > 0) || continue
        dphi = phase[a, na, ap] - phase[b, nb, ap]
        isfinite(dphi) || continue
        s = sbar[I...]
        s2 = abs2(s)
        (isfinite(s2) && s2 > 0) || continue
        z = imag((r / wr) * conj(s) * cis(-dphi)) / s2
        isfinite(z) || continue
        val[c...] = dphi + z
        w[c...] = s2 * wr
        mask[c...] = true
    end
    return (; val, w, mask, nodes)
end

"""
    solve_adhoc_phasing(rbar, wbar, stations; gauge, smoother, tying, prior) -> DimStack

Solve globally-closing adhoc phases from coherently frequency-averaged
residual baseline visibilities under the model

    y[station pair, feed pair, ap] = φ_na(ap) − φ_nb(ap) + x[station pair, feed pair],

with the station phases `φ` on parameter nodes and one free source term `x`
per (station pair, feed pair), constant over the scan. The free source term
carries the source's EVPA, D-terms, and closure phase, so the station tracks
are unbiased by source structure.

`rbar` is `Σ_chan w·V_residual` and `wbar` is `Σ_chan w`, both `DimArray`s
over `AntennaPair`, `FeedPair` and `Ti` in any storage order, with the same
lookups: station pairs labeled by station names, feed pairs by feed-index
pairs, and `Ti` by the AP epochs in seconds. The coherent SNR² is
`|rbar|²/wbar`. `stations` names the stations; each one's position in it is
its station index. The number of feeds per station is the largest feed index
in the `FeedPair` labels.

Returns a `DimStack`: `:phase` (`AntennaName(stations) × Feed × Ti`) is the
per-(station, feed) adhoc phase in radians, `NaN` where unsolved;
`:covered` marks the solved cells; `:source` (`AntennaPair × FeedPair`) is
the fitted source phase, `NaN` where unidentifiable; `:prior` (`AntennaName(stations)
× Feed`) is the prior each track was fit under, its hyperparameters resolved,
`nothing` where the track had no prior or no data.

Each track is its weighted mean over the scan plus the time structure: the
constant is a gauge, since a per-station constant trades against the source
terms, so every track is returned with zero weighted mean. The residual-rate
slope is kept.

`gauge` sets the per-AP convention, applied after smoothing so that it does
not change any baseline difference: `PinAntenna` holds its reference's phase
at 0, `ZeroSumPhase` centers each AP on zero mean.

`tying` is the adhoc component's [`AbstractFeedTying`](@ref). `PerFeed()`
(the default) solves an independent track per feed; `SharedFeeds()` solves
one feed-common phase, constrained by every correlation product and
contributing zero inter-feed phase. Under `PerFeed`, an AP's products that pair
different feed nodes join their blocks into one connected component with a
single gauge freedom; blocks no such product joins keep a freedom each. The
solved phases do not depend on how a station orders its feeds, and under
`SharedFeeds` they do not depend on its feed basis either.

`prior` is the adhoc component's prior along time — `nothing`, a
[`RandomWalkPrior`](@ref) or an [`OUPrior`](@ref) — for every station, or a
vector holding one per station.

`smoother` is an [`AbstractAdhocSmoother`](@ref) selecting how the tracks are
fit under `prior` after the solve; its [`AdhocOptions`](@ref) set the solve. The
`(φ, x)` blocks are fit by alternating minimization for at most
`options.source_iters` passes, stopping once no source term moves by more than
`options.source_tol` radians beyond what per-station constants absorb (a
constant moved between the tracks and the source terms leaves the model
unchanged); `source_iters = 1` fixes `x = 0`.
"""
function solve_adhoc_phasing(
        rbar::DimensionalData.AbstractDimArray{<:Complex, 3},
        wbar::DimensionalData.AbstractDimArray{<:Real, 3},
        stations::AbstractVector;
        gauge::AbstractGauge,
        smoother::AbstractAdhocSmoother = PerTrackAdhocSmoother(),
        tying::AbstractFeedTying = PerFeed(),
        prior = default_adhoc_prior(),
    )
    cell_dims = (AntennaPair, FeedPair, Ti)
    all(d -> DimensionalData.hasdim(rbar, d), cell_dims) || throw(
        ArgumentError(
            "`rbar` must be over AntennaPair, FeedPair and Ti; got " *
                join(DimensionalData.name(DimensionalData.dims(rbar)), ", "),
        ),
    )
    DimensionalData.comparedims(
        DimensionalData.dims(wbar, DimensionalData.dims(rbar)), DimensionalData.dims(rbar); val = true,
    )
    # Positions along `Ti` and in `stations` index the solve's 1-based arrays.
    Base.require_one_based_indexing(rbar, wbar, stations)
    allunique(stations) || throw(
        ArgumentError(
            "station names must be unique; repeated: " *
                join((n for n in unique(stations) if count(==(n), stations) > 1), ", "),
        ),
    )
    nfeed = maximum(maximum, lookup(rbar, FeedPair))
    nnode = nfeed_blocks(tying, nfeed)
    _requires_single_node(smoother) && nnode != 1 && error(
        "$(typeof(smoother)) requires one phase node per station (its Kalman state " *
            "is one dimension per station), but the adhoc component ties feeds as " *
            "$(typeof(tying)). Use PerTrackAdhocSmoother for an independent per-feed track.",
    )
    priors = _station_priors(prior, stations)
    T = something(smoother.options.eltype, float(real(eltype(rbar))))
    r = Complex{T}.(permutedims(rbar, cell_dims))
    w = T.(permutedims(wbar, cell_dims))
    return _solve_adhoc_phasing(r, w, _cell_nodes(r, stations, tying), stations, nfeed, nnode, gauge, smoother, tying, priors)
end

_station_priors(prior, stations) = _station_priors(fill(prior, length(stations)), stations)
function _station_priors(priors::AbstractVector, stations)
    length(priors) == length(stations) || throw(
        DimensionMismatch("$(length(priors)) priors for $(length(stations)) stations"),
    )
    for p in priors
        p isa Union{Nothing, RandomWalkPrior, OUPrior} || throw(
            ArgumentError("an adhoc prior must be `nothing`, a RandomWalkPrior or an OUPrior, got $(typeof(p))"),
        )
    end
    return priors
end

# Behind a function barrier: the working type comes from a runtime option.
# `rbar`/`wbar` are stored `(AntennaPair, FeedPair, Ti)`, as is every derived array.
function _solve_adhoc_phasing(rbar, wbar, nodes, stations, nfeed, nnode, gauge, smoother, tying, priors)
    T = real(eltype(rbar))
    nant = length(stations)
    nap = size(rbar, Ti)
    times = parent(lookup(rbar, Ti))

    # Solved on the (station, feed node) graph; expanded back onto the feed axis at the end.
    node_axes = (AntennaName(stations), FeedNode(1:nnode), DimensionalData.dims(rbar, Ti))
    phase = fill(T(NaN), node_axes)
    covered = DimArray(falses(nant, nnode, nap), node_axes)
    # Track per-(station, feed node) coherent weight for smoothing and the demean.
    track_w = zeros(T, node_axes)
    resolved = DimArray(Union{Nothing, AbstractPrior}[nothing for _ in 1:nant, _ in 1:nnode], node_axes[1:2])

    # The SNR gate does not depend on the source terms, so the gated observations
    # are built once and reused by every alternation pass (and by the joint smoother).
    raw = _adhoc_obs(rbar, wbar, nodes, smoother.options.snr_floor^2)

    # Source terms, one per (station pair, feed pair). `source_iters == 1` never
    # updates them, so the model reduces exactly to a pure station-difference
    # solve with no source term.
    x = zeros(T, DimensionalData.dims(nodes))
    keep = fill!(similar(nodes, Bool), true)
    fit_source = smoother.options.source_iters >= 2
    if fit_source
        # A (baseline, product) seen at a single AP is absorbed exactly by its own
        # source term: it constrains no node phase, and admitting it would only
        # inflate the apparent coverage. Two APs is the identifiability threshold,
        # not a tuning choice.
        keep .= dropdims(sum(raw.mask; dims = Ti); dims = Ti) .>= 2
        # Seed each source term from its own observations. The gauge the per-track
        # demean imposes leaves each node track with ~zero scan mean, so a row's
        # scan-mean is its source term to first order. Seeding from 0 instead would
        # make the first solve fit rows a full source phase away from their model,
        # which for a source phase near ±π locks the wrong 2π branch — one the
        # later passes inherit through the warm start and cannot leave.
        _update_source_terms!(x, raw, zeros(T, nant, nnode, nap))
    end
    obs = _source_corrected(raw, x, keep)

    # The per-AP solves' anchor: the station with the most gated weight in the
    # scan. The per-AP pin, the warm-start seed condition and the restitch key
    # on it being present. It is internal and independent of `gauge`, which is
    # applied once, after smoothing.
    anchor = let wtot = zeros(T, nant)
        for ap in axes(obs.w, Ti)
            wk, mk = view(obs.w, Ti(ap)), view(obs.mask, Ti(ap))
            for I in eachindex(wk, mk, nodes)
                mk[I] || continue
                (a, _), (b, _) = nodes[I]
                wtot[a] += wk[I]
                wtot[b] += wk[I]
            end
        end
        all(iszero, wtot) ? 1 : argmax(wtot)
    end

    cell_w = dropdims(sum(map((w, m) -> m ? w : zero(w), raw.w, raw.mask); dims = Ti); dims = Ti)
    # Carry solved node phases forward as a temporal warm-start for the next AP's
    # 2π-branch selection, so weakly-constrained stations do not flip branch per
    # AP. The seed is an anchor-gauged snapshot: refreshed only from APs where
    # the anchor has data, so every stored cell is in the anchor = 0 gauge, and
    # applied only when the current AP also has the anchor, so the per-AP
    # spanning-tree seed it partially overrides is anchor = 0 too. Mixing an
    # anchor = 0 seed with a differently-anchored tree seed would mis-pick the 2π
    # branch on a seeded↔tree boundary edge. The snapshot is rebuilt from scratch
    # by each alternation pass, so a pass never inherits a branch chosen against
    # a stale set of source terms.
    #
    # A seed is trusted only while the atmosphere cannot yet have drifted past
    # ±π, beyond which the old branch is no longer a safe guide; that takes a few
    # coherence times. `times` is in seconds and `T_AP` is its median spacing.
    dts0 = filter(>(0), diff(sort(times)))
    t_ap0 = isempty(dts0) ? 1.0 : float(median(dts0))
    max_stale = max(10, ceil(Int, 3 * _adhoc_coherence_time(smoother) / t_ap0))
    # Alternating minimization of the joint (node phases, source terms) least
    # squares: each pass re-solves every AP with the current source terms removed,
    # then re-estimates each source term from the residual it leaves. Both blocks
    # are conditionally linear, so the alternation descends a convex quadratic.
    # With `source_iters == 1` the source terms are never fitted and stay at 0, so
    # the model reduces to a pure station-difference solve.
    for iter in 1:max(smoother.options.source_iters, 1)
        _solve_ap_sweep!(
            phase, covered, track_w, obs, nant, anchor,
            smoother.options.phase_rewrap_iters, max_stale,
        )
        (fit_source && iter < smoother.options.source_iters) || break
        x_prev = copy(x)
        _update_source_terms!(x, raw, phase)
        obs = _source_corrected(raw, x, keep)
        moved = _source_move(x, x_prev, cell_w, keep, nodes, nant, nnode, anchor)
        moved <= smoother.options.source_tol && break
    end

    _smooth_tracks!(smoother, phase, covered, track_w; obs, priors, resolved)

    # Gauss–Newton refinement in the complex domain. Everything above is the
    # seed: the phase-extraction solve's spanning-tree unwrap and warm starts
    # settle the global 2π branch, which no local linearization can, and the SNR
    # gate is confined to that seeding role. Each pass here re-derives the
    # per-(baseline, product) complex source terms from the whole scan,
    # linearizes every AP's residual around the current tracks
    # (`_linearized_obs`), and re-solves and re-smooths on those observations,
    # every AP entering at its exact first-order information, ungated. With the
    # smoothing pass inside, each iteration is an extended-Kalman/RTS step and
    # the loop is Gauss–Newton on the MAP objective of the complex data; the
    # innovations start on the seed's branch, so the linearization stays inside
    # ±π by construction.
    sbar_ref = nothing
    nap_ref = nothing
    for _ in 1:max(smoother.options.complex_iters, 0)
        sbar, napu = _complex_source_means(rbar, wbar, phase, nodes)
        sbar_ref = sbar
        nap_ref = napu
        lin = _linearized_obs(rbar, wbar, nodes, phase, sbar)
        _solve_ap_sweep!(
            phase, covered, track_w, lin, nant, anchor,
            smoother.options.phase_rewrap_iters, max_stale,
        )
        _smooth_tracks!(smoother, phase, covered, track_w; obs = lin, priors, resolved)
    end

    # The per-station constant's gauge; see `_detrend_track!`.
    removed = [_detrend_track!(@view(phase[a, n, :]), @view(track_w[a, n, :])) for a in axes(phase, 1), n in axes(phase, 2)]

    # Impose the gauge last. Demeaning removes each track's temporal mean, which
    # offsets every AP by the same constant, so applying the gauge after it leaves
    # the per-AP convention exact and the demeaned tracks zero-mean up to one
    # global constant — and that constant is a per-AP common mode, which cancels on
    # every baseline. The two conventions are compatible only up to that constant;
    # this order is what makes the gauge the exact one.
    _apply_ap_gauge!(phase, covered, track_w, gauge)

    # Expand the (station, feed node) solution onto the feed axis the caller
    # indexes: feeds sharing a node get identical tracks (so a `SharedFeeds` adhoc
    # contributes exactly zero inter-feed phase), and a feed the component does not
    # parameterize (node 0) stays NaN/uncovered.
    feed_axes = (AntennaName(stations), Feed(1:nfeed), DimensionalData.dims(rbar, Ti))
    phase_out = fill(T(NaN), feed_axes)
    covered_out = DimArray(falses(nant, nfeed, nap), feed_axes)
    prior_out = DimArray(Union{Nothing, AbstractPrior}[nothing for _ in 1:nant, _ in 1:nfeed], feed_axes[1:2])
    for f in lookup(phase_out, Feed)
        n = _feed_node(tying, f)
        n == 0 && continue
        view(phase_out, Feed(At(f))) .= view(phase, FeedNode(At(n)))
        view(covered_out, Feed(At(f))) .= view(covered, FeedNode(At(n)))
        view(prior_out, Feed(At(f))) .= view(resolved, FeedNode(At(n)))
    end

    # The refinement's complex source means supersede the seed's phase-only
    # alternation estimates: same per-(baseline, product) constant, measured
    # against the final tracks over every usable AP. Two APs stays the
    # identifiability threshold.
    if sbar_ref !== nothing
        for I in eachindex(x, keep, sbar_ref, nap_ref)
            keep[I] = nap_ref[I] >= 2 && abs2(sbar_ref[I]) > 0
            keep[I] && (x[I] = angle(sbar_ref[I]))
        end
    end

    # Source terms in the demeaned tracks' gauge (`x_ab → x_ab + c_a − c_b`),
    # `NaN` where too poorly sampled to identify.
    source = map(x, keep, nodes) do xi, k, ends
        (a, na), (b, nb) = ends
        !k ? T(NaN) : _solvable(ends) ? xi + removed[a, na] - removed[b, nb] : xi
    end

    return DimensionalData.DimStack((phase = phase_out, covered = covered_out, source = source, prior = prior_out))
end

# ── Per-scan pipeline entry ───────────────────────────────────────────────────

"""
    adhoc_scan!(θ, group::XRadio.ProcessingSet, geom::DataGeometry, adhoc_plan,
                adhoc, gauge; executor = DynamicScheduler()) -> DimArray

The per-integration atmospheric-phase (adhoc) solve of one scan group — the
"caller" the module docstring above refers to. On data already gain-corrected:
sum the per-(station pair, feed pair, AP)
inverse-variance residual over every band (`weighted_sums`), solve the
globally-closing per-AP station phase through the pluggable `adhoc` smoother
under each station's prior (`solve_adhoc_phasing`), and write this scan's
`PerIntegration` θ slots (disjoint per scan — concurrent groups may solve in
parallel). The feed tying and the priors come from `adhoc_plan`, so the number
of phase nodes per station is the model's choice and needs no separate
argument.

Returns the prior each track was fit under, over `(AntennaName, Feed, Ti)` with the
scan's first AP epoch as its one `Ti` value.
"""
function adhoc_scan!(
        θ, group::XRadio.ProcessingSet, geom::DataGeometry, adhoc_plan, adhoc, gauge;
        executor = DynamicScheduler(),
    )
    (; rbar, wbar, ti) = _ap_sums(group, geom; executor)
    as = solve_adhoc_phasing(
        rbar, wbar, geom.stations;
        gauge = _gauge_for(gauge, adhoc_plan.path), smoother = adhoc, tying = adhoc_plan.tying,
        prior = map(p -> _prior_along(p, :Ti), adhoc_plan.priors),
    )
    adhoc_leaf = _component_leaf(adhoc_plan, θ)
    for gti in ti
        tseg = adhoc_plan.tseg_id[gti]
        at_t = view(as.phase, Ti(At(geom.times[gti])))
        for (ant, name) in pairs(geom.stations), feed in lookup(at_t, Feed)
            val = at_t[AntennaName(At(name)), Feed(At(feed))]
            isfinite(val) || continue
            node = _feed_node(adhoc_plan.tying, feed)
            node == 0 && continue
            adhoc_leaf[1, node, 1, tseg, ant] = val
        end
    end
    t0 = minimum(gti -> geom.times[gti], ti)
    return cat(as.prior; dims = Ti([t0]))
end

# Smoothers see each AP centered on the cells covered throughout the scan, a
# frame that depends on neither the per-AP solves' anchor nor the user's gauge.
# A track smoothed relative to one station carries that station's noise, which
# per-track priors do not pass through equally, so the smoothed baseline
# differences would depend on the station.
function _smooth_tracks!(smoother, phase, covered, track_w; obs, priors, resolved)
    _apply_ap_gauge!(phase, covered, track_w, ZeroSumPhase())
    return apply_adhoc!(smoother, phase, track_w; obs, priors, resolved)
end

# Put each AP on the gauge's own convention. The per-AP common mode is
# unobservable — it cancels on every baseline — so this changes how the tracks
# read, never the applied correction. One constant per AP spans every feed
# node, so differences between nodes in one component are left untouched.
#
# The constraint is built once, over the cells covered in every AP that has any
# coverage, and weighted by their scan totals: a constraint over whatever
# happens to be covered moves frame with coverage (a pin falling back to a
# different node per AP), putting steps into every track for a quantity that
# carries no information. Each AP is shifted wherever its phase is finite.
function _apply_ap_gauge!(phase, covered, track_w, gauge::AbstractGauge)
    T = eltype(phase)
    aps = [ap for ap in axes(covered, 3) if any(view(parent(covered), :, :, ap))]
    cells = CartesianIndices((axes(covered, 1), axes(covered, 2)))
    always = [I for I in cells if all(ap -> covered[I, ap], aps)]
    (isempty(aps) || isempty(always)) && return phase
    f = GaugeFreedom(;
        nodes = LinearIndices(cells)[always], station = [I[1] for I in always], feed = [I[2] for I in always],
        scan = zeros(Int, length(always)), component = fill((), length(always)),
        observable = fill(:phase, length(always)), direction = ones(T, length(always)),
        weight = T[sum(ap -> track_w[I, ap], aps) for I in always],
    )
    C, d = _gauge_system(gauge, GaugeFreedoms{T}([f], length(cells)))
    row = T[C[1, n] for n in f.nodes]
    for ap in axes(phase, 3)
        x = view(parent(phase), :, :, ap)
        vals = x[f.nodes]
        all(isfinite, vals) || continue
        c = (sum(row .* vals) - d[1]) / sum(row)
        x[isfinite.(x)] .-= c
    end
    return phase
end

# Re-reference anchor-absent APs to the trusted frame. Each connected component
# of an AP's gated graph carries one additive freedom, which that AP's solve pins
# at an anchor node when the component holds one and at some other node when it
# does not. A component without an anchor node is offset by an arbitrary,
# non-2π constant δ, estimated as the weighted circular mean of its cells'
# change from their last trusted values and subtracted from them. Every
# component then in the trusted frame refreshes those values, so real drift
# propagates.
# APs before any trusted value exists are left untouched; per-(station, node)
# means and slopes are left to `_detrend_track!`.
function _restitch_refant_gauge!(phase, covered, track_w, obs, ref_station::Integer)
    T = eltype(phase)
    nant, nnode = size(phase, 1), size(phase, 2)
    trusted = fill(T(NaN), nant, nnode)
    for ap in axes(phase, 3)
        mk = view(obs.mask, Ti(ap))
        edges = [_edge(obs.nodes[I], nant) for I in eachindex(mk, obs.nodes) if mk[I]]
        compid, ncomp, _ = connected_components(nant * nnode, edges)
        comp(a, n) = compid[_node(a, n, nant)]
        framed = falses(ncomp)
        for n in 1:nnode
            c = comp(ref_station, n)
            c > 0 && covered[ref_station, n, ap] && (framed[c] = true)
        end
        for c in 1:ncomp
            framed[c] && continue
            num = zero(Complex{T})
            for a in 1:nant, n in 1:nnode
                (comp(a, n) == c && covered[a, n, ap] && isfinite(trusted[a, n]) && isfinite(phase[a, n, ap])) || continue
                wk = (isfinite(track_w[a, n, ap]) && track_w[a, n, ap] > 0) ? T(track_w[a, n, ap]) : one(T)
                num += wk * cis(phase[a, n, ap] - trusted[a, n])
            end
            iszero(num) && continue
            δ = angle(num)
            for a in 1:nant, n in 1:nnode
                comp(a, n) == c && covered[a, n, ap] && (phase[a, n, ap] -= δ)
            end
            framed[c] = true
        end
        for a in 1:nant, n in 1:nnode
            c = comp(a, n)
            (c > 0 && framed[c] && covered[a, n, ap] && isfinite(phase[a, n, ap])) || continue
            trusted[a, n] = phase[a, n, ap]
        end
    end
    return phase
end

# Remove the weighted mean of a track: the gauge between a station's track and
# the source terms, which a per-station constant trades against. With
# `source_iters = 1` there are no source terms and the constant is measured;
# it is removed all the same and left to the fringe step's per-scan `atmos`.
# The slope is kept: it is the residual rate the fringe step's `Rate` left.
# Returns the mean removed, zero for a track without finite values.
function _detrend_track!(track::AbstractVector, w::AbstractVector)
    T = eltype(track)
    idx = [i for i in eachindex(track) if isfinite(track[i])]
    length(idx) >= 1 || return zero(T)
    ws = [(isfinite(w[i]) && w[i] > 0) ? T(w[i]) : one(T) for i in idx]
    m = sum(ws .* track[idx]) / sum(ws)
    for i in idx
        track[i] -= m
    end
    return m
end
