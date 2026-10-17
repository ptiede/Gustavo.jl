# ── Per-feed stationization with closure ─────────────────────────────────────
#
# Weighted least squares turning per-baseline fringe detections into
# per-(station, feed) delays and rates. The incidence, the use of every
# correlation product, the row weights and the gauge are derived in
# `docs/src/fringe_fitting.md`.
#
# The per-antenna response is taken as diagonal, so leakage goes to the
# residuals, and there is no parallactic-angle term. Modeling either means
# adding a component; neither is checked for here.

# ── Robust losses ───────────────────────────────────────────────────────────
#
# The station solves downweight inconsistent rows through a robust loss ρ on
# the normalized squared residual u = (z/f)², where z = resid·√w and f is the
# loss scale. Because `w` is the noise model's inverse variance (`_row_weight`),
# z is already in units of predicted σ and u is dimensionless. The scale must
# not be re-estimated from the residuals themselves: that makes the effective
# threshold independent of how many rows are in the system, whereas a fitted
# scale shrinks with the degrees of freedom, tightening a nominal cut on
# exactly the weak scans that can least afford it.
#
# Each loss supplies `robust_weight(loss, u) = ρ′(u)`, the IRLS multiplier on a
# row's weight. Conventions follow SciPy's `least_squares` loss family, which is
# where EHT-HOPS's `soft_l1` choice comes from.

"""
    AbstractRobustLoss

Robust loss family for the station solves — how strongly a row inconsistent
with the closing solution is downweighted. Members: [`LeastSquares`](@ref),
[`SoftL1`](@ref), [`Huber`](@ref), [`Cauchy`](@ref). Selected through
[`Stationization`](@ref)'s `loss` field, scaled by its `loss_scale`.
"""
abstract type AbstractRobustLoss end

"""
    LeastSquares()

No robust downweighting: every row keeps its full noise-model weight, and
the solve is plain weighted least squares.
"""
struct LeastSquares <: AbstractRobustLoss end

"""
    SoftL1()

Smooth L1: `ρ(u) = 2(√(1+u) − 1)`. Quadratic for `u ≪ 1` and linear far out, so
a grossly inconsistent row contributes a bounded gradient instead of dominating
the fit. The default, and EHT-HOPS's choice.
"""
struct SoftL1 <: AbstractRobustLoss end

"""
    Huber()

`ρ(u) = u` for `u ≤ 1`, `2√u − 1` beyond: full weight inside the scale, L1
outside. Sharper than [`SoftL1`](@ref) at the transition.
"""
struct Huber <: AbstractRobustLoss end

"""
    Cauchy()

`ρ(u) = ln(1 + u)`. Redescending — weight falls off as `1/u`, so a far outlier
is suppressed much harder than under [`SoftL1`](@ref) or [`Huber`](@ref), at
the cost of a loss that is no longer convex in the residual.
"""
struct Cauchy <: AbstractRobustLoss end

# ρ′(u): the IRLS weight multiplier at normalized squared residual `u ≥ 0`.
robust_weight(::LeastSquares, u::Real) = one(u)
robust_weight(::SoftL1, u::Real) = inv(sqrt(1 + u))
robust_weight(::Huber, u::Real) = u <= 1 ? one(u) : inv(sqrt(u))
robust_weight(::Cauchy, u::Real) = inv(1 + u)

"""
    Stationization(; eltype = Float64, pfa_max, weak_sys_scale, loss, loss_scale,
                     irls_iters, systematic_delay, systematic_rate)
    Stationization{T}(; kw...)

Options for [`solve_station_systems!`](@ref).

`eltype` (or `T`) is the floating-point type the station systems are built and
solved in, whatever the detections' own type. The default `Float64` is
deliberate: weak rows sit ~1e-5 below accepted ones in `√w`, near `Float32`'s
rank tolerance.

`pfa_max` is the solve's one detection threshold, read against each
detection's family-wise false-alarm probability (see [`Detection`](@ref)).
It governs connectivity alone: a detection at or below it joins its two
stations into one fringe group, and a (station, scan) is calibrated only
when such a detection reaches it. A detection above it still contributes its
row — with every systematic floor multiplied by `weak_sys_scale` — but never
joins stations into a group, so a marginal baseline is measured at the
fringe location the accepted detections fixed without being able to invent
one.

Every correlation product contributes a row to the delay and rate systems;
which parameters a row touches is set by the model's feed tying.

`loss`/`loss_scale`/`irls_iters` control robust downweighting: after each
solve, each row's weight is rescaled by
`robust_weight(loss, (z/loss_scale)^2)` at its noise-normalized residual
`z = resid·√w`, and the system re-solved, up to `irls_iters` times. A false
fringe is closure-inconsistent, lands far out in `z`, and is suppressed.
Weights come from the noise model rather than the residual spread, so
`loss_scale` cuts at the same effective threshold whatever the system size.
`loss = LeastSquares()` disables downweighting.

`systematic_delay`/`systematic_rate` (seconds / Hz) are floors added in
quadrature to a row's CRB uncertainty: `w = 1/(σ_CRB² + systematic²)`; without
one, data with no real systematics drives `z` to the numerical noise floor and
the loss reweights rounding noise. `weak_sys_scale` multiplies the total σ of
an above-threshold row, floor included.
"""
struct Stationization{T <: AbstractFloat, L <: AbstractRobustLoss}
    pfa_max::T
    weak_sys_scale::T
    loss::L
    loss_scale::T
    irls_iters::Int
    systematic_delay::T
    systematic_rate::T
end

Stationization(; eltype::Type{<:AbstractFloat} = Float64, kw...) = Stationization{eltype}(; kw...)

function Stationization{T}(;
        pfa_max = 1.0e-4, weak_sys_scale = 1.0e3,
        loss = SoftL1(), loss_scale = 8.0, irls_iters = 5,
        systematic_delay = 0.0, systematic_rate = 0.0,
    ) where {T}
    return Stationization{T, typeof(loss)}(
        pfa_max, weak_sys_scale, loss, loss_scale, irls_iters, systematic_delay, systematic_rate,
    )
end

# IRLS weight update. `w` (mutated) is the effective weight vector fed to the
# solver, `w0` the untouched noise-model weights that define the σ scale.
# Returns whether any weight moved enough to be worth another solve.
function _irls_weights!(w, w0, loss::AbstractRobustLoss, scale::Real, resid)
    changed = false
    f2 = scale^2
    for i in eachindex(w, w0, resid)
        u = resid[i]^2 * w0[i] / f2
        wi = w0[i] * robust_weight(loss, u)
        abs(wi - w[i]) > 1.0e-12 * w0[i] && (changed = true)
        w[i] = wi
    end
    return changed
end

# ── Row weights: the noise model, in each observable's own units ────────────
#
# Every row's weight is `1/σ²` with σ the CRB uncertainty of that measurement,
# derived in `docs/src/fringe_fitting.md`, plus a systematic floor in
# quadrature. The delay and rate σ carry the array's frequency and time extent.
#
# The spreads matter only for the robust loss. A weighted least-squares solution
# is invariant under scaling every weight in a system by a common constant, and
# σ_ν/σ_t are common to all rows of one scan, so getting them wrong cannot move
# the fit. What they fix is the meaning of `z = resid·√w`: with a true inverse
# variance, z is in units of σ and `loss_scale` is the same dimensionless number
# for both observables, matching EHT-HOPS's `f_scale = 8` on
# `(data − model)/err`.

# RMS spread of a coordinate about its mean — the CRB lever arm. For channels
# uniformly filling a band of width B this is B/√12, HOPS's `√12` factor; taking
# the actual spread instead generalizes correctly to sparse or unevenly spaced
# bands (VGOS), where assuming contiguity overstates the lever arm.
function _rms_spread(xs)
    n = length(xs)
    n > 1 || return 0.0
    μ = sum(xs) / n
    return sqrt(sum(x -> (x - μ)^2, xs) / n)
end

# Statistical σ of one observable, from the CRB.
_sigma_stat(kind::Symbol, snr::Real, σ_ν::Real, σ_t::Real) =
    kind === :delay ? inv(2π * σ_ν * snr) : inv(2π * σ_t * snr)

# Inverse-variance weight: statistical σ with a systematic floor in quadrature,
# the whole σ then inflated by `scale` (1 for an accepted detection, and
# `weak_sys_scale` for one above `pfa_max` — see `Stationization`).
_row_weight(σ_stat::Real, σ_sys::Real, scale::Real = 1.0) =
    inv(scale^2 * (σ_stat^2 + σ_sys^2))

# The scan geometry a delay/rate weight needs, or an error naming what is
# missing. A robust loss cannot normalize a delay residual without knowing the
# band it was measured over: the residual is in seconds and the threshold is in
# σ. Rather than silently leaving those rows at full weight — which reads as
# "robust" while downweighting nothing — refuse the combination outright.
function _require_spread(spread, kind::Symbol, loss::AbstractRobustLoss)
    (!isnothing(spread) && spread > 0) && return float(spread)
    loss isa LeastSquares && return 1.0     # unused: a common factor cancels
    what = kind === :delay ? ("freq_rms", "RMS frequency spread (Hz)") :
        ("time_rms", "RMS time spread (s)")
    throw(
        ArgumentError(
            "a $(nameof(typeof(loss))) loss on the $kind system needs the scan's " *
                "$(what[2]) to express residuals in units of σ, but `$(what[1])` is " *
                "$(isnothing(spread) ? "missing" : "$spread"). Supply it (see " *
                "`detection_stack`/`solve_station_systems!`), or set " *
                "`Stationization(loss = LeastSquares())` to weight rows by the " *
                "noise model alone.",
        ),
    )
end

# ── Generic, model-driven station solve ─────────────────────────────────────
#
# `solve_station_systems!` has no node space of its own: the unknowns are the θ
# columns the model declares. For each stage-B phase component,
# `_block_index(plan, _feed_node(tying, feed), 1, tseg_id[ti], ant)` is the θ
# slot a (station, feed, time) observation maps to. The plan's segmentation
# encodes the time basis — `PerScan` gives a distinct column per scan,
# `GlobalTime` one column shared across the track — and its tying the feed fold.
# So the same engine solves a per-scan model, whose columns are disjoint per
# scan and therefore block-diagonal, and a model with a track-global inter-feed
# offset, whose shared column couples scans. The model is the extension point;
# this solver only reads the θ columns each component declares.
#
# Every correlation product's detection becomes a row of the delay and rate
# systems, whatever feeds it relates: the tying alone decides what a row
# touches. Under `default_fringe_terms` the rows relating different feeds are
# what constrain `rel_delay`'s common mode.
#
# `scans` is a vector of `AntennaPair × FeedPair` Detection `DimStack`s, the shape
# `search_scan` returns, read by label: each antenna pair names its stations,
# which `stations` numbers, and a representative global time index `:ti` in
# metadata selects the `tseg_id`. θ slots are accumulated into
# with `+=`, so `rounds > 1` — search on the residual — adds each round's
# increment to the last.

"""
    detection_stack(D::AbstractMatrix{<:Detection}, antenna_pairs, feeds;
                    ti, freq_rms, time_rms) -> DimStack

Package a plain `[antenna pair, feed pair]` detection matrix as the
`AntennaPair × FeedPair` DimStack shape `search_scan` returns, labeled by the
antenna-name pairs `antenna_pairs` and the feed-index pairs `feeds` (see
[`feed_pairs`](@ref)), carrying `ti` (the representative global time index) in
metadata — so a scan built directly (the refine stage, or a
direct `solve_station_systems!` call) has the same shape as
one that came from the search, and every consumer reads pairs/feeds/ti off the
stack uniformly.
"""
function detection_stack(
        D::AbstractMatrix{<:Detection}, antenna_pairs, feeds;
        ti::Integer,
        freq_rms::Union{Nothing, Real} = nothing,
        time_rms::Union{Nothing, Real} = nothing,
    )
    gdims = (_station_pair_dim(collect(antenna_pairs)), FeedPair(collect(feeds)))
    layers = (;
        delay = DimArray(getfield.(D, :delay), gdims),
        rate = DimArray(getfield.(D, :rate), gdims),
        phase = DimArray(getfield.(D, :phase), gdims),
        amp = DimArray(getfield.(D, :amp), gdims),
        snr = DimArray(getfield.(D, :snr), gdims),
        pfa = DimArray(getfield.(D, :pfa), gdims),
        valid = DimArray(getfield.(D, :valid), gdims),
    )
    return DimensionalData.DimStack(
        layers; metadata = _scan_meta(ti, freq_rms, time_rms),
    )
end

# A detection stack's scan-level provenance: the representative global time
# index and the RMS frequency/time spreads the weights need.
_scan_meta(ti, freq_rms, time_rms) = (; ti = Int(ti), freq_rms, time_rms)

# Attach a representative global time index to an existing detection stack (the
# search's own `search_scan` return, which carries no `:ti` — the caller knows
# which window it searched).
_with_ti(
    stack::AbstractDimStack, ti::Integer;
    freq_rms::Union{Nothing, Real} = nothing, time_rms::Union{Nothing, Real} = nothing,
) = DimensionalData.rebuild(stack; metadata = _scan_meta(ti, freq_rms, time_rms))

_scan_pairs(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, AntennaPair))
_scan_feeds(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, FeedPair))

_station_slot(slot, name) = get(slot, name) do
    throw(ArgumentError("detection antenna `$name` is not among the stations " * join(keys(slot), ", ")))
end
_scan_ti(sc::AbstractDimStack) = DimensionalData.metadata(sc).ti
_scan_spread(sc::AbstractDimStack, key::Symbol) = getproperty(DimensionalData.metadata(sc), key)

"""
    solve_station_systems!(θ, scans, components, stations; gauge, opts) -> (ncomp, covered, unlinked)

Solve the stage-B fringe systems (delay and rate) over `scans` and
accumulate the per-(station, feed) values into `θ` at the columns the model
declares. `stations` are the station names the layout numbers; each
detection's antenna pair is matched to them by name. `components` is a vector
of `(plan::ComponentPlan, kind::Symbol)` with
`kind ∈ (:delay, :rate)`. Multiple components of the same kind are summed
per (station, feed) observation: e.g. a feed-common `PerScan × SharedFeeds` term
plus a `GlobalTime × ExceptFeed(1)` inter-feed offset both feed the delay
system, so a row on feed `k ≠ 1` touches both columns and a stable inter-feed offset is solved
once across the track (bright scans pin it; weak scans inherit it, tying feeds
that would otherwise split). Returns the delay system's component count and
`covered` — the `(station name, feed, scan index)` triples the solve CONSTRAINS
in every solved kind: those an accepted detection (`pfa ≤ pfa_max`) touches on
that feed. Coverage is independent of the gauge (see `Stationization` for how
inconsistent rows are weighted). A θ column no accepted detection touches —
per-scan or scan-spanning, per-feed or shared by every feed — is set to zero
(identity gain). With a single per-scan/per-feed component per kind and one scan,
each scan's system is independent and solves exactly as it would alone.

`unlinked` holds the `(kind, scan index)` pairs in which the accepted
detections leave the offset between some station's feeds free: no accepted
detection relates different feeds of that fringe group. The gauge then sets
it, at the gauge's preferred station, so it is not measured.
"""
function solve_station_systems!(
        θ::AbstractVector, scans, components, stations;
        gauge::AbstractGauge, opts::Stationization{T} = Stationization(),
    ) where {T}
    # θ columns are numbered from 1 (`_block_index`).
    Base.require_one_based_indexing(θ)
    slot = Dict(n => i for (i, n) in pairs(stations))
    ncomp = 0
    # (station, feed, scan-index) triples the solve constrains, intersected over
    # the solved kinds: a station's feed must be constrained in delay and rate to
    # count as calibrated. Everything else is flagged downstream.
    covered = Set{Tuple{Int, Int, Int}}()
    unconstrained = Set{Int}()
    unlinked = Set{Tuple{Symbol, Int}}()
    first_kind = true
    for kind in (:delay, :rate)
        plans = [c[1] for c in components if c[2] === kind]
        isempty(plans) && continue
        nc, cov, solved, constrained, free_scans = _solve_kind_cols!(θ, scans, plans, slot, gauge, opts, Val(kind))
        union!(unlinked, ((kind, si) for si in free_scans))
        union!(unconstrained, setdiff(keys(solved), constrained))
        covered = first_kind ? cov : intersect(covered, cov)
        first_kind = false
        kind === :delay && (ncomp = nc)
    end
    # Columns no accepted row touches hold values set by weak rows alone.
    for col in unconstrained
        θ[col] = zero(eltype(θ))
    end
    return ncomp, Set{Tuple{eltype(stations), Int, Int}}((stations[a], f, si) for (a, f, si) in covered), unlinked
end

# Solve one observable kind across all scans, accumulating into θ. Each detection
# becomes a station-difference row whose a-/b-side touch the sum of all `plans`'
# θ columns for that (station, feed, time) — a feed-common per-scan column and,
# when present, a global feed-offset column. Returns (ncomp, covered, solved,
# constrained): `solved` maps each θ column this kind touched to the value it
# just added, and `constrained` holds the θ columns an accepted row touches.
#
# `kind` is a type parameter so every kind-dependent choice below is made at
# compile time; the caller's `Val(kind)` is the only dynamic dispatch.
function _solve_kind_cols!(
        θ::AbstractVector, scans, plans, slot, gauge::AbstractGauge, opts::Stationization{T},
        ::Val{kind},
    ) where {T, kind}
    getval(d) = getfield(d, kind)
    sys = kind === :delay ? opts.systematic_delay : opts.systematic_rate
    spread_key = kind === :delay ? :freq_rms : :time_rms

    colnode = Dict{Int, Int}()               # θ column → local node id
    node_col = Int[]                         # local node → θ column
    node_feed = Int[]                        # exclusive feed, or 0 if a column is shared by several feeds
    node_station = Int[]
    node_scan = Int[]                        # scan id, or 0 if a column spans scans (global)
    node_path = Tuple{Vararg{Symbol}}[]      # the owning component's path
    function getnode(col, st, fd, sidx, path)
        if haskey(colnode, col)
            n = colnode[col]
            node_feed[n] == fd || (node_feed[n] = 0)        # touched by several feeds → shared
            node_scan[n] == sidx || (node_scan[n] = 0)      # spans scans → global
            return n
        end
        push!(node_col, col); push!(node_feed, fd); push!(node_station, st); push!(node_scan, sidx)
        push!(node_path, path)
        return colnode[col] = length(node_col)
    end

    # Rows in θ-column space: each side is the list of θ columns whose sum is
    # that station's value for this observable (+1 on a-side, −1 on b-side).
    # `rlinks` pairs each column with the same component's column on the other
    # side; they define the gauge components (see `_solve_tagged_system`).
    rowA = Vector{Int}[]; rowB = Vector{Int}[]; rlinks = Vector{Tuple{Int, Int}}[]
    rval = T[]; rw = T[]; rscan = Int[]
    rsta_a = Int[]; rsta_b = Int[]; rfeed_a = Int[]; rfeed_b = Int[]
    # Per row: whether its detection is real (`pfa <= pfa_max`). Only these
    # connect stations into a fringe group; the rest constrain and no more.
    raccept = Bool[]
    for (sidx, sc) in enumerate(scans)
        bl_pairs = map(_scan_pairs(sc)) do (a, b)
            (_station_slot(slot, a), _station_slot(slot, b))
        end
        feeds = _scan_feeds(sc)
        ti = _scan_ti(sc)
        # Per scan, not per row: the band and duration are properties of the
        # observation, so every row of one scan shares this lever arm.
        σ = T(_require_spread(_scan_spread(sc, spread_key), kind, opts.loss))
        σν = kind === :delay ? σ : one(T)
        σt = kind === :rate ? σ : one(T)
        for bi in eachindex(bl_pairs), p in eachindex(feeds)
            det = sc[AntennaPair(bi), FeedPair(p)]
            det.valid || continue                   # no data in this cell, no measurement
            a, b = bl_pairs[bi]
            a == b && continue
            fa, fb = feeds[p]
            accept = det.pfa <= opts.pfa_max
            nsA = Int[]; nsB = Int[]; links = Tuple{Int, Int}[]
            for plan in plans
                na = _feed_node(plan.tying, fa)
                ca = na == 0 ? 0 : _block_index(plan, na, 1, plan.tseg_id[ti], a)
                ca != 0 && push!(nsA, getnode(ca, a, fa, sidx, plan.path))
                nb = _feed_node(plan.tying, fb)
                cb = nb == 0 ? 0 : _block_index(plan, nb, 1, plan.tseg_id[ti], b)
                cb != 0 && push!(nsB, getnode(cb, b, fb, sidx, plan.path))
                ca != 0 && cb != 0 && push!(links, (last(nsA), last(nsB)))
            end
            (isempty(nsA) || isempty(nsB)) && continue
            push!(rowA, nsA); push!(rowB, nsB); push!(rlinks, links)
            push!(rval, getval(det))
            push!(rw, _row_weight(_sigma_stat(kind, det.snr, σν, σt), sys, accept ? one(T) : opts.weak_sys_scale))
            push!(raccept, accept)
            push!(rscan, sidx)
            push!(rsta_a, a); push!(rsta_b, b); push!(rfeed_a, fa); push!(rfeed_b, fb)
        end
    end
    isempty(rowA) && return (0, Set{Tuple{Int, Int, Int}}(), Dict{Int, T}(), Set{Int}(), Set{Int}())

    # Robust solve: IRLS over `opts.loss`, rescaling each row's noise-model
    # weight by the loss's derivative at that row's normalized residual (see
    # `Stationization`). No row leaves the system, so the graph's connectivity,
    # and hence which stations are solvable, is fixed before the first solve.
    w = copy(rw)
    rows = (;
        A = rowA, B = rowB, links = rlinks, val = rval, accept = raccept,
        sta_a = rsta_a, feed_a = rfeed_a, sta_b = rsta_b, feed_b = rfeed_b, scan = rscan,
    )
    nodes = (; feed = node_feed, station = node_station, scan = node_scan, component = node_path)
    x, ncomp, resid = _solve_tagged_system(rows, w, nodes, gauge, kind)
    if !(opts.loss isa LeastSquares)
        for _ in 1:max(opts.irls_iters, 0)
            _irls_weights!(w, rw, opts.loss, opts.loss_scale, resid) || break
            x, ncomp, resid = _solve_tagged_system(rows, w, nodes, gauge, kind)
        end
    end
    solved = Dict{Int, T}()
    for n in eachindex(node_col)
        θ[node_col[n]] += x[n]
        solved[node_col[n]] = x[n]
    end
    constrained = Set{Int}(
        node_col[n] for i in eachindex(rowA, rowB, raccept) if raccept[i]
            for n in Iterators.flatten((rowA[i], rowB[i]))
    )
    covered = _covered_station_feeds(rsta_a, rfeed_a, rsta_b, rfeed_b, rscan, raccept)
    A = _incidence(rows, length(node_col), T)
    return ncomp, covered, solved, constrained, _unlinked_scans(A, rows)
end

# The (station, feed, scan) triples this system calibrates: those carrying at
# least one accepted detection on that feed. Everything else is flagged
# downstream.
#
# Acceptance must be read from `raccept`, not from the rows: every measured cell
# contributes a row, so reading coverage off the rows alone would call a station
# calibrated on the strength of a noise peak, which sits at an arbitrary delay
# and fixes nothing. Only a detection at or below `pfa_max` joins stations into
# a fringe group.
#
# Coverage does not depend on the reference antenna, and so is invariant to the
# gauge pin: a correction enters the data only as the difference `g_a − g_b`
# along a baseline, and a baseline exists only within a connected component, so
# a component's gauge cancels wherever it is applied. Requiring reference
# connectivity instead would discard every station of a scan the reference sits
# out, including scans whose own closure is well determined.
#
# `pfa_max` decides which rows are real detections and therefore what the graph
# looks like; the robust loss then arbitrates inconsistency among those rows
# without removing any. A station with no accepted detection is uncalibrated
# however the surviving rows are weighted.
function _covered_station_feeds(sta_a, feed_a, sta_b, feed_b, rscan, raccept)
    cov = Set{Tuple{Int, Int, Int}}()
    for i in eachindex(sta_a, feed_a, sta_b, feed_b, rscan, raccept)
        raccept[i] || continue
        push!(cov, (sta_a[i], feed_a[i], rscan[i]))
        push!(cov, (sta_b[i], feed_b[i], rscan[i]))
    end
    return cov
end

# Constrained WLS over a tagged node graph, the column-space generalization of
# `_solve_observable!`. Row `i` observes `Σ x[rows.A[i]] − Σ x[rows.B[i]]` with
# weight `w[i]`; `nodes` tags each local node with its station, feed (0 = shared
# by several feeds), scan (0 = a column spanning scans) and component path.
# Returns the solution, the number of gauged components the accepted rows form,
# and the residuals.
#
# The gauge: `rows.links` join each column to the same component's column on
# the other side of a row. A component of those links whose columns can all
# shift together without changing any row it must hold is a gauge freedom, and
# gets one constraint row from `gauge`. That gives one per scan for per-scan
# columns, however columns spanning scans couple them, and one for a feed
# offset that no row relating different feeds fixes. A freedom that shifts
# columns of several components at once is pinned by `_pin_leftover_freedoms`.
# Any other freedom is an error.
function _solve_tagged_system(rows, w, nodes, gauge::AbstractGauge, kind::Symbol)
    T = eltype(rows.val)
    nnodes = length(nodes.feed)
    compid, ncomp, nfree = _gauge_components(rows, nodes)

    # Total row weight on each node — the score the pin falls back to when a
    # component holds no reference node.
    nodew = _node_weights(nnodes, zip(rows.A, rows.B), w)

    freedoms = map(1:nfree) do c
        cn = [n for n in eachindex(compid) if compid[n] == c]
        GaugeFreedom(;
            nodes = cn, station = nodes.station[cn], feed = nodes.feed[cn], scan = nodes.scan[cn],
            component = nodes.component[cn], observable = fill(kind, length(cn)),
            direction = ones(T, length(cn)), weight = nodew[cn],
        )
    end
    A = _incidence(rows, nnodes, T)
    C, d = _gauge_system(gauge, GaugeFreedoms{T}(freedoms, nnodes))
    # The null-space test and the constrained factorization work on dense rows.
    C = Matrix(C)
    C, d = _pin_leftover_freedoms(C, d, A, rows, nodew, gauge)
    solve_system = try
        ConstrainedWLS(A, w, C, d)
    catch err
        err isa ArgumentError || rethrow()
        throw(
            ArgumentError(
                "the $kind station system is not determined after its gauge ($(err.msg)): " *
                    "the detections leave a combination of the model's columns free, such " *
                    "as two components of this observable that every detection sees only " *
                    "as their sum.",
            ),
        )
    end
    x = solve_system(rows.val)
    return x, ncomp, rows.val .- A * x
end

# Row `i` observes `Σ x[rows.A[i]] − Σ x[rows.B[i]]`.
function _incidence(rows, nnodes::Integer, ::Type{T}) where {T}
    A = zeros(T, length(rows.A), nnodes)
    for i in eachindex(rows.A, rows.B)
        for n in rows.A[i]
            A[i, n] += one(T)
        end
        for n in rows.B[i]
            A[i, n] -= one(T)
        end
    end
    return A
end

# The rows a gauge must leave unchanged: the accepted ones and those touching a
# column no accepted row touches, as `_gauge_components` holds islands to
# every row.
function _held_rows(rows)
    touched = Set(n for i in eachindex(rows.A, rows.B, rows.accept) if rows.accept[i] for n in Iterators.flatten((rows.A[i], rows.B[i])))
    return [rows.accept[i] || any(n -> n ∉ touched, Iterators.flatten((rows.A[i], rows.B[i]))) for i in eachindex(rows.A, rows.B, rows.accept)]
end

# Each (station, feed, scan) a row side measures, and the columns whose sum is
# its value.
function _row_sides(rows)
    sides = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
    for i in eachindex(rows.A, rows.B)
        sides[(rows.sta_a[i], rows.feed_a[i], rows.scan[i])] = rows.A[i]
        sides[(rows.sta_b[i], rows.feed_b[i], rows.scan[i])] = rows.B[i]
    end
    return sides
end

# The scans in which a direction the held rows leave free changes the
# difference between two feeds of one station: the detections do not link that
# station's feeds, so only a gauge can set the offset between them.
function _unlinked_scans(A, rows)
    T = eltype(A)
    N = nullspace(A[_held_rows(rows), :])
    scans = Set{Int}()
    size(N, 2) == 0 && return scans
    feeds = Dict{Tuple{Int, Int}, Vector{Pair{Int, Vector{Int}}}}()
    for ((st, f, sc), cols) in _row_sides(rows)
        push!(get!(feeds, (st, sc), Pair{Int, Vector{Int}}[]), f => cols)
    end
    tol = sqrt(eps(T))
    for ((_, sc), fs) in feeds
        sort!(fs; by = first)
        for k in 2:length(fs)
            diff = zeros(T, size(A, 2))
            diff[last(fs[k])] .+= one(T)
            diff[last(fs[k - 1])] .-= one(T)
            maximum(abs, N' * diff) > tol && push!(scans, sc)
        end
    end
    return scans
end

# The directions the rows that must hold leave free after the gauge rows `C`,
# each fixed by setting one (station, feed, scan) value to zero. Such a
# direction moves columns of several components together: without products
# relating different feeds, a station whose feed order differs from the
# reference's has its feeds in different groups than the inter-feed offset
# columns assume. Values are tried in the gauge's station order, then by row
# weight, lowest feed first, and one is kept only if it fixes a further
# direction. A direction no value fixes is left for the solve to reject.
function _pin_leftover_freedoms(C, d, A, rows, nodew, gauge::AbstractGauge)
    T = eltype(C)
    K = nullspace(vcat(A[_held_rows(rows), :], C))
    size(K, 2) == 0 && return C, d

    sides = _row_sides(rows)
    weight = Dict{Int, T}()
    for ((st, _, _), cols) in sides
        weight[st] = get(weight, st, zero(T)) + sum(n -> nodew[n], cols)
    end
    preferred = collect(gauge_station_order(gauge))
    rank_of(st) = (something(findfirst(==(st), preferred), length(preferred) + 1), -weight[st], st)
    keys_in_order = sort!(collect(keys(sides)); by = ((st, f, sc),) -> (rank_of(st), sc, f))

    # A pin's entries are 0 or 1 and `K` is orthonormal, so a direction it fixes
    # shows up at order one; anything near rounding is no direction at all.
    tol = sqrt(eps(T))
    pins = zeros(T, 0, size(C, 2))
    fixed = 0
    for key in keys_in_order
        r = zeros(T, 1, size(C, 2))
        r[sides[key]] .= one(T)
        trial = vcat(pins, r)
        rank(trial * K; atol = tol) > fixed || continue
        pins = trial
        fixed += 1
        fixed == size(K, 2) && break
    end
    return vcat(C, pins), vcat(d, zeros(T, size(pins, 1)))
end

# The gauge components of a tagged system, numbered so that the gauged ones come
# first: `compid` (0 for a node in none), the number of gauged components the
# accepted rows form, and the number gauged in all.
#
# Components are built from accepted rows only. A detection above `pfa_max`
# sits at an arbitrary noise peak, so letting it define graph structure would
# hand the component count and the gauge pins to noise. Acceptance is a hard
# connectivity cut: where weak rows bridge two accepted components, each keeps
# its own gauge row, and those rows fix the offset between them. Nodes no
# accepted row links form islands over all rows, gauged the same way;
# `_covered_station_feeds` excludes them from coverage.
function _gauge_components(rows, nodes)
    nnodes = length(nodes.feed)
    accepted = (l for i in eachindex(rows.links, rows.accept) if rows.accept[i] for l in rows.links[i])
    compid, nacc, _ = connected_components(nnodes, accepted)
    unlinked = (l for ls in rows.links for l in ls if compid[l[1]] == 0 && compid[l[2]] == 0)
    island, nisl, _ = connected_components(nnodes, unlinked)
    for n in eachindex(compid, island)
        compid[n] == 0 && island[n] != 0 && (compid[n] = nacc + island[n])
    end
    free = _shift_invariant(compid, nacc + nisl, rows, c -> c > nacc)
    # Renumber: gauged accepted components, then gauged islands, then the rest.
    order = [findall(c -> free[c] && c <= nacc, 1:(nacc + nisl)); findall(c -> free[c] && c > nacc, 1:(nacc + nisl)); findall(!, free)]
    rank = invperm(order)
    for n in eachindex(compid)
        compid[n] == 0 || (compid[n] = rank[compid[n]])
    end
    return compid, count(c -> free[c], 1:nacc), count(free)
end

# Whether shifting each component's columns by a common constant leaves every
# row unchanged that the component must hold: accepted rows for all, and weak
# rows too for the components `all_rows(c)` selects.
function _shift_invariant(compid, ncomp::Integer, rows, all_rows)
    free = trues(ncomp)
    tally = Dict{Int, Int}()
    for i in eachindex(rows.A, rows.B, rows.accept)
        empty!(tally)
        for n in rows.A[i]
            compid[n] == 0 || (tally[compid[n]] = get(tally, compid[n], 0) + 1)
        end
        for n in rows.B[i]
            compid[n] == 0 || (tally[compid[n]] = get(tally, compid[n], 0) - 1)
        end
        for (c, k) in tally
            k == 0 || !(rows.accept[i] || all_rows(c)) || (free[c] = false)
        end
    end
    return free
end

# Total weight of the rows touching each of `nnodes` nodes. Each row is a pair of
# sides, each a node or a collection of nodes.
function _node_weights(nnodes::Integer, rows, w::AbstractVector{T}) where {T}
    nodew = zeros(T, nnodes)
    for ((sa, sb), wi) in zip(rows, w)
        for n in sa
            nodew[n] += wi
        end
        for n in sb
            nodew[n] += wi
        end
    end
    return nodew
end

"""
    station_closure_residuals(dets; observable = :phase, feeds = (1, 1), pfa_max) -> Vector

For every closed triangle of antenna pairs in the detection stack `dets` (the
`AntennaPair × FeedPair` shape [`search_scan`](@ref) returns), the residual
closure quantity of the chosen `observable` (`:delay`/`:rate`/`:phase`) on the
feed pair `feeds`, using the *measured* detections: the signed sum around the
triangle that station-based quantities cancel. On noiseless station-differenced
data a pair `(f, f)` closes to ≈ 0; a pair of different feeds does not, since
its legs read different feeds at the shared station. This is a property of the
data alone, so no solution is needed to evaluate it.

A triangle counts only when all three legs are accepted detections
(`pfa <= pfa_max`); a leg above that threshold carries an arbitrary value,
which would enter the sum as noise rather than as evidence of non-closure.
"""
function station_closure_residuals(
        dets::AbstractDimStack; observable::Symbol = :phase, feeds::Tuple{Integer, Integer} = (1, 1),
        pfa_max::Real = Stationization().pfa_max,
    )
    observable in (:delay, :rate, :phase) || throw(ArgumentError("observable must be :delay, :rate or :phase"))
    q = FeedPair(At(feeds))
    # A pair read in the other order negates only when both feeds are the same.
    leg = Dict{Tuple{String, String}, Float64}()
    for ab in _scan_pairs(dets)
        d = dets[AntennaPair(At(ab)), q]
        (d.valid && d.pfa <= pfa_max) || continue
        leg[ab] = getfield(d, observable)
        feeds[1] == feeds[2] && (leg[reverse(ab)] = -leg[ab])
    end
    ants = unique(Iterators.flatten(_scan_pairs(dets)))
    res = Float64[]
    for i in eachindex(ants), j in eachindex(ants), k in eachindex(ants)
        i < j < k || continue
        a, b, c = ants[i], ants[j], ants[k]
        (haskey(leg, (a, b)) && haskey(leg, (b, c)) && haskey(leg, (a, c))) || continue
        s = leg[(a, b)] + leg[(b, c)] - leg[(a, c)]
        observable === :phase && (s = rem2pi(s, RoundNearest))
        push!(res, s)
    end
    return res
end
