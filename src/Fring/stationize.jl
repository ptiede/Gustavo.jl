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

"""
    OutlierMixture()

Each detection is either explained by the station values, with its noise
model's variance plus a floor `s_c²`, or not, at probability `π_c`, with a
value spread evenly over the delay or rate window its search covered. The class
`c` is whether the detection relates the same feed index on its two stations
or different ones. `s_c` and `π_c` are estimated by EM along with the station
values, which a detection then weights by its probability of being explained,
starting from the [`SoftL1`](@ref) solution; no detection leaves the system.
A detection more likely unexplained than not is reported.
"""
struct OutlierMixture <: AbstractRobustLoss end

# ρ′(u): the IRLS weight multiplier at normalized squared residual `u ≥ 0`.
robust_weight(::LeastSquares, u::Real) = one(u)
robust_weight(::SoftL1, u::Real) = inv(sqrt(1 + u))
robust_weight(::Huber, u::Real) = u <= 1 ? one(u) : inv(sqrt(u))
robust_weight(::Cauchy, u::Real) = inv(1 + u)

"""
    Stationization(; eltype = Float64, pfa_max, weak_sys_scale, loss, loss_scale,
                     irls_iters, systematic_delay, systematic_delay_rate,
                     max_em_iterations, em_tolerance)
    Stationization{T}(; kw...)

Options for [`solve_antenna_systems!`](@ref).

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

`systematic_delay` (seconds) and `systematic_delay_rate` (seconds per second)
are floors added in quadrature to a row's CRB uncertainty:
`w = 1/(σ_CRB² + systematic²)`. A rate row's floor is `systematic_delay_rate`
times its scan's mean frequency (`freq_mean`), since a non-dispersive delay
rate τ̇ moves the fringe rate by `ν·τ̇`. The default delay-rate floor, 2.3e-15,
is the tropospheric floor measured at 43 GHz on the VLBA (a fringe-rate floor
of 1e-4 Hz). Without a floor, data with no real systematics drives `z` to the
numerical noise floor and the loss reweights rounding noise.
`weak_sys_scale` multiplies the total σ of an above-threshold row, floor
included.

A model whose priors along time give hyperparameters as densities, or the
[`OutlierMixture`](@ref) loss, makes the solve an EM over those parameters,
which stops once a cycle moves no log hyperparameter or outlier fraction, nor
any floor's square relative to its detections' median noise variance, by more
than `em_tolerance`, and warns after `max_em_iterations`.
"""
struct Stationization{T <: AbstractFloat, L <: AbstractRobustLoss}
    pfa_max::T
    weak_sys_scale::T
    loss::L
    loss_scale::T
    irls_iters::Int
    systematic_delay::T
    systematic_delay_rate::T
    max_em_iterations::Int
    em_tolerance::T
end

Stationization(; eltype::Type{<:AbstractFloat} = Float64, kw...) = Stationization{eltype}(; kw...)

function Stationization{T}(;
        pfa_max = 1.0e-4, weak_sys_scale = 1.0e3,
        loss = SoftL1(), loss_scale = 8.0, irls_iters = 5,
        systematic_delay = 0.0, systematic_delay_rate = 2.3e-15,
        max_em_iterations = 200, em_tolerance = 1.0e-3,
    ) where {T}
    return Stationization{T, typeof(loss)}(
        pfa_max, weak_sys_scale, loss, loss_scale, irls_iters, systematic_delay, systematic_delay_rate,
        max_em_iterations, em_tolerance,
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
                "`detection_stack`/`solve_antenna_systems!`), or set " *
                "`Stationization(loss = LeastSquares())` to weight rows by the " *
                "noise model alone.",
        ),
    )
end

# A rate row's systematic floor (Hz): the delay-rate floor at the scan's mean
# frequency.
function _rate_floor(delay_rate, freq_mean)
    iszero(delay_rate) && return zero(delay_rate)
    (!isnothing(freq_mean) && freq_mean > 0) || throw(
        ArgumentError(
            "`Stationization(; systematic_delay_rate = $delay_rate)` needs the scan's mean " *
                "frequency to give a rate row its floor, but `freq_mean` is " *
                "$(isnothing(freq_mean) ? "missing" : "$freq_mean"). Supply it (see " *
                "`detection_stack`), or set `systematic_delay_rate = 0`.",
        ),
    )
    return delay_rate * freq_mean
end

# ── Generic, model-driven station solve ─────────────────────────────────────
#
# `solve_antenna_systems!` has no node space of its own: the unknowns are the θ
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
# touches. Under `default_fringe_model` the rows relating different feeds are
# what constrain `rel_delay`'s common mode.
#
# `scans` is a vector of `AntennaPair × FeedPair` Detection `DimStack`s, the shape
# `search_scan` returns, read by label: each antenna pair names its stations,
# which `antennas` numbers, and a representative global time index `:ti` in
# metadata selects the `tseg_id`. θ slots are accumulated into
# with `+=`, so `rounds > 1` — search on the residual — adds each round's
# increment to the last.

"""
    detection_stack(D::AbstractMatrix{<:Detection}, antenna_pairs, feeds;
                    ti, freq_mean, freq_rms, time_rms) -> DimStack

Package a plain `[antenna pair, feed pair]` detection matrix as the
`AntennaPair × FeedPair` DimStack shape `search_scan` returns, labeled by the
antenna-name pairs `antenna_pairs` and the feed-index pairs `feeds` (see
[`feed_pairs`](@ref)), carrying `ti` (the representative global time index),
the scan's mean frequency `freq_mean` (Hz) and its RMS frequency and time
spreads in metadata — so a scan built directly (the refine stage, or a
direct `solve_antenna_systems!` call) has the same shape as
one that came from the search, and every consumer reads pairs/feeds/ti off the
stack uniformly.
"""
function detection_stack(
        D::AbstractMatrix{<:Detection}, antenna_pairs, feeds;
        ti::Integer,
        freq_mean::Union{Nothing, Real} = nothing,
        freq_rms::Union{Nothing, Real} = nothing,
        time_rms::Union{Nothing, Real} = nothing,
    )
    gdims = (_antenna_pair_dim(collect(antenna_pairs)), FeedPair(collect(feeds)))
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
        layers; metadata = _scan_meta(ti, freq_mean, freq_rms, time_rms),
    )
end

# A detection stack's scan-level provenance: the representative global time
# index, and the mean frequency and RMS frequency/time spreads the weights need.
_scan_meta(ti, freq_mean, freq_rms, time_rms, delay_window = nothing, rate_window = nothing) =
    (; ti = Int(ti), freq_mean, freq_rms, time_rms, delay_window, rate_window)

# Attach a representative global time index to an existing detection stack (the
# search's own `search_scan` return, which carries no `:ti` — the caller knows
# which window it searched).
_with_ti(
    stack::AbstractDimStack, ti::Integer; freq_mean::Union{Nothing, Real} = nothing,
    freq_rms::Union{Nothing, Real} = nothing, time_rms::Union{Nothing, Real} = nothing,
    delay_window = nothing, rate_window = nothing,
) = DimensionalData.rebuild(stack; metadata = _scan_meta(ti, freq_mean, freq_rms, time_rms, delay_window, rate_window))

_scan_pairs(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, AntennaPair))
_scan_feeds(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, FeedPair))

_antenna_slot(slot, name) = get(slot, name) do
    throw(ArgumentError("detection antenna `$name` is not among the antennas " * join(keys(slot), ", ")))
end
_scan_ti(sc::AbstractDimStack) = DimensionalData.metadata(sc).ti

"""
    solve_antenna_systems!(θ, scans, components, antennas; gauge, opts, times) -> (ncomp, covered, unlinked, fits)

Solve the stage-B fringe systems (delay and rate) over `scans` and
accumulate the per-(station, feed) values into `θ` at the columns the model
declares. `antennas` are the station names the layout numbers; each
detection's antenna pair is matched to them by name. `components` is a vector
of `(plan::ComponentPlan, kind::Symbol)` with
`kind ∈ (:delay, :rate)`. Multiple components of the same kind are summed
per (station, feed) observation: e.g. a feed-common `PerScan × SharedFeeds` term
plus a `GlobalTime × ExceptFeed(1)` inter-feed offset both feed the delay
system, so a row on feed `k ≠ 1` touches both columns and a stable inter-feed offset is solved
once across the track (bright scans pin it; weak scans inherit it, tying feeds
that would otherwise split). Returns the delay system's component count and
`covered` — the `(antenna name, feed, scan index)` triples the solve CONSTRAINS
in every solved kind: those an accepted detection (`pfa ≤ pfa_max`) touches on
that feed. Coverage is independent of the gauge (see `Stationization` for how
inconsistent rows are weighted). A θ column no accepted detection touches —
per-scan or scan-spanning, per-feed or shared by every feed — is set to zero
(identity gain), as is one under [`OutlierMixture`](@ref) that only
detections it leaves unexplained touch. With a single per-scan/per-feed component per kind and one scan,
each scan's system is independent and solves exactly as it would alone.

A component with a prior along `Ti` adds the prior's rows over its time
segments, placed at the mean of `times`, the solve geometry's sample times, in
each segment. The prior ties the segments' values, so a column it ties counts
as constrained, and a station's feed is covered where every column of its value
is constrained and one is tied.

`fits` holds, per kind that ran the EM of `Stationization`, the resolved prior
of each chain, the [`OutlierMixture`](@ref) parameters per row class, and the
rows more likely unexplained than not.

`unlinked` holds the `(kind, scan index)` pairs in which the accepted
detections leave the offset between some station's feeds free: no accepted
detection relates different feeds of that fringe group. The gauge then sets
it, at the gauge's preferred station, so it is not measured.
"""
function solve_antenna_systems!(
        θ::AbstractVector, scans, components, antennas;
        gauge::AbstractGauge, opts::Stationization{T} = Stationization(), times = nothing,
    ) where {T}
    # θ columns are numbered from 1 (`_block_index`).
    Base.require_one_based_indexing(θ)
    slot = Dict(n => i for (i, n) in pairs(antennas))
    ncomp = 0
    # (station, feed, scan-index) triples the solve constrains: touched by an
    # accepted detection in every solved kind, or determined in every kind and
    # tied by a prior in one. Everything else is flagged downstream.
    touched = Set{Tuple{Int, Int, Int}}()
    determined = Set{Tuple{Int, Int, Int}}()
    tied = Set{Tuple{Int, Int, Int}}()
    unconstrained = Set{Int}()
    unlinked = Set{Tuple{Symbol, Int}}()
    fits = Dict{Symbol, Any}()
    first_kind = true
    for kind in (:delay, :rate)
        plans = [c[1] for c in components if c[2] === kind]
        isempty(plans) && continue
        nc, cov, solved, constrained, free_scans, fit = _solve_kind_cols!(θ, scans, plans, slot, gauge, opts, Val(kind), times)
        isnothing(fit) || (fits[kind] = fit)
        union!(unlinked, ((kind, si) for si in free_scans))
        union!(unconstrained, setdiff(keys(solved), constrained))
        touched = first_kind ? cov.touched : intersect(touched, cov.touched)
        determined = first_kind ? cov.determined : intersect(determined, cov.determined)
        union!(tied, cov.tied)
        first_kind = false
        kind === :delay && (ncomp = nc)
    end
    covered = union(touched, intersect(determined, tied))
    # Columns no accepted row touches hold values set by weak rows alone.
    for col in unconstrained
        θ[col] = zero(eltype(θ))
    end
    return ncomp, Set{Tuple{eltype(antennas), Int, Int}}((antennas[a], f, si) for (a, f, si) in covered), unlinked, fits
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
        ::Val{kind}, times,
    ) where {T, kind}
    getval(d) = getfield(d, kind)
    spread_key = kind === :delay ? :freq_rms : :time_rms

    colnode = Dict{Int, Int}()               # θ column → local node id
    node_col = Int[]                         # local node → θ column
    node_feed = Int[]                        # exclusive feed, or 0 if a column is shared by several feeds
    node_antenna = Int[]
    node_scan = Int[]                        # scan id, or 0 if a column spans scans (global)
    node_path = Tuple{Vararg{Symbol}}[]      # the owning component's path
    function getnode(col, st, fd, sidx, path)
        if haskey(colnode, col)
            n = colnode[col]
            node_feed[n] == fd || (node_feed[n] = 0)        # touched by several feeds → shared
            node_scan[n] == sidx || (node_scan[n] = 0)      # spans scans → global
            return n
        end
        push!(node_col, col); push!(node_feed, fd); push!(node_antenna, st); push!(node_scan, sidx)
        push!(node_path, path)
        return colnode[col] = length(node_col)
    end
    # A state of a prior's chain beyond the column's own value: no θ column.
    function latentnode(st, fd, sidx, path)
        push!(node_col, 0); push!(node_feed, fd); push!(node_antenna, st); push!(node_scan, sidx)
        push!(node_path, path)
        return length(node_col)
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
            (_antenna_slot(slot, a), _antenna_slot(slot, b))
        end
        feeds = _scan_feeds(sc)
        ti = _scan_ti(sc)
        # Per scan, not per row: the band and duration are properties of the
        # observation, so every row of one scan shares this lever arm.
        σ = T(_require_spread(getproperty(DimensionalData.metadata(sc), spread_key), kind, opts.loss))
        σν = kind === :delay ? σ : one(T)
        σt = kind === :rate ? σ : one(T)
        sys = kind === :delay ? T(opts.systematic_delay) : T(_rate_floor(opts.systematic_delay_rate, DimensionalData.metadata(sc).freq_mean))
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
    if isempty(rowA)
        none = Set{Tuple{Int, Int, Int}}()
        return (0, (; touched = none, determined = copy(none), tied = copy(none)), Dict{Int, T}(), Set{Int}(), Set{Int}(), nothing)
    end
    chains = _prior_chains!(getnode, latentnode, colnode, node_feed, plans, scans, times)
    priors = Any[_hyper_start(ch.spec) for ch in chains]

    # Robust solve: IRLS over `opts.loss`, rescaling each row's noise-model
    # weight by the loss's derivative at that row's normalized residual (see
    # `Stationization`). No row leaves the system, so the graph's connectivity,
    # and hence which stations are solvable, is fixed before the first solve.
    # An `OutlierMixture` starts from the `SoftL1` solution.
    w = copy(rw)
    rows = (;
        A = rowA, B = rowB, links = rlinks, val = rval, accept = raccept,
        sta_a = rsta_a, feed_a = rfeed_a, sta_b = rsta_b, feed_b = rfeed_b, scan = rscan,
        prior = _prior_block(chains, priors, length(node_col), T),
    )
    nodes = (; feed = node_feed, antenna = node_antenna, scan = node_scan, component = node_path)
    x, ncomp, resid, _ = _solve_tagged_system(rows, w, nodes, gauge, kind)
    loss = opts.loss isa OutlierMixture ? SoftL1() : opts.loss
    if !(loss isa LeastSquares)
        for _ in 1:max(opts.irls_iters, 0)
            _irls_weights!(w, rw, loss, opts.loss_scale, resid) || break
            x, ncomp, resid, _ = _solve_tagged_system(rows, w, nodes, gauge, kind)
        end
    end
    fits = nothing
    informative = raccept
    if opts.loss isa OutlierMixture || any(ch -> _hyper_count(ch.spec) > 0, chains)
        width = [_search_width(scans[k], kind) for k in rscan]
        em = _closure_em!(rows, rw, w, nodes, gauge, kind, chains, priors, width, opts)
        x, ncomp, resid = em.x, em.ncomp, em.resid
        # A detection the mixture leaves unexplained still measures its
        # stations, but sets no value.
        informative = em.rows.accept
        rows = merge(em.rows, (; accept = raccept))
        fits = (; em.priors, em.mixture, em.outliers, em.converged, em.iterations, chains = [(; ch.plan.path, ch.antenna, ch.node) for ch in chains])
    end
    solved = Dict{Int, T}()
    for n in eachindex(node_col)
        node_col[n] == 0 && continue
        θ[node_col[n]] += x[n]
        solved[node_col[n]] = x[n]
    end
    accepted = Set{Int}(
        node_col[n] for i in eachindex(rowA, rowB, informative) if informative[i]
            for n in Iterators.flatten((rowA[i], rowB[i]))
    )
    chained = Set{Int}(node_col[first(ch.states[k])] for ch in chains for k in eachindex(ch.states))
    covered = _covered_antenna_feeds(rows, node_col, accepted, chained)
    return ncomp, covered, solved, union(accepted, chained), _unlinked_scans(_incidence(rows, length(node_col), T), rows), fits
end

# The chains a prior along time runs over in a kind's system, one per (station,
# feed node) of each component whose prior runs along `Ti`: over the
# component's time segments in time order, each segment's state is its column's
# node followed by the prior's further states, `latentnode`s. A chain is built
# where the station's feed node has a column some detection touches; the
# columns of segments no detection touches join it and take the prior's
# interpolation. Each chain holds its resolved prior `spec`, the coordinates
# `u` and unit `h` its model runs on, and its states.
function _prior_chains!(getnode, latentnode, colnode, node_feed, plans, scans, times)
    chains = NamedTuple[]
    for plan in plans
        any(p -> !isnothing(_prior_along(p, :Ti)), plan.priors) || continue
        epochs = [_segment_time(plan, times, k) for k in 1:plan.shape.Ti]
        order = sortperm(epochs)
        scan_of = Dict(plan.tseg_id[_scan_ti(sc)] => sidx for (sidx, sc) in pairs(scans))
        for a in eachindex(plan.priors), n in 1:plan.shape.node
            spec = _prior_along(plan.priors[a], :Ti)
            isnothing(spec) && continue
            cols = [_block_index(plan, n, 1, k, a) for k in order]
            present = findfirst(c -> haskey(colnode, c), cols)
            isnothing(present) && continue
            fd = node_feed[colnode[cols[present]]]
            u, h = _piece_coordinates(spec, epochs[order])
            states = map(eachindex(cols, order)) do k
                sidx = get(scan_of, order[k], 0)
                [getnode(cols[k], a, fd, sidx, plan.path); [latentnode(a, fd, sidx, plan.path) for _ in 2:_statedim(spec)]]
            end
            push!(chains, (; plan, antenna = a, node = n, spec, u, h, states, dim = Val(_statedim(spec))))
        end
    end
    return chains
end

# The prior rows of `chains` under the resolved `priors`, one per chain, as a
# dense matrix over `nnodes` nodes with right-hand sides: each prior's whitened
# transitions `L⁻¹(s′ − A s)` and, for a proper start, its whitened first
# state, so that with unit weights their squared sum is the prior's energy.
function _prior_block(chains, priors, nnodes::Integer, ::Type{T}) where {T}
    rows = Vector{Tuple{Int, T}}[]
    rhs = T[]
    for (ch, prior) in zip(chains, priors)
        _with_piece_models(prior, ch.dim, T) do model_of
            model = model_of(ch.h)
            start = initial(model, T)
            if !isnothing(start)
                μ0, P0 = start
                W = inv(cholesky(Symmetric(Matrix(P0))).L)
                _push_prior_rows!(rows, rhs, W, ch.states[1], W * μ0)
            end
            for k in 2:length(ch.states)
                A, Q = transition(model, ch.u[k] - ch.u[k - 1], T)
                W = inv(cholesky(Symmetric(Matrix(Q))).L)
                _push_prior_rows!(rows, rhs, hcat(W, -W * A), [ch.states[k]; ch.states[k - 1]], zeros(T, size(W, 1)))
            end
        end
    end
    M = zeros(T, length(rows), nnodes)
    for (r, entries) in pairs(rows), (n, v) in entries
        M[r, n] += v
    end
    return (; M, rhs)
end

# The rows `W * s = b` over the nodes `s`.
function _push_prior_rows!(rows, rhs, W, nodes, b)
    for i in axes(W, 1)
        push!(rows, [(nodes[j], W[i, j]) for j in axes(W, 2)])
        push!(rhs, b[i])
    end
    return rows
end

# The resolved prior `spec` with every hyperparameter it gives as a density at
# that density's median.
_hyper_start(spec) = _hyper_count(spec) == 0 ? spec :
    _with_log_hypers(spec, [log(median(h)) for h in _hyper_order(spec) if !is_fixed_hyper(h)])
_hyper_order(p::RandomWalkPrior) = (p.σ,)
_hyper_order(p::OUPrior) = (p.scale, p.σ)
_hyper_order(p::CARMAPrior) = _hyper_specs(p)

# The width of the window a scan's search covered in `kind`'s observable.
_search_width(sc, kind) = (w = getproperty(DimensionalData.metadata(sc), kind === :delay ? :delay_window : :rate_window); w[2] - w[1])

# Each row's class: `1` where it relates the same feed index on its two
# stations, `2` where different ones.
_row_classes(rows) = [fa == fb ? 1 : 2 for (fa, fb) in zip(rows.feed_a, rows.feed_b)]

"""
    _closure_em!(rows, rw, w, nodes, gauge, kind, chains, priors, width, opts)

EM over a kind's station values `x` and the parameters of its noise and prior
models, from the weights `w` of the starting solve. Under
[`OutlierMixture`](@ref), a row's inlier variance is `f = σ² + s_c²`: its
noise variance `σ² = 1/rw` and its class's floor `s_c²`. Each cycle's E-step
gives each row's probability `γ` of being unexplained, from its residual `e`
against `x`: `π_c/W` (`W` its search's `width`) against `(1 − π_c)·N(e; 0, f)`;
then the Gaussian posterior of `x`, `x̂` and `Σ`, with each row weighted by
`(1 − γ)/f`. The M-step sets `π_c` to the mean `γ` of its class and `s_c²` to
the mean over the class, weighted by `1 − γ`, of the floor's posterior second
moment, `s²(f − s²)/f + (s²/f)²(e² + v)` with `e` against `x̂` and `v` the
variance of the row's fitted value (Henderson's variance-component step); and
each chain whose prior gives hyperparameters as densities to the maximum of its
expected log prior plus hyperprior (`_m_step`). A row more likely unexplained
than not no longer joins stations into a fringe group, as a detection above
`pfa_max` does not; other losses keep reweighting as their IRLS does.
"""
function _closure_em!(rows, rw, w, nodes, gauge, kind, chains, priors, width, opts::Stationization{T}) where {T}
    mixture = opts.loss isa OutlierMixture
    loss = mixture ? SoftL1() : opts.loss
    nnodes = length(nodes.feed)
    A = _incidence(rows, nnodes, T)
    class = _row_classes(rows)
    σ2 = inv.(rw)
    present = [any(==(c), class) for c in 1:2]
    s2 = [present[c] ? median(σ2[class .== c]) : zero(T) for c in 1:2]
    frac = [present[c] ? T(0.05) : zero(T) for c in 1:2]
    estimated = [i for i in eachindex(chains) if _hyper_count(chains[i].spec) > 0]
    As = sparse(A)
    gauges = Dict{BitVector, @NamedTuple{A::Matrix{T}, C::Matrix{T}, d::Vector{T}, ncomp::Int}}()
    # The gauge rows given which rows the station values explain: the others
    # carry no information, so they neither link stations nor hold a gauge,
    # and a station's value only they measure is pinned like any other value
    # the rows leave free.
    sides = _row_sides(rows)
    gauge_for(rs, explained, wts) = get!(() -> _gauge_rows(_select_rows(rs, explained), wts[explained], nodes, gauge, kind; sides), gauges, explained)
    function solve(rs, explained, wts)
        local g = gauge_for(rs, explained, wts)
        basis = _gauge_basis(g.C, g.d)
        return _gauge_solve(As, wts, rs, basis)..., basis.Z
    end

    x0, _, _ = solve(rows, trues(length(rw)), w)

    # One EM cycle from station values `x`, updating the parameters in place:
    # the next station values, the largest parameter change, and the E-step and
    # solve it made.
    function cycle(x)
        local prior, resid, γ, change
        prior = _prior_block(chains, priors, nnodes, T)
        resid = rows.val .- As * x
        γ = zeros(T, length(rw))
        kept = ones(T, length(rw))
        f = σ2 .+ s2[class]
        wts = copy(w)
        if mixture
            for r in eachindex(resid, γ, kept)
                c = class[r]
                inlier = log1p(-frac[c]) - (resid[r]^2 / f[r] + log(2T(π) * f[r])) / 2
                outlier = log(frac[c]) - log(T(width[r]))
                γ[r] = inv(1 + exp(inlier - outlier))
                kept[r] = inv(1 + exp(outlier - inlier))
            end
            wts .= kept ./ f
        elseif !(loss isa LeastSquares)
            _irls_weights!(wts, rw, loss, opts.loss_scale, resid)
        end
        explained = BitVector(γ .<= 1 / 2)
        rs = merge(rows, (; prior, accept = rows.accept .& explained))
        x1, F, Z = solve(rs, explained, wts)
        resid1 = rows.val .- As * x1
        cov = _covariance(F, Z)
        v = mixture ? _fit_variances(cov, As) : nothing
        Σ = isempty(estimated) ? nothing : _node_covariance(cov, chains[estimated], nnodes)
        change = zero(T)
        if mixture
            # A floor's posterior second moment, averaged over its class's rows
            # weighted by their probability of being explained. Rows all
            # unexplained say nothing about the floor, which keeps its value.
            function second_moment(s2j, members)
                total = sum(kept[r] for r in members)
                iszero(total) && return s2j
                return sum(kept[r] * (s2j * (f[r] - s2j) / f[r] + (s2j / f[r])^2 * (resid1[r]^2 + v[r])) for r in members) / total
            end
            # Each change is against its rows' variance, the relative change in
            # their weights, as a variance tending to zero moves its logarithm
            # without bound.
            for c in 1:2
                present[c] || continue
                members = findall(==(c), class)
                π1 = sum(γ[members]) / length(members)
                s21 = second_moment(s2[c], members)
                change = max(change, abs(π1 - frac[c]), abs(s21 - s2[c]) / median(f[members]))
                frac[c], s2[c] = π1, s21
            end
        end
        for i in estimated
            ch = chains[i]
            priors[i], δ = _m_step(ch.spec, priors[i], [ch], [_state_moments(ch, x1, Σ)])
            change = max(change, δ)
        end
        return x1, change, (; γ, explained, wts, rows = rs)
    end

    x, change, last = cycle(x0)
    iterations = 1
    while !(change < opts.em_tolerance) && iterations < opts.max_em_iterations
        x, change, last = cycle(x)
        iterations += 1
    end
    converged = change < opts.em_tolerance
    prior = _prior_block(chains, priors, nnodes, T)
    final = merge(last.rows, (; prior))
    g = gauge_for(final, last.explained, last.wts)
    x = ConstrainedWLS(vcat(A, prior.M), vcat(last.wts, ones(T, size(prior.M, 1))), g.C, g.d)(vcat(final.val, prior.rhs))
    resid = final.val .- A * x
    γ = last.γ
    outliers = [
        (; sta_a = final.sta_a[r], feed_a = final.feed_a[r], sta_b = final.sta_b[r], feed_b = final.feed_b[r], scan = final.scan[r], residual = resid[r], probability = γ[r])
            for r in eachindex(γ) if γ[r] > 1 / 2
    ]
    return (; x, ncomp = g.ncomp, resid, rows = final, priors, converged, iterations, outliers,
        mixture = mixture ? (; floor = sqrt.(s2), outlier_fraction = frac) : nothing)
end

# The detection rows `keep` selects, with every prior row.
_select_rows(rows, keep) = merge(
    map(v -> v[keep], Base.structdiff(rows, NamedTuple{(:prior,)})),
    (; rows.prior),
)

# A chain's state moments under the posterior mean `x` and covariance `Σ`: its
# first state's mean and second moment, and the second moment of each
# consecutive pair `(s_k, s_{k−1})`, as `_m_step` reads them.
function _state_moments(ch, x, Σ)
    D = length(first(ch.states))
    second(idx) = x[idx] * x[idx]' + Σ[idx, idx]
    s1 = ch.states[1]
    return (;
        start = (SVector{D}(x[s1]), SMatrix{D, D}(second(s1))),
        pairs = [SMatrix{2D, 2D}(second([ch.states[k]; ch.states[k - 1]])) for k in 2:length(ch.states)],
    )
end

# The mean of the `times` in a component's time segment `k`.
function _segment_time(plan, times, k)
    ts = [times[i] for i in eachindex(plan.tseg_id, times) if plan.tseg_id[i] == k]
    return sum(ts) / length(ts)
end


# The (station, feed, scan) triples this system calibrates, as three sets:
# `touched`, those carrying at least one accepted detection on that feed;
# `determined`, those whose value sums only columns an accepted detection
# touches or a prior along time ties to its neighbors (`chained`); and `tied`,
# those whose value sums a chained column.
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
function _covered_antenna_feeds(rows, node_col, accepted, chained)
    touched = Set{Tuple{Int, Int, Int}}()
    determined = Set{Tuple{Int, Int, Int}}()
    tied = Set{Tuple{Int, Int, Int}}()
    for i in eachindex(rows.A, rows.B, rows.accept)
        sides = ((rows.sta_a[i], rows.feed_a[i], rows.A[i]), (rows.sta_b[i], rows.feed_b[i], rows.B[i]))
        for (st, f, nodes) in sides
            key = (st, f, rows.scan[i])
            cols = [node_col[n] for n in nodes]
            rows.accept[i] && push!(touched, key)
            all(c -> c in accepted || c in chained, cols) && push!(determined, key)
            any(in(chained), cols) && push!(tied, key)
        end
    end
    return (; touched, determined, tied)
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
    (; A, C, d, ncomp) = _gauge_rows(rows, w, nodes, gauge, kind)
    solve_system = ConstrainedWLS(vcat(A, rows.prior.M), vcat(w, ones(T, size(rows.prior.M, 1))), C, d)
    x = solve_system(vcat(rows.val, rows.prior.rhs))
    return x, ncomp, rows.val .- A * x, solve_system
end

# The gauge rows `Cx = d` of a kind's system under the weights `w`, the number
# of gauged components the accepted rows form, and the dense incidence `A`.
function _gauge_rows(rows, w, nodes, gauge::AbstractGauge, kind::Symbol; sides = _row_sides(rows))
    T = eltype(rows.val)
    nnodes = length(nodes.feed)
    compid, ncomp, nfree = _gauge_components(rows, nodes)

    # Total row weight on each node — the score the pin falls back to when a
    # component holds no reference node.
    nodew = _node_weights(nnodes, zip(rows.A, rows.B), w)

    freedoms = map(1:nfree) do c
        cn = [n for n in eachindex(compid) if compid[n] == c]
        GaugeFreedom(;
            nodes = cn, antenna = nodes.antenna[cn], feed = nodes.feed[cn], scan = nodes.scan[cn],
            component = nodes.component[cn], observable = fill(kind, length(cn)),
            direction = ones(T, length(cn)), weight = nodew[cn],
        )
    end
    A = _incidence(rows, nnodes, T)
    C, d = _gauge_system(gauge, GaugeFreedoms{T}(freedoms, nnodes))
    # The null-space test and the constrained factorization work on dense rows.
    C = Matrix(C)
    C, d = _pin_leftover_freedoms(C, d, A, rows, nodew, gauge, sides)
    return (; A, C, d, ncomp)
end

# The feasible set `x0 + Z·y` of the gauge rows `Cx = d`. An unknown no row
# touches is its own column of `Z`, and each set of unknowns the rows couple
# gets a null-space block of its own, so `Z` is as sparse as the rows are local.
function _gauge_basis(C, d)
    T = eltype(C)
    n = size(C, 2)
    group = collect(1:n)
    root(j) = group[j] == j ? j : (group[j] = root(group[j]))
    for row in eachrow(C)
        cols = findall(!iszero, row)
        for j in cols
            group[root(j)] = root(first(cols))
        end
    end
    constrained = falses(n)
    blocks = Dict{Int, Vector{Int}}()
    for (r, row) in pairs(eachrow(C))
        cols = findall(!iszero, row)
        constrained[cols] .= true
        push!(get!(blocks, root(first(cols)), Int[]), r)
    end
    x0 = zeros(T, n)
    I, J, V = Int[], Int[], T[]
    ncol = 0
    for j in findall(!, constrained)
        push!(I, j); push!(J, ncol += 1); push!(V, one(T))
    end
    for rs in values(blocks)
        cols = findall(j -> constrained[j] && root(j) == root(findfirst(!iszero, C[first(rs), :])), 1:n)
        Cb = C[rs, cols]
        x0[cols] = pinv(Cb) * d[rs]
        N = nullspace(Cb)
        for k in axes(N, 2), (i, j) in pairs(cols)
            push!(I, j); push!(J, ncol + k); push!(V, N[i, k])
        end
        ncol += size(N, 2)
    end
    return (; Z = sparse(I, J, V, n, ncol), x0)
end

# The station values from the normal equations of the detection rows (sparse
# incidence `As`, weights `w`) and prior rows on the gauge's feasible set
# `x0 + Z·y`, and the Cholesky factor `F` of the reduced normal matrix
# `Z'HZ`, from which `_covariance` reads the posterior covariance.
function _gauge_solve(As, w, rows, basis)
    (; Z, x0) = basis
    M = sparse(rows.prior.M)
    H = transpose(As) * Diagonal(w) * As + transpose(M) * M
    g = transpose(As) * (w .* rows.val) + transpose(M) * rows.prior.rhs
    F = cholesky(Symmetric(transpose(Z) * H * Z))
    return x0 + Z * (F \ (transpose(Z) * (g - H * x0))), F
end

# Overwrite the lower-triangular Cholesky factor `L` with the entries of
# `(LL')⁻¹` on its pattern, which a factor's pattern closes under (Takahashi,
# Fagan & Chen 1973): `Σᵢⱼ = −Σₖ LₖⱼΣᵢₖ/Lⱼⱼ` for `i > j` and
# `Σⱼⱼ = (1/Lⱼⱼ − Σₖ LₖⱼΣₖⱼ)/Lⱼⱼ`, over the `k > j` of column `j`. Column `j`
# reads only its own `L` entries and the `Σ` of later columns.
function _selected_inverse!(L::SparseMatrixCSC)
    rv, v = rowvals(L), nonzeros(L)
    for j in reverse(axes(L, 2))
        r = nzrange(L, j)
        d = v[first(r)]
        off = r[2:end]
        l = v[off]
        for p in off
            v[p] = -sum(l[k] * _lower_entry(L, rv[p], rv[q]) for (k, q) in pairs(off)) / d
        end
        v[first(r)] = (inv(d) - sum((l[k] * v[q] for (k, q) in pairs(off)); init = zero(d))) / d
    end
    return L
end

# Entry `(i, k)` of the symmetric matrix whose lower triangle `S` stores.
function _lower_entry(S, i, k)
    j, m = minmax(i, k)
    r = nzrange(S, j)
    return nonzeros(S)[r[searchsortedfirst(view(rowvals(S), r), m)]]
end

# The posterior covariance `Z (Z'HZ)⁻¹ Z'` of the station values from the
# factor `F` of `Z'HZ`, as a function of two unknowns. It reads `(Z'HZ)⁻¹` on
# the factor's pattern, which holds every pair of unknowns sharing a row or a
# prior row.
function _covariance(F, Z)
    Σ = _selected_inverse!(sparse(F.L))
    ip = invperm(F.p)
    Zt = sparse(transpose(Z))
    return function (i, j)
        ri, rj = nzrange(Zt, i), nzrange(Zt, j)
        return sum(
            (nonzeros(Zt)[p] * nonzeros(Zt)[q] * _lower_entry(Σ, ip[rowvals(Zt)[p]], ip[rowvals(Zt)[q]]) for p in ri for q in rj);
            init = zero(eltype(Zt)),
        )
    end
end

# Each row's fitted-value variance `aᵣ'Σaᵣ`.
function _fit_variances(cov, As)
    At = sparse(transpose(As))
    return map(axes(At, 2)) do r
        rng = nzrange(At, r)
        sum(nonzeros(At)[p] * nonzeros(At)[q] * cov(rowvals(At)[p], rowvals(At)[q]) for p in rng for q in rng)
    end
end

# The posterior covariance between consecutive states of each of `chains`, in
# an `nnodes × nnodes` sparse matrix holding only those entries.
function _node_covariance(cov, chains, nnodes)
    I, J = Int[], Int[]
    for ch in chains, k in eachindex(ch.states)
        idx = k == 1 ? ch.states[1] : [ch.states[k]; ch.states[k - 1]]
        append!(I, repeat(idx, outer = length(idx)))
        append!(J, repeat(idx, inner = length(idx)))
    end
    return sparse(I, J, [cov(i, j) for (i, j) in zip(I, J)], nnodes, nnodes, (a, b) -> a)
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

# The rows a gauge must leave unchanged, as a matrix over the nodes: the held
# detection rows of the incidence `A` and every prior row, scaled to unit size
# like the detection rows so a null-space tolerance applies to both.
_held_system(A, rows) = vcat(sparse(A[_held_rows(rows), :]), sparse(rows.prior.M ./ maximum(abs, rows.prior.M; dims = 2, init = zero(eltype(A)))))

# `X`'s null space, through `XᵀX`: an `n × n` problem however many rows `X` has.
_null_basis(X) = nullspace(Matrix(transpose(X) * X))

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
    N = _null_basis(_held_system(A, rows))
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
function _pin_leftover_freedoms(C, d, A, rows, nodew, gauge::AbstractGauge, sides = _row_sides(rows))
    T = eltype(C)
    K = _null_basis(vcat(_held_system(A, rows), C))
    size(K, 2) == 0 && return C, d

    weight = Dict{Int, T}()
    for ((st, _, _), cols) in sides
        weight[st] = get(weight, st, zero(T)) + sum(n -> nodew[n], cols)
    end
    preferred = collect(gauge_antenna_order(gauge))
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
# `_covered_antenna_feeds` excludes them from coverage.
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
    # A prior along time holds the shift of a chain's values unless the
    # shift leaves its rows unchanged, as a flat random walk's does.
    M = rows.prior.M
    for c in 1:ncomp
        members = findall(==(c), compid)
        isempty(members) && continue
        any(r -> abs(sum(view(M, r, members))) > sqrt(eps(eltype(M))) * maximum(abs, view(M, r, :)), axes(M, 1)) && (free[c] = false)
    end
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
    antenna_closure_residuals(dets; observable = :phase, feeds = (1, 1), pfa_max) -> Vector

For every closed triangle of antenna pairs in the detection stack `dets` (the
`AntennaPair × FeedPair` shape a fringe search returns), the residual
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
function antenna_closure_residuals(
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
