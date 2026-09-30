# ── Per-feed stationization with closure ─────────────────────────────────────
#
# Weighted least squares turning per-baseline fringe detections into
# per-(station, feed) delays, rates and phases on a graph of 2·nant nodes. The
# incidence, the use of all four correlation products, the row weights and the
# gauge are derived in `docs/src/fringe_fitting.md`.
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
    Stationization(; eltype = Float64, pfa_max, weak_sys_scale, phase_rewrap_iters,
                     loss, loss_scale, irls_iters, systematic_delay, systematic_rate,
                     systematic_phase, systematic_delay_cross, systematic_rate_cross)
    Stationization{T}(; kw...)

Options for [`solve_station_systems!`](@ref).

`eltype` (or `T`) is the floating-point type the station systems are built and
solved in, whatever the detections' own type. The default `Float64` is
deliberate: weak rows sit ~1e-5 below accepted ones in `√w`, near `Float32`'s
rank tolerance, and a phase row's epoch offset is taken from absolute times.

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
which parameters a row touches is set by the model's feed tying. The
feed-blind phase system alone withholds cross-hand rows (see the comment
above `solve_station_systems!`).

`loss`/`loss_scale`/`irls_iters` control robust downweighting: after each
solve, each row's weight is rescaled by
`robust_weight(loss, (z/loss_scale)^2)` at its noise-normalized residual
`z = resid·√w`, and the system re-solved, up to `irls_iters` times. A false
fringe is closure-inconsistent, lands far out in `z`, and is suppressed.
Weights come from the noise model rather than the residual spread, so
`loss_scale` cuts at the same effective threshold whatever the system size.
`loss = LeastSquares()` disables downweighting.

`systematic_delay`/`systematic_rate`/`systematic_phase` (seconds / Hz /
radians) are floors added in quadrature to a row's CRB uncertainty:
`w = 1/(σ_CRB² + systematic²)`; without one, data with no real systematics
drives `z` to the numerical noise floor and the loss reweights rounding
noise. The `_cross` variants floor cross-hand rows (default: the
parallel-hand values) for error that does not shrink with SNR, such as
leakage. `weak_sys_scale` multiplies the total σ of an above-threshold row,
floor included. `phase_rewrap_iters` re-wraps phase residuals exceeding ±π.
"""
struct Stationization{T <: AbstractFloat, L <: AbstractRobustLoss}
    pfa_max::T
    weak_sys_scale::T
    phase_rewrap_iters::Int
    loss::L
    loss_scale::T
    irls_iters::Int
    systematic_delay::T
    systematic_rate::T
    systematic_phase::T
    systematic_delay_cross::T
    systematic_rate_cross::T
end

Stationization(; eltype::Type{<:AbstractFloat} = Float64, kw...) = Stationization{eltype}(; kw...)

function Stationization{T}(;
        pfa_max = 1.0e-4, weak_sys_scale = 1.0e3, phase_rewrap_iters = 3,
        loss = SoftL1(), loss_scale = 8.0, irls_iters = 5,
        systematic_delay = 0.0, systematic_rate = 0.0, systematic_phase = 0.0,
        systematic_delay_cross = systematic_delay, systematic_rate_cross = systematic_rate,
    ) where {T}
    return Stationization{T, typeof(loss)}(
        pfa_max, weak_sys_scale, phase_rewrap_iters, loss, loss_scale, irls_iters,
        systematic_delay, systematic_rate, systematic_phase,
        systematic_delay_cross, systematic_rate_cross,
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
# quadrature. Only the delay and rate σ carry the array's frequency and time
# extent; phase is already dimensionless in σ units, which is why it alone needs
# no scan geometry.
#
# The spreads matter only for the robust loss. A weighted least-squares solution
# is invariant under scaling every weight in a system by a common constant, and
# σ_ν/σ_t are common to all rows of one scan, so getting them wrong cannot move
# the fit. What they fix is the meaning of `z = resid·√w`: with a true inverse
# variance, z is in units of σ and `loss_scale` is the same dimensionless number
# for all three observables, matching EHT-HOPS's `f_scale = 8` on
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
    kind === :delay ? inv(2π * σ_ν * snr) :
    kind === :rate ? inv(2π * σ_t * snr) : inv(snr)

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
# systems, with no special case for cross hands: the tying alone decides what a
# row touches. Under `default_fringe_terms` the cross-hand delay rows are what
# constrains `rel_delay`'s common mode. The phase system under a feed-blind
# model is the exception, described in `docs/src/fringe_fitting.md`;
# `_solve_kind_cols!` implements it by augmenting that system with
# per-(scan, station) nuisance feed-2 offset columns and withholding cross-hand
# rows from it.
#
# `scans` is a vector of `AntennaPair × FeedPair` Detection `DimStack`s, the shape
# `search_scan` returns, read by label: each antenna pair names its stations,
# which `stations` numbers, and a representative global time index `:ti` in
# metadata selects the `tseg_id`. θ slots are accumulated into
# with `+=`, so `rounds > 1` — search on the residual — adds each round's
# increment to the last.

"""
    detection_stack(D::AbstractMatrix{<:Detection}, antenna_pairs, feeds;
                    ti, epoch, freq_rms, time_rms) -> DimStack

Package a plain `[antenna pair, feed pair]` detection matrix as the
`AntennaPair × FeedPair` DimStack shape `search_scan` returns, labeled by the
antenna-name pairs `antenna_pairs` and the feed-index pairs `feeds` (see
[`feed_pairs`](@ref)), carrying `ti` (the representative global time index) in
metadata — so a scan built directly (the refine stage, or a
direct `solve_station_systems!` call) has the same shape as
one that came from the search, and every consumer reads pairs/feeds/ti off the
stack uniformly.

`epoch` (seconds) is where the phases were measured, which the station solve needs
to read them as constants. Omitting it asserts they sit wherever the model's
rate components are referenced, and is an error when those disagree among
themselves — see `Fring.scan_phase_epoch`.
"""
function detection_stack(
        D::AbstractMatrix{<:Detection}, antenna_pairs, feeds;
        ti::Integer, epoch::Union{Nothing, Real} = nothing,
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
        layers; metadata = _scan_meta(ti, epoch, freq_rms, time_rms),
    )
end

# A detection stack's scan-level provenance: the representative global time
# index, the epoch (seconds) the phases are referenced to, and the RMS
# frequency/time spreads the weights need.
_scan_meta(ti, epoch, freq_rms, time_rms) = (; ti = Int(ti), epoch, freq_rms, time_rms)

# Attach a representative global time index to an existing detection stack (the
# search's own `search_scan` return, which carries no `:ti` — the caller knows
# which window it searched).
_with_ti(
    stack::AbstractDimStack, ti::Integer; epoch::Union{Nothing, Real} = nothing,
    freq_rms::Union{Nothing, Real} = nothing, time_rms::Union{Nothing, Real} = nothing,
) = DimensionalData.rebuild(stack; metadata = _scan_meta(ti, epoch, freq_rms, time_rms))

_scan_pairs(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, AntennaPair))
_scan_feeds(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, FeedPair))

_station_slot(slot, name) = get(slot, name) do
    throw(ArgumentError("detection antenna `$name` is not among the stations " * join(keys(slot), ", ")))
end
_scan_ti(sc::AbstractDimStack) = DimensionalData.metadata(sc).ti
_scan_epoch(sc::AbstractDimStack) = DimensionalData.metadata(sc).epoch
_scan_spread(sc::AbstractDimStack, key::Symbol) = getproperty(DimensionalData.metadata(sc), key)

"""
    solve_station_systems!(θ, scans, components, stations; gauge, opts) -> (ncomp, covered)

Solve the stage-B fringe systems (delay, rate, constant phase) over `scans` and
accumulate the per-(station, feed) values into `θ` at the columns the model
declares. `stations` are the station names the layout numbers; each
detection's antenna pair is matched to them by name. `components` is a vector
of `(plan::ComponentPlan, kind::Symbol)` with
`kind ∈ (:delay, :rate, :phase)`. Multiple components of the same kind are summed
per (station, feed) observation: e.g. a feed-common `PerScan × SharedFeeds` term
plus a `GlobalTime × SingleFeed(2)` inter-feed offset both feed the delay
system, so a feed-2 row touches both columns and a stable inter-feed offset is solved
once across the track (bright scans pin it; weak scans inherit it, tying feeds
that would otherwise split). Returns the phase-system component count and
`covered` — the `(station name, scan index)` pairs the solve CONSTRAINS, which is
independent of the gauge (see `Stationization` for how inconsistent rows are
weighted). The columns of a (station, scan) outside `covered` are set to zero
(identity gain), as are a station's scan-spanning columns when it is covered
in no scan. With a single per-scan/per-feed component per kind and one scan,
each scan's system is independent and solves exactly as it would alone.
"""
function solve_station_systems!(
        θ::AbstractVector, scans, components, stations;
        gauge::AbstractGauge = PinAntenna(1), opts::Stationization{T} = Stationization(),
    ) where {T}
    # θ columns are numbered from 1 (`_block_index`).
    Base.require_one_based_indexing(θ)
    slot = Dict(n => i for (i, n) in pairs(stations))
    ncomp = 0
    # (station, scan-index) pairs the solve constrains, intersected over the
    # solved kinds: a station must be constrained in delay, rate and phase to
    # count as calibrated. Everything else is zeroed (identity gain) and
    # flagged downstream.
    covered = Set{Tuple{Int, Int}}()
    colkeys = Dict{Int, Tuple{Int, Int}}()
    first_kind = true
    rate_plans = [c[1] for c in components if c[2] === :rate]
    rate_solved = Dict{Int, T}()
    # :rate before :phase — a detection's phase is a constant only at the epoch
    # where every rate coordinate vanishes, so a rate referenced to some other
    # epoch has to be subtracted off the phase rows, and that needs it solved.
    for kind in (:delay, :rate, :phase)
        plans = [c[1] for c in components if c[2] === kind]
        isempty(plans) && continue
        nc, cov, solved, keys_ = _solve_kind_cols!(
            θ, scans, plans, slot, gauge, opts, Val(kind);
            rate_plans = kind === :phase ? rate_plans : ComponentPlan[],
            rate_solved,
        )
        kind === :rate && (rate_solved = solved)
        merge!(colkeys, keys_)
        covered = first_kind ? cov : intersect(covered, cov)
        first_kind = false
        kind === :phase && (ncomp = nc)
    end
    _zero_unconstrained!(θ, colkeys, covered)
    return ncomp, Set{Tuple{eltype(stations), Int}}((stations[a], si) for (a, si) in covered)
end

# Zero the θ columns of every (station, scan) no accepted detection constrains,
# and a station's scan-spanning columns when it is constrained in no scan:
# otherwise they hold values set by weak rows alone.
function _zero_unconstrained!(θ, colkeys, covered)
    constrained = Set(first(c) for c in covered)
    for (col, (st, sidx)) in colkeys
        (sidx == 0 ? st in constrained : (st, sidx) in covered) || (θ[col] = zero(eltype(θ)))
    end
    return θ
end

# Solve one observable kind across all scans, accumulating into θ. Each detection
# becomes a station-difference row whose a-/b-side touch the sum of all `plans`'
# θ columns for that (station, feed, time) — a feed-common per-scan column and,
# when present, a global feed-offset column. Returns (ncomp, covered, solved,
# colkeys): `solved` maps each θ column this kind touched to the value it just
# added, `colkeys` to its (station, scan index), scan 0 for a column spanning scans.
#
# `rate_plans`/`rate_solved` are non-empty only for `:phase`, and only matter
# where a rate component's origin differs from the epoch the phases were
# measured at — see `_phase_epoch_offset`.
#
# `kind` is a type parameter so every kind-dependent choice below is made at
# compile time; the caller's `Val(kind)` is the only dynamic dispatch.
function _solve_kind_cols!(
        θ::AbstractVector, scans, plans, slot, gauge::AbstractGauge, opts::Stationization{T},
        ::Val{kind}; rate_plans = ComponentPlan[], rate_solved::Dict{Int, T} = Dict{Int, T}(),
    ) where {T, kind}
    getval(d) = getfield(d, kind)
    # Phase has no cross-hand variant: its floor is dimensionless (radians), so
    # leakage and field rotation enter it at the same scale on either hand.
    sys_par = kind === :delay ? opts.systematic_delay :
        kind === :rate ? opts.systematic_rate : opts.systematic_phase
    sys_cross = kind === :delay ? opts.systematic_delay_cross :
        kind === :rate ? opts.systematic_rate_cross : opts.systematic_phase
    spread_key = kind === :delay ? :freq_rms : :time_rms
    rewrap = kind === :phase ? opts.phase_rewrap_iters : 0

    colnode = Dict{Int, Int}()               # θ column → local node id
    node_col = Int[]                         # local node → θ column (0: nuisance, never written to θ)
    node_feed = Int[]                        # exclusive feed (1/2), or 0 if a column is shared by both feeds
    node_station = Int[]
    node_scan = Int[]                        # scan id, or 0 if a column spans scans (global)
    function getnode(col, st, fd, sidx)
        if haskey(colnode, col)
            n = colnode[col]
            node_feed[n] == fd || (node_feed[n] = 0)        # touched by both feeds → shared
            node_scan[n] == sidx || (node_scan[n] = 0)      # spans scans → global
            return n
        end
        push!(node_col, col); push!(node_feed, fd); push!(node_station, st); push!(node_scan, sidx)
        return colnode[col] = length(node_col)
    end

    # A feed-blind phase system gets per-(scan, station) nuisance feed-2 offset
    # columns (see the header comment above `solve_station_systems!`). Solved
    # like any column, discarded at the θ write-out. Tagged feed 2 so the gauge
    # never anchors a component on one.
    feedblind = kind === :phase && all(p -> p.tying isa SharedFeeds, plans)
    nuis = Dict{Tuple{Int, Int}, Int}()      # (scan, station) → local node id
    function nuisnode(sidx, st)
        return get!(nuis, (sidx, st)) do
            push!(node_col, 0); push!(node_feed, 2); push!(node_station, st); push!(node_scan, sidx)
            length(node_col)
        end
    end

    # Rows in θ-column space: each side is the list of θ columns whose sum is
    # that station's value for this observable (+1 on a-side, −1 on b-side).
    # `rlinks` pairs each column with the same component's column on the other
    # side, and a nuisance column with its side's model column; they define the
    # gauge components (see `_solve_tagged_system`).
    rowA = Vector{Int}[]; rowB = Vector{Int}[]; rlinks = Vector{Tuple{Int, Int}}[]
    rval = T[]; rw = T[]; rcross = Bool[]; rscan = Int[]
    rsta_a = Int[]; rsta_b = Int[]
    # Per row: whether its detection is real (`pfa <= pfa_max`). Only these
    # connect stations into a fringe group; the rest constrain and no more.
    raccept = Bool[]
    for (sidx, sc) in enumerate(scans)
        bl_pairs = map(_scan_pairs(sc)) do (a, b)
            (_station_slot(slot, a), _station_slot(slot, b))
        end
        feeds = _scan_feeds(sc)
        ti = _scan_ti(sc)
        epoch = _scan_epoch(sc)
        isnothing(epoch) && _require_common_epoch(rate_plans, ti)
        # Per scan, not per row: the band and duration are properties of the
        # observation, so every row of one scan shares this lever arm.
        σ = kind === :phase ? one(T) :
            T(_require_spread(_scan_spread(sc, spread_key), kind, opts.loss))
        σν = kind === :delay ? σ : one(T)
        σt = kind === :rate ? σ : one(T)
        # Stations whose feed-1 phase this scan's parallel rows measure — the
        # set eligible for a nuisance feed-2 offset column. A station observed
        # only on feed 2 (a single-feed receiver, or a feed-1 dropout) gets
        # none: the shared column and the offset would be an exactly degenerate
        # pair. Its shared column then carries its feed-2 phase referenced to
        # the reference station's feed-2 frame — the parallel hands cannot
        # separate such a station's phase from the offsets' common mode, and
        # the nuisance-block gauge (see `_solve_tagged_system`) pins that mode
        # at the reference.
        f1 = Set{Int}()
        if feedblind
            for bi in eachindex(bl_pairs), p in eachindex(feeds)
                sc[AntennaPair(bi), FeedPair(p)].valid || continue
                fa, fb = feeds[p]
                (fa == 1 && fb == 1) || continue
                a, b = bl_pairs[bi]
                a == b && continue
                push!(f1, a); push!(f1, b)
            end
        end
        for bi in eachindex(bl_pairs), p in eachindex(feeds)
            det = sc[AntennaPair(bi), FeedPair(p)]
            det.valid || continue                   # no data in this cell, no measurement
            a, b = bl_pairs[bi]
            a == b && continue
            fa, fb = feeds[p]
            cross = fa != fb
            # Withheld from the feed-blind phase system only — see the header
            # comment above `solve_station_systems!`. Delay and rate keep every
            # row: their cross hands are consistent under the model and carry
            # the `rel_delay` common mode.
            feedblind && cross && continue
            accept = det.pfa <= opts.pfa_max
            nsA = Int[]; nsB = Int[]; links = Tuple{Int, Int}[]
            for plan in plans
                na = _feed_node(plan.tying, fa)
                ca = na == 0 ? 0 : _block_index(plan, na, 1, plan.tseg_id[ti], a)
                ca != 0 && push!(nsA, getnode(ca, a, fa, sidx))
                nb = _feed_node(plan.tying, fb)
                cb = nb == 0 ? 0 : _block_index(plan, nb, 1, plan.tseg_id[ti], b)
                cb != 0 && push!(nsB, getnode(cb, b, fb, sidx))
                ca != 0 && cb != 0 && push!(links, (last(nsA), last(nsB)))
            end
            (isempty(nsA) || isempty(nsB)) && continue
            # The nuisance column joins the side after the model columns, so a
            # row side's first entry is always a model column (`_seed_tagged`
            # reads sides that way).
            if feedblind && fa == 2 && a in f1
                push!(nsA, nuisnode(sidx, a))
                push!(links, (first(nsA), last(nsA)))
            end
            if feedblind && fb == 2 && b in f1
                push!(nsB, nuisnode(sidx, b))
                push!(links, (first(nsB), last(nsB)))
            end
            push!(rowA, nsA); push!(rowB, nsB); push!(rlinks, links)
            push!(
                rval, getval(det) -
                    _phase_epoch_offset(rate_plans, rate_solved, ti, epoch, a, fa) +
                    _phase_epoch_offset(rate_plans, rate_solved, ti, epoch, b, fb),
            )
            push!(
                rw, _row_weight(
                    _sigma_stat(kind, det.snr, σν, σt), cross ? sys_cross : sys_par,
                    accept ? one(T) : opts.weak_sys_scale,
                ),
            )
            push!(raccept, accept)
            push!(rcross, cross); push!(rscan, sidx)
            push!(rsta_a, a); push!(rsta_b, b)
        end
    end
    isempty(rowA) && return (0, Set{Tuple{Int, Int}}(), Dict{Int, T}(), Dict{Int, Tuple{Int, Int}}())

    # Robust solve: IRLS over `opts.loss`, rescaling each row's noise-model
    # weight by the loss's derivative at that row's normalized residual (see
    # `Stationization`). No row leaves the system, so the graph's connectivity,
    # and hence which stations are solvable, is fixed before the first solve.
    #
    # IRLS is the outer loop and the phase system's 2π re-wrap the inner one, so
    # every IRLS pass sees a converged branch assignment. Reversing them would
    # downweight rows whose residual is still a wrap away from its final value.
    w = copy(rw)
    rows = (; A = rowA, B = rowB, links = rlinks, val = rval, cross = rcross, accept = raccept)
    nodes = (; feed = node_feed, station = node_station, scan = node_scan, nuisance = node_col .== 0)
    x, ncomp, resid = _solve_tagged_system(rows, w, nodes, gauge, kind; rewrap)
    if !(opts.loss isa LeastSquares)
        for _ in 1:max(opts.irls_iters, 0)
            _irls_weights!(w, rw, opts.loss, opts.loss_scale, resid) || break
            x, ncomp, resid = _solve_tagged_system(rows, w, nodes, gauge, kind; rewrap)
        end
    end
    solved = Dict{Int, T}()
    colkeys = Dict{Int, Tuple{Int, Int}}()
    for n in eachindex(node_col)
        node_col[n] == 0 && continue          # nuisance offset: solved, discarded
        θ[node_col[n]] += x[n]
        solved[node_col[n]] = x[n]
        colkeys[node_col[n]] = (node_station[n], node_scan[n])
    end
    return ncomp, _covered_stations(rsta_a, rsta_b, rscan, raccept), solved, colkeys
end

# The phase a rate component contributes at `epoch` to one (station, feed):
# `2π·ṙ·(epoch − t0_k)` over the rate columns, in radians.
#
# A detection's phase is the phase at the epoch its search referenced, and the
# station solve reads it as a sum of constants. That holds only where every rate
# coordinate is zero, at each rate component's own origin. A component segmented
# like the constants beside it — the default, everything `PerScan` — has its
# origin exactly there and contributes nothing here, so the rows are the
# measured phases unchanged. A component segmented more coarsely, such as a
# track-global inter-feed rate against per-scan constants, is referenced
# elsewhere, and its share of the measured phase is removed here rather than
# left for a per-scan constant to absorb.
#
# `solved` is this round's rate increment per θ column, which is what the phases
# of this round, measured on the previous round's residual, contain.
# Epochs are absolute (s), so their difference is taken in Float64 before it
# narrows to the solve's element type `T`.
function _phase_epoch_offset(rate_plans, solved::AbstractDict{Int, T}, ti::Integer, epoch, a::Integer, feed::Integer) where {T}
    off = zero(T)
    isempty(rate_plans) && return off
    for plan in rate_plans
        node = _feed_node(plan.tying, feed)
        node == 0 && continue
        seg = plan.tseg_id[ti]
        Δt = isnothing(epoch) ? zero(T) : T(Float64(epoch) - Float64(plan.tstate[seg]))
        iszero(Δt) && continue
        col = _block_index(plan, node, 1, seg, a)
        col == 0 && continue
        off += 2 * T(π) * get(solved, col, zero(T)) * Δt
    end
    return off
end

# A detection stack that records no epoch says only "referenced wherever the
# model's rate columns vanish". That is a complete answer when they all vanish
# in the same place, and no answer at all when they do not.
function _require_common_epoch(rate_plans, ti::Integer)
    isempty(rate_plans) && return nothing
    o1 = Float64(first(rate_plans).tstate[first(rate_plans).tseg_id[ti]])
    for plan in rate_plans
        o = Float64(plan.tstate[plan.tseg_id[ti]])
        isapprox(o, o1; atol = _epoch_atol(o1)) || error(
            "solve_station_systems!: the rate components are referenced to different " *
                "epochs at time index $ti ($o1 s vs $o s), so no single epoch makes a " *
                "detection's phase a sum of constants. Record the epoch the phases were " *
                "measured at (`detection_stack(...; epoch)`) so the rates referenced " *
                "elsewhere can be subtracted from the phase rows.",
        )
    end
    return nothing
end

# The (station, scan) pairs this system calibrates: those carrying at least one
# accepted detection, and so constrained by the solve rather than left at θ = 0
# (identity gain). Everything else is flagged downstream.
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
function _covered_stations(rsta_a, rsta_b, rscan, raccept)
    cov = Set{Tuple{Int, Int}}()
    for i in eachindex(rsta_a, rsta_b, rscan, raccept)
        raccept[i] || continue
        push!(cov, (rsta_a[i], rscan[i]))
        push!(cov, (rsta_b[i], rscan[i]))
    end
    return cov
end

# Constrained WLS over a tagged node graph, the column-space generalization of
# `_solve_observable!`. Row `i` observes `Σ x[rows.A[i]] − Σ x[rows.B[i]]` with
# weight `w[i]`; `nodes` tags each local node with its station, feed (0 = shared
# by both feeds), scan (0 = a column spanning scans) and whether it is a
# nuisance offset. `rows.cross` marks cross-hand rows, which are excluded from
# the unwrap seed's spanning tree. Returns the solution, the number of gauged
# components the accepted rows form, and the residuals.
#
# The gauge: `rows.links` join each column to the same component's column on
# the other side of a row (and a nuisance column to its side's model column).
# A component of those links whose model columns can all shift together without
# changing any row it must hold is a gauge freedom, and gets one constraint row
# from `gauge`. That gives one per scan for per-scan columns, however columns
# spanning scans couple them, and one for a feed-2 offset that no cross-hand
# row fixes. Any other freedom is an error.
function _solve_tagged_system(rows, w, nodes, gauge::AbstractGauge, kind::Symbol; rewrap::Integer)
    T = eltype(rows.val)
    nnodes = length(nodes.feed)
    compid, ncomp, nfree = _gauge_components(rows, nodes)

    station_of(n) = nodes.station[n]
    feed_of(n) = nodes.feed[n]

    # Total row weight on each node — the score the pin falls back to when a
    # component holds no reference node.
    nodew = _node_weights(nnodes, zip(rows.A, rows.B), w)

    # One constraint row per gauged component; `anchors` names a real node per
    # component for the phase-unwrap seed.
    comps = [[n for n in eachindex(compid) if compid[n] == c] for c in 1:nfree]
    anchors = [gauge_anchor(gauge, cn, nodew, station_of, feed_of) for cn in comps]
    A = zeros(T, length(rows.A), nnodes)
    for i in eachindex(rows.A, rows.B)
        for n in rows.A[i]
            A[i, n] += one(T)
        end
        for n in rows.B[i]
            A[i, n] -= one(T)
        end
    end
    Cp = zeros(T, length(comps), nnodes)
    for (j, cn) in enumerate(comps)
        gauge_row!(view(Cp, j, :), gauge, cn, nodew, station_of, feed_of)
    end
    # Nuisance-block gauge: the nuisance offset columns of one scan carry a
    # common-mode freedom the data cannot fix — shifting them together, along
    # with the shared column of every station observed only on feed 2, changes
    # no row. It is pinned like any other gauge freedom: one row per
    # (component, scan) group of nuisance nodes, through `gauge`, which prefers
    # the ranked reference — a feed-2-only station's phase is thereby referenced
    # to the reference station's feed-2 frame, deterministically.
    if any(nodes.nuisance)
        groups = Dict{Tuple{Int, Int}, Vector{Int}}()
        for n in eachindex(nodes.nuisance)
            (nodes.nuisance[n] && compid[n] != 0) || continue
            push!(get!(groups, (compid[n], nodes.scan[n]), Int[]), n)
        end
        keyorder = sort!(collect(keys(groups)))
        Cn = zeros(T, length(keyorder), nnodes)
        for (j, k) in enumerate(keyorder)
            gauge_row!(view(Cn, j, :), gauge, groups[k], nodew, station_of, feed_of)
        end
        Cp = vcat(Cp, Cn)
    end
    solve_system = try
        ConstrainedWLS(A, w, Cp)
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

    if rewrap > 0
        x, bw = _rewrap_solve(solve_system, A, rows.val, _seed_tagged(rows, w, anchors, nnodes), rewrap)
        resid = bw .- A * x
    else
        x = solve_system(rows.val)
        resid = rows.val .- A * x
    end
    return x, ncomp, resid
end

# The gauge components of a tagged system, numbered so that the gauged ones come
# first: `compid` (0 for a node in none), the number of gauged components the
# accepted rows form, and the number gauged in all.
#
# Components are built from accepted rows only. A detection above `pfa_max`
# sits at an arbitrary noise peak, so letting it define graph structure would
# hand the component count, the gauge pins and the unwrap anchors to noise.
# Acceptance is a hard connectivity cut: where weak rows bridge two accepted
# components, each keeps its own gauge row, and those rows fix the offset
# between them. Nodes no accepted row links form islands over all rows, gauged
# the same way; `_covered_stations` excludes them from coverage.
function _gauge_components(rows, nodes)
    nnodes = length(nodes.feed)
    accepted = (l for i in eachindex(rows.links, rows.accept) if rows.accept[i] for l in rows.links[i])
    compid, nacc, _ = connected_components(nnodes, accepted)
    unlinked = (l for ls in rows.links for l in ls if compid[l[1]] == 0 && compid[l[2]] == 0)
    island, nisl, _ = connected_components(nnodes, unlinked)
    for n in eachindex(compid, island)
        compid[n] == 0 && island[n] != 0 && (compid[n] = nacc + island[n])
    end
    free = _shift_invariant(compid, nacc + nisl, rows, nodes.nuisance, c -> c > nacc)
    # Renumber: gauged accepted components, then gauged islands, then the rest.
    order = [findall(c -> free[c] && c <= nacc, 1:(nacc + nisl)); findall(c -> free[c] && c > nacc, 1:(nacc + nisl)); findall(!, free)]
    rank = invperm(order)
    for n in eachindex(compid)
        compid[n] == 0 || (compid[n] = rank[compid[n]])
    end
    return compid, count(c -> free[c], 1:nacc), count(free)
end

# Whether shifting each component's model columns by a common constant leaves
# every row unchanged that the component must hold: accepted rows for all, and
# weak rows too for the components `all_rows(c)` selects.
function _shift_invariant(compid, ncomp::Integer, rows, nuisance, all_rows)
    free = trues(ncomp)
    tally = Dict{Int, Int}()
    for i in eachindex(rows.A, rows.B, rows.accept)
        empty!(tally)
        for n in rows.A[i]
            (compid[n] == 0 || nuisance[n]) || (tally[compid[n]] = get(tally, compid[n], 0) + 1)
        end
        for n in rows.B[i]
            (compid[n] == 0 || nuisance[n]) || (tally[compid[n]] = get(tally, compid[n], 0) - 1)
        end
        for (c, k) in tally
            k == 0 || !(rows.accept[i] || all_rows(c)) || (free[c] = false)
        end
    end
    return free
end

# The spanning-tree phase seed in local-node space: each parallel-hand row is a
# tree edge between the first column of each side, which is a model column by
# row construction. Further columns on a side (a global feed offset, a nuisance
# feed-2 offset) are left out of the seed; the constrained WLS and the re-wrap
# iterations solve for them.
function _seed_tagged(rows, w, anchors, nnodes::Integer)
    parallel = (
        (rows.A[i][1], rows.B[i][1], rows.val[i], w[i]) for
            i in eachindex(rows.A, rows.B, rows.val, w, rows.cross) if !rows.cross[i]
    )
    return _prim_seed(eltype(rows.val), nnodes, parallel, anchors)
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

"""
    station_closure_residuals(dets; observable = :phase, feeds = (1, 1), pfa_max) -> Vector

For every closed triangle of antenna pairs in the detection stack `dets` (the
`AntennaPair × FeedPair` shape [`search_scan`](@ref) returns), the residual
closure quantity of the chosen `observable` (`:delay`/`:rate`/`:phase`) on the
feed pair `feeds`, using the *measured* detections: the signed sum around the
triangle that station-based quantities cancel. On noiseless station-differenced
data a parallel-hand feed pair closes to ≈ 0; a cross-hand one does not, since
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
    # A pair read in the other order negates only on a parallel hand.
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
