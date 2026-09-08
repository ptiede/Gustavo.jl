# ── Per-feed stationization with closure ─────────────────────────────────────
#
# Turn per-baseline fringe detections into per-(station, feed) delays, rates and
# phases by weighted least squares on the (station, feed) graph (2·nant nodes).
# Every observable differences station quantities with the same incidence:
#
#     baseline (a,b), product p with feeds (fa, fb) = correlation_feed_pair(p):
#         delay_ab^p = τ_{a,fa} − τ_{b,fb}
#         rate_ab^p  = ṙ_{a,fa} − ṙ_{b,fb}
#         phase_ab^p = φ_{a,fa} − φ_{b,fb}
#
# Using all FOUR products (not just parallel hands) is deliberate: cross-hand
# rows connect feed-1 and feed-2 nodes, so the inter-feed (feed-2 − feed-1)
# delay/phase offset is pinned by the data and falls out of the solution — no
# separate alignment stage. Closure holds by construction within each product
# and across mixed-hand triangles.
#
# What the model omits. The per-antenna response is taken as diagonal, so
# leakage goes to the residuals. There is no parallactic-angle term: with
# circular feeds field rotation is a per-feed phase the solve's own nodes
# absorb; with linear feeds it perturbs the parallel-hand amplitude through Q, U
# and reaches the phase only through the stations' differential rotation, at
# O(V/I). Neither is checked here — modeling either means adding a component.
#
# Gauge: each connected component of the (station, feed) graph has one additive
# freedom per observable, fixed by one constraint row per component supplied by an
# `AbstractGauge`. Cross hands merge the two feeds into a single component, so the
# inter-feed offset is fixed by the data and needs no constraint of its own.

# ── Robust losses ────────────────────────────────────────────────────────────
#
# The station solves downweight inconsistent rows through a robust loss ρ
# applied to the NORMALIZED squared residual u = (z/f)², where z = resid·√w and
# f is the loss scale. Because `w` is the noise model's inverse variance
# (`_row_weight`: the CRB σ of that observable, floored by a systematic term),
# z is already in units of predicted σ and u is dimensionless — the scale is
# never re-estimated from the residuals themselves. That is what makes
# the effective threshold independent of how many rows are in the system: a
# scale fitted from residuals shrinks with the degrees of freedom, so a nominal
# cut tightens on exactly the weak scans that can least afford it.
#
# Each loss supplies `robust_weight(loss, u) = ρ′(u)`, the IRLS multiplier on a
# row's weight. Conventions follow SciPy's `least_squares` loss family, which
# is where EHT-HOPS's `soft_l1` choice comes from.

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
    Stationization(; pfa_max, weak_sys_scale, phase_rewrap_iters, loss,
                     loss_scale, irls_iters, systematic_delay, systematic_rate,
                     systematic_phase, systematic_delay_cross, systematic_rate_cross)

Options for [`solve_station_systems!`](@ref).

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
Base.@kwdef struct Stationization
    pfa_max::Float64 = 1.0e-4
    weak_sys_scale::Float64 = 1.0e3
    phase_rewrap_iters::Int = 3
    loss::AbstractRobustLoss = SoftL1()
    loss_scale::Float64 = 8.0
    irls_iters::Int = 5
    systematic_delay::Float64 = 0.0
    systematic_rate::Float64 = 0.0
    systematic_phase::Float64 = 0.0
    systematic_delay_cross::Float64 = systematic_delay
    systematic_rate_cross::Float64 = systematic_rate
end

# IRLS weight update. `w` (mutated) is the effective weight vector fed to the
# solver, `w0` the untouched noise-model weights that define the σ scale.
# Returns whether any weight moved enough to be worth another solve.
#
# Kept behind its own function so the loss type — an abstract field on
# `Stationization` — is resolved once per iteration rather than per row.
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

# ── Row weights: the noise model, in each observable's own units ─────────────
#
# Every row's weight is `1/σ²` with σ the CRB uncertainty of that measurement:
#
#     σ_delay = 1 / (2π · σ_ν · snr)     seconds   (σ_ν = RMS frequency spread)
#     σ_rate  = 1 / (2π · σ_t · snr)     Hz        (σ_t = RMS time spread)
#     σ_phase = 1 / snr                  radians
#
# plus a systematic floor in quadrature. Only the delay and rate σ carry the
# array's frequency/time extent; phase is already dimensionless in σ units,
# which is why it alone needs no scan geometry.
#
# The spreads matter only for the robust loss. A weighted least-squares solution
# is invariant under scaling every weight in a system by a common constant, and
# σ_ν/σ_t are common to all rows of one scan — so getting them wrong (or right)
# cannot move the fit. What they fix is the meaning of `z = resid·√w`: with a
# true inverse variance, z is in units of σ and `loss_scale` is the same
# dimensionless number for all three observables, matching EHT-HOPS's
# `f_scale = 8` on `(data − model)/err`.

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
    (spread !== nothing && spread > 0) && return float(spread)
    loss isa LeastSquares && return 1.0     # unused: a common factor cancels
    what = kind === :delay ? ("freq_rms", "RMS frequency spread (Hz)") :
        ("time_rms", "RMS time spread (s)")
    throw(
        ArgumentError(
            "a $(nameof(typeof(loss))) loss on the $kind system needs the scan's " *
                "$(what[2]) to express residuals in units of σ, but `$(what[1])` is " *
                "$(spread === nothing ? "missing" : "$spread"). Supply it (see " *
                "`detection_stack`/`solve_station_systems!`), or set " *
                "`Stationization(loss = LeastSquares())` to weight rows by the " *
                "noise model alone.",
        ),
    )
end

# Node index on the (station, feed) graph: feed-1 block 1:nant, feed-2 nant+1:2nant.
_node(ant::Integer, feed::Integer, nant::Integer) = (feed - 1) * nant + ant

# One observation row contributing to a node system. `na`/`nb` are the PARAMETER
# NODES the row's two stations contribute to — `_feed_node(tying, feed)` of the
# correlation product's feeds, so they coincide with the feed indices only under
# `PerFeed`. A row with `na != nb` is cross-hand: it is the only kind that ties
# the two feed blocks together. `src` indexes the row's free source term in a
# system that carries one per (baseline, product) — the adhoc solve, where the
# observed source visibility phase is a per-scan constant absorbed alongside the
# station phases (see `solve_adhoc_phasing`). It is 0 in systems with no such term.
struct _ObsRow
    a::Int
    b::Int
    na::Int
    nb::Int
    val::Float64
    w::Float64
    src::Int
end
_ObsRow(a, b, na, nb, val, w) = _ObsRow(a, b, na, nb, val, w, 0)

# Robust wrapper around `_solve_observable`: IRLS over `opts.loss`. Each pass
# rescales every row's noise-model weight by the loss's derivative at that row's
# normalized residual and re-solves; no scale is estimated from the residuals.
#
# The phase system's own re-wrap iteration lives INSIDE `_solve_observable`, so
# the nesting is IRLS-outer / re-wrap-inner: every IRLS pass sees a fully
# converged 2π branch assignment. The other order would let the loss downweight
# rows whose residual is still one wrap away from its final value, which reads
# as a gross outlier and suppresses a perfectly good row.
function _solve_observable_robust(
        rows::Vector{_ObsRow}, nant::Integer, gauge::AbstractGauge, opts::Stationization;
        rewrap::Integer,
    )
    vals, cov, ncomp, resid = _solve_observable(rows, nant, gauge; rewrap = rewrap)
    (opts.loss isa LeastSquares || isempty(rows)) && return vals, cov, ncomp
    w0 = [r.w for r in rows]
    w = copy(w0)
    for _ in 1:max(opts.irls_iters, 0)
        _irls_weights!(w, w0, opts.loss, opts.loss_scale, resid) || break
        vals, cov, ncomp, resid =
            _solve_observable(rows, nant, gauge; rewrap = rewrap, weights = w)
    end
    return vals, cov, ncomp
end

# Solve one observable's WLS system on the (station, feed) graph. Returns
# (values::(nant,2), covered::(nant,2), ncomp, resid). `weights` overrides
# the rows' own noise-model weights (the IRLS driver's reweighted vector);
# `resid` comes back aligned with `rows`, already 2π-branch-corrected for a
# re-wrapped system, so the driver can normalize it without redoing the unwrap.
function _solve_observable(
        rows::Vector{_ObsRow}, nant::Integer, gauge::AbstractGauge;
        rewrap::Integer,
        seed_phase::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
    )
    vals = fill(NaN, nant, 2)
    cov = falses(nant, 2)
    nnodes = 2 * nant
    isempty(rows) && return vals, cov, 0, Float64[]

    edges = [(_node(r.a, r.na, nant), _node(r.b, r.nb, nant)) for r in rows]
    compid, ncomp, touched = connected_components(nnodes, edges)

    nrow = length(rows)

    # Gauge: `gauge` supplies one constraint row per component. `anchors` names a
    # real node per component as well — phase unwrapping propagates outward from
    # an actual node, which a summed constraint does not provide.
    nodew = zeros(Float64, nnodes)
    for (i, r) in enumerate(rows)
        wi = weights === nothing ? r.w : weights[i]
        nodew[_node(r.a, r.na, nant)] += wi
        nodew[_node(r.b, r.nb, nant)] += wi
    end
    # Inverse of `_node`: the feed-1 block is 1:nant, feed-2 is nant+1:2nant.
    station_of(n) = (n - 1) % nant + 1
    feed_of(n) = n > nant ? 2 : 1
    comps = [findall(==(c), compid) for c in 1:ncomp]
    anchors = [gauge_anchor(gauge, cn, nodew, station_of, feed_of) for cn in comps]
    # Constrain every UNTOUCHED node — an (antenna, feed) with no observation in
    # this solve, e.g. a station that dropped out. Its design column is all-zero,
    # which would make the constrained QR system rank-deficient and corrupt the
    # solve for the stations that DO have data. Fixing it at 0 (its value is
    # discarded; only `touched` cells are returned) keeps the system well-posed.
    # Components hold only touched nodes, so these never collide with a gauge row.
    idle = [n for n in 1:nnodes if !touched[n]]

    A = zeros(Float64, nrow, nnodes)
    b = zeros(Float64, nrow)
    w = zeros(Float64, nrow)
    for (i, r) in enumerate(rows)
        A[i, _node(r.a, r.na, nant)] += 1.0
        A[i, _node(r.b, r.nb, nant)] -= 1.0
        b[i] = r.val
        w[i] = weights === nothing ? r.w : weights[i]
    end
    C = zeros(eltype(A), ncomp + length(idle), nnodes)
    for (j, cn) in enumerate(comps)
        gauge_row!(view(C, j, :), gauge, cn, nodew, station_of, feed_of)
    end
    for (k, n) in enumerate(idle)
        C[ncomp + k, n] = one(eltype(C))
    end
    dgauge = zeros(eltype(C), size(C, 1))

    # Phase re-wrap: unwrap observations toward a model and re-solve, so
    # station-difference phases exceeding ±π are handled. The first model comes
    # from a maximum-weight spanning-tree traversal of the (station, feed) graph
    # (K1): propagating wrapped edge phases from each pin gives a globally
    # consistent unwrap that is correct even when a true station-difference
    # exceeds ±π — far more robust than starting the iteration from a WLS fit on
    # the raw wrapped observations, which can lock onto the wrong 2π branch. For
    # delay/rate (`rewrap == 0`, no wrapping) we solve the raw system directly.
    if rewrap > 0
        xseed = _spanning_tree_seed(rows, nant, anchors)
        # Temporal warm-start: where a `seed_phase` (e.g. the previous AP's solved
        # node phases) is available, OVERRIDE the per-solve spanning-tree seed with
        # it. The model is used only to pick each observation's 2π branch, and edge
        # predictions `φ_na − φ_nb` are gauge-invariant, so a warm-start from a
        # differently-anchored neighbouring AP is safe. This gives the per-AP adhoc
        # track temporal continuity, so a weakly-constrained station cannot flip
        # between two sub-2π branches AP-to-AP (which the integer-2π unwrap and the
        # smoother both leave intact). Stations the seed does not cover fall back to
        # the spanning-tree estimate.
        if seed_phase !== nothing
            for ant in 1:nant, feed in 1:2
                v = seed_phase[ant, feed]
                isfinite(v) && (xseed[_node(ant, feed, nant)] = float(v))
            end
        end
        model = A * xseed
        bw = similar(b)
        @. bw = b + 2π * round((model - b) / (2π))
        x = weighted_constrained_least_squares(A, bw, w, C, dgauge)
        for _ in 1:rewrap
            model = A * x
            @. bw = b + 2π * round((model - b) / (2π))
            x = weighted_constrained_least_squares(A, bw, w, C, dgauge)
        end
        resid = bw .- A * x
    else
        x = weighted_constrained_least_squares(A, b, w, C, dgauge)
        resid = b .- A * x
    end

    for ant in 1:nant, feed in 1:2
        n = _node(ant, feed, nant)
        if touched[n]
            vals[ant, feed] = x[n]
            cov[ant, feed] = true
        end
    end
    return vals, cov, ncomp, resid
end

# Maximum-weight spanning-tree phase seed (K1). Propagate wrapped edge phases
# from each pin over the parallel-hand (same-node-index, `na == nb`) edges of the
# (station, feed) graph, preferring high-weight edges, to build a globally
# consistent node-phase estimate. Cross-hand rows are excluded from the tree:
# they carry the inter-feed offset, which the tree has no way to place, so their
# nodes are reached through the parallel-hand subgraph (or seeded 0 and resolved
# by the WLS). The estimate is used only to unwrap the observations for the first
# constrained solve, so any edge it cannot place stays 0 — the re-wrap iterations
# refine from there.
function _spanning_tree_seed(rows::Vector{_ObsRow}, nant::Integer, anchors::AbstractVector{<:Integer})
    nnodes = 2 * nant
    x = zeros(Float64, nnodes)
    # Adjacency over parallel-hand edges: neighbor, phase to ADD (φ_v = φ_u + add), weight.
    adj = [Vector{Tuple{Int, Float64, Float64}}() for _ in 1:nnodes]
    for r in rows
        r.na == r.nb || continue
        na = _node(r.a, r.na, nant)
        nb = _node(r.b, r.nb, nant)
        # row: φ_na − φ_nb = r.val ⇒ from na, φ_nb = φ_na − r.val; from nb, φ_na = φ_nb + r.val.
        push!(adj[na], (nb, -r.val, r.w))
        push!(adj[nb], (na, r.val, r.w))
    end
    # Visit strongest edges first so the tree follows high-SNR connections.
    for n in 1:nnodes
        sort!(adj[n]; by = e -> e[3], rev = true)
    end
    # Grow a MAX-WEIGHT spanning tree per component (Prim): repeatedly attach the
    # highest-weight edge from the visited set to an unvisited node. Following the
    # strongest edges (not just any incident edge, as a plain BFS would) keeps the
    # unwrapping on the most reliable, smallest-|Δφ| connections — a hub node
    # directly joined to a far node by a low-SNR, >π edge does not get to define
    # that node's branch.
    visited = falses(nnodes)
    for p in anchors
        (1 <= p <= nnodes && !visited[p]) || continue
        visited[p] = true                       # pinned node phase stays 0
        while true
            best_w = -Inf
            best_u = 0
            best_v = 0
            best_add = 0.0
            for u in 1:nnodes
                visited[u] || continue
                for (v, add, w) in adj[u]
                    (!visited[v] && w > best_w) || continue
                    best_w = w
                    best_u = u
                    best_v = v
                    best_add = add
                end
            end
            best_v == 0 && break               # component exhausted
            x[best_v] = x[best_u] + best_add
            visited[best_v] = true
        end
    end
    return x
end

# ── Generic, model-driven station solve ──────────────────────────────────────
#
# `solve_station_systems!` has no node space of its own: the unknowns are the θ
# COLUMNS the model declares. For each stage-B phase component (a
# `ConstantTerm`/`Delay`/`Rate` × time-seg × tying), `_block_index(plan,
# _feed_node(tying, feed), 1, tseg_id[ti], ant)` is the θ slot a (station, feed,
# time) observation maps to. The plan's segmentation encodes the time basis
# (PerScan → a distinct column per scan; GlobalTime → one column shared across the
# whole track) and its tying the feed fold (PerFeed → distinct feed columns;
# SharedFeeds → one shared column). So the same engine solves a per-scan model
# (columns disjoint per scan ⇒ block-diagonal ⇒ scans solve independently) and a
# model with a track-global inter-feed offset (a column shared across scans
# couples them) — the model is the extension point, this solver just reads the θ
# columns each component declares.
#
# Every correlation product's detection becomes a row of the delay and rate
# systems. A cross-hand row is not special-cased there: the tying alone decides
# what it touches, so `SharedFeeds` reads it as `x_a − x_b` and `PerFeed` as
# `x_{a,p} − x_{b,q}` — and under the default term list the cross-hand delay
# rows are exactly what constrains `rel_delay`'s common mode.
#
# The phase system under a feed-blind model is the exception, because its rows
# are not mutually consistent: the model deliberately carries no feed-relative
# phase (the R–L offset — instrumental constant plus field rotation — is left
# in the data for a downstream polarization fit), so a QQ row sits a
# station-based offset away from its PP sibling, and a cross-hand row adds the
# source's cross-hand phase on top. Fitting all four families to one shared
# column would return a weighted compromise biased toward whichever feed
# carries more weight. `_solve_kind_cols!` therefore augments the feed-blind
# phase system with per-(scan, station) NUISANCE feed-2 offset columns —
# solved so the shared column is the feed-1 phase, then discarded so the
# offset survives in the data — and withholds cross-hand rows from that system
# alone: everything they constrain beyond the parallel hands (the offsets'
# common mode, i.e. the EVPA zero) is discarded anyway. A model whose phase
# component resolves feeds itself describes every row, so the nuisance
# machinery stays off there.
#
# `scans` is a vector of `Baseline × Pol` Detection `DimStack`s (the shape
# `search_scan` returns — see `detection_stack`/`_with_ti`), each carrying its
# `Baseline` lookup's `(a, b)` pairs, per-product feeds derived from its `Pol`
# lookup, and a representative global time index `:ti` in metadata for the
# `tseg_id` lookup. θ slots are ACCUMULATED into (`+=`), matching
# `_pack_station!`, so `rounds > 1` (search on the residual) stays correct.

"""
    detection_stack(D::AbstractMatrix{<:Detection}, bl_pairs, pol_products;
                    ti, freq_rms, time_rms) -> DimStack

Package a plain `[baseline, product]` detection matrix as the `Baseline × Pol`
DimStack shape `search_scan` returns, carrying `ti` (the representative global
time index) in metadata — so a scan built directly (the refine stage, or a
direct `solve_station_systems!` call) has the same shape as
one that came from the search, and every consumer reads pairs/feeds/ti off the
stack uniformly.

`epoch` (hours) is where the phases were measured, which the station solve needs
to read them as constants. Omitting it asserts they sit wherever the model's
rate components are referenced, and is an error when those disagree among
themselves — see `Fringe.scan_phase_epoch`.
"""
function detection_stack(
        D::AbstractMatrix{<:Detection}, bl_pairs, pol_products;
        ti::Integer, epoch::Union{Nothing, Real} = nothing,
        freq_rms::Union{Nothing, Real} = nothing,
        time_rms::Union{Nothing, Real} = nothing,
    )
    gdims = (Baseline(collect(Tuple{Int, Int}, bl_pairs)), Pol(collect(pol_products)))
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
# index, the epoch (hours) the phases are referenced to, and the RMS
# frequency/time spreads the weights need.
_scan_meta(ti, epoch, freq_rms, time_rms) = Dict{Symbol, Any}(
    :ti => Int(ti), :epoch => epoch, :freq_rms => freq_rms, :time_rms => time_rms,
)

# Attach a representative global time index to an existing detection stack (the
# search's own `search_scan` return, which carries no `:ti` — the caller knows
# which window it searched).
_with_ti(
    stack::AbstractDimStack, ti::Integer; epoch::Union{Nothing, Real} = nothing,
    freq_rms::Union{Nothing, Real} = nothing, time_rms::Union{Nothing, Real} = nothing,
) = DimensionalData.rebuild(stack; metadata = _scan_meta(ti, epoch, freq_rms, time_rms))

_scan_bl_pairs(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, Baseline))
_scan_pols(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, Pol))
_scan_feeds(sc::AbstractDimStack) = [correlation_feed_pair(p) for p in _scan_pols(sc)]
_scan_ti(sc::AbstractDimStack) = DimensionalData.metadata(sc)[:ti]::Int
_scan_epoch(sc::AbstractDimStack) = get(DimensionalData.metadata(sc), :epoch, nothing)
_scan_spread(sc::AbstractDimStack, key::Symbol) = get(DimensionalData.metadata(sc), key, nothing)

"""
    solve_station_systems!(θ, scans, components; gauge, opts) -> (ncomp, covered)

Solve the stage-B fringe systems (delay, rate, constant phase) over `scans` and
accumulate the per-(station, feed) values into `θ` at the columns the model
declares. `components` is a vector of `(plan::ComponentPlan, kind::Symbol)` with
`kind ∈ (:delay, :rate, :phase)`. Multiple components of the same kind are summed
per (station, feed) observation: e.g. a feed-common `PerScan × SharedFeeds` term
plus a `GlobalTime × SingleFeed(2)` inter-feed offset both feed the delay
system, so a feed-2 row touches both columns and a stable inter-feed offset is solved
once across the track (bright scans pin it; weak scans inherit it, tying feeds
that would otherwise split). Returns the phase-system component count and
`covered` — the `(station, scan-index)` pairs the solve CONSTRAINS, which is
independent of the gauge (see `Stationization` for how inconsistent rows are
weighted). With a single per-scan/per-feed component per kind and one scan,
each scan's system is independent and solves exactly as it would alone.
"""
function solve_station_systems!(
        θ::AbstractVector, scans, components;
        gauge::AbstractGauge = PinAntenna(1), opts::Stationization = Stationization(),
    )
    ncomp = 0
    # (station, scan-index) pairs the solve CONSTRAINS. A station with no
    # accepted detection in a scan keeps θ = 0 there ⇒ identity gain, and must
    # be FLAGGED downstream rather than silently passed through uncalibrated.
    # Intersected over the solved kinds: a station must be constrained in delay
    # And rate and phase to count as calibrated.
    covered = Set{Tuple{Int, Int}}()
    first_kind = true
    rate_plans = [c[1] for c in components if c[2] === :rate]
    rate_solved = Dict{Int, Float64}()
    # :rate before :phase — a detection's phase is a constant only at the epoch
    # where every rate coordinate vanishes, so a rate referenced to some other
    # epoch has to be subtracted off the phase rows, and that needs it solved.
    for kind in (:delay, :rate, :phase)
        plans = [c[1] for c in components if c[2] === kind]
        isempty(plans) && continue
        nc, cov, solved = _solve_kind_cols!(
            θ, scans, plans, gauge, opts, kind;
            rate_plans = kind === :phase ? rate_plans : ComponentPlan[],
            rate_solved,
        )
        kind === :rate && (rate_solved = solved)
        covered = first_kind ? cov : intersect(covered, cov)
        first_kind = false
        kind === :phase && (ncomp = nc)
    end
    return ncomp, covered
end

# Solve one observable kind across all scans, accumulating into θ. Each detection
# becomes a station-difference row whose a-/b-side touch the sum of all `plans`'
# θ columns for that (station, feed, time) — a feed-common per-scan column and,
# when present, a global feed-offset column. Returns (ncomp, covered, solved),
# `solved` mapping each θ column this kind touched to the value it just added.
#
# `rate_plans`/`rate_solved` are non-empty only for `:phase`, and only matter
# where a rate component's origin differs from the epoch the phases were
# measured at — see `_phase_epoch_offset`.
function _solve_kind_cols!(
        θ::AbstractVector, scans, plans, gauge::AbstractGauge, opts::Stationization, kind::Symbol;
        rate_plans = ComponentPlan[], rate_solved::Dict{Int, Float64} = Dict{Int, Float64}(),
    )
    getval = kind === :delay ? (d -> d.delay) : kind === :rate ? (d -> d.rate) : (d -> d.phase)
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
    rowA = Vector{Int}[]; rowB = Vector{Int}[]
    rval = Float64[]; rw = Float64[]; rcross = Bool[]; rscan = Int[]
    rsta_a = Int[]; rsta_b = Int[]
    # Per row: whether its detection is real (`pfa <= pfa_max`). Only these
    # connect stations into a fringe group; the rest constrain and no more.
    raccept = Bool[]
    for (sidx, sc) in enumerate(scans)
        nbl, npol = size(sc)
        bl_pairs = _scan_bl_pairs(sc)
        feeds = _scan_feeds(sc)
        ti = _scan_ti(sc)
        epoch = _scan_epoch(sc)
        epoch === nothing && _require_common_epoch(rate_plans, ti)
        # Per scan, not per row: the band and duration are properties of the
        # observation, so every row of one scan shares this lever arm.
        σ = kind === :phase ? 1.0 :
            _require_spread(_scan_spread(sc, spread_key), kind, opts.loss)
        σν = kind === :delay ? σ : 1.0
        σt = kind === :rate ? σ : 1.0
        # Stations whose feed-1 phase this scan's parallel rows measure — the
        # set eligible for a nuisance feed-2 offset column. A station observed
        # only on feed 2 (a single-feed receiver, or a feed-1 dropout) gets
        # none: the shared column and the offset would be an exactly degenerate
        # pair. Its shared column then carries its feed-2 phase referenced to
        # the reference station's feed-2 frame — the parallel hands cannot
        # separate such a station's phase from the offsets' common mode, and
        # the nuisance-block gauge (see `_solve_tagged_system`) pins that mode
        # at the reference rather than leaving it to the min-norm completion.
        f1 = Set{Int}()
        if feedblind
            for bi in 1:nbl, p in 1:npol
                sc[bi, p].valid || continue
                fa, fb = feeds[p]
                (fa == 1 && fb == 1) || continue
                a, b = bl_pairs[bi]
                a == b && continue
                push!(f1, a); push!(f1, b)
            end
        end
        for bi in 1:nbl, p in 1:npol
            det = sc[bi, p]
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
            nsA = Int[]; nsB = Int[]
            for plan in plans
                na = _feed_node(plan.tying, fa)
                ca = na == 0 ? 0 : _block_index(plan, na, 1, plan.tseg_id[ti], a)
                ca != 0 && push!(nsA, getnode(ca, a, fa, sidx))
                nb = _feed_node(plan.tying, fb)
                cb = nb == 0 ? 0 : _block_index(plan, nb, 1, plan.tseg_id[ti], b)
                cb != 0 && push!(nsB, getnode(cb, b, fb, sidx))
            end
            (isempty(nsA) || isempty(nsB)) && continue
            # The nuisance column joins the side after the model columns, so a
            # row side's first entry is always a model column (`_seed_tagged`
            # reads sides that way).
            if feedblind
                fa == 2 && a in f1 && push!(nsA, nuisnode(sidx, a))
                fb == 2 && b in f1 && push!(nsB, nuisnode(sidx, b))
            end
            push!(rowA, nsA); push!(rowB, nsB)
            push!(
                rval, getval(det) -
                    _phase_epoch_offset(rate_plans, rate_solved, ti, epoch, a, fa) +
                    _phase_epoch_offset(rate_plans, rate_solved, ti, epoch, b, fb),
            )
            push!(
                rw, _row_weight(
                    _sigma_stat(kind, det.snr, σν, σt), cross ? sys_cross : sys_par,
                    accept ? 1.0 : opts.weak_sys_scale,
                ),
            )
            push!(raccept, accept)
            push!(rcross, cross); push!(rscan, sidx)
            push!(rsta_a, a); push!(rsta_b, b)
        end
    end
    isempty(rowA) && return (0, Set{Tuple{Int, Int}}(), Dict{Int, Float64}())

    # Robust solve: IRLS over `opts.loss`, rescaling each row's noise-model
    # weight by the loss's derivative at that row's normalized residual (see
    # `Stationization`). No row leaves the system, so the graph's connectivity —
    # and hence which stations are solvable — is fixed before the first solve
    # and cannot be changed by the fit.
    #
    # Nesting: IRLS outer, the phase system's 2π re-wrap inner (it lives inside
    # `_solve_tagged_system`), so every IRLS pass sees a converged branch
    # assignment. Reversing them would downweight rows whose residual is still a
    # wrap away from its final value.
    w = copy(rw)
    nuisance = node_col .== 0
    local x, ncomp
    x, ncomp, resid = _solve_tagged_system(
        rowA, rowB, rval, w, rcross, length(node_col),
        node_feed, node_station, node_scan, gauge; rewrap = rewrap, nuisance, raccept,
    )
    if !(opts.loss isa LeastSquares)
        for _ in 1:max(opts.irls_iters, 0)
            _irls_weights!(w, rw, opts.loss, opts.loss_scale, resid) || break
            x, ncomp, resid = _solve_tagged_system(
                rowA, rowB, rval, w, rcross, length(node_col),
                node_feed, node_station, node_scan, gauge; rewrap = rewrap, nuisance, raccept,
            )
        end
    end
    solved = Dict{Int, Float64}()
    for n in eachindex(node_col)
        node_col[n] == 0 && continue          # nuisance offset: solved, discarded
        θ[node_col[n]] += x[n]
        solved[node_col[n]] = x[n]
    end
    return ncomp, _covered_stations(rsta_a, rsta_b, rscan, raccept), solved
end

# The phase a rate component contributes at `epoch` to one (station, feed) —
# `2π·ṙ·(epoch − t0_k)` over the rate columns, in radians.
#
# A detection's phase is the phase at the epoch its search referenced, and the
# station solve reads it as a sum of CONSTANTS. That holds only where every rate
# coordinate is zero, i.e. at each rate component's own origin. A component
# segmented like the constants beside it (the default: everything `PerScan`) has
# its origin exactly there and contributes nothing here — the subtraction is
# identically zero and the rows are the measured phases unchanged. A component
# segmented more coarsely (a track-global inter-feed rate against per-scan
# constants) is referenced elsewhere, and its share of the measured phase is
# removed here rather than being left for a per-scan constant to absorb.
#
# `solved` is this round's rate increment per θ column, which is what the phases
# of this round — measured on the residual of the previous one — contain.
function _phase_epoch_offset(rate_plans, solved, ti::Integer, epoch, a::Integer, feed::Integer)
    isempty(rate_plans) && return 0.0
    off = 0.0
    for plan in rate_plans
        node = _feed_node(plan.tying, feed)
        node == 0 && continue
        seg = plan.tseg_id[ti]
        Δt = epoch === nothing ? 0.0 : (Float64(epoch) - Float64(plan.tstate[seg]))
        Δt == 0.0 && continue
        col = _block_index(plan, node, 1, seg, a)
        col == 0 && continue
        off += 2π * get(solved, col, 0.0) * Δt * 3600.0
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
        isapprox(o, o1; atol = 1.0e-9) || error(
            "solve_station_systems!: the rate components are referenced to different " *
                "epochs at time index $ti ($o1 h vs $o h), so no single epoch makes a " *
                "detection's phase a sum of constants. Record the epoch the phases were " *
                "measured at (`detection_stack(...; epoch)`) so the rates referenced " *
                "elsewhere can be subtracted from the phase rows.",
        )
    end
    return nothing
end

# The (station, scan) pairs this system actually calibrates: those carrying at
# least one ACCEPTED detection, and so constrained by the solve rather than left
# at θ = 0 (identity gain). Everything else is flagged downstream.
#
# `raccept` is what keeps this honest. Every measured cell contributes a row, so
# reading coverage off the rows alone would call a station calibrated on the
# strength of a noise peak — which sits at an arbitrary delay and fixes nothing.
# Only a detection at or below `pfa_max` joins stations into a fringe group.
#
# Coverage does not depend on the reference antenna, and so is invariant to the
# gauge pin. Each connected component of the (station, feed) graph carries its own
# arbitrary additive zero, but a correction enters the data only as the difference
# `g_a − g_b` along a baseline, and a baseline exists only within a component — so
# a component's gauge cancels wherever it is applied, whether or not the reference
# is one of its stations. Requiring reference connectivity instead would discard
# every station of a scan the reference happens to sit out, including scans whose
# own closure is perfectly well determined.
#
# Connectivity and weighting are separate questions: `pfa_max` decides which rows
# are real detections and therefore what the graph looks like, and the robust loss
# then arbitrates inconsistency AMONG those rows without removing any. A station
# with no accepted detection is uncalibrated no matter how the surviving rows are
# weighted.
function _covered_stations(rsta_a, rsta_b, rscan, raccept)
    cov = Set{Tuple{Int, Int}}()
    for i in eachindex(rsta_a, rsta_b, rscan, raccept)
        raccept[i] || continue
        push!(cov, (rsta_a[i], rscan[i]))
        push!(cov, (rsta_b[i], rscan[i]))
    end
    return cov
end

# Constrained WLS over a tagged node graph (the column-space generalization of
# `_solve_observable`). `node_feed`/`node_station`/`node_scan` tag each local node
# (feed 0 = shared by both feeds; scan 0 = global column) so the gauge reproduces
# `_solve_observable`'s tie-breaks in the per-scan case. Rows may touch more than
# one column per side (a feed-common column plus a global feed-offset column).
# After the per-component gauge rows, any residual gauge freedom (e.g. the per-scan
# absolute level once scans are globally coupled) is removed by a minimum-norm
# null-space pin — so the engine is well-posed for any model `plan_parameters` can
# flatten, with no model-specific gauge code. `rcross` marks cross-hand rows,
# which are excluded from the unwrap seed's spanning tree.
function _solve_tagged_system(
        rowA, rowB, rval, rw, rcross, nnodes,
        node_feed, node_station, node_scan, gauge::AbstractGauge; rewrap::Integer,
        nuisance::Union{Nothing, AbstractVector{Bool}} = nothing,
        raccept::Union{Nothing, AbstractVector{Bool}} = nothing,
    )
    # Union the columns of each (possibly multi-term) row into one component —
    # ACCEPTED rows only. A detection above `pfa_max` sits at an arbitrary
    # noise peak, so letting it define graph structure would hand the component
    # count, the gauge pins and the unwrap anchors to noise; the two-tier
    # contract is a hard connectivity cut, with weak rows constraining the fit
    # (σ-inflated) and no more. A node reached only by weak rows joins no
    # component and gets no gauge row: its value is determined RELATIVE to the
    # accepted structure by those weak rows — the "constrains a parameter
    # nothing else constrains" case — or, for a genuinely isolated island,
    # falls to the min-norm completion; `_covered_stations` already excludes it
    # from coverage either way.
    edges = Tuple{Int, Int}[]
    for i in eachindex(rowA)
        (raccept === nothing || raccept[i]) || continue
        ns = vcat(rowA[i], rowB[i])
        for k in 2:length(ns)
            push!(edges, (ns[1], ns[k]))
        end
    end
    compid, ncomp, _ = connected_components(nnodes, edges)

    # Node tags: feed 0 marks a column shared by both feeds, so anything other
    # than 2 counts as feed-1/shared for gauge purposes.
    station_of(n) = node_station[n]
    feed_of(n) = node_feed[n]

    # Total row weight on each node — the score the pin falls back to when a
    # component holds no reference node.
    nodew = zeros(Float64, nnodes)
    for i in eachindex(rowA)
        for n in rowA[i]
            nodew[n] += rw[i]
        end
        for n in rowB[i]
            nodew[n] += rw[i]
        end
    end

    # One constraint row per component, from `gauge`; `anchors` names a real node
    # per component for the phase-unwrap seed.
    comps = Vector{Int}[]
    for c in 1:ncomp
        comp = [n for n in 1:nnodes if compid[n] == c]
        isempty(comp) && continue
        push!(comps, comp)
    end
    anchors = [gauge_anchor(gauge, cn, nodew, station_of, feed_of) for cn in comps]
    nrow = length(rowA)
    A = zeros(Float64, nrow, nnodes)
    b = zeros(Float64, nrow)
    w = zeros(Float64, nrow)
    @inbounds for i in 1:nrow
        for n in rowA[i]
            A[i, n] += 1.0
        end
        for n in rowB[i]
            A[i, n] -= 1.0
        end
        b[i] = rval[i]
        w[i] = rw[i]
    end
    Cp = zeros(eltype(A), length(comps), nnodes)
    for (j, cn) in enumerate(comps)
        gauge_row!(view(Cp, j, :), gauge, cn, nodew, station_of, feed_of)
    end
    # Nuisance-block gauge: the nuisance offset columns of one scan carry a
    # common-mode freedom the data cannot fix — shifting them together, along
    # with the shared column of every station observed only on feed 2, changes
    # no row. Left to the min-norm completion it would land partly on those
    # stations' shared columns, i.e. in the APPLIED correction, so it is pinned
    # like any other gauge freedom: one row per (component, scan) group of
    # nuisance nodes, through `gauge`, which prefers the ranked reference — a
    # feed-2-only station's phase is thereby referenced to the reference
    # station's feed-2 frame, deterministically.
    if nuisance !== nothing && any(nuisance)
        groups = Dict{Tuple{Int, Int}, Vector{Int}}()
        for n in 1:nnodes
            nuisance[n] || continue
            # A nuisance node outside every accepted component carries no
            # common-mode freedom worth pinning; its weak rows (or the
            # min-norm completion) settle it.
            compid[n] == 0 && continue
            push!(get!(groups, (compid[n], node_scan[n]), Int[]), n)
        end
        keyorder = sort!(collect(keys(groups)))
        Cn = zeros(eltype(A), length(keyorder), nnodes)
        for (j, k) in enumerate(keyorder)
            gauge_row!(view(Cn, j, :), gauge, groups[k], nodew, station_of, feed_of)
        end
        Cp = vcat(Cp, Cn)
    end
    # Min-norm completion: fix any gauge freedom the per-component rows leave (the
    # null space of [A; Cp]). Empty for the per-scan model (those rows suffice ⇒
    # byte-identical), non-trivial once a global column couples scans.
    nb = nullspace(vcat(A, Cp))
    C = size(nb, 2) > 0 ? vcat(Cp, permutedims(nb)) : Cp
    dgauge = zeros(eltype(C), size(C, 1))

    if rewrap > 0
        xseed = _seed_tagged(rowA, rowB, rval, rw, rcross, anchors, nnodes)
        model = A * xseed
        bw = similar(b)
        @. bw = b + 2π * round((model - b) / (2π))
        x = weighted_constrained_least_squares(A, bw, w, C, dgauge)
        for _ in 1:rewrap
            model = A * x
            @. bw = b + 2π * round((model - b) / (2π))
            x = weighted_constrained_least_squares(A, bw, w, C, dgauge)
        end
        resid = bw .- A * x
    else
        x = weighted_constrained_least_squares(A, b, w, C, dgauge)
        resid = b .- A * x
    end

    return x, ncomp, resid, compid
end

# Max-weight spanning-tree phase seed in local-node space (column-space twin of
# `_spanning_tree_seed`): propagate wrapped parallel-hand edge phases from each
# pin to unwrap the first constrained solve. Every parallel-hand row is a tree
# edge between the first column of each side — the primary model column, by row
# construction. Any further columns on a side (a global feed offset, a nuisance
# feed-2 offset) displace the edge phase by less than a wrap, which is all a
# branch-picking seed needs; the constrained WLS + re-wrap iterations resolve
# them exactly.
function _seed_tagged(rowA, rowB, rval, rw, rcross, anchors, nnodes::Integer)
    x = zeros(Float64, nnodes)
    adj = [Vector{Tuple{Int, Float64, Float64}}() for _ in 1:nnodes]
    for i in eachindex(rowA)
        rcross[i] && continue
        na, nb = rowA[i][1], rowB[i][1]
        push!(adj[na], (nb, -rval[i], rw[i]))
        push!(adj[nb], (na, rval[i], rw[i]))
    end
    for n in 1:nnodes
        sort!(adj[n]; by = e -> e[3], rev = true)
    end
    visited = falses(nnodes)
    for p in anchors
        (1 <= p <= nnodes && !visited[p]) || continue
        visited[p] = true
        while true
            best_w = -Inf
            best_u = 0; best_v = 0; best_add = 0.0
            for u in 1:nnodes
                visited[u] || continue
                for (v, add, ww) in adj[u]
                    (!visited[v] && ww > best_w) || continue
                    best_w = ww; best_u = u; best_v = v; best_add = add
                end
            end
            best_v == 0 && break
            x[best_v] = x[best_u] + best_add
            visited[best_v] = true
        end
    end
    return x
end

"""
    station_closure_residuals(detections, bl_pairs, pol_products;
                              observable = :phase, product = 1, pfa_max) -> Vector

For every closed triangle of baselines present in `bl_pairs`, the residual
closure quantity of the chosen `observable` (`:delay`/`:rate`/`:phase`) using the
*measured* detections — i.e. the signed sum around the triangle that station-based
quantities must cancel. For noiseless station-differenced data these are ≈ 0
(including mixed-hand triangles); large values flag non-closing data. This is a
property of the data alone, so no solution is needed to evaluate it.

A triangle counts only when all three legs are accepted detections
(`pfa <= pfa_max`); a leg above that threshold carries an arbitrary delay, which
would enter the sum as noise rather than as evidence of non-closure.
"""
function station_closure_residuals(
        detections::AbstractMatrix{<:Detection},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString};
        observable::Symbol = :phase,
        product::Integer = 1,
        pfa_max::Real = Stationization().pfa_max,
    )
    getval = observable === :delay ? (d -> d.delay) :
        observable === :rate ? (d -> d.rate) :
        observable === :phase ? (d -> d.phase) :
        error("observable must be :delay, :rate or :phase")
    blindex = Dict((bl_pairs[bi][1], bl_pairs[bi][2]) => bi for bi in eachindex(bl_pairs))
    res = Float64[]
    ants = sort(unique(Iterators.flatten(bl_pairs)))
    for i in eachindex(ants), j in eachindex(ants), k in eachindex(ants)
        a, b, c = ants[i], ants[j], ants[k]
        (a < b < c) || continue
        (haskey(blindex, (a, b)) && haskey(blindex, (b, c)) && haskey(blindex, (a, c))) || continue
        dab, dbc, dac = detections[blindex[(a, b)], product], detections[blindex[(b, c)], product], detections[blindex[(a, c)], product]
        # Accepted detections only: a triangle closed through a noise peak sits at
        # an arbitrary delay and its residual measures nothing.
        (dab.pfa <= pfa_max && dbc.pfa <= pfa_max && dac.pfa <= pfa_max) || continue
        s = getval(dab) + getval(dbc) - getval(dac)
        observable === :phase && (s = rem2pi(s, RoundNearest))
        push!(res, s)
    end
    return res
end
