# ── Per-feed stationization with closure ─────────────────────────────────────
#
# Turn per-baseline fringe detections into per-(station, feed) delays, rates and
# phases by weighted least squares on the (station, feed) graph (2·nant nodes).
# Every observable differences station quantities with the SAME incidence:
#
#     baseline (a,b), product p with feeds (fa, fb) = correlation_feed_pair(p):
#         delay_ab^p = τ_{a,fa} − τ_{b,fb}
#         rate_ab^p  = ṙ_{a,fa} − ṙ_{b,fb}
#         phase_ab^p = φ_{a,fa} − φ_{b,fb}
#
# Using ALL FOUR products (not just parallel hands) is deliberate: cross-hand
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
# freedom per observable; we pin the reference station's node per component. Cross
# hands merge the two feeds into a single component, so the inter-feed offset is
# fixed by the data and needs no pin of its own.

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

The identity element of [`AbstractRobustLoss`](@ref): every row keeps its full
noise-model weight. The IRLS iteration then converges on its first pass, so a
solve under `LeastSquares` is plain weighted least squares — this is how a
caller asks for no robust downweighting at all.
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
    Stationization(; snr_min, phase_rewrap_iters, loss,
                     loss_scale, irls_iters, systematic_delay, systematic_rate)

Options for [`solve_station_systems!`](@ref). `snr_min` drops detections below
this SNR; `phase_rewrap_iters` re-wraps phase residuals to handle differences
exceeding ±π.

Every correlation product a detection survives the SNR gate on contributes a row
to every observable's system — parallel and cross hands alike. Which parameters
a row touches is the MODEL's business, not this type's: under `SharedFeeds` both
sides map to one per-station column, under `PerFeed` to the row's own two feed
columns. Withholding cross-hand rows would silently substitute a different
estimator for the one the model declares.

`loss`/`loss_scale`/`irls_iters` control robust downweighting. After each solve,
every row's weight is rescaled by `robust_weight(loss, (z/loss_scale)^2)` at its
noise-normalized residual `z = resid·√w`, and the system is re-solved — up to
`irls_iters` times. A false fringe (e.g. tone/crosstalk correlation on a
co-located telescope pair) is closure-inconsistent with the true detections, so
it lands far out in `z` and is suppressed instead of dragging its stations'
solutions. `loss = LeastSquares()` keeps every row at full weight.

Because the weights come from the noise model rather than from a spread fitted
to the residuals, `loss_scale` means the same thing in every system regardless
of how many rows it holds — a two-station scan and a full-array scan are cut at
the same effective threshold.

`systematic_delay`/`systematic_rate` (seconds / Hz) are a systematic-error floor
added in quadrature to the CRB uncertainty of delay/rate rows:
`w = 1/(σ_CRB² + systematic²)`. Without a floor, an array with no real
systematics (or synthetic data) drives `z` to the numerical noise floor, where
the loss has no real outlier to find and merely reweights rounding noise; a
nonzero floor keeps the residual distribution meaningful at whatever precision
the instrument actually delivers. Phase rows carry no systematic term.
"""
Base.@kwdef struct Stationization
    snr_min::Float64 = 6.0
    phase_rewrap_iters::Int = 3
    loss::AbstractRobustLoss = SoftL1()
    loss_scale::Float64 = 8.0
    irls_iters::Int = 5
    systematic_delay::Float64 = 0.0
    systematic_rate::Float64 = 0.0
end

# IRLS weight update. `w` (mutated) is the effective weight vector fed to the
# solver, `w0` the untouched noise-model weights that define the σ scale.
# Returns whether any weight moved enough to be worth another solve.
#
# Kept behind its own function so the loss type — an abstract field on
# `Stationization` — is resolved ONCE per iteration rather than per row.
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
# The spreads matter ONLY for the robust loss. A weighted least-squares solution
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

# Inverse-variance weight: statistical σ with a systematic floor in quadrature.
_row_weight(σ_stat::Real, σ_sys::Real) = inv(σ_stat^2 + σ_sys^2)

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
        rows::Vector{_ObsRow}, nant::Integer, ref_ant::Integer, opts::Stationization;
        rewrap::Integer,
    )
    vals, cov, ncomp, resid = _solve_observable(rows, nant, ref_ant; rewrap = rewrap)
    (opts.loss isa LeastSquares || isempty(rows)) && return vals, cov, ncomp
    w0 = [r.w for r in rows]
    w = copy(w0)
    for _ in 1:max(opts.irls_iters, 0)
        _irls_weights!(w, w0, opts.loss, opts.loss_scale, resid) || break
        vals, cov, ncomp, resid =
            _solve_observable(rows, nant, ref_ant; rewrap = rewrap, weights = w)
    end
    return vals, cov, ncomp
end

# Solve one observable's WLS system on the (station, feed) graph. Returns
# (values::(nant,2), covered::(nant,2), ncomp, resid). `weights` overrides
# the rows' own noise-model weights (the IRLS driver's reweighted vector);
# `resid` comes back aligned with `rows`, already 2π-branch-corrected for a
# re-wrapped system, so the driver can normalize it without redoing the unwrap.
function _solve_observable(
        rows::Vector{_ObsRow}, nant::Integer, ref_ant::Integer;
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

    # Gauge pins: one reference node per component (prefer ref_ant's feed-1, then
    # feed-2, then the lowest node in the component).
    pins = Int[]
    for c in 1:ncomp
        comp_nodes = findall(==(c), compid)
        r1 = _node(ref_ant, 1, nant)
        r2 = _node(ref_ant, 2, nant)
        rn = r1 in comp_nodes ? r1 : (r2 in comp_nodes ? r2 : minimum(comp_nodes))
        push!(pins, rn)
    end
    # Pin every UNTOUCHED node — an (antenna, feed) with no observation in this
    # solve, e.g. a station that dropped out. Its design column is all-zero, which
    # would make the constrained QR system rank-deficient and corrupt the solve
    # for the stations that DO have data. Pinning it to 0 (its value is discarded;
    # only `touched` cells are returned) keeps the system full-rank and well-posed.
    for n in 1:nnodes
        touched[n] || n in pins || push!(pins, n)
    end

    A = zeros(Float64, nrow, nnodes)
    b = zeros(Float64, nrow)
    w = zeros(Float64, nrow)
    for (i, r) in enumerate(rows)
        A[i, _node(r.a, r.na, nant)] += 1.0
        A[i, _node(r.b, r.nb, nant)] -= 1.0
        b[i] = r.val
        w[i] = weights === nothing ? r.w : weights[i]
    end
    C = zeros(Float64, length(pins), nnodes)
    for (j, p) in enumerate(pins)
        C[j, p] = 1.0
    end
    dgauge = zeros(Float64, length(pins))

    # Phase re-wrap: unwrap observations toward a model and re-solve, so
    # station-difference phases exceeding ±π are handled. The first model comes
    # from a maximum-weight spanning-tree traversal of the (station, feed) graph
    # (K1): propagating wrapped edge phases from each pin gives a globally
    # consistent unwrap that is correct even when a true station-difference
    # exceeds ±π — far more robust than starting the iteration from a WLS fit on
    # the raw wrapped observations, which can lock onto the wrong 2π branch. For
    # delay/rate (`rewrap == 0`, no wrapping) we solve the raw system directly.
    if rewrap > 0
        xseed = _spanning_tree_seed(rows, nant, pins)
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
function _spanning_tree_seed(rows::Vector{_ObsRow}, nant::Integer, pins::AbstractVector{<:Integer})
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
    for p in pins
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
# SharedFeeds → one shared column). So the SAME engine solves a per-scan model
# (columns disjoint per scan ⇒ block-diagonal ⇒ scans solve independently) and a
# model with a track-global inter-feed offset (a column shared across scans
# couples them) — the model is the extension point, this solver just reads the θ
# columns each component declares.
#
# EVERY correlation product's detection becomes a row of EVERY observable's
# system. A cross-hand row is not special-cased and is never withheld: the tying
# alone decides what it touches, so `SharedFeeds` reads it as `x_a − x_b` and
# `PerFeed` as `x_{a,p} − x_{b,q}`. Dropping such rows would fit a different
# estimator than the model describes, and would hide a violated tying assumption
# that belongs in the residuals where the robust loss can act on it.
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
"""
function detection_stack(
        D::AbstractMatrix{<:Detection}, bl_pairs, pol_products;
        ti::Integer, freq_rms::Union{Nothing, Real} = nothing,
        time_rms::Union{Nothing, Real} = nothing,
    )
    gdims = (Baseline(collect(Tuple{Int, Int}, bl_pairs)), Pol(collect(pol_products)))
    layers = (;
        delay = DimArray(getfield.(D, :delay), gdims),
        rate = DimArray(getfield.(D, :rate), gdims),
        phase = DimArray(getfield.(D, :phase), gdims),
        amp = DimArray(getfield.(D, :amp), gdims),
        snr = DimArray(getfield.(D, :snr), gdims),
        valid = DimArray(getfield.(D, :valid), gdims),
    )
    return DimensionalData.DimStack(
        layers; metadata = _scan_meta(ti, freq_rms, time_rms),
    )
end

# A detection stack's scan-level provenance: the representative global time
# index plus the RMS frequency/time spreads its weights need.
_scan_meta(ti, freq_rms, time_rms) = Dict{Symbol, Any}(
    :ti => Int(ti), :freq_rms => freq_rms, :time_rms => time_rms,
)

# Attach a representative global time index to an existing detection stack (the
# search's own `search_scan` return, which carries no `:ti` — the caller knows
# which window it searched).
_with_ti(
    stack::AbstractDimStack, ti::Integer;
    freq_rms::Union{Nothing, Real} = nothing, time_rms::Union{Nothing, Real} = nothing,
) = DimensionalData.rebuild(stack; metadata = _scan_meta(ti, freq_rms, time_rms))

_scan_bl_pairs(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, Baseline))
_scan_pols(sc::AbstractDimStack) = collect(DimensionalData.lookup(sc, Pol))
_scan_feeds(sc::AbstractDimStack) = [correlation_feed_pair(p) for p in _scan_pols(sc)]
_scan_ti(sc::AbstractDimStack) = DimensionalData.metadata(sc)[:ti]::Int
_scan_spread(sc::AbstractDimStack, key::Symbol) = get(DimensionalData.metadata(sc), key, nothing)

"""
    solve_station_systems!(θ, scans, components; ref_ant, opts) -> (ncomp, covered)

Solve the stage-B fringe systems (delay, rate, constant phase) over `scans` and
accumulate the per-(station, feed) values into `θ` at the columns the model
declares. `components` is a vector of `(plan::ComponentPlan, kind::Symbol)` with
`kind ∈ (:delay, :rate, :phase)`. Multiple components of the SAME kind are summed
per (station, feed) observation: e.g. a feed-common `PerScan × SharedFeeds` term
plus a `GlobalTime × FeedComponent(2)` inter-feed offset both feed the delay
system, so a feed-2 row touches both columns and a stable inter-feed offset is solved
once across the track (bright scans pin it; weak scans inherit it, tying feeds
that would otherwise split). Returns the phase-system component count and
`covered` — the `(station, scan-index)` pairs
carrying a transferable solution (see `Stationization` for how inconsistent rows
are weighted). With a single per-scan/per-feed component per kind and one scan,
each scan's system is independent and solves exactly as it would alone.
"""
function solve_station_systems!(
        θ::AbstractVector, scans, components;
        ref_ant::Integer = 1, opts::Stationization = Stationization(),
    )
    ncomp = 0
    # (station, scan-index) pairs carrying a TRANSFERABLE solution — the
    # EHT-HOPS flag criterion, inverted: a station outside the reference's
    # fringe group in a scan is uncalibrated there (its θ is gauged to some
    # other arbitrary pin, or stays 0 ⇒ identity gain) and must be FLAGGED
    # downstream, not silently passed through. Intersected over the solved
    # kinds: a station must be constrained in delay AND rate AND phase to
    # count as calibrated.
    covered = Set{Tuple{Int, Int}}()
    first_kind = true
    for kind in (:delay, :rate, :phase)
        plans = [c[1] for c in components if c[2] === kind]
        isempty(plans) && continue
        nc, cov = _solve_kind_cols!(θ, scans, plans, ref_ant, opts, kind)
        covered = first_kind ? cov : intersect(covered, cov)
        first_kind = false
        kind === :phase && (ncomp = nc)
    end
    return ncomp, covered
end

# Solve one observable kind across all scans, accumulating into θ. Each detection
# becomes a station-difference row whose a-/b-side touch the sum of all `plans`'
# θ columns for that (station, feed, time) — a feed-common per-scan column and,
# when present, a global feed-offset column. Returns (ncomp, covered).
function _solve_kind_cols!(
        θ::AbstractVector, scans, plans, ref_ant::Integer, opts::Stationization, kind::Symbol,
    )
    getval = kind === :delay ? (d -> d.delay) : kind === :rate ? (d -> d.rate) : (d -> d.phase)
    sys_err = kind === :delay ? opts.systematic_delay : kind === :rate ? opts.systematic_rate : 0.0
    spread_key = kind === :delay ? :freq_rms : :time_rms
    rewrap = kind === :phase ? opts.phase_rewrap_iters : 0

    colnode = Dict{Int, Int}()               # θ column → local node id
    node_col = Int[]                         # local node → θ column
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

    # Rows in θ-column space: each side is the list of θ columns whose sum is
    # that station's value for this observable (+1 on a-side, −1 on b-side).
    rowA = Vector{Int}[]; rowB = Vector{Int}[]
    rval = Float64[]; rw = Float64[]; rcross = Bool[]; rscan = Int[]
    rsta_a = Int[]; rsta_b = Int[]
    for (sidx, sc) in enumerate(scans)
        nbl, npol = size(sc)
        bl_pairs = _scan_bl_pairs(sc)
        feeds = _scan_feeds(sc)
        ti = _scan_ti(sc)
        # Per SCAN, not per row: the band and duration are properties of the
        # observation, so every row of one scan shares this lever arm.
        σ = kind === :phase ? 1.0 :
            _require_spread(_scan_spread(sc, spread_key), kind, opts.loss)
        σν = kind === :delay ? σ : 1.0
        σt = kind === :rate ? σ : 1.0
        for bi in 1:nbl, p in 1:npol
            det = sc[bi, p]
            (det.valid && det.snr >= opts.snr_min) || continue
            a, b = bl_pairs[bi]
            a == b && continue
            fa, fb = feeds[p]
            cross = fa != fb
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
            push!(rowA, nsA); push!(rowB, nsB)
            push!(rval, getval(det))
            push!(rw, _row_weight(_sigma_stat(kind, det.snr, σν, σt), sys_err))
            push!(rcross, cross); push!(rscan, sidx)
            push!(rsta_a, a); push!(rsta_b, b)
        end
    end
    isempty(rowA) && return (0, Set{Tuple{Int, Int}}())

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
    local x, ncomp, compid
    x, ncomp, resid, compid = _solve_tagged_system(
        rowA, rowB, rval, w, rcross, length(node_col),
        node_feed, node_station, node_scan, ref_ant; rewrap = rewrap,
    )
    if !(opts.loss isa LeastSquares)
        for _ in 1:max(opts.irls_iters, 0)
            _irls_weights!(w, rw, opts.loss, opts.loss_scale, resid) || break
            x, ncomp, resid, compid = _solve_tagged_system(
                rowA, rowB, rval, w, rcross, length(node_col),
                node_feed, node_station, node_scan, ref_ant; rewrap = rewrap,
            )
        end
    end
    @inbounds for n in eachindex(node_col)
        θ[node_col[n]] += x[n]
    end
    return ncomp, _covered_stations(rowA, rsta_a, rsta_b, rscan, compid, node_station, ref_ant)
end

# The (station, scan) pairs this system actually calibrates: those reached by a
# row inside the REFERENCE's connected component. This is EHT-HOPS's fringe-group
# criterion — a station outside the reference's group of mutually-linked stations
# has a solution, but one gauged to a different arbitrary pin, so it is not
# transferable and must be flagged rather than silently applied.
#
# Connectivity and weighting are separate questions: the `snr_min` gate decides
# which rows are real detections and therefore what the graph looks like, and the
# robust loss then arbitrates inconsistency AMONG those rows without removing any.
# A station with no reference-linked detection is uncalibrated no matter how the
# surviving rows are weighted.
#
# With the reference absent from the system entirely there is no group to be
# transferable to, so nothing is covered.
function _covered_stations(rowA, rsta_a, rsta_b, rscan, compid, node_station, ref_ant)
    cov = Set{Tuple{Int, Int}}()
    # EVERY component holding a reference node, not just one: under a per-scan
    # model the scans are block-diagonal, so each scan contributes its own
    # reference-linked component. A track-global column instead fuses them into
    # one — either way this is "the groups the reference reaches".
    refcomps = Set(compid[n] for n in eachindex(node_station) if node_station[n] == ref_ant)
    isempty(refcomps) && return cov
    for i in eachindex(rowA)
        # A row's columns are unioned into one component, so any of its nodes
        # answers for the whole row.
        compid[first(rowA[i])] in refcomps || continue
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
# After the explicit reference pins, any residual gauge freedom (e.g. the per-scan
# absolute level once scans are globally coupled) is removed by a minimum-norm
# null-space pin — so the engine is well-posed for any model `plan_parameters` can
# flatten, with no model-specific gauge code. `rcross` marks cross-hand rows,
# which are excluded from the unwrap seed's spanning tree.
function _solve_tagged_system(
        rowA, rowB, rval, rw, rcross, nnodes,
        node_feed, node_station, node_scan, ref_ant; rewrap::Integer,
    )
    # Union the columns of each (possibly multi-term) row into one component.
    edges = Tuple{Int, Int}[]
    for i in eachindex(rowA)
        ns = vcat(rowA[i], rowB[i])
        for k in 2:length(ns)
            push!(edges, (ns[1], ns[k]))
        end
    end
    compid, ncomp, _ = connected_components(nnodes, edges)

    # feed-1-or-shared nodes sort before feed-2; then by station, then scan.
    nodekey(n) = (node_feed[n] == 2 ? 1 : 0, node_station[n], node_scan[n])
    is_ref(n) = node_station[n] == ref_ant
    is_feed1(n) = node_feed[n] != 2

    pins = Int[]
    # One reference pin per component: prefer ref_ant feed-1/shared, then feed-2,
    # then the lowest (feed, station, scan) node.
    for c in 1:ncomp
        comp = [n for n in 1:nnodes if compid[n] == c]
        isempty(comp) && continue
        r1 = findfirst(n -> is_ref(n) && is_feed1(n), comp)
        r2 = findfirst(n -> is_ref(n) && node_feed[n] == 2, comp)
        pin = r1 !== nothing ? comp[r1] :
            r2 !== nothing ? comp[r2] : comp[argmin(map(nodekey, comp))]
        push!(pins, pin)
    end
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
    Cp = zeros(Float64, length(pins), nnodes)
    for (j, p) in enumerate(pins)
        Cp[j, p] = 1.0
    end
    # Min-norm completion: pin any gauge freedom the explicit pins leave (the null
    # space of [A; Cp]). Empty for the per-scan model (explicit pins suffice ⇒
    # byte-identical), non-trivial once a global column couples scans.
    nb = nullspace(vcat(A, Cp))
    C = size(nb, 2) > 0 ? vcat(Cp, permutedims(nb)) : Cp
    dgauge = zeros(Float64, size(C, 1))

    if rewrap > 0
        xseed = _seed_tagged(rowA, rowB, rval, rw, rcross, pins, nnodes)
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
# pin to unwrap the first constrained solve. Only single-column-per-side
# parallel-hand rows are tree edges; multi-term (global-offset) rows are left to
# the constrained WLS + re-wrap iterations.
function _seed_tagged(rowA, rowB, rval, rw, rcross, pins, nnodes::Integer)
    x = zeros(Float64, nnodes)
    adj = [Vector{Tuple{Int, Float64, Float64}}() for _ in 1:nnodes]
    for i in eachindex(rowA)
        (!rcross[i] && length(rowA[i]) == 1 && length(rowB[i]) == 1) || continue
        na, nb = rowA[i][1], rowB[i][1]
        push!(adj[na], (nb, -rval[i], rw[i]))
        push!(adj[nb], (na, rval[i], rw[i]))
    end
    for n in 1:nnodes
        sort!(adj[n]; by = e -> e[3], rev = true)
    end
    visited = falses(nnodes)
    for p in pins
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
    station_closure_residuals(detections, bl_pairs, pol_products; observable = :phase) -> Vector

For every closed triangle of baselines present in `bl_pairs`, the residual
closure quantity of the chosen `observable` (`:delay`/`:rate`/`:phase`) using the
*measured* detections — i.e. the signed sum around the triangle that station-based
quantities must cancel. For noiseless station-differenced data these are ≈ 0
(including mixed-hand triangles); large values flag non-closing data. This is a
property of the data alone, so no solution is needed to evaluate it.
"""
function station_closure_residuals(
        detections::AbstractMatrix{<:Detection},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString};
        observable::Symbol = :phase,
        product::Integer = 1,
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
        (dab.valid && dbc.valid && dac.valid) || continue
        s = getval(dab) + getval(dbc) - getval(dac)
        observable === :phase && (s = rem2pi(s, RoundNearest))
        push!(res, s)
    end
    return res
end
