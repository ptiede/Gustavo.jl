# ── Per-feed stationization with closure ─────────────────────────────────────
#
# Turn per-baseline fringe detections into per-(station, feed) delays, rates and
# phases by weighted least squares on the (station, feed) graph (2·nant nodes).
# Every observable differences station quantities with the SAME incidence:
#
#     baseline (a,b), product p with feeds (fa, fb) = correlation_feed_pair(p):
#         delay_ab^p = τ_{a,fa} − τ_{b,fb}
#         rate_ab^p  = ṙ_{a,fa} − ṙ_{b,fb}
#         phase_ab^p = φ_{a,fa} − φ_{b,fb}  (+ source cross-hand phase, below)
#
# Using ALL FOUR products (not just parallel hands) is deliberate: cross-hand
# rows connect feed-1 and feed-2 nodes, so the relative feed-2−feed-1 offset (the
# "R-L delay/phase") is pinned by the data and falls out of the global solution
# — no separate R-L stage. Closure holds by construction within each product and
# across mixed-hand triangles.
#
# Cross-hand source phase: a point source contributes a constant phase χ to the
# cross-hand visibility — +χ on PQ rows, −χ on QP rows — flat in frequency, so it
# enters only the PHASE system (one nuisance unknown per scan), never delay. Per
# the no-field-rotation decision, rate is taken from parallel hands only by
# default (cross-hand rate would absorb intra-scan field-rotation drift unless χ
# is segmented finer than the scan).
#
# Gauge: each connected component of the (station, feed) graph has one additive
# freedom per observable; we pin the reference station's node per component. With
# cross hands the two feeds merge into one component, and the PHASE system gains
# an extra (feed-offset ↔ χ) degeneracy — shifting all feed-2 phases by Δ and χ by
# Δ is invisible — so we add a second pin on the reference feed-2 node (the EVPA
# gauge; absolute EVPA still needs external polarization calibration).

"""
    Stationization(; snr_min, cross_hand_rate, phase_rewrap_iters, reject_sigma, reject_iters)

Options for [`stationize_scan`](@ref). `snr_min` drops detections below this SNR;
`cross_hand_rate` includes cross-hand products in the rate solve (default false —
appropriate when χ is per-scan); `phase_rewrap_iters` re-wraps phase residuals to
handle differences exceeding ±π.

`reject_sigma`/`reject_iters` control robust outlier rejection: after each solve,
detections whose SNR-weighted residual is a > `reject_sigma` MAD outlier are
dropped and the system re-solved (up to `reject_iters` times). A false fringe —
e.g. tone/crosstalk correlation on a co-located telescope pair — is closure-
inconsistent with the true detections, so it lands far outside the residual
distribution and is excised instead of dragging its stations' solutions.
`reject_sigma = 0` disables rejection.
"""
Base.@kwdef struct Stationization
    snr_min::Float64 = 6.0
    cross_hand_rate::Bool = false
    phase_rewrap_iters::Int = 3
    reject_sigma::Float64 = 7.0
    reject_iters::Int = 5
end

# Node index on the (station, feed) graph: feed-1 block 1:nant, feed-2 nant+1:2nant.
_node(ant::Integer, feed::Integer, nant::Integer) = (feed - 1) * nant + ant

# Cross-hand source-phase sign for product with feeds (fa, fb): +1 on (1,2)
# (PQ-type), −1 on (2,1) (QP-type), 0 on parallel hands.
_chi_sign(fa::Integer, fb::Integer) = fa == fb ? 0 : (fa < fb ? 1 : -1)

# One observation row contributing to a node system.
struct _ObsRow
    a::Int
    b::Int
    fa::Int
    fb::Int
    val::Float64
    w::Float64
    chisign::Int
end

"""
    stationize_scan(detections, bl_pairs, pol_products, nant; ref_ant, opts) -> DimStack

Solve per-(station, feed) delay, rate and phase from a scan's per-baseline
`detections::AbstractMatrix{Detection}` (indexed `[baseline, product]`).
`bl_pairs` are the `(a, b)` antenna-index pairs, `pol_products` the MSv4
correlation labels (e.g. `["PP","PQ","QP","QQ"]`), `ref_ant` the gauge reference.

Returns a `DimStack` over `Ant × Feed` (feed axis 1/2): layers `:delay`/`:rate`/
`:phase` are `NaN` where a (station, feed) had no usable detection, `:covered`
marks the solved cells. Its metadata carries `:chi` (the per-scan cross-hand
source phase, `NaN` if no cross hands were used) and `:ncomp` (the connected-
component count of the phase graph; ≥2 ⇒ disconnected array or feeds untied by
cross hands).
"""
function stationize_scan(
        detections::AbstractMatrix{Detection},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString},
        nant::Integer;
        ref_ant::Integer = 1,
        opts::Stationization = Stationization(),
    )
    nbl, npol = size(detections)
    nbl == length(bl_pairs) || error("detections has $nbl baselines; bl_pairs has $(length(bl_pairs))")
    npol == length(pol_products) || error("detections has $npol products; pol_products has $(length(pol_products))")
    feeds = [correlation_feed_pair(p) for p in pol_products]

    # Closure pre-screen (see `_closure_screen`): drop every product of a
    # baseline whose delay/rate breaks triangle closure — a false fringe.
    excl = opts.reject_sigma > 0 ?
        _closure_screen((StationScanDetections(detections, collect(Tuple{Int, Int}, bl_pairs), feeds, 1),), opts) :
        Set{Tuple{Int, Int}}()

    # Gather valid observation rows per observable.
    delay_rows = _ObsRow[]
    rate_rows = _ObsRow[]
    phase_rows = _ObsRow[]
    for bi in 1:nbl, p in 1:npol
        det = detections[bi, p]
        (det.valid && det.snr >= opts.snr_min) || continue
        (1, bi) in excl && continue             # closure-inconsistent baseline
        a, b = bl_pairs[bi]
        a == b && continue                      # skip autocorrelations
        fa, fb = feeds[p]
        cs = _chi_sign(fa, fb)
        w = det.snr^2                           # CRB row weight ∝ SNR² (common Δf/Δt cancel per system)
        push!(delay_rows, _ObsRow(a, b, fa, fb, det.delay, w, cs))
        if cs == 0 || opts.cross_hand_rate
            push!(rate_rows, _ObsRow(a, b, fa, fb, det.rate, w, cs))
        end
        push!(phase_rows, _ObsRow(a, b, fa, fb, det.phase, w, cs))
    end

    delay, _, cov_d, _ = _solve_observable_robust(delay_rows, nant, ref_ant, opts; use_chi = false, rewrap = 0)
    rate, _, cov_r, _ = _solve_observable_robust(rate_rows, nant, ref_ant, opts; use_chi = false, rewrap = 0)
    phase, chi, cov_p, ncomp = _solve_observable_robust(phase_rows, nant, ref_ant, opts; use_chi = true, rewrap = opts.phase_rewrap_iters)

    covered = cov_d .| cov_r .| cov_p
    gdims = (Ant(1:nant), Feed(1:2))
    return DimensionalData.DimStack(
        (
            delay = DimArray(delay, gdims),
            rate = DimArray(rate, gdims),
            phase = DimArray(phase, gdims),
            covered = DimArray(covered, gdims),
        );
        metadata = Dict{Symbol, Any}(:chi => chi, :ncomp => ncomp),
    )
end

# Closure pre-screen: flag baselines whose DELAY or RATE detections break
# triangle closure. A false fringe (e.g. tone/crosstalk correlation on a
# co-located pair like the Onsala twins) is a high-SNR detection whose value is
# inconsistent with every triangle it participates in, while genuine detections
# close to within thermal noise. Unlike post-fit residual rejection, closure is
# FIT-FREE, so a bad baseline cannot mask itself by dragging the solution: one
# bad baseline corrupts at most 1 of (nsta − 2) triangles of any clean baseline,
# so the per-baseline MEDIAN closure misfit isolates exactly the culprit.
#
# Scores are noise-normalized (`|closure| / √(1/w₁+1/w₂+1/w₃)`, w = snr², CRB
# σ ∝ 1/snr) so one MAD cut applies across strong and weak triangles. Parallel
# hands only (a triangle needs one feed throughout); a flagged baseline drops
# ALL its products — cross-hand crosstalk accompanies parallel-hand crosstalk.
# Returns a set of `(scan_index, baseline_index)` to exclude.
function _closure_screen(scans, opts::Stationization)
    excl = Set{Tuple{Int, Int}}()
    for (sidx, sc) in enumerate(scans)
        nbl, npol = size(sc.det)
        for getval in ((d -> d.delay), (d -> d.rate)), feed in 1:2
            # Weighted per-baseline value on this feed's parallel-hand subgraph.
            acc = Dict{Tuple{Int, Int}, Tuple{Float64, Float64}}()   # (a,b) → (Σw·v, Σw)
            for bi in 1:nbl, p in 1:npol
                sc.feeds[p] == (feed, feed) || continue
                det = sc.det[bi, p]
                (det.valid && det.snr >= opts.snr_min) || continue
                a, b = sc.bl_pairs[bi]
                a == b && continue
                s0 = get(acc, (a, b), (0.0, 0.0))
                w = det.snr^2
                acc[(a, b)] = (s0[1] + w * getval(det), s0[2] + w)
            end
            length(acc) >= 6 || continue                             # need triangle redundancy
            val(a, b) = haskey(acc, (a, b)) ? acc[(a, b)][1] / acc[(a, b)][2] :
                haskey(acc, (b, a)) ? -acc[(b, a)][1] / acc[(b, a)][2] : NaN
            wgt(a, b) = haskey(acc, (a, b)) ? acc[(a, b)][2] :
                haskey(acc, (b, a)) ? acc[(b, a)][2] : 0.0
            stations = sort(unique(collect(Iterators.flatten(keys(acc)))))
            scores = Dict{Tuple{Int, Int}, Float64}()
            for (a, b) in keys(acc)
                zs = Float64[]
                for c in stations
                    (c == a || c == b) && continue
                    vac = val(a, c)
                    vbc = val(b, c)
                    (isnan(vac) || isnan(vbc)) && continue
                    cl = val(a, b) + vbc - vac
                    push!(zs, abs(cl) / sqrt(1 / wgt(a, b) + 1 / wgt(b, c) + 1 / wgt(a, c)))
                end
                length(zs) >= 2 || continue
                scores[(a, b)] = median(zs)
            end
            length(scores) >= 5 || continue
            svals = collect(values(scores))
            med = median(svals)
            s = 1.4826 * median(abs.(svals .- med))
            s > 0 || continue
            for ((a, b), sc_ab) in scores
                sc_ab - med > opts.reject_sigma * s || continue
                for bi in 1:nbl
                    (sc.bl_pairs[bi] == (a, b) || sc.bl_pairs[bi] == (b, a)) &&
                        push!(excl, (sidx, bi))
                end
            end
        end
    end
    return excl
end

# Robust wrapper around `_solve_observable`: iteratively re-solve, dropping rows
# whose SNR-weighted residual `(val − pred)·√w` is a > `reject_sigma` MAD outlier.
# With CRB weights (`w = snr²`, σ_val ∝ 1/snr) the weighted residuals share one
# scale across strong and weak rows, so a single cut is meaningful. Residuals of
# the (wrapping) phase system are re-wrapped to ±π before the cut.
function _solve_observable_robust(
        rows::Vector{_ObsRow}, nant::Integer, ref_ant::Integer, opts::Stationization;
        use_chi::Bool, rewrap::Integer,
    )
    vals, chi, cov, ncomp = _solve_observable(rows, nant, ref_ant; use_chi = use_chi, rewrap = rewrap)
    (opts.reject_sigma > 0 && !isempty(rows)) || return vals, chi, cov, ncomp
    wrap = rewrap > 0
    for _ in 1:opts.reject_iters
        z = map(rows) do r
            pa = vals[r.a, r.fa]
            pb = vals[r.b, r.fb]
            (isfinite(pa) && isfinite(pb)) || return 0.0
            pred = pa - pb + (use_chi && r.chisign != 0 && isfinite(chi) ? r.chisign * chi : 0.0)
            res = r.val - pred
            wrap && (res = rem2pi(res, RoundNearest))
            return res * sqrt(r.w)
        end
        med = median(z)
        s = 1.4826 * median(abs.(z .- med))
        s > 0 || break
        keep = abs.(z .- med) .<= opts.reject_sigma * s
        all(keep) && break
        rows = rows[keep]
        isempty(rows) && break
        vals, chi, cov, ncomp = _solve_observable(rows, nant, ref_ant; use_chi = use_chi, rewrap = rewrap)
    end
    return vals, chi, cov, ncomp
end

# Solve one observable's WLS system on the (station, feed) graph. Returns
# (values::(nant,2), chi, covered::(nant,2), ncomp).
function _solve_observable(
        rows::Vector{_ObsRow}, nant::Integer, ref_ant::Integer;
        use_chi::Bool, rewrap::Integer,
        seed_phase::Union{Nothing, AbstractMatrix{<:Real}} = nothing, seed_chi::Real = NaN,
    )
    vals = fill(NaN, nant, 2)
    cov = falses(nant, 2)
    nnodes = 2 * nant
    isempty(rows) && return vals, NaN, cov, 0

    edges = [(_node(r.a, r.fa, nant), _node(r.b, r.fb, nant)) for r in rows]
    compid, ncomp, touched = connected_components(nnodes, edges)

    has_chi = use_chi && any(r.chisign != 0 for r in rows)
    nchi = has_chi ? 1 : 0
    ncol = nnodes + nchi
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
    # EVPA gauge: the (feed-2 offset ↔ χ) degeneracy is a SINGLE global freedom
    # (χ is one per-scan unknown), so add exactly ONE extra feed-2 pin — in the
    # reference antenna's component if it spans both feeds, else the first such
    # component. Other disconnected islands' feed offsets are then tied through
    # the global χ, so they need no extra pin (an extra pin per island would
    # over-constrain χ). NOTE: this pin is a RANK device, not a physical zero —
    # under it the solved χ absorbs the reference's own R–L phase for this solve.
    # A caller that repeats this solve along an axis (e.g. `_solve_phase_bandpass!`
    # per channel) must reassign χ's along-axis structure back into the feed-2
    # block, or the reference's R–L variation is silently discarded.
    if has_chi
        ref_comp = compid[_node(ref_ant, 1, nant)]
        order = ref_comp == 0 ? (1:ncomp) : Iterators.flatten((ref_comp, (c for c in 1:ncomp if c != ref_comp)))
        for c in order
            comp_nodes = findall(==(c), compid)
            any(n -> n <= nant, comp_nodes) && any(n -> n > nant, comp_nodes) || continue
            r2 = _node(ref_ant, 2, nant)
            f2 = r2 in comp_nodes ? r2 : minimum(filter(n -> n > nant, comp_nodes))
            f2 in pins || push!(pins, f2)
            break
        end
    end
    # Pin every UNTOUCHED node — an (antenna, feed) with no observation in this
    # solve, e.g. a station that dropped out. Its design column is all-zero, which
    # would make the constrained QR system rank-deficient and corrupt the solve
    # for the stations that DO have data. Pinning it to 0 (its value is discarded;
    # only `touched` cells are returned) keeps the system full-rank and well-posed.
    for n in 1:nnodes
        touched[n] || n in pins || push!(pins, n)
    end

    A = zeros(Float64, nrow, ncol)
    b = zeros(Float64, nrow)
    w = zeros(Float64, nrow)
    for (i, r) in enumerate(rows)
        A[i, _node(r.a, r.fa, nant)] += 1.0
        A[i, _node(r.b, r.fb, nant)] -= 1.0
        has_chi && r.chisign != 0 && (A[i, nnodes + 1] = float(r.chisign))
        b[i] = r.val
        w[i] = r.w
    end
    C = zeros(Float64, length(pins), ncol)
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
        xseed = _spanning_tree_seed(rows, nant, pins, ncol)
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
            has_chi && isfinite(seed_chi) && (xseed[nnodes + 1] = float(seed_chi))
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
    else
        x = weighted_constrained_least_squares(A, b, w, C, dgauge)
    end

    for ant in 1:nant, feed in 1:2
        n = _node(ant, feed, nant)
        if touched[n]
            vals[ant, feed] = x[n]
            cov[ant, feed] = true
        end
    end
    chi = has_chi ? x[nnodes + 1] : NaN
    return vals, chi, cov, ncomp
end

# Maximum-weight spanning-tree phase seed (K1). Propagate wrapped edge phases
# from each pin over the parallel-hand (chisign == 0, same-feed) edges of the
# (station, feed) graph, preferring high-weight edges, to build a globally
# consistent node-phase estimate. Cross-hand rows (which carry the unknown χ) are
# excluded from the tree; their nodes are reached through the parallel-hand
# subgraph (or seeded 0 and resolved by the WLS + χ). Returns a length-`ncol`
# vector (χ column, if present, seeded 0). The estimate is used only to unwrap the
# observations for the first constrained solve, so any edge it cannot place stays
# 0 — the re-wrap iterations refine from there.
function _spanning_tree_seed(rows::Vector{_ObsRow}, nant::Integer, pins::AbstractVector{<:Integer}, ncol::Integer)
    nnodes = 2 * nant
    x = zeros(Float64, ncol)
    # Adjacency over parallel-hand edges: neighbor, phase to ADD (φ_v = φ_u + add), weight.
    adj = [Vector{Tuple{Int, Float64, Float64}}() for _ in 1:nnodes]
    for r in rows
        r.chisign == 0 || continue
        na = _node(r.a, r.fa, nant)
        nb = _node(r.b, r.fb, nant)
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
# `solve_station_systems!` is the segmentation/tying-aware generalization of
# `stationize_scan`. Instead of a fixed per-scan (station, feed) node space, the
# unknowns are the θ COLUMNS the model declares: for each stage-B phase component
# (a `ConstantTerm`/`Delay`/`Rate` × time-seg × tying), `_block_index(plan,
# _feed_node(tying, feed), 1, tseg_id[ti], ant)` is the θ slot a (station, feed,
# time) observation maps to. The plan's segmentation encodes the time basis
# (PerScan → a distinct column per scan; GlobalTime → one column shared across the
# whole track) and its tying the feed fold (PerFeed → distinct feed columns;
# SharedFeeds → one shared column). So the SAME engine solves a per-scan model
# (columns disjoint per scan ⇒ block-diagonal ⇒ identical to N independent
# `stationize_scan` calls) and a model with a global R–L offset (a column shared
# across scans couples them) — the model is the extension point, this solver just
# reads the θ columns each component declares.
#
# `scans` is a vector of `StationScanDetections`, each carrying one scan's
# detection matrix, its `(a, b)` pairs, per-product feeds, and a representative
# global time index `ti` for the `tseg_id` lookup. θ slots are ACCUMULATED into
# (`+=`), matching `_pack_station!`, so `rounds > 1` (search on the residual)
# stays correct.

struct StationScanDetections{D}
    det::D                                   # [baseline, product] of Detection cells:
                                             # a Baseline × Pol DimStack (search) or a
                                             # Matrix{Detection} (refine/direct solve)
    bl_pairs::Vector{Tuple{Int, Int}}
    feeds::Vector{Tuple{Int, Int}}           # feed pair per product
    ti::Int                                  # representative global time index (→ tseg)
end

"""
    solve_station_systems!(θ, scans, components; ref_ant, opts) -> (chi, ncomp, nrejected)

Solve the stage-B fringe systems (delay, rate, constant phase) over `scans` and
accumulate the per-(station, feed) values into `θ` at the columns the model
declares. `components` is a vector of `(plan::ComponentPlan, kind::Symbol)` with
`kind ∈ (:delay, :rate, :phase)`. Multiple components of the SAME kind are summed
per (station, feed) observation: e.g. a feed-common `PerScan × SharedFeeds` term
plus a global `GlobalTime × FeedComponent(2)` R–L offset both feed the delay
system, so a feed-2 row touches both columns and a stable R–L offset is solved
once across the track (bright scans pin it; weak scans inherit it, tying feeds
that would otherwise split). Returns the representative cross-hand `chi`, the
phase-system component count, and the number of detections excised by the robust
rejection (summed over the three systems; see `Stationization`). With a single
per-scan/per-feed component per kind and one scan, this is numerically identical
to `stationize_scan`.
"""
function solve_station_systems!(
        θ::AbstractVector, scans, components;
        ref_ant::Integer = 1, opts::Stationization = Stationization(),
    )
    chi = NaN
    ncomp = 0
    # Closure pre-screen: baselines carrying a closure-breaking false fringe are
    # excluded from every system up front (fit-free, so immune to the leverage
    # masking that can defeat the post-fit residual cut on small arrays).
    excl = opts.reject_sigma > 0 ? _closure_screen(scans, opts) : Set{Tuple{Int, Int}}()
    nrej = 0
    for (sidx, bi) in excl
        for p in axes(scans[sidx].det, 2)
            scans[sidx].det[bi, p].valid && (nrej += 1)
        end
    end
    # (station, scan-index) pairs CONSTRAINED by the surviving rows — the
    # EHT-HOPS flag criterion, inverted: a station with no strong detection
    # left on ANY of its baselines after the closure screen and the robust
    # rejection is uncalibrated for that scan (its θ stays 0 ⇒ identity gain)
    # and must be FLAGGED downstream, not silently passed through. Intersected
    # over the solved kinds: a station must be constrained in delay AND rate
    # AND phase to count as calibrated.
    covered = Set{Tuple{Int, Int}}()
    first_kind = true
    for kind in (:delay, :rate, :phase)
        plans = [c[1] for c in components if c[2] === kind]
        isempty(plans) && continue
        ch, nc, nr, cov = _solve_kind_cols!(θ, scans, plans, ref_ant, opts, kind, excl)
        nrej += nr
        covered = first_kind ? cov : intersect(covered, cov)
        first_kind = false
        if kind === :phase
            chi = ch
            ncomp = nc
        end
    end
    return chi, ncomp, nrej, covered
end

# Solve one observable kind across all scans, accumulating into θ. Each detection
# becomes a station-difference row whose a-/b-side touch the sum of all `plans`'
# θ columns for that (station, feed, time) — a feed-common per-scan column and,
# when present, a global feed-offset column. Returns (chi, ncomp).
function _solve_kind_cols!(
        θ::AbstractVector, scans, plans, ref_ant::Integer, opts::Stationization, kind::Symbol,
        excl::Set{Tuple{Int, Int}} = Set{Tuple{Int, Int}}(),
    )
    getval = kind === :delay ? (d -> d.delay) : kind === :rate ? (d -> d.rate) : (d -> d.phase)
    use_chi = kind === :phase
    rewrap = kind === :phase ? opts.phase_rewrap_iters : 0
    # Cross-hand rows: delay & phase always include them (they tie the feeds);
    # rate only when requested (cross-hand rate would absorb field rotation).
    include_cross = kind === :rate ? opts.cross_hand_rate : true

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
    rval = Float64[]; rw = Float64[]; rcs = Int[]; rscan = Int[]
    rsta_a = Int[]; rsta_b = Int[]
    for (sidx, sc) in enumerate(scans)
        nbl, npol = size(sc.det)
        for bi in 1:nbl, p in 1:npol
            det = sc.det[bi, p]
            (det.valid && det.snr >= opts.snr_min) || continue
            (sidx, bi) in excl && continue      # closure-inconsistent baseline
            a, b = sc.bl_pairs[bi]
            a == b && continue
            fa, fb = sc.feeds[p]
            cs = _chi_sign(fa, fb)
            (include_cross || cs == 0) || continue
            nsA = Int[]; nsB = Int[]
            for plan in plans
                na = _feed_node(plan.tying, fa)
                ca = na == 0 ? 0 : _block_index(plan, na, 1, plan.tseg_id[sc.ti], a)
                ca != 0 && push!(nsA, getnode(ca, a, fa, sidx))
                nb = _feed_node(plan.tying, fb)
                cb = nb == 0 ? 0 : _block_index(plan, nb, 1, plan.tseg_id[sc.ti], b)
                cb != 0 && push!(nsB, getnode(cb, b, fb, sidx))
            end
            (isempty(nsA) || isempty(nsB)) && continue
            push!(rowA, nsA); push!(rowB, nsB)
            push!(rval, getval(det)); push!(rw, det.snr^2); push!(rcs, cs); push!(rscan, sidx)
            push!(rsta_a, a); push!(rsta_b, b)
        end
    end
    isempty(rowA) && return (NaN, 0, 0, Set{Tuple{Int, Int}}())

    # Robust solve: re-solve dropping rows whose SNR-weighted residual is a
    # > `reject_sigma` MAD outlier (closure-breaking false fringes; see
    # `Stationization`). Weighted residuals `resid·√w` share one scale across
    # strong and weak rows (σ_val ∝ 1/snr, w = snr²). Phase residuals come back
    # from the re-wrapped system, so they are already branch-corrected.
    keep = trues(length(rowA))
    nrej = 0
    local x, chi, ncomp
    it = 0
    while true
        idx = findall(keep)
        isempty(idx) && return (NaN, 0, nrej, Set{Tuple{Int, Int}}())
        x, chi, ncomp, resid = _solve_tagged_system(
            rowA[idx], rowB[idx], rval[idx], rw[idx], rcs[idx], rscan[idx], length(node_col),
            node_feed, node_station, node_scan, ref_ant; use_chi = use_chi, rewrap = rewrap,
        )
        it += 1
        (opts.reject_sigma > 0 && it <= opts.reject_iters) || break
        z = resid .* sqrt.(rw[idx])
        med = median(z)
        s = 1.4826 * median(abs.(z .- med))
        s > 0 || break
        bad = findall(abs.(z .- med) .> opts.reject_sigma * s)
        isempty(bad) && break
        keep[idx[bad]] .= false
        nrej += length(bad)
    end
    @inbounds for n in eachindex(node_col)
        θ[node_col[n]] += x[n]
    end
    # (station, scan) pairs constrained by the SURVIVING rows of this system.
    cov = Set{Tuple{Int, Int}}()
    for i in findall(keep)
        push!(cov, (rsta_a[i], rscan[i]))
        push!(cov, (rsta_b[i], rscan[i]))
    end
    return chi, ncomp, nrej, cov
end

# Constrained WLS over a tagged node graph (the column-space generalization of
# `_solve_observable`). `node_feed`/`node_station`/`node_scan` tag each local node
# (feed 0 = shared by both feeds; scan 0 = global column) so the gauge reproduces
# `_solve_observable`'s tie-breaks in the per-scan case. Rows may touch more than
# one column per side (a feed-common column plus a global feed-offset column). χ
# is a per-scan nuisance. After the explicit reference/EVPA pins, any residual
# gauge freedom (e.g. the per-scan absolute level once scans are globally coupled)
# is removed by a minimum-norm null-space pin — so the engine is well-posed for
# any model `plan_parameters` can flatten, with no model-specific gauge code.
function _solve_tagged_system(
        rowA, rowB, rval, rw, rcs, rscan, nnodes,
        node_feed, node_station, node_scan, ref_ant; use_chi::Bool, rewrap::Integer,
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

    # χ columns: one per scan that carries a cross-hand row.
    chi_col = Dict{Int, Int}()
    if use_chi
        for i in eachindex(rcs)
            rcs[i] != 0 || continue
            get!(chi_col, rscan[i], length(chi_col) + 1)
        end
    end
    nchi = length(chi_col)
    ncol = nnodes + nchi

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
    # EVPA gauge: the (feed-2 offset ↔ χ) freedom. Add one feed-2 pin per scan that
    # has a both-feed component (deduped — a global R–L offset column is one node
    # shared by all scans, so this resolves to a single pin on the global offset).
    # As with the per-scan pin above, this fixes only the ONE conventional EVPA
    # constant; the per-scan χ columns absorb the source cross-hand phase (which
    # must NOT be calibrated out) plus any reference R–L drift, and are discarded
    # below (only the scan-min χ is returned, as a diagnostic).
    if nchi > 0
        for s in sort(collect(keys(chi_col)))
            scomps = unique(compid[n] for n in 1:nnodes if node_scan[n] == s)
            ref_first = sort(scomps; by = c -> any(n -> compid[n] == c && is_ref(n) && is_feed1(n), 1:nnodes) ? 0 : 1)
            for c in ref_first
                comp = [n for n in 1:nnodes if compid[n] == c]
                (any(is_feed1, comp) && any(n -> node_feed[n] == 2, comp)) || continue
                f2 = [n for n in comp if node_feed[n] == 2]
                r2 = findfirst(is_ref, f2)
                pin = r2 !== nothing ? f2[r2] : f2[argmin(map(nodekey, f2))]
                pin in pins || push!(pins, pin)
                break
            end
        end
    end

    nrow = length(rowA)
    A = zeros(Float64, nrow, ncol)
    b = zeros(Float64, nrow)
    w = zeros(Float64, nrow)
    @inbounds for i in 1:nrow
        for n in rowA[i]
            A[i, n] += 1.0
        end
        for n in rowB[i]
            A[i, n] -= 1.0
        end
        if nchi > 0 && rcs[i] != 0
            A[i, nnodes + chi_col[rscan[i]]] = float(rcs[i])
        end
        b[i] = rval[i]
        w[i] = rw[i]
    end
    Cp = zeros(Float64, length(pins), ncol)
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
        xseed = _seed_tagged(rowA, rowB, rval, rw, rcs, pins, ncol, nnodes)
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

    chi = NaN
    if nchi > 0
        s0 = minimum(keys(chi_col))
        chi = x[nnodes + chi_col[s0]]
    end
    return x[1:nnodes], chi, ncomp, resid
end

# Max-weight spanning-tree phase seed in local-node space (column-space twin of
# `_spanning_tree_seed`): propagate wrapped parallel-hand edge phases from each
# pin to unwrap the first constrained solve. Only single-column-per-side
# parallel-hand rows are tree edges; multi-term (global-offset) rows are left to
# the constrained WLS + re-wrap iterations.
function _seed_tagged(rowA, rowB, rval, rw, rcs, pins, ncol::Integer, nnodes::Integer)
    x = zeros(Float64, ncol)
    adj = [Vector{Tuple{Int, Float64, Float64}}() for _ in 1:nnodes]
    for i in eachindex(rowA)
        (rcs[i] == 0 && length(rowA[i]) == 1 && length(rowB[i]) == 1) || continue
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
    station_closure_residuals(detections, bl_pairs, pol_products, sol; observable = :phase) -> Vector

For every closed triangle of baselines present in `bl_pairs`, the residual
closure quantity of the chosen `observable` (`:delay`/`:rate`/`:phase`) using the
*measured* detections — i.e. the signed sum around the triangle that station-based
quantities must cancel. For noiseless station-differenced data these are ≈ 0
(including mixed-hand triangles); large values flag non-closing data. `sol` is
unused for the measured-closure check but accepted for API symmetry.
"""
function station_closure_residuals(
        detections::AbstractMatrix{Detection},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString},
        ::AbstractDimStack;
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
