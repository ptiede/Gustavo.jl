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
    Stationization(; snr_min, cross_hand_rate, phase_rewrap_iters)

Options for [`stationize_scan`](@ref). `snr_min` drops detections below this SNR;
`cross_hand_rate` includes cross-hand products in the rate solve (default false —
appropriate when χ is per-scan); `phase_rewrap_iters` re-wraps phase residuals to
handle differences exceeding ±π.
"""
Base.@kwdef struct Stationization
    snr_min::Float64 = 6.0
    cross_hand_rate::Bool = false
    phase_rewrap_iters::Int = 3
end

"""
    StationSolution

Per-(station, feed) fringe solution for one scan. `delay`/`rate`/`phase` are
`(nant, 2)` matrices (feed axis 1/2); entries are `NaN` where a (station, feed)
had no usable detection. `chi` is the per-scan cross-hand source phase (`NaN` if
no cross hands were used). `covered` marks solved (station, feed) cells. `ncomp`
gives the connected-component count of the phase graph (≥2 ⇒ disconnected array
or feeds untied by cross hands).
"""
struct StationSolution
    delay::Matrix{Float64}
    rate::Matrix{Float64}
    phase::Matrix{Float64}
    chi::Float64
    covered::BitMatrix
    ncomp::Int
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
    stationize_scan(detections, bl_pairs, pol_products, nant; ref_ant, opts) -> StationSolution

Solve per-(station, feed) delay, rate and phase from a scan's per-baseline
`detections::AbstractMatrix{FringeDetection}` (indexed `[baseline, product]`).
`bl_pairs` are the `(a, b)` antenna-index pairs, `pol_products` the MSv4
correlation labels (e.g. `["PP","PQ","QP","QQ"]`), `ref_ant` the gauge reference.
"""
function stationize_scan(
        detections::AbstractMatrix{FringeDetection},
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

    # Gather valid observation rows per observable.
    delay_rows = _ObsRow[]
    rate_rows = _ObsRow[]
    phase_rows = _ObsRow[]
    for bi in 1:nbl, p in 1:npol
        det = detections[bi, p]
        (det.valid && det.snr >= opts.snr_min) || continue
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

    delay, _, cov_d, _ = _solve_observable(delay_rows, nant, ref_ant; use_chi = false, rewrap = 0)
    rate, _, cov_r, _ = _solve_observable(rate_rows, nant, ref_ant; use_chi = false, rewrap = 0)
    phase, chi, cov_p, ncomp = _solve_observable(phase_rows, nant, ref_ant; use_chi = true, rewrap = opts.phase_rewrap_iters)

    covered = cov_d .| cov_r .| cov_p
    return StationSolution(delay, rate, phase, chi, covered, ncomp)
end

# Solve one observable's WLS system on the (station, feed) graph. Returns
# (values::(nant,2), chi, covered::(nant,2), ncomp).
function _solve_observable(rows::Vector{_ObsRow}, nant::Integer, ref_ant::Integer; use_chi::Bool, rewrap::Integer)
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
    # over-constrain χ).
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

    x = weighted_constrained_least_squares(A, b, w, C, dgauge)
    # Phase re-wrap: unwrap observations toward the current model and re-solve, so
    # station-difference phases exceeding ±π are handled.
    if rewrap > 0
        bw = copy(b)
        for _ in 1:rewrap
            model = A * x
            @. bw = b + 2π * round((model - b) / (2π))
            x = weighted_constrained_least_squares(A, bw, w, C, dgauge)
        end
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
        detections::AbstractMatrix{FringeDetection},
        bl_pairs::AbstractVector{<:Tuple{Integer, Integer}},
        pol_products::AbstractVector{<:AbstractString},
        ::StationSolution;
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
