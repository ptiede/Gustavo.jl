# ── Corrections: modify a Measurement Set in place ──────────────────────────
#
# A correction modifies the layers of the Measurement Set it is handed and
# returns it, as an `f!` function does.

function _check_corrected(x, ms, got)
    got === ms || throw(
        ArgumentError(
            "a correction modifies the Measurement Set it is handed and returns it; " *
                "$(nameof(typeof(x))) returned " *
                (got isa XRadio.MeasurementSet ? "a different MeasurementSet" : "a $(nameof(typeof(got)))")
        )
    )
    return got
end

_one_member(ms::XRadio.MeasurementSet) =
    XRadio.ProcessingSet(OrderedDict{Symbol, XRadio.MeasurementSet}(:ms => ms))

# The interval every time sample integrates over, from the time coordinate's
# `integration_time`; `nothing` when the Measurement Set does not state it.
function _time_span(ms::XRadio.MeasurementSet)
    it = get(DimensionalData.metadata(dims(ms, Ti)), :integration_time, nothing)
    return isnothing(it) ? nothing : Float64(it.value)
end

# ── Dividing out a solution's gains ──────────────────────────────────────────

# Target station index → the solution's own (0 = absent from the solution).
function _antenna_map(sol::Calibration._AppliedSolution, antennas)
    solnames = sol.geom.antennas
    m = [something(findfirst(==(n), solnames), 0) for n in antennas]
    all(iszero, m) && throw(
        ArgumentError(
            "the solution shares no antenna with this data, so it would correct " *
                "nothing. It knows $(join(map(repr, solnames), ", ")); the data has " *
                "$(join(map(repr, antennas), ", "))."
        )
    )
    return m
end

# Whether the solution's gains are the same at every time in the window: no
# component reads a time coordinate, and every sample places in one time segment.
# A precal is usually such a solution (PerScan × PerSpectralWindow with no time
# term), and evaluating one time column instead of `nti` of them saves the
# `cis`/`exp` work that dominates applying the correction.
#
# Placement runs over the whole window here, so a sample the solution cannot
# place — or whose span straddles a bin boundary — raises exactly as it would in
# the full evaluation. This decides how many columns to evaluate, never whether
# to check.
function _time_constant_over(sol::Calibration._AppliedSolution, win::GeometryWindow, tspan)
    length(win.ti_idx) <= 1 && return true
    for grp in sol.groups, plan in grp.layout.plans
        Ti in Calibration.term_axes(plan.term) && return false
        ids = Calibration.time_segment_ids(
            plan.tseg, sol.geom, win.geom; ti_idx = win.ti_idx, time_span = tspan,
        )
        allequal(ids) || return false
    end
    return true
end

# `sol`'s gains divided out of `ms` in place, placed through `win`. A cell
# whose gain is degenerate is left as it is (`flag_bad = false`), or given a NaN
# visibility and a flag (`flag_bad = true`).
function _divide_gains!(
        ms::XRadio.MeasurementSet, win::GeometryWindow, sol::Calibration._AppliedSolution;
        flag_bad::Bool, executor = DynamicScheduler(),
    )
    amap = _antenna_map(sol, win.geom.antennas)
    vis = ms[:visibility]
    weight = ms[:weight]
    flag = ms[:flag]
    feeds = feed_pairs(ms)
    tspan = _time_span(ms)
    tconst = _time_constant_over(sol, win, tspan)
    gwin = tconst ? GeometryWindow(win.geom, win.chan_idx, win.ti_idx[1:1]) : win
    g = gains(sol, gwin; time_span = tspan)
    V, W, F = _storage_order(vis), _storage_order(weight), _storage_order(flag)
    axes(V, 2) == axes(g, 1) && (tconst || axes(V, 4) == axes(g, 2)) || throw(
        DimensionMismatch("gains $(axes(g)) do not cover the data $(axes(V))"),
    )
    # Tasks own whole time samples: the products of a cell are adjacent in
    # storage, so tasks split by product would share and write cache lines.
    tforeach(axes(V, 4); scheduler = executor) do t
        gt = tconst ? firstindex(g, 2) : t
        _divide_sample!(V, W, F, g, t, gt, win.antennas, amap, feeds, flag_bad)
    end
    return ms
end

# Gain magnitude below which a cell is treated as unconstrained rather than
# divided through: the correction would amplify noise without bound.
const _GAIN_FLOOR = 1.0e-12

# Time sample `t` of `_divide_gains`, its gains at time index `gt`.
function _divide_sample!(V, W, F, g, t, gt, antennas, amap, feeds, flag_bad::Bool)
    for bi in axes(V, 3)
        a, b = antennas[bi]
        sa, sb = amap[a], amap[b]
        (sa == 0 || sb == 0) && continue
        for c in axes(V, 2), p in axes(V, 1)
            fa, fb = feeds[p, bi]
            ga = g[c, gt, sa, fa]
            gb = g[c, gt, sb, fb]
            den = ga * conj(gb)
            if !isfinite(den) || abs2(ga) < _GAIN_FLOOR^2 || abs2(gb) < _GAIN_FLOOR^2
                if flag_bad
                    V[p, c, bi, t] = convert(eltype(V), NaN)
                    F[p, c, bi, t] = true
                end
                continue
            end
            # Both gains are above the floor, so the plain reciprocal cannot
            # overflow; Base's robust complex division costs several times more.
            den2 = abs2(den)
            V[p, c, bi, t] *= conj(den) / den2
            W[p, c, bi, t] *= den2
        end
    end
    return nothing
end

# ── Per-station weight scaling ──────────────────────────────────────────────

"""
    scale_weights!(ms::MeasurementSet, scale::AbstractDict) -> ms

Scale the weights of `ms` per station in place, `w → w·s_a·s_b` on each
baseline `(a, b)`. `scale` maps station names (strings or symbols) to finite,
positive factors; a station it does not name keeps its weights. A factor above
1 raises a station's weights (a correlator claiming more noise than the data
carries). Visibilities are untouched.

    scale_weights!(ms, Dict("HS" => 2.0, "GL" => 2.0))
"""
function scale_weights!(ms::XRadio.MeasurementSet, scale::AbstractDict)
    all(x -> isfinite(x) && x > 0, values(scale)) || throw(
        ArgumentError("scale_weights!: every factor must be finite and positive, got $(scale).")
    )
    factor = Dict(String(n) => Float64(f) for (n, f) in scale)
    weight = ms[:weight]
    for (wb, (a, b)) in zip(eachslice(weight; dims = BaselineID), XRadio.baselines(ms))
        f = get(factor, String(a), 1.0) * get(factor, String(b), 1.0)
        f == 1 || (wb .*= f)
    end
    return ms
end

"""
    scale_weights!(ms::MeasurementSet, scale::AbstractDimArray) -> ms

Scale the weights of `ms` in place by `scale`, matched to them by dimension,
such as the factors per (polarization, baseline) of [`weight_scales`](@ref).

    scale_weights!(ms, weight_scales(ms))
"""
function scale_weights!(ms::XRadio.MeasurementSet, scale::AbstractDimArray)
    weight = ms[:weight]
    @d weight .*= scale
    return ms
end

"""
    weight_scales(ms::MeasurementSet) -> DimArray

The factor per (polarization, baseline) that makes the weights of `ms` inverse
variances of each visibility component, estimated from the scatter between
consecutive time samples of one channel. For two such samples,
`z = (V₁ − V₂)/√(1/w₁ + 1/w₂)` has `E|z|² = 2` when the weights are inverse
variances and the signal does not change between them, so the factor is
`2 ln 2 / median|z|²` (the median of an exponential with mean 2 is `2 ln 2`).
Pairs with either sample flagged are skipped.

The signal cancels as long as its phase moves little in one integration; an
uncorrected rate `r` moves it by `2π r τ` over an integration of length `τ`.

Autocorrelations, and cross-correlations with no unflagged consecutive pair,
get 1; a warning names the cross-correlations among them that hold unflagged
data. Apply the factors with [`scale_weights!`](@ref).
"""
function weight_scales(ms::XRadio.MeasurementSet)
    V, W, F = ms[:visibility], ms[:weight], ms[:flag]
    T = float(eltype(W))
    scale = ones(T, dims(W, (Polarization, BaselineID)))
    baselines = XRadio.baselines(ms)
    unestimated = String[]
    for cell in DimensionalData.DimIndices(scale)
        p, bi = map(DimensionalData.val, cell)
        a, b = baselines[bi]
        a == b && continue
        z2 = _consecutive_z2(view(V, cell), view(W, cell), view(F, cell))
        if !isempty(z2)
            scale[cell] = 2 * log(T(2)) / median!(z2)
        elseif !all(view(F, cell))
            push!(unestimated, "$(lookup(W, Polarization)[p]) on $(a)–$(b)")
        end
    end
    isempty(unestimated) || @warn "weight_scales: no consecutive unflagged time samples to " *
        "estimate the weight scale of $(join(unestimated, ", ")); their factor is 1"
    return scale
end

# |z|² of each pair of consecutive time samples of one channel, both unflagged.
function _consecutive_z2(v, w, f)
    nt = size(v, Ti)
    t1, t2 = Ti(1:(nt - 1)), Ti(2:nt)
    ok = @d .!(view(f, t1) .| view(f, t2)) (dims = (Frequency, Ti), strict = false)
    z2 = @d abs2.(view(v, t2) .- view(v, t1)) ./ (inv.(view(w, t1)) .+ inv.(view(w, t2))) (dims = (Frequency, Ti), strict = false)
    return parent(z2)[parent(ok)]
end
