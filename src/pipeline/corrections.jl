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

# The interval each time sample integrates over, from the time coordinate's
# `integration_time`; `nothing` when the Measurement Set does not state it.
function _time_span(ms::XRadio.MeasurementSet)
    md = DimensionalData.metadata(lookup(ms[:visibility], Ti))
    md isa AbstractDict || return nothing
    it = get(md, :integration_time, nothing)
    it === nothing && return nothing
    return fill(Float64(it.value), length(XRadio.times(ms)))
end

# ── Dividing out a solution's gains ──────────────────────────────────────────

# Target station index → the solution's own (0 = absent from the solution).
function _station_map(sol::Calibration._AppliedSolution, stations)
    solnames = sol.geom.stations
    m = [something(findfirst(==(n), solnames), 0) for n in stations]
    all(iszero, m) && throw(
        ArgumentError(
            "the solution shares no station with this data, so it would correct " *
                "nothing. It knows $(join(map(repr, solnames), ", ")); the data has " *
                "$(join(map(repr, stations), ", "))."
        )
    )
    if any(iszero, m)
        missing_names = [n for (n, k) in zip(stations, m) if k == 0]
        @warn "stations $(missing_names) are not in the solution — they keep identity gains." maxlog = 1
    end
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

# The span of just the first sample, to match a single evaluated time column.
_head_span(span) = span === nothing || isempty(span) ? span : span[1:1]

# `sol`'s gains divided out of `ms` in place, placed through `win`. A cell
# whose gain is degenerate is left as it is (`flag_bad = false`), or given a NaN
# visibility and a flag (`flag_bad = true`).
function _divide_gains!(
        ms::XRadio.MeasurementSet, win::GeometryWindow, sol::Calibration._AppliedSolution;
        flag_bad::Bool, executor = DynamicScheduler(),
    )
    amap = _station_map(sol, win.geom.stations)
    vis = ms[:visibility]
    weight = ms[:weight]
    flag = ms[:flag]
    feeds = feed_pairs(ms)
    tspan = _time_span(ms)
    tconst = _time_constant_over(sol, win, tspan)
    gwin = tconst ? GeometryWindow(win.geom, win.chan_idx, win.ti_idx[1:1]) : win
    g = parent(gains(sol, gwin; time_span = tconst ? _head_span(tspan) : tspan))
    V, W, F = UVData._storage_order(vis), UVData._storage_order(weight), UVData._storage_order(flag)
    axes(V, 2) == axes(g, 1) && (tconst || axes(V, 4) == axes(g, 2)) || throw(
        DimensionMismatch("gains $(axes(g)) do not cover the data $(axes(V))"),
    )
    # Tasks own whole time samples: the products of a cell are adjacent in
    # storage, so tasks split by product would share and write cache lines.
    tforeach(axes(V, 4); scheduler = executor) do t
        gt = tconst ? firstindex(g, 2) : t
        _divide_sample!(V, W, F, g, t, gt, win.stations, amap, feeds, flag_bad)
    end
    return ms
end

# Gain magnitude below which a cell is treated as unconstrained rather than
# divided through: the correction would amplify noise without bound.
const _GAIN_FLOOR = 1.0e-12

# Time sample `t` of `_divide_gains`, its gains at time index `gt`.
function _divide_sample!(V, W, F, g, t, gt, stations, amap, feeds, flag_bad::Bool)
    for bi in axes(V, 3)
        a, b = stations[bi]
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
    scale_weights!(ms::MeasurementSet, scale::AbstractDimVector) -> ms

Scale the weights of `ms` per station in place, `w → w·s_a·s_b` on each
baseline `(a, b)`. `scale` holds one finite, positive factor per station,
indexed by `AntennaName`; a station it does not name keeps its weights. A
factor above 1 raises a station's weights (a correlator claiming more noise
than the data carries). Visibilities are untouched.

    ws = DimArray([2.0, 2.0], AntennaName(["HS", "GL"]))
    scale_weights!(ms, ws)
"""
function scale_weights!(ms::XRadio.MeasurementSet, scale::AbstractDimVector)
    only(dims(scale)) isa AntennaName || throw(
        ArgumentError(
            "scale_weights!: index the factors by `AntennaName`, not " *
                "$(nameof(typeof(only(dims(scale)))))"
        )
    )
    allunique(lookup(scale, 1)) || throw(
        ArgumentError("scale_weights!: a station is named more than once: $(collect(lookup(scale, 1)))")
    )
    all(x -> isfinite(x) && x > 0, scale) || throw(
        ArgumentError("scale_weights!: every factor must be finite and positive, got $(collect(scale)).")
    )
    factor = Dict(String(n) => Float64(f) for (n, f) in zip(lookup(scale, 1), scale))
    weight = ms[:weight]
    for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
        f = get(factor, String(a), 1.0) * get(factor, String(b), 1.0)
        f == 1 && continue
        view(weight, BaselineID(bi)) .*= f
    end
    return ms
end

# ── Channel flagging ────────────────────────────────────────────────────────

"""
    flag_channels!(ms::MeasurementSet, mask::AbstractDimVector{Bool}) -> ms

Flag, in place, the channels of `ms` where `mask`, indexed by `Frequency`
(Hz), is `true`. Each channel takes the mask entry at its center frequency; a
channel the mask does not cover throws. Visibilities and weights are left as
they are.

    freqs = XRadio.frequencies(ms)
    flag_channels!(ms, DimArray(86.10e9 .< freqs .< 86.12e9, Frequency(freqs)))
"""
function flag_channels!(ms::XRadio.MeasurementSet, mask::AbstractDimVector{Bool})
    only(dims(mask)) isa Frequency || throw(
        ArgumentError(
            "flag_channels!: index the mask by `Frequency`, not $(nameof(typeof(only(dims(mask)))))"
        )
    )
    allunique(lookup(mask, 1)) || throw(ArgumentError("flag_channels!: a frequency appears more than once in the mask"))
    fm = Float64.(lookup(mask, 1))
    perm = sortperm(fm)
    sf = fm[perm]
    sel = Int[]
    for (c, f) in enumerate(XRadio.frequencies(ms))
        tol = Calibration._FREQ_RTOL * abs(f)
        j = searchsortedfirst(sf, f - tol)
        (j <= length(sf) && abs(sf[j] - f) <= tol) || throw(
            ArgumentError("flag_channels!: the mask does not cover the channel at $f Hz")
        )
        mask[perm[j]] && push!(sel, c)
    end
    isempty(sel) || (view(ms[:flag], Frequency(sel)) .= true)
    return ms
end
