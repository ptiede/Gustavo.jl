# ── Corrections: Measurement Set → Measurement Set ──────────────────────────
#
# A correction takes a Measurement Set and returns a corrected copy. Any such
# function may sit in a pipeline; the structs below are the ones a solution can
# record and replay. Inside `fit` and `calibrate` a correction is applied through
# `_correct(x, ms, geom)`, which hands the ones that address the run's index
# space (channels, stations) the run's geometry; called on its own, such a
# correction builds the geometry of the Measurement Set it is given.

"""
    AbstractDataTransform

A correction a solution records: a callable struct `t(ms) -> MeasurementSet`
returning a corrected copy of `ms`. In a pipeline it corrects the data every
later solve step reads, and `calibrate` replays it. Built-ins:
[`AutocorrelationNormalization`](@ref), [`ApplySolution`](@ref),
[`StationWeightScale`](@ref), [`FlagChannels`](@ref).

A plain function from a Measurement Set to a Measurement Set can sit in a
pipeline as well; a solution holding one cannot be replayed after
[`load_solution`](@ref), since functions are not saved.
"""
abstract type AbstractDataTransform end

_correct(x, ms::XRadio.MeasurementSet, geom::DataGeometry) = x(ms)

function _check_corrected(x, got)
    got isa XRadio.MeasurementSet || throw(
        ArgumentError(
            "a correction must return a MeasurementSet; $(nameof(typeof(x))) returned a " *
                "$(nameof(typeof(got)))"
        )
    )
    return got
end

# `corrections` applied to `ms` in order.
_apply_corrections(corrections, ms::XRadio.MeasurementSet, geom::DataGeometry) =
    foldl((acc, x) -> _check_corrected(x, _correct(x, acc, geom)), corrections; init = ms)

_one_member(ms::XRadio.MeasurementSet) =
    XRadio.ProcessingSet(OrderedDict{Symbol, XRadio.MeasurementSet}(:ms => ms))

# The copy of `ms` with the given layers replaced.
function _with_layers(ms::XRadio.MeasurementSet; layers...)
    out = copy(ms)
    for (k, v) in pairs(layers)
        out[k] = v
    end
    return out
end

# The interval each time sample integrates over, from the time coordinate's
# `integration_time`; `nothing` when the Measurement Set does not state it.
function _time_span(ms::XRadio.MeasurementSet)
    md = DimensionalData.metadata(lookup(ms[:visibility], Ti))
    md isa AbstractDict || return nothing
    it = get(md, :integration_time, nothing)
    it === nothing && return nothing
    return fill(Float64(it.value), length(XRadio.times(ms)))
end

# ── Built-in: normalization by the autocorrelations ─────────────────────────

"""
    AutocorrelationNormalization()

Correction: [`normalize_by_autocorrelations`](@ref), so cross-correlations
become correlation coefficients and the autocorrelation baselines are flagged.
The default pipelines start with it, and a solution records it, so
`calibrate` applies gains to data on the scale they were solved on.
"""
struct AutocorrelationNormalization <: AbstractDataTransform end

(::AutocorrelationNormalization)(ms::XRadio.MeasurementSet) = normalize_by_autocorrelations(ms)

# ── Built-in: dividing out a solution's gains ────────────────────────────────

"""
    ApplySolution(sol::CalibrationSolution)

Correction: divide `sol`'s gains out of the data (`V → V / (g_a·conj(g_b))`,
`w → w·|g_a g_b|²`), e.g. a phase-cal solution or an earlier fit. A solution
fit on top of it is the correction on top of `sol`. Cells where the gain is
non-finite or zero are left untouched.

The solution need not have been fit on this data, nor on its sampling. Each
sample is placed in the segment of `sol` it belongs to, by the identity the
segmentation is defined on: a `PerScan` component by scan name, a
`PerSpectralWindow` one by spectral window name, a `TimeBlocks` or
`InstrumentScans` one by the solve's own bin formula, a `PerIntegration` one by
exact epoch. So a solution segmented more coarsely than the data applies, while
one segmented more finely is refused. `ChannelBlocks` and `FreqGroups` cut the
channel-index axis, so they require the data to index the same channels: in a
pipeline or `calibrate` that is the whole processing set's channel axis, and
`ApplySolution(sol)(ms)` on its own uses the channels of `ms` alone.

Stations are matched by name against the solution's recorded `ant_names`;
stations the solution never solved keep identity gains, with a warning. A
solution that records no `ant_names`, or shares no station with the data, is
refused.
"""
struct ApplySolution{S <: CalibrationSolution} <: AbstractDataTransform
    sol::S
end

(t::ApplySolution)(ms::XRadio.MeasurementSet) = _correct(t, ms, DataGeometry(_one_member(ms)))

_correct(t::ApplySolution, ms::XRadio.MeasurementSet, geom::DataGeometry) =
    _divide_gains(ms, GeometryWindow(geom, ms), t.sol; flag_bad = false)

# Target station index → the solution's own (0 = absent from the solution).
function _station_map(sol::CalibrationSolution, stations)
    solnames = _solution_ant_names(sol)
    m = [something(findfirst(==(n), solnames), 0) for n in stations]
    all(iszero, m) && throw(
        ArgumentError(
            "ApplySolution: the solution shares no station with this data, so it would correct " *
                "nothing. It knows $(join(map(repr, solnames), ", ")); the data has " *
                "$(join(map(repr, stations), ", "))."
        )
    )
    if any(iszero, m)
        missing_names = [n for (n, k) in zip(stations, m) if k == 0]
        @warn "ApplySolution: stations $(missing_names) are not in the solution — they keep identity gains." maxlog = 1
    end
    return m
end

# The station names a solution matches on. One that records none has no station
# identity at all — only its own set's positional order, which means nothing
# anywhere else — so it cannot be applied.
function _solution_ant_names(sol::CalibrationSolution)
    (hasproperty(sol.info, :ant_names) && !isempty(sol.info.ant_names)) || throw(
        ArgumentError(
            "ApplySolution: the solution records no station names, so its θ rows could only be " *
                "matched to the data's stations by position — which means nothing across sets. " *
                "Re-solve so the solution records `ant_names`."
        )
    )
    return String.(collect(sol.info.ant_names))
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
function _time_constant_over(sol::CalibrationSolution, win::GeometryWindow, tspan)
    length(win.ti_idx) <= 1 && return true
    for s in sol.steps, plan in s.layout.plans
        :Ti in Calibration.term_axes(plan.term) && return false
        ids = Calibration.time_segment_ids(
            plan.tseg, sol.geom, win.geom; ti_idx = win.ti_idx, time_span = tspan,
        )
        allequal(ids) || return false
    end
    return true
end

# The span of just the first sample, to match a single evaluated time column.
_head_span(span) = span === nothing || isempty(span) ? span : span[1:1]

# A copy of `ms` with `sol`'s gains divided out, placed through `win`. A cell
# whose gain is degenerate is left as it is (`flag_bad = false`), or given a NaN
# visibility and a flag (`flag_bad = true`).
function _divide_gains(
        ms::XRadio.MeasurementSet, win::GeometryWindow, sol::CalibrationSolution;
        flag_bad::Bool, executor = SerialScheduler(),
    )
    amap = _station_map(sol, win.geom.stations)
    vis = modify(Array, ms[:visibility])
    weight = modify(Array, ms[:weight])
    flag = modify(Array, ms[:flag])
    feeds = feed_pairs(ms)
    tspan = _time_span(ms)
    tconst = _time_constant_over(sol, win, tspan)
    gwin = tconst ? GeometryWindow(win.geom, win.chan_idx, win.ti_idx[1:1]) : win
    g = parent(gains(sol, gwin; time_span = tconst ? _head_span(tspan) : tspan))
    cols = vec(CartesianIndices(feeds))
    tforeach(cols; scheduler = executor) do col
        p, bi = Tuple(col)
        a, b = win.stations[bi]
        sa, sb = amap[a], amap[b]
        (sa == 0 || sb == 0) && return
        fa, fb = feeds[p, bi]
        _divide_column!(
            UVData._cell_plane(vis, bi, p), UVData._cell_plane(weight, bi, p),
            UVData._cell_plane(flag, bi, p), g, sa, sb, fa, fb, tconst, flag_bad,
        )
    end
    return _with_layers(ms; visibility = vis, weight, flag)
end

# Gain magnitude below which a cell is treated as unconstrained rather than
# divided through: the correction would amplify noise without bound.
const _GAIN_FLOOR = 1.0e-12

# One `(Frequency, Ti)` plane of `_divide_gains`. The gains share its axes (one
# time when `tconst`).
function _divide_column!(V, W, F, g, a, b, fa, fb, tconst::Bool, flag_bad::Bool)
    axes(V, 1) == axes(g, 1) && (tconst || axes(V, 2) == axes(g, 2)) || throw(
        DimensionMismatch("gains $(axes(g)) do not cover the plane $(axes(V))"),
    )
    for t in axes(V, 2)
        gt = tconst ? firstindex(g, 2) : t
        for c in axes(V, 1)
            ga = g[c, gt, a, fa]
            gb = g[c, gt, b, fb]
            den = ga * conj(gb)
            if !isfinite(den) || abs(ga) < _GAIN_FLOOR || abs(gb) < _GAIN_FLOOR
                if flag_bad
                    V[c, t] = convert(eltype(V), NaN)
                    F[c, t] = true
                end
                continue
            end
            V[c, t] /= den
            W[c, t] *= abs2(den)
        end
    end
    return nothing
end

# ── Built-in: per-station weight scaling ─────────────────────────────────────

"""
    StationWeightScale(scale::AbstractDimVector)

Correction: per-station weight scaling, `w → w·s_a·s_b` on each baseline
`(a, b)`. `scale` holds one finite, positive factor per station, indexed by
`AntennaName`; a station it does not name keeps its weights. A factor above 1
raises a station's weights (a correlator claiming more noise than the data
carries). Visibilities are untouched.

    ws = DimArray([2.0, 2.0], AntennaName(["HS", "GL"]))
    fit(StationWeightScale(ws) |> BaselineFringeFit(), ps; gauge)
"""
struct StationWeightScale{S <: AbstractDimVector} <: AbstractDataTransform
    scale::S
    function StationWeightScale{S}(scale) where {S}
        only(dims(scale)) isa XRadio.AntennaName || throw(
            ArgumentError(
                "StationWeightScale: index the factors by `AntennaName`, not " *
                    "$(nameof(typeof(only(dims(scale)))))"
            )
        )
        allunique(lookup(scale, 1)) || throw(
            ArgumentError("StationWeightScale: a station is named more than once: $(collect(lookup(scale, 1)))")
        )
        all(x -> isfinite(x) && x > 0, scale) || throw(
            ArgumentError("StationWeightScale: every factor must be finite and positive, got $(collect(scale)).")
        )
        return new{S}(scale)
    end
end
StationWeightScale(scale::AbstractDimVector) = StationWeightScale{typeof(scale)}(scale)

function (t::StationWeightScale)(ms::XRadio.MeasurementSet)
    factor = Dict(String(n) => Float64(f) for (n, f) in zip(lookup(t.scale, 1), t.scale))
    weight = modify(Array, ms[:weight])
    for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
        f = get(factor, String(a), 1.0) * get(factor, String(b), 1.0)
        f == 1 && continue
        view(weight, BaselineID(bi)) .*= f
    end
    return _with_layers(ms; weight)
end

# ── Built-in: channel flagging ───────────────────────────────────────────────

"""
    FlagChannels(mask::AbstractVector{Bool})

Correction: flag the channels of the run's channel axis
(`DataGeometry(ps).channel_freqs`) where `mask` is `true`, e.g.
`tone_channel_mask`. Visibilities and weights are left as they are.
"""
struct FlagChannels <: AbstractDataTransform
    mask::BitVector
end
FlagChannels(mask::AbstractVector{Bool}) = FlagChannels(BitVector(mask))

(t::FlagChannels)(ms::XRadio.MeasurementSet) = _correct(t, ms, DataGeometry(_one_member(ms)))

function _correct(t::FlagChannels, ms::XRadio.MeasurementSet, geom::DataGeometry)
    length(t.mask) == length(geom.channel_freqs) || throw(
        DimensionMismatch(
            "FlagChannels: the mask covers $(length(t.mask)) channels, the data " *
                "$(length(geom.channel_freqs))"
        )
    )
    sel = findall(t.mask[GeometryWindow(geom, ms).chan_idx])
    isempty(sel) && return ms
    flag = modify(Array, ms[:flag])
    view(flag, Frequency(sel)) .= true
    return _with_layers(ms; flag)
end
