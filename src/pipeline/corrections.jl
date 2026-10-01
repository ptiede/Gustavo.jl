# ── Corrections: Measurement Set → Measurement Set ──────────────────────────
#
# A correction takes a Measurement Set and returns a corrected copy. Any such
# function may sit in a pipeline; the structs below are the built-in ones.
# Inside `fit` a correction is applied through
# `_correct(x, ms, geom)`, which hands the ones that address the run's index
# space (channels, stations) the run's geometry; called on its own, such a
# correction builds the geometry of the Measurement Set it is given.

"""
    AbstractDataTransform

A built-in correction: a callable struct `t(ms) -> MeasurementSet` returning a
corrected copy of `ms`. In a pipeline it corrects the data every later solve
step reads. Built-ins:
[`AutocorrelationNormalization`](@ref), [`ApplySolution`](@ref),
[`StationWeightScale`](@ref), [`FlagChannels`](@ref).

A plain function from a Measurement Set to a Measurement Set can sit in a
pipeline as well.
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
The default pipelines start with it; `calibrate(pipeline, sol, ps)` repeats it,
so the gains divide data on the scale they were solved on.
"""
struct AutocorrelationNormalization <: AbstractDataTransform end

(::AutocorrelationNormalization)(ms::XRadio.MeasurementSet) = normalize_by_autocorrelations(ms)

# ── Built-in: dividing out a solution's gains ────────────────────────────────

"""
    ApplySolution(sol::CalibrationSolution)

Correction: divide the gains of `sol` (a solution, or any selection of one) out
of the data (`V → V / (g_a·conj(g_b))`, `w → w·|g_a g_b|²`), e.g. a phase-cal
solution or an earlier fit. A solution
fit on top of it is the correction on top of `sol`. Cells where the gain is
non-finite or zero are left untouched.

The solution need not have been fit on this data, nor on its sampling. Each
sample is placed in the segment of `sol` it belongs to, by the identity the
segmentation is defined on: a `PerScan` component by scan name, a
`PerSpectralWindow` one by spectral window name, a `TimeBlocks` or
`InstrumentScans` one by the solve's own bin formula, a `PerIntegration` one by
exact epoch. So a solution segmented more coarsely than the data applies, while
one segmented more finely is refused. A `ChannelBlocks` or `FreqGroups`
component places each channel by its center frequency and width: the data's
channel must be a channel of the solve, no wider, so averaged channels are
refused.

Stations are matched by name against the solution's geometry; stations the
solution never solved keep identity gains, with a warning. A solution that
shares no station with the data is refused.
"""
struct ApplySolution{S <: Calibration._AppliedSolution} <: AbstractDataTransform
    sol::S
end

ApplySolution(sol::CalibrationSolution) = ApplySolution(Calibration._applied(sol))

Base.show(io::IO, t::ApplySolution) =
    print(io, "ApplySolution(", join(unique(g.step for g in t.sol.groups), ", "), ")")

(t::ApplySolution)(ms::XRadio.MeasurementSet) = _correct(t, ms, DataGeometry(_one_member(ms)))

_correct(t::ApplySolution, ms::XRadio.MeasurementSet, geom::DataGeometry) =
    _divide_gains(ms, GeometryWindow(geom, ms), t.sol; flag_bad = false)

# Target station index → the solution's own (0 = absent from the solution).
function _station_map(sol::Calibration._AppliedSolution, stations)
    solnames = sol.geom.stations
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

# A copy of `ms` with `sol`'s gains divided out, placed through `win`. A cell
# whose gain is degenerate is left as it is (`flag_bad = false`), or given a NaN
# visibility and a flag (`flag_bad = true`).
function _divide_gains(
        ms::XRadio.MeasurementSet, win::GeometryWindow, sol::Calibration._AppliedSolution;
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
    fit(StationWeightScale(ws) |> BaselineFringeFit(; gauge = PinAntenna("HS")), ps)
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
    FlagChannels(mask::AbstractDimVector{Bool})

Correction: flag the channels where `mask`, indexed by `Frequency` (Hz), is
`true`. Each channel of the data takes the mask entry at its center frequency;
a channel the mask does not cover throws. Visibilities and weights are left as
they are.

    freqs = XRadio.frequencies(ms)
    FlagChannels(DimArray(86.10e9 .< freqs .< 86.12e9, Frequency(freqs)))
"""
struct FlagChannels{M <: AbstractDimVector{Bool}} <: AbstractDataTransform
    mask::M
    function FlagChannels{M}(mask) where {M}
        only(dims(mask)) isa Frequency || throw(
            ArgumentError(
                "FlagChannels: index the mask by `Frequency`, not $(nameof(typeof(only(dims(mask)))))"
            )
        )
        allunique(lookup(mask, 1)) || throw(ArgumentError("FlagChannels: a frequency appears more than once in the mask"))
        return new{M}(mask)
    end
end
FlagChannels(mask::AbstractDimVector{Bool}) = FlagChannels{typeof(mask)}(mask)

Base.show(io::IO, t::FlagChannels) =
    print(io, "FlagChannels(", count(t.mask), " of ", length(t.mask), " channels)")

function (t::FlagChannels)(ms::XRadio.MeasurementSet)
    fm = Float64.(lookup(t.mask, 1))
    perm = sortperm(fm)
    sf = fm[perm]
    sel = Int[]
    for (c, f) in enumerate(XRadio.frequencies(ms))
        tol = Calibration._FREQ_RTOL * abs(f)
        j = searchsortedfirst(sf, f - tol)
        (j <= length(sf) && abs(sf[j] - f) <= tol) || throw(
            ArgumentError("FlagChannels: the mask does not cover the channel at $f Hz")
        )
        t.mask[perm[j]] && push!(sel, c)
    end
    isempty(sel) && return ms
    flag = modify(Array, ms[:flag])
    view(flag, Frequency(sel)) .= true
    return _with_layers(ms; flag)
end
