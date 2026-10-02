# ── Corrections: modify a Measurement Set in place ──────────────────────────
#
# A correction modifies the layers of the Measurement Set it is handed and
# returns it. Any such function may sit in a pipeline; the structs below are the
# built-in ones. Inside `fit` a correction is applied through
# `_correct(x, ms, geom)`, which hands the ones that address the run's index
# space (channels, stations) the run's geometry; called on its own, such a
# correction builds the geometry of the Measurement Set it is given.

"""
    AbstractDataTransform

A built-in correction: a callable struct `t(ms) -> ms` that modifies the
layers of `ms` in place and returns it, as an `f!` function would. To keep the
original, correct a copy that owns its arrays (`t(deepcopy(ms))`); `copy` and
`read` of an in-memory Measurement Set share them. In a pipeline it corrects
the data every later solve step reads; `fit` and `calibrate` hand it a group
read for the purpose, so the data passed to them is never modified. Built-ins:
[`AutocorrelationNormalization`](@ref), [`StationWeightScale`](@ref),
[`FlagChannels`](@ref), and [`GainCorrection`](@ref), which
[`calibrate!`](@ref)`(sol)` returns to carry in an earlier solution.

A plain function that modifies a Measurement Set and returns it can sit in a
pipeline as well.
"""
abstract type AbstractDataTransform end

_correct(x, ms::XRadio.MeasurementSet, geom::DataGeometry) = x(ms)

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

# `corrections` applied to `ms` in order, in place.
_apply_corrections!(corrections, ms::XRadio.MeasurementSet, geom::DataGeometry) =
    foldl((acc, x) -> _check_corrected(x, acc, _correct(x, acc, geom)), corrections; init = ms)

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

# ── Built-in: normalization by the autocorrelations ─────────────────────────

"""
    AutocorrelationNormalization()

Correction: [`normalize_by_autocorrelations!`](@ref), so cross-correlations
become correlation coefficients and the autocorrelation baselines are flagged.
The default pipelines start with it; `calibrate(pipeline, sol, ps)` repeats it,
so the gains divide data on the scale they were solved on.
"""
struct AutocorrelationNormalization <: AbstractDataTransform end

(::AutocorrelationNormalization)(ms::XRadio.MeasurementSet) = normalize_by_autocorrelations!(ms)

# ── Dividing out a solution's gains ──────────────────────────────────────────

"""
    GainCorrection

The correction [`calibrate!`](@ref)`(sol; flag_bad, apply_flags)` returns:
`g(ms)` is `calibrate!(sol, ms; flag_bad, apply_flags)`, with `sol` compiled
once. It sits in a pipeline beside solve steps and other corrections, and
composes with `∘`.
"""
struct GainCorrection{S <: Calibration._AppliedSolution, F} <: AbstractDataTransform
    app::S
    flag_bad::Bool
    apply_flags::Bool
    flagged::F
end

function Base.show(io::IO, g::GainCorrection)
    print(io, "calibrate!(", join(unique(grp.step for grp in g.app.groups), ", "))
    kw = [k for (k, on) in ("flag_bad = false" => g.flag_bad, "apply_flags = false" => g.apply_flags) if !on]
    isempty(kw) || print(io, "; ", join(kw, ", "))
    print(io, ")")
end

(g::GainCorrection)(ms::XRadio.MeasurementSet) = _correct(g, ms, DataGeometry(_one_member(ms)))

function _correct(g::GainCorrection, ms::XRadio.MeasurementSet, geom::DataGeometry)
    _divide_gains!(ms, GeometryWindow(geom, ms), g.app; g.flag_bad)
    return _flag_unconstrained!(ms, g.app.geom, geom, g.flagged)
end

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

# ── Built-in: per-station weight scaling ─────────────────────────────────────

"""
    StationWeightScale(scale::AbstractDimVector)

Correction: per-station weight scaling in place, `w → w·s_a·s_b` on each baseline
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
        only(dims(scale)) isa AntennaName || throw(
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
    weight = ms[:weight]
    for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
        f = get(factor, String(a), 1.0) * get(factor, String(b), 1.0)
        f == 1 && continue
        view(weight, BaselineID(bi)) .*= f
    end
    return ms
end

# ── Built-in: channel flagging ───────────────────────────────────────────────

"""
    FlagChannels(mask::AbstractDimVector{Bool})

Correction: flag, in place, the channels where `mask`, indexed by `Frequency` (Hz), is
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
    isempty(sel) || (view(ms[:flag], Frequency(sel)) .= true)
    return ms
end
