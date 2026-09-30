# ── CalibrationSolution: a list of solved components ─────────────────────────
#
# A solution is a flat list of `SolvedComponent`s, each holding one gain
# component's fitted parameters as a labeled array. Gains are separable over
# components (`exp(Σ logamp)·cis(Σ phase)`), so any subset of the list is a
# solution in its own right. Applying one re-plans the components it holds
# against the solution's geometry, one layout per (step, station set), and runs
# the forward map on θ built from their parameters.

using Statistics: mean
using DimensionalData: lookup, Ti, DimArray, Dim, Dimensions, AbstractDimArray
using DimensionalData.Lookups: Sampled, Explicit, Intervals, Center
using OrderedCollections: OrderedDict
using ..UVData: Frequency, Polarization, BaselineID, AntennaName, Feed

"""
    SolvedComponent(step, path, component, params)

One fitted gain component: `component` (a [`GainComponent`](@ref): term,
segmentation, feed tying, prior) and its parameters `params`, a `DimArray`
over `(param, Feed or node, Frequency, Ti, AntennaName)`. Each `Frequency`/`Ti`
segment is labeled by the midpoint of the channels or samples it covers, with
intervals so `Contains(x)` selects the segment covering `x`; `AntennaName` names the
stations that carry the component.

`step` is the solve step that fit it (`:fringe`, `:bandpass`, …) and `path`
its place in that step's model, starting at `:phase` or `:logamp`
(`(:phase, :mbd)`; `(:phase, :bandpass, :g1)` for one signature group of a
component whose specification differs across stations).
"""
struct SolvedComponent{C <: GainComponent, A <: AbstractDimArray}
    step::Symbol
    path::Tuple{Vararg{Symbol}}
    component::C
    params::A
    function SolvedComponent{C, A}(step, path, component, params) where {C, A}
        p = Tuple(path)
        (!isempty(p) && first(p) in (:phase, :logamp)) || throw(
            ArgumentError("a component's path starts at :phase or :logamp, got $(p)")
        )
        return new{C, A}(step, p, component, params)
    end
end

SolvedComponent(step::Symbol, path, component::GainComponent, params::AbstractDimArray) =
    SolvedComponent{typeof(component), typeof(params)}(step, path, component, params)

_label(c::SolvedComponent) = join((c.step, c.path...), '.')

Base.show(io::IO, c::SolvedComponent) = print(
    io, "SolvedComponent(", _label(c), ", ", nameof(typeof(c.component.term)), ", ",
    join(size(c.params), "×"), ")",
)

"""
    CalibrationSolution(geom, components, steps = OrderedDict(), info = (;); pipeline = "", gauge = "")
    CalibrationSolution(model, layout, geom, θ, info = (;); name = :solution, pipeline = "", gauge = "")

A solved calibration: the list `components` of [`SolvedComponent`](@ref)s over
the geometry `geom`, whose `stations` name every station a component may
carry. Gains compose multiplicatively across components.

- `steps` maps each solve step, in run order, to its diagnostics (detections,
  per-scan SNR, timing, the fringe step's unconstrained `flagged_ant` /
  `flagged_scan`, …).
- `info` holds run-wide diagnostics.
- `provenance` is the pipeline and gauge the run was given, as text for the
  record. A solution does not replay its pipeline: [`calibrate`](@ref) divides
  by its gains only.

Any subset of the components is a solution: `filter(pred, sol)`,
`sol[:fringe]` (one step), `sol[:fringe, :phase, :mbd]` (the components under a
path). The fields are the interface: iterate `sol.components`, read
`c.params`.

The second form builds a one-step solution from a flat parameter vector `θ`
laid out by `layout` (1-based, `length(θ) == layout.nθ`), copying θ; `info`
is that step's diagnostics.
"""
struct CalibrationSolution{G <: DataGeometry}
    geom::G
    components::Vector{SolvedComponent}
    steps::OrderedDict{Symbol, NamedTuple}
    info::NamedTuple
    provenance::@NamedTuple{pipeline::String, gauge::String}
    function CalibrationSolution{G}(geom, components, steps, info, provenance) where {G}
        isempty(geom.stations) && throw(
            ArgumentError(
                "a solution's geometry must name its stations; build it with `DataGeometry(ps)` " *
                    "or pass `stations`"
            )
        )
        labels = map(_label, components)
        allunique(labels) || throw(
            ArgumentError("CalibrationSolution: components repeat: $(join(unique(filter(l -> count(==(l), labels) > 1, labels)), ", "))")
        )
        return new{G}(geom, components, steps, info, provenance)
    end
end

function CalibrationSolution(
        geom::DataGeometry, components::AbstractVector{<:SolvedComponent},
        steps::AbstractDict = OrderedDict{Symbol, NamedTuple}(), info::NamedTuple = NamedTuple();
        pipeline::AbstractString = "", gauge::AbstractString = "",
    )
    return CalibrationSolution{typeof(geom)}(
        geom, collect(SolvedComponent, components), OrderedDict{Symbol, NamedTuple}(steps), info,
        (; pipeline = String(pipeline), gauge = String(gauge)),
    )
end

function CalibrationSolution(
        model::GainModel, layout::ParameterLayout, geom::DataGeometry,
        θ::AbstractVector, info::NamedTuple = NamedTuple();
        name::Symbol = :solution, pipeline::AbstractString = "", gauge::AbstractString = "",
    )
    return CalibrationSolution(
        geom, _solved_components(name, model, layout, θ, geom),
        OrderedDict{Symbol, NamedTuple}(name => info); pipeline, gauge,
    )
end

Base.:(==)(a::SolvedComponent, b::SolvedComponent) =
    a.step === b.step && a.path == b.path && a.component == b.component && a.params == b.params
Base.hash(c::SolvedComponent, h::UInt) =
    hash(c.params, hash(c.component, hash(c.path, hash(c.step, hash(:SolvedComponent, h)))))

# ── Selection ────────────────────────────────────────────────────────────────

# The solution holding `components`, keeping the diagnostics of the steps they
# came from.
function _with_components(sol::CalibrationSolution, components)
    steps = OrderedDict{Symbol, NamedTuple}(k => v for (k, v) in sol.steps if any(c -> c.step === k, components))
    return CalibrationSolution{typeof(sol.geom)}(sol.geom, collect(SolvedComponent, components), steps, sol.info, sol.provenance)
end

"""
    filter(pred, sol::CalibrationSolution) -> CalibrationSolution

The solution holding the components `c` of `sol` for which `pred(c)` is true.
"""
Base.filter(pred, sol::CalibrationSolution) = _with_components(sol, filter(pred, sol.components))

"""
    sol[step]
    sol[step, path...]

The components fit by `step`, or those of `step` whose path begins with
`path` (`sol[:fringe, :phase]`, `sol[:bandpass, :phase, :bandpass, :g1]`), as a
solution. Throws when nothing matches, naming what the solution holds;
`haskey(sol, step)` tests for a step first.
"""
function Base.getindex(sol::CalibrationSolution, step::Symbol, path::Symbol...)
    keep = filter(c -> c.step === step && length(c.path) >= length(path) && c.path[1:length(path)] == path, sol.components)
    isempty(keep) && throw(
        ArgumentError(
            "the solution holds no component under $(join((step, path...), '.')); it holds " *
                join(map(_label, sol.components), ", ")
        )
    )
    return _with_components(sol, keep)
end

Base.haskey(sol::CalibrationSolution, step::Symbol) = any(c -> c.step === step, sol.components)
Base.length(sol::CalibrationSolution) = length(sol.components)
Base.iterate(sol::CalibrationSolution, i...) = iterate(sol.components, i...)
Base.eltype(::Type{<:CalibrationSolution}) = SolvedComponent

function Base.show(io::IO, ::MIME"text/plain", sol::CalibrationSolution)
    println(io, "CalibrationSolution")
    println(io, "  Steps     : ", join(keys(sol.steps), ", "))
    println(io, "  Grid      : ", length(sol.geom.stations), " stations × ", nchannels(sol.geom), " channels × ", ntimes(sol.geom), " times")
    print(io, "  Components:")
    for c in sol.components
        print(io, "\n    ", _label(c), "  ", nameof(typeof(c.component.term)), "  ", join(size(c.params), "×"))
    end
    return io
end

Base.show(io::IO, sol::CalibrationSolution) =
    print(io, "CalibrationSolution(", length(sol.components), " components)")

# ── From a solved step ───────────────────────────────────────────────────────

# The components of one solved step: `layout` laid out `model` over the
# geometry's stations, and `θ` holds its parameters.
function _solved_components(step::Symbol, model::GainModel, layout::ParameterLayout, θ::AbstractVector, geom::DataGeometry)
    Base.require_one_based_indexing(θ)
    length(θ) == layout.nθ || throw(
        DimensionMismatch("θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)")
    )
    names = _station_labels(geom, layout.nant)
    out = SolvedComponent[]
    for group in (:phase, :logamp)
        _collect_components!(
            out, step, getproperty(layout.plantree, group), getproperty(layout.axes, group),
            (group,), θ, model, geom, names,
        )
    end
    return out
end

function _collect_components!(out, step, nt::NamedTuple, axnode, path, θ, model, geom, names)
    for k in keys(nt)
        _collect_components!(out, step, nt[k], axnode[k], (path..., k), θ, model, geom, names)
    end
    return out
end

function _collect_components!(out, step, plan::ComponentPlan, axnode, path, θ, model, geom, names)
    push!(out, SolvedComponent(step, path, _component_at(model, path, first(names)), _params_array(plan, axnode, θ, geom, names)))
    return out
end

function _collect_components!(out, step, g::GroupedComponentPlan, axnode, path, θ, model, geom, names)
    for (i, k) in enumerate(keys(g.groups))
        comp = _component_at(model, path, names[first(g.stations[i])])
        push!(out, SolvedComponent(step, (path..., k), comp, _params_array(g.groups[i], axnode[i], θ, geom, names)))
    end
    return out
end

# The `GainComponent` at `path` (`(:phase, name, …)`) in the component tree
# `station` solves under. Where stations differ only in prior, this is the
# named station's.
function _component_at(model::GainModel, path, station)
    node = station_components(model, station)
    for k in path
        node = getproperty(node, k)
    end
    node isa GainComponent || throw(
        ArgumentError("the model holds no component at $(join(path, '.')), which the layout lays out")
    )
    return node
end

function _params_array(plan::ComponentPlan, axnode, θ, geom, names)
    raw = Array{eltype(θ)}(undef, plan.shape)
    copyto!(raw, view(θ, plan.range))
    return DimArray(raw, _params_dims(plan, axnode, geom, names))
end

_params_dims(plan::ComponentPlan, axnode, geom, names) = ntuple(
    d -> _role_dim(axnode.roles[d], plan.shape[d], geom, names, plan; stations = get(axnode, :stations, nothing)),
    length(plan.shape),
)

# The station names a layout over `n` stations of `geom` addresses.
function _station_labels(geom::DataGeometry, n::Int)
    isempty(geom.stations) && throw(
        ArgumentError(
            "a solution's geometry must name its stations; build it with `DataGeometry(ps)` " *
                "or pass `stations`"
        )
    )
    length(geom.stations) == n || throw(
        DimensionMismatch("the layout covers $n stations but the geometry names $(length(geom.stations))")
    )
    return geom.stations
end

# ── Applying ─────────────────────────────────────────────────────────────────

# Components of one step over one station set, planned together: `stations`
# indexes `geom.stations`.
struct _EvalGroup{M <: GainModel, L <: ParameterLayout, V <: AbstractVector}
    step::Symbol
    model::M
    layout::L
    θ::V
    stations::Vector{Int}
end

# What a solution applies: its geometry and one planned group per
# (step, station set), in component order.
struct _AppliedSolution{G <: DataGeometry}
    geom::G
    groups::Vector{_EvalGroup}
end

function _applied(sol::CalibrationSolution)
    isempty(sol.components) && throw(
        ArgumentError("the solution holds no components, so applying it would change nothing")
    )
    geom = sol.geom
    keys_ = Tuple{Symbol, Vector{String}}[]
    members = Vector{SolvedComponent}[]
    for c in sol.components
        k = (c.step, String.(collect(lookup(c.params, AntennaName))))
        i = findfirst(==(k), keys_)
        if isnothing(i)
            push!(keys_, k)
            push!(members, SolvedComponent[c])
        else
            push!(members[i], c)
        end
    end
    groups = _EvalGroup[_eval_group(step, stations, cs, geom) for ((step, stations), cs) in zip(keys_, members)]
    return _AppliedSolution(geom, groups)
end

function _eval_group(step::Symbol, stations::Vector{String}, cs::Vector{SolvedComponent}, geom::DataGeometry)
    idx = map(stations) do s
        i = findfirst(==(s), geom.stations)
        isnothing(i) && throw(
            ArgumentError("component station `$s` is not among the geometry's stations $(join(geom.stations, ", "))")
        )
        i
    end
    keys_ = [Symbol(:c, i) for i in eachindex(cs)]
    group(g) = (sel = findall(c -> first(c.path) === g, cs); NamedTuple{Tuple(keys_[sel])}(Tuple(cs[i].component for i in sel)))
    model = GainModel(; phase = group(:phase), logamp = group(:logamp))
    layout = plan_parameters(model, length(stations), geom; require_nonempty = false)
    T = mapreduce(c -> eltype(c.params), promote_type, cs)
    θ = zeros(T, layout.nθ)
    for (c, key) in zip(cs, keys_)
        g = first(c.path)
        plan = getproperty(getproperty(layout.plantree, g), key)
        axnode = getproperty(getproperty(layout.axes, g), key)
        _check_params_dims(DimensionalData.dims(c.params), _params_dims(plan, axnode, geom, stations), _label(c))
        copyto!(view(θ, plan.range), parent(c.params))
    end
    return _EvalGroup(step, model, layout, θ, idx)
end

function _check_params_dims(got::Tuple, expected::Tuple, label)
    map(DimensionalData.name, got) == map(DimensionalData.name, expected) || throw(
        DimensionMismatch(
            "$label has dims $(map(DimensionalData.name, got)), but its component lays out " *
                "$(map(DimensionalData.name, expected))"
        )
    )
    for (a, b) in zip(got, expected)
        la, lb = collect(lookup(a)), collect(lookup(b))
        la == lb || throw(
            DimensionMismatch(
                "$label's $(DimensionalData.name(a)) axis is $(repr(la)), but its component lays " *
                    "out $(repr(lb))"
            )
        )
    end
    return nothing
end

_nant(sol::_AppliedSolution) = length(sol.geom.stations)

# The product of every group's gains over an `nchan × ntime` window, each
# group's placed on its own stations; a station a group does not carry takes
# unit gain from it. `args`/`kw` select the window as `evaluate_gains` does.
function _product_gains(sol::_AppliedSolution, nchan::Int, ntime::Int, args...; kw...)
    T = mapreduce(g -> float(eltype(g.θ)), promote_type, sol.groups)
    g = ones(Complex{T}, nchan, ntime, _nant(sol), 2)
    for grp in sol.groups
        view(g, :, :, grp.stations, :) .*= evaluate_gains(grp.layout, grp.θ, args...; kw...)
    end
    return g
end

"""
    gains(sol::CalibrationSolution; Frequency, Ti, AntennaName, Feed) -> DimArray

The complex antenna gains of `sol` — a whole solution or any selection of it
(`sol[:bandpass]`, `sol[:fringe, :phase, :mbd]`, `filter(pred, sol)`) — as a
`DimArray` over `(Frequency, Ti, AntennaName, Feed)`: channel frequencies (Hz),
integration times (seconds), the geometry's stations, and feed.
`gain = exp(Σ logamp) · cis(Σ phase)` over the components is the same forward
map `calibrate` divides by.

The keywords accept anything `DimArray` indexing accepts, under the invariant
`gains(sol; kw...) == gains(sol)[kw...]`. A `Frequency`/`Ti` selector
restricts the evaluation to the selected samples; `AntennaName`/`Feed` slice the
result.
"""
gains(sol::CalibrationSolution; kw...) = gains(_applied(sol); kw...)

function gains(sol::_AppliedSolution; kw...)
    nant = _nant(sol)
    nfeed = 2
    nchan = nchannels(sol.geom)
    ntime = ntimes(sol.geom)
    d = (Frequency(sol.geom.channel_freqs), Ti(sol.geom.times), AntennaName(sol.geom.stations), Feed(1:nfeed))
    isempty(kw) && return DimArray(_product_gains(sol, nchan, ntime), d)
    for k in keys(kw)
        k in (:Frequency, :Ti, :AntennaName, :Feed) || throw(
            ArgumentError(
                "gains: unknown dimension keyword $(repr(k)); the gain axes are " *
                    "Frequency, Ti, AntennaName, Feed."
            )
        )
    end
    # Resolve the selectors against the full-grid lookups without evaluating
    # anything (the reference array is index-valued and never materialized).
    ref = DimArray(CartesianIndices((nchan, ntime, nant, nfeed)), d)
    If, It, Ia, Ife = Dimensions.dims2indices(ref, Dimensions.kw2dims(kw))
    ci = _window_indices(If, nchan)
    ti = _window_indices(It, ntime)
    # Slice the lookups before evaluating: the evaluator reads its segment
    # tables under `@inbounds`, so an out-of-range selector must fail here
    # (a plain `BoundsError`), never inside the evaluation.
    fsel = sol.geom.channel_freqs[ci]
    tsel = sol.geom.times[ti]
    g = _product_gains(sol, length(ci), length(ti), ci, ti)
    A = DimArray(g, (Frequency(fsel), Ti(tsel), d[3], d[4]))
    # Indexing the windowed wrap reproduces `gains(sol)[kw...]` exactly —
    # including an integer selector dropping its dimension.
    return A[_post_index(If), _post_index(It), Ia, Ife]
end

# The evaluation window one resolved index implies, and the residual index
# into the windowed result. An integer selector windows to its single sample
# and then drops the dimension, as `getindex` would.
_window_indices(::Colon, n::Int) = Base.OneTo(n)
_window_indices(i::Integer, ::Int) = i:i
_window_indices(v::AbstractVector{Bool}, ::Int) = findall(v)
_window_indices(v, ::Int) = v
_post_index(::Integer) = 1
_post_index(_) = Colon()

# A lookup over segments of the samples at `coords`, `groups` giving each
# segment's sample indices: each segment is the interval from its lowest to its
# highest sample, labeled by that interval's midpoint, so `Contains(x)` finds
# the segment covering `x` and throws in a gap between segments.
function _segment_lookup(coords, groups)
    lo = [minimum(view(coords, g)) for g in groups]
    hi = [maximum(view(coords, g)) for g in groups]
    return Sampled(
        (lo .+ hi) ./ 2;
        span = Explicit(permutedims(hcat(lo, hi))), sampling = Intervals(Center()),
    )
end

_frequency_segment_lookup(plan::ComponentPlan, geom::DataGeometry, n::Integer = plan.shape[3]) =
    _segment_lookup(geom.channel_freqs, segment_groups(plan.fseg_id, n))
_time_segment_lookup(plan::ComponentPlan, geom::DataGeometry, n::Integer = plan.shape[4]) =
    _segment_lookup(geom.times, segment_groups(plan.tseg_id, n))

# The dimension for one axis of a component's parameters, from its role:
# segment axes over the geometry, `AntennaName` over station names (a signature
# group's over just its `stations`), `param`/`node` positional.
function _role_dim(role::Symbol, n::Int, geom::DataGeometry, names, plan::ComponentPlan; stations = nothing)
    if role === :Frequency
        return Frequency(_frequency_segment_lookup(plan, geom, n))
    elseif role === :Ti
        return Ti(_time_segment_lookup(plan, geom, n))
    elseif role === :AntennaName
        return AntennaName(isnothing(stations) ? collect(names) : collect(names)[stations])
    elseif role === :Feed
        return Feed(1:n)
    else
        return Dim{role}(1:n)
    end
end

# ── Geometry from a ProcessingSet ────────────────────────────────────────────────────

# Two timestamps within `_epoch_atol` of each other are the same instant. The
# axis is built by matching each new value against the canonical ones already
# seen, not by rounding to a grid: a grid still splits two values that straddle
# a bucket edge, which is the case this exists to remove. The canonical value is
# a real observed timestamp, so the axis keeps feeding physical `t - t0`
# arithmetic. `_time_indices` matches `geom.times` with the same tolerance, so
# the constructor and its readers agree.
function _find_canonical_time(canon::AbstractVector{Float64}, t::Float64)
    atol = _epoch_atol(t)
    i = searchsortedfirst(canon, t - atol)
    return (i <= length(canon) && abs(canon[i] - t) <= atol) ? i : nothing
end

function _canonical_time!(canon::Vector{Float64}, t::Float64)
    i = _find_canonical_time(canon, t)
    i === nothing || return canon[i]
    insert!(canon, searchsortedfirst(canon, t), t)
    return t
end

# Read-only counterpart for a second pass over times already canonicalized.
function _canonical_time(canon::AbstractVector{Float64}, t::Float64)
    i = _find_canonical_time(canon, t)
    i === nothing && throw(
        ArgumentError("time $t s was not canonicalized on the first pass")
    )
    return canon[i]
end

"""
    DataGeometry(ps::XRadio.ProcessingSet; f0 = nothing, t0 = nothing) -> DataGeometry

The geometry a solve over `ps` runs on: the sorted distinct channel
frequencies (Hz) and time samples (seconds) of every Measurement Set, each
channel's spectral window from [`XRadio.spectralwindow`](@ref), each time's
scan from the `scan_name` coordinate, and the stations, every antenna named
in an antenna dataset, in the order first seen. `f0` defaults to the mean
channel frequency, `t0` to the first time.

A Measurement Set may hold several scans. Sub-arrays observing different
scans at the same timestamps are supported when their station sets are
disjoint and they share the whole scan window, as for
`DataGeometry`. Throws when a Measurement Set states no `scan_name`,
when a frequency falls in two spectral windows, when two scans share a
timestamp and a station, and when a scan overlaps another over part of its
span.
"""
function DataGeometry(ps::XRadio.ProcessingSet; f0 = nothing, t0 = nothing)
    isempty(ps) && throw(ArgumentError("the processing set holds no Measurement Sets"))
    pieces = [
        (;
                freqs = Float64.(XRadio.frequencies(ms)), spw = XRadio.spectralwindow(ms),
                times = Float64.(XRadio.times(ms)), scans = _scan_labels(ms),
                active = Set{String}(Iterators.flatten(XRadio.baselines(ms))),
            ) for ms in ps
    ]
    stations = String[]
    for ms in ps, name in XRadio.antennas(ms)
        String(name) in stations || push!(stations, String(name))
    end
    return _span_geometry(pieces, stations; f0, t0)
end

function _scan_labels(ms::XRadio.MeasurementSet)
    haskey(ms, :scan_name) || throw(
        ArgumentError(
            "a Measurement Set states no `scan_name`; the geometry labels every time " *
                "sample with its scan"
        )
    )
    return String.(collect(ms[:scan_name]))
end

# The union geometry of `pieces`, each holding channels `freqs` of one spectral
# window `spw` and time samples `times` labelled per sample by `scans`, observed
# by the stations in `active`.
function _span_geometry(pieces, stations; f0, t0)
    freq_spw = Dict{Float64, String}()
    time_scan = Dict{Float64, String}()
    time_stations = Dict{Float64, Set{String}}()
    time_canon = Float64[]
    for piece in pieces
        for f in piece.freqs
            prev = get(freq_spw, f, nothing)
            (prev === nothing || prev == piece.spw) || throw(
                ArgumentError(
                    "channel frequency $f Hz appears in conflicting spectral " *
                        "windows '$prev' and '$(piece.spw)' — a single concatenated channel " *
                        "axis cannot dense-rank it to one spw. Partition the set so each " *
                        "frequency belongs to one spw, or rename the spws consistently."
                )
            )
            freq_spw[f] = piece.spw
        end
        for (t, scan) in zip(piece.times, piece.scans)
            tk = _canonical_time!(time_canon, t)
            prev = get(time_scan, tk, nothing)
            if prev === nothing
                time_scan[tk] = scan
                time_stations[tk] = copy(piece.active)
            elseif prev == scan
                union!(time_stations[tk], piece.active)
            else
                shared = intersect(time_stations[tk], piece.active)
                isempty(shared) || throw(
                    ArgumentError(
                        "station(s) $(join(sort!(collect(shared)), ", ")) are in " *
                            "both scan '$prev' and scan '$scan' at time $tk s. A " *
                            "station cannot be in two scans at one instant, and a single " *
                            "concatenated time axis cannot dense-rank the timestamp to one scan. " *
                            "Check for overlapping scan windows or inconsistent scan names."
                    )
                )
                union!(time_stations[tk], piece.active)
            end
        end
    end

    # Sub-arrays observing different sources at the same timestamps are
    # admitted above, since disjoint station sets carry no contradiction. The
    # timestamp still dense-ranks to one scan, so a scan whose times land under
    # more than one label would have its scan split into separate per-scan
    # parameter segments partway through. Reject that rather than solve it.
    for piece in pieces, name in unique(piece.scans)
        labels = unique(
            time_scan[_canonical_time(time_canon, t)]
                for (t, scan) in zip(piece.times, piece.scans) if scan == name
        )
        length(labels) == 1 || throw(
            ArgumentError(
                "scan '$name' spans timestamps that dense-rank to " *
                    "$(join(("'" * l * "'" for l in labels), ", ")) — it overlaps another " *
                    "scan over part of its span but not all of it, so a single concatenated " *
                    "time axis would segment it into more than one scan. Sub-arrays sharing " *
                    "an entire scan window are supported; partial overlap is not."
            )
        )
    end

    freqs = sort!(collect(keys(freq_spw)))
    times = sort!(collect(keys(time_scan)))

    spw_labels = [freq_spw[f] for f in freqs]
    scan_labels = [time_scan[t] for t in times]

    spw_of_chan, _ = _dense_rank_labels(spw_labels)
    scan_of_time, _ = _dense_rank_labels(scan_labels)

    f0v = isnothing(f0) ? (isempty(freqs) ? 0.0 : mean(freqs)) : Float64(f0)
    t0v = isnothing(t0) ? (isempty(times) ? 0.0 : first(times)) : Float64(t0)

    return DataGeometry(;
        times, channel_freqs = freqs, scan_of_time, spw_of_chan, t0 = t0v, f0 = f0v,
        scan_names = _unique_in_order(scan_labels), spw_names = _unique_in_order(spw_labels),
        stations,
    )
end

# Dense-rank string labels to 1..k preserving first-appearance order.
function _dense_rank_labels(labels::AbstractVector{<:AbstractString})
    seen = Dict{String, Int}()
    out = Vector{Int}(undef, length(labels))
    next = 0
    for i in eachindex(labels)
        id = get(seen, labels[i], 0)
        if id == 0
            next += 1
            seen[labels[i]] = next
            id = next
        end
        out[i] = id
    end
    return out, next
end

function _unique_in_order(labels::AbstractVector{<:AbstractString})
    seen = Set{String}()
    out = String[]
    for l in labels
        if !(l in seen)
            push!(seen, l)
            push!(out, l)
        end
    end
    return out
end

"""
    GeometryWindow(geom::DataGeometry, ms::XRadio.MeasurementSet)
    GeometryWindow(geom::DataGeometry, chan_idx, ti_idx)

A window into a [`DataGeometry`](@ref): the solve's index space projected onto
one Measurement Set, with no data attached.

- `geom` — the solve's geometry, the index space the window addresses.
- `chan_idx`, `ti_idx` — global indices into `geom.channel_freqs` / `geom.times`
  of each channel and time of the Measurement Set, in its own order.
- `stations` — for each baseline, in `baseline_id` order, the pair of indices
  into `geom.stations` of its two antennas.
- `feed_order` — the feed pairs `(feed_a, feed_b)` the baselines relate, sorted.
- `feeds` — `feeds[k, b]` is the stored product of baseline `b` that relates
  `feed_order[k]`.

θ is addressed by POSITION — a component's leaf indexed `(param, node, fseg_id,
tseg_id, ant)` over global-length segment-id tables — while a Measurement Set's
coordinates are PHYSICAL (Hz, seconds, antenna names, correlation labels), so
this join cannot be recovered from the data alone.

`GeometryWindow(geom, ms)` matches channels by frequency (`isapprox`, rtol
1e-9), times by `_epoch_atol` on absolute seconds, and antennas by name, and
throws on any that `geom` does not hold. The three-argument form addresses
channels and times only, which is all evaluating gains needs; its station and
feed maps are empty.
"""
struct GeometryWindow
    geom::DataGeometry
    chan_idx::Vector{Int}
    ti_idx::Vector{Int}
    stations::Vector{Tuple{Int, Int}}
    feed_order::Vector{Tuple{Int, Int}}
    feeds::Matrix{Int}
end

GeometryWindow(geom::DataGeometry, chan_idx, ti_idx) =
    GeometryWindow(geom, chan_idx, ti_idx, Tuple{Int, Int}[], Tuple{Int, Int}[], zeros(Int, 0, 0))

function GeometryWindow(geom::DataGeometry, ms::XRadio.MeasurementSet)
    isempty(geom.stations) && throw(
        ArgumentError("the geometry names no stations; build it with `DataGeometry(ps)`")
    )
    slot = Dict(n => i for (i, n) in pairs(geom.stations))
    station(n) = get(slot, String(n)) do
        throw(
            ArgumentError(
                "baseline antenna `$n` is not among the geometry's stations, " *
                    join(geom.stations, ", ")
            )
        )
    end
    stations = [(station(a), station(b)) for (a, b) in XRadio.baselines(ms)]
    order, perm = UVData._feed_permutation(feed_pairs(ms))
    return GeometryWindow(
        geom, _channel_indices(geom, XRadio.frequencies(ms)),
        _time_indices(geom, XRadio.times(ms)), stations, order, perm,
    )
end

_channel_indices(geom::DataGeometry, fs) = [_channel_index(geom, Float64(f)) for f in fs]

function _channel_index(geom::DataGeometry, f)
    j = findfirst(g -> isapprox(g, f; rtol = 1.0e-9), geom.channel_freqs)
    isnothing(j) && throw(ArgumentError("channel frequency $f Hz is not in the geometry"))
    return j
end

_time_indices(geom::DataGeometry, ts) = [_time_index(geom, Float64(t)) for t in ts]

function _time_index(geom::DataGeometry, t)
    j = findfirst(g -> isapprox(g, t; atol = _epoch_atol(t)), geom.times)
    isnothing(j) && throw(ArgumentError("time $t s is not in the geometry"))
    return j
end

"""
    gains(sol::CalibrationSolution, win::GeometryWindow; time_span = nothing) -> DimArray

The gains of `sol` at the samples `win` addresses: `win.chan_idx` and
`win.ti_idx` index `win.geom`, which may be `sol.geom` itself or the geometry
of other data. On a foreign geometry each sample is placed in the solve
segment it belongs to (see [`evaluate_gains`](@ref)); `time_span[k]` is the
interval the `k`-th selected time integrates over, so a sample straddling a
segment boundary is rejected. Labeled like [`gains`](@ref)`(sol)`.
"""
gains(sol::CalibrationSolution, win::GeometryWindow; time_span = nothing) =
    gains(_applied(sol), win; time_span)

function gains(sol::_AppliedSolution, win::GeometryWindow; time_span = nothing)
    nc, nt = length(win.chan_idx), length(win.ti_idx)
    g = win.geom === sol.geom ? _product_gains(sol, nc, nt, win.chan_idx, win.ti_idx) :
        _product_gains(
            sol, nc, nt, sol.geom, win.geom; chan_idx = win.chan_idx, ti_idx = win.ti_idx, time_span,
        )
    return DimArray(
        g, (
            Frequency(win.geom.channel_freqs[win.chan_idx]), Ti(win.geom.times[win.ti_idx]),
            AntennaName(sol.geom.stations), Feed(1:2),
        ),
    )
end
