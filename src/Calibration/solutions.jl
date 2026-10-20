# ── CalibrationSolution: one step's solved components ───────────────────────
#
# A solution is what one solve step fit: a flat list of `ComponentFit`s,
# each holding one gain component's fitted parameters as a labeled array, and
# the step's diagnostics. Gains are separable over components
# (`exp(Σ logamp)·cis(Σ phase)`), so any subset of the list is a solution in its
# own right. Applying one re-plans the components it holds against the
# solution's geometry, one layout per station set, and runs the forward map on
# θ built from their parameters.

using Statistics: mean
using DimensionalData: lookup, Ti, DimArray, Dim, Dimensions, AbstractDimArray
using DimensionalData.Lookups: Sampled, Explicit, Intervals, Center
using OrderedCollections: OrderedDict

"""
    ComponentFit(path, component, params)

One fitted gain component: `component` (a [`GainComponent`](@ref): term,
segmentation, feed tying, and one prior for all of its parameters) and its
parameters `params`, a `DimArray` over `(param, Feed or node, Frequency, Ti,
AntennaName)`. Each `Frequency`/`Ti`
segment is labeled by the midpoint of the channels or samples it covers, with
intervals so `Contains(x)` selects the segment covering `x`; `AntennaName` names the
stations that carry the component.

`path` is its place in the step's model, starting at `:phase` or `:logamp`
(`(:phase, :mbd)`; `(:phase, :bandpass, :g1)` for one signature group of a
component whose specification differs across stations).
"""
struct ComponentFit{C <: GainComponent, A <: AbstractDimArray}
    path::Tuple{Vararg{Symbol}}
    component::C
    params::A
    function ComponentFit{C, A}(path, component, params) where {C, A}
        p = Tuple(path)
        (!isempty(p) && first(p) in (:phase, :logamp)) || throw(
            ArgumentError("a component's path starts at :phase or :logamp, got $(p)")
        )
        return new{C, A}(p, component, params)
    end
end

ComponentFit(path, component::GainComponent, params::AbstractDimArray) =
    ComponentFit{typeof(component), typeof(params)}(path, component, params)

_label(c::ComponentFit) = join(c.path, '.')

Base.show(io::IO, c::ComponentFit) = print(
    io, "ComponentFit(", _label(c), ", ", nameof(typeof(c.component.term)), ", ",
    join(size(c.params), "×"), ")",
)

"""
    CalibrationSolution(geom, name, components, info = (;); step = "")
    CalibrationSolution(model, layout, geom, θ, info = (;); name = :solution, step = "")

What one solve step fit: the list `components` of [`ComponentFit`](@ref)s
over the geometry `geom`, whose `antennas` name every station a component may
carry. Gains compose multiplicatively across components.

- `name` names the step's solution (`:fringe`, `:bandpass`, `:adhoc`).
- `info` holds the step's diagnostics (detections, per-scan SNR, timing, the
  fringe step's unconstrained `flagged_ant` / `flagged_feed` / `flagged_scan`,
  …) and the run's (`nscan`, task counts).
- `step` is the solve step that was fit, its gauge included, as text for the
  record.

Any subset of the components is a solution: `filter(pred, sol)`,
`sol[:phase, :mbd]` (the components under a path). The fields are the
interface: iterate `sol.components`, read `c.params`. Steps compose by applying
one solution to the data ([`calibrate!`](@ref Gustavo.calibrate!)) before
fitting the next.

The second form builds a solution from a flat parameter vector `θ` laid out by
`layout` (1-based, `length(θ) == layout.nθ`), copying θ.
"""
struct CalibrationSolution{G <: DataGeometry}
    geom::G
    name::Symbol
    components::Vector{ComponentFit}
    info::NamedTuple
    step::String
    function CalibrationSolution{G}(geom, name, components, info, step) where {G}
        isempty(geom.antennas) && throw(
            ArgumentError(
                "a solution's geometry must name its antennas; build it with `DataGeometry(ps)` " *
                    "or pass `antennas`"
            )
        )
        labels = map(_label, components)
        allunique(labels) || throw(
            ArgumentError("CalibrationSolution: components repeat: $(join(unique(filter(l -> count(==(l), labels) > 1, labels)), ", "))")
        )
        return new{G}(geom, name, components, info, step)
    end
end

function CalibrationSolution(
        geom::DataGeometry, name::Symbol, components::AbstractVector{<:ComponentFit},
        info::NamedTuple = NamedTuple(); step::AbstractString = "",
    )
    return CalibrationSolution{typeof(geom)}(geom, name, collect(ComponentFit, components), info, String(step))
end

function CalibrationSolution(
        model::GainModel, layout::ParameterLayout, geom::DataGeometry,
        θ::AbstractVector, info::NamedTuple = NamedTuple();
        name::Symbol = :solution, step::AbstractString = "",
    )
    return CalibrationSolution(geom, name, _solved_components(model, layout, θ, geom), info; step)
end

Base.:(==)(a::ComponentFit, b::ComponentFit) =
    a.path == b.path && a.component == b.component && a.params == b.params
Base.hash(c::ComponentFit, h::UInt) =
    hash(c.params, hash(c.component, hash(c.path, hash(:ComponentFit, h))))

# ── Selection ────────────────────────────────────────────────────────────────

# The solution holding `components`, with the step's diagnostics.
_with_components(sol::CalibrationSolution, components) =
    CalibrationSolution{typeof(sol.geom)}(sol.geom, sol.name, collect(ComponentFit, components), sol.info, sol.step)

"""
    filter(pred, sol::CalibrationSolution) -> CalibrationSolution

The solution holding the components `c` of `sol` for which `pred(c)` is true.
"""
Base.filter(pred, sol::CalibrationSolution) = _with_components(sol, filter(pred, sol.components))

"""
    sol[path...]

The components whose path begins with `path` (`sol[:phase]`,
`sol[:phase, :bandpass, :g1]`), as a solution. Throws when nothing matches,
naming what the solution holds; `haskey(sol, path...)` tests first.
"""
function Base.getindex(sol::CalibrationSolution, path::Symbol...)
    keep = filter(c -> _under(c, path), sol.components)
    isempty(keep) && throw(
        ArgumentError(
            "the solution holds no component under $(join(path, '.')); it holds " *
                join(map(_label, sol.components), ", ")
        )
    )
    return _with_components(sol, keep)
end

_under(c::ComponentFit, path) = length(c.path) >= length(path) && c.path[1:length(path)] == path

Base.haskey(sol::CalibrationSolution, path::Symbol...) = any(c -> _under(c, path), sol.components)
Base.length(sol::CalibrationSolution) = length(sol.components)
Base.iterate(sol::CalibrationSolution, i...) = iterate(sol.components, i...)
Base.eltype(::Type{<:CalibrationSolution}) = ComponentFit

function Base.show(io::IO, ::MIME"text/plain", sol::CalibrationSolution)
    println(io, "CalibrationSolution")
    println(io, "  Step      : ", sol.name)
    println(io, "  Grid      : ", length(sol.geom.antennas), " antennas × ", nchannels(sol.geom), " channels × ", ntimes(sol.geom), " times")
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
function _solved_components(model::GainModel, layout::ParameterLayout, θ::AbstractVector, geom::DataGeometry)
    Base.require_one_based_indexing(θ)
    length(θ) == layout.nθ || throw(
        DimensionMismatch("θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)")
    )
    names = _antenna_labels(geom, layout.nant)
    out = ComponentFit[]
    for group in (:phase, :logamp)
        _collect_components!(
            out, getproperty(layout.plantree, group), getproperty(layout.axes, group),
            (group,), θ, model, geom, names,
        )
    end
    return out
end

function _collect_components!(out, nt::NamedTuple, axnode, path, θ, model, geom, names)
    for k in keys(nt)
        _collect_components!(out, nt[k], axnode[k], (path..., k), θ, model, geom, names)
    end
    return out
end

function _collect_components!(out, plan::ComponentPlan, axnode, path, θ, model, geom, names)
    push!(out, ComponentFit(path, _component_at(model, path, first(names)), _params_array(plan, axnode, θ, geom, names)))
    return out
end

function _collect_components!(out, g::GroupedComponentPlan, axnode, path, θ, model, geom, names)
    for (i, k) in enumerate(keys(g.groups))
        comp = _component_at(model, path, names[first(g.antennas[i])])
        push!(out, ComponentFit((path..., k), comp, _params_array(g.groups[i], axnode[i], θ, geom, names)))
    end
    return out
end

# The `GainComponent` at `path` (`(:phase, name, …)`) in the component tree
# `antenna` solves under. Where stations differ only in prior, this is the
# named station's.
function _component_at(model::GainModel, path, antenna)
    node = antenna_components(model, antenna)
    for k in path
        node = getproperty(node, k)
    end
    node isa GainComponent || throw(
        ArgumentError("the model holds no component at $(join(path, '.')), which the layout lays out")
    )
    return node
end

function _params_array(plan::ComponentPlan, axnode, θ, geom, names)
    raw = Array{eltype(θ)}(undef, Tuple(plan.shape))
    copyto!(raw, view(θ, plan.range))
    return DimArray(raw, _params_dims(plan, axnode, geom, names))
end

_params_dims(plan::ComponentPlan, axnode, geom, names) = ntuple(
    d -> _role_dim(axnode.roles[d], plan.shape[d], geom, names, plan; antennas = get(axnode, :antennas, nothing)),
    length(plan.shape),
)

# The station names a layout over `n` stations of `geom` addresses.
function _antenna_labels(geom::DataGeometry, n::Int)
    isempty(geom.antennas) && throw(
        ArgumentError(
            "a solution's geometry must name its antennas; build it with `DataGeometry(ps)` " *
                "or pass `antennas`"
        )
    )
    length(geom.antennas) == n || throw(
        DimensionMismatch("the layout covers $n antennas but the geometry names $(length(geom.antennas))")
    )
    return geom.antennas
end

# ── Applying ─────────────────────────────────────────────────────────────────

# Components over one station set, planned together: `antennas` indexes
# `geom.antennas`.
struct _EvalGroup{M <: GainModel, L <: ParameterLayout, V <: AbstractVector}
    model::M
    layout::L
    θ::V
    antennas::Vector{Int}
end

# What a solution applies: its geometry and one planned group per station set,
# in component order.
struct _AppliedSolution{G <: DataGeometry}
    geom::G
    groups::Vector{_EvalGroup}
end

function _applied(sol::CalibrationSolution)
    isempty(sol.components) && throw(
        ArgumentError("the solution holds no components, so applying it would change nothing")
    )
    geom = sol.geom
    keys_ = Vector{String}[]
    members = Vector{ComponentFit}[]
    for c in sol.components
        k = String.(collect(lookup(c.params, AntennaName)))
        i = findfirst(==(k), keys_)
        if isnothing(i)
            push!(keys_, k)
            push!(members, ComponentFit[c])
        else
            push!(members[i], c)
        end
    end
    groups = _EvalGroup[_eval_group(antennas, cs, geom) for (antennas, cs) in zip(keys_, members)]
    return _AppliedSolution(geom, groups)
end

function _eval_group(antennas::Vector{String}, cs::Vector{ComponentFit}, geom::DataGeometry)
    idx = map(antennas) do s
        i = findfirst(==(s), geom.antennas)
        isnothing(i) && throw(
            ArgumentError("component antenna `$s` is not among the geometry's antennas $(join(geom.antennas, ", "))")
        )
        i
    end
    keys_ = [Symbol(:c, i) for i in eachindex(cs)]
    group(g) = (sel = findall(c -> first(c.path) === g, cs); NamedTuple{Tuple(keys_[sel])}(Tuple(cs[i].component for i in sel)))
    model = GainModel(; phase = group(:phase), logamp = group(:logamp))
    layout = plan_parameters(model, length(antennas), geom; require_nonempty = false)
    T = mapreduce(c -> eltype(c.params), promote_type, cs)
    θ = zeros(T, layout.nθ)
    for (c, key) in zip(cs, keys_)
        g = first(c.path)
        plan = getproperty(getproperty(layout.plantree, g), key)
        axnode = getproperty(getproperty(layout.axes, g), key)
        _check_params_dims(DimensionalData.dims(c.params), _params_dims(plan, axnode, geom, antennas), _label(c))
        copyto!(view(θ, plan.range), parent(c.params))
    end
    return _EvalGroup(model, layout, θ, idx)
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

_nant(sol::_AppliedSolution) = length(sol.geom.antennas)

# The product of every group's gains over an `nchan × ntime` window, each
# group's placed on its own stations; a station a group does not carry takes
# unit gain from it. `args`/`kw` select the window as `evaluate_gains` does.
function _product_gains(sol::_AppliedSolution, nchan::Int, ntime::Int, args...; kw...)
    T = mapreduce(g -> float(eltype(g.θ)), promote_type, sol.groups)
    g = ones(Complex{T}, nchan, ntime, _nant(sol), sol.geom.nfeed)
    for grp in sol.groups
        view(g, :, :, grp.antennas, :) .*= evaluate_gains(grp.layout, grp.θ, args...; kw...)
    end
    return g
end

"""
    gains(sol::CalibrationSolution; Frequency, Ti, AntennaName, Feed) -> DimArray

The complex antenna gains of `sol` — a whole solution or any selection of it
(`sol[:phase]`, `sol[:phase, :mbd]`, `filter(pred, sol)`) — as a
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
    nfeed = sol.geom.nfeed
    nchan = nchannels(sol.geom)
    ntime = ntimes(sol.geom)
    d = (Frequency(sol.geom.channel_freqs), Ti(sol.geom.times), AntennaName(sol.geom.antennas), Feed(1:nfeed))
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

_frequency_segment_lookup(plan::ComponentPlan, geom::DataGeometry, n::Integer = plan.shape.Frequency) =
    _segment_lookup(geom.channel_freqs, segment_groups(plan.fseg_id, n))
_time_segment_lookup(plan::ComponentPlan, geom::DataGeometry, n::Integer = plan.shape.Ti) =
    _segment_lookup(geom.times, segment_groups(plan.tseg_id, n))

# The dimension for one axis of a component's parameters, from its role:
# segment axes over the geometry, `AntennaName` over station names (a signature
# group's over just its `antennas`), `param`/`node` positional.
function _role_dim(role::Symbol, n::Int, geom::DataGeometry, names, plan::ComponentPlan; antennas = nothing)
    if role === :Frequency
        return Frequency(_frequency_segment_lookup(plan, geom, n))
    elseif role === :Ti
        return Ti(_time_segment_lookup(plan, geom, n))
    elseif role === :AntennaName
        return AntennaName(isnothing(antennas) ? collect(names) : collect(names)[antennas])
    elseif role === :Feed
        return Feed(_block_feeds(plan.tying, geom.nfeed))
    else
        return Dim{role}(1:n)
    end
end

# ── Geometry from a ProcessingSet ────────────────────────────────────────────────────

# Two timestamps within `_epoch_atol` of each other are the same instant, and
# two channel frequencies within `_freq_atol` the same channel. An axis is built
# by matching each new value against the canonical ones already seen, not by
# rounding to a grid: a grid still splits two values that straddle a bucket
# edge, which is the case this exists to remove. The canonical value is a real
# observed value, so the axis keeps feeding physical `t - t0` and `f - f0`
# arithmetic. `_time_indices` and `_channel_indices` match the geometry with the
# same tolerances, so the constructor and its readers agree.
_freq_atol(f::Real) = 1.0e-9 * abs(Float64(f))

function _find_canonical(canon::AbstractVector{Float64}, x::Float64, atol)
    i = searchsortedfirst(canon, x - atol)
    return (i <= length(canon) && abs(canon[i] - x) <= atol) ? i : nothing
end

function _canonical!(canon::Vector{Float64}, x::Float64, atol)
    i = _find_canonical(canon, x, atol)
    isnothing(i) || return canon[i]
    insert!(canon, searchsortedfirst(canon, x), x)
    return x
end

# Read-only counterpart for a second pass over times already canonicalized.
function _canonical_time(canon::AbstractVector{Float64}, t::Float64)
    i = _find_canonical(canon, t, _epoch_atol(t))
    i === nothing && throw(
        ArgumentError("time $t s was not canonicalized on the first pass")
    )
    return canon[i]
end

"""
    DataGeometry(ps::XRadio.ProcessingSet; f0 = nothing, t0 = nothing) -> DataGeometry

The geometry a solve over `ps` runs on: the sorted distinct channel
frequencies (Hz) and time samples (seconds) of every Measurement Set, each
channel's spectral window from `XRadio.spectralwindow` and width from
the frequency coordinate's `channel_width`, each time's
scan from the `scan_name` coordinate, and the stations, every antenna named
in an antenna dataset, in the order first seen. The feed count is each
Measurement Set's receptor count, `size(XRadio.polarization_types(ms), 1)`.
`f0` defaults to the mean channel frequency, `t0` to the first time.

A Measurement Set may hold several scans. Sub-arrays observing different
scans at the same timestamps are supported when their station sets are
disjoint and they share the whole scan window, as for
`DataGeometry`. Throws when a Measurement Set states no `scan_name` or `channel_width`,
when a frequency falls in two spectral windows or has two widths, when two scans share a
timestamp and a station, when a scan overlaps another over part of its
span, and when the Measurement Sets' receptor counts differ.
"""
function DataGeometry(ps::XRadio.ProcessingSet; f0 = nothing, t0 = nothing)
    isempty(ps) && throw(ArgumentError("the processing set holds no Measurement Sets"))
    pieces = [
        (;
                freqs = Float64.(XRadio.frequencies(ms)), spw = XRadio.spectralwindow(ms),
                width = _channel_width(ms),
                times = Float64.(XRadio.times(ms)), scans = _scan_labels(ms),
                active = Set{String}(Iterators.flatten(XRadio.baselines(ms))),
            ) for ms in ps
    ]
    antennas = String[]
    for ms in ps, name in XRadio.antennas(ms)
        String(name) in antennas || push!(antennas, String(name))
    end
    return _span_geometry(pieces, antennas, _feed_count(ps); f0, t0)
end

function _feed_count(ps::XRadio.ProcessingSet)
    counts = Dict(name => size(XRadio.polarization_types(ms), 1) for (name, ms) in pairs(ps))
    length(unique(values(counts))) == 1 || throw(
        ArgumentError(
            "the Measurement Sets hold different numbers of receptors per antenna, " *
                join(("$name: $n" for (name, n) in sort!(collect(counts); by = first)), ", ") *
                "; a solve runs on one feed count"
        )
    )
    return first(values(counts))
end

function _channel_width(ms::XRadio.MeasurementSet)
    meta = DimensionalData.metadata(DimensionalData.lookup(ms[:visibility], XRadio.Frequency))
    (meta isa AbstractDict && haskey(meta, :channel_width)) || throw(
        ArgumentError(
            "a Measurement Set's frequency coordinate states no `channel_width`; the " *
                "geometry records each channel's width"
        )
    )
    return abs(Float64(XRadio.value(meta[:channel_width])))
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
# window `spw` and time samples `times` labeled per sample by `scans`, observed
# by the stations in `active`.
function _span_geometry(pieces, antennas, nfeed; f0, t0)
    freq_spw = Dict{Float64, String}()
    freq_width = Dict{Float64, Float64}()
    time_scan = Dict{Float64, String}()
    time_antennas = Dict{Float64, Set{String}}()
    time_canon = Float64[]
    freq_canon = Float64[]
    for piece in pieces
        for f in piece.freqs
            f = Float64(f)
            f = _canonical!(freq_canon, f, _freq_atol(f))
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
            w = get!(freq_width, f, piece.width)
            w == piece.width || throw(
                ArgumentError(
                    "channel frequency $f Hz has widths $w Hz and $(piece.width) Hz in " *
                        "different Measurement Sets of spectral window '$(piece.spw)'"
                )
            )
        end
        for (t, scan) in zip(piece.times, piece.scans)
            tk = _canonical!(time_canon, t, _epoch_atol(t))
            prev = get(time_scan, tk, nothing)
            if prev === nothing
                time_scan[tk] = scan
                time_antennas[tk] = copy(piece.active)
            elseif prev == scan
                union!(time_antennas[tk], piece.active)
            else
                shared = intersect(time_antennas[tk], piece.active)
                isempty(shared) || throw(
                    ArgumentError(
                        "antenna(s) $(join(sort!(collect(shared)), ", ")) are in " *
                            "both scan '$prev' and scan '$scan' at time $tk s. A " *
                            "antenna cannot be in two scans at one instant, and a single " *
                            "concatenated time axis cannot dense-rank the timestamp to one scan. " *
                            "Check for overlapping scan windows or inconsistent scan names."
                    )
                )
                union!(time_antennas[tk], piece.active)
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
        times, channel_freqs = freqs, scan_of_time, spw_of_chan,
        channel_widths = [freq_width[f] for f in freqs], t0 = t0v, f0 = f0v,
        scan_names = _unique_in_order(scan_labels), spw_names = _unique_in_order(spw_labels),
        antennas, nfeed,
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
- `antennas` — for each baseline, in `baseline_id` order, the pair of indices
  into `geom.antennas` of its two antennas.

θ is addressed by POSITION — a component's leaf indexed `(param, node, fseg_id,
tseg_id, ant)` over global-length segment-id tables — while a Measurement Set's
coordinates are PHYSICAL (Hz, seconds, antenna names, correlation labels), so
this join cannot be recovered from the data alone.

`GeometryWindow(geom, ms)` matches channels by frequency (`isapprox`, rtol
1e-9), times by `_epoch_atol` on absolute seconds, and antennas by name, and
throws on any that `geom` does not hold. The three-argument form addresses
channels and times only, which is all evaluating gains needs; its station
map is empty.
"""
struct GeometryWindow
    geom::DataGeometry
    chan_idx::Vector{Int}
    ti_idx::Vector{Int}
    antennas::Vector{Tuple{Int, Int}}
end

GeometryWindow(geom::DataGeometry, chan_idx, ti_idx) =
    GeometryWindow(geom, chan_idx, ti_idx, Tuple{Int, Int}[])

function GeometryWindow(geom::DataGeometry, ms::XRadio.MeasurementSet)
    isempty(geom.antennas) && throw(
        ArgumentError("the geometry names no antennas; build it with `DataGeometry(ps)`")
    )
    slot = Dict(n => i for (i, n) in pairs(geom.antennas))
    antenna(n) = get(slot, String(n)) do
        throw(
            ArgumentError(
                "baseline antenna `$n` is not among the geometry's antennas, " *
                    join(geom.antennas, ", ")
            )
        )
    end
    antennas = [(antenna(a), antenna(b)) for (a, b) in XRadio.baselines(ms)]
    return GeometryWindow(
        geom, _channel_indices(geom, XRadio.frequencies(ms)), _time_indices(geom, XRadio.times(ms)), antennas,
    )
end

_channel_indices(geom::DataGeometry, fs) = [_channel_index(geom, Float64(f)) for f in fs]

function _channel_index(geom::DataGeometry, f)
    j = findfirst(g -> abs(g - f) <= _freq_atol(f), geom.channel_freqs)
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
segment it belongs to (see [`evaluate_gains`](@ref)); `time_span` is the
interval the selected times integrate over, one number for all of them or one
per time, so a sample straddling a segment boundary is rejected. Labeled like [`gains`](@ref)`(sol)`.
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
            AntennaName(sol.geom.antennas), Feed(1:sol.geom.nfeed),
        ),
    )
end
