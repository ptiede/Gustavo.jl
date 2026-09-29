# ── CalibrationSolution: container, geometry, apply, serialization ───────────
#
# A `CalibrationSolution` is the composition of every pipeline step's own
# finished `StepSolution` — each step solved its own private `(model, layout,
# θ)`, never merged with another step's (see `StepSolution`), and the
# solution's gains are the ELEMENTWISE PRODUCT of every step's own gains
# (equivalently a log-space sum, since `evaluate_gains` already returns
# `exp(Σ logamp)·cis(Σ phase)` and a product of `cis`/`exp` factors is a sum of
# their arguments). It is the hand-off object between a solver (e.g.
# `Gustavo.fit`) and the data: `Gustavo.calibrate(sol, ps)` divides the data by
# each step's gains, and `save_solution`/`load_solution`
# round-trip it through the `Serialization` stdlib.

using Serialization: serialize, deserialize
using Statistics: mean
using DimensionalData: lookup, Ti, DimArray, Dim, Dimensions
using DimensionalData.Lookups: Sampled, Explicit, Intervals, Center
using ..UVData: Frequency, Polarization, BaselineID, Ant, Feed

"""
    StepSolution(name, model, layout, θ, info)

One finished pipeline step's own solved gain model: `model`/`layout` compiled
from that step's own `model_components` alone — never merged with another
step's — and `θ` its own solved parameter vector, plus `info`, that step's own
solver diagnostics. A [`CalibrationSolution`](@ref) is the ordered
`steps::Vector{StepSolution}` a pipeline run produced, one per solve step, in
run order; gains compose multiplicatively across them ([`gains`](@ref),
`calibrate`), and `name` (the step's `provides(step)`
capability, e.g. `:fringe`/`:bandpass`/`:adhoc`) is how a later
a user's [`stage_info`](@ref) or `sol[name]` looks a step up.

`θ` keeps whatever array type it is given — a labelled `DimArray` as readily as
a `Vector` — subject to two requirements `layout` imposes: `length(θ) ==
layout.nθ`, and 1-based indexing, since `layout` addresses θ by absolute
position. Both are checked on construction. A step stores a copy, never an
alias.

`selection` is the component-tree path the step is restricted to — `()` (the
whole model) unless the step came out of a component selection
(`sol[step, path...]`, see `getindex`). A selected step evaluates only the
selected subtree; every other component contributes unit gain.
"""
struct StepSolution{M <: GainModel, L <: ParameterLayout, V <: AbstractVector{<:Real}}
    name::Symbol
    model::M
    layout::L
    θ::V
    info::NamedTuple
    selection::Tuple{Vararg{Symbol}}

    function StepSolution{M, L, V}(name, model, layout, θ, info, selection = ()) where {M, L, V}
        # `layout` addresses θ by absolute 1-based position (`ComponentPlan.range`)
        # and the term kernels read those blocks under `@inbounds`, so an array
        # with other axes would read out of bounds silently rather than throw.
        Base.require_one_based_indexing(θ)
        length(θ) == layout.nθ || throw(
            DimensionMismatch(
                "StepSolution($(repr(name))): θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)"
            )
        )
        return new{M, L, V}(name, model, layout, θ, info, selection)
    end
end

# `copy(θ)`, not an alias: the fused output tail builds a step solution per
# scan group from the run's working θ while sibling groups are still writing
# their own slots, and each group must correct against its own snapshot. A
# selection (`_select_component`) calls the inner constructor directly and
# deliberately shares θ with the step it narrows.
StepSolution(name::Symbol, model::GainModel, layout::ParameterLayout, θ::AbstractVector, info::NamedTuple = NamedTuple()) =
    StepSolution{typeof(model), typeof(layout), typeof(θ)}(name, model, layout, copy(θ), info)

# Fieldwise equality, so a selection of a solution compares as the same
# solution content. `layout` (like `geom` below) has no value equality of its
# own and compares by identity, so equality is meaningful among selections
# and snapshots of one solve, not across serialization round-trips.
Base.:(==)(a::StepSolution, b::StepSolution) =
    a.name === b.name && a.model == b.model && a.layout == b.layout &&
    a.θ == b.θ && a.info == b.info && a.selection == b.selection

"""
    CalibrationSolution(steps, geom, info = NamedTuple(); sequence = (), gauge = nothing)
    CalibrationSolution(model, layout, geom, θ, info = NamedTuple(); name = :solution,
                        sequence = (), gauge = nothing)

A solved calibration: the composition of every pipeline step's own finished
[`StepSolution`](@ref). The second form is the common single-model
convenience — a hand-built or extracted solution with exactly one step, named
`name`. `geom::DataGeometry` is the grid every step's own layout was planned
over, and `info` a NamedTuple of solution-level diagnostics (per-scan SNR,
residuals, …) — distinct from each step's own `info`.

Component names are local to each step's own model and may repeat across
steps (e.g. a `bandpass` step and a `fringe` step can both carry an `atmos`
component) — a component selection (`sol[step, path...]`, see `getindex`)
always takes the step explicitly rather than searching for a name across
`steps`.

`sequence` records the pipeline the solve ran, as a tuple, in order: its solve
steps and corrections. `calibrate(sol, ps)` replays it in that order, each
solve step as its own gains, and `fit(sol.sequence, ps; sol.gauge)` repeats
the run. `gauge` is the gauge
the solve was given. Both are empty for a hand-built solution; an element that
did not survive serialization is `missing`.
"""
struct CalibrationSolution{G <: DataGeometry}
    steps::Vector{StepSolution}
    geom::G
    info::NamedTuple
    sequence::Tuple
    gauge::Any
end

function CalibrationSolution(
        steps::AbstractVector{<:StepSolution}, geom::DataGeometry, info::NamedTuple = NamedTuple();
        sequence = (), gauge = nothing,
    )
    isempty(steps) && throw(ArgumentError("CalibrationSolution: at least one step is required."))
    return CalibrationSolution(collect(StepSolution, steps), geom, info, Tuple(sequence), gauge)
end

function CalibrationSolution(
        model::GainModel, layout::ParameterLayout, geom::DataGeometry,
        θ::AbstractVector, info::NamedTuple = NamedTuple();
        name::Symbol = :solution, sequence = (), gauge = nothing,
    )
    return CalibrationSolution(
        [StepSolution(name, model, layout, θ, info)], geom, info; sequence, gauge,
    )
end

"""
    recorded_transforms(sol::CalibrationSolution) -> Vector

The corrections in `sol.sequence`, in order, with any `missing` element kept
so that a caller replaying them can refuse.
"""
function recorded_transforms end

Base.:(==)(a::CalibrationSolution, b::CalibrationSolution) =
    a.steps == b.steps && a.geom == b.geom && a.info == b.info &&
    a.sequence == b.sequence && a.gauge == b.gauge

# `nant` is a run-wide constant every step's own layout was planned with
# (`plan_parameters(step_model, nant, geom)`), so any step's own layout reports
# the same value.
_nant(sol::CalibrationSolution) = sol.steps[1].layout.nant

# A selected step shows its component path beside the stage name.
_step_label(s::StepSolution) =
    isempty(s.selection) ? string(s.name) : string(s.name, "[", join(s.selection, '.'), "]")

function Base.show(io::IO, ::MIME"text/plain", sol::CalibrationSolution)
    println(io, "CalibrationSolution")
    println(io, "  Steps     : ", join(map(_step_label, sol.steps), ", "))
    println(
        io, "  Grid      : ", _nant(sol), " antennas × ", nchannels(sol.geom), " channels × ",
        ntimes(sol.geom), " times",
    )
    println(io, "  Parameters: ", sum(s.layout.nθ for s in sol.steps))
    println(io, "  Components: ", join(component_names(sol), ", "))
    print(
        io, "  Pipeline  : ", length(sol.sequence), " element(s), gauge ",
        sol.gauge === nothing ? "not recorded" : sol.gauge,
    )
    return io
end

Base.show(io::IO, sol::CalibrationSolution) = print(
    io, "CalibrationSolution(", length(sol.steps), " step(s), ",
    sum(s.layout.nθ for s in sol.steps), " parameters)",
)

# ── Per-stage views: snapshots of the solution as of each pipeline stage ─────

"""
    getindex(sol::CalibrationSolution, index) -> CalibrationSolution
    getindex(sol::CalibrationSolution, step, path::Symbol...) -> CalibrationSolution

Select a subtree of `sol` — steps, or one step's model components — as a
solution in its own right.

The first index selects steps, in run order. A `Symbol` names one step (see
`keys`; a duplicated stage name is an error — index positionally); anything
`sol.steps` accepts selects positionally — `sol[2]` the second step alone,
`sol[1:2]` the first two, `sol[[1, 3]]` the first and third, `sol[:]` every
one.

Further `Symbol`s descend into that one step's component tree, starting at
`:phase` or `:logamp`: `sol[:fringe, :phase]` is the fringe step's phase
components alone, `sol[:fringe, :phase, :mbd]` one named component,
`sol[:fringe, :phase, :sbd, :delay]` a wrapper's nested leaf, and a
station-heterogeneous name descends into its signature groups
(`sol[:bandpass, :phase, :bandpass, :g1]`). Indexing a component selection
descends further into its subtree (`sol[:fringe, :phase][:fringe, :mbd]`).
The selection remembers the tree path — θ is never copied or zeroed.

Gains compose multiplicatively across steps and across components, so a
selection's gain is exactly the product of the selected parts' own gains — a
part left out contributes no gain at all. A leading run `sol[1:i]` is thus the
solution AS OF step `i`, and a component selection's gain is that component's
own contribution: apply it, plot it, or difference it against another.
Geometry, `info` and both provenance chains carry over unchanged, so every
selection is a valid solution for `calibrate` and `ApplySolution`,
and `sol[:]` reproduces `sol`.

Selecting no step at all is an error: a solution has at least one.
"""
function Base.getindex(sol::CalibrationSolution, index)
    return step_solution(sol, index)
end

function Base.getindex(sol::CalibrationSolution, step::Union{Symbol, Integer}, path1::Symbol, path::Symbol...)
    ssel = step_solution(sol, step)
    s = _select_component(ssel.steps[1], (path1, path...))
    return CalibrationSolution(
        [s], sol.geom, sol.info;
        sequence = sol.sequence, gauge = sol.gauge,
    )
end

Base.firstindex(sol::CalibrationSolution) = firstindex(sol.steps)
Base.lastindex(sol::CalibrationSolution) = lastindex(sol.steps)

"""
    length(sol::CalibrationSolution), iterate, keys, haskey, eachindex

`CalibrationSolution` is a container of its pipeline steps: `length` counts
them, `eachindex` gives their positions, `keys` their stage names in run
order, `haskey` membership by name, and iteration yields `sol[i]` — each step
as a single-step solution, so `collect(sol) == [sol[i] for i in
eachindex(sol)]`.
"""
Base.length(sol::CalibrationSolution) = length(sol.steps)
Base.eachindex(sol::CalibrationSolution) = eachindex(sol.steps)
Base.keys(sol::CalibrationSolution) = Symbol[s.name for s in sol.steps]
Base.haskey(sol::CalibrationSolution, name::Symbol) = any(s -> s.name === name, sol.steps)
Base.iterate(sol::CalibrationSolution, i::Int = 1) =
    i > length(sol.steps) ? nothing : (sol[i], i + 1)
Base.eltype(::Type{S}) where {S <: CalibrationSolution} = S

# Narrow one step to a component subtree: extend its selection by `path`,
# validated against the layout's plantree. Shares θ with the step it narrows
# (inner-constructor call — see the outer constructor's copy note).
_select_component(s::StepSolution{M, L, V}, path::Tuple{Vararg{Symbol}}) where {M, L, V} =
    StepSolution{M, L, V}(s.name, s.model, s.layout, s.θ, s.info, _extend_selection(s, path))

# The step's selection extended by `path`, validated by walking the plantree:
# a `GroupedComponentPlan` descends into its signature groups; naming an
# absent key, or descending past a single component, errors naming what is
# available at that point.
function _extend_selection(s::StepSolution, path::Tuple{Vararg{Symbol}})
    full = (s.selection..., path...)
    node = s.layout.plantree
    walked = Symbol[]
    for k in full
        children = node isa GroupedComponentPlan ? node.groups : node
        children isa NamedTuple || throw(
            ArgumentError(
                "step $(repr(s.name)): $(join(walked, '.')) is a single component " *
                    "with no subcomponent $(repr(k)) to select."
            )
        )
        haskey(children, k) || throw(
            ArgumentError(
                "step $(repr(s.name)) has no component " *
                    "$(join(vcat(walked, k), '.')); available under " *
                    "$(isempty(walked) ? "the model" : join(walked, '.')): " *
                    "$(join(keys(children), ", "))."
            )
        )
        push!(walked, k)
        node = getproperty(children, k)
    end
    return full
end

# The plantree restricted to a selection path: the full tree when the path is
# empty, otherwise singleton NamedTuples nested along the path, the other
# top-level group left empty. Component `range`s address absolute θ positions,
# so the pruned tree evaluates against the step's full θ — the selected
# components contribute their solved values and every other component
# contributes nothing (a unit gain factor), with no zeroed-θ copy.
_selected_plantree(layout::ParameterLayout, ::Tuple{}) = layout.plantree
function _selected_plantree(layout::ParameterLayout, sel::Tuple{Vararg{Symbol}})
    group = first(sel)
    sub = _prune_node(getproperty(layout.plantree, group), Base.tail(sel))
    return group === :phase ? (; phase = sub, logamp = (;)) : (; phase = (;), logamp = sub)
end

_prune_node(node, ::Tuple{}) = node
_prune_node(node::NamedTuple, path::Tuple{Symbol, Vararg{Symbol}}) = NamedTuple{(first(path),)}(
    (_prune_node(getproperty(node, first(path)), Base.tail(path)),)
)
# Selecting one signature group keeps the station routing: only that group's
# stations carry the component; every other station contributes nothing.
function _prune_node(g::GroupedComponentPlan, path::Tuple{Symbol, Vararg{Symbol}})
    k = first(path)
    gi = findfirst(==(k), keys(g.groups))
    group_of = [go == gi ? 1 : 0 for go in g.group_of]
    local_of = [go == gi ? lo : 0 for (go, lo) in zip(g.group_of, g.local_of)]
    return GroupedComponentPlan(
        NamedTuple{(k,)}((getproperty(g.groups, k),)), [g.stations[gi]], group_of, local_of,
    )
end

# One step's layout, honoring its selection: the plantree pruned to the
# selected subtree. `nθ`, the grid dims, and the flat `plans` stay those of the
# full solve — pruning never re-lays-out θ.
function _selected_layout(s::StepSolution)
    isempty(s.selection) && return s.layout
    lay = s.layout
    return ParameterLayout(
        lay.nθ, lay.nant, lay.ntime, lay.nchan, lay.nphase, lay.plans,
        _selected_plantree(lay, s.selection), lay.axes,
    )
end

"""
    stage_info(sol::CalibrationSolution, name::Symbol) -> NamedTuple

The diagnostics recorded by the step named `name` (detections and per-scan SNR
for the fringe stage, calibrator choice for the bandpass stage, …).
"""
stage_info(sol::CalibrationSolution, name::Symbol) = sol[name].steps[1].info

"""
    component_names(sol::CalibrationSolution) -> Vector{Symbol}

The top-level model component names across every step of `sol` (phase and
log-amplitude groups combined, in step then declaration order, duplicates
dropped) — what `show` lists under `Components:`. A name here can belong to
several steps at once (component names are local to each step's own model);
a step-qualified selection (`sol[step, :phase, name]`) reaches one
unambiguously.
"""
function component_names(sol::CalibrationSolution)
    names = Symbol[]
    for s in sol.steps, k in (keys(s.model.phase)..., keys(s.model.logamp)...)
        k in names || push!(names, k)
    end
    return names
end

# The elementwise product of every step's own gains, each evaluated by
# `evaluate_gains(layout, θ, args...; kw...)`. A step reads as identity gain
# wherever it has no component, so the product across steps is the same total
# gain a single merged model would give, without ever building one.
function _product_gains(sol::CalibrationSolution, args...; kw...)
    s1 = sol.steps[1]
    g = evaluate_gains(_selected_layout(s1), s1.θ, args...; kw...)
    for s in view(sol.steps, 2:length(sol.steps))
        g .*= evaluate_gains(_selected_layout(s), s.θ, args...; kw...)
    end
    return g
end

"""
    gains(sol::CalibrationSolution; Frequency, Ti, Ant, Feed) -> DimArray

The solved complex antenna gains of `sol` — the whole solution, or any
selection of it (`sol[:bandpass]`, `sol[:fringe, :phase, :mbd]`, …) — labelled
for inspection: a `DimArray` over `(Frequency, Ti, Ant, Feed)` — channel
frequencies (Hz), integration times (seconds), antennas (named when `sol.info`
carries `ant_names`, else `1:nant`), and feed. `gain = exp(Σ logamp) · cis(Σ
phase)`, summed over the selection's components, is the same forward map
`calibrate` divides by;
`abs.(gains(sol))` and `angle.(gains(sol))` recover amplitude and phase.

The keywords are the dimension names and accept anything `DimArray` indexing
accepts — integers, ranges, `At`, `Near`, `Where`, intervals — under the
invariant `gains(sol; kw...) == gains(sol)[kw...]`. A `Frequency`/`Ti`
selector restricts the evaluation window (the gains are computed only at the
selected samples, not sliced from the full cube); `Ant`/`Feed` slice the
result.
"""
function gains(sol::CalibrationSolution; kw...)
    nant = _nant(sol)
    nfeed = 2
    nchan = nchannels(sol.geom)
    ntime = ntimes(sol.geom)
    d = (Frequency(sol.geom.channel_freqs), Ti(sol.geom.times), Ant(_ant_labels(sol, nant)), Feed(1:nfeed))
    isempty(kw) && return DimArray(_product_gains(sol), d)
    for k in keys(kw)
        k in (:Frequency, :Ti, :Ant, :Feed) || throw(
            ArgumentError(
                "gains: unknown dimension keyword $(repr(k)); the gain axes are " *
                    "Frequency, Ti, Ant, Feed."
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
    g = _product_gains(sol, ci, ti)
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

# ── parameters: the selection's θ as labelled DimArray leaves ────────────────

"""
    parameters(sol::CalibrationSolution)

What the selection solved: its raw θ, wrapped as labelled `DimArray` leaves.

For a single-component selection (`parameters(sol[:fringe, :phase, :mbd])`)
the result is that component's θ leaf: a `DimArray` over the five axes
`(param, feed, Frequency, Ti, Ant)` — the second is `Feed` when the component
is fit per feed, else the tied `node` axis — with coordinates materialized
from `sol`'s geometry: each `Frequency` and `Ti` segment is an interval from
its lowest to its highest channel frequency (Hz) or sample epoch (seconds),
labeled by its midpoint, so `Frequency(Contains(ν))` or `Ti(Contains(t))`
selects the segment covering `ν` or `t`; then feed/node ids, and antenna names
(from `sol.info.ant_names` when present, else `1:nant`).

A wider selection returns the NamedTuple tree of those leaves, mirroring the
model: `parameters(sol[:fringe, :phase])` the fringe step's phase components,
`parameters(sol[:fringe])` its `(; phase, logamp)` trees, and a multi-step
solution one tree per step, keyed by stage name (a duplicated stage name is
an error — select the step positionally first). A station-heterogeneous
component appears as one leaf per signature group (`g1, g2, …`), each `Ant`
axis labelled with just its own group's stations.

The leaves share data with the selection's θ (no copy): an inspection view,
never stored on the solution and never fed through the solve or AD. There are
no selector keywords — index the returned `DimArray`.
"""
function parameters(sol::CalibrationSolution)
    if length(sol.steps) > 1
        ks = keys(sol)
        allunique(ks) || throw(
            ArgumentError(
                "parameters: stage names $(ks) repeat; select one step positionally " *
                    "(`parameters(sol[i])`)."
            )
        )
        return NamedTuple{Tuple(ks)}(ntuple(i -> parameters(sol[i]), length(ks)))
    end
    s = sol.steps[1]
    ok, plan = _try_descend(s.layout.plantree, s.selection)
    ok || error("parameters: selection $(s.selection) no longer resolves in the layout")
    _, axnode = _try_descend(s.layout.axes, s.selection)
    name = isempty(s.selection) ? s.name : last(s.selection)
    return _parameters_node(plan, axnode, sol, s, name)
end

# Recursion mirrors the plantree; the axes tree runs alongside it (its leaves
# are `(; dims, roles[, stations])` records, so the plan node drives dispatch).
_parameters_node(nt::NamedTuple, axnode, sol, s, name) = NamedTuple{keys(nt)}(
    ntuple(i -> _parameters_node(nt[i], axnode[i], sol, s, keys(nt)[i]), length(nt))
)
_parameters_node(plan::ComponentPlan, axnode, sol, s, name) =
    _leaf_dimarray(plan, axnode, sol, s, name)
_parameters_node(g::GroupedComponentPlan, axnode, sol, s, name) = NamedTuple{keys(g.groups)}(
    ntuple(i -> _leaf_dimarray(g.groups[i], axnode[i], sol, s, keys(g.groups)[i]), length(g.groups))
)

function _leaf_dimarray(plan::ComponentPlan, axnode, sol::CalibrationSolution, s::StepSolution, name::Symbol)
    stations = hasproperty(axnode, :stations) ? axnode.stations : nothing
    raw = _component_leaf(plan, s.θ)
    dims = ntuple(d -> _role_dim(axnode.roles[d], size(raw, d), sol, plan; stations), ndims(raw))
    return DimArray(raw, dims; name)
end

# Attempt to descend `tree` (a step's own `layout.plantree` or `layout.axes`)
# by `path`; `(false, nothing)` without throwing when a name is absent along
# the way. A `GroupedComponentPlan` descends into its signature groups, so a
# path may address one group's leaf directly (`:phase, :bandpass, :g1`).
function _try_descend(tree, path::Tuple{Vararg{Symbol}})
    node = tree
    for s in path
        children = node isa GroupedComponentPlan ? node.groups : node
        (children isa NamedTuple && haskey(children, s)) || return false, nothing
        node = getproperty(children, s)
    end
    return true, node
end

# The DimensionalData dimension for one leaf axis, from its role and the
# solution's geometry. A segment axis spans each segment's extent, from its
# lowest to its highest channel or sample, labeled by the midpoint; the
# antenna axis takes station names when the solution carries them — for a
# signature group's leaf, the names of just the group's `stations`; `:param`
# and `:node` are positional id spaces with no physical coordinate.
# Station labels for an axis of length `n`: the solution's recorded names when
# they cover it, else positional ids.
function _ant_labels(sol::CalibrationSolution, n::Int)
    an = hasproperty(sol.info, :ant_names) ? sol.info.ant_names : nothing
    return an !== nothing && length(an) == n ? collect(an) : (1:n)
end

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

# The lookups of a plan's `n` frequency segments (over its channels) and time
# segments (over its samples).
_frequency_segment_lookup(plan::ComponentPlan, geom::DataGeometry, n::Integer = plan.shape[3]) =
    _segment_lookup(geom.channel_freqs, segment_groups(plan.fseg_id, n))
_time_segment_lookup(plan::ComponentPlan, geom::DataGeometry, n::Integer = plan.shape[4]) =
    _segment_lookup(geom.times, segment_groups(plan.tseg_id, n))

function _role_dim(role::Symbol, n::Int, sol::CalibrationSolution, plan::ComponentPlan; stations = nothing)
    if role === :Frequency
        return Frequency(_frequency_segment_lookup(plan, sol.geom, n))
    elseif role === :Ti
        return Ti(_time_segment_lookup(plan, sol.geom, n))
    elseif role === :Ant
        stations === nothing && return Ant(_ant_labels(sol, n))
        an = hasproperty(sol.info, :ant_names) ? sol.info.ant_names : nothing
        ants = an !== nothing && all(i -> 1 <= i <= length(an), stations) ?
            collect(an)[stations] : collect(stations)
        return Ant(ants)
    elseif role === :Feed
        return Feed(1:n)
    else
        return Dim{role}(1:n)
    end
end

function step_solution(sol::CalibrationSolution, index)
    stp = sol.steps[index]
    stpout = stp isa AbstractVector ? stp : [stp]
    return CalibrationSolution(
        stpout, sol.geom, sol.info;
        sequence = sol.sequence, gauge = sol.gauge,
    )
end

function step_solution(sol::CalibrationSolution, name::Symbol)
    idx = findall(s -> s.name === name, sol.steps)
    isempty(idx) && throw(
        ArgumentError("solution has no stage $(repr(name)); recorded stages: $(keys(sol)).")
    )
    length(idx) == 1 || throw(
        ArgumentError(
            "stage $(repr(name)) is recorded $(length(idx)) times (positions $(idx)); " *
                "select it positionally."
        )
    )
    return step_solution(sol, idx[1])
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
`win.ti_idx` index `win.geom`, which may be `sol.geom` itself or the geometry of
other data. On a foreign geometry each sample is placed in the solve segment it
belongs to (see [`evaluate_gains`](@ref)); `time_span[k]` is the interval the
`k`-th selected time integrates over, so a sample straddling a segment boundary
is rejected. Labelled like [`gains`](@ref)`(sol)`.
"""
function gains(sol::CalibrationSolution, win::GeometryWindow; time_span = nothing)
    g = win.geom === sol.geom ? _product_gains(sol, win.chan_idx, win.ti_idx) :
        _product_gains(
            sol, sol.geom, win.geom; chan_idx = win.chan_idx, ti_idx = win.ti_idx, time_span,
        )
    nant = _nant(sol)
    return DimArray(
        g, (
            Frequency(win.geom.channel_freqs[win.chan_idx]), Ti(win.geom.times[win.ti_idx]),
            Ant(_ant_labels(sol, nant)), Feed(1:2),
        ),
    )
end

# ── Serialization ────────────────────────────────────────────────────────────

"""
    save_solution(path, sol::CalibrationSolution)

Serialize `sol` to `path` via the `Serialization` stdlib inside a versioned
wrapper NamedTuple. The current version is 8 (the solution records its
pipeline as `sequence` and its `gauge`); earlier versions are refused on load.
A pipeline element that fails to serialize, such as a closure, is recorded as
`missing` with a warning rather than failing the save.
"""
function save_solution(path::AbstractString, sol::CalibrationSolution)
    wrapper = (;
        version = 8, sol.steps, sol.geom, sol.info,
        sequence = Tuple(_serializable_elements(sol.sequence)), sol.gauge,
    )
    serialize(path, wrapper)
    return path
end

function _serializable_elements(xs)
    out = Any[]
    for x in xs
        ok = try
            serialize(IOBuffer(), x)
            true
        catch
            false
        end
        if ok
            push!(out, x)
        else
            @warn "save_solution: pipeline element $(typeof(x)) is not serializable — recorded as `missing`."
            push!(out, missing)
        end
    end
    return out
end

"""
    load_solution(path) -> CalibrationSolution

Inverse of [`save_solution`](@ref). Only current-format (version 8) files are
supported; files from an earlier Gustavo used a different solution shape and
are refused — re-solve to produce a current-format solution.
"""
function load_solution(path::AbstractString)
    w = deserialize(path)
    w.version == 8 || error(
        "load_solution: unsupported version $(w.version) — saved by an incompatible " *
            "Gustavo (the solution shape changed); re-solve to produce a current file.",
    )
    return CalibrationSolution(w.steps, w.geom, w.info; w.sequence, w.gauge)
end
