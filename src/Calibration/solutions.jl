# ── CalibrationSolution: container, geometry, apply, serialization ───────────
#
# A `CalibrationSolution` is the composition of every pipeline step's own
# finished `StepSolution` — each step solved its own private `(model, layout,
# θ)`, never merged with another step's (see `StepSolution`), and the
# solution's gains are the ELEMENTWISE PRODUCT of every step's own gains
# (equivalently a log-space sum, since `evaluate_gains` already returns
# `exp(Σ logamp)·cis(Σ phase)` and a product of `cis`/`exp` factors is a sum of
# their arguments). It is the hand-off object between a solver (e.g.
# `Gustavo.fit`) and the data: `apply_calibration(uvset, sol)` divides every
# leaf's visibilities by the composed gains, and `save_solution`/`load_solution`
# round-trip it through the `Serialization` stdlib.

using Serialization: serialize, deserialize
using Statistics: mean
using DimensionalData: lookup, Ti, DimArray, Dim, Dimensions
using ..UVData: Frequency, Pol, Baseline, Ant, Feed

"""
    StepSolution(name, model, layout, θ, info)

One finished pipeline step's own solved gain model: `model`/`layout` compiled
from that step's own `model_components` alone — never merged with another
step's — and `θ` its own solved parameter vector, plus `info`, that step's own
solver diagnostics. A [`CalibrationSolution`](@ref) is the ordered
`steps::Vector{StepSolution}` a pipeline run produced, one per solve step, in
run order; gains compose multiplicatively across them ([`gains`](@ref),
[`apply_calibration`](@ref)), and `name` (the step's `provides(step)`
capability, e.g. `:fringe`/`:bandpass`/`:refine`/`:adhoc`) is how a later
step's `fit_selection` or a user's [`stage_info`](@ref) or `sol[name]`
looks a step up.

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
struct StepSolution{M <: StationGainModel, L <: ParameterLayout, V <: AbstractVector{<:Real}}
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
StepSolution(name::Symbol, model::StationGainModel, layout::ParameterLayout, θ::AbstractVector, info::NamedTuple = NamedTuple()) =
    StepSolution{typeof(model), typeof(layout), typeof(θ)}(name, model, layout, copy(θ), info)

# Fieldwise equality, so a selection of a solution compares as the same
# solution content. `layout` (like `geom` below) has no value equality of its
# own and compares by identity, so equality is meaningful among selections
# and snapshots of one solve, not across serialization round-trips.
Base.:(==)(a::StepSolution, b::StepSolution) =
    a.name === b.name && a.model == b.model && a.layout == b.layout &&
    a.θ == b.θ && a.info == b.info && a.selection == b.selection

"""
    CalibrationSolution(steps, geom, info = NamedTuple();
                        transforms = (), postcal = (), pipeline = nothing)
    CalibrationSolution(model, layout, geom, θ, info = NamedTuple(); name = :solution,
                        transforms = (), postcal = (), pipeline = nothing)

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

`transforms` records the data-transform chain the solve materialized its scans
through (precal, weight scaling, caller hooks), so diagnostics can replay it and
the standalone apply path can reproduce `transforms ∘ solution`. `postcal`
records the OUTPUT-chain calibration steps (a-priori amplitude) applied after
the gains and before any reductions, so the standalone apply reproduces them
without re-passing their inputs. Both default empty.

`pipeline` records the `CalibrationPipeline` the solve ran — step order, model
specs, execution config, and gauge — so a run is reproducible from its output
(`fit(sol.pipeline, uvset)`). It is provenance only: applying the solution
never reads it, and it does not participate in `==`. `nothing` for a
hand-built solution; `missing` when a recorded pipeline did not survive
serialization.
"""
struct CalibrationSolution{G <: DataGeometry, T, P}
    steps::Vector{StepSolution}
    geom::G
    info::NamedTuple
    transforms::Vector{T}
    postcal::Vector{P}
    # The pipeline the solve ran (`nothing` for a hand-built solution;
    # `missing` when a recorded pipeline did not survive serialization).
    # Provenance only — it does not participate in applying the solution.
    pipeline::Any
end

function CalibrationSolution(
        steps::AbstractVector{<:StepSolution}, geom::DataGeometry, info::NamedTuple = NamedTuple();
        transforms = (), postcal = (), pipeline = nothing,
    )
    isempty(steps) && throw(ArgumentError("CalibrationSolution: at least one step is required."))
    return CalibrationSolution(
        collect(StepSolution, steps), geom, info,
        UVData._narrow_eltype(transforms), UVData._narrow_eltype(postcal), pipeline,
    )
end

function CalibrationSolution(
        model::StationGainModel, layout::ParameterLayout, geom::DataGeometry,
        θ::AbstractVector, info::NamedTuple = NamedTuple();
        name::Symbol = :solution, transforms = (), postcal = (), pipeline = nothing,
    )
    return CalibrationSolution(
        [StepSolution(name, model, layout, θ, info)], geom, info;
        transforms, postcal, pipeline,
    )
end

Base.:(==)(a::CalibrationSolution, b::CalibrationSolution) =
    a.steps == b.steps && a.geom == b.geom && a.info == b.info &&
    a.transforms == b.transforms && a.postcal == b.postcal

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
        io, "  Provenance: ", length(sol.transforms), " transform(s), ",
        length(sol.postcal), " postcal step(s), pipeline ",
        sol.pipeline === nothing ? "not recorded" :
            sol.pipeline === missing ? "missing" : "recorded",
    )
    return io
end

Base.show(io::IO, sol::CalibrationSolution) = print(
    io, "CalibrationSolution(", length(sol.steps), " step(s), ",
    sum(s.layout.nθ for s in sol.steps), " parameters)",
)

# ── Per-stage views: snapshots of the solution as of each pipeline stage ─────

"""
    component_ranges(layout::ParameterLayout) -> Vector{UnitRange{Int}}

The contiguous θ range owned by each component plan (phase components first,
then log-amplitude, matching `layout.plans`) — each plan's own `range`, the span
`plan_parameters` reserved for its leaf.
"""
component_ranges(layout::ParameterLayout) = [p.range for p in layout.plans]

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

Gains compose multiplicatively across steps AND across components, so a
selection's gain is exactly the product of the selected parts' own gains — a
part left out contributes no gain at all. A leading run `sol[1:i]` is thus the
solution AS OF step `i`, and a component selection's gain is that component's
own contribution: apply it, plot it, or difference it against another.
Geometry, `info` and both provenance chains carry over unchanged, so every
selection is a valid solution for [`apply_calibration`](@ref) and `calibrate`,
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
        transforms = sol.transforms, postcal = sol.postcal, pipeline = sol.pipeline,
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

# The evaluator for one step, honoring its selection: the step's layout with
# the plantree pruned to the selected subtree. `nθ`, the grid dims, and the
# flat `plans` stay those of the full solve — pruning never re-lays-out θ.
function _evaluator(s::StepSolution)
    isempty(s.selection) && return GainEvaluator(s.model, s.layout)
    lay = s.layout
    return GainEvaluator(
        s.model,
        ParameterLayout(
            lay.nθ, lay.nant, lay.ntime, lay.nchan, lay.nphase, lay.plans,
            _selected_plantree(lay, s.selection), lay.template, lay.axes,
        ),
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

# The elementwise product of every step's own gains over the full grid —
# each step's own `evaluate_gains` already reads as identity gain wherever
# that step has no component, so the product across steps is the exact same
# total gain a single merged evaluator would have produced, without ever
# building one.
function _composed_gains(sol::CalibrationSolution)
    s1 = sol.steps[1]
    g = evaluate_gains(_evaluator(s1), s1.θ)
    for s in view(sol.steps, 2:length(sol.steps))
        g .*= evaluate_gains(_evaluator(s), s.θ)
    end
    return g
end

# Windowed counterpart of `_composed_gains(sol)` — see `evaluate_gains`'s
# windowed method.
function _composed_gains(
        sol::CalibrationSolution,
        chan_idx::AbstractVector{<:Integer}, ti_idx::AbstractVector{<:Integer},
    )
    s1 = sol.steps[1]
    g = evaluate_gains(_evaluator(s1), s1.θ, chan_idx, ti_idx)
    for s in view(sol.steps, 2:length(sol.steps))
        g .*= evaluate_gains(_evaluator(s), s.θ, chan_idx, ti_idx)
    end
    return g
end

# Foreign-grid counterpart: each target sample placed in the solve segment it
# belongs to, so the data need not be sampled on the solve grid.
function _composed_gains(
        sol::CalibrationSolution, target::DataGeometry;
        chan_idx::AbstractVector{<:Integer} = Base.OneTo(nchannels(target)),
        ti_idx::AbstractVector{<:Integer} = Base.OneTo(ntimes(target)),
        time_span = nothing,
    )
    s1 = sol.steps[1]
    g = evaluate_gains(
        _evaluator(s1), s1.θ, sol.geom, target; chan_idx, ti_idx, time_span,
    )
    for s in view(sol.steps, 2:length(sol.steps))
        g .*= evaluate_gains(
            _evaluator(s), s.θ, sol.geom, target; chan_idx, ti_idx, time_span,
        )
    end
    return g
end

"""
    gains(sol::CalibrationSolution; Frequency, Ti, Ant, Feed) -> DimArray

The solved complex antenna gains of `sol` — the whole solution, or any
selection of it (`sol[:bandpass]`, `sol[:fringe, :phase, :mbd]`, …) — labelled
for inspection: a `DimArray` over `(Frequency, Ti, Ant, Feed)` — channel
frequencies (Hz), integration times (hours), antennas (named when `sol.info`
carries `ant_names`, else `1:nant`), and feed. `gain = exp(Σ logamp) · cis(Σ
phase)`, summed over the selection's components, is the same forward map
[`apply_calibration`](@ref) divides by (and [`save_solution_hdf5`](@ref)
writes); `abs.(gains(sol))` and `angle.(gains(sol))` recover amplitude and
phase.

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
    isempty(kw) && return DimArray(_composed_gains(sol), d)
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
    # Slice the lookups BEFORE evaluating: the evaluator reads its segment
    # tables under `@inbounds`, so an out-of-range selector must fail here
    # (a plain `BoundsError`), never inside the evaluation.
    fsel = sol.geom.channel_freqs[ci]
    tsel = sol.geom.times[ti]
    g = _composed_gains(sol, ci, ti)
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
from `sol`'s geometry: a `Frequency` segment's centre channel frequency (Hz),
a `Ti` segment's mean epoch (hours), feed/node ids, and antenna names (from
`sol.info.ant_names` when present, else `1:nant`).

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
# solution's geometry. A segment axis takes a representative coordinate per
# segment (a frequency segment's centre, a time segment's mean epoch); the
# antenna axis takes station names when the solution carries them — for a
# signature group's leaf, the names of just the group's `stations`; `:param`
# and `:node` are positional id spaces with no physical coordinate.
# Station labels for an axis of length `n`: the solution's recorded names when
# they cover it, else positional ids.
function _ant_labels(sol::CalibrationSolution, n::Int)
    an = hasproperty(sol.info, :ant_names) ? sol.info.ant_names : nothing
    return an !== nothing && length(an) == n ? collect(an) : (1:n)
end

function _role_dim(role::Symbol, n::Int, sol::CalibrationSolution, plan::ComponentPlan; stations = nothing)
    if role === :Frequency
        groups = segment_groups(plan.fseg_id, n)
        return Frequency([mean(view(sol.geom.channel_freqs, g)) for g in groups])
    elseif role === :Ti
        groups = segment_groups(plan.tseg_id, n)
        return Ti([mean(view(sol.geom.times, g)) for g in groups])
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
        transforms = sol.transforms, postcal = sol.postcal, pipeline = sol.pipeline,
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


# ── Geometry from a UVSet ────────────────────────────────────────────────────

"""
    build_geometry(uvset; f0 = nothing, t0 = nothing) -> DataGeometry

Build the union `DataGeometry` spanning every leaf of `uvset`: the sorted unique
channel frequencies (Hz) of all bands, the sorted unique integration times
(hours) of all scans, with `spw_of_chan` / `scan_of_time` dense-ranked from the
leaves' `spw_name` / `scan_name`. `f0` defaults to the mean channel frequency,
`t0` to the first time. Assumes a single consistent antenna table across the
set (used only to size the solve elsewhere).
"""
function build_geometry(uvset::UVSet; f0 = nothing, t0 = nothing)
    # Collect (freq, spw_name) and (time, scan_name) observations from all leaves.
    freq_spw = Dict{Float64, String}()
    time_scan = Dict{Float64, String}()
    for (_, leaf) in UVData.branches(uvset)
        info = UVData.metadata(leaf)
        fs = channel_freqs(info.freq_setup)
        for f in fs
            fk = Float64(f)
            prev = get(freq_spw, fk, nothing)
            (prev === nothing || prev == info.spw_name) || throw(
                ArgumentError(
                    "build_geometry: channel frequency $fk Hz appears in conflicting spectral " *
                        "windows '$prev' and '$(info.spw_name)' — a single concatenated channel " *
                        "axis cannot dense-rank it to one spw. Partition the set so each " *
                        "frequency belongs to one spw, or rename the spws consistently."
                )
            )
            freq_spw[fk] = info.spw_name
        end
        ts = lookup(leaf[:vis], Ti)
        for t in ts
            tk = Float64(t)
            prev = get(time_scan, tk, nothing)
            (prev === nothing || prev == info.scan_name) || throw(
                ArgumentError(
                    "build_geometry: time $tk h appears in conflicting scans '$prev' and " *
                        "'$(info.scan_name)' — a single concatenated time axis cannot dense-rank " *
                        "it to one scan. Check for overlapping scan windows or inconsistent " *
                        "scan names."
                )
            )
            time_scan[tk] = info.scan_name
        end
    end

    freqs = sort!(collect(keys(freq_spw)))
    times = sort!(collect(keys(time_scan)))

    spw_labels = [freq_spw[f] for f in freqs]
    scan_labels = [time_scan[t] for t in times]

    spw_of_chan, _ = _dense_rank_labels(spw_labels)
    scan_of_time, _ = _dense_rank_labels(scan_labels)

    spw_names = _unique_in_order(spw_labels)
    scan_names = _unique_in_order(scan_labels)

    f0v = isnothing(f0) ? (isempty(freqs) ? 0.0 : mean(freqs)) : Float64(f0)
    t0v = isnothing(t0) ? (isempty(times) ? 0.0 : first(times)) : Float64(t0)

    return DataGeometry(;
        times = times,
        channel_freqs = freqs,
        scan_of_time = scan_of_time,
        spw_of_chan = spw_of_chan,
        t0 = t0v,
        f0 = f0v,
        scan_names = scan_names,
        spw_names = spw_names,
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
    GeometryWindow

A window into a [`DataGeometry`](@ref): everything needed to locate one scan's
channels and times in the solve's index space, with no data attached.

- `geom` — the solve's geometry, the index space the window addresses.
- `chan_idx`, `ti_idx` — GLOBAL indices into `geom.channel_freqs` / `geom.times`
  of the channels and times the window covers.

θ is addressed by POSITION — a component's leaf indexed `(param, node, fseg_id,
tseg_id, ant)` over global-length segment-id tables — while a `DimStack`'s
coordinates are PHYSICAL (Hz, hours), so this join cannot be recovered from the
data alone.
Build one with [`leaf_window`](@ref); pass it alongside the scan's `DimStack`
to anything that needs both.
"""
struct GeometryWindow
    geom::DataGeometry
    chan_idx::Vector{Int}
    ti_idx::Vector{Int}
end

"""
    leaf_window(geom::DataGeometry, leaf) -> GeometryWindow

The [`GeometryWindow`](@ref) addressing the channels and times `leaf` carries,
matched by value against `geom` (frequency by `isapprox` rtol 1e-9, time by atol
1e-9 h). Errors if any leaf sample has no match in the geometry.
"""
leaf_window(geom::DataGeometry, leaf) =
    GeometryWindow(geom, _channel_indices(geom, leaf), _time_indices(geom, leaf))

function _channel_indices(geom::DataGeometry, leaf)
    fs = lookup(leaf[:vis], Frequency)
    idx = Vector{Int}(undef, length(fs))
    for (i, f) in enumerate(fs)
        j = findfirst(g -> isapprox(g, Float64(f); rtol = 1.0e-9), geom.channel_freqs)
        isnothing(j) &&
            throw(ArgumentError("leaf_window: channel frequency $f not found in geometry"))
        idx[i] = j
    end
    return idx
end

function _time_indices(geom::DataGeometry, leaf)
    ts = lookup(leaf[:vis], Ti)
    idx = Vector{Int}(undef, length(ts))
    for (i, t) in enumerate(ts)
        j = findfirst(g -> isapprox(g, Float64(t); atol = 1.0e-9), geom.times)
        isnothing(j) && throw(ArgumentError("leaf_window: time $t not found in geometry"))
        idx[i] = j
    end
    return idx
end

# ── Apply ────────────────────────────────────────────────────────────────────

"""
    apply_calibration(uvset::UVSet, sol::CalibrationSolution; apply_flags = true,
                      transforms = sol.transforms) -> UVSet

Apply the solution's RECORDED transform chain (`sol.transforms` — e.g. a
station weight scale and an earlier solution applied as a data transform), then
divide every leaf's visibilities by the solution's per-antenna gains. For a
baseline `(a, b)` and correlation product `p` with feeds `(fa, fb)`:

    V_corr = V / (g_a[fa] · conj(g_b[fb])),    W_corr = W · |g_a · g_b|²

Samples where either gain magnitude underflows are flagged (weight 0, vis NaN).

`transforms` defaults to the solution's own recorded chain, so the corrected
set carries the SAME total correction `calibrate(sol, uvset)` produces (minus
its `postcal`/`reduce` tail) — weights included, which matters to anything
that reads them as noise claims. Pass `transforms = ()` to apply the gains
alone: the right call when the data has already been transform-corrected (the
streaming passes do this), and the semantics every internal replay site uses.

Each channel and time is placed in the segment of `sol` it belongs to — matched
by spw and scan identity — so `uvset` may be sampled differently from the solve:
a bandpass fit on scan-averaged data corrects data at full time resolution. A
sample the solution has no segment for is rejected, as is one whose recorded
span crosses a bin boundary — see `evaluate_gains`.

`apply_flags` (default `true`) additionally zero-weights baselines touching a
(station, scan) the solve left UNCONSTRAINED, when `sol.info` records them —
identity gains, i.e. the data would pass through uncalibrated. This is the
EHT-HOPS flag semantic: a station is flagged per scan only when, after the
closure-screened global solve, no strong detection constrains it; a merely weak
baseline between two constrained stations is NOT flagged (it is calibrated by
SNR transfer).
"""
# Whole-set replay of a recorded transform chain. The transform types (and the
# working method) live in the Streaming layer, which loads after this module;
# the stub exists so `apply_calibration` can replay a chain without a layering
# inversion.
function _replay_transforms end

function UVData.apply_calibration(
        uvset::UVSet, sol::CalibrationSolution;
        apply_flags::Bool = true, executor = DynamicScheduler(),
        transforms = sol.transforms,
    )
    for t in transforms
        t === missing && throw(
            ArgumentError(
                "apply_calibration: this solution records a transform that did not survive " *
                    "serialization (saved as `missing`) — re-fit, or apply the original " *
                    "transform chain manually (pass `transforms = ()` to apply gains alone).",
            )
        )
    end
    isempty(transforms) || (uvset = _replay_transforms(uvset, transforms))
    flagged = apply_flags ? _solution_flag_sets(sol.info) : nothing
    # Placement is against the TARGET SET's own geometry, not a leaf's: a
    # channel-index segmentation (`ChannelBlocks`, `FreqGroups`) is defined on
    # the set's whole concatenated channel axis, which one band's leaf does not
    # carry. `leaf_window` then says which of its samples each leaf holds.
    target = build_geometry(uvset)
    return UVData.apply(uvset) do leaf, info, root
        # A lazy leaf materializes to freshly-decoded private arrays we correct
        # in place; an eager leaf is caller-owned, so copy it first. Capture
        # laziness before `materialize_leaf` collapses it.
        private = is_lazy(leaf)
        leaf = materialize_leaf(leaf)
        private || (
            leaf = rebuild_visibilities(
                leaf, copy(parent(leaf[:vis])), copy(parent(leaf[:weights])),
            )
        )
        win = leaf_window(target, leaf)
        g = _composed_gains(                                 # (nchan_leaf, nti_leaf, nant, 2)
            sol, target;
            chan_idx = win.chan_idx, ti_idx = win.ti_idx, time_span = info.time_span,
        )
        _apply_gains!(leaf, g; executor)
        _flag_solution_rows!(
            leaf[:vis], leaf[:weights], UVData.baselines(leaf).pairs,
            _geom_scan_id(sol.geom, info.scan_name), flagged,
        )
        return leaf
    end
end

# The solution's unconstrained (station, geometry scan id) pairs as a lookup set,
# `nothing` when the solution records none.
function _solution_flag_sets(info::NamedTuple)
    (haskey(info, :flagged_ant) && !isempty(info.flagged_ant)) || return nothing
    return Set{Tuple{Int, Int}}(
        (Int(info.flagged_ant[i]), Int(info.flagged_scan[i]))
            for i in eachindex(info.flagged_ant)
    )
end

# The solution's own scan id for a scan label, 0 when the solve never saw it.
# Flags are recorded against these ids, so a label join is what locates them —
# the leaf's epochs need not appear in the solve grid.
_geom_scan_id(geom::DataGeometry, scan_name) =
    something(findfirst(==(String(scan_name)), geom.scan_names), 0)

# Zero-weight (and NaN) whole baseline rows touching a (station, scan) the solve
# left unconstrained — identity gains, so the data would pass through
# uncalibrated. A leaf spans ONE scan, hence one scan id.
function _flag_solution_rows!(Vc, Wc, bl_pairs, scanid::Integer, flagged)
    flagged === nothing && return nothing
    @inbounds for bi in eachindex(bl_pairs)
        a, b = bl_pairs[bi]
        a == b && continue
        ((a, scanid) in flagged || (b, scanid) in flagged) || continue
        Wc[:, :, bi, :] .= zero(eltype(Wc))
        Vc[:, :, bi, :] .= convert(eltype(Vc), NaN)
    end
    return nothing
end

# Gain magnitude below which a cell is treated as unconstrained rather than
# divided through: the correction would amplify noise without bound.
const _GAIN_FLOOR = 1.0e-12

# Correct one (baseline, product) column in place over its (Frequency, Ti)
# plane: `V ← V / (g_a conj(g_b))` and `w ← w · |g_a g_b|²`, where `ga`/`gb` are
# the two stations' gains on that same plane. A degenerate or non-finite gain
# blanks the cell — NaN visibility, zero weight — which is what marks it flagged
# downstream. Each cell reads then writes its own index, so the update is exact
# even though the read and the write hit the same array.
function _correct_column!(vis, w, ga, gb)
    for i in eachindex(vis, w, ga, gb)
        gai = ga[i]
        gbi = gb[i]
        den = gai * conj(gbi)
        if abs(gai) < _GAIN_FLOOR || abs(gbi) < _GAIN_FLOOR || !isfinite(den)
            vis[i] = convert(eltype(vis), NaN)
            w[i] = zero(eltype(w))
        else
            vis[i] = vis[i] / den
            w[i] = w[i] * abs2(gai * gbi)
        end
    end
    return nothing
end

# Correct a leaf's visibilities in place by its complex antenna gains
# `g[c, ti, ant, feed]`. Baselines and correlation products are read off the
# leaf, so they cannot disagree with the arrays they index. The caller owns the
# copy-or-mutate decision: pass a private leaf to overwrite it, or a copy to
# leave the source untouched.
#
# Each (baseline, product) column is independent — disjoint writes over
# read-only gains — so the columns fan out over the inner `executor`. That
# fan-out is load-bearing: this kernel is ≈80% of `apply_calibration` and the
# split is worth ~5× on 8 threads. It is also bit-identical to the serial loop,
# because every cell is the same scalar expression whatever the partition.
function _apply_gains!(leaf, g::AbstractArray{<:Complex, 4}; executor = SerialScheduler())
    vis = leaf[:vis]
    w = leaf[:weights]
    ants = UVData.baselines(leaf).pairs
    feeds = map(correlation_feed_pair, pol_products(leaf))
    columns = vec(CartesianIndices((axes(vis, 3), axes(vis, 4))))
    tforeach(columns; scheduler = executor) do col
        bi, p = Tuple(col)
        a, b = ants[bi]
        fa, fb = feeds[p]
        _correct_column!(
            view(vis, :, :, bi, p), view(w, :, :, bi, p),
            view(g, :, :, a, fa), view(g, :, :, b, fb),
        )
    end
    return leaf
end

# ── Serialization ────────────────────────────────────────────────────────────

"""
    save_solution(path, sol::CalibrationSolution)

Serialize `sol` to `path` via the `Serialization` stdlib inside a versioned
wrapper NamedTuple. The current version is 7 (the solve's
`CalibrationPipeline` is recorded on the solution); earlier versions are
refused on load. Transforms that close over caller code (e.g. a
`CalFunction`) serialize only within the same code state; a transform, a
postcal step, or the recorded pipeline that fails to serialize is recorded as
`missing` with a warning rather than failing the save.
"""
function save_solution(path::AbstractString, sol::CalibrationSolution)
    wrapper = (;
        version = 7, sol.steps, sol.geom, sol.info,
        transforms = _serializable_transforms(sol.transforms),
        postcal = _serializable_transforms(sol.postcal),
        pipeline = _serializable_pipeline(sol.pipeline),
    )
    serialize(path, wrapper)
    return path
end

function _serializable_transforms(ts)
    out = Any[]
    for t in ts
        ok = try
            serialize(IOBuffer(), t)
            true
        catch
            false
        end
        if ok
            push!(out, t)
        else
            @warn "save_solution: transform $(typeof(t)) is not serializable — recorded as `missing`."
            push!(out, missing)
        end
    end
    return out
end

# Same fallback policy as `_serializable_transforms`, for the one recorded
# pipeline: an unserializable pipeline (e.g. a step configured with caller
# code) degrades to `missing` instead of failing the save.
function _serializable_pipeline(p)
    p === nothing && return nothing
    ok = try
        serialize(IOBuffer(), p)
        true
    catch
        false
    end
    ok && return p
    @warn "save_solution: the recorded pipeline is not serializable — recorded as `missing`."
    return missing
end

"""
    load_solution(path) -> CalibrationSolution

Inverse of [`save_solution`](@ref). Only current-format (version 7) files are
supported; files from an earlier Gustavo used a different solution shape and
are refused — re-solve to produce a current-format solution.
"""
function load_solution(path::AbstractString)
    w = deserialize(path)
    w.version == 7 || error(
        "load_solution: unsupported version $(w.version) — saved by an incompatible " *
            "Gustavo (the solution shape changed); re-solve to produce a current file.",
    )
    return CalibrationSolution(
        w.steps, w.geom, w.info;
        transforms = w.transforms, postcal = w.postcal, pipeline = w.pipeline,
    )
end

"""
    save_solution_hdf5(path, sol::CalibrationSolution; gains = true, time_block = 1024)

Write `sol` to an HDF5 caltable readable from any language (Python/h5py, CASA, …),
NOT just Julia. Provided by `GustavoHDF5Ext` — load `HDF5` to enable it.

Layout:
- `gain/real`, `gain/imag` — the evaluated complex antenna gains on the
  `(channel, time, antenna, feed)` grid (`Float32`, chunked + gzip). Written in
  blocks of `time_block` integrations so the full ~GB cube is never resident. Omit
  with `gains = false` to write only the compact parametric form.
- `axes/*` — `channel_freq_hz`, `time`, `scan_of_time`, `spw_of_chan`, `f0`, `t0`.
- `info/*` — the solution-level (run-wide) diagnostics: antenna/scan counts,
  station names, flags, executor/timing summary counters.
- `info/steps/<name>/*` — each pipeline step's OWN diagnostics
  (`stage_info(sol, name)`), one subgroup per step, written generically —
  a third-party `SolveStep`'s custom diagnostics appear here automatically,
  with no changes needed to the writer. A `NamedTuple`- or `DimStack`-valued
  entry (e.g. the fringe step's per-scan `scan_snr`/detection table, any
  step's `timing`) recurses into its own further subgroup.
- root attributes — format/version, units, and the gain convention
  `V_corr = V / (g_a · conj(g_b))`, `weight ×= |g_a g_b|²`.
- `julia/blob` — the `Serialization` bytes of the full solution (steps, geom,
  info, transforms, postcal, pipeline) so Julia can round-trip it losslessly
  (external readers ignore it).
"""
function save_solution_hdf5 end

"""
    load_solution_hdf5(path) -> CalibrationSolution

Reconstruct a `CalibrationSolution` from an HDF5 file written by
[`save_solution_hdf5`](@ref) (via its `julia/blob`). Provided by `GustavoHDF5Ext`.
"""
function load_solution_hdf5 end
