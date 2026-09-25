# ── Step protocol ────────────────────────────────────────────────────────────
#
# A pipeline is an ordered tuple of solve steps and corrections, each step
# solving its own compiled gain model: at fit time a `SolveStep` declares its
# model components and `plan_parameters` lays out that step's own θ alone — no
# step's θ block is ever shared with, or visible to, another step's. A step
# reads the data the corrections before it and the earlier steps' gains
# produced (each finished step's solution joins the corrections of every later
# step), and every stage remains individually inspectable through the
# solution's per-stage records.
#
# Hooks a step may implement:
# - `model_components(step, spec)` — the gain-model components this step solves.
# - `provides(step)`               — names the step's solution slot.
# - `solve(step, ctx)`             — fills the step's θ, reading the data through
#                                    `each_group(f, ctx)`, and returns the step's
#                                    diagnostics.
#
# Run-wide resources (task/memory budgets, progress) live on the
# `ExecutionConfig` passed to `fit`; the gauge is a run-wide choice passed to
# `fit` and recorded on the solution. Anything that changes what a given step
# solves is model specification and lives on that step.

"""
    SolveStep

A pipeline stage that solves its own gain model (fringe fit, bandpass
estimation, adhoc phasing, …). A solve step declares its model components
with [`model_components`](@ref) and fits them in [`solve`](@ref), reading the
data with [`each_group`](@ref).
"""
abstract type SolveStep end

"""
    model_components(step::SolveStep, spec) -> GainModel

The [`GainModel`](@ref) `step` solves. `spec = (; geom)` carries the data
geometry (with the run's `stations`) the step may consult (e.g. to resolve an
`:auto` option). Default: no components.

A method of the same generic compiles a data-dependent model element —
`model_components(element, spec)`, e.g. a [`DispersionModel`](@ref) — so steps
and model elements compose through one mechanism.
"""
model_components(step::SolveStep, spec) = GainModel()

"""
    supports_station_heterogeneity(step::SolveStep) -> Bool

Whether `step`'s solver handles a gain model whose component signatures differ
across stations. Default `false`: the runner then rejects a heterogeneous
compiled model aimed at the step
([`Calibration.require_station_uniform`](@ref)), naming the differing
components and their per-station signatures, so an undeclared solver never
receives θ blocks it would silently leave unsolved.

A step that declares `true` must write its solve as a loop over the station
blocks the layout supplies — [`Calibration.station_blocks`](@ref) — rather
than assuming one rectangular `(…, nant)` leaf per component. A step whose
solving is delegated to a pluggable solver object forwards this question to it,
so the generic also answers for solver objects — [`Bandpass`](@ref) asks its
smoother.
"""
supports_station_heterogeneity(step::SolveStep) = false

# How the heterogeneity rejection names its subject. A step that delegates its
# solve is not itself the thing that cannot take the model, so it names the
# solver and the configuration that would accept it; the step type alone leaves
# the user nothing to change.
heterogeneity_rejector(step::SolveStep) = string(nameof(typeof(step)))

"""
    provides(step::SolveStep) -> Symbol

The capability this step contributes (`:fringe`, `:bandpass`, `:adhoc`, …):
names its `StepSolution` slot (`sol[:name]`)
and labels its progress-callback stage. Two steps in the same pipeline must
not share a non-`:nothing` value — their solutions would collide under the
same name. Default: `:nothing`.
"""
provides(step::SolveStep) = :nothing

"""
    solve(step::SolveStep, ctx::SolveContext) -> NamedTuple

Fit `step`'s own model: fill `ctx.θ` and return the step's diagnostics, which
become its `StepSolution.info` (`stage_info(sol, name)`). The data are read
with [`each_group`](@ref), once per pass the solve needs; a solve that
iterates (a residual re-search, say) calls it once per round. The runner adds
`t_pass`, the solve's wall time, and `timing`, each scan group's decode and
work seconds summed over the step's passes.

Every `SolveStep` must define a method.
"""
function solve end

solve(step::SolveStep, ctx) = throw(
    ArgumentError(
        "$(nameof(typeof(step))) does not define `Gustavo.solve(::$(nameof(typeof(step))), ctx)`, " *
            "which fits the step's θ and returns its diagnostics NamedTuple.",
    ),
)

# ── The solve context ────────────────────────────────────────────────────────

"""
    SolveContext

What a step's [`solve`](@ref) works with: the step's own compiled model
(`model`, `layout`, and `θ`, the flat parameter vector over `layout`, which
`solve` fills; a component's block is `reshape(view(θ, plan.range),
plan.shape)`), the data geometry `geom` (its `stations` are the run's station
table), the resolved `gauge`, `nant`, the scan groups of the data
(`groupby(ps, ByScan())`), and the corrections the step's data pass through:
the pipeline's corrections before the step and every earlier step's gains.
Another step's θ is never visible here; it reaches the step only as a
correction of the data.
"""
const _PassTiming = @NamedTuple{decode::Vector{Float64}, work::Vector{Float64}}

struct SolveContext{
        M <: GainModel, L <: ParameterLayout, V <: AbstractVector{Float64},
        G <: AbstractDict, X <: ExecutionConfig,
    }
    model::M
    layout::L
    geom::DataGeometry
    θ::V
    gauge::AbstractGauge
    nant::Int
    groups::G
    charges::Vector{Int}
    corrections::Vector{Any}
    exec::X
    stage::Symbol
    passes::Vector{_PassTiming}
end

"""
    each_group(f, ctx::SolveContext) -> Vector

Read each scan group of the step's data and return `f(group)` for every
group, in group order. `group` is a `ProcessingSet` of in-memory Measurement
Sets, one per spectral window of the scan, with the corrections before the
step applied. Groups run on the run's outer scheduler, heaviest first, and the
progress callback is told of each.

`f` may run concurrently across groups, so it must not write shared state
other than θ slots belonging to its own scan; it returns its scan's
contribution instead, and the caller combines the returned values.
"""
function each_group(f::F, ctx::SolveContext) where {F}
    out = _map_groups(ctx.groups, ctx.charges, ctx.exec; stage = ctx.stage) do group
        ta = time_ns()
        corrected = _read_group(group, ctx.corrections, ctx.geom, inner_executor(ctx.exec))
        tb = time_ns()
        r = f(corrected)
        (; decode = (tb - ta) / 1.0e9, work = (time_ns() - tb) / 1.0e9, r)
    end
    push!(ctx.passes, (; decode = Float64[o.decode for o in out], work = Float64[o.work for o in out]))
    return map(o -> o.r, out)
end

# One scan group read into memory, each Measurement Set passed through
# `corrections`.
function _read_group(group::XRadio.ProcessingSet, corrections, geom::DataGeometry, executor)
    named = collect(pairs(group))
    # Typed `tmap`: the untyped form rejects `GreedyScheduler`.
    members = tmap(XRadio.MeasurementSet, named; scheduler = executor) do (_, ms)
        _apply_corrections(corrections, read(ms), geom)
    end
    return XRadio.ProcessingSet(
        OrderedDict{Symbol, XRadio.MeasurementSet}(first.(named) .=> members),
        copy(DimensionalData.metadata(group)),
    )
end

# ── Group scheduling ─────────────────────────────────────────────────────────

const _PROGRESS_LOCK = ReentrantLock()

_report_progress(::Nothing, stage, done, total) = nothing
function _report_progress(cb, stage, done, total)
    lock(_PROGRESS_LOCK) do
        try
            cb(stage, Int(done), Int(total))
        catch err
            @warn "progress callback failed" stage err maxlog = 1
        end
    end
    return nothing
end

# `work(group)` over `groups` (their values) on the outer scheduler, heaviest
# first by `charges`, with the results in group order and
# `(stage, done, total)` reported to the run's progress callback.
function _map_groups(work::F, groups, charges, exec::ExecutionConfig; stage::Symbol) where {F}
    items = collect(values(groups))
    total = length(items)
    progress = progress_callback(exec)
    _report_progress(progress, stage, 0, total)
    done = Threads.Atomic{Int}(0)
    function wrapped(group)
        r = work(group)
        _report_progress(progress, stage, Threads.atomic_add!(done, 1) + 1, total)
        return r
    end
    return _scheduled_map(wrapped, items, charges; executor = outer_executor(exec))
end

# Largest-first parallel map: run `work` over `items` on `executor`, dispatching
# the heaviest item (by `charges`) first so the long poles start immediately.
# Results in `items` order. A failed worker rethrows after the other workers
# drain the queue. `executor` is used exactly as configured — how many items run
# at once is its decision, checked against the memory budget upstream.
#
# Each backend fills an `Any` sink and returns `map(identity, sink)`: `work`'s
# return type is not known before it runs, and tasks write their slots
# concurrently, so the sink has to admit any value; `map` then recovers the
# concrete element type for whatever consumes the pass.
function _scheduled_map(work::F, items, charges; executor = SerialScheduler()) where {F}
    length(items) == length(charges) || throw(
        DimensionMismatch("items and charges must match: $(length(items)) vs $(length(charges))"),
    )
    Base.require_one_based_indexing(items, charges)
    return _scheduled_map(executor, work, items, charges)
end

# With one worker the dispatch order cannot matter.
_scheduled_map(::SerialScheduler, work::F, items, charges) where {F} = map(work, items)

# `GreedyScheduler` is the one that keeps largest-first meaningful under uneven
# charges — it hands each task the next group off the queue — where a chunking
# scheduler assigns groups to tasks up front. A backend with different
# task-lifetime needs adds its own method on its executor type.
function _scheduled_map(sched::Scheduler, work::F, items, charges) where {F}
    out = Vector{Any}(undef, length(items))
    tforeach(sortperm(charges; rev = true); scheduler = sched) do k
        out[k] = work(items[k])
    end
    return map(identity, out)
end

# ── Pipelines ────────────────────────────────────────────────────────────────

# The pipeline elements `|>` joins: solve steps and recorded corrections. A plain
# function may also sit in a pipeline, written as a tuple or vector, since
# `f |> x` is Base's function application.
const PipelineElement = Union{SolveStep, AbstractDataTransform, AprioriAmplitude}

"""
    a |> b

Build a pipeline, the tuple `(a, b)`, from solve steps and corrections;
`pipeline |> c` appends and `c |> pipeline` prepends:
`AutocorrelationNormalization() |> BaselineFringeFit() |> Bandpass()`. Two pipelines
join by splatting, `(p..., q...)`.
"""
Base.:|>(a::PipelineElement, b::PipelineElement) = (a, b)
Base.:|>(a::Tuple, b::PipelineElement) = (a..., b)
Base.:|>(a::PipelineElement, b::Tuple) = (a, b...)

function _check_pipeline(seq)
    for x in seq
        x isa Union{PipelineElement, Function} || throw(
            ArgumentError(
                "a pipeline holds solve steps and corrections (an AbstractDataTransform, " *
                    "AprioriAmplitude, or a function of a Measurement Set), not $(nameof(typeof(x)))"
            )
        )
    end
    return seq
end
