# ── Step protocol: the composable-pipeline contract ──────────────────────────
#
# A pipeline is an ordered list of steps, each solving its OWN compiled gain
# model: at fit time a `SolveStep` declares its model components and
# `plan_parameters` lays out that step's own θ alone — no step's θ block is
# ever shared with, or visible to, another step's. Gain correction between
# steps flows through the scan stream's transform chain (each finished step's
# solution is appended, so later steps read already-corrected data), and every
# stage remains individually inspectable through the solution's per-stage
# records (composed from the run's finished `StepSolution`s once every step
# has solved).
#
# Hooks a step may implement (all have working defaults):
# - `model_components(step, spec)` — the gain-model components this step solves.
# - `transforms(step)`             — data transforms it contributes to the
#                                    materialization chain (see `CalFunction`).
# - `fit_selection(step, prior_solutions)` — which scans feed its accumulation
#                                    (fit-on-subset / apply-everywhere); reads
#                                    non-data info from earlier steps (e.g.
#                                    per-scan SNR) off their `StepSolution`s.
# - `provides(step)` / `requires(step)` — stage ordering contract.
# - `required_grouping(step)`      — leaf-grouping constraint.
# - `start_pass!` / `process_scan!` / `finish_pass!` — the VISITOR CONTRACT:
#   the executor owns all streaming (the scan group is the only unit of data
#   flow — steps never see the uvset); a step accumulates from each
#   materialized, transform-corrected scan view and runs its global solve when
#   the pass completes. Every step gets its own streaming pass, ordered by
#   `requires`/`provides`.
#
# Run-wide resources (task/memory budgets, progress) live on the pipeline's
# `ExecutionConfig`, NOT on steps: they are properties of a run, shared by
# every pass. Anything that changes WHAT is solved (the reference antenna
# included) is model specification and lives on the steps.

"""
    SolveStep <: CalibrationStep

A pipeline stage that solves its own gain model (fringe fit, bandpass
estimation, temporal smoothing, …). Solve steps declare their model components
via [`model_components`](@ref) and run under the executor-driven visitor
contract — [`start_pass!`](@ref) / [`process_scan!`](@ref) /
[`finish_pass!`](@ref); reduce steps ([`ReduceStep`](@ref)) transform data
instead.
"""
abstract type SolveStep <: CalibrationStep end

"""
    model_components(step::SolveStep, spec) -> (; phase, logamp)

The gain-model components `step` solves, as named component trees (`NamedTuple`s
of `TiedComponent`s) to merge into the compiled `StationGainModel`'s phase /
log-amplitude groups. `spec` carries the data geometry and antenna table the
step may consult (e.g. to resolve an `:auto` option). Default: no components.

A method of the same generic that compiles a `FringeModel` term-list element —
`model_components(element, geom::DataGeometry)` — so steps and model-list
elements compose through one mechanism.
"""
model_components(step::SolveStep, spec) = (; phase = (;), logamp = (;))

"""
    transforms(step::CalibrationStep) -> Tuple

Data transforms ([`Fringe.AbstractDataTransform`](@ref)) this step contributes
to the materialization chain, applied in pipeline order to every scan group as
it is materialized. Default: none.
"""
transforms(step::CalibrationStep) = ()

"""
    fit_selection(step::CalibrationStep, prior_solutions) -> Fringe.AbstractScanSelection

Which scans feed this step's accumulation. Time-global components solved by
the step still apply to EVERY scan — fitting a bandpass or a global R–L delay
from a few bright calibrator scans and applying it across the board. `prior_solutions`
is the ordered `Vector{StepSolution}` of every earlier step's finished solution —
a step wanting non-data info from an earlier step (e.g. per-scan SNR) reads it
off there. Default: [`Fringe.AllScans`](@ref)`()`.
"""
fit_selection(step::CalibrationStep, prior_solutions) = Fringe.AllScans()

"""
    provides(step::CalibrationStep) -> Symbol

The capability this step contributes (`:fringe`, `:bandpass`, `:adhoc`, …),
consumed by later steps' [`requires`](@ref). Default: `:nothing`.
"""
provides(step::CalibrationStep) = :nothing

"""
    requires(step::CalibrationStep) -> Tuple{Vararg{Symbol}}

Capabilities that must have been provided by EARLIER steps (e.g. the bandpass
estimator requires `:fringe` — it accumulates the fringe-corrected residual).
Checked when the pipeline is compiled. Default: none.
"""
requires(step::CalibrationStep) = ()

"""
    required_grouping(step::CalibrationStep) -> Symbol

The leaf-grouping this step can run under: `:any`, or `:scan_complete` (the
step needs every band of a scan materialized together — true of the fringe
search's multi-band concat and of per-scan θ-slot disjointness). Default `:any`.
"""
required_grouping(step::CalibrationStep) = :any

"""
    start_pass!(step::SolveStep, ctx) -> nothing

Called once when the streaming pass containing `step` begins, before any scan
is materialized — allocate the step's accumulators here. Default: no-op.
"""
start_pass!(step::SolveStep, ctx) = nothing

"""
    process_scan!(step::SolveStep, ctx::SolveContext, stack, win::GeometryWindow) -> Any

Accumulate one scan group into the step's state. The EXECUTOR has already
materialized the group, applied the pipeline's transform chain, and admitted it
against the memory budget — the step only consumes the scan's `DimStack` and the
[`GeometryWindow`](@ref) addressing it in the solve's index space (and may
read/write its own per-scan θ slots through `ctx`). Called once per selected scan group
(see [`fit_selection`](@ref)), possibly concurrently across groups; per-scan θ
slots are disjoint. The RETURN VALUE is collected by the runner — one entry per
selected group, in group-index order, delivered to [`finish_pass!`](@ref) via
`ctx.scratch[:pass_results]` — so a step needs no locking: return the scan's
contribution and fold in `finish_pass!` (the fold is then deterministic at any
concurrency). Default: no-op returning `nothing`.
"""
process_scan!(step::SolveStep, ctx, stack, win) = nothing

"""
    finish_pass!(step::SolveStep, ctx::SolveContext) -> NamedTuple

Called once when the pass's streaming completes: run the step's global solve
(stationization, bandpass solve, smoother fit, …), fill its θ block, and return
the stage's diagnostics NamedTuple (recorded on the solution's `StageRecord`).
`ctx.scratch[:pass_results]` holds the pass's collected per-group results —
`(; index, decode, work, r)` per selected group in group-index order, where
`index` is the stream group index, `decode`/`work` are seconds spent
materializing / in `process_scan!`, and `r` is the step's per-scan return.
A step may request pass repetition (residual re-search rounds) by including
`repeat_pass = true` in the returned NamedTuple — the runner streams the pass
again (that key is stripped from the recorded diagnostics). Default: empty
diagnostics.

Together with [`start_pass!`](@ref) and [`process_scan!`](@ref) this is the
step execution contract: the runner gives each solve step its own streaming
pass, ordered and validated by `requires`/`provides` (every built-in stage
consumes the residual of all previously-solved θ, so passes never share), and
drives each pass through the streaming layer (`Fringe.map_groups`).
"""
finish_pass!(step::SolveStep, ctx) = NamedTuple()

# ── Execution configuration (run-wide, not per-step) ─────────────────────────

"""
    ExecutionConfig(; ntasks = Threads.nthreads(), mem_fraction = 0.6,
                    mem_budget = nothing, progress = nothing, exclude_colocated = true,
                    outer_executor = ThreadsExecutor(), inner_executor = DynamicScheduler())

Run-wide RESOURCES for a pipeline execution, shared by every pass — as opposed
to per-step options, which shape an estimator or the model. Anything that
changes WHAT is solved (including the reference antenna: the gauge pin is part
of the model specification) lives on the steps' model options, never here —
two runs differing only in their `ExecutionConfig` solve the same problem.

- `ntasks`, `mem_fraction` / `mem_budget` — group concurrency under a
  DETERMINISTIC memory budget (`mem_budget` bytes if set, else `mem_fraction`
  of physical RAM).
- `progress` — `(stage, done, total)` callback per completed scan of each pass.
- `exclude_colocated` — drop intra-site (co-located twin) baselines from the
  bandpass/adhoc accumulations (their non-closing crosstalk pollutes both).
- `outer_executor` — the ACROSS-scan (group scheduling) backend:
  [`ThreadsExecutor`](@ref) by default (the budget-admission worker pool on
  `Threads.@spawn`), or [`DaggerExecutor`](@ref) for a distributed run. The
  admission policy (memory budget, largest-first, `ntasks` cap) is identical
  under both.
- `inner_executor` — the WITHIN-scan fan-out scheduler, an OhMyThreads
  `Scheduler` (`DynamicScheduler()` by default; `SerialScheduler()` to run the
  within-scan solves single-threaded). `scan_stream` fixes its chunk count to
  the inner parallelism the memory budget leaves per group.
"""
Base.@kwdef struct ExecutionConfig{P, O, I}
    ntasks::Int = Threads.nthreads()
    mem_fraction::Float64 = 0.6
    mem_budget::Union{Nothing, Float64} = nothing
    progress::P = nothing
    exclude_colocated::Bool = true
    outer_executor::O = Executors.DEFAULT_EXECUTOR[]
    inner_executor::I = DynamicScheduler()
end

# ── The solve context (shared state of a pipeline run) ───────────────────────

"""
    SolveContext

The shared state of one pipeline solve, threaded through every visitor hook:
the step's OWN compiled model (`model`/`layout`/`ev`/`θ` — that step's private
gain model, never merged with another step's, see [`StepSolution`](@ref)), the
data geometry, the resolved gauge pin (`ref_ant`), the streaming layer
(`stream` — REBUILT between steps as each finished solution is appended to its
transform chain, see `_run_pipeline`), the run's `exec` resources, the
per-stage provenance records accumulated so far (`stages`), and `scratch` — a
`Dict{Symbol, Any}` for state PRIVATE to this step's own pass (e.g. per-group
scratch accumulators across search rounds). Non-data info a LATER step wants
from an earlier one (e.g. per-scan SNR) is never read through `scratch` — it's
read off the ordered list of finished `StepSolution`s instead (see
[`fit_selection`](@ref)); gain correction between steps is never read through
`scratch` either — it flows through `stream`'s transform chain, so no step
evaluates or mutates another step's θ.

`θ` is a [`ComponentVector`](@ref) over `layout.template`'s axes — its named
blocks (`θ.phase.<name>` / `θ.logamp.<name>`) are directly addressable; wrap a
component's block as a labelled, dimensioned `DimArray` on demand with
[`@comp`](@ref).
"""
mutable struct SolveContext{
        M <: StationGainModel, L <: ParameterLayout, E <: GainEvaluator,
        A <: UVData.AntennaTable, S <: Streaming.ScanStream, X <: ExecutionConfig,
        V <: AbstractVector{Float64},
    }
    model::M
    layout::L
    geom::DataGeometry
    ev::E
    θ::V
    ref_ant::Int
    nant::Int
    antennas::A
    stream::S
    exec::X
    stages::Vector{StageRecord}
    scratch::Dict{Symbol, Any}
end

"""
    StepSolution(name, model, layout, θ, info)

One finished [`SolveStep`](@ref)'s own gain model, solved parameters, and
diagnostics — `model`/`layout` compiled from that step's own
[`model_components`](@ref) alone, never merged with another step's, and `θ` a
named [`ComponentVector`](@ref) over `layout.template`'s axes. The runner
(`_run_pipeline`) keeps the ordered list of every step's `StepSolution` as
it runs the pipeline, both to chain gain correction between steps (each
finished one is appended to the scan stream's transform chain) and as the
non-data-input channel hooks like [`fit_selection`](@ref) read (e.g.
`BandpassEstimator`'s SNR-aware selection reads the fringe stage's
`info.scan_snr`). Internal: `CalibrationSolution`'s public shape is unchanged
by this — the runner composes one legacy-shaped solution from the finished
`StepSolution`s at the end of the run.
"""
struct StepSolution{M <: StationGainModel, L <: ParameterLayout, V <: AbstractVector{Float64}}
    name::Symbol
    model::M
    layout::L
    θ::V
    info::NamedTuple
end

# ── Transforms as pipeline steps ─────────────────────────────────────────────

"""
    DataTransformStep(t::Fringe.AbstractDataTransform)

Lifts a data transform into a pipeline step so transforms compose in the step
chain: `CalFunction(f) |> FringeFit(...)`. Raw transforms are lifted
automatically by `|>` and the `CalibrationPipeline` constructors, so you rarely
construct this directly.
"""
struct DataTransformStep{T <: Fringe.AbstractDataTransform} <: CalibrationStep
    t::T
end
transforms(s::DataTransformStep) = (s.t,)

function run_step(s::DataTransformStep, ctx::CalibrationContext)
    return error(
        "DataTransformStep($(typeof(s.t))) cannot run through the sequential `run_step` " *
            "chain — use `fit`/`fitcalibrate`, which thread transforms into the streaming solve."
    )
end

# Lift pipeline elements to steps: transforms wrap, steps pass through.
_lift_step(s::CalibrationStep) = s
_lift_step(t::Fringe.AbstractDataTransform) = DataTransformStep(t)
_lift_step(x) = error(
    "not a pipeline element: $(typeof(x)) — expected a CalibrationStep or an AbstractDataTransform."
)

# ── Step chaining ────────────────────────────────────────────────────────────

"""
    StepChain

An ordered chain of pipeline steps built with `|>`:
`CalFunction(f) |> FringeFit(...) |> BandpassEstimator(...)`. Pass it to
[`CalibrationPipeline`](@ref) (or directly to [`fit`](@ref)).
"""
struct StepChain
    steps::Vector{CalibrationStep}
end

const _Chainable = Union{CalibrationStep, Fringe.AbstractDataTransform}
Base.:|>(a::_Chainable, b::_Chainable) = StepChain([_lift_step(a), _lift_step(b)])
Base.:|>(c::StepChain, b::_Chainable) = StepChain(vcat(c.steps, _lift_step(b)))
Base.:|>(a::_Chainable, c::StepChain) = StepChain(vcat(_lift_step(a), c.steps))
Base.:|>(a::StepChain, b::StepChain) = StepChain(vcat(a.steps, b.steps))

# ── The pipeline ─────────────────────────────────────────────────────────────

"""
    CalibrationPipeline(steps...; exec = ExecutionConfig())
    CalibrationPipeline(chain::StepChain; exec = ExecutionConfig())
    CalibrationPipeline(steps::AbstractVector; exec = ExecutionConfig())

An ordered list of [`CalibrationStep`](@ref)s (raw
`Fringe.AbstractDataTransform`s are lifted automatically) plus the run-wide
[`ExecutionConfig`](@ref). Solve with [`fit`](@ref) / [`fitcalibrate`](@ref).
"""
struct CalibrationPipeline{X <: ExecutionConfig}
    steps::Vector{CalibrationStep}
    exec::X
end
CalibrationPipeline(steps::AbstractVector; exec::ExecutionConfig = ExecutionConfig()) =
    CalibrationPipeline(CalibrationStep[_lift_step(s) for s in steps], exec)
CalibrationPipeline(steps::_Chainable...; exec::ExecutionConfig = ExecutionConfig()) =
    CalibrationPipeline(collect(steps); exec)
CalibrationPipeline(chain::StepChain; exec::ExecutionConfig = ExecutionConfig()) =
    CalibrationPipeline(chain.steps, exec)

# Label a step by kind; a lifted transform is named for the transform it wraps.
_step_label(s::CalibrationStep) = string(nameof(typeof(s)))
_step_label(s::DataTransformStep) = string(nameof(typeof(s.t)))

function Base.show(io::IO, ::MIME"text/plain", p::CalibrationPipeline)
    println(io, "CalibrationPipeline (", length(p.steps), " step(s))")
    for (i, s) in enumerate(p.steps)
        println(io, "  ", i, ". ", _step_label(s))
    end
    print(
        io, "  exec: outer=", nameof(typeof(p.exec.outer_executor)),
        ", inner=", nameof(typeof(p.exec.inner_executor)), ", ntasks=", p.exec.ntasks,
    )
    return io
end

Base.show(io::IO, p::CalibrationPipeline) =
    print(io, "CalibrationPipeline(", length(p.steps), " steps)")

# Ordered container over its steps.
Base.length(p::CalibrationPipeline) = length(p.steps)
Base.getindex(p::CalibrationPipeline, i) = p.steps[i]
Base.iterate(p::CalibrationPipeline, args...) = iterate(p.steps, args...)
Base.eltype(::Type{<:CalibrationPipeline}) = CalibrationStep
