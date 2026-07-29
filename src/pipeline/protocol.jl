# ── Step protocol: the composable-pipeline contract ──────────────────────────
#
# A pipeline is an ordered list of steps sharing ONE compiled gain model: at
# fit time each `SolveStep` declares its model components, `plan_parameters`
# lays out a single θ, and each step fills its own block — so gauge accounting
# (feed tying, R–L pins) lives under one model while every stage remains
# individually inspectable through the solution's per-stage records.
#
# Hooks a step may implement (all have working defaults):
# - `model_components(step, spec)` — the gain-model components this step solves.
# - `transforms(step)`             — data transforms it contributes to the
#                                    materialization chain (see `CalFunction`).
# - `fit_selection(step)`          — which scans feed its accumulation
#                                    (fit-on-subset / apply-everywhere).
# - `provides(step)` / `requires(step)` — stage ordering contract.
# - `required_grouping(step)`      — leaf-grouping constraint.
# - `start_pass!` / `process_scan!` / `finish_pass!` — the VISITOR CONTRACT:
#   the executor owns all streaming (the scan group is the only unit of data
#   flow — steps never see the uvset); a step accumulates from each
#   materialized, transform-corrected scan view and runs its global solve when
#   the pass completes. The compiler packs steps into a minimal number of
#   full-data passes from `requires`/`provides`: independent steps share one
#   pass; a step needing another's finalized θ starts a new one.
#
# Run-wide resources (task/memory budgets, progress) live on the pipeline's
# `ExecutionConfig`, NOT on steps: they are properties of a run, shared by
# every pass. Anything that changes WHAT is solved (the reference antenna
# included) is model specification and lives on the steps.

"""
    SolveStep <: CalibrationStep

A pipeline stage that solves part of the shared gain model (fringe fit,
bandpass estimation, temporal smoothing, …). Solve steps declare their model
components via [`model_components`](@ref) and run under the executor-driven
visitor contract — [`start_pass!`](@ref) / [`process_scan!`](@ref) /
[`finish_pass!`](@ref); reduce steps ([`ReduceStep`](@ref)) transform data
instead.
"""
abstract type SolveStep <: CalibrationStep end

"""
    model_components(step::SolveStep, spec) -> (; phase, logamp)

The gain-model components `step` solves, as tuples of `TiedComponent`s to
append to the compiled `StationGainModel`'s phase / log-amplitude lists.
`spec` carries the data geometry and antenna table the step may consult (e.g.
to resolve an `:auto` option). Default: no components.

A method of the same generic that compiles a `FringeModel` term-list element —
`model_components(element, geom::DataGeometry)` — so steps and model-list
elements compose through one mechanism.
"""
model_components(step::SolveStep, spec) = (; phase = (), logamp = ())

"""
    transforms(step::CalibrationStep) -> Tuple

Data transforms ([`Fringe.AbstractDataTransform`](@ref)) this step contributes
to the materialization chain, applied in pipeline order to every scan group as
it is materialized. Default: none.
"""
transforms(step::CalibrationStep) = ()

"""
    fit_selection(step::CalibrationStep) -> Fringe.AbstractScanSelection

Which scans feed this step's accumulation. Time-global components solved by
the step still apply to EVERY scan — fitting a bandpass or a global R–L delay
from a few bright calibrator scans and applying it across the board. Default:
[`Fringe.AllScans`](@ref)`()`.
"""
fit_selection(step::CalibrationStep) = Fringe.AllScans()

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
                    mem_budget = nothing, progress = nothing, exclude_colocated = true)

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
- `executor` — the task-scheduling backend ([`ThreadsExecutor`](@ref) by
  default, via the ambient [`with_executor`](@ref) scope;
  [`DaggerExecutor`](@ref) opt-in/experimental). Identical
  admission/chunking/fold order under both, so θ and outputs are bit-identical
  across executors.
"""
Base.@kwdef struct ExecutionConfig{P, E <: Executors.AbstractExecutor}
    ntasks::Int = Threads.nthreads()
    mem_fraction::Float64 = 0.6
    mem_budget::Union{Nothing, Float64} = nothing
    progress::P = nothing
    exclude_colocated::Bool = true
    executor::E = Executors.current_executor()
end

# ── The solve context (shared state of a pipeline run) ───────────────────────

"""
    SolveContext

The shared state of one pipeline solve, threaded through every visitor hook:
the COMPILED model (`model`/`layout`/`ev` — one θ for all stages, each step
filling its own block), the data geometry, the resolved gauge pin (`ref_ant`),
the streaming layer (`stream`), the run's `exec` resources, the per-stage
provenance records accumulated so far (`stages`), and `scratch` — a
`Dict{Symbol, Any}` for cross-step state (the runner puts each pass's collected
per-group results in `:pass_results`; the fringe stage publishes `:scan_snr`
for SNR-aware scan selections and its `:refine` service for the dTEC/SBD
θ slots it owns; the bandpass stage records `:refined_scans`).
"""
mutable struct SolveContext{
        M <: StationGainModel, E <: GainEvaluator, A <: UVData.AntennaTable,
        S <: Streaming.ScanStream, X <: ExecutionConfig,
    }
    model::M
    layout::ParameterLayout
    geom::DataGeometry
    ev::E
    θ::Vector{Float64}
    ref_ant::Int
    nant::Int
    antennas::A
    stream::S
    exec::X
    stages::Vector{StageRecord}
    scratch::Dict{Symbol, Any}
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
