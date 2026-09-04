# ── Step protocol: the composable-pipeline contract ──────────────────────────
#
# A pipeline is an ordered list of steps, each solving its own compiled gain
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
# - `provides(step)`               — names the step's solution slot.
# - `required_grouping(step)`      — leaf-grouping constraint.
# - `fusable_grouping(step)`       — accumulation scope; `:scan` lets the step
#                                    share one streaming pass with its
#                                    neighbors.
# - `start_pass!` / `process_scan!` / `finish_pass!` — the VISITOR CONTRACT:
#   the executor owns all streaming (the scan group is the only unit of data
#   flow — steps never see the uvset); a step accumulates from each
#   materialized, transform-corrected scan view and runs its global solve when
#   the pass completes. Steps are run in the order the pipeline declares them,
#   each in its own streaming pass unless consecutive steps declare themselves
#   scan-local (`fusable_grouping`), in which case they share one — a step that
#   needs an earlier step's correction and does not have it fails from its own
#   solve kernel, not from pipeline construction.
#
# Run-wide resources (task/memory budgets, progress) live on the pipeline's
# `ExecutionConfig`, not on steps: they are properties of a run, shared by
# every pass. Anything that changes what a given step solves is model
# specification and lives on that step. The reference antenna is neither: it is
# a run-wide choice shared by every step's pass rather than a resource, so it
# lives on `CalibrationPipeline` itself (`gauge`), not on any one step or on
# `ExecutionConfig`.

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
    model_components(step::SolveStep, spec) -> AbstractGainModel | (; phase, logamp)

The gain model `step` solves: an [`Calibration.AbstractGainModel`](@ref), or a
bare `(; phase, logamp)` tree of named `GainComponent`s that the runner lifts
to a station-uniform [`StationGainModel`](@ref)
([`Calibration.as_gain_model`](@ref)). `spec = (; geom, antennas)` carries the
data geometry and antenna table the step may consult (e.g. to resolve an
`:auto` option). Default: no components.

A method of the same generic that compiles a `FringeModel` term-list element —
`model_components(element, spec)` — so steps and model-list elements compose
through one mechanism.
"""
model_components(step::SolveStep, spec) = (; phase = (;), logamp = (;))

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
solving is delegated to a pluggable solver object should forward this question
to it.
"""
supports_station_heterogeneity(step::CalibrationStep) = false

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
the step still apply to every scan — fitting a bandpass or a track-global delay
from a few bright calibrator scans and applying it across the board. `prior_solutions`
is the ordered `Vector{StepSolution}` of every earlier step's finished solution —
a step wanting non-data info from an earlier step (e.g. per-scan SNR) reads it
off there. Default: [`Fringe.AllScans`](@ref)`()`.
"""
fit_selection(step::CalibrationStep, prior_solutions) = Fringe.AllScans()

"""
    provides(step::CalibrationStep) -> Symbol

The capability this step contributes (`:fringe`, `:bandpass`, `:adhoc`, …):
names its `StepSolution` slot (`sol[:name]`)
and labels its progress-callback stage. Two steps in the same pipeline must
not share a non-`:nothing` value — their solutions would collide under the
same name. Default: `:nothing`.
"""
provides(step::CalibrationStep) = :nothing

"""
    required_grouping(step::CalibrationStep) -> Symbol

The leaf-grouping this step can run under: `:any`, or `:scan_complete` (the
step needs every spw of a scan materialized together — true of the fringe
search's multi-band concat and of per-scan θ-slot disjointness). Default `:any`.
"""
required_grouping(step::CalibrationStep) = :any

"""
    fusable_grouping(step::CalibrationStep) -> Symbol

The accumulation scope this step's solve needs: `:scan` when the step is
finalizable from one scan group alone — every θ slot it writes for a scan is
written from that scan's data, by the time its `process_scan!` returns — or
`:global` when finishing needs the whole pass (one system closed over every
scan, a statistic pooled across scans, a residual re-search round). Default
`:global`, so a step that has not declared otherwise is never fused.

Consecutive `:scan` steps share one streaming pass: the scan group is
materialized once, each step's [`process_scan!`](@ref) runs on it in declared
order, and each step's just-solved gains are divided out of the resident scan
before the next step sees it — the same correction the transform chain applies
between un-fused passes, on data already in memory. A lazy `UVSet` re-reads and
decodes the dataset once per pass, so this is the difference between N reads of
the data and one; it changes no step's result.

Fusion additionally requires every step in the run to accumulate from every
scan ([`fit_selection`](@ref) returning [`Fringe.AllScans`](@ref)`()`), since
one pass materializes one set of groups, and forbids pass repetition
(`repeat_pass`), which is by definition not scan-local.

Dispatches on the step INSTANCE, not just its type, so a step whose
configuration decides the answer can answer for itself.
"""
fusable_grouping(step::CalibrationStep) = :global

"""
    scan_flags(step::CalibrationStep, r) -> Vector{Tuple{Int, Int}}

The `(station, geometry scan id)` pairs `step`'s solve left unconstrained in
one scan, read off that scan's [`process_scan!`](@ref) return `r`. A fused
output tail corrects each scan group while it is resident — before any step's
`finish_pass!` runs — so flags a `:scan` step finishes inside `process_scan!`
reach the tail through this accessor: like θ, they must be complete for the
scan when `process_scan!` returns. Default: none.
"""
scan_flags(step::CalibrationStep, r) = Tuple{Int, Int}[]

"""
    start_pass!(step::SolveStep, ctx) -> nothing

Called once when the streaming pass containing `step` begins, before any scan
is materialized — allocate the step's accumulators here. Default: no-op.
"""
start_pass!(step::SolveStep, ctx) = nothing

"""
    process_scan!(step::SolveStep, ctx::SolveContext, stack, win::GeometryWindow) -> Any

Accumulate one scan group into the step's state. The EXECUTOR has already
materialized the group and applied the pipeline's transform chain — the step
only consumes the scan's `DimStack` and the
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

Called once when the pass's streaming completes: run the step's global
solve, fill its θ block, and return the stage's diagnostics `NamedTuple`.
This is the step logging interface: every key returned lands on the step's
own `StepSolution.info` (`stage_info(sol, name)`) and — for
`Number`/`AbstractVector`/`String`/nested `NamedTuple`/`DimStack` values —
is written to `save_solution_hdf5`'s `info/steps/<name>/*` automatically.

`ctx.scratch[:pass_results]` holds the collected per-group results:
`(; index, decode, work, reduce, r, out)` per selected group, in no
particular order — `index` is the group's true position, and
[`scan_values`](@ref) scatters a per-result quantity back into global
scan-group order. `decode`/`work` are seconds spent materializing / in
`process_scan!`; `r` is the step's own per-scan return. The runner adds
`t_pass` and a per-scan `timing` `DimStack` to whatever this returns.

A step may request pass repetition by including `repeat_pass = true` in the
return (the key is stripped from the recorded diagnostics). A step declaring
itself scan-local ([`fusable_grouping`](@ref)) may not — a step needing the
pass run again is not finalizable from one scan. Default: empty
diagnostics.
"""
finish_pass!(step::SolveStep, ctx) = NamedTuple()

"""
    scan_values(f, results, ngroups::Integer; default) -> Vector

The per-scan-group values `f(res)` extracts from `results`
(`ctx.scratch[:pass_results]`, see [`finish_pass!`](@ref)), scattered into a
dense length-`ngroups` array in global scan-group index order. A group a
step's [`fit_selection`](@ref) did not select reads back as `default`, not
garbage — `results` need not be sorted, and need not cover every group. The
shared primitive behind the runner's own per-step `timing` (in
[`finish_pass!`](@ref)'s docs) and the built-in fringe estimator's per-scan
`scan_snr`/`scan_ncells`/detection log — reach for it in a CUSTOM step's
`finish_pass!` to publish a per-scan diagnostic the same way, e.g.

    scan_values(res -> res.r.residual_rms, results, ngroups; default = NaN)
"""
function scan_values(f, results, ngroups::Integer; default)
    out = fill(default, ngroups)
    for res in results
        out[res.index] = f(res)
    end
    return out
end

# ── The solve context (shared state of a pipeline run) ───────────────────────

"""
    SolveContext

The shared state of one pipeline solve, threaded through every visitor hook:
the step's own compiled model (`model`/`layout`/`ev`/`θ` — that step's private
gain model, never merged with another step's, see [`StepSolution`](@ref)), the
data geometry, the resolved gauge (`gauge`), the streaming layer
(`stream` — REBUILT between steps as each finished solution is appended to its
transform chain, see `_run_pipeline` — which also carries the run's
[`ExecutionConfig`](@ref) resources), and
`scratch` — a `Dict{Symbol, Any}` for state private to this step's own pass
(e.g. per-group scratch accumulators across search rounds). Non-data info a
Later step wants from an earlier one (e.g. per-scan SNR) is never read through
`scratch` — it's read off the ordered list of finished `StepSolution`s instead
(see [`fit_selection`](@ref)); gain correction between steps is never read
through `scratch` either — it flows through `stream`'s transform chain, so no
step evaluates or mutates another step's θ.

`θ` is a `ComponentVector` over `layout.template`'s axes — its named
blocks (`θ.phase.<name>` / `θ.logamp.<name>`) are directly addressable. Once a
step is finished and wrapped in a [`CalibrationSolution`](@ref),
[`parameters`](@ref) wraps its components' blocks as labelled, dimensioned
`DimArray`s on demand.
"""
mutable struct SolveContext{
        M <: StationGainModel, L <: ParameterLayout, E <: GainEvaluator,
        A <: UVData.AntennaTable, S <: Streaming.ScanStream,
        V <: AbstractVector{Float64},
    }
    model::M
    layout::L
    geom::DataGeometry
    ev::E
    θ::V
    gauge::AbstractGauge
    nant::Int
    antennas::A
    stream::S
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
`CalFunction(f) |> FringeFit(...) |> Bandpass(...)`. Pass it to
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
    CalibrationPipeline(steps...; exec = ExecutionConfig(), gauge = PinAntenna(1))
    CalibrationPipeline(chain::StepChain; exec = ExecutionConfig(), gauge = PinAntenna(1))
    CalibrationPipeline(steps::AbstractVector; exec = ExecutionConfig(), gauge = PinAntenna(1))

An ordered list of [`CalibrationStep`](@ref)s (raw
`Fringe.AbstractDataTransform`s are lifted automatically) plus the run-wide
[`ExecutionConfig`](@ref), `gauge` — the gauge convention every solve step reads
(`ctx.gauge`): an [`AbstractGauge`](@ref), e.g. `PinAntenna("PT")`,
`PinAntenna(["PT", "LA"])` for a ranked fallback, or `ZeroSumPhase()`. A
pipeline needs no [`FringeFit`](@ref) step; any `SolveStep`
composition is legal, including a single standalone step (e.g. a `Bandpass`
fit over data already corrected by an earlier run) — the single-step solve is
the primitive a multi-step pipeline is built from (see [`fit`](@ref)'s
docstring). Solve with [`fit`](@ref) / [`fitcalibrate`](@ref).
"""
struct CalibrationPipeline{X <: ExecutionConfig}
    steps::Vector{CalibrationStep}
    exec::X
    gauge::AbstractGauge
end
CalibrationPipeline(
    steps::AbstractVector; exec::ExecutionConfig = ExecutionConfig(),
    gauge::AbstractGauge = PinAntenna(1),
) = CalibrationPipeline(CalibrationStep[_lift_step(s) for s in steps], exec, gauge)
CalibrationPipeline(
    steps::_Chainable...; exec::ExecutionConfig = ExecutionConfig(),
    gauge::AbstractGauge = PinAntenna(1),
) = CalibrationPipeline(collect(steps); exec, gauge)
CalibrationPipeline(
    chain::StepChain; exec::ExecutionConfig = ExecutionConfig(),
    gauge::AbstractGauge = PinAntenna(1),
) = CalibrationPipeline(chain.steps, exec, gauge)

# Label a step by kind; a lifted transform is named for the transform it wraps.
_step_label(s::CalibrationStep) = string(nameof(typeof(s)))
_step_label(s::DataTransformStep) = string(nameof(typeof(s.t)))

function Base.show(io::IO, ::MIME"text/plain", p::CalibrationPipeline)
    println(io, "CalibrationPipeline (", length(p.steps), " step(s))")
    for (i, s) in enumerate(p.steps)
        println(io, "  ", i, ". ", _step_label(s))
    end
    print(
        io, "  exec: outer=", nameof(typeof(outer_executor(p.exec))),
        ", inner=", nameof(typeof(inner_executor(p.exec))),
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
