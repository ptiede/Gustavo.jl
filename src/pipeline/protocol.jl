# ── Step protocol ────────────────────────────────────────────────────────────
#
# A pipeline is an ordered tuple of solve steps and corrections, each step
# solving its own compiled gain model: at fit time a `SolveStep` declares its
# model components and `plan_parameters` lays out that step's own θ alone — no
# step's θ block is ever shared with, or visible to, another step's. A step
# reads the data the corrections before it and the earlier steps' gains
# produced (each finished step's solution joins the scan stream's transform
# chain), and every stage remains individually inspectable through the
# solution's per-stage records.
#
# Hooks a step may implement (all have working defaults):
# - `model_components(step, spec)` — the gain-model components this step solves.
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
#   the pass completes. Consecutive scan-local steps (`fusable_grouping`) with
#   no correction between them share one pass.
#
# Run-wide resources (task/memory budgets, progress) live on the
# `ExecutionConfig` passed to `fit`; the gauge is a run-wide choice passed to
# `fit` and recorded on the solution. Anything that changes what a given step
# solves is model specification and lives on that step.

"""
    SolveStep

A pipeline stage that solves its own gain model (fringe fit, bandpass
estimation, adhoc phasing, …). Solve steps declare their model components
via [`model_components`](@ref) and run under the executor-driven visitor
contract — [`start_pass!`](@ref) / [`process_scan!`](@ref) /
[`finish_pass!`](@ref).
"""
abstract type SolveStep end

"""
    model_components(step::SolveStep, spec) -> GainModel

The [`GainModel`](@ref) `step` solves. `spec = (; geom, antennas)` carries the
data geometry and antenna table the step may consult (e.g. to resolve an
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
    fit_selection(step::SolveStep, prior_solutions) -> Fring.AbstractScanSelection

Which scans feed this step's accumulation. Time-global components solved by
the step still apply to every scan — fitting a bandpass or a track-global delay
from a few bright calibrator scans and applying it across the board. `prior_solutions`
is the ordered `Vector{StepSolution}` of every earlier step's finished solution —
a step wanting non-data info from an earlier step (e.g. per-scan SNR) reads it
off there. Default: [`Fring.AllScans`](@ref)`()`.
"""
fit_selection(step::SolveStep, prior_solutions) = Fring.AllScans()

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
    required_grouping(step::SolveStep) -> Symbol

The leaf-grouping this step can run under: `:any`, or `:scan_complete` (the
step needs every spw of a scan materialized together — true of the fringe
search's multi-band concat and of per-scan θ-slot disjointness). Default `:any`.
"""
required_grouping(step::SolveStep) = :any

"""
    fusable_grouping(step::SolveStep) -> Symbol

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
scan ([`fit_selection`](@ref) returning [`Fring.AllScans`](@ref)`()`), since
one pass materializes one set of groups, and forbids pass repetition
(`repeat_pass`), which is by definition not scan-local.

Dispatches on the step INSTANCE, not just its type, so a step whose
configuration decides the answer can answer for itself.
"""
fusable_grouping(step::SolveStep) = :global

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
own `StepSolution.info` (`stage_info(sol, name)`).

`ctx.scratch[:pass_results]` holds the collected per-group results:
`(; index, decode, work, r)` per selected group, in no particular order — `index` is the group's true position, and
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

`θ` is the flat parameter vector over `layout`; a component's block is
`reshape(view(θ, plan.range), plan.shape)`. Once a step is finished and wrapped in a [`CalibrationSolution`](@ref),
[`parameters`](@ref) wraps its components' blocks as labelled, dimensioned
`DimArray`s on demand.
"""
mutable struct SolveContext{
        M <: GainModel, L <: ParameterLayout,
        A <: UVData.AntennaTable, S <: Streaming.ScanStream,
        V <: AbstractVector{Float64},
    }
    model::M
    layout::L
    geom::DataGeometry
    θ::V
    gauge::AbstractGauge
    nant::Int
    antennas::A
    stream::S
    scratch::Dict{Symbol, Any}
end

# ── Pipelines ────────────────────────────────────────────────────────────────

# What a pipeline may hold: solve steps, corrections applied to the data every
# later step reads, and output-only a-priori calibration.
const PipelineElement = Union{SolveStep, Fring.AbstractDataTransform, AprioriAmplitude}

"""
    a |> b

Build a pipeline, the tuple `(a, b)`, from solve steps and corrections;
`pipeline |> c` appends and `c |> pipeline` prepends:
`StationWeightScale(ws) |> BaselineFringeFit() |> Bandpass()`. Two pipelines
join by splatting, `(p..., q...)`.
"""
Base.:|>(a::PipelineElement, b::PipelineElement) = (a, b)
Base.:|>(a::Tuple, b::PipelineElement) = (a..., b)
Base.:|>(a::PipelineElement, b::Tuple) = (a, b...)

function _check_pipeline(seq)
    for x in seq
        x isa PipelineElement || throw(
            ArgumentError(
                "a pipeline holds solve steps, data transforms and AprioriAmplitude, " *
                    "not $(nameof(typeof(x)))"
            )
        )
    end
    return seq
end
