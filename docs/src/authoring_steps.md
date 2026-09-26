```@meta
CurrentModule = Gustavo
```

# [Authoring a pipeline step](@id authoring-steps)

A pipeline step is a solver that reads the data itself: it declares the gain
model it solves and fits it, reading each scan group through
[`each_group`](@ref) as many times as its solve needs. This page is the step contract, followed by a
worked example: the shipped per-integration phase step, [`AdhocPhase`](@ref).

A step is a [`SolveStep`](@ref), which solves gains. Corrections, which
change what later steps read, are functions from a Measurement Set to a
Measurement Set ([`AbstractDataTransform`](@ref)) instead.

## The contract at a glance

A `SolveStep` subtype implements [`solve`](@ref) and some of these hooks:

| Hook | Answers | Default |
|:-----|:--------|:--------|
| [`model_components`](@ref) | what gain model does this step solve? | no components |
| [`provides`](@ref) | the step's solution slot name (`sol[:name]`) | `:nothing` |
| [`solve`](@ref) | fill θ; return diagnostics | required |
| [`supports_station_heterogeneity`](@ref) | can the solver loop over ragged station blocks? | `false` |

The execution model behind them:

- **A step reads scan groups, never the whole set.** `each_group(f, ctx)`
  reads each scan group into memory through the corrections before the step
  (so the step reads data already corrected by every earlier step's finished
  solution) and calls `f(group)`, where `group` is a `ProcessingSet` holding
  one Measurement Set per spectral window of the scan.
  `GeometryWindow(ctx.geom, ms)` addresses a Measurement Set in the solve's
  index space. `each_group` returns `f`'s results in group order. A solve that iterates, such as a residual
  re-search, calls `each_group` once per round.
- **Each step solves its own private θ.** `model_components(step, spec)`
  compiles to a per-step parameter layout; no step's θ block is shared with
  or visible to another step's. Gains compose multiplicatively across steps,
  so a step needing "the same" column as an earlier step compiles its own
  private copy instead of writing into the other's.
- **`f` may run concurrently across groups.** It returns each scan's
  contribution and `solve` combines the returned vector, so a step needs no
  locking. Per-scan θ slots are disjoint, so `f` may also write the θ slots
  of its own scan directly.

## Worked example: the adhoc step

[`AdhocPhase`](@ref) solves a per-integration station phase on the residual
of the steps before it. A new gain term, if an effect needs one, is authored
separately (see [Authoring a new gain term](@ref authoring-terms)); this step
uses the ordinary `ConstantTerm`.

### The declaration and its model

The step carries the `GainModel` it solves and the solver options:

```julia
Base.@kwdef struct AdhocPhase{M <: GainModel, S <: Fring.AbstractAdhocSmoother} <: SolveStep
    model::M = Fring.default_adhoc_terms()
    smoother::S = Fring.SavitzkyGolaySmoother()
end
provides(::AdhocPhase) = :adhoc
```

`model_components` returns that model after vetting it against the solver,
before any data is read: every component must pass
[`can_fit`](@ref Gustavo.Fring.can_fit) for the smoother, and the whole tree
must pass [`validate_model`](@ref Gustavo.Fring.validate_model). A rejected
component throws, naming what the solver accepts, rather than leaving a θ
block at zero:

```julia
model_components(s::AdhocPhase, spec) = _vet_step_model(
    s.smoother, s.model,
    "The adhoc smoothers fit `GainComponent(ConstantTerm(); Ti = PerIntegration(), " *
        "Frequency = GlobalFrequency(), Feed = SharedFeeds() or PerFeed())` ...",
    spec,
)

can_fit(::AbstractAdhocSmoother, tc, geom) =
    tc.term isa ConstantTerm && tc.Ti isa PerIntegration &&
    tc.Frequency isa GlobalFrequency && (tc.Feed isa PerFeed || tc.Feed isa SharedFeeds)
```

### The solve

`solve` locates the step's θ block once, then solves each scan group. The
adhoc phase is per scan, so each group writes only its own θ slots and the
per-group function writes θ directly. The returned `NamedTuple` is the step's
diagnostics:

```julia
function solve(s::AdhocPhase, ctx::SolveContext)
    adhoc_plan = Fring._adhoc_plan(ctx.model, ctx.layout)
    results = each_group(ctx) do group
        Fring.adhoc_scan!(
            ctx.θ, group, ctx.geom, adhoc_plan, s.smoother, ctx.gauge, ctx.nant;
            executor = inner_executor(ctx.exec),
        )
    end
    return (; nscans = length(results))
end
```

`_adhoc_plan` is a **signature router**: a `findfirst` over the compiled
components' types (here, the one with a `PerIntegration` time segmentation),
never a hardcoded index. It only has to be unambiguous within the step's own
model, which `can_fit` has already restricted. A step whose model names the
component it needs can skip routers and reach it through
`ctx.layout.plantree`.

## Diagnostics and logging

The `NamedTuple` `solve` returns **is** the step logging interface: every key
lands on the step's own `StepSolution.info`, readable via
[`stage_info`](@ref Gustavo.Calibration.stage_info)`(sol, :adhoc)`. The
runner adds `t_pass` and a per-scan `timing` `DimStack`. A per-scan quantity
is a vector built from `each_group`'s results, which are already in scan-group
order:

```julia
results = each_group(group -> residual_rms(group, ctx.geom), ctx)
return (; residual_rms = results)
```

## Station heterogeneity

By default a step receives station-uniform models only: the runner rejects a
model whose component signatures differ across stations, naming the differing
component, so an undeclared solver never gets θ blocks it would silently
leave unsolved. A step whose solver can handle raggedness declares

```julia
supports_station_heterogeneity(::MyStep) = true
```

and writes its solve as a loop over
[`station_blocks`](@ref Gustavo.Calibration.station_blocks)`(layout, θ, :phase, :name)`
— each block is `(; stations, θ, plan)`: the global station indices it spans,
a shaped view into θ, and the block's own segment tables. A uniform model
yields exactly one block spanning every station, so the loop costs nothing in
the common case. Gauge caution: the blocks of one component share a *single*
physical degeneracy — place the gauge constraint once across the union of the
blocks' stations, not per block.

## Selecting scans

A step reads every scan of the data `fit` is given; there is no per-step
selection. To fit a step on a subset of the scans, fit it on that subset and
carry its solution into the full-data fit as a correction:

    fr = fit(BaselineFringeFit(), data; gauge)
    bp = fit(ApplySolution(fr) |> Bandpass(), calibrator_scans; gauge)
    sol = fit(ApplySolution(fr) |> ApplySolution(bp) |> AdhocPhase(), data; gauge)
