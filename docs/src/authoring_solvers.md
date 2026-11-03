```@meta
CurrentModule = Gustavo
```

# [Authoring a solver](@id authoring-solvers)

A solver reads the data itself: it declares the gain
model it solves and fits it, reading each scan group through
[`each_group`](@ref) as many times as its solve needs. This page is the solver contract, followed by a
worked example: the shipped per-integration phase solver, [`AdhocPhase`](@ref).

A solver is an [`AbstractSolver`](@ref), which solves gains. Corrections, which
change the data a later fit reads, are functions that modify a Measurement Set
in place and return it, such as [`scale_weights!`](@ref) and
[`calibrate!`](@ref); they are applied to the data before `fit` is called.

## The contract at a glance

An `AbstractSolver` subtype implements [`solve`](@ref) and some of these hooks:

| Hook | Answers | Default |
|:-----|:--------|:--------|
| [`model_components`](@ref) | what gain model does this solver solve? | no components |
| [`solution_name`](@ref) | the solver's solution slot name (`sol[:name]`) | `:nothing` |
| [`solver_gauge`](@ref) | the gauge that fixes the solver's undetermined station values | `nothing` |
| [`solve`](@ref) | fill θ; return diagnostics | required |
| [`supports_antenna_heterogeneity`](@ref) | can the solver loop over ragged station blocks? | `false` |

The execution model behind them:

- **A solver reads scan groups, never the whole set.** `each_group(f, ctx)`
  reads each scan group of the data `fit` was given into memory and calls
  `f(group)`, where `group` is a `ProcessingSet` holding one Measurement Set
  per spectral window of the scan.
  `GeometryWindow(ctx.geom, ms)` addresses a Measurement Set in the solve's
  index space. `each_group` returns `f`'s results in group order. A solve that iterates, such as a residual
  re-search, calls `each_group` once per round.
- **A solver carries its own gauge.** The built-in solvers take a `gauge` keyword,
  `ZeroSumPhase()` by default, and `solver_gauge(solver)` returns it. `fit` resolves the station codes
  of the returned gauge against the run's antenna table and hands the result to
  `solve` as `ctx.gauge`. A solver without a `solver_gauge` method has none:
  `ctx.gauge` is `nothing`.
- **Each solver solves its own private θ.** `model_components(solver, spec)`
  compiles to a per-solver parameter layout; no solver's θ block is shared with
  or visible to another fit's. Gains compose multiplicatively across fits,
  so a solver needing "the same" column as an earlier fit compiles its own
  private copy instead of writing into the other's.
- **`f` may run concurrently across groups.** It returns each scan's
  contribution and `solve` combines the returned vector, so a solver needs no
  locking. Per-scan θ slots are disjoint, so `f` may also write the θ slots
  of its own scan directly.

## Worked example: the adhoc solver

[`AdhocPhase`](@ref) solves a per-integration station phase on data the
fringe (and bandpass) solutions have already corrected. A new gain term, if an effect needs one, is authored
separately (see [Authoring a new gain term](@ref authoring-terms)); this solver
uses the ordinary `ConstantTerm`.

### The declaration and its model

The solver carries the `GainModel` it solves and the smoother that solves it:

```julia
Base.@kwdef struct AdhocPhase{M <: GainModel, S <: Fring.AbstractAdhocSmoother} <: AbstractSolver
    model::M = Fring.default_adhoc_model()
    smoother::S = Fring.PerTrackAdhocSmoother()
end
solution_name(::AdhocPhase) = :adhoc
```

`model_components` returns that model after vetting it against the solver,
before any data is read: every component must pass
[`can_fit`](@ref Gustavo.Fring.can_fit) for the smoother, and the whole tree
must pass [`validate_model`](@ref Gustavo.Fring.validate_model). A rejected
component throws, naming what the solver accepts, rather than leaving a θ
block at zero:

```julia
model_components(s::AdhocPhase, spec) = _vet_model(
    s.smoother, s.model,
    "The adhoc smoothers fit `GainComponent(ConstantTerm(); Ti = PerIntegration(), " *
        "Frequency = GlobalFrequency(), Feed = SharedFeeds() or PerFeed())` ...",
    spec,
)

can_fit(::AbstractAdhocSmoother, tc, geom) =
    tc.term isa ConstantTerm && tc.Ti isa PerIntegration &&
    tc.Frequency isa GlobalFrequency && (tc.Feed isa PerFeed || tc.Feed isa SharedFeeds) &&
    _is_time_prior(resolve_prior(tc))
```

### The solve

`solve` locates the solver's θ block once, then solves each scan group. The
adhoc phase is per scan, so each group writes only its own θ slots and the
per-group function writes θ directly. It fixes the per-scan constant with
`ctx.gauge`, the solver's own gauge. The returned `NamedTuple` is the solver's
diagnostics:

```julia
function solve(s::AdhocPhase, ctx::SolveContext)
    adhoc_plan = Fring._adhoc_plan(ctx.model, ctx.layout)
    results = each_group(ctx) do group
        Fring.adhoc_scan!(
            ctx.θ, group, ctx.geom, adhoc_plan, s.smoother, ctx.gauge;
            executor = inner_executor(ctx.exec),
        )
    end
    isempty(results) && return (; nscans = 0)
    sort!(results; by = r -> only(lookup(r, Ti)))
    return (; nscans = length(results), priors = cat(results...; dims = Ti))
end
```

`_adhoc_plan` is a **signature router**: a `findfirst` over the compiled
components' types (here, the one with a `PerIntegration` time segmentation),
never a hardcoded index. It only has to be unambiguous within the solver's own
model, which `can_fit` has already restricted. A solver whose model names the
component it needs can skip routers and reach it through
`ctx.layout.plantree`.

## Diagnostics and logging

The `NamedTuple` `solve` returns **is** the solver logging interface: every key
lands in the solution's diagnostics, `sol.info`. The runner adds `t_pass` and `timing`, a `DimStack` over `Scan` (scan names). A per-scan quantity
is a vector built from `each_group`'s results, which are already in scan-group
order:

```julia
results = each_group(group -> residual_rms(group, ctx.geom), ctx)
return (; residual_rms = results)
```

## Station heterogeneity

By default a solver receives station-uniform models only: the runner rejects a
model whose component signatures differ across stations, naming the differing
component, so an undeclared solver never gets θ blocks it would silently
leave unsolved. A solver that can handle raggedness declares

```julia
supports_antenna_heterogeneity(::MySolver) = true
```

and writes its solve as a loop over
[`antenna_blocks`](@ref Gustavo.Calibration.antenna_blocks)`(layout, θ, :phase, :name)`
— each block is `(; antennas, θ, plan)`: the global station indices it spans,
a shaped view into θ, and the block's own segment tables. A uniform model
yields exactly one block spanning every station, so the loop costs nothing in
the common case. Gauge caution: the blocks of one component share a *single*
physical degeneracy — place the gauge constraint once across the union of the
blocks' stations, not per block.

## Selecting scans

A fit reads every scan of the data it is given; there is no per-fit
selection. To fit a subset of the scans, fit that subset, then
divide its solution out of the full data before the next fit:

    fr = fit(FringeClosure(; gauge), fit(FringeSearch(), data))
    bp = fit(Bandpass(; gauge), calibrate(fr, calibrator_scans; flag_bad = false, apply_flags = false))
    corrected = calibrate(fr, data; flag_bad = false, apply_flags = false)
    calibrate!(bp, corrected; flag_bad = false, apply_flags = false)
    ad = fit(AdhocPhase(; gauge), corrected)
