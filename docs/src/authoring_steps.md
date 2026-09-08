```@meta
CurrentModule = Gustavo
```

# [Authoring a pipeline step](@id authoring-steps)

A pipeline step is a solver with its own streaming pass: it declares the gain
model it solves, accumulates from each materialized scan group, and closes its
solve when the pass completes. This page is the step contract, followed by a
worked end-to-end example — adding a new physical effect with a specialized
solver — using the shipped ionospheric-dispersion step as the model.

Two kinds of step exist. A [`ReduceStep`](@ref) is a `UVSet -> UVSet` data
transform (averaging, spw merging); it implements one method,
[`prepare_reducer`](@ref), and is not covered further here. A
[`SolveStep`](@ref) solves gains; the rest of this page is about those.

## The contract at a glance

A `SolveStep` subtype implements some of these hooks — every one has a
working default:

| Hook | Answers | Default |
|:-----|:--------|:--------|
| [`model_components`](@ref) | what gain model does this step solve? | no components |
| [`provides`](@ref) | the step's solution slot name (`sol[:name]`) | `:nothing` |
| [`start_pass!`](@ref) | allocate accumulators before the pass | no-op |
| [`process_scan!`](@ref) | accumulate one scan group | no-op |
| [`finish_pass!`](@ref) | close the solve; return diagnostics | empty |
| [`fusable_grouping`](@ref) | can the step share a pass with its neighbors? | `:global` |
| [`required_grouping`](@ref) | leaf-grouping constraint | `:any` |
| [`fit_selection`](@ref) | which scans feed the accumulation | all scans |
| [`scan_flags`](@ref) | per-scan unconstrained (station, scan) pairs | none |
| [`supports_station_heterogeneity`](@ref) | can the solver loop over ragged station blocks? | `false` |
| [`transforms`](@ref Gustavo.transforms) | data transforms the step contributes | none |

The execution model behind them:

- **The executor owns all streaming.** A step never sees the `UVSet`; the
  runner materializes each scan group through the pipeline's transform chain
  (so the step reads data already corrected by every earlier step's finished
  solution) and hands the step a `DimStack` plus the `GeometryWindow`
  addressing it in the solve's index space.
- **Each step solves its own private θ.** `model_components(step, spec)`
  compiles to a per-step parameter layout; no step's θ block is shared with
  or visible to another step's. Gains compose multiplicatively across steps,
  so a step needing "the same" column as an earlier step compiles its own
  private copy instead of writing into the other's (the worked example's
  `delay_refine` column below).
- **`process_scan!` may run concurrently across groups.** Its return value is
  collected by the runner — one entry per selected group — and delivered to
  `finish_pass!` via `ctx.scratch[:pass_results]`, so a step needs no
  locking: return each scan's contribution and fold at the end. Per-scan θ
  slots are disjoint, so a scan-local step may also write θ directly.

## Pass fusion: `fusable_grouping`

Declaring `fusable_grouping(step) = :scan` promises the step is finalizable
from one scan group alone: every θ slot it writes for a scan is written from
that scan's data by the time `process_scan!` returns. Consecutive `:scan`
steps then share **one** streaming pass — on a lazy dataset, the difference
between N reads of the data and one.

The hook dispatches on the step *instance*, so configuration can change the
answer. The shipped `FringeFit` is the example:

```julia
fusable_grouping(s::FringeFit) =
    Fringe.scan_local_solve(s.estimator, s.model) ? :scan : :global
```

With the default [`MatchedFilter`](@ref Gustavo.Fringe.MatchedFilter) (one
round) and an all-per-scan term list, each scan's station systems close
inside `process_scan!` and the step fuses. Setting `MatchedFilter(rounds = 2)`
— a residual re-search — needs the whole pass finished before the next round
can start, so it flips the step to `:global` and it takes its own pass. Any
track-global model column (`GlobalTime`-tied inter-feed delay) does the same.
When your step's answer depends on its model or solver options, follow this
pattern rather than hardcoding one value.

## Worked example: the dispersion step

The shipped [`DispersionSBDFit`](@ref) step adds a physical effect — the
dispersive ionosphere — that the fringe search cannot fit (over a finite band
the 1/ν curvature is nearly degenerate with a linear delay, so it needs a
joint specialized fit). It exercises every layer of the recipe for adding an
effect with its own solver:

1. a **term** — the effect in the forward gain model;
2. a **wrapper element** — the user-facing configuration, with geometry
   gating;
3. a **step** — the solver, locating its θ columns with signature routers.

### 1. The term

The term is the forward model's side: how the effect contributes phase at one
(channel, time) cell. `Dispersion` is an ordinary gain term (see
[Authoring a new gain term](@ref authoring-terms) for the six hooks):

```julia
struct Dispersion <: AbstractGainTerm end

term_axes(::Dispersion) = (:Frequency,)
param_shapes(::Dispersion, nchan_seg) = (dtec = (),)
freq_coord_state(::Dispersion, geom, fseg_id, nfseg) = geom.f0
freq_coordinate(::Dispersion, f, f0, seg) = DISPERSION_K * (1.0 / f0 - 1.0 / f)
@inline term_eval(::Dispersion, p, x) = p.dtec * x.Frequency
term_label(::Dispersion) = "dtec"
```

Once the θ column is filled — by *any* solver — `evaluate_gains` and every
downstream consumer (correction, diagnostics, HDF5 export) handle the effect
with no further code.

### 2. The wrapper element, with geometry gating

The element is the user's configuration surface: a small struct carrying the
effect's options, compiled to component(s) by a `model_components` method.
This is where data-dependent judgment lives:

```julia
Base.@kwdef struct DispersionModel
    require_band_separation::Bool = true
    colocated_sep::Union{Nothing, Float64} = 1000.0
end

model_components(dm::DispersionModel, spec) =
    _dispersion_enabled(dm, spec.geom) ?
    GainComponent(Dispersion(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds()) : nothing
```

`spec = (; geom, antennas)` carries the data geometry, so the element can
decline to compile at all when the data cannot constrain the effect — here,
when the band layout cannot separate 1/ν from a linear delay
(`_dispersion_enabled` requires several sub-bands over a wide fractional
bandwidth). The specific thresholds are *this element's* judgment call, not
framework policy: your element writes its own gate, or none. An element that
compiles to `nothing` contributes no θ, no solve work, and no provenance — the
machinery has zero footprint until the data supports the effect.

### 3. The step

The step owns the solve. Its declaration surface:

```julia
Base.@kwdef struct DispersionSBDFit{D, S} <: SolveStep
    dispersion::D = Fringe.DispersionModel()
    sbd::S = Fringe.SingleBandDelay()
end
provides(::DispersionSBDFit) = :refine
required_grouping(::DispersionSBDFit) = :scan_complete
fusable_grouping(::DispersionSBDFit) = :scan     # both fits are per-scan
```

Note the step's model surface is its two element fields — deliberately not a
free-form component tree, because the solver is a rigid specialized fit. A
step whose solver genuinely generalizes takes a tree instead (as `Bandpass`
does) and vets it with `can_fit`/`validate_model`.

Its `model_components` compiles the step's private model:

```julia
function model_components(s::DispersionSBDFit, spec)
    dispc = s.dispersion === nothing ? nothing : model_components(s.dispersion, spec)
    sbdc = s.sbd === nothing ? nothing : model_components(s.sbd, spec)
    phase = merge(
        dispc === nothing ? (;) : (;
                delay_refine = GainComponent(Delay(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds()),
                dtec = dispc,
            ),
        sbdc === nothing ? (;) : (; sbd = sbdc),
    )
    return (; phase, logamp = (;))
end
```

`delay_refine` is the private-θ principle in action: the joint (Δτ, dTEC)
fit must update a per-scan delay alongside the dTEC (the two covary), but
that column belongs to *this* step's model, not to `FringeFit`'s — gains
compose multiplicatively, so this step's delay column times `FringeFit`'s is
the same total correction, and neither step touches the other's θ.

`start_pass!` resolves where the step's θ columns live, once, before any data
streams:

```julia
function start_pass!(s::DispersionSBDFit, ctx::SolveContext)
    disp_plan = Calibration._dispersion_plan(ctx.model, ctx.layout)
    delay_plan = disp_plan === nothing ? nothing :
        Fringe._perscan_delay_plan(ctx.model, ctx.layout)
    ctx.scratch[:disp_sbd_setup] = (;
        delay_plan, disp_plan,
        sbd_plans = Fringe._sbd_plans(ctx.model, ctx.layout),
        ties = _dtec_ties(s.dispersion, ctx.antennas),
    )
    return nothing
end
```

The plan lookups are **signature routers**: a `findfirst` over the compiled
components' `(term, Ti, Frequency, Feed)` types —

```julia
_is_dispersion(tc) = tc.term isa Dispersion

function _dispersion_plan(model, layout)
    i = findfirst(_is_dispersion, phase_components(model))
    return i === nothing ? nothing : layout.plans[i]
end
```

— never a hardcoded index or name, so the component may sit anywhere in the
model's order, and its absence (the geometry gate declined) reads back as
`nothing` rather than a wrong column. The router only has to be unambiguous
within the step's *own* private model, which the step itself compiled — that
is what keeps `findfirst` honest. A step whose model names the component it
needs can skip routers entirely and reach it through `ctx.layout.plantree`.

`process_scan!` runs the specialized fits per scan (both write per-scan θ
slots, which are disjoint across groups — hence `fusable_grouping = :scan`),
and `finish_pass!` publishes what was actually solved:

```julia
function finish_pass!(s::DispersionSBDFit, ctx::SolveContext)
    setup = ctx.scratch[:disp_sbd_setup]
    return (;
        nscans = length(ctx.scratch[:pass_results]),
        dispersion_applied = setup.disp_plan !== nothing,
        sbd_applied = setup.sbd_plans !== nothing,
    )
end
```

## Diagnostics and logging

`finish_pass!`'s returned `NamedTuple` **is** the step logging interface:
every key lands on the step's own `StepSolution.info`, readable via
[`stage_info`](@ref Gustavo.Calibration.stage_info)`(sol, :refine)` and
written automatically to [`save_solution_hdf5`](@ref
Gustavo.Calibration.save_solution_hdf5)'s `info/steps/<name>/*` groups
(`Number`, `AbstractVector`, `String`, and nested `NamedTuple`/`DimStack`
values). The runner adds `t_pass` and a per-scan `timing` `DimStack` for
free. Publish per-scan quantities in global scan order with
[`scan_values`](@ref):

```julia
scan_values(res -> res.r.residual_rms, ctx.scratch[:pass_results], ngroups; default = NaN)
```

### Exporting an opaque config: `external_info`

A step (or estimator) may record a custom struct in its diagnostics — the
fringe estimator records its whole `FringeSearch` config, which its
diagnostics replay. The HDF5 exporter cannot write an arbitrary struct; the
seam is [`external_info`](@ref Gustavo.Calibration.external_info): define one
method returning the plain-data form, and the exporter writes it instead of
reporting the entry as omitted:

```julia
Calibration.external_info(s::MyEstimatorConfig) =
    (; window_ns = collect(s.window), algorithm = string(s.algorithm))
```

Without a method, the entry is omitted from the HDF5 `info/*` groups (with a
report) and survives only in the file's Julia blob.

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

[`fit_selection`](@ref)`(step, prior_solutions)` restricts which scans feed
the accumulation (fit-on-subset / apply-everywhere — a bandpass fit from a
few bright calibrator scans still applies to every scan). `prior_solutions`
is the ordered list of earlier steps' finished `StepSolution`s: the supported
way to read non-data info from an earlier step, e.g. a
[`ScanWhere`](@ref Gustavo.Fringe.ScanWhere) selection keeping scans whose
fringe SNR cleared a floor. Gain corrections are *never* read this way — they
flow through the stream's transform chain automatically.
