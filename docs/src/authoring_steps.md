```@meta
CurrentModule = Gustavo
```

# [Authoring a pipeline step](@id authoring-steps)

A pipeline step is a solver that reads the data itself: it declares the gain
model it solves and fits it, reading each scan group through
[`each_group`](@ref) as many times as its solve needs. This page is the step contract, followed by a
worked end-to-end example — adding a new physical effect with a specialized
solver — using the shipped ionospheric-dispersion step as the model.

A step is a [`SolveStep`](@ref), which solves gains. Data transforms, which
change what later steps read, are covered by the transform contract
(`apply_transform!`) instead.

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
  materializes each scan group through the pipeline's transform chain (so the
  step reads data already corrected by every earlier step's finished solution)
  and calls `f(stack, win)` with the group's `DimStack` and the
  `GeometryWindow` addressing it in the solve's index space. It returns `f`'s
  results in group order. A solve that iterates, such as a residual
  re-search, calls `each_group` once per round.
- **Each step solves its own private θ.** `model_components(step, spec)`
  compiles to a per-step parameter layout; no step's θ block is shared with
  or visible to another step's. Gains compose multiplicatively across steps,
  so a step needing "the same" column as an earlier step compiles its own
  private copy instead of writing into the other's (the worked example's
  `delay_refine` column below).
- **`f` may run concurrently across groups.** It returns each scan's
  contribution and `solve` combines the returned vector, so a step needs no
  locking. Per-scan θ slots are disjoint, so `f` may also write the θ slots
  of its own scan directly.

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
```

Once the θ column is filled — by *any* solver — `evaluate_gains` and every
downstream consumer (correction, diagnostics) handle the effect
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
    GainComponent(Dispersion(); Ti = PerScan(), Feed = SharedFeeds()) : nothing
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
    dispersion::D = Fring.DispersionModel()
    sbd::S = Fring.SingleBandDelay()
end
provides(::DispersionSBDFit) = :refine
```

Note the step's model surface is its two element fields — deliberately not a
free-form `GainModel`, because the solver is a rigid specialized fit. A
step whose solver genuinely generalizes takes a `GainModel` instead (as
`Bandpass` does) and vets it with `can_fit`/`validate_model`.

Its `model_components` compiles the step's private model:

```julia
function model_components(s::DispersionSBDFit, spec)
    dispc = s.dispersion === nothing ? nothing : model_components(s.dispersion, spec)
    sbdc = s.sbd === nothing ? nothing : model_components(s.sbd, spec)
    phase = merge(
        dispc === nothing ? (;) : (;
                delay_refine = GainComponent(Delay(); Ti = PerScan(), Feed = SharedFeeds()),
                dtec = dispc,
            ),
        sbdc === nothing ? (;) : (; sbd = sbdc),
    )
    return GainModel(; phase)
end
```

`delay_refine` is the private-θ principle in action: the joint (Δτ, dTEC)
fit must update a per-scan delay alongside the dTEC (the two covary), but
that column belongs to *this* step's model, not to `BaselineFringeFit`'s — gains
compose multiplicatively, so this step's delay column times `BaselineFringeFit`'s is
the same total correction, and neither step touches the other's θ.

`solve` resolves where the step's θ columns live, once, before any data is
read, then runs the specialized fits per scan. Both fits write per-scan θ
slots, which are disjoint across groups, so the per-group function writes θ
directly and returns nothing. The returned `NamedTuple` publishes what was
actually solved:

```julia
function solve(s::DispersionSBDFit, ctx::SolveContext)
    disp_plan = Calibration._dispersion_plan(ctx.model, ctx.layout)
    delay_plan = disp_plan === nothing ? nothing :
        Fring._perscan_delay_plan(ctx.model, ctx.layout)
    sbd_plans = Fring._sbd_plans(ctx.model, ctx.layout)
    ties = _dtec_ties(s.dispersion, ctx.antennas)
    results = each_group(ctx) do stack, win
        Fring.refine_scan_dispersion!(
            ctx.θ, stack, win, delay_plan, disp_plan, ctx.gauge, ctx.nant; ties,
        )
        Fring.refine_scan_sbd!(ctx.θ, stack, win, sbd_plans, ctx.gauge, ctx.nant)
        nothing
    end
    return (;
        nscans = length(results),
        dispersion_applied = disp_plan !== nothing,
        sbd_applied = sbd_plans !== nothing,
    )
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

## Diagnostics and logging

The `NamedTuple` `solve` returns **is** the step logging interface: every key
lands on the step's own `StepSolution.info`, readable via
[`stage_info`](@ref Gustavo.Calibration.stage_info)`(sol, :refine)`. The
runner adds `t_pass` and a per-scan `timing` `DimStack`. A per-scan quantity
is a vector built from `each_group`'s results, which are already in scan-group
order:

```julia
results = each_group((stack, win) -> residual_rms(stack, win), ctx)
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
