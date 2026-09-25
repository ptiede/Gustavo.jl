```@meta
CurrentModule = Gustavo
```

# Gustavo

Gustavo is a modular VLBI fringe-fitting and station-gain calibration package.
It reads FITS-IDI or UVFITS data into a lazy, scan-partitioned [`UVSet`](@ref),
solves an ordered pipeline of calibration steps, and streams corrected,
reduced visibilities back out one scan group at a time — a full-track dataset
is never resident in memory.

Gustavo is experimental and unregistered: the API changes freely and without
deprecation.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ptiede/Gustavo.jl")
```

The FITS-IDI/UVFITS reader and writer live in a package extension: also load
`FITSFiles` to enable them. Loading `CairoMakie` (or another Makie backend)
enables the diagnostic plots.

## A calibration run

```julia
using Gustavo
using FITSFiles

uvset = load_fitsidi("track.idifits")            # lazy: header tables only

pipeline = BaselineFringeFit() |> DispersionSBDFit() |> Bandpass() |> AdhocPhase()
sol = fit(pipeline, uvset; gauge = PinAntenna("AA"))   # run-wide reference antenna

out = calibrate(
    sol, uvset;
    post = AverageTime(seconds = 10.0) ∘ CombineSpw() ∘ AverageFrequency(nout = 1),
)

write_uvfits("track_cal.uvfits", out)
save_solution("track.jls", sol)
```

## The pieces

**Data.** [`load_fitsidi`](@ref) / [`load_uvfits`](@ref) return a `UVSet`: a
tree of per-scan leaves, each carrying dimension-named
`(Ti, BaselineID, Polarization, Frequency)` visibility cubes. The `UV_DATA` payload
stays on disk until a scan group is materialized, so the solvers stream it.

**Pipeline.** A pipeline is a tuple of steps, usually built with `|>`,
solved in order. The built-in solve steps are [`BaselineFringeFit`](@ref) (delay /
rate / phase search), [`DispersionSBDFit`](@ref) (ionospheric dTEC and
per-band-group delay refinement), [`Bandpass`](@ref) (time-stable station
bandpass), and [`AdhocPhase`](@ref) (per-integration atmospheric
phase). A-priori amplitude calibration ([`AprioriAmplitude`](@ref)) joins
the same pipeline and scales the output. Every step is optional and
reorderable; a single standalone step is a legal pipeline.

**Transforms.** A step scales what it *produces*; an
[`AbstractDataTransform`](@ref Gustavo.Fring.AbstractDataTransform) scales what
every step after it *reads* — it runs on each scan group as it is
materialized, inside the streaming pass. Chain one into a pipeline like any step
(`AprioriPreCal(uvset, antab) |> Bandpass()`). The built-ins are
[`AprioriPreCal`](@ref) (ANTAB SEFD scaling before the solve, the pre-fit
counterpart of [`AprioriAmplitude`](@ref)), [`ApplySolution`](@ref),
[`StationWeightScale`](@ref), [`FlagChannels`](@ref), and
[`CalFunction`](@ref) for arbitrary caller code. Writing your own means one
method, `apply_transform!(t, stack, win; executor)`.

**Models.** Each solve step separates WHAT it solves — a gain model, built
from the vocabulary in [Specifying gain models](@ref specifying-models) —
from HOW it is solved (a pluggable estimator or smoother object on the step).

**Verbs.** [`fit`](@ref) solves and returns a
[`CalibrationSolution`](@ref Gustavo.Calibration.CalibrationSolution) without
producing corrected data; [`calibrate`](@ref) applies a finished solution to
this or another dataset with the same geometry, replaying the recorded
transforms and a-priori steps, and runs its `post` function — averaging
([`AverageFrequency`](@ref), [`AverageTime`](@ref)), spw merging
([`CombineSpw`](@ref)), edge flagging ([`FlagSpwEdges`](@ref)) — on each
corrected scan group.

**Solutions.** A solution is inspectable per stage: `sol[:fringe]` selects
one step (any selection is itself a solution that applies, plots, and
differences like the whole), [`gains`](@ref Gustavo.Calibration.gains)
evaluates a selection's complex station gains as a labelled `DimArray`,
[`parameters`](@ref Gustavo.Calibration.parameters) shows the solved θ, and
[`stage_info`](@ref Gustavo.Calibration.stage_info) returns a step's
diagnostics. [`save_solution`](@ref Gustavo.Calibration.save_solution) /
[`load_solution`](@ref Gustavo.Calibration.load_solution) round-trip it.

## Extending Gustavo

Three seams, in increasing scope:

- a new **gain term** — a physical effect in the forward model:
  [Authoring a new gain term](@ref authoring-terms);
- a new **estimator or smoother** behind an existing step — a different way
  to solve the same model: see
  [`AbstractFringeEstimator`](@ref Gustavo.Fring.AbstractFringeEstimator) and
  [`AbstractBandpassSmoother`](@ref Gustavo.Fring.AbstractBandpassSmoother);
- a new **pipeline step** — a solver with its own streaming pass:
  [Authoring a pipeline step](@ref authoring-steps).
