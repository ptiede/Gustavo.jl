```@meta
CurrentModule = Gustavo
```

# Gustavo

Gustavo is a modular VLBI fringe-fitting and station-gain calibration package.
It solves an ordered pipeline of calibration steps on MSv4 data, an XRadio
`ProcessingSet`, reading one scan group at a time, so a full-track dataset is
never resident in memory, and applies the solution to the data.

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
using XRadio

ps = open(ProcessingSet, "track.ps.zarr")       # lazy: no visibilities read

pipeline = AutocorrelationNormalization() |> BaselineFringeFit() |>
    DispersionSBDFit() |> Bandpass() |> AdhocPhase()
sol = fit(pipeline, ps; gauge = PinAntenna("AA"))   # run-wide reference antenna

out = calibrate(sol, ps)                         # corrected, in memory
save_solution("track.jls", sol)
```

## The pieces

**Data.** A `ProcessingSet` holds Measurement Sets, each one spectral window
with dimension-named visibility, weight and flag layers. `fit` groups it by
scan (`groupby(ps, ByScan())`) and reads one group at a time, so an opened
store stays on disk until a step reads it. Narrow the data by subsetting the
`ProcessingSet` before fitting.

**Pipeline.** A pipeline is a tuple of solve steps and corrections, usually
built with `|>`, run in order. The built-in solve steps are [`BaselineFringeFit`](@ref) (delay /
rate / phase search), [`DispersionSBDFit`](@ref) (ionospheric dTEC and
per-band-group delay refinement), [`Bandpass`](@ref) (time-stable station
bandpass), and [`AdhocPhase`](@ref) (per-integration atmospheric
phase). Every step is optional and reorderable; a single standalone step is
a legal pipeline.

**Corrections.** A step solves gains; a correction changes what every step
after it *reads*. A correction is a function from a Measurement Set to a
corrected Measurement Set, applied to each Measurement Set of a scan group as
the group is read. The ones a solution records and [`calibrate`](@ref)
replays are [`AbstractDataTransform`](@ref) structs:
[`AutocorrelationNormalization`](@ref), [`ApplySolution`](@ref),
[`StationWeightScale`](@ref) and [`FlagChannels`](@ref). A plain function
can sit in a pipeline too, written as a tuple or vector
(`(my_flagging, BaselineFringeFit())`), since `f |> step` is Base's function
application.

**Models.** Each solve step separates WHAT it solves — a gain model, built
from the vocabulary in [Specifying gain models](@ref specifying-models) —
from HOW it is solved (the step's options, or a pluggable smoother object).

**Verbs.** [`fit`](@ref) solves and returns a
[`CalibrationSolution`](@ref Gustavo.Calibration.CalibrationSolution) without
producing corrected data; [`calibrate`](@ref) applies a finished solution to
this or other data, replaying the recorded corrections and each step's gains
in order, and runs its `post` function on each corrected Measurement Set.

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
- a new **smoother** behind an existing step — a different way to solve the
  same model: see
  [`AbstractBandpassSmoother`](@ref Gustavo.Fring.AbstractBandpassSmoother);
- a new **pipeline step** — a solver that reads the data through
  [`each_group`](@ref Gustavo.each_group):
  [Authoring a pipeline step](@ref authoring-steps).
