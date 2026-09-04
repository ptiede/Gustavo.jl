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
`FITSFiles` to enable them. Loading `HDF5` enables the language-neutral
caltable writer ([`save_solution_hdf5`](@ref Gustavo.Calibration.save_solution_hdf5)),
and loading `CairoMakie` (or another Makie backend) enables the diagnostic plots.

## A calibration run

```julia
using Gustavo
using FITSFiles

uvset = load_fitsidi("track.idifits")            # lazy: header tables only

pipe = CalibrationPipeline(
    FringeFit() |> DispersionSBDFit() |> Bandpass() |> TemporalSmoother();
    gauge = PinAntenna("AA"),                    # run-wide reference antenna
)

sol, out = fitcalibrate(
    pipe, uvset;
    reduce = [AverageFrequency(nout = 1), CombineSpw(), AverageTime(seconds = 10.0)],
)

write_uvfits("track_cal.uvfits", out)
save_solution("track.jls", sol)
```

## The pieces

**Data.** [`load_fitsidi`](@ref) / [`load_uvfits`](@ref) return a `UVSet`: a
tree of per-scan leaves, each carrying dimension-named
`(Ti, Baseline, Pol, Frequency)` visibility cubes. The `UV_DATA` payload
stays on disk until a scan group is materialized, so the solvers stream it.

**Pipeline.** A [`CalibrationPipeline`](@ref) is an ordered list of steps
chained with `|>`. The built-in solve steps are [`FringeFit`](@ref) (delay /
rate / phase search), [`DispersionSBDFit`](@ref) (ionospheric dTEC and
per-band-group delay refinement), [`Bandpass`](@ref) (time-stable station
bandpass), and [`TemporalSmoother`](@ref) (per-integration atmospheric
phase). Data reductions ([`AverageFrequency`](@ref), [`AverageTime`](@ref),
[`CombineSpw`](@ref), [`FlagSpwEdges`](@ref)) and a-priori amplitude
calibration ([`AprioriAmplitude`](@ref)) compose into the same pipeline.
Every step is optional and reorderable; a single standalone step is a legal
pipeline.

**Models.** Each solve step separates WHAT it solves — a gain model, built
from the vocabulary in [Specifying gain models](@ref specifying-models) —
from HOW it is solved (a pluggable estimator or smoother object on the step).

**Verbs.** [`fit`](@ref) solves and returns a
[`CalibrationSolution`](@ref Gustavo.Calibration.CalibrationSolution) without
producing corrected data; [`calibrate`](@ref) applies a finished solution to
this or another dataset with the same geometry; [`fitcalibrate`](@ref) does
both in one streaming run and is the production path.

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
  [`AbstractFringeEstimator`](@ref Gustavo.Fringe.AbstractFringeEstimator) and
  [`AbstractBandpassSmoother`](@ref Gustavo.Fringe.AbstractBandpassSmoother);
- a new **pipeline step** — a solver with its own streaming pass:
  [Authoring a pipeline step](@ref authoring-steps).
