# Gustavo

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/Gustavo.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/Gustavo.jl/dev/)
[![Build Status](https://github.com/ptiede/Gustavo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/Gustavo.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/Gustavo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/Gustavo.jl)

Gustavo is a modular VLBI fringe-fitting and station-gain calibration package
for radio interferometry. It reads FITS-IDI or UVFITS data into a lazy,
scan-partitioned `UVSet`, solves an ordered pipeline of calibration steps —
fringe search (delay/rate/phase), ionospheric dispersion and single-band
delay refinement, station bandpass, per-integration atmospheric phase — and
streams corrected, reduced visibilities back out, one scan group at a time,
so a full-track dataset is never resident in memory.

Gustavo is experimental and unregistered: the API changes freely and without
deprecation. (It also operates entirely on vibes and fried chicken.)

## A calibration run

```julia
using Gustavo
using FITSFiles   # enables the FITS-IDI/UVFITS reader and writer extension

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
save_solution("track.jls", sol)                  # or save_solution_hdf5 (needs HDF5)
```

Every step is optional and reorderable — a pipeline can equally be a single
`Bandpass()` fit over data an earlier run already corrected. `fit` solves
without producing output; `calibrate(sol, uvset)` applies a finished solution
to this or another dataset with the same geometry.

The solution is inspectable per stage: `sol[:fringe]` selects one step (any
selection is itself a solution), `gains(sol[:bandpass])` evaluates its complex
station gains as a labelled `DimArray`, `parameters(sol[:fringe, :phase, :mbd])`
shows the solved θ of one model component, and `stage_info(sol, :fringe)`
returns the step's diagnostics.

## What is pluggable

Each step separates WHAT it solves (a gain model: named components, each a
term at a time/frequency resolution with a feed tying) from HOW it is solved
(an estimator or smoother object). Third-party code can add

- a new **gain term** (a physical effect in the forward model) — six small
  methods;
- a new **pipeline step** (a solver with its own streaming pass) — the
  `start_pass!`/`process_scan!`/`finish_pass!` visitor contract;
- a new **fringe estimator** or **bandpass smoother** behind an existing step.

The [documentation](https://ptiede.github.io/Gustavo.jl/dev/) has a worked
example for each, plus the model-specification vocabulary
(per-station heterogeneous models included).
