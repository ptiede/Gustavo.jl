# Gustavo

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/Gustavo.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/Gustavo.jl/dev/)
[![Build Status](https://github.com/ptiede/Gustavo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/Gustavo.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/Gustavo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/Gustavo.jl)

Gustavo is a modular VLBI fringe-fitting and station-gain calibration package
for radio interferometry. It solves an ordered pipeline of calibration steps
on MSv4 data (an XRadio `ProcessingSet`) — fringe search (delay/rate/phase),
ionospheric dispersion and single-band delay refinement, station bandpass,
per-integration atmospheric phase — reading one scan group at a time, so a
full-track dataset is never resident in memory, and applies the solution to
the data.

Gustavo is experimental and unregistered: the API changes freely and without
deprecation. (It also operates entirely on vibes and fried chicken.)

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

Every step is optional and reorderable — a pipeline can equally be a single
`Bandpass()` fit over data an earlier run already corrected. `fit` solves
without producing output; `calibrate(sol, ps)` applies a finished solution to
this or other data, replaying the pipeline's corrections and each step's gains
in order.

The solution is inspectable per stage: `sol[:fringe]` selects one step (any
selection is itself a solution), `gains(sol[:bandpass])` evaluates its complex
station gains as a labelled `DimArray`, `parameters(sol[:fringe, :phase, :mbd])`
shows the solved θ of one model component, and `stage_info(sol, :fringe)`
returns the step's diagnostics.

## What is pluggable

Each step separates WHAT it solves (a gain model: named components, each a
term at a time/frequency resolution with a feed tying) from HOW it is solved
(the step's options, or a smoother object). Third-party code can add

- a new **gain term** (a physical effect in the forward model) — six small
  methods;
- a new **pipeline step** — a `solve` method that reads the data through
  `each_group`;
- a new **bandpass** or **adhoc smoother** behind an existing step.

The [documentation](https://ptiede.github.io/Gustavo.jl/dev/) has a worked
example for each, plus the model-specification vocabulary
(per-station heterogeneous models included).
