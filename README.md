# Gustavo

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/Gustavo.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/Gustavo.jl/dev/)
[![Build Status](https://github.com/ptiede/Gustavo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/Gustavo.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/Gustavo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/Gustavo.jl)

Gustavo is a modular VLBI fringe-fitting and station-gain calibration package
for radio interferometry. Data are MSv4: an XRadio.jl `ProcessingSet` of
`MeasurementSet`s, each holding one spectral window. Gustavo fits
calibration steps on them (fringe search for delay, rate and phase; ionospheric
dispersion and single-band delay; station bandpass; per-integration
atmospheric phase), one scan group at a time, and applies the solutions to the
data.

Gustavo is experimental and unregistered: the API changes freely and without
deprecation. (It also operates entirely on vibes and fried chicken.)

## Getting data in and out

Loading `FITSFiles` enables the FITS readers and writers of both packages:

```julia
using Gustavo, XRadio
using FITSFiles

fitsidi2msv4("track.idifits", "track.ps.zarr")  # FITS-IDI to an MSv4 Zarr store
ps = open(ProcessingSet, "track.ps.zarr")       # lazy: no visibilities read
read_antab!(ps, "track.antab")                  # Tsys and gain curves, when the file has none

ps = load_uvfits("track.uvfits")                # AIPS UVFITS, read into memory
write_uvfits("out.uvfits", ps)                  # one source, frequency setup and subarray
```

`apriori_calibrate!(ms)` puts one Measurement Set's visibilities in janskys
from the system temperatures and gain curves it records. Like every correction
it modifies a Measurement Set held in memory, so it runs member by member, for
example `foreach(apriori_calibrate!, values(g))` inside the `mapsets` bodies
below.

## A calibration run

`fit(step, data)` solves one step; `calibrate!(sol, data)` divides a solution's
gains out of data in memory, and `calibrate(sol, data)` does the same to a copy.
To fit a step on data an earlier solution has corrected, correct the data
first. `mapsets` does this one scan group at a time, holding only that group
in memory:

```julia
using Gustavo
using XRadio

ps = open(ProcessingSet, "track.ps.zarr")       # lazy: no visibilities read

gauge = PinAntenna("AA")                        # reference antenna
sols = mapsets(groupby(ps, ByScan())) do g      # g: one scan, in memory
    foreach(normalize_by_autocorrelations!, values(g))
    fr = fit(BaselineFringeFit(; gauge), g)
    calibrate!(fr, g; flag_bad = false, apply_flags = false)
    ad = fit(AdhocPhase(; gauge), g)
    return (; fr, ad)
end
```

A bandpass is fit over the whole track: each scan group is fringe-corrected
and averaged over the scan (`average(g, ByScan())`, one sample per scan), and
the bandpass is fit on the averaged groups, so one scan group is in memory at
a time. The bandpass solution applies to the unaveraged data:

```julia
averaged = mapsets(groupby(ps, ByScan())) do g
    foreach(normalize_by_autocorrelations!, values(g))
    fr = fit(BaselineFringeFit(; gauge), g)
    calibrate!(fr, g; flag_bad = false, apply_flags = false)
    return average(g, ByScan())
end
bp = fit(Bandpass(; gauge), ProcessingSet(averaged))

sols = mapsets(groupby(ps, ByScan())) do g
    foreach(normalize_by_autocorrelations!, values(g))
    calibrate!(bp, g; flag_bad = false, apply_flags = false)
    fr = fit(BaselineFringeFit(; gauge), g)
    calibrate!(fr, g; flag_bad = false, apply_flags = false)
    return (; fr, ad = fit(AdhocPhase(; gauge), g))
end
```

Each solve step takes its own `gauge`, which fixes the station values the data
leave undetermined. `fit` never modifies the data it is given. `flag_bad =
false, apply_flags = false` divides out the gains without flagging, as suits
data handed to a later fit; the defaults also flag samples whose gain is
degenerate and the baselines of stations the fringe fit left unconstrained.

A solution is inspectable per component: `fr[:fringe, :phase, :mbd]` selects
the components under a path (any selection is itself a solution),
`gains(bp)` evaluates the complex station gains as a labeled `DimArray`, and
`fr.steps[:fringe]` holds the step's diagnostics.

`mapsets` returns its results keyed like its input, one per scan group here.
A collection of per-scan solutions saves to one store and loads back with the
same keys:

```julia
using OrderedCollections: OrderedDict

save_solution("fringe.sol.zarr", OrderedDict(k => s.fr for (k, s) in sols))
fringe = load_solution("fringe.sol.zarr")       # OrderedDict: scan group => solution
```

## What is pluggable

Each step separates WHAT it solves (a gain model: named components, each a
term at a time/frequency resolution with a feed tying) from HOW it is solved
(the step's options, or a smoother object). Third-party code can add

- a new **gain term** (a physical effect in the forward model) — six
  methods;
- a new **solve step** — a `solve` method that reads the data through
  `each_group`;
- a new **bandpass** or **adhoc smoother** behind an existing step.

The [documentation](https://ptiede.github.io/Gustavo.jl/dev/) has a worked
example for each, plus the model-specification vocabulary
(per-station heterogeneous models included).
