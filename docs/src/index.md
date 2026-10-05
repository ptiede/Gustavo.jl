```@meta
CurrentModule = Gustavo
```

# Gustavo

Gustavo is a modular VLBI fringe-fitting and station-gain calibration package.
Data are MSv4: an XRadio.jl `ProcessingSet` of `MeasurementSet`s, each holding
one spectral window. Gustavo fits calibration steps on them, reading one scan
group at a time, and applies the solutions to the data.

Gustavo is experimental and unregistered: the API changes freely and without
deprecation.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ptiede/Gustavo.jl")
```

Loading `FITSFiles` enables the FITS-IDI converter (`XRadio.fitsidi2msv4`)
and the UVFITS reader and writer ([`load_uvfits`](@ref),
[`write_uvfits`](@ref)), which live in package extensions. Loading
`CairoMakie` (or another Makie backend) enables the diagnostic plots.

## Getting data in and out

```julia
using Gustavo, XRadio
using FITSFiles

fitsidi2msv4("track.idifits", "track.ps.zarr")  # FITS-IDI to an MSv4 Zarr store
ps = open(ProcessingSet, "track.ps.zarr")       # lazy: no visibilities read
read_antab!(ps, "track.antab")                  # Tsys and gain curves, when the file has none

ps = load_uvfits("track.uvfits")                # AIPS UVFITS, read into memory
write_uvfits("out.uvfits", ps)                  # one source, frequency setup and subarray
```

A FITS-IDI file is converted once to a Zarr store, which `open` reads lazily.
A UVFITS file is read whole. [Conventions](@ref conventions) gives the phase
sense, units and weight and flag mapping at each boundary.

## A calibration run

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

Each solve step takes its own `gauge`, which fixes the station values the data
leave undetermined. A bandpass is fit over the
whole track: each scan group is fringe-corrected and averaged over the scan
(`average(g, ByScan())`, one sample per scan), and the bandpass is fit on the
averaged groups, so one scan group is in memory at a time. The bandpass
solution applies to the unaveraged data:

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

Corrected data go to a new store the same way, one scan group at a time: write
a template of the store first, then fill it from inside the body with
[`write!`](@ref), which is safe from concurrent units:

```julia
write(out, ps; data = false, chunks = (; time = 1), schemas = [UVData.GUSTAVO_VISIBILITY_SCHEMA])
dest = open(ProcessingSet, out; mode = "r+")
mapsets(groupby(ps, ByScan())) do g
    calibrate!(bp, g)
    write!(dest, g)
    return nothing
end
```

## The pieces

**Data.** A `ProcessingSet` holds Measurement Sets, each one spectral window
with dimension-named visibility, weight and flag layers. `fit` groups it by
scan (`groupby(ps, ByScan())`) and reads one group at a time, so an opened
store stays on disk until a step reads it. Narrow the data by subsetting the
`ProcessingSet` before fitting.

**Solve steps.** A solve step fits one gain model. The built-in steps are
[`BaselineFringeFit`](@ref) (delay / rate / phase search),
[`Bandpass`](@ref) (time-stable station bandpass), and
[`AdhocPhase`](@ref) (per-integration atmospheric phase). [`fit`](@ref)
solves one step on the data it is given and never modifies that data. To fit
a step on data an earlier solution has corrected, correct the data first,
either on data held in memory or inside [`mapsets`](@ref), which reads one
unit of data (a Measurement Set, or a scan group of a `groupby` result) at a
time and hands it to a function.

**Corrections.** A correction is a function that modifies a Measurement Set in
place and returns it. The built-in ones are
[`normalize_by_autocorrelations!`](@ref Gustavo.UVData.normalize_by_autocorrelations!),
[`scale_weights!`](@ref),
[`apriori_calibrate!`](@ref Gustavo.UVData.apriori_calibrate!), which puts the
visibilities in janskys from the system temperatures and gain curves the
Measurement Set records (XRadio's `read_antab!` adds them from an ANTAB file)
and corrects the correlator's quantization loss from each antenna's
`digitizer_levels` or a `quantization_efficiency` keyword, and [`calibrate!`](@ref), which divides a solution's gains out of the data. Each
applies to one Measurement Set; a `ProcessingSet` is corrected member by
member, `foreach(ms -> scale_weights!(ms, ws), values(ps))`, except
`calibrate!`, which also takes a `ProcessingSet`. Data opened lazily must be
read into memory before it is corrected: [`Gustavo.materialize`](@ref Gustavo.UVData.materialize) reads it with
arrays of its own, so the source is left as it was.

**Flagging.** XRadio's `flags(ms)` is the Measurement Set's own flag array,
so flagging is assignment through DimensionalData's selectors and XRadio's
baseline selectors, which combine freely:

```julia
using DimensionalData, XRadio
F = flags(ms)
F[Frequency = 86.10e9 .. 86.12e9] .= true                 # a frequency range (Hz)
F[Frequency = Begin:(Begin + k - 1)] .= true              # the first k channels
F[Frequency = (End - k + 1):End] .= true                  # the last k channels
F[Frequency = Where(f -> f > 86.2e9)] .= true             # any predicate
F[Ti = t1 .. t2] .= true                                  # a time range
F[baselines(ms, "LA"), Frequency(f1 .. f2)] .= true       # every baseline of a station
F[baseline(ms, ("LA", "PV"))] .= true                     # one baseline
```

A Measurement Set holds one spectral window, so the channel selections act per
window; flag a `ProcessingSet` member by member. A range no channel of a
window falls in selects nothing there. FITSFiles also exports `End`; with it
loaded, write `DimensionalData.End`.

**Models.** Each solve step separates WHAT it solves — a gain model, built
from the vocabulary in [Specifying gain models](@ref specifying-models) —
from HOW it is solved (the step's options, or a pluggable smoother object).

**Verbs.** [`fit`](@ref) solves and returns a
[`CalibrationSolution`](@ref Gustavo.Calibration.CalibrationSolution) without
producing corrected data. [`calibrate!`](@ref) divides a solution's gains out
of data in memory, applies the solution's flags, and runs its `post` function
on each corrected Measurement Set; [`calibrate`](@ref) does the same to a copy
and leaves the data passed to it as it was. `flag_bad = false, apply_flags =
false` divides out the gains without flagging, as suits data handed to a later
fit.

**Solutions.** A solution is a list of solved components
(`sol.components`), each a [`SolvedComponent`](@ref Gustavo.Calibration.SolvedComponent)
holding its step, its path in that step's model, its `GainComponent` and its
parameters as a labeled `DimArray` (`c.params`). Any selection is itself a
solution that applies, plots and differences like the whole: `sol[:fringe]`,
`sol[:fringe, :phase, :mbd]` or `filter(pred, sol)`.
[`gains`](@ref Gustavo.Calibration.gains) evaluates a selection's complex
station gains as a labeled `DimArray`, and `sol.steps[:fringe]` holds a step's
diagnostics. [`save_solution`](@ref Gustavo.Calibration.save_solution) /
[`load_solution`](@ref Gustavo.Calibration.load_solution) round-trip it through
a Zarr store, as they do a collection of solutions such as the per-scan fits
`mapsets` returns (`OrderedDict(k => s.fr for (k, s) in sols)`), keys and order
included.

## Extending Gustavo

Three seams, in increasing scope:

- a new **gain term** — a physical effect in the forward model:
  [Authoring a new gain term](@ref authoring-terms);
- a new **smoother** behind an existing step — a different way to solve the
  same model: see
  [`AbstractBandpassSmoother`](@ref Gustavo.Fring.AbstractBandpassSmoother);
- a new **solve step** — a solver that reads the data through
  [`each_group`](@ref Gustavo.each_group):
  [Authoring a solve step](@ref authoring-steps).
