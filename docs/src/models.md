```@meta
CurrentModule = Gustavo.Calibration
```

# [Specifying gain models](@id specifying-models)

Every solve step in a Gustavo pipeline separates WHAT it solves from HOW.
The WHAT is a *gain model*: named parametric contributions to each station's
complex gain,

```
gain(station, feed, channel, time) = exp(Σ logamp) · cis(Σ phase)
```

where each sum runs over that station's named components. This page is the
model vocabulary: what a component is, the terms, resolutions, and feed
tyings it is built from, how the built-in steps accept one, and what the
standard pipeline's model solves and why.

## The atom: `GainComponent`

A model is a `(; phase, logamp)` pair of named trees of
[`GainComponent`](@ref)s:

```julia
GainComponent(term; Ti, Frequency, Feed = PerFeed())
```

A component is **a term at a resolution with a feed tying**: one physical
`term` (delay, rate, constant, …), replicated over a `Ti` (time) segmentation
and a `Frequency` segmentation — one parameter block per (time segment,
frequency segment, feed block). The keywords are the same dimension names used
everywhere else in Gustavo (`term_eval` coordinates, θ leaf axes,
`gains(sol; ...)` selectors). `Ti` and `Frequency` have no defaults: a
resolution is a modeling choice, stated explicitly.

Two rules of thumb anchor the vocabulary:

- *A new term is a new component; a different resolution is not.* A per-scan
  delay and a track-global delay are the same `Delay()` term under different
  `Ti` segmentations.
- *How finely a value varies is said by its segmentation, never by the term.*
  A value free per channel is `ConstantTerm()` paired with `ChannelBlocks(1)`,
  not a vector-valued term.

### Terms

| Term | Contribution at `(f, t)` | Parameters |
|:-----|:----|:----|
| [`ConstantTerm`](@ref)`()` | `c` | one constant |
| [`Delay`](@ref)`()` | `2π·τ·(f − f0)` | delay `τ` (s) |
| [`Rate`](@ref)`()` | `2π·ṙ·(t − t0)` | fringe rate `ṙ` (Hz), `t0` the segment's own mean epoch |
| [`Dispersion`](@ref)`()` | `K·θ·(1/f0 − 1/f)` | differential TEC `θ` (TECU) |
| [`PolynomialFreq`](@ref)`(n)` / [`PolynomialTime`](@ref)`(n)` | `Σ cᵈ·xᵈ`, `d = 1…n`, over the segment-normalized axis coordinate | `n` coefficients per segment (the constant belongs to a `ConstantTerm`) |

A term contributes to whichever group (`phase` or `logamp`) its component is
placed in. New terms are added with six small methods — see
[Authoring a new gain term](@ref authoring-terms).

### Time segmentations

| Segmentation | One parameter block per |
|:-----|:----|
| [`GlobalTime`](@ref)`()` | whole track (time-invariant) |
| [`PerScan`](@ref)`()` | scan |
| [`PerIntegration`](@ref)`()` | accumulation period (`Ti` sample) |
| [`TimeBlocks`](@ref)`(hours)` | fixed wall-clock block |
| [`InstrumentScans`](@ref)`(boundaries_hr)` | user-specified window |

### Frequency segmentations

| Segmentation | One parameter block per |
|:-----|:----|
| [`GlobalFrequency`](@ref)`()` | whole band |
| [`PerSpectralWindow`](@ref)`()` | spectral window |
| [`ChannelBlocks`](@ref)`(n)` | `n` consecutive channels (`ChannelBlocks(1)` = free per channel) |
| [`FreqGroups`](@ref)`(ranges)` | explicit channel range |
| [`BandGroups`](@ref)`(; gap_factor)` | gap-detected band group |

`BandGroups` is data-dependent: at plan time it
[`materialize`](@ref Gustavo.UVData.materialize)s into the `FreqGroups` that
[`fringe_freq_groups`](@ref) detects from the actual channel frequencies, so
a configuration written before the data is seen adapts to the band layout it
meets.

### Feed tyings

| Tying | Blocks | Meaning |
|:-----|:----|:----|
| [`PerFeed`](@ref)`()` | 2 | each feed solves its own value |
| [`SharedFeeds`](@ref)`()` | 1 | one value, read by both feeds |
| [`SingleFeed`](@ref)`(k)` | 1 | feed `k` only; the other feed gets no contribution |
| [`ReferenceRelative`](@ref)`(k)` | 2 | partner feed = reference block + relative block |

The tying is physics: a feed-common quantity solved `PerFeed` lets the two
feeds' solve noise diverge, injecting spurious cross-hand structure (see the
rate and adhoc components below), while a genuinely instrumental per-feed
quantity solved `SharedFeeds` averages away a real signal.

## Names are load-bearing

Components compose by *named* `NamedTuple` entries, never positionally: θ is
addressed as `θ.phase.<name>`, diagnostics label by name, and
`parameters(sol[:fringe, :phase, :mbd])` selects by the same name. A value may
itself be a named subtree — one model element that compiles to several
components nests them under its key (the SBD element's `sbd.delay` /
`sbd.constant` pair, below).

## Where a model plugs in

Each built-in solve step takes its model as a constructor argument, vetted at
compile time — before any data is read — against the step's solver: the
solver declares [`can_fit`](@ref Gustavo.Fringe.can_fit) per component
(default `false`, so an undeclared combination fails loudly with the
component's constructor spelling, never solves to silent zeros), plus
whole-tree requirements via
[`validate_model`](@ref Gustavo.Fringe.validate_model).

- [`Bandpass`](@ref Gustavo.Bandpass)`(model = default_bandpass_terms(), smoother = ...)`
  and [`TemporalSmoother`](@ref Gustavo.TemporalSmoother)`(model = default_adhoc_terms(), smoother = ...)`
  take a full `(; phase, logamp)` tree (or a `StationGainModel`).
- [`FringeFit`](@ref Gustavo.FringeFit)`(model = FringeModel(), estimator = ...)`
  takes a [`FringeModel`](@ref Gustavo.Fringe.FringeModel): an ordered, named
  phase-term list ([`default_fringe_terms`](@ref Gustavo.Fringe.default_fringe_terms)).
- [`DispersionSBDFit`](@ref Gustavo.DispersionSBDFit)`(dispersion = DispersionModel(), sbd = SingleBandDelay())`
  deliberately has no free-form tree: its solver is a rigid specialized fit,
  so its whole surface is the two element fields (either may be `nothing`).

Each step compiles and solves its own model on its own private θ — no step's
parameter block is shared with or visible to another's. Gains compose
multiplicatively across steps, so e.g. `DispersionSBDFit`'s private per-scan
delay-refinement column times `FringeFit`'s wideband delay is the same total
correction as incrementing one shared column would be.

## The standard pipeline's model, component by component

The four-step pipeline `FringeFit() |> DispersionSBDFit() |> Bandpass() |>
TemporalSmoother()` solves, across its steps, the following phase components
— this is the standard VLBI calibration model, and each tying below is a
physics decision:

**`atmos` — per-scan constant phase, feed-common.** The atmosphere/clock
phase that varies scan to scan and is the same for both polarization feeds.

**`mbd` — per-scan wideband (multi-band) delay, feed-common.** One slope
across the whole band per scan.

**`rel_delay` — inter-feed delay offset, `SingleFeed(2)`.** The instrumental
feed-2 − feed-1 group-delay offset. `PerScan()` (the default) fits it per
scan, so its scan-to-scan scatter is an instrument-stability diagnostic and
no column couples scans; `GlobalTime()` fits one offset per station for the
whole track (the EHT-HOPS / rPICARD assumption), so bright scans pin it and
weak scans inherit it through the shared column. There is deliberately no
inter-feed *phase* offset: a feed-2 constant is not separable from the
source's cross-hand phase, so fitting one would remove the source's
polarization angle along with the instrument's offset — see
[`default_fringe_terms`](@ref Gustavo.Fringe.default_fringe_terms) for the
full argument.

**`rate` — per-scan fringe rate, feed-common.** The fringe rate is common to
both feeds, so it is tied across them exactly like the per-scan constant and
delay. Solving it `PerFeed` instead lets a spurious inter-feed rate
(`ṙ₂ − ṙ₁`) float on noise — and, multiplied by the hours-long rate lever
arm, inject arbitrary scan-to-scan cross-hand phase jumps. The inter-feed
rate is negligible (EHT-HOPS), so it is tied; a genuine offset would be
opted into as a separate `PerScan × SingleFeed(2)` rate component, not by
untying this one.

**`dtec` — per-scan differential TEC, feed-common** (`DispersionSBDFit`'s
`dispersion` half). Ionospheric dispersion, phase `K·θ·(1/f0 − 1/f)` with θ
in TECU. It is *not* solved by the FFT fringe search: over any finite band
the 1/ν curvature is nearly degenerate with a linear delay, so the refinement
stage fits (Δτ, dTEC) jointly per scan, updating its own private delay column
alongside. It is only compiled when the band layout can constrain the
curvature (several sub-bands over a wide fractional bandwidth — see
[`DispersionModel`](@ref)).

**`sbd.delay` + `sbd.constant` — per-scan per-band-group single-band delay**
(`DispersionSBDFit`'s `sbd` half, fourfit's SBD). A station's per-band signal
path can move relative to its phase-cal tones between scans (tens of ns),
which neither the wideband delay (one slope across all groups) nor the
time-invariant per-channel bandpass can track. It is measured from
within-band chunk slopes — nearly orthogonal to the cross-band observables
that set `mbd` and `dtec`. The pair exists because the `Delay` coordinate is
`(f − f0)` with the *global* `f0`: correcting a group's slope about its own
centre `νg` needs the companion per-group constant `−2πτ(νg − f0)`, making
the net phase `2πτ(f − νg)` — zero at the group centre, so the cross-band
solution is untouched.

**`bandpass` — per-channel constant phase, time-global, per-feed** (the
`Bandpass` step's phase half). The residual nonlinear-in-frequency
instrumental phase that a per-scan linear delay cannot represent, stable
across the observation (HOPS-style).

**`adhoc` — per-integration constant phase, feed-common** (the
`TemporalSmoother` step). Residual atmospheric phase is non-birefringent
(common to both feeds), so it is solved feed-common — which both denoises it
and contributes exactly zero inter-feed phase. A `PerFeed` adhoc lets per-AP
solve noise differ between feeds and injects spurious cross-hand scatter on
top of the real instrumental inter-feed offset.

And one log-amplitude component:

**`bandpass` — per-channel constant log-amplitude, time-global, per-feed**
(the `Bandpass` step's logamp half). The instrumental amplitude passband
(filterbank shape), flattened from the calibrator under a pluggable shape
spec. The absolute flux scale is *not* its job — that stays with the a-priori
amplitude calibration ([`AprioriAmplitude`](@ref Gustavo.AprioriAmplitude)).

## Per-station heterogeneity

Stations may differ in segmentation and in terms. The rule the design
follows: *the model shouldn't change shape because one station behaves a
little differently.* A [`StationGainModel`](@ref)'s `stations` keyword maps a
station code to a replacement entry used verbatim:

```julia
StationGainModel(
    phase = (bandpass = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1)),),
    stations = (
        AA = (; phase = (bandpass = GainComponent(PolynomialFreq(3); Ti = GlobalTime(), Frequency = GlobalFrequency()),)),
    ),
)
```

Here every station solves a free per-channel phase bandpass except AA, which
gets a smooth cubic — the right model for a station too weak to constrain
per-channel values. Replacement is **whole-group**: a station entry supplies
complete `phase` and/or `logamp` trees (a group the entry omits is inherited
from the base), because components within a group interact — they sum and
share degeneracies — while the two groups do not. The model never merges
within a group.

Station codes resolve against the observation's antenna table when the model
is materialized for a solve; an unknown code errors there, naming the known
stations. A rule-based model ("every station matching a prefix gets a
smoother bandpass") is an [`AbstractGainModel`](@ref) subtype implementing
the one seam method, [`station_components`](@ref).

Heterogeneity is opt-in per solver
([`supports_station_heterogeneity`](@ref Gustavo.supports_station_heterogeneity)):
a solver that has not declared support rejects a heterogeneous model at
compile time with an error naming the differing component, rather than
silently leaving θ blocks unsolved. No shipped step declares support yet;
the seam exists for solvers written against the station-block iterator (see
[Authoring a pipeline step](@ref authoring-steps)).

## Inspecting a model

[`component_label`](@ref) prints a component as its constructor call — a form
you can paste back — and error messages use the same spelling.
[`station_model_summary`](@ref Gustavo.Calibration.station_model_summary)
summarizes a whole model, station overrides included. After a fit,
[`parameters`](@ref) shows each component's solved θ as a labelled
`DimArray`.
