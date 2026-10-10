```@meta
CurrentModule = Gustavo.Calibration
```

# [Specifying gain models](@id specifying-models)

Every solve step in Gustavo separates WHAT it solves from HOW.
The WHAT is a *gain model*: named parametric contributions to each station's
complex gain,

```
gain(station, feed, channel, time) = exp(Σ logamp) · cis(Σ phase)
```

where each sum runs over that station's named components. This page is the
model vocabulary: what a component is, the terms, resolutions, and feed
tyings it is built from, how the built-in steps accept one, and what the
standard steps' models solve and why.

## The atom: `GainComponent`

A model is a [`GainModel`](@ref): a `phase` group and a `logamp` group, each
a `NamedTuple` of named [`GainComponent`](@ref)s:

```julia
GainComponent(term; Ti, Frequency = GlobalFrequency(), Feed = PerFeed(), prior = nothing)

bp = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1))
GainModel(phase = (; bp), logamp = (; bp))
```

A component is **a term at a resolution with a feed tying**: one physical
`term` (delay, rate, constant, …), replicated over a `Ti` (time) segmentation
and a `Frequency` segmentation — one parameter block per (time segment,
frequency segment, feed block). The keywords are the same dimension names used
everywhere else in Gustavo (`term_eval` coordinates, θ leaf axes,
`gains(sol; ...)` selectors). `Ti` has no default: the time resolution
decides what is solved, so it is stated explicitly. `Frequency` defaults to
the whole band.

Two rules of thumb anchor the vocabulary:

- *A new term is a new component; a different resolution is not.* A per-scan
  delay and a track-global delay are the same `Delay()` term under different
  `Ti` segmentations.
- *How finely a value varies is said by its segmentation, never by the term.*
  A value free per channel is `ConstantTerm()` paired with `ChannelBlocks(1)`,
  not a vector-valued term. A value with structure at several resolutions is a
  sum of components, one per resolution: a bandpass may be a `level` per
  spectral window plus a per-channel `shape` under a zero-mean prior.

### Terms

| Term | Contribution at `(f, t)` | Parameters |
|:-----|:----|:----|
| [`ConstantTerm`](@ref)`()` | `c` | one constant |
| [`Delay`](@ref)`()` | `2π·τ·(f − f0)` | delay `τ` (s) |
| [`Rate`](@ref)`()` | `2π·ṙ·(t − t0)` | fringe rate `ṙ` (Hz), `t0` the segment's own mean epoch |
| [`Dispersion`](@ref)`()` | `K·θ·(1/f0 − 1/f)` | differential TEC `θ` (TECU) |
| [`PolynomialFreq`](@ref)`(n)` / [`PolynomialTime`](@ref)`(n)` | `Σ cᵈ·xᵈ`, `d = 1…n`, over the segment-normalized axis coordinate | `n` coefficients per segment (the constant belongs to a `ConstantTerm`) |

A term contributes to whichever group (`phase` or `logamp`) its component is
placed in. New terms are added with five small methods — see
[Authoring a new gain term](@ref authoring-terms).

### Priors

A component's `prior` is the Gaussian belief about its parameters before the
data:

| Prior | Density |
|:------|:--------|
| `nothing` | none: every value is free |
| [`IIDPrior`](@ref)`(σ)` | each parameter `N(0, σ²)` |
| [`RandomWalkPrior`](@ref)`(; order, σ, init)` | the `(order − 1)`-times integrated Brownian motion along the axis; `σ²` per unit of `x^(2·order − 1)`; `σ` a number or a hyperprior; `init` a Gaussian on the initial value and its derivatives, or `nothing` (free) |
| [`OUPrior`](@ref)`(; scale, σ)` | a zero-mean Ornstein–Uhlenbeck process along the axis; `scale` and `σ` each a number or a hyperprior |

`σ` is in the parameter's own units, per the power of the axis coordinate (s or Hz) the random walk states. A correlated prior runs along the one axis
the component segments (the one whose segmentation is not `GlobalTime` or
`GlobalFrequency`). A component segmenting both keys the prior by axis,
`prior = (Ti = OUPrior(; …), Frequency = RandomWalkPrior(; …))`, and may leave
either key out; see [`resolve_prior`](@ref). Along frequency, a prior relates
values within one spectral window and never across windows. Stations that
differ only in their priors share one parameter layout. Which priors a solver
fits is part of its `can_fit`. How a solver fits a track under a prior, and estimates a
hyperprior, is derived in [Fitting under priors](@ref fitting-under-priors).

A level is its own component, with no prior. The zero mean of the shape's prior
is what separates the two:

```julia
shape = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1),
                      prior = OUPrior(; scale = 50e6, σ = 0.1))
level = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = PerSpectralWindow())
GainModel(phase = (; level, shape), logamp = (; level, shape))
```

The bandpass smoother estimates the level by generalized least squares under the
shape's prior, and an `OUPrior` hyperprior with the level integrated out. A
random walk beside a level is rejected, since the walk leaves its own level
free, as is a shape with no prior.

### Time segmentations

| Segmentation | One parameter block per |
|:-----|:----|
| [`GlobalTime`](@ref)`()` | whole track (time-invariant) |
| [`PerScan`](@ref)`()` | scan |
| [`PerIntegration`](@ref)`()` | accumulation period (`Ti` sample) |
| [`TimeBlocks`](@ref)`(seconds)` | fixed wall-clock block |
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
[`resolve`](@ref Gustavo.Calibration.resolve)s into the `FreqGroups` that
[`fringe_freq_groups`](@ref) detects from the actual channel frequencies, so
a configuration written before the data is seen adapts to the band layout it
meets.

### Feed tyings

| Tying | Blocks | Meaning |
|:-----|:----|:----|
| [`PerFeed`](@ref)`()` | ``n`` | each feed solves its own value |
| [`SharedFeeds`](@ref)`()` | 1 | one value, read by every feed |
| [`SingleFeed`](@ref)`(k)` | 1 | feed `k` only; the other feeds get no contribution |
| [`ExceptFeed`](@ref)`(k)` | ``n - 1`` | every feed but `k`, each its own value; feed `k` gets no contribution |

``n`` is the number of feeds per station. A reference/relative model is a
`SharedFeeds` component for the common part plus an `ExceptFeed(k)` component
for every other feed's deviation from feed `k`.

The tying is physics: a feed-common quantity solved `PerFeed` lets the
feeds' solve noise diverge, injecting spurious inter-feed structure (see the
rate and adhoc components below), while a genuinely instrumental per-feed
quantity solved `SharedFeeds` averages away a real signal.

## Names are load-bearing

Components compose by *named* `NamedTuple` entries, never positionally: θ is
addressed as `θ.phase.<name>`, diagnostics label by name, and
`sol[:fringe, :phase, :mbd]` selects by the same name. A value may
itself be a named subtree of components, addressed through its key
(`θ.phase.<name>.<part>`).

## Where a model plugs in

Each built-in solve step takes its model as a constructor argument, vetted at
compile time — before any data is read — against the step's solver: the
solver declares [`can_fit`](@ref Gustavo.Fring.can_fit) per component
(default `false`, so an undeclared combination fails loudly with the
component's constructor spelling, never solves to silent zeros), plus
whole-tree requirements via
[`validate_model`](@ref Gustavo.Fring.validate_model).

- [`Bandpass`](@ref Gustavo.Bandpass)`(model = default_bandpass_terms(), smoother = ...)`,
  [`AdhocPhase`](@ref Gustavo.AdhocPhase)`(model = default_adhoc_terms(), smoother = ...)`
  and [`BaselineFringeFit`](@ref Gustavo.BaselineFringeFit)`(model = default_fringe_terms(), search = ...)`
  take a `GainModel`; the fringe step's model has phase components only
  ([`default_fringe_terms`](@ref Gustavo.Fring.default_fringe_terms)).

[`merge`](@ref Base.merge(::GainModel)) builds a variant of a model by adding
or replacing named components:

```julia
BaselineFringeFit(model = merge(default_fringe_terms();
    phase = (; rel_rate = GainComponent(Rate(); Ti = PerScan(), Feed = ExceptFeed(1)))))
```

Each step compiles and solves its own model on its own private θ — no step's
parameter block is shared with or visible to another's. Gains compose
multiplicatively across steps.

## The standard model, component by component

The three steps `BaselineFringeFit()`, `Bandpass()` and `AdhocPhase()`, fit in
turn, solve the following phase components
— this is the standard VLBI calibration model, and each tying below is a
physics decision:

There is no per-scan constant phase. A station's per-scan phase trades
against the source's own phase, so the data do not determine it without a
source model; it is left for imaging or self-calibration — see
[`default_fringe_terms`](@ref Gustavo.Fring.default_fringe_terms).

**`mbd` — per-scan wideband (multi-band) delay, feed-common.** One slope
across the whole band per scan.

**`rel_delay` — inter-feed delay offsets, `ExceptFeed(1)`.** Each other
feed's instrumental group-delay offset from feed 1; with one feed per station
the component has no parameters. `PerScan()` (the default) fits it per
scan, so its scan-to-scan scatter is an instrument-stability diagnostic and
no column couples scans; `GlobalTime()` fits one offset per station for the
whole track (the EHT-HOPS / rPICARD assumption), so bright scans pin it and
weak scans inherit it through the shared column.

**`rate` — per-scan fringe rate, feed-common.** The fringe rate is common to
every feed, so it is tied across them exactly like the delay. Solving it `PerFeed` instead lets a spurious inter-feed rate
(`ṙ₂ − ṙ₁`) float on noise — and, multiplied by the hours-long rate lever
arm, inject arbitrary scan-to-scan inter-feed phase jumps. The inter-feed
rate is negligible (EHT-HOPS), so it is tied; a genuine offset would be
opted into as a separate `PerScan × ExceptFeed(1)` rate component, not by
untying this one.

**`bandpass` — per-channel constant phase, time-global, per-feed** (the
`Bandpass` step's phase half). The residual nonlinear-in-frequency
instrumental phase that a per-scan linear delay cannot represent, stable
across the observation (HOPS-style).

**`adhoc` — per-integration constant phase, feed-common** (the
`AdhocPhase` step). Residual atmospheric phase is non-birefringent
(common to every feed), so it is solved feed-common — which both denoises it
and contributes exactly zero inter-feed phase in any feed basis. A `PerFeed`
adhoc lets per-AP solve noise differ between feeds and scatters the inter-feed
phase on top of the real instrumental offset. Its prior along time is an
`OUPrior` with weakly informative hyperpriors by default
([`default_adhoc_prior`](@ref Gustavo.Fring.default_adhoc_prior)); each
track's weighted mean over a scan is removed, since a per-station constant
trades against the source terms and so is a gauge rather than a parameter.

And one log-amplitude component:

**`bandpass` — per-channel constant log-amplitude, time-global, per-feed**
(the `Bandpass` step's logamp half). The instrumental amplitude passband
(filterbank shape), measured on the calibrator under the component's prior,
if any. The absolute flux scale is *not* its job — that stays with the a-priori
amplitude calibration.

## Per-station heterogeneity

Stations may differ in segmentation and in terms. The rule the design
follows: *the model shouldn't change shape because one station behaves a
little differently.* [`with_station`](@ref) gives one station its own
groups, used verbatim:

```julia
bp(prior) = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1), prior)
m = GainModel(phase = (; bandpass = bp(nothing)))
with_station(m, "AA"; phase = (; bandpass = bp(RandomWalkPrior(; order = 2, σ = 1.0e-11))))
```

Here every station solves a free per-channel phase bandpass except AA, whose
channels are tied by a second-order random walk (`σ` in rad/Hz^(3/2): over
2 MHz channels the second difference has a standard deviation of about 0.02
rad) — the right model for a station too weak to constrain per-channel values. Replacement is **whole-group**: a station entry supplies
complete `phase` and/or `logamp` trees (a group the entry omits is inherited
from the base), because components within a group interact — they sum and
share degeneracies — while the two groups do not. The model never merges
within a group. Entries that differ from the base only in their priors, as AA's
does here, keep the model uniform: the stations share one parameter layout, and
each carries its own prior.

Station codes resolve against the observation's antenna table when the model
is resolved for a solve; an unknown code errors there, naming the known
stations. A rule ("every station matching a prefix gets a smoother
bandpass") is a loop of `with_station` calls over the station codes.

Heterogeneity is opt-in per solver
([`supports_station_heterogeneity`](@ref Gustavo.supports_station_heterogeneity)):
a solver that has not declared support rejects a heterogeneous model at
compile time with an error naming the differing component, rather than
silently leaving θ blocks unsolved. Among the shipped steps, `Bandpass` with a
`JointSmoother` declares support; solvers written against the station-block
iterator can declare it too (see [Authoring a solve step](@ref authoring-steps)).

## Inspecting a model

[`component_label`](@ref) prints a component as its constructor call — a form
you can paste back — and error messages use the same spelling.
Showing a `GainModel` lists its components the same way, and the stations
with their own groups. After a fit, each
[`SolvedComponent`](@ref) of the solution holds its solved θ as a labeled
`DimArray` (`c.params`).
