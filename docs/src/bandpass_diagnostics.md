```@meta
CurrentModule = Gustavo.Fring
```

# [Bandpass diagnostics](@id bandpass-diagnostics)

Once a bandpass is divided out, every spectrum should be the source alone,
flat in amplitude and phase. These functions measure how far corrected data
are from that, compare bandpass models, and find where a station's bandpass
changed between scans. They report what is left, a residual delay included,
rather than fitting anything out of it.

- [`tss`](@ref): how far each corrected spectrum is from flat.
- [`chi2`](@ref): the χ² of data against a model.
- [`normalized_residuals`](@ref): each spectrum over its mean along an axis.
- [`step_tests`](@ref): an F test for a change in one station's bandpass at
  each gap between its scans.

## Weights

All three read the weights as the inverse variance of each real component of
the visibilities. A χ²/dof of 1 then means a perfect fit. If the weights are
right only up to a factor per baseline, [`weight_scales`](@ref Gustavo.weight_scales)
measures that factor from the scatter between consecutive samples and
[`scale_weights!`](@ref Gustavo.scale_weights!) applies it:

```julia
foreach(ms -> scale_weights!(ms, weight_scales(ms)), values(ps))
```

A comparison between two models fitted to the same data does not need this:
both χ² values scale with the weights, and the ratios below cancel the scale.

## Input

Each function takes a processing set or a single Measurement Set. It lays the
data out as [`baseline_spectra`](@ref) does, over
`(Ti, AntennaPair, FeedPair, Frequency)` with the channels of every spectral
window on one `Frequency` axis, and skips flagged samples. A
[`baseline_spectra`](@ref) result works too, which saves laying the data out
again for each call. Results keep those dimensions, less the ones reduced.

A bandpass error is the same in every integration of a scan, while the noise
averages down, so scan averages show it most clearly: pass
`average(ps, ByScan())`. Unaveraged data give one value per integration.

## Goodness of fit

Both functions return a DimStack with a degrees-of-freedom layer `dof`.

[`tss`](@ref) is the weighted sum of squares of each cut along `dims` about its
own weighted mean, one value per cut. Along `Frequency`, the default, it is
the χ² of a source that is constant across the band, the model a corrected
spectrum should satisfy, with that constant fitted to each spectrum. Totals
are sums:

```julia
using XRadio: average, ByScan
q = tss(average(calibrate(sol, ps), ByScan()))[FeedPair = At([(1, 1), (2, 2)])]
sum(q.tss) / sum(q.dof)                              # the whole set
sum(q.tss; dims = Ti) ./ sum(q.dof; dims = Ti)       # per baseline and feed pair
mine = [p for p in lookup(q.tss, AntennaPair) if "S1" in p]
sum(q.tss[AntennaPair = At(mine)]) / sum(q.dof[AntennaPair = At(mine)])   # one station
```

With `dims = Ti`, each channel is taken about its mean over time instead.
On unaveraged data, `tss(ms; dims = Ti)` tests whether a Measurement Set's
channels hold steady through its scan.

A value per degree of freedom well above 1 says the correction left structure
across the band. A residual delay is such structure: a phase slope that, if it
changes from scan to scan, is a missing delay term in the gain model rather
than anything a bandpass shape can follow. [`normalized_residuals`](@ref)
shows which it is.

[`chi2`](@ref) compares the data with any model of the visibilities, a
`DimArray` over some of the data's dimensions, which is used for every sample
along the dimensions it lacks. Here `dims` only names what is summed:

```julia
chi2(average(ps, ByScan()), model; dims = (Frequency, Ti))   # Σ w |v - model|² per baseline and feed pair
```

Every usable sample counts 2 degrees of freedom. The model's own free
parameters are not charged; subtract them from `dof`.

To compare a model B with a model A that adds `Δp` parameters (a break, say),
fit both to the same data and form

```math
F = \frac{(\chi^2_B - \chi^2_A)/\Delta p}{\chi^2_A/\nu_A}.
```

Here χ² is each model's [`tss`](@ref) on the data it corrected, or its
[`chi2`](@ref) against a common model. If A's residuals were noise, F would
follow F(Δp, ν_A), whose upper quantiles
are close to 1 for large `ν_A`. Residuals of real data are rarely noise. They
carry structure that changes from scan to scan, and nearly any extra
parameters absorb some of it. So the quantile alone accepts almost any
addition. Read F as the factor by which the addition beats the average misfit,
and require a large one.

## Residuals

[`normalized_residuals`](@ref) divides every cut of the data along an axis by
its weighted mean along that axis, and gives the variance of each
component. Along `Frequency`, the default, this removes each spectrum's phase,
amplitude scale and band-mean source visibility. What is left is the structure
across the band that the correction did not remove. That includes a residual
delay, seen as a phase slope, which is the sign of a missing delay term:

```julia
using XRadio: average, ByScan
res = normalized_residuals(average(ps, ByScan()))
k = findfirst(==("SRC_3"), res.scan_name)
angle.(res.r[Ti = k, FeedPair = At((1, 1))])         # residual phase per baseline
```

Scan averages, shorter averages and unaveraged data all work. Along `Ti`, each
channel is taken about its mean over time instead:

```julia
normalized_residuals(ms; dims = Ti)
```

Scan averages raise the signal-to-noise each spectrum is normalized with, and
lose nothing for a bandpass that holds over a scan. They assume the phase is
coherent through the scan. A phase common to all channels cancels in the
ratio, but a delay drifting within the scan decorrelates the outer channels
more than the inner ones.

## Step tests

[`step_tests`](@ref) asks, for each gap between two of a station's
consecutive scans, whether the residual spectra on the station's baselines
are better described by one mean spectrum on each side of the gap than by one
overall:

```julia
intervals = scan_intervals(ps)
rows = step_tests(res, intervals, "S1")
rows[argmax([r.F for r in rows])]          # the gap with the strongest step
```

F is normalized by the scatter of the scans about the two means, so scatter
between scans that is not a step leaves F near 1. A residual delay that
changes from scan to scan is such scatter, and lowers F at every gap. A delay
that changes across a gap is a step. Reading a station's F
against time:

- **A step peaks.** The F of a step is largest at the gap where it happened.
  It falls off as the split moves away and mixes the two bandpasses on one
  side. Gaps only minutes from the step move one short scan to the other side
  and read almost the same F. So compare the peak with gaps well away from it,
  not with its neighbors.
- **A drift plateaus.** A bandpass changing slowly through a track gives
  similar F at many gaps, and its maximum moves when the data change a little.
  Every cut through a drift is significant, but no single gap is where it
  happened.
- **A step leaks through shared baselines.** A step at station S1 shows on
  every baseline to S1, including S1–S2. It therefore raises the F of S2,
  most when S1–S2 is S2's most sensitive baseline. Fitting the break to each
  candidate in turn tells them apart: the break at the station that owns the
  step removes more χ².
- **A station step is common to the station's baselines.** A bandpass
  multiplies every baseline to its station alike, so a real change at S1
  appears on all of S1's sensitive baselines at the same gap. Testing each
  baseline alone shows whether it does:

```julia
using DimensionalData: lookup, At
for p in lookup(res.r, AntennaPair)
    "S1" in p || continue
    rows = step_tests(res[AntennaPair(At([p]))], intervals, "S1")
    println(p, " ", maximum(r -> r.F, rows))
end
```

A change that shows on one baseline, or at different times on different
baselines, is not a station bandpass change. It does not close. A station term
can only move it onto the station's other baselines.
