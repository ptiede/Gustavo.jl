```@meta
CurrentModule = Gustavo.Fring
```

# [Bandpass diagnostics](@id bandpass-diagnostics)

Once a bandpass is divided out, every spectrum should be the source alone,
flat in amplitude and phase. These functions measure how far corrected data
are from that, compare bandpass models, and find where a station's bandpass
changed between scans. They report what is left, a residual delay included,
rather than fitting anything out of it.

- [`bandpass_chi2`](@ref): the goodness of fit of a corrected set.
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

## Goodness of fit

[`bandpass_chi2`](@ref) fits each spectrum (one baseline, parallel feed pair
and time sample) with a complex constant and with a line in frequency, and sums
the residual χ²:

```julia
res = bandpass_chi2(calibrate(sol, ps))
print_chi2(res; nparams = sum(c -> length(c.params), sol.components))
total_chi2(res; model = :lin)       # (; chi2, dof) at the scan resolution
station_chi2(res)["S1"]             # one station's baselines
```

Two resolutions are reported:
- `ap`: every integration.
- `scan`: every scan average.

A bandpass error is the same in every integration of a scan, while the noise
averages down, so the `scan` χ² shows bandpass error most clearly.

The `lin` model absorbs a phase slope across the band: a residual delay, or
the frequency dependence of a resolved source. A `const` χ² well above the
`lin` one therefore says the corrected data still carry delays. That is a
missing delay term in the gain model, which no bandpass shape can follow when
the delay changes from scan to scan.

To compare a model B with a model A that adds `Δp` parameters (a break, say),
fit both to the same data and form

```math
F = \frac{(\chi^2_B - \chi^2_A)/\Delta p}{\chi^2_A/\nu_A}.
```

If A's residuals were noise, F would follow F(Δp, ν_A), whose upper quantiles
are close to 1 for large `ν_A`. Residuals of real data are rarely noise. They
carry structure that changes from scan to scan, and nearly any extra
parameters absorb some of it. So the quantile alone accepts almost any
addition. Read F as the factor by which the addition beats the average misfit,
and require a large one.

## Residuals

[`normalized_residuals`](@ref) divides every cut of a set of spectra along an
axis by its weighted mean along that axis, and gives the variance of each
component. Along `Frequency`, the default, this removes each spectrum's phase,
amplitude scale and band-mean source visibility. What is left is the structure
across the band that the correction did not remove. That includes a residual
delay, seen as a phase slope, which is the sign of a missing delay term:

```julia
using XRadio: average, ByScan
res = normalized_residuals(baseline_spectra(average(ps, ByScan())))
angle.(res.r[Scan = At("SRC_3"), FeedPair = At((1, 1))])   # residual phase per baseline
```

Any layout with a `Frequency` axis works: scan averages as above, shorter
averages, or a Measurement Set's integrations. Along `Ti`, each channel is
taken about its mean over time instead:

```julia
w = DimArray(ifelse.(parent(ms[:flag]), 0.0f0, parent(ms[:weight])), dims(ms[:weight]))
normalized_residuals(DimStack((; vis = ms[:visibility], weight = w)); dims = Ti)
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
