```@meta
CurrentModule = Gustavo.Fring
```

# [Fringe fitting](@id fringe-fitting)

Fringe fitting estimates, for each station, the delay, fringe rate and phase
that align the visibility phasor across frequency and time. It proceeds in two
stages. The first estimates a delay, rate and phase for each baseline and
correlation product separately, by maximizing a matched filter. The second
treats those estimates as differences of per-station quantities and solves for
the stations by weighted least squares.

Both stages are derived below. The choice of which parameters are solved, at
what time and frequency resolution, and with what tying across feeds belongs to
the gain model, described in
[Specifying gain models](@ref specifying-models); the fringe stage's model is a
[`FringeModel`](@ref) and its estimator a [`MatchedFilter`](@ref). Phase sense,
parameter signs and units are fixed in [Conventions](@ref conventions).

## Matched filter

Write the visibility measured on one baseline, for one correlation product, at
frequency ``f`` and time ``t`` as

```math
V(f, t) = A \exp\!\big(i[\varphi + 2\pi\tau(f - f_0) + 2\pi\dot{r}(t - t_0)]\big) + n(f, t),
```

with amplitude ``A``, group delay ``\tau``, fringe rate ``\dot r``, and phase
``\varphi`` referred to a reference frequency ``f_0`` and epoch ``t_0``. The
noise ``n`` is circular complex Gaussian, independent between samples, with
variance ``\sigma^2(f,t) = 1/w(f,t)``; the weights ``w`` stored beside the
visibilities are inverse variances.

The maximum-likelihood estimate minimizes

```math
\chi^2(A, \varphi, \tau, \dot r) = \sum_{f,t} w \left| V - A e^{i\psi} \right|^2,
\qquad \psi = \varphi + 2\pi\tau(f - f_0) + 2\pi\dot r(t - t_0).
```

Expanding the square,

```math
\chi^2 = \sum w |V|^2 - 2\,\mathrm{Re}\Big[A e^{i\varphi} \overline{D(\tau, \dot r)}\Big] + A^2 \sum w,
```

where every dependence on the data has collected into

```math
D(\tau, \dot r) = \sum_{f,t} w(f,t)\, V(f,t)\, \exp\!\big(-2\pi i[\tau(f - f_0) + \dot r(t - t_0)]\big).
```

For fixed ``(\tau, \dot r)`` the minimum over the linear parameters is at

```math
\hat\varphi = \arg D(\tau, \dot r), \qquad \hat A = \frac{|D(\tau, \dot r)|}{\sum w},
```

and substituting them back gives

```math
\chi^2_{\min}(\tau, \dot r) = \sum w |V|^2 - \frac{|D(\tau, \dot r)|^2}{\sum w}.
```

Minimizing ``\chi^2`` is therefore equivalent to maximizing ``|D|``. The two
nonlinear parameters are estimated by a search over ``(\tau, \dot r)``, and the
two linear parameters follow in closed form. ``D`` is the matched filter for
this model.

A sample contributes to the sums when its flag is unset, its visibility is
finite, and its weight is finite and positive. Flags and weights are
independent, so clearing a flag restores a sample at the weight it already
carried.

[`baseline_fringe_search`](@ref) performs this search for one block, given a
`DimStack` over `(Frequency, Ti)` with `:vis`, `:weights` and `:flags` layers.
[`search_scan`](@ref) applies it to every cross baseline and product of a scan;
autocorrelations are excluded. Each search returns a [`Detection`](@ref),
holding ``\hat\tau``, ``\hat{\dot r}``, ``\hat\varphi``, ``\hat A``, a
signal-to-noise ratio and a false-alarm probability.

## FFT evaluation

Let the channel frequencies and timestamps lie on uniform grids,
``f_k = f_\mathrm{lo} + k\,\Delta f`` for ``k = 0,\dots,N-1`` and
``t_j = t_\mathrm{lo} + j\,\Delta t`` for ``j = 0,\dots,M-1``. Absorbing the
constant offsets into a phase factor,

```math
D(\tau, \dot r) \propto \sum_{k,j} G_{kj} \exp\!\big(-2\pi i [\tau \Delta f\, k + \dot r \Delta t\, j]\big),
\qquad G_{kj} = w_{kj} V_{kj},
```

a two-dimensional DFT of the weighted visibilities: delay is conjugate to
frequency and rate is conjugate to time. Evaluating it on the FFT grid
``\tau_m = m/(N\Delta f)``, ``\dot r_l = l/(M \Delta t)`` costs
``O(NM\log NM)`` for all cells together, against ``O(NM)`` for each trial pair
evaluated directly.

Grid points with no contributing sample, whether flagged, unweighted, or lying
in a gap between bands, are set to zero. They then contribute nothing to ``D``
and nothing to ``\sum w``.

### Delay resolution and zero padding

Take uniform weights over ``N`` contiguous channels and hold ``\dot r`` at its
true value. Then

```math
|D(\tau)| \propto \left| \sum_{k=0}^{N-1} e^{-2\pi i \tau \Delta f k} \right|
= \left| \frac{\sin(\pi N \Delta f \tau)}{\sin(\pi \Delta f \tau)} \right|,
```

a Dirichlet kernel whose first null is at ``\tau = 1/(N\Delta f) = 1/B``, with
``B`` the spanned bandwidth. The response to a delay error is therefore a peak
of width of order ``1/B``. This is the delay resolution, and it is set by the
span of the frequency axis rather than by the number of samples on it.

The FFT samples this peak at spacing ``1/B``, approximately one sample per main
lobe. The largest sampled cell is then displaced from the true maximum by up to
half a cell, an effect termed scalloping. Padding the grid with ``P-1`` zeros
per sample, the `oversample` factor ``P``, interpolates the same continuous
response onto a grid of spacing

```math
\delta\tau = \frac{1}{P N \Delta f} = \frac{1}{PB},
```

so ``P`` is the number of grid cells laid across the main lobe. Padding adds no
information; it evaluates an existing response more finely.

Zero padding does not remove scalloping, so the grid cell is not reported. The
FFT peak is used as a starting point, and ``|D|`` is maximized off the grid by
per-axis parabolic steps (`_polish_peak_exact!`), the sum being evaluated
directly at each trial. The amplitude and phase are read from that exact
value,

```math
D_\mathrm{ref} = \sum w V \exp\!\big(-2\pi i[\hat\tau(f - f_0) + \hat{\dot r}(t - t_0)]\big),
\qquad \hat A = \frac{|D_\mathrm{ref}|}{\sum w},
\qquad \hat\varphi = \arg D_\mathrm{ref},
```

referred directly to ``(f_0, t_0)``, so no grid-origin rotation enters the
result.

``P`` therefore controls which local maximum the refinement starts in, not the
precision with which that maximum is then located. Setting
`quad_interp = false` disables the refinement and reports the grid cell itself.

### Multi-band aliasing

For contiguous channels the delay window contains a single peak, and scalloping
affects only precision, which the refinement recovers. A gapped axis does not
have this property.

Partition the channels into bands ``b`` with origins ``f_b``. Splitting the sum
over bands,

```math
D(\tau, \dot r) = \sum_b e^{-2\pi i \tau (f_b - f_1)} D_b(\tau, \dot r),
\qquad
D_b(\tau, \dot r) = \sum_{f \in b,\, t} w V e^{-2\pi i [\tau(f - f_b) + \dot r (t - t_0)]}.
```

Each ``|D_b|`` varies with ``\tau`` on the scale ``1/B_\mathrm{band}`` set by
one band's width, while the exponential prefactors vary on the scale
``1/\mathrm{span}``. If the band origins are commensurate with spacing
``\delta``, the prefactor sum is periodic in ``\tau`` with period ``1/\delta``,
so the response carries a comb of alias peaks at that spacing, of comparable
height, under an envelope of width ``1/B_\mathrm{band}``.

A scalloped main peak may then fall below one of its aliases. The refinement
does not recover this case, since it converges to the local maximum it was
started in. The required ``P`` is therefore set by the need to distinguish comb
teeth, not by the width of a single peak. The relevant quantity is the
sparsity

```math
s = \frac{N_\mathrm{grid}}{N_\mathrm{chan}},
```

the number of bins in the common grid over the number of channels occupied;
``s = 1`` for a contiguous axis and ``s \gg 1`` for a sparse one.
`oversample = :auto` (the default) reads ``P`` from it:

| Sparsity ``s`` | ``P`` |
|:---------------|:------|
| ``s < 2`` | 2 |
| ``2 \le s < 3`` | 4 |
| ``s \ge 3`` | 8 |

The thresholds are empirical. Injected fringes were scored on how often the
recovered delay landed on the correct peak, swept over sparsity and
signal-to-noise: ``P = 2`` suffices to ``s \approx 1.5``; ``P = 4`` is
indistinguishable from ``P = 8`` up to ``s \approx 3``; at ``s = 4``, ``P = 8``
exceeds ``P = 4`` by one to three percentage points. At ``P = 1``
identification fails at all signal-to-noise ratios, the failure being geometric
rather than statistical. See [`_resolve_oversample`](@ref).

The cost is ``O(P^2)``, the grid being ``PN \times PM``, so ``P = 8`` is about
sixteen times the work of ``P = 2``. An explicit positive integer overrides
`:auto`. Since `:auto` is a function of the frequency axis alone, replaying a
recorded [`FringeSearch`](@ref) on the same data reproduces the same grid.

The `delay_window` and `rate_window` restrict where the peak is sought. They do
not reduce the cost of the FFT, which spans the whole plane in either case;
they reduce the number of independent trials entering the false-alarm
probability below. A window narrower than the true spread of a parameter
excludes the correct solution. A rate window reaching beyond
``\pm 1/(2\Delta t)`` admits peaks that are aliases of the rate's own wrap.

## Search algorithms

[`FullGrid`](@ref) evaluates the transform above on a single grid spanning the
whole frequency axis. When narrow bands are spread over a wide span that grid
is mostly zeros: its size is ``N_\mathrm{grid} = s N_\mathrm{chan}`` with
``s \gg 1``, and both the transform and the memory traffic scale with it.

[`HierarchicalMBD`](@ref) evaluates the same filter through the band
factorization derived above, in two stages:

1. For each band ``b``, a small two-dimensional FFT over that band's channels
   and the time axis gives ``D_b`` on a coarse grid of in-band delay — the
   single-band delay (SBD) — and rate. Only the windowed block is kept.
2. For each SBD bin, a small one-dimensional FFT over the band origins,
   gridded on a coarse common spacing ``\Delta_{bc}``, gives the multi-band
   delay (MBD) and rate. The peak is then taken over the resulting
   three-dimensional (SBD, MBD, rate) cube.

The second stage's transform is periodic with ambiguity
``\mathcal{A} = 1/\Delta_{bc}``, and the SBD bin width ``1/B_\mathrm{band}``
can be comparable to ``\mathcal{A}``, so the correct cycle cannot be chosen by
rounding to the SBD estimate. Every candidate
``\tau = \mathrm{mbd} + k\mathcal{A}`` lying within about 1.5 SBD bins of the
SBD estimate and inside the delay window is instead evaluated on the exact
filter, and the largest is retained. The exact filter responds to the in-band
slope, which is the quantity that distinguishes the candidates. The selected
candidate is then refined as on the full-grid path.

`algorithm = :auto` (the default) selects `HierarchicalMBD` when the common
grid would exceed four times the occupied channel count, on a sorted,
non-degenerate axis, and `FullGrid` otherwise. Contiguous data takes the full
grid; sparse layouts take the hierarchical path. `HierarchicalMBD` reverts to
the full grid where the band geometry cannot support the factorization, in
particular on an axis with fewer than two bands. Both paths use the same noise
and detection statistics, so their detections are directly comparable.

## Detection statistics

Under the null hypothesis ``V = n``, the filter is a weighted sum of
independent circular Gaussians, so ``D`` is itself circular Gaussian with

```math
\mathbb{E}[D] = 0,
\qquad
\mathrm{Var}(D) = \sum w^2 \sigma^2 = \sum w^2 \frac{1}{w} = \sum w .
```

Hence ``|D|^2/\sum w`` is exponentially distributed with unit mean, and for the
normalized statistic ``\rho = |D|/\sqrt{\sum w}``,

```math
\Pr(\rho > s) = e^{-s^2}.
```

This identity holds only where the weights are true inverse variances. Raw
correlator output frequently carries uniform or uncalibrated weights, for which
``\sqrt{\sum w}`` misestimates the noise and hence every probability derived
from it. The search therefore estimates the denominator from the data. It takes
a strided sample of ``|D|^2`` over the search plane and uses the median, since
for an exponential distribution ``\mathrm{median} = \ln 2 \cdot \mathrm{mean}``:

```math
\widehat{\mathrm{Var}}(D) = \frac{\mathrm{median}(|D|^2)}{\ln 2},
\qquad
\mathrm{SNR} = \frac{|D_\mathrm{ref}|}{\sqrt{\widehat{\mathrm{Var}}(D)}} .
```

The median is used in place of the mean so that the peak and its sidelobes do
not inflate the estimate. The sample is taken over the whole plane rather than
over the search window, which lies on the sidelobe ridge of the fringe. Where
the weights are calibrated the estimate reduces to
``|D_\mathrm{ref}|/\sqrt{\sum w}``, and with too few samples to form a median it
falls back to ``\sum w``.

A search reports a peak whether or not a fringe is present, so each detection
carries the probability that noise alone produced it. With ``N_c`` independent
cells searched and single-cell exceedance ``e^{-\mathrm{SNR}^2}``,

```math
\mathrm{PFA} = 1 - \left(1 - e^{-\mathrm{SNR}^2}\right)^{N_c},
```

computed by [`fringe_pfa`](@ref). The number of independent cells is the window
span divided by the resolution on each axis,

```math
N_c = \big(\Delta\tau_\mathrm{win} \cdot B\big) \times \big(\Delta\dot r_\mathrm{win} \cdot T\big),
```

each factor clamped below at one and above at the number of gridded samples on
that axis, since zero-padding refines the sampling without adding independent
trials. [`fringe_snr_cut`](@ref) inverts the relation; the implied threshold
grows as ``\sqrt{\log(N_c/\mathrm{PFA})}``, so it varies slowly with both
arguments.

``N_c`` is counted over a family of searches rather than one:
[`search_scan`](@ref) uses ``n_\mathrm{baseline} \times n_\mathrm{pol} \times
n_\mathrm{groups}``, a Bonferroni correction over a scan, or over a whole track
when the caller supplies its scan count. A recorded PFA is therefore directly
comparable with [`Stationization`](@ref)'s `pfa_max`, and is not a per-search
quantity.

No threshold is applied during the search. Every block with usable data yields
a detection, and acceptance is decided in the station solve.

## Fringe closure

Each measured quantity is a difference between two (station, feed) values. For
baseline ``(a,b)`` and correlation product ``p`` with feeds ``(f_a, f_b)``,

```math
\tau^p_{ab} = \tau_{a,f_a} - \tau_{b,f_b},
\qquad
\dot r^p_{ab} = \dot r_{a,f_a} - \dot r_{b,f_b},
\qquad
\varphi^p_{ab} = \varphi_{a,f_a} - \varphi_{b,f_b},
```

so each observable gives a linear system ``M x = y`` on a graph of
``2 n_\mathrm{ant}`` nodes, whose rows are ``e_i - e_j``. Each system is solved
by weighted least squares.

### Row weights

A row's weight is ``1/\sigma^2``, with ``\sigma`` the Cramér–Rao bound of that
measurement. Let ``\rho`` denote the signal-to-noise ratio. With ``A`` and
``\varphi`` profiled out of ``\chi^2`` as above, the Fisher information for the
phase, and for the coefficient ``c`` of a linear phase term
``2\pi c (x - \bar x)`` in a coordinate of weighted RMS spread ``\sigma_x``, is

```math
I_{\varphi\varphi} = \rho^2,
\qquad
I_{cc} = (2\pi)^2 \sigma_x^2 \rho^2 ,
```

giving

```math
\sigma_\varphi = \frac{1}{\rho},
\qquad
\sigma_\tau = \frac{1}{2\pi\sigma_\nu\rho},
\qquad
\sigma_{\dot r} = \frac{1}{2\pi\sigma_t\rho},
```

with ``\sigma_\nu`` and ``\sigma_t`` the RMS spreads of the channel frequencies
and of the timestamps. For channels uniformly filling a band of width ``B`` this
gives ``\sigma_\nu = B/\sqrt{12}``. The actual spread is used instead, which
remains correct for sparse or unevenly spaced bands, where the contiguous
expression overstates the lever arm.

A systematic floor is added in quadrature,
``w = 1/(\sigma_\mathrm{CRB}^2 + \sigma_\mathrm{sys}^2)``. Without it, data free
of systematics drives the normalized residuals to the numerical noise floor,
and the robust loss below then reweights rounding error.

A weighted least-squares solution is invariant under scaling every weight of a
system by a common constant, and ``\sigma_\nu``, ``\sigma_t`` are common to all
rows of one scan, so these factors do not affect the fit. They fix the meaning
of the normalized residual ``z = r\sqrt{w}``, on which the robust loss
thresholds.

### Cross-hand rows

All four correlation products contribute rows. A cross-hand row connects a
feed-1 node to a feed-2 node, merging the two feeds into a single connected
component. The inter-feed offset is then determined by the data, and no
separate alignment step is required. Which parameters a row enters follows from
the model's feed tying; the product is not treated as a special case.

The phase system is an exception when the model carries no feed-relative phase
term. This is the default, the R–L offset being left in the data for a
subsequent polarization fit. The four product families are then mutually
inconsistent: a QQ row sits a station-based offset away from its PP
counterpart, and a cross-hand row adds the source's cross-hand phase. Fitting
them to a single shared parameter returns a weighted compromise between them.
That system is therefore augmented with per-(scan, station) nuisance feed-2
offset parameters, so that the shared parameter is the feed-1 phase, and
cross-hand rows are withheld from it. What those rows would constrain beyond
the parallel hands is the common mode of the offsets, which is discarded.

### Gauge freedom

The rows ``e_i - e_j`` annihilate any vector constant on a connected component
of the graph, so ``M`` has a null space of dimension equal to the number of
components and the system is rank-deficient. One constraint row per component
is supplied by an
[`AbstractGauge`](@ref Gustavo.Calibration.AbstractGauge):
[`PinAntenna`](@ref Gustavo.Calibration.PinAntenna) sets one node to zero, and
[`ZeroSumPhase`](@ref Gustavo.Calibration.ZeroSumPhase) constrains a weighted
sum of the nodes.

The constraint changes no gauge-invariant quantity: baseline differences,
closure phases and the applied calibration are unaffected. It does determine
which per-station values are reported, and hence whether values from different
scans are comparable.

### Detection threshold and robust weighting

`pfa_max` is the single detection threshold, and it governs connectivity: a
detection at or below it joins its two stations into one fringe group, and a
(station, scan) is calibrated only if such a detection reaches it. A detection
above the threshold still contributes a row, with its systematic floors
multiplied by `weak_sys_scale`, but cannot join stations into a group. A
marginal baseline is therefore measured at the fringe position determined by
the accepted detections, and does not establish one itself.

A false fringe is inconsistent with closure, so it appears as a large
normalized residual ``z``. The systems are solved by iteratively reweighted
least squares: after each solve a row's weight is multiplied by ``\rho'(u)`` at
``u = (z/\mathrm{loss\_scale})^2`` for the chosen loss ([`SoftL1`](@ref) by
default, ``\rho(u) = 2(\sqrt{1+u} - 1)``; [`LeastSquares`](@ref) disables the
reweighting), and the system is re-solved. Because the weights come from the
noise model rather than from the spread of the residuals, `loss_scale`
corresponds to the same effective threshold in units of ``\sigma`` whatever the
size of the system. [`station_closure_residuals`](@ref) reports the residuals
that remain.

Stations the solve does not constrain retain ``\theta = 0``, the identity gain.
They are reported as uncovered and are to be flagged on that basis.

Where the solve is scan-local, every block is re-measured at the delay and rate
predicted by the solution, without a threshold. A station within the fringe
group then has its baselines measured at a known fringe position, and so to
arbitrarily low signal-to-noise. The significance of these steered measurements
is recorded and is not used as a threshold.

## Diagnostics

[`baseline_fringe_map`](@ref) returns the windowed delay–rate surface itself as
a [`FringeSearchMap`](@ref), rather than only its maximum. This is required to
determine why a detection occurred at a given delay and rate, and whether a
competing peak of comparable height was present.
[`fringe_search_map`](@ref) builds one from the detections
recorded on a solution, [`suspect_fringes`](@ref) lists detections whose
statistics are inconsistent, and [`delay_closure`](@ref) evaluates delay
closure around triangles. With a Makie backend loaded,
[`plot_fringe_search`](@ref) plots the surface.
