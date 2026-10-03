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
phase-only [`GainModel`](@ref Gustavo.Calibration.GainModel), and the search
and station-solve options are fields of [`BaselineFringeFit`](@ref Gustavo.BaselineFringeFit). Phase sense,
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
[`search_scan`](@ref) applies it to every cross baseline and feed pair of a
scan group, joining each baseline's spectral windows into one plane so the
delay search spans them all; autocorrelations are excluded. Each search returns a [`Detection`](@ref),
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

### Noise model

Each visibility is ``V_k = s_k + n_k``, where the noise ``n_k`` has independent
real and imaginary parts of equal variance, and the weight is the inverse of
that per-component variance:

```math
\operatorname{Re} n_k,\ \operatorname{Im} n_k \sim \mathcal{N}\!\left(0,\ \frac{1}{w_k}\right),
\qquad
\mathbb{E}|n_k|^2 = \frac{2}{w_k}.
```

This is the convention of the radiometer equation: XRadio's FITS-IDI reader
writes ``w = f \cdot 2\Delta\nu\,\tau\,\eta_a\eta_b`` for a correlation
coefficient, the inverse variance of each of its real and imaginary parts. A
weight column that departs from it (a station whose weights are off by a
factor) is corrected before fitting with `scale_weights!`.

### The matched filter under noise

The matched filter at a trial (delay, rate) is
``D = \sum_k w_k V_k e^{-i\theta_k}`` with ``\theta_k`` the trial phase of
sample ``k``. Under the null hypothesis ``V = n``, ``D`` is a weighted sum of
independent circular Gaussians, and rotating a circular Gaussian by
``e^{-i\theta_k}`` leaves its distribution unchanged, so each component of
``D`` is Gaussian with

```math
\mathbb{E}[\operatorname{Re} D] = \mathbb{E}[\operatorname{Im} D] = 0,
\qquad
\operatorname{Var}(\operatorname{Re} D) = \operatorname{Var}(\operatorname{Im} D)
= \sum_k w_k^2 \cdot \frac{1}{w_k} = \sum_k w_k ,
```

and the two components are independent.

### SNR

The fringe amplitude is ``A = |D|/\sum w`` and the noise on each component of
``D/\sum w`` is ``1/\sqrt{\sum w}``, so the signal-to-noise ratio is the
amplitude over its per-component noise,

```math
\mathrm{SNR} = \frac{|D|}{\sqrt{\sum w}} .
```

At high SNR the phase of ``D`` has standard deviation ``1/\mathrm{SNR}`` (the
noise component perpendicular to ``D``, divided by ``|D|``), and so does
``\log A``. The bandpass and adhoc steps use the same quantity for a coherent
sum of visibilities, ``|\sum w V|^2 / \sum w``, as the inverse variance of its
phase and log-amplitude.

### Single-cell exceedance

Under the null, ``\mathrm{SNR}^2 = (\operatorname{Re} D)^2/\sum w + (\operatorname{Im} D)^2/\sum w``
is the sum of the squares of two independent standard normals, a ``\chi^2``
variable with two degrees of freedom, which is exponential with mean 2.
Equivalently, ``\mathrm{SNR}`` is Rayleigh distributed with unit scale. The
probability that one cell of pure noise reaches ``s`` is therefore

```math
p_1(s) = \Pr(\mathrm{SNR} > s) = e^{-s^2/2}.
```

### Probability of false alarm

A search reports a peak whether or not a fringe is present, so each detection
carries the probability that noise alone produced it. The search takes the
largest of ``N_c`` independent cells; the probability that none of them
exceeds ``s`` is ``(1 - p_1)^{N_c}``, so

```math
\mathrm{PFA}(s) = 1 - \left(1 - e^{-s^2/2}\right)^{N_c}
\;\approx\; N_c\, e^{-s^2/2} \quad (\mathrm{PFA} \ll 1),
```

computed by [`fringe_pfa`](@ref). The number of independent cells is the window
span divided by the resolution on each axis,

```math
N_c = \big(\Delta\tau_\mathrm{win} \cdot B\big) \times \big(\Delta\dot r_\mathrm{win} \cdot T\big),
```

each factor clamped below at one and above at the number of gridded samples on
that axis, since zero-padding refines the sampling without adding independent
trials. [`fringe_snr_cut`](@ref) inverts the relation,

```math
s = \sqrt{-2 \log\!\left(1 - (1 - \mathrm{PFA})^{1/N_c}\right)}
\;\approx\; \sqrt{2 \log(N_c / \mathrm{PFA})},
```

so the implied threshold varies slowly with both arguments: for
``N_c = 10^4`` cells, ``\mathrm{PFA} = 10^{-3}`` corresponds to
``\mathrm{SNR} \approx 5.7``.

``N_c`` is counted over a family of searches rather than one:
[`search_scan`](@ref) uses ``n_\mathrm{baseline} \times n_\mathrm{pol}``, a
Bonferroni correction over the scan. A recorded PFA is therefore directly
comparable with [`Stationization`](@ref)'s `pfa_max`, is not a per-search
quantity, and does not depend on how many other scans a solve holds.

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
is supplied by the step's `gauge`, an
[`AbstractGauge`](@ref Gustavo.Calibration.AbstractGauge):
[`PinAntenna`](@ref Gustavo.Calibration.PinAntenna) sets one node to zero, and
[`ZeroSumPhase`](@ref Gustavo.Calibration.ZeroSumPhase) constrains a weighted
sum of the nodes.

When a station's value is a sum of several components, such as a per-scan
delay plus a feed-2 offset, the components are formed per model column: a
detection links each column to the same component's column at the other
station. A component receives a constraint row when shifting all of its
columns by a constant leaves every accepted detection unchanged. Per-scan
columns therefore give one constraint per scan, even when a column spanning
scans (a track-global feed-2 offset) couples them, and a feed-2 offset receives
its own constraint only when no accepted cross-hand detection determines it.
Under `PinAntenna` the reference station then reads zero in every scan, and its
feed-2 offset reads zero when the data carry no cross hands. Any other
undetermined combination of columns, such as two feed-2 offsets that every
detection sees only as their sum, is an error.

The constraint changes no gauge-invariant quantity: baseline differences,
closure phases and the applied calibration are unaffected. It does determine
which per-station values are reported, and hence whether values from different
scans are comparable.

[`ByComponent`](@ref Gustavo.Calibration.ByComponent) chooses a gauge per
model component, by the component names of the step's model (those a solution
lists, such as `fringe.phase.rate`):

```julia
gauge = ByComponent((; rate = PinAntenna("A2")); default = PinAntenna("A1"))
sol = fit(BaselineFringeFit(; gauge), ps)
```

The step rejects a name its model does not have, listing the components it
does have.

A new convention subtypes `AbstractGauge`. Each freedom reaches it as a
[`GaugeFreedom`](@ref Gustavo.Calibration.GaugeFreedom): its nodes and, per
node, the station, feed, scan, component path and observable (`:delay`,
`:rate` or `:phase`), with the shift direction and the row weight on each node.
The gauge implements
[`gauge_constraint`](@ref Gustavo.Calibration.gauge_constraint)`(g, freedom)`,
returning a row over the freedom's nodes and its target value; the target need
not be zero. A convention that ties freedoms together, such as holding the
reference's value equal across scans, implements
[`gauge_constraints`](@ref Gustavo.Calibration.gauge_constraints)`(g, freedoms)`
instead, returning `C` and `d` for all freedoms of one system.

The phase-unwrap seed and the joint bandpass solve also need one node per
freedom; [`gauge_anchor`](@ref Gustavo.Calibration.gauge_anchor) supplies it,
by default the first station of
[`gauge_station_order`](@ref Gustavo.Calibration.gauge_station_order) present
in the freedom. The joint bandpass solve pins that node, and the adhoc phase
solve pins the first station of `gauge_station_order` it sees in the scan;
neither reads a gauge's constraints.

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

A fringe solution's diagnostics are labeled arrays keyed by scan name,
station and feed: [`fringe_snr_table`](@ref) (per-scan peak SNR and
false-alarm probability), [`fringe_detections`](@ref) (every measured
baseline and product), [`fringe_station_solutions`](@ref) (per-scan station
delay, rate and phase) and [`fringe_station_flags`](@ref) (stations a scan
left unconstrained). Per-scan solutions from [`mapsets`](@ref) combine with
[`cat_scans`](@ref):

```julia
sols = mapsets(g -> fit(BaselineFringeFit(; gauge), g), groupby(ps, ByScan()))
det = cat_scans(fringe_detections.(values(sols)))
findall(det.detected .& (det.pfa .> 1e-6))     # accepted, but marginal
```

The data themselves show what a solution removed. XRadio's `average`
reduces a scan to its inverse-variance weighted means, and
[`baseline_spectra`](@ref) lays a time-averaged set out by station pair, feed
pair and frequency across its spectral windows. Spectra of the raw and the
corrected scan show the delay slope a good fit flattens, and plot directly
with DimensionalData's Makie recipes. [`baseline_delays`](@ref) and
[`delay_closure`](@ref) test the station-based structure: delays close around
triangles before correction and vanish after it. [`freq_group_coherence`](@ref)
localises residual structure to a frequency group.

```julia
using XRadio: average, ByScan
before = baseline_spectra(average(g, ByScan()))
after = baseline_spectra(average(calibrate(fr, g; flag_bad = false, apply_flags = false), ByScan()))
series(angle.(after.vis[Scan = 1, FeedPair = At((1, 1))]))   # one line per baseline
delay_closure(before)
```

[`baseline_fringe_map`](@ref) returns the windowed delay–rate surface itself as
a [`FringeSearchMap`](@ref), rather than only its maximum. This is required to
determine why a detection occurred at a given delay and rate, and whether a
competing peak of comparable height was present. With a Makie backend loaded,
[`plot_fringe_search`](@ref) plots the surface.
