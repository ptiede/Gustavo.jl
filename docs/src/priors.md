```@meta
CurrentModule = Gustavo.Fring
```

# [Fitting under priors](@id fitting-under-priors)

A component's prior (see [Specifying gain models](@ref specifying-models))
relates its parameter values along one axis: time for the adhoc phase,
frequency for the bandpass. This page derives how a solver fits a track of
values under such a prior: the maximum a posteriori (MAP) values, the Kalman
filter and Rauch–Tung–Striebel (RTS) smoother that compute them for the
Ornstein–Uhlenbeck (OU) and random-walk priors, and the type-II MAP estimate of the prior's own
hyperparameters. The last section says where each solver uses them.

## One track

A solver reduces its data to real tracks: an unwrapped phase or a
log-amplitude per sample, with an inverse-variance weight. Write one track as

```math
y_k = L + f_k + \varepsilon_k, \qquad \varepsilon_k \sim \mathcal N(0, r_k), \quad r_k = 1/w_k,
```

for samples ``k = 1, \dots, n`` at coordinates ``x_1 < \dots < x_n`` (seconds
or Hz). ``f`` is the structure the prior describes, ``L`` a level the prior
says nothing about, and ``\varepsilon`` the measurement noise. A sample whose
value is not finite, or whose weight is not positive and finite, carries no
data. The spacing ``x_{k+1} - x_k`` need not be uniform.

For the bandpass, a track is one spectral window of one (station, feed, time
segment), sampled at the frequency segments' mean channel frequencies, and
``L`` is the level component's value for that window. For the adhoc phase, a
track is one (station, feed node) over the accumulation periods (APs) of one
scan, and ``L`` is the per-scan constant, a gauge that the adhoc step removes
afterwards (see [`solve_adhoc_phasing`](@ref)).

## The OU prior and its MAP values

Under an [`OUPrior`](@ref Gustavo.Calibration.OUPrior)`(; scale = τ, σ)`, ``f``
is a zero-mean Gaussian process with covariance

```math
K_{jk} = \sigma^2 \exp\!\big(-|x_j - x_k| / \tau\big).
```

With ``L`` fixed, the posterior of ``f`` is Gaussian, so its MAP value is also
its posterior mean:

```math
\hat f = K H^\top \big(H K H^\top + R\big)^{-1} (y - L),
```

where ``H`` selects the samples with data and ``R = \mathrm{diag}(r_k)``. At
samples without data this interpolates, or extrapolates toward zero, from the
covariance. Solved densely this costs ``O(n^3)``; the OU process has a
structure that brings it to ``O(n)``.

### The OU process is Markov

The exponential covariance is that of a first-order autoregression with
coefficients that depend on the spacing. With ``\Delta_k = x_k - x_{k-1}``,

```math
f_1 \sim \mathcal N(0, \sigma^2), \qquad
f_k = a_k f_{k-1} + \eta_k, \quad a_k = e^{-\Delta_k/\tau}, \quad
\eta_k \sim \mathcal N\!\big(0,\, q_k\big), \quad q_k = \sigma^2 (1 - a_k^2)
```

(`ou_step`). Every ``f_k`` then has variance ``\sigma^2``, and
``\mathrm{cov}(f_j, f_k) = \sigma^2 \prod a = \sigma^2 e^{-|x_j - x_k|/\tau}``,
the covariance above. A long gap makes ``a_k`` small, so the process forgets
across it; a short one keeps neighboring values close. This state-space form of the
exponential (Matérn-1/2) covariance is standard; see Särkkä and Solin,
*Applied Stochastic Differential Equations* (2019).

### Kalman filter

Both priors on this page are linear-Gaussian state-space models: a small state
``s_k`` per sample, whose first entry is ``f_k``, evolving over each step as

```math
s_k = A_k s_{k-1} + \eta_k, \qquad \eta_k \sim \mathcal N(0, Q_k).
```

For the OU prior the state is ``f_k`` alone, with ``A_k = a_k`` and
``Q_k = q_k`` ([`OUModel`](@ref)); the random-walk prior below has ``m``
states. One filter and smoother fit either.

The forward pass ([`kalman_filter`](@ref)) carries the mean ``\mu_k`` and
covariance ``P_k`` of ``s_k`` given samples ``1, \dots, k``. Each step predicts

```math
\mu_k^- = A_k \mu_{k-1}, \qquad P_k^- = A_k P_{k-1} A_k^\top + Q_k,
```

starting from the prior's start (for OU, ``\mu_1^- = 0``,
``P_1^- = \sigma^2``), and, if sample ``k`` carries data, updates with the
innovation ``v_k`` and its variance ``S_k``:

```math
v_k = y_k - L - (\mu_k^-)_1, \quad S_k = (P_k^-)_{11} + r_k, \quad
K_k = P_k^- e_1 / S_k, \quad
\mu_k = \mu_k^- + K_k v_k, \quad P_k = P_k^- - K_k S_k K_k^\top,
```

with ``e_1`` the first unit vector; ``P_k`` is computed in Joseph form so it
stays symmetric and positive definite. A sample without data is predicted
through: ``\mu_k = \mu_k^-``, ``P_k = P_k^-``.

### RTS smoother

The backward pass ([`rts_smooth`](@ref)) conditions each state on the samples
after it as well:

```math
G_k = P_k A_{k+1}^\top (P_{k+1}^-)^{-1}, \qquad
\hat s_k = \mu_k + G_k \big(\hat s_{k+1} - \mu_{k+1}^-\big),
```

starting from ``\hat s_n = \mu_n``. The first entry of ``\hat s_k`` equals the
dense posterior mean above exactly, in ``O(n)`` time and memory, for any
spacing ([`smooth_track`](@ref), [`smooth_ou_track`](@ref)).

## The marginal likelihood

The innovations are independent, so the forward pass also gives the likelihood
of the data with ``f`` integrated out:

```math
\log p(y \mid \tau, \sigma, L) = -\frac12 \sum_{k \text{ with data}}
\Big[\log(2\pi S_k) + \frac{v_k^2}{S_k}\Big].
```

This is what the hyperparameters are estimated from.

### The level

The filter is linear in its observations, so the innovations of ``y - L`` are
``v^y_k - L\, v^1_k``, where ``v^y`` are the innovations of ``y`` and ``v^1``
those of a track of ones, both run with the same gains (`_level_sums`).
With

```math
b = \sum_k \frac{v^y_k v^1_k}{S_k}, \qquad c = \sum_k \frac{(v^1_k)^2}{S_k},
```

the likelihood is quadratic in ``L`` and maximized at the generalized least
squares level ``\hat L = b/c``. Tracks that share one level (the spectral
windows of one level segment) sum their ``b`` and ``c``.

``L`` is not known when the hyperparameters are estimated, and plugging in
``\hat L`` would bias ``\sigma`` low on short tracks, since the level absorbs
part of the structure. Instead ``L`` is integrated out under a flat prior (the
restricted likelihood), which adds, per level,

```math
\frac{b^2}{2c} - \frac12 \log c
```

to the log likelihood of the zero-mean tracks, up to a constant. Given the
hyperparameters, the fit is then ``\hat L`` plus the MAP of ``y - \hat L``.

## Type-II MAP of the hyperparameters

Each of `scale` and `σ` in an `OUPrior` is either a number, held fixed, or a
hyperprior density (any object implementing DensityInterface's
`logdensityof`, such as a Distributions.jl distribution). A hyperprior is
resolved by maximizing the marginal likelihood times the hyperprior density,
searched over ``(\log\tau, \log\sigma)``:

```math
\max_{\tau, \sigma}\;
\log p(y \mid \tau, \sigma) + \log \pi_\tau(\tau) + \log \pi_\sigma(\sigma)
+ \log\tau + \log\sigma,
```

where ``p(y \mid \tau, \sigma)`` is the restricted likelihood above when the
track has a level, the zero-mean one otherwise, summed over every track the
solver pools, and ``\log\tau + \log\sigma`` is the Jacobian of the change to
logarithmic coordinates. This is type-II MAP (also called empirical Bayes with
a hyperprior): the values are integrated out, the hyperparameters are
maximized. The track is then fit at the resolved values, so the hyperparameters'
own uncertainty does not enter the fitted values.

The search is a Nelder–Mead simplex (`_map_ou_hypers`). ``\tau`` is
confined to between one median sample spacing and ten times the span of the
samples (for pooled tracks, the widest such range any one track allows), since
outside it the data cannot distinguish values of ``\tau``. The search for
``\sigma^2`` starts from the tracks' weighted variance less the expected noise
contribution, and is floored at ``10^{-8}``.

The hyperpriors must be proper. With a flat hyperprior, the likelihood alone
drives ``\sigma \to 0`` on a track whose structure is below the noise, so a weak
track would be fit as a constant regardless of what it is. A proper
hyperprior on ``\sigma`` keeps it at a plausible value. The adhoc default,
[`default_adhoc_prior`](@ref), puts LogNormal hyperpriors on ``\tau`` (median
10 s) and ``\sigma`` (median 0.3 rad), each with log-standard deviation 1.

## The random-walk prior

A [`RandomWalkPrior`](@ref Gustavo.Calibration.RandomWalkPrior)`(; order = m, σ)`
is the ``(m-1)``-times integrated Brownian motion along the coordinate. Its
state at each sample is ``s_k = (f_k, f'_k, \dots, f^{(m-1)}_k)``, and over a
step ``\Delta_k = |x_k - x_{k-1}|`` it evolves as

```math
s_k = A_k s_{k-1} + \eta_k, \qquad
(A_k)_{ij} = \frac{\Delta_k^{\,j-i}}{(j-i)!}\ (j \ge i), \qquad
(Q_k)_{ij} = \frac{\sigma^2\, \Delta_k^{\,2m-1-i-j}}{(2m-1-i-j)\,(m-1-i)!\,(m-1-j)!},
```

with ``\eta_k \sim \mathcal N(0, Q_k)`` and indices ``i, j`` from 0. For
``m = 1`` this is ``f_k - f_{k-1} \sim \mathcal N(0, \sigma^2 \Delta_k)``;
for ``m = 2`` the posterior mean is a cubic smoothing spline. ``\sigma^2`` is
per unit of ``x^{2m-1}``. Because the prior is defined in the coordinate, not
by counting samples, it holds for uneven spacing and gaps and keeps its meaning
when the segments are made coarser or finer.

The starting value and its first ``m - 1`` derivatives are left free (a flat
start), so the prior is improper in those directions ([`RandomWalkModel`](@ref)).
The same Kalman filter and RTS smoother fit it. Until the samples determine the
state, the filter carries it in information form, the density
``\exp(c + \eta^\top s - \tfrac12 s^\top \Lambda s)`` of the samples so far and
the state, starting from ``\Lambda = 0``, ``\eta = 0``, ``c = 0``. A sample
with data adds ``e_1 e_1^\top / r_k`` to ``\Lambda``, ``e_1 y_k / r_k`` to
``\eta`` and ``-\tfrac12[\log(2\pi r_k) + y_k^2/r_k]`` to ``c``; a step, with
``M = A_k^{-\top} \Lambda A_k^{-1}``, ``\tilde\eta = A_k^{-\top}\eta`` and
``F = I + Q_k M``, sets

```math
\Lambda \leftarrow F^{-\top} M, \qquad \eta \leftarrow F^{-\top} \tilde\eta, \qquad
c \leftarrow c + \tfrac12 \tilde\eta^\top Q_k F^{-\top} \tilde\eta
- \log\lvert\det A_k\rvert - \tfrac12 \log\det F,
```

which needs no ``Q_k^{-1}``. Once ``\Lambda`` is positive definite, after ``m``
samples with data, the filter continues in covariance form from
``P = \Lambda^{-1}``, ``\mu = P\eta``. This is exact, not a large finite
starting variance. The smoother runs back through those first samples with the
conditional of ``s_k`` given ``s_{k+1}`` in the same form.

The likelihood with the flat start integrated out (the restricted likelihood,
as for the level above) is ``c + \tfrac12 \eta^\top \Lambda^{-1} \eta -
\tfrac12 \log\det\Lambda + \tfrac m2 \log 2\pi`` at the switch, plus the usual
innovation terms after it (`_random_walk_loglik`). It is taken under the
Lebesgue measure on the first sample's state, derivatives in units of the
coordinate. The coordinate is first shifted and rescaled by its median spacing
so the states have comparable magnitudes, and the fit is in at least `Float64`.
A track with fewer than ``m`` samples with data does not determine the walk and
keeps its measured values. Since the walk leaves its own level free, it cannot
be separated from a level component, and a random walk beside a level is
rejected.

`σ` may be a hyperprior, resolved by the same type-II MAP as an OU prior's,
maximizing

```math
\sum_{\text{tracks}} \log p_R(y \mid \sigma) + \log \pi_\sigma(\sigma) + \log\sigma
```

over ``\log\sigma``, where ``p_R`` is the restricted likelihood above. The
search starts from the variance of the ``m``-th differences of the samples with
data, less their noise contribution, set equal to the variance a walk gives an
``m``-th difference at the median spacing ``h``, ``c_m \sigma^2 h^{2m-1}``
(``c_1 = 1``, ``c_2 = 2/3``). There are no search bounds, and the hyperprior
must be proper: as ``\sigma \to 0`` the restricted likelihood tends to that
of a polynomial of degree ``m - 1`` fit to the samples, which stays finite, so
with a flat hyperprior a track whose structure is below the noise would be fit
as a polynomial. The solvers pool tracks as they do for an OU prior; since a
walk has no separate level, no levels are integrated out (`_estimate_hypers`).

## Where the solvers use this

**Bandpass, [`PerTrackSmoother`](@ref).** Each (station, feed, time segment)
has one track per spectral window along frequency. A hyperprior is resolved
once for all of that track's windows together, with the level of each level
segment integrated out; then the levels are estimated and each window is fit.

**Bandpass, [`JointSmoother`](@ref).** The same fit runs inside each sweep of
the gain update, on the tracks linearized around the current gains, with the
hyperparameters and levels re-estimated every sweep. Iterated to convergence
this is MAP estimation of the gains under both priors, with type-II MAP
hyperparameters.

**Adhoc phase, [`PerTrackAdhocSmoother`](@ref).** After the per-AP solve, each
(station, feed node) track of a scan is fit on its own: hyperparameters per
(station, node, scan) with the level integrated out, then the level plus the
MAP of the zero-mean part. The per-scan mean is then removed from every track,
since a per-station constant trades against the baseline source terms.

**Adhoc phase, [`JointKalmanSmoother`](@ref).** The state is the vector of
station phases ``\theta_1, \dots, \theta_N``, each an independent OU process
with its own ``(\tau_i, \sigma_i)``, so the transition is diagonal with entries
``a_{i,k} = e^{-\Delta_k/\tau_i}``. Each gated (baseline, correlation product)
cell at an AP is an observation

```math
y_{ab} = \theta_a - \theta_b + \varepsilon_{ab}, \qquad
\varepsilon_{ab} \sim \mathcal N\!\big(0, 1/\mathrm{SNR}_{ab}^2\big),
```

with its source term already removed. The same filter and smoother, in matrix
form ([`kalman_mv_filter`](@ref), [`rts_smooth_mv`](@ref)), give the
posterior mean of every station's track from the baseline data directly,
without first solving each AP. Differences leave the sum of all station
phases unconstrained; only the OU prior holds it near zero, and the result is
re-referenced to the anchor station at every AP afterwards. Each observed phase
is moved by a multiple of 2π toward the current model before the filter runs,
and this is repeated a few times as the model improves.

Each station's hyperparameters are resolved by type-II MAP on its track from
the per-AP solve, with the level integrated out, and the track is centered on
its level for the filter. The anchor's per-AP track is zero by construction,
and an unobserved station has none, so both take the median of the other
stations' resolved values, unless their own are fixed.

In both adhoc solvers, the complex-domain refinement (see
[`AdhocOptions`](@ref)) repeats the fit on observations linearized around the
current tracks, re-resolving the hyperparameters each pass.
