```@meta
CurrentModule = Gustavo.Fring
```

# [Fitting under priors](@id fitting-under-priors)

A component's prior (see [Specifying gain models](@ref specifying-models))
relates its parameter values along one axis: time for the adhoc phase,
frequency for the bandpass. This page derives how a solver fits a track of
values under such a prior: the maximum a posteriori (MAP) values, the Kalman
filter and Rauch–Tung–Striebel (RTS) smoother that compute them for the
Ornstein–Uhlenbeck (OU) and random-walk priors, the type-II MAP estimate of the prior's own
hyperparameters, and the choice of a phase track's 2π branches. The last section
says where each solver uses them.

## One track

A solver reduces its data to real tracks: a phase (on branches chosen as in
[Phase tracks and their 2π branches](@ref phase-branches)) or a log-amplitude per
sample, with an inverse-variance weight. Write one track as

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

### [The level](@id prior-level)

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

The search is a Nelder–Mead simplex (`_map_ou_hypers`), stopped once the
negative log posterior agrees across the simplex to ``10^{-8}`` nats, which
places the optimum within about ``10^{-4}`` of its own standard error: an
optimum resolved only relative to the objective's size would wander between
calls by a sizable fraction of its standard error on long tracks. ``\tau`` is
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
keeps its measured values. Since a walk with a flat start leaves its own level
free, it cannot be separated from a level component, and such a walk beside a
level is rejected.

An `init` gives the first sample's state a Gaussian prior
``s_1 \sim \mathcal N(\mu_0, P_0)`` instead, in units of the coordinate and
relative to the track's level, and the walk is then proper. The filter starts
in covariance form from it, exactly as for OU, and the likelihood is the
ordinary one. A level beside it is estimated, and integrated out for the
hyperparameters, as for OU above, so the init's spread describes how far the
track's initial state may lie from its level.

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
as a polynomial. The solvers pool tracks as they do for an OU prior. A walk
with a flat start has no separate level, so no levels are integrated out; with
an `init` they are, as for OU (`_estimate_hypers`).

## [Phase tracks and their 2π branches](@id phase-branches)

A phase is measured from a complex gain, so each sample's value ``\theta_k`` is
known only modulo ``2\pi``. The likelihood does not depend on the branch, but a
prior on the real-valued track does: moving one sample by ``2\pi`` leaves the
data's fit unchanged and adds a step the prior penalizes. Fitting a phase track
under a prior therefore means choosing an integer ``m_k`` for each sample as well,
the track being observed as

```math
\theta_k + 2\pi m_k = L + f_k + \varepsilon_k .
```

Choosing ``m_k`` by stepping from each sample to its neighbor (nearest-branch
unwrapping, `unwrap_phase_track`) works while the noise on a step stays well
inside ``\pm\pi``. At lower signal-to-noise ratio one noisy step puts every later
sample a turn off; the smoother then fits that step as a smooth ramp, and an
iterative solve that refines each sample about the current fit keeps it, since
removing it means moving many samples by ``2\pi`` at once. The branches must be
chosen under the prior, not before it.

### A sum of Gaussians per sample

If the lifted phase has Gaussian noise of variance ``r_k``, the likelihood of the
wrapped sample is the wrapped normal, a sum over branches:

```math
p(\theta_k \mid f_k) = \sum_{m \in \mathbb Z}
\mathcal N\!\big(\theta_k + 2\pi m;\; L + f_k,\; r_k\big).
```

Under a state-space prior, the posterior over the state and a branch history
``m_{1:k}`` is then a mixture with one Kalman filter per history, each weighted
by the marginal likelihood of its lifted samples (a Gaussian-sum filter; Sorenson
and Alspach 1971; Alspach and Sorenson 1972). At each sample with data, every
history ``h`` branches into one child per ``m``: the child updates ``h``'s filter
with the lifted value ``\theta_k + 2\pi m`` and multiplies its weight by the
innovation density,

```math
w_{h,m} = w_h\, \mathcal N\!\big(\theta_k + 2\pi m - (\mu^-_{h,k})_1;\; 0,\; S_k\big).
```

The predicted covariance, gain and innovation variance do not depend on the
samples' values, so all histories share them; a history is its state mean, its
log weight and its branches. Summed over every history, the weights give the
marginal likelihood of the wrapped track,
``p(\theta) = \sum_h w_h``, and the most probable history gives the branches the
track is fit on (`_branch_filter`).

The sum over histories grows geometrically, so it is truncated as in
multiple-hypothesis tracking (Reid 1979):

- a history branches only to the values within 5 predictive standard
  deviations ``\sqrt{S_k}`` of its prediction, and always to the nearest;
  a farther branch's density is below ``e^{-12.5}`` of the peak;
- a history whose log weight falls 25 below the best one's is dropped, a
  relative weight below ``10^{-10}``;
- at most 64 histories are carried;
- two histories whose predicted states agree to ``10^{-3}`` of the state's
  standard deviations have the same future, so they are merged: their weights
  are summed, which keeps the marginal likelihood, and the more probable one's
  branches are kept, as in the Viterbi algorithm (Viterbi 1967).

At high signal-to-noise ratio only the nearest branch is within reach of each
prediction and one history is carried; the filter is then the ordinary one on the
lifted track. Histories multiply only where a sample's branch is in doubt.

**A flat start.** A random walk with a flat start has no prediction until its
first ``m`` samples with data fix the state, and every branch of those samples is
equally probable. Each takes the branch nearest the previous one. For samples on
consecutive points of an evenly spaced grid this loses nothing: any other choice
differs by a polynomial of degree below ``m`` whose values at the grid points are
multiples of ``2\pi``, which leaves every sample's phase unchanged and which the
flat start absorbs. A track whose first samples have gaps between them is
restricted to that one choice.

**A level.** With a level beside the prior, the level is integrated out under a
flat prior as in [The level](@ref prior-level) above. Each history carries its own running
``b``, and ``c`` is shared, so a history's prediction of sample ``k`` with the
level integrated out is ``(\mu^-_{h,k})_1 + (b_h/c)\, v^1_k`` with variance
``S_k + (v^1_k)^2/c``, and its weight is
``\log w_h = \ell_h + b_h^2/2c - \tfrac12\log c``. Before any sample has data the
level is unknown, so the first sample takes its nearest branch, which the level
absorbs. Tracks that share a level are then put on one branch, the one bringing
each track's weighted mean nearest the first's.

A track whose level is free, under a flat start or beside a level, is finally
moved by the multiple of ``2\pi`` that keeps it nearest its input, so an
iterative solve does not see it jump by whole turns between iterations.

### The hyperparameters over every branch

A phase track's hyperparameters can be estimated by the type-II MAP above with the likelihood
summed over the branch histories, ``\log \sum_h w_h``, in place of the
likelihood of one lifted track. Where several tracks share a level, the pooled
restricted likelihood of their most probable lifts is used, plus, for each track,
the log of its histories' total weight relative to its most probable one's
(`_branch_pooled_loglik`); without shared levels that is exactly the summed
likelihood. Each evaluation of the objective runs the filter again at the trial
hyperparameters, so the branches and the hyperparameters are estimated together.
The random walk's search starts from the ``m``-th differences reduced modulo
``2\pi``, which their integer coefficients make well defined.
The bandpass solve uses this for the track alone to start its hyperparameters,
then fits them on the chosen branches together with every other track's (see
[The bandpass posterior](@ref bandpass-posterior)).

### The probability of a slip

The histories left at the end measure how well the data determine the branches.
The slip probability of a track is the total weight of the histories whose fitted
track differs from the most probable one's by more than ``\pi/2`` at some sample,
once a common phase is removed (`_slip_probability`). The smoother is linear, so
two histories' fits differ by the fit of the difference of their lifts, which is
what is compared. A value near zero means one branch history dominates; a large
one means the data at that signal-to-noise ratio do not determine the track, and
the fitted track is one of several the posterior allows.

### Limits

The wrapped normal is the likelihood when the phase noise is Gaussian. A phase
derived from a complex measurement has non-Gaussian noise at low
signal-to-noise ratio; treating it as Gaussian with the weight's variance is the
same linearization the solvers already make. The filter chooses the most
probable branch history, which need not be the branches of the single most
probable track, and the truncation drops histories whose total weight is small
by construction.

The same structure, an integer per observation estimated alongside a real-valued
state, is the carrier-phase ambiguity problem of satellite navigation, where the
integers are resolved by integer least squares (Teunissen 1995). There the
integers enter a general least-squares problem; here the Markov prior makes the
search sequential. Kalman filtering has also been used to unwrap interferometric
phase in radar imaging (Loffeld et al. 2008).

**References.**
D. L. Alspach and H. W. Sorenson (1972), Nonlinear Bayesian estimation using
Gaussian sum approximations, *IEEE Transactions on Automatic Control* 17(4),
439–448.
O. Loffeld, H. Nies, S. Knedlik, and Y. Wang (2008), Phase unwrapping for SAR
interferometry—a data fusion approach by Kalman filtering, *IEEE Transactions on
Geoscience and Remote Sensing* 46(1), 47–58.
D. B. Reid (1979), An algorithm for tracking multiple targets, *IEEE Transactions
on Automatic Control* 24(6), 843–854.
H. W. Sorenson and D. L. Alspach (1971), Recursive Bayesian estimation using
Gaussian sums, *Automatica* 7(4), 465–479.
P. J. G. Teunissen (1995), The least-squares ambiguity decorrelation adjustment:
a method for fast GPS integer ambiguity estimation, *Journal of Geodesy* 70(1–2),
65–82.
A. J. Viterbi (1967), Error bounds for convolutional codes and an asymptotically
optimum decoding algorithm, *IEEE Transactions on Information Theory* 13(2),
260–269.

## [The bandpass posterior](@id bandpass-posterior)

[`solve_joint_bandpass!`](@ref) fits the complex visibilities against
``V_{ab}(\nu) \approx g_a(\nu)\, S\, \overline{g_b(\nu)}``, with one complex
gain ``g = e^{\ell + i\phi}`` per (station, feed, frequency segment, time
segment) slot and one frequency-flat source term ``S`` per (scan, station pair,
feed pair). Its objective is the negative log posterior

```math
O(\ell, \phi) = \tfrac12 \chi^2\big(g, \hat S(g)\big) + \sum_{\text{tracks}} E(\ell_{\text{track}}) + \sum_{\text{tracks}} E(\phi_{\text{track}}),
```

with ``\hat S(g)`` the closed-form least-squares source term for the gains and
``E`` each track's prior energy, its levels profiled out.

### The prior as a sparse quadratic

A state-space prior's energy over a track, written over its states rather than
through the filter, is a sum of small quadratic terms: for each step between
samples, ``\tfrac12 (s_k - A_k s_{k-1})^\top Q_k^{-1} (s_k - A_k s_{k-1})``, and
for a proper start ``\tfrac12 (s_1 - \mu_0)^\top P_0^{-1} (s_1 - \mu_0)``, where
the first entry of ``s_k`` is the track's value less its level. Minimized over
the latent entries of the states (the derivatives of a random walk) and the
levels, this is the energy the Kalman filter gives with the values observed
exactly, and a flat start is exact: no start term at all, rather than a large
starting variance (`_track_energy`). The unknowns of one track are its values,
its latent states and its levels, and the terms couple only neighboring
samples, so the energy of every track is one sparse quadratic. This is the
smoothing problem written as least squares, which the RTS smoother solves by
elimination along the chain (Bell 1994).

The source term is frequency-flat, so it absorbs any constant added to a track:
the data cannot measure a track's band constant. The posterior of everything
else is the same as under each prior profiled over that constant, so a proper
prior (an OU prior, or a random walk with an `init`) acts on its track less a
free constant, exactly as beside a level; a flat-start walk is unchanged by a
constant.

### The step

The χ² to second order in each slot's log-gain about the current gains, with
``S`` held, has curvature ``q = w|m|^2`` per correlation between the two slots it
joins (``m = g_a S \bar g_b``) and decouples into a log-amplitude part, which
measures ``\delta\ell_a + \delta\ell_b``, and a phase part, which measures
``\delta\phi_a - \delta\phi_b``. Each part, plus the exact prior quadratic of its
tracks, is one sparse system over the slots, the latent states and the levels,
solved by a sparse Cholesky factorization (`_map_step`). A gauge other than a
pin is imposed exactly by solving each of its constraints for one slot, which
leaves the system positive definite. The unknowns are ordered by frequency for
the factorization: the data couple stations only at one frequency and a prior
couples frequencies only along a track, so the matrix is banded. Where the
linearization is poor, a step can lower the χ² in a direction the full
objective does not support (shrinking a gain whose phase is far off, rather than
turning it), so the step is damped by Levenberg–Marquardt: ``\lambda/2\,
\lVert x' - x \rVert^2`` over the slots' log-gains, a trust region in the units in
which the linearization's error is measured, with ``\lambda`` a multiple ``\mu``
of the largest data curvature and ``\mu`` updated from the ratio of the
objective's actual decrease to the quadratic model's predicted one (Madsen,
Nielsen & Tingleff 2004). A step is accepted when it does not raise the
objective, with ``S`` re-solved.

The iteration starts from each slot's own gain estimate given unit gains
elsewhere and stops when the undamped step would move no gain by more than
`tolerance` of its standard error from the data, or on a plateau: when an
accepted step lowers the negative log posterior by less than
``\mathrm{tolerance}^2/2``, what a step of `tolerance` standard errors would.
``S`` absorbs a band-constant change of a station's gains, so at low SNR the
gains can keep moving along such directions while the posterior barely
changes. Where a slot's phase is held
more than ``\pi/2`` from its own data, by a phase prior or the gauge, the
amplitude that fits it best is zero; without an amplitude prior the posterior
then has no mode, and the solve warns.

### Hyperparameters

A hyperparameter ``\theta`` given as a density is estimated by type-II MAP:
the maximum of the hyperprior density times the marginal likelihood of the
data, the gains integrated out. Every iteration this is done for the quadratic
the step solves, the χ² to second order about the current gains with ``S``
held. That quadratic is Gaussian in its unknowns ``u`` (the slots, latent
states and levels), so for it the Laplace approximation (Tierney & Kadane
1986) is exact:

```math
\log p(\text{data} \mid \theta) = -\min_u f_\theta(u) - \tfrac12 \log\det \nabla^2 f_\theta
  - \tfrac12 \sum_k \log\det(2\pi Q_k) - \tfrac12 \log\det(2\pi P_0) + \text{const},
```

with ``f_\theta`` the step's objective, the sums over the prior terms of the
previous section, and the levels and flat starts integrated under their flat
priors. One sparse factorization gives the minimum and the determinant. For a
single track whose slots are measured independently, this is the restricted
Kalman marginal of [The marginal likelihood](@ref).

All tracks' hyperparameters are fit together, because the data couple the
stations. The gradient in one track's hyperparameters is that of its prior's
log density in expectation over the posterior of its states (Louis 1982):

```math
\partial_\theta \log p(\text{data} \mid \theta) = \mathrm{E}\big[\partial_\theta \log p(u \mid \theta)\big]
= \partial_\theta \sum_k \Big(-\tfrac12 \log\det(2\pi Q_k)
  - \tfrac12 \operatorname{tr}\big(Q_k^{-1} G_k\, \mathrm{E}[v_k v_k^\top]\, G_k^\top\big)\Big),
```

where ``v_k = (s_k, s_{k-1})`` and ``G_k = [I, -A_k]``. The second moments need
only the posterior covariances of unknowns that share a term. Those lie on the
sparsity pattern of the Cholesky factor, whose entries of the inverse the
recursion of Takahashi, Fagan & Chen (1973) gives without forming the rest. The
search is quasi-Newton (BFGS; Nocedal & Wright 2006) with one block per track,
started from the curvature of the expected log density and kept between
iterations, with each step backtracked until the log posterior is no lower. It
stops once the Newton step is within a tolerance of its standard error, its
curvature measured along a step of its own.

That optimum moves with the linearization, so it is sought only as closely as
the last step moved the gains, in their standard errors, and to `tolerance`
once the gains have converged: an inexact inner solve in the sense of
Eisenstat & Walker (1996). At convergence the gains are the MAP under the
hyperparameters, and the hyperparameters are the type-II MAP for the posterior
linearized at those gains. This is the linearize-then-estimate scheme of
penalized quasi-likelihood (Breslow & Clayton 1993), here with a Gaussian
likelihood in the visibilities. ``S`` is held at its estimate, not
integrated: integrating it would couple every pair of slots on a baseline. It
absorbs only band constants, along which the priors are flat.

The solve starts the hyperparameters from type-II MAP on each track alone,
taking each slot's own estimate given every other gain and ``S`` as an
independent measurement: the gain ``\hat g`` that minimizes the χ² of its own
correlations, with weight ``\sum w |S\, g_{\text{other}}|^2 \cdot |\hat g|^2``
in the log-gain. For a phase track, that start is summed over the branch
histories.

### Branches

The phase branches are chosen every iteration from the same slot estimates
under the current hyperparameters (see
[Phase tracks and their 2π branches](@ref phase-branches)). A branch change
moves a phase by a multiple of ``2\pi``, which leaves its gain unchanged.

**References.**
B. M. Bell (1994), The iterated Kalman smoother as a Gauss–Newton method,
*SIAM Journal on Optimization* 4(3), 626–636.
N. E. Breslow and D. G. Clayton (1993), Approximate inference in generalized
linear mixed models, *Journal of the American Statistical Association* 88(421),
9–25.
S. C. Eisenstat and H. F. Walker (1996), Choosing the forcing terms in an
inexact Newton method, *SIAM Journal on Scientific Computing* 17(1), 16–32.
T. A. Louis (1982), Finding the observed information matrix when using the EM
algorithm, *Journal of the Royal Statistical Society B* 44(2), 226–233.
K. Madsen, H. B. Nielsen, and O. Tingleff (2004), *Methods for Non-Linear Least
Squares Problems*, 2nd ed., Technical University of Denmark.
J. Nocedal and S. J. Wright (2006), *Numerical Optimization*, 2nd ed., Springer.
K. Takahashi, J. Fagan, and M.-S. Chen (1973), Formation of a sparse bus
impedance matrix and its application to short circuit study, *Proceedings of
the 8th Power Industry Computer Applications (PICA) Conference*.
L. Tierney and J. B. Kadane (1986), Accurate approximations for posterior
moments and marginal densities, *Journal of the American Statistical
Association* 81(393), 82–86.

## Where the solvers use this

**Bandpass, [`JointSmoother`](@ref).** Each (station, feed, time segment)
has one track per spectral window along frequency, and the solve maximizes the
posterior of every track at once, as described in
[The bandpass posterior](@ref bandpass-posterior) below. Each iteration
puts each phase window on its most probable branches; fits every track's
hyperparameters together to the marginal likelihood of the linearized
posterior; and takes one damped Gauss–Newton step of the posterior. The report's `phase_slip` holds each
window's slip probability from the last iteration.

**Adhoc phase, [`PerTrackAdhocSmoother`](@ref).** After the per-AP solve, each
(station, feed node) track of a scan is fit on its own: hyperparameters per
(station, node, scan) with the level integrated out, then the level plus the
MAP of the zero-mean part. The per-scan mean is then removed from every track,
since a per-station constant trades against the baseline source terms.

**Adhoc phase, [`JointKalmanSmoother`](@ref).** The state stacks every
station's state under its own prior: its phase ``\theta_i`` for an OU prior,
or ``(\theta_i, \theta_i', \dots)`` for a random walk with an `init`. The
stations are independent, so the transition is block diagonal, one block per
station. Each gated (baseline, correlation product) cell at an AP is an
observation

```math
y_{ab} = \theta_a - \theta_b + \varepsilon_{ab}, \qquad
\varepsilon_{ab} \sim \mathcal N\!\big(0, 1/\mathrm{SNR}_{ab}^2\big),
```

with its source term already removed. The same filter and smoother, in matrix
form ([`kalman_mv_filter`](@ref), [`rts_smooth_mv`](@ref)), give the
posterior mean of every station's track from the baseline data directly,
without first solving each AP. Differences leave the sum of all station
phases unconstrained; only the priors hold it near zero, which is why every
station needs a proper prior (an OU prior or a walk with an `init`), and the result is
re-referenced to the anchor station at every AP afterwards. Each observed phase
is moved by a multiple of 2π toward the current model before the filter runs,
and this is repeated a few times as the model improves.

Each station's hyperparameters are resolved by type-II MAP on its track from
the per-AP solve, with the level integrated out, and the track is centered on
its level for the filter. The anchor's per-AP track is zero by construction,
and an unobserved station has none, so both take the median of the resolved
values of the stations whose prior has the same form (OU, or a walk of the same
order), unless their own are fixed.

In both adhoc solvers, the complex-domain refinement (see
[`AdhocOptions`](@ref)) repeats the fit on observations linearized around the
current tracks, re-resolving the hyperparameters each pass.
