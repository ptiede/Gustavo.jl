# ── Fringe estimator seam ────────────────────────────────────────────────────
#
# The strategy that turns scan data into station fringe parameters (delays,
# rates, phases) is pluggable. Today's implementation — a per-baseline
# delay/rate matched-filter search followed by a closure-screened per-station
# WLS ("stationization") — is ONE estimator; a Schwab–Cotton-style global least
# squares fit of station parameters directly to the visibilities is another,
# and needs no stationization at all. The search/closure machinery therefore
# belongs to the ESTIMATOR that uses it, not to the `FringeFit` step itself.
#
# This file declares the seam: the supertype and the two hooks a `FringeFit`
# step calls. `MatchedFilter`'s methods are defined with the step itself, which
# is where the pipeline's solve context is in scope.

"""
    AbstractFringeEstimator

The strategy a `FringeFit` step uses to estimate station fringe parameters
from scan data. Implementations own their machinery entirely — e.g. the
matched-filter estimator carries a search configuration and a `Stationization`;
a global least-squares estimator would carry neither.

# Implementing an estimator

Subtype it and define two methods, which `FringeFit` calls once per scan group
and once per pass:

    Gustavo.Fringe.estimate_scan!(est::MyEstimator, ctx, step, view) -> NamedTuple
    Gustavo.Fringe.finish_estimate!(est::MyEstimator, ctx, step) -> NamedTuple

See [`estimate_scan!`](@ref) and [`finish_estimate!`](@ref) for what each
receives and must return. Both have fallbacks that error, so an estimator that
implements neither fails with a message naming what is missing rather than
being silently skipped.

Then declare what it can fit, which the step checks when it compiles the
model — before any data is read:

    Gustavo.Fringe.can_fit(est::MyEstimator, tc) -> Bool
    Gustavo.Fringe.validate_model(est::MyEstimator, comps)   # optional

[`can_fit`](@ref) defaults to `false`, so an estimator that declares nothing
is rejected rather than quietly leaving θ columns unwritten;
[`validate_model`](@ref) defaults to a no-op, since requiring nothing is
legitimate. See both for the two directions of the check.

The step, not the estimator, owns the θ slots `FringeModel` declared and the
dTEC/SBD refine service it publishes for later stages — an estimator only has
to fill θ and report.
"""
abstract type AbstractFringeEstimator end

"""
    estimate_scan!(est::AbstractFringeEstimator, ctx, step, view) -> NamedTuple

One scan group's contribution to the fringe estimate. `ctx` is the pipeline's
solve context (`ctx.θ`, `ctx.ev`, `ctx.geom`, `ctx.stream`, `ctx.scratch`),
`step` the [`FringeFit`](@ref) being run — read its `model` for WHAT is solved —
and `view` the materialized `ScanDataView`.

Runs concurrently across scan groups, so it may write only θ columns private to
this scan; anything global belongs in [`finish_estimate!`](@ref). The returned
NamedTuple is collected in group order and handed back there.

Include a `max_snr::Real` field: scan selections that rank groups by strength
([`BrightestCalibrator`](@ref)) read it, and an estimator with no notion of SNR
should return `NaN` rather than omit it.
"""
function estimate_scan! end

estimate_scan!(est::AbstractFringeEstimator, ctx, step, view) = error(
    "$(typeof(est)) does not implement the fringe estimator interface: define " *
        "Gustavo.Fringe.estimate_scan!(::$(typeof(est)), ctx, step, view) " *
        "returning this scan group's contribution (see `AbstractFringeEstimator`)."
)

"""
    finish_estimate!(est::AbstractFringeEstimator, ctx, step) -> NamedTuple

The global half of the estimate, run once per pass after every scan group has
been through [`estimate_scan!`](@ref). The per-group returns are in
`ctx.scratch[:pass_results]`, in group order, each wrapped as
`(; index, r, decode, work, reduce)` — `r` is what `estimate_scan!` returned.
This is where θ's cross-scan columns are solved.

The returned NamedTuple becomes the stage's recorded diagnostics. Include
`repeat_pass = true` to have the runner stream the whole pass again — how a
residual-refinement round is requested. The step publishes its refine service
once the pass is NOT repeated, so an estimator that iterates must report every
intermediate round as `repeat_pass = true`.
"""
function finish_estimate! end

finish_estimate!(est::AbstractFringeEstimator, ctx, step) = error(
    "$(typeof(est)) does not implement the fringe estimator interface: define " *
        "Gustavo.Fringe.finish_estimate!(::$(typeof(est)), ctx, step) " *
        "solving the cross-scan parameters (see `AbstractFringeEstimator`)."
)

"""
    can_fit(est::AbstractFringeEstimator, tc::Calibration.TiedComponent) -> Bool

Whether `est` fits the θ block of the compiled component `tc`. `FringeFit`
calls this once per component ITS OWN model contributed, at model-compile
time, and throws an `ArgumentError` naming the estimator and the component on
the first `false` — a term nothing writes is a silent no-fit, not a smaller
solve.

**The default is `false`**, and the step drives the loop, so an estimator whose
author never considered capability fails loudly on the first model term rather
than returning a solution with zeros in it. Declaring capability is therefore
part of implementing the interface, not an optional refinement.

Components contributed by OTHER steps — the per-integration adhoc phase, the
per-channel bandpass — are not asked about: the step that contributes a
component vouches for it.
"""
can_fit(::AbstractFringeEstimator, tc) = false

"""
    validate_model(est::AbstractFringeEstimator, comps) -> nothing

Check that the compiled components `comps` (again, only the `FringeFit`'s own
contributions) contain everything `est` REQUIRES, throwing an `ArgumentError`
naming the estimator and the missing signature. The mirror of
[`can_fit`](@ref): that one rejects terms the estimator cannot fit, this one
rejects a model missing terms the estimator assumes exist.

The default is a no-op — requiring nothing is legitimate. Define a method when
an absent term would make the estimator discard its own results: an estimator
that searches for delays and finds no delay component to write them into
produces a solution that looks fitted and is not.

Express a requirement as the routing signature the estimator's machinery
actually looks up, not as a term type. `MatchedFilter` requires a per-scan
feed-common delay; a `Delay` segmented by `GlobalTime` satisfies a term-level
check and still leaves the router with nothing to return.
"""
validate_model(::AbstractFringeEstimator, comps) = nothing

"""
    estimator_info(est::AbstractFringeEstimator) -> NamedTuple

The estimator's own provenance, merged into the fitted solution's `info` so a
solution records HOW it was estimated. `MatchedFilter` reports its
[`FringeSearch`](@ref) configuration as `search`.

Optional: the default is empty, and a solution from an estimator that defines no
method simply carries no estimator-specific fields. Field names must not collide
with the run-level ones the pipeline records (`nant`, `nscan`, `ant_names`, the
timings, …); a collision silently wins over the pipeline's value.
"""
estimator_info(::AbstractFringeEstimator) = NamedTuple()
