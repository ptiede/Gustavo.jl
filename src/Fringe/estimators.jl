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
