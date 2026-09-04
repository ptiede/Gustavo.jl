# ── Fringe estimator seam ────────────────────────────────────────────────────
#
# The strategy that turns scan data into station fringe parameters (delays,
# rates, phases) is pluggable. Today's implementation — a per-baseline
# delay/rate matched-filter search followed by a closure-screened per-station
# WLS ("stationization") — is one estimator; a Schwab–Cotton-style global least
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
from scan data. Implementations own their machinery: the matched-filter
estimator carries a search configuration and a `Stationization`; a global
least-squares estimator would carry neither.

# Implementing an estimator

Subtype it and define two methods, called once per scan group and once per
pass:

    Gustavo.Fringe.estimate_scan!(est::MyEstimator, ctx, step, stack, win) -> NamedTuple
    Gustavo.Fringe.finish_estimate!(est::MyEstimator, ctx, step) -> NamedTuple

(see [`estimate_scan!`](@ref) and [`finish_estimate!`](@ref); both fallbacks
error, naming what is missing). Then declare what it can fit, checked when
the step compiles the model, before any data is read:

    Gustavo.Fringe.can_fit(est::MyEstimator, tc) -> Bool
    Gustavo.Fringe.validate_model(est::MyEstimator, comps)   # optional

[`can_fit`](@ref) defaults to `false`, so an estimator that declares nothing
is rejected rather than leaving θ columns unwritten; [`validate_model`](@ref)
defaults to a no-op.

An estimator whose solve completes per scan under some configurations may
also declare [`scan_local_solve`](@ref), which lets a `FringeFit` carrying
it share one streaming pass with adjacent scan-local steps; the default
(`false`) is never fused.

The step, not the estimator, owns the θ slots `FringeModel` declared; an
estimator only fills θ and reports.
"""
abstract type AbstractFringeEstimator end

"""
    estimate_scan!(est::AbstractFringeEstimator, ctx, step, stack, win) -> NamedTuple

One scan group's contribution to the fringe estimate. `ctx` is the pipeline's
solve context (`ctx.θ`, `ctx.ev`, `ctx.geom`, `ctx.stream`, `ctx.scratch`),
`step` the [`FringeFit`](@ref Gustavo.FringeFit) being run — read its `model` for what is solved —
`stack` the materialized scan group's `DimStack`, and `win` its
[`GeometryWindow`](@ref) into the solve's index space.

Runs concurrently across scan groups, so it may write only θ columns private to
this scan; anything global belongs in [`finish_estimate!`](@ref). A scan-local
configuration ([`scan_local_solve`](@ref)) writes all of this scan's columns
here. The returned NamedTuple is collected in group order and handed back there.

Include a `max_snr::Real` field: a scan selection may rank or filter groups by
strength (e.g. a [`ScanWhere`](@ref) predicate reading `s.snr`), and an
estimator with no notion of SNR should return `NaN` rather than omit it.
"""
function estimate_scan! end

estimate_scan!(est::AbstractFringeEstimator, ctx, step, stack, win) = error(
    "$(typeof(est)) does not implement the fringe estimator interface: define " *
        "Gustavo.Fringe.estimate_scan!(::$(typeof(est)), ctx, step, stack, win) " *
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
once the pass is not repeated, so an estimator that iterates must report every
intermediate round as `repeat_pass = true`.
"""
function finish_estimate! end

finish_estimate!(est::AbstractFringeEstimator, ctx, step) = error(
    "$(typeof(est)) does not implement the fringe estimator interface: define " *
        "Gustavo.Fringe.finish_estimate!(::$(typeof(est)), ctx, step) " *
        "solving the cross-scan parameters (see `AbstractFringeEstimator`)."
)

"""
    scan_local_solve(est::AbstractFringeEstimator, model) -> Bool

Whether `est`, fitting `model` (the `FringeFit`'s own `FringeModel`), finalizes
each scan from that scan's data alone: every θ column of a scan is written by
the time its [`estimate_scan!`](@ref) returns, and [`finish_estimate!`](@ref)
neither writes θ nor requests `repeat_pass`. `FringeFit` declares itself
scan-local (`fusable_grouping` = `:scan`) exactly when this is `true`, letting
it share one streaming pass with adjacent scan-local steps.

**The default is `false`** — a pass-global estimator is always correct, just
never fused. Declare `true` only for configurations whose solve genuinely
completes per scan. The answer is read before any data is, so it must depend
only on the estimator's and model's declared options, never on the geometry.
"""
scan_local_solve(::AbstractFringeEstimator, model) = false

"""
    can_fit(est::AbstractFringeEstimator, tc::Calibration.GainComponent) -> Bool

Whether `est` fits the θ block of the compiled component `tc`. `FringeFit`
calls this once per component its own model contributed, at model-compile
time, and throws an `ArgumentError` naming the estimator and the component on
the first `false` — a term nothing writes is a silent no-fit, not a smaller
solve.

**The default is `false`**, and the step drives the loop, so an estimator whose
author never considered capability fails loudly on the first model term rather
than returning a solution with zeros in it. Declaring capability is therefore
part of implementing the interface, not an optional refinement.

Components contributed by other steps — the per-integration adhoc phase, the
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
solution records how it was estimated. `MatchedFilter` reports its
[`FringeSearch`](@ref) configuration as `search`.

Optional: the default is empty, and a solution from an estimator that defines no
method simply carries no estimator-specific fields. Field names must not collide
with the run-level ones the pipeline records (`nant`, `nscan`, `ant_names`, the
timings, …); a collision silently wins over the pipeline's value.
"""
estimator_info(::AbstractFringeEstimator) = NamedTuple()
