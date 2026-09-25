# ── Fringe estimator seam ────────────────────────────────────────────────────
#
# The supertype and the two hooks a `BaselineFringeFit` step calls to turn scan data
# into station fringe parameters. `MatchedFilter`'s methods are defined with the
# step itself, which is where the pipeline's solve context is in scope.

"""
    AbstractFringeEstimator

The strategy a `BaselineFringeFit` step uses to estimate station fringe parameters
from scan data. Implementations own their machinery: the matched-filter
estimator carries a search configuration and a `Stationization`; a global
least-squares estimator would carry neither.

# Implementing an estimator

Subtype it and define two methods, called once per scan group and once per
pass:

    Gustavo.Fring.estimate_scan!(est::MyEstimator, ctx, step, stack, win) -> NamedTuple
    Gustavo.Fring.finish_estimate!(est::MyEstimator, ctx, step) -> NamedTuple

(see [`estimate_scan!`](@ref) and [`finish_estimate!`](@ref); both fallbacks
error, naming what is missing). Then declare what it can fit, checked when
the step compiles the model, before any data is read:

    Gustavo.Fring.can_fit(est::MyEstimator, tc, geom) -> Bool
    Gustavo.Fring.validate_model(est::MyEstimator, model)   # optional

[`can_fit`](@ref) defaults to `false`, so an estimator that declares nothing
is rejected rather than leaving θ columns unwritten; [`validate_model`](@ref)
defaults to a no-op.

An estimator whose solve completes per scan under some configurations may
also declare [`scan_local_solve`](@ref), which lets a `BaselineFringeFit` carrying
it share one streaming pass with adjacent scan-local steps; the default
(`false`) is never fused.

The step, not the estimator, owns the θ slots its model declared; an
estimator only fills θ and reports.
"""
abstract type AbstractFringeEstimator end

"""
    estimate_scan!(est::AbstractFringeEstimator, ctx, step, stack, win) -> NamedTuple

One scan group's contribution to the fringe estimate. `ctx` is the pipeline's
solve context (`ctx.θ`, `ctx.layout`, `ctx.geom`, `ctx.stream`, `ctx.scratch`),
`step` the [`BaselineFringeFit`](@ref Gustavo.BaselineFringeFit) being run — read its `model` for what is solved —
`stack` the materialized scan group's `DimStack`, and `win` its
[`GeometryWindow`](@ref) into the solve's index space.

Runs concurrently across scan groups, so it may write only θ columns private to
this scan; anything global belongs in [`finish_estimate!`](@ref). A scan-local
configuration ([`scan_local_solve`](@ref)) writes all of this scan's columns
here. The returned NamedTuple is collected in group order and handed back there.

Include a `max_snr::Real` field: it becomes the step's per-scan `scan_snr`
diagnostic, and an estimator with no notion of SNR should return `NaN` rather than omit it.
"""
function estimate_scan! end

estimate_scan!(est::AbstractFringeEstimator, ctx, step, stack, win) = error(
    "$(typeof(est)) does not implement the fringe estimator interface: define " *
        "Gustavo.Fring.estimate_scan!(::$(typeof(est)), ctx, step, stack, win) " *
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
        "Gustavo.Fring.finish_estimate!(::$(typeof(est)), ctx, step) " *
        "solving the cross-scan parameters (see `AbstractFringeEstimator`)."
)

"""
    scan_local_solve(est::AbstractFringeEstimator, model) -> Bool

Whether `est`, fitting `model` (the `BaselineFringeFit`'s own model), finalizes
each scan from that scan's data alone: every θ column of a scan is written by
the time its [`estimate_scan!`](@ref) returns, and [`finish_estimate!`](@ref)
neither writes θ nor requests `repeat_pass`. `BaselineFringeFit` declares itself
scan-local (`fusable_grouping` = `:scan`) exactly when this is `true`, letting
it share one streaming pass with adjacent scan-local steps.

**The default is `false`** — a pass-global estimator is always correct, just
never fused. Declare `true` only for configurations whose solve genuinely
completes per scan. The answer is read before any data is, so it must depend
only on the estimator's and model's declared options, never on the geometry.
"""
scan_local_solve(::AbstractFringeEstimator, model) = false

"""
    can_fit(est, tc::Calibration.GainComponent, geom::Calibration.DataGeometry) -> Bool

Whether `est` fits the θ block of the compiled component `tc` on the geometry
`geom`. The step calls this once per component its own model contributed, at
model-compile time, and throws an `ArgumentError` naming the estimator and the
component on the first `false` — a term nothing writes is a silent no-fit, not
a smaller solve. The rejection is generic by design; state which models the
family fits in the estimator's own docstring and point the message there.

`geom` is what makes a capability that depends on the data's own sampling
answerable: whether a time segmentation splits a scan, for instance, is a
property of the segmentation and the scan lengths together, not of the term
alone. Answer from the term wherever that suffices.

**The default is `false`**, and the step drives the loop, so an estimator whose
author never considered capability fails loudly on the first model term rather
than returning a solution with zeros in it. Declaring capability is therefore
part of implementing the interface, not an optional refinement.

Shared with the bandpass and adhoc smoothers, which answer for their own steps'
models. Components contributed by other steps are not asked about: the step
that contributes a component vouches for it.
"""
can_fit(::AbstractFringeEstimator, tc, geom) = false

"""
    validate_model(est::AbstractFringeEstimator, model) -> nothing

Check that the `(; phase, logamp)` component tree `model` (again, only the
`BaselineFringeFit`'s own; once per distinct station tree) contains everything `est`
REQUIRES, throwing an `ArgumentError` naming the estimator and the missing
signature. The mirror of
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
validate_model(::AbstractFringeEstimator, model) = nothing

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
