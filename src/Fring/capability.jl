# ── Solver capability ────────────────────────────────────────────────────────
#
# The compile-time check a step runs on its model before any data is read:
# every component must be one its solver fits (`can_fit`), and the model must
# hold what the solver requires (`validate_model`). The solver is the step
# itself or the smoother it delegates to.

"""
    can_fit(solver, tc::Calibration.GainComponent, geom::Calibration.DataGeometry) -> Bool

Whether `solver` (a solve step, or the smoother a step delegates to) fits the
θ block of the compiled component `tc` on the geometry `geom`. The step calls
this once per component of its own model, when the model compiles, and throws
an `ArgumentError` naming the solver and the component on the first `false`:
a term nothing writes is a silent no-fit, not a smaller solve. State which
models the solver fits in its own docstring.

`geom` makes a capability that depends on the data's sampling answerable:
whether a time segmentation splits a scan is a property of the segmentation
and the scan lengths together. Answer from the term wherever that suffices.

The default is `false`, so a solver that declares nothing fails on its first
model term rather than returning a solution with zeros in it.
"""
can_fit(solver, tc, geom) = false

"""
    validate_model(solver, model) -> nothing

Check that the `(; phase, logamp)` component tree `model` (once per distinct
station tree) contains everything `solver` requires, throwing an
`ArgumentError` naming the solver and what is missing. The mirror of
[`can_fit`](@ref): that one rejects terms the solver cannot fit, this one
rejects a model missing terms the solver assumes exist.

The default is a no-op. Define a method when an absent term would make the
solver discard its own results: a fringe search with no delay component to
write into produces a solution that looks fitted and is not. Express a
requirement as the routing signature the solver looks up, not as a term type.
"""
validate_model(solver, model) = nothing
