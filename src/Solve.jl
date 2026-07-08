# ── Gustavo.Solve — global forward-modeling fringe fitter (Stage 2) ───────────
#
# A single global MAP forward-model fit that replaces the old two-stage
# FFT-search → closure-stationize → per-stage-WLS pipeline. The objective
#
#     f(θ) = Σ_leaf loglik_leaf(θ) + logprior(θ)
#
# is minimized by a gradient optimizer (Optimization.jl + Enzyme, behind
# `GustavoSolveExt`), distributed over `UVSet` partitions on the `Gustavo.Graph`
# (Dagger) substrate. Parameters are per-site 2D `(time × frequency)` arrays
# (ComponentArrays); phase/amplitude offsets are Cartesian/complex so the
# landscape is wrap-free (no phase unwrapping). The existing `Fringe` FFT search +
# stationization is kept as the delay/rate warm-start initializer.
#
# Built incrementally (see the plan). Milestone 1: the per-leaf coherency-matrix
# WLS objective on the existing flat-θ forward map.

module Solve

using ..UVData
using ..UVData: UVSet
using ..Calibration
using ..Calibration:
    GainEvaluator, predict_visibilities,
    DataGeometry, build_geometry, leaf_window, correlation_feed_pair
# Extended with new methods for the per-site `GainPlan` (Milestone 2), so import.
import ..Calibration: evaluate_gains, nparameters
# Non-exported forward-map internals reused by the per-site plan (Milestone 2).
using ..Calibration:
    term, time_segmentation, freq_segmentation, coord_kind, COORD_FREQ, COORD_TIME,
    freq_coordinate, time_coordinate, _term_contribution, nfeed_blocks
using ..Graph
using DimensionalData
using ComponentArrays
using ComponentArrays: getdata, getaxes
import LogDensityProblems

include("Solve/sitemodel.jl")
include("Solve/plan.jl")
include("Solve/leaf_objective.jl")
include("Solve/logdensity.jl")

# Milestone 1 — per-leaf coherency WLS objective.
export build_leaf_ctx, point_source_coherency, leaf_loglik_gains, leaf_loglik
# Milestone 2 — per-array model, plan, per-site parameter container, forward map.
export ArrayGainModel, site_model
export GainPlan, GroupPlan, SiteComponent, plan_gains
export zero_params, flatten, unflatten, group_arrays, evaluate_gains
# Milestone 3 — per-leaf reverse-mode value + structured gradient (Enzyme ext).
export leaf_value_and_grad
# Milestone 4 — distributed objective+gradient over the Graph substrate + posterior.
export fringe_objective, fringe_objective_and_grad, FringePosterior

end # module Solve
