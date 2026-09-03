"""
    Calibration

Shared calibration infrastructure for Gustavo: feed/correlation-product
conventions, weighted least-squares solvers, phase-track utilities, and the
unified station gain-model framework used by `Gustavo.Fringe` and the
calibration pipeline.
"""
module Calibration

using OhMyThreads: tforeach, DynamicScheduler, SerialScheduler
using ..UVData
using ..UVData: correlation_feed_pair, is_parallel_hand,
    parallel_hand_indices, cross_hand_indices, phase_relative_to_ref
using ..UVData: PolTypes, UVSet, channel_freqs, rebuild_visibilities, materialize_leaf,
    is_lazy, pol_products, baselines
using LinearAlgebra
using LinearSolve
using ComponentArrays: ComponentVector, ComponentArray, getaxes

include("Calibration/gauge.jl")
include("Calibration/lsq.jl")
include("Calibration/segmentation.jl")
include("Calibration/terms.jl")
include("Calibration/models.jl")
include("Calibration/parameters.jl")
include("Calibration/propagation.jl")
include("Calibration/evaluate.jl")
include("Calibration/solutions.jl")

# Feed / correlation-product conventions
export correlation_feed_pair, is_parallel_hand
export parallel_hand_indices, cross_hand_indices

# Gauge conventions for station-based solves
export AbstractGauge, PinAntenna, ZeroSumPhase
export gauge_anchor, gauge_row!, regauge!, resolve_gauge, gauge_primary, remap_gauge
export gauge_station_order

# Weighted least squares (weights are always inverse variances)
export design_matrices
export weighted_least_squares, weighted_regularized_least_squares,
    weighted_constrained_least_squares
export WLSEstimator
export weighted_phase_mean, weighted_complex_correction
export connected_components, savitzky_golay_smooth

# Phase-track utilities
export unwrap_phase_track, phase_unwrap_ambiguity, phase_relative_to_ref

# ── Unified gain-model framework ─────────────────────────────────────────────
# Segmentation vocabulary
export AbstractTimeSegmentation, AbstractFrequencySegmentation
export GlobalTime, PerScan, PerIntegration, TimeBlocks, InstrumentScans
export GlobalFrequency, PerSpectralWindow, ChannelBlocks, FreqGroups
export DataGeometry, time_segment_ids, freq_segment_ids, segment_groups

# Terms
export AbstractGainTerm
export ConstantTerm, Delay, Dispersion, Rate, Polynomial, PolynomialFreq, PolynomialTime
# The term-authoring interface: the hooks a new `AbstractGainTerm` implements.
export term_axes, param_shapes, term_eval, term_label
export freq_coordinate, time_coordinate, freq_coord_state, time_coord_state

# Models and tying
export GainComponent, StationGainModel
export AbstractFeedTying, PerFeed, SharedFeeds, ReferenceRelative, SingleFeed
export phase_components, logamp_components, model_components
export phase_is_per_scan, amplitude_is_per_scan, component_is_per_scan
export validate_station_gain_model, station_model_summary, component_label

# Propagation
export DispersionModel

# Parameter layout and pure evaluation
export ParameterLayout, plan_parameters
export GainEvaluator, evaluate_gains, predict_visibilities, nparameters

# Calibration solution container, apply, serialization
export CalibrationSolution, StepSolution, build_geometry, GeometryWindow, leaf_window
export save_solution, load_solution
export save_solution_hdf5, load_solution_hdf5

# Per-stage provenance and snapshots (composable pipeline)
export stage_names, stage_info
export component_ranges, component_gains, component_names, component_dimarray, gains
# Fit-once / apply-later extraction (a single step, e.g. a portable bandpass)
export step_solution

end
