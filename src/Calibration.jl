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
using ..UVData: correlation_feed_pair, is_parallel_hand, same_feed_label,
    parallel_hand_indices, cross_hand_indices, build_parallel_hand_mask,
    phase_relative_to_ref
using ..UVData: PolTypes, UVSet, channel_freqs, rebuild_visibilities, materialize_leaf,
    is_lazy, pol_products, baselines
using LinearAlgebra
using LinearSolve

include("Calibration/lsq.jl")
include("Calibration/segmentation.jl")
include("Calibration/terms.jl")
include("Calibration/models.jl")
include("Calibration/parameters.jl")
include("Calibration/propagation.jl")
include("Calibration/evaluate.jl")
include("Calibration/solutions.jl")

# Feed / correlation-product conventions
export correlation_feed_pair, is_parallel_hand, same_feed_label
export parallel_hand_indices, cross_hand_indices, build_parallel_hand_mask

# Weighted least squares (weights are always inverse variances)
export design_matrices
export weighted_least_squares, weighted_regularized_least_squares,
    weighted_constrained_least_squares
export weighted_phase_mean, weighted_complex_correction
export connected_components, savitzky_golay_smooth

# Phase-track utilities
export unwrap_phase_track, phase_relative_to_ref

# ── Unified gain-model framework ─────────────────────────────────────────────
# Segmentation vocabulary
export AbstractTimeSegmentation, AbstractFrequencySegmentation
export GlobalTime, PerScan, PerIntegration, TimeBlocks, InstrumentScans
export GlobalFrequency, PerSpectralWindow, ChannelBlocks, FrequencyBands
export DataGeometry, time_segment_ids, freq_segment_ids, segment_groups

# Terms
export AbstractGainTerm
export ConstantTerm, Delay, Dispersion, Rate, Polynomial, PolynomialFreq, PolynomialTime
export nparams_per_block

# Models and tying
export GainComponent, TiedComponent, StationGainModel
export AbstractFeedTying, PerFeed, SharedFeeds, ReferenceRelative, FeedComponent
export phase_components, logamp_components, model_components
export phase_is_per_scan, amplitude_is_per_scan, component_is_per_scan
export validate_station_gain_model, station_model_summary, component_label

# Propagation
export DispersionModel

# Parameter layout and pure evaluation
export ParameterLayout, plan_parameters
export GainEvaluator, evaluate_gains, predict_visibilities, nparameters

# Calibration solution container, apply, serialization
export CalibrationSolution, build_geometry, GeometryWindow, leaf_window, save_solution, load_solution
export save_solution_hdf5, load_solution_hdf5

# Per-stage provenance and snapshots (composable pipeline)
export StageRecord, StageView, stage_names, stage_solution, stage_info
export component_ranges, component_gains, gains
# Fit-once / apply-later extraction (portable time-constant bandpass)
export bandpass_solution

end
