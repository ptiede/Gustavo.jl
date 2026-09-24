"""
    Calibration

Shared calibration infrastructure for Gustavo: feed/correlation-product
conventions, weighted least-squares solvers, phase-track utilities, and the
unified station gain-model framework used by `Gustavo.Fring` and the
calibration pipeline.
"""
module Calibration

using OhMyThreads: tforeach, DynamicScheduler, SerialScheduler
using ..UVData
using ..UVData: feed_pairs, phase_relative_to_ref
using ..UVData: PolTypes, UVSet, channel_freqs, rebuild_visibilities, materialize_leaf,
    is_lazy, pol_products, baselines
# Segmentation materialization is a METHOD of the data layer's `materialize`
# generic — one package-wide verb for resolving a deferred form (a lazy leaf, a
# data-dependent segmentation) into its concrete one. A bare `using` definition
# would mint a second function of the same name and leave the two ambiguous
# wherever both modules are in scope.
import ..UVData: materialize
using LinearAlgebra
using LinearSolve
using Statistics: median
using Dates: Period, Nanosecond, DateTime, datetime2unix

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
export feed_pairs

# Gauge conventions for station-based solves
export AbstractGauge, PinAntenna, ZeroSumPhase
export gauge_anchor, gauge_row!, resolve_gauge, remap_gauge
export gauge_station_order

# ── Unified gain-model framework ─────────────────────────────────────────────
# Segmentation vocabulary
export AbstractTimeSegmentation, AbstractFrequencySegmentation
export GlobalTime, PerScan, PerIntegration, TimeBlocks, InstrumentScans
export GlobalFrequency, PerSpectralWindow, ChannelBlocks, FreqGroups, BandGroups
export DataGeometry, time_segment_ids, freq_segment_ids, segment_groups, common_refinement
export materialize, segment_ranges, fringe_freq_groups

# Terms
export AbstractGainTerm
export ConstantTerm, Delay, Dispersion, Rate, Polynomial, PolynomialFreq, PolynomialTime
# The term-authoring interface: the hooks a new `AbstractGainTerm` implements.
export term_axes, param_shapes, term_eval
export freq_coordinate, time_coordinate, freq_coord_state, time_coord_state

# Models and tying
export GainComponent, GainModel, with_station
export AbstractFeedTying, PerFeed, SharedFeeds, SingleFeed
export phase_components, logamp_components, model_components
export station_components
export validate_gain_model, component_label

# Propagation
export DispersionModel

# Parameter layout and pure evaluation
export ParameterLayout, plan_parameters, station_blocks
export evaluate_gains

# Calibration solution container, apply, serialization
export CalibrationSolution, StepSolution, build_geometry, GeometryWindow, leaf_window
export save_solution, load_solution

# Per-stage provenance and snapshots (composable pipeline)
export stage_info
export component_names, gains, parameters

end
