"""
    Calibration

Shared calibration infrastructure for Gustavo: feed/correlation-product
conventions, weighted least-squares solvers, phase-track utilities, and the
unified station gain-model framework used by `Gustavo.Fring` and the
calibration pipeline.
"""
module Calibration

using OhMyThreads: tforeach, DynamicScheduler, SerialScheduler
import XRadio
import DimensionalData
import DensityInterface
import Distributions
using Distributions: Distribution, Normal, AbstractMvNormal, mean, cov, var
using ..UVData
using ..UVData: feed_pairs
using LinearAlgebra
using LinearSolve
using Statistics: median
using Dates: Period, Nanosecond, DateTime, datetime2unix

include("Calibration/gauge.jl")
include("Calibration/lsq.jl")
include("Calibration/segmentation.jl")
include("Calibration/terms.jl")
include("Calibration/priors.jl")
include("Calibration/models.jl")
include("Calibration/parameters.jl")
include("Calibration/evaluate.jl")
include("Calibration/solutions.jl")
include("Calibration/solution_store.jl")

"""
    plot_gain_phases(g)
    plot_gain_phases(parent, g)

Gain phase of `g`, a [`gains`](@ref) array reduced to `AntennaName`, `Feed` and
one more dimension: one panel per antenna (rows) and feed (columns), with the
remaining dimension on the x axis. Select that dimension by indexing, e.g.
`plot_gain_phases(gains(sol; Ti = 1))` for phase against frequency or
`plot_gain_phases(gains(sol[:bandpass]; Ti = 1))` for the bandpass alone.
Provided by `GustavoMakieExt`.
"""
function plot_gain_phases end

# Feed / correlation-product conventions
export feed_pairs

# Gauge conventions for station-based solves
export AbstractGauge, PinAntenna, ZeroSumPhase, ByComponent
export GaugeFreedom, GaugeFreedoms, gauge_constraints, gauge_constraint, gauge_anchor
export resolve_gauge, remap_gauge, gauge_station_order, check_gauge_components

# ── Unified gain-model framework ─────────────────────────────────────────────
# Segmentation vocabulary
export AbstractTimeSegmentation, AbstractFrequencySegmentation
export GlobalTime, PerScan, PerIntegration, TimeBlocks, InstrumentScans
export GlobalFrequency, PerSpectralWindow, ChannelBlocks, FreqGroups, BandGroups
export DataGeometry, time_segment_ids, freq_segment_ids, segment_groups, common_refinement
export resolve, segment_ranges, fringe_freq_groups

# Terms
export AbstractGainTerm
export ConstantTerm, Delay, Dispersion, Rate, Polynomial, PolynomialFreq, PolynomialTime
# The term-authoring interface: the hooks a new `AbstractGainTerm` implements.
export term_axes, param_shapes, term_eval
export freq_coordinate, time_coordinate, freq_coord_state, time_coord_state

# Models and tying
export GainComponent, GainModel, with_station
export AbstractPrior, IIDPrior, RandomWalkPrior, OUPrior, resolve_prior, is_fixed_hyper
export AbstractFeedTying, PerFeed, SharedFeeds, SingleFeed
export phase_components, logamp_components, model_components
export station_components
export validate_gain_model, component_label


# Parameter layout and pure evaluation
export ParameterLayout, plan_parameters, station_blocks
export evaluate_gains

# Calibration solution container, apply, serialization
export CalibrationSolution, SolvedComponent, GeometryWindow
export gains, plot_gain_phases
export save_solution, load_solution, storage_constructor, storage_arguments

end
