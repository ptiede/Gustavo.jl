"""
FRING
Only the best chicken in the world. We sell nothing else and live on pure vibes
"""
module Gustavo

import DimensionalData
using DimensionalData: lookup, modify, groupby, dims, Ti, At, AbstractDimVector
using OrderedCollections: OrderedDict
import XRadio
import Zarr
import DiskArrays

# The scheduler types users select for either fan-out level; re-exported so a
# bare `using Gustavo` can name them in
# `ExecutionConfig(outer_executor = …, inner_executor = …)`.
using OhMyThreads: DynamicScheduler, StaticScheduler, GreedyScheduler, SerialScheduler
using OhMyThreads: Scheduler, tforeach, tmap
using OhMyThreads.Schedulers: chunking_enabled, has_nchunks, nchunks

using LinearAlgebra: BLAS
import StatsAPI
using StatsAPI: fit

using DimensionalData: AbstractDimArray, hasdim, branches, metadata
using Statistics: mean
using AstroLib: ct2lst

include("data/dimensions.jl")
include("data/io.jl")
include("data/apriori.jl")
include("data/utilities.jl")
include("data/msv4_schema.jl")
include("data/measurementset.jl")
include("data/autocorrelations.jl")

include("Calibration.jl")
using .Calibration
# The pipeline layer's step methods extend the model layer's
# element-compilation generic (see pipeline/protocol.jl).
import .Calibration: model_components

include("Fring.jl")
using .Fring

# Top-level modular calibration pipeline (orchestrates all three submodules).
include("pipeline.jl")

export Calibration, Fring
# Axis names for every array Gustavo stores or returns, so scripts can index
# and slice them without reaching into XRadio or `DimensionalData`.
# `Ti` is DimensionalData's own dim, re-exported here for the same reason.
export Polarization, Frequency, AntennaName, BaselineID, Ti, UVW, Feed, Scan, AntennaPair, FeedPair
# UVFITS entry: the reader and writer of a `ProcessingSet`.
export load_uvfits, write_uvfits
export GUSTAVO_VISIBILITY_SCHEMA, feed_pairs
export DynamicScheduler, StaticScheduler, GreedyScheduler, SerialScheduler
export AbstractGauge, PinAntenna, ZeroSumPhase, ByComponent, resolve_gauge
export calibrate, calibrate!
export BaselineFringeFit, default_fringe_terms, Bandpass, default_bandpass_terms, AdhocPhase,
    default_adhoc_terms, AbstractBandpassSmoother, PerTrackSmoother, JointSmoother
# The gain-model vocabulary: everything a step's `model =` argument is written
# in — components, terms, segmentations, feed tyings — under a bare
# `using Gustavo`.
export GainComponent, GainModel, with_station
export AbstractPrior, IIDPrior, RandomWalkPrior, OUPrior
export station_components, component_label
export AbstractGainTerm, ConstantTerm, Delay, Dispersion, Rate, Polynomial,
    PolynomialFreq, PolynomialTime
export AbstractFeedTying, PerFeed, SharedFeeds, SingleFeed
export AbstractTimeSegmentation, GlobalTime, PerScan, PerIntegration, TimeBlocks,
    InstrumentScans
export AbstractFrequencySegmentation, GlobalFrequency, PerSpectralWindow,
    ChannelBlocks, FreqGroups, BandGroups
# Verbs, step protocol, execution config.
export fit
export SolveStep, ExecutionConfig, ProgressLogger, mapsets, write!
export outer_executor, inner_executor
export each_group, model_components, provides, step_gauge
export supports_station_heterogeneity
export scale_weights!, normalize_by_autocorrelations!
export apriori_calibrate!, TsysPlacement, ScanMean, LinearInTime, NearestInTime
export search_scan
# The solution surface: the container, its components, and serialization.
export CalibrationSolution, SolvedComponent
export gains, plot_gain_phases
export save_solution, load_solution

# Unexported because the names are common: a step author extends `solve`.
@static if VERSION >= v"1.11"
    eval(Meta.parse("public solve, SolveContext"))
end
end
