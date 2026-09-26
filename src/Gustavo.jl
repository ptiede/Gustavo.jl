"""
FRING
Only the best chicken in the world. We sell nothing else and live on pure vibes
"""
module Gustavo

import DimensionalData
using DimensionalData: lookup, modify, groupby, dims, Ti, AbstractDimVector
using OrderedCollections: OrderedDict
import XRadio

# The scheduler types users select for either fan-out level; re-exported so a
# bare `using Gustavo` can name them in
# `ExecutionConfig(outer_executor = …, inner_executor = …)`.
using OhMyThreads: DynamicScheduler, StaticScheduler, GreedyScheduler, SerialScheduler
using OhMyThreads: Scheduler, tforeach, tmap
using OhMyThreads.Schedulers: chunking_enabled, has_nchunks, nchunks

using LinearAlgebra: BLAS
import StatsAPI
using StatsAPI: fit

include("UVData/UVData.jl")
using .UVData

include("Calibration.jl")
using .Calibration
# The pipeline layer's step methods extend the model layer's
# element-compilation generic (see pipeline/protocol.jl).
import .Calibration: model_components

include("Fring.jl")
using .Fring

# Top-level modular calibration pipeline (orchestrates all three submodules).
include("pipeline.jl")

export UVData, Calibration, Fring
# Axis names for every array Gustavo stores or returns, so scripts can index
# and slice leaves without reaching into `UVData` or `DimensionalData`.
# `Ti` is DimensionalData's own dim, re-exported here for the same reason.
export Polarization, Frequency, Ant, BaselineID, Ti, UVW, Feed, Scan
# Data entry and exit: the set type plus the reader/writer pair for each
# supported format, so a bare `using Gustavo` spans load → fit → calibrate → write.
export UVSet, load_uvfits, load_fitsidi, write_uvfits, write_fitsidi
export DynamicScheduler, StaticScheduler, GreedyScheduler, SerialScheduler
export AbstractGauge, PinAntenna, ZeroSumPhase, resolve_gauge
export calibrate
export BaselineFringeFit, default_fringe_terms, Bandpass, default_bandpass_terms, AdhocPhase,
    default_adhoc_terms, AbstractBandpassSmoother, PerTrackSmoother, JointSmoother
# The gain-model vocabulary: everything a step's `model =` argument is written
# in — components, terms, segmentations, feed tyings — under a bare
# `using Gustavo`.
export GainComponent, GainModel, with_station
export station_components, component_label
export AbstractGainTerm, ConstantTerm, Delay, Dispersion, Rate, Polynomial,
    PolynomialFreq, PolynomialTime
export AbstractFeedTying, PerFeed, SharedFeeds, SingleFeed
export AbstractTimeSegmentation, GlobalTime, PerScan, PerIntegration, TimeBlocks,
    InstrumentScans
export AbstractFrequencySegmentation, GlobalFrequency, PerSpectralWindow,
    ChannelBlocks, FreqGroups, BandGroups
export AprioriAmplitude, AverageFrequency, CombineSpw, AverageTime, FlagSpwEdges
# Composable-pipeline surface: verbs, step protocol, execution config.
export fit
export SolveStep, ExecutionConfig, ProgressLogger
export outer_executor, inner_executor
export each_group, model_components, provides
export supports_station_heterogeneity
# Re-export the transform vocabulary and stage accessors so
# pipelines read naturally with a bare `using Gustavo`.
export AbstractDataTransform, AutocorrelationNormalization, ApplySolution,
    StationWeightScale, FlagChannels
export search_scan
# The solution surface: the container, selection (`sol[...]`), the two verbs,
# and serialization.
export CalibrationSolution, StepSolution, stage_info
export component_names, gains, parameters
export save_solution, load_solution, recorded_transforms

# A step author extends `solve`; unexported because the name is common.
@static if VERSION >= v"1.11"
    eval(Meta.parse("public solve, SolveContext"))
end
end
