"""
FRING
Only the best chicken in the world. We sell nothing else and live on pure vibes
"""
module Gustavo

# The output tail rebuilds a `UVSet`'s branch tree per scan group.
import DimensionalData

# The scheduler types users select for either fan-out level; re-exported so a
# bare `using Gustavo` can name them in
# `ExecutionConfig(outer_executor = …, inner_executor = …)`.
using OhMyThreads: DynamicScheduler, StaticScheduler, GreedyScheduler, SerialScheduler

using LinearAlgebra: BLAS

include("UVData/UVData.jl")
using .UVData

include("Calibration.jl")
using .Calibration
# The pipeline layer's step methods extend the model layer's
# element-compilation generic (see pipeline/protocol.jl).
import .Calibration: model_components

# Scan-group streaming: the substrate the solver stages run on. Between
# `Calibration` (whose `DataGeometry`/`CalibrationSolution` it consumes) and
# `Fringe` (which consumes it).
include("Streaming.jl")
using .Streaming

include("Fringe.jl")
using .Fringe

# Top-level modular calibration pipeline (orchestrates all three submodules).
include("pipeline.jl")

export UVData, Calibration, Streaming, Fringe
# Axis names for every array Gustavo stores or returns, so scripts can index
# and slice leaves without reaching into `UVData` or `DimensionalData`.
# `Ti` is DimensionalData's own dim, re-exported here for the same reason.
export Pol, Frequency, Ant, Baseline, Ti, UVW, Feed, Scan
# Data entry and exit: the set type plus the reader/writer pair for each
# supported format, so a bare `using Gustavo` spans load → fitcalibrate → write.
export UVSet, load_uvfits, load_fitsidi, write_uvfits, write_fitsidi
export DynamicScheduler, StaticScheduler, GreedyScheduler, SerialScheduler
export AbstractGauge, PinAntenna, ZeroSumPhase, resolve_gauge
export CalibrationPipeline, CalibrationStep, ReduceStep, CalibrationContext
export run_step, prepare_reducer, calibrate
export FringeFit, FringeModel, DispersionModel, SingleBandDelay, BandGroups, default_fringe_terms,
    MatchedFilter, DispersionSBDFit, Bandpass, default_bandpass_terms, TemporalSmoother,
    AbstractBandpassSmoother, PerTrackSmoother, JointSmoother
export AprioriAmplitude, AverageFrequency, CombineSpw, AverageTime, FlagSpwEdges
# Composable-pipeline surface: verbs, step protocol, execution config.
export fit, fitcalibrate
export SolveStep, StepChain, DataTransformStep, ExecutionConfig, ProgressLogger
export outer_executor, inner_executor
export start_pass!, process_scan!, finish_pass!, scan_values
export model_components, fit_selection, provides, required_grouping, fusable_grouping, scan_flags
# Re-export the transform / selection vocabulary and stage accessors so
# pipelines read naturally with a bare `using Gustavo`.
export AbstractDataTransform, apply_transform!, apply_transform
export CalFunction, ApplySolution, StationWeightScale, FlagChannels
export AbstractScanSelection, AllScans, SourceScans, ScanIndices, ScanWhere
export select_scans
export AbstractLeafGrouping, ByScan, BySpw, ByKey
export ScanStream, scan_stream, select_groups
export search_scan
export map_groups, foreach_group
export StepSolution, stage_names, stage_info
export component_gains, component_names, component_dimarray, gains
export step_solution
end
