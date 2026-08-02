"""
FRING
Only the best chicken in the world. We sell nothing else and live on pure vibes
"""
module Gustavo

# The output tail rebuilds a `UVSet`'s branch tree per scan group.
import DimensionalData

# The executor seam: included FIRST so every submodule — the FITS-ext decode
# fan-outs, the Fringe kernels, the pass runner — spawns through it.
include("executors.jl")
using .Executors
# Inner-executor types users select for within-scan fan-out; re-exported so a
# bare `using Gustavo` can name them in `ExecutionConfig(inner_executor = …)`.
using OhMyThreads: DynamicScheduler, StaticScheduler, SerialScheduler

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
# Data entry and exit: the set type plus the reader/writer pair for each
# supported format, so a bare `using Gustavo` spans load → fitcalibrate → write.
export UVSet, load_uvfits, load_fitsidi, write_uvfits, write_fitsidi
export ThreadsExecutor, DaggerExecutor
export DynamicScheduler, StaticScheduler, SerialScheduler
export CalibrationPipeline, CalibrationStep, ReduceStep, CalibrationContext
export run_step, prepare_reducer, calibrate
export FringeFit, FringeModel, DispersionModel, SingleBandDelay, default_fringe_terms,
    MatchedFilter, DispersionSBDFit, BandpassEstimator, TemporalSmoother
export AprioriAmplitude, AverageFrequency, CombineSpw, AverageTime, FlagBandEdges
# Composable-pipeline surface: verbs, step protocol, execution config.
export fit, fitcalibrate
export SolveStep, StepChain, DataTransformStep, ExecutionConfig
export start_pass!, process_scan!, finish_pass!, scan_values
export model_components, fit_selection, provides, required_grouping
# Re-export the transform / selection vocabulary and stage accessors so
# pipelines read naturally with a bare `using Gustavo`.
export AbstractDataTransform, apply_transform!, apply_transform
export CalFunction, ApplySolution, StationWeightScale, FlagChannels
export AbstractScanSelection, AllScans, SourceScans, ScanIndices, ScanWhere
export select_scans
export AbstractLeafGrouping, ByScan, ByBand, ByKey
export ScanStream, scan_stream, select_groups
export search_scan
export map_groups, foreach_group
export StepSolution, stage_names, stage_solution, stage_info
export component_gains, component_names, gains, @comp
export step_solution
end
