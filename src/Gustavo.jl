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

include("UVData/UVData.jl")
using .UVData

include("Calibration.jl")
using .Calibration

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
export AbstractExecutor, ThreadsExecutor, DaggerExecutor, with_executor
export CalibrationPipeline, CalibrationStep, ReduceStep, CalibrationContext
export run_step, prepare_reducer, calibrate
export FringeFit, FringeModel, DispersionModel, CrossFeed, MatchedFilter, BandpassEstimator, TemporalSmoother
export AprioriAmplitude, AverageFrequency, CombineSpw, AverageTime, FlagBandEdges
# Composable-pipeline surface: verbs, step protocol, execution config.
export fit, fitcalibrate
export SolveStep, StepChain, DataTransformStep, ExecutionConfig
export start_pass!, process_scan!, finish_pass!
export model_components, fit_selection, provides, requires, required_grouping
# Re-export the transform / selection vocabulary and stage accessors so
# pipelines read naturally with a bare `using Gustavo`.
export AbstractDataTransform, ScanDataView, apply_transform!, apply_transform
export CalFunction, ApplySolution, StationWeightScale, FlagChannels
export AbstractScanSelection, AllScans, SourceScans, BrightestCalibrator, ScanIndices, ScanWhere
export select_scans
export AbstractLeafGrouping, ByScan, ByBand, ByKey
export ScanStream, scan_stream, ScanGroup, select_groups
export ScanSearchResult, search_scan
export map_groups, foreach_group
export StageRecord, StageView, stage_names, stage_solution, stage_info, component_gains
export bandpass_solution
end
