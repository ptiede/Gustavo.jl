"""
    Streaming

Scan-group streaming over a (possibly lazy) `UVSet`: the unit of data flow the
calibration pipeline runs on. Domain-independent — it knows about memory
budgets, executors and `DimTree` leaves, not about fringes:

- `execution.jl` — [`ExecutionConfig`](@ref), the run-wide memory budget,
  schedulers and progress callback a stream is built from.
- `transforms.jl` — [`AbstractDataTransform`](@ref), the per-materialization
  hook chain, and the scan `DimStack` + [`GeometryWindow`](@ref) it mutates.
- `selections.jl` — [`AbstractScanSelection`](@ref), naming which scan groups a
  pass reads.
- `stream.jl` — [`ScanStream`](@ref) construction, group materialization, and
  the concurrent pass runner [`map_groups`](@ref).

This layer owns the group/budget/transform machinery and never names a
consumer's kernels: it hands a materialized scan `DimStack` to whatever reads it
(`Gustavo.Fringe`'s search allocates its own per-task FFT scratch).
"""
module Streaming

using OhMyThreads: tforeach, tmap, Scheduler
using OhMyThreads: DynamicScheduler, StaticScheduler, GreedyScheduler, SerialScheduler
using OhMyThreads.Schedulers: chunking_enabled, has_nchunks, nchunks
using ..UVData
using ..UVData: Baseline, Frequency, Pol
using ..Calibration
using ..Calibration: GeometryWindow
import DimensionalData
using DimensionalData: DimArray, DimStack, AbstractDimStack, lookup, Ti

include("Streaming/execution.jl")
include("Streaming/transforms.jl")
include("Streaming/selections.jl")
include("Streaming/stream.jl")

export ExecutionConfig, ProgressLogger, outer_executor, inner_executor
export AbstractDataTransform, apply_transform!, apply_transform
export ApplySolution, StationWeightScale, FlagChannels, CalFunction
export station_weight_scale
export AbstractScanSelection, AllScans, SourceScans, ScanIndices, ScanWhere
export select_scans
export AbstractLeafGrouping, ByScan, BySpw, ByKey
export ScanStream, scan_stream, ScanGroupSpec, select_groups
export materialize_cube, materialize_leaves
export map_groups, foreach_group

end
