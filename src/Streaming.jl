"""
    Streaming

Scan-group streaming over a (possibly lazy) `UVSet`: the unit of data flow the
calibration pipeline runs on. Domain-independent — it knows about memory
budgets, executors and `DimTree` leaves, not about fringes:

- `transforms.jl` — [`AbstractDataTransform`](@ref), the per-materialization
  hook chain, and the scan `DimStack` + [`GeometryWindow`](@ref) it mutates.
- `selections.jl` — [`AbstractScanSelection`](@ref), naming WHICH scan groups a
  pass reads.
- `stream.jl` — [`ScanStream`](@ref) construction, group materialization, and
  the budget-admitted pass runner [`map_groups`](@ref).

Per-task scratch is the layer's one hook into a consumer's kernels: a stream
carries a `Channel{W}` pool whose element type comes from `scan_stream`'s
`workspace` factory, so a consumer that needs scratch (`Gustavo.Fringe`'s FFT
workspaces) supplies it and this module never names it.
"""
module Streaming

using ..Executors
using ..Executors: exec_foreach
using ..UVData
using ..UVData: Baseline, Frequency, Pol
using ..Calibration
using ..Calibration: GeometryWindow
import DimensionalData
using DimensionalData: DimArray, DimStack, AbstractDimStack, lookup, Ti

include("Streaming/transforms.jl")
include("Streaming/selections.jl")
include("Streaming/stream.jl")

export AbstractDataTransform, apply_transform!, apply_transform
export ApplySolution, StationWeightScale, FlagChannels, CalFunction
export station_weight_scale
export AbstractScanSelection, AllScans, SourceScans, BrightestCalibrator, ScanIndices, ScanWhere
export select_scans
export AbstractLeafGrouping, ByScan, ByBand, ByKey
export ScanStream, scan_stream, ScanGroupSpec, select_groups
export materialize_cube, materialize_leaves
export map_groups, foreach_group

end
