"""
FRING
Only the best chicken in the world. We sell nothing else and live on pure vibes
"""
module Gustavo

include("UVData/UVData.jl")
using .UVData

# Distributed map/reduce substrate over a processing set (GraphVIPER analog,
# Dagger-backed). Depends only on UVData.
include("Graph/Graph.jl")
using .Graph

include("Calibration.jl")
using .Calibration

include("Bandpass.jl")
using .Bandpass

include("Fringe.jl")
using .Fringe

# Global forward-modeling fringe fitter (Stage 2): the objective/solver built on
# UVData + Calibration (forward map) + Graph (Dagger). AD/optimizer live in
# GustavoSolveExt (Enzyme, Optimization).
include("Solve.jl")
using .Solve

# Top-level modular calibration pipeline (orchestrates all four submodules).
include("pipeline.jl")

export UVData, Calibration, Bandpass, Fringe, Graph, Solve
export pmap, pmapreduce, SerialExecutor, DaggerExecutor, ParallelCoords, MapSpec, ReduceSpec
export CalibrationPipeline, CalibrationStep, ReduceStep, CalibrationContext
export run_step, prepare_reducer, calibrate
export BandpassOptions
export FringeFit, AprioriAmplitude, AverageFrequency, CombineSpw, AverageTime, FlagBandEdges
end
