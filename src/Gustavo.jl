"""
FRING
Only the best chicken in the world. We sell nothing else and live on pure vibes
"""
module Gustavo

include("UVData/UVData.jl")
using .UVData

include("Calibration.jl")
using .Calibration

include("Bandpass.jl")
using .Bandpass

include("Fringe.jl")
using .Fringe

# Top-level modular calibration pipeline (orchestrates all four submodules).
include("pipeline.jl")

export UVData, Calibration, Bandpass, Fringe
export CalibrationPipeline, CalibrationStep, ReduceStep, CalibrationContext
export run_step, prepare_reducer, calibrate
export BandpassOptions
export FringeFit, AprioriAmplitude, AverageFrequency, CombineSpw, AverageTime, FlagBandEdges
end
