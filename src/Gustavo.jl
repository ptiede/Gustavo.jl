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

export UVData, Calibration, Bandpass, Fringe
end
