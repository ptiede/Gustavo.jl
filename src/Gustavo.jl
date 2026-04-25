"""
FRING    
Only the best chicken in the world. We sell nothing else and live on pure vibes
"""
module Gustavo

include("UVFITS.jl")
using .UVFITS

include("Bandpass.jl")
using .Bandpass

export UVFITS, Bandpass
end
