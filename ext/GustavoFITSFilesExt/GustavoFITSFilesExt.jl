# FITS-backed I/O for Gustavo. Split by format:
#   uvfits.jl          — AIPS random-groups UVFITS (Memo 117) reader
#   fitsidi_phasecal.jl — FITS-IDI phase-cal tone table (PHASE-CAL) reader
module GustavoFITSFilesExt

include("uvfits.jl")
include("fitsidi_phasecal.jl")

end # module
