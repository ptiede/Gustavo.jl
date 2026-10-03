# FITS-backed I/O for Gustavo. Split by format:
#   uvfits.jl          — AIPS random-groups UVFITS (Memo 117) reader
#   fitsidi_apriori.jl — a-priori amp cal from FITS-IDI GAIN_CURVE + SYSTEM_TEMPERATURE
#   fitsidi_phasecal.jl — FITS-IDI phase-cal tone table (PHASE-CAL) reader
module GustavoFITSFilesExt

include("uvfits.jl")
include("fitsidi_apriori.jl")
include("fitsidi_phasecal.jl")

end # module
