# FITS-backed I/O for Gustavo. Split by format:
#   uvfits.jl          — AIPS random-groups UVFITS (Memo 117), eager read/write
#   fitsidi_read.jl    — FITS-IDI (Memo 114) streaming/lazy reader
#   fitsidi_write.jl   — FITS-IDI writer
#   fitsidi_apriori.jl — a-priori amp cal from GAIN_CURVE + SYSTEM_TEMPERATURE
#   fitsidi_phasecal.jl — phase-cal tone table (PHASE-CAL) reader
module GustavoFITSFilesExt

include("uvfits.jl")
include("fitsidi_read.jl")
include("fitsidi_write.jl")
include("fitsidi_apriori.jl")
include("fitsidi_phasecal.jl")

end # module
