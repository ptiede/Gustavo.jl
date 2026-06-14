# FITS-backed I/O for Gustavo. Split by format:
#   uvfits.jl        — AIPS random-groups UVFITS (Memo 117), eager read/write
#   fitsidi_read.jl  — FITS-IDI (Memo 114) streaming/lazy reader   (planned)
#   fitsidi_write.jl — FITS-IDI writer                             (planned)
module GustavoFITSFilesExt

include("uvfits.jl")

end # module
