# FITS-backed I/O for Gustavo:
#   uvfits.jl          — AIPS random-groups UVFITS (Memo 117) reader
#   uvfits_write.jl    — its writer
module GustavoFITSFilesExt

include("uvfits.jl")
include("uvfits_write.jl")

end # module
