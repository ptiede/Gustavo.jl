module GustavoMakieExt

using Makie
using Makie:
    Figure, Axis, Colorbar, Label,
    scatter!, lines!, hlines!,
    linkxaxes!, linkyaxes!, ylims!,
    hidexdecorations!, hideydecorations!,
    colsize!

using Printf: @sprintf

include("GustavoMakieExt_fringe.jl")
include("GustavoMakieExt_gains.jl")
include("GustavoMakieExt_coherence.jl")

end # module
