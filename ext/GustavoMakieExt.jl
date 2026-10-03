module GustavoMakieExt

using Makie
using Makie:
    Figure, Axis, Colorbar, Label,
    scatter!, lines!, linesegments!, hlines!, text!,
    linkxaxes!, linkyaxes!, ylims!, axislegend,
    MarkerElement, LineElement,
    hidexdecorations!, hideydecorations!,
    colsize!, Fixed, resize_to_layout!,
    cgrad

using Printf: @sprintf

include("GustavoMakieExt_fringe.jl")
include("GustavoMakieExt_gains.jl")
include("GustavoMakieExt_coherence.jl")

end # module
