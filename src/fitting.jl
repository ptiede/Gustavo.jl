# ── Fitting: fit, calibrate, mapsets ─────────────────────────────────────────
#
# `fit` solves one specification on a set of scan groups; `calibrate!` divides a
# solution's gains out of the data; `mapsets` runs a function over units of data
# read into memory.

include("fitting/execution.jl")
include("fitting/corrections.jl")
include("fitting/protocol.jl")
include("fitting/solvers.jl")
include("fitting/verbs.jl")
include("fitting/fringe.jl")
include("fitting/mapsets.jl")
