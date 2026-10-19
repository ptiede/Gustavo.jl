# ── Steps: fit, calibrate, mapsets ───────────────────────────────────────────
#
# `fit` solves one step on a set of scan groups; `calibrate!` divides a
# solution's gains out of the data; `mapsets` runs a function over units of data
# read into memory.

include("steps/execution.jl")
include("steps/corrections.jl")
include("steps/protocol.jl")
include("steps/steps.jl")
include("steps/verbs.jl")
include("steps/mapsets.jl")
