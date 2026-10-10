# ── Calibration pipeline ──────────────────────────────────────────────────────
#
# `fit` solves one step on a set of scan groups; `calibrate!` divides a
# solution's gains out of the data; `mapsets` runs a function over units of data
# read into memory.

include("pipeline/execution.jl")
include("pipeline/corrections.jl")
include("pipeline/protocol.jl")
include("pipeline/steps.jl")
include("pipeline/verbs.jl")
include("pipeline/mapsets.jl")
