# ── Calibration pipeline ──────────────────────────────────────────────────────
#
# `fit` solves one step on a set of scan groups; `calibrate!` divides a
# solution's gains out of the data; `mapsets` runs a function over units of data
# read into memory.

# Run-wide resources: schedulers, progress.
include("pipeline/execution.jl")

# Corrections: functions that modify a Measurement Set in place.
include("pipeline/corrections.jl")

# The step protocol: SolveStep and its hooks.
include("pipeline/protocol.jl")

# The built-in solve steps: BaselineFringeFit, Bandpass, AdhocPhase.
include("pipeline/steps.jl")

# ── Runner ────────────────────────────────────────────────────────────────────

# The verbs: fit and calibrate.
include("pipeline/verbs.jl")

# `mapsets`: a function over units of data, each read into memory.
include("pipeline/mapsets.jl")
