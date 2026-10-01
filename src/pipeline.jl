# ── Calibration pipeline ──────────────────────────────────────────────────────
#
# A pipeline is an ordered tuple of solve steps and corrections. `fit` solves
# the steps in order, each on the data the corrections and steps before it
# produced, and records the sequence on the solution; `calibrate` replays it.

# Run-wide resources: schedulers, progress.
include("pipeline/execution.jl")

# Corrections: Measurement Set → Measurement Set, the recorded ones as structs.
include("pipeline/corrections.jl")

# The step protocol: SolveStep, its hooks, and `|>` building a pipeline.
include("pipeline/protocol.jl")

# The built-in solve steps: BaselineFringeFit, Bandpass, AdhocPhase.
include("pipeline/steps.jl")

# ── Runner ────────────────────────────────────────────────────────────────────

# The pipeline verbs: fit and calibrate.
include("pipeline/verbs.jl")
