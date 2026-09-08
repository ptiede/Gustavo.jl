# ── Calibration pipeline ──────────────────────────────────────────────────────
#
# A modular, ordered calibration pipeline. The user lists the steps and their
# order; `calibrate` threads a `CalibrationContext` through them. New steps are
# added by subtyping `CalibrationStep` (an action) or `ReduceStep` (a UVSet ->
# UVSet transform) and defining one method — `run_step` or `prepare_reducer`.
#
# Memory model ("solver owns its reductions"): fringe solve+correct+reduce is one
# streaming pass over the (lazy) data. Reduce steps placed in `FringeFit(reduce =
# [...])` are folded into that pass's `postprocess`, so they never materialize the
# full file. The same `ReduceStep`s also run standalone at the top level (eagerly,
# on the already-reduced `output`).

# The per-operation `UVSet -> UVSet` kernels backing the `ReduceStep` types;
# internal to `UVData` (the step types are their one public spelling).
using .UVData: scan_average, time_bin_average, frequency_average, flag_spw_edges,
    combine_spw

"""
    CalibrationStep

Abstract supertype of a pipeline stage. Implement `run_step(step, ctx)` to add
an action step (loads, solves, …).
"""
abstract type CalibrationStep end

"""
    ReduceStep <: CalibrationStep

A `UVSet -> UVSet` transform that can be *fused* into a solver's streaming pass
(via `FringeFit(reduce = [...])`) or run standalone. Implement
`prepare_reducer(step, ctx) -> (transform, ctx)`.
"""
abstract type ReduceStep <: CalibrationStep end

"""
    CalibrationContext

State threaded through the pipeline (immutable; rebuilt per step). Fields:
`uvset` (current, maybe lazy), `solution` (`CalibrationSolution`), `output`
(corrected/reduced `UVSet`), `spw_cals` (a-priori amplitude cal, if an
[`AprioriAmplitude`](@ref) step ran). Data loading is the caller's job
(`load_fitsidi`/`load_fitsidi_apriori`); the pipeline only transforms `uvset`.
"""
Base.@kwdef struct CalibrationContext
    uvset = nothing
    solution::Union{Nothing, CalibrationSolution} = nothing
    output = nothing
    spw_cals = nothing
end

# Rebuild a context with selected fields overridden.
function _with(
        ctx::CalibrationContext;
        uvset = ctx.uvset, solution = ctx.solution,
        output = ctx.output, spw_cals = ctx.spw_cals,
    )
    return CalibrationContext(uvset, solution, output, spw_cals)
end

"""
    run_step(step::CalibrationStep, ctx::CalibrationContext) -> CalibrationContext

Run an action `step`, returning the updated context. The extension point for
action steps. The fallback for a [`ReduceStep`](@ref) applies its transform
eagerly to `ctx.output` (or `ctx.uvset`).
"""
function run_step end

"""
    prepare_reducer(step::ReduceStep, ctx) -> (transform::Any, ctx::CalibrationContext)

Return a `UVSet -> UVSet` `transform` and a (possibly updated) context. Called by
[`FringeFit`](@ref) to fold the step into its streaming pass, and by the default
[`run_step`](@ref) for standalone use.
"""
function prepare_reducer end

# A ReduceStep at the top level runs eagerly on the materialized output.
function run_step(step::ReduceStep, ctx::CalibrationContext)
    f, ctx = prepare_reducer(step, ctx)
    target = ctx.output !== nothing ? ctx.output : ctx.uvset
    target === nothing && error("$(nameof(typeof(step))): no `output`/`uvset` in context to reduce.")
    if is_lazy(target)
        @warn "Applying $(nameof(typeof(step))) eagerly to a lazy UVSet forces a full " *
            "materialization; put it in `FringeFit(reduce = [...])` to fuse it into the streaming pass."
    end
    return _with(ctx; output = f(target))
end

"""
    (step::ReduceStep)(uvset::UVSet) -> UVSet

Apply a reduce step eagerly to a whole `UVSet`, so the pipeline vocabulary
also composes in scripts: `uvset |> AverageFrequency(nout = 1) |> CombineSpw()`.
On a lazy set this forces a full materialization; put the step in a
`reduce = [...]` list to fuse it into a streaming pass instead.
"""
function (step::ReduceStep)(uvset::UVSet)
    f, _ = prepare_reducer(step, CalibrationContext(; uvset))
    return f(uvset)
end

# The composable-pipeline step protocol: SolveStep, the step hooks, step
# chaining (`|>`), and the CalibrationPipeline itself.
include("pipeline/protocol.jl")

# The built-in solve steps: FringeFit (model + estimator), Bandpass,
# TemporalSmoother.
include("pipeline/steps.jl")


# ── Antenna-name lookup (shared by the bridge and the runner) ───────────────

function _antenna_names(uvset)
    leaf = first(values(UVData.branches(uvset)))
    return collect(UVData.metadata(leaf).antennas.name)
end

"""
    AprioriAmplitude(spw_cals; min_elevation_deg = 0.0, on_missing_station = :warn)

Output-chain pipeline step (not a reduction): a-priori amplitude calibration.

Being an output step, this scales the data a solve *produces*, never the data it
*reads*: a `Bandpass` in the same pipeline is fit on unscaled amplitudes, so a
per-channel SEFD lands in the fitted gains instead of being divided out ahead of
them. Use [`AprioriPreCal`](@ref) — a data transform, applied to each scan group
as it is materialized — when the solvers should see calibrated amplitudes.

Applies a pre-built `spw_cals` (`load_fitsidi_apriori(path)` — the caller's
job) to the fringe-corrected data, after the solution's gains and
interleaved with any `ReduceStep`s in whatever relative order the
`CalibrationPipeline` declares them — e.g. placed before a `ReduceStep` that
merges spws, it sees the native per-spw channels; placed after, it sees the
reduced ones. It is recorded on the fitted solution (`sol.postcal`), so the
standalone `calibrate(sol, uvset)` reproduces it without re-passing
`spw_cals`.
"""
struct AprioriAmplitude{C} <: CalibrationStep
    spw_cals::C
    min_elevation_deg::Float64
    on_missing_station::Symbol
end
AprioriAmplitude(spw_cals; min_elevation_deg::Real = 0.0, on_missing_station::Symbol = :warn) =
    AprioriAmplitude(spw_cals, Float64(min_elevation_deg), on_missing_station)

"""
    AverageFrequency(; nout = 1)

Reduce step: inverse-variance-average each spw's `Frequency` axis into `nout`
channels (`frequency_average`).
"""
Base.@kwdef struct AverageFrequency <: ReduceStep
    nout::Int = 1
end
prepare_reducer(s::AverageFrequency, ctx::CalibrationContext) =
    ((uv -> frequency_average(uv; nout = s.nout)), ctx)

"""
    CombineSpw()

Reduce step: merge sibling spw leaves into one `Frequency` axis (`combine_spw`).
Apply after [`AverageFrequency`](@ref) so each spw is one channel.
"""
struct CombineSpw <: ReduceStep end
prepare_reducer(::CombineSpw, ctx::CalibrationContext) = (combine_spw, ctx)


"""
    AverageTime(; seconds = nothing)

Reduce step: inverse-variance-average each leaf's `Ti` axis — into
`seconds`-wide bins, or, with no `seconds`, collapsing each scan to a single
sample.
"""
Base.@kwdef struct AverageTime <: ReduceStep
    seconds::Union{Nothing, Float64} = nothing
end
prepare_reducer(s::AverageTime, ctx::CalibrationContext) =
    (s.seconds === nothing ? scan_average : (uv -> time_bin_average(uv, s.seconds)), ctx)

"""
    FlagSpwEdges(; mode = :flag_fraction, fraction = 0.0)

Reduce step: handle polyphase-filterbank spectral-window edges (`flag_spw_edges`).
`:flag_fraction` zeros the outer `fraction` of channels' weights at each edge;
`:trim` drops them.
"""
Base.@kwdef struct FlagSpwEdges <: ReduceStep
    mode::Symbol = :flag_fraction
    fraction::Float64 = 0.0
end
prepare_reducer(s::FlagSpwEdges, ctx::CalibrationContext) =
    ((uv -> flag_spw_edges(uv; mode = s.mode, fraction = s.fraction)), ctx)


# ── Runner ────────────────────────────────────────────────────────────────────

# The composable-pipeline verbs: fit / calibrate(sol, uvset) / fitcalibrate —
# solve steps share one compiled model and streaming passes.
include("pipeline/verbs.jl")
