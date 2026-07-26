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
(corrected/reduced `UVSet`), `band_cals` (a-priori amplitude cal, if an
[`AprioriAmplitude`](@ref) step ran). Data loading is the caller's job
(`load_fitsidi`/`load_fitsidi_apriori`); the pipeline only transforms `uvset`.
"""
Base.@kwdef struct CalibrationContext
    uvset = nothing
    solution::Union{Nothing, CalibrationSolution} = nothing
    output = nothing
    band_cals = nothing
end

# Rebuild a context with selected fields overridden.
function _with(
        ctx::CalibrationContext;
        uvset = ctx.uvset, solution = ctx.solution,
        output = ctx.output, band_cals = ctx.band_cals,
    )
    return CalibrationContext(uvset, solution, output, band_cals)
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

# The composable-pipeline step protocol: SolveStep, the step hooks,
# ExecutionConfig, step chaining (`|>`), and the CalibrationPipeline itself.
include("pipeline/protocol.jl")

# The built-in solve steps: FringeFit (model + estimator), BandpassEstimator,
# TemporalSmoother.
include("pipeline/steps.jl")


# ── Reference-antenna resolution (shared by the bridge and the runner) ───────

# Resolve a reference antenna to its 1-based index. An Integer passes through; a
# station code (String/Symbol) is matched against the data's antenna table (the
# reader maps NOSTA → 1-based row, so the row index is the solver's station index).
_resolve_ref_ant(r::Integer, uvset) = Int(r)
function _resolve_ref_ant(code::Union{AbstractString, Symbol}, uvset)
    names = _antenna_names(uvset)
    i = findfirst(==(String(code)), names)
    i === nothing &&
        error("FringeFit ref_ant: station code \"$code\" not in antenna table $(names).")
    return i
end

function _antenna_names(uvset)
    leaf = first(values(UVData.branches(uvset)))
    return collect(UVData.metadata(leaf).antennas.name)
end

"""
    AprioriAmplitude(band_cals; min_elevation_deg = 0.0, on_missing_station = :warn)

Output-chain pipeline step (NOT a reduction): a-priori amplitude calibration.
Applies a pre-built `band_cals` (`load_fitsidi_apriori(path)` — the caller's
job) via `apply_calibration` on the fringe-corrected native channels, after the
solution's gains and before any `ReduceStep`s. Place it in the
`CalibrationPipeline`; it is RECORDED on the fitted solution (`sol.postcal`),
so the standalone `calibrate(sol, uvset)` reproduces it without re-passing
`band_cals`.
"""
struct AprioriAmplitude <: CalibrationStep
    band_cals::Any
    min_elevation_deg::Float64
    on_missing_station::Symbol
end
AprioriAmplitude(band_cals; min_elevation_deg::Real = 0.0, on_missing_station::Symbol = :warn) =
    AprioriAmplitude(band_cals, Float64(min_elevation_deg), on_missing_station)

"""
    AverageFrequency(; nout = 1)

Reduce step: inverse-variance-average each band's `Frequency` axis into `nout`
channels (`frequency_average`).
"""
Base.@kwdef struct AverageFrequency <: ReduceStep
    nout::Int = 1
end
prepare_reducer(s::AverageFrequency, ctx::CalibrationContext) =
    ((uv -> frequency_average(uv; nout = s.nout)), ctx)

"""
    CombineSpw()

Reduce step: merge sibling band leaves into one `Frequency` axis (`combine_spw`).
Apply after [`AverageFrequency`](@ref) so each band is one channel.
"""
struct CombineSpw <: ReduceStep end
prepare_reducer(::CombineSpw, ctx::CalibrationContext) = (combine_spw, ctx)

"""
    AverageTime(; seconds)

Reduce step: inverse-variance-average the `Ti` axis into `seconds`-wide bins
(`time_bin_average`).
"""
Base.@kwdef struct AverageTime <: ReduceStep
    seconds::Float64
end
prepare_reducer(s::AverageTime, ctx::CalibrationContext) =
    ((uv -> time_bin_average(uv, s.seconds)), ctx)

"""
    FlagBandEdges(; mode = :flag_fraction, fraction = 0.0)

Reduce step: handle polyphase-filterbank band edges (`flag_band_edges`).
`:flag_fraction` zeros the outer `fraction` of channels' weights at each edge;
`:trim` drops them.
"""
Base.@kwdef struct FlagBandEdges <: ReduceStep
    mode::Symbol = :flag_fraction
    fraction::Float64 = 0.0
end
prepare_reducer(s::FlagBandEdges, ctx::CalibrationContext) =
    ((uv -> flag_band_edges(uv; mode = s.mode, fraction = s.fraction)), ctx)


# ── Runner ────────────────────────────────────────────────────────────────────

# The composable-pipeline verbs: fit / calibrate(sol, uvset) / fitcalibrate —
# solve steps share one compiled model and streaming passes.
include("pipeline/verbs.jl")
