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
    CalibrationPipeline(steps)
    CalibrationPipeline(step1, step2, …)

An ordered list of [`CalibrationStep`](@ref)s, run by [`calibrate`](@ref).
"""
struct CalibrationPipeline
    steps::Vector{CalibrationStep}
end
CalibrationPipeline(steps::CalibrationStep...) = CalibrationPipeline(collect(steps))

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

# Compose the reduce steps (in order) into a single postprocess closure, threading
# the context (so e.g. a-priori can stash its band_cals).
function _compose_reducers(steps, ctx::CalibrationContext)
    transforms = Vector{Any}(undef, length(steps))
    for (i, st) in enumerate(steps)
        f, ctx = prepare_reducer(st, ctx)
        transforms[i] = f
    end
    post = isempty(transforms) ? identity :
        uv -> foldl((acc, f) -> f(acc), transforms; init = uv)
    return post, ctx
end


# ── Option holders ────────────────────────────────────────────────────────────

"""
    BandpassOptions(; phase = true, amp = true, amp_smoother = PolynomialBandpass(4), source = nothing)

Fringe relative phase/amplitude bandpass toggles for [`FringeFit`](@ref).
`source` names the bandpass calibrator (default: the brightest). `amp_smoother`
selects the amplitude-bandpass estimator — an `AbstractBandpassSmoother`, e.g.
[`PolynomialBandpass`](@ref) (degree), [`PenalizedBandpass`](@ref) (λ), or
[`FreeBandpass`](@ref).
"""
Base.@kwdef struct BandpassOptions
    phase::Bool = true
    amp::Bool = true
    amp_smoother::AbstractBandpassSmoother = PolynomialBandpass(4)
    source = nothing
end


# ── Built-in steps ────────────────────────────────────────────────────────────

"""
    FringeFit(; ref_ant = 1, rounds = 1, search = FringeSearch(), adhoc = AdhocPhasing(),
              ntasks = Threads.nthreads(), mem_fraction = 0.6, mem_budget = nothing,
              bandpass = BandpassOptions(), reduce = ReduceStep[])

Action step: the fringe solver. Folds `reduce` (a list of [`ReduceStep`](@ref)s,
in order) into the single streaming `postprocess` pass of `solve_and_reduce_fringes`,
so reductions never materialize the full file. Sets `ctx.solution`, `ctx.output`,
and (if a-priori is in `reduce`) `ctx.band_cals`.

`ref_ant` may be a 1-based antenna index OR a station code (`String`/`Symbol`,
e.g. `"PT"`) resolved against the data's antenna table. `rounds` re-runs the
delay/rate/phase search pass on the residual (each round divides out the current
solution and accumulates the leftover); >1 helps when one matched-filter pass
leaves residual delay/rate, at the cost of another full streaming read.

Group concurrency is bounded by a DETERMINISTIC memory budget: `mem_budget`
(absolute bytes) if set, else `mem_fraction` of TOTAL physical RAM — so `ntasks`
(and the solve's wall time) is reproducible across runs. Lower `mem_fraction` (or
set `mem_budget`) on a shared box; pass an explicit `mem_budget` for identical
behavior across machines.
"""
Base.@kwdef struct FringeFit <: CalibrationStep
    ref_ant::Any = 1
    rounds::Int = 1
    search::FringeSearch = FringeSearch()
    adhoc::AdhocPhasing = AdhocPhasing()
    ntasks::Int = Threads.nthreads()
    mem_fraction::Float64 = 0.6
    mem_budget::Union{Nothing, Float64} = nothing
    bandpass::BandpassOptions = BandpassOptions()
    reduce::Vector{ReduceStep} = ReduceStep[]
end

function run_step(s::FringeFit, ctx::CalibrationContext)
    ctx.uvset === nothing &&
        error("FringeFit: no `uvset` in context — add a Load step or call `calibrate(uvset, pipe)`.")
    ref = _resolve_ref_ant(s.ref_ant, ctx.uvset)
    post, ctx = _compose_reducers(s.reduce, ctx)
    sol, output = solve_and_reduce_fringes(
        ctx.uvset;
        postprocess = post,
        ref_ant = ref, rounds = s.rounds,
        search = s.search, adhoc = s.adhoc,
        ntasks = s.ntasks, mem_fraction = s.mem_fraction, mem_budget = s.mem_budget,
        phase_bandpass = s.bandpass.phase, amp_bandpass = s.bandpass.amp,
        amp_smoother = s.bandpass.amp_smoother,
        bandpass_source = s.bandpass.source,
    )
    return _with(ctx; solution = sol, output = output)
end

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

Reduce step: a-priori amplitude calibration. Applies a pre-built `band_cals`
(`load_fitsidi_apriori(path)` — the caller's job) via `apply_calibration` on the
fringe-corrected native channels.
"""
struct AprioriAmplitude <: ReduceStep
    band_cals::Any
    min_elevation_deg::Float64
    on_missing_station::Symbol
end
AprioriAmplitude(band_cals; min_elevation_deg::Real = 0.0, on_missing_station::Symbol = :warn) =
    AprioriAmplitude(band_cals, Float64(min_elevation_deg), on_missing_station)

function prepare_reducer(s::AprioriAmplitude, ctx::CalibrationContext)
    f = uv -> apply_calibration(
        uv, s.band_cals;
        min_elevation_deg = s.min_elevation_deg, on_missing_station = s.on_missing_station,
    )
    return f, _with(ctx; band_cals = s.band_cals)
end

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

"""
    calibrate(uvset::UVSet, pipe::CalibrationPipeline) -> CalibrationContext

Run `pipe`'s steps in order on an already-loaded `uvset`, threading a
[`CalibrationContext`](@ref). The result exposes `.uvset`, `.solution`, `.output`,
and `.band_cals`. Loading (`load_fitsidi`/`load_fitsidi_apriori`) is done by the
caller and kept out of the pipeline.
"""
function calibrate(uvset::UVSet, pipe::CalibrationPipeline)
    ctx = CalibrationContext(; uvset = uvset)
    for step in pipe.steps
        ctx = run_step(step, ctx)
    end
    return ctx
end
