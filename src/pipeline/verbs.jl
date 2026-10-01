# ── Pipeline verbs: fit / calibrate ──────────────────────────────────────────
#
#     sol = fit(pipeline, ps)                      # solve
#     out = calibrate(sol, ps; post)               # apply
#
# `fit` runs each solve step's `solve` on the data as the corrections and steps
# before it leave it. `calibrate` divides each Measurement Set by the solution's
# gains, then applies its flags and `post`.

# A solve's parallelism is across scan groups and, within a group, across
# baselines; each task's WLS/QR solve is small. Multithreaded BLAS underneath
# that multiplies out to `tasks × BLAS.get_num_threads()` threads competing for
# the same cores, which contend rather than progress. One BLAS thread is the
# correct split, and only the caller can set it process-wide.
function _check_blas_threads()
    n = BLAS.get_num_threads()
    n == 1 || @warn """
    BLAS is using $n threads; a solve parallelizes across scan groups and baselines, so \
    threaded BLAS oversubscribes the cores and slows the solve. Set \
    `BLAS.set_num_threads(1)` (`using LinearAlgebra`) before fitting.""" maxlog = 1
    return nothing
end

"""
    fit(pipeline, ps::ProcessingSet; exec = ExecutionConfig()) -> CalibrationSolution
    fit(pipeline, ms::MeasurementSet; exec = ExecutionConfig()) -> CalibrationSolution

Solve the pipeline's solve steps on `ps`, in order, one scan group
(`groupby(ps, ByScan())`) at a time. `pipeline` is a tuple (or vector) of
solve steps ([`BaselineFringeFit`](@ref), [`Bandpass`](@ref),
[`AdhocPhase`](@ref) or a third-party `SolveStep`) and corrections (an
[`AbstractDataTransform`](@ref) such as [`AutocorrelationNormalization`](@ref),
or any function from a Measurement Set to a Measurement Set), usually built
with `|>`, or a single one of them. Each solve step reads the data every
earlier correction and step has corrected.
Narrow the data by subsetting `ps` first. A Measurement Set is fit as a
processing set of one.

Each solve step carries its own gauge ([`step_gauge`](@ref)); `fit` resolves
its station codes against the data's antenna table. `exec` (an
[`ExecutionConfig`](@ref)) supplies the run's schedulers, memory budget and
progress callback.

The solution is a [`CalibrationSolution`](@ref): a list of solved components,
each step's diagnostics in `sol.steps`, and the pipeline, its steps' gauges
included, as text in `sol.provenance`; [`calibrate`](@ref)`(pipeline, sol, ps)` applies it along the
same data path. Solving `A |> B` is equivalent to `sa = fit(A, ps)` followed
by `fit(ApplySolution(sa[provides(A)]) |> B, ps)`.
"""
function StatsAPI.fit(
        pipeline::Union{Tuple, AbstractVector}, ps::XRadio.ProcessingSet;
        exec::ExecutionConfig = ExecutionConfig(),
    )
    _check_blas_threads()
    return _run_pipeline(_parse_pipeline(pipeline), exec, ps)
end

StatsAPI.fit(pipeline::Union{Tuple, AbstractVector}, ms::XRadio.MeasurementSet; kwargs...) =
    fit(pipeline, _one_member(ms); kwargs...)
StatsAPI.fit(x::PipelineElement, data::Union{XRadio.ProcessingSet, XRadio.MeasurementSet}; kwargs...) =
    fit((x,), data; kwargs...)

"""
    calibrate(sol::CalibrationSolution, ps::ProcessingSet;
              post = identity, apply_flags = true, exec = ExecutionConfig()) -> ProcessingSet
    calibrate(pipeline, sol::CalibrationSolution, ps::ProcessingSet; kwargs...) -> ProcessingSet

Apply a fitted solution, or any selection of one (`sol[:bandpass]`,
`sol[:fringe, :phase, :mbd]`, `filter(pred, sol)`), to data: each Measurement
Set of `ps` is read and divided by the gains of the components `sol` holds,
then the flags of its steps and `post`, a function from a Measurement Set to a
Measurement Set, are applied. A gain cell that is zero or non-finite gives a
NaN visibility and a flag.

The first form divides by the gains only. The second repeats the data path of
the fit: `pipeline` is the one `sol` was fit with, walked in order, each
correction applied as written and each solve step replaced by that step's
gains in `sol`. Its solve steps must be exactly the steps `sol` holds.

```julia
gauge = PinAntenna("AA")
pipeline = AutocorrelationNormalization() |> BaselineFringeFit(; gauge) |> Bandpass(; gauge)
sol = fit(pipeline, ps)
out = calibrate(pipeline, sol, ps)   # normalized, then divided by each step's gains
```

`apply_flags` (default `true`) flags the baselines of each (station, scan)
a step of `sol` left unconstrained (the fringe step records these): their gains
are identity, so the data would pass through uncalibrated. The flagged samples
keep their visibilities and weights.

Each sample is placed in the solution by scan, spectral window and time as
[`ApplySolution`](@ref) places it, so `ps` may be other data than the solution
was fit on. `exec` supplies the schedulers and progress callback.
"""
calibrate(sol::CalibrationSolution, ps::XRadio.ProcessingSet; kwargs...) =
    _calibrate(Any[_StepGains(Calibration._applied(sol))], sol, ps; kwargs...)

calibrate(pipeline::Union{Tuple, AbstractVector, PipelineElement}, sol::CalibrationSolution, ps::XRadio.ProcessingSet; kwargs...) =
    _calibrate(_replay_chain(pipeline, sol), sol, ps; kwargs...)

function _calibrate(
        chain, sol::CalibrationSolution, ps::XRadio.ProcessingSet;
        post = identity, apply_flags::Bool = true,
        exec::ExecutionConfig = ExecutionConfig(),
    )
    geom = DataGeometry(ps)
    flagged = apply_flags ? _solution_flag_sets(sol) : nothing
    names = collect(keys(ps))
    members = collect(values(ps))
    out = _map_groups(members, zeros(Int, length(members)), exec; stage = :output) do ms
        corrected = _apply_corrections(chain, read(ms), geom)
        post(_flag_unconstrained(corrected, sol.geom, geom, flagged))
    end
    return XRadio.ProcessingSet(
        OrderedDict{Symbol, XRadio.MeasurementSet}(names .=> out),
        copy(DimensionalData.metadata(ps)),
    )
end

# The gains of one solved step, as `calibrate` applies them: a degenerate gain
# cell flags the sample.
struct _StepGains{S <: Calibration._AppliedSolution}
    app::S
end

_correct(g::_StepGains, ms::XRadio.MeasurementSet, geom::DataGeometry) =
    _divide_gains(ms, GeometryWindow(geom, ms), g.app; flag_bad = true)

# `pipeline` as corrections, each solve step replaced by its gains in `sol`; a
# step that compiled no components contributes none.
function _replay_chain(pipeline, sol::CalibrationSolution)
    seq = collect(Any, _check_pipeline(pipeline isa PipelineElement ? (pipeline,) : pipeline))
    steps = [provides(x) for x in seq if x isa SolveStep]
    Set(steps) == Set(keys(sol.steps)) || throw(
        ArgumentError(
            "calibrate: the pipeline's solve steps $(steps) are not the solution's steps " *
                "$(collect(keys(sol.steps))); pass the pipeline the solution was fit with, " *
                "or select the solution to match"
        )
    )
    chain = Any[]
    for x in seq
        if !(x isa SolveStep)
            push!(chain, x)
        elseif haskey(sol, provides(x))
            push!(chain, _StepGains(Calibration._applied(sol[provides(x)])))
        end
    end
    return chain
end

"""
    calibrate(antab::UVData.AntabCalibration, uvset::UVSet; kwargs...) -> UVSet
    calibrate(spw_cals::AbstractDict{<:Integer, UVData.AntabCalibration}, uvset::UVSet; kwargs...) -> UVSet

Apply an a-priori amplitude calibration — a single ANTAB table, or one per
1-based band index as [`load_fitsidi_apriori`](@ref) returns — scaling
visibilities and weights by the SEFD-derived gains. Keywords
(`min_elevation_deg`, `on_missing_station`) pass through. To record the
application on a fitted solution instead, put an [`AprioriAmplitude`](@ref)
in the pipeline.
"""
calibrate(antab::UVData.AntabCalibration, uvset::UVSet; kwargs...) =
    UVData.apply_calibration(uvset, antab; kwargs...)
calibrate(
    spw_cals::AbstractDict{<:Integer, <:UVData.AntabCalibration}, uvset::UVSet;
    kwargs...,
) = UVData.apply_calibration(uvset, spw_cals; kwargs...)

# The unconstrained (station name, scan id) pairs the steps of `sol` record, as
# a lookup set, `nothing` when they record none.
function _solution_flag_sets(sol::CalibrationSolution)
    flagged = Set{Tuple{String, Int}}()
    for info in values(sol.steps)
        haskey(info, :flagged_ant) || continue
        for i in eachindex(info.flagged_ant, info.flagged_scan)
            push!(flagged, (String(info.flagged_ant[i]), Int(info.flagged_scan[i])))
        end
    end
    return isempty(flagged) ? nothing : flagged
end

# Flag the samples of each baseline touching a (station, scan) in `flagged`,
# scans indexing the solution's geometry `solgeom`, matched by name.
function _flag_unconstrained(ms::XRadio.MeasurementSet, solgeom::DataGeometry, geom::DataGeometry, flagged)
    flagged === nothing && return ms
    station = geom.stations
    scan = [something(findfirst(==(String(c)), solgeom.scan_names), 0) for c in ms[:scan_name]]
    flag = modify(Array, ms[:flag])
    hit = false
    for (bi, (a, b)) in pairs(GeometryWindow(geom, ms).stations)
        a == b && continue
        sa, sb = station[a], station[b]
        for ti in eachindex(scan)
            ((sa, scan[ti]) in flagged || (sb, scan[ti]) in flagged) || continue
            view(flag, BaselineID(bi), Ti(ti)) .= true
            hit = true
        end
    end
    hit || return ms
    return _with_layers(ms; flag)
end

# ── The runner ───────────────────────────────────────────────────────────────

_resolve_step_gauge(g::AbstractGauge, ant_names) = resolve_gauge(g, ant_names)
_resolve_step_gauge(::Nothing, ant_names) = nothing

# Solve a pipeline: each solve step compiles and solves its own private
# (model, layout, θ), never a merged one. A step reads the data through the
# corrections as they stand at the step's position: those listed before it, in
# order, with each earlier step's own solution (`ApplySolution`) where that
# step sat.
function _run_pipeline(
        br, exec::ExecutionConfig, ps::XRadio.ProcessingSet,
    )
    geom = DataGeometry(ps)
    spec = (; geom)
    gauges = [_resolve_step_gauge(step_gauge(st), geom.stations) for st in br.solve_steps]
    groups = groupby(ps, XRadio.ByScan())
    charges = [_group_charge(g) for g in values(groups)]
    _check_memory_budget(charges, exec)
    corrections = Any[]
    components = SolvedComponent[]
    steps = OrderedDict{Symbol, NamedTuple}()
    for (st, before, gauge) in zip(br.solve_steps, br.before, gauges)
        append!(corrections, before)
        ctx = _step_context(st, spec, gauge, groups, charges, copy(corrections), exec)
        t0 = time_ns()
        info = solve(st, ctx)
        info isa NamedTuple || throw(
            ArgumentError(
                "`solve(::$(nameof(typeof(st))), ctx)` must return a NamedTuple of " *
                    "diagnostics, got a $(typeof(info))",
            ),
        )
        stepsol = CalibrationSolution(ctx.model, ctx.layout, geom, ctx.θ; name = provides(st))
        append!(components, stepsol.components)
        steps[provides(st)] = _with_timing(info, ctx, t0)
        # Every later step reads data corrected by this one; a step that
        # compiled no components corrects nothing.
        isempty(stepsol.components) || push!(corrections, ApplySolution(stepsol))
    end
    return CalibrationSolution(
        geom, components, steps, _run_info(br, groups, exec);
        pipeline = join((sprint(show, x; context = :limit => true) for x in br.sequence), " |> "),
    )
end

# A step compiles its own model against the run's geometry, and may
# legitimately compile no components at all; it still runs, with nothing to
# solve. The model is materialized against the run's stations here, so the
# context (and the solution's provenance) holds the concrete per-station trees
# the solve uses; a step that has not opted into station heterogeneity
# (`supports_station_heterogeneity`) is handed uniform models only — anything else is rejected before any data is read.
function _step_context(st::SolveStep, spec, gauge, groups, charges, corrections, exec)
    stations = spec.geom.stations
    model = Calibration.materialize(model_components(st, spec), stations, spec.geom)
    supports_station_heterogeneity(st) ||
        Calibration.require_station_uniform(model, stations, heterogeneity_rejector(st))
    layout = plan_parameters(model, stations, spec.geom; require_nonempty = false)
    return SolveContext(
        model, layout, spec.geom, zeros(layout.nθ), gauge, length(stations),
        groups, charges, corrections, exec, provides(st), _PassTiming[],
    )
end

# A step's diagnostics plus `t_pass`, the wall time of its `solve`, and
# `timing`, a `DimStack` over `Scan` of each group's `decode`/`work` seconds
# summed over the step's passes.
function _with_timing(info::NamedTuple, ctx::SolveContext, t0::UInt64)
    ngroups = length(ctx.groups)
    decode, work = zeros(ngroups), zeros(ngroups)
    for pass in ctx.passes
        decode .+= pass.decode
        work .+= pass.work
    end
    timing = DimensionalData.DimStack((; decode, work), (UVData.Scan(1:ngroups),))
    return (; info..., t_pass = (time_ns() - t0) / 1.0e9, timing)
end

# The run-wide diagnostics. Each step's own diagnostics are `sol.steps[name]`.
function _run_info(br, groups, exec)
    return (;
        nscan = length(groups),
        precal_applied = any(t -> t isa ApplySolution, Iterators.flatten(br.before)),
        ntasks_used = max_tasks(outer_executor(exec)),
        inner_tasks = max_tasks(inner_executor(exec)),
    )
end

# ── Pipeline parsing ─────────────────────────────────────────────────────────

# Step order is never validated here — a step that cannot do its job with the
# data it is handed fails at the point of use (its own solve kernel). Only
# `provides`'s naming role is checked: a step's components and diagnostics are
# keyed by it, so two steps sharing a value would collide.
function _check_unique_provides(solve_steps::Vector{SolveStep})
    provided = Symbol[]
    for s in solve_steps
        p = provides(s)
        p in provided && throw(
            ArgumentError(
                "fit: more than one step provides :$p; a step's solution is keyed by " *
                    "`provides`, so the names must be distinct."
            )
        )
        push!(provided, p)
    end
    return nothing
end

# Split a pipeline into its solve steps and, for each, the corrections listed
# between it and the step before it (`before[k]`).
function _parse_pipeline(pipeline)
    sequence = collect(Any, _check_pipeline(pipeline))
    solve_steps = SolveStep[]
    before = Vector{Any}[]
    pending = Any[]
    for x in sequence
        if x isa SolveStep
            push!(solve_steps, x)
            push!(before, pending)
            pending = Any[]
        else
            push!(pending, x)
        end
    end
    isempty(solve_steps) && throw(ArgumentError("fit: the pipeline holds no solve step"))
    _check_unique_provides(solve_steps)
    return (; sequence, solve_steps, before)
end
