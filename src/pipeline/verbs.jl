# ── Pipeline verbs: fit / calibrate ──────────────────────────────────────────
#
#     sol = fit(pipeline, ps; gauge)               # solve
#     out = calibrate(sol, ps; post)               # apply
#
# `fit` runs each solve step's `solve` on the data as the corrections and steps
# before it leave it. `calibrate` replays the recorded sequence over each
# Measurement Set, then the flags and `post`.

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
    fit(pipeline, ps::ProcessingSet; gauge, exec = ExecutionConfig()) -> CalibrationSolution
    fit(pipeline, ms::MeasurementSet; gauge, exec = ExecutionConfig()) -> CalibrationSolution

Solve the pipeline's solve steps on `ps`, in order, one scan group
(`groupby(ps, ByScan())`) at a time. `pipeline` is a tuple (or vector) of
solve steps ([`BaselineFringeFit`](@ref), [`DispersionSBDFit`](@ref),
[`Bandpass`](@ref), [`AdhocPhase`](@ref) or a third-party `SolveStep`) and
corrections (an [`AbstractDataTransform`](@ref) such as
[`AutocorrelationNormalization`](@ref), or any function from a Measurement Set
to a Measurement Set), usually built with `|>`, or a single one of them. Each
solve step reads the data every earlier correction and step has corrected.
Narrow the data by subsetting `ps` first. A Measurement Set is fit as a
processing set of one.

`gauge` is required: an [`AbstractGauge`](@ref) such as `PinAntenna("PT")`,
`PinAntenna(["PT", "LA"])` for a ranked fallback, or `ZeroSumPhase()`; without
one `fit` throws, naming the stations. `exec` (an [`ExecutionConfig`](@ref))
supplies the run's schedulers, memory budget and progress callback.

The solution records the pipeline as a tuple (`sol.sequence`) and the gauge
(`sol.gauge`), so `fit(sol.sequence, ps; sol.gauge)` repeats the run and
[`calibrate`](@ref)`(sol, ps)` replays it. `sol[:fringe]` selects one step,
`sol[1:i]` the cumulative view through step `i`, and [`stage_info`](@ref) a
step's diagnostics. Solving `A |> B` is equivalent to `sa = fit(A, ps)`
followed by `fit(ApplySolution(sa[provides(A)]) |> B, ps)`.
"""
function StatsAPI.fit(
        pipeline::Union{Tuple, AbstractVector}, ps::XRadio.ProcessingSet;
        gauge::Union{Nothing, AbstractGauge} = nothing,
        exec::ExecutionConfig = ExecutionConfig(),
    )
    _check_blas_threads()
    return _run_pipeline(_parse_pipeline(pipeline), exec, gauge, ps)
end

StatsAPI.fit(pipeline::Union{Tuple, AbstractVector}, ms::XRadio.MeasurementSet; kwargs...) =
    fit(pipeline, _one_member(ms); kwargs...)
StatsAPI.fit(x::PipelineElement, data::Union{XRadio.ProcessingSet, XRadio.MeasurementSet}; kwargs...) =
    fit((x,), data; kwargs...)

"""
    calibrate(sol::CalibrationSolution, ps::ProcessingSet;
              post = identity, apply_flags = true, exec = ExecutionConfig()) -> ProcessingSet

Apply a fitted solution to data: each Measurement Set of `ps` is read, passed
through `sol.sequence` in order (each correction as recorded, each solve step
as that step's gains), then the solution's flags, then `post`, a function from
a Measurement Set to a Measurement Set. A gain cell that is zero or non-finite
gives a NaN visibility and a flag.

`apply_flags` (default `true`) flags the baselines of each (station, scan)
the fringe solve left unconstrained: their gains are identity, so the data
would pass through uncalibrated. The flagged samples keep their visibilities
and weights.

Each sample is placed in the solution by scan, spectral window and time as
[`ApplySolution`](@ref) places it, so `ps` may be other data than the solution
was fit on. `exec` supplies the schedulers and progress callback.
"""
function calibrate(
        sol::CalibrationSolution, ps::XRadio.ProcessingSet;
        post = identity, apply_flags::Bool = true,
        exec::ExecutionConfig = ExecutionConfig(),
    )
    any(ismissing, sol.sequence) && throw(
        ArgumentError(
            "calibrate: this solution records a pipeline element that did not survive " *
                "serialization (saved as `missing`) — re-fit, or apply it manually."
        )
    )
    geom = DataGeometry(ps)
    replay = _replay_sequence(sol)
    flagged = apply_flags ? _solution_flag_sets(sol.info) : nothing
    names = collect(keys(ps))
    members = collect(values(ps))
    out = _map_groups(members, zeros(Int, length(members)), exec; stage = :output) do ms
        corrected = _apply_corrections(replay, read(ms), geom)
        post(_flag_unconstrained(corrected, sol, geom, flagged))
    end
    return XRadio.ProcessingSet(
        OrderedDict{Symbol, XRadio.MeasurementSet}(names .=> out),
        copy(DimensionalData.metadata(ps)),
    )
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

# The gains of one solved step, as `calibrate` applies them: a degenerate gain
# cell flags the sample.
struct _StepGains{S <: CalibrationSolution}
    sol::S
end

_correct(g::_StepGains, ms::XRadio.MeasurementSet, geom::DataGeometry) =
    _divide_gains(ms, GeometryWindow(geom, ms), g.sol; flag_bad = true)

# `sol.sequence` as corrections, each solve step replaced by its own gains.
function _replay_sequence(sol::CalibrationSolution)
    replay = Any[]
    k = 0
    for x in sol.sequence
        if x isa SolveStep
            k += 1
            push!(replay, _StepGains(sol[k]))
        else
            push!(replay, x)
        end
    end
    k == length(sol.steps) || throw(
        ArgumentError(
            "calibrate: the solution records $(length(sol.steps)) solved step(s) but its " *
                "sequence lists $k"
        )
    )
    return replay
end

Calibration.recorded_transforms(sol::CalibrationSolution) =
    Any[x for x in sol.sequence if !(x isa SolveStep)]

# The solution's unconstrained (station, scan id) pairs as a lookup set,
# `nothing` when the solution records none.
function _solution_flag_sets(info::NamedTuple)
    (haskey(info, :flagged_ant) && !isempty(info.flagged_ant)) || return nothing
    return Set{Tuple{Int, Int}}(
        (Int(info.flagged_ant[i]), Int(info.flagged_scan[i]))
            for i in eachindex(info.flagged_ant)
    )
end

# Flag the samples of each baseline touching a (station, scan) in `flagged`,
# stations and scans being the solution's own, matched by name.
function _flag_unconstrained(ms::XRadio.MeasurementSet, sol::CalibrationSolution, geom::DataGeometry, flagged)
    flagged === nothing && return ms
    solnames = _solution_ant_names(sol)
    station = [something(findfirst(==(n), solnames), 0) for n in geom.stations]
    scan = [something(findfirst(==(String(c)), sol.geom.scan_names), 0) for c in ms[:scan_name]]
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

_resolve_run_gauge(g::AbstractGauge, ant_names) = resolve_gauge(g, ant_names)
_resolve_run_gauge(::Nothing, ant_names) = throw(
    ArgumentError(
        "no gauge given; pass `gauge = PinAntenna(station)` or `gauge = ZeroSumPhase()`. " *
            "Stations: $(join(ant_names, ", ")).",
    ),
)

# Solve a pipeline: each solve step compiles and solves its own private
# (model, layout, θ), never a merged one. A step reads the data through the
# corrections as they stand at the step's position: those listed before it, in
# order, with each earlier step's own solution (`ApplySolution`) where that
# step sat.
function _run_pipeline(
        br, exec::ExecutionConfig, gauge_spec::Union{Nothing, AbstractGauge}, ps::XRadio.ProcessingSet,
    )
    geom = DataGeometry(ps)
    gauge = _resolve_run_gauge(gauge_spec, geom.stations)
    groups = groupby(ps, XRadio.ByScan())
    charges = [_group_charge(g) for g in values(groups)]
    _check_memory_budget(charges, exec)
    spec = (; geom)
    corrections = Any[]
    step_solutions = StepSolution[]
    for (st, before) in zip(br.solve_steps, br.before)
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
        push!(
            step_solutions,
            StepSolution(provides(st), ctx.model, ctx.layout, ctx.θ, _with_timing(info, ctx, t0)),
        )
        # Each step's θ is undivided (it solves its own private model), so it
        # joins the corrections as-is and every later step reads corrected data.
        push!(corrections, ApplySolution(_step_precal(st, ctx, geom)))
    end
    return CalibrationSolution(
        step_solutions, geom, _run_info(step_solutions, br, geom, groups, exec);
        sequence = Tuple(br.sequence), gauge = gauge_spec,
    )
end

# A step compiles its own model against the run's geometry, and may
# legitimately compile no components at all (e.g. `DispersionSBDFit` with
# dispersion disabled and a band layout that can't support SBD either); it
# still runs, with nothing to solve. The model is materialized against the
# run's stations here, so the context (and the solution's provenance) holds the
# concrete per-station trees the solve uses; a step that has not opted into
# station heterogeneity (`supports_station_heterogeneity`) is handed uniform
# models only — anything else is rejected before any data is read.
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

# One finished step's own solution, as the correction a later step divides out.
# `ApplySolution` matches stations by NAME and refuses a solution that names
# none, so the station table is recorded here as it is on the run's own
# solution.
_step_precal(st, c::SolveContext, geom::DataGeometry) = CalibrationSolution(
    c.model, c.layout, geom, c.θ, (; ant_names = copy(geom.stations));
    name = provides(st),
)

# The solution-level `info`: run-wide fields, plus the fringe step's
# unconstrained (station, scan) flags, which `calibrate` applies, and its
# search configuration. Every other per-step diagnostic lives on that step's
# `StepSolution.info` (`stage_info(sol, name)`).
function _run_info(step_solutions, br, geom, groups, exec)
    ffi = findfirst(st -> st isa BaselineFringeFit, br.solve_steps)
    ff_info = ffi === nothing ? nothing : step_solutions[ffi].info
    return (;
        nant = length(geom.stations),
        nscan = length(groups),
        flagged_ant = ff_info === nothing ? Int[] : ff_info.flagged_ant,
        flagged_scan = ff_info === nothing ? Int[] : ff_info.flagged_scan,
        ant_names = copy(geom.stations),
        (ffi === nothing ? (;) : (; search = br.solve_steps[ffi].search))...,
        precal_applied = any(t -> t isa ApplySolution, Iterators.flatten(br.before)),
        ntasks_used = max_tasks(outer_executor(exec)),
        inner_tasks = max_tasks(inner_executor(exec)),
    )
end

# ── Pipeline parsing ─────────────────────────────────────────────────────────

# Step order is never validated here — a step that cannot do its job with the
# data it is handed fails at the point of use (its own solve kernel), the same
# pattern `apply_calibration`'s data-level guards use. Only
# `provides`'s NAMING role is checked: two steps sharing a non-`:nothing`
# capability would silently collide in the by-name lookup behind `sol[name]`
# (a `findfirst`, so the second step's solution would be unreachable) — that is a naming conflict, not an ordering rule, so it stays
# a construction-time error regardless of where the two steps sit.
function _check_unique_provides(solve_steps::Vector{SolveStep})
    provided = Symbol[]
    for s in solve_steps
        p = provides(s)
        if p !== :nothing
            p in provided && throw(
                ArgumentError(
                    "fit: more than one step provides :$p — exactly one is " *
                        "allowed per pipeline."
                )
            )
            push!(provided, p)
        end
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
