# ── Verbs: fit / calibrate ───────────────────────────────────────────────────
#
#     sol = fit(step, ps)                          # solve
#     calibrate!(sol, ps; post)                    # apply in place
#
# `fit` runs a solve step's `solve` on the data. `calibrate!` divides each
# Measurement Set by the solution's gains, then applies its flags and `post`.

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
    fit(step::SolveStep, ps::ProcessingSet; exec = ExecutionConfig()) -> CalibrationSolution
    fit(step::SolveStep, ms::MeasurementSet; exec = ExecutionConfig()) -> CalibrationSolution

Solve `step` ([`BaselineFringeFit`](@ref), [`Bandpass`](@ref),
[`AdhocPhase`](@ref) or a third-party `SolveStep`) on `ps`, one scan group
(`groupby(ps, ByScan())`) at a time. `ps` is never modified. Narrow the data by
subsetting `ps` first. A Measurement Set is fit as a processing set of one.

To fit a step on data an earlier solution has corrected, correct the data
first: [`calibrate!`](@ref) on data read into memory, or inside
[`mapsets`](@ref).

The step carries its own gauge ([`step_gauge`](@ref)); `fit` resolves its
station codes against the data's antenna table. `exec` (an
[`ExecutionConfig`](@ref)) supplies the schedulers and progress callback.

The solution is a [`CalibrationSolution`](@ref): the solved components, the
step's diagnostics in `sol.steps`, and the step, its gauge included, as text in
`sol.provenance`.
"""
function StatsAPI.fit(
        step::SolveStep, ps::XRadio.ProcessingSet;
        exec::ExecutionConfig = ExecutionConfig(),
    )
    _check_blas_threads()
    return _run_step(step, exec, ps)
end

StatsAPI.fit(step::SolveStep, ms::XRadio.MeasurementSet; kwargs...) =
    fit(step, _one_member(ms); kwargs...)

"""
    calibrate(sol::CalibrationSolution, ps::ProcessingSet; kwargs...) -> ProcessingSet
    calibrate(sol::CalibrationSolution, ms::MeasurementSet; kwargs...) -> MeasurementSet

[`calibrate!`](@ref) applied to an in-memory copy of the data whose arrays are
its own; the data passed in is left as it was.
"""
calibrate(sol::CalibrationSolution, data::Union{XRadio.ProcessingSet, XRadio.MeasurementSet}; kwargs...) =
    calibrate!(sol, materialize(data); kwargs...)

"""
    calibrate!(sol::CalibrationSolution, ps::ProcessingSet;
               flag_bad = true, apply_flags = true, post = identity,
               exec = ExecutionConfig()) -> ps
    calibrate!(sol::CalibrationSolution, ms::MeasurementSet; flag_bad = true, apply_flags = true) -> ms

Apply a fitted solution, or any selection of one (`sol[:bandpass]`,
`sol[:fringe, :phase, :mbd]`, `filter(pred, sol)`), to the data in place: each
Measurement Set has the gains of the components `sol` holds divided out
(`V → V / (g_a·conj(g_b))`, `w → w·|g_a g_b|²`), then the flags of its step
and `post` applied. `post` modifies the Measurement Set it is handed and
returns it. To keep the data, use [`calibrate`](@ref); `copy` and `read` of
in-memory data share its arrays.

```julia
gauge = PinAntenna("AA")
fr = fit(BaselineFringeFit(; gauge), ps)
mapsets(groupby(ps, ByScan())) do g
    calibrate!(fr, g; flag_bad = false, apply_flags = false)
    fit(AdhocPhase(; gauge), g)
end
```

A gain cell that is zero or non-finite gives a NaN visibility and a flag
(`flag_bad = true`) or leaves the sample as it is (`flag_bad = false`).
`apply_flags` flags the baselines of each (station, scan) a step of `sol` left
unconstrained (the fringe step records these): their gains are identity, so the
data would pass through uncalibrated. The flagged samples keep their
visibilities and weights.

The solution need not have been fit on this data, nor on its sampling. Each
sample is placed in the segment of `sol` it belongs to, by the identity the
segmentation is defined on: a `PerScan` component by scan name, a
`PerSpectralWindow` one by spectral window name, a `TimeBlocks` or
`InstrumentScans` one by the solve's own bin formula, a `PerIntegration` one by
exact epoch. So a solution segmented more coarsely than the data applies, while
one segmented more finely is refused. A `ChannelBlocks` or `FreqGroups`
component places each channel by its center frequency and width: the data's
channel must be a channel of the solve, no wider, so averaged channels are
refused.

Stations are matched by name against the solution's geometry; stations the
solution never solved keep identity gains, with a warning. A solution that
shares no station with the data is refused. `exec` supplies the schedulers and
progress callback.
"""
function calibrate!(
        sol::CalibrationSolution, ps::XRadio.ProcessingSet;
        flag_bad::Bool = true, apply_flags::Bool = true, post = identity,
        exec::ExecutionConfig = ExecutionConfig(),
    )
    app = Calibration._applied(sol)
    flagged = apply_flags ? _solution_flag_sets(sol) : nothing
    geom = DataGeometry(ps)
    members = collect(values(ps))
    _map_groups(members, zeros(Int, length(members)), exec; stage = :output) do ms
        _apply_solution!(ms, app, geom, flagged; flag_bad, executor = inner_executor(exec))
        _check_corrected(post, ms, post(ms))
    end
    return ps
end

function calibrate!(
        sol::CalibrationSolution, ms::XRadio.MeasurementSet;
        flag_bad::Bool = true, apply_flags::Bool = true,
    )
    flagged = apply_flags ? _solution_flag_sets(sol) : nothing
    return _apply_solution!(ms, Calibration._applied(sol), DataGeometry(_one_member(ms)), flagged; flag_bad)
end

# `app`'s gains divided out of `ms`, then the baselines of `flagged` flagged.
function _apply_solution!(
        ms::XRadio.MeasurementSet, app::Calibration._AppliedSolution, geom::DataGeometry, flagged;
        flag_bad::Bool, executor = DynamicScheduler(),
    )
    _divide_gains!(ms, GeometryWindow(geom, ms), app; flag_bad, executor)
    return _flag_unconstrained!(ms, app.geom, geom, flagged)
end

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
function _flag_unconstrained!(ms::XRadio.MeasurementSet, solgeom::DataGeometry, geom::DataGeometry, flagged)
    flagged === nothing && return ms
    station = geom.stations
    scan = [something(findfirst(==(String(c)), solgeom.scan_names), 0) for c in ms[:scan_name]]
    flag = ms[:flag]
    for (bi, (a, b)) in pairs(GeometryWindow(geom, ms).stations)
        a == b && continue
        sa, sb = station[a], station[b]
        for ti in eachindex(scan)
            ((sa, scan[ti]) in flagged || (sb, scan[ti]) in flagged) || continue
            view(flag, BaselineID(bi), Ti(ti)) .= true
        end
    end
    return ms
end

# ── The runner ───────────────────────────────────────────────────────────────

_resolve_step_gauge(g::AbstractGauge, ant_names) = resolve_gauge(g, ant_names)
_resolve_step_gauge(::Nothing, ant_names) = nothing

# Solve one step: compile its model against the data's geometry, then run its
# `solve` over the scan groups.
function _run_step(st::SolveStep, exec::ExecutionConfig, ps::XRadio.ProcessingSet)
    geom = DataGeometry(ps)
    spec = (; geom)
    gauge = _resolve_step_gauge(step_gauge(st), geom.stations)
    groups = groupby(ps, XRadio.ByScan())
    sizes = [_group_bytes(g) for g in values(groups)]
    ctx = _step_context(st, spec, gauge, groups, sizes, exec)
    t0 = time_ns()
    info = solve(st, ctx)
    info isa NamedTuple || throw(
        ArgumentError(
            "`solve(::$(nameof(typeof(st))), ctx)` must return a NamedTuple of " *
                "diagnostics, got a $(typeof(info))",
        ),
    )
    stepsol = CalibrationSolution(ctx.model, ctx.layout, geom, ctx.θ; name = provides(st))
    steps = OrderedDict{Symbol, NamedTuple}(provides(st) => _with_timing(info, ctx, t0))
    return CalibrationSolution(
        geom, stepsol.components, steps, _run_info(groups, exec);
        pipeline = sprint(show, st; context = :limit => true),
    )
end

# A step compiles its own model against the run's geometry, and may
# legitimately compile no components at all; it still runs, with nothing to
# solve. The model is resolved against the run's stations here, so the
# context (and the solution's provenance) holds the concrete per-station trees
# the solve uses; a step that has not opted into station heterogeneity
# (`supports_station_heterogeneity`) is handed uniform models only — anything else is rejected before any data is read.
function _step_context(st::SolveStep, spec, gauge, groups, sizes, exec)
    stations = spec.geom.stations
    model = Calibration.resolve(model_components(st, spec), stations, spec.geom)
    supports_station_heterogeneity(st) ||
        Calibration.require_station_uniform(model, stations, heterogeneity_rejector(st))
    layout = plan_parameters(model, stations, spec.geom; require_nonempty = false)
    return SolveContext(
        model, layout, spec.geom, zeros(layout.nθ), gauge, length(stations),
        groups, sizes, exec, provides(st), _PassTiming[],
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
    timing = DimensionalData.DimStack((; decode, work), (Fring._scan_dim(_group_scan_names(ctx.groups)),))
    return (; info..., t_pass = (time_ns() - t0) / 1.0e9, timing)
end

# The scan name of each `ByScan` group, in group order.
function _group_scan_names(groups)
    names = String[String(k.scan) for k in keys(groups)]
    allunique(names) || throw(
        ArgumentError("scan names repeat across scan groups: $(join(unique(filter(n -> count(==(n), names) > 1, names)), ", "))")
    )
    return names
end

# The run-wide diagnostics. Each step's own diagnostics are `sol.steps[name]`.
function _run_info(groups, exec)
    return (;
        nscan = length(groups),
        ntasks_used = max_tasks(outer_executor(exec)),
        inner_tasks = max_tasks(inner_executor(exec)),
    )
end
