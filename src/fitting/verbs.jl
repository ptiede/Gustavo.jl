# ── Verbs: fit / calibrate ───────────────────────────────────────────────────
#
#     sol = fit(solver, ps)                        # solve
#     calibrate!(sol, ps; post)                    # apply in place
#
# `fit` runs a solver's `solve` on the data. `calibrate!` divides each
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
    fit(solver::AbstractSolver, ps::ProcessingSet; exec = ExecutionConfig()) -> CalibrationSolution
    fit(solver::AbstractSolver, ms::MeasurementSet; exec = ExecutionConfig()) -> CalibrationSolution

Solve `solver` ([`Bandpass`](@ref), [`AdhocPhase`](@ref) or a third-party
`AbstractSolver`) on `ps`, one scan group
(`groupby(ps, ByScan())`) at a time. `ps` is never modified. Narrow the data by
subsetting `ps` first. A Measurement Set is fit as a processing set of one.

To fit data an earlier solution has corrected, correct the data first:
[`calibrate!`](@ref) on data read into memory, or inside [`mapsets`](@ref).

The solver carries its own gauge ([`solver_gauge`](@ref)); `fit` resolves its
station codes against the data's antenna table. `exec` (an
[`ExecutionConfig`](@ref)) supplies the schedulers and progress callback.

The solution is a [`CalibrationSolution`](@ref): the solved components, the
solver's diagnostics in `sol.info`, and the solver, its gauge included, as
text in `sol.spec`.
"""
function StatsAPI.fit(
        solver::AbstractSolver, ps::XRadio.ProcessingSet;
        exec::ExecutionConfig = ExecutionConfig(),
    )
    _check_blas_threads()
    return _run_solver(solver, exec, ps)
end

StatsAPI.fit(solver::AbstractSolver, ms::XRadio.MeasurementSet; kwargs...) =
    fit(solver, _one_member(ms); kwargs...)

"""
    calibrate(sol::CalibrationSolution, ps::ProcessingSet; exec = ExecutionConfig(), kwargs...) -> ProcessingSet
    calibrate(sol::CalibrationSolution, ms::MeasurementSet; exec = ExecutionConfig(), kwargs...) -> MeasurementSet

[`calibrate!`](@ref) applied to an in-memory copy of the data whose arrays are
its own; the data passed in is left as it was. The Measurement Sets are read on
`exec`'s `inner_executor`. The whole copy is held in memory; for data larger
than memory, calibrate each unit of a [`mapsets`](@ref) pass with `calibrate!`
and store it with [`write!`](@ref).
"""
calibrate(
    sol::CalibrationSolution, data::Union{XRadio.ProcessingSet, XRadio.MeasurementSet};
    exec::ExecutionConfig = ExecutionConfig(), kwargs...,
) = calibrate!(sol, _read_unit(data, inner_executor(exec)); exec, kwargs...)

"""
    calibrate!(sol::CalibrationSolution, ps::ProcessingSet;
               flag_bad = true, apply_flags = true, post = identity,
               exec = ExecutionConfig()) -> ps
    calibrate!(sol::CalibrationSolution, ms::MeasurementSet;
               flag_bad = true, apply_flags = true, exec = ExecutionConfig()) -> ms

Apply a fitted solution, or any selection of one (`sol[:phase]`,
`sol[:phase, :mbd]`, `filter(pred, sol)`), to the data in place: each
Measurement Set has the gains of the components `sol` holds divided out
(`V → V / (g_a·conj(g_b))`, `w → w·|g_a g_b|²`), then the flags `sol`
records and `post` applied. `post` modifies the Measurement Set it is handed
and returns it. To keep the data, use [`calibrate`](@ref); `copy` and `read` of
in-memory data share its arrays.

```julia
gauge = PinAntenna("AA")
fr = fit(FringeClosure(; gauge), fit(FringeSearch(), ps))
mapsets(groupby(ps, ByScan())) do g
    calibrate!(fr, g; flag_bad = false, apply_flags = false)
    fit(AdhocPhase(; gauge), g)
end
```

A gain cell that is zero or non-finite gives a NaN visibility and a flag
(`flag_bad = true`) or leaves the sample as it is (`flag_bad = false`).
`apply_flags` flags the baselines of each (station, scan) the fit behind `sol`
left unconstrained (the fringe fit records these): their gains are identity, so
the data would pass through uncalibrated. The flagged samples keep their
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

Stations are matched by name against the solution's geometry. The baselines of
a station the solution does not name are flagged under `apply_flags`, as an
unconstrained station's are, and otherwise left uncorrected; either way with a
warning. A solution that shares no station with the data is refused. A store
opened with `mode = "r+"` is calibrated in place only if its visibility, weight
and flag layers hold one time per chunk; one chunked over several times is
refused, since tasks writing into one chunk would lose each other's writes. `exec` supplies the schedulers and
progress callback; a single Measurement Set uses only its `inner_executor`.
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
        _apply_solution!(ms, app, geom, flagged; flag_bad, apply_flags, executor = inner_executor(exec))
        _check_corrected(post, ms, post(ms))
    end
    return ps
end

function calibrate!(
        sol::CalibrationSolution, ms::XRadio.MeasurementSet;
        flag_bad::Bool = true, apply_flags::Bool = true, exec::ExecutionConfig = ExecutionConfig(),
    )
    flagged = apply_flags ? _solution_flag_sets(sol) : nothing
    return _apply_solution!(
        ms, Calibration._applied(sol), DataGeometry(_one_member(ms)), flagged;
        flag_bad, apply_flags, executor = inner_executor(exec),
    )
end

# `app`'s gains divided out of `ms`, then the baselines of `flagged` flagged.
function _apply_solution!(
        ms::XRadio.MeasurementSet, app::Calibration._AppliedSolution, geom::DataGeometry, flagged;
        flag_bad::Bool, apply_flags::Bool, executor = DynamicScheduler(),
    )
    _check_time_chunks(ms)
    win = GeometryWindow(geom, ms)
    _divide_gains!(ms, win, app; flag_bad, executor)
    _flag_absent_antennas!(ms, win, app.geom.antennas; apply_flags)
    return _flag_unconstrained!(ms, win, app.geom, flagged)
end

# Tasks divide out whole time samples, so a stored layer chunked over several
# times loses writes: each task reads a chunk, changes its own samples and
# writes the whole chunk back.
function _check_time_chunks(ms::XRadio.MeasurementSet)
    for key in (:visibility, :weight, :flag)
        z = parent(ms[key])
        z isa Zarr.ZArray || continue
        n = maximum(length, DiskArrays.eachchunk(z).chunks[DimensionalData.dimnum(ms[key], Ti)])
        n == 1 || throw(
            ArgumentError(
                "`$key` is stored in chunks of $n times, and `calibrate!` writes each time " *
                    "from its own task, so writes into one chunk would be lost. Calibrate the " *
                    "data in memory (`calibrate`, or `calibrate!` on each unit of `mapsets`) and " *
                    "store it with `write!`, or write the store with `chunks = (; time = 1)`"
            )
        )
    end
    return nothing
end

# The cross baselines of stations `solved` does not name have no gains to divide
# out; flag them (`apply_flags`) or leave them as they are, and warn either way.
function _flag_absent_antennas!(ms::XRadio.MeasurementSet, win::GeometryWindow, solved; apply_flags::Bool)
    absent = [!(n in solved) for n in win.geom.antennas]
    names = Set{String}()
    flag = ms[:flag]
    for (bi, (a, b)) in pairs(win.antennas)
        (a != b && (absent[a] || absent[b])) || continue
        absent[a] && push!(names, String(win.geom.antennas[a]))
        absent[b] && push!(names, String(win.geom.antennas[b]))
        apply_flags && (view(flag, BaselineID(bi)) .= true)
    end
    isempty(names) || @warn "antennas $(sort!(collect(names))) are not in the solution; their baselines are " *
        (apply_flags ? "flagged" : "left uncorrected") maxlog = 1
    return ms
end

# The unconstrained (station name, feed, scan id) triples `sol` records,
# as a lookup set, `nothing` when it records none.
function _solution_flag_sets(sol::CalibrationSolution)
    info = sol.info
    haskey(info, :flagged_ant) || return nothing
    flagged = Set(
        (String(info.flagged_ant[i]), Int(info.flagged_feed[i]), Int(info.flagged_scan[i]))
            for i in eachindex(info.flagged_ant, info.flagged_feed, info.flagged_scan)
    )
    return isempty(flagged) ? nothing : flagged
end

# Flag the samples of each product whose (station, feed) on either side is in
# `flagged` for that scan, scans indexing the solution's geometry `solgeom`,
# matched by name.
function _flag_unconstrained!(ms::XRadio.MeasurementSet, win::GeometryWindow, solgeom::DataGeometry, flagged)
    isnothing(flagged) && return ms
    antenna = win.geom.antennas
    scan = [_solution_scan(solgeom, String(c)) for c in ms[:scan_name]]
    feeds = feed_pairs(ms)
    flag = ms[:flag]
    for (bi, (a, b)) in pairs(win.antennas)
        a == b && continue
        sa, sb = antenna[a], antenna[b]
        for ti in eachindex(scan), p in axes(feeds, 1)
            fa, fb = feeds[p, bi]
            ((sa, fa, scan[ti]) in flagged || (sb, fb, scan[ti]) in flagged) || continue
            view(flag, Polarization(p), BaselineID(bi), Ti(ti)) .= true
        end
    end
    return ms
end

function _solution_scan(solgeom::DataGeometry, name::String)
    s = findfirst(==(name), solgeom.scan_names)
    isnothing(s) && throw(
        ArgumentError(
            "scan $(repr(name)) is not in the solution, which covers " *
                "$(join(map(repr, solgeom.scan_names), ", ")); its unconstrained-antenna flags " *
                "cannot be applied. Calibrate only the scans the solution was solved on, or pass " *
                "`apply_flags = false`."
        )
    )
    return s
end

# ── The runner ───────────────────────────────────────────────────────────────

# Run one solver: compile its model against the data's geometry, then run its
# `solve` over the scan groups.
function _run_solver(solver::AbstractSolver, exec::ExecutionConfig, ps::XRadio.ProcessingSet)
    geom = DataGeometry(ps)
    spec = (; geom)
    unresolved = solver_gauge(solver)
    gauge = isnothing(unresolved) ? nothing : resolve_gauge(unresolved, geom.antennas)
    groups = groupby(ps, XRadio.ByScan())
    sizes = [_group_bytes(g) for g in values(groups)]
    ctx = _solver_context(solver, spec, gauge, groups, sizes, exec)
    t0 = time_ns()
    info = solve(solver, ctx)
    info isa NamedTuple || throw(
        ArgumentError(
            "`solve(::$(nameof(typeof(solver))), ctx)` must return a NamedTuple of " *
                "diagnostics, got a $(typeof(info))",
        ),
    )
    return CalibrationSolution(
        ctx.model, ctx.layout, geom, ctx.θ, (; _run_info(groups, exec)..., _with_timing(info, ctx, t0)...);
        name = solution_name(solver), spec = sprint(show, solver; context = :limit => true),
    )
end

# A solver compiles its own model against the run's geometry and may compile no
# components at all; it still runs, with nothing to solve. The model is resolved
# against the run's stations here, so the context holds the per-station trees
# the solve uses. A solver that does not declare
# `supports_antenna_heterogeneity` is handed uniform models only; anything else
# is rejected before any data is read.
function _solver_context(solver::AbstractSolver, spec, gauge, groups, sizes, exec)
    antennas = spec.geom.antennas
    model = Calibration.resolve(model_components(solver, spec), antennas, spec.geom)
    supports_antenna_heterogeneity(solver) ||
        Calibration.require_antenna_uniform(model, antennas, string(nameof(typeof(solver))))
    layout = plan_parameters(model, antennas, spec.geom; require_nonempty = false)
    return SolveContext(
        model, layout, spec.geom, zeros(layout.nθ), gauge, length(antennas),
        groups, sizes, exec, solution_name(solver), _PassTiming[],
    )
end

# A solver's diagnostics plus `t_pass`, the wall time of its `solve`, and
# `timing`, a `DimStack` over `Scan` of each group's `decode`/`work` seconds
# summed over the solver's passes.
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

# The run's diagnostics, which `sol.info` holds beside the solver's own.
function _run_info(groups, exec)
    return (;
        nscan = length(groups),
        ntasks_used = max_tasks(outer_executor(exec)),
        inner_tasks = max_tasks(inner_executor(exec)),
    )
end
