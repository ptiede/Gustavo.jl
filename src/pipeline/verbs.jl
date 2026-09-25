# ── Pipeline verbs: fit / calibrate ──────────────────────────────────────────
#
#     sol = fit(pipeline, uvset; gauge)               # solve
#     out = calibrate(sol, uvset; post)               # apply, one scan group at a time
#
# `fit` runs each solve step's `solve` on the data as the corrections and steps
# before it leave it. `calibrate` streams the recorded corrections, the gains and flags, and the
# output steps over each scan group, then `post`.

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
    fit(pipeline, uvset; gauge, exec = ExecutionConfig()) -> CalibrationSolution

Solve the pipeline's solve steps on `uvset`, in order. `pipeline` is a tuple
(or vector) of solve steps ([`BaselineFringeFit`](@ref), [`DispersionSBDFit`](@ref),
[`Bandpass`](@ref), [`AdhocPhase`](@ref) or a third-party `SolveStep`), data
transforms and [`AprioriAmplitude`](@ref), usually built with `|>`, or a
single one of them. Each solve step reads the data every earlier transform
and step has corrected; an `AprioriAmplitude` scales only the output.

`gauge` is required: an [`AbstractGauge`](@ref) such as `PinAntenna("PT")`,
`PinAntenna(["PT", "LA"])` for a ranked fallback, or `ZeroSumPhase()`; without
one `fit` throws, naming the stations. `exec` (an [`ExecutionConfig`](@ref))
supplies the run's schedulers, memory budget and progress callback.

The solution records the pipeline as a tuple (`sol.sequence`) and the gauge
(`sol.gauge`), so `fit(sol.sequence, uvset; sol.gauge)` repeats the run and
[`calibrate`](@ref)`(sol, uvset)` replays it. `sol[:fringe]` selects one step,
`sol[1:i]` the cumulative view through step `i`, and [`stage_info`](@ref) a
step's diagnostics. A pipeline needs no `BaselineFringeFit`: solving `A |> B`
is equivalent to `sa = fit(A, uvset)` followed by
`fit(Fring.ApplySolution(sa[provides(A)]) |> B, uvset)`.
"""
function StatsAPI.fit(
        pipeline::Union{Tuple, AbstractVector}, uvset::UVSet;
        gauge::Union{Nothing, AbstractGauge} = nothing,
        exec::ExecutionConfig = ExecutionConfig(),
    )
    _check_blas_threads()
    return _run_pipeline(_parse_pipeline(pipeline), exec, gauge, uvset)
end

StatsAPI.fit(x::PipelineElement, uvset::UVSet; kwargs...) = fit((x,), uvset; kwargs...)

"""
    calibrate(sol::CalibrationSolution, uvset::UVSet;
              post = identity, apply_flags = true, exec = ExecutionConfig()) -> UVSet

Apply a fitted solution to data, one scan group at a time (a lazy set is
never fully materialized). Each group passes through the transforms recorded
in `sol.sequence`, in order, then the solution's gains and flags, then any
recorded [`AprioriAmplitude`](@ref), then `post`, a `UVSet -> UVSet`
function such as `CombineSpw() ∘ AverageFrequency(nout = 1)`. Use it on the
data the solution was fit on, or another set with the same geometry. `exec`
supplies this pass's schedulers, memory budget and progress callback.
"""
function calibrate(
        sol::CalibrationSolution, uvset::UVSet;
        post = identity, apply_flags::Bool = true,
        exec::ExecutionConfig = ExecutionConfig(),
    )
    any(ismissing, sol.sequence) && throw(
        ArgumentError(
            "calibrate: this solution records a pipeline element that did not survive " *
                "serialization (saved as `missing`) — re-fit, or apply it manually."
        )
    )
    # Same station-axis reason as `fit`: leaves that saw different sub-arrays
    # number stations differently, and the gains are applied against one axis.
    uvset = UVData.unify_antennas(uvset)
    stream = Fring.scan_stream(uvset; transforms = recorded_transforms(sol), exec = exec)
    apriori = filter(x -> x isa AprioriAmplitude, sol.sequence)
    output = uv -> post(foldl((acc, a) -> _apply_apriori(a, acc), apriori; init = uv))
    group_pairs = _map_groups(stream; stage = :output) do spec
        keyed = Fring.materialize_leaves(stream, spec)
        reduce_scan_output(
            stream.uvset, keyed, sol, output;
            executor = inner_executor(stream), apply_flags = apply_flags,
        )
    end
    return assemble_output(uvset, group_pairs)
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

# One scan group's output: the materialized leaves `keyed` (already through the
# recorded transforms) as a sub-`UVSet`, with `sol`'s gains and flags applied
# and then `postprocess`, returned as output branches.
function reduce_scan_output(
        uvset::UVSet, keyed, sol::CalibrationSolution, postprocess;
        executor = SerialScheduler(), apply_flags::Bool = true,
    )
    sub_branches = DimensionalData.TreeDict()
    for (k, leaf) in keyed
        sub_branches[k] = leaf
    end
    sub = DimensionalData.rebuild(uvset; branches = sub_branches)
    reduced = postprocess(UVData.apply_calibration(sub, sol; executor, apply_flags, transforms = ()))
    return collect(pairs(UVData.branches(reduced)))
end

# The output `UVSet` rebuilt from the per-group branch pairs, in group order.
function assemble_output(uvset::UVSet, group_pairs)
    out_branches = DimensionalData.TreeDict()
    for r in group_pairs
        r === nothing && continue
        for (k, leaf) in r
            out_branches[k] = leaf
        end
    end
    return DimensionalData.rebuild(uvset; branches = out_branches)
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
# transform chain as it stands at the step's position: the transforms listed
# before it, in order, with each earlier step's own solution (`ApplySolution`)
# appended where the step finished.
function _run_pipeline(
        br, exec::ExecutionConfig, gauge_spec::Union{Nothing, AbstractGauge}, uvset::UVSet,
    )
    # A solve has ONE station axis, and a leaf's baseline pairs index that
    # leaf's own antenna table — so leaves that saw different sub-arrays number
    # the same station differently. Put them all on the union table first; a set
    # whose leaves already share one is returned untouched and stays lazy.
    uvset = UVData.unify_antennas(uvset)
    gauge = _resolve_run_gauge(gauge_spec, _antenna_names(uvset))
    geom = build_geometry(uvset)
    antennas = UVData.union_antennas(uvset)
    spec = (; geom, antennas)
    chain = Fring.AbstractDataTransform[]
    step_solutions = StepSolution[]
    stream = nothing
    for (st, before) in zip(br.solve_steps, br.before)
        append!(chain, before)
        # The stream fixes its transform types as a type parameter, so a longer
        # chain means a new stream (geometry only, no data read).
        stream = Fring.scan_stream(uvset; geom, transforms = copy(chain), exec)
        ctx = _step_context(st, spec, gauge, stream)
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
        # joins the chain as-is and every later step reads corrected data.
        push!(chain, Fring.ApplySolution(_step_precal(st, ctx, geom)))
    end
    return CalibrationSolution(
        step_solutions, geom, _run_info(step_solutions, br, antennas, stream);
        sequence = Tuple(br.sequence), gauge = gauge_spec,
    )
end

# A step compiles its own model against the run's geometry, and may
# legitimately compile no components at all (e.g. `DispersionSBDFit` with
# dispersion disabled and a band layout that can't support SBD either); it
# still runs, with nothing to solve. The model is materialized against the
# run's antenna table here, so the context (and the solution's provenance)
# holds the concrete per-station trees the solve uses; a step that has not
# opted into station heterogeneity (`supports_station_heterogeneity`) is handed
# uniform models only — anything else is rejected before any data is read.
function _step_context(st::SolveStep, spec, gauge, stream)
    (; geom, antennas) = spec
    model = Calibration.materialize(model_components(st, spec), antennas, geom)
    supports_station_heterogeneity(st) ||
        Calibration.require_station_uniform(model, antennas, heterogeneity_rejector(st))
    layout = plan_parameters(model, antennas, geom; require_nonempty = false)
    return SolveContext(
        model, layout, geom, zeros(layout.nθ), gauge, length(antennas), antennas,
        stream, provides(st), _PassTiming[],
    )
end

# A step's diagnostics plus `t_pass`, the wall time of its `solve`, and
# `timing`, a `DimStack` over `Scan` of each group's `decode`/`work` seconds
# summed over the step's passes.
function _with_timing(info::NamedTuple, ctx::SolveContext, t0::UInt64)
    ngroups = length(ctx.stream.groups)
    decode, work = zeros(ngroups), zeros(ngroups)
    for pass in ctx.passes
        decode .+= pass.decode
        work .+= pass.work
    end
    timing = DimensionalData.DimStack((; decode, work), (UVData.Scan(1:ngroups),))
    return (; info..., t_pass = (time_ns() - t0) / 1.0e9, timing)
end

# One finished step's own solution, as the precal a later step divides out.
# `ApplySolution` matches stations by NAME and refuses a solution that names
# none, so the station table is recorded here as it is on the run's own
# solution — a solve-time transform is no exception to the apply contract.
_step_precal(st, c::SolveContext, geom::DataGeometry) = CalibrationSolution(
    c.model, c.layout, geom, c.θ, (; ant_names = String.(collect(c.antennas.name)));
    name = provides(st),
)

# The solution-level `info`: run-wide fields, plus the fringe step's
# unconstrained (station, scan) flags, which `calibrate` applies, and its
# search configuration, which `fringe_search_map` replays. Every other per-step
# diagnostic lives on that step's `StepSolution.info` (`stage_info(sol, name)`).
function _run_info(step_solutions, br, antennas, stream)
    ffi = findfirst(st -> st isa BaselineFringeFit, br.solve_steps)
    ff_info = ffi === nothing ? nothing : step_solutions[ffi].info
    return (;
        nant = length(antennas),
        nscan = length(stream.groups),
        flagged_ant = ff_info === nothing ? Int[] : ff_info.flagged_ant,
        flagged_scan = ff_info === nothing ? Int[] : ff_info.flagged_scan,
        ant_names = String.(collect(antennas.name)),
        (ffi === nothing ? (;) : (; search = br.solve_steps[ffi].search))...,
        precal_applied = any(t -> t isa Fring.ApplySolution, Iterators.flatten(br.before)),
        ntasks_used = Streaming.max_tasks(outer_executor(stream)),
        inner_tasks = Streaming.max_tasks(inner_executor(stream)),
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

# Split a pipeline into its solve steps and, for each, the transforms listed
# between it and the step before it (`before[k]`).
function _parse_pipeline(pipeline)
    sequence = collect(Any, _check_pipeline(pipeline))
    solve_steps = SolveStep[]
    before = Vector{Fring.AbstractDataTransform}[]
    pending = Fring.AbstractDataTransform[]
    for x in sequence
        if x isa SolveStep
            push!(solve_steps, x)
            push!(before, pending)
            pending = Fring.AbstractDataTransform[]
        elseif x isa Fring.AbstractDataTransform
            push!(pending, x)
        end
    end
    isempty(solve_steps) && throw(ArgumentError("fit: the pipeline holds no solve step"))
    _check_unique_provides(solve_steps)
    return (; sequence, solve_steps, before)
end
