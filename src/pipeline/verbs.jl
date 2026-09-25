# ── Pipeline verbs: fit / calibrate ──────────────────────────────────────────
#
#     sol = fit(pipeline, uvset; gauge)               # solve
#     out = calibrate(sol, uvset; post)               # apply, one scan group at a time
#
# `fit` runs each solve step in its own streaming pass, except where
# consecutive scan-local steps share one (see `fusable_grouping`).
# `calibrate` streams the recorded corrections, the gains and flags, and the
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
    group_pairs = Fring.map_groups(stream; stage = :output) do spec
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

# ── The new-engine path: the compiled model + visitor pass runner ────────────

_resolve_run_gauge(g::AbstractGauge, ant_names) = resolve_gauge(g, ant_names)
_resolve_run_gauge(::Nothing, ant_names) = throw(
    ArgumentError(
        "no gauge given; pass `gauge = PinAntenna(station)` or `gauge = ZeroSumPhase()`. " *
            "Stations: $(join(ant_names, ", ")).",
    ),
)

# Solve a pipeline: each solve step compiles and solves its own private
# (model, layout, θ) — never a merged one — under the visitor contract
# (start_pass!/process_scan!/finish_pass!). The steps are partitioned into
# RUNS (`_fusable_run`), one streaming pass each: a run of consecutive
# scan-local steps with no transform between them shares its pass, and
# everything else runs alone. A step reads the data through the transform
# chain as it stands at the step's position: the transforms listed before it,
# in order, with each earlier step's own solution (`ApplySolution`) appended
# where the step finished. Non-data info an earlier step published (e.g. the
# fringe stage's per-scan SNR) reaches a later step through the ordered list
# of finished `StepSolution`s, not a shared scratch dict.
function _run_pipeline(
        br, exec::ExecutionConfig, gauge_spec::Union{Nothing, AbstractGauge}, uvset::UVSet,
    )
    solve_steps = br.solve_steps

    # A solve has ONE station axis, and a leaf's baseline pairs index that
    # leaf's own antenna table — so leaves that saw different sub-arrays number
    # the same station differently. Put them all on the union table first; a set
    # whose leaves already share one is returned untouched and stays lazy.
    uvset = UVData.unify_antennas(uvset)
    gauge = _resolve_run_gauge(gauge_spec, _antenna_names(uvset))
    geom = build_geometry(uvset)
    antennas = UVData.union_antennas(uvset)
    nant = length(antennas)
    spec = (; geom, antennas)
    # `ScanStream`/`SolveContext` fix the transform-vector element type as a
    # type parameter, so growing the chain means building a new stream (cheap:
    # geometry only, no data read).
    _build_stream(tfs) = Fring.scan_stream(uvset; geom = geom, transforms = tfs, exec = exec)
    # Run-wide state shared across every step's own SolveContext.
    scratch = Dict{Symbol, Any}()
    chain = Fring.AbstractDataTransform[]
    step_solutions = StepSolution[]
    ctx = nothing
    # A step compiles its own model against the run's geometry, and may
    # legitimately compile no components at all (e.g. `DispersionSBDFit` with
    # dispersion disabled and a band layout that can't support SBD either); it
    # still runs under the visitor contract, with nothing to solve. The model is
    # materialized against the run's antenna table here, so the context (and
    # the solution's provenance) holds the concrete per-station trees the solve
    # uses; a step that has not opted into station heterogeneity
    # (`supports_station_heterogeneity`) is handed uniform models only —
    # anything else is rejected before any data is read.
    function _step_context(st, stream)
        step_model = Calibration.materialize(model_components(st, spec), antennas, geom)
        supports_station_heterogeneity(st) ||
            Calibration.require_station_uniform(step_model, antennas, heterogeneity_rejector(st))
        step_layout = plan_parameters(step_model, antennas, geom; require_nonempty = false)
        return SolveContext(
            step_model, step_layout, geom, zeros(step_layout.nθ),
            gauge, nant, antennas, stream, scratch,
        )
    end
    si = 1
    while si <= length(solve_steps)
        append!(chain, br.before[si])
        # A transform between two steps ends the run: the later step must read
        # its output.
        run = _fusable_run(solve_steps, si)
        stop = findfirst(k -> !isempty(br.before[k]), (si + 1):last(run))
        run = stop === nothing ? run : si:(si + stop - 1)
        si = last(run) + 1
        stream = _build_stream(copy(chain))
        run_steps = solve_steps[run]
        contexts = [_step_context(st, stream) for st in run_steps]
        ctx = last(contexts)
        infos = length(run) == 1 ?
            [_run_pass!(run_steps[1], contexts[1])] :
            _run_fused_pass!(run_steps, contexts)
        for (st, c, info) in zip(run_steps, contexts, infos)
            push!(step_solutions, StepSolution(provides(st), c.model, c.layout, c.θ, info))
        end
        # Each step's own θ is undivided (it solves its own private model), so
        # it joins the chain as-is and every later pass reads corrected data.
        # Within a fused run the same corrections were applied to the
        # resident scan instead, in the same order.
        for (st, c) in zip(run_steps, contexts)
            push!(chain, Fring.ApplySolution(_step_precal(st, c, geom)))
        end
    end
    return CalibrationSolution(
        step_solutions, geom, _new_engine_info(ctx, br);
        sequence = Tuple(br.sequence), gauge = gauge_spec,
    )
end

# ── Pass partitioning: which steps share one read of the data ────────────────

# The maximal run of steps starting at `first_i` that can share one streaming
# pass: consecutive steps that each declare themselves scan-local
# (`fusable_grouping === :scan`). The run's first step need not be, so every
# pipeline partitions into runs: a `:global` step simply lands in a run of its
# own and gets a pass to itself.
function _fusable_run(solve_steps, first_i::Integer)
    _fusable(st) = fusable_grouping(st) === :scan
    _fusable(solve_steps[first_i]) || return first_i:first_i
    last_i = first_i
    while last_i < length(solve_steps) && _fusable(solve_steps[last_i + 1])
        last_i += 1
    end
    return first_i:last_i
end

# One streaming pass per solve step: materialize each selected group, hand its
# stack and geometry window to `process_scan!`, collect the per-group returns IN
# group-INDEX ORDER into `ctx.scratch[:pass_results]`, then `finish_pass!`
# (repeating the pass while it returns `repeat_pass = true` — residual
# re-search rounds). Returns the step's diagnostics NamedTuple (`repeat_pass`
# stripped, `t_pass`/`timing` added — see below) — the caller wraps it into a
# `StepSolution` alongside this step's own `ctx.model`/`layout`/`θ`.
function _run_pass!(step::SolveStep, ctx::SolveContext)
    stage = provides(step)
    ngroups = length(ctx.stream.groups)
    t0 = time_ns()
    while true
        start_pass!(step, ctx)
        results = Fring.map_groups(ctx.stream; stage) do gspec
            ta = time_ns()
            stack, win = Fring.materialize_cube(ctx.stream, gspec)
            tb = time_ns()
            r = process_scan!(step, ctx, stack, win)
            (; index = gspec.index, decode = (tb - ta) / 1.0e9, work = (time_ns() - tb) / 1.0e9, r)
        end
        ctx.scratch[:pass_results] = results
        info = finish_pass!(step, ctx)
        get(info, :repeat_pass, false) || return _pass_diagnostics(info, results, ngroups, t0)
    end
    return
end

# One streaming pass shared by a run of scan-local steps (see
# `fusable_grouping` / `_fusable_run`): the group is materialized once, then
# each step's `process_scan!` runs on it in declared order, and every step but
# the last has its just-solved gains divided out of the resident scan before
# the next one reads it. That division is the same `ApplySolution` the
# transform chain applies between un-fused passes, on the same values in the
# same order — so the run's θ is what separate passes would have produced,
# reading and decoding the data once instead of once per step. Returns one
# diagnostics NamedTuple per step, in step order.
#
# `t_pass` runs from the start of the shared pass, so each step reports the
# run's elapsed time through its own `finish_pass!`; the per-scan `timing`
# splits the group's cost the way it was actually incurred — the single decode
# charged to the first step and each step's own `process_scan!` to itself.
#
function _run_fused_pass!(steps, contexts)
    n = length(steps)
    stream = first(contexts).stream
    ngroups = length(stream.groups)
    t0 = time_ns()
    for (st, ctx) in zip(steps, contexts)
        start_pass!(st, ctx)
    end
    results = Fring.map_groups(stream; stage = provides(last(steps))) do gspec
        ta = time_ns()
        stack, win = Fring.materialize_cube(stream, gspec)
        tb = time_ns()
        work = zeros(Float64, n)
        rs = Vector{Any}(undef, n)
        for k in eachindex(steps, contexts)
            tk = time_ns()
            rs[k] = process_scan!(steps[k], contexts[k], stack, win)
            if k < n
                as = Fring.ApplySolution(_step_precal(steps[k], contexts[k], contexts[k].geom))
                apply_transform!(as, stack, win; executor = inner_executor(stream))
            end
            work[k] = (time_ns() - tk) / 1.0e9
        end
        (; index = gspec.index, decode = (tb - ta) / 1.0e9, work, rs)
    end
    infos = NamedTuple[]
    for k in eachindex(steps, contexts)
        ctx = contexts[k]
        # Each step sees the pass results in the shape a step of its own always
        # gets: its own per-scan return under `r`, and the share of the group's
        # cost it caused.
        ctx.scratch[:pass_results] = [
            (;
                res.index, decode = k == 1 ? res.decode : 0.0, work = res.work[k], r = res.rs[k],
            ) for res in results
        ]
        info = finish_pass!(steps[k], ctx)
        get(info, :repeat_pass, false) && throw(
            ArgumentError(
                "$(nameof(typeof(steps[k]))) declares `fusable_grouping` = :scan but its " *
                    "`finish_pass!` requested `repeat_pass` — a step that needs the pass run " *
                    "again is not scan-local. Declare :global."
            )
        )
        push!(infos, _pass_diagnostics(info, ctx.scratch[:pass_results], ngroups, t0))
    end
    return infos
end

# Every step's diagnostics, whatever `finish_pass!` chose to return
# (`repeat_pass` stripped), plus TWO fields the runner computes generically for
# Any step from data `map_groups` already collected — no per-step code, no
# per-step-name knowledge: `t_pass` (total pass wall time) and `timing` (a
# `DimStack` over `Scan`, one row per selected group, `decode`/`work`
# task-seconds, built with `scan_values` — the same primitive a custom step
# uses for its own per-scan diagnostics). A third-party `SolveStep` gets both
# automatically.
function _pass_diagnostics(out, results, ngroups::Integer, t0::UInt64)
    timing = DimensionalData.DimStack(
        (;
            decode = scan_values(res -> res.decode, results, ngroups; default = 0.0),
            work = scan_values(res -> res.work, results, ngroups; default = 0.0),
        ),
        (UVData.Scan(1:ngroups),),
    )
    return (;
        (k => v for (k, v) in pairs(out) if k !== :repeat_pass)...,
        t_pass = (time_ns() - t0) / 1.0e9, timing = timing,
    )
end

# One finished step's own solution, as the precal a later pass divides out.
# `ApplySolution` matches stations by NAME and refuses a solution that names
# none, so the station table is recorded here as it is on the run's own
# solution — a solve-time transform is no exception to the apply contract.
_step_precal(st, c::SolveContext, geom::DataGeometry) = CalibrationSolution(
    c.model, c.layout, geom, c.θ, (; ant_names = String.(collect(c.antennas.name)));
    name = provides(st),
)

# The solution-level `info` NamedTuple of a new-engine fit: RUN-WIDE fields
# only. Every per-step diagnostic (per-scan SNR/detections, pass timing, …) now
# lives on that step's own `StepSolution.info` instead (`stage_info(sol,
# name)`) — published generically by `_run_pass!` (`t_pass`/`timing`, any
# step) or by the step itself (e.g. the fringe estimator's `scan_snr`,
# detection table). This function no longer needs to know any step's name to
# expose its diagnostics; a third-party `SolveStep` needs no changes here.
# `br.ff` is `nothing` for a BaselineFringeFit-less pipeline, so the estimator info is
# omitted rather than assumed present.
function _new_engine_info(ctx::SolveContext, br)
    return (;
        nant = ctx.nant,
        nscan = length(ctx.stream.groups),
        Fring.flag_table(_fringe_flags(ctx))...,
        ant_names = String.(collect(ctx.antennas.name)),
        (br.ff === nothing ? NamedTuple() : Fring.estimator_info(br.ff.estimator))...,
        precal_applied = any(t -> t isa Fring.ApplySolution, Iterators.flatten(br.before)),
        ntasks_used = Streaming.max_tasks(outer_executor(ctx.stream)),
        inner_tasks = Streaming.max_tasks(inner_executor(ctx.stream)),
    )
end

# The stage-B-unconstrained (station, scan) pairs an estimator reported, or none.
_fringe_flags(ctx::SolveContext) = get(() -> Tuple{Int, Int}[], ctx.scratch, :fringe_flags)

# ── Pipeline parsing ─────────────────────────────────────────────────────────

# Step order is never validated here — a step that cannot do its job with the
# data it is handed fails at the point of use (its own estimator/solve
# kernel), the same pattern `apply_calibration`'s data-level guards use. Only
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
    ffi = findfirst(s -> s isa BaselineFringeFit, solve_steps)
    return (; sequence, solve_steps, before, ff = ffi === nothing ? nothing : solve_steps[ffi])
end
