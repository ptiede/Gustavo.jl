# ── Pipeline verbs: fit / calibrate / fitcalibrate ───────────────────────────
#
# The three user-facing entry points of the composable pipeline:
#
#     sol      = fit(pipe, uvset)                      # solve only
#     out      = calibrate(sol, uvset; reduce = [...]) # apply + optional reduce
#     sol, out = fitcalibrate(pipe, uvset; reduce)     # fused single pass
#
# Every pipeline runs on the new engine (`_run_pipeline`): each solve step
# compiles and solves its OWN private model under the visitor contract, one
# streaming pass per step except where consecutive scan-local steps share one
# (see `fusable_grouping`), and — for the
# output verbs — a per-scan-group output tail (`reduce_scan_output`)
# that applies the solution and the reduce chain while the group is resident.
# When the final solve step is a `TemporalSmoother` the tail FUSES into its
# pass (the group is corrected and reduced right after its per-AP solve, the
# production single-read path); otherwise a dedicated output pass streams the
# same tail. `calibrate(sol, uvset)` is that output pass standalone, so fused
# and standalone outputs are identical by construction.

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
    fit(pipe::CalibrationPipeline, uvset::UVSet) -> CalibrationSolution
    fit(step_or_chain, uvset; exec = ExecutionConfig(), gauge = PinAntenna(1)) -> CalibrationSolution

Solve the pipeline's calibration on `uvset` WITHOUT producing corrected data —
the estimation half of [`fitcalibrate`](@ref). The returned solution carries
per-stage provenance (`sol[:fringe]`, `sol[:bandpass]`, `sol[:adhoc]` — each a
step alone; `sol[1:i]` for the cumulative view through step `i`,
[`stage_info`](@ref) for its diagnostics) and records
the data-transform chain the scans were materialized through
(`sol.transforms`), so diagnostics and the standalone [`calibrate`](@ref)
reproduce exactly what the solve saw.

`ReduceStep`s and [`AprioriAmplitude`](@ref) in the pipeline do not affect the
solution's θ and are ignored here (`AprioriAmplitude` is still RECORDED on the
solution — `sol.postcal` — so `calibrate(sol, uvset)` replays it); they run in
[`fitcalibrate`](@ref)'s output tail.

**A single `SolveStep` fit alone is the primitive** — the pipeline needs no
[`FringeFit`](@ref) step, and any composition of `SolveStep`s is legal,
including one standalone step (e.g. fitting a `Bandpass` alone over
data already corrected by an earlier run). A multi-step `CalibrationPipeline`
is a fusion convenience built FROM repeated single-step solves: solving
`A |> B` in one call is equivalent to `sa = fit(A, uvset)` followed by
`fit(Fringe.ApplySolution(sa[provides(A)]) |> B, uvset)` — the
SAME mechanism ([`Fringe.ApplySolution`](@ref) dividing a finished step's
gains out of the stream before the next step's pass), just run within one
call instead of across two. Selection (`sol[name]` — see the
`getindex` docstring on `CalibrationSolution`) extracts the finished step for
the cross-run form of this composition.
"""
function fit(pipe::CalibrationPipeline, uvset::UVSet)
    _check_blas_threads()
    sol, _ = _run_pipeline(
        _parse_pipeline(pipe), pipe.exec, pipe.gauge, uvset,
    )
    return sol
end

fit(
    x::Union{CalibrationStep, Fringe.AbstractDataTransform}, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(),
    gauge::AbstractGauge = PinAntenna(1),
) = fit(CalibrationPipeline(x; exec, gauge), uvset)
fit(
    chain::StepChain, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(),
    gauge::AbstractGauge = PinAntenna(1),
) = fit(CalibrationPipeline(chain; exec, gauge), uvset)

"""
    calibrate(sol::CalibrationSolution, uvset::UVSet;
              reduce = ReduceStep[], apply_flags = true, exec = ExecutionConfig()) -> UVSet

Apply a fitted solution to data, streaming one scan group at a time (a lazy set
is never fully materialized): each group is materialized through the transform
chain recorded on `sol` (so the output carries `transforms ∘ solution`, exactly
as the fused path would), corrected with the solution's gains and flags, run
through the recorded [`AprioriAmplitude`](@ref) chain (`sol.postcal`) and then
the `reduce` steps in order, and collected. This is the SAME per-group tail
[`fitcalibrate`](@ref) fuses into its final pass. Use it to calibrate the data
the solution was fit on, or another set with the same geometry. `exec` (an
[`ExecutionConfig`](@ref)) supplies this pass's schedulers, memory budget and
progress callback, exactly as it does for a [`fit`](@ref).
"""
function calibrate(
        sol::CalibrationSolution, uvset::UVSet;
        reduce = ReduceStep[], apply_flags::Bool = true,
        exec::ExecutionConfig = ExecutionConfig(),
    )
    any(t -> t === missing, sol.transforms) && throw(
        ArgumentError(
            "calibrate: this solution records a transform that did not survive serialization " *
                "(saved as `missing`) — re-fit, or apply the original transform chain manually."
        )
    )
    any(t -> t === missing, sol.postcal) && throw(
        ArgumentError(
            "calibrate: this solution records an a-priori amplitude cal that did not survive " *
                "serialization (saved as `missing`) — re-fit, or apply it manually."
        )
    )
    stream = Fringe.scan_stream(uvset; transforms = sol.transforms, exec = exec)
    post = _compose_output_chain(sol.postcal, collect(reduce))
    group_pairs = Fringe.map_groups(stream; stage = :output) do spec
        keyed = Fringe.materialize_leaves(stream, spec)
        reduce_scan_output(
            stream.uvset, keyed, sol, post;
            executor = inner_executor(stream), apply_flags = apply_flags,
        )
    end
    return assemble_output(uvset, group_pairs)
end

"""
    fitcalibrate(pipe::CalibrationPipeline, uvset::UVSet; reduce = ReduceStep[])
        -> (sol::CalibrationSolution, out::UVSet)

Solve AND produce the corrected, reduced output — the production path. The
output chain is: the solution's gains/flags, then any [`AprioriAmplitude`](@ref)
and `ReduceStep`s the pipeline declares, run in THEIR declared relative order,
then the `reduce` kwarg's (always last) — run per scan group while it is
resident. When the pipeline ends in a [`TemporalSmoother`](@ref) the whole
tail fuses into that final streaming pass (one read solves, corrects, and
reduces each group).
"""
function fitcalibrate(pipe::CalibrationPipeline, uvset::UVSet; reduce = ReduceStep[])
    _check_blas_threads()
    sol, ctx = _run_fitcalibrate(pipe, uvset, reduce)
    return sol, ctx.output
end

fitcalibrate(
    x::Union{CalibrationStep, Fringe.AbstractDataTransform}, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(),
    gauge::AbstractGauge = PinAntenna(1),
    kwargs...,
) = fitcalibrate(CalibrationPipeline(x; exec, gauge), uvset; kwargs...)
fitcalibrate(
    chain::StepChain, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(),
    gauge::AbstractGauge = PinAntenna(1),
    kwargs...,
) = fitcalibrate(CalibrationPipeline(chain; exec, gauge), uvset; kwargs...)

# Shared driver for fitcalibrate: returns (sol, ctx).
function _run_fitcalibrate(pipe::CalibrationPipeline, uvset::UVSet, reduce)
    br = _parse_pipeline(pipe)
    post = _compose_output_chain(br.post_steps, collect(reduce))
    sol, output = _run_pipeline(
        br, pipe.exec, pipe.gauge, uvset; sink = OutputSink(post),
    )
    ctx = CalibrationContext(
        uvset, sol, output,
        isempty(br.apriori) ? nothing : last(br.apriori).spw_cals,
    )
    return sol, ctx
end

# ── The output sink ──────────────────────────────────────────────────────────

"""
    reduce_scan_output(uvset::UVSet, keyed, sol::CalibrationSolution, postprocess;
                       ntasks = 1, apply_flags = true) -> Vector{Pair}

The per-scan-group output tail: rebuild the group's `(key, leaf)` pairs as a
sub-`UVSet`, apply `sol`'s gains/flags (`apply_calibration` — zero-weighting
unconstrained stations and excluded baselines exactly like a full-set apply
would), run `postprocess` (a `UVSet -> UVSet` map), and return the reduced
output branches. Every output path — the fused `fitcalibrate` tail and the
standalone `calibrate(sol, uvset)` stream — runs THIS function per group, so
fused ≡ standalone by construction and a lazy set is never fully materialized.
"""
function reduce_scan_output(
        uvset::UVSet, keyed, sol::CalibrationSolution, postprocess;
        executor = SerialScheduler(), apply_flags::Bool = true,
    )
    sub_branches = DimensionalData.TreeDict()
    for (k, leaf) in keyed
        sub_branches[k] = leaf
    end
    sub = DimensionalData.rebuild(uvset; branches = sub_branches)
    # Gains only: the streaming pass that produced `keyed` already applied
    # `sol.transforms`; the default recorded-chain replay would apply them twice.
    reduced = postprocess(UVData.apply_calibration(sub, sol; executor, apply_flags, transforms = ()))
    return collect(pairs(UVData.branches(reduced)))
end

"""
    assemble_output(uvset::UVSet, group_pairs) -> UVSet

Rebuild an output `UVSet` from the per-group branch pairs
[`reduce_scan_output`](@ref) returned (in group order).
"""
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

# The fused output tail's configuration: the composed `UVSet -> UVSet`
# postprocess (a-priori amplitude chain + reductions) applied after the
# solution's gains, per scan group.
struct OutputSink
    postprocess::Any
    apply_flags::Bool
end
OutputSink(postprocess) = OutputSink(postprocess, true)

# Compose the output chain: `declared` runs first, in ITS OWN declared
# relative order (an AprioriAmplitude and a ReduceStep interleave exactly as
# `pipe.steps` listed them — no hardcoded "apriori before reductions" rule),
# then `extra_reduces` (the `reduce` kwarg, supplied after the fact and always
# appended last) — into one `UVSet -> UVSet` function.
function _compose_output_chain(declared, extra_reduces)
    fs = Any[]
    ctx = CalibrationContext()
    for s in declared
        if s isa AprioriAmplitude
            push!(
                fs, uv -> apply_calibration(
                    uv, s.spw_cals;
                    min_elevation_deg = s.min_elevation_deg, on_missing_station = s.on_missing_station,
                )
            )
        elseif s isa ReduceStep
            f, ctx = prepare_reducer(s, ctx)
            push!(fs, f)
        else
            throw(
                ArgumentError(
                    "output chain: recorded postcal entry $(typeof(s)) is not an " *
                        "AprioriAmplitude or a ReduceStep."
                )
            )
        end
    end
    for st in extra_reduces
        st isa AprioriAmplitude && throw(
            ArgumentError(
                "fitcalibrate/calibrate: AprioriAmplitude is a pipeline step, not a reduction " *
                    "— place it in the CalibrationPipeline (it is recorded on the solution and " *
                    "replayed by `calibrate(sol, uvset)`)."
            )
        )
        st isa ReduceStep || throw(
            ArgumentError(
                "fitcalibrate/calibrate: `reduce` accepts ReduceSteps only (got $(typeof(st)))."
            )
        )
        f, ctx = prepare_reducer(st, ctx)
        push!(fs, f)
    end
    isempty(fs) && return identity
    return uv -> foldl((acc, f) -> f(acc), fs; init = uv)
end

# ── The new-engine path: the compiled model + visitor pass runner ────────────

# Solve a pipeline on the new engine: each solve step compiles and solves its
# OWN private (model, layout, θ) — never a merged one — under the visitor
# contract (start_pass!/process_scan!/finish_pass!). The steps are partitioned
# into RUNS (`_fusable_run`), one streaming pass each: a run of consecutive
# scan-local steps shares its pass, and everything else runs alone.
# Gain correction between runs flows through the scan stream's transform
# chain, not a live θ evaluation: after a run finishes, each of its steps'
# solutions (its own model/layout/θ, undivided — no other step's contribution
# to zero out) is wrapped as an `ApplySolution` and appended, so every LATER
# pass reads already-corrected data (this is solve-time-only bookkeeping — the
# returned solution still records just `br.tfs`, the caller's own precal
# chain). This is the SAME `ApplySolution`-chaining a caller does explicitly
# across separate `fit` calls via selection (`sol[name]`) — a
# multi-step pipeline just runs the append within one call instead of two, and
# within a fused run applies the identical division to the resident scan.
# Non-data info an earlier step published (e.g. the fringe stage's
# per-scan SNR) reaches a later step through the plain, ordered list of
# finished `StepSolution`s (`fit_selection`), not a shared scratch dict. With a
# `sink`, the output tail runs per group — fused into the final pass when that
# pass is a TemporalSmoother's (it never repeats and its θ writes precede the
# tail), else as a dedicated output pass after the solves. Once every step has
# solved, the runner hands the finished `StepSolution`s straight to
# `CalibrationSolution` — there is no merged model/layout/θ to assemble.
# Returns `(sol, output)` (`output === nothing` without a sink).
function _run_pipeline(
        br, exec::ExecutionConfig, gauge_spec::AbstractGauge,
        uvset::UVSet; sink = nothing,
    )
    solve_steps = br.solve_steps

    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    antennas = UVData.metadata(first_leaf).antennas
    nant = length(antennas)
    spec = (; geom, antennas)
    gauge = resolve_gauge(gauge_spec, _antenna_names(uvset))
    # A fresh `ScanStream` over the given transform list — cheap (geometry-only,
    # no data read). Used both for the initial stream and to grow the SOLVE-time
    # transform chain between steps (below): `ScanStream`/`SolveContext` fix the
    # transform-vector element type as a type parameter, so appending a
    # transform means building a new stream, not mutating one in place.
    _build_stream(tfs) = Fringe.scan_stream(uvset; geom = geom, transforms = tfs, exec = exec)
    stream = _build_stream(br.tfs)
    # Run-wide state shared, unmodified in identity, across every step's own
    # SolveContext below (only `model`/`layout`/`ev`/`θ` and `stream` change
    # per step — the rest is the run-wide part CHUNK-069 splits out).
    scratch = Dict{Symbol, Any}()
    # The final pass fuses the output tail only when it is a TemporalSmoother's:
    # that pass never repeats and finishes each group's θ before the tail runs
    # (the monolith's pass-2 structure). Anything else gets a dedicated pass.
    fused = sink !== nothing && last(solve_steps) isa TemporalSmoother
    tfs_solve = copy(br.tfs)
    step_solutions = StepSolution[]
    ctx = nothing
    # A step compiles its own model against the run's geometry; a step may
    # legitimately compile NO components at all (e.g. `DispersionSBDFit` with
    # dispersion disabled and a band layout that can't support SBD either) — it
    # still runs under the visitor contract, just with nothing to solve, so the
    # empty-model guard (meant for a whole pipeline's compiled model) doesn't
    # apply per-step. The model is materialized against the run's antenna
    # table here, so the context (and the solution's provenance) holds the
    # concrete per-station trees the solve uses; a step that has not opted
    # into station heterogeneity (`supports_station_heterogeneity`) is handed
    # uniform models only — anything else is rejected before any data is read.
    function _step_context(st, stream)
        step_model = Calibration.materialize(
            Calibration.as_gain_model(model_components(st, spec)), antennas, geom,
        )
        supports_station_heterogeneity(st) ||
            Calibration.require_station_uniform(step_model, antennas, string(nameof(typeof(st))))
        step_layout = plan_parameters(step_model, antennas, geom; require_nonempty = false)
        return SolveContext(
            step_model, step_layout, geom, GainEvaluator(step_model, step_layout),
            Calibration.component_vector(step_layout, zeros(step_layout.nθ)),
            gauge, nant, antennas, stream, scratch,
        )
    end
    si = 1
    while si <= length(solve_steps)
        # The partition is recomputed here rather than up front because
        # `fit_selection` reads the solutions finished so far.
        run = _fusable_run(solve_steps, si, step_solutions)
        si = last(run) + 1
        run_steps = solve_steps[run]
        contexts = [_step_context(st, stream) for st in run_steps]
        ctx = last(contexts)
        last_run = last(run) == length(solve_steps)
        run_sink = (fused && last_run) ? sink : nothing
        infos = length(run) == 1 ?
            [_run_pass!(run_steps[1], contexts[1], step_solutions; sink = run_sink)] :
            _run_fused_pass!(run_steps, contexts; sink = run_sink)
        for (st, c, info) in zip(run_steps, contexts, infos)
            push!(step_solutions, StepSolution(provides(st), c.model, c.layout, c.θ, info))
        end
        last_run && break
        # Gain correction to the NEXT run flows through the transform chain, not
        # a live θ evaluation: each step's own θ is already undivided (no other
        # step's contribution to zero out — each step solves its own private
        # model), so it appends directly; every later pass then reads
        # already-corrected data with no step evaluating or mutating another
        # step's θ block. Within a fused run the same corrections were applied
        # to the resident scan instead, in the same order.
        for (st, c) in zip(run_steps, contexts)
            push!(tfs_solve, Fringe.ApplySolution(_step_precal(st, c, geom)))
        end
        stream = _build_stream(tfs_solve)
    end
    sol = CalibrationSolution(
        step_solutions, geom, _new_engine_info(ctx, br, step_solutions);
        transforms = br.tfs, postcal = br.apriori,
    )
    sink === nothing && return sol, nothing
    if !fused
        # A FRESH stream over just `br.tfs` (matching `sol.transforms`), not
        # `ctx.stream` (which carries the internal, solve-time-only transform
        # chain above) — `sol`'s θ is already the complete, composed
        # correction, so this output pass must divide it out exactly once,
        # the same as the standalone `calibrate(sol, uvset)` path does.
        out_stream = _build_stream(br.tfs)
        group_pairs = Fringe.map_groups(out_stream; stage = :output) do gspec
            keyed = Fringe.materialize_leaves(out_stream, gspec)
            reduce_scan_output(
                out_stream.uvset, keyed, sol, sink.postprocess;
                executor = inner_executor(out_stream), apply_flags = sink.apply_flags,
            )
        end
        return sol, assemble_output(uvset, group_pairs)
    end
    return sol, assemble_output(uvset, ctx.scratch[:sink_pairs])
end

# The per-scan SNR a LATER step's selection may want (e.g. a `ScanWhere`
# predicate filtering on `s.snr`), read off the most recent finished
# `StepSolution` that published one — never a shared scratch dict
# (CHUNK-067c). `nothing` when no prior step published SNR (an estimator with
# no notion of it, or none yet).
function _scan_snr(prior_solutions)
    for s in Iterators.reverse(prior_solutions)
        haskey(s.info, :scan_snr) && return s.info.scan_snr
    end
    return nothing
end

# ── Pass partitioning: which steps share one read of the data ────────────────

# The maximal run of steps starting at `first_i` that can share ONE streaming
# pass. A run extends while each further step declares itself scan-local
# (`fusable_grouping === :scan`) AND accumulates from every scan: one pass
# materializes one set of groups, so steps disagreeing on their `fit_selection`
# need passes of their own. `AllScans` is required rather than mere agreement
# because a fused step also cannot read its run siblings' `StepSolution`s —
# they do not exist until the run ends — and a narrower selection is exactly
# where a step would want them.
#
# The run's FIRST step needs neither property, so every pipeline partitions
# into runs: a `:global` step, or one fitting on a subset, simply lands in a
# run of its own and gets a pass to itself.
function _fusable_run(solve_steps, first_i::Integer, prior_solutions)
    _fusable(st) = fusable_grouping(st) === :scan &&
        fit_selection(st, prior_solutions) isa Fringe.AllScans
    _fusable(solve_steps[first_i]) || return first_i:first_i
    last_i = first_i
    while last_i < length(solve_steps) && _fusable(solve_steps[last_i + 1])
        last_i += 1
    end
    return first_i:last_i
end

# One streaming pass per solve step: materialize each selected group, hand its
# stack and geometry window to `process_scan!`, collect the per-group returns IN GROUP-INDEX ORDER
# into `ctx.scratch[:pass_results]`, then `finish_pass!` (repeating the pass
# while it returns `repeat_pass = true` — residual re-search rounds). With a
# `sink` (the fused fitcalibrate tail) the group is materialized as LEAVES
# first — the output must preserve the per-band leaf structure — the solve cube
# is stacked from them, and after `process_scan!` finishes the group's θ the
# tail corrects and reduces those same leaves in place. Returns the step's diagnostics
# NamedTuple (`repeat_pass` stripped, `t_pass`/`timing` added — see below) —
# the caller wraps it into a `StepSolution` alongside this step's own
# `ctx.model`/`layout`/`θ`.
function _run_pass!(step::SolveStep, ctx::SolveContext, prior_solutions; sink = nothing)
    stage = provides(step)
    ngroups = length(ctx.stream.groups)
    t0 = time_ns()
    while true
        start_pass!(step, ctx)
        flag_nt = sink === nothing ? nothing :
            Fringe.flag_table(_fringe_flags(ctx))
        results = Fringe.map_groups(
            ctx.stream; selection = fit_selection(step, prior_solutions),
            snr = _scan_snr(prior_solutions), stage = stage,
        ) do gspec
            ta = time_ns()
            if sink === nothing
                stack, win = Fringe.materialize_cube(ctx.stream, gspec)
                keyed = nothing
            else
                keyed = Fringe.materialize_leaves(ctx.stream, gspec)
                stack, win = Streaming._stacked_scan_group(
                    [m for (_, m) in keyed], ctx.stream.geom,
                )
            end
            tb = time_ns()
            r = process_scan!(step, ctx, stack, win)
            tc = time_ns()
            out = nothing
            if sink !== nothing
                # `keyed` was materialized through `ctx.stream`'s transform
                # chain, which already carries every EARLIER step's finished
                # solution (see `_run_pipeline`) — so the group-local
                # solution here needs only THIS step's own (model, layout, θ),
                # undivided, on top of it.
                sol_local = CalibrationSolution(ctx.model, ctx.layout, ctx.geom, ctx.θ, flag_nt; name = stage)
                out = reduce_scan_output(
                    ctx.stream.uvset, keyed, sol_local, sink.postprocess;
                    executor = inner_executor(ctx.stream), apply_flags = sink.apply_flags,
                )
            end
            (;
                index = gspec.index, decode = (tb - ta) / 1.0e9,
                work = (tc - tb) / 1.0e9, reduce = (time_ns() - tc) / 1.0e9, r, out,
            )
        end
        ctx.scratch[:pass_results] = results
        sink === nothing || (ctx.scratch[:sink_pairs] = [res.out for res in results])
        # NOT named `out`: that would be the same binding the per-group closure
        # above writes, so every concurrent group would share one box.
        info = finish_pass!(step, ctx)
        get(info, :repeat_pass, false) || return _pass_diagnostics(info, results, ngroups, t0)
    end
    return
end

# ONE streaming pass shared by a run of scan-local steps (see
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
# charged to the first step, each step's own `process_scan!` to itself, and the
# output tail to the last.
#
# Every step in the run selects every scan (`_fusable_run`), which is why the
# pass takes the default selection and needs no per-scan SNR from the steps
# solved before it.
function _run_fused_pass!(steps, contexts; sink = nothing)
    n = length(steps)
    stream = first(contexts).stream
    ngroups = length(stream.groups)
    t0 = time_ns()
    for (st, ctx) in zip(steps, contexts)
        start_pass!(st, ctx)
    end
    ctx_n = last(contexts)
    # Flags already finished by EARLIER passes (shared scratch). A `:scan` step
    # in THIS run finishes each scan's flags inside `process_scan!`
    # (`scan_flags`), never pass-wide — the tail collects them per group below.
    base_flags = sink === nothing ? nothing : _fringe_flags(ctx_n)
    results = Fringe.map_groups(stream; stage = provides(last(steps))) do gspec
        ta = time_ns()
        if sink === nothing
            stack, win = Fringe.materialize_cube(stream, gspec)
            keyed = nothing
        else
            keyed = _private_leaves(stream, gspec)
            stack, win = Streaming._stacked_scan_group([m for (_, m) in keyed], stream.geom)
        end
        tb = time_ns()
        work = zeros(Float64, n)
        rs = Vector{Any}(undef, n)
        for k in 1:n
            tk = time_ns()
            rs[k] = process_scan!(steps[k], contexts[k], stack, win)
            if k < n
                # `_stacked_scan_group` copied the leaves' data into the cube, so
                # the solve stack and the output leaves are separate arrays and
                # each needs the correction applied to it.
                as = Fringe.ApplySolution(_step_precal(steps[k], contexts[k], contexts[k].geom))
                apply_transform!(as, stack, win; executor = inner_executor(stream))
                if keyed !== nothing
                    for (_, m) in keyed
                        apply_transform!(
                            as, m[(:vis, :weights)], leaf_window(stream.geom, m);
                            executor = SerialScheduler(),
                        )
                    end
                end
            end
            work[k] = (time_ns() - tk) / 1.0e9
        end
        tc = time_ns()
        out = nothing
        if sink !== nothing
            # Every earlier step of the run has already been divided out of
            # `keyed` above, exactly as an un-fused pass's transform chain would
            # have, so the tail's group-local solution carries only the LAST
            # step's own (model, layout, θ).
            flags = copy(base_flags)
            for k in 1:n
                append!(flags, scan_flags(steps[k], rs[k]))
            end
            sol_local = CalibrationSolution(
                ctx_n.model, ctx_n.layout, ctx_n.geom, ctx_n.θ, Fringe.flag_table(flags);
                name = provides(last(steps)),
            )
            out = reduce_scan_output(
                stream.uvset, keyed, sol_local, sink.postprocess;
                executor = inner_executor(stream), apply_flags = sink.apply_flags,
            )
        end
        (;
            index = gspec.index, decode = (tb - ta) / 1.0e9, work,
            reduce = (time_ns() - tc) / 1.0e9, rs, out,
        )
    end
    sink === nothing || (ctx_n.scratch[:sink_pairs] = [res.out for res in results])
    infos = NamedTuple[]
    for k in 1:n
        ctx = contexts[k]
        # Each step sees the pass results in the shape a step of its own always
        # gets: its OWN per-scan return under `r`, and the share of the group's
        # cost it caused.
        ctx.scratch[:pass_results] = [
            (;
                res.index, decode = k == 1 ? res.decode : 0.0, work = res.work[k],
                reduce = k == n ? res.reduce : 0.0, r = res.rs[k], res.out,
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

# A scan group's materialized leaves, guaranteed writable: a fused run divides
# each step's gains out of them in place, and `materialize_leaves` can hand back
# an EAGER source's own arrays. A leaf sharing its array with the set's is
# rewrapped around copies — the caller's `UVSet` is never mutated. Tested by
# array identity rather than by re-deriving when the copy already happened, so
# no assumption about the materialization path is carried here.
function _private_leaves(stream, gspec)
    keyed = Fringe.materialize_leaves(stream, gspec)
    return [
        parent(m[:vis]) === parent(src[:vis]) ?
            (k, UVData.rebuild_visibilities(m, copy(parent(m[:vis])), copy(parent(m[:weights])))) :
            (k, m)
            for ((_, src), (k, m)) in zip(gspec.leaves, keyed)
    ]
end

# Every step's diagnostics, whatever `finish_pass!` chose to return
# (`repeat_pass` stripped), plus TWO fields the runner computes generically for
# ANY step from data `map_groups` already collected — no per-step code, no
# per-step-name knowledge: `t_pass` (total pass wall time) and `timing` (a
# `DimStack` over `Scan`, one row per selected group, `decode`/`work`/`reduce`
# task-seconds, built with `scan_values` — the same primitive a custom step
# uses for its own per-scan diagnostics). A third-party `SolveStep` gets both
# automatically.
function _pass_diagnostics(out, results, ngroups::Integer, t0::UInt64)
    timing = DimensionalData.DimStack(
        (;
            decode = scan_values(res -> res.decode, results, ngroups; default = 0.0),
            work = scan_values(res -> res.work, results, ngroups; default = 0.0),
            reduce = scan_values(res -> res.reduce, results, ngroups; default = 0.0),
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
# `br.ff` is `nothing` for a FringeFit-less pipeline, so the estimator info is
# omitted rather than assumed present.
function _new_engine_info(ctx::SolveContext, br, step_solutions::Vector{StepSolution})
    return (;
        nant = ctx.nant,
        nscan = length(ctx.stream.groups),
        Fringe.flag_table(_fringe_flags(ctx))...,
        ant_names = String.(collect(ctx.antennas.name)),
        (br.ff === nothing ? NamedTuple() : Fringe.estimator_info(br.ff.estimator))...,
        precal_applied = any(t -> t isa Fringe.ApplySolution, br.tfs),
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
                    "fit/fitcalibrate: more than one step provides :$p — exactly one is " *
                        "allowed per pipeline."
                )
            )
            push!(provided, p)
        end
    end
    return nothing
end

# Parse a pipeline into (transforms, the FringeFit if present, every SolveStep
# in declared order, and the output-tail steps — AprioriAmplitude and
# ReduceStep — in their OWN declared relative order) — no per-step-type branch
# to maintain as new `SolveStep`s appear. A pipeline needs no FringeFit step —
# `ff` is `nothing` when none is declared; a step that needs an earlier
# FringeFit's correction and does not have one fails from its own solve
# kernel, not from pipeline construction.
function _parse_pipeline(pipe::CalibrationPipeline)
    tfs = Fringe.AbstractDataTransform[]
    solve_steps = SolveStep[]
    post_steps = Union{AprioriAmplitude, ReduceStep}[]
    for s in pipe.steps
        if s isa DataTransformStep
            push!(tfs, s.t)
        elseif s isa SolveStep
            push!(solve_steps, s)
        elseif s isa Union{AprioriAmplitude, ReduceStep}
            push!(post_steps, s)
        else
            throw(
                ArgumentError(
                    "fit/fitcalibrate: step $(typeof(s)) is not runnable — supported: data " *
                        "transforms, SolveSteps (FringeFit, DispersionSBDFit, " *
                        "Bandpass, TemporalSmoother, or a third-party SolveStep), " *
                        "AprioriAmplitude, and ReduceSteps."
                )
            )
        end
    end
    _check_unique_provides(solve_steps)
    ffi = findfirst(s -> s isa FringeFit, solve_steps)
    apriori = filter(s -> s isa AprioriAmplitude, post_steps)
    return (; tfs, ff = ffi === nothing ? nothing : solve_steps[ffi], solve_steps, apriori, post_steps)
end
