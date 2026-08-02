# ── Pipeline verbs: fit / calibrate / fitcalibrate ───────────────────────────
#
# The three user-facing entry points of the composable pipeline:
#
#     sol      = fit(pipe, uvset)                      # solve only
#     out      = calibrate(sol, uvset; reduce = [...]) # apply + optional reduce
#     sol, out = fitcalibrate(pipe, uvset; reduce)     # fused single pass
#
# Every pipeline runs on the new engine (`_run_pipeline`): each solve step
# compiles and solves its OWN private model, one streaming pass per step under
# the visitor contract, and — for the
# output verbs — a per-scan-group output tail (`reduce_scan_output`)
# that applies the solution and the reduce chain while the group is resident.
# When the final solve step is a `TemporalSmoother` the tail FUSES into its
# pass (the group is corrected and reduced right after its per-AP solve, the
# production single-read path); otherwise a dedicated output pass streams the
# same tail. `calibrate(sol, uvset)` is that output pass standalone, so fused
# and standalone outputs are identical by construction.

"""
    fit(pipe::CalibrationPipeline, uvset::UVSet) -> CalibrationSolution
    fit(step_or_chain, uvset; exec = ExecutionConfig(), ref_ant = 1) -> CalibrationSolution

Solve the pipeline's calibration on `uvset` WITHOUT producing corrected data —
the estimation half of [`fitcalibrate`](@ref). The returned solution carries
per-stage provenance (`sol[:fringe]`, `sol[:bandpass]`, `sol[:adhoc]` —
[`stage_solution`](@ref)/[`stage_info`](@ref)) and records the data-transform
chain the scans were materialized through (`sol.transforms`), so diagnostics
and the standalone [`calibrate`](@ref) reproduce exactly what the solve saw.

`ReduceStep`s and [`AprioriAmplitude`](@ref) in the pipeline do not affect the
solution's θ and are ignored here (`AprioriAmplitude` is still RECORDED on the
solution — `sol.postcal` — so `calibrate(sol, uvset)` replays it); they run in
[`fitcalibrate`](@ref)'s output tail.

**A single `SolveStep` fit alone is the primitive** — the pipeline needs no
[`FringeFit`](@ref) step, and any composition of `SolveStep`s is legal,
including one standalone step (e.g. fitting a `BandpassEstimator` alone over
data already corrected by an earlier run). A multi-step `CalibrationPipeline`
is a fusion convenience built FROM repeated single-step solves: solving
`A |> B` in one call is equivalent to `sa = fit(A, uvset)` followed by
`fit(Fringe.ApplySolution(step_solution(sa, provides(A))) |> B, uvset)` — the
SAME mechanism ([`Fringe.ApplySolution`](@ref) dividing a finished step's
gains out of the stream before the next step's pass), just run within one
call instead of across two. See [`step_solution`](@ref) for the cross-run form
of this composition.
"""
function fit(pipe::CalibrationPipeline, uvset::UVSet)
    sol, _ = _run_pipeline(_parse_pipeline(pipe), pipe.exec, pipe.ref_ant, uvset)
    return sol
end

fit(
    x::Union{CalibrationStep, Fringe.AbstractDataTransform}, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(),
    ref_ant::Union{Integer, AbstractString, Symbol} = 1,
) = fit(CalibrationPipeline(x; exec, ref_ant), uvset)
fit(
    chain::StepChain, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(),
    ref_ant::Union{Integer, AbstractString, Symbol} = 1,
) = fit(CalibrationPipeline(chain; exec, ref_ant), uvset)

"""
    calibrate(sol::CalibrationSolution, uvset::UVSet;
              reduce = ReduceStep[], apply_flags = true, ntasks = Threads.nthreads()) -> UVSet

Apply a fitted solution to data, streaming one scan group at a time (a lazy set
is never fully materialized): each group is materialized through the transform
chain recorded on `sol` (so the output carries `transforms ∘ solution`, exactly
as the fused path would), corrected with the solution's gains and flags, run
through the recorded [`AprioriAmplitude`](@ref) chain (`sol.postcal`) and then
the `reduce` steps in order, and collected. This is the SAME per-group tail
[`fitcalibrate`](@ref) fuses into its final pass. Use it to calibrate the data
the solution was fit on, or another set with the same geometry.
"""
function calibrate(
        sol::CalibrationSolution, uvset::UVSet;
        reduce = ReduceStep[], apply_flags::Bool = true, ntasks::Integer = Threads.nthreads(),
        outer_executor = Executors.DEFAULT_EXECUTOR[], inner_executor = DynamicScheduler(),
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
    stream = Fringe.scan_stream(
        uvset; transforms = sol.transforms, ntasks = ntasks,
        outer_executor = outer_executor, inner_executor = inner_executor,
    )
    post = _compose_output_chain(sol.postcal, collect(reduce))
    group_pairs = Fringe.map_groups(stream; stage = :output) do spec
        keyed = Fringe.materialize_leaves(stream, spec)
        reduce_scan_output(
            stream.uvset, keyed, sol, post;
            executor = stream.inner_executor, apply_flags = apply_flags,
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
    sol, ctx = _run_fitcalibrate(pipe, uvset, reduce)
    return sol, ctx.output
end

fitcalibrate(
    x::Union{CalibrationStep, Fringe.AbstractDataTransform}, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(),
    ref_ant::Union{Integer, AbstractString, Symbol} = 1, kwargs...,
) = fitcalibrate(CalibrationPipeline(x; exec, ref_ant), uvset; kwargs...)
fitcalibrate(
    chain::StepChain, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(),
    ref_ant::Union{Integer, AbstractString, Symbol} = 1, kwargs...,
) = fitcalibrate(CalibrationPipeline(chain; exec, ref_ant), uvset; kwargs...)

# Shared driver for fitcalibrate: returns (sol, ctx).
function _run_fitcalibrate(pipe::CalibrationPipeline, uvset::UVSet, reduce)
    br = _parse_pipeline(pipe)
    post = _compose_output_chain(br.post_steps, collect(reduce))
    sol, output = _run_pipeline(br, pipe.exec, pipe.ref_ant, uvset; sink = OutputSink(post))
    ctx = CalibrationContext(
        uvset, sol, output,
        isempty(br.apriori) ? nothing : last(br.apriori).band_cals,
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
    reduced = postprocess(UVData.apply_calibration(sub, sol; executor, apply_flags))
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
            push!(fs, uv -> apply_calibration(
                uv, s.band_cals;
                min_elevation_deg = s.min_elevation_deg, on_missing_station = s.on_missing_station,
            ))
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
# OWN private (model, layout, θ) — never a merged one — under its own
# streaming pass (the visitor contract: start_pass!/process_scan!/finish_pass!).
# Gain correction between steps flows through the scan stream's transform
# chain, not a live θ evaluation: after each step finishes, its solution
# (its own model/layout/θ, undivided — no other step's contribution to zero
# out) is wrapped as an `ApplySolution` and appended, so every LATER step's
# pass reads already-corrected data (this is solve-time-only bookkeeping — the
# returned solution still records just `br.tfs`, the caller's own precal
# chain). This is the SAME `ApplySolution`-chaining a caller does explicitly
# across separate `fit` calls via `step_solution` (see its docstring) — a
# multi-step pipeline just runs the append within one call instead of two.
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
        br, exec::ExecutionConfig, ref_ant_spec::Union{Integer, AbstractString, Symbol},
        uvset::UVSet; sink = nothing,
    )
    solve_steps = br.solve_steps

    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    antennas = UVData.metadata(first_leaf).antennas
    nant = length(antennas)
    spec = (; geom, antennas)
    ref_ant = _resolve_ref_ant(ref_ant_spec, uvset)
    # A fresh `ScanStream` over the given transform list — cheap (geometry-only,
    # no data read). Used both for the initial stream and to grow the SOLVE-time
    # transform chain between steps (below): `ScanStream`/`SolveContext` fix the
    # transform-vector element type as a type parameter, so appending a
    # transform means building a new stream, not mutating one in place.
    _build_stream(tfs) = Fringe.scan_stream(
        uvset; geom = geom, transforms = tfs,
        ntasks = exec.ntasks, mem_fraction = exec.mem_fraction, mem_budget = exec.mem_budget,
        outer_executor = exec.outer_executor, inner_executor = exec.inner_executor,
    )
    stream = _build_stream(br.tfs)
    # Run-wide state shared, unmodified in identity, across every step's own
    # SolveContext below (only `model`/`layout`/`ev`/`θ` and `stream` change
    # per step — the rest is the run-wide part CHUNK-069 splits out).
    scratch = Dict{Symbol, Any}()
    # Intra-site (co-located) baseline exclusion, shared by the residual-pooling
    # stages and the exported flags (the monolith's `excl`).
    scratch[:excl] = if exec.exclude_colocated
        s = UVData._colocated_pair_set(antennas)
        isempty(s) ? nothing : s
    else
        nothing
    end
    # The final pass fuses the output tail only when it is a TemporalSmoother's:
    # that pass never repeats and finishes each group's θ before the tail runs
    # (the monolith's pass-2 structure). Anything else gets a dedicated pass.
    fused = sink !== nothing && last(solve_steps) isa TemporalSmoother
    tfs_solve = copy(br.tfs)
    step_solutions = StepSolution[]
    ctx = nothing
    # The whole solve runs under the monolith's thread-environment wrappers:
    # FITS decode helpers on `inner` tasks, single-threaded FFTW plans (the
    # per-baseline fan-out owns the parallelism), single-threaded BLAS.
    Fringe._with_decode_threads(stream.inner) do
        Fringe._with_fft_threads(1) do
            Fringe._with_single_blas_thread() do
                for (si, st) in enumerate(solve_steps)
                    mc = model_components(st, spec)
                    step_model = StationGainModel(phase = mc.phase, logamp = mc.logamp)
                    # A step may legitimately compile NO components at all (e.g.
                    # `DispersionSBDFit` with dispersion disabled and a band
                    # layout that can't support SBD either) — it still gets its
                    # own pass under the visitor contract, just with nothing to
                    # solve, so the empty-model guard (meant for a whole
                    # pipeline's compiled model) doesn't apply per-step.
                    step_layout = plan_parameters(step_model, nant, geom; require_nonempty = false)
                    ctx = SolveContext(
                        step_model, step_layout, geom, GainEvaluator(step_model, step_layout),
                        Calibration.component_vector(step_layout, zeros(step_layout.nθ)),
                        ref_ant, nant, antennas, stream, exec, scratch,
                    )
                    info = _run_pass!(
                        st, ctx, step_solutions;
                        sink = (fused && si == length(solve_steps)) ? sink : nothing,
                    )
                    push!(step_solutions, StepSolution(provides(st), step_model, step_layout, ctx.θ, info))
                    si == length(solve_steps) && break
                    # Gain correction between steps flows through the transform
                    # chain, not a live θ evaluation: this step's own θ is
                    # already undivided (no other step's contribution to zero
                    # out — each step solves its own private model), so it
                    # appends directly; every LATER step's pass then reads
                    # already-corrected data with no step evaluating or
                    # mutating another step's θ block.
                    push!(
                        tfs_solve,
                        Fringe.ApplySolution(
                            CalibrationSolution(
                                step_model, step_layout, geom, ctx.θ, NamedTuple(); name = provides(st),
                            ),
                        ),
                    )
                    stream = _build_stream(tfs_solve)
                end
            end
        end
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
        group_pairs = Fringe.map_groups(out_stream; progress = exec.progress, stage = :output) do gspec
            keyed = Fringe.materialize_leaves(out_stream, gspec)
            reduce_scan_output(
                out_stream.uvset, keyed, sol, sink.postprocess;
                executor = out_stream.inner_executor, apply_flags = sink.apply_flags,
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

# One streaming pass per solve step: materialize each selected group, hand its
# stack and geometry window to `process_scan!`, collect the per-group returns IN GROUP-INDEX ORDER
# into `ctx.scratch[:pass_results]`, then `finish_pass!` (repeating the pass
# while it returns `repeat_pass = true` — residual re-search rounds). With a
# `sink` (the fused fitcalibrate tail) the group is materialized as LEAVES
# first — the output must preserve the per-band leaf structure — the solve cube
# is stacked from them, and after `process_scan!` finishes the group's θ the
# tail corrects and reduces those same leaves in place. This is the executor
# seam the Dagger runner replaces at M7. Returns the step's diagnostics
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
            Fringe.flag_table(_fringe_flags(ctx), ctx.scratch[:excl])
        results = Fringe.map_groups(
            ctx.stream; selection = fit_selection(step, prior_solutions),
            snr = _scan_snr(prior_solutions),
            progress = ctx.exec.progress, stage = stage,
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
                    executor = ctx.stream.inner_executor, apply_flags = sink.apply_flags,
                )
            end
            (; index = gspec.index, decode = (tb - ta) / 1.0e9,
                work = (tc - tb) / 1.0e9, reduce = (time_ns() - tc) / 1.0e9, r, out)
        end
        ctx.scratch[:pass_results] = results
        sink === nothing || (ctx.scratch[:sink_pairs] = [res.out for res in results])
        out = finish_pass!(step, ctx)
        get(out, :repeat_pass, false) || return _pass_diagnostics(out, results, ngroups, t0)
    end
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
        Fringe.flag_table(_fringe_flags(ctx), ctx.scratch[:excl])...,
        ant_names = String.(collect(ctx.antennas.name)),
        (br.ff === nothing ? NamedTuple() : Fringe.estimator_info(br.ff.estimator))...,
        precal_applied = any(t -> t isa Fringe.ApplySolution, br.tfs),
        ntasks_used = ctx.stream.ntasks,
        inner_tasks = ctx.stream.inner,
    )
end

# The stage-B-unconstrained (station, scan) pairs an estimator reported, or none.
_fringe_flags(ctx::SolveContext) = get(() -> Tuple{Int, Int}[], ctx.scratch, :fringe_flags)

# ── Pipeline parsing ─────────────────────────────────────────────────────────

# Step order is never validated here — a step that cannot do its job with the
# data it is handed fails at the point of use (its own estimator/solve
# kernel), the same pattern `apply_calibration`'s data-level guards use. Only
# `provides`'s NAMING role is checked: two steps sharing a non-`:nothing`
# capability would silently collide in `stage_solution`/`step_solution`'s
# by-name lookup (a `findfirst`, so the second step's solution would be
# unreachable) — that is a naming conflict, not an ordering rule, so it stays
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
                        "BandpassEstimator, TemporalSmoother, or a third-party SolveStep), " *
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
