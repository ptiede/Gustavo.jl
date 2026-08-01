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
    fit(step_or_chain, uvset; exec = ExecutionConfig()) -> CalibrationSolution

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
"""
function fit(pipe::CalibrationPipeline, uvset::UVSet)
    sol, _ = _run_pipeline(_parse_pipeline(pipe), pipe.exec, uvset)
    return sol
end

fit(x::Union{CalibrationStep, Fringe.AbstractDataTransform}, uvset::UVSet; exec::ExecutionConfig = ExecutionConfig()) =
    fit(CalibrationPipeline(x; exec), uvset)
fit(chain::StepChain, uvset::UVSet; exec::ExecutionConfig = ExecutionConfig()) =
    fit(CalibrationPipeline(chain; exec), uvset)

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
step in the pipeline, then the pipeline's `ReduceStep`s and the `reduce` kwarg's
(in that order) — run per scan group while it is resident. When the pipeline
ends in a [`TemporalSmoother`](@ref) the whole tail fuses into that final
streaming pass (one read solves, corrects, and reduces each group).
"""
function fitcalibrate(pipe::CalibrationPipeline, uvset::UVSet; reduce = ReduceStep[])
    sol, ctx = _run_fitcalibrate(pipe, uvset, reduce)
    return sol, ctx.output
end

fitcalibrate(x::Union{CalibrationStep, Fringe.AbstractDataTransform}, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(), kwargs...) =
    fitcalibrate(CalibrationPipeline(x; exec), uvset; kwargs...)
fitcalibrate(chain::StepChain, uvset::UVSet;
    exec::ExecutionConfig = ExecutionConfig(), kwargs...) =
    fitcalibrate(CalibrationPipeline(chain; exec), uvset; kwargs...)

# Shared driver for fitcalibrate: returns (sol, ctx).
function _run_fitcalibrate(pipe::CalibrationPipeline, uvset::UVSet, reduce)
    br = _parse_pipeline(pipe)
    post = _compose_output_chain(br.apriori, vcat(br.post_reduce, collect(reduce)))
    sol, output = _run_pipeline(br, pipe.exec, uvset; sink = OutputSink(post))
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

# Compose the output chain: the AprioriAmplitude steps (after the fringe gains,
# before any reductions — they need the native channels), then the ReduceSteps
# in order, into one `UVSet -> UVSet` function.
function _compose_output_chain(apriori, reduces)
    fs = Any[]
    for s in apriori
        s isa AprioriAmplitude || throw(
            ArgumentError(
                "output chain: recorded postcal entry $(typeof(s)) is not an AprioriAmplitude."
            )
        )
        push!(fs, uv -> apply_calibration(
            uv, s.band_cals;
            min_elevation_deg = s.min_elevation_deg, on_missing_station = s.on_missing_station,
        ))
    end
    ctx = CalibrationContext()
    for st in reduces
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
# chain). Non-data info an earlier step published (e.g. the fringe stage's
# per-scan SNR) reaches a later step through the plain, ordered list of
# finished `StepSolution`s (`fit_selection`), not a shared scratch dict. With a
# `sink`, the output tail runs per group — fused into the final pass when that
# pass is a TemporalSmoother's (it never repeats and its θ writes precede the
# tail), else as a dedicated output pass after the solves. Once every step has
# solved, the runner composes ONE legacy-shaped `CalibrationSolution` (merged
# model/layout/θ, `StageRecord` provenance) from the finished `StepSolution`s —
# `CalibrationSolution`'s public shape is unchanged by the per-step split.
# Returns `(sol, output)` (`output === nothing` without a sink).
function _run_pipeline(br, exec::ExecutionConfig, uvset::UVSet; sink = nothing)
    ff = br.ff
    solve_steps = SolveStep[ff]
    br.ds === nothing || push!(solve_steps, br.ds)
    br.bp === nothing || push!(solve_steps, br.bp)
    br.sm === nothing || push!(solve_steps, br.sm)

    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    antennas = UVData.metadata(first_leaf).antennas
    nant = length(antennas)
    spec = (; geom, antennas)
    ref_ant = _resolve_ref_ant(ff.model.ref_ant, uvset)
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
                        ref_ant, nant, antennas, stream, exec, StageRecord[], scratch,
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
                        Fringe.ApplySolution(CalibrationSolution(step_model, step_layout, geom, ctx.θ, NamedTuple())),
                    )
                    stream = _build_stream(tfs_solve)
                end
            end
        end
    end
    model, layout, θ, stages = _compose_legacy_solution(step_solutions, nant, geom)
    sol = CalibrationSolution(
        model, layout, geom, θ, _new_engine_info(ctx, br, model, layout, stages);
        stages, transforms = br.tfs, postcal = br.apriori,
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

# The `Vector{StepSolution}` a run has finished so far, composed into ONE
# legacy-shaped `(model, layout, θ, stages)` — `CalibrationSolution`'s public
# shape is unchanged by the per-step split (CHUNK-069): each step solved on
# its own private model, so `model`/`layout` here are built by merging those
# models (in step order, matching `_merge_components`'s duplicate-name guard)
# and `θ` by copying each step's own values into the merged layout's matching
# component ranges — components pair up positionally between a step's own
# layout and the merged one because merging a `NamedTuple` preserves key order
# as a concatenation, so a step's phase (resp. log-amp) leaves occupy the same
# relative position in the merged phase (resp. log-amp) group as in its own.
function _compose_legacy_solution(step_solutions::Vector{StepSolution}, nant::Integer, geom::DataGeometry)
    model = StationGainModel(
        phase = reduce(Calibration._merge_components, (s.model.phase for s in step_solutions); init = (;)),
        logamp = reduce(Calibration._merge_components, (s.model.logamp for s in step_solutions); init = (;)),
    )
    layout = plan_parameters(model, nant, geom)
    θ = Calibration.component_vector(layout, zeros(layout.nθ))
    stages = StageRecord[]
    phase_off = 0
    logamp_off = 0
    for (si, ssol) in enumerate(step_solutions)
        npi = ssol.layout.nphase
        nli = length(ssol.layout.plans) - npi
        for k in 1:npi
            θ[layout.plans[phase_off + k].range] .= ssol.θ[ssol.layout.plans[k].range]
        end
        for k in 1:nli
            θ[layout.plans[layout.nphase + logamp_off + k].range] .= ssol.θ[ssol.layout.plans[npi + k].range]
        end
        push!(
            stages,
            StageRecord(
                ssol.name, si, collect((phase_off + 1):(phase_off + npi)),
                collect((logamp_off + 1):(logamp_off + nli)), ssol.info,
            ),
        )
        phase_off += npi
        logamp_off += nli
    end
    return model, layout, θ, stages
end

# The per-scan SNR a LATER step's selection may want (e.g. `BandpassEstimator`'s
# `BrightestCalibrator`), read off the most recent finished `StepSolution` that
# published one — never a shared scratch dict (CHUNK-067c). `nothing` when no
# prior step published SNR (an estimator with no notion of it, or none yet).
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
# NamedTuple (`repeat_pass` stripped) — the caller wraps it into a
# `StepSolution` alongside this step's own `ctx.model`/`layout`/`θ`.
function _run_pass!(step::SolveStep, ctx::SolveContext, prior_solutions; sink = nothing)
    stage = provides(step)
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
                sol_local = CalibrationSolution(ctx.model, ctx.layout, ctx.geom, ctx.θ, flag_nt)
                out = reduce_scan_output(
                    ctx.stream.uvset, keyed, sol_local, sink.postprocess;
                    executor = ctx.stream.inner_executor, apply_flags = sink.apply_flags,
                )
            end
            (; index = gspec.index, decode = (tb - ta) / 1.0e9,
                work = (tc - tb) / 1.0e9, reduce = (time_ns() - tc) / 1.0e9, r, out)
        end
        ctx.scratch[:pass_results] = results
        if sink !== nothing
            ngroups = length(ctx.stream.groups)
            tred = get!(() -> zeros(ngroups), ctx.scratch, :scan_t_reduce)::Vector{Float64}
            for res in results
                tred[res.index] = res.reduce
            end
            ctx.scratch[:sink_pairs] = [res.out for res in results]
        end
        out = finish_pass!(step, ctx)
        if !get(out, :repeat_pass, false)
            times = get!(() -> Dict{Symbol, Float64}(), ctx.scratch, :pass_times)::Dict{Symbol, Float64}
            times[stage] = get(times, stage, 0.0) + (time_ns() - t0) / 1.0e9
            return (; (k => v for (k, v) in pairs(out) if k !== :repeat_pass)...)
        end
    end
end

# The solution-level `info` NamedTuple of a new-engine fit, assembled from the
# passes' scratch state plus the composed `(model, layout, stages)` (see
# `_compose_legacy_solution`). The per-scan detection tables are what the
# estimator published: an estimator with no notion of matched-filter
# detections leaves them empty rather than being required to fake them.
function _new_engine_info(ctx::SolveContext, br, model::StationGainModel, layout::ParameterLayout, stages)
    ngroups = length(ctx.stream.groups)
    fringe = _fringe_stage_info(stages)
    times = ctx.scratch[:pass_times]::Dict{Symbol, Float64}
    refine_i = findfirst(r -> r.name === :refine, stages)
    refine_info = refine_i === nothing ? NamedTuple() : stages[refine_i].info
    return (;
        nant = ctx.nant,
        nscan = ngroups,
        scan_max_snr = get(fringe, :scan_snr, zeros(ngroups))::Vector{Float64},
        scan_ncells = get(() -> zeros(ngroups), ctx.scratch, :scan_ncells)::Vector{Float64},
        scan_chi = fill(fringe.chi, ngroups),
        scan_ncomp = fill(fringe.ncomp, ngroups),
        stageB_rejected = fringe.rejected,
        Fringe.flag_table(_fringe_flags(ctx), ctx.scratch[:excl])...,
        # `DispersionSBDFit`'s own pass (when present) covers every selected
        # scan unconditionally, so the compiled model alone (not which OTHER
        # steps ran) says whether dTEC/SBD were fit.
        dispersion_applied = Calibration._dispersion_plan(model, layout) !== nothing,
        sbd_applied = Fringe._sbd_plans(model, layout) !== nothing,
        dtec_rejected = get(refine_info, :dtec_rejected, 0),
        ant_names = String.(collect(ctx.antennas.name)),
        Fringe.estimator_info(br.ff.estimator)...,
        precal_applied = any(t -> t isa Fringe.ApplySolution, br.tfs),
        ntasks_used = ctx.stream.ntasks,
        inner_tasks = ctx.stream.inner,
        t_search_pass = get(times, :fringe, 0.0),
        t_refine_pass = get(times, :refine, 0.0),
        t_bandpass_stage = get(times, :bandpass, 0.0),
        t_adhoc_pass = get(times, :adhoc, 0.0),
        scan_t_decode = get(() -> zeros(ngroups), ctx.scratch, :scan_t_decode)::Vector{Float64},
        scan_t_search = get(() -> zeros(ngroups), ctx.scratch, :scan_t_search)::Vector{Float64},
        scan_t_decode2 = get(ctx.scratch, :scan_t_decode2, zeros(ngroups))::Vector{Float64},
        scan_t_adhoc = get(ctx.scratch, :scan_t_adhoc, zeros(ngroups))::Vector{Float64},
        scan_t_reduce = get(ctx.scratch, :scan_t_reduce, zeros(ngroups))::Vector{Float64},
        Fringe.detection_table(get(() -> Vector{Fringe.DetectionRow}[], ctx.scratch, :scan_dets))...,
    )
end

# The stage-B-unconstrained (station, scan) pairs an estimator reported, or none.
_fringe_flags(ctx::SolveContext) = get(() -> Tuple{Int, Int}[], ctx.scratch, :fringe_flags)

# The fringe stage's recorded diagnostics (chi/ncomp/rejected/scan_snr).
function _fringe_stage_info(stages)
    i = findfirst(r -> r.name === :fringe, stages)
    i === nothing && error("internal: no fringe stage record among the run's StageRecords")
    return stages[i].info
end

# ── Pipeline parsing ─────────────────────────────────────────────────────────

# Parse a pipeline into (transforms, the FringeFit, the optional
# DispersionSBDFit / BandpassEstimator / TemporalSmoother, AprioriAmplitude
# steps, trailing ReduceSteps), validating step order and multiplicity.
function _parse_pipeline(pipe::CalibrationPipeline)
    tfs = Fringe.AbstractDataTransform[]
    ff = nothing
    ds = nothing
    bp = nothing
    sm = nothing
    apriori = AprioriAmplitude[]
    post_reduce = ReduceStep[]
    for s in pipe.steps
        if s isa DataTransformStep
            push!(tfs, s.t)
        elseif s isa FringeFit
            ff === nothing ||
                throw(ArgumentError("fit/fitcalibrate: exactly ONE FringeFit per pipeline."))
            ff = s
        elseif s isa DispersionSBDFit
            ds === nothing || throw(
                ArgumentError("fit/fitcalibrate: exactly ONE DispersionSBDFit per pipeline.")
            )
            ff === nothing && throw(
                ArgumentError(
                    "fit/fitcalibrate: DispersionSBDFit requires :fringe — place FringeFit " *
                        "before it."
                )
            )
            ds = s
        elseif s isa BandpassEstimator
            bp === nothing || throw(
                ArgumentError("fit/fitcalibrate: exactly ONE BandpassEstimator per pipeline.")
            )
            ff === nothing && throw(
                ArgumentError(
                    "fit/fitcalibrate: BandpassEstimator requires :fringe — place FringeFit " *
                        "before it."
                )
            )
            bp = s
        elseif s isa TemporalSmoother
            sm === nothing || throw(
                ArgumentError("fit/fitcalibrate: exactly ONE TemporalSmoother per pipeline.")
            )
            ff === nothing && throw(
                ArgumentError(
                    "fit/fitcalibrate: TemporalSmoother requires :fringe — place FringeFit " *
                        "before it."
                )
            )
            sm = s
        elseif s isa AprioriAmplitude
            push!(apriori, s)
        elseif s isa ReduceStep
            push!(post_reduce, s)
        else
            throw(
                ArgumentError(
                    "fit/fitcalibrate: step $(typeof(s)) is not runnable — supported: data " *
                        "transforms, FringeFit, DispersionSBDFit, BandpassEstimator, " *
                        "TemporalSmoother, AprioriAmplitude, and ReduceSteps."
                )
            )
        end
    end
    ff === nothing && throw(
        ArgumentError("fit/fitcalibrate: the pipeline contains no FringeFit step.")
    )
    return (; tfs, ff, ds, bp, sm, apriori, post_reduce)
end
