# ── Pipeline verbs: fit / calibrate / fitcalibrate ───────────────────────────
#
# The three user-facing entry points of the composable pipeline:
#
#     sol      = fit(pipe, uvset)                      # solve only
#     out      = calibrate(sol, uvset; reduce = [...]) # apply + optional reduce
#     sol, out = fitcalibrate(pipe, uvset; reduce)     # fused single pass
#
# Every pipeline runs on the new engine (`_fit_new_engine`): one compiled model,
# one streaming pass per solve step under the visitor contract, and — for the
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
    sol, _ = _fit_new_engine(_parse_pipeline(pipe), pipe.exec, uvset)
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
        executor::Executors.AbstractExecutor = Executors.current_executor(),
    )
    any(t -> t === missing, sol.transforms) && error(
        "calibrate: this solution records a transform that did not survive serialization " *
            "(saved as `missing`) — re-fit, or apply the original transform chain manually."
    )
    any(t -> t === missing, sol.postcal) && error(
        "calibrate: this solution records an a-priori amplitude cal that did not survive " *
            "serialization (saved as `missing`) — re-fit, or apply it manually."
    )
    stream = Fringe.scan_stream(uvset; transforms = sol.transforms, ntasks = ntasks, executor = executor)
    post = _compose_output_chain(sol.postcal, collect(reduce))
    group_pairs = Fringe.map_groups(stream; stage = :output) do spec
        keyed = Fringe.materialize_leaves(stream, spec)
        reduce_scan_output(
            stream.uvset, keyed, sol, post;
            ntasks = stream.inner, apply_flags = apply_flags,
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
    sol, output = _fit_new_engine(br, pipe.exec, uvset; sink = OutputSink(post))
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
        ntasks::Integer = 1, apply_flags::Bool = true,
    )
    sub_branches = DimensionalData.TreeDict()
    for (k, leaf) in keyed
        sub_branches[k] = leaf
    end
    sub = DimensionalData.rebuild(uvset; branches = sub_branches)
    reduced = postprocess(UVData.apply_calibration(sub, sol; ntasks = ntasks, apply_flags = apply_flags))
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
        s isa AprioriAmplitude || error(
            "output chain: recorded postcal entry $(typeof(s)) is not an AprioriAmplitude."
        )
        push!(fs, uv -> apply_calibration(
            uv, s.band_cals;
            min_elevation_deg = s.min_elevation_deg, on_missing_station = s.on_missing_station,
        ))
    end
    ctx = CalibrationContext()
    for st in reduces
        st isa AprioriAmplitude && error(
            "fitcalibrate/calibrate: AprioriAmplitude is a pipeline step, not a reduction — " *
                "place it in the CalibrationPipeline (it is recorded on the solution and " *
                "replayed by `calibrate(sol, uvset)`)."
        )
        st isa ReduceStep || error(
            "fitcalibrate/calibrate: `reduce` accepts ReduceSteps only (got $(typeof(st)))."
        )
        f, ctx = prepare_reducer(st, ctx)
        push!(fs, f)
    end
    isempty(fs) && return identity
    return uv -> foldl((acc, f) -> f(acc), fs; init = uv)
end

# ── The new-engine path: the compiled model + visitor pass runner ────────────

# Solve a pipeline on the new engine: collect each solve step's model
# components IN STEP ORDER into ONE compiled StationGainModel/θ, build the scan
# stream, and give each solve step its own streaming pass under the visitor
# contract (start_pass!/process_scan!/finish_pass!). With a `sink`, the output
# tail runs per group — fused into the final pass when that pass is a
# TemporalSmoother's (it never repeats and its θ writes precede the tail),
# else as a dedicated output pass after the solves. Returns `(sol, output)`
# (`output === nothing` without a sink).
function _fit_new_engine(br, exec::ExecutionConfig, uvset::UVSet; sink = nothing)
    ff = br.ff
    solve_steps = SolveStep[ff]
    br.bp === nothing || push!(solve_steps, br.bp)
    br.sm === nothing || push!(solve_steps, br.sm)

    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    antennas = UVData.metadata(first_leaf).antennas
    nant = length(antennas)
    spec = (; geom, antennas)
    phase = ()
    logamp = ()
    comp_owner = NamedTuple[]
    for st in solve_steps
        mc = model_components(st, spec)
        push!(comp_owner, (;
            phase = collect((length(phase) + 1):(length(phase) + length(mc.phase))),
            logamp = collect((length(logamp) + 1):(length(logamp) + length(mc.logamp))),
        ))
        phase = (phase..., mc.phase...)
        logamp = (logamp..., mc.logamp...)
    end
    model = StationGainModel(phase = phase, logamp = logamp)
    layout = plan_parameters(model, nant, geom)
    ev = GainEvaluator(model, layout)
    stream = Fringe.scan_stream(
        uvset; geom = geom, transforms = br.tfs,
        ntasks = exec.ntasks, mem_fraction = exec.mem_fraction, mem_budget = exec.mem_budget,
        workspace = Fringe.FringeWorkspace, executor = exec.executor,
    )
    ctx = SolveContext(
        model, layout, geom, ev, zeros(layout.nθ),
        _resolve_ref_ant(ff.model.ref_ant, uvset), nant, antennas,
        stream, exec, StageRecord[], Dict{Symbol, Any}(),
    )
    # Intra-site (co-located) baseline exclusion, shared by the residual-pooling
    # stages and the exported flags (the monolith's `excl`).
    ctx.scratch[:excl] = if exec.exclude_colocated
        s = UVData._colocated_pair_set(antennas)
        isempty(s) ? nothing : s
    else
        nothing
    end
    # The final pass fuses the output tail only when it is a TemporalSmoother's:
    # that pass never repeats and finishes each group's θ before the tail runs
    # (the monolith's pass-2 structure). Anything else gets a dedicated pass.
    fused = sink !== nothing && last(solve_steps) isa TemporalSmoother
    # The whole solve runs under the monolith's thread-environment wrappers:
    # FITS decode helpers on `inner` tasks, single-threaded FFTW plans (the
    # per-baseline fan-out owns the parallelism), single-threaded BLAS.
    Fringe._with_decode_threads(stream.inner) do
        Fringe._with_fft_threads(1) do
            Fringe._with_single_blas_thread() do
                for (si, st) in enumerate(solve_steps)
                    _run_pass!(
                        st, ctx, si, comp_owner[si];
                        sink = (fused && si == length(solve_steps)) ? sink : nothing,
                    )
                end
            end
        end
    end
    sol = CalibrationSolution(
        model, layout, geom, ctx.θ, _new_engine_info(ctx, br);
        stages = ctx.stages, transforms = br.tfs, postcal = br.apriori,
    )
    sink === nothing && return sol, nothing
    if !fused
        group_pairs = Fringe.map_groups(ctx.stream; progress = exec.progress, stage = :output) do gspec
            keyed = Fringe.materialize_leaves(ctx.stream, gspec)
            reduce_scan_output(
                ctx.stream.uvset, keyed, sol, sink.postprocess;
                ntasks = ctx.stream.inner, apply_flags = sink.apply_flags,
            )
        end
        return sol, assemble_output(uvset, group_pairs)
    end
    return sol, assemble_output(uvset, ctx.scratch[:sink_pairs])
end

# One streaming pass per solve step: materialize each selected group, hand the
# view to `process_scan!`, collect the per-group returns IN GROUP-INDEX ORDER
# into `ctx.scratch[:pass_results]`, then `finish_pass!` (repeating the pass
# while it returns `repeat_pass = true` — residual re-search rounds). With a
# `sink` (the fused fitcalibrate tail) the group is materialized as LEAVES
# first — the output must preserve the per-band leaf structure — the solve cube
# is stacked from them, and after `process_scan!` finishes the group's θ the
# tail corrects and reduces those same leaves in place. This is the executor
# seam the Dagger runner replaces at M7.
function _run_pass!(step::SolveStep, ctx::SolveContext, step_index::Int, comps; sink = nothing)
    stage = provides(step)
    t0 = time_ns()
    while true
        start_pass!(step, ctx)
        flag_nt = sink === nothing ? nothing :
            Fringe.flag_table(_fringe_flags(ctx), ctx.scratch[:excl])
        results = Fringe.map_groups(
            ctx.stream; selection = fit_selection(step),
            snr = get(ctx.scratch, :scan_snr, nothing),
            progress = ctx.exec.progress, stage = stage,
        ) do gspec
            ta = time_ns()
            if sink === nothing
                grp = Fringe.materialize_cube(ctx.stream, gspec)
                keyed = nothing
            else
                keyed = Fringe.materialize_leaves(ctx.stream, gspec)
                grp = Streaming._stacked_scan_group([m for (_, m) in keyed], ctx.stream.geom)
            end
            v = Fringe.scan_view(ctx.stream, grp)
            tb = time_ns()
            r = process_scan!(step, ctx, v)
            tc = time_ns()
            out = nothing
            if sink !== nothing
                # θ is complete for this group (its per-scan slots were just
                # written; the global blocks were finalized by earlier passes),
                # so the group-local solution corrects identically to the final
                # global one — the monolith's pass-2 invariant.
                sol_local = CalibrationSolution(ctx.model, ctx.layout, ctx.geom, ctx.θ, flag_nt)
                out = reduce_scan_output(
                    ctx.stream.uvset, keyed, sol_local, sink.postprocess;
                    ntasks = ctx.stream.inner, apply_flags = sink.apply_flags,
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
            info = (; (k => v for (k, v) in pairs(out) if k !== :repeat_pass)...)
            push!(ctx.stages, StageRecord(stage, step_index, comps.phase, comps.logamp, info))
            break
        end
    end
    times = get!(() -> Dict{Symbol, Float64}(), ctx.scratch, :pass_times)::Dict{Symbol, Float64}
    times[stage] = get(times, stage, 0.0) + (time_ns() - t0) / 1.0e9
    return nothing
end

# The solution-level `info` NamedTuple of a new-engine fit, assembled from the
# passes' scratch state. The per-scan detection tables are what the estimator
# published: an estimator with no notion of matched-filter detections leaves
# them empty rather than being required to fake them.
function _new_engine_info(ctx::SolveContext, br)
    ngroups = length(ctx.stream.groups)
    fringe = _fringe_stage_info(ctx)
    rf = get(ctx.scratch, :refine, nothing)
    refined = get(ctx.scratch, :refined_scans, Set{Int}())
    sm_ran = any(r -> r.name === :adhoc, ctx.stages)
    times = ctx.scratch[:pass_times]::Dict{Symbol, Float64}
    return (;
        nant = ctx.nant,
        nscan = ngroups,
        scan_max_snr = get(() -> zeros(ngroups), ctx.scratch, :scan_snr)::Vector{Float64},
        scan_ncells = get(() -> zeros(ngroups), ctx.scratch, :scan_ncells)::Vector{Float64},
        scan_chi = fill(fringe.chi, ngroups),
        scan_ncomp = fill(fringe.ncomp, ngroups),
        stageB_rejected = fringe.rejected,
        Fringe.flag_table(_fringe_flags(ctx), ctx.scratch[:excl])...,
        # dTEC/SBD cover the whole track once the temporal-smoother pass ran;
        # without it only the scans the bandpass stage read were refined
        # (`refined_scans`) and the rest stay 0.
        dispersion_applied = rf !== nothing && rf.disp_plan !== nothing &&
            (sm_ran || !isempty(refined)),
        sbd_applied = rf !== nothing && rf.sbd_plans !== nothing &&
            (sm_ran || !isempty(refined)),
        dtec_rejected = get(ctx.scratch, :bp_dtec_rejected, 0) +
            get(ctx.scratch, :adhoc_dtec_rejected, 0),
        ant_names = String.(collect(ctx.antennas.name)),
        Fringe.estimator_info(br.ff.estimator)...,
        precal_applied = any(t -> t isa Fringe.ApplySolution, br.tfs),
        ntasks_used = ctx.stream.ntasks,
        inner_tasks = ctx.stream.inner,
        t_search_pass = get(times, :fringe, 0.0),
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

# The fringe stage's recorded diagnostics (chi/ncomp/rejected) off the context.
function _fringe_stage_info(ctx::SolveContext)
    i = findfirst(r -> r.name === :fringe, ctx.stages)
    i === nothing && error("internal: no fringe stage record on the SolveContext")
    return ctx.stages[i].info
end

# ── Pipeline parsing ─────────────────────────────────────────────────────────

# Parse a pipeline into (transforms, the FringeFit, the optional
# BandpassEstimator / TemporalSmoother, AprioriAmplitude steps, trailing
# ReduceSteps), validating step order and multiplicity.
function _parse_pipeline(pipe::CalibrationPipeline)
    tfs = Fringe.AbstractDataTransform[]
    ff = nothing
    bp = nothing
    sm = nothing
    apriori = AprioriAmplitude[]
    post_reduce = ReduceStep[]
    for s in pipe.steps
        if s isa DataTransformStep
            push!(tfs, s.t)
        elseif s isa FringeFit
            ff === nothing || error("fit/fitcalibrate: exactly ONE FringeFit per pipeline.")
            ff = s
        elseif s isa BandpassEstimator
            bp === nothing || error("fit/fitcalibrate: exactly ONE BandpassEstimator per pipeline.")
            ff === nothing && error(
                "fit/fitcalibrate: BandpassEstimator requires :fringe — place FringeFit before it."
            )
            bp = s
        elseif s isa TemporalSmoother
            sm === nothing || error("fit/fitcalibrate: exactly ONE TemporalSmoother per pipeline.")
            ff === nothing && error(
                "fit/fitcalibrate: TemporalSmoother requires :fringe — place FringeFit before it."
            )
            sm = s
        elseif s isa AprioriAmplitude
            push!(apriori, s)
        elseif s isa ReduceStep
            push!(post_reduce, s)
        else
            error(
                "fit/fitcalibrate: step $(typeof(s)) is not runnable — supported: data " *
                    "transforms, FringeFit, BandpassEstimator, TemporalSmoother, " *
                    "AprioriAmplitude, and ReduceSteps."
            )
        end
    end
    ff === nothing && error("fit/fitcalibrate: the pipeline contains no FringeFit step.")
    return (; tfs, ff, bp, sm, apriori, post_reduce)
end
