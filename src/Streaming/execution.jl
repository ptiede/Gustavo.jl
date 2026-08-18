# ── Execution configuration (run-wide, not per-step) ─────────────────────────

"""
    ExecutionConfig(; mem_fraction = 0.6, mem_budget = nothing, progress = nothing,
                    outer_executor = SerialScheduler(), inner_executor = DynamicScheduler())

Run-wide RESOURCES for a streaming run, shared by every pass over the data — as
opposed to per-step options, which shape an estimator or a step's own model, and
as opposed to the gauge convention (`gauge`, on `CalibrationPipeline` itself),
which is shared by every pass but is not a resource. Two runs differing only in
their `ExecutionConfig` solve the same problem with the same gauge.

A [`ScanStream`](@ref) carries the config it was built from, so every pass over
that stream draws its schedulers and progress reporting from one place; read
them back with [`outer_executor`](@ref) / [`inner_executor`](@ref).

THE TWO SCHEDULERS OWN THE RUN'S PARALLELISM. Each is used exactly as
configured — nothing here rewrites a scheduler you passed — so the task count
you set on one is the concurrency you get.

- `mem_fraction` / `mem_budget` — the DETERMINISTIC memory budget (`mem_budget`
  bytes if set, else `mem_fraction` of physical RAM) the outer scheduler's task
  count is checked against when a stream is built. A configuration that cannot
  fit is an error, not something silently reduced.
- `progress` — `(stage, done, total)` callback per completed scan of each pass.
- `outer_executor` — the ACROSS-scan (group scheduling) scheduler, and thus how
  many scan groups are resident at once: `SerialScheduler()` by default (one
  group at a time on the calling task); any other OhMyThreads `Scheduler` to run
  its task count of groups concurrently (`GreedyScheduler(; ntasks)` keeps a
  task pulling the next-heaviest group as it frees up, which balances uneven
  scan lengths best). Groups are dispatched heaviest-first under every scheduler.
- `inner_executor` — the WITHIN-scan fan-out scheduler, an OhMyThreads
  `Scheduler` (`DynamicScheduler()` by default; `SerialScheduler()` to run the
  within-scan solves single-threaded).
"""
Base.@kwdef struct ExecutionConfig{P, O, I}
    mem_fraction::Float64 = 0.6
    mem_budget::Union{Nothing, Float64} = nothing
    progress::P = nothing
    outer_executor::O = SerialScheduler()
    inner_executor::I = DynamicScheduler()
end

"""
    outer_executor(x) -> Scheduler

The ACROSS-scan scheduler of an [`ExecutionConfig`](@ref) or of the
[`ScanStream`](@ref) built from one: how many scan groups [`map_groups`](@ref)
keeps resident at once.
"""
outer_executor(x::ExecutionConfig) = x.outer_executor

"""
    inner_executor(x) -> Scheduler

The WITHIN-scan fan-out scheduler of an [`ExecutionConfig`](@ref) or of the
[`ScanStream`](@ref) built from one: the default `executor` for the per-group
kernels ([`materialize_cube`](@ref) and the solves that run on its output).
"""
inner_executor(x::ExecutionConfig) = x.inner_executor

# The `(stage, done, total)` callback, or `nothing`. Unexported: pass reporting
# is plumbed by `map_groups`, not assembled by callers.
progress_callback(x::ExecutionConfig) = x.progress

"""
    ProgressLogger(; min_interval = 5.0, io = stdout)

A ready-made [`ExecutionConfig`](@ref) `progress` callback: prints each pass's
`(stage, done, total)` progress, throttled to at most one line every
`min_interval` seconds, with an ETA extrapolated from the pass's mean
completion rate so far. A pass's start (`done == 0`) and finish
(`done == total`) always print, regardless of the throttle. Stateful — build
one `ProgressLogger` per `fit`/`fitcalibrate` call; sharing an instance across
concurrent runs mixes their timers.

    fit(pipe, uvset; exec = ExecutionConfig(progress = ProgressLogger()))
"""
mutable struct ProgressLogger{IOT}
    min_interval::Float64
    io::IOT
    stage::Symbol
    t_start::Float64
    t_last::Float64
end
ProgressLogger(; min_interval::Real = 5.0, io = stdout) =
    ProgressLogger(Float64(min_interval), io, :nothing, 0.0, 0.0)

function (p::ProgressLogger)(stage::Symbol, done::Integer, total::Integer)
    t = time()
    if stage !== p.stage || done == 0
        p.stage, p.t_start, p.t_last = stage, t, t
        println(p.io, "[$stage] starting ($total scan group$(total == 1 ? "" : "s"))")
        flush(p.io)
        return nothing
    end
    finished = done == total
    if !finished && (t - p.t_last) < p.min_interval
        return nothing
    end
    p.t_last = t
    elapsed = t - p.t_start
    if finished
        println(p.io, "[$stage] $done/$total scan groups done ($(round(elapsed; digits = 1))s)")
    else
        eta = elapsed * (total - done) / done
        pct = round(100 * done / total; digits = 1)
        println(p.io, "[$stage] $done/$total scan groups ($(pct)%), ETA $(round(eta; digits = 1))s")
    end
    # A redirected stdout is block-buffered, so progress on a long pass would otherwise
    # not reach the file until the stream closes — exactly when it is no longer useful.
    flush(p.io)
    return nothing
end
