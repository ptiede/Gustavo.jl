# ── Execution configuration (run-wide, not per-step) ─────────────────────────

"""
    ExecutionConfig(; progress = nothing, outer_executor = SerialScheduler(),
                    inner_executor = DynamicScheduler())

Run-wide resources for `fit` and `calibrate`, shared by every pass over the
data; per-step options, the gauge included, live on the steps. Two runs differing only in their `ExecutionConfig` solve the same
problem. Each scheduler is used exactly as configured, so the task count you
set is the concurrency you get.

- `progress` — `(stage, done, total)` callback per completed scan of each
  pass.
- `outer_executor` — the across-scan scheduler, and thus how many scan
  groups are resident at once: `SerialScheduler()` by default; any other
  OhMyThreads `Scheduler` runs its task count of groups concurrently
  (`GreedyScheduler(; ntasks)` balances uneven scan lengths best). Groups
  are dispatched largest-first under every scheduler. Each resident group
  holds its visibilities, weights and flags plus the step's working memory,
  so the task count is bounded by memory as well as cores.
- `inner_executor` — the within-scan fan-out scheduler
  (`DynamicScheduler()` by default; `SerialScheduler()` for
  single-threaded within-scan solves).
"""
Base.@kwdef struct ExecutionConfig{P, O, I}
    progress::P = nothing
    outer_executor::O = SerialScheduler()
    inner_executor::I = DynamicScheduler()
end

"""
    outer_executor(x) -> Scheduler

The across-scan scheduler of an [`ExecutionConfig`](@ref): how many scan
groups a pass keeps resident at once.
"""
outer_executor(x::ExecutionConfig) = x.outer_executor

"""
    inner_executor(x) -> Scheduler

The within-scan fan-out scheduler of an [`ExecutionConfig`](@ref): reading a
group's Measurement Sets and the per-group kernels.
"""
inner_executor(x::ExecutionConfig) = x.inner_executor

# The `(stage, done, total)` callback, or `nothing`.
progress_callback(x::ExecutionConfig) = x.progress

"""
    ProgressLogger(; min_interval = 5.0, io = stdout)

A ready-made [`ExecutionConfig`](@ref) `progress` callback: prints each pass's
`(stage, done, total)` progress, throttled to at most one line every
`min_interval` seconds, with an ETA extrapolated from the pass's mean
completion rate so far. A pass's start (`done == 0`) and finish
(`done == total`) always print, regardless of the throttle. Stateful — build
one `ProgressLogger` per `fit` or `calibrate` call; sharing an instance across
concurrent runs mixes their timers.

    fit(pipe, ps; exec = ExecutionConfig(progress = ProgressLogger()))
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

# ── Scheduling ────────────────────────────────────────────────────────────────

# A group's size in bytes: every cell's visibility, weight and flag. Orders
# concurrent groups largest-first.
function _group_bytes(group)
    bytes = 0
    for ms in values(group)
        bytes += sum(k -> length(ms[k]) * sizeof(eltype(ms[k])), (:visibility, :weight, :flag))
    end
    return bytes
end

# The most tasks `sched` runs at once. An upper bound — a scheduler whose chunk
# count follows the collection (`chunksize`), or that spawns one task per
# element (`chunking = false`), is bounded by the thread count instead.
max_tasks(::SerialScheduler) = 1
max_tasks(s::GreedyScheduler) = s.ntasks
max_tasks(s::Union{DynamicScheduler, StaticScheduler}) =
    (chunking_enabled(s) && has_nchunks(s)) ? nchunks(s) : Threads.nthreads()
