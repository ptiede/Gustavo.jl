# ── Execution configuration (run-wide, not per-step) ─────────────────────────

"""
    ExecutionConfig(; mem_fraction = 0.6, mem_budget = nothing, progress = nothing,
                    outer_executor = SerialScheduler(), inner_executor = DynamicScheduler())

Run-wide resources for `fit` and `calibrate`, shared by every pass over the
data; per-step options live on the steps, and the gauge is an argument of
`fit`. Two runs differing only in their `ExecutionConfig` solve the same
problem. Each scheduler is used exactly as configured, so the task count you
set is the concurrency you get.

- `mem_fraction` / `mem_budget` — the memory budget (`mem_budget` bytes if
  set, else `mem_fraction` of physical RAM) the outer scheduler's task count
  is checked against before any data is read. A configuration that cannot fit
  is an error rather than being silently reduced.
- `progress` — `(stage, done, total)` callback per completed scan of each
  pass.
- `outer_executor` — the across-scan scheduler, and thus how many scan
  groups are resident at once: `SerialScheduler()` by default; any other
  OhMyThreads `Scheduler` runs its task count of groups concurrently
  (`GreedyScheduler(; ntasks)` balances uneven scan lengths best). Groups
  are dispatched heaviest-first under every scheduler.
- `inner_executor` — the within-scan fan-out scheduler
  (`DynamicScheduler()` by default; `SerialScheduler()` for
  single-threaded within-scan solves).
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

    fit(pipe, ps; gauge, exec = ExecutionConfig(progress = ProgressLogger()))
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

# ── The memory budget ─────────────────────────────────────────────────────────

# Peak resident bytes charged for one group: every cell's visibility, weight
# and flag, with a 2.5× headroom factor over the measured ~2× peak of reading
# and correcting a group (decoded layers, corrected copies, GC slack).
function _group_charge(group)
    bytes = 0
    for ms in values(group)
        bytes += sum(k -> length(ms[k]) * sizeof(eltype(ms[k])), (:visibility, :weight, :flag))
    end
    return round(Int, 2.5 * bytes)
end

# The deterministic memory budget (bytes): an explicit `mem_budget` if given,
# else `mem_fraction` of TOTAL physical RAM — total, not available, so the
# group concurrency a run picks is reproducible on a given box.
function _memory_budget(exec::ExecutionConfig)
    exec.mem_budget === nothing || return Float64(exec.mem_budget)
    return exec.mem_fraction * Float64(Sys.total_memory())
end

# The most tasks `sched` runs at once: what the memory budget is checked
# against. An UPPER BOUND — a scheduler whose chunk count follows the
# collection (`chunksize`), or that spawns one task per element
# (`chunking = false`), is bounded by the thread count instead. Read-only: a
# scheduler is used exactly as its owner configured it, never rebuilt
# (`nchunks` and `chunksize` are mutually exclusive, so there is no lossless
# way to override one).
max_tasks(::SerialScheduler) = 1
max_tasks(s::GreedyScheduler) = s.ntasks
max_tasks(s::Union{DynamicScheduler, StaticScheduler}) =
    (chunking_enabled(s) && has_nchunks(s)) ? nchunks(s) : Threads.nthreads()

# The memory gate, checked before any data is read rather than imposed by
# capping the caller's scheduler: the outer scheduler decides how many groups
# run at once, so a configuration that cannot fit is an error, not something
# to silently rewrite.
function _check_memory_budget(charges, exec::ExecutionConfig)
    peak = maximum(charges; init = 0)
    peak > 0 || return nothing
    concurrent = min(max_tasks(outer_executor(exec)), max(length(charges), 1))
    # One group at a time is the floor: a single group larger than the budget is
    # the data's problem, not the schedule's, and there is no task count that
    # would fix it — run it and let the machine decide.
    concurrent <= 1 && return nothing
    budget = _memory_budget(exec)
    concurrent * peak <= budget && return nothing
    gib(x) = round(x / 2^30; digits = 2)
    throw(
        ArgumentError(
            "the outer executor runs up to $concurrent scan groups at once (≈ $(gib(concurrent * peak)) " *
                "GiB at the largest group's charge) but the memory budget is $(gib(budget)) GiB — " *
                "lower the outer scheduler's task count, or raise the ExecutionConfig's " *
                "mem_fraction/mem_budget."
        )
    )
end
