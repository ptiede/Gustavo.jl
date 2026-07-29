# ── Executors: the task backends a run spawns through ─────────────────────────
#
# A run parallelizes at two independent levels, each with its own executor (see
# `ExecutionConfig`):
#
#   • INNER — the within-scan fan-outs (per-baseline search, per-band refine and
#     adhoc, leaf transforms) run through an OhMyThreads `Scheduler`, via
#     `tforeach`. `scan_stream` fixes the scheduler's chunk count to the
#     memory-budgeted inner parallelism.
#   • OUTER — the across-scan group scheduling runs through the budget-admission
#     worker pool in `Streaming` (`_scheduled_map`), whose task backend is the
#     outer executor: [`ThreadsExecutor`](@ref) (the default, `Threads.@spawn`)
#     or [`DaggerExecutor`](@ref) for a distributed run.
#
# The Dagger backend is a package EXTENSION (`GustavoDaggerExt`, activated by
# `using Dagger`): plain Gustavo never loads or installs Dagger.

module Executors

using OhMyThreads: Scheduler, DynamicScheduler, StaticScheduler, SerialScheduler
using OhMyThreads.Schedulers: chunking_enabled, chunksplit, threadpool, nchunks

export ThreadsExecutor, DaggerExecutor, exec_fetch

"""
    ThreadsExecutor()

The default OUTER (across-scan) task backend: the group-scheduler worker pool
runs on `Threads.@spawn`. Select a different backend with
`ExecutionConfig(outer_executor = …)`.
"""
struct ThreadsExecutor end

"""
    DaggerExecutor()

OUTER (across-scan) task backend that schedules scan-group passes on Dagger
(`Dagger.@spawn`) via the `GustavoDaggerExt` extension — load Dagger
(`using Dagger`) and set `ExecutionConfig(outer_executor = DaggerExecutor())`.

This backend exists for FUTURE DISTRIBUTED (multi-machine) computing — the use
case Dagger is built for — and is NOT recommended in-process: on the
single-process BT164A benchmark (Dagger v0.21) it produced matching results but
+64% wall-clock and ~+20 GB peak RSS over `ThreadsExecutor` (per-task scheduler
latency at our 0.1–2 s task granularity; thunk-state retention until GC
finalization, bounded by a per-group forced collection). The seam keeps the
whole group schedule expressible on it, so a multi-worker deployment only has to
revisit data movement (chunks/datadeps), not the pipeline.
"""
struct DaggerExecutor end

# Process-wide default OUTER executor: the group-scheduler backend a run uses
# when `ExecutionConfig` is built without one. A default SOURCE read once when
# the config is built — not an ambient scope; every level receives its executor
# explicitly thereafter.
const DEFAULT_EXECUTOR = Ref{Any}(ThreadsExecutor())

"""
    exec_fetch(handle)

Await a handle from the outer scheduler's task backend and return its value,
rethrowing the task's own exception. The Dagger extension adds a method for its
handle type that unwraps Dagger's thunk-failure wrappers, so callers see the
same exception surface under both backends.
"""
exec_fetch(t::Task) = fetch(t)

"""
    Executors._spawn(ex, f, blocking::Bool) -> handle

Spawn the zero-argument `f` on the outer backend `ex` and return a handle to
await with [`exec_fetch`](@ref). `blocking` declares that `f` spends most of its
time BLOCKED awaiting tasks it spawns (a group parent that fans out decode /
search / refine work); a backend that reserves processor slots can use it to
avoid starvation. `ThreadsExecutor` spawns with `Threads.@spawn` directly, so
only the Dagger backend implements this.

The least-specific method errors: it must stay more general than every backend's
method — those dispatch on their own concrete executor type — because a stub
sharing a backend's signature would be overwritten when the extension loads,
which precompilation forbids.
"""
_spawn(ex, f, blocking::Bool) = error(
    "$(nameof(typeof(ex))) has no spawn backend loaded — load the package " *
        "providing its extension (`using Dagger` for DaggerExecutor)."
)

"""
    with_nchunks(sched::OhMyThreads.Scheduler, n::Integer) -> Scheduler

The inner `sched` reconfigured to fan out over `n` chunks — the per-block inner
parallelism the stream budgets against its memory cap. ChunkSplitters clamps `n`
down to each collection's length, so one scheduler serves every within-scan
fan-out regardless of how many leaves or baselines it spans. A `SerialScheduler`
(or any scheduler with chunking disabled) passes through unchanged.
"""
with_nchunks(s::SerialScheduler, ::Integer) = s
function with_nchunks(s::DynamicScheduler, n::Integer)
    chunking_enabled(s) || return s
    return DynamicScheduler(;
        nchunks = max(1, Int(n)), split = chunksplit(s), threadpool = threadpool(s),
    )
end
function with_nchunks(s::StaticScheduler, n::Integer)
    chunking_enabled(s) || return s
    return StaticScheduler(; nchunks = max(1, Int(n)), split = chunksplit(s))
end

end # module
