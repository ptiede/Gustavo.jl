# ── Executors: WHO runs the tasks (M7) ────────────────────────────────────────
#
# One tiny module, included before everything else, so every layer — the
# UVData/FITS decode fan-outs, the Fringe kernels' inner chunk loops, and the
# pipeline's group-level pass runner — draws its task spawning from the same
# seam. WHAT runs (chunking, fold order, budget admission) is fixed by the
# callers and identical under every executor, so θ and outputs are
# bit-identical across executors; the executor only decides which scheduler
# carries the tasks.
#
# The Dagger backend is a package EXTENSION (`GustavoDaggerExt`, activated by
# `using Dagger`): plain Gustavo never loads or installs Dagger.

module Executors

using OhMyThreads: tforeach
using Base.ScopedValues: ScopedValue, @with

export AbstractExecutor, ThreadsExecutor, DaggerExecutor
export with_executor, current_executor, exec_spawn, exec_fetch, exec_foreach

"""
    AbstractExecutor

The task-scheduling backend of a run: every task — scan-group passes AND the
nested per-baseline/decode loops — spawns through it.
[`ThreadsExecutor`](@ref) (the default) is plain `Threads.@spawn`;
[`DaggerExecutor`](@ref) runs the same task graph on Dagger (opt-in via the
`GustavoDaggerExt` extension — `using Dagger` activates it). Select per run
with `ExecutionConfig(executor = …)`, per scope with [`with_executor`](@ref),
or process-wide via `Executors.DEFAULT_EXECUTOR[]`. Results are bit-identical
across executors — chunking, budget admission, and fold order are fixed by the
callers; only the scheduler differs.
"""
abstract type AbstractExecutor end

"""
    ThreadsExecutor()

Plain `Threads.@spawn` task scheduling — the default.
"""
struct ThreadsExecutor <: AbstractExecutor end

"""
    DaggerExecutor()

Task scheduling on Dagger (`Dagger.@spawn`) via the `GustavoDaggerExt`
extension — load Dagger (`using Dagger`) to activate it. This backend exists
for FUTURE DISTRIBUTED (multi-machine) computing — the use case Dagger is
built for — and is NOT recommended in-process: on the single-process BT164A
benchmark (Dagger v0.21) it produced bit-identical results but +64%
wall-clock and ~+20 GB peak RSS over `ThreadsExecutor` (per-task scheduler
latency at our 0.1–2 s task granularity; thunk-state retention until GC
finalization, since bounded by a per-group forced collection). The seam keeps
the whole task graph expressible on it, so a multi-worker deployment only has
to revisit data movement (chunks/datadeps), not the pipeline.
"""
struct DaggerExecutor <: AbstractExecutor end

# The ambient executor: bound per run by the pass runner (or a driver) and
# inherited by every nested fan-out — including the FITS-ext decode loops,
# which cannot be reached by argument plumbing. Outside any `with_executor`
# scope the process-wide default applies.
const DEFAULT_EXECUTOR = Ref{AbstractExecutor}(ThreadsExecutor())
const CURRENT_EXECUTOR = ScopedValue{Union{Nothing, AbstractExecutor}}(nothing)

"""
    with_executor(f, ex::AbstractExecutor)

Run `f()` with `ex` as the ambient executor: every [`exec_spawn`](@ref) in the
dynamic extent — however deeply nested — uses it.
"""
with_executor(f, ex::AbstractExecutor) = @with(CURRENT_EXECUTOR => ex, f())

"""
    current_executor() -> AbstractExecutor

The ambient executor: the innermost [`with_executor`](@ref) binding, else the
process-wide default `Executors.DEFAULT_EXECUTOR[]` (a `ThreadsExecutor`
unless reassigned).
"""
current_executor() = something(CURRENT_EXECUTOR[], DEFAULT_EXECUTOR[])

"""
    exec_spawn(f; blocking = false) -> task handle

Spawn the 0-arg closure `f` on the ambient executor. Await with
[`exec_fetch`](@ref). Spawned tasks inherit the ambient executor, so spawns
nested inside `f` stay on the same backend.

`blocking = true` declares that `f` spends most of its time BLOCKED waiting on
tasks it spawns (a parent/fan-out task). Under Dagger such a task is spawned
at zero occupancy so it never reserves a processor slot while blocked (see
`GustavoDaggerExt` for why Dagger's automatic in-thunk-fetch handling is not
relied on); under Threads it is a no-op.
"""
exec_spawn(f; blocking::Bool = false) = _spawn(current_executor(), f, blocking)
_spawn(::ThreadsExecutor, f, blocking::Bool) = Threads.@spawn f()
_spawn(::DaggerExecutor, f, blocking::Bool) = error(
    "DaggerExecutor requires the GustavoDaggerExt extension — load Dagger " *
        "first (`using Dagger`)."
)

"""
    exec_fetch(t)

Await a handle from [`exec_spawn`](@ref) and return its value, rethrowing the
task's error. The Dagger extension unwraps Dagger's thunk-failure wrappers, so
callers see the same exception surface under both executors.
"""
exec_fetch(t::Task) = fetch(t)

"""
    exec_foreach(f, items; ntasks = length(items))

Run `f(item)` for each item, fanned out over at most `ntasks` contiguous
chunks on the ambient executor and awaited to completion. The chunking depends
only on `ntasks` (never on the executor or on runtime timing), so any
per-chunk state an `f` builds is identical across backends. Under
[`ThreadsExecutor`](@ref) the threading construct is OhMyThreads' `tforeach`;
other executors chunk explicitly and spawn per chunk.
"""
function exec_foreach(f, items; ntasks::Integer = length(items))
    n = length(items)
    n == 0 && return nothing
    nchunk = clamp(Int(ntasks), 1, n)
    if nchunk == 1
        foreach(f, items)
        return nothing
    end
    ex = current_executor()
    if ex isa ThreadsExecutor
        tforeach(f, items; ntasks = nchunk)
        return nothing
    end
    tasks = map(Iterators.partition(items, cld(n, nchunk))) do chunk
        _spawn(ex, () -> foreach(f, chunk), false)
    end
    foreach(exec_fetch, tasks)
    return nothing
end

end # module
