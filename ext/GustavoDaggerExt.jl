# ── GustavoDaggerExt: the Dagger backend of the executor seam ────────────────
#
# Activated by `using Dagger`. Implements `DaggerExecutor`'s spawn/fetch on
# Dagger's eager scheduler; everything else about a run — chunking, budget
# admission, fold order — lives executor-independently in Gustavo, so results
# are bit-identical to `ThreadsExecutor`.
#
# This backend exists for FUTURE DISTRIBUTED (multi-machine) runs. In-process
# it is measurably slower than Threads at Gustavo's task granularity (see the
# `DaggerExecutor` docstring); the findings below were profiled on v0.21:
#
# - In-thunk `fetch` of nested eager tasks DEADLOCKS at ≤ 2 threads — Dagger's
#   own `Sch.thunk_yield` (which should make this safe automatically) fails to
#   let the children schedule; in-thunk SPAWN alone is fine. Hence `blocking`
#   parents are spawned at zero occupancy, which sidesteps the reclaim
#   machinery entirely. Remove when fixed upstream.
# - Marking EVERY task zero-occupancy makes the scheduler co-schedule the
#   whole pool and thrash its per-processor queues (~1000 lock conflicts per
#   group task, 2× compute slowdown) — hence the split: only blocking parents
#   are zero-occupancy, compute leaves keep the default.
# - Completed thunks (and the scan-sized data their closures capture) are
#   freed only after a GC run collects the dropped `DTask` handles, their
#   finalizers enqueue scheduler cleanup, and the scheduler processes it —
#   Julia's GC pacing lags that pipeline, so RSS climbs by ~a cube per scan.
#   The group scheduler (`Fringe._scheduled_map_dagger`) forces a collection
#   as each group retires.

module GustavoDaggerExt

import Gustavo.Executors as Executors
using Gustavo.Executors: DaggerExecutor
using Dagger
using Distributed: Distributed, CapturedException

function Executors._spawn(ex::DaggerExecutor, f, blocking::Bool)
    if blocking
        return Dagger.@spawn occupancy = Dict(Dagger.ThreadProc => 0) f()
    else
        return Dagger.@spawn f()
    end
end

function Executors.exec_fetch(t::Dagger.DTask)
    try
        return fetch(t)
    catch err
        rethrow(_unwrap_dagger(err))
    end
end

# Peel Dagger's transport wrappers (DTaskFailedException / RemoteException /
# CapturedException, possibly stacked) down to the task's own exception, so
# callers see the same error surface under both executors.
function _unwrap_dagger(err)
    while true
        if err isa Dagger.DTaskFailedException
            err = err.ex
        elseif err isa Distributed.RemoteException
            err = err.captured
        elseif err isa CapturedException
            err = err.ex
        else
            return err
        end
    end
end

end # module
