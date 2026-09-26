# The `off1` index table of a `ComponentPlan`, so tests can address θ slots by
# `(ant, feed, tseg, fseg)`: the absolute θ index of each block's first
# parameter (0 = no block for that feed).
const _CALoff = Gustavo.Calibration

function plan_off1(plan)
    _, _, nfseg, ntseg, nant = plan.shape
    off = zeros(Int, nant, 2, ntseg, nfseg)
    for a in 1:nant, feed in 1:2, ts in 1:ntseg, fs in 1:nfseg
        node = _CALoff._feed_node(plan.tying, feed)
        off[a, feed, ts, fs] = node == 0 ? 0 : _CALoff._block_index(plan, node, fs, ts, a)
    end
    return off
end
