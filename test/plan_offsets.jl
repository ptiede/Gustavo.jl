# Reconstruct the legacy `off1`/`off2` index tables from a `ComponentPlan`, so
# tests can address θ slots by `(ant, feed, tseg, fseg)` as before. `off1` is the
# absolute θ index of each block's first parameter (0 = no block for that feed);
# `off2` is the secondary (`ReferenceRelative` partner's relative) block.
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

function plan_off2(plan)
    _, _, nfseg, ntseg, nant = plan.shape
    off = zeros(Int, nant, 2, ntseg, nfseg)
    for a in 1:nant, feed in 1:2, ts in 1:ntseg, fs in 1:nfseg
        node2 = _CALoff._feed_node2(plan.tying, feed)
        off[a, feed, ts, fs] = node2 == 0 ? 0 : _CALoff._block_index(plan, node2, fs, ts, a)
    end
    return off
end
