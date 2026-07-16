# Gustavo.Graph — distributed map/reduce substrate (Stage-1 refactor, Step 5/6).
#
# The SerialExecutor is the reference: pmap/pmapreduce under it must reproduce a
# plain apply / mapleaves / manual-reduce. The DaggerExecutor (Dagger owns the
# parallelism) must return bit-identical results in deterministic partition
# order. Reuses `_build_fringe_uvset` from test_pipeline.jl.

using Gustavo: pmap, pmapreduce, SerialExecutor, DaggerExecutor
import Gustavo.UVData as UV
import DimensionalData

# Clone a 1-scan fixture into a 2-scan set (two (source, scan) load-groups) so
# the Dagger path runs more than one task and cross-group ordering is exercised.
function _two_scan_uvset()
    uv1, _ = _build_fringe_uvset(nbands = 3, nchan = 4)
    new_branches = DimensionalData.TreeDict()
    for (k, leaf) in UV.branches(uv1)
        new_branches[k] = leaf                                   # scan "1"
    end
    for (_, leaf) in UV.branches(uv1)
        info2 = UV.update(UV.metadata(leaf); scan_name = "2")     # scan "2" clone
        leaf2 = UV._build_leaf(leaf[:vis], leaf[:weights], leaf[:uvw]; partition_info = info2)
        new_branches[UV.partition_key(info2)] = leaf2
    end
    return DimensionalData.rebuild(uv1; branches = new_branches)
end

# A leaf→leaf transform (tree-out): scale visibilities by 2.
_scale(leaf) = UV._build_leaf(
    2.0f0 .* leaf[:vis], leaf[:weights], leaf[:uvw];
    partition_info = UV.metadata(leaf),
)

@testset "Graph: load_groups groups sibling bands by (source, scan)" begin
    uvset = _two_scan_uvset()
    groups = Gustavo.Graph.load_groups(uvset)
    @test length(groups) == 2                 # two scans
    @test all(length(g) == 3 for g in groups) # three bands each
    # Every leaf appears exactly once across the groups, keys preserved.
    gkeys = sort([kv.first for g in groups for kv in g])
    @test gkeys == sort(collect(keys(UV.branches(uvset))))
end

@testset "Graph: pmap tree-out reproduces apply (Serial)" begin
    uvset = _two_scan_uvset()
    ua = UV.apply(_scale, uvset)
    up = pmap(_scale, uvset)                   # SerialExecutor default
    @test up isa UV.UVSet
    @test collect(keys(UV.branches(up))) == collect(keys(UV.branches(ua)))
    for k in keys(UV.branches(uvset))
        @test parent(UV.branches(up)[k][:vis]) == parent(UV.branches(ua)[k][:vis])
    end
end

@testset "Graph: pmap dict-out reproduces mapleaves" begin
    uvset = _two_scan_uvset()
    node = leaf -> sum(abs2, leaf[:vis])
    dm = UV.mapleaves(node, uvset)
    dp = pmap(node, uvset)
    @test dp isa AbstractDict
    @test collect(keys(dp)) == collect(keys(dm))
    @test all(dp[k] == dm[k] for k in keys(dm))
end

@testset "Graph: node coords carry partition info" begin
    uvset = _two_scan_uvset()
    scans = pmap((data, coords) -> coords.info.scan_name, uvset)
    @test Set(values(scans)) == Set(("1", "2"))
end

@testset "Graph: pmapreduce (tree vs single vs init)" begin
    uvset = _two_scan_uvset()
    ncell = leaf -> length(leaf[:vis])
    manual = sum(length(leaf[:vis]) for (_, leaf) in UV.branches(uvset))
    @test pmapreduce(ncell, +, uvset) == manual                       # tree (default)
    @test pmapreduce(ncell, +, uvset; tree = false) == manual         # single-node fold
    @test pmapreduce(ncell, +, uvset; init = 1000) == manual + 1000   # seeded
end

@testset "Graph: DaggerExecutor equals SerialExecutor" begin
    uvset = _two_scan_uvset()
    ncell = leaf -> length(leaf[:vis])
    manual = sum(length(leaf[:vis]) for (_, leaf) in UV.branches(uvset))

    # Reduce: Dagger owns the parallelism, result matches serial/manual.
    @test pmapreduce(ncell, +, uvset; executor = DaggerExecutor()) == manual

    # Map (tree-out): Dagger result is bit-identical and in partition order.
    ua = UV.apply(_scale, uvset)
    ud = pmap(_scale, uvset; executor = DaggerExecutor())
    @test ud isa UV.UVSet
    @test collect(keys(UV.branches(ud))) == collect(keys(UV.branches(ua)))
    for k in keys(UV.branches(uvset))
        @test parent(UV.branches(ud)[k][:vis]) == parent(UV.branches(ua)[k][:vis])
    end

    # Map (dict-out): same keys and values under Dagger.
    node = leaf -> sum(abs2, leaf[:vis])
    ds = pmap(node, uvset; executor = SerialExecutor())
    dd = pmap(node, uvset; executor = DaggerExecutor())
    @test collect(keys(dd)) == collect(keys(ds))
    @test all(dd[k] == ds[k] for k in keys(ds))
end
