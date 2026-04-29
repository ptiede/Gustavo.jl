"""
    apply(f, uvset::UVSet) -> UVSet

Walk the leaves of `uvset` and rebuild a tree with each leaf replaced by
`f(leaf, partition_info, root_meta)`, where `partition_info` is the leaf's
`metadata::PartitionInfo` and `root_meta` is `metadata(uvset)`. Tree shape is
preserved. Built-in reducers (`TimeAverage`, `BandpassCorrection`) implement
this signature.
"""
function apply(f, uvset::UVSet)
    root_meta = DimensionalData.metadata(uvset)
    src = DimensionalData.branches(uvset)
    new_branches = DimensionalData.TreeDict()
    sizehint!(new_branches, length(src))
    for (k, leaf) in src
        new_branches[k] = f(leaf, DimensionalData.metadata(leaf), root_meta)
    end
    return DimensionalData.rebuild(uvset; branches = new_branches)
end
