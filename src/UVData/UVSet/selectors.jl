"""
    select_source(uvset::UVSet, name::AbstractString) -> UVSet

Sub-`UVSet` containing only leaves whose `partition_info.source_name == name`.
Root-level metadata is shared with the parent.
"""
function select_source(uvset::UVSet, name::AbstractString)
    new_branches = DimensionalData.TreeDict()
    for (k, leaf) in DimensionalData.branches(uvset)
        DimensionalData.metadata(leaf).source_name == name || continue
        new_branches[k] = leaf
    end
    return DimensionalData.rebuild(uvset; branches = new_branches)
end

"""
    select_scan(uvset::UVSet, source::AbstractString, scan_name) -> DimTree

Return the single leaf for `(source, scan_name)`. `scan_name` may be a
string (the leaf's primary scan label) or an integer (treated as
`string(scan_name)` for back-compat with prior numeric scan handles).
Errors if no such leaf exists.
"""
function select_scan(uvset::UVSet, source::AbstractString, scan_name::AbstractString)
    for (_, leaf) in DimensionalData.branches(uvset)
        m = DimensionalData.metadata(leaf)
        if m.source_name == source && m.scan_name == scan_name
            return leaf
        end
    end
    return error("select_scan: no leaf for source=$(source), scan_name=$(scan_name)")
end

select_scan(uvset::UVSet, source::AbstractString, scan_idx::Integer) =
    select_scan(uvset, source, string(scan_idx))

"""
    select_partition(uvset::UVSet; source=nothing, scan=nothing,
                     spw=nothing, intent=nothing) -> UVSet

Predicate filter over leaves. Each kwarg, when not `nothing`, must equal
the corresponding `partition_info` field for a leaf to be retained.
"""
function select_partition(
        uvset::UVSet;
        source::Union{Nothing, AbstractString} = nothing,
        scan::Union{Nothing, AbstractString, Integer} = nothing,
        spw::Union{Nothing, AbstractString} = nothing,
        intent::Union{Nothing, AbstractString} = nothing,
    )
    scan_label = scan === nothing ? nothing :
        (scan isa Integer ? string(scan) : String(scan))
    new_branches = DimensionalData.TreeDict()
    for (k, leaf) in DimensionalData.branches(uvset)
        m = DimensionalData.metadata(leaf)
        source === nothing || m.source_name == source || continue
        scan_label === nothing || m.scan_name == scan_label || continue
        spw === nothing || m.spw_name == spw || continue
        intent === nothing || m.intent == intent || continue
        new_branches[k] = leaf
    end
    return DimensionalData.rebuild(uvset; branches = new_branches)
end

"""
    select_station(uvset::UVSet, name::AbstractString) -> UVSet

Per-leaf filter retaining only baselines that touch `name`. Leaves where
the station did not participate are dropped.
"""
select_station(uvset::UVSet, name::AbstractString) =
    _filter_uvset(uvset, (Station = name,))

"""
    select_baseline(uvset::UVSet, label) -> UVSet
    select_baseline(uvset::UVSet, (a, b)) -> UVSet

Per-leaf filter to a single baseline. Leaves missing the baseline are
dropped.
"""
select_baseline(uvset::UVSet, label::AbstractString) =
    _filter_uvset(uvset, (Baseline = label,))
select_baseline(uvset::UVSet, bl::Tuple{AbstractString, AbstractString}) =
    select_baseline(uvset, string(bl[1], "-", bl[2]))

"""
    time_window(uvset::UVSet, t_lo, t_hi) -> UVSet

Per-leaf filter restricting `Ti` to `t_lo ≤ t ≤ t_hi`.
"""
time_window(uvset::UVSet, t_lo::Real, t_hi::Real) =
    _filter_uvset(uvset, (Ti = DimensionalData.Lookups.Where(t -> t_lo <= t <= t_hi),))

# Walk the tree applying per-leaf filters; drop leaves the filter empties.
function _filter_uvset(uvset::UVSet, kw::NamedTuple)
    new_branches = DimensionalData.TreeDict()
    for (k, leaf) in DimensionalData.branches(uvset)
        filtered = _filter_partition(leaf, kw)
        filtered === nothing && continue
        new_branches[k] = filtered
    end
    return DimensionalData.rebuild(uvset; branches = new_branches)
end

# Apply Station/Baseline/Ti/Pol/IF selectors to a single leaf DimTree.
# Returns `nothing` if the filter would empty the leaf.
function _filter_partition(leaf::DimensionalData.DimTree, kw::NamedTuple)
    bls = baselines(leaf)
    nbl = length(bls.pairs)

    keep_bl = trues(nbl)
    if haskey(kw, :Station)
        name = String(kw.Station)
        @inbounds for bi in 1:nbl
            keep_bl[bi] &= (bls.ant1_names[bi] == name) || (bls.ant2_names[bi] == name)
        end
    end
    if haskey(kw, :Baseline)
        target = String(kw.Baseline isa AbstractString ? kw.Baseline : kw.Baseline.val)
        @inbounds for bi in 1:nbl
            keep_bl[bi] &= bls.labels[bi] == target
        end
    end
    bl_inds = findall(keep_bl)
    isempty(bl_inds) && return nothing

    vis_l = leaf[:vis]
    weights_l = leaf[:weights]
    flag_l = leaf[:flag]
    uvw_l = leaf[:uvw]

    nti_full = length(obs_time(leaf))
    ti_inds = if haskey(kw, :Ti)
        ti_lookup = lookup(vis_l, Ti)
        sel = DimensionalData.Lookups.selectindices(ti_lookup, kw.Ti)
        sel isa Integer ? Int[sel] : collect(sel)
    else
        collect(1:nti_full)
    end
    isempty(ti_inds) && return nothing

    vis_p = parent(vis_l)[ti_inds, bl_inds, :, :]
    w_p = parent(weights_l)[ti_inds, bl_inds, :, :]
    flag_p = parent(flag_l)[ti_inds, bl_inds, :, :]
    uvw_p = parent(uvw_l)[ti_inds, bl_inds, :]
    obs_time_new = obs_time(leaf)[ti_inds]

    new_labels = bls.labels[bl_inds]
    pol_dim = dims(vis_l, Pol)
    if_dim = dims(vis_l, IF)
    vis_da = DimArray(vis_p, (Ti(obs_time_new), Baseline(new_labels), pol_dim, if_dim))
    weights_da = DimArray(w_p, dims(vis_da))
    flag_da = DimArray(flag_p, dims(vis_da))
    uvw_da = DimArray(uvw_p, (Ti(obs_time_new), Baseline(new_labels), UVW(["U", "V", "W"])))

    pol_if_kw = NamedTuple(kk => v for (kk, v) in pairs(kw) if kk in (:Pol, :IF))
    if !isempty(pol_if_kw)
        vis_da = getindex(vis_da; pol_if_kw...)
        weights_da = getindex(weights_da; pol_if_kw...)
        flag_da = getindex(flag_da; pol_if_kw...)
    end

    new_pairs = bls.pairs[bl_inds]
    new_unique_codes = bls.unique_codes[bl_inds]
    new_lookup = Dict(c => i for (i, c) in enumerate(new_unique_codes))
    new_ant1 = bls.ant1_names[bl_inds]
    new_ant2 = bls.ant2_names[bl_inds]
    new_baselines = BaselineIndex(
        eltype(bls.codes)[], new_pairs, new_lookup, new_unique_codes,
        new_labels, new_ant1, new_ant2,
    )

    ti_remap = Dict(orig => newi for (newi, orig) in enumerate(ti_inds))
    bi_remap = Dict(orig => newi for (newi, orig) in enumerate(bl_inds))
    record_order_new = Tuple{Int, Int}[]
    rec_keep_inds = Int[]
    for (rec_i, (ti_orig, bi_orig)) in enumerate(record_order(leaf))
        new_ti = get(ti_remap, ti_orig, 0)
        new_bi = get(bi_remap, bi_orig, 0)
        (new_ti == 0 || new_bi == 0) && continue
        push!(record_order_new, (new_ti, new_bi))
        push!(rec_keep_inds, rec_i)
    end

    date_param_new = if isempty(rec_keep_inds) || size(date_param(leaf), 1) == 0
        zeros(eltype(date_param(leaf)), 0, size(date_param(leaf), 2))
    else
        date_param(leaf)[rec_keep_inds, :]
    end
    extras_new = if isempty(rec_keep_inds)
        NamedTuple()
    else
        NamedTuple{keys(extra_columns(leaf))}(
            ntuple(i -> extra_columns(leaf)[i][rec_keep_inds], length(extra_columns(leaf)))
        )
    end

    return _build_leaf(
        vis_da, weights_da, uvw_da, flag_da;
        partition_info = update(
            DimensionalData.metadata(leaf);
            baselines = new_baselines,
            record_order = record_order_new,
            date_param = date_param_new,
            extra_columns = extras_new,
        ),
    )
end


"""
    merge_uvsets(uvsets::UVSet...) -> UVSet

Strictly combine multiple UVSets into one multi-source UVSet. All inputs
must share identical array-wide metadata: `antennas`, `array_config`,
`array_obs`, and the same polarization products on the `Pol` axis. The
root `primary_cards` is taken from the first input verbatim — write-back
round-trips after `select_source(merged, name)`.

Branch keys must be unique across inputs (collisions would indicate two
leaves describing the same `(source, scan, sub_scan)` triple, which is not
representable). For sub-arrays observing the same `(source, scan)`
simultaneously, set `sub_scan_name` distinctly on each leaf to disambiguate
— mirrors xradio's `SUB_SCAN_NUMBER` partitioning.
"""
function merge_uvsets(uvsets::UVSet...)
    isempty(uvsets) && error("merge_uvsets: at least one UVSet required")
    length(uvsets) == 1 && return only(uvsets)
    base = first(uvsets)
    base_root = DimensionalData.metadata(base)
    base_pp = pol_products(base)
    for (i, u) in enumerate(uvsets[2:end])
        m = DimensionalData.metadata(u)
        m.antennas == base_root.antennas ||
            error("merge_uvsets: antenna table mismatch (input $(i + 1) vs 1)")
        m.array_config == base_root.array_config ||
            error("merge_uvsets: array_config mismatch (input $(i + 1) vs 1)")
        m.array_obs == base_root.array_obs ||
            error("merge_uvsets: array_obs mismatch (input $(i + 1) vs 1)")
        pol_products(u) == base_pp ||
            error("merge_uvsets: polarization product mismatch (input $(i + 1) vs 1)")
    end

    new_branches = DimensionalData.TreeDict()
    for u in uvsets
        for (k, leaf) in DimensionalData.branches(u)
            haskey(new_branches, k) &&
                error("merge_uvsets: duplicate partition key $(k)")
            new_branches[k] = leaf
        end
    end
    return DimensionalData.rebuild(base; branches = new_branches)
end
