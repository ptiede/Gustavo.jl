"""
    BandpassDataset

Internal adapter consumed by the bandpass solver. Wraps the dense
`(Scan, Baseline, Pol, IF)` cubes assembled from a `UVSet` (via
`_to_bandpass_dataset`) together with the observation-level globals the
solver needs (`antennas`, `metadata`, `scans`) and the union
`BaselineIndex`.

"""
struct BandpassDataset{TVis, TW, TUVW, TBl, TAnt, TMeta, TFS <: UVData.AbstractFrequencySetup, TScans}
    vis::TVis
    weights::TW
    uvw::TUVW
    baselines::TBl
    antennas::TAnt
    metadata::TMeta
    freq_setup::TFS
    scans::TScans
end

UVData.pol_products(data::BandpassDataset) = UVData.pol_products(data.vis)
UVData.freq_setup(data::BandpassDataset) = data.freq_setup
UVData.channel_freqs(data::BandpassDataset) = UVData.channel_freqs(data.freq_setup)

"""
    _group_leaves_by_spw(uvset::UVSet) -> OrderedDict{String, Vector{Symbol}}

Walk leaves and group their branch keys by `spw_name` in first-seen
order. Used by `solve_bandpass(::UVSet)` to dispatch one solver run
per SPW.
"""
function _group_leaves_by_spw(uvset::UVSet)
    groups = OrderedDict{String, Vector{Symbol}}()
    for (key, leaf) in DimensionalData.branches(uvset)
        spw = DimensionalData.metadata(leaf).spw_name
        push!(get!(() -> Symbol[], groups, spw), key)
    end
    return groups
end

"""
    _uvset_with_branches(uvset::UVSet, leaf_keys) -> UVSet

Rebuild a UVSet with only the named branches. Goes through
`DimensionalData.rebuild` so format-extension shadow state (e.g. FITS
primary cards) follows automatically via `_propagate_extension_state!`.
"""
function _uvset_with_branches(uvset::UVSet, leaf_keys)
    full = DimensionalData.branches(uvset)
    sub = DimensionalData.TreeDict()
    for k in leaf_keys
        sub[k] = full[k]
    end
    return DimensionalData.rebuild(uvset; branches = sub)
end

"""
    _to_bandpass_dataset(uvset::UVSet) -> BandpassDataset

Assemble a dense `(Scan, Baseline_union, Pol, IF)` averaged cube from a
`UVSet` by inverse-variance averaging each leaf's `Ti` axis (one row per
leaf / scan) and unioning baselines across leaves.
"""
function _to_bandpass_dataset(uvset::UVSet)
    branches_d = DimensionalData.branches(uvset)
    isempty(branches_d) && error("_to_bandpass_dataset: empty UVSet")

    # Asserts every leaf shares one frequency setup; multi-SPW UVSets
    # should be split via `_group_leaves_by_spw` upstream (see
    # `solve_bandpass(::UVSet)`).
    fs = UVData.freq_setup(uvset)

    # Concretely-typed leaf vector keeps inner loops type-stable despite the
    # `OrderedDict{Symbol, AbstractDimTree}` abstract eltype.
    leaf_list = collect(values(branches_d))

    union_pair_set = Set{Tuple{Int, Int}}()
    for p in leaf_list
        union!(union_pair_set, baselines(p).pairs)
    end
    union_pairs = sort!(collect(union_pair_set))
    union_lookup = Dict(p => i for (i, p) in enumerate(union_pairs))
    pair_to_ant1 = Dict{Tuple{Int, Int}, String}()
    pair_to_ant2 = Dict{Tuple{Int, Int}, String}()
    for p in leaf_list
        for (i, pr) in enumerate(baselines(p).pairs)
            pair_to_ant1[pr] = baselines(p).ant1_names[i]
            pair_to_ant2[pr] = baselines(p).ant2_names[i]
        end
    end
    union_ant1 = [pair_to_ant1[pr] for pr in union_pairs]
    union_ant2 = [pair_to_ant2[pr] for pr in union_pairs]
    union_labels = string.(union_ant1, "-", union_ant2)
    union_baselines = UVData.BaselineIndex(
        Tuple{Int, Int}[], union_pairs, union_lookup,
        union_labels, union_ant1, union_ant2,
    )
    nbl = length(union_pairs)

    root = DimensionalData.metadata(uvset)
    array_obs = root.array_obs
    nchan = UVData.nchannels(fs)

    # Enumerate leaves in (scan_window.lower, sub_scan_name) sort order so the
    # `Scan` dim has a stable ordering even with sub-arrays. After this, each
    # leaf maps to exactly one Scan slot at position `sid`.
    sorted_leaves = sort(
        leaf_list;
        by = leaf -> (UVData.scan_window(leaf)[1], DimensionalData.metadata(leaf).sub_scan_name),
    )
    nscan = length(sorted_leaves)
    scan_windows = [UVData.scan_window(leaf) for leaf in sorted_leaves]
    scan_centers = [(lo + hi) / 2 for (lo, hi) in scan_windows]

    first_part = first(sorted_leaves)
    pol_labels = pol_products(first_part)
    npol = length(pol_labels)
    Tvis = eltype(parent(first_part[:vis]))
    Tw = eltype(parent(first_part[:weights]))
    Tuvw = eltype(parent(first_part[:uvw]))

    # Memory layout: vis/weights = (Frequency, Ti, Baseline, Pol);
    # uvw = (Ti, Baseline, UVW). Frequency fastest, pol slowest.
    V_num = zeros(Tvis, nchan, nscan, nbl, npol)
    W_sum = zeros(Tw, nchan, nscan, nbl, npol)
    UVW_num = zeros(Tuvw, nscan, nbl, 3)
    UVW_w = zeros(Tw, nscan, nbl)

    for (sid, part) in enumerate(sorted_leaves)
        # Function barrier: extract concretely-typed parent arrays and pass
        # to a kernel so the inner-loop scalar accesses don't box.
        _accumulate_avg_into!(
            V_num, W_sum, UVW_num, UVW_w,
            parent(part[:vis]), parent(part[:weights]), parent(part[:uvw]),
            baselines(part).pairs, union_lookup, sid,
        )
    end

    V = similar(V_num)
    @inbounds for k in eachindex(V)
        V[k] = W_sum[k] > 0 ? V_num[k] / W_sum[k] : Tvis(NaN, NaN)
    end

    UVW_out = fill(Tuvw(NaN), nscan, nbl, 3)
    @inbounds for s in 1:nscan, bi in 1:nbl
        if UVW_w[s, bi] > 0
            for k in 1:3
                UVW_out[s, bi, k] = UVW_num[s, bi, k] / UVW_w[s, bi]
            end
        end
    end

    vis_avg = DimensionalData.DimArray(
        V,
        (
            Frequency(UVData.channel_freqs(fs)), Ti(scan_centers),
            Baseline(union_labels), Pol(pol_labels),
        ),
    )
    weights_avg = DimensionalData.DimArray(W_sum, dims(vis_avg))
    uvw_avg = DimensionalData.DimArray(
        UVW_out,
        (Ti(scan_centers), Baseline(union_labels), UVW(["U", "V", "W"])),
    )

    return BandpassDataset(
        vis_avg, weights_avg, uvw_avg, union_baselines,
        UVData.union_antennas(uvset), array_obs, fs, scan_windows,
    )
end

# Type-stable inner kernel for `_to_bandpass_dataset`. Accumulates one
# leaf's contribution to the union (Scan, Baseline, Pol, IF) cube.
function _accumulate_avg_into!(
        V_num::AbstractArray{Tvis, 4},      # (Frequency, Ti, Baseline, Pol)
        W_sum::AbstractArray{Tw, 4},        # (Frequency, Ti, Baseline, Pol)
        UVW_num::AbstractArray{Tuvw, 3},    # (Ti, Baseline, UVW)
        UVW_w::AbstractArray{Tw, 2},        # (Ti, Baseline)
        vis_p::AbstractArray{Tvis, 4},      # leaf: (Frequency, Ti, Baseline, Pol)
        w_p::AbstractArray{Tw, 4},
        uvw_p::AbstractArray{Tuvw, 3},      # leaf: (Ti, Baseline, UVW)
        pairs_local, union_lookup, sid::Integer,
    ) where {Tvis, Tw, Tuvw}
    nchan, nti_p, nbl_p, npol = size(vis_p)
    @inbounds for ti in 1:nti_p, bi_local in 1:nbl_p
        bi = union_lookup[pairs_local[bi_local]]
        tot_w = zero(Tw)
        for p in 1:npol, c in 1:nchan
            w = w_p[c, ti, bi_local, p]
            v = vis_p[c, ti, bi_local, p]
            (w > 0 && isfinite(w) && isfinite(real(v))) || continue
            V_num[c, sid, bi, p] += w * v
            W_sum[c, sid, bi, p] += w
            tot_w += w
        end
        (tot_w > 0 && isfinite(tot_w)) || continue
        for k in 1:3
            u = uvw_p[ti, bi_local, k]
            isfinite(u) || continue
            UVW_num[sid, bi, k] += tot_w * u
        end
        UVW_w[sid, bi] += tot_w
    end
    return nothing
end

"""
    baseline_visibilities(data::BandpassDataset, bl::Tuple{String,String})

Return visibilities for a single baseline as a `DimArray` of dims
`(Scan, Pol, IF)`.
"""
baseline_visibilities(data::BandpassDataset, bl::Tuple{String, String}) =
    _baseline_slice(data.vis, data, bl, :vis)

baseline_weights(data::BandpassDataset, bl::Tuple{String, String}) =
    _baseline_slice(data.weights, data, bl, :weights)

function _baseline_slice(A::DimensionalData.AbstractDimArray, data::BandpassDataset, bl::Tuple{String, String}, kind::Symbol)
    bi = baseline_number_in_dataset(data, bl)
    metadata = Dict(
        :baseline => join(bl, "-"),
        :Sites => bl,
        :kind => kind,
        :pol_products => collect(lookup(A, Pol)),
        :channel_freqs => UVData.channel_freqs(data.freq_setup),
        :band_center_frequency => band_center_frequency_dataset(data),
    )
    return rebuild(@view(A[Baseline = bi]); metadata = metadata)
end

function baseline_number_in_dataset(data::BandpassDataset, bl::Tuple{String, String})
    a_idx = findfirst(==(bl[1]), data.antennas.name)
    b_idx = findfirst(==(bl[2]), data.antennas.name)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl")
    bi = findfirst(==((a_idx, b_idx)), data.baselines.pairs)
    isnothing(bi) && error("Baseline $bl not in dataset")
    return bi
end

band_center_frequency_dataset(data::BandpassDataset) =
    (first(UVData.channel_freqs(data.freq_setup)) + last(UVData.channel_freqs(data.freq_setup))) / 2

scan_time_centers_dataset(data::BandpassDataset) =
    [(lo + hi) / 2 for (lo, hi) in data.scans]

"""
    baseline_visibilities(uvset::UVSet, bl::Tuple{String,String})

Return an `OrderedDict{Symbol, DimArray}` mapping each branch key to the
visibility slice on baseline `bl` for that leaf.
"""
function baseline_visibilities(uvset::UVSet, bl::Tuple{String, String})
    out = OrderedDict{Symbol, Any}()
    for (key, leaf) in DimensionalData.branches(uvset)
        bl[1] in baselines(leaf).ant1_names || bl[1] in baselines(leaf).ant2_names || continue
        bl[2] in baselines(leaf).ant1_names || bl[2] in baselines(leaf).ant2_names || continue
        try
            out[key] = _partition_baseline_slice(leaf, bl, :vis)
        catch
        end
    end
    return out
end

function baseline_weights(uvset::UVSet, bl::Tuple{String, String})
    out = OrderedDict{Symbol, Any}()
    for (key, leaf) in DimensionalData.branches(uvset)
        bl[1] in baselines(leaf).ant1_names || bl[1] in baselines(leaf).ant2_names || continue
        bl[2] in baselines(leaf).ant1_names || bl[2] in baselines(leaf).ant2_names || continue
        try
            out[key] = _partition_baseline_slice(leaf, bl, :weights)
        catch
        end
    end
    return out
end

function _partition_baseline_slice(leaf::DimensionalData.AbstractDimTree, bl::Tuple{String, String}, kind::Symbol)
    bi = UVData.baseline_number(leaf, bl)
    A = kind === :vis ? leaf[:vis] : leaf[:weights]
    return @view(A[Baseline = bi])
end

"""
    getindex(data::BandpassDataset; kwargs...)

Slice the visibility, weights, and UVW arrays by named DD selectors.
"""
function Base.getindex(data::BandpassDataset; kwargs...)
    return DimStack(
        (
            vis = _slice_dim(data.vis; kwargs...),
            weights = _slice_dim(data.weights; kwargs...),
            uvw = _slice_dim(data.uvw; kwargs...),
        )
    )
end

function _slice_dim(A::DimensionalData.AbstractDimArray; kwargs...)
    keys_keep = Tuple(k for k in keys(kwargs) if hasdim(A, name2dim(k)))
    isempty(keys_keep) && return A
    keep = NamedTuple{keys_keep}(values(kwargs))
    return getindex(A; keep...)
end

"""
    wrap_gain_solutions(gains, data::BandpassDataset; pol_keys=1:2,
                        spw_name="spw_0") -> DimArray

Wrap the solved gain cube in a `DimArray` with `(Frequency, Ti, Ant, Pol)`
axes — frequency-fastest, pol-slowest in memory. The `spw_name` rides
on the DimArray's metadata so callers can pass the result directly to
`apply_bandpass` without extra wrapping. Pol axis labels feed indices
`1:2` (each correlation product uses one feed per side); pol-product
names live on `data.vis`'s Pol axis.
"""
function wrap_gain_solutions(
        gains, data::BandpassDataset;
        pol_keys = 1:2,
        spw_name::AbstractString = "spw_0",
    )
    size(gains, 1) == length(UVData.channel_freqs(data.freq_setup)) || error("Gain channel axis does not match dataset channels")
    size(gains, 2) == length(data.scans) || error("Gain Ti axis does not match dataset scans")
    size(gains, 3) == length(data.antennas) || error("Gain antenna axis does not match dataset antennas")
    size(gains, 4) == length(pol_keys) || error("pol_keys length must match gain polarisation axis")

    metadata = Dict{Symbol, Any}(
        :band_center_frequency => band_center_frequency_dataset(data),
        :spw_name => String(spw_name),
    )

    return DimensionalData.DimArray(
        gains, (
            Frequency(UVData.channel_freqs(data.freq_setup)),
            Ti(scan_time_centers_dataset(data)),
            Ant(data.antennas.name),
            Pol(collect(pol_keys)),
        ); metadata = metadata,
    )
end
