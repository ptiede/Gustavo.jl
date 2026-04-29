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
    _to_bandpass_dataset(uvset::UVSet) -> BandpassDataset

Assemble a dense `(Scan, Baseline_union, Pol, IF)` averaged cube from a
`UVSet` by inverse-variance averaging each leaf's `Ti` axis (one row per
leaf / scan) and unioning baselines across leaves.
"""
function _to_bandpass_dataset(uvset::UVSet)
    branches_d = DimensionalData.branches(uvset)
    isempty(branches_d) && error("_to_bandpass_dataset: empty UVSet")

    # Asserts every leaf shares one frequency setup; per-SPW grouping is a
    # future extension and would dispatch on a `Vector{FrequencySetup}` here.
    fs = UVData.freq_setup(uvset)

    # Concretely-typed leaf vector keeps inner loops type-stable despite the
    # `OrderedDict{Symbol, AbstractDimTree}` abstract eltype.
    leaf_list = collect(values(branches_d))

    union_codes_set = Set{Int}()
    for p in leaf_list
        union!(union_codes_set, Int.(baselines(p).unique_codes))
    end
    union_codes = sort!(collect(union_codes_set))
    union_lookup = Dict(c => i for (i, c) in enumerate(union_codes))
    code_to_pair = Dict{Int, Tuple{Int, Int}}()
    code_to_ant1 = Dict{Int, String}()
    code_to_ant2 = Dict{Int, String}()
    for p in leaf_list
        for (i, c) in enumerate(baselines(p).unique_codes)
            ci = Int(c)
            code_to_pair[ci] = baselines(p).pairs[i]
            code_to_ant1[ci] = baselines(p).ant1_names[i]
            code_to_ant2[ci] = baselines(p).ant2_names[i]
        end
    end
    union_pairs = [code_to_pair[c] for c in union_codes]
    union_ant1 = [code_to_ant1[c] for c in union_codes]
    union_ant2 = [code_to_ant2[c] for c in union_codes]
    union_labels = string.(union_ant1, "-", union_ant2)
    union_baselines = UVData.BaselineIndex(
        Int[], union_pairs, union_lookup, union_codes,
        union_labels, union_ant1, union_ant2,
    )
    nbl = length(union_codes)

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

    V_num = zeros(Tvis, nscan, nbl, npol, nchan)
    W_sum = zeros(Tw, nscan, nbl, npol, nchan)
    UVW_num = zeros(Tuvw, nscan, nbl, 3)
    UVW_w = zeros(Tw, nscan, nbl)

    for (sid, part) in enumerate(sorted_leaves)
        # Function barrier: extract concretely-typed parent arrays and pass
        # to a kernel so the inner-loop scalar accesses don't box.
        _accumulate_avg_into!(
            V_num, W_sum, UVW_num, UVW_w,
            parent(part[:vis]), parent(part[:weights]), parent(part[:uvw]),
            baselines(part).unique_codes, union_lookup, sid,
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
            Scan(scan_centers), Baseline(union_labels),
            Pol(pol_labels), IF(UVData.channel_freqs(fs)),
        ),
    )
    weights_avg = DimensionalData.DimArray(W_sum, dims(vis_avg))
    uvw_avg = DimensionalData.DimArray(
        UVW_out,
        (Scan(scan_centers), Baseline(union_labels), UVW(["U", "V", "W"])),
    )

    return BandpassDataset(
        vis_avg, weights_avg, uvw_avg, union_baselines,
        root.antennas, array_obs, fs, scan_windows,
    )
end

# Type-stable inner kernel for `_to_bandpass_dataset`. Accumulates one
# leaf's contribution to the union (Scan, Baseline, Pol, IF) cube.
function _accumulate_avg_into!(
        V_num::AbstractArray{Tvis, 4},
        W_sum::AbstractArray{Tw, 4},
        UVW_num::AbstractArray{Tuvw, 3},
        UVW_w::AbstractArray{Tw, 2},
        vis_p::AbstractArray{Tvis, 4},
        w_p::AbstractArray{Tw, 4},
        uvw_p::AbstractArray{Tuvw, 3},
        unique_codes_local, union_lookup, sid::Integer,
    ) where {Tvis, Tw, Tuvw}
    nti_p, nbl_p, npol, nchan = size(vis_p)
    @inbounds for ti in 1:nti_p, bi_local in 1:nbl_p
        bl_code = Int(unique_codes_local[bi_local])
        bi = union_lookup[bl_code]
        tot_w = zero(Tw)
        for p in 1:npol, c in 1:nchan
            w = w_p[ti, bi_local, p, c]
            v = vis_p[ti, bi_local, p, c]
            (w > 0 && isfinite(w) && isfinite(real(v))) || continue
            V_num[sid, bi, p, c] += w * v
            W_sum[sid, bi, p, c] += w
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
    wrap_gain_solutions(gains, data::BandpassDataset; pol_keys=1:2)

Wrap the solved gain cube in a `DimArray` with scan/site/pol/IF axes.
"""
function wrap_gain_solutions(gains, data::BandpassDataset; pol_keys = 1:2)
    size(gains, 1) == length(data.scans) || error("Gain scan axis does not match dataset scans")
    size(gains, 2) == length(data.antennas) || error("Gain antenna axis does not match dataset antennas")
    size(gains, 3) == length(pol_keys) || error("pol_keys length must match gain polarisation axis")
    size(gains, 4) == length(UVData.channel_freqs(data.freq_setup)) || error("Gain channel axis does not match dataset channels")

    return DimensionalData.DimArray(
        gains, (
            Scan(scan_time_centers_dataset(data)),
            Ant(data.antennas.name),
            Pol(collect(pol_keys)),
            IF(UVData.channel_freqs(data.freq_setup)),
        ); metadata = Dict(
            :band_center_frequency => band_center_frequency_dataset(data),
        )
    )
end

"""
    wrap_xy_correction(xy_correction, data::BandpassDataset, ref_ant; applies_to_pol, reference_pol)
"""
function wrap_xy_correction(xy_correction, data::BandpassDataset, ref_ant; applies_to_pol, reference_pol)
    size(xy_correction, 1) == length(data.scans) || error("XY correction scan axis does not match dataset scans")
    size(xy_correction, 2) == length(UVData.channel_freqs(data.freq_setup)) || error("XY correction IF axis does not match dataset channels")
    1 <= ref_ant <= length(data.antennas) || error("ref_ant index out of bounds")

    metadata = Dict(
        :SiteIndices => [ref_ant],
        :Sites => [data.antennas.name[ref_ant]],
        :applies_to_pol => applies_to_pol,
        :reference_pol => reference_pol,
        :channel_freqs => UVData.channel_freqs(data.freq_setup),
        :band_center_frequency => band_center_frequency_dataset(data),
    )

    return DimensionalData.DimArray(
        xy_correction, (
            Scan(scan_time_centers_dataset(data)),
            IF(UVData.channel_freqs(data.freq_setup)),
        ); metadata = metadata
    )
end
