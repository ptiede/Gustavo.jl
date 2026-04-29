"""
    AbstractPartitionReducer

Abstract type for callables that map a leaf `DimTree` to a transformed leaf,
to be composed with `apply(reducer, uvset)`. Built-in reducers (`TimeAverage`,
`BandpassCorrection`) implement the
`(leaf::DimTree, partition_info::PartitionInfo, root_meta::UVMetadata) -> DimTree`
signature.
"""
abstract type AbstractPartitionReducer end


"""
    TimeAverage()

Per-leaf reducer that collapses the `Ti` axis to length 1 by computing the
inverse-variance-weighted mean of `vis`, the weight sum of `weights`, and the
weight-mean of `uvw`. Cells with no valid contributions become `NaN+NaN*im`
(vis), `0` (weights), `NaN` (uvw).
"""
struct TimeAverage <: AbstractPartitionReducer end

(r::TimeAverage)(leaf::DimensionalData.AbstractDimTree, ::PartitionInfo, ::UVMetadata) =
    _time_average_partition(leaf)

function _time_average_partition(leaf::DimensionalData.AbstractDimTree)
    vis_l = leaf[:vis]
    weights_l = leaf[:weights]
    uvw_l = leaf[:uvw]
    # Function barrier: dispatch through `_time_average_kernel` so the inner
    # loops see concrete eltypes. Without this, `leaf[:vis]` returns a DimArray
    # backed by `data(leaf)::DataDict = OrderedDict{Symbol, Any}`, so every
    # scalar access boxes.
    V, W_sum, UVW_out, t_center = _time_average_kernel(parent(vis_l), parent(weights_l), parent(uvw_l), obs_time(leaf))

    nbl = size(V, 2)
    new_obs_time = [t_center]

    pol_dim = dims(vis_l, Pol)
    if_dim = dims(vis_l, IF)
    bl_dim = dims(vis_l, Baseline)
    vis_da = DimArray(V, (Ti(new_obs_time), bl_dim, pol_dim, if_dim))
    weights_da = DimArray(W_sum, dims(vis_da))
    uvw_da = DimArray(UVW_out, (Ti(new_obs_time), bl_dim, UVW(["U", "V", "W"])))
    flag_da = DimArray(W_sum .<= 0, dims(vis_da))

    info = DimensionalData.metadata(leaf)
    new_info = update(
        info;
        record_order = Tuple{Int, Int}[],
        date_param = zeros(eltype(info.date_param), 0, size(info.date_param, 2)),
        extra_columns = NamedTuple(),
    )
    return _build_leaf(vis_da, weights_da, uvw_da, flag_da; partition_info = new_info)
end

# Type-stable kernel for inverse-variance time-averaging. Hot loop sees
# concrete `vis_p::Array{Tvis,4}`, `w_p::Array{Tw,4}`, `uvw_p::Array{Tuvw,3}`.
function _time_average_kernel(
        vis_p::AbstractArray{Tvis, 4},
        w_p::AbstractArray{Tw, 4},
        uvw_p::AbstractArray{Tuvw, 3},
        obs_t,
    ) where {Tvis, Tw, Tuvw}
    nti, nbl, npol, nchan = size(vis_p)
    V_num = zeros(Tvis, 1, nbl, npol, nchan)
    W_sum = zeros(Tw, 1, nbl, npol, nchan)
    UVW_num = zeros(Tuvw, 1, nbl, 3)
    UVW_w = zeros(Tw, 1, nbl)

    @inbounds for ti in 1:nti, bi in 1:nbl
        tot_w = zero(Tw)
        for p in 1:npol, c in 1:nchan
            w = w_p[ti, bi, p, c]
            v = vis_p[ti, bi, p, c]
            (w > 0 && isfinite(w) && isfinite(real(v))) || continue
            V_num[1, bi, p, c] += w * v
            W_sum[1, bi, p, c] += w
            tot_w += w
        end
        (tot_w > 0 && isfinite(tot_w)) || continue
        for k in 1:3
            u = uvw_p[ti, bi, k]
            isfinite(u) || continue
            UVW_num[1, bi, k] += tot_w * u
        end
        UVW_w[1, bi] += tot_w
    end

    V = similar(V_num)
    @inbounds for k in eachindex(V)
        V[k] = W_sum[k] > 0 ? V_num[k] / W_sum[k] : Tvis(NaN, NaN)
    end

    UVW_out = fill(Tuvw(NaN), 1, nbl, 3)
    @inbounds for bi in 1:nbl
        if UVW_w[1, bi] > 0
            for k in 1:3
                UVW_out[1, bi, k] = UVW_num[1, bi, k] / UVW_w[1, bi]
            end
        end
    end

    t_center = isempty(obs_t) ? 0.0 : (minimum(obs_t) + maximum(obs_t)) / 2
    return V, W_sum, UVW_out, t_center
end


"""
    scan_average(uvset::UVSet) -> UVSet

Time-average each leaf's `Ti` axis to length 1 (one timestamp per scan),
preserving the tree shape. Equivalent to `apply(TimeAverage(), uvset)`.
"""
scan_average(uvset::UVSet) = apply(TimeAverage(), uvset)
