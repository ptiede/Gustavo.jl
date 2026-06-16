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
    V, W_sum, UVW_out, t_center = _time_average_kernel(vis_l, weights_l, uvw_l, obs_time(leaf))


    new_obs_time = [t_center]

    pol_dim = dims(vis_l, Pol)
    if_dim = dims(vis_l, Frequency)
    bl_dim = dims(vis_l, Baseline)
    # Storage layout: (Frequency, Ti, Baseline, Pol). uvw: (Ti, Baseline, UVW).
    vis_da = DimArray(V, (if_dim, Ti(new_obs_time), bl_dim, pol_dim))
    weights_da = DimArray(W_sum, dims(vis_da))
    uvw_da = DimArray(UVW_out, (Ti(new_obs_time), bl_dim, UVW(["U", "V", "W"])))
    flag_da = DimArray(W_sum .<= 0, dims(vis_da))

    info = DimensionalData.metadata(leaf)
    new_info = update(
        info;
        record_order = Tuple{Int, Int}[],
        extra_columns = NamedTuple(),
    )
    return _build_leaf(vis_da, weights_da, uvw_da, flag_da; partition_info = new_info)
end

# Type-stable kernel for inverse-variance time-averaging. Hot loop sees
# concrete `vis_p::Array{Tvis,4}`, `w_p::Array{Tw,4}`, `uvw_p::Array{Tuvw,3}`.
# Layout: vis/weights are (Frequency, Ti, Baseline, Pol); uvw is (Ti, Baseline, UVW).
function _time_average_kernel(
        vis_p::AbstractArray{Tvis, 4},
        w_p::AbstractArray{Tw, 4},
        uvw_p::AbstractArray{Tuvw, 3},
        obs_t,
    ) where {Tvis, Tw, Tuvw}
    nchan, nti, nbl, npol = size(vis_p)
    V_num = zeros(Tvis, nchan, 1, nbl, npol)
    W_sum = zeros(Tw, nchan, 1, nbl, npol)
    UVW_num = zeros(Tuvw, 1, nbl, 3)
    UVW_w = zeros(Tw, 1, nbl)

    @inbounds for ti in 1:nti, bi in 1:nbl
        tot_w = zero(Tw)
        for p in 1:npol, c in 1:nchan
            w = w_p[c, ti, bi, p]
            v = vis_p[c, ti, bi, p]
            (w > 0 && isfinite(w) && isfinite(real(v))) || continue
            V_num[c, 1, bi, p] += w * v
            W_sum[c, 1, bi, p] += w
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


# ── Frequency averaging ──────────────────────────────────────────────────────

"""
    FrequencyAverage(nout = 1)

Per-leaf reducer that inverse-variance-averages the `Frequency` axis into `nout`
contiguous output channels (default 1 — collapse each band leaf to a single
channel). The leaf's `freq_setup` is updated to the averaged channels (center =
mean channel frequency of each group, `ch_width`/`total_bandwidth` = the group's
summed channel widths). Post-fringe-fit continuum reduction: a band's delay
structure has been removed, so coherently averaging channels is lossless for
imaging while shrinking the data by `nchan/nout`.
"""
struct FrequencyAverage <: AbstractPartitionReducer
    nout::Int
    function FrequencyAverage(nout::Integer = 1)
        nout >= 1 || error("FrequencyAverage nout must be ≥ 1")
        return new(Int(nout))
    end
end

(r::FrequencyAverage)(leaf::DimensionalData.AbstractDimTree, ::PartitionInfo, ::UVMetadata) =
    _frequency_average_partition(leaf, r.nout)

# Contiguous near-equal channel groups (group g = output channel g).
function _channel_groups(nchan::Integer, nout::Integer)
    nout = min(nout, nchan)
    bs = cld(nchan, nout)
    return [((g - 1) * bs + 1):min(g * bs, nchan) for g in 1:cld(nchan, bs)]
end

function _frequency_average_partition(leaf::DimensionalData.AbstractDimTree, nout::Integer)
    vis_l = leaf[:vis]
    weights_l = leaf[:weights]
    uvw_l = leaf[:uvw]
    info = DimensionalData.metadata(leaf)
    fs = info.freq_setup
    cfreqs = collect(channel_freqs(fs))
    cwidths = collect(ch_widths(fs))
    groups = _channel_groups(length(cfreqs), nout)

    V, W = _frequency_average_kernel(vis_l, weights_l, groups)

    # New per-group channel axis + freq_setup.
    new_freqs = [sum(@view cfreqs[g]) / length(g) for g in groups]
    new_widths = [sum(@view cwidths[g]) for g in groups]
    new_fs = FrequencySetup(;
        name = fs.name, ref_freq = fs.ref_freq,
        channel_freqs = new_freqs, ch_widths = new_widths,
        total_bandwidths = new_widths,
        sidebands = [sidebands(fs)[first(g)] for g in groups],
        extras = fs.extras,
    )

    pol_dim = dims(vis_l, Pol)
    bl_dim = dims(vis_l, Baseline)
    ti_dim = dims(vis_l, Ti)
    vis_da = DimArray(V, (Frequency(new_freqs), ti_dim, bl_dim, pol_dim))
    weights_da = DimArray(W, dims(vis_da))
    flag_da = DimArray(W .<= 0, dims(vis_da))
    new_info = update(info; freq_setup = new_fs)
    # uvw is frequency-independent — carry it through unchanged.
    return _build_leaf(vis_da, weights_da, uvw_l, flag_da; partition_info = new_info)
end

# Type-stable kernel: average `vis`/`weights` over each channel group.
# Layout (Frequency, Ti, Baseline, Pol).
function _frequency_average_kernel(
        vis_p::AbstractArray{Tvis, 4}, w_p::AbstractArray{Tw, 4}, groups,
    ) where {Tvis, Tw}
    nchan, nti, nbl, npol = size(vis_p)
    ng = length(groups)
    Vnum = zeros(Tvis, ng, nti, nbl, npol)
    Wsum = zeros(Tw, ng, nti, nbl, npol)
    @inbounds for p in 1:npol, bi in 1:nbl, ti in 1:nti
        for (g, grp) in enumerate(groups)
            for c in grp
                w = w_p[c, ti, bi, p]
                v = vis_p[c, ti, bi, p]
                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                Vnum[g, ti, bi, p] += w * v
                Wsum[g, ti, bi, p] += w
            end
        end
    end
    V = similar(Vnum)
    @inbounds for k in eachindex(V)
        V[k] = Wsum[k] > 0 ? Vnum[k] / Wsum[k] : Tvis(NaN, NaN)
    end
    return V, Wsum
end

"""
    frequency_average(uvset::UVSet; nout = 1) -> UVSet

Average each leaf's `Frequency` axis into `nout` channels. `apply(FrequencyAverage(nout), uvset)`.
"""
frequency_average(uvset::UVSet; nout::Integer = 1) = apply(FrequencyAverage(nout), uvset)


# ── Time-bin averaging ───────────────────────────────────────────────────────

"""
    TimeBinAverage(dt_seconds)

Per-leaf reducer that inverse-variance-averages the `Ti` axis into consecutive
bins spanning `dt_seconds` (the `Ti` lookup is in hours). Bins are formed by
`floor((t − t₀)/Δt)` over the leaf's sorted times; each output sample sits at its
bin's weighted-mean epoch, with summed weights and weight-mean `uvw`. After
fringe + adhoc phasing the per-AP residual phase is flat, so averaging to a
coarser cadence is lossless — the core payoff of fringe fitting.
"""
struct TimeBinAverage <: AbstractPartitionReducer
    dt_seconds::Float64
    function TimeBinAverage(dt_seconds::Real)
        dt_seconds > 0 || error("TimeBinAverage dt_seconds must be positive")
        return new(Float64(dt_seconds))
    end
end

(r::TimeBinAverage)(leaf::DimensionalData.AbstractDimTree, ::PartitionInfo, ::UVMetadata) =
    _time_bin_average_partition(leaf, r.dt_seconds)

# Bin id (1..nbin, contiguous) for each (sorted) time sample under width dt (s).
function _time_bins(ts::AbstractVector, dt_seconds::Real)
    n = length(ts)
    n == 0 && return (Int[], Float64[], 0)
    t0 = float(first(ts))
    raw = [floor(Int, (float(t) - t0) * 3600.0 / dt_seconds) for t in ts]
    ids = Vector{Int}(undef, n)
    nbin = 0
    last = typemin(Int)
    @inbounds for i in 1:n
        if raw[i] != last
            nbin += 1
            last = raw[i]
        end
        ids[i] = nbin
    end
    return ids, Float64.(ts), nbin
end

function _time_bin_average_partition(leaf::DimensionalData.AbstractDimTree, dt_seconds::Real)
    vis_l = leaf[:vis]
    weights_l = leaf[:weights]
    uvw_l = leaf[:uvw]
    ts = collect(obs_time(leaf))
    ids, tvals, nbin = _time_bins(ts, dt_seconds)

    V, W, UVW_out, tcenters = _time_bin_average_kernel(vis_l, weights_l, uvw_l, ids, tvals, nbin)

    pol_dim = dims(vis_l, Pol)
    if_dim = dims(vis_l, Frequency)
    bl_dim = dims(vis_l, Baseline)
    vis_da = DimArray(V, (if_dim, Ti(tcenters), bl_dim, pol_dim))
    weights_da = DimArray(W, dims(vis_da))
    uvw_da = DimArray(UVW_out, (Ti(tcenters), bl_dim, UVW(["U", "V", "W"])))
    flag_da = DimArray(W .<= 0, dims(vis_da))
    info = DimensionalData.metadata(leaf)
    new_info = update(info; record_order = Tuple{Int, Int}[], extra_columns = NamedTuple())
    return _build_leaf(vis_da, weights_da, uvw_da, flag_da; partition_info = new_info)
end

# Type-stable kernel: inverse-variance average into `nbin` time bins.
# Layout: vis/weights (Frequency, Ti, Baseline, Pol); uvw (Ti, Baseline, UVW).
function _time_bin_average_kernel(
        vis_p::AbstractArray{Tvis, 4}, w_p::AbstractArray{Tw, 4},
        uvw_p::AbstractArray{Tuvw, 3}, ids::AbstractVector{<:Integer},
        tvals::AbstractVector, nbin::Integer,
    ) where {Tvis, Tw, Tuvw}
    nchan, nti, nbl, npol = size(vis_p)
    Vnum = zeros(Tvis, nchan, nbin, nbl, npol)
    Wsum = zeros(Tw, nchan, nbin, nbl, npol)
    UVWnum = zeros(Tuvw, nbin, nbl, 3)
    UVWw = zeros(Tw, nbin, nbl)
    tnum = zeros(Float64, nbin)
    tw = zeros(Float64, nbin)

    @inbounds for ti in 1:nti
        b = ids[ti]
        for bi in 1:nbl
            tot_w = zero(Tw)
            for p in 1:npol, c in 1:nchan
                w = w_p[c, ti, bi, p]
                v = vis_p[c, ti, bi, p]
                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                Vnum[c, b, bi, p] += w * v
                Wsum[c, b, bi, p] += w
                tot_w += w
            end
            (tot_w > 0 && isfinite(tot_w)) || continue
            for k in 1:3
                u = uvw_p[ti, bi, k]
                isfinite(u) || continue
                UVWnum[b, bi, k] += tot_w * u
            end
            UVWw[b, bi] += tot_w
            tnum[b] += tot_w * tvals[ti]
            tw[b] += tot_w
        end
    end

    V = similar(Vnum)
    @inbounds for k in eachindex(V)
        V[k] = Wsum[k] > 0 ? Vnum[k] / Wsum[k] : Tvis(NaN, NaN)
    end
    UVW_out = fill(Tuvw(NaN), nbin, nbl, 3)
    @inbounds for bi in 1:nbl, b in 1:nbin
        if UVWw[b, bi] > 0
            for k in 1:3
                UVW_out[b, bi, k] = UVWnum[b, bi, k] / UVWw[b, bi]
            end
        end
    end
    tcenters = [tw[b] > 0 ? tnum[b] / tw[b] : (isempty(tvals) ? 0.0 : tvals[1]) for b in 1:nbin]
    return V, Wsum, UVW_out, tcenters
end

"""
    time_bin_average(uvset::UVSet, dt_seconds) -> UVSet

Average each leaf's `Ti` axis into `dt_seconds`-wide bins. `apply(TimeBinAverage(dt_seconds), uvset)`.
"""
time_bin_average(uvset::UVSet, dt_seconds::Real) = apply(TimeBinAverage(dt_seconds), uvset)


# ── Combine spectral windows (bands) into one IF axis ────────────────────────

"""
    combine_spw(uvset::UVSet) -> UVSet

Merge the sibling spectral-window (band) leaves of each (source, scan, subarray)
into a single leaf whose `Frequency` axis is the concatenation of all bands'
channels (sorted by frequency). The per-band `FrequencySetup`s collapse into one
setup whose `channel_freqs`/`ch_widths`/`total_bandwidths`/`sidebands` are the
concatenated band values.

This is the shape a UVFITS export wants: `write_uvfits` maps a leaf's `Frequency`
axis onto the AIPS IF axis (one channel per IF), so the combined bands become the
IFs of a single FREQID — the standard continuum layout — rather than one FREQID
per band (which a reader also cannot serialize when a band carries a single
channel). Apply after [`frequency_average`](@ref) so each band is one channel.

All sibling leaves of a group must share their `Ti`, `Baseline`, and `Pol` axes
(true for bands read from one FITS-IDI `UV_DATA` table); a mismatch errors.
Groups with a single band are returned unchanged.
"""
function combine_spw(uvset::UVSet)
    groups = Dict{Tuple{String, String, String}, Vector{Any}}()
    order = Tuple{String, String, String}[]
    for (_, leaf) in DimensionalData.branches(uvset)
        info = DimensionalData.metadata(leaf)
        key = (String(info.source_name), String(info.scan_name), String(info.subarray_name))
        if !haskey(groups, key)
            groups[key] = Any[]
            push!(order, key)
        end
        push!(groups[key], leaf)
    end

    new_branches = DimensionalData.TreeDict()
    for key in order
        leaves = groups[key]
        merged = length(leaves) == 1 ? only(leaves) : _combine_band_leaves(leaves)
        new_branches[partition_key(DimensionalData.metadata(merged))] = merged
    end
    return DimensionalData.rebuild(uvset; branches = new_branches)
end

# Concatenate sibling band leaves along Frequency into one leaf (one merged
# FrequencySetup). Leaves must agree on Ti/Baseline/Pol; uvw is freq-independent
# so the first leaf's is carried through.
function _combine_band_leaves(leaves)
    band_min(l) = minimum(channel_freqs(DimensionalData.metadata(l).freq_setup))
    leaves = sort(collect(leaves); by = band_min)
    l0 = first(leaves)
    info0 = DimensionalData.metadata(l0)

    ti0 = lookup(l0[:vis], Ti)
    bl0 = lookup(l0[:vis], Baseline)
    pol0 = lookup(l0[:vis], Pol)
    for l in leaves
        (lookup(l[:vis], Ti) == ti0 && lookup(l[:vis], Baseline) == bl0 && lookup(l[:vis], Pol) == pol0) ||
            error(
            "combine_spw: sibling band leaves must share Ti/Baseline/Pol axes. " *
                "If you time-averaged first, each band got its own weighted bin-center " *
                "epochs — apply combine_spw BEFORE time_bin_average (after frequency_average, " *
                "which preserves the integration axis).",
        )
    end

    vis_cat = cat(map(l -> parent(l[:vis]), leaves)...; dims = 1)
    w_cat = cat(map(l -> parent(l[:weights]), leaves)...; dims = 1)

    setups = [DimensionalData.metadata(l).freq_setup for l in leaves]
    fs0 = first(setups)
    new_freqs = reduce(vcat, [collect(channel_freqs(fs)) for fs in setups])
    new_fs = FrequencySetup(;
        name = setup_name(fs0), ref_freq = ref_freq(fs0),
        channel_freqs = new_freqs,
        ch_widths = reduce(vcat, [collect(ch_widths(fs)) for fs in setups]),
        total_bandwidths = reduce(vcat, [collect(total_bandwidths(fs)) for fs in setups]),
        sidebands = reduce(vcat, [collect(sidebands(fs)) for fs in setups]),
        extras = (; frqsel = Int32(1)),
    )

    ti_dim = dims(l0[:vis], Ti)
    bl_dim = dims(l0[:vis], Baseline)
    pol_dim = dims(l0[:vis], Pol)
    vis_da = DimArray(vis_cat, (Frequency(new_freqs), ti_dim, bl_dim, pol_dim))
    w_da = DimArray(w_cat, dims(vis_da))
    flag_da = DimArray(parent(w_da) .<= 0, dims(vis_da))

    new_info = update(info0; freq_setup = new_fs, spw_name = "combined", ddi = 0)
    return _build_leaf(vis_da, w_da, l0[:uvw], flag_da; partition_info = new_info)
end
