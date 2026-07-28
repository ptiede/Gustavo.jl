# ── Segmentation vocabulary ──────────────────────────────────────────────────
#
# A calibration *component* is replicated across a partition of the time axis
# (one parameter block per time segment) and the frequency axis (one block per
# frequency segment). The segmentation types below name those partitions; they
# carry no data themselves — they are resolved against a concrete `DataGeometry`
# into per-sample segment-id vectors by `time_segment_ids` / `freq_segment_ids`.
#
# This single vocabulary covers everything both the bandpass solver and the
# fringe fitter need: accumulation periods (`PerIntegration`), scans
# (`PerScan`), instrument scans (`InstrumentScans`), fixed-duration stability
# blocks (`TimeBlocks`); and spectral windows/IFs/bands (`PerSpectralWindow`)
# and channel blocks (`ChannelBlocks`, with `ChannelBlocks(1)` the free
# per-channel bandpass).

abstract type AbstractTimeSegmentation end

"One segment spanning the whole time axis (time-invariant parameter)."
struct GlobalTime <: AbstractTimeSegmentation end

"One segment per scan."
struct PerScan <: AbstractTimeSegmentation end

"One segment per integration / accumulation period (each `Ti` sample)."
struct PerIntegration <: AbstractTimeSegmentation end

"One segment per fixed wall-clock block of `duration_hr` hours."
struct TimeBlocks <: AbstractTimeSegmentation
    duration_hr::Float64
    function TimeBlocks(duration_hr::Real)
        duration_hr > 0 ||
            throw(ArgumentError("TimeBlocks duration_hr must be positive, got $duration_hr"))
        return new(Float64(duration_hr))
    end
end

"""
One segment per user-specified instrument-scan window. `boundaries_hr` lists
the interior boundaries (hours); `n+1` segments result from `n` boundaries.
"""
struct InstrumentScans <: AbstractTimeSegmentation
    boundaries_hr::Vector{Float64}
end
InstrumentScans(b::AbstractVector{<:Real}) = InstrumentScans(sort!(Float64.(collect(b))))

abstract type AbstractFrequencySegmentation end

"One segment spanning all channels of all spectral windows."
struct GlobalFrequency <: AbstractFrequencySegmentation end

"One segment per spectral window / IF / band."
struct PerSpectralWindow <: AbstractFrequencySegmentation end

"Blocks of `block_size` consecutive channels, never straddling a spw boundary."
struct ChannelBlocks <: AbstractFrequencySegmentation
    block_size::Int
    function ChannelBlocks(block_size::Integer)
        block_size >= 1 ||
            throw(ArgumentError("ChannelBlocks block_size must be at least 1, got $block_size"))
        return new(Int(block_size))
    end
end

"""
One segment per explicit global-channel range — for caller-defined frequency
groupings the other segmentations cannot express, e.g. the widely-separated
VGOS band groups (`Fringe.fringe_band_groups`). Ranges must be ascending,
contiguous, start at channel 1, and (checked against the geometry at plan
time) cover every channel exactly once.
"""
struct FrequencyBands <: AbstractFrequencySegmentation
    ranges::Vector{UnitRange{Int}}
    function FrequencyBands(ranges::AbstractVector{<:UnitRange{<:Integer}})
        rs = [UnitRange{Int}(r) for r in ranges]
        isempty(rs) && throw(ArgumentError("FrequencyBands: at least one range required"))
        first(rs[1]) == 1 || throw(
            ArgumentError("FrequencyBands: ranges must start at channel 1, got $(rs[1])")
        )
        for i in 2:length(rs)
            first(rs[i]) == last(rs[i - 1]) + 1 || throw(
                ArgumentError(
                    "FrequencyBands: ranges must be contiguous and ascending; " *
                        "$(rs[i]) does not follow $(rs[i - 1])"
                )
            )
        end
        return new(rs)
    end
end

# ── DataGeometry ─────────────────────────────────────────────────────────────

"""
    DataGeometry

The concrete time/frequency grid a calibration solve runs over — the bridge
from the abstract segmentation vocabulary to integer sample indices. Built once
per solve (from a `UVSet` scan-group, or directly in tests).

Fields:
- `times`         : `Ti` sample epochs (hours since the dataset reference time).
- `scan_of_time`  : scan id for each time sample (any integer labels; need not
                    be 1-based or contiguous — `PerScan` dense-ranks them).
- `channel_freqs` : concatenated channel center frequencies (Hz) across spws.
- `spw_of_chan`   : spw/band id for each global channel (same labelling freedom).
- `t0`            : rate reference epoch (hours); rate phase ∝ (t − t0).
- `f0`            : delay reference frequency (Hz); delay phase ∝ (f − f0).
- `scan_names`    : human-readable scan labels (for diagnostics; optional).
- `spw_names`     : human-readable spw labels (for diagnostics; optional).
"""
struct DataGeometry
    times::Vector{Float64}
    scan_of_time::Vector{Int}
    channel_freqs::Vector{Float64}
    spw_of_chan::Vector{Int}
    t0::Float64
    f0::Float64
    scan_names::Vector{String}
    spw_names::Vector{String}
end

function DataGeometry(;
        times::AbstractVector{<:Real},
        channel_freqs::AbstractVector{<:Real},
        scan_of_time::AbstractVector{<:Integer} = ones(Int, length(times)),
        spw_of_chan::AbstractVector{<:Integer} = ones(Int, length(channel_freqs)),
        t0::Real = isempty(times) ? 0.0 : first(times),
        f0::Real = isempty(channel_freqs) ? 0.0 : sum(channel_freqs) / length(channel_freqs),
        scan_names::AbstractVector{<:AbstractString} = String[],
        spw_names::AbstractVector{<:AbstractString} = String[],
    )
    length(scan_of_time) == length(times) || throw(
        DimensionMismatch(
            "scan_of_time length $(length(scan_of_time)) ≠ times length $(length(times))"
        )
    )
    length(spw_of_chan) == length(channel_freqs) || throw(
        DimensionMismatch(
            "spw_of_chan length $(length(spw_of_chan)) ≠ " *
                "channel_freqs length $(length(channel_freqs))"
        )
    )
    return DataGeometry(
        Float64.(collect(times)), Int.(collect(scan_of_time)),
        Float64.(collect(channel_freqs)), Int.(collect(spw_of_chan)),
        Float64(t0), Float64(f0),
        String.(collect(scan_names)), String.(collect(spw_names)),
    )
end

ntimes(geom::DataGeometry) = length(geom.times)
nchannels(geom::DataGeometry) = length(geom.channel_freqs)

# Map arbitrary integer labels to dense 1..k ids preserving first-appearance
# order. The basis of every segment-id computation below.
function _dense_rank(labels::AbstractVector{<:Integer})
    seen = Dict{Int, Int}()
    out = Vector{Int}(undef, length(labels))
    next = 0
    @inbounds for i in eachindex(labels)
        id = get(seen, labels[i], 0)
        if id == 0
            next += 1
            seen[labels[i]] = next
            id = next
        end
        out[i] = id
    end
    return out, next
end

# ── Time segment ids ─────────────────────────────────────────────────────────

"""
    time_segment_ids(seg, geom) -> (ids::Vector{Int}, nseg::Int)

Segment id (1..nseg) for each time sample under time segmentation `seg`.
"""
time_segment_ids(::GlobalTime, geom::DataGeometry) = (ones(Int, ntimes(geom)), 1)
time_segment_ids(::PerIntegration, geom::DataGeometry) =
    (collect(1:ntimes(geom)), ntimes(geom))
time_segment_ids(::PerScan, geom::DataGeometry) = _dense_rank(geom.scan_of_time)

function time_segment_ids(seg::TimeBlocks, geom::DataGeometry)
    n = ntimes(geom)
    n == 0 && return (Int[], 0)
    t_start = minimum(geom.times)
    blocks = [floor(Int, (t - t_start) / seg.duration_hr) for t in geom.times]
    return _dense_rank(blocks)
end

function time_segment_ids(seg::InstrumentScans, geom::DataGeometry)
    bins = [searchsortedlast(seg.boundaries_hr, t) + 1 for t in geom.times]
    return _dense_rank(bins)
end

# ── Frequency segment ids ────────────────────────────────────────────────────

"""
    freq_segment_ids(seg, geom) -> (ids::Vector{Int}, nseg::Int)

Segment id (1..nseg) for each global channel under frequency segmentation `seg`.
"""
freq_segment_ids(::GlobalFrequency, geom::DataGeometry) = (ones(Int, nchannels(geom)), 1)
freq_segment_ids(::PerSpectralWindow, geom::DataGeometry) = _dense_rank(geom.spw_of_chan)

function freq_segment_ids(seg::FrequencyBands, geom::DataGeometry)
    n = nchannels(geom)
    last(seg.ranges[end]) == n || throw(
        DimensionMismatch(
            "FrequencyBands: ranges cover $(last(seg.ranges[end])) channels; geometry has $n"
        )
    )
    ids = Vector{Int}(undef, n)
    for (k, r) in enumerate(seg.ranges), c in r
        ids[c] = k
    end
    return ids, length(seg.ranges)
end

function freq_segment_ids(seg::ChannelBlocks, geom::DataGeometry)
    spw = geom.spw_of_chan
    n = length(spw)
    # Per-channel composite label (spw, block-within-spw). A block boundary
    # falls every `block_size` channels counted *within each spw* in channel
    # order, so blocks never straddle a spw change.
    labels = Vector{Int}(undef, n)
    spw_counter = Dict{Int, Int}()
    nextlabel = Dict{Tuple{Int, Int}, Int}()
    nlab = 0
    @inbounds for c in 1:n
        k = get(spw_counter, spw[c], 0)
        block = k ÷ seg.block_size
        spw_counter[spw[c]] = k + 1
        key = (spw[c], block)
        id = get(nextlabel, key, 0)
        if id == 0
            nlab += 1
            nextlabel[key] = nlab
            id = nlab
        end
        labels[c] = id
    end
    return _dense_rank(labels)
end

# Convenience: convert segment-id vector into the list of index groups.
function segment_groups(ids::AbstractVector{<:Integer}, nseg::Integer)
    groups = [Int[] for _ in 1:nseg]
    @inbounds for i in eachindex(ids)
        push!(groups[ids[i]], i)
    end
    return groups
end
