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
# blocks (`TimeBlocks`); and spectral windows (`PerSpectralWindow`) and channel
# blocks (`ChannelBlocks`, with `ChannelBlocks(1)` the free per-channel
# bandpass).

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

"One segment per spectral window."
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
VGOS frequency groups ([`fringe_freq_groups`](@ref)). Ranges must be ascending,
contiguous, start at channel 1, and (checked against the geometry at plan
time) cover every channel exactly once.
"""
struct FreqGroups <: AbstractFrequencySegmentation
    ranges::Vector{UnitRange{Int}}
    function FreqGroups(ranges::AbstractVector{<:UnitRange{<:Integer}})
        rs = [UnitRange{Int}(r) for r in ranges]
        isempty(rs) && throw(ArgumentError("FreqGroups: at least one range required"))
        first(rs[1]) == 1 || throw(
            ArgumentError("FreqGroups: ranges must start at channel 1, got $(rs[1])")
        )
        for i in 2:length(rs)
            first(rs[i]) == last(rs[i - 1]) + 1 || throw(
                ArgumentError(
                    "FreqGroups: ranges must be contiguous and ascending; " *
                        "$(rs[i]) does not follow $(rs[i - 1])"
                )
            )
        end
        return new(rs)
    end
end

# `ranges` is a `Vector`, so the default struct `==` would compare by identity.
Base.:(==)(a::FreqGroups, b::FreqGroups) = a.ranges == b.ranges
Base.hash(s::FreqGroups, h::UInt) = hash(s.ranges, hash(:FreqGroups, h))

"""
    BandGroups(; gap_factor = 4.0)

Frequency groups read off the channel-frequency axis' own gap structure
([`fringe_freq_groups`](@ref)): the widely-separated VGOS 3/5/6/10 GHz groups,
or one full-range group on a contiguous axis. `gap_factor` is the ratio an
inter-block gap must exceed to count as a between-group one.

Data-dependent: it names the partition RULE, not the partition.
[`materialize`](@ref) resolves it against a `DataGeometry` into the concrete
[`FreqGroups`](@ref) that layouts, solutions, and foreign-grid placement work
with, so `BandGroups` itself needs no `freq_segment_ids` methods.
"""
Base.@kwdef struct BandGroups <: AbstractFrequencySegmentation
    gap_factor::Float64 = 4.0
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
- `spw_of_chan`   : spw id for each global channel (same labelling freedom).
- `t0`            : rate reference epoch (hours); rate phase ∝ (t − t0).
- `f0`            : delay reference frequency (Hz); delay phase ∝ (f − f0).
- `scan_names`    : scan label of each distinct `scan_of_time` id, in the order
                    the ids are dense-ranked (`scan_names[s]` names segment `s`).
- `spw_names`     : spw label of each distinct `spw_of_chan` id, likewise.

The names are the identity a solution is applied on: a foreign sample is
placed in a `PerScan` or `PerSpectralWindow` segment by matching the name,
never the raw integer id or the coordinate. Name vectors may be left empty,
in which case a solution carrying such a segmentation applies only to a grid
with identical labelling. A non-empty name vector must have one entry per
distinct id (checked here).
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
    _check_names("scan_names", scan_names, scan_of_time)
    _check_names("spw_names", spw_names, spw_of_chan)
    return DataGeometry(
        Float64.(collect(times)), Int.(collect(scan_of_time)),
        Float64.(collect(channel_freqs)), Int.(collect(spw_of_chan)),
        Float64(t0), Float64(f0),
        String.(collect(scan_names)), String.(collect(spw_names)),
    )
end

ntimes(geom::DataGeometry) = length(geom.times)
nchannels(geom::DataGeometry) = length(geom.channel_freqs)

# Segment id `s` is named `names[s]`, so a name vector either covers every
# distinct label or is absent entirely. A partial one would silently name the
# wrong segment in a foreign apply.
function _check_names(what::String, names, labels)
    isempty(names) && return nothing
    n = length(Set(labels))
    length(names) == n || throw(
        DimensionMismatch(
            "DataGeometry: $what has $(length(names)) entries but the geometry has $n " *
                "distinct segments; a name vector must name every segment or be empty."
        )
    )
    return nothing
end

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

time_segment_ids(seg::Union{TimeBlocks, InstrumentScans}, geom::DataGeometry) =
    _dense_rank(map(_time_binner(seg, geom), geom.times))

# The raw-bin formula of a time segmentation that bins a coordinate, closed over
# the geometry parameters it reads. Placing a foreign sample evaluates the solve
# geometry's binner at the target's epoch, so both grids are binned identically;
# a `TimeBlocks` origin is the solve's first epoch, never the target's.
function _time_binner(seg::TimeBlocks, geom::DataGeometry)
    t_start = isempty(geom.times) ? 0.0 : minimum(geom.times)
    return t -> floor(Int, (t - t_start) / seg.duration_hr)
end
_time_binner(seg::InstrumentScans, ::DataGeometry) =
    t -> searchsortedlast(seg.boundaries_hr, t) + 1

# ── Frequency segment ids ────────────────────────────────────────────────────

"""
    freq_segment_ids(seg, geom) -> (ids::Vector{Int}, nseg::Int)

Segment id (1..nseg) for each global channel under frequency segmentation `seg`.
"""
freq_segment_ids(::GlobalFrequency, geom::DataGeometry) = (ones(Int, nchannels(geom)), 1)
freq_segment_ids(::PerSpectralWindow, geom::DataGeometry) = _dense_rank(geom.spw_of_chan)

function freq_segment_ids(seg::FreqGroups, geom::DataGeometry)
    n = nchannels(geom)
    last(seg.ranges[end]) == n || throw(
        DimensionMismatch(
            "FreqGroups: ranges cover $(last(seg.ranges[end])) channels; geometry has $n"
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

# ── Materialization and the range form ───────────────────────────────────────

"""
    materialize(seg::AbstractFrequencySegmentation, geom::DataGeometry)
        -> AbstractFrequencySegmentation

Resolve a frequency segmentation against the concrete geometry it partitions.
Identity for a segmentation that already names its partition; a data-dependent
one ([`BandGroups`](@ref)) resolves to the concrete segmentation its rule finds
on `geom`. Parameter layouts materialize every component's frequency
segmentation, so segment ids, solutions, and foreign-grid placement only ever
see the materialized form.
"""
materialize(seg::AbstractFrequencySegmentation, ::DataGeometry) = seg
materialize(b::BandGroups, geom::DataGeometry) =
    FreqGroups(fringe_freq_groups(geom.channel_freqs; gap_factor = b.gap_factor))

"""
    fringe_freq_groups(freqs; gap_factor = 4.0) -> Vector{UnitRange{Int}}

Group the contiguous sub-band blocks of a channel-frequency axis into frequency
GROUPS. The inter-block gaps are split into "within-group" vs "between-group"
scales at the largest ratio jump in their sorted values (must exceed
`gap_factor`); when the gaps carry no such two-scale structure the axis is one
group (a single far-flung pair is arbitrated against the block widths instead).
On VGOS this recovers the four widely-separated 3/5/6/10 GHz groups (each
holding several 32 MHz sub-bands); on a contiguous axis (e.g. VLBA) it returns
one full-range group. Channel ranges index the stacked frequency axis.
"""
function fringe_freq_groups(freqs::AbstractVector{<:Real}; gap_factor::Real = 4.0)
    blocks = _freq_group_ranges(freqs)
    length(blocks) <= 1 && return blocks
    gaps = [Float64(freqs[first(blocks[i + 1])] - freqs[last(blocks[i])]) for i in 1:(length(blocks) - 1)]
    thr = Inf
    if length(gaps) == 1
        # No gap statistics: a lone pair of blocks splits when the gap dwarfs the
        # blocks themselves.
        wmed = median([Float64(freqs[last(r)] - freqs[first(r)]) for r in blocks])
        gaps[1] > gap_factor * wmed && (thr = gap_factor * wmed)
    else
        s = sort(gaps)
        best = 0.0
        for i in 1:(length(s) - 1)
            s[i] > 0 || continue
            r = s[i + 1] / s[i]
            if r > best
                best = r
                thr = 0.5 * (s[i] + s[i + 1])
            end
        end
        best > gap_factor || (thr = Inf)
    end
    groups = UnitRange{Int}[]
    lo = first(blocks[1])
    for i in 1:(length(blocks) - 1)
        if gaps[i] > thr
            push!(groups, lo:last(blocks[i]))
            lo = first(blocks[i + 1])
        end
    end
    push!(groups, lo:last(blocks[end]))
    return groups
end

# Contiguous frequency-group ranges of a channel-frequency axis: split where the
# step jumps by more than 3× the median spacing (the VGOS sub-band gaps).
function _freq_group_ranges(freqs::AbstractVector{<:Real})
    n = length(freqs)
    n == 0 && return UnitRange{Int}[]
    n == 1 && return [1:1]
    dfs = abs.(diff(Float64.(freqs)))
    step = median(dfs)
    ranges = UnitRange{Int}[]
    lo = 1
    for i in 1:(n - 1)
        if dfs[i] > 3 * step
            push!(ranges, lo:i)
            lo = i + 1
        end
    end
    push!(ranges, lo:n)
    return ranges
end

"""
    segment_ranges(seg::AbstractFrequencySegmentation, geom::DataGeometry)
        -> Vector{UnitRange{Int}}

The global-channel range of each frequency segment of `seg` on `geom`, in
segment-id order. Defined wherever every segment is a contiguous channel run —
true of each shipped segmentation on the layouts it is meant for; a
segmentation whose segments interleave on `geom` (e.g. `PerSpectralWindow`
over interleaved spectral windows) has no range form and throws.
"""
function segment_ranges(seg::AbstractFrequencySegmentation, geom::DataGeometry)
    ids, nseg = freq_segment_ids(seg, geom)
    lo = fill(typemax(Int), nseg)
    hi = zeros(Int, nseg)
    count = zeros(Int, nseg)
    for (c, s) in pairs(ids)
        lo[s] = min(lo[s], c)
        hi[s] = max(hi[s], c)
        count[s] += 1
    end
    for s in 1:nseg
        count[s] > 0 || throw(
            ArgumentError("$(_seg_label(seg)): segment $s of $nseg covers no channel")
        )
        hi[s] - lo[s] + 1 == count[s] || throw(
            ArgumentError(
                "$(_seg_label(seg)): segment $s spans channels $(lo[s]):$(hi[s]) but holds " *
                    "only $(count[s]) of them — its channels interleave with another " *
                    "segment's, so it has no contiguous channel range."
            )
        )
    end
    return [lo[s]:hi[s] for s in 1:nseg]
end

# ── Placement on a foreign grid ──────────────────────────────────────────────
#
# Applying a solution to data it was not fit on asks one question per target
# sample: which segment of the solve does this sample belong to? It is answered
# by the identity the segmentation is defined on — a scan is a scan name, a spw
# is a spw name, a block is a bin of the solve's own formula — never by
# comparing coordinates against segment intervals, which would replace an exact
# answer with midpoints and boundary tolerances.
#
# A target sample with no such segment is an error. That is the whole of the
# "coarser is fine, finer is not" contract: a segmentation coarser than the data
# has a segment covering every sample by construction, and a finer one does not.

# Epoch/frequency identity tolerances — the same ones `leaf_window` joins a leaf
# to a geometry with, so a solution and the data it was solved on always agree.
const _EPOCH_ATOL = 1.0e-9      # hours
const _FREQ_RTOL = 1.0e-9

"""
    time_segment_ids(seg, solve::DataGeometry, target::DataGeometry;
                     ti_idx = eachindex(target.times), time_span = nothing) -> Vector{Int}

The solve-side time segment id of each `target` epoch selected by `ti_idx` — the
space `ComponentPlan.tseg_id` and a component's θ leaf are indexed by, so a
solution evaluates on `target`'s grid by reading these ids. `ti_idx` selects the
window to place; `time_span[k]` is the interval the `k`-th selected sample
integrates over (see `PartitionInfo.time_span`), checked against the solution's
own bins where the segmentation places by a formula on a coordinate.

Throws when a target sample falls in no segment of `seg` as the solve resolved
it, naming the segmentation and the sample.
"""
function time_segment_ids end

"""
    freq_segment_ids(seg, solve::DataGeometry, target::DataGeometry;
                     chan_idx = eachindex(target.channel_freqs)) -> Vector{Int}

The solve-side frequency segment id of each `target` channel selected by
`chan_idx`; the frequency counterpart of the foreign-grid
[`time_segment_ids`](@ref).
"""
function freq_segment_ids end

time_segment_ids(
    ::GlobalTime, ::DataGeometry, target::DataGeometry;
    ti_idx = eachindex(target.times), time_span = nothing,
) = ones(Int, length(ti_idx))

freq_segment_ids(
    ::GlobalFrequency, ::DataGeometry, target::DataGeometry;
    chan_idx = eachindex(target.channel_freqs),
) = ones(Int, length(chan_idx))

function time_segment_ids(
        seg::PerScan, solve::DataGeometry, target::DataGeometry;
        ti_idx = eachindex(target.times), time_span = nothing,
    )
    ids, _ = _dense_rank(solve.scan_of_time)
    # An identical labelling is the identity — the two grids agree sample for
    # sample, so no name is needed to say which scan a sample belongs to.
    solve.scan_of_time == target.scan_of_time && return ids[ti_idx]
    _require_names(seg, "scan", solve.scan_names, target.scan_names)
    tids, _ = _dense_rank(target.scan_of_time)
    of_name = Dict(solve.scan_names[s] => s for s in eachindex(solve.scan_names))
    return [
        _named_segment(seg, "scan", target.scan_names[tids[i]], of_name, solve.scan_names)
            for i in ti_idx
    ]
end

function freq_segment_ids(
        seg::PerSpectralWindow, solve::DataGeometry, target::DataGeometry;
        chan_idx = eachindex(target.channel_freqs),
    )
    ids, _ = _dense_rank(solve.spw_of_chan)
    solve.spw_of_chan == target.spw_of_chan && return ids[chan_idx]
    _require_names(seg, "spectral-window", solve.spw_names, target.spw_names)
    tids, _ = _dense_rank(target.spw_of_chan)
    of_name = Dict(solve.spw_names[s] => s for s in eachindex(solve.spw_names))
    return [
        _named_segment(seg, "spectral window", target.spw_names[tids[c]], of_name, solve.spw_names)
            for c in chan_idx
    ]
end

function _require_names(seg, what::String, solve_names, target_names)
    isempty(solve_names) && throw(
        ArgumentError(
            "$(_seg_label(seg)): the SOLUTION's geometry carries no $what names, so a foreign " *
                "grid cannot be placed — matching raw $what ids across two geometries is " *
                "positional matching, not identity. Build it with `build_geometry`, which names them."
        )
    )
    isempty(target_names) && throw(
        ArgumentError(
            "$(_seg_label(seg)): the TARGET geometry carries no $what names, so its samples " *
                "cannot be matched to the solution's $what segments by identity."
        )
    )
    return nothing
end

function _named_segment(seg, what::String, name::String, of_name, solved)
    s = get(of_name, name, 0)
    s == 0 && throw(
        ArgumentError(
            "$(_seg_label(seg)): the target $what $(repr(name)) is not in the solution, which " *
                "covers $(join(map(repr, solved), ", ")). A solution has no segment for data " *
                "it never saw; it is not extended to one."
        )
    )
    return s
end

function time_segment_ids(
        seg::Union{TimeBlocks, InstrumentScans}, solve::DataGeometry, target::DataGeometry;
        ti_idx = eachindex(target.times), time_span = nothing,
    )
    bin = _time_binner(seg, solve)
    of_bin = _bin_ids(map(bin, solve.times))
    _check_span_length(seg, time_span, ti_idx)
    out = Vector{Int}(undef, length(ti_idx))
    for (k, i) in enumerate(ti_idx)
        t = target.times[i]
        b = bin(t)
        # A sample integrating ACROSS a bin boundary would silently take
        # whichever bin its center landed in — the one place identity placement
        # still rests on a coordinate, so the span closes it.
        w = _span_at(time_span, k)
        (w <= 0 || (bin(t - w / 2) == b && bin(t + w / 2) == b)) || throw(
            ArgumentError(
                "$(_seg_label(seg)): the target epoch $t h integrates over $w h and crosses a " *
                    "segment boundary of the solution, which is segmented more finely than the " *
                    "data — no single segment applies."
            )
        )
        s = get(of_bin, b, 0)
        s == 0 && throw(
            ArgumentError(
                "$(_seg_label(seg)): the target epoch $t h falls in bin $b, which no solve " *
                    "epoch populated, so the solution has no segment covering it."
            )
        )
        out[k] = s
    end
    return out
end

function time_segment_ids(
        seg::PerIntegration, solve::DataGeometry, target::DataGeometry;
        ti_idx = eachindex(target.times), time_span = nothing,
    )
    perm = sortperm(solve.times)
    st = solve.times[perm]
    _check_span_length(seg, time_span, ti_idx)
    out = Vector{Int}(undef, length(ti_idx))
    for (k, i) in enumerate(ti_idx)
        t = target.times[i]
        j = searchsortedfirst(st, t - _EPOCH_ATOL)
        (j <= length(st) && abs(st[j] - t) <= _EPOCH_ATOL) || throw(
            ArgumentError(
                "$(_seg_label(seg)): the target epoch $t h is not a solve epoch — the nearest " *
                    "is $(_nearest(st, t)) h. One segment per solve integration cannot be " *
                    "resampled onto a different time grid."
            )
        )
        # Averaging epochs {1, 2, 3} h yields 2.0 h, which is a solve epoch, so
        # the match above does not by itself catch time-averaged data; the span
        # does — it still covers the epochs that were averaged away.
        w = _span_at(time_span, k)
        if w > 0
            lo = searchsortedfirst(st, t - w / 2)
            hi = searchsortedlast(st, t + w / 2)
            hi > lo && throw(
                ArgumentError(
                    "$(_seg_label(seg)): the target epoch $t h integrates over $w h, covering " *
                        "solve epochs $(join(st[lo:hi], ", ")) h — the solution is segmented " *
                        "more finely than the data and no single segment applies."
                )
            )
        end
        out[k] = perm[j]
    end
    return out
end

# `FreqGroups` and `ChannelBlocks` cut the channel INDEX axis, so they mean the
# same thing on another grid only when that grid indexes the same channels.
function freq_segment_ids(
        seg::Union{FreqGroups, ChannelBlocks}, solve::DataGeometry, target::DataGeometry;
        chan_idx = eachindex(target.channel_freqs),
    )
    _require_same_channels(seg, solve, target)
    ids, _ = freq_segment_ids(seg, solve)
    return ids[chan_idx]
end

function _require_same_channels(seg, solve::DataGeometry, target::DataGeometry)
    ns = nchannels(solve)
    nt = nchannels(target)
    ns == nt || throw(
        ArgumentError(
            "$(_seg_label(seg)) segments the channel axis by index, so it applies only to an " *
                "identical channel layout; the target has $nt channels and the solution $ns."
        )
    )
    for c in 1:ns
        isapprox(target.channel_freqs[c], solve.channel_freqs[c]; rtol = _FREQ_RTOL) || throw(
            ArgumentError(
                "$(_seg_label(seg)) segments the channel axis by index, so it applies only to " *
                    "an identical channel layout; target channel $c is at " *
                    "$(target.channel_freqs[c]) Hz and the solution's at " *
                    "$(solve.channel_freqs[c]) Hz."
            )
        )
    end
    return nothing
end

# First-appearance raw bin → dense segment id, the map `_dense_rank` builds
# implicitly. A bin no solve sample populated is absent from it and has no id,
# so a target sample landing there fails rather than reading the next
# populated bin's parameters.
function _bin_ids(bins)
    ids = Dict{Int, Int}()
    for b in bins
        get!(ids, b, length(ids) + 1)
    end
    return ids
end

_span_at(::Nothing, k) = 0.0
_span_at(span, k) = isempty(span) ? 0.0 : span[k]

function _check_span_length(seg, span, ti_idx)
    (span === nothing || isempty(span) || length(span) == length(ti_idx)) || throw(
        DimensionMismatch(
            "$(_seg_label(seg)): time_span has $(length(span)) entries for $(length(ti_idx)) " *
                "placed samples."
        )
    )
    return nothing
end

_nearest(sorted, t) = argmin(x -> abs(x - t), sorted)

_seg_label(seg::TimeBlocks) = "TimeBlocks($(seg.duration_hr))"
_seg_label(seg::ChannelBlocks) = "ChannelBlocks($(seg.block_size))"
_seg_label(seg) = string(nameof(typeof(seg)))
