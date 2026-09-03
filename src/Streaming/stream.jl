# ── The scan group: the streaming layer's unit of data flow ──────────────────
#
# A pass sees a lazy `UVSet` one SCAN GROUP at a time and never the set itself:
#
#   stream = scan_stream(uvset; transforms = [...])   # group + schedule, no read
#   stack, win = materialize_cube(stream, spec)       # one group, transforms applied
#   map_groups(stream) do spec ... end                # concurrent pass runner
#
# The data hook is the stream's TRANSFORM CHAIN (`AbstractDataTransform`, see
# transforms.jl), applied at every materialization, so a solve, a re-run, and a
# diagnostic that share a stream see identical data.

# ── Leaf grouping ─────────────────────────────────────────────────────────────

"""
    AbstractLeafGrouping

How a `UVSet`'s spw leaves are grouped into scan groups for streaming.
Built-ins: [`ByScan`](@ref) (the default), [`BySpw`](@ref), [`ByKey`](@ref).
"""
abstract type AbstractLeafGrouping end

"""
    ByScan()

Group leaves by `(source, scan)` — every spw of a scan is materialized
together (required by the multi-band fringe search and per-scan solves).
"""
struct ByScan <: AbstractLeafGrouping end

"""
    BySpw()

One group per spw leaf (`(source, scan, spw)` granularity) — no frequency
concatenation. For per-spw streaming work; steps that declare
`required_grouping(step) == :scan_complete` cannot run under it.
"""
struct BySpw <: AbstractLeafGrouping end

"""
    ByKey(f)

Custom grouping: leaves with equal `f(partition_key, leaf)` stream together.
Group order is first appearance in the set's branch order.
"""
struct ByKey{F} <: AbstractLeafGrouping
    f::F
end

_group_key(::ByScan, k, leaf) = begin
    info = UVData.metadata(leaf)
    (info.source_name, info.scan_name)
end
_group_key(::BySpw, k, leaf) = begin
    info = UVData.metadata(leaf)
    (info.source_name, info.scan_name, k)
end
_group_key(g::ByKey, k, leaf) = g.f(k, leaf)

# ── Stream construction ───────────────────────────────────────────────────────

"""
    ScanGroupSpec

One scheduled scan group: its `index` in stream order, `source`/`scan` names,
the `(partition_key, lazy_leaf)` pairs, and the memory `charge` (bytes) the
group costs while resident — what the outer scheduler's task count is checked
against, and what orders dispatch (heaviest first). Nothing is read until the
group is materialized.
"""
struct ScanGroupSpec{L}
    index::Int
    source::String
    scan::String
    leaves::Vector{L}            # (partition_key, lazy leaf) pairs
    charge::Int
end

"""
    ScanStream

A `UVSet` prepared for scan-group streaming: the ordered [`ScanGroupSpec`](@ref)s,
the data-transform chain applied at every materialization, and the
[`ExecutionConfig`](@ref) the run's parallelism, memory budget and progress
reporting come from. Build with [`scan_stream`](@ref); consume with
[`materialize_cube`](@ref) / [`materialize_leaves`](@ref) /
[`map_groups`](@ref).

`S` and `T` carry the group-spec and transform types the stream was built from.
Reach the schedulers with [`outer_executor`](@ref) / [`inner_executor`](@ref).
"""
struct ScanStream{G <: AbstractLeafGrouping, S <: ScanGroupSpec, T, X <: ExecutionConfig}
    uvset::UVSet
    geom::DataGeometry
    grouping::G
    groups::Vector{S}
    transforms::Vector{T}
    ant_names::Vector{String}
    exec::X
end

outer_executor(s::ScanStream) = outer_executor(s.exec)
inner_executor(s::ScanStream) = inner_executor(s.exec)
progress_callback(s::ScanStream) = progress_callback(s.exec)

# Ordered container over its scan-group specs — iteration reads only the
# precomputed group metadata, never visibilities.
Base.length(s::ScanStream) = length(s.groups)
Base.getindex(s::ScanStream, i) = s.groups[i]
Base.iterate(s::ScanStream, args...) = iterate(s.groups, args...)
Base.eltype(::Type{<:ScanStream{G, S}}) where {G, S} = S

Base.show(io::IO, s::ScanStream) =
    print(io, "ScanStream(", length(s.groups), " scan group(s), ", length(s.ant_names), " antennas)")

# One group's spec, labelled from its first leaf's partition metadata.
function _scan_group_spec(index::Int, keyed_leaves)
    info = UVData.metadata(last(first(keyed_leaves)))
    return ScanGroupSpec(
        index, String(info.source_name), String(info.scan_name),
        keyed_leaves, _spec_peak_bytes(keyed_leaves),
    )
end

# Peak resident bytes charged for one group: 12 B/cell native (8 vis + 4 weight)
# with a 2.5× headroom factor over the measured ~2× materialization peak (decode
# span + stacked cube + GC slack).
function _spec_peak_bytes(keyed_leaves)
    cells = 0
    for (_, leaf) in keyed_leaves
        cells += prod(size(leaf[:vis]))
    end
    return round(Int, 2.5 * 12 * cells)
end

# The deterministic memory budget (bytes): an explicit `mem_budget` if given,
# else `mem_fraction` of TOTAL physical RAM — total, not available, so the
# group concurrency a run picks is reproducible on a given box.
function _stream_budget(mem_fraction, mem_budget)
    mem_budget === nothing || return Float64(mem_budget)
    return mem_fraction * Float64(Sys.total_memory())
end

# The most tasks `sched` runs at once: what the memory budget is checked
# against, and what the nested decode pool is sized to. An UPPER BOUND — a
# scheduler whose chunk count follows the collection (`chunksize`), or that
# spawns one task per element (`chunking = false`), is bounded by the thread
# count instead. Read-only: a scheduler is used exactly as its owner configured
# it, never rebuilt (`nchunks` and `chunksize` are mutually exclusive, so there
# is no lossless way to override one).
max_tasks(::SerialScheduler) = 1
max_tasks(s::GreedyScheduler) = s.ntasks
max_tasks(s::Union{DynamicScheduler, StaticScheduler}) =
    (chunking_enabled(s) && has_nchunks(s)) ? nchunks(s) : Threads.nthreads()

"""
    scan_stream(uvset::UVSet; grouping = ByScan(), transforms = (),
                geom = build_geometry(uvset), exec = ExecutionConfig()) -> ScanStream

Prepare `uvset` for scan-group streaming WITHOUT reading data: group the lazy
leaves under `grouping` and charge each group's peak bytes. `exec` (an
[`ExecutionConfig`](@ref)) supplies the run-wide resources and is carried on the
stream: its two schedulers own the parallelism — the outer one decides how many
scan groups are resident at once, the inner one the within-scan fan-out — and
each is used exactly as configured. Construction FAILS if the outer scheduler's
task count times the largest group's charge exceeds `exec`'s memory budget.
`transforms` (a sequence of [`AbstractDataTransform`](@ref)) are applied, in
order, to every group as it is materialized — every consumer of the stream sees
identical, consistently-corrected data.
"""
function scan_stream(
        uvset::UVSet;
        grouping::AbstractLeafGrouping = ByScan(),
        transforms = (),
        geom::DataGeometry = build_geometry(uvset),
        exec::ExecutionConfig = ExecutionConfig(),
    )
    # Group lazy leaf references in branch order, first-seen key order.
    groups = Dict{Any, Vector{Any}}()
    order = Any[]
    for (k, leaf) in UVData.branches(uvset)
        key = _group_key(grouping, k, leaf)
        if !haskey(groups, key)
            groups[key] = Any[]
            push!(order, key)
        end
        push!(groups[key], (k, leaf))
    end
    specs = [
        _scan_group_spec(i, UVData._narrow_eltype(groups[key]))
            for (i, key) in enumerate(order)
    ]

    first_leaf = last(first(UVData.branches(uvset)))
    ant_names = String.(UVData.metadata(first_leaf).antennas.name)

    # Fail fast on transforms that cannot apply to this set (e.g. an
    # ApplySolution with a different channel layout) — a plain error here beats
    # a TaskFailedException out of a worker mid-pass.
    for t in transforms
        validate_transform(t, geom, ant_names)
    end

    _check_memory_budget(
        specs, outer_executor(exec), _stream_budget(exec.mem_fraction, exec.mem_budget),
    )
    return ScanStream(
        uvset, geom, grouping, specs, UVData._narrow_eltype(transforms), ant_names, exec,
    )
end

# The memory gate, checked once at construction rather than imposed by capping
# the caller's scheduler: the outer scheduler decides how many groups run at
# once, so a configuration that cannot fit is an error, not something to
# silently rewrite.
function _check_memory_budget(specs, outer, budget)
    peak = maximum(s -> s.charge, specs; init = 0)
    peak > 0 || return nothing
    concurrent = min(max_tasks(outer), max(length(specs), 1))
    # One group at a time is the floor: a single group larger than the budget is
    # the data's problem, not the schedule's, and there is no task count that
    # would fix it — run it and let the machine decide.
    concurrent <= 1 && return nothing
    concurrent * peak <= budget && return nothing
    gib(x) = round(x / 2^30; digits = 2)
    throw(
        ArgumentError(
            "the outer executor runs up to $concurrent scan groups at once (≈ $(gib(concurrent * peak)) " *
                "GiB at the largest group's charge) but the memory budget is $(gib(budget)) GiB — " *
                "lower the outer scheduler's task count, or raise the ExecutionConfig's " *
                "mem_fraction/mem_budget."
        )
    )
end

"""
    select_groups(stream::ScanStream, sel::AbstractScanSelection; snr = nothing)
        -> Vector{ScanGroupSpec}

The stream's groups filtered by a scan selection (fit-on-subset: a selective
pass materializes — and reads — only these). `snr` optionally supplies per-group
stage-A SNRs (by stream group index) for selections that need them (e.g. a
[`ScanWhere`](@ref) predicate reading `s.snr`); without it they see `NaN`.
Each selection record also carries the group's `stations` set (from lazy-leaf
metadata — no reads), consumed by coverage-aware selections and available to
`ScanWhere` predicates.
"""
function select_groups(stream::ScanStream, sel::AbstractScanSelection; snr = nothing)
    recs = [
        (;
            index = s.index, source = s.source, scan = s.scan,
            snr = snr === nothing ? NaN : Float64(snr[s.index]),
            stations = _spec_stations(s),
        )
            for s in stream.groups
    ]
    return stream.groups[select_scans(sel, recs)]
end

# Station set of one scan group, from lazy-leaf metadata (no reads).
function _spec_stations(spec::ScanGroupSpec)
    sts = Set{Int}()
    for (_, leaf) in spec.leaves
        for (a, b) in UVData.baselines(leaf).pairs
            a == b && continue
            push!(sts, a)
            push!(sts, b)
        end
    end
    return sts
end

# ── Materialization (the transform-chain choke points) ────────────────────────

"""
    materialize_cube(stream::ScanStream, spec::ScanGroupSpec;
                     executor = inner_executor(stream)) -> (stack, win)

Materialize one scan group as a frequency-concatenated cube and apply the
stream's transform chain to it. Fast path decodes each band directly into its
contiguous channel block of the stacked cube (one sequential read, no per-band
intermediates); falls back to materialize-then-copy when the group is not a
single sibling-band IDI span.

`stack` is a leaf-shaped `DimStack` built ONCE where the concatenated cube is
born: native-precision `:vis`/`:weights` layers on `(Frequency, Ti, Baseline,
Pol)` dims, with the first band leaf's `PartitionInfo` as metadata (source/scan
identity, antennas and baselines are group-wide; the frequency truth for the
concatenated axis lives on the `Frequency` lookup, NOT in
`metadata.frequencies`, which still describes that one band). `win` is the
group's [`GeometryWindow`](@ref) into the solve's index space.
"""
function materialize_cube(stream::ScanStream, spec::ScanGroupSpec; executor = inner_executor(stream))
    grp = _direct_scan_group(spec, stream.geom, executor)
    if grp === nothing
        leaves = UVData.materialize_group([l for (_, l) in spec.leaves]; executor)
        grp = _stacked_scan_group(leaves, stream.geom)
    end
    stack, win = grp
    apply_transforms!(stream.transforms, stack, win; executor)
    return stack, win
end

# Direct decode into the stacked cube: returns `nothing` (caller falls back) unless
# every band leaf maps to ONE full contiguous ascending channel block of the
# stacked frequency axis. Metadata comes from the LAZY leaves, so nothing is
# materialized until the decode, which runs under `executor`.
function _direct_scan_group(spec::ScanGroupSpec, geom::DataGeometry, executor)
    lazy = [l for (_, l) in spec.leaves]
    l0 = first(lazy)
    bl_pairs = collect(UVData.baselines(l0).pairs)
    pols = String.(pol_products(l0))
    tg = Float64.(lookup(l0[:vis], Ti))
    nti = length(tg)
    nbl = length(bl_pairs)
    npol = length(pols)

    chan_entries = Tuple{Int, Float64, Int, Int}[]      # (g_ci, freq, leafidx, local_c)
    nchan_leaf = Vector{Int}(undef, length(lazy))
    for (li, leaf) in enumerate(lazy)
        ci = leaf_window(geom, leaf).chan_idx
        fs = Float64.(lookup(leaf[:vis], Frequency))
        nchan_leaf[li] = length(ci)
        for (lc, gc) in enumerate(ci)
            push!(chan_entries, (gc, fs[lc], li, lc))
        end
    end
    sort!(chan_entries; by = e -> e[1])
    nchan = length(chan_entries)

    blocks = Vector{UnitRange{Int}}(undef, length(lazy))
    fill!(blocks, 1:0)
    i = 1
    while i <= nchan
        li = chan_entries[i][3]
        chan_entries[i][4] == 1 || return nothing        # run must start at local channel 1
        j = i
        while j < nchan && chan_entries[j + 1][3] == li &&
                chan_entries[j + 1][4] == chan_entries[j][4] + 1
            j += 1
        end
        (j - i + 1) == nchan_leaf[li] || return nothing   # run must cover the whole band
        isempty(blocks[li]) || return nothing             # each leaf exactly once
        blocks[li] = i:j
        i = j + 1
    end
    any(isempty, blocks) && return nothing

    Vg = Array{ComplexF32}(undef, nchan, nti, nbl, npol)
    Wg = Array{Float32}(undef, nchan, nti, nbl, npol)
    fg = Vector{Float64}(undef, nchan)
    g_ci = Vector{Int}(undef, nchan)
    for (row, e) in enumerate(chan_entries)
        fg[row] = e[2]
        g_ci[row] = e[1]
    end

    dests = [
        (view(Vg, blocks[li], :, :, :), view(Wg, blocks[li], :, :, :))
            for li in eachindex(lazy)
    ]
    UVData.materialize_group_into!(dests, lazy; executor) || return nothing

    info = UVData.metadata(l0)
    d = (Frequency(fg), Ti(tg), Baseline(copy(info.baselines.labels)), Pol(pols))
    return (
        DimStack((vis = DimArray(Vg, d), weights = DimArray(Wg, d)); metadata = info),
        GeometryWindow(geom, g_ci, leaf_window(geom, l0).ti_idx),
    )
end

# Copy a contiguous channel-block into the stacked cubes (function barrier —
# `V`/`W` from `parent(leaf[:vis])` are type-unstable at the call site; the
# `@simd` loop over the stride-1 channel axis needs the specialization).
function _cube_block!(
        Vg::Array{ComplexF32, 4}, Wg::Array{Float32, 4}, V, W,
        dst0::Int, lc0::Int, nbc::Int,
    )
    _, nti, nbl, npol = size(Vg)
    @inbounds for p in 1:npol, bl in 1:nbl, ti in 1:nti
        @simd for c in 0:(nbc - 1)
            Vg[dst0 + c, ti, bl, p] = V[lc0 + c, ti, bl, p]
            Wg[dst0 + c, ti, bl, p] = W[lc0 + c, ti, bl, p]
        end
    end
    return nothing
end

# Stack already-materialized sibling band leaves along frequency, sorted by
# global channel index; block copies walk maximal same-leaf channel runs —
# cache-friendly on the stride-1 axis.
function _stacked_scan_group(leaves, geom::DataGeometry)
    l0 = first(leaves)
    bl_pairs = collect(UVData.baselines(l0).pairs)
    pols = String.(pol_products(l0))
    tg = Float64.(lookup(l0[:vis], Ti))
    nti = length(tg)
    nbl = length(bl_pairs)
    npol = length(pols)

    chan_entries = Tuple{Int, Float64, Int, Int}[]      # (g_ci, freq, leafidx, local_c)
    for (li, leaf) in enumerate(leaves)
        ci = leaf_window(geom, leaf).chan_idx
        fs = Float64.(lookup(leaf[:vis], Frequency))
        for (lc, gc) in enumerate(ci)
            push!(chan_entries, (gc, fs[lc], li, lc))
        end
    end
    sort!(chan_entries; by = e -> e[1])
    nchan = length(chan_entries)

    Vg = Array{ComplexF32}(undef, nchan, nti, nbl, npol)
    Wg = Array{Float32}(undef, nchan, nti, nbl, npol)
    fg = Vector{Float64}(undef, nchan)
    g_ci = Vector{Int}(undef, nchan)
    for (row, e) in enumerate(chan_entries)
        fg[row] = e[2]
        g_ci[row] = e[1]
    end

    i = 1
    @inbounds while i <= nchan
        li = chan_entries[i][3]
        lc0 = chan_entries[i][4]
        j = i
        while j < nchan && chan_entries[j + 1][3] == li &&
                chan_entries[j + 1][4] == chan_entries[j][4] + 1
            j += 1
        end
        nbc = j - i + 1
        V = parent(leaves[li][:vis])
        W = parent(leaves[li][:weights])
        _cube_block!(Vg, Wg, V, W, i, lc0, nbc)
        i = j + 1
    end

    info = UVData.metadata(l0)
    d = (Frequency(fg), Ti(tg), Baseline(copy(info.baselines.labels)), Pol(pols))
    return (
        DimStack((vis = DimArray(Vg, d), weights = DimArray(Wg, d)); metadata = info),
        GeometryWindow(geom, g_ci, leaf_window(geom, l0).ti_idx),
    )
end

"""
    materialize_leaves(stream::ScanStream, spec::ScanGroupSpec;
                       executor = inner_executor(stream)) -> Vector{Tuple}

Materialize one scan group as its per-band `(partition_key, leaf)` pairs — no
concatenated copy (the memory-lean path for leaf-wise passes) — with the
stream's transform chain applied to each leaf. Lazy-sourced leaves are
transformed in place (their arrays are freshly materialized, hence private); an
eager source's leaf is the caller's own data, so it is copied first — the
caller's `UVSet` is never mutated.
"""
function materialize_leaves(stream::ScanStream, spec::ScanGroupSpec; executor = inner_executor(stream))
    keyed = [
        (k, m) for ((k, _), m) in zip(
                spec.leaves,
                UVData.materialize_group([l for (_, l) in spec.leaves]; executor),
            )
    ]
    isempty(stream.transforms) && return keyed
    private = all(((_, l),) -> UVData.is_lazy(l), spec.leaves)
    # `Tuple` is spelled out because the untyped `tmap` rejects `GreedyScheduler`,
    # which `inner_executor` accepts; the typed form writes the output by index
    # under every scheduler.
    return tmap(Tuple, keyed; scheduler = executor) do (k, m)
        (k, _transform_leaf(stream, spec, m; copy_arrays = !private))
    end
end

# Run the transform chain over one materialized band leaf: the chain's stack is
# the layer selection off the leaf ITSELF (`leaf[(:vis, :weights)]` — metadata
# and all; no decomposition), which shares the leaf's arrays, so the chain
# mutates the leaf in place and `base` is the transformed leaf. `copy_arrays`
# guards user-owned data (eager sources): the leaf is rewrapped around array
# copies first, exactly as `apply_calibration` does; private (freshly-materialized)
# leaves are transformed in place with no scan-sized copy.
function _transform_leaf(stream::ScanStream, spec::ScanGroupSpec, leaf; copy_arrays::Bool)
    base = copy_arrays ?
        rebuild_visibilities(leaf, copy(parent(leaf[:vis])), copy(parent(leaf[:weights]))) : leaf
    # Per-leaf work is already fanned out across leaves; keep transforms serial here.
    apply_transforms!(
        stream.transforms, base[(:vis, :weights)], leaf_window(stream.geom, base);
        executor = SerialScheduler(),
    )
    return base
end

# ── The pass runner: concurrent group execution (the executor seam) ──────────

const _STREAM_PROGRESS_LOCK = ReentrantLock()

_stream_progress(::Nothing, stage, done, total) = nothing
function _stream_progress(cb, stage, done, total)
    lock(_STREAM_PROGRESS_LOCK) do
        try
            cb(stage, Int(done), Int(total))
        catch err
            @warn "progress callback failed" stage err maxlog = 1
        end
    end
    return nothing
end

"""
    map_groups(work, stream::ScanStream; selection = AllScans(), snr = nothing,
               progress = progress_callback(stream), stage = :pass) -> Vector

Run `work(spec::ScanGroupSpec)` over the stream's (selected) groups on the
stream's outer scheduler, heaviest group first so the long poles start
immediately. Results return in group order. `work` must be independent across
groups (all fringe passes qualify: disjoint per-scan θ slots / per-scan outputs).

The outer scheduler alone decides how many groups run at once; [`scan_stream`](@ref)
has already checked that many against the run's memory budget.

This is the EXECUTOR SEAM: every full-data pass of the pipeline runs through
here. `progress` defaults to the callback on the stream's
[`ExecutionConfig`](@ref) and, when not `nothing`, is called
`(stage, done, total)` per completed group.
"""
function map_groups(
        work::F, stream::ScanStream;
        selection::AbstractScanSelection = AllScans(), snr = nothing,
        progress = progress_callback(stream), stage::Symbol = :pass,
    ) where {F}
    specs = selection isa AllScans ? stream.groups : select_groups(stream, selection; snr)
    total = length(specs)
    _stream_progress(progress, stage, 0, total)
    done = Threads.Atomic{Int}(0)
    function wrapped(spec)
        r = work(spec)
        _stream_progress(progress, stage, Threads.atomic_add!(done, 1) + 1, total)
        return r
    end
    return _scheduled_map(
        wrapped, specs, [s.charge for s in specs];
        executor = outer_executor(stream),
    )
end

"""
    foreach_group(work, stream; kwargs...) -> nothing

[`map_groups`](@ref) discarding results.
"""
foreach_group(work::F, stream::ScanStream; kwargs...) where {F} =
    (map_groups(work, stream; kwargs...); nothing)

# Largest-first parallel map: run `work` over `items` on `executor`, dispatching
# the heaviest item (by `charges`) first so the long poles start immediately.
# Results in `items` order. A failed worker rethrows after the other workers
# drain the queue. `executor` is used exactly as configured — how many items run
# at once is ITS decision, checked against the memory budget upstream.
#
# Each backend fills an `Any` sink and returns `map(identity, sink)`: `work`'s
# return type is not known before it runs, and tasks write their slots
# concurrently, so the sink has to admit any value; `map` then recovers the
# concrete element type for whatever consumes the pass.
function _scheduled_map(work::F, items, charges; executor = SerialScheduler()) where {F}
    length(items) == length(charges) || throw(
        DimensionMismatch("items and charges must match: $(length(items)) vs $(length(charges))"),
    )
    Base.require_one_based_indexing(items, charges)
    return _scheduled_map(executor, work, items, charges)
end

# The SERIAL outer backend: each group runs to completion on the calling task in
# `items` order — with one worker the dispatch order cannot matter. Within-scan
# fan-out still goes through the inner executor.
function _scheduled_map(::SerialScheduler, work::F, items, charges) where {F}
    return map(work, items)
end

# The THREADED outer backend: `sched` runs the groups over the charge-sorted
# index order. `GreedyScheduler` is the one that keeps largest-first meaningful
# under uneven charges — it hands each task the next group off the queue — where
# a chunking scheduler assigns groups to tasks up front. A backend with
# different task-lifetime needs adds its own method on its executor type; the
# seam is open by dispatch.
function _scheduled_map(sched::Scheduler, work::F, items, charges) where {F}
    out = Vector{Any}(undef, length(items))
    tforeach(sortperm(charges; rev = true); scheduler = sched) do k
        out[k] = work(items[k])
    end
    return map(identity, out)
end
