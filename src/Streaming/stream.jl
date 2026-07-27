# ── The scan group: the streaming layer's unit of data flow ──────────────────
#
# A pass sees a lazy `UVSet` one SCAN GROUP at a time and never the set itself:
#
#   stream = scan_stream(uvset; transforms = [...])   # group + budget, no read
#   grp    = materialize_cube(stream, spec)           # one group, transforms applied
#   map_groups(stream) do spec ... end                # budget-admitted pass runner
#
# The data hook is the stream's TRANSFORM CHAIN (`AbstractDataTransform`, see
# transforms.jl), applied at every materialization, so a solve, a re-run, and a
# diagnostic that share a stream see identical data.

# ── Leaf grouping ─────────────────────────────────────────────────────────────

"""
    AbstractLeafGrouping

How a `UVSet`'s band leaves are grouped into scan groups for streaming.
Built-ins: [`ByScan`](@ref) (the default), [`ByBand`](@ref), [`ByKey`](@ref).
"""
abstract type AbstractLeafGrouping end

"""
    ByScan()

Group leaves by `(source, scan)` — every band of a scan is materialized
together (required by the multi-band fringe search and per-scan solves).
"""
struct ByScan <: AbstractLeafGrouping end

"""
    ByBand()

One group per band leaf (`(source, scan, band)` granularity) — no frequency
concatenation. For per-band streaming work; steps that declare
`required_grouping(step) == :scan_complete` cannot run under it.
"""
struct ByBand <: AbstractLeafGrouping end

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
_group_key(::ByBand, k, leaf) = begin
    info = UVData.metadata(leaf)
    (info.source_name, info.scan_name, k)
end
_group_key(g::ByKey, k, leaf) = g.f(k, leaf)

# ── Stream construction ───────────────────────────────────────────────────────

"""
    ScanGroupSpec

One scheduled scan group: its `index` in stream order, `source`/`scan` names,
the `(partition_key, lazy_leaf)` pairs, and the memory `charge` (bytes) used by
the budget scheduler. Nothing is read until the group is materialized.
"""
struct ScanGroupSpec
    index::Int
    source::String
    scan::String
    leaves::Vector{Any}          # (partition_key, lazy leaf) pairs
    charge::Int
end

"""
    ScanStream

A `UVSet` prepared for scan-group streaming: the ordered [`ScanGroupSpec`](@ref)s,
the data-transform chain applied at every materialization, and the run's
deterministic resource sizing (memory budget, group concurrency `ntasks`,
per-group task budget `inner`, per-task scratch pool). Build with
[`scan_stream`](@ref); consume with [`materialize_cube`](@ref) /
[`materialize_leaves`](@ref) / [`map_groups`](@ref).

`W` is the element type of the scratch `pool`, set by `scan_stream`'s
`workspace` factory. It is `Nothing` — an empty pool — unless a factory is
given, so a consumer whose kernels need scratch (`Gustavo.Fringe`'s FFT
workspaces) asks for it by name and this module never names it.
"""
struct ScanStream{G <: AbstractLeafGrouping, W}
    uvset::UVSet
    geom::DataGeometry
    grouping::G
    groups::Vector{ScanGroupSpec}
    transforms::Vector{Any}
    ant_names::Vector{String}
    budget::Float64
    ntasks::Int
    inner::Int
    pool::Channel{W}
    executor::Executors.AbstractExecutor
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
# admission decisions a run makes are reproducible on a given box.
function _stream_budget(mem_fraction, mem_budget)
    mem_budget === nothing || return Float64(mem_budget)
    return mem_fraction * Float64(Sys.total_memory())
end

# The stream's per-task scratch pool: `n` objects from the `workspace` factory,
# or an empty `Channel{Nothing}` when no factory is given. The element type is
# what a consumer checks to tell a stream that carries its scratch from one
# that does not.
_workspace_pool(::Nothing, ::Int) = Channel{Nothing}(1)
function _workspace_pool(workspace, n::Int)
    ws = [workspace() for _ in 1:n]
    pool = Channel{eltype(ws)}(n)
    foreach(w -> put!(pool, w), ws)
    return pool
end

"""
    scan_stream(uvset::UVSet; grouping = ByScan(), transforms = (),
                geom = build_geometry(uvset), ntasks = Threads.nthreads(),
                mem_fraction = 0.6, mem_budget = nothing, workspace = nothing,
                executor = current_executor()) -> ScanStream

Prepare `uvset` for scan-group streaming WITHOUT reading data: group the lazy
leaves under `grouping`, charge each group's peak bytes, and size the run's
deterministic concurrency — `ntasks` is capped so `ntasks × largest-group
charge` fits the memory budget, and the leftover threads become each group's
`inner` fan-out. `transforms` (a sequence of [`AbstractDataTransform`](@ref))
are applied, in order, to every group as it is materialized — every consumer of
the stream sees identical, consistently-corrected data.

`workspace` is a zero-argument factory for the stream's per-task scratch pool,
one object per thread, borrowed and returned by consumers that need scratch
that outlives a single group. `nothing` builds no pool. The fringe search
needs one:

    stream = scan_stream(uvset; workspace = Gustavo.Fringe.FringeWorkspace)
"""
function scan_stream(
        uvset::UVSet;
        grouping::AbstractLeafGrouping = ByScan(),
        transforms = (),
        geom::DataGeometry = build_geometry(uvset),
        ntasks::Integer = Threads.nthreads(),
        mem_fraction::Real = 0.6,
        mem_budget = nothing,
        workspace = nothing,
        executor::Executors.AbstractExecutor = Executors.current_executor(),
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
    specs = Vector{ScanGroupSpec}(undef, length(order))
    for (i, key) in enumerate(order)
        kl = groups[key]
        info = UVData.metadata(last(first(kl)))
        specs[i] = ScanGroupSpec(
            i, String(info.source_name), String(info.scan_name), kl, _spec_peak_bytes(kl),
        )
    end

    first_leaf = last(first(UVData.branches(uvset)))
    ant_names = String.(UVData.metadata(first_leaf).antennas.name)

    # Fail fast on transforms that cannot apply to this set (e.g. an
    # ApplySolution with a different channel layout) — a plain error here beats
    # a TaskFailedException out of a worker mid-pass.
    for t in transforms
        validate_transform(t, geom, ant_names)
    end

    budget = _stream_budget(mem_fraction, mem_budget)
    requested = max(1, min(Int(ntasks), max(length(specs), 1)))
    peak = maximum(s -> s.charge, specs; init = 0)
    ntasks_use = peak <= 0 ? requested : min(requested, max(1, Int(floor(budget / peak))))
    inner = max(1, Threads.nthreads() ÷ ntasks_use)

    pool = _workspace_pool(workspace, max(Threads.nthreads(), 1))

    return ScanStream(
        uvset, geom, grouping, specs, collect(Any, transforms), ant_names,
        budget, ntasks_use, inner, pool, executor,
    )
end

"""
    select_groups(stream::ScanStream, sel::AbstractScanSelection; snr = nothing)
        -> Vector{ScanGroupSpec}

The stream's groups filtered by a scan selection (fit-on-subset: a selective
pass materializes — and reads — only these). `snr` optionally supplies per-group
stage-A SNRs (by stream group index) for selections that need them
([`BrightestCalibrator`](@ref)); without it they see `NaN`. Each selection
record also carries the group's `stations` set (from lazy-leaf metadata — no
reads), consumed by coverage-aware selections and available to `ScanWhere`
predicates.
"""
function select_groups(stream::ScanStream, sel::AbstractScanSelection; snr = nothing)
    recs = [
        (; index = s.index, source = s.source, scan = s.scan,
            snr = snr === nothing ? NaN : Float64(snr[s.index]),
            stations = _spec_stations(s))
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
    ScanGroup

One materialized scan group, frequency-concatenated across its band leaves.
`data` is the leaf-shaped `DimStack` built ONCE where the concatenated cube is
born: native-precision `:vis`/`:weights` layers on
`(Frequency, Ti, Baseline, Pol)` dims, with the first band leaf's
`PartitionInfo` as metadata (source/scan identity, antennas, and baselines are
group-wide; the frequency truth for the concatenated axis lives on the
`Frequency` lookup, NOT in `metadata.frequencies`, which still describes that
one band). `g_ci`/`g_ti` are the global geometry indices.

Convenience properties, all derived from `data`: `Vg`/`Wg` (the parent cubes,
`(nchan, nti, nbl, npol)`), `fg` (Hz), `tg` (hours), `bl_pairs`,
`pol_products`, `source`, `scan`.
"""
struct ScanGroup{S <: AbstractDimStack}
    data::S
    g_ci::Vector{Int}
    g_ti::Vector{Int}
end

function Base.getproperty(g::ScanGroup, s::Symbol)
    s === :Vg && return parent(getfield(g, :data)[:vis])
    s === :Wg && return parent(getfield(g, :data)[:weights])
    s === :fg && return parent(lookup(getfield(g, :data)[:vis], Frequency))
    s === :tg && return parent(lookup(getfield(g, :data)[:vis], Ti))
    s === :pol_products && return parent(lookup(getfield(g, :data)[:vis], Pol))
    s === :bl_pairs && return DimensionalData.metadata(getfield(g, :data)).baselines.pairs
    s === :source && return String(DimensionalData.metadata(getfield(g, :data)).source_name)
    s === :scan && return String(DimensionalData.metadata(getfield(g, :data)).scan_name)
    return getfield(g, s)
end
Base.propertynames(::ScanGroup) = (
    :data, :g_ci, :g_ti, :Vg, :Wg, :fg, :tg, :bl_pairs, :pol_products,
    :source, :scan,
)

"""
    scan_view(stream::ScanStream, grp::ScanGroup) -> ScanDataView

The transform-facing window over a materialized group — a zero-cost wrap: the
view's `data` IS the group's stack (mutating the view mutates the group).
"""
scan_view(stream::ScanStream, grp::ScanGroup) =
    ScanDataView(grp.data, grp.g_ci, grp.g_ti, stream.geom)

"""
    materialize_cube(stream::ScanStream, spec::ScanGroupSpec;
                     inner = stream.inner) -> ScanGroup

Materialize one scan group as the frequency-concatenated [`ScanGroup`](@ref)
cube and apply the stream's transform chain to it. Fast path decodes each band
directly into its contiguous channel block of the stacked cube (one sequential
read, no per-band intermediates); falls back to materialize-then-copy when the
group is not a single sibling-band IDI span.
"""
function materialize_cube(stream::ScanStream, spec::ScanGroupSpec; inner::Integer = stream.inner)
    grp = _direct_scan_group(spec, stream.geom)
    if grp === nothing
        leaves = UVData.materialize_group(
            [l for (_, l) in spec.leaves]; layers = (:vis, :weights, :uvw),
        )
        grp = _stacked_scan_group(leaves, stream.geom)
    end
    apply_transforms!(stream.transforms, scan_view(stream, grp); inner = inner)
    return grp
end

# Direct decode into the stacked cube: returns `nothing` (caller falls back) unless
# every band leaf maps to ONE full contiguous ascending channel block of the
# stacked frequency axis. Metadata comes from the LAZY leaves, so nothing is
# materialized until the decode.
function _direct_scan_group(spec::ScanGroupSpec, geom::DataGeometry)
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
        ci, _ = leaf_window(geom, leaf)
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
    UVData.materialize_group_into!(dests, lazy; layers = (:vis, :weights)) || return nothing

    _, g_ti = leaf_window(geom, l0)
    info = UVData.metadata(l0)
    d = (Frequency(fg), Ti(tg), Baseline(copy(info.baselines.labels)), Pol(pols))
    return ScanGroup(
        DimStack((vis = DimArray(Vg, d), weights = DimArray(Wg, d)); metadata = info),
        g_ci, g_ti,
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
        ci, _ = leaf_window(geom, leaf)
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

    _, g_ti = leaf_window(geom, l0)
    info = UVData.metadata(l0)
    d = (Frequency(fg), Ti(tg), Baseline(copy(info.baselines.labels)), Pol(pols))
    return ScanGroup(
        DimStack((vis = DimArray(Vg, d), weights = DimArray(Wg, d)); metadata = info),
        g_ci, g_ti,
    )
end

"""
    materialize_leaves(stream::ScanStream, spec::ScanGroupSpec;
                       inner = stream.inner) -> Vector{Tuple}

Materialize one scan group as its per-band `(partition_key, leaf)` pairs — no
concatenated copy (the memory-lean path for leaf-wise passes) — with the
stream's transform chain applied to each leaf. Lazy-sourced leaves are
transformed in place (their arrays are freshly materialized, hence private); an
eager source's leaf is the caller's own data, so it is copied first — the
caller's `UVSet` is never mutated.
"""
function materialize_leaves(stream::ScanStream, spec::ScanGroupSpec; inner::Integer = stream.inner)
    keyed = [
        (k, m) for ((k, _), m) in zip(
            spec.leaves,
            UVData.materialize_group([l for (_, l) in spec.leaves]; layers = (:vis, :weights, :uvw)),
        )
    ]
    isempty(stream.transforms) && return keyed
    private = all(((_, l),) -> UVData.is_lazy(l), spec.leaves)
    out = Vector{Any}(undef, length(keyed))
    nt = clamp(Int(inner), 1, length(keyed))
    exec_foreach(eachindex(keyed); ntasks = nt) do i
        k, m = keyed[i]
        out[i] = (k, _transform_leaf(stream, spec, m; copy_arrays = !private))
    end
    return [out[i]::Tuple for i in eachindex(out)]
end

# Run the transform chain over one materialized band leaf: the view's stack is
# the layer selection off the leaf ITSELF (`leaf[(:vis, :weights)]` — metadata
# and all; no decomposition). `copy_arrays` guards user-owned data (eager
# sources): the leaf is rewrapped around array copies first, exactly as
# `apply_calibration` does; private (freshly-materialized) leaves are
# transformed in place with no scan-sized copy. Either way the returned leaf's
# `flag` layer is re-derived from the TRANSFORMED weights — the pre-transform
# flag would be stale after e.g. `FlagChannels`.
function _transform_leaf(stream::ScanStream, spec::ScanGroupSpec, leaf; copy_arrays::Bool)
    base = copy_arrays ?
        with_visibilities(leaf, copy(parent(leaf[:vis])), copy(parent(leaf[:weights]))) : leaf
    ci, ti = leaf_window(stream.geom, base)
    v = ScanDataView(base[(:vis, :weights)], collect(Int, ci), collect(Int, ti), stream.geom)
    # Per-leaf work is already fanned out across leaves; keep transforms serial here.
    apply_transforms!(stream.transforms, v; inner = 1)
    return with_visibilities(base, v.vis, v.weights)
end

# ── The pass runner: budget-admitted group execution (the executor seam) ─────

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
               progress = nothing, stage = :pass) -> Vector

Run `work(spec::ScanGroupSpec)` over the stream's (selected) groups under
budget-admitted concurrency: each group is admitted against the stream's memory
budget by ITS OWN charge — big groups run (nearly) alone, small groups pack
into the leftover budget, largest-first so the long poles start immediately.
Results return in group order. `work` must be independent across groups (all
fringe passes qualify: disjoint per-scan θ slots / per-scan outputs).

This is the EXECUTOR SEAM: every full-data pass of the pipeline runs through
here (and the Dagger executor later replaces only this implementation).
`progress`, if given, is called `(stage, done, total)` per completed group.
"""
function map_groups(
        work::F, stream::ScanStream;
        selection::AbstractScanSelection = AllScans(), snr = nothing,
        progress = nothing, stage::Symbol = :pass,
    ) where {F}
    specs = selection isa AllScans ? stream.groups : select_groups(stream, selection; snr)
    total = length(specs)
    _stream_progress(progress, stage, 0, total)
    done = Threads.Atomic{Int}(0)
    wrapped = function (spec)
        r = work(spec)
        _stream_progress(progress, stage, Threads.atomic_add!(done, 1) + 1, total)
        return r
    end
    # Bind the stream's executor as the ambient one for the whole pass, so the
    # group tasks AND every fan-out nested inside `work` (decode, search
    # chunks, refine loops) run on the same backend.
    results, _ = Executors.with_executor(stream.executor) do
        _scheduled_map(
            wrapped, specs, [s.charge for s in specs], stream.budget;
            max_tasks = stream.ntasks, executor = stream.executor,
        )
    end
    return results
end

"""
    foreach_group(work, stream; kwargs...) -> nothing

[`map_groups`](@ref) discarding results.
"""
foreach_group(work::F, stream::ScanStream; kwargs...) where {F} =
    (map_groups(work, stream; kwargs...); nothing)

# Budget-admission scheduler: admit each item by its own charge against a shared budget; among the items
# that currently fit, the LARGEST starts first; an item charged more than the
# whole budget is clamped so it still runs (alone). Results in `items` order;
# returns `(results, peak_concurrency)`. A failed worker rethrows after the
# other workers drain the queue. The `executor` picks the task backend — the
# ADMISSION policy (budget, largest-first, max_tasks cap) is identical under
# both, so which groups ever run concurrently does not depend on the backend.
function _scheduled_map(
        work::F, items, charges, budget;
        max_tasks::Integer, executor::Executors.AbstractExecutor = Executors.ThreadsExecutor(),
    ) where {F}
    n = length(items)
    n == 0 && return Any[], 0
    executor isa Executors.DaggerExecutor &&
        return _scheduled_map_dagger(work, items, charges, budget; max_tasks)
    out = Vector{Any}(undef, n)
    remaining = sort(collect(1:n); by = k -> -Float64(charges[k]))
    cond = Threads.Condition()
    avail = Ref(Float64(budget))
    inflight = Ref(0)
    peak = Ref(0)
    workers = map(1:max(1, min(Int(max_tasks), n))) do _
        Threads.@spawn while true
            k = 0
            amt = 0.0
            lock(cond)
            try
                while true
                    isempty(remaining) && break
                    # First (= largest) not-yet-started item that fits; when none
                    # fits, wait for a release. An over-budget item is clamped, so
                    # with nothing in flight SOMETHING always fits — no deadlock.
                    j = findfirst(kk -> min(Float64(charges[kk]), Float64(budget)) <= avail[], remaining)
                    if j === nothing
                        wait(cond)
                    else
                        k = remaining[j]
                        deleteat!(remaining, j)
                        amt = min(Float64(charges[k]), Float64(budget))
                        avail[] -= amt
                        inflight[] += 1
                        peak[] = max(peak[], inflight[])
                        break
                    end
                end
            finally
                unlock(cond)
            end
            k == 0 && break
            try
                out[k] = work(items[k])
            finally
                lock(cond)
                try
                    avail[] += amt
                    inflight[] -= 1
                    notify(cond)
                finally
                    unlock(cond)
                end
            end
        end
    end
    foreach(wait, workers)
    return map(identity, out), peak[]
end

# The Dagger backend of `_scheduled_map`: one `Dagger.@spawn` per item instead
# of a worker pool, submitted by the SAME admission policy (largest-first, own
# charge against the shared budget, `max_tasks` in-flight cap). Each task
# releases its charge as it finishes; the submitter blocks on the condition
# when nothing fits. Results are fetched — and a failed task rethrown — in
# `items` order after all submissions.
function _scheduled_map_dagger(work::F, items, charges, budget; max_tasks::Integer) where {F}
    n = length(items)
    cap = max(1, min(Int(max_tasks), n))
    tasks = Vector{Any}(undef, n)
    remaining = sort(collect(1:n); by = k -> -Float64(charges[k]))
    cond = Threads.Condition()
    avail = Ref(Float64(budget))
    inflight = Ref(0)
    peak = Ref(0)
    while !isempty(remaining)
        k = 0
        amt = 0.0
        lock(cond)
        try
            while true
                # First (= largest) not-yet-started item that fits; when none
                # fits (or the in-flight cap is reached), wait for a release.
                # An over-budget item is clamped, so with nothing in flight
                # SOMETHING always fits — no deadlock.
                j = inflight[] < cap ?
                    findfirst(kk -> min(Float64(charges[kk]), Float64(budget)) <= avail[], remaining) :
                    nothing
                if j === nothing
                    wait(cond)
                else
                    k = remaining[j]
                    deleteat!(remaining, j)
                    amt = min(Float64(charges[k]), Float64(budget))
                    avail[] -= amt
                    inflight[] += 1
                    peak[] = max(peak[], inflight[])
                    break
                end
            end
        finally
            unlock(cond)
        end
        body = let k = k, amt = amt
            function ()
                try
                    return work(items[k])
                finally
                    lock(cond)
                    try
                        avail[] += amt
                        inflight[] -= 1
                        notify(cond)
                    finally
                        unlock(cond)
                    end
                    # Dagger retention: a completed thunk — and the scan-sized
                    # data its closure captured — is freed only after a GC run
                    # collects the dropped DTask handles, their finalizers
                    # enqueue the scheduler cleanup, and the scheduler
                    # processes it. Julia's GC pacing lags that pipeline under
                    # load (RSS climbs by ~a cube per scan), so force the
                    # collection as each group retires — the same per-scan GC
                    # discipline the drivers use (<1 s against a ~15 s scan).
                    GC.gc()
                end
            end
        end
        # blocking = true: group tasks spend much of their time fetching their
        # own nested fan-outs (decode/search/refine chunks).
        tasks[k] = Executors._spawn(Executors.DaggerExecutor(), body, true)
    end
    out = Vector{Any}(undef, n)
    for k in 1:n
        out[k] = Executors.exec_fetch(tasks[k])
    end
    return map(identity, out), peak[]
end

