# ── solve_fringes: the end-to-end fringe-fitting pipeline ────────────────────
#
# Tie the tested building blocks together into one solve over a whole `UVSet`:
#
#   1. build a shared per-feed fringe `StationGainModel` (per-scan constant phase
#      + delay + rate, plus a per-integration adhoc-phase component);
#   2. for each (source, scan) group of sibling band leaves: concatenate bands
#      along frequency, fringe-search every (baseline, product), stationize to
#      per-(station, feed) delay/rate/phase, pack into θ;
#   3. divide out the Stage-B station gains, coherently frequency-average the
#      residual per (baseline, product, AP), solve the globally-closing adhoc
#      phase track, pack it into θ.
#
# The output is a `CalibrationSolution` that `apply_calibration` flattens the
# data with. Time units: the search/stationize layer works in SECONDS, so
# scan times (hours) and t0 (hours) are multiplied by 3600 on the way in; the
# Rate term stores its parameter in Hz, matching what `stationize_scan` returns.

import DimensionalData
using DimensionalData: lookup, dims, Ti
using ..UVData: Frequency
using Statistics: mean
using LinearAlgebra: BLAS

# Run `f` with BLAS pinned to a single thread, restoring the prior setting after.
# The fringe solve threads over scan groups with OhMyThreads; if BLAS also
# spawns threads, every task's WLS/QR solve fans out `BLAS.get_num_threads()`
# threads, so `ntasks × blas_threads` (e.g. 8 × 8 = 64) oversubscribe the cores
# and contend — threads sit "runnable" while only ~1 core makes progress. One
# BLAS thread per task is the correct split when the parallelism is across tasks.
function _with_single_blas_thread(f)
    old = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(old)
    end
end

# Run `f` with FFTW using `n` threads per transform, restoring 1 (FFTW's default)
# after. The fringe SEARCH is ~96% FFT, but the memory cap pins group-parallelism
# (`ntasks`) well below the core count on a big file — leaving cores idle DURING
# the search. Giving each FFT `n = nthreads ÷ ntasks` threads uses them
# (`ntasks × n ≈ nthreads`), so the otherwise-idle cores accelerate the transforms
# instead of sitting out the bottleneck phase. Plans are built lazily inside the
# threaded passes, so this must wrap them to take effect.
function _with_fft_threads(f, n::Int)
    FFTW.set_num_threads(max(1, n))
    try
        return f()
    finally
        FFTW.set_num_threads(1)
    end
end

# Run `f` with the bulk reader decoding each leaf over `n` tasks, restoring 1
# after. Decode (byte-swap + complex repack + pol permute) is CPU-bound, so like
# the FFT it can use the cores the memory cap leaves idle. Nested under the group
# `tmap`, but Julia's scheduler caps live tasks at `nthreads`, so it composes.
function _with_decode_threads(f, n::Int)
    old = UVData._DECODE_NTASKS[]
    UVData._DECODE_NTASKS[] = max(1, n)
    try
        return f()
    finally
        UVData._DECODE_NTASKS[] = old
    end
end

# The shared fringe model. Phase, over the global frequency band:
#   - feed-COMMON per-scan constant + delay (`SharedFeeds`): the atmosphere/clock
#     terms that vary scan-to-scan and are the same for both polarization feeds;
#   - a GLOBAL R–L offset constant + delay (`GlobalTime × FeedComponent(2)`): the
#     instrumental feed-2−feed-1 offset, stable across the whole observation
#     (EHT-HOPS / rPICARD assumption). Solved once from all scans' cross hands so
#     the bright polarized scans pin it and weak scans inherit it (feeds that
#     would otherwise split into `ncomp = 2` are tied);
#   - per-scan rate (`PerFeed`, unchanged — R–L rate is negligible);
#   - a per-AP adhoc-phase constant (`PerFeed`).
# Log-amplitude empty. `solve_station_systems!` reads the off1 columns this model
# declares — so the global-vs-per-scan split is a model choice, not solver code.
function _fringe_model()
    return StationGainModel(
        phase = (
            TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), SharedFeeds()),
            TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), GlobalFrequency()), FeedComponent(2)),
            TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), SharedFeeds()),
            TiedComponent(GainComponent(Delay(), GlobalTime(), GlobalFrequency()), FeedComponent(2)),
            TiedComponent(GainComponent(Rate(), PerScan(), GlobalFrequency()), PerFeed()),
            # Phase bandpass: per-channel, stable across the observation (HOPS-style),
            # per feed. Captures the residual nonlinear-in-frequency instrumental phase
            # that the per-scan (linear) delay cannot represent. Solved by a dedicated
            # frequency-stationization stage (`_solve_phase_bandpass!`), not by the
            # delay/rate search — `PerChannel` is its signature, used to route it.
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), GlobalFrequency()), PerFeed()),
            TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), PerFeed()),
        ),
        # Amplitude bandpass: per-channel, time-stable, per-feed log-amplitude — the
        # station-RELATIVE instrumental amplitude shape (the common-mode part is
        # degenerate with the source spectrum and left to amplitude cal). Solved by
        # the same bandpass stage from the calibrator; the SNR gate leaves low-signal
        # band edges uncorrected (gain 1) rather than dividing by ~0.
        logamp = (
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), GlobalFrequency()), PerFeed()),
        ),
    )
end

# Stage-B engine components `(plan, kind)` — the delay/rate/const terms solved by
# the fringe search + stationization. EXCLUDES the per-integration adhoc term
# (`PerIntegration`, solved per-AP) and the phase bandpass (`PerChannel`, solved
# per-channel). `kind` is derived from the term type, so the routing is structural
# (by term/segmentation), not by hardcoded indices.
function _stageB_components(model, layout)
    comps = Tuple{ComponentPlan, Symbol}[]
    for (i, tc) in enumerate(model.phase)
        tc.component.time isa PerIntegration && continue   # adhoc (per-AP)
        tc.component.term isa PerChannel && continue        # bandpass (per-channel)
        term = tc.component.term
        kind = term isa Delay ? :delay : term isa Rate ? :rate : :phase
        push!(comps, (layout.plans[i], kind))
    end
    return comps
end

# The adhoc component's plan (the per-integration phase term).
_adhoc_plan(model, layout) =
    layout.plans[findfirst(tc -> tc.component.time isa PerIntegration, model.phase)]

# The phase-bandpass component's plan (the per-channel phase term), or `nothing`
# if the model carries no bandpass component.
function _bandpass_plan(model, layout)
    i = findfirst(tc -> tc.component.term isa PerChannel, model.phase)
    return i === nothing ? nothing : layout.plans[i]
end

# The amplitude-bandpass component's plan (the per-channel log-amp term), or
# `nothing`. Log-amp plans follow the phase plans in `layout.plans` (offset
# `nphase`), so index the `PerChannel` position within `model.logamp`.
function _amp_bandpass_plan(model, layout)
    j = findfirst(tc -> tc.component.term isa PerChannel, model.logamp)
    return j === nothing ? nothing : layout.plans[layout.nphase + j]
end

# One (source, scan) group of band leaves, already materialized, with the data
# concatenated along frequency.
struct _ScanGroup
    Vg::Array{ComplexF32, 4}             # (chan, ti, bl, pol) — native precision (halves RAM)
    Wg::Array{Float32, 4}
    fg::Vector{Float64}                  # channel freqs (Hz), all bands stacked
    tg::Vector{Float64}                  # times (hours)
    bl_pairs::Vector{Tuple{Int, Int}}
    pol_products::Vector{String}
    g_ci::Vector{Int}                    # global channel indices of fg
    g_ti::Vector{Int}                    # global time indices of tg
end

# Group leaf references by (source_name, scan_name) WITHOUT materializing — the
# leaves stay lazy so the caller can materialize one scan group at a time. A
# whole-dataset eager materialization here is what fills RAM on a real (24 GB)
# file: each band leaf is ~80 MB once promoted to ComplexF64, ×144 leaves ≈ 18 GB.
# Each group entry is a `(partition_key, lazy_leaf)` pair; the key is kept so the
# fused solve+reduce path can reassemble the output tree in the original layout.
function _scan_group_leaves(uvset::UVSet)
    groups = Dict{Tuple{String, String}, Vector{Any}}()
    order = Tuple{String, String}[]
    for (k, leaf) in UVData.branches(uvset)
        info = UVData.metadata(leaf)
        key = (info.source_name, info.scan_name)
        if !haskey(groups, key)
            groups[key] = Any[]
            push!(order, key)
        end
        push!(groups[key], (k, leaf))       # (partition key, lazy leaf reference)
    end
    return [groups[k] for k in order]
end

# Materialize ONE (source, scan) group and build its concatenated `_ScanGroup`.
# The solve reads only vis + weights (flag is redundant with w <= 0); skip the flag
# layer. `materialize_group` reads the scan's whole contiguous row span in ONE
# sequential read (FITS-IDI bulk path), extracting all bands — vs per-leaf, per-row
# seeks — then falls back to per-leaf for eager/non-IDI leaves.
_materialize_group_leaves(keyed_leaves_lazy) =
    UVData.materialize_group([l for (_, l) in keyed_leaves_lazy]; layers = (:vis, :weights, :uvw))

# Build the concatenated `_ScanGroup` (frequency stacked) for the fringe search /
# bandpass stage. Fast path: decode each band's vis+weights DIRECTLY into a
# contiguous channel-block of the stacked cube (`_try_build_scan_group_direct`),
# avoiding the per-band intermediate dense arrays + the explicit concat copy (one
# full in-RAM data copy and one read+write pass eliminated) and the unused uvw
# read. Falls back to materialize-then-copy when the group is not a single
# sibling-band IDI span or its channels do not stack contiguously.
function _materialize_concat_group(keyed_leaves_lazy, geom::DataGeometry)
    direct = _try_build_scan_group_direct(keyed_leaves_lazy, geom)
    direct === nothing || return direct
    leaves = _materialize_group_leaves(keyed_leaves_lazy)
    return _build_scan_group(leaves, geom)
end

# Try to build the stacked `_ScanGroup` by decoding straight into the cube. Returns
# `nothing` (caller falls back) unless every band leaf maps to ONE full contiguous
# ascending channel block of the stacked frequency axis — the normal case (distinct,
# non-interleaved sub-bands). Metadata (baselines/pols/times/freqs) comes from the
# LAZY leaves (dims are eager), so nothing is materialized until the decode.
function _try_build_scan_group_direct(keyed_leaves_lazy, geom::DataGeometry)
    lazy = [l for (_, l) in keyed_leaves_lazy]
    l0 = first(lazy)
    bl_pairs = collect(UVData.baselines(l0).pairs)
    pols = String.(pol_products(l0))
    tg = Float64.(lookup(l0[:vis], Ti))
    nti = length(tg)
    nbl = length(bl_pairs)
    npol = length(pols)

    # (global channel index, freq, leafidx, local channel) for every channel, sorted
    # by global index — the stacked-cube channel order (same as `_build_scan_group`).
    chan_entries = Tuple{Int, Float64, Int, Int}[]
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

    # Require each leaf to be exactly one full contiguous ascending run (local
    # channels 1..no_chan land on a consecutive cube block). Anything else (a band
    # split or interleaved with another) → bail to the general copy path.
    blocks = Vector{UnitRange{Int}}(undef, length(lazy))
    fill!(blocks, 1:0)
    i = 1
    while i <= nchan
        li = chan_entries[i][3]
        chan_entries[i][4] == 1 || return nothing          # run must start at local channel 1
        j = i
        while j < nchan && chan_entries[j + 1][3] == li &&
                chan_entries[j + 1][4] == chan_entries[j][4] + 1
            j += 1
        end
        (j - i + 1) == nchan_leaf[li] || return nothing     # run must cover the whole band
        isempty(blocks[li]) || return nothing               # each leaf exactly once
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
    return _ScanGroup(Vg, Wg, fg, tg, bl_pairs, pols, g_ci, g_ti)
end

# Materialize a group as its per-band `(key, leaf)` pairs WITHOUT building the
# concatenated copy — used by pass 2 (adhoc runs directly on the leaves, then they
# are corrected/reduced/reassembled in place). Holds a single data copy.
function _materialize_leaf_group(keyed_leaves_lazy)
    leaves = _materialize_group_leaves(keyed_leaves_lazy)
    return [(k, m) for ((k, _), m) in zip(keyed_leaves_lazy, leaves)]
end

# Copy a contiguous channel-block from one band leaf's vis/weights (`V`/`W`) into
# the concatenated cubes at destination channel offset `dst0`. A FUNCTION BARRIER:
# `V`/`W` come from `parent(leaf[:vis])`, which is type-unstable at the call site,
# so doing the scalar copy inline made every element a dynamic dispatch (~1 MB/s).
# Passing them as arguments forces Julia to specialize this loop on their concrete
# runtime types; the `@simd` inner loop over the leading (stride-1) channel axis
# then runs at native speed.
function _concat_block!(
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

# Build the concatenated `_ScanGroup` from a group's already-materialized sibling
# band leaves: stack the frequency axis (sorted by channel frequency) and resolve
# global geom indices.
function _build_scan_group(leaves, geom::DataGeometry)
    let
        # Reference baselines/pols/times from the first band leaf.
        l0 = first(leaves)
        bl_pairs = collect(UVData.baselines(l0).pairs)
        pols = String.(pol_products(l0))
        tg = Float64.(lookup(l0[:vis], Ti))
        nti = length(tg)
        nbl = length(bl_pairs)
        npol = length(pols)

        # Collect (global channel index, freq, source leaf, local channel) for
        # every channel of every band, then sort by global channel index so the
        # concatenated frequency axis matches geom order.
        chan_entries = Tuple{Int, Float64, Int, Int}[]   # (g_ci, freq, leafidx, local_c)
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

        # Cache-friendly concat: walk maximal runs of consecutive channels coming
        # from the same band leaf and copy each as a contiguous channel-block
        # (channel is the leading, stride-1 axis of both V and Vg). The old
        # `Vg[row, :, :, :] .= V[lc, :, :, :]` fixed the *leading* axis, scattering
        # single elements at stride `nchan` across the whole cube one channel-row at
        # a time (nchan times) — cache-hostile and ~5× slower than this block copy.
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
            _concat_block!(Vg, Wg, V, W, i, lc0, nbc)
            i = j + 1
        end

        _, g_ti = leaf_window(geom, l0)
        return _ScanGroup(Vg, Wg, fg, tg, bl_pairs, pols, g_ci, g_ti)
    end
end

# Fringe-search every (baseline, product) of a materialized group on `Vsearch`
# (the raw cube, or the residual on rounds > 1). Returns the detection matrix and
# the group's max valid SNR.
function _search_group(grp::_ScanGroup, Vsearch, f0, t0_sec, search, ws)
    nbl = length(grp.bl_pairs)
    npol = length(grp.pol_products)
    det = Matrix{FringeDetection}(undef, nbl, npol)
    maxsnr = 0.0
    # Build the search-grid geometry ONCE for the group (freqs/times are shared by
    # every baseline×product), and pass `view`s of the cube into the core so each
    # call neither re-sorts the axes nor copies its (chan, ti) slice.
    times = grp.tg .* 3600.0
    ax = _search_axes(grp.fg, times, search)
    for p in 1:npol, bi in 1:nbl
        a, b = grp.bl_pairs[bi]
        if a == b
            det[bi, p] = FringeDetection(0.0, 0.0, 0.0, 0.0, 0.0, false)
            continue
        end
        d = _baseline_fringe_search(
            view(Vsearch, :, :, bi, p), view(grp.Wg, :, :, bi, p),
            grp.fg, times, f0, t0_sec, ax, ws, search,
        )
        det[bi, p] = d
        d.valid && (maxsnr = max(maxsnr, d.snr))
    end
    return det, maxsnr
end

# Globally-closing adhoc phase for one group: residual after the (already-solved,
# global) stage-B gains, coherently freq-averaged per AP, solved and written into
# `θ`'s per-integration adhoc slots (disjoint per group). Fused accumulation
# avoids materializing a full residual cube.
function _adhoc_group!(θ, grp::_ScanGroup, ev, adhoc_plan, adhoc, ref_ant, nant)
    nbl = length(grp.bl_pairs)
    npol = length(grp.pol_products)
    nap = length(grp.tg)
    rbar = zeros(ComplexF64, nbl, npol, nap)
    wbar = zeros(Float64, nbl, npol, nap)
    _accumulate_residual_rbar!(rbar, wbar, ev, θ, grp)
    as = solve_adhoc_phasing(
        rbar, wbar, grp.bl_pairs, grp.pol_products, nant, grp.tg;
        ref_ant = ref_ant, opts = adhoc,
    )
    for (ap, gti) in enumerate(grp.g_ti)
        tseg = adhoc_plan.tseg_id[gti]
        for ant in 1:nant, feed in 1:2
            v = as.phase[ant, feed, ap]
            isfinite(v) || continue
            off = adhoc_plan.off1[ant, feed, tseg, 1]
            off == 0 && continue
            θ[off] = v
        end
    end
    return θ
end

# Adhoc for one group from its per-band LEAVES (no concatenated `_ScanGroup` — pass
# 2's memory-lean path). The per-AP residual is summed over each band leaf's
# channels (a function barrier per leaf, gains evaluated on the leaf window); the
# sum over all bands' channels is identical to accumulating over the concatenated
# cube, so the result matches `_adhoc_group!` exactly. `keyed` is the group's
# `(key, leaf)` pairs.
function _adhoc_group_leaves!(θ, keyed, geom::DataGeometry, ev, adhoc_plan, adhoc, ref_ant, nant)
    leaves = [m for (_, m) in keyed]
    l0 = first(leaves)
    bl_pairs = collect(UVData.baselines(l0).pairs)
    pols = String.(pol_products(l0))
    tg = Float64.(lookup(l0[:vis], Ti))
    _, g_ti = leaf_window(geom, l0)
    nbl = length(bl_pairs); npol = length(pols); nap = length(tg)
    rbar = zeros(ComplexF64, nbl, npol, nap)
    wbar = zeros(Float64, nbl, npol, nap)
    for leaf in leaves
        ci, ti = leaf_window(geom, leaf)
        g = evaluate_gains(ev, θ, ci, ti)               # (nchan_leaf, nti, nant, 2)
        _accumulate_leaf_rbar!(rbar, wbar, parent(leaf[:vis]), parent(leaf[:weights]), g, bl_pairs, pols)
    end
    as = solve_adhoc_phasing(rbar, wbar, bl_pairs, pols, nant, tg; ref_ant = ref_ant, opts = adhoc)
    for (ap, gti) in enumerate(g_ti)
        tseg = adhoc_plan.tseg_id[gti]
        for ant in 1:nant, feed in 1:2
            v = as.phase[ant, feed, ap]
            isfinite(v) || continue
            off = adhoc_plan.off1[ant, feed, tseg, 1]
            off == 0 && continue
            θ[off] = v
        end
    end
    return θ
end

# Accumulate one band leaf's per-AP residual `Σ_chan w·(V/gain)` into `rbar`/`wbar`.
# Function barrier: `V`/`W` from `parent(leaf[...])` are type-unstable at the call
# site, so the per-cell loop must be a specialized function (else dynamic dispatch
# per element). Mirrors `_accumulate_residual_rbar!` but for one leaf's channels.
function _accumulate_leaf_rbar!(rbar, wbar, V, W, g, bl_pairs, pols)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(pols[p])
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            for tt in 1:nti, c in 1:nchan
                w = W[c, tt, bi, p]
                (w > 0 && isfinite(w)) || continue
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, tt, bi, p] / den
                isfinite(v) || continue
                rbar[bi, p, tt] += w * v
                wbar[bi, p, tt] += w
            end
        end
    end
    return rbar, wbar
end

# Estimate the PEAK resident bytes for processing one scan group through the bulk
# reader. The bulk path holds, at once: the transient row-span buffer (~one copy
# of the raw bytes), the per-band materialized leaves, AND the concatenated
# `_ScanGroup` copy — plus a windowed gain cube. That is ~3–4× the raw vis size;
# we charge 4× (vis ComplexF32 = 8 B + weights Float32 = 4 B ⇒ 12 B/cell, ×4) as a
# safe peak. `group_leaves[i]` is a vector of `(key, lazy_leaf)`; summing
# `prod(size(leaf[:vis]))` over its bands gives the group's cell count without
# materializing anything.
# Estimate the PEAK resident bytes for one scan group (vis ComplexF32 8 B + weights
# Float32 4 B = 12 B/cell). During decode the search pass holds the stacked cube
# (one copy) PLUS the transient bulk-read span (≈ one copy — on-disk FLUX is the
# same 8 B/cell, measured span/cube ≈ 0.8–1.1 here), so peak ≈ 2× cube; pass 2
# similarly holds the per-band leaves plus the corrected copy `apply_calibration`
# allocates. We charge 2.5× for GC/heap headroom over that ~2.25× peak. The
# decode-into-cube change removed the OLD second copy (leaves AND cube held at once),
# but the now-larger in-RAM span (the 3 GiB cap lets big scans use the fast read)
# brings the peak back to ~2× — so the charge stays 2.5×.
function _group_peak_bytes(group)
    cells = 0
    for (_, leaf) in group
        cells += prod(size(leaf[:vis]))
    end
    return round(Int, 2.5 * 12 * cells)
end

# The DETERMINISTIC memory budget for the solve (bytes): an explicit `mem_budget`
# if given, else `mem_fraction` of TOTAL physical RAM. Total RAM is a machine
# constant, so the budget — and therefore `ntasks`, and therefore the whole solve's
# wall time — is REPRODUCIBLE across runs on a box. The earlier version budgeted
# from `/proc/meminfo` `MemAvailable`, which swings as other processes (IDE/language
# servers) come and go and so flipped the cap between e.g. 2 and 3 tasks on
# identical inputs — making the solve impossible to benchmark. The trade-off of
# using total: it can over-commit a heavily LOADED box, so `mem_fraction` must leave
# headroom for everything else (lower it on a shared box, or pass an explicit
# `mem_budget` for machine-independent control). The conservative per-group peak
# charge (`_group_peak_bytes`, 2.5× the real ~2× footprint) provides further slack.
function _memory_budget(mem_fraction, mem_budget)
    mem_budget === nothing || return Float64(mem_budget)
    return mem_fraction * Float64(Sys.total_memory())
end

# Bound `ntasks` so peak RAM (`ntasks × per-group peak`) stays under the
# deterministic memory budget. Defaulting `ntasks` to `Threads.nthreads()` OOMs on a
# real (24 GB) file: each concurrent group peaks at several GB. We size the cap from
# the LARGEST group and the budget. Returns ≥ 1.
function _bounded_ntasks(group_leaves, ntasks, ngroups; mem_fraction = 0.6, mem_budget = nothing)
    requested = max(1, min(Int(ntasks), ngroups))
    peak = maximum(_group_peak_bytes, group_leaves; init = 0)
    peak <= 0 && return requested
    budget = _memory_budget(mem_fraction, mem_budget)
    cap = max(1, Int(floor(budget / peak)))
    used = min(requested, cap)
    if used < requested
        @info "Fringe solve: capping ntasks for memory" requested cap used peak_GB = round(peak / 2^30; digits = 2) total_GB = round(Sys.total_memory() / 2^30; digits = 2) budget_GB = round(budget / 2^30; digits = 2)
    end
    return used
end

# Pass 1: search every group (threaded; on the residual for rounds > 1), then a
# SINGLE global stage-B solve over all groups' detections — `solve_station_systems!`
# couples scans through the global R–L offset column, so it cannot be per-group.
# Mutates `θ` (stage-B slots) and `scan_snr`; returns (chi, ncomp).
function _search_and_stationize!(
        θ, scan_snr, group_leaves, geom, ev, stageB, f0, t0_sec,
        search, rounds, ref_ant, ntasks_use, ws_tlv,
    )
    ngroups = length(group_leaves)
    chi = NaN
    ncomp = 0
    for round in 1:max(rounds, 1)
        dets = tmap(1:ngroups; ntasks = ntasks_use) do gi
            grp = _materialize_concat_group(group_leaves[gi], geom)
            Vsearch = round > 1 ? _residual_vis(ev, θ, grp) : grp.Vg
            det, snr = _search_group(grp, Vsearch, f0, t0_sec, search, ws_tlv[])
            scan_snr[gi] = snr
            feeds = [correlation_feed_pair(p) for p in grp.pol_products]
            StationScanDetections(det, grp.bl_pairs, feeds, first(grp.g_ti))
        end
        chi, ncomp = solve_station_systems!(θ, dets, stageB; ref_ant = ref_ant)
    end
    return chi, ncomp
end

"""
    solve_fringes(uvset; search, adhoc, rounds, ref_ant) -> CalibrationSolution

Fringe-fit `uvset`. Builds the shared fringe model (feed-common per-scan phase +
delay, a global stable R–L offset, per-scan rate, and a per-integration adhoc
term) and solves it in two passes over the (source, scan) groups: (1) a multi-band
fringe search on every group followed by ONE global stationization that ties the
feeds with a track-wide R–L offset; (2) the globally-closing adhoc phase per
group on the stage-B residual. Returns the packed `CalibrationSolution`.

Because the R–L offset couples scans, the stage-B solve is global — so the data
is read once for the search pass and once for the adhoc pass (`rounds` re-runs
the search pass on the residual).
"""
function solve_fringes(
        uvset::UVSet;
        search::FringeSearch = FringeSearch(),
        adhoc::AdhocPhasing = AdhocPhasing(),
        rounds::Int = 1,
        ref_ant::Integer = 1,
        ntasks::Integer = Threads.nthreads(),
        mem_fraction::Real = 0.6,
        mem_budget = nothing,
        phase_bandpass::Bool = true,
        amp_bandpass::Bool = true,
        bandpass_source = nothing,
    )
    model = _fringe_model()
    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    nant = length(UVData.metadata(first_leaf).antennas)
    layout = plan_parameters(model, nant, geom)
    ev = GainEvaluator(model, layout)
    stageB = _stageB_components(model, layout)
    adhoc_plan = _adhoc_plan(model, layout)
    bp_plan = _bandpass_plan(model, layout)
    amp_bp_plan = _amp_bandpass_plan(model, layout)
    f0 = geom.f0
    t0_sec = geom.t0 * 3600.0

    group_leaves = _scan_group_leaves(uvset)
    group_sources = _group_sources(group_leaves)
    ngroups = length(group_leaves)
    scan_snr = zeros(ngroups)
    θ = zeros(layout.nθ)
    ntasks_use = _bounded_ntasks(group_leaves, ntasks, ngroups; mem_fraction = mem_fraction, mem_budget = mem_budget)
    nfft = max(1, Threads.nthreads() ÷ ntasks_use)
    ws_tlv = TaskLocalValue{FringeWorkspace}(FringeWorkspace)

    chi, ncomp = _with_decode_threads(nfft) do
        _with_fft_threads(nfft) do
            _with_single_blas_thread() do
                ch, nc = _search_and_stationize!(
                    θ, scan_snr, group_leaves, geom, ev, stageB, f0, t0_sec,
                    search, rounds, ref_ant, ntasks_use, ws_tlv,
                )
                # Phase + amplitude bandpass (between Stage-B and adhoc; orthogonal
                # frequency structure). Solved once from the brightest calibrator,
                # time-stable, from one shared per-channel residual accumulation.
                if (phase_bandpass && bp_plan !== nothing) || (amp_bandpass && amp_bp_plan !== nothing)
                    cal = _bandpass_calibrator(bandpass_source, group_sources, scan_snr)
                    _solve_bandpass_stage!(
                        θ, group_leaves, group_sources, cal, geom, ev,
                        phase_bandpass ? bp_plan : nothing, nant;
                        amp_plan = amp_bandpass ? amp_bp_plan : nothing, ref_ant = ref_ant,
                    )
                end
                # Pass 2: adhoc per group on the residual after the global stage-B (each
                # group writes its own disjoint per-integration slots of the shared θ).
                tmap(1:ngroups; ntasks = ntasks_use) do gi
                    keyed = _materialize_leaf_group(group_leaves[gi])
                    _adhoc_group_leaves!(θ, keyed, geom, ev, adhoc_plan, adhoc, ref_ant, nant)
                    nothing
                end
                (ch, nc)
            end
        end
    end

    info = (;
        nant = nant,
        nscan = ngroups,
        scan_max_snr = scan_snr,
        scan_chi = fill(chi, ngroups),
        scan_ncomp = fill(ncomp, ngroups),
        ant_names = String.(collect(UVData.metadata(first_leaf).antennas.name)),
    )
    return CalibrationSolution(model, layout, geom, θ, info)
end

"""
    solve_and_reduce_fringes(uvset; postprocess, search, adhoc, rounds, ref_ant, ntasks)
        -> (sol::CalibrationSolution, output::UVSet)

Fringe fit **and** correction/reduction. The stage-B solve ties the feeds with a
track-wide R–L offset (so it is global, not per-scan); the data is therefore read
in two threaded passes: (1) search every group → one global stationization; (2)
per group, adhoc-phase → gain-correct → run through `postprocess` (a `UVSet ->
UVSet` map, default `identity`) → free. `postprocess` runs while the corrected
group is resident (no extra I/O), so the natural use is post-fringe reduction,
e.g.

    postprocess = uv -> combine_spw(time_bin_average(frequency_average(uv; nout = 1), 2.0))

`output` is a full `UVSet` mirroring the input tree with every leaf corrected and
reduced; `sol` is the same solution `solve_fringes` would return.

`ntasks` requests how many groups are processed concurrently, but it is bounded
down so peak RAM (≈ `ntasks` × per-group size) stays under a DETERMINISTIC budget:
`mem_budget` (absolute bytes) if given, else `mem_fraction` of TOTAL physical RAM.
Using total — a machine constant — makes `ntasks` reproducible across runs (the
old MemAvailable basis swung the cap with whatever else was running). The bulk
reader holds a whole scan's row span plus its vis cube at once (several GB on a
real file), so a raw thread-count default would OOM. On a shared/loaded box leave
headroom: lower `mem_fraction` (or pass an explicit `mem_budget`); raise it on a
dedicated box. A capping decision is logged via `@info`.
"""
function solve_and_reduce_fringes(
        uvset::UVSet;
        postprocess = identity,
        search::FringeSearch = FringeSearch(),
        adhoc::AdhocPhasing = AdhocPhasing(),
        rounds::Int = 1,
        ref_ant::Integer = 1,
        ntasks::Integer = Threads.nthreads(),
        mem_fraction::Real = 0.6,
        mem_budget = nothing,
        phase_bandpass::Bool = true,
        amp_bandpass::Bool = true,
        bandpass_source = nothing,
    )
    model = _fringe_model()
    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    nant = length(UVData.metadata(first_leaf).antennas)
    layout = plan_parameters(model, nant, geom)
    ev = GainEvaluator(model, layout)
    stageB = _stageB_components(model, layout)
    adhoc_plan = _adhoc_plan(model, layout)
    bp_plan = _bandpass_plan(model, layout)
    amp_bp_plan = _amp_bandpass_plan(model, layout)
    f0 = geom.f0
    t0_sec = geom.t0 * 3600.0

    group_leaves = _scan_group_leaves(uvset)
    group_sources = _group_sources(group_leaves)
    ngroups = length(group_leaves)
    scan_snr = zeros(ngroups)
    θ = zeros(layout.nθ)
    ntasks_use = _bounded_ntasks(group_leaves, ntasks, ngroups; mem_fraction = mem_fraction, mem_budget = mem_budget)
    nfft = max(1, Threads.nthreads() ÷ ntasks_use)
    ws_tlv = TaskLocalValue{FringeWorkspace}(FringeWorkspace)

    out_pairs, chi, ncomp = _with_decode_threads(nfft) do
        _with_fft_threads(nfft) do
            _with_single_blas_thread() do
                ch, nc = _search_and_stationize!(
                    θ, scan_snr, group_leaves, geom, ev, stageB, f0, t0_sec,
                    search, rounds, ref_ant, ntasks_use, ws_tlv,
                )
                # Phase + amplitude bandpass from the brightest calibrator (time-
                # stable), applied to every group's correction below via the full θ.
                if (phase_bandpass && bp_plan !== nothing) || (amp_bandpass && amp_bp_plan !== nothing)
                    cal = _bandpass_calibrator(bandpass_source, group_sources, scan_snr)
                    _solve_bandpass_stage!(
                        θ, group_leaves, group_sources, cal, geom, ev,
                        phase_bandpass ? bp_plan : nothing, nant;
                        amp_plan = amp_bandpass ? amp_bp_plan : nothing, ref_ant = ref_ant,
                    )
                end
                # Pass 2: per group, adhoc → correct → reduce. θ is fully populated for
                # this group (global stage-B + this group's just-written adhoc slots), so
                # the group-local solution corrects identically to the global one.
                results = tmap(1:ngroups; ntasks = ntasks_use) do gi
                    keyed = _materialize_leaf_group(group_leaves[gi])
                    _adhoc_group_leaves!(θ, keyed, geom, ev, adhoc_plan, adhoc, ref_ant, nant)
                    sub_branches = DimensionalData.TreeDict()
                    for (k, leaf) in keyed
                        sub_branches[k] = leaf
                    end
                    sub = DimensionalData.rebuild(uvset; branches = sub_branches)
                    sol_local = CalibrationSolution(model, layout, geom, θ, (;))
                    reduced = postprocess(UVData.apply_calibration(sub, sol_local))
                    collect(pairs(UVData.branches(reduced)))
                end
                (results, ch, nc)
            end
        end
    end

    out_branches = DimensionalData.TreeDict()
    for r in out_pairs
        for (k, leaf) in r
            out_branches[k] = leaf
        end
    end
    output = DimensionalData.rebuild(uvset; branches = out_branches)

    info = (;
        nant = nant,
        nscan = ngroups,
        scan_max_snr = scan_snr,
        scan_chi = fill(chi, ngroups),
        scan_ncomp = fill(ncomp, ngroups),
        ant_names = String.(collect(UVData.metadata(first_leaf).antennas.name)),
    )
    sol = CalibrationSolution(model, layout, geom, θ, info)
    return sol, output
end

# Accumulate the coherent per-(baseline, product, AP) residual sum
# `rbar = Σ_chan w·(V/gain)`, `wbar = Σ_chan w` — evaluating gains once and
# streaming over channels so no full residual cube is allocated (the adhoc stage
# only needs the frequency-collapsed residual). Matches `_residual_vis` + the old
# explicit accumulation exactly.
function _accumulate_residual_rbar!(rbar, wbar, ev::GainEvaluator, θ::AbstractVector, grp::_ScanGroup)
    g = evaluate_gains(ev, θ, grp.g_ci, grp.g_ti)    # (nchan, nti, nant, 2)
    nchan, nti, nbl, npol = size(grp.Vg)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(grp.pol_products[p])
        for bi in 1:nbl
            a, b = grp.bl_pairs[bi]
            for tt in 1:nti, c in 1:nchan
                w = grp.Wg[c, tt, bi, p]
                (w > 0 && isfinite(w)) || continue
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                denom = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(denom)) || continue
                v = grp.Vg[c, tt, bi, p] / denom
                isfinite(v) || continue
                rbar[bi, p, tt] += w * v
                wbar[bi, p, tt] += w
            end
        end
    end
    return rbar, wbar
end

# ── Phase bandpass (HOPS-style passband): per-channel station phase, time-stable ──
#
# Accumulate one group's contribution to the per-(global-baseline, product, GLOBAL
# channel) coherent residual `rbar_bp` (and weight `wbar_bp`), for the phase-
# bandpass solve. The residual is V / (stage-B gains); BEFORE summing over time we
# counter-rotate each AP by its OWN band-averaged residual phase, removing the
# per-AP time phase (residual rate/drift, and what the adhoc would later remove)
# so the time-average is coherent and isolates the per-channel SHAPE. `blidx` maps
# `(a, b) -> row` in the global baseline table. Mirrors the adhoc accumulation but
# collapses over time per channel instead of over frequency per AP.
function _accumulate_bandpass_rbar!(rbar_bp, wbar_bp, blidx, ev::GainEvaluator, θ::AbstractVector, grp::_ScanGroup)
    g = evaluate_gains(ev, θ, grp.g_ci, grp.g_ti)    # (nchan, nti, nant, 2)
    nchan, nti, nbl, npol = size(grp.Vg)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(grp.pol_products[p])
        for bi in 1:nbl
            a, b = grp.bl_pairs[bi]
            a == b && continue
            idx = get(blidx, (a, b), 0)
            idx == 0 && continue
            for tt in 1:nti
                # Band-averaged residual phase for this AP (the per-AP time phase).
                acc = zero(ComplexF64)
                for c in 1:nchan
                    w = grp.Wg[c, tt, bi, p]
                    (w > 0 && isfinite(w)) || continue
                    ga = g[c, tt, a, fa]; gb = g[c, tt, b, fb]
                    den = ga * conj(gb)
                    (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                    v = grp.Vg[c, tt, bi, p] / den
                    isfinite(v) && (acc += w * v)
                end
                abs(acc) > 0 || continue
                rot = conj(acc) / abs(acc)            # cis(-angle(acc)): de-rotate this AP
                for c in 1:nchan
                    w = grp.Wg[c, tt, bi, p]
                    (w > 0 && isfinite(w)) || continue
                    ga = g[c, tt, a, fa]; gb = g[c, tt, b, fb]
                    den = ga * conj(gb)
                    (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                    v = grp.Vg[c, tt, bi, p] / den
                    isfinite(v) || continue
                    gc = grp.g_ci[c]
                    rbar_bp[idx, p, gc] += w * v * rot
                    wbar_bp[idx, p, gc] += w
                end
            end
        end
    end
    return rbar_bp, wbar_bp
end

# Solve the per-(station, feed) phase bandpass from the accumulated per-channel
# residual and write it into `θ`'s `PerChannel` slots. For each global channel,
# the globally-closing per-feed phase is solved exactly like the adhoc (reusing
# `_solve_observable` over the (station, feed) graph) with the same scale-invariant
# SNR gate (`_track_noise2`, robust to uncalibrated weights). Each (station, feed)
# track is then referenced to its CIRCULAR-mean phase over channels (zero net phase
# ⇒ does not alias the Stage-B constant phase). No cross-channel unwrap/smoothing —
# the per-channel phase is applied as `cis(φ)`, for which wrapping is irrelevant,
# and unwrapping across the sub-band gaps would be unsafe.
function _solve_phase_bandpass!(
        θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan;
        ref_ant::Integer = 1, snr_floor::Real = 1.0,
    )
    nbl, npol, nchan = size(rbar_bp)
    feeds = [correlation_feed_pair(p) for p in pol_products]
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in 1:nbl, p in 1:npol]
    phase = fill(NaN, nant, 2, nchan)
    for gc in 1:nchan
        rows = _ObsRow[]
        for bi in 1:nbl, p in 1:npol
            a, b = bl_pairs[bi]
            a == b && continue
            r = rbar_bp[bi, p, gc]; w = wbar_bp[bi, p, gc]
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            n2 = noise2[bi, p]
            snr2 = isfinite(n2) && n2 > 0 ? abs2(r / w) / n2 : abs2(r) / w
            snr2 >= snr_floor^2 || continue
            fa, fb = feeds[p]
            push!(rows, _ObsRow(a, b, fa, fb, angle(r), snr2, _chi_sign(fa, fb)))
        end
        ph, _, _, _ = _solve_observable(rows, nant, ref_ant; use_chi = true, rewrap = 0)
        phase[:, :, gc] .= ph
    end

    # Circular-mean reference per (station, feed) → zero net applied phase (gauge).
    for a in 1:nant, f in 1:2
        acc = zero(ComplexF64)
        @inbounds for gc in 1:nchan
            v = phase[a, f, gc]
            isfinite(v) && (acc += cis(v))
        end
        abs(acc) > 0 || continue
        m = angle(acc)
        @inbounds for gc in 1:nchan
            v = phase[a, f, gc]
            isfinite(v) || continue
            off = plan.off1[a, f, 1, 1]
            off == 0 && continue
            θ[off + plan.clocal[gc] - 1] = rem2pi(v - m, RoundNearest)
        end
    end
    return θ
end

# Solve the per-(station, feed) AMPLITUDE bandpass (log-amp) from the SAME
# accumulated residual and write it into `θ`'s log-amp `PerChannel` slots. For each
# channel the coherent baseline amplitude obeys `log|V̄_ab| = la + lb` (a SUM
# closure, +1/+1 incidence — unlike the phase difference), solved per channel by a
# ridge-regularized WLS over the (station, feed) nodes (ridge stabilizes any
# rank-deficient/bipartite component; the gauge below removes the resulting offset).
# Same scale-invariant SNR gate as the phase path, so low-signal band edges are
# left uncorrected (log-amp 0 ⇒ gain 1) rather than dividing by ~0. Gauge: zero
# band-mean log-amp per (station, feed) — the absolute/common-mode amplitude is
# degenerate with the source spectrum and intentionally NOT recovered here.
function _solve_amp_bandpass!(
        θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan;
        snr_floor::Real = 1.0, ridge::Real = 1.0e-6, max_logamp::Real = log(2.0),
    )
    nbl, npol, nchan = size(rbar_bp)
    feeds = [correlation_feed_pair(p) for p in pol_products]
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in 1:nbl, p in 1:npol]
    nnodes = 2 * nant
    pen = fill(float(ridge), nnodes)
    la = fill(NaN, nant, 2, nchan)
    for gc in 1:nchan
        n1 = Int[]; n2 = Int[]; vals = Float64[]; wts = Float64[]
        touched = falses(nnodes)
        for bi in 1:nbl, p in 1:npol
            a, b = bl_pairs[bi]
            a == b && continue
            r = rbar_bp[bi, p, gc]; w = wbar_bp[bi, p, gc]
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            nz = noise2[bi, p]
            snr2 = isfinite(nz) && nz > 0 ? abs2(r / w) / nz : abs2(r) / w
            snr2 >= snr_floor^2 || continue
            amp = abs(r / w)
            amp > 0 || continue
            fa, fb = feeds[p]
            push!(n1, _node(a, fa, nant)); push!(n2, _node(b, fb, nant))
            push!(vals, log(amp)); push!(wts, snr2)
            touched[_node(a, fa, nant)] = true; touched[_node(b, fb, nant)] = true
        end
        isempty(vals) && continue
        A = zeros(length(vals), nnodes)
        @inbounds for ri in eachindex(vals)
            A[ri, n1[ri]] += 1.0
            A[ri, n2[ri]] += 1.0
        end
        sol = weighted_regularized_least_squares(A, vals, wts, pen)
        for node in 1:nnodes
            touched[node] || continue
            ant = (node - 1) % nant + 1
            feed = (node - 1) ÷ nant + 1
            la[ant, feed, gc] = sol[node]
        end
    end

    # Remove the per-channel COMMON MODE (mean over stations, per feed). The +1/+1
    # closure with no source term attributes the common instrumental bandpass (the
    # filterbank roll-off) AND the source spectrum to the stations, so each station's
    # `la` carries the whole band shape (|g| 0.1→1.6, →0 at edges). Subtracting the
    # per-channel station mean leaves only the station-RELATIVE bandpass (|g|~1, no
    # roll-off, no edge blow-up) — the common mode is degenerate with the source and
    # intentionally NOT corrected. This is the amplitude analog of the phase solve's
    # ±1 difference closure (which cancels the common mode automatically).
    @inbounds for f in 1:2, gc in 1:nchan
        acc = 0.0; n = 0
        for a in 1:nant
            v = la[a, f, gc]
            isfinite(v) && (acc += v; n += 1)
        end
        n == 0 && continue
        m = acc / n
        for a in 1:nant
            isfinite(la[a, f, gc]) && (la[a, f, gc] -= m)
        end
    end

    # Zero band-mean log-amp gauge per (station, feed), then write the log-amp slots.
    for a in 1:nant, f in 1:2
        acc = 0.0; n = 0
        @inbounds for gc in 1:nchan
            v = la[a, f, gc]
            isfinite(v) && (acc += v; n += 1)
        end
        n == 0 && continue
        m = acc / n
        off = plan.off1[a, f, 1, 1]
        off == 0 && continue
        @inbounds for gc in 1:nchan
            v = la[a, f, gc]
            isfinite(v) || continue
            val = v - m
            # Leave implausibly large corrections UNAPPLIED (|g| = 1): these are the
            # low-SNR band-edge channels where the relative solve is unstable, and
            # where a |g| > 1 would even up-weight noise (apply scales weight ×|g|²).
            θ[off + plan.clocal[gc] - 1] = abs(val) > max_logamp ? 0.0 : val
        end
    end
    return θ
end

# Bandpass stage: solve the phase bandpass from one (bright calibrator) source and
# write it into `θ`. Accumulates the per-channel residual over that source's scans
# (after the global Stage-B), then one closing per-channel solve. Sequential over
# the calibrator's groups (one source, a handful of scans) — only one group is
# resident at a time. Time-stable, so it applies to ALL scans via `GlobalTime`.
function _solve_bandpass_stage!(
        θ, group_leaves, group_sources, cal_source, geom, ev, plan, nant;
        amp_plan = nothing, ref_ant::Integer = 1, snr_floor::Real = 1.0,
    )
    nchan = length(geom.channel_freqs)
    bl_pairs = [(a, b) for a in 1:nant for b in (a + 1):nant]
    blidx = Dict(bl_pairs[i] => i for i in eachindex(bl_pairs))
    rbar_bp = nothing; wbar_bp = nothing; pols = String[]
    for gi in eachindex(group_leaves)
        group_sources[gi] == cal_source || continue
        grp = _materialize_concat_group(group_leaves[gi], geom)
        if rbar_bp === nothing
            pols = grp.pol_products
            rbar_bp = zeros(ComplexF64, length(bl_pairs), length(pols), nchan)
            wbar_bp = zeros(Float64, length(bl_pairs), length(pols), nchan)
        end
        _accumulate_bandpass_rbar!(rbar_bp, wbar_bp, blidx, ev, θ, grp)
    end
    rbar_bp === nothing && return θ            # calibrator absent → leave bandpass at 0
    plan === nothing ||
        _solve_phase_bandpass!(θ, rbar_bp, wbar_bp, bl_pairs, pols, nant, plan; ref_ant = ref_ant, snr_floor = snr_floor)
    amp_plan === nothing ||
        _solve_amp_bandpass!(θ, rbar_bp, wbar_bp, bl_pairs, pols, nant, amp_plan; snr_floor = snr_floor)
    return θ
end

# Source name of each (source, scan) group, without materializing.
_group_sources(group_leaves) =
    [UVData.metadata(last(first(g))).source_name for g in group_leaves]

# The calibrator for the bandpass: the explicit `bandpass_source`, else the source
# carrying the highest-SNR scan (brightest), else the first group's source.
function _bandpass_calibrator(bandpass_source, group_sources, scan_snr)
    bandpass_source !== nothing && return String(bandpass_source)
    isempty(group_sources) && return ""
    gi = all(!isfinite, scan_snr) ? 1 : argmax(i -> (isfinite(scan_snr[i]) ? scan_snr[i] : -Inf), eachindex(scan_snr))
    return group_sources[gi]
end

# Residual visibilities for one scan group: Vg divided by the current θ gains
# evaluated at the group's (global chan, global ti) window. Used for rounds > 1
# (the search needs a full residual cube); the adhoc stage uses the fused
# `_accumulate_residual_rbar!` above instead.
function _residual_vis(ev::GainEvaluator, θ::AbstractVector, grp::_ScanGroup)
    g = evaluate_gains(ev, θ, grp.g_ci, grp.g_ti)    # (nchan, nti, nant, 2)
    nchan, nti, nbl, npol = size(grp.Vg)
    out = similar(grp.Vg)
    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(grp.pol_products[p])
        for bi in 1:nbl
            a, b = grp.bl_pairs[bi]
            for tt in 1:nti, c in 1:nchan
                ga = g[c, tt, a, fa]
                gb = g[c, tt, b, fb]
                denom = ga * conj(gb)
                if abs(ga) < 1.0e-12 || abs(gb) < 1.0e-12 || !isfinite(denom)
                    out[c, tt, bi, p] = ComplexF64(NaN, NaN)
                else
                    out[c, tt, bi, p] = grp.Vg[c, tt, bi, p] / denom
                end
            end
        end
    end
    return out
end
