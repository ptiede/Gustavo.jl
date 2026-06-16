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
            TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), PerFeed()),
        ),
    )
end

# Stage-B engine components `(plan, kind)` — every phase component except the
# per-integration adhoc term, with `kind` derived from the term type (so a new
# term/segmentation is picked up automatically; no hardcoded indices).
function _stageB_components(model, layout)
    comps = Tuple{ComponentPlan, Symbol}[]
    for (i, tc) in enumerate(model.phase)
        tc.component.time isa PerIntegration && continue
        term = tc.component.term
        kind = term isa Delay ? :delay : term isa Rate ? :rate : :phase
        push!(comps, (layout.plans[i], kind))
    end
    return comps
end

# The adhoc component's plan (the per-integration phase term).
_adhoc_plan(model, layout) =
    layout.plans[findfirst(tc -> tc.component.time isa PerIntegration, model.phase)]

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
# Returns `(grp, keyed)` where `keyed` is the vector of `(partition_key,
# materialized_leaf)` pairs — kept so the fused path can correct+reduce the SAME
# materialized leaves (no second disk read) and reassemble the output tree.
# Called per group inside the solve loop so only one group's worth of data
# (~hundreds of MB) is resident at a time.
function _materialize_scan_group(keyed_leaves_lazy, geom::DataGeometry)
    # The solve reads only vis + weights (flag is redundant with w <= 0); skip the
    # flag layer. `materialize_group` reads the scan's whole contiguous row span in
    # ONE sequential read (FITS-IDI bulk path), extracting all bands — vs per-leaf,
    # per-row seeks — then falls back to per-leaf for eager/non-IDI leaves.
    leaves = UVData.materialize_group([l for (_, l) in keyed_leaves_lazy]; layers = (:vis, :weights, :uvw))
    keyed = [(k, m) for ((k, _), m) in zip(keyed_leaves_lazy, leaves)]
    return _build_scan_group(leaves, geom), keyed
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
            gc, f, li, lc = e
            fg[row] = f
            g_ci[row] = gc
            V = parent(leaves[li][:vis])
            W = parent(leaves[li][:weights])
            @views Vg[row, :, :, :] .= V[lc, :, :, :]
            @views Wg[row, :, :, :] .= W[lc, :, :, :]
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
    for p in 1:npol, bi in 1:nbl
        a, b = grp.bl_pairs[bi]
        if a == b
            det[bi, p] = FringeDetection(0.0, 0.0, 0.0, 0.0, 0.0, false)
            continue
        end
        d = baseline_fringe_search(
            Vsearch[:, :, bi, p], grp.Wg[:, :, bi, p],
            grp.fg, grp.tg .* 3600.0, f0, t0_sec; opts = search, workspace = ws,
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
            grp, _ = _materialize_scan_group(group_leaves[gi], geom)
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
    )
    model = _fringe_model()
    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    nant = length(UVData.metadata(first_leaf).antennas)
    layout = plan_parameters(model, nant, geom)
    ev = GainEvaluator(model, layout)
    stageB = _stageB_components(model, layout)
    adhoc_plan = _adhoc_plan(model, layout)
    f0 = geom.f0
    t0_sec = geom.t0 * 3600.0

    group_leaves = _scan_group_leaves(uvset)
    ngroups = length(group_leaves)
    scan_snr = zeros(ngroups)
    θ = zeros(layout.nθ)
    ntasks_use = max(1, min(Int(ntasks), ngroups))
    ws_tlv = TaskLocalValue{FringeWorkspace}(FringeWorkspace)

    chi, ncomp = _with_single_blas_thread() do
        ch, nc = _search_and_stationize!(
            θ, scan_snr, group_leaves, geom, ev, stageB, f0, t0_sec,
            search, rounds, ref_ant, ntasks_use, ws_tlv,
        )
        # Pass 2: adhoc per group on the residual after the global stage-B (each
        # group writes its own disjoint per-integration slots of the shared θ).
        tmap(1:ngroups; ntasks = ntasks_use) do gi
            grp, _ = _materialize_scan_group(group_leaves[gi], geom)
            _adhoc_group!(θ, grp, ev, adhoc_plan, adhoc, ref_ant, nant)
            nothing
        end
        (ch, nc)
    end

    info = (;
        nant = nant,
        nscan = ngroups,
        scan_max_snr = scan_snr,
        scan_chi = fill(chi, ngroups),
        scan_ncomp = fill(ncomp, ngroups),
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
reduced; `sol` is the same solution `solve_fringes` would return. `ntasks` caps
how many groups are resident at once (peak RAM ≈ `ntasks` × per-group size).
"""
function solve_and_reduce_fringes(
        uvset::UVSet;
        postprocess = identity,
        search::FringeSearch = FringeSearch(),
        adhoc::AdhocPhasing = AdhocPhasing(),
        rounds::Int = 1,
        ref_ant::Integer = 1,
        ntasks::Integer = Threads.nthreads(),
    )
    model = _fringe_model()
    geom = build_geometry(uvset)
    first_leaf = first(values(UVData.branches(uvset)))
    nant = length(UVData.metadata(first_leaf).antennas)
    layout = plan_parameters(model, nant, geom)
    ev = GainEvaluator(model, layout)
    stageB = _stageB_components(model, layout)
    adhoc_plan = _adhoc_plan(model, layout)
    f0 = geom.f0
    t0_sec = geom.t0 * 3600.0

    group_leaves = _scan_group_leaves(uvset)
    ngroups = length(group_leaves)
    scan_snr = zeros(ngroups)
    θ = zeros(layout.nθ)
    ntasks_use = max(1, min(Int(ntasks), ngroups))
    ws_tlv = TaskLocalValue{FringeWorkspace}(FringeWorkspace)

    out_pairs, chi, ncomp = _with_single_blas_thread() do
        ch, nc = _search_and_stationize!(
            θ, scan_snr, group_leaves, geom, ev, stageB, f0, t0_sec,
            search, rounds, ref_ant, ntasks_use, ws_tlv,
        )
        # Pass 2: per group, adhoc → correct → reduce. θ is fully populated for
        # this group (global stage-B + this group's just-written adhoc slots), so
        # the group-local solution corrects identically to the global one.
        results = tmap(1:ngroups; ntasks = ntasks_use) do gi
            grp, keyed = _materialize_scan_group(group_leaves[gi], geom)
            _adhoc_group!(θ, grp, ev, adhoc_plan, adhoc, ref_ant, nant)
            grp = nothing
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
