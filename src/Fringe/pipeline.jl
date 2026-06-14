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

using DimensionalData: lookup, dims, Ti
using ..UVData: Frequency
using Statistics: mean

# The shared fringe model: phase = per-scan {const, delay, rate} + per-AP adhoc
# const, all per-feed, all over the global frequency band. Log-amplitude empty.
function _fringe_model()
    return StationGainModel(
        phase = (
            TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), PerFeed()),
            TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            TiedComponent(GainComponent(Rate(), PerScan(), GlobalFrequency()), PerFeed()),
            TiedComponent(GainComponent(ConstantTerm(), PerIntegration(), GlobalFrequency()), PerFeed()),
        ),
    )
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
function _scan_group_leaves(uvset::UVSet)
    groups = Dict{Tuple{String, String}, Vector{Any}}()
    order = Tuple{String, String}[]
    for (_, leaf) in UVData.branches(uvset)
        info = UVData.metadata(leaf)
        key = (info.source_name, info.scan_name)
        if !haskey(groups, key)
            groups[key] = Any[]
            push!(order, key)
        end
        push!(groups[key], leaf)            # lazy leaf reference
    end
    return [groups[k] for k in order]
end

# Materialize ONE (source, scan) group: concatenate its sibling band leaves along
# the frequency axis (sorted by channel frequency) and resolve global geom
# indices. Called per group inside the solve loop so only one group's worth of
# data (~hundreds of MB) is resident at a time.
function _materialize_scan_group(leaves_lazy, geom::DataGeometry)
    leaves = [materialize_leaf(l) for l in leaves_lazy]
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

# Accumulate a per-(ant,feed) station value (NaN-skipping) into θ at the given
# component plan's (tseg, fseg=1) block. Accumulation (not overwrite) is what
# makes `rounds > 1` correct: each round searches the residual (data ÷ current
# gains) and finds the *incremental* delay/rate/phase, which adds in the gain
# exponent. Round 1 starts from θ = 0, so `+=` still sets the initial value.
function _pack_station!(θ, plan::ComponentPlan, vals::AbstractMatrix, tseg::Int)
    nant = size(vals, 1)
    for ant in 1:nant, feed in 1:2
        v = vals[ant, feed]
        isfinite(v) || continue
        off = plan.off1[ant, feed, tseg, 1]
        off == 0 && continue
        θ[off] += v
    end
    return θ
end

# Solve one materialized scan group into `θ` (the chunk's buffer): search →
# stationize → pack Stage-B, repeated `rounds` times on the residual, then the
# globally-closing adhoc phase. Writes only this group's (disjoint) θ slots.
# Returns `(max_snr, chi, ncomp)` from the final round for diagnostics.
function _solve_one_group!(θ, grp::_ScanGroup, ev, plans, f0, t0_sec, search, adhoc, rounds, ref_ant, nant, ws)
    const_plan, delay_plan, rate_plan, adhoc_plan = plans
    nbl = length(grp.bl_pairs)
    npol = length(grp.pol_products)
    stseg = const_plan.tseg_id[first(grp.g_ti)]   # PerScan segment for this group
    maxsnr = 0.0
    chi = NaN
    ncomp = 0

    for round in 1:max(rounds, 1)
        # On rounds > 1, divide out the current θ gains so the search runs on the
        # residual; round 1 searches the raw data.
        Vsearch = round > 1 ? _residual_vis(ev, θ, grp) : grp.Vg

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

        ss = stationize_scan(det, grp.bl_pairs, grp.pol_products, nant; ref_ant = ref_ant)
        _pack_station!(θ, const_plan, ss.phase, stseg)
        _pack_station!(θ, delay_plan, ss.delay, stseg)
        _pack_station!(θ, rate_plan, ss.rate, stseg)
        chi = ss.chi
        ncomp = ss.ncomp
    end

    # ── Adhoc: residual after Stage-B, coherently freq-averaged per AP. ──
    # Fused: evaluate gains once and accumulate the per-AP coherent sum directly,
    # without materializing a full residual cube (saves a Vg-sized array/group).
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
    return maxsnr, chi, ncomp
end

"""
    solve_fringes(uvset; search, adhoc, rounds, ref_ant) -> CalibrationSolution

Fringe-fit `uvset`. Builds a shared per-feed fringe model (per-scan constant
phase + delay + rate, plus a per-integration adhoc-phase term), solves each
(source, scan) group of sibling band leaves with a multi-band fringe search →
stationization → globally-closing adhoc phasing, and returns the packed
`CalibrationSolution`.

`search::FringeSearch` and `adhoc::AdhocPhasing` tune the two stages; `rounds`
re-runs the search/stationize stage on the (previous round's) corrected data;
`ref_ant` is the gauge reference station.
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
    # nant from the first leaf's antenna table.
    first_leaf = first(values(UVData.branches(uvset)))
    nant = length(UVData.metadata(first_leaf).antennas)
    layout = plan_parameters(model, nant, geom)
    ev = GainEvaluator(model, layout)
    plans = (layout.plans[1], layout.plans[2], layout.plans[3], layout.plans[4])
    f0 = geom.f0
    t0_sec = geom.t0 * 3600.0

    # Lazy grouping — leaves are materialized one group at a time inside the loop
    # so peak memory is one scan group (hundreds of MB), not the whole file.
    group_leaves = _scan_group_leaves(uvset)
    ngroups = length(group_leaves)
    scan_snr = zeros(ngroups)
    scan_chi = fill(NaN, ngroups)
    scan_ncomp = zeros(Int, ngroups)

    # Threaded over scan groups via OhMyThreads with a bounded task count. Each
    # (source, scan) group is independent and writes a DISJOINT set of θ slots
    # (its own PerScan / PerIntegration time segment), so each group fills its own
    # (cheap, ~1 MB) θ contribution which are summed at the end — the result is
    # identical to the sequential solve (the nonzero slots never overlap)
    # regardless of task count. `ntasks` caps how many groups are resident at once
    # (peak RAM ≈ ntasks × per-group size), so it is the knob to scale threads to
    # the machine's MEMORY, not just its core count. The FFT workspace is task-
    # local (one per task, reused across that task's groups).
    ntasks_use = max(1, min(Int(ntasks), ngroups))
    ws_tlv = TaskLocalValue{FringeWorkspace}(FringeWorkspace)
    contribs = tmap(1:ngroups; ntasks = ntasks_use) do gi
        θloc = zeros(layout.nθ)
        grp = _materialize_scan_group(group_leaves[gi], geom)
        snr, chi, nc = _solve_one_group!(
            θloc, grp, ev, plans, f0, t0_sec, search, adhoc, rounds, ref_ant, nant, ws_tlv[],
        )
        scan_snr[gi] = snr
        scan_chi[gi] = chi
        scan_ncomp[gi] = nc
        θloc
    end
    θ = isempty(contribs) ? zeros(layout.nθ) : reduce(+, contribs)

    info = (;
        nant = nant,
        nscan = ngroups,
        scan_max_snr = scan_snr,
        scan_chi = scan_chi,
        scan_ncomp = scan_ncomp,
    )
    return CalibrationSolution(model, layout, geom, θ, info)
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
