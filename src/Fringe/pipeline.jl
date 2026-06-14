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
    Vg::Array{ComplexF64, 4}             # (chan, ti, bl, pol)
    Wg::Array{Float64, 4}
    fg::Vector{Float64}                  # channel freqs (Hz), all bands stacked
    tg::Vector{Float64}                  # times (hours)
    bl_pairs::Vector{Tuple{Int, Int}}
    pol_products::Vector{String}
    g_ci::Vector{Int}                    # global channel indices of fg
    g_ti::Vector{Int}                    # global time indices of tg
end

# Group leaves by (source_name, scan_name), concatenate sibling bands along the
# frequency axis (sorted by channel frequency), and resolve their global geom
# indices.
function _build_scan_groups(uvset::UVSet, geom::DataGeometry)
    groups = Dict{Tuple{String, String}, Vector{Any}}()
    order = Tuple{String, String}[]
    for (_, leaf) in UVData.branches(uvset)
        info = UVData.metadata(leaf)
        key = (info.source_name, info.scan_name)
        if !haskey(groups, key)
            groups[key] = Any[]
            push!(order, key)
        end
        push!(groups[key], materialize_leaf(leaf))
    end

    out = _ScanGroup[]
    for key in order
        leaves = groups[key]
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

        Vg = Array{ComplexF64}(undef, nchan, nti, nbl, npol)
        Wg = Array{Float64}(undef, nchan, nti, nbl, npol)
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
        push!(out, _ScanGroup(Vg, Wg, fg, tg, bl_pairs, pols, g_ci, g_ti))
    end
    return out
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
    )
    model = _fringe_model()
    geom = build_geometry(uvset)
    # nant from the first leaf's antenna table.
    first_leaf = first(values(UVData.branches(uvset)))
    nant = length(UVData.metadata(first_leaf).antennas)
    layout = plan_parameters(model, nant, geom)
    θ = zeros(layout.nθ)
    ev = GainEvaluator(model, layout)

    groups = _build_scan_groups(uvset, geom)

    const_plan = layout.plans[1]
    delay_plan = layout.plans[2]
    rate_plan = layout.plans[3]
    adhoc_plan = layout.plans[4]

    f0 = geom.f0
    t0_sec = geom.t0 * 3600.0

    scan_snr = Float64[]
    scan_chi = Float64[]
    scan_ncomp = Int[]

    for grp in groups
        nbl = length(grp.bl_pairs)
        npol = length(grp.pol_products)
        # PerScan tseg id for this group (constant across the group's times).
        stseg = const_plan.tseg_id[first(grp.g_ti)]

        for round in 1:max(rounds, 1)
            # On rounds > 1, divide out the current θ gains (Stage-B + adhoc) so
            # the search runs on the residual; round 1 searches the raw data.
            Vsearch = grp.Vg
            if round > 1
                Vsearch = _residual_vis(ev, θ, grp)
            end

            # Per-baseline fringe search over the concatenated band.
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
                    grp.fg, grp.tg .* 3600.0, f0, t0_sec; opts = search,
                )
                det[bi, p] = d
                d.valid && (maxsnr = max(maxsnr, d.snr))
            end

            ss = stationize_scan(det, grp.bl_pairs, grp.pol_products, nant; ref_ant = ref_ant)

            # Accumulate Stage-B station gains; on rounds > 1 this adds the
            # residual delay/rate/phase found on the corrected data.
            _pack_station!(θ, const_plan, ss.phase, stseg)
            _pack_station!(θ, delay_plan, ss.delay, stseg)
            _pack_station!(θ, rate_plan, ss.rate, stseg)

            if round == max(rounds, 1)
                push!(scan_snr, maxsnr)
                push!(scan_chi, ss.chi)
                push!(scan_ncomp, ss.ncomp)
            end
        end

        # ── Adhoc: residual after Stage-B, coherently freq-averaged per AP. ──
        Vresid = _residual_vis(ev, θ, grp)
        nap = length(grp.tg)
        rbar = zeros(ComplexF64, nbl, npol, nap)
        wbar = zeros(Float64, nbl, npol, nap)
        nchan = length(grp.fg)
        @inbounds for p in 1:npol, bi in 1:nbl, ap in 1:nap, c in 1:nchan
            w = grp.Wg[c, ap, bi, p]
            v = Vresid[c, ap, bi, p]
            (isfinite(w) && w > 0 && isfinite(v)) || continue
            rbar[bi, p, ap] += w * v
            wbar[bi, p, ap] += w
        end
        as = solve_adhoc_phasing(
            rbar, wbar, grp.bl_pairs, grp.pol_products, nant, grp.tg;
            ref_ant = ref_ant, opts = adhoc,
        )

        # Pack adhoc per-AP station phases into the PerIntegration component.
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
    end

    info = (;
        nant = nant,
        nscan = length(groups),
        scan_max_snr = scan_snr,
        scan_chi = scan_chi,
        scan_ncomp = scan_ncomp,
    )
    return CalibrationSolution(model, layout, geom, θ, info)
end

# Residual visibilities for one scan group: Vg divided by the current θ gains
# evaluated at the group's (global chan, global ti) window.
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
