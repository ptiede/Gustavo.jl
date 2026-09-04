# ── Per-scan dTEC/SBD refinement over scan windows ────────────────────────────
#
# The per-scan (Δτ, dTEC) and per-band-group SBD refinement, operating on a
# scan's `DimStack` and its `GeometryWindow` (descended verbatim from the
# monolithic solver's concat-cube variants, deleted at M5). Both are driven by
# `DispersionSBDFit`'s visitor hooks (`src/pipeline/steps.jl`), on data already
# fringe-corrected through the pipeline's transform chain — neither kernel
# evaluates a gain itself. The per-baseline matched-filter measurements
# (`_fit_band_dispersion`, `_fit_chunk_delay`, and their band/chunk-phasor
# accumulators) live in `refine_search.jl`; this file assembles those
# measurements per scan and station-solves them (closure screen + robust
# rejection, via `solve_station_systems!`) into the private per-scan θ columns
# `DispersionSBDFit` owns.

# The view's grp-local channel ranges per spectral window (the concat cube's
# per-band blocks, recovered from the global channel indices).
function _spw_blocks(geom::DataGeometry, chan_idx)
    soc = geom.spw_of_chan
    blocks = UnitRange{Int}[]
    lo = 1
    for c in 2:length(chan_idx)
        if soc[chan_idx[c]] != soc[chan_idx[c - 1]]
            push!(blocks, lo:(c - 1))
            lo = c
        end
    end
    push!(blocks, lo:length(chan_idx))
    return blocks
end

"""
    refine_scan_dispersion!(θ, stack, win::GeometryWindow, delay_plan, disp_plan,
                            gauge, nant; opts, tau_max, dtec_max,
                            executor, ties) -> nothing

Joint per-scan (Δτ, dTEC) refinement of one scan window, on data already
gain-corrected through the pipeline's transform chain: per-spw band phasors →
per-baseline joint fits → two station solves accumulating into `delay_plan`
(a private per-scan delay REFINEMENT column, distinct from and additional to
the wideband delay the fringe search already solved — gains compose
multiplicatively, so this is the same total delay as incrementing one shared
column) and `disp_plan` (the dispersion θ block). The window's per-spw channel
blocks stand in for the band leaves (the concat-cube path). Returns the number
of detections the robust station solves excised; a no-op (0) when
`disp_plan === nothing` or fewer than 4 bands are present (1/ν is
unconstrainable).
"""
function refine_scan_dispersion!(
        θ, stack::AbstractDimStack, win::GeometryWindow, delay_plan, disp_plan, gauge, nant;
        opts::Stationization = Stationization(loss = LeastSquares()),
        tau_max::Real = 2.0e-8, dtec_max::Real = 45.0, executor = DynamicScheduler(),
        ties = nothing,
    )
    disp_plan === nothing && return 0
    geom = win.geom
    ci = win.chan_idx
    ti = win.ti_idx
    blocks = _spw_blocks(geom, ci)
    length(blocks) >= 4 || return 0              # < 4 bands can't constrain 1/ν
    V = stack[:vis]
    W = stack[:weights]
    fg = frequencies(stack)
    bl_pairs = UVData.baselines(stack).pairs
    pols = pol_products(stack)
    feeds = [correlation_feed_pair(p) for p in pols]
    nbl = length(bl_pairs)
    npol = length(pols)
    nlf = length(blocks)
    z = zeros(ComplexF64, nbl, npol, nlf)
    w = zeros(Float64, nbl, npol, nlf)
    fb = [sum(@view fg[r]) / length(r) for r in blocks]
    tforeach(1:nlf; scheduler = executor) do li
        r = blocks[li]
        _accumulate_leaf_band_phasor!(
            view(z, :, :, li), view(w, :, :, li),
            view(V, r, :, :, :), view(W, r, :, :, :),
            bl_pairs, pols,
        )
    end
    return _dispersion_fit_stationize!(
        θ, z, w, fb, bl_pairs, pols, feeds, first(ti), geom,
        delay_plan, disp_plan, gauge, opts,
        Float64(tau_max), Float64(dtec_max), ties,
    )
end

"""
    refine_scan_sbd!(θ, stack, win::GeometryWindow, sbd, gauge, nant;
                     nchunk = 4, tau_max = 6.0e-8, executor = DynamicScheduler()) -> nrej

Per-scan per-band-group SBD refinement of one scan window (fourfit's single-band
delay), on data already gain-corrected through the pipeline's transform chain:
each spw block's sub-band chunk phasors → per-(baseline, group) exact
matched-filter slope fits → guarded station solves accumulating into the
per-scan `Delay × FreqGroups` column and its companion constant. Run after
the dispersion refinement so the within-band slopes it fits are
dispersion-corrected. A no-op (0) when `sbd === nothing`.
"""
function refine_scan_sbd!(
        θ, stack::AbstractDimStack, win::GeometryWindow, sbd, gauge, nant;
        nchunk::Integer = 4, tau_max::Real = 6.0e-8, executor = DynamicScheduler(),
    )
    sbd === nothing && return 0
    geom = win.geom
    ci = win.chan_idx
    ti = win.ti_idx
    blocks = _spw_blocks(geom, ci)
    V = stack[:vis]
    W = stack[:weights]
    fg = frequencies(stack)
    bl_pairs = UVData.baselines(stack).pairs
    pols = pol_products(stack)
    feeds = [correlation_feed_pair(p) for p in pols]
    nbl = length(bl_pairs)
    npol = length(pols)
    nlf = length(blocks)
    ntot = nlf * Int(nchunk)
    z = zeros(ComplexF64, nbl, npol, ntot)
    w = zeros(Float64, nbl, npol, ntot)
    chunkf = zeros(Float64, ntot)
    chunkgrp = zeros(Int, ntot)
    tforeach(1:nlf; scheduler = executor) do li
        r = blocks[li]
        fs = fg[r]
        nc = length(fs)
        bgrp = findfirst(rr -> ci[first(r)] in rr, sbd.freqgroups)
        bgrp === nothing && return
        edges = round.(Int, range(0, nc; length = Int(nchunk) + 1))
        coc = Vector{Int}(undef, nc)
        for k in 1:Int(nchunk)
            klo, khi = edges[k] + 1, edges[k + 1]
            khi >= klo || continue
            kk = (li - 1) * Int(nchunk) + k
            coc[klo:khi] .= kk
            chunkf[kk] = sum(@view fs[klo:khi]) / (khi - klo + 1)
            chunkgrp[kk] = bgrp
        end
        _accumulate_leaf_chunks!(
            view(z, :, :, ((li - 1) * Int(nchunk) + 1):(li * Int(nchunk))),
            view(w, :, :, ((li - 1) * Int(nchunk) + 1):(li * Int(nchunk))),
            view(V, r, :, :, :), view(W, r, :, :, :),
            bl_pairs, pols,
            coc .- (li - 1) * Int(nchunk),
        )
    end
    return _sbd_fit_stationize!(
        θ, z, w, chunkf, chunkgrp, bl_pairs, pols, feeds,
        first(ti), geom, sbd, gauge, nant;
        tau_max = Float64(tau_max),
    )
end

# Refine one materialized scan group: band phasors → per-baseline (Δτ, dTEC) →
# two station solves accumulating into the per-scan delay and dispersion columns
# (disjoint per scan, so pass-2 groups can refine concurrently). Returns the
# number of detections the robust station solves excised.
function _dispersion_fit_stationize!(
        θ, z, w, fb, bl_pairs, pols, feeds, ti0, geom,
        delay_plan, disp_plan, gauge, opts, tau_max, dtec_max, ties = nothing,
    )
    nbl, npol, nlf = size(z)
    Dτ = fill(_invalid_detection(Float64), nbl, npol)
    Dd = fill(_invalid_detection(Float64), nbl, npol)
    for p in 1:npol
        feeds[p][1] == feeds[p][2] || continue
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            rows_f = Float64[]
            rows_z = ComplexF64[]
            rows_w = Float64[]
            for li in 1:nlf
                w[bi, p, li] > 0 || continue
                push!(rows_f, fb[li])
                push!(rows_z, z[bi, p, li])
                push!(rows_w, w[bi, p, li])
            end
            length(rows_f) >= 4 || continue
            fit = _fit_band_dispersion(
                rows_f, rows_z, rows_w, geom.f0;
                tau_max = _band_delay_halfwindow(rows_f, tau_max), dtec_max = dtec_max,
            )
            # Every fit is recorded with its false-alarm probability; `opts.pfa_max`
            # decides which are real, exactly as it does for a search detection.
            pfa = fringe_pfa(fit.snr, fit.ncells)
            Dτ[bi, p] = Detection{Float64}((fit.tau, 0.0, 0.0, fit.amp, fit.snr, pfa, true))
            Dd[bi, p] = Detection{Float64}((fit.dtec, 0.0, 0.0, fit.amp, fit.snr, pfa, true))
        end
    end

    if haskey(ENV, "GUSTAVO_DTEC_DEBUG")
        for p in 1:npol, bi in 1:nbl
            Dτ[bi, p].valid || continue
            println(
                "  dtec-fit bl=", bl_pairs[bi], " p=", p,
                " Δτ=", round(Dτ[bi, p].delay * 1.0e12; digits = 1), "ps",
                " dtec=", round(Dd[bi, p].delay; digits = 3),
                " snr=", round(Dτ[bi, p].snr; digits = 1),
            )
        end
    end
    tied = ties !== nothing && any(ties[i] != i for i in eachindex(ties))
    # These solves fit only a delay, from the per-band phasors, so the band
    # centres are the lever arm and no time spread is consulted.
    freq_rms = _rms_spread(fb)
    time_rms = nothing
    for (D, plan, tie) in ((Dτ, delay_plan, false), (Dd, disp_plan, tied))
        plan === nothing && continue
        any(d -> d.valid, D) || continue
        pairs_s = bl_pairs
        Ds = D
        if tie
            # Solve on representative nodes; a baseline whose endpoints collapse
            # to the same representative (the co-located pair itself) carries no
            # differential-TEC information and is dropped.
            pairs_s = [(ties[a], ties[b]) for (a, b) in bl_pairs]
            Ds = copy(D)
            for bi in eachindex(pairs_s)
                if pairs_s[bi][1] == pairs_s[bi][2]
                    for p in axes(Ds, 2)
                        Ds[bi, p] = _invalid_detection(Float64)
                    end
                end
            end
            any(d -> d.valid, Ds) || continue
        end
        solve_station_systems!(
            θ, (detection_stack(Ds, pairs_s, pols; ti = ti0, freq_rms, time_rms),), ((plan, :delay),);
            gauge = tie ? remap_gauge(gauge, ties) : gauge, opts = opts,
        )
        if tie
            # Members inherit the representative's solved value (assignment, not
            # increment — both start equal, so totals stay equal).
            ts = plan.tseg_id[ti0]
            leaf = _component_leaf(plan, θ)
            for a in eachindex(ties)
                ties[a] == a && continue
                for f in 1:2
                    node = _feed_node(plan.tying, f)
                    node == 0 && continue
                    leaf[1, node, 1, ts, a] = leaf[1, node, 1, ts, ties[a]]
                end
            end
        end
    end
    return nothing
end

# Shared back half: per-(baseline, group) fits over the accumulated chunk
# phasors — the within-group SLOPE (exact matched filter) plus the slope-
# corrected group phasor's phase — then one feed-common station solve of each
# per group. Both are needed: the wideband delay's decomposition against the
# per-group slopes is ambiguous (a common-mode slope shift leaves per-group
# constants of 2πΔτ(f0 − νg) behind that no single per-scan constant can
# absorb), and real instruments carry genuine per-band phase offsets. θ gets
# the per-scan per-group delay and its constant: net correction
# 2πτ(f − νg) + φg, referenced to the group centre. `chunkf`/`chunkgrp` label
# each accumulated chunk with its centre frequency and band-group id.
function _sbd_fit_stationize!(
        θ, z, w, chunkf, chunkgrp, bl_pairs, pols, feeds, ti0, geom, sbd, gauge, nant;
        opts::Stationization = Stationization(loss = LeastSquares()),
        tau_max::Float64 = 6.0e-8,
    )
    nbl, npol, _ = size(z)
    ngrp = length(sbd.freqgroups)
    nrej = 0
    # ── Tier 1: per-group fits + within-group SLOPE guard ───────────────────
    # The per-baseline SNR gate cannot catch a BIASED fit: on a group whose
    # chunk phasors are internally decoherent (real per-channel bandpass
    # structure, not a delay — VGOS band 2 is the worst), the matched filter
    # happily returns a confident wrong slope, and applying it bends every
    # baseline of a previously-better group (the VR2505 full run: +SBD
    # degraded g1/g2 η_f on 1803+784 and 4C39.25 and band-2 on 3C454.3 while
    # genuinely fixing 0607-157 and YJ). So: keep a group's stationized slope
    # only when it RAISES the group's within-band coherence on the very chunk
    # phasors it was fit from. The group CONSTANT is judged separately — it
    # cancels inside the within-group coherence (|Σ z·e^{-iφ}| = |Σ z|), so
    # this tier is structurally blind to it. A failing slope is ZEROED and the
    # group phase refit at τ = 0 (not dropped with the group: φg carries the
    # cross-band alignment, and dropping it collapsed 1803+784's cross-band η
    # in the first, group-dropping version of this guard).
    gsols = NamedTuple[]
    for gidx in 1:ngrp
        ks = findall(==(gidx), chunkgrp)
        length(ks) >= 3 || continue
        fc = sum(chunkf[ks]) / length(ks)       # FIXED group centre (matches the θ write)
        rows_τ = _ObsRow[]
        # per accepted baseline row: (a, b, φ at fitted slope, φ at τ = 0, weight)
        φrows = Tuple{Int, Int, Float64, Float64, Float64}[]
        for p in 1:npol
            feeds[p][1] == feeds[p][2] || continue
            for bi in 1:nbl
                a, b = bl_pairs[bi]
                a == b && continue
                fs = Float64[]
                zs = ComplexF64[]
                ws = Float64[]
                for k in ks
                    w[bi, p, k] > 0 || continue
                    push!(fs, chunkf[k])
                    push!(zs, z[bi, p, k])
                    push!(ws, w[bi, p, k])
                end
                length(fs) >= 3 || continue
                fit = _fit_chunk_delay(fs, zs, ws, fc; tau_max = _band_delay_halfwindow(fs, tau_max))
                # A fit above `pfa_max` still constrains, at `weak_sys_scale`-inflated
                # σ — i.e. its weight (an inverse variance) divided by that squared.
                accept = fringe_pfa(fit.snr, fit.ncells) <= opts.pfa_max
                rw = accept ? fit.snr^2 : fit.snr^2 / opts.weak_sys_scale^2
                φs = angle(sum(zs[k] * cis(-2π * fit.tau * (fs[k] - fc)) for k in eachindex(fs)))
                φ0 = angle(sum(zs))
                # feed-common node (SharedFeeds model): both hands constrain feed 1.
                push!(rows_τ, _ObsRow(a, b, 1, 1, fit.tau, rw))
                push!(φrows, (a, b, φs, φ0, rw))
            end
        end
        isempty(rows_τ) && continue
        τv, covτ, _ = _solve_observable_robust(rows_τ, nant, gauge, opts; rewrap = 0)
        num0 = 0.0
        num1 = 0.0
        for p in 1:npol
            feeds[p][1] == feeds[p][2] || continue
            for bi in 1:nbl
                a, b = bl_pairs[bi]
                a == b && continue
                τab = (covτ[a, 1] ? τv[a, 1] : 0.0) - (covτ[b, 1] ? τv[b, 1] : 0.0)
                acc0 = zero(ComplexF64)
                acc1 = zero(ComplexF64)
                for k in ks
                    w[bi, p, k] > 0 || continue
                    zk = z[bi, p, k]
                    acc0 += zk
                    acc1 += zk * cis(-2π * τab * (chunkf[k] - fc))
                end
                num0 += abs(acc0)
                num1 += abs(acc1)
            end
        end
        slope_ok = num1 > num0
        if !slope_ok
            nrej += 1
            fill!(τv, 0.0)
        end
        rows_φ = [_ObsRow(r[1], r[2], 1, 1, slope_ok ? r[3] : r[4], r[5]) for r in φrows]
        φv, covφ, _ = _solve_observable_robust(rows_φ, nant, gauge, opts; rewrap = 2)
        push!(gsols, (; gidx, ks, fc, τv, covτ, φv, covφ))
    end
    isempty(gsols) && return nrej
    # ── Tier 2: scan-level JOINT guard on the CROSS-band statistic ──────────
    # The group constants' whole job is aligning the groups' band phasors, so
    # judge them (together with the surviving slopes) on the coherent sum over
    # All groups per baseline — the ηx-flavoured statistic — and apply the
    # scan's SBD solution all-or-nothing.
    num0 = 0.0
    num1 = 0.0
    for p in 1:npol
        feeds[p][1] == feeds[p][2] || continue
        for bi in 1:nbl
            a, b = bl_pairs[bi]
            a == b && continue
            acc0 = zero(ComplexF64)
            acc1 = zero(ComplexF64)
            for gs in gsols
                τab = (gs.covτ[a, 1] ? gs.τv[a, 1] : 0.0) - (gs.covτ[b, 1] ? gs.τv[b, 1] : 0.0)
                φab = (gs.covφ[a, 1] ? gs.φv[a, 1] : 0.0) - (gs.covφ[b, 1] ? gs.φv[b, 1] : 0.0)
                for k in gs.ks
                    w[bi, p, k] > 0 || continue
                    zk = z[bi, p, k]
                    acc0 += zk
                    acc1 += zk * cis(-(2π * τab * (chunkf[k] - gs.fc) + φab))
                end
            end
            num0 += abs(acc0)
            num1 += abs(acc1)
        end
    end
    num1 > num0 || return nrej + length(gsols)
    ts = sbd.dplan.tseg_id[ti0]
    dleaf = _component_leaf(sbd.dplan, θ)
    cleaf = _component_leaf(sbd.cplan, θ)
    for gs in gsols
        for a in 1:nant
            τa = (gs.covτ[a, 1] && isfinite(gs.τv[a, 1])) ? gs.τv[a, 1] : 0.0
            φa = (gs.covφ[a, 1] && isfinite(gs.φv[a, 1])) ? gs.φv[a, 1] : 0.0
            (τa == 0.0 && φa == 0.0) && continue
            nd = _feed_node(sbd.dplan.tying, 1)
            nd == 0 && continue
            dleaf[1, nd, gs.gidx, ts, a] += τa
            nc = _feed_node(sbd.cplan.tying, 1)
            nc == 0 && continue
            cleaf[1, nc, gs.gidx, ts, a] += φa - 2π * τa * (gs.fc - geom.f0)
        end
    end
    return nrej
end
