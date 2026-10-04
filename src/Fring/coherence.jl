# ── Coherence diagnostics (stage-agnostic) ───────────────────────────────────
#
# How much amplitude does *coherent* (vector) averaging lose relative to
# *incoherent* (scalar) averaging? For a set of visibility cells the coherence
# factor is
#
#     η = |Σ w·V| / Σ (w·|V|)   ∈ (0, 1].
#
# η = 1 means the cells share a phase, so coherently averaging them is lossless;
# η < 1 means residual phase scatter within the averaging cell decorrelates the
# vector sum, and (1 − η) of the amplitude (and SNR) is lost by averaging. This
# is the fundamental QA metric for a fringe/bandpass solution: after a correct
# fit the per-baseline residual phase is flat in time (rate/adhoc removed) and in
# frequency (delay/bandpass removed), so averaging the data — the whole point of
# fringe fitting — stays coherent out to long timescales and wide bandwidths.
#
# `coherence_report` measures η of one scan group against time-averaging
# interval Δt and frequency-averaging width Δν, per (antenna pair, feed pair)
# and pooled over antenna pairs. The accumulation kernels below work on
# `(Frequency, Ti, BaselineID, Polarization)` cubes.
#
# CAVEAT: η is computed against the data's own finest resolution (a single
# integration / channel gives η ≡ 1), so it is self-normalized and needs no
# external gain model. By default it is not thermal-noise-debiased: pure thermal
# noise alone pulls η below 1 at coarse averaging (the incoherent |V| is
# noise-inflated while the coherent sum averages noise down), so at native per-cell
# SNR the absolute η UNDERSTATES a good solution — read the *shape* and the
# *before/after* comparison, not the absolute value. Pass `debias = true` to
# remove that bias; it takes the weights as inverse variances per real
# component (`w = 1/Var(Re V)`).
#
# The debiased estimator works in POWER, with a single square root at the end:
# per averaging bin, `|Σ w·V|² − 2Σw` is an unbiased estimate of the bin's
# coherent signal power `|s̄|²(Σw)²` at any SNR, so
# bins (and cells, and baselines) are pooled as `Σ (|Σ w·V|² − 2Σw)/Σw` — no
# per-bin clip, no per-bin square root — and η = √(pooled power at Δ / pooled
# power at native binning). A per-bin amplitude `√(max(|Σ w·V|² − 2Σw, 0))`
# would fold noise at bin SNR ≲ 1 (the clip and the root are both nonlinear),
# which draws a spurious dip-and-recover curve for a perfectly coherent weak
# baseline and drags a pooled η down through its noise-only members; in the
# power domain those fluctuations cancel in expectation instead. The price is
# variance, and it is reported honestly: a denominator that does not DETECT
# signal power at 3σ of its null fluctuation (see `_curve_from_sums`) has no
# coherence measurement and reads NaN — never a clamped ratio of noise over
# noise — and a weak baseline's trace is noisy-but-unbiased rather than
# smoothly wrong.

# Doubling sweep of averaging intervals from the native spacing until one
# interval exceeds the largest span, so the last bin holds the whole extent.
# The native spacing is the *minimum* positive step, not the median: on a
# non-uniform grid a median step merges the closest pair into one bin and
# breaks the η ≡ 1 anchor at native resolution. Empty when the axis has no
# spacing (one sample or channel everywhere).
function _auto_intervals(diffs::Vector{Float64}, spans::Vector{Float64})
    isempty(diffs) && return Float64[]
    native = minimum(diffs)
    (native > 0 && isfinite(native)) || return Float64[]
    full = isempty(spans) ? 0.0 : maximum(spans)
    out = [native]
    while last(out) <= full
        push!(out, 2 * last(out))
    end
    return out
end

# Per-baseline η and the pooled aggregate, per interval.
#
# Amplitude mode (`power = false`, the raw estimator): η = num/den with
# num = Σ|Σ w·V| and den = Σ w·|V|; ≤ 1 by the triangle inequality, clamped to
# the ceiling against rounding. A baseline with no data (den = 0) is skipped.
#
# Power mode (`power = true`, the debiased estimator): num and den are pooled
# unbiased signal powers, and η = √(num/den) — one square root, on the ratio.
# The ratio is clamped to [0, 1] (its fluctuations are unbounded either way for
# weak data; the estimand is). A baseline whose own measured power is not
# positive has no coherence measurement and reads NaN — but it still enters the
# pooled sums: its expected contribution to both is zero, and excluding it on
# the realized sign of a noise fluctuation would bias the aggregate.
# In power mode a ratio is reported only where its denominator DETECTS signal
# power: `den > 3·√denvar`, with `denvar` the exact null variance of the pooled
# power sum (4 per cell). Below that there is no coherence
# measurement — the ratio of two near-zero fluctuations would clamp into
# [0, 1] and read as a confident number — so the entry is NaN instead.
function _curve_from_sums(
        num::Matrix{Float64}, den::Vector{Float64}, denvar::Vector{Float64}, power::Bool,
    )
    nrow, nbl = size(num)
    eta_bl = fill(NaN, nrow, nbl)
    agg = fill(NaN, nrow)
    for k in axes(num, 1)
        sn = 0.0; sd = 0.0; sv = 0.0
        for bl in axes(num, 2)
            d = den[bl]
            if power
                d == 0.0 && num[k, bl] == 0.0 && continue      # no data at all
                d > 3 * sqrt(max(denvar[bl], 0.0)) &&
                    (eta_bl[k, bl] = sqrt(clamp(num[k, bl] / d, 0.0, 1.0)))
                sn += num[k, bl]; sd += d; sv += denvar[bl]
            else
                d > 0 || continue
                eta_bl[k, bl] = min(num[k, bl] / d, 1.0)
                sn += num[k, bl]; sd += d
            end
        end
        if power
            sd > 3 * sqrt(max(sv, 0.0)) && (agg[k] = sqrt(clamp(sn / sd, 0.0, 1.0)))
        else
            sd > 0 && (agg[k] = min(sn / sd, 1.0))
        end
    end
    return eta_bl, agg
end

# Accumulate one Measurement Set's contribution. `V`/`W` are `(Frequency, Ti, BaselineID, Polarization)`;
# `blmap[bli]` maps a local baseline to its global index (0 = skip: autocorr or
# unmapped). A function barrier so the hot loops specialize on the concrete eltypes
# of the layer arrays.
#
# Binning is made independent of on-disk storage direction: lower-sideband bands
# have negative CH_WIDTH, so `freqs` (and occasionally `times_sec`) can be stored
# DESCENDING. We anchor each axis's bin ids at its MINIMUM coordinate (`t0`/`f0`)
# and iterate samples in ascending-coordinate order via `tperm`/`cperm`, so the
# half-open bin id `floor((x − x0)/Δ)` is non-decreasing along the walk and the
# flush-on-id-change stays contiguous regardless of direction. (V/W are still
# indexed by the original `c`/`ti`; only the *visit order* is permuted.)
#
# TIME bins (per channel): walk ascending time, flush the coherent sum `|Σ w·V|`
# when the bin id changes; FREQ bins (per AP): same over ascending freq. Bin ids
# depend only on (sample, interval) — not on baseline/pol — so they are tabulated
# Once (`tid`/`fid`) and each (c, ti) cell is read O(1) times w.r.t. the number of
# intervals: a single walk fans every cell into all nT (or nF) running
# accumulators. The denominator Σ w·|V| and `npts` are interval-independent, summed
# once. Intervals are used as given.
# `Fl` is the flag layer over `V`, or `nothing` for an already-collapsed cube:
# `_collapse_axis` has excluded the flagged samples, so the reduced cells carry
# no flag of their own.
function _coherence_accumulate!(
        numT::Matrix{Float64}, numF::Matrix{Float64}, den::Vector{Float64}, dvar::Vector{Float64},
        npts::Vector{Int},
        V::AbstractArray{Tv, 4}, W::AbstractArray{Tw, 4}, Fl, blmap::Vector{Int}, plist::Vector{Int},
        times_sec::Vector{Float64}, freqs::Vector{Float64}, dts::Vector{Float64}, dnus::Vector{Float64},
        debias::Bool,
    ) where {Tv, Tw}
    Fl === nothing || UVData.check_layer_axes(V, W, Fl)
    nchan, nti, nbl, npol = size(V)
    nT = length(dts); nF = length(dnus)
    tperm = sortperm(times_sec)
    cperm = sortperm(freqs)
    t0 = isempty(times_sec) ? 0.0 : times_sec[tperm[1]]
    f0 = isempty(freqs) ? 0.0 : freqs[cperm[1]]

    # Bin-id tables (sample × interval), indexed by the ORIGINAL sample index.
    tid = Matrix{Int}(undef, nti, nT)
    for k in eachindex(dts)
        dt = dts[k]
        for ti in eachindex(times_sec)
            tid[ti, k] = dt > 0 ? floor(Int, (times_sec[ti] - t0) / dt) : 0
        end
    end
    fid = Matrix{Int}(undef, nchan, nF)
    for k in eachindex(dnus)
        dnu = dnus[k]
        for c in eachindex(freqs)
            fid[c, k] = dnu > 0 ? floor(Int, (freqs[c] - f0) / dnu) : 0
        end
    end

    # Running per-interval accumulators, reused across (baseline, pol, channel/AP).
    # `swT`/`swF` carry the per-bin Σw for the optional thermal debias.
    accT = Vector{ComplexF64}(undef, nT); swT = Vector{Float64}(undef, nT)
    curT = Vector{Int}(undef, nT); haveT = Vector{Bool}(undef, nT)
    accF = Vector{ComplexF64}(undef, nF); swF = Vector{Float64}(undef, nF)
    curF = Vector{Int}(undef, nF); haveF = Vector{Bool}(undef, nF)

    # Per-bin contribution. The weight is the inverse variance of one REAL
    # COMPONENT (`w = 1/σ²` with `Var(Re n) = Var(Im n) = 1/w`), so a cell's complex
    # noise power is `E|n|² = 2/w` — the factor of 2 below is that, not a fudge.
    # Hence `|Σ w·V|²` is noise-inflated by `Σ w²·E|n|² = 2Σw`.
    #
    # With `debias` the contribution is the unbiased per-BIN SIGNAL POWER
    # `(|Σ w·V|² − 2Σw)/Σw` — unclipped, no per-bin square root, so noise
    # fluctuations cancel across bins instead of folding at bin SNR ≲ 1; the
    # single square root is taken on the pooled ratio in `_curve_from_sums`.
    # Without it, the raw coherent amplitude `|Σ w·V|`. At native resolution
    # (one cell: `s = w·V`, `sw = w`) either form equals its denominator cell,
    # so η ≡ 1 there — an identity both estimators preserve.
    binval(s::ComplexF64, sw::Float64) = debias ? (abs2(s) - 2 * sw) / sw : abs(s)

    # Spanning the cube's own axes drops the checks on the `V`/`W`/`Fl` reads;
    # the annotation stays for `blmap`, `den`/`npts` and the per-bin
    # accumulators, which are indexed through values rather than loop ranges.
    @inbounds for bli in axes(V, 3)
        bl = blmap[bli]
        bl == 0 && continue
        for pli in eachindex(plist)
            p = plist[pli]
            p in 1:npol || continue

            # Denominator (interval-independent). Without `debias`, the incoherent
            # Σ w·|V|. With it, the numerator's own expression at NATIVE binning —
            # the per-cell unbiased power `(w²|V|² − 2w)/w = w|V|² − 2` — so η ≡ 1
            # at native resolution stays an identity and the ratio's denominator
            # is unbiased at any SNR (a per-cell clipped amplitude would sit
            # noise-inflated above the true signal for weak cells, holding η
            # below 1 even after the numerator's bins reach high SNR).
            for ti in axes(V, 2), c in axes(V, 1)
                _flagged(Fl, c, ti, bli, p) && continue
                w = W[c, ti, bli, p]; v = V[c, ti, bli, p]
                (w > 0 && isfinite(w) && isfinite(v)) || continue
                a2 = abs2(ComplexF64(v))
                if debias
                    den[bl] += Float64(w) * a2 - 2.0
                    dvar[bl] += 4.0
                else
                    den[bl] += Float64(w) * sqrt(a2)
                end
                npts[bl] += 1
            end

            # Time-binned coherent amplitude, per channel. One ascending-time walk
            # fans each cell into all nT intervals.
            for c in axes(V, 1)
                for k in eachindex(accT)
                    accT[k] = zero(ComplexF64); swT[k] = 0.0; haveT[k] = false
                end
                for ti in tperm
                    _flagged(Fl, c, ti, bli, p) && continue
                    w = W[c, ti, bli, p]; v = V[c, ti, bli, p]
                    (w > 0 && isfinite(w) && isfinite(v)) || continue
                    wv = w * ComplexF64(v)
                    for k in eachindex(accT)
                        id = tid[ti, k]
                        if !haveT[k]
                            curT[k] = id; haveT[k] = true
                        elseif id != curT[k]
                            numT[k, bl] += binval(accT[k], swT[k])
                            accT[k] = zero(ComplexF64); swT[k] = 0.0; curT[k] = id
                        end
                        accT[k] += wv; swT[k] += w
                    end
                end
                for k in eachindex(accT)
                    haveT[k] && (numT[k, bl] += binval(accT[k], swT[k]))
                end
            end

            # Frequency-binned coherent amplitude, per AP. One ascending-freq walk
            # fans each cell into all nF intervals.
            for ti in axes(V, 2)
                for k in eachindex(accF)
                    accF[k] = zero(ComplexF64); swF[k] = 0.0; haveF[k] = false
                end
                for c in cperm
                    _flagged(Fl, c, ti, bli, p) && continue
                    w = W[c, ti, bli, p]; v = V[c, ti, bli, p]
                    (w > 0 && isfinite(w) && isfinite(v)) || continue
                    wv = w * ComplexF64(v)
                    for k in eachindex(accF)
                        id = fid[c, k]
                        if !haveF[k]
                            curF[k] = id; haveF[k] = true
                        elseif id != curF[k]
                            numF[k, bl] += binval(accF[k], swF[k])
                            accF[k] = zero(ComplexF64); swF[k] = 0.0; curF[k] = id
                        end
                        accF[k] += wv; swF[k] += w
                    end
                end
                for k in eachindex(accF)
                    haveF[k] && (numF[k, bl] += binval(accF[k], swF[k]))
                end
            end
        end
    end
    return nothing
end

# Coherently weighted-average `V` (with weights `W`) over `axis` (1 = Frequency,
# 2 = Ti), returning `(Vbar, Wbar)` with that axis collapsed to length 1: `Vbar` is
# the weighted mean `Σ w·V / Σ w` and `Wbar = Σ w` (so its noise variance is `1/Wbar`,
# preserving the inverse-variance convention the debias relies on). For corrected
# data this is the high-SNR band-average (axis 1) or time-average (axis 2) a
# marginalized coherence curve is measured on; cells with no weight become `NaN`.
function _collapse_axis(
        V::AbstractArray{<:Any, 4}, W::AbstractArray{<:Any, 4},
        Fl::AbstractArray{Bool, 4}, axis::Int,
    )
    nchan, nti, nbl, npol = size(V)
    # The collapsed axis keeps a length-1 slot; the surviving one keeps the
    # cube's own axis, so a caller's axes flow into the result.
    kept, gone = axis == 1 ? (axes(V, 2), axes(V, 1)) : (axes(V, 1), axes(V, 2))
    Vbar = fill(ComplexF64(NaN), axis == 1 ? 1 : nchan, axis == 1 ? nti : 1, nbl, npol)
    Wbar = zeros(Float64, axis == 1 ? 1 : nchan, axis == 1 ? nti : 1, nbl, npol)
    @inbounds for p in axes(V, 4), bl in axes(V, 3), j in kept
        s = zero(ComplexF64); w = 0.0
        for i in gone
            c, ti = axis == 1 ? (i, j) : (j, i)
            Fl[c, ti, bl, p] && continue
            wc = W[c, ti, bl, p]; vc = V[c, ti, bl, p]
            (wc > 0 && isfinite(wc) && isfinite(vc)) || continue
            s += Float64(wc) * ComplexF64(vc); w += Float64(wc)
        end
        if w > 0
            oj1, oj2 = axis == 1 ? (1, j) : (j, 1)
            Vbar[oj1, oj2, bl, p] = s / w
            Wbar[oj1, oj2, bl, p] = w
        end
    end
    return Vbar, Wbar
end


DimensionalData.@dim AveragingTime "Time-averaging interval (s)"
DimensionalData.@dim AveragingBandwidth "Frequency-averaging width (Hz)"

"""
    coherence_report(group; timescales = nothing, bandwidths = nothing,
                     debias = false, marginalize = true) -> DimStack

The coherence factor η = |Σ w·V| / Σ(w·|V|) of `group`, one scan
(`groupby(ps, ByScan())`), against time-averaging interval and
frequency-averaging width. Call it on the data before and after a solution
to see how much coherence each averaging costs: a good solution keeps η ≈ 1
out to the whole scan and the whole spectral window. Layers:

- `time` — over `(Scan, AntennaPair, FeedPair, AveragingTime)` (s).
- `freq` — over `(Scan, AntennaPair, FeedPair, AveragingBandwidth)` (Hz);
  frequency bins stay within one spectral window.
- `time_pooled`, `freq_pooled` — the same over `(Scan, FeedPair, …)`, pooled
  over antenna pairs.

Each interval bins the data along its axis, sums coherently within every bin
and pools the bins, so η ≡ 1 at native resolution and no gain model is
needed. A cell with no usable data is `NaN`. `timescales` (s) and
`bandwidths` (Hz) replace the default sweeps, which double from the native
spacing until one interval spans the whole extent.

`debias` removes the thermal-noise bias, which otherwise pulls η below 1 at
coarse averaging; it takes the weights as inverse variances per real
component (`w = 1/Var(Re V)`). The debiased η pools unbiased per-bin powers and
takes one square root at the end; where the pooled power does not clear 3σ of
its null fluctuation it is `NaN`. `marginalize` measures each curve on data
coherently averaged over the other axis first (the time curve on per-integration
window averages, the frequency curve on per-channel scan averages), which
raises the per-sample SNR the debias needs.

Per-scan results combine with [`cat_scans`](@ref); pass the same `timescales`
and `bandwidths` to every scan so the intervals line up.
"""
function coherence_report(
        group::XRadio.ProcessingSet;
        timescales = nothing, bandwidths = nothing, debias::Bool = false, marginalize::Bool = true,
    )
    scan = _group_scan(group)
    members = collect(values(group))
    member_pairs = map(_member_station_pairs, members)
    stations, feeds = _cross_cell_labels(member_pairs, map(feed_pairs, members), DataGeometry(group))
    isempty(stations) && throw(ArgumentError("scan $scan holds no cross baselines"))
    times = [Float64.(collect(XRadio.times(ms))) for ms in members]
    freqs = [Float64.(collect(XRadio.frequencies(ms))) for ms in members]
    dts = isnothing(timescales) ? _sweep(times) : sort!(Float64.(collect(timescales)))
    dnus = isnothing(bandwidths) ? _sweep(freqs) : sort!(Float64.(collect(bandwidths)))

    cell = LinearIndices((length(stations), length(feeds)))
    pair_i = Dict(p => j for (j, p) in pairs(stations))
    feed_i = Dict(f => q for (q, f) in pairs(feeds))
    numT, numF = zeros(length(dts), length(cell)), zeros(length(dnus), length(cell))
    denT, denF, dvarT, dvarF = (zeros(length(cell)) for _ in 1:4)
    npts = zeros(Int, length(cell))
    for (ms, ps, ts, fs) in zip(members, member_pairs, times, freqs)
        V, W, F = map(_coherence_cube, _member_layers(ms))
        fp = feed_pairs(ms)
        blmap(p) = [haskey(pair_i, ps[bi]) ? cell[pair_i[ps[bi]], feed_i[fp[p, bi]]] : 0 for bi in axes(V, 3)]
        if marginalize
            Vt, Wt = _collapse_axis(V, W, F, 1)
            Vf, Wf = _collapse_axis(V, W, F, 2)
            for p in axes(V, 4)
                _coherence_accumulate!(
                    numT, zeros(1, length(cell)), denT, dvarT, npts, Vt, Wt, nothing, blmap(p), [p],
                    ts, [sum(fs) / length(fs)], dts, [1.0], debias,
                )
                _coherence_accumulate!(
                    zeros(1, length(cell)), numF, denF, dvarF, zeros(Int, length(cell)), Vf, Wf, nothing,
                    blmap(p), [p], [sum(ts) / length(ts)], fs, [1.0], dnus, debias,
                )
            end
        else
            for p in axes(V, 4)
                _coherence_accumulate!(numT, numF, denT, dvarT, npts, V, W, F, blmap(p), [p], ts, fs, dts, dnus, debias)
            end
        end
    end
    marginalize || (denF, dvarF = denT, dvarT)

    sd, pd, fd = _scan_dim([scan]), _station_pair_dim(stations), FeedPair(feeds)
    time, time_pooled = _feed_curves(numT, denT, dvarT, debias, (sd, pd, fd), AveragingTime(dts))
    freq, freq_pooled = _feed_curves(numF, denF, dvarF, debias, (sd, pd, fd), AveragingBandwidth(dnus))
    return DimStack((; time, freq, time_pooled, freq_pooled))
end

# A Measurement Set layer as a plain `(Frequency, Ti, BaselineID, Polarization)` array.
_coherence_cube(L) = PermutedDimsArray(parent(L), dimnum(L, (Frequency, Ti, BaselineID, Polarization)))

function _sweep(coords)
    diffs, spans = Float64[], Float64[]
    for x in coords
        length(x) > 1 || continue
        s = sort(x)
        append!(diffs, filter(>(0), diff(s)))
        push!(spans, last(s) - first(s))
    end
    return _auto_intervals(diffs, spans)
end

# The per-cell and per-feed-pair curves from the interval × cell sums, whose
# cells are `LinearIndices` over (antenna pair, feed pair).
function _feed_curves(num, den, dvar, power::Bool, (sd, pd, fd), xd)
    eta = fill(NaN, 1, length(pd), length(fd), length(xd))
    pooled = fill(NaN, 1, length(fd), length(xd))
    cell = LinearIndices((length(pd), length(fd)))
    for q in axes(cell, 2)
        cols = cell[:, q]
        e, agg = _curve_from_sums(num[:, cols], den[cols], dvar[cols], power)
        eta[1, :, q, :] = permutedims(e)
        pooled[1, q, :] = agg
    end
    return DimArray(eta, (sd, pd, fd, xd)), DimArray(pooled, (sd, fd, xd))
end
