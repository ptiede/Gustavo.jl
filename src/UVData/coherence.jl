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
# These helpers operate on any `UVSet` (raw, fringe-corrected, frequency- or
# time-averaged), so the same metric can be read at every pipeline stage by
# calling `coherence_report` on each stage's `UVSet`. `coherence_report` returns
# two curves — η versus time-averaging interval Δt and η versus frequency-
# averaging width Δν — plus the headline numbers at full-scan / full-band
# averaging. Pure (no Makie); the `plot_coherence` stub is implemented by
# `GustavoMakieExt`.
#
# CAVEAT: η is computed against the data's own finest resolution (a single
# integration / channel gives η ≡ 1), so it is self-normalized and needs no
# external gain model. By default it is not thermal-noise-debiased: pure thermal
# noise alone pulls η below 1 at coarse averaging (the incoherent |V| is
# noise-inflated while the coherent sum averages noise down), so at native per-cell
# SNR the absolute η UNDERSTATES a good solution — read the *shape* and the
# *before/after* comparison, not the absolute value. Pass `debias = true` to
# remove that bias; the weights then set the RELATIVE cell weighting
# (`w ∝ 1/Var(Re V)`), while the absolute noise scale is estimated from the
# data itself (`_noise_scale`), so a mis-calibrated WEIGHT column does not
# corrupt η.
#
# The debiased estimator works in POWER, with a single square root at the end:
# per averaging bin, `|Σ w·V|² − 2αΣw` is an unbiased estimate of the bin's
# coherent signal power `|s̄|²(Σw)²` at any SNR (α the measured noise scale), so
# bins (and cells, and baselines) are pooled as `Σ (|Σ w·V|² − 2αΣw)/Σw` — no
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

"""
    CoherenceCurve

Coherence factor versus averaging interval along one axis (`:time` or `:freq`),
produced by [`coherence_report`](@ref).

- `axis` — `:time` (averaging the `Ti` axis) or `:freq` (the `Frequency` axis).
- `intervals` — the averaging intervals (seconds for `:time`, Hz for `:freq`),
  ascending from the data's native spacing to its full extent.
- `eta` — the aggregate coherence factor at each interval (weight-pooled over all
  baselines/products), `length(intervals)`.
- `eta_baseline` — per-baseline coherence, `(length(intervals), nbaseline)`; `NaN`
  for a baseline with no valid data.

`eta` starts at ≈1 (native resolution, one sample per cell) and decreases as the
interval grows if residual phase decorrelates the average.
"""
struct CoherenceCurve
    axis::Symbol
    intervals::Vector{Float64}
    eta::Vector{Float64}
    eta_baseline::Matrix{Float64}
end

"""
    CoherenceReport

Stage-agnostic coherence summary of a `UVSet`, from [`coherence_report`](@ref).
`bl_pairs`/`ant_names` label the baseline axis of the curves; `pol_products` are
the correlation products that were included (parallel-hand by default); `npts` is
the valid-cell count per baseline. `time` and `freq` are the [`CoherenceCurve`](@ref)s
versus Δt and Δν. The headline stage numbers are `time.eta[end]` (coherence
retained averaging the whole scan to one sample) and `freq.eta[end]` (averaging
the whole band to one channel) — see [`coherence_headline`](@ref).
"""
struct CoherenceReport
    bl_pairs::Vector{Tuple{Int, Int}}
    ant_names::Vector{String}
    pol_products::Vector{String}
    npts::Vector{Int}
    time::CoherenceCurve
    freq::CoherenceCurve
end

# Parallel-hand correlation product ("PP"/"QQ", or "RR"/"LL"): both feed letters
# equal. A pure-label test, kept local so this file needs no Calibration import.
_is_parallel_product(s::AbstractString) = length(s) == 2 && s[1] == s[2]

# Resolve a `pols` selector against this leaf's product labels → local indices.
function _select_coherence_pols(labels::Vector{String}, pols)
    if pols === :parallel
        idx = findall(_is_parallel_product, labels)
        return isempty(idx) ? collect(eachindex(labels)) : idx
    elseif pols === :all
        return collect(eachindex(labels))
    elseif pols isa Integer
        return [Int(pols)]
    elseif pols isa AbstractVector{<:Integer}
        return collect(Int.(pols))
    elseif pols isa Union{AbstractString, Symbol}
        i = findfirst(==(String(pols)), labels)
        return i === nothing ? error("coherence: product $(pols) not in $labels") : [i]
    elseif pols isa AbstractVector
        return [(j = findfirst(==(String(p)), labels); j === nothing ? error("coherence: product $p not in $labels") : j) for p in pols]
    else
        error("coherence: `pols` must be :parallel, :all, an index, or product label(s)")
    end
end

# Geometric sweep of averaging intervals from the native spacing to the full
# extent (largest per-leaf span), doubling each step. The finest interval is the
# *minimum* positive consecutive difference (not the median): with a uniform grid
# the two agree, but on a non-uniform grid the median can be larger than the
# closest pair, merging those two samples into one bin at the "native" step and
# breaking the η ≡ 1 anchor. Anchoring at the minimum step guarantees
# one-sample-per-bin (η ≡ 1) at native resolution regardless of grid uniformity.
# Empty when the axis has no spacing (single sample/channel everywhere) — that
# axis simply gets no coherence curve.
function _auto_intervals(diffs::Vector{Float64}, spans::Vector{Float64})
    isempty(diffs) && return Float64[]
    native = minimum(diffs)
    (native > 0 && isfinite(native)) || return Float64[]
    full = isempty(spans) ? native : maximum(spans)
    out = Float64[]
    v = native
    while v < full
        push!(out, v)
        v *= 2
    end
    # Terminal "average everything" interval. Bins are half-open [k·Δ, (k+1)·Δ),
    # so a sample sitting exactly at `full` (span an exact multiple of Δ — the
    # common uniform grid) would floor into a spurious singleton top bin; nudging
    # the terminal interval just above the span keeps the whole leaf in one bin.
    push!(out, full * (1 + 1.0e-9))
    return unique!(out)
end

"""
    coherence_report(uvset::UVSet; timescales = nothing, bandwidths = nothing,
                     pols = :parallel) -> CoherenceReport

Measure the per-baseline coherence factor η = |Σ w·V| / Σ(w·|V|) of `uvset` as a
function of time-averaging interval Δt and frequency-averaging width Δν, pooling
over the selected correlation products. Works on any `UVSet` — call it on the raw
data, the fringe/bandpass-corrected data, and each reduction stage to see how much
coherence each averaging step costs (the run is correct when averaging stays
coherent: η ≈ 1 out to long Δt / wide Δν).

For each interval the data is binned along that axis (within each leaf, per
channel for time / per integration for frequency), coherently summed inside every
bin, and the bin amplitudes pooled — so η is self-normalized (η ≡ 1 at the native
one-sample-per-bin resolution) and needs no gain model. Lazy leaves are
materialized one at a time (memory-safe on a streamed set).

`timescales` (seconds) / `bandwidths` (Hz) override the default geometric sweeps
(native spacing → full extent). `pols` selects products: `:parallel` (default,
parallel-hand only — cross hands are mostly noise and would bias η down), `:all`,
an index, or product label(s).

`debias` (default `false`) removes the thermal-noise bias from η. The
weights set the relative cell weighting; the absolute noise scale is
estimated per (baseline, product) from adjacent-sample differences, so a
mis-calibrated WEIGHT column does not corrupt η. Without `debias` the raw η
is pulled below 1 at coarse averaging by noise alone. The debiased estimator
pools unbiased per-bin power estimates `(|Σ w·V|² − 2αΣw)/Σw` and takes one
square root at the end, so η ≈ 1 for a flat-phase solution at any SNR. A
baseline (or pooled aggregate) whose signal power does not clear 3σ of its
null fluctuation reads `NaN`; a NaN baseline still enters the pooled
aggregate, since excluding it on the realized sign would bias the pool.

`marginalize` (default `true`) measures each curve on the data coherently
averaged over the other axis first: the time curve on the per-AP
band-average, the frequency curve on the per-channel time-average. This
boosts the per-sample SNR (so `debias` is reliable); at native per-cell
SNR ≲ 1 the unmarginalized curves measure noise and understate coherence.
See [`CoherenceReport`](@ref) / [`print_coherence_report`](@ref) /
`plot_coherence`.
"""
function coherence_report(
        uvset::UVSet;
        timescales = nothing, bandwidths = nothing, pols = :parallel,
        debias = false, marginalize = true,
    )
    src = DimensionalData.branches(uvset)
    isempty(src) && error("coherence_report: uvset has no leaves")

    # Pass 1 — metadata/dims only (eager even on lazy leaves): global baseline
    # list, antenna labels, included products, and the native/full extents that
    # set the default interval sweeps. Nothing is read from disk here.
    blidx = OrderedDict{Tuple{Int, Int}, Int}()
    bl_pairs = Tuple{Int, Int}[]
    ant_names = String[]
    pol_labels = String[]
    tdiffs = Float64[]; tspans = Float64[]
    fdiffs = Float64[]; fspans = Float64[]
    for leaf in values(src)
        for (a, b) in baselines(leaf).pairs
            a == b && continue
            key = (a, b)
            if !haskey(blidx, key)
                push!(bl_pairs, key)
                blidx[key] = length(bl_pairs)
            end
        end
        isempty(ant_names) && (ant_names = String.(collect(metadata(leaf).antennas.name)))
        if isempty(pol_labels)
            labels = String.(pol_products(leaf))
            pol_labels = labels[_select_coherence_pols(labels, pols)]
        end
        ts = sort(Float64.(lookup(leaf[:vis], Ti))) .* 3600.0   # hours → seconds
        if length(ts) > 1
            append!(tdiffs, filter(>(0), diff(ts)))
            push!(tspans, last(ts) - first(ts))
        end
        fs = sort(Float64.(lookup(leaf[:vis], Frequency)))
        if length(fs) > 1
            append!(fdiffs, filter(>(0), diff(fs)))
            push!(fspans, last(fs) - first(fs))
        end
    end
    isempty(bl_pairs) && error("coherence_report: uvset has no cross baselines")

    dts = timescales === nothing ? _auto_intervals(tdiffs, tspans) : sort(Float64.(collect(timescales)))
    dnus = bandwidths === nothing ? _auto_intervals(fdiffs, fspans) : sort(Float64.(collect(bandwidths)))
    nbl = length(bl_pairs); nT = length(dts); nF = length(dnus)

    numT = zeros(Float64, nT, nbl)
    numF = zeros(Float64, nF, nbl)
    # In `marginalize` mode the two curves use DIFFERENT denominators (the time curve
    # references the per-AP band-averaged amplitude, the freq curve the per-channel
    # time-averaged amplitude), so keep them separate; non-marginalized shares one.
    denT = zeros(Float64, nbl)
    denF = zeros(Float64, nbl)
    # Null variance of each debiased denominator (Σ 4α² over its cells): the
    # scale against which a denominator counts as a DETECTION of signal power.
    dvarT = zeros(Float64, nbl)
    dvarF = zeros(Float64, nbl)
    npts = zeros(Int, nbl)
    dumF = zeros(Float64, 1, nbl)
    dumT = zeros(Float64, 1, nbl)
    dumN = zeros(Int, nbl)

    # Pass 2 — materialize each leaf and accumulate.
    for leaf in values(src)
        m = materialize_leaf(leaf)
        V = parent(m[:vis]); W = parent(m[:weights])
        labels = String.(pol_products(m))
        plist = _select_coherence_pols(labels, pols)
        # Pass 1 captured the first leaf's labels/antennas, but pass 2 pools all
        # leaves into one global baseline index — heterogeneous leaves would be
        # silently mislabelled, so guard instead of trusting the first leaf.
        leaf_labels = labels[plist]
        leaf_labels == pol_labels || error(
            "coherence_report: heterogeneous correlation products across leaves " *
                "($(leaf_labels) vs $(pol_labels)); cannot pool into one report",
        )
        leaf_ants = String.(collect(metadata(m).antennas.name))
        leaf_ants == ant_names || error(
            "coherence_report: heterogeneous antenna names across leaves; " *
                "cannot pool into one report",
        )
        blmap = [get(blidx, p, 0) for p in baselines(m).pairs]
        times_sec = Float64.(lookup(m[:vis], Ti)) .* 3600.0
        freqs = Float64.(lookup(m[:vis], Frequency))
        # Per-(baseline, product) noise scale α, from adjacent-sample differences
        # of the NATIVE cube (signal cancels in the difference; noise does not),
        # so the debias trusts the weights only for RELATIVE cell weighting. α is
        # a property of the weights against the true noise and is preserved
        # exactly by the weighted averaging `_collapse_axis` performs.
        alpha = debias ? _noise_scale(V, W, plist) : ones(Float64, size(V, 3), size(V, 4))
        if marginalize
            # Coherently average the other axis first (incoherent/segmented style), so
            # each curve is measured on high-SNR samples and the debias is reliable.
            f0m = isempty(freqs) ? 0.0 : sum(freqs) / length(freqs)
            t0m = isempty(times_sec) ? 0.0 : sum(times_sec) / length(times_sec)
            Vt, Wt = _collapse_axis(V, W, 1)        # band-average per AP → time curve
            _coherence_accumulate!(numT, dumF, denT, dvarT, npts, Vt, Wt, blmap, plist, times_sec, [f0m], dts, [1.0], debias, alpha)
            Vf, Wf = _collapse_axis(V, W, 2)        # time-average per channel → freq curve
            _coherence_accumulate!(dumT, numF, denF, dvarF, dumN, Vf, Wf, blmap, plist, [t0m], freqs, [1.0], dnus, debias, alpha)
        else
            _coherence_accumulate!(numT, numF, denT, dvarT, npts, V, W, blmap, plist, times_sec, freqs, dts, dnus, debias, alpha)
        end
    end

    etaT, aggT = _curve_from_sums(numT, denT, dvarT, debias)
    etaF, aggF = _curve_from_sums(numF, marginalize ? denF : denT, marginalize ? dvarF : dvarT, debias)
    return CoherenceReport(
        bl_pairs, ant_names, pol_labels, npts,
        CoherenceCurve(:time, dts, aggT, etaT),
        CoherenceCurve(:freq, dnus, aggF, etaF),
    )
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
# power sum (Σ 4α² over its cells). Below that there is no coherence
# measurement — the ratio of two near-zero fluctuations would clamp into
# [0, 1] and read as a confident number — so the entry is NaN instead.
function _curve_from_sums(
        num::Matrix{Float64}, den::Vector{Float64}, denvar::Vector{Float64}, power::Bool,
    )
    nrow, nbl = size(num)
    eta_bl = fill(NaN, nrow, nbl)
    agg = fill(NaN, nrow)
    for k in 1:nrow
        sn = 0.0; sd = 0.0; sv = 0.0
        for bl in 1:nbl
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

# Accumulate one leaf's contribution. `V`/`W` are `(Frequency, Ti, Baseline, Pol)`;
# `blmap[bli]` maps a local baseline to its global index (0 = skip: autocorr or
# unmapped). A function barrier so the hot loops specialize on the concrete eltypes
# of `parent(leaf[...])` (type-unstable at the call site, as in the reducers).
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
# once. Intervals are used as given (the auto sweep nudges its terminal just above
# the span so the whole leaf lands in one bin).
function _coherence_accumulate!(
        numT::Matrix{Float64}, numF::Matrix{Float64}, den::Vector{Float64}, dvar::Vector{Float64},
        npts::Vector{Int},
        V::AbstractArray{Tv, 4}, W::AbstractArray{Tw, 4}, blmap::Vector{Int}, plist::Vector{Int},
        times_sec::Vector{Float64}, freqs::Vector{Float64}, dts::Vector{Float64}, dnus::Vector{Float64},
        debias::Bool, alpha::Matrix{Float64},
    ) where {Tv, Tw}
    nchan, nti, nbl, npol = size(V)
    nT = length(dts); nF = length(dnus)
    tperm = sortperm(times_sec)
    cperm = sortperm(freqs)
    t0 = isempty(times_sec) ? 0.0 : times_sec[tperm[1]]
    f0 = isempty(freqs) ? 0.0 : freqs[cperm[1]]

    # Bin-id tables (sample × interval), indexed by the ORIGINAL sample index.
    tid = Matrix{Int}(undef, nti, nT)
    @inbounds for k in 1:nT
        dt = dts[k]
        for ti in 1:nti
            tid[ti, k] = dt > 0 ? floor(Int, (times_sec[ti] - t0) / dt) : 0
        end
    end
    fid = Matrix{Int}(undef, nchan, nF)
    @inbounds for k in 1:nF
        dnu = dnus[k]
        for c in 1:nchan
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
    binval(s::ComplexF64, sw::Float64, a::Float64) = debias ? (abs2(s) - 2 * a * sw) / sw : abs(s)

    @inbounds for bli in 1:nbl
        bl = blmap[bli]
        bl == 0 && continue
        for pli in eachindex(plist)
            p = plist[pli]
            p in 1:npol || continue
            a = alpha[bli, p]

            # Denominator (interval-independent). Without `debias`, the incoherent
            # Σ w·|V|. With it, the numerator's own expression at NATIVE binning —
            # the per-cell unbiased power `(w²|V|² − 2w)/w = w|V|² − 2` — so η ≡ 1
            # at native resolution stays an identity and the ratio's denominator
            # is unbiased at any SNR (a per-cell clipped amplitude would sit
            # noise-inflated above the true signal for weak cells, holding η
            # below 1 even after the numerator's bins reach high SNR).
            for ti in 1:nti, c in 1:nchan
                w = W[c, ti, bli, p]; v = V[c, ti, bli, p]
                (w > 0 && isfinite(w) && isfinite(v)) || continue
                a2 = abs2(ComplexF64(v))
                if debias
                    den[bl] += Float64(w) * a2 - 2.0 * a
                    dvar[bl] += 4.0 * a^2
                else
                    den[bl] += Float64(w) * sqrt(a2)
                end
                npts[bl] += 1
            end

            # Time-binned coherent amplitude, per channel. One ascending-time walk
            # fans each cell into all nT intervals.
            for c in 1:nchan
                for k in 1:nT
                    accT[k] = zero(ComplexF64); swT[k] = 0.0; haveT[k] = false
                end
                for ti in tperm
                    w = W[c, ti, bli, p]; v = V[c, ti, bli, p]
                    (w > 0 && isfinite(w) && isfinite(v)) || continue
                    wv = w * ComplexF64(v)
                    for k in 1:nT
                        id = tid[ti, k]
                        if !haveT[k]
                            curT[k] = id; haveT[k] = true
                        elseif id != curT[k]
                            numT[k, bl] += binval(accT[k], swT[k], a)
                            accT[k] = zero(ComplexF64); swT[k] = 0.0; curT[k] = id
                        end
                        accT[k] += wv; swT[k] += w
                    end
                end
                for k in 1:nT
                    haveT[k] && (numT[k, bl] += binval(accT[k], swT[k], a))
                end
            end

            # Frequency-binned coherent amplitude, per AP. One ascending-freq walk
            # fans each cell into all nF intervals.
            for ti in 1:nti
                for k in 1:nF
                    accF[k] = zero(ComplexF64); swF[k] = 0.0; haveF[k] = false
                end
                for c in cperm
                    w = W[c, ti, bli, p]; v = V[c, ti, bli, p]
                    (w > 0 && isfinite(w) && isfinite(v)) || continue
                    wv = w * ComplexF64(v)
                    for k in 1:nF
                        id = fid[c, k]
                        if !haveF[k]
                            curF[k] = id; haveF[k] = true
                        elseif id != curF[k]
                            numF[k, bl] += binval(accF[k], swF[k], a)
                            accF[k] = zero(ComplexF64); swF[k] = 0.0; curF[k] = id
                        end
                        accF[k] += wv; swF[k] += w
                    end
                end
                for k in 1:nF
                    haveF[k] && (numF[k, bl] += binval(accF[k], swF[k], a))
                end
            end
        end
    end
    return nothing
end

# Per-(baseline, product) noise scale α = w·Var(Re V), from the data itself.
#
# The debias terms need the ABSOLUTE per-cell noise power, and the weights only
# promise it up to calibration: a WEIGHT column mis-scaled by 10% in σ is a 21%
# error in noise power, which — summed over ~10⁵ near-zero-signal cells — can
# exceed a faint source's entire measured power and drive the pooled
# denominator negative. Adjacent-sample differences measure the true noise
# independently of that calibration: the (continuum, or slowly-varying) signal
# cancels in `V₁ − V₂` while the noise adds, so
# `E|V₁ − V₂|² = 2α(1/w₁ + 1/w₂)`. α is estimated per (baseline, product) — a
# per-station weight miscalibration factorizes onto baselines — as a MEDIAN
# (robust to outliers; |ΔV|² is exponential under the null, so the median is
# `ln 2` of the mean). Adjacent channels are differenced first; a
# single-channel axis (already band-averaged data) falls back to adjacent
# integrations. Fewer than 32 usable pairs leaves α = 1 (trust the weights).
#
# Residual signal leakage into the difference (a delay slope across adjacent
# channels of RAW data) inflates α slightly at high SNR, where the debias is
# negligible anyway; on corrected data the signal is flat and cancels exactly.
function _noise_scale(V::AbstractArray{Tv, 4}, W::AbstractArray{Tw, 4}, plist::Vector{Int}) where {Tv, Tw}
    nchan, nti, nbl, npol = size(V)
    alpha = ones(Float64, nbl, npol)
    buf = Float64[]
    @inbounds for bli in 1:nbl, p in plist
        p in 1:npol || continue
        empty!(buf)
        if nchan > 1
            for ti in 1:nti, c in 1:(nchan - 1)
                w1 = W[c, ti, bli, p]; w2 = W[c + 1, ti, bli, p]
                v1 = V[c, ti, bli, p]; v2 = V[c + 1, ti, bli, p]
                (w1 > 0 && w2 > 0 && isfinite(w1) && isfinite(w2) && isfinite(v1) && isfinite(v2)) || continue
                push!(
                    buf, abs2(ComplexF64(v1) - ComplexF64(v2)) /
                        (2 * (inv(Float64(w1)) + inv(Float64(w2))))
                )
            end
        end
        if length(buf) < 32 && nti > 1
            empty!(buf)
            for c in 1:nchan, ti in 1:(nti - 1)
                w1 = W[c, ti, bli, p]; w2 = W[c, ti + 1, bli, p]
                v1 = V[c, ti, bli, p]; v2 = V[c, ti + 1, bli, p]
                (w1 > 0 && w2 > 0 && isfinite(w1) && isfinite(w2) && isfinite(v1) && isfinite(v2)) || continue
                push!(
                    buf, abs2(ComplexF64(v1) - ComplexF64(v2)) /
                        (2 * (inv(Float64(w1)) + inv(Float64(w2))))
                )
            end
        end
        length(buf) >= 32 || continue
        alpha[bli, p] = median(buf) / log(2)
    end
    return alpha
end

# Coherently weighted-average `V` (with weights `W`) over `axis` (1 = Frequency,
# 2 = Ti), returning `(Vbar, Wbar)` with that axis collapsed to length 1: `Vbar` is
# the weighted mean `Σ w·V / Σ w` and `Wbar = Σ w` (so its noise variance is `1/Wbar`,
# preserving the inverse-variance convention the debias relies on). For corrected
# data this is the high-SNR band-average (axis 1) or time-average (axis 2) used by
# `coherence_report(marginalize = true)`; cells with no weight become `NaN`.
function _collapse_axis(V::AbstractArray{<:Any, 4}, W::AbstractArray{<:Any, 4}, axis::Int)
    nchan, nti, nbl, npol = size(V)
    on, no = axis == 1 ? (nchan, nti) : (nti, nchan)
    Vbar = fill(ComplexF64(NaN), axis == 1 ? 1 : nchan, axis == 1 ? nti : 1, nbl, npol)
    Wbar = zeros(Float64, axis == 1 ? 1 : nchan, axis == 1 ? nti : 1, nbl, npol)
    @inbounds for p in 1:npol, bl in 1:nbl, j in 1:no
        s = zero(ComplexF64); w = 0.0
        for i in 1:on
            c, ti = axis == 1 ? (i, j) : (j, i)
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

"""
    coherence_headline(report::CoherenceReport) -> NamedTuple

The two stage-summary numbers: `(; eta_time, eta_freq, loss_time, loss_freq)`,
where `eta_time` is the aggregate coherence retained averaging the whole scan to
one sample (`report.time.eta[end]`), `eta_freq` averaging the whole band to one
channel, and `loss_* = 1 − eta_*`. `NaN` if an axis had no averaging to do.
"""
function coherence_headline(report::CoherenceReport)
    et = isempty(report.time.eta) ? NaN : last(report.time.eta)
    ef = isempty(report.freq.eta) ? NaN : last(report.freq.eta)
    return (; eta_time = et, eta_freq = ef, loss_time = 1 - et, loss_freq = 1 - ef)
end

_cohfmt(x::Real) = isfinite(x) ? string(round(x; digits = 4)) : "NaN"

"""
    print_coherence_report(report::CoherenceReport; io = stdout, nworst = 5)

Pretty-print a [`CoherenceReport`](@ref): the aggregate η versus Δt and Δν, the
headline full-scan / full-band coherence (and loss), and the `nworst` least-
coherent baselines at full averaging.
"""
function print_coherence_report(report::CoherenceReport; io = stdout, nworst::Integer = 5)
    println(io)
    println(io, "Coherence report [", join(report.pol_products, ","), "], ", length(report.bl_pairs), " baselines")

    _print_curve(io, "time-averaging", report.time, "Δt", 1.0, "s")
    _print_curve(io, "frequency-averaging", report.freq, "Δν", 1.0e-6, "MHz")

    h = coherence_headline(report)
    println(io, "  headline:")
    println(io, "    full-scan  η = ", _cohfmt(h.eta_time), "   (loss ", _cohpct(h.loss_time), ")")
    println(io, "    full-band  η = ", _cohfmt(h.eta_freq), "   (loss ", _cohpct(h.loss_freq), ")")

    _print_worst(io, report, nworst)
    return nothing
end

_cohpct(x::Real) = isfinite(x) ? string(round(100x; digits = 2), "%") : "NaN"

function _print_curve(io, title, curve::CoherenceCurve, xlabel, scale, unit)
    if isempty(curve.intervals)
        println(io, "  ", title, ": (single sample — no averaging)")
        return nothing
    end
    println(io, "  ", title, ":")
    println(io, "    ", rpad(string(xlabel, " (", unit, ")"), 14), "η")
    for k in eachindex(curve.intervals)
        println(io, "    ", rpad(_cohfmt(curve.intervals[k] * scale), 14), _cohfmt(curve.eta[k]))
    end
    return nothing
end

# The "na–nb" station-pair code for baseline `bi`, with an `ant{i}` fallback when
# an antenna index is past the name table. Non-exported; reused by sibling tools.
function coherence_baseline_label(report::CoherenceReport, bi::Integer)::String
    a, b = report.bl_pairs[bi]
    na = a <= length(report.ant_names) ? report.ant_names[a] : string("ant", a)
    nb = b <= length(report.ant_names) ? report.ant_names[b] : string("ant", b)
    return string(na, "–", nb)
end

# Indices of the `n` least-coherent baselines at the coarsest swept interval,
# ranked on the time curve (falling back to the freq curve when there is no time
# curve), NaN excluded. Non-exported; reused by sibling tools.
function worst_baselines(report::CoherenceReport, n::Integer; axis::Symbol = :time)::Vector{Int}
    curve = axis === :freq ? report.freq : report.time
    isempty(curve.eta) && (curve = axis === :freq ? report.time : report.freq)
    isempty(curve.eta) && return Int[]
    k = lastindex(curve.eta)
    rows = [(bi, curve.eta_baseline[k, bi]) for bi in eachindex(report.bl_pairs)]
    finite = filter(r -> isfinite(r[2]), rows)
    isempty(finite) && return Int[]
    sort!(finite; by = r -> r[2])
    m = min(n, length(finite))
    return [finite[i][1] for i in 1:m]
end

# The least-coherent baselines at full time-averaging (the headline-worst), shown
# only when there is a time curve to rank by.
function _print_worst(io, report::CoherenceReport, nworst::Integer)
    (nworst <= 0 || isempty(report.time.eta)) && return nothing
    worst = worst_baselines(report, nworst; axis = :time)
    isempty(worst) && return nothing
    k = lastindex(report.time.eta)
    println(io, "  least coherent (full-scan):")
    for bi in worst
        e = report.time.eta_baseline[k, bi]
        println(io, "    ", rpad(coherence_baseline_label(report, bi), 12), "η = ", _cohfmt(e))
    end
    return nothing
end

# ── Plot stub — implemented by `GustavoMakieExt`. ─────────────────────────────
"""
    plot_coherence(report::CoherenceReport; baselines = :all)
    plot_coherence(parent, report::CoherenceReport; baselines = :all)

Coherence factor η versus time-averaging interval Δt and frequency-averaging width
Δν: the aggregate as a bold line plus faint per-baseline traces. A correct
solution stays near η = 1 across the swept intervals. Provided by `GustavoMakieExt`
(load Makie/CairoMakie). See [`coherence_report`](@ref).
"""
function plot_coherence end

"""
    plot_coherence_matrix(report::CoherenceReport; axis = :time, sortworst = true)
    plot_coherence_matrix(parent, report::CoherenceReport; ...)

Heatmap of the per-baseline coherence factor η: one row per baseline (labelled by
station-pair code, sorted worst-first by default), one column per averaging
interval, colour = η ∈ `[0, 1]` (red = decorrelated, green = coherent). The direct
"which baseline/station is bad" view — a problem station shows as a band of red
rows. `axis = :time` (Δt columns) or `:freq` (Δν columns). Choose the columns by
passing `timescales` / `bandwidths` to [`coherence_report`](@ref) (e.g.
`timescales = [1, 5, 10, 30, 60]`). Provided by `GustavoMakieExt`.
"""
function plot_coherence_matrix end
