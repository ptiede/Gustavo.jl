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
# A `CoherenceReport` holds two curves — η versus time-averaging interval Δt
# and η versus frequency-averaging width Δν — plus the headline numbers at
# full-scan / full-band averaging. The accumulation kernels below work on
# `(Frequency, Ti, BaselineID, Polarization)` cubes. Pure (no Makie); the
# `plot_coherence` stub is implemented by `GustavoMakieExt`.
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

"""
    CoherenceCurve

Coherence factor versus averaging interval along one axis (`:time` or `:freq`),
held by a [`CoherenceReport`](@ref).

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

Stage-agnostic coherence summary of a visibility set.
`bl_pairs`/`ant_names` label the baseline axis of the curves; `feeds` are the
feed pairs of the correlation products that were included; `npts` is
the valid-cell count per baseline. `time` and `freq` are the [`CoherenceCurve`](@ref)s
versus Δt and Δν. The headline stage numbers are `time.eta[end]` (coherence
retained averaging the whole scan to one sample) and `freq.eta[end]` (averaging
the whole band to one channel) — see [`coherence_headline`](@ref).
"""
struct CoherenceReport
    bl_pairs::Vector{Tuple{Int, Int}}
    ant_names::Vector{String}
    feeds::Vector{Tuple{Int, Int}}
    npts::Vector{Int}
    time::CoherenceCurve
    freq::CoherenceCurve
end

# Resolve a `pols` selector against this leaf's feed pairs → local indices.
function _select_coherence_pols(feeds::Vector{Tuple{Int, Int}}, pols)
    if pols === :all
        return collect(eachindex(feeds))
    elseif pols isa Integer
        return [Int(pols)]
    elseif pols isa Tuple{Integer, Integer}
        return [_pol_index_lookup(feeds, pols)]
    elseif pols isa AbstractVector && all(p -> p isa Union{Integer, Tuple{Integer, Integer}}, pols)
        return [p isa Integer ? Int(p) : _pol_index_lookup(feeds, p) for p in pols]
    else
        throw(ArgumentError("coherence: `pols` must be :all, an index, a feed pair such as (1, 1), or a vector of these; got $(repr(pols))"))
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

# Accumulate one leaf's contribution. `V`/`W` are `(Frequency, Ti, BaselineID, Polarization)`;
# `blmap[bli]` maps a local baseline to its global index (0 = skip: autocorr or
# unmapped). A function barrier so the hot loops specialize on the concrete eltypes
# of `parent(leaf[...])`.
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
# `Fl` is the flag layer over `V`, or `nothing` for an already-collapsed cube:
# `_collapse_axis` has excluded the flagged samples, so the reduced cells carry
# no flag of their own.
@inline _flagged(::Nothing, I...) = false
Base.@propagate_inbounds _flagged(Fl, I...) = Fl[I...]

function _coherence_accumulate!(
        numT::Matrix{Float64}, numF::Matrix{Float64}, den::Vector{Float64}, dvar::Vector{Float64},
        npts::Vector{Int},
        V::AbstractArray{Tv, 4}, W::AbstractArray{Tw, 4}, Fl, blmap::Vector{Int}, plist::Vector{Int},
        times_sec::Vector{Float64}, freqs::Vector{Float64}, dts::Vector{Float64}, dnus::Vector{Float64},
        debias::Bool,
    ) where {Tv, Tw}
    Fl === nothing || check_layer_axes(V, W, Fl)
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
    println(io, "Coherence report [", join(report.feeds, ","), "], ", length(report.bl_pairs), " baselines")

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
(load Makie/CairoMakie).
"""
function plot_coherence end

"""
    plot_coherence_matrix(report::CoherenceReport; axis = :time, sortworst = true)
    plot_coherence_matrix(parent, report::CoherenceReport; ...)

Heatmap of the per-baseline coherence factor η: one row per baseline (labelled by
station-pair code, sorted worst-first by default), one column per averaging
interval, colour = η ∈ `[0, 1]` (red = decorrelated, green = coherent). The direct
"which baseline/station is bad" view — a problem station shows as a band of red
rows. `axis = :time` (Δt columns) or `:freq` (Δν columns); the columns are the
report's own intervals. Provided by `GustavoMakieExt`.
"""
function plot_coherence_matrix end
