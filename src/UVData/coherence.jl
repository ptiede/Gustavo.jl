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
# These helpers operate on ANY `UVSet` (raw, fringe-corrected, frequency- or
# time-averaged), so the same metric can be read at every pipeline stage by
# calling `coherence_report` on each stage's `UVSet`. `coherence_report` returns
# two curves — η versus time-averaging interval Δt and η versus frequency-
# averaging width Δν — plus the headline numbers at full-scan / full-band
# averaging. Pure (no Makie); the `plot_coherence` stub is implemented by
# `GustavoMakieExt`.
#
# CAVEAT: η is computed against the data's own finest resolution (a single
# integration / channel gives η ≡ 1), so it is self-normalized and needs no
# external gain model — but it is NOT thermal-noise-debiased. Pure thermal noise
# alone pulls η below 1 at coarse averaging (the incoherent |V| is noise-inflated
# while the coherent sum averages noise down). The diagnostic value is therefore
# the *shape* of the curve and the *before/after* (or stage-to-stage) comparison,
# not the absolute η: a correct solution tracks the thermal floor, a residual
# phase error drops faster.

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
an index, or product label(s). See [`CoherenceReport`](@ref) /
[`print_coherence_report`](@ref) / `plot_coherence`.
"""
function coherence_report(
        uvset::UVSet;
        timescales = nothing, bandwidths = nothing, pols = :parallel,
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
    den = zeros(Float64, nbl)
    npts = zeros(Int, nbl)

    # Pass 2 — materialize each leaf and accumulate. The denominator Σ w·|V| is
    # binning-independent, so it (and the cell count) is summed once per baseline.
    for leaf in values(src)
        m = materialize_leaf(leaf; layers = (:vis, :weights))
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
        _coherence_accumulate!(numT, numF, den, npts, V, W, blmap, plist, times_sec, freqs, dts, dnus)
    end

    etaT, aggT = _curve_from_sums(numT, den)
    etaF, aggF = _curve_from_sums(numF, den)
    return CoherenceReport(
        bl_pairs, ant_names, pol_labels, npts,
        CoherenceCurve(:time, dts, aggT, etaT),
        CoherenceCurve(:freq, dnus, aggF, etaF),
    )
end

# Per-baseline η = num/den and the pooled aggregate η = Σnum/Σden, per interval.
function _curve_from_sums(num::Matrix{Float64}, den::Vector{Float64})
    nrow, nbl = size(num)
    eta_bl = fill(NaN, nrow, nbl)
    agg = fill(NaN, nrow)
    for k in 1:nrow
        sn = 0.0; sd = 0.0
        for bl in 1:nbl
            d = den[bl]
            d > 0 || continue
            eta_bl[k, bl] = num[k, bl] / d
            sn += num[k, bl]; sd += d
        end
        sd > 0 && (agg[k] = sn / sd)
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
# ONCE (`tid`/`fid`) and each (c, ti) cell is read O(1) times w.r.t. the number of
# intervals: a single walk fans every cell into all nT (or nF) running
# accumulators. The denominator Σ w·|V| and `npts` are interval-independent, summed
# once. Intervals are used as given (the auto sweep nudges its terminal just above
# the span so the whole leaf lands in one bin).
function _coherence_accumulate!(
        numT::Matrix{Float64}, numF::Matrix{Float64}, den::Vector{Float64}, npts::Vector{Int},
        V::AbstractArray{Tv, 4}, W::AbstractArray{Tw, 4}, blmap::Vector{Int}, plist::Vector{Int},
        times_sec::Vector{Float64}, freqs::Vector{Float64}, dts::Vector{Float64}, dnus::Vector{Float64},
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
    accT = Vector{ComplexF64}(undef, nT); curT = Vector{Int}(undef, nT); haveT = Vector{Bool}(undef, nT)
    accF = Vector{ComplexF64}(undef, nF); curF = Vector{Int}(undef, nF); haveF = Vector{Bool}(undef, nF)

    @inbounds for bli in 1:nbl
        bl = blmap[bli]
        bl == 0 && continue
        for pli in eachindex(plist)
            p = plist[pli]
            p in 1:npol || continue

            # Denominator Σ w·|V| and cell count (interval-independent).
            for ti in 1:nti, c in 1:nchan
                w = W[c, ti, bli, p]; v = V[c, ti, bli, p]
                (w > 0 && isfinite(w) && isfinite(v)) || continue
                den[bl] += w * abs(ComplexF64(v))
                npts[bl] += 1
            end

            # Time-binned coherent amplitude, per channel. One ascending-time walk
            # fans each cell into all nT intervals.
            for c in 1:nchan
                for k in 1:nT
                    accT[k] = zero(ComplexF64); haveT[k] = false
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
                            numT[k, bl] += abs(accT[k]); accT[k] = zero(ComplexF64); curT[k] = id
                        end
                        accT[k] += wv
                    end
                end
                for k in 1:nT
                    haveT[k] && (numT[k, bl] += abs(accT[k]))
                end
            end

            # Frequency-binned coherent amplitude, per AP. One ascending-freq walk
            # fans each cell into all nF intervals.
            for ti in 1:nti
                for k in 1:nF
                    accF[k] = zero(ComplexF64); haveF[k] = false
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
                            numF[k, bl] += abs(accF[k]); accF[k] = zero(ComplexF64); curF[k] = id
                        end
                        accF[k] += wv
                    end
                end
                for k in 1:nF
                    haveF[k] && (numF[k, bl] += abs(accF[k]))
                end
            end
        end
    end
    return nothing
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
interval, colour = η ∈ [0, 1] (red = decorrelated, green = coherent). The direct
"which baseline/station is bad" view — a problem station shows as a band of red
rows. `axis = :time` (Δt columns) or `:freq` (Δν columns). Choose the columns by
passing `timescales` / `bandwidths` to [`coherence_report`](@ref) (e.g.
`timescales = [1, 5, 10, 30, 60]`). Provided by `GustavoMakieExt`.
"""
function plot_coherence_matrix end
