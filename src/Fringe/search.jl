# ── Per-baseline FFT delay/rate fringe search ────────────────────────────────
#
# For one (baseline, correlation product, scan) block of visibilities the fringe
# model is
#
#     V(f, t) ≈ A · exp(i[φ + 2π·τ·(f − f0) + 2π·ṙ·(t − t0)])
#
# with a station-pair delay τ (s), fringe rate ṙ (Hz), constant phase φ (rad).
# The matched filter
#
#     D(τ, ṙ) = Σ_{f,t} w(f,t)·V(f,t)·exp(−2πi[τ·(f − f0) + ṙ·(t − t0)])
#
# is a 2-D DFT once the (weighted) visibilities are placed on a uniform
# frequency × time grid: delay is conjugate to frequency, rate to time. We grid,
# zero-pad (oversample), FFT, take the windowed peak of |D|, refine each axis by
# 3-point quadratic interpolation, and read φ off the complex peak.
#
# Conventions: weights are inverse variances (1/σ²); at the matched point
# |D| ≈ A·Σw and the noise on D has variance Σw, so SNR = |D_peak| / √(Σw) and
# amplitude A = |D_peak| / Σw. Multi-band (gapped) frequency axes are gridded
# onto a common Δf grid with empty bins zero-filled. NOTE: for narrow bands that
# are widely separated, the multi-band matched filter has near-equal-height alias
# peaks at the inverse band-spacing, and the FFT can lock onto an alias unless the
# delay window is narrower than the alias spacing or `oversample` is large — so
# multi-band delay is only as unambiguous as the a-priori `delay_window` allows.
#
# For VGOS-style layouts (narrow bands spread over a huge span) the common-Δf
# grid is almost entirely zeros and the single big FFT is prohibitively slow; a
# hierarchical single-band → multi-band delay (SBD/MBD) search — HOPS/fourfit's
# decomposition — replaces it, selected by `FringeSearch.algorithm` (see the
# "Hierarchical SBD → MBD search" section below).

"""
    FringeSearch(; delay_window, rate_window, oversample, snr_min, quad_interp, algorithm)

Options for [`baseline_fringe_search`](@ref).

- `delay_window`  : `(lo, hi)` delay search window in seconds. Default ±1 µs.
- `rate_window`   : `(lo, hi)` fringe-rate search window in Hz. Default ±50 mHz.
- `oversample`    : zero-padding factor per axis (finer delay/rate grid). Default 8.
  Widely-separated narrow bands may need a larger value (or a tight
  `delay_window`) to avoid locking onto a multi-band alias peak.
- `snr_min`       : detection threshold; `valid = snr ≥ snr_min`. Default 6.
- `quad_interp`   : refine the peak by 3-point quadratic interpolation. Default true.
- `algorithm`     : `:auto` (default), `:full`, or `:mbd`. `:full` is the single
  brute-force FFT over the common-Δf grid spanning the whole frequency axis;
  `:mbd` is the hierarchical single-band → multi-band delay search (HOPS/fourfit
  style), which is far cheaper when narrow bands are spread over a wide span
  (VGOS-style) and resolves the multi-band delay ambiguity explicitly against
  the in-band delay. `:auto` picks `:mbd` when the axis has ≥ 2 band blocks and
  the common grid would be > 4× the real channel count (mostly zero-padding),
  else `:full` — contiguous-band data (e.g. VLBA) always takes the `:full` path.
"""
Base.@kwdef struct FringeSearch
    delay_window::Tuple{Float64, Float64} = (-1.0e-6, 1.0e-6)
    rate_window::Tuple{Float64, Float64} = (-0.05, 0.05)
    oversample::Int = 8
    snr_min::Float64 = 6.0
    quad_interp::Bool = true
    algorithm::Symbol = :auto
end

"""
    FringeDetection(delay, rate, phase, amp, snr, valid)

Result of a single-baseline fringe search. `delay` (s) and `rate` (Hz) are the
station-pair group delay and fringe rate; `phase` (rad) is the constant phase φ
referenced to `(f0, t0)`; `amp` is the coherent amplitude; `snr` the detection
signal-to-noise; `valid = snr ≥ snr_min`.
"""
struct FringeDetection
    delay::Float64
    rate::Float64
    phase::Float64
    amp::Float64
    snr::Float64
    valid::Bool
end

const _INVALID_DETECTION = FringeDetection(0.0, 0.0, 0.0, 0.0, 0.0, false)

"""
    FringeWorkspace()

Reusable scratch for [`baseline_fringe_search`](@ref): the zero-padded gridding
buffer `G`, the FFT output `D`, and a cached FFT plan, lazily (re)allocated when
the padded grid size changes. Pass one per thread to avoid allocating the grid
(tens of MB at `oversample = 8`) on every call — the search runs thousands of
times per solve, so reuse removes essentially all of its allocation/GC churn.
"""
mutable struct FringeWorkspace
    nf::Int
    nt::Int
    G::Matrix{ComplexF64}
    D::Matrix{ComplexF64}
    plan::Any
    dwin::Vector{Float64}      # scratch for the windowed |D|² noise estimate
    mbd::Any                   # lazily-built `_MBDWorkspace` for the hierarchical path
end
FringeWorkspace() = FringeWorkspace(0, 0, Matrix{ComplexF64}(undef, 0, 0), Matrix{ComplexF64}(undef, 0, 0), nothing, Float64[], nothing)

# Ensure `ws` is sized for an `nf × nt` grid (reallocating + replanning only when
# the size changes). `nothing` makes a fresh workspace (single-call fallback).
_ensure_workspace!(::Nothing, nf::Integer, nt::Integer) = _ensure_workspace!(FringeWorkspace(), nf, nt)
function _ensure_workspace!(ws::FringeWorkspace, nf::Integer, nt::Integer)
    if ws.nf != nf || ws.nt != nt || ws.plan === nothing
        ws.G = zeros(ComplexF64, nf, nt)
        ws.D = similar(ws.G)
        # FFT is ~96% of the search cost, so plan with FFTW.MEASURE (≈1.8× faster
        # transforms than the default ESTIMATE) and let the plan pick up the
        # process-wide FFTW thread count set by the solve. The plan is built once
        # per (thread-local) workspace per grid size and reused thousands of times,
        # so MEASURE's one-off planning cost is amortized to nothing; MEASURE may
        # scribble on `G`, but every search `fill!`s `G` before gridding. Results
        # are bit-identical to ESTIMATE — only the algorithm/speed differs.
        ws.plan = plan_fft(ws.G; flags = MEASURE)
        ws.nf = Int(nf)
        ws.nt = Int(nt)
    end
    return ws
end

# Uniform-grid descriptor for one axis: the origin, spacing, and grid length such
# that every sample `x` lands at `round((x - origin)/Δ) + 1 ∈ 1:n`. Δ is the
# median adjacent spacing (robust to gaps); n spans min→max. A length-1 axis is
# degenerate (Δ = 1, n = 1) — its conjugate (delay or rate) is not searched.
function _uniform_axis(x::AbstractVector)
    n = length(x)
    n <= 1 && return (origin = (isempty(x) ? 0.0 : float(first(x))), step = 1.0, n = 1, degenerate = true)
    xs = sort(collect(float.(x)))
    diffs = diff(xs)
    Δ = median(diffs)
    Δ = Δ > 0 ? Δ : 1.0
    span = xs[end] - xs[1]
    ngrid = round(Int, span / Δ) + 1
    return (origin = xs[1], step = Δ, n = ngrid, degenerate = false)
end

# Next FFT-friendly size ≥ m (products of small primes are fast in FFTW).
_fast_fft_size(m::Integer) = nextprod((2, 3, 5, 7), max(1, m))

# A `_uniform_axis` descriptor (concrete NamedTuple type, so `_SearchAxes` fields
# stay type-stable).
const _Axis = @NamedTuple{origin::Float64, step::Float64, n::Int, degenerate::Bool}

# Per-(scan group) search-grid geometry: the frequency/time uniform-axis
# descriptors, the padded FFT sizes, and the conjugate delay/rate coordinate
# vectors. These depend ONLY on (freqs, times, oversample) — identical across every
# (baseline, product) of a group — so the search builds them ONCE per group via
# `_search_axes` instead of re-sorting the freq/time axes (an O(nchan log nchan)
# sort of the same 1024 channels) and re-`collect`ing the two fftfreq vectors on
# each of the group's nbl×npol calls.
struct _SearchAxes{M}
    fax::_Axis
    tax::_Axis
    nf_pad::Int
    nt_pad::Int
    delays::Vector{Float64}
    rates::Vector{Float64}
    mbd::M                     # `_MBDAxes` for the hierarchical path, else `nothing`
end

function _search_axes(freqs::AbstractVector, times::AbstractVector, opts::FringeSearch)
    fax = _uniform_axis(freqs)
    tax = _uniform_axis(times)
    nf_pad = fax.degenerate ? 1 : _fast_fft_size(opts.oversample * fax.n)
    nt_pad = tax.degenerate ? 1 : _fast_fft_size(opts.oversample * tax.n)
    delays = fax.degenerate ? [0.0] : collect(fftfreq(nf_pad, 1.0 / fax.step))
    rates = tax.degenerate ? [0.0] : collect(fftfreq(nt_pad, 1.0 / tax.step))
    mbd = _maybe_mbd_axes(freqs, fax, tax, rates, opts)
    return _SearchAxes(fax, tax, nf_pad, nt_pad, delays, rates, mbd)
end

"""
    baseline_fringe_search(V, W, freqs, times, f0, t0; opts = FringeSearch()) -> FringeDetection

Search one block of visibilities `V[chan, time]` (with inverse-variance weights
`W[chan, time]`) for the group delay, fringe rate, and phase that align the
visibility phasor. `freqs` (Hz) and `times` (s) label the channel and time axes;
`f0`, `t0` are the delay/rate reference frequency and epoch (the returned phase
is referenced to them). Flagged samples (`W ≤ 0` or non-finite `V`) are dropped.

`V`/`W` may also be passed as vectors for a single-time (`(nchan,)`, delay only)
or single-channel block (`times`-length, rate only).
"""
function baseline_fringe_search(
        V::AbstractMatrix, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real;
        opts::FringeSearch = FringeSearch(),
        workspace::Union{Nothing, FringeWorkspace} = nothing,
    )
    size(V) == size(W) || error("V and W must have the same shape")
    nchan, ntime = size(V)
    (nchan == length(freqs) && ntime == length(times)) ||
        error("V is $(size(V)); expected (length(freqs), length(times)) = ($(length(freqs)), $(length(times)))")
    ax = _search_axes(freqs, times, opts)
    return _baseline_fringe_search(V, W, freqs, times, f0, t0, ax, workspace, opts)
end

# Grid the weighted visibilities onto the workspace's zero-padded uniform grid
# `ws.G` (zeroed here); returns Σw. Shared by the search hot path and the map
# extractor (`baseline_fringe_map`) so the gridding convention has one home.
function _grid_visibilities!(
        ws::FringeWorkspace, V::AbstractMatrix, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, ax::_SearchAxes,
    )
    nchan, ntime = size(V)
    fax = ax.fax
    tax = ax.tax
    nf_pad = ax.nf_pad
    nt_pad = ax.nt_pad
    G = ws.G
    fill!(G, zero(ComplexF64))
    Wsum = 0.0
    @inbounds for ti in 1:ntime, ci in 1:nchan
        w = W[ci, ti]
        v = V[ci, ti]
        (isfinite(w) && w > 0 && isfinite(v)) || continue
        bf = fax.degenerate ? 1 : round(Int, (freqs[ci] - fax.origin) / fax.step) + 1
        bt = tax.degenerate ? 1 : round(Int, (times[ti] - tax.origin) / tax.step) + 1
        (1 <= bf <= nf_pad && 1 <= bt <= nt_pad) || continue
        G[bf, bt] += w * v
        Wsum += w
    end
    return Wsum
end

# Data-driven noise variance of the D plane: median of a strided sample of |D|²
# over the FULL plane (not the narrow search window, whose fringe/sidelobe ridge
# would bias it), converted Rayleigh-median → mean (`median(|D|²) = ln2·mean`).
# Falls back to Σw when the plane is too small to sample. See the SNR note in
# `_baseline_fringe_search`.
function _plane_noise2!(ws::FringeWorkspace, Wsum::Float64)
    dwin = ws.dwin
    empty!(dwin)
    D = ws.D
    ntot = length(D)
    stride = max(1, ntot ÷ 20000)
    @inbounds for idx in 1:stride:ntot
        push!(dwin, abs2(D[idx]))
    end
    return length(dwin) > 2 ? max(median(dwin) / log(2), eps(Float64)) : Wsum
end

# Core matched-filter search on a PRECOMPUTED `_SearchAxes` — the hot path called
# once per (baseline, product). `baseline_fringe_search` above is the public,
# one-off wrapper that builds the axes then calls this; the group search builds the
# axes once and calls this directly for every baseline. Numerically identical to the
# previously-inlined body.
function _baseline_fringe_search(
        V::AbstractMatrix, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        ax::_SearchAxes, workspace::Union{Nothing, FringeWorkspace}, opts::FringeSearch,
    )
    # Hierarchical multi-band path (never allocates the big common-Δf grid).
    ax.mbd === nothing ||
        return _mbd_fringe_search(V, W, freqs, times, f0, t0, ax, workspace, opts)

    nchan, ntime = size(V)
    fax = ax.fax
    tax = ax.tax
    nf_pad = ax.nf_pad
    nt_pad = ax.nt_pad

    # Reuse the workspace's gridding buffer + FFT plan (zeroed each call) instead
    # of allocating an `nf_pad × nt_pad` ComplexF64 grid every call.
    ws = _ensure_workspace!(workspace, nf_pad, nt_pad)
    Wsum = _grid_visibilities!(ws, V, W, freqs, times, ax)
    Wsum > 0 || return _INVALID_DETECTION

    D = ws.D
    mul!(D, ws.plan, ws.G)

    # Conjugate-axis coordinates: delays (s) ↔ frequency grid, rates (Hz) ↔ time.
    # Precomputed once per group in `ax` (identical every call).
    delays = ax.delays
    rates = ax.rates

    # Windowed peak of |D|, plus Σ|D|² over the window for a data-driven noise
    # estimate (see SNR below).
    kbest = lbest = 0
    peakabs = -1.0
    @inbounds for l in eachindex(rates)
        _in_window(rates[l], opts.rate_window, tax.degenerate) || continue
        for k in eachindex(delays)
            _in_window(delays[k], opts.delay_window, fax.degenerate) || continue
            a = abs(D[k, l])
            if a > peakabs
                peakabs = a
                kbest = k
                lbest = l
            end
        end
    end
    peakabs >= 0 || return _INVALID_DETECTION

    # Noise estimate from the full |D|² plane (see `_plane_noise2!`).
    noise2 = _plane_noise2!(ws, Wsum)

    # Quadratic peak refinement on |D| along each non-degenerate axis. The bin
    # spacings are 1/(nf_pad·Δf) for delay and 1/(nt_pad·Δt) for rate.
    delay = delays[kbest]
    rate = rates[lbest]
    if opts.quad_interp
        if !fax.degenerate
            δ = _quad_offset(abs(D[_wrap(kbest - 1, nf_pad), lbest]), peakabs, abs(D[_wrap(kbest + 1, nf_pad), lbest]))
            delay += δ / (nf_pad * fax.step)
        end
        if !tax.degenerate
            δ = _quad_offset(abs(D[kbest, _wrap(lbest - 1, nt_pad)]), peakabs, abs(D[kbest, _wrap(lbest + 1, nt_pad)]))
            rate += δ / (nt_pad * tax.step)
        end
    end

    # The FFT localizes (delay, rate) to a fraction of a bin, but reading φ and
    # amplitude off the discrete peak bin suffers FFT scalloping and a grid-origin
    # phase offset. Re-evaluate the matched filter EXACTLY at the refined
    # (delay, rate), referenced directly to (f0, t0):
    #     Dref = Σ w·V·exp(−2πi[delay·(f−f0) + rate·(t−t0)])
    # so amp = |Dref|/Σw, φ = angle(Dref) — exact, no scalloping loss / origin
    # rotation.
    Dref = _exact_matched_filter(V, W, freqs, times, f0, t0, delay, rate)
    absref = abs(Dref)
    amp = absref / Wsum
    # Data-driven SNR: the matched-filter noise is estimated from the spread of
    # |D| over the search plane, robustly (median) so the bright peak and its
    # sidelobes don't bias it. For Rayleigh-distributed noise bins
    # `median(|D|²) = ln2 · mean(|D|²)`, and `mean(|D|²) = Σw` when the weights are
    # true inverse-variances — so this reduces to the matched-filter |Dref|/√Σw
    # for calibrated data, but stays correct when the WEIGHT column is
    # uncalibrated / uniform (common in raw correlator output), where √Σw badly
    # mis-scales the SNR and silently fails the snr_min gate. Falls back to √Σw if
    # the window is too small to estimate noise.
    snr = absref / sqrt(noise2)
    phase = rem2pi(angle(Dref), RoundNearest)

    return FringeDetection(delay, rate, phase, amp, snr, snr >= opts.snr_min)
end

# Vector overloads: single-time (delay only) and the general fallback.
function baseline_fringe_search(
        V::AbstractVector, W::AbstractVector,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real;
        opts::FringeSearch = FringeSearch(),
    )
    # Interpret a vector as (nchan, 1) when it matches freqs, else (1, ntime).
    if length(V) == length(freqs) && length(times) == 1
        return baseline_fringe_search(reshape(V, :, 1), reshape(W, :, 1), freqs, times, f0, t0; opts)
    elseif length(V) == length(times) && length(freqs) == 1
        return baseline_fringe_search(reshape(V, 1, :), reshape(W, 1, :), freqs, times, f0, t0; opts)
    else
        error("vector baseline_fringe_search: length(V)=$(length(V)) matches neither freqs ($(length(freqs))) nor times ($(length(times)))")
    end
end

# Exact (grid-free) matched filter Σ w·V·exp(−2πi[τ(f−f0) + ṙ(t−t0)]) at one
# (delay, rate) point — the scalloping-free evaluation shared by both search
# paths (final phase/amp readout, and the MBD ambiguity arbitration, which
# evaluates several candidates per search). The phase is separable, so the
# phasors are precomputed per axis: O(nchan + ntime) `cis` calls instead of
# O(nchan·ntime) — `cis` dominates the naive loop.
function _exact_matched_filter(
        V::AbstractMatrix, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        delay::Real, rate::Real,
    )
    nchan, ntime = size(V)
    cf = Vector{ComplexF64}(undef, nchan)
    @inbounds for ci in 1:nchan
        cf[ci] = cis(-2π * delay * (freqs[ci] - f0))
    end
    Dref = zero(ComplexF64)
    @inbounds for ti in 1:ntime
        acc = zero(ComplexF64)
        for ci in 1:nchan
            w = W[ci, ti]
            v = V[ci, ti]
            (isfinite(w) && w > 0 && isfinite(v)) || continue
            acc += w * v * cf[ci]
        end
        Dref += acc * cis(-2π * rate * (times[ti] - t0))
    end
    return Dref
end

# 3-point quadratic vertex offset (in bins) given neighbour magnitudes `ym, y0,
# yp` straddling the peak `y0`. Returns 0 if the curvature is non-concave.
function _quad_offset(ym::Real, y0::Real, yp::Real)
    denom = ym - 2y0 + yp
    denom < 0 || return 0.0
    δ = 0.5 * (ym - yp) / denom
    return clamp(δ, -0.5, 0.5)
end

_wrap(i::Int, n::Int) = mod(i - 1, n) + 1

# Window test; a degenerate axis (single sample) always passes (its conjugate
# coordinate is identically 0).
_in_window(x::Real, window::Tuple{<:Real, <:Real}, degenerate::Bool) =
    degenerate || (window[1] <= x <= window[2])

# ── Hierarchical SBD → MBD search (HOPS/fourfit-style) ─────────────────────────
#
# For narrow bands spread over a wide span, factor the matched filter over the
# band blocks b (with per-band grid origin f_b):
#
#     D(τ, ṙ) = Σ_b exp(−2πi·τ·(f_b − f_1)) · D_b(τ, ṙ)
#     D_b(τ, ṙ) = Σ_{f∈b, t} w·V·exp(−2πi[τ·(f − f_b) + ṙ·(t − t0)])
#
# |D_b| varies with τ on the SLOW in-band scale 1/BW_band (the single-band delay,
# SBD), while the band phase factor varies on the FAST scale 1/span (the multi-
# band delay, MBD). So:
#
#   stage 1: per band, one SMALL 2-D FFT (in-band delay × rate) — D_b on a coarse
#            SBD grid; keep only the windowed (SBD, rate) block.
#   stage 2: per SBD bin, a SMALL 1-D FFT across the band ORIGINS (gridded on a
#            coarse common spacing Δbc) → the (MBD, rate) plane; scan the 3-D
#            (SBD, MBD, rate) cube for the windowed peak.
#
# The MBD is periodic with ambiguity A = 1/Δbc; the total delay is the MBD
# unfolded near the SBD estimate. Because the SBD bin (1/BW_band) can be
# comparable to A, the ambiguity k is ARBITRATED BY THE EXACT MATCHED FILTER:
# every candidate mbd + k·A within ~1.5 SBD bins of the SBD estimate (and inside
# the window) is evaluated exactly and the strongest wins — the exact filter sees
# the true in-band slope, which is precisely what distinguishes the aliases. The
# final fine delay is then polished by a parabolic step on the exact filter, so
# the result carries no grid scalloping.
#
# Cost: the giant nf_pad×nt_pad grid (mostly zeros for VGOS layouts) is replaced
# by nband small per-band FFTs plus nsbd tiny band-center FFTs — orders of
# magnitude less compute AND memory traffic. Noise/SNR conventions are identical
# to the full path: every cube cell is a matched-filter output with variance Σw
# under noise, so the same strided-median estimate applies.

# Approximate GCD of nonnegative reals within absolute tolerance `tol` (folded
# Euclid: remainders within `tol` of 0 OR of the divisor count as exact).
# Returns 0 when no value exceeds `tol`.
function _approx_gcd(vals::AbstractVector{<:Real}, tol::Real)
    g = 0.0
    for v0 in vals
        v = abs(float(v0))
        v <= tol && continue
        if g == 0.0
            g = v
            continue
        end
        a, b = max(g, v), min(g, v)
        while b > tol
            r = mod(a, b)
            a, b = b, min(r, b - r)
        end
        g = a
    end
    return g
end

# Split a SORTED frequency axis into contiguous band blocks: a new block starts
# wherever the spacing exceeds 1.5× the in-band spacing Δf.
function _detect_bands(freqs::AbstractVector, Δf::Real)
    bands = UnitRange{Int}[]
    lo = 1
    for i in 1:(length(freqs) - 1)
        if freqs[i + 1] - freqs[i] > 1.5 * Δf
            push!(bands, lo:i)
            lo = i + 1
        end
    end
    push!(bands, lo:length(freqs))
    return bands
end

# Precomputed hierarchical-search geometry (per scan group, like `_SearchAxes`).
struct _MBDAxes
    bands::Vector{UnitRange{Int}}   # channel-index blocks (ascending frequency)
    f_lo::Vector{Float64}           # per-band grid origin (first channel freq)
    bc_bin::Vector{Int}             # band-origin bin on the Δbc grid (1-based)
    bc_step::Float64                # Δbc: common band-origin grid spacing
    nbc_pad::Int                    # padded band-center FFT length
    nfb_pad::Int                    # padded per-band (SBD) FFT length, shared
    ambig::Float64                  # MBD ambiguity A = 1/Δbc
    sbd_bin::Float64                # SBD grid spacing 1/(nfb_pad·Δf)
    mbd_bin::Float64                # MBD grid spacing A/nbc_pad
    sbd_idx::Vector{Int}            # SBD FFT rows kept (sorted by SBD value)
    sbd_val::Vector{Float64}        # SBD value per kept row
    mbd::Vector{Float64}            # MBD value per band-center FFT row (FFT order)
    rate_idx::Vector{Int}           # rate FFT cols kept (sorted by value; window ±1)
    rate_val::Vector{Float64}       # rate value per kept col
    rate_scan::Vector{Int}          # positions in rate_idx inside the rate window
end

# Decide whether the hierarchical path applies (see `FringeSearch.algorithm`),
# and build its geometry; `nothing` selects the full path.
function _maybe_mbd_axes(freqs::AbstractVector, fax::_Axis, tax::_Axis, rates::Vector{Float64}, opts::FringeSearch)
    alg = opts.algorithm
    alg in (:auto, :full, :mbd) || error("FringeSearch: algorithm must be :auto, :full, or :mbd (got $(alg))")
    alg === :full && return nothing
    (fax.degenerate || !issorted(freqs)) && return nothing
    bands = _detect_bands(freqs, fax.step)
    length(bands) >= 2 || return nothing
    if alg === :auto
        # Hierarchical only when the common grid is mostly padding (> 4× the real
        # channel count) — else the single FFT is simple and fast enough.
        fax.n > 4 * length(freqs) || return nothing
    end
    return _build_mbd_axes(freqs, bands, fax, tax, rates, opts)
end

function _build_mbd_axes(
        freqs::AbstractVector, bands::Vector{UnitRange{Int}},
        fax::_Axis, tax::_Axis, rates::Vector{Float64}, opts::FringeSearch,
    )
    Δf = fax.step
    f_lo = [Float64(freqs[first(b)]) for b in bands]
    maxbins = maximum(round(Int, (freqs[last(b)] - freqs[first(b)]) / Δf) + 1 for b in bands)
    # The internal axes are tiny, so oversample them at least 4× regardless of
    # `opts.oversample` (which still governs the rate axis via `nt_pad`).
    osb = max(opts.oversample, 4)
    nfb_pad = _fast_fft_size(osb * maxbins)

    # Common band-origin grid spacing Δbc: the approximate GCD of the origin
    # offsets. Real layouts mix origin spacings (VGOS: 64/96/192 MHz within and
    # between band groups) whose common divisor (32 MHz there) is what defines
    # the MBD ambiguity. The tolerance is physical: an off-grid residual ε
    # decoheres the stage-2 band sum by 2π·τ·ε at trial delay τ, so residuals
    # must satisfy 2π·τmax·ε ≤ 0.3 rad across the delay window (the exact
    # matched filter, which uses the true frequencies, then removes even that
    # from the final delay/amp/phase/SNR).
    offs = f_lo .- f_lo[1]
    τmax = max(abs(opts.delay_window[1]), abs(opts.delay_window[2]), 1.0e-12)
    tol = 0.3 / (2π * τmax)
    Δbc = _approx_gcd(offs, tol)
    Δbc > 0 || return nothing
    maximum(abs(o - round(o / Δbc) * Δbc) for o in offs) <= tol || return nothing
    bc_bin = [round(Int, o / Δbc) + 1 for o in offs]
    allunique(bc_bin) || return nothing
    nbc_pad = _fast_fft_size(osb * maximum(bc_bin))
    # A near-continuum of origin bins means the hierarchy buys nothing (the
    # band-center FFT approaches the full grid) — let the full path handle it.
    nbc_pad <= 65536 || return nothing
    A = 1.0 / Δbc

    # SBD rows: window ± (A/2 + one bin) — the peak's SBD may sit half an
    # ambiguity from the true delay, and quad refinement needs a neighbour.
    sbd_all = fftfreq(nfb_pad, 1.0 / Δf)
    sbd_bin = 1.0 / (nfb_pad * Δf)
    lo, hi = opts.delay_window
    margin = A / 2 + sbd_bin
    sbd_idx = [k for k in eachindex(sbd_all) if (lo - margin) <= sbd_all[k] <= (hi + margin)]
    isempty(sbd_idx) && return nothing
    sort!(sbd_idx; by = k -> sbd_all[k])
    sbd_val = Float64[sbd_all[k] for k in sbd_idx]

    # Rate cols: the window bins plus one value-neighbour each side (for quad).
    ord = sortperm(rates)
    sel = [i for i in eachindex(ord) if _in_window(rates[ord[i]], opts.rate_window, tax.degenerate)]
    isempty(sel) && return nothing
    i1 = max(first(sel) - 1, 1)
    i2 = min(last(sel) + 1, length(ord))
    rate_idx = ord[i1:i2]
    rate_val = rates[rate_idx]
    rate_scan = collect((first(sel) - i1 + 1):(last(sel) - i1 + 1))

    mbd = collect(fftfreq(nbc_pad, 1.0 / Δbc))
    return _MBDAxes(
        bands, f_lo, bc_bin, Δbc, nbc_pad, nfb_pad, A, sbd_bin, A / nbc_pad,
        sbd_idx, sbd_val, mbd, rate_idx, rate_val, rate_scan,
    )
end

# Scratch for the hierarchical path, hung off `FringeWorkspace.mbd` (lazily
# (re)built when the group geometry changes, like the main workspace).
mutable struct _MBDWorkspace
    nfb::Int
    nt::Int
    nband::Int
    nsbd::Int
    nrw::Int
    nbc::Int
    Gb::Matrix{ComplexF64}          # per-band gridding buffer (nfb_pad × nt_pad)
    Db::Matrix{ComplexF64}
    planb::Any
    X::Array{ComplexF64, 3}         # stage-1 outputs, windowed: (nsbd, nrw, nband)
    Mc::Matrix{ComplexF64}          # stage-2 input (nbc_pad × nrw)
    Dc::Matrix{ComplexF64}
    planc::Any
end

function _ensure_mbd_workspace!(ws::FringeWorkspace, mx::_MBDAxes, nt_pad::Int)
    nsbd = length(mx.sbd_idx)
    nrw = length(mx.rate_idx)
    nband = length(mx.bands)
    w = ws.mbd
    if !(w isa _MBDWorkspace) || w.nfb != mx.nfb_pad || w.nt != nt_pad ||
            w.nband != nband || w.nsbd != nsbd || w.nrw != nrw || w.nbc != mx.nbc_pad
        Gb = zeros(ComplexF64, mx.nfb_pad, nt_pad)
        Db = similar(Gb)
        planb = plan_fft(Gb; flags = MEASURE)
        X = Array{ComplexF64, 3}(undef, nsbd, nrw, nband)
        Mc = zeros(ComplexF64, mx.nbc_pad, nrw)
        Dc = similar(Mc)
        planc = plan_fft(Mc, 1; flags = MEASURE)
        w = _MBDWorkspace(mx.nfb_pad, nt_pad, nband, nsbd, nrw, mx.nbc_pad, Gb, Db, planb, X, Mc, Dc, planc)
        ws.mbd = w
    end
    return w::_MBDWorkspace
end

# Stage 2 for one SBD row: scatter the bands onto the Δbc grid and FFT along it
# (dim 1) → the (MBD, rate) plane in `w.Dc`.
function _stage2_plane!(w::_MBDWorkspace, mx::_MBDAxes, sj::Int)
    Mc = w.Mc
    fill!(Mc, zero(ComplexF64))
    @inbounds for b in 1:w.nband, rj in 1:w.nrw
        Mc[mx.bc_bin[b], rj] += w.X[sj, rj, b]
    end
    mul!(w.Dc, w.planc, Mc)
    return w.Dc
end

# One (MBD row, rate col) value of the stage-2 sum for an arbitrary SBD row —
# the direct nband-term sum, matching the FFT's index phase exactly. Used for
# quad refinement along the SBD axis without rebuilding whole planes.
function _stage2_value(w::_MBDWorkspace, mx::_MBDAxes, sj::Int, m::Int, rj::Int)
    acc = zero(ComplexF64)
    ph = -2π * (m - 1) / mx.nbc_pad
    @inbounds for b in 1:w.nband
        acc += w.X[sj, rj, b] * cis(ph * (mx.bc_bin[b] - 1))
    end
    return acc
end

# The hierarchical search core (same contract as the full-path body of
# `_baseline_fringe_search`: identical FringeDetection semantics and SNR/noise
# conventions).
function _mbd_fringe_search(
        V::AbstractMatrix, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        ax::_SearchAxes, workspace::Union{Nothing, FringeWorkspace}, opts::FringeSearch,
    )
    mx = ax.mbd
    tax = ax.tax
    nt_pad = ax.nt_pad
    ws = workspace === nothing ? FringeWorkspace() : workspace
    w = _ensure_mbd_workspace!(ws, mx, nt_pad)
    ntime = size(V, 2)
    Δf = ax.fax.step
    nband = w.nband

    # Stage 1: per band, grid + 2-D FFT (in-band delay × rate); keep the windowed
    # (SBD row, rate col) block. The noise is estimated HERE, from a strided
    # median of each band's FULL |D_b|² plane (the windowed stage-2 cube sits on
    # the fringe's sidelobe ridge and would bias it — same rationale as the full
    # path's whole-plane sample): the band contributions to D are independent, so
    # Var(D) = Σ_b Var(D_b), each `median|D_b|²/ln2` (Rayleigh median → mean).
    Wsum = 0.0
    noise2 = 0.0
    nnoise = 0
    dwin = ws.dwin
    for bi in 1:nband
        Gb = w.Gb
        fill!(Gb, zero(ComplexF64))
        flo = mx.f_lo[bi]
        @inbounds for ti in 1:ntime, ci in mx.bands[bi]
            wgt = W[ci, ti]
            v = V[ci, ti]
            (isfinite(wgt) && wgt > 0 && isfinite(v)) || continue
            bf = round(Int, (freqs[ci] - flo) / Δf) + 1
            bt = tax.degenerate ? 1 : round(Int, (times[ti] - tax.origin) / tax.step) + 1
            (1 <= bf <= w.nfb && 1 <= bt <= nt_pad) || continue
            Gb[bf, bt] += wgt * v
            Wsum += wgt
        end
        mul!(w.Db, w.planb, Gb)
        empty!(dwin)
        ntot = length(w.Db)
        stride = max(1, ntot ÷ max(64, 20000 ÷ nband))
        @inbounds for idx in 1:stride:ntot
            push!(dwin, abs2(w.Db[idx]))
        end
        if length(dwin) > 2
            noise2 += median(dwin) / log(2)
            nnoise += 1
        end
        @inbounds for (rj, l) in enumerate(mx.rate_idx), (sj, k) in enumerate(mx.sbd_idx)
            w.X[sj, rj, bi] = w.Db[k, l]
        end
    end
    Wsum > 0 || return _INVALID_DETECTION
    noise2 = (nnoise == nband && noise2 > 0) ? max(noise2, eps(Float64)) : Wsum

    # Stage 2: scan the (SBD, MBD, rate) cube for the windowed peak.
    nsbd = w.nsbd
    nbc = mx.nbc_pad
    A = mx.ambig
    lo, hi = opts.delay_window
    peak = -1.0
    ps = pm = pr = 0
    for sj in 1:nsbd
        Dc = _stage2_plane!(w, mx, sj)
        sv = mx.sbd_val[sj]
        @inbounds for rj in mx.rate_scan, m in 1:nbc
            a = abs(Dc[m, rj])
            a > peak || continue
            # Total-delay candidate: the MBD unfolded to the branch nearest this
            # SBD row; only cells whose unfolded delay is in the window compete.
            du = mx.mbd[m] + A * round((sv - mx.mbd[m]) / A)
            (lo <= du <= hi) || continue
            peak = a
            ps = sj
            pm = m
            pr = rj
        end
    end
    peak >= 0 || return _INVALID_DETECTION

    # Refinement. Rebuild the winning plane (the loop reused the buffers), then
    # quad-refine MBD (periodic → wrapped neighbours), rate, and SBD.
    Dc = _stage2_plane!(w, mx, ps)
    mbd_ref = mx.mbd[pm]
    rate_ref = mx.rate_val[pr]
    sbd_ref = mx.sbd_val[ps]
    if opts.quad_interp
        ym = abs(Dc[_wrap(pm - 1, nbc), pr])
        yp = abs(Dc[_wrap(pm + 1, nbc), pr])
        mbd_ref += _quad_offset(ym, peak, yp) * mx.mbd_bin
        if !tax.degenerate && 1 < pr < w.nrw
            rate_ref += _quad_offset(abs(Dc[pm, pr - 1]), peak, abs(Dc[pm, pr + 1])) / (nt_pad * tax.step)
        end
        if 1 < ps < nsbd
            ym = abs(_stage2_value(w, mx, ps - 1, pm, pr))
            yp = abs(_stage2_value(w, mx, ps + 1, pm, pr))
            sbd_ref += _quad_offset(ym, peak, yp) * mx.sbd_bin
        end
    end

    # Ambiguity arbitration: candidate total delays mbd_ref + k·A near the SBD
    # estimate (within ±1.5 SBD bins, inside the window), decided by the EXACT
    # matched filter — it sees the true in-band slope, which is what separates
    # the aliases. Usually one candidate (A ≫ sbd_bin); a handful otherwise.
    half = 1.5 * mx.sbd_bin
    kmin = ceil(Int, (max(lo, sbd_ref - half) - mbd_ref) / A)
    kmax = floor(Int, (min(hi, sbd_ref + half) - mbd_ref) / A)
    if kmin > kmax
        kmin = kmax = round(Int, (clamp(sbd_ref, lo, hi) - mbd_ref) / A)
    end
    delay = mbd_ref + kmin * A
    Dref = _exact_matched_filter(V, W, freqs, times, f0, t0, delay, rate_ref)
    for k in (kmin + 1):kmax
        d = mbd_ref + k * A
        Dk = _exact_matched_filter(V, W, freqs, times, f0, t0, d, rate_ref)
        if abs(Dk) > abs(Dref)
            Dref = Dk
            delay = d
        end
    end

    # Parabolic polish of the fine delay on the exact filter (plays the role of
    # the full path's on-grid quad refinement, but scalloping-free). Two passes:
    # the second re-centers after any shift left by Δbc-grid residuals.
    if opts.quad_interp
        h = mx.mbd_bin / 2
        for _ in 1:2
            ym = abs(_exact_matched_filter(V, W, freqs, times, f0, t0, delay - h, rate_ref))
            yp = abs(_exact_matched_filter(V, W, freqs, times, f0, t0, delay + h, rate_ref))
            δ = _quad_offset(ym, abs(Dref), yp)
            δ == 0.0 && break
            delay += δ * h
            Dref = _exact_matched_filter(V, W, freqs, times, f0, t0, delay, rate_ref)
        end
    end

    absref = abs(Dref)
    amp = absref / Wsum
    snr = absref / sqrt(noise2)
    phase = rem2pi(angle(Dref), RoundNearest)
    return FringeDetection(delay, rate_ref, phase, amp, snr, snr >= opts.snr_min)
end

# ── False-fringe statistics + the delay–rate map extractor ─────────────────────

# Effective number of INDEPENDENT search cells inside the (delay, rate) window.
# The delay resolution is 1/(gridded bandwidth span) = 1/(fax.n·fax.step) and the
# rate resolution 1/(time span), so cells-per-axis = window span / resolution,
# clamped to [1, n gridded samples] (zero-pad `oversample` refines the peak but
# adds NO independent cells). A degenerate axis contributes a factor 1.
function _search_cells(ax::_SearchAxes, opts::FringeSearch)
    nd = ax.fax.degenerate ? 1.0 :
        clamp((opts.delay_window[2] - opts.delay_window[1]) * ax.fax.n * ax.fax.step, 1.0, Float64(ax.fax.n))
    nr = ax.tax.degenerate ? 1.0 :
        clamp((opts.rate_window[2] - opts.rate_window[1]) * ax.tax.n * ax.tax.step, 1.0, Float64(ax.tax.n))
    return nd * nr
end

"""
    fringe_pfa(snr, ncells) -> Float64

Probability of false alarm of a fringe detection: the probability that pure
noise, searched over `ncells` independent (delay, rate) cells, produces a peak
of at least `snr` (HOPS-style). With this module's SNR convention
(`snr = |D| / √E[|D|²]` under noise, so a noise cell exceeds `s` with
probability `exp(−s²)`),

    pfa = 1 − (1 − exp(−snr²))^ncells

evaluated stably (`≈ ncells·exp(−snr²)` when small). `pfa ≪ 1` marks a secure
detection; `pfa ≳ 0.01` means the peak is consistent with the noise sidelobe
forest — a likely FALSE fringe. Returns `NaN` for non-finite inputs.
"""
function fringe_pfa(snr::Real, ncells::Real)
    (isfinite(snr) && isfinite(ncells)) || return NaN
    snr <= 0 && return 1.0
    p1 = exp(-float(snr)^2)                    # single-cell exceedance
    p1 >= 1 && return 1.0
    return -expm1(max(ncells, 1.0) * log1p(-p1))
end

"""
    FringeSearchMap

The windowed delay–rate matched-filter surface of one visibility block, produced
by [`baseline_fringe_map`](@ref) — the classic false-fringe diagnostic. Fields:

- `delays` (s) / `rates` (Hz) — the in-window grid coordinates, ascending.
- `snr` — `(ndelay, nrate)` map of `|D| / noise` in the SAME units as
  `FringeDetection.snr`, so the map's peak sits at ≈ `detection.snr`.
- `detection` — the refined peak, exactly as [`baseline_fringe_search`](@ref)
  returns it.
- `ncells` — effective number of independent search cells (see `fringe_pfa`).
- `pfa` — `fringe_pfa(detection.snr, ncells)` for this single search.

A real fringe is one sharp peak far above the sidelobe forest (`pfa ≪ 1`); a
false fringe barely clears the forest (`pfa` not small) and typically shows
several comparable-height peaks.
"""
struct FringeSearchMap
    delays::Vector{Float64}
    rates::Vector{Float64}
    snr::Matrix{Float64}
    detection::FringeDetection
    ncells::Float64
    pfa::Float64
end

"""
    baseline_fringe_map(V, W, freqs, times, f0, t0; opts = FringeSearch(), workspace = nothing)
        -> FringeSearchMap

Compute the full delay–rate SNR surface for one visibility block — the
brute-force common-Δf grid/FFT (`algorithm = :full`), returning the windowed
`|D|` plane (in SNR units) instead of only the peak. The PLANE is always the
full grid — showing the complete sidelobe/alias structure is the point of the
diagnostic — while the embedded `detection` honours `opts.algorithm`, so it is
identical to what [`baseline_fringe_search`](@ref) (and the solver) returns. A
diagnostic (a full-grid FFT per call — on VGOS-style axes this single plane is
much more expensive than the hierarchical search that produced the solution),
not a hot path.
"""
function baseline_fringe_map(
        V::AbstractMatrix, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real;
        opts::FringeSearch = FringeSearch(),
        workspace::Union{Nothing, FringeWorkspace} = nothing,
    )
    size(V) == size(W) || error("V and W must have the same shape")
    (size(V, 1) == length(freqs) && size(V, 2) == length(times)) ||
        error("V is $(size(V)); expected (length(freqs), length(times)) = ($(length(freqs)), $(length(times)))")
    ax = _search_axes(freqs, times, opts)                # detection axes (honour opts.algorithm)
    axf = ax.mbd === nothing ? ax :                      # plane axes: always the full grid
        _search_axes(freqs, times, FringeSearch(opts.delay_window, opts.rate_window, opts.oversample, opts.snr_min, opts.quad_interp, :full))
    ncells = _search_cells(axf, opts)
    ws = _ensure_workspace!(workspace, axf.nf_pad, axf.nt_pad)
    Wsum = _grid_visibilities!(ws, V, W, freqs, times, axf)
    Wsum > 0 || return FringeSearchMap(Float64[], Float64[], zeros(0, 0), _INVALID_DETECTION, ncells, NaN)
    D = ws.D
    mul!(D, ws.plan, ws.G)
    noise2 = _plane_noise2!(ws, Wsum)

    # In-window bins of each conjugate axis, in ascending coordinate order (the
    # fftfreq vectors are in FFT order [0, +…, −…]).
    kidx = [k for k in eachindex(axf.delays) if _in_window(axf.delays[k], opts.delay_window, axf.fax.degenerate)]
    lidx = [l for l in eachindex(axf.rates) if _in_window(axf.rates[l], opts.rate_window, axf.tax.degenerate)]
    sort!(kidx; by = k -> axf.delays[k])
    sort!(lidx; by = l -> axf.rates[l])
    inv_noise = 1.0 / sqrt(noise2)
    snrmap = Matrix{Float64}(undef, length(kidx), length(lidx))
    @inbounds for (j, l) in enumerate(lidx), (i, k) in enumerate(kidx)
        snrmap[i, j] = abs(D[k, l]) * inv_noise
    end

    # The refined peak, via the standard search (re-grids + re-FFTs the same data
    # in `ws` — the map above is already copied out, and reusing the search keeps
    # the peak/refinement logic in one place).
    det = _baseline_fringe_search(V, W, freqs, times, f0, t0, ax, ws, opts)
    return FringeSearchMap(axf.delays[kidx], axf.rates[lidx], snrmap, det, ncells, fringe_pfa(det.snr, ncells))
end
