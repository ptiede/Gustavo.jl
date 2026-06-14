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
# onto a common Δf grid with empty bins zero-filled, so a concatenated-spw block
# searches multi-band delay correctly.

"""
    FringeSearch(; delay_window, rate_window, oversample, snr_min, quad_interp)

Options for [`baseline_fringe_search`](@ref).

- `delay_window`  : `(lo, hi)` delay search window in seconds. Default ±1 µs.
- `rate_window`   : `(lo, hi)` fringe-rate search window in Hz. Default ±50 mHz.
- `oversample`    : zero-padding factor per axis (finer delay/rate grid). Default 4.
- `snr_min`       : detection threshold; `valid = snr ≥ snr_min`. Default 6.
- `quad_interp`   : refine the peak by 3-point quadratic interpolation. Default true.
"""
Base.@kwdef struct FringeSearch
    delay_window::Tuple{Float64, Float64} = (-1.0e-6, 1.0e-6)
    rate_window::Tuple{Float64, Float64} = (-0.05, 0.05)
    oversample::Int = 4
    snr_min::Float64 = 6.0
    quad_interp::Bool = true
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
    )
    size(V) == size(W) || error("V and W must have the same shape")
    nchan, ntime = size(V)
    (nchan == length(freqs) && ntime == length(times)) ||
        error("V is $(size(V)); expected (length(freqs), length(times)) = ($(length(freqs)), $(length(times)))")

    fax = _uniform_axis(freqs)
    tax = _uniform_axis(times)

    # Pad each axis (oversample), then round up to a fast FFT size. A degenerate
    # axis keeps length 1 (its conjugate parameter stays 0).
    nf_pad = fax.degenerate ? 1 : _fast_fft_size(opts.oversample * fax.n)
    nt_pad = tax.degenerate ? 1 : _fast_fft_size(opts.oversample * tax.n)

    # Place weighted visibilities on the uniform grid.
    G = zeros(ComplexF64, nf_pad, nt_pad)
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
    Wsum > 0 || return _INVALID_DETECTION

    D = fft(G)

    # Conjugate-axis coordinates: delays (s) ↔ frequency grid, rates (Hz) ↔ time.
    delays = fax.degenerate ? [0.0] : collect(fftfreq(nf_pad, 1.0 / fax.step))
    rates = tax.degenerate ? [0.0] : collect(fftfreq(nt_pad, 1.0 / tax.step))

    # Windowed peak of |D|.
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
    # so amp = |Dref|/Σw, snr = |Dref|/√Σw, φ = angle(Dref) — all exact, no
    # scalloping loss, no origin rotation.
    Dref = zero(ComplexF64)
    @inbounds for ti in 1:ntime, ci in 1:nchan
        w = W[ci, ti]
        v = V[ci, ti]
        (isfinite(w) && w > 0 && isfinite(v)) || continue
        ph = 2π * (delay * (freqs[ci] - f0) + rate * (times[ti] - t0))
        Dref += w * v * cis(-ph)
    end
    absref = abs(Dref)
    amp = absref / Wsum
    snr = absref / sqrt(Wsum)
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
