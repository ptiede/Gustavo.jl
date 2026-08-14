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
# zero-pad (oversample), FFT, take the windowed peak of |D|, then polish that
# coarse (delay, rate) cell on the EXACT matched filter (`_polish_peak_exact!`) and
# read φ off it. The FFT only LOCATES the main lobe; the sub-cell peak comes from
# the exact objective, so accuracy no longer needs a fine `oversample` grid.
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
#
# Precision: the compute type `C` (complex; `T = real(C)`) flows from the
# visibility block's own eltype rather than being fixed at `ComplexF64` — a
# `ComplexF32` caller's FFT runs natively in `Float32` (FFTW itself supports no
# other complex type). This applies to the FFT grid/plan (`FringeWorkspace{C,T}`,
# `_MBDWorkspace{C}`) and the small FFT-conjugate delay/rate axes (`_SearchAxes`'s
# `delays`/`rates`, `_MBDAxes`'s `sbd_val`/`mbd`/`rate_val`/etc), which are the
# compute-dominant pieces. The channel/time AXIS ORIGIN AND SPACING
# (`_uniform_axis`'s `origin`/`step`, `_MBDAxes`'s `f_lo`/`bc_step`) and the raw
# `freqs`/`times` used directly in phase computation (`_exact_matched_filter`)
# stay `Float64` regardless of `C`: these are absolute physical values (e.g.
# ~10¹¹ Hz), and `Float32`'s ~7 significant digits would quantize a GHz-scale
# origin to within a few kHz — enough to misround channel bins on fine spacing.
# Narrowing to `T` happens only for small, already-relative quantities (grid
# reciprocals, FFT outputs).

"""
    AbstractSearchAlgorithm

How the delay search lays out its FFT grid. Built-ins: [`FullGrid`](@ref) and
[`HierarchicalMBD`](@ref).

A custom algorithm subtypes this and defines

    Gustavo.Fringe._mbd_axes(alg, freqs, fax, tax, rates, opts, ::Type{C}) -> mbd_axes or nothing

returning `nothing` to run the single full-grid FFT, or the hierarchical band
geometry to run the two-stage search. `C` is the search's compute type (the
visibility block's eltype), needed to build any FFT plans the geometry carries.
There is no fallback: an algorithm with no method is an error rather than a
silent switch to a different search.
"""
abstract type AbstractSearchAlgorithm end

"""
    FullGrid()

One brute-force FFT over the common-Δf grid spanning the whole frequency axis.
Simple and fast for contiguous bands (e.g. VLBA); on widely-separated narrow
bands the grid is almost entirely zero-padding.
"""
struct FullGrid <: AbstractSearchAlgorithm end

"""
    HierarchicalMBD()

Hierarchical single-band → multi-band delay search (HOPS/fourfit style): a
per-band in-band delay, then a multi-band delay over the band origins, which
resolves the MBD ambiguity explicitly against the in-band delay. Far cheaper
than [`FullGrid`](@ref) when narrow bands are spread over a wide span
(VGOS-style). Falls back to the full grid when the axis cannot support it —
fewer than two band blocks, a degenerate or unsorted frequency axis.
"""
struct HierarchicalMBD <: AbstractSearchAlgorithm end

"""
    FringeSearch(; delay_window, rate_window, oversample, quad_interp, algorithm)

Options for [`baseline_fringe_search`](@ref).

The search measures; it does not judge. Every cell holding usable data yields a
peak with its signal-to-noise and false-alarm probability, and no detection
threshold is applied here — admission to the station solve is
[`Stationization`](@ref)'s decision, taken on the recorded `pfa`.

- `delay_window`  : `(lo, hi)` delay search window in seconds. Default ±1 µs.
- `rate_window`   : `(lo, hi)` fringe-rate search window in Hz. Default ±0.8 Hz.
  Narrowing it buys no speed — the FFT spans the whole plane either way — only a
  smaller false-alarm trial count, while a window narrower than the true rate
  spread hides a station outright. A station off the array's clock reaches
  several hundred mHz at mm wavelengths, so the default is generous; keep any
  choice well inside the ±1/(2Δt) Nyquist rate, past which a peak is an alias of
  its own wrap.
- `oversample`    : zero-padding factor per axis (finer delay/rate grid). Default 8.
  Widely-separated narrow bands may need a larger value (or a tight
  `delay_window`) to avoid locking onto a multi-band alias peak.
- `quad_interp`   : polish the peak on the exact matched filter (sub-cell, per-axis
  parabolic steps on the true objective — `_polish_peak_exact!`). Default true.
- `algorithm`     : an [`AbstractSearchAlgorithm`](@ref) — [`FullGrid`](@ref) or
  [`HierarchicalMBD`](@ref) — or the sentinel `:auto` (default), which picks
  between them from the frequency axis: `HierarchicalMBD` when the axis has ≥ 2
  band blocks and the common grid would be > 4× the real channel count (mostly
  zero-padding), else `FullGrid`. Contiguous-band data (e.g. VLBA) always takes
  the full-grid path.
"""
Base.@kwdef struct FringeSearch
    delay_window::Tuple{Float64, Float64} = (-1.0e-6, 1.0e-6)
    rate_window::Tuple{Float64, Float64} = (-0.8, 0.8)
    oversample::Int = 8
    quad_interp::Bool = true
    algorithm::Union{Symbol, AbstractSearchAlgorithm} = :auto
end

"""
    Detection{T} = @NamedTuple{delay, rate, phase, amp, snr, pfa, valid}

The result of a single-baseline fringe search. `delay` (s) and `rate` (Hz) are
the station-pair group delay and fringe rate; `phase` (rad) is the constant
phase φ referenced to `(f0, t0)`; `amp` is the coherent amplitude; `snr` the
detection signal-to-noise.

`pfa` is the probability that noise alone would produce a peak this strong
somewhere in the search family the measurement belongs to — the whole family of
(baseline × product × scan) searches sharing one false-alarm budget, not this
one search in isolation, so it is directly comparable to
`Stationization.pfa_max` (see [`search_scan`](@ref)).

`valid` says a peak was MEASURED here, not that it passed any threshold: it is
false only for a cell with no usable data (zero total weight, no peak), whose
other fields are all zero and carry no information. `pfa` is the quantity that
separates a real fringe from noise.

`T` is the search's compute precision (`real(C)` for a `ComplexF32`/`ComplexF64`
visibility block).
"""
const Detection{T} = @NamedTuple{
    delay::T, rate::T, phase::T,
    amp::T, snr::T, pfa::T, valid::Bool,
}

# The zeroed detection of a cell with no usable data, at precision `T`. `pfa` is
# 1 — certainty that noise explains it — so a cell that reaches an admission test
# despite `valid = false` is rejected rather than admitted on a zero `pfa`. The
# second method lets a caller derive `T` from an existing cell's own type
# (`_invalid_detection(typeof(d))`) without naming `T` explicitly.
_invalid_detection(::Type{T}) where {T} =
    Detection{T}((zero(T), zero(T), zero(T), zero(T), zero(T), one(T), false))
_invalid_detection(::Type{Detection{T}}) where {T} = _invalid_detection(T)

"""
    FringeWorkspace(::Type{C})

Reusable scratch for [`baseline_fringe_search`](@ref) at compute type `C`
(`ComplexF32` or `ComplexF64`, matching a visibility block's own eltype): the
zero-padded gridding buffer `G` and the FFT output `D`, lazily (re)allocated
when the padded grid size changes. Pass one per task to avoid allocating the
grid (tens of MB at `oversample = 8`) on every call — the search runs thousands
of times per solve, so reuse removes essentially all of its allocation/GC churn.
The FFT plan is not here: it is a pure function of the grid size (and `C`), so
the search-grid geometry carries it (built once per scan, shared read-only
across the workspaces that execute it).
"""
mutable struct FringeWorkspace{C, T}
    nf::Int
    nt::Int
    G::Matrix{C}
    D::Matrix{C}
    dwin::Vector{T}            # scratch for the windowed |D|² noise estimate
    mbd::Any                   # lazily-built `_MBDWorkspace{C}` for the hierarchical path
end
FringeWorkspace{C}() where {C} = FringeWorkspace{C, real(C)}(
    0, 0, Matrix{C}(undef, 0, 0), Matrix{C}(undef, 0, 0), real(C)[], nothing,
)
FringeWorkspace(::Type{C}) where {C} = FringeWorkspace{C}()

# Ensure `ws` is sized for an `nf × nt` grid at compute type `C` (reallocating
# only when the size changes). `nothing` makes a fresh workspace (single-call
# fallback); `C` must match an existing `ws`'s own compute type.
_ensure_workspace!(::Nothing, ::Type{C}, nf::Integer, nt::Integer) where {C} =
    _ensure_workspace!(FringeWorkspace(C), C, nf, nt)
function _ensure_workspace!(ws::FringeWorkspace{C}, ::Type{C}, nf::Integer, nt::Integer) where {C}
    if ws.nf != nf || ws.nt != nt
        ws.G = zeros(C, nf, nt)
        ws.D = similar(ws.G)
        ws.nf = Int(nf)
        ws.nt = Int(nt)
    end
    return ws
end

# The scan's shared FFT plan at compute type `C`. FFTW.ESTIMATE selects a plan
# by a fixed heuristic, with no timing-based benchmarking of the machine —
# unlike MEASURE, whose algorithm choice depends on wall-clock trials and can
# therefore pick a different transform (and different floating-point rounding)
# on different runs of the same problem size. The plan depends only on the
# padded grid size and `C`, so it is built ONCE per scan and shared read-only
# across every baseline/task: FFTW executes one plan concurrently across
# threads through out-of-place `mul!(D, plan, G)`, which never mutates the plan.
_plan_grid(::Type{C}, nf::Integer, nt::Integer) where {C} = plan_fft(zeros(C, nf, nt); flags = ESTIMATE)

# Uniform-grid descriptor for one axis: the origin, spacing, and grid length such
# that every sample `x` lands at `round((x - origin)/Δ) + 1 ∈ 1:n`. Δ is the
# median adjacent spacing (robust to gaps); n spans min→max. A length-1 axis is
# degenerate (Δ = 1, n = 1) — its conjugate (delay or rate) is not searched.
# Always Float64: `x` is an absolute physical (Hz, s) axis, and origin/step keep
# full precision regardless of the search's compute type `C` (see the header
# note on precision scope).
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
# stay type-stable). Always Float64 — see `_uniform_axis`.
const _Axis = @NamedTuple{origin::Float64, step::Float64, n::Int, degenerate::Bool}

# Per-(scan group) search-grid geometry: the frequency/time uniform-axis
# descriptors, the padded FFT sizes, and the conjugate delay/rate coordinate
# vectors (at compute precision `T`). These depend ONLY on (freqs, times,
# oversample, C) — identical across every (baseline, product) of a group — so
# the search builds them ONCE per group via `_search_axes` instead of re-sorting
# the freq/time axes (an O(nchan log nchan) sort of the same 1024 channels) and
# re-`collect`ing the two fftfreq vectors on each of the group's nbl×npol calls.
struct _SearchAxes{T, M}
    fax::_Axis
    tax::_Axis
    nf_pad::Int
    nt_pad::Int
    delays::Vector{T}
    rates::Vector{T}
    mbd::M                     # `_MBDAxes{T}` for the hierarchical path, else `nothing`
    plan::Any                  # full-grid FFT plan (at compute type C); `nothing` when `mbd` owns the plans
end

# The rate axis spans ±1/(2Δt); a window reaching into that wrap admits peaks
# indistinguishable from aliases of their own conjugate. The window is a
# configuration choice rather than a per-group event, hence `maxlog`.
function _check_rate_window(tax::_Axis, opts::FringeSearch)
    tax.degenerate && return nothing
    nyquist = 1 / (2 * tax.step)
    half = max(abs(opts.rate_window[1]), abs(opts.rate_window[2]))
    half > 0.8 * nyquist && @warn(
        "FringeSearch rate_window half-width $(round(half, sigdigits = 3)) Hz reaches " *
            "$(round(Int, 100 * half / nyquist))% of the ±$(round(nyquist, sigdigits = 3)) Hz Nyquist " *
            "rate of a $(round(tax.step, sigdigits = 3)) s integration — peaks near the wrap are aliases.",
        maxlog = 1,
    )
    return nothing
end

# False-alarm level at which an edge-pinned peak is worth reporting. This is a
# diagnostic level only; admission to the station solve is
# `Stationization.pfa_max`'s decision.
const _EDGE_WARN_PFA = 1.0e-4

"""
    _warn_edge_peaks(scube, opts, ax)

Warn when a peak strong enough to be a detection sits on the `rate_window`
boundary. The window, not the data, then chose that peak: the true fringe rate
lies outside it, and the station it belongs to is being hidden rather than
measured.
"""
function _warn_edge_peaks(scube, opts::FringeSearch, ax::_SearchAxes)
    ax.tax.degenerate && return nothing
    lo, hi = opts.rate_window
    (isfinite(lo) && isfinite(hi)) || return nothing
    drate = 1 / (ax.tax.step * ax.nt_pad)
    rate = scube[:rate]
    pfa = scube[:pfa]
    valid = scube[:valid]
    n = 0
    for i in eachindex(rate, pfa, valid)
        (valid[i] && pfa[i] <= _EDGE_WARN_PFA) || continue
        (rate[i] - lo <= drate || hi - rate[i] <= drate) && (n += 1)
    end
    n > 0 && @warn(
        "$n detection(s) peak within one grid step of the rate_window boundary " *
            "(±$(round(max(abs(lo), abs(hi)), sigdigits = 3)) Hz): their fringe rate lies outside the window.",
        maxlog = 1,
    )
    return nothing
end

function _search_axes(freqs::AbstractVector, times::AbstractVector, opts::FringeSearch, ::Type{C}) where {C}
    T = real(C)
    fax = _uniform_axis(freqs)
    tax = _uniform_axis(times)
    _check_rate_window(tax, opts)
    nf_pad = fax.degenerate ? 1 : _fast_fft_size(opts.oversample * fax.n)
    nt_pad = tax.degenerate ? 1 : _fast_fft_size(opts.oversample * tax.n)
    delays = fax.degenerate ? [zero(T)] : collect(fftfreq(nf_pad, T(1.0 / fax.step)))
    rates = tax.degenerate ? [zero(T)] : collect(fftfreq(nt_pad, T(1.0 / tax.step)))
    mbd = _maybe_mbd_axes(freqs, fax, tax, rates, opts, C)
    # The full-grid path executes `plan`; the hierarchical path carries its own
    # plans on `mbd` and never touches this one, so only build it when needed.
    plan = mbd === nothing ? _plan_grid(C, nf_pad, nt_pad) : nothing
    return _SearchAxes(fax, tax, nf_pad, nt_pad, delays, rates, mbd, plan)
end

"""
    baseline_fringe_search(V, W, freqs, times, f0, t0; opts = FringeSearch()) -> Detection

Search one block of visibilities `V[chan, time]` (with inverse-variance weights
`W[chan, time]`) for the group delay, fringe rate, and phase that align the
visibility phasor. `freqs` (Hz) and `times` (s) label the channel and time axes;
`f0`, `t0` are the delay/rate reference frequency and epoch (the returned phase
is referenced to them). Flagged samples (`W ≤ 0` or non-finite `V`) are dropped.

The search runs natively at `V`'s own compute type `C = eltype(V)` (`ComplexF32`
or `ComplexF64`; FFTW supports no other complex type) — a `ComplexF32` block
returns a `Detection{Float32}`.

`V`/`W` may also be passed as vectors for a single-time (`(nchan,)`, delay only)
or single-channel block (`times`-length, rate only).
"""
function baseline_fringe_search(
        V::AbstractMatrix{C}, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real;
        opts::FringeSearch = FringeSearch(),
        workspace::Union{Nothing, FringeWorkspace} = nothing,
    ) where {C}
    size(V) == size(W) || throw(
        DimensionMismatch("V is $(size(V)) and W is $(size(W)); they must have the same shape")
    )
    nchan, ntime = size(V)
    (nchan == length(freqs) && ntime == length(times)) || throw(
        DimensionMismatch(
            "V is $(size(V)); expected (length(freqs), length(times)) = " *
                "($(length(freqs)), $(length(times)))"
        )
    )
    ax = _search_axes(freqs, times, opts, C)
    # A standalone search is a family of one, so its own cell count is the family's.
    return _baseline_fringe_search(V, W, freqs, times, f0, t0, ax, workspace, opts, _search_cells(ax, opts))
end

# Grid the weighted visibilities onto the workspace's zero-padded uniform grid
# `ws.G` (zeroed here); returns Σw. Shared by the search hot path and the map
# extractor (`baseline_fringe_map`) so the gridding convention has one home.
function _grid_visibilities!(
        ws::FringeWorkspace{C}, V::AbstractMatrix, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, ax::_SearchAxes,
    ) where {C}
    nchan, ntime = size(V)
    fax = ax.fax
    tax = ax.tax
    nf_pad = ax.nf_pad
    nt_pad = ax.nt_pad
    G = ws.G
    fill!(G, zero(C))
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
        V::AbstractMatrix{C}, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        ax::_SearchAxes, workspace::Union{Nothing, FringeWorkspace}, opts::FringeSearch,
        family_cells::Real,
    ) where {C}
    # Hierarchical multi-band path (never allocates the big common-Δf grid).
    ax.mbd === nothing ||
        return _mbd_fringe_search(V, W, freqs, times, f0, t0, ax, workspace, opts, family_cells)

    T = real(C)
    nchan, ntime = size(V)
    fax = ax.fax
    tax = ax.tax
    nf_pad = ax.nf_pad
    nt_pad = ax.nt_pad

    # Reuse the workspace's gridding buffer (zeroed each call) and the scan's
    # shared FFT plan instead of allocating an `nf_pad × nt_pad` `C` grid and
    # replanning on every call.
    ws = _ensure_workspace!(workspace, C, nf_pad, nt_pad)
    Wsum = _grid_visibilities!(ws, V, W, freqs, times, ax)
    Wsum > 0 || return _invalid_detection(T)

    D = ws.D
    mul!(D, ax.plan, ws.G)

    # Conjugate-axis coordinates: delays (s) ↔ frequency grid, rates (Hz) ↔ time.
    # Precomputed once per group in `ax` (identical every call).
    delays = ax.delays
    rates = ax.rates

    # Windowed peak of |D|, plus Σ|D|² over the window for a data-driven noise
    # estimate (see SNR below).
    kbest = lbest = 0
    peakabs = -one(T)
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
    peakabs >= 0 || return _invalid_detection(T)

    # Noise estimate from the full |D|² plane (see `_plane_noise2!`).
    noise2 = _plane_noise2!(ws, Wsum)

    # Refine the peak on the EXACT matched filter (scalloping-free), seeded at the
    # FFT peak cell. This replaces the old on-grid 3-point parabola: the FFT only
    # LOCATES the main lobe, and `_polish_peak_exact!` finds the sub-cell peak of
    # the true objective — so accuracy no longer leans on a fine `oversample` grid.
    # It also reads φ and amplitude off the exact complex peak, referenced directly
    # to (f0, t0) with no FFT scalloping / grid-origin rotation:
    #     Dref = Σ w·V·exp(−2πi[delay·(f−f0) + rate·(t−t0)])
    # so amp = |Dref|/Σw, φ = angle(Dref).
    delay = delays[kbest]
    rate = rates[lbest]
    if opts.quad_interp
        pk = _polish_peak_exact!(
            V, W, freqs, times, f0, t0, delay, rate;
            delay_bin = fax.degenerate ? 0.0 : 1.0 / (nf_pad * fax.step),
            rate_bin = tax.degenerate ? 0.0 : 1.0 / (nt_pad * tax.step),
            refine_delay = !fax.degenerate, refine_rate = !tax.degenerate,
        )
        delay, rate, Dref = pk.delay, pk.rate, pk.Dref
    else
        Dref = _exact_matched_filter(V, W, freqs, times, f0, t0, delay, rate)
    end
    absref = abs(Dref)
    amp = absref / Wsum
    # Data-driven SNR: the matched-filter noise is estimated from the spread of
    # |D| over the search plane, robustly (median) so the bright peak and its
    # sidelobes don't bias it. For Rayleigh-distributed noise bins
    # `median(|D|²) = ln2 · mean(|D|²)`, and `mean(|D|²) = Σw` when the weights are
    # true inverse-variances — so this reduces to the matched-filter |Dref|/√Σw
    # for calibrated data, but stays correct when the WEIGHT column is
    # uncalibrated / uniform (common in raw correlator output), where √Σw badly
    # mis-scales the SNR and so its false-alarm probability. Falls back to √Σw if
    # the window is too small to estimate noise.
    snr = absref / sqrt(noise2)
    phase = rem2pi(angle(Dref), RoundNearest)

    return Detection{T}((delay, rate, phase, amp, snr, T(fringe_pfa(snr, family_cells)), true))
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
        throw(
            DimensionMismatch(
                "vector baseline_fringe_search: length(V)=$(length(V)) matches neither " *
                    "freqs ($(length(freqs))) nor times ($(length(times)))"
            )
        )
    end
end

# Exact (grid-free) matched filter Σ w·V·exp(−2πi[τ(f−f0) + ṙ(t−t0)]) at one
# (delay, rate) point — the scalloping-free evaluation shared by both search
# paths (final phase/amp readout, and the MBD ambiguity arbitration, which
# evaluates several candidates per search). The phase is separable, so the
# phasors are precomputed per axis: O(nchan + ntime) `cis` calls instead of
# O(nchan·ntime) — `cis` dominates the naive loop. The phase ANGLES
# (`freqs[ci] - f0`, an absolute-Hz difference) are computed in `Float64`
# regardless of `V`'s compute type `C` — see the header note on precision scope
# — but the phasors themselves, and the accumulator, are natively `C`.
function _exact_matched_filter(
        V::AbstractMatrix{C}, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        delay::Real, rate::Real,
    ) where {C}
    nchan, ntime = size(V)
    cf = Vector{C}(undef, nchan)
    @inbounds for ci in 1:nchan
        cf[ci] = cis(-2π * delay * (freqs[ci] - f0))
    end
    Dref = zero(C)
    @inbounds for ti in 1:ntime
        acc = zero(C)
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

# ── Separable evaluation: collapse one axis, then sweep the other ─────────────
#
# The matched-filter phase is separable, so collapsing the cube along one axis
# leaves a vector the CONJUGATE coordinate can be swept over in O(length):
#
#     S[c] = Σ_t w·V[c,t]·cis(−2π·ṙ(t−t0))   ⇒  D(τ, ṙ) = Σ_c S[c]·cis(−2π·τ(f_c−f0))
#     T[t] = Σ_c w·V[c,t]·cis(−2π·τ(f_c−f0)) ⇒  D(τ, ṙ) = Σ_t T[t]·cis(−2π·ṙ(t−t0))
#
# A coordinate-descent step therefore costs ONE sweep of the cube for the whole
# axis, not one per probe — which is what `_polish_peak_exact!` is built on.
# `_collapse_freq!` sums in `_exact_matched_filter`'s own order, so the two agree
# exactly; `_collapse_time!` sums time-major and agrees to float rounding.

function _collapse_time!(S, V::AbstractMatrix{C}, W, times, t0::Real, rate::Real) where {C}
    fill!(S, zero(C))
    @inbounds for ti in axes(V, 2)
        ph = cis(-2π * rate * (times[ti] - t0))
        for ci in axes(V, 1)
            w = W[ci, ti]
            v = V[ci, ti]
            (isfinite(w) && w > 0 && isfinite(v)) || continue
            S[ci] += w * v * ph
        end
    end
    return S
end

function _collapse_freq!(Tt, cf, V::AbstractMatrix{C}, W, freqs, f0::Real, delay::Real) where {C}
    @inbounds for ci in axes(V, 1)
        cf[ci] = cis(-2π * delay * (freqs[ci] - f0))
    end
    @inbounds for ti in axes(V, 2)
        acc = zero(C)
        for ci in axes(V, 1)
            w = W[ci, ti]
            v = V[ci, ti]
            (isfinite(w) && w > 0 && isfinite(v)) || continue
            acc += w * v * cf[ci]
        end
        Tt[ti] = acc
    end
    return Tt
end

# `Σ_k z[k]·cis(−2π·x·(coord[k] − origin))` — the collapsed cube evaluated at one
# conjugate coordinate.
function _phase_sum(z, coord, origin::Real, x::Real)
    D = zero(eltype(z))
    @inbounds for k in eachindex(z, coord)
        D += z[k] * cis(-2π * x * (coord[k] - origin))
    end
    return D
end

# Refine a coarse (delay, rate) peak by maximizing the EXACT matched filter
# |Σ w·V·exp(−2πi[τ(f−f0)+ṙ(t−t0)])| directly, rather than fitting a parabola to
# the coarse FFT |D| (whose bias grows with the grid cell size, i.e. shrinks with
# `oversample`). Four passes of a per-axis 3-point parabolic step evaluated on the
# exact objective, seeded at the FFT peak cell with a half-bin probe — the same
# coordinate-descent polish `_fit_band_dispersion` uses over its band phasors.
# Each axis is swept on its collapsed vector (`_collapse_time!`/`_collapse_freq!`),
# so a pass costs two sweeps of the cube rather than one per probe.
#
# Four passes shrink the probe bracket to `bin/32`, over which the main lobe
# (first null at ~1/B, i.e. `oversample` bins away) is quadratic to well within
# the noise; the accepted step is the parabola VERTEX, a continuous estimate, so
# the achieved resolution is not the bracket width.
# Optimizing the true objective can only raise |D|, so the refined SNR is ≥ the
# on-grid SNR; steps that leave the seed cell or head downhill are rejected, so a
# coarse (low-`oversample`) grid still seeds it safely. Returns the refined
# `(delay, rate, Dref)`. A degenerate axis passes `refine_* = false` (its conjugate
# coordinate is fixed at 0), so the polish reduces to a single exact evaluation.
# `delay`/`rate`/the bin widths are plain `Real`: they arrive at the search's
# compute precision `T` but the bin widths derive from the (always-`Float64`)
# axis spacing, so the refined values end up `Float64` — narrowed back to `T`
# only when the caller builds the final `Detection{T}`.
function _polish_peak_exact!(
        V, W, freqs, times, f0, t0, delay::Real, rate::Real;
        delay_bin::Real, rate_bin::Real,
        refine_delay::Bool, refine_rate::Bool,
    )
    Dref = _exact_matched_filter(V, W, freqs, times, f0, t0, delay, rate)
    # Per-axis probe half-width, shrunk geometrically each pass so the search hones
    # from the coarse seed cell down to well below the fringe resolution regardless
    # of `oversample`. The seed is the FFT argmax, so the true peak lies within
    # ±half a grid bin; probing ±bin/2 brackets it. Where the three probes are
    # concave we take the parabolic vertex, else step toward the taller side; the
    # step is CLAMPED to one probe width (a coarse-grid parabola can overshoot the
    # sinc peak) rather than rejected, so the point always walks toward the peak,
    # and only an uphill move is kept. Each axis' three probes and its centre are
    # read off the SAME collapsed vector, so the parabola is built from mutually
    # consistent values.
    hd = refine_delay ? delay_bin / 2 : 0.0
    hr = refine_rate ? rate_bin / 2 : 0.0
    (hd > 0 || hr > 0) || return (delay = delay, rate = rate, Dref = Dref)
    # Index-matched to the cube's own axes, so `_phase_sum`'s `eachindex(z, coord)`
    # checks each collapsed vector against the coordinate it is swept over.
    C = eltype(V)
    S = similar(V, C, (axes(V, 1),))
    Tt = similar(V, C, (axes(V, 2),))
    cf = similar(V, C, (axes(V, 1),))
    for _ in 1:4
        if hd > 0
            _collapse_time!(S, V, W, times, t0, rate)
            b0 = abs(_phase_sum(S, freqs, f0, delay))
            am = abs(_phase_sum(S, freqs, f0, delay - hd))
            ap = abs(_phase_sum(S, freqs, f0, delay + hd))
            den = am - 2 * b0 + ap
            δ = den < 0 ? clamp(0.5 * hd * (am - ap) / den, -hd, hd) : (ap > am ? hd : (am > ap ? -hd : 0.0))
            if δ != 0.0
                Dn = _phase_sum(S, freqs, f0, delay + δ)
                if abs(Dn) >= b0
                    Dref = Dn; delay += δ
                end
            end
            hd *= 0.5
        end
        if hr > 0
            _collapse_freq!(Tt, cf, V, W, freqs, f0, delay)
            b0 = abs(_phase_sum(Tt, times, t0, rate))
            am = abs(_phase_sum(Tt, times, t0, rate - hr))
            ap = abs(_phase_sum(Tt, times, t0, rate + hr))
            den = am - 2 * b0 + ap
            δ = den < 0 ? clamp(0.5 * hr * (am - ap) / den, -hr, hr) : (ap > am ? hr : (am > ap ? -hr : 0.0))
            if δ != 0.0
                Dn = _phase_sum(Tt, times, t0, rate + δ)
                if abs(Dn) >= b0
                    Dref = Dn; rate += δ
                end
            end
            hr *= 0.5
        end
    end
    return (delay = delay, rate = rate, Dref = Dref)
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
# by nfreqgroup small per-band FFTs plus nsbd tiny band-center FFTs — orders of
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
function _detect_freq_groups(freqs::AbstractVector, Δf::Real)
    freqgroups = UnitRange{Int}[]
    lo = 1
    for i in 1:(length(freqs) - 1)
        if freqs[i + 1] - freqs[i] > 1.5 * Δf
            push!(freqgroups, lo:i)
            lo = i + 1
        end
    end
    push!(freqgroups, lo:length(freqs))
    return freqgroups
end

# Precomputed hierarchical-search geometry (per scan group, like `_SearchAxes`).
# `f_lo`/`bc_step` are absolute band-origin (Hz) quantities and stay `Float64`,
# like `_Axis`'s `origin`/`step`; the SBD/MBD/rate coordinates (`ambig`,
# `sbd_bin`, `mbd_bin`, `sbd_val`, `mbd`, `rate_val`) are small FFT-conjugate
# values and track the search's compute precision `T`, like `_SearchAxes`'s
# `delays`/`rates`.
struct _MBDAxes{T}
    freqgroups::Vector{UnitRange{Int}}   # channel-index blocks (ascending frequency)
    f_lo::Vector{Float64}           # per-band grid origin (first channel freq)
    bc_bin::Vector{Int}             # band-origin bin on the Δbc grid (1-based)
    bc_step::Float64                # Δbc: common band-origin grid spacing
    nbc_pad::Int                    # padded band-center FFT length
    nfb_pad::Int                    # padded per-band (SBD) FFT length, shared
    ambig::T                        # MBD ambiguity A = 1/Δbc
    sbd_bin::T                      # SBD grid spacing 1/(nfb_pad·Δf)
    mbd_bin::T                      # MBD grid spacing A/nbc_pad
    sbd_idx::Vector{Int}            # SBD FFT rows kept (sorted by SBD value)
    sbd_val::Vector{T}              # SBD value per kept row
    mbd::Vector{T}                  # MBD value per band-center FFT row (FFT order)
    rate_idx::Vector{Int}           # rate FFT cols kept (sorted by value; window ±1)
    rate_val::Vector{T}             # rate value per kept col
    rate_scan::Vector{Int}          # positions in rate_idx inside the rate window
    planb::Any                      # per-band 2-D FFT plan (nfb_pad × nt_pad), compute type C
    planc::Any                      # stage-2 band-center FFT plan (nbc_pad × nrw, dim 1), compute type C
end

# Resolve `FringeSearch.algorithm` to a concrete algorithm. The `:auto` sentinel
# reads the frequency axis; an explicit algorithm passes through untouched.
_resolve_algorithm(alg::AbstractSearchAlgorithm, freqs, fax::_Axis) = alg
function _resolve_algorithm(alg::Symbol, freqs, fax::_Axis)
    alg === :auto || throw(ArgumentError(
            "FringeSearch: algorithm must be :auto or an AbstractSearchAlgorithm " *
                "(FullGrid(), HierarchicalMBD()); got :$(alg)"
        ))
    # Hierarchical only when the common grid is mostly padding (> 4× the real
    # channel count) — else the single FFT is simple and fast enough.
    hierarchical = !fax.degenerate && issorted(freqs) && fax.n > 4 * length(freqs)
    return hierarchical ? HierarchicalMBD() : FullGrid()
end

# The hierarchical band geometry for `alg`, or `nothing` to run the single
# full-grid FFT. The extension point for a custom `AbstractSearchAlgorithm`:
# dispatch is on the algorithm alone, so the remaining arguments stay
# unannotated and an out-of-package method is unambiguously more specific.
_mbd_axes(alg::AbstractSearchAlgorithm, freqs, fax, tax, rates, opts, ::Type{C}) where {C} =
    throw(ArgumentError(
        "FringeSearch: $(typeof(alg)) defines no `Gustavo.Fringe._mbd_axes` method"
    ))

_mbd_axes(::FullGrid, freqs, fax, tax, rates, opts, ::Type{C}) where {C} = nothing

function _mbd_axes(::HierarchicalMBD, freqs, fax, tax, rates, opts, ::Type{C}) where {C}
    (fax.degenerate || !issorted(freqs)) && return nothing
    freqgroups = _detect_freq_groups(freqs, fax.step)
    length(freqgroups) >= 2 || return nothing
    return _build_mbd_axes(freqs, freqgroups, fax, tax, rates, opts, C)
end

_maybe_mbd_axes(freqs::AbstractVector, fax::_Axis, tax::_Axis, rates::AbstractVector, opts::FringeSearch, ::Type{C}) where {C} =
    _mbd_axes(_resolve_algorithm(opts.algorithm, freqs, fax), freqs, fax, tax, rates, opts, C)

function _build_mbd_axes(
        freqs::AbstractVector, freqgroups::Vector{UnitRange{Int}},
        fax::_Axis, tax::_Axis, rates::AbstractVector{T}, opts::FringeSearch, ::Type{C},
    ) where {T, C}
    Δf = fax.step
    f_lo = [Float64(freqs[first(b)]) for b in freqgroups]
    maxbins = maximum(round(Int, (freqs[last(b)] - freqs[first(b)]) / Δf) + 1 for b in freqgroups)
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
    A = T(1.0 / Δbc)

    # SBD rows: window ± (A/2 + one bin) — the peak's SBD may sit half an
    # ambiguity from the true delay, and quad refinement needs a neighbour.
    sbd_all = fftfreq(nfb_pad, T(1.0 / Δf))
    sbd_bin = T(1.0 / (nfb_pad * Δf))
    lo, hi = opts.delay_window
    margin = A / 2 + sbd_bin
    sbd_idx = [k for k in eachindex(sbd_all) if (lo - margin) <= sbd_all[k] <= (hi + margin)]
    isempty(sbd_idx) && return nothing
    sort!(sbd_idx; by = k -> sbd_all[k])
    sbd_val = T[sbd_all[k] for k in sbd_idx]

    # Rate cols: the window bins plus one value-neighbour each side (for quad).
    ord = sortperm(rates)
    sel = [i for i in eachindex(ord) if _in_window(rates[ord[i]], opts.rate_window, tax.degenerate)]
    isempty(sel) && return nothing
    i1 = max(first(sel) - 1, 1)
    i2 = min(last(sel) + 1, length(ord))
    rate_idx = ord[i1:i2]
    rate_val = rates[rate_idx]
    rate_scan = collect((first(sel) - i1 + 1):(last(sel) - i1 + 1))

    mbd = collect(fftfreq(nbc_pad, T(1.0 / Δbc)))
    # Plans are pure functions of the padded band/band-center grid sizes (the rate
    # axis reuses the scan's `nt_pad`, which is `length(rates)`) and the compute
    # type `C`, so build them here, once per scan, and share them across the
    # workspaces (see `_plan_grid`).
    nt_pad = length(rates)
    planb = plan_fft(zeros(C, nfb_pad, nt_pad); flags = ESTIMATE)
    planc = plan_fft(zeros(C, nbc_pad, length(rate_idx)), 1; flags = ESTIMATE)
    return _MBDAxes{T}(
        freqgroups, f_lo, bc_bin, Δbc, nbc_pad, nfb_pad, A, sbd_bin, A / nbc_pad,
        sbd_idx, sbd_val, mbd, rate_idx, rate_val, rate_scan, planb, planc,
    )
end

# Scratch for the hierarchical path, hung off `FringeWorkspace.mbd` (lazily
# (re)built when the group geometry changes, like the main workspace). The
# stage-1/stage-2 FFT plans are carried by `_MBDAxes`, not here.
mutable struct _MBDWorkspace{C}
    nfb::Int
    nt::Int
    nfreqgroup::Int
    nsbd::Int
    nrw::Int
    nbc::Int
    Gb::Matrix{C}          # per-band gridding buffer (nfb_pad × nt_pad)
    Db::Matrix{C}
    X::Array{C, 3}         # stage-1 outputs, windowed: (nsbd, nrw, nfreqgroup)
    Mc::Matrix{C}          # stage-2 input (nbc_pad × nrw)
    Dc::Matrix{C}
end

function _ensure_mbd_workspace!(ws::FringeWorkspace{C}, mx::_MBDAxes, nt_pad::Int) where {C}
    nsbd = length(mx.sbd_idx)
    nrw = length(mx.rate_idx)
    nfreqgroup = length(mx.freqgroups)
    w = ws.mbd
    if !(w isa _MBDWorkspace{C}) || w.nfb != mx.nfb_pad || w.nt != nt_pad ||
            w.nfreqgroup != nfreqgroup || w.nsbd != nsbd || w.nrw != nrw || w.nbc != mx.nbc_pad
        Gb = zeros(C, mx.nfb_pad, nt_pad)
        Db = similar(Gb)
        X = Array{C, 3}(undef, nsbd, nrw, nfreqgroup)
        Mc = zeros(C, mx.nbc_pad, nrw)
        Dc = similar(Mc)
        w = _MBDWorkspace{C}(mx.nfb_pad, nt_pad, nfreqgroup, nsbd, nrw, mx.nbc_pad, Gb, Db, X, Mc, Dc)
        ws.mbd = w
    end
    return w::_MBDWorkspace{C}
end

# Stage 2 for one SBD row: scatter the bands onto the Δbc grid and FFT along it
# (dim 1) → the (MBD, rate) plane in `w.Dc`.
function _stage2_plane!(w::_MBDWorkspace{C}, mx::_MBDAxes, sj::Int) where {C}
    Mc = w.Mc
    fill!(Mc, zero(C))
    @inbounds for b in 1:w.nfreqgroup, rj in 1:w.nrw
        Mc[mx.bc_bin[b], rj] += w.X[sj, rj, b]
    end
    mul!(w.Dc, mx.planc, Mc)
    return w.Dc
end

# One (MBD row, rate col) value of the stage-2 sum for an arbitrary SBD row —
# the direct nfreqgroup-term sum, matching the FFT's index phase exactly. Used for
# quad refinement along the SBD axis without rebuilding whole planes.
function _stage2_value(w::_MBDWorkspace{C}, mx::_MBDAxes, sj::Int, m::Int, rj::Int) where {C}
    acc = zero(C)
    ph = -2π * (m - 1) / mx.nbc_pad
    @inbounds for b in 1:w.nfreqgroup
        acc += w.X[sj, rj, b] * cis(ph * (mx.bc_bin[b] - 1))
    end
    return acc
end

# The hierarchical search core (same contract as the full-path body of
# `_baseline_fringe_search`: identical Detection semantics and SNR/noise
# conventions).
function _mbd_fringe_search(
        V::AbstractMatrix{C}, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        ax::_SearchAxes, workspace::Union{Nothing, FringeWorkspace}, opts::FringeSearch,
        family_cells::Real,
    ) where {C}
    T = real(C)
    mx = ax.mbd
    tax = ax.tax
    nt_pad = ax.nt_pad
    ws = workspace === nothing ? FringeWorkspace(C) : workspace
    w = _ensure_mbd_workspace!(ws, mx, nt_pad)
    ntime = size(V, 2)
    Δf = ax.fax.step
    nfreqgroup = w.nfreqgroup

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
    for bi in 1:nfreqgroup
        Gb = w.Gb
        fill!(Gb, zero(C))
        flo = mx.f_lo[bi]
        @inbounds for ti in 1:ntime, ci in mx.freqgroups[bi]
            wgt = W[ci, ti]
            v = V[ci, ti]
            (isfinite(wgt) && wgt > 0 && isfinite(v)) || continue
            bf = round(Int, (freqs[ci] - flo) / Δf) + 1
            bt = tax.degenerate ? 1 : round(Int, (times[ti] - tax.origin) / tax.step) + 1
            (1 <= bf <= w.nfb && 1 <= bt <= nt_pad) || continue
            Gb[bf, bt] += wgt * v
            Wsum += wgt
        end
        mul!(w.Db, mx.planb, Gb)
        empty!(dwin)
        ntot = length(w.Db)
        stride = max(1, ntot ÷ max(64, 20000 ÷ nfreqgroup))
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
    Wsum > 0 || return _invalid_detection(T)
    noise2 = (nnoise == nfreqgroup && noise2 > 0) ? max(noise2, eps(Float64)) : Wsum

    # Stage 2: scan the (SBD, MBD, rate) cube for the windowed peak.
    nsbd = w.nsbd
    nbc = mx.nbc_pad
    A = mx.ambig
    lo, hi = opts.delay_window
    peak = -one(T)
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
    peak >= 0 || return _invalid_detection(T)

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

    # Final refinement on the EXACT matched filter (scalloping-free), shared with
    # the full path. The stage-2 FFT and the ambiguity arbitration above have
    # located the main lobe and its correct alias branch; `_polish_peak_exact!`
    # then finds the sub-cell (delay, rate) peak of the true objective. Its
    # per-axis half-bin probe re-centres any residual left by the Δbc-grid
    # quantization — what the old two-pass delay-only polish did, now also refining
    # rate on the exact objective instead of on the (biased) stage-2 grid.
    if opts.quad_interp
        pk = _polish_peak_exact!(
            V, W, freqs, times, f0, t0, delay, rate_ref;
            delay_bin = mx.mbd_bin,
            rate_bin = tax.degenerate ? 0.0 : 1.0 / (nt_pad * tax.step),
            refine_delay = true, refine_rate = !tax.degenerate,
        )
        delay, rate_ref, Dref = pk.delay, pk.rate, pk.Dref
    end

    absref = abs(Dref)
    amp = absref / Wsum
    snr = absref / sqrt(noise2)
    phase = rem2pi(angle(Dref), RoundNearest)
    return Detection{T}((delay, rate_ref, phase, amp, snr, T(fringe_pfa(snr, family_cells)), true))
end

# ── False-fringe statistics + the delay–rate map extractor ─────────────────────

# Effective number of INDEPENDENT search cells inside the (delay, rate) window.
# The delay resolution is 1/(gridded bandwidth span) = 1/(fax.n·fax.step) and the
# rate resolution 1/(time span), so cells-per-axis = window span / resolution,
# clamped to [1, n gridded samples] (zero-pad `oversample` refines the peak but
# adds NO independent cells). A degenerate axis contributes a factor 1.
_search_cells(ax::_SearchAxes, opts::FringeSearch) = _search_cells(ax.fax, ax.tax, opts)
_search_cells(freqs::AbstractVector, times::AbstractVector, opts::FringeSearch) =
    _search_cells(_uniform_axis(freqs), _uniform_axis(times), opts)

# Effective independent (delay, rate) cells inside the search window — the
# null-hypothesis trial count behind `fringe_pfa`. Depends only on the axis
# geometry and the windows, not on the FFT plan, so it is cheap to recompute
# from `freqs`/`times` without building a `_SearchAxes`.
function _search_cells(fax::_Axis, tax::_Axis, opts::FringeSearch)
    nd = fax.degenerate ? 1.0 :
        clamp((opts.delay_window[2] - opts.delay_window[1]) * fax.n * fax.step, 1.0, Float64(fax.n))
    nr = tax.degenerate ? 1.0 :
        clamp((opts.rate_window[2] - opts.rate_window[1]) * tax.n * tax.step, 1.0, Float64(tax.n))
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
    fringe_snr_cut(pfa, ncells) -> Float64

Inverse of [`fringe_pfa`](@ref) in `snr`: the SNR at which a search over
`ncells` independent (delay, rate) cells reaches false-alarm probability `pfa` —
i.e. the effective SNR detection threshold implied by a PFA gate
(`fringe_pfa(fringe_snr_cut(pfa, ncells), ncells) = pfa`). Because the cut only
grows as `√log(ncells/pfa)`, it moves slowly with both arguments: a PFA-gated
acceptance threshold is nearly flat across scans while still self-adjusting to
the search size. Returns `0.0` for `pfa ≥ 1` and `Inf` for `pfa ≤ 0`.
"""
function fringe_snr_cut(pfa::Real, ncells::Real)
    (isfinite(pfa) && isfinite(ncells)) || return NaN
    pfa >= 1 && return 0.0
    pfa <= 0 && return Inf
    p1 = -expm1(log1p(-float(pfa)) / max(ncells, 1.0))   # per-cell exceedance
    return sqrt(-log(p1))
end

"""
    FringeSearchMap

The windowed delay–rate matched-filter surface of one visibility block, produced
by [`baseline_fringe_map`](@ref) — the classic false-fringe diagnostic. Always
`Float64`-valued (a diagnostic/plotting artifact, unlike the precision-generic
[`baseline_fringe_search`](@ref)). Fields:

- `delays` (s) / `rates` (Hz) — the in-window grid coordinates, ascending.
- `snr` — `(ndelay, nrate)` map of `|D| / noise` in the SAME units as
  the detection's `snr`, so the map's peak sits at ≈ `detection.snr`.
- `detection` — the refined peak, exactly as [`baseline_fringe_search`](@ref)
  returns it (narrowed to `Float64` regardless of the search's own precision).
- `ncells` — effective number of independent search cells (see `fringe_pfa`).
- `pfa` — `fringe_pfa(detection.snr, ncells)` for this single search.
"""
struct FringeSearchMap{D, R, S, Det}
    delays::D
    rates::R
    snr::S
    detection::Det
    ncells::Float64
    pfa::Float64
end

# Narrow a `Detection{T}` to `Float64` for `FringeSearchMap`'s always-`Float64`
# fields (a diagnostic artifact, out of the search kernel's precision scope).
_detection64(d::Detection) =
    Detection{Float64}((Float64(d.delay), Float64(d.rate), Float64(d.phase), Float64(d.amp), Float64(d.snr), d.valid))

"""
    baseline_fringe_map(V, W, freqs, times, f0, t0; opts = FringeSearch(), workspace = nothing)
        -> FringeSearchMap

Compute the full delay–rate SNR surface for one visibility block — the
brute-force common-Δf grid/FFT ([`FullGrid`](@ref)), returning the windowed
`|D|` plane (in SNR units) instead of only the peak. The PLANE is always the
full grid — showing the complete sidelobe/alias structure is the point of the
diagnostic — while the embedded `detection` honours `opts.algorithm`, so it is
identical to what [`baseline_fringe_search`](@ref) (and the solver) returns. A
diagnostic (a full-grid FFT per call — on VGOS-style axes this single plane is
much more expensive than the hierarchical search that produced the solution),
not a hot path.
"""
function baseline_fringe_map(
        V::AbstractMatrix{C}, W::AbstractMatrix,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real;
        opts::FringeSearch = FringeSearch(),
        workspace::Union{Nothing, FringeWorkspace} = nothing,
    ) where {C}
    size(V) == size(W) || throw(
        DimensionMismatch("V is $(size(V)) and W is $(size(W)); they must have the same shape")
    )
    (size(V, 1) == length(freqs) && size(V, 2) == length(times)) || throw(
        DimensionMismatch(
            "V is $(size(V)); expected (length(freqs), length(times)) = " *
                "($(length(freqs)), $(length(times)))"
        )
    )
    ax = _search_axes(freqs, times, opts, C)              # detection axes (honour opts.algorithm)
    axf = ax.mbd === nothing ? ax :                       # plane axes: always the full grid
        _search_axes(freqs, times, FringeSearch(opts.delay_window, opts.rate_window, opts.oversample, opts.quad_interp, FullGrid()), C)
    ncells = _search_cells(axf, opts)
    ws = _ensure_workspace!(workspace, C, axf.nf_pad, axf.nt_pad)
    Wsum = _grid_visibilities!(ws, V, W, freqs, times, axf)
    Wsum > 0 || return FringeSearchMap(Float64[], Float64[], zeros(0, 0), _invalid_detection(Float64), ncells, NaN)
    D = ws.D
    mul!(D, axf.plan, ws.G)
    noise2 = _plane_noise2!(ws, Wsum)

    # In-window bins of each conjugate axis, in ascending coordinate order (the
    # fftfreq vectors are in FFT order [0, +…, −…]).
    kidx = [k for k in eachindex(axf.delays) if _in_window(axf.delays[k], opts.delay_window, axf.fax.degenerate)]
    lidx = [l for l in eachindex(axf.rates) if _in_window(axf.rates[l], opts.rate_window, axf.tax.degenerate)]
    sort!(kidx; by = k -> axf.delays[k])
    sort!(lidx; by = l -> axf.rates[l])
    inv_noise = 1.0 / sqrt(noise2)
    snrmap = abs.(D[kidx, lidx]) .* inv_noise

    # The refined peak, via the standard search (re-grids + re-FFTs the same data
    # in `ws` — the map above is already copied out, and reusing the search keeps
    # the peak/refinement logic in one place).
    det = _baseline_fringe_search(V, W, freqs, times, f0, t0, ax, ws, opts, ncells)
    return FringeSearchMap(axf.delays[kidx], axf.rates[lidx], snrmap, det, ncells, fringe_pfa(det.snr, ncells))
end
