# ── Per-baseline FFT delay/rate fringe search ────────────────────────────────
#
# The algorithm — the matched filter, its evaluation as a 2-D FFT over a
# uniform frequency × time grid, oversampling, multi-band aliasing and the
# hierarchical SBD → MBD decomposition — is derived in
# `docs/src/fringe_fitting.md`.
#
# Precision: the compute type `C` follows the visibility block's own eltype, so
# a `ComplexF32` caller's FFT runs natively in `Float32`. Axis origins and
# spacings (`_uniform_axis`'s `origin`/`step`, `_MBDAxes`'s `f_lo`/`bc_step`)
# and the `freqs`/`times` that `_exact_matched_filter` uses must stay `Float64`
# whatever `C` is: they are absolute physical values near 10¹¹ Hz, and
# `Float32`'s ~7 significant digits would quantize a GHz-scale origin to within
# a few kHz, enough to misround channel bins on fine spacing. Only relative
# quantities — grid reciprocals, FFT outputs, a coordinate's offset from its
# origin — narrow to `T`; every sum, phase and refined peak is at `T`.

"""
    AbstractSearchAlgorithm

How the delay search lays out its FFT grid. Built-ins: [`FullGrid`](@ref) and
[`HierarchicalMBD`](@ref).

A custom algorithm subtypes this and defines

    Gustavo.Fring._mbd_axes(alg, freqs, fax, tax, rates, opts, ::Type{C}) -> mbd_axes or nothing

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
(VGOS-style). Throws when the frequency axis cannot support it: a degenerate
or unsorted axis, fewer than two band blocks, band origins on no common grid
or sharing a grid bin, a band-origin grid over 65536 points, or no delay or
rate inside the search windows. `algorithm = :auto` runs the full grid in
those cases instead.
"""
struct HierarchicalMBD <: AbstractSearchAlgorithm end

"""
    FringeSearch(; delay_window, rate_window, oversample, quad_interp, algorithm)

Options for [`baseline_fringe_search`](@ref).

Every cell holding usable data yields a peak with its signal-to-noise and
false-alarm probability; no detection threshold is applied here — admission
to the station solve is [`Stationization`](@ref)'s decision, taken on the
recorded `pfa`.

- `delay_window`  : `(lo, hi)` delay search window in seconds. Default ±1 µs.
- `rate_window`   : `(lo, hi)` fringe-rate search window in Hz. Default ±0.8 Hz.
  Narrowing it buys no speed (the FFT spans the whole plane), only a smaller
  false-alarm trial count, while a window narrower than the true rate spread
  hides a station outright. Keep any choice well inside the ±1/(2Δt) Nyquist
  rate, past which a peak is an alias of its own wrap.
- `oversample`    : zero-padding factor per axis, equivalently the number of grid
  cells laid across the main lobe of the delay response. It sets which lobe the
  search finds, not how accurately the delay is then measured — `quad_interp`
  polishes the peak off-grid — so it only has to be fine enough to tell the main
  lobe from the alias lobes a gapped frequency axis puts beside it. `:auto`
  (default) therefore reads it off that axis: 2 for contiguous channels, rising
  to 8 as the bands sparsify toward the [`HierarchicalMBD`](@ref) crossover.
  An explicit positive `Int` overrides it; raise it if detections are landing on
  a multi-band alias, or lower it if the axis is contiguous and the search is the
  bottleneck. Cost is quadratic — the grid is `oversample·nchan ×
  oversample·ntime`, so 8 is ~16× the work of 2. [`_resolve_oversample`](@ref)
  gives the rule and the measurements behind its thresholds.
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
    oversample::Union{Symbol, Int} = :auto
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

`valid` says a peak was measured here, not that it passed any threshold: it is
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
# despite `valid = false` is rejected rather than admitted on a zero `pfa`.
_invalid_detection(::Type{T}) where {T} =
    Detection{T}((zero(T), zero(T), zero(T), zero(T), zero(T), one(T), false))

# The detection at the refined peak `Dref` of the matched filter, from the
# plane's total weight `Wsum`: each component of `D` has variance `Wsum` under
# noise, so `|Dref| / √Wsum` is the amplitude over its per-component σ.
function _detection(Dref, delay, rate, Wsum::T, family_cells::Real) where {T}
    absref = abs(Dref)
    snr = absref / sqrt(Wsum)
    phase = rem2pi(angle(Dref), RoundNearest)
    return Detection{T}((delay, rate, phase, absref / Wsum, snr, T(fringe_pfa(snr, family_cells)), true))
end

"""
    FringeWorkspace(::Type{C})

Reusable scratch for [`baseline_fringe_search`](@ref) at compute type `C`
(`ComplexF32` or `ComplexF64`, matching a visibility block's own eltype): the
zero-padded gridding buffer `G` and the FFT output `D`, lazily (re)allocated
when the padded grid size changes. Pass one per task to avoid allocating the
grid (tens of MB where `oversample` resolves high) on every call — the search runs thousands
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
    mbd::Any                   # lazily-built `_MBDWorkspace{C}` for the hierarchical path
    # The cell's gathered `(vis, weights, flags)` planes. Untyped: the weight and
    # flag eltypes come from the data, so the kernels take them through a
    # function barrier.
    planes::Any
end
FringeWorkspace{C}() where {C} = FringeWorkspace{C, real(C)}(
    0, 0, Matrix{C}(undef, 0, 0), Matrix{C}(undef, 0, 0), nothing, nothing,
)
FringeWorkspace(::Type{C}) where {C} = FringeWorkspace{C}()

# Ensure `ws` is sized for an `nf × nt` grid at compute type `C` (reallocating
# only when the size changes); `C` must match `ws`'s own compute type.
function _ensure_workspace!(ws::FringeWorkspace{C}, ::Type{C}, nf::Integer, nt::Integer) where {C}
    if ws.nf != nf || ws.nt != nt
        ws.G = zeros(C, nf, nt)
        ws.D = similar(ws.G)
        ws.nf = Int(nf)
        ws.nt = Int(nt)
    end
    return ws
end

# The scan's shared FFT plan at compute type `C`. `FFTW.ESTIMATE` selects a plan
# by a fixed heuristic, with no timing-based benchmarking of the machine —
# unlike `FFTW.MEASURE`, whose algorithm choice depends on wall-clock trials and can
# therefore pick a different transform (and different floating-point rounding)
# on different runs of the same problem size. The plan depends only on the
# padded grid size and `C`, so it is built once per scan and shared read-only
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
# vectors (at compute precision `T`). These depend only on (freqs, times,
# oversample, C), so `_search_axes` builds them once per scan group for every
# (antenna pair, feed pair) cell.
struct _SearchAxes{T, M, P}
    fax::_Axis
    tax::_Axis
    nf_pad::Int
    nt_pad::Int
    delays::Vector{T}
    rates::Vector{T}
    mbd::M                     # `_MBDAxes{T}` for the hierarchical path, else `nothing`
    plan::P                    # full-grid FFT plan (at compute type C); `nothing` when `mbd` owns the plans
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
    # Resolve the padding factor once per group, here at the boundary, so every
    # grid below reads a concrete Int (`_resolve_algorithm` resolves the same way).
    ropts = _with_oversample(opts, _resolve_oversample(opts.oversample, fax, length(freqs)))
    nf_pad = fax.degenerate ? 1 : _fast_fft_size(ropts.oversample * fax.n)
    nt_pad = tax.degenerate ? 1 : _fast_fft_size(ropts.oversample * tax.n)
    delays = fax.degenerate ? [zero(T)] : collect(fftfreq(nf_pad, T(1.0 / fax.step)))
    rates = tax.degenerate ? [zero(T)] : collect(fftfreq(nt_pad, T(1.0 / tax.step)))
    mbd = _maybe_mbd_axes(freqs, fax, tax, rates, ropts, C)
    # The full-grid path executes `plan`; the hierarchical path carries its own
    # plans on `mbd` and never touches this one, so only build it when needed.
    plan = isnothing(mbd) ? _plan_grid(C, nf_pad, nt_pad) : nothing
    return _SearchAxes(fax, tax, nf_pad, nt_pad, delays, rates, mbd, plan)
end

"""
    fringe_plane(V, W, freqs, times; flags = nothing) -> DimStack

Wrap one (baseline, correlation product) block into the `(Frequency, Ti)` stack
[`baseline_fringe_search`](@ref) takes: layers `:vis`, `:weights` and `:flags`,
with `freqs` (Hz) and `times` (s) as the two lookups. `flags` defaults to
nothing flagged.

`V`/`W`/`flags` may be vectors instead, for a single-time (`(nchan,)`, delay
only) or single-channel (`times`-length, rate only) block.

"""
function fringe_plane(V, W, freqs, times; flags = nothing)
    # The returned stack's lookups are `freqs`/`times` themselves, so the block
    # is read 1-based. Declared here, where the argument can be named: the
    # vector path reshapes, which would otherwise drop an offset silently.
    Base.require_one_based_indexing(V, W)
    isnothing(flags) || Base.require_one_based_indexing(flags)
    size(V) == size(W) || throw(
        DimensionMismatch("V is $(size(V)) and W is $(size(W)); they must have the same shape")
    )
    isnothing(flags) || size(flags) == size(V) || throw(
        DimensionMismatch("flags is $(size(flags)) and V is $(size(V)); they must have the same shape")
    )
    Vm, Wm, Fm = _as_plane_arrays(V, W, flags, freqs, times)
    d = (Frequency(collect(freqs)), Ti(collect(times)))
    return DimStack((
        vis = DimArray(Vm, d), weights = DimArray(Wm, d), flags = DimArray(Fm, d),
    ))
end

# Reshape a block to `(nchan, ntime)`. A vector is read as one time when its
# length matches `freqs` and as one channel when it matches `times`.
function _as_plane_arrays(V::AbstractMatrix, W, flags, freqs, times)
    (size(V, 1) == length(freqs) && size(V, 2) == length(times)) || throw(
        DimensionMismatch(
            "V is $(size(V)); expected (length(freqs), length(times)) = " *
                "($(length(freqs)), $(length(times)))"
        )
    )
    return V, W, isnothing(flags) ? falses(size(V)) : flags
end

function _as_plane_arrays(V::AbstractVector, W, flags, freqs, times)
    if length(V) == length(freqs) && length(times) == 1
        sz = (length(V), 1)
    elseif length(V) == length(times) && length(freqs) == 1
        sz = (1, length(V))
    else
        throw(
            DimensionMismatch(
                "vector fringe_plane: length(V)=$(length(V)) matches neither " *
                    "freqs ($(length(freqs))) nor times ($(length(times)))"
            )
        )
    end
    return reshape(V, sz), reshape(W, sz), isnothing(flags) ? falses(sz) : reshape(flags, sz)
end

"""
    baseline_fringe_search(plane, f0, t0; opts = FringeSearch()) -> Detection

Search one (baseline, correlation product) block for the group delay, fringe
rate, and phase that align the visibility phasor. `plane` is a `DimStack` on
`(Frequency, Ti)` carrying `:vis`, `:weights` (inverse-variance) and `:flags`,
such as one built from plain arrays by [`fringe_plane`](@ref). `f0`, `t0` are the delay/rate reference
frequency and epoch, and the returned phase is referenced to them.

A sample is used when its flag is unset and it carries usable statistics — a
finite visibility and a finite, positive weight. The two are independent: a
flag that a caller clears makes the sample available again at the weight it
already had.

The search runs natively at the visibility layer's own compute type
`C = eltype(plane[:vis])` (`ComplexF32` or `ComplexF64`; FFTW supports no other
complex type), so a `ComplexF32` block returns a `Detection{Float32}`.
"""
function baseline_fringe_search(
        plane::AbstractDimStack, f0::Real, t0::Real;
        opts::FringeSearch = FringeSearch(),
        workspace::Union{Nothing, FringeWorkspace} = nothing,
    )
    freqs, times = _plane_axes(plane)
    C = eltype(plane[:vis])
    ax = _search_axes(freqs, times, opts, C)
    # A standalone search is a family of one, so its own cell count is the family's.
    return _baseline_fringe_search(
        plane, freqs, times, f0, t0, ax, workspace, opts, _search_cells(ax, opts),
    )
end

# The plane's own frequency and time lookups, which label its two axes.
_plane_axes(plane) = (lookup(plane[:vis], Frequency), lookup(plane[:vis], Ti))

# The three layers every search kernel reads, in the order they are passed on,
# each as a plain `(Frequency, Ti)` matrix whatever order the plane is stored in:
# the kernels index them by position.
_plane_layers(plane) = map(
    L -> PermutedDimsArray(parent(L), dimnum(L, (Frequency, Ti))),
    (plane[:vis], plane[:weights], plane[:flags]),
)

# A plane's layers copied into `ws` frequency-fastest. The kernels sweep a plane
# many times, so one copy costs less than strided reads.
function _gather_plane!(ws::FringeWorkspace, plane)
    layers = _plane_layers(plane)
    bufs = _plane_buffers!(ws, layers, size(first(layers)))
    foreach(copyto!, bufs, layers)
    return bufs
end

function _plane_buffers!(ws::FringeWorkspace, layers, shape)
    fits(buf, L) = buf isa Matrix{eltype(L)} && size(buf) == shape
    bufs = ws.planes
    bufs isa NTuple{3, Matrix} && all(map(fits, bufs, layers)) && return bufs
    ws.planes = map(L -> Matrix{eltype(L)}(undef, shape), layers)
    return ws.planes
end

# The flag layer for the (baseline, product) plane under search, or `nothing`
# from a caller that has none. Which method applies follows from the argument
# type, so the test below costs nothing where there are no flags.
@inline _flagged(::Nothing, I...) = false
Base.@propagate_inbounds _flagged(F, I...) = F[I...]

# The kernels index the three planes with one set of loop variables, so their
# axes must agree. A caller with no flag layer passes `nothing`.
_check_plane_axes(V, W, ::Nothing) = check_layer_axes(V, W)
_check_plane_axes(V, W, F) = check_layer_axes(V, W, F)

# `x` labels dimension `d` of the plane `V`.
function _check_coord(V, d, x)
    axes(x, 1) == axes(V, d) || throw(
        DimensionMismatch("a coordinate vector has axes $(axes(x, 1)); dimension $d of the plane has $(axes(V, d))"),
    )
    return nothing
end

# `cis(−2π·x·(coord − origin))` at precision `T`. The offset is taken before
# narrowing: `coord` and `origin` are absolute (Hz, s) values.
@inline _phasor(::Type{T}, x, coord, origin) where {T} = cis(-2 * T(π) * T(x) * T(coord - origin))

# Grid the weighted visibilities of channels `rows` onto `G` (zeroed here), the
# frequency axis starting at `forigin` with spacing `fax.step`; returns `Wsum`
# plus their total weight. Samples off `G` are dropped.
function _grid!(
        G::AbstractMatrix{C}, V, W, F, freqs, times, rows, forigin::Real, fax::_Axis, tax::_Axis,
        Wsum::T,
    ) where {C, T}
    fill!(G, zero(C))
    for ti in axes(V, 2), ci in rows
        w = W[ci, ti]
        v = V[ci, ti]
        _usable(_flagged(F, ci, ti), w, v) || continue
        bf = fax.degenerate ? 1 : round(Int, (freqs[ci] - forigin) / fax.step) + 1
        bt = tax.degenerate ? 1 : round(Int, (times[ti] - tax.origin) / tax.step) + 1
        (1 <= bf <= size(G, 1) && 1 <= bt <= size(G, 2)) || continue
        G[bf, bt] += T(w) * v
        Wsum += T(w)
    end
    return Wsum
end

# The full-grid gridding onto `ws.G`, shared by the search and
# `baseline_fringe_map`; returns Σw.
function _grid_visibilities!(
        ws::FringeWorkspace{C, T}, V::AbstractMatrix, W::AbstractMatrix, F,
        freqs::AbstractVector, times::AbstractVector, ax::_SearchAxes,
    ) where {C, T}
    _check_plane_axes(V, W, F)
    _check_coord(V, 1, freqs)
    _check_coord(V, 2, times)
    return _grid!(ws.G, V, W, F, freqs, times, axes(V, 1), ax.fax.origin, ax.fax, ax.tax, zero(T))
end

# Core matched-filter search on a precomputed `_SearchAxes` — the hot path called
# once per (baseline, product). `baseline_fringe_search` above is the public,
# one-off wrapper that builds the axes then calls this; the group search builds the
# axes once and calls this directly for every baseline.
# The numeric kernels below work on the plane's three layers as plain arrays:
# they are swept thousands of times per solve and index by position, so the
# stack is destructured once here rather than at every call.
function _baseline_fringe_search(
        plane::AbstractDimStack,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        ax::_SearchAxes, workspace::Union{Nothing, FringeWorkspace}, opts::FringeSearch,
        family_cells::Real,
    )
    ws = something(workspace, FringeWorkspace(eltype(plane[:vis])))
    V, W, F = _gather_plane!(ws, plane)
    return _baseline_fringe_search(V, W, F, freqs, times, f0, t0, ax, ws, opts, family_cells)
end

function _baseline_fringe_search(
        V::AbstractMatrix, W::AbstractMatrix, F,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        ax::_SearchAxes, workspace::FringeWorkspace, opts::FringeSearch,
        family_cells::Real,
    )
    C = eltype(V)
    # Hierarchical multi-band path (never allocates the big common-Δf grid).
    isnothing(ax.mbd) ||
        return _mbd_fringe_search(V, W, F, freqs, times, f0, t0, ax, workspace, opts, family_cells)

    T = real(C)
    fax = ax.fax
    tax = ax.tax
    nf_pad = ax.nf_pad
    nt_pad = ax.nt_pad

    # Reuse the workspace's gridding buffer (zeroed each call) and the scan's
    # shared FFT plan instead of allocating an `nf_pad × nt_pad` `C` grid and
    # replanning on every call.
    ws = _ensure_workspace!(workspace, C, nf_pad, nt_pad)
    Wsum = _grid_visibilities!(ws, V, W, F, freqs, times, ax)
    Wsum > 0 || return _invalid_detection(T)

    D = ws.D
    mul!(D, ax.plan, ws.G)

    # Conjugate-axis coordinates: delays (s) ↔ frequency grid, rates (Hz) ↔ time.
    # Precomputed once per group in `ax` (identical every call).
    delays = ax.delays
    rates = ax.rates

    # Peak of |D| inside the search windows.
    kbest = lbest = 0
    peakabs = -one(T)
    kidx = [k for k in eachindex(delays) if _in_window(delays[k], opts.delay_window, fax.degenerate)]
    lidx = [l for l in eachindex(rates) if _in_window(rates[l], opts.rate_window, tax.degenerate)]
    for l in lidx, k in kidx
        a = abs(D[k, l])
        if a > peakabs
            peakabs = a
            kbest = k
            lbest = l
        end
    end
    peakabs >= 0 || return _invalid_detection(T)

    # Refine the peak on the exact matched filter (scalloping-free), seeded at the
    # FFT peak cell: the FFT only locates the main lobe, and `_polish_peak_exact!`
    # finds the sub-cell peak of the true objective, so accuracy does not rest on
    # a fine `oversample` grid. φ and amplitude are read off the exact complex
    # peak, referenced directly to (f0, t0) with no FFT scalloping or grid-origin
    # rotation:
    #     Dref = Σ w·V·exp(−2πi[delay·(f−f0) + rate·(t−t0)])
    # so amp = |Dref|/Σw, φ = angle(Dref).
    delay = delays[kbest]
    rate = rates[lbest]
    if opts.quad_interp
        pk = _polish_peak_exact!(
            V, W, F, freqs, times, f0, t0, delay, rate;
            delay_bin = fax.degenerate ? zero(T) : T(1 / (nf_pad * fax.step)),
            rate_bin = tax.degenerate ? zero(T) : T(1 / (nt_pad * tax.step)),
            refine_delay = !fax.degenerate, refine_rate = !tax.degenerate,
        )
        delay, rate, Dref = pk.delay, pk.rate, pk.Dref
    else
        Dref = _exact_matched_filter(V, W, F, freqs, times, f0, t0, delay, rate)
    end
    return _detection(Dref, delay, rate, Wsum, family_cells)
end

function _exact_matched_filter(
        V::AbstractMatrix{C}, W::AbstractMatrix, F,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        delay::Real, rate::Real,
    ) where {C}
    _check_plane_axes(V, W, F)
    _check_coord(V, 1, freqs)
    _check_coord(V, 2, times)
    T = real(C)
    cf = similar(V, C, (axes(V, 1),))
    for ci in eachindex(cf, freqs)
        cf[ci] = _phasor(T, delay, freqs[ci], f0)
    end
    Dref = zero(C)
    for ti in axes(V, 2)
        acc = zero(C)
        for ci in axes(V, 1)
            w = W[ci, ti]
            v = V[ci, ti]
            _usable(_flagged(F, ci, ti), w, v) || continue
            acc += T(w) * v * cf[ci]
        end
        Dref += acc * _phasor(T, rate, times[ti], t0)
    end
    return Dref
end

# ── Separable evaluation: collapse one axis, then sweep the other ───────────
#
# The matched-filter phase is separable, so collapsing the cube along one axis
# leaves a vector the conjugate coordinate can be swept over in O(length):
#
#     S[c] = Σ_t w·V[c,t]·cis(−2π·ṙ(t−t0))   ⇒  D(τ, ṙ) = Σ_c S[c]·cis(−2π·τ(f_c−f0))
#     T[t] = Σ_c w·V[c,t]·cis(−2π·τ(f_c−f0)) ⇒  D(τ, ṙ) = Σ_t T[t]·cis(−2π·ṙ(t−t0))
#
# A coordinate-descent step therefore costs one sweep of the cube for the whole
# axis rather than one per probe, which is what `_polish_peak_exact!` is built
# on. `_collapse_freq!` sums in `_exact_matched_filter`'s own order, so the two
# agree exactly; `_collapse_time!` sums time-major and agrees to float rounding.

function _collapse_time!(S, V::AbstractMatrix{C}, W, F, times, t0::Real, rate::Real) where {C}
    _check_plane_axes(V, W, F)
    _check_coord(V, 1, S)
    _check_coord(V, 2, times)
    T = real(C)
    fill!(S, zero(C))
    for ti in axes(V, 2)
        ph = _phasor(T, rate, times[ti], t0)
        for ci in axes(V, 1)
            w = W[ci, ti]
            v = V[ci, ti]
            _usable(_flagged(F, ci, ti), w, v) || continue
            S[ci] += T(w) * v * ph
        end
    end
    return S
end

function _collapse_freq!(Tt, cf, V::AbstractMatrix{C}, W, F, freqs, f0::Real, delay::Real) where {C}
    _check_plane_axes(V, W, F)
    _check_coord(V, 1, freqs)
    _check_coord(V, 1, cf)
    _check_coord(V, 2, Tt)
    T = real(C)
    for ci in axes(V, 1)
        cf[ci] = _phasor(T, delay, freqs[ci], f0)
    end
    for ti in axes(V, 2)
        acc = zero(C)
        for ci in axes(V, 1)
            w = W[ci, ti]
            v = V[ci, ti]
            _usable(_flagged(F, ci, ti), w, v) || continue
            acc += T(w) * v * cf[ci]
        end
        Tt[ti] = acc
    end
    return Tt
end

# `Σ_k z[k]·cis(−2π·x·(coord[k] − origin))` — the collapsed cube evaluated at one
# conjugate coordinate.
function _phase_sum(z, coord, origin::Real, x::Real)
    T = real(eltype(z))
    D = zero(eltype(z))
    for k in eachindex(z, coord)
        D += z[k] * _phasor(T, x, coord[k], origin)
    end
    return D
end

# Refine a coarse (delay, rate) peak by maximizing the exact matched filter
# |Σ w·V·exp(−2πi[τ(f−f0)+ṙ(t−t0)])| directly, rather than fitting a parabola to
# the coarse FFT |D|, whose bias grows with the grid cell size. Four passes of a
# per-axis 3-point parabolic step on the exact objective, seeded at the FFT peak
# cell with a half-bin probe. Each axis is swept on its collapsed vector
# (`_collapse_time!`/`_collapse_freq!`), so a pass costs two sweeps of the cube
# rather than one per probe.
#
# Four passes shrink the probe bracket to `bin/32`, over which the main lobe is
# quadratic to well within the noise; the accepted step is the parabola vertex,
# a continuous estimate, so the achieved resolution is not the bracket width.
# Optimizing the true objective can only raise |D|, so the refined SNR is at
# least the on-grid SNR; steps that leave the seed cell or head downhill are
# rejected, so a coarse grid still seeds it safely. A degenerate axis passes
# `refine_* = false`, its conjugate coordinate fixed at 0, and the polish
# reduces to a single exact evaluation. Returns `(delay, rate, Dref)` at the
# compute precision `real(eltype(V))`.
function _polish_peak_exact!(
        V, W, F, freqs, times, f0, t0, delay0::Real, rate0::Real;
        delay_bin::Real, rate_bin::Real,
        refine_delay::Bool, refine_rate::Bool,
    )
    T = real(eltype(V))
    delay, rate = T(delay0), T(rate0)
    Dref = _exact_matched_filter(V, W, F, freqs, times, f0, t0, delay, rate)
    # Per-axis probe half-width, shrunk geometrically each pass so the search
    # hones from the coarse seed cell to well below the fringe resolution
    # whatever `oversample` is. The seed is the FFT argmax, so the true peak lies
    # within ±half a grid bin and probing ±bin/2 brackets it. Where the three
    # probes are concave the parabolic vertex is taken, else a step toward the
    # taller side; the step is clamped to one probe width rather than rejected,
    # since a coarse-grid parabola can overshoot the sinc peak, so the point
    # always walks toward the peak and only an uphill move is kept. Each axis'
    # three probes and its center are read off the same collapsed vector, so the
    # parabola is built from mutually consistent values.
    hd = refine_delay ? T(delay_bin) / 2 : zero(T)
    hr = refine_rate ? T(rate_bin) / 2 : zero(T)
    (hd > 0 || hr > 0) || return (delay = delay, rate = rate, Dref = Dref)
    # Index-matched to the cube's own axes, so `_phase_sum`'s `eachindex(z, coord)`
    # checks each collapsed vector against the coordinate it is swept over.
    C = eltype(V)
    S = similar(V, C, (axes(V, 1),))
    Tt = similar(V, C, (axes(V, 2),))
    cf = similar(V, C, (axes(V, 1),))
    for _ in 1:4
        if hd > 0
            _collapse_time!(S, V, W, F, times, t0, rate)
            b0 = abs(_phase_sum(S, freqs, f0, delay))
            am = abs(_phase_sum(S, freqs, f0, delay - hd))
            ap = abs(_phase_sum(S, freqs, f0, delay + hd))
            δ = _probe_step(am, b0, ap, hd)
            if !iszero(δ)
                Dn = _phase_sum(S, freqs, f0, delay + δ)
                if abs(Dn) >= b0
                    Dref = Dn; delay += δ
                end
            end
            hd /= 2
        end
        if hr > 0
            _collapse_freq!(Tt, cf, V, W, F, freqs, f0, delay)
            b0 = abs(_phase_sum(Tt, times, t0, rate))
            am = abs(_phase_sum(Tt, times, t0, rate - hr))
            ap = abs(_phase_sum(Tt, times, t0, rate + hr))
            δ = _probe_step(am, b0, ap, hr)
            if !iszero(δ)
                Dn = _phase_sum(Tt, times, t0, rate + δ)
                if abs(Dn) >= b0
                    Dref = Dn; rate += δ
                end
            end
            hr /= 2
        end
    end
    return (delay = delay, rate = rate, Dref = Dref)
end

# The polish's step from probes `am, b0, ap` at `−h, 0, +h`: the parabola vertex
# where they are concave, else `h` toward the taller side.
function _probe_step(am, b0, ap, h)
    den = am - 2 * b0 + ap
    den < 0 && return clamp(h * (am - ap) / (2 * den), -h, h)
    return ap > am ? h : am > ap ? -h : zero(h)
end

# 3-point quadratic vertex offset (in bins) given neighbor magnitudes `ym, y0,
# yp` straddling the peak `y0`. Returns 0 if the curvature is non-concave.
function _quad_offset(ym::Real, y0::Real, yp::Real)
    denom = ym - 2y0 + yp
    denom < 0 || return zero(denom)
    δ = (ym - yp) / (2 * denom)
    return clamp(δ, -one(δ) / 2, one(δ) / 2)
end

_wrap(i::Int, n::Int) = mod(i - 1, n) + 1

# Window test; a degenerate axis (single sample) always passes (its conjugate
# coordinate is identically 0).
_in_window(x::Real, window::Tuple{<:Real, <:Real}, degenerate::Bool) =
    degenerate || (window[1] <= x <= window[2])

# ── Hierarchical SBD → MBD search (HOPS/fourfit-style) ───────────────────────
#
# The band factorization, the two FFT stages, the MBD ambiguity and how the
# exact matched filter arbitrates it are derived in
# `docs/src/fringe_fitting.md`. Noise and SNR conventions are identical to the
# full-grid path: every cube cell is a matched-filter output with variance Σw
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

# Split a sorted frequency axis into contiguous band blocks: a new block starts
# wherever the spacing exceeds 1.5× the in-band spacing Δf.
function _detect_freq_groups(freqs::AbstractVector, Δf::Real)
    # The returned ranges index the stacked channel axis, which is 1-based.
    Base.require_one_based_indexing(freqs)
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
struct _MBDAxes{T, PB, PC}
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
    planb::PB                       # per-band 2-D FFT plan (nfb_pad × nt_pad), compute type C
    planc::PC                       # stage-2 band-center FFT plan (nbc_pad × nrw, dim 1), compute type C
end

# Resolve `FringeSearch.algorithm` to a concrete algorithm. The `:auto` sentinel
# reads the frequency axis; an explicit algorithm passes through untouched.
_resolve_algorithm(alg::AbstractSearchAlgorithm, freqs, fax::_Axis) = alg
function _resolve_algorithm(alg::Symbol, freqs, fax::_Axis)
    alg === :auto || throw(
        ArgumentError(
            "FringeSearch: algorithm must be :auto or an AbstractSearchAlgorithm " *
                "(FullGrid(), HierarchicalMBD()); got :$(alg)"
        )
    )
    # Hierarchical only when the common grid is mostly padding (> 4× the real
    # channel count) — else the single FFT is simple and fast enough.
    hierarchical = !fax.degenerate && issorted(freqs) && fax.n > 4 * length(freqs)
    return hierarchical ? HierarchicalMBD() : FullGrid()
end

"""
    _resolve_oversample(oversample, fax, nchan) -> Int

The zero-padding factor to build the search grid at: an explicit positive `Int`
passes through, `:auto` reads it off the frequency axis.

The padded grid's delay cell is `1/(oversample·n·Δf)` and the main lobe of the
synthesized delay response is about `1/(n·Δf)` wide, so `oversample` is the
number of cells laid across the main lobe. One cell never suffices: the search
reports whichever cell is highest, and the sampled peak is scalloped away from
the true one by up to half a cell.

How finely the lobe must be sampled depends on what it is competing against.
Contiguous channels put a single lobe in the delay window, and scalloping costs
only accuracy — which `_polish_peak_exact!` then recovers off-grid. Channels
gathered into separated bands put a comb of alias lobes beside it, spaced by the
reciprocal of the band-origin spacing, and a scalloped main lobe can be reported
below one of them; the polish cannot undo that, because it refines whichever
lobe it was seeded in. The comb's density tracks `sparsity` — the common grid's
size over the real channel count, the same quantity `_resolve_algorithm` reads
to hand a very gapped axis to [`HierarchicalMBD`](@ref) — so the sampling
requirement tracks it too.

The thresholds come from injected fringes scored on how often the recovered
delay lands on the true lobe, swept over band sparsity and signal-to-noise: 2
cells hold to sparsity ≈ 1.5; 4 is indistinguishable from 8 up to sparsity ≈ 3;
only at the `HierarchicalMBD` crossover (sparsity 4) does 8 measurably beat 4,
and by 1–3 points. Below 2 cells the identification collapses — to 20–50% — at
every signal-to-noise, because it is a grid failure rather than a noise one.

`:auto` is a pure function of the frequency axis, so a recorded `FringeSearch`
still determines the grid: replaying it on the same data resolves identically.
[`HierarchicalMBD`](@ref) floors its own band and band-center grids at 4
regardless, and takes only its rate axis from this.
"""
function _resolve_oversample(oversample::Symbol, fax::_Axis, nchan::Integer)
    oversample === :auto || throw(
        ArgumentError(
            "FringeSearch: oversample must be :auto or a positive Int; got :$(oversample)"
        )
    )
    (fax.degenerate || nchan <= 0) && return 2
    sparsity = fax.n / nchan
    return sparsity < 2 ? 2 : sparsity < 3 ? 4 : 8
end
function _resolve_oversample(oversample::Integer, ::_Axis, ::Integer)
    oversample >= 1 || throw(
        ArgumentError("FringeSearch: oversample must be ≥ 1; got $(oversample)")
    )
    return Int(oversample)
end

# `s` with `oversample` resolved to the Int the grid is actually built at, so
# everything below `_search_axes` reads a concrete factor.
_with_oversample(s::FringeSearch, oversample::Int) = FringeSearch(;
    s.delay_window, s.rate_window, oversample, s.quad_interp, s.algorithm,
)

# The hierarchical band geometry for `alg`, or `nothing` to run the single
# full-grid FFT. The extension point for a custom `AbstractSearchAlgorithm`:
# dispatch is on the algorithm alone, so the remaining arguments stay
# unannotated and an out-of-package method is unambiguously more specific.
_mbd_axes(alg::AbstractSearchAlgorithm, freqs, fax, tax, rates, opts, ::Type{C}) where {C} =
    throw(
    ArgumentError(
        "FringeSearch: $(typeof(alg)) defines no `Gustavo.Fring._mbd_axes` method"
    )
)

_mbd_axes(::FullGrid, freqs, fax, tax, rates, opts, ::Type{C}) where {C} = nothing

function _mbd_axes(::HierarchicalMBD, freqs, fax, tax, rates, opts, ::Type{C}) where {C}
    mbd = _hierarchical_axes(freqs, fax, tax, rates, opts, C)
    mbd isa AbstractString && throw(
        ArgumentError("HierarchicalMBD cannot search this scan: $mbd. Use FullGrid() or algorithm = :auto.")
    )
    return mbd
end

# The hierarchical band geometry, or the reason the axis cannot support one.
function _hierarchical_axes(freqs, fax, tax, rates, opts, ::Type{C}) where {C}
    fax.degenerate && return "the frequency axis is degenerate"
    issorted(freqs) || return "the channel frequencies are not sorted"
    freqgroups = _detect_freq_groups(freqs, fax.step)
    length(freqgroups) >= 2 || return "the channels form $(length(freqgroups)) band block(s), not at least two"
    return _build_mbd_axes(freqs, freqgroups, fax, tax, rates, opts, C)
end

# `:auto` runs the full grid wherever the hierarchical search cannot run.
function _maybe_mbd_axes(freqs::AbstractVector, fax::_Axis, tax::_Axis, rates::AbstractVector, opts::FringeSearch, ::Type{C}) where {C}
    alg = _resolve_algorithm(opts.algorithm, freqs, fax)
    (opts.algorithm === :auto && alg isa HierarchicalMBD) || return _mbd_axes(alg, freqs, fax, tax, rates, opts, C)
    mbd = _hierarchical_axes(freqs, fax, tax, rates, opts, C)
    return mbd isa AbstractString ? nothing : mbd
end

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
    (Δbc > 0 && maximum(abs(o - round(o / Δbc) * Δbc) for o in offs) <= tol) ||
        return "the band origins lie on no common grid within $(tol) Hz"
    bc_bin = [round(Int, o / Δbc) + 1 for o in offs]
    allunique(bc_bin) || return "two band origins fall in one band-origin grid bin"
    nbc_pad = _fast_fft_size(osb * maximum(bc_bin))
    # A near-continuum of origin bins means the hierarchy buys nothing (the
    # band-center FFT approaches the full grid).
    nbc_pad <= 65536 || return "the band-origin grid needs $nbc_pad points, over 65536"
    A = T(1.0 / Δbc)

    # SBD rows: window ± (A/2 + one bin) — the peak's SBD may sit half an
    # ambiguity from the true delay, and quad refinement needs a neighbor.
    sbd_all = fftfreq(nfb_pad, T(1.0 / Δf))
    sbd_bin = T(1.0 / (nfb_pad * Δf))
    lo, hi = opts.delay_window
    margin = A / 2 + sbd_bin
    sbd_idx = [k for k in eachindex(sbd_all) if (lo - margin) <= sbd_all[k] <= (hi + margin)]
    isempty(sbd_idx) && return "no single-band delay falls inside the delay window"
    sort!(sbd_idx; by = k -> sbd_all[k])
    sbd_val = T[sbd_all[k] for k in sbd_idx]

    # Rate cols: the window bins plus one value-neighbor each side (for quad).
    ord = sortperm(rates)
    sel = [i for i in eachindex(ord) if _in_window(rates[ord[i]], opts.rate_window, tax.degenerate)]
    isempty(sel) && return "no rate falls inside the rate window"
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
    return _MBDAxes(
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
    for b in axes(w.X, 3), rj in axes(w.X, 2)
        Mc[mx.bc_bin[b], rj] += w.X[sj, rj, b]
    end
    mul!(w.Dc, mx.planc, Mc)
    return w.Dc
end

# One (MBD row, rate col) value of the stage-2 sum for an arbitrary SBD row —
# the direct nfreqgroup-term sum, matching the FFT's index phase exactly. Used for
# quad refinement along the SBD axis without rebuilding whole planes.
function _stage2_value(w::_MBDWorkspace{C}, mx::_MBDAxes, sj::Int, m::Int, rj::Int) where {C}
    T = real(C)
    acc = zero(C)
    ph = -2π * (m - 1) / mx.nbc_pad
    for b in axes(w.X, 3)
        acc += w.X[sj, rj, b] * cis(T(ph * (mx.bc_bin[b] - 1)))
    end
    return acc
end

# The hierarchical search core (same contract as the full-path body of
# `_baseline_fringe_search`: identical Detection semantics and SNR/noise
# conventions).
function _mbd_fringe_search(
        V::AbstractMatrix{C}, W::AbstractMatrix, F,
        freqs::AbstractVector, times::AbstractVector, f0::Real, t0::Real,
        ax::_SearchAxes, ws::FringeWorkspace, opts::FringeSearch,
        family_cells::Real,
    ) where {C}
    _check_plane_axes(V, W, F)
    _check_coord(V, 1, freqs)
    _check_coord(V, 2, times)
    T = real(C)
    mx = ax.mbd
    tax = ax.tax
    nt_pad = ax.nt_pad
    w = _ensure_mbd_workspace!(ws, mx, nt_pad)
    nfreqgroup = w.nfreqgroup

    # Stage 1: per band, grid + 2-D FFT (in-band delay × rate); keep the windowed
    # (SBD row, rate col) block.
    Wsum = zero(T)
    for bi in eachindex(mx.freqgroups)
        Wsum = _grid!(w.Gb, V, W, F, freqs, times, mx.freqgroups[bi], mx.f_lo[bi], ax.fax, tax, Wsum)
        mul!(w.Db, mx.planb, w.Gb)
        for (rj, l) in zip(axes(w.X, 2), mx.rate_idx),
                (sj, k) in zip(axes(w.X, 1), mx.sbd_idx)
            w.X[sj, rj, bi] = w.Db[k, l]
        end
    end
    Wsum > 0 || return _invalid_detection(T)

    # Stage 2: scan the (SBD, MBD, rate) cube for the windowed peak.
    nsbd = w.nsbd
    nbc = mx.nbc_pad
    A = mx.ambig
    lo, hi = opts.delay_window
    peak = -one(T)
    ps = pm = pr = 0
    for sj in eachindex(mx.sbd_val)
        Dc = _stage2_plane!(w, mx, sj)
        sv = mx.sbd_val[sj]
        for rj in mx.rate_scan, m in axes(Dc, 1)
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
    # quad-refine MBD (periodic → wrapped neighbors), rate, and SBD.
    Dc = _stage2_plane!(w, mx, ps)
    mbd_ref = mx.mbd[pm]
    rate_ref = mx.rate_val[pr]
    sbd_ref = mx.sbd_val[ps]
    rate_bin = tax.degenerate ? zero(T) : T(1 / (nt_pad * tax.step))
    if opts.quad_interp
        ym = abs(Dc[_wrap(pm - 1, nbc), pr])
        yp = abs(Dc[_wrap(pm + 1, nbc), pr])
        mbd_ref += _quad_offset(ym, peak, yp) * mx.mbd_bin
        if !tax.degenerate && 1 < pr < w.nrw
            rate_ref += _quad_offset(abs(Dc[pm, pr - 1]), peak, abs(Dc[pm, pr + 1])) * rate_bin
        end
        if 1 < ps < nsbd
            ym = abs(_stage2_value(w, mx, ps - 1, pm, pr))
            yp = abs(_stage2_value(w, mx, ps + 1, pm, pr))
            sbd_ref += _quad_offset(ym, peak, yp) * mx.sbd_bin
        end
    end

    # Ambiguity arbitration: candidate total delays mbd_ref + k·A near the SBD
    # estimate (within ±1.5 SBD bins, inside the window), decided by the exact
    # matched filter — it sees the true in-band slope, which is what separates
    # the aliases. Usually one candidate (A ≫ sbd_bin); a handful otherwise.
    half = 1.5 * mx.sbd_bin
    kmin = ceil(Int, (max(lo, sbd_ref - half) - mbd_ref) / A)
    kmax = floor(Int, (min(hi, sbd_ref + half) - mbd_ref) / A)
    if kmin > kmax
        kmin = kmax = round(Int, (clamp(sbd_ref, lo, hi) - mbd_ref) / A)
    end
    delay = mbd_ref + kmin * A
    Dref = _exact_matched_filter(V, W, F, freqs, times, f0, t0, delay, rate_ref)
    for k in (kmin + 1):kmax
        d = mbd_ref + k * A
        Dk = _exact_matched_filter(V, W, F, freqs, times, f0, t0, d, rate_ref)
        if abs(Dk) > abs(Dref)
            Dref = Dk
            delay = d
        end
    end

    # Final refinement on the exact matched filter (scalloping-free), shared with
    # the full path. The stage-2 FFT and the ambiguity arbitration above have
    # located the main lobe and its correct alias branch; `_polish_peak_exact!`
    # then finds the sub-cell (delay, rate) peak of the true objective, its
    # half-bin probes re-centering any residual left by the Δbc-grid
    # quantization, and refines the rate on the exact objective rather than the
    # biased stage-2 grid.
    if opts.quad_interp
        pk = _polish_peak_exact!(
            V, W, F, freqs, times, f0, t0, delay, rate_ref;
            delay_bin = mx.mbd_bin,
            rate_bin,
            refine_delay = true, refine_rate = !tax.degenerate,
        )
        delay, rate_ref, Dref = pk.delay, pk.rate, pk.Dref
    end

    return _detection(Dref, delay, rate_ref, Wsum, family_cells)
end

# ── False-fringe statistics + the delay–rate map extractor ─────────────────────

# Effective number of independent search cells inside the (delay, rate) window.
# The delay resolution is 1/(gridded bandwidth span) = 1/(fax.n·fax.step) and the
# rate resolution 1/(time span), so cells-per-axis = window span / resolution,
# clamped to [1, n gridded samples] (zero-pad `oversample` refines the peak but
# adds no independent cells). A degenerate axis contributes a factor 1.
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
of at least `snr` (HOPS-style). With `snr` the amplitude over its per-component
σ (`|D| / √Σw` for weights that are inverse variances per real component), a
noise cell exceeds `s` with probability `exp(−s²/2)`, so

    pfa = 1 − (1 − exp(−snr²/2))^ncells

evaluated stably (`≈ ncells·exp(−snr²/2)` when small). See the "Detection
statistics" section of the fringe-fitting manual for the derivation. `pfa ≪ 1` marks a secure
detection; `pfa ≳ 0.01` means the peak is consistent with the noise sidelobe
forest — a likely FALSE fringe. Returns `NaN` for non-finite inputs.
"""
function fringe_pfa(snr::Real, ncells::Real)
    (isfinite(snr) && isfinite(ncells)) || return NaN
    snr <= 0 && return 1.0
    p1 = exp(-float(snr)^2 / 2)                # single-cell exceedance
    p1 >= 1 && return 1.0
    return -expm1(max(ncells, 1.0) * log1p(-p1))
end

"""
    fringe_snr_cut(pfa, ncells) -> Float64

Inverse of [`fringe_pfa`](@ref) in `snr`: the SNR at which a search over
`ncells` independent (delay, rate) cells reaches false-alarm probability `pfa` —
i.e. the effective SNR detection threshold implied by a PFA gate
(`fringe_pfa(fringe_snr_cut(pfa, ncells), ncells) = pfa`). Because the cut only
grows as `√(2 log(ncells/pfa))`, it moves slowly with both arguments: a PFA-gated
acceptance threshold is nearly flat across scans while still self-adjusting to
the search size. Returns `0.0` for `pfa ≥ 1` and `Inf` for `pfa ≤ 0`.
"""
function fringe_snr_cut(pfa::Real, ncells::Real)
    (isfinite(pfa) && isfinite(ncells)) || return NaN
    pfa >= 1 && return 0.0
    pfa <= 0 && return Inf
    p1 = -expm1(log1p(-float(pfa)) / max(ncells, 1.0))   # per-cell exceedance
    return sqrt(-2 * log(p1))
end

DimensionalData.@dim FringeDelay "Fringe delay (s)"
DimensionalData.@dim FringeRate "Fringe rate (Hz)"

"""
    FringeSearchMap

The windowed delay–rate matched-filter surface of one visibility block, produced
by [`baseline_fringe_map`](@ref) and [`fringe_search_map`](@ref) — the classic
false-fringe diagnostic. The coordinates, map and detection are at the search's
compute precision, the real type of the visibilities. Fields:

- `snr` — `DimArray` of `|D| / noise` over `(FringeDelay, FringeRate)` (s, Hz;
  the in-window grid, ascending), in the same units as the detection's `snr`,
  so the map's peak sits at ≈ `detection.snr`. Its `refdims` name the scan,
  baseline and feed pair when the map came from [`fringe_search_map`](@ref).
- `detection` — the refined peak, exactly as [`baseline_fringe_search`](@ref)
  returns it.
- `ncells` — effective number of independent search cells (see `fringe_pfa`).
- `pfa` — `fringe_pfa(detection.snr, ncells)`.
"""
struct FringeSearchMap{S, Det}
    snr::S
    detection::Det
    ncells::Float64
    pfa::Float64
end

"""
    baseline_fringe_map(plane, f0, t0; opts = FringeSearch(), workspace = nothing)
        -> FringeSearchMap

Compute the full delay–rate SNR surface for one visibility block — the
brute-force common-Δf grid/FFT ([`FullGrid`](@ref)), returning the windowed
`|D|` plane (in SNR units) instead of only the peak. The PLANE is always the
full grid — showing the complete sidelobe/alias structure is the point of the
diagnostic — while the embedded `detection` honors `opts.algorithm`, so it is
identical to what [`baseline_fringe_search`](@ref) (and the solver) returns. A
diagnostic (a full-grid FFT per call — on VGOS-style axes this single plane is
much more expensive than the hierarchical search that produced the solution),
not a hot path. `ncells` and `pfa` are those of this one search.
"""
function baseline_fringe_map(
        plane::AbstractDimStack, f0::Real, t0::Real;
        opts::FringeSearch = FringeSearch(),
        workspace::Union{Nothing, FringeWorkspace} = nothing,
    )
    freqs, times = _plane_axes(plane)
    ws = something(workspace, FringeWorkspace(eltype(plane[:vis])))
    V, W, F = _gather_plane!(ws, plane)
    return _fringe_map(V, W, F, freqs, times, f0, t0, opts, ws, nothing)
end

# Explicit lookups: an empty map would otherwise infer a different lookup type.
_ascending(x) = DimensionalData.Lookups.Sampled(
    x; order = DimensionalData.Lookups.ForwardOrdered(), span = DimensionalData.Lookups.Irregular(),
    sampling = DimensionalData.Lookups.Points(),
)
_snr_map(snr, delays, rates) = DimArray(snr, (FringeDelay(_ascending(delays)), FringeRate(_ascending(rates))); name = :snr)

# `family_cells` is the false-alarm family of the detection's `pfa`; `nothing`
# takes this one search's cells.
function _fringe_map(V, W, F, freqs, times, f0, t0, opts::FringeSearch, ws::FringeWorkspace, family_cells)
    C = eltype(V)
    T = real(C)
    ax = _search_axes(freqs, times, opts, C)              # detection axes (honor opts.algorithm)
    axf = isnothing(ax.mbd) ? ax :                       # plane axes: always the full grid
        _search_axes(freqs, times, FringeSearch(opts.delay_window, opts.rate_window, opts.oversample, opts.quad_interp, FullGrid()), C)
    ncells = something(family_cells, _search_cells(axf, opts))
    _ensure_workspace!(ws, C, axf.nf_pad, axf.nt_pad)
    Wsum = _grid_visibilities!(ws, V, W, F, freqs, times, axf)
    Wsum > 0 || return FringeSearchMap(_snr_map(zeros(T, 0, 0), T[], T[]), _invalid_detection(T), ncells, NaN)
    D = ws.D
    mul!(D, axf.plan, ws.G)

    # In-window bins of each conjugate axis, in ascending coordinate order (the
    # fftfreq vectors are in FFT order [0, +…, −…]).
    kidx = [k for k in eachindex(axf.delays) if _in_window(axf.delays[k], opts.delay_window, axf.fax.degenerate)]
    lidx = [l for l in eachindex(axf.rates) if _in_window(axf.rates[l], opts.rate_window, axf.tax.degenerate)]
    sort!(kidx; by = k -> axf.delays[k])
    sort!(lidx; by = l -> axf.rates[l])
    inv_noise = inv(sqrt(Wsum))
    snrmap = _snr_map(abs.(D[kidx, lidx]) .* inv_noise, axf.delays[kidx], axf.rates[lidx])

    # The refined peak, via the standard search (re-grids + re-FFTs the same data
    # in `ws` — the map above is already copied out, and reusing the search keeps
    # the peak/refinement logic in one place).
    det = _baseline_fringe_search(V, W, F, freqs, times, f0, t0, ax, ws, opts, ncells)
    return FringeSearchMap(snrmap, det, ncells, fringe_pfa(det.snr, ncells))
end
