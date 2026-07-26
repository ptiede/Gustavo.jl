# ── Data transforms: the materialization hook chain ──────────────────────────
#
# An `AbstractDataTransform` is a caller-supplied operation applied to each scan
# group's visibilities/weights AS IT IS MATERIALIZED, before any solver stage
# sees it. The chain generalizes (and will replace) the fixed precal carrier
# built by `_make_precal` — precal division, per-station weight scaling, and
# channel flagging become three built-in transforms, and `CalFunction` opens the
# same choke point to arbitrary caller code (e.g. rescaling the weights of ONE
# baseline on ONE scan) with no edits to Gustavo internals.
#
# The contract: implement `apply_transform!(t, v::ScanDataView; inner)` mutating
# `v.vis`/`v.weights` in place. Transforms run in chain order at every
# materialization, so a solve, a re-run, and a diagnostic that share the chain
# see identical data. Solutions record the chain they were solved with
# (`sol.transforms`), so diagnostics can replay it automatically.

"""
    AbstractDataTransform

A per-scan-group data hook applied at materialization, before any solver stage
reads the scan. Implement [`apply_transform!`](@ref) for the in-place streaming
form; optionally [`apply_transform`](@ref) for an eager whole-`UVSet` form.
Built-ins: [`ApplySolution`](@ref), [`StationWeightScale`](@ref),
[`FlagChannels`](@ref), [`CalFunction`](@ref).
"""
abstract type AbstractDataTransform end

using DimensionalData: DimensionalData, DimArray, DimStack, AbstractDimStack, lookup, Ti
using ..UVData: Baseline

"""
    ScanDataView

The mutable window a transform sees: one scan's data in leaf `DimStack` layout,
plus the window that addresses it in the solve's index space. NOT a separate
data format — the stack is never assembled by decomposing an existing
container:

- `data` — a `DimStack` with layers `:vis`/`:weights` on
  `(Frequency, Ti, Baseline, Pol)` dims (frequencies in Hz, times in hours,
  correlation products on the `Pol` lookup) and the leaf's `PartitionInfo`
  metadata. On the leaf paths this is literally `leaf[(:vis, :weights)]` — the
  layer selection off the leaf tree, metadata and all; on the concatenated-cube
  path it is the [`ScanGroup`](@ref)'s stack, built once where the cube is
  born. There is no other construction route: a view always wraps data that
  already lives in leaf form. DimensionalData selectors work directly, e.g.
  `v.data[:vis][Pol = pol_at("PP")]`. Mutate the layers IN PLACE
  (`parent(v.data[:weights]) .= …`).
- `chan_idx`, `ti_idx` — the view's GLOBAL channel/time indices into `geom`
  (`geom.channel_freqs[chan_idx] == v.freqs`) — the solve-side index window a
  coordinate axis cannot carry.
- `geom` — the solve's `DataGeometry`.

Convenience properties, all derived from `data` (the raw-array forms the
built-in transforms and hot kernels use): `v.vis` / `v.weights` (the parent
`(nchan, nti, nbl, npol)` arrays — mutating them mutates `data` and vice
versa), `v.freqs`, `v.times`, `v.bl_pairs`, `v.pol_products`, `v.source`,
`v.scan`, `v.ant_names`.
"""
struct ScanDataView{S <: AbstractDimStack}
    data::S
    chan_idx::Vector{Int}
    ti_idx::Vector{Int}
    geom::DataGeometry
end

function Base.getproperty(v::ScanDataView, s::Symbol)
    s === :vis && return parent(getfield(v, :data)[:vis])
    s === :weights && return parent(getfield(v, :data)[:weights])
    s === :freqs && return parent(lookup(getfield(v, :data)[:vis], Frequency))
    s === :times && return parent(lookup(getfield(v, :data)[:vis], Ti))
    s === :pol_products && return parent(lookup(getfield(v, :data)[:vis], Pol))
    s === :bl_pairs && return DimensionalData.metadata(getfield(v, :data)).baselines.pairs
    s === :source && return DimensionalData.metadata(getfield(v, :data)).source_name
    s === :scan && return DimensionalData.metadata(getfield(v, :data)).scan_name
    s === :ant_names && return String.(DimensionalData.metadata(getfield(v, :data)).antennas.name)
    return getfield(v, s)
end
Base.propertynames(::ScanDataView) = (
    :data, :chan_idx, :ti_idx, :geom, :vis, :weights, :freqs, :times,
    :bl_pairs, :pol_products, :source, :scan, :ant_names,
)

"""
    apply_transform!(t::AbstractDataTransform, v::ScanDataView; inner = 1)

Apply `t` to one materialized scan window, mutating `v.vis`/`v.weights` in
place. `inner` is the task budget for transforms that fan out over baselines.
The extension point for custom transforms.
"""
function apply_transform!(t::AbstractDataTransform, v::ScanDataView; inner::Integer = 1)
    return error(
        "apply_transform! not implemented for $(typeof(t)) — implement " *
            "`apply_transform!(t, v::ScanDataView; inner)` mutating v.vis/v.weights in place."
    )
end

"""
    apply_transform(uvset::UVSet, t::AbstractDataTransform) -> UVSet

Eager whole-set form of a transform (used by the standalone
`calibrate(sol, uvset)` path on transforms recorded in a solution). Only
transforms with a well-defined whole-set meaning provide it.
"""
function apply_transform(uvset::UVSet, t::AbstractDataTransform)
    return error(
        "apply_transform (eager whole-UVSet form) is not available for $(typeof(t)); " *
            "it is applied in the streaming pass only."
    )
end

# Run a chain in order (the choke-point entry; `nothing` chain = no-op).
function apply_transforms!(ts, v::ScanDataView; inner::Integer = 1)
    ts === nothing && return v
    for t in ts
        apply_transform!(t, v; inner = inner)
    end
    return v
end

"""
    validate_transform(t::AbstractDataTransform, geom::DataGeometry, ant_names)

Fail-fast compatibility check of a transform against the target set's geometry,
run once at stream construction — so an incompatible transform (e.g. an
`ApplySolution` from a different correlator setup) raises a plain error before
any data is read, not a `TaskFailedException` from a worker mid-pass. The
default accepts anything; transforms with compatibility requirements add
methods.
"""
validate_transform(t::AbstractDataTransform, geom::DataGeometry, ant_names) = nothing

"""
    apply_transforms(uvset::UVSet, transforms; geom = build_geometry(uvset)) -> UVSet

Eagerly apply a transform chain to a whole `UVSet`, leaf by leaf — each leaf is
wrapped in a [`ScanDataView`](@ref) with its global geometry window, so EVERY
transform works here, including [`CalFunction`](@ref) and
[`FlagChannels`](@ref) (which need global indices and have no standalone
whole-set form). This is the replay of the chain a solution records
(`sol.transforms`), used by the standalone `calibrate`. Leaf arrays are
copied; the input set is never mutated.
"""
function apply_transforms(uvset::UVSet, transforms; geom::DataGeometry = build_geometry(uvset))
    ts = collect(Any, transforms)
    isempty(ts) && return uvset
    return UVData.apply(uvset) do leaf, info, root
        ml = UVData.materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
        # Rewrap around array copies (the input set is never mutated), then the
        # view is just the layer selection off that leaf — metadata included.
        mlc = with_visibilities(ml, copy(parent(ml[:vis])), copy(parent(ml[:weights])))
        ci, ti = leaf_window(geom, mlc)
        v = ScanDataView(mlc[(:vis, :weights)], collect(Int, ci), collect(Int, ti), geom)
        apply_transforms!(ts, v)
        # Re-derive the flag layer from the transformed weights.
        return with_visibilities(mlc, v.vis, v.weights)
    end
end

# ── Built-in: precal solution division ────────────────────────────────────────

"""
    ApplySolution(sol::CalibrationSolution)

Transform: divide `sol`'s gains out of each scan as it is materialized
(`V → V / (g_a·conj(g_b))`, `w → w·|g_a g_b|²`), e.g. a phase-cal solution.
The fringe solution fit on top is then the correction ON TOP of `sol` (total
gain = `sol ∘ solution`). Cells where the gain is non-finite or zero are left
untouched (matching the solver's precal semantics — no data is invented or
destroyed by a bad precal cell).

A solution fit on the SAME set applies index-aligned. A solution from ANOTHER
run — the fit-once / apply-later workflow, e.g. a
[`Gustavo.Calibration.bandpass_solution`](@ref) extraction — is portable when
it is globally TIME-CONSTANT (every component `GlobalTime`): stations are then
matched BY NAME against the solution's recorded `ant_names` (stations it never
solved keep identity gains, with a warning), and the channel layout must be
identical (same channel frequencies — same correlator setup).
"""
struct ApplySolution{S <: CalibrationSolution} <: AbstractDataTransform
    sol::S
end

# Fail-fast compatibility check of an ApplySolution against a target geometry —
# run at STREAM CONSTRUCTION (`scan_stream` / the verbs), so an incompatible
# precal raises a plain error before any data is read instead of surfacing as a
# TaskFailedException from a worker mid-pass.
function validate_transform(t::ApplySolution, geom::DataGeometry, ant_names)
    sol = t.sol
    sol.geom.channel_freqs == geom.channel_freqs || error(
        "ApplySolution: channel layout differs from the solution's " *
            "($(length(geom.channel_freqs)) vs $(length(sol.geom.channel_freqs)) channels, " *
            "or different frequencies) — a per-channel solution only ports across sets with " *
            "the identical correlator setup."
    )
    sol.geom.times == geom.times && return nothing
    # Cross-set apply (fit once, apply later — possibly another file/epoch).
    ev = GainEvaluator(sol.model, sol.layout)
    _globally_time_constant(ev) || error(
        "ApplySolution: the data's time axis differs from the solution's, and the solution " *
            "is not time-constant (it carries per-scan/per-integration components) — only " *
            "time-constant solutions (e.g. `bandpass_solution(sol)`) are portable across sets."
    )
    hasproperty(sol.info, :ant_names) || error(
        "ApplySolution: cross-set application needs the solution's station names " *
            "(`sol.info.ant_names`) to match stations by name — this solution has none."
    )
    return nothing
end

function apply_transform!(t::ApplySolution, v::ScanDataView; inner::Integer = 1)
    sol = t.sol
    ev = GainEvaluator(sol.model, sol.layout)
    if sol.geom.channel_freqs == v.geom.channel_freqs && sol.geom.times == v.geom.times
        # Same-set apply: stations index-aligned, full time mapping.
        _divide_gains!(v.vis, v.weights, ev, sol.θ, v.chan_idx, v.ti_idx, v.bl_pairs, v.pol_products, inner)
        return nothing
    end
    # Cross-set apply; the compatibility contract was already enforced by
    # `validate_transform` at stream construction, re-checked here for direct
    # (stream-less) callers.
    validate_transform(t, v.geom, v.ant_names)
    solnames = String.(collect(sol.info.ant_names))
    amap = [something(findfirst(==(n), solnames), 0) for n in v.ant_names]
    if any(iszero, amap)
        missing_names = [n for (n, m) in zip(v.ant_names, amap) if m == 0]
        @warn "ApplySolution: stations $(missing_names) are not in the solution — they keep identity gains." maxlog = 1
    end
    _divide_gains_mapped!(
        v.vis, v.weights, ev, sol.θ, v.chan_idx, amap, v.bl_pairs, v.pol_products, inner,
    )
    return nothing
end

# Every component time-constant (GlobalTime, no time coordinate) — the
# portability requirement for applying a solution onto a different time axis.
function _globally_time_constant(ev::GainEvaluator)
    for plan in ev.layout.plans
        plan.coord === Calibration.COORD_TIME && return false
        ts1 = plan.tseg_id[1]
        all(==(ts1), plan.tseg_id) || return false
    end
    return true
end

apply_transform(uvset::UVSet, t::ApplySolution) = UVData.apply_calibration(uvset, t.sol)

# Divide evaluated gains out of a (nchan, nti, nbl, npol) window in place —
# the transform-chain port of `_divide_precal!`'s gain branch, with the same
# time-constant fast path and the same skip-bad-cell semantics.
function _divide_gains!(V, W, ev::GainEvaluator, θ, ci, ti, bl_pairs, pols, inner::Integer)
    nchan, nti, nbl, npol = size(V)
    tconst = _precal_time_constant(ev, ti)
    g = tconst ? evaluate_gains(ev, θ, ci, ti[1]:ti[1]) : evaluate_gains(ev, θ, ci, ti)
    cols = [(bi, p) for p in 1:npol for bi in 1:nbl]
    nt = clamp(Int(inner), 1, length(cols))
    do_chunk = function (chunk)
        @inbounds for (bi, p) in chunk
            fa, fb = correlation_feed_pair(pols[p])
            a, b = bl_pairs[bi]
            for t in 1:nti
                gt = tconst ? 1 : t
                for c in 1:nchan
                    den = g[c, gt, a, fa] * conj(g[c, gt, b, fb])
                    (isfinite(den) && abs2(den) > 0) || continue
                    V[c, t, bi, p] /= den
                    W[c, t, bi, p] *= abs2(den)
                end
            end
        end
    end
    if nt <= 1
        do_chunk(cols)
    else
        chunks = collect(Iterators.partition(cols, cld(length(cols), nt)))
        exec_foreach(do_chunk, chunks; ntasks = length(chunks))
    end
    return nothing
end

# Cross-set variant of `_divide_gains!`: the (time-constant) gains are
# evaluated once in the SOLUTION's station space and each data station is
# remapped through `amap` (target ant index → solution ant index, 0 = not in
# the solution → identity, cell untouched).
function _divide_gains_mapped!(V, W, ev::GainEvaluator, θ, ci, amap, bl_pairs, pols, inner::Integer)
    nchan, nti, nbl, npol = size(V)
    g = evaluate_gains(ev, θ, ci, 1:1)               # time-constant: one column
    cols = [(bi, p) for p in 1:npol for bi in 1:nbl]
    nt = clamp(Int(inner), 1, length(cols))
    do_chunk = function (chunk)
        @inbounds for (bi, p) in chunk
            fa, fb = correlation_feed_pair(pols[p])
            a, b = bl_pairs[bi]
            am = amap[a]
            bm = amap[b]
            (am == 0 || bm == 0) && continue
            for t in 1:nti, c in 1:nchan
                den = g[c, 1, am, fa] * conj(g[c, 1, bm, fb])
                (isfinite(den) && abs2(den) > 0) || continue
                V[c, t, bi, p] /= den
                W[c, t, bi, p] *= abs2(den)
            end
        end
    end
    if nt <= 1
        do_chunk(cols)
    else
        chunks = collect(Iterators.partition(cols, cld(length(cols), nt)))
        exec_foreach(do_chunk, chunks; ntasks = length(chunks))
    end
    return nothing
end

# ── Built-in: per-station weight scaling ──────────────────────────────────────

"""
    StationWeightScale(s::AbstractVector{<:Real})

Transform: per-station weight correction, `w → w·s_a·s_b` per baseline
(NOISE-ESTIMATE fix only; visibilities untouched). Build `s` with
[`station_weight_scale`](@ref). Every factor must be finite and positive.
"""
struct StationWeightScale <: AbstractDataTransform
    s::Vector{Float64}
    function StationWeightScale(s::AbstractVector{<:Real})
        sv = Vector{Float64}(s)
        all(x -> isfinite(x) && x > 0, sv) ||
            error("StationWeightScale: every factor must be finite and positive, got $(sv).")
        return new(sv)
    end
end

function apply_transform!(t::StationWeightScale, v::ScanDataView; inner::Integer = 1)
    s = t.s
    n = maximum(max(a, b) for (a, b) in v.bl_pairs; init = 0)
    n <= length(s) ||
        error("StationWeightScale: factor vector has $(length(s)) entries but the data references station index $n.")
    W = v.weights
    @inbounds for p in axes(W, 4), (bi, (a, b)) in enumerate(v.bl_pairs)
        f = s[a] * s[b]
        f == 1 && continue
        @views W[:, :, bi, p] .*= f
    end
    return nothing
end

function apply_transform(uvset::UVSet, t::StationWeightScale)
    s = t.s
    return UVData.apply(uvset) do leaf, info, root
        leaf = UVData.materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
        W = copy(parent(leaf[:weights]))
        for (bi, (a, b)) in enumerate(UVData.baselines(leaf).pairs)
            f = s[a] * s[b]
            f == 1 && continue
            @views W[:, :, bi, :] .*= f
        end
        return with_visibilities(leaf, copy(parent(leaf[:vis])), W)
    end
end

# ── Built-in: global channel flagging ─────────────────────────────────────────

"""
    FlagChannels(mask::BitVector)

Transform: zero-weight the flagged GLOBAL channels (mask indexed by the solve
geometry's channel axis, `true` = flag), e.g. `tone_channel_mask`.
"""
struct FlagChannels <: AbstractDataTransform
    mask::BitVector
end
FlagChannels(mask::AbstractVector{Bool}) = FlagChannels(BitVector(mask))

function apply_transform!(t::FlagChannels, v::ScanDataView; inner::Integer = 1)
    length(t.mask) == length(v.geom.channel_freqs) ||
        error("FlagChannels: mask length $(length(t.mask)) ≠ nchan $(length(v.geom.channel_freqs))")
    W = v.weights
    @inbounds for (c, gc) in enumerate(v.chan_idx)
        t.mask[gc] && (W[c, :, :, :] .= 0)
    end
    return nothing
end

# ── The open hook: arbitrary caller code ──────────────────────────────────────

"""
    CalFunction(f)

Transform: run caller code `f(v::ScanDataView)` on each scan group as it is
materialized. `f` may mutate `v.vis`/`v.weights` in place and can address any
(scan, baseline, channel, pol) via the view's metadata — the general escape
hatch for per-datum corrections that have no dedicated option:

    # halve the weight of the PT–LA baseline on scan "No0012" only
    fix = CalFunction() do v
        v.scan == "No0012" || return
        pt = findfirst(==("PT"), v.ant_names); la = findfirst(==("LA"), v.ant_names)
        for (bi, (a, b)) in enumerate(v.bl_pairs)
            (minmax(a, b) == minmax(pt, la)) && (v.weights[:, :, bi, :] .*= 0.5)
        end
    end

`f`'s work should stay cheap relative to a scan's solve (~seconds); it runs on
every materialization of every scan group.
"""
struct CalFunction{F} <: AbstractDataTransform
    f::F
end

function apply_transform!(t::CalFunction, v::ScanDataView; inner::Integer = 1)
    t.f(v)
    return nothing
end

# ── Per-station weight-scale construction + precal fast-path probe ───────────

"""
    station_weight_scale(uvset_or_names, factors; default = 1.0) -> Vector{Float64}

Per-station weight correction factors, indexed by the solver's station index, built
from a station-code => factor map (e.g. `Dict("HS" => 0.5, "GL" => 0.5)`). Wrap the
result in a [`StationWeightScale`](@ref) transform in the pipeline's step chain
and every scan's weights are corrected as
`w → w·s_a·s_b` at materialization — so a baseline with ONE affected station gets
`s`, and a baseline between two affected stations gets `s²`, automatically.

This corrects the NOISE ESTIMATE only; visibilities are untouched. Use it when the
correlator's weights are miscalibrated for particular stations (an upstream bug),
which otherwise biases every weighted quantity downstream — fringe SNR (∝ √s), the
PFA gate that consumes it, the bandpass and adhoc accumulations, and the exported
weights. Codes absent from the data are ignored; stations absent from `factors`
get `default`.

    uvset = load_fitsidi(path; lazy = true)
    ws = station_weight_scale(uvset, Dict("HS" => 0.5, "GL" => 0.5))
    FringeFit(weight_scale = ws, …)
"""
station_weight_scale(uvset::UVSet, factors; default::Real = 1.0) = station_weight_scale(
    String.(UVData.metadata(first(values(UVData.branches(uvset)))).antennas.name),
    factors; default,
)

function station_weight_scale(names::AbstractVector{<:AbstractString}, factors; default::Real = 1.0)
    s = fill(Float64(default), length(names))
    for (code, f) in pairs(factors)
        i = findfirst(==(String(code)), names)
        i === nothing && continue
        s[i] = Float64(f)
    end
    return s
end

# A precal's gains are usually CONSTANT IN TIME across one scan's window (the
# phase-cal model is PerScan × PerSpectralWindow with no time-coordinate term),
# so evaluating the full (nchan × nti) gain cube wastes nti× the `cis` work —
# the dominant cost of applying the correction. True when no component reads a
# time coordinate and every component sees a single time segment in the window.
function _precal_time_constant(ev::GainEvaluator, g_ti)
    length(g_ti) <= 1 && return true
    for plan in ev.layout.plans
        plan.coord === Calibration.COORD_TIME && return false
        ts = plan.tseg_id[g_ti[1]]
        for gti in g_ti
            plan.tseg_id[gti] == ts || return false
        end
    end
    return true
end
