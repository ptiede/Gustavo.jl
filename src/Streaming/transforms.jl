# ── Data transforms: the materialization hook chain ──────────────────────────
#
# An `AbstractDataTransform` is a caller-supplied operation applied to each scan
# group's visibilities/weights AS IT IS MATERIALIZED, before any solver stage
# sees it. Precal division, per-station weight scaling and channel flagging are
# built-in transforms; `CalFunction` opens the same choke point to arbitrary
# caller code (e.g. rescaling the weights of ONE baseline on ONE scan) with no
# edits to Gustavo internals.
#
# The contract: implement `apply_transform!(t, stack, win; executor)` mutating the
# stack's `:vis`/`:weights` layers in place. Transforms run in chain order at
# every materialization, so a solve, a re-run, and a diagnostic that share the
# chain see identical data. Solutions record the chain they were solved with
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

"""
    apply_transform!(t::AbstractDataTransform, stack::AbstractDimStack,
                     win::GeometryWindow; executor = SerialScheduler())

Apply `t` to one materialized scan window, mutating `stack`'s `:vis`/`:weights`
layers in place. The extension point for custom transforms.

`stack` carries layers `:vis`/`:weights` on `(Frequency, Ti, Baseline, Pol)`
dims (frequencies in Hz, times in hours, correlation products on the `Pol`
lookup) and the leaf's `PartitionInfo` metadata, so DimensionalData selectors
and the `UVData` accessors both work on it directly — `stack[:vis][Pol =
pol_at("PP")]`, [`frequencies`](@ref), [`baselines`](@ref), [`source_name`](@ref).
`win` addresses the same channels and times in the solve's index space, which a
coordinate axis cannot carry (`win.geom.channel_freqs[win.chan_idx] ==
frequencies(stack)`).

`executor` is the OhMyThreads scheduler for transforms that fan out over
baselines (the within-scan inner executor — see [`ExecutionConfig`](@ref)).
"""
# Only the transform type is annotated: a third-party method that leaves `stack`
# and `win` unannotated — the natural spelling — must be strictly MORE specific
# than this fallback, not ambiguous with it.
function apply_transform!(t::AbstractDataTransform, stack, win; executor = SerialScheduler())
    return error(
        "apply_transform! not implemented for $(typeof(t)) — implement " *
            "`apply_transform!(t, stack, win::GeometryWindow; executor)` mutating the " *
            "stack's :vis/:weights layers in place."
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
function apply_transforms!(ts, stack::AbstractDimStack, win::GeometryWindow; executor = SerialScheduler())
    ts === nothing && return stack
    for t in ts
        apply_transform!(t, stack, win; executor)
    end
    return stack
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
paired with its global [`GeometryWindow`](@ref), so EVERY transform works here,
including [`CalFunction`](@ref) and [`FlagChannels`](@ref) (which need global
indices and have no standalone whole-set form). This is the replay of the chain
a solution records (`sol.transforms`), used by the standalone `calibrate`. Leaf
arrays are copied; the input set is never mutated.
"""
function apply_transforms(uvset::UVSet, transforms; geom::DataGeometry = build_geometry(uvset))
    ts = collect(Any, transforms)
    isempty(ts) && return uvset
    return UVData.apply(uvset) do leaf, info, root
        ml = UVData.materialize_leaf(leaf)
        # Rewrap around array copies (the input set is never mutated); the
        # transform's stack is then just the layer selection off that leaf —
        # metadata included. The selection shares `mlc`'s arrays, so the chain
        # mutates `mlc` in place and it is the transformed leaf.
        mlc = rebuild_visibilities(ml, copy(parent(ml[:vis])), copy(parent(ml[:weights])))
        apply_transforms!(ts, mlc[(:vis, :weights)], leaf_window(geom, mlc))
        return mlc
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
[`Gustavo.Calibration.step_solution`](@ref) extraction — is portable when
it is globally TIME-CONSTANT (every component `GlobalTime`): stations are then
matched BY NAME against the solution's recorded `ant_names` (stations it never
solved keep identity gains, with a warning), and the channel layout must be
identical (same channel frequencies — same correlator setup).
"""
struct ApplySolution{S <: CalibrationSolution} <: AbstractDataTransform
    sol::S
end

function apply_transform!(
        t::ApplySolution, stack::AbstractDimStack, win::GeometryWindow; executor = SerialScheduler(),
    )
    sol = t.sol
    ant_names = String.(UVData.antennas(stack).name)
    solnames = hasproperty(sol.info, :ant_names) ? String.(collect(sol.info.ant_names)) : ant_names
    amap = if solnames == ant_names
        nothing                                      # index-aligned
    else
        m = [something(findfirst(==(n), solnames), 0) for n in ant_names]
        if any(iszero, m)
            missing_names = [n for (n, k) in zip(ant_names, m) if k == 0]
            @warn "ApplySolution: stations $(missing_names) are not in the solution — they keep identity gains." maxlog = 1
        end
        m
    end
    _divide_gains!(stack, win, sol, executor; amap)
    return nothing
end

apply_transform(uvset::UVSet, t::ApplySolution) = UVData.apply_calibration(uvset, t.sol)

# Divide evaluated gains out of one scan window in place — the transform-chain
# port of `_divide_precal!`'s gain branch, with the same time-constant fast path
# and the same skip-bad-cell semantics. The gain divided out is the ELEMENTWISE
# PRODUCT of every step's own evaluator (see `Calibration._composed_gains`),
# computed here rather than via that helper so the time-constant fast path
# below still applies (one evaluation of `tsel`, shared by every step).
#
# `amap` selects the SAME-set or CROSS-set reading. `nothing` (same set) reads
# gains at the data's own station indices over the window's times. A vector
# (target ant index → solution ant index, 0 = absent from the solution → cell
# untouched) matches stations BY NAME and evaluates at the solution's single
# time column, which is well-defined because `validate_transform` has already
# established the solution is globally time-constant — the window's `ti_idx`
# addresses the DATA's time axis and means nothing in the solution's.
#
# The elementwise write loop is the one place a raw `Array` earns its keep: DD's
# `setindex!` is not `@propagate_inbounds`, so a `DimArray` here keeps bounds
# checks the loop is written to elide.
function _divide_gains!(
        stack::AbstractDimStack, win::GeometryWindow, sol::CalibrationSolution,
        executor; amap = nothing,
    )
    V = parent(stack[:vis])
    W = parent(stack[:weights])
    bl_pairs = UVData.baselines(stack).pairs
    pols = pol_products(stack)
    nchan, nti, nbl, npol = size(V)
    ti = win.ti_idx
    evs = [GainEvaluator(s.model, s.layout) for s in sol.steps]
    tconst = amap === nothing ? all(ev -> _precal_time_constant(ev, ti), evs) : true
    tsel = amap === nothing ? (tconst ? (ti[1]:ti[1]) : ti) : (1:1)
    g = evaluate_gains(evs[1], sol.steps[1].θ, win.chan_idx, tsel)
    for (ev, s) in Iterators.drop(zip(evs, sol.steps), 1)
        g .*= evaluate_gains(ev, s.θ, win.chan_idx, tsel)
    end
    cols = [(bi, p) for p in 1:npol for bi in 1:nbl]
    tforeach(cols; scheduler = executor) do col
        bi, p = col
        @inbounds begin
            fa, fb = correlation_feed_pair(pols[p])
            a, b = bl_pairs[bi]
            if amap !== nothing
                a = amap[a]
                b = amap[b]
                (a == 0 || b == 0) && return
            end
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

function apply_transform!(
        t::StationWeightScale, stack::AbstractDimStack, ::GeometryWindow; executor = SerialScheduler(),
    )
    s = t.s
    bl_pairs = UVData.baselines(stack).pairs
    n = maximum(max(a, b) for (a, b) in bl_pairs; init = 0)
    n <= length(s) ||
        error("StationWeightScale: factor vector has $(length(s)) entries but the data references station index $n.")
    W = stack[:weights]
    for p in axes(W, Pol), (bi, (a, b)) in enumerate(bl_pairs)
        f = s[a] * s[b]
        f == 1 && continue
        @views W[:, :, bi, p] .*= f
    end
    return nothing
end

function apply_transform(uvset::UVSet, t::StationWeightScale)
    s = t.s
    return UVData.apply(uvset) do leaf, info, root
        leaf = UVData.materialize_leaf(leaf)
        W = copy(parent(leaf[:weights]))
        for (bi, (a, b)) in enumerate(UVData.baselines(leaf).pairs)
            f = s[a] * s[b]
            f == 1 && continue
            @views W[:, :, bi, :] .*= f
        end
        return rebuild_visibilities(leaf, copy(parent(leaf[:vis])), W)
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

function apply_transform!(
        t::FlagChannels, stack::AbstractDimStack, win::GeometryWindow; executor = SerialScheduler(),
    )
    length(t.mask) == length(win.geom.channel_freqs) ||
        error("FlagChannels: mask length $(length(t.mask)) ≠ nchan $(length(win.geom.channel_freqs))")
    # `t.mask` is indexed by GLOBAL channel; gathering it through `win.chan_idx`
    # gives the mask over this window's own frequency axis.
    stack[:weights][Frequency = t.mask[win.chan_idx]] .= 0
    return nothing
end

# ── The open hook: arbitrary caller code ──────────────────────────────────────

"""
    CalFunction(f)

Transform: run caller code `f(stack, win)` on each scan group as it is
materialized. `f` may mutate the stack's `:vis`/`:weights` layers in place and
can address any (scan, baseline, channel, pol) through the stack's dims and
metadata — the general escape hatch for per-datum corrections that have no
dedicated option:

    # halve the weight of the PT–LA baseline on scan "No0012" only
    fix = CalFunction() do stack, win
        scan_name(stack) == "No0012" || return
        names = antennas(stack).name
        pt = findfirst(==("PT"), names); la = findfirst(==("LA"), names)
        for (bi, (a, b)) in enumerate(baselines(stack).pairs)
            (minmax(a, b) == minmax(pt, la)) && (stack[:weights][Baseline = bi] .*= 0.5)
        end
    end

`win` is the scan's [`GeometryWindow`](@ref), for code that needs to address the
solve's global channel/time indices. `f`'s work should stay cheap relative to a
scan's solve (~seconds); it runs on every materialization of every scan group.
"""
struct CalFunction{F} <: AbstractDataTransform
    f::F
end

function apply_transform!(
        t::CalFunction, stack::AbstractDimStack, win::GeometryWindow; executor = SerialScheduler(),
    )
    t.f(stack, win)
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
        :Ti in Calibration.term_axes(plan.term) && return false
        ts = plan.tseg_id[g_ti[1]]
        for gti in g_ti
            plan.tseg_id[gti] == ts || return false
        end
    end
    return true
end
