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

Fail-fast compatibility check of a transform against the target set, run once at
stream construction — so a transform that can never apply raises a plain error
before any data is read, not a `TaskFailedException` from a worker mid-pass. The
default accepts anything; transforms with requirements add methods.

Reserve this for what is wrong about the PAIRING of transform and set, whatever
data is read: an [`ApplySolution`](@ref) checks that its stations can be matched
to `ant_names` at all. Per-sample compatibility does not belong here — it is
checked where the sample is used, against the window actually materialized,
rather than eagerly against every sample the run might never visit.
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
# `apply_calibration`'s recorded-chain replay (see the stub in
# `Calibration/solutions.jl` for why it is wired through this layer).
Calibration._replay_transforms(uvset::UVSet, transforms) = apply_transforms(uvset, transforms)

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

The solution need not have been fit on this set, nor on its sampling. Each
target sample is placed in the segment of `sol` it BELONGS to, by the identity
the segmentation is defined on: a `PerScan` component by scan name, a
`PerSpectralWindow` one by spw name, a `TimeBlocks` or `InstrumentScans` one by
the solve's own bin formula, a `PerIntegration` one by exact epoch. So a
solution segmented more COARSELY than the target applies — a bandpass fit on
scan-averaged data corrects at full time resolution — while one segmented more
finely has no answer and is refused. `ChannelBlocks` and `FreqGroups` cut the
channel-index axis, so they require the target to index the same channels.

Stations are matched BY NAME against the solution's recorded `ant_names` — a
name is the whole of a station's identity here, never its position — and
stations the solution never solved keep identity gains, with a warning. A
solution that records no `ant_names`, or that shares no station with the set at
all, is refused by [`validate_transform`](@ref) at stream construction. Placement
is checked per scan group, as each window is materialized.
"""
struct ApplySolution{S <: CalibrationSolution} <: AbstractDataTransform
    sol::S
end

function apply_transform!(
        t::ApplySolution, stack::AbstractDimStack, win::GeometryWindow; executor = SerialScheduler(),
    )
    _divide_gains!(stack, win, t.sol, executor; amap = _station_map(t.sol, String.(UVData.antennas(stack).name)))
    return nothing
end

apply_transform(uvset::UVSet, t::ApplySolution) = UVData.apply_calibration(uvset, t.sol)

# Target station index → the solution's own (0 = absent from the solution).
# `nothing` when the two agree position for position and no mapping is needed.
function _station_map(sol::CalibrationSolution, ant_names)
    solnames = _solution_ant_names(sol)
    solnames == ant_names && return nothing
    m = [something(findfirst(==(n), solnames), 0) for n in ant_names]
    if any(iszero, m)
        missing_names = [n for (n, k) in zip(ant_names, m) if k == 0]
        @warn "ApplySolution: stations $(missing_names) are not in the solution — they keep identity gains." maxlog = 1
    end
    return m
end

# The station names a solution matches on. One that records none has no station
# identity at all — only its own set's positional order, which means nothing
# anywhere else — so it cannot be applied, here or at the construction gate.
function _solution_ant_names(sol::CalibrationSolution)
    (hasproperty(sol.info, :ant_names) && !isempty(sol.info.ant_names)) || throw(
        ArgumentError(
            "ApplySolution: the solution records no station names, so its θ rows could only be " *
                "matched to a set's stations by position — which means nothing across sets. " *
                "Re-solve so the solution records `ant_names`."
        )
    )
    return String.(collect(sol.info.ant_names))
end

"""
    validate_transform(t::ApplySolution, geom::DataGeometry, ant_names)

Check that the solution's stations can be matched to this set's at all: a
station is identified by its NAME and nothing else, so the solution must record
`ant_names`, and at least one of `ant_names` must appear in them. Stations the
solution is missing are not an error — they keep identity gains, with a warning
from the apply — but a solution sharing NO station with the set corrects
nothing at all, and silently doing nothing is the outcome worth refusing.

Whether an individual channel or time can be PLACED in the solution is not
checked here. It is checked where the sample is used, against the window
actually materialized; validating the whole geometry up front would reject a
run over samples it never visits.
"""
function validate_transform(t::ApplySolution, geom::DataGeometry, ant_names)
    solnames = _solution_ant_names(t.sol)
    any(in(solnames), ant_names) || throw(
        ArgumentError(
            "ApplySolution: the solution shares no station with this set, so it would correct " *
                "nothing. It knows $(join(map(repr, solnames), ", ")); the set has " *
                "$(join(map(repr, ant_names), ", "))."
        )
    )
    return nothing
end

# Whether the solution's gains are the same at every time in the window: no
# component reads a time coordinate, and every sample places in one time segment.
# A precal is usually such a solution (PerScan × PerSpectralWindow with no time
# term), and evaluating ONE time column instead of `nti` of them saves the
# `cis`/`exp` work that dominates applying the correction.
#
# Placement runs over the WHOLE window here, so a sample the solution cannot
# place — or whose span straddles a bin boundary — raises exactly as it would in
# the full evaluation. This decides how many columns to evaluate, never whether
# to check.
function _time_constant_over(sol::CalibrationSolution, win::GeometryWindow, tspan)
    length(win.ti_idx) <= 1 && return true
    for s in sol.steps, plan in s.layout.plans
        :Ti in Calibration.term_axes(plan.term) && return false
        ids = Calibration.time_segment_ids(
            plan.tseg, sol.geom, win.geom; ti_idx = win.ti_idx, time_span = tspan,
        )
        allequal(ids) || return false
    end
    return true
end

# Divide evaluated gains out of one scan window in place, with the same
# skip-bad-cell semantics as `_divide_precal!`'s gain branch. Gains are placed
# through `win` — the target set's own geometry plus the global indices this
# window covers — so each sample reads the solution segment it belongs to
# whatever the solve was sampled on.
#
# `amap` maps a target station index to the solution's (0 = absent from the
# solution → cell untouched).
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
    tspan = UVData.metadata(stack).time_span
    tconst = _time_constant_over(sol, win, tspan)
    g = Calibration._composed_gains(
        sol, win.geom;
        chan_idx = win.chan_idx,
        ti_idx = tconst ? win.ti_idx[1:1] : win.ti_idx,
        time_span = tconst ? _head_span(tspan) : tspan,
    )
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

# The span of just the first sample, to match a single evaluated time column.
_head_span(span) = span === nothing || isempty(span) ? span : span[1:1]

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
from a station-code => factor map (e.g. `Dict("HS" => 2.0, "GL" => 2.0)`). A factor
ABOVE 1 raises a station's weights, which is what a correlator claiming MORE noise
than the data carries needs. Wrap the
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
    ws = station_weight_scale(uvset, Dict("HS" => 2.0, "GL" => 2.0))
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
