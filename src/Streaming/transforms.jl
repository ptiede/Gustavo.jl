# ── Data transforms: the materialization hook chain ──────────────────────────
#
# An `AbstractDataTransform` is a caller-supplied operation applied to each scan
# group's visibilities/weights AS IT is MATERIALIZED, before any solver stage
# sees it. Precal division, per-station weight scaling and channel flagging are
# built-in transforms; `CalFunction` opens the same choke point to arbitrary
# caller code (e.g. rescaling the weights of one baseline on one scan) with no
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

# Only the transform type is annotated: a third-party method that leaves `stack`
# and `win` unannotated — the natural spelling — must be strictly more specific
# than this fallback, not ambiguous with it.
"""
    apply_transform!(t::AbstractDataTransform, stack::AbstractDimStack,
                     win::GeometryWindow; executor = SerialScheduler())

Apply `t` to one materialized scan window, mutating `stack`'s `:vis`/`:weights`
layers in place. The extension point for custom transforms.

`stack` carries layers `:vis`/`:weights` on `(Frequency, Ti, Baseline, Pol)`
dims (frequencies in Hz, times in hours, correlation products on the `Pol`
lookup) and the leaf's `PartitionInfo` metadata, so DimensionalData selectors
and the `UVData` accessors both work on it directly — `stack[:vis][Pol =
pol_at("PP")]`, [`frequencies`](@ref), `baselines`, [`source_name`](@ref).
`win` addresses the same channels and times in the solve's index space, which a
coordinate axis cannot carry (`win.geom.channel_freqs[win.chan_idx] ==
frequencies(stack)`).

`executor` is the OhMyThreads scheduler for transforms that fan out over
baselines (the within-scan inner executor — see [`ExecutionConfig`](@ref)).
"""
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

# `apply_calibration`'s recorded-chain replay (see the stub in
# `Calibration/solutions.jl` for why it is wired through this layer).
Calibration._replay_transforms(uvset::UVSet, transforms) = apply_transforms(uvset, transforms)

"""
    apply_transforms(uvset::UVSet, transforms; geom = build_geometry(uvset)) -> UVSet

Eagerly apply a transform chain to a whole `UVSet`, leaf by leaf — each leaf is
paired with its global [`GeometryWindow`](@ref), so every transform works here,
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

The solution need not have been fit on this set, nor on its sampling. Each
target sample is placed in the segment of `sol` it belongs to, by the identity
the segmentation is defined on: a `PerScan` component by scan name, a
`PerSpectralWindow` one by spw name, a `TimeBlocks` or `InstrumentScans` one by
the solve's own bin formula, a `PerIntegration` one by exact epoch. So a
solution segmented more COARSELY than the target applies — a bandpass fit on
scan-averaged data corrects at full time resolution — while one segmented more
finely has no answer and is refused. `ChannelBlocks` and `FreqGroups` cut the
channel-index axis, so they require the target to index the same channels.

Stations are matched by NAME against the solution's recorded `ant_names` — a
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
from the apply — but a solution sharing no station with the set corrects
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
# term), and evaluating one time column instead of `nti` of them saves the
# `cis`/`exp` work that dominates applying the correction.
#
# Placement runs over the whole window here, so a sample the solution cannot
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

Transform: zero-weight the flagged global channels (mask indexed by the solve
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
    # `t.mask` is indexed by global channel; gathering it through `win.chan_idx`
    # gives the mask over this window's own frequency axis.
    stack[:weights][Frequency = t.mask[win.chan_idx]] .= 0
    return nothing
end

# ── Built-in: a-priori amplitude (SEFD) calibration ──────────────────────────

"""
    AprioriPreCal(uvset::UVSet, antab::AntabCalibration;
                  min_elevation_deg = 0.0, on_missing_station = :warn)

Transform: a-priori amplitude calibration from a parsed ANTAB, applied to each
scan group as it is materialized — so every solve step reads visibilities
already scaled to Jy. Same correction as
[`apply_calibration`](@ref Gustavo.UVData.apply_calibration)`(uvset, antab)`:
`V → V / (g_a·g_b)` and `w → w·(g_a·g_b)²` with `g_a = 1/√SEFD_a`,
`SEFD = T_sys / (DPFU · g_E(elevation))`. Samples whose Tsys is missing or
non-positive, and those below `min_elevation_deg`, are flagged (weight ← 0);
autocorrelations are flagged, being total power rather than a visibility.

This is the streaming counterpart of the
[`AprioriAmplitude`](@ref) pipeline step. The two differ in *what they
calibrate*, not merely in when: `AprioriAmplitude` runs in the output tail,
after the solved gains, so a bandpass fit alongside it is fit on uncalibrated
amplitudes; this transform runs before any solver reads the scan, so the SEFD
scaling — per channel, where the ANTAB declares per-channel Tsys — is divided
out first and does not land in the fitted gains. Use this one when the solve
should see calibrated amplitudes.

`uvset` supplies the array's `rdate` (the epoch the ANTAB's timestamps are
resolved against) and the per-spw channel numbering the ANTAB indexes, both
read from metadata only — the set stays lazy.

The correction leaves the set's `BUNIT` untouched: it is data, not metadata, that
this transform rewrites. Stamp the finished output with
`set_bunit(out, "JY")` if the unit needs to be recorded.
"""
struct AprioriPreCal{A <: UVData.AntabCalibration} <: AbstractDataTransform
    antab::A
    rdate::String
    spw_channel_freqs::Dict{String, Vector{Float64}}
    min_elevation_deg::Float64
    on_missing_station::Symbol
end

function AprioriPreCal(
        uvset::UVSet, antab::UVData.AntabCalibration;
        min_elevation_deg::Real = 0.0, on_missing_station::Symbol = :warn,
    )
    on_missing_station in (:warn, :error, :ignore) || throw(
        ArgumentError(
            "AprioriPreCal: on_missing_station must be :warn, :error, or :ignore " *
                "(got :$(on_missing_station))"
        )
    )
    rdate = String(DimensionalData.metadata(uvset).array_obs.rdate)
    isempty(rdate) && throw(
        ArgumentError(
            "AprioriPreCal: the set's root metadata has an empty `rdate`, so the ANTAB's " *
                "DOY+UT timestamps cannot be resolved to the data's time axis."
        )
    )
    # An ANTAB numbers channels within a spw, so recovering that numbering for a
    # multi-spw scan group needs each spw's own channel order — which is the
    # leaf's, not the geometry's ascending-frequency one (a lower-sideband spw
    # reverses between the two).
    spw_freqs = Dict{String, Vector{Float64}}()
    for leaf in values(DimensionalData.branches(uvset))
        info = DimensionalData.metadata(leaf)
        get!(spw_freqs, String(info.spw_name)) do
            Float64.(collect(info.freq_setup.channel_freqs))
        end
    end
    return AprioriPreCal(
        antab, rdate, spw_freqs, Float64(min_elevation_deg), on_missing_station,
    )
end

"""
    validate_transform(t::AprioriPreCal, geom::DataGeometry, ant_names)

Check that the ANTAB describes stations this set actually has, and that every
spw of the solve geometry was seen when the transform was built. An ANTAB
sharing no station with the set would scale nothing at all, and silently doing
nothing is the outcome worth refusing; `on_missing_station` decides what a
*partially* matching ANTAB does (`:error` refuses, `:warn` reports and leaves
those baselines unscaled).
"""
function validate_transform(t::AprioriPreCal, geom::Calibration.DataGeometry, ant_names)
    known = [n for n in ant_names if haskey(t.antab, n)]
    isempty(known) && throw(
        ArgumentError(
            "AprioriPreCal: the ANTAB $(repr(t.antab.track_label)) shares no station with " *
                "this set, so it would scale nothing. It knows " *
                "$(join(map(repr, sort(collect(String, keys(t.antab)))), ", ")); " *
                "the set has $(join(map(repr, ant_names), ", ")). Check the ANTAB matches " *
                "this track and band."
        )
    )
    absent = [String(n) for n in ant_names if !haskey(t.antab, n)]
    if !isempty(absent)
        if t.on_missing_station === :error
            throw(
                ArgumentError(
                    "AprioriPreCal: the ANTAB $(repr(t.antab.track_label)) has no record for " *
                        "stations $(absent); baselines involving them would go through " *
                        "unscaled. Pass `on_missing_station = :warn` or `:ignore` to allow that."
                )
            )
        elseif t.on_missing_station === :warn
            @warn "AprioriPreCal: the ANTAB has no record for these stations; baselines involving them are left unscaled." stations =
                absent track = t.antab.track_label
        end
    end
    for s in unique(geom.spw_of_chan)
        name = isempty(geom.spw_names) ? "" : geom.spw_names[s]
        haskey(t.spw_channel_freqs, name) || throw(
            ArgumentError(
                "AprioriPreCal: spw $(repr(name)) is in the solve geometry but was not in the " *
                    "`UVSet` the transform was built from, so its channel numbering is unknown. " *
                    "Build the transform from the set being solved."
            )
        )
    end
    return nothing
end

function apply_transform!(
        t::AprioriPreCal, stack::AbstractDimStack, win::GeometryWindow; executor = SerialScheduler(),
    )
    info = DimensionalData.metadata(stack)
    times = Float64.(lookup(stack[:vis], Ti))
    isempty(times) && return nothing
    base_dt = DateTime(Date(t.rdate))
    base_jd = datetime2julian(base_dt)
    jds = [base_jd + h / 24.0 for h in times]
    lo_h, hi_h = extrema(times)
    t_lo = base_dt + Millisecond(round(Int, lo_h * 3_600_000))
    t_hi = base_dt + Millisecond(round(Int, hi_h * 3_600_000))

    pkg = UVData.apriori_gains(
        t.antab, info.antennas, _antab_channel_index(t, win), jds, t_lo, t_hi,
        info.ra, info.dec;
        # Station coverage is settled once, at stream construction
        # (`validate_transform`); repeating it here would warn once per scan group.
        on_missing_station = :ignore, min_elevation_deg = t.min_elevation_deg,
    )
    _scale_apriori!(stack, pkg.gains, executor)
    return nothing
end

apply_transform(uvset::UVSet, t::AprioriPreCal) = UVData.apply_calibration(
    uvset, t.antab;
    on_missing_station = :ignore, min_elevation_deg = t.min_elevation_deg,
)

# The ANTAB channel number of each channel of this window: the position the
# channel holds in its own spw's leaf-order frequency list. Identity for a
# single-spw window whose channels are in file order, and the reason a
# multi-spw or lower-sideband window still reads the right per-channel Tsys.
function _antab_channel_index(t::AprioriPreCal, win::GeometryWindow)
    geom = win.geom
    idx = Vector{Int}(undef, length(win.chan_idx))
    for (c, gc) in pairs(win.chan_idx)
        name = isempty(geom.spw_names) ? "" : geom.spw_names[geom.spw_of_chan[gc]]
        freqs = t.spw_channel_freqs[name]
        f = geom.channel_freqs[gc]
        k = findfirst(g -> isapprox(g, f; rtol = 1.0e-9), freqs)
        k === nothing && error(
            "AprioriPreCal: channel $(f) Hz is not among spw $(repr(name))'s channels; the " *
                "transform was built from a set with a different frequency setup."
        )
        idx[c] = k
    end
    return idx
end

# Divide out real, positive per-(channel, integration, antenna, feed) amplitude
# gains in place. Mirrors `UVData._apply_apriori_kernel`, which does the same on
# a whole leaf: NaN flags the sample, autocorrelations are flagged outright.
function _scale_apriori!(stack::AbstractDimStack, gains::Array{Float64, 4}, executor)
    V = parent(stack[:vis])
    W = parent(stack[:weights])
    bl_pairs = UVData.baselines(stack).pairs
    pols = UVData.pol_products(stack)
    nchan, nti, nbl, npol = size(V)
    cols = [(bi, p) for p in 1:npol for bi in 1:nbl]
    tforeach(cols; scheduler = executor) do col
        bi, p = col
        a, b = bl_pairs[bi]
        if a == b
            for t in 1:nti, c in 1:nchan
                W[c, t, bi, p] = zero(eltype(W))
            end
            return
        end
        fa, fb = UVData.correlation_feed_pair(pols[p])
        for t in 1:nti
            for c in 1:nchan
                w = W[c, t, bi, p]
                (w > 0 && isfinite(w)) || continue
                ga = gains[c, t, a, fa]
                gb = gains[c, t, b, fb]
                if !(isfinite(ga) && isfinite(gb))
                    W[c, t, bi, p] = zero(eltype(W))
                    continue
                end
                V[c, t, bi, p] /= ga * gb
                W[c, t, bi, p] *= (ga * gb)^2
            end
        end
    end
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

Per-station weight correction factors, indexed by the solver's station
index, built from a station-code => factor map. A factor above 1 raises a
station's weights (a correlator claiming more noise than the data carries).
Wrap the result in a [`StationWeightScale`](@ref) transform in the
pipeline's step chain and every scan's weights are corrected as
`w → w·s_a·s_b` at materialization.

This corrects the noise estimate only; visibilities are untouched.
Miscalibrated station weights otherwise bias every weighted quantity
downstream: fringe SNR (∝ √s), the PFA gate, the bandpass and adhoc
accumulations, and the exported weights. Codes absent from the data are
ignored; stations absent from `factors` get `default`.

    uvset = load_fitsidi(path; lazy = true)
    ws = station_weight_scale(uvset, Dict("HS" => 2.0, "GL" => 2.0))
    fit(StationWeightScale(ws) |> FringeFit(), uvset)
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
