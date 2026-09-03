# ── CalibrationSolution: container, geometry, apply, serialization ───────────
#
# A `CalibrationSolution` is the composition of every pipeline step's own
# finished `StepSolution` — each step solved its own private `(model, layout,
# θ)`, never merged with another step's (see `StepSolution`), and the
# solution's gains are the ELEMENTWISE PRODUCT of every step's own gains
# (equivalently a log-space sum, since `evaluate_gains` already returns
# `exp(Σ logamp)·cis(Σ phase)` and a product of `cis`/`exp` factors is a sum of
# their arguments). It is the hand-off object between a solver (e.g.
# `Gustavo.fit`) and the data: `apply_calibration(uvset, sol)` divides every
# leaf's visibilities by the composed gains, and `save_solution`/`load_solution`
# round-trip it through the `Serialization` stdlib.

using Serialization: serialize, deserialize
using Statistics: mean
using DimensionalData: lookup, Ti, DimArray, Dim
using ..UVData: Frequency, Pol, Baseline, Ant, Feed

"""
    StepSolution(name, model, layout, θ, info)

One finished pipeline step's own solved gain model: `model`/`layout` compiled
from that step's own `model_components` alone — never merged with another
step's — and `θ` its own solved parameter vector, plus `info`, that step's own
solver diagnostics. A [`CalibrationSolution`](@ref) is the ordered
`steps::Vector{StepSolution}` a pipeline run produced, one per solve step, in
run order; gains compose multiplicatively across them ([`gains`](@ref),
[`apply_calibration`](@ref)), and `name` (the step's `provides(step)`
capability, e.g. `:fringe`/`:bandpass`/`:refine`/`:adhoc`) is how a later
step's `fit_selection` or a user's [`stage_info`](@ref) or `sol[name]`
looks a step up.

`θ` keeps whatever array type it is given — a labelled `DimArray` as readily as
a `Vector` — subject to two requirements `layout` imposes: `length(θ) ==
layout.nθ`, and 1-based indexing, since `layout` addresses θ by absolute
position. Both are checked on construction. A step stores a copy, never an
alias.
"""
struct StepSolution{M <: StationGainModel, L <: ParameterLayout, V <: AbstractVector{<:Real}}
    name::Symbol
    model::M
    layout::L
    θ::V
    info::NamedTuple

    function StepSolution{M, L, V}(name, model, layout, θ, info) where {M, L, V}
        # `layout` addresses θ by absolute 1-based position (`ComponentPlan.range`)
        # and the term kernels read those blocks under `@inbounds`, so an array
        # with other axes would read out of bounds silently rather than throw.
        Base.require_one_based_indexing(θ)
        length(θ) == layout.nθ || throw(
            DimensionMismatch(
                "StepSolution($(repr(name))): θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)"
            )
        )
        # `copy`, not an alias: the fused output tail builds a step solution per
        # scan group from the run's working θ while sibling groups are still
        # writing their own slots, and each group must correct against its own
        # snapshot.
        return new{M, L, V}(name, model, layout, copy(θ), info)
    end
end

StepSolution(name::Symbol, model::StationGainModel, layout::ParameterLayout, θ::AbstractVector, info::NamedTuple = NamedTuple()) =
    StepSolution{typeof(model), typeof(layout), typeof(θ)}(name, model, layout, θ, info)

"""
    CalibrationSolution(steps, geom, info = NamedTuple(); transforms = (), postcal = ())
    CalibrationSolution(model, layout, geom, θ, info = NamedTuple(); name = :solution,
                        transforms = (), postcal = ())

A solved calibration: the composition of every pipeline step's own finished
[`StepSolution`](@ref). The second form is the common single-model
convenience — a hand-built or extracted solution with exactly one step, named
`name`. `geom::DataGeometry` is the grid every step's own layout was planned
over, and `info` a NamedTuple of solution-level diagnostics (per-scan SNR,
residuals, …) — distinct from each step's own `info`.

Component names are local to each step's own model and may repeat across
steps (e.g. a `bandpass` step and a `fringe` step can both carry an `atmos`
component) — [`component_dimarray`](@ref) and [`component_gains`](@ref)
always take the step explicitly rather than searching for a name across
`steps`.

`transforms` records the data-transform chain the solve materialized its scans
through (precal, weight scaling, caller hooks), so diagnostics can replay it and
the standalone apply path can reproduce `transforms ∘ solution`. `postcal`
records the OUTPUT-chain calibration steps (a-priori amplitude) applied after
the gains and before any reductions, so the standalone apply reproduces them
without re-passing their inputs. Both default empty.
"""
struct CalibrationSolution{G <: DataGeometry, T, P}
    steps::Vector{StepSolution}
    geom::G
    info::NamedTuple
    transforms::Vector{T}
    postcal::Vector{P}
end

function CalibrationSolution(
        steps::AbstractVector{<:StepSolution}, geom::DataGeometry, info::NamedTuple = NamedTuple();
        transforms = (), postcal = (),
    )
    isempty(steps) && throw(ArgumentError("CalibrationSolution: at least one step is required."))
    return CalibrationSolution(
        collect(StepSolution, steps), geom, info,
        UVData._narrow_eltype(transforms), UVData._narrow_eltype(postcal),
    )
end

function CalibrationSolution(
        model::StationGainModel, layout::ParameterLayout, geom::DataGeometry,
        θ::AbstractVector, info::NamedTuple = NamedTuple();
        name::Symbol = :solution, transforms = (), postcal = (),
    )
    return CalibrationSolution([StepSolution(name, model, layout, θ, info)], geom, info; transforms, postcal)
end

# `nant` is a run-wide constant every step's own layout was planned with
# (`plan_parameters(step_model, nant, geom)`), so any step's own layout reports
# the same value.
_nant(sol::CalibrationSolution) = sol.steps[1].layout.nant

function Base.show(io::IO, ::MIME"text/plain", sol::CalibrationSolution)
    println(io, "CalibrationSolution")
    println(io, "  Steps     : ", join(stage_names(sol), ", "))
    println(
        io, "  Grid      : ", _nant(sol), " antennas × ", nchannels(sol.geom), " channels × ",
        ntimes(sol.geom), " times",
    )
    println(io, "  Parameters: ", sum(s.layout.nθ for s in sol.steps))
    println(io, "  Components: ", join(component_names(sol), ", "))
    print(
        io, "  Provenance: ", length(sol.transforms), " transform(s), ",
        length(sol.postcal), " postcal step(s)",
    )
    return io
end

Base.show(io::IO, sol::CalibrationSolution) = print(
    io, "CalibrationSolution(", length(sol.steps), " step(s), ",
    sum(s.layout.nθ for s in sol.steps), " parameters)",
)

# ── Per-stage views: snapshots of the solution as of each pipeline stage ─────

"""
    component_ranges(layout::ParameterLayout) -> Vector{UnitRange{Int}}

The contiguous θ range owned by each component plan (phase components first,
then log-amplitude, matching `layout.plans`) — each plan's own `range`, the span
`plan_parameters` reserved for its leaf.
"""
component_ranges(layout::ParameterLayout) = [p.range for p in layout.plans]

"""
    stage_names(sol::CalibrationSolution) -> Vector{Symbol}

The pipeline steps recorded on `sol`, in run order.
"""
stage_names(sol::CalibrationSolution) = Symbol[s.name for s in sol.steps]

"""
    getindex(sol::CalibrationSolution, name::Symbol) -> CalibrationSolution
    getindex(sol::CalibrationSolution, index) -> CalibrationSolution

Select steps out of `sol`, in run order, as a solution in its own right. A
`Symbol` names one step (see [`stage_names`](@ref)); anything `sol.steps`
accepts selects positionally — `sol[2]` the second step alone, `sol[1:2]` the
first two, `sol[[1, 3]]` the first and third, `sol[:]` every one.

Gains compose multiplicatively across steps, so a selection's gain is exactly
the product of the selected steps' own gains — a step left out contributes no
gain at all, rather than an explicit zeroed θ block over a shared layout. A
leading run `sol[1:i]` is thus the solution AS OF step `i`: apply it, plot it,
or difference it against the next one. Geometry, `info` and both provenance
chains carry over unchanged, so every selection is a valid solution for
[`apply_calibration`](@ref) and `calibrate`, and `sol[:]` reproduces `sol`.

Selecting no step at all is an error: a solution has at least one.
"""
function Base.getindex(sol::CalibrationSolution, index)
    return step_solution(sol, index)
end

Base.firstindex(sol::CalibrationSolution) = firstindex(sol.steps)
Base.lastindex(sol::CalibrationSolution) = lastindex(sol.steps)

"""
    stage_info(sol::CalibrationSolution, name::Symbol) -> NamedTuple

The diagnostics recorded by the step named `name` (detections and per-scan SNR
for the fringe stage, calibrator choice for the bandpass stage, …).
"""
stage_info(sol::CalibrationSolution, name::Symbol) = sol[name].steps[1].info

"""
    component_gains(sol::CalibrationSolution, step::Symbol, plan_index::Integer; ci = :, ti = :)

The complex antenna gains contributed by ONE model component of the step named
`step` alone (θ zeroed everywhere else in that step; every OTHER step already
contributes identity gain here, by construction — no separate zeroing needed),
on the `(channel, time, antenna, feed)` window `ci × ti`. Exact, because
station gains factor multiplicatively over components — the product over
every component of every step reproduces [`gains`](@ref). `plan_index` follows
that step's OWN `layout.plans` (phase components first, then log-amplitude).
"""
function component_gains(sol::CalibrationSolution, step::Symbol, plan_index::Integer; ci = Colon(), ti = Colon())
    s = sol[step].steps[1]
    1 <= plan_index <= length(s.layout.plans) || throw(
        ArgumentError(
            "component_gains: plan_index $plan_index out of range 1:$(length(s.layout.plans)) " *
                "for step $(repr(step))"
        )
    )
    rng = component_ranges(s.layout)
    θm = fill!(similar(s.θ), 0)
    θm[rng[plan_index]] = s.θ[rng[plan_index]]
    ev = GainEvaluator(s.model, s.layout)
    civ = ci === Colon() ? (1:nchannels(sol.geom)) : ci
    tiv = ti === Colon() ? (1:ntimes(sol.geom)) : ti
    return evaluate_gains(ev, θm, civ, tiv)
end

"""
    component_names(sol::CalibrationSolution) -> Vector{Symbol}

The top-level model component names across every step of `sol` (phase and
log-amplitude groups combined, in step then declaration order, duplicates
dropped) — what `show` lists under `Components:`. A name here can belong to
several steps at once (component names are local to each step's own model);
use [`stage_names`](@ref) and a step-qualified [`component_dimarray`](@ref) or
[`component_gains`](@ref) call to reach one unambiguously.
"""
function component_names(sol::CalibrationSolution)
    names = Symbol[]
    for s in sol.steps, k in (keys(s.model.phase)..., keys(s.model.logamp)...)
        k in names || push!(names, k)
    end
    return names
end

# The elementwise product of every step's own gains over the full grid —
# each step's own `evaluate_gains` already reads as identity gain wherever
# that step has no component, so the product across steps is the exact same
# total gain a single merged evaluator would have produced, without ever
# building one.
function _composed_gains(sol::CalibrationSolution)
    s1 = sol.steps[1]
    g = evaluate_gains(GainEvaluator(s1.model, s1.layout), s1.θ)
    for s in view(sol.steps, 2:length(sol.steps))
        g .*= evaluate_gains(GainEvaluator(s.model, s.layout), s.θ)
    end
    return g
end

# Windowed counterpart of `_composed_gains(sol)` — see `evaluate_gains`'s
# windowed method.
function _composed_gains(
        sol::CalibrationSolution,
        chan_idx::AbstractVector{<:Integer}, ti_idx::AbstractVector{<:Integer},
    )
    s1 = sol.steps[1]
    g = evaluate_gains(GainEvaluator(s1.model, s1.layout), s1.θ, chan_idx, ti_idx)
    for s in view(sol.steps, 2:length(sol.steps))
        g .*= evaluate_gains(GainEvaluator(s.model, s.layout), s.θ, chan_idx, ti_idx)
    end
    return g
end

# Foreign-grid counterpart: each target sample placed in the solve segment it
# belongs to, so the data need not be sampled on the solve grid.
function _composed_gains(
        sol::CalibrationSolution, target::DataGeometry;
        chan_idx::AbstractVector{<:Integer} = Base.OneTo(nchannels(target)),
        ti_idx::AbstractVector{<:Integer} = Base.OneTo(ntimes(target)),
        time_span = nothing,
    )
    s1 = sol.steps[1]
    g = evaluate_gains(
        GainEvaluator(s1.model, s1.layout), s1.θ, sol.geom, target; chan_idx, ti_idx, time_span,
    )
    for s in view(sol.steps, 2:length(sol.steps))
        g .*= evaluate_gains(
            GainEvaluator(s.model, s.layout), s.θ, sol.geom, target; chan_idx, ti_idx, time_span,
        )
    end
    return g
end

"""
    gains(sol::CalibrationSolution) -> DimArray

The solved complex antenna gains on the full grid of `sol`'s geometry, labelled
for inspection: a `DimArray` over `(Frequency, Ti, Ant, Feed)` — channel
frequencies (Hz), integration times (hours), antennas (named when `sol.info`
carries `ant_names`, else `1:nant`), and feed. `gain = exp(Σ logamp) · cis(Σ
phase)`, summed over every step's own components, is the same forward map
[`apply_calibration`](@ref) divides by (and [`save_solution_hdf5`](@ref)
writes); `abs.(gains(sol))` and `angle.(gains(sol))` recover amplitude and
phase.
"""
function gains(sol::CalibrationSolution)
    g = _composed_gains(sol)   # (nchan, ntime, nant, nfeed)
    ants = hasproperty(sol.info, :ant_names) && length(sol.info.ant_names) == size(g, 3) ?
        collect(sol.info.ant_names) : (1:size(g, 3))
    return DimArray(
        g,
        (
            Frequency(sol.geom.channel_freqs), Ti(sol.geom.times),
            Ant(ants), Feed(1:size(g, 4)),
        ),
    )
end

# ── component_dimarray: one component's θ leaf as a labelled DimArray ────────

"""
    component_dimarray(sol::CalibrationSolution, step, group::Symbol, name::Symbol...) -> DimArray

The raw θ leaf of one model component, wrapped as a labelled `DimArray` for
inspection. `step` selects the owning step, by name (`Symbol`, see
[`stage_names`](@ref)) or by position (`Integer`, into `sol.steps`) — always
explicit, never searched for: component names are local to a step's own model
and routinely repeat across steps (e.g. a `bandpass` step and a `fringe` step
can both carry an `atmos` component). `group` is `:phase` or `:logamp`;
further `name`s descend into a multi-emit wrapper's subtree
(`component_dimarray(sol, :fringe, :phase, :sbd, :delay)`).

The result carries the leaf's five axes `(param, feed, Frequency, Ti, Ant)` —
the second is `Feed` when the component is fit per feed, else the tied `node`
axis — with coordinates materialized from `sol`'s geometry: a `Frequency`
segment's centre channel frequency (Hz), a `Ti` segment's mean epoch (hours),
feed/node ids, and antenna names (from `sol.info.ant_names` when present,
else `1:nant`).

The wrap shares data with θ (no copy): an inspection view, never stored on the
solution and never fed through the solve or AD.
"""
function component_dimarray(sol::CalibrationSolution, step, group::Symbol, name::Symbol...)
    path = (group, name...)
    s = sol[step].steps[1]
    ok, plan = _try_descend(s.layout.plantree, path)
    ok || throw(
        ArgumentError("component_dimarray: step $(repr(step)) has no component named $(join(path, '.')).")
    )
    if plan isa GroupedComponentPlan
        gk = join(keys(plan.groups), ", ")
        throw(
            ArgumentError(
                "component_dimarray: $(join(path, '.')) is station-heterogeneous; " *
                    "descend to one of its signature groups ($gk), or iterate them " *
                    "with `station_blocks`."
            )
        )
    end
    plan isa ComponentPlan || throw(
        ArgumentError(
            "component_dimarray: path $(join(path, '.')) names a component group, not a leaf; " *
                "descend to a named component."
        )
    )
    _, axnode = _try_descend(s.layout.axes, path)
    stations = hasproperty(axnode, :stations) ? axnode.stations : nothing
    raw = _component_leaf(plan, s.θ)
    dims = ntuple(d -> _role_dim(axnode.roles[d], size(raw, d), sol, plan; stations), ndims(raw))
    return DimArray(raw, dims; name = last(path))
end

# Attempt to descend `tree` (a step's own `layout.plantree` or `layout.axes`)
# by `path`; `(false, nothing)` without throwing when a name is absent along
# the way. A `GroupedComponentPlan` descends into its signature groups, so a
# path may address one group's leaf directly (`:phase, :bandpass, :g1`).
function _try_descend(tree, path::Tuple{Vararg{Symbol}})
    node = tree
    for s in path
        children = node isa GroupedComponentPlan ? node.groups : node
        (children isa NamedTuple && haskey(children, s)) || return false, nothing
        node = getproperty(children, s)
    end
    return true, node
end

# The DimensionalData dimension for one leaf axis, from its role and the
# solution's geometry. A segment axis takes a representative coordinate per
# segment (a frequency segment's centre, a time segment's mean epoch); the
# antenna axis takes station names when the solution carries them — for a
# signature group's leaf, the names of just the group's `stations`; `:param`
# and `:node` are positional id spaces with no physical coordinate.
function _role_dim(role::Symbol, n::Int, sol::CalibrationSolution, plan::ComponentPlan; stations = nothing)
    if role === :Frequency
        groups = segment_groups(plan.fseg_id, n)
        return Frequency([mean(view(sol.geom.channel_freqs, g)) for g in groups])
    elseif role === :Ti
        groups = segment_groups(plan.tseg_id, n)
        return Ti([mean(view(sol.geom.times, g)) for g in groups])
    elseif role === :Ant
        an = hasproperty(sol.info, :ant_names) ? sol.info.ant_names : nothing
        ants = if stations === nothing
            an !== nothing && length(an) == n ? collect(an) : (1:n)
        else
            an !== nothing && all(i -> 1 <= i <= length(an), stations) ?
                collect(an)[stations] : collect(stations)
        end
        return Ant(ants)
    elseif role === :Feed
        return Feed(1:n)
    else
        return Dim{role}(1:n)
    end
end

function step_solution(sol::CalibrationSolution, index)
    stp = sol.steps[index]
    stpout = stp isa AbstractVector ? stp : [stp]
    return CalibrationSolution(
        stpout, sol.geom, sol.info; transforms = sol.transforms, postcal = sol.postcal,
    )
end

function step_solution(sol::CalibrationSolution, name::Symbol)
    i = findfirst(s -> s.name === name, sol.steps)
    i === nothing && throw(
        ArgumentError("solution has no stage $(repr(name)); recorded stages: $(stage_names(sol)).")
    )
    return step_solution(sol, i)
end


# ── Geometry from a UVSet ────────────────────────────────────────────────────

"""
    build_geometry(uvset; f0 = nothing, t0 = nothing) -> DataGeometry

Build the union `DataGeometry` spanning every leaf of `uvset`: the sorted unique
channel frequencies (Hz) of all bands, the sorted unique integration times
(hours) of all scans, with `spw_of_chan` / `scan_of_time` dense-ranked from the
leaves' `spw_name` / `scan_name`. `f0` defaults to the mean channel frequency,
`t0` to the first time. Assumes a single consistent antenna table across the
set (used only to size the solve elsewhere).
"""
function build_geometry(uvset::UVSet; f0 = nothing, t0 = nothing)
    # Collect (freq, spw_name) and (time, scan_name) observations from all leaves.
    freq_spw = Dict{Float64, String}()
    time_scan = Dict{Float64, String}()
    for (_, leaf) in UVData.branches(uvset)
        info = UVData.metadata(leaf)
        fs = channel_freqs(info.freq_setup)
        for f in fs
            fk = Float64(f)
            prev = get(freq_spw, fk, nothing)
            (prev === nothing || prev == info.spw_name) || throw(
                ArgumentError(
                    "build_geometry: channel frequency $fk Hz appears in conflicting spectral " *
                        "windows '$prev' and '$(info.spw_name)' — a single concatenated channel " *
                        "axis cannot dense-rank it to one spw. Partition the set so each " *
                        "frequency belongs to one spw, or rename the spws consistently."
                )
            )
            freq_spw[fk] = info.spw_name
        end
        ts = lookup(leaf[:vis], Ti)
        for t in ts
            tk = Float64(t)
            prev = get(time_scan, tk, nothing)
            (prev === nothing || prev == info.scan_name) || throw(
                ArgumentError(
                    "build_geometry: time $tk h appears in conflicting scans '$prev' and " *
                        "'$(info.scan_name)' — a single concatenated time axis cannot dense-rank " *
                        "it to one scan. Check for overlapping scan windows or inconsistent " *
                        "scan names."
                )
            )
            time_scan[tk] = info.scan_name
        end
    end

    freqs = sort!(collect(keys(freq_spw)))
    times = sort!(collect(keys(time_scan)))

    spw_labels = [freq_spw[f] for f in freqs]
    scan_labels = [time_scan[t] for t in times]

    spw_of_chan, _ = _dense_rank_labels(spw_labels)
    scan_of_time, _ = _dense_rank_labels(scan_labels)

    spw_names = _unique_in_order(spw_labels)
    scan_names = _unique_in_order(scan_labels)

    f0v = isnothing(f0) ? (isempty(freqs) ? 0.0 : mean(freqs)) : Float64(f0)
    t0v = isnothing(t0) ? (isempty(times) ? 0.0 : first(times)) : Float64(t0)

    return DataGeometry(;
        times = times,
        channel_freqs = freqs,
        scan_of_time = scan_of_time,
        spw_of_chan = spw_of_chan,
        t0 = t0v,
        f0 = f0v,
        scan_names = scan_names,
        spw_names = spw_names,
    )
end

# Dense-rank string labels to 1..k preserving first-appearance order.
function _dense_rank_labels(labels::AbstractVector{<:AbstractString})
    seen = Dict{String, Int}()
    out = Vector{Int}(undef, length(labels))
    next = 0
    for i in eachindex(labels)
        id = get(seen, labels[i], 0)
        if id == 0
            next += 1
            seen[labels[i]] = next
            id = next
        end
        out[i] = id
    end
    return out, next
end

function _unique_in_order(labels::AbstractVector{<:AbstractString})
    seen = Set{String}()
    out = String[]
    for l in labels
        if !(l in seen)
            push!(seen, l)
            push!(out, l)
        end
    end
    return out
end

"""
    GeometryWindow

A window into a [`DataGeometry`](@ref): everything needed to locate one scan's
channels and times in the solve's index space, with no data attached.

- `geom` — the solve's geometry, the index space the window addresses.
- `chan_idx`, `ti_idx` — GLOBAL indices into `geom.channel_freqs` / `geom.times`
  of the channels and times the window covers.

θ is addressed by POSITION — a component's leaf indexed `(param, node, fseg_id,
tseg_id, ant)` over global-length segment-id tables — while a `DimStack`'s
coordinates are PHYSICAL (Hz, hours), so this join cannot be recovered from the
data alone.
Build one with [`leaf_window`](@ref); pass it alongside the scan's `DimStack`
to anything that needs both.
"""
struct GeometryWindow
    geom::DataGeometry
    chan_idx::Vector{Int}
    ti_idx::Vector{Int}
end

"""
    leaf_window(geom::DataGeometry, leaf) -> GeometryWindow

The [`GeometryWindow`](@ref) addressing the channels and times `leaf` carries,
matched by value against `geom` (frequency by `isapprox` rtol 1e-9, time by atol
1e-9 h). Errors if any leaf sample has no match in the geometry.
"""
leaf_window(geom::DataGeometry, leaf) =
    GeometryWindow(geom, _channel_indices(geom, leaf), _time_indices(geom, leaf))

function _channel_indices(geom::DataGeometry, leaf)
    fs = lookup(leaf[:vis], Frequency)
    idx = Vector{Int}(undef, length(fs))
    for (i, f) in enumerate(fs)
        j = findfirst(g -> isapprox(g, Float64(f); rtol = 1.0e-9), geom.channel_freqs)
        isnothing(j) &&
            throw(ArgumentError("leaf_window: channel frequency $f not found in geometry"))
        idx[i] = j
    end
    return idx
end

function _time_indices(geom::DataGeometry, leaf)
    ts = lookup(leaf[:vis], Ti)
    idx = Vector{Int}(undef, length(ts))
    for (i, t) in enumerate(ts)
        j = findfirst(g -> isapprox(g, Float64(t); atol = 1.0e-9), geom.times)
        isnothing(j) && throw(ArgumentError("leaf_window: time $t not found in geometry"))
        idx[i] = j
    end
    return idx
end

# ── Apply ────────────────────────────────────────────────────────────────────

"""
    apply_calibration(uvset::UVSet, sol::CalibrationSolution; apply_flags = true,
                      transforms = sol.transforms) -> UVSet

Apply the solution's RECORDED transform chain (`sol.transforms` — e.g. a
station weight scale and an earlier solution applied as a data transform), then
divide every leaf's visibilities by the solution's per-antenna gains. For a
baseline `(a, b)` and correlation product `p` with feeds `(fa, fb)`:

    V_corr = V / (g_a[fa] · conj(g_b[fb])),    W_corr = W · |g_a · g_b|²

Samples where either gain magnitude underflows are flagged (weight 0, vis NaN).

`transforms` defaults to the solution's own recorded chain, so the corrected
set carries the SAME total correction `calibrate(sol, uvset)` produces (minus
its `postcal`/`reduce` tail) — weights included, which matters to anything
that reads them as noise claims. Pass `transforms = ()` to apply the gains
alone: the right call when the data has already been transform-corrected (the
streaming passes do this), and the semantics every internal replay site uses.

Each channel and time is placed in the segment of `sol` it belongs to — matched
by spw and scan identity — so `uvset` may be sampled differently from the solve:
a bandpass fit on scan-averaged data corrects data at full time resolution. A
sample the solution has no segment for is rejected, as is one whose recorded
span crosses a bin boundary — see `evaluate_gains`.

`apply_flags` (default `true`) additionally zero-weights baselines touching a
(station, scan) the solve left UNCONSTRAINED, when `sol.info` records them —
identity gains, i.e. the data would pass through uncalibrated. This is the
EHT-HOPS flag semantic: a station is flagged per scan only when, after the
closure-screened global solve, no strong detection constrains it; a merely weak
baseline between two constrained stations is NOT flagged (it is calibrated by
SNR transfer).
"""
# Whole-set replay of a recorded transform chain. The transform types (and the
# working method) live in the Streaming layer, which loads after this module;
# the stub exists so `apply_calibration` can replay a chain without a layering
# inversion.
function _replay_transforms end

function UVData.apply_calibration(
        uvset::UVSet, sol::CalibrationSolution;
        apply_flags::Bool = true, executor = DynamicScheduler(),
        transforms = sol.transforms,
    )
    for t in transforms
        t === missing && throw(
            ArgumentError(
                "apply_calibration: this solution records a transform that did not survive " *
                    "serialization (saved as `missing`) — re-fit, or apply the original " *
                    "transform chain manually (pass `transforms = ()` to apply gains alone).",
            )
        )
    end
    isempty(transforms) || (uvset = _replay_transforms(uvset, transforms))
    flagged = apply_flags ? _solution_flag_sets(sol.info) : nothing
    # Placement is against the TARGET SET's own geometry, not a leaf's: a
    # channel-index segmentation (`ChannelBlocks`, `FreqGroups`) is defined on
    # the set's whole concatenated channel axis, which one band's leaf does not
    # carry. `leaf_window` then says which of its samples each leaf holds.
    target = build_geometry(uvset)
    return UVData.apply(uvset) do leaf, info, root
        # A lazy leaf materializes to freshly-decoded private arrays we correct
        # in place; an eager leaf is caller-owned, so copy it first. Capture
        # laziness before `materialize_leaf` collapses it.
        private = is_lazy(leaf)
        leaf = materialize_leaf(leaf)
        private || (
            leaf = rebuild_visibilities(
                leaf, copy(parent(leaf[:vis])), copy(parent(leaf[:weights])),
            )
        )
        win = leaf_window(target, leaf)
        g = _composed_gains(                                 # (nchan_leaf, nti_leaf, nant, 2)
            sol, target;
            chan_idx = win.chan_idx, ti_idx = win.ti_idx, time_span = info.time_span,
        )
        _apply_gains!(leaf, g; executor)
        _flag_solution_rows!(
            leaf[:vis], leaf[:weights], UVData.baselines(leaf).pairs,
            _geom_scan_id(sol.geom, info.scan_name), flagged,
        )
        return leaf
    end
end

# The solution's unconstrained (station, geometry scan id) pairs as a lookup set,
# `nothing` when the solution records none.
function _solution_flag_sets(info::NamedTuple)
    (haskey(info, :flagged_ant) && !isempty(info.flagged_ant)) || return nothing
    return Set{Tuple{Int, Int}}(
        (Int(info.flagged_ant[i]), Int(info.flagged_scan[i]))
            for i in eachindex(info.flagged_ant)
    )
end

# The solution's own scan id for a scan label, 0 when the solve never saw it.
# Flags are recorded against these ids, so a label join is what locates them —
# the leaf's epochs need not appear in the solve grid.
_geom_scan_id(geom::DataGeometry, scan_name) =
    something(findfirst(==(String(scan_name)), geom.scan_names), 0)

# Zero-weight (and NaN) whole baseline rows touching a (station, scan) the solve
# left unconstrained — identity gains, so the data would pass through
# uncalibrated. A leaf spans ONE scan, hence one scan id.
function _flag_solution_rows!(Vc, Wc, bl_pairs, scanid::Integer, flagged)
    flagged === nothing && return nothing
    @inbounds for bi in eachindex(bl_pairs)
        a, b = bl_pairs[bi]
        a == b && continue
        ((a, scanid) in flagged || (b, scanid) in flagged) || continue
        Wc[:, :, bi, :] .= zero(eltype(Wc))
        Vc[:, :, bi, :] .= convert(eltype(Vc), NaN)
    end
    return nothing
end

# Gain magnitude below which a cell is treated as unconstrained rather than
# divided through: the correction would amplify noise without bound.
const _GAIN_FLOOR = 1.0e-12

# Correct one (baseline, product) column in place over its (Frequency, Ti)
# plane: `V ← V / (g_a conj(g_b))` and `w ← w · |g_a g_b|²`, where `ga`/`gb` are
# the two stations' gains on that same plane. A degenerate or non-finite gain
# blanks the cell — NaN visibility, zero weight — which is what marks it flagged
# downstream. Each cell reads then writes its own index, so the update is exact
# even though the read and the write hit the same array.
function _correct_column!(vis, w, ga, gb)
    for i in eachindex(vis, w, ga, gb)
        gai = ga[i]
        gbi = gb[i]
        den = gai * conj(gbi)
        if abs(gai) < _GAIN_FLOOR || abs(gbi) < _GAIN_FLOOR || !isfinite(den)
            vis[i] = convert(eltype(vis), NaN)
            w[i] = zero(eltype(w))
        else
            vis[i] = vis[i] / den
            w[i] = w[i] * abs2(gai * gbi)
        end
    end
    return nothing
end

# Correct a leaf's visibilities in place by its complex antenna gains
# `g[c, ti, ant, feed]`. Baselines and correlation products are read off the
# leaf, so they cannot disagree with the arrays they index. The caller owns the
# copy-or-mutate decision: pass a private leaf to overwrite it, or a copy to
# leave the source untouched.
#
# Each (baseline, product) column is independent — disjoint writes over
# read-only gains — so the columns fan out over the inner `executor`. That
# fan-out is load-bearing: this kernel is ≈80% of `apply_calibration` and the
# split is worth ~5× on 8 threads. It is also bit-identical to the serial loop,
# because every cell is the same scalar expression whatever the partition.
function _apply_gains!(leaf, g::AbstractArray{<:Complex, 4}; executor = SerialScheduler())
    vis = leaf[:vis]
    w = leaf[:weights]
    ants = UVData.baselines(leaf).pairs
    feeds = map(correlation_feed_pair, pol_products(leaf))
    columns = vec(CartesianIndices((axes(vis, 3), axes(vis, 4))))
    tforeach(columns; scheduler = executor) do col
        bi, p = Tuple(col)
        a, b = ants[bi]
        fa, fb = feeds[p]
        _correct_column!(
            view(vis, :, :, bi, p), view(w, :, :, bi, p),
            view(g, :, :, a, fa), view(g, :, :, b, fb),
        )
    end
    return leaf
end

# ── Serialization ────────────────────────────────────────────────────────────

"""
    save_solution(path, sol::CalibrationSolution)

Serialize `sol` to `path` via the `Serialization` stdlib inside a versioned
wrapper NamedTuple. The current version is 5 (`CalibrationSolution` composed
from per-step `StepSolution`s rather than one merged model); earlier versions
are refused on load. Transforms that close over caller code (e.g. a
`CalFunction`) serialize only within the same code state; a transform (or
postcal step) that fails to serialize is recorded as `missing` with a warning
rather than failing the save.
"""
function save_solution(path::AbstractString, sol::CalibrationSolution)
    wrapper = (;
        version = 5, sol.steps, sol.geom, sol.info,
        transforms = _serializable_transforms(sol.transforms),
        postcal = _serializable_transforms(sol.postcal),
    )
    serialize(path, wrapper)
    return path
end

function _serializable_transforms(ts)
    out = Any[]
    for t in ts
        ok = try
            serialize(IOBuffer(), t)
            true
        catch
            false
        end
        if ok
            push!(out, t)
        else
            @warn "save_solution: transform $(typeof(t)) is not serializable — recorded as `missing`."
            push!(out, missing)
        end
    end
    return out
end

"""
    load_solution(path) -> CalibrationSolution

Inverse of [`save_solution`](@ref). Only current-format (version 5) files are
supported; files from an earlier Gustavo used a different solution shape and
are refused — re-solve to produce a current-format solution.
"""
function load_solution(path::AbstractString)
    w = deserialize(path)
    w.version == 5 || error(
        "load_solution: unsupported version $(w.version) — saved by an incompatible " *
            "Gustavo (the solution shape changed); re-solve to produce a current file.",
    )
    return CalibrationSolution(w.steps, w.geom, w.info; transforms = w.transforms, postcal = w.postcal)
end

"""
    save_solution_hdf5(path, sol::CalibrationSolution; gains = true, time_block = 1024)

Write `sol` to an HDF5 caltable readable from any language (Python/h5py, CASA, …),
NOT just Julia. Provided by `GustavoHDF5Ext` — load `HDF5` to enable it.

Layout:
- `gain/real`, `gain/imag` — the evaluated complex antenna gains on the
  `(channel, time, antenna, feed)` grid (`Float32`, chunked + gzip). Written in
  blocks of `time_block` integrations so the full ~GB cube is never resident. Omit
  with `gains = false` to write only the compact parametric form.
- `axes/*` — `channel_freq_hz`, `time`, `scan_of_time`, `spw_of_chan`, `f0`, `t0`.
- `info/*` — the solution-level (run-wide) diagnostics: antenna/scan counts,
  station names, flags, executor/timing summary counters.
- `info/steps/<name>/*` — each pipeline step's OWN diagnostics
  (`stage_info(sol, name)`), one subgroup per step, written generically —
  a third-party `SolveStep`'s custom diagnostics appear here automatically,
  with no changes needed to the writer. A `NamedTuple`- or `DimStack`-valued
  entry (e.g. the fringe step's per-scan `scan_snr`/detection table, any
  step's `timing`) recurses into its own further subgroup.
- root attributes — format/version, units, and the gain convention
  `V_corr = V / (g_a · conj(g_b))`, `weight ×= |g_a g_b|²`.
- `julia/blob` — the `Serialization` bytes of `(steps, geom, info)` so Julia can
  round-trip the solution losslessly (external readers ignore it).
"""
function save_solution_hdf5 end

"""
    load_solution_hdf5(path) -> CalibrationSolution

Reconstruct a `CalibrationSolution` from an HDF5 file written by
[`save_solution_hdf5`](@ref) (via its `julia/blob`). Provided by `GustavoHDF5Ext`.
"""
function load_solution_hdf5 end
