# ── CalibrationSolution: container, geometry, apply, serialization ───────────
#
# A `CalibrationSolution` bundles a solved `StationGainModel`, its
# `ParameterLayout`, the `DataGeometry` it was solved over, and the flattened
# parameter vector θ, plus a free-form `info` NamedTuple of diagnostics. It is
# the hand-off object between a solver (e.g. `Gustavo.fit`) and
# the data: `apply_calibration(uvset, sol)` divides every leaf's visibilities by
# the model's per-antenna gains, and `save_solution`/`load_solution` round-trip
# it through the `Serialization` stdlib.

using Serialization: serialize, deserialize
using Statistics: mean
using DimensionalData: lookup, Ti
using ..UVData: Frequency, Pol, Baseline

"""
    StageRecord(name, step_index, phase_comps, logamp_comps, info)

Per-stage provenance on a [`CalibrationSolution`](@ref): which model components
(indices into `model.phase` / `model.logamp`) the pipeline stage `name` owns —
i.e. whose θ blocks it solved — plus that stage's own diagnostics. The records
are ordered as the stages ran, which is what makes "the solution as of stage k"
well-defined (see [`stage_solution`](@ref)).
"""
struct StageRecord
    name::Symbol
    step_index::Int
    phase_comps::Vector{Int}
    logamp_comps::Vector{Int}
    info::NamedTuple
end

"""
    CalibrationSolution(model, layout, geom, θ, info; stages = StageRecord[],
                        transforms = (), postcal = ())

A solved station gain model. `model::StationGainModel` is the shared model,
`layout::ParameterLayout` its flattened parameter plan over `geom::DataGeometry`,
`θ` the solved parameter vector, and `info` a NamedTuple of solver diagnostics
(per-scan SNR, χ, residuals, …).

`θ` keeps whatever array type it is given — a labelled `DimArray` as readily as a
`Vector` — subject to two requirements the layout imposes: `length(θ) == layout.nθ`,
and 1-based indexing, since `layout` addresses θ by absolute position. Both are
checked on construction. The solution stores a copy, never an alias.

`stages` records per-pipeline-stage provenance ([`StageRecord`](@ref)) — index
the solution by stage name (`sol[:bandpass]`) for the solution-as-of-that-stage
and its diagnostics. `transforms` records the data-transform chain the solve
materialized its scans through (precal, weight scaling, caller hooks), so
diagnostics can replay it and the standalone apply path can reproduce
`transforms ∘ solution`. `postcal` records the OUTPUT-chain calibration steps
(a-priori amplitude) applied after the gains and before any reductions, so the
standalone apply reproduces them without re-passing their inputs. All default
empty (a plain single-stage solution).
"""
struct CalibrationSolution{
        M <: StationGainModel, L <: ParameterLayout, G <: DataGeometry,
        V <: AbstractVector{<:Real}, T, P,
    }
    model::M
    layout::L
    geom::G
    θ::V
    info::NamedTuple
    stages::Vector{StageRecord}
    transforms::Vector{T}
    postcal::Vector{P}
end

function CalibrationSolution(
        model::StationGainModel, layout::ParameterLayout, geom::DataGeometry,
        θ::AbstractVector, info::NamedTuple = NamedTuple();
        stages = StageRecord[], transforms = (), postcal = (),
    )
    # The layout addresses θ by absolute 1-based position (`ComponentPlan.off1`)
    # and the term kernels read those offsets under `@inbounds`, so an array with
    # other axes would read out of bounds silently rather than throw.
    Base.require_one_based_indexing(θ)
    length(θ) == layout.nθ || throw(
        DimensionMismatch(
            "CalibrationSolution: θ has length $(length(θ)), expected layout.nθ = $(layout.nθ)"
        )
    )
    # `copy`, not an alias: the fused output tail builds a solution per scan group
    # from the run's working θ while sibling groups are still writing their own
    # slots, and each group must correct against its own snapshot.
    return CalibrationSolution(
        model, layout, geom, copy(θ), info,
        collect(StageRecord, stages),
        UVData._narrow_eltype(transforms), UVData._narrow_eltype(postcal),
    )
end

# ── Per-stage views: snapshots of the solution as of each pipeline stage ─────

"""
    component_ranges(layout::ParameterLayout) -> Vector{UnitRange{Int}}

The contiguous θ range owned by each component plan (phase components first,
then log-amplitude, matching `layout.plans`). Exact because `plan_parameters`
assigns blocks strictly sequentially: a component's parameters are the
contiguous run from its first assigned offset up to the next component's first.
A component that planned no parameters gets an empty range.
"""
function component_ranges(layout::ParameterLayout)
    starts = Vector{Int}(undef, length(layout.plans))
    for (i, p) in enumerate(layout.plans)
        s = typemax(Int)
        for v in p.off1
            0 < v < s && (s = v)
        end
        for v in p.off2
            0 < v < s && (s = v)
        end
        starts[i] = s == typemax(Int) ? 0 : s
    end
    ends = zeros(Int, length(starts))
    nxt = layout.nθ + 1
    for i in reverse(eachindex(starts))
        starts[i] == 0 && continue
        ends[i] = nxt - 1
        nxt = starts[i]
    end
    return [starts[i] == 0 ? (1:0) : (starts[i]:ends[i]) for i in eachindex(starts)]
end

"""
    StageView

A view of one pipeline stage of a [`CalibrationSolution`](@ref), obtained by
indexing the solution with the stage name: `sol[:fringe]`, `sol[:bandpass]`, ….
[`stage_solution`](@ref) gives the full solution AS OF that stage;
[`stage_info`](@ref) gives the stage's own diagnostics.
"""
struct StageView{S <: CalibrationSolution}
    sol::S
    k::Int
end

"""
    stage_names(sol::CalibrationSolution) -> Vector{Symbol}

The pipeline stages recorded on `sol`, in run order (empty for a plain
single-stage solution).
"""
stage_names(sol::CalibrationSolution) = Symbol[r.name for r in sol.stages]

function Base.getindex(sol::CalibrationSolution, name::Symbol)
    k = findfirst(r -> r.name === name, sol.stages)
    k === nothing && throw(
        ArgumentError("solution has no stage $(repr(name)); recorded stages: $(stage_names(sol)).")
    )
    return StageView(sol, k)
end

"""
    stage_info(sv::StageView) -> NamedTuple

The diagnostics recorded by this stage (detections and per-scan SNR for the
fringe stage, calibrator choice for the bandpass stage, …).
"""
stage_info(sv::StageView) = sv.sol.stages[sv.k].info

"""
    stage_solution(sv::StageView) -> CalibrationSolution

The full solution AS OF this stage: the θ blocks of every later stage's
components are zeroed, which is exactly identity gains for every term (phase 0,
log-amplitude 0). Gauge pins live in the model structure and are untouched, so
the snapshot stays on the same per-scan gauge the solve chose. Apply it, plot
it, or difference it against the next stage's snapshot.
"""
function stage_solution(sv::StageView)
    sol = sv.sol
    rng = component_ranges(sol.layout)
    θm = copy(sol.θ)
    for j in (sv.k + 1):length(sol.stages)
        r = sol.stages[j]
        for ci in r.phase_comps
            θm[rng[ci]] .= 0.0
        end
        for cj in r.logamp_comps
            θm[rng[sol.layout.nphase + cj]] .= 0.0
        end
    end
    return CalibrationSolution(
        sol.model, sol.layout, sol.geom, θm, sol.info;
        stages = sol.stages[1:sv.k], transforms = sol.transforms, postcal = sol.postcal,
    )
end

"""
    component_gains(sol::CalibrationSolution, plan_index::Integer; ci = :, ti = :)

The complex antenna gains contributed by ONE model component alone (θ zeroed
everywhere else), on the `(channel, time, antenna, feed)` window `ci × ti`.
Exact, because station gains factor multiplicatively over components — the
product over all components reproduces `evaluate_gains` on the full θ. Plan
indices follow `layout.plans` (phase components first, then log-amplitude).
"""
function component_gains(sol::CalibrationSolution, plan_index::Integer; ci = Colon(), ti = Colon())
    1 <= plan_index <= length(sol.layout.plans) || throw(
        ArgumentError(
            "component_gains: plan_index $plan_index out of range 1:$(length(sol.layout.plans))"
        )
    )
    rng = component_ranges(sol.layout)
    θm = fill!(similar(sol.θ), 0)
    θm[rng[plan_index]] = sol.θ[rng[plan_index]]
    ev = GainEvaluator(sol.model, sol.layout)
    civ = ci === Colon() ? (1:nchannels(sol.geom)) : ci
    tiv = ti === Colon() ? (1:ntimes(sol.geom)) : ti
    return evaluate_gains(ev, θm, civ, tiv)
end

"""
    bandpass_solution(sol::CalibrationSolution) -> CalibrationSolution

Extract the BANDPASS alone from a fitted solution: a new solution whose model
carries only the frequency-resolved (`ChannelBlocks`) phase / log-amplitude
components, with their θ blocks copied — nothing else (no per-scan delays/rates,
no adhoc).
Because the bandpass is time-stable (`GlobalTime`), the extracted solution is
PORTABLE: apply it in a LATER pipeline run as a precal transform at the head of
the chain,

    bp = bandpass_solution(sol_calibrators)
    fit(ApplySolution(bp) |> FringeFit(...), uvset_full)

so the fringe search runs on bandpass-corrected data and the new solve fits the
correction ON TOP of it (fit-on-a-few-scans / apply-everywhere, across runs —
the transform is recorded on the new solution and replayed by `calibrate`).
Cross-set application matches stations BY NAME and requires the identical
channel layout (see [`Gustavo.Fringe.ApplySolution`](@ref)); stations absent
from the extraction get identity gains.
"""
function bandpass_solution(sol::CalibrationSolution)
    pcs = collect(sol.model.phase)
    lcs = collect(sol.model.logamp)
    pidx = findall(_is_bandpass, pcs)
    lidx = findall(_is_bandpass, lcs)
    isempty(pidx) && isempty(lidx) && throw(
        ArgumentError("bandpass_solution: the solution's model carries no bandpass component.")
    )
    model = StationGainModel(phase = Tuple(pcs[pidx]), logamp = Tuple(lcs[lidx]))
    nant = sol.layout.nant
    layout = plan_parameters(model, nant, sol.geom)
    # The extracted model has its own layout, so θ shares neither length nor
    # parameter identity with `sol.θ` — only its element type carries over.
    θ = zeros(eltype(sol.θ), layout.nθ)
    # Component plans are laid out deterministically from (component, nant,
    # geom), so each extracted component's θ block is a straight range copy.
    ro = component_ranges(sol.layout)
    rn = component_ranges(layout)
    for (k, i) in enumerate(pidx)
        length(rn[k]) == length(ro[i]) ||
            error("bandpass_solution: internal block-size mismatch (phase component $i)")
        θ[rn[k]] = sol.θ[ro[i]]
    end
    for (k, j) in enumerate(lidx)
        length(rn[layout.nphase + k]) == length(ro[sol.layout.nphase + j]) ||
            error("bandpass_solution: internal block-size mismatch (logamp component $j)")
        θ[rn[layout.nphase + k]] = sol.θ[ro[sol.layout.nphase + j]]
    end
    names = hasproperty(sol.info, :ant_names) ?
        (; ant_names = sol.info.ant_names) : NamedTuple()
    info = (; nant = nant, nscan = 0, extracted = :bandpass, names...)
    return CalibrationSolution(model, layout, sol.geom, θ, info)
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
    leaf_window(geom, leaf) -> (chan_idx::Vector{Int}, ti_idx::Vector{Int})

Indices into `geom.channel_freqs` / `geom.times` of the channels and times the
leaf carries, matched by value (frequency by `isapprox` rtol 1e-9, time by atol
1e-9 h). Errors if any leaf sample has no match in the geometry.
"""
function leaf_window(geom::DataGeometry, leaf)
    fs = lookup(leaf[:vis], Frequency)
    ts = lookup(leaf[:vis], Ti)
    chan_idx = Vector{Int}(undef, length(fs))
    for (i, f) in enumerate(fs)
        j = findfirst(g -> isapprox(g, Float64(f); rtol = 1.0e-9), geom.channel_freqs)
        isnothing(j) &&
            throw(ArgumentError("leaf_window: channel frequency $f not found in geometry"))
        chan_idx[i] = j
    end
    ti_idx = Vector{Int}(undef, length(ts))
    for (i, t) in enumerate(ts)
        j = findfirst(g -> isapprox(g, Float64(t); atol = 1.0e-9), geom.times)
        isnothing(j) && throw(ArgumentError("leaf_window: time $t not found in geometry"))
        ti_idx[i] = j
    end
    return chan_idx, ti_idx
end

# ── Apply ────────────────────────────────────────────────────────────────────

"""
    apply_calibration(uvset::UVSet, sol::CalibrationSolution; apply_flags = true) -> UVSet

Divide every leaf's visibilities by the solution's per-antenna gains. For a
baseline `(a, b)` and correlation product `p` with feeds `(fa, fb)`:

    V_corr = V / (g_a[fa] · conj(g_b[fb])),    W_corr = W · |g_a · g_b|²

Samples where either gain magnitude underflows are flagged (weight 0, vis NaN).

`apply_flags` (default `true`) additionally zero-weights the solution's
recorded flags, when present in `sol.info` (the fringe solver records both):
(station, scan) pairs the solve left UNCONSTRAINED — identity gains, i.e. the
data would pass through uncalibrated — and baselines excluded for cause (the
intra-site crosstalk pairs). This is the EHT-HOPS flag semantic: a station is
flagged per scan only when, after the closure-screened global solve, no strong
detection constrains it; a merely weak baseline between two constrained
stations is NOT flagged (it is calibrated by SNR transfer).
"""
function UVData.apply_calibration(
        uvset::UVSet, sol::CalibrationSolution;
        apply_flags::Bool = true, ntasks::Integer = Threads.nthreads(),
    )
    ev = GainEvaluator(sol.model, sol.layout)
    flagged, exclbl = apply_flags ? _solution_flag_sets(sol.info) : (nothing, nothing)
    return UVData.apply(uvset) do leaf, info, root
        # Correction reads vis + weights; the output flag is re-derived from the
        # corrected weights downstream, so skip the redundant on-disk flag layer.
        leaf = materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
        ci, ti = leaf_window(sol.geom, leaf)
        g = evaluate_gains(ev, sol.θ, ci, ti)      # (nchan_leaf, nti_leaf, nant, 2)
        bl_pairs = UVData.baselines(leaf).pairs
        pols = pol_products(leaf)
        vis_corr, w_corr = _apply_gain_kernel(leaf[:vis], leaf[:weights], g, bl_pairs, pols; ntasks = ntasks)
        _flag_solution_rows!(parent(vis_corr), parent(w_corr), bl_pairs, sol.geom, ti, flagged, exclbl)
        return with_visibilities(leaf, vis_corr, w_corr)
    end
end

# The solution's recorded flags as lookup sets: `flagged` = (station, geometry
# scan id) pairs with no constraint (identity gains), `exclbl` = excluded
# baselines (both orders). `nothing` when the solution carries none.
function _solution_flag_sets(info::NamedTuple)
    flagged = if haskey(info, :flagged_ant) && !isempty(info.flagged_ant)
        Set{Tuple{Int, Int}}(
            (Int(info.flagged_ant[i]), Int(info.flagged_scan[i]))
                for i in eachindex(info.flagged_ant)
        )
    else
        nothing
    end
    exclbl = if haskey(info, :excluded_ant_a) && !isempty(info.excluded_ant_a)
        s = Set{Tuple{Int, Int}}()
        for i in eachindex(info.excluded_ant_a)
            a, b = Int(info.excluded_ant_a[i]), Int(info.excluded_ant_b[i])
            push!(s, (a, b))
            push!(s, (b, a))
        end
        s
    else
        nothing
    end
    return flagged, exclbl
end

# Zero-weight (and NaN) whole baseline rows per the solution flags: baselines
# touching a (station, scan) the solve left unconstrained, and the excluded
# (intra-site) baselines. A leaf spans ONE scan, so the scan id comes from its
# first time index.
function _flag_solution_rows!(Vc, Wc, bl_pairs, geom, ti_idx, flagged, exclbl)
    (flagged === nothing && exclbl === nothing) && return nothing
    isempty(ti_idx) && return nothing
    scanid = geom.scan_of_time[first(ti_idx)]
    @inbounds for bi in eachindex(bl_pairs)
        a, b = bl_pairs[bi]
        a == b && continue
        bad = (exclbl !== nothing && (a, b) in exclbl) || (
            flagged !== nothing && ((a, scanid) in flagged || (b, scanid) in flagged)
        )
        bad || continue
        Wc[:, :, bi, :] .= zero(eltype(Wc))
        Vc[:, :, bi, :] .= convert(eltype(Vc), NaN)
    end
    return nothing
end

# Divide visibilities by complex antenna gains `g[c, ti, ant, feed]`. The vis /
# weight DimArrays are (Frequency, Ti, Baseline, Pol). The per-(baseline, pol)
# output columns are independent (disjoint writes, read-only gains), so the loop
# fans out over `ntasks` tasks — this kernel is the dominant cost of applying a
# solution (≈80% of `apply_calibration`), and one thread wastes a 16-core box.
# Reordering across independent cells is bit-identical: every cell is computed by
# the same scalar expression regardless of task partition.
function _apply_gain_kernel(
        vis_p::AbstractArray, w_p::AbstractArray,
        g::AbstractArray{<:Complex, 4},
        bl_pairs, pols; ntasks::Integer = 1,
    )
    V = parent(vis_p)
    W = parent(w_p)
    Vc = copy(V)
    Wc = copy(W)
    nchan, nti, nbl, npol = size(V)
    geps = 1.0e-12
    # One (bi, p) column of work per unit; disjoint outputs → safe to spawn.
    cols = [(bi, p) for p in 1:npol for bi in 1:nbl]
    nt = clamp(Int(ntasks), 1, max(1, length(cols)))
    do_col = @inline function (bi, p)
        fa, fb = correlation_feed_pair(pols[p])
        a, b = bl_pairs[bi]
        return @inbounds for tt in 1:nti, c in 1:nchan
            w = W[c, tt, bi, p]
            ga = g[c, tt, a, fa]
            gb = g[c, tt, b, fb]
            denom = ga * conj(gb)
            if abs(ga) < geps || abs(gb) < geps || !isfinite(denom)
                Wc[c, tt, bi, p] = zero(eltype(Wc))
                Vc[c, tt, bi, p] = convert(eltype(Vc), NaN)
                continue
            end
            Vc[c, tt, bi, p] = V[c, tt, bi, p] / denom
            Wc[c, tt, bi, p] = w * abs2(ga * gb)
        end
    end
    if nt <= 1
        for (bi, p) in cols
            do_col(bi, p)
        end
    else
        exec_foreach(cols; ntasks = nt) do (bi, p)
            do_col(bi, p)
        end
    end
    return Vc, Wc
end

# ── Serialization ────────────────────────────────────────────────────────────

"""
    save_solution(path, sol::CalibrationSolution)

Serialize `sol` to `path` via the `Serialization` stdlib inside a versioned
wrapper NamedTuple (version 2 adds `stages` + `transforms`; version 3 adds
`postcal`). Transforms that close over caller code (e.g. a `CalFunction`)
serialize only within the same code state; a transform (or postcal step) that
fails to serialize is recorded as `missing` with a warning rather than failing
the save.
"""
function save_solution(path::AbstractString, sol::CalibrationSolution)
    wrapper = (;
        version = 3, sol.model, sol.layout, sol.geom, sol.θ, sol.info,
        sol.stages, transforms = _serializable_transforms(sol.transforms),
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

Inverse of [`save_solution`](@ref). Loads version 1 (pre-stage) files as
solutions with empty `stages`/`transforms`, and version ≤ 2 (pre-postcal) with
empty `postcal`.
"""
function load_solution(path::AbstractString)
    w = deserialize(path)
    w.version in (1, 2, 3) || error("load_solution: unsupported version $(w.version)")
    stages = w.version >= 2 ? w.stages : StageRecord[]
    transforms = w.version >= 2 ? w.transforms : Any[]
    postcal = w.version >= 3 ? w.postcal : Any[]
    return CalibrationSolution(w.model, w.layout, w.geom, w.θ, w.info; stages, transforms, postcal)
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
- `info/*` — the solver diagnostics (per-scan SNR/χ/ncomp, counts).
- root attributes — format/version, units, and the gain convention
  `V_corr = V / (g_a · conj(g_b))`, `weight ×= |g_a g_b|²`.
- `julia/blob` — the `Serialization` bytes of `(model, layout, geom, θ, info)` so
  Julia can round-trip the solution losslessly (external readers ignore it).
"""
function save_solution_hdf5 end

"""
    load_solution_hdf5(path) -> CalibrationSolution

Reconstruct a `CalibrationSolution` from an HDF5 file written by
[`save_solution_hdf5`](@ref) (via its `julia/blob`). Provided by `GustavoHDF5Ext`.
"""
function load_solution_hdf5 end
