# ── Fringe diagnostics ───────────────────────────────────────────────────────
#
# Diagnostics read off the detections of a fringe search and a fringe
# `CalibrationSolution`, returned as labeled `DimStack`s and `DimArray`s keyed
# by scan name, station name and feed, plus the per-baseline data helpers the
# Makie plot stubs consume. The plot entry points themselves are stubs in
# `Fring.jl`, implemented by `GustavoMakieExt`.

DimensionalData.@dim FreqGroup "Frequency group"
DimensionalData.@dim Triangle "Antenna triangle"

function _fringe_info(sol::CalibrationSolution)
    sol.name === :fringe || throw(ArgumentError("the solution is not a fringe solution; it holds $(repr(sol.name))"))
    return sol.info
end

_scan_dim(names) = Scan(DimensionalData.Lookups.Categorical(collect(String, names); order = DimensionalData.Lookups.Unordered()))

"""
    fringe_snr_table(dets) -> DimStack

Per-scan summary of the fringe search `dets` (the detection stacks of
`fit(FringeSearch(...), ps)`), over `Scan` (scan names):

- `max_snr` — the largest SNR the scan's search measured.
- `pfa` — its false-alarm probability [`fringe_pfa`](@ref): the chance that
  pure noise, searched over the scan's full delay × rate × baseline × product
  space, gives a peak of at least `max_snr`. `pfa ≪ 1` marks a secure
  detection, `pfa` near 1 a likely false fringe.
"""
function fringe_snr_table(dets::AbstractVector)
    max_snr = [maximum((d.snr[i] for i in eachindex(d.snr, d.valid) if d.valid[i]); init = 0.0) for d in dets]
    pfa = fringe_pfa.(max_snr, [DimensionalData.metadata(d).ncells for d in dets])
    return DimStack((; max_snr, pfa), (_scan_dim([DimensionalData.metadata(d).scan for d in dets]),))
end

"""
    fringe_detections(dets) -> DimStack

Every (baseline, product) cell the fringe search `dets` measured, over
`(Scan, AntennaPair, FeedPair)` — scan name, `(antenna, antenna)` and
`(feed, feed)`. Layers: `snr`, `pfa` (family-wise, [`fringe_pfa`](@ref)),
`delay` (s), `rate` (Hz), `phase` (rad) and `amp`. A cell a scan did not
measure holds `NaN`.

The detections a [`FringeClosure`](@ref Gustavo.FringeClosure) accepts are
those with `pfa ≤ pfa_max`; the accepted ones a stricter threshold would
reject are the marginal ones worth inspecting:

```julia
det = fringe_detections(dets)
findall((det.pfa .<= Stationization().pfa_max) .& (det.pfa .> 1e-6))
```
"""
function fringe_detections(dets::AbstractVector)
    antenna_pairs = unique(p for d in dets for p in lookup(d, AntennaPair))
    feeds = sort!(unique(f for d in dets for f in lookup(d, FeedPair)))
    dm = (_scan_dim([DimensionalData.metadata(d).scan for d in dets]), _antenna_pair_dim(antenna_pairs), FeedPair(feeds))
    n = map(length, dm)
    layers = (:snr, :pfa, :delay, :rate, :phase, :amp)
    stack = DimStack(NamedTuple{layers}(ntuple(_ -> fill(NaN, n), length(layers))), dm)
    for (k, d) in pairs(dets), p in lookup(d, AntennaPair), f in lookup(d, FeedPair)
        src = (AntennaPair(At(p)), FeedPair(At(f)))
        d.valid[src...] || continue
        cell = (Scan(k), AntennaPair(At(p)), FeedPair(At(f)))
        for layer in layers
            stack[layer][cell...] = d[layer][src...]
        end
    end
    return stack
end

"""
    fringe_antenna_solutions(sol::CalibrationSolution) -> DimStack

The fringe closure's per-scan station delay and rate, decoded
from its solved parameters (no data is read), over
`(Scan, AntennaName, Feed)`:

- `delay` — station group delay (s): the feed-common delay plus, on each feed
  the model gives one, the fitted inter-feed offset.
- `rate` — station fringe rate (Hz).

Each sums every delay and rate term of the closure's model; a term
the model lacks reads `NaN`. Values are fixed to the solve's gauge; a
within-scan difference against the same feed of a reference station is
gauge-invariant.

The parameters are dense: a (scan, station, feed) the solve never constrained
reads back as 0, indistinguishable from a solved zero. Mask it with
[`fringe_antenna_flags`](@ref).
"""
function fringe_antenna_solutions(sol::CalibrationSolution)
    _fringe_info(sol)
    groups = Calibration._applied(sol).groups
    length(groups) == 1 || error(
        "fringe_antenna_solutions: the fringe components differ across antennas; decode each " *
            "antenna set's components from their `params` instead"
    )
    (; model, layout, θ) = only(groups)
    nant = layout.nant
    comps = fringe_stage_components(model, layout)   # (plan, kind ∈ :delay/:rate)
    refplan = _perscan_delay_plan(model, layout)
    refplan === nothing &&
        error("fringe_antenna_solutions: model has no per-scan (feed-common) delay component")
    nscan = refplan.shape.Ti                                # PerScan ⇒ ntseg == #scan groups
    names = sol.geom.scan_names
    length(names) == nscan || error(
        "fringe_antenna_solutions: the solution's geometry names $(length(names)) scans for " *
            "$nscan scan segments"
    )
    # First time index landing in each scan segment — used to look up every plan's
    # own segment id for this scan (a `GlobalTime` inter-feed plan maps them all to 1, a
    # `PerScan` one to the scan itself, so the same lookup handles both bases).
    t0 = zeros(Int, nscan)
    for ti in eachindex(refplan.tseg_id)
        k = refplan.tseg_id[ti]
        (1 <= k <= nscan && t0[k] == 0) && (t0[k] = ti)
    end
    scans = findall(!=(0), t0)
    d = (_scan_dim(names[scans]), _antenna_dim(sol.geom.antennas), Feed(1:sol.geom.nfeed))
    n = map(length, d)
    delay, rate = fill(NaN, n), fill(NaN, n)
    for (s, k) in enumerate(scans), a in 1:nant, f in 1:sol.geom.nfeed
        ti = t0[k]
        for (plan, kind) in comps
            node = _feed_node(plan.tying, f)            # fseg 1: stage-B terms are GlobalFrequency
            node == 0 && continue
            v = _component_leaf(plan, θ)[1, node, 1, plan.tseg_id[ti], a]
            out = kind === :delay ? delay : rate
            out[s, a, f] = isnan(out[s, a, f]) ? v : out[s, a, f] + v
        end
    end
    return DimStack((; delay, rate), d)
end

"""
    fringe_antenna_flags(sol::CalibrationSolution) -> DimArray{Bool}

Over `(Scan, AntennaName, Feed)`: `true` where the fringe closure left the
station's feed unconstrained in the scan — no accepted detection (`pfa ≤
Stationization.pfa_max`) on any of its products with that feed, so nothing put
it in a fringe group. A measured but rejected product does not rescue it: such
a row constrains the fit without fixing a fringe location. `calibrate` flags
every product with a flagged (station, feed) on either side in that scan
(`apply_flags = true`).
"""
function fringe_antenna_flags(sol::CalibrationSolution)
    info = _fringe_info(sol)
    d = (_scan_dim(sol.geom.scan_names), _antenna_dim(sol.geom.antennas), Feed(1:sol.geom.nfeed))
    flags = fill(false, d)
    for i in eachindex(info.flagged_ant, info.flagged_feed, info.flagged_scan)
        flags[
            Scan(At(sol.geom.scan_names[info.flagged_scan[i]])),
            AntennaName(At(String(info.flagged_ant[i]))), Feed(At(info.flagged_feed[i])),
        ] = true
    end
    return flags
end

"""
    cat_scans(xs) -> DimStack or DimArray

Concatenate per-scan diagnostics — each a `DimStack` or `DimArray` with a
`Scan` dimension, such as [`fringe_snr_table`](@ref) of each solution
[`mapsets`](@ref Gustavo.mapsets) returns — along `Scan`. Every other dimension takes the
union of the inputs' labels, in first-seen order, so scans whose stations,
baselines or products differ line up by name; a cell an input lacks holds
`NaN`, or `false` in a `Bool` layer. The inputs share their dimensions'
types and order.

```julia
sols = mapsets(g -> fit(FringeClosure(; gauge), fit(FringeSearch(), g)), groupby(ps, ByScan()))
flags = cat_scans(fringe_antenna_flags.(sols))
```
"""
function cat_scans(xs)
    xs = collect(xs)
    isempty(xs) && throw(ArgumentError("cat_scans: nothing to concatenate"))
    ref = dims(first(xs))
    for x in xs
        map(DimensionalData.basetypeof, dims(x)) == map(DimensionalData.basetypeof, ref) || throw(
            DimensionMismatch("cat_scans: dimensions differ: $(map(DimensionalData.name, dims(x))) vs $(map(DimensionalData.name, ref))")
        )
    end
    scans = reduce(vcat, (collect(lookup(x, Scan)) for x in xs))
    allunique(scans) || throw(ArgumentError("cat_scans: scans repeat: $(join(unique(filter(s -> count(==(s), scans) > 1, scans)), ", "))"))
    out_dims = map(ref) do d
        D = DimensionalData.basetypeof(d)
        D === Scan && return _scan_dim(scans)
        all(x -> lookup(x, D) == lookup(d), xs) && return d
        labels = unique(reduce(vcat, (collect(lookup(x, D)) for x in xs)))
        return D(DimensionalData.Lookups.Categorical(labels; order = DimensionalData.Lookups.Unordered()))
    end
    return _cat_scans(first(xs), xs, out_dims)
end

_filler(::Type{Bool}) = false
_filler(::Type{T}) where {T <: AbstractFloat} = T(NaN)

function _cat_scans(::AbstractDimArray, xs, out_dims)
    out = fill(_filler(eltype(first(xs))), out_dims)
    for x in xs
        out[map(d -> DimensionalData.basetypeof(d)(At(collect(lookup(d)))), dims(x))] = parent(x)
    end
    return out
end

function _cat_scans(::AbstractDimStack, xs, out_dims)
    names = keys(first(xs))
    layers = map(names) do k
        _cat_scans(first(xs)[k], [x[k] for x in xs], dims(out_dims, dims(first(xs)[k])))
    end
    return DimStack(NamedTuple{names}(layers))
end

# ── Per-baseline spectra (the fringe-fit quality check) ──────────────────────

"""
    baseline_spectra(avg) -> DimStack

The time-averaged spectra of `avg`, a processing set XRadio's `average`
reduced to one sample per scan (`average(ps, ByScan())`), laid out by label
across its spectral windows: layers `vis` (the inverse-variance weighted mean
visibility) and `weight` (its summed weight, so `1/√weight` is its width per
real component), over `(Scan, AntennaPair, FeedPair, Frequency)`, cross
baselines only. A cell no Measurement Set holds, or holds flagged, is `NaN`
with weight zero. Throws when a Measurement Set holds more than one sample of
a scan.

Spectra of a scan before and after a solution show what it removed; a good
fringe fit flattens the phase slope. DimensionalData's Makie recipes plot them
directly, one line per baseline:

```julia
before = baseline_spectra(average(g, ByScan()))
after = baseline_spectra(average(calibrate(fr, g; flag_bad = false, apply_flags = false), ByScan()))
series(angle.(after.vis[Scan = 1, FeedPair = At((1, 1))]))
```

Per-scan results combine with [`cat_scans`](@ref).
"""
function baseline_spectra(avg::XRadio.ProcessingSet)
    isempty(avg) && throw(ArgumentError("the processing set holds no Measurement Sets"))
    members = collect(values(avg))
    geom = DataGeometry(avg)
    pairs = [p for p in _canonical_antenna_pairs(reduce(vcat, map(_member_antenna_pairs, members)), geom) if p[1] != p[2]]
    feeds = sort!(unique!(reduce(vcat, (vec(feed_pairs(ms)) for ms in members))))
    scans = unique(reduce(vcat, (String.(collect(ms[:scan_name])) for ms in members)))
    freqs = sort!(unique!(reduce(vcat, (collect(XRadio.frequencies(ms)) for ms in members))))
    d = (_scan_dim(scans), _antenna_pair_dim(pairs), FeedPair(feeds), Frequency(freqs))
    n = map(length, d)
    V = mapreduce(ms -> eltype(ms[:visibility]), promote_type, members)
    W = mapreduce(ms -> eltype(ms[:weight]), promote_type, members)
    vis = fill(V(NaN), n)
    weight = zeros(W, n)
    seen = falses(n)
    index = (
        Dict(s => i for (i, s) in enumerate(scans)), Dict(p => i for (i, p) in enumerate(pairs)),
        Dict(f => i for (i, f) in enumerate(feeds)), Dict(f => i for (i, f) in enumerate(freqs)),
    )
    for ms in members
        _place_spectra!(vis, weight, seen, index, _member_layers(ms)..., ms)
    end
    return DimStack((; vis = DimArray(vis, d), weight = DimArray(weight, d)))
end

function _place_spectra!(vis, weight, seen, (scan_i, pair_i, feed_i, freq_i), V, W, F, ms)
    antennas = _member_antenna_pairs(ms)
    feeds = feed_pairs(ms)
    scans = String.(collect(ms[:scan_name]))
    freqs = collect(lookup(V, Frequency))
    for ti in axes(V, Ti), bi in axes(V, BaselineID), p in axes(V, Polarization)
        haskey(pair_i, antennas[bi]) || continue                # autocorrelations
        s, j, q = scan_i[scans[ti]], pair_i[antennas[bi]], feed_i[feeds[p, bi]]
        for c in axes(V, Frequency)
            k = freq_i[freqs[c]]
            seen[s, j, q, k] && throw(
                ArgumentError(
                    "the set holds more than one sample of scan $(scans[ti]) on $(antennas[bi]); " *
                        "average over time first: `average(ps, ByScan())`"
                )
            )
            seen[s, j, q, k] = true
            cell = (Frequency(c), Ti(ti), BaselineID(bi), Polarization(p))
            _usable(F[cell], W[cell], V[cell]) || continue
            vis[s, j, q, k] = V[cell]
            weight[s, j, q, k] = W[cell]
        end
    end
    return nothing
end

"""
    freq_group_coherence(spectra) -> DimArray

The coherence `η = Σ |Σ_c z_c| / Σ Σ_c |z_c|` of each frequency group
([`fringe_freq_groups`](@ref)) of `spectra`, a [`baseline_spectra`](@ref)
result, pooled over cross baselines: the inner sums run
over the group's channels, the outer over baselines. Over
`(Scan, FeedPair, FreqGroup)`, each `FreqGroup` labeled by its lowest and
highest channel frequency. A group whose coherence lags its neighbours after a
fit localises residual frequency structure (RFI, a station passband defect)
to that group.
"""
function freq_group_coherence(spectra::AbstractDimStack)
    spec = spectra.vis
    freqs = collect(lookup(spec, Frequency))
    groups = _freq_group_ranges(freqs)
    d = (
        dims(spec, Scan), dims(spec, FeedPair),
        FreqGroup(DimensionalData.Lookups.Categorical([(freqs[first(r)], freqs[last(r)]) for r in groups]; order = DimensionalData.Lookups.Unordered())),
    )
    η = fill(NaN, d)
    for s in axes(spec, Scan), p in axes(spec, FeedPair), (k, r) in enumerate(groups)
        num = 0.0
        den = 0.0
        for bl in axes(spec, AntennaPair)
            z = filter(isfinite, view(spec, Scan(s), AntennaPair(bl), FeedPair(p), Frequency(r)))
            num += abs(sum(z; init = zero(eltype(spec))))
            den += sum(abs, z; init = 0.0)
        end
        den > 0 && (η[Scan(s), FeedPair(p), FreqGroup(k)] = num / den)
    end
    return η
end

# Coherence-weighted group delay (s) of one baseline's spectrum `z` over `freqs`
# from the per-channel phase increment: τ = ⟨angle(z[c+1] z[c]*)⟩ / (2π Δf). Uses
# the wrapped increment (no unwrap needed) and weights by |z|; skips flagged cells
# and the sub-band-boundary jumps (Δf ≫ in-band spacing) where the increment wraps.
function _baseline_delay(z::AbstractVector, freqs::AbstractVector)
    # `z` and `freqs` are read at the same index, so they must share one axis.
    eachindex(z, freqs)
    df = Float64[]
    for c in firstindex(freqs):(lastindex(freqs) - 1)
        d = freqs[c + 1] - freqs[c]
        d > 0 && push!(df, d)
    end
    isempty(df) && return NaN
    dfmed = median(df)
    num = 0.0; den = 0.0
    for c in firstindex(z):(lastindex(z) - 1)
        z1 = z[c]; z2 = z[c + 1]
        (isfinite(z1) && isfinite(z2) && abs(z1) > 0 && abs(z2) > 0) || continue
        d = freqs[c + 1] - freqs[c]
        (d > 0 && d <= 3 * dfmed) || continue         # skip sub-band boundary jumps
        w = abs(z1) * abs(z2)
        num += w * angle(z2 * conj(z1))
        den += w * 2π * d
    end
    return den > 0 ? num / den : NaN
end

"""
    baseline_delays(spectra) -> DimArray

Each cross baseline's group delay (s) in `spectra`, a
[`baseline_spectra`](@ref) result, over `(Scan, AntennaPair, FeedPair)`: the
coherence-weighted mean phase increment between adjacent channels over 2π
times their spacing. On data a correct delay solution has corrected, every
baseline's delay is ≈ 0.
"""
function baseline_delays(spectra::AbstractDimStack)
    spec = spectra.vis
    freqs = collect(lookup(spec, Frequency))
    d = (dims(spec, Scan), dims(spec, AntennaPair), dims(spec, FeedPair))
    delay = fill(NaN, d)
    for I in DimensionalData.DimIndices(delay)
        delay[I] = _baseline_delay(collect(spec[I]), freqs)
    end
    return delay
end

"""
    delay_closure(spectra) -> DimArray

The triangle delay closure `τ_ab + τ_bc + τ_ca` of the
[`baseline_delays`](@ref) of `spectra`, a [`baseline_spectra`](@ref) result,
around every triangle of stations whose three baselines were measured. Over
`(Scan, Triangle, FeedPair)`, each `Triangle` labeled by its station names,
for the products that pair a feed with itself: on a product `(f, g)` with
`f ≠ g` each station enters the triangle once with each feed, so the feeds'
delay difference does not cancel.

Station-based delays cancel around a triangle, so the closure of raw data is
≈ 0 up to noise, and a correct station-based solution keeps it there. A
station-structure or sign mistake shows as residual
[`baseline_delays`](@ref) after correction and, if it breaks closure, as a
nonzero closure.
"""
function delay_closure(spectra::AbstractDimStack)
    delay = baseline_delays(spectra)
    pairs = collect(lookup(delay, AntennaPair))
    antennas = unique(Iterators.flatten(pairs))
    index = Dict(p => i for (i, p) in enumerate(pairs))
    triangles = [
        (a, b, c) for (i, a) in enumerate(antennas) for (j, b) in enumerate(antennas) for (k, c) in enumerate(antennas)
            if i < j < k && all(e -> haskey(index, e) || haskey(index, reverse(e)), ((a, b), (b, c), (c, a)))
    ]
    parallel = [f for f in lookup(delay, FeedPair) if f[1] == f[2]]
    d = (
        dims(delay, Scan),
        Triangle(DimensionalData.Lookups.Categorical(triangles; order = DimensionalData.Lookups.Unordered())),
        FeedPair(parallel),
    )
    closure = fill(NaN, d)
    # A baseline stored as (y, x) has the negated delay of (x, y).
    τ(s, e, f) = haskey(index, e) ? delay[Scan(s), AntennaPair(index[e]), FeedPair(At(f))] :
        -delay[Scan(s), AntennaPair(index[reverse(e)]), FeedPair(At(f))]
    for s in axes(closure, Scan), (t, (a, b, c)) in enumerate(triangles), f in parallel
        closure[Scan(s), Triangle(t), FeedPair(At(f))] = τ(s, (a, b), f) + τ(s, (b, c), f) + τ(s, (c, a), f)
    end
    return closure
end

# ── Delay–rate search map (the false-fringe check) ─────────────────────────────

"""
    fringe_search_map(det, group; baseline = nothing, feeds = nothing) -> FringeSearchMap

The delay–rate search surface of one cross baseline of `group`, the scan
(`groupby(ps, ByScan())`) whose search gave `det`, one detection stack of
`fit(FringeSearch(...), ps)`: the search `det` records, with the same
[`FringeSearch`](@ref) options, keeping the whole windowed matched-filter plane
in SNR units instead of only its peak. A real fringe is a single sharp peak far
above the sidelobes (`pfa ≪ 1`); a false fringe barely clears them.

- `baseline` — a station-name pair such as `("A1", "A3")`, in either order.
- `feeds` — a feed-index pair such as `(1, 1)`.

Either one left out is the scan's strongest cell in `det` that matches the
other. The map's `pfa` is over the scan's search family, as `det` records it,
so it compares with `det.pfa` and `Stationization.pfa_max`. The detection
phase is referenced to the scan's mean time.

The map reproduces the recorded detection only for the data that was searched:
pass the group with the same flags, weights and corrections.
"""
function fringe_search_map(
        det::AbstractDimStack, group::XRadio.ProcessingSet; baseline = nothing, feeds = nothing,
    )
    meta = DimensionalData.metadata(det)
    scan = _group_scan(group)
    scan == meta.scan || throw(ArgumentError("the detections are of scan $(meta.scan), the group of scan $scan"))
    baseline, feeds = _strongest_cell(det, baseline, feeds)
    gc = _GroupCells(group, meta.geom)
    j = findfirst(p -> p == baseline || p == reverse(baseline), gc.antenna_pairs)
    isnothing(j) && throw(
        ArgumentError("$(baseline) is not a cross baseline of scan $scan; it holds $(join(gc.antenna_pairs, ", "))")
    )
    q = findfirst(==(feeds), gc.feeds)
    isnothing(q) && throw(ArgumentError("scan $scan holds no feed pair $(feeds); it holds $(join(gc.feeds, ", "))"))
    ws = FringeWorkspace(eltype(first(first(gc.layers))))
    X, Wm = _gather_cell!(ws, gc, j, q)
    m = _fringe_map(X, Wm, gc.freqs, gc.times, meta.geom.f0, meta.epoch, meta.search, ws, meta.ncells)
    refdims = (
        _scan_dim([scan]), _antenna_pair_dim([gc.antenna_pairs[j]]),
        FeedPair(DimensionalData.Lookups.Categorical([gc.feeds[q]]; order = DimensionalData.Lookups.Unordered())),
    )
    return FringeSearchMap(DimensionalData.rebuild(m.snr; refdims), m.detection, m.ncells, m.pfa)
end

function _group_scan(group::XRadio.ProcessingSet)
    names = unique(String(n) for ms in values(group) for n in ms[:scan_name])
    length(names) == 1 || throw(
        ArgumentError("the group must hold one scan, as `groupby(ps, ByScan())` yields; it holds $(join(names, ", "))")
    )
    return only(names)
end

function _strongest_cell(det::AbstractDimStack, baseline, feeds)
    isnothing(baseline) || isnothing(feeds) || return baseline, feeds
    cells = [
        (p, f) for p in lookup(det, AntennaPair), f in lookup(det, FeedPair)
            if (isnothing(baseline) || p == baseline || p == reverse(baseline)) &&
            (isnothing(feeds) || f == feeds) && det.valid[AntennaPair(At(p)), FeedPair(At(f))]
    ]
    isempty(cells) && throw(
        ArgumentError("the search measured no cell of scan $(DimensionalData.metadata(det).scan) with baseline = $(repr(baseline)), feeds = $(repr(feeds))")
    )
    return argmax(((p, f),) -> det.snr[AntennaPair(At(p)), FeedPair(At(f))], cells)
end
