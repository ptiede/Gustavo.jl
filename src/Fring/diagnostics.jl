# ── Fringe diagnostics ───────────────────────────────────────────────────────
#
# Diagnostics read off a fringe `CalibrationSolution` — its solved parameters
# and the fringe step's `info` — returned as labeled `DimStack`s and
# `DimArray`s keyed by scan name, station name and feed, plus the per-baseline
# data helpers the Makie plot stubs consume. The plot entry points themselves
# are stubs in `Fring.jl`, implemented by `GustavoMakieExt`.

function _fringe_info(sol::CalibrationSolution)
    haskey(sol.steps, :fringe) || throw(
        ArgumentError(
            "the solution has no :fringe step; it holds $(join(repr.(collect(keys(sol.steps))), ", "))"
        )
    )
    return sol.steps[:fringe]
end

_scan_dim(names) = Scan(DimensionalData.Lookups.Categorical(collect(String, names); order = DimensionalData.Lookups.Unordered()))

"""
    fringe_snr_table(sol::CalibrationSolution) -> DimStack

Per-scan fringe-search summary of the fringe step, over `Scan` (scan names):

- `max_snr` — the largest detection SNR in the scan.
- `pfa` — the scan's false-alarm probability [`fringe_pfa`](@ref): the chance
  that pure noise, searched over the scan's full delay × rate × baseline ×
  product space, gives a peak of at least `max_snr`. `pfa ≪ 1` marks a secure
  detection, `pfa` near 1 a likely false fringe.

Throws when `sol` has no `:fringe` step. Per-scan results of
[`mapsets`](@ref) combine with [`cat_scans`](@ref).
"""
function fringe_snr_table(sol::CalibrationSolution)
    info = _fringe_info(sol)
    pfa = fringe_pfa.(info.scan_snr, info.scan_ncells)
    return DimStack((; max_snr = info.scan_snr, pfa), (_scan_dim(info.scan_names),))
end

"""
    fringe_detections(sol::CalibrationSolution) -> DimStack

Every (baseline, product) cell the fringe search measured, over
`(Scan, AntennaPair, FeedPair)` — scan name, `(station, station)` and
`(feed, feed)`. Layers: `snr`, `pfa` (family-wise, [`fringe_pfa`](@ref)),
`delay` (s), `rate` (Hz), `phase` (rad), and `detected`, whether the search
accepted the cell as a fringe (`pfa ≤ Stationization.pfa_max`). A cell a scan
did not measure holds `NaN` and `detected = false`.

The accepted detections that a stricter threshold would reject are the
marginal ones worth inspecting:

```julia
det = fringe_detections(sol)
findall(det.detected .& (det.pfa .> 1e-6))
```
"""
function fringe_detections(sol::CalibrationSolution)
    info = _fringe_info(sol)
    pairs = unique(tuple.(String.(info.det_ant_a), String.(info.det_ant_b)))
    feeds = sort!(unique(tuple.(Int.(info.det_feed_a), Int.(info.det_feed_b))))
    d = (_scan_dim(info.scan_names), _station_pair_dim(pairs), FeedPair(feeds))
    n = map(length, d)
    stack = DimStack(
        (;
            snr = fill(NaN, n), pfa = fill(NaN, n), delay = fill(NaN, n),
            rate = fill(NaN, n), phase = fill(NaN, n), detected = fill(false, n),
        ), d,
    )
    pair_index = Dict(p => i for (i, p) in enumerate(pairs))
    feed_index = Dict(f => i for (i, f) in enumerate(feeds))
    for i in eachindex(info.det_scan)
        cell = (
            Scan(info.det_scan[i]),
            AntennaPair(pair_index[(String(info.det_ant_a[i]), String(info.det_ant_b[i]))]),
            FeedPair(feed_index[(Int(info.det_feed_a[i]), Int(info.det_feed_b[i]))]),
        )
        stack.snr[cell] = info.det_snr[i]
        stack.pfa[cell] = info.det_pfa[i]
        stack.delay[cell] = info.det_delay[i]
        stack.rate[cell] = info.det_rate[i]
        stack.phase[cell] = info.det_phase[i]
        stack.detected[cell] = info.det_detected[i]
    end
    return stack
end

"""
    fringe_station_solutions(sol::CalibrationSolution) -> DimStack

The fringe step's per-scan station delay, rate and constant phase, decoded
from its solved parameters (no data is read), over
`(Scan, AntennaName, Feed)`:

- `delay` — station group delay (s): the feed-common delay plus, on feed 2,
  the fitted inter-feed offset.
- `rate` — station fringe rate (Hz).
- `phase` — station constant phase (rad), at the epoch the scan's rate is
  referenced to (the scan's own mean time), so it compares across scans only
  through a difference taken within one scan.

Each sums every delay, rate and constant term the fringe step owns; a term
the model lacks reads `NaN`. Values are fixed to the solve's gauge; a
within-scan difference against the same feed of a reference station is
gauge-invariant.

The parameters are dense: a (scan, station, feed) the solve never constrained
reads back as 0, indistinguishable from a solved zero. Mask it with
[`fringe_station_flags`](@ref).
"""
function fringe_station_solutions(sol::CalibrationSolution)
    groups = Calibration._applied(sol[:fringe]).groups
    length(groups) == 1 || error(
        "fringe_station_solutions: the fringe components differ across stations; decode each " *
            "station set's components from their `params` instead"
    )
    (; model, layout, θ) = only(groups)
    nant = layout.nant
    comps = fringe_stage_components(model, layout)   # (plan, kind ∈ :delay/:rate/:phase)
    refplan = _perscan_delay_plan(model, layout)
    refplan === nothing &&
        error("fringe_station_solutions: model has no per-scan (feed-common) delay component")
    nscan = refplan.shape[4]                                # PerScan ⇒ ntseg == #scan groups
    names = sol.geom.scan_names
    length(names) == nscan || error(
        "fringe_station_solutions: the solution's geometry names $(length(names)) scans for " *
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
    d = (_scan_dim(names[scans]), _station_dim(sol.geom.stations), Feed(1:2))
    n = map(length, d)
    delay, rate, phase = fill(NaN, n), fill(NaN, n), fill(NaN, n)
    for (s, k) in enumerate(scans), a in 1:nant, f in 1:2
        ti = t0[k]
        for (plan, kind) in comps
            node = _feed_node(plan.tying, f)            # fseg 1: stage-B terms are GlobalFrequency
            node == 0 && continue
            v = _component_leaf(plan, θ)[1, node, 1, plan.tseg_id[ti], a]
            out = kind === :delay ? delay : kind === :rate ? rate : phase
            out[s, a, f] = isnan(out[s, a, f]) ? v : out[s, a, f] + v
        end
    end
    return DimStack((; delay, rate, phase), d)
end

"""
    fringe_station_flags(sol::CalibrationSolution) -> DimArray{Bool}

Over `(Scan, AntennaName)`: `true` where the fringe step left the station
unconstrained in the scan — no accepted detection (`pfa ≤
Stationization.pfa_max`) on any of its baselines, so nothing put it in a
fringe group. A measured but rejected baseline does not rescue it: such a row
constrains the fit without fixing a fringe location. These stations carry
identity gains for those scans, and `calibrate` flags their baselines there
(`apply_flags = true`).
"""
function fringe_station_flags(sol::CalibrationSolution)
    info = _fringe_info(sol)
    d = (_scan_dim(info.scan_names), _station_dim(sol.geom.stations))
    flags = DimArray(fill(false, map(length, d)), d)
    for i in eachindex(info.flagged_ant, info.flagged_scan)
        flags[Scan(At(sol.geom.scan_names[info.flagged_scan[i]])), AntennaName(At(String(info.flagged_ant[i])))] = true
    end
    return flags
end

"""
    cat_scans(xs) -> DimStack or DimArray

Concatenate per-scan diagnostics — each a `DimStack` or `DimArray` with a
`Scan` dimension, such as [`fringe_snr_table`](@ref) of each solution
[`mapsets`](@ref) returns — along `Scan`. Every other dimension takes the
union of the inputs' labels, in first-seen order, so scans whose stations,
baselines or products differ line up by name; a cell an input lacks holds
`NaN`, or `false` in a `Bool` layer. The inputs share their dimensions'
types and order.

```julia
sols = mapsets(g -> fit(BaselineFringeFit(; gauge), g), groupby(ps, ByScan()))
snr = cat_scans(fringe_snr_table.(values(sols)))
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
    out = DimArray(fill(_filler(eltype(first(xs))), map(length, out_dims)), out_dims)
    for x in xs
        out[map(d -> DimensionalData.basetypeof(d)(At(collect(lookup(d)))), dims(x))] = parent(x)
    end
    return out
end

function _cat_scans(::AbstractDimStack, xs, out_dims)
    names = keys(first(xs))
    layers = map(names) do k
        _cat_scans(first(xs)[k], [x[k] for x in xs], out_dims)
    end
    return DimStack(NamedTuple{names}(layers))
end

# ── Per-baseline before/after data (the fringe-fit quality check) ──────────────

"""
    BaselineFringeData

Per-baseline coherent visibility averages for one scan, before and after applying
a fringe `CalibrationSolution`, consumed by `plot_baseline_fringes`.

Fields: `source`/`scan`/`scan_index`/`max_snr` identify the scan; `bl_pairs` and
`feeds` (each product's feed pair, see [`feed_pairs`](@ref)) label the baseline
and correlation axes; `freqs` (Hz, all spws
stacked) and `times` (h) the data axes. The four data arrays are weighted coherent
means (vector averages, `NaN` where a cell has no unflagged data):

- `spec_before`/`spec_after` — `(nchan, nbl, npol)`, averaged over time. `angle`
  vs frequency shows the group-delay slope (flat after a good fit); `abs` shows
  the group-averaged coherence.
- `tser_before`/`tser_after` — `(ntime, nbl, npol)`, averaged over frequency.
  `angle` vs time shows the fringe-rate slope (flat after a good fit).

Wide-band multi-group data (e.g. the four VGOS 3/5/6/10 GHz frequency groups) also
carries the per-frequency-group split, so diagnostics can be viewed one frequency
group at a time (`plot_baseline_fringes(data; freqgroup = k)`):

- `freq_groups` — channel ranges of each frequency group ([`fringe_freq_groups`](@ref)
  of `freqs`; a single full range on contiguous data).
- `tser_freqgroup_before`/`tser_freqgroup_after` — `(ntime, nbl, npol, ngroups)`, the time
  series averaged over only that frequency group's channels.
"""
struct BaselineFringeData
    source::String
    scan::String
    scan_index::Int
    max_snr::Float64
    bl_pairs::Vector{Tuple{Int, Int}}
    ant_names::Vector{String}        # station codes, indexed by antenna number
    feeds::Vector{Tuple{Int, Int}}
    freqs::Vector{Float64}
    times::Vector{Float64}
    spec_before::Array{ComplexF64, 3}
    spec_after::Array{ComplexF64, 3}
    tser_before::Array{ComplexF64, 3}
    tser_after::Array{ComplexF64, 3}
    freq_groups::Vector{UnitRange{Int}}
    tser_freqgroup_before::Array{ComplexF64, 4}
    tser_freqgroup_after::Array{ComplexF64, 4}
    # Thermal 1σ on each coherent mean above, in visibility units — the radial
    # width of the complex sample, from which a phase error is σ/|V|. `NaN`
    # where the mean has no contributing data, or where the caller did not
    # supply weights.
    spec_sigma_before::Array{Float64, 3}
    spec_sigma_after::Array{Float64, 3}
    tser_sigma_before::Array{Float64, 3}
    tser_sigma_after::Array{Float64, 3}
    tser_freqgroup_sigma_before::Array{Float64, 4}
    tser_freqgroup_sigma_after::Array{Float64, 4}
end

# 1σ on a weighted coherent mean: with `w = 1/σ_vis²` per sample, the mean
# `Σwv / Σw` has variance `1/Σw`. `_coherent_mean!` leaves its weight-sum
# argument untouched, so this reads the same accumulator the mean divided by.
_mean_sigma(wsum::AbstractArray) = map(x -> x > 0 ? 1 / sqrt(x) : NaN, wsum)

# The 1σ error bar on a plotted view of a complex sample, given the sample `z`
# and its radial width `σ`: `abs` sees σ itself, `angle` sees the angle σ
# subtends at radius |z|. Once σ reaches |z| the phase is unconstrained, so the bar saturates at
# π rather than reporting a misleadingly finite width.
_plotted_sigma(::typeof(abs), z, σ) = σ
_plotted_sigma(::typeof(angle), z, σ) = abs(z) > 0 ? min(σ / abs(z), Float64(π)) : Float64(π)

# Backwards-compatible constructor (no frequency-group split): one group spanning
# all channels, per-group time series = the full-span ones.
function BaselineFringeData(
        source, scan, scan_index, max_snr, bl_pairs, ant_names, feeds,
        freqs, times, spec_before, spec_after, tser_before, tser_after,
    )
    nti, nbl, npol = size(tser_before)
    return BaselineFringeData(
        source, scan, scan_index, max_snr, bl_pairs, ant_names, feeds,
        freqs, times, spec_before, spec_after, tser_before, tser_after,
        [1:length(freqs)],
        reshape(copy(tser_before), nti, nbl, npol, 1),
        reshape(copy(tser_after), nti, nbl, npol, 1),
        # No weights were supplied, so the thermal widths are unknown; plots
        # draw no error bars for a NaN σ.
        fill(NaN, size(spec_before)), fill(NaN, size(spec_after)),
        fill(NaN, size(tser_before)), fill(NaN, size(tser_after)),
        fill(NaN, nti, nbl, npol, 1), fill(NaN, nti, nbl, npol, 1),
    )
end

"""
    baseline_pol_index(data, pol) -> Int

Resolve a correlation-product selector, an `Integer` index or a feed pair such as
`(1, 1)`, against `data.feeds`.
"""
baseline_pol_index(data::BaselineFringeData, pol) = _pol_index(data.feeds, pol)

"""
    fringe_freq_group_stats(data::BaselineFringeData; pol)
        -> Vector{@NamedTuple{f_lo, f_hi, nchan, eta_before, eta_after}}

Per-frequency-group coherence summary of one scan's [`BaselineFringeData`](@ref):
for each contiguous frequency group, the within-group coherence `|Σ_c z_c| / Σ_c |z_c|`
of the per-channel time-averaged visibilities, pooled over cross baselines —
before and after the fringe solution. A frequency group whose `eta_after` lags its
neighbours localises residual frequency structure (RFI, station passband
defect) to that group.
"""
function fringe_freq_group_stats(data::BaselineFringeData; pol)
    p = _pol_index(data.feeds, pol)
    out = @NamedTuple{f_lo::Float64, f_hi::Float64, nchan::Int, eta_before::Float64, eta_after::Float64}[]
    for r in _freq_group_ranges(data.freqs)
        stats = map((data.spec_before, data.spec_after)) do spec
            num = 0.0
            den = 0.0
            for (bi, (a, b)) in enumerate(data.bl_pairs)
                a == b && continue
                acc = zero(ComplexF64)
                s = 0.0
                for c in r
                    z = spec[c, bi, p]
                    (isfinite(real(z)) && isfinite(imag(z))) || continue
                    acc += z
                    s += abs(z)
                end
                num += abs(acc)
                den += s
            end
            den > 0 ? num / den : NaN
        end
        push!(
            out, (
                f_lo = data.freqs[first(r)], f_hi = data.freqs[last(r)],
                nchan = length(r), eta_before = stats[1], eta_after = stats[2],
            )
        )
    end
    return out
end

# Selector resolution shared by `baseline_pol_index` and `fringe_search_map`.
_pol_index(feeds, pol::Integer) = Int(pol)
_pol_index(feeds, pol::Tuple{Integer, Integer}) = UVData._pol_index_lookup(feeds, pol)
_pol_index(feeds, pol) = throw(
    ArgumentError("select a correlation product by index or by feed pair such as (1, 1), got $(repr(pol))")
)

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
    delay_closure(data::BaselineFringeData; pol) -> NamedTuple

Triangle delay-closure check, the consistency test a station-based delay solution
must pass. For every closed triangle `(a,b,c)` it forms `τ_ab + τ_bc − τ_ac` from
the per-baseline group delays fitted to the coherent spectra:

- `closure_before` — from the raw data. A property of the *data*: real station-
  based delays cancel around a triangle, so these are ≈ 0 (up to noise). Large
  values would mean the data itself is non-closing (not something a fit can fix).
- `closure_after` — from the corrected data. Must stay ≈ 0 (a correct station-based
  solution cannot create closure errors).
- `resid_delay` — the per-baseline residual group delay after correction; a correct
  delay solution drives these to ≈ 0 on every baseline.

Returns `(; pol, triangles, closure_before, closure_after, resid_delay, bl_pairs)`.
A station-structure or sign mistake shows up as nonzero `resid_delay` (and, if it
breaks closure, nonzero `closure_after`).
"""
function delay_closure(data::BaselineFringeData; pol)
    p = baseline_pol_index(data, pol)
    nbl = length(data.bl_pairs)
    τb = fill(NaN, nbl); τa = fill(NaN, nbl)
    for bi in eachindex(τb, τa)
        a, b = data.bl_pairs[bi]
        a == b && continue
        τb[bi] = _baseline_delay(view(data.spec_before, :, bi, p), data.freqs)
        τa[bi] = _baseline_delay(view(data.spec_after, :, bi, p), data.freqs)
    end
    blindex = Dict(data.bl_pairs[bi] => bi for bi in eachindex(data.bl_pairs))
    ants = sort(unique(Iterators.flatten(data.bl_pairs)))
    tris = NTuple{3, Int}[]; cb = Float64[]; ca = Float64[]
    for a in ants, b in ants, c in ants
        (a < b < c) || continue
        (haskey(blindex, (a, b)) && haskey(blindex, (b, c)) && haskey(blindex, (a, c))) || continue
        ab, bc, ac = blindex[(a, b)], blindex[(b, c)], blindex[(a, c)]
        (isfinite(τb[ab]) && isfinite(τb[bc]) && isfinite(τb[ac])) || continue
        push!(tris, (a, b, c))
        push!(cb, τb[ab] + τb[bc] - τb[ac])
        push!(ca, τa[ab] + τa[bc] - τa[ac])
    end
    return (;
        pol = data.feeds[p], triangles = tris, closure_before = cb,
        closure_after = ca, data_delay = τb, resid_delay = τa, bl_pairs = copy(data.bl_pairs),
    )
end

# ── Delay–rate search map (the false-fringe check) ─────────────────────────────

"""
    BaselineFringeMap

The delay–rate search map of one (baseline, correlation product) of one scan,
with its scan/baseline labels, consumed by `plot_fringe_search`. `source`/`scan`/`scan_index` identify the scan; `bl_pair`
(antenna indices into `ant_names`) and `pol` (the product's feed pair) the searched block; `map` is the
[`FringeSearchMap`](@ref) (axes, SNR surface, refined detection, `ncells`,
`pfa`).
"""
struct BaselineFringeMap
    source::String
    scan::String
    scan_index::Int
    bl_pair::Tuple{Int, Int}
    ant_names::Vector{String}
    pol::Tuple{Int, Int}
    map::FringeSearchMap
end

"""
    print_delay_closure(c; io = stdout)

Summarize [`delay_closure`](@ref): RMS/max triangle closure (data and residual)
and the RMS/max residual per-baseline delay, all in ns.
"""
function print_delay_closure(c; io = stdout)
    rms(v) = (u = filter(isfinite, v); isempty(u) ? NaN : sqrt(sum(abs2, u) / length(u)))
    mx(v) = (u = abs.(filter(isfinite, v)); isempty(u) ? NaN : maximum(u))
    ns(x) = 1.0e9 * x
    println(io, "Delay closure [", c.pol, "], ", length(c.triangles), " triangles:")
    println(io, @sprintf("  data     closure rms = %9.4f ns  max = %9.4f ns", ns(rms(c.closure_before)), ns(mx(c.closure_before))))
    println(io, @sprintf("  residual closure rms = %9.4f ns  max = %9.4f ns", ns(rms(c.closure_after)), ns(mx(c.closure_after))))
    println(io, @sprintf("  residual delay   rms = %9.4f ns  max = %9.4f ns", ns(rms(c.resid_delay)), ns(mx(c.resid_delay))))
    return nothing
end
