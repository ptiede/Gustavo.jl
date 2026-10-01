# ── Fringe diagnostics ───────────────────────────────────────────────────────
#
# Report-style (non-plotting) diagnostics for a fringe `CalibrationSolution`,
# plus the pure gain-extraction helpers the Makie plot stubs consume. Everything
# here works off the solved model/θ/geometry and the solver's `info` NamedTuple
# (per-scan max SNR / component count), so it is Makie-free and unit-testable
# without loading a plotting backend. The plot entry points themselves
# (`plot_fringe_spectrum`, `plot_fringe_phases`, `plot_fringe_snr`) are stubs in
# `Fring.jl`, implemented by `GustavoMakieExt`.

# The `:fringe` step's diagnostics, or an empty NamedTuple when the solution
# has no fringe stage — every diagnostic below then returns an empty result.
_fringe_info(sol::CalibrationSolution) = get(sol.steps, :fringe, (;))

"""
    fringe_snr_table(sol::CalibrationSolution) -> Vector{NamedTuple}

Per-scan fringe-fit summary rows `(scan, max_snr, ncomp, pfa)`, all pulled
from the fringe step's own diagnostics (`sol.steps[:fringe]`). `ncomp` is a
solve-wide scalar (the same value on every row, not literally per-scan). `pfa` is the scan's false-alarm probability [`fringe_pfa`](@ref):
the chance that pure noise, searched over the scan's full
delay×rate×baseline×product space, would produce a peak of at least `max_snr`
— `pfa ≪ 1` marks a secure detection, `pfa` near 1 a likely FALSE fringe (`NaN`
when the solution predates `scan_ncells`). Returns an empty vector if the
solution carries no `:fringe` stage, or that stage no per-scan diagnostics.
"""
function fringe_snr_table(sol::CalibrationSolution)
    info = _fringe_info(sol)
    (haskey(info, :scan_snr) && haskey(info, :ncomp)) || return NamedTuple[]
    snr = info.scan_snr
    ncomp = Int(info.ncomp)
    ncells = get(info, :scan_ncells, Float64[])
    return [
        (;
            scan = s, max_snr = Float64(snr[s]), ncomp = ncomp,
            pfa = s <= length(ncells) ? fringe_pfa(snr[s], ncells[s]) : NaN,
        )
            for s in eachindex(snr)
    ]
end

"""
    print_fringe_snr_table(rows; io = stdout)

Pretty-print the rows from [`fringe_snr_table`](@ref).
"""
function print_fringe_snr_table(rows; io = stdout)
    isempty(rows) && return println(io, "No per-scan fringe diagnostics available")
    println(io)
    println(io, "Fringe per-scan summary")
    println(io, "scan   max_snr   ncomp        pfa")
    for r in rows
        println(
            io,
            lpad(string(r.scan), 4), "   ",
            lpad(_fmt(r.max_snr), 7), "   ",
            lpad(string(r.ncomp), 5), "   ",
            lpad(_fmt_pfa(get(r, :pfa, NaN)), 8),
        )
    end
    return nothing
end

_fmt(x::Real) = isfinite(x) ? string(round(x; digits = 3)) : "NaN"

# PFA formatting: probabilities span many decades, so switch to scientific
# notation below 10⁻³ instead of rounding to "0.0".
_fmt_pfa(x::Real) = !isfinite(x) ? "NaN" :
    (x > 0 && x < 1.0e-3 ? @sprintf("%.1e", x) : string(round(x; digits = 3)))

"""
    fringe_solution_summary(sol::CalibrationSolution) -> String

One-line summary of a fringe solution: antenna/scan/parameter counts and the
median per-scan max SNR.
"""
function fringe_solution_summary(sol::CalibrationSolution)
    rows = fringe_snr_table(sol)
    nant = length(sol.geom.stations)
    nscan = get(sol.info, :nscan, length(rows))
    snrs = [r.max_snr for r in rows if isfinite(r.max_snr)]
    medsnr = isempty(snrs) ? NaN : median(snrs)
    nθ = sum(c -> length(c.params), sol.components; init = 0)
    return string(
        "FringeSolution: ", nant, " antennas, ", nscan, " scans, ",
        nθ, " parameters; median scan max-SNR = ", _fmt(medsnr),
    )
end

"""
    fringe_station_solutions(sol::CalibrationSolution) -> Vector{NamedTuple}

Decode the stationized per-scan delay/rate/constant-phase parameters from
the fringe step's θ into a per-`(scan, station, feed)` table; no data is
read. One row per (scan-group index `scan`, 1-based `station`,
`feed ∈ {1, 2}`):

- `delay_ns`  — station group delay (ns): the feed-common delay plus, on
  feed 2, the fitted inter-feed offset.
- `rate_mHz`  — station fringe rate (mHz).
- `phase_deg` — station constant phase (deg), at the epoch the scan's rate
  column is referenced to (the scan's own mean time), so it is comparable
  across scans only through a difference taken within one scan.

Summed from every stage-B component the fringe stage owns (delay/rate/
constant terms — not adhoc or bandpass).
Values are gauge-fixed to the solve's reference pin; a within-scan
difference against the same feed of a reference station is gauge-invariant
(the reported `delay_rel`/`rate_rel`).

The table is dense: a (station, feed, scan) the solve never constrained
reads back as the identity 0, indistinguishable from a genuine gauge-zero.
Mask it with the detection/flag info (`suspect_fringes`,
`info.flagged_ant`/`flagged_scan`); this accessor is a pure θ-decode and
does not consult the detections.

Scan index matches the scan-group ordering of [`fringe_snr_table`](@ref) and
`info.det_scan`.
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
    # First time index landing in each scan segment — used to look up every plan's
    # own segment id for this scan (a `GlobalTime` inter-feed plan maps them all to 1, a
    # `PerScan` one to the scan itself, so the same lookup handles both bases).
    t0 = zeros(Int, nscan)
    for ti in eachindex(refplan.tseg_id)
        k = refplan.tseg_id[ti]
        (1 <= k <= nscan && t0[k] == 0) && (t0[k] = ti)
    end
    out = NamedTuple[]
    for k in eachindex(t0)
        ti = t0[k]
        ti == 0 && continue
        for a in 1:nant, f in 1:2
            d = 0.0; r = 0.0; p = 0.0
            hd = false; hr = false; hp = false
            for (plan, kind) in comps
                node = _feed_node(plan.tying, f)            # fseg 1: stage-B terms are GlobalFrequency
                node == 0 && continue
                v = _component_leaf(plan, θ)[1, node, 1, plan.tseg_id[ti], a]
                if kind === :delay
                    d += v; hd = true
                elseif kind === :rate
                    r += v; hr = true
                else
                    p += v; hp = true
                end
            end
            push!(
                out, (;
                    scan = k, station = a, feed = f,
                    delay_ns = hd ? d * 1.0e9 : NaN,       # τ (s) → ns
                    rate_mHz = hr ? r * 1.0e3 : NaN,       # ṙ (Hz) → mHz
                    phase_deg = hp ? rad2deg(p) : NaN,   # φ (rad) → deg
                ),
            )
        end
    end
    return out
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
# subtends at radius |z| (the same small-angle form `phase_series_with_noise`
# uses). Once σ reaches |z| the phase is unconstrained, so the bar saturates at
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
    fringe_station_flags(sol::CalibrationSolution) -> Vector{NamedTuple}

The (station, scan) pairs the stage-B solve left UNCONSTRAINED — no ACCEPTED
detection (`pfa <= Stationization.pfa_max`) on any of the station's baselines,
so nothing put it in a fringe group (the EHT-HOPS flag criterion). A measured
but rejected baseline does not rescue it: such a row constrains the fit without
fixing a fringe location. These stations carry
identity gains for those scans, and `calibrate` flags their baselines there
(`apply_flags = true`). Rows
`(; scan, scan_name, ant, station)`; empty when every participating station
was constrained (or the solution predates flag recording).
"""
function fringe_station_flags(sol::CalibrationSolution)
    info = _fringe_info(sol)
    haskey(info, :flagged_ant) || return NamedTuple[]
    ant(name) = something(findfirst(==(name), sol.geom.stations), 0)
    scname(s) = s <= length(sol.geom.scan_names) ? String(sol.geom.scan_names[s]) : string(s)
    rows = [
        (;
            scan = Int(info.flagged_scan[i]), scan_name = scname(Int(info.flagged_scan[i])),
            ant = ant(info.flagged_ant[i]), station = String(info.flagged_ant[i]),
        )
            for i in eachindex(info.flagged_ant)
    ]
    sort!(rows; by = r -> (r.scan, r.ant))
    return rows
end

"""
    suspect_fringes(sol::CalibrationSolution; pfa_max = 1.0e-4) -> Vector{NamedTuple}

Screen the solution's ACCEPTED detections against a false-alarm probability of
`pfa_max`: the rows the stage-B solve treated as real fringes whose PFA exceeds
the threshold given here. The search pass records every measured cell, accepted
or not, so this reads the `detected` ones only — a rejected cell is not a suspect
fringe, it is a non-detection.

At the default this returns the empty set by construction, since acceptance is a
PFA test at the solve's own `Stationization.pfa_max`. It earns its keep when
passed something STRICTER than the solve used: those are the accepted detections
that would flip under a tighter threshold, i.e. the marginal ones worth eyeballing.

Rows `(; scan, a, b, sta_a, sta_b, pol, snr, pfa)`, most-suspect (largest `pfa`)
first. Needs no data read.
"""
function suspect_fringes(sol::CalibrationSolution; pfa_max::Real = 1.0e-4)
    info = _fringe_info(sol)
    haskey(info, :det_pfa) || return NamedTuple[]
    ant(name) = something(findfirst(==(name), sol.geom.stations), 0)
    # A solution written before the table recorded rejected cells holds detections
    # only, so every row of one counts as accepted.
    detected = get(info, :det_detected, nothing)
    accepted(i) = detected === nothing || detected[i]
    rows = [
        (;
            scan = Int(info.det_scan[i]), a = ant(info.det_ant_a[i]), b = ant(info.det_ant_b[i]),
            sta_a = String(info.det_ant_a[i]), sta_b = String(info.det_ant_b[i]),
            pol = (Int(info.det_feed_a[i]), Int(info.det_feed_b[i])), snr = Float64(info.det_snr[i]), pfa = Float64(info.det_pfa[i]),
        )
            for i in eachindex(info.det_pfa) if accepted(i) && info.det_pfa[i] > pfa_max
    ]
    sort!(rows; by = r -> r.pfa, rev = true)
    return rows
end

"""
    print_solve_timing(sol::CalibrationSolution; io = stdout, top = 5)

Profiling summary of a solve, GENERIC over every step (built-in or
third-party): one line per step that published timing (`sol.steps[name].t_pass`, the pass's total wall time, and `.timing`, a `Scan`-indexed
`DimStack` of `decode`/`work`/`reduce` task-seconds — with N concurrent group
tasks the wall share is up to N× smaller), then the `top` slowest scans of
whichever step spent the most per-scan time. Prints a notice when the
solution carries no step timing at all.
"""
function print_solve_timing(sol::CalibrationSolution; io = stdout, top::Integer = 5)
    timed = [(; name, info) for (name, info) in sol.steps if haskey(info, :t_pass)]
    isempty(timed) && return println(io, "No solve timing recorded in this solution")
    println(io)
    println(
        io, "Solve timing (peak ", get(sol.info, :ntasks_used, "?"), " concurrent groups × ",
        get(sol.info, :inner_tasks, "?"), " inner tasks)",
    )
    for s in timed
        line = @sprintf("  %-10s %8.1f s wall", String(s.name), s.info.t_pass)
        if haskey(s.info, :timing)
            t = s.info.timing
            line *= @sprintf(
                "   (Σ decode %8.1f s, Σ work %8.1f s)", sum(t.decode), sum(t.work),
            )
        end
        println(io, line)
    end
    heaviest = argmax(s -> haskey(s.info, :timing) ? sum(s.info.timing.work) : -Inf, timed)
    haskey(heaviest.info, :timing) || return nothing
    t = heaviest.info.timing
    tot = t.decode .+ t.work
    ord = sortperm(tot; rev = true)
    println(io, "  slowest scans (", heaviest.name, ", decode/work s):")
    for i in ord[1:min(Int(top), length(ord))]
        println(io, @sprintf("    scan %3d  %6.1f = %.1f/%.1f", i, tot[i], t.decode[i], t.work[i]))
    end
    return nothing
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
