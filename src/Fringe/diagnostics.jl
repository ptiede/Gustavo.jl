# ── Fringe diagnostics ───────────────────────────────────────────────────────
#
# Report-style (non-plotting) diagnostics for a fringe `CalibrationSolution`,
# plus the pure gain-extraction helpers the Makie plot stubs consume. Everything
# here works off the solved model/θ/geometry and the solver's `info` NamedTuple
# (per-scan max SNR / χ / component count), so it is Makie-free and unit-testable
# without loading a plotting backend. The plot entry points themselves
# (`plot_fringe_spectrum`, `plot_fringe_phases`, `plot_fringe_snr`) are stubs in
# `Fringe.jl`, implemented by `GustavoMakieExt`.

"""
    fringe_snr_table(sol::CalibrationSolution) -> Vector{NamedTuple}

Per-scan fringe-fit summary rows `(scan, max_snr, chi, ncomp)` pulled from the
solver's `info` (`scan_max_snr` / `scan_chi` / `scan_ncomp`, as populated by
[`solve_fringes`](@ref)). Returns an empty vector if the solution carries no
per-scan diagnostics.
"""
function fringe_snr_table(sol::CalibrationSolution)
    info = sol.info
    (haskey(info, :scan_max_snr) && haskey(info, :scan_chi) && haskey(info, :scan_ncomp)) || return NamedTuple[]
    snr = info.scan_max_snr
    chi = info.scan_chi
    ncomp = info.scan_ncomp
    n = min(length(snr), length(chi), length(ncomp))
    return [(; scan = s, max_snr = Float64(snr[s]), chi = Float64(chi[s]), ncomp = Int(ncomp[s])) for s in 1:n]
end

"""
    print_fringe_snr_table(rows; io = stdout)

Pretty-print the rows from [`fringe_snr_table`](@ref).
"""
function print_fringe_snr_table(rows; io = stdout)
    isempty(rows) && return println(io, "No per-scan fringe diagnostics available")
    println(io)
    println(io, "Fringe per-scan summary")
    println(io, "scan   max_snr      chi   ncomp")
    for r in rows
        println(
            io,
            lpad(string(r.scan), 4), "   ",
            lpad(_fmt(r.max_snr), 7), "   ",
            lpad(_fmt(r.chi), 6), "   ",
            lpad(string(r.ncomp), 5),
        )
    end
    return nothing
end

_fmt(x::Real) = isfinite(x) ? string(round(x; digits = 3)) : "NaN"

"""
    fringe_solution_summary(sol::CalibrationSolution) -> String

One-line summary of a fringe solution: antenna/scan/parameter counts and the
median per-scan max SNR.
"""
function fringe_solution_summary(sol::CalibrationSolution)
    rows = fringe_snr_table(sol)
    nant = get(sol.info, :nant, sol.layout.nant)
    nscan = get(sol.info, :nscan, length(rows))
    snrs = [r.max_snr for r in rows if isfinite(r.max_snr)]
    medsnr = isempty(snrs) ? NaN : median(snrs)
    return string(
        "FringeSolution: ", nant, " antennas, ", nscan, " scans, ",
        sol.layout.nθ, " parameters; median scan max-SNR = ", _fmt(medsnr),
    )
end

"""
    fringe_scan_groups(uvset, sol::CalibrationSolution) -> Vector{NamedTuple}

Per scan-group metadata in solver order: `(; scan_index, source, scan, max_snr)`.
Lazy — reads only leaf metadata (no visibilities), so it is cheap on a streamed
`UVSet`. Use it to choose which scans to inspect with [`baseline_fringe_data`](@ref)
/ `plot_baseline_fringes`, e.g. the highest-SNR scan of each source:

    g = fringe_scan_groups(uvset, sol)
    best = argmax(r -> r.max_snr, filter(r -> r.source == "M87", g))
"""
function fringe_scan_groups(uvset::UVSet, sol::CalibrationSolution)
    groups = _scan_group_leaves(uvset)
    snr = get(sol.info, :scan_max_snr, Float64[])
    out = NamedTuple[]
    for (gi, g) in enumerate(groups)
        info = UVData.metadata(last(first(g)))
        push!(
            out, (;
                scan_index = gi, source = info.source_name, scan = info.scan_name,
                max_snr = gi <= length(snr) ? Float64(snr[gi]) : NaN,
            ),
        )
    end
    return out
end

# ── Gain extractors (pure; consumed by the Makie plot stubs and tests) ─────────

"""
    fringe_gain_spectrum(sol::CalibrationSolution; ti = 1) -> (freqs, gains)

Evaluate the solution's complex antenna gains at time index `ti` over the full
geometry. Returns `(channel_freqs::Vector, gains::Array{Complex,3})` with `gains`
shaped `(nchan, nant, 2)` (last axis = feed). The fringe model is phase-only, so
`angle.(gains)` is the diagnostic of interest (delay slope + constant + rate/adhoc
offset at this time).
"""
function fringe_gain_spectrum(sol::CalibrationSolution; ti::Integer = 1)
    ev = GainEvaluator(sol.model, sol.layout)
    ntime = sol.layout.ntime
    1 <= ti <= ntime || error("fringe_gain_spectrum: ti=$ti out of range 1:$ntime")
    # Window to the single requested time — evaluating the full (nchan × ntime)
    # cube just to slice one column is O(ntime) wasted work (and memory) on long
    # tracks.
    g = evaluate_gains(ev, sol.θ, 1:(sol.layout.nchan), ti:ti)   # (nchan, 1, nant, 2)
    return sol.geom.channel_freqs, g[:, 1, :, :]
end

"""
    fringe_gain_time_series(sol::CalibrationSolution; ci = 1) -> (times, gains)

Evaluate the solution's complex antenna gains at channel index `ci` over the full
geometry. Returns `(times::Vector, gains::Array{Complex,3})` with `gains` shaped
`(ntime, nant, 2)`. `angle.(gains)` vs time shows the rate + adhoc phase evolution
per (station, feed).
"""
function fringe_gain_time_series(sol::CalibrationSolution; ci::Integer = 1)
    ev = GainEvaluator(sol.model, sol.layout)
    nchan = sol.layout.nchan
    1 <= ci <= nchan || error("fringe_gain_time_series: ci=$ci out of range 1:$nchan")
    # Window to the single requested channel (see fringe_gain_spectrum).
    g = evaluate_gains(ev, sol.θ, ci:ci, 1:(sol.layout.ntime))   # (1, ntime, nant, 2)
    return sol.geom.times, g[1, :, :, :]
end

# ── Per-baseline before/after data (the fringe-fit quality check) ──────────────

"""
    BaselineFringeData

Per-baseline coherent visibility averages for ONE scan, before and after applying
a fringe `CalibrationSolution`. Produced by [`baseline_fringe_data`](@ref) and
consumed by `plot_baseline_fringes`.

Fields: `source`/`scan`/`scan_index`/`max_snr` identify the scan; `bl_pairs` and
`pol_products` label the baseline and correlation axes; `freqs` (Hz, all bands
stacked) and `times` (h) the data axes. The four data arrays are weighted coherent
means (vector averages, `NaN` where a cell has no unflagged data):

- `spec_before`/`spec_after` — `(nchan, nbl, npol)`, averaged over time. `angle`
  vs frequency shows the group-delay slope (flat after a good fit); `abs` shows
  the band-averaged coherence.
- `tser_before`/`tser_after` — `(ntime, nbl, npol)`, averaged over frequency.
  `angle` vs time shows the fringe-rate slope (flat after a good fit).
"""
struct BaselineFringeData
    source::String
    scan::String
    scan_index::Int
    max_snr::Float64
    bl_pairs::Vector{Tuple{Int, Int}}
    ant_names::Vector{String}        # station codes, indexed by antenna number
    pol_products::Vector{String}
    freqs::Vector{Float64}
    times::Vector{Float64}
    spec_before::Array{ComplexF64, 3}
    spec_after::Array{ComplexF64, 3}
    tser_before::Array{ComplexF64, 3}
    tser_after::Array{ComplexF64, 3}
end

# Scan group with the largest detection SNR (the most informative to inspect),
# falling back to the first group when no per-scan SNR is recorded.
function _max_snr_scan(sol::CalibrationSolution, ngroups::Integer)
    haskey(sol.info, :scan_max_snr) || return 1
    snr = sol.info.scan_max_snr
    (isempty(snr) || all(!isfinite, snr)) && return 1
    best = argmax(i -> (isfinite(snr[i]) ? snr[i] : -Inf), 1:min(length(snr), ngroups))
    return best
end

# Divide a weighted sum by its weight, leaving NaN where there was no data.
function _coherent_mean!(sum::Array{ComplexF64}, w::Array{Float64})
    @inbounds for i in eachindex(sum, w)
        sum[i] = w[i] > 0 ? sum[i] / w[i] : ComplexF64(NaN, NaN)
    end
    return sum
end

"""
    baseline_fringe_data(uvset, sol; scan_index = nothing) -> BaselineFringeData

Materialize one scan of `uvset` and compute, per baseline and correlation product,
the weighted coherent visibility average vs frequency and vs time, BEFORE and AFTER
dividing out the fringe solution `sol`. This is the per-baseline before/after check:
a good fit flattens the phase slopes (delay in frequency, rate in time) and lifts
the coherent amplitude.

`scan_index` selects which `(source, scan)` group (in the same order
[`solve_fringes`](@ref) used); the default is the highest-SNR scan. The "after"
visibility is `V / (g_a · conj(g_b))` with gains evaluated from `sol` exactly as the
solver applies them — no second disk read of the full set, just this one scan.
"""
function baseline_fringe_data(
        uvset::UVSet, sol::CalibrationSolution;
        scan_index::Union{Integer, Nothing} = nothing,
    )
    groups = _scan_group_leaves(uvset)
    isempty(groups) && error("baseline_fringe_data: uvset has no scan groups")
    gi = scan_index === nothing ? _max_snr_scan(sol, length(groups)) : Int(scan_index)
    (1 <= gi <= length(groups)) || error("baseline_fringe_data: scan_index $gi out of range 1:$(length(groups))")

    info = UVData.metadata(last(first(groups[gi])))   # source/scan from the lazy leaf
    grp = _materialize_concat_group(groups[gi], sol.geom)
    ev = GainEvaluator(sol.model, sol.layout)
    g = evaluate_gains(ev, sol.θ, grp.g_ci, grp.g_ti)   # (nchan, nti, nant, 2)
    nchan, nti, nbl, npol = size(grp.Vg)

    sb = zeros(ComplexF64, nchan, nbl, npol); swb = zeros(Float64, nchan, nbl, npol)
    sa = zeros(ComplexF64, nchan, nbl, npol); swa = zeros(Float64, nchan, nbl, npol)
    tb = zeros(ComplexF64, nti, nbl, npol); twb = zeros(Float64, nti, nbl, npol)
    ta = zeros(ComplexF64, nti, nbl, npol); twa = zeros(Float64, nti, nbl, npol)

    @inbounds for p in 1:npol
        fa, fb = correlation_feed_pair(grp.pol_products[p])
        for bi in 1:nbl
            a, b = grp.bl_pairs[bi]
            a == b && continue                          # skip autocorrelations
            for ti in 1:nti, c in 1:nchan
                w = grp.Wg[c, ti, bi, p]
                v = grp.Vg[c, ti, bi, p]
                (w > 0 && isfinite(w) && isfinite(v)) || continue
                sb[c, bi, p] += w * v; swb[c, bi, p] += w
                tb[ti, bi, p] += w * v; twb[ti, bi, p] += w
                ga = g[c, ti, a, fa]; gb = g[c, ti, b, fb]
                denom = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(denom)) || continue
                vc = v / denom
                isfinite(vc) || continue
                sa[c, bi, p] += w * vc; swa[c, bi, p] += w
                ta[ti, bi, p] += w * vc; twa[ti, bi, p] += w
            end
        end
    end

    msnr = (haskey(sol.info, :scan_max_snr) && gi <= length(sol.info.scan_max_snr)) ?
        Float64(sol.info.scan_max_snr[gi]) : NaN
    return BaselineFringeData(
        info.source_name, info.scan_name, gi, msnr,
        copy(grp.bl_pairs), String.(collect(info.antennas.name)), copy(grp.pol_products),
        copy(grp.fg), copy(grp.tg),
        _coherent_mean!(sb, swb), _coherent_mean!(sa, swa),
        _coherent_mean!(tb, twb), _coherent_mean!(ta, twa),
    )
end

"""
    baseline_pol_index(data, pol) -> Int

Resolve a correlation-product selector (`Integer` index, or `String`/`Symbol`
label like `"PP"`) against `data.pol_products`. With `pol = :parallel` (the
default used by the plots) returns the first parallel-hand product.
"""
function baseline_pol_index(data::BaselineFringeData, pol)
    return if pol === :parallel
        idx = findfirst(p -> (fp = correlation_feed_pair(p); fp[1] == fp[2]), data.pol_products)
        idx === nothing ? 1 : idx
    elseif pol isa Integer
        Int(pol)
    else
        idx = findfirst(==(String(pol)), data.pol_products)
        idx === nothing ? error("pol $(pol) not in $(data.pol_products)") : idx
    end
end

# Coherence-weighted group delay (s) of one baseline's spectrum `z` over `freqs`
# from the per-channel phase increment: τ = ⟨angle(z[c+1] z[c]*)⟩ / (2π Δf). Uses
# the wrapped increment (no unwrap needed) and weights by |z|; skips flagged cells
# and the sub-band-boundary jumps (Δf ≫ in-band spacing) where the increment wraps.
function _baseline_delay(z::AbstractVector, freqs::AbstractVector)
    df = Float64[]
    @inbounds for c in 1:(length(freqs) - 1)
        d = freqs[c + 1] - freqs[c]
        d > 0 && push!(df, d)
    end
    isempty(df) && return NaN
    dfmed = median(df)
    num = 0.0; den = 0.0
    @inbounds for c in 1:(length(z) - 1)
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
    delay_closure(data::BaselineFringeData; pol = :parallel) -> NamedTuple

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
function delay_closure(data::BaselineFringeData; pol = :parallel)
    p = baseline_pol_index(data, pol)
    nbl = length(data.bl_pairs)
    τb = fill(NaN, nbl); τa = fill(NaN, nbl)
    for bi in 1:nbl
        a, b = data.bl_pairs[bi]
        a == b && continue
        τb[bi] = _baseline_delay(view(data.spec_before, :, bi, p), data.freqs)
        τa[bi] = _baseline_delay(view(data.spec_after, :, bi, p), data.freqs)
    end
    blindex = Dict(data.bl_pairs[bi] => bi for bi in 1:nbl)
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
        pol = data.pol_products[p], triangles = tris, closure_before = cb,
        closure_after = ca, data_delay = τb, resid_delay = τa, bl_pairs = copy(data.bl_pairs),
    )
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
