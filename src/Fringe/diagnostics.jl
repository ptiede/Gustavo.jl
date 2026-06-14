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
    g = evaluate_gains(ev, sol.θ)            # (nchan, ntime, nant, 2)
    1 <= ti <= size(g, 2) || error("fringe_gain_spectrum: ti=$ti out of range 1:$(size(g, 2))")
    return sol.geom.channel_freqs, g[:, ti, :, :]
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
    g = evaluate_gains(ev, sol.θ)            # (nchan, ntime, nant, 2)
    1 <= ci <= size(g, 1) || error("fringe_gain_time_series: ci=$ci out of range 1:$(size(g, 1))")
    return sol.geom.times, g[ci, :, :, :]
end
