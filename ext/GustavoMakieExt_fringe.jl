# ── Fringe-solution plots (GustavoMakieExt) ──────────────────────────────────
#
# Implements the plot stubs declared in `Gustavo.Fringe`. Each entry point has a
# `(parent, sol; …)` form that draws into a `Figure`/`GridPosition` and a
# `(sol; …)` convenience form that creates and returns a `Figure`. Gains are
# pulled through the pure extractors in `Fringe/diagnostics.jl`, so all the data
# wrangling stays Makie-free and tested without a backend.

import Gustavo.Fringe
using Gustavo.Calibration: CalibrationSolution
using Gustavo.Fringe: fringe_gain_spectrum, fringe_gain_time_series, fringe_snr_table

# Resolve a `sites`/`feeds` selector into a vector of integer indices.
_fringe_indices(sel::Colon, n::Integer) = collect(1:n)
_fringe_indices(sel::Integer, n::Integer) = [Int(sel)]
_fringe_indices(sel::AbstractVector{<:Integer}, n::Integer) = collect(Int.(sel))
_fringe_indices(sel::Symbol, n::Integer) = sel === :all ? collect(1:n) :
    error("site/feed selector Symbol must be :all")

_site_label(i::Integer) = string("ant", i)
_feed_label(f::Integer) = string("feed", f)

# ── plot_fringe_spectrum: phase vs frequency, rows = sites, cols = feeds ───────
function Fringe.plot_fringe_spectrum(
        parent, sol::CalibrationSolution;
        sites = :all, feeds = :all, ti::Integer = 1,
    )
    freqs, g = fringe_gain_spectrum(sol; ti = ti)
    nant = size(g, 2)
    site_idx = _fringe_indices(sites, nant)
    feed_idx = _fringe_indices(feeds, 2)
    fghz = freqs ./ 1.0e9
    for (row, ai) in enumerate(site_idx)
        axrow = Axis[]
        for (col, fi) in enumerate(feed_idx)
            ax = Axis(
                parent[row, col];
                xlabel = "frequency (GHz)", ylabel = _site_label(ai),
                title = (row == 1 ? string(_feed_label(fi), " phase (rad)") : ""),
            )
            scatter!(ax, fghz, vec(angle.(g[:, ai, fi])); markersize = 5, color = :steelblue)
            push!(axrow, ax)
        end
        for ax in axrow[2:end]
            linkxaxes!(axrow[1], ax)
            linkyaxes!(axrow[1], ax)
        end
    end
    return parent
end

function Fringe.plot_fringe_spectrum(sol::CalibrationSolution; sites = :all, feeds = :all, ti::Integer = 1)
    nrow = length(_fringe_indices(sites, sol.layout.nant))
    ncol = length(_fringe_indices(feeds, 2))
    fig = Figure(size = (480 * ncol + 40, 220 * nrow + 40))
    Fringe.plot_fringe_spectrum(fig, sol; sites = sites, feeds = feeds, ti = ti)
    return fig
end

# ── plot_fringe_phases: phase vs time, rows = sites, cols = feeds ──────────────
function Fringe.plot_fringe_phases(
        parent, sol::CalibrationSolution;
        sites = :all, feeds = :all, ci::Integer = 1,
    )
    times, g = fringe_gain_time_series(sol; ci = ci)
    nant = size(g, 2)
    site_idx = _fringe_indices(sites, nant)
    feed_idx = _fringe_indices(feeds, 2)
    for (row, ai) in enumerate(site_idx)
        axrow = Axis[]
        for (col, fi) in enumerate(feed_idx)
            ax = Axis(
                parent[row, col];
                xlabel = "time (h)", ylabel = _site_label(ai),
                title = (row == 1 ? string(_feed_label(fi), " phase (rad)") : ""),
            )
            scatter!(ax, times, vec(angle.(g[:, ai, fi])); markersize = 5, color = :darkorange)
            push!(axrow, ax)
        end
        for ax in axrow[2:end]
            linkxaxes!(axrow[1], ax)
            linkyaxes!(axrow[1], ax)
        end
    end
    return parent
end

function Fringe.plot_fringe_phases(sol::CalibrationSolution; sites = :all, feeds = :all, ci::Integer = 1)
    nrow = length(_fringe_indices(sites, sol.layout.nant))
    ncol = length(_fringe_indices(feeds, 2))
    fig = Figure(size = (480 * ncol + 40, 220 * nrow + 40))
    Fringe.plot_fringe_phases(fig, sol; sites = sites, feeds = feeds, ci = ci)
    return fig
end

# ── plot_fringe_snr: per-scan max SNR (+ χ) ───────────────────────────────────
function Fringe.plot_fringe_snr(parent, sol::CalibrationSolution)
    rows = fringe_snr_table(sol)
    scans = [Float64(r.scan) for r in rows]
    snr = [r.max_snr for r in rows]
    chi = [r.chi for r in rows]
    ax = Axis(parent[1, 1]; xlabel = "scan", ylabel = "max detection SNR", title = "Fringe per-scan SNR")
    if !isempty(scans)
        scatter!(ax, scans, snr; markersize = 8, color = :seagreen)
        lines!(ax, scans, snr; color = (:seagreen, 0.5))
    end
    axχ = Axis(parent[2, 1]; xlabel = "scan", ylabel = "stationization χ", title = "")
    if !isempty(scans)
        finite = isfinite.(chi)
        any(finite) && scatter!(axχ, scans[finite], chi[finite]; markersize = 8, color = :firebrick)
    end
    linkxaxes!(ax, axχ)
    return parent
end

function Fringe.plot_fringe_snr(sol::CalibrationSolution)
    fig = Figure(size = (640, 520))
    Fringe.plot_fringe_snr(fig, sol)
    return fig
end
