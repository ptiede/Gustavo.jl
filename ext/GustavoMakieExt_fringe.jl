# ── Fringe-solution plots (GustavoMakieExt) ──────────────────────────────────
#
# Implements the plot stubs declared in `Gustavo.Fringe`. Each entry point has a
# `(parent, sol; …)` form that draws into a `Figure`/`GridPosition` and a
# `(sol; …)` convenience form that creates and returns a `Figure`. Gains are
# pulled through the pure extractors in `Fringe/diagnostics.jl`, so all the data
# wrangling stays Makie-free and tested without a backend.

import Gustavo.Fringe
using Gustavo.UVData: UVSet
using Gustavo.Calibration: CalibrationSolution
using Gustavo.Fringe: fringe_gain_spectrum, fringe_gain_time_series, fringe_snr_table
using Gustavo.Fringe: BaselineFringeData, baseline_fringe_data, baseline_pol_index

# Resolve a `sites`/`feeds` selector into a vector of integer indices.
_fringe_indices(sel::Colon, n::Integer) = collect(1:n)
_fringe_indices(sel::Integer, n::Integer) = [Int(sel)]
_fringe_indices(sel::AbstractVector{<:Integer}, n::Integer) = collect(Int.(sel))
_fringe_indices(sel::Symbol, n::Integer) = sel === :all ? collect(1:n) :
    error("site/feed selector Symbol must be :all")

# Station label: the actual code when available (from `sol.info.ant_names` /
# `BaselineFringeData.ant_names`), else a generic `ant{i}` fallback.
_site_label(names, i::Integer) =
    (names !== nothing && i <= length(names)) ? String(names[i]) : string("ant", i)
_feed_label(f::Integer) = string("feed", f)

# ── plot_fringe_spectrum: phase vs frequency, rows = sites, cols = feeds ───────
function Fringe.plot_fringe_spectrum(
        parent, sol::CalibrationSolution;
        sites = :all, feeds = :all, ti::Integer = 1,
    )
    freqs, g = fringe_gain_spectrum(sol; ti = ti)
    nant = size(g, 2)
    names = get(sol.info, :ant_names, nothing)
    site_idx = _fringe_indices(sites, nant)
    feed_idx = _fringe_indices(feeds, 2)
    fghz = freqs ./ 1.0e9
    for (row, ai) in enumerate(site_idx)
        axrow = Axis[]
        for (col, fi) in enumerate(feed_idx)
            ax = Axis(
                parent[row, col];
                xlabel = "frequency (GHz)", ylabel = _site_label(names, ai),
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
    names = get(sol.info, :ant_names, nothing)
    site_idx = _fringe_indices(sites, nant)
    feed_idx = _fringe_indices(feeds, 2)
    for (row, ai) in enumerate(site_idx)
        axrow = Axis[]
        for (col, fi) in enumerate(feed_idx)
            ax = Axis(
                parent[row, col];
                xlabel = "time (h)", ylabel = _site_label(names, ai),
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

# ── plot_baseline_fringes: per-baseline before/after, grid of panels ───────────
_bl_label((a, b)::Tuple, names) = string(_site_label(names, a), "–", _site_label(names, b))

# Cross-baseline column indices (skip autocorrelations), honoring a selector.
function _baseline_indices(data::BaselineFringeData, sel)
    cross = [bi for (bi, (a, b)) in enumerate(data.bl_pairs) if a != b]
    sel === :all && return cross
    sel isa Integer && return [Int(sel)]
    sel isa AbstractVector && return collect(Int.(sel))
    error("baselines selector must be :all, an Integer, or a vector of indices")
end

function Fringe.plot_baseline_fringes(
        parent, data::BaselineFringeData;
        kind::Symbol = :freq, show::Symbol = :phase, pol = :parallel, baselines = :all,
    )
    kind in (:freq, :time) || error("kind must be :freq or :time")
    show in (:phase, :amp) || error("show must be :phase or :amp")
    p = baseline_pol_index(data, pol)
    bls = _baseline_indices(data, baselines)
    if kind === :freq
        x = data.freqs ./ 1.0e9
        before, after = data.spec_before, data.spec_after
        xlab = "frequency (GHz)"
    else
        x = data.times
        before, after = data.tser_before, data.tser_after
        xlab = "time (h)"
    end
    reduce_y = show === :phase ? angle : abs
    ylab = show === :phase ? "phase (rad)" : "amplitude"

    n = length(bls)
    ncols = max(1, ceil(Int, sqrt(n)))
    nrows = ceil(Int, n / ncols)
    local axfirst = nothing
    for (k, bi) in enumerate(bls)
        row, col = fldmod1(k, ncols)
        ax = Axis(
            parent[row, col];
            title = _bl_label(data.bl_pairs[bi], data.ant_names),
            xlabel = (row == nrows ? xlab : ""), ylabel = (col == 1 ? ylab : ""),
        )
        yb = reduce_y.(@view before[:, bi, p])
        ya = reduce_y.(@view after[:, bi, p])
        sb = scatter!(ax, x, yb; markersize = 4, color = (:steelblue, 0.6))
        sa = scatter!(ax, x, ya; markersize = 4, color = (:firebrick, 0.8))
        show === :phase && ylims!(ax, -π, π)
        if axfirst === nothing
            axfirst = ax
            axislegend(ax, [sb, sa], ["before", "after"]; position = :rt, labelsize = 9, framevisible = false)
        end
    end
    Label(
        parent[0, :],
        string(
            data.source, "  scan ", data.scan, "  [", data.pol_products[p], "]  ",
            show, " vs ", kind === :freq ? "freq" : "time",
            isfinite(data.max_snr) ? string("  (max SNR ", round(data.max_snr; digits = 1), ")") : ""
        );
        fontsize = 14, font = :bold,
    )
    return parent
end

function Fringe.plot_baseline_fringes(data::BaselineFringeData; kind::Symbol = :freq, kwargs...)
    bls = _baseline_indices(data, get(kwargs, :baselines, :all))
    n = max(1, length(bls))
    ncols = max(1, ceil(Int, sqrt(n)))
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (360 * ncols + 40, 240 * nrows + 60))
    Fringe.plot_baseline_fringes(fig, data; kind = kind, kwargs...)
    return fig
end

function Fringe.plot_baseline_fringes(
        uvset::UVSet, sol::CalibrationSolution;
        scan_index = nothing, kind::Symbol = :freq, kwargs...,
    )
    data = baseline_fringe_data(uvset, sol; scan_index = scan_index)
    return Fringe.plot_baseline_fringes(data; kind = kind, kwargs...)
end
