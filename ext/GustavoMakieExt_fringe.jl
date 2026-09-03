# ── Fringe-solution plots (GustavoMakieExt) ──────────────────────────────────
#
# Implements the plot stubs declared in `Gustavo.Fringe`. Each entry point has a
# `(parent, sol; …)` form that draws into a `Figure`/`GridPosition` and a
# `(sol; …)` convenience form that creates and returns a `Figure`. Gains are
# pulled through `gains` on a solution selection, so all the data wrangling
# stays Makie-free and tested without a backend.

import Gustavo.Fringe
using Gustavo.UVData: UVSet, Frequency
using Gustavo.Calibration: CalibrationSolution, gains, _freq_group_ranges
using DimensionalData: lookup, Ti
using Gustavo.Fringe: fringe_snr_table
using Gustavo.Fringe: BaselineFringeData, baseline_fringe_data, baseline_pol_index
using Gustavo.Fringe: FringeSearchMap, BaselineFringeMap, fringe_search_map, _fmt_pfa

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
        sites = :all, feeds = :all, ti::Integer = 1, freqgroup = nothing, residual::Bool = false,
    )
    # `residual = true`: plot the per-channel bandpass ripple with the per-scan delay
    # slope removed (readable — otherwise a big station delay wraps 2π·τ·(f−f0) across
    # the band and hides the ripple; the bandpass is time-invariant, so any time
    # sample works). `false`: the full solved gain phase at time `ti`.
    # NB: `parent` here is the Figure argument — keep the labelled DimArray and
    # index it positionally; its lookups carry the coordinates the plot needs.
    g = residual ? gains(sol[:bandpass, :phase, :bandpass]; Ti = 1) : gains(sol; Ti = ti)
    freqs = lookup(g, Frequency)
    phaselab = residual ? "bandpass phase (rad)" : "phase (rad)"
    # Optional restriction to one frequency group of the (possibly gappy) channel axis.
    fglab = ""
    if freqgroup !== nothing
        fgs = Fringe.fringe_freq_groups(freqs)
        (1 <= Int(freqgroup) <= length(fgs)) || error("freqgroup must be in 1:$(length(fgs)) (got $freqgroup)")
        r = fgs[Int(freqgroup)]
        fglab = @sprintf(" — freqgroup %d/%d", Int(freqgroup), length(fgs))
        freqs = freqs[r]
        g = g[r, :, :]
    end
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
                title = (row == 1 ? string(_feed_label(fi), " ", phaselab, fglab) : ""),
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

function Fringe.plot_fringe_spectrum(sol::CalibrationSolution; sites = :all, feeds = :all, ti::Integer = 1, freqgroup = nothing, residual::Bool = false)
    nrow = length(_fringe_indices(sites, sol.steps[1].layout.nant))
    ncol = length(_fringe_indices(feeds, 2))
    fig = Figure(size = (480 * ncol + 40, 220 * nrow + 40))
    Fringe.plot_fringe_spectrum(fig, sol; sites = sites, feeds = feeds, ti = ti, freqgroup = freqgroup, residual = residual)
    return fig
end

# ── plot_fringe_phases: phase vs time, rows = sites, cols = feeds ──────────────
# `ci = 0` (default) evaluates at the channel nearest the reference frequency f0,
# where the per-scan delay term contributes ~nothing. Any other channel adds
# 2π·τ·(f_ci − f0) — thousands of radians on real data — so the scan-to-scan
# phase track is wrap-scrambled by the delay and unreadable there.
function Fringe.plot_fringe_phases(
        parent, sol::CalibrationSolution;
        sites = :all, feeds = :all, ci::Integer = 0,
    )
    ci0 = ci > 0 ? Int(ci) : argmin(abs.(sol.geom.channel_freqs .- sol.geom.f0))
    g = gains(sol; Frequency = ci0)
    times = lookup(g, Ti)
    fghz = sol.geom.channel_freqs[ci0] / 1.0e9
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
                title = (row == 1 ? string(_feed_label(fi), @sprintf(" phase (rad) @ %.2f GHz", fghz)) : ""),
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

function Fringe.plot_fringe_phases(sol::CalibrationSolution; sites = :all, feeds = :all, ci::Integer = 0)
    nrow = length(_fringe_indices(sites, sol.steps[1].layout.nant))
    ncol = length(_fringe_indices(feeds, 2))
    fig = Figure(size = (480 * ncol + 40, 220 * nrow + 40))
    Fringe.plot_fringe_phases(fig, sol; sites = sites, feeds = feeds, ci = ci)
    return fig
end

# ── plot_fringe_snr: per-scan max SNR ─────────────────────────────────────────
function Fringe.plot_fringe_snr(parent, sol::CalibrationSolution)
    rows = fringe_snr_table(sol)
    scans = [Float64(r.scan) for r in rows]
    snr = [r.max_snr for r in rows]
    ax = Axis(parent[1, 1]; xlabel = "scan", ylabel = "max detection SNR", title = "Fringe per-scan SNR")
    if !isempty(scans)
        scatter!(ax, scans, snr; markersize = 8, color = :seagreen)
        lines!(ax, scans, snr; color = (:seagreen, 0.5))
    end
    return parent
end

function Fringe.plot_fringe_snr(sol::CalibrationSolution)
    fig = Figure(size = (640, 280))
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

# Triangle (baseline-matrix) layout: the stations present among `bls` are ordered
# by antenna index and placed on the diagonal; baseline (a, b) lives in the upper
# triangle at (row = pos[min], col = pos[max]). Returns the ordered antenna ids,
# a pos lookup, and the matrix dimension M.
function _triangle_positions(data::BaselineFringeData, bls)
    present = Int[]
    for bi in bls
        a, b = data.bl_pairs[bi]
        a in present || push!(present, a)
        b in present || push!(present, b)
    end
    sort!(present)
    pos = Dict(a => i for (i, a) in enumerate(present))
    return present, pos, length(present)
end

# Coherently bin per-channel complex means in bins of `bin` channels within one
# frequency-group range (NaN channels skipped); returns (x_fraction_within_group, values).
function _bin_freqgroup(spec_col, r::UnitRange{Int}, bin::Int)
    xs = Float64[]
    zs = ComplexF64[]
    lo = first(r)
    for i in lo:bin:last(r)
        hi = min(i + bin - 1, last(r))
        acc = zero(ComplexF64)
        n = 0
        for c in i:hi
            z = spec_col[c]
            (isfinite(real(z)) && isfinite(imag(z))) || continue
            acc += z
            n += 1
        end
        n == 0 && continue
        push!(xs, (0.5 * (i + hi) - lo) / max(length(r) - 1, 1))
        push!(zs, acc / n)
    end
    return xs, zs
end

# Unit phasor of a complex sample's vector mean (the panel's mean phase direction),
# or 1 if there is no finite signal. Used to re-centre phase panels so a flat "after"
# track sitting near ±π is drawn as one group instead of split across the wrap.
_unit_phasor(zs) = (s = sum(z -> isfinite(z) ? z : zero(z), zs); abs(s) > 0 ? s / abs(s) : one(ComplexF64))

function Fringe.plot_baseline_fringes(
        parent, data::BaselineFringeData;
        kind::Symbol = :freq, show::Symbol = :phase, pol = :parallel, baselines = :all,
        layout::Symbol = :triangle, bin::Integer = 0, freqgroup = nothing,
        recenter::Bool = true, drop_empty::Bool = true,
    )
    kind in (:freq, :time) || error("kind must be :freq or :time")
    show in (:phase, :amp) || error("show must be :phase or :amp")
    layout in (:triangle, :grid) || error("layout must be :triangle or :grid")
    p = baseline_pol_index(data, pol)
    bls = _baseline_indices(data, baselines)
    # Optional restriction to ONE frequency group (`freqgroup = k`): `:freq` panels
    # show only that group's channels on the REAL frequency axis (delay slopes
    # physical); `:time` panels average over only that group's channels. Essential
    # on wide multi-group data (VGOS), where the all-group view hides which group
    # misfits.
    nbg = length(data.freq_groups)
    fgsel = freqgroup === nothing ? 0 : Int(freqgroup)
    fgsel == 0 || (1 <= fgsel <= nbg) || error("freqgroup must be in 1:$nbg (got $freqgroup)")
    grpr = fgsel == 0 ? (1:length(data.freqs)) : data.freq_groups[fgsel]
    # Frequency-group structure of the frequency axis: each group gets its own
    # equal-width segment on a compressed x-axis (no dead space at the VGOS gaps),
    # channels coherently binned so every plotted point carries real SNR. `bin = 0`
    # picks ~12 points per group on many-channel data (and 1:1 below 32 channels/group).
    freqgroups = kind === :freq ? _freq_group_ranges(data.freqs) : UnitRange{Int}[]
    if fgsel != 0 && kind === :freq
        freqgroups = [r for r in freqgroups if first(r) >= first(grpr) && last(r) <= last(grpr)]
    end
    compressed = kind === :freq && fgsel == 0 && length(freqgroups) > 1
    if kind === :freq
        binw = bin > 0 ? Int(bin) : max(1, maximum(length, freqgroups) ÷ 12)
        x = data.freqs ./ 1.0e9
        before, after = data.spec_before, data.spec_after
        xlab = compressed ? "group (centre GHz)" : "frequency (GHz)"
    else
        binw = 1
        x = data.times
        before, after = fgsel == 0 ? (data.tser_before, data.tser_after) :
            (view(data.tser_freqgroup_before, :, :, :, fgsel), view(data.tser_freqgroup_after, :, :, :, fgsel))
        xlab = "time (h)"
    end
    reduce_y = show === :phase ? angle : abs
    ylab = show === :phase ? (recenter ? "phase − ⟨after⟩ (rad)" : "phase (rad)") : "amplitude"
    # Compressed-axis tick positions/labels: group centres, thinned to ≤ 8 labels.
    tickpos = Float64[]
    ticklab = String[]
    if compressed
        stride = max(1, ceil(Int, length(freqgroups) / 8))
        for (k, r) in enumerate(freqgroups)
            (k - 1) % stride == 0 || continue
            push!(tickpos, k - 0.5)
            push!(ticklab, string(round(sum(data.freqs[r]) / length(r) / 1.0e9; digits = 2)))
        end
    end

    # Drop baselines carrying no finite data for this product (e.g. a station flagged
    # out of this scan), so the grid isn't padded with blank panels. Guard against
    # dropping everything.
    if drop_empty
        keep = [
            bi for bi in bls
                if any(isfinite, @view(after[:, bi, p])) || any(isfinite, @view(before[:, bi, p]))
        ]
        isempty(keep) || (bls = keep)
    end

    # Panel placement: (row, col) per selected baseline, plus a flag for which
    # cells sit on the bottom/left edge of their column/row (for sparse axis
    # labels). In :triangle mode the diagonal carries station-name labels.
    present, pos, M = _triangle_positions(data, bls)
    if layout === :triangle
        cells = [
            (
                min(pos[data.bl_pairs[bi][1]], pos[data.bl_pairs[bi][2]]),
                max(pos[data.bl_pairs[bi][1]], pos[data.bl_pairs[bi][2]]),
            ) for bi in bls
        ]
        nrows, ncols = M, M
    else
        n = length(bls)
        ncols = max(1, ceil(Int, sqrt(n)))
        nrows = ceil(Int, n / ncols)
        cells = [fldmod1(k, ncols) for k in 1:n]
    end
    # Bottom-most filled row in each column / left-most filled col in each row.
    botrow = Dict{Int, Int}(); leftcol = Dict{Int, Int}()
    for (row, col) in cells
        botrow[col] = max(get(botrow, col, 0), row)
        leftcol[row] = min(get(leftcol, row, typemax(Int)), col)
    end

    local axfirst = nothing
    for (k, bi) in enumerate(bls)
        row, col = cells[k]
        ax = Axis(
            parent[row, col];
            title = _bl_label(data.bl_pairs[bi], data.ant_names),
            xlabel = (row == botrow[col] ? xlab : ""), ylabel = (col == leftcol[row] ? ylab : ""),
        )
        compressed && !isempty(tickpos) && (ax.xticks = (tickpos, ticklab))
        local sb, sa
        if kind === :freq
            xb = Float64[]; zbv = ComplexF64[]
            xa = Float64[]; zav = ComplexF64[]
            for (k, r) in enumerate(freqgroups)
                x0 = compressed ? k - 1 + 0.05 : data.freqs[first(r)] / 1.0e9
                xw = compressed ? 0.9 : (data.freqs[last(r)] - data.freqs[first(r)]) / 1.0e9
                fx, fz = _bin_freqgroup(view(before, :, bi, p), r, binw)
                append!(xb, x0 .+ xw .* fx); append!(zbv, fz)
                fx, fz = _bin_freqgroup(view(after, :, bi, p), r, binw)
                append!(xa, x0 .+ xw .* fx); append!(zav, fz)
            end
            # Phase only: rotate both traces so the "after" mean phase sits at 0, so a
            # flat corrected track near ±π reads as one group, not a split at the wrap.
            rot = (show === :phase && recenter) ? conj(_unit_phasor(zav)) : one(ComplexF64)
            sb = scatter!(ax, xb, reduce_y.(rot .* zbv); markersize = 5, color = (:steelblue, 0.6))
            sa = scatter!(ax, xa, reduce_y.(rot .* zav); markersize = 5, color = (:firebrick, 0.8))
            compressed && length(freqgroups) > 1 &&
                vlines!(ax, collect(1.0:(length(freqgroups) - 1)); color = (:gray, 0.3), linewidth = 0.5)
        else
            rot = (show === :phase && recenter) ? conj(_unit_phasor(@view after[:, bi, p])) : one(ComplexF64)
            yb = reduce_y.(rot .* @view before[:, bi, p])
            ya = reduce_y.(rot .* @view after[:, bi, p])
            sb = scatter!(ax, x, yb; markersize = 4, color = (:steelblue, 0.6))
            sa = scatter!(ax, x, ya; markersize = 4, color = (:firebrick, 0.8))
        end
        show === :phase && ylims!(ax, -π, π)
        if axfirst === nothing
            axfirst = ax
            axislegend(ax, [sb, sa], ["before", "after"]; position = :rt, labelsize = 9, framevisible = false)
        end
    end
    if layout === :triangle
        for (i, a) in enumerate(present)
            Label(parent[i, i], _site_label(data.ant_names, a); fontsize = 22, font = :bold, tellwidth = false, tellheight = false)
        end
    end
    Label(
        parent[0, :],
        string(
            data.source, "  scan ", data.scan, "  [", data.pol_products[p], "]  ",
            show, " vs ", kind === :freq ? "freq" : "time",
            isfinite(data.max_snr) ? string("  (max SNR ", round(data.max_snr; digits = 1), ")") : "",
            fgsel == 0 ? "" : @sprintf(
                    "   group %d/%d (%.2f–%.2f GHz)", fgsel, nbg,
                    data.freqs[first(grpr)] / 1.0e9, data.freqs[last(grpr)] / 1.0e9,
                ),
        );
        fontsize = 14, font = :bold,
    )
    return parent
end

function Fringe.plot_baseline_fringes(data::BaselineFringeData; kind::Symbol = :freq, kwargs...)
    bls = _baseline_indices(data, get(kwargs, :baselines, :all))
    layout = get(kwargs, :layout, :triangle)
    if layout === :triangle
        _, _, M = _triangle_positions(data, bls)
        ncols = nrows = max(1, M)
    else
        n = max(1, length(bls))
        ncols = max(1, ceil(Int, sqrt(n)))
        nrows = ceil(Int, n / ncols)
    end
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

# ── plot_fringe_search: delay–rate SNR surface + peak cross-sections ───────────
#
# HOPS-style fringe plot: the windowed matched-filter plane as a heatmap with the
# delay and rate cross-sections through the map peak alongside, and the refined
# detection (SNR / delay / rate / PFA) annotated. Axis units: delay in ns, rate
# in mHz (the FringeSearch window units are s / Hz).

# Number of main-lobe widths the default (`zoom = true`) view spans.
const _FRINGE_ZOOM_SPAN = 48.0

# Half-width of the main lobe along one axis: half the extent of the run through
# the peak `i` that stays above half the peak value, floored at one grid step so
# a lobe resolved by a single cell still gives a finite window.
function _lobe_halfwidth(x::AbstractVector, prof::AbstractVector, i::Integer)
    n = length(x)
    n < 2 && return zero(float(eltype(x)))
    xmin, xmax = extrema(x)
    step = (xmax - xmin) / (n - 1)
    half = prof[i] / 2
    lo = i
    while lo > firstindex(prof) && prof[lo - 1] >= half
        lo -= 1
    end
    hi = i
    while hi < lastindex(prof) && prof[hi + 1] >= half
        hi += 1
    end
    return max(abs(x[hi] - x[lo]) / 2, step)
end

# View spanning `span` main-lobe widths around the peak, holding both the refined
# detection `c` and the discrete map peak `x[i]` (which disagree exactly when the
# map is worth looking at), clipped to the searched window. `nothing` for an axis
# too short to zoom (a single-integration scan has one rate cell).
function _peak_limits(x::AbstractVector, prof::AbstractVector, i::Integer, c::Real, span::Real)
    hw = _lobe_halfwidth(x, prof, i)
    xmin, xmax = extrema(x)
    lo = max(min(c, x[i]) - span * hw / 2, xmin)
    hi = min(max(c, x[i]) + span * hw / 2, xmax)
    return hi > lo ? (lo, hi) : nothing
end

_zoom_span(zoom::Bool) = zoom ? _FRINGE_ZOOM_SPAN : nothing
_zoom_span(zoom::Real) = (zoom > 0 || error("plot_fringe_search: zoom must be positive"); Float64(zoom))

function Fringe.plot_fringe_search(
        parent, fsm::FringeSearchMap;
        title::AbstractString = "", zoom::Union{Bool, Real} = true,
    )
    isempty(fsm.delays) && error("plot_fringe_search: empty map (no unflagged data)")
    det = fsm.detection
    dns = fsm.delays .* 1.0e9                 # ns
    rmhz = fsm.rates .* 1.0e3                 # mHz
    pk = argmax(fsm.snr)                      # discrete map peak (cross-section anchor)
    peaksnr = max(fsm.snr[pk], 1.0)
    dprof = fsm.snr[:, pk[2]]
    rprof = fsm.snr[pk[1], :]

    ax = Axis(
        parent[2, 1];
        xlabel = "delay (ns)", ylabel = "rate (mHz)",
    )
    hm = heatmap!(ax, dns, rmhz, fsm.snr; colormap = :viridis, colorrange = (0.0, peaksnr))
    vlines!(ax, [det.delay * 1.0e9]; color = (:white, 0.8), linestyle = :dash, linewidth = 1)
    hlines!(ax, [det.rate * 1.0e3]; color = (:white, 0.8), linestyle = :dash, linewidth = 1)

    axd = Axis(parent[1, 1]; ylabel = "SNR", title = title, titlesize = 12)
    lines!(axd, dns, dprof; color = :steelblue)
    vlines!(axd, [det.delay * 1.0e9]; color = (:firebrick, 0.7), linestyle = :dash, linewidth = 1)
    hidexdecorations!(axd; grid = false)
    linkxaxes!(ax, axd)

    # Few ticks: this panel is a fifth of the width and a strong fringe puts four
    # digits on every label, which run together at the default tick density.
    axr = Axis(parent[2, 2]; xlabel = "SNR", xticks = Makie.LinearTicks(3))
    lines!(axr, rprof, rmhz; color = :steelblue)
    hlines!(axr, [det.rate * 1.0e3]; color = (:firebrick, 0.7), linestyle = :dash, linewidth = 1)
    hideydecorations!(axr; grid = false)
    linkyaxes!(ax, axr)

    Colorbar(parent[2, 3], hm; label = "SNR")

    span = _zoom_span(zoom)
    if span !== nothing
        # Linked axes carry these to the cross-section panels.
        dlim = _peak_limits(dns, dprof, pk[1], det.delay * 1.0e9, span)
        rlim = _peak_limits(rmhz, rprof, pk[2], det.rate * 1.0e3, span)
        dlim === nothing || xlims!(ax, dlim...)
        rlim === nothing || ylims!(ax, rlim...)
    end

    # Give the map panel most of the area, leaving the rate profile wide enough to
    # read (only when we own the layout — resizing a GridPosition parent's grid
    # would reshape the caller's figure).
    if parent isa Figure
        colsize!(parent.layout, 1, Makie.Relative(0.62))
        colsize!(parent.layout, 2, Makie.Relative(0.2))
        Makie.rowsize!(parent.layout, 2, Makie.Relative(0.75))
    end
    return parent
end

function Fringe.plot_fringe_search(parent, m::BaselineFringeMap; kwargs...)
    det = m.map.detection
    title = string(
        m.source, "  scan ", m.scan, "  ",
        _bl_label(m.bl_pair, m.ant_names), " [", m.pol, "]   ",
        "SNR ", round(det.snr; digits = 1),
        det.valid ? "" : " (below snr_min)",
        @sprintf("   τ = %.3f ns", det.delay * 1.0e9),
        @sprintf("   ṙ = %.3f mHz", det.rate * 1.0e3),
        "   PFA ", _fmt_pfa(m.map.pfa),
    )
    return Fringe.plot_fringe_search(parent, m.map; title = title, kwargs...)
end

function Fringe.plot_fringe_search(m::Union{BaselineFringeMap, FringeSearchMap}; kwargs...)
    fig = Figure(size = (900, 640))
    Fringe.plot_fringe_search(fig, m; kwargs...)
    return fig
end

function Fringe.plot_fringe_search(
        uvset::UVSet, sol::CalibrationSolution;
        zoom::Union{Bool, Real} = true, kwargs...,
    )
    return Fringe.plot_fringe_search(fringe_search_map(uvset, sol; kwargs...); zoom = zoom)
end
