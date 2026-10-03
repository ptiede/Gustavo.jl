# ── Fringe-search plot (GustavoMakieExt) ──────────────────────────────────
#
# Implements `plot_fringe_search`, declared in `Gustavo.Fring`.

import Gustavo.Fring
using Gustavo.Fring: FringeSearchMap, BaselineFringeMap

# Station label: the name from `sol.geom.stations` when available, else a
# generic `ant{i}` fallback.
_site_label(names, i::Integer) =
    (names !== nothing && i <= length(names)) ? String(names[i]) : string("ant", i)

# Probabilities span many decades: scientific notation below 10⁻³.
_fmt_pfa(x::Real) = !isfinite(x) ? "NaN" :
    (x > 0 && x < 1.0e-3 ? @sprintf("%.1e", x) : string(round(x; digits = 3)))

_bl_label((a, b)::Tuple, names) = string(_site_label(names, a), "–", _site_label(names, b))

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

function Fring.plot_fringe_search(
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

function Fring.plot_fringe_search(parent, m::BaselineFringeMap; kwargs...)
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
    return Fring.plot_fringe_search(parent, m.map; title = title, kwargs...)
end

function Fring.plot_fringe_search(m::Union{BaselineFringeMap, FringeSearchMap}; kwargs...)
    fig = Figure(size = (900, 640))
    Fring.plot_fringe_search(fig, m; kwargs...)
    return fig
end
