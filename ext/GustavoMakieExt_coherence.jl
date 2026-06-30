# ── Coherence-loss plots (GustavoMakieExt) ───────────────────────────────────
#
# Implements the `plot_coherence` / `plot_coherence_matrix` stubs from
# `Gustavo.UVData`. The data is computed Makie-free by `coherence_report`; this
# only draws.
#
# - `plot_coherence`        — η versus Δt and Δν as curves; the aggregate bold,
#   per-baseline traces faint, and (with `nlabel`) the worst baselines coloured
#   and labelled in a legend.
# - `plot_coherence_matrix` — a baseline × interval heatmap (rows labelled by
#   station-pair code, sorted worst-first): the readable "which station is bad"
#   view when 45 spaghetti traces are not.

import Gustavo.UVData
using Gustavo.UVData: CoherenceReport, CoherenceCurve, coherence_headline
using Makie: heatmap!, Legend

# Distinct colours for highlighted (worst) baselines.
const _COH_HL_COLORS = [
    :red, :royalblue, :seagreen, :darkorange, :purple,
    :saddlebrown, :magenta, :teal, :goldenrod, :crimson,
]

# Per-baseline column indices honoring a selector (cross baselines are all rows).
function _coherence_baseline_indices(report::CoherenceReport, sel)
    sel === :all && return collect(eachindex(report.bl_pairs))
    sel isa Integer && return [Int(sel)]
    sel isa AbstractVector && return collect(Int.(sel))
    error("plot_coherence baselines selector must be :all, an Integer, or a vector of indices")
end

# The worst `n` baselines (lowest η at the coarsest interval) among `bls`, ranked
# on the time curve when present else the frequency curve. Empty if `n ≤ 0`. Reuses
# the global ranking from `UVData.worst_baselines`, then filters to the subset `bls`
# (preserving rank order) — `worst_baselines` ranks across all baselines.
function _coherence_worst(report::CoherenceReport, bls::Vector{Int}, n::Integer)
    n <= 0 && return Int[]
    sel = Set(bls)
    cand = [bi for bi in UVData.worst_baselines(report, length(report.bl_pairs)) if bi in sel]
    return cand[1:min(Int(n), length(cand))]
end

# Draw one curve into `ax`: faint per-baseline traces, the highlighted ones in
# colour, the aggregate bold on top. Returns the highlighted line handles (for the
# shared legend), keyed to `hl` order.
function _draw_coherence_curve!(ax, curve::CoherenceCurve, bls, hl, xscale)
    handles = Any[]
    isempty(curve.intervals) && return handles
    x = curve.intervals .* xscale
    hlset = Set(hl)
    for bi in bls
        bi in hlset && continue
        y = curve.eta_baseline[:, bi]
        any(isfinite, y) && lines!(ax, x, y; color = (:gray, 0.3), linewidth = 1)
    end
    for (j, bi) in enumerate(hl)
        c = _COH_HL_COLORS[(j - 1) % length(_COH_HL_COLORS) + 1]
        h = lines!(ax, x, curve.eta_baseline[:, bi]; color = c, linewidth = 2)
        push!(handles, h)
    end
    lines!(ax, x, curve.eta; color = :black, linewidth = 3)
    scatter!(ax, x, curve.eta; color = :black, markersize = 7)
    return handles
end

function UVData.plot_coherence(parent, report::CoherenceReport; baselines = :all, nlabel::Integer = 0)
    bls = _coherence_baseline_indices(report, baselines)
    hl = _coherence_worst(report, bls, nlabel)
    h = coherence_headline(report)

    axt = Axis(
        parent[1, 1];
        xscale = log10, xlabel = "time-averaging Δt (s)", ylabel = "coherence η",
        title = isfinite(h.eta_time) ? string("vs Δt   (aggregate η = ", round(h.eta_time; digits = 3), ")") : "vs Δt",
    )
    handles = _draw_coherence_curve!(axt, report.time, bls, hl, 1.0)

    axf = Axis(
        parent[1, 2];
        xscale = log10, xlabel = "freq-averaging Δν (MHz)", ylabel = "coherence η",
        title = isfinite(h.eta_freq) ? string("vs Δν   (aggregate η = ", round(h.eta_freq; digits = 3), ")") : "vs Δν",
    )
    _draw_coherence_curve!(axf, report.freq, bls, hl, 1.0e-6)

    for ax in (axt, axf)
        ylims!(ax, 0.0, 1.05)
    end
    if !isempty(handles)
        Legend(parent[1, 3], handles, [UVData.coherence_baseline_label(report, bi) for bi in hl], "worst baselines"; labelsize = 9)
    end
    Label(
        parent[0, :],
        string("Coherence loss  [", join(report.pol_products, ","), "]  ", length(report.bl_pairs), " baselines");
        fontsize = 14, font = :bold,
    )
    return parent
end

function UVData.plot_coherence(report::CoherenceReport; baselines = :all, nlabel::Integer = 0)
    fig = Figure(size = (nlabel > 0 ? 1180 : 980, 420))
    UVData.plot_coherence(fig, report; baselines = baselines, nlabel = nlabel)
    return fig
end

function UVData.plot_coherence_matrix(parent, report::CoherenceReport; axis::Symbol = :time, sortworst::Bool = true)
    axis in (:time, :freq) || error("plot_coherence_matrix: axis must be :time or :freq")
    curve = axis === :time ? report.time : report.freq
    isempty(curve.intervals) && error("plot_coherence_matrix: no $(axis) intervals (single sample/channel)")
    M = curve.eta_baseline                       # (ninterval, nbaseline)
    nint, nbl = size(M)

    # Row order: worst (lowest η at the coarsest interval) first; NaN rows last.
    rank(bi) = isfinite(M[nint, bi]) ? M[nint, bi] : Inf
    order = sortworst ? sortperm([rank(bi) for bi in 1:nbl]) : collect(1:nbl)
    Z = M[:, order]                              # heatmap: x = interval, y = baseline row
    labels = [UVData.coherence_baseline_label(report, bi) for bi in order]

    scale = axis === :time ? 1.0 : 1.0e-6
    xunit = axis === :time ? "Δt (s)" : "Δν (MHz)"
    xt = [string(round(curve.intervals[k] * scale; sigdigits = 3)) for k in 1:nint]

    ax = Axis(
        parent[1, 1];
        xlabel = xunit, ylabel = "baseline (worst → best)",
        title = string("per-baseline coherence η  [", join(report.pol_products, ","), "]"),
        xticks = (1:nint, xt), yticks = (1:nbl, labels),
        yticklabelsize = 8, yreversed = true,
    )
    hm = heatmap!(
        ax, 1:nint, 1:nbl, Z;
        colorrange = (0.0, 1.0), nan_color = (:gray, 0.35),
        colormap = cgrad([:firebrick, :orange, :gold, :yellowgreen, :seagreen]),
    )
    Colorbar(parent[1, 2], hm; label = "coherence η")
    return parent
end

function UVData.plot_coherence_matrix(report::CoherenceReport; axis::Symbol = :time, sortworst::Bool = true)
    nbl = length(report.bl_pairs)
    fig = Figure(size = (620, max(320, 16 * nbl + 130)))
    UVData.plot_coherence_matrix(fig, report; axis = axis, sortworst = sortworst)
    return fig
end
