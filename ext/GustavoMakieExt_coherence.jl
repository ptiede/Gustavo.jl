# ── Coherence plot (GustavoMakieExt) ─────────────────────────────────────────
#
# Implements `plot_coherence`, declared in `Gustavo.Fring`: η against Δt and Δν
# for one scan and feed pair of a `coherence_report` result, a faint trace per
# antenna pair, the pooled η bold, and the `nlabel` worst pairs colored.

using Gustavo.Fring: AveragingTime, AveragingBandwidth
using DimensionalData: AbstractDimStack, At, hasdim
using Makie: Legend

const _COH_HL_COLORS = [
    :red, :royalblue, :seagreen, :darkorange, :purple,
    :saddlebrown, :magenta, :teal, :goldenrod, :crimson,
]

function _check_coherence_selection(coh::AbstractDimStack)
    ok = hasdim(coh[:time], AntennaPair) && ndims(coh[:time]) == 2 && ndims(coh[:freq]) == 2
    ok || throw(
        ArgumentError(
            "plot_coherence: expected one scan and feed pair; select them first, " *
                "e.g. coh[Scan = 1, FeedPair = At((1, 1))]"
        )
    )
    return nothing
end

# The `n` antenna pairs with the lowest finite η at the longest time interval.
function _coherence_worst(eta, n::Integer)
    n <= 0 && return eltype(lookup(eta, AntennaPair))[]
    last_eta = eta[AveragingTime = lastindex(lookup(eta, AveragingTime))]
    pairs = [p for p in lookup(eta, AntennaPair) if isfinite(last_eta[AntennaPair = At(p)])]
    sort!(pairs; by = p -> last_eta[AntennaPair = At(p)])
    return pairs[1:min(Int(n), length(pairs))]
end

function _draw_coherence_curve!(ax, eta, pooled, X, hl, xscale)
    handles = Any[]
    isempty(lookup(eta, X)) && return handles
    x = collect(lookup(eta, X)) .* xscale
    for p in lookup(eta, AntennaPair)
        p in hl && continue
        y = collect(eta[AntennaPair = At(p)])
        any(isfinite, y) && lines!(ax, x, y; color = (:gray, 0.3), linewidth = 1)
    end
    for (j, p) in enumerate(hl)
        c = _COH_HL_COLORS[(j - 1) % length(_COH_HL_COLORS) + 1]
        push!(handles, lines!(ax, x, collect(eta[AntennaPair = At(p)]); color = c, linewidth = 2))
    end
    lines!(ax, x, collect(pooled); color = :black, linewidth = 3)
    scatter!(ax, x, collect(pooled); color = :black, markersize = 7)
    return handles
end

_pooled_title(label, pooled) =
    isempty(pooled) || !isfinite(last(pooled)) ? label : string(label, "   (pooled η = ", round(last(pooled); digits = 3), ")")

function Fring.plot_coherence(parent, coh::AbstractDimStack; nlabel::Integer = 0)
    _check_coherence_selection(coh)
    hl = _coherence_worst(coh[:time], nlabel)

    axt = Axis(
        parent[1, 1];
        xscale = log10, xlabel = "time-averaging Δt (s)", ylabel = "coherence η",
        title = _pooled_title("vs Δt", coh[:time_pooled]),
    )
    handles = _draw_coherence_curve!(axt, coh[:time], coh[:time_pooled], AveragingTime, hl, 1.0)
    axf = Axis(
        parent[1, 2];
        xscale = log10, xlabel = "freq-averaging Δν (MHz)", ylabel = "coherence η",
        title = _pooled_title("vs Δν", coh[:freq_pooled]),
    )
    _draw_coherence_curve!(axf, coh[:freq], coh[:freq_pooled], AveragingBandwidth, hl, 1.0e-6)
    for ax in (axt, axf)
        ylims!(ax, 0.0, 1.05)
    end
    isempty(handles) || Legend(parent[1, 3], handles, [join(p, "–") for p in hl], "worst baselines"; labelsize = 9)
    Label(
        parent[0, :], join(["Coherence"; map(_refdim_label, refdims(coh[:time]))...], "  ");
        fontsize = 14, font = :bold,
    )
    return parent
end

function Fring.plot_coherence(coh::AbstractDimStack; nlabel::Integer = 0)
    fig = Figure(size = (nlabel > 0 ? 1180 : 980, 420))
    Fring.plot_coherence(fig, coh; nlabel)
    return fig
end
