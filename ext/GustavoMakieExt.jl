module GustavoMakieExt

using Makie
using Makie:
    Figure, Axis, Colorbar, Label,
    scatter!, lines!, linesegments!, hlines!, text!,
    linkxaxes!, linkyaxes!, ylims!, axislegend,
    MarkerElement, LineElement,
    hidexdecorations!, hideydecorations!,
    colsize!, Fixed, resize_to_layout!,
    cgrad

using Printf: @sprintf

import Gustavo.UVData
using Gustavo.UVData:
    UVSet, pol_products,
    _scans, _baseline_scan_blocks, _concat_scan_blocks,
    resolve_plot_polarizations,
    resolve_gain_polarizations, resolve_gain_sites,
    gain_quantity_label, gain_quantity_series,
    finite_series_ylims, shared_track,
    coherence_label, residual_phase_coherence,
    phase_series, phase_noise_series,
    amplitude_series, amplitude_noise_series,
    scan_averaged_amplitude_series, amplitude_range_label

# ── Color and annotation helpers ────────────────────────────────────────────

UVData.diagnostic_scan_colormap(nscan) =
    cgrad(:tol_muted, max(nscan, 1); categorical = true)

const diagnostic_scan_colormap = UVData.diagnostic_scan_colormap

_scan_colorrange(nscan::Integer) =
    nscan <= 1 ? (0.5, 1.5) : (1.0, Float64(nscan))

function UVData.annotate_coherence!(ax, stats; fontsize = 11)
    return text!(
        ax, 0.98, 0.96;
        text = coherence_label(stats),
        space = :relative, align = (:right, :top), fontsize = fontsize
    )
end
const annotate_coherence! = UVData.annotate_coherence!

function UVData.plot_noise_segments!(
        ax, series, noise, scan_index, scan_wheel, nscan;
        alpha = 0.75, linewidth = 1.8, cap_width = 0.22
    )
    xs = Float64[]
    ys = Float64[]
    cap_xs = Float64[]
    cap_ys = Float64[]
    for (channel, (value, sigma)) in enumerate(zip(series, noise))
        (isfinite(value) && isfinite(sigma) && sigma > 0) || continue
        push!(xs, channel, channel)
        push!(ys, value - sigma, value + sigma)
        push!(cap_xs, channel - cap_width, channel + cap_width)
        push!(cap_ys, value - sigma, value - sigma)
        push!(cap_xs, channel - cap_width, channel + cap_width)
        push!(cap_ys, value + sigma, value + sigma)
    end
    isempty(xs) && return ax
    linesegments!(
        ax, xs, ys;
        color = scan_index,
        alpha = alpha,
        colormap = scan_wheel,
        colorrange = _scan_colorrange(nscan),
        linewidth = linewidth,
    )
    isempty(cap_xs) || linesegments!(
        ax, cap_xs, cap_ys;
        color = scan_index,
        alpha = alpha,
        colormap = scan_wheel,
        colorrange = _scan_colorrange(nscan),
        linewidth = linewidth,
    )
    return ax
end
const plot_noise_segments! = UVData.plot_noise_segments!

# ── stability_plotting_config (closure carries Makie text! call) ────────────

function UVData.stability_plotting_config(quantity; relative = false)
    if quantity == :phase
        ylabel = relative ? "phase relative to ref (rad)" : "absolute phase (rad)"
        summarize = (vis_block, weight_block; groups = nothing) -> phase_series(vis_block, weight_block; relative = relative)
        scatter_series = summarize
        scatter_noise = (vis_block, weight_block; groups = nothing) -> phase_noise_series(vis_block, weight_block; relative = relative)
        annotate_metric! = function (ax, vis_block, weight_block, scan_groups)
            return annotate_coherence!(ax, residual_phase_coherence(vis_block, weight_block; groups = scan_groups); fontsize = 12)
        end
    elseif quantity == :amplitude
        ylabel = relative ? "amplitude / ref" : "amplitude"
        summarize = (vis_block, weight_block; groups = nothing) -> scan_averaged_amplitude_series(
            vis_block, weight_block; relative = relative, groups = groups
        )
        scatter_series = (vis_block, weight_block; groups = nothing) -> amplitude_series(
            vis_block, weight_block; relative = relative
        )
        scatter_noise = (vis_block, weight_block; groups = nothing) -> amplitude_noise_series(
            vis_block, weight_block; relative = relative
        )
        annotate_metric! = function (ax, vis_block, weight_block, scan_groups)
            return text!(
                ax, 0.98, 0.96;
                text = amplitude_range_label(vis_block, weight_block; groups = scan_groups),
                space = :relative, align = (:right, :top), fontsize = 12
            )
        end
    else
        error("quantity must be :phase or :amplitude")
    end

    return ylabel, summarize, scatter_series, scatter_noise, annotate_metric!
end
const stability_plotting_config = UVData.stability_plotting_config

# ── plot_stability ──────────────────────────────────────────────────────────

function UVData.plot_stability(
        parent,
        data::UVSet, corr::UVSet, bl_plot;
        quantity = :phase, pol = :parallel, relative = false, comparison_weights = :input,
    )
    nscan = length(_scans(data))
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_idx, pol_labels = resolve_plot_polarizations(data; pol = pol)
    ylabel, summarize, scatter_series, scatter_noise, annotate_metric! =
        stability_plotting_config(quantity; relative = relative)

    for (row, (pi, lab)) in enumerate(zip(pol_idx, pol_labels))
        ax_b = Axis(
            parent[row, 1]; title = "$(join(bl_plot, "-")) $lab  before",
            xlabel = "channel", ylabel = ylabel,
        )
        ax_a = Axis(
            parent[row, 2]; title = "$(join(bl_plot, "-")) $lab  after",
            xlabel = "channel", ylabel = ylabel,
        )
        linkxaxes!(ax_b, ax_a)
        linkyaxes!(ax_b, ax_a)

        blocks = _baseline_scan_blocks(data, corr, bl_plot, pi)
        use_input_weights = comparison_weights === :input
        if !(comparison_weights in (:input, :native))
            error("comparison_weights must be :input or :native")
        end

        vis_b_cat, scan_groups = _concat_scan_blocks(blocks; field = :vis_b)
        vis_a_cat, _ = _concat_scan_blocks(blocks; field = :vis_a)
        w_b_cat, _ = _concat_scan_blocks(blocks; field = :w_b)
        w_a_native_cat, _ = _concat_scan_blocks(blocks; field = :w_a)
        w_a_cat = use_input_weights ? w_b_cat : w_a_native_cat

        summary_before = summarize(vis_b_cat, w_b_cat; groups = scan_groups)
        summary_after = summarize(vis_a_cat, w_a_cat; groups = scan_groups)
        lines!(ax_b, summary_before; color = :black, linewidth = 2, linestyle = :dot)
        lines!(ax_a, summary_after; color = :black, linewidth = 2, linestyle = :dot)
        plotted_series = Any[summary_before, summary_after]
        plotted_noise = Any[]

        annotate_metric!(ax_b, vis_b_cat, w_b_cat, scan_groups)
        annotate_metric!(ax_a, vis_a_cat, w_a_cat, scan_groups)

        for b in blocks
            s = b.sid
            kw = (color = s, colormap = scan_wheel, colorrange = _scan_colorrange(nscan), markersize = 9)
            w_b_before = b.w_b
            w_b_after = use_input_weights ? b.w_b : b.w_a
            before_phase = scatter_series(b.vis_b, w_b_before; groups = nothing)
            after_phase = scatter_series(b.vis_a, w_b_after; groups = nothing)
            before_noise = scatter_noise(b.vis_b, w_b_before; groups = nothing)
            after_noise = scatter_noise(b.vis_a, w_b_after; groups = nothing)
            scatter!(ax_b, before_phase; kw...)
            scatter!(ax_a, after_phase; kw...)
            plot_noise_segments!(ax_b, before_phase, before_noise, s, scan_wheel, nscan)
            plot_noise_segments!(ax_a, after_phase, after_noise, s, scan_wheel, nscan)
            push!(plotted_series, before_phase, after_phase)
            push!(plotted_noise, before_noise, after_noise)
        end

        ylims = finite_series_ylims(plotted_series, plotted_noise)
        isnothing(ylims) || ylims!(ax_b, ylims...)
    end

    Colorbar(parent[1:length(pol_idx), 3], colormap = scan_wheel, limits = _scan_colorrange(nscan), label = "Scan")
    return parent
end

function UVData.plot_stability(
        data::UVSet, corr::UVSet, bl_plot;
        quantity = :phase, pol = :parallel, relative = false, comparison_weights = :input,
    )
    fig = Figure(size = (900, 280 * length(resolve_plot_polarizations(data; pol = pol)[1]) + 40))
    UVData.plot_stability(fig, data, corr, bl_plot; quantity = quantity, pol = pol, relative = relative, comparison_weights = comparison_weights)
    return fig
end

# ── plot_baseline_phases ────────────────────────────────────────────────────

function UVData.plot_baseline_phases(
        parent,
        data::UVSet, corr::UVSet, bl_plot;
        relative = true, comparison_weights = :input,
    )
    nscan = length(_scans(data))
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_labels = collect(pol_products(data))
    ylabel = relative ? "phase relative to ref (rad)" : "absolute phase (rad)"
    use_input_weights = comparison_weights === :input
    if !(comparison_weights in (:input, :native))
        error("comparison_weights must be :input or :native")
    end

    for (row, (pi, lab)) in enumerate(zip(eachindex(pol_labels), pol_labels))
        ax_b = Axis(
            parent[row, 1]; title = "$(join(bl_plot, "-")) $lab before",
            xlabel = "channel", ylabel = ylabel,
        )
        ax_a = Axis(
            parent[row, 2]; title = "$(join(bl_plot, "-")) $lab after",
            xlabel = "channel", ylabel = ylabel,
        )
        linkxaxes!(ax_b, ax_a)
        linkyaxes!(ax_b, ax_a)

        blocks = _baseline_scan_blocks(data, corr, bl_plot, pi)
        vis_b_cat, scan_groups = _concat_scan_blocks(blocks; field = :vis_b)
        vis_a_cat, _ = _concat_scan_blocks(blocks; field = :vis_a)
        w_b_cat, _ = _concat_scan_blocks(blocks; field = :w_b)
        w_a_native_cat, _ = _concat_scan_blocks(blocks; field = :w_a)
        w_a_cat = use_input_weights ? w_b_cat : w_a_native_cat
        plotted_series = Any[]

        annotate_coherence!(ax_b, residual_phase_coherence(vis_b_cat, w_b_cat; groups = scan_groups))
        annotate_coherence!(ax_a, residual_phase_coherence(vis_a_cat, w_a_cat; groups = scan_groups))

        for b in blocks
            s = b.sid
            kw = (color = s, colormap = scan_wheel, colorrange = _scan_colorrange(nscan), markersize = 4)
            w_b_before = b.w_b
            w_b_after = use_input_weights ? b.w_b : b.w_a
            before_phase = phase_series(b.vis_b, w_b_before; relative = relative)
            after_phase = phase_series(b.vis_a, w_b_after; relative = relative)
            before_noise = phase_noise_series(b.vis_b, w_b_before; relative = relative)
            after_noise = phase_noise_series(b.vis_a, w_b_after; relative = relative)
            scatter!(ax_b, before_phase; kw...)
            scatter!(ax_a, after_phase; kw...)
            plot_noise_segments!(ax_b, before_phase, before_noise, s, scan_wheel, nscan)
            plot_noise_segments!(ax_a, after_phase, after_noise, s, scan_wheel, nscan)
            push!(plotted_series, before_phase, after_phase)
        end

        ylims = finite_series_ylims(plotted_series)
        isnothing(ylims) || ylims!(ax_b, ylims...)
    end

    Colorbar(parent[1:length(pol_labels), 3], colormap = scan_wheel, limits = _scan_colorrange(nscan), label = "Scan")
    return parent
end

function UVData.plot_baseline_phases(
        data::UVSet, corr::UVSet, bl_plot;
        relative = true, comparison_weights = :input,
    )
    fig = Figure(size = (1100, 900))
    UVData.plot_baseline_phases(fig, data, corr, bl_plot; relative = relative, comparison_weights = comparison_weights)
    return fig
end

# ── plot_gain_solutions ─────────────────────────────────────────────────────

function UVData.plot_gain_solutions(parent, gains, data::UVSet; quantity = :phase, pol = :all, sites = :all, relative = true)
    nscan = length(_scans(data))
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_idx, pol_labels = resolve_gain_polarizations(data; pol = pol)
    site_idx, site_labels = resolve_gain_sites(data; sites = sites)
    ylabel = gain_quantity_label(quantity; relative = relative)
    series = gain_quantity_series(quantity; relative = relative)

    for (row, ai) in enumerate(site_idx)
        axes_row = Axis[]
        for (col, (pi, lab)) in enumerate(zip(pol_idx, pol_labels))
            ax = Axis(
                parent[row, col];
                ylabel = site_labels[row], xlabel = "channel",
                title = (row == 1 ? "$(lab) gain  $ylabel" : ""),
            )
            push!(axes_row, ax)
        end

        for ax in axes_row[2:end]
            linkxaxes!(axes_row[1], ax)
            linkyaxes!(axes_row[1], ax)
        end

        for (ax, pi) in zip(axes_row, pol_idx)
            # gains layout: (Frequency, Ti, Ant, Pol). Pull each Ti slice
            # — yields a length-nchan vector for the (ai, pi) site/pol.
            # `Base.parent` qualified because `parent` is the
            # GridPosition argument above; on plain Arrays it's identity.
            gains_arr = Base.parent(gains)
            tracks = [series(vec(gains_arr[:, s, ai, pi])) for s in 1:nscan]
            shared = shared_track(tracks)
            if isnothing(shared)
                for s in 1:nscan
                    kw = (color = s, colormap = scan_wheel, colorrange = _scan_colorrange(nscan), markersize = 4)
                    scatter!(ax, tracks[s]; kw...)
                end
            else
                lines!(ax, shared; color = :black, linewidth = 2.0)
            end
        end
    end
    Colorbar(parent[1:length(site_idx), length(pol_idx) + 1], colormap = scan_wheel, limits = _scan_colorrange(nscan), label = "Scan")
    return parent
end

function UVData.plot_gain_solutions(gains, data::UVSet; quantity = :phase, pol = :all, sites = :all, relative = true)
    site_idx, _ = resolve_gain_sites(data; sites = sites)
    fig = Figure(size = (900, 180 * length(site_idx)))
    UVData.plot_gain_solutions(fig, gains, data; quantity = quantity, pol = pol, sites = sites, relative = relative)
    return fig
end

include("GustavoMakieExt_fringe.jl")
include("GustavoMakieExt_coherence.jl")

end # module
