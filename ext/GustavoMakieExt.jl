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

import Gustavo.Bandpass
using Gustavo.UVData: UVSet
using Gustavo.UVData: pol_products
using Gustavo.Bandpass:
    BandpassDataset, BandpassSolverSetup, BandpassSolverState,
    _DataLike, _scans,
    _baseline_scan_blocks, _concat_scan_blocks,
    resolve_plot_polarizations,
    resolve_gain_polarizations, resolve_gain_sites,
    gain_quantity_label, gain_quantity_series,
    finite_series_ylims, shared_track,
    coherence_label, residual_phase_coherence,
    phase_series, phase_noise_series,
    amplitude_series, amplitude_noise_series,
    scan_averaged_amplitude_series, amplitude_range_label,
    baseline_index, baseline_bandpass_diagnostics,
    bandpass_residual_stats, residual_stats_annotation

# ── Color and annotation helpers ────────────────────────────────────────────

Bandpass.diagnostic_scan_colormap(nscan) =
    cgrad(:tol_muted, max(nscan, 1); categorical = true)

const diagnostic_scan_colormap = Bandpass.diagnostic_scan_colormap

function Bandpass.annotate_coherence!(ax, stats; fontsize = 11)
    return text!(
        ax, 0.98, 0.96;
        text = coherence_label(stats),
        space = :relative, align = (:right, :top), fontsize = fontsize
    )
end
const annotate_coherence! = Bandpass.annotate_coherence!

function Bandpass.plot_noise_segments!(
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
        colorrange = (1, max(nscan, 1)),
        linewidth = linewidth,
    )
    isempty(cap_xs) || linesegments!(
        ax, cap_xs, cap_ys;
        color = scan_index,
        alpha = alpha,
        colormap = scan_wheel,
        colorrange = (1, max(nscan, 1)),
        linewidth = linewidth,
    )
    return ax
end
const plot_noise_segments! = Bandpass.plot_noise_segments!

# ── stability_plotting_config (closure carries Makie text! call) ────────────

function Bandpass.stability_plotting_config(quantity; relative = false)
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
const stability_plotting_config = Bandpass.stability_plotting_config

# ── plot_stability ──────────────────────────────────────────────────────────

function Bandpass.plot_stability(
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
            kw = (color = s, colormap = scan_wheel, colorrange = (1, max(nscan, 1)), markersize = 9)
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

    Colorbar(parent[1:length(pol_idx), 3], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return parent
end

function Bandpass.plot_stability(
        data::UVSet, corr::UVSet, bl_plot;
        quantity = :phase, pol = :parallel, relative = false, comparison_weights = :input,
    )
    fig = Figure(size = (900, 280 * length(resolve_plot_polarizations(data; pol = pol)[1]) + 40))
    Bandpass.plot_stability(fig, data, corr, bl_plot; quantity = quantity, pol = pol, relative = relative, comparison_weights = comparison_weights)
    return fig
end

# ── plot_baseline_phases ────────────────────────────────────────────────────

function Bandpass.plot_baseline_phases(
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
            kw = (color = s, colormap = scan_wheel, colorrange = (1, max(nscan, 1)), markersize = 4)
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

    Colorbar(parent[1:length(pol_labels), 3], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return parent
end

function Bandpass.plot_baseline_phases(
        data::UVSet, corr::UVSet, bl_plot;
        relative = true, comparison_weights = :input,
    )
    fig = Figure(size = (1100, 900))
    Bandpass.plot_baseline_phases(fig, data, corr, bl_plot; relative = relative, comparison_weights = comparison_weights)
    return fig
end

# ── plot_gain_solutions ─────────────────────────────────────────────────────

function Bandpass.plot_gain_solutions(parent, gains, data::_DataLike; quantity = :phase, pol = :all, sites = :all, relative = true)
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
            tracks = [series(vec(gains[s, ai, pi, :])) for s in 1:nscan]
            shared = shared_track(tracks)
            if isnothing(shared)
                for s in 1:nscan
                    kw = (color = s, colormap = scan_wheel, colorrange = (1, max(nscan, 1)), markersize = 4)
                    scatter!(ax, tracks[s]; kw...)
                end
            else
                lines!(ax, shared; color = :black, linewidth = 2.0)
            end
        end
    end
    Colorbar(parent[1:length(site_idx), length(pol_idx) + 1], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return parent
end

function Bandpass.plot_gain_solutions(gains, data::_DataLike; quantity = :phase, pol = :all, sites = :all, relative = true)
    site_idx, _ = resolve_gain_sites(data; sites = sites)
    fig = Figure(size = (900, 180 * length(site_idx)))
    Bandpass.plot_gain_solutions(fig, gains, data; quantity = quantity, pol = pol, sites = sites, relative = relative)
    return fig
end

# ── plot_baseline_bandpass ──────────────────────────────────────────────────

function Bandpass.plot_baseline_bandpass(
        parent,
        setup::BandpassSolverSetup, gains, bl_plot;
        pol = :parallel,
        normalize_by_source = false,
    )
    data = setup.data
    bi = baseline_index(data, bl_plot)
    nscan = length(data.scans)
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_idx, pol_labels = resolve_plot_polarizations(data; pol = pol)
    baseline_label = join(bl_plot, "-")

    for (row, (pi, lab)) in enumerate(zip(pol_idx, pol_labels))
        amp_title = normalize_by_source ? "$(baseline_label) $lab |V / S|" : "$(baseline_label) $lab |V|"
        phase_title = normalize_by_source ? "$(baseline_label) $lab arg(V / S)" : "$(baseline_label) $lab arg(V)"
        amp_ylabel = normalize_by_source ? "amp / S" : "amplitude"
        phase_ylabel = normalize_by_source ? "phase - S" : "phase (rad)"
        ax_amp = Axis(parent[row, 1]; title = amp_title, xlabel = "channel", ylabel = amp_ylabel)
        ax_phase = Axis(parent[row, 2]; title = phase_title, xlabel = "channel", ylabel = phase_ylabel)
        linkxaxes!(ax_amp, ax_phase)

        observed, observed_weights, model, _, weights, gain_product, source_per_scan =
            baseline_bandpass_diagnostics(setup, gains, bi, pi)
        amp_series = Vector{Float64}[]
        phase_series_blocks = Vector{Float64}[]
        amp_noise_blocks = Vector{Float64}[]
        phase_noise_blocks = Vector{Float64}[]
        plotted_scans = NamedTuple[]

        for s in 1:nscan
            valid_scan = vec(weights[s, :]) .> 0
            any(valid_scan) || continue

            nchan = size(observed, 2)
            obs_amp = fill(NaN, nchan)
            obs_amp_noise = fill(NaN, nchan)
            obs_phase = fill(NaN, nchan)
            obs_phase_noise = fill(NaN, nchan)
            model_amp = fill(NaN, nchan)
            model_phase = fill(NaN, nchan)
            for c in 1:nchan
                v = observed[s, c]
                w = observed_weights[s, c]
                m_full = model[s, c]
                gp = gain_product[s, c]
                src = source_per_scan[s, c]
                (isfinite(real(v)) && isfinite(imag(v)) && w > 0 && isfinite(w)) || continue
                sigma = 1.0 / sqrt(w)

                if normalize_by_source
                    (isfinite(real(src)) && isfinite(imag(src)) && abs(src) > 0) || continue
                    r = v / src
                    obs_amp[c] = abs(r)
                    obs_phase[c] = angle(r)
                    obs_amp_noise[c] = sigma / abs(src)
                    obs_phase_noise[c] = sigma / abs(src)
                    if isfinite(real(gp)) && isfinite(imag(gp))
                        model_amp[c] = abs(gp)
                        model_phase[c] = angle(gp)
                    end
                else
                    obs_amp[c] = abs(v)
                    obs_phase[c] = angle(v)
                    obs_amp_noise[c] = sigma
                    obs_phase_noise[c] = abs(v) > 0 ? sigma / abs(v) : NaN
                    if isfinite(real(m_full)) && isfinite(imag(m_full))
                        model_amp[c] = abs(m_full)
                        model_phase[c] = angle(m_full)
                    end
                end
            end

            push!(amp_series, obs_amp, model_amp)
            push!(phase_series_blocks, obs_phase, model_phase)
            push!(amp_noise_blocks, obs_amp_noise)
            push!(phase_noise_blocks, obs_phase_noise)
            push!(
                plotted_scans, (;
                    scan = s,
                    obs_amp,
                    obs_amp_noise,
                    model_amp,
                    obs_phase,
                    obs_phase_noise,
                    model_phase,
                ),
            )
        end

        shared_amp_track = normalize_by_source ? shared_track(getfield.(plotted_scans, :model_amp)) : nothing
        shared_phase_track = normalize_by_source ? shared_track(getfield.(plotted_scans, :model_phase)) : nothing

        for entry in plotted_scans
            color_kw = (color = entry.scan, colormap = scan_wheel, colorrange = (1, max(nscan, 1)))
            marker_kw = merge(color_kw, (markersize = 8,))
            line_kw = merge(color_kw, (linewidth = 2.0, alpha = 0.9))

            scatter!(ax_amp, entry.obs_amp; marker_kw...)
            plot_noise_segments!(ax_amp, entry.obs_amp, entry.obs_amp_noise, entry.scan, scan_wheel, nscan)
            scatter!(ax_phase, entry.obs_phase; marker_kw...)
            plot_noise_segments!(ax_phase, entry.obs_phase, entry.obs_phase_noise, entry.scan, scan_wheel, nscan)
            isnothing(shared_amp_track) && lines!(ax_amp, entry.model_amp; line_kw...)
            isnothing(shared_phase_track) && lines!(ax_phase, entry.model_phase; line_kw...)
        end

        if !isnothing(shared_amp_track)
            lines!(ax_amp, shared_amp_track; color = :black, linewidth = 2.4)
        end
        if !isnothing(shared_phase_track)
            lines!(ax_phase, shared_phase_track; color = :black, linewidth = 2.4)
        end

        amp_lims = finite_series_ylims(amp_series, amp_noise_blocks)
        phase_lims = finite_series_ylims(phase_series_blocks, phase_noise_blocks)
        if !isnothing(amp_lims)
            ylims!(ax_amp, max(0.0, amp_lims[1]), amp_lims[2])
        else
            ylims!(ax_amp, low = 0.0)
        end
        isnothing(phase_lims) || ylims!(ax_phase, phase_lims...)

        model_color = (!isnothing(shared_amp_track) || !isnothing(shared_phase_track)) ? :black : :gray30
        model_label = normalize_by_source ? "G_a · conj(G_b)" : "G_a · S · conj(G_b)"
        legend_elements = [
            MarkerElement(color = :gray30, marker = :circle, markersize = 8),
            LineElement(color = model_color, linewidth = 2.0),
        ]
        axislegend(ax_amp, legend_elements, ["data", model_label]; position = :rt, framevisible = false)
    end

    Colorbar(parent[1:length(pol_idx), 3], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return parent
end

function Bandpass.plot_baseline_bandpass(
        setup::BandpassSolverSetup, gains, bl_plot;
        pol = :parallel,
        normalize_by_source = false,
    )
    npol = length(resolve_plot_polarizations(setup.data; pol = pol)[1])
    fig = Figure(size = (1100, 280 * npol + 40))
    Bandpass.plot_baseline_bandpass(fig, setup, gains, bl_plot; pol = pol, normalize_by_source = normalize_by_source)
    return fig
end

function Bandpass.plot_baseline_bandpass(
        setup::BandpassSolverSetup, state::BandpassSolverState, bl_plot;
        pol = :parallel,
    )
    return Bandpass.plot_baseline_bandpass(setup, state.gains, bl_plot; pol = pol)
end

# ── plot_baseline_bandpass_residuals ────────────────────────────────────────

function Bandpass.plot_baseline_bandpass_residuals(
        parent,
        setup::BandpassSolverSetup, gains, bl_plot;
        pol = :parallel,
    )
    data = setup.data
    bi = baseline_index(data, bl_plot)
    nscan = length(data.scans)
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_idx, pol_labels = resolve_plot_polarizations(data; pol = pol)
    residual_rows = bandpass_residual_stats(setup, gains; by = :baseline)
    baseline_label = join(bl_plot, "-")

    for (row, (pi, lab)) in enumerate(zip(pol_idx, pol_labels))
        ax_amp = Axis(parent[row, 1]; title = "$(baseline_label) $lab |V / S|", xlabel = "channel", ylabel = "amp / S")
        ax_phase = Axis(parent[row, 2]; title = "$(baseline_label) $lab arg(V / S)", xlabel = "channel", ylabel = "phase - S")
        ax_real_res = Axis(parent[row, 3]; title = "$(baseline_label) $lab residual Re", xlabel = "channel", ylabel = "sqrt(w) * Re(v - m)")
        ax_imag_res = Axis(parent[row, 4]; title = "$(baseline_label) $lab residual Im", xlabel = "channel", ylabel = "sqrt(w) * Im(v - m)")

        for ax in (ax_phase, ax_real_res, ax_imag_res)
            linkxaxes!(ax_amp, ax)
        end

        observed, observed_weights, _model, normalized_residual, weights, gain_product, source_per_scan =
            baseline_bandpass_diagnostics(setup, gains, bi, pi)
        amp_series = Vector{Float64}[]
        phase_series_blocks = Vector{Float64}[]
        amp_noise_blocks = Vector{Float64}[]
        phase_noise_blocks = Vector{Float64}[]
        real_res_series = Vector{Float64}[[0.0]]
        imag_res_series = Vector{Float64}[[0.0]]
        plotted_scans = NamedTuple[]

        hlines!(ax_real_res, [0.0]; color = (:black, 0.35), linestyle = :dash)
        hlines!(ax_imag_res, [0.0]; color = (:black, 0.35), linestyle = :dash)
        text!(
            ax_imag_res, 0.98, 0.96;
            text = residual_stats_annotation(residual_rows, baseline_label, lab),
            space = :relative, align = (:right, :top), fontsize = 11,
        )

        for s in 1:nscan
            valid_scan = vec(weights[s, :]) .> 0
            any(valid_scan) || continue

            nchan = size(observed, 2)
            obs_amp = fill(NaN, nchan)
            obs_amp_noise = fill(NaN, nchan)
            obs_phase = fill(NaN, nchan)
            obs_phase_noise = fill(NaN, nchan)
            model_amp = fill(NaN, nchan)
            model_phase = fill(NaN, nchan)
            for c in 1:nchan
                v = observed[s, c]
                w = observed_weights[s, c]
                gp = gain_product[s, c]
                src = source_per_scan[s, c]
                (isfinite(real(v)) && isfinite(imag(v)) && w > 0 && isfinite(w)) || continue
                (isfinite(real(src)) && isfinite(imag(src)) && abs(src) > 0) || continue
                sigma = 1.0 / sqrt(w)
                r = v / src
                obs_amp[c] = abs(r)
                obs_phase[c] = angle(r)
                obs_amp_noise[c] = sigma / abs(src)
                obs_phase_noise[c] = sigma / abs(src)
                if isfinite(real(gp)) && isfinite(imag(gp))
                    model_amp[c] = abs(gp)
                    model_phase[c] = angle(gp)
                end
            end
            res_real = real.(vec(normalized_residual[s, :]))
            res_imag = imag.(vec(normalized_residual[s, :]))

            push!(amp_series, obs_amp, model_amp)
            push!(phase_series_blocks, obs_phase, model_phase)
            push!(amp_noise_blocks, obs_amp_noise)
            push!(phase_noise_blocks, obs_phase_noise)
            push!(real_res_series, res_real)
            push!(imag_res_series, res_imag)
            push!(
                plotted_scans, (;
                    scan = s,
                    obs_amp,
                    obs_amp_noise,
                    model_amp,
                    obs_phase,
                    obs_phase_noise,
                    model_phase,
                    res_real,
                    res_imag,
                ),
            )
        end

        shared_model_amp = shared_track(getfield.(plotted_scans, :model_amp))
        shared_model_phase = shared_track(getfield.(plotted_scans, :model_phase))

        for entry in plotted_scans
            color_kw = (color = entry.scan, colormap = scan_wheel, colorrange = (1, max(nscan, 1)))
            marker_kw = merge(color_kw, (markersize = 8,))
            line_kw = merge(color_kw, (linewidth = 2.0, alpha = 0.9))

            scatter!(ax_amp, entry.obs_amp; marker_kw...)
            plot_noise_segments!(ax_amp, entry.obs_amp, entry.obs_amp_noise, entry.scan, scan_wheel, nscan)
            isnothing(shared_model_amp) && lines!(ax_amp, entry.model_amp; line_kw...)
            scatter!(ax_phase, entry.obs_phase; marker_kw...)
            plot_noise_segments!(ax_phase, entry.obs_phase, entry.obs_phase_noise, entry.scan, scan_wheel, nscan)
            isnothing(shared_model_phase) && lines!(ax_phase, entry.model_phase; line_kw...)
            scatter!(ax_real_res, entry.res_real; marker_kw...)
            scatter!(ax_imag_res, entry.res_imag; marker_kw...)
        end

        if !isnothing(shared_model_amp)
            lines!(ax_amp, shared_model_amp; color = :black, linewidth = 2.4)
        end
        if !isnothing(shared_model_phase)
            lines!(ax_phase, shared_model_phase; color = :black, linewidth = 2.4)
        end

        amp_lims = finite_series_ylims(amp_series, amp_noise_blocks)
        phase_lims = finite_series_ylims(phase_series_blocks, phase_noise_blocks)
        real_res_lims = finite_series_ylims(real_res_series)
        imag_res_lims = finite_series_ylims(imag_res_series)
        if !isnothing(amp_lims)
            ylims!(ax_amp, max(0.0, amp_lims[1]), amp_lims[2])
        else
            ylims!(ax_amp, low = 0.0)
        end
        isnothing(phase_lims) || ylims!(ax_phase, phase_lims...)
        isnothing(real_res_lims) || ylims!(ax_real_res, real_res_lims...)
        isnothing(imag_res_lims) || ylims!(ax_imag_res, imag_res_lims...)

        model_color = (!isnothing(shared_model_amp) || !isnothing(shared_model_phase)) ? :black : :gray30
        legend_elements = [
            MarkerElement(color = :gray30, marker = :circle, markersize = 8),
            LineElement(color = model_color, linewidth = 2.0),
        ]
        axislegend(ax_amp, legend_elements, ["data", "G_a · conj(G_b)"]; position = :rt, framevisible = false)
    end

    Colorbar(parent[1:length(pol_idx), 5], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return parent
end

function Bandpass.plot_baseline_bandpass_residuals(
        setup::BandpassSolverSetup, gains, bl_plot;
        pol = :parallel,
    )
    npol = length(resolve_plot_polarizations(setup.data; pol = pol)[1])
    fig = Figure(size = (1500, 280 * npol + 40))
    Bandpass.plot_baseline_bandpass_residuals(fig, setup, gains, bl_plot; pol = pol)
    return fig
end

function Bandpass.plot_baseline_bandpass_residuals(
        setup::BandpassSolverSetup, state::BandpassSolverState, bl_plot;
        pol = :parallel,
    )
    return Bandpass.plot_baseline_bandpass_residuals(setup, state.gains, bl_plot; pol = pol)
end

end # module
