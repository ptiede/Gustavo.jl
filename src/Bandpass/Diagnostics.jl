function diagnostic_weight_pair(weights, weights_corr; comparison_weights = :input)
    if comparison_weights == :input
        return weights, weights
    elseif comparison_weights == :native
        return weights, weights_corr
    else
        error("comparison_weights must be :input or :native")
    end
end

diagnostic_scan_colormap(nscan) = cgrad(get(ColorSchemes.tol_muted, range(0, 1, length = max(nscan, 1))), categorical = true)

function annotate_coherence!(ax, stats; fontsize = 11)
    return text!(
        ax, 0.98, 0.96;
        text = coherence_label(stats),
        space = :relative, align = (:right, :top), fontsize = fontsize
    )
end

"""
    plot_stability(data, corr, bl_plot;
                   quantity=:phase, pol=:parallel,
                   relative=false, comparison_weights=:input)

Two-column figure showing scan-averaged baseline stability versus channel for
`bl_plot` before (left) and after (right) correction. Set `quantity=:phase` or
`:amplitude` and choose which polarization products to plot with `pol`.

For phase plots, `relative=true` suppresses scan-constant offsets by plotting
phase relative to the reference channel. For amplitude plots, `relative=true`
normalizes each spectrum by its reference-channel amplitude.

The default `pol=:parallel` plots the two parallel-hand products. Use
`pol=:all` to include every product, or pass a single polarization label/index
or a collection of labels/indices.

For before/after comparison, the default `comparison_weights=:input` uses the
same pre-correction weights on both panels so gain-amplitude rescaling does not
change the apparent scatter purely through reweighting. Use
`comparison_weights=:native` to plot each panel with its own weights.
"""
function plot_stability(
        data::UVData, corr::UVData, bl_plot;
        quantity = :phase, pol = :parallel, relative = false, comparison_weights = :input
    )
    a_idx = findfirst(==(bl_plot[1]), data.antennas.name)
    b_idx = findfirst(==(bl_plot[2]), data.antennas.name)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl_plot")

    bl_idx = findfirst(p -> p == (a_idx, b_idx), data.bl_pairs)
    isnothing(bl_idx) && error("Baseline $bl_plot not in data")
    target_code = data.unique_bls[bl_idx]
    int_inds = findall(c -> c == target_code, data.bl_codes)

    nscan = length(data.scans)
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_idx, pol_labels = resolve_plot_polarizations(data; pol = pol)
    ylabel, summarize, scatter_series, scatter_noise, annotate_metric! = stability_plotting_config(quantity; relative = relative)
    plot_weights_before, plot_weights_after = diagnostic_weight_pair(
        data.weights, corr.weights;
        comparison_weights = comparison_weights
    )

    fig = Figure(size = (900, 280 * length(pol_idx) + 40))
    for (row, (pi, lab)) in enumerate(zip(pol_idx, pol_labels))
        ax_b = Axis(
            fig[row, 1]; title = "$(join(bl_plot, "-")) $lab  before",
            xlabel = "channel", ylabel = ylabel
        )
        ax_a = Axis(
            fig[row, 2]; title = "$(join(bl_plot, "-")) $lab  after",
            xlabel = "channel", ylabel = ylabel
        )
        linkxaxes!(ax_b, ax_a)
        linkyaxes!(ax_b, ax_a)

        vis_before = @view data.vis[int_inds, pi, :]
        vis_after = @view corr.vis[int_inds, pi, :]
        w_before = @view plot_weights_before[int_inds, pi, :]
        w_after = @view plot_weights_after[int_inds, pi, :]
        scan_groups = data.scan_idx[int_inds]

        summary_before = summarize(vis_before, w_before; groups = scan_groups)
        summary_after = summarize(vis_after, w_after; groups = scan_groups)
        lines!(ax_b, summary_before; color = :black, linewidth = 2)
        lines!(ax_a, summary_after; color = :black, linewidth = 2)
        plotted_series = Any[summary_before, summary_after]
        plotted_noise = Any[]

        annotate_metric!(ax_b, vis_before, w_before, scan_groups)
        annotate_metric!(ax_a, vis_after, w_after, scan_groups)

        for s in 1:nscan
            ii = int_inds[findall(i -> data.scan_idx[int_inds[i]] == s, eachindex(int_inds))]
            isempty(ii) && continue
            kw = (color = s, colormap = scan_wheel, colorrange = (1, max(nscan, 1)), markersize = 9)
            before_phase = scatter_series(data.vis[ii, pi, :], plot_weights_before[ii, pi, :]; groups = nothing)
            after_phase = scatter_series(corr.vis[ii, pi, :], plot_weights_after[ii, pi, :]; groups = nothing)
            before_noise = scatter_noise(data.vis[ii, pi, :], plot_weights_before[ii, pi, :]; groups = nothing)
            after_noise = scatter_noise(corr.vis[ii, pi, :], plot_weights_after[ii, pi, :]; groups = nothing)
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

    Colorbar(fig[1:length(pol_idx), 3], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return fig
end

function weighted_channel_average(vis_block, weight_block)
    nchan = size(vis_block, 2)
    avg = Vector{ComplexF64}(undef, nchan)
    for c in 1:nchan
        weights_c = vec(weight_block[:, c])
        vis_c = vec(vis_block[:, c])
        valid = (weights_c .> 0) .& isfinite.(weights_c) .& isfinite.(real.(vis_c)) .& isfinite.(imag.(vis_c))
        if any(valid)
            avg[c] = sum(weights_c[valid] .* vis_c[valid]) / sum(weights_c[valid])
        else
            avg[c] = NaN + NaN * im
        end
    end
    return avg
end

function summed_channel_weights(weight_block)
    nchan = size(weight_block, 2)
    sums = zeros(Float64, nchan)
    for c in 1:nchan
        weights_c = vec(weight_block[:, c])
        valid = (weights_c .> 0) .& isfinite.(weights_c)
        any(valid) || continue
        sums[c] = sum(weights_c[valid])
    end
    return sums
end

function thermal_noise_series(weight_block)
    sums = summed_channel_weights(weight_block)
    noise = fill(NaN, length(sums))
    valid = sums .> 0
    noise[valid] .= 1.0 ./ sqrt.(sums[valid])
    return noise
end

function amplitude_reference_index(amps, ref_idx = 1)
    if 1 <= ref_idx <= length(amps)
        ref = amps[ref_idx]
        isfinite(ref) && ref > 0 && return ref_idx
    end
    return findfirst(a -> isfinite(a) && a > 0, amps)
end

function phase_series_with_noise(vis_block, weight_block; relative = true, ref_idx = 1)
    avg = weighted_channel_average(vis_block, weight_block)
    amp = abs.(avg)
    phase = angle.(avg)
    sigma_amp = thermal_noise_series(weight_block)
    sigma_phase = fill(NaN, length(phase))

    for i in eachindex(phase)
        (isfinite(amp[i]) && amp[i] > 0 && isfinite(sigma_amp[i])) || continue
        sigma_phase[i] = sigma_amp[i] / amp[i]
    end

    relative || return phase, sigma_phase

    relative_phase = phase_relative_to_ref(phase, ref_idx)
    phase_noise = fill(NaN, length(phase))
    ref = amplitude_reference_index(amp, ref_idx)
    isnothing(ref) && return relative_phase, phase_noise

    for i in eachindex(phase)
        isfinite(relative_phase[i]) && isfinite(sigma_phase[i]) || continue
        phase_noise[i] = i == ref ? 0.0 : sigma_phase[i]
    end
    return relative_phase, phase_noise
end

function phase_series(vis_block, weight_block; relative = true)
    phase, _ = phase_series_with_noise(vis_block, weight_block; relative = relative)
    return phase
end

function phase_noise_series(vis_block, weight_block; relative = true)
    _, noise = phase_series_with_noise(vis_block, weight_block; relative = relative)
    return noise
end

function amplitude_series_with_noise(vis_block, weight_block; relative = false, ref_idx = 1)
    amp = abs.(weighted_channel_average(vis_block, weight_block))
    sigma_amp = thermal_noise_series(weight_block)
    relative || return amp, sigma_amp

    relative_amp = amplitude_relative_to_ref(amp, ref_idx)
    relative_noise = fill(NaN, length(amp))
    ref = amplitude_reference_index(amp, ref_idx)
    isnothing(ref) && return relative_amp, relative_noise

    for i in eachindex(amp)
        isfinite(relative_amp[i]) && isfinite(amp[i]) && amp[i] > 0 && isfinite(sigma_amp[i]) || continue
        relative_noise[i] = i == ref ? 0.0 : relative_amp[i] * (sigma_amp[i] / amp[i])
    end
    return relative_amp, relative_noise
end

function amplitude_series(vis_block, weight_block; relative = false)
    amp, _ = amplitude_series_with_noise(vis_block, weight_block; relative = relative)
    return amp
end

function amplitude_noise_series(vis_block, weight_block; relative = false)
    _, noise = amplitude_series_with_noise(vis_block, weight_block; relative = relative)
    return noise
end

function scan_averaged_amplitude_series(vis_block, weight_block; relative = false, groups = nothing)
    isnothing(groups) && return amplitude_series(vis_block, weight_block; relative = relative)

    nchan = size(vis_block, 2)
    accum = zeros(Float64, nchan)
    accum_weights = zeros(Float64, nchan)

    for group in sort(unique(groups))
        group == 0 && continue
        ii = findall(==(group), groups)
        isempty(ii) && continue

        spectrum = amplitude_series(view(vis_block, ii, :), view(weight_block, ii, :); relative = relative)
        spectrum_weights = summed_channel_weights(view(weight_block, ii, :))
        for c in 1:nchan
            v = spectrum[c]
            w = spectrum_weights[c]
            (isfinite(v) && isfinite(w) && w > 0) || continue
            accum[c] += w * v
            accum_weights[c] += w
        end
    end

    summary = fill(NaN, nchan)
    valid = accum_weights .> 0
    summary[valid] .= accum[valid] ./ accum_weights[valid]
    return summary
end

function amplitude_relative_to_ref(amps, ref_idx = 1)
    relative = fill(NaN, length(amps))
    (1 <= ref_idx <= length(amps)) || return relative

    ref = amps[ref_idx]
    if !(isfinite(ref) && ref > 0)
        ref_idx = findfirst(a -> isfinite(a) && a > 0, amps)
        isnothing(ref_idx) && return relative
        ref = amps[ref_idx]
    end

    for i in eachindex(amps)
        (isfinite(amps[i]) && amps[i] > 0) || continue
        relative[i] = amps[i] / ref
    end
    return relative
end

function amplitude_range_label(vis_block, weight_block; groups = nothing)
    ranges = Float64[]

    if isnothing(groups)
        for s in axes(vis_block, 1)
            amp = amplitude_series(view(vis_block, s:s, :), view(weight_block, s:s, :); relative = true)
            valid = isfinite.(amp)
            count(valid) >= 2 || continue
            push!(ranges, maximum(amp[valid]) - minimum(amp[valid]))
        end
    else
        for group in sort(unique(groups))
            group == 0 && continue
            ii = findall(==(group), groups)
            isempty(ii) && continue
            amp = amplitude_series(view(vis_block, ii, :), view(weight_block, ii, :); relative = true)
            valid = isfinite.(amp)
            count(valid) >= 2 || continue
            push!(ranges, maximum(amp[valid]) - minimum(amp[valid]))
        end
    end

    isempty(ranges) && return "no valid scans"
    return @sprintf("median rel amp span %.3f", median(ranges))
end

function resolve_plot_polarizations(data::UVData; pol = :parallel)
    if pol == :parallel
        pol_idx = parallel_hand_indices(data.metadata.pol_codes)
    elseif pol == :all
        pol_idx = eachindex(data.metadata.pol_codes)
    elseif pol isa Integer
        pol_idx = [Int(pol)]
    elseif pol isa AbstractString
        pol_idx = [resolve_single_polarization(data, pol)]
    elseif pol isa AbstractVector || pol isa Tuple
        pol_idx = Int[resolve_single_polarization(data, item) for item in pol]
    else
        error("Unsupported polarization selector: $pol")
    end

    all(1 .<= pol_idx .<= length(data.metadata.pol_codes)) || error("Polarization index out of bounds: $pol_idx")
    return pol_idx, collect(data.metadata.pol_labels[collect(pol_idx)])
end

resolve_single_polarization(data::UVData, pol::Integer) = Int(pol)
function resolve_single_polarization(data::UVData, pol::AbstractString)
    idx = findfirst(==(pol), data.metadata.pol_labels)
    isnothing(idx) && error("Polarization $pol not found in $(collect(data.metadata.pol_labels))")
    return idx
end

function stability_plotting_config(quantity; relative = false)
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

function finite_series_ylims(
        series_blocks, noise_blocks = ();
        pad_fraction = 0.08, min_pad = 1.0e-3, noise_cap_fraction = 0.5
    )
    values = Float64[]
    for block in series_blocks
        for value in block
            isfinite(value) || continue
            push!(values, value)
        end
    end

    isempty(values) && return nothing
    ymin = minimum(values)
    ymax = maximum(values)
    span = ymax - ymin
    scale = span > 0 ? span : max(abs(ymin), abs(ymax), 1.0)
    pad = max(min_pad, pad_fraction * scale)

    max_noise = 0.0
    for block in noise_blocks
        for sigma in block
            (isfinite(sigma) && sigma > 0) || continue
            max_noise = max(max_noise, sigma)
        end
    end
    pad += min(max_noise, noise_cap_fraction * scale)

    return ymin - pad, ymax + pad
end

function plot_noise_segments!(ax, series, noise, scan_index, scan_wheel, nscan; alpha = 0.75, linewidth = 1.8, cap_width = 0.22)
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
        linewidth = linewidth
    )
    isempty(cap_xs) || linesegments!(
        ax, cap_xs, cap_ys;
        color = scan_index,
        alpha = alpha,
        colormap = scan_wheel,
        colorrange = (1, max(nscan, 1)),
        linewidth = linewidth
    )
    return ax
end

"""
    residual_phase_coherence(vis_block, weight_block; groups=nothing)

Estimate the coherence factor expected from channel-to-channel phase scatter
within each scan, then average the scan-level losses.

Returns `(coherence, loss_percent, phase_rms_deg, nscan_valid)` where each
metric is first computed on a per-scan spectrum and then averaged over scans.
The optional `groups` argument lets raw integrations be grouped by scan before
forming the per-scan spectra.
"""
function spectrum_phase_coherence(vis_spectrum, weight_spectrum)
    valid = (weight_spectrum .> 0) .& isfinite.(weight_spectrum) .&
        isfinite.(real.(vis_spectrum)) .& isfinite.(imag.(vis_spectrum)) .&
        (abs.(vis_spectrum) .> 0)
    any(valid) || return (NaN, NaN, NaN, 0)

    total_weight = sum(weight_spectrum[valid])
    total_weight > 0 || return (NaN, NaN, NaN, 0)

    phasors = vis_spectrum[valid] ./ abs.(vis_spectrum[valid])
    coherence = clamp(abs(sum(weight_spectrum[valid] .* phasors) / total_weight), 0.0, 1.0)
    loss_percent = 100 * (1 - coherence)
    phase_rms_deg = coherence > 0 ? rad2deg(sqrt(max(0.0, -2 * log(coherence)))) : Inf
    return (coherence, loss_percent, phase_rms_deg, count(valid))
end

function residual_phase_coherence(vis_block, weight_block; groups = nothing)
    scan_coherences = Float64[]
    scan_losses = Float64[]
    scan_phase_rms = Float64[]

    if isnothing(groups)
        for s in axes(vis_block, 1)
            coherence, loss_percent, phase_rms_deg, nsamp = spectrum_phase_coherence(
                vec(vis_block[s, :]),
                vec(weight_block[s, :])
            )
            nsamp == 0 && continue
            push!(scan_coherences, coherence)
            push!(scan_losses, loss_percent)
            push!(scan_phase_rms, phase_rms_deg)
        end
    else
        for group in sort(unique(groups))
            group == 0 && continue
            ii = findall(==(group), groups)
            isempty(ii) && continue

            vis_spectrum = weighted_channel_average(view(vis_block, ii, :), view(weight_block, ii, :))
            weight_spectrum = summed_channel_weights(view(weight_block, ii, :))
            coherence, loss_percent, phase_rms_deg, nsamp = spectrum_phase_coherence(vis_spectrum, weight_spectrum)
            nsamp == 0 && continue
            push!(scan_coherences, coherence)
            push!(scan_losses, loss_percent)
            push!(scan_phase_rms, phase_rms_deg)
        end
    end

    isempty(scan_coherences) && return (NaN, NaN, NaN, 0)
    return (mean(scan_coherences), mean(scan_losses), mean(scan_phase_rms), length(scan_coherences))
end

function coherence_label(stats)
    _, loss_percent, phase_rms_deg, nscan = stats
    nscan == 0 && return "no valid scans"
    loss_text = isfinite(loss_percent) ? @sprintf("scan loss %.2f%%", loss_percent) : "scan loss n/a"
    rms_text = isfinite(phase_rms_deg) ? @sprintf("scan rms %.1f deg", phase_rms_deg) : "scan rms n/a"
    return string(loss_text, "\n", rms_text)
end

"""
    coherence_loss_table(avg::UVData, avg_corr::UVData;
                         pol_idx=[parallel-hand indices], pol_labels=[parallel-hand labels],
                         comparison_weights=:input)

Build a baseline-by-baseline diagnostic table of expected coherence loss from
within-scan channel phase scatter before and after bandpass correction.
`avg` and `avg_corr` are scan-averaged `UVData` objects (returned by `scan_average`).
By default this uses the original input weights on both sides so the comparison
isolates phase improvement instead of gain-amplitude reweighting. Set
`comparison_weights=:native` to evaluate the corrected data with its updated weights.
"""
function coherence_loss_table(
        avg::UVData, avg_corr::UVData;
        pol_idx = nothing, pol_labels = nothing, comparison_weights = :input
    )
    V = avg.vis
    W = avg.weights
    V_corr = avg_corr.vis
    _, W_corr = diagnostic_weight_pair(
        avg.weights, avg_corr.weights;
        comparison_weights = comparison_weights
    )
    if isnothing(pol_idx)
        rr, ll = parallel_hand_indices(avg.metadata.pol_codes)
        pol_idx = [rr, ll]
    end
    isnothing(pol_labels) && (pol_labels = avg.metadata.pol_labels[pol_idx])
    rows = NamedTuple[]

    for bi in eachindex(avg.bl_pairs)
        a, b = avg.bl_pairs[bi]
        baseline = string(avg.antennas.name[a], "-", avg.antennas.name[b])

        for (pi, lab) in zip(pol_idx, pol_labels)
            before = residual_phase_coherence(view(V, :, bi, pi, :), view(W, :, bi, pi, :))
            after = residual_phase_coherence(view(V_corr, :, bi, pi, :), view(W_corr, :, bi, pi, :))
            nsamp = max(before[4], after[4])
            nsamp == 0 && continue

            push!(
                rows, (;
                    baseline, pol = lab, nsamp,
                    coherence_before = before[1], loss_before = before[2], phase_rms_before = before[3],
                    coherence_after = after[1], loss_after = after[2], phase_rms_after = after[3],
                    loss_improvement = before[2] - after[2],
                    phase_rms_improvement = before[3] - after[3],
                )
            )
        end
    end

    sort!(rows; by = row -> (isfinite(row.loss_improvement) ? -row.loss_improvement : Inf, row.baseline, row.pol))
    return rows
end

function print_coherence_loss_table(rows; io = stdout)
    println(io)
    println(io, "Expected coherence loss from within-scan channel phase scatter")
    println(io, "baseline  pol  scans  coh_before  loss_before(%)  coh_after  loss_after(%)  improve(%)  rms_after(deg)")

    for row in rows
        coh_before = isfinite(row.coherence_before) ? @sprintf("%10.3f", row.coherence_before) : "       n/a"
        loss_before = isfinite(row.loss_before) ? @sprintf("%14.2f", row.loss_before) : "           n/a"
        coh_after = isfinite(row.coherence_after) ? @sprintf("%9.3f", row.coherence_after) : "      n/a"
        loss_after = isfinite(row.loss_after) ? @sprintf("%13.2f", row.loss_after) : "          n/a"
        improve = isfinite(row.loss_improvement) ? @sprintf("%10.2f", row.loss_improvement) : "       n/a"
        rms_after = isfinite(row.phase_rms_after) ? @sprintf("%14.2f", row.phase_rms_after) : "           n/a"

        println(
            io, rpad(row.baseline, 8), "  ",
            rpad(row.pol, 3), "  ",
            lpad(string(row.nsamp), 5), "  ",
            coh_before, "  ",
            loss_before, "  ",
            coh_after, "  ",
            loss_after, "  ",
            improve, "  ",
            rms_after
        )
    end
    return
end

function site_coherence_rows(rows, site::String; pol = nothing)
    filtered = NamedTuple[
        row for row in rows if begin
                sites = split(row.baseline, "-")
                (site in sites) && (isnothing(pol) || row.pol == pol)
            end
    ]

    sort!(
        filtered; by = row -> (
            isfinite(row.loss_after) ? -row.loss_after : Inf,
            isfinite(row.loss_improvement) ? row.loss_improvement : Inf,
            row.baseline,
        )
    )
    return filtered
end

function print_site_coherence_rows(rows, site::String; pol = nothing, io = stdout, limit = nothing)
    filtered = site_coherence_rows(rows, site; pol = pol)
    isempty(filtered) && begin
        println(
            io, "No coherence rows found for site ", site,
            isnothing(pol) ? "" : " and pol $pol"
        )
        return filtered
    end

    title = isnothing(pol) ?
        "Coherence summary for site $site" :
        "Coherence summary for site $site, pol $pol"
    println(io, title)
    println(io, "baseline  partner  pol  scans  loss_before(%)  loss_after(%)  improve(%)  rms_after(deg)")

    shown = isnothing(limit) ? filtered : first(filtered, min(limit, length(filtered)))
    for row in shown
        a, b = split(row.baseline, "-")
        partner = a == site ? b : a
        loss_before = isfinite(row.loss_before) ? @sprintf("%14.2f", row.loss_before) : "           n/a"
        loss_after = isfinite(row.loss_after) ? @sprintf("%13.2f", row.loss_after) : "          n/a"
        improve = isfinite(row.loss_improvement) ? @sprintf("%10.2f", row.loss_improvement) : "       n/a"
        rms_after = isfinite(row.phase_rms_after) ? @sprintf("%14.2f", row.phase_rms_after) : "           n/a"

        println(
            io, rpad(row.baseline, 8), "  ",
            rpad(partner, 7), "  ",
            rpad(row.pol, 3), "  ",
            lpad(string(row.nsamp), 5), "  ",
            loss_before, "  ",
            loss_after, "  ",
            improve, "  ",
            rms_after
        )
    end

    return filtered
end

function baseline_in_data(data::UVData, bl_plot)
    a_idx = findfirst(==(bl_plot[1]), data.antennas.name)
    b_idx = findfirst(==(bl_plot[2]), data.antennas.name)
    (isnothing(a_idx) || isnothing(b_idx)) && return false
    return any(p -> p == (a_idx, b_idx), data.bl_pairs)
end

function choose_diagnostic_baseline(
        avg::UVData;
        preferred = [("AA", "AX"), ("AA", "NN"), ("AA", "PV"), ("KT", "PV")]
    )
    for bl in preferred
        baseline_in_data(avg, bl) && return bl
    end

    W = avg.weights
    pols = collect(parallel_hand_indices(avg.metadata.pol_codes))
    best_score = -Inf
    best_bl = nothing

    for (bi, (a, b)) in enumerate(avg.bl_pairs)
        score = sum(@view W[:, bi, pols, :])
        if score > best_score
            best_score = score
            best_bl = (avg.antennas.name[a], avg.antennas.name[b])
        end
    end

    isnothing(best_bl) && error("No populated baseline available for diagnostics")
    return best_bl
end

"""
    plot_baseline_phases(data, corr, bl_plot; relative=true, comparison_weights=:input)

Show scan-averaged phase versus channel for all four correlation products on a
single baseline before and after correction. With `relative=false`, the figure
shows absolute phase and is sensitive to scan-constant offsets that the default
relative view intentionally suppresses. The default `comparison_weights=:input`
uses the same pre-correction weights on both panels for a like-for-like visual
comparison.
"""
function plot_baseline_phases(
        data::UVData, corr::UVData, bl_plot;
        relative = true, comparison_weights = :input
    )
    a_idx = findfirst(==(bl_plot[1]), data.antennas.name)
    b_idx = findfirst(==(bl_plot[2]), data.antennas.name)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl_plot")

    bl_idx = findfirst(p -> p == (a_idx, b_idx), data.bl_pairs)
    isnothing(bl_idx) && error("Baseline $bl_plot not in data")
    target_code = data.unique_bls[bl_idx]
    int_inds = findall(c -> c == target_code, data.bl_codes)

    nscan = length(data.scans)
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_labels = collect(data.metadata.pol_labels)
    ylabel = relative ? "phase relative to ref (rad)" : "absolute phase (rad)"
    plot_weights_before, plot_weights_after = diagnostic_weight_pair(
        data.weights, corr.weights;
        comparison_weights = comparison_weights
    )

    fig = Figure(size = (1100, 900))
    for (row, (pi, lab)) in enumerate(zip(eachindex(pol_labels), pol_labels))
        ax_b = Axis(
            fig[row, 1]; title = "$(join(bl_plot, "-")) $lab before",
            xlabel = "channel", ylabel = ylabel
        )
        ax_a = Axis(
            fig[row, 2]; title = "$(join(bl_plot, "-")) $lab after",
            xlabel = "channel", ylabel = ylabel
        )
        linkxaxes!(ax_b, ax_a)
        linkyaxes!(ax_b, ax_a)

        vis_before = @view data.vis[int_inds, pi, :]
        vis_after = @view corr.vis[int_inds, pi, :]
        w_before = @view plot_weights_before[int_inds, pi, :]
        w_after = @view plot_weights_after[int_inds, pi, :]
        scan_groups = data.scan_idx[int_inds]
        plotted_series = Any[]

        annotate_coherence!(ax_b, residual_phase_coherence(vis_before, w_before; groups = scan_groups))
        annotate_coherence!(ax_a, residual_phase_coherence(vis_after, w_after; groups = scan_groups))

        for s in 1:nscan
            ii = int_inds[findall(i -> data.scan_idx[int_inds[i]] == s, eachindex(int_inds))]
            isempty(ii) && continue
            kw = (color = s, colormap = scan_wheel, colorrange = (1, max(nscan, 1)), markersize = 4)
            before_phase = phase_series(data.vis[ii, pi, :], plot_weights_before[ii, pi, :]; relative = relative)
            after_phase = phase_series(corr.vis[ii, pi, :], plot_weights_after[ii, pi, :]; relative = relative)
            before_noise = phase_noise_series(data.vis[ii, pi, :], plot_weights_before[ii, pi, :]; relative = relative)
            after_noise = phase_noise_series(corr.vis[ii, pi, :], plot_weights_after[ii, pi, :]; relative = relative)
            scatter!(ax_b, before_phase; kw...)
            scatter!(ax_a, after_phase; kw...)
            plot_noise_segments!(ax_b, before_phase, before_noise, s, scan_wheel, nscan)
            plot_noise_segments!(ax_a, after_phase, after_noise, s, scan_wheel, nscan)
            push!(plotted_series, before_phase, after_phase)
        end

        ylims = finite_series_ylims(plotted_series)
        isnothing(ylims) || ylims!(ax_b, ylims...)
    end

    Colorbar(fig[1:length(pol_labels), 3], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return fig
end

"""
    plot_gain_solutions(gains, data; quantity=:phase, pol=:all, sites=:all, relative=true)

Grid of solved gain tracks versus channel, one row per selected site and one
column per selected polarization/feed. Set `quantity=:phase` or `:amplitude`,
choose feeds with `pol`, and restrict rows with `sites`.

The default behavior matches the legacy plot: both feeds, all sites, and
relative phase.
"""
function plot_gain_solutions(gains, data::UVData; quantity=:phase, pol=:all, sites=:all, relative=true)
    nscan = length(data.scans)
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_idx, pol_labels = resolve_gain_polarizations(data; pol = pol)
    site_idx, site_labels = resolve_gain_sites(data; sites = sites)
    ylabel = gain_quantity_label(quantity; relative = relative)
    series = gain_quantity_series(quantity; relative = relative)

    fig = Figure(size = (900, 180 * length(site_idx)))
    for (row, ai) in enumerate(site_idx)
        axes_row = Axis[]
        for (col, (pi, lab)) in enumerate(zip(pol_idx, pol_labels))
            ax = Axis(
                fig[row, col];
                ylabel = site_labels[row], xlabel = "channel",
                title = (row == 1 ? "$(lab) gain  $ylabel" : "")
            )
            push!(axes_row, ax)
        end

        for ax in axes_row[2:end]
            linkxaxes!(axes_row[1], ax)
            linkyaxes!(axes_row[1], ax)
        end

        for s in 1:nscan
            kw = (color = s, colormap = scan_wheel, colorrange = (1, max(nscan, 1)), markersize = 4)
            for (ax, pi) in zip(axes_row, pol_idx)
                scatter!(ax, series(vec(gains[s, ai, pi, :])); kw...)
            end
        end
    end
    Colorbar(fig[1:length(site_idx), length(pol_idx) + 1], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return fig
end

function baseline_bandpass_diagnostics(setup::BandpassSolverSetup, state::BandpassSolverState, bi, pi)
    data = setup.data
    nscan = size(data.vis, 1)
    nchan = size(data.vis, 4)
    observed = fill(NaN + NaN * im, nscan, nchan)
    model = fill(NaN + NaN * im, nscan, nchan)
    ratio = fill(NaN + NaN * im, nscan, nchan)
    weights = Array{Float64}(undef, nscan, nchan)
    source = fit_bandpass_source_coherencies(setup, state.gains)

    a, b = setup.bl_pairs[bi]
    fa, fb = stokes_feed_pair(setup.pol_codes[pi])
    for s in 1:nscan, c in 1:nchan
        v = data.vis[s, bi, pi, c]
        w = data.weights[s, bi, pi, c]
        weights[s, c] = w
        (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue

        src = source[s, bi, fa, fb]
        gain_model = state.gains[s, a, fa, c] * conj(state.gains[s, b, fb, c])
        model[s, c] = gain_model

        if isfinite(real(src)) && isfinite(imag(src)) && abs(src) > 0
            observed[s, c] = v / src
            if isfinite(real(gain_model)) && isfinite(imag(gain_model)) && abs(gain_model) > 0
                ratio[s, c] = observed[s, c] / gain_model
            end
        end
    end

    return observed, model, ratio, weights
end

function residual_stats_annotation(rows, baseline, pol)
    row = findfirst(r -> r.baseline == baseline && r.pol == pol, rows)
    isnothing(row) && return "no residual stats"
    stats = rows[row]
    return @sprintf(
        "chi2/real %.2f\nnorm rms %.2f\nmed |r|sqrt(w) %.2f",
        stats.chi2_per_real_component,
        stats.normalized_residual_rms,
        stats.median_abs_normalized_residual
    )
end

"""
    plot_baseline_bandpass_residuals(setup, state, bl_plot; pol=:parallel)

Diagnostic figure for one baseline showing the fitted baseline bandpass shape
and residuals for the current solver state. The left panels show the
source-normalized observed baseline bandpass with the fitted model overlaid. The
right panels show residual amplitude ratios and residual phases relative to that
model, which should sit near `1` and `0` respectively when the fit is good.
"""
function plot_baseline_bandpass_residuals(
        setup::BandpassSolverSetup, state::BandpassSolverState, bl_plot;
        pol = :parallel
    )
    data = setup.data
    bi = baseline_index(data, bl_plot)
    nscan = length(data.scans)
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_idx, pol_labels = resolve_plot_polarizations(data; pol = pol)
    residual_rows = bandpass_residual_stats(setup, state; by = :baseline)
    baseline_label = join(bl_plot, "-")

    fig = Figure(size = (1500, 280 * length(pol_idx) + 40))
    for (row, (pi, lab)) in enumerate(zip(pol_idx, pol_labels))
        ax_amp = Axis(fig[row, 1]; title = "$(baseline_label) $lab bandpass amp/ref", xlabel = "channel", ylabel = "amp / ref")
        ax_phase = Axis(fig[row, 2]; title = "$(baseline_label) $lab bandpass phase", xlabel = "channel", ylabel = "phase rel. to ref (rad)")
        ax_amp_res = Axis(fig[row, 3]; title = "$(baseline_label) $lab residual amp", xlabel = "channel", ylabel = "|obs / model|")
        ax_phase_res = Axis(fig[row, 4]; title = "$(baseline_label) $lab residual phase", xlabel = "channel", ylabel = "angle(obs / model) (rad)")

        for ax in (ax_phase, ax_amp_res, ax_phase_res)
            linkxaxes!(ax_amp, ax)
        end

        observed, model, ratio, weights = baseline_bandpass_diagnostics(setup, state, bi, pi)
        amp_series = Any[[1.0]]
        phase_series_blocks = Any[]
        amp_res_series = Any[[1.0]]
        phase_res_series = Any[[0.0]]

        hlines!(ax_amp_res, [1.0]; color = (:black, 0.35), linestyle = :dash)
        hlines!(ax_phase_res, [0.0]; color = (:black, 0.35), linestyle = :dash)
        text!(
            ax_phase_res, 0.98, 0.96;
            text = residual_stats_annotation(residual_rows, baseline_label, lab),
            space = :relative, align = (:right, :top), fontsize = 11
        )

        for s in 1:nscan
            valid_scan = vec(weights[s, :]) .> 0
            any(valid_scan) || continue

            color_kw = (color = s, colormap = scan_wheel, colorrange = (1, max(nscan, 1)))
            marker_kw = merge(color_kw, (markersize = 8,))
            line_kw = merge(color_kw, (linewidth = 2.0, alpha = 0.9))

            obs_amp = amplitude_relative_to_ref(abs.(vec(observed[s, :])))
            model_amp = amplitude_relative_to_ref(abs.(vec(model[s, :])))
            obs_phase = phase_relative_to_ref(angle.(vec(observed[s, :])))
            model_phase = phase_relative_to_ref(angle.(vec(model[s, :])))
            res_amp = abs.(vec(ratio[s, :]))
            res_phase = angle.(vec(ratio[s, :]))

            scatter!(ax_amp, obs_amp; marker_kw...)
            lines!(ax_amp, model_amp; line_kw...)
            scatter!(ax_phase, obs_phase; marker_kw...)
            lines!(ax_phase, model_phase; line_kw...)
            scatter!(ax_amp_res, res_amp; marker_kw...)
            scatter!(ax_phase_res, res_phase; marker_kw...)

            push!(amp_series, obs_amp, model_amp)
            push!(phase_series_blocks, obs_phase, model_phase)
            push!(amp_res_series, res_amp)
            push!(phase_res_series, res_phase)
        end

        amp_lims = finite_series_ylims(amp_series)
        phase_lims = finite_series_ylims(phase_series_blocks)
        amp_res_lims = finite_series_ylims(amp_res_series)
        phase_res_lims = finite_series_ylims(phase_res_series)
        isnothing(amp_lims) || ylims!(ax_amp, amp_lims...)
        isnothing(phase_lims) || ylims!(ax_phase, phase_lims...)
        isnothing(amp_res_lims) || ylims!(ax_amp_res, amp_res_lims...)
        isnothing(phase_res_lims) || ylims!(ax_phase_res, phase_res_lims...)
    end

    Colorbar(fig[1:length(pol_idx), 5], colormap = scan_wheel, limits = (1, max(nscan, 1)), label = "Scan")
    return fig
end

function resolve_gain_polarizations(data::UVData; pol = :all)
    if pol == :all
        pol_idx = [1, 2]
    elseif pol == :parallel
        pol_idx = [1, 2]
    elseif pol isa Integer
        pol_idx = [Int(pol)]
    elseif pol isa AbstractString
        pol_idx = [resolve_single_gain_polarization(data, pol)]
    elseif pol isa AbstractVector || pol isa Tuple
        pol_idx = Int[resolve_single_gain_polarization(data, item) for item in pol]
    else
        error("Unsupported gain polarization selector: $pol")
    end

    all(1 .<= pol_idx .<= 2) || error("Gain polarization index must be 1 or 2: $pol_idx")
    return pol_idx, ["Pol $pi" for pi in pol_idx]
end

resolve_single_gain_polarization(data::UVData, pol::Integer) = Int(pol)
function resolve_single_gain_polarization(data::UVData, pol::AbstractString)
    pol in ("1", "Pol 1", "11", "RR") && return 1
    pol in ("2", "Pol 2", "22", "LL") && return 2
    error("Unsupported gain polarization label: $pol")
end

function resolve_gain_sites(data::UVData; sites = :all)
    if sites == :all
        site_idx = collect(eachindex(data.antennas.name))
    elseif sites isa Integer
        site_idx = [Int(sites)]
    elseif sites isa AbstractString
        site_idx = [resolve_single_gain_site(data, sites)]
    elseif sites isa AbstractVector || sites isa Tuple
        site_idx = Int[resolve_single_gain_site(data, site) for site in sites]
    else
        error("Unsupported gain site selector: $sites")
    end

    all(1 .<= site_idx .<= length(data.antennas.name)) || error("Gain site index out of bounds: $site_idx")
    return site_idx, collect(data.antennas.name[site_idx])
end

resolve_single_gain_site(data::UVData, site::Integer) = Int(site)
function resolve_single_gain_site(data::UVData, site::AbstractString)
    idx = findfirst(==(site), data.antennas.name)
    isnothing(idx) && error("Site $site not found in $(collect(data.antennas.name))")
    return idx
end

function gain_quantity_series(quantity; relative = true)
    if quantity == :phase
        return values -> begin
            phase = angle.(values)
            relative ? phase_relative_to_ref(phase) : phase
        end
    elseif quantity == :amplitude
        return values -> begin
            amp = abs.(values)
            relative ? amplitude_relative_to_ref(amp) : amp
        end
    else
        error("quantity must be :phase or :amplitude")
    end
end

function gain_quantity_label(quantity; relative = true)
    if quantity == :phase
        return relative ? "gain phase rel. to ref (rad)" : "gain phase (rad)"
    elseif quantity == :amplitude
        return relative ? "gain amp / ref" : "gain amp"
    else
        error("quantity must be :phase or :amplitude")
    end
end

function baseline_index(data::UVData, bl::Tuple{String,String})
    a_idx = findfirst(==(bl[1]), data.antennas.name)
    b_idx = findfirst(==(bl[2]), data.antennas.name)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl")

    bl_idx = findfirst(==((a_idx, b_idx)), data.bl_pairs)
    isnothing(bl_idx) && error("Baseline $bl not in data")
    return bl_idx
end

function parallel_hand_support_summary(avg::UVData, bl::Tuple{String, String})
    bi = baseline_index(avg, bl)
    rr, ll = parallel_hand_indices(avg.metadata.pol_codes)
    pol_map = [(avg.metadata.pol_labels[rr], rr), (avg.metadata.pol_labels[ll], ll)]
    rows = NamedTuple[]

    for (lab, pi) in pol_map
        W = @view avg.weights[:, bi, pi, :]
        V = @view avg.vis[:, bi, pi, :]
        valid = (W .> 0) .& isfinite.(W) .& isfinite.(real.(V)) .& isfinite.(imag.(V))
        scan_valid = vec(sum(valid, dims = 2))
        if_valid = vec(sum(valid, dims = 1))
        total_weight = sum(W[valid])
        mean_phase_by_scan = [
            begin
                    idx = vec(valid[s, :])
                    any(idx) ? angle(sum(W[s, idx] .* (V[s, idx] ./ abs.(V[s, idx]))) / sum(W[s, idx])) : NaN
                end for s in axes(W, 1)
        ]

        push!(
            rows, (;
                baseline = join(bl, "-"),
                pol = lab,
                valid_samples = count(valid),
                total_weight = total_weight,
                scans_with_data = count(>(0), scan_valid),
                min_scan_valid = minimum(scan_valid),
                max_scan_valid = maximum(scan_valid),
                min_if_valid = minimum(if_valid),
                max_if_valid = maximum(if_valid),
                mean_phase_by_scan,
            )
        )
    end

    rr, ll = rows
    return (;
        rows,
        valid_ratio = ll.valid_samples / max(rr.valid_samples, 1),
        weight_ratio = ll.total_weight / max(rr.total_weight, eps()),
    )
end

function site_parallel_hand_support(avg::UVData, site::String)
    rows = NamedTuple[]
    for (bi, (a, b)) in enumerate(avg.bl_pairs)
        names = (avg.antennas.name[a], avg.antennas.name[b])
        site in names || continue
        summary = parallel_hand_support_summary(avg, names)
        rr, ll = summary.rows
        push!(
            rows, (;
                baseline = join(names, "-"),
                rr_valid = rr.valid_samples,
                ll_valid = ll.valid_samples,
                rr_weight = rr.total_weight,
                ll_weight = ll.total_weight,
                ll_to_rr_valid = summary.valid_ratio,
                ll_to_rr_weight = summary.weight_ratio,
            )
        )
    end
    sort!(rows; by = row -> row.ll_to_rr_weight)
    return rows
end

function print_parallel_hand_support(summary; io = stdout)
    println(io, "Parallel-hand support summary for ", summary.rows[1].baseline)
    println(io, "pol  valid_samples  total_weight   scans_with_data  min/max_scan_valid  min/max_IF_valid")
    for row in summary.rows
        println(
            io,
            lpad(row.pol, 3), "  ",
            lpad(string(row.valid_samples), 13), "  ",
            lpad(@sprintf("%.6g", row.total_weight), 12), "  ",
            lpad(string(row.scans_with_data), 15), "  ",
            lpad("$(row.min_scan_valid)/$(row.max_scan_valid)", 18), "  ",
            lpad("$(row.min_if_valid)/$(row.max_if_valid)", 16)
        )
    end
    pol_a, pol_b = getindex.(summary.rows, :pol)
    println(io, "$(pol_b)/$(pol_a) valid ratio  = ", @sprintf("%.4f", summary.valid_ratio))
    return println(io, "$(pol_b)/$(pol_a) weight ratio = ", @sprintf("%.4f", summary.weight_ratio))
end
