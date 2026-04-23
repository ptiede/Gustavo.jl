function diagnostic_weight_pair(weights, weights_corr; comparison_weights=:input)
    if comparison_weights == :input
        return weights, weights
    elseif comparison_weights == :native
        return weights, weights_corr
    else
        error("comparison_weights must be :input or :native")
    end
end

diagnostic_scan_colormap(nscan) = cgrad(get(ColorSchemes.tol_muted, range(0, 1, length=max(nscan, 1))), categorical=true)

function annotate_coherence!(ax, stats; fontsize=11)
    text!(ax, 0.98, 0.96;
        text=coherence_label(stats),
        space=:relative, align=(:right, :top), fontsize=fontsize)
end

"""
    plot_phase_stability(data, vis_corr, weights_corr, bl_plot;
                         relative=false, comparison_weights=:input)

Two-column figure showing scan-averaged phase vs channel for `bl_plot`
(e.g. ("AA","KT")) before (left) and after (right) correction, for RR and LL.
By default this shows absolute phase so the dominant bandpass trend remains
visible; set `relative=true` to suppress scan-constant offsets.

For before/after comparison, the default `comparison_weights=:input` uses the
same pre-correction weights on both panels so gain-amplitude rescaling does not
change the apparent scatter purely through reweighting. Use
`comparison_weights=:native` to plot each panel with its own weights.
"""
function plot_phase_stability(data::UVData, corr::UVData, bl_plot;
    relative=false, comparison_weights=:input)
    a_idx = findfirst(==(bl_plot[1]), data.ant_names)
    b_idx = findfirst(==(bl_plot[2]), data.ant_names)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl_plot")

    bl_idx = findfirst(p -> p == (a_idx, b_idx), data.bl_pairs)
    isnothing(bl_idx) && error("Baseline $bl_plot not in data")
    target_code = data.unique_bls[bl_idx]
    int_inds = findall(c -> c == target_code, data.bl_codes)

    nscan = length(data.sc)
    scan_wheel = diagnostic_scan_colormap(nscan)

    pol_labels = ["RR", "LL"]
    pol_idx = [1, 4]

    ylabel = relative ? "phase relative to ref (rad)" : "absolute phase (rad)"
    plot_weights_before, plot_weights_after = diagnostic_weight_pair(data.weights, corr.weights;
        comparison_weights=comparison_weights)

    fig = Figure(size=(900, 600))
    for (row, (pi, lab)) in enumerate(zip(pol_idx, pol_labels))
        ax_b = Axis(fig[row, 1]; title="$(join(bl_plot,"-")) $lab  before",
            xlabel="channel", ylabel=ylabel)
        ax_a = Axis(fig[row, 2]; title="$(join(bl_plot,"-")) $lab  after",
            xlabel="channel", ylabel=ylabel)
        linkxaxes!(ax_b, ax_a)
        linkyaxes!(ax_b, ax_a)

        vis_before = @view data.vis[int_inds, pi, :]
        vis_after = @view corr.vis[int_inds, pi, :]
        w_before = @view plot_weights_before[int_inds, pi, :]
        w_after = @view plot_weights_after[int_inds, pi, :]
        scan_groups = data.scan_idx[int_inds]

        summary_before = phase_series(vis_before, w_before; relative=relative)
        summary_after = phase_series(vis_after, w_after; relative=relative)
        lines!(ax_b, summary_before; color=:black, linewidth=2)
        lines!(ax_a, summary_after; color=:black, linewidth=2)

        annotate_coherence!(ax_b, residual_phase_coherence(vis_before, w_before; groups=scan_groups); fontsize=12)
        annotate_coherence!(ax_a, residual_phase_coherence(vis_after, w_after; groups=scan_groups); fontsize=12)

        for s in 1:nscan
            ii = int_inds[findall(i -> data.scan_idx[int_inds[i]] == s, eachindex(int_inds))]
            isempty(ii) && continue
            kw = (color=s, colormap=scan_wheel, colorrange=(1, max(nscan, 1)), markersize=9)
            before_phase = phase_series(data.vis[ii, pi, :], plot_weights_before[ii, pi, :]; relative=relative)
            after_phase = phase_series(corr.vis[ii, pi, :], plot_weights_after[ii, pi, :]; relative=relative)
            scatter!(ax_b, before_phase; kw...)
            scatter!(ax_a, after_phase; kw...)
        end
    end

    Colorbar(fig[1:2, 3], colormap=scan_wheel, limits=(1, max(nscan, 1)), label="Scan")
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

function phase_series(vis_block, weight_block; relative=true)
    phase = angle.(weighted_channel_average(vis_block, weight_block))
    return relative ? phase_relative_to_ref(phase) : phase
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

function residual_phase_coherence(vis_block, weight_block; groups=nothing)
    scan_coherences = Float64[]
    scan_losses = Float64[]
    scan_phase_rms = Float64[]

    if isnothing(groups)
        for s in axes(vis_block, 1)
            coherence, loss_percent, phase_rms_deg, nsamp = spectrum_phase_coherence(
                vec(vis_block[s, :]),
                vec(weight_block[s, :]))
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
                         pol_idx=[1, 4], pol_labels=["RR", "LL"],
                         comparison_weights=:input)

Build a baseline-by-baseline diagnostic table of expected coherence loss from
within-scan channel phase scatter before and after bandpass correction.
`avg` and `avg_corr` are scan-averaged `UVData` objects (returned by `scan_average`).
By default this uses the original input weights on both sides so the comparison
isolates phase improvement instead of gain-amplitude reweighting. Set
`comparison_weights=:native` to evaluate the corrected data with its updated weights.
"""
function coherence_loss_table(avg::UVData, avg_corr::UVData;
    pol_idx=[1, 4], pol_labels=["RR", "LL"], comparison_weights=:input)
    V = avg.vis
    W = avg.weights
    V_corr = avg_corr.vis
    _, W_corr = diagnostic_weight_pair(avg.weights, avg_corr.weights;
        comparison_weights=comparison_weights)
    rows = NamedTuple[]

    for bi in eachindex(avg.bl_pairs)
        a, b = avg.bl_pairs[bi]
        baseline = string(avg.ant_names[a], "-", avg.ant_names[b])

        for (pi, lab) in zip(pol_idx, pol_labels)
            before = residual_phase_coherence(view(V, :, bi, pi, :), view(W, :, bi, pi, :))
            after = residual_phase_coherence(view(V_corr, :, bi, pi, :), view(W_corr, :, bi, pi, :))
            nsamp = max(before[4], after[4])
            nsamp == 0 && continue

            push!(rows, (; baseline, pol=lab, nsamp,
                coherence_before=before[1], loss_before=before[2], phase_rms_before=before[3],
                coherence_after=after[1], loss_after=after[2], phase_rms_after=after[3],
                loss_improvement=before[2] - after[2],
                phase_rms_improvement=before[3] - after[3]))
        end
    end

    sort!(rows; by=row -> (isfinite(row.loss_improvement) ? -row.loss_improvement : Inf, row.baseline, row.pol))
    return rows
end

function print_coherence_loss_table(rows; io=stdout)
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

        println(io, rpad(row.baseline, 8), "  ",
            rpad(row.pol, 3), "  ",
            lpad(string(row.nsamp), 5), "  ",
            coh_before, "  ",
            loss_before, "  ",
            coh_after, "  ",
            loss_after, "  ",
            improve, "  ",
            rms_after)
    end
end

function baseline_in_data(data::UVData, bl_plot)
    a_idx = findfirst(==(bl_plot[1]), data.ant_names)
    b_idx = findfirst(==(bl_plot[2]), data.ant_names)
    (isnothing(a_idx) || isnothing(b_idx)) && return false
    return any(p -> p == (a_idx, b_idx), data.bl_pairs)
end

function choose_diagnostic_baseline(avg::UVData;
    preferred=[("AA", "AX"), ("AA", "NN"), ("AA", "PV"), ("KT", "PV")])
    for bl in preferred
        baseline_in_data(avg, bl) && return bl
    end

    W = avg.weights
    npol = size(W, 3)
    pols = npol >= 4 ? [1, npol] : collect(1:npol)
    best_score = -Inf
    best_bl = nothing

    for (bi, (a, b)) in enumerate(avg.bl_pairs)
        score = sum(@view W[:, bi, pols, :])
        if score > best_score
            best_score = score
            best_bl = (avg.ant_names[a], avg.ant_names[b])
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
function plot_baseline_phases(data::UVData, corr::UVData, bl_plot;
    relative=true, comparison_weights=:input)
    a_idx = findfirst(==(bl_plot[1]), data.ant_names)
    b_idx = findfirst(==(bl_plot[2]), data.ant_names)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl_plot")

    bl_idx = findfirst(p -> p == (a_idx, b_idx), data.bl_pairs)
    isnothing(bl_idx) && error("Baseline $bl_plot not in data")
    target_code = data.unique_bls[bl_idx]
    int_inds = findall(c -> c == target_code, data.bl_codes)

    nscan = length(data.sc)
    scan_wheel = diagnostic_scan_colormap(nscan)
    pol_labels = ["RR (R-R)", "LR (L-R)", "RL (R-L)", "LL (L-L)"]
    ylabel = relative ? "phase relative to ref (rad)" : "absolute phase (rad)"
    plot_weights_before, plot_weights_after = diagnostic_weight_pair(data.weights, corr.weights;
        comparison_weights=comparison_weights)

    fig = Figure(size=(1100, 900))
    for (row, (pi, lab)) in enumerate(zip(1:4, pol_labels))
        ax_b = Axis(fig[row, 1]; title="$(join(bl_plot, "-")) $lab before",
            xlabel="channel", ylabel=ylabel)
        ax_a = Axis(fig[row, 2]; title="$(join(bl_plot, "-")) $lab after",
            xlabel="channel", ylabel=ylabel)
        linkxaxes!(ax_b, ax_a)
        linkyaxes!(ax_b, ax_a)

        vis_before = @view data.vis[int_inds, pi, :]
        vis_after = @view corr.vis[int_inds, pi, :]
        w_before = @view plot_weights_before[int_inds, pi, :]
        w_after = @view plot_weights_after[int_inds, pi, :]
        scan_groups = data.scan_idx[int_inds]

        annotate_coherence!(ax_b, residual_phase_coherence(vis_before, w_before; groups=scan_groups))
        annotate_coherence!(ax_a, residual_phase_coherence(vis_after, w_after; groups=scan_groups))

        for s in 1:nscan
            ii = int_inds[findall(i -> data.scan_idx[int_inds[i]] == s, eachindex(int_inds))]
            isempty(ii) && continue
            kw = (color=s, colormap=scan_wheel, colorrange=(1, max(nscan, 1)), markersize=4)
            scatter!(ax_b, phase_series(data.vis[ii, pi, :], plot_weights_before[ii, pi, :]; relative=relative); kw...)
            scatter!(ax_a, phase_series(corr.vis[ii, pi, :], plot_weights_after[ii, pi, :]; relative=relative); kw...)
        end
    end

    Colorbar(fig[1:4, 3], colormap=scan_wheel, limits=(1, max(nscan, 1)), label="Scan")
    return fig
end

"""
    plot_gain_solutions(gains, data)

Grid of relative feed phases of the solved gains, one row per antenna and one
colour per scan.
"""
function plot_gain_solutions(gains, data::UVData)
    nscan = length(data.sc)
    nant = size(gains, 2)
    scan_wheel = diagnostic_scan_colormap(nscan)

    fig = Figure(size=(900, 180 * nant))
    for ai in 1:nant
        ax_phase1 = Axis(fig[ai, 1];
            ylabel=data.ant_names[ai], xlabel="channel",
            title=(ai == 1 ? "R/X gain  ∠g (rad)" : ""))

        ax_phase2 = Axis(fig[ai, 2];
            ylabel=data.ant_names[ai], xlabel="channel",
            title=(ai == 1 ? "L/Y gain  ∠g (rad)" : ""))
        linkxaxes!(ax_phase1, ax_phase2)
        linkyaxes!(ax_phase1, ax_phase2)
        for s in 1:nscan
            kw = (color=s, colormap=scan_wheel, colorrange=(1, max(nscan, 1)), markersize=4)
            scatter!(ax_phase1, phase_relative_to_ref(angle.(gains[s, ai, 1, :])); kw...)
            scatter!(ax_phase2, phase_relative_to_ref(angle.(gains[s, ai, 2, :])); kw...)
        end
    end
    Colorbar(fig[1:nant, 3], colormap=scan_wheel, limits=(1, max(nscan, 1)), label="Scan")
    return fig
end
