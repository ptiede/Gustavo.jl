# Plotting helpers consume either a `UVSet` (canonical, leaf-walking) or a
# `BandpassDataset` (per-scan-averaged cube the solver consumes). Accessors
# below return the same logical fields from either shape so internal helpers
# don't have to branch on type.
const _DataLike = Union{UVSet, BandpassDataset}

_scans(data::BandpassDataset) = data.scans
function _scans(data::UVSet)
    out = Tuple{Float64, Float64}[]
    for (_, leaf) in DimensionalData.branches(data)
        push!(out, UVData.scan_window(leaf))
    end
    return out
end
_antenna_names_v(data::BandpassDataset) = data.antennas.name
_antenna_names_v(data::UVSet) = UVData.union_antennas(data).name

# Build the union baseline pair list from a UVSet (every (a,b) that appears
# in any leaf). For BandpassDataset, the per-scan-averaged cube already
# carries the union as `data.baselines.pairs`.
_baseline_pairs(data::BandpassDataset) = data.baselines.pairs
function _baseline_pairs(data::UVSet)
    seen = Set{Tuple{Int, Int}}()
    out = Tuple{Int, Int}[]
    for (_, leaf) in DimensionalData.branches(data)
        for p in baselines(leaf).pairs
            p in seen && continue
            push!(seen, p)
            push!(out, p)
        end
    end
    return out
end

# Locate `bl` (e.g. ("AA", "AX")) within a single leaf's local baseline
# index. Returns `nothing` if the baseline is absent from that leaf.
function _local_baseline_idx(leaf::DimensionalData.AbstractDimTree, bl::Tuple{<:AbstractString, <:AbstractString})
    bls = baselines(leaf)
    a, b = String(bl[1]), String(bl[2])
    @inbounds for i in eachindex(bls.pairs)
        bls.ant1_names[i] == a && bls.ant2_names[i] == b && return i
    end
    return nothing
end

# Collect per-scan (leaf) blocks of (vis_before, vis_after, w_before, w_after)
# for one (baseline, pol) selection. Returns a Vector of NamedTuples ordered
# by branch insertion order. The `sid` field is the leaf's index in branch
# order — used as a stable per-block ordinal for plotting/grouping; downstream
# consumers should not assume it indexes into any global scan table.
# Leaves that don't carry the baseline are skipped.
function _baseline_scan_blocks(data::UVSet, corr::UVSet, bl_plot, pol_index::Integer)
    blocks = NamedTuple[]
    src_d = DimensionalData.branches(data)
    src_c = DimensionalData.branches(corr)
    for (sid, (k, leaf_d)) in enumerate(src_d)
        leaf_c = src_c[k]
        bi_d = _local_baseline_idx(leaf_d, bl_plot)
        isnothing(bi_d) && continue
        bi_c = _local_baseline_idx(leaf_c, bl_plot)
        isnothing(bi_c) && continue
        # Layout: (Frequency, Ti, Baseline, Pol). Slice to (Frequency, Ti)
        # for fixed (baseline, pol), then transpose to (Ti, Frequency) so
        # downstream concat yields (nrec, nchan).
        push!(
            blocks, (
                sid = sid,
                vis_b = copy(transpose(parent(leaf_d[:vis])[:, :, bi_d, pol_index])),
                vis_a = copy(transpose(parent(leaf_c[:vis])[:, :, bi_c, pol_index])),
                w_b = copy(transpose(parent(leaf_d[:weights])[:, :, bi_d, pol_index])),
                w_a = copy(transpose(parent(leaf_c[:weights])[:, :, bi_c, pol_index])),
            )
        )
    end
    return blocks
end

# vcat per-scan blocks into a single (nrec, nchan) matrix and a parallel
# `groups::Vector{Int}` of per-record scan ids.
function _concat_scan_blocks(blocks; field::Symbol)
    isempty(blocks) && return Matrix{ComplexF64}(undef, 0, 0), Int[]
    cols = [getproperty(b, field) for b in blocks]
    cat = vcat(cols...)
    groups = vcat([fill(b.sid, size(getproperty(b, field), 1)) for b in blocks]...)
    return cat, groups
end

function diagnostic_weight_pair(weights, weights_corr; comparison_weights = :input)
    if comparison_weights == :input
        return weights, weights
    elseif comparison_weights == :native
        return weights, weights_corr
    else
        error("comparison_weights must be :input or :native")
    end
end

# `diagnostic_scan_colormap`, `annotate_coherence!`, and `plot_stability`
# live in the `GustavoMakieExt` extension (load Makie or CairoMakie to
# enable plotting).

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

function resolve_plot_polarizations(data::_DataLike; pol = :parallel)
    pp = pol_products(data)
    if pol == :parallel
        pol_idx = collect(parallel_hand_indices(pp))
    elseif pol == :all
        pol_idx = collect(eachindex(pp))
    elseif pol isa Integer
        pol_idx = [Int(pol)]
    elseif pol isa AbstractString
        pol_idx = [resolve_single_polarization(data, pol)]
    elseif pol isa AbstractVector || pol isa Tuple
        pol_idx = Int[resolve_single_polarization(data, item) for item in pol]
    else
        error("Unsupported polarization selector: $pol")
    end

    all(1 .<= pol_idx .<= length(pp)) || error("Polarization index out of bounds: $pol_idx")
    return pol_idx, collect(pp[pol_idx])
end

resolve_single_polarization(data::_DataLike, pol::Integer) = Int(pol)
function resolve_single_polarization(data::_DataLike, pol::AbstractString)
    pp = pol_products(data)
    idx = findfirst(==(pol), pp)
    isnothing(idx) || return idx

    if pol in ("11", "22")
        p_idx, q_idx = parallel_hand_indices(pp)
        return pol == "11" ? p_idx : q_idx
    end
    if pol in ("12", "21")
        cross = cross_hand_indices(pp)
        isnothing(cross) && error("Cross-hand pol $pol not found in $(collect(pp))")
        return pol == "12" ? cross.pq : cross.qp
    end
    error("Polarization $pol not found in $(collect(pp))")
end

# `stability_plotting_config` lives in `GustavoMakieExt` because the
# amplitude branch's `annotate_metric!` closure calls Makie's `text!`.
function stability_plotting_config end

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

# `plot_noise_segments!` lives in the `GustavoMakieExt` extension.

function shared_track(tracks; atol = 1.0e-12, rtol = 1.0e-9)
    representative = nothing
    representative_valid = nothing

    for track in tracks
        valid = isfinite.(track)
        any(valid) || continue

        if isnothing(representative)
            representative = copy(track)
            representative_valid = valid
            continue
        end

        valid == representative_valid || return nothing
        all(isapprox.(track[valid], representative[valid]; atol = atol, rtol = rtol)) || return nothing
    end

    return representative
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
    coherence_loss_table(avg::_DataLike, avg_corr::_DataLike;
                         pol_idx=[parallel-hand indices], pol_labels=[parallel-hand labels],
                         comparison_weights=:input)

Build a baseline-by-baseline diagnostic table of expected coherence loss from
within-scan channel phase scatter before and after bandpass correction.
`avg` and `avg_corr` are scan-averaged `UVPartition` objects (returned by `scan_average`).
By default this uses the original input weights on both sides so the comparison
isolates phase improvement instead of gain-amplitude reweighting. Set
`comparison_weights=:native` to evaluate the corrected data with its updated weights.
"""
coherence_loss_table(avg::UVSet, avg_corr::UVSet; kwargs...) =
    coherence_loss_table(_to_bandpass_dataset(avg), _to_bandpass_dataset(avg_corr); kwargs...)

function coherence_loss_table(
        avg::_DataLike, avg_corr::_DataLike;
        pol_idx = nothing, pol_labels = nothing, comparison_weights = :input
    )
    V = avg.vis
    W = avg.weights
    V_corr = avg_corr.vis
    _, W_corr = diagnostic_weight_pair(
        avg.weights, avg_corr.weights;
        comparison_weights = comparison_weights
    )
    pp = pol_products(avg)
    if isnothing(pol_idx)
        p_idx, q_idx = parallel_hand_indices(pp)
        pol_idx = [p_idx, q_idx]
    end
    isnothing(pol_labels) && (pol_labels = pp[pol_idx])
    rows = NamedTuple[]

    for bi in eachindex(avg.baselines.pairs)
        a, b = avg.baselines.pairs[bi]
        baseline = string(avg.antennas.name[a], "-", avg.antennas.name[b])

        for (pi, lab) in zip(pol_idx, pol_labels)
            # BandpassDataset cubes are (Frequency, Ti, Baseline, Pol), while
            # `residual_phase_coherence` consumes (Ti, Frequency) blocks.
            before = residual_phase_coherence(
                transpose(view(V, Baseline=bi, Pol=pi)),
                transpose(view(W, Baseline=bi, Pol=pi)),
            )
            after = residual_phase_coherence(
                transpose(view(V_corr, Baseline=bi, Pol=pi)),
                transpose(view(W_corr, Baseline=bi, Pol=pi)),
            )
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

function baseline_in_data(data::_DataLike, bl_plot)
    ant_names = _antenna_names_v(data)
    a_idx = findfirst(==(bl_plot[1]), ant_names)
    b_idx = findfirst(==(bl_plot[2]), ant_names)
    (isnothing(a_idx) || isnothing(b_idx)) && return false
    return any(p -> p == (a_idx, b_idx), _baseline_pairs(data))
end

choose_diagnostic_baseline(avg::UVSet; kwargs...) =
    choose_diagnostic_baseline(_to_bandpass_dataset(avg); kwargs...)

function choose_diagnostic_baseline(
        avg::BandpassDataset;
        preferred = [("AA", "AX"), ("AA", "NN"), ("AA", "PV"), ("KT", "PV")]
    )
    for bl in preferred
        baseline_in_data(avg, bl) && return bl
    end

    W = avg.weights
    pols = collect(parallel_hand_indices(pol_products(avg)))
    best_score = -Inf
    best_bl = nothing

    for (bi, (a, b)) in enumerate(avg.baselines.pairs)
        # W layout: (Frequency, Ti, Baseline, Pol).
        score = sum(@view W[:, :, bi, pols])
        if score > best_score
            best_score = score
            best_bl = (avg.antennas.name[a], avg.antennas.name[b])
        end
    end

    isnothing(best_bl) && error("No populated baseline available for diagnostics")
    return best_bl
end

# `plot_baseline_phases` and `plot_gain_solutions` live in the
# `GustavoMakieExt` extension.

function baseline_bandpass_diagnostics(setup::BandpassSolverSetup, gains, bi, pi)
    # data.vis/weights: (Frequency, Ti, Baseline, Pol). gains:
    # (Frequency, Ti, Ant, Feed). Output arrays keep (Ti, Frequency)
    # layout for plotting friendliness.
    data = setup.data
    nchan, nti, _, _ = size(parent(data.vis))
    observed = fill(NaN + NaN * im, nti, nchan)
    observed_weights = zeros(Float64, nti, nchan)
    model = fill(NaN + NaN * im, nti, nchan)
    gain_product = fill(NaN + NaN * im, nti, nchan)
    source_per_scan = fill(NaN + NaN * im, nti, nchan)
    normalized_residual = fill(NaN + NaN * im, nti, nchan)
    weights = Array{Float64}(undef, nti, nchan)
    source = fit_bandpass_source_coherencies(setup, gains)

    a, b = setup.bl_pairs[bi]
    fa, fb = correlation_feed_pair(setup.pol_products[pi])
    for s in 1:nti
        src = source[s, bi, fa, fb]
        @inbounds for c in 1:nchan
            v = data.vis[c, s, bi, pi]
            w = data.weights[c, s, bi, pi]
            weights[s, c] = w
            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue

            gain_model = gains[c, s, a, fa] * conj(gains[c, s, b, fb])
            full_model = gain_model * src
            observed[s, c] = v
            observed_weights[s, c] = w
            model[s, c] = full_model
            gain_product[s, c] = gain_model
            source_per_scan[s, c] = src
            normalized_residual[s, c] = sqrt(w) * (v - full_model)
        end
    end

    return observed, observed_weights, model, normalized_residual, weights, gain_product, source_per_scan
end

baseline_bandpass_diagnostics(setup::BandpassSolverSetup, state::BandpassSolverState, bi, pi) =
    baseline_bandpass_diagnostics(setup, state.gains, bi, pi)

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


function resolve_gain_polarizations(data::_DataLike; pol = :all)
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

resolve_single_gain_polarization(::_DataLike, pol::Integer) = Int(pol)
function resolve_single_gain_polarization(::_DataLike, pol::AbstractString)
    pol in ("11", "Pol 1") && return 1
    pol in ("22", "Pol 2") && return 2
    error("Unsupported gain polarization label: $pol; use \"11\" (POLA) or \"22\" (POLB)")
end

function resolve_gain_sites(data::_DataLike; sites = :all)
    ant_names = _antenna_names_v(data)
    if sites == :all
        site_idx = collect(eachindex(ant_names))
    elseif sites isa Integer
        site_idx = [Int(sites)]
    elseif sites isa AbstractString
        site_idx = [resolve_single_gain_site(data, sites)]
    elseif sites isa AbstractVector || sites isa Tuple
        site_idx = Int[resolve_single_gain_site(data, site) for site in sites]
    else
        error("Unsupported gain site selector: $sites")
    end

    all(1 .<= site_idx .<= length(ant_names)) || error("Gain site index out of bounds: $site_idx")
    return site_idx, collect(ant_names[site_idx])
end

resolve_single_gain_site(data::_DataLike, site::Integer) = Int(site)
function resolve_single_gain_site(data::_DataLike, site::AbstractString)
    ant_names = _antenna_names_v(data)
    idx = findfirst(==(site), ant_names)
    isnothing(idx) && error("Site $site not found in $(collect(ant_names))")
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

function baseline_index(data::_DataLike, bl::Tuple{String, String})
    ant_names = _antenna_names_v(data)
    a_idx = findfirst(==(bl[1]), ant_names)
    b_idx = findfirst(==(bl[2]), ant_names)
    (isnothing(a_idx) || isnothing(b_idx)) && error("Antenna not found: $bl")

    pairs_v = _baseline_pairs(data)
    bl_idx = findfirst(==((a_idx, b_idx)), pairs_v)
    isnothing(bl_idx) && error("Baseline $bl not in data")
    return bl_idx
end

parallel_hand_support_summary(avg::UVSet, bl::Tuple{String, String}) =
    parallel_hand_support_summary(_to_bandpass_dataset(avg), bl)

function parallel_hand_support_summary(avg::BandpassDataset, bl::Tuple{String, String})
    bi = baseline_index(avg, bl)
    pp = pol_products(avg)
    p_idx, q_idx = parallel_hand_indices(pp)
    pol_map = [(pp[p_idx], p_idx), (pp[q_idx], q_idx)]
    rows = NamedTuple[]

    for (lab, pi) in pol_map
        # avg.weights/vis layout: (Frequency, Ti, Baseline, Pol). Slice to a
        # (Frequency, Ti) 2-D view; iterate Ti as outer "scan" axis.
        W = @view avg.weights[:, :, bi, pi]
        V = @view avg.vis[:, :, bi, pi]
        valid = (W .> 0) .& isfinite.(W) .& isfinite.(real.(V)) .& isfinite.(imag.(V))
        scan_valid = vec(sum(valid, dims = 1))   # per-Ti valid count (length nti)
        if_valid = vec(sum(valid, dims = 2))    # per-Frequency valid count (length nchan)
        total_weight = sum(W[valid])
        mean_phase_by_scan = [
            begin
                    idx = vec(valid[:, s])
                    any(idx) ? angle(sum(W[idx, s] .* (V[idx, s] ./ abs.(V[idx, s]))) / sum(W[idx, s])) : NaN
                end for s in axes(W, 2)
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

site_parallel_hand_support(avg::UVSet, site::String) =
    site_parallel_hand_support(_to_bandpass_dataset(avg), site)

function site_parallel_hand_support(avg::BandpassDataset, site::String)
    rows = NamedTuple[]
    for (bi, (a, b)) in enumerate(avg.baselines.pairs)
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
