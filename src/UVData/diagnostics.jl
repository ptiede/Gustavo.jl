# Baseline-level diagnostic series computed by walking `UVSet` leaves.
#
# The series functions below consume `(Ti, Frequency)` blocks — a single
# (baseline, pol) selection concatenated across leaves — rather than a `UVSet`,
# so they compose with any source of visibility/weight matrices. Use
# `_baseline_scan_blocks` and `_concat_scan_blocks` to produce those blocks from
# a `UVSet` pair.

# Per-leaf scan windows, in branch insertion order.
function _scans(data::UVSet)
    out = Tuple{Float64, Float64}[]
    for (_, leaf) in branches(data)
        push!(out, scan_window(leaf))
    end
    return out
end

_antenna_names_v(data::UVSet) = union_antennas(data).name

# Union baseline pair list: every (a, b) that appears in any leaf.
function _baseline_pairs(data::UVSet)
    seen = Set{Tuple{Int, Int}}()
    out = Tuple{Int, Int}[]
    for (_, leaf) in branches(data)
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
function _local_baseline_idx(leaf::AbstractDimTree, bl::Tuple{<:AbstractString, <:AbstractString})
    bls = baselines(leaf)
    a, b = String(bl[1]), String(bl[2])
    for i in eachindex(bls.pairs)
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
    src_d = branches(data)
    src_c = branches(corr)
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

# ── Weighted channel statistics ─────────────────────────────────────────────

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

# ── Referencing ─────────────────────────────────────────────────────────────

"""
    phase_relative_to_ref(phases, ref_idx=1) -> Vector{Float64}

Wrap each finite entry of `phases` into `(-π, π]` relative to `phases[ref_idx]`.
Non-finite entries stay `NaN`. If `phases[ref_idx]` is not finite the first
finite entry is used instead; if none is, every entry is `NaN`.
"""
function phase_relative_to_ref(phases, ref_idx = 1)
    relative = fill(NaN, length(phases))
    (1 <= ref_idx <= length(phases)) || return relative

    ref = phases[ref_idx]
    if !isfinite(ref)
        ref_idx = findfirst(isfinite, phases)
        isnothing(ref_idx) && return relative
        ref = phases[ref_idx]
    end

    for i in eachindex(phases)
        isfinite(phases[i]) || continue
        relative[i] = angle(cis(phases[i] - ref))
    end
    return relative
end

"""
    amplitude_relative_to_ref(amps, ref_idx=1) -> Vector{Float64}

Divide each positive finite entry of `amps` by `amps[ref_idx]`. Entries that are
not finite and positive stay `NaN`. If `amps[ref_idx]` is not finite and
positive the first entry that is gets used instead; if none is, every entry is
`NaN`.
"""
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

# Index of the reference channel used to normalize amplitudes: `ref_idx` when
# it names a positive finite amplitude, otherwise the first channel that does.
function amplitude_reference_index(amps, ref_idx = 1)
    if 1 <= ref_idx <= length(amps)
        ref = amps[ref_idx]
        isfinite(ref) && ref > 0 && return ref_idx
    end
    return findfirst(a -> isfinite(a) && a > 0, amps)
end

# ── Phase and amplitude series ──────────────────────────────────────────────

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

# ── Polarization selection ──────────────────────────────────────────────────

function resolve_plot_polarizations(data::UVSet; pol = :parallel)
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

resolve_single_polarization(data::UVSet, pol::Integer) = Int(pol)
function resolve_single_polarization(data::UVSet, pol::AbstractString)
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

# ── Axis and track helpers ──────────────────────────────────────────────────

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

# The single track shared by every entry of `tracks`, or `nothing` if they
# disagree in value or in which entries are finite.
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

# ── Within-scan phase coherence ─────────────────────────────────────────────

# Weighted vector average of the unit phasors of one spectrum. Returns
# `(coherence, loss_percent, phase_rms_deg, nvalid)`.
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

"""
    residual_phase_coherence(vis_block, weight_block; groups=nothing)

Estimate the coherence factor expected from channel-to-channel phase scatter
within each scan, then average the scan-level losses.

Returns `(coherence, loss_percent, phase_rms_deg, nscan_valid)` where each
metric is first computed on a per-scan spectrum and then averaged over scans.
The optional `groups` argument lets raw integrations be grouped by scan before
forming the per-scan spectra.
"""
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

# Two-line plot annotation summarizing a `residual_phase_coherence` result.
function coherence_label(stats)
    _, loss_percent, phase_rms_deg, nscan = stats
    nscan == 0 && return "no valid scans"
    loss_text = isfinite(loss_percent) ? @sprintf("scan loss %.2f%%", loss_percent) : "scan loss n/a"
    rms_text = isfinite(phase_rms_deg) ? @sprintf("scan rms %.1f deg", phase_rms_deg) : "scan rms n/a"
    return string(loss_text, "\n", rms_text)
end

# ── Gain-track selection ────────────────────────────────────────────────────

function resolve_gain_polarizations(data::UVSet; pol = :all)
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

resolve_single_gain_polarization(::UVSet, pol::Integer) = Int(pol)
function resolve_single_gain_polarization(::UVSet, pol::AbstractString)
    pol in ("11", "Pol 1") && return 1
    pol in ("22", "Pol 2") && return 2
    error("Unsupported gain polarization label: $pol; use \"11\" (POLA) or \"22\" (POLB)")
end

function resolve_gain_sites(data::UVSet; sites = :all)
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

resolve_single_gain_site(data::UVSet, site::Integer) = Int(site)
function resolve_single_gain_site(data::UVSet, site::AbstractString)
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

# ── Plot entry points — implemented by `GustavoMakieExt` ────────────────────
# Load Makie or CairoMakie to enable plotting.

"""
    plot_stability(data, corr, bl_plot; quantity, pol, relative, comparison_weights)

Per-leaf time-stability scatter for a baseline. Provided by `GustavoMakieExt`.
"""
function plot_stability end

"""
    plot_baseline_phases(data, corr, bl_plot; relative, comparison_weights)

Per-leaf phase-vs-channel scatter for a baseline. Provided by `GustavoMakieExt`.
"""
function plot_baseline_phases end

"""
    plot_gain_solutions(gains, data; quantity, pol, sites, relative)

Grid of solved gain tracks. Provided by `GustavoMakieExt`.
"""
function plot_gain_solutions end

"""
    stability_plotting_config(quantity; relative)

Axis labels and annotation callback for `plot_stability`. Provided by
`GustavoMakieExt`, whose amplitude branch annotates with Makie's `text!`.
"""
function stability_plotting_config end

"""
    plot_noise_segments!(ax, series, noise, scan_index, scan_wheel, nscan; ...)

Draw per-channel error bars onto `ax`. Provided by `GustavoMakieExt`.
"""
function plot_noise_segments! end

"""
    annotate_coherence!(ax, stats; fontsize)

Annotate `ax` with `coherence_label(stats)`. Provided by `GustavoMakieExt`.
"""
function annotate_coherence! end

"""
    diagnostic_scan_colormap(nscan)

Categorical scan colormap for plot helpers. Provided by `GustavoMakieExt`.
"""
function diagnostic_scan_colormap end
