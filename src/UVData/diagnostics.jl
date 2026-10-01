# Baseline-level diagnostic series. They consume `(Ti, Frequency)` blocks — a
# single (baseline, product) selection — so they compose with any source of
# visibility/weight matrices.

# ── Weighted channel statistics ─────────────────────────────────────────────

function weighted_channel_average(vis_block, weight_block)
    avg = similar(vis_block, ComplexF64, (axes(vis_block, 2),))
    for c in axes(vis_block, 2)
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
    sums = fill!(similar(weight_block, Float64, (axes(weight_block, 2),)), 0.0)
    for c in axes(weight_block, 2)
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
    relative = fill!(similar(phases, Float64), NaN)
    (ref_idx in eachindex(phases)) || return relative

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
    relative = fill!(similar(amps, Float64), NaN)
    (ref_idx in eachindex(amps)) || return relative

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
    if ref_idx in eachindex(amps)
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
        for c in eachindex(spectrum, spectrum_weights)
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
