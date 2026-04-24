function collect_parallel_hand_rows(Vblock, Wblock, pol, c0, c)
    D = ComplexF64[]
    row_weights = Float64[]
    rows = Int[]

    if ndims(Vblock) == 3
        for bi in axes(Vblock, 1)
            v_c = Vblock[bi, pol, c]
            v_c0 = Vblock[bi, pol, c0]
            w_c = Wblock[bi, pol, c]
            w_c0 = Wblock[bi, pol, c0]
            (w_c > 0 && w_c0 > 0 && isfinite(real(v_c)) && isfinite(real(v_c0))) || continue
            push!(D, v_c / v_c0)
            push!(row_weights, sqrt(w_c * w_c0))
            push!(rows, bi)
        end
    elseif ndims(Vblock) == 4
        for s in axes(Vblock, 1), bi in axes(Vblock, 2)
            v_c = Vblock[s, bi, pol, c]
            v_c0 = Vblock[s, bi, pol, c0]
            w_c = Wblock[s, bi, pol, c]
            w_c0 = Wblock[s, bi, pol, c0]
            (w_c > 0 && w_c0 > 0 && isfinite(real(v_c)) && isfinite(real(v_c0))) || continue
            push!(D, v_c / v_c0)
            push!(row_weights, sqrt(w_c * w_c0))
            push!(rows, bi)
        end
    else
        error("Unsupported visibility block rank: $(ndims(Vblock))")
    end

    return D, row_weights, rows
end

function solve_parallel_channel!(gains, solved, Vblock, Wblock, bl_pairs, nant, phase_ref_ant, c0, c, A_amp, A_phase,
        station_models, parallel_pols; min_baselines=3)
    for (pol, feed) in zip(parallel_pols, (1, 2))
        D, row_weights, rows = collect_parallel_hand_rows(Vblock, Wblock, pol, c0, c)
        length(rows) < min_baselines && continue

        active = sort(unique(vcat([[bl_pairs[bi][1], bl_pairs[bi][2]] for bi in rows]...)))
        conn = zeros(Int, nant)
        for bi in rows
            a, b = bl_pairs[bi]
            conn[a] += 1
            conn[b] += 1
        end
        local_ref = choose_local_phase_reference(active, phase_ref_ant, station_models, conn, feed)
        active_free = filter(≠(local_ref), active)
        isempty(active_free) && continue

        log_amp_active = weighted_least_squares(
            A_amp[rows, active],
            log.(abs.(D)),
            row_weights)

        φ_free = weighted_least_squares(
            A_phase[rows, active_free],
            angle.(D),
            row_weights)

        log_amp = zeros(nant)
        log_amp[active] = log_amp_active
        φ = zeros(nant)
        φ[active_free] = φ_free

        gains[:, feed, c] = exp.(log_amp) .* cis.(φ)
        isnothing(solved) || (solved[active, feed, c] .= true)
    end

    return gains
end

function antenna_phase_weights(Wblock, bl_pairs, nant, pol)
    nchan = size(Wblock, ndims(Wblock))
    channel_weights = zeros(Float64, nant, nchan)

    if ndims(Wblock) == 4
        for s in axes(Wblock, 1), bi in axes(Wblock, 2), c in axes(Wblock, 4)
            w = Wblock[s, bi, pol, c]
            (w > 0 && isfinite(w)) || continue
            a, b = bl_pairs[bi]
            channel_weights[a, c] += w
            channel_weights[b, c] += w
        end
    elseif ndims(Wblock) == 3
        for bi in axes(Wblock, 1), c in axes(Wblock, 3)
            w = Wblock[bi, pol, c]
            (w > 0 && isfinite(w)) || continue
            a, b = bl_pairs[bi]
            channel_weights[a, c] += w
            channel_weights[b, c] += w
        end
    else
        error("Unsupported weight block rank: $(ndims(Wblock))")
    end

    return channel_weights
end

function amplitude_support_weights(Wblock, bl_pairs, nant, parallel_pols)
    reference_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[1])
    partner_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[2])
    support = zeros(Float64, nant, 2, size(reference_weights, 2))
    support[:, 1, :] .= reference_weights
    support[:, 2, :] .= partner_weights
    return support
end

function phase_block_design_matrix(nchan, c0, phase_block_size, valid)
    phase_block_size <= 1 && return zeros(Float64, nchan, 0)

    nblocks = cld(nchan, phase_block_size)
    columns = Vector{Vector{Float64}}()

    for block in 1:nblocks
        lo = (block - 1) * phase_block_size + 1
        hi = min(block * phase_block_size, nchan)
        inds = lo:hi
        any(valid[inds]) || continue

        column = zeros(Float64, nchan)
        column[inds] .= 1.0
        push!(columns, column)
    end

    isempty(columns) && return zeros(Float64, nchan, 0)
    return hcat(columns...)
end

frequency_segments(::GlobalFrequencySegmentation, nchan) = [1:nchan]

function frequency_segments(segmentation::BlockFrequencySegmentation, nchan)
    return [((block - 1) * segmentation.block_size + 1):min(block * segmentation.block_size, nchan)
        for block in 1:cld(nchan, segmentation.block_size)]
end

function model_basis_columns(::PerChannelBandpassModel, x, x_scaled, segment)
    return [Float64[i == channel ? 1.0 : 0.0 for i in eachindex(x)] for channel in segment]
end

function model_basis_columns(::FlatBandpassModel, x, x_scaled, segment)
    return [Float64[i in segment ? 1.0 : 0.0 for i in eachindex(x)]]
end

function model_basis_columns(::DelayBandpassModel, x, x_scaled, segment)
    return [Float64[i in segment ? x[i] : 0.0 for i in eachindex(x)]]
end

function model_basis_columns(model::PolynomialBandpassModel, x, x_scaled, segment)
    return [Float64[i in segment ? x_scaled[i]^degree : 0.0 for i in eachindex(x)] for degree in 1:model.degree]
end

function segment_design_coordinate(::GlobalFrequencySegmentation, x, segment, valid)
    scale = maximum(abs.(x[valid]))
    return scale > 0 ? x ./ scale : zeros(length(x))
end

function segment_design_coordinate(::BlockFrequencySegmentation, x, segment, valid)
    x_local = zeros(length(x))
    valid_segment = segment[valid[segment]]
    isempty(valid_segment) && return x_local

    center = mean(x[valid_segment])
    x_centered = x[segment] .- center
    scale = maximum(abs.(x_centered[valid[segment]]))
    x_local[segment] .= scale > 0 ? x_centered ./ scale : 0.0
    return x_local
end

function independent_segment_columns(columns, valid)
    independent = Vector{Vector{Float64}}()
    current_rank = 0

    for column in columns
        any(!iszero, column[valid]) || continue

        candidate = isempty(independent) ? hcat(column) : hcat(independent..., column)
        candidate_rank = rank(candidate[valid, :])
        candidate_rank > current_rank || continue

        push!(independent, column)
        current_rank = candidate_rank
    end

    return independent
end

function component_design_columns(component::SegmentedBandpassModel, x, valid)
    segments = frequency_segments(component.segmentation, length(x))
    columns = Vector{Vector{Float64}}()
    for segment in segments
        any(valid[segment]) || continue
        x_segment = segment_design_coordinate(component.segmentation, x, segment, valid)
        segment_columns = model_basis_columns(component.model, x_segment, x_segment, segment)
        append!(columns, independent_segment_columns(segment_columns, valid))
    end
    return columns
end

function fit_phase_model(phase_track, channel_weights, channel_freqs, c0, phase_model::AbstractBandpassModel,
    default_segmentation::AbstractFrequencySegmentation)
    components = model_components(phase_model, default_segmentation)
    length(components) == 1 && components[1].model isa PerChannelBandpassModel && return phase_track

    phase_unwrapped = unwrap_phase_track(phase_track, c0)

    x = 2π .* (channel_freqs .- channel_freqs[c0])
    valid = (channel_weights .> 0) .& isfinite.(channel_weights) .& isfinite.(phase_unwrapped) .& isfinite.(x)
    count(valid) >= 2 || return phase_track

    basis = Vector{Vector{Float64}}()
    for component in components
        append!(basis, component_design_columns(component, x, valid))
    end

    if isempty(basis)
        fitted = zeros(length(phase_track))
        fitted[c0] = 0.0
        return fitted
    end

    A = hcat(basis...)
    count(valid) >= size(A, 2) || return phase_track
    coeffs = weighted_least_squares(A[valid, :], phase_unwrapped[valid], channel_weights[valid])
    fitted = A * coeffs
    fitted .-= fitted[c0]
    fitted[c0] = 0.0
    return fitted
end

function fit_amplitude_model(log_amp_track, channel_weights, channel_freqs, c0, ::PerChannelBandpassModel,
    default_segmentation::AbstractFrequencySegmentation)
    return log_amp_track
end

function fit_amplitude_model(log_amp_track, channel_weights, channel_freqs, c0, amp_model::AbstractBandpassModel,
    default_segmentation::AbstractFrequencySegmentation)
    components = model_components(amp_model, default_segmentation)
    length(components) == 1 && components[1].model isa PerChannelBandpassModel && return log_amp_track

    x = channel_freqs .- channel_freqs[c0]
    valid = (channel_weights .> 0) .& isfinite.(channel_weights) .& isfinite.(log_amp_track) .& isfinite.(x)
    count(valid) >= 1 || return log_amp_track

    basis = Vector{Vector{Float64}}()
    for component in components
        append!(basis, component_design_columns(component, x, valid))
    end

    isempty(basis) && return log_amp_track
    A = hcat(basis...)
    count(valid) >= size(A, 2) || return log_amp_track
    coeffs = weighted_least_squares(A[valid, :], log_amp_track[valid], channel_weights[valid])
    fitted = A * coeffs
    fitted[c0] = 0.0
    return fitted
end

function fit_relative_correction_track(raw_correction, channel_weights, channel_freqs, c0, station_model)
    fitted_correction = copy(raw_correction)
    fitted_correction[c0] = 1.0 + 0.0im

    relative_log_amp = log.(abs.(fitted_correction))
    relative_phase = angle.(fitted_correction)

    fitted_log_amp = station_model.relative.amplitude.model isa PerChannelBandpassModel ?
        relative_log_amp :
        fit_amplitude_model(
            relative_log_amp,
            channel_weights,
            channel_freqs,
            c0,
            station_model.relative.amplitude.model,
            station_model.relative.amplitude.segmentation.frequency)

    fitted_phase = station_model.relative.phase.model isa PerChannelBandpassModel ?
        relative_phase :
        fit_phase_model(
            relative_phase,
            channel_weights,
            channel_freqs,
            c0,
            station_model.relative.phase.model,
            station_model.relative.phase.segmentation.frequency)

    fitted = exp.(fitted_log_amp) .* cis.(fitted_phase)
    fitted[c0] = 1.0 + 0.0im
    return fitted
end

function replacement_amplitude_scale(amps, support, c, c0; neighbor_window=2)
    local_values = Float64[]
    lo = max(1, c - neighbor_window)
    hi = min(length(amps), c + neighbor_window)
    for j in lo:hi
        j == c && continue
        j == c0 && continue
        support[j] > 0 || continue
        amp = amps[j]
        isfinite(amp) && amp > 0 || continue
        push!(local_values, amp)
    end
    length(local_values) >= 2 && return median(local_values)

    global_values = Float64[]
    for j in eachindex(amps)
        j == c && continue
        j == c0 && continue
        support[j] > 0 || continue
        amp = amps[j]
        isfinite(amp) && amp > 0 || continue
        push!(global_values, amp)
    end
    isempty(global_values) && return nothing
    return median(global_values)
end

function sanitize_gain_amplitudes!(gains, support_weights, c0;
        collapse_fraction=0.05, min_gain_amplitude=1e-2, neighbor_window=2)
    amps = abs.(gains)
    repaired = NamedTuple[]

    for ant in axes(gains, 1), feed in axes(gains, 2), c in axes(gains, 3)
        c == c0 && continue
        support_weights[ant, feed, c] > 0 || continue

        amp = amps[ant, feed, c]
        local_scale = replacement_amplitude_scale(view(amps, ant, feed, :), view(support_weights, ant, feed, :), c, c0;
            neighbor_window=neighbor_window)
        isnothing(local_scale) && continue

        threshold = max(min_gain_amplitude, collapse_fraction * local_scale)
        if !(isfinite(amp) && amp >= threshold)
            gains[ant, feed, c] = local_scale * cis(angle(gains[ant, feed, c]))
            push!(repaired, (; ant, feed, channel=c, amplitude=amp, replacement=local_scale))
        end
    end

    return repaired
end

function constrain_gain_amplitudes!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols)
    nant = size(gains, 1)
    reference_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[1])
    partner_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[2])

    for ant in 1:nant
        model = station_models[ant]
        reference_feed = model.reference_feed
        partner_feed = partner_feed_index(model.reference_feed)

        abs_amp_model = model.reference.amplitude.model
        if !(abs_amp_model isa PerChannelBandpassModel)
            reference_log_amp = log.(abs.(gains[ant, reference_feed, :]))
            fitted_reference_log_amp = fit_amplitude_model(
                vec(reference_log_amp),
                vec(reference_weights[ant, :]),
                channel_freqs,
                c0,
                abs_amp_model,
                model.reference.amplitude.segmentation.frequency)
            gains[ant, reference_feed, :] = exp.(fitted_reference_log_amp) .* cis.(angle.(gains[ant, reference_feed, :]))
        end

        relative_amp_model = model.relative.amplitude.model
        if !(relative_amp_model isa PerChannelBandpassModel)
            ratio = gains[ant, partner_feed, :] ./ gains[ant, reference_feed, :]
            relative_log_amp = log.(abs.(ratio))
            relative_weights = sqrt.(reference_weights[ant, :] .* partner_weights[ant, :])
            fitted_relative_log_amp = fit_amplitude_model(
                vec(relative_log_amp),
                vec(relative_weights),
                channel_freqs,
                c0,
                relative_amp_model,
                model.relative.amplitude.segmentation.frequency)
            gains[ant, partner_feed, :] = abs.(gains[ant, reference_feed, :]) .* exp.(fitted_relative_log_amp) .* cis.(angle.(gains[ant, partner_feed, :]))
        end
    end

    return gains
end

function constrain_gain_phases!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols)
    nant = size(gains, 1)
    reference_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[1])
    partner_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[2])

    for ant in 1:nant
        model = station_models[ant]
        reference_feed = model.reference_feed
        partner_feed = partner_feed_index(model.reference_feed)

        reference_phase_model = model.reference.phase.model
        if !(reference_phase_model isa PerChannelBandpassModel)
            reference_phase_track = vec(angle.(gains[ant, reference_feed, :]))
            fitted_reference_phase = fit_phase_model(
                reference_phase_track,
                vec(reference_weights[ant, :]),
                channel_freqs,
                c0,
                reference_phase_model,
                model.reference.phase.segmentation.frequency)
            gains[ant, reference_feed, :] = abs.(gains[ant, reference_feed, :]) .* cis.(fitted_reference_phase)
        end

        relative_phase_model = model.relative.phase.model
        if !(relative_phase_model isa PerChannelBandpassModel)
            ratio = gains[ant, partner_feed, :] ./ gains[ant, reference_feed, :]
            relative_phase_track = vec(angle.(ratio))
            relative_weights = sqrt.(reference_weights[ant, :] .* partner_weights[ant, :])
            fitted_relative_phase = fit_phase_model(
                relative_phase_track,
                vec(relative_weights),
                channel_freqs,
                c0,
                relative_phase_model,
                model.relative.phase.segmentation.frequency)
            gains[ant, partner_feed, :] = abs.(gains[ant, partner_feed, :]) .* cis.(angle.(gains[ant, reference_feed, :]) .+ fitted_relative_phase)
        end
    end

    return gains
end

function constrain_gain_models!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols)
    constrain_gain_amplitudes!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols)
    constrain_gain_phases!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols)
    sanitize_gain_amplitudes!(gains, amplitude_support_weights(Wblock, bl_pairs, size(gains, 1), parallel_pols), c0)
    return gains
end

"""
    solve_ref_xy_correction(V, W, bl_pairs, gains, ref_ant, c0, channel_freqs, station_models; min_samples=2)

Estimate the differential feed correction for the reference antenna after
the parallel-hand solve. This uses cross-hand to parallel-hand ratios on
baselines touching `ref_ant`, so only one feed phase is held fixed overall.

The estimator first solves a raw per-channel complex feed-ratio track per scan,
then projects that track onto the configured relative amplitude and phase basis
for the reference station before applying it.
"""
function solve_ref_xy_correction(V, W, bl_pairs, gains, ref_ant, c0, channel_freqs, station_models, pol_codes; min_samples=2)
    nscan, nbl, npol, nchan = size(V)
    cross_pols = cross_hand_indices(pol_codes)
    isnothing(cross_pols) && return ones(ComplexF64, nscan, nchan)
    rr, ll = parallel_hand_indices(pol_codes)
    rl, lr = cross_pols.rl, cross_pols.lr

    xy_correction = ones(ComplexF64, nscan, nchan)
    ref_model = station_models[ref_ant]

    for s in 1:nscan
        raw_correction = ones(ComplexF64, nchan)
        channel_weights = zeros(Float64, nchan)

        for c in 1:nchan
            c == c0 && continue

            corrections = ComplexF64[]
            correction_weights = Float64[]

            for bi in 1:nbl
                a, b = bl_pairs[bi]
                ref_ant ∉ (a, b) && continue

                if a == ref_ant
                    estimators = [
                        (lr, rr,  W[s, bi, lr, c], W[s, bi, rr, c], W[s, bi, lr, c0], W[s, bi, rr, c0]),
                        (ll, rl,  W[s, bi, ll, c], W[s, bi, rl, c], W[s, bi, ll, c0], W[s, bi, rl, c0]),
                    ]
                    sign = 1.0
                else
                    estimators = [
                        (rl, rr,  W[s, bi, rl, c], W[s, bi, rr, c], W[s, bi, rl, c0], W[s, bi, rr, c0]),
                        (ll, lr,  W[s, bi, ll, c], W[s, bi, lr, c], W[s, bi, ll, c0], W[s, bi, lr, c0]),
                    ]
                    sign = -1.0
                end

                for (num_pol, den_pol, w_num_c, w_den_c, w_num_c0, w_den_c0) in estimators
                    min(w_num_c, w_den_c, w_num_c0, w_den_c0) > 0 || continue

                    num_c = corrected_visibility(V, gains, pol_codes, bi, a, b, num_pol, s, c)
                    den_c = corrected_visibility(V, gains, pol_codes, bi, a, b, den_pol, s, c)
                    num_c0 = corrected_visibility(V, gains, pol_codes, bi, a, b, num_pol, s, c0)
                    den_c0 = corrected_visibility(V, gains, pol_codes, bi, a, b, den_pol, s, c0)

                    vals = (num_c, den_c, num_c0, den_c0)
                    all(isfinite(real(v)) && isfinite(imag(v)) && abs(v) > 0 for v in vals) || continue

                    ratio = (num_c / den_c) / (num_c0 / den_c0)
                    push!(corrections, sign > 0 ? ratio : inv(ratio))
                    push!(correction_weights, sqrt(w_num_c * w_den_c * w_num_c0 * w_den_c0))
                end
            end

            length(corrections) < min_samples && continue
            correction = weighted_complex_correction(corrections, correction_weights)
            isnothing(correction) && continue
            raw_correction[c] = correction
            channel_weights[c] = sum(correction_weights)
        end

        xy_correction[s, :] .= fit_relative_correction_track(raw_correction, channel_weights, channel_freqs, c0, ref_model)
    end

    return xy_correction
end

function solve_bandpass_single_scan(Vs, Ws, bl_pairs, nant, phase_ref_ant, c0, channel_freqs, station_models, parallel_pols;
    min_baselines=3)
    nbl, npol, nchan = size(Vs)
    A_amp, A_phase = design_matrices(bl_pairs, nant)

    gains = ones(ComplexF64, nant, 2, nchan)
    solved = falses(nant, 2, nchan)

    for c in 1:nchan
        c == c0 && continue
        solve_parallel_channel!(gains, solved, Vs, Ws, bl_pairs, nant, phase_ref_ant, c0, c, A_amp, A_phase,
            station_models, parallel_pols; min_baselines=min_baselines)
    end

    constrain_gain_models!(gains, Ws, bl_pairs, channel_freqs, c0, station_models, parallel_pols)
    return gains, solved
end

function solve_bandpass_template(V, W, bl_pairs, nant, phase_ref_ant, c0, channel_freqs, station_models, parallel_pols;
    min_baselines=3)
    nscan, nbl, npol, nchan = size(V)
    A_amp, A_phase = design_matrices(bl_pairs, nant)

    gains = ones(ComplexF64, nant, 2, nchan)

    for c in 1:nchan
        c == c0 && continue
        solve_parallel_channel!(gains, nothing, V, W, bl_pairs, nant, phase_ref_ant, c0, c, A_amp, A_phase,
            station_models, parallel_pols; min_baselines=min_baselines)
    end

    constrain_gain_models!(gains, W, bl_pairs, channel_freqs, c0, station_models, parallel_pols)
    return gains
end

function merge_scan_gains!(gain_slice, scan_gains, solved, phase_variable_mask, amplitude_variable_mask)
    nant, nfeed, nchan = size(gain_slice)
    for a in 1:nant, feed in 1:nfeed, c in 1:nchan
        solved[a, feed, c] || continue

        amp = amplitude_variable_mask[a, feed] ? abs(scan_gains[a, feed, c]) : abs(gain_slice[a, feed, c])
        phase = phase_variable_mask[a, feed] ? angle(scan_gains[a, feed, c]) : angle(gain_slice[a, feed, c])
        gain_slice[a, feed, c] = amp * cis(phase)
    end

    return gain_slice
end

"""
    solve_bandpass(avg::UVData, ref_ant; min_baselines=3,
                   station_models=nothing, phase_ref_ant=ref_ant, apply_relative_correction=true)

Solve for per-antenna complex bandpass gains relative to a reference channel.
The solver uses a staged architecture: the two parallel-hand products (`11` and
`22`) are solved independently to build a stable backbone, then an optional
reference-antenna relative feed correction is recovered afterward from the
cross-hands (`12`/`21`).

Algorithm per (scan s, channel c, feed):
  1. Form ratios  D[a,b] = V[s,a,b,c] / V[s,a,b,c₀]   (source cancels)
  2. Amplitude:   log|g_a| + log|g_b| = log|D[a,b]|    (least squares)
  3. Phase:       φ_a    - φ_b        = ∠ D[a,b]        (least squares, ref_ant fixed)
  4. Optionally use `12`/`21` relative to `11`/`22` on baselines touching `ref_ant`
      to estimate the reference antenna's differential feed correction.
  5. g_a = exp(log|g_a|) · exp(iφ_a), with feed 2 rephased by the solved
      differential term where available.

Returns:
  gains                         – `DimArray` with scan/antenna/feed/channel axes
  c0                            – reference channel index (gains = 1 there)
  xy_correction                 – `DimArray` with scan/IF axes and metadata for the target site/pol
"""
function solve_bandpass(avg::UVData, ref_ant;
        min_baselines=3, station_models=nothing, phase_ref_ant=ref_ant,
        apply_relative_correction=true)
    V = avg.vis
    W = avg.weights
    nscan, nbl, npol, nchan = size(V)
    nant = length(avg.ant_names)
    bl_pairs = avg.bl_pairs
    channel_freqs = avg.channel_freqs

    c0 = best_ref_channel(avg)
    @info "Reference channel: $c0 of $nchan"

    if isnothing(station_models)
        station_models = [StationBandpassModel() for _ in 1:nant]
    else
        length(station_models) == nant || error("station_models length does not match antenna count")
        station_models = validate_station_bandpass_model.(station_models)
    end

    parallel_pols = parallel_hand_indices(avg.pol_codes)
    gains_template = solve_bandpass_template(V, W, bl_pairs, nant, phase_ref_ant, c0, channel_freqs, station_models, parallel_pols;
        min_baselines=min_baselines)
    gains = repeat(reshape(gains_template, 1, nant, 2, nchan), nscan, 1, 1, 1)
    phase_variable_mask = falses(nant, 2)
    amplitude_variable_mask = falses(nant, 2)
    for ant in 1:nant, feed in 1:2
        phase_variable_mask[ant, feed] = phase_is_per_scan(station_models[ant], feed)
        amplitude_variable_mask[ant, feed] = amplitude_is_per_scan(station_models[ant], feed)
    end

    for s in 1:nscan
        scan_gains, solved = solve_bandpass_single_scan(
            view(V, s, :, :, :),
            view(W, s, :, :, :),
            bl_pairs,
            nant,
            phase_ref_ant,
            c0,
            channel_freqs,
            station_models,
            parallel_pols,
            min_baselines=min_baselines)
        merge_scan_gains!(
            view(gains, s, :, :, :),
            scan_gains,
            solved,
            phase_variable_mask,
            amplitude_variable_mask)
    end

    xy_correction = ones(ComplexF64, nscan, nchan)
    ref_partner_feed = partner_feed_index(station_models[ref_ant].reference_feed)
    if apply_relative_correction
        xy_correction = solve_ref_xy_correction(V, W, bl_pairs, gains, ref_ant, c0, channel_freqs, station_models, avg.pol_codes)
        gains[:, ref_ant, ref_partner_feed, :] .*= xy_correction
    end

    wrapped_xy = wrap_xy_correction(
        xy_correction,
        avg,
        ref_ant;
        applies_to_pol=ref_partner_feed,
        reference_pol=station_models[ref_ant].reference_feed)

    return wrap_gain_solutions(gains, avg), c0, wrapped_xy
end
