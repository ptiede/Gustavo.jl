function log_visibility_precision(v, w)
    (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || return 0.0
    amp2 = abs2(v)
    (amp2 > 0 && isfinite(amp2)) || return 0.0
    return w * amp2
end

function log_visibility_variance(v, w)
    precision = log_visibility_precision(v, w)
    precision > 0 || return Inf
    return inv(precision)
end

function propagated_log_ratio_weight(v_num, w_num, v_den, w_den)
    variance = log_visibility_variance(v_num, w_num) + log_visibility_variance(v_den, w_den)
    isfinite(variance) || return 0.0
    variance > 0 || return 0.0

    # `weighted_least_squares` scales each row directly, so this needs the
    # inverse standard deviation of log(v_num / v_den).
    return inv(sqrt(variance))
end

function propagated_log_double_ratio_weight(v_num, w_num, v_den, w_den, v_num_ref, w_num_ref, v_den_ref, w_den_ref)
    variance = (
        log_visibility_variance(v_num, w_num) +
        log_visibility_variance(v_den, w_den) +
        log_visibility_variance(v_num_ref, w_num_ref) +
        log_visibility_variance(v_den_ref, w_den_ref)
    )
    isfinite(variance) || return 0.0
    variance > 0 || return 0.0
    return inv(sqrt(variance))
end

function collect_parallel_hand_rows(Vblock, Wblock, pol, c0, c, baseline_mask = nothing)
    D = ComplexF64[]
    row_weights = Float64[]
    rows = Int[]

    if ndims(Vblock) == 3
        for bi in axes(Vblock, 1)
            (isnothing(baseline_mask) || baseline_mask[bi]) || continue
            v_c = Vblock[bi, pol, c]
            v_c0 = Vblock[bi, pol, c0]
            w_c = Wblock[bi, pol, c]
            w_c0 = Wblock[bi, pol, c0]
            row_weight = propagated_log_ratio_weight(v_c, w_c, v_c0, w_c0)
            row_weight > 0 || continue
            push!(D, v_c / v_c0)
            push!(row_weights, row_weight)
            push!(rows, bi)
        end
    elseif ndims(Vblock) == 4
        for s in axes(Vblock, 1), bi in axes(Vblock, 2)
            (isnothing(baseline_mask) || baseline_mask[bi]) || continue
            v_c = Vblock[s, bi, pol, c]
            v_c0 = Vblock[s, bi, pol, c0]
            w_c = Wblock[s, bi, pol, c]
            w_c0 = Wblock[s, bi, pol, c0]
            row_weight = propagated_log_ratio_weight(v_c, w_c, v_c0, w_c0)
            row_weight > 0 || continue
            push!(D, v_c / v_c0)
            push!(row_weights, row_weight)
            push!(rows, bi)
        end
    else
        error("Unsupported visibility block rank: $(ndims(Vblock))")
    end

    return D, row_weights, rows
end


function solve_parallel_channel!(
        gains, solved, Vblock, Wblock, bl_pairs, nant, gauge, c0, c, A_amp, A_phase,
        station_models, parallel_pols; min_baselines = 3, parallel_hand_mask = nothing
    )
    for (pol, feed) in zip(parallel_pols, (1, 2))
        bl_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, feed])
        D, row_weights, rows = collect_parallel_hand_rows(Vblock, Wblock, pol, c0, c, bl_mask)
        length(rows) < min_baselines && continue

        active = sort(unique(vcat([[bl_pairs[bi][1], bl_pairs[bi][2]] for bi in rows]...)))
        conn = zeros(Int, nant)
        for bi in rows
            a, b = bl_pairs[bi]
            conn[a] += 1
            conn[b] += 1
        end
        local_ref = choose_local_phase_reference(active, gauge, station_models, conn, feed)
        active_free = filter(≠(local_ref), active)
        isempty(active_free) && continue

        log_amp_active = weighted_least_squares(
            A_amp[rows, active],
            log.(abs.(D)),
            row_weights
        )

        φ_free = weighted_least_squares(
            A_phase[rows, active_free],
            angle.(D),
            row_weights
        )

        log_amp = zeros(nant)
        log_amp[active] = log_amp_active
        φ = zeros(nant)
        φ[active_free] = φ_free

        gains[:, feed, c] = exp.(log_amp) .* cis.(φ)
        isnothing(solved) || (solved[active, feed, c] .= true)
    end
    return gains
end

function antenna_phase_weights(Wblock, bl_pairs, nant, pol, baseline_mask = nothing)
    nchan = size(Wblock, ndims(Wblock))
    channel_weights = zeros(Float64, nant, nchan)

    if ndims(Wblock) == 4
        for s in axes(Wblock, 1), bi in axes(Wblock, 2), c in axes(Wblock, 4)
            (isnothing(baseline_mask) || baseline_mask[bi]) || continue
            w = Wblock[s, bi, pol, c]
            (w > 0 && isfinite(w)) || continue
            a, b = bl_pairs[bi]
            channel_weights[a, c] += w
            channel_weights[b, c] += w
        end
    elseif ndims(Wblock) == 3
        for bi in axes(Wblock, 1), c in axes(Wblock, 3)
            (isnothing(baseline_mask) || baseline_mask[bi]) || continue
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

function amplitude_support_weights(Wblock, bl_pairs, nant, parallel_pols, parallel_hand_mask = nothing)
    ref_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 1])
    par_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 2])
    reference_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[1], ref_mask)
    partner_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[2], par_mask)
    support = zeros(Float64, nant, 2, size(reference_weights, 2))
    support[:, 1, :] .= reference_weights
    support[:, 2, :] .= partner_weights
    return support
end

abstract type AbstractBandpassGauge end

struct ZeroMeanBandpassGauge <: AbstractBandpassGauge end

struct ReferenceAntennaBandpassGauge <: AbstractBandpassGauge
    ref_ant::Int
end

ReferenceAntennaBandpassGauge(ref_ant::Integer) = ReferenceAntennaBandpassGauge(Int(ref_ant))

validate_bandpass_gauge(gauge::ZeroMeanBandpassGauge, nant) = gauge
function validate_bandpass_gauge(gauge::ReferenceAntennaBandpassGauge, nant)
    1 <= gauge.ref_ant <= nant || error("ReferenceAntennaBandpassGauge ref_ant=$(gauge.ref_ant) is out of bounds for $nant antennas")
    return gauge
end

function apply_zero_mean_bandpass_gauge_track!(track, weights, ref_idx)
    valid = (weights .> 0) .& isfinite.(weights) .& isfinite.(real.(track)) .& isfinite.(imag.(track))
    any(valid) || return track

    log_amp = log.(abs.(track))
    amp_offset = sum(weights[valid] .* log_amp[valid]) / sum(weights[valid])
    track ./= exp(amp_offset)

    phase_track = unwrap_phase_track(vec(angle.(track)), ref_idx)
    phase_offset = sum(weights[valid] .* phase_track[valid]) / sum(weights[valid])
    track .*= cis.(-phase_offset)
    return track
end

function antenna_feed_support_weights(Wblock, bl_pairs, pol_codes, nant)
    nchan = size(Wblock, ndims(Wblock))
    support = zeros(Float64, nant, 2, nchan)

    if ndims(Wblock) == 4
        for s in axes(Wblock, 1), bi in axes(Wblock, 2), pol in axes(Wblock, 3), c in axes(Wblock, 4)
            w = Wblock[s, bi, pol, c]
            (w > 0 && isfinite(w)) || continue
            a, b = bl_pairs[bi]
            fa, fb = stokes_feed_pair(pol_codes[pol])
            support[a, fa, c] += w
            support[b, fb, c] += w
        end
    elseif ndims(Wblock) == 3
        for bi in axes(Wblock, 1), pol in axes(Wblock, 2), c in axes(Wblock, 3)
            w = Wblock[bi, pol, c]
            (w > 0 && isfinite(w)) || continue
            a, b = bl_pairs[bi]
            fa, fb = stokes_feed_pair(pol_codes[pol])
            support[a, fa, c] += w
            support[b, fb, c] += w
        end
    else
        error("Unsupported weight block rank: $(ndims(Wblock))")
    end

    return support
end

function bandpass_track_gauge_factor(track, weights, ref_idx)
    valid = (weights .> 0) .& isfinite.(weights) .& isfinite.(real.(track)) .& isfinite.(imag.(track))
    any(valid) || return 1.0 + 0.0im

    log_amp = log.(abs.(track))
    amp_offset = sum(weights[valid] .* log_amp[valid]) / sum(weights[valid])
    phase_track = unwrap_phase_track(vec(angle.(track)), ref_idx)
    phase_offset = sum(weights[valid] .* phase_track[valid]) / sum(weights[valid])
    return exp(amp_offset) * cis(phase_offset)
end

function zero_mean_bandpass_gauge_factors(gains, support_weights, ref_idx)
    gamma = ones(ComplexF64, size(gains, 1), size(gains, 2))
    for ant in axes(gains, 1), feed in axes(gains, 2)
        gamma[ant, feed] = bandpass_track_gauge_factor(@view(gains[ant, feed, :]), vec(support_weights[ant, feed, :]), ref_idx)
    end
    return gamma
end

function reference_antenna_bandpass_gauge_factors(gains, support_weights, ref_idx, ref_ant)
    gamma = ones(ComplexF64, size(gains, 1), size(gains, 2))
    for feed in axes(gains, 2)
        factor = bandpass_track_gauge_factor(@view(gains[ref_ant, feed, :]), vec(support_weights[ref_ant, feed, :]), ref_idx)
        gamma[:, feed] .= factor
    end
    return gamma
end

function bandpass_gauge_factors(gains, support_weights, ref_idx, ::ZeroMeanBandpassGauge)
    return zero_mean_bandpass_gauge_factors(gains, support_weights, ref_idx)
end

function bandpass_gauge_factors(gains, support_weights, ref_idx, gauge::ReferenceAntennaBandpassGauge)
    return reference_antenna_bandpass_gauge_factors(gains, support_weights, ref_idx, gauge.ref_ant)
end

function apply_bandpass_gauge!(gains, support_weights, ref_idx, gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge())
    if ndims(gains) == 3
        gamma = bandpass_gauge_factors(gains, support_weights, ref_idx, gauge)
        for ant in axes(gains, 1), feed in axes(gains, 2)
            gains[ant, feed, :] ./= gamma[ant, feed]
        end
    elseif ndims(gains) == 4
        for scan in axes(gains, 1)
            apply_bandpass_gauge!(@view(gains[scan, :, :, :]), support_weights, ref_idx, gauge)
        end
    else
        error("Unsupported gain array rank: $(ndims(gains))")
    end

    return gains
end

function apply_zero_mean_bandpass_gauge!(gains, support_weights, ref_idx)
    return apply_bandpass_gauge!(gains, support_weights, ref_idx, ZeroMeanBandpassGauge())
end

function allocate_source_coherencies(Vblock)
    nscan = ndims(Vblock) == 4 ? size(Vblock, 1) : 1
    nbl = ndims(Vblock) == 4 ? size(Vblock, 2) : size(Vblock, 1)
    return ones(ComplexF64, nscan, nbl, 2, 2)
end

function solve_source_coherencies!(source, gains, Vblock, Wblock, bl_pairs, pol_codes)
    numer = zeros(ComplexF64, size(source))
    denom = zeros(Float64, size(source))

    if ndims(Vblock) == 4
        for s in axes(Vblock, 1), bi in axes(Vblock, 2), pol in axes(Vblock, 3), c in axes(Vblock, 4)
            v = Vblock[s, bi, pol, c]
            w = Wblock[s, bi, pol, c]
            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
            a, b = bl_pairs[bi]
            fa, fb = stokes_feed_pair(pol_codes[pol])
            if ndims(gains) == 4
                model = gains[s, a, fa, c] * conj(gains[s, b, fb, c])
            else
                model = gains[a, fa, c] * conj(gains[b, fb, c])
            end
            amp2 = abs2(model)
            (amp2 > 0 && isfinite(amp2)) || continue
            numer[s, bi, fa, fb] += w * v * conj(model)
            denom[s, bi, fa, fb] += w * amp2
        end
    elseif ndims(Vblock) == 3
        for bi in axes(Vblock, 1), pol in axes(Vblock, 2), c in axes(Vblock, 3)
            v = Vblock[bi, pol, c]
            w = Wblock[bi, pol, c]
            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
            a, b = bl_pairs[bi]
            fa, fb = stokes_feed_pair(pol_codes[pol])
            model = gains[a, fa, c] * conj(gains[b, fb, c])
            amp2 = abs2(model)
            (amp2 > 0 && isfinite(amp2)) || continue
            numer[1, bi, fa, fb] += w * v * conj(model)
            denom[1, bi, fa, fb] += w * amp2
        end
    else
        error("Unsupported visibility block rank: $(ndims(Vblock))")
    end

    fill!(source, 0.0 + 0.0im)
    for i in eachindex(source)
        denom[i] > 0 || continue
        source[i] = numer[i] / denom[i]
    end
    return source
end

function joint_bandpass_objective(gains, source, Vblock, Wblock, bl_pairs, pol_codes)
    objective = 0.0

    if ndims(Vblock) == 4
        for s in axes(Vblock, 1), bi in axes(Vblock, 2), pol in axes(Vblock, 3), c in axes(Vblock, 4)
            v = Vblock[s, bi, pol, c]
            w = Wblock[s, bi, pol, c]
            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
            a, b = bl_pairs[bi]
            fa, fb = stokes_feed_pair(pol_codes[pol])
            if ndims(gains) == 4
                model = gains[s, a, fa, c] * source[s, bi, fa, fb] * conj(gains[s, b, fb, c])
            else
                model = gains[a, fa, c] * source[s, bi, fa, fb] * conj(gains[b, fb, c])
            end
            residual = v - model
            objective += w * abs2(residual)
        end
    elseif ndims(Vblock) == 3
        for bi in axes(Vblock, 1), pol in axes(Vblock, 2), c in axes(Vblock, 3)
            v = Vblock[bi, pol, c]
            w = Wblock[bi, pol, c]
            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
            a, b = bl_pairs[bi]
            fa, fb = stokes_feed_pair(pol_codes[pol])
            model = gains[a, fa, c] * source[1, bi, fa, fb] * conj(gains[b, fb, c])
            residual = v - model
            objective += w * abs2(residual)
        end
    else
        error("Unsupported visibility block rank: $(ndims(Vblock))")
    end

    return objective
end

function weighted_visibility_count(Wblock)
    count = 0
    for w in Wblock
        (w > 0 && isfinite(w)) || continue
        count += 1
    end
    return count
end

function joint_bandpass_parameter_count(gains, source)
    gain_params = 2 * count(isfinite, gains)
    source_params = 2 * count(isfinite, source)
    return gain_params + source_params
end

function constrained_real_track_parameter_count(valid)
    n = count(valid)
    return n > 0 ? n - 1 : 0
end

function effective_gain_parameter_count(setup, state)
    template_params = 0
    for ant in axes(state.gains_template, 1), feed in axes(state.gains_template, 2)
        valid = isfinite.(view(state.gains_template, ant, feed, :))
        if !setup.amplitude_variable_mask[ant, feed]
            template_params += constrained_real_track_parameter_count(valid)
        end
        if !setup.phase_variable_mask[ant, feed]
            template_params += constrained_real_track_parameter_count(valid)
        end
    end

    scan_params = 0
    for s in axes(state.scan_gains, 1), ant in axes(state.scan_gains, 2), feed in axes(state.scan_gains, 3)
        valid = isfinite.(view(state.scan_gains, s, ant, feed, :)) .& view(state.scan_solved, s, ant, feed, :)
        if setup.amplitude_variable_mask[ant, feed]
            scan_params += constrained_real_track_parameter_count(valid)
        end
        if setup.phase_variable_mask[ant, feed]
            scan_params += constrained_real_track_parameter_count(valid)
        end
    end

    return template_params + scan_params
end

function observed_source_parameter_count(Vblock, Wblock, pol_codes)
    nscan = ndims(Vblock) == 4 ? size(Vblock, 1) : 1
    nbl = ndims(Vblock) == 4 ? size(Vblock, 2) : size(Vblock, 1)
    observed = falses(nscan, nbl, 2, 2)

    if ndims(Vblock) == 4
        for s in axes(Vblock, 1), bi in axes(Vblock, 2), pol in axes(Vblock, 3), c in axes(Vblock, 4)
            v = Vblock[s, bi, pol, c]
            w = Wblock[s, bi, pol, c]
            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
            fa, fb = stokes_feed_pair(pol_codes[pol])
            observed[s, bi, fa, fb] = true
        end
    elseif ndims(Vblock) == 3
        for bi in axes(Vblock, 1), pol in axes(Vblock, 2), c in axes(Vblock, 3)
            v = Vblock[bi, pol, c]
            w = Wblock[bi, pol, c]
            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
            fa, fb = stokes_feed_pair(pol_codes[pol])
            observed[1, bi, fa, fb] = true
        end
    else
        error("Unsupported visibility block rank: $(ndims(Vblock))")
    end

    return 2 * count(observed)
end

function apply_bandpass_gauge_with_source!(gains, source, support_weights, bl_pairs, ref_idx, gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge())
    ndims(gains) == 3 || error("Gauge/source refinement expects a rank-3 gain cube")

    gamma = bandpass_gauge_factors(gains, support_weights, ref_idx, gauge)
    for ant in axes(gains, 1), feed in axes(gains, 2)
        gains[ant, feed, :] ./= gamma[ant, feed]
    end

    for s in axes(source, 1), bi in axes(source, 2)
        a, b = bl_pairs[bi]
        left = Diagonal(vec(gamma[a, :]))
        right = Diagonal(conj.(vec(gamma[b, :])))
        source[s, bi, :, :] .= left * Matrix(@view(source[s, bi, :, :])) * right
    end

    return gains, source
end

function apply_zero_mean_bandpass_gauge_with_source!(gains, source, support_weights, bl_pairs, ref_idx)
    return apply_bandpass_gauge_with_source!(gains, source, support_weights, bl_pairs, ref_idx, ZeroMeanBandpassGauge())
end

function refine_joint_bandpass_als!(
        gains, solved, Vblock, Wblock, bl_pairs, pol_codes, c0;
        max_iterations = 8, tolerance = 1.0e-6,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge()
    )
    source = allocate_source_coherencies(Vblock)
    support_weights = antenna_feed_support_weights(Wblock, bl_pairs, pol_codes, size(gains, 1))
    solve_source_coherencies!(source, gains, Vblock, Wblock, bl_pairs, pol_codes)

    previous_objective = joint_bandpass_objective(gains, source, Vblock, Wblock, bl_pairs, pol_codes)

    for _ in 1:max_iterations
        for c in axes(gains, 3), ant in axes(gains, 1), feed in axes(gains, 2)
            numer = 0.0 + 0.0im
            denom = 0.0

            if ndims(Vblock) == 4
                for s in axes(Vblock, 1), bi in axes(Vblock, 2), pol in axes(Vblock, 3)
                    v = Vblock[s, bi, pol, c]
                    w = Wblock[s, bi, pol, c]
                    (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                    a, b = bl_pairs[bi]
                    fa, fb = stokes_feed_pair(pol_codes[pol])

                    if a == ant && fa == feed
                        coeff = source[s, bi, feed, fb] * conj(gains[b, fb, c])
                        amp2 = abs2(coeff)
                        amp2 > 0 || continue
                        numer += w * conj(coeff) * v
                        denom += w * amp2
                    end

                    if b == ant && fb == feed
                        coeff = conj(gains[a, fa, c] * source[s, bi, fa, feed])
                        amp2 = abs2(coeff)
                        amp2 > 0 || continue
                        numer += w * conj(coeff) * conj(v)
                        denom += w * amp2
                    end
                end
            elseif ndims(Vblock) == 3
                for bi in axes(Vblock, 1), pol in axes(Vblock, 2)
                    v = Vblock[bi, pol, c]
                    w = Wblock[bi, pol, c]
                    (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                    a, b = bl_pairs[bi]
                    fa, fb = stokes_feed_pair(pol_codes[pol])

                    if a == ant && fa == feed
                        coeff = source[1, bi, feed, fb] * conj(gains[b, fb, c])
                        amp2 = abs2(coeff)
                        amp2 > 0 || continue
                        numer += w * conj(coeff) * v
                        denom += w * amp2
                    end

                    if b == ant && fb == feed
                        coeff = conj(gains[a, fa, c] * source[1, bi, fa, feed])
                        amp2 = abs2(coeff)
                        amp2 > 0 || continue
                        numer += w * conj(coeff) * conj(v)
                        denom += w * amp2
                    end
                end
            else
                error("Unsupported visibility block rank: $(ndims(Vblock))")
            end

            denom > 0 || continue
            gains[ant, feed, c] = numer / denom
            isnothing(solved) || (solved[ant, feed, c] = true)
        end

        apply_bandpass_gauge_with_source!(gains, source, support_weights, bl_pairs, c0, gauge)
        solve_source_coherencies!(source, gains, Vblock, Wblock, bl_pairs, pol_codes)
        objective = joint_bandpass_objective(gains, source, Vblock, Wblock, bl_pairs, pol_codes)

        improvement = previous_objective - objective
        improvement <= 0 && break
        (improvement / max(previous_objective, eps(Float64))) <= tolerance && break
        previous_objective = objective
    end

    return gains, source
end


frequency_segments(::GlobalFrequencySegmentation, nchan) = [1:nchan]

function frequency_segments(segmentation::BlockFrequencySegmentation, nchan)
    return [
        ((block - 1) * segmentation.block_size + 1):min(block * segmentation.block_size, nchan)
            for block in 1:cld(nchan, segmentation.block_size)
    ]
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

function fit_phase_model(
        phase_track, channel_weights, channel_freqs, c0, phase_model::AbstractBandpassModel,
        default_segmentation::AbstractFrequencySegmentation
    )
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

function fit_amplitude_model(
        log_amp_track, channel_weights, channel_freqs, c0, ::PerChannelBandpassModel,
        default_segmentation::AbstractFrequencySegmentation
    )
    return log_amp_track
end

function fit_amplitude_model(
        log_amp_track, channel_weights, channel_freqs, c0, amp_model::AbstractBandpassModel,
        default_segmentation::AbstractFrequencySegmentation
    )
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
            station_model.relative.amplitude.segmentation.frequency
        )

    fitted_phase = station_model.relative.phase.model isa PerChannelBandpassModel ?
        relative_phase :
        fit_phase_model(
            relative_phase,
            channel_weights,
            channel_freqs,
            c0,
            station_model.relative.phase.model,
            station_model.relative.phase.segmentation.frequency
        )

    fitted = exp.(fitted_log_amp) .* cis.(fitted_phase)
    fitted[c0] = 1.0 + 0.0im
    return fitted
end

function replacement_amplitude_scale(amps, support, c, c0; neighbor_window = 2)
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

function sanitize_gain_amplitudes!(
        gains, support_weights, c0;
        collapse_fraction = 0.05, min_gain_amplitude = 1.0e-2, neighbor_window = 2
    )
    amps = abs.(gains)
    repaired = NamedTuple[]

    for ant in axes(gains, 1), feed in axes(gains, 2), c in axes(gains, 3)
        c == c0 && continue
        support_weights[ant, feed, c] > 0 || continue

        amp = amps[ant, feed, c]
        local_scale = replacement_amplitude_scale(
            view(amps, ant, feed, :), view(support_weights, ant, feed, :), c, c0;
            neighbor_window = neighbor_window
        )
        isnothing(local_scale) && continue

        threshold = max(min_gain_amplitude, collapse_fraction * local_scale)
        if !(isfinite(amp) && amp >= threshold)
            gains[ant, feed, c] = local_scale * cis(angle(gains[ant, feed, c]))
            push!(repaired, (; ant, feed, channel = c, amplitude = amp, replacement = local_scale))
        end
    end

    return repaired
end

function warn_sanitized_gain_amplitudes(repaired, ant_names = nothing; context = "")
    isempty(repaired) && return nothing

    grouped = Dict{Tuple{Int, Int}, Vector{NamedTuple}}()
    for repair in repaired
        key = (repair.ant, repair.feed)
        push!(get!(grouped, key, NamedTuple[]), repair)
    end

    details = String[]
    for ((ant, feed), entries) in sort!(collect(grouped); by = first)
        ant_label = isnothing(ant_names) ? string("ant", ant) : string(ant_names[ant])
        channels = sort(getfield.(entries, :channel))
        min_amp = minimum(getfield.(entries, :amplitude))
        max_replacement = maximum(getfield.(entries, :replacement))
        push!(
            details,
            string(
                ant_label, "/feed", feed,
                " channels=", join(channels, ","),
                " min_amp=", round(min_amp; digits = 4),
                " replacement<=", round(max_replacement; digits = 4)
            )
        )
    end

    prefix = isempty(context) ? "" : string(context, ": ")
    @warn string(prefix, "repaired collapsed gain amplitudes; likely data/support issue") repaired_count = length(repaired) details
    return nothing
end

function constrain_gain_amplitudes!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols, parallel_hand_mask = nothing)
    nant = size(gains, 1)
    ref_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 1])
    par_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 2])
    reference_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[1], ref_mask)
    partner_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[2], par_mask)

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
                model.reference.amplitude.segmentation.frequency
            )
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
                model.relative.amplitude.segmentation.frequency
            )
            gains[ant, partner_feed, :] = abs.(gains[ant, reference_feed, :]) .* exp.(fitted_relative_log_amp) .* cis.(angle.(gains[ant, partner_feed, :]))
        end
    end

    return gains
end

function constrain_gain_phases!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols, parallel_hand_mask = nothing)
    nant = size(gains, 1)
    ref_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 1])
    par_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 2])
    reference_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[1], ref_mask)
    partner_weights = antenna_phase_weights(Wblock, bl_pairs, nant, parallel_pols[2], par_mask)

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
                model.reference.phase.segmentation.frequency
            )
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
                model.relative.phase.segmentation.frequency
            )
            gains[ant, partner_feed, :] = abs.(gains[ant, partner_feed, :]) .* cis.(angle.(gains[ant, reference_feed, :]) .+ fitted_relative_phase)
        end
    end

    return gains
end

function constrain_gain_models!(
        gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols, parallel_hand_mask = nothing;
        ant_names = nothing, context = "",
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge()
    )
    constrain_gain_amplitudes!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols, parallel_hand_mask)
    constrain_gain_phases!(gains, Wblock, bl_pairs, channel_freqs, c0, station_models, parallel_pols, parallel_hand_mask)
    support_weights = amplitude_support_weights(Wblock, bl_pairs, size(gains, 1), parallel_pols, parallel_hand_mask)
    repaired = sanitize_gain_amplitudes!(gains, support_weights, c0)
    warn_sanitized_gain_amplitudes(repaired, ant_names; context = context)
    apply_bandpass_gauge!(gains, support_weights, c0, gauge)
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
function solve_ref_xy_correction(V, W, bl_pairs, gains, ref_ant, c0, channel_freqs, station_models, pol_codes; min_samples = 2)
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
                        (lr, rr, W[s, bi, lr, c], W[s, bi, rr, c], W[s, bi, lr, c0], W[s, bi, rr, c0]),
                        (ll, rl, W[s, bi, ll, c], W[s, bi, rl, c], W[s, bi, ll, c0], W[s, bi, rl, c0]),
                    ]
                    sign = 1.0
                else
                    estimators = [
                        (rl, rr, W[s, bi, rl, c], W[s, bi, rr, c], W[s, bi, rl, c0], W[s, bi, rr, c0]),
                        (ll, lr, W[s, bi, ll, c], W[s, bi, lr, c], W[s, bi, ll, c0], W[s, bi, lr, c0]),
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
                    push!(correction_weights, propagated_log_double_ratio_weight(
                        num_c, w_num_c, den_c, w_den_c, num_c0, w_num_c0, den_c0, w_den_c0
                    ))
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

function solve_bandpass_single_scan(
        Vs, Ws, bl_pairs, nant, c0, channel_freqs, station_models, pol_codes, parallel_pols;
        ant_names = nothing, context = "",
        min_baselines = 3, joint_als_iterations = 8, joint_als_tolerance = 1.0e-6,
        parallel_hand_mask = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge()
    )
    _, _, nchan = size(Vs)
    A_amp, A_phase = design_matrices(bl_pairs, nant)

    gains = ones(ComplexF64, nant, 2, nchan)
    solved = falses(nant, 2, nchan)

    for c in 1:nchan
        c == c0 && continue
        solve_parallel_channel!(
            gains, solved, Vs, Ws, bl_pairs, nant, gauge, c0, c, A_amp, A_phase,
            station_models, parallel_pols; min_baselines = min_baselines, parallel_hand_mask = parallel_hand_mask
        )
    end

    joint_als_iterations > 0 && refine_joint_bandpass_als!(
        gains, solved, Vs, Ws, bl_pairs, pol_codes, c0;
        max_iterations = joint_als_iterations, tolerance = joint_als_tolerance,
        gauge = gauge
    )

    constrain_gain_models!(
        gains, Ws, bl_pairs, channel_freqs, c0, station_models, parallel_pols, parallel_hand_mask;
        ant_names = ant_names, context = context,
        gauge = gauge
    )
    return gains, solved
end

function solve_bandpass_template(
        V, W, bl_pairs, nant, c0, channel_freqs, station_models, pol_codes, parallel_pols;
        ant_names = nothing, context = "template",
        min_baselines = 3, joint_als_iterations = 8, joint_als_tolerance = 1.0e-6,
        parallel_hand_mask = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge()
    )
    _, _, _, nchan = size(V)
    A_amp, A_phase = design_matrices(bl_pairs, nant)

    gains = ones(ComplexF64, nant, 2, nchan)

    for c in 1:nchan
        c == c0 && continue
        solve_parallel_channel!(
            gains, nothing, V, W, bl_pairs, nant, gauge, c0, c, A_amp, A_phase,
            station_models, parallel_pols; min_baselines = min_baselines, parallel_hand_mask = parallel_hand_mask
        )
    end

    joint_als_iterations > 0 && refine_joint_bandpass_als!(
        gains, nothing, V, W, bl_pairs, pol_codes, c0;
        max_iterations = joint_als_iterations, tolerance = joint_als_tolerance,
        gauge = gauge
    )

    constrain_gain_models!(
        gains, W, bl_pairs, channel_freqs, c0, station_models, parallel_pols, parallel_hand_mask;
        ant_names = ant_names, context = context,
        gauge = gauge
    )
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

struct BandpassSolverSetup{
        D <: UVData,
        B <: AbstractVector{<:Tuple{<:Integer, <:Integer}},
        F <: AbstractVector{<:Real},
        S <: AbstractVector{<:StationBandpassModel},
        C <: AbstractVector{<:Integer},
        G <: AbstractBandpassGauge,
}
    data::D
    ref_ant::Int
    gauge::G
    min_baselines::Int
    bl_pairs::B
    channel_freqs::F
    station_models::S
    parallel_pols::Tuple{Int, Int}
    parallel_hand_mask::BitMatrix
    pol_codes::C
    c0::Int
    phase_variable_mask::BitMatrix
    amplitude_variable_mask::BitMatrix
end

mutable struct BandpassSolverState
    gains_template::Array{ComplexF64, 3}
    scan_gains::Array{ComplexF64, 4}
    scan_solved::BitArray{4}
    gains::Array{ComplexF64, 4}
    template_source::Array{ComplexF64, 4}
    scan_sources::Array{ComplexF64, 4}
    template_objective::Float64
    scan_objectives::Vector{Float64}
    xy_correction::Array{ComplexF64, 2}
    als_iterations_completed::Int
end

struct BandpassSolverResult{G, X}
    gains::G
    c0::Int
    xy_correction::X
end

abstract type AbstractBandpassInitializer end

struct RatioBandpassInitializer <: AbstractBandpassInitializer end

struct RandomBandpassInitializer{R <: AbstractRNG} <: AbstractBandpassInitializer
    rng::R
    amplitude_sigma::Float64
    phase_sigma::Float64
    scan_perturbation::Float64
end

function RandomBandpassInitializer(;
        rng = default_rng(),
        amplitude_sigma = 0.05,
        phase_sigma = 0.2,
        scan_perturbation = 0.02
    )
    amplitude_sigma >= 0 || error("amplitude_sigma must be nonnegative")
    phase_sigma >= 0 || error("phase_sigma must be nonnegative")
    scan_perturbation >= 0 || error("scan_perturbation must be nonnegative")
    return RandomBandpassInitializer(rng, Float64(amplitude_sigma), Float64(phase_sigma), Float64(scan_perturbation))
end

abstract type AbstractBandpassRefinement end

struct BandpassALS <: AbstractBandpassRefinement
    iterations::Int
    tolerance::Float64
    refine_template::Bool
    refine_scans::Bool
end

function BandpassALS(; iterations = 1, tolerance = 1.0e-6, refine_template = true, refine_scans = true)
    iterations >= 0 || error("iterations must be nonnegative")
    tolerance >= 0 || error("tolerance must be nonnegative")
    return BandpassALS(Int(iterations), Float64(tolerance), Bool(refine_template), Bool(refine_scans))
end

function prepare_bandpass_solver(
        avg::UVData, ref_ant;
        min_baselines = 3, station_models = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge()
    )
    ndims(avg.vis) == 4 || error("prepare_bandpass_solver expects scan-averaged rank-4 visibilities")

    nant = length(avg.antennas)
    if isnothing(station_models)
        station_models = [StationBandpassModel() for _ in 1:nant]
    else
        length(station_models) == nant || error("station_models length does not match antenna count")
        station_models = validate_station_bandpass_model.(station_models)
    end

    gauge = validate_bandpass_gauge(gauge, nant)
    parallel_pols = parallel_hand_indices(avg.metadata.pol_codes)
    parallel_hand_mask = build_parallel_hand_mask(avg.antennas, avg.baselines.pairs)
    c0 = best_ref_channel(avg)
    phase_variable_mask = falses(nant, 2)
    amplitude_variable_mask = falses(nant, 2)
    for ant in 1:nant, feed in 1:2
        phase_variable_mask[ant, feed] = phase_is_per_scan(station_models[ant], feed)
        amplitude_variable_mask[ant, feed] = amplitude_is_per_scan(station_models[ant], feed)
    end

    return BandpassSolverSetup(
        avg,
        ref_ant,
        gauge,
        min_baselines,
        avg.baselines.pairs,
        avg.metadata.channel_freqs,
        station_models,
        parallel_pols,
        parallel_hand_mask,
        avg.metadata.pol_codes,
        c0,
        phase_variable_mask,
        amplitude_variable_mask,
    )
end

function update_state_sources_and_objectives!(setup::BandpassSolverSetup, state::BandpassSolverState)
    solve_source_coherencies!(state.template_source, state.gains_template, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_codes)
    state.template_objective = joint_bandpass_objective(
        state.gains_template, state.template_source, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_codes
    )

    for s in axes(state.scan_gains, 1)
        scan_source = allocate_source_coherencies(view(setup.data.vis, s, :, :, :))
        solve_source_coherencies!(
            scan_source,
            view(state.scan_gains, s, :, :, :),
            view(setup.data.vis, s, :, :, :),
            view(setup.data.weights, s, :, :, :),
            setup.bl_pairs,
            setup.pol_codes
        )
        state.scan_sources[s, :, :, :] .= scan_source[1, :, :, :]
        state.scan_objectives[s] = joint_bandpass_objective(
            view(state.scan_gains, s, :, :, :),
            scan_source,
            view(setup.data.vis, s, :, :, :),
            view(setup.data.weights, s, :, :, :),
            setup.bl_pairs,
            setup.pol_codes
        )
    end

    return state
end

function assemble_bandpass_state_gains!(
        merged_gains, gains_template, scan_gains, scan_solved, phase_variable_mask, amplitude_variable_mask
    )
    merged_gains .= reshape(gains_template, 1, size(gains_template, 1), size(gains_template, 2), size(gains_template, 3))
    for s in axes(scan_gains, 1)
        merge_scan_gains!(
            view(merged_gains, s, :, :, :),
            view(scan_gains, s, :, :, :),
            view(scan_solved, s, :, :, :),
            phase_variable_mask,
            amplitude_variable_mask
        )
    end
    return merged_gains
end

function build_bandpass_solver_state(
        setup::BandpassSolverSetup,
        gains_template,
        scan_gains,
        scan_solved
    )
    data = setup.data
    nscan, _, _, nchan = size(data.vis)
    nbl = length(setup.bl_pairs)

    gains = repeat(reshape(gains_template, 1, size(gains_template, 1), size(gains_template, 2), size(gains_template, 3)), nscan, 1, 1, 1)
    assemble_bandpass_state_gains!(
        gains, gains_template, scan_gains, scan_solved, setup.phase_variable_mask, setup.amplitude_variable_mask
    )

    state = BandpassSolverState(
        gains_template,
        scan_gains,
        scan_solved,
        gains,
        ones(ComplexF64, nscan, nbl, 2, 2),
        ones(ComplexF64, nscan, nbl, 2, 2),
        0.0,
        zeros(Float64, nscan),
        ones(ComplexF64, nscan, nchan),
        0,
    )
    return update_state_sources_and_objectives!(setup, state)
end

function initialize_bandpass_state(setup::BandpassSolverSetup, ::RatioBandpassInitializer)
    data = setup.data
    nscan, _, _, nchan = size(data.vis)
    nant = length(data.antennas)

    gains_template = solve_bandpass_template(
        data.vis,
        data.weights,
        setup.bl_pairs,
        nant,
        setup.c0,
        setup.channel_freqs,
        setup.station_models,
        setup.pol_codes,
        setup.parallel_pols;
        ant_names = data.antennas.name,
        min_baselines = setup.min_baselines,
        joint_als_iterations = 0,
        parallel_hand_mask = setup.parallel_hand_mask,
        gauge = setup.gauge,
    )

    scan_gains = ones(ComplexF64, nscan, nant, 2, nchan)
    scan_solved = falses(nscan, nant, 2, nchan)
    for s in 1:nscan
        gains_scan, solved = solve_bandpass_single_scan(
            view(data.vis, s, :, :, :),
            view(data.weights, s, :, :, :),
            setup.bl_pairs,
            nant,
            setup.c0,
            setup.channel_freqs,
            setup.station_models,
            setup.pol_codes,
            setup.parallel_pols;
            ant_names = data.antennas.name,
            context = string("scan ", s),
            min_baselines = setup.min_baselines,
            joint_als_iterations = 0,
            parallel_hand_mask = setup.parallel_hand_mask,
            gauge = setup.gauge,
        )
        scan_gains[s, :, :, :] .= gains_scan
        scan_solved[s, :, :, :] .= solved
    end

    return build_bandpass_solver_state(setup, gains_template, scan_gains, scan_solved)
end

function initialize_bandpass_state(setup::BandpassSolverSetup, initializer::RandomBandpassInitializer)
    data = setup.data
    nscan, _, _, nchan = size(data.vis)
    nant = length(data.antennas)
    rng = initializer.rng

    gains_template = exp.(initializer.amplitude_sigma .* randn(rng, nant, 2, nchan)) .* cis.(initializer.phase_sigma .* randn(rng, nant, 2, nchan))
    support_template = antenna_feed_support_weights(data.weights, setup.bl_pairs, setup.pol_codes, nant)
    apply_bandpass_gauge!(gains_template, support_template, setup.c0, setup.gauge)

    scan_gains = repeat(reshape(gains_template, 1, nant, 2, nchan), nscan, 1, 1, 1)
    if initializer.scan_perturbation > 0
        scan_gains .*= exp.(initializer.scan_perturbation .* randn(rng, nscan, nant, 2, nchan)) .* cis.(initializer.scan_perturbation .* randn(rng, nscan, nant, 2, nchan))
    end
    for s in 1:nscan
        support_scan = antenna_feed_support_weights(view(data.weights, s, :, :, :), setup.bl_pairs, setup.pol_codes, nant)
        apply_bandpass_gauge!(view(scan_gains, s, :, :, :), support_scan, setup.c0, setup.gauge)
    end
    scan_solved = trues(nscan, nant, 2, nchan)

    return build_bandpass_solver_state(setup, gains_template, scan_gains, scan_solved)
end

initialize_bandpass_state(setup::BandpassSolverSetup) = initialize_bandpass_state(setup, RatioBandpassInitializer())

bandpass_state_objective(state::BandpassSolverState) = state.template_objective + sum(state.scan_objectives)

function fit_bandpass_source_coherencies(setup::BandpassSolverSetup, gains = nothing)
    isnothing(gains) && error("gains must be provided")
    source = allocate_source_coherencies(setup.data.vis)
    solve_source_coherencies!(source, gains, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_codes)
    return source
end

function compute_bandpass_model_and_residuals(setup::BandpassSolverSetup, gains, source)
    V = setup.data.vis
    W = setup.data.weights
    model = fill(NaN + NaN * im, size(V))
    residual = similar(model)

    for s in axes(V, 1), bi in axes(V, 2), pol in axes(V, 3), c in axes(V, 4)
        v = V[s, bi, pol, c]
        w = W[s, bi, pol, c]
        (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue

        a, b = setup.bl_pairs[bi]
        fa, fb = stokes_feed_pair(setup.pol_codes[pol])
        m = gains[s, a, fa, c] * source[s, bi, fa, fb] * conj(gains[s, b, fb, c])
        model[s, bi, pol, c] = m
        residual[s, bi, pol, c] = v - m
    end

    return model, residual
end

function summarize_bandpass_residual_block(residual_block, weight_block)
    chi2 = 0.0
    sumw = 0.0
    sum_abs2 = 0.0
    normalized_abs = Float64[]
    nvis = 0

    for (r, w) in zip(residual_block, weight_block)
        (w > 0 && isfinite(w) && isfinite(real(r)) && isfinite(imag(r))) || continue
        ar2 = abs2(r)
        chi2 += w * ar2
        sumw += w
        sum_abs2 += ar2
        nvis += 1
        push!(normalized_abs, abs(r) * sqrt(w))
    end

    nvis == 0 && return nothing

    nreal = 2 * nvis
    chi2_per_visibility = chi2 / nvis
    chi2_per_real_component = chi2 / nreal
    return (
        nvis = nvis,
        nreal = nreal,
        chi2 = chi2,
        sum_weight = sumw,
        residual_rms = sqrt(sum_abs2 / nvis),
        weighted_residual_rms = sumw > 0 ? sqrt(chi2 / sumw) : NaN,
        normalized_residual_rms = sqrt(chi2_per_real_component),
        chi2_per_visibility = chi2_per_visibility,
        chi2_per_real_component = chi2_per_real_component,
        median_abs_normalized_residual = isempty(normalized_abs) ? NaN : median(normalized_abs),
        max_abs_normalized_residual = isempty(normalized_abs) ? NaN : maximum(normalized_abs),
    )
end

"""
    bandpass_residual_stats(setup, gains; by=:baseline)

Summarize weighted complex residuals for the provided merged bandpass gains.
With `by=:baseline`, rows are grouped by `(baseline, pol)` across all scans and
channels. With `by=:scan_baseline`, rows are grouped by `(scan, baseline, pol)`.
"""
function bandpass_residual_stats(setup::BandpassSolverSetup, gains; by = :baseline)
    source = fit_bandpass_source_coherencies(setup, gains)
    _, residual = compute_bandpass_model_and_residuals(setup, gains, source)
    rows = NamedTuple[]

    if by == :baseline
        for (bi, (a, b)) in enumerate(setup.bl_pairs), pol in eachindex(setup.pol_codes)
            stats = summarize_bandpass_residual_block(
                view(residual, :, bi, pol, :),
                view(setup.data.weights, :, bi, pol, :)
            )
            isnothing(stats) && continue
            push!(rows, merge((
                baseline = string(setup.data.antennas.name[a], "-", setup.data.antennas.name[b]),
                pol = setup.data.metadata.pol_labels[pol],
            ), stats))
        end
    elseif by == :scan_baseline
        for s in axes(setup.data.vis, 1), (bi, (a, b)) in enumerate(setup.bl_pairs), pol in eachindex(setup.pol_codes)
            stats = summarize_bandpass_residual_block(
                view(residual, s, bi, pol, :),
                view(setup.data.weights, s, bi, pol, :)
            )
            isnothing(stats) && continue
            push!(rows, merge((
                scan = s,
                baseline = string(setup.data.antennas.name[a], "-", setup.data.antennas.name[b]),
                pol = setup.data.metadata.pol_labels[pol],
            ), stats))
        end
    else
        error("Unsupported residual grouping $by. Use :baseline or :scan_baseline")
    end

    sort!(rows; by = row -> (isfinite(row.chi2_per_real_component) ? -row.chi2_per_real_component : Inf, row.baseline, row.pol))
    return rows
end

bandpass_residual_stats(setup::BandpassSolverSetup, state::BandpassSolverState; by = :baseline) =
    bandpass_residual_stats(setup, state.gains; by = by)

function print_bandpass_residual_stats(rows; io = stdout, limit = nothing)
    isempty(rows) && return println(io, "No residual statistics available")

    shown = isnothing(limit) ? rows : first(rows, min(limit, length(rows)))
    has_scan = haskey(first(shown), :scan)

    println(io)
    println(io, "Bandpass residual summary")
    if has_scan
        println(io, "scan  baseline  pol   nvis   chi2/real   norm_rms   med|r|sqrt(w)   max|r|sqrt(w)")
    else
        println(io, "baseline  pol   nvis   chi2/real   norm_rms   med|r|sqrt(w)   max|r|sqrt(w)")
    end

    for row in shown
        prefix = has_scan ? string(lpad(string(row.scan), 4), "  ") : ""
        println(
            io,
            prefix,
            rpad(row.baseline, 8), "  ",
            rpad(row.pol, 3), "  ",
            lpad(string(row.nvis), 5), "  ",
            lpad(@sprintf("%.3f", row.chi2_per_real_component), 10), "  ",
            lpad(@sprintf("%.3f", row.normalized_residual_rms), 8), "  ",
            lpad(@sprintf("%.3f", row.median_abs_normalized_residual), 15), "  ",
            lpad(@sprintf("%.3f", row.max_abs_normalized_residual), 15)
        )
    end
    return nothing
end

function bandpass_fit_stats(setup::BandpassSolverSetup, gains)
    merged_source = fit_bandpass_source_coherencies(setup, gains)

    nvis = weighted_visibility_count(setup.data.weights)
    nreal = 2 * nvis
    chi2 = joint_bandpass_objective(gains, merged_source, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_codes)

    return (
        chi2 = chi2,
        nvis = nvis,
        nreal = nreal,
        nparams = missing,
        dof = missing,
        chi2_per_visibility = chi2 / max(nvis, 1),
        chi2_per_real_component = chi2 / max(nreal, 1),
        reduced_chi2 = missing,
    )
end

function bandpass_fit_stats(setup::BandpassSolverSetup, state::BandpassSolverState)
    stats = bandpass_fit_stats(setup, state.gains)
    nparams = effective_gain_parameter_count(setup, state) + observed_source_parameter_count(
        setup.data.vis,
        setup.data.weights,
        setup.pol_codes
    )
    dof = max(stats.nreal - nparams, 1)
    return merge(stats, (
        nparams = nparams,
        dof = dof,
        reduced_chi2 = stats.chi2 / dof,
    ))
end

function refine_bandpass!(
        setup::BandpassSolverSetup,
        state::BandpassSolverState,
        refinement::BandpassALS = BandpassALS()
    )
    refinement.iterations > 0 || return state

    if refinement.refine_template
        refine_joint_bandpass_als!(
            state.gains_template, nothing, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_codes, setup.c0;
            max_iterations = refinement.iterations, tolerance = refinement.tolerance,
            gauge = setup.gauge
        )
    end

    if refinement.refine_scans
        for s in axes(state.scan_gains, 1)
            refine_joint_bandpass_als!(
                view(state.scan_gains, s, :, :, :),
                view(state.scan_solved, s, :, :, :),
                view(setup.data.vis, s, :, :, :),
                view(setup.data.weights, s, :, :, :),
                setup.bl_pairs,
                setup.pol_codes,
                setup.c0;
                max_iterations = refinement.iterations,
                tolerance = refinement.tolerance,
                gauge = setup.gauge
            )
        end
    end

    assemble_bandpass_state_gains!(
        state.gains,
        state.gains_template,
        state.scan_gains,
        state.scan_solved,
        setup.phase_variable_mask,
        setup.amplitude_variable_mask
    )
    state.xy_correction .= 1.0 + 0.0im
    state.als_iterations_completed += refinement.iterations
    return update_state_sources_and_objectives!(setup, state)
end


function finalize_bandpass_state(
        setup::BandpassSolverSetup,
        state::BandpassSolverState;
        apply_relative_correction = state.als_iterations_completed == 0,
        project_models = true
    )
    data = setup.data
    nscan = size(state.scan_gains, 1)
    nant = size(state.scan_gains, 2)
    nchan = size(state.scan_gains, 4)

    gains_template = copy(state.gains_template)
    scan_gains = copy(state.scan_gains)
    merged_gains = repeat(reshape(gains_template, 1, nant, 2, nchan), nscan, 1, 1, 1)

    if project_models
        constrain_gain_models!(
            gains_template,
            data.weights,
            setup.bl_pairs,
            setup.channel_freqs,
            setup.c0,
            setup.station_models,
            setup.parallel_pols,
            setup.parallel_hand_mask;
            ant_names = data.antennas.name,
            context = "template",
            gauge = setup.gauge
        )

        for s in 1:nscan
            constrain_gain_models!(
                view(scan_gains, s, :, :, :),
                view(data.weights, s, :, :, :),
                setup.bl_pairs,
                setup.channel_freqs,
                setup.c0,
                setup.station_models,
                setup.parallel_pols,
                setup.parallel_hand_mask;
                ant_names = data.antennas.name,
                context = string("scan ", s),
                gauge = setup.gauge
            )
        end
    end

    assemble_bandpass_state_gains!(
        merged_gains, gains_template, scan_gains, state.scan_solved, setup.phase_variable_mask, setup.amplitude_variable_mask
    )

    xy_correction = ones(ComplexF64, nscan, nchan)
    ref_partner_feed = partner_feed_index(setup.station_models[setup.ref_ant].reference_feed)
    if apply_relative_correction
        xy_correction = solve_ref_xy_correction(
            data.vis,
            data.weights,
            setup.bl_pairs,
            merged_gains,
            setup.ref_ant,
            setup.c0,
            setup.channel_freqs,
            setup.station_models,
            setup.pol_codes
        )
        merged_gains[:, setup.ref_ant, ref_partner_feed, :] .*= xy_correction
    end

    apply_bandpass_gauge!(
        merged_gains,
        amplitude_support_weights(data.weights, setup.bl_pairs, nant, setup.parallel_pols, setup.parallel_hand_mask),
        setup.c0,
        setup.gauge
    )

    return BandpassSolverResult(
        wrap_gain_solutions(merged_gains, data),
        setup.c0,
        wrap_xy_correction(
            xy_correction,
            data,
            setup.ref_ant;
            applies_to_pol = ref_partner_feed,
            reference_pol = setup.station_models[setup.ref_ant].reference_feed
        )
    )
end

"""
    solve_bandpass(avg::UVData, ref_ant; min_baselines=3,
                   station_models=nothing,
                   gauge=ZeroMeanBandpassGauge(),
                   apply_relative_correction=true, joint_als_iterations=8,
                   joint_als_tolerance=1e-6)

Solve for per-antenna complex bandpass gains using a staged procedure:
parallel-hand channel ratios provide the initialization, then an optional joint
complex ALS refinement fits all available hands with a per-scan/per-baseline
2×2 nuisance coherency that is held constant across channel. The final gains are
reported in the requested bandpass gauge across frequency.

Algorithm per (scan s, channel c, feed):
  1. Form ratios `D[a,b] = V[s,a,b,c] / V[s,a,b,c₀]` so the scan/baseline source term cancels.
  2. Amplitude: `log|g_a[c]/g_a[c₀]| + log|g_b[c]/g_b[c₀]| = log|D[a,b]|`.
  3. Phase: `φ_a[c]-φ_a[c₀] - (φ_b[c]-φ_b[c₀]) = ∠D[a,b]`, with one local antenna held fixed.
  4. If `joint_als_iterations > 0`, refine those gains jointly against all
     available hands by alternating between source-coherency and gain updates
     in the complex domain, while gauge-fixing only channel-constant per-track
     offsets.
  5. Fit/project the refined tracks onto the configured station models and
     apply the requested final gain gauge.
  6. If joint ALS is disabled, optionally use `12`/`21` relative to `11`/`22`
     on baselines touching `ref_ant` to estimate the reference antenna's
     differential feed correction.

Returns:
  gains                         – `DimArray` with scan/antenna/feed/channel axes
  c0                            – internal reference channel used for ratios, phase unwrapping, and relative-feed correction
  xy_correction                 – `DimArray` with scan/IF axes and metadata for the target site/pol
"""
function solve_bandpass(
        avg::UVData, ref_ant;
        min_baselines = 3, station_models = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge(),
        apply_relative_correction = true, joint_als_iterations = 8, joint_als_tolerance = 1.0e-6
    )
    setup = prepare_bandpass_solver(
        avg,
        ref_ant;
        min_baselines = min_baselines,
        station_models = station_models,
        gauge = gauge
    )
    @info "Reference channel: $(setup.c0) of $(length(setup.channel_freqs))"

    state = initialize_bandpass_state(setup)
    joint_als_iterations > 0 && refine_bandpass!(
        setup,
        state,
        BandpassALS(iterations = joint_als_iterations, tolerance = joint_als_tolerance)
    )

    result = finalize_bandpass_state(
        setup,
        state;
        apply_relative_correction = apply_relative_correction && joint_als_iterations <= 0
    )
    return result.gains, result.c0, result.xy_correction
end
