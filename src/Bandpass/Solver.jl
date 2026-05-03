function log_visibility_precision(v, w)
    (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || return 0.0
    amp2 = abs2(v)
    (amp2 > 0 && isfinite(amp2)) || return zero(amp2)
    return w * amp2
end

function log_visibility_variance(v, w)
    precision = log_visibility_precision(v, w)
    precision > 0 || return convert(typeof(w), Inf)
    return inv(precision)
end

function propagated_log_ratio_weight(v_num, w_num, v_den, w_den)
    variance = log_visibility_variance(v_num, w_num) + log_visibility_variance(v_den, w_den)
    isfinite(variance) || return zero(variance)
    variance > 0 || return zero(variance)

    # Returns the inverse variance (precision) of log(v_num / v_den), matching
    # the inverse-variance convention used throughout Gustavo's WLS.
    return inv(variance)
end

function propagated_log_double_ratio_weight(v_num, w_num, v_den, w_den, v_num_ref, w_num_ref, v_den_ref, w_den_ref)
    variance = (
        log_visibility_variance(v_num, w_num) +
            log_visibility_variance(v_den, w_den) +
            log_visibility_variance(v_num_ref, w_num_ref) +
            log_visibility_variance(v_den_ref, w_den_ref)
    )
    isfinite(variance) || return zero(variance)
    variance > 0 || return zero(variance)
    return inv(variance)
end

function collect_parallel_hand_rows(Vblock, Wblock, pol, c0, c, baseline_mask = nothing)
    D = eltype(Vblock)[]
    row_weights = eltype(Wblock)[]
    rows = Int[]

    Vp = view(Vblock; Pol = pol)
    Wp = view(Wblock; Pol = pol)
    for bi in axes(Vp, Baseline)
        (isnothing(baseline_mask) || baseline_mask[bi]) || continue
        v_c_slice = view(Vp; Baseline = bi, Frequency = c)
        v_c0_slice = view(Vp; Baseline = bi, Frequency = c0)
        w_c_slice = view(Wp; Baseline = bi, Frequency = c)
        w_c0_slice = view(Wp; Baseline = bi, Frequency = c0)
        for (v_c, v_c0, w_c, w_c0) in zip(v_c_slice, v_c0_slice, w_c_slice, w_c0_slice)
            row_weight = propagated_log_ratio_weight(v_c, w_c, v_c0, w_c0)
            row_weight > 0 || continue
            push!(D, v_c / v_c0)
            push!(row_weights, row_weight)
            push!(rows, bi)
        end
    end

    return D, row_weights, rows
end


function solve_parallel_channel!(
        gains, solved, Vblock, Wblock, bl_pairs, nant, gauge, c0, c, A_amp, A_phase,
        station_models, parallel_pols; min_baselines = 3, parallel_hand_mask = nothing,
        ref_ant = nothing,
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
        local_ref = choose_local_phase_reference(active, gauge, station_models, conn, feed, ref_ant)
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

        # Gains layout (3-D template): (Frequency, Ant, Feed). Write the
        # solved channel slice along dim 1.
        for ant in 1:nant
            gains[c, ant, feed] = exp(log_amp[ant]) * cis(φ[ant])
        end
        if !isnothing(solved)
            for ant in active
                solved[c, ant, feed] = true
            end
        end
    end
    return gains
end

function antenna_phase_weights(Vblock, Wblock, bl_pairs, nant, pol, baseline_mask = nothing)
    # Inverse variance of the per-antenna phase / log-amplitude *track* at
    # channel `c`, propagated from per-baseline visibility precisions
    # `|v|²·w_raw` (`log_visibility_precision`). Earlier versions summed bare
    # `w_raw`, which only matched the inverse-variance convention that
    # downstream `weighted_least_squares` expects up to a sqrt — leaving
    # high-SNR channels (e.g. AA-AX-rich) under-weighted in the polynomial
    # bandpass fits. See `log_visibility_variance` for the per-baseline
    # derivation.
    Vp = view(Vblock; Pol = pol)
    Wp = view(Wblock; Pol = pol)
    nchan = size(Vp, Frequency)
    T = float(eltype(Wblock))
    channel_weights = zeros(T, nant, nchan)
    for bi in axes(Vp, Baseline)
        (isnothing(baseline_mask) || baseline_mask[bi]) || continue
        a, b = bl_pairs[bi]
        for c in 1:nchan
            for (v, w) in zip(view(Vp; Baseline = bi, Frequency = c), view(Wp; Baseline = bi, Frequency = c))
                prec = log_visibility_precision(v, w)
                prec > 0 || continue
                channel_weights[a, c] += prec
                channel_weights[b, c] += prec
            end
        end
    end
    return channel_weights
end

function amplitude_support_weights(Wblock, bl_pairs, nant, parallel_pols, parallel_hand_mask = nothing)
    # Sanitization-gate weights: a per-antenna×feed×channel coverage tally that
    # only needs to be monotone in "how much data backs this gain" — uses raw
    # FITS weights so gauge / collapse-detection thresholds stay calibrated.
    ref_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 1])
    par_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 2])
    nchan = size(Wblock, Frequency)
    support = zeros(Float64, nant, 2, nchan)
    for (feed_idx, (pol, mask)) in enumerate(((parallel_pols[1], ref_mask), (parallel_pols[2], par_mask)))
        Wp = view(Wblock; Pol = pol)
        for bi in axes(Wp, Baseline)
            (isnothing(mask) || mask[bi]) || continue
            a, b = bl_pairs[bi]
            for c in 1:nchan
                for w in view(Wp; Baseline = bi, Frequency = c)
                    (w > 0 && isfinite(w)) || continue
                    support[a, feed_idx, c] += w
                    support[b, feed_idx, c] += w
                end
            end
        end
    end
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

# Weighted mean of a real vector, restricted to channels marked `valid`
# (a Bool mask). Returns 0 when no channel is valid or weights sum to 0,
# so callers can use the result as an offset that vanishes for empty
# tracks.
function _weighted_mean_valid(values, weights, valid)
    wsum = sum(weights[valid])
    wsum > 0 || return zero(eltype(values))
    return sum(weights[valid] .* values[valid]) / wsum
end

function apply_zero_mean_bandpass_gauge_track!(track, weights)
    valid = (weights .> 0) .& isfinite.(weights) .& isfinite.(real.(track)) .& isfinite.(imag.(track))
    any(valid) || return track

    log_amp = log.(abs.(track))
    amp_offset = _weighted_mean_valid(log_amp, weights, valid)
    track .*= exp(-amp_offset)

    phase_track = unwrap_phase_track(vec(angle.(track)); weights = weights)
    phase_offset = _weighted_mean_valid(phase_track, weights, valid)
    track .*= cis.(-phase_offset)
    return track
end

function antenna_feed_support_weights(Wblock, bl_pairs, pol_products, nant)
    # Weights layout: 4-D = (Frequency, Ti, Baseline, Pol);
    # 3-D = (Frequency, Baseline, Pol). Stride-1 inner loop over Frequency.
    nchan = size(Wblock, 1)
    support = zeros(eltype(Wblock), nant, 2, nchan)

    if ndims(Wblock) == 4
        for pol in axes(Wblock, 4), bi in axes(Wblock, 3), s in axes(Wblock, 2)
            a, b = bl_pairs[bi]
            fa, fb = correlation_feed_pair(pol_products[pol])
            @inbounds for c in axes(Wblock, 1)
                w = Wblock[c, s, bi, pol]
                (w > 0 && isfinite(w)) || continue
                support[a, fa, c] += w
                support[b, fb, c] += w
            end
        end
    elseif ndims(Wblock) == 3
        for pol in axes(Wblock, 3), bi in axes(Wblock, 2)
            a, b = bl_pairs[bi]
            fa, fb = correlation_feed_pair(pol_products[pol])
            @inbounds for c in axes(Wblock, 1)
                w = Wblock[c, bi, pol]
                (w > 0 && isfinite(w)) || continue
                support[a, fa, c] += w
                support[b, fb, c] += w
            end
        end
    else
        error("Unsupported weight block rank: $(ndims(Wblock))")
    end

    return support
end

function bandpass_track_gauge_factor(track, weights)
    valid = (weights .> 0) .& isfinite.(weights) .& isfinite.(real.(track)) .& isfinite.(imag.(track))
    any(valid) || return 1.0 + 0.0im

    log_amp = log.(abs.(track))
    amp_offset = _weighted_mean_valid(log_amp, weights, valid)
    phase_track = unwrap_phase_track(vec(angle.(track)); weights = weights)
    phase_offset = _weighted_mean_valid(phase_track, weights, valid)
    return exp(amp_offset) * cis(phase_offset)
end

function zero_mean_bandpass_gauge_factors(gains, support_weights)
    # Gains 3-D layout: (Frequency, Ant, Feed). gamma is (Ant, Feed).
    nant = size(gains, 2)
    nfeed = size(gains, 3)
    gamma = ones(eltype(gains), nant, nfeed)
    for ant in 1:nant, feed in 1:nfeed
        gamma[ant, feed] = bandpass_track_gauge_factor(
            @view(gains[:, ant, feed]),
            vec(support_weights[ant, feed, :]),
        )
    end
    return gamma
end

function reference_antenna_bandpass_gauge_factors(gains, support_weights, ref_ant)
    nant = size(gains, 2)
    nfeed = size(gains, 3)
    gamma = ones(eltype(gains), nant, nfeed)
    for feed in 1:nfeed
        factor = bandpass_track_gauge_factor(
            @view(gains[:, ref_ant, feed]),
            vec(support_weights[ref_ant, feed, :]),
        )
        gamma[:, feed] .= factor
    end
    return gamma
end

function bandpass_gauge_factors(gains, support_weights, ::ZeroMeanBandpassGauge)
    return zero_mean_bandpass_gauge_factors(gains, support_weights)
end

function bandpass_gauge_factors(gains, support_weights, gauge::ReferenceAntennaBandpassGauge)
    return reference_antenna_bandpass_gauge_factors(gains, support_weights, gauge.ref_ant)
end

function apply_bandpass_gauge!(gains, support_weights, gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge())
    if ndims(gains) == 3
        # Gains 3-D layout: (Frequency, Ant, Feed). Divide stride-1 along
        # the channel axis.
        gamma = bandpass_gauge_factors(gains, support_weights, gauge)
        for ant in axes(gains, 2), feed in axes(gains, 3)
            for c in axes(gains, 1)
                gains[c, ant, feed] /= gamma[ant, feed]
            end
        end
    elseif ndims(gains) == 4
        # Gains 4-D layout: (Frequency, Ti, Ant, Feed). Recurse on each Ti
        # slice — yields a 3-D (Frequency, Ant, Feed) view.
        for s in axes(gains, 2)
            apply_bandpass_gauge!(@view(gains[:, s, :, :]), support_weights, gauge)
        end
    else
        error("Unsupported gain array rank: $(ndims(gains))")
    end

    return gains
end

function apply_zero_mean_bandpass_gauge!(gains, support_weights)
    return apply_bandpass_gauge!(gains, support_weights, ZeroMeanBandpassGauge())
end

function allocate_source_coherencies(Vblock)
    # Vblock layouts: 4-D = (Frequency, Ti, Baseline, Pol);
    # 3-D = (Frequency, Baseline, Pol). Source axes stay (Ti, Baseline, 2, 2)
    # — the trailing (2, 2) are pol-pair indices, not labelled Pol.
    nti = ndims(Vblock) == 4 ? size(Vblock, 2) : 1
    nbl = ndims(Vblock) == 4 ? size(Vblock, 3) : size(Vblock, 2)
    return ones(ComplexF64, nti, nbl, 2, 2)
end

function solve_source_coherencies!(source, gains, Vblock, Wblock, bl_pairs, pol_products)
    numer = zeros(ComplexF64, size(source))
    denom = zeros(Float64, size(source))

    if ndims(Vblock) == 4
        # Vblock/Wblock: (Frequency, Ti, Baseline, Pol). Stride-1 inner
        # loop over Frequency.
        for pol in axes(Vblock, 4), bi in axes(Vblock, 3), s in axes(Vblock, 2)
            a, b = bl_pairs[bi]
            fa, fb = correlation_feed_pair(pol_products[pol])
            @inbounds for c in axes(Vblock, 1)
                v = Vblock[c, s, bi, pol]
                w = Wblock[c, s, bi, pol]
                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                model = if ndims(gains) == 4
                    gains[c, s, a, fa] * conj(gains[c, s, b, fb])
                else
                    gains[c, a, fa] * conj(gains[c, b, fb])
                end
                amp2 = abs2(model)
                (amp2 > 0 && isfinite(amp2)) || continue
                numer[s, bi, fa, fb] += w * v * conj(model)
                denom[s, bi, fa, fb] += w * amp2
            end
        end
    elseif ndims(Vblock) == 3
        # Single-Ti slice: (Frequency, Baseline, Pol).
        for pol in axes(Vblock, 3), bi in axes(Vblock, 2)
            a, b = bl_pairs[bi]
            fa, fb = correlation_feed_pair(pol_products[pol])
            @inbounds for c in axes(Vblock, 1)
                v = Vblock[c, bi, pol]
                w = Wblock[c, bi, pol]
                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                model = gains[c, a, fa] * conj(gains[c, b, fb])
                amp2 = abs2(model)
                (amp2 > 0 && isfinite(amp2)) || continue
                numer[1, bi, fa, fb] += w * v * conj(model)
                denom[1, bi, fa, fb] += w * amp2
            end
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

function joint_bandpass_objective(gains, source, Vblock, Wblock, bl_pairs, pol_products)
    objective = 0.0

    if ndims(Vblock) == 4
        # (Frequency, Ti, Baseline, Pol).
        for pol in axes(Vblock, 4), bi in axes(Vblock, 3), s in axes(Vblock, 2)
            a, b = bl_pairs[bi]
            fa, fb = correlation_feed_pair(pol_products[pol])
            src = source[s, bi, fa, fb]
            @inbounds for c in axes(Vblock, 1)
                v = Vblock[c, s, bi, pol]
                w = Wblock[c, s, bi, pol]
                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                model = if ndims(gains) == 4
                    gains[c, s, a, fa] * src * conj(gains[c, s, b, fb])
                else
                    gains[c, a, fa] * src * conj(gains[c, b, fb])
                end
                residual = v - model
                objective += w * abs2(residual)
            end
        end
    elseif ndims(Vblock) == 3
        # (Frequency, Baseline, Pol).
        for pol in axes(Vblock, 3), bi in axes(Vblock, 2)
            a, b = bl_pairs[bi]
            fa, fb = correlation_feed_pair(pol_products[pol])
            src = source[1, bi, fa, fb]
            @inbounds for c in axes(Vblock, 1)
                v = Vblock[c, bi, pol]
                w = Wblock[c, bi, pol]
                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                model = gains[c, a, fa] * src * conj(gains[c, b, fb])
                residual = v - model
                objective += w * abs2(residual)
            end
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
    # gains_template layout: (Frequency, Ant, Feed). scan_gains layout:
    # (Frequency, Ti, Ant, Feed). Slice along the Frequency axis (dim 1).
    template_params = 0
    for ant in axes(state.gains_template, 2), feed in axes(state.gains_template, 3)
        valid = isfinite.(view(state.gains_template, :, ant, feed))
        if !setup.amplitude_variable_mask[ant, feed]
            template_params += constrained_real_track_parameter_count(valid)
        end
        if !setup.phase_variable_mask[ant, feed]
            template_params += constrained_real_track_parameter_count(valid)
        end
    end

    scan_params = 0
    for s in axes(state.scan_gains, 2), ant in axes(state.scan_gains, 3), feed in axes(state.scan_gains, 4)
        valid = isfinite.(view(state.scan_gains, :, s, ant, feed)) .& view(state.scan_solved, :, s, ant, feed)
        if setup.amplitude_variable_mask[ant, feed]
            scan_params += constrained_real_track_parameter_count(valid)
        end
        if setup.phase_variable_mask[ant, feed]
            scan_params += constrained_real_track_parameter_count(valid)
        end
    end

    return template_params + scan_params
end

function observed_source_parameter_count(Vblock, Wblock, pol_products)
    nti = ndims(Vblock) == 4 ? size(Vblock, 2) : 1
    nbl = ndims(Vblock) == 4 ? size(Vblock, 3) : size(Vblock, 2)
    observed = falses(nti, nbl, 2, 2)

    if ndims(Vblock) == 4
        for pol in axes(Vblock, 4), bi in axes(Vblock, 3), s in axes(Vblock, 2)
            fa, fb = correlation_feed_pair(pol_products[pol])
            @inbounds for c in axes(Vblock, 1)
                v = Vblock[c, s, bi, pol]
                w = Wblock[c, s, bi, pol]
                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                observed[s, bi, fa, fb] = true
            end
        end
    elseif ndims(Vblock) == 3
        for pol in axes(Vblock, 3), bi in axes(Vblock, 2)
            fa, fb = correlation_feed_pair(pol_products[pol])
            @inbounds for c in axes(Vblock, 1)
                v = Vblock[c, bi, pol]
                w = Wblock[c, bi, pol]
                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                observed[1, bi, fa, fb] = true
            end
        end
    else
        error("Unsupported visibility block rank: $(ndims(Vblock))")
    end

    return 2 * count(observed)
end

function apply_bandpass_gauge_with_source!(gains, source, support_weights, bl_pairs, gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge())
    ndims(gains) == 3 || error("Gauge/source refinement expects a rank-3 gain cube")

    # Gains 3-D layout: (Frequency, Ant, Feed). gamma is (Ant, Feed).
    gamma = bandpass_gauge_factors(gains, support_weights, gauge)
    for ant in axes(gains, 2), feed in axes(gains, 3)
        for c in axes(gains, 1)
            gains[c, ant, feed] /= gamma[ant, feed]
        end
    end

    for s in axes(source, 1), bi in axes(source, 2)
        a, b = bl_pairs[bi]
        left = Diagonal(vec(gamma[a, :]))
        right = Diagonal(conj.(vec(gamma[b, :])))
        source[s, bi, :, :] .= left * Matrix(@view(source[s, bi, :, :])) * right
    end

    return gains, source
end

function apply_zero_mean_bandpass_gauge_with_source!(gains, source, support_weights, bl_pairs)
    return apply_bandpass_gauge_with_source!(gains, source, support_weights, bl_pairs, ZeroMeanBandpassGauge())
end

function refine_joint_bandpass_als!(
        gains, solved, Vblock, Wblock, bl_pairs, pol_products,
        station_models::AbstractVector{<:StationBandpassModel},
        channel_freqs::AbstractVector{<:Real},
        parallel_pols::Tuple{Int, Int},
        parallel_hand_mask = nothing;
        max_iterations = 8, tolerance = 1.0e-6,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge(),
    )
    # Gains 3-D layout: (Frequency, Ant, Feed). Vblock/Wblock 4-D:
    # (Frequency, Ti, Baseline, Pol); 3-D: (Frequency, Baseline, Pol).
    nant = size(gains, 2)
    nfeed = size(gains, 3)
    source = allocate_source_coherencies(Vblock)
    support_weights = antenna_feed_support_weights(Wblock, bl_pairs, pol_products, nant)
    solve_source_coherencies!(source, gains, Vblock, Wblock, bl_pairs, pol_products)

    previous_objective = joint_bandpass_objective(gains, source, Vblock, Wblock, bl_pairs, pol_products)
    nchan = size(gains, 1)
    numer_ch = zeros(ComplexF64, nchan)
    denom_ch = zeros(Float64, nchan)
    # Per-channel Fisher info for each (ant, feed), used as projection
    # weight after the per-channel update (option (ii)).
    denom_af = zeros(Float64, nchan, nant, nfeed)

    for iter in 1:max_iterations
        fill!(denom_af, 0.0)
        for ant in axes(gains, 2), feed in axes(gains, 3)
            fill!(numer_ch, 0.0 + 0.0im)
            fill!(denom_ch, 0.0)

            if ndims(Vblock) == 4
                for pol in axes(Vblock, 4), bi in axes(Vblock, 3)
                    a, b = bl_pairs[bi]
                    fa, fb = correlation_feed_pair(pol_products[pol])
                    a_match = (a == ant && fa == feed)
                    b_match = (b == ant && fb == feed)
                    (a_match || b_match) || continue

                    for s in axes(Vblock, 2)
                        if a_match
                            src = source[s, bi, feed, fb]
                            @inbounds for c in 1:nchan
                                v = Vblock[c, s, bi, pol]
                                w = Wblock[c, s, bi, pol]
                                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                                coeff = src * conj(gains[c, b, fb])
                                amp2 = abs2(coeff)
                                amp2 > 0 || continue
                                numer_ch[c] += w * conj(coeff) * v
                                denom_ch[c] += w * amp2
                            end
                        end
                        if b_match
                            src = source[s, bi, fa, feed]
                            @inbounds for c in 1:nchan
                                v = Vblock[c, s, bi, pol]
                                w = Wblock[c, s, bi, pol]
                                (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                                coeff = conj(gains[c, a, fa] * src)
                                amp2 = abs2(coeff)
                                amp2 > 0 || continue
                                numer_ch[c] += w * conj(coeff) * conj(v)
                                denom_ch[c] += w * amp2
                            end
                        end
                    end
                end
            elseif ndims(Vblock) == 3
                for pol in axes(Vblock, 3), bi in axes(Vblock, 2)
                    a, b = bl_pairs[bi]
                    fa, fb = correlation_feed_pair(pol_products[pol])
                    a_match = (a == ant && fa == feed)
                    b_match = (b == ant && fb == feed)
                    (a_match || b_match) || continue

                    if a_match
                        src = source[1, bi, feed, fb]
                        @inbounds for c in 1:nchan
                            v = Vblock[c, bi, pol]
                            w = Wblock[c, bi, pol]
                            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                            coeff = src * conj(gains[c, b, fb])
                            amp2 = abs2(coeff)
                            amp2 > 0 || continue
                            numer_ch[c] += w * conj(coeff) * v
                            denom_ch[c] += w * amp2
                        end
                    end
                    if b_match
                        src = source[1, bi, fa, feed]
                        @inbounds for c in 1:nchan
                            v = Vblock[c, bi, pol]
                            w = Wblock[c, bi, pol]
                            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue
                            coeff = conj(gains[c, a, fa] * src)
                            amp2 = abs2(coeff)
                            amp2 > 0 || continue
                            numer_ch[c] += w * conj(coeff) * conj(v)
                            denom_ch[c] += w * amp2
                        end
                    end
                end
            else
                error("Unsupported visibility block rank: $(ndims(Vblock))")
            end

            @inbounds for c in 1:nchan
                denom_af[c, ant, feed] = denom_ch[c]
                denom_ch[c] > 0 || continue
                gains[c, ant, feed] = numer_ch[c] / denom_ch[c]
                isnothing(solved) || (solved[c, ant, feed] = true)
            end
        end

        # Project the per-channel update for every (ant, feed) onto the
        # user-specified bandpass model subspace using per-iter Fisher-info
        # weights `denom_af · |g|²`. Reference feed first, then partner-feed
        # ratio (handled inside the helpers).
        constrain_gain_amplitudes_with_weights!(gains, denom_af, channel_freqs, station_models)
        constrain_gain_phases_with_weights!(gains, denom_af, channel_freqs, station_models)

        apply_bandpass_gauge_with_source!(gains, source, support_weights, bl_pairs, gauge)
        solve_source_coherencies!(source, gains, Vblock, Wblock, bl_pairs, pol_products)
        objective = joint_bandpass_objective(gains, source, Vblock, Wblock, bl_pairs, pol_products)

        improvement = previous_objective - objective
        # Skip the monotone-improvement guard on iter 1: the projection
        # step is non-monotonic in unconstrained χ², so a single iter can
        # legitimately raise the measured objective above the per-channel
        # initial state. From iter 2 onward both endpoints lie on the
        # constraint manifold and the descent argument applies.
        iter > 1 && improvement <= 0 && break
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
        phase_track, channel_weights, channel_freqs, phase_model::AbstractBandpassModel,
        default_segmentation::AbstractFrequencySegmentation,
    )
    components = model_components(phase_model, default_segmentation)
    length(components) == 1 && components[1].model isa PerChannelBandpassModel && return phase_track

    phase_unwrapped = unwrap_phase_track(phase_track; weights = channel_weights)

    # Basis is fit in a centered frequency coordinate. The center choice
    # is gauge-immaterial (a constant offset rotates the basis but not
    # the fitted predictions), so use the plain `mean(channel_freqs)`.
    x = 2π .* (channel_freqs .- mean(channel_freqs))
    valid = (channel_weights .> 0) .& isfinite.(channel_weights) .& isfinite.(phase_unwrapped) .& isfinite.(x)
    count(valid) >= 2 || return phase_track

    basis = Vector{Vector{Float64}}()
    for component in components
        append!(basis, component_design_columns(component, x, valid))
    end

    if isempty(basis)
        return zeros(length(phase_track))
    end

    A = hcat(basis...)
    count(valid) >= size(A, 2) || return phase_track
    coeffs = weighted_least_squares(A[valid, :], phase_unwrapped[valid], channel_weights[valid])
    fitted = A * coeffs
    # Weighted-zero-mean gauge: subtract the channel-weighted mean of
    # the fitted phase track. Channel-symmetric (no anchor).
    fitted .-= _weighted_mean_valid(fitted, channel_weights, valid)
    return fitted
end

function fit_amplitude_model(
        log_amp_track, channel_weights, channel_freqs, ::PerChannelBandpassModel,
        default_segmentation::AbstractFrequencySegmentation,
    )
    return log_amp_track
end

function fit_amplitude_model(
        log_amp_track, channel_weights, channel_freqs, amp_model::AbstractBandpassModel,
        default_segmentation::AbstractFrequencySegmentation,
    )
    components = model_components(amp_model, default_segmentation)
    length(components) == 1 && components[1].model isa PerChannelBandpassModel && return log_amp_track

    x = channel_freqs .- mean(channel_freqs)
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
    fitted .-= _weighted_mean_valid(fitted, channel_weights, valid)
    return fitted
end

function replacement_amplitude_scale(amps, support, c; neighbor_window = 2)
    local_values = Float64[]
    lo = max(1, c - neighbor_window)
    hi = min(length(amps), c + neighbor_window)
    for j in lo:hi
        j == c && continue
        support[j] > 0 || continue
        amp = amps[j]
        isfinite(amp) && amp > 0 || continue
        push!(local_values, amp)
    end
    length(local_values) >= 2 && return median(local_values)

    global_values = Float64[]
    for j in eachindex(amps)
        j == c && continue
        support[j] > 0 || continue
        amp = amps[j]
        isfinite(amp) && amp > 0 || continue
        push!(global_values, amp)
    end
    isempty(global_values) && return nothing
    return median(global_values)
end

function sanitize_gain_amplitudes!(
        gains, support_weights;
        neighbor_window = 2,
    )
    # Gains 3-D layout: (Frequency, Ant, Feed). support_weights stays
    # (Ant, Feed, Frequency).
    amps = abs.(gains)
    repaired = NamedTuple[]

    for ant in axes(gains, 2), feed in axes(gains, 3), c in axes(gains, 1)
        support_weights[ant, feed, c] > 0 || continue

        amp = amps[c, ant, feed]
        # Only repair genuinely broken values (NaN / Inf / negative / exact zero).
        # Legitimately small amplitudes (e.g. EHT IF-edge bandpass rolloff) must be
        # left alone — replacing them with a neighbor median produces a model that
        # vastly overpredicts visibilities at those channels.
        (isfinite(amp) && amp > 0) && continue

        local_scale = replacement_amplitude_scale(
            view(amps, :, ant, feed), view(support_weights, ant, feed, :), c;
            neighbor_window = neighbor_window,
        )
        isnothing(local_scale) && continue

        gains[c, ant, feed] = local_scale * cis(angle(gains[c, ant, feed]))
        push!(repaired, (; ant, feed, channel = c, amplitude = amp, replacement = local_scale))
    end

    return repaired
end

"""
    inspect_collapsed_gain_amplitudes(gains, support_weights;
        collapse_fraction = 0.05, min_gain_amplitude = 1.0e-2,
        neighbor_window = 2)

Non-mutating diagnostic that flags channels whose gain amplitude is
suspiciously small relative to its neighbours. Returns a `Vector{NamedTuple}`
with `(ant, feed, channel, amplitude, neighbor_median)` rows for any channel
where the amplitude is finite-positive but either below `min_gain_amplitude`
or below `collapse_fraction * neighbor_median`.

This is the "warn but don't sanitize" companion to
[`sanitize_gain_amplitudes!`](@ref): it surfaces likely-broken bandpass solves
(e.g. a single dead channel that nevertheless produced a finite, tiny gain)
without overwriting the solution, so legitimately small amplitudes (e.g. EHT
IF-edge rolloff) are left alone but obvious collapses are still reported.
"""
function inspect_collapsed_gain_amplitudes(
        gains, support_weights;
        collapse_fraction = 0.05, min_gain_amplitude = 1.0e-2,
        neighbor_window = 2,
    )
    # Gains 3-D layout: (Frequency, Ant, Feed). support_weights:
    # (Ant, Feed, Frequency).
    amps = abs.(gains)
    suspects = NamedTuple[]

    for ant in axes(gains, 2), feed in axes(gains, 3), c in axes(gains, 1)
        support_weights[ant, feed, c] > 0 || continue

        amp = amps[c, ant, feed]
        (isfinite(amp) && amp > 0) || continue

        neighbor_median = replacement_amplitude_scale(
            view(amps, :, ant, feed), view(support_weights, ant, feed, :), c;
            neighbor_window = neighbor_window,
        )
        isnothing(neighbor_median) && continue

        is_collapsed = amp < min_gain_amplitude || amp < collapse_fraction * neighbor_median
        is_collapsed || continue

        push!(suspects, (; ant, feed, channel = c, amplitude = amp, neighbor_median))
    end

    return suspects
end

function warn_collapsed_gain_amplitudes(suspects, ant_names = nothing; context = "")
    isempty(suspects) && return nothing

    grouped = Dict{Tuple{Int, Int}, Vector{NamedTuple}}()
    for s in suspects
        key = (s.ant, s.feed)
        push!(get!(grouped, key, NamedTuple[]), s)
    end

    details = String[]
    for ((ant, feed), entries) in sort!(collect(grouped); by = first)
        ant_label = isnothing(ant_names) ? string("ant", ant) : string(ant_names[ant])
        channels = sort(getfield.(entries, :channel))
        min_amp = minimum(getfield.(entries, :amplitude))
        max_neighbor = maximum(getfield.(entries, :neighbor_median))
        push!(
            details,
            string(
                ant_label, "/feed", feed,
                " channels=", join(channels, ","),
                " min_amp=", round(min_amp; digits = 4),
                " neighbor_median<=", round(max_neighbor; digits = 4)
            )
        )
    end

    prefix = isempty(context) ? "" : string(context, ": ")
    @warn string(prefix, "collapsed gain amplitudes detected (not sanitized); inspect data/support") suspect_count = length(suspects) details
    return nothing
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

function constrain_gain_amplitudes!(gains, Vblock, Wblock, bl_pairs, channel_freqs, station_models, parallel_pols, parallel_hand_mask = nothing)
    # Gains 3-D layout: (Frequency, Ant, Feed). Slice along the Frequency axis.
    nant = size(gains, 2)
    ref_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 1])
    par_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 2])
    reference_weights = antenna_phase_weights(Vblock, Wblock, bl_pairs, nant, parallel_pols[1], ref_mask)
    partner_weights = antenna_phase_weights(Vblock, Wblock, bl_pairs, nant, parallel_pols[2], par_mask)

    for ant in 1:nant
        model = station_models[ant]
        reference_feed = model.reference_feed
        partner_feed = partner_feed_index(model.reference_feed)

        abs_amp_model = model.reference.amplitude.model
        if !(abs_amp_model isa PerChannelBandpassModel)
            reference_log_amp = log.(abs.(gains[:, ant, reference_feed]))
            fitted_reference_log_amp = fit_amplitude_model(
                vec(reference_log_amp),
                vec(reference_weights[ant, :]),
                channel_freqs,
                abs_amp_model,
                model.reference.amplitude.segmentation.frequency,
            )
            gains[:, ant, reference_feed] = exp.(fitted_reference_log_amp) .* cis.(angle.(gains[:, ant, reference_feed]))
        end

        relative_amp_model = model.relative.amplitude.model
        if !(relative_amp_model isa PerChannelBandpassModel)
            ratio = gains[:, ant, partner_feed] ./ gains[:, ant, reference_feed]
            relative_log_amp = log.(abs.(ratio))
            relative_weights = sqrt.(reference_weights[ant, :] .* partner_weights[ant, :])
            fitted_relative_log_amp = fit_amplitude_model(
                vec(relative_log_amp),
                vec(relative_weights),
                channel_freqs,
                relative_amp_model,
                model.relative.amplitude.segmentation.frequency,
            )
            gains[:, ant, partner_feed] = abs.(gains[:, ant, reference_feed]) .* exp.(fitted_relative_log_amp) .* cis.(angle.(gains[:, ant, partner_feed]))
        end
    end

    return gains
end

function constrain_gain_phases!(gains, Vblock, Wblock, bl_pairs, channel_freqs, station_models, parallel_pols, parallel_hand_mask = nothing)
    # Gains 3-D layout: (Frequency, Ant, Feed).
    nant = size(gains, 2)
    ref_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 1])
    par_mask = isnothing(parallel_hand_mask) ? nothing : @view(parallel_hand_mask[:, 2])
    reference_weights = antenna_phase_weights(Vblock, Wblock, bl_pairs, nant, parallel_pols[1], ref_mask)
    partner_weights = antenna_phase_weights(Vblock, Wblock, bl_pairs, nant, parallel_pols[2], par_mask)

    for ant in 1:nant
        model = station_models[ant]
        reference_feed = model.reference_feed
        partner_feed = partner_feed_index(model.reference_feed)

        reference_phase_model = model.reference.phase.model
        if !(reference_phase_model isa PerChannelBandpassModel)
            reference_phase_track = vec(angle.(gains[:, ant, reference_feed]))
            fitted_reference_phase = fit_phase_model(
                reference_phase_track,
                vec(reference_weights[ant, :]),
                channel_freqs,
                reference_phase_model,
                model.reference.phase.segmentation.frequency,
            )
            gains[:, ant, reference_feed] = abs.(gains[:, ant, reference_feed]) .* cis.(fitted_reference_phase)
        end

        relative_phase_model = model.relative.phase.model
        if !(relative_phase_model isa PerChannelBandpassModel)
            ratio = gains[:, ant, partner_feed] ./ gains[:, ant, reference_feed]
            relative_phase_track = vec(angle.(ratio))
            relative_weights = sqrt.(reference_weights[ant, :] .* partner_weights[ant, :])
            fitted_relative_phase = fit_phase_model(
                relative_phase_track,
                vec(relative_weights),
                channel_freqs,
                relative_phase_model,
                model.relative.phase.segmentation.frequency,
            )
            gains[:, ant, partner_feed] = abs.(gains[:, ant, partner_feed]) .* cis.(angle.(gains[:, ant, reference_feed]) .+ fitted_relative_phase)
        end
    end

    return gains
end

# Per-iter Fisher-info-weighted projection. `denom_af[c, ant, feed]` is the
# per-channel ALS denominator from `refine_joint_bandpass_als!` — i.e. the
# Fisher information of `g[c, ant, feed]` at the current source / other-gain
# iterate. The WLS weight for projecting `log|g|` and `arg g` onto the user's
# basis is `denom_af · |g|²` (linearized log/arg variance for a complex
# Gaussian estimator). Unlike `constrain_gain_amplitudes!` /
# `constrain_gain_phases!` these helpers do not need `Vblock`/`Wblock`/
# `bl_pairs`/`parallel_pols` because the weight is precomputed.
function constrain_gain_amplitudes_with_weights!(
        gains, denom_af, channel_freqs, station_models,
    )
    nant = size(gains, 2)
    @assert size(denom_af) == size(gains)
    for ant in 1:nant
        model = station_models[ant]
        reference_feed = model.reference_feed
        partner_feed = partner_feed_index(model.reference_feed)

        ref_g = view(gains, :, ant, reference_feed)
        par_g = view(gains, :, ant, partner_feed)
        ref_w = view(denom_af, :, ant, reference_feed) .* abs2.(ref_g)
        par_w = view(denom_af, :, ant, partner_feed) .* abs2.(par_g)

        abs_amp_model = model.reference.amplitude.model
        if !(abs_amp_model isa PerChannelBandpassModel)
            reference_log_amp = log.(abs.(ref_g))
            fitted_reference_log_amp = fit_amplitude_model(
                vec(reference_log_amp),
                vec(ref_w),
                channel_freqs,
                abs_amp_model,
                model.reference.amplitude.segmentation.frequency,
            )
            gains[:, ant, reference_feed] = exp.(fitted_reference_log_amp) .* cis.(angle.(ref_g))
        end

        relative_amp_model = model.relative.amplitude.model
        if !(relative_amp_model isa PerChannelBandpassModel)
            ratio = par_g ./ ref_g
            relative_log_amp = log.(abs.(ratio))
            relative_weights = sqrt.(ref_w .* par_w)  # geometric-mean precision (existing convention)
            fitted_relative_log_amp = fit_amplitude_model(
                vec(relative_log_amp),
                vec(relative_weights),
                channel_freqs,
                relative_amp_model,
                model.relative.amplitude.segmentation.frequency,
            )
            gains[:, ant, partner_feed] = abs.(gains[:, ant, reference_feed]) .* exp.(fitted_relative_log_amp) .* cis.(angle.(par_g))
        end
    end
    return gains
end

function constrain_gain_phases_with_weights!(
        gains, denom_af, channel_freqs, station_models,
    )
    nant = size(gains, 2)
    @assert size(denom_af) == size(gains)
    for ant in 1:nant
        model = station_models[ant]
        reference_feed = model.reference_feed
        partner_feed = partner_feed_index(model.reference_feed)

        ref_g = view(gains, :, ant, reference_feed)
        par_g = view(gains, :, ant, partner_feed)
        ref_w = view(denom_af, :, ant, reference_feed) .* abs2.(ref_g)
        par_w = view(denom_af, :, ant, partner_feed) .* abs2.(par_g)

        reference_phase_model = model.reference.phase.model
        if !(reference_phase_model isa PerChannelBandpassModel)
            reference_phase_track = vec(angle.(ref_g))
            fitted_reference_phase = fit_phase_model(
                reference_phase_track,
                vec(ref_w),
                channel_freqs,
                reference_phase_model,
                model.reference.phase.segmentation.frequency,
            )
            gains[:, ant, reference_feed] = abs.(ref_g) .* cis.(fitted_reference_phase)
        end

        relative_phase_model = model.relative.phase.model
        if !(relative_phase_model isa PerChannelBandpassModel)
            ratio = par_g ./ gains[:, ant, reference_feed]
            relative_phase_track = vec(angle.(ratio))
            relative_weights = sqrt.(ref_w .* par_w)
            fitted_relative_phase = fit_phase_model(
                relative_phase_track,
                vec(relative_weights),
                channel_freqs,
                relative_phase_model,
                model.relative.phase.segmentation.frequency,
            )
            gains[:, ant, partner_feed] = abs.(par_g) .* cis.(angle.(gains[:, ant, reference_feed]) .+ fitted_relative_phase)
        end
    end
    return gains
end

function sanitize_and_gauge_gains!(
        gains, Wblock, bl_pairs, parallel_pols, parallel_hand_mask = nothing;
        ant_names = nothing, context = "",
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge(),
    )
    # Sanitization-and-gauge only. ALS already produced the per-channel
    # solution; this step does NOT refit gains onto a smoothed model
    # basis. Per-station frequency models are an ALS-time concern (and
    # an init concern via `solve_bandpass_template`/`single_scan`); at
    # finalize we just repair NaN/zero amplitudes and apply the user's
    # gauge choice (channel-symmetric weighted-zero-mean per (ant, feed),
    # or pinned to a reference antenna's track).
    support_weights = amplitude_support_weights(Wblock, bl_pairs, size(gains, 2), parallel_pols, parallel_hand_mask)
    repaired = sanitize_gain_amplitudes!(gains, support_weights)
    warn_sanitized_gain_amplitudes(repaired, ant_names; context = context)
    apply_bandpass_gauge!(gains, support_weights, gauge)
    return gains
end

# Pick a per-call init reference channel for the per-channel ratio basis
# in `solve_parallel_channel!`. Internal to solver init only — the
# downstream gauge is channel-symmetric and does not depend on this
# choice. Falls back to channel 1 if support is uniformly zero.
function _init_ref_channel_from_weights(Wblock, parallel_pols)
    # Wblock layouts: (Frequency, Ti, Baseline, Pol) or (Frequency, Baseline, Pol).
    rr, ll = parallel_pols
    nchan = size(Wblock, 1)
    chan_weight = zeros(Float64, nchan)
    if ndims(Wblock) == 4
        for pol in (rr, ll), bi in axes(Wblock, 3), s in axes(Wblock, 2)
            @inbounds for c in 1:nchan
                w = Wblock[c, s, bi, pol]
                (w > 0 && isfinite(w)) || continue
                chan_weight[c] += w
            end
        end
    else
        for pol in (rr, ll), bi in axes(Wblock, 2)
            @inbounds for c in 1:nchan
                w = Wblock[c, bi, pol]
                (w > 0 && isfinite(w)) || continue
                chan_weight[c] += w
            end
        end
    end
    return any(chan_weight .> 0) ? argmax(chan_weight) : 1
end

function solve_bandpass_single_scan(
        Vs, Ws, bl_pairs, nant, channel_freqs, station_models, pol_products, parallel_pols;
        ant_names = nothing, context = "",
        min_baselines = 3, joint_als_iterations = 8, joint_als_tolerance = 1.0e-6,
        parallel_hand_mask = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge(),
        ref_ant = nothing,
    )
    # Vs/Ws layout: (Frequency, Baseline, Pol). gains layout (3-D template):
    # (Frequency, Ant, Feed).
    nchan = size(Vs, 1)
    A_amp, A_phase = design_matrices(bl_pairs, nant)
    init_ref_chan = _init_ref_channel_from_weights(Ws, parallel_pols)

    gains = ones(ComplexF64, nchan, nant, 2)
    solved = falses(nchan, nant, 2)

    for c in 1:nchan
        c == init_ref_chan && continue
        solve_parallel_channel!(
            gains, solved, Vs, Ws, bl_pairs, nant, gauge, init_ref_chan, c, A_amp, A_phase,
            station_models, parallel_pols; min_baselines = min_baselines, parallel_hand_mask = parallel_hand_mask,
            ref_ant = ref_ant,
        )
    end

    # Init-time projection: pull the per-channel init onto the user's
    # bandpass model subspace before ALS starts. Uses the
    # `antenna_phase_weights`-flavored helpers because we don't yet have the
    # per-iter `denom_af` (that's an artifact of the ALS loop). The first
    # ALS iter immediately re-projects with proper Fisher-info weights.
    constrain_gain_amplitudes!(
        gains, Vs, Ws, bl_pairs, channel_freqs,
        station_models, parallel_pols, parallel_hand_mask,
    )
    constrain_gain_phases!(
        gains, Vs, Ws, bl_pairs, channel_freqs,
        station_models, parallel_pols, parallel_hand_mask,
    )

    joint_als_iterations > 0 && refine_joint_bandpass_als!(
        gains, solved, Vs, Ws, bl_pairs, pol_products,
        station_models, channel_freqs, parallel_pols, parallel_hand_mask;
        max_iterations = joint_als_iterations, tolerance = joint_als_tolerance,
        gauge = gauge,
    )

    sanitize_and_gauge_gains!(
        gains, Ws, bl_pairs, parallel_pols, parallel_hand_mask;
        ant_names = ant_names, context = context,
        gauge = gauge,
    )
    return gains, solved
end

function solve_bandpass_template(
        V, W, bl_pairs, nant, channel_freqs, station_models, pol_products, parallel_pols;
        ant_names = nothing, context = "template",
        min_baselines = 3, joint_als_iterations = 8, joint_als_tolerance = 1.0e-6,
        parallel_hand_mask = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge(),
        ref_ant = nothing,
    )
    # V/W layout: (Frequency, Ti, Baseline, Pol). gains layout (3-D template):
    # (Frequency, Ant, Feed).
    nchan = size(V, 1)
    A_amp, A_phase = design_matrices(bl_pairs, nant)
    init_ref_chan = _init_ref_channel_from_weights(W, parallel_pols)

    gains = ones(ComplexF64, nchan, nant, 2)

    for c in 1:nchan
        c == init_ref_chan && continue
        solve_parallel_channel!(
            gains, nothing, V, W, bl_pairs, nant, gauge, init_ref_chan, c, A_amp, A_phase,
            station_models, parallel_pols; min_baselines = min_baselines, parallel_hand_mask = parallel_hand_mask,
            ref_ant = ref_ant,
        )
    end

    # Init-time projection (see solve_bandpass_single_scan for rationale).
    constrain_gain_amplitudes!(
        gains, V, W, bl_pairs, channel_freqs,
        station_models, parallel_pols, parallel_hand_mask,
    )
    constrain_gain_phases!(
        gains, V, W, bl_pairs, channel_freqs,
        station_models, parallel_pols, parallel_hand_mask,
    )

    joint_als_iterations > 0 && refine_joint_bandpass_als!(
        gains, nothing, V, W, bl_pairs, pol_products,
        station_models, channel_freqs, parallel_pols, parallel_hand_mask;
        max_iterations = joint_als_iterations, tolerance = joint_als_tolerance,
        gauge = gauge,
    )

    sanitize_and_gauge_gains!(
        gains, W, bl_pairs, parallel_pols, parallel_hand_mask;
        ant_names = ant_names, context = context,
        gauge = gauge,
    )
    return gains
end

function merge_scan_gains!(gain_slice, scan_gains, solved, phase_variable_mask, amplitude_variable_mask)
    # 3-D layout: (Frequency, Ant, Feed). Stride-1 inner loop over Frequency.
    nchan, nant, nfeed = size(gain_slice)
    for a in 1:nant, feed in 1:nfeed
        @inbounds for c in 1:nchan
            solved[c, a, feed] || continue

            amp = amplitude_variable_mask[a, feed] ? abs(scan_gains[c, a, feed]) : abs(gain_slice[c, a, feed])
            phase = phase_variable_mask[a, feed] ? angle(scan_gains[c, a, feed]) : angle(gain_slice[c, a, feed])
            gain_slice[c, a, feed] = amp * cis(phase)
        end
    end

    return gain_slice
end

struct BandpassSolverSetup{
        D,
        B <: AbstractVector{<:Tuple{<:Integer, <:Integer}},
        F <: AbstractVector{<:Real},
        S <: AbstractVector{<:StationBandpassModel},
        C <: AbstractVector{<:AbstractString},
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
    pol_products::C
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
    als_iterations_completed::Int
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

prepare_bandpass_solver(avg::UVSet, ref_ant; kwargs...) =
    prepare_bandpass_solver(_to_bandpass_dataset(avg), ref_ant; kwargs...)

prepare_bandpass_solver(avg::UVSet, source::AbstractString, ref_ant; kwargs...) =
    prepare_bandpass_solver(_to_bandpass_dataset(select_source(avg, source)), ref_ant; kwargs...)

function prepare_bandpass_solver(
        avg::BandpassDataset, ref_ant;
        min_baselines = 3, station_models = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge(),
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
    pols = pol_products(avg)
    parallel_pols = parallel_hand_indices(pols)
    parallel_hand_mask = build_parallel_hand_mask(avg.antennas, avg.baselines.pairs)
    phase_variable_mask = falses(nant, 2)
    amplitude_variable_mask = falses(nant, 2)
    for ant in 1:nant, feed in 1:2
        phase_variable_mask[ant, feed] = phase_is_per_scan(station_models[ant], feed)
        amplitude_variable_mask[ant, feed] = amplitude_is_per_scan(station_models[ant], feed)
    end

    setup = BandpassSolverSetup(
        avg,
        ref_ant,
        gauge,
        min_baselines,
        avg.baselines.pairs,
        UVData.channel_freqs(avg.freq_setup),
        station_models,
        parallel_pols,
        parallel_hand_mask,
        pols,
        phase_variable_mask,
        amplitude_variable_mask,
    )

    warn_underdetermined_per_scan_supports(setup)

    return setup
end

# Per-(ant, scan, feed) free-parameter count for one scan, given the
# user's station model. PerChannel is treated as `nchan` parameters
# (full channel-by-channel freedom). For Composite models the per-block
# basis count is `parameter_count(model) + 1` per block (the `+1`
# accounts for the per-block `Flat` constant; `parameter_count` returns
# only the polynomial-degree count). When time segmentation is global
# this returns 0 (the per-scan slice doesn't add free DoF).
function _per_scan_free_params(spec::BandpassSpec, nchan::Integer)
    is_per_scan(spec.segmentation.time) || return 0
    model = spec.model
    seg = spec.segmentation.frequency
    return _model_freedom(model, seg, nchan)
end

_model_freedom(::PerChannelBandpassModel, _seg, nchan::Integer) = nchan
_model_freedom(model::SegmentedBandpassModel, _outer_seg, nchan::Integer) =
    _segmented_freedom(model, nchan)
_model_freedom(model::CompositeBandpassModel, _seg, nchan::Integer) =
    sum(_segmented_freedom(component, nchan) for component in model.components)
_model_freedom(model::AbstractBandpassModel, _seg, _nchan::Integer) =
    something(parameter_count(model), 0)

function _segmented_freedom(component::SegmentedBandpassModel, nchan::Integer)
    component.model isa PerChannelBandpassModel && return nchan
    nblocks = length(frequency_segments(component.segmentation, nchan))
    is_flat = component.model isa FlatBandpassModel
    per_block = is_flat ? 1 : something(parameter_count(component.model), 0)
    return nblocks * per_block
end

# Count of distinct baselines touching `(ant, feed)` with non-zero
# parallel-hand weight on a given scan. Used as a heuristic constraint
# count for the per-scan WLS gain solve.
function _baselines_with_parallel_data(setup::BandpassSolverSetup, ant::Integer, scan::Integer, parallel_pol::Integer)
    Wp = parent(setup.data.weights)
    bl_pairs = setup.bl_pairs
    n = 0
    for bi in eachindex(bl_pairs)
        a, b = bl_pairs[bi]
        (ant == a || ant == b) || continue
        @inbounds for c in axes(Wp, 1)
            w = Wp[c, scan, bi, parallel_pol]
            if w > 0 && isfinite(w)
                n += 1
                break
            end
        end
    end
    return n
end

"""
    warn_underdetermined_per_scan_supports(setup)

Emit `@warn` for any `(ant, scan, feed)` whose per-scan station model
has more free parameters than the data can plausibly constrain — i.e.
where `nchan × n_baselines_with_parallel_data` is below `model_dof`.
This catches cases like NN on early EHT scans (only 2 baselines with
data) configured with `PerChannelBandpassModel` per scan: each scan
has 32 free per-channel DoF but only 2 baselines × 32 channels = 64
real WLS constraints minus a per-channel source DoF, leaving the
solve under-determined.

Pure diagnostic; does not change the solver's behaviour.
"""
function warn_underdetermined_per_scan_supports(setup::BandpassSolverSetup)
    nchan = length(setup.channel_freqs)
    nti = size(parent(setup.data.weights), 2)
    nant = length(setup.data.antennas)
    rr_pol, ll_pol = setup.parallel_pols
    rows = NamedTuple[]
    for ant in 1:nant
        model = setup.station_models[ant]
        for feed in 1:2
            spec_phase, spec_amp = if feed == model.reference_feed
                model.reference.phase, model.reference.amplitude
            else
                model.relative.phase, model.relative.amplitude
            end
            phase_dof = _per_scan_free_params(spec_phase, nchan)
            amp_dof = _per_scan_free_params(spec_amp, nchan)
            (phase_dof + amp_dof > 0) || continue   # global model — not per-scan
            parallel_pol = feed == 1 ? rr_pol : ll_pol
            model_dof = phase_dof + amp_dof
            for s in 1:nti
                nbl = _baselines_with_parallel_data(setup, ant, s, parallel_pol)
                # Skip scans where the antenna has no data — those leave
                # the gain at its initialization value but aren't a
                # model-vs-data mismatch, just a coverage gap.
                nbl > 0 || continue
                # The per-channel ALS for one (ant, feed) at one channel
                # sees `2·nbl` real measurements vs at least `2 + 2·nbl`
                # unknowns (this gain plus per-baseline source DoF), so
                # it is always 2 real DoF short per channel — relying on
                # the joint (gains, source) solve over many channels and
                # baselines to constrain the remaining DoF. In practice
                # the per-scan solve is well-conditioned only when an
                # antenna participates in at least `min_baselines` non-
                # degenerate baselines (3 by default). Flag any per-
                # scan (ant, feed) below that threshold whose model
                # carries any per-scan freedom — those configurations
                # let the per-channel gain track drift along the under-
                # determined null space, which surfaces as visibly noisy
                # corrected visibilities.
                if nbl < setup.min_baselines
                    push!(rows, (
                        ant = setup.data.antennas.name[ant],
                        scan = s,
                        feed = feed,
                        nbaselines = nbl,
                        model_dof = model_dof,
                        min_baselines = setup.min_baselines,
                    ))
                end
            end
        end
    end
    isempty(rows) && return
    sort!(rows; by = r -> (r.ant, r.scan, r.feed))
    samples = first(rows, min(length(rows), 8))
    detail = join(
        [
            "$(r.ant) scan=$(r.scan) feed=$(r.feed) nbl=$(r.nbaselines) (< min_baselines=$(r.min_baselines))"
                for r in samples
        ],
        "; ",
    )
    extra = length(rows) > length(samples) ? " (+$(length(rows) - length(samples)) more)" : ""
    @warn (
        "prepare_bandpass_solver: $(length(rows)) (antenna, scan, feed) configurations are " *
            "under-determined — the antenna participates in fewer than `min_baselines` " *
            "baselines on those scans while its station model carries per-scan freedom. " *
            "The per-scan gain solve will drift along the under-determined null space and " *
            "corrected visibilities on those baselines will look unstable. Consider " *
            "downgrading those antennas to a global-time / coarser-frequency station model " *
            "(e.g. Composite(Flat ⊕ Poly2) with GlobalTimeSegmentation, or " *
            "BlockFrequencySegmentation), or excluding the affected scans. Sample: " *
            detail * extra
    )
    return
end

function update_state_sources_and_objectives!(setup::BandpassSolverSetup, state::BandpassSolverState)
    # data.vis/weights layout: (Frequency, Ti, Baseline, Pol). scan_gains:
    # (Frequency, Ti, Ant, Feed). scan_sources: (Ti, Baseline, 2, 2).
    # Slice along Ti (dim 2 for vis/weights/scan_gains, dim 1 for scan_sources).
    solve_source_coherencies!(state.template_source, state.gains_template, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_products)
    state.template_objective = joint_bandpass_objective(
        state.gains_template, state.template_source, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_products
    )

    for s in axes(state.scan_gains, 2)
        vis_s = view(setup.data.vis, :, s, :, :)
        w_s = view(setup.data.weights, :, s, :, :)
        scan_source = allocate_source_coherencies(vis_s)
        solve_source_coherencies!(
            scan_source,
            view(state.scan_gains, :, s, :, :),
            vis_s, w_s,
            setup.bl_pairs,
            setup.pol_products
        )
        state.scan_sources[s, :, :, :] .= scan_source[1, :, :, :]
        state.scan_objectives[s] = joint_bandpass_objective(
            view(state.scan_gains, :, s, :, :),
            scan_source,
            vis_s, w_s,
            setup.bl_pairs,
            setup.pol_products
        )
    end

    return state
end

function assemble_bandpass_state_gains!(
        merged_gains, gains_template, scan_gains, scan_solved, phase_variable_mask, amplitude_variable_mask
    )
    # merged_gains/scan_gains/scan_solved layout: (Frequency, Ti, Ant, Feed).
    # gains_template layout: (Frequency, Ant, Feed). Broadcast template across
    # Ti (dim 2).
    nchan = size(gains_template, 1)
    nant = size(gains_template, 2)
    nfeed = size(gains_template, 3)
    merged_gains .= reshape(gains_template, nchan, 1, nant, nfeed)
    for s in axes(scan_gains, 2)
        merge_scan_gains!(
            view(merged_gains, :, s, :, :),
            view(scan_gains, :, s, :, :),
            view(scan_solved, :, s, :, :),
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
    # data.vis layout: (Frequency, Ti, Baseline, Pol).
    data = setup.data
    nchan, nti, _, _ = size(parent(data.vis))
    nbl = length(setup.bl_pairs)
    nant = size(gains_template, 2)
    nfeed = size(gains_template, 3)

    # gains: (Frequency, Ti, Ant, Feed) — template broadcast across Ti.
    gains = repeat(reshape(gains_template, nchan, 1, nant, nfeed), 1, nti, 1, 1)
    assemble_bandpass_state_gains!(
        gains, gains_template, scan_gains, scan_solved, setup.phase_variable_mask, setup.amplitude_variable_mask
    )

    state = BandpassSolverState(
        gains_template,
        scan_gains,
        scan_solved,
        gains,
        ones(ComplexF64, nti, nbl, 2, 2),
        ones(ComplexF64, nti, nbl, 2, 2),
        0.0,
        zeros(Float64, nti),
        0,
    )
    return update_state_sources_and_objectives!(setup, state)
end

function initialize_bandpass_state(setup::BandpassSolverSetup, ::RatioBandpassInitializer)
    # data.vis layout: (Frequency, Ti, Baseline, Pol). gains_template:
    # (Frequency, Ant, Feed). scan_gains/scan_solved: (Frequency, Ti, Ant, Feed).
    data = setup.data
    nchan, nti, _, _ = size(parent(data.vis))
    nant = length(data.antennas)

    gains_template = solve_bandpass_template(
        data.vis,
        data.weights,
        setup.bl_pairs,
        nant,
        setup.channel_freqs,
        setup.station_models,
        setup.pol_products,
        setup.parallel_pols;
        ant_names = data.antennas.name,
        min_baselines = setup.min_baselines,
        joint_als_iterations = 0,
        parallel_hand_mask = setup.parallel_hand_mask,
        gauge = setup.gauge,
        ref_ant = setup.ref_ant,
    )

    scan_gains = ones(ComplexF64, nchan, nti, nant, 2)
    scan_solved = falses(nchan, nti, nant, 2)
    for s in 1:nti
        gains_scan, solved = solve_bandpass_single_scan(
            view(data.vis, :, s, :, :),
            view(data.weights, :, s, :, :),
            setup.bl_pairs,
            nant,
            setup.channel_freqs,
            setup.station_models,
            setup.pol_products,
            setup.parallel_pols;
            ant_names = data.antennas.name,
            context = string("scan ", s),
            min_baselines = setup.min_baselines,
            joint_als_iterations = 0,
            parallel_hand_mask = setup.parallel_hand_mask,
            gauge = setup.gauge,
            ref_ant = setup.ref_ant,
        )
        scan_gains[:, s, :, :] .= gains_scan
        scan_solved[:, s, :, :] .= solved
    end

    return build_bandpass_solver_state(setup, gains_template, scan_gains, scan_solved)
end

function initialize_bandpass_state(setup::BandpassSolverSetup, initializer::RandomBandpassInitializer)
    # gains_template: (Frequency, Ant, Feed). scan_gains: (Frequency, Ti, Ant, Feed).
    data = setup.data
    nchan, nti, _, _ = size(parent(data.vis))
    nant = length(data.antennas)
    rng = initializer.rng

    gains_template = exp.(initializer.amplitude_sigma .* randn(rng, nchan, nant, 2)) .* cis.(initializer.phase_sigma .* randn(rng, nchan, nant, 2))
    support_template = antenna_feed_support_weights(data.weights, setup.bl_pairs, setup.pol_products, nant)
    apply_bandpass_gauge!(gains_template, support_template, setup.gauge)

    scan_gains = repeat(reshape(gains_template, nchan, 1, nant, 2), 1, nti, 1, 1)
    if initializer.scan_perturbation > 0
        scan_gains .*= exp.(initializer.scan_perturbation .* randn(rng, nchan, nti, nant, 2)) .* cis.(initializer.scan_perturbation .* randn(rng, nchan, nti, nant, 2))
    end
    for s in 1:nti
        support_scan = antenna_feed_support_weights(view(data.weights, :, s, :, :), setup.bl_pairs, setup.pol_products, nant)
        apply_bandpass_gauge!(view(scan_gains, :, s, :, :), support_scan, setup.gauge)
    end
    scan_solved = trues(nchan, nti, nant, 2)

    return build_bandpass_solver_state(setup, gains_template, scan_gains, scan_solved)
end

initialize_bandpass_state(setup::BandpassSolverSetup) = initialize_bandpass_state(setup, RatioBandpassInitializer())

bandpass_state_objective(state::BandpassSolverState) = state.template_objective + sum(state.scan_objectives)

function fit_bandpass_source_coherencies(setup::BandpassSolverSetup, gains = nothing)
    isnothing(gains) && error("gains must be provided")
    source = allocate_source_coherencies(setup.data.vis)
    solve_source_coherencies!(source, gains, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_products)
    return source
end

function compute_bandpass_model_and_residuals(setup::BandpassSolverSetup, gains, source)
    # V/W layout: (Frequency, Ti, Baseline, Pol). gains: (Frequency, Ti, Ant, Feed).
    V = setup.data.vis
    W = setup.data.weights
    model = fill(NaN + NaN * im, size(V))
    residual = similar(model)

    for pol in axes(V, 4), bi in axes(V, 3), s in axes(V, 2)
        a, b = setup.bl_pairs[bi]
        fa, fb = correlation_feed_pair(setup.pol_products[pol])
        src = source[s, bi, fa, fb]
        @inbounds for c in axes(V, 1)
            v = V[c, s, bi, pol]
            w = W[c, s, bi, pol]
            (w > 0 && isfinite(w) && isfinite(real(v)) && isfinite(imag(v))) || continue

            m = gains[c, s, a, fa] * src * conj(gains[c, s, b, fb])
            model[c, s, bi, pol] = m
            residual[c, s, bi, pol] = v - m
        end
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

    # residual / weights layout: (Frequency, Ti, Baseline, Pol). Slice
    # along Frequency (dim 1) and Ti (dim 2).
    if by == :baseline
        for (bi, (a, b)) in enumerate(setup.bl_pairs), pol in eachindex(setup.pol_products)
            stats = summarize_bandpass_residual_block(
                view(residual, :, :, bi, pol),
                view(setup.data.weights, :, :, bi, pol)
            )
            isnothing(stats) && continue
            push!(
                rows, merge(
                    (
                        baseline = string(setup.data.antennas.name[a], "-", setup.data.antennas.name[b]),
                        pol = setup.pol_products[pol],
                    ), stats
                )
            )
        end
    elseif by == :scan_baseline
        for s in axes(setup.data.vis, 2), (bi, (a, b)) in enumerate(setup.bl_pairs), pol in eachindex(setup.pol_products)
            stats = summarize_bandpass_residual_block(
                view(residual, :, s, bi, pol),
                view(setup.data.weights, :, s, bi, pol)
            )
            isnothing(stats) && continue
            push!(
                rows, merge(
                    (
                        scan = s,
                        baseline = string(setup.data.antennas.name[a], "-", setup.data.antennas.name[b]),
                        pol = setup.pol_products[pol],
                    ), stats
                )
            )
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
    chi2 = joint_bandpass_objective(gains, merged_source, setup.data.vis, setup.data.weights, setup.bl_pairs, setup.pol_products)

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
        setup.pol_products
    )
    dof = max(stats.nreal - nparams, 1)
    return merge(
        stats, (
            nparams = nparams,
            dof = dof,
            reduced_chi2 = stats.chi2 / dof,
        )
    )
end

function refine_bandpass!(
        setup::BandpassSolverSetup,
        state::BandpassSolverState,
        refinement::BandpassALS = BandpassALS()
    )
    refinement.iterations > 0 || return state

    if refinement.refine_template
        refine_joint_bandpass_als!(
            state.gains_template, nothing,
            setup.data.vis, setup.data.weights,
            setup.bl_pairs, setup.pol_products,
            setup.station_models, setup.channel_freqs,
            setup.parallel_pols, setup.parallel_hand_mask;
            max_iterations = refinement.iterations, tolerance = refinement.tolerance,
            gauge = setup.gauge,
        )
    end

    if refinement.refine_scans
        # scan_gains/scan_solved: (Frequency, Ti, Ant, Feed). data.vis/weights:
        # (Frequency, Ti, Baseline, Pol). Slice on Ti (dim 2).
        for s in axes(state.scan_gains, 2)
            refine_joint_bandpass_als!(
                view(state.scan_gains, :, s, :, :),
                view(state.scan_solved, :, s, :, :),
                view(setup.data.vis, :, s, :, :),
                view(setup.data.weights, :, s, :, :),
                setup.bl_pairs,
                setup.pol_products,
                setup.station_models, setup.channel_freqs,
                setup.parallel_pols, setup.parallel_hand_mask;
                max_iterations = refinement.iterations,
                tolerance = refinement.tolerance,
                gauge = setup.gauge,
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
    state.als_iterations_completed += refinement.iterations
    return update_state_sources_and_objectives!(setup, state)
end


function finalize_bandpass_state(
        setup::BandpassSolverSetup,
        state::BandpassSolverState;
        spw_name::AbstractString = "spw_0",
    )
    # state.scan_gains layout: (Frequency, Ti, Ant, Feed).
    data = setup.data
    nchan = size(state.scan_gains, 1)
    nti = size(state.scan_gains, 2)
    nant = size(state.scan_gains, 3)

    gains_template = copy(state.gains_template)
    scan_gains = copy(state.scan_gains)
    merged_gains = repeat(reshape(gains_template, nchan, 1, nant, 2), 1, nti, 1, 1)

    sanitize_and_gauge_gains!(
        gains_template, data.weights, setup.bl_pairs,
        setup.parallel_pols, setup.parallel_hand_mask;
        ant_names = data.antennas.name,
        context = "template",
        gauge = setup.gauge,
    )

    for s in 1:nti
        sanitize_and_gauge_gains!(
            view(scan_gains, :, s, :, :),
            view(data.weights, :, s, :, :),
            setup.bl_pairs,
            setup.parallel_pols, setup.parallel_hand_mask;
            ant_names = data.antennas.name,
            context = string("scan ", s),
            gauge = setup.gauge,
        )
    end

    assemble_bandpass_state_gains!(
        merged_gains, gains_template, scan_gains, state.scan_solved, setup.phase_variable_mask, setup.amplitude_variable_mask
    )

    apply_bandpass_gauge!(
        merged_gains,
        amplitude_support_weights(data.weights, setup.bl_pairs, nant, setup.parallel_pols, setup.parallel_hand_mask),
        setup.gauge,
    )

    return wrap_gain_solutions(
        merged_gains, data;
        spw_name = String(spw_name),
    )
end

"""
    solve_bandpass(uvset::UVSet, ref_ant; ...) -> OrderedDict{String, DimArray}

Per-SPW orchestrator. Groups leaves by `spw_name`, builds a
`BandpassDataset` per group via `_to_bandpass_dataset`, runs the
inner solver, and collects per-SPW gain `DimArray`s into an ordered
Dict keyed by `spw_name`. Single-SPW UVSets return a length-1 Dict.

Within an SPW group, `union_antennas` (called by
`_to_bandpass_dataset`) enforces antenna-metadata consistency across
multi-subarray leaves; conflicts error with a clear message.
"""
function solve_bandpass(
        uvset::UVSet, ref_ant;
        min_baselines = 3, station_models = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge(),
        joint_als_iterations = 8, joint_als_tolerance = 1.0e-6,
    )
    groups = _group_leaves_by_spw(uvset)
    isempty(groups) && error("solve_bandpass: UVSet has no leaves")

    sols = OrderedDict{String, DimensionalData.DimArray}()
    for (spw_name, leaf_keys) in groups
        sub_uvset = _uvset_with_branches(uvset, leaf_keys)
        avg = _to_bandpass_dataset(sub_uvset)
        sols[spw_name] = solve_bandpass(
            avg, ref_ant;
            min_baselines = min_baselines, station_models = station_models,
            gauge = gauge,
            joint_als_iterations = joint_als_iterations,
            joint_als_tolerance = joint_als_tolerance,
            spw_name = spw_name,
        )
    end
    return sols
end

solve_bandpass(uvset::UVSet, source::AbstractString, ref_ant; kwargs...) =
    solve_bandpass(select_source(uvset, source), ref_ant; kwargs...)

# Per-leaf solve: `key` is a branch key like `:M3C273_spw_0_scan_1`.
function solve_bandpass(uvset::UVSet, key::Symbol, ref_ant; kwargs...)
    branches_d = DimensionalData.branches(uvset)
    haskey(branches_d, key) || error("UVSet has no partition $key")
    info = DimensionalData.metadata(branches_d[key])
    sub = select_partition(uvset; source = info.source_name, scan = info.scan_name)
    return solve_bandpass(sub, ref_ant; kwargs...)
end

function solve_bandpass(
        avg::BandpassDataset, ref_ant;
        min_baselines = 3, station_models = nothing,
        gauge::AbstractBandpassGauge = ZeroMeanBandpassGauge(),
        joint_als_iterations = 8, joint_als_tolerance = 1.0e-6,
        spw_name::AbstractString = "spw_0",
    )
    setup = prepare_bandpass_solver(
        avg,
        ref_ant;
        min_baselines = min_baselines,
        station_models = station_models,
        gauge = gauge,
    )
    @info "Bandpass solver setup: $(length(setup.channel_freqs)) channels × $(length(setup.data.antennas)) antennas, gauge=$(typeof(setup.gauge))"

    state = initialize_bandpass_state(setup)
    joint_als_iterations > 0 && refine_bandpass!(
        setup,
        state,
        BandpassALS(iterations = joint_als_iterations, tolerance = joint_als_tolerance),
    )

    return finalize_bandpass_state(setup, state; spw_name = spw_name)
end
