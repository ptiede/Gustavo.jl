function parallel_hand_indices(pol_codes)
    rr = findfirst(==(-1), pol_codes)
    ll = findfirst(==(-2), pol_codes)
    !isnothing(rr) && !isnothing(ll) && return rr, ll
    xx = findfirst(==(-5), pol_codes)
    yy = findfirst(==(-6), pol_codes)
    !isnothing(xx) && !isnothing(yy) && return xx, yy
    error("Parallel-hand products not found in pol_codes=$(collect(pol_codes))")
end

function cross_hand_indices(pol_codes)
    rl = findfirst(==(-3), pol_codes)
    lr = findfirst(==(-4), pol_codes)
    !isnothing(rl) && !isnothing(lr) && return (; rl, lr)
    xy = findfirst(==(-7), pol_codes)
    yx = findfirst(==(-8), pol_codes)
    !isnothing(xy) && !isnothing(yx) && return (; xy, yx)
    return nothing
end

function build_parallel_hand_mask(antennas, bl_pairs)
    mask = falses(length(bl_pairs), 2)
    for (bi, (a, b)) in enumerate(bl_pairs)
        mask[bi, 1] = same_feed_type(antennas.feed_a[a], antennas.feed_a[b])
        mask[bi, 2] = same_feed_type(antennas.feed_b[a], antennas.feed_b[b])
    end
    return mask
end

polarization_feeds(data::UVData, pol_index::Integer) = stokes_feed_pair(data.metadata.pol_codes[pol_index])

function best_ref_channel(data::UVData)
    rr, ll = parallel_hand_indices(data.metadata.pol_codes)
    pols = [rr, ll]
    return argmax(vec(sum(data.weights[:, :, pols, :], dims = (1, 2, 3))))
end

function design_matrices(bl_pairs, nant)
    nbl = length(bl_pairs)
    A_amp = zeros(Float64, nbl, nant)
    A_phase = zeros(Float64, nbl, nant)
    for (i, (a, b)) in enumerate(bl_pairs)
        A_amp[i, a] = 1.0
        A_amp[i, b] = 1.0
        A_phase[i, a] = 1.0
        A_phase[i, b] = -1.0
    end
    return A_amp, A_phase
end

function weighted_phase_mean(phases, weights)
    sin_sum = sum(weights .* sin.(phases))
    cos_sum = sum(weights .* cos.(phases))
    return atan(sin_sum, cos_sum)
end

function weighted_complex_correction(samples, weights)
    isempty(samples) && return nothing
    sum(weights) > 0 || return nothing
    log_amp = sum(weights .* log.(abs.(samples))) / sum(weights)
    phase = weighted_phase_mean(angle.(samples), weights)
    return exp(log_amp) * cis(phase)
end

# Convention across Gustavo: a "weight" is always an *inverse variance*
# (precision, 1/σ²), matching both the Gaussian-likelihood derivation of
# weighted least squares and the AIPS UVFITS convention (Memo 117: visibility
# weights are in Jy⁻² = variance⁻¹).
#
# For y_i = A_i x + ε_i with independent ε_i ~ N(0, σ_i²) the MLE is
#   x* = (Aᵀ W A)⁻¹ Aᵀ W y,   W = diag(1/σ_i²).
# To solve this with QR we need S with SᵀS = W; since W is diagonal-positive,
# S = diag(√W_ii) = diag(1/σ_i), so we scale each row by √(weight) before
# handing to QR. **Callers pass inverse variance and never take the sqrt
# themselves.**
_row_scale(inv_variances) = sqrt.(inv_variances)

function weighted_least_squares(A, b, inv_variances)
    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    return solve(LinearProblem(Aw, bw), QRFactorization()).u
end

function weighted_regularized_least_squares(A, b, inv_variances, penalties)
    isempty(penalties) && return weighted_least_squares(A, b, inv_variances)
    all(≤(0), penalties) && return weighted_least_squares(A, b, inv_variances)

    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    reg = sqrt.(penalties)
    Areg = Matrix(Diagonal(reg))
    breg = zeros(size(A, 2))
    return solve(LinearProblem(vcat(Aw, Areg), vcat(bw, breg)), QRFactorization()).u
end

function weighted_constrained_least_squares(A, b, inv_variances, C, d; constraint_weight = 1.0e6)
    isempty(C) && return weighted_least_squares(A, b, inv_variances)

    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    Acon = constraint_weight .* C
    bcon = constraint_weight .* d
    return solve(LinearProblem(vcat(Aw, Acon), vcat(bw, bcon)), QRFactorization()).u
end

function unwrap_phase_track(phases, ref_idx = 1)
    unwrapped = copy(phases)
    n = length(unwrapped)
    (1 <= ref_idx <= n) || return unwrapped

    if !isfinite(unwrapped[ref_idx])
        ref_idx = findfirst(isfinite, unwrapped)
        isnothing(ref_idx) && return unwrapped
    end

    last = unwrapped[ref_idx]
    for i in (ref_idx + 1):n
        isfinite(unwrapped[i]) || continue
        unwrapped[i] += 2π * round((last - unwrapped[i]) / (2π))
        last = unwrapped[i]
    end

    last = unwrapped[ref_idx]
    for i in (ref_idx - 1):-1:1
        isfinite(unwrapped[i]) || continue
        unwrapped[i] += 2π * round((last - unwrapped[i]) / (2π))
        last = unwrapped[i]
    end

    return unwrapped
end

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

function corrected_visibility(V, gains, pol_codes, bi, a, b, pol, s, c)
    fa, fb = stokes_feed_pair(pol_codes[pol])
    return V[s, bi, pol, c] / (gains[s, a, fa, c] * conj(gains[s, b, fb, c]))
end


function choose_local_phase_reference(active_ants, gauge, station_models, connectivity, feed)
    if gauge isa ReferenceAntennaBandpassGauge
        gauge.ref_ant ∈ active_ants && return gauge.ref_ant
    end

    stable_active = [ant for ant in active_ants if !phase_is_per_scan(station_models[ant], feed)]
    candidates = isempty(stable_active) ? active_ants : stable_active
    isempty(candidates) && error("No active antennas available for local phase reference")

    scores = [connectivity[ant] for ant in candidates]
    return candidates[argmax(scores)]
end

function choose_phase_reference(avg::UVData, variable_ants)
    W = avg.weights
    nant = length(avg.antennas)
    blocked = falses(nant)
    blocked[variable_ants] .= true
    pols = parallel_hand_indices(avg.metadata.pol_codes)
    scores = zeros(Float64, nant)

    for s in axes(W, 1), bi in axes(W, 2), pol in pols, c in axes(W, 4)
        a, b = avg.baselines.pairs[bi]
        w = W[s, bi, pol, c]
        w > 0 || continue
        blocked[a] || (scores[a] += w)
        blocked[b] || (scores[b] += w)
    end

    phase_ref = argmax(scores)
    scores[phase_ref] > 0 || error("No stable antenna available for phase reference")
    return phase_ref
end

function build_station_models(
        ant_names, station_model_map;
        default = StationBandpassModel()
    )
    default_model = validate_station_bandpass_model(default)
    station_models = StationBandpassModel[default_model for _ in ant_names]
    for (name, model) in station_model_map
        ant_idx = findfirst(==(name), ant_names)
        isnothing(ant_idx) && error("Unknown station in station_model_map: $name")
        station_models[ant_idx] = validate_station_bandpass_model(model)
    end
    return station_models
end

function station_model_summary(name, model)
    reference_summary_phase = effective_bandpass_model_label(model.reference.phase.model, model.reference.phase.segmentation.frequency)
    relative_summary_phase = effective_bandpass_model_label(model.relative.phase.model, model.relative.phase.segmentation.frequency)
    reference_summary_amp = effective_bandpass_model_label(model.reference.amplitude.model, model.reference.amplitude.segmentation.frequency)
    relative_summary_amp = effective_bandpass_model_label(model.relative.amplitude.model, model.relative.amplitude.segmentation.frequency)
    return string(
        name,
        " ref=", reference_feed_label(model.reference_feed),
        " abs(phase=", reference_summary_phase,
        ", phase_time=", time_segmentation_label(model.reference.phase.segmentation.time),
        ", amp=", reference_summary_amp,
        ", amp_time=", time_segmentation_label(model.reference.amplitude.segmentation.time), ")",
        " rel(phase=", relative_summary_phase,
        ", phase_time=", time_segmentation_label(model.relative.phase.segmentation.time),
        ", amp=", relative_summary_amp,
        ", amp_time=", time_segmentation_label(model.relative.amplitude.segmentation.time), ")"
    )
end
