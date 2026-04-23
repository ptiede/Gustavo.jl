stokes_feed_pair(code::Integer) = code == -1 ? (1, 1) :
    code == -2 ? (2, 2) :
    code == -3 ? (1, 2) :
    code == -4 ? (2, 1) :
    error("Unsupported Stokes code for feed mapping: $code")

feed_pair_label(feeds::Tuple{<:Integer,<:Integer}) = string(feeds[1], feeds[2])
polarization_label(code::Integer) = feed_pair_label(stokes_feed_pair(code))

function parallel_hand_indices(pol_codes)
    rr = findfirst(==(-1), pol_codes)
    ll = findfirst(==(-2), pol_codes)
    (isnothing(rr) || isnothing(ll)) && error("Parallel-hand 11/22 products not found in pol_codes=$(collect(pol_codes))")
    return rr, ll
end

function cross_hand_indices(pol_codes)
    rl = findfirst(==(-3), pol_codes)
    lr = findfirst(==(-4), pol_codes)
    (isnothing(rl) || isnothing(lr)) && return nothing
    return (; rl, lr)
end

polarization_feeds(data::UVData, pol_index::Integer) = stokes_feed_pair(data.pol_codes[pol_index])

function best_ref_channel(data::UVData)
    rr, ll = parallel_hand_indices(data.pol_codes)
    pols = [rr, ll]
    return argmax(vec(sum(data.weights[:, :, pols, :], dims=(1, 2, 3))))
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

function weighted_least_squares(A, b, weights)
    Aw = A .* reshape(weights, :, 1)
    bw = b .* weights
    return solve(LinearProblem(Aw, bw), QRFactorization()).u
end

function weighted_regularized_least_squares(A, b, weights, penalties)
    isempty(penalties) && return weighted_least_squares(A, b, weights)
    all(≤(0), penalties) && return weighted_least_squares(A, b, weights)

    Aw = A .* reshape(weights, :, 1)
    bw = b .* weights
    reg = sqrt.(penalties)
    Areg = Matrix(Diagonal(reg))
    breg = zeros(size(A, 2))
    return solve(LinearProblem(vcat(Aw, Areg), vcat(bw, breg)), QRFactorization()).u
end

function unwrap_phase_track(phases, ref_idx=1)
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

function phase_relative_to_ref(phases, ref_idx=1)
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

feed_node(ant, feed, nant) = (feed - 1) * nant + ant

function decode_feed_node(node, nant)
    ant = mod1(node, nant)
    feed = fld(node - 1, nant) + 1
    return ant, feed
end

function choose_local_phase_reference(active_ants, phase_ref_ant, station_models, connectivity)
    phase_ref_ant ∈ active_ants && return phase_ref_ant

    stable_active = [ant for ant in active_ants if !is_per_scan(station_models[ant].segmentation.time)]
    candidates = isempty(stable_active) ? active_ants : stable_active
    isempty(candidates) && error("No active antennas available for local phase reference")

    scores = [connectivity[ant] for ant in candidates]
    return candidates[argmax(scores)]
end

function choose_phase_reference(avg::UVData, variable_ants)
    W = avg.weights
    nant = length(avg.ant_names)
    blocked = falses(nant)
    blocked[variable_ants] .= true
    pols = parallel_hand_indices(avg.pol_codes)
    scores = zeros(Float64, nant)

    for s in axes(W, 1), bi in axes(W, 2), pol in pols, c in axes(W, 4)
        a, b = avg.bl_pairs[bi]
        w = W[s, bi, pol, c]
        w > 0 || continue
        blocked[a] || (scores[a] += w)
        blocked[b] || (scores[b] += w)
    end

    phase_ref = argmax(scores)
    scores[phase_ref] > 0 || error("No stable antenna available for phase reference")
    return phase_ref
end

function build_station_models(ant_names, station_model_map;
        default=StationBandpassModel())
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
    reference_summary_phase = effective_bandpass_model_label(model.reference.phase, model.segmentation.frequency)
    relative_summary_phase = effective_bandpass_model_label(model.relative.phase, model.segmentation.frequency)
    reference_summary_amp = effective_bandpass_model_label(model.reference.amplitude, model.segmentation.frequency)
    relative_summary_amp = effective_bandpass_model_label(model.relative.amplitude, model.segmentation.frequency)
    return string(name,
        ": time=", time_segmentation_label(model.segmentation.time),
        " ref=", reference_feed_label(model.reference_feed),
        " abs(phase=", reference_summary_phase, " amp=", reference_summary_amp, ")",
        " rel(phase=", relative_summary_phase, " amp=", relative_summary_amp, ")")
end
