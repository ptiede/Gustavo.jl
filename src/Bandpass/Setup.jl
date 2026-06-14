polarization_feeds(data::BandpassDataset, pol_index::Integer) =
    correlation_feed_pair(pol_products(data)[pol_index])
polarization_feeds(uvset::UVSet, pol_index::Integer) =
    correlation_feed_pair(pol_products(uvset)[pol_index])

function best_ref_channel(data::BandpassDataset)
    # data.weights layout: (Frequency, Ti, Baseline, Pol). Sum the
    # parallel-hand pol slice along (Ti, Baseline, Pol) → length-nchan
    # vector; argmax picks the strongest channel.
    pp, qq = parallel_hand_indices(pol_products(data))
    pols = [pp, qq]
    return argmax(vec(sum(view(parent(data.weights), :, :, :, pols), dims = (2, 3, 4))))
end

function corrected_visibility(V, gains, pol_products, bi, a, b, pol, s, c)
    # V layout: (Frequency, Ti, Baseline, Pol). gains: (Frequency, Ti, Ant, Feed).
    fa, fb = correlation_feed_pair(pol_products[pol])
    return V[c, s, bi, pol] / (gains[c, s, a, fa] * conj(gains[c, s, b, fb]))
end


function choose_local_phase_reference(active_ants, gauge, station_models, connectivity, feed, ref_ant = nothing)
    if gauge isa ReferenceAntennaBandpassGauge
        gauge.ref_ant ∈ active_ants && return gauge.ref_ant
    end

    # The user-supplied `ref_ant` takes precedence as the local phase reference
    # when it is among the active antennas. Otherwise, prefer antennas whose
    # phase model is *not* per-scan (so the per-channel pin propagates cleanly
    # through the gauge step). Fall back to active_ants only if no station has
    # a stable phase model. Within the candidate pool we prefer the *least*
    # connected antenna so the most informative tracks remain free.
    !isnothing(ref_ant) && ref_ant ∈ active_ants && return ref_ant

    stable_active = [ant for ant in active_ants if !phase_is_per_scan(station_models[ant], feed)]
    candidates = isempty(stable_active) ? active_ants : stable_active
    isempty(candidates) && error("No active antennas available for local phase reference")

    scores = [connectivity[ant] for ant in candidates]
    return candidates[argmax(scores)]
end

function choose_phase_reference(avg::BandpassDataset, variable_ants)
    # W layout: (Frequency, Ti, Baseline, Pol). Stride-1 inner loop over
    # Frequency.
    W = avg.weights
    nant = length(avg.antennas)
    blocked = falses(nant)
    blocked[variable_ants] .= true
    pols = parallel_hand_indices(pol_products(avg))
    scores = zeros(Float64, nant)

    for pol in pols, bi in axes(W, 3), s in axes(W, 2)
        a, b = avg.baselines.pairs[bi]
        @inbounds for c in axes(W, 1)
            w = W[c, s, bi, pol]
            w > 0 || continue
            blocked[a] || (scores[a] += w)
            blocked[b] || (scores[b] += w)
        end
    end

    phase_ref = argmax(scores)
    scores[phase_ref] > 0 || error("No stable antenna available for phase reference")
    return phase_ref
end

choose_phase_reference(uvset::UVSet, variable_ants) =
    choose_phase_reference(_to_bandpass_dataset(uvset), variable_ants)

function build_station_models(
        ant_names, station_model_map;
        default::StationGainModel,
    )
    default_model = validate_station_gain_model(default)
    station_models = StationGainModel[default_model for _ in ant_names]
    for (name, model) in station_model_map
        ant_idx = findfirst(==(name), ant_names)
        isnothing(ant_idx) && error("Unknown station in station_model_map: $name")
        station_models[ant_idx] = validate_station_gain_model(model)
    end
    return station_models
end
