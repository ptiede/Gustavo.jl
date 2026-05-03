"""
    correlation_feed_pair(label::AbstractString) -> Tuple{Int, Int}

Map a generic correlation-product label `"PP"`/`"PQ"`/`"QP"`/`"QQ"` to the
`(feed_a_idx, feed_b_idx)` pair (each ∈ {1, 2}) that the visibility
`V[a, b, p]` relates to per-antenna gains via
`V[a, b, p] = g_a[feed_a] · S · conj(g_b[feed_b])`.
"""
function correlation_feed_pair(label::AbstractString)
    length(label) == 2 || error("correlation_feed_pair: expected 2-char label, got \"$label\"")
    a = _feed_index(label[1])
    b = _feed_index(label[2])
    return a, b
end
_feed_index(c::Char) = c == 'P' ? 1 : c == 'Q' ? 2 : error("Unsupported feed letter '$c' (expected 'P' or 'Q')")

"""
    is_parallel_hand(label::AbstractString) -> Bool

True for the parallel-hand correlations `"PP"` and `"QQ"`; false for
`"PQ"` and `"QP"`.
"""
is_parallel_hand(label::AbstractString) =
    length(label) == 2 && label[1] == label[2]

"""
    same_feed_label(a, b) -> Bool

True iff two `PolTypes` (RPol/LPol/XPol/YPol) carry the same feed label.
Used by `build_parallel_hand_mask` to determine, per-baseline, whether
the two antennas' feed-1 (resp. feed-2) form a parallel-hand correlation.
"""
same_feed_label(a::T, b::T) where {T <: PolTypes} = true
same_feed_label(::PolTypes, ::PolTypes) = false

"""
    parallel_hand_indices(pol_products) -> Tuple{Int, Int}

Indices of `"PP"` and `"QQ"` in `pol_products`. Errors if either is
missing.
"""
function parallel_hand_indices(pol_products)
    pp = findfirst(==("PP"), pol_products)
    qq = findfirst(==("QQ"), pol_products)
    isnothing(pp) || isnothing(qq) &&
        error("Parallel-hand products not found in pol_products=$(collect(pol_products))")
    return pp, qq
end

"""
    cross_hand_indices(pol_products) -> NamedTuple{(:pq, :qp), Tuple{Int, Int}} or nothing

Indices of the two cross-hand correlations `"PQ"` and `"QP"`, or `nothing`
if either is absent.
"""
function cross_hand_indices(pol_products)
    pq = findfirst(==("PQ"), pol_products)
    qp = findfirst(==("QP"), pol_products)
    !isnothing(pq) && !isnothing(qp) && return (; pq, qp)
    return nothing
end

# `build_parallel_hand_mask(antennas, bl_pairs)` answers, for each
# `(baseline, feed_index ∈ {1, 2})`, whether antenna A's feed-`feed_index`
# and antenna B's feed-`feed_index` carry the same nominal label (so the
# corresponding parallel-hand correlation on this baseline is meaningful).
# Mixed-feed arrays produce `false` for baselines crossing different
# nominal labels; uniform-feed arrays produce all `true`.
function build_parallel_hand_mask(antennas, bl_pairs)
    mask = falses(length(bl_pairs), 2)
    nb = antennas.nominal_basis  # Vector{NTuple{2, PolTypes}} via StructArray forwarding
    for (bi, (a, b)) in enumerate(bl_pairs)
        mask[bi, 1] = same_feed_label(nb[a][1], nb[b][1])
        mask[bi, 2] = same_feed_label(nb[a][2], nb[b][2])
    end
    return mask
end

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
# weighted least squares and the AIPS UVData convention (Memo 117: visibility
# weights are in Jy⁻² = variance⁻¹).
#
# For y_i = A_i x + ε_i with independent ε_i ~ N(0, σ_i²) the MLE is
#   x* = (Aᵀ W A)⁻¹ Aᵀ W y,   W = diag(1/σ_i²).
# To solve this with QR we need S with SᵀS = W; since W is diagonal-positive,
# S = diag(√W_ii) = diag(1/σ_i), so we scale each row by √(weight) before
# handing to QR. **Callers pass inverse variance and never take the sqrt
# themselves.**
_row_scale(inv_variances) = sqrt.(inv_variances)

# Promote the WLS triple `(A, b, inv_variances)` to a single common element
# type. The design matrix `A` is built in Float64 by `design_matrices`, but
# `b` (log-amplitudes / phases) and `inv_variances` (UVData weights) come in
# at Float32 since AIPS Memo 117 stores weights as `1E`. LinearSolve's QR
# `ldiv!` rejects a Float64 factorization against a Float32 RHS, so we have
# to align eltypes up front.
function _promote_lsq(A, b, inv_variances)
    T = promote_type(eltype(A), eltype(b), real(eltype(inv_variances)))
    return convert(AbstractMatrix{T}, A), convert(AbstractVector{T}, b),
        convert(AbstractVector{T}, inv_variances)
end

function weighted_least_squares(A, b, inv_variances)
    A, b, inv_variances = _promote_lsq(A, b, inv_variances)
    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    return solve(LinearProblem(Aw, bw), QRFactorization()).u
end

function weighted_regularized_least_squares(A, b, inv_variances, penalties)
    isempty(penalties) && return weighted_least_squares(A, b, inv_variances)
    all(≤(0), penalties) && return weighted_least_squares(A, b, inv_variances)

    A, b, inv_variances = _promote_lsq(A, b, inv_variances)
    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    reg = sqrt.(penalties)
    Areg = Matrix(Diagonal(reg))
    breg = zeros(eltype(Aw), size(A, 2))
    return solve(LinearProblem(vcat(Aw, Areg), vcat(bw, breg)), QRFactorization()).u
end

function weighted_constrained_least_squares(A, b, inv_variances, C, d; constraint_weight = 1.0e6)
    isempty(C) && return weighted_least_squares(A, b, inv_variances)

    A, b, inv_variances = _promote_lsq(A, b, inv_variances)
    sw = _row_scale(inv_variances)
    Aw = A .* reshape(sw, :, 1)
    bw = b .* sw
    Acon = constraint_weight .* C
    bcon = constraint_weight .* d
    return solve(LinearProblem(vcat(Aw, Acon), vcat(bw, bcon)), QRFactorization()).u
end

"""
    unwrap_phase_track(phases; weights=nothing) -> Vector

Unwrap a phase track to remove ±2π discontinuities between adjacent
finite samples. The walk seeds from a single reference index, picked
internally as `argmax(weights[finite])` when `weights` is provided, or
the first finite phase otherwise. The returned track is bit-identical
to the prior `unwrap_phase_track(phases, ref_idx)` form when `ref_idx`
is the highest-weight finite channel.

This is an algorithmic anchor only — downstream gauge code should
always center via a weighted mean (or remove a per-feed factor in the
`ReferenceAntennaBandpassGauge` case), not pin the unwrap reference.
"""
function unwrap_phase_track(phases; weights = nothing)
    unwrapped = copy(phases)
    n = length(unwrapped)
    n == 0 && return unwrapped

    finite = isfinite.(unwrapped)
    any(finite) || return unwrapped

    ref_idx = if weights === nothing
        findfirst(finite)
    else
        @assert length(weights) == n "weights length must match phases length"
        # Choose the highest-weight finite channel as the unwrap seed.
        best_w = -Inf
        best_i = findfirst(finite)
        @inbounds for i in 1:n
            (finite[i] && isfinite(weights[i])) || continue
            if weights[i] > best_w
                best_w = weights[i]
                best_i = i
            end
        end
        best_i
    end
    isnothing(ref_idx) && return unwrapped

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
