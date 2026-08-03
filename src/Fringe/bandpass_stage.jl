# ── Bandpass stage: per-channel station phase/log-amp over scan windows ───────
#
# The carved-out bandpass stage of the composable pipeline: the per-scan
# residual accumulation ([`accumulate_bandpass!`](@ref)) and the two per-channel
# solves ([`solve_phase_bandpass!`](@ref) / [`solve_amp_bandpass!`](@ref)) —
# descended verbatim from the monolithic solver (deleted at M5), retargeted
# from its concat cube to a scan `DimStack` where they touch data. The
# BandpassEstimator step visits each selected scan (refine → accumulate → return
# the scan's contribution) and its `finish_pass!` folds the contributions in
# GROUP-INDEX order — deterministic at ANY concurrency (unlike the monolith's
# ntasks-dependent chunk fold; the two agree to float-rounding, gated at
# rtol ≤ 1e-12).
#
# The graph/solve helpers (`_ObsRow`, `_solve_observable`, `_track_noise2`,
# `_node`, `_chi_sign`) live in stationize.jl/adhoc.jl; the amp-smoother family
# (`AbstractBandpassSmoother`, `_fit_bandpass_segment`) lives below.

# ── Amplitude-bandpass smoothers (pluggable estimators) ───────────────────────────
#
# The per-(station, feed) log-amp bandpass is solved from the SUM closure
# `log|V̄_ab(ν)| = la_a(ν) + lb_b(ν)` over each spw (a +1/+1, signless-Laplacian
# incidence — FULL RANK, so no reference state). HOW the per-channel shape is
# estimated — and how low-/no-signal channels are filled — is a pluggable strategy:
# add an `AbstractBandpassSmoother` subtype and a `_fit_bandpass_segment` method
# (at the foot of this file) to extend.
abstract type AbstractBandpassSmoother end

"""
    FreeBandpass()

Free per-channel closure: one independent WLS per channel, no regularisation.
Follows the data but does NOT estimate low-/no-signal channels (left at |g| = 1).
The `λ → 0` / `degree → ∞` limit of the others.
"""
struct FreeBandpass <: AbstractBandpassSmoother end

"""
    PolynomialBandpass(degree = 4)

Smooth per-spw polynomial of `degree` in a centred/scaled frequency coordinate, fit
by a single closure WLS (the `PolynomialFreq` design convention). Estimates gaps by
the fit. Assumes the in-spw bandpass is ~a low-order polynomial (smooth passband +
gentle roll-off); a high degree can ring (Runge) at the edges.
"""
struct PolynomialBandpass <: AbstractBandpassSmoother
    degree::Int
    function PolynomialBandpass(degree::Integer = 4)
        degree >= 1 || error("PolynomialBandpass: degree must be ≥ 1")
        return new(Int(degree))
    end
end

"""
    PenalizedBandpass(lambda = 1.0)

Roughness-penalised per-channel bandpass (a Whittaker smoother): a free value per
channel plus a 2nd-difference smoothness penalty of strength `lambda` (relative to
the per-channel data weight). Makes NO shape assumption — follows real structure
where the SNR supports it and smoothly interpolates gaps where it does not.
`lambda → 0` ⇒ [`FreeBandpass`](@ref); large `lambda` ⇒ flat.
"""
struct PenalizedBandpass <: AbstractBandpassSmoother
    lambda::Float64
    function PenalizedBandpass(lambda::Real = 1.0)
        lambda >= 0 || error("PenalizedBandpass: lambda must be ≥ 0")
        return new(Float64(lambda))
    end
end

"""
    accumulate_bandpass!(rbar_bp, wbar_bp, blidx, stack, win::GeometryWindow)

Accumulate one scan window's contribution to the per-(global-baseline, product,
GLOBAL channel) coherent residual `rbar_bp` (and weight `wbar_bp`) for the
bandpass solves, on data already gain-corrected through the pipeline's
transform chain. BEFORE summing over time each AP is counter-rotated by its
OWN band-averaged residual phase, removing the per-AP time phase (residual
rate/drift, and what the adhoc stage would later remove) so the time-average
is coherent and isolates the per-channel SHAPE. `blidx` maps `(a, b) -> row` in
the global baseline table (co-located pairs excluded there never contribute).
"""
# Fresh per-(baseline row, product, GLOBAL channel) bandpass accumulators. They
# carry (Baseline, Pol, Frequency) dims so the accumulate/solve kernels below
# address axes BY NAME (the house style of the Bandpass module) instead of by
# position; indexing stays plain-positional and costs nothing.
function bandpass_accumulators(nbl::Integer, npol::Integer, nchan::Integer)
    d = (Baseline(1:nbl), Pol(1:npol), Frequency(1:nchan))
    return (
        DimensionalData.DimArray(zeros(ComplexF64, nbl, npol, nchan), d),
        DimensionalData.DimArray(zeros(Float64, nbl, npol, nchan), d),
    )
end

function accumulate_bandpass!(
        rbar_bp, wbar_bp, blidx, stack::AbstractDimStack, win::GeometryWindow,
    )
    V = stack[:vis]                                      # the dims-carrying layers —
    W = stack[:weights]                                  # the loops below address axes BY NAME
    bl_pairs = UVData.baselines(stack).pairs
    g_ci = win.chan_idx
    for p in axes(V, Pol)
        for bi in axes(V, Baseline)
            a, b = bl_pairs[bi]
            a == b && continue # autocorrelation skip
            idx = get(blidx, (a, b), 0)
            idx == 0 && continue # baseline doesn't exist so skip
            for tt in axes(V, Ti)
                # Band-averaged residual phase for this AP (the per-AP time phase).
                acc = zero(eltype(V))
                for c in axes(V, Frequency)
                    w = W[c, tt, bi, p]
                    (w > 0 && isfinite(w)) || continue
                    vv = V[c, tt, bi, p]
                    acc += ifelse(isfinite(vv), w * vv, zero(eltype(V)))
                end
                abs(acc) > 0 || continue
                rot = conj(acc) / abs(acc)            # cis(-angle(acc)): de-rotate this AP
                for c in axes(V, Frequency)
                    w = W[c, tt, bi, p]
                    (w > 0 && isfinite(w)) || continue
                    vv = V[c, tt, bi, p]
                    isfinite(vv) || continue
                    gc = g_ci[c]
                    rbar_bp[idx, p, gc] += w * vv * rot
                    wbar_bp[idx, p, gc] += w
                end
            end
        end
    end
    return rbar_bp, wbar_bp
end

"""
    solve_phase_bandpass!(θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan;
                          ref_ant = 1, snr_floor = 1.0)

Solve the per-(station, feed) phase bandpass from the accumulated per-channel
residual and write it into the bandpass component's θ blocks. `plan`'s frequency
segmentation sets the resolution: one solved value per (station, feed, frequency
segment), from the residual of every channel the segment holds. Per segment, the
globally-closing per-feed phase is solved on the (station, feed) graph with a
scale-invariant SNR gate; the per-segment χ's band-structure (the reference's
R–L phase shape, parked in χ by the per-segment EVPA pin) is re-gauged into the
feed-2 block, and each (station, feed) track is referenced to its circular-mean
phase over segments (zero net applied phase).
"""
function solve_phase_bandpass!(
        θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan;
        ref_ant::Integer = 1, snr_floor::Real = 1.0,
    )
    nbl, npol, nchan = size(rbar_bp)
    feeds = [correlation_feed_pair(p) for p in pol_products]
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in 1:nbl, p in 1:npol]
    segs = segment_groups(plan.fseg_id, length(plan.nchan_seg))
    nseg = length(segs)
    phase = fill(NaN, nant, 2, nseg)
    chis = fill(NaN, nseg)
    for (fs, chans) in enumerate(segs)
        rows = _ObsRow[]
        for bi in axes(rbar_bp, Baseline), p in axes(rbar_bp, Pol)
            a, b = bl_pairs[bi]
            a == b && continue
            r, w, w2 = _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            snr2 = _segment_snr2(r, w, w2, noise2[bi, p])
            snr2 >= snr_floor^2 || continue
            fa, fb = feeds[p]
            push!(rows, _ObsRow(a, b, fa, fb, angle(r), snr2, _chi_sign(fa, fb)))
        end
        ph, chi, _, _ = _solve_observable(rows, nant, ref_ant; use_chi = true, rewrap = 0)
        phase[:, :, fs] .= ph
        chis[fs] = chi
    end

    # Reassign χ's band-structure into the feed-2 block: the per-segment EVPA pin
    # parked the reference's R–L phase shape in χ̂_s; subtract δ_s = χ̂_s − χ̄ from
    # all feed-2 nodes so only the band-constant χ̄ stays conventional. Segments
    # with no finite χ̂ (no cross-hand row cleared the SNR gate) are left
    # unshifted — the R–L alignment is unobservable there.
    acc_chi = zero(eltype(rbar_bp))
    for fs in 1:nseg
        isfinite(chis[fs]) && (acc_chi += cis(chis[fs]))
    end
    if abs(acc_chi) > 0
        chibar = angle(acc_chi)
        for fs in 1:nseg
            isfinite(chis[fs]) || continue
            δ = rem2pi(chis[fs] - chibar, RoundNearest)
            @inbounds for a in 1:nant
                v = phase[a, 2, fs]
                isfinite(v) && (phase[a, 2, fs] = v - δ)
            end
        end
    end

    # Circular-mean reference per (station, feed) → zero net applied phase (gauge).
    leaf = _component_leaf(plan, θ)
    for a in 1:nant, f in 1:2
        acc = zero(ComplexF64)
        for fs in 1:nseg
            v = phase[a, f, fs]
            isfinite(v) && (acc += cis(v))
        end
        abs(acc) > 0 || continue
        m = angle(acc)
        for fs in 1:nseg
            v = phase[a, f, fs]
            isfinite(v) || continue
            node = _feed_node(plan.tying, f)
            node == 0 && continue
            leaf[1, node, fs, 1, a] = rem2pi(v - m, RoundNearest)
        end
    end
    return θ
end

# One frequency segment's coherent residual `(r, w, w2)`: the sums of the
# accumulators over the channels it holds, plus `w2 = Σ wᶜ²`, which converts a
# PER-CHANNEL noise variance into the variance of this segment's normalized
# value `r/w` — `n2 · w2 / w²`, i.e. `n2/k` for `k` equally-weighted channels.
# Scaling the noise the other way (or not at all) would make a wide block look
# WORSE than its channels and the SNR gate would reject the very observations
# grouping exists to strengthen.
#
# A one-channel segment leaves all three quantities at that channel's own, so
# `ChannelBlocks(1)` reproduces a free per-channel bandpass exactly; a wider
# block pools its channels' signal into the one value they share.
function _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
    r = zero(eltype(rbar_bp))
    w = zero(eltype(wbar_bp))
    w2 = zero(eltype(wbar_bp))
    for gc in chans
        rc = rbar_bp[bi, p, gc]
        wc = wbar_bp[bi, p, gc]
        (isfinite(rc) && isfinite(wc) && wc > 0) || continue
        r += rc
        w += wc
        w2 += wc^2
    end
    return r, w, w2
end

# Segment SNR² under the per-channel noise estimate `n2` (`NaN` when the track
# had too few channels to estimate one — then fall back to the weight itself).
_segment_snr2(r, w, w2, n2) =
    isfinite(n2) && n2 > 0 ? abs2(r / w) * w^2 / (n2 * w2) : abs2(r) / w

"""
    solve_amp_bandpass!(θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan,
                        channel_freqs; snr_floor = 1.0, ridge = 1.0e-6,
                        spw_of_chan = Int[], smoother = PenalizedBandpass(1.0),
                        max_logamp = log(10.0), spike_sigma = 5.0)

Solve the per-(station, feed) AMPLITUDE bandpass (log-amp) from the accumulated
residual and write it into the log-amp bandpass component's θ blocks —
flattening the per-station instrumental frequency response. `plan`'s frequency
segmentation sets the resolution: one solved value per (station, feed, frequency
segment). Gated closure observations are gathered PER SPW and handed to
`smoother` (an [`AbstractBandpassSmoother`](@ref)); narrow positive log-amp
spikes (pcal tones, RFI — additive contamination the multiplicative model must
not up-weight) are excised, and a zero-band-mean gauge per (station, feed) keeps
the bandpass to SHAPE only.
"""
function solve_amp_bandpass!(
        θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan, channel_freqs;
        snr_floor::Real = 1.0, ridge::Real = 1.0e-6,
        spw_of_chan::AbstractVector{<:Integer} = Int[],
        smoother::AbstractBandpassSmoother = PenalizedBandpass(1.0),
        max_logamp::Real = log(10.0),
        spike_sigma::Real = 5.0,
    )
    nbl, npol, nchan = size(rbar_bp)
    feeds = [correlation_feed_pair(p) for p in pol_products]
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in 1:nbl, p in 1:npol]
    nnodes = 2 * nant
    soc = isempty(spw_of_chan) ? ones(Int, nchan) : collect(spw_of_chan)
    fsegs = segment_groups(plan.fseg_id, length(plan.nchan_seg))
    nfseg = length(fsegs)
    # A frequency segment is the unit solved for, so it must lie within one spw:
    # the smoothers fit a shape per spw and could not place a straddling segment.
    seg_spw = map(fsegs) do chans
        s = soc[first(chans)]
        all(gc -> soc[gc] == s, chans) || throw(
            ArgumentError(
                "solve_amp_bandpass!: a frequency segment straddles a spectral-window " *
                    "boundary; the bandpass segmentation must refine the spw partition.",
            ),
        )
        return s
    end
    seg_freq = [sum(channel_freqs[gc] for gc in chans) / length(chans) for chans in fsegs]
    la = fill(NaN, nant, 2, nfseg)

    for bnd in sort(unique(seg_spw))
        sidx = [s for s in 1:nfseg if seg_spw[s] == bnd]
        nseg = length(sidx); nseg == 0 && continue
        fs = Float64[seg_freq[s] for s in sidx]
        center = sum(fs) / nseg
        scale = maximum(abs.(fs .- center)); scale = scale > 0 ? scale : 1.0
        xseg = [(fs[ci] - center) / scale for ci in 1:nseg]

        # Gated closure observations for this spw (`ci` indexes the frequency
        # segments it holds).
        na = Int[]; nbn = Int[]; cii = Int[]; vals = Float64[]; wts = Float64[]
        for (ci, s) in enumerate(sidx), bi in 1:nbl, p in 1:npol
            a, b = bl_pairs[bi]
            a == b && continue
            r, w, w2 = _segment_residual(rbar_bp, wbar_bp, bi, p, fsegs[s])
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            snr2 = _segment_snr2(r, w, w2, noise2[bi, p])
            snr2 >= snr_floor^2 || continue
            amp = abs(r / w); amp > 0 || continue
            fa, fb = feeds[p]
            push!(na, _node(a, fa, nant)); push!(nbn, _node(b, fb, nant)); push!(cii, ci)
            push!(vals, log(amp)); push!(wts, snr2)
        end
        isempty(vals) && continue
        la_seg = _fit_bandpass_segment(smoother, na, nbn, cii, vals, wts, nnodes, nseg, xseg, ridge)
        for node in 1:nnodes
            ant = (node - 1) % nant + 1; feed = (node - 1) ÷ nant + 1
            for ci in 1:nseg
                v = la_seg[node, ci]
                isfinite(v) && (la[ant, feed, sidx[ci]] = v)
            end
        end
    end

    # Narrow-spike guard: additive contamination (pcal tones, RFI) violates the
    # multiplicative gain model — a contaminated channel shows EXCESS amplitude,
    # the fit hands it |g| > 1, and `apply_calibration` would then UP-weight it
    # (w → w·|g|²), amplifying exactly the channels that should be distrusted.
    # Genuine passband structure is smooth or negative (roll-off), so narrow
    # POSITIVE log-amp outliers vs the per-(station, feed, spw) robust scale are
    # excised (left unapplied, |g| = 1) instead of trusted. `spike_sigma = 0`
    # disables the guard.
    if spike_sigma > 0
        for a in 1:nant, f in 1:2, bnd in sort(unique(seg_spw))
            sidx = [s for s in 1:nfseg if seg_spw[s] == bnd]
            v = [la[a, f, s] for s in sidx if isfinite(la[a, f, s])]
            length(v) >= 8 || continue
            med = median(v)
            s = 1.4826 * median(abs.(v .- med))
            cut = spike_sigma * max(s, 0.02)
            for si in sidx
                isfinite(la[a, f, si]) || continue
                la[a, f, si] - med > cut && (la[a, f, si] = NaN)
            end
        end
    end

    # Zero band-mean log-amp gauge per (station, feed) — SHAPE only. Write the slots.
    leaf = _component_leaf(plan, θ)
    for a in 1:nant, f in 1:2
        acc = 0.0; n = 0
        for s in 1:nfseg
            v = la[a, f, s]
            isfinite(v) && (acc += v; n += 1)
        end
        n == 0 && continue
        m = acc / n
        for s in 1:nfseg
            v = la[a, f, s]
            isfinite(v) || continue
            node = _feed_node(plan.tying, f)
            node == 0 && continue
            val = v - m
            # Leave implausibly-large corrections UNAPPLIED (|g| = 1). A smoother (the
            # default) interpolates gaps and self-regularizes, but `FreeBandpass` (or a
            # near-zero `lambda`) can hand a low-SNR band-edge channel that barely
            # clears the gate a huge log-amp; applying it would up-weight that channel's
            # noise, since `apply_calibration` scales weights by |g|². The bound is
            # generous (|g| ≤ 10) so real passband roll-off/structure passes unchanged —
            # only pathological noise blow-ups are gated.
            leaf[1, node, s, 1, a] = abs(val) > max_logamp ? 0.0 : val
        end
    end
    return θ
end

# ── Coverage top-up selection (stations the calibrator never observed) ────────

# Wraps the bandpass step's user selection: stations absent from every selected
# scan would get NO bandpass (g = 1), so for each such station the highest-SNR
# scan (any source) containing it is added — mixing sources is safe for the
# bandpass SHAPE (a source's structure phase is flat in frequency per baseline,
# so it biases every channel identically and cancels in the shape; per-scan
# ionosphere differences land in the frozen curve's mean, which each scan's
# dTEC is measured relative to). Applied to ANY selection, matching the frozen
# monolith's `_bandpass_coverage_topup`; a selection that already covers every
# station (e.g. `AllScans`) is returned unchanged. Requires the per-scan
# `stations` record field `select_groups` provides.
struct CoverageTopup{S <: AbstractScanSelection} <: AbstractScanSelection
    inner::S
end

function select_scans(sel::CoverageTopup, scans)
    picked = select_scans(sel.inner, scans)
    (isempty(scans) || !hasproperty(first(scans), :stations)) && return picked
    by = Dict(s.index => s for s in scans)
    covered = Set{Int}()
    for gi in picked
        union!(covered, by[gi].stations)
    end
    pickset = Set(picked)
    extra = Int[]
    order = sortperm([isfinite(s.snr) ? s.snr : -Inf for s in scans]; rev = true)
    for oi in order
        s = scans[oi]
        s.index in pickset && continue
        isempty(setdiff(s.stations, covered)) && continue
        push!(extra, s.index)
        union!(covered, s.stations)
    end
    return sort!(vcat(picked, extra))
end

# ── Per-segment amplitude-shape fitters (relocated verbatim from the monolith) ──

# smoother types (defined at the top of this file). Each takes ONE spw's gated closure
# observations — `na`/`nb` node indices, `ci` the index of the frequency segment within
# the spw, `val = log|V̄|`, `w = SNR²`, `xseg` the centred/scaled in-spw frequency
# coordinate — and returns `la_seg::Matrix` (nnodes × nseg, `NaN` where unestimable).

# Bucket observation indices by their frequency segment.
function _bandpass_obs_by_segment(ci, nseg)
    byc = [Int[] for _ in 1:nseg]
    for i in eachindex(ci)
        push!(byc[ci[i]], i)
    end
    return byc
end

# Free per-segment closure: independent signless-Laplacian WLS per frequency segment.
function _fit_bandpass_segment(::FreeBandpass, na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
    la = fill(NaN, nnodes, nseg)
    pen = fill(float(ridge), nnodes)
    for (c, idx) in enumerate(_bandpass_obs_by_segment(ci, nseg))
        isempty(idx) && continue
        A = zeros(length(idx), nnodes)
        touched = falses(nnodes)
        for (r, i) in enumerate(idx)
            A[r, na[i]] += 1.0; A[r, nb[i]] += 1.0
            touched[na[i]] = true; touched[nb[i]] = true
        end
        sol = weighted_regularized_least_squares(A, val[idx], w[idx], pen)
        for node in 1:nnodes
            touched[node] && (la[node, c] = sol[node])
        end
    end
    return la
end

# Per-spw polynomial: one closure WLS over `nb = degree+1` coefficients per node;
# θ-column for (node, k) is `(node-1)*nb + k`. Evaluated at every channel (incl. gaps).
function _fit_bandpass_segment(sm::PolynomialBandpass, na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
    deg = clamp(sm.degree, 1, max(1, nseg - 1))
    nbf = deg + 1
    B = Float64[xseg[c]^k for c in 1:nseg, k in 0:deg]      # nseg × nbf basis
    ncol = nnodes * nbf
    A = zeros(length(val), ncol)
    touched = falses(nnodes)
    @inbounds for i in eachindex(val)
        oa = (na[i] - 1) * nbf; ob = (nb[i] - 1) * nbf
        for k in 1:nbf
            A[i, oa + k] += B[ci[i], k]; A[i, ob + k] += B[ci[i], k]
        end
        touched[na[i]] = true; touched[nb[i]] = true
    end
    coef = weighted_regularized_least_squares(A, val, w, fill(float(ridge), ncol))
    la = fill(NaN, nnodes, nseg)
    for node in 1:nnodes
        touched[node] || continue
        c0 = (node - 1) * nbf
        for c in 1:nseg
            v = 0.0
            @inbounds for k in 1:nbf
                v += coef[c0 + k] * B[c, k]
            end
            la[node, c] = v
        end
    end
    return la
end

# Roughness-penalised: free per-channel closure, then a per-node Whittaker
# (2nd-difference) penalised WLS across channels — interpolating gated channels via
# the penalty, following the data elsewhere.
function _fit_bandpass_segment(sm::PenalizedBandpass, na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
    la0 = _fit_bandpass_segment(FreeBandpass(), na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
    (sm.lambda <= 0 || nseg < 3) && return la0
    prec = zeros(nnodes, nseg)                              # per-(node, channel) precision Σ w
    for i in eachindex(val)
        prec[na[i], ci[i]] += w[i]; prec[nb[i], ci[i]] += w[i]
    end
    la = copy(la0)
    for node in 1:nnodes
        any(>(0), @view prec[node, :]) || continue
        la[node, :] .= _whittaker_smooth(view(la0, node, :), view(prec, node, :), sm.lambda, ridge)
    end
    return la
end

# 1-D Whittaker smoother: minimise  Σ w_i (x_i − y_i)² + (λ·w̄) Σ (x_{i−1} − 2x_i + x_{i+1})²
# + ridge Σ x_i², via the generic penalized WLS solve with `A = I`, the ridge
# term stacked as `√ridge · I` and the roughness term as `√λ · D` (`D` the
# 2nd-difference operator). `w_i = 0` (and `y_i` non-finite) where a channel had
# no data → the penalty alone sets it (interpolation). `λ` is scaled by the
# median positive weight so it is data-relative.
function _whittaker_smooth(y, w, lambda::Real, ridge::Real)
    n = length(y)
    pos = [w[i] for i in 1:n if w[i] > 0]
    λ = lambda * (isempty(pos) ? 1.0 : median(pos))
    wi = [(w[i] > 0 && isfinite(y[i])) ? float(w[i]) : 0.0 for i in 1:n]
    yi = [wi[i] > 0 ? float(y[i]) : 0.0 for i in 1:n]
    D = zeros(n - 2, n)
    @inbounds for i in 1:(n - 2)                            # 2nd-difference rows [1, −2, 1]
        D[i, i] = 1.0; D[i, i + 1] = -2.0; D[i, i + 2] = 1.0
    end
    R = vcat(sqrt(ridge) .* Matrix(1.0I, n, n), sqrt(λ) .* D)
    return weighted_regularized_least_squares(Matrix(1.0I, n, n), yi, wi, R)
end
