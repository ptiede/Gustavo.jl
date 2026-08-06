# ── Bandpass stage: per-channel station phase/log-amp over scan windows ───────
#
# The carved-out bandpass stage of the composable pipeline: the per-scan
# residual accumulation ([`accumulate_bandpass!`](@ref)) and the two per-channel
# closure solves ([`solve_phase_bandpass!`](@ref) / [`solve_amp_bandpass!`](@ref)) —
# descended verbatim from the monolithic solver (deleted at M5), retargeted
# from its concat cube to a scan `DimStack` where they touch data. The
# `Bandpass` step visits every scan (refine → accumulate → return the scan's
# contribution) and its `finish_pass!` folds the contributions in GROUP-INDEX
# order — deterministic at ANY concurrency (unlike the monolith's
# ntasks-dependent chunk fold; the two agree to float-rounding, gated at
# rtol ≤ 1e-12). WHAT is fit lives on `BandpassModel`; HOW it is solved is
# pluggable through `AbstractBandpassEstimator` — `SplitWLS` runs the closure
# solves above, `JointALS` runs [`solve_joint_bandpass!`](@ref), an alternative,
# per-scan-aware solve that fits the actual complex visibilities against an
# explicit per-scan source term instead of a closure that assumes it cancels.
#
# The graph/solve helpers (`_ObsRow`, `_solve_observable`, `_track_noise2`,
# `_node`) live in stationize.jl/adhoc.jl; the amp-bandpass
# `WLSEstimator` presets (`free_bandpass`, `polynomial_bandpass`,
# `penalized_bandpass`) live below.

# ── Amplitude-bandpass estimators (pluggable WLSEstimator presets) ────────────
#
# The per-(station, feed) log-amp bandpass is solved from the SUM closure
# `log|V̄_ab(ν)| = la_a(ν) + lb_b(ν)` over each spw (a +1/+1, signless-Laplacian
# incidence — FULL RANK, so no reference state). HOW the per-channel shape is
# estimated — and how low-/no-signal channels are filled — is a pluggable
# strategy: `free_bandpass`/`polynomial_bandpass`/`penalized_bandpass` each
# return a callable of `(na, nb, ci, val, w, nnodes, nseg, xseg, ridge) ->
# la_seg::Matrix` (nnodes × nseg, `NaN` where unestimable), built on
# `WLSEstimator`; add a new one the same way to extend.

# Signless-Laplacian (SUM) incidence for one frequency segment's gated
# closure observations, restricted to the rows `idx` — the shared
# `observation_model` behind both `free_bandpass` and `polynomial_bandpass`'s
# per-segment special case.
function _signless_incidence(na, nb, idx, nnodes, val, w)
    A = zeros(length(idx), nnodes)
    for (r, i) in enumerate(idx)
        A[r, na[i]] += 1.0; A[r, nb[i]] += 1.0
    end
    return A, val[idx], w[idx]
end

"""
    free_bandpass()

Free per-channel closure: one independent [`WLSEstimator`](@ref) call per
frequency segment (ridge-only regularization), no roughness penalty. Follows
the data but does NOT estimate low-/no-signal channels (left at |g| = 1).
The `λ → 0` / `degree → ∞` limit of the others.
"""
function free_bandpass()
    return function (na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
        est = WLSEstimator((idx) -> _signless_incidence(na, nb, idx, nnodes, val, w), A -> fill(ridge, size(A, 2)))
        la = fill(NaN, nnodes, nseg)
        for (c, idx) in enumerate(_bandpass_obs_by_segment(ci, nseg))
            isempty(idx) && continue
            sol = est(idx)
            for i in idx
                la[na[i], c] = sol[na[i]]
                la[nb[i], c] = sol[nb[i]]
            end
        end
        return la
    end
end

# Signless-Laplacian incidence over per-node polynomial-coefficient columns
# (θ-column for (node, k) is `(node-1)*nbf + k`), evaluated at every segment
# (incl. gaps) via the fitted basis coefficients.
function _polynomial_incidence(na, nb, ci, val, w, nnodes, nbf, B)
    A = zeros(length(val), nnodes * nbf)
    @inbounds for i in eachindex(val)
        oa = (na[i] - 1) * nbf; ob = (nb[i] - 1) * nbf
        for k in 1:nbf
            A[i, oa + k] += B[ci[i], k]; A[i, ob + k] += B[ci[i], k]
        end
    end
    return A, val, w
end

"""
    polynomial_bandpass(degree = 4)

Smooth per-spw polynomial of `degree` in a centred/scaled frequency coordinate, fit
by a single closure [`WLSEstimator`](@ref) call (the `PolynomialFreq` design
convention). Estimates gaps by the fit. Assumes the in-spw bandpass is ~a
low-order polynomial (smooth passband + gentle roll-off); a high degree can
ring (Runge) at the edges.
"""
function polynomial_bandpass(degree::Integer = 4)
    degree >= 1 || throw(ArgumentError("polynomial_bandpass: degree must be ≥ 1"))
    return function (na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
        deg = clamp(degree, 1, max(1, nseg - 1))
        nbf = deg + 1
        B = Float64[xseg[c]^k for c in 1:nseg, k in 0:deg]      # nseg × nbf basis
        est = WLSEstimator((na, nb, ci, val, w) -> _polynomial_incidence(na, nb, ci, val, w, nnodes, nbf, B), A -> fill(ridge, size(A, 2)))
        coef = est(na, nb, ci, val, w)
        touched = falses(nnodes)
        for i in eachindex(val)
            touched[na[i]] = true; touched[nb[i]] = true
        end
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
end

"""
    penalized_bandpass(lambda = 1.0)

Roughness-penalised per-channel bandpass (a Whittaker smoother): [`free_bandpass`](@ref)'s
per-segment closure, then a per-node [`WLSEstimator`](@ref) 2nd-difference
smoothness penalty of strength `lambda` (relative to the per-channel data
weight) across segments. Makes NO shape assumption — follows real structure
where the SNR supports it and smoothly interpolates gaps where it does not.
`lambda → 0` ⇒ [`free_bandpass`](@ref); large `lambda` ⇒ flat.
"""
function penalized_bandpass(lambda::Real = 1.0)
    lambda >= 0 || throw(ArgumentError("penalized_bandpass: lambda must be ≥ 0"))
    free = free_bandpass()
    return function (na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
        la0 = free(na, nb, ci, val, w, nnodes, nseg, xseg, ridge)
        (lambda <= 0 || nseg < 3) && return la0
        prec = zeros(nnodes, nseg)                              # per-(node, channel) precision Σ w
        for i in eachindex(val)
            prec[na[i], ci[i]] += w[i]; prec[nb[i], ci[i]] += w[i]
        end
        la = copy(la0)
        for node in 1:nnodes
            any(>(0), @view prec[node, :]) || continue
            la[node, :] .= _whittaker_smooth(view(la0, node, :), view(prec, node, :), lambda, ridge)
        end
        return la
    end
end

"""
    BandpassModel(; phase = true, amp = true, freq = ChannelBlocks(1),
                  amp_model = penalized_bandpass(1.0))

WHAT the [`Bandpass`](@ref) step fits: whether to solve the phase bandpass,
the log-amplitude bandpass, or both; how finely each is resolved in frequency
(`freq`, a [`ChannelBlocks`](@ref) — the default is one free value per
channel); and `amp_model`, the amplitude-shape estimator
([`penalized_bandpass`](@ref) / [`polynomial_bandpass`](@ref) /
[`free_bandpass`](@ref)) used by the [`SplitWLS`](@ref) estimator. Uniform
across every antenna — no per-station segmentation.
"""
Base.@kwdef struct BandpassModel
    phase::Bool = true
    amp::Bool = true
    freq::ChannelBlocks = ChannelBlocks(1)
    amp_model = penalized_bandpass(1.0)
end

"""
    AbstractBandpassEstimator

HOW the [`Bandpass`](@ref) step solves the model `BandpassModel` describes.
Concretely [`SplitWLS`](@ref) (independent phase/log-amp closures, the
default) or [`JointALS`](@ref) (a joint complex-gain + per-scan source-term
solve). Mirrors [`AbstractFringeEstimator`](@ref)'s split between WHAT a step
fits and HOW.

# Implementing an estimator

Define:

    Gustavo.Fringe.solve_bandpass!(est::MyEstimator, θ, results, setup, model::BandpassModel; ref_ant) -> nothing

writing into `θ`'s bandpass blocks. `results` is the per-scan
`(; rl, wl, pols, source)` accumulator list, in group-index order; `setup` is
`(; bl_pairs, blidx, nant, bp_plan, amp_plan, channel_freqs, spw_of_chan)`,
built once per pass. The fallback errors, naming what is missing.

Two optional hooks:

    Gustavo.Fringe.bandpass_derotate(est::MyEstimator) -> Bool   # default true
    Gustavo.Fringe.validate_bandpass(est::MyEstimator, model::BandpassModel)

[`bandpass_derotate`](@ref) controls whether [`accumulate_bandpass!`](@ref)
counter-rotates each AP before accumulating (see its docstring) —
`SplitWLS` needs this (it sums scans together), `JointALS` does not (it fits
each scan's own coherent visibility). [`validate_bandpass`](@ref) is checked
at model-compile time, before any data is read; a method for a new estimator
REPLACES the default, so it must state at least as strong a requirement.
"""
abstract type AbstractBandpassEstimator end

bandpass_derotate(::AbstractBandpassEstimator) = true

"""
    validate_bandpass(est::AbstractBandpassEstimator, model::BandpassModel)

Reject a [`BandpassModel`](@ref) that `est` cannot solve, at the point the
[`Bandpass`](@ref) step compiles its components — before any data is read.

The default requires a model that fits SOMETHING: with both `phase` and `amp`
off the step compiles no components at all, so it would accumulate every scan
and write nowhere. An estimator with stricter needs defines its own method
(see [`JointALS`](@ref)), which replaces this one.
"""
function validate_bandpass(::AbstractBandpassEstimator, model::BandpassModel)
    # The step indexes its compiled components by name
    # (`layout.plantree.phase.bandpass` and its log-amp twin) under exactly
    # these two flags, so a model with neither set has nothing to solve into.
    model.phase || model.amp || throw(
        ArgumentError("BandpassModel fits nothing: at least one of `phase` or `amp` must be true."),
    )
    return nothing
end
function solve_bandpass! end
solve_bandpass!(est::AbstractBandpassEstimator, θ, results, setup, model::BandpassModel; ref_ant) =
    error("$(typeof(est)) does not implement the bandpass estimator interface: define " *
        "Gustavo.Fringe.solve_bandpass!(::$(typeof(est)), θ, results, setup, model; ref_ant).")

"""
    accumulate_bandpass!(rbar_bp, wbar_bp, blidx, stack, win::GeometryWindow; derotate = true)

Accumulate one scan window's contribution to the per-(global-baseline, product,
GLOBAL channel) coherent residual `rbar_bp` (and weight `wbar_bp`) for the
bandpass solves, on data already gain-corrected through the pipeline's
transform chain. `blidx` maps `(a, b) -> row` in the global baseline table
(co-located pairs excluded there never contribute).

`derotate` (default `true`) counter-rotates each AP, BEFORE summing over time,
by its OWN band-averaged residual phase — removing the per-AP time phase
(residual rate/drift, and what the adhoc stage would later remove) so summing
COHERENT SCANS TOGETHER (the closure-based [`solve_phase_bandpass!`](@ref) /
[`solve_amp_bandpass!`](@ref) path, which combines every selected scan's
residual into one accumulator before solving) isolates the per-channel SHAPE
despite each scan's uncontrolled source phase. [`solve_joint_bandpass!`](@ref)
fits each scan's OWN coherent visibility against an explicit per-scan source
term instead of summing scans together, so it passes `derotate = false` — the
per-AP trick would otherwise erase the very source phase/amplitude that term
is meant to absorb.
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
        rbar_bp, wbar_bp, blidx, stack::AbstractDimStack, win::GeometryWindow;
        derotate::Bool = true,
    )
    V = stack[:vis]                                      # the dims-carrying layers —
    W = stack[:weights]                                  # the loops below address axes BY NAME
    bl_pairs = UVData.baselines(stack).pairs
    g_ci = win.chan_idx
    for p in axes(V, Pol), bi in axes(V, Baseline)
        a, b = bl_pairs[bi]
        a == b && continue # autocorrelation skip
        idx = get(blidx, (a, b), 0)
        idx == 0 && continue # baseline doesn't exist so skip
        for tt in axes(V, Ti)
            rot = one(eltype(V))
            if derotate
                # Band-averaged residual phase for this AP (the per-AP time phase).
                acc = zero(eltype(V))
                for c in axes(V, Frequency)
                    w = W[Frequency=c, Ti=tt, Baseline=bi, Pol=p]
                    vv = V[Frequency=c, Ti=tt, Baseline=bi, Pol=p]
                    cond = (w > 0 && isfinite(w) && isfinite(vv))
                    acc += ifelse(cond, w * vv, zero(eltype(V)))
                end
                rot = ifelse(abs(acc) > 0, conj(acc) / abs(acc), one(eltype(V))) # cis(-angle(acc)): de-rotate this AP
            end
            for c in axes(V, Frequency)
                w = W[Frequency=c, Ti=tt, Baseline=bi, Pol=p]
                vv = V[Frequency=c, Ti=tt, Baseline=bi, Pol=p]
                cond = (w > 0 && isfinite(w) && isfinite(vv))
                gc = g_ci[c]
                rbar_bp[idx, p, gc] += ifelse(cond, w * vv * rot, zero(eltype(V)))
                wbar_bp[idx, p, gc] += ifelse(cond, w, zero(eltype(W)))
            end
        end
    end
    return rbar_bp, wbar_bp
end

"""
    SplitWLS()

The default [`AbstractBandpassEstimator`](@ref): independent per-channel
closure solves for phase ([`solve_phase_bandpass!`](@ref)) and log-amplitude
([`solve_amp_bandpass!`](@ref)), each summing every scan's residual into one
accumulator before solving — the closure assumes a baseline's source term
cancels out of the per-channel phase-difference/log-amp-sum, so it is not
appropriate for a resolved or polarized calibrator (see [`JointALS`](@ref)).
"""
struct SplitWLS <: AbstractBandpassEstimator end

function solve_bandpass!(::SplitWLS, θ, results, setup, model::BandpassModel; ref_ant::Integer)
    pols = results[1].pols
    nchan = length(setup.channel_freqs)
    rbar, wbar = bandpass_accumulators(length(setup.bl_pairs), length(pols), nchan)
    for res in results
        rbar .+= res.rl
        wbar .+= res.wl
    end
    setup.bp_plan === nothing || solve_phase_bandpass!(
        θ, rbar, wbar, setup.bl_pairs, pols, setup.nant, setup.bp_plan; ref_ant,
    )
    setup.amp_plan === nothing || solve_amp_bandpass!(
        θ, rbar, wbar, setup.bl_pairs, pols, setup.nant, setup.amp_plan, setup.channel_freqs;
        spw_of_chan = setup.spw_of_chan, smoother = model.amp_model,
    )
    return nothing
end

"""
    solve_phase_bandpass!(θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan;
                          ref_ant = 1, snr_floor = 1.0)

Solve the per-(station, feed) phase bandpass from the accumulated per-channel
residual and write it into the bandpass component's θ blocks. `plan`'s frequency
segmentation sets the resolution: one solved value per (station, feed, frequency
segment), from the residual of every channel the segment holds. Per segment, the
globally-closing per-feed phase is solved on the (station, feed) graph with a
scale-invariant SNR gate, and each (station, feed) track is referenced to its
circular-mean phase over segments (zero net applied phase). Cross-hand rows tie
the two feed blocks per segment, so the inter-feed phase shape across the band is
solved rather than conventional; the one band-constant inter-feed offset is not
separable from the source's cross-hand phase and is removed by the circular-mean
referencing.
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
            push!(rows, _ObsRow(a, b, fa, fb, angle(r), snr2))
        end
        ph, _, _, _ = _solve_observable(rows, nant, ref_ant; rewrap = 0)
        phase[:, :, fs] .= ph
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
                        spw_of_chan = Int[], smoother = penalized_bandpass(1.0),
                        max_logamp = log(10.0), spike_sigma = 5.0)

Solve the per-(station, feed) AMPLITUDE bandpass (log-amp) from the accumulated
residual and write it into the log-amp bandpass component's θ blocks —
flattening the per-station instrumental frequency response. `plan`'s frequency
segmentation sets the resolution: one solved value per (station, feed, frequency
segment). Gated closure observations are gathered PER SPW and handed to
`smoother` (a callable as returned by [`free_bandpass`](@ref) /
[`polynomial_bandpass`](@ref) / [`penalized_bandpass`](@ref)); narrow positive
log-amp spikes (pcal tones, RFI — additive contamination the multiplicative
model must not up-weight) are excised, and a zero-band-mean gauge per
(station, feed) keeps the bandpass to SHAPE only.
"""
function solve_amp_bandpass!(
        θ, rbar_bp, wbar_bp, bl_pairs, pol_products, nant, plan, channel_freqs;
        snr_floor::Real = 1.0, ridge::Real = 1.0e-6,
        spw_of_chan::AbstractVector{<:Integer} = Int[],
        smoother = penalized_bandpass(1.0),
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
        la_seg = smoother(na, nbn, cii, vals, wts, nnodes, nseg, xseg, ridge)
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
            # default) interpolates gaps and self-regularizes, but `free_bandpass` (or a
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

# ── Joint complex bandpass + per-scan source coherence (ALS) ──────────────────
#
# solve_phase_bandpass!/solve_amp_bandpass! assume a baseline's source term
# cancels out of the per-channel phase-difference/log-amp-sum closure — true
# only for an unresolved, unpolarized source. solve_joint_bandpass! instead
# fits the actual complex visibilities against
#   V_ab(ν) | scan  ≈  g_a(ν) · S_{scan,ab,pol} · conj(g_b(ν)),
# one frequency-flat complex `S` per (scan, baseline, polarization product),
# so a resolved/polarized source's per-baseline structure is absorbed into `S`
# instead of biasing the station bandpass. `g` is bilinear with `S`, so this
# alternates a closed-form per-(scan, baseline, pol) solve of `S` (given the
# current `g`, [`_update_source_coherence!`](@ref)) with a Gauss-Seidel
# per-(station, feed) solve of `g` (given `S` and every OTHER station's
# current gain, [`_update_station_gains!`](@ref)) at `phase_plan`/`amp_plan`'s
# shared frequency-segment resolution, to convergence.
#
# Every array below carries (Scan, Baseline, Pol, Frequency) or (Ant, Feed,
# Frequency) dims — the house style of this module — so the loops read by
# axis NAME; `Frequency` here is the SEGMENT index (as elsewhere once a
# solve moves past the raw per-channel accumulator). Indexing itself stays
# plain-positional.

# One scan's per-(baseline, pol, SEGMENT) coherent residual, written directly
# into `rview`/`wview` (a (Baseline, Pol, Frequency) slice of the multi-scan
# accumulator — no intermediate allocation) and SNR-gated exactly as the
# closure solves gate it: `_track_noise2`/`_segment_snr2` read straight off
# `sc.rl`/`sc.wl` (a fresh per-scan accumulator, not summed across scans), so
# no new noise-estimation machinery is needed.
function _reduce_scan_segments!(rview, wview, sc, segs)
    nchan = size(sc.rl, Frequency)
    for p in axes(rview, Pol), bi in axes(rview, Baseline)
        noise2 = _track_noise2(sc.rl, sc.wl, bi, p, nchan)
        for (fs, chans) in enumerate(segs)
            rc, wc, w2c = _segment_residual(sc.rl, sc.wl, bi, p, chans)
            keep = isfinite(rc) && abs(rc) > 0 && isfinite(wc) && wc > 0 &&
                _segment_snr2(rc, wc, w2c, noise2) >= 1.0
            rview[bi, p, fs] = keep ? rc : zero(rc)
            wview[bi, p, fs] = keep ? wc : zero(wc)
        end
    end
    return nothing
end

# Every scan's (Baseline, Pol, SEGMENT) residual, stacked over an added Scan
# axis — NOT summed across scans (unlike solve_phase_bandpass!'s/
# solve_amp_bandpass!'s fold), since the per-scan source coherence needs each
# scan's own coherent visibility. Element types follow the scan accumulators'
# own (`bandpass_accumulators`'), not a hardcoded precision.
function _reduce_all_scans(scans, segs)
    nscan = length(scans)
    nbl, npol = size(first(scans).rl, Baseline), size(first(scans).rl, Pol)
    nseg = length(segs)
    C = eltype(first(scans).rl)
    T = real(eltype(first(scans).wl))
    d = (Scan(1:nscan), Baseline(1:nbl), Pol(1:npol), Frequency(1:nseg))
    rseg = DimensionalData.DimArray(zeros(C, nscan, nbl, npol, nseg), d)
    wseg = DimensionalData.DimArray(zeros(T, nscan, nbl, npol, nseg), d)
    for (si, sc) in enumerate(scans)
        _reduce_scan_segments!(view(rseg, si, :, :, :), view(wseg, si, :, :, :), sc, segs)
    end
    return rseg, wseg
end

# Every (baseline, pol) touching (ant, feed), tagged by which side of the
# baseline it is — built once, reused by every ALS iteration's gain update.
function _joint_bandpass_touching(bl_pairs, feeds, nant)
    touching = [Tuple{Int, Int, Symbol}[] for _ in 1:nant, _ in 1:2]
    for p in eachindex(feeds), bi in eachindex(bl_pairs)
        a, b = bl_pairs[bi]
        a == b && continue
        fa, fb = feeds[p]
        push!(touching[a, fa], (bi, p, :a))
        push!(touching[b, fb], (bi, p, :b))
    end
    return touching
end

_ant_feed(node, nant) = ((node - 1) % nant + 1, (node - 1) ÷ nant + 1)

# One reference (station, feed) node per connected component of the
# (station, feed) graph, and which nodes the graph's baselines/pols actually
# touch — mirrors `_solve_observable`'s pin selection in stationize.jl. The
# pinned node is held fixed at `1.0 + 0.0im` for every segment throughout the
# ALS iteration: per-channel-free `g` gives every (station, feed) its OWN
# unconstrained value at every segment, so the model is invariant under
# multiplying every station's gain at a given segment by ANY shared complex
# factor and dividing every touching `S` by the same, AT THAT SEGMENT
# INDEPENDENTLY — a per-antenna final gauge only removes a single constant per
# (station, feed), so it cannot remove a factor that varies freely from
# segment to segment the way this pin does.
function _joint_bandpass_pins(bl_pairs, feeds, nant, ref_ant)
    nnodes = 2 * nant
    edges = Tuple{Int, Int}[]
    for p in eachindex(feeds), bi in eachindex(bl_pairs)
        a, b = bl_pairs[bi]
        a == b && continue
        fa, fb = feeds[p]
        push!(edges, (_node(a, fa, nant), _node(b, fb, nant)))
    end
    compid, ncomp, graph_touched = connected_components(nnodes, edges)
    pins = Set{Int}()
    for c in 1:ncomp
        comp_nodes = findall(==(c), compid)
        r1, r2 = _node(ref_ant, 1, nant), _node(ref_ant, 2, nant)
        push!(pins, r1 in comp_nodes ? r1 : (r2 in comp_nodes ? r2 : minimum(comp_nodes)))
    end
    return pins, graph_touched
end

# Closed-form per-(scan, baseline, pol) solve of the source coherence `S`
# given the current station gains `g`: the weighted-least-squares minimizer of
# `Σ_segment wseg·|rseg/wseg − g_a·S·conj(g_b)|²` over the single complex
# unknown `S`.
function _update_source_coherence!(S, g, rseg, wseg, bl_pairs, feeds)
    T = real(eltype(S))
    for p in axes(rseg, Pol), bi in axes(rseg, Baseline)
        a, b = bl_pairs[bi]
        a == b && continue
        fa, fb = feeds[p]
        for si in axes(rseg, Scan)
            numer = zero(eltype(S))
            denom = zero(T)
            for fs in axes(rseg, Frequency)
                w = wseg[si, bi, p, fs]
                w > 0 || continue
                u = g[a, fa, fs] * conj(g[b, fb, fs])
                abs2(u) > 0 || continue
                numer += conj(u) * rseg[si, bi, p, fs]
                denom += w * abs2(u)
            end
            S[si, bi, p] = denom > 0 ? numer / denom : zero(eltype(S))
        end
    end
    return nothing
end

# One Gauss-Seidel sweep over every non-pinned (station, feed): closed-form
# per-segment solve of its complex gain given the current source coherence `S`
# and every OTHER station's current gain (immediately visible to later
# antennas in the same sweep — Gauss-Seidel, not Jacobi). Returns the largest
# relative gain change, for the caller's convergence check.
function _update_station_gains!(g, touched, S, rseg, wseg, touching, bl_pairs, feeds, pinned)
    T = real(eltype(g))
    maxrel = zero(T)
    for feed in axes(g, Feed), ant in axes(g, Ant)
        pinned[ant, feed] && continue
        entries = touching[ant, feed]
        isempty(entries) && continue
        for fs in axes(g, Frequency)
            numer = zero(eltype(g))
            denom = zero(T)
            for (bi, p, role) in entries
                a, b = bl_pairs[bi]
                fa, fb = feeds[p]
                for si in axes(rseg, Scan)
                    w = wseg[si, bi, p, fs]
                    w > 0 || continue
                    s = S[si, bi, p]
                    if role === :a
                        coeff = s * conj(g[b, fb, fs])
                        abs2(coeff) > 0 || continue
                        numer += conj(coeff) * rseg[si, bi, p, fs]
                        denom += w * abs2(coeff)
                    else
                        coeff = conj(g[a, fa, fs] * s)
                        abs2(coeff) > 0 || continue
                        numer += conj(coeff) * conj(rseg[si, bi, p, fs])
                        denom += w * abs2(coeff)
                    end
                end
            end
            denom > 0 || continue
            gold = g[ant, feed, fs]
            gnew = numer / denom
            g[ant, feed, fs] = gnew
            touched[ant, feed, fs] = true
            maxrel = max(maxrel, abs(gnew - gold) / max(abs(gold), abs(gnew), eps(T)))
        end
    end
    return maxrel
end

# Gauge-fix each (station, feed) track — zero band-mean log-amplitude,
# circular-mean reference phase, matching solve_phase_bandpass!'s/
# solve_amp_bandpass!'s convention — and write into phase_plan's/amp_plan's θ
# blocks. A (station, feed) `touched` nowhere (no data ever reached it) is
# left unwritten (still whatever θ already held, i.e. unit gain).
function _write_joint_bandpass!(θ, phase_plan, amp_plan, g, touched, max_logamp)
    phase_leaf = _component_leaf(phase_plan, θ)
    amp_leaf = _component_leaf(amp_plan, θ)
    for a in axes(g, Ant), f in axes(g, Feed)
        valid = @view touched[a, f, :]
        any(valid) || continue
        logs = log.(abs.(@view g[a, f, :]))
        m = sum(view(logs, valid)) / count(valid)
        mphase = angle(sum(cis(angle(g[a, f, fs])) for fs in axes(g, Frequency) if touched[a, f, fs]))
        pnode = _feed_node(phase_plan.tying, f)
        anode = _feed_node(amp_plan.tying, f)
        for fs in axes(g, Frequency)
            touched[a, f, fs] || continue
            la = logs[fs] - m
            ph = rem2pi(angle(g[a, f, fs]) - mphase, RoundNearest)
            anode == 0 || (amp_leaf[1, anode, fs, 1, a] = abs(la) > max_logamp ? 0.0 : la)
            pnode == 0 || (phase_leaf[1, pnode, fs, 1, a] = ph)
        end
    end
    return θ
end

"""
    JointALS(; max_iterations = 8, tolerance = 1.0e-6)

An [`AbstractBandpassEstimator`](@ref) that fits the actual complex
visibilities per scan against an explicit per-scan, per-baseline,
per-polarization source coherence ([`solve_joint_bandpass!`](@ref)), instead
of the [`SplitWLS`](@ref) closure that assumes a baseline's source term
cancels — the right choice when the calibrator is resolved or polarized
enough that its per-baseline structure would otherwise bias the station
bandpass. Requires both `model.phase` and `model.amp` (it solves one complex
gain per (station, feed, segment), not independent phase/amp closures), and
ignores `model.amp_model` — no post-hoc smoothing pass.
"""
Base.@kwdef struct JointALS <: AbstractBandpassEstimator
    max_iterations::Int = 8
    tolerance::Real = 1.0e-6
end

bandpass_derotate(::JointALS) = false

function validate_bandpass(::JointALS, model::BandpassModel)
    model.phase && model.amp || throw(ArgumentError(
        "JointALS requires both model.phase = true and model.amp = true — it solves one " *
            "complex gain per (station, feed, segment), not independent phase/log-amp closures.",
    ))
    return nothing
end

function solve_bandpass!(est::JointALS, θ, results, setup, model::BandpassModel; ref_ant::Integer)
    solve_joint_bandpass!(
        θ, results, setup.bl_pairs, results[1].pols, setup.nant, setup.bp_plan, setup.amp_plan;
        ref_ant, max_iterations = est.max_iterations, tolerance = est.tolerance,
    )
    return nothing
end

"""
    solve_joint_bandpass!(θ, scans, bl_pairs, pol_products, nant, phase_plan, amp_plan;
                          ref_ant = 1, max_iterations = 8, tolerance = 1.0e-6,
                          max_logamp = log(10.0))

Jointly solve the per-(station, feed) COMPLEX bandpass gain and a per-scan,
per-baseline, per-polarization constant source coherence (see the module
comment above [`_reduce_scan_segments!`](@ref) for the model and the
alternating scheme), then gauge-fix each (station, feed) track and write the
result into `phase_plan`'s and `amp_plan`'s θ blocks
([`_write_joint_bandpass!`](@ref)).

`scans` is the per-scan `(rl, wl)` accumulator pairs from
[`accumulate_bandpass!`](@ref)`(...; derotate = false)` — NOT summed across
scans, since the source term needs each scan's own coherent visibility.
`phase_plan` and `amp_plan` must share one frequency segmentation (true by
construction: [`BandpassModel`](@ref) compiles both from the same `freq`
setting).

Convergence is judged on the largest relative per-iteration gain change, not a
tracked χ² (which would need a per-channel power accumulator this stage does
not keep).
"""
function solve_joint_bandpass!(
        θ, scans, bl_pairs, pol_products, nant, phase_plan, amp_plan;
        ref_ant::Integer = 1, max_iterations::Integer = 8, tolerance::Real = 1.0e-6,
        max_logamp::Real = log(10.0),
    )
    phase_plan.fseg_id == amp_plan.fseg_id || throw(
        ArgumentError(
            "solve_joint_bandpass!: phase_plan and amp_plan must share one frequency segmentation",
        ),
    )
    isempty(scans) && return θ

    feeds = [correlation_feed_pair(p) for p in pol_products]
    segs = segment_groups(phase_plan.fseg_id, length(phase_plan.nchan_seg))
    nseg = length(segs)

    rseg, wseg = _reduce_all_scans(scans, segs)
    touching = _joint_bandpass_touching(bl_pairs, feeds, nant)
    pins, graph_touched = _joint_bandpass_pins(bl_pairs, feeds, nant, ref_ant)
    pinned = [_node(ant, feed, nant) in pins for ant in 1:nant, feed in 1:2]

    C = eltype(rseg)
    gd = (Ant(1:nant), Feed(1:2), Frequency(1:nseg))
    g = DimensionalData.DimArray(ones(C, nant, 2, nseg), gd)
    touched = DimensionalData.DimArray(falses(nant, 2, nseg), gd)
    for node in pins
        graph_touched[node] || continue
        ant, feed = _ant_feed(node, nant)
        touched[ant, feed, :] .= true
    end
    S = DimensionalData.DimArray(
        zeros(C, length(scans), length(bl_pairs), length(pol_products)),
        (Scan(1:length(scans)), Baseline(1:length(bl_pairs)), Pol(1:length(pol_products))),
    )

    _update_source_coherence!(S, g, rseg, wseg, bl_pairs, feeds)
    for _iter in 1:max_iterations
        maxrel = _update_station_gains!(g, touched, S, rseg, wseg, touching, bl_pairs, feeds, pinned)
        _update_source_coherence!(S, g, rseg, wseg, bl_pairs, feeds)
        maxrel < tolerance && break
    end

    return _write_joint_bandpass!(θ, phase_plan, amp_plan, g, touched, max_logamp)
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

# Bucket observation indices by their frequency segment.
function _bandpass_obs_by_segment(ci, nseg)
    byc = [Int[] for _ in 1:nseg]
    for i in eachindex(ci)
        push!(byc[ci[i]], i)
    end
    return byc
end

# 1-D Whittaker smoother: minimise  Σ w_i (x_i − y_i)² + (λ·w̄) Σ (x_{i−1} − 2x_i + x_{i+1})²
# + ridge Σ x_i², via a WLSEstimator with `A = I`, the ridge term stacked as
# `√ridge · I` and the roughness term as `√λ · D` (`D` the 2nd-difference
# operator). `w_i = 0` (and `y_i` non-finite) where a channel had no data →
# the penalty alone sets it (interpolation). `λ` is scaled by the median
# positive weight so it is data-relative.
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
    est = WLSEstimator(
        (y, w) -> (Matrix(1.0I, n, n), y, w),
        A -> vcat(sqrt(ridge) .* Matrix(1.0I, n, n), sqrt(λ) .* D),
    )
    return est(yi, wi)
end
