# ── Bandpass stage: per-channel station phase/log-amp over scan windows ───────
#
# The carved-out bandpass stage of the composable pipeline: the per-scan
# residual accumulation ([`accumulate_bandpass!`](@ref)) and the two per-channel
# closure seed solves — descended from the monolithic solver (deleted at M5),
# retargeted from its concat cube to a scan `DimStack` where they touch data. The
# `Bandpass` step visits every scan (refine → accumulate → return the scan's
# contribution) and its `finish_pass!` folds the contributions in GROUP-INDEX
# order — deterministic at ANY concurrency (unlike the monolith's
# ntasks-dependent chunk fold; the two agree to float-rounding, gated at
# rtol ≤ 1e-12). WHAT is fit lives on `BandpassModel`; HOW it is solved is
# pluggable through `AbstractBandpassSmoother`, in two tiers. `PerTrackSmoother`
# sums every scan's residual into one accumulator, runs the per-channel closure
# solves (which assume a baseline's source term cancels) and fits each resulting
# (station, feed, spw) track under its shape spec. `JointSmoother` runs
# [`solve_joint_bandpass!`](@ref) instead, fitting the actual complex
# visibilities against an explicit per-scan source term — the right choice when
# that assumption fails — with the specs entering as priors inside the gain
# update. Both carry one [`AbstractShapeSpec`](@ref) per observable.
#
# The graph/solve helpers (`_ObsRow`, `_solve_observable`, `_track_noise2`,
# `_node`) live in stationize.jl/adhoc.jl; the shape specs live in shapes.jl.

# ── Amplitude closure incidence ───────────────────────────────────────────────
#
# The per-(station, feed) log-amp bandpass is solved from the SUM closure
# `log|V̄_ab(ν)| = la_a(ν) + lb_b(ν)` over each spw (a +1/+1, signless-Laplacian
# incidence — FULL RANK, so no reference state).

# Signless-Laplacian (SUM) incidence for one frequency segment's gated closure
# observations, restricted to the rows `idx`.
function _signless_incidence(na, nb, idx, nnodes, val, w)
    A = zeros(length(idx), nnodes)
    for (r, i) in enumerate(idx)
        A[r, na[i]] += 1.0; A[r, nb[i]] += 1.0
    end
    return A, val[idx], w[idx]
end

"""
    BandpassModel(; phase = true, amp = true, freq = ChannelBlocks(1))

WHAT the [`Bandpass`](@ref) step fits: whether to solve the phase bandpass, the
log-amplitude bandpass, or both, and how finely each is resolved in frequency
(`freq`, a [`ChannelBlocks`](@ref) — the default is one free value per channel).
Uniform across every antenna — no per-station segmentation. HOW each observable
is shaped lives on the step's smoother (see [`AbstractBandpassSmoother`](@ref)).
"""
Base.@kwdef struct BandpassModel
    phase::Bool = true
    amp::Bool = true
    freq::ChannelBlocks = ChannelBlocks(1)
end

"""
    AbstractBandpassSmoother

HOW the [`Bandpass`](@ref) step turns the accumulated per-channel residual into
station bandpass tracks, given the shape assumption each observable is fit under
(an [`AbstractShapeSpec`](@ref) per observable). Concretely
[`PerTrackSmoother`](@ref), which solves the per-channel closures and fits each
track, or [`JointSmoother`](@ref), which fits the complex visibilities against an
explicit per-scan source term with the specs as priors. Mirrors
[`AbstractFringeEstimator`](@ref)'s split between WHAT a step fits and HOW.

# Implementing a smoother

Define:

    Gustavo.Fringe.solve_bandpass!(sm::MySmoother, θ, results, setup, model::BandpassModel; gauge) -> report

writing into `θ`'s bandpass blocks. `results` is the per-scan
`(; rl, wl, pols, source)` accumulator list, in group-index order; `setup` is
`(; bl_pairs, blidx, nant, bp_plan, amp_plan, channel_freqs, spw_of_chan)`,
built once per pass. The fallback errors, naming what is missing.

`report` is published on the [`Bandpass`](@ref) step's solution record and should
say which tracks the solve actually measured — see
[`bandpass_track_report`](@ref), which builds it from per-track outcome codes.
Return `nothing` if a smoother has nothing to report; θ alone cannot express the
difference between a measured flat response and an unfitted one, so a smoother
that can tell them apart should.

Two optional hooks:

    Gustavo.Fringe.bandpass_derotate(sm::MySmoother) -> Bool   # default true
    Gustavo.Fringe.validate_bandpass(sm::MySmoother, model::BandpassModel)

[`bandpass_derotate`](@ref) controls whether [`accumulate_bandpass!`](@ref)
counter-rotates each AP before accumulating (see its docstring) — a smoother that
sums scans together needs it, one that fits each scan's own coherent visibility
does not. [`validate_bandpass`](@ref) is checked at model-compile time, before any
data is read; a method for a new smoother REPLACES the default, so it must state
at least as strong a requirement.
"""
abstract type AbstractBandpassSmoother end

bandpass_derotate(::AbstractBandpassSmoother) = true

"""
    validate_bandpass(sm::AbstractBandpassSmoother, model::BandpassModel)

Reject a [`BandpassModel`](@ref) that `sm` cannot solve, at the point the
[`Bandpass`](@ref) step compiles its components — before any data is read.

The default requires a model that fits SOMETHING: with both `phase` and `amp`
off the step compiles no components at all, so it would accumulate every scan
and write nowhere. A smoother with stricter needs defines its own method
(see [`JointSmoother`](@ref)), which replaces this one.
"""
function validate_bandpass(::AbstractBandpassSmoother, model::BandpassModel)
    # The step indexes its compiled components by name
    # (`layout.plantree.phase.bandpass` and its log-amp twin) under exactly
    # these two flags, so a model with neither set has nothing to solve into.
    model.phase || model.amp || throw(
        ArgumentError("BandpassModel fits nothing: at least one of `phase` or `amp` must be true."),
    )
    return nothing
end
function solve_bandpass! end
solve_bandpass!(sm::AbstractBandpassSmoother, θ, results, setup, model::BandpassModel; gauge) =
    error(
    "$(typeof(sm)) does not implement the bandpass smoother interface: define " *
        "Gustavo.Fringe.solve_bandpass!(::$(typeof(sm)), θ, results, setup, model; gauge)."
)

"""
    accumulate_bandpass!(rbar_bp, wbar_bp, blidx, stack, win::GeometryWindow; derotate = true)

Accumulate one scan window's contribution to the per-(global-baseline, product,
GLOBAL channel) coherent residual `rbar_bp` (and weight `wbar_bp`) for the
bandpass solves, on data already gain-corrected through the pipeline's
transform chain. `blidx` maps `(a, b) -> row` in the global baseline table;
a pair absent from it never contributes.

`derotate` (default `true`) counter-rotates each AP, BEFORE summing over time,
by its OWN band-averaged residual phase — removing the per-AP time phase
(residual rate/drift, and what the adhoc stage would later remove) so summing
COHERENT SCANS TOGETHER ([`PerTrackSmoother`](@ref), which combines every
selected scan's residual into one accumulator before solving) isolates the per-channel SHAPE
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
                    w = W[Frequency = c, Ti = tt, Baseline = bi, Pol = p]
                    vv = V[Frequency = c, Ti = tt, Baseline = bi, Pol = p]
                    cond = (w > 0 && isfinite(w) && isfinite(vv))
                    acc += ifelse(cond, w * vv, zero(eltype(V)))
                end
                rot = ifelse(abs(acc) > 0, conj(acc) / abs(acc), one(eltype(V))) # cis(-angle(acc)): de-rotate this AP
            end
            for c in axes(V, Frequency)
                w = W[Frequency = c, Ti = tt, Baseline = bi, Pol = p]
                vv = V[Frequency = c, Ti = tt, Baseline = bi, Pol = p]
                cond = (w > 0 && isfinite(w) && isfinite(vv))
                gc = g_ci[c]
                rbar_bp[idx, p, gc] += ifelse(cond, w * vv * rot, zero(eltype(V)))
                wbar_bp[idx, p, gc] += ifelse(cond, w, zero(eltype(W)))
            end
        end
    end
    return rbar_bp, wbar_bp
end


# Free per-segment closure seed for the phase bandpass: the globally-closing
# per-feed phase solved independently in each frequency segment, plus each
# (station, feed, segment)'s Fisher weight — the summed weight of the gated rows
# touching it, which is the diagonal of the segment's normal matrix and so the
# per-segment precision a shape fit weights the track by.
function _seed_phase_tracks(
        rbar_bp, wbar_bp, bl_pairs, pol_products, nant, segs;
        gauge::AbstractGauge = PinAntenna(1), snr_floor::Real = 1.0,
    )
    nbl, npol, nchan = size(rbar_bp)
    feeds = [correlation_feed_pair(p) for p in pol_products]
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in 1:nbl, p in 1:npol]
    nseg = length(segs)
    phase = fill(NaN, nant, 2, nseg)
    prec = zeros(nant, 2, nseg)
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
            prec[a, fa, fs] += snr2
            prec[b, fb, fs] += snr2
        end
        ph, _, _, _ = _solve_observable(rows, nant, gauge; rewrap = 0)
        phase[:, :, fs] .= ph
    end
    return phase, prec
end

# Gauge and write a solved phase bandpass: each (station, feed) track is
# referenced to its circular-mean phase over segments, so the bandpass carries
# SHAPE only and applies zero net phase.
function _write_phase_bandpass!(θ, plan, phase)
    nant, _, nseg = size(phase)
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


# Per-segment spw label and mean frequency for a bandpass plan's segmentation,
# with `spw_of_chan` empty meaning a single band. A frequency segment is the unit
# solved for, so it must lie within one spw: a shape is fit per spw and could not
# place a straddling segment.
function _segment_bands(plan, channel_freqs, spw_of_chan)
    nchan = length(channel_freqs)
    soc = isempty(spw_of_chan) ? ones(Int, nchan) : collect(spw_of_chan)
    fsegs = segment_groups(plan.fseg_id, length(plan.nchan_seg))
    seg_spw = map(fsegs) do chans
        s = soc[first(chans)]
        all(gc -> soc[gc] == s, chans) || throw(
            ArgumentError(
                "bandpass: a frequency segment straddles a spectral-window " *
                    "boundary; the bandpass segmentation must refine the spw partition.",
            ),
        )
        return s
    end
    seg_freq = [sum(channel_freqs[gc] for gc in chans) / length(chans) for chans in fsegs]
    return fsegs, seg_spw, seg_freq
end

# Narrow-spike guard: additive contamination (pcal tones, RFI) violates the
# multiplicative gain model — a contaminated channel shows EXCESS amplitude,
# the fit hands it |g| > 1, and `apply_calibration` would then UP-weight it
# (w → w·|g|²), amplifying exactly the channels that should be distrusted.
# Genuine passband structure is smooth or negative (roll-off), so narrow
# POSITIVE log-amp outliers vs the per-(station, feed, spw) robust scale are
# excised (left unapplied, |g| = 1) instead of trusted. `spike_sigma = 0`
# disables the guard.
function _spike_guard!(la, seg_spw, spike_sigma::Real)
    spike_sigma > 0 || return la
    nant, _, nfseg = size(la)
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
    return la
end

# Gauge and write a solved log-amp bandpass: zero band-mean per (station, feed),
# so the bandpass carries SHAPE only and applies unit net amplitude.
function _write_amp_bandpass!(θ, plan, la, max_logamp::Real)
    nant, _, nfseg = size(la)
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
            # Leave implausibly-large corrections UNAPPLIED (|g| = 1). A shape that
            # interpolates gaps self-regularizes, but an unconstrained fit can hand a
            # low-SNR band-edge segment that barely clears the gate a huge log-amp;
            # applying it would up-weight that segment's noise, since
            # `apply_calibration` scales weights by |g|². The bound is generous
            # (|g| ≤ 10) so real passband roll-off/structure passes unchanged — only
            # pathological noise blow-ups are gated.
            leaf[1, node, s, 1, a] = abs(val) > max_logamp ? 0.0 : val
        end
    end
    return θ
end

# Free per-segment closure seed for the log-amp bandpass: the SUM closure
# `log|V̄_ab| = la_a + la_b` solved independently in each frequency segment on the
# signless-Laplacian incidence (full rank, so no reference state), plus each
# (station, feed, segment)'s summed gate weight as its precision. Segments with no
# gated observation are left `NaN` for a shape fit to estimate — or not.
function _seed_amp_tracks(
        rbar_bp, wbar_bp, bl_pairs, pol_products, nant, fsegs;
        snr_floor::Real = 1.0, ridge::Real = 1.0e-6,
    )
    nbl, npol, nchan = size(rbar_bp)
    feeds = [correlation_feed_pair(p) for p in pol_products]
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in 1:nbl, p in 1:npol]
    nnodes = 2 * nant
    nfseg = length(fsegs)
    la = fill(NaN, nant, 2, nfseg)
    prec = zeros(nant, 2, nfseg)
    for (fs, chans) in enumerate(fsegs)
        na = Int[]; nbn = Int[]; vals = Float64[]; wts = Float64[]
        for bi in 1:nbl, p in 1:npol
            a, b = bl_pairs[bi]
            a == b && continue
            r, w, w2 = _segment_residual(rbar_bp, wbar_bp, bi, p, chans)
            (isfinite(r) && abs(r) > 0 && isfinite(w) && w > 0) || continue
            snr2 = _segment_snr2(r, w, w2, noise2[bi, p])
            snr2 >= snr_floor^2 || continue
            amp = abs(r / w); amp > 0 || continue
            fa, fb = feeds[p]
            push!(na, _node(a, fa, nant)); push!(nbn, _node(b, fb, nant))
            push!(vals, log(amp)); push!(wts, snr2)
            prec[a, fa, fs] += snr2
            prec[b, fb, fs] += snr2
        end
        isempty(vals) && continue
        A, bvec, wvec = _signless_incidence(na, nbn, eachindex(vals), nnodes, vals, wts)
        sol = weighted_regularized_least_squares(A, bvec, wvec, fill(float(ridge), nnodes))
        for i in eachindex(vals), node in (na[i], nbn[i])
            ant = (node - 1) % nant + 1
            feed = (node - 1) ÷ nant + 1
            la[ant, feed, fs] = sol[node]
        end
    end
    return la, prec
end

# ── Per-track bandpass smoothing (seed closure solve → per-track shape fit) ────

const _BP_SPIKE_SIGMA = 5.0
const _BP_MAX_LOGAMP = log(10.0)

# Outcome of fitting ONE (station, feed, spw) bandpass track, reported per track by
# `bandpass_track_report` so a caller can tell a measurement from a placeholder.
# `θ` carries no such distinction: an unfitted track reads back as unit gain and a
# starved one as a constant, both indistinguishable from a real flat response.
const _BP_TRACK_NODATA = Int8(0)      # no usable segment; left at unit gain
const _BP_TRACK_SOLVED = Int8(1)      # fit, with frequency structure
const _BP_TRACK_FLAT = Int8(2)        # fit, but constant to within `_BP_FLAT_SPAN`
const _BP_TRACK_DECLINED = Int8(3)    # phase branch undetermined; not fit

const _BP_TRACK_LABELS = ("nodata", "solved", "flat", "declined")

# A fitted track this flat carries no shape: reported as `_BP_TRACK_FLAT` rather
# than silently passed off as a measured response. In radians for a phase track and
# nepers for a log-amplitude one — both are the observable's own natural unit and a
# hundredth of it is far below any real passband feature.
const _BP_FLAT_SPAN = 0.01

# Above this fraction of coin-flip steps (`phase_unwrap_ambiguity`) a phase track's
# 2π branch is not determined by the data, and the smooth trend a shape spec then
# fits through the unwrap's random walk is an artifact of the walk. Such a track is
# declined rather than fit: unit gain is honest about knowing nothing, an invented
# multi-radian ramp is not.
const _BP_MAX_UNWRAP_AMBIGUITY = 0.25

# Fraction of a solve's tracks that may come back flat or declined before the
# bandpass as a whole is worth a warning.
const _BP_DEGENERATE_WARN_FRACTION = 0.25

"""
    PerTrackSmoother(; phase = FreeShape(), amp = WhittakerShape(1.0))

Solve the bandpass one (station, feed, spw) track at a time: the per-segment
closure graph solve seeds real phase and log-amp tracks, then each track is fit
under its own shape spec ([`fit_track`](@ref)) — `phase` and `amp` are
parameterized separately because the two carry different frequency smoothness
(nonlinear instrumental phase vs. passband roll-off).

Phase tracks are unwrapped along frequency before the fit, so a shape sees a
continuous branch rather than a sawtooth. A spec that estimates gaps
([`PolynomialShape`](@ref), [`WhittakerShape`](@ref), [`ARShape`](@ref)) fills
segments the closure solve had no data for; [`FreeShape`](@ref) leaves them
unapplied.

The closure assumes a baseline's source term cancels out of the per-channel
phase-difference / log-amp-sum, so it is not appropriate for a resolved or
polarized calibrator (see [`JointSmoother`](@ref)).
"""
struct PerTrackSmoother{P <: AbstractShapeSpec, A <: AbstractShapeSpec} <: AbstractBandpassSmoother
    phase::P
    amp::A
end
PerTrackSmoother(; phase::AbstractShapeSpec = FreeShape(), amp::AbstractShapeSpec = WhittakerShape(1.0)) =
    PerTrackSmoother(phase, amp)

# The outcome code for one fitted spw track: nothing estimated, a constant, or a
# real shape.
function _band_track_status(fitted)
    obs = [v for v in fitted if isfinite(v)]
    isempty(obs) && return _BP_TRACK_NODATA
    return (maximum(obs) - minimum(obs)) < _BP_FLAT_SPAN ? _BP_TRACK_FLAT : _BP_TRACK_SOLVED
end

# Fit one segment-indexed track under `spec`, split at the spw boundaries — a
# shape describes the response WITHIN a band, so segments never pool across spws
# and each band keeps its own free level. `fit_track_group` decides what, if
# anything, the bands share: only a spec that ESTIMATES its shape parameters pools
# them, and it pools the parameters alone, never the levels.
#
# `unwrap` re-references a phase track to a continuous branch along frequency
# first: the specs fit a real track, and the ±π branch cuts of a raw phase solve
# would otherwise read as genuine structure. A band whose branch the data does not
# determine (`phase_unwrap_ambiguity` past `_BP_MAX_UNWRAP_AMBIGUITY`) is dropped
# instead — see that constant.
#
# `status` receives one `_BP_TRACK_*` code per band, in ascending band order.
function _fit_track_bands(
        spec::AbstractShapeSpec, y, w, seg_spw, seg_freq;
        unwrap::Bool = false, status = nothing,
    )
    out = fill(NaN, length(y))
    bands = sort(unique(seg_spw))
    members = [[s for s in eachindex(seg_spw) if seg_spw[s] == bnd] for bnd in bands]
    ys = Vector{Vector{Float64}}(undef, length(bands))
    ws = Vector{Vector{Float64}}(undef, length(bands))
    xs = Vector{Vector{Float64}}(undef, length(bands))
    declined = falses(length(bands))
    for (j, sidx) in enumerate(members)
        yy = Float64[y[s] for s in sidx]
        ws[j] = Float64[w[s] for s in sidx]
        xs[j] = Float64[seg_freq[s] for s in sidx]
        if unwrap
            if phase_unwrap_ambiguity(yy; weights = ws[j]) > _BP_MAX_UNWRAP_AMBIGUITY
                declined[j] = true
                fill!(yy, NaN)
            else
                yy = unwrap_phase_track(yy; weights = ws[j])
            end
        end
        ys[j] = yy
    end
    fitted = fit_track_group(spec, ys, ws, xs)
    for (j, sidx) in enumerate(members)
        for (i, s) in enumerate(sidx)
            out[s] = fitted[j][i]
        end
        status === nothing && continue
        status[j] = declined[j] ? _BP_TRACK_DECLINED : _band_track_status(fitted[j])
    end
    return out
end

# Fit every (station, feed, spw) track of `tracks` in place under `spec`, each
# segment weighted by its seed precision. `status`, when given, is an
# `(Ant, Feed, band)` array receiving each track's `_BP_TRACK_*` outcome.
function _shape_tracks!(
        tracks, prec, seg_spw, seg_freq, spec::AbstractShapeSpec; unwrap::Bool, status = nothing,
    )
    nant, _, nfseg = size(tracks)
    for a in 1:nant, f in 1:2
        y = [tracks[a, f, s] for s in 1:nfseg]
        w = [prec[a, f, s] for s in 1:nfseg]
        st = status === nothing ? nothing : view(status, a, f, :)
        fitted = _fit_track_bands(spec, y, w, seg_spw, seg_freq; unwrap, status = st)
        for s in 1:nfseg
            tracks[a, f, s] = fitted[s]
        end
    end
    return tracks
end

"""
    bandpass_track_report(phase_status, amp_status, band_ids) -> NamedTuple

Summarize a bandpass solve's per-(station, feed, spw) outcomes into the record the
[`Bandpass`](@ref) step publishes. `phase_status`/`amp_status` are `(Ant, Feed,
band)` arrays of `_BP_TRACK_*` codes (either may be `nothing` when that half was
not fit); `band_ids` names the spw each band slot came from.

Returns the two arrays as `phase_status`/`amp_status` alongside `band_ids`,
`track_labels` (the code → name mapping, so a reader needs no constant from this
module) and the counts `n_solved`/`n_flat`/`n_declined`/`n_nodata` summed over both
observables. `flat` and `declined` are the two ways a track can occupy a slot
without measuring anything, and they are what the counts exist to expose: θ itself
records an unfitted track as unit gain and a starved one as a constant, neither
distinguishable there from a genuinely flat response.
"""
function bandpass_track_report(phase_status, amp_status, band_ids)
    counts = zeros(Int, 4)
    for st in (phase_status, amp_status), c in something(st, Int8[])
        counts[Int(c) + 1] += 1
    end
    # Concrete arrays throughout — the record is serialized with the solution, and
    # an observable that was not fit is an EMPTY status rather than a missing field.
    empty_status = Array{Int8, 3}(undef, 0, 0, 0)
    return (;
        phase_status = something(phase_status, empty_status),
        amp_status = something(amp_status, empty_status),
        band_ids = collect(Int, band_ids),
        track_labels = collect(String, _BP_TRACK_LABELS),
        n_nodata = counts[1], n_solved = counts[2],
        n_flat = counts[3], n_declined = counts[4],
    )
end

# Warn when a large share of the tracks measured nothing. Silence here would leave
# a bandpass that is mostly placeholder looking exactly like one that is mostly
# measured — the caller cannot tell from θ, which is why this is a warning and not
# only a record.
function _warn_degenerate_bandpass(report)
    total = report.n_nodata + report.n_solved + report.n_flat + report.n_declined
    total > 0 || return nothing
    degenerate = report.n_flat + report.n_declined + report.n_nodata
    frac = degenerate / total
    frac > _BP_DEGENERATE_WARN_FRACTION || return nothing
    @warn """
    Bandpass: $(round(100 * frac; digits = 1))% of (station, feed, spw) tracks carry no measured \
    frequency shape — $(report.n_flat) fit flat, $(report.n_declined) declined for an \
    undetermined phase branch, $(report.n_nodata) with no usable data (of $total). \
    They are unit gain or a constant in the solution, not a measured response. \
    Check per-spw SNR and the detection coverage of the bandpass scans.
    """ n_solved = report.n_solved
    return nothing
end

function solve_bandpass!(sm::PerTrackSmoother, θ, results, setup, model::BandpassModel; gauge::AbstractGauge)
    pols = results[1].pols
    nchan = length(setup.channel_freqs)
    rbar, wbar = bandpass_accumulators(length(setup.bl_pairs), length(pols), nchan)
    for res in results
        rbar .+= res.rl
        wbar .+= res.wl
    end
    phase_status = nothing
    amp_status = nothing
    band_ids = Int[]
    if setup.bp_plan !== nothing
        plan = setup.bp_plan
        fsegs, seg_spw, seg_freq = _segment_bands(plan, setup.channel_freqs, setup.spw_of_chan)
        band_ids = sort(unique(seg_spw))
        phase, prec = _seed_phase_tracks(
            rbar, wbar, setup.bl_pairs, pols, setup.nant, fsegs; gauge,
        )
        phase_status = fill(_BP_TRACK_NODATA, setup.nant, 2, length(band_ids))
        _shape_tracks!(phase, prec, seg_spw, seg_freq, sm.phase; unwrap = true, status = phase_status)
        _write_phase_bandpass!(θ, plan, phase)
    end
    if setup.amp_plan !== nothing
        plan = setup.amp_plan
        fsegs, seg_spw, seg_freq = _segment_bands(plan, setup.channel_freqs, setup.spw_of_chan)
        band_ids = sort(unique(seg_spw))
        la, prec = _seed_amp_tracks(rbar, wbar, setup.bl_pairs, pols, setup.nant, fsegs)
        _spike_guard!(la, seg_spw, _BP_SPIKE_SIGMA)
        amp_status = fill(_BP_TRACK_NODATA, setup.nant, 2, length(band_ids))
        _shape_tracks!(la, prec, seg_spw, seg_freq, sm.amp; unwrap = false, status = amp_status)
        _write_amp_bandpass!(θ, plan, la, _BP_MAX_LOGAMP)
    end
    report = bandpass_track_report(phase_status, amp_status, band_ids)
    _warn_degenerate_bandpass(report)
    return report
end

# ── Joint complex bandpass + per-scan source coherence (ALS) ──────────────────
#
# The per-channel closure solves assume a baseline's source term
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
# accumulator — no intermediate allocation).
#
# NOT SNR-gated, deliberately. The joint solve consumes these as COMPLEX
# residuals under inverse-variance weights, and that accumulation is unbiased
# at any SNR — a weak cell contributes its information at its honest weight
# and costs variance, never validity. An SNR gate here (the closure tier's,
# which is justified THERE because that tier extracts a per-segment PHASE, a
# meaningless quantity below the noise) would preferentially delete the
# cross-hand cells — the only rows that tie the feed-2 gain block to feed-1
# and so the only measurement of the relative (R–L) bandpass — leaving that
# block at its initialization. Outlier handling is a pipeline concern (a
# dedicated flagging step upstream of the solve), not a cell gate's.
function _reduce_scan_segments!(rview, wview, sc, segs)
    for p in axes(rview, Pol), bi in axes(rview, Baseline)
        for (fs, chans) in enumerate(segs)
            rc, wc, _ = _segment_residual(sc.rl, sc.wl, bi, p, chans)
            keep = isfinite(rc) && isfinite(wc) && wc > 0
            rview[bi, p, fs] = keep ? rc : zero(rc)
            wview[bi, p, fs] = keep ? wc : zero(wc)
        end
    end
    return nothing
end

# Every scan's (Baseline, Pol, SEGMENT) residual, stacked over an added Scan
# axis — NOT summed across scans (unlike the closure tier's fold), since the per-scan source coherence needs each
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

# One reference (station, feed) node per connected component of the
# (station, feed) graph — mirrors `_solve_observable`'s pin selection in
# stationize.jl. The
# pinned node's PHASE is held at zero for every segment throughout the ALS
# iteration; its amplitude is solved like any other node's.
#
# Only the phase is a gauge freedom. Multiplying every station's gain at one
# segment by a shared `c` sends `g_a·S·conj(g_b)` to `|c|²·g_a·S·conj(g_b)`: the
# phase of `c` cancels between the two conjugated factors, so a common phase per
# segment is unobservable and must be pinned there, one constraint per segment.
# The magnitude does NOT cancel, and `S` is frequency-flat, so it can only absorb
# `|c|²` when `|c|` is constant across the band — leaving exactly ONE free
# amplitude parameter overall, which the zero-band-mean gauge in
# [`_write_joint_bandpass!`](@ref) removes. Pinning `|g|` per segment as well
# would assert the reference antenna has a flat amplitude bandpass, discarding
# structure that IS identifiable (mean-removing `log|V_ab| = la_a + la_b + ls_ab`
# over the band eliminates `ls` and leaves the full-rank signless-Laplacian
# system) and biasing every other station through the inconsistency.
function _joint_bandpass_pins(bl_pairs, feeds, nant, gauge)
    nnodes = 2 * nant
    edges = Tuple{Int, Int}[]
    # Node degree stands in for the row weight the gauge scores elsewhere: this
    # graph is built from the baselines that EXIST, before any per-channel gating,
    # so the number of correlations touching a node is the observation count
    # available to anchor it.
    deg = zeros(Int, nnodes)
    for p in eachindex(feeds), bi in eachindex(bl_pairs)
        a, b = bl_pairs[bi]
        a == b && continue
        fa, fb = feeds[p]
        na, nb = _node(a, fa, nant), _node(b, fb, nant)
        push!(edges, (na, nb))
        deg[na] += 1
        deg[nb] += 1
    end
    compid, ncomp, _ = connected_components(nnodes, edges)
    # Inverse of `_node`: feed-1 block 1:nant, feed-2 nant+1:2nant.
    station_of(n) = (n - 1) % nant + 1
    feed_of(n) = n > nant ? 2 : 1
    pins = Set{Int}()
    for c in 1:ncomp
        comp_nodes = findall(==(c), compid)
        push!(pins, gauge_anchor(gauge, comp_nodes, deg, station_of, feed_of))
    end
    return pins
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
# and every OTHER station's current gain (immediately visible to later antennas
# in the same sweep — Gauss-Seidel, not Jacobi), with each observable's shape
# spec acting as a PRIOR on the resulting track rather than a post-hoc smooth.
#
# Solving each segment independently would be the `FreeShape`/`FreeShape` case;
# the prior enters exactly where that independence is dropped. Around the
# unconstrained per-segment estimate `ĝ` the residual linearizes as
# `Σ denom·|ĝ|²·(δlogamp² + δphase²)`, so `denom·|ĝ|²` is the Fisher weight BOTH
# real tracks are fit under, and the fit is a penalized WLS against the spec.
# Iterated to convergence with the relinearization this is MAP estimation under
# the two priors.
#
# `φ` carries each (station, feed)'s UNWRAPPED phase track across sweeps: it is
# unwrapped once, when `seed` (the first sweep) initializes it, and thereafter
# advanced by wrapped increments about its own current value, so no global unwrap
# is needed inside the loop and the 2π branch cannot flip between iterations.
# Returns the largest relative gain change, for the caller's convergence check.
function _update_station_gains!(
        g, φ, touched, S, rseg, wseg, touching, bl_pairs, feeds, pinned,
        phase_spec, amp_spec, seg_spw, seg_freq; seed::Bool,
        phase_status = nothing, amp_status = nothing,
    )
    T = real(eltype(g))
    C = eltype(g)
    maxrel = zero(T)
    nseg = size(g, Frequency)
    ĝ = Vector{C}(undef, nseg)
    wf = Vector{T}(undef, nseg)
    la = Vector{T}(undef, nseg)
    φ̃ = Vector{T}(undef, nseg)
    for feed in axes(g, Feed), ant in axes(g, Ant)
        entries = touching[ant, feed]
        isempty(entries) && continue
        # The gauge pin fixes this node's PHASE at every segment; its amplitude
        # is solved like any other node's (see `_joint_bandpass_pins`).
        ispin = pinned[ant, feed]
        fill!(ĝ, zero(C))
        fill!(wf, zero(T))
        for fs in axes(g, Frequency)
            numer = zero(C)
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
            gh = denom > 0 ? numer / denom : zero(C)
            ĝ[fs] = gh
            wf[fs] = (denom > 0 && abs(gh) > 0) ? denom * abs2(gh) : zero(T)
        end
        for fs in 1:nseg
            if wf[fs] > 0
                la[fs] = log(abs(ĝ[fs]))
                # The wrapped increment about this track's current value keeps the
                # candidate on the same 2π branch as the iterate it refines.
                φ̃[fs] = φ[ant, feed, fs] +
                    rem2pi(angle(ĝ[fs]) - φ[ant, feed, fs], RoundNearest)
            else
                la[fs] = T(NaN)
                φ̃[fs] = T(NaN)
            end
        end
        # The status of the LAST sweep is the status of the solve: each sweep
        # overwrites the previous one's codes for this node.
        ast = amp_status === nothing ? nothing : view(amp_status, ant, feed, :)
        pst = phase_status === nothing ? nothing : view(phase_status, ant, feed, :)
        la_new = _fit_track_bands(amp_spec, la, wf, seg_spw, seg_freq; status = ast)
        φ_new = if ispin
            # The pin's phase is fixed by the gauge at every segment, so it is known
            # rather than fitted: report it as such instead of leaving it at NODATA.
            pst === nothing || fill!(pst, _BP_TRACK_SOLVED)
            zeros(T, nseg)
        else
            _fit_track_bands(phase_spec, φ̃, wf, seg_spw, seg_freq; unwrap = seed, status = pst)
        end
        for fs in 1:nseg
            (isfinite(la_new[fs]) && isfinite(φ_new[fs])) || continue
            gold = g[ant, feed, fs]
            gnew = exp(C(la_new[fs], φ_new[fs]))
            g[ant, feed, fs] = gnew
            φ[ant, feed, fs] = φ_new[fs]
            touched[ant, feed, fs] = true
            maxrel = max(maxrel, abs(gnew - gold) / max(abs(gold), abs(gnew), eps(T)))
        end
    end
    return maxrel
end

# Gauge-fix each (station, feed) track — zero band-mean log-amplitude,
# circular-mean reference phase, matching the closure tier's convention — and write into phase_plan's/amp_plan's θ
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
    JointSmoother(; phase = FreeShape(), amp = FreeShape(),
                  max_iterations = 8, tolerance = 1.0e-6)

Fit the station bandpass against the actual complex visibilities, with each
observable's shape spec acting as a PRIOR inside the solve rather than a fit
applied to it afterwards. The alternating complex-gain / per-scan
source-coherence scheme is [`solve_joint_bandpass!`](@ref)'s; what `phase` and
`amp` change is the per-(station, feed) gain update, which fits the whole track
under the spec instead of solving each frequency segment on its own. Iterated to
convergence that is MAP estimation under the two priors.

Because the per-scan source term absorbs a baseline's own structure, this suits a
resolved or polarized calibrator, where [`PerTrackSmoother`](@ref)'s closure
assumption would bias the bandpass. It solves one complex gain per
(station, feed, segment), so it requires both `model.phase` and `model.amp`.
"""
struct JointSmoother{P <: AbstractShapeSpec, A <: AbstractShapeSpec} <: AbstractBandpassSmoother
    phase::P
    amp::A
    max_iterations::Int
    tolerance::Float64
end
function JointSmoother(;
        phase::AbstractShapeSpec = FreeShape(), amp::AbstractShapeSpec = FreeShape(),
        max_iterations::Integer = 8, tolerance::Real = 1.0e-6,
    )
    return JointSmoother(phase, amp, Int(max_iterations), Float64(tolerance))
end

# The joint tier fits each scan's OWN coherent visibility against an explicit
# source term, so the per-AP derotation that lets scans be summed together would
# erase the very source phase that term absorbs.
bandpass_derotate(::JointSmoother) = false

function validate_bandpass(::JointSmoother, model::BandpassModel)
    model.phase && model.amp || throw(
        ArgumentError(
            "JointSmoother requires both model.phase = true and model.amp = true — it solves " *
                "one complex gain per (station, feed, segment), not independent phase/log-amp tracks.",
        ),
    )
    return nothing
end

function solve_bandpass!(sm::JointSmoother, θ, results, setup, model::BandpassModel; gauge::AbstractGauge)
    _, seg_spw, seg_freq = _segment_bands(setup.bp_plan, setup.channel_freqs, setup.spw_of_chan)
    band_ids = sort(unique(seg_spw))
    phase_status = fill(_BP_TRACK_NODATA, setup.nant, 2, length(band_ids))
    amp_status = fill(_BP_TRACK_NODATA, setup.nant, 2, length(band_ids))
    solve_joint_bandpass!(
        θ, results, setup.bl_pairs, results[1].pols, setup.nant, setup.bp_plan, setup.amp_plan;
        gauge, max_iterations = sm.max_iterations, tolerance = sm.tolerance,
        phase_spec = sm.phase, amp_spec = sm.amp, seg_spw, seg_freq,
        phase_status, amp_status,
    )
    report = bandpass_track_report(phase_status, amp_status, band_ids)
    _warn_degenerate_bandpass(report)
    return report
end

"""
    solve_joint_bandpass!(θ, scans, bl_pairs, pol_products, nant, phase_plan, amp_plan;
                          gauge = PinAntenna(1), max_iterations = 8, tolerance = 1.0e-6,
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

`phase_status`/`amp_status`, when given, are `(Ant, Feed, band)` arrays that
receive each track's `_BP_TRACK_*` outcome code from the final sweep.
"""
function solve_joint_bandpass!(
        θ, scans, bl_pairs, pol_products, nant, phase_plan, amp_plan;
        gauge::AbstractGauge = PinAntenna(1), max_iterations::Integer = 8, tolerance::Real = 1.0e-6,
        max_logamp::Real = _BP_MAX_LOGAMP,
        phase_spec::AbstractShapeSpec = FreeShape(),
        amp_spec::AbstractShapeSpec = FreeShape(),
        seg_spw::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        seg_freq::Union{Nothing, AbstractVector{<:Real}} = nothing,
        phase_status = nothing,
        amp_status = nothing,
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
    pins = _joint_bandpass_pins(bl_pairs, feeds, nant, gauge)
    pinned = [_node(ant, feed, nant) in pins for ant in 1:nant, feed in 1:2]

    # A shape describes the response within one band; with no segmentation given
    # the whole solve is one band indexed by segment.
    bands = seg_spw === nothing ? ones(Int, nseg) : seg_spw
    coords = seg_freq === nothing ? collect(1.0:nseg) : seg_freq

    C = eltype(rseg)
    gd = (Ant(1:nant), Feed(1:2), Frequency(1:nseg))
    g = DimensionalData.DimArray(ones(C, nant, 2, nseg), gd)
    # The unwrapped phase track behind `g`, carried across sweeps so the shape fit
    # never sees a 2π branch cut.
    φ = DimensionalData.DimArray(zeros(real(C), nant, 2, nseg), gd)
    # Every node — pinned or not — is marked solved by the gain update, at the
    # segments it actually has data for. A pinned node's phase is known
    # everywhere by the gauge, but its amplitude is not, so it earns its slots
    # the same way the rest do.
    touched = DimensionalData.DimArray(falses(nant, 2, nseg), gd)
    S = DimensionalData.DimArray(
        zeros(C, length(scans), length(bl_pairs), length(pol_products)),
        (Scan(1:length(scans)), Baseline(1:length(bl_pairs)), Pol(1:length(pol_products))),
    )

    _update_source_coherence!(S, g, rseg, wseg, bl_pairs, feeds)
    for iter in 1:max_iterations
        maxrel = _update_station_gains!(
            g, φ, touched, S, rseg, wseg, touching, bl_pairs, feeds, pinned,
            phase_spec, amp_spec, bands, coords; seed = iter == 1,
            phase_status, amp_status,
        )
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
