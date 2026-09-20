# ── Bandpass stage: per-channel station phase/log-amp over scan windows ───────
#
# The carved-out bandpass stage of the composable pipeline: the per-scan
# residual accumulation (`accumulate_bandpass!`) and the two per-channel
# closure seed solves — descended from the monolithic solver (deleted at M5),
# retargeted from its concat cube to a scan `DimStack` where they touch data. The
# `Bandpass` step visits every scan (refine → accumulate → return the scan's
# contribution) and its `finish_pass!` folds the contributions in group-INDEX
# order — deterministic at any concurrency (unlike the monolith's
# ntasks-dependent chunk fold; the two agree to float-rounding, gated at
# rtol ≤ 1e-12). What is fit is the step's model tree (see
# [`default_bandpass_terms`](@ref)); how it is solved is
# pluggable through `AbstractBandpassSmoother`, in two tiers. `PerTrackSmoother`
# sums every scan's residual into one accumulator, runs the per-channel closure
# solves (which assume a baseline's source term cancels) and fits each resulting
# (station, feed, spw) track under its shape spec. `JointSmoother` runs
# [`solve_joint_bandpass!`](@ref) instead, fitting the actual complex
# visibilities against an explicit per-scan source term — the right choice when
# that assumption fails — with the specs entering as priors inside the gain
# update. Both carry one [`AbstractShapeSpec`](@ref Gustavo.Fringe.AbstractShapeSpec) per observable.
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
    default_bandpass_terms(; freq = ChannelBlocks(1)) -> NamedTuple

The default [`Bandpass`](@ref Gustavo.Bandpass) step model: a time-stable, per-feed constant per
frequency segment for each observable — a `phase.bandpass` and a
`logamp.bandpass` component, both resolved by `freq` (an
`AbstractFrequencySegmentation`; the default is one free value per channel).

A `Bandpass` model is a `(; phase, logamp)` tree of named
`Calibration.GainComponent`s (or a `StationGainModel`); either group may be
absent or empty. Fit one observable only by keeping just that group, e.g.

    Bandpass(model = (; phase = default_bandpass_terms().phase),
             smoother = PerTrackSmoother())

fits the phase bandpass alone. Uniform across every antenna. How each
observable is shaped lives on the step's smoother (see
[`AbstractBandpassSmoother`](@ref)).
"""
default_bandpass_terms(; freq::AbstractFrequencySegmentation = ChannelBlocks(1)) = (;
    phase = (; bandpass = _bandpass_component(freq)),
    logamp = (; bandpass = _bandpass_component(freq)),
)

_bandpass_component(freq) =
    GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = freq, Feed = PerFeed())

"""
    AbstractBandpassSmoother

How the [`Bandpass`](@ref Gustavo.Bandpass) step turns the accumulated
per-channel residual into station bandpass tracks, given a shape assumption
per observable (an [`AbstractShapeSpec`](@ref Gustavo.Fringe.AbstractShapeSpec)).
Concretely [`PerTrackSmoother`](@ref), which solves the per-channel closures
and fits each track, or [`JointSmoother`](@ref), which fits the complex
visibilities against an explicit per-scan source term.

# Implementing a smoother

Define:

    Gustavo.Fringe.can_fit(sm::MySmoother, tc::Calibration.GainComponent) -> Bool
    Gustavo.Fringe.solve_bandpass!(sm::MySmoother, θ, results, setup; gauge) -> report

[`can_fit`](@ref) declares which model components the smoother can solve; it
defaults to `false`, so an undeclared model is rejected at compile time
rather than leaving θ blocks unsolved. Both shipped smoothers accept
`GainComponent(ConstantTerm(); Ti = <GlobalTime, InstrumentScans or TimeBlocks>,
Frequency = <any segmentation>, Feed = PerFeed())` and nothing else — a time
segmentation whose segments each span several scans, solved one segment at a time.

`solve_bandpass!` writes into `θ`'s bandpass blocks. `results` is the
per-scan `(; rl, wl, pols, ti, source)` accumulator list in group-index order,
`ti` being the scan's first sample on the solve's time axis (hence which time
segment it falls in);
`setup` is `(; bl_pairs, blidx, nant, layout, bp_path, amp_path, channel_freqs, spw_of_chan)`,
built once per pass. Reach each observable's parameters through
[`bandpass_blocks`](@ref)`(setup, θ, :phase)` / `(…, :logamp)` rather than the
paths directly. `report` is published on the step's solution record and
should say which tracks were measured (see [`bandpass_track_report`](@ref));
θ alone cannot distinguish a measured flat response from an unfitted one.
Return `nothing` to report nothing.

Two optional hooks:

    Gustavo.Fringe.bandpass_derotate(sm::MySmoother) -> Bool   # default true
    Gustavo.Fringe.validate_model(sm::MySmoother, model)

`bandpass_derotate` controls whether `accumulate_bandpass!` counter-rotates
each AP before accumulating: a smoother that sums scans together needs it;
one that fits each scan's own coherent visibility does not. `validate_model`
receives the whole `(; phase, logamp)` tree at compile time for requirements
`can_fit` cannot express per component. A new method replaces the default,
so it must re-establish the base checks, available as
[`validate_bandpass_groups`](@ref).
"""
abstract type AbstractBandpassSmoother end

bandpass_derotate(::AbstractBandpassSmoother) = true

# `can_fit`/`validate_model` are the same compile-time capability seam the
# fringe estimators use (see estimators.jl); the `false` default makes an
# undeclared smoother reject loudly instead of accepting silently.
can_fit(::AbstractBandpassSmoother, tc) = false

# A bandpass track is one free constant per frequency segment, per feed, held
# over a stretch of time. Both smoothers pool the scans of a time segment and
# solve that stretch as a unit, so any segmentation whose segments span scans
# fits: `GlobalTime` is the whole track, `InstrumentScans` breaks it at named
# epochs, `TimeBlocks` at a fixed cadence. `PerScan` and `PerIntegration` do not
# — a segment holding one scan leaves the band mean of the gain degenerate with
# that scan's own source coherence, which needs the deviation-from-template
# scheme neither smoother implements.
_fits_bandpass_track(tc) =
    tc.term isa ConstantTerm &&
    tc.Ti isa Union{GlobalTime, InstrumentScans, TimeBlocks} &&
    tc.Feed isa PerFeed

"""
    validate_bandpass_groups(model)

The structural requirement the [`Bandpass`](@ref Gustavo.Bandpass) step
places on every smoother's model: at least one component overall, and at
most one per group (the step hands `solve_bandpass!` one plan per
observable). The default `validate_model` for
[`AbstractBandpassSmoother`](@ref), and the base a smoother's own method
must re-establish.
"""
function validate_bandpass_groups(model)
    np = length(Calibration._flatten_components(model.phase))
    na = length(Calibration._flatten_components(model.logamp))
    np + na >= 1 || throw(
        ArgumentError(
            "Bandpass model fits nothing: the model compiles no components, so the step " *
                "would accumulate every scan and write nowhere. Add a phase and/or logamp " *
                "component — `default_bandpass_terms()` is the standard model.",
        ),
    )
    for (name, n) in ((:phase, np), (:logamp, na))
        n <= 1 || throw(
            ArgumentError(
                "Bandpass model group `$name` holds $n components; the bandpass solve fits " *
                    "one track set per observable, so each group holds at most one.",
            ),
        )
    end
    return nothing
end

validate_model(sm::AbstractBandpassSmoother, model) = validate_bandpass_groups(model)

# The path to a bandpass observable's component in a step layout's plantree,
# `nothing` when that group compiles none. `validate_bandpass_groups` caps each
# group at one component, so every level of the descent carries exactly one key
# and the path is unambiguous however the user nested the names. The plantree's
# type is a compile-time constant, so accumulating the path as a tuple keeps the
# descent inferrable and the splat into `station_blocks` type-stable.
_bandpass_path(plantree, group::Symbol) = _component_path(plantree[group], (group,))

# Descend a plantree subtree to its single component; a node that is not a named
# subtree is that component.
_component_path(_, path::Tuple{Vararg{Symbol}}) = path
function _component_path(nt::NamedTuple, path::Tuple{Vararg{Symbol}})
    isempty(nt) && return nothing
    k = only(keys(nt))
    return _component_path(nt[k], (path..., k))
end

"""
    bandpass_blocks(setup, θ, group::Symbol) -> Vector

The station blocks of the bandpass's `:phase` or `:logamp` observable —
[`Calibration.station_blocks`](@ref) resolved against `setup`'s recorded path —
empty when the model compiles no component for that observable. Each block is
`(; stations, θ, plan)`, and a station-uniform model yields exactly one block
spanning every station.

The path is resolved by NAME through the layout's component tree: the flat
`layout.plans` list holds one entry per signature group, so its positions do not
name the two observables once a model differs across stations.
"""
function bandpass_blocks(setup, θ, group::Symbol)
    path = group === :phase ? setup.bp_path : setup.amp_path
    path === nothing && return NamedTuple[]
    return station_blocks(setup.layout, θ, path...)
end

function solve_bandpass! end
solve_bandpass!(sm::AbstractBandpassSmoother, θ, results, setup; gauge) =
    error(
    "$(typeof(sm)) does not implement the bandpass smoother interface: define " *
        "Gustavo.Fringe.solve_bandpass!(::$(typeof(sm)), θ, results, setup; gauge)."
)

# Fresh per-(baseline row, product, global channel) bandpass accumulators. They
# carry (Baseline, Pol, Frequency) dims so the accumulate/solve kernels below
# address axes by NAME (the house style of the Bandpass module) instead of by
# position; indexing stays plain-positional and costs nothing.
function bandpass_accumulators(nbl::Integer, npol::Integer, nchan::Integer)
    d = (Baseline(1:nbl), Pol(1:npol), Frequency(1:nchan))
    return (
        DimensionalData.DimArray(zeros(ComplexF64, nbl, npol, nchan), d),
        DimensionalData.DimArray(zeros(Float64, nbl, npol, nchan), d),
    )
end

"""
    accumulate_bandpass!(rbar_bp, wbar_bp, blidx, stack, win::GeometryWindow; derotate = true)

Accumulate one scan window's contribution to the per-(global-baseline,
product, global channel) coherent residual `rbar_bp` (and weight `wbar_bp`) for the
bandpass solves, on data already gain-corrected through the pipeline's
transform chain. `blidx` maps `(a, b) -> row` in the global baseline table;
a pair absent from it never contributes.

`derotate` (default `true`) counter-rotates each AP, before summing over time,
by its own band-averaged residual phase — removing the per-AP time phase
(residual rate/drift, and what the adhoc stage would later remove) so summing
COHERENT SCANS TOGETHER ([`PerTrackSmoother`](@ref), which combines every
selected scan's residual into one accumulator before solving) isolates the per-channel shape
despite each scan's uncontrolled source phase. [`solve_joint_bandpass!`](@ref)
fits each scan's own coherent visibility against an explicit per-scan source
term instead of summing scans together, so it passes `derotate = false` — the
per-AP trick would otherwise erase the very source phase/amplitude that term
is meant to absorb.
"""
function accumulate_bandpass!(
        rbar_bp, wbar_bp, blidx, stack::AbstractDimStack, win::GeometryWindow;
        derotate::Bool = true,
    )
    V = stack[:vis]                                      # the dims-carrying layers —
    W = stack[:weights]                                  # the loops below address axes BY NAME
    F = stack[:flags]
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
                    fl = F[Frequency = c, Ti = tt, Baseline = bi, Pol = p]
                    cond = (!fl && w > 0 && isfinite(w) && isfinite(vv))
                    acc += ifelse(cond, w * vv, zero(eltype(V)))
                end
                rot = ifelse(abs(acc) > 0, conj(acc) / abs(acc), one(eltype(V))) # cis(-angle(acc)): de-rotate this AP
            end
            for c in axes(V, Frequency)
                w = W[Frequency = c, Ti = tt, Baseline = bi, Pol = p]
                vv = V[Frequency = c, Ti = tt, Baseline = bi, Pol = p]
                fl = F[Frequency = c, Ti = tt, Baseline = bi, Pol = p]
                cond = (!fl && w > 0 && isfinite(w) && isfinite(vv))
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
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in axes(rbar_bp, 1), p in axes(rbar_bp, 2)]
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
# Shape only and applies zero net phase.
function _write_phase_bandpass!(θ, plan, phase, ts::Integer = 1)
    nant, _, nseg = size(phase)
    leaf = _component_leaf(plan, θ)
    for a in axes(phase, 1), f in axes(phase, 2)
        acc = zero(ComplexF64)
        for fs in axes(phase, 3)
            v = phase[a, f, fs]
            isfinite(v) && (acc += cis(v))
        end
        abs(acc) > 0 || continue
        m = angle(acc)
        for fs in axes(phase, 3)
            v = phase[a, f, fs]
            isfinite(v) || continue
            node = _feed_node(plan.tying, f)
            node == 0 && continue
            leaf[1, node, fs, ts, a] = rem2pi(v - m, RoundNearest)
        end
    end
    return θ
end

# One frequency segment's coherent residual `(r, w, w2)`: the sums of the
# accumulators over the channels it holds, plus `w2 = Σ wᶜ²`, which converts a
# Per-channel noise variance into the variance of this segment's normalized
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
    for a in axes(la, 1), f in axes(la, 2), bnd in sort(unique(seg_spw))
        sidx = [s for s in eachindex(seg_spw) if seg_spw[s] == bnd]
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
# so the bandpass carries shape only and applies unit net amplitude.
function _write_amp_bandpass!(θ, plan, la, max_logamp::Real, ts::Integer = 1)
    nant, _, nfseg = size(la)
    leaf = _component_leaf(plan, θ)
    for a in axes(la, 1), f in axes(la, 2)
        acc = 0.0; n = 0
        for s in axes(la, 3)
            v = la[a, f, s]
            isfinite(v) && (acc += v; n += 1)
        end
        n == 0 && continue
        m = acc / n
        for s in axes(la, 3)
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
            leaf[1, node, s, ts, a] = abs(val) > max_logamp ? 0.0 : val
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
    noise2 = [_track_noise2(rbar_bp, wbar_bp, bi, p, nchan) for bi in axes(rbar_bp, 1), p in axes(rbar_bp, 2)]
    nnodes = 2 * nant
    nfseg = length(fsegs)
    la = fill(NaN, nant, 2, nfseg)
    prec = zeros(nant, 2, nfseg)
    for (fs, chans) in enumerate(fsegs)
        na = Int[]; nbn = Int[]; vals = Float64[]; wts = Float64[]
        for bi in axes(rbar_bp, 1), p in axes(rbar_bp, 2)
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

# Outcome of fitting one (station, feed, spw) bandpass track, reported per track by
# `bandpass_track_report` so a caller can tell a measurement from a placeholder.
# `θ` carries no such distinction: an unfitted track reads back as unit gain and a
# starved one as a constant, both indistinguishable from a real flat response.
const _BP_TRACK_NODATA = Int8(0)      # no usable segment; left at unit gain
const _BP_TRACK_SOLVED = Int8(1)      # fit, with frequency structure
const _BP_TRACK_FLAT = Int8(2)        # fit, but constant to within `_BP_FLAT_SPAN`
const _BP_TRACK_DECLINED = Int8(3)    # phase branch undetermined; not fit
# A (station, time segment) cell the report's rectangular shape has but this
# station's own segmentation does not — the station carries no parameter there.
# Distinct from `_BP_TRACK_NODATA`, which is a cell that exists and went unfit.
const _BP_TRACK_NA = Int8(4)

const _BP_TRACK_LABELS = ("nodata", "solved", "flat", "declined", "na")

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

can_fit(::PerTrackSmoother, tc) = _fits_bandpass_track(tc)

# The outcome code for one fitted spw track: nothing estimated, a constant, or a
# real shape.
function _band_track_status(fitted)
    obs = [v for v in fitted if isfinite(v)]
    isempty(obs) && return _BP_TRACK_NODATA
    return (maximum(obs) - minimum(obs)) < _BP_FLAT_SPAN ? _BP_TRACK_FLAT : _BP_TRACK_SOLVED
end

# Fit one segment-indexed track under `spec`, split at the spw boundaries — a
# shape describes the response within a band, so segments never pool across spws
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
    for a in axes(tracks, 1), f in axes(tracks, 2)
        y = [tracks[a, f, s] for s in axes(tracks, 3)]
        w = [prec[a, f, s] for s in axes(prec, 3)]
        st = status === nothing ? nothing : view(status, a, f, :)
        fitted = _fit_track_bands(spec, y, w, seg_spw, seg_freq; unwrap, status = st)
        for (s, v) in zip(axes(tracks, 3), fitted)
            tracks[a, f, s] = v
        end
    end
    return tracks
end

"""
    bandpass_track_report(phase_status, amp_status, band_ids) -> NamedTuple

Summarize a bandpass solve's per-(station, feed, spw, time segment) outcomes into
the record the [`Bandpass`](@ref Gustavo.Bandpass) step publishes.
`phase_status`/`amp_status` are `(Ant, Feed, band, time segment)` arrays of
`_BP_TRACK_*` codes (either may be `nothing` when that half was not fit);
`band_ids` names the spw each band slot came from. A time-stable bandpass has one
time segment, so its arrays are `(Ant, Feed, band, 1)`.

The time-segment axis spans the union of every station's segments, so where the
model gives stations different time segmentations the arrays are rectangular over
a grid some stations do not fill; a cell a station's own segmentation lacks
carries the `na` code and is counted in `n_na`, apart from the four outcomes a
track that exists can have.

Returns the two arrays as `phase_status`/`amp_status` alongside `band_ids`,
`track_labels` (the code → name mapping, so a reader needs no constant from this
module) and the counts `n_solved`/`n_flat`/`n_declined`/`n_nodata`/`n_na` summed
over both observables. `flat` and `declined` are the two ways a track can occupy a
slot without measuring anything, and they are what the counts exist to expose: θ
itself records an unfitted track as unit gain and a starved one as a constant,
neither distinguishable there from a genuinely flat response.
"""
function bandpass_track_report(phase_status, amp_status, band_ids)
    counts = zeros(Int, length(_BP_TRACK_LABELS))
    for st in (phase_status, amp_status), c in something(st, Int8[])
        counts[Int(c) + 1] += 1
    end
    # Concrete arrays throughout — the record is serialized with the solution, and
    # an observable that was not fit is an EMPTY status rather than a missing field.
    empty_status = Array{Int8, 4}(undef, 0, 0, 0, 0)
    return (;
        phase_status = something(phase_status, empty_status),
        amp_status = something(amp_status, empty_status),
        band_ids = collect(Int, band_ids),
        track_labels = collect(String, _BP_TRACK_LABELS),
        n_nodata = counts[1], n_solved = counts[2],
        n_flat = counts[3], n_declined = counts[4], n_na = counts[5],
    )
end

# Warn when a large share of the tracks measured nothing. Silence here would leave
# a bandpass that is mostly placeholder looking exactly like one that is mostly
# measured — the caller cannot tell from θ, which is why this is a warning and not
# only a record. The fraction is over the tracks that EXIST: a cell a station's
# own segmentation does not have (`n_na`) is not a track that failed to measure
# anything, and counting it would make the warning fire on raggedness alone.
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

"""
    time_segment_scans(plan, results) -> Vector{Vector{Int}}

The `results` indices belonging to each of `plan`'s time segments, in segment
order. A scan lies wholly inside one segment of any segmentation the smoothers
accept, so its first sample (`res.ti`) names the segment. Segments the pass
never visited come back empty.
"""
function time_segment_scans(plan, results)
    nts = isempty(plan.tseg_id) ? 1 : maximum(plan.tseg_id)
    groups = [Int[] for _ in 1:nts]
    for (i, res) in pairs(results)
        push!(groups[plan.tseg_id[res.ti]], i)
    end
    return groups
end

"""
    _station_time_segments(blocks, results, nant) -> Matrix{Int}

Each station's own time segment for each scan: `tseg[ant, i]` is the segment
`results[i]` falls in under the segmentation `blocks` gives that station, and `0`
for a station no block covers — such a station carries no bandpass parameter and
stays at unit gain. A scan lies wholly inside one segment of any segmentation the
smoothers accept, so its first sample (`res.ti`) names the segment.

A station-uniform model has one block spanning every station, so every row is the
same and the table reduces to [`time_segment_scans`](@ref)' single grouping.
"""
function _station_time_segments(blocks, results, nant)
    tseg = zeros(Int, Base.OneTo(nant), axes(results, 1))
    for b in blocks, a in b.stations
        for (i, res) in pairs(results)
            tseg[a, i] = b.plan.tseg_id[res.ti]
        end
    end
    return tseg
end

"""
    _station_freq_segments(blocks, nant) -> (fseg, cells)

Each station's own frequency segment for each cell of the solve's refinement
grid. `cells` is the channel groups of the COMMON REFINEMENT of the blocks'
frequency segmentations ([`Calibration.common_refinement`](@ref)) — the coarsest
channel partition every block's own segmentation is a union of cells of — and
`fseg[a, k]` is the segment station `a` carries over cell `k`, `0` for a station
no block covers.

Stations that share one frequency segmentation give the identity refinement, so
`cells` is that segmentation's own channel groups and every station's cell maps
to itself.
"""
function _station_freq_segments(blocks, nant)
    cell_ids, ncell = common_refinement([b.plan.fseg_id for b in blocks])
    cells = segment_groups(cell_ids, ncell)
    fseg = zeros(Int, Base.OneTo(nant), Base.OneTo(ncell))
    for b in blocks, a in b.stations
        for (k, chans) in pairs(cells)
            # A cell lies wholly inside one segment of every block's
            # segmentation, so its first channel names them all.
            fseg[a, k] = b.plan.fseg_id[first(chans)]
        end
    end
    return fseg, cells
end

# The two per-station tables `_fit_track_bands` fits a track against: the spw
# each of that station's frequency segments belongs to, and the segment's
# frequency coordinate. Both are the station's block's own, so stations sharing a
# block share the vectors. `seg_spw`/`seg_freq` hold one entry per block, in
# `blocks` order; `nothing` falls back to a single band indexed by segment, which
# is what a caller that named no channel table gets.
function _station_band_tables(blocks, block_of, seg_spw, seg_freq, nant)
    nfs(a) = iszero(block_of[a]) ? 0 : blocks[block_of[a]].plan.shape[3]
    bands = [
        (seg_spw === nothing || iszero(block_of[a])) ? ones(Int, nfs(a)) : seg_spw[block_of[a]]
            for a in eachindex(block_of)
    ]
    coords = [
        (seg_freq === nothing || iszero(block_of[a])) ? collect(1.0:nfs(a)) : seg_freq[block_of[a]]
            for a in eachindex(block_of)
    ]
    return bands, coords
end

# The `results` indices of each independently solvable scan group, given a
# per-station time-segment table.
#
# Two scans share a parameter only through a station that is in the same one of
# ITS OWN time segments in both, so the coupling graph has one node per
# (station, segment) and one edge per pair of stations a scan holds together; the
# ALS then runs once per connected component. With one segmentation for the whole
# array the components are exactly the array-wide time segments. With one station
# broken mid-track and the rest constant the constant stations bridge the break
# and the whole track is one component.
function _joint_scan_groups(tseg)
    # `connected_components` numbers its nodes densely from 1, which is what
    # `LinearIndices` hands back for a (station, segment) pair.
    ntseg = maximum(tseg; init = zero(eltype(tseg)))
    nodes = LinearIndices((axes(tseg, 1), Base.OneTo(ntseg)))
    edges = Tuple{Int, Int}[]
    anchor = zeros(eltype(nodes), axes(tseg, 2))
    for si in axes(tseg, 2)
        prev = zero(eltype(nodes))
        for a in axes(tseg, 1)
            ts = tseg[a, si]
            iszero(ts) && continue
            n = nodes[a, ts]
            iszero(prev) ? (anchor[si] = n) : push!(edges, (prev, n))
            prev = n
        end
        # A scan holding a single station still has to name a component: the
        # self-edge marks the node visited without joining it to anything.
        iszero(anchor[si]) || push!(edges, (anchor[si], anchor[si]))
    end
    compid, ncomp, _ = connected_components(length(nodes), edges)
    groups = [Int[] for _ in 1:ncomp]
    for si in axes(tseg, 2)
        iszero(anchor[si]) || push!(groups[compid[anchor[si]]], si)
    end
    return filter!(!isempty, groups)
end

# Sum the per-scan residual accumulators of `idx` into one pooled pair.
function _pool_scans(results, idx, nbl, npol, nchan)
    rbar, wbar = bandpass_accumulators(nbl, npol, nchan)
    for i in idx
        rbar .+= results[i].rl
        wbar .+= results[i].wl
    end
    return rbar, wbar
end

function solve_bandpass!(sm::PerTrackSmoother, θ, results, setup; gauge::AbstractGauge)
    pols = results[1].pols
    nchan = length(setup.channel_freqs)
    nbl = length(setup.bl_pairs)
    phase_status = nothing
    amp_status = nothing
    band_ids = Int[]
    # The two observables are solved independently here, so each partitions the
    # scans by its OWN time segmentation — a phase bandpass that breaks mid-track
    # can sit beside an amplitude one held over the whole of it.
    bp_blocks = bandpass_blocks(setup, θ, :phase)
    amp_blocks = bandpass_blocks(setup, θ, :logamp)
    if !isempty(bp_blocks)
        plan = only(bp_blocks).plan
        fsegs, seg_spw, seg_freq = _segment_bands(plan, setup.channel_freqs, setup.spw_of_chan)
        band_ids = sort(unique(seg_spw))
        groups = time_segment_scans(plan, results)
        phase_status = fill(_BP_TRACK_NODATA, setup.nant, 2, length(band_ids), length(groups))
        for (ts, idx) in pairs(groups)
            isempty(idx) && continue
            rbar, wbar = _pool_scans(results, idx, nbl, length(pols), nchan)
            phase, prec = _seed_phase_tracks(
                rbar, wbar, setup.bl_pairs, pols, setup.nant, fsegs; gauge,
            )
            _shape_tracks!(
                phase, prec, seg_spw, seg_freq, sm.phase;
                unwrap = true, status = view(phase_status, :, :, :, ts),
            )
            _write_phase_bandpass!(θ, plan, phase, ts)
        end
    end
    if !isempty(amp_blocks)
        plan = only(amp_blocks).plan
        fsegs, seg_spw, seg_freq = _segment_bands(plan, setup.channel_freqs, setup.spw_of_chan)
        band_ids = sort(unique(seg_spw))
        groups = time_segment_scans(plan, results)
        amp_status = fill(_BP_TRACK_NODATA, setup.nant, 2, length(band_ids), length(groups))
        for (ts, idx) in pairs(groups)
            isempty(idx) && continue
            rbar, wbar = _pool_scans(results, idx, nbl, length(pols), nchan)
            la, prec = _seed_amp_tracks(rbar, wbar, setup.bl_pairs, pols, setup.nant, fsegs)
            _spike_guard!(la, seg_spw, _BP_SPIKE_SIGMA)
            _shape_tracks!(
                la, prec, seg_spw, seg_freq, sm.amp;
                unwrap = false, status = view(amp_status, :, :, :, ts),
            )
            _write_amp_bandpass!(θ, plan, la, _BP_MAX_LOGAMP, ts)
        end
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
# per-(station, feed) solve of `g` (given `S` and every other station's
# current gain, [`_update_station_gains!`](@ref)) at `phase_plan`/`amp_plan`'s
# shared frequency-segment resolution, to convergence.
#
# Every array below carries (Scan, Baseline, Pol, Frequency) or (Ant, Feed, Ti,
# Frequency) dims — the house style of this module — so the loops read by
# axis NAME; `Frequency` here is the frequency-segment index and `Ti` the
# time-segment one (as elsewhere once a solve moves past the raw per-channel
# accumulator). Because each station carries its own time segmentation, `Ti`
# spans the union of them and a station reaches only its own slots. Indexing
# itself stays plain-positional.

# One scan's per-(baseline, pol, segment) coherent residual, written directly
# into `rview`/`wview` (a (Baseline, Pol, Frequency) slice of the multi-scan
# accumulator — no intermediate allocation).
#
# Not SNR-gated, deliberately. The joint solve consumes these as complex
# residuals under inverse-variance weights, and that accumulation is unbiased
# at any SNR — a weak cell contributes its information at its honest weight
# and costs variance, never validity. An SNR gate here (the closure tier's,
# which is justified THERE because that tier extracts a per-segment phase, a
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

# Every scan's (Baseline, Pol, segment) residual, stacked over an added Scan
# axis — not summed across scans (unlike the closure tier's fold), since the per-scan source coherence needs each
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

# The phase-gauge graph of a joint bandpass solve: one node per (station, feed,
# that station's own time segment, that station's own frequency segment), one
# edge per (scan, baseline, pol, refinement cell) correlation, joining its two
# ends in the segments they are in for that scan and that cell. Node degree
# stands in for the row weight the gauge scores elsewhere — the graph is built
# from the correlations that EXIST, before any per-channel gating, so a node's
# degree is the observation count available to anchor it.
#
# The node set IS the gauge condition. An edge names the two gain slots one
# correlation reads, and a shared phase cancels out of `g_a·S·conj(g_b)` only
# when both ends of every internal edge carry it, so the unobservable phases are
# one constant per connected component — no coarser node set states that, and a
# station whose frequency segments are finer than the modes the array leaves free
# would be over-constrained by any pin that fixed its whole track.
#
# `nodes` is the dense `(station, feed, time segment, frequency segment)`
# numbering `connected_components` works in; `compid[n]` is `0` for a node no
# correlation touches.
function _joint_bandpass_graph(bl_pairs, feeds, nant, tseg, fseg, nfsmax)
    ntseg = maximum(tseg; init = zero(eltype(tseg)))
    nodes = LinearIndices(
        (Base.OneTo(nant), Base.OneTo(2), Base.OneTo(ntseg), Base.OneTo(nfsmax)),
    )
    edges = Tuple{Int, Int}[]
    deg = zeros(Int, length(nodes))
    for si in axes(tseg, 2), p in eachindex(feeds), bi in eachindex(bl_pairs)
        a, b = bl_pairs[bi]
        a == b && continue
        # A station no block covers (segment 0) carries no bandpass parameter, so
        # the correlations touching it constrain nothing.
        ta, tb = tseg[a, si], tseg[b, si]
        (iszero(ta) || iszero(tb)) && continue
        fa, fb = feeds[p]
        for cell in axes(fseg, 2)
            na = nodes[a, fa, ta, fseg[a, cell]]
            nb = nodes[b, fb, tb, fseg[b, cell]]
            push!(edges, (na, nb))
            deg[na] += 1
            deg[nb] += 1
        end
    end
    compid, ncomp, _ = connected_components(length(nodes), edges)
    return nodes, compid, ncomp, deg
end

# One reference node per connected component of that graph — mirrors
# `_solve_observable`'s pin selection in stationize.jl. The pinned node's phase is
# held at zero in the one (time segment, frequency segment) slot it names; its
# amplitude is solved like any other node's.
#
# Only the phase is a gauge freedom. Multiplying the gains of a set of nodes by a
# shared `c` sends `g_a·S·conj(g_b)` to `|c|²·g_a·S·conj(g_b)` on every
# correlation internal to that set: the phase of `c` cancels between the two
# conjugated factors. The sets on which it cancels everywhere are exactly the
# components above, so the phase carries one unobservable constant per component
# and needs one constraint there — no more. A station-uniform segmentation splits
# the graph along the array-wide (time segment, frequency segment) cells and
# recovers one pin per cell, which together zero the reference station's whole
# track; a model in which one station breaks mid-track keeps the whole track in
# one component, because the stations that hold one gain over it bridge the
# broken station's two segments, and the relative phase across that break is then
# measured rather than gauged away.
#
# The same bridging on the frequency axis makes a pin PARTIAL: where a station
# holds one gain across cells the others split, those cells lie in one component,
# and its single pin fixes ONE segment of the pinned station's track while the
# rest of that track is fitted.
#
# The magnitude does not cancel, and `S` is frequency-flat, so it can only absorb
# `|c|²` when `|c|` is constant across the band — leaving exactly one free
# amplitude parameter overall, which the zero-band-mean gauge in
# `_write_joint_bandpass!` removes. Pinning `|g|` as well
# would assert the reference antenna has a flat amplitude bandpass, discarding
# structure that is identifiable (mean-removing `log|V_ab| = la_a + la_b + ls_ab`
# over the band eliminates `ls` and leaves the full-rank signless-Laplacian
# system) and biasing every other station through the inconsistency.
function _joint_bandpass_pins(bl_pairs, feeds, nant, tseg, fseg, nfsmax, gauge)
    nodes, compid, ncomp, deg =
        _joint_bandpass_graph(bl_pairs, feeds, nant, tseg, fseg, nfsmax)
    ci = CartesianIndices(nodes)
    station_of(n) = ci[n][1]
    feed_of(n) = ci[n][2]
    pins = Set{Int}()
    for c in 1:ncomp
        comp_nodes = findall(==(c), compid)
        push!(pins, gauge_anchor(gauge, comp_nodes, deg, station_of, feed_of))
    end
    return nodes, pins
end

# Closed-form per-(scan, baseline, pol) solve of the source coherence `S`
# given the current station gains `g`: the weighted-least-squares minimizer of
# `Σ_cell wseg·|rseg/wseg − g_a·S·conj(g_b)|²` over the single complex
# unknown `S`.
#
# `rseg`/`wseg` are reduced onto the refinement grid the two stations have in
# common, while `g` is held in each station's OWN frequency segments, so each end
# is read through its own `fseg` row.
function _update_source_coherence!(S, g, rseg, wseg, bl_pairs, feeds, tseg, fseg)
    T = real(eltype(S))
    for p in axes(rseg, Pol), bi in axes(rseg, Baseline)
        a, b = bl_pairs[bi]
        a == b && continue
        fa, fb = feeds[p]
        for si in axes(rseg, Scan)
            # Each station is in its own time segment for this scan; a station no
            # block covers (segment 0) has no gain to fit, so the baselines that
            # touch it carry no source term either.
            ta, tb = tseg[a, si], tseg[b, si]
            (iszero(ta) || iszero(tb)) && continue
            numer = zero(eltype(S))
            denom = zero(T)
            for cell in axes(rseg, Frequency)
                w = wseg[si, bi, p, cell]
                w > 0 || continue
                u = g[a, fa, ta, fseg[a, cell]] * conj(g[b, fb, tb, fseg[b, cell]])
                abs2(u) > 0 || continue
                numer += conj(u) * rseg[si, bi, p, cell]
                denom += w * abs2(u)
            end
            S[si, bi, p] = denom > 0 ? numer / denom : zero(eltype(S))
        end
    end
    return nothing
end

# One Gauss-Seidel sweep over every (station, feed, time segment): closed-form
# per-frequency-segment solve of its complex gain given the current source
# coherence `S` and every other station's current gain (immediately visible to later antennas
# in the same sweep — Gauss-Seidel, not Jacobi), with each observable's shape
# spec acting as a PRIOR on the resulting track rather than a post-hoc smooth.
#
# Solving each segment independently would be the `FreeShape`/`FreeShape` case;
# the prior enters exactly where that independence is dropped. Around the
# unconstrained per-segment estimate `ĝ` the residual linearizes as
# `Σ denom·|ĝ|²·(δlogamp² + δphase²)`, so `denom·|ĝ|²` is the Fisher weight both
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
        g, φ, touched, S, rseg, wseg, touching, bl_pairs, feeds, pinned, tseg, fseg, present,
        phase_spec, amp_spec, bands, coords; seed::Bool,
        phase_status = nothing, amp_status = nothing,
    )
    T = real(eltype(g))
    C = eltype(g)
    maxrel = zero(T)
    nfsmax = size(g, Frequency)
    num = Vector{C}(undef, nfsmax)
    den = Vector{T}(undef, nfsmax)
    ĝ = Vector{C}(undef, nfsmax)
    wf = Vector{T}(undef, nfsmax)
    la = Vector{T}(undef, nfsmax)
    φ̃ = Vector{T}(undef, nfsmax)
    for feed in axes(g, Feed), ant in axes(g, Ant), ts in present[ant]
        entries = touching[ant, feed]
        isempty(entries) && continue
        seg_spw, seg_freq = bands[ant], coords[ant]
        nfs = length(seg_spw)
        # The gauge fixes this node's phase in the frequency segments it pins —
        # every one of them where the array leaves this track no free structure,
        # a single one where a coarser station ties the band into one mode. The
        # amplitude is solved like any other node's (see `_joint_bandpass_pins`).
        pins = view(pinned, ant, feed, ts, 1:nfs)
        for buf in (num, den, ĝ, wf, la, φ̃)
            resize!(buf, nfs)
        end
        fill!(num, zero(C))
        fill!(den, zero(T))
        # The data live on the refinement grid every station shares; this station's
        # gain is constant over its OWN segment, so a segment's estimate pools the
        # numerator and denominator of every cell inside it.
        for cell in axes(rseg, Frequency)
            sa = fseg[ant, cell]
            for (bi, p, role) in entries
                a, b = bl_pairs[bi]
                fa, fb = feeds[p]
                for si in axes(rseg, Scan)
                    # Only the scans this node's OWN segment covers constrain it.
                    tseg[ant, si] == ts || continue
                    w = wseg[si, bi, p, cell]
                    w > 0 || continue
                    s = S[si, bi, p]
                    if role === :a
                        tb = tseg[b, si]
                        iszero(tb) && continue
                        coeff = s * conj(g[b, fb, tb, fseg[b, cell]])
                        abs2(coeff) > 0 || continue
                        num[sa] += conj(coeff) * rseg[si, bi, p, cell]
                        den[sa] += w * abs2(coeff)
                    else
                        ta = tseg[a, si]
                        iszero(ta) && continue
                        coeff = conj(g[a, fa, ta, fseg[a, cell]] * s)
                        abs2(coeff) > 0 || continue
                        num[sa] += conj(coeff) * conj(rseg[si, bi, p, cell])
                        den[sa] += w * abs2(coeff)
                    end
                end
            end
        end
        for fs in 1:nfs
            gh = den[fs] > 0 ? num[fs] / den[fs] : zero(C)
            ĝ[fs] = gh
            wf[fs] = (den[fs] > 0 && abs(gh) > 0) ? den[fs] * abs2(gh) : zero(T)
            if wf[fs] > 0
                la[fs] = log(abs(gh))
                # The wrapped increment about this track's current value keeps the
                # candidate on the same 2π branch as the iterate it refines.
                φ̃[fs] = φ[ant, feed, ts, fs] +
                    rem2pi(angle(gh) - φ[ant, feed, ts, fs], RoundNearest)
            else
                la[fs] = T(NaN)
                φ̃[fs] = T(NaN)
            end
        end
        # The status of the LAST sweep is the status of the solve: each sweep
        # overwrites the previous one's codes for this node.
        ast = amp_status === nothing ? nothing : view(amp_status, ant, feed, :, ts)
        pst = phase_status === nothing ? nothing : view(phase_status, ant, feed, :, ts)
        la_new = _fit_track_bands(amp_spec, la, wf, seg_spw, seg_freq; status = ast)
        φ_new = if all(pins)
            # A wholly pinned track is known rather than fitted: report it as such
            # instead of leaving it at NODATA.
            pst === nothing || fill!(pst, _BP_TRACK_SOLVED)
            zeros(T, nfs)
        else
            fitted = _fit_track_bands(
                phase_spec, φ̃, wf, seg_spw, seg_freq; unwrap = seed, status = pst,
            )
            # A partial pin holds its own segments and leaves the rest fitted.
            # With segments fit independently that is the constrained fit itself;
            # `solve_joint_bandpass!` rejects a `phase_spec` that pools them
            # rather than pass off a joint fit with one segment overwritten.
            for fs in eachindex(fitted)
                pins[fs] && (fitted[fs] = zero(T))
            end
            fitted
        end
        for fs in 1:nfs
            (isfinite(la_new[fs]) && isfinite(φ_new[fs])) || continue
            gold = g[ant, feed, ts, fs]
            gnew = exp(C(la_new[fs], φ_new[fs]))
            g[ant, feed, ts, fs] = gnew
            φ[ant, feed, ts, fs] = φ_new[fs]
            touched[ant, feed, ts, fs] = true
            maxrel = max(maxrel, abs(gnew - gold) / max(abs(gold), abs(gnew), eps(T)))
        end
    end
    return maxrel
end

# Gauge-fix each (station, feed, time segment) track — zero band-mean
# log-amplitude, circular-mean reference phase, matching the closure tier's
# convention — and write it into the station block that carries it. Both gauges
# are over the FREQUENCY track of one time segment, so each of a station's
# segments is normalized on its own. A (station, feed, segment) `touched` nowhere
# (no data ever reached it) is left unwritten (still whatever θ already held,
# i.e. unit gain), as is a station no block covers.
#
# A block's `:Ant` axis spans its own stations, so the leaf is indexed by the
# station's position within `block.stations`, and `ts` is already in that block's
# own segment numbering (`_station_time_segments` reads each station's segment
# from its own block's `tseg_id`).
function _write_joint_bandpass!(θ, phase_blocks, amp_blocks, g, touched, max_logamp, present)
    for block in phase_blocks
        for (ai, a) in pairs(block.stations), f in axes(g, Feed)
            node = _feed_node(block.plan.tying, f)
            node == 0 && continue
            for ts in present[a]
                any(view(touched, a, f, ts, :)) || continue
                mphase = angle(sum(cis(angle(g[a, f, ts, fs])) for fs in axes(g, Frequency) if touched[a, f, ts, fs]))
                for fs in axes(g, Frequency)
                    touched[a, f, ts, fs] || continue
                    block.θ[1, node, fs, ts, ai] =
                        rem2pi(angle(g[a, f, ts, fs]) - mphase, RoundNearest)
                end
            end
        end
    end
    for block in amp_blocks
        for (ai, a) in pairs(block.stations), f in axes(g, Feed)
            node = _feed_node(block.plan.tying, f)
            node == 0 && continue
            for ts in present[a]
                valid = @view touched[a, f, ts, :]
                any(valid) || continue
                logs = log.(abs.(@view g[a, f, ts, :]))
                m = sum(view(logs, valid)) / count(valid)
                for fs in axes(g, Frequency)
                    valid[fs] || continue
                    la = logs[fs] - m
                    block.θ[1, node, fs, ts, ai] = abs(la) > max_logamp ? 0.0 : la
                end
            end
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
(station, feed, segment), so it requires a model with both a phase and a
log-amplitude component, sharing one frequency segmentation per station.
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

# The joint tier fits each scan's own coherent visibility against an explicit
# source term, so the per-AP derotation that lets scans be summed together would
# erase the very source phase that term absorbs.
bandpass_derotate(::JointSmoother) = false

can_fit(::JointSmoother, tc) = _fits_bandpass_track(tc)

function validate_model(::JointSmoother, model)
    validate_bandpass_groups(model)
    ph = Calibration._flatten_components(model.phase)
    la = Calibration._flatten_components(model.logamp)
    length(ph) == 1 && length(la) == 1 || throw(
        ArgumentError(
            "JointSmoother requires both a phase and a logamp component — it solves one " *
                "COMPLEX gain per (station, feed, segment), not independent phase/log-amp " *
                "tracks. Use `smoother = PerTrackSmoother()` for a phase-only or " *
                "amplitude-only bandpass.",
        ),
    )
    only(ph).Frequency == only(la).Frequency || throw(
        ArgumentError(
            "JointSmoother requires the phase and logamp components to share one frequency " *
                "segmentation — it solves one COMPLEX gain per (station, feed, segment). Got " *
                "$(repr(only(ph).Frequency)) (phase) vs $(repr(only(la).Frequency)) (logamp).",
        ),
    )
    only(ph).Ti == only(la).Ti || throw(
        ArgumentError(
            "JointSmoother requires the phase and logamp components to share one time " *
                "segmentation — one complex gain per (station, feed, segment) is solved over " *
                "one stretch of time, not two. Got $(repr(only(ph).Ti)) (phase) vs " *
                "$(repr(only(la).Ti)) (logamp). Use `smoother = PerTrackSmoother()` to give " *
                "the two observables different time resolutions.",
        ),
    )
    return nothing
end

function solve_bandpass!(sm::JointSmoother, θ, results, setup; gauge::AbstractGauge)
    phase_blocks = bandpass_blocks(setup, θ, :phase)
    amp_blocks = bandpass_blocks(setup, θ, :logamp)
    # One band table per block, since the blocks need not share a frequency
    # segmentation. Every segmentation refines the spw partition, so the bands
    # they name are the same set however finely each block cuts them.
    tables = [_segment_bands(b.plan, setup.channel_freqs, setup.spw_of_chan) for b in phase_blocks]
    seg_spw = [t[2] for t in tables]
    seg_freq = [t[3] for t in tables]
    band_ids = sort(unique(Iterators.flatten(seg_spw)))
    # One complex gain per (station, feed, segment) means one time segmentation
    # for both observables — `validate_model` holds each station's two plans to
    # it — so the phase side's table is the whole solve's.
    tseg = _station_time_segments(phase_blocks, results, setup.nant)
    phase_status = _joint_status_array(phase_blocks, setup.nant, length(band_ids))
    amp_status = _joint_status_array(amp_blocks, setup.nant, length(band_ids))
    for idx in _joint_scan_groups(tseg)
        solve_joint_bandpass!(
            θ, results[idx], setup.bl_pairs, results[1].pols, setup.nant,
            phase_blocks, amp_blocks;
            gauge, max_iterations = sm.max_iterations, tolerance = sm.tolerance,
            phase_spec = sm.phase, amp_spec = sm.amp, seg_spw, seg_freq,
            tseg = view(tseg, :, idx), phase_status, amp_status,
        )
    end
    report = bandpass_track_report(phase_status, amp_status, band_ids)
    _warn_degenerate_bandpass(report)
    return report
end

# One observable's `(Ant, Feed, band, time segment)` status array, rectangular
# over the union of the blocks' time segmentations. A cell outside a station's
# own segmentation — including every cell of a station no block covers — is
# `_BP_TRACK_NA`: it holds no parameter, so no solve will ever write it and it is
# not a track that measured nothing. The rest start at `_BP_TRACK_NODATA`, which
# is what they stay if no scan reaches them.
function _joint_status_array(blocks, nant, nband)
    ntseg = maximum((b.plan.shape[4] for b in blocks); init = 0)
    status = fill(_BP_TRACK_NA, nant, 2, nband, ntseg)
    for b in blocks, a in b.stations
        fill!(view(status, a, :, :, 1:b.plan.shape[4]), _BP_TRACK_NODATA)
    end
    return status
end

"""
    solve_joint_bandpass!(θ, scans, bl_pairs, pol_products, nant, phase_blocks, amp_blocks;
                          gauge = PinAntenna(1), max_iterations = 8, tolerance = 1.0e-6,
                          max_logamp = log(10.0))

Jointly solve the per-(station, feed) complex bandpass gain and a per-scan,
per-baseline, per-polarization constant source coherence (see the module
comment above `_reduce_scan_segments!` for the model and the
alternating scheme), then gauge-fix each (station, feed, time segment) track and
write the result into the station blocks of the two observables
(`_write_joint_bandpass!`).

`phase_blocks` and `amp_blocks` are [`bandpass_blocks`](@ref)`(setup, θ, :phase)`
and `(…, :logamp)` — station blocks over the SAME `θ` this call is handed, since
each block's `θ` is a view into it. A station-uniform model gives one block per
observable spanning every station; where the model differs across stations, each
block carries its own stations, feed tying and segment numbering, and a station
no block covers is left at unit gain. Blocks need not share a frequency
segmentation: the accumulators are reduced onto the common refinement of the
blocks' segmentations ([`_station_freq_segments`](@ref)) and each station's gain
is solved on its own segments, one gain over however many refinement cells a
segment spans. A station's two components must still resolve the SAME
segmentation as each other, which `validate_model(::JointSmoother, model)`
enforces per station at model-compile time; the throw here guards direct callers.

`seg_spw`/`seg_freq`, when given, hold one entry per block of `phase_blocks`, in
that order: the spw each of the block's frequency segments belongs to and the
segment's frequency coordinate, as [`_segment_bands`](@ref) returns them. Omitted,
each station is fit as one band indexed by segment.

Scaling a set of stations by one phase leaves every visibility internal to that
set unchanged, so the gauge fixes one constant per such set. The sets are the
connected components of the (station, feed, time segment, frequency segment)
graph the correlations span, and one node of each is pinned
([`_joint_bandpass_pins`](@ref)). Stations sharing one frequency segmentation
split that graph per cell, so the pinned station's whole track is held at zero;
where one station holds a gain across cells the others split, those cells lie in
one component and the pin holds ONE segment of its track while the rest is
fitted. That partial pin is the constrained fit only where the segments are fit
independently, so a `phase_spec` other than [`FreeShape`](@ref) is rejected in
that case rather than solved as a joint fit with a segment overwritten.

`scans` is the per-scan `(rl, wl)` accumulator pairs from
`accumulate_bandpass!``(...; derotate = false)` — not summed across
scans, since the source term needs each scan's own coherent visibility.

`tseg`, when given, is the `(station, scan)` time-segment table
([`_station_time_segments`](@ref)): station `a`'s gain is solved separately for
each distinct `tseg[a, :]` value, in that station's own segment numbering, and a
station whose entry is `0` is left out of the solve. Every station shares one
segment when it is omitted. `scans` must hold the scans `tseg`'s columns
describe, in the same order. The phase gauge acts on the
(station, feed, time segment, frequency segment) graph these scans span, pinning
one node per connected component ([`_joint_bandpass_pins`](@ref)), so a break at
one station is measured
against the stations that hold one gain across it rather than gauged away.
Solved phases are comparable only WITHIN a component: where that graph splits —
disjoint sub-arrays, or every station segmented at the same epoch — each piece
carries its own arbitrary constant, and a per-station change read across the
split is that constant plus the change.

Convergence is judged on the largest relative per-iteration gain change over
every (station, feed, segment) node solved here, not a tracked χ² (which would
need a per-channel power accumulator this stage does not keep). The scans handed
to one call are a connected piece of the coupling graph, so this is one
criterion over one coupled problem: nodes that share no data are solved by
separate calls rather than being averaged into a common tolerance.

`phase_status`/`amp_status`, when given, are `(Ant, Feed, band, time segment)`
arrays that receive each track's `_BP_TRACK_*` outcome code from the final sweep;
they are indexed by the station's own segment number, so a call solving part of a
track writes only its own slots, and a slot outside a station's own segmentation
is never written (the caller marks those `_BP_TRACK_NA`).
"""
function solve_joint_bandpass!(
        θ, scans, bl_pairs, pol_products, nant, phase_blocks, amp_blocks;
        gauge::AbstractGauge = PinAntenna(1), max_iterations::Integer = 8, tolerance::Real = 1.0e-6,
        max_logamp::Real = _BP_MAX_LOGAMP,
        phase_spec::AbstractShapeSpec = FreeShape(),
        amp_spec::AbstractShapeSpec = FreeShape(),
        seg_spw::Union{Nothing, AbstractVector} = nothing,
        seg_freq::Union{Nothing, AbstractVector} = nothing,
        phase_status = nothing,
        amp_status = nothing,
        tseg::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
    )
    isempty(scans) && return θ

    feeds = [correlation_feed_pair(p) for p in pol_products]
    # The data are reduced onto the refinement of every block's frequency
    # segmentation, and each station's gain is held in its OWN segments — one
    # gain over however many refinement cells that segment spans.
    fseg, segs = _station_freq_segments(phase_blocks, nant)
    block_of = zeros(Int, nant)
    for (bi, b) in pairs(phase_blocks), a in b.stations
        block_of[a] = bi
    end
    # Both observables carry ONE complex gain per (station, feed, segment), so a
    # station's two components must resolve the same frequency segmentation
    # (`validate_model(::JointSmoother, model)` holds each station's tree to it;
    # the throw guards direct callers).
    for b in amp_blocks, a in b.stations
        iszero(block_of[a]) && continue
        b.plan.fseg_id == phase_blocks[block_of[a]].plan.fseg_id || throw(
            ArgumentError(
                "solve_joint_bandpass!: station $a's phase and logamp components resolve " *
                    "different frequency segmentations — the solve carries one COMPLEX gain " *
                    "per (station, feed, segment), not independent phase/log-amp tracks.",
            ),
        )
    end
    bands, coords = _station_band_tables(phase_blocks, block_of, seg_spw, seg_freq, nant)
    nfsmax = maximum(length, bands; init = 0)

    # The time segments each station is actually solved for here, in its own
    # segmentation's numbering — the numbering θ is written in, so the arrays
    # below are rectangular over `1:ntseg` and a station simply skips the slots
    # its segmentation does not reach.
    tsg = tseg === nothing ? ones(Int, nant, length(scans)) : tseg
    present = [sort!(filter!(!iszero, unique(view(tsg, a, :)))) for a in axes(tsg, 1)]
    ntseg = maximum(tsg; init = zero(eltype(tsg)))

    rseg, wseg = _reduce_all_scans(scans, segs)
    touching = _joint_bandpass_touching(bl_pairs, feeds, nant)
    nodes, pins = _joint_bandpass_pins(bl_pairs, feeds, nant, tsg, fseg, nfsmax, gauge)
    pinned = [
        nodes[ant, feed, ts, fs] in pins
            for ant in axes(nodes, 1), feed in axes(nodes, 2), ts in axes(nodes, 3), fs in axes(nodes, 4)
    ]

    # A pin covering part of a track leaves the rest of it fitted, and zeroing
    # part of a track a spec fits jointly is not the constrained fit that spec
    # asks for. Segments fit on their own admit the constraint exactly.
    if !(phase_spec isa FreeShape)
        for ant in axes(pinned, 1), feed in axes(pinned, 2), ts in axes(pinned, 3)
            nfs = length(bands[ant])
            np = count(view(pinned, ant, feed, ts, 1:nfs))
            (iszero(np) || np == nfs) && continue
            throw(
                ArgumentError(
                    "solve_joint_bandpass!: the phase gauge pins $np of the $nfs frequency " *
                        "segments of station $ant (feed $feed, time segment $ts), because " *
                        "the stations' frequency segmentations tie those channels into " *
                        "fewer independent common-phase modes than this station has " *
                        "segments. $(nameof(typeof(phase_spec))) fits a band's segments " *
                        "jointly, so holding part of the fitted track at zero is not the " *
                        "constrained fit it asks for. Use `FreeShape` for the phase, or " *
                        "give the stations frequency segmentations that nest.",
                ),
            )
        end
    end

    C = eltype(rseg)
    # The gain arrays are indexed by each station's OWN frequency segment, which
    # is θ's own axis; a station with fewer segments than the widest one leaves
    # the tail slots untouched (and at unit gain).
    gd = (Ant(1:nant), Feed(1:2), Ti(1:ntseg), Frequency(1:nfsmax))
    g = DimensionalData.DimArray(ones(C, nant, 2, ntseg, nfsmax), gd)
    # The unwrapped phase track behind `g`, carried across sweeps so the shape fit
    # never sees a 2π branch cut.
    φ = DimensionalData.DimArray(zeros(real(C), nant, 2, ntseg, nfsmax), gd)
    # Every node — pinned or not — is marked solved by the gain update, at the
    # segments it actually has data for. A pinned node's phase is known
    # everywhere by the gauge, but its amplitude is not, so it earns its slots
    # the same way the rest do.
    touched = DimensionalData.DimArray(falses(nant, 2, ntseg, nfsmax), gd)
    S = DimensionalData.DimArray(
        zeros(C, length(scans), length(bl_pairs), length(pol_products)),
        (Scan(1:length(scans)), Baseline(1:length(bl_pairs)), Pol(1:length(pol_products))),
    )

    _update_source_coherence!(S, g, rseg, wseg, bl_pairs, feeds, tsg, fseg)
    for iter in 1:max_iterations
        maxrel = _update_station_gains!(
            g, φ, touched, S, rseg, wseg, touching, bl_pairs, feeds, pinned, tsg, fseg, present,
            phase_spec, amp_spec, bands, coords; seed = iter == 1,
            phase_status, amp_status,
        )
        _update_source_coherence!(S, g, rseg, wseg, bl_pairs, feeds, tsg, fseg)
        maxrel < tolerance && break
    end

    return _write_joint_bandpass!(θ, phase_blocks, amp_blocks, g, touched, max_logamp, present)
end

# ── Coverage top-up selection (stations the calibrator never observed) ────────

# Wraps the bandpass step's user selection: stations absent from every selected
# scan would get no bandpass (g = 1), so for each such station the highest-SNR
# scan (any source) containing it is added — mixing sources is safe for the
# bandpass shape (a source's structure phase is flat in frequency per baseline,
# so it biases every channel identically and cancels in the shape; per-scan
# ionosphere differences land in the frozen curve's mean, which each scan's
# dTEC is measured relative to). Applied to any selection, matching the frozen
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
