# ── Hybrid solve: direct linear per-slice phase block steps ──────────────────
#
# A per-CHANNEL phase bandpass and a per-AP adhoc phase are both LINEAR in their
# parameters given the other gains, and both are large and weakly coupled — so
# LBFGS crawls over them (150+ iters on real VLBA data). Solve each DIRECTLY:
# divide out the current gains, coherently average the residual over the OTHER
# axis, and stationize the residual phase per slice with the globally-closing WLS
# `solve_adhoc_phasing`. The bandpass and the adhoc are the SAME operation on
# orthogonal axes:
#
#   * `axis = :freq` — bandpass: average over TIME, one phase per (station, feed)
#     per CHANNEL.  (`solve_adhoc_phasing` with the channel axis as its "AP" axis.)
#   * `axis = :time` — adhoc: average over FREQUENCY, one phase per (station, feed)
#     per AP.  (`solve_adhoc_phasing`'s native use.)
#
# A block-coordinate driver alternates LBFGS (the nonlinear delay/rate, both blocks
# frozen) with these linear steps.

using ..Fringe: solve_adhoc_phasing, NoSmoothing, AbstractAdhocSmoother
using ..UVData: materialize_leaf, branches, baselines, pol_products
using ..Calibration: correlation_feed_pair

# Accumulate one leaf's residual phasor per (baseline, product, SLICE), coherently
# over the averaged axis, after dividing out the current gains `g`. `axis = :freq`
# slices on the GLOBAL channel (`gci`), averaging over time; `axis = :time` slices
# on the GLOBAL AP (`gti`), averaging over frequency. The inverse-variance weight
# `w·|den|²` matches the adhoc's `_accumulate_leaf_rbar!`.
function _accumulate_leaf_phasor!(z, wsum, V, W, g, bl_a, bl_b, feed_a, feed_b, gci, gti, freqaxis::Bool)
    nchan, nti, nbl, npol = size(V)
    @inbounds for p in 1:npol
        fa = feed_a[p]
        fb = feed_b[p]
        for bi in 1:nbl
            a = bl_a[bi]
            b = bl_b[bi]
            a == b && continue
            for t in 1:nti, c in 1:nchan
                ww = W[c, t, bi, p]
                (ww > 0 && isfinite(ww)) || continue
                ga = g[c, t, a, fa]
                gb = g[c, t, b, fb]
                den = ga * conj(gb)
                (abs(ga) > 1.0e-12 && abs(gb) > 1.0e-12 && isfinite(den)) || continue
                v = V[c, t, bi, p] / den
                isfinite(v) || continue
                wd = ww * abs2(den)
                s = freqaxis ? gci[c] : gti[t]
                z[bi, p, s] += wd * v
                wsum[bi, p, s] += wd
            end
        end
    end
    return z, wsum
end

# ADD a per-(antenna, feed, slice) phase increment into the named component's
# block(s). Feeds sharing a feed block are AVERAGED (so a SharedFeeds component is
# written once, not double-added; a PerFeed one gets each feed in its own block).
# `axis = :freq` writes `A[c_local, 1, fseg, fb, la]` (PerChannel × GlobalTime);
# `axis = :time` writes `A[1, tseg, 1, fb, la]` (scalar × PerIntegration × GlobalFreq).
function _add_phase_component!(p, plan::GainPlan, phase::Array{Float64, 3}, name::Symbol, freqaxis::Bool)
    nslice = freqaxis ? plan.nchan : plan.ntime
    for gi in eachindex(plan.groups)
        gp = plan.groups[gi]
        ci = findfirst(v -> _sym(v) === name, gp.phase_syms)
        ci === nothing && continue
        comp = gp.phase[ci]
        parr, _ = group_arrays(plan, p, gi)
        A = parr[ci]
        @inbounds for (la, ant) in enumerate(gp.ants), b in 1:comp.nfb
            for s in 1:nslice
                acc = 0.0
                n = 0
                for feed in 1:2
                    comp.fb1[feed] == b || continue
                    v = phase[ant, feed, s]
                    isfinite(v) || continue
                    acc += v
                    n += 1
                end
                n == 0 && continue
                val = acc / n
                if freqaxis
                    A[comp.clocal[s], 1, comp.fseg_id[s], b, la] += val
                else
                    A[1, comp.tseg_id[s], 1, b, la] += val
                end
            end
        end
    end
    return p
end

"""
    refine_phase_component!(p, plan, uvset, geom; component, axis = :freq,
        ref_ant = 1, shared_feeds = false,
        smoother = NoSmoothing(detrend = false, phase_rewrap_iters = 0)) -> p

Solve a per-slice phase component DIRECTLY (in place, one linear step): divide out
`p`'s current gains, coherently average the residual over the axis ORTHOGONAL to
`axis`, and stationize the residual phase per slice with the globally-closing
[`solve_adhoc_phasing`](@ref), adding the result into `component`.

- `axis = :freq` — the per-channel phase BANDPASS (average over time).
- `axis = :time` — the per-AP ADHOC phase (average over frequency).

`shared_feeds = true` solves one feed-common phase per slice (for a `SharedFeeds`
component, e.g. the adhoc). Far faster and better-conditioned than LBFGS over the
~10³–10⁴ weakly-coupled params of these blocks.
"""
function refine_phase_component!(
        p, plan::GainPlan, uvset, geom;
        component::Symbol, axis::Symbol = :freq,
        ref_ant::Integer = 1, shared_feeds::Bool = false,
        smoother::AbstractAdhocSmoother = NoSmoothing(detrend = false, phase_rewrap_iters = 0),
    )
    axis in (:freq, :time) || error("refine_phase_component!: axis must be :freq or :time (got :$axis)")
    freqaxis = axis === :freq
    nant = plan.nant
    nslice = freqaxis ? plan.nchan : plan.ntime

    l0 = materialize_leaf(first(values(branches(uvset))); layers = (:vis, :weights, :uvw))
    bl = baselines(l0).pairs
    bl_a = Int[Int(q[1]) for q in bl]
    bl_b = Int[Int(q[2]) for q in bl]
    pols = String.(pol_products(l0))
    feeds = correlation_feed_pair.(pols)
    feed_a = Int[f[1] for f in feeds]
    feed_b = Int[f[2] for f in feeds]
    nbl = length(bl)
    npol = length(pols)

    z = zeros(ComplexF64, nbl, npol, nslice)
    wsum = zeros(Float64, nbl, npol, nslice)
    for (_, leaf) in branches(uvset)
        leaf = materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
        ci, ti = leaf_window(geom, leaf)
        g = evaluate_gains(plan, p, ci, ti)
        _accumulate_leaf_phasor!(
            z, wsum, parent(leaf[:vis]), parent(leaf[:weights]), g,
            bl_a, bl_b, feed_a, feed_b, ci, ti, freqaxis,
        )
    end

    # The slice index stands in for `solve_adhoc_phasing`'s AP axis: unit "times"
    # for channels (freq), physical epochs (s) for APs (time).
    times = freqaxis ? collect(1.0:nslice) : (geom.times .* 3600.0)
    as = solve_adhoc_phasing(
        z, wsum, [(bl_a[i], bl_b[i]) for i in 1:nbl], pols, nant, times;
        ref_ant = ref_ant, smoother = smoother, shared_feeds = shared_feeds,
    )
    _add_phase_component!(p, plan, as.phase, component, freqaxis)
    return p
end

"""
    refine_phase_bandpass!(p, plan, uvset, geom; bandpass = :bandpass, kw...) -> p

The per-channel phase-bandpass case of [`refine_phase_component!`](@ref) (`axis =
:freq`).
"""
refine_phase_bandpass!(p, plan::GainPlan, uvset, geom; bandpass::Symbol = :bandpass, kw...) =
    refine_phase_component!(p, plan, uvset, geom; component = bandpass, axis = :freq, kw...)
