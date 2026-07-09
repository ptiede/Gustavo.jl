# ── FFT + stationization warm-start (Stage 2, Milestone 5c) ───────────────────
#
# The forward-model MAP objective is non-convex in the delay/rate directions:
# `phase = 2π·τ·(ν − f0)` wraps once the delay exceeds `1/BW`, so a gradient
# optimizer from a zero start locks onto whichever sidelobe of the matched filter
# is nearest the origin. The multimodality is aliasing/sidelobe, NOT a 2π phase
# ambiguity — the Cartesian forward map has no wraps — so it is resolved exactly
# by the tool built for it: the KEPT FFT matched-filter search + closure
# stationization (`Gustavo.Fringe`). This module turns that per-scan, per-station
# `(delay, rate, phase)` solution into a seed for the per-site `ComponentVector`
# the optimizer polishes.
#
# The mapping is DIRECT: the term coordinates were chosen so a term's parameter IS
# the physical observable the search reports — `Delay` θ = delay (s), `Rate` θ =
# rate (Hz), `ConstantTerm` θ = phase (rad), all referenced to the SAME `(f0, t0)`
# the search uses (`geom.f0`, `geom.t0`). So seeding is a structural walk over the
# plan's components BY TERM TYPE, writing the stationized value into the cell for
# the scan's time segment (and every frequency segment — a wideband search gives
# one delay per scan; a band-resolved model gets the same seed in each band and
# the optimizer refines). Amplitude and per-integration (adhoc) phase are left at
# 0 — the profiled source handles the amplitude scale, and the adhoc residual is
# not a search observable.

import ..Fringe
using ..Fringe: FringeSearch, Stationization, StationSolution, stationize_scan
using ..Calibration: AbstractGainTerm, Delay, Rate, ConstantTerm

# The per-(station, feed) observable a term is seeded from, or `nothing` when the
# term carries no search observable (bandpass, dispersion, …, left at 0). Dispatch
# by term type keeps the routing structural — a new seedable term adds one method.
_seed_observable(::Delay, ss::StationSolution) = ss.delay
_seed_observable(::Rate, ss::StationSolution) = ss.rate
_seed_observable(::ConstantTerm, ss::StationSolution) = ss.phase
_seed_observable(::AbstractGainTerm, ::StationSolution) = nothing

"""
    seed_from_stationization!(p0, plan, ss::StationSolution, g_ti) -> p0

Write one scan's per-(station, feed) [`StationSolution`](@ref) into the warm-start
parameter `ComponentVector` `p0` (in place). `g_ti` are the scan's GLOBAL time
indices (into `plan`'s geometry), used to resolve each component's time segment.

Only phase components whose term is a search observable are seeded — `Delay` from
`ss.delay`, `Rate` from `ss.rate`, `ConstantTerm` from `ss.phase`; every other term
(and all log-amplitude) is left untouched. `NaN` entries (no detection; the
reference antenna) leave the cell at its current value (0). A `ConstantTerm` is
seeded only when the scan maps to a SINGLE time segment for that component — a
per-integration (adhoc) constant is a within-scan residual, not the per-scan
fringe phase, so it is left at 0.
"""
function seed_from_stationization!(p0, plan::GainPlan, ss::StationSolution, g_ti)
    for g in eachindex(plan.groups)
        gp = plan.groups[g]
        parr, _ = group_arrays(plan, p0, g)      # views into p0 (phase, logamp)
        for (ci, comp) in enumerate(gp.phase)
            obs = _seed_observable(comp.term, ss)
            obs === nothing && continue
            _seed_component!(parr[ci], comp, gp.ants, obs, g_ti)
        end
    end
    return p0
end

# Seed one component's per-group array `A[nparam, ntseg, nfseg, nfb, nant]` from
# the scan's per-(station, feed) observable `obs[ant, feed]`. The value goes into
# every time segment the scan touches (usually one — PerScan) and every frequency
# segment (a wideband search value seeds each band). Feed → feed-block routing
# reuses the component's `fb1` table so it is correct for any tying: PerFeed writes
# each feed's own block, SharedFeeds averages both feeds into the shared block,
# FeedComponent writes the single feed's block, and a ReferenceRelative reference
# block gets the feed average while its relative block stays 0 (a benign seed the
# optimizer refines).
function _seed_component!(A, comp::SiteComponent, ants, obs, g_ti)
    tsegs = unique(comp.tseg_id[gti] for gti in g_ti)   # time segments THIS scan touches
    # A per-scan phase must not seed a finer (per-integration) constant — its
    # per-AP mean is degenerate with the coarser per-scan constant.
    (comp.term isa ConstantTerm && length(tsegs) != 1) && return nothing
    for (la, ant) in enumerate(ants)
        for b in 1:comp.nfb
            acc = 0.0
            n = 0
            for feed in 1:2
                comp.fb1[feed] == b || continue   # feeds routed to block b
                v = obs[ant, feed]
                isfinite(v) || continue
                acc += v
                n += 1
            end
            n == 0 && continue
            val = acc / n
            for ts in tsegs, fs in 1:comp.nfseg
                @inbounds A[1, ts, fs, b, la] = val
            end
        end
    end
    return nothing
end

"""
    fft_warmstart(plan, uvset, geom; ref_ant = 1, search = FringeSearch(),
                  stationization = Stationization(), inner = 1) -> ComponentVector

A warm-start parameter `ComponentVector` for `plan` seeded from the FFT
matched-filter fringe search + closure stationization over `uvset`. For each
(source, scan) group of band leaves: concatenate the bands along frequency,
search every `(baseline, product)` for delay/rate/phase, stationize to
per-(station, feed) values referenced to `ref_ant`, and
[`seed_from_stationization!`](@ref) them into the plan's `Delay`/`Rate`/
`ConstantTerm` phase components. `search`/`stationization` configure the two
stages; `ref_ant` must match the solve's [`ReferenceAntenna`](@ref) gauge so the
seed lands on the pinned gauge (refant = 0).

The seed resolves the delay/rate non-convexity (sidelobe locking) that a zero
start cannot escape; `fringe_solve(uvset, model; warmstart = :fft)` uses it.
"""
function fft_warmstart(
        plan::GainPlan, uvset, geom;
        ref_ant::Integer = 1,
        search::FringeSearch = FringeSearch(),
        stationization::Stationization = Stationization(),
        inner::Integer = 1,
    )
    p0 = zero_params(plan)
    nant = plan.nant
    t0_sec = geom.t0 * 3600.0                     # search works in seconds (times are hours)
    groups = Fringe._scan_group_leaves(uvset)
    pool = Fringe._ws_pool(max(Int(inner), 1))
    for keyed in groups
        grp = Fringe._materialize_concat_group(keyed, geom)
        det, _, _, _ = Fringe._search_group(grp, grp.Vg, geom.f0, t0_sec, search, pool, inner)
        ss = stationize_scan(
            det, grp.bl_pairs, grp.pol_products, nant;
            ref_ant = ref_ant, opts = stationization,
        )
        seed_from_stationization!(p0, plan, ss, grp.g_ti)
    end
    return p0
end
