# ── FringeFit stage: the term-list fringe model + the matched-filter estimator ─
#
# The composable pipeline's fringe stage in three parts:
#
# - `FringeModel` — WHAT is solved: the gauge pin (`ref_ant`) plus an ordered
#   list of phase-term elements. Each element declares its own feed scope
#   through its tying (`SharedFeeds`, `FeedComponent(2)`, …), so the model is
#   specified feed by feed; adding an effect is adding an element.
#   `fringe_phase_components` compiles each element through
#   `model_components(element, geom)` and concatenates in list order.
# - `MatchedFilter <: AbstractFringeEstimator` — HOW it is estimated: today's
#   stage A (per-baseline delay/rate matched-filter search + closure-screened
#   station WLS). The search, `Stationization`, and the cross-hand
#   fit-on-subset selection live HERE, not on the model — an alternative
#   estimator (e.g. a Schwab–Cotton-style global LS) plugs in with no
#   vestigial search/stationization options.
# - The stage machinery the runner drives through the streaming layer:
#   residual cubes for `rounds > 1`, R–L fit-on-subset masking, the stage-B
#   component filter, and the detection/flag tables recorded on the solution.

"""
    SingleBandDelay()

Per-scan per-band-group single-band delay (fourfit's SBD) — a
[`FringeModel`](@ref) term-list element. A station's per-band signal path can
move relative to its phase-cal tones between scans (~30 ns has been observed),
which neither the wideband delay (one slope across all band groups) nor the
time-invariant per-channel bandpass can track. Instrumental, not propagation.

Compiles to a coupled per-band-group pair — a per-scan `Delay` plus its
companion per-scan constant, over `FrequencyBands` ranges computed from the
data geometry ([`fringe_band_groups`](@ref)) — or to nothing when the
frequency axis has fewer than 2 band groups (a single group is fully
degenerate with the wideband delay). Fit from within-band chunk slopes by the
refine stage, nearly orthogonal to the cross-band observables that set the
wideband delay and dTEC.
"""
struct SingleBandDelay end

function model_components(::SingleBandDelay, geom::DataGeometry)
    bands = fringe_band_groups(geom.channel_freqs)
    length(bands) >= 2 || return nothing
    # The Delay coordinate is (f − f0) with the GLOBAL f0, so correcting a
    # group slope about the group's own centre νg needs the companion per-group
    # constant −2πτ(νg − f0): net phase 2πτ(f − νg), zero at the group centre —
    # the cross-band solution is untouched. The pair nests under the element's
    # key (`θ.phase.<key>.delay` / `.constant`).
    return (
        delay = TiedComponent(Delay(), PerScan(), FrequencyBands(bands), SharedFeeds()),
        constant = TiedComponent(ConstantTerm(), PerScan(), FrequencyBands(bands), SharedFeeds()),
    )
end

"""
    default_fringe_terms() -> NamedTuple

The default [`FringeModel`](@ref) term list — the standard VLBI fringe model,
specified feed by feed as a named list (the key names the component; compiled
component order = list order):

1. per-scan constant phase, feed-common (`SharedFeeds`): atmosphere/clock.
2. R–L constant offset, `GlobalTime × FeedComponent(2)`: the instrumental
   feed-2 − feed-1 phase offset, stable across the observation (EHT-HOPS /
   rPICARD assumption) — solved once from all scans' cross hands, so bright
   polarized scans pin it and weak scans inherit it.
3. per-scan wideband (multi-band) delay, feed-common.
4. R–L delay offset, `GlobalTime × FeedComponent(2)`: one instrumental offset
   per station for the whole track. Replace `GlobalTime()` with `PerScan()` to
   fit it per scan — its scatter is an instrument-stability diagnostic, at the
   cost that a scan with no cross-hand detection leaves feed 2 untied.
5. per-scan rate, feed-common: the fringe rate is common to both feeds. There
   is deliberately NO feed-specific rate here — the Rate phase is
   `2π·rate·(t − t0_global)` with the WHOLE-TRACK reference, so per-feed rate
   noise is levered by hours into large, arbitrary scan-to-scan R–L phase
   jumps; R–L rate is negligible (EHT-HOPS convention). A genuine offset is
   opted into by ADDING `TiedComponent(Rate(), GlobalTime(),
   GlobalFrequency(), FeedComponent(2))` — cross-hand rows then join the rate
   solve.

Ionospheric dispersion (dTEC) and single-band delay (SBD) are NOT modeled
here — they are fit by a separate [`DispersionSBDFit`](@ref) pipeline step,
on the fringe-corrected residual.

Omit an element to drop the effect; add a `Calibration.TiedComponent` (term ×
time segmentation × frequency segmentation × feed tying) to model a new one.
"""
default_fringe_terms() = (
    atmos = TiedComponent(ConstantTerm(), PerScan(), GlobalFrequency(), SharedFeeds()),
    rl_phase = TiedComponent(ConstantTerm(), GlobalTime(), GlobalFrequency(), FeedComponent(2)),
    mbd = TiedComponent(Delay(), PerScan(), GlobalFrequency(), SharedFeeds()),
    rl_delay = TiedComponent(Delay(), GlobalTime(), GlobalFrequency(), FeedComponent(2)),
    rate = TiedComponent(Rate(), PerScan(), GlobalFrequency(), SharedFeeds()),
)

"""
    FringeModel(; ref_ant = 1, terms = default_fringe_terms())

WHAT the fringe stage solves — the model specification of a `FringeFit` step:
the gauge pin plus an ordered list of phase-term elements.

- `ref_ant` — the gauge pin: a 1-based antenna index or a station code
  (`"PT"`). Part of the MODEL (it changes what is solved), not of the
  execution configuration.
- `terms` — the ordered, NAMED term list (a `NamedTuple`; each key names the
  component it compiles to). Each value is a bare `Calibration.TiedComponent`
  (a gain term × time segmentation × frequency segmentation × feed tying).
  Adding an effect is adding a named element; the list order is the compiled
  component order. See [`default_fringe_terms`](@ref) for the default list and
  how to modify it.

[`DispersionModel`](@ref) and [`SingleBandDelay`](@ref) are NOT valid `terms`
elements: they are fit by a separate [`DispersionSBDFit`](@ref) pipeline step,
not by the fringe search, so a `FringeModel` carrying one would compile a θ
column no stage ever fits (a silent no-fit) — rejected at construction instead.
"""
struct FringeModel{T <: NamedTuple}
    ref_ant::Union{Integer, AbstractString, Symbol}
    terms::T
    function FringeModel{T}(ref_ant, terms) where {T}
        for (k, t) in pairs(terms)
            t isa DispersionModel && throw(
                ArgumentError(
                    "FringeModel: `terms.$k` is a DispersionModel — dispersion is fit by a " *
                        "separate DispersionSBDFit pipeline step, not by FringeModel's term " *
                        "list. Remove it from `terms` and add `DispersionSBDFit(; dispersion = " *
                        "$(t))` to the pipeline instead.",
                ),
            )
            t isa SingleBandDelay && throw(
                ArgumentError(
                    "FringeModel: `terms.$k` is a SingleBandDelay — SBD is fit by a separate " *
                        "DispersionSBDFit pipeline step, not by FringeModel's term list. Remove " *
                        "it from `terms` and add `DispersionSBDFit(; sbd = $(t))` to the " *
                        "pipeline instead.",
                ),
            )
        end
        return new{T}(ref_ant, terms)
    end
end
function FringeModel(; ref_ant = 1, terms = default_fringe_terms())
    terms isa NamedTuple || throw(
        ArgumentError("FringeModel: `terms` must be a NamedTuple naming each element."),
    )
    return FringeModel{typeof(terms)}(ref_ant, terms)
end

"""
    MatchedFilter(; search = FringeSearch(), closure = Stationization(), rounds = 1,
                  cross_hand_fit_on = AllScans())

HOW the fringe stage is estimated (an [`AbstractFringeEstimator`](@ref)):
today's stage A — a per-baseline delay/rate matched-filter `search` on every
scan group, then ONE global closure-screened station WLS (`closure`) that ties
the feeds and solves any track-global columns. `rounds` re-runs the search on
the residual (each round divides out the current solution and accumulates the
leftover) — an iteration knob of THIS estimator.

`cross_hand_fit_on` selects which scans' CROSS-HAND rows feed the station
solve (fit-on-subset / apply-everywhere: fit the R–L offsets from a few bright
polarized scans, apply them track-wide). Parallel-hand rows are unaffected.

When `search.pfa_max` is finite and `closure` is left at its default, the
stationization's fixed SNR floor is dropped (`snr_min = 0`): the PFA gate IS
the acceptance decision. A custom `closure` is used as given.
"""
Base.@kwdef struct MatchedFilter{S <: AbstractScanSelection} <: AbstractFringeEstimator
    search::FringeSearch = FringeSearch()
    closure::Stationization = Stationization()
    rounds::Int = 1
    cross_hand_fit_on::S = AllScans()
end

# ── Model compilation ─────────────────────────────────────────────────────────

"""
    fringe_phase_components(fm::FringeModel, geom::DataGeometry) -> NamedTuple

The fringe stage's gain-model phase components as a named tree: each element of
`fm.terms` compiled through `model_components(element, geom)` under its list key,
in list order — the list order IS the compiled component order. An element that
compiles to nothing contributes no key; one that compiles to several components
nests them under its key.

Throws `ArgumentError` when two compiled components share a routing signature:
the structural plan routers (`_perscan_delay_plan`, `_sbd_plans`,
`Calibration._dispersion_plan`) locate θ blocks by `findfirst` over (term,
segmentation, tying) types, so a second matching component would compile θ
columns no stage ever writes — a silent no-fit.
"""
function fringe_phase_components(fm::FringeModel, geom::DataGeometry)
    comps = _compile_named(fm.terms, geom)
    _validate_fringe_components(_flatten_components(comps))
    return comps
end

# Compile a named term list into a named component tree: each element keyed by
# its list name, dropped when it emits nothing, nested when it emits several.
# The names come from the type parameter so the keys stay compile-time constants
# and the tree's type is inferred.
_compile_named(terms::NamedTuple{names}, geom::DataGeometry) where {names} =
    _compile_named(names, terms, geom)
_compile_named(::Tuple{}, terms, geom::DataGeometry) = (;)
function _compile_named(names::Tuple, terms, geom::DataGeometry)
    k = first(names)
    v = model_components(terms[k], geom)
    return _prepend_named(Val(k), v, _compile_named(Base.tail(names), terms, geom))
end
_prepend_named(::Val, ::Nothing, rest) = rest
_prepend_named(::Val{k}, v, rest) where {k} = merge(NamedTuple{(k,)}((v,)), rest)

# Reject compiled component sets a findfirst router cannot address uniquely,
# and exact duplicates (indistinguishable θ blocks are degenerate columns).
function _validate_fringe_components(comps::Tuple)
    for i in eachindex(comps), j in (i + 1):length(comps)
        _same_component_signature(comps[i], comps[j]) && throw(
            ArgumentError(
                "FringeModel: terms compile two identical components " *
                    "($(component_label(comps[i]))) — the routers and solvers cannot " *
                    "distinguish their θ blocks. Remove the duplicate element.",
            ),
        )
    end
    _at_most_one(_is_perscan_delay, comps, "the per-scan feed-common delay signature")
    return comps
end

function _at_most_one(pred, comps::Tuple, what::String)
    n = count(pred, comps)
    n <= 1 || throw(
        ArgumentError(
            "FringeModel: $n compiled components match $what — the plan router " *
                "routes to the first and the rest would never be fit. Remove the " *
                "duplicate element(s).",
        ),
    )
    return nothing
end

_same_component_signature(a::TiedComponent, b::TiedComponent) =
    typeof(a.component.term) === typeof(b.component.term) &&
    a.component.time == b.component.time &&
    _same_freq_segmentation(a.component.freq, b.component.freq) &&
    a.tying == b.tying

_same_freq_segmentation(a, b) = a == b
_same_freq_segmentation(a::FrequencyBands, b::FrequencyBands) = a.ranges == b.ranges

# Whether the compiled fringe components opt into a solvable feed-specific
# rate (an R–L rate column) — cross-hand rows must then join the rate system.
# Accepts the named component tree (or any component collection).
_has_feed_rate(comps::NamedTuple) = _has_feed_rate(_flatten_components(comps))
_has_feed_rate(comps) =
    any(tc -> tc.component.term isa Rate && tc.tying isa FeedComponent, comps)

# What `MatchedFilter` does with a compiled component's θ block:
#
#   :delay / :rate / :phase — stage B's station system of that kind writes it,
#                             from the per-baseline search observable of the
#                             same name.
#   nothing                 — the matched filter does not touch it.
#
# ONE estimator's vocabulary, hence private: a global least-squares fringe
# fitter has no use for it, fitting θ through `evaluate_gains` directly.
#
# The stage-B kinds cover exactly the single-parameter, band-wide terms whose
# observable the search measures. Everything else is `nothing` and so unfittable
# by this estimator rather than approximated: `_solve_kind_cols!` writes one θ
# column per (station, feed, time) node — the block's first parameter at the
# FIRST frequency segment — so a multi-parameter term (a polynomial) would have its
# trailing parameters left at zero, and a frequency-resolved term (the bandpass,
# `ConstantTerm × ChannelBlocks`) would have every segment but the first left at
# zero while the search wrote its band-wide phase into that one. A `Dispersion`
# term or a `FrequencyBands`-segmented one (SBD) already falls through to
# `nothing` here — the term/freq-type checks below exclude them without special
# casing — so `can_fit` correctly rejects either if found in a `FringeModel`'s
# own term list (they belong to a separate `DispersionSBDFit` step instead).
function matched_kind(tc)
    tc.component.time isa PerIntegration && return nothing
    # The per-baseline search measures ONE delay/rate/phase across the whole
    # band, so only a component spanning it can receive that estimate.
    tc.component.freq isa GlobalFrequency || return nothing
    term = tc.component.term
    term isa Delay && return :delay
    term isa Rate && return :rate
    term isa ConstantTerm && return :phase
    return nothing
end

# Stage-B engine components `(plan, kind)` — the delay/rate/phase terms the
# search + stationization solve, as declared by `matched_kind`, over EVERY
# phase component of `model`. Called on a single step's own private model
# (the fringe step's), so every component here genuinely belongs to that step
# — no later step's component can structurally collide with a stage-B
# signature, since it lives in a separate step's own model entirely.
function fringe_stage_components(model, layout)
    comps = Tuple{ComponentPlan, Symbol}[]
    for (i, tc) in enumerate(phase_components(model))
        kind = matched_kind(tc)
        kind in (:delay, :rate, :phase) || continue
        push!(comps, (layout.plans[i], kind))
    end
    return comps
end

# ── MatchedFilter's capability ───────────────────────────────────────────────

can_fit(::MatchedFilter, tc) = matched_kind(tc) !== nothing

# What the matched filter REQUIRES to exist. Each absent item costs the
# estimator its own output silently rather than crashing: `solve_station_systems!`
# skips a kind with no components, dropping every scan's search estimate for
# that observable, and `refine_scan_dispersion!` skips the Δτ half of the joint
# (Δτ, dTEC) fit when the per-scan delay plan is missing, biasing the dTEC it
# does report by exactly the degeneracy the joint fit exists to break.
function validate_model(est::MatchedFilter, comps)
    kinds = map(matched_kind, comps)
    for (kind, what) in (
            (:delay, "a delay component (the per-baseline delay search has nowhere to go)"),
            (:rate, "a rate component (the per-baseline rate search has nowhere to go)"),
            (:phase, "a constant-phase component (the station phase solve has nowhere to go)"),
        )
        kind in kinds || throw(
            ArgumentError(
                "$(nameof(typeof(est))) requires $what. Add one to the FringeModel's " *
                    "`terms` — see `default_fringe_terms`.",
            ),
        )
    end
    any(_is_perscan_delay, comps) || throw(
        ArgumentError(
            "$(nameof(typeof(est))) requires a per-scan feed-common wideband delay " *
                "component (`Delay` × a non-`GlobalTime` time segmentation × " *
                "`GlobalFrequency` × `SharedFeeds`): the refine stage fits it jointly " *
                "with dTEC, and a delay tied any other way leaves that fit with only " *
                "its degenerate half. Add one to the FringeModel's `terms` — see " *
                "`default_fringe_terms`.",
        ),
    )
    return nothing
end

# The effective Stationization for a MatchedFilter run: with the PFA gate
# active and the DEFAULT closure, the search's `valid` IS the acceptance
# decision — drop the fixed SNR floor (the legacy solver's rule). A customized
# closure is honored as given. An opt-in R–L rate needs the cross-hand rows in
# the rate system, so `cross_hand_rate` is forced on.
function resolve_closure(est::MatchedFilter, rl_rate_on::Bool)
    c = est.closure
    if isfinite(est.search.pfa_max) && c == Stationization()
        c = Stationization(snr_min = 0.0)
    end
    if rl_rate_on && !c.cross_hand_rate
        c = Stationization(c.snr_min, true, c.phase_rewrap_iters, c.reject_sigma, c.reject_iters)
    end
    return c
end

# ── Stage machinery over the streaming layer ─────────────────────────────────

# One residual cell: the visibility divided by its baseline's gain product. A
# degenerate gain yields NaN, which the search's weight handling excludes. The
# result keeps the visibility's element type, so a native-precision cube stays
# native precision.
function _residual_cell(v, ga, gb)
    den = ga * conj(gb)
    degenerate = abs(ga) < 1.0e-12 || abs(gb) < 1.0e-12 || !isfinite(den)
    return oftype(v, degenerate ? complex(NaN, NaN) : v / den)
end

"""
    residual_vis(ev::GainEvaluator, θ, stack, win::GeometryWindow) -> DimArray

The scan's residual visibilities: `stack`'s `:vis` layer divided by the current
θ gains evaluated on `win`'s (global chan, global ti) window — the search input
for residual re-search rounds. Carries the visibilities' dims and element type.
"""
function residual_vis(
        ev::GainEvaluator, θ::AbstractVector, stack::AbstractDimStack, win::GeometryWindow,
    )
    g = evaluate_gains(ev, θ, win.chan_idx, win.ti_idx)   # (nchan, nti, nant, 2)
    ants = UVData.baselines(stack).pairs
    feeds = map(correlation_feed_pair, pol_products(stack))
    # Indexing the gains by the baselines' antenna vector and the products' feed
    # vector is an outer product over (Baseline, Pol) — the cube's last two axes
    # — so the whole residual is one fused broadcast with no intermediate.
    ga = view(g, :, :, first.(ants), first.(feeds))
    gb = view(g, :, :, last.(ants), last.(feeds))
    return _residual_cell.(stack[:vis], ga, gb)
end

# R–L fit-on-subset: invalidate the CROSS-HAND detections of every scan the
# estimator's `cross_hand_fit_on` selection does NOT pick, so only the selected
# scans' cross-hand rows feed the station solve — the solved time-global R–L
# components still apply to every scan. Parallel-hand rows are untouched. `dets` is the per-scan
# `StationScanDetections` vector (mutated); `snr` supplies per-scan SNRs for
# selections that need them.
function mask_unselected_cross_hands!(dets, fit_on::AbstractScanSelection, groups, snr)
    fit_on isa AllScans && return dets
    recs = [
        (; index = s.index, source = s.source, scan = s.scan, snr = Float64(snr[s.index]))
            for s in groups
    ]
    sel = Set(select_scans(fit_on, recs))
    for (gi, d) in enumerate(dets)
        gi in sel && continue
        for p in eachindex(d.feeds)
            fa, fb = d.feeds[p]
            fa == fb && continue
            for bi in axes(d.det, 1)
                d.det[bi, p] = _INVALID_DETECTION
            end
        end
    end
    return dets
end

# EHT-HOPS-style station flags: a station that PARTICIPATES in a scan (has
# baselines there) but is left UNCONSTRAINED by the surviving stage-B rows
# keeps identity gains — record it as (station, geometry scan id) so
# `apply_calibration` zero-weights its baselines instead of passing raw phases
# through at full weight.
function unconstrained_flags(dets, covered, geom::DataGeometry)
    flags = Tuple{Int, Int}[]
    for gi in eachindex(dets)
        scanid = geom.scan_of_time[dets[gi].ti]
        stations = Set{Int}()
        for (a, b) in dets[gi].bl_pairs
            a == b && continue
            push!(stations, a); push!(stations, b)
        end
        for st in stations
            (st, gi) in covered || push!(flags, (st, scanid))
        end
    end
    return flags
end

# Flatten per-scan detection rows into parallel plain vectors for the solution
# `info` — HDF5-representable and cheap to filter (`suspect_fringes`).
function detection_table(scan_dets)
    n = sum(length, scan_dets; init = 0)
    det_scan = Vector{Int}(undef, n); det_ant_a = Vector{Int}(undef, n)
    det_ant_b = Vector{Int}(undef, n); det_pol = Vector{String}(undef, n)
    det_snr = Vector{Float64}(undef, n); det_pfa = Vector{Float64}(undef, n)
    i = 0
    for (gi, rows) in enumerate(scan_dets), r in rows
        i += 1
        det_scan[i] = gi; det_ant_a[i] = r.a; det_ant_b[i] = r.b
        det_pol[i] = r.pol; det_snr[i] = r.snr; det_pfa[i] = r.pfa
    end
    return (; det_scan, det_ant_a, det_ant_b, det_pol, det_snr, det_pfa)
end

# The flag block for the solution `info` (plain parallel vectors,
# HDF5-representable): stage-B-unconstrained (station, scan) pairs and any
# intra-site baselines excluded for crosstalk.
function flag_table(station_flags, excl)
    pairs_ab = excl === nothing ? Tuple{Int, Int}[] : sort!([p for p in excl if p[1] < p[2]])
    return (;
        flagged_ant = Int[f[1] for f in station_flags],
        flagged_scan = Int[f[2] for f in station_flags],
        excluded_ant_a = Int[p[1] for p in pairs_ab],
        excluded_ant_b = Int[p[2] for p in pairs_ab],
    )
end
