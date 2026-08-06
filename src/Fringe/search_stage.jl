# ── FringeFit stage: the term-list fringe model + the matched-filter estimator ─
#
# The composable pipeline's fringe stage in three parts:
#
# - `FringeModel` — WHAT is solved: an ordered list of phase-term elements
#   (the gauge pin, `ref_ant`, is run-wide — see `CalibrationPipeline` in
#   pipeline/protocol.jl). Each element declares its own feed scope
#   through its tying (`SharedFeeds`, `FeedComponent(2)`, …), so the model is
#   specified feed by feed; adding an effect is adding an element.
#   `fringe_phase_components` compiles each element through
#   `model_components(element, geom)` and concatenates in list order.
# - `MatchedFilter <: AbstractFringeEstimator` — HOW it is estimated: today's
#   stage A (per-baseline delay/rate matched-filter search + closure-screened
#   station WLS). The search and `Stationization` live HERE, not on the model —
#   an alternative estimator (e.g. a Schwab–Cotton-style global LS) plugs in
#   with no vestigial search/stationization options.
# - The stage machinery the runner drives through the streaming layer:
#   residual cubes for `rounds > 1`, the stage-B component filter, and the
#   detection/flag tables recorded on the solution.

"""
    SingleBandDelay()

Per-scan per-band-group single-band delay (fourfit's SBD) — a
[`FringeModel`](@ref) term-list element. A station's per-band signal path can
move relative to its phase-cal tones between scans (~30 ns has been observed),
which neither the wideband delay (one slope across all band groups) nor the
time-invariant per-channel bandpass can track. Instrumental, not propagation.

Compiles to a coupled per-band-group pair — a per-scan `Delay` plus its
companion per-scan constant, over `FreqGroups` ranges computed from the
data geometry ([`fringe_freq_groups`](@ref)) — or to nothing when the
frequency axis has fewer than 2 band groups (a single group is fully
degenerate with the wideband delay). Fit from within-band chunk slopes by the
refine stage, nearly orthogonal to the cross-band observables that set the
wideband delay and dTEC.
"""
struct SingleBandDelay end

function model_components(::SingleBandDelay, geom::DataGeometry)
    freqgroups = fringe_freq_groups(geom.channel_freqs)
    length(freqgroups) >= 2 || return nothing
    # The Delay coordinate is (f − f0) with the GLOBAL f0, so correcting a
    # group slope about the group's own centre νg needs the companion per-group
    # constant −2πτ(νg − f0): net phase 2πτ(f − νg), zero at the group centre —
    # the cross-band solution is untouched. The pair nests under the element's
    # key (`θ.phase.<key>.delay` / `.constant`).
    return (
        delay = TiedComponent(Delay(), PerScan(), FreqGroups(freqgroups), SharedFeeds()),
        constant = TiedComponent(ConstantTerm(), PerScan(), FreqGroups(freqgroups), SharedFeeds()),
    )
end

"""
    default_fringe_terms(; rel_time = PerScan()) -> NamedTuple

The default [`FringeModel`](@ref) term list — the standard VLBI fringe model,
specified feed by feed as a named list (the key names the component; compiled
component order = list order):

1. per-scan constant phase, feed-common (`SharedFeeds`): atmosphere/clock.
2. relative constant offset, `rel_time × FeedComponent(2)`: the instrumental
   feed-2 − feed-1 phase offset.
3. per-scan wideband (multi-band) delay, feed-common.
4. relative delay offset, `rel_time × FeedComponent(2)`: the instrumental
   feed-2 − feed-1 group-delay offset.
5. per-scan rate, feed-common: the fringe rate is common to both feeds. There
   is deliberately NO feed-specific rate here — the Rate phase is
   `2π·rate·(t − t0_global)` with the WHOLE-TRACK reference, so per-feed rate
   noise is levered by hours into large, arbitrary scan-to-scan inter-feed
   phase jumps; the inter-feed rate is negligible (EHT-HOPS convention). A
   genuine offset is opted into by ADDING `TiedComponent(Rate(), GlobalTime(),
   GlobalFrequency(), FeedComponent(2))`, which gives the inter-feed rate its
   own column. Every correlation product's rate row enters the system either
   way; the tying alone decides whether it reads as `ṙ_a − ṙ_b` (feed-common)
   or `ṙ_{a,p} − ṙ_{b,q}` (feed-specific).

`rel_time` is the time segmentation of BOTH inter-feed offsets (elements 2 and
4), an `AbstractTimeSegmentation`:

- `PerScan()` (default) fits an offset per scan, so its scan-to-scan scatter is
  a direct instrument-stability diagnostic, and the model has no cross-scan
  column — each scan's system is independent, which is what lets consecutive
  scan-local steps share one pass over the data.
- `GlobalTime()` fits ONE offset per station for the whole track (the EHT-HOPS
  / rPICARD assumption that the instrumental offset is stable): bright
  polarized scans pin it and weak scans inherit it, so a scan with no
  cross-hand detection still has feed 2 tied. The cost is that a track-global
  column couples every scan into one system, which forgoes scan fusion.

The phase offset (element 2) also absorbs the source's cross-hand phase, which
is not separable from it — both are a rigid shift of the feed-2 block. So the
offset is estimable only up to that constant, and calibrated cross-hand phase
carries one conventional constant per segment: per scan under `PerScan()`
(scan-to-scan jitter, since each scan estimates its own), one for the track
under `GlobalTime()`. `GlobalTime()` is therefore also how to ask for a
track-constant cross-hand phase convention. A source whose cross-hand phase
genuinely varies scan to scan cannot be represented under `GlobalTime()`: the
variation lands in the residuals, where the robust loss downweights it.

Ionospheric dispersion (dTEC) and single-band delay (SBD) are NOT modeled
here — they are fit by a separate [`DispersionSBDFit`](@ref) pipeline step,
on the fringe-corrected residual.

Omit an element to drop the effect; add a `Calibration.TiedComponent` (term ×
time segmentation × frequency segmentation × feed tying) to model a new one.
"""
default_fringe_terms(; rel_time::AbstractTimeSegmentation = PerScan()) = (
    atmos = TiedComponent(ConstantTerm(), PerScan(), GlobalFrequency(), SharedFeeds()),
    rel_phase = TiedComponent(ConstantTerm(), rel_time, GlobalFrequency(), FeedComponent(2)),
    mbd = TiedComponent(Delay(), PerScan(), GlobalFrequency(), SharedFeeds()),
    rel_delay = TiedComponent(Delay(), rel_time, GlobalFrequency(), FeedComponent(2)),
    rate = TiedComponent(Rate(), PerScan(), GlobalFrequency(), SharedFeeds()),
)

"""
    FringeModel(; terms = default_fringe_terms())

WHAT the fringe stage solves — the model specification of a `FringeFit` step:
an ordered list of phase-term elements. The gauge pin (`ref_ant`) is run-wide,
not part of any one step's model — see [`CalibrationPipeline`](@ref).

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
    terms::T
    function FringeModel{T}(terms) where {T}
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
        return new{T}(terms)
    end
end
function FringeModel(; terms = default_fringe_terms())
    terms isa NamedTuple || throw(
        ArgumentError("FringeModel: `terms` must be a NamedTuple naming each element."),
    )
    return FringeModel{typeof(terms)}(terms)
end

"""
    MatchedFilter(; search = FringeSearch(), closure = Stationization(), rounds = 1)

HOW the fringe stage is estimated (an [`AbstractFringeEstimator`](@ref)):
today's stage A — a per-baseline delay/rate matched-filter `search` on every
scan group, then ONE global closure-screened station WLS (`closure`) that ties
the feeds and solves any track-global columns. `rounds` re-runs the search on
the residual (each round divides out the current solution and accumulates the
leftover) — an iteration knob of THIS estimator.

When `search.pfa_max` is finite and `closure` is left at its default, the
stationization's fixed SNR floor is dropped (`snr_min = 0`): the PFA gate IS
the acceptance decision. A custom `closure` is used as given.
"""
Base.@kwdef struct MatchedFilter <: AbstractFringeEstimator
    search::FringeSearch = FringeSearch()
    closure::Stationization = Stationization()
    rounds::Int = 1
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
_same_freq_segmentation(a::FreqGroups, b::FreqGroups) = a.ranges == b.ranges

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
# term or a `FreqGroups`-segmented one (SBD) already falls through to
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
# decision — drop the fixed SNR floor. A customized closure is honored as given.
function resolve_closure(est::MatchedFilter)
    c = est.closure
    return isfinite(est.search.pfa_max) && c == Stationization() ?
        Stationization(snr_min = 0.0) : c
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

# EHT-HOPS-style station flags: a station that PARTICIPATES in a scan (has
# baselines there) but is left UNCONSTRAINED by the surviving stage-B rows
# keeps identity gains — record it as (station, geometry scan id) so
# `apply_calibration` zero-weights its baselines instead of passing raw phases
# through at full weight.
function unconstrained_flags(dets, covered, geom::DataGeometry)
    flags = Tuple{Int, Int}[]
    for gi in eachindex(dets)
        scanid = geom.scan_of_time[_scan_ti(dets[gi])]
        stations = Set{Int}()
        for (a, b) in _scan_bl_pairs(dets[gi])
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
