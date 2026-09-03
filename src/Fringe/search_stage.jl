# ── FringeFit stage: the term-list fringe model + the matched-filter estimator ─
#
# The composable pipeline's fringe stage in three parts:
#
# - `FringeModel` — WHAT is solved: an ordered list of phase-term elements
#   (the gauge pin, `gauge`, is run-wide — see `CalibrationPipeline` in
#   pipeline/protocol.jl). Each element declares its own feed scope
#   through its tying (`SharedFeeds`, `SingleFeed(2)`, …), so the model is
#   specified feed by feed; adding a component is adding an element.
#   `fringe_phase_components` compiles each element through
#   `model_components(element, spec)` and concatenates in list order.
# - `MatchedFilter <: AbstractFringeEstimator` — HOW it is estimated: today's
#   stage A (per-baseline delay/rate matched-filter search + closure-screened
#   station WLS). The search and `Stationization` live HERE, not on the model —
#   an alternative estimator (e.g. a Schwab–Cotton-style global LS) plugs in
#   with no vestigial search/stationization options.
# - The stage machinery the runner drives through the streaming layer:
#   residual cubes for `rounds > 1`, the stage-B component filter, and the
#   detection/flag tables recorded on the solution.

"""
    SingleBandDelay(; freq = BandGroups())

Per-scan single-band delay (fourfit's SBD) — a [`FringeModel`](@ref) term-list
element. A station's signal path can move relative to its phase-cal tones
between scans (~30 ns has been observed), which neither the wideband delay (one
slope across the whole band) nor the time-invariant per-channel bandpass can
track. Instrumental, not propagation.

`freq` is the frequency partition the delay is resolved on — any
`AbstractFrequencySegmentation`, resolved through `Calibration.materialize` and
`Calibration.segment_ranges`. `BandGroups()` (the default) gives one delay per
gap-detected band group; `PerSpectralWindow()` gives every spectral window its
own delay and offset, which is what a per-spw signal-path difference that MOVES
between scans needs — a `GlobalTime` bandpass can only fit such a step's track
average. A `FreqGroups` is taken as the partition itself.

Compiles to a coupled per-group pair — a per-scan `Delay` plus its companion
per-scan constant over that partition — or to nothing when the partition holds
fewer than 2 groups (a single group is fully degenerate with the wideband
delay). Fit from within-group chunk slopes by the refine stage, nearly
orthogonal to the cross-band observables that set the wideband delay and dTEC.
"""
Base.@kwdef struct SingleBandDelay{F}
    freq::F = BandGroups()
end

function model_components(s::SingleBandDelay, spec)
    geom = spec.geom
    freqgroups = segment_ranges(materialize(s.freq, geom), geom)
    length(freqgroups) >= 2 || return nothing
    # The Delay coordinate is (f − f0) with the GLOBAL f0, so correcting a
    # group slope about the group's own centre νg needs the companion per-group
    # constant −2πτ(νg − f0): net phase 2πτ(f − νg), zero at the group centre —
    # the cross-band solution is untouched. The pair nests under the element's
    # key (`θ.phase.<key>.delay` / `.constant`).
    return (
        delay = GainComponent(Delay(); Ti = PerScan(), Frequency = FreqGroups(freqgroups), Feed = SharedFeeds()),
        constant = GainComponent(ConstantTerm(); Ti = PerScan(), Frequency = FreqGroups(freqgroups), Feed = SharedFeeds()),
    )
end

"""
    default_fringe_terms(; rel_time = PerScan()) -> NamedTuple

The default [`FringeModel`](@ref) term list — the standard VLBI fringe model,
specified feed by feed as a named list (the key names the component; compiled
component order = list order):

1. per-scan constant phase, feed-common (`SharedFeeds`): atmosphere/clock.
2. per-scan wideband (multi-band) delay, feed-common.
3. relative delay offset, `Ti = rel_time`, `Feed = SingleFeed(2)`: the instrumental
   feed-2 − feed-1 group-delay offset.
4. per-scan rate, feed-common: the fringe rate is common to both feeds. There
   is deliberately NO feed-specific rate here — the inter-feed rate is
   negligible (EHT-HOPS convention), so a column for it would buy little but
   the noise of fitting one. A genuine offset is opted into by ADDING
   `GainComponent(Rate(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SingleFeed(2))`,
   which gives the inter-feed rate its own column. Every correlation product's
   rate row enters the system either way; the tying alone decides whether it
   reads as `ṙ_a − ṙ_b` (feed-common) or `ṙ_{a,p} − ṙ_{b,q}` (feed-specific).
   An added rate component must carry the same time segmentation as the
   constants beside it, so that one epoch zeroes every rate coordinate at once —
   see [`scan_phase_epoch`](@ref).

`rel_time` is the inter-feed delay offset's time segmentation, an
`AbstractTimeSegmentation`:

- `PerScan()` (default) fits an offset per scan, so its scan-to-scan scatter is
  a direct instrument-stability diagnostic, and the model has no cross-scan
  column — each scan's system is independent, which is what lets consecutive
  scan-local steps share one pass over the data.
- `GlobalTime()` fits ONE offset per station for the whole track (the EHT-HOPS
  / rPICARD assumption that the instrumental offset is stable): bright scans pin
  it and weak scans inherit it. The cost is that a track-global column couples
  every scan into one system, which forgoes scan fusion.

There is deliberately NO inter-feed PHASE offset. A feed-2 constant is not
separable from the source's cross-hand phase — both are a rigid shift of the
feed-2 block, and the model has no source column — so fitting one and dividing
it out removes the source's polarization angle along with the instrument's
offset. Omitting it removes nothing instead: the R–L phase is left in the data
for a downstream polarization fit, which is where the source and the instrument
can be separated with a source model. Nothing else in the model is disturbed by
the omission, because the offset it would have carried lands in the feed-COMMON
constant, and a feed-common term cancels identically in every feed difference —
`QQ − PP` and the cross hands are untouched.

The R–L phase is still MEASURABLE without the column, and better: `QQ − PP` on
one baseline is `ρ_a − ρ_b` with the source and atmospheric terms cancelling
algebraically, at parallel-hand SNR and with no fit in between (see
`DetectionRow`'s `phase`).

The inter-feed DELAY is a different case and is kept: a delay decoheres across
the band, so leaving it in costs signal, while a constant phase costs nothing to
carry.

Ionospheric dispersion (dTEC) and single-band delay (SBD) are NOT modeled
here — they are fit by a separate [`DispersionSBDFit`](@ref) pipeline step,
on the fringe-corrected residual.

Omit an element to drop the component; add a `Calibration.GainComponent` (a term with its
time segmentation, frequency segmentation, and feed tying) to model a new one.
"""
default_fringe_terms(; rel_time::AbstractTimeSegmentation = PerScan()) = (
    atmos = GainComponent(ConstantTerm(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds()),
    mbd = GainComponent(Delay(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds()),
    rel_delay = GainComponent(Delay(); Ti = rel_time, Frequency = GlobalFrequency(), Feed = SingleFeed(2)),
    rate = GainComponent(Rate(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds()),
)

"""
    FringeModel(; terms = default_fringe_terms())

WHAT the fringe stage solves — the model specification of a `FringeFit` step:
an ordered list of phase-term elements. The gauge pin (`gauge`) is run-wide,
not part of any one step's model — see [`CalibrationPipeline`](@ref).

- `terms` — the ordered, NAMED term list (a `NamedTuple`; each key names the
  component it compiles to). Each value is a bare `Calibration.GainComponent`
  (a gain term with its time segmentation, frequency segmentation, and feed tying).
  Adding a component is adding a named element; the list order is the compiled
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
scan group, then the closure-screened station WLS (`closure`) that ties the
feeds. With one round and an all-per-scan term list the station systems are
block-diagonal, and each scan's system is solved as its scan is searched, so
the pass is scan-local ([`scan_local_solve`](@ref)); a track-global column or
`rounds > 1` instead pools every scan's detections into one solve at the end
of the pass. `rounds` re-runs the search on the residual (each round divides
out the current solution and accumulates the leftover) — an iteration knob of
THIS estimator.

The `search` measures every baseline and gates nothing; `closure.pfa_max` is the
one detection threshold, deciding which measurements are real fringes and so
which stations are calibrated (see [`Stationization`](@ref)).

After the station solve closes, every cell is re-measured AT the delay and rate
that solution predicts for it ([`steer_scan`](@ref)). That measurement is NOT
gated: `pfa_max` decides fringe-group membership on the blind pass, and a
station in that group has its baselines measured at the known fringe location to
arbitrarily low SNR. `steer_cells` sizes the trial count of the recorded
`pfa_steer` — a significance a caller may read, not a threshold the pass
applies — and `steer_cells = 0` skips the pass entirely. It runs only where the
solve is scan-local; a pooled solve has no station parameters while the data is
still resident, and the steered columns come back `NaN`.
"""
Base.@kwdef struct MatchedFilter <: AbstractFringeEstimator
    search::FringeSearch = FringeSearch()
    closure::Stationization = Stationization()
    rounds::Int = 1
    steer_cells::Float64 = 9.0
end

# ── Model compilation ─────────────────────────────────────────────────────────

"""
    fringe_phase_components(fm::FringeModel, spec) -> NamedTuple

The fringe stage's gain-model phase components as a named tree: each element of
`fm.terms` compiled through `model_components(element, spec)` under its list
key, in list order — the list order IS the compiled component order
(`spec = (; geom, antennas)`, the step compile spec). An element that
compiles to nothing contributes no key; one that compiles to several components
nests them under its key.

Throws `ArgumentError` when two compiled components share a routing signature:
the structural plan routers (`_perscan_delay_plan`, `_sbd_plans`,
`Calibration._dispersion_plan`) locate θ blocks by `findfirst` over (term,
segmentation, tying) types, so a second matching component would compile θ
columns no stage ever writes — a silent no-fit.
"""
function fringe_phase_components(fm::FringeModel, spec)
    comps = _compile_named(fm.terms, spec)
    _validate_fringe_components(_flatten_components(comps))
    return comps
end

# Compile a named term list into a named component tree: each element keyed by
# its list name, dropped when it emits nothing, nested when it emits several.
# The names come from the type parameter so the keys stay compile-time constants
# and the tree's type is inferred.
_compile_named(terms::NamedTuple{names}, spec) where {names} =
    _compile_named(names, terms, spec)
_compile_named(::Tuple{}, terms, spec) = (;)
function _compile_named(names::Tuple, terms, spec)
    k = first(names)
    v = model_components(terms[k], spec)
    return _prepend_named(Val(k), v, _compile_named(Base.tail(names), terms, spec))
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

_same_component_signature(a::GainComponent, b::GainComponent) =
    typeof(a.term) === typeof(b.term) &&
    a.Ti == b.Ti && a.Frequency == b.Frequency && a.Feed == b.Feed

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
# `ConstantTerm` over `ChannelBlocks`) would have every segment but the first left at
# zero while the search wrote its band-wide phase into that one. A `Dispersion`
# term or a `FreqGroups`-segmented one (SBD) already falls through to
# `nothing` here — the term/freq-type checks below exclude them without special
# casing — so `can_fit` correctly rejects either if found in a `FringeModel`'s
# own term list (they belong to a separate `DispersionSBDFit` step instead).
function matched_kind(tc)
    tc.Ti isa PerIntegration && return nothing
    # The per-baseline search measures ONE delay/rate/phase across the whole
    # band, so only a component spanning it can receive that estimate.
    tc.Frequency isa GlobalFrequency || return nothing
    term = tc.term
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

# One round and an all-per-scan term list make the station systems
# block-diagonal per scan, so each scan's WLS closes inside `estimate_scan!`
# and the pass is scan-local. A cross-scan time segmentation (a `GlobalTime`
# inter-feed offset shares a column across scans), an opaque term (its
# compiled segmentation is unknowable without the geometry), or `rounds > 1`
# (the re-search reads the whole pass's residual) each force the pooled path.
scan_local_solve(est::MatchedFilter, fm::FringeModel) =
    est.rounds <= 1 &&
    all(t -> t isa GainComponent && component_is_per_scan(t), values(fm.terms))

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
                "component (a `Delay` with a non-`GlobalTime` time segmentation, " *
                "`Frequency = GlobalFrequency()`, `Feed = SharedFeeds()`): the refine stage fits it jointly " *
                "with dTEC, and a delay tied any other way leaves that fit with only " *
                "its degenerate half. Add one to the FringeModel's `terms` — see " *
                "`default_fringe_terms`.",
        ),
    )
    return nothing
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

# Flatten per-scan search rows into parallel plain vectors for the solution
# `info` — HDF5-representable and cheap to filter (`suspect_fringes`). Every
# measured cell is here, so `det_detected` is what selects the real fringes.
function detection_table(scan_dets)
    n = sum(length, scan_dets; init = 0)
    det_scan = Vector{Int}(undef, n); det_ant_a = Vector{Int}(undef, n)
    det_ant_b = Vector{Int}(undef, n); det_pol = Vector{String}(undef, n)
    det_snr = Vector{Float64}(undef, n); det_pfa = Vector{Float64}(undef, n)
    det_delay = Vector{Float64}(undef, n); det_rate = Vector{Float64}(undef, n)
    det_phase = Vector{Float64}(undef, n)
    det_detected = Vector{Bool}(undef, n)
    det_snr_steer = Vector{Float64}(undef, n); det_pfa_steer = Vector{Float64}(undef, n)
    det_delay_steer = Vector{Float64}(undef, n); det_rate_steer = Vector{Float64}(undef, n)
    det_steered = Vector{Bool}(undef, n)
    i = 0
    for (gi, rows) in enumerate(scan_dets), r in rows
        i += 1
        det_scan[i] = gi; det_ant_a[i] = r.a; det_ant_b[i] = r.b
        det_pol[i] = r.pol; det_snr[i] = r.snr; det_pfa[i] = r.pfa
        det_delay[i] = r.delay; det_rate[i] = r.rate; det_phase[i] = r.phase
        det_detected[i] = r.detected
        det_snr_steer[i] = r.snr_steer; det_pfa_steer[i] = r.pfa_steer
        det_delay_steer[i] = r.delay_steer; det_rate_steer[i] = r.rate_steer
        det_steered[i] = r.steered
    end
    return (;
        det_scan, det_ant_a, det_ant_b, det_pol, det_snr, det_pfa,
        det_delay, det_rate, det_phase, det_detected,
        det_snr_steer, det_pfa_steer, det_delay_steer, det_rate_steer, det_steered,
    )
end

"""
    scan_station_terms(model, layout, θ, ti) -> (delay, rate)

Per-`(station, feed)` group delay (s) and fringe rate (Hz) at time index `ti`,
summed over the stage-B components of `model`. This is the decode
[`fringe_station_solutions`](@ref) reports, for ONE scan and without a finished
solution, so a solve step can read its own station parameters while the scan's
data is still resident. Entries are `NaN` where no component constrains that node.
"""
function scan_station_terms(model, layout, θ, ti::Integer)
    nant = layout.nant
    comps = fringe_stage_components(model, layout)
    delay = fill(NaN, nant, 2)
    rate = fill(NaN, nant, 2)
    for a in 1:nant, f in 1:2
        d = 0.0; r = 0.0; hd = false; hr = false
        for (plan, kind) in comps
            node = _feed_node(plan.tying, f)        # fseg 1: stage-B terms are GlobalFrequency
            node == 0 && continue
            v = _component_leaf(plan, θ)[1, node, 1, plan.tseg_id[ti], a]
            if kind === :delay
                d += v; hd = true
            elseif kind === :rate
                r += v; hr = true
            end
        end
        hd && (delay[a, f] = d)
        hr && (rate[a, f] = r)
    end
    return (delay, rate)
end

"""
    scan_phase_epoch(model, layout, ti) -> Union{Float64, Nothing}

The epoch (hours) at which `model`'s constant phase terms are the phase, for the
time segment holding time index `ti`: the origin of the rate columns covering
that segment, since a rate contributes `2π·ṙ·(t − t0)` and vanishes only there.
`nothing` when the model carries no rate component, which leaves the constant
free of any epoch.

A search must reference its detection phases to this epoch, or the station
solve reads them as a constant they are not. Getting it wrong is not a bias but
a variance: a phase quoted a lever arm `Δt` from where it was measured inherits
`2π·σ_ṙ·Δt` of the rate's own uncertainty, which the feed-common columns can
absorb through their rate but a feed-relative offset — modeled with no rate of
its own — cannot.

Rate components of different time segmentations put their origins in different
places, and no single epoch then zeroes them all; that model is rejected rather
than silently referenced to one of them.
"""
function scan_phase_epoch(model, layout, ti::Integer)
    epoch = nothing
    for (plan, kind) in fringe_stage_components(model, layout)
        kind === :rate || continue
        o = Float64(plan.tstate[plan.tseg_id[ti]])
        if epoch === nothing
            epoch = o
        elseif !isapprox(o, epoch; atol = 1.0e-9)
            error(
                "scan_phase_epoch: the model's rate components disagree on the epoch of " *
                    "time index $ti ($epoch h vs $o h). A constant phase is the phase at " *
                    "the epoch where every rate coordinate vanishes, and rate components " *
                    "with different time segmentations have no such epoch in common. Give " *
                    "every Rate term the same time segmentation as the constant it " *
                    "accompanies (`PerScan()` for the default fringe term list).",
            )
        end
    end
    return epoch
end

"""
    steer_scan(stack, res, bl_pairs, pols, f0, t0, sta_delay, sta_rate; cells)

Re-measure every `(baseline, product)` of a materialized scan group AT the delay
and rate the station solution predicts for it — `τ_{a,fa} − τ_{b,fb}` and the
same difference in rate — rather than wherever a blind search found its peak.

This is what recovers a fringe too weak to survive a blind search: the trial
count collapses from the search plane's ~1e4 cells to the handful `cells`
covering the prediction's uncertainty, so the SNR needed to clear a given
false-alarm probability drops by roughly 1.5σ. Nothing about the measurement
changes — only how many chances noise had to fake it.

The steered SNR is directly comparable to the blind one because it is formed
against the SAME noise: the blind pass reports `snr = |D_blind|/σ`, so
re-evaluating the exact matched filter at the blind peak recovers `σ = |D_blind|/snr`
without needing the search plane. Cells with no usable data, or whose two
stations are not both solved this scan, come back `NaN`.
"""
function steer_scan(
        stack, res, bl_pairs, pols, f0::Real, t0::Real,
        sta_delay::AbstractMatrix, sta_rate::AbstractMatrix;
        cells::Real = 9.0,
    )
    V = stack[:vis]
    W = stack[:weights]
    freqs = frequencies(stack)
    times = timestamps(stack) .* 3600.0
    # `res` covers only cross baselines; `keep` maps its column back to the cube's.
    keep = findall(pr -> pr[1] != pr[2], UVData.baselines(stack).pairs)
    dims = (length(bl_pairs), length(pols))
    sdelay = fill(NaN, dims); srate = fill(NaN, dims)
    samp = fill(NaN, dims); ssnr = fill(NaN, dims); spfa = fill(NaN, dims)
    for p in eachindex(pols), j in eachindex(bl_pairs)
        res[:valid][j, p] || continue
        snr0 = res[:snr][j, p]
        snr0 > 0 || continue
        a, b = bl_pairs[j]
        fa, fb = correlation_feed_pair(pols[p])
        dpred = sta_delay[a, fa] - sta_delay[b, fb]
        rpred = sta_rate[a, fa] - sta_rate[b, fb]
        (isfinite(dpred) && isfinite(rpred)) || continue
        bi = keep[j]
        Vb = view(V, :, :, bi, p)
        Wb = view(W, :, :, bi, p)
        # σ of the blind pass, recovered from its own reported SNR.
        σ = abs(
            _exact_matched_filter(
                Vb, Wb, freqs, times, f0, t0,
                res[:delay][j, p], res[:rate][j, p]
            )
        ) / snr0
        σ > 0 || continue
        D = _exact_matched_filter(Vb, Wb, freqs, times, f0, t0, dpred, rpred)
        Wsum = 0.0
        for i in eachindex(Wb)
            w = Wb[i]
            (isfinite(w) && w > 0) && (Wsum += w)
        end
        sdelay[j, p] = dpred
        srate[j, p] = rpred
        samp[j, p] = Wsum > 0 ? abs(D) / Wsum : NaN
        ssnr[j, p] = abs(D) / σ
        spfa[j, p] = fringe_pfa(ssnr[j, p], cells)
    end
    return (delay = sdelay, rate = srate, amp = samp, snr = ssnr, pfa = spfa)
end

# The flag block for the solution `info` (plain parallel vectors,
# HDF5-representable): the stage-B-unconstrained (station, scan) pairs.
function flag_table(station_flags)
    return (;
        flagged_ant = Int[f[1] for f in station_flags],
        flagged_scan = Int[f[2] for f in station_flags],
    )
end
