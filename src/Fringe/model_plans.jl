# ── Fringe structural plan lookups ───────────────────────────────────────────
#
# The structural plan routers that locate each stage's θ components by term
# Type (per-scan delay, SBD, adhoc) — never by hardcoded index. The dispersion
# term's router is `Calibration._dispersion_plan`, beside the model that
# configures it.
#
# A step whose own model names the component it needs reaches it directly
# through `layout.plantree` instead; a router is only for a component a stage
# must recognize in a model it did not name itself.

# ── Router signatures ────────────────────────────────────────────────────────
#
# Each plan router locates a θ block by `findfirst` over a compiled component's
# (term, time segmentation, frequency segmentation, feed tying) types. The predicate
# is the signature: it is what a model must contain for the router to find
# anything, so `validate_model` checks the same predicates the routers use
# rather than a separate description of them. A term-level check is too coarse —
# `Delay × GlobalTime` is a `Delay` and still leaves `_perscan_delay_plan`
# empty-handed.

# The feed-common wideband delay: not the per-band-group SBD delay
# (`FreqGroups`), not a feed-specific inter-feed delay (`SingleFeed`). Shared by
# `FringeFit`'s own wideband delay (`mbd`) and `DispersionSBDFit`'s private
# per-scan delay-refinement column — but each lives in its own step's private
# model, so `findfirst` over either step's own component list finds the right
# one without ambiguity; no positional trick is needed to tell them apart.
_is_perscan_delay(tc) =
    tc.term isa Delay && !(tc.Ti isa GlobalTime) &&
    tc.Frequency isa GlobalFrequency && tc.Feed isa SharedFeeds

# The per-band-group single-band delay and its companion constant, which are
# fit together by `refine_scan_sbd!`.
_is_sbd_delay(tc) = tc.term isa Delay && tc.Frequency isa FreqGroups
_is_sbd_constant(tc) =
    tc.term isa ConstantTerm && tc.Frequency isa FreqGroups

# The step's own feed-common wideband delay — `FringeFit`'s `mbd` when called
# on the fringe step's own `(model, layout)`, or `DispersionSBDFit`'s private
# delay-refinement column when called on the refine step's own — either way
# the only `_is_perscan_delay`-signatured component in that step's private
# model.
function _perscan_delay_plan(model, layout)
    i = findfirst(_is_perscan_delay, phase_components(model))
    return i === nothing ? nothing : layout.plans[i]
end

# The SBD components' plans `(dplan, cplan, freqgroups)` (per-scan per-band-group
# delay + companion constant), or `nothing` when the model carries none.
function _sbd_plans(model, layout)
    pcs = phase_components(model)
    i = findfirst(_is_sbd_delay, pcs)
    i === nothing && return nothing
    j = findfirst(_is_sbd_constant, pcs)
    j === nothing && error("SBD delay component present without its companion constant")
    return (dplan = layout.plans[i], cplan = layout.plans[j], freqgroups = pcs[i].Frequency.ranges)
end

# Index of the adhoc component (the per-integration phase term) in the flat
# phase-component order (= `layout.plans` order).
_adhoc_idx(model) = findfirst(tc -> tc.Ti isa PerIntegration, phase_components(model))

# The adhoc component's plan (the per-integration phase term).
_adhoc_plan(model, layout) = layout.plans[_adhoc_idx(model)]
