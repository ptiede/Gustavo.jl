# ── Fringe structural plan lookups ───────────────────────────────────────────
#
# The structural plan routers that locate each stage's θ components by term
# Type (per-scan delay, adhoc) — never by hardcoded index.
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

# The feed-common wideband delay: not a per-band-group delay (`FreqGroups`),
# not a feed-specific inter-feed delay (`SingleFeed`).
_is_perscan_delay(tc) =
    tc.term isa Delay && !(tc.Ti isa GlobalTime) &&
    tc.Frequency isa GlobalFrequency && tc.Feed isa SharedFeeds

# The fringe step's feed-common wideband delay (`mbd` by default), the only
# `_is_perscan_delay`-signatured component its model may hold.
function _perscan_delay_plan(model, layout)
    i = findfirst(_is_perscan_delay, phase_components(model))
    return i === nothing ? nothing : layout.plans[i]
end

# Index of the adhoc component (the per-integration phase term) in the flat
# phase-component order (= `layout.plans` order).
_adhoc_idx(model) = findfirst(tc -> tc.Ti isa PerIntegration, phase_components(model))

# The adhoc component's plan (the per-integration phase term).
_adhoc_plan(model, layout) = layout.plans[_adhoc_idx(model)]
