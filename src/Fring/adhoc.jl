# ── Per-integration station gains (adhoc) ────────────────────────────────────
#
# After the per-scan fringe solution a fast, time-variable gain per station
# remains: atmospheric phase turbulence, which a per-scan constant cannot
# follow, and amplitude changes within a scan (pointing, opacity, phased-array
# efficiency). Each scan is solved on its own by the joint solver
# (`solve_joint_gains!`): the residual visibilities, frequency-averaged to one
# weighted sum per (station pair, feed pair, AP), are fit by one phase and one
# log-amplitude per (station, feed node, AP), with the source coherence per
# (station pair, feed pair) profiled out and each station's priors along time.
#
# The source coherence stays free per (station pair, feed pair) and constant
# over the scan: that keeps the model independent of the polarization basis,
# since the source EVPA, the D-terms and the source's closure phase land in it
# rather than in the station phases, and it holds the source's flux, so an
# amplitude change common to every station within a scan lands on the
# stations. The data see only phase differences within an AP; the priors hold
# the common phase and the gauge sets it after the fit.

"""
    default_adhoc_model(; feed = SharedFeeds(), phase = default_adhoc_phase_prior(),
                        logamp = default_adhoc_logamp_prior()) -> GainModel

The default [`Adhoc`](@ref Gustavo.Adhoc) model: one per-AP constant phase and
one per-AP constant log-amplitude over the global band — `phase.adhoc` and
`logamp.adhoc` components with `Ti = PerIntegration()`. `feed` is their feed
tying: `SharedFeeds()` (the default) solves one track per station — residual
atmospheric phase is non-birefringent, and a feed-common track contributes zero
inter-feed phase in any feed basis, where a `PerFeed()` solve lets per-AP noise
differ between feeds and so scatters the inter-feed phase. `PerFeed()` fits
each feed's own tracks when the per-feed structure is real.

`phase` and `logamp` are the components' priors: by default [`OUPrior`](@ref)s
along time with weakly informative hyperpriors ([`default_adhoc_phase_prior`](@ref),
[`default_adhoc_logamp_prior`](@ref)); `logamp = nothing` leaves the logamp
out, a phase-only model. Any prior the joint solve accepts will do: an
`IIDPrior` fits every AP on its own, and a state-space prior along `Ti` must
have a proper start (an `OUPrior`, a `CARMAPrior`, or a `RandomWalkPrior` with
an `init`).

The adhoc stage solves exactly one phase component and at most one logamp
component.
"""
function default_adhoc_model(;
        feed::AbstractFeedTying = SharedFeeds(), phase = default_adhoc_phase_prior(), logamp = default_adhoc_logamp_prior(),
    )
    component(prior) = GainComponent(ConstantTerm(); Ti = PerIntegration(), Feed = feed, prior)
    return GainModel(; phase = (; adhoc = component(phase)), logamp = isnothing(logamp) ? (;) : (; adhoc = component(logamp)))
end

"""
    default_adhoc_phase_prior() -> OUPrior

The default prior of the adhoc phase along time: an [`OUPrior`](@ref) whose
`scale` has a `LogNormal` hyperprior with median 10 s and whose `σ` has one
with median 0.3 rad, each with log-standard deviation 1.
"""
default_adhoc_phase_prior() = OUPrior(; scale = LogNormal(log(10.0), 1.0), σ = LogNormal(log(0.3), 1.0))

"""
    default_adhoc_logamp_prior() -> OUPrior

The default prior of the adhoc log-amplitude along time: an [`OUPrior`](@ref)
whose `scale` has a `LogNormal` hyperprior with median 60 s and whose `σ` has
one with median 0.05 (5% in amplitude), each with log-standard deviation 1.
"""
default_adhoc_logamp_prior() = OUPrior(; scale = LogNormal(log(60.0), 1.0), σ = LogNormal(log(0.05), 1.0))

# The component the adhoc stage fits: one constant per (feed node, AP) over the
# global band, under a prior the joint solve accepts.
_fits_adhoc_track(tc) =
    tc.term isa ConstantTerm && tc.Ti isa PerIntegration && tc.Frequency isa GlobalFrequency &&
    (tc.Feed isa PerFeed || tc.Feed isa SharedFeeds) && _adhoc_prior(tc)

function _adhoc_prior(tc)
    tc.prior isa IIDPrior && return true
    p = resolve_prior(tc)
    return p isa NamedTuple && keys(p) == (:Ti,) && _proper_prior(p.Ti)
end

# The structure of every adhoc model: exactly one per-integration phase
# component and at most one per-integration logamp component.
function _validate_adhoc_model(model)
    ph = Calibration._flatten_components(model.phase)
    la = Calibration._flatten_components(model.logamp)
    length(ph) == 1 || throw(
        ArgumentError(
            "the adhoc stage solves exactly one per-integration phase component; the " *
                "model's phase group holds $(length(ph)). A second (e.g. per-scan) " *
                "component is not identifiable: a constant on an antenna's track trades " *
                "against every baseline's free source term. `default_adhoc_model()` is " *
                "the standard model.",
        ),
    )
    length(la) <= 1 || throw(
        ArgumentError(
            "the adhoc stage solves at most one per-integration logamp component; the " *
                "model's logamp group holds $(length(la)). A second (e.g. per-scan) " *
                "component is not identifiable: a constant on an antenna's track trades " *
                "against every baseline's free source term.",
        ),
    )
    return nothing
end

# The adhoc components' plans, `(; phase, logamp)`: the per-integration phase
# term, and the per-integration logamp term or `nothing` without one.
function _adhoc_plans(model, layout)
    i = findfirst(tc -> tc.Ti isa PerIntegration, phase_components(model))
    j = findfirst(tc -> tc.Ti isa PerIntegration, logamp_components(model))
    return (; phase = layout.plans[i], logamp = isnothing(j) ? nothing : layout.plans[layout.nphase + j])
end

# Fit the adhoc components `blocks` (`(; phase, logamp)`, each its station
# blocks, `logamp` `nothing` without one) to one scan's weighted sums
# `rbar = Σ w·v` and `wbar = Σ w` over `(AntennaPair, FeedPair, Ti)`, AP `k` at
# the global time index `ti[k]`, and return per observable the prior each
# (station, feed) was fit under, over `(AntennaName, Feed)`, `nothing` where it
# carries no data or the observable is not fit.
function _solve_adhoc!(blocks, rbar, wbar, ti, geom::DataGeometry, gauge::AbstractGauge; tolerance = 0.1, kw...)
    cell_dims = (AntennaPair, FeedPair, Ti)
    r, w = map((rbar, wbar)) do x
        y = permutedims(x, cell_dims)
        DimArray(reshape(parent(y), 1, size(y)...), (Scan(1:1), DimensionalData.dims(y)...))
    end
    amp = !isnothing(blocks.logamp)
    fits = solve_joint_gains!(
        (; phase = [blocks.phase], logamp = amp ? [blocks.logamp] : typeof(blocks.phase)[]), (; r, w), geom;
        cover = [(t, 1) for _ in 1:1, t in ti], gauge, tolerance, kw...,
    )
    (; sg) = fits.gains
    return (;
        phase = _fit_priors(fits.phase, blocks.phase, geom, sg, fits.gains.phase),
        logamp = amp ? _fit_priors(fits.amp, blocks.logamp, geom, sg, fits.gains.amp) : _unfit_priors(geom),
    )
end

# The prior each (station, feed) of `blocks` was fit under, over
# `(AntennaName, Feed)`, `nothing` where it carries no data: a state-space
# prior from the solve's `chains`, an `IIDPrior` (which has none) on the feeds
# of each station gain of `sg` with a value in `x`.
function _fit_priors(chains, blocks, geom, sg, x)
    prior = _unfit_priors(geom)
    tying = first(blocks).plan.tying
    for chain in chains, f in 1:geom.nfeed
        _feed_node(tying, f) == chain.node || continue
        prior[blocks[chain.block].antennas[chain.station], f] = chain.prior
    end
    stated(p) = p isa NamedTuple ? only(values(p)) : p
    iid = Dict(a => stated(p) for b in blocks for (a, p) in zip(b.antennas, b.plan.priors) if stated(p) isa IIDPrior)
    for (n, (a, f, _)) in pairs(sg.at), f′ in 1:geom.nfeed
        (haskey(iid, a) && isfinite(x[n]) && _feed_node(tying, f′) == _feed_node(tying, f)) && (prior[a, f′] = iid[a])
    end
    return prior
end

_unfit_priors(geom) = DimArray(
    Union{Nothing, AbstractPrior}[nothing for _ in geom.antennas, _ in 1:geom.nfeed], (AntennaName(geom.antennas), Feed(1:geom.nfeed)),
)

"""
    solve_adhoc(rbar, wbar, antennas; gauge, model = default_adhoc_model(), kw...) -> DimStack

The per-AP station gains of one scan: `rbar = Σ w·v` and `wbar = Σ w`, the
weighted sums of the residual visibilities over frequency, over `AntennaPair`,
`FeedPair` and `Ti` (the AP epochs in seconds), fit by `model`'s adhoc
components ([`default_adhoc_model`](@ref)) with `antennas` the station names.
`kw` passes on to the joint solve ([`solve_joint_gains!`](@ref)), whose
`tolerance` defaults here to 0.1: the fit stops once a cycle moves no gain by
a tenth of its standard error.

The model is `V_ab,p(t) = exp(η_na(t) + η_nb(t) + i(φ_na(t) − φ_nb(t)))·S_ab,p`
with `S` per (station pair, feed pair) profiled out, `n` a station's feed
node, and each station's priors on its own phase and log-amplitude along time.
The data see only the phase differences within an AP: the priors hold each
AP's common phase, and `gauge` then sets it (`PinAntenna` holds the reference
at 0, `ZeroSumPhase` centers each AP on zero mean). A constant on a station's
track trades against its `S`, so the priors set it too. `S` is constant over
the scan, so a change of the source's flux within it lands on the stations'
log-amplitudes. Without a logamp component the model is phase-only.

Returns `phase` and `logamp` over `(AntennaName, Feed, Ti)`, feeds sharing a
node carrying the same values and `NaN` where a station has no data (all of
`logamp` without a logamp component), and `phase_prior` and `logamp_prior`,
the prior each (station, feed) was fit under, its hyperparameters resolved,
over `(AntennaName, Feed)`.
"""
function solve_adhoc(
        rbar::DimensionalData.AbstractDimArray{<:Complex, 3},
        wbar::DimensionalData.AbstractDimArray{<:Real, 3},
        antennas::AbstractVector;
        gauge::AbstractGauge, model::GainModel = default_adhoc_model(), kw...,
    )
    times = collect(Float64, lookup(rbar, Ti))
    nfeed = maximum(maximum, lookup(rbar, FeedPair))
    geom = DataGeometry(; times, channel_freqs = [0.0], antennas = String.(antennas), nfeed)
    layout = plan_parameters(model, geom.antennas, geom)
    plans = _adhoc_plans(model, layout)
    θ = zeros(layout.nθ)
    blocks = map(p -> isnothing(p) ? nothing : antenna_blocks(layout, θ, p.path...), plans)
    priors = _solve_adhoc!(blocks, rbar, wbar, eachindex(times), geom, _gauge_for(gauge, plans.phase.path); kw...)
    station = Dict(n => i for (i, n) in pairs(geom.antennas))
    covered = falses(length(antennas), nfeed, length(times))
    for I in CartesianIndices(wbar)
        wbar[I] > 0 || continue
        a, b = lookup(wbar, AntennaPair)[I[DimensionalData.dimnum(wbar, AntennaPair)]]
        a == b && continue
        fa, fb = lookup(wbar, FeedPair)[I[DimensionalData.dimnum(wbar, FeedPair)]]
        t = I[DimensionalData.dimnum(wbar, Ti)]
        covered[station[a], fa, t] = covered[station[b], fb, t] = true
    end
    phase, logamp = (fill(NaN, AntennaName(geom.antennas), Feed(1:nfeed), Ti(times)) for _ in 1:2)
    foreach(b -> _write_adhoc_values!(phase, b, covered), blocks.phase)
    isnothing(blocks.logamp) || foreach(b -> _write_adhoc_values!(logamp, b, covered), blocks.logamp)
    return DimensionalData.DimStack((; phase, logamp, phase_prior = priors.phase, logamp_prior = priors.logamp))
end

# Whether any cross-correlation of the AP sums carries weight.
_has_cross_data(wbar) = any(pairs(lookup(wbar, AntennaPair))) do (i, (a, b))
    a != b && any(>(0), selectdim(wbar, DimensionalData.dimnum(wbar, AntennaPair), i))
end

# Writes block `b`'s values into `out` at each (station, feed, AP) whose feed
# node has data on some feed (`covered`).
function _write_adhoc_values!(out, b, covered)
    feeds = axes(covered, 2)
    for (ai, a) in pairs(b.antennas), f in feeds, t in axes(covered, 3)
        node = _feed_node(b.plan.tying, f)
        node == 0 && continue
        any(f′ -> _feed_node(b.plan.tying, f′) == node && covered[a, f′, t], feeds) || continue
        out[a, f, t] = b.θ[1, node, 1, t, ai]
    end
    return out
end

"""
    adhoc_scan!(θ, group::XRadio.ProcessingSet, geom::DataGeometry, setup, gauge;
                executor = SerialScheduler(), hyper_optimizer = LBFGSB(m = 20)) -> NamedTuple

The adhoc solve of one scan group, on data already gain-corrected: sum the
per-(station pair, feed pair, AP) inverse-variance residual over every band
(`weighted_sums`), fit the adhoc components under each station's priors
(`solve_joint_gains!`, its hyperparameters by `hyper_optimizer`), and write
this scan's `PerIntegration` θ entries
(disjoint per scan, so scan groups may solve in parallel). `setup` holds the
parameter `layout` and the adhoc components' `plans` (`(; phase, logamp)`).

Returns per observable (`(; phase, logamp)`) the prior each (station, feed)
was fit under, over `(AntennaName, Feed, Ti)` with the scan's first AP epoch as
its one `Ti` value, `nothing` for an observable the model leaves out. A scan
with no unflagged cross-correlation keeps its θ entries, with a warning, and every prior
`nothing`.
"""
function adhoc_scan!(
        θ, group::XRadio.ProcessingSet, geom::DataGeometry, setup, gauge;
        executor = SerialScheduler(), hyper_optimizer = LBFGSB(m = 20),
    )
    (; rbar, wbar, ti) = _ap_sums(group, geom; executor)
    t0 = minimum(gti -> geom.times[gti], ti)
    per_scan(prior) = cat(prior; dims = Ti([t0]))
    if !_has_cross_data(wbar)
        scans = join(unique(String(s) for ms in values(group) for s in ms[:scan_name]), ", ")
        @warn "Adhoc: scan $scans carries no unflagged cross-correlation; its gains are not solved."
        return map(_ -> per_scan(_unfit_priors(geom)), setup.plans)
    end
    blocks = map(p -> isnothing(p) ? nothing : antenna_blocks(setup.layout, θ, p.path...), setup.plans)
    priors = _solve_adhoc!(blocks, rbar, wbar, ti, geom, _gauge_for(gauge, setup.plans.phase.path); executor, hyper_optimizer)
    return map(per_scan, priors)
end
