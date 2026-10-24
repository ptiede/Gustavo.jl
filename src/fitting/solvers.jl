# ── Built-in solvers ─────────────────────────────────────────────────────────
#
# The bandpass and adhoc solves as `AbstractSolver`s: each declares its model
# components and implements `solve`, reading the data through `each_group`.

"""
    Bandpass(; gauge = ZeroSumPhase(), model = default_bandpass_model(), smoother = JointSmoother())

The bandpass stage: the time-global phase / log-amplitude station bandpass,
solved from the residual of whichever earlier solutions have already been
applied, over every scan it is given. What is fit
is `model`, a [`GainModel`](@ref) — see [`Fring.default_bandpass_model`](@ref) for the
default and the component form `JointSmoother` accepts; each component's prior
states how its values relate within a spectral window. How it is solved
lives on `smoother`, a [`Fring.JointSmoother`](@ref), which fits the complex visibilities
against an explicit per-scan source coherence and so does not assume the
calibrator is unresolved and unpolarized. `gauge`, an
[`AbstractGauge`](@ref), references the bandpass phase: `JointSmoother`
imposes its constraints exactly in every step. `JointSmoother` solves one complex gain per
(station, feed, frequency segment); a phase-only or amplitude-only model holds
the other half at zero. The model is self-contained, so fitting the bandpass
before or after the fringe closure is equally legal.

The phase the data leave free is a spectrum common to every station.
`PinAntenna` moves the reference station's bandpass into every other station's
track, and so into its prior, which suits a reference with a clean, stable
bandpass; `ZeroSumPhase()` takes the mean over stations and biases none.

The bandpass is fit on data the fringe solution has already corrected, for
example on calibrator scans held in memory:

    fr = fit(FringeClosure(; gauge), fit(FringeSearch(), calibrator_scans))
    bp = fit(Bandpass(; gauge), calibrate(fr, calibrator_scans; flag_bad = false, apply_flags = false))
"""
Base.@kwdef struct Bandpass{M <: GainModel, G <: AbstractGauge} <: AbstractSolver
    model::M = Fring.default_bandpass_model()
    smoother::Fring.JointSmoother = Fring.JointSmoother()
    gauge::G = ZeroSumPhase()
    function Bandpass(model::M, smoother, gauge::G) where {M, G}
        _check_solver_gauge(gauge, model)
        return new{M, G}(model, smoother, gauge)
    end
end
solution_name(::Bandpass) = :bandpass

"""
    AdhocPhase(; gauge = ZeroSumPhase(), model = default_adhoc_model(), smoother = PerTrackAdhocSmoother())
    AdhocPhase(smoother; gauge = ZeroSumPhase())

The per-integration atmospheric-phase stage (adhoc phasing): solves the
globally-closing per-AP station phase on the fringe/bandpass residual. What is
fit is `model`, a [`GainModel`](@ref) holding the single adhoc component and
its prior along time — see [`Fring.default_adhoc_model`](@ref) for the default
(feed-common, OU prior) form. How the tracks are fit under that prior lives on
`smoother`, a pluggable [`Fring.AbstractAdhocSmoother`](@ref)
([`Fring.PerTrackAdhocSmoother`](@ref) or [`Fring.JointKalmanSmoother`](@ref), which
requires the feed-common (`SharedFeeds`) model). The one-argument form takes
the smoother and keeps the default model. `gauge`, an
[`AbstractGauge`](@ref), sets each AP's convention after smoothing, so the
applied correction does not depend on it: `PinAntenna` holds the first of its
references present in that AP at 0, and `ZeroSumPhase` centers every AP on the
stations covered throughout the scan.

The solution's info holds `nscans` and `priors`: the prior each (station, feed)
track was fit under, its hyperparameters resolved, over `(AntennaName, Feed, Ti)` with
one `Ti` value per scan, its first AP epoch.
"""
Base.@kwdef struct AdhocPhase{M <: GainModel, S <: Fring.AbstractAdhocSmoother, G <: AbstractGauge} <: AbstractSolver
    model::M = Fring.default_adhoc_model()
    smoother::S = Fring.PerTrackAdhocSmoother()
    gauge::G = ZeroSumPhase()
    function AdhocPhase(model::M, smoother::S, gauge::G) where {M, S, G}
        _check_solver_gauge(gauge, model)
        return new{M, S, G}(model, smoother, gauge)
    end
end
AdhocPhase(smoother::Fring.AbstractAdhocSmoother; gauge = ZeroSumPhase()) = AdhocPhase(; smoother, gauge)
solution_name(::AdhocPhase) = :adhoc

solver_gauge(s::Union{Bandpass, AdhocPhase}) = s.gauge

# Every component path of `model` without its `:phase`/`:logamp` root, over the
# base tree and each station entry.
function _model_component_names(model::GainModel)
    names = Tuple[]
    for (_, tree) in _antenna_variants(model)
        append!(names, Calibration._leaf_paths(tree.phase), Calibration._leaf_paths(tree.logamp))
    end
    return unique!(names)
end

_check_solver_gauge(gauge::AbstractGauge, model::GainModel) =
    check_gauge_components(gauge, _model_component_names(model))
_check_solver_gauge(gauge, ::GainModel) = throw(
    ArgumentError(
        "`gauge` must be an AbstractGauge, got $(repr(gauge)) of type $(typeof(gauge)): " *
            "`PinAntenna(\"A1\")` or `PinAntenna(2)` (a reference antenna, or a ranked list), " *
            "`ZeroSumPhase()`, or `ByComponent((; component = gauge, …); default = gauge)`.",
    ),
)

# ── Model components ─────────────────────────────────────────────────────────

# Vet a fit's model argument against its solver's declared capability, at
# compile time, before any data is read: no component the solver cannot fit
# (`can_fit`; its θ block would stay at zero and the solution would look
# fitted), then the solver's own whole-tree requirements (`validate_model`).
# Both checks run once per distinct station tree — the base and each
# `antennas` entry's effective pair — with the can_fit error naming the station
# whose entry carries the component. `accepted` finishes that error with the
# component form the solver family does fit. Returns `m`; the runner
# resolves it against the antenna table.
function _vet_model(solver, m::GainModel, accepted, spec = nothing)
    for (antenna, tree) in _antenna_variants(m)
        at = antenna === nothing ? "" : " (antenna $(repr(antenna)) entry)"
        for tc in Calibration._flatten_components(tree)
            # Without a geometry `can_fit` gets `nothing`, so a capability that
            # needs one fails there instead of being answered from a default.
            Fring.can_fit(solver, tc, isnothing(spec) ? nothing : spec.geom) || throw(
                ArgumentError(
                    "$(nameof(typeof(solver))) cannot fit the component " *
                        "$(Calibration.component_label(tc))$at; its parameters would " *
                        "never be solved. " * accepted,
                ),
            )
        end
        Fring.validate_model(solver, tree)
    end
    return m
end

# The distinct station trees a model assigns, as `antenna => (; phase, logamp)`
# pairs (`nothing` for the base).
function _antenna_variants(m::Calibration.GainModel)
    vars = Pair{Any, Any}[nothing => (; phase = m.phase, logamp = m.logamp)]
    for k in keys(m.antennas)
        push!(vars, k => Calibration.antenna_components(m, k))
    end
    return vars
end

model_components(s::Bandpass, spec) = _vet_model(
    s.smoother, s.model,
    "`JointSmoother` fits `GainComponent(ConstantTerm(); " *
        "Ti = <GlobalTime, InstrumentScans or TimeBlocks>, Frequency = <any " *
        "segmentation>, Feed = PerFeed(), prior = <nothing, or a RandomWalkPrior, " *
        "OUPrior or CARMAPrior along Frequency>)` — a time segmentation whose segments each span " *
        "several scans. A segment holding one scan (`PerScan`, `PerIntegration`) is " *
        "not fitted: the band mean of its gain is degenerate with that scan's source " *
        "coherence. See `default_bandpass_model`.",
    spec,
)

supports_antenna_heterogeneity(::Bandpass) = true

model_components(s::AdhocPhase, spec) = _vet_model(
    s.smoother, s.model,
    "The adhoc smoothers fit `GainComponent(ConstantTerm(); Ti = PerIntegration(), " *
        "Frequency = GlobalFrequency(), Feed = SharedFeeds() or PerFeed(), prior = " *
        "<nothing, or a RandomWalkPrior or OUPrior along Ti>)` " *
        "(`JointKalmanSmoother`: `SharedFeeds()` and an `OUPrior` only) — see `default_adhoc_model`.",
    spec,
)

# ── Bandpass: accumulate per scan group, then one joint solve ────────────────

function _group_setup(::Bandpass, ctx::SolveContext)
    layout = ctx.layout
    # The accumulators' labels, shared by every scan group: each stored cross
    # pair and feed pair of the set.
    members = [ms for group in values(ctx.groups) for ms in values(group)]
    antenna_pairs, feeds = Fring._cross_cell_labels(
        map(Fring._member_antenna_pairs, members), map(feed_pairs, members), ctx.geom,
    )
    # `ctx.layout` holds only the bandpass model's components. The two observables are
    # located by name through the layout's component tree: the flat `plans` list
    # carries one entry per station-signature group, so its positions stop naming
    # them as soon as a model differs across stations.
    return (;
        antenna_pairs, feeds, layout, geom = ctx.geom,
        paths = (;
            phase = Fring._bandpass_paths(layout.plantree, :phase),
            logamp = Fring._bandpass_paths(layout.plantree, :logamp),
        ),
    )
end

function _solve_group(s::Bandpass, ctx::SolveContext, setup, group)
    rl, wl = Fring.accumulate_bandpass(
        group, ctx.geom, setup.antenna_pairs, setup.feeds; executor = inner_executor(ctx.exec),
    )
    # `ti` locates this scan on the solve's global time axis, which is how a
    # time-segmented bandpass tells which segment the scan belongs to. A scan
    # lies within one segment of any segmentation coarser than a scan, so its
    # first sample names the segment.
    t0 = minimum(minimum(XRadio.times(ms)) for ms in values(group))
    sources = unique(only(XRadio.sources(ms)) for ms in values(group))
    length(sources) == 1 || throw(
        ArgumentError("a scan group observes several sources: $(join(sources, ", "))")
    )
    return (; rl, wl, ti = Calibration._time_index(ctx.geom, t0), source = only(sources))
end

function solve(s::Bandpass, ctx::SolveContext)
    setup = _group_setup(s, ctx)
    results = each_group(group -> _solve_group(s, ctx, setup, group), ctx)
    isempty(results) && return (; nscans = 0)     # no scans → bandpass stays 0
    report = Fring._solve_bandpass!(s.smoother, ctx.θ, results, setup; gauge = ctx.gauge, executor = inner_executor(ctx.exec))
    # The solve's per-track record travels with the solution's info, so a
    # consumer can tell a measured bandpass track from a placeholder without
    # re-deriving it from the gains (where the two look identical).
    return (;
        nscans = length(results),
        sources = unique(String[r.source for r in results]),
        report...,
    )
end

# ── AdhocPhase: per-scan per-AP phase solve ──────────────────────────────────

_group_setup(::AdhocPhase, ctx::SolveContext) = Fring._adhoc_plan(ctx.model, ctx.layout)

_solve_group(s::AdhocPhase, ctx::SolveContext, adhoc_plan, group) = Fring.adhoc_scan!(
    ctx.θ, group, ctx.geom, adhoc_plan, s.smoother, ctx.gauge;
    executor = inner_executor(ctx.exec),
)

function solve(s::AdhocPhase, ctx::SolveContext)
    adhoc_plan = _group_setup(s, ctx)
    results = each_group(group -> _solve_group(s, ctx, adhoc_plan, group), ctx)
    isempty(results) && return (; nscans = 0)
    sort!(results; by = r -> only(lookup(r, Ti)))
    return (; nscans = length(results), priors = cat(results...; dims = Ti))
end
