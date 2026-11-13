# ── Built-in solvers ─────────────────────────────────────────────────────────
#
# The bandpass and adhoc solves as `AbstractSolver`s: each declares its model
# components and implements `solve`, reading the data through `each_group`.

"""
    Bandpass(; model, gauge = ZeroSumPhase(), smoother = JointSmoother())

The bandpass stage: the time-global phase / log-amplitude station bandpass,
solved from the residual of whichever earlier solutions have already been
applied, over every scan it is given. What is fit
is `model`, a [`GainModel`](@ref) whose every component carries a prior — see
[`Fring.JointSmoother`](@ref) for the components it accepts; a prior along
frequency states how a component's values relate within a spectral window. How it is solved
lives on `smoother`, a [`Fring.JointSmoother`](@ref), which fits the complex visibilities
against an explicit per-scan source coherence and so does not assume the
calibrator is unresolved and unpolarized. `gauge`, an
[`AbstractGauge`](@ref), references the bandpass phase: `JointSmoother`
moves the solved phases onto its constraints after the fit, along the phases
the data leave free. A phase-only or amplitude-only model holds the other half
at zero. The model is self-contained, so fitting the bandpass
before or after the fringe closure is equally legal.

The phase the data leave free is a spectrum common to every station.
`PinAntenna` moves the reference station's bandpass into every other station's
track, and so into its prior, which suits a reference with a clean, stable
bandpass; `ZeroSumPhase()` takes the mean over stations and biases none.

The bandpass is fit on data the fringe solution has already corrected, for
example on calibrator scans held in memory:

    fr = fit(FringeClosure(; gauge), fit(FringeSearch(), calibrator_scans))
    bp = fit(Bandpass(; model, gauge), calibrate(fr, calibrator_scans; flag_bad = false, apply_flags = false))
"""
Base.@kwdef struct Bandpass{M <: GainModel, G <: AbstractGauge} <: AbstractSolver
    model::M
    smoother::Fring.JointSmoother = Fring.JointSmoother()
    gauge::G = ZeroSumPhase()
    function Bandpass(model::M, smoother, gauge::G) where {M, G}
        _check_solver_gauge(gauge, model)
        return new{M, G}(model, smoother, gauge)
    end
end
solution_name(::Bandpass) = :bandpass

"""
    Adhoc(; model = default_adhoc_model(), gauge = ZeroSumPhase(), hyper_optimizer = LBFGSB(m = 20))

The per-integration station-gain stage (adhoc): solves each scan's per-AP
station phases and log-amplitudes on the fringe/bandpass residual. What is fit
is `model`, a [`GainModel`](@ref) holding the adhoc phase component, an
optional adhoc logamp component, and their priors — see
[`Fring.default_adhoc_model`](@ref) for the default (feed-common, OU priors
along time) form. Each scan is one joint solve ([`Fring.solve_joint_gains!`](@ref))
of the scan's frequency-averaged visibilities, with the source coherence per
station pair and feed pair profiled out and the priors' hyperparameters
resolved per (station, feed node, scan). The data see only the phase
differences within an AP, so the priors hold each AP's common phase; `gauge`,
an [`AbstractGauge`](@ref), then sets it, so the applied correction does not
depend on it: `PinAntenna` holds the reference at 0, and `ZeroSumPhase`
centers every AP on zero mean. The source coherence is constant over the scan,
so a change of the source's flux within a scan lands on the stations'
log-amplitudes.

The solution's info holds `nscans` and `priors`: per observable (`phase`,
`logamp`) the prior each (station, feed) was fit under, its hyperparameters
resolved, over `(AntennaName, Feed, Ti)` with one `Ti` value per scan, its
first AP epoch (`nothing` where not fit).

The priors' hyperparameters are at their type-II MAP, fit by `hyper_optimizer`,
an [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) optimizer that
takes a gradient and bounds (L-BFGS-B by default), on their Laplace evidence.
"""
Base.@kwdef struct Adhoc{M <: GainModel, G <: AbstractGauge, O} <: AbstractSolver
    model::M = Fring.default_adhoc_model()
    gauge::G = ZeroSumPhase()
    hyper_optimizer::O = Fring.LBFGSB(m = 20)
    function Adhoc(model::M, gauge::G, hyper_optimizer::O) where {M, G, O}
        _check_solver_gauge(gauge, model)
        return new{M, G, O}(model, gauge, hyper_optimizer)
    end
end
solution_name(::Adhoc) = :adhoc

solver_gauge(s::Union{Bandpass, Adhoc}) = s.gauge

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
    "`JointSmoother` fits components of any term with `Ti = <PerScan, GlobalTime, " *
        "InstrumentScans or TimeBlocks>` — a time segmentation whose segments are whole " *
        "scans — and `prior = <an IIDPrior, or an OUPrior, a CARMAPrior or a " *
        "RandomWalkPrior with an init along Frequency>`. See `JointSmoother`.",
    spec,
)

supports_antenna_heterogeneity(::Bandpass) = true

Fring.can_fit(::Adhoc, tc, geom) = Fring._fits_adhoc_track(tc)
Fring.validate_model(::Adhoc, model) = Fring._validate_adhoc_model(model)

model_components(s::Adhoc, spec) = _vet_model(
    s, s.model,
    "`Adhoc` fits a phase and an optional logamp `GainComponent(ConstantTerm(); Ti = PerIntegration(), " *
        "Frequency = GlobalFrequency(), Feed = SharedFeeds() or PerFeed(), prior = " *
        "<an IIDPrior, or an OUPrior, a CARMAPrior or a RandomWalkPrior with an init along Ti>)` " *
        "— see `default_adhoc_model`.",
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

# ── Adhoc: per-scan per-AP phase solve ──────────────────────────────────

_group_setup(::Adhoc, ctx::SolveContext) = (; ctx.layout, plans = Fring._adhoc_plans(ctx.model, ctx.layout))

_solve_group(s::Adhoc, ctx::SolveContext, setup, group) = Fring.adhoc_scan!(
    ctx.θ, group, ctx.geom, setup, ctx.gauge;
    executor = inner_executor(ctx.exec), s.hyper_optimizer,
)

function solve(s::Adhoc, ctx::SolveContext)
    setup = _group_setup(s, ctx)
    results = each_group(group -> _solve_group(s, ctx, setup, group), ctx)
    isempty(results) && return (; nscans = 0)
    sort!(results; by = r -> only(lookup(r.phase, Ti)))
    priors = DimensionalData.DimStack(map(obs -> cat((r[obs] for r in results)...; dims = Ti), (; phase = :phase, logamp = :logamp)))
    return (; nscans = length(results), priors)
end
