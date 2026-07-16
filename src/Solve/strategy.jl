# ── Pluggable solve strategies ────────────────────────────────────────────────
#
# The objective (a `FringePosterior`) and the SOLVE STRATEGY are decoupled: the
# same posterior can be minimized by pure gradient descent OR by a block-coordinate
# scheme that alternates a gradient step with direct linear block solves (e.g. the
# per-channel phase bandpass, which is linear and crawls under LBFGS). Both are
# first-class, and a user extends the scheme by subtyping `AbstractSolveStep` and
# adding one `apply_step` method.
#
#   strategy = GradientDescent(LBFGS())                         # pure gradient (default)
#   strategy = BlockCoordinate(                                 # hybrid block-coordinate
#       GradientStep(LBFGS(); frozen = (:bandpass, :adhoc)),    #   nonlinear params
#       FreqStep(:bandpass),                                    #   linear bandpass (freq axis)
#       TimeStep(:adhoc);                                       #   linear adhoc (time axis)
#       rounds = 3,
#   )

# The optimizer seam: `_optimize_map(reparam_posterior, y0, optimizer; …)` → (u, info).
# No core method; `GustavoOptimizationExt` (load `Optimization` + `OptimizationOptimJL`)
# builds the `OptimizationProblem` and calls `solve`. `optimizer = nothing` lets it
# pick a default (LBFGS).
function _optimize_map end

abstract type AbstractSolveStrategy end
abstract type AbstractSolveStep end

"""
    GradientDescent(optimizer = nothing; gauge = nothing, scaling = AutoScale(),
                    maxiters = 1000)

Pure gradient descent: one `Optimization.jl` solve of the reparameterized (gauge-
fixed, scaled) MAP problem with `optimizer` (any Optimization.jl optimizer, or
`nothing` for the extension's default LBFGS). The default `fringe_solve` strategy.
"""
struct GradientDescent{O, G, S} <: AbstractSolveStrategy
    optimizer::O
    gauge::G
    scaling::S
    maxiters::Int
end
GradientDescent(optimizer = nothing; gauge = nothing, scaling = AutoScale(), maxiters::Integer = 1000) =
    GradientDescent(optimizer, gauge, scaling, Int(maxiters))

"""
    BlockCoordinate(steps...; rounds = 3)

Block-coordinate strategy: run the ordered [`AbstractSolveStep`](@ref)s (each
updates the parameter vector in place) `rounds` times. Compose e.g. a
[`GradientStep`](@ref) freezing the bandpass with a [`FreqStep`](@ref) that solves
it directly.
"""
struct BlockCoordinate{T <: Tuple} <: AbstractSolveStrategy
    steps::T
    rounds::Int
end
BlockCoordinate(steps::AbstractSolveStep...; rounds::Integer = 3) = BlockCoordinate(steps, Int(rounds))

"""
    GradientStep(optimizer = nothing; frozen = (), scaling = AutoScale(), maxiters = 1000)

A block step: a gradient solve over the free parameters with the named `frozen`
components held fixed (in addition to the reference-antenna gauge).
"""
struct GradientStep{O, S} <: AbstractSolveStep
    optimizer::O
    frozen::Tuple
    scaling::S
    maxiters::Int
end
GradientStep(optimizer = nothing; frozen = (), scaling = AutoScale(), maxiters::Integer = 1000) =
    GradientStep(optimizer, Tuple(frozen), scaling, Int(maxiters))

"""
    LinearPhaseStep(component; axis = :freq, shared_feeds = false,
                    smoother = NoSmoothing(detrend = false, phase_rewrap_iters = 0))

A block step that solves a per-slice phase `component` DIRECTLY (a linear closure
solve) via [`refine_phase_component!`](@ref), instead of by gradient descent:

- `axis = :freq` — a phase varying along FREQUENCY, one per channel (see
  [`FreqStep`](@ref); the typical use is the instrumental bandpass).
- `axis = :time` — a phase varying along TIME, one per AP (see [`TimeStep`](@ref);
  the typical use is the atmospheric adhoc phase).

`shared_feeds = true` solves one feed-common phase per slice (`SharedFeeds`). This
is the extension point: point it at any per-slice phase block your model defines.

`prior` selects the per-slice solve. `nothing` (the default) inherits the posterior's
[`ComponentPriors`](@ref) entry for `component` (falling back to the robust one-shot
[`solve_adhoc_phasing`](@ref) when there is none); a concrete `AbstractPrior`
overrides it (`NoPrior()` forces the one-shot path). A [`BandpassARPrior`](@ref)
(`:freq`) or [`OUPrior`](@ref) (`:time`) triggers the EXACT prior-coupled MAP block
update (matrix-free CG), which regularizes weak channels/APs. `map_maxgn`/`map_tol`
bound the outer Gauss–Newton loop, `cg_tol`/`cg_maxit` the inner CG.
"""
struct LinearPhaseStep{SM, PR} <: AbstractSolveStep
    component::Symbol
    axis::Symbol
    shared_feeds::Bool
    smoother::SM
    prior::PR
    map_maxgn::Int
    map_tol::Float64
    cg_tol::Float64
    cg_maxit::Int
end
LinearPhaseStep(
    component::Symbol; axis::Symbol = :freq, shared_feeds::Bool = false,
    smoother = NoSmoothing(detrend = false, phase_rewrap_iters = 0), prior = nothing,
    map_maxgn::Integer = 5, map_tol::Real = 1.0e-8, cg_tol::Real = 1.0e-10, cg_maxit::Integer = 2000,
) = LinearPhaseStep(
    component, axis, shared_feeds, smoother, prior,
    Int(map_maxgn), Float64(map_tol), Float64(cg_tol), Int(cg_maxit),
)

"""
    FreqStep(component = :bandpass; shared_feeds = false, prior = nothing, smoother = …) -> LinearPhaseStep

Solve a phase that varies along FREQUENCY directly (`axis = :freq`) — the per-channel
instrumental bandpass. Per-feed by default (bandpass is instrumental, not
non-birefringent). See [`LinearPhaseStep`](@ref) for `prior` / MAP-block options.
"""
FreqStep(component::Symbol = :bandpass; shared_feeds::Bool = false, kw...) =
    LinearPhaseStep(component; axis = :freq, shared_feeds = shared_feeds, kw...)

"""
    TimeStep(component = :adhoc; shared_feeds = true, prior = nothing, smoother = …) -> LinearPhaseStep

Solve a phase that varies along TIME directly (`axis = :time`) — the per-AP
atmospheric adhoc phase. Feed-common by default (atmospheric phase is
non-birefringent). (Not a numerical time step — a per-AP phase solve.) See
[`LinearPhaseStep`](@ref) for `prior` / MAP-block options.
"""
TimeStep(component::Symbol = :adhoc; shared_feeds::Bool = true, kw...) =
    LinearPhaseStep(component; axis = :time, shared_feeds = shared_feeds, kw...)

# ── Running a strategy ────────────────────────────────────────────────────────
#
# `ctx` bundles the shared problem: `(; post, ref_ant, solve_kwargs)`. `run_strategy`
# and `apply_step` take `(strategy/step, ctx, p) -> (p, info)`.

function run_strategy(strat::GradientDescent, ctx, p0)
    plan = ctx.post.plan
    gge = strat.gauge === nothing ? ReferenceAntenna(ctx.ref_ant) : strat.gauge
    reparam = build_reparam(plan, gge, strat.scaling, p0)
    rpost = ReparamPosterior(ctx.post, reparam)
    u, sinfo = _optimize_map(
        rpost, collect(to_free(reparam, p0)), strat.optimizer;
        maxiters = strat.maxiters, ctx.solve_kwargs...,
    )
    return to_full(reparam, u), merge((; nfree = nfree(reparam)), sinfo)
end

function run_strategy(strat::BlockCoordinate, ctx, p0)
    p = p0
    for _ in 1:strat.rounds
        for step in strat.steps
            p, _ = apply_step(step, ctx, p)
        end
    end
    # Report the true final objective at the block-coordinate optimum (one eval),
    # since the last step (e.g. a linear bandpass solve) carries none.
    final_obj = LogDensityProblems.logdensity(ctx.post, flatten(p))
    return p, (; rounds = strat.rounds, final_objective = final_obj)
end

apply_step(step::GradientStep, ctx, p) =
    run_strategy(
    GradientDescent(step.optimizer, _freeze_gauge(ctx.post.plan, ctx.ref_ant, step.frozen), step.scaling, step.maxiters),
    ctx, p,
)

function apply_step(step::LinearPhaseStep, ctx, p)
    prior = _resolve_step_prior(step, ctx.post.prior)
    refine_phase_component!(
        p, ctx.post.plan, ctx.post.uvset, ctx.post.geom;
        component = step.component, axis = step.axis, ref_ant = ctx.ref_ant,
        shared_feeds = step.shared_feeds, smoother = step.smoother, prior = prior,
        map_maxgn = step.map_maxgn, map_tol = step.map_tol,
        cg_tol = step.cg_tol, cg_maxit = step.cg_maxit,
    )
    return p, (;)
end

# The step's explicit `prior` wins; otherwise inherit the posterior's `ComponentPriors`
# entry for the step's component (so `strategy = BlockCoordinate(FreqStep(:bandpass))`
# with `prior = BandpassARPrior(...)` on the *solve* automatically does the MAP block).
_resolve_step_prior(step::LinearPhaseStep, ::ComponentPriors) = step.prior
_resolve_step_prior(step::LinearPhaseStep{<:Any, Nothing}, cp::ComponentPriors) =
    get(cp.priors, step.component, NoPrior())

# Gauge that pins the reference antenna AND holds the named components fixed.
function _freeze_gauge(plan::GainPlan, ref_ant::Integer, frozen::Tuple)
    idx = _fixed_indices(ReferenceAntenna(ref_ant), plan)
    for name in frozen
        append!(idx, _component_flat_indices(plan, name))
    end
    return FixParams(sort!(unique!(idx)))
end

# The parameter-block names available to freeze — the model's component names as
# they appear in the parameter ComponentVector (`p.g<k>.<name>`). Antenna groups
# are an internal detail, so a name is user-facing across all groups it appears in.
function _param_block_names(plan::GainPlan)
    names = Symbol[]
    for gp in plan.groups
        append!(names, propertynames(_sub(plan.template, gp.groupval)))
    end
    return unique!(names)
end

# Flat `p` indices of the named parameter block, resolved through the
# ComponentVector's OWN named structure (`propertynames`/`getproperty`), so it
# targets the real parameters — and errors on an unknown name (catches typos).
function _component_flat_indices(plan::GainPlan, name::Symbol)
    name in _param_block_names(plan) || error(
        "GradientStep frozen: no parameter block named `$name` — the model's blocks are " *
            "$(_param_block_names(plan)).",
    )
    mask = similar(plan.template, Bool)
    fill!(mask, false)
    for gp in plan.groups
        pg = _sub(mask, gp.groupval)
        name in propertynames(pg) && (getproperty(pg, name) .= true)
    end
    return findall(flatten(mask))
end
