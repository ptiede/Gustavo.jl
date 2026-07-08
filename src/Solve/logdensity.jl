# ── Per-leaf value + gradient (Stage 2, Milestone 3) ─────────────────────────
#
# `leaf_value_and_grad(plan, p, ctx, S)` reverse-differentiates the per-leaf
# log-likelihood `_leaf_loglik` with respect to the per-site parameters `p`,
# returning `(value, grad)` where `grad::ComponentVector` shares `p`'s layout.
# The gradient is STRUCTURALLY SPARSE: only the time/frequency segments the
# leaf's window (`ctx.chan_idx`/`ctx.ti_idx`) touches are nonzero.
#
# Enzyme cannot differentiate across a Dagger boundary, so the design is that
# each leaf self-differentiates INSIDE its own Dagger task (all of
# `plan`/`ctx`/`S` captured `Const`); `Graph.pmapreduce` then sums the per-leaf
# `(value, grad)` over the processing set with
# `((v1, g1), (v2, g2)) -> (v1 + v2, g1 .+ g2)` (ComponentVector addition), and
# the driver adds `logprior` once. The reverse-mode method itself lives in
# `GustavoEnzymeExt` (an Enzyme forward/reverse pass is a heavy compile
# dependency, kept out of the core load path).

"""
    leaf_value_and_grad(plan::GainPlan, p, ctx, S) -> (value, grad)

Reverse-mode value and structured gradient of the per-leaf log-likelihood
[`leaf_loglik`](@ref)`(plan, p, ctx; S)` with respect to the per-site parameter
`ComponentVector` `p`. `grad` shares `p`'s axes and is structurally sparse (zero
outside the segments the leaf window touches).

Implemented in `GustavoEnzymeExt`; run `using Enzyme` to enable it.
"""
function leaf_value_and_grad(plan::GainPlan, p, ctx, S)
    return error(
        "leaf_value_and_grad requires the Enzyme extension — run `using Enzyme` " *
            "to load GustavoEnzymeExt",
    )
end

# ── Distributed objective over the processing set (Stage 2, Milestone 4) ──────
#
# The total forward-model log-likelihood is a sum over the `UVSet`'s partitions,
# evaluated on the `Gustavo.Graph` (Dagger) substrate: each leaf materializes,
# builds its AD-inactive context, and computes its own `(value, gradient)` INSIDE
# its own task (Enzyme can't cross the Dagger boundary); `pmapreduce` sums them.
# Because the reduce is order-deterministic, `SerialExecutor` and `DaggerExecutor`
# give bit-identical results. The value-only path needs no Enzyme.

# Per-leaf node functions for `pmap`/`pmapreduce`. `params = (plan, p, S, flux)`;
# `coords.root` carries the geometry (threaded through as the map's `params` would
# not know it) — we pass geometry in `params` too.
function _leaf_value_node(data, coords, params)
    plan, p, geom, S, flux = params
    ctx = build_leaf_ctx(geom, data)
    Smat = S === nothing ? point_source_coherency(length(ctx.bl_a); flux = flux) : S
    return _leaf_loglik(plan, p, ctx, Smat)
end
function _leaf_valuegrad_node(data, coords, params)
    plan, p, geom, S, flux = params
    ctx = build_leaf_ctx(geom, data)
    Smat = S === nothing ? point_source_coherency(length(ctx.bl_a); flux = flux) : S
    return leaf_value_and_grad(plan, p, ctx, Smat)
end

# Reduce (value, structured-gradient) pairs: sum values, add ComponentVectors.
@inline _vg_add((v1, g1), (v2, g2)) = (v1 + v2, g1 + g2)

"""
    fringe_objective(plan, p, uvset, geom; S = nothing, flux = 1.0,
                     executor = SerialExecutor()) -> Real

Total forward-model log-likelihood `Σ_leaf leaf_loglik` over the partitions of
`uvset`, on the Graph substrate. Value only — no Enzyme required.
"""
function fringe_objective(
        plan::GainPlan, p, uvset, geom;
        S = nothing, flux::Real = 1.0, executor = SerialExecutor(),
    )
    return pmapreduce(
        _leaf_value_node, +, uvset;
        executor = executor, params = (plan, p, geom, S, Float64(flux)),
        init = zero(float(eltype(p))),
    )
end

"""
    fringe_objective_and_grad(plan, p, uvset, geom; S = nothing, flux = 1.0,
                              executor = SerialExecutor()) -> (value, grad)

Total forward-model log-likelihood AND its gradient wrt `p`, summed over the
partitions of `uvset` on the Graph substrate. Each leaf self-differentiates in
its own task ([`leaf_value_and_grad`](@ref), Enzyme) and `pmapreduce` sums
`(value, grad)`. `grad` is a `ComponentVector` sharing `p`'s layout. Serial and
Dagger executors give bit-identical results. Requires the Enzyme extension.
"""
function fringe_objective_and_grad(
        plan::GainPlan, p, uvset, geom;
        S = nothing, flux::Real = 1.0, executor = SerialExecutor(),
    )
    return pmapreduce(
        _leaf_valuegrad_node, _vg_add, uvset;
        executor = executor, params = (plan, p, geom, S, Float64(flux)),
        init = (zero(float(eltype(p))), zero(p)),
    )
end

# ── LogDensityProblems interface (order-1) ────────────────────────────────────

"""
    FringePosterior(plan, uvset, geom; S = nothing, flux = 1.0,
                    executor = SerialExecutor())

The forward-model fringe-fit log-density over the flat parameter vector, wrapping
the distributed objective as a `LogDensityProblems` problem (order 1). `logdensity`
returns the total log-likelihood; `logdensity_and_gradient` returns it with the
flat gradient. (v1 = likelihood only; the `logprior` layer is Milestone 7.)
"""
struct FringePosterior{P <: GainPlan, U, G, S, E}
    plan::P
    uvset::U
    geom::G
    source::S
    flux::Float64
    executor::E
end
function FringePosterior(
        plan::GainPlan, uvset, geom;
        S = nothing, flux::Real = 1.0, executor = SerialExecutor(),
    )
    return FringePosterior(plan, uvset, geom, S, Float64(flux), executor)
end

LogDensityProblems.dimension(post::FringePosterior) = nparameters(post.plan)
LogDensityProblems.capabilities(::Type{<:FringePosterior}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity(post::FringePosterior, x::AbstractVector)
    p = unflatten(post.plan, x)
    return fringe_objective(
        post.plan, p, post.uvset, post.geom;
        S = post.source, flux = post.flux, executor = post.executor,
    )
end

function LogDensityProblems.logdensity_and_gradient(post::FringePosterior, x::AbstractVector)
    p = unflatten(post.plan, x)
    v, g = fringe_objective_and_grad(
        post.plan, p, post.uvset, post.geom;
        S = post.source, flux = post.flux, executor = post.executor,
    )
    return v, flatten(g)
end
