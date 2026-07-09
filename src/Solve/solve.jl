# ── FringeSolution + fringe_solve driver (Stage 2, Milestone 5) ──────────────
#
# `fringe_solve` is the top-level driver: build the geometry, plan the per-site
# gain model, warm-start, then minimize the forward-model MAP objective with a
# pluggable SOLVE STRATEGY (the objective and the strategy are decoupled — the
# same `FringePosterior` can be minimized by gradient LBFGSFit today or block-ALS
# later). The result is a `FringeSolution` the data is calibrated by via
# `apply_calibration` (reusing the pure `evaluate_gains` forward map).

using ..UVData: with_visibilities, materialize_leaf, pol_products, branches
import ..UVData: apply_calibration
import ..Calibration: _apply_gain_kernel

"""
    FringeSolution(model, plan, geom, p, info)

A solved forward-model gain fit: the `ArrayGainModel`, its [`GainPlan`](@ref) over
`geom`, the solved per-site parameter `ComponentVector` `p`, and a NamedTuple of
solver `info`. Calibrate data with `apply_calibration(uvset, sol)`.
"""
struct FringeSolution{M, PL <: GainPlan, G <: DataGeometry, CV}
    model::M
    plan::PL
    geom::G
    p::CV
    info::NamedTuple
end

# ── Optimizer seam ───────────────────────────────────────────────────────────
#
# The objective is a `LogDensityProblems` posterior, so the solver takes ANY
# Optimization.jl optimizer and hands it to `solve` (mirrors Comrade's
# `comrade_opt(post, opt)`) — no bespoke per-optimizer type. `_optimize_map` has
# no core methods; `GustavoOptimizationExt` (load `Optimization` +
# `OptimizationOptimJL`) adds the one that builds the `OptimizationProblem` and
# calls `solve(prob, optimizer)`. `optimizer = nothing` lets the ext pick a
# sensible default (LBFGS). A future block-coordinate/ALS solver would be a
# separate driver, not an "optimizer" for `solve`.
function _optimize_map end

# ── Driver ───────────────────────────────────────────────────────────────────

# Number of antennas = highest antenna index appearing on any baseline.
function _nant(uvset)
    n = 0
    for (_, leaf) in branches(uvset)
        for (a, b) in UVData.baselines(leaf).pairs
            n = max(n, Int(a), Int(b))
        end
    end
    return n
end

"""
    fringe_solve(uvset, model; optimizer = nothing, source = ProfiledPointSource(),
                 warmstart = nothing, ref_ant = 1, executor = SerialExecutor(),
                 maxiters = 1000, f0 = nothing, t0 = nothing, solve_kwargs...)
        -> FringeSolution

Fit `model` (an `ArrayGainModel` or a bare `StationGainModel`) to `uvset` by
minimizing the forward-model MAP objective with `optimizer` — ANY Optimization.jl
optimizer (e.g. `OptimizationOptimJL.LBFGS()`), or `nothing` for the extension's
default (LBFGS). Requires `GustavoOptimizationExt` (load `Optimization` +
`OptimizationOptimJL`). `source` defaults to a profiled point source (generic,
source-agnostic, no absolute amplitude scale); `warmstart` is a `ComponentVector`
(defaults to zeros); extra `solve_kwargs` pass through to `solve`.
"""
function fringe_solve(
        uvset, model;
        optimizer = nothing,
        source::AbstractSourceModel = ProfiledPointSource(),
        warmstart = nothing,
        ref_ant::Integer = 1,
        executor = SerialExecutor(),
        maxiters::Integer = 1000,
        f0 = nothing, t0 = nothing,
        solve_kwargs...,
    )
    isempty(methods(_optimize_map)) && error(
        "fringe_solve needs the Optimization extension — run " *
            "`using Optimization, OptimizationOptimJL`.",
    )
    geom = build_geometry(uvset; f0 = f0, t0 = t0)
    am = model isa ArrayGainModel ? model : ArrayGainModel(model)
    nant = _nant(uvset)
    plan = plan_gains(am, nant, geom)
    p0 = warmstart === nothing ? zero_params(plan) : warmstart
    post = FringePosterior(plan, uvset, geom; source = source, executor = executor)
    p, sinfo = _optimize_map(post, p0, optimizer; maxiters = Int(maxiters), solve_kwargs...)
    info = merge((; ref_ant = Int(ref_ant), nant = nant), sinfo)
    return FringeSolution(am, plan, geom, p, info)
end

# ── Applying the solution ─────────────────────────────────────────────────────

"""
    apply_calibration(uvset, sol::FringeSolution; ntasks = nthreads()) -> UVSet

Divide every leaf's visibilities by the solved per-antenna gains
(`V_corr = V / (gₐ·conj(g_b))`, `W_corr = W·|gₐg_b|²`), reusing the pure
`evaluate_gains(plan, p, …)` forward map. Samples where a gain magnitude
underflows are flagged (weight 0, vis NaN).
"""
function apply_calibration(
        uvset::UVSet, sol::FringeSolution;
        ntasks::Integer = Threads.nthreads(),
    )
    return UVData.apply(uvset) do leaf, info, root
        leaf = materialize_leaf(leaf; layers = (:vis, :weights, :uvw))
        ci, ti = leaf_window(sol.geom, leaf)
        g = evaluate_gains(sol.plan, sol.p, ci, ti)
        bl_pairs = UVData.baselines(leaf).pairs
        pols = pol_products(leaf)
        vis_corr, w_corr = _apply_gain_kernel(leaf[:vis], leaf[:weights], g, bl_pairs, pols; ntasks = ntasks)
        return with_visibilities(leaf, vis_corr, w_corr)
    end
end
