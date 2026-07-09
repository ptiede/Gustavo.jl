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

# Resolve the `warmstart` option into a starting parameter `ComponentVector`:
# `nothing` → zeros; `:fft`/`:auto` → the FFT search + stationization seed; an
# explicit `ComponentVector` is passed through unchanged.
_resolve_warmstart(::Nothing, plan, uvset, geom; kw...) = zero_params(plan)
function _resolve_warmstart(
        w::Symbol, plan, uvset, geom;
        ref_ant, search, stationization,
    )
    (w === :fft || w === :auto) || error(
        "warmstart symbol must be :fft or :auto (got :$w); pass a ComponentVector " *
            "or nothing otherwise",
    )
    return fft_warmstart(
        plan, uvset, geom;
        ref_ant = ref_ant, search = search, stationization = stationization,
    )
end
_resolve_warmstart(w, plan, uvset, geom; kw...) = w

"""
    fringe_solve(uvset, model; optimizer = nothing, source = ProfiledPointSource(),
                 gauge = ReferenceAntenna(ref_ant), scaling = AutoScale(),
                 warmstart = nothing, ref_ant = 1, executor = SerialExecutor(),
                 maxiters = 1000, f0 = nothing, t0 = nothing, solve_kwargs...)
        -> FringeSolution

Fit `model` (an `ArrayGainModel` or a bare `StationGainModel`) to `uvset` by
minimizing the forward-model MAP objective with `optimizer` — ANY Optimization.jl
optimizer (e.g. `OptimizationOptimJL.LBFGS()`), or `nothing` for the extension's
default (LBFGS). Requires `GustavoOptimizationExt` (load `Optimization` +
`OptimizationOptimJL`).

The optimizer runs on a REPARAMETERIZED problem: `gauge` fixes the unobservable
gauge (default [`ReferenceAntenna`](@ref)`(ref_ant)` — pins the refant's phase and
amplitude), and `scaling` conditions the variables (default [`AutoScale`](@ref)).
Both are user-specifiable. `source` defaults to a profiled point source (generic,
source-agnostic, no absolute amplitude scale); extra `solve_kwargs` pass through
to `solve`.

`warmstart` selects the optimizer's start: `nothing` (zeros), `:fft`/`:auto` (the
FFT matched-filter search + closure stationization seed — [`fft_warmstart`](@ref),
which resolves the delay/rate non-convexity a zero start cannot escape), or an
explicit `ComponentVector`. `search`/`stationization` configure the two warm-start
stages (used only for `:fft`/`:auto`).
"""
function fringe_solve(
        uvset, model;
        optimizer = nothing,
        source::AbstractSourceModel = ProfiledPointSource(),
        gauge::Union{AbstractGauge, Nothing} = nothing,
        scaling::AbstractScaling = AutoScale(),
        prior::ComponentPriors = ComponentPriors(),
        warmstart = nothing,
        search::FringeSearch = FringeSearch(),
        stationization::Stationization = Stationization(),
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
    p0 = _resolve_warmstart(
        warmstart, plan, uvset, geom;
        ref_ant = ref_ant, search = search, stationization = stationization,
    )
    post = FringePosterior(plan, uvset, geom; source = source, prior = prior, executor = executor)

    gge = gauge === nothing ? ReferenceAntenna(ref_ant) : gauge
    reparam = build_reparam(plan, gge, scaling, p0)
    rpost = ReparamPosterior(post, reparam)
    y0 = to_free(reparam, p0)
    u, sinfo = _optimize_map(rpost, collect(y0), optimizer; maxiters = Int(maxiters), solve_kwargs...)
    p = to_full(reparam, u)

    info = merge((; ref_ant = Int(ref_ant), nant = nant, nfree = nfree(reparam)), sinfo)
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
