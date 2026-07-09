# Optimization.jl solve for the Gustavo.Solve forward-model fringe fitter.
# Implements the `_optimize_map` seam declared in `Gustavo.Solve`: wrap the
# `FringePosterior` (a LogDensityProblems order-1 problem) in an
# `OptimizationProblem` and minimize `-logdensity` with ANY Optimization.jl
# optimizer (default LBFGS), using the Enzyme gradient.
#
# Kept in an extension because Optimization + OptimizationOptimJL are a heavy
# compile dependency; the core `Gustavo.Solve` load path stays free of them.
module GustavoOptimizationExt

using Optimization: OptimizationFunction, OptimizationProblem, solve
import OptimizationOptimJL
import LogDensityProblems as LDP
import Gustavo.Solve: _optimize_map, flatten, unflatten

function _optimize_map(post, p0, optimizer; maxiters::Integer = 1000, solve_kwargs...)
    x0 = collect(flatten(p0))
    # Minimize the negative log-density; gradient from the order-1 posterior.
    f = OptimizationFunction(
        (x, _p) -> -LDP.logdensity(post, x);
        grad = (G, x, _p) -> begin
            _, g = LDP.logdensity_and_gradient(post, x)
            G .= .-g
            return G
        end,
    )
    prob = OptimizationProblem(f, x0)
    opt = optimizer === nothing ? OptimizationOptimJL.LBFGS() : optimizer
    sol = solve(prob, opt; maxiters = maxiters, solve_kwargs...)
    p = unflatten(post.plan, collect(sol.u))
    info = (; final_objective = -Float64(sol.objective), retcode = Symbol(sol.retcode))
    return p, info
end

end # module GustavoOptimizationExt
