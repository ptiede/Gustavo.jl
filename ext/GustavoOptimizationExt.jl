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
import Gustavo.Solve: _optimize_map

# Minimize `-logdensity(post)` over the flat vector `x0` with `optimizer` (any
# Optimization.jl optimizer; default LBFGS). `post` is any order-1
# LogDensityProblems problem; the caller handles the parameterization. Returns
# `(u, info)` — the optimized vector and solver diagnostics.
function _optimize_map(post, x0, optimizer; maxiters::Integer = 1000, solve_kwargs...)
    f = OptimizationFunction(
        (x, _p) -> -LDP.logdensity(post, x);
        grad = (G, x, _p) -> begin
            _, g = LDP.logdensity_and_gradient(post, x)
            G .= .-g
            return G
        end,
    )
    prob = OptimizationProblem(f, collect(x0))
    opt = optimizer === nothing ? OptimizationOptimJL.LBFGS() : optimizer
    sol = solve(prob, opt; maxiters = maxiters, solve_kwargs...)
    info = (; final_objective = -Float64(sol.objective), retcode = Symbol(sol.retcode))
    return collect(sol.u), info
end

end # module GustavoOptimizationExt
