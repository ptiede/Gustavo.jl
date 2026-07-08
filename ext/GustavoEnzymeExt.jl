# Enzyme reverse-mode gradient for the Gustavo.Solve forward-model fringe fitter.
# Implements the `leaf_value_and_grad` stub declared in `Gustavo.Solve`: the
# per-leaf value + structured gradient of the coherency-matrix log-likelihood
# with respect to the per-site parameter `ComponentVector`.
#
# Kept in an extension because an Enzyme reverse pass is a heavy compile
# dependency; the core `Gustavo.Solve` load path stays Enzyme-free. `set_runtime_
# activity` handles the mixed activity through the complex `cis`/`exp` gain map
# and the data-dependent branches in `leaf_loglik_gains` (all on `Const` data —
# weights/finiteness — never on the differentiated gains).
module GustavoEnzymeExt

using Enzyme
using ComponentArrays: ComponentVector
import Gustavo.Solve: leaf_value_and_grad, _leaf_loglik, GainPlan

function leaf_value_and_grad(plan::GainPlan, p::ComponentVector, ctx, S)
    dp = zero(p)                                  # structured gradient shadow
    _, value = Enzyme.autodiff(
        set_runtime_activity(ReverseWithPrimal),
        _leaf_loglik, Active,
        Const(plan), Duplicated(p, dp), Const(ctx), Const(S),
    )
    return value, dp
end

end # module GustavoEnzymeExt
