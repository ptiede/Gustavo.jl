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
