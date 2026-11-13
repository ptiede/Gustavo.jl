# ── Station gains ────────────────────────────────────────────────────────────
#
# A joint solve fits parameters, the entries of its components; the data see
# only the station gains, one log-gain per observable at each (station, feed,
# data cell), the sum of the components' terms there. Two cells share a station
# gain where every component reads the same parameters at the same
# coordinates, and the station-gain values depend on the parameters through the
# sparse Jacobian `J`, built by sparse forward-mode differentiation through the
# model layer, so any term gets its columns from its `term_eval`.

"""
    StationGains(components, nant, nfeed, cover)

The station gains of a joint solve over the data entries `cover`
(`cover[si, c] = (t, ch)`, a time and a channel index of the plans): one per
station and set of (feed, entry) pairs at which every fitted component reads
the same parameters at the same coordinates. `components` is
`(; phase, logamp)`, each a vector of the observable's fitted components, each
the station blocks `(; antennas, θ, plan)` that [`antenna_blocks`](@ref)
returns.

- `index[a, p, si, c]`: the station gain of station `a` and feed `p` at entry
  `(si, c)`; `0` where no component reaches station `a` and feed `p`.
- `at[n]`: station gain `n`'s station, feed and one of its entries `(t, ch)`.
- `params`: per observable, the parameters its components fit at these
  entries, each `(component, block, index)` into that block's `θ`.
- `J`: per observable, the sparse Jacobian of its station-gain values
  (`_station_gain_values`) in its `params`.
"""
struct StationGains{I <: AbstractArray{<:Integer, 4}, A <: AbstractVector, P <: NamedTuple, M <: NamedTuple}
    index::I
    at::A
    params::P
    J::M
end

function StationGains(components, nant, nfeed, cover)
    locs = map(cs -> [_block_locations(c, nant) for c in cs], components)
    touched = map(cs -> [[Set{NTuple{4, Int}}() for _ in c] for c in cs], components)
    index = zeros(Int, nant, nfeed, size(cover)...)
    at = Tuple{Int, Int, eltype(cover)}[]
    for a in 1:nant
        seen = Dict{Any, Int}()
        for p in 1:nfeed, I in CartesianIndices(cover)
            t, ch = cover[I]
            key = map((cs, ls, ts) -> _gain_key!(ts, cs, ls, a, p, t, ch), components, locs, touched)
            all(k -> all(isnothing, k), key) && continue
            index[a, p, I] = get!(seen, key) do
                push!(at, (a, p, cover[I]))
                length(at)
            end
        end
    end
    params = map(components, touched) do cs, ts
        reduce(vcat, [_component_params(c, ts[k], k) for (k, c) in pairs(cs)]; init = NTuple{3, Int}[])
    end
    J = map((cs, ps, ls) -> _gain_jacobian(cs, ps, ls, at), components, params, locs)
    return StationGains(index, at, params, J)
end

# What each of `components` reads at station `a`, feed `p` and entry `(t, ch)`:
# its feed node, segments and coordinates, or `nothing` where it does not reach
# that station and feed. Marks the parameter blocks read in `touched`.
function _gain_key!(touched, components, locs, a, p, t, ch)
    return map(eachindex(components)) do k
        j, ai = locs[k][a]
        iszero(j) && return nothing
        plan = components[k][j].plan
        node = _feed_node(plan.tying, p)
        iszero(node) && return nothing
        fs, ts = plan.fseg_id[ch], plan.tseg_id[t]
        push!(touched[k][j], (ai, node, ts, fs))
        return (node, fs, ts, Calibration._cell_coordinates(plan.term, plan, t, ch))
    end
end

# The parameters of component `k` in its blocks' parameter blocks `touched`,
# `(ai, node, ts, fs)` per block, as `(k, block, index)` in station, feed-node
# and segment order.
function _component_params(blocks, touched, k)
    return [
        (k, j, i) for (j, b) in pairs(blocks) for (ai, node, ts, fs) in sort!(collect(touched[j]))
            for i in view(LinearIndices(b.θ), 1:nparams_per_block(b.plan.term, b.plan.nchan_seg[fs]), node, fs, ts, ai)
    ]
end

# The values of `params` in their blocks' θ.
_param_values(components, params) = [components[k][j].θ[i] for (k, j, i) in params]

# Writes `values` into the blocks' θ at `params`.
function _write_params!(components, params, values)
    for n in eachindex(params, values)
        k, j, i = params[n]
        components[k][j].θ[i] = values[n]
    end
    return components
end

# Each block's θ in the element type of `values`, with `values` at `params`.
function _leaves_with(components, params, values)
    leaves = [[copyto!(similar(b.θ, eltype(values)), b.θ) for b in c] for c in components]
    for n in eachindex(params, values)
        k, j, i = params[n]
        leaves[k][j][i] = values[n]
    end
    return leaves
end

# The station-gain values of one observable, into `x`: at each station gain
# `at[n]`, the sum of `components`' terms from their blocks' parameters
# `leaves`.
function _station_gain_values!(x, components, leaves, locs, at)
    fill!(x, zero(eltype(x)))
    for n in eachindex(x, at)
        a, p, (t, ch) = at[n]
        for k in eachindex(components)
            j, ai = locs[k][a]
            iszero(j) || (x[n] += Calibration._leaf_value(components[k][j].plan, leaves[k][j], ai, p, t, ch))
        end
    end
    return x
end

# The Jacobian of one observable's station-gain values in its `params`, by
# sparse forward-mode differentiation: the pattern is traced without values and
# colored once, then one forward pass per color.
function _gain_jacobian(components, params, locs, at)
    isempty(params) && return spzeros(length(at), 0)
    f(values) = _station_gain_values!(similar(values, length(at)), components, _leaves_with(components, params, values), locs, at)
    backend = AutoSparse(
        AutoForwardDiff(); sparsity_detector = TracerSparsityDetector(), coloring_algorithm = GreedyColoringAlgorithm(),
    )
    values = _param_values(components, params)
    return jacobian(f, prepare_jacobian(f, backend, values), backend, values)
end
