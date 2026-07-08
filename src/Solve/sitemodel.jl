# ── Per-array gain model (Stage 2, Milestone 2) ──────────────────────────────
#
# `ArrayGainModel` is the whole-array instrument model: one DEFAULT
# `StationGainModel` applied to every antenna, plus optional PER-SITE OVERRIDES
# for antennas that need a different structure (an extra dTEC term, a finer
# bandpass, …). This mirrors Comrade's `ArrayPrior` (a default + a handful of
# site overrides) — the common case is one model for the whole array, so the
# override dict is usually empty.
#
# The key downstream property: antennas are later GROUPED BY IDENTICAL MODEL
# STRUCTURE (`==`) so the forward map evaluates each group under one concrete,
# compile-time `StationGainModel` type — heterogeneous arrays stay type-stable
# without one specialization per antenna (see `plan_gains`).

"""
    ArrayGainModel(default; overrides = Dict{Int,StationGainModel}())

The gain model for a whole array: a `default` [`StationGainModel`](@ref) used for
every antenna, with `overrides` mapping a 1-based antenna index to a replacement
`StationGainModel` for that antenna only. Antennas absent from `overrides` use
`default`.

A bare `StationGainModel` (`ArrayGainModel(default)`) is a homogeneous array;
pass a `Dict` of antenna-index ⇒ model for overrides
(`ArrayGainModel(default, Dict(3 => other_model))`).
"""
struct ArrayGainModel{D <: StationGainModel}
    default::D
    overrides::Dict{Int, StationGainModel}
end

ArrayGainModel(default::StationGainModel) =
    ArrayGainModel(default, Dict{Int, StationGainModel}())
function ArrayGainModel(default::StationGainModel, overrides::AbstractDict)
    ov = Dict{Int, StationGainModel}()
    for (k, v) in overrides
        ov[Int(k)] = v
    end
    return ArrayGainModel(default, ov)
end

"""
    site_model(am::ArrayGainModel, ant::Integer) -> StationGainModel

The `StationGainModel` antenna `ant` (1-based) is fit under: its override if one
is registered, else the array default.
"""
site_model(am::ArrayGainModel, ant::Integer) = get(am.overrides, Int(ant), am.default)

# Group 1..nant into runs of antennas sharing an identical (`==`) model. Returns
# `(models, ants)` where `models[g]` is the group's `StationGainModel` and
# `ants[g]` the (ascending) global antenna indices in that group. Equality of the
# value-typed model structs means one parameter-array layout serves the group.
function _group_by_model(am::ArrayGainModel, nant::Integer)
    models = StationGainModel[]
    ants = Vector{Int}[]
    for a in 1:Int(nant)
        m = site_model(am, a)
        idx = findfirst(==(m), models)
        if idx === nothing
            push!(models, m)
            push!(ants, [a])
        else
            push!(ants[idx], a)
        end
    end
    return models, ants
end
