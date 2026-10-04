# Whole-set functions on an MSv4 `ProcessingSet`, answered by walking its
# Measurement Sets through the per-partition accessors.

"""
    leaves(ps::XRadio.ProcessingSet)

Iterator over `key => MeasurementSet` pairs of `ps`, in insertion order.
"""
leaves(ps::XRadio.ProcessingSet) = pairs(ps)

"""
    union_frequency_axis(ps::XRadio.ProcessingSet) -> Vector{FrequencySetup}

Vector of `FrequencySetup`s spanning every partition, deduplicated by `==`/`hash`,
in first-seen order, like xradio's `ProcessingSet.get_freq_axis()`.
"""
function union_frequency_axis(data::XRadio.ProcessingSet)
    out = FrequencySetup[]
    seen = Set{FrequencySetup}()
    for (_, leaf) in leaves(data)
        fs = freq_setup(leaf)
        if !(fs in seen)
            push!(out, fs)
            push!(seen, fs)
        end
    end
    return out
end

"""
    freq_setup(ps::XRadio.ProcessingSet) -> FrequencySetup

Single-SPW shorthand: returns the unique `FrequencySetup` if every partition
shares one, otherwise throws `ArgumentError`. Multi-SPW callers should
use `union_frequency_axis(data)` or read each partition's `freq_setup`
individually.
"""
function freq_setup(data::XRadio.ProcessingSet)
    setups = union_frequency_axis(data)
    n = length(setups)
    n == 1 && return setups[1]
    n == 0 && throw(ArgumentError("the set has no partitions; no frequency setup"))
    throw(
        ArgumentError(
            "the set has $(n) distinct frequency setups; " *
                "use union_frequency_axis(data) or freq_setup(partition)",
        )
    )
end

"""
    union_antennas(ps::XRadio.ProcessingSet) -> AntennaTable

Walk partitions and union their antennas by name, in first-seen order. Errors
if the same antenna name has different metadata (mount, station_xyz,
nominal_basis, pol_angles) across partitions — a multi-track observation that
should be split and processed per-SPW. An `extras` column is kept when every
partition's table has it, taking each antenna's value from the table it was
first seen in.
"""
function union_antennas(data::XRadio.ProcessingSet)
    tables = [antennas(leaf) for (_, leaf) in leaves(data)]
    isempty(tables) && error("union_antennas: the set has no partitions")
    template = first(tables)
    all(t -> t === template, tables) && return template
    rows = eltype(getfield(template, :antennas))[]
    origin = Tuple{Int, Int}[]
    slot = Dict{String, Int}()
    for (ti, tab) in pairs(tables), (ai, ant) in pairs(getfield(tab, :antennas))
        idx = get(slot, ant.name, 0)
        if idx == 0
            push!(rows, ant)
            push!(origin, (ti, ai))
            slot[ant.name] = length(rows)
        else
            isequal(rows[idx], ant) || error(
                "union_antennas: antenna '$(ant.name)' has " *
                    "inconsistent metadata across partitions; split the set " *
                    "and process per-SPW.",
            )
        end
    end
    common = filter(k -> all(t -> haskey(extras(t), k), tables), keys(extras(template)))
    ext = NamedTuple{common}(
        Tuple([extras(tables[ti])[k][ai] for (ti, ai) in origin] for k in common)
    )
    return AntennaTable(StructArray(rows), array_name(template), ext)
end

"""
    union_pol_products(ps::XRadio.ProcessingSet) -> Vector{String}

Pol product set shared across partitions. Errors if they disagree —
mirrors `union_antennas` for the polarization axis.
"""
function union_pol_products(data::XRadio.ProcessingSet)
    isempty(leaves(data)) && error("union_pol_products: the set has no partitions")
    first_pp = pol_products(last(first(leaves(data))))
    for (_, leaf) in leaves(data)
        Set(pol_products(leaf)) == Set(first_pp) ||
            error(
            "union_pol_products: leaves have different pol product sets; " *
                "got $first_pp vs $(pol_products(leaf)).",
        )
    end
    return first_pp
end

nchannels(data::XRadio.ProcessingSet) = nchannels(freq_setup(data))
