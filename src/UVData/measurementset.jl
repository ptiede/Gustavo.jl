# The per-partition accessors, answered from an MSv4 `MeasurementSet`.
#
# Each is rebuilt from the store on every call.

"""
    feed_pairs(ms::XRadio.MeasurementSet) -> Matrix{Tuple{Int, Int}}

The `(feed_a, feed_b)` pair each stored product relates on each baseline,
indexed `[product, baseline]` in `ms`'s own order. A product names one receptor
of each antenna, resolved against that antenna's `polarization_type`: the
letter's position there, or, for an antenna whose receptors are all of the
other basis (`X`/`Y` for an `R`/`L` letter, or the reverse), the letter's
position in its own basis (`R`, `X` first; `L`, `Y` second). Any other letter
throws. The same stored product can so relate different feed pairs on
different baselines.
"""
function feed_pairs(ms::XRadio.MeasurementSet)
    types = XRadio.polarization_types(ms)
    names = XRadio.antennas(ms)
    receptors = [String.(collect(types[:, a])) for a in axes(types, 2)]
    products = XRadio.polarizations(ms)
    for p in products
        length(p) == 2 || throw(ArgumentError("a product label has two receptors, got \"$p\""))
    end
    return [
        (_receptor_feed(p[1], receptors[a], names[a]), _receptor_feed(p[2], receptors[b], names[b]))
            for p in products, (a, b) in baselines(ms).pairs
    ]
end

const _BASES = (("R", "L"), ("X", "Y"))

function _receptor_feed(letter::Char, receptors, antenna)
    r = string(letter)
    i = findfirst(==(r), receptors)
    i === nothing || return i
    own = findfirst(basis -> r in basis, _BASES)
    other = own === nothing ? nothing : _BASES[3 - own]
    (other !== nothing && all(in(other), receptors)) && return findfirst(==(r), _BASES[own])
    throw(
        ArgumentError(
            "product letter `$r` names no receptor of antenna `$antenna`, whose receptors are $(join(receptors, ", "))"
        )
    )
end

# The shared feed-pair order of a solver cube, and for each baseline the stored
# product holding each pair: `perm[k, bi]` is the product of baseline `bi` that
# relates `order[k]`. Every baseline must relate each pair exactly once.
function _feed_permutation(pairs::AbstractMatrix{Tuple{Int, Int}})
    order = sort!(unique(vec(pairs)))
    perm = similar(pairs, Int, (eachindex(order), axes(pairs, 2)))
    for bi in axes(pairs, 2)
        col = view(pairs, :, bi)
        (allunique(col) && length(col) == length(order)) || throw(
            ArgumentError(
                "baseline $bi relates feed pairs $(collect(col)); every baseline must relate each of $order once"
            )
        )
        for k in eachindex(order)
            perm[k, bi] = findfirst(==(order[k]), col)
        end
    end
    return order, perm
end

"""
    baselines(ms::XRadio.MeasurementSet) -> BaselineIndex

The baselines of `ms` in `baseline_id` order, their antenna indices counting
into [`antennas(ms)`](@ref antennas(::XRadio.MeasurementSet)).
"""
baselines(ms::XRadio.MeasurementSet) =
    _baselines_into(ms, XRadio.antennas(ms), "the antenna dataset")

"""
    baselines(ms::XRadio.MeasurementSet, stations::AntennaTable) -> BaselineIndex

The baselines of `ms` in `baseline_id` order, their antenna indices counting
into `stations`, matched by name. This is how Measurement Sets that saw
different sub-arrays share one station axis: MSv4 names each baseline's
antennas, and `stations` fixes the numbering.
"""
baselines(ms::XRadio.MeasurementSet, stations::AntennaTable) =
    _baselines_into(ms, collect(String.(stations.name)), "the station table")

function _baselines_into(ms, names, what)
    slot = Dict(n => i for (i, n) in pairs(names))
    index(n) = get(slot, n) do
        throw(
            ArgumentError(
                "baseline antenna `$n` is not in $what, which names " * join(names, ", ")
            )
        )
    end
    pairs_v = [(index(a), index(b)) for (a, b) in XRadio.baselines(ms)]
    return BaselineIndex(pairs_v, pairs_v; antenna_names = names)
end

const _POL_TYPES = Dict("R" => RPol(), "L" => LPol(), "X" => XPol(), "Y" => YPol())

"""
    antennas(ms::XRadio.MeasurementSet) -> AntennaTable

The antenna table of `ms`, from its antenna dataset: geocentric positions,
mounts, receptor polarizations and angles, and dish diameters where stated
(under `extras(tab).DIAMETER`). An unstated receptor angle is `NaN`.
"""
function antennas(ms::XRadio.MeasurementSet)
    haskey(branches(ms), :antenna) ||
        throw(ArgumentError("this Measurement Set has no antenna dataset"))
    xds = branches(ms)[:antenna]
    names = XRadio.antennas(ms)
    positions = XRadio.antenna_positions(ms)
    mounts = XRadio.mounts(ms)
    types = XRadio.polarization_types(ms)
    angles = XRadio.receptor_angles(ms)
    size(types, 1) == 2 || throw(
        ArgumentError(
            "each antenna has $(size(types, 1)) receptors; an `Antenna` describes two"
        )
    )
    polarization(label) = get(_POL_TYPES, label) do
        throw(ArgumentError("receptor polarization `$label` is not one of R, L, X, Y"))
    end
    ants = [
        Antenna(;
                name = String(n),
                station_xyz = collect(positions[:, a]),
                mount = mounts[a],
                nominal_basis = (polarization(types[1, a]), polarization(types[2, a])),
                pol_angles = (angles[1, a], angles[2, a]),
            ) for (a, n) in zip(axes(positions, 2), names)
    ]
    ext = haskey(xds, :antenna_dish_diameter) ?
        (; DIAMETER = collect(xds[:antenna_dish_diameter])) : NamedTuple()
    array = get(DimensionalData.metadata(xds), :overall_telescope_name, "")
    return AntennaTable(StructArray(ants), String(array), ext)
end

"""
    freq_setup(ms::XRadio.MeasurementSet) -> FrequencySetup

The frequency setup of `ms`, from its `frequency` coordinate. MSv4 states one
`channel_width` for the window, so every channel gets it.

The sideband and total bandwidth are the coordinate's `sideband` and
`total_bandwidth` where the store states them ([`GUSTAVO_VISIBILITY_SCHEMA`](@ref)).
Otherwise the sideband is the direction of the channel axis, upper where the
frequency rises, and the total bandwidth is the channel count times the channel
width. A single-channel window without a stated sideband throws.
"""
function freq_setup(ms::XRadio.MeasurementSet)
    freq = dims(ms, XRadio.Frequency)
    freq === nothing && throw(ArgumentError("this Measurement Set has no frequency axis"))
    meta = DimensionalData.metadata(lookup(freq))
    channels = collect(lookup(freq))
    n = length(channels)
    width = XRadio.value(meta[:channel_width])
    sideband = haskey(meta, :sideband) ? meta[:sideband] : _channel_direction(channels)
    bandwidth = haskey(meta, :total_bandwidth) ?
        XRadio.value(meta[:total_bandwidth]) : n * abs(width)
    return FrequencySetup(;
        name = XRadio.spectralwindow(ms),
        ref_freq = XRadio.value(meta[:reference_frequency]),
        channel_freqs = channels,
        ch_widths = fill(width, n),
        total_bandwidths = fill(bandwidth, n),
        sidebands = fill(Float64(sideband), n),
    )
end

function _channel_direction(channels)
    length(channels) > 1 || throw(
        ArgumentError(
            "the store states no sideband, and a window of $(length(channels)) " *
                "channel(s) has no direction to derive one from"
        )
    )
    first(channels) < last(channels) && return 1
    first(channels) > last(channels) && return -1
    throw(ArgumentError("the channel frequencies neither rise nor fall"))
end

"""
    materialize(ms::MeasurementSet) -> MeasurementSet
    materialize(ps::ProcessingSet) -> ProcessingSet

`ms` or `ps` read into memory with arrays of its own, which no other
Measurement Set shares, so in-place corrections such as `calibrate!` never
reach the source. `read` of in-memory data shares its arrays; this copies
those it would share.

```julia
map(values(groupby(ps, ByScan()))) do g
    g = Gustavo.materialize(g)
    calibrate!(sol, g)
    ...
end
```
"""
function materialize(ms::XRadio.MeasurementSet)
    out = read(ms)
    for k in keys(out)
        parent(out[k]) === parent(ms[k]) && (out[k] = copy(out[k]))
    end
    return out
end

materialize(ps::XRadio.ProcessingSet) = XRadio.ProcessingSet(
    OrderedDict{Symbol, XRadio.MeasurementSet}(k => materialize(ms) for (k, ms) in pairs(ps)),
    copy(DimensionalData.metadata(ps)),
)
