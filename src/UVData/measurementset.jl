# The per-partition accessors, answered from an MSv4 `MeasurementSet`.
#
# Each returns what the same accessor returns for a `UVSet` leaf, rebuilt from
# the store on every call.

"""
    scan_name(ms::XRadio.MeasurementSet) -> String

The one scan `ms` holds. Throws when it holds none or several, since MSv4 does
not limit a Measurement Set to one scan.
"""
scan_name(ms::XRadio.MeasurementSet) = _only_name(XRadio.scans(ms), "scan")
primary_scan_name(ms::XRadio.MeasurementSet) = scan_name(ms)

"""
    source_name(ms::XRadio.MeasurementSet) -> String

The one source `ms` observes, from the `field_and_source` dataset of its `base`
data group. Throws when it names none or several.
"""
source_name(ms::XRadio.MeasurementSet) =
    _only_name(XRadio.partition_info(ms).source_name, "source")

"""
    sub_scan_name(ms::XRadio.MeasurementSet) -> String

The one sub-scan `ms` holds, from the `sub_scan_name` coordinate of
[`GUSTAVO_VISIBILITY_SCHEMA`](@ref). A store without that coordinate has no
sub-scans, and gives `""`.
"""
function sub_scan_name(ms::XRadio.MeasurementSet)
    haskey(ms, :sub_scan_name) || return ""
    return _only_name(unique(String.(collect(ms[:sub_scan_name]))), "sub-scan")
end

function _only_name(names, what)
    length(names) == 1 && return String(only(names))
    throw(
        ArgumentError(
            "this Measurement Set holds $(length(names)) $(what) names " *
                "($(join(names, ", "))), and a partition accessor needs exactly one"
        )
    )
end

"""
    scan_intents(ms::XRadio.MeasurementSet) -> Vector{String}

The scan intents as the store writes them, such as `"OBSERVE_TARGET#ON_SOURCE"`:
the `intents` of `observation_info` where stated, else the `scan_intents` of the
`scan_name` coordinate, else none.
"""
function scan_intents(ms::XRadio.MeasurementSet)
    info = get(DimensionalData.metadata(ms), :observation_info, Dict{Symbol, Any}())
    haskey(info, :intents) && return String.(info[:intents])
    haskey(ms, :scan_name) || return String[]
    return String.(get(DimensionalData.metadata(ms[:scan_name]), :scan_intents, String[]))
end

obs_time(ms::XRadio.MeasurementSet) = lookup(dims(ms, Ti))

pol_products(ms::XRadio.MeasurementSet) = XRadio.polarizations(ms)

"""
    baselines(ms::XRadio.MeasurementSet) -> BaselineIndex

The baselines of `ms` in `baseline_id` order, their antenna indices counting
into [`antennas(ms)`](@ref antennas(::XRadio.MeasurementSet)).
"""
function baselines(ms::XRadio.MeasurementSet)
    names = XRadio.antennas(ms)
    slot = Dict(n => i for (i, n) in pairs(names))
    index(n) = get(slot, n) do
        throw(
            ArgumentError(
                "baseline antenna `$n` is not in the antenna dataset, which names " *
                    join(names, ", ")
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
