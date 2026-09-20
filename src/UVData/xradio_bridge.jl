# A `UVSet` rendered as an MSv4 `ProcessingSet`.
#
# Gustavo's partition tree and MSv4's store hold the same observation in
# different shapes, so the same fixtures can drive both and a result computed
# from one can be checked against the other.

"""
    uvset_to_processingset(uvset::UVSet) -> XRadio.ProcessingSet

Build a `ProcessingSet` holding the same observation as `uvset`, one
`MeasurementSet` per leaf under the leaf's own key.

Each `MeasurementSet` conforms to `XRadio.VISIBILITY_SCHEMA`: the science
arrays are transposed into the standard's `(time, baseline_id, frequency,
polarization)` order, the antenna table becomes an `antenna` sub-dataset, and
the source position becomes a `field_and_source` one.

Two of the set's fields have no MSv4 field and are dropped rather than
approximated: the per-channel `sidebands` and `total_bandwidths` of the
[`FrequencySetup`](@ref), and the Earth-orientation block of
[`ObsArrayMetadata`](@ref). `FLAG` and `WEIGHT` come from the leaf's own
`:flags` and `:weights` layers, which carry the same independent meanings
MSv4 gives them.
"""
function uvset_to_processingset(uvset::UVSet)
    sets = OrderedDict{Symbol, XRadio.MeasurementSet}()
    for (key, leaf) in branches(uvset)
        sets[key] = _leaf_to_measurementset(leaf, metadata(uvset))
    end
    return XRadio.ProcessingSet(sets)
end

# The nominal integration time: the set's own `time_span` where it records one,
# and otherwise the axis' smallest positive spacing.
function _nominal_integration(times::AbstractVector, time_span::AbstractVector)
    isempty(time_span) || return Float64(first(time_span))
    length(times) < 2 && return 0.0
    spacings = filter(>(0), diff(sort(collect(times))))
    return isempty(spacings) ? 0.0 : Float64(minimum(spacings))
end

_measure(v, units) = XRadio.Measure(
    v, Dict{Symbol, Any}(:units => units, :type => "quantity"),
)

function _leaf_to_measurementset(leaf, root)
    part = is_lazy(leaf) ? materialize_leaf(leaf) : leaf
    info = metadata(part)
    fs = info.freq_setup

    times = Float64.(collect(obs_time(part)))
    freqs = Float64.(collect(channel_freqs(fs)))
    pols = String.(collect(pol_products(part)))
    nbl = length(info.baselines.pairs)

    time = Ti(
        DimensionalData.Lookups.Sampled(
            times;
            metadata = Dict{Symbol, Any}(
                :type => "time", :units => "s", :scale => "utc", :format => "unix",
                :integration_time => _measure(
                    _nominal_integration(times, info.time_span), "s",
                ),
            ),
        ),
    )
    freq = XRadio.Frequency(
        DimensionalData.Lookups.Sampled(
            freqs;
            metadata = Dict{Symbol, Any}(
                :type => "spectral_coord", :units => "Hz", :observer => "icrs",
                :spectral_window_name => String(info.spw_name),
                :spectral_window_intents => String[String(info.intent)],
                :reference_frequency => XRadio.Measure(
                    Float64(ref_freq(fs)),
                    Dict{Symbol, Any}(
                        :units => "Hz", :type => "spectral_coord", :observer => "icrs",
                    ),
                ),
                # MSv4's channel_width is one scalar for the window; a setup
                # with varying widths keeps only the first here.
                :channel_width => _measure(Float64(first(ch_widths(fs))), "Hz"),
            ),
        ),
    )
    pol = XRadio.Polarization(DimensionalData.Lookups.Categorical(pols))
    base = XRadio.BaselineID(DimensionalData.Lookups.Sampled(collect(1:nbl)))
    uvwlab = XRadio.UVWLabel(DimensionalData.Lookups.Categorical(["u", "v", "w"]))

    # Gustavo holds `(Frequency, Ti, Baseline, Pol)` and the standard's Julia
    # order is `(polarization, frequency, baseline_id, time)`.
    vis = permutedims(parent(part[:vis]), (4, 1, 3, 2))
    wgt = permutedims(parent(part[:weights]), (4, 1, 3, 2))
    flg = permutedims(parent(part[:flags]), (4, 1, 3, 2))
    uvw = permutedims(parent(part[:uvw]), (3, 2, 1))
    sci = (pol, freq, base, time)

    ms = XRadio.MeasurementSet(; metadata = _ms_metadata(root))
    ms[:visibility] = DimArray(vis, sci; metadata = Dict{Symbol, Any}(:units => "Jy"))
    ms[:flag] = DimArray(flg, sci)
    ms[:weight] = DimArray(wgt, sci)
    ms[:uvw] = DimArray(
        uvw, (uvwlab, base, time);
        metadata = Dict{Symbol, Any}(:type => "uvw", :units => "m", :frame => "icrs"),
    )
    ms[:field_name] = DimArray(fill(String(info.field_name), length(times)), (time,))
    ms[:scan_name] = DimArray(
        fill(String(info.scan_name), length(times)), (time,);
        metadata = Dict{Symbol, Any}(
            :scan_intents => String.(collect(info.scan_intents)),
        ),
    )
    ms[:baseline_antenna1_name] = DimArray(String.(info.baselines.ant1_names), (base,))
    ms[:baseline_antenna2_name] = DimArray(String.(info.baselines.ant2_names), (base,))

    branches(ms)[:antenna] = _antenna_dataset(info.antennas)
    branches(ms)[:field_and_source_base] = _field_and_source_dataset(info)
    return ms
end

function _ms_metadata(root)
    obs = root.array_obs
    return Dict{Symbol, Any}(
        :type => "visibility",
        :schema_version => string(XRadio.MSV4_SCHEMA_VERSION),
        :creation_date => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sss"),
        :creator => Dict{Symbol, Any}(
            :software_name => "Gustavo.jl", :version => string(pkgversion(UVData)),
        ),
        :observation_info => Dict{Symbol, Any}(
            :observer => String[String(obs.telescope)],
            :release_date => "",
            :project_UID => "",
        ),
        :processor_info => Dict{Symbol, Any}(
            :type => "CORRELATOR", :sub_type => String(obs.instrume),
        ),
        :data_groups => Dict{Symbol, Any}(
            :base => Dict{Symbol, Any}(
                :correlated_data => "VISIBILITY", :flag => "FLAG",
                :weight => "WEIGHT", :uvw => "UVW",
                :field_and_source => "field_and_source_base_xds",
                :description => "Correlator output",
                :date => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sss"),
            ),
        ),
    )
end

# AIPS mount codes, from the rotation-rate coefficients `Mount` carries.
function _mount_name(m::Mount)
    par, elev = parallactic_mount(m), elevation_mount(m)
    par == 0 && return "EQUATORIAL"
    elev == 1 && return "NASMYTH-R"
    elev == -1 && return "NASMYTH-L"
    return "ALT-AZ"
end

_receptor_label(p::PolTypes) =
    p isa RPol ? "R" : p isa LPol ? "L" : p isa XPol ? "X" : "Y"

function _antenna_dataset(tab::AntennaTable)
    names = String.(collect(tab.name))
    n = length(names)
    ant = XRadio.AntennaName(DimensionalData.Lookups.Categorical(names))
    receptor = XRadio.ReceptorLabel(DimensionalData.Lookups.Categorical(["1", "2"]))
    cart = XRadio.CartesianPosLabel(DimensionalData.Lookups.Categorical(["x", "y", "z"]))

    positions = Matrix{Float64}(undef, 3, n)
    for (i, xyz) in enumerate(tab.station_xyz)
        positions[:, i] .= Float64.(xyz)
    end
    basis = collect(tab.nominal_basis)

    ds = XRadio.Dataset(;
        metadata = Dict{Symbol, Any}(
            :type => "antenna",
            :overall_telescope_name => String(array_name(tab)),
            :relocatable_antennas => false,
        ),
    )
    ds[:antenna_position] = DimArray(
        positions, (cart, ant);
        metadata = Dict{Symbol, Any}(
            :type => "location", :units => "m", :frame => "ITRS",
            :coordinate_system => "geocentric", :origin_object_name => "earth",
        ),
    )
    ds[:station_name] = DimArray(names, (ant,))
    ds[:mount] = DimArray([_mount_name(m) for m in tab.mount], (ant,))
    ds[:telescope_name] = DimArray(fill(String(array_name(tab)), n), (ant,))
    ds[:polarization_type] = DimArray(
        [_receptor_label(basis[a][r]) for r in 1:2, a in eachindex(basis)], (receptor, ant),
    )
    return ds
end

function _field_and_source_dataset(info)
    field = XRadio.FieldName(
        DimensionalData.Lookups.Categorical([String(info.field_name)]),
    )
    skylab = XRadio.SkyDirLabel(DimensionalData.Lookups.Categorical(["ra", "dec"]))
    ds = XRadio.Dataset(; metadata = Dict{Symbol, Any}(:type => "field_and_source"))
    ds[:source_name] = DimArray([String(info.source_name)], (field,))
    ds[:field_phase_center_direction] = DimArray(
        reshape(Float64[info.ra, info.dec], 2, 1), (skylab, field);
        metadata = Dict{Symbol, Any}(
            :type => "sky_coord", :units => "rad", :frame => "icrs",
        ),
    )
    return ds
end
