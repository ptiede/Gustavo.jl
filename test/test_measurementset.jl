# ── Gustavo's MSv4 schema extension, and the accessors on a Measurement Set ───
#
# What MSv4 has no field for travels as attributes and as a coordinate over a
# dimension the standard already defines, so a store carrying it still conforms.
# The specification is what makes `check` examine it and `write` name it.

using XRadio: XRadio, ProcessingSet, MeasurementSet, Testing
using StructArrays

@isdefined(uvfits_fixture) || include("synthetic_uvfits.jl")

@testset "GUSTAVO_VISIBILITY_SCHEMA" begin
    UV = Gustavo.UVData
    ps = load_uvfits(first(uvfits_fixture()))
    spec = UV.GUSTAVO_VISIBILITY_SCHEMA

    @testset "it extends the standard rather than replacing it" begin
        @test spec.type == XRadio.VISIBILITY_SCHEMA.type
        freq = only(filter(c -> c.name === :frequency, spec.coords))
        named = [a.name for a in freq.attrs]
        # The standard's own frequency attributes survive the merge.
        @test :channel_width in named
        @test :reference_frequency in named
        @test :sideband in named
        @test :total_bandwidth in named
        @test any(c -> c.name === :sub_scan_name, spec.coords)
        @test any(a -> a.name === :earth_orientation, spec.attrs)
    end

    @testset "every addition is optional" begin
        # A store written by anything else still passes, which is what lets
        # Gustavo check data it did not write.
        freq = only(filter(c -> c.name === :frequency, spec.coords))
        @test only(filter(a -> a.name === :sideband, freq.attrs)).optional
        @test only(filter(a -> a.name === :total_bandwidth, freq.attrs)).optional
        @test only(filter(c -> c.name === :sub_scan_name, spec.coords)).optional
        @test only(filter(a -> a.name === :earth_orientation, spec.attrs)).optional
        @test isempty(XRadio.check(Testing.processing_set(); schemas = [spec]))
    end

    @testset "the earth-orientation block is all-or-nothing" begin
        # A half-filled block is the case worth reporting: read as zeros it
        # would silently mis-state the rotation.
        block = only(filter(a -> a.name === :earth_orientation, spec.attrs)).nested
        @test !any(f -> f.optional, block.fields)
        @test sort([f.name for f in block.fields]) == sort(
            [
                :gst_iat0, :earth_rot_rate, :ut1utc, :polarx,
                :polary, :datutc, :xyzhand, :poltype,
            ]
        )
        half = load_uvfits(first(uvfits_fixture()))
        for ms in values(half)
            delete!(DimensionalData.metadata(ms)[:earth_orientation], :poltype)
        end
        @test !isempty(XRadio.check(half; schemas = [spec]))
    end

    @testset "a store carrying the extension conforms" begin
        @test isempty(XRadio.check(ps; schemas = [spec]))
        # And still conforms when nothing is told about the extension: the
        # additions are an attribute and a coordinate over `time`, both of
        # which MSv4 permits unconditionally.
        @test isempty(XRadio.check(ps))
    end

    @testset "the schema names `sub_scan_name` on disk" begin
        # Without the specification the layer is written `SUB_SCAN_NAME`, which
        # a Python reader sees as a data variable rather than a coordinate.
        named = load_uvfits(first(uvfits_fixture()))
        for ms in values(named)
            t = dims(ms, Ti)
            ms[:sub_scan_name] = DimArray(fill("sub_0", length(t)), (t,))
        end
        path = joinpath(mktempdir(), "named.ps.zarr")
        write(path, named; schemas = [spec], validate = false)
        node = joinpath(path, String(first(keys(named))))
        @test isdir(joinpath(node, "sub_scan_name"))
        @test !isdir(joinpath(node, "SUB_SCAN_NAME"))
        # The variables it labels list it among their coordinates, which is how
        # xarray tells a coordinate from a data variable.
        @test occursin(
            "sub_scan_name", read(joinpath(node, "VISIBILITY", ".zattrs"), String)
        )

        back = read(XRadio.ProcessingSet, path)
        key = first(keys(named))
        @test collect(back[key][:sub_scan_name]) == collect(named[key][:sub_scan_name])
        @test DimensionalData.metadata(
            lookup(back[key][:visibility], XRadio.Frequency)
        )[:sideband] ==
            DimensionalData.metadata(
            lookup(named[key][:visibility], XRadio.Frequency)
        )[:sideband]
    end
end

# ── Partition accessors on a MeasurementSet ───────────────────────────────────
#
# Each accessor answers what the Measurement Set was built from.

@testset "partition accessors on a MeasurementSet" begin
    UV = Gustavo.UVData
    names = ["A1", "A2", "A3", "A4"]
    times = 1.614816e9 .+ (0:5) .* 30.0
    freqs = 2.3e11 .+ 2.0e6 .* (0:7)
    positions = [1.0e4 * c * i for c in 1:3, i in 1:4]
    mounts = [XRadio.MountAltAz(), XRadio.MountEquatorial(), XRadio.MountNasmythR((1.5, 0.0, 0.0)), XRadio.MountAltAz()]
    make() = Testing.measurement_set(;
        antennas = names, times, frequencies = freqs, spectral_window = "spw_3",
        scan = "7", source = "3C273", field = "3C273",
        antenna_xds = Testing.antenna(names; positions, mounts, diameter = 12.0),
    )
    ms = make()

    function agrees(ms)
        fs = UV.freq_setup(ms)
        @test UV.ref_freq(fs) == first(freqs)
        @test UV.channel_freqs(fs) == freqs
        @test all(==(2.0e6), UV.ch_widths(fs))

        am = UV._antenna_table(ms)
        @test am.name == names
        @test am.station_xyz == [positions[:, i] for i in 1:4]
        @test am.mount == mounts
        @test all(==((UV.RPol(), UV.LPol())), am.nominal_basis)
        @test all(==((0.0, 0.0)), am.pol_angles)
        @test UV.array_name(am) == "SYNTHETIC"
        return @test UV.extras(am).DIAMETER == fill(12.0, 4)
    end

    @testset "in memory" begin
        agrees(ms)
    end

    @testset "read back from disk" begin
        path = joinpath(mktempdir(), "accessors.ps.zarr")
        write(path, ProcessingSet(OrderedDict(:one => ms)); schemas = [UV.GUSTAVO_VISIBILITY_SCHEMA])
        agrees(read(XRadio.ProcessingSet, path)[:one])
    end

    @testset "sideband and bandwidth derived where the store states none" begin
        for sideband in (1, -1)
            m = Testing.measurement_set(; frequencies = sideband > 0 ? freqs : reverse(freqs))
            meta = DimensionalData.metadata(lookup(dims(m, XRadio.Frequency)))
            meta[:sideband] = sideband
            meta[:total_bandwidth] = XRadio.Measure(
                5.0e7, Dict{Symbol, Any}(:units => "Hz", :type => "quantity")
            )
            # A stated bandwidth wider than the channels span, so reading the
            # stored value and deriving one give different answers.
            @test all(==(5.0e7), UV.total_bandwidths(UV.freq_setup(m)))
            delete!(meta, :sideband)
            delete!(meta, :total_bandwidth)
            derived = UV.freq_setup(m)
            @test all(==(sideband), UV.sidebands(derived))
            @test all(==(8 * 2.0e6), UV.total_bandwidths(derived))
        end
        @test_throws "no direction to derive one from" UV._channel_direction([2.3e11])
    end
end

# ── Set-level functions on a ProcessingSet ────────────────────────────────────

@testset "set-level functions on a ProcessingSet" begin
    UV = Gustavo.UVData
    ps = Testing.processing_set(; nantenna = 4, nspw = 2, nscan = 2)

    @testset "one frequency setup across the set" begin
        @test_throws "2 distinct frequency setups" UV.freq_setup(ps)
        one_ps = Testing.processing_set(; nantenna = 4, nspw = 1, nscan = 2)
        @test UV.channel_freqs(UV.freq_setup(one_ps)) == 2.3e11 .+ 2.0e6 .* (0:7)
    end
end
