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
    mounts = [UV.MountAltAz(), UV.MountEquatorial(), UV.MountNasmythR((1.5, 0.0, 0.0)), UV.MountAltAz()]
    make() = Testing.measurement_set(;
        antennas = names, times, frequencies = freqs, spectral_window = "spw_3",
        scan = "7", source = "3C273", field = "3C273",
        antenna_xds = Testing.antenna(names; positions, mounts, diameter = 12.0),
    )
    ms = make()

    function agrees(ms)
        @test UV.scan_name(ms) == "7"
        @test UV.primary_scan_name(ms) == "7"
        @test UV.source_name(ms) == "3C273"
        @test UV.sub_scan_name(ms) == ""
        @test UV.scan_intents(ms) == ["OBSERVE_TARGET#ON_SOURCE"]
        @test collect(UV.obs_time(ms)) == times
        @test UV.pol_products(ms) == ["RR", "RL", "LR", "LL"]

        bm = UV.baselines(ms)
        @test bm.pairs == [(a, b) for a in 1:4 for b in (a + 1):4]
        @test bm.ant1_names == [names[a] for (a, _) in bm.pairs]
        @test bm.ant2_names == [names[b] for (_, b) in bm.pairs]
        @test bm.labels == string.(bm.ant1_names, "-", bm.ant2_names)

        fs = UV.freq_setup(ms)
        @test UV.setup_name(fs) == "spw_3"
        @test UV.ref_freq(fs) == first(freqs)
        @test UV.channel_freqs(fs) == freqs
        @test all(==(2.0e6), UV.ch_widths(fs))

        am = UV.antennas(ms)
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

    @testset "a named sub-scan and stated intents" begin
        intents = ["OBSERVE_TARGET#ON_SOURCE", "CALIBRATE_DELAY#ON_SOURCE"]
        named = make()
        t = dims(named, Ti)
        named[:sub_scan_name] = DimArray(fill("sub_1", length(t)), (t,))
        DimensionalData.metadata(named[:scan_name])[:scan_intents] = intents
        @test UV.sub_scan_name(named) == "sub_1"
        @test UV.scan_intents(named) == intents
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

    @testset "a Measurement Set holding two scans has no scan name" begin
        two = make()
        t = dims(two, Ti)
        two[:scan_name] = DimArray([i <= length(t) ÷ 2 ? "1" : "2" for i in eachindex(t)], (t,))
        @test_throws "holds 2 scan names (1, 2)" UV.scan_name(two)
        @test_throws "holds 2 scan names" UV.primary_scan_name(two)
    end
end

# ── Set-level functions on a ProcessingSet ────────────────────────────────────

@testset "set-level functions on a ProcessingSet" begin
    UV = Gustavo.UVData
    ps = Testing.processing_set(; nantenna = 4, nspw = 2, nscan = 2)

    @testset "leaves are the Measurement Sets, by name" begin
        @test collect(UV.leaves(ps)) == collect(pairs(ps))
    end

    @testset "frequency and polarization unions" begin
        mine = UV.union_frequency_axis(ps)
        @test length(mine) == 2
        @test UV.channel_freqs.(mine) == [
            2.3e11 .+ 2.0e6 .* (0:7), 2.3e11 + 1.0e8 .+ 2.0e6 .* (0:7),
        ]
        @test UV.union_pol_products(ps) == ["RR", "RL", "LR", "LL"]
        @test_throws "2 distinct frequency setups" UV.freq_setup(ps)

        one_ps = Testing.processing_set(; nantenna = 4, nspw = 1, nscan = 2)
        @test UV.channel_freqs(UV.freq_setup(one_ps)) == 2.3e11 .+ 2.0e6 .* (0:7)
        @test UV.nchannels(one_ps) == 8
    end

    @testset "the antenna union" begin
        tab = UV.union_antennas(ps)
        @test tab.name == ["A1", "A2", "A3", "A4"]
        @test tab.station_xyz == [[1.0e4 * c * i for c in 1:3] for i in 1:4]
        @test all(==(UV.MountAltAz()), tab.mount)
        @test UV.extras(tab).DIAMETER == fill(25.0, 4)
    end

    @testset "members that saw different sub-arrays" begin
        # The second scan drops the second antenna from its antenna dataset, so
        # the union must restore the full order.
        names = ["A1", "A2", "A3", "A4"]
        keep = names[[1, 3, 4]]
        positions = [1.0e4 * c * i for c in 1:3, i in 1:4]
        full = Testing.measurement_set(; antennas = names, antenna_xds = Testing.antenna(names; positions))
        sub = Testing.measurement_set(;
            antennas = keep, times = [1.614816e9 + 300, 1.614816e9 + 330],
            scan = "2", antenna_xds = Testing.antenna(keep; positions = positions[:, [1, 3, 4]]),
        )
        sub_ps = ProcessingSet(OrderedDict(:one => full, :two => sub))
        @test length(unique(XRadio.antennas.(values(sub_ps)))) == 2

        tab = UV.union_antennas(sub_ps)
        @test tab.name == names
        @test UV.extras(tab).DIAMETER == fill(25.0, 4)
    end

    @testset "unstated receptor angles still union" begin
        bare = Testing.processing_set(; nantenna = 4, nspw = 2, nscan = 2)
        for ms in values(bare)
            delete!(DimensionalData.branches(ms)[:antenna], :antenna_receptor_angle)
        end
        tab = UV.union_antennas(bare)
        @test tab.name == ["A1", "A2", "A3", "A4"]
        @test all(a -> all(isnan, a), tab.pol_angles)
    end

    @testset "one antenna stated two ways is refused" begin
        moved = Testing.antenna(["A1", "A2", "A3"])
        parent(moved[:antenna_position])[1, 1] += 1.0
        clash = ProcessingSet(
            OrderedDict(
                :one => Testing.measurement_set(),
                :two => Testing.measurement_set(; scan = "2", antenna_xds = moved),
            )
        )
        @test_throws "inconsistent metadata across partitions" UV.union_antennas(clash)
    end
end
