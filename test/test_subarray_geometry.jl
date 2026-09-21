# Concurrent sub-arrays on one time axis: which overlaps `build_geometry`
# admits and which it refuses.

using Dates: DateTime, Date, datetime2unix

const CALsub = Gustavo.Calibration
const UVsub = Gustavo.UVData

# A UVSet of one leaf per (scan name, antenna index set), each leaf observing
# `times`. Every leaf shares one antenna table, so a station set is fixed by
# which baselines the leaf carries.
function _subarray_uvset(specs; nchan = 4, nant = 6)
    ants = [
        UVsub.Antenna(;
            name = "A$(i)",
            station_xyz = Float64[1.0e4 * i, 2.0e4 * i, 3.0e4 * i],
            mount = UVsub.MountAltAz(),
            nominal_basis = (RPol(), LPol()),
            response = Diagonal(ones(ComplexF32, 2)),
            pol_angles = (0.0f0, 0.0f0),
        ) for i in 1:nant
    ]
    antennas = UVsub.AntennaTable(
        StructArray(ants), Float64[1.0:nant;], "SYNTH",
        (; NOSTA = Int32.(1:nant), DIAMETER = fill(25.0f0, nant)),
    )
    chf = 230.0e9 .+ (0:(nchan - 1)) .* 2.0e6
    fs = UVsub.FrequencySetup(;
        name = "band_1", ref_freq = 230.0e9, channel_freqs = collect(chf),
        ch_widths = fill(2.0e6, nchan), total_bandwidths = fill(2.0e6 * nchan, nchan),
        sidebands = fill(1.0, nchan),
    )
    array_obs = UVsub.ObsArrayMetadata(;
        telescope = "SYNTH", instrume = "SYNTH", date_obs = "2021-03-04",
        equinox = 2000.0f0, bunit = "JY", rdate = "2021-03-04",
        earth_rot_rate = 360.0f0,
    )
    pol_labels = ["PP", "QQ"]

    branches = DimensionalData.TreeDict()
    for (scan, src, antidx, times) in specs
        pairs = [(a, b) for a in antidx for b in antidx if a < b]
        bls = UVsub.BaselineIndex(pairs, pairs; antenna_names = collect(antennas.name))
        nbl, nt = length(pairs), length(times)
        vis = DimArray(
            fill(ComplexF32(1), nchan, nt, nbl, length(pol_labels)),
            (Frequency(collect(chf)), Ti(collect(times)),
                Baseline(bls.labels), Pol(pol_labels)),
        )
        w = DimArray(ones(Float32, size(vis)), dims(vis))
        fl = DimArray(falses(size(vis)), dims(vis))
        uvw = DimArray(
            zeros(Float32, nt, nbl, 3),
            (Ti(collect(times)), Baseline(bls.labels), UVW(["U", "V", "W"])),
        )
        info = UVsub.PartitionInfo(;
            source_name = src, source_key = UVsub.sanitize_source(src),
            scan_name = scan, ra = 1.234, dec = -0.56,
            antennas = antennas, baselines = bls,
            record_order = [(ti, bl) for bl in 1:nbl for ti in 1:nt],
            freq_setup = fs, spw_name = "band_1", ddi = 0, basename = "synth_sub",
        )
        branches[UVsub.partition_key(info)] =
            UVsub._build_leaf(vis, w, uvw, fl; partition_info = info)
    end
    return UVSet(; metadata = UVsub.UVMetadata(array_obs), branches = branches)
end

@testset "build_geometry: concurrent sub-arrays" begin
    t0 = datetime2unix(DateTime(Date("2021-03-04")))
    tspan = collect(t0 .+ (0:5) .* 30.0)

    @testset "disjoint stations over the same window are admitted" begin
        uvset = _subarray_uvset(
            [
                ("sA", "3C279", [1, 2, 3], tspan),
                ("sB", "J0423-0120", [4, 5, 6], tspan),
            ],
        )
        geom = CALsub.build_geometry(uvset)
        @test geom.times == tspan
        # One label per timestamp, and every timestamp resolved.
        @test length(geom.scan_of_time) == length(tspan)
        @test all(>(0), geom.scan_of_time)
        # Both leaves land in one segment each, so neither scan is split.
        @test length(unique(geom.scan_of_time)) == 1
    end

    @testset "a station in two scans at one instant is refused" begin
        uvset = _subarray_uvset(
            [
                ("sA", "3C279", [1, 2, 3], tspan),
                ("sB", "J0423-0120", [3, 4, 5], tspan),   # A3 in both
            ],
        )
        @test_throws "cannot be in two scans at one instant" CALsub.build_geometry(uvset)
        @test_throws "A3" CALsub.build_geometry(uvset)
    end

    @testset "partial overlap would split a scan and is refused" begin
        # sB runs past sA's window, so its later timestamps carry a different
        # label from its earlier ones.
        uvset = _subarray_uvset(
            [
                ("sA", "3C279", [1, 2, 3], tspan[1:3]),
                ("sB", "J0423-0120", [4, 5, 6], tspan),
            ],
        )
        @test_throws "partial overlap is not" CALsub.build_geometry(uvset)
    end

    @testset "timestamps within _epoch_atol are one instant" begin
        # Two sub-arrays whose axes differ by one ULP per sample, as separately
        # loaded correlator files can.
        nudged = [nextfloat(t) for t in tspan]
        @test nudged != tspan
        uvset = _subarray_uvset(
            [
                ("sA", "3C279", [1, 2, 3], tspan),
                ("sB", "J0423-0120", [4, 5, 6], nudged),
            ],
        )
        geom = CALsub.build_geometry(uvset)
        # One axis entry per integration, not two.
        @test length(geom.times) == length(tspan)
        @test all(t -> any(g -> isapprox(g, t; atol = CALsub._epoch_atol(t)), geom.times), nudged)

        # And the shared-station check still fires across the nudged axes.
        clash = _subarray_uvset(
            [
                ("sA", "3C279", [1, 2, 3], tspan),
                ("sB", "J0423-0120", [3, 4, 5], nudged),
            ],
        )
        @test_throws "cannot be in two scans at one instant" CALsub.build_geometry(clash)
    end

    @testset "a single sub-array is unaffected" begin
        uvset = _subarray_uvset([("sA", "3C279", [1, 2, 3], tspan)])
        geom = CALsub.build_geometry(uvset)
        @test geom.times == tspan
        @test geom.scan_names == ["sA"]
    end
end
