@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

const CALg = Gustavo.Calibration

_relabel(ms, name, values) = (ms[name] = rebuild(ms[name], values); ms)

function _subarray_ms(antennas, scan; times = 1.6e9 .+ 30.0 .* (0:3), spw = "band_1")
    return XRadio.Testing.measurement_set(;
        antennas, times, frequencies = 230.0e9 .+ 2.0e6 .* (0:3),
        spectral_window = spw, scan, field = "F$scan", source = "S$scan",
    )
end

@testset "the geometry of a ProcessingSet" begin
    ps, truth = _build_fringe_ps(; nant = 3, nspw = 2, nchan = 4, ntime = 5, nscans = 2)
    geom = CALg.DataGeometry(ps)

    @test length(geom.times) == 10
    @test geom.scan_of_time == repeat([1, 2]; inner = 5)
    @test geom.scan_names == ["1", "2"]
    @test geom.spw_of_chan == repeat([1, 2]; inner = 4)
    @test geom.spw_names == ["band_1", "band_2"]
    @test geom.channel_freqs == sort!(unique!(reduce(vcat, [XRadio.frequencies(ms) for ms in ps])))
    @test geom.stations == ["A1", "A2", "A3"]
    @test geom.f0 ≈ truth.f0
    @test geom.t0 == truth.t0_sec

    @testset "a station on no baseline is still a station" begin
        dropped, _ = _build_fringe_ps(; nant = 4, nspw = 1, omit_station = 2)
        @test CALg.DataGeometry(dropped).stations == ["A1", "A2", "A3", "A4"]
    end

    @testset "a Measurement Set holding several scans" begin
        one, _ = _build_fringe_ps(; nant = 3, nspw = 2, nchan = 4, ntime = 6)
        for ms in one
            _relabel(ms, :scan_name, ["1", "1", "1", "2", "2", "2"])
        end
        g = CALg.DataGeometry(one)
        @test g.scan_of_time == [1, 1, 1, 2, 2, 2]
        @test g.scan_names == ["1", "2"]
    end

    @testset "sub-arrays" begin
        pair = XRadio.ProcessingSet(
            OrderedDict(:a => _subarray_ms(["A1", "A2"], "1"), :b => _subarray_ms(["B1", "B2"], "2"))
        )
        g = CALg.DataGeometry(pair)
        @test g.stations == ["A1", "A2", "B1", "B2"]
        @test length(g.times) == 4
        @test CALg.GeometryWindow(g, pair[:b]).stations == [(3, 4)]

        clash = XRadio.ProcessingSet(
            OrderedDict(:a => _subarray_ms(["A1", "A2"], "1"), :b => _subarray_ms(["A2", "B2"], "2"))
        )
        @test_throws "cannot be in two scans at one instant" CALg.DataGeometry(clash)
        @test_throws "A2" CALg.DataGeometry(clash)
    end

    @testset "errors" begin
        @test_throws "holds no Measurement Sets" CALg.DataGeometry(filter(_ -> false, ps))
        unlabelled = copy(first(ps))
        delete!(unlabelled, :scan_name)
        bare = XRadio.ProcessingSet(OrderedDict(:x => unlabelled))
        @test_throws "states no `scan_name`" CALg.DataGeometry(bare)
    end
end

@testset "a Measurement Set's window" begin
    ps, truth = _build_fringe_ps(; nant = 3, nspw = 2, nchan = 4, ntime = 5, nscans = 2)
    geom = CALg.DataGeometry(ps)
    ms = ps[:synth_fringe_2_2]
    win = CALg.GeometryWindow(geom, ms)

    @test win.geom === geom
    @test win.chan_idx == 5:8
    @test win.ti_idx == 6:10
    @test win.stations == truth.bl_pairs
    @test win.feed_order == [(1, 1), (1, 2), (2, 1), (2, 2)]
    pairs = Gustavo.UVData.feed_pairs(ms)
    @test all(pairs[win.feeds[k, b], b] == win.feed_order[k] for k in axes(win.feeds, 1), b in axes(win.feeds, 2))

    @testset "a view addresses its own samples" begin
        part = view(ms, XRadio.Ti(At(XRadio.times(ms)[2:3])))
        @test CALg.GeometryWindow(geom, part).ti_idx == [7, 8]
    end

    @testset "a station whose receptors are ordered L, R" begin
        names = ["A1", "A2", "A3"]
        ax = XRadio.Testing.antenna(names)
        types = copy(parent(ax[:polarization_type]))
        types[:, 2] = ["L", "R"]
        ax[:polarization_type] = rebuild(ax[:polarization_type], types)
        lr = XRadio.Testing.measurement_set(;
            antennas = names, antenna_xds = ax, times = 1.6e9 .+ 30.0 .* (0:3),
            frequencies = 230.0e9 .+ 2.0e6 .* (0:3), scan = "1",
        )
        w = CALg.GeometryWindow(CALg.DataGeometry(XRadio.ProcessingSet(OrderedDict(:lr => lr))), lr)
        lr_pairs = Gustavo.UVData.feed_pairs(lr)
        a1a2 = findfirst(==((1, 2)), w.stations)
        rr = findfirst(==("RR"), XRadio.polarizations(lr))
        # RR on A1–A2 relates A1's feed 1 to A2's feed 2.
        @test lr_pairs[rr, a1a2] == (1, 2)
        @test w.feeds[findfirst(==((1, 2)), w.feed_order), a1a2] == rr
        @test all(lr_pairs[w.feeds[k, b], b] == w.feed_order[k] for k in axes(w.feeds, 1), b in axes(w.feeds, 2))
    end

    @testset "errors" begin
        other = XRadio.ProcessingSet(OrderedDict(:o => _subarray_ms(["A1", "Z9"], "1")))
        @test_throws "`Z9` is not among the geometry's stations" CALg.GeometryWindow(
            CALg.DataGeometry(XRadio.ProcessingSet(OrderedDict(:a => _subarray_ms(["A1", "A2"], "1")))),
            other[:o],
        )
        late = _subarray_ms(["A1", "A2"], "1"; times = 2.0e9 .+ 30.0 .* (0:3))
        @test_throws "is not in the geometry" CALg.GeometryWindow(geom, late)
        @test_throws "names no stations" CALg.GeometryWindow(
            CALg.DataGeometry(; times = geom.times, channel_freqs = geom.channel_freqs), ms,
        )
    end
end
