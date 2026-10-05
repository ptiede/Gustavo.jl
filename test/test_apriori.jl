@testset "a-priori amplitude calibration" begin
    # Every antenna stands at the north pole, where a source's elevation is its
    # declination at all times, so the expected gains need no sidereal time.
    names = ["AA", "MG", "ZZ"]
    pole = [0.0 0.0 0.0; 0.0 0.0 0.0; 6356752.314245 6356752.314245 6356752.314245]
    dec = 0.5
    t0 = 1.614816e9                         # 2021-03-04T00:00:00, day 63
    function uncalibrated()
        ms = XRadio.Testing.measurement_set(;
            antennas = names, antenna_xds = XRadio.Testing.antenna(names; positions = pole),
            direction = (1.0, dec), times = t0 .+ [0.0, 30.0, 120.0, 150.0],
        )
        # Two scans of two 30 s integrations, ninety seconds apart.
        ms[:scan_name] = DimArray(["1", "1", "2", "2"], dims(ms[:scan_name]); metadata = metadata(ms[:scan_name]))
        DimensionalData.metadata(ms[:visibility])[:units] = "UNCALIB"
        ax = DimensionalData.branches(ms)[:antenna]
        ax[:digitizer_levels] = DimArray([2, 4, 4], dims(ax, XRadio.AntennaName))
        return ms
    end
    one_member(ms) = XRadio.ProcessingSet(OrderedDict{Symbol, XRadio.MeasurementSet}(:ms => ms))

    # The row at 75 s is a slew between the scans.
    antab = tempname()
    write(antab, """
    GAIN AA ELEV DPFU = 0.1, 0.2 POLY = 1.0, 0.01 /
    GAIN MG ALTAZ DPFU = 0.5 POLY = 2.0 /
    TSYS AA INDEX = 'R1', 'L1' /
    63 00:00:00   100.0  200.0
    63 00:00:30   110.0  210.0
    63 00:01:15   1.0e6  1.0e6
    63 00:02:00   120.0  220.0
    63 00:02:30   130.0  230.0
    /
    TSYS MG INDEX = 'R1:1', 'L1:1' /
    63 00:00:00   50.0   60.0
    63 00:02:00   70.0   80.0
    /
    """)
    function from_antab()
        ms = uncalibrated()
        @test_logs (:warn,) XRadio.read_antab!(one_member(ms), antab; ifs = Dict("spw_0" => 1))
        return ms
    end

    gain_aa = 1.0 + 0.01 * rad2deg(dec)
    sefd_aa(p, scan) = (p == 1 ? (105.0, 125.0) : (205.0, 225.0))[scan] / ((p == 1 ? 0.1 : 0.2) * gain_aa)
    sefd_mg(p, scan) = (p == 1 ? (50.0, 70.0) : (60.0, 80.0))[scan] / (0.5 * 2.0)
    scan_of(t) = t <= 2 ? 1 : 2
    # AA samples 2 levels and MG 4.
    η_aa_mg = (2 / π) * 0.8825
    feeds = [(1, 1), (1, 2), (2, 1), (2, 2)]

    @testset "into janskys, per receptor and per scan" begin
        ms = from_antab()
        @test apriori_calibrate!(ms) === ms
        @test DimensionalData.metadata(ms[:visibility])[:units] == "Jy"
        V, W, F = parent(ms[:visibility]), parent(ms[:weight]), parent(ms[:flag])
        # Baseline 1 is AA–MG; the slew row never reaches a scan.
        for t in 1:4, (p, (fa, fb)) in pairs(feeds)
            s = sefd_aa(fa, scan_of(t)) * sefd_mg(fb, scan_of(t))
            @test all(V[p, :, 1, t] .≈ sqrt(s) / η_aa_mg)
            @test all(W[p, :, 1, t] .≈ η_aa_mg^2 / s)
            @test !any(F[p, :, 1, t])
        end
        # ZZ has no system temperature: flagged, and left as it was.
        @test all(F[:, :, 2:3, :])
        @test all(==(1), V[:, :, 2:3, :]) && all(==(1), W[:, :, 2:3, :])

        @test_throws "already in Jy" apriori_calibrate!(ms)
    end

    @testset "placing the system temperatures" begin
        rows = [0.0, 30.0, 75.0, 120.0, 150.0]
        values = [100.0, NaN, 1.0e6, 120.0, 130.0]
        times = [0.0, 30.0, 120.0, 150.0]
        scans = ["1", "1", "2", "2"]
        place(rule) = Gustavo.UVData.place_tsys(rule, rows, values, times, scans, 15.0)
        @test place(ScanMean()) == [100.0, 100.0, 125.0, 125.0]
        @test place(NearestInTime()) == [100.0, 100.0, 120.0, 130.0]
        # Interpolation does cross the slew row, which is what `ScanMean` avoids.
        @test place(LinearInTime()) ≈ [100.0, 100.0 + 30 / 75 * (1.0e6 - 100.0), 120.0, 130.0]
        @test isnan(only(Gustavo.UVData.place_tsys(LinearInTime(), rows, values, [200.0], ["3"], 15.0)))
        @test isnan(only(Gustavo.UVData.place_tsys(ScanMean(), rows, values, [200.0], ["3"], 15.0)))

        ms = from_antab()
        apriori_calibrate!(ms; tsys = NearestInTime())
        @test parent(ms[:visibility])[1, 1, 1, 1] ≈ sqrt(100.0 / (0.1 * gain_aa) * sefd_mg(1, 1)) / η_aa_mg
    end

    @testset "implausible system temperatures count as unmeasured" begin
        screened = tempname()
        write(screened, """
        GAIN AA ELEV DPFU = 0.1, 0.2 POLY = 1.0, 0.01 /
        GAIN MG ALTAZ DPFU = 0.5 POLY = 2.0 /
        TSYS AA INDEX = 'R1', 'L1' /
        63 00:00:00   100.0  200.0
        63 00:00:30   999.0  -5.0
        63 00:02:00   2.0e4  2.0e4
        /
        TSYS MG INDEX = 'R1:1', 'L1:1' /
        63 00:00:00   50.0   60.0
        63 00:02:00   70.0   80.0
        /
        """)
        function filled()
            ms = uncalibrated()
            @test_logs (:warn,) XRadio.read_antab!(one_member(ms), screened; ifs = Dict("spw_0" => 1))
            return ms
        end
        ms = apriori_calibrate!(filled())
        V, F = parent(ms[:visibility]), parent(ms[:flag])
        # The placeholder and the negative row leave the first scan's mean alone.
        @test V[1, 1, 1, 1] ≈ sqrt(100.0 / (0.1 * gain_aa) * sefd_mg(1, 1)) / η_aa_mg
        @test V[4, 1, 1, 2] ≈ sqrt(200.0 / (0.2 * gain_aa) * sefd_mg(2, 1)) / η_aa_mg
        # The second scan's only row is above `max_tsys`, so AA is unmeasured there.
        @test all(F[:, :, 1:2, 3:4]) && all(==(1), V[:, :, 1:2, 3:4])
        # The store keeps what the file said.
        @test 999.0 in XRadio.system_temperatures(ms)

        raised = apriori_calibrate!(filled(); max_tsys = 3.0e4)
        @test parent(raised[:visibility])[1, 1, 1, 3] ≈ sqrt(2.0e4 / (0.1 * gain_aa) * sefd_mg(1, 2)) / η_aa_mg
    end

    @testset "below the elevation limit" begin
        ms = from_antab()
        apriori_calibrate!(ms; min_elevation = dec + 0.01)
        @test all(parent(ms[:flag]))
        @test all(==(1), parent(ms[:visibility]))
    end

    @testset "from datasets laid out as fitsidi2msv4 writes them" begin
        # Temperatures on the calibration clock beside antenna temperatures,
        # and curves in elevation, one per band.
        ms = uncalibrated()
        ax = DimensionalData.branches(ms)[:antenna]
        roster = (;
            antenna_name = names, station_name = names, receptor_label = ["pol_0", "pol_1"],
            polarization_type = [r for r in ("R", "L"), _ in 1:3],
        )
        tsys = fill(NaN, 2, 2, 3)
        tsys[:, :, 1] = [100.0 120.0; 200.0 220.0]
        tsys[:, :, 2] = [50.0 70.0; 60.0 80.0]
        ms[:system_calibration] = XRadio.subdataset(
            :system_calibration; roster..., time_system_cal = t0 .+ [10.0, 130.0],
            tsys, tant = tsys ./ 4,
        )
        coefficients = zeros(2, 2, 3)
        coefficients[:, :, 1] .= [1.0 0.01; 1.0 0.01]
        coefficients[:, 1, 2] .= 2.0
        ms[:gain_curve] = XRadio.subdataset(
            :gain_curve; roster..., mount = collect(ax[:mount]), telescope_name = collect(ax[:telescope_name]),
            poly_term = [0, 1], gain_curve_type = fill("POWER(EL)", 3), gain_curve = coefficients,
            gain_curve_sensitivity = [0.1 0.5 NaN; 0.2 0.5 NaN], gain_curve_interval = fill(150.0, 3),
            metadata = (; measured_date = "1858-11-17T00:00:00.000"),
        )
        @test isempty(XRadio.check(ms))
        apriori_calibrate!(ms)
        V = parent(ms[:visibility])
        for t in 1:4, (p, (fa, fb)) in pairs(feeds)
            aa = tsys[fa, scan_of(t), 1] / ((fa == 1 ? 0.1 : 0.2) * gain_aa)
            mg = tsys[fb, scan_of(t), 2] / (0.5 * 2.0)
            @test all(V[p, :, 1, t] .≈ sqrt(aa * mg) / η_aa_mg)
        end
    end

    @testset "quantization efficiency" begin
        recorded = apriori_calibrate!(from_antab())
        function unrecorded()
            ms = from_antab()
            delete!(DimensionalData.branches(ms)[:antenna], :digitizer_levels)
            return ms
        end
        A = XRadio.AntennaName

        per_antenna = Dict("AA" => 2 / π, "MG" => 0.8825, :ZZ => 0.5, "XX" => 0.7)
        given = apriori_calibrate!(unrecorded(); quantization_efficiency = per_antenna)
        @test parent(given[:visibility]) ≈ parent(recorded[:visibility])
        @test parent(given[:weight]) ≈ parent(recorded[:weight])

        unit = apriori_calibrate!(unrecorded(); quantization_efficiency = 1)
        V, W = parent(recorded[:visibility]), parent(recorded[:weight])
        @test V[:, :, 1, :] ≈ parent(unit[:visibility])[:, :, 1, :] ./ η_aa_mg
        @test W[:, :, 1, :] ≈ parent(unit[:weight])[:, :, 1, :] .* η_aa_mg^2
        uniform = apriori_calibrate!(unrecorded(); quantization_efficiency = 0.5)
        @test parent(uniform[:visibility])[:, :, 1, :] ≈ parent(unit[:visibility])[:, :, 1, :] ./ 0.25

        @test_throws "no value for antenna ZZ" apriori_calibrate!(
            unrecorded(); quantization_efficiency = Dict("AA" => 2 / π, "MG" => 0.8825)
        )
        @test_throws "tabulated quantization efficiency for every antenna; `quantization_efficiency` cannot also be given" apriori_calibrate!(from_antab(); quantization_efficiency = 2 / π)
        @test_throws "pass `quantization_efficiency` (e.g. `2/π` for 2-level sampling)" apriori_calibrate!(unrecorded())
        # An untabulated level count: the keyword supplies that antenna's η, and
        # only that antenna's.
        function eight()
            ms = from_antab()
            DimensionalData.branches(ms)[:antenna][:digitizer_levels][A(At("MG"))] = 8
            return ms
        end
        @test_throws "antennas MG have 8 digitizer levels, for which no quantization efficiency is tabulated" apriori_calibrate!(eight())
        filled = apriori_calibrate!(eight(); quantization_efficiency = Dict("MG" => 0.8825))
        @test parent(filled[:visibility]) ≈ parent(recorded[:visibility])
        scalar = apriori_calibrate!(eight(); quantization_efficiency = 0.8825)
        @test parent(scalar[:visibility]) ≈ parent(recorded[:visibility])
        @test_throws "gives antennas AA, whose `digitizer_levels` fix it" apriori_calibrate!(
            eight(); quantization_efficiency = Dict("AA" => 2 / π, "MG" => 0.8825)
        )
    end

    @testset "what it refuses" begin
        @test_throws "records no system temperatures" apriori_calibrate!(uncalibrated())
        ms = from_antab()
        delete!(DimensionalData.branches(ms), :gain_curve)
        @test_throws "records no gain curves" apriori_calibrate!(ms)
    end
end
