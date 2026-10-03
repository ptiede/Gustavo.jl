# ── Baselines of a Measurement Set on the solve's station axis ────────────────
#
# MSv4 names each baseline's antennas; a solve numbers stations once, for the
# whole set. Measurement Sets that saw different sub-arrays must count into
# that one table, not their own antenna datasets.

using XRadio: XRadio, ProcessingSet

@testset "Measurement Set baselines on the station table" begin
    UV = Gustavo.UVData
    Testing = XRadio.Testing

    ant = Testing.antenna(["A1", "A2", "A3", "A4"])
    full = Testing.measurement_set(; antennas = ["A1", "A2", "A3", "A4"], antenna_xds = ant)
    # The same antennas as `full`'s, at the same positions.
    sub_ant = Testing.antenna(["A2", "A4"]; positions = [1.0e4 * c * i for c in 1:3, i in (2, 4)])
    sub = Testing.measurement_set(; antennas = ["A2", "A4"], antenna_xds = sub_ant)
    stations = UV.union_antennas(ProcessingSet(OrderedDict(:full => full, :sub => sub)))
    @test collect(stations.name) == ["A1", "A2", "A3", "A4"]

    # Its own antenna dataset numbers A2 and A4 as 1 and 2; the station table
    # numbers them 2 and 4.
    @test UV.baselines(sub).pairs == [(1, 2)]
    @test UV.baselines(sub, stations).pairs == [(2, 4)]
    @test UV.baselines(full, stations).pairs == UV.baselines(full).pairs
    @test UV.baselines(sub, stations).labels == ["A2-A4"]

    few = UV.antennas(Testing.measurement_set(; antennas = ["A1", "A2"]))
    @test_throws "baseline antenna `A3` is not in the station table" UV.baselines(full, few)
end
