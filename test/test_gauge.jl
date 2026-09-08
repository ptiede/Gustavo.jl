# Gauge conventions for station-based solves.
# Standalone-runnable and included from runtests.jl.

using Gustavo
using Test

const CALg = Gustavo.Calibration

# A 4-station, 2-feed node vector, laid out as the station solves lay it out:
# feed 1 occupies 1:nant, feed 2 occupies nant+1:2nant.
const GN = 4
_station_of(n) = (n - 1) % GN + 1
_feed_of(n) = n > GN ? 2 : 1

# Stations 1–3 present on both feeds; station 4 absent entirely.
const GCOMP = [1, 2, 3, 5, 6, 7]
const GW = Float64[10, 5, 3, 0, 8, 4, 2, 0]

_row(g, comp = GCOMP, T = Float64) = begin
    r = zeros(T, 2GN)
    CALg.gauge_row!(view(r, :), g, comp, GW, _station_of, _feed_of)
    r
end

@testset "PinAntenna picks the ranked reference" begin
    # Feed 1 before feed 2, so a station's values stay referenced to one feed.
    @test _row(PinAntenna(2)) == [0, 1, 0, 0, 0, 0, 0, 0]
    @test CALg.gauge_anchor(PinAntenna(2), GCOMP, GW, _station_of, _feed_of) == 2

    # A ranked list falls to the next entry when the leading one is absent —
    # station 9 does not exist, station 4 is absent from this component.
    @test _row(PinAntenna([9, 3, 2])) == [0, 0, 1, 0, 0, 0, 0, 0]
    @test _row(PinAntenna([4, 2])) == [0, 1, 0, 0, 0, 0, 0, 0]

    # With no listed reference present, the best-OBSERVED node carries the gauge.
    @test _row(PinAntenna([4])) == [1, 0, 0, 0, 0, 0, 0, 0]

    # A component holding only feed-2 nodes gauges on feed 2.
    @test _row(PinAntenna(2), [5, 6, 7]) == [0, 0, 0, 0, 0, 1, 0, 0]
end

@testset "ZeroSumPhase spreads the gauge over the component" begin
    r = _row(ZeroSumPhase())
    @test sum(r) ≈ 1
    @test count(!iszero, r) == length(GCOMP)      # station 4 contributes nothing
    @test all(r[n] ≈ 1 / length(GCOMP) for n in GCOMP)

    # Restricting to a fixed station set is what makes the sum comparable across
    # scans; only that set's nodes appear.
    rs = _row(ZeroSumPhase(antennas = [2, 3]))
    @test sum(rs) ≈ 1
    @test findall(!iszero, rs) == [2, 3, 6, 7]

    # A restricted set disjoint from the component would leave an empty row and a
    # rank-deficient system, so it falls back to the whole component.
    @test _row(ZeroSumPhase(antennas = [4])) ≈ _row(ZeroSumPhase())

    # Weights must not sum to zero — that is the rank-deficient case, and it fails
    # loudly rather than producing a silently unconstrained solve.
    @test_throws ErrorException _row(ZeroSumPhase(weights = zeros(2GN)))
end

@testset "gauge rows follow the solve's element type" begin
    for T in (Float32, Float64)
        @test eltype(_row(PinAntenna(2), GCOMP, T)) === T
        @test eltype(_row(ZeroSumPhase(), GCOMP, T)) === T
        @test sum(_row(ZeroSumPhase(), GCOMP, T)) ≈ one(T)
    end
end

@testset "regauge! shifts a per-station vector" begin
    # PinAntenna subtracts its reference; non-finite entries are left alone rather
    # than fabricated into a value.
    x = [1.0, 2.0, 4.0, NaN]
    CALg.regauge!(x, PinAntenna(2))
    @test x[1:3] == [-1.0, 0.0, 2.0]
    @test isnan(x[4])

    # Ranked: the first reference with a finite value wins.
    y = [1.0, NaN, 4.0, 8.0]
    CALg.regauge!(y, PinAntenna([2, 3]))
    @test y[[1, 3, 4]] == [-3.0, 0.0, 4.0]

    # ZeroSumPhase centers on the mean of the finite entries.
    z = [1.0, 2.0, 4.0, NaN]
    CALg.regauge!(z, ZeroSumPhase())
    @test sum(z[1:3]) ≈ 0 atol = 1.0e-12

    # Nothing finite to reference leaves the vector untouched.
    w = [NaN, NaN]
    CALg.regauge!(w, PinAntenna(1))
    @test all(isnan, w)
end

@testset "resolve_gauge maps station codes to indices" begin
    names = ["A1", "A2", "A3", "A4"]
    @test resolve_gauge(PinAntenna("A2"), names).refs == [2]
    @test resolve_gauge(PinAntenna(3), names).refs == [3]
    @test resolve_gauge(PinAntenna(:A4), names).refs == [4]
    # Rank order is preserved, and codes and indices may be mixed.
    @test resolve_gauge(PinAntenna(["A4", 1]), names).refs == [4, 1]
    @test resolve_gauge(ZeroSumPhase(antennas = ["A2", "A4"]), names).antennas == [2, 4]
    @test resolve_gauge(ZeroSumPhase(), names).antennas === nothing
    # A code the antenna table does not carry is an error, not a silent fallback.
    @test_throws ErrorException resolve_gauge(PinAntenna("ZZ"), names)
    @test_throws ErrorException resolve_gauge(ZeroSumPhase(antennas = ["ZZ"]), names)
end

@testset "gauge_station_order and remap_gauge" begin
    @test CALg.gauge_station_order(PinAntenna([3, 1]), GN) == [3, 1]
    # A summed gauge names no station, so a caller must choose on its own criterion.
    @test isempty(CALg.gauge_station_order(ZeroSumPhase(), GN))

    # Tying stations into representatives rewrites the gauge through the map.
    map = [1, 1, 3, 3]
    @test CALg.remap_gauge(PinAntenna([2, 4]), map).refs == [1, 3]
    @test CALg.remap_gauge(ZeroSumPhase(antennas = [2, 4]), map).antennas == [1, 3]

    @test CALg.gauge_primary(PinAntenna([3, 1])) == 3
    @test CALg.gauge_primary(ZeroSumPhase()) === nothing
end
