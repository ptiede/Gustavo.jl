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

_freedom(comp = GCOMP, T = Float64) = CALg.GaugeFreedom(;
    nodes = comp, station = _station_of.(comp), feed = _feed_of.(comp), scan = zeros(Int, length(comp)),
    component = fill((:phase, :x), length(comp)), observable = fill(:phase, length(comp)),
    direction = ones(T, length(comp)), weight = GW[comp],
)
_row(g, comp = GCOMP, T = Float64) = vec(first(CALg.gauge_constraints(g, CALg.GaugeFreedoms{T}([_freedom(comp, T)], 2GN))))

@testset "PinAntenna picks the ranked reference" begin
    # Feed 1 before feed 2, so a station's values stay referenced to one feed.
    @test _row(PinAntenna(2)) == [0, 1, 0, 0, 0, 0, 0, 0]
    @test CALg.gauge_anchor(PinAntenna(2), _freedom()) == 2

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
end

@testset "gauge rows follow the solve's element type" begin
    for T in (Float32, Float64)
        @test eltype(_row(PinAntenna(2), GCOMP, T)) === T
        @test eltype(_row(ZeroSumPhase(), GCOMP, T)) === T
        @test sum(_row(ZeroSumPhase(), GCOMP, T)) ≈ one(T)
    end
end

@testset "resolve_gauge maps station codes to indices" begin
    names = ["A1", "A2", "A3", "A4"]
    @test resolve_gauge(PinAntenna("A2"), names).refs == [2]
    @test resolve_gauge(PinAntenna(3), names).refs == [3]
    @test resolve_gauge(PinAntenna(:A4), names).refs == [4]
    # Rank order is preserved, and codes and indices may be mixed.
    @test resolve_gauge(PinAntenna(["A4", 1]), names).refs == [4, 1]
    @test resolve_gauge(ZeroSumPhase(), names) === ZeroSumPhase()
    # A code the antenna table does not carry is an error, not a silent fallback.
    @test_throws ErrorException resolve_gauge(PinAntenna("ZZ"), names)
end

@testset "gauge_station_order and remap_gauge" begin
    @test CALg.gauge_station_order(PinAntenna([3, 1])) == [3, 1]
    # A summed gauge names no station, so a caller must choose on its own criterion.
    @test isempty(CALg.gauge_station_order(ZeroSumPhase()))

    # Tying stations into representatives rewrites the gauge through the map.
    map = [1, 1, 3, 3]
    @test CALg.remap_gauge(PinAntenna([2, 4]), map).refs == [1, 3]
    @test CALg.remap_gauge(ZeroSumPhase(), map) === ZeroSumPhase()
end

struct PreferStation <: CALg.AbstractGauge
    station::Int
end
CALg.gauge_station_order(g::PreferStation) = (g.station,)

struct NoPreference <: CALg.AbstractGauge end

@testset "gauge_anchor's default" begin
    # The preferred station's feed-1 node, else its feed-2 node.
    @test CALg.gauge_anchor(PreferStation(3), _freedom()) == 3
    @test CALg.gauge_anchor(PreferStation(3), _freedom([5, 6, 7])) == 7
    # A gauge naming no station, or one absent from the freedom, anchors on the
    # best-observed node; ties go to the lowest node.
    @test CALg.gauge_anchor(NoPreference(), _freedom()) == 1
    @test CALg.gauge_anchor(PreferStation(4), _freedom()) == 1
    @test CALg.gauge_anchor(NoPreference(), _freedom([2, 5])) == 5
    @test CALg.gauge_anchor(NoPreference(), _freedom([8, 4])) == 4
end

@testset "gauge_constraints stacks one row per freedom" begin
    fs = CALg.GaugeFreedoms{Float64}([_freedom([1, 2]), _freedom([6, 7])], 2GN)
    C, d = CALg.gauge_constraints(PinAntenna(2), fs)
    @test C == [0 1 0 0 0 0 0 0; 0 0 0 0 0 1 0 0]
    @test d == [0, 0]
end

@testset "ByComponent chooses a gauge by component name" begin
    g = ByComponent((; rate = PinAntenna(2), adhoc = (; scan = ZeroSumPhase())); default = PinAntenna(1))
    @test CALg._component_choice(g, (:phase, :rate)) === g.choices.rate
    @test CALg._component_choice(g, (:phase, :adhoc, :scan)) === g.choices.adhoc.scan
    @test CALg._component_choice(g, (:phase, :mbd)) === g.default
    @test CALg._component_choice(g, ()) === g.default
    # A subtree's gauge covers every component under it.
    @test CALg._component_choice(ByComponent((; adhoc = PinAntenna(3)); default = PinAntenna(1)), (:phase, :adhoc, :scan)).refs == 3

    mixed = CALg.GaugeFreedom(;
        nodes = [1, 2], station = [1, 2], feed = [1, 1], scan = [0, 0],
        component = [(:phase, :rate), (:phase, :mbd)], observable = [:phase, :phase],
        direction = ones(2), weight = ones(2),
    )
    @test_throws "spans components with different gauges" CALg.gauge_anchor(g, mixed)

    r = resolve_gauge(ByComponent((; rate = PinAntenna("A2")); default = PinAntenna("A1")), ["A1", "A2"])
    @test r.choices.rate.refs == [2] && r.default.refs == [1]

    names = [(:rate,), (:adhoc, :scan), (:mbd,)]
    @test isnothing(CALg.check_gauge_components(g, names))
    @test_throws "no component adhoc.integration" CALg.check_gauge_components(
        ByComponent((; adhoc = (; integration = PinAntenna(2))); default = PinAntenna(1)), names,
    )
end
