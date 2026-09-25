# ── A scan group on the run's geometry, and the steps that read it ───────────
#
# `GroupTables` places each Measurement Set of a scan group on the run's
# geometry; the bandpass, adhoc and refine kernels accumulate one member at a
# time into the group's tables and fit once per group.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")
@isdefined(CAL) || const CAL = Gustavo.Calibration
@isdefined(FP) || const FP = Gustavo.Fring

_swap_baselines(ms) = Gustavo._with_layers(
    ms; baseline_antenna1_name = ms[:baseline_antenna2_name],
    baseline_antenna2_name = ms[:baseline_antenna1_name],
)

@testset "GroupTables" begin
    ps, _ = _build_fringe_ps(; nant = 4, nspw = 3, nchan = 4, ntime = 5, nscans = 2)
    geom = CAL.DataGeometry(ps)
    group = first(values(DimensionalData.groupby(ps, XRadio.ByScan())))
    rev = XRadio.ProcessingSet(OrderedDict(reverse(collect(pairs(group)))), DimensionalData.metadata(group))

    tabs = FP.GroupTables(rev, geom)
    @test issorted([first(w.chan_idx) for w in tabs.wins])
    @test [first(w.chan_idx) for w in tabs.wins] == [1, 5, 9]
    @test tabs.bl_pairs == sort(unique(reduce(vcat, (w.stations for w in tabs.wins))))
    @test tabs.feeds == [(1, 1), (1, 2), (2, 1), (2, 2)]
    @test tabs.ti == 1:5
    for m in eachindex(tabs.wins)
        w = tabs.wins[m]
        @test tabs.bl_pairs[tabs.blrow[m]] == w.stations
        @test tabs.feeds[tabs.feedrow[m]] == w.feed_order
        @test tabs.ti[tabs.tpos[m]] == w.ti_idx
    end

    members = OrderedDict(pairs(group))
    k = first(keys(members))
    members[k] = _swap_baselines(read(members[k]))
    swapped = XRadio.ProcessingSet(members, DimensionalData.metadata(group))
    @test_throws "stored in both orders" FP.GroupTables(swapped, geom)
    @test_throws "holds no Measurement Sets" FP.GroupTables(XRadio.ProcessingSet(OrderedDict{Symbol, XRadio.MeasurementSet}()), geom)
end

@testset "station positions for co-located ties" begin
    positions = [[0.0, 0.0, 0.0], [1.0e5, 0.0, 0.0], [2.0e5, 0.0, 0.0], [2.0e5 + 60.0, 0.0, 0.0]]
    ps, _ = _build_fringe_ps(; nant = 4, nspw = 1, station_positions = positions)
    geom = CAL.DataGeometry(ps)
    groups = DimensionalData.groupby(ps, XRadio.ByScan())
    @test Gustavo._station_positions(groups, geom.stations) == positions
    @test Gustavo._station_positions(groups, reverse(geom.stations)) == reverse(positions)
    @test_throws "no Measurement Set states a position for A9" Gustavo._station_positions(
        groups, [geom.stations; "A9"],
    )

    # DispersionSBDFit ties the co-located pair's dTEC to one value.
    dtec = [0.0, 3.0, -5.0, -5.0]
    psd, _ = _build_fringe_ps(;
        nant = 4, nspw = 8, nchan = 8, ref_freq = 3.0e9, spw_sep = 0.5e9, dtec,
        feed_common = true, seed = 77, station_positions = positions,
    )
    step = only(fit(DispersionSBDFit(), psd; gauge = PinAntenna(1)).steps)
    dplan = CAL._dispersion_plan(step.model, step.layout)
    leaf = CAL._component_leaf(dplan, step.θ)
    @test leaf[1, 1, 1, 1, 3] == leaf[1, 1, 1, 1, 4]
    @test isapprox(leaf[1, 1, 1, 1, 2], dtec[2] - dtec[1]; atol = 0.05)
    @test isapprox(leaf[1, 1, 1, 1, 3], dtec[3] - dtec[1]; atol = 0.05)
end

@testset "per-group steps on a ProcessingSet" begin
    rng = MersenneTwister(5)
    nant, nspw, nchan = 4, 2, 8
    bp = 0.3 .* randn(rng, nant, 2, nspw * nchan)
    ps, _ = _build_fringe_ps(; nant, nspw, nchan, ntime = 6, nscans = 3, bandpass = bp, seed = 3)
    serial = ExecutionConfig(inner_executor = SerialScheduler())
    wide = ExecutionConfig(outer_executor = DynamicScheduler(), inner_executor = DynamicScheduler(; nchunks = 4))
    for st in (Bandpass(), Bandpass(smoother = FP.PerTrackSmoother()), AdhocPhase(), DispersionSBDFit())
        a = fit(st, ps; gauge = PinAntenna(1), exec = serial)
        b = fit(st, ps; gauge = PinAntenna(1), exec = wide)
        @test a.steps[1].θ == b.steps[1].θ
        @test stage_info(a, only(keys(a))).nscans == 3
    end
    @test stage_info(fit(Bandpass(), ps; gauge = PinAntenna(1)), :bandpass).sources == ["SRC1"]
end
