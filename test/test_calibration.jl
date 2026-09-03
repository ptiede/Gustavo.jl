# Phase 1 — unified Calibration framework core.
# Standalone-runnable (`julia --project=test test/test_calibration.jl`) and
# included from runtests.jl.

using Gustavo
using Test
using LinearAlgebra
using DimensionalData: DimArray, Dim, lookup, Ti, name, dims
using Statistics: mean, median
using Random
import OffsetArrays

const UVD = Gustavo.UVData

const CAL = Gustavo.Calibration

@testset "Calibration segmentation ids" begin
    # 5 times across 2 scans; 6 channels across 2 spws.
    geom = CAL.DataGeometry(;
        times = [0.0, 0.1, 0.2, 1.0, 1.1],
        scan_of_time = [7, 7, 7, 9, 9],          # arbitrary labels → dense-ranked
        channel_freqs = collect(1.0:6.0) .* 1.0e9,
        spw_of_chan = [2, 2, 2, 5, 5, 5],
        t0 = 0.0, f0 = 3.5e9,
    )

    @test CAL.time_segment_ids(CAL.GlobalTime(), geom) == (fill(1, 5), 1)
    @test CAL.time_segment_ids(CAL.PerIntegration(), geom) == (collect(1:5), 5)
    @test CAL.time_segment_ids(CAL.PerScan(), geom) == ([1, 1, 1, 2, 2], 2)

    # TimeBlocks of 0.5 h: t=0,0.1,0.2 → block 0; t=1.0,1.1 → block 2 → dense-rank 1,2.
    ids, n = CAL.time_segment_ids(CAL.TimeBlocks(0.5), geom)
    @test ids == [1, 1, 1, 2, 2] && n == 2

    # InstrumentScans boundary at 0.5 h splits the same way.
    @test CAL.time_segment_ids(CAL.InstrumentScans([0.5]), geom) == ([1, 1, 1, 2, 2], 2)

    @test CAL.freq_segment_ids(CAL.GlobalFrequency(), geom) == (fill(1, 6), 1)
    @test CAL.freq_segment_ids(CAL.PerSpectralWindow(), geom) == ([1, 1, 1, 2, 2, 2], 2)
    # ChannelBlocks(2) within each spw: spw1 → [1,1,2], spw2 → [3,3,4].
    @test CAL.freq_segment_ids(CAL.ChannelBlocks(2), geom) == ([1, 1, 2, 3, 3, 4], 4)
    # ChannelBlocks never straddles a spw: block_size 4 still splits at the spw edge.
    @test CAL.freq_segment_ids(CAL.ChannelBlocks(4), geom) == ([1, 1, 1, 2, 2, 2], 2)

    @test CAL.segment_groups([1, 1, 2, 3, 3, 4], 4) == [[1, 2], [3], [4, 5], [6]]

    # A name vector names every segment or none; a partial one would silently
    # label the wrong segment when a foreign grid is placed against it.
    @test_throws "must name every segment or be empty" CAL.DataGeometry(;
        times = [0.0, 1.0], scan_of_time = [1, 2], channel_freqs = [1.0e9],
        scan_names = ["only-one"],
    )
end

@testset "Materialization and segment_ranges" begin
    # 4 channels: a gap-separated pair of groups, in 3 spws.
    geom = CAL.DataGeometry(;
        times = [0.0, 1.0],
        channel_freqs = [1.0e9, 1.1e9, 1.2e9, 5.0e9],
        spw_of_chan = [1, 1, 2, 3],
    )

    # Concrete segmentations materialize to themselves; the data-dependent
    # `BandGroups` resolves to the `FreqGroups` its gap detection finds.
    @test CAL.BandGroups() isa CAL.AbstractFrequencySegmentation
    for seg in (
            CAL.GlobalFrequency(), CAL.PerSpectralWindow(), CAL.ChannelBlocks(2),
            CAL.FreqGroups([1:3, 4:4]),
        )
        @test CAL.materialize(seg, geom) === seg
    end
    @test CAL.materialize(CAL.BandGroups(), geom) == CAL.FreqGroups([1:3, 4:4])
    # `gap_factor` reaches the gap detection: an unreachable ratio keeps one group.
    @test CAL.materialize(CAL.BandGroups(gap_factor = 1.0e9), geom) ==
        CAL.FreqGroups([1:4])

    # The range form of each shipped concrete segmentation.
    @test CAL.segment_ranges(CAL.GlobalFrequency(), geom) == [1:4]
    @test CAL.segment_ranges(CAL.PerSpectralWindow(), geom) == [1:2, 3:3, 4:4]
    @test CAL.segment_ranges(CAL.ChannelBlocks(2), geom) == [1:2, 3:3, 4:4]
    @test CAL.segment_ranges(CAL.FreqGroups([1:1, 2:4]), geom) == [1:1, 2:4]

    # A segmentation whose segments interleave has no range form.
    inter = CAL.DataGeometry(;
        times = [0.0], channel_freqs = [1.0e9, 1.1e9, 1.2e9], spw_of_chan = [1, 2, 1],
    )
    @test_throws "no contiguous channel range" CAL.segment_ranges(
        CAL.PerSpectralWindow(), inter
    )

    # Layouts materialize: a plan built from a `BandGroups` component records the
    # concrete `FreqGroups`, so solutions and foreign-grid placement never see
    # the data-dependent form.
    model = CAL.StationGainModel(
        phase = (
            sbd = CAL.GainComponent(
                CAL.Delay(); Ti = CAL.PerScan(), Frequency = CAL.BandGroups(),
                Feed = CAL.SharedFeeds(),
            ),
        ),
    )
    layout = CAL.plan_parameters(model, 3, geom)
    plan = only(layout.plans)
    @test plan.fseg == CAL.FreqGroups([1:3, 4:4])
    @test plan.fseg_id == [1, 1, 1, 2]
end

@testset "Placement on a foreign grid" begin
    # Solve grid: 2 scans × 2 epochs, 2 spws × 3 channels.
    solve = CAL.DataGeometry(;
        times = [0.0, 0.1, 1.0, 1.1], scan_of_time = [7, 7, 9, 9],
        channel_freqs = [1.0, 1.1, 1.2, 2.0, 2.1, 2.2] .* 1.0e9,
        spw_of_chan = [3, 3, 3, 4, 4, 4],
        scan_names = ["No001", "No002"], spw_names = ["A", "B"],
    )
    # The same scans and spws, sampled three times as finely in time.
    fine = CAL.DataGeometry(;
        times = [0.0, 0.05, 0.1, 1.0, 1.05, 1.1], scan_of_time = [1, 1, 1, 2, 2, 2],
        channel_freqs = solve.channel_freqs, spw_of_chan = solve.spw_of_chan,
        scan_names = ["No001", "No002"], spw_names = ["A", "B"],
    )

    @testset "a coarser solution covers every finer sample" begin
        @test CAL.time_segment_ids(CAL.GlobalTime(), solve, fine) == fill(1, 6)
        @test CAL.time_segment_ids(CAL.PerScan(), solve, fine) == [1, 1, 1, 2, 2, 2]
        @test CAL.freq_segment_ids(CAL.GlobalFrequency(), solve, fine) == fill(1, 6)
        @test CAL.freq_segment_ids(CAL.PerSpectralWindow(), solve, fine) == [1, 1, 1, 2, 2, 2]
        # A window places exactly the samples it selects, in their order.
        @test CAL.time_segment_ids(CAL.PerScan(), solve, fine; ti_idx = [5, 2]) == [2, 1]
        @test CAL.freq_segment_ids(CAL.PerSpectralWindow(), solve, fine; chan_idx = 4:6) == [2, 2, 2]

        # A scan-averaged solve — one epoch per scan — applies at full time
        # resolution: the scan name says which segment, not the epoch.
        averaged = CAL.DataGeometry(;
            times = [0.05, 1.05], scan_of_time = [7, 9],
            channel_freqs = solve.channel_freqs, spw_of_chan = solve.spw_of_chan,
            scan_names = ["No001", "No002"], spw_names = ["A", "B"],
        )
        @test CAL.time_segment_ids(CAL.PerScan(), averaged, fine) == [1, 1, 1, 2, 2, 2]
    end

    @testset "a scan or spw the solve never saw errors" begin
        stranger = CAL.DataGeometry(;
            times = [0.0, 2.0], scan_of_time = [1, 2],
            channel_freqs = solve.channel_freqs, spw_of_chan = solve.spw_of_chan,
            scan_names = ["No001", "No009"], spw_names = ["A", "B"],
        )
        # Named, not borrowed from the neighbouring scan it sits closest to.
        @test_throws "PerScan" CAL.time_segment_ids(CAL.PerScan(), solve, stranger)
        @test_throws "\"No009\" is not in the solution" CAL.time_segment_ids(
            CAL.PerScan(), solve, stranger,
        )
        other_band = CAL.DataGeometry(;
            times = solve.times, scan_of_time = solve.scan_of_time,
            channel_freqs = solve.channel_freqs, spw_of_chan = [1, 1, 1, 2, 2, 2],
            scan_names = ["No001", "No002"], spw_names = ["A", "C"],
        )
        @test_throws "\"C\" is not in the solution" CAL.freq_segment_ids(
            CAL.PerSpectralWindow(), solve, other_band,
        )
    end

    @testset "identity placement needs names, not raw ids" begin
        # Raw ids agree with themselves, so an identical labelling still places:
        # the two grids then correspond sample for sample and no name is needed.
        unnamed = CAL.DataGeometry(;
            times = solve.times, scan_of_time = solve.scan_of_time,
            channel_freqs = solve.channel_freqs, spw_of_chan = solve.spw_of_chan,
        )
        @test CAL.time_segment_ids(CAL.PerScan(), unnamed, unnamed) == [1, 1, 2, 2]
        @test CAL.freq_segment_ids(CAL.PerSpectralWindow(), unnamed, unnamed) == [1, 1, 1, 2, 2, 2]

        # Across two differently-labelled grids they do not: matching id 1 to
        # id 1 is positional matching, which identity placement exists to avoid.
        unnamed_fine = CAL.DataGeometry(;
            times = fine.times, scan_of_time = fine.scan_of_time,
            channel_freqs = fine.channel_freqs, spw_of_chan = [1, 1, 1, 2, 2, 2],
        )
        @test_throws "SOLUTION's geometry carries no scan names" CAL.time_segment_ids(
            CAL.PerScan(), unnamed, fine,
        )
        @test_throws "TARGET geometry carries no scan names" CAL.time_segment_ids(
            CAL.PerScan(), solve, unnamed_fine,
        )
        @test_throws "SOLUTION's geometry carries no spectral-window names" CAL.freq_segment_ids(
            CAL.PerSpectralWindow(), unnamed, unnamed_fine,
        )
    end

    @testset "formula rows: raw bins map through the solve's own dense ranking" begin
        # Solve epochs in blocks 0 and 2 — block 1 is never populated, so it has
        # no segment id at all and block 2's parameters live at id 2, not 3.
        gapped = CAL.DataGeometry(; times = [0.0, 0.1, 2.0, 2.1], channel_freqs = [1.0e9])
        seg = CAL.TimeBlocks(1.0)
        @test CAL.time_segment_ids(seg, gapped) == ([1, 1, 2, 2], 2)
        either_side = CAL.DataGeometry(; times = [0.5, 2.5], channel_freqs = [1.0e9])
        @test CAL.time_segment_ids(seg, gapped, either_side) == [1, 2]
        inside = CAL.DataGeometry(; times = [1.5], channel_freqs = [1.0e9])
        @test_throws "no solve epoch populated" CAL.time_segment_ids(seg, gapped, inside)

        # The solve's own block origin is used, never the target's: a target
        # starting an hour later must still land in the solve's blocks.
        later = CAL.DataGeometry(; times = [2.05], channel_freqs = [1.0e9])
        @test CAL.time_segment_ids(seg, gapped, later) == [2]

        # A sample integrating ACROSS a block boundary has no single segment.
        @test_throws "crosses a segment boundary" CAL.time_segment_ids(
            seg, gapped, either_side; time_span = [1.2, 0.1],
        )
        iscans = CAL.InstrumentScans([1.0])
        @test CAL.time_segment_ids(iscans, gapped, either_side) == [1, 2]
        @test_throws "InstrumentScans" CAL.time_segment_ids(
            iscans, gapped, either_side; time_span = [1.2, 0.1],
        )
    end

    @testset "PerIntegration: exact epochs, and averaging caught by the span" begin
        aps = CAL.DataGeometry(; times = [1.0, 2.0, 3.0], channel_freqs = [1.0e9])
        @test CAL.time_segment_ids(CAL.PerIntegration(), aps, aps) == [1, 2, 3]
        # Untouched data carries a span narrower than the AP spacing.
        @test CAL.time_segment_ids(CAL.PerIntegration(), aps, aps; time_span = fill(0.9, 3)) ==
            [1, 2, 3]

        # A near miss names both epochs, so a unit slip is diagnosable at a glance.
        off = CAL.DataGeometry(; times = [2.25], channel_freqs = [1.0e9])
        @test_throws "2.25" CAL.time_segment_ids(CAL.PerIntegration(), aps, off)
        @test_throws "nearest is 2.0" CAL.time_segment_ids(CAL.PerIntegration(), aps, off)

        # Averaging {1, 2, 3} h lands exactly ON a solve epoch, so the epoch
        # match alone would accept it; the span it now carries does not.
        avg = CAL.DataGeometry(; times = [2.0], channel_freqs = [1.0e9])
        @test CAL.time_segment_ids(CAL.PerIntegration(), aps, avg) == [2]
        @test_throws "covering solve epochs 1.0, 2.0, 3.0" CAL.time_segment_ids(
            CAL.PerIntegration(), aps, avg; time_span = [2.5],
        )
    end

    @testset "channel-index segmentations require the same channel layout" begin
        narrow = CAL.DataGeometry(;
            times = solve.times, scan_of_time = solve.scan_of_time,
            channel_freqs = solve.channel_freqs[1:4], spw_of_chan = solve.spw_of_chan[1:4],
        )
        @test_throws "target has 4 channels and the solution 6" CAL.freq_segment_ids(
            CAL.ChannelBlocks(1), solve, narrow,
        )
        shifted = CAL.DataGeometry(;
            times = solve.times, scan_of_time = solve.scan_of_time,
            channel_freqs = solve.channel_freqs .+ 1.0e6, spw_of_chan = solve.spw_of_chan,
        )
        # The near miss prints both frequencies, so a unit slip or a shifted
        # correlator setup is diagnosable at a glance.
        @test_throws "target channel 1 is at 1.001e9 Hz" CAL.freq_segment_ids(
            CAL.ChannelBlocks(2), solve, shifted,
        )
        @test_throws "solution's at 1.0e9 Hz" CAL.freq_segment_ids(
            CAL.ChannelBlocks(2), solve, shifted,
        )
        @test CAL.freq_segment_ids(CAL.ChannelBlocks(2), solve, solve) == [1, 1, 2, 3, 3, 4]
        groups = CAL.FreqGroups([1:3, 4:6])
        @test CAL.freq_segment_ids(groups, solve, solve) == [1, 1, 1, 2, 2, 2]
        @test_throws "FreqGroups" CAL.freq_segment_ids(groups, solve, narrow)
    end

    @testset "gains evaluate on the foreign grid in the solve's own basis" begin
        model = CAL.StationGainModel(
            phase = (
                atmos = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),
                mbd = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),
            ),
        )
        nant = 2
        ev = CAL.GainEvaluator(model, solve; nant)
        θ = collect(range(0.1; step = 0.05, length = ev.layout.nθ))
        θ[(end ÷ 2 + 1):end] .*= 1.0e-9         # the delay block, in seconds

        # Placing the solve grid against itself reproduces the index form
        # exactly — same-grid apply is the degenerate case, not a separate path.
        @test CAL.evaluate_gains(ev, θ, solve, solve) == CAL.evaluate_gains(ev, θ)

        # On the finer grid, each sample takes its own scan's parameters, and
        # the delay still reads (f − f0) against the SOLUTION's f0.
        gf = CAL.evaluate_gains(ev, θ, solve, fine)
        gs = CAL.evaluate_gains(ev, θ)
        @test size(gf) == (6, 6, nant, 2)
        for (k, ti) in enumerate([1, 1, 2, 3, 3, 4])   # fine sample → a solve sample in its scan
            @test gf[:, k, :, :] ≈ gs[:, ti, :, :]
        end

        # A window of the foreign grid is the same gains, sliced.
        @test CAL.evaluate_gains(ev, θ, solve, fine; chan_idx = 2:4, ti_idx = [2, 5]) ≈
            gf[2:4, [2, 5], :, :]

        # A scan the solution never saw is refused, naming the segmentation.
        stranger = CAL.DataGeometry(;
            times = [5.0], scan_of_time = [1],
            channel_freqs = solve.channel_freqs, spw_of_chan = solve.spw_of_chan,
            scan_names = ["No042"], spw_names = ["A", "B"],
        )
        @test_throws "PerScan" CAL.evaluate_gains(ev, θ, solve, stranger)
    end
end

@testset "Calibration terms: names, nparams, eval" begin
    # A term names its parameters and shapes them; the count is derived, so it
    # cannot disagree with the names.
    @test CAL.param_shapes(CAL.Delay(), 7) == (delay = (),)
    @test CAL.param_shapes(CAL.PolynomialFreq(3), 7) == (coeffs = (3,),)

    # One `Polynomial` term; the axis it reads is a type parameter, and the
    # convenience constructors are functions returning it.
    @test CAL.PolynomialFreq(3) isa CAL.Polynomial{:Frequency}
    @test CAL.PolynomialTime(3) isa CAL.Polynomial{:Ti}
    @test CAL.term_axes(CAL.PolynomialFreq(3)) == (:Frequency,)
    @test CAL.term_axes(CAL.PolynomialTime(3)) == (:Ti,)
    @test CAL.term_label(CAL.PolynomialFreq(3)) == "polyf3"
    @test CAL.term_label(CAL.PolynomialTime(2)) == "polyt2"

    # `term_label` defaults to the type name, so a term author only needs to
    # override it for a more evocative label.
    @eval CAL struct _UnlabeledTerm <: AbstractGainTerm end
    @test CAL.term_label(CAL._UnlabeledTerm()) == "_UnlabeledTerm"

    @test_throws ArgumentError CAL.PolynomialFreq(0)
    @test_throws "axis must be one of" CAL.Polynomial{:nope}(2)

    # A term is handed the axes it declares, under the dimension names, and a
    # channel index is never one of them.
    @test CAL.term_axes(CAL.Delay()) == (:Frequency,)
    @test CAL.term_axes(CAL.Rate()) == (:Ti,)
    @test CAL.term_axes(CAL.ConstantTerm()) == ()

    # A block's size never depends on how many channels its segment holds: how
    # finely a term varies in frequency is said by the segmentation.
    @test CAL.nparams_per_block(CAL.ConstantTerm(), 7) == 1
    @test CAL.nparams_per_block(CAL.Delay(), 7) == 1
    @test CAL.nparams_per_block(CAL.Rate(), 7) == 1
    @test CAL.nparams_per_block(CAL.PolynomialFreq(3), 7) == 3
    @test CAL.nparams_per_block(CAL.PolynomialTime(2), 7) == 2

    # scalar term_eval primitives: a term sees its own named parameters and the
    # coordinates `term_axes` declares, under those names.
    @test CAL.term_eval(CAL.ConstantTerm(), (offset = 1.5,), NamedTuple()) == 1.5
    @test CAL.term_eval(CAL.Delay(), (delay = 0.3,), (Frequency = 4.0,)) ≈ 2π * 0.3 * 4.0
    @test CAL.term_eval(CAL.Rate(), (rate = 0.3,), (Ti = 5.0,)) ≈ 2π * 0.3 * 5.0
    @test CAL.term_eval(CAL.Dispersion(), (dtec = 0.3,), (Frequency = 4.0,)) ≈ 0.3 * 4.0
    @test CAL.term_eval(
        CAL.PolynomialFreq(3), (coeffs = [0.3, 1.5, -0.7],), (Frequency = 2.0,)
    ) ≈ 0.3 * 2 + 1.5 * 4 + (-0.7) * 8
    @test CAL.term_eval(
        CAL.PolynomialTime(3), (coeffs = [0.3, 1.5, -0.7],), (Ti = 2.0,)
    ) ≈ 0.3 * 2 + 1.5 * 4 + (-0.7) * 8

    # The named parameters of a block are addressed over the block's own view.
    θ = [0.0, 0.3, 1.5, -0.7]
    @test CAL._block_params(CAL.param_shapes(CAL.Delay(), 1), view(θ, 2:2)) == (delay = 0.3,)
    @test CAL._block_params(CAL.param_shapes(CAL.PolynomialFreq(2), 1), view(θ, 3:4)).coeffs ==
        [1.5, -0.7]
    # Declaration order, and each name gets exactly the size it declared.
    shapes = (a = (), b = (2,), c = ())
    p = CAL._block_params(shapes, θ)
    @test p.a == 0.0 && p.b == [0.3, 1.5] && p.c == -0.7

    # `basis_columns` is gone: WLS solvers build their own systems.
    @test !isdefined(CAL, :basis_columns)
end

@testset "GainComponent: construction, label, equality" begin
    # The single vocabulary: the flat `GainComponent` replaces the nested
    # TiedComponent wrapper.
    @test :GainComponent in names(CAL)
    @test !isdefined(CAL, :TiedComponent)

    # The keyword form is the one public spelling; `feeds` defaults to PerFeed.
    e = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency())
    @test e.term isa CAL.Delay
    @test e.Ti isa CAL.PerScan
    @test e.Frequency isa CAL.GlobalFrequency
    @test e.Feed isa CAL.PerFeed

    # `component_label` prints the constructor call, and the printed form
    # evaluates back to an equal GainComponent.
    lbl = CAL.component_label(
        CAL.GainComponent(
            CAL.Delay(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(),
            Feed = CAL.SharedFeeds(),
        )
    )
    @test lbl ==
        "GainComponent(Delay(); Ti = PerScan(), Frequency = GlobalFrequency(), Feed = SharedFeeds())"
    @test Core.eval(CAL, Meta.parse(lbl)) == CAL.GainComponent(
        CAL.Delay(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(),
        Feed = CAL.SharedFeeds(),
    )
    # Types whose public constructor is not `TypeName(fields...)` print their
    # public form; parameterized fields print pasteable values.
    for (x, want) in (
            (CAL.PolynomialFreq(2), "PolynomialFreq(2)"),
            (CAL.PolynomialTime(3), "PolynomialTime(3)"),
            (CAL.TimeBlocks(1.5), "TimeBlocks(1.5)"),
            (CAL.ChannelBlocks(4), "ChannelBlocks(4)"),
            (CAL.ReferenceRelative(1), "ReferenceRelative(1)"),
            (CAL.SingleFeed(2), "SingleFeed(2)"),
        )
        @test CAL._call_string(x) == want
    end

    # Value equality reaches through Vector-holding fields: two independently
    # built FreqGroups (and components/models holding them) compare and hash equal.
    fg1 = CAL.FreqGroups([1:4, 5:8])
    fg2 = CAL.FreqGroups([1:4, 5:8])
    @test fg1 == fg2 && hash(fg1) == hash(fg2)
    @test fg1 != CAL.FreqGroups([1:2, 3:8])
    mk(fg) = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Frequency = fg, Feed = CAL.SharedFeeds())
    @test mk(fg1) == mk(fg2) && hash(mk(fg1)) == hash(mk(fg2))
    @test mk(fg1) != CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Frequency = fg1)  # Feed differs
    @test mk(fg1) != CAL.GainComponent(CAL.Rate(); Ti = CAL.PerScan(), Frequency = fg1, Feed = CAL.SharedFeeds())
    m1 = CAL.StationGainModel(phase = (sbd = mk(fg1),))
    m2 = CAL.StationGainModel(phase = (sbd = mk(fg2),))
    @test m1 == m2 && hash(m1) == hash(m2)
    @test m1 != CAL.StationGainModel(logamp = (sbd = mk(fg1),))

    # Unnamed components are rejected with the constructor the message shows.
    @test_throws "pass a NamedTuple" CAL.StationGainModel(phase = (mk(fg1),))
end

@testset "Calibration feed tying offset algebra" begin
    geom = CAL.DataGeometry(; times = [0.0, 1.0], channel_freqs = [1.0e9, 2.0e9])
    # One ConstantTerm, GlobalTime × GlobalFrequency, 2 antennas.
    mk(tying) = CAL.StationGainModel(
        phase = (c = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = tying),),
    )

    lay_pf = CAL.plan_parameters(mk(CAL.PerFeed()), 2, geom)
    @test lay_pf.nθ == 4                                    # 2 ant × 2 feeds
    p = lay_pf.plans[1]
    @test plan_off1(p)[1, 1, 1, 1] != plan_off1(p)[1, 2, 1, 1]          # feeds independent
    @test all(plan_off2(p) .== 0)

    lay_sf = CAL.plan_parameters(mk(CAL.SharedFeeds()), 2, geom)
    @test lay_sf.nθ == 2                                    # 2 ant, shared across feeds
    q = lay_sf.plans[1]
    @test plan_off1(q)[1, 1, 1, 1] == plan_off1(q)[1, 2, 1, 1]          # feeds share a block
    @test all(plan_off2(q) .== 0)

    lay_rr = CAL.plan_parameters(mk(CAL.ReferenceRelative(1)), 2, geom)
    @test lay_rr.nθ == 4                                    # ref + relative per ant
    r = lay_rr.plans[1]
    @test plan_off1(r)[1, 1, 1, 1] == plan_off1(r)[1, 2, 1, 1]          # both feeds reference the ref block
    @test plan_off2(r)[1, 1, 1, 1] == 0                           # reference feed has no relative
    @test plan_off2(r)[1, 2, 1, 1] != 0                           # partner feed adds a relative block
end

@testset "Calibration ComponentVector template" begin
    freqs = [1.0e9, 2.0e9, 3.0e9]                    # 3 channels
    geom = CAL.DataGeometry(; times = [0.0, 1.0], channel_freqs = freqs)
    nant = 2
    model = CAL.StationGainModel(
        phase = (
            a = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),
            bp = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.ChannelBlocks(1), Feed = CAL.SharedFeeds()),
            grp = (                                  # one element compiling to a nested subtree
                d = CAL.GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),
                c = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),
            ),
        ),
    )
    layout = CAL.plan_parameters(model, nant, geom)
    θ = Float64.(1:layout.nθ)
    cv = CAL.component_vector(layout, θ)

    # The named/shaped view spans exactly the flat θ (one source of truth) and
    # nests where the model does.
    @test length(layout.template) == layout.nθ
    @test propertynames(cv) == (:phase, :logamp)
    @test propertynames(cv.phase) == (:a, :bp, :grp)
    @test propertynames(cv.phase.grp) == (:d, :c)

    # Each leaf is the full-rank shape its tying × segmentation imply
    # (param, feed-node, freq-seg, time-seg, ant), size-1 axes kept.
    @test size(cv.phase.a) == (1, 2, 1, 1, nant)                 # PerFeed: two feed nodes
    @test size(cv.phase.bp) == (1, 1, length(freqs), 1, nant)    # ChannelBlocks(1): one freq-seg per channel
    @test size(cv.phase.grp.d) == (1, 1, 1, 1, nant)             # a nested part is a plain leaf
    @test size(cv.phase.grp.c) == (1, 1, 1, 1, nant)

    # The stored axis roles name each dimension.
    @test layout.axes.phase.a.roles == (:param, :Feed, :Frequency, :Ti, :Ant)
    @test layout.axes.phase.bp.roles == (:param, :node, :Frequency, :Ti, :Ant)
    @test layout.axes.phase.grp.d.roles == (:param, :node, :Frequency, :Ti, :Ant)

    # A named leaf is exactly the component's θ block (matches component_ranges:
    # phase components depth-first — a, bp, grp.d, grp.c).
    rng = CAL.component_ranges(layout)
    @test vec(cv.phase.a) == θ[rng[1]]
    @test vec(cv.phase.bp) == θ[rng[2]]
    @test vec(cv.phase.grp.d) == θ[rng[3]]
    @test vec(cv.phase.grp.c) == θ[rng[4]]

    # The wrap shares data (no copy), and the forward map reads it identically to
    # the flat vector.
    ev = CAL.GainEvaluator(model, layout)
    @test CAL.evaluate_gains(ev, cv) == CAL.evaluate_gains(ev, θ)
    cv[1] = -99.0
    @test θ[1] == -99.0
end

@testset "component_dimarray: a component leaf as a labelled DimArray" begin
    freqs = [1.0e9, 2.0e9, 3.0e9, 4.0e9]
    times = [0.0, 1.0, 2.0, 3.0]
    geom = CAL.DataGeometry(;
        times, channel_freqs = freqs,
        scan_of_time = [1, 1, 2, 2], spw_of_chan = [1, 1, 1, 1], t0 = 0.0, f0 = 2.5e9,
    )
    nant = 3
    ants = ["PT", "LM", "AA"]
    model = CAL.StationGainModel(
        phase = (
            atmos = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),
            bp = CAL.GainComponent(CAL.PolynomialFreq(2); Ti = CAL.GlobalTime(), Frequency = CAL.ChannelBlocks(2), Feed = CAL.SharedFeeds()),
            rl = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.ReferenceRelative(1)),
        ),
        logamp = (
            amp = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),
        ),
    )
    layout = CAL.plan_parameters(model, nant, geom)
    θ = Float64.(1:layout.nθ)
    sol = CAL.CalibrationSolution(model, layout, geom, θ, (; ant_names = ants))

    # A leaf's DimArray is shaped and rolled exactly as the layout stored it.
    # The step (`:solution`, the single-model constructor's default) is always
    # explicit: components are addressed within one step, never searched for
    # by a bare name across steps.
    a = CAL.component_dimarray(sol, :solution, :phase, :atmos)
    @test a isa DimArray
    @test size(a) == layout.axes.phase.atmos.dims
    @test name.(dims(a)) == layout.axes.phase.atmos.roles
    @test name(a) == :atmos

    # PerFeed carries a physical Feed axis; the tied tyings carry a positional
    # node axis (ReferenceRelative: reference + relative, two nodes).
    @test name(dims(a, 2)) == :Feed
    @test name(dims(CAL.component_dimarray(sol, :solution, :phase, :bp), 2)) == :node
    @test size(CAL.component_dimarray(sol, :solution, :phase, :rl), 2) == 2

    # The same lookup by step POSITION (not name) reaches the same leaf.
    @test CAL.component_dimarray(sol, 1, :phase, :atmos) == a

    # Segment axes carry a representative physical coordinate per segment: a
    # frequency segment's centre, a time segment's mean epoch.
    @test lookup(a, UVD.Frequency) == [mean(freqs)]              # GlobalFrequency: one centre
    @test lookup(a, Ti) == [mean(times[1:2]), mean(times[3:4])]  # PerScan: per-scan mean epoch
    @test lookup(CAL.component_dimarray(sol, :solution, :phase, :bp), UVD.Frequency) ==
        [mean(freqs[1:2]), mean(freqs[3:4])]                     # ChannelBlocks(2): block centres

    # Antennas take the solution's station names; feed is 1:2.
    @test lookup(a, UVD.Ant) == ants
    @test lookup(a, UVD.Feed) == 1:2

    # logamp descends the same way.
    @test CAL.component_dimarray(sol, :solution, :logamp, :amp) isa DimArray

    # The leaf is a view onto θ (no copy): its data is the component's block, and
    # writing through it mirrors into θ.
    rng = CAL.component_ranges(layout)
    @test vec(parent(a)) == θ[rng[1]]
    a[1, 1, 1, 1, 1] = -7.0
    @test sol.steps[1].θ[first(rng[1])] == -7.0

    # No ant_names in info → the antenna axis falls back to 1:nant.
    soln = CAL.CalibrationSolution(model, layout, geom, θ, (; nant))
    @test lookup(CAL.component_dimarray(soln, :solution, :phase, :atmos), UVD.Ant) == 1:nant

    # A multi-emit wrapper's nested subtree is reached leaf by leaf, the shape
    # SingleBandDelay compiles to (`sbd.delay` / `sbd.constant`).
    gb = CAL.DataGeometry(;
        times = [0.0, 1.0], channel_freqs = [1.0e9, 1.1e9, 5.0e9, 5.1e9],
        scan_of_time = [1, 1], spw_of_chan = [1, 1, 2, 2], t0 = 0.0, f0 = 3.0e9,
    )
    sbd = CAL.model_components(SingleBandDelay(), gb)
    msbd = CAL.StationGainModel(phase = (sbd = sbd,))
    lsbd = CAL.plan_parameters(msbd, 2, gb)
    ssbd = CAL.CalibrationSolution(msbd, lsbd, gb, Float64.(1:lsbd.nθ), (;))
    dl = CAL.component_dimarray(ssbd, :solution, :phase, :sbd, :delay)
    @test dl isa DimArray
    @test name(dl) == :delay
    @test lookup(dl, UVD.Frequency) == [mean([1.0e9, 1.1e9]), mean([5.0e9, 5.1e9])]
    @test vec(parent(dl)) == ssbd.steps[1].θ[lsbd.plantree.phase.sbd.delay.range]

    # A path that stops at a group is rejected.
    @test_throws ArgumentError CAL.component_dimarray(ssbd, :solution, :phase, :sbd)
    @test_throws "names a component group" CAL.component_dimarray(ssbd, :solution, :phase, :sbd)

    # An unknown step NAME needs the recorded stages spelled out; an
    # out-of-range index does not — `BoundsError` already says it.
    @test_throws ArgumentError CAL.component_dimarray(sol, :nosuchstep, :phase, :atmos)
    @test_throws "recorded stages: [:solution]" CAL.component_dimarray(sol, :nosuchstep, :phase, :atmos)
    @test_throws BoundsError CAL.component_dimarray(sol, 2, :phase, :atmos)

    # Component names are local to each step and may repeat across steps —
    # splitting a solve into steps is exactly what makes that legal, so
    # `component_dimarray` must resolve by step, never by searching for a name
    # across steps.
    atmos2 = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds())
    model2 = CAL.StationGainModel(phase = (atmos = atmos2,))
    layout2 = CAL.plan_parameters(model2, nant, geom)
    θ2 = fill(-1.0, layout2.nθ)
    twostep = CAL.CalibrationSolution(
        [
            CAL.StepSolution(:bandpass, model, layout, θ, (;)),
            CAL.StepSolution(:fringe, model2, layout2, θ2, (;)),
        ],
        geom, (; ant_names = ants),
    )
    @test vec(parent(CAL.component_dimarray(twostep, :bandpass, :phase, :atmos))) == θ[rng[1]]
    @test all(==(-1.0), CAL.component_dimarray(twostep, :fringe, :phase, :atmos))
    @test CAL.component_dimarray(twostep, 1, :phase, :atmos) == CAL.component_dimarray(twostep, :bandpass, :phase, :atmos)
    @test CAL.component_dimarray(twostep, 2, :phase, :atmos) == CAL.component_dimarray(twostep, :fringe, :phase, :atmos)
end

@testset "Calibration evaluate_gains: correctness, purity, inference" begin
    nant = 3
    freqs = collect(2.28e11:1.0e8:(2.28e11 + 5.0e8))         # 6 channels
    f0 = sum(freqs) / length(freqs)
    times = [0.0, 0.5, 1.0]
    geom = CAL.DataGeometry(; times, channel_freqs = freqs, t0 = 0.0, f0)

    # Pure per-feed delay model: phase = 2π τ (f − f0), one τ per (ant, feed).
    model = CAL.StationGainModel(
        phase = (delay = CAL.GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),),
    )
    ev = CAL.GainEvaluator(model, geom; nant)
    @test CAL.nparameters(ev) == nant * 2

    τ = 1.0e-9 .* collect(1:(nant * 2))                    # distinct delays in ns
    g = CAL.evaluate_gains(ev, τ)
    @test size(g) == (length(freqs), length(times), nant, 2)

    # Hand-check a couple of cells: |g| == 1 (no amplitude), phase = 2π τ (f−f0).
    p = ev.layout.plans[1]
    for ant in 1:nant, feed in 1:2, ci in eachindex(freqs)
        off = plan_off1(p)[ant, feed, 1, 1]
        expected = cis(2π * τ[off] * (freqs[ci] - f0))
        @test g[ci, 1, ant, feed] ≈ expected
        @test g[ci, 2, ant, feed] ≈ expected            # time-invariant (GlobalTime)
    end

    # Purity: θ untouched, repeated evaluation bit-identical, no aliasing.
    τ_copy = copy(τ)
    g2 = CAL.evaluate_gains(ev, τ)
    @test τ == τ_copy
    @test g == g2
    @test g !== g2

    # Type stability of the forward map.
    @inferred CAL.evaluate_gains(ev, τ)

    # Rate term: phase grows linearly in time, flat in frequency, about the
    # segment's OWN mean epoch (one segment here, so the whole track's).
    rate_model = CAL.StationGainModel(
        phase = (rate = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),),
    )
    evr = CAL.GainEvaluator(rate_model, geom; nant)
    ṙ = 1.0e-3 .* collect(1:nant)                          # mHz-scale rates
    gr = CAL.evaluate_gains(evr, ṙ)
    pr = evr.layout.plans[1]
    @test pr.tstate ≈ [sum(times) / length(times)]
    for ant in 1:nant, ti in eachindex(times)
        off = plan_off1(pr)[ant, 1, 1, 1]
        expected = cis(2π * ṙ[off] * (times[ti] - pr.tstate[1]) * 3600.0)
        @test gr[1, ti, ant, 1] ≈ expected
        @test gr[1, ti, ant, 2] ≈ expected               # shared across feeds
    end
end

@testset "Calibration predict_visibilities closes" begin
    nant = 3
    geom = CAL.DataGeometry(; times = [0.0], channel_freqs = [2.28e11, 2.281e11], t0 = 0.0)
    model = CAL.StationGainModel(
        phase = (offset = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),),
        logamp = (offset = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),),
    )
    ev = CAL.GainEvaluator(model, geom; nant)
    θ = randn(CAL.nparameters(ev))
    g = CAL.evaluate_gains(ev, θ)

    bl_pairs = [(1, 2), (1, 3), (2, 3)]
    bl_a = first.(bl_pairs)
    bl_b = last.(bl_pairs)
    # Parallel + cross hands: PP, PQ, QP, QQ → feed pairs.
    pol_products = ["PP", "PQ", "QP", "QQ"]
    feed_a = [CAL.correlation_feed_pair(p)[1] for p in pol_products]
    feed_b = [CAL.correlation_feed_pair(p)[2] for p in pol_products]

    coh = zeros(ComplexF64, length(bl_pairs), 2, 2)
    for bi in eachindex(bl_pairs)
        coh[bi, :, :] .= ComplexF64[1.0 0.1; 0.1 0.9]
    end
    V = CAL.predict_visibilities(g, coh, bl_a, bl_b, feed_a, feed_b)
    @test size(V) == (2, 1, 3, 4)

    # Phase closure of the gain factors on triangle (1,2,3) for the PP product:
    # arg(V12) + arg(V23) − arg(V13), with a point source the source phase is 0,
    # so the closure phase is exactly 0.
    pp = 1
    for c in 1:2
        cphase = angle(V[c, 1, 1, pp]) + angle(V[c, 1, 3, pp]) - angle(V[c, 1, 2, pp])
        # source coherency PP is real positive → contributes 0 to closure.
        @test isapprox(rem2pi(cphase, RoundNearest), 0.0; atol = 1.0e-10)
    end
end

# ── Audit-driven regression tests ────────────────────────────────────────────

@testset "Calibration parallel_hand_indices errors when PP/QQ missing" begin
    @test CAL.parallel_hand_indices(["PP", "PQ", "QP", "QQ"]) == (1, 4)
    # Regression for the operator-precedence bug: a missing PP must error, not
    # silently return (nothing, idx).
    @test_throws ErrorException CAL.parallel_hand_indices(["RL", "QQ"])
    @test_throws ErrorException CAL.parallel_hand_indices(["PP", "RL"])
    @test_throws ErrorException CAL.parallel_hand_indices(["RR", "LL"])
end

@testset "Calibration savitzky_golay_smooth: isolated finite sample" begin
    # Regression: a window containing one finite sample (ord = 0) must not crash.
    y = [NaN, 1.5, NaN]
    out = CAL.savitzky_golay_smooth(y, [0.0, 2.0, 0.0]; window = 7, order = 2)
    @test out[2] ≈ 1.5
    @test all(isfinite, out)                      # gaps interpolated from the one sample
    # A fully-finite smooth track is preserved.
    t = sin.(range(0, 2π; length = 30))
    sm = CAL.savitzky_golay_smooth(t; window = 7, order = 2)
    @test maximum(abs.(sm .- t)) < 0.1
end

@testset "Calibration: misdeclared coordinate term errors loudly (N3)" begin
    # A new term that declares the :Frequency axis but defines no
    # freq_coordinate must error at plan time, not silently evaluate at x = 0.
    @eval CAL begin
        struct _AuditBadFreqTerm <: AbstractGainTerm end
        term_axes(::_AuditBadFreqTerm) = (:Frequency,)
        param_shapes(::_AuditBadFreqTerm, n) = (scale = (),)
        # NOTE: deliberately no freq_coordinate method.
    end
    geom = CAL.DataGeometry(; times = [0.0], channel_freqs = [1.0e9, 2.0e9])
    model = CAL.StationGainModel(
        phase = (bad = CAL.GainComponent(CAL._AuditBadFreqTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),),
    )
    @test_throws MethodError CAL.plan_parameters(model, 1, geom)
end

@testset "CalibrationSolution θ keeps its array type" begin
    nant = 3
    freqs = collect(2.28e11:1.0e8:(2.28e11 + 5.0e8))
    times = [0.0, 0.5, 1.0]
    geom = CAL.DataGeometry(;
        times, channel_freqs = freqs, t0 = 0.0, f0 = sum(freqs) / length(freqs),
    )
    # A delay term plus a per-channel bandpass, so `step_solution` has
    # something to extract.
    model = CAL.StationGainModel(
        phase = (
            delay = CAL.GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),
            bandpass = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.ChannelBlocks(1), Feed = CAL.SharedFeeds()),
        ),
    )
    layout = CAL.plan_parameters(model, nant, geom)
    θv = collect(1:(layout.nθ)) ./ 1.0e10

    solv = CAL.CalibrationSolution(model, layout, geom, θv, (; nant))
    @test solv.steps[1].θ isa Vector{Float64}

    @testset "a DimArray θ survives construction and every derived path" begin
        # Split into the delay-only :fringe step and the bandpass-only
        # :bandpass step so the step-scoped accessors (`sol[i]`,
        # `component_gains`) have real steps to address.
        model_fr = CAL.StationGainModel(phase = (delay = model.phase.delay,))
        model_bp = CAL.StationGainModel(phase = (bandpass = model.phase.bandpass,))
        layout_fr = CAL.plan_parameters(model_fr, nant, geom)
        layout_bp = CAL.plan_parameters(model_bp, nant, geom)
        θv_fr = θv[1:(layout_fr.nθ)]
        θv_bp = θv[(layout_fr.nθ + 1):end]
        two_step(θfr, θbp) = CAL.CalibrationSolution(
            [
                CAL.StepSolution(:fringe, model_fr, layout_fr, θfr),
                CAL.StepSolution(:bandpass, model_bp, layout_bp, θbp),
            ],
            geom, (; nant),
        )
        solv2 = two_step(θv_fr, θv_bp)

        θd_fr = DimArray(copy(θv_fr), Dim{:param}(1:(layout_fr.nθ)))
        θd_bp = DimArray(copy(θv_bp), Dim{:param}(1:(layout_bp.nθ)))
        sold2 = two_step(θd_fr, θd_bp)
        @test sold2.steps[1].θ isa DimArray
        @test sold2.steps[1].θ == θd_fr
        # Same numbers as the Vector-backed solution, not merely close.
        ev = CAL.GainEvaluator(model_fr, layout_fr)
        @test CAL.evaluate_gains(ev, sold2.steps[1].θ) == CAL.evaluate_gains(ev, solv2.steps[1].θ)
        @test CAL.component_gains(sold2, :fringe, 1) == CAL.component_gains(solv2, :fringe, 1)
        # A step selection is index-matched to θ, so it propagates the array type.
        @test sold2[1:1].steps[1].θ isa DimArray
        @test sold2[1:1].steps[1].θ == solv2[1:1].steps[1].θ
        # `step_solution` returns the named step's own solution — its θ shares
        # no parameter identity with the merged original, only the element type.
        @test CAL.step_solution(sold2, :bandpass).steps[1].θ == CAL.step_solution(solv2, :bandpass).steps[1].θ
    end

    @testset "gains(sol) labels the forward map for inspection" begin
        ev = CAL.GainEvaluator(model, layout)
        g = gains(solv)
        @test g isa DimArray
        @test size(g) == (length(freqs), length(times), nant, 2)
        # The same numbers `evaluate_gains` / `apply_calibration` use.
        @test parent(g) == CAL.evaluate_gains(ev, solv.steps[1].θ)
        # Axes carry the geometry, so a user can index by physical coordinate.
        @test lookup(g, UVD.Frequency) == freqs
        @test lookup(g, Ti) == times
        @test lookup(g, UVD.Ant) == 1:nant           # no ant_names in info → 1:nant
        @test lookup(g, UVD.Feed) == 1:2
        # amp/phase recover from the complex gain, no separate accessor needed.
        @test abs.(g) == abs.(CAL.evaluate_gains(ev, solv.steps[1].θ))
        @test angle.(g) == angle.(CAL.evaluate_gains(ev, solv.steps[1].θ))
        # Indifferent to θ's array type.
        θd = DimArray(copy(θv), Dim{:param}(1:(layout.nθ)))
        @test gains(CAL.CalibrationSolution(model, layout, geom, θd, (; nant))) == g
    end

    @testset "the 1-based contract is enforced, not assumed" begin
        # A component's `range` holds absolute positions and the term kernels read
        # its block under `@inbounds`, so a shifted-axes θ must be refused here
        # rather than read out of bounds later.
        θoff = OffsetArrays.OffsetArray(copy(θv), 0:(layout.nθ - 1))
        @test_throws ArgumentError CAL.CalibrationSolution(model, layout, geom, θoff, (;))
        @test_throws DimensionMismatch CAL.CalibrationSolution(
            model, layout, geom, θv[1:(end - 1)], (;)
        )
        @test_throws "θ has length" CAL.CalibrationSolution(model, layout, geom, θv[1:(end - 1)], (;))
    end

    @testset "the element type is carried, not coerced to Float64" begin
        @test CAL.CalibrationSolution(model, layout, geom, Float32.(θv), (;)).steps[1].θ isa Vector{Float32}
    end

    @testset "the solution copies θ rather than aliasing it" begin
        # The fused output tail builds a solution per scan group from the run's
        # live θ while sibling groups are still writing their own slots.
        θmut = copy(θv)
        s = CAL.CalibrationSolution(model, layout, geom, θmut, (;))
        θmut[1] = -999.0
        @test s.steps[1].θ[1] == θv[1]
    end
end

@testset "argument validation is typed" begin
    # Constructor and argument validation throws `ArgumentError` or
    # `DimensionMismatch`; bare `error` is reserved for algorithmic failure, so a
    # caller can tell "you passed me nonsense" from "the solve did not converge".
    @testset "segmentation constructors" begin
        @test_throws ArgumentError CAL.TimeBlocks(0.0)
        @test_throws "duration_hr must be positive" CAL.TimeBlocks(-1.0)
        @test_throws ArgumentError CAL.ChannelBlocks(0)
        @test_throws "block_size must be at least 1" CAL.ChannelBlocks(-2)
        @test_throws ArgumentError CAL.FreqGroups(UnitRange{Int}[])
        @test_throws "at least one range" CAL.FreqGroups(UnitRange{Int}[])
        @test_throws "must start at channel 1" CAL.FreqGroups([2:4])
        @test_throws "contiguous and ascending" CAL.FreqGroups([1:4, 6:8])
    end

    @testset "geometry axis lengths" begin
        @test_throws DimensionMismatch CAL.DataGeometry(;
            times = [0.0, 1.0], channel_freqs = [1.0e9], scan_of_time = [1]
        )
        @test_throws "scan_of_time length" CAL.DataGeometry(;
            times = [0.0, 1.0], channel_freqs = [1.0e9], scan_of_time = [1]
        )
        @test_throws DimensionMismatch CAL.DataGeometry(;
            times = [0.0], channel_freqs = [1.0e9, 2.0e9], spw_of_chan = [1]
        )
        @test_throws "spw_of_chan length" CAL.DataGeometry(;
            times = [0.0], channel_freqs = [1.0e9, 2.0e9], spw_of_chan = [1]
        )

        # `FreqGroups` is structurally valid but must also cover the geometry.
        geom = CAL.DataGeometry(; times = [0.0], channel_freqs = collect(1.0:6.0) .* 1.0e9)
        @test_throws DimensionMismatch CAL.freq_segment_ids(CAL.FreqGroups([1:4]), geom)
        @test_throws "geometry has 6" CAL.freq_segment_ids(CAL.FreqGroups([1:4]), geom)
    end

    @testset "feed tying and empty models" begin
        @test_throws ArgumentError CAL.ReferenceRelative(3)
        @test_throws "reference_feed must be 1 or 2" CAL.ReferenceRelative(0)
        @test_throws ArgumentError CAL.SingleFeed(3)
        @test_throws "feed must be 1 or 2" CAL.SingleFeed(0)
        @test_throws ArgumentError CAL.validate_station_gain_model(CAL.StationGainModel())
        @test_throws "neither phase nor log-amplitude" CAL.validate_station_gain_model(
            CAL.StationGainModel()
        )
    end
end

@testset "unwrap_phase_track moves samples by 2π and nothing else" begin
    n = 64
    k = 0:(n - 1)
    w = fill(100.0, n)

    # Every sample comes back on some 2π branch of the value it went in as: the walk
    # chooses a branch, it does not adjust the track toward a trend or a smooth
    # shape.
    raw = rem2pi.(1.1 .* k .+ 0.3 .* sin.(k ./ 7), RoundNearest)
    got = CAL.unwrap_phase_track(raw; weights = w)
    @test all(abs(rem2pi(got[i] - raw[i], RoundNearest)) < 1.0e-9 for i in 1:n)

    # A trend gentle enough that every step's increment stays well inside ±π is
    # recovered exactly by that walk alone.
    gentle = 1.1 .* k .+ 0.3 .* sin.(k ./ 7)
    @test maximum(abs, (got .- got[1]) .- (gentle .- gentle[1])) < 1.0e-8

    # A track with no trend keeps the branch it arrived on.
    flat = collect(rem2pi.(0.2 .* sin.(k ./ 5), RoundNearest))
    @test CAL.unwrap_phase_track(flat; weights = w) ≈ flat

    # Past π per sample the trend is aliased in the samples themselves, so no walk
    # recovers it — the result must still be finite rather than an error.
    @test all(isfinite, CAL.unwrap_phase_track(rem2pi.(4.0 .* k, RoundNearest); weights = w))

    # Gaps are carried through untouched and do not anchor the walk.
    gappy = collect(rem2pi.(gentle, RoundNearest))
    gappy[20:30] .= NaN
    wg = copy(w)
    wg[20:30] .= 0
    out = CAL.unwrap_phase_track(gappy; weights = wg)
    @test all(isnan, out[20:30])
    @test maximum(abs, (out[31:end] .- out[31]) .- (gentle[31:end] .- gentle[31])) < 1.0e-8
end

@testset "phase_unwrap_ambiguity flags an undetermined branch" begin
    rng = MersenneTwister(5150)
    n = 64
    k = 0:(n - 1)
    w = fill(100.0, n)

    # A clean track: its increments sit in a tight cluster, so there is no scatter
    # to report however steep the trend that cluster is centred on. Steepness is a
    # separate failure of the walk and deliberately not this statistic's business.
    clean = rem2pi.(1.1 .* k .+ 0.3 .* sin.(k ./ 7), RoundNearest)
    @test CAL.phase_unwrap_ambiguity(clean; weights = w) == 0
    @test CAL.phase_unwrap_ambiguity(rem2pi.(2.7 .* k, RoundNearest); weights = w) == 0

    # Pure noise well below a radian stays resolvable; past it the walk is a coin
    # flip at a large fraction of its steps and the branch is not determined.
    amb(σ) = median(
        [
            CAL.phase_unwrap_ambiguity(rem2pi.(σ .* randn(rng, n), RoundNearest); weights = fill(1 / σ^2, n))
                for _ in 1:40
        ],
    )
    @test amb(0.2) < 0.05
    @test amb(0.4) < 0.1
    @test amb(1.5) > 0.25

    # No adjacent pair to compare ⇒ nothing to be ambiguous about.
    @test CAL.phase_unwrap_ambiguity(fill(NaN, 8)) == 0
    @test CAL.phase_unwrap_ambiguity(Float64[]) == 0
end
