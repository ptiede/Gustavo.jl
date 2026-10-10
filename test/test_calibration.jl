# Phase 1 — unified Calibration framework core.
# Standalone-runnable (`julia --project=test test/test_calibration.jl`) and
# included from runtests.jl.

using Gustavo
using Test
using LinearAlgebra
using DimensionalData: DimArray, Dim, lookup, Ti, name, dims, At
using Statistics: mean, median
using Dates: Minute, Second, Nanosecond, Month, DateTime, datetime2unix
using Random
import OffsetArrays
using Distributions: LogNormal, MvNormal


const CAL = Gustavo.Calibration

@testset "Calibration segmentation ids" begin
    # 5 times across 2 scans; 6 channels across 2 spws.
    geom = CAL.DataGeometry(; nfeed = 2,
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

    # The common refinement: two indices share a cell only where they share a
    # segment under every member. Members that agree come back unchanged…
    @test CAL.common_refinement([[1, 1, 2, 2], [1, 1, 2, 2]]) == ([1, 1, 2, 2], 2)
    # …and segmentations that cut the axis at different places give a partition
    # strictly finer than either, still numbered in axis order.
    @test CAL.common_refinement([[1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 3, 3]]) ==
        ([1, 1, 2, 3, 4, 4], 4)
    # A coarsening of another member changes nothing: refining is one-sided.
    @test CAL.common_refinement([[1, 1, 2, 2], [1, 1, 1, 1]]) == ([1, 1, 2, 2], 2)
    @test_throws DimensionMismatch CAL.common_refinement([[1, 1], [1, 1, 2]])

    # A name vector names every segment or none; a partial one would silently
    # label the wrong segment when a foreign grid is placed against it.
    @test_throws "must name every segment or be empty" CAL.DataGeometry(; nfeed = 2,
        times = [0.0, 1.0], scan_of_time = [1, 2], channel_freqs = [1.0e9],
        scan_names = ["only-one"],
    )
end

@testset "Materialization and segment_ranges" begin
    # 4 channels: a gap-separated pair of groups, in 3 spws.
    geom = CAL.DataGeometry(; nfeed = 2,
        times = [0.0, 1.0],
        channel_freqs = [1.0e9, 1.1e9, 1.2e9, 5.0e9],
        spw_of_chan = [1, 1, 2, 3],
    )

    # Concrete segmentations resolve to themselves; the data-dependent
    # `BandGroups` resolves to the `FreqGroups` its gap detection finds.
    @test CAL.BandGroups() isa CAL.AbstractFrequencySegmentation
    for seg in (
            CAL.GlobalFrequency(), CAL.PerSpectralWindow(), CAL.ChannelBlocks(2),
            CAL.FreqGroups([1:3, 4:4]),
        )
        @test CAL.resolve(seg, geom) === seg
    end
    @test CAL.resolve(CAL.BandGroups(), geom) == CAL.FreqGroups([1:3, 4:4])
    # `gap_factor` reaches the gap detection: an unreachable ratio keeps one group.
    @test CAL.resolve(CAL.BandGroups(gap_factor = 1.0e9), geom) ==
        CAL.FreqGroups([1:4])

    # The range form of each shipped concrete segmentation.
    @test CAL.segment_ranges(CAL.GlobalFrequency(), geom) == [1:4]
    @test CAL.segment_ranges(CAL.PerSpectralWindow(), geom) == [1:2, 3:3, 4:4]
    @test CAL.segment_ranges(CAL.ChannelBlocks(2), geom) == [1:2, 3:3, 4:4]
    @test CAL.segment_ranges(CAL.FreqGroups([1:1, 2:4]), geom) == [1:1, 2:4]

    # A segmentation whose segments interleave has no range form.
    inter = CAL.DataGeometry(; nfeed = 2,
        times = [0.0], channel_freqs = [1.0e9, 1.1e9, 1.2e9], spw_of_chan = [1, 2, 1],
    )
    @test_throws "no contiguous channel range" CAL.segment_ranges(
        CAL.PerSpectralWindow(), inter
    )

    # Layouts resolve: a plan built from a `BandGroups` component records the
    # concrete `FreqGroups`, so solutions and foreign-grid placement never see
    # the data-dependent form.
    model = CAL.GainModel(
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
    solve = CAL.DataGeometry(; nfeed = 2,
        times = [0.0, 0.1, 1.0, 1.1], scan_of_time = [7, 7, 9, 9],
        channel_freqs = [1.0, 1.1, 1.2, 2.0, 2.1, 2.2] .* 1.0e9,
        spw_of_chan = [3, 3, 3, 4, 4, 4],
        scan_names = ["No001", "No002"], spw_names = ["A", "B"],
    )
    # The same scans and spws, sampled three times as finely in time.
    fine = CAL.DataGeometry(; nfeed = 2,
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
        averaged = CAL.DataGeometry(; nfeed = 2,
            times = [0.05, 1.05], scan_of_time = [7, 9],
            channel_freqs = solve.channel_freqs, spw_of_chan = solve.spw_of_chan,
            scan_names = ["No001", "No002"], spw_names = ["A", "B"],
        )
        @test CAL.time_segment_ids(CAL.PerScan(), averaged, fine) == [1, 1, 1, 2, 2, 2]
    end

    @testset "a scan or spw the solve never saw errors" begin
        stranger = CAL.DataGeometry(; nfeed = 2,
            times = [0.0, 2.0], scan_of_time = [1, 2],
            channel_freqs = solve.channel_freqs, spw_of_chan = solve.spw_of_chan,
            scan_names = ["No001", "No009"], spw_names = ["A", "B"],
        )
        # Named, not borrowed from the neighbouring scan it sits closest to.
        @test_throws "PerScan" CAL.time_segment_ids(CAL.PerScan(), solve, stranger)
        @test_throws "\"No009\" is not in the solution" CAL.time_segment_ids(
            CAL.PerScan(), solve, stranger,
        )
        other_band = CAL.DataGeometry(; nfeed = 2,
            times = solve.times, scan_of_time = solve.scan_of_time,
            channel_freqs = solve.channel_freqs, spw_of_chan = [1, 1, 1, 2, 2, 2],
            scan_names = ["No001", "No002"], spw_names = ["A", "C"],
        )
        @test_throws "\"C\" is not in the solution" CAL.freq_segment_ids(
            CAL.PerSpectralWindow(), solve, other_band,
        )
    end

    @testset "identity placement needs names, not raw ids" begin
        # Raw ids agree with themselves, so an identical labeling still places:
        # the two grids then correspond sample for sample and no name is needed.
        unnamed = CAL.DataGeometry(; nfeed = 2,
            times = solve.times, scan_of_time = solve.scan_of_time,
            channel_freqs = solve.channel_freqs, spw_of_chan = solve.spw_of_chan,
        )
        @test CAL.time_segment_ids(CAL.PerScan(), unnamed, unnamed) == [1, 1, 2, 2]
        @test CAL.freq_segment_ids(CAL.PerSpectralWindow(), unnamed, unnamed) == [1, 1, 1, 2, 2, 2]

        # Across two differently-labeled grids they do not: matching id 1 to
        # id 1 is positional matching, which identity placement exists to avoid.
        unnamed_fine = CAL.DataGeometry(; nfeed = 2,
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
        gapped = CAL.DataGeometry(; nfeed = 2, times = [0.0, 0.1, 2.0, 2.1], channel_freqs = [1.0e9])
        seg = CAL.TimeBlocks(1.0)
        @test CAL.time_segment_ids(seg, gapped) == ([1, 1, 2, 2], 2)
        either_side = CAL.DataGeometry(; nfeed = 2, times = [0.5, 2.5], channel_freqs = [1.0e9])
        @test CAL.time_segment_ids(seg, gapped, either_side) == [1, 2]
        inside = CAL.DataGeometry(; nfeed = 2, times = [1.5], channel_freqs = [1.0e9])
        @test_throws "no solve epoch populated" CAL.time_segment_ids(seg, gapped, inside)

        # The solve's own block origin is used, never the target's: a target
        # starting an hour later must still land in the solve's blocks.
        later = CAL.DataGeometry(; nfeed = 2, times = [2.05], channel_freqs = [1.0e9])
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
        aps = CAL.DataGeometry(; nfeed = 2, times = [1.0, 2.0, 3.0], channel_freqs = [1.0e9])
        @test CAL.time_segment_ids(CAL.PerIntegration(), aps, aps) == [1, 2, 3]
        # Untouched data carries a span narrower than the AP spacing.
        @test CAL.time_segment_ids(CAL.PerIntegration(), aps, aps; time_span = fill(0.9, 3)) ==
            [1, 2, 3]

        # A near miss names both epochs, so a unit slip is diagnosable at a glance.
        off = CAL.DataGeometry(; nfeed = 2, times = [2.25], channel_freqs = [1.0e9])
        @test_throws "2.25" CAL.time_segment_ids(CAL.PerIntegration(), aps, off)
        @test_throws "nearest is 2.0" CAL.time_segment_ids(CAL.PerIntegration(), aps, off)

        # Averaging {1, 2, 3} h lands exactly ON a solve epoch, so the epoch
        # match alone would accept it; the span it now carries does not.
        avg = CAL.DataGeometry(; nfeed = 2, times = [2.0], channel_freqs = [1.0e9])
        @test CAL.time_segment_ids(CAL.PerIntegration(), aps, avg) == [2]
        @test_throws "covering solve epochs 1.0, 2.0, 3.0" CAL.time_segment_ids(
            CAL.PerIntegration(), aps, avg; time_span = [2.5],
        )
    end

    @testset "channel segmentations place by frequency and width" begin
        widths = fill(0.1e9, 6)
        solvew = CAL.DataGeometry(; nfeed = 2,
            solve.times, solve.scan_of_time, solve.channel_freqs, solve.spw_of_chan,
            channel_widths = widths, scan_names = solve.scan_names, spw_names = solve.spw_names,
        )
        sub(idx; freqs = solve.channel_freqs[idx], w = widths[idx]) = CAL.DataGeometry(; nfeed = 2,
            times = solve.times, scan_of_time = solve.scan_of_time,
            channel_freqs = freqs, spw_of_chan = solve.spw_of_chan[idx], channel_widths = w,
        )
        @test CAL.freq_segment_ids(CAL.ChannelBlocks(2), solve, solve) == [1, 1, 2, 3, 3, 4]
        groups = CAL.FreqGroups([1:3, 4:6])
        @test CAL.freq_segment_ids(groups, solve, solve) == [1, 1, 1, 2, 2, 2]
        # One window, or any subset of channels, takes the segments of its own channels.
        @test CAL.freq_segment_ids(CAL.ChannelBlocks(2), solvew, sub(4:6)) == [3, 3, 4]
        @test CAL.freq_segment_ids(groups, solvew, sub([5, 2])) == [2, 1]
        @test CAL.freq_segment_ids(groups, solvew, sub([5, 2]); chan_idx = [2]) == [1]
        # A channel outside every solve channel: the message prints the nearest, so a
        # unit slip or a shifted correlator setup is diagnosable at a glance.
        @test_throws "at 1.5e9 Hz lies in no channel of the solution — the nearest is at 1.2e9 Hz" CAL.freq_segment_ids(
            CAL.ChannelBlocks(2), solvew, sub([3]; freqs = [1.5e9]),
        )
        shifted = sub(1:6; freqs = solve.channel_freqs .+ 1.0e6)
        @test_throws "extends beyond the solution's 1.0e8 Hz channel at 1.0e9 Hz" CAL.freq_segment_ids(
            CAL.ChannelBlocks(2), solvew, shifted,
        )
        # Three channels averaged into one: its center is a solve channel's, its band is wider.
        averaged = sub([2]; w = [0.3e9])
        @test_throws "extends beyond the solution's" CAL.freq_segment_ids(groups, solvew, averaged)
        # A solution on averaged channels places the channels each one averaged.
        coarse(freqs, w) = CAL.DataGeometry(; nfeed = 2,
            times = solve.times, scan_of_time = solve.scan_of_time, channel_freqs = freqs,
            spw_of_chan = [3, 4], channel_widths = w,
        )
        twobins = coarse([1.05, 2.05] .* 1.0e9, [0.2e9, 0.2e9])
        @test CAL.freq_segment_ids(CAL.ChannelBlocks(1), twobins, sub([1, 2, 4, 5])) == [1, 1, 2, 2]
        @test_throws "at 1.2e9 Hz lies in no channel" CAL.freq_segment_ids(CAL.ChannelBlocks(1), twobins, sub([3]))
        overlapping = coarse([1.05, 1.1] .* 1.0e9, [0.2e9, 0.2e9])
        @test_throws "whose stated widths overlap" CAL.freq_segment_ids(CAL.ChannelBlocks(1), overlapping, sub([2]))
        # Widths are needed on both sides once the grids differ.
        @test_throws "SOLUTION geometry states no channel widths" CAL.freq_segment_ids(groups, solve, sub(4:6))
        @test_throws "TARGET geometry states no channel widths" CAL.freq_segment_ids(groups, solvew, sub(4:6; w = Float64[]))
        @test_throws "channel_widths length" sub(1:2; w = [1.0])
        @test_throws "finite and positive" sub(1:2; w = [1.0, 0.0])
    end

    @testset "gains evaluate on the foreign grid in the solve's own basis" begin
        model = CAL.GainModel(
            phase = (
                atmos = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),
                mbd = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),
            ),
        )
        nant = 2
        ev = CAL.plan_parameters(model, nant, solve)
        θ = collect(range(0.1; step = 0.05, length = ev.nθ))
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
        stranger = CAL.DataGeometry(; nfeed = 2,
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
    @test CAL.PolynomialFreq(3) isa CAL.Polynomial{Frequency}
    @test CAL.PolynomialTime(3) isa CAL.Polynomial{Ti}
    @test CAL.term_axes(CAL.PolynomialFreq(3)) == (Frequency,)
    @test CAL.term_axes(CAL.PolynomialTime(3)) == (Ti,)

    @test_throws ArgumentError CAL.PolynomialFreq(0)
    @test_throws "axis must be one of" CAL.Polynomial{Gustavo.AntennaName}(2)

    # A term is handed the axes it declares, under the dimension names, and a
    # channel index is never one of them.
    @test CAL.term_axes(CAL.Delay()) == (Frequency,)
    @test CAL.term_axes(CAL.Rate()) == (Ti,)
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
            (CAL.SingleFeed(2), "SingleFeed(2)"),
            (CAL.ExceptFeed(1), "ExceptFeed(1)"),
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
    m1 = CAL.GainModel(phase = (sbd = mk(fg1),))
    m2 = CAL.GainModel(phase = (sbd = mk(fg2),))
    @test m1 == m2 && hash(m1) == hash(m2)
    @test m1 != CAL.GainModel(logamp = (sbd = mk(fg1),))

    # Unnamed components are rejected with the constructor the message shows.
    @test_throws "pass a NamedTuple" CAL.GainModel(phase = (mk(fg1),))
end

@testset "Calibration feed tying offset algebra" begin
    geom = CAL.DataGeometry(; nfeed = 2, times = [0.0, 1.0], channel_freqs = [1.0e9, 2.0e9])
    # One ConstantTerm, GlobalTime × GlobalFrequency, 2 antennas.
    mk(tying) = CAL.GainModel(
        phase = (c = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = tying),),
    )

    lay_pf = CAL.plan_parameters(mk(CAL.PerFeed()), 2, geom)
    @test lay_pf.nθ == 4                                    # 2 ant × 2 feeds
    p = lay_pf.plans[1]
    @test plan_off1(p)[1, 1, 1, 1] != plan_off1(p)[1, 2, 1, 1]          # feeds independent

    lay_sf = CAL.plan_parameters(mk(CAL.SharedFeeds()), 2, geom)
    @test lay_sf.nθ == 2                                    # 2 ant, shared across feeds
    q = lay_sf.plans[1]
    @test plan_off1(q)[1, 1, 1, 1] == plan_off1(q)[1, 2, 1, 1]          # feeds share a block

    lay_1f = CAL.plan_parameters(mk(CAL.SingleFeed(2)), 2, geom)
    @test lay_1f.nθ == 2                                    # one block per ant
    r = lay_1f.plans[1]
    @test plan_off1(r)[1, 1, 1, 1] == 0                           # feed 1 has no block
    @test plan_off1(r)[1, 2, 1, 1] != 0

    geom3 = CAL.DataGeometry(; nfeed = 3, times = [0.0, 1.0], channel_freqs = [1.0e9, 2.0e9])
    lay_ex = CAL.plan_parameters(mk(CAL.ExceptFeed(2)), 2, geom3)
    @test lay_ex.nθ == 4                                    # 2 ant × feeds 1 and 3
    @test lay_ex.plans[1].shape == (1, 2, 1, 1, 2)
    @test [CAL._feed_node(CAL.ExceptFeed(2), f) for f in 1:3] == [1, 0, 2]   # feed 2 has no block
    geom1 = CAL.DataGeometry(; nfeed = 1, times = [0.0, 1.0], channel_freqs = [1.0e9, 2.0e9])
    @test CAL.plan_parameters(mk(CAL.ExceptFeed(1)), 2, geom1).nθ == 0
    @test_throws "ExceptFeed(3) names a feed the data does not have" CAL.plan_parameters(mk(CAL.ExceptFeed(3)), 2, geom)
    @test_throws "ExceptFeed feed must be at least 1" CAL.ExceptFeed(0)
end

@testset "Calibration parameter layout" begin
    freqs = [1.0e9, 2.0e9, 3.0e9]                    # 3 channels
    geom = CAL.DataGeometry(; nfeed = 2, times = [0.0, 1.0], channel_freqs = freqs)
    nant = 2
    model = CAL.GainModel(
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

    # The axes tree nests where the model does.
    @test keys(layout.axes) == (:phase, :logamp)
    @test keys(layout.axes.phase) == (:a, :bp, :grp)
    @test keys(layout.axes.phase.grp) == (:d, :c)

    # Each leaf is the full-rank shape its tying × segmentation imply
    # (param, feed-node, freq-seg, time-seg, ant), size-1 axes kept.
    @test layout.axes.phase.a.dims == (1, 2, 1, 1, nant)                 # PerFeed: two feed nodes
    @test layout.axes.phase.bp.dims == (1, 1, length(freqs), 1, nant)    # ChannelBlocks(1): one freq-seg per channel
    @test layout.axes.phase.grp.d.dims == (1, 1, 1, 1, nant)             # a nested part is a plain leaf
    @test layout.axes.phase.grp.c.dims == (1, 1, 1, 1, nant)

    # The stored axis roles name each dimension.
    @test layout.axes.phase.a.roles == (:param, :Feed, :Frequency, :Ti, :AntennaName)
    @test layout.axes.phase.bp.roles == (:param, :node, :Frequency, :Ti, :AntennaName)
    @test layout.axes.phase.grp.d.roles == (:param, :node, :Frequency, :Ti, :AntennaName)

    # The plans tile θ depth-first — a, bp, grp.d, grp.c — each over its leaf.
    rng = [p.range for p in layout.plans]
    @test reduce(vcat, collect.(rng)) == 1:layout.nθ
    @test [p.shape for p in layout.plans] ==
        [layout.axes.phase.a.dims, layout.axes.phase.bp.dims, layout.axes.phase.grp.d.dims, layout.axes.phase.grp.c.dims]

    # The forward map follows θ's element type, so a caller can differentiate
    # through it or evaluate it in higher precision.
    θs = 1.0e-10 .* θ                  # delays of order 0.1 ns keep the phases O(1)
    gb = CAL.evaluate_gains(layout, BigFloat.(θs))
    @test eltype(gb) == Complex{BigFloat}
    @test gb ≈ CAL.evaluate_gains(layout, θs)
end

@testset "a solution is a list of labeled solved components" begin
    freqs = [1.0e9, 2.0e9, 3.0e9, 4.0e9]
    times = [0.0, 1.0, 2.0, 3.0]
    ants = ["PT", "LM", "AA"]
    geom = CAL.DataGeometry(; nfeed = 2,
        times, channel_freqs = freqs, stations = ants,
        scan_of_time = [1, 1, 2, 2], spw_of_chan = [1, 1, 1, 1], t0 = 0.0, f0 = 2.5e9,
    )
    nant = 3
    model = CAL.GainModel(
        phase = (
            atmos = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),
            bp = CAL.GainComponent(CAL.PolynomialFreq(2); Ti = CAL.GlobalTime(), Frequency = CAL.ChannelBlocks(2), Feed = CAL.SharedFeeds()),
            rl = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SingleFeed(2)),
        ),
        logamp = (
            amp = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),
        ),
    )
    layout = CAL.plan_parameters(model, nant, geom)
    θ = Float64.(1:layout.nθ)
    sol = CAL.CalibrationSolution(model, layout, geom, θ, (;))

    # One component per model entry, in model order, each labeled by step and path.
    @test [(c.step, c.path) for c in sol.components] ==
        [(:solution, (:phase, :atmos)), (:solution, (:phase, :bp)), (:solution, (:phase, :rl)), (:solution, (:logamp, :amp))]
    @test length(sol) == 4 && collect(sol) == sol.components
    @test haskey(sol, :solution) && !haskey(sol, :other)
    # The one-step form records `info` as the step's diagnostics only.
    @test CAL.CalibrationSolution(model, layout, geom, θ, (; note = 1)).steps[:solution] == (; note = 1)
    @test CAL.CalibrationSolution(model, layout, geom, θ, (; note = 1)).info == (;)
    c = first(sol.components)
    @test c isa CAL.SolvedComponent
    @test c.component == model.phase.atmos
    a = c.params
    @test a isa DimArray
    @test size(a) == layout.axes.phase.atmos.dims
    @test name.(dims(a)) == layout.axes.phase.atmos.roles

    # PerFeed carries a physical Feed axis; the tied tyings carry a positional
    # node axis.
    @test name(dims(a, 2)) == :Feed
    bpp = only(sol[:solution, :phase, :bp].components).params
    @test name(dims(bpp, 2)) == :node
    @test size(only(sol[:solution, :phase, :rl].components).params, 2) == 1

    # Segment axes span each segment's channels or samples, labeled by the
    # midpoint, so `Contains` finds the segment covering a coordinate.
    @test lookup(a, Gustavo.Frequency) == [mean(freqs)]              # GlobalFrequency: one segment
    @test DimensionalData.intervalbounds(a, Gustavo.Frequency) == [(1.0e9, 4.0e9)]
    @test lookup(a, Ti) == [0.5, 2.5]                            # PerScan: one per scan
    @test DimensionalData.intervalbounds(a, Ti) == [(0.0, 1.0), (2.0, 3.0)]
    @test a[Ti(Contains(2.2)), Gustavo.AntennaName(At("PT")), Gustavo.Feed(1)] == a[Ti(2), Gustavo.AntennaName(1), Gustavo.Feed(1)]
    @test_throws "No interval contains" a[Ti(Contains(1.5))]     # between scans
    @test lookup(bpp, Gustavo.Frequency) == [mean(freqs[1:2]), mean(freqs[3:4])]   # ChannelBlocks(2)

    # Antennas take the geometry's station names; feed is 1:2.
    @test lookup(a, Gustavo.AntennaName) == ants
    @test lookup(a, Gustavo.Feed) == 1:2

    # The parameters are a copy of θ's blocks.
    rng = [p.range for p in layout.plans]
    @test vec(parent(a)) == θ[rng[1]]
    a[1, 1, 1, 1, 1] = -7.0
    @test θ[first(rng[1])] == first(rng[1])

    # Selections are solutions: by step, by path prefix, or by any predicate.
    @test sol[:solution] isa CAL.CalibrationSolution
    @test length(sol[:solution, :phase]) == 3
    @test [c.path for c in filter(c -> c.component.Feed isa CAL.SharedFeeds, sol)] == [(:phase, :bp), (:logamp, :amp)]
    @test_throws "holds no component under solution.phase.nope" sol[:solution, :phase, :nope]
    @test_throws "holds no component under other" sol[:other]

    # Equal components hash equally, so they collapse in sets and `unique`.
    twin = CAL.SolvedComponent(c.step, c.path, c.component, copy(c.params))
    @test twin == c && hash(twin) == hash(c)
    @test length(unique([c, twin])) == 1

    # An empty selection is a solution, but applying it is refused.
    none = filter(_ -> false, sol)
    @test isempty(none.components) && isempty(none.steps)
    @test_throws "holds no components" gains(none)

    # A nested subtree (`sbd.delay` / `sbd.constant`) keeps its path.
    gb = CAL.DataGeometry(; nfeed = 2,
        times = [0.0, 1.0], channel_freqs = [1.0e9, 1.1e9, 5.0e9, 5.1e9], stations = ["X", "Y"],
        scan_of_time = [1, 1], spw_of_chan = [1, 1, 2, 2], t0 = 0.0, f0 = 3.0e9,
    )
    groups = CAL.FreqGroups([1:2, 3:4])
    sbd = (
        delay = CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan(), Frequency = groups, Feed = CAL.SharedFeeds()),
        constant = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = groups, Feed = CAL.SharedFeeds()),
    )
    msbd = CAL.GainModel(phase = (sbd = sbd,))
    lsbd = CAL.plan_parameters(msbd, 2, gb)
    θsbd = Float64.(1:lsbd.nθ)
    ssbd = CAL.CalibrationSolution(msbd, lsbd, gb, θsbd, (;))
    @test [c.path for c in ssbd.components] == [(:phase, :sbd, :delay), (:phase, :sbd, :constant)]
    dl = ssbd.components[1].params
    @test lookup(dl, Gustavo.Frequency) == [mean([1.0e9, 1.1e9]), mean([5.0e9, 5.1e9])]
    @test vec(parent(dl)) == θsbd[lsbd.plantree.phase.sbd.delay.range]
    @test parent(gains(ssbd)) ≈ CAL.evaluate_gains(lsbd, θsbd)

    # Components from different steps may share a name.
    atmos2 = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds())
    model2 = CAL.GainModel(phase = (atmos = atmos2,))
    layout2 = CAL.plan_parameters(model2, nant, geom)
    θ2 = fill(-1.0, layout2.nθ)
    twostep = CAL.CalibrationSolution(
        geom,
        vcat(
            CAL.CalibrationSolution(model, layout, geom, θ; name = :bandpass).components,
            CAL.CalibrationSolution(model2, layout2, geom, θ2; name = :fringe).components,
        ),
    )
    @test vec(parent(only(twostep[:bandpass, :phase, :atmos].components).params)) == θ[rng[1]]
    @test all(==(-1.0), only(twostep[:fringe, :phase, :atmos].components).params)
    @test parent(gains(twostep)) ≈ parent(gains(twostep[:bandpass])) .* parent(gains(twostep[:fringe]))
    @test_throws "components repeat: bandpass.phase.atmos" CAL.CalibrationSolution(
        geom, vcat(twostep.components, twostep[:bandpass, :phase, :atmos].components),
    )

    # A path starts at :phase or :logamp; a geometry must name its stations.
    @test_throws "starts at :phase or :logamp" CAL.SolvedComponent(:s, (:atmos,), c.component, c.params)
    nostations = CAL.DataGeometry(; nfeed = 2, times, channel_freqs = freqs, scan_of_time = [1, 1, 2, 2], spw_of_chan = [1, 1, 1, 1], t0 = 0.0, f0 = 2.5e9)
    @test_throws "must name its stations" CAL.CalibrationSolution(model, layout, nostations, θ)
end

@testset "Calibration evaluate_gains: correctness, purity, inference" begin
    nant = 3
    freqs = collect(2.28e11:1.0e8:(2.28e11 + 5.0e8))         # 6 channels
    f0 = sum(freqs) / length(freqs)
    times = [0.0, 0.5, 1.0]
    geom = CAL.DataGeometry(; nfeed = 2, times, channel_freqs = freqs, t0 = 0.0, f0)

    # Pure per-feed delay model: phase = 2π τ (f − f0), one τ per (ant, feed).
    model = CAL.GainModel(
        phase = (delay = CAL.GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),),
    )
    ev = CAL.plan_parameters(model, nant, geom)
    @test ev.nθ == nant * 2

    τ = 1.0e-9 .* collect(1:(nant * 2))                    # distinct delays in ns
    g = CAL.evaluate_gains(ev, τ)
    @test size(g) == (length(freqs), length(times), nant, 2)

    # Hand-check a couple of cells: |g| == 1 (no amplitude), phase = 2π τ (f−f0).
    p = ev.plans[1]
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

    # A windowed evaluation rejects out-of-grid indices up front: the loop
    # reads the plans' grid tables under `@inbounds`, so this is the last
    # point where a bad index can fail loudly.
    @test_throws ArgumentError CAL.evaluate_gains(ev, τ, [1], [10_000])
    @test_throws "outside the layout's time grid" CAL.evaluate_gains(ev, τ, [1], [10_000])
    @test_throws "outside the layout's channel grid" CAL.evaluate_gains(ev, τ, [0], [1])

    # Type stability of the forward map.
    @inferred CAL.evaluate_gains(ev, τ)

    # Rate term: phase grows linearly in time, flat in frequency, about the
    # segment's OWN mean epoch (one segment here, so the whole track's).
    rate_model = CAL.GainModel(
        phase = (rate = CAL.GainComponent(CAL.Rate(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.SharedFeeds()),),
    )
    evr = CAL.plan_parameters(rate_model, nant, geom)
    ṙ = 1.0e-3 .* collect(1:nant)                          # mHz-scale rates
    gr = CAL.evaluate_gains(evr, ṙ)
    pr = evr.plans[1]
    @test pr.tstate ≈ [sum(times) / length(times)]
    for ant in 1:nant, ti in eachindex(times)
        off = plan_off1(pr)[ant, 1, 1, 1]
        expected = cis(2π * ṙ[off] * (times[ti] - pr.tstate[1]))
        @test gr[1, ti, ant, 1] ≈ expected
        @test gr[1, ti, ant, 2] ≈ expected               # shared across feeds
    end
end

# ── Audit-driven regression tests ────────────────────────────────────────────

@testset "Priors on a component" begin
    @testset "construction and validation" begin
        @test CAL.IIDPrior(0.1).σ == 0.1
        rw = CAL.RandomWalkPrior(; order = 2, σ = 0.01f0)
        @test rw isa CAL.RandomWalkPrior{Float32} && rw.order == 2
        @test CAL.RandomWalkPrior(; σ = 1.0).order == 1
        ou = CAL.OUPrior(; scale = 5, σ = 0.2)
        @test ou isa CAL.OUPrior{Float64, Float64} && ou.scale === 5.0

        @test_throws "IIDPrior σ must be positive and finite, got 0" CAL.IIDPrior(0)
        @test_throws "RandomWalkPrior σ must be positive and finite" CAL.RandomWalkPrior(; σ = Inf)
        @test_throws "RandomWalkPrior order must be ≥ 1, got 0" CAL.RandomWalkPrior(; order = 0, σ = 1.0)
        @test_throws "OUPrior scale must be positive" CAL.OUPrior(; scale = -1.0, σ = 1.0)

        # OU hyperparameters are fixed numbers or hyperpriors.
        hp = CAL.OUPrior(; scale = LogNormal(log(1.0e7), 1.0), σ = 0.1)
        @test hp.scale isa LogNormal && hp.σ === 0.1
        @test !CAL.is_fixed_hyper(hp.scale) && CAL.is_fixed_hyper(hp.σ)
        @test hp == CAL.OUPrior(; scale = LogNormal(log(1.0e7), 1.0), σ = 0.1)
        @test_throws "OUPrior σ must be a positive number or a density implementing " *
            "DensityInterface.logdensityof, got String" CAL.OUPrior(; scale = 1.0, σ = "wide")
    end

    @testset "the axis comes from the component" begin
        comp(; Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), prior) =
            CAL.GainComponent(CAL.ConstantTerm(); Ti, Frequency, prior)
        rw = CAL.RandomWalkPrior(; order = 2, σ = 0.01)
        ou = CAL.OUPrior(; scale = 10.0, σ = 1.0)
        @test isnothing(CAL.resolve_prior(CAL.GainComponent(CAL.Delay(); Ti = CAL.PerScan())))
        @test CAL.resolve_prior(comp(; Frequency = CAL.ChannelBlocks(1), prior = rw)) == (Frequency = rw,)
        @test CAL.resolve_prior(comp(; Ti = CAL.PerIntegration(), prior = ou)) == (Ti = ou,)
        # IID relates nothing, so it needs no axis and fits any component.
        iid = CAL.IIDPrior(1.0)
        @test CAL.resolve_prior(comp(; prior = iid)) === iid

        both = (; Ti = CAL.PerIntegration(), Frequency = CAL.ChannelBlocks(1))
        @test CAL.resolve_prior(comp(; both..., prior = (Ti = ou, Frequency = rw))) == (Ti = ou, Frequency = rw)
        @test CAL.resolve_prior(comp(; both..., prior = (Frequency = rw,))) == (Frequency = rw,)

        @test_throws "relates nothing, the component segments neither Ti nor Frequency" comp(; prior = rw)
        @test_throws "GainComponent(ConstantTerm(); Ti = GlobalTime()" comp(; prior = rw)
        @test_throws "is ambiguous, the component segments both Ti and Frequency; key it by axis, " *
            "e.g. `prior = (Frequency = RandomWalkPrior(; order = 2, σ = 0.01),)`" comp(; both..., prior = rw)
        @test_throws "relates nothing, the component does not segment Ti" comp(;
            Frequency = CAL.ChannelBlocks(1), prior = (Ti = ou,),
        )
        @test_throws "prior key AntennaName is not an axis" comp(; both..., prior = (AntennaName = ou,))
        @test_throws "a keyed prior must be a RandomWalkPrior or OUPrior" comp(; both..., prior = (Ti = iid,))
        @test_throws "a keyed prior needs at least one axis" comp(; both..., prior = (;))
    end

    @testset "attached to a component" begin
        bp(prior) = CAL.GainComponent(
            CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.PerSpectralWindow(), prior,
        )
        rw = CAL.RandomWalkPrior(; order = 2, σ = 0.01)
        @test bp(rw).prior === rw

        # The prior is part of the component's value and its printed form.
        @test bp(rw) == bp(CAL.RandomWalkPrior(; order = 2, σ = 0.01))
        @test hash(bp(rw)) == hash(bp(CAL.RandomWalkPrior(; order = 2, σ = 0.01)))
        @test bp(rw) != bp(nothing)
        @test bp(rw) != bp(CAL.RandomWalkPrior(; order = 1, σ = 0.01))
        @test !occursin("prior", CAL.component_label(bp(nothing)))
        keyed = CAL.GainComponent(
            CAL.ConstantTerm(); Ti = CAL.PerIntegration(), Frequency = CAL.ChannelBlocks(1),
            prior = (Ti = CAL.OUPrior(; scale = 30.0, σ = 0.1), Frequency = rw),
        )
        for c in (bp(rw), bp(CAL.OUPrior(; scale = 2.0e6, σ = 0.3)), bp(CAL.IIDPrior(0.5)), keyed)
            lbl = CAL.component_label(c)
            @test occursin("prior = ", lbl)
            @test Core.eval(CAL, Meta.parse(lbl)) == c
        end
        @test CAL.component_label(bp(rw)) ==
            "GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = PerSpectralWindow(), " *
            "Feed = PerFeed(), prior = RandomWalkPrior(; order = 2, σ = 0.01))"

        rw_init = CAL.RandomWalkPrior(; order = 2, σ = 0.01, init = MvNormal(zeros(2), [1.0 0.0; 0.0 2.0]))
        ou_hyper = CAL.OUPrior(; scale = LogNormal(18.0, 1.0), σ = 0.1)
        for p in (rw_init, ou_hyper, CAL.IIDPrior(0.5))
            twin = deepcopy(bp(p))
            @test twin == bp(p) && hash(twin) == hash(bp(p))
        end
        @test bp(rw_init) != bp(CAL.RandomWalkPrior(; order = 2, σ = 0.01, init = MvNormal(zeros(2), [1.0 0.0; 0.0 3.0])))
        @test bp(rw_init) != bp(CAL.RandomWalkPrior(; order = 2, σ = 0.02, init = rw_init.init))
        @test bp(ou_hyper) != bp(CAL.OUPrior(; scale = LogNormal(18.0, 2.0), σ = 0.1))
        @test bp(ou_hyper) != bp(CAL.OUPrior(; scale = LogNormal(18.0, 1.0), σ = 0.2))
        @test bp(CAL.IIDPrior(0.5)) != bp(CAL.IIDPrior(0.6))
    end

    @testset "stations differing only in prior share a plan" begin
        geom = CAL.DataGeometry(; nfeed = 2,
            times = [0.0, 1.0], scan_of_time = [1, 2], stations = ["AA", "BB", "CC"],
            channel_freqs = [1.0, 1.1, 1.2, 2.0, 2.1] .* 1.0e9, spw_of_chan = [1, 1, 1, 2, 2],
            t0 = 0.0, f0 = 1.5e9,
        )
        names = ["AA", "BB", "CC"]
        bp(prior; Ti = CAL.GlobalTime()) = CAL.GainComponent(
            CAL.ConstantTerm(); Ti, Frequency = CAL.PerSpectralWindow(), prior,
        )
        rw = CAL.RandomWalkPrior(; order = 2, σ = 0.01)
        ou = CAL.OUPrior(; scale = LogNormal(18.0, 1.0), σ = 0.1)
        m = CAL.GainModel(
            phase = (bp = bp(rw),),
            stations = (BB = (; phase = (bp = bp(ou),)),),
        )
        uniform = CAL.plan_parameters(CAL.GainModel(phase = (bp = bp(rw),)), names, geom)
        layout = CAL.plan_parameters(m, names, geom)
        plan = layout.plantree.phase.bp
        @test plan isa CAL.ComponentPlan
        @test layout.nθ == uniform.nθ && plan.range == uniform.plantree.phase.bp.range
        @test plan.priors == [(Frequency = rw,), (Frequency = ou,), (Frequency = rw,)]
        @test eltype(uniform.plantree.phase.bp.priors) == typeof((Frequency = rw,))
        @test CAL.require_station_uniform(m, names, "test") === m

        # With a second signature, each group still carries its stations' own priors.
        m2 = CAL.with_station(m, "CC"; phase = (bp = bp(nothing; Ti = CAL.PerScan()),))
        g = CAL.plan_parameters(m2, names, geom).plantree.phase.bp
        @test g isa CAL.GroupedComponentPlan && g.stations == [[1, 2], [3]]
        @test g.groups.g1.priors == [(Frequency = rw,), (Frequency = ou,)]
        @test g.groups.g2.priors == [nothing]

        mat = CAL.resolve(m, names, geom)
        @test mat.phase.bp.prior === rw
        soln = CAL.CalibrationSolution(mat, layout, geom, collect(1.0:layout.nθ))
        path = joinpath(mktempdir(), "prior.zarr")
        CAL.save_solution(path, soln)
        back = CAL.load_solution(path)
        @test back.components == soln.components
        @test only(back.components).component == mat.phase.bp
    end
end

@testset "Calibration: misdeclared coordinate term errors loudly (N3)" begin
    # A new term that declares the Frequency axis but defines no
    # freq_coordinate must error at plan time, not silently evaluate at x = 0.
    @eval CAL begin
        struct _AuditBadFreqTerm <: AbstractGainTerm end
        term_axes(::_AuditBadFreqTerm) = (Frequency,)
        param_shapes(::_AuditBadFreqTerm, n) = (scale = (),)
        # NOTE: deliberately no freq_coordinate method.
    end
    geom = CAL.DataGeometry(; nfeed = 2, times = [0.0], channel_freqs = [1.0e9, 2.0e9])
    model = CAL.GainModel(
        phase = (bad = CAL.GainComponent(CAL._AuditBadFreqTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),),
    )
    @test_throws MethodError CAL.plan_parameters(model, 1, geom)

    # A dimension no term coordinate is built along fails at plan time, named.
    @eval CAL begin
        struct _AuditAntTerm <: AbstractGainTerm end
        term_axes(::_AuditAntTerm) = ($(Gustavo.AntennaName),)
        param_shapes(::_AuditAntTerm, n) = (scale = (),)
    end
    antmodel = CAL.GainModel(
        phase = (bad = CAL.GainComponent(CAL._AuditAntTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),),
    )
    @test_throws "declares unknown coordinate axes (:AntennaName,)" CAL.plan_parameters(antmodel, 1, geom)
end

@testset "CalibrationSolution parameters keep θ's element type" begin
    nant = 3
    freqs = collect(2.28e11:1.0e8:(2.28e11 + 5.0e8))
    times = [0.0, 0.5, 1.0]
    geom = CAL.DataGeometry(; nfeed = 2,
        times, channel_freqs = freqs, stations = ["A", "B", "C"], t0 = 0.0, f0 = sum(freqs) / length(freqs),
    )
    model = CAL.GainModel(
        phase = (
            delay = CAL.GainComponent(CAL.Delay(); Ti = CAL.GlobalTime(), Frequency = CAL.GlobalFrequency(), Feed = CAL.PerFeed()),
            bandpass = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.ChannelBlocks(1), Feed = CAL.SharedFeeds()),
        ),
    )
    layout = CAL.plan_parameters(model, nant, geom)
    θv = collect(1:(layout.nθ)) ./ 1.0e10

    solv = CAL.CalibrationSolution(model, layout, geom, θv, (; nant))
    delay(s) = only(s[:solution, :phase, :delay].components).params
    @test eltype(delay(solv)) == Float64

    @testset "a DimArray θ gives the same components and gains" begin
        θd = DimArray(copy(θv), Dim{:param}(1:(layout.nθ)))
        sold = CAL.CalibrationSolution(model, layout, geom, θd, (; nant))
        @test sold.components == solv.components
        @test parent(gains(sold)) == parent(gains(solv))
    end

    @testset "gains(sol) labels the forward map for inspection" begin
        ev = layout
        g = gains(solv)
        @test g isa DimArray
        @test size(g) == (length(freqs), length(times), nant, 2)
        # The same numbers `evaluate_gains` returns.
        @test parent(g) == CAL.evaluate_gains(ev, θv)
        # Axes carry the geometry, so a user can index by physical coordinate.
        @test lookup(g, Gustavo.Frequency) == freqs
        @test lookup(g, Ti) == times
        @test lookup(g, Gustavo.AntennaName) == geom.stations
        @test lookup(g, Gustavo.Feed) == 1:2
        # amp/phase recover from the complex gain, no separate accessor needed.
        @test abs.(g) == abs.(CAL.evaluate_gains(ev, θv))
        @test angle.(g) == angle.(CAL.evaluate_gains(ev, θv))
        # Indifferent to θ's array type.
        θd = DimArray(copy(θv), Dim{:param}(1:(layout.nθ)))
        @test gains(CAL.CalibrationSolution(model, layout, geom, θd, (; nant))) == g
    end

    @testset "gains(sol, win) at a window of the solve grid or of other data" begin
        ci, ti = [2, 5], [1, 3]
        gw = gains(solv, CAL.GeometryWindow(geom, ci, ti))
        @test parent(gw) == parent(gains(solv))[ci, ti, :, :]
        @test lookup(gw, Gustavo.Frequency) == freqs[ci]
        @test lookup(gw, Ti) == times[ti]
        # Another geometry holding the same samples places each in the same
        # solve segment.
        other = CAL.DataGeometry(; nfeed = 2, times, channel_freqs = freqs, stations = geom.stations, t0 = 0.0, f0 = geom.f0)
        @test parent(gains(solv, CAL.GeometryWindow(other, ci, ti))) ≈ parent(gw)
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
        s32 = CAL.CalibrationSolution(model, layout, geom, Float32.(θv), (;))
        @test all(c -> eltype(c.params) == Float32, s32.components)
        @test eltype(gains(s32)) == ComplexF32
    end

    @testset "the solution copies θ rather than aliasing it" begin
        θmut = copy(θv)
        s = CAL.CalibrationSolution(model, layout, geom, θmut, (;))
        θmut[1] = -999.0
        @test delay(s)[1] == θv[1]
    end

    @testset "an edited parameter value is what gains evaluate" begin
        s = CAL.CalibrationSolution(model, layout, geom, θv, (; nant))
        delay(s) .*= 2
        θ2 = copy(θv)
        θ2[layout.plantree.phase.delay.range] .*= 2
        @test parent(gains(s)) == CAL.evaluate_gains(layout, θ2)
    end

    @testset "parameters whose labels or shape differ from their component are refused" begin
        c = only(solv[:solution, :phase, :bandpass].components)
        d = dims(c.params)
        relabeled = CAL.SolvedComponent(c.step, c.path, c.component, DimArray(parent(c.params), (d[1:3]..., Ti([9.0]), d[5])))
        @test_throws "Ti axis is [9.0]" gains(CAL.CalibrationSolution(geom, [relabeled]))
        other_stations = CAL.SolvedComponent(c.step, c.path, c.component, DimArray(parent(c.params), (d[1:4]..., Gustavo.AntennaName(["A", "B", "Z"]))))
        @test_throws "station `Z` is not among" gains(CAL.CalibrationSolution(geom, [other_stations]))
        short = CAL.SolvedComponent(c.step, c.path, c.component, c.params[Ti(1:1), Gustavo.Frequency(1:2)])
        @test_throws DimensionMismatch gains(CAL.CalibrationSolution(geom, [short]))
    end

    @testset "a component over some stations leaves the others at unit gain" begin
        c = only(solv[:solution, :phase, :delay].components)
        sub = CAL.SolvedComponent(c.step, c.path, c.component, c.params[Gustavo.AntennaName(At(["A", "C"]))])
        g = parent(gains(CAL.CalibrationSolution(geom, [sub])))
        full = parent(gains(solv[:solution, :phase, :delay]))
        @test all(==(1), g[:, :, 2, :])
        @test g[:, :, [1, 3], :] == full[:, :, [1, 3], :]
    end
end

# The `Ti` axis is in seconds, so a segmentation's width is too. A `Period`
# says which unit the caller meant; the stored field stays a bare Float64, so
# equality and hashing are untouched by the spelling.
@testset "segmentations accept a Period" begin
    @test CAL.TimeBlocks(Minute(10)) == CAL.TimeBlocks(600.0)
    @test CAL.TimeBlocks(Second(30)).duration_s == 30.0
    @test hash(CAL.TimeBlocks(Minute(10))) == hash(CAL.TimeBlocks(600.0))
    # A Float64 duration has ~1e-23 s of resolution at this magnitude, so the
    # finest `Dates` unit is exact.
    @test CAL.TimeBlocks(Nanosecond(100)).duration_s == 1.0e-7
    # Month has no fixed length, so it cannot name a width.
    @test_throws MethodError CAL.TimeBlocks(Month(1))

    # InstrumentScans boundaries are epochs on the axis, not widths.
    t0 = DateTime(2021, 3, 4, 1, 0, 0)
    @test CAL.InstrumentScans([t0]) == CAL.InstrumentScans([datetime2unix(t0)])
    @test CAL.InstrumentScans([3.0, 1.0]).boundaries_s == [1.0, 3.0]
    @test CAL.InstrumentScans([3, 1]).boundaries_s == [1.0, 3.0]
    @test CAL.InstrumentScans([t0 + Hour(1), t0]).boundaries_s ==
        datetime2unix.([t0, t0 + Hour(1)])
end

@testset "argument validation is typed" begin
    # Constructor and argument validation throws `ArgumentError` or
    # `DimensionMismatch`; bare `error` is reserved for algorithmic failure, so a
    # caller can tell "you passed me nonsense" from "the solve did not converge".
    @testset "segmentation constructors" begin
        @test_throws ArgumentError CAL.TimeBlocks(0.0)
        @test_throws "duration_s must be positive" CAL.TimeBlocks(-1.0)
        @test_throws ArgumentError CAL.ChannelBlocks(0)
        @test_throws "block_size must be at least 1" CAL.ChannelBlocks(-2)
        @test_throws ArgumentError CAL.FreqGroups(UnitRange{Int}[])
        @test_throws "at least one range" CAL.FreqGroups(UnitRange{Int}[])
        @test_throws "must start at channel 1" CAL.FreqGroups([2:4])
        @test_throws "contiguous and ascending" CAL.FreqGroups([1:4, 6:8])
    end

    @testset "geometry axis lengths" begin
        @test_throws DimensionMismatch CAL.DataGeometry(; nfeed = 2,
            times = [0.0, 1.0], channel_freqs = [1.0e9], scan_of_time = [1]
        )
        @test_throws "scan_of_time length" CAL.DataGeometry(; nfeed = 2,
            times = [0.0, 1.0], channel_freqs = [1.0e9], scan_of_time = [1]
        )
        @test_throws DimensionMismatch CAL.DataGeometry(; nfeed = 2,
            times = [0.0], channel_freqs = [1.0e9, 2.0e9], spw_of_chan = [1]
        )
        @test_throws "spw_of_chan length" CAL.DataGeometry(; nfeed = 2,
            times = [0.0], channel_freqs = [1.0e9, 2.0e9], spw_of_chan = [1]
        )

        # `FreqGroups` is structurally valid but must also cover the geometry.
        geom = CAL.DataGeometry(; nfeed = 2, times = [0.0], channel_freqs = collect(1.0:6.0) .* 1.0e9)
        @test_throws DimensionMismatch CAL.freq_segment_ids(CAL.FreqGroups([1:4]), geom)
        @test_throws "geometry has 6" CAL.freq_segment_ids(CAL.FreqGroups([1:4]), geom)
    end

    @testset "feed tying and empty models" begin
        @test CAL.SingleFeed(3).feed == 3
        @test_throws "feed must be at least 1" CAL.SingleFeed(0)
        @test_throws ArgumentError CAL.validate_gain_model(CAL.GainModel())
        @test_throws "neither phase nor log-amplitude" CAL.validate_gain_model(
            CAL.GainModel()
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
    # to report however steep the trend that cluster is centered on. Steepness is a
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

@testset "Per-station heterogeneity" begin
    geom = CAL.DataGeometry(; nfeed = 2,
        times = [0.0, 0.1, 1.0, 1.1],
        scan_of_time = [1, 1, 2, 2],
        channel_freqs = collect(1.0:6.0) .* 1.0e9,
        spw_of_chan = [1, 1, 1, 2, 2, 2], stations = ["AA", "BB", "CC"],
        t0 = 0.0, f0 = 3.5e9,
    )
    names = ["AA", "BB", "CC"]
    bp(seg) = CAL.GainComponent(
        CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = seg, Feed = CAL.PerFeed(),
    )
    atmos = CAL.GainComponent(
        CAL.ConstantTerm(); Ti = CAL.PerScan(), Frequency = CAL.GlobalFrequency(),
        Feed = CAL.SharedFeeds(),
    )
    base = (; phase = (bandpass = bp(CAL.ChannelBlocks(1)), atmos))

    @testset "stations constructor and seam" begin
        m = CAL.GainModel(;
            base..., stations = (AA = (; phase = (bandpass = bp(CAL.GlobalFrequency()),)),),
        )
        # Replacement is verbatim and whole-group: AA's phase tree is the
        # entry's alone (no `atmos`), its logamp inherited from the base.
        aa = CAL.station_components(m, "AA")
        @test keys(aa.phase) == (:bandpass,)
        @test aa.phase.bandpass.Frequency isa CAL.GlobalFrequency
        @test aa.logamp == m.logamp
        # A station without an entry gets the base pair, and Symbol/String
        # address the same station.
        @test CAL.station_components(m, "BB") == (; phase = m.phase, logamp = m.logamp)
        @test CAL.station_components(m, :AA) == aa

        # Entry keys other than phase/logamp error — the likely mistake is a
        # component name at the top level.
        @test_throws "unexpected key" CAL.GainModel(;
            base..., stations = (AA = (; bandpass = bp(CAL.GlobalFrequency())),),
        )
        @test_throws "must be a `(; phase, logamp)` NamedTuple" CAL.GainModel(;
            base..., stations = (AA = bp(CAL.GlobalFrequency()),),
        )

        # Equality and hashing are order-insensitive in `stations`.
        ma = CAL.GainModel(;
            base..., stations = (AA = (; phase = (;)), BB = (; logamp = (;))),
        )
        mb = CAL.GainModel(;
            base..., stations = (BB = (; logamp = (;)), AA = (; phase = (;))),
        )
        @test ma == mb
        @test hash(ma) == hash(mb)
        @test ma != CAL.GainModel(; base...)
    end

    @testset "with_station" begin
        m = CAL.GainModel(; base...)
        cc = (bandpass = bp(CAL.GlobalFrequency()),)
        mc = CAL.with_station(m, "CC"; phase = cc)
        @test mc == CAL.GainModel(; base..., stations = (CC = (; phase = cc),))
        @test m == CAL.GainModel(; base...)          # the input is unchanged
        # A later call keeps the groups it does not name; Symbol/String agree.
        la = (amp = atmos,)
        mcl = CAL.with_station(mc, :CC; logamp = la)
        @test CAL.station_components(mcl, "CC") == (; phase = cc, logamp = la)
        @test CAL.with_station(mcl, "CC"; phase = base.phase).stations.CC ==
            (; phase = base.phase, logamp = la)
        @test_throws "unexpected key" CAL.with_station(m, "CC"; bandpass = bp(CAL.GlobalFrequency()))
    end

    @testset "merge" begin
        m = CAL.with_station(CAL.GainModel(; base...), "CC"; phase = (; bandpass = bp(CAL.GlobalFrequency())))
        amp = (; amp = atmos)
        mm = merge(m; phase = (; atmos = bp(CAL.GlobalFrequency())), logamp = amp)
        # A name the model has is replaced in place; a new name is appended.
        @test keys(mm.phase) == (:bandpass, :atmos)
        @test mm.phase.atmos == bp(CAL.GlobalFrequency())
        @test mm.logamp == amp
        @test mm.stations == m.stations                       # entries are kept
        @test merge(m) == m
        @test_throws "must be named" merge(m; phase = (atmos,))
        @test_throws "must be a `GainComponent`" merge(m; phase = (; x = 1.0))
    end

    @testset "resolve" begin
        # Segmentations resolve inside per-station trees, and an entry
        # equal to the base (after resolution) collapses away.
        m = CAL.GainModel(
            phase = (bandpass = bp(CAL.BandGroups()), atmos),
            stations = (
                BB = (; phase = (bandpass = bp(CAL.BandGroups()), atmos)),
                CC = (; phase = (bandpass = bp(CAL.GlobalFrequency()),)),
            ),
        )
        mat = CAL.resolve(m, names, geom)
        @test mat.phase.bandpass.Frequency isa CAL.FreqGroups
        @test keys(mat.stations) == (:CC,)
        @test CAL.resolve(mat, names, geom) == mat        # idempotent

        # Unknown station codes error, naming the known stations.
        bad = CAL.GainModel(; base..., stations = (XX = (; phase = base.phase),))
        @test_throws "unknown station :XX" CAL.resolve(bad, names, geom)
        @test_throws "AA, BB, CC" CAL.resolve(bad, names, geom)
    end

    mu = CAL.GainModel(; base...)
    mh = CAL.GainModel(;
        base...,
        stations = (AA = (; phase = (bandpass = bp(CAL.GlobalFrequency()), atmos)),),
    )

    @testset "uniform model reproduces the rectangular layout exactly" begin
        l1 = CAL.plan_parameters(mu, 3, geom)
        l2 = CAL.plan_parameters(mu, names, geom)
        @test l1.nθ == l2.nθ
        @test [p.range for p in l1.plans] == [p.range for p in l2.plans]
        @test [p.shape for p in l1.plans] == [p.shape for p in l2.plans]
        @test typeof(l1.plantree) == typeof(l2.plantree)
        @test l1.axes == l2.axes
        # The count form cannot resolve station codes, so a model carrying
        # `stations` demands the antenna table.
        @test_throws "antenna COUNT" CAL.plan_parameters(mh, 3, geom)
    end

    lh = CAL.plan_parameters(mh, names, geom)

    @testset "canonicalizer groups by signature" begin
        node = lh.plantree.phase.bandpass
        @test node isa CAL.GroupedComponentPlan
        @test keys(node.groups) == (:g1, :g2)
        @test node.stations == [[1], [2, 3]]
        @test node.group_of == [1, 2, 2]
        @test node.local_of == [1, 1, 2]
        # Ragged: AA's global-frequency block vs the others' per-channel one.
        @test node.groups.g1.shape == (1, 2, 1, 1, 1)
        @test node.groups.g2.shape == (1, 2, 6, 1, 2)
        # θ leaves nest under the group keys and spans stay contiguous.
        @test length(node.groups.g1.range) == 2
        @test node.groups.g2.range == last(node.groups.g1.range) .+ (1:24)
        # A component every station shares (here at a different resolution per
        # the entry, but with an identical signature) stays a plain plan with
        # the full antenna axis: the grouped machinery has zero footprint on it.
        @test lh.plantree.phase.atmos isa CAL.ComponentPlan
        @test lh.plantree.phase.atmos.shape[5] == 3

        # A name that is a leaf at one station and a subtree at another has no
        # honest layout.
        conflict = CAL.GainModel(;
            base...,
            stations = (AA = (; phase = (bandpass = (; a = bp(CAL.GlobalFrequency())),)),),
        )
        @test_throws "disagree structurally" CAL.plan_parameters(conflict, names, geom)
    end

    @testset "evaluator routes stations through their groups" begin
        θ = zeros(lh.nθ)
        g1, g2 = lh.plantree.phase.bandpass.groups
        CAL._component_leaf(g1, θ) .= 0.5                    # AA: one phase, all channels
        CAL._component_leaf(g2, θ)[1, 1, 3, 1, 2] = 0.25     # CC (local index 2), feed 1, chan 3
        ev = lh
        g = @inferred CAL.evaluate_gains(ev, θ)
        @test size(g) == (6, 4, 3, 2)
        @test all(angle.(g[:, :, 1, :]) .≈ 0.5)
        @test angle(g[3, 1, 3, 1]) ≈ 0.25
        @test angle(g[2, 1, 3, 1]) ≈ 0.0
        @test all(angle.(g[:, :, 2, :]) .≈ 0.0)

        # The windowed and foreign-grid maps run the same grouped walk.
        gw = CAL.evaluate_gains(ev, θ, [3], [1])
        @test angle(gw[1, 1, 3, 1]) ≈ 0.25
        gf = CAL.evaluate_gains(ev, θ, geom, geom)
        @test gf ≈ g
    end

    @testset "station_blocks" begin
        θ = collect(1.0:lh.nθ)
        blocks = CAL.station_blocks(lh, θ, :phase, :bandpass)
        @test length(blocks) == 2
        @test blocks[1].stations == [1]
        @test blocks[2].stations == [2, 3]
        @test size(blocks[2].θ) == (1, 2, 6, 1, 2)
        @test vec(blocks[1].θ) == θ[blocks[1].plan.range]
        # Writing through a block's θ view writes into the solve's θ.
        blocks[1].θ[1] = -3.0
        @test θ[first(blocks[1].plan.range)] == -3.0
        # A uniform component yields one block spanning every station.
        ub = CAL.station_blocks(lh, θ, :phase, :atmos)
        @test length(ub) == 1
        @test ub[1].stations == [1, 2, 3]
        # Path errors: an unknown name, and a stop at a subtree.
        @test_throws "no component at path" CAL.station_blocks(lh, θ, :phase, :nope)
        @test_throws "names a component subtree" CAL.station_blocks(lh, θ, :phase)
    end

    @testset "require_station_uniform" begin
        mat = CAL.resolve(mh, names, geom)
        err = try
            CAL.require_station_uniform(mat, names, "Bandpass")
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("station-uniform", err.msg)
        @test occursin("phase.bandpass", err.msg)
        @test occursin("[AA]", err.msg)
        @test occursin("[BB, CC]", err.msg)
        @test occursin("supports_station_heterogeneity", err.msg)
        # `atmos` is identical everywhere, so it is not reported.
        @test !occursin("atmos", err.msg)
        # A uniform model passes through, entries collapsed or not.
        @test CAL.require_station_uniform(
            CAL.resolve(mu, names, geom), names, "Bandpass",
        ) isa CAL.GainModel
    end

    @testset "components and gains of a grouped component" begin
        θh = collect(1.0:lh.nθ)
        soln = CAL.CalibrationSolution(CAL.resolve(mh, names, geom), lh, geom, θh)
        # The grouped name is one component per signature group; each group's
        # parameters carry ITS stations on the AntennaName axis and its own specification.
        bp = soln[:solution, :phase, :bandpass]
        @test [c.path for c in bp.components] == [(:phase, :bandpass, :g1), (:phase, :bandpass, :g2)]
        c1, c2 = bp.components
        @test lookup(c1.params, Gustavo.AntennaName) == ["AA"]
        @test lookup(c2.params, Gustavo.AntennaName) == ["BB", "CC"]
        @test vec(parent(c2.params)) == θh[lh.plantree.phase.bandpass.groups.g2.range]
        @test c1.component != c2.component
        # A group's gain is identity off its own stations, and the groups'
        # product reproduces the whole name.
        gg1 = parent(gains(soln[:solution, :phase, :bandpass, :g1]))
        gg2 = parent(gains(soln[:solution, :phase, :bandpass, :g2]))
        @test all(gg1[:, :, 2:3, :] .== 1)
        @test all(gg2[:, :, 1:1, :] .== 1)
        @test gg1 .* gg2 ≈ parent(gains(bp))
        # The whole solution evaluates as the layout it came from, on its own
        # geometry or on another holding the same samples.
        @test parent(gains(soln)) ≈ CAL.evaluate_gains(lh, θh)
        other = CAL.DataGeometry(; nfeed = 2,
            geom.times, geom.channel_freqs, geom.scan_of_time, geom.spw_of_chan, geom.stations, geom.t0, geom.f0,
        )
        ci, ti = [2, 5], [1, 4]
        @test parent(gains(bp, CAL.GeometryWindow(other, ci, ti))) ≈ parent(gains(bp))[ci, ti, :, :]
        # The uniform component still labels with the full station list.
        @test lookup(only(soln[:solution, :phase, :atmos].components).params, Gustavo.AntennaName) == names

        # A station-heterogeneous solution round-trips through `save_solution`.
        path = joinpath(mktempdir(), "het.zarr")
        CAL.save_solution(path, soln)
        back = CAL.load_solution(path)
        @test back.components == soln.components
        @test parent(gains(back)) == parent(gains(soln))
    end
end
