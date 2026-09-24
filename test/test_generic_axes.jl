using Test
using Random
using OffsetArrays: OffsetArray
using DimensionalData: DimArray
using Gustavo.UVData: Frequency, Ti, BaselineID, Polarization

# An `AbstractArray` annotation promises the function works for any array — any
# axes, and lazy wrappers as well as `Array`. These tests hold the package to
# that promise where it makes it, and pin the refusal where it does not: a
# function that is honestly 1-based must say so and throw, not read the wrong
# cells. `_baseline_delay` did the latter under `@inbounds`, returning a
# plausible delay built partly from memory past the end of its input.
@testset "generic axes" begin
    UV = Gustavo.UVData
    FR = Gustavo.Fring
    CAL = Gustavo.Calibration
    ST = Gustavo.Streaming

    shift(a) = OffsetArray(a, ntuple(_ -> -2, ndims(a)))
    whole(a) = view(a, ntuple(_ -> Colon(), ndims(a))...)

    @testset "the leaf boundary is what enforces 1-based layers" begin
        # Every solver kernel reads its layers off a `DimArray`, whose lookups
        # are 1-based, so an offset layer cannot reach them. The refusal lives
        # here; this is the test that says so.
        V = randn(ComplexF64, 4, 3, 2, 2)
        d = (
            Frequency(collect(1.0:4.0)), Ti(collect(1.0:3.0)),
            BaselineID(["a", "b"]), Polarization(["PP", "QQ"]),
        )
        @test_throws DimensionMismatch DimArray(OffsetArray(V, -1, -1, 0, 0), d)
        # A lazy wrapper is not an offset and must still be accepted.
        @test parent(DimArray(whole(V), d)) === whole(V)
    end

    @testset "phase/amplitude relative to a reference track their input's axes" begin
        ph = [0.1, 0.5, NaN, 1.2, -0.3]
        am = [2.0, 4.0, NaN, 1.0, 8.0]
        for (f, x) in ((UV.phase_relative_to_ref, ph), (UV.amplitude_relative_to_ref, am))
            ref = f(x)
            out = f(shift(x))
            @test collect(out) ≈ collect(ref) nans = true
            @test axes(out) == axes(shift(x))
            @test collect(f(whole(x))) ≈ collect(ref) nans = true
        end
        # `ref_idx` names a position in the input's own axes, not a count from
        # the start of storage: the shifted vector's first index picks the same
        # reference that `1` picks on the plain one, and an index outside those
        # axes is refused the same way.
        @test collect(UV.phase_relative_to_ref(shift(ph), firstindex(shift(ph)))) ≈
            collect(UV.phase_relative_to_ref(ph, 1)) nans = true
        @test all(isnan, UV.phase_relative_to_ref(shift(ph), lastindex(shift(ph)) + 1))
        @test all(isnan, UV.phase_relative_to_ref(shift(ph), firstindex(shift(ph)) - 1))
    end

    @testset "station_weight_scale tracks its name vector's axes" begin
        names = ["AA", "BB", "CC"]
        factors = Dict("BB" => 2.0)
        ref = ST.station_weight_scale(names, factors)
        out = ST.station_weight_scale(shift(names), factors)
        @test collect(out) == collect(ref)
        @test axes(out) == axes(shift(names))
        @test ST.station_weight_scale(whole(names), factors) == ref
    end

    @testset "the per-baseline delay reads its own indices" begin
        freqs = collect(range(2.2e10, step = 2.0e6, length = 8))
        z = cis.(range(0.0, 2.1, length = 8))
        ref = FR._baseline_delay(z, freqs)
        # A scalar out of a whole-vector reduction: the value cannot depend on
        # where the caller's axes happen to start.
        @test FR._baseline_delay(shift(z), shift(freqs)) == ref
        @test FR._baseline_delay(whole(z), whole(freqs)) == ref
        # The two are read at one index, so mismatched axes are an error rather
        # than a silent read of whichever cells line up.
        @test_throws DimensionMismatch FR._baseline_delay(z, freqs[1:(end - 1)])
    end

    @testset "the 1-based paths declare themselves" begin
        y = randn(MersenneTwister(7), 12)
        @test_throws ArgumentError CAL.savitzky_golay_smooth(shift(y); window = 5)
        @test_throws "offset arrays are not supported" CAL.savitzky_golay_smooth(
            shift(y); window = 5
        )
        @test CAL.savitzky_golay_smooth(whole(y); window = 5) ==
            CAL.savitzky_golay_smooth(y; window = 5)

        # `fringe_plane`'s lookups are the caller's `freqs`/`times`, so the
        # block is read 1-based. Both the matrix and the vector path refuse —
        # the vector path reshapes, which would otherwise drop the offset.
        freqs = collect(range(2.2e10, step = 2.0e6, length = 6))
        times = [0.0, 4.0]
        Vm = randn(MersenneTwister(8), ComplexF64, 6, 2)
        Wm = fill(1.0, 6, 2)
        @test_throws ArgumentError FR.fringe_plane(shift(Vm), shift(Wm), freqs, times)
        @test_throws ArgumentError FR.fringe_plane(
            Vm, Wm, freqs, times; flags = shift(falses(6, 2))
        )
        Vv = Vm[:, 1]
        @test_throws ArgumentError FR.fringe_plane(shift(Vv), shift(Wm[:, 1]), freqs, [0.0])
    end

    @testset "the fringe search is indifferent to lazy wrappers" begin
        rng = MersenneTwister(42)
        nf, nt = 16, 8
        freqs = collect(range(2.2e10, step = 2.0e6, length = nf))
        times = collect(range(0.0, step = 4.0, length = nt))
        f0, t0 = first(freqs), first(times)
        V = [
            cis(2π * (3.0e-9 * (f - f0) + 0.012 * (t - t0)) + 0.7) +
                0.05 * randn(rng, ComplexF64) for f in freqs, t in times
        ]
        W = fill(1.0, nf, nt)
        opts = FR.FringeSearch(rate_window = (-0.1, 0.1))
        d(v, w) = FR.baseline_fringe_search(FR.fringe_plane(v, w, freqs, times), f0, t0; opts)
        ref = d(V, W)

        # A contiguous view and a strided one carry the same data, so they must
        # give the same answer to the last bit — not merely to a tolerance.
        got = d(whole(V), whole(W))
        @test (got.delay, got.rate, got.phase) == (ref.delay, ref.rate, ref.phase)

        Vs = zeros(ComplexF64, 2nf, 2nt)
        Ws = zeros(2nf, 2nt)
        Vs[1:2:end, 1:2:end] .= V
        Ws[1:2:end, 1:2:end] .= W
        gs = d(view(Vs, 1:2:2nf, 1:2:2nt), view(Ws, 1:2:2nf, 1:2:2nt))
        @test (gs.delay, gs.rate, gs.phase) == (ref.delay, ref.rate, ref.phase)
    end
end
