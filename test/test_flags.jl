using Test

# `FLAG` and `WEIGHT` answer different questions — "must this datum be used?"
# and "how good is it?" — so a positive weight must not smuggle a flagged sample
# back in. `test_uvfits.jl` pins the two apart where the reader writes them.
@testset "flag layer" begin
    FP = Gustavo.Fring

    @testset "the search honors the flag, not the weight" begin
        # Flagging must be exactly equivalent to deleting the samples, and a
        # positive weight must not smuggle them back in.
        freqs = 43.0e9 .+ (0:31) .* 0.5e6
        times = collect(0:7) .* 1.0
        delay, rate = 1.5e-9, 2.0e-3
        fringe(fs, ts, d, r) = ComplexF32[
            cis(2π * (d * (f - freqs[1]) + r * (t - times[1]))) for f in fs, t in ts
        ]
        V = fringe(freqs, times, delay, rate)
        W = ones(Float32, size(V))

        # Corrupt the first half of the band with a contradictory delay, leave its
        # weight untouched, and flag it.
        Vc = copy(V)
        Vc[1:16, :] = fringe(freqs[1:16], times, -8.0e-9, rate)
        fl = falses(size(V))
        fl[1:16, :] .= true

        honored = FP.baseline_fringe_search(
            FP.fringe_plane(Vc, W, freqs, times; flags = fl), freqs[1], times[1],
        )
        ignored = FP.baseline_fringe_search(
            FP.fringe_plane(Vc, W, freqs, times), freqs[1], times[1],
        )
        @test isapprox(honored.delay, delay; atol = 2.0e-11)
        # Reading the corrupted half lands somewhere else entirely.
        @test !isapprox(ignored.delay, delay; atol = 1.0e-9)

        # Deleting the flagged half outright gives the same answer as flagging it.
        deleted = FP.baseline_fringe_search(
            FP.fringe_plane(Vc[17:32, :], W[17:32, :], freqs[17:32], times),
            freqs[1], times[1],
        )
        @test isapprox(honored.delay, deleted.delay; atol = 2.0e-11)
    end
end
