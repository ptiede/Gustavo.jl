using Test
using DimensionalData
using DimensionalData: DimArray, dims, lookup, Ti
using Gustavo.UVData: Polarization, Frequency, BaselineID

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

# `:flags` and `:weights` answer different questions — "must this datum be used?"
# and "how good is it?" — so every test here pins the two apart. A leaf whose
# flags could be recovered from `weights .<= 0` would pass a weaker suite and
# still have collapsed the two layers back into one.
@testset "flag layer" begin
    UV = Gustavo.UVData
    DD = Gustavo.UVData.DimensionalData

    base, _ = _build_fringe_uvset(
        nant = 3, nspw = 2, nchan = 4, ntime = 3, pol_labels = ["PP", "QQ"],
    )

    # Replace every leaf of `set` by `f(leaf)`, keeping the partition keys.
    function _maptree(f, set)
        return DD.rebuild(
            set; branches = DD.TreeDict(k => f(l) for (k, l) in UV.branches(set)),
        )
    end

    # A leaf carrying both a flagged sample with a positive weight and an
    # unflagged sample with zero weight: the two configurations a single-layer
    # model cannot express.
    function _crossed_leaf(leaf)
        w = copy(parent(leaf[:weights]))
        f = copy(parent(leaf[:flags]))
        f[1, 1, 1, 1] = true        # flagged, weight left positive
        w[2, 1, 1, 1] = 0           # zero weight, left unflagged
        d = dims(leaf[:vis])
        return UV._build_leaf(
            leaf[:vis], DimArray(w, d), leaf[:uvw], DimArray(f, d);
            partition_info = UV.metadata(leaf),
        )
    end

    @testset "every leaf carries the layer" begin
        for leaf in values(UV.branches(base))
            @test haskey(leaf, :flags)
            @test eltype(leaf[:flags]) === Bool
            @test dims(leaf[:flags]) == dims(leaf[:vis])
        end
    end

    @testset "a leaf built from positive weights is unflagged" begin
        for leaf in values(UV.branches(base))
            @test all(parent(leaf[:weights]) .> 0)
            @test !any(parent(leaf[:flags]))
        end
    end

    @testset "_build_leaf rejects flags off the vis axes" begin
        leaf = first(values(UV.branches(base)))
        d = dims(leaf[:vis])
        short = DimArray(
            parent(leaf[:flags])[1:1, :, :, :],
            (Frequency(lookup(d[1])[1:1]), d[2], d[3], d[4]),
        )
        @test_throws "flags must share the vis axes" UV._build_leaf(
            leaf[:vis], leaf[:weights], leaf[:uvw], short;
            partition_info = UV.metadata(leaf),
        )
    end

    @testset "the two layers stay independent" begin
        leaf = _crossed_leaf(first(values(UV.branches(base))))
        @test parent(leaf[:flags])[1, 1, 1, 1]
        @test parent(leaf[:weights])[1, 1, 1, 1] > 0
        @test !parent(leaf[:flags])[2, 1, 1, 1]
        @test parent(leaf[:weights])[2, 1, 1, 1] == 0
    end

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

    @testset "the bridge takes FLAG from the layer, not the weight sign" begin
        crossed = _maptree(_crossed_leaf, base)
        ms = first(values(UV.uvset_to_processingset(crossed)))
        # Gustavo's (Frequency, Ti, BaselineID, Polarization) cell [c, 1, 1, 1] is MSv4's
        # (polarization, frequency, baseline_id, time) cell [1, c, 1, 1].
        @test parent(ms[:flag])[1, 1, 1, 1]
        @test parent(ms[:weight])[1, 1, 1, 1] > 0
        @test !parent(ms[:flag])[1, 2, 1, 1]
        @test parent(ms[:weight])[1, 2, 1, 1] == 0
    end
end
