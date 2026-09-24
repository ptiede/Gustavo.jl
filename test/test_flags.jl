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

    _with_flags(leaf, f) = UV.rebuild_visibilities(
        leaf, parent(leaf[:vis]), parent(leaf[:weights]), parent(leaf[:uvw]), f,
    )

    # A leaf carrying both a flagged sample with a positive weight and an
    # unflagged sample with zero weight: the two configurations a single-layer
    # model cannot express.
    function _crossed_leaf(leaf)
        w = copy(parent(leaf[:weights]))
        f = copy(parent(leaf[:flags]))
        f[1, 1, 1, 1] = true        # flagged, weight left positive
        w[2, 1, 1, 1] = 0           # zero weight, left unflagged
        return UV.rebuild_visibilities(
            leaf, parent(leaf[:vis]), w, parent(leaf[:uvw]), f,
        )
    end

    @testset "every leaf carries the layer" begin
        for leaf in values(UV.leaves(base))
            @test haskey(leaf, :flags)
            @test eltype(leaf[:flags]) === Bool
            @test dims(leaf[:flags]) == dims(leaf[:vis])
        end
    end

    @testset "a leaf built from positive weights is unflagged" begin
        for leaf in values(UV.leaves(base))
            @test all(parent(leaf[:weights]) .> 0)
            @test !any(parent(leaf[:flags]))
        end
    end

    @testset "_build_leaf rejects flags off the vis axes" begin
        leaf = first(values(UV.leaves(base)))
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
        leaf = _crossed_leaf(first(values(UV.leaves(base))))
        @test parent(leaf[:flags])[1, 1, 1, 1]
        @test parent(leaf[:weights])[1, 1, 1, 1] > 0
        @test !parent(leaf[:flags])[2, 1, 1, 1]
        @test parent(leaf[:weights])[2, 1, 1, 1] == 0

        # Round trips that rebuild the leaf must preserve both configurations.
        m = UV.materialize_leaf(leaf)
        @test parent(m[:flags])[1, 1, 1, 1]
        @test parent(m[:weights])[1, 1, 1, 1] > 0
        @test !parent(m[:flags])[2, 1, 1, 1]
        @test parent(m[:weights])[2, 1, 1, 1] == 0

        bl = UV.baseline(leaf, UV.baselines(leaf).labels[1])
        @test parent(bl[:flags])[1, 1, 1]
        @test parent(bl[:weights])[1, 1, 1] > 0
        @test !parent(bl[:flags])[2, 1, 1]
        @test parent(bl[:weights])[2, 1, 1] == 0
    end

    @testset "a reduction flags an output only where every input was flagged" begin
        leaf = first(values(UV.leaves(base)))
        nchan, nti, _, _ = size(parent(leaf[:vis]))
        @test nti >= 2
        f = copy(parent(leaf[:flags]))
        f[1, :, 1, 1] .= true       # flagged in every integration
        f[2, 1, 1, 1] = true        # flagged in one of them only
        one = DD.rebuild(
            base; branches = DD.TreeDict(:only => _with_flags(leaf, f)),
        )

        avg = first(values(UV.leaves(UV.scan_average(one))))
        @test size(parent(avg[:flags]), 2) == 1
        @test parent(avg[:flags])[1, 1, 1, 1]
        @test !parent(avg[:flags])[2, 1, 1, 1]

        # One bin wide enough to swallow the whole scan reduces like the above.
        binned = first(values(UV.leaves(UV.time_bin_average(one, 1.0e6))))
        @test size(parent(binned[:flags]), 2) == 1
        @test parent(binned[:flags])[1, 1, 1, 1]
        @test !parent(binned[:flags])[2, 1, 1, 1]

        # One output channel per input channel leaves the flags untouched;
        # collapsing them all leaves nothing flagged, since channel 1 is not
        # flagged in every integration of the group it now shares.
        same = first(values(UV.leaves(UV.frequency_average(one; nout = nchan))))
        @test parent(same[:flags]) == f
        whole = first(values(UV.leaves(UV.frequency_average(one; nout = 1))))
        @test !any(parent(whole[:flags]))
    end

    @testset "spw-edge handling" begin
        leaf0 = first(values(UV.leaves(base)))
        nchan = size(parent(leaf0[:vis]), 1)
        @test nchan >= 4
        fraction = 1 / nchan

        flagged = UV.flag_spw_edges(base; mode = :flag_fraction, fraction = fraction)
        for leaf in values(UV.leaves(flagged))
            f = parent(leaf[:flags])
            @test all(f[1, :, :, :])
            @test all(f[end, :, :, :])
            @test !any(f[2:(end - 1), :, :, :])
            # Flagging an edge channel does not cost it its weight: clearing
            # the flag has to give the sample back.
            @test all(parent(leaf[:weights])[1, :, :, :] .> 0)
            @test all(parent(leaf[:weights])[end, :, :, :] .> 0)
        end

        # `:trim` carries each surviving channel's flag through unchanged.
        marked = _maptree(base) do leaf
            f = copy(parent(leaf[:flags]))
            f[2, :, :, :] .= true
            return _with_flags(leaf, f)
        end
        trimmed = UV.flag_spw_edges(marked; mode = :trim, fraction = fraction)
        for leaf in values(UV.leaves(trimmed))
            @test size(parent(leaf[:flags]), 1) == nchan - 2
            @test all(parent(leaf[:flags])[1, :, :, :])
            @test !any(parent(leaf[:flags])[2:end, :, :, :])
        end
    end

    @testset "combine_spw concatenates the flags along frequency" begin
        marked = _maptree(base) do leaf
            f = copy(parent(leaf[:flags]))
            UV.metadata(leaf).ddi == 0 && (f[1, :, :, :] .= true)
            return _with_flags(leaf, f)
        end
        combined = UV.combine_spw(marked)
        for leaf in values(UV.leaves(combined))
            f = parent(leaf[:flags])
            nchan = size(parent(first(values(UV.leaves(base)))[:vis]), 1)
            @test size(f, 1) == 2 * nchan
            @test all(f[1, :, :, :])                  # first spw's first channel
            @test !any(f[2:nchan, :, :, :])
            @test !any(f[(nchan + 1):end, :, :, :])   # second spw untouched
        end
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

    @testset "a zero weight is not a flag" begin
        # A zero-weight sample carries no information, so a reduction skips it —
        # but it was never flagged, and the reduction must not claim it was.
        leaf = first(values(UV.leaves(base)))
        w = copy(parent(leaf[:weights]))
        w[1, :, 1, 1] .= 0
        zeroed = UV.rebuild_visibilities(
            leaf, parent(leaf[:vis]), w, parent(leaf[:uvw]), parent(leaf[:flags]),
        )
        one = DD.rebuild(base; branches = DD.TreeDict(:only => zeroed))
        avg = first(values(UV.leaves(UV.scan_average(one))))
        @test parent(avg[:weights])[1, 1, 1, 1] == 0
        @test !parent(avg[:flags])[1, 1, 1, 1]
    end

    @testset "the a-priori kernel flags without spending the weight" begin
        # The two branches that flag: an autocorrelation, which is total power
        # rather than a visibility, and a non-finite gain, which leaves the
        # correction undefined. Neither is a statement about the datum's
        # quality, so neither touches the weight.
        leaf = first(values(UV.leaves(base)))
        # The kernel spans its axes by dimension (`axes(vis, Ti)`), so it takes
        # the leaf's `DimArray`s, not their parents.
        vis = leaf[:vis]
        w = leaf[:weights]
        f = leaf[:flags]
        bl_pairs = UV.baselines(leaf).pairs
        pols = collect(UV.pol_products(leaf))
        nchan, nti, _, _ = size(vis)
        nant = maximum(maximum(p) for p in bl_pairs)

        gains = ones(Float64, nchan, nti, nant, 2)
        gains[1, 1, 1, :] .= NaN                 # station 1 unusable at (chan 1, ti 1)
        auto = findfirst(p -> p[1] == p[2], bl_pairs)
        cross1 = findfirst(p -> p[1] != p[2] && 1 in p, bl_pairs)
        @test cross1 !== nothing

        _, w_out, f_out = UV._apply_apriori_kernel(vis, w, f, gains, bl_pairs, pols)
        @test parent(w_out) == parent(w)
        @test f_out[1, 1, cross1, 1]
        auto === nothing || @test all(f_out[:, :, auto, :])
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
