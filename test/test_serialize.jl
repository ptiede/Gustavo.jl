# Distribution-readiness (Stage-1 refactor, Step 4).
#
# For a Dagger worker to materialize a leaf it was shipped, a lazy `UVSet` must
# (a) serialize/deserialize and then materialize BIT-IDENTICALLY, and (b) expose
# the source path each leaf reopens (the serialization anchor; workers must share
# the filesystem). We also run `pmapreduce` over the lazy set under the
# `DaggerExecutor` — materialization happens inside the Dagger task — and require
# it to equal the serial result. Reuses `build_synth_idi_uvset` from
# test_fitsidi.jl (included earlier in runtests.jl).

using Serialization: serialize, deserialize
using Gustavo: pmapreduce, SerialExecutor, DaggerExecutor

@testset "Serialize: lazy UVSet round-trips + materializes identically" begin
    UV = Gustavo.UVData
    uvset = build_synth_idi_uvset(; nbands = 2, nchan = 4, nscan = 2, ntime = 3)
    path = tempname() * ".idifits"
    try
        UV.write_fitsidi(path, uvset)
        lazy = UV.load_fitsidi(path; lazy = true)
        @test UV.is_lazy(lazy)

        # Every lazy leaf exposes its source path (eager leaves would give nothing).
        for (_, leaf) in UV.branches(lazy)
            @test UV.leaf_source_path(leaf) == path
        end

        # Serialize → deserialize the WHOLE lazy UVSet; leaves stay lazy.
        buf = IOBuffer()
        serialize(buf, lazy)
        seekstart(buf)
        round = deserialize(buf)
        @test round isa UV.UVSet
        @test UV.is_lazy(round)
        @test collect(keys(UV.branches(round))) == collect(keys(UV.branches(lazy)))
        for (_, leaf) in UV.branches(round)
            @test UV.leaf_source_path(leaf) == path
        end

        # Materializing the deserialized set reproduces the original exactly
        # (isequal so NaN "missing" cells compare equal).
        mat_orig = UV.materialize(lazy)
        mat_round = UV.materialize(round)
        for k in keys(UV.branches(mat_orig))
            lo = UV.branches(mat_orig)[k]
            lr = UV.branches(mat_round)[k]
            @test isequal(parent(lo[:vis]), parent(lr[:vis]))
            @test isequal(parent(lo[:weights]), parent(lr[:weights]))
        end
    finally
        isfile(path) && rm(path)
    end
end

@testset "Serialize: DaggerExecutor over a lazy UVSet equals serial" begin
    UV = Gustavo.UVData
    uvset = build_synth_idi_uvset(; nbands = 2, nchan = 4, nscan = 2, ntime = 3)
    path = tempname() * ".idifits"
    try
        UV.write_fitsidi(path, uvset)
        lazy = UV.load_fitsidi(path; lazy = true)
        @test UV.is_lazy(lazy)

        # The substrate materializes each group inside the (Dagger) task; the node
        # sees an eager leaf. Reduce = total finite visibility power over the set.
        node = function (leaf)
            v = parent(leaf[:vis])
            s = 0.0
            @inbounds for x in v
                isfinite(x) && (s += abs2(Float64(real(x))) + abs2(Float64(imag(x))))
            end
            return s
        end

        ser = pmapreduce(node, +, lazy; executor = SerialExecutor())
        dag = pmapreduce(node, +, lazy; executor = DaggerExecutor())
        @test ser > 0
        @test dag ≈ ser
    finally
        isfile(path) && rm(path)
    end
end
