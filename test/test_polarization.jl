# Graceful missing-polarization helpers (Stage-1 refactor, Step 3).
#
# `pol_index_or` and `coherency_slots` are the format-neutral seam a solver
# reads to assemble a 2×2 coherency / Jones matrix from whatever polarization
# products a leaf actually carries — with `0` marking an absent product. These
# tests exercise the pure `Vector{String}` path (no data fixture needed); the
# DimArray/leaf/UVSet overloads just forward through `pol_products`.

using Gustavo.UVData: pol_index, pol_index_or, coherency_slots

@testset "pol_index_or: present / absent / alias folding" begin
    full = ["PP", "PQ", "QP", "QQ"]
    # Present products resolve identically to pol_index.
    for (i, p) in enumerate(full)
        @test pol_index_or(full, p) == i
        @test pol_index_or(full, p) == pol_index(full, p)
    end
    # Alias folding: circular (R/L) and linear (X/Y) fold onto canonical P/Q.
    @test pol_index_or(full, "RR") == pol_index_or(full, "PP") == 1
    @test pol_index_or(full, "RL") == pol_index_or(full, "PQ") == 2
    @test pol_index_or(full, "XY") == pol_index_or(full, "PQ") == 2
    @test pol_index_or(full, "LL") == pol_index_or(full, "QQ") == 4

    # Absent product returns the default (0) instead of throwing.
    parallel = ["PP", "QQ"]
    @test pol_index_or(parallel, "PQ") == 0
    @test pol_index_or(parallel, "QP") == 0
    @test pol_index_or(parallel, "PP") == 1
    @test pol_index_or(parallel, "QQ") == 2
    @test pol_index_or(parallel, "RL") == 0          # alias of the absent PQ
    @test pol_index_or(parallel, "PQ", -1) == -1     # custom default
    # pol_index still throws on the same absent product.
    @test_throws KeyError pol_index(parallel, "PQ")
end

@testset "coherency_slots: 2×2 layout with gaps" begin
    # Full polarization: [PP PQ; QP QQ] → identity-ish index table.
    full = ["PP", "PQ", "QP", "QQ"]
    @test coherency_slots(full) == [1 2; 3 4]

    # Parallel-hand only: off-diagonals are absent (0).
    @test coherency_slots(["PP", "QQ"]) == [1 0; 0 2]

    # Ordering of the stored products is respected (not assumed canonical).
    @test coherency_slots(["QQ", "PP"]) == [2 0; 0 1]

    # Single product (e.g. Stokes-I-like / one parallel hand).
    @test coherency_slots(["PP"]) == [1 0; 0 0]

    # A cross-hand-only exotic set still lands in the right cells.
    @test coherency_slots(["PQ", "QP"]) == [0 1; 2 0]

    # Stored labels are always canonical PP/PQ/QP/QQ (the readers map both
    # circular *and* linear feeds onto them), so a linear-feed (VGOS)
    # parallel-hand leaf is exactly the ["PP","QQ"] case above. Feed-letter
    # aliases (R/L, X/Y) are a query-side convenience — covered by the
    # pol_index_or testset, not by the stored product set.

    # Always a 2×2 Int matrix regardless of how many products are present.
    for products in (["PP"], ["PP", "QQ"], full)
        cs = coherency_slots(products)
        @test cs isa Matrix{Int}
        @test size(cs) == (2, 2)
    end
end
