# ── Transform chain semantics ────────────────────────────────────────────────
#
# The materialization transform chain (ApplySolution → StationWeightScale →
# FlagChannels + the `CalFunction` open hook): the cube and leaf choke paths
# must agree bit-for-bit through the same chain, user data is never mutated,
# and the view metadata/mutation semantics hold. (The frozen-monolith precal
# oracle these were originally gated against was deleted at M5 after the parity
# gates passed — see test_smoother_step.jl for the end-to-end gates.)

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "Transform chain semantics" begin
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)

    # A real solution to divide out (per-scan delays/rates + per-AP adhoc — the
    # time-VARYING precal branch; `_precal_time_constant` guards the fast path).
    precal = fit(
        FringeFit(model = FringeModel(ref_ant = 1)) |> BandpassEstimator() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset,
    )
    nant = length(UVP.union_antennas(uvset).name)
    ws = [1.0 + 0.25 * i for i in 1:nant]
    nchan = length(geom.channel_freqs)
    mask = falses(nchan); mask[1] = true; mask[end] = true

    chain = [FP.ApplySolution(precal), FP.StationWeightScale(ws), FP.FlagChannels(mask)]
    st = FP.scan_stream(uvset; geom = geom, transforms = chain)

    @testset "cube path ≡ leaf path through the same chain" begin
        for spec in st.groups
            grp = FP.materialize_cube(st, spec; inner = 2)
            keyed = FP.materialize_leaves(st, spec; inner = 2)
            # Rebuild the cube from the transformed leaves: identical data —
            # the two choke paths apply the chain identically.
            grp_l = FP._stacked_scan_group([m for (_, m) in keyed], geom)
            @test isequal(grp.Vg, grp_l.Vg)
            @test isequal(grp.Wg, grp_l.Wg)
            # The mask really zero-weighted the flagged global channels.
            for (c, gc) in enumerate(grp.g_ci)
                mask[gc] && @test all(iszero, @view grp.Wg[c, :, :, :])
            end
        end
    end

    @testset "eager source data is never mutated" begin
        snap = [copy(parent(leaf[:weights])) for (_, leaf) in UVP.branches(uvset)]
        FP.materialize_leaves(st, st.groups[1]; inner = 2)
        FP.materialize_cube(st, st.groups[1])
        @test all(
            isequal(s, parent(leaf[:weights]))
                for (s, (_, leaf)) in zip(snap, UVP.branches(uvset))
        )
    end

    @testset "CalFunction: view metadata + targeted mutation" begin
        seen = []
        probe = FP.CalFunction() do v
            push!(seen, (; v.source, v.scan, nchan = length(v.freqs), v.ant_names,
                ci = copy(v.chan_idx), ti = copy(v.ti_idx), v.pol_products))
        end
        stp = FP.scan_stream(uvset; geom = geom, transforms = (probe,))
        grp = FP.materialize_cube(stp, stp.groups[1])
        @test length(seen) == 1
        m = seen[1]
        @test m.source == stp.groups[1].source && m.scan == stp.groups[1].scan
        @test m.nchan == length(grp.fg) && m.ci == grp.g_ci && m.ti == grp.g_ti
        @test m.ant_names == stp.ant_names
        @test m.pol_products == grp.pol_products

        # Targeted per-datum correction: halve the weights of baseline (1, 2).
        fix = FP.CalFunction() do v
            for (bi, (a, b)) in enumerate(v.bl_pairs)
                minmax(a, b) == (1, 2) && (v.weights[:, :, bi, :] .*= 0.5)
            end
        end
        stf = FP.scan_stream(uvset; geom = geom, transforms = (fix,))
        grp_f = FP.materialize_cube(stf, stf.groups[1])
        grp_0 = FP.materialize_cube(FP.scan_stream(uvset; geom = geom), stf.groups[1])
        for (bi, (a, b)) in enumerate(grp_0.bl_pairs)
            expect = minmax(a, b) == (1, 2) ? Float32(0.5) .* grp_0.Wg[:, :, bi, :] :
                grp_0.Wg[:, :, bi, :]
            @test isequal(grp_f.Wg[:, :, bi, :], expect)
        end
        @test isequal(grp_f.Vg, grp_0.Vg)

        # The chain hits the leaf path too.
        keyed_f = FP.materialize_leaves(stf, stf.groups[1])
        @test length(keyed_f) == length(stf.groups[1].leaves)
    end

    @testset "view = leaf-shaped DimStack + geometry window" begin
        stv = FP.scan_stream(uvset; geom = geom)
        grp = FP.materialize_cube(stv, stv.groups[1])
        v = FP.scan_view(stv, grp)
        @test v.data isa DimStack
        # The view's stack IS the group's stack (built once at the cube's
        # birth); the layers ALIAS the cubes — mutation flows both ways.
        @test v.data === grp.data
        @test v.vis === parent(v.data[:vis]) === grp.Vg
        @test v.weights === parent(v.data[:weights]) === grp.Wg
        parent(v.data[:weights])[1, 1, 1, 1] = 42
        @test grp.Wg[1, 1, 1, 1] == 42.0f0
        # Leaf-format axes: lookups carry freqs/times/products; UVData helpers
        # and DimensionalData selectors work on the stack directly.
        @test collect(lookup(v.data[:vis], Frequency)) == grp.fg == v.freqs
        @test collect(lookup(v.data[:vis], Ti)) == grp.tg == v.times
        @test pol_products(v.data[:vis]) == grp.pol_products == v.pol_products
        p1 = grp.pol_products[1]
        @test parent(v.data[:vis][Pol = At(p1)]) == grp.Vg[:, :, :, 1]
        # The metadata is the first band leaf's own PartitionInfo — moved, not
        # rebuilt: real BaselineIndex, antenna table, source/scan identity.
        md = DimensionalData.metadata(v.data)
        @test md isa UVP.PartitionInfo
        @test md.baselines isa UVP.BaselineIndex
        @test md.baselines.pairs == grp.bl_pairs == v.bl_pairs
        @test md.baselines[grp.bl_pairs[2]] == 2
        @test md.source_name == grp.source == v.source
        @test md.scan_name == grp.scan == v.scan
        @test String.(md.antennas.name) == stv.ant_names == v.ant_names
    end

    @testset "leaf path: view stack selected off the leaf, flag re-derived" begin
        lk = ReentrantLock()
        captured = []
        probe = FP.CalFunction() do v
            lock(() -> push!(captured, (; stack = v.data, v.source, v.ant_names)), lk)
        end
        stp = FP.scan_stream(uvset; geom = geom, transforms = (probe, FP.FlagChannels(mask)))
        keyed = FP.materialize_leaves(stp, stp.groups[1])
        @test length(captured) == length(keyed)
        # The view's stack is `leaf[(:vis, :weights)]` — the leaf's own
        # PartitionInfo comes along with the selection, nothing is rebuilt.
        for c in captured
            @test DimensionalData.metadata(c.stack) isa UVP.PartitionInfo
            @test c.ant_names == stp.ant_names
            @test c.source == stp.groups[1].source
        end
        # The transformed arrays flow through to the returned leaves unchanged.
        @test Set(objectid(parent(c.stack[:vis])) for c in captured) ==
            Set(objectid(parent(l[:vis])) for (_, l) in keyed)
        # The flag layer is re-derived from the TRANSFORMED weights (FlagChannels
        # zeroed whole channels above), matching the monolith's leaf semantics.
        for (_, l) in keyed
            @test parent(l[:flag]) == (parent(l[:weights]) .<= 0)
            @test any(parent(l[:flag]))
        end
    end

    @testset "validation errors" begin
        # An incompatible ApplySolution is rejected at stream construction.
        other, _ = _build_fringe_uvset(nbands = 3, nchan = 4)
        @test_throws ErrorException FP.scan_stream(other; transforms = [FP.ApplySolution(precal)])
        @test_throws ErrorException FP.StationWeightScale([1.0, -1.0])
        st_len = FP.scan_stream(uvset; geom = geom, transforms = [FP.FlagChannels(falses(3))])
        @test_throws ErrorException FP.materialize_cube(st_len, st_len.groups[1])
    end
end
