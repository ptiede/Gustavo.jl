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
            stack, win = FP.materialize_cube(st, spec; executor = DynamicScheduler(; nchunks = 2))
            keyed = FP.materialize_leaves(st, spec; executor = DynamicScheduler(; nchunks = 2))
            # Rebuild the cube from the transformed leaves: identical data —
            # the two choke paths apply the chain identically.
            stack_l, _ = ST._stacked_scan_group([m for (_, m) in keyed], geom)
            @test isequal(stack[:vis], stack_l[:vis])
            @test isequal(stack[:weights], stack_l[:weights])
            # The mask really zero-weighted the flagged global channels.
            for (c, gc) in enumerate(win.chan_idx)
                mask[gc] && @test all(iszero, @view stack[:weights][c, :, :, :])
            end
        end
    end

    @testset "eager source data is never mutated" begin
        snap = [copy(parent(leaf[:weights])) for (_, leaf) in UVP.branches(uvset)]
        FP.materialize_leaves(st, st.groups[1]; executor = DynamicScheduler(; nchunks = 2))
        FP.materialize_cube(st, st.groups[1])
        @test all(
            isequal(s, parent(leaf[:weights]))
                for (s, (_, leaf)) in zip(snap, UVP.branches(uvset))
        )
    end

    @testset "CalFunction: stack metadata + targeted mutation" begin
        seen = []
        probe = FP.CalFunction() do stack, win
            push!(seen, (;
                source = source_name(stack), scan = scan_name(stack),
                nchan = length(frequencies(stack)),
                ant_names = String.(antennas(stack).name),
                ci = copy(win.chan_idx), ti = copy(win.ti_idx),
                pol_products = pol_products(stack),
            ))
        end
        stp = FP.scan_stream(uvset; geom = geom, transforms = (probe,))
        stack, win = FP.materialize_cube(stp, stp.groups[1])
        @test length(seen) == 1
        m = seen[1]
        @test m.source == stp.groups[1].source && m.scan == stp.groups[1].scan
        @test m.nchan == length(frequencies(stack)) &&
            m.ci == win.chan_idx && m.ti == win.ti_idx
        @test m.ant_names == stp.ant_names
        @test m.pol_products == pol_products(stack)

        # Targeted per-datum correction: halve the weights of baseline (1, 2).
        fix = FP.CalFunction() do stack, win
            for (bi, (a, b)) in enumerate(baselines(stack).pairs)
                minmax(a, b) == (1, 2) && (stack[:weights][Baseline = bi] .*= 0.5)
            end
        end
        stf = FP.scan_stream(uvset; geom = geom, transforms = (fix,))
        stack_f, _ = FP.materialize_cube(stf, stf.groups[1])
        stack_0, _ = FP.materialize_cube(FP.scan_stream(uvset; geom = geom), stf.groups[1])
        W0 = stack_0[:weights]
        for (bi, (a, b)) in enumerate(baselines(stack_0).pairs)
            expect = minmax(a, b) == (1, 2) ? Float32(0.5) .* W0[:, :, bi, :] : W0[:, :, bi, :]
            @test isequal(stack_f[:weights][:, :, bi, :], expect)
        end
        @test isequal(stack_f[:vis], stack_0[:vis])

        # The chain hits the leaf path too.
        keyed_f = FP.materialize_leaves(stf, stf.groups[1])
        @test length(keyed_f) == length(stf.groups[1].leaves)
    end

    @testset "materialized group = leaf-shaped DimStack + geometry window" begin
        stv = FP.scan_stream(uvset; geom = geom)
        stack, win = FP.materialize_cube(stv, stv.groups[1])
        @test stack isa DimStack
        @test win isa CAL.GeometryWindow && win.geom === geom
        # The layers ALIAS the cubes the decode wrote — mutation flows both ways.
        parent(stack[:weights])[1, 1, 1, 1] = 42
        @test stack[:weights][1, 1, 1, 1] == 42.0f0
        # Leaf-format axes: lookups carry freqs/times/products; UVData helpers
        # and DimensionalData selectors work on the stack directly.
        @test collect(lookup(stack[:vis], Frequency)) == frequencies(stack)
        @test collect(lookup(stack[:vis], Ti)) == timestamps(stack)
        @test pol_products(stack[:vis]) == pol_products(stack)
        p1 = pol_products(stack)[1]
        @test parent(stack[:vis][Pol = At(p1)]) == parent(stack[:vis])[:, :, :, 1]
        # The window addresses the same channels in the solve's index space.
        @test geom.channel_freqs[win.chan_idx] ≈ frequencies(stack)
        # The metadata is the first band leaf's own PartitionInfo — moved, not
        # rebuilt: real BaselineIndex, antenna table, source/scan identity.
        md = DimensionalData.metadata(stack)
        @test md isa UVP.PartitionInfo
        @test md.baselines isa UVP.BaselineIndex
        @test md.baselines.pairs == baselines(stack).pairs
        @test md.baselines[baselines(stack).pairs[2]] == 2
        @test md.source_name == source_name(stack)
        @test md.scan_name == scan_name(stack)
        @test String.(md.antennas.name) == stv.ant_names
    end

    @testset "leaf path: stack selected off the leaf, flag re-derived" begin
        lk = ReentrantLock()
        captured = []
        probe = FP.CalFunction() do stack, win
            lock(lk) do
                push!(captured, (;
                    stack, source = source_name(stack),
                    ant_names = String.(antennas(stack).name),
                ))
            end
        end
        stp = FP.scan_stream(uvset; geom = geom, transforms = (probe, FP.FlagChannels(mask)))
        keyed = FP.materialize_leaves(stp, stp.groups[1])
        @test length(captured) == length(keyed)
        # The stack is `leaf[(:vis, :weights)]` — the leaf's own
        # PartitionInfo comes along with the selection, nothing is rebuilt.
        for c in captured
            @test DimensionalData.metadata(c.stack) isa UVP.PartitionInfo
            @test c.ant_names == stp.ant_names
            @test c.source == stp.groups[1].source
        end
        # The transformed arrays flow through to the returned leaves unchanged.
        @test Set(objectid(parent(c.stack[:vis])) for c in captured) ==
            Set(objectid(parent(l[:vis])) for (_, l) in keyed)
        # FlagChannels zeroed whole channels above, so the transformed weights
        # carry the flag (a cell is flagged iff its weight is ≤ 0).
        for (_, l) in keyed
            @test any(parent(l[:weights]) .<= 0)
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
