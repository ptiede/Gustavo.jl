# ── Streaming layer (Gustavo.Streaming) ──────────────────────────────────────
#
# Grouping, materialization (direct-decode fast path ≡ stacked fallback),
# search determinism, selection, scheduling, and the layer's independence from
# the fringe kernels. End-to-end gates live in test_smoother_step.jl.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "Scan streaming layer" begin
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)

    st = FP.scan_stream(uvset; geom = geom)

    @testset "grouping: order, identity, charges" begin
        @test length(st.groups) == length(unique(s.scan for s in st.groups))
        for spec in st.groups
            @test spec.index == findfirst(==(spec), st.groups)
            @test spec.charge == ST._spec_peak_bytes(spec.leaves)
            info = UVP.metadata(last(first(spec.leaves)))
            @test spec.source == String(info.source_name)
            @test spec.scan == String(info.scan_name)
        end
    end

    @testset "direct-decode fast path ≡ stacked fallback" begin
        for spec in st.groups
            leaves = UVP.materialize_group([l for (_, l) in spec.leaves])
            stack_s, win_s = ST._stacked_scan_group(leaves, geom)
            stack_n, win_n = FP.materialize_cube(st, spec)
            @test isequal(stack_n[:vis], stack_s[:vis]) &&
                isequal(stack_n[:weights], stack_s[:weights])
            # The direct-decode fast path, WHEN it fires (lazy sibling-spw IDI
            # spans), must agree with the stacked fallback bit-for-bit.
            direct = ST._direct_scan_group(spec, geom, ST.inner_executor(st))
            if direct !== nothing
                stack_d, win_d = direct
                @test isequal(stack_d[:vis], stack_s[:vis]) &&
                    isequal(stack_d[:weights], stack_s[:weights])
                @test win_d.chan_idx == win_s.chan_idx && win_d.ti_idx == win_s.ti_idx
            end
            @test win_s.chan_idx == win_n.chan_idx && win_s.ti_idx == win_n.ti_idx
            @test frequencies(stack_n) == frequencies(stack_s)
            @test timestamps(stack_n) == timestamps(stack_s)
            @test baselines(stack_n).pairs == baselines(stack_s).pairs
            @test pol_products(stack_n) == pol_products(stack_s)
            @test source_name(stack_n) == spec.source && scan_name(stack_n) == spec.scan
        end
    end

    @testset "search_scan determinism + cell accounting" begin
        search = FP.FringeSearch()
        spec = st.groups[1]
        stack_n, _ = FP.materialize_cube(st, spec)
        res = FP.search_scan(stack_n, st.geom, search; executor = DynamicScheduler(; nchunks = 2), ngroups = length(st.groups))
        @test any(res.valid)      # the synthetic fringes are strong → detections
        # Effective search cells are FRACTIONAL on real grids (the oversampled
        # plane is divided by the oversampling) — a Float64, never an Int
        # (regression: an Int once survived the whole synthetic suite because the
        # fixtures give integral counts, then threw InexactError on a real file).
        @test FP._search_cells(frequencies(stack_n), timestamps(stack_n) .* 3600.0, search) isa Float64

        # inner fan-out is bit-identical to the serial loop, per detection layer.
        res_ser = FP.search_scan(stack_n, st.geom, search; executor = SerialScheduler(), ngroups = length(st.groups))
        @test all(all(res_ser[k] .=== res[k]) for k in keys(res))
    end

    @testset "BySpw / ByKey grouping" begin
        stb = FP.scan_stream(uvset; grouping = FP.BySpw(), geom = geom)
        nleaves = length(collect(UVP.branches(uvset)))
        @test length(stb.groups) == nleaves
        @test all(s -> length(s.leaves) == 1, stb.groups)
        # A single-leaf group still materializes (one spw's cube) with the
        # leaf's own geometry window.
        _, win1 = FP.materialize_cube(stb, stb.groups[1])
        w1 = CAL.leaf_window(geom, last(first(stb.groups[1].leaves)))
        @test win1.chan_idx == w1.chan_idx && win1.ti_idx == w1.ti_idx

        stk = FP.scan_stream(uvset; grouping = FP.ByKey((k, leaf) -> 1), geom = geom)
        @test length(stk.groups) == 1
        @test length(stk.groups[1].leaves) == nleaves
    end

    @testset "map_groups: order, selection, progress" begin
        stb = FP.scan_stream(uvset; grouping = FP.BySpw(), geom = geom)
        n = length(stb.groups)
        # Results in group order regardless of scheduling.
        @test map_groups(spec -> spec.index, stb) == collect(1:n)

        events = Tuple{Symbol, Int, Int}[]
        cb = (stage, done, total) -> push!(events, (stage, done, total))
        map_groups(spec -> nothing, stb; progress = cb, stage = :probe)
        @test events[1] == (:probe, 0, n)
        @test sort(last.(events[2:end])) == fill(n, n) && sort([e[2] for e in events[2:end]]) == collect(1:n)

        sel = Gustavo.ScanIndices(2)
        picked = FP.select_groups(stb, sel)
        @test length(picked) == 1 && picked[1].index == 2
        @test map_groups(spec -> spec.index, stb; selection = sel) == [2]

        # snr-aware selection resolves through select_groups' snr keyword —
        # any ScanWhere predicate can read it generically.
        snrs = fill(NaN, n); snrs[1] = 10.0
        finite = FP.select_groups(stb, Gustavo.ScanWhere(s -> isfinite(s.snr)); snr = snrs)
        @test [s.index for s in finite] == [1]
        every = FP.select_groups(stb, Gustavo.ScanWhere(s -> true); snr = snrs)
        @test [s.index for s in every] == collect(1:n)

        # A failing group rethrows after the pass drains. A concurrent
        # scheduler task-wraps it — the contract is that the ROOT CAUSE
        # surfaces.
        err = try
            map_groups(stb) do spec
                spec.index == 1 ? error("boom") : nothing
            end
            nothing
        catch e
            e
        end
        root = err isa TaskFailedException ? err.task.exception : err
        @test root isa ErrorException && root.msg == "boom"
    end

    @testset "the memory budget gates the outer scheduler" begin
        # The schedulers are used as configured, so a budget that cannot hold
        # the groups they would keep resident is an error at construction —
        # never a silently reduced task count. `BySpw` so the set has more than
        # one group to hold at once.
        @test_throws "memory budget" FP.scan_stream(
            uvset; geom = geom, grouping = FP.BySpw(),
            exec = ExecutionConfig(
                mem_budget = 1.0, outer_executor = GreedyScheduler(; ntasks = 4),
            ),
        )
        # One group at a time is the floor — no task count would make an
        # oversized group fit, so a serial stream builds however tight the
        # budget, and runs.
        st_tight = FP.scan_stream(
            uvset; geom = geom, grouping = FP.BySpw(),
            exec = ExecutionConfig(mem_budget = 1.0),
        )
        @test map_groups(spec -> spec.index, st_tight) == collect(1:length(st_tight.groups))

        @test ST.max_tasks(SerialScheduler()) == 1
        @test ST.max_tasks(GreedyScheduler(; ntasks = 3)) == 3
        @test ST.max_tasks(DynamicScheduler(; nchunks = 5)) == 5
        # A chunksize-configured scheduler's count follows the data, so the
        # bound falls back to the thread count.
        @test ST.max_tasks(DynamicScheduler(; chunksize = 2)) == Threads.nthreads()
    end
end

# The streaming layer drives a pass with nothing from `Gustavo.Fringe` in
# scope: `using Gustavo.Streaming` alone must supply the stream, the grouping,
# the transform contract and the pass runner.
module StreamingWithoutFringe

    using Gustavo.Streaming
    import Gustavo.Streaming: apply_transform!

    struct HalveWeights <: AbstractDataTransform end
    apply_transform!(::HalveWeights, stack, win; executor = SerialScheduler()) =
        (stack[:weights] .*= 0.5; nothing)

    # One pass: materialize every group through the chain and
    # report each group's total weight.
    function pass(uvset, geom)
        stream = scan_stream(uvset; geom = geom, transforms = (HalveWeights(),))
        sums = map_groups(stream) do spec
            sum(first(materialize_cube(stream, spec))[:weights])
        end
        return stream, sums
    end

end

@testset "streaming layer stands alone" begin
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)

    stream, halved = StreamingWithoutFringe.pass(uvset, geom)
    plain = map_groups(
        s -> sum(first(FP.materialize_cube(FP.scan_stream(uvset; geom = geom), s))[:weights]),
        FP.scan_stream(uvset; geom = geom),
    )
    @test length(halved) == length(stream.groups) == length(plain)
    @test all(isapprox(h, 0.5 * p; rtol = 1.0e-6) for (h, p) in zip(halved, plain))

    # `Streaming` never names the fringe kernels — the module is the assertion,
    # since it is loaded before `Fringe` and cannot reach back into it.
    for n in (:FringeWorkspace, :FringeSearch, :search_scan)
        @test !isdefined(Gustavo.Streaming, n)
    end

    # `Fringe`'s own selection extends the streaming generic rather than
    # shadowing it: one function, reachable unambiguously at the top level.
    @test FP.select_scans === ST.select_scans === Gustavo.select_scans
    @test isdefined(Gustavo, :select_scans)
    recs = [(; index = i, source = "S", scan = "s$i", snr = 1.0, stations = Set(1:3)) for i in 1:3]
    @test select_scans(FP.CoverageTopup(AllScans()), recs) == [1, 2, 3]

    # The stream carries no kernel scratch: the search allocates its own per-task
    # FFT workspace, so any stream searches without prior setup.
    @test !hasproperty(stream, :pool)
    stack, _ = FP.materialize_cube(stream, stream.groups[1])
    res = FP.search_scan(stack, stream.geom, FP.FringeSearch())
    @test res isa DimStack
    @test keys(res) == (:delay, :rate, :phase, :amp, :snr, :pfa, :valid)
end

# ── Sub-array leaves: one station axis across differing antenna tables ────────
#
# A leaf's `(a, b)` baseline pairs index that leaf's OWN antenna table, so a
# leaf that saw a sub-array numbers stations differently from a full one. Every
# solver has a single station axis, so the set must be put on one table first —
# `unify_antennas` — or one station's data is attributed to another.

@testset "unify_antennas: sub-array leaves share one station axis" begin
    uvset, _ = _build_fringe_uvset(nant = 4, nspw = 1, nchan = 4, nscans = 2, ntime = 4)
    names = String.(UVP.union_antennas(uvset).name)

    # Rebuild the SECOND scan as a sub-array that dropped the second antenna:
    # its own table lists 3 stations, so its index 2 is what the full table
    # calls index 3.
    keep = [names[1], names[3], names[4]]
    sub = UVP.apply(uvset) do leaf, info, root
        info.scan_name == "2" || return leaf
        full = String.(info.antennas.name)
        local_of = Dict(n => i for (i, n) in pairs(keep))
        rows = [r for r in getfield(info.antennas, :antennas) if String(r.name) in keep]
        tbl = UVP.AntennaTable(
            StructArray(rows), UVP.array_xyz(info.antennas),
            UVP.array_name(info.antennas), UVP.extras(info.antennas),
        )
        remap(ps) = [
            (local_of[full[a]], local_of[full[b]]) for (a, b) in ps
                if full[a] in keep && full[b] in keep
        ]
        b = info.baselines
        idx = [i for (i, (a, c)) in enumerate(b.pairs) if full[a] in keep && full[c] in keep]
        newb = UVP.BaselineIndex(remap(b.pairs_per_record), remap(b.pairs); antenna_names = keep)
        v = leaf[:vis][Baseline = idx]
        w = leaf[:weights][Baseline = idx]
        u = leaf[:uvw][Baseline = idx]
        return UVP._build_leaf(
            v, w, u;
            partition_info = UVP.update(
                info; antennas = tbl, baselines = newb, record_order = Tuple{Int, Int}[],
            ),
        )
    end

    @testset "leaves really do disagree before unification" begin
        tables = unique([String.(UVP.metadata(l).antennas.name) for l in values(UVP.leaves(sub))])
        @test length(tables) == 2
        @test keep in tables && names in tables
    end

    @testset "unification puts every leaf on the union table" begin
        u = UVP.unify_antennas(sub)
        @test all(String.(UVP.metadata(l).antennas.name) == names for l in values(UVP.leaves(u)))
        # The sub-array leaf's pairs now name the same stations they did locally.
        for (k, l) in UVP.branches(u)
            info = UVP.metadata(l)
            UVP.metadata(UVP.branches(sub)[k]).scan_name == "2" || continue
            orig = UVP.metadata(UVP.branches(sub)[k]).baselines
            for (p_new, p_old) in zip(info.baselines.pairs, orig.pairs)
                @test names[p_new[1]] == keep[p_old[1]]
                @test names[p_new[2]] == keep[p_old[2]]
            end
        end
        # A set whose leaves already agree is handed back untouched.
        @test UVP.unify_antennas(uvset) === uvset
    end

    @testset "the solve spans both, and stations keep their identity" begin
        # Solving the mixed set must reach every station of the union table and
        # name them as the full-array set does.
        sol = fit(Bandpass(), sub)
        @test sol.info.ant_names == names
        # The sub-array scan contributes: its stations are solved, not skipped.
        g = CAL.gains(sol[:bandpass]; Ti = 1)
        @test size(g, 2) == length(names)
    end
end
