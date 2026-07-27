# ── Streaming layer (Gustavo.Streaming) ──────────────────────────────────────
#
# Grouping, materialization (direct-decode fast path ≡ stacked fallback),
# search determinism, selection, scheduling, and the layer's independence from
# the fringe kernels. End-to-end gates live in test_smoother_step.jl.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "Scan streaming layer" begin
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)

    st = FP.scan_stream(uvset; geom = geom, workspace = FP.FringeWorkspace)

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
            leaves = UVP.materialize_group(
                [l for (_, l) in spec.leaves]; layers = (:vis, :weights, :uvw),
            )
            grp_s = ST._stacked_scan_group(leaves, geom)
            grp_n = FP.materialize_cube(st, spec)
            @test isequal(grp_n.Vg, grp_s.Vg) && isequal(grp_n.Wg, grp_s.Wg)
            # The direct-decode fast path, WHEN it fires (lazy sibling-band IDI
            # spans), must agree with the stacked fallback bit-for-bit.
            grp_d = ST._direct_scan_group(spec, geom)
            if grp_d !== nothing
                @test isequal(grp_d.Vg, grp_s.Vg) && isequal(grp_d.Wg, grp_s.Wg)
                @test grp_d.g_ci == grp_s.g_ci && grp_d.g_ti == grp_s.g_ti
            end
            @test grp_s.g_ci == grp_n.g_ci && grp_s.g_ti == grp_n.g_ti
            @test grp_n.fg == grp_s.fg && grp_n.tg == grp_s.tg
            @test grp_n.bl_pairs == grp_s.bl_pairs
            @test grp_n.pol_products == grp_s.pol_products
            @test grp_n.source == spec.source && grp_n.scan == spec.scan
        end
    end

    @testset "search_scan determinism + cell accounting" begin
        search = FP.FringeSearch()
        spec = st.groups[1]
        grp_n = FP.materialize_cube(st, spec)
        res = FP.search_scan(st, grp_n, search; inner = 2)
        ncross = count(pr -> pr[1] != pr[2], grp_n.bl_pairs)
        @test res.ncells == res.cells1 * max(ncross * length(grp_n.pol_products), 1)
        # Effective cells are FRACTIONAL on real grids (the oversampled plane is
        # divided by the oversampling) — the fields must be Float64, not Int
        # (regression: an Int field survived the whole synthetic suite because
        # these fixtures happen to give integral counts, then threw
        # InexactError on the first real file).
        @test res.cells1 isa Float64 && res.ncells isa Float64
        @test !isempty(res.rows)      # the synthetic fringes are strong

        # `ngroups = 1` (standalone per-scan gating) only tightens/loosens the
        # per-search PFA threshold; the search grid — and cells1 — are unchanged.
        res1 = FP.search_scan(st, grp_n, search; inner = 2, ngroups = 1)
        @test res1.cells1 == res.cells1

        # inner fan-out is bit-identical to the serial loop.
        res_ser = FP.search_scan(st, grp_n, search; inner = 1)
        @test all(res_ser.det .=== res.det)
        @test res_ser.rows == res.rows
    end

    @testset "ByBand / ByKey grouping" begin
        stb = FP.scan_stream(uvset; grouping = FP.ByBand(), geom = geom)
        nleaves = length(collect(UVP.branches(uvset)))
        @test length(stb.groups) == nleaves
        @test all(s -> length(s.leaves) == 1, stb.groups)
        # A single-leaf group still materializes (one band's cube) with the
        # leaf's own geometry window.
        grp1 = FP.materialize_cube(stb, stb.groups[1])
        ci1, ti1 = CAL.leaf_window(geom, last(first(stb.groups[1].leaves)))
        @test grp1.g_ci == collect(ci1) && grp1.g_ti == collect(ti1)

        stk = FP.scan_stream(uvset; grouping = FP.ByKey((k, leaf) -> 1), geom = geom)
        @test length(stk.groups) == 1
        @test length(stk.groups[1].leaves) == nleaves
    end

    @testset "map_groups: order, selection, progress" begin
        stb = FP.scan_stream(uvset; grouping = FP.ByBand(), geom = geom)
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

        # snr-aware selection resolves through select_groups' snr keyword. The
        # synthetic set has ONE source, so uncapped BrightestCalibrator takes
        # every group; max_scans = 1 keeps only the highest-SNR one.
        snrs = fill(NaN, n); snrs[1] = 10.0
        bright = FP.select_groups(stb, Gustavo.BrightestCalibrator(); snr = snrs)
        @test [s.index for s in bright] == collect(1:n)
        top = FP.select_groups(stb, Gustavo.BrightestCalibrator(max_scans = 1); snr = snrs)
        @test [s.index for s in top] == [1]

        # A failing group rethrows after the pass drains. The wrapper differs
        # by executor (Threads task-wraps, the Dagger fetch unwraps to the
        # original) — the contract is that the ROOT CAUSE surfaces.
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

    @testset "deterministic sizing" begin
        @test st.ntasks ≥ 1
        @test st.inner == max(1, Threads.nthreads() ÷ st.ntasks)
        # Tiny synthetic charges never cap ntasks below the request.
        @test st.ntasks == max(1, min(Threads.nthreads(), length(st.groups)))
        # An explicit budget smaller than one group's charge still admits it (clamped).
        st_small = FP.scan_stream(uvset; geom = geom, mem_budget = 1.0)
        @test st_small.ntasks == 1
        @test map_groups(spec -> spec.index, st_small) == collect(1:length(st_small.groups))
    end
end

# The streaming layer drives a pass with nothing from `Gustavo.Fringe` in
# scope: `using Gustavo.Streaming` alone must supply the stream, the grouping,
# the transform contract and the pass runner.
module StreamingWithoutFringe

using Gustavo.Streaming
import Gustavo.Streaming: apply_transform!

struct HalveWeights <: AbstractDataTransform end
apply_transform!(::HalveWeights, v::ScanDataView; inner::Integer = 1) =
    (v.weights .*= 0.5; nothing)

# One budget-admitted pass: materialize every group through the chain and
# report each group's total weight.
function pass(uvset, geom)
    stream = scan_stream(uvset; geom = geom, transforms = (HalveWeights(),))
    sums = map_groups(stream) do spec
        sum(materialize_cube(stream, spec).Wg)
    end
    return stream, sums
end

end

@testset "streaming layer stands alone" begin
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)

    stream, halved = StreamingWithoutFringe.pass(uvset, geom)
    plain = map_groups(s -> sum(FP.materialize_cube(FP.scan_stream(uvset; geom = geom), s).Wg),
                       FP.scan_stream(uvset; geom = geom))
    @test length(halved) == length(stream.groups) == length(plain)
    @test all(isapprox(h, 0.5 * p; rtol = 1.0e-6) for (h, p) in zip(halved, plain))

    # `Streaming` never names the fringe kernels — the module is the assertion,
    # since it is loaded before `Fringe` and cannot reach back into it.
    for n in (:FringeWorkspace, :FringeSearch, :search_scan, :ScanSearchResult)
        @test !isdefined(Gustavo.Streaming, n)
    end

    # `Fringe`'s own selection extends the streaming generic rather than
    # shadowing it: one function, reachable unambiguously at the top level.
    @test FP.select_scans === ST.select_scans === Gustavo.select_scans
    @test isdefined(Gustavo, :select_scans)
    recs = [(; index = i, source = "S", scan = "s$i", snr = 1.0, stations = Set(1:3)) for i in 1:3]
    @test select_scans(FP.CoverageTopup(AllScans()), recs) == [1, 2, 3]

    # The scratch pool is the one seam, and it is empty by default: a stream
    # built without a `workspace` factory carries no fringe workspaces, and
    # `search_scan` says so rather than blocking on an empty channel.
    @test eltype(stream.pool) === Nothing
    @test eltype(FP.scan_stream(uvset; geom = geom, workspace = FP.FringeWorkspace).pool) ===
        FP.FringeWorkspace
    grp = FP.materialize_cube(stream, stream.groups[1])
    @test_throws ArgumentError FP.search_scan(stream, grp, FP.FringeSearch())
    @test_throws "workspace = FringeWorkspace" FP.search_scan(stream, grp, FP.FringeSearch())
end
