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
            stack_s, win_s = ST._stacked_scan_group(leaves, geom)
            stack_n, win_n = FP.materialize_cube(st, spec)
            @test isequal(stack_n[:vis], stack_s[:vis]) &&
                isequal(stack_n[:weights], stack_s[:weights])
            # The direct-decode fast path, WHEN it fires (lazy sibling-band IDI
            # spans), must agree with the stacked fallback bit-for-bit.
            direct = ST._direct_scan_group(spec, geom)
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
        res = FP.search_scan(st, stack_n, search; inner = 2)
        ncross = count(pr -> pr[1] != pr[2], baselines(stack_n).pairs)
        @test res.ncells == res.cells1 * max(ncross * length(pol_products(stack_n)), 1)
        # Effective cells are FRACTIONAL on real grids (the oversampled plane is
        # divided by the oversampling) — the fields must be Float64, not Int
        # (regression: an Int field survived the whole synthetic suite because
        # these fixtures happen to give integral counts, then threw
        # InexactError on the first real file).
        @test res.cells1 isa Float64 && res.ncells isa Float64
        @test !isempty(res.rows)      # the synthetic fringes are strong

        # `ngroups = 1` (standalone per-scan gating) only tightens/loosens the
        # per-search PFA threshold; the search grid — and cells1 — are unchanged.
        res1 = FP.search_scan(st, stack_n, search; inner = 2, ngroups = 1)
        @test res1.cells1 == res.cells1

        # inner fan-out is bit-identical to the serial loop.
        res_ser = FP.search_scan(st, stack_n, search; inner = 1)
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
        _, win1 = FP.materialize_cube(stb, stb.groups[1])
        w1 = CAL.leaf_window(geom, last(first(stb.groups[1].leaves)))
        @test win1.chan_idx == w1.chan_idx && win1.ti_idx == w1.ti_idx

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
apply_transform!(::HalveWeights, stack, win; inner::Integer = 1) =
    (stack[:weights] .*= 0.5; nothing)

# One budget-admitted pass: materialize every group through the chain and
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
    stack, _ = FP.materialize_cube(stream, stream.groups[1])
    @test_throws ArgumentError FP.search_scan(stream, stack, FP.FringeSearch())
    @test_throws "workspace = FringeWorkspace" FP.search_scan(stream, stack, FP.FringeSearch())
end
