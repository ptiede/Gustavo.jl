# Composable-pipeline interface tests (protocol, transforms, selections, stage
# provenance/snapshots, and the fit/calibrate/fitcalibrate verbs). Every
# pipeline runs on the new engine. Reuses `_build_fringe_uvset` and the
# CAL/FP/UVP aliases from test_pipeline.jl (included earlier in runtests.jl).

# A throwaway solve step proving the protocol defaults exist.
struct _ProtoProbe <: Gustavo.SolveStep end

# A THIRD-PARTY-shaped solve step: declares provides like a built-in one, but
# `_parse_pipeline` knows nothing about its type — proving it routes by
# abstract type alone, not an isa-chain in disguise.
struct _ThirdPartyStep <: Gustavo.SolveStep end
Gustavo.provides(::_ThirdPartyStep) = :thirdparty

# A transform with no apply_transform! implementation (error-path probe).
struct _NoImpl <: Gustavo.Fringe.AbstractDataTransform end

# The full three-stage production pipeline at defaults.
_full_chain() = FringeFit() |> BandpassEstimator() |> TemporalSmoother()

@testset "Composable pipeline interface" begin
    @testset "step protocol defaults + visitor hooks" begin
        s = _ProtoProbe()
        @test Gustavo.model_components(s, nothing) == (; phase = (;), logamp = (;))
        @test Gustavo.transforms(s) == ()
        @test Gustavo.fit_selection(s, Gustavo.StepSolution[]) isa AllScans
        @test Gustavo.provides(s) == :nothing
        @test Gustavo.required_grouping(s) == :any
        # The executor-driven visitor contract has working defaults.
        @test Gustavo.start_pass!(s, nothing) === nothing
        @test Gustavo.process_scan!(s, nothing, nothing, nothing) === nothing
        @test Gustavo.finish_pass!(s, nothing) == NamedTuple()
    end

    @testset "built-in step declarations" begin
        @test Gustavo.provides(FringeFit()) == :fringe
        @test Gustavo.provides(BandpassEstimator()) == :bandpass
        @test Gustavo.provides(TemporalSmoother()) == :adhoc
        @test Gustavo.required_grouping(FringeFit()) == :scan_complete
        # The bandpass pass streams the user's selection wrapped in the station
        # coverage top-up; the fringe pass always streams every scan
        # (the estimator's cross_hand_fit_on masks rows, not scans).
        bsel = Gustavo.fit_selection(BandpassEstimator(select = SourceScans("X")), Gustavo.StepSolution[])
        @test bsel isa FP.CoverageTopup && bsel.inner.sources == ["X"]
        @test Gustavo.fit_selection(
            FringeFit(estimator = MatchedFilter(cross_hand_fit_on = ScanIndices(1))), Gustavo.StepSolution[],
        ) isa AllScans
        # Solve steps refuse the sequential run_step chain.
        @test_throws ErrorException Gustavo.run_step(FringeFit(), Gustavo.CalibrationContext())
    end

    @testset "steps compose in any declared order" begin
        # A third-party SolveStep composes purely by declaring provides — no
        # isa-case anywhere in _parse_pipeline for it, and no construction-time
        # veto on where it sits.
        @test Gustavo._check_unique_provides(
            Gustavo.SolveStep[FringeFit(), _ThirdPartyStep()]
        ) === nothing
        @test Gustavo._check_unique_provides(
            Gustavo.SolveStep[_ThirdPartyStep(), FringeFit()]
        ) === nothing
        # Two steps providing the same capability are rejected, regardless of
        # their concrete types or position — a naming conflict, not an
        # ordering rule.
        @test_throws "more than one step provides :fringe" Gustavo._check_unique_provides(
            Gustavo.SolveStep[FringeFit(), FringeFit()]
        )
        # provides(step) === :nothing never collides with itself.
        @test Gustavo._check_unique_provides(
            Gustavo.SolveStep[_ProtoProbe(), _ProtoProbe()]
        ) === nothing
        # _parse_pipeline routes any SolveStep (built-in or third-party) into
        # solve_steps by abstract type alone, in declared order — it does not
        # reorder or reject based on that order.
        br = Gustavo._parse_pipeline(
            CalibrationPipeline(FringeFit(), _ThirdPartyStep())
        )
        @test br.solve_steps == [FringeFit(), _ThirdPartyStep()]
        @test br.ff == FringeFit()
    end

    @testset "chaining and lifting" begin
        cf = CalFunction((stack, win) -> nothing)
        chain = cf |> FringeFit() |> AverageFrequency(nout = 1)
        @test chain isa StepChain
        @test length(chain.steps) == 3
        @test chain.steps[1] isa DataTransformStep
        @test Gustavo.transforms(chain.steps[1]) == (cf,)
        @test chain.steps[2] isa FringeFit

        p = CalibrationPipeline(chain; exec = ExecutionConfig(mem_fraction = 0.4))
        @test p.steps == chain.steps
        @test p.exec.mem_fraction == 0.4

        # Vector and vararg constructors lift raw transforms too.
        p2 = CalibrationPipeline([cf, FringeFit()])
        @test p2.steps[1] isa DataTransformStep
        p3 = CalibrationPipeline(cf, FringeFit())
        @test p3.steps[1] isa DataTransformStep && p3.exec == ExecutionConfig()
    end

    @testset "scan selections" begin
        scans = [
            (index = 1, source = "A", scan = "No1", snr = 10.0),
            (index = 2, source = "B", scan = "No2", snr = 50.0),
            (index = 3, source = "A", scan = "No3", snr = 20.0),
            (index = 4, source = "B", scan = "No4", snr = 5.0),
        ]
        @test select_scans(AllScans(), scans) == [1, 2, 3, 4]
        @test select_scans(SourceScans("A"), scans) == [1, 3]
        @test select_scans(SourceScans("A", "B"), scans) == [1, 2, 3, 4]
        @test select_scans(ScanIndices(3, 1), scans) == [1, 3]
        @test select_scans(ScanWhere(s -> s.snr > 15), scans) == [2, 3]
    end

    @testset "transforms on a scan stack + geometry window" begin
        # Built the way production builds them — a real (tiny) leaf, the stack
        # selected off it — there is no raw-array assembly path.
        uvsmall, _ = _build_fringe_uvset(
            nant = 3, nspw = 1, nchan = 4, ntime = 3, pol_labels = ["PP", "QQ"],
        )
        geom = CAL.build_geometry(uvsmall)
        leaf = UVP.materialize_leaf(last(first(UVP.branches(uvsmall))))
        scanname = UVP.metadata(leaf).scan_name
        function mkwindow()
            parent(leaf[:vis]) .= 1
            parent(leaf[:weights]) .= 1
            return leaf[(:vis, :weights)], CAL.leaf_window(geom, leaf)
        end

        # StationWeightScale: w → w·s_a·s_b per baseline; vis untouched.
        stack, win = mkwindow()
        apply_transform!(StationWeightScale([2.0, 3.0, 5.0]), stack, win)
        W = stack[:weights]
        @test all(W[:, :, 1, :] .== 6)
        @test all(W[:, :, 2, :] .== 10)
        @test all(W[:, :, 3, :] .== 15)
        @test all(stack[:vis] .== 1)
        @test_throws ErrorException StationWeightScale([1.0, -1.0, 1.0])
        @test_throws ErrorException apply_transform!(StationWeightScale([1.0]), mkwindow()...)

        # FlagChannels: zero-weights flagged GLOBAL channels only.
        stack, win = mkwindow()
        apply_transform!(FlagChannels(BitVector([true, false, false, true])), stack, win)
        W = stack[:weights]
        @test all(W[1, :, :, :] .== 0) && all(W[4, :, :, :] .== 0)
        @test all(W[2:3, :, :, :] .== 1)
        @test_throws ErrorException apply_transform!(FlagChannels(trues(3)), mkwindow()...)

        # CalFunction: arbitrary per-(scan, baseline) mutation — the pain point.
        stack, win = mkwindow()
        hook = CalFunction() do st, w
            scan_name(st) == scanname || return
            for (bi, (a, b)) in enumerate(baselines(st).pairs)
                minmax(a, b) == (1, 3) && (st[:weights][Baseline = bi] .*= 0.5)
            end
        end
        apply_transform!(hook, stack, win)
        W = stack[:weights]
        @test all(W[:, :, 2, :] .== 0.5)
        @test all(W[:, :, 1, :] .== 1) && all(W[:, :, 3, :] .== 1)

        # Chains run in order; `nothing` chain is a no-op.
        stack, win = mkwindow()
        ST.apply_transforms!([StationWeightScale([2.0, 1.0, 1.0]), hook], stack, win)
        @test all(stack[:weights][:, :, 2, :] .== 1.0)   # (1,3): 2·1 then ×0.5
        ST.apply_transforms!(nothing, stack, win)

        # A transform without an implementation errors loudly.
        @test_throws ErrorException apply_transform!(_NoImpl(), mkwindow()...)
        @test_throws ErrorException FP.apply_transform(first(_build_fringe_uvset()), _NoImpl())
    end

    @testset "full pipeline: stage provenance and snapshots" begin
        uvset, _ = _build_fringe_uvset()
        sol = fit(CalibrationPipeline(_full_chain()), uvset)

        @test sol isa CAL.CalibrationSolution
        @test stage_names(sol) == [:fringe, :bandpass, :adhoc]
        @test_throws ArgumentError sol[:bogus]
        @test_throws "recorded stages: [:fringe, :bandpass, :adhoc]" sol[:bogus]

        # Component θ ranges: contiguous, disjoint, and tile 1:nθ — a per-step
        # property now (each step owns its own layout, not a merged one).
        for step in sol.steps
            rng = CAL.component_ranges(step.layout)
            @test length(rng) == length(step.layout.plans)
            nonempty = [r for r in rng if !isempty(r)]
            isempty(nonempty) && continue
            @test first(first(nonempty)) == 1
            @test last(last(nonempty)) == step.layout.nθ
            for i in 2:length(nonempty)
                @test first(nonempty[i]) == last(nonempty[i - 1]) + 1
            end
        end

        # `sol[name]` extracts that step alone (== `step_solution`); the
        # cumulative view through a stage is `stage_solution`, a separate call.
        @test sol[:adhoc].steps[1].θ == CAL._step(sol, :adhoc).θ
        @test stage_names(sol[:adhoc]) == [:adhoc]

        # The last stage's cumulative snapshot IS the full solution.
        @test stage_names(stage_solution(sol, :adhoc)) == stage_names(sol)
        @test parent(gains(stage_solution(sol, :adhoc))) == parent(gains(sol))

        # Earlier snapshots carry only the steps up to and including that stage
        # — a later stage contributes no gain there at all, rather than an
        # explicit zeroed θ block over a shared layout.
        fr = stage_solution(sol, :fringe)
        @test stage_names(fr) == [:fringe]
        @test fr.steps[1].θ == sol.steps[1].θ
        @test fr.steps[1].θ == sol[:fringe].steps[1].θ   # same step, extracted either way

        bp = stage_solution(sol, :bandpass)
        @test stage_names(bp) == [:fringe, :bandpass]
        @test bp.steps[1].θ == sol.steps[1].θ
        @test bp.steps[2].θ == sol.steps[2].θ

        # A snapshot is a valid solution: it applies cleanly.
        @test UVP.apply_calibration(uvset, fr) isa UVP.UVSet
        @test stage_info(sol, :fringe) isa NamedTuple

        # Gains factor multiplicatively over components: the elementwise
        # product of every step's every component's gain reproduces the full
        # composed evaluation.
        g_full = parent(gains(sol))
        g_prod = ones(ComplexF64, size(g_full))
        for step in sol.steps, pi in eachindex(step.layout.plans)
            g_prod .*= CAL.component_gains(sol, step.name, pi)
        end
        @test g_prod ≈ g_full
    end

    @testset "fit + calibrate ≡ fitcalibrate (weight-scale transform)" begin
        uvset, _ = _build_fringe_uvset()
        ws = [1.0, 0.5, 1.0, 2.0]
        pipe = CalibrationPipeline(StationWeightScale(ws) |> _full_chain())
        red = [AverageFrequency(nout = 1)]

        sol_f, out_f = fitcalibrate(pipe, uvset; reduce = red)
        sol = fit(pipe, uvset)
        @test parent(gains(sol)) ≈ parent(gains(sol_f))
        # The solve's transform chain is recorded on the solution.
        @test length(sol.transforms) == 1
        @test sol.transforms[1] isa StationWeightScale
        @test sol.transforms[1].s == ws

        out = calibrate(sol, uvset; reduce = red)
        @test Set(keys(DimensionalData.branches(out))) ==
            Set(keys(DimensionalData.branches(out_f)))
        for (k, leaf) in DimensionalData.branches(out_f)
            Vf = parent(leaf[:vis])
            Wf = parent(leaf[:weights])
            V = parent(DimensionalData.branches(out)[k][:vis])
            W = parent(DimensionalData.branches(out)[k][:weights])
            @test size(V) == size(Vf)
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vf, V))
            @test W ≈ Wf
        end

        # Two weight-scale transforms COMPOSE (the chain applies both in
        # order, w·(s_a s_b)²) — the old bridge's "specified twice" error died
        # with it. Both are recorded on the solution.
        both = CalibrationPipeline(
            StationWeightScale(ws) |> StationWeightScale(ws) |> _full_chain())
        sol_b = fit(both, uvset)
        @test length(sol_b.transforms) == 2
        @test parent(gains(fit(CalibrationPipeline(StationWeightScale(ws .* ws) |> _full_chain()), uvset))) ≈
            parent(gains(sol_b))

        # Chain convenience form ≡ the pipeline form.
        @test parent(gains(fit(StationWeightScale(ws) |> _full_chain(), uvset))) ≈ parent(gains(sol_f))
    end

    @testset "solution serialization v5 round-trip; pre-v5 files refused" begin
        uvset, _ = _build_fringe_uvset()
        sol = fit(CalibrationPipeline(StationWeightScale([1.0, 0.5, 1.0, 1.0]) |> _full_chain()), uvset)
        path = joinpath(mktempdir(), "sol.jls")
        CAL.save_solution(path, sol)
        back = CAL.load_solution(path)
        @test all(s1.θ == s2.θ for (s1, s2) in zip(back.steps, sol.steps))
        @test stage_names(back) == stage_names(sol)
        @test back.transforms[1] isa StationWeightScale
        @test back[:fringe].steps[1].θ == sol[:fringe].steps[1].θ

        # Pre-v5 wrappers used a different solution shape; they are refused
        # rather than misread, so a caller re-solves instead of loading a stale
        # parameter vector.
        v1path = joinpath(mktempdir(), "sol_v1.jls")
        Gustavo.Calibration.serialize(
            v1path, (; version = 1, sol.steps, sol.geom, sol.info)
        )
        @test_throws "unsupported version" CAL.load_solution(v1path)
    end
end
