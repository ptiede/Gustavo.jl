# Composable-pipeline interface tests (protocol, transforms, selections, stage
# provenance/snapshots, and the fit/calibrate/fitcalibrate verbs). Every
# pipeline runs on the new engine. Reuses `_build_fringe_uvset` and the
# CAL/FP/UVP aliases from test_pipeline.jl (included earlier in runtests.jl).

# A throwaway solve step proving the protocol defaults exist.
struct _ProtoProbe <: Gustavo.SolveStep end

# A transform with no apply_transform! implementation (error-path probe).
struct _NoImpl <: Gustavo.Fringe.AbstractDataTransform end

# The full three-stage production pipeline at defaults.
_full_chain() = FringeFit() |> BandpassEstimator() |> TemporalSmoother()

@testset "Composable pipeline interface" begin
    @testset "step protocol defaults + visitor hooks" begin
        s = _ProtoProbe()
        @test Gustavo.model_components(s, nothing) == (; phase = (), logamp = ())
        @test Gustavo.transforms(s) == ()
        @test Gustavo.fit_selection(s) isa AllScans
        @test Gustavo.provides(s) == :nothing
        @test Gustavo.requires(s) == ()
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
        @test Gustavo.requires(BandpassEstimator()) == (:fringe,)
        @test Gustavo.requires(TemporalSmoother()) == (:fringe,)
        @test Gustavo.required_grouping(FringeFit()) == :scan_complete
        # The bandpass pass streams the user's selection wrapped in the station
        # coverage top-up; the fringe pass always streams every scan
        # (the estimator's cross_hand_fit_on masks rows, not scans).
        bsel = Gustavo.fit_selection(BandpassEstimator(select = SourceScans("X")))
        @test bsel isa FP.CoverageTopup && bsel.inner.sources == ["X"]
        @test Gustavo.fit_selection(FringeFit(
            estimator = MatchedFilter(cross_hand_fit_on = ScanIndices(1)))) isa AllScans
        # Solve steps refuse the sequential run_step chain.
        @test_throws ErrorException Gustavo.run_step(FringeFit(), Gustavo.CalibrationContext())
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
        # Brightest calibrator: source B has total 55 > A's 30; cap keeps the
        # highest-SNR scans of the winning source only.
        @test select_scans(BrightestCalibrator(), scans) == [2, 4]
        @test select_scans(BrightestCalibrator(max_scans = 1), scans) == [2]
        # No finite SNRs yet (pre-stage-A) → explicit error, not a silent pick.
        blind = [(index = 1, source = "A", scan = "No1", snr = NaN)]
        @test_throws ErrorException select_scans(BrightestCalibrator(), blind)
    end

    @testset "transforms on a scan stack + geometry window" begin
        # Built the way production builds them — a real (tiny) leaf, the stack
        # selected off it — there is no raw-array assembly path.
        uvsmall, _ = _build_fringe_uvset(
            nant = 3, nbands = 1, nchan = 4, ntime = 3, pol_labels = ["PP", "QQ"],
        )
        geom = CAL.build_geometry(uvsmall)
        leaf = UVP.materialize_leaf(
            last(first(UVP.branches(uvsmall))); layers = (:vis, :weights, :uvw),
        )
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

        # Component θ ranges: contiguous, disjoint, and they tile 1:nθ.
        rng = CAL.component_ranges(sol.layout)
        @test length(rng) == length(sol.layout.plans)
        nonempty = [r for r in rng if !isempty(r)]
        @test first(first(nonempty)) == 1
        @test last(last(nonempty)) == sol.layout.nθ
        for i in 2:length(nonempty)
            @test first(nonempty[i]) == last(nonempty[i - 1]) + 1
        end

        # The last stage's snapshot IS the full solution.
        @test CAL.stage_solution(sol[:adhoc]).θ == sol.θ

        # Earlier snapshots zero exactly the later stages' blocks.
        owned(name) = begin
            r = sol.stages[findfirst(s -> s.name === name, sol.stages)]
            ix = Int[]
            for c in r.phase_comps
                append!(ix, rng[c])
            end
            for c in r.logamp_comps
                append!(ix, rng[sol.layout.nphase + c])
            end
            ix
        end
        fr = CAL.stage_solution(sol[:fringe])
        @test fr.θ[owned(:fringe)] == sol.θ[owned(:fringe)]
        @test all(fr.θ[owned(:bandpass)] .== 0)
        @test all(fr.θ[owned(:adhoc)] .== 0)
        @test stage_names(fr) == [:fringe]

        bp = CAL.stage_solution(sol[:bandpass])
        @test bp.θ[owned(:fringe)] == sol.θ[owned(:fringe)]
        @test bp.θ[owned(:bandpass)] == sol.θ[owned(:bandpass)]
        @test all(bp.θ[owned(:adhoc)] .== 0)

        # A snapshot is a valid solution: it applies cleanly.
        @test UVP.apply_calibration(uvset, fr) isa UVP.UVSet
        @test stage_info(sol[:fringe]) isa NamedTuple

        # Gains factor multiplicatively over components: the elementwise
        # product of per-component gains reproduces the full evaluation.
        ev = CAL.GainEvaluator(sol.model, sol.layout)
        g_full = CAL.evaluate_gains(ev, sol.θ, 1:CAL.nchannels(sol.geom), 1:CAL.ntimes(sol.geom))
        g_prod = ones(ComplexF64, size(g_full))
        for pi in eachindex(sol.layout.plans)
            g_prod .*= CAL.component_gains(sol, pi)
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
        @test sol.θ ≈ sol_f.θ
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
        @test fit(CalibrationPipeline(StationWeightScale(ws .* ws) |> _full_chain()), uvset).θ ≈ sol_b.θ

        # Chain convenience form ≡ the pipeline form.
        @test fit(StationWeightScale(ws) |> _full_chain(), uvset).θ ≈ sol_f.θ
    end

    @testset "solution serialization v2 round-trip (+ v1 compat)" begin
        uvset, _ = _build_fringe_uvset()
        sol = fit(CalibrationPipeline(StationWeightScale([1.0, 0.5, 1.0, 1.0]) |> _full_chain()), uvset)
        path = joinpath(mktempdir(), "sol.jls")
        CAL.save_solution(path, sol)
        back = CAL.load_solution(path)
        @test back.θ == sol.θ
        @test stage_names(back) == stage_names(sol)
        @test back.transforms[1] isa StationWeightScale
        @test CAL.stage_solution(back[:fringe]).θ == CAL.stage_solution(sol[:fringe]).θ

        # v1 wrapper (pre-stage files) loads as a plain solution.
        v1path = joinpath(mktempdir(), "sol_v1.jls")
        Gustavo.Calibration.serialize(
            v1path, (; version = 1, sol.model, sol.layout, sol.geom, sol.θ, sol.info)
        )
        old = CAL.load_solution(v1path)
        @test old.θ == sol.θ
        @test isempty(old.stages) && isempty(old.transforms)
    end
end
