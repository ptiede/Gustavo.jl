# Composable-pipeline interface tests (protocol, transforms, selections, stage
# provenance/snapshots, and the fit/calibrate verbs). Every
# pipeline runs on the new engine. Reuses `_build_fringe_uvset` and the
# CAL/FP/UVP aliases from test_pipeline.jl (included earlier in runtests.jl).

# A throwaway solve step proving the protocol defaults exist.
struct _ProtoProbe <: Gustavo.SolveStep end

# A THIRD-PARTY-shaped solve step: declares provides like a built-in one, but
# `_parse_pipeline` knows nothing about its type — proving it routes by
# abstract type alone, not an isa-chain in disguise.
struct _ThirdPartyStep <: Gustavo.SolveStep end
Gustavo.provides(::_ThirdPartyStep) = :thirdparty

# A scan-local step that accumulates from a SUBSET of the scans — the one
# property that keeps a `:scan` step out of a shared pass.
struct _SubsetScanStep <: Gustavo.SolveStep end
Gustavo.fusable_grouping(::_SubsetScanStep) = :scan
Gustavo.fit_selection(::_SubsetScanStep, _) = ScanIndices(1)

# A step whose declaration contradicts itself: scan-local, yet asking for the
# pass to be run again.
struct _RepeatingScanStep <: Gustavo.SolveStep end
Gustavo.fusable_grouping(::_RepeatingScanStep) = :scan
Gustavo.finish_pass!(::_RepeatingScanStep, ctx) = (; repeat_pass = true)

# A transform with no apply_transform! implementation (error-path probe).
struct _NoImpl <: Gustavo.Fring.AbstractDataTransform end

# A model value that is not a `GainComponent`.
struct _OpaqueTerm end

# An estimator that declares nothing — `scan_local_solve`'s safe default.
struct _OpaqueEstimator <: Gustavo.Fring.AbstractFringeEstimator end

# The full three-stage production pipeline at defaults.
_full_chain() = BaselineFringeFit() |> Bandpass() |> AdhocPhase()

@testset "Composable pipeline interface" begin
    @testset "step protocol defaults + visitor hooks" begin
        s = _ProtoProbe()
        @test Gustavo.model_components(s, nothing) == GainModel()
        @test Gustavo.fit_selection(s, Gustavo.StepSolution[]) isa AllScans
        @test Gustavo.provides(s) == :nothing
        @test Gustavo.required_grouping(s) == :any
        # A step that has not declared itself scan-local is never fused.
        @test Gustavo.fusable_grouping(s) == :global
        # The executor-driven visitor contract has working defaults.
        @test Gustavo.start_pass!(s, nothing) === nothing
        @test Gustavo.process_scan!(s, nothing, nothing, nothing) === nothing
        @test Gustavo.finish_pass!(s, nothing) == NamedTuple()
    end

    @testset "built-in step declarations" begin
        @test Gustavo.provides(BaselineFringeFit()) == :fringe
        @test Gustavo.provides(Bandpass()) == :bandpass
        @test Gustavo.provides(AdhocPhase()) == :adhoc
        @test Gustavo.required_grouping(BaselineFringeFit()) == :scan_complete
        # Steps that finalize a scan inside `process_scan!` declare themselves
        # scan-local; a pass that closes one system over every scan does not.
        @test Gustavo.fusable_grouping(DispersionSBDFit()) == :scan
        @test Gustavo.fusable_grouping(AdhocPhase()) == :scan
        @test Gustavo.fusable_grouping(Bandpass()) == :global
        # BaselineFringeFit answers per instance (`Fring.scan_local_solve`): the
        # default — one round, every term per-scan — solves each scan's
        # station systems as the scan is searched, so the pass is scan-local.
        @test Gustavo.fusable_grouping(BaselineFringeFit()) == :scan
        # Any cross-scan coupling forces the pooled pass: a residual re-search
        # round, a track-global inter-feed column (in the base model or in one
        # station's entry), or an estimator that declares nothing (the seam's
        # safe default).
        @test Gustavo.fusable_grouping(BaselineFringeFit(estimator = MatchedFilter(rounds = 2))) == :global
        @test Gustavo.fusable_grouping(
            BaselineFringeFit(model = default_fringe_terms(rel_time = CAL.GlobalTime())),
        ) == :global
        @test Gustavo.fusable_grouping(
            BaselineFringeFit(
                model = with_station(
                    default_fringe_terms(), "AA";
                    phase = default_fringe_terms(rel_time = CAL.GlobalTime()).phase,
                ),
            ),
        ) == :global
        @test_throws "must be a `GainComponent`" merge(default_fringe_terms(); phase = (; x = _OpaqueTerm()))
        # A step's model is a GainModel, never a bare NamedTuple.
        @test_throws MethodError BaselineFringeFit(model = (; phase = default_fringe_terms().phase))
        @test Gustavo.fusable_grouping(BaselineFringeFit(estimator = _OpaqueEstimator())) == :global
        # Neither Bandpass nor BaselineFringeFit overrides fit_selection — both passes
        # stream every scan.
        @test Gustavo.fit_selection(Bandpass(), Gustavo.StepSolution[]) isa AllScans
        @test Gustavo.fit_selection(BaselineFringeFit(), Gustavo.StepSolution[]) isa AllScans
    end

    @testset "pass partitioning: which steps share one read" begin
        prior = Gustavo.StepSolution[]
        steps = Gustavo.SolveStep[
            BaselineFringeFit(), DispersionSBDFit(), AdhocPhase(), Bandpass(),
        ]
        @test Gustavo._fusable_run(steps, 1, prior) == 1:3   # the scan-local run shares a pass
        @test Gustavo._fusable_run(steps, 4, prior) == 4:4
        # A pooled BaselineFringeFit instance (rounds > 1) runs alone, and the
        # scan-local pair after it still shares its own pass.
        steps2 = Gustavo.SolveStep[
            BaselineFringeFit(estimator = MatchedFilter(rounds = 2)),
            DispersionSBDFit(), AdhocPhase(), Bandpass(),
        ]
        @test Gustavo._fusable_run(steps2, 1, prior) == 1:1
        @test Gustavo._fusable_run(steps2, 2, prior) == 2:3
        @test Gustavo._fusable_run(steps2, 4, prior) == 4:4
        # A scan-local step accumulating from a SUBSET of the scans still runs
        # alone: one pass materializes one set of groups.
        @test Gustavo._fusable_run(
            Gustavo.SolveStep[DispersionSBDFit(), _SubsetScanStep()], 1, prior
        ) == 1:1
        @test Gustavo._fusable_run(
            Gustavo.SolveStep[_SubsetScanStep(), DispersionSBDFit()], 1, prior
        ) == 1:1
    end

    @testset "a fused step may not ask for its pass again" begin
        uvset, _ = _build_fringe_uvset()
        @test_throws "is not scan-local" fit(
            AdhocPhase() |> _RepeatingScanStep(), uvset,
            gauge = PinAntenna(1),
        )
    end

    @testset "a fit without a gauge throws, naming the stations" begin
        uvset, _ = _build_fringe_uvset()
        @test_throws "no gauge given" fit(BaselineFringeFit(), uvset)
        @test_throws join(Gustavo._antenna_names(uvset), ", ") fit(BaselineFringeFit(), uvset)
    end

    @testset "steps compose in any declared order" begin
        # A third-party SolveStep composes purely by declaring provides — no
        # isa-case anywhere in _parse_pipeline for it, and no construction-time
        # veto on where it sits.
        @test Gustavo._check_unique_provides(
            Gustavo.SolveStep[BaselineFringeFit(), _ThirdPartyStep()]
        ) === nothing
        @test Gustavo._check_unique_provides(
            Gustavo.SolveStep[_ThirdPartyStep(), BaselineFringeFit()]
        ) === nothing
        # Two steps providing the same capability are rejected, regardless of
        # their concrete types or position — a naming conflict, not an
        # ordering rule.
        @test_throws "more than one step provides :fringe" Gustavo._check_unique_provides(
            Gustavo.SolveStep[BaselineFringeFit(), BaselineFringeFit()]
        )
        # provides(step) === :nothing never collides with itself.
        @test Gustavo._check_unique_provides(
            Gustavo.SolveStep[_ProtoProbe(), _ProtoProbe()]
        ) === nothing
        # _parse_pipeline routes any SolveStep (built-in or third-party) into
        # solve_steps by abstract type alone, in declared order — it does not
        # reorder or reject based on that order.
        br = Gustavo._parse_pipeline([BaselineFringeFit(), _ThirdPartyStep()])
        @test br.solve_steps == [BaselineFringeFit(), _ThirdPartyStep()]
        @test br.ff == BaselineFringeFit()
    end

    @testset "chaining builds a vector" begin
        cf = CalFunction((stack, win) -> nothing)
        chain = cf |> BaselineFringeFit() |> Bandpass()
        @test chain isa Tuple
        @test chain[1] === cf && chain[2] isa BaselineFringeFit && chain[3] isa Bandpass
        @test (cf |> (BaselineFringeFit(),))[1] === cf
        # A transform belongs to the steps after it.
        @test Gustavo._parse_pipeline(chain).before == [[cf], []]
        # Only pipeline elements chain; anything else is function application.
        @test_throws MethodError BaselineFringeFit() |> AverageFrequency(nout = 1)
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
            parent(leaf[:flags]) .= false
            return leaf[(:vis, :weights, :flags)], CAL.leaf_window(geom, leaf)
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

        # FlagChannels: sets the flag on the named GLOBAL channels only, and
        # leaves every weight alone.
        stack, win = mkwindow()
        apply_transform!(FlagChannels(BitVector([true, false, false, true])), stack, win)
        F = stack[:flags]
        @test all(F[1, :, :, :]) && all(F[4, :, :, :])
        @test !any(F[2:3, :, :, :])
        @test all(stack[:weights] .== 1)
        @test_throws ErrorException apply_transform!(FlagChannels(trues(3)), mkwindow()...)

        # CalFunction: arbitrary per-(scan, baseline) mutation — the pain point.
        stack, win = mkwindow()
        hook = CalFunction() do st, w
            scan_name(st) == scanname || return
            for (bi, (a, b)) in enumerate(baselines(st).pairs)
                minmax(a, b) == (1, 3) && (st[:weights][BaselineID = bi] .*= 0.5)
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
        sol = fit(_full_chain(), uvset; gauge = PinAntenna(1))

        @test sol isa CAL.CalibrationSolution
        @test keys(sol) == [:fringe, :bandpass, :adhoc]
        @test_throws ArgumentError sol[:bogus]
        @test_throws "recorded stages: [:fringe, :bandpass, :adhoc]" sol[:bogus]

        # Component θ ranges: contiguous, disjoint, and tile 1:nθ — a per-step
        # property now (each step owns its own layout, not a merged one).
        for step in sol.steps
            rng = [p.range for p in step.layout.plans]
            @test length(rng) == length(step.layout.plans)
            nonempty = [r for r in rng if !isempty(r)]
            isempty(nonempty) && continue
            @test first(first(nonempty)) == 1
            @test last(last(nonempty)) == step.layout.nθ
            for i in 2:length(nonempty)
                @test first(nonempty[i]) == last(nonempty[i - 1]) + 1
            end
        end

        # A `Symbol` selects that step alone; a range selects a run of them.
        @test keys(sol[:adhoc]) == [:adhoc]

        # Selecting every step reproduces the solution.
        @test keys(sol[:]) == keys(sol)
        @test parent(gains(sol[:])) == parent(gains(sol))

        # A leading run carries only the steps up to and including that one —
        # a later step contributes no gain there at all, rather than an
        # explicit zeroed θ block over a shared layout.
        fr = sol[1:1]
        @test keys(fr) == [:fringe]
        @test fr.steps[1].θ == sol.steps[1].θ
        @test fr.steps[1].θ == sol[:fringe].steps[1].θ   # same step, selected either way

        bp = sol[begin:2]
        @test keys(bp) == [:fringe, :bandpass]
        @test bp.steps[1].θ == sol.steps[1].θ
        @test bp.steps[2].θ == sol.steps[2].θ

        # `end` addresses the last step, and a selection keeps the provenance
        # chains, so it stays replayable by `calibrate`.
        @test keys(sol[end]) == [last(keys(sol))]
        @test sol[:].sequence == sol.sequence
        @test sol[:].gauge === sol.gauge
        # A solution has at least one step, so an empty selection is refused.
        @test_throws ArgumentError sol[2:1]
        @test_throws "at least one step" sol[2:1]

        # A snapshot is a valid solution: it applies cleanly.
        @test UVP.apply_calibration(uvset, fr) isa UVP.UVSet
        @test stage_info(sol, :fringe) isa NamedTuple

        # Gains factor multiplicatively over components: the elementwise
        # product of every step's every top-level component selection's gain
        # reproduces the full composed evaluation.
        g_full = parent(gains(sol))
        g_prod = ones(ComplexF64, size(g_full))
        for (si, step) in enumerate(sol.steps), group in (:phase, :logamp)
            for cname in keys(step.layout.plantree[group])
                g_prod .*= parent(gains(sol[si, group, cname]))
            end
        end
        @test g_prod ≈ g_full
    end

    @testset "solution container and selection algebra" begin
        uvset, _ = _build_fringe_uvset()
        sol = fit(_full_chain(), uvset; gauge = PinAntenna(1))

        # Container contract: length/eachindex/keys/haskey, and iteration
        # yields each step as a single-step solution.
        @test length(sol) == 3
        @test eachindex(sol) == 1:3
        @test keys(sol) == [:fringe, :bandpass, :adhoc]
        @test haskey(sol, :bandpass) && !haskey(sol, :bogus)
        @test collect(sol) == [sol[i] for i in eachindex(sol)]
        @test [only(s.steps).name for s in sol] == [:fringe, :bandpass, :adhoc]

        # A duplicated stage name refuses Symbol lookup; positions still work.
        st = sol[:fringe].steps[1]
        dup = CAL.CalibrationSolution([st, st], sol.geom, sol.info)
        @test_throws ArgumentError dup[:fringe]
        @test_throws "recorded 2 times" dup[:fringe]
        @test dup[1].steps[1].name === :fringe

        # A component selection is a solution: it applies like any other, and
        # composing it with its complement reproduces the full step.
        fr = sol[:fringe]
        comps = keys(fr.steps[1].layout.plantree.phase)
        @test !isempty(comps)
        csel = sol[:fringe, :phase, first(comps)]
        @test csel isa CAL.CalibrationSolution
        @test UVP.apply_calibration(uvset, csel) isa UVP.UVSet
        # Chained selection descends into the selected subtree.
        @test parent(gains(sol[:fringe, :phase][:fringe, first(comps)])) ==
            parent(gains(csel))

        # `gains` keywords are DD dimension selectors, windowing the
        # evaluation under the invariant gains(sol; kw...) == gains(sol)[kw...].
        G = gains(sol)
        @test gains(sol; Ti = 1) == G[Ti = 1]
        @test gains(sol; Frequency = 2:3, Feed = 2) == G[Frequency = 2:3, Feed = 2]
        f2 = sol.geom.channel_freqs[2]
        @test gains(sol; Frequency = At(f2), Ant = 2) == G[Frequency = At(f2), Ant = 2]
        @test gains(sol; Ti = Near(sol.geom.times[end])) == G[Ti = Near(sol.geom.times[end])]
        @test_throws ArgumentError gains(sol; Polarization = 1)
        @test_throws "unknown dimension keyword" gains(sol; Polarization = 1)
    end

    @testset "calibrate replays the recorded transforms (weight scale)" begin
        uvset, _ = _build_fringe_uvset()
        ws = [1.0, 0.5, 1.0, 2.0]
        sol = fit(StationWeightScale(ws) |> _full_chain(), uvset; gauge = PinAntenna(1))
        @test length(recorded_transforms(sol)) == 1
        @test recorded_transforms(sol)[1] isa StationWeightScale
        @test recorded_transforms(sol)[1].s == ws

        out = calibrate(sol, uvset; post = AverageFrequency(nout = 1))
        ref = AverageFrequency(nout = 1)(Gustavo.UVData.apply_calibration(uvset, sol))
        @test Set(keys(DimensionalData.branches(out))) == Set(keys(DimensionalData.branches(ref)))
        for (k, leaf) in DimensionalData.branches(ref)
            V = parent(DimensionalData.branches(out)[k][:vis])
            @test size(V) == size(parent(leaf[:vis]))
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(parent(leaf[:vis]), V))
            @test parent(DimensionalData.branches(out)[k][:weights]) ≈ parent(leaf[:weights])
        end

        # Two weight-scale transforms compose (w·(s_a s_b)²), and both are
        # recorded. Two pipelines join by splatting.
        sol_b = fit(((StationWeightScale(ws) |> StationWeightScale(ws))..., _full_chain()...), uvset; gauge = PinAntenna(1))
        @test length(recorded_transforms(sol_b)) == 2
        @test parent(gains(fit(StationWeightScale(ws .* ws) |> _full_chain(), uvset; gauge = PinAntenna(1)))) ≈
            parent(gains(sol_b))

        # A transform listed after a step does not reach that step.
        early = fit(BaselineFringeFit(), uvset; gauge = PinAntenna(1))
        late = fit(BaselineFringeFit() |> StationWeightScale(ws), uvset; gauge = PinAntenna(1))
        @test parent(gains(late)) == parent(gains(early))
    end

    @testset "rate components must share the constant-phase epoch" begin
        # Two scans: the per-scan rate's origins (scan centers) then cannot
        # all coincide with the track-wide rate's single origin.
        uvset, _ = _build_fringe_uvset(nscans = 2)
        # A second rate on a different time segmentation puts its origin in a
        # different place, so no single epoch zeroes both rate coordinates;
        # the fit rejects the model before its fringe pass reads any data.
        bad = merge(
            default_fringe_terms();
            phase = (; rate2 = GainComponent(Rate(); Ti = GlobalTime(), Feed = SingleFeed(2))),
        )
        @test_throws "disagree on the epoch" fit(BaselineFringeFit(model = bad), uvset; gauge = PinAntenna(1))
    end

    @testset "solution serialization round-trip; older files refused" begin
        uvset, _ = _build_fringe_uvset()
        sol = fit(StationWeightScale([1.0, 0.5, 1.0, 1.0]) |> _full_chain(), uvset; gauge = PinAntenna(1))
        path = joinpath(mktempdir(), "sol.jls")
        CAL.save_solution(path, sol)
        back = CAL.load_solution(path)
        @test all(s1.θ == s2.θ for (s1, s2) in zip(back.steps, sol.steps))
        @test keys(back) == keys(sol)
        @test recorded_transforms(back)[1] isa StationWeightScale
        @test back[:fringe].steps[1].θ == sol[:fringe].steps[1].θ

        # The pipeline and gauge are recorded, survive the round-trip, and ride
        # along through selections.
        @test length(back.sequence) == length(sol.sequence) == 4
        @test back.gauge.refs == 1
        @test sol[:fringe].sequence == sol.sequence

        # An element that did not survive an earlier save round-trips as
        # `missing`, and `calibrate` refuses rather than skip it.
        solm = CAL.CalibrationSolution(
            sol.steps, sol.geom, sol.info; sequence = (missing, sol.sequence[2:end]...), sol.gauge,
        )
        pathm = joinpath(mktempdir(), "solm.jls")
        CAL.save_solution(pathm, solm)
        @test ismissing(first(CAL.load_solution(pathm).sequence))
        @test_throws "did not survive" calibrate(solm, uvset)

        # Pre-v6 wrappers used a different solution shape; they are refused
        # rather than misread, so a caller re-solves instead of loading a stale
        # parameter vector.
        v1path = joinpath(mktempdir(), "sol_v1.jls")
        Gustavo.Calibration.serialize(
            v1path, (; version = 1, sol.steps, sol.geom, sol.info)
        )
        @test_throws "unsupported version" CAL.load_solution(v1path)
    end
end
