# Modular calibration-pipeline tests: the pipeline-level surface
# (`CalibrationPipeline`, `fitcalibrate`, reduce steps, defaults).
# Reuses `_build_fringe_uvset` and the CAL/FP/UVP aliases from test_pipeline.jl
# (included earlier in runtests.jl).

# A throwaway reduce step proving the ReduceStep extension point: subtype + one
# `prepare_reducer` method (identity transform that records it ran).
struct _ProbeReduce <: Gustavo.ReduceStep
    seen::Base.RefValue{Bool}
end
Gustavo.prepare_reducer(s::_ProbeReduce, ctx::Gustavo.CalibrationContext) =
    ((uv -> (s.seen[] = true; uv)), ctx)

@testset "Calibration pipeline" begin
    @testset "fitcalibrate defaults (solve + corrected output)" begin
        uvset, _ = _build_fringe_uvset()
        pipe = CalibrationPipeline([FringeFit(), BandpassEstimator(), TemporalSmoother()])
        sol, out = fitcalibrate(pipe, uvset)
        @test sol isa CAL.CalibrationSolution
        @test out !== nothing                   # the full pipeline sets corrected output
        # The fused output tail does not perturb the solve: gains ≡ the fit-only gains.
        @test parent(gains(sol)) == parent(gains(fit(pipe, uvset)))
        @test sol.info.nant == 4
        @test isempty(sol.postcal)              # no a-priori step
    end

    @testset "pipeline ReduceSteps fuse into the streaming pass" begin
        uvset, _ = _build_fringe_uvset(nbands = 3, nchan = 4)
        chain = [FringeFit(), BandpassEstimator(), TemporalSmoother()]
        pipe = CalibrationPipeline(vcat(
            chain, [AverageFrequency(nout = 1), CombineSpw(), AverageTime(seconds = 1.0e6)],
        ))
        sol, out = fitcalibrate(pipe, uvset)
        # Equivalent hand-written reducer chain (same order) on the two-pass
        # corrected set.
        reducer = uv -> UVP.time_bin_average(UVP.combine_spw(UVP.frequency_average(uv; nout = 1)), 1.0e6)
        out_ref = reducer(Gustavo.apply_calibration(uvset, sol))
        @test Set(keys(DimensionalData.branches(out))) ==
            Set(keys(DimensionalData.branches(out_ref)))
        for (k, leaf) in DimensionalData.branches(out_ref)
            Vr = parent(leaf[:vis])
            Vf = parent(DimensionalData.branches(out)[k][:vis])
            @test size(Vf) == size(Vr)
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
        end
    end

    @testset "fitcalibrate reduce kwarg ≡ pipeline ReduceStep" begin
        uvset, _ = _build_fringe_uvset()
        chain = FringeFit() |> BandpassEstimator() |> TemporalSmoother()
        _, out_kw = fitcalibrate(chain, uvset; reduce = [AverageFrequency(nout = 1)])
        _, out_pl = fitcalibrate(
            CalibrationPipeline([
                FringeFit(), BandpassEstimator(), TemporalSmoother(), AverageFrequency(nout = 1),
            ]),
            uvset,
        )
        for (k, leaf) in DimensionalData.branches(out_kw)
            Vr = parent(leaf[:vis])
            Vf = parent(DimensionalData.branches(out_pl)[k][:vis])
            @test size(Vf) == size(Vr)
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
        end
    end

    @testset "extensibility: custom ReduceStep runs in the pipeline" begin
        uvset, _ = _build_fringe_uvset()
        seen = Ref(false)
        sol, _ = fitcalibrate(
            CalibrationPipeline([
                FringeFit(), BandpassEstimator(), TemporalSmoother(), _ProbeReduce(seen),
            ]),
            uvset,
        )
        @test seen[]
        @test sol isa CAL.CalibrationSolution

        # Arbitrary run_step-based action steps no longer thread through a
        # pipeline (solve steps share one compiled model + streaming passes).
        struct_probe = Gustavo.DataTransformStep(CalFunction((stack, win) -> nothing))
        @test_throws ErrorException Gustavo.run_step(struct_probe, Gustavo.CalibrationContext())
    end

    @testset "band edges" begin
        uvset, _ = _build_fringe_uvset(nbands = 2, nchan = 8)   # fraction 0.2 → 1 edge chan

        flagged = UVP.flag_band_edges(uvset; mode = :flag_fraction, fraction = 0.2)
        for (k, leaf) in DimensionalData.branches(flagged)
            W = parent(leaf[:weights])
            W0 = parent(DimensionalData.branches(uvset)[k][:weights])
            @test all(W[1, :, :, :] .== 0)
            @test all(W[end, :, :, :] .== 0)
            @test W[2:(end - 1), :, :, :] == W0[2:(end - 1), :, :, :]
        end

        trimmed = UVP.flag_band_edges(uvset; mode = :trim, fraction = 0.2)
        for (_, leaf) in DimensionalData.branches(trimmed)
            @test size(parent(leaf[:vis]), 1) == 6
            @test length(channel_freqs(DimensionalData.metadata(leaf).freq_setup)) == 6
        end

        @test_throws ErrorException UVP.flag_band_edges(uvset; mode = :bogus, fraction = 0.1)

        # As a fused reduce step on the corrected output.
        _, out = fitcalibrate(
            CalibrationPipeline(
                [
                    FringeFit(), BandpassEstimator(), TemporalSmoother(),
                    FlagBandEdges(mode = :flag_fraction, fraction = 0.2),
                ]
            ),
            uvset,
        )
        for (_, leaf) in DimensionalData.branches(out)
            W = parent(leaf[:weights])
            @test all(W[1, :, :, :] .== 0)
            @test all(W[end, :, :, :] .== 0)
        end
    end

    @testset "ref_ant by station code" begin
        uvset, _ = _build_fringe_uvset()    # antennas named A1..A4
        @test Gustavo._resolve_ref_ant(3, uvset) == 3
        @test Gustavo._resolve_ref_ant("A2", uvset) == 2
        @test Gustavo._resolve_ref_ant(:A4, uvset) == 4
        @test_throws ErrorException Gustavo._resolve_ref_ant("ZZ", uvset)
        # End-to-end (new engine): code "A1" resolves to index 1 → identical solve.
        by_code = fit(FringeFit(), uvset; ref_ant = "A1")
        by_idx = fit(FringeFit(), uvset; ref_ant = 1)
        @test parent(gains(by_code)) ≈ parent(gains(by_idx))
    end

    @testset "FringeFit-less pipeline" begin
        uvset, _ = _build_fringe_uvset()
        # A standalone BandpassEstimator fit needs no FringeFit step, no
        # pipeline-level anchor check, and no fringe-estimator diagnostics.
        sol = fit(BandpassEstimator(), uvset; ref_ant = 2)
        @test stage_names(sol) == [:bandpass]
        @test !haskey(sol.info, :search)
        @test sol.info.nant == 4
        _, out = fitcalibrate(BandpassEstimator(), uvset; ref_ant = 2)
        @test out !== nothing
    end

    @testset "defaults" begin
        f = FringeFit()
        @test f.model.terms == default_fringe_terms()
        # The default list: 5 feed-by-feed instrument components — dispersion
        # (dTEC) and SBD are no longer part of FringeModel's term list; they
        # are a separate DispersionSBDFit step.
        @test length(f.model.terms) == 5
        # No feed-specific Rate element: the R–L rate is tied ≡ 0 by default.
        @test !any(
            t -> t isa CAL.TiedComponent && t.component.term isa CAL.Rate &&
                t.tying isa CAL.FeedComponent,
            f.model.terms,
        )
        @test f.estimator isa MatchedFilter
        @test f.estimator.search == FP.FringeSearch()
        @test f.estimator.closure == FP.Stationization()
        @test f.estimator.rounds == 1
        @test f.estimator.cross_hand_fit_on isa AllScans

        d = DispersionSBDFit()
        @test d.dispersion == DispersionModel()
        @test d.sbd isa SingleBandDelay

        b = BandpassEstimator()
        @test b.phase && b.amp
        @test b.amp_model == FP.PenalizedBandpass(1.0)
        @test b.select == AllScans()

        t = TemporalSmoother()
        @test t.smoother == FP.SavitzkyGolaySmoother()

        e = ExecutionConfig()
        @test e.mem_fraction == 0.6 && e.mem_budget === nothing && e.exclude_colocated

        # ref_ant is run-wide, on CalibrationPipeline, not on FringeModel.
        @test CalibrationPipeline([FringeFit()]).ref_ant == 1

        # AprioriAmplitude carries a pre-built band_cals (loading is the caller's job).
        bc = Dict(1 => :dummy)
        ap = AprioriAmplitude(bc; min_elevation_deg = 10.0)
        @test ap.band_cals === bc
        @test ap.min_elevation_deg == 10.0
        @test ap.on_missing_station == :warn
    end
end

# The run-state types the pipeline layer threads through its solve loop carry
# their contents as type parameters rather than as `Any`, so the whole solve
# context is a concrete type. This is an interface property, not a speed one:
# an `Any` field advertises no contract, and it drifts back silently.
@testset "pipeline run state is concretely typed" begin
    uvset, _ = _build_fringe_uvset()

    @testset "SolveContext" begin
        ff = FringeFit(model = FringeModel())
        geom = CAL.build_geometry(uvset)
        antennas = UVP.metadata(first(values(UVP.branches(uvset)))).antennas
        nant = length(antennas)
        mc = Gustavo.model_components(ff, (; geom, antennas))
        model = CAL.StationGainModel(phase = mc.phase, logamp = mc.logamp)
        layout = CAL.plan_parameters(model, nant, geom)
        ctx = Gustavo.SolveContext(
            model, layout, geom, CAL.GainEvaluator(model, layout), zeros(layout.nθ),
            1, nant, antennas, ST.scan_stream(uvset; geom),
            ExecutionConfig(), Dict{Symbol, Any}(),
        )
        @test isconcretetype(typeof(ctx))
        for f in (:model, :layout, :geom, :ev, :antennas, :stream, :exec)
            @test isconcretetype(fieldtype(typeof(ctx), f))
        end
        # `scratch` stays a `Dict{Symbol, Any}` by design — it is the untyped
        # cross-step channel, and its readers assert on retrieval.
        @test fieldtype(typeof(ctx), :scratch) == Dict{Symbol, Any}
    end

    @testset "ExecutionConfig carries the progress callback's type" begin
        cb = (stage, done, total) -> nothing
        e = ExecutionConfig(progress = cb)
        @test fieldtype(typeof(e), :progress) === typeof(cb)
        @test isconcretetype(typeof(ExecutionConfig()))
        @test fieldtype(typeof(ExecutionConfig()), :progress) === Nothing
        @test isconcretetype(fieldtype(typeof(e), :outer_executor))
        @test isconcretetype(fieldtype(typeof(e), :inner_executor))
        # The pipeline propagates the config at its own concrete type.
        @test fieldtype(typeof(CalibrationPipeline([FringeFit()]; exec = e)), :exec) === typeof(e)
    end

    @testset "stream group specs and transforms" begin
        t = StationWeightScale(ones(4))
        stream = ST.scan_stream(uvset; transforms = (t,))
        @test isconcretetype(eltype(stream.groups))
        @test isconcretetype(eltype(first(stream.groups).leaves))
        @test eltype(stream.transforms) === typeof(t)
        # No transforms: nothing to join, so the vector stays `Any`-typed rather
        # than becoming a `Vector{Union{}}` that could never accept an entry.
        @test eltype(ST.scan_stream(uvset).transforms) === Any
    end

    @testset "solution records its chains at their own type" begin
        t = StationWeightScale(ones(4))
        sol = fit(t |> FringeFit(), uvset)
        @test eltype(sol.transforms) === typeof(t)
        @test eltype(sol.postcal) === Any        # empty
        # A stage snapshot rebuilds the solution without widening the chain.
        @test eltype(CAL.stage_solution(sol, :fringe).transforms) === eltype(sol.transforms)
    end


    @testset "AprioriAmplitude and CalibrationPipeline.ref_ant" begin
        bc = Dict(1 => :dummy)
        @test fieldtype(typeof(AprioriAmplitude(bc)), :band_cals) === typeof(bc)
        # ref_ant's domain is what `_resolve_ref_ant` accepts: an antenna index
        # or a station code.
        @test CalibrationPipeline([FringeFit()]; ref_ant = 2).ref_ant == 2
        @test CalibrationPipeline([FringeFit()]; ref_ant = "A1").ref_ant == "A1"
        @test CalibrationPipeline([FringeFit()]; ref_ant = :A1).ref_ant === :A1
        @test_throws TypeError CalibrationPipeline([FringeFit()]; ref_ant = 2.5)
    end
end

# The solve produces a `Vector`-backed θ, but a caller may rewrap it — e.g. as a
# labelled `DimArray` — and the whole apply path must be indifferent to that.
@testset "a rewrapped θ corrects data identically" begin
    uvset, _ = _build_fringe_uvset()
    sol = fit(FringeFit(model = FringeModel()) |> BandpassEstimator(), uvset)
    sold_steps = [
        CAL.StepSolution(
            s.name, s.model, s.layout, DimArray(copy(s.θ), Dim{:param}(1:(s.layout.nθ))), s.info,
        )
            for s in sol.steps
    ]
    sold = CAL.CalibrationSolution(
        sold_steps, sol.geom, sol.info; transforms = sol.transforms, postcal = sol.postcal,
    )
    @test all(s.θ isa DimArray for s in sold.steps)

    a = Gustavo.apply_calibration(uvset, sol)
    b = Gustavo.apply_calibration(uvset, sold)
    @test Set(keys(DimensionalData.branches(a))) == Set(keys(DimensionalData.branches(b)))
    for (k, leaf) in DimensionalData.branches(a)
        Va = parent(leaf[:vis])
        Vb = parent(DimensionalData.branches(b)[k][:vis])
        # Bit-identical, not approximate: the same arithmetic on the same numbers.
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || x === y, zip(Va, Vb))
    end
end
