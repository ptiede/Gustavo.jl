# Modular calibration-pipeline tests: the pipeline-level surface (a pipeline
# as a vector, `fit`, `calibrate` and its `post`, the output reducers, defaults).
# Reuses `_build_fringe_uvset` and the CAL/FP/UVP aliases from test_pipeline.jl
# (included earlier in runtests.jl).

@testset "Calibration pipeline" begin
    @testset "fit, then calibrate" begin
        uvset, _ = _build_fringe_uvset()
        pipe = (BaselineFringeFit(), Bandpass(), AdhocPhase())
        sol = fit(pipe, uvset; gauge = PinAntenna(1))
        @test sol isa CAL.CalibrationSolution
        @test sol.info.nant == 4
        @test sol.sequence == pipe
        out = calibrate(sol, uvset)
        @test out isa UVP.UVSet
        ref = Gustavo.UVData.apply_calibration(uvset, sol)
        for (k, leaf) in DimensionalData.branches(ref)
            @test isequal(parent(DimensionalData.branches(out)[k][:vis]), parent(leaf[:vis]))
        end
    end

    @testset "calibrate's post runs on each corrected group" begin
        uvset, _ = _build_fringe_uvset(nspw = 3, nchan = 4)
        sol = fit(BaselineFringeFit() |> Bandpass() |> AdhocPhase(), uvset; gauge = PinAntenna(1))
        out = calibrate(
            sol, uvset; post = AverageTime(seconds = 1.0e6) ∘ CombineSpw() ∘ AverageFrequency(nout = 1),
        )
        reducer = uv -> UVP.time_bin_average(UVP.combine_spw(UVP.frequency_average(uv; nout = 1)), 1.0e6)
        out_ref = reducer(Gustavo.UVData.apply_calibration(uvset, sol))
        @test Set(keys(DimensionalData.branches(out))) ==
            Set(keys(DimensionalData.branches(out_ref)))
        for (k, leaf) in DimensionalData.branches(out_ref)
            Vr = parent(leaf[:vis])
            Vf = parent(DimensionalData.branches(out)[k][:vis])
            @test size(Vf) == size(Vr)
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
        end

        seen = Ref(0)
        calibrate(sol, uvset; post = uv -> (seen[] += 1; uv))
        @test seen[] == length(ST.scan_stream(uvset).groups)
    end

    @testset "a pipeline holds solve steps, transforms and a-priori steps" begin
        uvset, _ = _build_fringe_uvset()
        @test_throws "not AverageFrequency" fit(
            [BaselineFringeFit(), AverageFrequency(nout = 1)], uvset; gauge = PinAntenna(1),
        )
        @test_throws "holds no solve step" fit([StationWeightScale(ones(4))], uvset; gauge = PinAntenna(1))
    end

    @testset "reduce steps apply eagerly as functors" begin
        uvset, _ = _build_fringe_uvset(nspw = 2, nchan = 8)

        # The reducer types are the one public spelling of each reduction;
        # calling one on a `UVSet` (or piping into it) applies it eagerly and
        # matches the internal kernels.
        red = uvset |> AverageFrequency(nout = 1) |> CombineSpw()
        red_ref = UVP.combine_spw(UVP.frequency_average(uvset; nout = 1))
        @test Set(keys(DimensionalData.branches(red))) ==
            Set(keys(DimensionalData.branches(red_ref)))
        for (k, leaf) in DimensionalData.branches(red)
            @test isequal(
                parent(leaf[:vis]), parent(DimensionalData.branches(red_ref)[k][:vis])
            )
        end

        # `AverageTime()` with no bin width collapses each scan to one sample.
        per_scan = AverageTime()(uvset)
        per_scan_ref = UVP.scan_average(uvset)
        for (k, leaf) in DimensionalData.branches(per_scan)
            @test size(parent(leaf[:vis]), 2) == 1
            @test isequal(
                parent(leaf[:vis]), parent(DimensionalData.branches(per_scan_ref)[k][:vis])
            )
        end

        flag_eager = FlagSpwEdges(mode = :flag_fraction, fraction = 0.2)(uvset)
        flag_ref = UVP.flag_spw_edges(uvset; mode = :flag_fraction, fraction = 0.2)
        for (k, leaf) in DimensionalData.branches(flag_eager)
            @test parent(leaf[:weights]) ==
                parent(DimensionalData.branches(flag_ref)[k][:weights])
        end
    end

    @testset "band edges" begin
        uvset, _ = _build_fringe_uvset(nspw = 2, nchan = 8)   # fraction 0.2 → 1 edge chan

        flagged = UVP.flag_spw_edges(uvset; mode = :flag_fraction, fraction = 0.2)
        for (k, leaf) in DimensionalData.branches(flagged)
            F = parent(leaf[:flags])
            @test all(F[1, :, :, :])
            @test all(F[end, :, :, :])
            @test !any(F[2:(end - 1), :, :, :])
            # Edge flagging records a decision; it does not rewrite the data.
            @test parent(leaf[:weights]) ==
                parent(DimensionalData.branches(uvset)[k][:weights])
        end

        trimmed = UVP.flag_spw_edges(uvset; mode = :trim, fraction = 0.2)
        for (_, leaf) in DimensionalData.branches(trimmed)
            @test size(parent(leaf[:vis]), 1) == 6
            @test length(channel_freqs(DimensionalData.metadata(leaf).freq_setup)) == 6
        end

        @test_throws ErrorException UVP.flag_spw_edges(uvset; mode = :bogus, fraction = 0.1)

        # As `calibrate`'s `post` on the corrected output.
        sol = fit([BaselineFringeFit(), Bandpass(), AdhocPhase()], uvset; gauge = PinAntenna(1))
        out = calibrate(sol, uvset; post = FlagSpwEdges(mode = :flag_fraction, fraction = 0.2))
        for (_, leaf) in DimensionalData.branches(out)
            F = parent(leaf[:flags])
            @test all(F[1, :, :, :])
            @test all(F[end, :, :, :])
        end
    end

    @testset "gauge by station code" begin
        uvset, _ = _build_fringe_uvset()    # antennas named A1..A4
        names = Gustavo._antenna_names(uvset)
        @test resolve_gauge(PinAntenna(3), names).refs == [3]
        @test resolve_gauge(PinAntenna("A2"), names).refs == [2]
        @test resolve_gauge(PinAntenna(:A4), names).refs == [4]
        # A ranked list resolves entry by entry, keeping order.
        @test resolve_gauge(PinAntenna(["A4", 1]), names).refs == [4, 1]
        @test resolve_gauge(ZeroSumPhase(antennas = ["A2", "A4"]), names).antennas == [2, 4]
        @test_throws ErrorException resolve_gauge(PinAntenna("ZZ"), names)
        @test_throws ErrorException resolve_gauge(ZeroSumPhase(antennas = ["ZZ"]), names)
        # End-to-end (new engine): code "A1" resolves to index 1 → identical solve.
        by_code = fit(BaselineFringeFit(), uvset; gauge = PinAntenna("A1"))
        by_idx = fit(BaselineFringeFit(), uvset; gauge = PinAntenna(1))
        @test parent(gains(by_code)) ≈ parent(gains(by_idx))
    end

    @testset "BaselineFringeFit-less pipeline" begin
        uvset, _ = _build_fringe_uvset()
        # A standalone Bandpass fit needs no BaselineFringeFit step, no
        # pipeline-level anchor check, and no fringe-estimator diagnostics.
        sol = fit(Bandpass(), uvset; gauge = PinAntenna(2))
        @test keys(sol) == [:bandpass]
        @test !haskey(sol.info, :search)
        @test sol.info.nant == 4
        @test calibrate(sol, uvset) isa UVP.UVSet
    end

    @testset "defaults" begin
        f = BaselineFringeFit()
        @test f.model == default_fringe_terms()
        # The default model: 4 feed-by-feed instrument components. Dispersion
        # (dTEC) and SBD are a separate DispersionSBDFit step, and there is no
        # inter-feed PHASE offset — see `default_fringe_terms`.
        @test isempty(f.model.logamp)
        @test length(f.model.phase) == 4
        @test !haskey(f.model.phase, :rel_phase)
        # No feed-specific Rate element: the inter-feed rate is tied ≡ 0 by default.
        @test !any(
            t -> t isa CAL.GainComponent && t.term isa CAL.Rate &&
                t.Feed isa CAL.SingleFeed,
            f.model.phase,
        )
        @test f.search == FP.FringeSearch()
        @test f.closure == FP.Stationization()
        @test f.rounds == 1
        @test f.steer_cells == 9.0

        d = DispersionSBDFit()
        @test d.dispersion == DispersionModel()
        @test d.sbd isa SingleBandDelay

        b = Bandpass()
        @test b.model == default_bandpass_terms()
        @test b.smoother isa FP.JointSmoother
        @test b.smoother.phase == FP.FreeShape()
        @test b.smoother.amp == FP.FreeShape()

        t = AdhocPhase()
        @test t.model == default_adhoc_terms()
        @test t.smoother == FP.SavitzkyGolaySmoother()

        e = ExecutionConfig()
        @test e.mem_fraction == 0.6 && e.mem_budget === nothing

        # AprioriAmplitude carries a pre-built spw_cals (loading is the caller's job).
        bc = Dict(1 => :dummy)
        ap = AprioriAmplitude(bc; min_elevation_deg = 10.0)
        @test ap.spw_cals === bc
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
        ff = BaselineFringeFit()
        geom = CAL.build_geometry(uvset)
        antennas = UVP.metadata(first(values(UVP.branches(uvset)))).antennas
        nant = length(antennas)
        mc = Gustavo.model_components(ff, (; geom, antennas))
        model = CAL.GainModel(phase = mc.phase, logamp = mc.logamp)
        layout = CAL.plan_parameters(model, nant, geom)
        ctx = Gustavo.SolveContext(
            model, layout, geom, zeros(layout.nθ),
            PinAntenna(1), nant, antennas, ST.scan_stream(uvset; geom), :fringe,
            Gustavo._PassTiming[],
        )
        @test isconcretetype(typeof(ctx))
        for f in (:model, :layout, :geom, :antennas, :stream, :passes)
            @test isconcretetype(fieldtype(typeof(ctx), f))
        end
    end

    @testset "ExecutionConfig carries the progress callback's type" begin
        cb = (stage, done, total) -> nothing
        e = ExecutionConfig(progress = cb)
        @test fieldtype(typeof(e), :progress) === typeof(cb)
        @test isconcretetype(typeof(ExecutionConfig()))
        @test fieldtype(typeof(ExecutionConfig()), :progress) === Nothing
        @test isconcretetype(fieldtype(typeof(e), :outer_executor))
        @test isconcretetype(fieldtype(typeof(e), :inner_executor))
    end

    @testset "ProgressLogger" begin
        buf = IOBuffer()
        p = ProgressLogger(min_interval = 0, io = buf)
        p(:fringe, 0, 3)
        p(:fringe, 1, 3)
        p(:fringe, 2, 3)
        p(:fringe, 3, 3)
        lines = split(strip(String(take!(buf))), '\n')
        @test length(lines) == 4
        @test occursin("starting", lines[1]) && occursin("3 scan groups", lines[1])
        @test occursin("ETA", lines[2]) && occursin("1/3", lines[2])
        @test occursin("ETA", lines[3]) && occursin("2/3", lines[3])
        @test occursin("done", lines[4]) && occursin("3/3", lines[4])

        # Intermediate scans are throttled; only the always-on start/finish print.
        buf2 = IOBuffer()
        p2 = ProgressLogger(min_interval = 3600, io = buf2)
        p2(:fringe, 0, 3)
        p2(:fringe, 1, 3)
        p2(:fringe, 2, 3)
        p2(:fringe, 3, 3)
        lines2 = split(strip(String(take!(buf2))), '\n')
        @test length(lines2) == 2
        @test occursin("starting", lines2[1])
        @test occursin("done", lines2[2])

        # A new stage (or a repeated pass restarting at done == 0) always
        # reports, regardless of the throttle.
        buf3 = IOBuffer()
        p3 = ProgressLogger(min_interval = 3600, io = buf3)
        p3(:fringe, 0, 2)
        p3(:fringe, 2, 2)
        p3(:bandpass, 0, 1)
        p3(:bandpass, 1, 1)
        lines3 = split(strip(String(take!(buf3))), '\n')
        @test length(lines3) == 4
        @test occursin("fringe", lines3[1]) && occursin("fringe", lines3[2])
        @test occursin("bandpass", lines3[3]) && occursin("bandpass", lines3[4])

        # Wired through a real fit: the pipeline's stages are named in the log.
        buf4 = IOBuffer()
        sol = fit(
            BaselineFringeFit() |> Bandpass(),
            uvset;
            exec = ExecutionConfig(progress = ProgressLogger(min_interval = 0, io = buf4)),
            gauge = PinAntenna(1),
        )
        @test sol isa CAL.CalibrationSolution
        out = String(take!(buf4))
        @test occursin("fringe", out) && occursin("bandpass", out)
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

    @testset "the solution records its pipeline and gauge" begin
        t = StationWeightScale(ones(4))
        ff = BaselineFringeFit()
        sol = fit(t |> ff, uvset; gauge = PinAntenna("A1"))
        @test sol.sequence == (t, ff)
        # Recorded as a tuple whatever the pipeline was given as.
        @test fit([t, ff], uvset; gauge = PinAntenna("A1")).sequence == (t, ff)
        @test recorded_transforms(sol) == Any[t]
        # The gauge as given, not as resolved against the stations.
        @test sol.gauge.refs == "A1"
        # A step selection carries both along.
        @test sol[1:1].sequence == sol.sequence
        @test sol[1:1].gauge === sol.gauge

        bc = Dict(1 => :dummy)
        @test fieldtype(typeof(AprioriAmplitude(bc)), :spw_cals) === typeof(bc)
        # The gauge keyword is typed, so a bare index is rejected rather than
        # silently treated as a gauge.
        @test_throws TypeError fit(ff, uvset; gauge = 2)
    end
end

# The solve produces a `Vector`-backed θ, but a caller may rewrap it — e.g. as a
# labelled `DimArray` — and the whole apply path must be indifferent to that.
@testset "a rewrapped θ corrects data identically" begin
    uvset, _ = _build_fringe_uvset()
    sol = fit(BaselineFringeFit() |> Bandpass(), uvset; gauge = PinAntenna(1))
    sold_steps = [
        CAL.StepSolution(
            s.name, s.model, s.layout, DimArray(copy(s.θ), Dim{:param}(1:(s.layout.nθ))), s.info,
        )
            for s in sol.steps
    ]
    sold = CAL.CalibrationSolution(sold_steps, sol.geom, sol.info; sol.sequence, sol.gauge)
    @test all(s.θ isa DimArray for s in sold.steps)

    a = Gustavo.UVData.apply_calibration(uvset, sol)
    b = Gustavo.UVData.apply_calibration(uvset, sold)
    @test Set(keys(DimensionalData.branches(a))) == Set(keys(DimensionalData.branches(b)))
    for (k, leaf) in DimensionalData.branches(a)
        Va = parent(leaf[:vis])
        Vb = parent(DimensionalData.branches(b)[k][:vis])
        # Bit-identical, not approximate: the same arithmetic on the same numbers.
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || x === y, zip(Va, Vb))
    end
end
