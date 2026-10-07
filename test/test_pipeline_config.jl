# The calibration surface: `fit`, `calibrate` and its `post`, gauges,
# provenance, defaults.
# Reuses `_build_fringe_ps` and the CAL/FP aliases from test_pipeline.jl
# (included earlier in runtests.jl).

@testset "Calibration surface" begin
    @testset "fit, then calibrate" begin
        ps, _ = _build_fringe_ps()
        gauge = PinAntenna(1)
        sols = _fit_chain((BaselineFringeFit(; gauge), Bandpass(; gauge), AdhocPhase(; gauge)), ps)
        @test all(s -> s isa CAL.CalibrationSolution, sols)
        @test all(s -> length(s.geom.stations) == 4, sols)
        @test [only(keys(s.steps)) for s in sols] == [:fringe, :bandpass, :adhoc]
        @test startswith(first(sols).provenance.pipeline, "BaselineFringeFit")
        out = _calibrate_chain(sols, ps)
        @test out isa XRadio.ProcessingSet
        @test collect(keys(out)) == collect(keys(ps))
    end

    @testset "calibrate's post runs on each corrected Measurement Set" begin
        ps, _ = _build_fringe_ps(nspw = 3, nchan = 4)
        gauge = PinAntenna(1)
        sol = fit(BaselineFringeFit(; gauge), ps)
        seen = Threads.Atomic{Int}(0)
        calibrate(sol, ps; post = ms -> (Threads.atomic_add!(seen, 1); ms))
        @test seen[] == length(ps)
    end

    @testset "gauge by station code" begin
        ps, _ = _build_fringe_ps()    # antennas named A1..A4
        names = CAL.DataGeometry(ps).stations
        @test resolve_gauge(PinAntenna(3), names).refs == [3]
        @test resolve_gauge(PinAntenna("A2"), names).refs == [2]
        @test resolve_gauge(PinAntenna(:A4), names).refs == [4]
        # A ranked list resolves entry by entry, keeping order.
        @test resolve_gauge(PinAntenna(["A4", 1]), names).refs == [4, 1]
        @test resolve_gauge(ZeroSumPhase(antennas = ["A2", "A4"]), names).antennas == [2, 4]
        @test_throws ErrorException resolve_gauge(PinAntenna("ZZ"), names)
        @test_throws ErrorException resolve_gauge(ZeroSumPhase(antennas = ["ZZ"]), names)
        # End-to-end (new engine): code "A1" resolves to index 1 → identical solve.
        by_code = fit(BaselineFringeFit(; gauge = PinAntenna("A1")), ps)
        by_idx = fit(BaselineFringeFit(; gauge = PinAntenna(1)), ps)
        @test parent(gains(by_code)) ≈ parent(gains(by_idx))
    end

    @testset "Bandpass without BaselineFringeFit" begin
        ps, _ = _build_fringe_ps()
        # A standalone Bandpass fit needs no BaselineFringeFit step, no anchor
        # check, and no fringe-estimator diagnostics.
        sol = fit(Bandpass(; gauge = PinAntenna(2)), ps)
        @test collect(keys(sol.steps)) == [:bandpass]
        @test !haskey(sol.steps[:bandpass], :search)
        @test length(sol.geom.stations) == 4
        @test calibrate(sol, ps) isa XRadio.ProcessingSet
    end

    @testset "defaults" begin
        f = BaselineFringeFit(; gauge = PinAntenna(1))
        @test f.model == default_fringe_terms()
        # The default model: delay, inter-feed delay and rate, and no constant
        # phase — see `default_fringe_terms`.
        @test isempty(f.model.logamp)
        @test length(f.model.phase) == 3
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

        b = Bandpass(; gauge = PinAntenna(1))
        @test b.model == default_bandpass_terms()
        @test b.smoother isa FP.JointSmoother
        @test isnothing(b.model.phase.bandpass.prior) && isnothing(b.model.logamp.bandpass.prior)

        t = AdhocPhase(; gauge = PinAntenna(1))
        @test t.model == default_adhoc_terms()
        @test t.smoother == FP.PerTrackAdhocSmoother()
        @test t.model.phase.adhoc.prior == FP.default_adhoc_prior()
    end
end

# The run-state types `fit` threads through its solve loop carry
# their contents as type parameters rather than as `Any`, so the whole solve
# context is a concrete type. This is an interface property, not a speed one:
# an `Any` field advertises no contract, and it drifts back silently.
@testset "run state is concretely typed" begin
    ps, _ = _build_fringe_ps()

    @testset "SolveContext" begin
        ff = BaselineFringeFit(; gauge = PinAntenna(1))
        geom = CAL.DataGeometry(ps)
        groups = XRadio.groupby(ps, XRadio.ByScan())
        sizes = [Gustavo._group_bytes(g) for g in values(groups)]
        gauge = resolve_gauge(PinAntenna(1), geom.stations)
        ctx = Gustavo._step_context(ff, (; geom), gauge, groups, sizes, ExecutionConfig())
        @test isconcretetype(typeof(ctx))
        for f in (:model, :layout, :geom, :θ, :gauge, :groups, :exec, :passes)
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

        # Wired through real fits: each step's stage is named in the log.
        buf4 = IOBuffer()
        sols = _fit_chain(
            (BaselineFringeFit(; gauge = PinAntenna(1)), Bandpass(; gauge = PinAntenna(1))), ps;
            exec = ExecutionConfig(progress = ProgressLogger(min_interval = 0, io = buf4)),
        )
        @test all(s -> s isa CAL.CalibrationSolution, sols)
        out = String(take!(buf4))
        @test occursin("fringe", out) && occursin("bandpass", out)
    end

    @testset "the solution records its step, gauge included" begin
        ff = BaselineFringeFit(; gauge = PinAntenna("A1"))
        sol = fit(ff, ps)
        @test sol.provenance.pipeline == sprint(show, ff; context = :limit => true)
        # The gauge as given, not as resolved against the stations.
        @test occursin(sprint(show, PinAntenna("A1")), sol.provenance.pipeline)
        # A step selection carries it along.
        @test sol[:fringe].provenance == sol.provenance

        # A bare index is not a gauge, so it is rejected rather than silently
        # treated as one.
        @test_throws "`gauge` must be an AbstractGauge, got 2" BaselineFringeFit(; gauge = 2)
        @test_throws "BaselineFringeFit needs a `gauge`" BaselineFringeFit()
    end
end

# The solve stores each component's parameters over an `Array`, but a caller may
# rewrap them — e.g. over a view — and the whole apply path must be indifferent to that.
@testset "rewrapped parameters correct data identically" begin
    ps, _ = _build_fringe_ps()
    sol = last(_fit_chain((BaselineFringeFit(; gauge = PinAntenna(1)), Bandpass(; gauge = PinAntenna(1))), ps))
    rewrapped = [
        CAL.SolvedComponent(
            c.step, c.path, c.component,
            DimArray(view(copy(parent(c.params)), axes(c.params)...), dims(c.params)),
        )
            for c in sol.components
    ]
    sold = CAL.CalibrationSolution(sol.geom, rewrapped, sol.steps, sol.info)
    @test all(parent(c.params) isa SubArray for c in sold.components)

    a = calibrate(sol, ps)
    b = calibrate(sold, ps)
    @test collect(keys(a)) == collect(keys(b))
    for (k, ms) in pairs(a)
        Va = parent(ms[:visibility])
        Vb = parent(b[k][:visibility])
        # Bit-identical, not approximate: the same arithmetic on the same numbers.
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || x === y, zip(Va, Vb))
    end
end
