# Solve-step interface tests (protocol, step record, solutions of a chain of
# steps, and the fit/calibrate verbs), on ProcessingSets from `_build_fringe_ps`. Uses
# the CAL/FP aliases from test_pipeline.jl (included earlier in runtests.jl).

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")
using Distributions: LogNormal, Gamma, MvNormal

# A throwaway solve step proving the protocol defaults exist.
struct _ProtoProbe <: Gustavo.SolveStep end

# A third-party step that reads the data twice and reports each group's first
# time.
struct _TwoPassStep <: Gustavo.SolveStep end
Gustavo.provides(::_TwoPassStep) = :twopass
_first_time(group) = minimum(ms -> minimum(XRadio.times(ms)), values(group))
function Gustavo.solve(::_TwoPassStep, ctx)
    first_t = Gustavo.each_group(_first_time, ctx)
    again = Gustavo.each_group(_first_time, ctx)
    return (; first_t, again)
end

# A step whose `solve` returns something other than its diagnostics.
struct _BadInfoStep <: Gustavo.SolveStep end
Gustavo.solve(::_BadInfoStep, ctx) = 1.0

# A model value that is not a `GainComponent`.
struct _OpaqueTerm end

# The three production steps at defaults.
_full_chain(; gauge = PinAntenna(1)) = (BaselineFringeFit(; gauge), Bandpass(; gauge), AdhocPhase(; gauge))

# Every parameter of a solution, in component order.
_all_params(sol) = reduce(vcat, [vec(parent(c.params)) for c in sol.components])

@testset "Solve-step interface" begin
    gauge = PinAntenna(1)
    @testset "step protocol defaults" begin
        s = _ProtoProbe()
        @test Gustavo.model_components(s, nothing) == GainModel()
        @test Gustavo.provides(s) == :nothing
        ps, _ = _build_fringe_ps()
        @test_throws "does not define `Gustavo.solve(::_ProtoProbe, ctx)`" fit(
            _ProtoProbe(), ps,
        )
        @test_throws "must return a NamedTuple" fit(_BadInfoStep(), ps)
    end

    @testset "a step drives its own passes with each_group" begin
        ps, _ = _build_fringe_ps(; nscans = 3)
        sol = fit(_TwoPassStep(), ps)
        info = sol.info
        # One result per scan group, in the same group order on every pass.
        @test length(info.first_t) == sol.info.nscan == 3
        @test allunique(info.first_t)
        @test info.again == info.first_t
        # The runner's timing covers both passes of every group.
        @test info.t_pass > 0
        @test length(info.timing.work) == 3 && all(>(0), info.timing.decode)
    end

    @testset "built-in step declarations" begin
        @test Gustavo.provides(BaselineFringeFit(; gauge)) == :fringe
        @test Gustavo.provides(Bandpass(; gauge)) == :bandpass
        @test Gustavo.provides(AdhocPhase(; gauge)) == :adhoc
        # The default fringe fit (one round, every term per-scan) solves each
        # scan's station systems as the scan is searched.
        @test Gustavo._scan_local_solve(BaselineFringeFit(; gauge))
        # Any cross-scan coupling pools every scan's detections into one solve:
        # a residual re-search round, or a track-global inter-feed column (in
        # the base model or in one station's entry).
        @test !Gustavo._scan_local_solve(BaselineFringeFit(; rounds = 2, gauge))
        @test !Gustavo._scan_local_solve(
            BaselineFringeFit(; model = default_fringe_model(rel_time = CAL.GlobalTime()), gauge),
        )
        @test !Gustavo._scan_local_solve(
            BaselineFringeFit(;
                model = with_antenna(
                    default_fringe_model(), "AA";
                    phase = default_fringe_model(rel_time = CAL.GlobalTime()).phase,
                ),
                gauge,
            ),
        )
        @test_throws "must be a `GainComponent`" merge(default_fringe_model(); phase = (; x = _OpaqueTerm()))
        # A step's model is a GainModel, never a bare NamedTuple.
        @test_throws MethodError BaselineFringeFit(; model = (; phase = default_fringe_model().phase), gauge)
    end

    @testset "a step fit on a scan subset, carried into the full fit" begin
        nant, nspw, nchan = 4, 2, 8
        bp = [a == 1 ? 0.0 : 0.3 * sin(0.4 * c + a + f) for a in 1:nant, f in 1:2, c in 1:(nspw * nchan)]
        # ComplexF64: a Float32 search's delays leave ~3e-5 rad of bandpass
        # difference, above the tolerance below.
        ps, _ = _build_fringe_ps(; nant, nspw, nchan, nscans = 3, bandpass = bp, eltype = ComplexF64)
        sub = XRadio.query(ps; scan_name = "3")
        fr = fit(BaselineFringeFit(; gauge), ps)
        # The scans' fringe phases differ, so the full-set solution corrects the
        # subset exactly as a fit on the subset alone does only if each of its
        # scans' gains lands on that scan.
        fr3 = fit(BaselineFringeFit(; gauge), sub)
        bp_sub = fit(Bandpass(; gauge), _precal(fr, sub))
        @test _all_params(bp_sub) ≈ _all_params(fit(Bandpass(; gauge), _precal(fr3, sub))) atol = 1.0e-10
        # Noise-free and time-constant: one scan determines the bandpass all three do.
        @test maximum(abs, _all_params(bp_sub) .- _all_params(fit(Bandpass(; gauge), _precal(fr, ps)))) < 1.0e-6

        corrected = _precal(fr, ps)
        calibrate!(bp_sub, corrected; flag_bad = false, apply_flags = false)
        sol = fit(AdhocPhase(; gauge), corrected)
        @test calibrate(sol, ps) isa XRadio.ProcessingSet
    end

    @testset "every step defaults to ZeroSumPhase; an unknown antenna is named" begin
        ps, _ = _build_fringe_ps()
        @test all(st -> st.gauge === ZeroSumPhase(), (BaselineFringeFit(), Bandpass(), AdhocPhase(), AdhocPhase(FP.PerTrackAdhocSmoother())))
        @test_throws "antenna code \"ZZ\" not in antenna table" fit(BaselineFringeFit(; gauge = PinAntenna("ZZ")), ps)
    end

    @testset "a chain of steps: one solution per step" begin
        ps, _ = _build_fringe_ps()
        sols = _fit_chain(_full_chain(), ps)
        fr, bp, ad = sols

        @test all(s -> s isa CAL.CalibrationSolution, sols)
        @test [s.name for s in sols] == [:fringe, :bandpass, :adhoc]
        @test all(s -> all(c -> c isa CAL.ComponentFit, s.components), sols)
        @test all(s -> s.info isa NamedTuple, sols)
        @test all(s -> occursin("PinAntenna{Int64}(1)", s.step), sols)
        @test_throws "holds no component under bogus" fr[:bogus]

        # The fringe step's unconstrained stations and search settings are its
        # own diagnostics.
        @test haskey(fr.info, :flagged_ant)
        @test fr.info.search isa Gustavo.Fring.FringeSearch
        @test all(s -> !haskey(s.info, :flagged_ant) && !haskey(s.info, :search), (bp, ad))

        # Gains factor multiplicatively over components.
        for sol in sols
            g_full = parent(gains(sol))
            g_prod = ones(ComplexF64, size(g_full))
            for c in sol.components
                g_prod .*= parent(gains(filter(==(c), sol)))
            end
            @test g_prod ≈ g_full
        end

        # Any selection applies. Calibrating component by component is
        # calibrating by the whole solution.
        whole = calibrate(fr, ps; apply_flags = false)
        componentwise = ps
        for c in fr.components
            componentwise = calibrate(filter(==(c), fr), componentwise; apply_flags = false)
        end
        for (k, ms) in pairs(whole)
            @test isapprox(parent(componentwise[k][:visibility]), parent(ms[:visibility]); nans = true)
            @test parent(componentwise[k][:weight]) ≈ parent(ms[:weight])
        end
        @test calibrate(fr[:phase, :mbd], ps) isa XRadio.ProcessingSet
        @test calibrate(filter(c -> c.component.term isa Delay, fr), ps) isa XRadio.ProcessingSet
        @test_throws "holds no components" calibrate(filter(_ -> false, fr), ps)
        @test_throws "holds no components" calibrate!(filter(_ -> false, fr), deepcopy(ps))
    end

    @testset "selections and gains selectors" begin
        ps, _ = _build_fringe_ps()
        sol = fit(first(_full_chain()), ps)

        # A path prefix selects the components under it; their gains multiply.
        ph = sol[:phase]
        @test !isempty(ph.components)
        @test ph.name === :fringe
        @test haskey(sol, :phase) && haskey(sol, :phase, :mbd) && !haskey(sol, :phase, :bogus)
        g_group = ones(ComplexF64, size(gains(sol)))
        for c in ph.components
            g_group .*= parent(gains(sol[c.path...]))
        end
        @test parent(gains(ph)) ≈ g_group

        # `gains` keywords are DD dimension selectors, windowing the
        # evaluation under the invariant gains(sol; kw...) == gains(sol)[kw...].
        G = gains(sol)
        geom = sol.geom
        @test gains(sol; Ti = 1) == G[Ti = 1]
        @test gains(sol; Frequency = 2:3, Feed = 2) == G[Frequency = 2:3, Feed = 2]
        f2 = geom.channel_freqs[2]
        @test gains(sol; Frequency = At(f2), AntennaName = 2) == G[Frequency = At(f2), AntennaName = 2]
        @test gains(sol; Ti = Near(geom.times[end])) == G[Ti = Near(geom.times[end])]
        @test_throws ArgumentError gains(sol; Polarization = 1)
        @test_throws "unknown dimension keyword" gains(sol; Polarization = 1)
    end

    @testset "flags come from the solution's step" begin
        ps, _ = _build_fringe_ps()
        fr, bp, _ = _fit_chain(_full_chain(), ps)
        sol = CAL.CalibrationSolution(
            fr.geom, fr.name, fr.components,
            (; fr.info..., flagged_ant = ["A2"], flagged_feed = [2], flagged_scan = [1]); step = fr.step,
        )
        @test Gustavo._solution_flag_sets(sol) == Set([("A2", 2, 1)])
        @test Gustavo._solution_flag_sets(sol[:phase, :mbd]) == Set([("A2", 2, 1)])
        @test Gustavo._solution_flag_sets(bp) === nothing
        flagged(out) = any(ms -> any(parent(ms[:flag])), values(out))
        @test flagged(calibrate(sol, ps)) && !flagged(calibrate(sol, ps; apply_flags = false))
        @test !flagged(calibrate(bp, ps))
    end

    @testset "calibrate divides by the gains alone" begin
        ps, _ = _build_fringe_ps()
        ws = Dict("A1" => 1.0, "A2" => 0.5, "A3" => 1.0, "A4" => 2.0)
        scale(set, w = ws) = XRadio.ProcessingSet(
            OrderedDict{Symbol, XRadio.MeasurementSet}(
                k => scale_weights!(deepcopy(read(ms)), w) for (k, ms) in pairs(set)
            ),
            copy(DimensionalData.metadata(set)),
        )
        sols = _fit_chain(_full_chain(), scale(ps))

        # Scaling the data before or after `calibrate` gives the same result.
        before = _calibrate_chain(sols, scale(ps))
        after = scale(_calibrate_chain(sols, ps))
        @test collect(keys(before)) == collect(keys(after))
        for (k, ms) in pairs(after)
            @test isequal(parent(before[k][:visibility]), parent(ms[:visibility]))
            @test parent(before[k][:weight]) ≈ parent(ms[:weight])
        end

        # Two weight scales compose (w·(s_a s_b)²).
        sols_b = _fit_chain(_full_chain(), scale(scale(ps)))
        sols_sq = _fit_chain(_full_chain(), scale(ps, Dict(k => v^2 for (k, v) in ws)))
        for (a, b) in zip(sols_sq, sols_b)
            @test parent(gains(a)) ≈ parent(gains(b))
        end
    end

    @testset "solution Zarr round-trip" begin
        ps, _ = _build_fringe_ps()
        ws = Dict("A1" => 1.0, "A2" => 0.5, "A3" => 1.0, "A4" => 1.0)
        scaled = Gustavo.materialize(ps)
        foreach(ms -> scale_weights!(ms, ws), values(scaled))
        sols = _fit_chain(_full_chain(), scaled)
        dir = mktempdir()
        for sol in sols
            path = joinpath(dir, "$(sol.name).zarr")
            CAL.save_solution(path, sol)
            back = CAL.load_solution(path)
            @test back.components == sol.components
            @test back.name == sol.name
            @test back.info == sol.info
            @test back.step == sol.step
            @test back.geom.antennas == sol.geom.antennas && back.geom.times == sol.geom.times
            @test parent(gains(back)) == parent(gains(sol))
        end

        sol = first(sols)
        path = joinpath(dir, "fringe.zarr")
        @test_throws "exists" CAL.save_solution(path, sol)
        @test_throws "not a Gustavo solution store" CAL.load_solution(joinpath(path, "geometry"))

        # A diagnostic with no stored form fails the save and leaves nothing behind.
        bad = CalibrationSolution(sol.geom, sol.name, sol.components, (; hook = x -> x))
        badpath = joinpath(dir, "bad.zarr")
        @test_throws "cannot save info.hook" CAL.save_solution(badpath, bad)
        @test !ispath(badpath)
    end

    @testset "components with hyperpriors round-trip through Zarr" begin
        rw = RandomWalkPrior(order = 2, σ = LogNormal(0, 1), init = MvNormal(zeros(2), [1.0 0.1; 0.1 2.0]))
        c = GainComponent(
            ConstantTerm(); Ti = InstrumentScans([1.0, 5.0]), Frequency = FreqGroups([1:3, 4:8]),
            Feed = SingleFeed(2), prior = (Ti = rw, Frequency = OUPrior(scale = 3.0, σ = Gamma(2.0, 1.0))),
        )
        back = CAL._decode(CAL._encode_checked(c, "c"))
        @test typeof(back) == typeof(c)
        @test back.prior.Ti.init == rw.init && back.prior.Ti.σ == rw.σ
        @test back.prior.Frequency == c.prior.Frequency
        @test (back.term, back.Ti, back.Frequency, back.Feed) == (c.term, c.Ti, c.Frequency, c.Feed)
        @test CAL._decode(CAL._encode_checked(PolynomialFreq(3), "p")) === PolynomialFreq(3)
    end
end
