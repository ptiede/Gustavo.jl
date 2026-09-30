# Composable-pipeline interface tests (protocol, transforms, stage
# provenance/snapshots, and the fit/calibrate verbs), on ProcessingSets from
# `_build_fringe_ps`. Uses the CAL/FP/UVP aliases from test_pipeline.jl
# (included earlier in runtests.jl).

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")
using Distributions: LogNormal, Gamma, MvNormal

# A throwaway solve step proving the protocol defaults exist.
struct _ProtoProbe <: Gustavo.SolveStep end

# A THIRD-PARTY-shaped solve step: declares provides like a built-in one, but
# `_parse_pipeline` knows nothing about its type — proving it routes by
# abstract type alone, not an isa-chain in disguise.
struct _ThirdPartyStep <: Gustavo.SolveStep end
Gustavo.provides(::_ThirdPartyStep) = :thirdparty

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

# The full three-stage production pipeline at defaults.
_full_chain() = BaselineFringeFit() |> Bandpass() |> AdhocPhase()

# Every parameter of a solution, in component order.
_all_params(sol) = reduce(vcat, [vec(parent(c.params)) for c in sol.components])

@testset "Composable pipeline interface" begin
    @testset "step protocol defaults" begin
        s = _ProtoProbe()
        @test Gustavo.model_components(s, nothing) == GainModel()
        @test Gustavo.provides(s) == :nothing
        ps, _ = _build_fringe_ps()
        @test_throws "does not define `Gustavo.solve(::_ProtoProbe, ctx)`" fit(
            _ProtoProbe(), ps; gauge = PinAntenna(1),
        )
        @test_throws "must return a NamedTuple" fit(_BadInfoStep(), ps; gauge = PinAntenna(1))
    end

    @testset "a step drives its own passes with each_group" begin
        ps, _ = _build_fringe_ps(; nscans = 3)
        sol = fit(_TwoPassStep(), ps; gauge = PinAntenna(1))
        info = sol.steps[:twopass]
        # One result per scan group, in the same group order on every pass.
        @test length(info.first_t) == sol.info.nscan == 3
        @test allunique(info.first_t)
        @test info.again == info.first_t
        # The runner's timing covers both passes of every group.
        @test info.t_pass > 0
        @test length(info.timing.work) == 3 && all(>(0), info.timing.decode)
    end

    @testset "built-in step declarations" begin
        @test Gustavo.provides(BaselineFringeFit()) == :fringe
        @test Gustavo.provides(Bandpass()) == :bandpass
        @test Gustavo.provides(AdhocPhase()) == :adhoc
        # The default fringe fit (one round, every term per-scan) solves each
        # scan's station systems as the scan is searched.
        @test Gustavo._scan_local_solve(BaselineFringeFit())
        # Any cross-scan coupling pools every scan's detections into one solve:
        # a residual re-search round, or a track-global inter-feed column (in
        # the base model or in one station's entry).
        @test !Gustavo._scan_local_solve(BaselineFringeFit(rounds = 2))
        @test !Gustavo._scan_local_solve(
            BaselineFringeFit(model = default_fringe_terms(rel_time = CAL.GlobalTime())),
        )
        @test !Gustavo._scan_local_solve(
            BaselineFringeFit(
                model = with_station(
                    default_fringe_terms(), "AA";
                    phase = default_fringe_terms(rel_time = CAL.GlobalTime()).phase,
                ),
            ),
        )
        @test_throws "must be a `GainComponent`" merge(default_fringe_terms(); phase = (; x = _OpaqueTerm()))
        # A step's model is a GainModel, never a bare NamedTuple.
        @test_throws MethodError BaselineFringeFit(model = (; phase = default_fringe_terms().phase))
    end

    @testset "a step fit on a scan subset, carried into the full fit" begin
        nant, nspw, nchan = 4, 2, 8
        bp = [a == 1 ? 0.0 : 0.3 * sin(0.4 * c + a + f) for a in 1:nant, f in 1:2, c in 1:(nspw * nchan)]
        # ComplexF64: a Float32 search's delays leave ~3e-5 rad of bandpass
        # difference, above the tolerance below.
        ps, _ = _build_fringe_ps(; nant, nspw, nchan, nscans = 3, bandpass = bp, eltype = ComplexF64)
        sub = XRadio.query(ps; scan_name = "3")
        gauge = PinAntenna(1)
        fr = fit(BaselineFringeFit(), ps; gauge)
        # The scans' fringe phases differ, so the full-set solution corrects the
        # subset exactly as a fit on the subset alone does only if each of its
        # scans' gains lands on that scan.
        fr3 = fit(BaselineFringeFit(), sub; gauge)
        bp_sub = fit(ApplySolution(fr) |> Bandpass(), sub; gauge)
        @test _all_params(bp_sub) ≈ _all_params(fit(ApplySolution(fr3) |> Bandpass(), sub; gauge)) atol = 1.0e-10
        # Noise-free and time-constant: one scan determines the bandpass all three do.
        @test maximum(abs, _all_params(bp_sub) .- _all_params(fit(ApplySolution(fr) |> Bandpass(), ps; gauge))) < 1.0e-6

        sol = fit(ApplySolution(fr) |> ApplySolution(bp_sub) |> AdhocPhase(), ps; gauge)
        @test startswith(sol.provenance.pipeline, "ApplySolution(fringe) |> ApplySolution(bandpass) |> AdhocPhase")
        @test calibrate(sol, ps) isa XRadio.ProcessingSet
    end

    @testset "a fit without a gauge throws, naming the stations" begin
        ps, _ = _build_fringe_ps()
        @test_throws "no gauge given" fit(BaselineFringeFit(), ps)
        @test_throws "A1, A2, A3, A4" fit(BaselineFringeFit(), ps)
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
        # Each step's solution is the branch named by `provides`, so the default
        # `:nothing` collides with itself too.
        @test_throws "more than one step provides :nothing" Gustavo._check_unique_provides(
            Gustavo.SolveStep[_ProtoProbe(), _ProtoProbe()]
        )
        # _parse_pipeline routes any SolveStep (built-in or third-party) into
        # solve_steps by abstract type alone, in declared order — it does not
        # reorder or reject based on that order.
        br = Gustavo._parse_pipeline([BaselineFringeFit(), _ThirdPartyStep()])
        @test br.solve_steps == [BaselineFringeFit(), _ThirdPartyStep()]
    end

    @testset "chaining builds a tuple" begin
        cf = StationWeightScale(DimArray([2.0], XRadio.AntennaName(["A1"])))
        chain = cf |> BaselineFringeFit() |> Bandpass()
        @test chain isa Tuple
        @test chain[1] === cf && chain[2] isa BaselineFringeFit && chain[3] isa Bandpass
        @test (cf |> (BaselineFringeFit(),))[1] === cf
        # A transform belongs to the steps after it.
        @test Gustavo._parse_pipeline(chain).before == [[cf], []]
        # Only pipeline elements chain; anything else is function application.
        @test_throws MethodError BaselineFringeFit() |> AverageFrequency(nout = 1)
    end

    @testset "full pipeline: components keyed by step" begin
        ps, _ = _build_fringe_ps()
        sol = fit(_full_chain(), ps; gauge = PinAntenna(1))

        @test sol isa CAL.CalibrationSolution
        @test collect(keys(sol.steps)) == [:fringe, :bandpass, :adhoc]
        @test unique(c.step for c in sol.components) == [:fringe, :bandpass, :adhoc]
        @test all(c -> c isa CAL.SolvedComponent, sol.components)
        @test sol.steps[:fringe] isa NamedTuple
        @test sol.provenance.gauge == "PinAntenna{Int64}(1)"
        @test_throws "holds no component under bogus" sol[:bogus]

        # The fringe step's unconstrained stations and search settings are its
        # own diagnostics.
        @test haskey(sol.steps[:fringe], :flagged_ant)
        @test sol.steps[:fringe].search isa Gustavo.Fring.FringeSearch
        @test !haskey(sol.info, :flagged_ant) && !haskey(sol.info, :search)

        # Gains factor multiplicatively over steps and over components.
        g_full = parent(gains(sol))
        @test parent(gains(sol[:fringe])) .* parent(gains(sol[:bandpass])) .* parent(gains(sol[:adhoc])) == g_full
        g_prod = ones(ComplexF64, size(g_full))
        for c in sol.components
            g_prod .*= parent(gains(filter(==(c), sol)))
        end
        @test g_prod ≈ g_full

        # Any selection applies. Calibrating step by step is calibrating by the
        # whole solution.
        whole = calibrate(sol, ps; apply_flags = false)
        stepwise = ps
        for k in (:fringe, :bandpass, :adhoc)
            stepwise = calibrate(sol[k], stepwise; apply_flags = false)
        end
        for (k, ms) in pairs(whole)
            @test isapprox(parent(stepwise[k][:visibility]), parent(ms[:visibility]); nans = true)
            @test parent(stepwise[k][:weight]) ≈ parent(ms[:weight])
        end
        @test calibrate(sol[:fringe, :phase, :mbd], ps) isa XRadio.ProcessingSet
        @test calibrate(filter(c -> c.component.term isa Delay, sol), ps) isa XRadio.ProcessingSet
        @test_throws "holds no components" calibrate(filter(_ -> false, sol), ps)
        @test_throws "holds no components" ApplySolution(filter(_ -> false, sol))
    end

    @testset "selections and gains selectors" begin
        ps, _ = _build_fringe_ps()
        sol = fit(_full_chain(), ps; gauge = PinAntenna(1))

        # A path prefix selects the components under it; their gains multiply.
        ph = sol[:fringe, :phase]
        @test !isempty(ph.components)
        @test collect(keys(ph.steps)) == [:fringe]
        g_group = ones(ComplexF64, size(gains(sol)))
        for c in ph.components
            g_group .*= parent(gains(sol[:fringe, c.path...]))
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

    @testset "flags come from the steps a selection holds" begin
        ps, _ = _build_fringe_ps()
        fitted = fit(_full_chain(), ps; gauge = PinAntenna(1))
        steps = copy(fitted.steps)
        steps[:fringe] = (; steps[:fringe]..., flagged_ant = ["A2"], flagged_scan = [1])
        sol = CAL.CalibrationSolution(fitted.geom, fitted.components, steps, fitted.info)
        @test Gustavo._solution_flag_sets(sol) == Set([("A2", 1)])
        @test Gustavo._solution_flag_sets(sol[:fringe, :phase, :mbd]) == Set([("A2", 1)])
        @test Gustavo._solution_flag_sets(sol[:bandpass]) === nothing
        flagged(out) = any(ms -> any(parent(ms[:flag])), values(out))
        @test flagged(calibrate(sol, ps)) && !flagged(calibrate(sol, ps; apply_flags = false))
        @test !flagged(calibrate(sol[:bandpass], ps))
    end

    @testset "calibrate: gains alone, or the fit's data path" begin
        ps, _ = _build_fringe_ps()
        ws = DimArray([1.0, 0.5, 1.0, 2.0], XRadio.AntennaName(["A1", "A2", "A3", "A4"]))
        sol = fit(StationWeightScale(ws) |> _full_chain(), ps; gauge = PinAntenna(1))
        @test startswith(sol.provenance.pipeline, "StationWeightScale")

        # The weight scale the fit applied is not replayed: scaling the data
        # before or after `calibrate` gives the same result.
        scale(set) = XRadio.ProcessingSet(
            OrderedDict{Symbol, XRadio.MeasurementSet}(
                k => StationWeightScale(ws)(read(ms)) for (k, ms) in pairs(set)
            ),
            copy(DimensionalData.metadata(set)),
        )
        before = calibrate(sol, scale(ps))
        after = scale(calibrate(sol, ps))
        @test collect(keys(before)) == collect(keys(after))
        for (k, ms) in pairs(after)
            @test isequal(parent(before[k][:visibility]), parent(ms[:visibility]))
            @test parent(before[k][:weight]) ≈ parent(ms[:weight])
        end

        # Given the pipeline, `calibrate` repeats the fit's data path, dividing
        # step by step rather than by the gains' product.
        replayed = calibrate(StationWeightScale(ws) |> _full_chain(), sol, ps)
        for (k, ms) in pairs(before)
            @test isapprox(parent(replayed[k][:visibility]), parent(ms[:visibility]); nans = true)
            @test parent(replayed[k][:weight]) ≈ parent(ms[:weight])
        end
        # Its solve steps must be the solution's.
        @test_throws "not the solution's steps" calibrate(StationWeightScale(ws) |> _full_chain(), sol[:fringe], ps)
        @test_throws "not the solution's steps" calibrate(BaselineFringeFit(), sol, ps)

        # Two weight-scale transforms compose (w·(s_a s_b)²). Two pipelines
        # join by splatting.
        sol_b = fit(((StationWeightScale(ws) |> StationWeightScale(ws))..., _full_chain()...), ps; gauge = PinAntenna(1))
        @test parent(gains(fit(StationWeightScale(ws .* ws) |> _full_chain(), ps; gauge = PinAntenna(1)))) ≈
            parent(gains(sol_b))

        # A transform listed after a step does not reach that step.
        early = fit(BaselineFringeFit(), ps; gauge = PinAntenna(1))
        late = fit(BaselineFringeFit() |> StationWeightScale(ws), ps; gauge = PinAntenna(1))
        @test parent(gains(late)) == parent(gains(early))

        # A correction between two steps is repeated in its place.
        between = BaselineFringeFit() |> StationWeightScale(ws) |> Bandpass()
        solb = fit(between, ps; gauge = PinAntenna(1))
        by_hand = calibrate(solb[:bandpass], scale(calibrate(solb[:fringe], ps; apply_flags = false)); apply_flags = false)
        for (k, ms) in pairs(calibrate(between, solb, ps; apply_flags = false))
            @test isapprox(parent(ms[:visibility]), parent(by_hand[k][:visibility]); nans = true)
            @test parent(ms[:weight]) ≈ parent(by_hand[k][:weight])
        end
    end

    @testset "rate components must share the constant-phase epoch" begin
        # Two scans: the per-scan rate's origins (scan centers) then cannot
        # all coincide with the track-wide rate's single origin.
        ps, _ = _build_fringe_ps(nscans = 2)
        # A second rate on a different time segmentation puts its origin in a
        # different place, so no single epoch zeroes both rate coordinates;
        # the fit rejects the model before its fringe pass reads any data.
        bad = merge(
            default_fringe_terms();
            phase = (; rate2 = GainComponent(Rate(); Ti = GlobalTime(), Feed = SingleFeed(2))),
        )
        @test_throws "disagree on the epoch" fit(BaselineFringeFit(model = bad), ps; gauge = PinAntenna(1))
    end

    @testset "solution Zarr round-trip" begin
        ps, _ = _build_fringe_ps()
        ws = DimArray([1.0, 0.5, 1.0, 1.0], XRadio.AntennaName(["A1", "A2", "A3", "A4"]))
        sol = fit(StationWeightScale(ws) |> _full_chain(), ps; gauge = PinAntenna(1))
        dir = mktempdir()
        path = joinpath(dir, "sol.zarr")
        CAL.save_solution(path, sol)
        back = CAL.load_solution(path)
        @test back.components == sol.components
        @test back.steps == sol.steps
        @test back.info == sol.info
        @test back.provenance == sol.provenance
        @test back.geom.stations == sol.geom.stations && back.geom.times == sol.geom.times
        @test parent(gains(back)) == parent(gains(sol))

        @test_throws "exists" CAL.save_solution(path, sol)
        @test_throws "not a Gustavo solution store" CAL.load_solution(joinpath(dir, "sol.zarr", "geometry"))

        # A diagnostic with no stored form fails the save and leaves nothing behind.
        bad = CalibrationSolution(sol.geom, sol.components, sol.steps, (; hook = x -> x))
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
