# Composable-pipeline interface tests (protocol, transforms, stage
# provenance/snapshots, and the fit/calibrate verbs), on ProcessingSets from
# `_build_fringe_ps`. Uses the CAL/FP/UVP aliases from test_pipeline.jl
# (included earlier in runtests.jl).

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

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
        info = stage_info(sol, :twopass)
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
        ps, _ = _build_fringe_ps(; nant, nspw, nchan, nscans = 3, bandpass = bp)
        sub = XRadio.query(ps; scan_name = "3")
        gauge = PinAntenna(1)
        fr = fit(BaselineFringeFit(), ps; gauge)
        # The scans' fringe phases differ, so the full-set solution corrects the
        # subset exactly as a fit on the subset alone does only if each of its
        # scans' gains lands on that scan.
        fr3 = fit(BaselineFringeFit(), sub; gauge)
        bp_sub = fit(ApplySolution(fr) |> Bandpass(), sub; gauge)
        @test bp_sub.steps[1].θ ≈ fit(ApplySolution(fr3) |> Bandpass(), sub; gauge).steps[1].θ atol = 1.0e-10
        # Noise-free and time-constant: one scan determines the bandpass all three do.
        @test maximum(abs, bp_sub.steps[1].θ .- fit(ApplySolution(fr) |> Bandpass(), ps; gauge).steps[1].θ) < 1.0e-6

        sol = fit(ApplySolution(fr) |> ApplySolution(bp_sub) |> AdhocPhase(), ps; gauge)
        @test sol.sequence[1] isa ApplySolution && sol.sequence[2] isa ApplySolution
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
        # provides(step) === :nothing never collides with itself.
        @test Gustavo._check_unique_provides(
            Gustavo.SolveStep[_ProtoProbe(), _ProtoProbe()]
        ) === nothing
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

    @testset "full pipeline: stage provenance and snapshots" begin
        ps, _ = _build_fringe_ps()
        sol = fit(_full_chain(), ps; gauge = PinAntenna(1))

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
        @test calibrate(fr, ps) isa XRadio.ProcessingSet
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
        ps, _ = _build_fringe_ps()
        sol = fit(_full_chain(), ps; gauge = PinAntenna(1))

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
        @test calibrate(csel, ps) isa XRadio.ProcessingSet
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
        ps, _ = _build_fringe_ps()
        ws = DimArray([1.0, 0.5, 1.0, 2.0], XRadio.AntennaName(["A1", "A2", "A3", "A4"]))
        sol = fit(StationWeightScale(ws) |> _full_chain(), ps; gauge = PinAntenna(1))
        @test length(recorded_transforms(sol)) == 1
        @test recorded_transforms(sol)[1] isa StationWeightScale
        @test recorded_transforms(sol)[1].scale == ws

        # Replaying the recorded weight scale is the same as scaling the data
        # first and calibrating with a solution that records no correction.
        scaled = XRadio.ProcessingSet(
            OrderedDict{Symbol, XRadio.MeasurementSet}(
                k => StationWeightScale(ws)(read(ms)) for (k, ms) in pairs(ps)
            ),
            copy(DimensionalData.metadata(ps)),
        )
        bare = CAL.CalibrationSolution(
            sol.steps, sol.geom, sol.info; sequence = sol.sequence[2:end], sol.gauge,
        )
        out = calibrate(sol, ps)
        ref = calibrate(bare, scaled)
        @test collect(keys(out)) == collect(keys(ref))
        for (k, ms) in pairs(ref)
            @test isequal(parent(out[k][:visibility]), parent(ms[:visibility]))
            @test parent(out[k][:weight]) == parent(ms[:weight])
        end

        # Two weight-scale transforms compose (w·(s_a s_b)²), and both are
        # recorded. Two pipelines join by splatting.
        sol_b = fit(((StationWeightScale(ws) |> StationWeightScale(ws))..., _full_chain()...), ps; gauge = PinAntenna(1))
        @test length(recorded_transforms(sol_b)) == 2
        @test parent(gains(fit(StationWeightScale(ws .* ws) |> _full_chain(), ps; gauge = PinAntenna(1)))) ≈
            parent(gains(sol_b))

        # A transform listed after a step does not reach that step.
        early = fit(BaselineFringeFit(), ps; gauge = PinAntenna(1))
        late = fit(BaselineFringeFit() |> StationWeightScale(ws), ps; gauge = PinAntenna(1))
        @test parent(gains(late)) == parent(gains(early))
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

    @testset "solution serialization round-trip; older files refused" begin
        ps, _ = _build_fringe_ps()
        ws = DimArray([1.0, 0.5, 1.0, 1.0], XRadio.AntennaName(["A1", "A2", "A3", "A4"]))
        sol = fit(StationWeightScale(ws) |> _full_chain(), ps; gauge = PinAntenna(1))
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
        @test_throws "did not survive" calibrate(solm, ps)

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
