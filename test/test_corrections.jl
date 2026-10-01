# ── Corrections and `calibrate` on Measurement Sets ──────────────────────────
#
# Each built-in correction against a direct computation, on a hand-built
# solution, so these tests do not depend on any solver.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")
@isdefined(_autocorrelated_ms) || include("test_autocorrelations.jl")
@isdefined(GroupProbe) || include("test_each_group.jl")

const CALc = Gustavo.Calibration

# A solution over `geom` with a per-(scan, window) phase and a per-channel
# amplitude, each per feed, set to seeded random values.
function _hand_solution(geom; info = (;), seed = 7)
    model = CALc.GainModel(;
        phase = (; ph = CALc.GainComponent(CALc.ConstantTerm(); Ti = CALc.PerScan(), Frequency = CALc.PerSpectralWindow(), Feed = CALc.PerFeed())),
        logamp = (; la = CALc.GainComponent(CALc.ConstantTerm(); Ti = CALc.GlobalTime(), Frequency = CALc.ChannelBlocks(1), Feed = CALc.PerFeed())),
    )
    layout = CALc.plan_parameters(model, geom.stations, geom)
    θ = 0.3 .* randn(StableRNG(seed), layout.nθ)
    return CALc.CalibrationSolution(
        model, layout, geom, θ, info; name = :hand,
    )
end

_with_stations(geom, stations) = CALc.DataGeometry(;
    geom.times, geom.channel_freqs, geom.scan_of_time, geom.spw_of_chan, geom.channel_widths, geom.t0, geom.f0,
    geom.scan_names, geom.spw_names, stations,
)

# The largest relative error of `out` against `ms` divided by `sol`'s gains,
# over every cell, visibilities and weights.
function _division_error(out, ms, sol, geom)
    win = CALc.GeometryWindow(geom, ms)
    g = parent(CALc.gains(sol, win))
    feeds = Gustavo.UVData.feed_pairs(ms)
    worst = 0.0
    for p in axes(feeds, 1), bi in axes(feeds, 2), c in eachindex(win.chan_idx), t in eachindex(win.ti_idx)
        a, b = win.stations[bi]
        fa, fb = feeds[p, bi]
        den = g[c, t, a, fa] * conj(g[c, t, b, fb])
        cell = (Polarization(p), BaselineID(bi), Frequency(c), Ti(t))
        v = ms[:visibility][cell...] / den
        w = ms[:weight][cell...] * abs2(den)
        worst = max(worst, abs(out[:visibility][cell...] - v) / abs(v), abs(out[:weight][cell...] - w) / w)
    end
    return worst
end

@testset "corrections" begin
    ps, _ = _build_fringe_ps(; nscans = 2, nspw = 2)
    geom = CALc.DataGeometry(ps)
    sol = _hand_solution(geom)
    ms = read(first(ps))
    fresh() = deepcopy(ms)

    @testset "calibrate! divides out the gains in place" begin
        target = fresh()
        out = calibrate!(sol, target; apply_flags = false)
        @test out === target
        @test _division_error(out, ms, sol, geom) < 1.0e-6
        @test parent(out[:flag]) == parent(ms[:flag])
    end

    @testset "calibrate! on one window matches the run's geometry" begin
        # A per-channel segmentation of the whole set places each channel of one
        # window by frequency: the same correction a pipeline applies.
        for m in values(ps)
            one = read(m)
            @test isequal(
                parent(calibrate!(sol, deepcopy(one))[:visibility]),
                parent(Gustavo._correct(calibrate!(sol), deepcopy(one), geom)[:visibility]),
            )
        end
        # A solution segmented only by name applies with the window's own geometry.
        own = CALc.DataGeometry(Gustavo._one_member(ms))
        onewin = _hand_solution(own)
        @test _division_error(calibrate!(onewin, fresh()), ms, onewin, own) < 1.0e-6
    end

    @testset "calibrate! matches stations by name" begin
        @test_throws "must name its stations" CALc.CalibrationSolution(_with_stations(geom, String[]), sol.components)
        strangers = _hand_solution(_with_stations(geom, ["X1", "X2", "X3", "X4"]))
        @test_throws "shares no station" calibrate!(strangers, fresh())
        # A station the solution lacks keeps identity gains.
        partial = _hand_solution(_with_stations(geom, ["A1", "A2", "A3", "X4"]))
        out = @test_logs (:warn, r"not in the solution") calibrate!(partial, fresh())
        a4 = findall(p -> "A4" in p, collect(XRadio.baselines(ms)))
        @test parent(out[:visibility][BaselineID = a4]) == parent(ms[:visibility][BaselineID = a4])
    end

    @testset "calibrate!(sol) is a correction that composes" begin
        g = calibrate!(sol; apply_flags = false)
        @test g isa GainCorrection
        @test sprint(show, g) == "calibrate!(hand; apply_flags = false)"
        ref = calibrate!(sol, fresh(); apply_flags = false)
        @test isequal(parent(g(fresh())[:visibility]), parent(ref[:visibility]))
        twice = (g ∘ g)(fresh())
        @test isequal(parent(twice[:visibility]), parent(g(ref)[:visibility]))
        @test g |> GroupProbe() isa Tuple{GainCorrection, GroupProbe}
        @test g |> g isa Tuple{GainCorrection, GainCorrection}
        @test startswith(fit(g |> GroupProbe(), ps).provenance.pipeline, "calibrate!(hand; apply_flags = false) |> ")
    end

    @testset "StationWeightScale" begin
        scale = DimArray([2.0, 5.0], AntennaName(["A2", "ZZ"]))
        target = fresh()
        out = StationWeightScale(scale)(target)
        @test out === target
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
            f = ("A2" in (a, b)) ? 2.0 : 1.0
            @test parent(out[:weight][BaselineID = bi]) ≈ f .* parent(ms[:weight][BaselineID = bi])
        end
        @test parent(out[:visibility]) == parent(ms[:visibility])
        # In a pipeline it is the same correction.
        @test parent(Gustavo._correct(StationWeightScale(scale), fresh(), geom)[:weight]) == parent(out[:weight])
        @test_throws "finite and positive" StationWeightScale(DimArray([1.0, 0.0], AntennaName(["A1", "A2"])))
        @test_throws "named more than once" StationWeightScale(DimArray([1.0, 2.0], AntennaName(["A1", "A1"])))
        @test_throws "index the factors by `AntennaName`" StationWeightScale(DimArray([1.0], XRadio.StationName(["A1"])))
    end

    @testset "FlagChannels" begin
        freqs = geom.channel_freqs
        hit = freqs[[1, end]]
        mask = DimArray(map(in(hit), freqs), Frequency(freqs))
        for m in values(ps)
            one = deepcopy(read(m))
            out = FlagChannels(mask)(one)
            @test out === one
            for (c, f) in enumerate(XRadio.frequencies(one))
                @test all(parent(out[:flag][Frequency = c])) == (f in hit)
            end
        end
        @test sprint(show, FlagChannels(mask)) == "FlagChannels(2 of $(length(freqs)) channels)"
        partial = DimArray(trues(2), Frequency(freqs[1:2]))
        @test_throws "does not cover the channel" FlagChannels(partial)(fresh())
        @test_throws "index the mask by `Frequency`" FlagChannels(DimArray(trues(2), Ti([1.0, 2.0])))
        @test_throws "more than once" FlagChannels(DimArray(trues(2), Frequency([1.0, 1.0])))
    end

    @testset "AutocorrelationNormalization" begin
        auto = _set_autocorrelations!(_autocorrelated_ms())
        ref = Gustavo.UVData.normalize_by_autocorrelations(auto)
        target = deepcopy(auto)
        @test AutocorrelationNormalization()(target) === target
        @test isequal(parent(target[:visibility]), parent(ref[:visibility]))
    end

    @testset "a correction returns the Measurement Set it modified" begin
        copying(m) = deepcopy(m)
        @test_throws "returned a different MeasurementSet" fit((copying, GroupProbe()), ps)
        @test_throws "returned a Nothing" fit((m -> nothing, GroupProbe()), ps)
    end

    @testset "fit hands every correction one private copy of the group" begin
        before = deepcopy(ps)
        seen = []
        record(m) = (push!(seen, parent(m[:visibility])); m)
        serial = ExecutionConfig(; inner_executor = SerialScheduler())
        _probe((record, _halve_weights, record) |> GroupProbe(), ps; exec = serial)
        # Both records of a member see the same array, never the caller's.
        @test all(seen[k] === seen[k + 1] for k in 1:2:length(seen))
        callers = [parent(m[:visibility]) for m in values(ps)]
        @test !any(v -> any(c -> c === v, callers), seen)
        for k in keys(ps)
            @test isequal(parent(ps[k][:visibility]), parent(before[k][:visibility]))
            @test parent(ps[k][:weight]) == parent(before[k][:weight])
        end
    end
end

@testset "calibrate" begin
    ps, _ = _build_fringe_ps(; nscans = 2, nspw = 2)
    geom = CALc.DataGeometry(ps)
    sol = _hand_solution(geom; info = (; flagged_ant = ["A2"], flagged_scan = [1]))

    @testset "divides by the gains only" begin
        out = calibrate(sol, ps; apply_flags = false)
        @test collect(keys(out)) == collect(keys(ps))
        for (name, lazy) in pairs(ps)
            ref = Gustavo._correct(calibrate!(sol; apply_flags = false), deepcopy(read(lazy)), geom)
            @test isequal(parent(out[name][:visibility]), parent(ref[:visibility]))
            @test parent(out[name][:weight]) == parent(ref[:weight])
            @test !any(parent(out[name][:flag]))
        end
    end

    @testset "flags the baselines of an unconstrained station, on that scan only" begin
        out = calibrate(sol, ps)
        for ms in values(out), (bi, pair) in pairs(collect(XRadio.baselines(ms)))
            on_scan1 = ms[:scan_name] .== "1"
            fl = ms[:flag][BaselineID = bi]
            for (t, s1) in pairs(on_scan1)
                @test all(parent(fl[Ti = t])) == ("A2" in pair && s1)
            end
        end
    end

    @testset "post runs on each corrected Measurement Set" begin
        out = calibrate(sol, ps; post = _halve_weights, apply_flags = false)
        ref = calibrate(sol, ps; apply_flags = false)
        @test all(parent(out[k][:weight]) == parent(ref[k][:weight]) ./ 2 for k in keys(ps))
    end

    @testset "leaves its input as it was" begin
        before = deepcopy(ps)
        calibrate(sol, ps)
        @test all(isequal(parent(ps[k][:visibility]), parent(before[k][:visibility])) for k in keys(ps))
        @test all(parent(ps[k][:flag]) == parent(before[k][:flag]) for k in keys(ps))
    end

    @testset "a degenerate gain flags the sample" begin
        bad = deepcopy(sol)
        foreach(c -> parent(c.params) .= -1.0e4, bad.components)    # gain amplitude exp(-1e4) underflows to zero
        out = calibrate(bad, ps; apply_flags = false)
        @test all(parent(first(out)[:flag]))
        @test all(isnan, parent(first(out)[:visibility]))
    end

    @testset "refuses an empty solution" begin
        @test_throws "holds no components" calibrate(filter(_ -> false, sol), ps)
    end
end
