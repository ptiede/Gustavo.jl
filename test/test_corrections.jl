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
function _hand_solution(geom; info = (;), sequence = (), seed = 7)
    model = CALc.GainModel(;
        phase = (; ph = CALc.GainComponent(CALc.ConstantTerm(); Ti = CALc.PerScan(), Frequency = CALc.PerSpectralWindow(), Feed = CALc.PerFeed())),
        logamp = (; la = CALc.GainComponent(CALc.ConstantTerm(); Ti = CALc.GlobalTime(), Frequency = CALc.ChannelBlocks(1), Feed = CALc.PerFeed())),
    )
    layout = CALc.plan_parameters(model, geom.stations, geom)
    θ = 0.3 .* randn(StableRNG(seed), layout.nθ)
    return CALc.CalibrationSolution(
        model, layout, geom, θ, (; ant_names = geom.stations, info...); name = :hand, sequence,
    )
end

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

    @testset "ApplySolution divides out the gains" begin
        out = Gustavo._correct(ApplySolution(sol), ms, geom)
        @test _division_error(out, ms, sol, geom) < 1.0e-6
        @test parent(out[:flag]) == parent(ms[:flag])
        # The input is not modified.
        @test isequal(parent(ms[:visibility]), parent(read(first(ps))[:visibility]))
    end

    @testset "ApplySolution on a Measurement Set alone" begin
        # A channel-index segmentation needs the whole set's channel axis, which
        # one window does not carry.
        @test_throws "identical channel layout" ApplySolution(sol)(ms)
        # A solution segmented only by name applies with the window's own geometry.
        onewin = _hand_solution(CALc.DataGeometry(Gustavo._one_member(ms)))
        own = CALc.DataGeometry(Gustavo._one_member(ms))
        @test _division_error(ApplySolution(onewin)(ms), ms, onewin, own) < 1.0e-6
    end

    @testset "ApplySolution matches stations by name" begin
        anon = CALc.CalibrationSolution(sol.steps, sol.geom, (;))
        @test_throws "records no station names" Gustavo._correct(ApplySolution(anon), ms, geom)
        strangers = CALc.CalibrationSolution(sol.steps, sol.geom, (; ant_names = ["X1", "X2", "X3", "X4"]))
        @test_throws "shares no station" Gustavo._correct(ApplySolution(strangers), ms, geom)
        # A station the solution lacks keeps identity gains.
        partial = CALc.CalibrationSolution(sol.steps, sol.geom, (; ant_names = ["A1", "A2", "A3", "X4"]))
        out = @test_logs (:warn, r"not in the solution") Gustavo._correct(ApplySolution(partial), ms, geom)
        a4 = findall(p -> "A4" in p, collect(XRadio.baselines(ms)))
        @test parent(out[:visibility][BaselineID = a4]) == parent(ms[:visibility][BaselineID = a4])
    end

    @testset "StationWeightScale" begin
        scale = DimArray([2.0, 5.0], XRadio.AntennaName(["A2", "ZZ"]))
        out = StationWeightScale(scale)(ms)
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
            f = ("A2" in (a, b)) ? 2.0 : 1.0
            @test parent(out[:weight][BaselineID = bi]) ≈ f .* parent(ms[:weight][BaselineID = bi])
        end
        @test parent(out[:visibility]) == parent(ms[:visibility])
        # In a pipeline it is the same correction.
        @test parent(Gustavo._correct(StationWeightScale(scale), ms, geom)[:weight]) == parent(out[:weight])
        @test_throws "finite and positive" StationWeightScale(DimArray([1.0, 0.0], XRadio.AntennaName(["A1", "A2"])))
        @test_throws "named more than once" StationWeightScale(DimArray([1.0, 2.0], XRadio.AntennaName(["A1", "A1"])))
        @test_throws "index the factors by `AntennaName`" StationWeightScale(DimArray([1.0], Gustavo.Ant(["A1"])))
    end

    @testset "FlagChannels" begin
        mask = falses(length(geom.channel_freqs))
        mask[[1, length(mask)]] .= true
        out = Gustavo._correct(FlagChannels(mask), ms, geom)
        flagged = mask[CALc.GeometryWindow(geom, ms).chan_idx]
        for c in eachindex(flagged)
            @test all(parent(out[:flag][Frequency = c])) == flagged[c]
        end
        @test_throws DimensionMismatch Gustavo._correct(FlagChannels(falses(3)), ms, geom)
    end

    @testset "AutocorrelationNormalization" begin
        auto = _set_autocorrelations!(_autocorrelated_ms())
        @test isequal(
            parent(AutocorrelationNormalization()(auto)[:visibility]),
            parent(Gustavo.UVData.normalize_by_autocorrelations(auto)[:visibility]),
        )
    end
end

@testset "calibrate" begin
    ps, _ = _build_fringe_ps(; nscans = 2, nspw = 2)
    geom = CALc.DataGeometry(ps)
    sol = _hand_solution(geom; info = (; flagged_ant = [2], flagged_scan = [1]), sequence = (_halve_weights, GroupProbe()))

    @testset "replays the sequence in order, then flags" begin
        out = calibrate(sol, ps; apply_flags = false)
        @test collect(keys(out)) == collect(keys(ps))
        for (name, lazy) in pairs(ps)
            ref = Gustavo._correct(ApplySolution(sol), _halve_weights(read(lazy)), geom)
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

    @testset "a degenerate gain flags the sample" begin
        bad = deepcopy(sol)
        bad.steps[1].θ .= -1.0e4    # gain amplitude exp(-1e4) underflows to zero
        out = calibrate(bad, ps; apply_flags = false)
        @test all(parent(first(out)[:flag]))
        @test all(isnan, parent(first(out)[:visibility]))
    end

    @testset "refuses what it cannot replay" begin
        lost = CALc.CalibrationSolution(sol.steps, sol.geom, sol.info; sequence = (missing, GroupProbe()))
        @test_throws "did not survive serialization" calibrate(lost, ps)
        short = CALc.CalibrationSolution(sol.steps, sol.geom, sol.info; sequence = (_halve_weights,))
        @test_throws "sequence lists 0" calibrate(short, ps)
    end
end
