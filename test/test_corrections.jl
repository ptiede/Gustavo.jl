# ── Corrections and `calibrate` on Measurement Sets ──────────────────────────
#
# Each built-in correction against a direct computation, on a hand-built
# solution, so these tests do not depend on any solver.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

const CALc = Gustavo.Calibration

_halve_weights(ms) = (parent(ms[:weight]) ./= 2; ms)

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
        # window by frequency: the same correction the whole set receives.
        whole = calibrate(sol, ps)
        for (k, m) in pairs(ps)
            @test isequal(
                parent(calibrate!(sol, deepcopy(read(m)))[:visibility]),
                parent(whole[k][:visibility]),
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

    @testset "scale_weights!" begin
        scale = DimArray([2.0, 5.0], AntennaName(["A2", "ZZ"]))
        target = fresh()
        out = scale_weights!(target, scale)
        @test out === target
        for (bi, (a, b)) in pairs(collect(XRadio.baselines(ms)))
            f = ("A2" in (a, b)) ? 2.0 : 1.0
            @test parent(out[:weight][BaselineID = bi]) ≈ f .* parent(ms[:weight][BaselineID = bi])
        end
        @test parent(out[:visibility]) == parent(ms[:visibility])
        @test_throws "scale_weights!: every factor must be finite and positive" scale_weights!(
            fresh(), DimArray([1.0, 0.0], AntennaName(["A1", "A2"]))
        )
        @test_throws "scale_weights!: a station is named more than once" scale_weights!(
            fresh(), DimArray([1.0, 2.0], AntennaName(["A1", "A1"]))
        )
        @test_throws "scale_weights!: index the factors by `AntennaName`" scale_weights!(
            fresh(), DimArray([1.0], XRadio.StationName(["A1"]))
        )
    end

    @testset "flag_channels!" begin
        freqs = geom.channel_freqs
        hit = freqs[[1, end]]
        mask = DimArray(map(in(hit), freqs), Frequency(freqs))
        for m in values(ps)
            one = deepcopy(read(m))
            out = flag_channels!(one, mask)
            @test out === one
            for (c, f) in enumerate(XRadio.frequencies(one))
                @test all(parent(out[:flag][Frequency = c])) == (f in hit)
            end
        end
        partial = DimArray(trues(2), Frequency(freqs[1:2]))
        @test_throws "flag_channels!: the mask does not cover the channel" flag_channels!(fresh(), partial)
        @test_throws "flag_channels!: index the mask by `Frequency`" flag_channels!(
            fresh(), DimArray(trues(2), Ti([1.0, 2.0]))
        )
        @test_throws "flag_channels!: a frequency appears more than once" flag_channels!(
            fresh(), DimArray(trues(2), Frequency([1.0, 1.0]))
        )
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
            @test _division_error(out[name], read(lazy), sol, geom) < 1.0e-6
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

    @testset "post returns the Measurement Set it modified" begin
        @test_throws "returned a different MeasurementSet" calibrate(sol, ps; post = deepcopy)
        @test_throws "returned a Nothing" calibrate(sol, ps; post = m -> nothing)
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
