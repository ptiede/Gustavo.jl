# ── StationGains: the station gains of a joint solve and their Jacobian ─────

@isdefined(CAL) || const CAL = Gustavo.Calibration
@isdefined(FP) || const FP = Gustavo.Fring

@testset "StationGains" begin
    names = ["A1", "A2", "A3"]
    nchan = 6
    freqs = 230.0e9 .+ (0:(nchan - 1)) .* 1.0e6
    geom = CAL.DataGeometry(;
        nfeed = 2, times = [0.0, 1.0, 10.0, 11.0], scan_of_time = [1, 1, 2, 2],
        channel_freqs = freqs, spw_of_chan = [1, 1, 1, 2, 2, 2],
        scan_names = ["s1", "s2"], spw_names = ["A", "B"], antennas = names,
    )
    cover = [(t, c) for t in (1, 3), c in 1:nchan]
    component(term; kw...) = CAL.GainComponent(term; Ti = CAL.GlobalTime(), Feed = CAL.PerFeed(), kw...)
    shape = component(CAL.ConstantTerm(); Frequency = CAL.ChannelBlocks(1))
    level = component(CAL.ConstantTerm(); Frequency = CAL.PerSpectralWindow())
    delay = component(CAL.Delay())
    rng = MersenneTwister(1)

    # The station-gain values at each station gain, summed from the model layer.
    function values(sg, comps, obs, v)
        cs, ps = comps[obs], sg.params[obs]
        locs = [FP._block_locations(c, length(names)) for c in cs]
        return FP._station_gain_values!(zeros(length(sg.at)), cs, FP._leaves_with(cs, ps, v), locs, sg.at)
    end

    @testset "a shape, a level and a delay" begin
        l = CAL.plan_parameters(CAL.GainModel(; phase = (; level, delay, shape), logamp = (; shape)), names, geom)
        θ = randn(rng, l.nθ)
        comps = (;
            phase = [CAL.antenna_blocks(l, θ, :phase, k) for k in (:level, :delay, :shape)],
            logamp = [CAL.antenna_blocks(l, θ, :logamp, :shape)],
        )
        sg = FP.StationGains(comps, length(names), 2, cover)
        # GlobalTime: the two scans share every station gain; PerFeed: the
        # feeds do not.
        @test length(sg.at) == length(names) * 2 * nchan
        @test sg.index[:, :, 1, :] == sg.index[:, :, 2, :]
        @test allunique(sg.index[:, :, 1, :])
        # Each phase station gain reads one level, one delay and one shape
        # value; each log-amplitude one shape value.
        nread(J) = [count(!iszero, view(J, n, :)) for n in axes(J, 1)]
        @test all(==(3), nread(sg.J.phase))
        @test all(==(1), nread(sg.J.logamp))
        n = sg.index[2, 1, 1, 5]
        delay_col = only(i for (i, (k, _, _)) in pairs(sg.params.phase) if k == 2 && !iszero(sg.J.phase[n, i]))
        @test sg.J.phase[n, delay_col] ≈ 2π * (freqs[5] - geom.f0)
        # The terms are linear, so the values are `J` times the parameters.
        for obs in (:phase, :logamp)
            v = FP._param_values(comps[obs], sg.params[obs])
            @test values(sg, comps, obs, v) ≈ sg.J[obs] * v
        end
    end

    @testset "shared feeds share a station gain" begin
        shared = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.ChannelBlocks(1), Feed = CAL.SharedFeeds())
        l = CAL.plan_parameters(CAL.GainModel(; phase = (; shared)), names, geom)
        θ = randn(rng, l.nθ)
        comps = (; phase = [CAL.antenna_blocks(l, θ, :phase, :shared)], logamp = [])
        sg = FP.StationGains(comps, length(names), 2, cover)
        @test sg.index[:, 1, :, :] == sg.index[:, 2, :, :]
        @test length(sg.at) == length(names) * nchan
        @test size(sg.J.logamp) == (length(sg.at), 0)
    end

    @testset "a rate splits the station gains by time" begin
        rate = component(CAL.Rate())
        l = CAL.plan_parameters(CAL.GainModel(; phase = (; shape, rate)), names, geom)
        θ = randn(rng, l.nθ)
        comps = (; phase = [CAL.antenna_blocks(l, θ, :phase, k) for k in (:shape, :rate)], logamp = [])
        sg = FP.StationGains(comps, length(names), 2, cover)
        @test length(sg.at) == 2 * length(names) * 2 * nchan
        v = FP._param_values(comps.phase, sg.params.phase)
        @test values(sg, comps, :phase, v) ≈ sg.J.phase * v
    end
end
