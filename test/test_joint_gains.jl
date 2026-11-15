# ── solve_joint_gains!: the joint solve over components' parameters ─────────
#
# Synthetic visibilities `V = g_a·S·conj(g_b)` with a random `S` per (scan,
# station pair, feed pair), which absorbs a constant per (station, feed) and
# leaves one phase per connected group of station gains free; the gains are
# compared with those freedoms removed.

@isdefined(CAL) || const CAL = Gustavo.Calibration
@isdefined(FP) || const FP = Gustavo.Fring

@testset "solve_joint_gains!" begin
    feeds = [(1, 1), (1, 2), (2, 1), (2, 2)]
    pairs_of(nant) = [(a, b) for a in 1:nant for b in (a + 1):nant]

    @testset "per-AP phase and log-amplitude" begin
        nant, nap = 4, 20
        names = ["A$i" for i in 1:nant]
        geom = CAL.DataGeometry(;
            nfeed = 2, times = collect(0.0:(nap - 1)), scan_of_time = ones(Int, nap),
            channel_freqs = [230.0e9], scan_names = ["No0001"], spw_names = ["A"], antennas = names,
        )
        comp = CAL.GainComponent(
            CAL.ConstantTerm(); Ti = CAL.PerIntegration(), Feed = CAL.SharedFeeds(), prior = CAL.OUPrior(; scale = 10.0, σ = 0.3),
        )
        l = CAL.plan_parameters(CAL.GainModel(; phase = (; adhoc = comp), logamp = (; adhoc = comp)), names, geom)
        θ = zeros(l.nθ)
        pb, ab = CAL.antenna_blocks(l, θ, :phase, :adhoc), CAL.antenna_blocks(l, θ, :logamp, :adhoc)
        rng = MersenneTwister(20261106)
        φ = cumsum(0.1 .* randn(rng, nant, nap); dims = 2)
        η = cumsum(0.03 .* randn(rng, nant, nap); dims = 2)
        bl = pairs_of(nant)
        S = [(1 + rand(rng)) * cis(2π * rand(rng)) for _ in bl, _ in feeds]
        ax = (FP.Scan(1:1), FP._antenna_pair_dim([(names[a], names[b]) for (a, b) in bl]), FP.FeedPair(feeds), Ti(1:nap))
        g = exp.(complex.(η, φ))
        w = fill(1.0e4, ax)
        r = zeros(ComplexF64, ax)
        for t in 1:nap, p in eachindex(feeds), (bi, (a, b)) in pairs(bl)
            r[1, bi, p, t] = w[1, bi, p, t] * g[a, t] * S[bi, p] * conj(g[b, t])
        end
        res = FP.solve_joint_gains!(
            (; phase = [pb], logamp = [ab]), (; r, w), geom;
            cover = [(t, 1) for _ in 1:1, t in 1:nap], gauge = PinAntenna(1), tolerance = 1.0e-8,
        )
        @test res.converged
        phase = [only(pb).θ[1, 1, 1, t, a] for a in 1:nant, t in 1:nap]
        logamp = [only(ab).θ[1, 1, 1, t, a] for a in 1:nant, t in 1:nap]
        # The reference's phase is zero on every AP; the rest are relative to it,
        # each up to the constant its `S` absorbs.
        @test maximum(abs, phase[1, :]) < 1.0e-12
        dφ = phase .- (φ .- φ[1:1, :])
        dη = logamp .- η
        @test maximum(abs, dφ .- mean(dφ; dims = 2)) < 1.0e-3
        @test maximum(abs, dη .- mean(dη; dims = 2)) < 1.0e-3
    end

    @testset "a bandpass shape beside a level and a delay" begin
        nant, nchan, nscan = 5, 32, 3
        names = ["A$i" for i in 1:nant]
        freqs = 230.0e9 .+ (0:(nchan - 1)) .* 2.0e6
        spw = [c <= nchan ÷ 2 ? 1 : 2 for c in 1:nchan]
        geom = CAL.DataGeometry(;
            nfeed = 2, times = 100.0 .* (1:nscan), scan_of_time = collect(1:nscan), channel_freqs = freqs,
            spw_of_chan = spw, scan_names = ["s$i" for i in 1:nscan], spw_names = ["A", "B"], antennas = names,
        )
        component(term; kw...) = CAL.GainComponent(term; Ti = CAL.GlobalTime(), Feed = CAL.PerFeed(), kw...)
        shape = component(
            CAL.ConstantTerm(); Frequency = CAL.ChannelBlocks(1),
            prior = CAL.OUPrior(; scale = LogNormal(log(10.0e6), 1.0), σ = LogNormal(log(0.1), 1.0)),
        )
        level = component(CAL.ConstantTerm(); Frequency = CAL.PerSpectralWindow(), prior = CAL.IIDPrior(1.0))
        delay = component(CAL.Delay(); prior = CAL.IIDPrior(1.0e-9))
        amp = component(CAL.ConstantTerm(); Frequency = CAL.ChannelBlocks(1), prior = CAL.OUPrior(; scale = 10.0e6, σ = 0.1))
        rng = MersenneTwister(3)
        lv, τ = 0.5 .* randn(rng, nant, 2, 2), 0.3e-9 .* randn(rng, nant, 2)
        φ = [lv[a, p, spw[c]] + 2π * (freqs[c] - geom.f0) * τ[a, p] + 0.2 * sin(c / 5 + a + p) for a in 1:nant, p in 1:2, c in 1:nchan]
        η = [0.05 * cos(c / 7 + a - p) for a in 1:nant, p in 1:2, c in 1:nchan]
        bl = pairs_of(nant)
        S = [(1 + rand(rng)) * cis(2π * rand(rng)) for _ in 1:nscan, _ in bl, _ in feeds]
        ax = (FP.Scan(1:nscan), FP._antenna_pair_dim([(names[a], names[b]) for (a, b) in bl]), FP.FeedPair(feeds), Frequency(1:nchan))
        g = exp.(complex.(η, φ))
        w = fill(1.0e4, ax)
        r = zeros(ComplexF64, ax)
        for si in 1:nscan, c in 1:nchan, (p, (fa, fb)) in pairs(feeds), (bi, (a, b)) in pairs(bl)
            r[si, bi, p, c] = w[si, bi, p, c] * g[a, fa, c] * S[si, bi, p] * conj(g[b, fb, c])
        end
        cover = [(si, c) for si in 1:nscan, c in 1:nchan]
        function solve(gauge)
            l = CAL.plan_parameters(CAL.GainModel(; phase = (; level, delay, shape), logamp = (; shape = amp)), names, geom)
            θ = zeros(l.nθ)
            comps = (;
                phase = [CAL.antenna_blocks(l, θ, :phase, k) for k in (:level, :delay, :shape)],
                logamp = [CAL.antenna_blocks(l, θ, :logamp, :shape)],
            )
            res = FP.solve_joint_gains!(comps, (; r, w), geom; cover, gauge, tolerance = 1.0e-6)
            sg = FP.StationGains(comps, nant, 2, cover)
            gains(obs) = sg.J[obs] * FP._param_values(comps[obs], sg.params[obs])
            xφ, xη = gains(:phase), gains(:logamp)
            at(x) = [x[sg.index[a, p, 1, c]] for a in 1:nant, p in 1:2, c in 1:nchan]
            return (; res, φ = at(xφ), η = at(xη))
        end
        # The data leave a phase per channel common to every station and feed,
        # and `S` a constant per (station, feed).
        free_removed(d) = (d = d .- mean(d; dims = (1, 2)); d .- mean(d; dims = 3))
        pinned = solve(PinAntenna(1))
        @test pinned.res.converged
        @test maximum(abs, pinned.φ[1, 1, :]) < 1.0e-9
        @test maximum(abs, free_removed(pinned.φ .- φ)) < 1.0e-3
        @test maximum(abs, (d = pinned.η .- η; d .- mean(d; dims = 3))) < 1.0e-3
        zero_sum = solve(ZeroSumPhase())
        @test maximum(abs, sum(zero_sum.φ; dims = (1, 2))) < 1.0e-9
        # The gauge moves only what the data do not see.
        @test free_removed(zero_sum.φ) ≈ free_removed(pinned.φ) atol = 1.0e-8
    end

    @testset "every fitted component takes a prior" begin
        names = ["A1", "A2"]
        geom = CAL.DataGeometry(;
            nfeed = 1, times = [0.0, 1.0], channel_freqs = [230.0e9, 231.0e9], scan_names = ["s1"],
            spw_names = ["A"], antennas = names,
        )
        ax = (FP.Scan(1:1), FP._antenna_pair_dim([("A1", "A2")]), FP.FeedPair([(1, 1)]), Frequency(1:2))
        data = (; r = ones(ComplexF64, ax), w = ones(ax))
        function solve(prior)
            c = CAL.GainComponent(CAL.ConstantTerm(); Ti = CAL.GlobalTime(), Frequency = CAL.ChannelBlocks(1), Feed = CAL.PerFeed(), prior)
            l = CAL.plan_parameters(CAL.GainModel(; phase = (; c)), names, geom)
            comps = (; phase = [CAL.antenna_blocks(l, zeros(l.nθ), :phase, :c)], logamp = [])
            return FP.solve_joint_gains!(comps, data, geom; cover = [(1, c) for _ in 1:1, c in 1:2], gauge = PinAntenna(1))
        end
        @test_throws "has no prior" solve(nothing)
        @test_throws "whose start is proper" solve(CAL.RandomWalkPrior(; σ = 0.1))
    end
end
