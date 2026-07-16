# Gustavo.Solve — reparameterization: gauge fixing + parameter scaling
# (Stage 2, Milestone 5b). Pure machinery (no optimization / Enzyme): the free /
# fixed split, the y↔p maps, AutoScale via param_scale, and the user-specifiable
# CustomScale / FixParams / NoGauge / NoScaling knobs.

using Gustavo.Solve
using Gustavo.Solve: build_reparam, to_full, to_free, nfree, param_scale,
    ReferenceAntenna, NoGauge, FixParams, AutoScale, NoScaling, CustomScale,
    plan_gains, zero_params, flatten
import Gustavo.Calibration as CAL
using Gustavo.Calibration:
    DataGeometry, StationGainModel, GainComponent, TiedComponent,
    Delay, ConstantTerm, PerChannel, PerScan, GlobalTime, GlobalFrequency,
    PerSpectralWindow, PerFeed
using Random: MersenneTwister
using Test

@testset "Solve M5b: gauge fixing + parameter scaling" begin
    geom = DataGeometry(
        times = [0.0, 0.1, 1.0, 1.1],
        scan_of_time = [1, 1, 2, 2],
        channel_freqs = [1.0e9, 1.1e9, 2.0e9, 2.1e9],
        spw_of_chan = [1, 1, 2, 2],
    )
    nant = 3
    model = StationGainModel(
        phase = (
            clock = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            fringe = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), PerFeed()),
        ),
        logamp = (
            bandpass = TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
    )
    plan = plan_gains(model, nant, geom)
    nθ = length(plan.template)

    @testset "AutoScale via param_scale" begin
        cdelay = plan.groups[1].phase[1]       # Delay SiteComponent
        cconst = plan.groups[1].phase[2]       # ConstantTerm
        dfmax = maximum(abs, geom.channel_freqs .- geom.f0)
        @test param_scale(Delay(), cdelay) ≈ 1 / (2π * dfmax)
        @test param_scale(ConstantTerm(), cconst) ≈ 1.0            # unit scale
        @test param_scale(PerChannel(), plan.groups[1].logamp[1]) ≈ 1.0
    end

    @testset "ReferenceAntenna gauge: free/fixed split + round-trip" begin
        rp = build_reparam(plan, ReferenceAntenna(1), AutoScale(), zero_params(plan))
        # Refant (ant 1) phase + amp blocks are fixed → removed from the free set.
        # Each component fixes nparam·ntseg·nfseg·nfb entries for the one refant.
        nfixed = sum(c.nparam * c.ntseg * c.nfseg * c.nfb
            for c in (plan.groups[1].phase..., plan.groups[1].logamp...))
        @test nfree(rp) == nθ - nfixed
        # to_full ∘ to_free is identity on a random full p (up to the pinned zeros).
        rng = MersenneTwister(7)
        p = zero_params(plan)
        p .= randn(rng, nθ)
        p.g1.clock[:, :, :, :, 1] .= 0        # match the gauge (refant pinned)
        p.g1.fringe[:, :, :, :, 1] .= 0
        p.g1.bandpass[:, :, :, :, 1] .= 0
        y = to_free(rp, p)
        @test length(y) == nfree(rp)
        @test flatten(to_full(rp, y)) ≈ flatten(p)
        # A fresh y always maps to a p whose refant blocks are exactly 0.
        pf = to_full(rp, randn(rng, nfree(rp)))
        @test all(iszero, pf.g1.clock[:, :, :, :, 1])
        @test all(iszero, pf.g1.bandpass[:, :, :, :, 1])
        # Scaling actually rescales: a unit y on a delay slot is ~1/(2π Δf) in p.
        y1 = zeros(nfree(rp))
        y1[1] = 1.0
        @test maximum(abs, flatten(to_full(rp, y1))) ≈ param_scale(Delay(), plan.groups[1].phase[1])
    end

    @testset "user-specifiable knobs" begin
        # NoGauge: nothing fixed. NoScaling: unit scale.
        rp0 = build_reparam(plan, NoGauge(), NoScaling(), zero_params(plan))
        @test nfree(rp0) == nθ
        @test all(rp0.scale .== 1)
        # phase-only refant gauge leaves the amp blocks free.
        rpp = build_reparam(plan, ReferenceAntenna(1; amp = false), AutoScale(), zero_params(plan))
        @test nfree(rpp) > nfree(build_reparam(plan, ReferenceAntenna(1), AutoScale(), zero_params(plan)))
        # CustomScale overrides a named component; others keep AutoScale.
        rpc = build_reparam(plan, NoGauge(), CustomScale((clock = 2.5,)), zero_params(plan))
        sc = zeros(nθ)
        sc[1] = 1.0
        # the delay ('clock') scale is now the custom 2.5, not the auto value
        yc = zeros(nθ)
        pc = zero_params(plan)
        pc.g1.clock[1, 1, 1, 1, 1] = 1.0        # a single clock parameter
        yy = to_free(rpc, pc)
        @test yy[findfirst(!=(0), yy)] ≈ 1.0 / 2.5
        # FixParams: an explicit index set is held at 0.
        rpx = build_reparam(plan, FixParams([1, 2, 3]), NoScaling(), zero_params(plan))
        @test nfree(rpx) == nθ - 3
        px = to_full(rpx, ones(nfree(rpx)))
        @test all(iszero, flatten(px)[1:3])
    end
end
