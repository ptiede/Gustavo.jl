# Gustavo.Solve — per-site 2D-array parameter container + grouped forward map
# (Stage 2, Milestone 2).
#
# `plan_gains` groups antennas by identical model structure and lays out their
# parameters as per-site arrays inside a ComponentVector; the generalized
# `evaluate_gains(plan, p)` must (1) be numerically identical to the existing
# flat-θ `GainEvaluator` path, (2) stay `@inferred` for both homogeneous and
# heterogeneous (multi-structure) arrays, and (3) round-trip through
# flatten/unflatten.

using Gustavo.Solve
using Gustavo.Solve: nparameters, flatten, unflatten
import Gustavo.Calibration as CAL
using Gustavo.Calibration:
    DataGeometry, StationGainModel, GainComponent, TiedComponent, GainEvaluator,
    ConstantTerm, Delay, Dispersion, PerChannel,
    PerScan, GlobalTime, GlobalFrequency, PerSpectralWindow,
    PerFeed, SharedFeeds, ReferenceRelative, FeedComponent
using Random: MersenneTwister
using Test

# Copy a flat θ (for the shared `model`/`ev`) into the matching per-site arrays of
# `p`, so both representations encode the SAME gains. Walks each component's old
# `off1`/`off2` offset tables in lockstep with the new feed-block routing — both
# derive from the same tying, so they align by construction. (Homogeneous only:
# one group, local antenna index == global.)
function _fill_matching!(p, plan, ev, θ)
    lay = ev.layout
    parr, aarr = group_arrays(plan, p, 1)
    gp = plan.groups[1]
    for (j, c) in enumerate(gp.phase)
        _copy_component!(parr[j], c, lay.plans[j], θ, plan.nant)
    end
    for (j, c) in enumerate(gp.logamp)
        _copy_component!(aarr[j], c, lay.plans[lay.nphase + j], θ, plan.nant)
    end
    return p
end
function _copy_component!(A, c::SiteComponent, oldplan, θ, nant)
    bl = c.nparam                       # block length (uniform — no ragged padding here)
    for ant in 1:nant, feed in 1:2, ts in 1:c.ntseg, fs in 1:c.nfseg
        fb1 = c.fb1[feed]
        if fb1 != 0
            o = oldplan.off1[ant, feed, ts, fs]
            o != 0 && (A[1:bl, ts, fs, fb1, ant] .= @view θ[o:(o + bl - 1)])
        end
        fb2 = c.fb2[feed]
        if fb2 != 0
            o2 = oldplan.off2[ant, feed, ts, fs]
            o2 != 0 && (A[1:bl, ts, fs, fb2, ant] .= @view θ[o2:(o2 + bl - 1)])
        end
    end
    return A
end

@testset "Solve M2: per-site plan + grouped evaluate" begin
    # 4 times over 2 scans; 4 channels over 2 equal-width spws (no ragged padding).
    geom = DataGeometry(
        times = [0.0, 0.1, 1.0, 1.1],
        scan_of_time = [1, 1, 2, 2],
        channel_freqs = [1.0e9, 1.1e9, 2.0e9, 2.1e9],
        spw_of_chan = [1, 1, 2, 2],
    )
    nant = 4

    # A model exercising PerFeed, ReferenceRelative (fb2), and PerChannel/per-spw.
    model = StationGainModel(
        phase = (
            TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
            TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), GlobalFrequency()), ReferenceRelative(1)),
        ),
        logamp = (
            TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
        ),
    )

    plan = plan_gains(model, nant, geom)
    ev = GainEvaluator(model, geom; nant = nant)

    @testset "layout" begin
        @test length(plan.groups) == 1                     # homogeneous → one group
        @test plan.groups[1].ants == collect(1:nant)
        @test plan.nant == nant
        @test (plan.ntime, plan.nchan) == (4, 4)
        # No ragged padding here, so the per-site layout has exactly nθ params.
        @test nparameters(plan) == ev.layout.nθ
        @test length(zero_params(plan)) == ev.layout.nθ
    end

    # Random truth in the flat vector, mirrored into the per-site container.
    rng = MersenneTwister(20260708)
    θ = randn(rng, ev.layout.nθ)
    p = zero_params(plan)
    _fill_matching!(p, plan, ev, θ)

    @testset "equals flat-θ evaluate" begin
        # Bit-identical: same term_eval, same coordinate tables, same param values.
        @test evaluate_gains(plan, p) == evaluate_gains(ev, θ)
        # Windowed (a subset of global channels/times) also matches.
        ci = [2, 3]
        ti = [1, 4]
        @test evaluate_gains(plan, p, ci, ti) == evaluate_gains(ev, θ, ci, ti)
        # Shape is (nchan, ntime, nant, feed).
        @test size(evaluate_gains(plan, p)) == (4, 4, nant, 2)
    end

    @testset "flatten / unflatten round-trip" begin
        x = flatten(p)
        @test x isa AbstractVector{Float64}
        @test length(x) == nparameters(plan)
        p2 = unflatten(plan, copy(x))
        @test flatten(p2) == x
        @test evaluate_gains(plan, p2) == evaluate_gains(plan, p)
    end

    @testset "type stability (homogeneous)" begin
        @test (@inferred evaluate_gains(plan, p, [2, 3], [1, 4])) isa Array{ComplexF64, 4}
        @test (@inferred evaluate_gains(plan, p)) isa Array{ComplexF64, 4}
    end

    @testset "heterogeneous array (per-site override)" begin
        # Antenna 2 gets an extra Dispersion phase term → a second model structure.
        model2 = StationGainModel(
            phase = (
                TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
                TiedComponent(GainComponent(ConstantTerm(), GlobalTime(), GlobalFrequency()), ReferenceRelative(1)),
                TiedComponent(GainComponent(Dispersion(), PerScan(), GlobalFrequency()), SharedFeeds()),
            ),
            logamp = (
                TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
            ),
        )
        am = ArrayGainModel(model, Dict(2 => model2))
        @test site_model(am, 2) === model2
        @test site_model(am, 1) === model

        plan_h = plan_gains(am, nant, geom)
        @test length(plan_h.groups) == 2
        # Group membership: the default-model antennas vs the overridden one.
        antsets = Set(Set(gp.ants) for gp in plan_h.groups)
        @test antsets == Set([Set([1, 3, 4]), Set([2])])
        # The overridden group carries the extra (3rd) phase component.
        gov = plan_h.groups[findfirst(gp -> gp.ants == [2], plan_h.groups)]
        @test length(gov.phase) == 3
        gdef = plan_h.groups[findfirst(gp -> gp.ants == [1, 3, 4], plan_h.groups)]
        @test length(gdef.phase) == 2

        ph = zero_params(plan_h)
        @test size(evaluate_gains(plan_h, ph)) == (4, 4, nant, 2)
        # Grouping keeps the heterogeneous forward map inferrable.
        @test (@inferred evaluate_gains(plan_h, ph, [2, 3], [1, 4])) isa Array{ComplexF64, 4}
    end
end
