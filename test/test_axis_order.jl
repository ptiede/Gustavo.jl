# The solver reads each Measurement Set by dimension name and each product's
# feeds from `feed_pairs`, so the same data with its layers stored in another
# axis order, as views, or with the members of a scan group in another order,
# gives the same answer as MSv4's `(Polarization, Frequency, BaselineID, Ti)`.

@isdefined(_build_fringe_ps) || include("synthetic_ps.jl")

@testset "solver steps are independent of the layers' axis order" begin
    CAL = Gustavo.Calibration
    FP = Gustavo.Fring

    function relaid(ms, f)
        out = copy(ms)
        for k in (:visibility, :weight, :flag)
            out[k] = f(ms[k])
        end
        return out
    end
    function relaid(ps::XRadio.ProcessingSet, f; order = identity)
        members = order([k => relaid(ms, f) for (k, ms) in pairs(ps)])
        return XRadio.ProcessingSet(
            OrderedDict{Symbol, XRadio.MeasurementSet}(members), copy(DimensionalData.metadata(ps)),
        )
    end

    # Sub-bands spread over a wide fractional bandwidth, so the fringe search
    # spans windows far apart in frequency.
    ps, _ = _build_fringe_ps(nspw = 4, noise = 0.05, ref_freq = 3.0e9, spw_sep = 2.0e9)
    geom = CAL.DataGeometry(ps)
    θ(sol, name) = sol[name].steps[1].θ
    reference = fit(BaselineFringeFit(), ps; gauge = PinAntenna(1))
    ref_search = FP.search_scan(ps, geom, FP.FringeSearch())
    @test any(parent(ref_search[:valid]))

    @testset "$variant" for (variant, other) in (
            "time first" => relaid(ps, A -> permutedims(A, (Ti, BaselineID, Frequency, Polarization))),
            "frequency first" => relaid(ps, A -> permutedims(A, (Frequency, Ti, BaselineID, Polarization))),
            "views" => relaid(ps, A -> view(A, Ti(1:size(A, Ti)))),
            "members reversed" => relaid(ps, identity; order = reverse),
        )
        @testset "search_scan" begin
            b = FP.search_scan(other, geom, FP.FringeSearch())
            @test all(k -> isequal(parent(ref_search[k]), parent(b[k])), keys(ref_search))
        end

        @testset "residual_group" begin
            layout = reference[:fringe].steps[1].layout
            a = FP.residual_group(layout, θ(reference, :fringe), ps, geom)
            b = FP.residual_group(layout, θ(reference, :fringe), other, geom)
            for (k, ms) in pairs(b)
                @test dims(ms[:visibility]) == dims(other[k][:visibility])
                @test isequal(parent(permutedims(ms[:visibility], dims(a[k][:visibility]))), parent(a[k][:visibility]))
            end
        end

        @testset "$(nameof(typeof(step)))" for step in (BaselineFringeFit(), AdhocPhase(), Bandpass())
            a = fit(step, ps; gauge = PinAntenna(1))
            b = fit(step, other; gauge = PinAntenna(1))
            name = Gustavo.provides(step)
            @test maximum(abs, θ(a, name)) > 0
            @test isequal(θ(a, name), θ(b, name))
        end
    end
end
