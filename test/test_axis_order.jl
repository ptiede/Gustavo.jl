# The solver reads a scan cube by dimension name, so the same data stored in
# MSv4's axis order `(Polarization, Frequency, BaselineID, Ti)` gives the same
# answer as Gustavo's `(Frequency, Ti, BaselineID, Polarization)`.

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "scan kernels are independent of the cube's axis order" begin
    CAL = Gustavo.Calibration
    FP = Gustavo.Fring
    ST = Gustavo.Streaming
    UV = Gustavo.UVData

    # Sub-bands spread over a wide fractional bandwidth, so the dispersion and
    # single-band-delay terms compile and their kernels run.
    uvset, _ = _build_fringe_uvset(nspw = 4, noise = 0.05, ref_freq = 3.0e9, spw_sep = 2.0e9)
    geom = CAL.build_geometry(uvset)
    stream = FP.scan_stream(uvset; geom)
    stack, win = FP.materialize_cube(stream, stream.groups[1])

    relaid(s, order) = DimStack(
        map(A -> permutedims(A, order), DimensionalData.layers(s));
        metadata = DimensionalData.metadata(s),
    )
    deepcopy_stack(s) = relaid(s, dims(s))
    same_cells(a, b) = all(k -> isequal(parent(permutedims(b[k], dims(a[k]))), parent(a[k])), keys(a))

    antennas = UV.union_antennas(uvset)
    function step_context(step)
        model = CAL.materialize(
            CAL.as_gain_model(Gustavo.model_components(step, (; geom, antennas))), antennas, geom,
        )
        layout = Gustavo.plan_parameters(model, antennas, geom; require_nonempty = false)
        gauge = CAL.resolve_gauge(CAL.ZeroSumPhase(), String.(antennas.name))
        return Gustavo.SolveContext(
            model, layout, geom, CAL.GainEvaluator(model, layout),
            CAL.component_vector(layout, zeros(layout.nθ)), gauge, length(antennas),
            antennas, stream, Dict{Symbol, Any}(),
        )
    end
    function process(step, s)
        ctx = step_context(step)
        Gustavo.start_pass!(step, ctx)
        r = Gustavo.process_scan!(step, ctx, s, win)
        return ctx, r
    end

    fringe_ctx, _ = process(FringeFit(), stack)
    sol = CAL.CalibrationSolution(
        fringe_ctx.model, fringe_ctx.layout, geom, fringe_ctx.θ,
        (; ant_names = String.(antennas.name)); name = :fringe,
    )

    # MSv4's order, and one that puts time before frequency.
    @testset "$order" for order in (
            (Polarization, Frequency, BaselineID, Ti),
            (Ti, Polarization, BaselineID, Frequency),
        )
        permuted = relaid(stack, order)
        @test size(permuted[:vis]) != size(stack[:vis])

        @testset "search_scan" begin
            a = FP.search_scan(stack, geom, FP.FringeSearch())
            b = FP.search_scan(permuted, geom, FP.FringeSearch())
            @test any(parent(a[:valid]))
            @test all(k -> isequal(parent(a[k]), parent(b[k])), keys(a))
        end

        @testset "$(nameof(typeof(step)))" for step in (
                FringeFit(),
                DispersionSBDFit(;
                    dispersion = DispersionModel(),
                    sbd = SingleBandDelay(; freq = CAL.FreqGroups([1:16, 17:32])),
                ),
                TemporalSmoother(),
            )
            a, _ = process(step, stack)
            b, _ = process(step, permuted)
            @test maximum(abs, a.θ) > 0
            @test isequal(a.θ, b.θ)
        end

        @testset "Bandpass accumulators" begin
            _, a = process(Bandpass(), stack)
            _, b = process(Bandpass(), permuted)
            @test maximum(abs, a.rl) > 0
            @test isequal(a.rl, b.rl) && isequal(a.wl, b.wl)
        end

        @testset "residual_vis" begin
            a = FP.residual_vis(fringe_ctx.ev, fringe_ctx.θ, stack, win)
            b = FP.residual_vis(fringe_ctx.ev, fringe_ctx.θ, permuted, win)
            @test dims(b) == dims(permuted[:vis])
            @test isequal(parent(permutedims(b, dims(a))), parent(a))
        end

        @testset "$(nameof(typeof(t)))" for t in (
                ST.ApplySolution(sol), ST.StationWeightScale([1.0, 2.0, 0.5, 3.0]),
            )
            a = deepcopy_stack(stack)
            b = deepcopy_stack(permuted)
            ST.apply_transform!(t, a, win)
            ST.apply_transform!(t, b, win)
            @test !same_cells(stack, a)
            @test same_cells(a, b)
        end

        @testset "gain application and solution row flags" begin
            info = UV.metadata(stack)
            g = CAL._composed_gains(
                sol, geom; chan_idx = win.chan_idx, ti_idx = win.ti_idx, time_span = info.time_span,
            )
            scanid = CAL._geom_scan_id(geom, info.scan_name)
            a = deepcopy_stack(stack)
            b = deepcopy_stack(permuted)
            for s in (a, b)
                CAL._apply_gains!(s, g)
                CAL._flag_solution_rows!(s[:flags], UV.baselines(s).pairs, scanid, Set([(2, scanid)]))
            end
            @test count(parent(a[:flags])) > count(parent(stack[:flags]))
            @test same_cells(a, b)
        end
    end
end
