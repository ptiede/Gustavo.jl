# Fringe diagnostics + Makie plot smoke tests (Phase 8). Reuses the synthetic
# multi-band UVSet builder `_build_fringe_uvset` from test_pipeline.jl (included
# earlier in runtests.jl) and CairoMakie (loaded at the top of runtests.jl).

using HDF5

@testset "Fringe diagnostics" begin
    uvset, _truth = _build_fringe_uvset()
    sol = FP.solve_fringes(uvset; ref_ant = 1, adhoc = FP.AdhocPhasing(; window = 7, order = 2, snr_floor = 0.0))

    @testset "snr table + summary" begin
        rows = FP.fringe_snr_table(sol)
        @test rows isa Vector{<:NamedTuple}
        @test !isempty(rows)
        @test length(rows) == sol.info.nscan
        r1 = first(rows)
        @test haskey(r1, :scan) && haskey(r1, :max_snr) && haskey(r1, :chi) && haskey(r1, :ncomp)
        @test all(r -> r.max_snr >= 0, rows)

        buf = IOBuffer()
        @test_nowarn FP.print_fringe_snr_table(rows; io = buf)
        @test occursin("Fringe per-scan summary", String(take!(buf)))
        # Empty-info path prints a friendly message rather than erroring.
        @test_nowarn FP.print_fringe_snr_table(NamedTuple[]; io = IOBuffer())

        s = FP.fringe_solution_summary(sol)
        @test s isa String
        @test occursin("FringeSolution", s)
    end

    @testset "gain extractors" begin
        freqs, g = FP.fringe_gain_spectrum(sol; ti = 1)
        @test length(freqs) == length(sol.geom.channel_freqs)
        @test size(g) == (length(sol.geom.channel_freqs), sol.layout.nant, 2)
        @test eltype(g) <: Complex

        times, gt = FP.fringe_gain_time_series(sol; ci = 1)
        @test length(times) == length(sol.geom.times)
        @test size(gt) == (length(sol.geom.times), sol.layout.nant, 2)

        @test_throws ErrorException FP.fringe_gain_spectrum(sol; ti = 10_000)
        @test_throws ErrorException FP.fringe_gain_time_series(sol; ci = 10_000)
    end

    @testset "Makie plot smoke" begin
        @test !isnothing(FP.plot_fringe_spectrum(sol))
        @test !isnothing(FP.plot_fringe_spectrum(sol; sites = 1, feeds = [1]))
        @test !isnothing(FP.plot_fringe_phases(sol))
        @test !isnothing(FP.plot_fringe_phases(sol; sites = [1, 2], feeds = :all, ci = 2))
        @test !isnothing(FP.plot_fringe_snr(sol))

        fig = Figure(size = (1600, 500))
        @test !isnothing(FP.plot_fringe_spectrum(fig[1, 1], sol; sites = 1))
        @test !isnothing(FP.plot_fringe_phases(fig[1, 2], sol; sites = 1))
        @test !isnothing(FP.plot_fringe_snr(fig[1, 3], sol))

        # Smoke: rendering must run to completion without throwing. (We do not
        # use @test_nowarn — the Makie/PlotUtils backend emits benign cosmetic
        # "No strict ticks found" warnings on sparse synthetic axes.)
        fig_spec = FP.plot_fringe_spectrum(sol; sites = 1)
        @test (show(IOBuffer(), MIME("image/png"), fig_spec); true)
        @test (show(IOBuffer(), MIME("image/png"), fig); true)
    end

    @testset "baseline before/after data" begin
        data = FP.baseline_fringe_data(uvset, sol)
        @test data isa FP.BaselineFringeData
        nchan = length(sol.geom.channel_freqs)
        nbl = length(data.bl_pairs)
        npol = length(data.pol_products)
        @test size(data.spec_before) == (nchan, nbl, npol)
        @test size(data.spec_after) == (nchan, nbl, npol)
        @test size(data.tser_before, 1) == length(data.times)
        @test data.scan_index == FP._max_snr_scan(sol, length(FP._scan_group_leaves(uvset)))

        p = FP.baseline_pol_index(data, :parallel)
        @test 1 <= p <= npol
        @test FP.baseline_pol_index(data, data.pol_products[1]) == 1

        # Quality: dividing out the solution should ALIGN the per-channel phases
        # (flatten the delay slope), so the coherent concentration R = |Σe^{iφ}|/N
        # over frequency should not drop on any cross baseline.
        concentration(z) = (v = filter(isfinite, z); isempty(v) ? 0.0 : abs(sum(cis, angle.(v))) / length(v))
        improved = 0
        total = 0
        for bi in 1:nbl
            a, b = data.bl_pairs[bi]
            a == b && continue
            rb = concentration(@view data.spec_before[:, bi, p])
            ra = concentration(@view data.spec_after[:, bi, p])
            (rb == 0 && ra == 0) && continue
            total += 1
            ra >= rb - 1.0e-6 && (improved += 1)
        end
        @test total > 0
        @test improved == total          # no baseline gets LESS coherent after the fit
    end

    @testset "delay closure" begin
        data = FP.baseline_fringe_data(uvset, sol)
        c = FP.delay_closure(data)
        @test !isempty(c.triangles)
        @test length(c.closure_before) == length(c.triangles)

        finite(v) = filter(isfinite, v)
        mx(v) = (u = abs.(finite(v)); isempty(u) ? 0.0 : maximum(u))
        τscale = mx(c.data_delay)                         # spread of the data's baseline delays
        @test τscale > 0                                  # the synthetic injected real delays

        # Closure is the defining consistency property: triangle sums of the DATA
        # delays cancel (≪ the individual delays) because real delays are station-
        # based — this is what would break if the station model were wrong.
        @test mx(c.closure_before) < 0.05 * τscale
        # A correct delay solution removes the delay on every baseline (residual ≪
        # data) and cannot introduce closure errors.
        @test mx(c.resid_delay) < 0.05 * τscale
        @test mx(c.closure_after) < 0.05 * τscale

        @test_nowarn FP.print_delay_closure(c; io = IOBuffer())
        buf = IOBuffer(); FP.print_delay_closure(c; io = buf)
        @test occursin("Delay closure", String(take!(buf)))
    end

    @testset "plot_baseline_fringes smoke" begin
        data = FP.baseline_fringe_data(uvset, sol)
        @test !isnothing(FP.plot_baseline_fringes(data))                                   # freq/phase
        @test !isnothing(FP.plot_baseline_fringes(data; kind = :time))
        @test !isnothing(FP.plot_baseline_fringes(data; kind = :freq, show = :amp))
        @test !isnothing(FP.plot_baseline_fringes(data; baselines = 2, pol = 1))
        @test !isnothing(FP.plot_baseline_fringes(uvset, sol; kind = :time))               # full path
        fig = Figure(size = (900, 700))
        @test !isnothing(FP.plot_baseline_fringes(fig[1, 1], data; kind = :freq))
        figbl = FP.plot_baseline_fringes(data)
        @test (show(IOBuffer(), MIME("image/png"), figbl); true)
    end

    @testset "HDF5 caltable: round-trip + external-readable" begin
        path = tempname() * ".h5"
        try
            CAL.save_solution_hdf5(path, sol; time_block = 4)

            # Lossless Julia round-trip via the embedded blob.
            sol2 = CAL.load_solution_hdf5(path)
            @test sol2.θ == sol.θ
            @test sol2.layout.nθ == sol.layout.nθ
            @test sol2.geom.channel_freqs == sol.geom.channel_freqs

            # Language-neutral content: gains + axes + diagnostics readable directly.
            nchan = sol.layout.nchan; ntime = sol.layout.ntime; nant = sol.layout.nant
            HDF5.h5open(path, "r") do f
                @test read(HDF5.attributes(f)["format"]) == "GustavoCalibrationSolution"
                @test haskey(f, "gain") && haskey(f, "axes")
                gr = read(f["gain"]["real"]); gi = read(f["gain"]["imag"])
                @test size(gr) == (nchan, ntime, nant, 2)
                @test read(f["axes"]["channel_freq_hz"]) == sol.geom.channel_freqs
                @test haskey(f["info"], "scan_max_snr")
                # gains in the file match the evaluator exactly (Float32 precision).
                ev = CAL.GainEvaluator(sol.model, sol.layout)
                g = CAL.evaluate_gains(ev, sol.θ, 1:nchan, 1:ntime)
                @test gr ≈ Float32.(real.(g))
                @test gi ≈ Float32.(imag.(g))
            end

            # gains = false → compact (blob-only) file still round-trips.
            path2 = tempname() * ".h5"
            CAL.save_solution_hdf5(path2, sol; gains = false)
            @test CAL.load_solution_hdf5(path2).θ == sol.θ
            @test !HDF5.h5open(ff -> haskey(ff, "gain"), path2, "r")
            isfile(path2) && rm(path2)
        finally
            isfile(path) && rm(path)
        end
    end
end
