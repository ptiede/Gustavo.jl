# Fringe diagnostics + Makie plot smoke tests (Phase 8). Reuses the synthetic
# multi-band UVSet builder `_build_fringe_uvset` from test_pipeline.jl (included
# earlier in runtests.jl) and CairoMakie (loaded at the top of runtests.jl).

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
end
