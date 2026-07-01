# Fringe diagnostics + Makie plot smoke tests (Phase 8). Reuses the synthetic
# multi-band UVSet builder `_build_fringe_uvset` from test_pipeline.jl (included
# earlier in runtests.jl) and CairoMakie (loaded at the top of runtests.jl).

using HDF5

@testset "Fringe diagnostics" begin
    uvset, _truth = _build_fringe_uvset()
    sol = FP.solve_fringes(uvset; ref_ant = 1, adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0))

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

    @testset "station codes available for plot labels" begin
        # Spectrum/phase plots read codes from sol.info; baseline plots from the data.
        @test sol.info.ant_names == ["A1", "A2", "A3", "A4"]
        data = FP.baseline_fringe_data(uvset, sol)
        @test data.ant_names == ["A1", "A2", "A3", "A4"]
        a, b = data.bl_pairs[1]
        @test data.ant_names[a] isa String && data.ant_names[b] isa String
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

    @testset "fringe_scan_groups" begin
        g = FP.fringe_scan_groups(uvset, sol)
        ngroups = length(FP._scan_group_leaves(uvset))
        @test length(g) == ngroups
        @test all(r -> haskey(r, :scan_index) && haskey(r, :source) && haskey(r, :scan) && haskey(r, :max_snr), g)
        @test [r.scan_index for r in g] == collect(1:ngroups)        # solver order, 1-based
        @test Set(r.source for r in g) ⊆ Set(UVP.sources(uvset))
        # max_snr agrees with the per-scan SNR table / the default scan picker.
        @test g[FP._max_snr_scan(sol, ngroups)].max_snr == maximum(r.max_snr for r in g)
        # the per-source best-scan selection used by run_pipeline resolves to a valid group
        for s in unique(r.source for r in g)
            rows = filter(r -> r.source == s, g)
            best = rows[argmax([isfinite(r.max_snr) ? r.max_snr : -Inf for r in rows])]
            @test 1 <= best.scan_index <= ngroups
        end
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

    @testset "coherence report (stage-agnostic)" begin
        corr = Gustavo.apply_calibration(uvset, sol)
        raw = UVP.coherence_report(uvset)
        rep = UVP.coherence_report(corr)

        @test rep isa UVP.CoherenceReport
        nbl = length(rep.bl_pairs)
        @test nbl > 0
        @test all(p -> length(p) == 2 && p[1] == p[2], rep.pol_products)   # parallel-hand default
        @test size(rep.time.eta_baseline) == (length(rep.time.intervals), nbl)
        @test size(rep.freq.eta_baseline) == (length(rep.freq.intervals), nbl)
        @test length(rep.time.eta) == length(rep.time.intervals)
        @test rep.time.intervals == sort(rep.time.intervals)               # ascending sweep

        # Native resolution (one sample/channel per bin) is the η ≡ 1 anchor.
        @test rep.time.eta[1] ≈ 1.0 atol = 1.0e-6
        @test rep.freq.eta[1] ≈ 1.0 atol = 1.0e-6

        # The fringe fit removes the rate + screen, so the corrected data stays
        # coherent when the whole scan is averaged to one sample; the raw data
        # (with the injected time phase) decorrelates and is strictly worse.
        h = UVP.coherence_headline(rep)
        hraw = UVP.coherence_headline(raw)
        @test h.loss_time ≈ 1 - h.eta_time
        @test h.eta_time > 0.9
        @test hraw.eta_time < h.eta_time
        @test hraw.eta_time < 0.95                                          # raw genuinely decorrelated
        @test h.eta_freq ≥ hraw.eta_freq - 1.0e-6                           # band-averaging no worse

        # Selectors / overrides.
        @test UVP.coherence_report(corr; pols = :all) isa UVP.CoherenceReport
        rep2 = UVP.coherence_report(corr; timescales = [30.0, 120.0, 360.0], bandwidths = [4.0e6, 1.6e7])
        @test rep2.time.intervals == [30.0, 120.0, 360.0]
        @test rep2.freq.intervals == [4.0e6, 1.6e7]

        # Re-exported at the package top level.
        @test Gustavo.coherence_report(corr) isa UVP.CoherenceReport

        buf = IOBuffer()
        @test_nowarn UVP.print_coherence_report(rep; io = buf)
        @test occursin("Coherence report", String(take!(buf)))

        fig = UVP.plot_coherence(rep)
        @test !isnothing(fig)
        @test !isnothing(UVP.plot_coherence(rep; baselines = 1))
        @test !isnothing(UVP.plot_coherence(rep; nlabel = 3))          # worst baselines coloured + legend
        parent = Figure(size = (1000, 420))
        @test !isnothing(UVP.plot_coherence(parent[1, 1], rep))
        @test (show(IOBuffer(), MIME("image/png"), fig); true)

        # Per-baseline heatmap (rows = baselines, cols = intervals).
        figm = UVP.plot_coherence_matrix(rep)
        @test !isnothing(figm)
        @test !isnothing(UVP.plot_coherence_matrix(rep; axis = :freq, sortworst = false))
        pm = Figure(size = (700, 500))
        @test !isnothing(UVP.plot_coherence_matrix(pm[1, 1], rep))
        @test (show(IOBuffer(), MIME("image/png"), figm); true)
    end

    @testset "coherence thermal debias" begin
        # One (baseline, pol) of `nchan` cells with inverse-variance weights (w = 1/σ²):
        # flat phase + noise should debias to η ≈ 1 (raw is pulled below by the noise);
        # a real per-cell phase scatter must keep η < 1 even debiased.
        function freq_eta(; sigma, phase_rms, debias, nchan = 600, seed = 7)
            rng = MersenneTwister(seed); w = 1 / sigma^2
            V = Array{ComplexF64}(undef, nchan, 1, 1, 1); W = fill(w, nchan, 1, 1, 1)
            for c in 1:nchan
                V[c, 1, 1, 1] = cis(phase_rms * randn(rng)) + sigma * (randn(rng) + im * randn(rng)) / sqrt(2)
            end
            freqs = collect(range(1.0e9, 1.1e9; length = nchan))
            numF = zeros(1, 1); den = zeros(1); npts = zeros(Int, 1)
            UVP._coherence_accumulate!(
                zeros(0, 1), numF, den, npts, V, W, [1], [1], [0.0], freqs, Float64[], [2.0e8], debias,
            )
            return den[1] > 0 ? min(numF[1, 1] / den[1], 1.0) : NaN   # clamp as `_curve_from_sums` does
        end
        # Flat phase, high per-cell SNR: raw is biased below 1, debias ≈ 1.
        @test freq_eta(; sigma = 0.2, phase_rms = 0.0, debias = false) < 0.995
        @test freq_eta(; sigma = 0.2, phase_rms = 0.0, debias = true) > 0.99
        # Real residual phase (0.4 rad): debias must NOT hide it (η stays < 1).
        @test freq_eta(; sigma = 0.2, phase_rms = 0.4, debias = true) < 0.97
    end

    @testset "coherence marginalize (incoherent)" begin
        # Faint FLAT-phase source (per-cell amplitude SNR ≈ 0.4, like a resolved/weak
        # baseline): the per-channel time curve understates coherence even debiased,
        # but marginalizing (band-average per AP → high SNR) recovers η ≈ 1.
        rng = MersenneTwister(9)
        nchan, nti = 64, 40
        sigma = 2.5; w = 1 / sigma^2
        V = Array{ComplexF64}(undef, nchan, nti, 1, 1); W = fill(w, nchan, nti, 1, 1)
        for c in 1:nchan, t in 1:nti
            V[c, t, 1, 1] = 1.0 + sigma * (randn(rng) + im * randn(rng)) / sqrt(2)
        end
        times = collect(1.0:nti); freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6)
        dts = [Float64(nti)]
        # per-channel time η at full averaging (debiased)
        nT = zeros(1, 1); den = zeros(1)
        UVP._coherence_accumulate!(nT, zeros(1, 1), den, zeros(Int, 1), V, W, [1], [1], times, freqs, dts, [1.0e8], true)
        eta_perchan = den[1] > 0 ? min(nT[1, 1] / den[1], 1.0) : NaN
        # marginalized: band-average per AP, then time η
        Vt, Wt = UVP._collapse_axis(V, W, 1)
        @test size(Vt) == (1, nti, 1, 1)
        nT2 = zeros(1, 1); denT = zeros(1)
        UVP._coherence_accumulate!(nT2, zeros(1, 1), denT, zeros(Int, 1), Vt, Wt, [1], [1], times, [1.5e9], dts, [1.0], true)
        eta_marg = denT[1] > 0 ? min(nT2[1, 1] / denT[1], 1.0) : NaN
        @test eta_marg > eta_perchan        # marginalize recovers what per-channel loses
        @test eta_marg > 0.95               # ...to ≈ 1 for a flat-phase source
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
