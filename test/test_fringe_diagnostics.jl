# Fringe diagnostics + Makie plot smoke tests. Reuses the synthetic builder
# `_build_fringe_ps` from test_pipeline.jl (included earlier in runtests.jl)
# and CairoMakie (loaded at the top of runtests.jl).


@testset "Fringe diagnostics" begin
    # Rates large enough that the uncorrected scan average decorrelates.
    station_rate = [0.0, 0.8e-3, -0.9e-3, 1.0e-3]
    ps, _truth = _build_fringe_ps(; station_rate)
    gauge = PinAntenna(1)
    sol = _combined(
        _fit_chain(
            (
                BaselineFringeFit(; gauge), Bandpass(; gauge),
                AdhocPhase(FP.PerTrackAdhocSmoother(; options = FP.AdhocOptions(; snr_floor = 0.0)); gauge),
            ), ps,
        )
    )

    @testset "snr table + summary" begin
        rows = FP.fringe_snr_table(sol)
        @test rows isa Vector{<:NamedTuple}
        @test !isempty(rows)
        @test length(rows) == sol.info.nscan
        r1 = first(rows)
        @test haskey(r1, :scan) && haskey(r1, :max_snr) && haskey(r1, :ncomp)
        @test all(r -> r.max_snr >= 0, rows)

        # PFA column: the solve records the per-scan effective search cells, and
        # the synthetic fringes are strong → secure detections on every scan.
        @test haskey(r1, :pfa)
        fringe = sol.steps[:fringe]
        @test length(fringe.scan_ncells) == sol.info.nscan
        @test all(>=(1), fringe.scan_ncells)
        @test fringe.search isa FP.FringeSearch
        for r in rows
            @test r.pfa ≈ FP.fringe_pfa(r.max_snr, fringe.scan_ncells[r.scan])
            r.max_snr > 10 && @test r.pfa < 1.0e-10
        end
        # A marginal SNR on the same search space would NOT be secure.
        @test FP.fringe_pfa(3.0, fringe.scan_ncells[1]) > 0.01

        buf = IOBuffer()
        @test_nowarn FP.print_fringe_snr_table(rows; io = buf)
        @test occursin("Fringe per-scan summary", String(take!(buf)))
        # Empty-info path prints a friendly message rather than erroring.
        @test_nowarn FP.print_fringe_snr_table(NamedTuple[]; io = IOBuffer())

        s = FP.fringe_solution_summary(sol)
        @test s isa String
        @test occursin("FringeSolution", s)
    end

    @testset "suspect_fringes (recorded detection table)" begin
        # The solve records every MEASURED cell as parallel plain vectors on
        # the fringe step's own diagnostics, with `det_detected` marking the ones
        # it accepted as real fringes.
        info = sol.info
        inf = sol.steps[:fringe]
        n = length(inf.det_pfa)
        @test n > 0
        @test length(inf.det_scan) == length(inf.det_ant_a) == length(inf.det_ant_b) ==
            length(inf.det_feed_a) == length(inf.det_feed_b) == length(inf.det_snr) == n
        @test all(s -> 1 <= s <= info.nscan, inf.det_scan)
        @test length(inf.det_detected) == n
        @test all(p -> 0.0 <= p <= 1.0, inf.det_pfa)
        # `detected` IS the PFA test at the solve's own threshold — nothing else.
        pfa_max = FP.Stationization().pfa_max
        @test inf.det_detected == (inf.det_pfa .<= pfa_max)
        # Strong synthetic fringes: every accepted row outscores every rejected one.
        @test all(inf.det_snr[inf.det_detected] .> maximum(inf.det_snr[.!inf.det_detected]; init = -Inf))

        # Strong synthetic fringes → nothing suspect at the default threshold.
        @test isempty(FP.suspect_fringes(sol))
        # ...and an all-pass threshold returns every recorded row, most-suspect first.
        rows = FP.suspect_fringes(sol; pfa_max = -1.0)
        @test length(rows) == count(inf.det_detected)     # accepted rows only
        @test issorted([r.pfa for r in rows]; rev = true)
        r = first(rows)
        @test r.sta_a == sol.geom.stations[r.a] && r.sta_b == sol.geom.stations[r.b]

        # Solutions without the table (e.g. loaded from an older file) degrade cleanly.
        old = CAL.CalibrationSolution(sol.geom, sol[:fringe].components, OrderedDict(:fringe => (; nscan = 1)))
        @test isempty(FP.suspect_fringes(old))
    end

    @testset "windowed gain evaluation for plots" begin
        # The plot entry points read a spectrum as `gains(sol; Ti = ti)` and a
        # time series as `gains(sol; Frequency = ci)` — an integer selector
        # windows the evaluation to that sample and drops the dimension.
        g = gains(sol; Ti = 1)
        @test size(g) == (length(sol.geom.channel_freqs), length(sol.geom.stations), 2)
        @test eltype(g) <: Complex
        @test lookup(g, UVP.Frequency) == sol.geom.channel_freqs

        gt = gains(sol; Frequency = 1)
        @test size(gt) == (length(sol.geom.times), length(sol.geom.stations), 2)
        @test lookup(gt, Ti) == sol.geom.times

        @test_throws BoundsError gains(sol; Ti = 10_000)
    end

    @testset "station codes available for plot labels" begin
        # Spectrum/phase plots read codes from the solution's geometry.
        @test sol.geom.stations == ["A1", "A2", "A3", "A4"]
    end

    @testset "Makie plot smoke" begin
        @test !isnothing(FP.plot_fringe_spectrum(sol))
        @test !isnothing(FP.plot_fringe_spectrum(sol; sites = 1, feeds = [1]))
        @test !isnothing(FP.plot_fringe_spectrum(sol; sites = 1, residual = true))
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

    @testset "coherence thermal debias" begin
        # One (baseline, pol) of `nchan` cells with inverse-variance weights:
        # flat phase + noise should debias to η ≈ 1 (raw is pulled below by the noise);
        # a real per-cell phase scatter must keep η < 1 even debiased.
        #
        # The weight is the inverse variance of ONE REAL COMPONENT (`w = 1/Var(Re V)`,
        # the FITS-IDI convention the reader delivers), so the noise is generated with
        # per-component σ and the cell's complex noise power is `2σ²` — matching the
        # factor of 2 the debias subtracts. Generating `σ·(randn + im·randn)/√2`
        # instead would be the complex convention (`E|n|² = 1/w`) and would silently
        # test the debias against data no reader produces.
        function freq_eta(; sigma, phase_rms, debias, nchan = 600, seed = 7)
            rng = MersenneTwister(seed); w = 1 / sigma^2
            V = Array{ComplexF64}(undef, nchan, 1, 1, 1); W = fill(w, nchan, 1, 1, 1)
            for c in 1:nchan
                V[c, 1, 1, 1] = cis(phase_rms * randn(rng)) + sigma * (randn(rng) + im * randn(rng))
            end
            freqs = collect(range(1.0e9, 1.1e9; length = nchan))
            numF = zeros(1, 1); den = zeros(1); dvar = zeros(1); npts = zeros(Int, 1)
            UVP._coherence_accumulate!(
                zeros(0, 1), numF, den, dvar, npts, V, W, nothing, [1], [1], [0.0], freqs,
                Float64[], [2.0e8], debias, ones(1, 1),
            )
            # ratio as `_curve_from_sums` forms it: debiased sums are POWERS
            # (η = √ of the clamped ratio), raw sums are amplitudes.
            den[1] > 0 || return NaN
            r = numF[1, 1] / den[1]
            return debias ? sqrt(clamp(r, 0.0, 1.0)) : min(r, 1.0)
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
        # Per-component σ (see the convention note in the debias testset above), chosen
        # so the cell's complex noise power `2σ²` — and hence the per-cell amplitude
        # SNR `|S|/√(2σ²)` ≈ 0.4 — a faint/resolved-baseline regime.
        sigma = 2.5 / sqrt(2); w = 1 / sigma^2
        V = Array{ComplexF64}(undef, nchan, nti, 1, 1); W = fill(w, nchan, nti, 1, 1)
        for c in 1:nchan, t in 1:nti
            V[c, t, 1, 1] = 1.0 + sigma * (randn(rng) + im * randn(rng))
        end
        times = collect(1.0:nti); freqs = collect(1.0e9 .+ (0:(nchan - 1)) .* 1.0e6)
        dts = [Float64(nti)]
        eta(num, den) = den[1] > 0 ? sqrt(clamp(num[1, 1] / den[1], 0.0, 1.0)) : NaN
        # per-channel time η at full averaging (debiased): unbiased even at
        # per-cell SNR ≈ 0.4, so ≈ 1 for a flat-phase source — just noisier
        # than the marginalized version below.
        nT = zeros(1, 1); den = zeros(1)
        UVP._coherence_accumulate!(nT, zeros(1, 1), den, zeros(1), zeros(Int, 1), V, W, nothing, [1], [1], times, freqs, dts, [1.0e8], true, ones(1, 1))
        eta_perchan = eta(nT, den)
        @test eta_perchan > 0.9
        # marginalized: band-average per AP, then time η — same estimand,
        # measured on high-SNR samples, so lower variance.
        Vt, Wt = UVP._collapse_axis(V, W, falses(size(V)), 1)
        @test size(Vt) == (1, nti, 1, 1)
        nT2 = zeros(1, 1); denT = zeros(1)
        UVP._coherence_accumulate!(nT2, zeros(1, 1), denT, zeros(1), zeros(Int, 1), Vt, Wt, nothing, [1], [1], times, [1.5e9], dts, [1.0], true, ones(1, 1))
        eta_marg = eta(nT2, denT)
        @test eta_marg > 0.95               # ≈ 1 for a flat-phase source
    end
end

@testset "fringe_freq_group_stats: per-band coherence + band splitting" begin
    # 3 bands of 4 channels with gaps; after = flat phase (η=1), before = ramp.
    freqs = vcat(1.0e9 .+ (0:3) .* 1.0e6, 1.1e9 .+ (0:3) .* 1.0e6, 1.3e9 .+ (0:3) .* 1.0e6)
    @test FP._freq_group_ranges(freqs) == [1:4, 5:8, 9:12]
    nchan = length(freqs)
    bl = [(1, 2)]
    spec_b = reshape(ComplexF64[cis(2π * c / 6) for c in 1:nchan], nchan, 1, 1)
    spec_a = reshape(fill(1.0 + 0.0im, nchan), nchan, 1, 1)
    data = FP.BaselineFringeData(
        "S", "1", 1, 100.0, bl, ["A", "B"], [(1, 1)], freqs, [0.0],
        spec_b, spec_a,
        zeros(ComplexF64, 1, 1, 1), zeros(ComplexF64, 1, 1, 1),
    )
    stats = FP.fringe_freq_group_stats(data; pol = 1)
    @test length(stats) == 3
    @test all(r -> r.nchan == 4, stats)
    @test all(r -> r.eta_after ≈ 1.0, stats)
    @test all(r -> r.eta_before < 0.9, stats)
    @test stats[1].f_lo == freqs[1] && stats[3].f_hi == freqs[end]

    # Band GROUPS: comparable inter-block gaps merge into one group; a far-away
    # block splits off its own group (the VGOS 3/5/6/10 GHz situation).
    @test FP.fringe_freq_groups(freqs) == [1:12]
    freqs2 = vcat(freqs, 5.0e9 .+ (0:3) .* 1.0e6)
    @test FP.fringe_freq_groups(freqs2) == [1:12, 13:16]
    @test FP.fringe_freq_groups(freqs2[1:1]) == [1:1]

    # Compat constructor: one full-range group whose band time series mirror the
    # full-band ones.
    @test data.freq_groups == [1:12]
    @test data.tser_freqgroup_before[:, :, :, 1] == data.tser_before
    @test data.tser_freqgroup_after[:, :, :, 1] == data.tser_after
end

# ── Thermal error bars on the baseline-fringe panels ─────────────────────────
#
# `plot_baseline_fringes` draws DATA — coherent visibility averages — so every
# point has a thermal width and no gain uncertainty is involved. The width is
# `σ = 1/√Σw` on the weighted mean, and the panels derive an amplitude bar
# (σ itself) and a phase bar (σ/|V|) from the same complex sample.

@testset "baseline fringe error bars" begin
    @testset "phase bar saturates rather than lying" begin
        @test FP._plotted_sigma(abs, 2.0 + 0im, 0.5) == 0.5
        @test FP._plotted_sigma(angle, 2.0 + 0im, 0.5) ≈ 0.25
        # Once the width reaches the sample the phase is unconstrained.
        @test FP._plotted_sigma(angle, 0.1 + 0im, 5.0) == Float64(π)
        @test FP._plotted_sigma(angle, 0.0 + 0im, 1.0) == Float64(π)
    end

    @testset "unknown widths draw no bars" begin
        # The weightless constructor cannot know a width; it reports NaN, and
        # every panel still renders.
        rng = MersenneTwister(0xBA25)
        freqs = vcat(230.0e9 .+ (0:7) .* 2.0e6, 230.1e9 .+ (0:7) .* 2.0e6)
        times = collect(0.0:30.0:150.0)
        bl_pairs = [(1, 2), (1, 3), (2, 3)]
        feeds = [(1, 1), (2, 2)]
        spec(n) = randn(rng, ComplexF64, n, length(bl_pairs), length(feeds))
        bare = FP.BaselineFringeData(
            "S", "1", 1, 100.0, bl_pairs, ["A1", "A2", "A3"], feeds, freqs, times,
            spec(length(freqs)), spec(length(freqs)), spec(length(times)), spec(length(times)),
        )
        @test all(isnan, bare.spec_sigma_before)
        for kind in (:freq, :time), show in (:phase, :amp)
            @test FP.plot_baseline_fringes(bare; pol = 1, kind = kind, show = show) isa Figure
        end
    end
end
