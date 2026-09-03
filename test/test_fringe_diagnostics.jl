# Fringe diagnostics + Makie plot smoke tests (Phase 8). Reuses the synthetic
# multi-band UVSet builder `_build_fringe_uvset` from test_pipeline.jl (included
# earlier in runtests.jl) and CairoMakie (loaded at the top of runtests.jl).

using HDF5

@testset "Fringe diagnostics" begin
    uvset, _truth = _build_fringe_uvset()
    sol = fit(
        FringeFit(model = FringeModel()) |> Bandpass() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset,
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
        fringe = sol[:fringe].steps[1]
        @test length(fringe.info.scan_ncells) == sol.info.nscan
        @test all(>=(1), fringe.info.scan_ncells)
        @test sol.info.search isa FP.FringeSearch
        for r in rows
            @test r.pfa ≈ FP.fringe_pfa(r.max_snr, fringe.info.scan_ncells[r.scan])
            r.max_snr > 10 && @test r.pfa < 1.0e-10
        end
        # A marginal SNR on the same search space would NOT be secure.
        @test FP.fringe_pfa(3.0, fringe.info.scan_ncells[1]) > 0.01

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
        # The solve records every MEASURED cell as parallel plain vectors
        # (HDF5-representable), on the fringe step's own info, with `det_detected`
        # marking the ones it accepted as real fringes.
        info = sol.info
        inf = sol[:fringe].steps[1].info
        n = length(inf.det_pfa)
        @test n > 0
        @test length(inf.det_scan) == length(inf.det_ant_a) == length(inf.det_ant_b) ==
            length(inf.det_pol) == length(inf.det_snr) == n
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
        @test r.sta_a == info.ant_names[r.a] && r.sta_b == info.ant_names[r.b]
        # A flagged row is directly inspectable with fringe_search_map (same
        # scan/baseline/product → the same detection, up to FFT-plan noise).
        m = FP.fringe_search_map(uvset, sol; scan_index = r.scan, baseline = (r.a, r.b), pol = r.pol)
        @test m.bl_pair in ((r.a, r.b), (r.b, r.a))
        @test m.pol == r.pol
        @test m.map.detection.snr ≈ r.snr rtol = 1.0e-6

        # Solutions without the table (e.g. loaded from an older file) degrade cleanly.
        fs = sol[:fringe].steps[1]
        old = CAL.CalibrationSolution(fs.model, fs.layout, sol.geom, fs.θ, (; nscan = 1); name = :fringe)
        @test isempty(FP.suspect_fringes(old))
    end

    @testset "windowed gain evaluation for plots" begin
        # The plot entry points read a spectrum as `gains(sol; Ti = ti)` and a
        # time series as `gains(sol; Frequency = ci)` — an integer selector
        # windows the evaluation to that sample and drops the dimension.
        g = gains(sol; Ti = 1)
        @test size(g) == (length(sol.geom.channel_freqs), CAL._nant(sol), 2)
        @test eltype(g) <: Complex
        @test lookup(g, UVP.Frequency) == sol.geom.channel_freqs

        gt = gains(sol; Frequency = 1)
        @test size(gt) == (length(sol.geom.times), CAL._nant(sol), 2)
        @test lookup(gt, Ti) == sol.geom.times

        @test_throws BoundsError gains(sol; Ti = 10_000)
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

    @testset "baseline before/after data" begin
        data = FP.baseline_fringe_data(uvset, sol)
        @test data isa FP.BaselineFringeData
        nchan = length(sol.geom.channel_freqs)
        nbl = length(data.bl_pairs)
        npol = length(data.pol_products)
        @test size(data.spec_before) == (nchan, nbl, npol)
        @test size(data.spec_after) == (nchan, nbl, npol)
        @test size(data.tser_before, 1) == length(data.times)
        @test data.scan_index == FP._max_snr_scan(sol, length(FP.scan_stream(uvset).groups))

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
        ngroups = length(FP.scan_stream(uvset).groups)
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

    @testset "fringe_search_map (delay–rate surface)" begin
        m = FP.fringe_search_map(uvset, sol)
        @test m isa FP.BaselineFringeMap
        # Defaults: the highest-SNR scan, the strongest baseline, a parallel hand.
        @test m.scan_index == FP._max_snr_scan(sol, length(FP.scan_stream(uvset).groups))
        @test m.ant_names == ["A1", "A2", "A3", "A4"]
        fa, fb = CAL.correlation_feed_pair(m.pol)
        @test fa == fb
        fsm = m.map
        @test size(fsm.snr) == (length(fsm.delays), length(fsm.rates))
        @test fsm.detection.valid
        # The strongest baseline's map peak is the scan's recorded max SNR (up to
        # peak refinement; the scan max is over all baselines/products searched).
        @test fsm.detection.snr <= sol[:fringe].steps[1].info.scan_snr[m.scan_index] * (1 + 1.0e-9)
        @test fsm.pfa < 1.0e-6
        # The map peak sits at the detection's (delay, rate) within a grid bin.
        pk = argmax(fsm.snr)
        @test isapprox(fsm.delays[pk[1]], fsm.detection.delay; atol = fsm.delays[2] - fsm.delays[1])
        @test isapprox(fsm.rates[pk[2]], fsm.detection.rate; atol = fsm.rates[2] - fsm.rates[1])

        # Explicit selectors: scan, baseline by station codes / indices / column, pol.
        a, b = m.bl_pair
        m2 = FP.fringe_search_map(
            uvset, sol;
            scan_index = m.scan_index, baseline = (m.ant_names[a], m.ant_names[b]), pol = m.pol,
        )
        @test m2.bl_pair == m.bl_pair
        @test m2.map.detection.snr ≈ fsm.detection.snr rtol = 1.0e-10
        @test FP.fringe_search_map(uvset, sol; baseline = (b, a)).bl_pair == m.bl_pair  # order-insensitive
        m3 = FP.fringe_search_map(uvset, sol; baseline = 2, pol = 1)
        @test m3.pol == "PP"

        @test_throws ErrorException FP.fringe_search_map(uvset, sol; scan_index = 10_000)
        @test_throws ErrorException FP.fringe_search_map(uvset, sol; baseline = ("A1", "nope"))
        @test_throws ErrorException FP.fringe_search_map(uvset, sol; pol = "XX")
    end

    @testset "plot_fringe_search smoke" begin
        m = FP.fringe_search_map(uvset, sol)
        @test !isnothing(FP.plot_fringe_search(m))
        @test !isnothing(FP.plot_fringe_search(m.map))                    # unlabeled low-level map
        @test !isnothing(FP.plot_fringe_search(uvset, sol; baseline = m.bl_pair))
        fig = Figure(size = (900, 700))
        @test !isnothing(FP.plot_fringe_search(fig[1, 1], m))
        figm = FP.plot_fringe_search(m)
        @test (show(IOBuffer(), MIME("image/png"), figm); true)

        # Zoom: the default view is a window around the peak, `false` the whole
        # searched plane, a number that span in main-lobe widths.
        @test !isnothing(FP.plot_fringe_search(m; zoom = 30))
        @test !isnothing(FP.plot_fringe_search(uvset, sol; baseline = m.bl_pair, zoom = false))
        @test_throws ErrorException FP.plot_fringe_search(m; zoom = 0)

        figfull = FP.plot_fringe_search(m; zoom = false)
        show(IOBuffer(), MIME("image/png"), figfull)          # lay out, so limits are final
        map_axis(f) = only(filter(c -> c isa Axis && c.xlabel[] == "delay (ns)", contents(f.layout)))
        zoomed = map_axis(figm).finallimits[]
        full = map_axis(figfull).finallimits[]
        @test zoomed.widths[1] < full.widths[1]
        @test zoomed.widths[2] < full.widths[2]
        @test zoomed.origin[1] <= m.map.detection.delay * 1.0e9 <= zoomed.origin[1] + zoomed.widths[1]
        @test zoomed.origin[2] <= m.map.detection.rate * 1.0e3 <= zoomed.origin[2] + zoomed.widths[2]
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
        # per-frequency-group view (freq restricts channels; time uses the freqgroup tser)
        nbg = length(data.freq_groups)
        @test !isnothing(FP.plot_baseline_fringes(data; kind = :freq, freqgroup = nbg))
        @test !isnothing(FP.plot_baseline_fringes(data; kind = :time, freqgroup = 1))
        @test_throws Exception FP.plot_baseline_fringes(data; freqgroup = nbg + 1)
        @test !isnothing(FP.plot_fringe_spectrum(sol; freqgroup = 1))
    end

    @testset "coherence report (stage-agnostic)" begin
        corr = Gustavo.UVData.apply_calibration(uvset, sol)
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
                zeros(0, 1), numF, den, dvar, npts, V, W, [1], [1], [0.0], freqs,
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
        UVP._coherence_accumulate!(nT, zeros(1, 1), den, zeros(1), zeros(Int, 1), V, W, [1], [1], times, freqs, dts, [1.0e8], true, ones(1, 1))
        eta_perchan = eta(nT, den)
        @test eta_perchan > 0.9
        # marginalized: band-average per AP, then time η — same estimand,
        # measured on high-SNR samples, so lower variance.
        Vt, Wt = UVP._collapse_axis(V, W, 1)
        @test size(Vt) == (1, nti, 1, 1)
        nT2 = zeros(1, 1); denT = zeros(1)
        UVP._coherence_accumulate!(nT2, zeros(1, 1), denT, zeros(1), zeros(Int, 1), Vt, Wt, [1], [1], times, [1.5e9], dts, [1.0], true, ones(1, 1))
        eta_marg = eta(nT2, denT)
        @test eta_marg > 0.95               # ≈ 1 for a flat-phase source
    end

    @testset "HDF5 caltable: round-trip + external-readable" begin
        path = tempname() * ".h5"
        try
            CAL.save_solution_hdf5(path, sol; time_block = 4)

            # Lossless Julia round-trip via the embedded blob.
            sol2 = CAL.load_solution_hdf5(path)
            @test sol2.pipeline isa CalibrationPipeline
            @test [s.layout.nθ for s in sol2.steps] == [s.layout.nθ for s in sol.steps]
            @test parent(gains(sol2)) == parent(gains(sol))
            @test sol2.geom.channel_freqs == sol.geom.channel_freqs

            # Language-neutral content: gains + axes + diagnostics readable directly.
            nchan = length(sol.geom.channel_freqs); ntime = length(sol.geom.times); nant = CAL._nant(sol)
            HDF5.h5open(path, "r") do f
                @test read(HDF5.attributes(f)["format"]) == "GustavoCalibrationSolution"
                @test haskey(f, "gain") && haskey(f, "axes")
                gr = read(f["gain"]["real"]); gi = read(f["gain"]["imag"])
                @test size(gr) == (nchan, ntime, nant, 2)
                @test read(f["axes"]["channel_freq_hz"]) == sol.geom.channel_freqs
                # Run-wide diagnostics at `info/*`, each step's own under
                # `info/steps/<name>/*` (generic — no per-step-name knowledge
                # needed to write or read them).
                @test haskey(f["info"], "ant_names")
                # The search configuration is exported flat, so external
                # readers get the full search provenance.
                @test read(f["info"]["search"]["algorithm"]) == "auto"
                @test read(f["info"]["search"]["oversample"]) == 8
                @test read(f["info"]["search"]["delay_window_s"]) == [-1.0e-6, 1.0e-6]
                @test haskey(f["info"]["steps"], "fringe")
                @test haskey(f["info"]["steps"]["fringe"], "scan_snr")
                @test haskey(f["info"]["steps"]["fringe"], "timing")
                @test haskey(f["info"]["steps"]["fringe"]["timing"], "decode")
                # gains in the file match the composed (product-over-steps) gains exactly (Float32 precision).
                g = CAL._composed_gains(sol, 1:nchan, 1:ntime)
                @test gr ≈ Float32.(real.(g))
                @test gi ≈ Float32.(imag.(g))
            end

            # gains = false → compact (blob-only) file still round-trips.
            path2 = tempname() * ".h5"
            CAL.save_solution_hdf5(path2, sol; gains = false)
            @test parent(gains(CAL.load_solution_hdf5(path2))) == parent(gains(sol))
            @test !HDF5.h5open(ff -> haskey(ff, "gain"), path2, "r")
            isfile(path2) && rm(path2)

            # An info entry with no HDF5 form is omitted from the external
            # file and reported, by name, in one warning per save.
            solx = CAL.CalibrationSolution(
                sol.steps, sol.geom, (; sol.info..., opaque = Ref(1));
                transforms = sol.transforms, postcal = sol.postcal,
            )
            path3 = tempname() * ".h5"
            @test_logs (:warn, r"no HDF5 representation.*opaque"s) match_mode = :any CAL.save_solution_hdf5(
                path3, solx; gains = false,
            )
            isfile(path3) && rm(path3)
        finally
            isfile(path) && rm(path)
        end
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
        "S", "1", 1, 100.0, bl, ["A", "B"], ["PP"], freqs, [0.0],
        spec_b, spec_a,
        zeros(ComplexF64, 1, 1, 1), zeros(ComplexF64, 1, 1, 1),
    )
    stats = FP.fringe_freq_group_stats(data)
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
