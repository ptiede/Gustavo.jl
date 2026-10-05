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

    @testset "snr table" begin
        t = FP.fringe_snr_table(sol)
        fringe = sol.steps[:fringe]
        @test t isa DimStack
        @test keys(t) == (:max_snr, :pfa)
        @test collect(lookup(t, Gustavo.Scan)) == fringe.scan_names
        @test length(fringe.scan_names) == sol.info.nscan
        @test all(>=(0), t.max_snr)

        # The solve records the per-scan effective search cells, and the
        # synthetic fringes are strong → secure detections on every scan.
        @test all(>=(1), fringe.scan_ncells)
        @test fringe.search isa FP.FringeSearch
        @test parent(t.pfa) ≈ FP.fringe_pfa.(fringe.scan_snr, fringe.scan_ncells)
        @test all(t.pfa[t.max_snr .> 10] .< 1.0e-10)
        # A marginal SNR on the same search space would NOT be secure.
        @test FP.fringe_pfa(3.0, fringe.scan_ncells[1]) > 0.01

        bp_only = sol[:bandpass]
        @test_throws "the solution has no :fringe step" FP.fringe_snr_table(bp_only)
    end

    @testset "fringe_detections" begin
        # The solve records every MEASURED cell; `det_detected` marks the ones it
        # accepted as real fringes.
        inf = sol.steps[:fringe]
        det = FP.fringe_detections(sol)
        @test keys(det) == (:snr, :pfa, :delay, :rate, :phase, :detected)
        @test map(DimensionalData.basetypeof, dims(det)) == (Gustavo.Scan, Gustavo.AntennaPair, Gustavo.FeedPair)
        measured = .!isnan.(det.snr)
        @test count(measured) == length(inf.det_snr)
        @test count(det.detected) == count(inf.det_detected)
        @test all(p -> 0.0 <= p <= 1.0, det.pfa[measured])
        # `detected` IS the PFA test at the solve's own threshold — nothing else.
        pfa_max = FP.Stationization().pfa_max
        @test det.detected == (det.pfa .<= pfa_max)
        # Strong synthetic fringes: every accepted cell outscores every rejected one.
        @test minimum(det.snr[det.detected]) > maximum(det.snr[measured .& .!det.detected]; init = -Inf)

        # Cells are labeled by name.
        i = findfirst(inf.det_detected)
        cell = det[
            Gustavo.Scan(At(inf.scan_names[inf.det_scan[i]])),
            Gustavo.AntennaPair(At((inf.det_ant_a[i], inf.det_ant_b[i]))),
            Gustavo.FeedPair(At((inf.det_feed_a[i], inf.det_feed_b[i]))),
        ]
        @test cell.snr == inf.det_snr[i] && cell.delay == inf.det_delay[i] && cell.detected
    end

    @testset "cat_scans combines per-scan results" begin
        ps2, _ = _build_fringe_ps(; nscans = 2, noise = 0.05)
        whole = fit(BaselineFringeFit(; gauge), ps2)
        per_scan = mapsets(g -> fit(BaselineFringeFit(; gauge), g), XRadio.groupby(ps2, XRadio.ByScan()))
        @test length(per_scan) == 2

        t = FP.cat_scans(FP.fringe_snr_table.(values(per_scan)))
        @test lookup(t, Gustavo.Scan) == lookup(FP.fringe_snr_table(whole), Gustavo.Scan)
        @test t.max_snr == FP.fringe_snr_table(whole).max_snr
        st = FP.cat_scans(FP.fringe_station_solutions.(values(per_scan)))
        @test st.delay ≈ FP.fringe_station_solutions(whole).delay
        det = FP.cat_scans(FP.fringe_detections.(values(per_scan)))
        @test count(det.detected) == count(FP.fringe_detections(whole).detected)

        # Differing labels form a union; a cell an input lacks is NaN (false for Bool).
        a = DimArray([1.0 2.0], (FP._scan_dim(["s1"]), FP._station_dim(["A", "B"])))
        b = DimArray([3.0 4.0], (FP._scan_dim(["s2"]), FP._station_dim(["B", "C"])))
        ab = FP.cat_scans([a, b])
        @test collect(lookup(ab, Gustavo.AntennaName)) == ["A", "B", "C"]
        @test isequal(parent(ab), [1.0 2.0 NaN; NaN 3.0 4.0])
        fa = DimArray([true;;], (FP._scan_dim(["s1"]), FP._station_dim(["A"])))
        fb = DimArray([true;;], (FP._scan_dim(["s2"]), FP._station_dim(["B"])))
        @test parent(FP.cat_scans([fa, fb])) == [true false; false true]

        @test_throws "cat_scans: scans repeat: s1" FP.cat_scans([a, a])
        @test_throws "cat_scans: nothing to concatenate" FP.cat_scans(DimArray[])
        @test_throws "cat_scans: dimensions differ" FP.cat_scans([a, DimArray([1.0], (FP._scan_dim(["s3"]),))])
    end

    @testset "windowed gain evaluation for plots" begin
        # The plot entry points read a spectrum as `gains(sol; Ti = ti)` and a
        # time series as `gains(sol; Frequency = ci)` — an integer selector
        # windows the evaluation to that sample and drops the dimension.
        g = gains(sol; Ti = 1)
        @test size(g) == (length(sol.geom.channel_freqs), length(sol.geom.stations), 2)
        @test eltype(g) <: Complex
        @test lookup(g, Gustavo.Frequency) == sol.geom.channel_freqs

        gt = gains(sol; Frequency = 1)
        @test size(gt) == (length(sol.geom.times), length(sol.geom.stations), 2)
        @test lookup(gt, Ti) == sol.geom.times

        @test_throws BoundsError gains(sol; Ti = 10_000)
    end

    @testset "station codes available for plot labels" begin
        # Spectrum/phase plots read codes from the solution's geometry.
        @test sol.geom.stations == ["A1", "A2", "A3", "A4"]
    end

    scan_average(data) = XRadio.average(data, XRadio.ByScan())
    before = FP.baseline_spectra(scan_average(ps))
    after = FP.baseline_spectra(scan_average(calibrate(sol[:fringe], ps; flag_bad = false, apply_flags = false)))

    @testset "baseline spectra before/after" begin
        @test keys(before) == (:vis, :weight)
        pairs = collect(lookup(before, AntennaPair))
        @test all(((a, b),) -> a != b, pairs)
        @test length(pairs) == 6
        @test collect(lookup(before, FeedPair)) == [(1, 1), (1, 2), (2, 1), (2, 2)]
        @test collect(lookup(before, Gustavo.Scan)) == sol.geom.scan_names
        @test size(before.vis) == (1, 6, 4, length(sol.geom.channel_freqs))
        @test lookup(before, Frequency) == sol.geom.channel_freqs

        # Dividing out the fringe solution aligns the per-channel phases, so the
        # concentration R = |Σe^{iφ}|/N over frequency drops on no cross baseline.
        concentration(z) = (v = filter(isfinite, z); isempty(v) ? 0.0 : abs(sum(cis, angle.(v))) / length(v))
        for pr in pairs
            sel = (Gustavo.Scan(1), AntennaPair(At(pr)), FeedPair(At((1, 1))))
            @test concentration(after.vis[sel]) >= concentration(before.vis[sel]) - 1.0e-6
        end

        @test_throws "average over time first" FP.baseline_spectra(ps)
        ps2, _ = _build_fringe_ps(; nscans = 2)
        two = FP.baseline_spectra(scan_average(ps2))
        @test length(lookup(two, Gustavo.Scan)) == 2
    end

    @testset "delay closure" begin
        mx(v) = maximum(abs, filter(isfinite, v); init = 0.0)
        τscale = mx(FP.baseline_delays(before))
        @test τscale > 0                                  # the fixture injects real delays
        c = FP.delay_closure(before)
        @test collect(lookup(c, FeedPair)) == [(1, 1), (2, 2)]
        @test collect(lookup(c, FP.Triangle)) == [("A1", "A2", "A3"), ("A1", "A2", "A4"), ("A1", "A3", "A4"), ("A2", "A3", "A4")]
        # Station-based delays cancel around a triangle; a correct delay solution
        # removes the delay on every baseline and introduces no closure error.
        @test mx(c) < 0.05 * τscale
        @test mx(FP.baseline_delays(after)[FeedPair(At([(1, 1), (2, 2)]))]) < 0.05 * τscale
        @test mx(FP.delay_closure(after)) < 0.05 * τscale
    end

    @testset "baseline spectra plot with DimensionalData's recipes" begin
        fig = series(angle.(after.vis[Gustavo.Scan(1), FeedPair(At((1, 1)))]))
        @test (show(IOBuffer(), MIME("image/png"), fig); true)
    end

    @testset "plot_gain_phases" begin
        fig = plot_gain_phases(gains(sol; Ti = 1))
        @test (show(IOBuffer(), MIME("image/png"), fig); true)
        @test !isnothing(plot_gain_phases(gains(sol; Frequency = 1, AntennaName = At(["A1", "A2"]))))
        @test !isnothing(plot_gain_phases(gains(sol; Frequency = Near(sol.geom.f0))))
        grid = Figure()
        @test !isnothing(plot_gain_phases(grid[1, 2], gains(sol; Ti = 1, Feed = 1:1)))
        @test_throws "AntennaName, Feed and one more dimension" plot_gain_phases(gains(sol))
    end

    @testset "per-scan SNR with DimensionalData's recipes" begin
        fig = plot(FP.fringe_snr_table(sol).max_snr)
        @test (show(IOBuffer(), MIME("image/png"), fig); true)
    end

    det = FP.fringe_detections(sol)
    labels(m) = map(d -> only(lookup(d)), DimensionalData.refdims(m.snr))
    recorded(scan, pair, feeds) = det[Gustavo.Scan(At(scan)), AntennaPair(At(pair)), FeedPair(At(feeds))]

    @testset "fringe_search_map (delay–rate surface)" begin
        m = FP.fringe_search_map(sol, ps)
        @test m isa FP.FringeSearchMap
        scan, pair, feeds = labels(m)
        @test scan == only(sol.geom.scan_names)
        # Default: the scan's strongest measured cell; the map's peak is that
        # recorded detection, with the pfa over the scan's search family.
        rec = recorded(scan, pair, feeds)
        @test rec.snr == maximum(filter(!isnan, det.snr))
        @test m.detection.snr ≈ rec.snr rtol = 1.0e-6
        @test m.detection.delay ≈ rec.delay rtol = 1.0e-6
        @test m.detection.rate ≈ rec.rate rtol = 1.0e-6
        @test m.ncells == only(sol.steps[:fringe].scan_ncells)
        @test m.pfa == FP.fringe_pfa(m.detection.snr, m.ncells)
        @test m.pfa < 1.0e-6
        # The map's discrete peak sits at the detection within a grid bin.
        delays, rates = lookup(m.snr, FP.FringeDelay), lookup(m.snr, FP.FringeRate)
        pk = argmax(m.snr)
        @test isapprox(delays[pk[1]], m.detection.delay; atol = delays[2] - delays[1])
        @test isapprox(rates[pk[2]], m.detection.rate; atol = rates[2] - rates[1])

        # Selectors: baseline in either order, either one alone.
        m2 = FP.fringe_search_map(sol, ps; baseline = reverse(pair), feeds)
        @test labels(m2) == (scan, pair, feeds)
        @test m2.detection.snr ≈ m.detection.snr rtol = 1.0e-10
        other = first(p for p in lookup(det, AntennaPair) if p != pair)
        @test labels(FP.fringe_search_map(sol, ps; baseline = other))[2] == other
        @test labels(FP.fringe_search_map(sol, ps; feeds = (2, 2)))[3] == (2, 2)

        # The weakest accepted detection reproduces through its map.
        I = argmin(ifelse.(det.detected, det.snr, Inf))
        weak = (lookup(det, AntennaPair)[I[2]], lookup(det, FeedPair)[I[3]])
        mw = FP.fringe_search_map(sol, ps; baseline = weak[1], feeds = weak[2])
        @test mw.detection.snr ≈ recorded(scan, weak...).snr rtol = 1.0e-6

        @test_throws "is not a cross baseline" FP.fringe_search_map(sol, ps; baseline = ("A1", "A1"), feeds = (1, 1))
        @test_throws "holds no feed pair" FP.fringe_search_map(sol, ps; baseline = pair, feeds = (1, 3))
        @test_throws "measured no cell" FP.fringe_search_map(sol, ps; baseline = ("A1", "nope"))
        @test_throws "the solution has no :fringe step" FP.fringe_search_map(sol[:bandpass], ps)
        ps2, _ = _build_fringe_ps(; nscans = 2)
        @test_throws "must hold one scan" FP.fringe_search_map(sol, ps2)
    end

    @testset "fringe_search_map after a hierarchical (MBD) search" begin
        # The solve's detection comes from the hierarchical search; the plotted
        # surface is always the full grid, and its peak must sit on that detection.
        search = FP.FringeSearch(algorithm = FP.HierarchicalMBD())
        gc = FP._GroupCells(ps, sol.geom)
        @test !isnothing(FP._search_axes(gc.freqs, gc.times, search, ComplexF32).mbd)
        mbd = fit(BaselineFringeFit(; gauge, search), ps)
        mdet = FP.fringe_detections(mbd)
        m = FP.fringe_search_map(mbd, ps)
        rec = mdet[Gustavo.Scan(At(labels(m)[1])), AntennaPair(At(labels(m)[2])), FeedPair(At(labels(m)[3]))]
        @test m.detection.snr ≈ rec.snr rtol = 1.0e-6
        @test m.detection.delay ≈ rec.delay rtol = 1.0e-6
        @test m.detection.rate ≈ rec.rate rtol = 1.0e-6
        delays, rates = lookup(m.snr, FP.FringeDelay), lookup(m.snr, FP.FringeRate)
        pk = argmax(m.snr)
        @test isapprox(delays[pk[1]], m.detection.delay; atol = delays[2] - delays[1])
        @test isapprox(rates[pk[2]], m.detection.rate; atol = rates[2] - rates[1])
    end

    @testset "plot_fringe_search smoke" begin
        m = FP.fringe_search_map(sol, ps)
        figm = FP.plot_fringe_search(m)
        @test (show(IOBuffer(), MIME("image/png"), figm); true)
        map_axis(f) = only(filter(c -> c isa Axis && c.xlabel[] == "delay (ns)", contents(f.layout)))
        title_axis(f) = only(filter(c -> c isa Axis && !isempty(c.title[]), contents(f.layout)))
        @test occursin(join(labels(m)[2], "–"), title_axis(figm).title[])
        fig = Figure(size = (900, 700))
        @test !isnothing(FP.plot_fringe_search(fig[1, 1], m))
        unlabeled = FP.FringeSearchMap(DimensionalData.rebuild(m.snr; refdims = ()), m.detection, m.ncells, m.pfa)
        @test !isnothing(FP.plot_fringe_search(unlabeled))
        @test (show(IOBuffer(), MIME("image/png"), heatmap(m.snr)); true)

        # Zoom: the default view is a window around the peak, `false` the whole
        # searched plane, a number that span in main-lobe widths.
        @test !isnothing(FP.plot_fringe_search(m; zoom = 30))
        @test_throws "zoom must be positive" FP.plot_fringe_search(m; zoom = 0)
        figfull = FP.plot_fringe_search(m; zoom = false)
        show(IOBuffer(), MIME("image/png"), figfull)          # lay out, so limits are final
        zoomed = map_axis(figm).finallimits[]
        full = map_axis(figfull).finallimits[]
        @test zoomed.widths[1] < full.widths[1]
        @test zoomed.widths[2] < full.widths[2]
        @test zoomed.origin[1] <= m.detection.delay * 1.0e9 <= zoomed.origin[1] + zoomed.widths[1]
        @test zoomed.origin[2] <= m.detection.rate * 1.0e3 <= zoomed.origin[2] + zoomed.widths[2]
    end

    @testset "coherence report" begin
        raw = FP.coherence_report(ps)
        coh = FP.coherence_report(calibrate(sol, ps; flag_bad = false, apply_flags = false))
        @test keys(coh) == (:time, :freq, :time_pooled, :freq_pooled)
        @test map(DimensionalData.basetypeof, dims(coh.time)) == (Gustavo.Scan, AntennaPair, FeedPair, FP.AveragingTime)
        @test map(DimensionalData.basetypeof, dims(coh.freq_pooled)) == (Gustavo.Scan, FeedPair, FP.AveragingBandwidth)
        @test collect(lookup(coh, Gustavo.Scan)) == sol.geom.scan_names
        @test length(lookup(coh, AntennaPair)) == 6
        @test collect(lookup(coh, FeedPair)) == [(1, 1), (1, 2), (2, 1), (2, 2)]
        dts = collect(lookup(coh, FP.AveragingTime))
        times = sol.geom.times
        @test first(dts) ≈ times[2] - times[1]
        @test all(dts[2:end] .== 2 .* dts[1:(end - 1)])
        @test last(dts) > last(times) - first(times) >= dts[end - 1]

        # Native resolution (one sample per bin) is the η ≡ 1 anchor.
        @test all(coh.time_pooled[FP.AveragingTime(1)] .≈ 1)
        @test all(coh.freq_pooled[FP.AveragingBandwidth(1)] .≈ 1)

        # The solution removes the rates, so the corrected parallel hands stay
        # coherent averaged over the whole scan while the raw data decorrelate;
        # averaging over the band is no worse.
        par = FeedPair(At([(1, 1), (2, 2)]))
        full_t(c) = c.time_pooled[par, FP.AveragingTime(length(dts))]
        full_f(c) = c.freq_pooled[par, FP.AveragingBandwidth(length(lookup(c, FP.AveragingBandwidth)))]
        @test all(>(0.9), full_t(coh))
        @test all(full_t(raw) .< 0.95)
        @test all(full_t(raw) .< full_t(coh))
        @test all(full_f(coh) .>= full_f(raw) .- 1.0e-6)

        # Explicit intervals; per-scan results combine along `Scan`.
        given = FP.coherence_report(ps; timescales = [30.0, 120.0, 360.0], bandwidths = [4.0e6, 1.6e7])
        @test lookup(given, FP.AveragingTime) == [30.0, 120.0, 360.0]
        @test lookup(given, FP.AveragingBandwidth) == [4.0e6, 1.6e7]
        ps2, _ = _build_fringe_ps(; nscans = 2)
        per_scan = [
            FP.coherence_report(g; timescales = [30.0, 120.0], bandwidths = [4.0e6])
                for g in values(XRadio.groupby(ps2, XRadio.ByScan()))
        ]
        both = FP.cat_scans(per_scan)
        @test length(lookup(both, Gustavo.Scan)) == 2
        @test size(both.time_pooled) == (2, 4, 2)
        @test_throws "must hold one scan" FP.coherence_report(ps2)
    end

    @testset "plot_coherence" begin
        coh = FP.coherence_report(calibrate(sol, ps; flag_bad = false, apply_flags = false))
        one = coh[Gustavo.Scan(1), FeedPair(At((1, 1)))]
        fig = FP.plot_coherence(one)
        @test (show(IOBuffer(), MIME("image/png"), fig); true)
        @test !isnothing(FP.plot_coherence(one; nlabel = 3))
        @test !isnothing(FP.plot_coherence(Figure(size = (1000, 420))[1, 1], one))
        @test_throws "select them first" FP.plot_coherence(coh)
        # Antenna pairs worst first, by indexing.
        full = one.time[FP.AveragingTime(length(lookup(one, FP.AveragingTime)))]
        worst = full[AntennaPair(sortperm(collect(full)))]
        @test issorted(collect(worst))
        @test Set(lookup(worst, AntennaPair)) == Set(lookup(one, AntennaPair))
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
            FP._coherence_accumulate!(
                zeros(0, 1), numF, den, dvar, npts, V, W, nothing, [1], [1], [0.0], freqs,
                Float64[], [2.0e8], debias,
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
        FP._coherence_accumulate!(nT, zeros(1, 1), den, zeros(1), zeros(Int, 1), V, W, nothing, [1], [1], times, freqs, dts, [1.0e8], true)
        eta_perchan = eta(nT, den)
        @test eta_perchan > 0.9
        # marginalized: band-average per AP, then time η — same estimand,
        # measured on high-SNR samples, so lower variance.
        Vt, Wt = FP._collapse_axis(V, W, falses(size(V)), 1)
        @test size(Vt) == (1, nti, 1, 1)
        nT2 = zeros(1, 1); denT = zeros(1)
        FP._coherence_accumulate!(nT2, zeros(1, 1), denT, zeros(1), zeros(Int, 1), Vt, Wt, nothing, [1], [1], times, [1.5e9], dts, [1.0], true)
        eta_marg = eta(nT2, denT)
        @test eta_marg > 0.95               # ≈ 1 for a flat-phase source
    end
end

@testset "freq_group_coherence: per-group coherence + group splitting" begin
    # 3 bands of 4 channels with gaps; flat phase gives η = 1, a ramp less.
    freqs = vcat(1.0e9 .+ (0:3) .* 1.0e6, 1.1e9 .+ (0:3) .* 1.0e6, 1.3e9 .+ (0:3) .* 1.0e6)
    @test FP._freq_group_ranges(freqs) == [1:4, 5:8, 9:12]
    d = (FP._scan_dim(["1"]), FP._station_pair_dim([("A", "B")]), FeedPair([(1, 1)]), Frequency(freqs))
    spectra(z) = DimStack((; vis = DimArray(reshape(z, 1, 1, 1, :), d)))
    ramp = FP.freq_group_coherence(spectra(ComplexF64[cis(2π * c / 6) for c in eachindex(freqs)]))
    flat = FP.freq_group_coherence(spectra(fill(1.0 + 0.0im, length(freqs))))
    @test map(DimensionalData.basetypeof, dims(flat)) == (Gustavo.Scan, FeedPair, FP.FreqGroup)
    @test collect(lookup(flat, FP.FreqGroup)) == [(freqs[1], freqs[4]), (freqs[5], freqs[8]), (freqs[9], freqs[12])]
    @test all(≈(1.0), flat)
    @test all(<(0.9), ramp)

    # Band GROUPS: comparable inter-block gaps merge into one group; a far-away
    # block splits off its own group (the VGOS 3/5/6/10 GHz situation).
    @test FP.fringe_freq_groups(freqs) == [1:12]
    freqs2 = vcat(freqs, 5.0e9 .+ (0:3) .* 1.0e6)
    @test FP.fringe_freq_groups(freqs2) == [1:12, 13:16]
    @test FP.fringe_freq_groups(freqs2[1:1]) == [1:1]
end

@testset "baseline spectra carry the averages' widths" begin
    noise = 0.05
    w = 2 / noise^2                       # the fixture's weight per real component
    nti = 6
    ps, _ = _build_fringe_ps(; nant = 4, nspw = 2, nchan = 8, ntime = nti, noise)
    fr = fit(BaselineFringeFit(; gauge = PinAntenna(1)), ps)
    corrected = calibrate(fr, ps; flag_bad = false, apply_flags = false)
    spectra = FP.baseline_spectra(XRadio.average(corrected, XRadio.ByScan()))

    # Each spectral point averages a baseline over `nti` integrations.
    @test all(≈(nti * w), FP.baseline_spectra(XRadio.average(ps, XRadio.ByScan())).weight)

    # Corrected visibilities on a baseline scatter about their mean by 1/√weight
    # per real component. Pooled over every baseline and channel this is a tight
    # check even with few samples.
    z = Float64[]
    for pr in lookup(spectra, AntennaPair)
        sel = (Gustavo.Scan(1), AntennaPair(At(pr)), FeedPair(At((1, 1))))
        col = collect(spectra.vis[sel])
        σ = inv.(sqrt.(collect(spectra.weight[sel])))
        μ = sum(col) / length(col)
        append!(z, real.(col .- μ) ./ σ, imag.(col .- μ) ./ σ)
    end
    @test length(z) > 100
    @test 0.5 < sqrt(sum(abs2, z) / length(z)) < 2.0
end
