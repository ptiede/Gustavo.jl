# End-to-end fringe-fitter pipeline test. Builds a small synthetic multi-band
# in-memory UVSet with KNOWN injected per-(station, feed) delay/rate/constant
# phase plus a per-(station, feed, AP) atmospheric phase screen, then verifies
# that `fit` → `apply_calibration` flattens the residual baseline
# phases (the coherence test) and that save/load round-trips.

# Shared synthetic-UVSet generator + usings/aliases (CAL/FP/UVP).
include("synthetic_uvset.jl")


@testset "Fringe pipeline end-to-end" begin
    uvset, _truth = _build_fringe_uvset()

    # Adhoc smoother window 7 (< the 12-AP scan) tracks the screen; snr_floor 0
    # keeps every well-determined AP in this high-SNR synthetic.
    sol = fit(
        FringeFit(model = FringeModel()) |> BandpassEstimator() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset,
    )
    @test sol isa CAL.CalibrationSolution
    for step in sol.steps
        @test length(step.θ) == step.layout.nθ
    end

    corr = Gustavo.apply_calibration(uvset, sol)

    @testset "parallel-hand coherence ≈ 1" begin
        worst = 1.0
        for (_, leaf) in DimensionalData.branches(corr)
            V = parent(leaf[:vis])
            W = parent(leaf[:weights])
            bl_pairs = UVP.baselines(leaf).pairs
            lp = pol_products(leaf)
            for p in eachindex(lp)
                Gustavo.Calibration.is_parallel_hand(lp[p]) || continue
                for bi in eachindex(bl_pairs)
                    a, b = bl_pairs[bi]
                    a == b && continue
                    coh = _coherence(@view(V[:, :, bi, p]), @view(W[:, :, bi, p]))
                    worst = min(worst, coh)
                    @test coh > 0.99
                end
            end
        end
        @info "worst parallel-hand coherence" worst
    end

    @testset "cross-hand coherence ≈ 1" begin
        # Source is unpolarized here, so cross hands should also flatten.
        for (_, leaf) in DimensionalData.branches(corr)
            V = parent(leaf[:vis])
            W = parent(leaf[:weights])
            bl_pairs = UVP.baselines(leaf).pairs
            lp = pol_products(leaf)
            for p in eachindex(lp)
                Gustavo.Calibration.is_parallel_hand(lp[p]) && continue
                for bi in eachindex(bl_pairs)
                    a, b = bl_pairs[bi]
                    a == b && continue
                    coh = _coherence(@view(V[:, :, bi, p]), @view(W[:, :, bi, p]))
                    @test coh > 0.99
                end
            end
        end
    end

    @testset "save / load round-trip" begin
        path = tempname()
        CAL.save_solution(path, sol)
        sol2 = CAL.load_solution(path)
        @test sol2.geom.times == sol.geom.times
        @test sol2.geom.channel_freqs == sol.geom.channel_freqs
        @test length(sol2.steps) == length(sol.steps)
        for step in sol.steps
            step2 = CAL._step(sol2, step.name)
            @test step2.θ == step.θ
            @test step2.layout.nθ == step.layout.nθ
            @test length(CAL.phase_components(step2.model)) == length(CAL.phase_components(step.model))
        end

        corr2 = Gustavo.apply_calibration(uvset, sol2)
        for (k, leaf) in DimensionalData.branches(corr)
            V = parent(leaf[:vis])
            V2 = parent(DimensionalData.branches(corr2)[k][:vis])
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x == y, zip(V, V2))
        end
        rm(path; force = true)
    end

    @testset "eager input is left unmodified" begin
        # An eager leaf is the caller's own data, so `apply_calibration` corrects
        # a copy. The synthetic input carries no NaNs, so `==` is exact.
        snap = Dict(
            k => (copy(parent(l[:vis])), copy(parent(l[:weights])))
                for (k, l) in DimensionalData.branches(uvset)
        )
        Gustavo.apply_calibration(uvset, sol)
        for (k, l) in DimensionalData.branches(uvset)
            @test parent(l[:vis]) == snap[k][1]
            @test parent(l[:weights]) == snap[k][2]
        end
    end

    @testset "standard interfaces: show and iteration" begin
        # `show` gives each type its own summary line instead of a raw dump.
        @test occursin("CalibrationSolution", sprint(show, sol))
        @test occursin("CalibrationSolution", sprint(show, MIME"text/plain"(), sol))
        @test occursin("StationGainModel", sprint(show, CAL._step(sol, :fringe).model))

        # CalibrationPipeline is an ordered container over its steps.
        pipe = CalibrationPipeline(FringeFit(model = FringeModel()) |> BandpassEstimator())
        @test length(pipe) == length(pipe.steps)
        @test eltype(typeof(pipe)) == CalibrationStep
        @test collect(pipe) == pipe.steps
        @test pipe[1] === pipe.steps[1]
        @test [s for s in pipe] == pipe.steps
        @test occursin("CalibrationPipeline", sprint(show, pipe))
        @test occursin("CalibrationPipeline", sprint(show, MIME"text/plain"(), pipe))

        # ScanStream is an ordered container over its scan-group specs.
        stream = ST.scan_stream(uvset)
        @test length(stream) == length(stream.groups)
        @test eltype(typeof(stream)) == eltype(stream.groups)
        @test collect(stream) == stream.groups
        @test first(stream) === stream.groups[1]
        @test occursin("ScanStream", sprint(show, stream))
    end
end

# `_correct_column!` overwrites its `vis`/`w` in place while reading them, so a
# private (lazy-materialized) leaf can be corrected without an output copy. The
# aliased read-then-write must land the same values as an out-of-place fold.
@testset "gain correction folds in place exactly" begin
    rng = MersenneTwister(0xC011)
    nchan, nti = 4, 3
    vis = rand(rng, ComplexF64, nchan, nti)
    w = rand(rng, nchan, nti)
    ga = rand(rng, ComplexF64, nchan, nti) .+ 1   # magnitudes well above _GAIN_FLOOR
    gb = rand(rng, ComplexF64, nchan, nti) .+ 1
    ga[1] = 0                                       # force the degenerate branch
    vis0, w0 = copy(vis), copy(w)
    CAL._correct_column!(vis, w, ga, gb)
    for i in eachindex(vis0)
        den = ga[i] * conj(gb[i])
        if abs(ga[i]) < CAL._GAIN_FLOOR || abs(gb[i]) < CAL._GAIN_FLOOR || !isfinite(den)
            @test isnan(vis[i]) && iszero(w[i])
        else
            @test vis[i] ≈ vis0[i] / den
            @test w[i] ≈ w0[i] * abs2(ga[i] * gb[i])
        end
    end
end

@testset "Rate & adhoc are feed-tied (SharedFeeds regression)" begin
    # The fringe model MUST tie the per-scan RATE and the per-AP ADHOC phase across
    # feeds (`SharedFeeds`). Both are feed-common physics: the fringe rate is shared
    # by the two feeds and the residual atmospheric screen is non-birefringent. A
    # revert to `PerFeed` lets a spurious R–L rate (rate₂ − rate₁) / adhoc phase
    # float on noise and — multiplied by the whole-track Rate lever arm
    # `2π·rate·(t − t0_global)`, hours long — inject large, arbitrary scan-to-scan
    # R–L (RL/RR) phase jumps (see the rationale comment on `_fringe_model` in
    # model_plans.jl and the `fringe-rate-must-be-sharedfeeds` decision).
    #
    # The end-to-end coherence test injects FEED-COMMON truth, so a `PerFeed` revert
    # would still recover it at high SNR and pass — it does NOT guard this decision.
    # Assert the tying structurally (the model) AND that the layout realises it (both
    # feeds share one θ column per (station, time-seg), so the solved R–L is ≡ 0).
    model = FP._fringe_model()
    phase = CAL.phase_components(model)

    rate_i = findfirst(tc -> tc.component.term isa CAL.Rate, phase)
    @test rate_i !== nothing
    @test phase[rate_i].tying isa CAL.SharedFeeds
    @test !(phase[rate_i].tying isa CAL.PerFeed)

    adhoc_i = findfirst(tc -> tc.component.time isa CAL.PerIntegration, phase)
    @test adhoc_i !== nothing
    @test phase[adhoc_i].tying isa CAL.SharedFeeds
    @test !(phase[adhoc_i].tying isa CAL.PerFeed)

    # Layout: `SharedFeeds` folds both feeds to ONE node (θ column), `PerFeed` keeps
    # two distinct ones. So feed-1 and feed-2 share every column iff the tie holds —
    # a `PerFeed` revert breaks this on any solved (station, seg).
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)
    first_leaf = first(values(DimensionalData.branches(uvset)))
    nant = length(UVP.metadata(first_leaf).antennas)
    layout = CAL.plan_parameters(model, nant, geom)

    for (ci, label) in ((rate_i, "rate"), (adhoc_i, "adhoc"))
        off1 = plan_off1(layout.plans[ci])                 # (ant, feed, ntseg, nfseg)
        @test off1[:, 1, :, :] == off1[:, 2, :, :]   # both feeds → same θ columns
        @test any(!=(0), off1[:, 1, :, :])           # ...and the plan is non-trivial
    end
end

@testset "Fused fitcalibrate ≡ two-pass fit + apply" begin
    # The fused single-pass driver must produce the SAME result as the explicit
    # two-pass `apply_calibration(uvset, fit(pipe, uvset))`, because each
    # leaf's gains depend only on its own (disjoint) θ slots.
    uvset, _ = _build_fringe_uvset()
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    chain = FringeFit(model = FringeModel()) |> BandpassEstimator() |>
        TemporalSmoother(adhoc)

    sol_ref = fit(chain, uvset)
    corr_ref = Gustavo.apply_calibration(uvset, sol_ref)

    # No reduce steps → fused correction only (no reduction).
    sol_fused, out_fused = fitcalibrate(chain, uvset)

    @test parent(gains(sol_fused)) ≈ parent(gains(sol_ref))
    # Same tree keys.
    @test Set(keys(DimensionalData.branches(out_fused))) ==
        Set(keys(DimensionalData.branches(corr_ref)))
    # Corrected visibilities/weights per leaf agree to Float32 precision
    # (NaN-aware) — NOT bit-identical: the two-pass reference divides by ONE
    # combined gain (`apply_calibration`), while the fused path composes
    # earlier steps' corrections through the transform chain as SEPARATE
    # sequential divisions (mathematically the same total gain, different
    # floating-point rounding).
    for (k, leaf) in DimensionalData.branches(corr_ref)
        Vr = parent(leaf[:vis]); Wr = parent(leaf[:weights])
        lf = DimensionalData.branches(out_fused)[k]
        Vf = parent(lf[:vis]); Wf = parent(lf[:weights])
        @test size(Vf) == size(Vr)
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || isapprox(x, y; rtol = 1.0e-5), zip(Vr, Vf))
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || isapprox(x, y; rtol = 1.0e-5), zip(Wr, Wf))
    end

    # With a reducer the fused output must equal applying the same reducer to the
    # two-pass corrected set.
    red_ref = UVP.frequency_average(corr_ref; nout = 1)
    _, out_red = fitcalibrate(chain, uvset; reduce = [AverageFrequency(nout = 1)])
    for (k, leaf) in DimensionalData.branches(red_ref)
        Vr = parent(leaf[:vis])
        Vf = parent(DimensionalData.branches(out_red)[k][:vis])
        @test size(Vf) == size(Vr)
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
    end
end

@testset "Cross-run step composition ≡ within-run pipeline" begin
    # A multi-step pipeline's within-run composition (`_run_pipeline` appends
    # each finished step's solution as an `ApplySolution` before the next
    # step's pass) is the SAME mechanism a caller invokes by hand across
    # separate `fit` calls via `step_solution`/`ApplySolution`. Fitting `A |>
    # B` in one call must solve the SAME B-step θ as fitting `A` alone, then
    # fitting `ApplySolution(step_solution(sol_a, :fringe)) |> B` in a later,
    # unrelated call.
    uvset, _ = _build_fringe_uvset()
    ff = FringeFit(model = FringeModel())
    bp = BandpassEstimator()

    sol_within = fit(ff |> bp, uvset)

    sol_a = fit(ff, uvset)
    sol_cross = fit(FP.ApplySolution(CAL.step_solution(sol_a, :fringe)) |> bp, uvset)

    θ_within = CAL.step_solution(sol_within, :bandpass).steps[1].θ
    θ_cross = CAL.step_solution(sol_cross, :bandpass).steps[1].θ
    @test θ_within == θ_cross
end

@testset "combine_spw: spws → Frequency axis" begin
    uvset, _ = _build_fringe_uvset(nspw = 3, nchan = 4)
    avg = UVP.frequency_average(uvset; nout = 1)          # each spw → 1 channel
    @test length(UVP.union_frequency_axis(avg)) == 3      # 3 distinct spw setups

    combined = UVP.combine_spw(avg)
    @test length(DimensionalData.branches(combined)) == 1 # one leaf per (src,scan)
    setups = UVP.union_frequency_axis(combined)
    @test length(setups) == 1                             # single FREQID
    cf = collect(channel_freqs(first(setups)))
    @test length(cf) == 3                                 # 3 channels
    @test issorted(cf)                                    # ascending channel freqs
    # The combined channel frequencies are exactly the per-spw averaged centers.
    spw_centers = sort([only(channel_freqs(fs)) for fs in UVP.union_frequency_axis(avg)])
    @test cf ≈ spw_centers
    leaf = first(values(DimensionalData.branches(combined)))
    @test size(parent(leaf[:vis]), 1) == 3                # Frequency axis = 3 channels
end

@testset "write_uvfits on FITS-IDI-style UVSet (synthesized primary cards)" begin
    # `_build_fringe_uvset` registers NO primary cards, so write_uvfits must
    # synthesize them. Reduce + combine spws to channels, then round-trip via UVFITS.
    uvset, _ = _build_fringe_uvset(nspw = 2, nchan = 6)
    sol, reduced = fitcalibrate(
        FringeFit(model = FringeModel()) |> BandpassEstimator() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset;
        reduce = [AverageFrequency(nout = 1), CombineSpw(), AverageTime(seconds = 0.02)],
    )
    @test length(DimensionalData.branches(reduced)) == 1
    @test length(UVP.union_frequency_axis(reduced)) == 1
    @test length(channel_freqs(first(UVP.union_frequency_axis(reduced)))) == 2

    path = tempname() * ".uvfits"
    try
        @test Gustavo.UVData.write_uvfits(path, reduced) == path
        @test isfile(path) && filesize(path) > 0
        rt = Gustavo.UVData.load_uvfits(path)
        @test length(DimensionalData.branches(rt)) == 1
        rt_setups = UVP.union_frequency_axis(rt)
        @test length(rt_setups) == 1
        @test length(channel_freqs(first(rt_setups))) == 2          # 2 channels survive
        @test channel_freqs(first(rt_setups)) ≈ channel_freqs(first(UVP.union_frequency_axis(reduced)))
        rt_leaf = first(values(DimensionalData.branches(rt)))
        @test Set(String.(pol_products(rt_leaf))) == Set(["PP", "PQ", "QP", "QQ"])
        @test all(isfinite, filter(isfinite, parent(rt_leaf[:vis])))  # no NaN explosion
    finally
        isfile(path) && rm(path; force = true)
    end
end

@testset "write_uvfits convention toggle (:aips conjugates, :fitsidi verbatim)" begin
    # :aips writes conj(vis); :fitsidi writes vis verbatim; (u,v,w) never negated.
    # load_uvfits always conjugates on read (assumes a standard :aips file), so the
    # :fitsidi round-trip comes back as the conjugate of the :aips round-trip — and
    # since both files traverse the identical write/load path, their records align
    # exactly and differ ONLY by that conjugation. This isolates the convention.
    uvset, _ = _build_fringe_uvset(nspw = 2, nchan = 6)
    _, reduced = fitcalibrate(
        FringeFit(model = FringeModel()) |> BandpassEstimator() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset;
        reduce = [AverageFrequency(nout = 1), CombineSpw(), AverageTime(seconds = 0.02)],
    )
    pa = tempname() * ".uvfits"
    pf = tempname() * ".uvfits"
    try
        @test UVP.write_uvfits(pa, reduced; convention = :aips) == pa
        @test UVP.write_uvfits(pf, reduced; convention = :fitsidi) == pf
        la = first(values(DimensionalData.branches(UVP.load_uvfits(pa))))
        lf = first(values(DimensionalData.branches(UVP.load_uvfits(pf))))
        Va = parent(la[:vis])
        Vf = parent(lf[:vis])
        finite = findall(v -> isfinite(real(v)) && isfinite(imag(v)), Va)
        @test !isempty(finite)
        @test Vf[finite] ≈ conj.(Va[finite])                      # convention flip
        @test any(i -> abs(imag(Va[i])) > 1.0f-6, finite)         # non-trivial imag
        @test parent(lf[:uvw]) ≈ parent(la[:uvw])                 # (u,v,w) not negated
        @test_throws ErrorException UVP.write_uvfits(pa, reduced; convention = :bogus)
    finally
        isfile(pa) && rm(pa; force = true)
        isfile(pf) && rm(pf; force = true)
    end
end

@testset "Residual accumulation is a plain weighted mean of pre-corrected data" begin
    # `_accumulate_leaf_rbar!`/`_accumulate_leaf_band_phasor!` no longer divide
    # out a gain themselves — the pipeline's transform chain (`_divide_gains!`,
    # transforms.jl) corrects `V`/`W` BEFORE these kernels ever see them,
    # including the |g|² weight reweighting (Var(V/g) = 1/(w·|g|²); this is
    # what once tripled K2's adhoc track noise on VR2505 when a low-|g| amp
    # channel's noise was up-weighted by a RAW, un-reweighted `w`). So feeding
    # these kernels ALREADY-reweighted (V, W) must give back exactly that
    # inverse-variance mean, with no further correction applied.
    nchan, nti, nbl, npol = 2, 1, 1, 1
    V = zeros(ComplexF32, nchan, nti, nbl, npol)
    W = zeros(Float32, nchan, nti, nbl, npol)
    V[1, 1, 1, 1] = 1.0 + 0.0im          # channel 1: unit gain ⇒ unchanged, weight 1
    V[2, 1, 1, 1] = 10.0im               # channel 2: pre-corrected by |g|² = 0.01,
    W[1, 1, 1, 1] = 1.0                  # so its weight is reweighted to 0.0001
    W[2, 1, 1, 1] = 0.0001
    bl = [(1, 2)]
    rbar = zeros(ComplexF64, nbl, npol, nti)
    wbar = zeros(Float64, nbl, npol, nti)
    FP._accumulate_leaf_rbar!(rbar, wbar, V, W)
    @test wbar[1, 1, 1] ≈ 1.0001
    @test rbar[1, 1, 1] / wbar[1, 1, 1] ≈ (1.0 + 0.001im) / 1.0001
    z = zeros(ComplexF64, nbl, npol)
    wz = zeros(Float64, nbl, npol)
    FP._accumulate_leaf_band_phasor!(z, wz, V, W, bl, ["PP"])
    @test wz[1, 1] ≈ 1.0001
    @test z[1, 1] / wz[1, 1] ≈ (1.0 + 0.001im) / 1.0001
end

@testset "Fringe pipeline: rounds > 1 accumulates (no corruption)" begin
    # Regression for the θ-overwrite bug: even rounds previously wiped the
    # round-1 solution (coherence collapsed). Accumulation keeps all rounds good.
    uvset, _ = _build_fringe_uvset()
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    for r in (1, 2, 3)
        sol = fit(
            FringeFit(model = FringeModel(), estimator = FP.MatchedFilter(rounds = r)) |>
                BandpassEstimator() |> TemporalSmoother(adhoc),
            uvset,
        )
        corr = Gustavo.apply_calibration(uvset, sol)
        worst = 1.0
        for (_, leaf) in DimensionalData.branches(corr)
            V = parent(leaf[:vis])
            W = parent(leaf[:weights])
            bl_pairs = UVP.baselines(leaf).pairs
            lp = pol_products(leaf)
            for p in eachindex(lp)
                Gustavo.Calibration.is_parallel_hand(lp[p]) || continue
                for bi in eachindex(bl_pairs)
                    a, b = bl_pairs[bi]
                    a == b && continue
                    worst = min(worst, _coherence(@view(V[:, :, bi, p]), @view(W[:, :, bi, p])))
                end
            end
        end
        @test worst > 0.99
    end
end

@testset "build_geometry conflict detection (N1)" begin
    # Normal multi-band set: each channel frequency belongs to exactly one spw.
    uvset_ok, _ = _build_fringe_uvset()
    @test CAL.build_geometry(uvset_ok) isa CAL.DataGeometry

    # spw_sep = 0 ⇒ the two bands share identical channel frequencies but carry
    # distinct spw_names ("band_1"/"band_2"), so a single concatenated channel
    # axis cannot dense-rank a frequency to one spw — build_geometry must error
    # instead of silently last-write-wins.
    uvset_conflict, _ = _build_fringe_uvset(spw_sep = 0.0)
    @test_throws ArgumentError CAL.build_geometry(uvset_conflict)
    @test_throws "conflicting spectral windows" CAL.build_geometry(uvset_conflict)
end

@testset "Phase bandpass: per-channel phase recovered" begin
    # Inject a smooth per-(station, feed, channel) phase bandpass (ref ant 1 = 0)
    # on top of the usual delay/rate/phase/screen. The bandpass stage should
    # flatten the per-channel phase; with it OFF the bandpass survives uncorrected.
    nant, nspw, nchan = 4, 2, 8
    nchg = nspw * nchan
    rng = MersenneTwister(0xBA9D)
    bp = zeros(nant, 2, nchg)
    for a in 2:nant, f in 1:2
        off = (rand(rng) - 0.5)
        for gc in 1:nchg
            bp[a, f, gc] = off + 1.0 * sin(2π * gc / nchg + a + f)   # smooth shape + offset
        end
    end
    uvset, _ = _build_fringe_uvset(; nant = nant, nspw = nspw, nchan = nchan, bandpass = bp)
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    ff = FringeFit(model = FringeModel())
    sol_on = fit(ff |> BandpassEstimator(phase = true) |> TemporalSmoother(adhoc), uvset)
    sol_off = fit(ff |> BandpassEstimator(phase = false) |> TemporalSmoother(adhoc), uvset)

    don = FP.baseline_fringe_data(uvset, sol_on)
    doff = FP.baseline_fringe_data(uvset, sol_off)
    p = FP.baseline_pol_index(don, :parallel)
    # Per-channel phase coherence per cross baseline: R = |Σ_c V̄_c| / Σ_c |V̄_c|
    # (1 ⇒ flat per-channel phase). Mean over baselines.
    function freq_coh(spec)
        rs = Float64[]
        for bi in eachindex(don.bl_pairs)
            a, b = don.bl_pairs[bi]
            a == b && continue
            z = filter(isfinite, spec[:, bi, p])
            isempty(z) && continue
            push!(rs, abs(sum(z)) / sum(abs.(z)))
        end
        return sum(rs) / length(rs)
    end
    R_on = freq_coh(don.spec_after)
    R_off = freq_coh(doff.spec_after)
    @test R_on > R_off                  # the bandpass stage flattens per-channel phase
    @test R_on > 0.97                   # nearly flat after the bandpass
    @test R_off < 0.95                  # bandpass survives without the stage

    # Bandpass is station-based ⇒ triangle delay closure is unchanged by it.
    c_on = FP.delay_closure(don)
    c_off = FP.delay_closure(doff)
    mx(v) = (u = abs.(filter(isfinite, v)); isempty(u) ? 0.0 : maximum(u))
    @test isapprox(mx(c_on.closure_before), mx(c_off.closure_before); rtol = 0.2)
end

@testset "Phase bandpass: reference R–L shape solved (χ re-gauge)" begin
    # The per-channel bandpass solve pins the REFERENCE's feed-2 node per channel
    # (a rank device against the χ ↔ feed-2-offset degeneracy); the χ re-gauge in
    # `_solve_phase_bandpass!` must reassign χ's band-structure into the feed-2
    # block so the reference's relative R–L phase bandpass is SOLVED rather than
    # zeroed — otherwise it survives uncorrected in every cross-hand visibility.
    nant, nspw, nchan = 4, 2, 8
    nchg = nspw * nchan
    rng = MersenneTwister(0xEC9A)
    bp = zeros(nant, 2, nchg)
    # Reference (ant 1): feed 1 flat (the true per-channel gauge), feed 2 a smooth
    # NONZERO shape — the relative R–L phase bandpass under test. The shape is
    # orthogonalized against per-band constant + slope: those components are
    # legitimately (and feed-COMMONLY) absorbed by the per-band SBD delay/const
    # and R–L delay stages, so leaving them in would make the earlier solves fight
    # an unmodelled feed-2-only per-band ramp instead of exercising the χ
    # re-gauge this test is about.
    for b in 1:nspw
        cs = ((b - 1) * nchan + 1):(b * nchan)
        s = [0.6 * sin(2π * k / nchan + 0.9 * b) for k in 1:nchan]   # full period per band
        X = hcat(ones(nchan), collect(Float64, 1:nchan))
        bp[1, 2, cs] .= s .- X * (X \ s)                             # ⊥ {const, slope} per band
    end
    for a in 2:nant, f in 1:2
        off = (rand(rng) - 0.5)
        for gc in 1:nchg
            bp[a, f, gc] = off + 1.0 * sin(2π * gc / nchg + a + f)
        end
    end
    uvset, _ = _build_fringe_uvset(; nant = nant, nspw = nspw, nchan = nchan, bandpass = bp)
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    ff = FringeFit(model = FringeModel())
    sol_on = fit(ff |> BandpassEstimator(phase = true) |> TemporalSmoother(adhoc), uvset)
    sol_off = fit(ff |> BandpassEstimator(phase = false) |> TemporalSmoother(adhoc), uvset)

    don = FP.baseline_fringe_data(uvset, sol_on)
    doff = FP.baseline_fringe_data(uvset, sol_off)
    # Per-channel phase coherence R = |Σ_c V̄_c| / Σ_c |V̄_c| over a product set.
    # The source is unpolarized, so a correct solve leaves the corrected CROSS
    # spectra flat; a SIGN error in the re-gauge doubles the ref R–L ripple
    # instead of removing it, so the cross coherence is also the convention check.
    function freq_coh(spec, ps)
        rs = Float64[]
        for bi in eachindex(don.bl_pairs), p in ps
            a, b = don.bl_pairs[bi]
            a == b && continue
            z = filter(isfinite, spec[:, bi, p])
            isempty(z) && continue
            push!(rs, abs(sum(z)) / sum(abs.(z)))
        end
        return sum(rs) / length(rs)
    end
    cross_ps = findall(p -> (fp = CAL.correlation_feed_pair(p); fp[1] != fp[2]), don.pol_products)
    R_on = freq_coh(don.spec_after, cross_ps)
    R_off = freq_coh(doff.spec_after, cross_ps)
    @test R_on > R_off
    @test R_on > 0.97                   # ref R–L shape corrected in the cross hands
    @test R_off < 0.95                  # ...and survives with the stage off

    # θ recovery: the (ref, feed 2) bandpass slots carry the injected shape up
    # to per-band constant + slope components (the parts the per-band SBD and R–L
    # delay terms legitimately absorb). Remove the per-band best-fit constant +
    # slope from the difference; the residual shape must match.
    bp_on = CAL._step(sol_on, :bandpass)
    plan = FP._bandpass_plan(bp_on.model, bp_on.layout)
    @test plan_off1(plan)[1, 2, 1, 1] != 0
    rec = [bp_on.θ[plan_off1(plan)[1, 2, 1, plan.fseg_id[gc]]] for gc in 1:nchg]
    worst = 0.0
    for b in 1:nspw
        cs = ((b - 1) * nchan + 1):(b * nchan)
        d = [rem2pi(rec[gc] - bp[1, 2, gc], RoundNearest) for gc in cs]
        X = hcat(ones(nchan), collect(Float64, 1:nchan))
        worst = max(worst, maximum(abs.(d .- X * (X \ d))))
    end
    @test worst < 0.15

    # Parallel hands are invariant under the feed-2 common-mode re-gauge.
    @test freq_coh(don.spec_after, (FP.baseline_pol_index(don, :parallel),)) > 0.97
end

@testset "Amplitude bandpass: pluggable estimators (poly / penalized / free)" begin
    # Inject a per-BAND log-amp roll-off (the filterbank passband, deep toward each
    # band's high-channel edge) shared by all stations, plus a small per-station
    # ripple. THEN kill one interior channel per band (zero its weight on every
    # baseline). The two SMOOTH estimators (polynomial, penalized) must flatten the
    # band AND estimate the killed channels from the in-spw shape; free_bandpass must
    # leave the killed channels untouched (|g| = 1).
    nant, nspw, nchan = 4, 2, 8
    nchg = nspw * nchan
    rng = MersenneTwister(0x5A11)
    locof(gc) = (gc - 1) % nchan + 1                                # local channel within its band
    rolloff(gc) = -0.5 * ((locof(gc) - 1) / (nchan - 1))^2          # per-band: 0 → −0.5 toward high edge
    abp = zeros(nant, 2, nchg)
    for a in 1:nant, f in 1:2
        dev = a == 1 ? 0.0 : (rand(rng) - 0.5) * 0.3               # per-station offset (gauge removes it)
        for gc in 1:nchg
            wig = a == 1 ? 0.0 : 0.05 * sin(2π * gc / nchg + a + f) # small per-station ripple
            abp[a, f, gc] = rolloff(gc) + dev + wig
        end
    end
    uvset, _ = _build_fringe_uvset(; nant = nant, nspw = nspw, nchan = nchan, amp_bandpass = abp)

    # Kill local channel 7 in every band (globals 7 and 15): zero its weight on all
    # baselines so the per-channel solve has NO data there.
    dead_local = 7
    dead_globals = [(b - 1) * nchan + dead_local for b in 1:nspw]
    for (_, leaf) in DimensionalData.branches(uvset)
        parent(leaf[:weights])[dead_local, :, :, :] .= 0.0f0
    end

    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    larec(θ, plan, a, f, gc) = (off = plan_off1(plan)[a, f, 1, plan.fseg_id[gc]]; off == 0 ? NaN : θ[off])
    function amp_ripple(spec, don, p)
        rs = Float64[]
        for bi in eachindex(don.bl_pairs)
            a, b = don.bl_pairs[bi]
            a == b && continue
            m = abs.(filter(isfinite, spec[:, bi, p]))
            (isempty(m) || minimum(m) <= 0) && continue
            push!(rs, maximum(m) / minimum(m))
        end
        isempty(rs) ? NaN : sum(rs) / length(rs)
    end

    ff = FringeFit(model = FringeModel())
    sol_off = fit(ff |> BandpassEstimator(amp = false) |> TemporalSmoother(adhoc), uvset)
    doff = FP.baseline_fringe_data(uvset, sol_off)
    poff = FP.baseline_pol_index(doff, :parallel)
    @test amp_ripple(doff.spec_after, doff, poff) > 1.3    # roll-off ripple without the stage

    # The smooth estimators flatten the band AND fill the killed channels onto the
    # in-spw curve (≈ the mean of the live neighbours, well away from log-amp 0).
    for sm in (FP.polynomial_bandpass(4), FP.penalized_bandpass(0.1))
        sol = fit(ff |> BandpassEstimator(amp_model = sm) |> TemporalSmoother(adhoc), uvset)
        don = FP.baseline_fringe_data(uvset, sol)
        p = FP.baseline_pol_index(don, :parallel)
        @test amp_ripple(don.spec_after, don, p) < 1.08
        bp = CAL._step(sol, :bandpass)
        plan = FP._amp_bandpass_plan(bp.model, bp.layout)
        for dg in dead_globals, a in 2:nant, f in 1:2
            nbr = 0.5 * (larec(bp.θ, plan, a, f, dg - 1) + larec(bp.θ, plan, a, f, dg + 1))
            @test isfinite(larec(bp.θ, plan, a, f, dg))
            @test abs(larec(bp.θ, plan, a, f, dg) - nbr) < 0.1    # estimated, on the smooth curve
            @test abs(nbr) > 0.12                                 # ...curve far from |g|=1 (meaningful)
        end
    end

    # free_bandpass does NOT estimate the killed channels — their θ slot is untouched
    # (log-amp 0 ⇒ |g| = 1), the contrast that motivates the smoothers.
    solf = fit(ff |> BandpassEstimator(amp_model = FP.free_bandpass()) |> TemporalSmoother(adhoc), uvset)
    bpf = CAL._step(solf, :bandpass)
    planf = FP._amp_bandpass_plan(bpf.model, bpf.layout)
    for dg in dead_globals, a in 2:nant, f in 1:2
        @test larec(bpf.θ, planf, a, f, dg) == 0.0
    end
end

@testset "Adhoc auto window (:auto) flattens end-to-end" begin
    # The default `SavitzkyGolaySmoother()` now uses `window = :auto` (EHT-HOPS coherence-time
    # selection). Verify the default path still flattens the per-baseline phase.
    uvset, _ = _build_fringe_uvset()
    sol = fit(
        FringeFit(model = FringeModel()) |> BandpassEstimator() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother()),   # window = :auto
        uvset,
    )
    corr = Gustavo.apply_calibration(uvset, sol)
    worst = 1.0
    for (_, leaf) in DimensionalData.branches(corr)
        V = parent(leaf[:vis]); W = parent(leaf[:weights])
        bl_pairs = UVP.baselines(leaf).pairs
        lp = pol_products(leaf)
        for p in eachindex(lp)
            Gustavo.Calibration.is_parallel_hand(lp[p]) || continue
            for bi in eachindex(bl_pairs)
                a, b = bl_pairs[bi]
                a == b && continue
                worst = min(worst, _coherence(@view(V[:, :, bi, p]), @view(W[:, :, bi, p])))
            end
        end
    end
    @test worst > 0.99
end

@testset "Fringe pipeline via hierarchical MBD search" begin
    # Narrow bands widely separated (4 × 8 ch, origins every 150 MHz): the
    # common-Δf grid is ≈ 7× the real channel count, so the group search
    # auto-selects the hierarchical SBD→MBD path — verify, then check the
    # end-to-end solve flattens the data exactly like the full path does.
    uvset, _ = _build_fringe_uvset(nspw = 4, nchan = 8, spw_sep = 1.5e8)
    geom = CAL.build_geometry(uvset)
    ax = FP._search_axes(geom.channel_freqs, geom.times .* 3600.0, FP.FringeSearch(), ComplexF64)
    @test ax.mbd !== nothing

    sol = fit(
        FringeFit(model = FringeModel()) |> BandpassEstimator() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset,
    )
    @test all(>(10), filter(isfinite, CAL._step(sol, :fringe).info.scan_snr))
    @test isempty(FP.suspect_fringes(sol))                 # all detections secure

    corr = Gustavo.apply_calibration(uvset, sol)
    worst = 1.0
    for (_, leaf) in DimensionalData.branches(corr)
        V = parent(leaf[:vis])
        W = parent(leaf[:weights])
        bl_pairs = UVP.baselines(leaf).pairs
        lp = pol_products(leaf)
        for p in eachindex(lp), bi in eachindex(bl_pairs)
            a, b = bl_pairs[bi]
            a == b && continue
            worst = min(worst, _coherence(@view(V[:, :, bi, p]), @view(W[:, :, bi, p])))
        end
    end
    @test worst > 0.99
end

@testset "Threaded per-baseline search ≡ serial, and stage timers" begin
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)
    stq = FP.scan_stream(uvset; geom = geom)
    stack, _ = FP.materialize_cube(stq, stq.groups[1])
    det1 = FP.search_scan(stack, stq.geom, FP.FringeSearch(); executor = SerialScheduler(), ngroups = 1)
    det4 = FP.search_scan(stack, stq.geom, FP.FringeSearch(); executor = DynamicScheduler(; nchunks = 4), ngroups = 1)
    # Same detections regardless of the inner task count (≈ only because the two
    # runs plan separate FFTW MEASURE transforms).
    @test size(det1) == size(det4)
    @test all(
        isapprox(det1[i].delay, det4[i].delay; atol = 1.0e-15) &&
            isapprox(det1[i].rate, det4[i].rate; atol = 1.0e-12) &&
            isapprox(det1[i].snr, det4[i].snr; rtol = 1.0e-9) &&
            det1[i].valid == det4[i].valid
            for i in eachindex(det1)
    )

    # Stage timers land in the solution info and print; the progress callback
    # fires per completed scan of each pass (plus a done=0 pass announcement).
    events = Tuple{Symbol, Int, Int}[]
    sol = fit(
        FringeFit(model = FringeModel()) |> BandpassEstimator() |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset;
        exec = ExecutionConfig(progress = (st, d, t) -> push!(events, (st, d, t))),
    )
    ngroups = length(FP.scan_stream(uvset; ntasks = 1).groups)
    # Stages report under the pass names of the composable pipeline
    # (:fringe/:bandpass/:adhoc — the monolith's :search stage died with it).
    for st in (:fringe, :adhoc)
        ev = [(d, t) for (s2, d, t) in events if s2 == st]
        @test !isempty(ev)
        total = ev[1][2]
        st === :fringe && @test total == ngroups
        @test sort([d for (d, _) in ev]) == collect(0:total)   # announcement + every scan
        @test all(t == total for (_, t) in ev)
    end
    @test any(e -> e[1] === :bandpass, events)                 # bandpass enabled by default
    inf = sol.info
    fringe_step, bandpass_step, adhoc_step = CAL._step(sol, :fringe), CAL._step(sol, :bandpass), CAL._step(sol, :adhoc)
    @test fringe_step.info.t_pass > 0 && adhoc_step.info.t_pass > 0 && bandpass_step.info.t_pass >= 0
    @test inf.ntasks_used >= 1 && inf.inner_tasks >= 1
    # Every step publishes the SAME generic per-scan timing shape — no
    # per-step-name code in the runner, so a third-party step gets this too.
    for step in (fringe_step, bandpass_step, adhoc_step)
        t = step.info.timing
        @test length(t.decode) == inf.nscan
        @test all(>=(0), t.decode) && all(>=(0), t.work) && all(>=(0), t.reduce)
    end
    @test sum(fringe_step.info.timing.work) > 0
    buf = IOBuffer()
    FP.print_solve_timing(sol; io = buf)
    out = String(take!(buf))
    @test occursin("Solve timing", out) && occursin("fringe", out)

    # Capping the bandpass accumulation to a single scan still produces a
    # working solve (the stage-B/bandpass/adhoc chain is intact).
    ev2 = Tuple{Symbol, Int, Int}[]
    solc = fit(
        FringeFit(model = FringeModel()) |>
            BandpassEstimator(select = Gustavo.ScanIndices(1)) |>
            TemporalSmoother(FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)),
        uvset;
        exec = ExecutionConfig(progress = (st, d, t) -> push!(ev2, (st, d, t))),
    )
    @test solc isa CAL.CalibrationSolution
    bp2 = [(d, t) for (s2, d, t) in ev2 if s2 === :bandpass]
    @test !isempty(bp2) && bp2[1][2] == 1                     # capped to one scan
    corr2 = Gustavo.apply_calibration(uvset, solc)
    l2 = first(values(UVP.branches(corr2)))
    bl2 = UVP.baselines(l2).pairs
    p2 = findfirst(pr -> pr[1] != pr[2], collect(bl2))
    @test _coherence(@view(parent(l2[:vis])[:, :, p2, 1]), @view(parent(l2[:weights])[:, :, p2, 1])) > 0.99
    # Solutions without timers degrade cleanly.
    fringe = CAL._step(sol, :fringe)
    old = CAL.CalibrationSolution(fringe.model, fringe.layout, sol.geom, fringe.θ, (;))
    @test_nowarn FP.print_solve_timing(old; io = IOBuffer())
end

@testset "Dispersion (dTEC) refinement recovers injected station TEC" begin
    # VGOS-like layout: 8 sub-bands over 3.0-6.5 GHz — wide enough fractional
    # bandwidth that 1/ν separates from a linear delay (`dispersion = :auto`
    # turns the term on). Phase bandpass OFF: on a single-scan synthetic a
    # global per-channel bandpass is degenerate with the dispersion curvature
    # (on real multi-scan data the bandpass absorbs only the CALIBRATOR scan's
    # ionosphere and per-scan dTEC is measured relative to it).
    # `feed_common = true` (zero true R-L offset): dispersion smears the stage-B
    # delay peak, so each correlation product's argmax scatters within the smeared
    # peak and that scatter lands in the GLOBAL R-L delay offset — which the
    # (correctly feed-common) refinement cannot repair. A multi-scan track
    # averages that offset error to ~10 ps; a single-scan noiseless synthetic
    # eats the full smear, so the test removes the coupling to isolate the
    # dispersion machinery itself.
    dtec_true = [0.0, 3.0, -5.0, 1.5]
    uvset, _ = _build_fringe_uvset(
        nant = 4, nspw = 8, nchan = 8, ref_freq = 3.0e9, spw_sep = 0.5e9,
        dtec = dtec_true, seed = 77, feed_common = true,
    )
    geom = CAL.build_geometry(uvset)
    @test CAL._dispersion_enabled(CAL.DispersionModel(), geom)

    sol = fit(
        FringeFit(
            model = FringeModel(),
            estimator = FP.MatchedFilter(search = FP.FringeSearch(algorithm = FP.FullGrid())),
        ) |> DispersionSBDFit() |> TemporalSmoother(),        # no bandpass stage (see comment above)
        uvset,
    )
    refine = CAL._step(sol, :refine)
    @test stage_info(sol, :refine).dispersion_applied
    dplan = CAL._dispersion_plan(refine.model, refine.layout)
    @test dplan !== nothing
    for a in 1:4
        off = plan_off1(dplan)[a, 1, 1, 1]
        off == 0 && continue
        @test isapprox(refine.θ[off], dtec_true[a] - dtec_true[1]; atol = 0.05)
    end

    # CROSS-BAND coherence: collapse each band leaf to one phasor per
    # (baseline, parallel product), then |Σ_bands| / Σ|·| pooled. This is the
    # metric dispersion decoheres — `coherence_report`'s frequency sweep bins
    # WITHIN each leaf (one band), so it cannot see cross-band structure.
    function _crossband_eta(uv)
        zsum = Dict{Tuple{Int, Int}, ComplexF64}()
        zabs = Dict{Tuple{Int, Int}, Float64}()
        for (_, l) in Gustavo.UVData.leaves(uv)
            V = parent(l[:vis])
            W = parent(l[:weights])
            for p in (1, 4), bi in axes(V, 3)
                acc = zero(ComplexF64)
                for t in axes(V, 2), c in axes(V, 1)
                    w = W[c, t, bi, p]
                    w > 0 || continue
                    acc += w * V[c, t, bi, p]
                end
                zsum[(bi, p)] = get(zsum, (bi, p), zero(ComplexF64)) + acc
                zabs[(bi, p)] = get(zabs, (bi, p), 0.0) + abs(acc)
            end
        end
        return sum(abs, values(zsum)) / sum(values(zabs))
    end

    # The full correction aligns the bands: cross-band coherence ≈ 1.
    corr = Gustavo.UVData.apply_calibration(uvset, sol)
    @test _crossband_eta(corr) > 0.99

    # Without the term (no DispersionSBDFit step) the dispersion survives as
    # cross-band decoherence.
    sol0 = fit(
        FringeFit(
            model = FringeModel(),
            estimator = FP.MatchedFilter(search = FP.FringeSearch(algorithm = FP.FullGrid())),
        ) |> TemporalSmoother(),
        uvset,
    )
    @test :refine ∉ stage_names(sol0)     # no DispersionSBDFit step in this pipeline at all
    corr0 = Gustavo.UVData.apply_calibration(uvset, sol0)
    @test _crossband_eta(corr0) < 0.9

    # HDF5 round-trip carries the Dispersion term (generic serialization).
    mktempdir() do dir
        path = joinpath(dir, "disp.h5")
        CAL.save_solution_hdf5(path, sol)
        sol2 = CAL.load_solution_hdf5(path)
        refine2 = CAL._step(sol2, :refine)
        @test refine2.θ == refine.θ
        @test any(tc -> tc.component.term isa CAL.Dispersion, CAL.phase_components(refine2.model))
    end
end

@testset "SBD: per-scan band-group delay recovered" begin
    # Two band GROUPS (4 sub-bands at 3.0–3.3 GHz, 4 at 5.0–5.3 GHz — the gap
    # ratio splits them) with an injected per-GROUP delay of ±2 ns on station 2,
    # zero-mean across groups so the wideband stage-B delay cannot absorb it.
    # This is the fourfit-SBD situation: a per-band instrumental slope that
    # neither the wideband delay nor a time-invariant bandpass owns (VR2505's
    # YJ drifts by ~30 ns between scans).
    origins = [3.0e9, 3.1e9, 3.2e9, 3.3e9, 5.0e9, 5.1e9, 5.2e9, 5.3e9]
    nb, nch = length(origins), 16
    chf = vcat([o .+ (0:(nch - 1)) .* 2.0e6 for o in origins]...)
    groups = FP.fringe_freq_groups(chf)
    @test length(groups) == 2
    τ2 = [2.0e-9, -2.0e-9]
    ph = zeros(4, 2, nb * nch)
    for (g, r) in enumerate(groups)
        νc = sum(chf[r]) / length(r)
        for c in r
            ph[2, :, c] .= 2π * τ2[g] * (chf[c] - νc)
        end
    end
    uvset, _ = _build_fringe_uvset(
        nant = 4, nspw = nb, nchan = nch, ref_freq = 3.0e9, chan_bw = 2.0e6,
        spw_origins = origins, bandpass = ph, seed = 99, feed_common = true,
    )
    geom = CAL.build_geometry(uvset)
    # The SingleBandDelay element emits its per-band pair on this geometry.
    @test length(CAL.model_components(SingleBandDelay(), geom)) == 2

    # Per-channel pooled coherence of one baseline after correction (time-avg
    # per channel, |Σ_c z| / Σ_c |z| across all channels).
    function _perchan_eta(uv, bi)
        acc = ComplexF64[]
        for (_, l) in Gustavo.UVData.leaves(uv)
            V = parent(l[:vis])
            W = parent(l[:weights])
            for c in axes(V, 1)
                z = zero(ComplexF64)
                for t in axes(V, 2)
                    w = W[c, t, bi, 1]
                    w > 0 || continue
                    z += w * V[c, t, bi, 1]
                end
                abs(z) > 0 && push!(acc, z)
            end
        end
        return abs(sum(acc)) / sum(abs, acc)
    end

    # dispersion OFF: this geometry's spw count/fractional bandwidth would also
    # enable the dTEC term, and this test isolates the SBD machinery.
    sol = fit(
        FringeFit(
            model = FringeModel(),
            estimator = FP.MatchedFilter(search = FP.FringeSearch(algorithm = FP.FullGrid())),
        ) |> DispersionSBDFit(dispersion = nothing) |> TemporalSmoother(),
        uvset,
    )
    refine = CAL._step(sol, :refine)
    @test stage_info(sol, :refine).sbd_applied
    @test !stage_info(sol, :refine).dispersion_applied
    sbd = FP._sbd_plans(refine.model, refine.layout)
    @test sbd !== nothing
    # A common-mode slope across groups is gauge-shared with the wideband
    # stage-B delay (and its constants land in the SBD phase columns), so the
    # gauge-invariant recovery check is the ACROSS-GROUP DIFFERENCE.
    Δ(a) = refine.θ[plan_off1(sbd.dplan)[a, 1, 1, 1]] - refine.θ[plan_off1(sbd.dplan)[a, 1, 1, 2]]
    @test plan_off1(sbd.dplan)[2, 1, 1, 1] != 0
    @test isapprox(Δ(2), τ2[1] - τ2[2]; atol = 0.1e-9)          # injected 4 ns split
    @test abs(Δ(3)) < 0.1e-9                                    # clean station ≈ 0
    corr = Gustavo.UVData.apply_calibration(uvset, sol)
    @test _perchan_eta(corr, 1) > 0.98                          # baseline (1,2) flat

    # Without the term (no DispersionSBDFit step) the per-group slope survives
    # as within-group decoherence.
    sol0 = fit(
        FringeFit(
            model = FringeModel(),
            estimator = FP.MatchedFilter(search = FP.FringeSearch(algorithm = FP.FullGrid())),
        ) |> TemporalSmoother(),
        uvset,
    )
    @test :refine ∉ stage_names(sol0)     # no DispersionSBDFit step in this pipeline at all
    corr0 = Gustavo.UVData.apply_calibration(uvset, sol0)
    @test _perchan_eta(corr0, 1) < 0.9
end

@testset "dTEC co-located tie (the Onsala-twin constraint)" begin
    # Station 4 sits 60 m from station 3 (same ionosphere); stations are
    # otherwise 100 km apart. `_colocated_ties` groups them; the dispersion
    # solve then fits ONE dTEC for the pair and both θ columns carry it.
    positions = [[0.0, 0.0, 0.0], [1.0e5, 0.0, 0.0], [2.0e5, 0.0, 0.0], [2.0e5 + 60.0, 0.0, 0.0]]
    dtec_true = [0.0, 3.0, -5.0, -5.0]
    uvset, _ = _build_fringe_uvset(
        nant = 4, nspw = 8, nchan = 8, ref_freq = 3.0e9, spw_sep = 0.5e9,
        dtec = dtec_true, seed = 77, feed_common = true,
        station_positions = positions,
    )
    ants = Gustavo.UVData.metadata(first(values(Gustavo.UVData.branches(uvset)))).antennas
    @test UVP._colocated_ties(ants) == [1, 2, 3, 3]

    # Degenerate positions (the default synthetic table, max sep ≪ 10 km) must
    # NOT tie anything — the guard against missing/zero station_xyz.
    uvd, _ = _build_fringe_uvset(nant = 4, nspw = 2, nchan = 4)
    antd = Gustavo.UVData.metadata(first(values(Gustavo.UVData.branches(uvd)))).antennas
    @test UVP._colocated_ties(antd) == [1, 2, 3, 4]

    # The intra-site baseline set derived from the same grouping: exactly the
    # (3,4) twin pair, both orders; empty when the position guard trips.
    @test UVP._colocated_pair_set(ants) == Set([(3, 4), (4, 3)])
    @test isempty(UVP._colocated_pair_set(antd))

    # TWO co-located pairs must BOTH tie (regression: a `break` in the old
    # comma-nested loop exited both levels after the first pair — on VR2505
    # it tied Onsala and silently skipped the Wettzell twins).
    pos2 = [
        [0.0, 0.0, 0.0], [1.0e5, 0.0, 0.0], [1.0e5 + 70.0, 0.0, 0.0],
        [2.0e5, 0.0, 0.0], [2.0e5 + 60.0, 0.0, 0.0],
    ]
    uv2, _ = _build_fringe_uvset(
        nant = 5, nspw = 2, nchan = 4, station_positions = pos2,
    )
    ant2 = Gustavo.UVData.metadata(first(values(Gustavo.UVData.branches(uv2)))).antennas
    @test UVP._colocated_ties(ant2) == [1, 2, 2, 4, 4]
    @test UVP._colocated_pair_set(ant2) == Set([(2, 3), (3, 2), (4, 5), (5, 4)])

    sol = fit(
        FringeFit(
            model = FringeModel(),
            estimator = FP.MatchedFilter(search = FP.FringeSearch(algorithm = FP.FullGrid())),
        ) |> DispersionSBDFit(sbd = nothing) |> TemporalSmoother(),
        uvset,
    )
    refine = CAL._step(sol, :refine)
    dplan = CAL._dispersion_plan(refine.model, refine.layout)
    @test dplan !== nothing
    o3 = plan_off1(dplan)[3, 1, 1, 1]
    o4 = plan_off1(dplan)[4, 1, 1, 1]
    @test o3 != 0 && o4 != 0
    @test refine.θ[o3] == refine.θ[o4]                                # tied EXACTLY
    @test isapprox(refine.θ[o3], dtec_true[3] - dtec_true[1]; atol = 0.05)
end

@testset "Bandpass calibrator: SourceScans override + coverage top-up" begin
    srcs = ["A", "A", "A", "B"]
    snr = [500.0, 600.0, 550.0, 900.0]
    recs = [
        (index = i, source = srcs[i], scan = "No$i", snr = snr[i], stations = Set([1, 2]))
            for i in eachindex(srcs)
    ]
    @test select_scans(Gustavo.SourceScans("B"), recs) == [4]   # explicit override

    # Top-up is a no-op when the selected scans already cover every station.
    full = [
        (index = i, source = "A", scan = "No$i", snr = 1.0, stations = Set(1:4))
            for i in 1:3
    ]
    @test select_scans(FP.CoverageTopup(Gustavo.AllScans()), full) == [1, 2, 3]
end

@testset "Budget-scheduled group map" begin
    # Results come back in index order regardless of completion order, and
    # groups pack by their own charge (not the largest group's).
    res, peak = ST._scheduled_map(x -> x * 10, 1:8, [10, 3, 3, 3, 1, 2, 2, 2], 12; max_tasks = 4)
    @test res == [10, 20, 30, 40, 50, 60, 70, 80]
    @test 1 <= peak <= 4

    # A group charged more than the whole budget is clamped so it still runs.
    res2, _ = ST._scheduled_map(x -> x + 1, 1:3, [100, 1, 1], 10; max_tasks = 4)
    @test res2 == [2, 3, 4]

    # Empty input and worker-error propagation.
    res3, peak3 = ST._scheduled_map(identity, Int[], Float64[], 10; max_tasks = 2)
    @test isempty(res3) && peak3 == 0
    @test_throws Exception ST._scheduled_map(
        x -> x == 2 ? error("boom") : x, 1:3, [1, 1, 1], 10; max_tasks = 2,
    )
end

@testset "EHT-HOPS station flags: unconstrained station zero-weighted" begin
    # Station 4 participates (baselines with valid weights) but carries NO
    # fringe — pure weak noise — so after the closure-screened global solve no
    # strong detection constrains it: the EHT-HOPS flag criterion. Its gains
    # stay identity, and apply_calibration must zero-weight its baselines
    # instead of passing the uncalibrated data through at full weight.
    uvset, _ = _build_fringe_uvset(nant = 4, nspw = 2, nchan = 8, ntime = 12, feed_common = true)
    rng = MersenneTwister(0xF1A6)
    for (_, leaf) in Gustavo.UVData.leaves(uvset)
        V = parent(leaf[:vis])
        prs = Gustavo.UVData.baselines(leaf).pairs
        for bi in eachindex(prs)
            (prs[bi][1] == 4 || prs[bi][2] == 4) || continue
            for idx in CartesianIndices((axes(V, 1), axes(V, 2), axes(V, 4)))
                V[idx[1], idx[2], bi, idx[3]] = 0.01 * (randn(rng) + im * randn(rng))
            end
        end
    end
    sol = fit(
        FringeFit(
            model = FringeModel(terms = _fringe_terms(dispersion = false, sbd = false)),
            estimator = FP.MatchedFilter(search = FP.FringeSearch(algorithm = FP.FullGrid())),
        ) |> TemporalSmoother(),
        uvset,
    )
    flags = FP.fringe_station_flags(sol)
    @test !isempty(flags)
    @test all(r -> r.ant == 4, flags)                 # only station 4 unconstrained
    @test any(r -> r.station == "A4", flags)

    corr = Gustavo.UVData.apply_calibration(uvset, sol)
    for (_, leaf) in Gustavo.UVData.leaves(corr)
        W = parent(leaf[:weights])
        prs = Gustavo.UVData.baselines(leaf).pairs
        for bi in eachindex(prs)
            prs[bi][1] == prs[bi][2] && continue
            if prs[bi][1] == 4 || prs[bi][2] == 4
                @test all(iszero, @view W[:, :, bi, :])
            else
                @test any(>(0), @view W[:, :, bi, :])
            end
        end
    end
    # Opting out keeps the (identity-gain) data.
    corr0 = Gustavo.UVData.apply_calibration(uvset, sol; apply_flags = false)
    l0f = last(first(Gustavo.UVData.leaves(corr0)))
    prs0 = Gustavo.UVData.baselines(l0f).pairs
    bi4 = findfirst(p -> p[1] != p[2] && (p[1] == 4 || p[2] == 4), prs0)
    @test any(>(0), @view parent(l0f[:weights])[:, :, bi4, :])
end
