# Gustavo.Solve — the exact, prior-coupled MAP block update (matrix-free CG).
#
# The one-shot `solve_adhoc_phasing` solves the per-slice ML closure; with a prior
# on the block (`BandpassARPrior` across channels, `OUPrior` across APs) the slices
# COUPLE and the MAP block update becomes one coupled linear system solved
# matrix-free by CG. Tests: (1) the matrix-free operator == a dense solve; (2) the
# AR prior regularizes a low-SNR recovery; (3) `_map_block_solve` recovers a smooth
# truth end-to-end on the freq (per-feed) and time (shared-feeds) axes; (4) the
# strategy wires a `BandpassARPrior` into `FreqStep` (explicitly and via the
# posterior's `ComponentPriors`) and still reaches a correct calibration.
#
# Reuses `_build_fringe_uvset` (test_pipeline.jl) and `_forward_uvset`
# (test_solve_fit.jl), so it is included after both in runtests.jl.

using Gustavo.Solve
using Gustavo.Solve: fringe_solve, fringe_objective, plan_gains, zero_params, flatten,
    BlockCoordinate, GradientStep, FreqStep, LinearPhaseStep,
    BandpassARPrior, OUPrior, NoPrior, ComponentPriors, PointSource,
    SiteComponent, _map_block_solve, _apply_data!, _apply_banded_blocks!,
    _ar_bands, _data_rhs!, _map_cg_solver
import Gustavo.UVData as UV
using Gustavo.Calibration:
    build_geometry, StationGainModel, GainComponent, TiedComponent,
    Delay, ConstantTerm, PerChannel, PerScan, GlobalTime, GlobalFrequency,
    PerSpectralWindow, PerFeed, correlation_feed_pair
using Enzyme, Optimization, OptimizationOptimJL
using Random: MersenneTwister
using Statistics: mean
using Test

# Synthetic per-edge residual phasor for a station×slice phase truth `xtrue`
# (`node = (feed-1)*nant + a`). `z = wsum·cis(Δφ)`, `wsum = snr²`, so the solve's
# `W = |z|²/wsum = snr²` and `angle(z) = Δφ`.
_gauge(X, ref) = X .- X[ref:ref, :]
_rmse(A, B) = sqrt(mean(abs2, A .- B))

@testset "Solve: exact prior-coupled MAP block (matrix-free CG)" begin

    @testset "matrix-free operator ≡ dense solve" begin
        rng = MersenneTwister(1)
        nant, nslice, ref = 5, 10, 1
        edges = [(a, b) for a in 1:nant for b in (a + 1):nant]
        na = [e[1] for e in edges]
        nb = [e[2] for e in edges]
        xt = zeros(nant, nslice)
        for a in 2:nant
            xt[a, :] = 0.4 .* sin.(2π .* (1:nslice) ./ nslice .* 0.5 .+ 0.3a)
        end
        Z = ComplexF64[
            cis(xt[na[e], s] - xt[nb[e], s]) + (randn(rng) + im * randn(rng)) / (sqrt(2) * 30)
                for e in eachindex(edges), s in 1:nslice
        ]
        W = fill(30.0^2, length(edges), nslice)

        groups = [collect(1:nslice)]          # one spw
        φ = [2.0, -1.0]                        # 2nd-difference curvature
        a0, aε = 1.0 / 1.0^2, 1.0 / 0.1^2
        qbands = [_ar_bands(nslice, φ, a0, aε)]
        applyM! = function (Y, X)
            _apply_data!(Y, X, na, nb, W)
            _apply_banded_blocks!(Y, X, groups, qbands, length(φ))
            @views Y[ref, :] .= 0.0
            return Y
        end
        B = zeros(nant, nslice)
        _data_rhs!(B, Z, W, na, nb)
        @views B[ref, :] .= 0.0

        solve_cg = _map_cg_solver(applyM!, nant, nslice; abstol = 1.0e-12, reltol = 1.0e-14, maxiters = 2000)
        Xcg = solve_cg(B)

        # Dense reference by probing the operator column by column.
        N = nant * nslice
        M = zeros(N, N)
        e = zeros(nant, nslice)
        Y = similar(e)
        for j in 1:N
            fill!(e, 0.0); e[j] = 1.0
            applyM!(Y, e); M[:, j] = vec(Y)
        end
        Bv = vec(B)
        for s in 1:nslice                      # pin refant row/col in the dense system
            idx = (s - 1) * nant + ref
            M[idx, :] .= 0; M[:, idx] .= 0; M[idx, idx] = 1; Bv[idx] = 0
        end
        Xdense = reshape(M \ Bv, nant, nslice)
        @test maximum(abs, Xcg .- Xdense) < 1.0e-8
    end

    @testset "AR prior regularizes a low-SNR recovery" begin
        nant, nslice, ref = 8, 64, 1
        edges = [(a, b) for a in 1:nant for b in (a + 1):nant]
        na = [e[1] for e in edges]
        nb = [e[2] for e in edges]
        xt = zeros(nant, nslice)
        for a in 2:nant
            xt[a, :] = 0.5 .* sin.(2π .* (1:nslice) ./ nslice .* 0.5) .+
                0.15 * (a - 1) .* ((1:nslice) ./ nslice)
        end
        xt = _gauge(xt, ref)
        groups = [collect(1:nslice)]
        φ = [2.0, -1.0]
        solve(a0, aε, Z, W) = begin
            qbands = [_ar_bands(nslice, φ, a0, aε)]
            applyM! = function (Y, X)
                _apply_data!(Y, X, na, nb, W)
                _apply_banded_blocks!(Y, X, groups, qbands, length(φ))
                @views Y[ref, :] .= 0.0
                return Y
            end
            B = zeros(nant, nslice); _data_rhs!(B, Z, W, na, nb); @views B[ref, :] .= 0.0
            solve_cg = _map_cg_solver(applyM!, nant, nslice; abstol = 1.0e-10, reltol = 1.0e-12, maxiters = 2000)
            solve_cg(B)
        end
        for snr in (5.0, 2.0)
            rng = MersenneTwister(42)
            Z = ComplexF64[
                cis(xt[na[e], s] - xt[nb[e], s]) + (randn(rng) + im * randn(rng)) / (sqrt(2) * snr)
                    for e in eachindex(edges), s in 1:nslice
            ]
            W = fill(snr^2, length(edges), nslice)
            Xml = solve(1.0e-6, 1.0e-6, Z, W)          # ~no prior
            Xmap = solve(1.0 / 10.0^2, 1.0 / 0.15^2, Z, W)
            @test _rmse(_gauge(Xmap, ref), xt) < _rmse(_gauge(Xml, ref), xt)
        end
    end

    @testset "_map_block_solve ties feeds via cross-hands (per-feed, BandpassAR)" begin
        # Cross hands (PQ/QP) carry the source cross-hand phase χ_s and TIE the two
        # feed gauges: without them feed-1 and feed-2 are disconnected graphs and the
        # per-station R–L relative phase is unconstrained. Generate all four products
        # from a true per-feed bandpass + a true per-slice χ, and check the MAP block
        # recovers BOTH feeds AND the R–L relative (feed2 − feed1) — refant pinned in
        # both feeds.
        nant, nchan, ref = 6, 24, 1
        pols = ["PP", "PQ", "QP", "QQ"]
        feeds = correlation_feed_pair.(pols)
        feed_a = [f[1] for f in feeds]
        feed_b = [f[2] for f in feeds]
        chisign(fa, fb) = fa == fb ? 0 : (fa < fb ? 1 : -1)
        bl = [(a, b) for a in 1:nant for b in (a + 1):nant]
        bl_a = [q[1] for q in bl]
        bl_b = [q[2] for q in bl]
        nbl, npol = length(bl), length(pols)

        # Smooth per (station, feed) bandpass truth (feeds DIFFER → a real R–L), ref = 0.
        xt = zeros(nant, 2, nchan)
        for a in 2:nant, f in 1:2
            xt[a, f, :] = 0.4 .* sin.(2π .* (1:nchan) ./ nchan .* 0.7 .+ (a + 0.9f))
        end
        χt = 0.5 .* sin.(2π .* (1:nchan) ./ nchan .* 0.3)    # true source cross-hand phase
        snr = 50.0
        z = zeros(ComplexF64, nbl, npol, nchan)
        wsum = zeros(Float64, nbl, npol, nchan)
        for s in 1:nchan, p in 1:npol, bi in 1:nbl
            a, b = bl_a[bi], bl_b[bi]
            fa, fb = feed_a[p], feed_b[p]
            dphi = xt[a, fa, s] - xt[b, fb, s] + chisign(fa, fb) * χt[s]
            wsum[bi, p, s] = snr^2
            z[bi, p, s] = snr^2 * cis(dphi)
        end

        comp = SiteComponent(
            PerChannel(), [1], ones(Int, nchan), zeros(nchan), zeros(1),
            collect(1:nchan), (1, 2), (0, 0), nchan, 1, 1, 2,
        )
        prior = BandpassARPrior(phi = [2.0, -1.0], sigma_eps = 0.05, sigma0 = 5.0)
        phase = _map_block_solve(
            z, wsum, bl_a, bl_b, feed_a, feed_b, nant, nchan, prior, comp, (;), true;
            ref_ant = ref, shared_feeds = false,
        )
        for f in 1:2
            @test _rmse(_gauge(phase[:, f, :], ref), _gauge(xt[:, f, :], ref)) < 5.0e-3
        end
        # The R–L relative (feed2 − feed1) — the quantity the cross-hand tie enables.
        rl_rec = phase[:, 2, :] .- phase[:, 1, :]
        rl_true = xt[:, 2, :] .- xt[:, 1, :]
        @test _rmse(_gauge(rl_rec, ref), _gauge(rl_true, ref)) < 5.0e-3
    end

    @testset "block-Jacobi preconditioner handles flagged channels (2 spw)" begin
        # Two spws with FLAGGED (zero-weight) edge-channel runs — the real-VLBA case
        # that makes unpreconditioned CG crawl (long pure-biharmonic stretches). The
        # block-Jacobi (D+Q)⁻¹ preconditioner must still recover the unflagged channels
        # (flagged ones interpolated by the AR prior) and stay finite.
        nant, ref, nspw, chanper = 6, 1, 2, 16
        nchan = nspw * chanper
        pols = ["PP", "PQ", "QP", "QQ"]
        feeds = correlation_feed_pair.(pols)
        feed_a = [f[1] for f in feeds]
        feed_b = [f[2] for f in feeds]
        chisign(fa, fb) = fa == fb ? 0 : (fa < fb ? 1 : -1)
        bl = [(a, b) for a in 1:nant for b in (a + 1):nant]
        bl_a = [q[1] for q in bl]
        bl_b = [q[2] for q in bl]
        nbl, npol = length(bl), length(pols)
        fseg_id = vcat((fill(k, chanper) for k in 1:nspw)...)
        clocal = vcat((collect(1:chanper) for _ in 1:nspw)...)
        flagged(s) = let lc = clocal[s]
            lc <= 3 || lc > chanper - 3
        end   # outer 3 chans/spw

        xt = zeros(nant, 2, nchan)
        for a in 2:nant, f in 1:2, sp in 1:nspw
            rng2 = (chanper * (sp - 1) + 1):(chanper * sp)
            xt[a, f, rng2] = 0.3 .* sin.(2π .* (1:chanper) ./ chanper .* 0.6 .+ (a + f))
        end
        χt = 0.4 .* sin.(2π .* (1:nchan) ./ nchan .* 0.3)
        z = zeros(ComplexF64, nbl, npol, nchan)
        wsum = zeros(Float64, nbl, npol, nchan)
        for s in 1:nchan, p in 1:npol, bi in 1:nbl
            flagged(s) && continue                       # zero weight on flagged channels
            a, b = bl_a[bi], bl_b[bi]
            fa, fb = feed_a[p], feed_b[p]
            dphi = xt[a, fa, s] - xt[b, fb, s] + chisign(fa, fb) * χt[s]
            wsum[bi, p, s] = 50.0^2
            z[bi, p, s] = 50.0^2 * cis(dphi)
        end
        comp = SiteComponent(
            PerChannel(), [1], fseg_id, zeros(nchan), zeros(1),
            clocal, (1, 2), (0, 0), chanper, 1, nspw, 2,
        )
        prior = BandpassARPrior(phi = [2.0, -1.0], sigma_eps = 0.05, sigma0 = 5.0)
        phase = _map_block_solve(
            z, wsum, bl_a, bl_b, feed_a, feed_b, nant, nchan, prior, comp, (;), true;
            ref_ant = ref, shared_feeds = false,
        )
        @test all(isfinite, phase)                        # flagged channels interpolated
        unflagged = [s for s in 1:nchan if !flagged(s)]
        for f in 1:2
            @test _rmse(_gauge(phase[:, f, unflagged], ref), _gauge(xt[:, f, unflagged], ref)) < 1.0e-2
        end
    end

    @testset "_map_block_solve recovers a smooth time truth (shared-feeds, OU)" begin
        nant, nap, ref = 6, 40, 1
        pols = ["PP", "PQ", "QP", "QQ"]
        feeds = correlation_feed_pair.(pols)
        feed_a = [f[1] for f in feeds]
        feed_b = [f[2] for f in feeds]
        bl = [(a, b) for a in 1:nant for b in (a + 1):nant]
        bl_a = [q[1] for q in bl]
        bl_b = [q[2] for q in bl]
        nbl, npol = length(bl), length(pols)
        times_hr = collect(range(0.0, 0.05, length = nap))    # ~3 min, ~4.6 s APs

        # Feed-common smooth atmospheric track, refant = 0.
        xt = zeros(nant, nap)
        for a in 2:nant
            xt[a, :] = 0.6 .* sin.(2π .* (1:nap) ./ nap .* 0.6 .+ 0.9a)
        end
        snr = 50.0
        z = zeros(ComplexF64, nbl, npol, nap)
        wsum = zeros(Float64, nbl, npol, nap)
        for s in 1:nap, p in 1:npol, bi in 1:nbl
            a, b = bl_a[bi], bl_b[bi]
            fa, fb = feed_a[p], feed_b[p]
            fa == fb || continue                # feed-common: only parallel hands feed it
            wsum[bi, p, s] = snr^2
            z[bi, p, s] = snr^2 * cis(xt[a, s] - xt[b, s])
        end

        comp = SiteComponent(
            ConstantTerm(), collect(1:nap), [1], zeros(1), zeros(nap),
            [1], (1, 1), (0, 0), 1, nap, 1, 1,
        )
        prior = OUPrior(τ = 30.0, σ2 = 1.0)
        phase = _map_block_solve(
            z, wsum, bl_a, bl_b, feed_a, feed_b, nant, nap, prior, comp,
            (; times = times_hr), false;
            ref_ant = ref, shared_feeds = true,
        )
        @test phase[:, 1, :] ≈ phase[:, 2, :]      # shared feeds replicated
        @test _rmse(_gauge(phase[:, 1, :], ref), _gauge(xt, ref)) < 5.0e-3
    end

    @testset "strategy wires the MAP block into FreqStep" begin
        uvset0, _ = _build_fringe_uvset(nant = 4, nbands = 2, nchan = 5, ntime = 5, feed_common = true)
        geom = build_geometry(uvset0)
        model = StationGainModel(
            phase = (
                clock = TiedComponent(GainComponent(Delay(), PerScan(), GlobalFrequency()), PerFeed()),
                fringe = TiedComponent(GainComponent(ConstantTerm(), PerScan(), GlobalFrequency()), PerFeed()),
                bandpass = TiedComponent(GainComponent(PerChannel(), GlobalTime(), PerSpectralWindow()), PerFeed()),
            ),
            logamp = (),
        )
        plan = plan_gains(model, 4, geom)
        rng = MersenneTwister(20260709)
        p_true = zero_params(plan)
        for a in 2:4, f in 1:2
            p_true.g1.clock[1, :, 1, f, a] .= (rand(rng) - 0.5) * 1.0e-9
            p_true.g1.fringe[1, :, 1, f, a] .= (rand(rng) - 0.5) * 1.0
            p_true.g1.bandpass[:, 1, :, f, a] .= 0.2 .* randn(rng, size(p_true.g1.bandpass, 1), size(p_true.g1.bandpass, 3))
        end
        uvset = _forward_uvset(uvset0, plan, p_true, geom)
        prior = BandpassARPrior(phi = [2.0, -1.0], sigma_eps = 0.3, sigma0 = 3.0)
        # Likelihood (no prior) at the zero start — the bar the MAP block must clear.
        obj0 = fringe_objective(plan, zero_params(plan), uvset, geom; source = PointSource(1.0))
        lik(sol) = fringe_objective(plan, sol.p, uvset, geom; source = PointSource(1.0))

        # (a) prior on the STEP. A BandpassARPrior REGULARIZES the (here white-noise)
        # bandpass, so a MAP block trades data-misfit for smoothness — it does NOT fit
        # noiseless data perfectly. Assert the coupled block still drives the likelihood
        # far above the raw start (the plumbing runs and helps).
        strat = BlockCoordinate(
            GradientStep(LBFGS(); frozen = (:bandpass,), maxiters = 600),
            FreqStep(:bandpass; prior = prior);
            rounds = 5,
        )
        sol = fringe_solve(uvset, model; strategy = strat, source = PointSource(1.0), warmstart = zero_params(plan))
        @test lik(sol) > obj0 + 1.0
        @test UV.apply_calibration(uvset, sol) isa UV.UVSet

        # (b) prior inherited from the posterior's ComponentPriors (step prior = nothing).
        strat2 = BlockCoordinate(
            GradientStep(LBFGS(); frozen = (:bandpass,), maxiters = 600),
            FreqStep(:bandpass);
            rounds = 5,
        )
        sol2 = fringe_solve(
            uvset, model; strategy = strat2, source = PointSource(1.0),
            warmstart = zero_params(plan), prior = ComponentPriors(bandpass = prior),
        )
        @test lik(sol2) > obj0 + 1.0
    end
end
