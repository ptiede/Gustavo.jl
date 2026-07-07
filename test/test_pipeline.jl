# End-to-end fringe-fitter pipeline test. Builds a small synthetic multi-band
# in-memory UVSet with KNOWN injected per-(station, feed) delay/rate/constant
# phase plus a per-(station, feed, AP) atmospheric phase screen, then verifies
# that `solve_fringes` → `apply_calibration` flattens the residual baseline
# phases (the coherence test) and that save/load round-trips.

using Gustavo
using Test
using Random
using LinearAlgebra: Diagonal
using StructArrays
using DimensionalData
using DimensionalData: DimArray, Ti, dims, lookup
using PolarizedTypes: RPol, LPol
using Gustavo.UVData: Integration, Pol, Frequency, UVW, Baseline, UVSet, pol_products, channel_freqs
using HDF5   # triggers GustavoHDF5Ext (solution save/load round-trip)
using FITSFiles   # triggers GustavoFITSFilesExt (write_uvfits/load_uvfits round-trip)

const CAL = Gustavo.Calibration
const FP = Gustavo.Fringe
const UVP = Gustavo.UVData

# ── Synthetic UVSet with injected station fringe parameters ──────────────────
#
# For baseline (a, b) and product p with feeds (fa, fb):
#   V[c, ti] = A · exp(i·[ Δφ + 2π·Δτ·(f_c − f0) + 2π·Δṙ·(t_sec − t0_sec)
#                          + Δscreen[ti] ])
# with Δx = x[a, fa] − x[b, fb]. Times in seconds use t0_sec; freqs in Hz.
function _build_fringe_uvset(;
        nant = 4, nbands = 2, nchan = 8, ntime = 12,
        pol_labels = ["PP", "PQ", "QP", "QQ"],
        ref_freq = 230.0e9, chan_bw = 2.0e6, band_sep = 1.0e8,
        seed = 1234,
        bandpass = nothing,    # optional (nant, 2, nbands*nchan) per-channel phase (rad)
        amp_bandpass = nothing, # optional (nant, 2, nbands*nchan) per-channel log-amp
        dtec = nothing,         # optional (nant,) station TEC (TECU, feed-common)
        feed_common = false,    # tie delay/phi across feeds (zero true R-L offset)
        band_origins = nothing, # optional (nbands,) explicit band start freqs (Hz) — overrides band_sep
        station_positions = nothing, # optional (nant,) xyz vectors (m) — for co-location tests
    )
    UV = Gustavo.UVData
    rng = MersenneTwister(seed)
    npol = length(pol_labels)

    ants_v = [
        UV.Antenna(;
                name = "A$(i)",
                station_xyz = station_positions === nothing ?
                Float64[100.0 * i, 200.0 * i, 300.0 * i] : Float64.(station_positions[i]),
                mount = UV.MountAltAz(),
                nominal_basis = (RPol(), LPol()),
                response = Diagonal(ones(ComplexF32, 2)),
                pol_angles = (0.0f0, 0.0f0),
            )
            for i in 1:nant
    ]
    antennas = UV.AntennaTable(
        StructArray(ants_v), Float64[1.0:nant;], "SYNTH",
        (; NOSTA = Int32.(1:nant), DIAMETER = fill(25.0f0, nant)),
    )

    bl_pairs = Tuple{Int, Int}[(a, b) for a in 1:nant for b in (a + 1):nant]
    nbl = length(bl_pairs)
    baselines = UV.BaselineIndex(bl_pairs, bl_pairs; antenna_names = collect(antennas.name))

    setups = UV.FrequencySetup[]
    for b in 1:nbands
        f_lo = band_origins === nothing ? ref_freq + (b - 1) * band_sep : Float64(band_origins[b])
        chf = f_lo .+ (0:(nchan - 1)) .* chan_bw
        push!(
            setups, UV.FrequencySetup(;
                name = "band_$(b)",
                ref_freq = ref_freq,
                channel_freqs = collect(chf),
                ch_widths = fill(chan_bw, nchan),
                total_bandwidths = fill(chan_bw * nchan, nchan),
                sidebands = fill(1.0, nchan),
                extras = (; bandfreq = (b - 1) * band_sep, band = b),
            ),
        )
    end

    array_obs = UV.ObsArrayMetadata(;
        telescope = "SYNTH", instrume = "SYNTH",
        date_obs = "2021-03-04", equinox = 2000.0f0, bunit = "JY",
        rdate = "2021-03-04", earth_rot_rate = 360.0f0,
        extras = (; correlat = "DiFX", obscode = "SY001"),
    )

    # Times (hours): one scan, 12 APs spaced 30 s.
    ti_vals = collect((0:(ntime - 1)) .* (30.0 / 3600.0))

    # Reference geometry constants (must match what solve_fringes derives):
    # f0 = mean of all channel freqs across both bands; t0 = first time (hours).
    allfreqs = Float64[]
    for fs in setups
        append!(allfreqs, collect(channel_freqs(fs)))
    end
    sort!(unique!(allfreqs))
    f0 = sum(allfreqs) / length(allfreqs)
    t0_sec = ti_vals[1] * 3600.0

    # Injected parameters. Reference antenna 1 = 0 (so the recovered solution matches
    # the gauge), others drawn small. `delay`/`phi` are PER-FEED (their constant R–L
    # offset is recovered by the model's global `FeedComponent(2)` delay/const terms);
    # `rate` is FEED-COMMON because the fringe model ties rate `SharedFeeds` (R–L rate
    # is not solved — see the model in pipeline.jl), so a per-feed rate would be
    # physically inconsistent and left uncorrected.
    delay = zeros(nant, 2)         # seconds (per feed)
    rate = zeros(nant, 2)          # Hz (feed-common)
    phi = zeros(nant, 2)           # rad (per feed)
    for a in 2:nant
        rc = (rand(rng) - 0.5) * 2.0e-3               # ±1 mHz, feed-common
        for f in 1:2
            delay[a, f] = (rand(rng) - 0.5) * 2.0e-9      # ±1 ns  (« 1/chan_bw)
            rate[a, f] = rc
            phi[a, f] = (rand(rng) - 0.5) * 2.0           # ±1 rad
        end
    end

    if feed_common
        delay[:, 2] .= delay[:, 1]
        phi[:, 2] .= phi[:, 1]
    end

    # Per-(station, AP) atmospheric screen (rad), reference held at 0, FEED-COMMON:
    # the atmosphere is non-birefringent, so both feeds see the same screen, and the
    # model's adhoc term is `SharedFeeds`. A smooth (slowly-varying) phase track — the
    # regime the Savitzky–Golay adhoc smoother is designed for — plus a per-station
    # constant offset that Stage-B's per-scan constant phase absorbs.
    screen = zeros(nant, 2, ntime)
    for a in 2:nant
        base = (rand(rng) - 0.5) * 1.0
        for ti in 1:ntime
            s = base + 0.2 * sin(0.5 * ti + a)
            screen[a, 1, ti] = s
            screen[a, 2, ti] = s
        end
    end

    feeds = [CAL.correlation_feed_pair(p) for p in pol_labels]
    A0 = 2.5

    src_name = "SRC1"
    branches = DimensionalData.TreeDict()
    for b in 1:nbands
        fs = setups[b]
        fch = collect(channel_freqs(fs))
        vis_dense = Array{ComplexF32}(undef, nchan, ntime, nbl, npol)
        w_dense = fill(1.0f3, nchan, ntime, nbl, npol)   # high weight → high SNR
        for p in 1:npol, bl in 1:nbl, ti in 1:ntime, c in 1:nchan
            a, bb = bl_pairs[bl]
            fa, fb = feeds[p]
            f = fch[c]
            tsec = ti_vals[ti] * 3600.0
            dτ = delay[a, fa] - delay[bb, fb]
            dṙ = rate[a, fa] - rate[bb, fb]
            dφ = phi[a, fa] - phi[bb, fb]
            dscr = screen[a, fa, ti] - screen[bb, fb, ti]
            gc = (b - 1) * nchan + c          # global channel index (bands stacked by freq)
            dbp = bandpass === nothing ? 0.0 : (bandpass[a, fa, gc] - bandpass[bb, fb, gc])
            dla = amp_bandpass === nothing ? 0.0 : (amp_bandpass[a, fa, gc] + amp_bandpass[bb, fb, gc])  # log-amp SUMS
            ddt = dtec === nothing ? 0.0 :
                CAL.DISPERSION_K * (dtec[a] - dtec[bb]) * (1.0 / f0 - 1.0 / f)
            ph = dφ + 2π * dτ * (f - f0) + 2π * dṙ * (tsec - t0_sec) + dscr + dbp + ddt
            vis_dense[c, ti, bl, p] = ComplexF32(A0 * exp(dla) * cis(ph))
        end
        vis_part = DimArray(
            vis_dense,
            (
                Frequency(fch), Ti(ti_vals),
                Baseline(baselines.labels), Pol(pol_labels),
            ),
        )
        w_part = DimArray(w_dense, dims(vis_part))

        uvw_dense = zeros(Float32, ntime, nbl, 3)
        for ti in 1:ntime, bl in 1:nbl
            uvw_dense[ti, bl, 1] = Float32(ti + 0.1 * bl)
            uvw_dense[ti, bl, 2] = Float32(2 * ti + 0.2 * bl)
            uvw_dense[ti, bl, 3] = Float32(3 * ti + 0.3 * bl)
        end
        uvw_part = DimArray(
            uvw_dense,
            (Ti(ti_vals), Baseline(baselines.labels), UVP.UVW(["U", "V", "W"])),
        )

        info = UV.PartitionInfo(;
            source_name = src_name,
            source_key = UV.sanitize_source(src_name),
            scan_name = "1",
            ra = 1.234, dec = -0.56,
            antennas = antennas,
            baselines = baselines,
            record_order = [(ti, bl) for bl in 1:nbl for ti in 1:ntime],
            freq_setup = fs,
            spw_name = "band_$(b)",
            ddi = b - 1,
            basename = "synth_fringe",
        )
        leaf = UV._build_leaf(vis_part, w_part, uvw_part; partition_info = info)
        branches[UV.partition_key(info)] = leaf
    end

    uvset = Gustavo.UVData.UVSet(; metadata = UV.UVMetadata(array_obs), branches = branches)
    return uvset, (; delay, rate, phi, screen, bandpass, amp_bandpass, dtec, f0, t0_sec, bl_pairs, pol_labels, feeds)
end

# Coherence of a (baseline, product) block: |Σ w·V| / Σ (w·|V|). 1 ⇒ phase flat.
function _coherence(V, W)
    num = zero(ComplexF64)
    den = 0.0
    for i in eachindex(V)
        v = V[i]
        w = W[i]
        (isfinite(v) && isfinite(w) && w > 0) || continue
        num += w * v
        den += w * abs(v)
    end
    den == 0 && return 0.0
    return abs(num) / den
end

@testset "Fringe pipeline end-to-end" begin
    uvset, _truth = _build_fringe_uvset()

    # Adhoc smoother window 7 (< the 12-AP scan) tracks the screen; snr_floor 0
    # keeps every well-determined AP in this high-SNR synthetic.
    sol = FP.solve_fringes(
        uvset; ref_ant = 1,
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0),
    )
    @test sol isa CAL.CalibrationSolution
    @test length(sol.θ) == sol.layout.nθ

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
        @test sol2.θ == sol.θ
        @test sol2.geom.times == sol.geom.times
        @test sol2.geom.channel_freqs == sol.geom.channel_freqs
        @test sol2.layout.nθ == sol.layout.nθ
        @test length(CAL.phase_components(sol2.model)) == length(CAL.phase_components(sol.model))

        corr2 = Gustavo.apply_calibration(uvset, sol2)
        for (k, leaf) in DimensionalData.branches(corr)
            V = parent(leaf[:vis])
            V2 = parent(DimensionalData.branches(corr2)[k][:vis])
            @test all(((x, y),) -> (isnan(x) && isnan(y)) || x == y, zip(V, V2))
        end
        rm(path; force = true)
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
    # pipeline.jl and the `fringe-rate-must-be-sharedfeeds` decision).
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

    # Layout: `SharedFeeds` assigns ONE θ column to both feeds, `PerFeed` two distinct
    # ones (Calibration `_assign_blocks!`). So feed-1 and feed-2 share every off1 slot
    # iff the tie holds — a `PerFeed` revert breaks this on any solved (station, seg).
    uvset, _ = _build_fringe_uvset()
    geom = CAL.build_geometry(uvset)
    first_leaf = first(values(DimensionalData.branches(uvset)))
    nant = length(UVP.metadata(first_leaf).antennas)
    layout = CAL.plan_parameters(model, nant, geom)

    for (ci, label) in ((rate_i, "rate"), (adhoc_i, "adhoc"))
        off1 = layout.plans[ci].off1                 # (ant, feed, ntseg, nfseg)
        @test off1[:, 1, :, :] == off1[:, 2, :, :]   # both feeds → same θ columns
        @test any(!=(0), off1[:, 1, :, :])           # ...and the plan is non-trivial
    end
end

@testset "Fused solve_and_reduce_fringes ≡ two-pass" begin
    # The fused single-pass driver must produce the SAME result as the explicit
    # two-pass `apply_calibration(uvset, solve_fringes(uvset))`, because each
    # leaf's gains depend only on its own (disjoint) θ slots.
    uvset, _ = _build_fringe_uvset()
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)

    sol_ref = FP.solve_fringes(uvset; ref_ant = 1, adhoc = adhoc)
    corr_ref = Gustavo.apply_calibration(uvset, sol_ref)

    # postprocess = identity → fused correction only (no reduction).
    sol_fused, out_fused = FP.solve_and_reduce_fringes(uvset; ref_ant = 1, adhoc = adhoc)

    @test sol_fused.θ ≈ sol_ref.θ
    # Same tree keys.
    @test Set(keys(DimensionalData.branches(out_fused))) ==
        Set(keys(DimensionalData.branches(corr_ref)))
    # Identical corrected visibilities/weights per leaf (NaN-aware).
    for (k, leaf) in DimensionalData.branches(corr_ref)
        Vr = parent(leaf[:vis]); Wr = parent(leaf[:weights])
        lf = DimensionalData.branches(out_fused)[k]
        Vf = parent(lf[:vis]); Wf = parent(lf[:weights])
        @test size(Vf) == size(Vr)
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || x == y, zip(Vr, Vf))
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || x == y, zip(Wr, Wf))
    end

    # With a reducer the fused output must equal applying the same reducer to the
    # two-pass corrected set.
    red_ref = UVP.frequency_average(corr_ref; nout = 1)
    _, out_red = FP.solve_and_reduce_fringes(
        uvset; ref_ant = 1, adhoc = adhoc,
        postprocess = uv -> UVP.frequency_average(uv; nout = 1),
    )
    for (k, leaf) in DimensionalData.branches(red_ref)
        Vr = parent(leaf[:vis])
        Vf = parent(DimensionalData.branches(out_red)[k][:vis])
        @test size(Vf) == size(Vr)
        @test all(((x, y),) -> (isnan(x) && isnan(y)) || x ≈ y, zip(Vr, Vf))
    end
end

@testset "combine_spw: bands → IF axis" begin
    uvset, _ = _build_fringe_uvset(nbands = 3, nchan = 4)
    avg = UVP.frequency_average(uvset; nout = 1)          # each band → 1 channel
    @test length(UVP.union_frequency_axis(avg)) == 3      # 3 distinct band setups

    combined = UVP.combine_spw(avg)
    @test length(DimensionalData.branches(combined)) == 1 # one leaf per (src,scan)
    setups = UVP.union_frequency_axis(combined)
    @test length(setups) == 1                             # single FREQID
    cf = collect(channel_freqs(first(setups)))
    @test length(cf) == 3                                 # 3 IFs
    @test issorted(cf)                                    # ascending IF freqs
    # The combined channel frequencies are exactly the per-band averaged centers.
    band_centers = sort([only(channel_freqs(fs)) for fs in UVP.union_frequency_axis(avg)])
    @test cf ≈ band_centers
    leaf = first(values(DimensionalData.branches(combined)))
    @test size(parent(leaf[:vis]), 1) == 3                # Frequency axis = 3 IFs
end

@testset "write_uvfits on FITS-IDI-style UVSet (synthesized primary cards)" begin
    # `_build_fringe_uvset` registers NO primary cards, so write_uvfits must
    # synthesize them. Reduce + combine bands to IFs, then round-trip via UVFITS.
    uvset, _ = _build_fringe_uvset(nbands = 2, nchan = 6)
    sol, reduced = FP.solve_and_reduce_fringes(
        uvset; ref_ant = 1,
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0),
        postprocess = uv -> UVP.time_bin_average(
            UVP.combine_spw(UVP.frequency_average(uv; nout = 1)), 0.02,
        ),
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
        @test length(channel_freqs(first(rt_setups))) == 2          # 2 IFs survive
        @test channel_freqs(first(rt_setups)) ≈ channel_freqs(first(UVP.union_frequency_axis(reduced)))
        rt_leaf = first(values(DimensionalData.branches(rt)))
        @test Set(String.(pol_products(rt_leaf))) == Set(["PP", "PQ", "QP", "QQ"])
        @test all(isfinite, filter(isfinite, parent(rt_leaf[:vis])))  # no NaN explosion
    finally
        isfile(path) && rm(path; force = true)
    end
end

@testset "Residual accumulation is inverse-variance in the CORRECTED data" begin
    # Regression: `Σ w·(V/g)` with the RAW weight `w` is only correct for
    # |g| = 1. Var(V/g) = 1/(w·|g|²), so the weight must be w·|g|² — otherwise
    # channels the amp bandpass marked low-|g| get their amplitude-inflated
    # noise UP-weighted (this tripled K2's adhoc track noise on VR2505). The
    # accumulated (rbar, wbar) must equal the inverse-variance mean.
    nchan, nti, nbl, npol = 2, 1, 1, 1
    V = zeros(ComplexF32, nchan, nti, nbl, npol)
    W = zeros(Float32, nchan, nti, nbl, npol)
    V[1, 1, 1, 1] = 1.0 + 0.0im          # channel 1: unit gain
    V[2, 1, 1, 1] = 0.1im                # channel 2: |g|² = 0.01, data rotated 90°
    W .= 1.0
    g = ones(ComplexF64, nchan, nti, 2, 2)
    g[2, 1, :, :] .= 0.1                 # station gains 0.1 ⇒ den = 0.01
    bl = [(1, 2)]
    rbar = zeros(ComplexF64, nbl, npol, nti)
    wbar = zeros(Float64, nbl, npol, nti)
    FP._accumulate_leaf_rbar!(rbar, wbar, V, W, g, bl, ["PP"])
    # corrected data: ch1 → 1+0i (weight 1), ch2 → 10i (weight 0.0001):
    # inverse-variance mean ≈ ch1, NOT the raw-weight mean ≈ (1 + 10i)/2.
    @test wbar[1, 1, 1] ≈ 1.0001
    @test rbar[1, 1, 1] / wbar[1, 1, 1] ≈ (1.0 + 0.001im) / 1.0001
    z = zeros(ComplexF64, nbl, npol)
    wz = zeros(Float64, nbl, npol)
    FP._accumulate_leaf_band_phasor!(z, wz, V, W, g, bl, ["PP"])
    @test wz[1, 1] ≈ 1.0001
    @test z[1, 1] / wz[1, 1] ≈ (1.0 + 0.001im) / 1.0001
end

@testset "Fringe pipeline: rounds > 1 accumulates (no corruption)" begin
    # Regression for the θ-overwrite bug: even rounds previously wiped the
    # round-1 solution (coherence collapsed). Accumulation keeps all rounds good.
    uvset, _ = _build_fringe_uvset()
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    for r in (1, 2, 3)
        sol = FP.solve_fringes(uvset; ref_ant = 1, rounds = r, adhoc = adhoc)
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

    # band_sep = 0 ⇒ the two bands share identical channel frequencies but carry
    # distinct spw_names ("band_1"/"band_2"), so a single concatenated channel
    # axis cannot dense-rank a frequency to one spw — build_geometry must error
    # instead of silently last-write-wins.
    uvset_conflict, _ = _build_fringe_uvset(band_sep = 0.0)
    @test_throws ErrorException CAL.build_geometry(uvset_conflict)
end

@testset "Phase bandpass: per-channel phase recovered" begin
    # Inject a smooth per-(station, feed, channel) phase bandpass (ref ant 1 = 0)
    # on top of the usual delay/rate/phase/screen. The bandpass stage should
    # flatten the per-channel phase; with it OFF the bandpass survives uncorrected.
    nant, nbands, nchan = 4, 2, 8
    nchg = nbands * nchan
    rng = MersenneTwister(0xBA9D)
    bp = zeros(nant, 2, nchg)
    for a in 2:nant, f in 1:2
        off = (rand(rng) - 0.5)
        for gc in 1:nchg
            bp[a, f, gc] = off + 1.0 * sin(2π * gc / nchg + a + f)   # smooth shape + offset
        end
    end
    uvset, _ = _build_fringe_uvset(; nant = nant, nbands = nbands, nchan = nchan, bandpass = bp)
    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    sol_on = FP.solve_fringes(uvset; ref_ant = 1, adhoc = adhoc, phase_bandpass = true)
    sol_off = FP.solve_fringes(uvset; ref_ant = 1, adhoc = adhoc, phase_bandpass = false)

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

@testset "Amplitude bandpass: pluggable estimators (poly / penalized / free)" begin
    # Inject a per-BAND log-amp roll-off (the filterbank passband, deep toward each
    # band's high-channel edge) shared by all stations, plus a small per-station
    # ripple. THEN kill one interior channel per band (zero its weight on every
    # baseline). The two SMOOTH estimators (polynomial, penalized) must flatten the
    # band AND estimate the killed channels from the in-spw shape; FreeBandpass must
    # leave the killed channels untouched (|g| = 1).
    nant, nbands, nchan = 4, 2, 8
    nchg = nbands * nchan
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
    uvset, _ = _build_fringe_uvset(; nant = nant, nbands = nbands, nchan = nchan, amp_bandpass = abp)

    # Kill local channel 7 in every band (globals 7 and 15): zero its weight on all
    # baselines so the per-channel solve has NO data there.
    dead_local = 7
    dead_globals = [(b - 1) * nchan + dead_local for b in 1:nbands]
    for (_, leaf) in DimensionalData.branches(uvset)
        parent(leaf[:weights])[dead_local, :, :, :] .= 0.0f0
    end

    adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0)
    larec(sol, plan, a, f, gc) = (off = plan.off1[a, f, 1, 1]; off == 0 ? NaN : sol.θ[off + plan.clocal[gc] - 1])
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

    sol_off = FP.solve_fringes(uvset; ref_ant = 1, adhoc = adhoc, amp_bandpass = false)
    doff = FP.baseline_fringe_data(uvset, sol_off)
    poff = FP.baseline_pol_index(doff, :parallel)
    @test amp_ripple(doff.spec_after, doff, poff) > 1.3    # roll-off ripple without the stage

    # The smooth estimators flatten the band AND fill the killed channels onto the
    # in-spw curve (≈ the mean of the live neighbours, well away from log-amp 0).
    for sm in (FP.PolynomialBandpass(4), FP.PenalizedBandpass(0.1))
        sol = FP.solve_fringes(uvset; ref_ant = 1, adhoc = adhoc, amp_bandpass = true, amp_smoother = sm)
        don = FP.baseline_fringe_data(uvset, sol)
        p = FP.baseline_pol_index(don, :parallel)
        @test amp_ripple(don.spec_after, don, p) < 1.08
        plan = FP._amp_bandpass_plan(sol.model, sol.layout)
        for dg in dead_globals, a in 2:nant, f in 1:2
            nbr = 0.5 * (larec(sol, plan, a, f, dg - 1) + larec(sol, plan, a, f, dg + 1))
            @test isfinite(larec(sol, plan, a, f, dg))
            @test abs(larec(sol, plan, a, f, dg) - nbr) < 0.1    # estimated, on the smooth curve
            @test abs(nbr) > 0.12                                 # ...curve far from |g|=1 (meaningful)
        end
    end

    # FreeBandpass does NOT estimate the killed channels — their θ slot is untouched
    # (log-amp 0 ⇒ |g| = 1), the contrast that motivates the smoothers.
    solf = FP.solve_fringes(uvset; ref_ant = 1, adhoc = adhoc, amp_bandpass = true, amp_smoother = FP.FreeBandpass())
    planf = FP._amp_bandpass_plan(solf.model, solf.layout)
    for dg in dead_globals, a in 2:nant, f in 1:2
        @test larec(solf, planf, a, f, dg) == 0.0
    end
end

@testset "Adhoc auto window (:auto) flattens end-to-end" begin
    # The default `SavitzkyGolaySmoother()` now uses `window = :auto` (EHT-HOPS coherence-time
    # selection). Verify the default path still flattens the per-baseline phase.
    uvset, _ = _build_fringe_uvset()
    sol = FP.solve_fringes(uvset; ref_ant = 1, adhoc = FP.SavitzkyGolaySmoother())   # window = :auto
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
    uvset, _ = _build_fringe_uvset(nbands = 4, nchan = 8, band_sep = 1.5e8)
    geom = CAL.build_geometry(uvset)
    ax = FP._search_axes(geom.channel_freqs, geom.times .* 3600.0, FP.FringeSearch())
    @test ax.mbd !== nothing

    sol = FP.solve_fringes(
        uvset; ref_ant = 1,
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0),
    )
    @test all(>(10), filter(isfinite, sol.info.scan_max_snr))
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
    groups = FP._scan_group_leaves(uvset)
    grp = FP._materialize_concat_group(groups[1], geom)
    f0 = geom.f0
    t0s = geom.t0 * 3600.0
    s = FP.FringeSearch()
    det1, snr1, nc1, rows1 = FP._search_group(grp, grp.Vg, f0, t0s, s, FP._ws_pool(1), 1)
    det4, snr4, nc4, rows4 = FP._search_group(grp, grp.Vg, f0, t0s, s, FP._ws_pool(4), 4)
    # Same detections regardless of the inner task count (≈ only because the two
    # runs plan separate FFTW MEASURE transforms), and the recorded detection
    # table in the same order.
    @test nc1 == nc4
    @test isapprox(snr1, snr4; rtol = 1.0e-9)
    @test size(det1) == size(det4)
    @test all(
        isapprox(det1[i].delay, det4[i].delay; atol = 1.0e-15) &&
            isapprox(det1[i].rate, det4[i].rate; atol = 1.0e-12) &&
            isapprox(det1[i].snr, det4[i].snr; rtol = 1.0e-9) &&
            det1[i].valid == det4[i].valid
            for i in eachindex(det1)
    )
    @test length(rows1) == length(rows4)
    @test all(
        r1.a == r4.a && r1.b == r4.b && r1.pol == r4.pol && isapprox(r1.snr, r4.snr; rtol = 1.0e-9)
            for (r1, r4) in zip(rows1, rows4)
    )

    # Stage timers land in the solution info and print; the progress callback
    # fires per completed scan of each pass (plus a done=0 pass announcement).
    events = Tuple{Symbol, Int, Int}[]
    sol = FP.solve_fringes(
        uvset; ref_ant = 1,
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0),
        progress = (st, d, t) -> push!(events, (st, d, t)),
    )
    ngroups = length(FP._scan_group_leaves(uvset))
    for st in (:search, :adhoc)
        ev = [(d, t) for (s2, d, t) in events if s2 == st]
        @test !isempty(ev)
        total = ev[1][2]
        st === :search && @test total == ngroups
        @test sort([d for (d, _) in ev]) == collect(0:total)   # announcement + every scan
        @test all(t == total for (_, t) in ev)
    end
    @test any(e -> e[1] === :bandpass, events)                 # bandpass enabled by default
    inf = sol.info
    @test inf.t_search_pass > 0 && inf.t_adhoc_pass > 0 && inf.t_bandpass_stage >= 0
    @test inf.ntasks_used >= 1 && inf.inner_tasks >= 1
    for k in (:scan_t_decode, :scan_t_search, :scan_t_decode2, :scan_t_adhoc)
        v = getproperty(inf, k)
        @test length(v) == inf.nscan && all(>=(0), v)
    end
    @test sum(inf.scan_t_search) > 0
    buf = IOBuffer()
    FP.print_solve_timing(sol; io = buf)
    out = String(take!(buf))
    @test occursin("Fringe solve timing", out) && occursin("search pass", out)

    # Capping the bandpass accumulation to the best calibrator scan still
    # produces a working solve (the stage-B/bandpass/adhoc chain is intact).
    ev2 = Tuple{Symbol, Int, Int}[]
    solc = FP.solve_fringes(
        uvset; ref_ant = 1, bandpass_max_scans = 1,
        adhoc = FP.SavitzkyGolaySmoother(; window = 7, order = 2, snr_floor = 0.0),
        progress = (st, d, t) -> push!(ev2, (st, d, t)),
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
    old = CAL.CalibrationSolution(sol.model, sol.layout, sol.geom, sol.θ, (;))
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
        nant = 4, nbands = 8, nchan = 8, ref_freq = 3.0e9, band_sep = 0.5e9,
        dtec = dtec_true, seed = 77, feed_common = true,
    )
    geom = CAL.build_geometry(uvset)
    @test FP._dispersion_enabled(:auto, geom)

    sol = FP.solve_fringes(
        uvset; ref_ant = 1, phase_bandpass = false, amp_bandpass = false,
        search = FP.FringeSearch(algorithm = :full),
    )
    @test sol.info.dispersion_applied
    dplan = FP._dispersion_plan(sol.model, sol.layout)
    @test dplan !== nothing
    for a in 1:4
        off = dplan.off1[a, 1, 1, 1]
        off == 0 && continue
        @test isapprox(sol.θ[off], dtec_true[a] - dtec_true[1]; atol = 0.05)
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

    # Without the term the dispersion survives as cross-band decoherence.
    sol0 = FP.solve_fringes(
        uvset; ref_ant = 1, phase_bandpass = false, amp_bandpass = false,
        search = FP.FringeSearch(algorithm = :full), dispersion = false,
    )
    @test !sol0.info.dispersion_applied
    corr0 = Gustavo.UVData.apply_calibration(uvset, sol0)
    @test _crossband_eta(corr0) < 0.9

    # HDF5 round-trip carries the Dispersion term (generic serialization).
    mktempdir() do dir
        path = joinpath(dir, "disp.h5")
        CAL.save_solution_hdf5(path, sol)
        sol2 = CAL.load_solution_hdf5(path)
        @test sol2.θ == sol.θ
        @test any(tc -> tc.component.term isa CAL.Dispersion, sol2.model.phase)
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
    groups = FP.fringe_band_groups(chf)
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
        nant = 4, nbands = nb, nchan = nch, ref_freq = 3.0e9, chan_bw = 2.0e6,
        band_origins = origins, bandpass = ph, seed = 99, feed_common = true,
    )
    geom = CAL.build_geometry(uvset)
    @test FP._sbd_bands(:auto, geom) !== nothing
    @test FP._sbd_bands(false, geom) === nothing

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

    sol = FP.solve_fringes(
        uvset; ref_ant = 1, phase_bandpass = false, amp_bandpass = false,
        dispersion = false, search = FP.FringeSearch(algorithm = :full),
    )
    @test sol.info.sbd_applied
    sbd = FP._sbd_plans(sol.model, sol.layout)
    @test sbd !== nothing
    # A common-mode slope across groups is gauge-shared with the wideband
    # stage-B delay (and its constants land in the SBD phase columns), so the
    # gauge-invariant recovery check is the ACROSS-GROUP DIFFERENCE.
    Δ(a) = sol.θ[sbd.dplan.off1[a, 1, 1, 1]] - sol.θ[sbd.dplan.off1[a, 1, 1, 2]]
    @test sbd.dplan.off1[2, 1, 1, 1] != 0
    @test isapprox(Δ(2), τ2[1] - τ2[2]; atol = 0.1e-9)          # injected 4 ns split
    @test abs(Δ(3)) < 0.1e-9                                    # clean station ≈ 0
    corr = Gustavo.UVData.apply_calibration(uvset, sol)
    @test _perchan_eta(corr, 1) > 0.98                          # baseline (1,2) flat

    # Without the term the per-group slope survives as within-group decoherence.
    sol0 = FP.solve_fringes(
        uvset; ref_ant = 1, phase_bandpass = false, amp_bandpass = false,
        dispersion = false, sbd = false, search = FP.FringeSearch(algorithm = :full),
    )
    @test !sol0.info.sbd_applied
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
        nant = 4, nbands = 8, nchan = 8, ref_freq = 3.0e9, band_sep = 0.5e9,
        dtec = dtec_true, seed = 77, feed_common = true,
        station_positions = positions,
    )
    ants = Gustavo.UVData.metadata(first(values(Gustavo.UVData.branches(uvset)))).antennas
    @test FP._colocated_ties(ants) == [1, 2, 3, 3]

    # Degenerate positions (the default synthetic table, max sep ≪ 10 km) must
    # NOT tie anything — the guard against missing/zero station_xyz.
    uvd, _ = _build_fringe_uvset(nant = 4, nbands = 2, nchan = 4)
    antd = Gustavo.UVData.metadata(first(values(Gustavo.UVData.branches(uvd)))).antennas
    @test FP._colocated_ties(antd) == [1, 2, 3, 4]

    # The intra-site baseline set derived from the same grouping: exactly the
    # (3,4) twin pair, both orders; empty when the position guard trips.
    @test FP._colocated_pair_set(ants) == Set([(3, 4), (4, 3)])
    @test isempty(FP._colocated_pair_set(antd))

    # TWO co-located pairs must BOTH tie (regression: a `break` in the old
    # comma-nested loop exited both levels after the first pair — on VR2505
    # it tied Onsala and silently skipped the Wettzell twins).
    pos2 = [
        [0.0, 0.0, 0.0], [1.0e5, 0.0, 0.0], [1.0e5 + 70.0, 0.0, 0.0],
        [2.0e5, 0.0, 0.0], [2.0e5 + 60.0, 0.0, 0.0],
    ]
    uv2, _ = _build_fringe_uvset(
        nant = 5, nbands = 2, nchan = 4, station_positions = pos2,
    )
    ant2 = Gustavo.UVData.metadata(first(values(Gustavo.UVData.branches(uv2)))).antennas
    @test FP._colocated_ties(ant2) == [1, 2, 2, 4, 4]
    @test FP._colocated_pair_set(ant2) == Set([(2, 3), (3, 2), (4, 5), (5, 4)])

    sol = FP.solve_fringes(
        uvset; ref_ant = 1, phase_bandpass = false, amp_bandpass = false,
        sbd = false, search = FP.FringeSearch(algorithm = :full),
    )
    dplan = FP._dispersion_plan(sol.model, sol.layout)
    @test dplan !== nothing
    o3 = dplan.off1[3, 1, 1, 1]
    o4 = dplan.off1[4, 1, 1, 1]
    @test o3 != 0 && o4 != 0
    @test sol.θ[o3] == sol.θ[o4]                                # tied EXACTLY
    @test isapprox(sol.θ[o3], dtec_true[3] - dtec_true[1]; atol = 0.05)
end

@testset "Bandpass calibrator: total-SNR selection + coverage top-up" begin
    # Total source SNR picks the workhorse source, not the single brightest scan
    # (a one-scan source can carry the top scan while covering a fraction of the
    # array — on VR2505 that left 10 of 17 stations with NO bandpass).
    srcs = ["A", "A", "A", "B"]
    snr = [500.0, 600.0, 550.0, 900.0]
    @test FP._bandpass_calibrator(nothing, srcs, snr) == "A"
    @test FP._bandpass_calibrator("B", srcs, snr) == "B"        # explicit override
    @test FP._bandpass_calibrator(nothing, srcs, fill(NaN, 4)) == "A"

    # Top-up is a no-op when the calibrator scans already cover every station.
    uvset, _ = _build_fringe_uvset()
    groups = FP._scan_group_leaves(uvset)
    @test isempty(FP._bandpass_coverage_topup(collect(eachindex(groups)), groups, ones(length(groups))))
end

@testset "Budget-scheduled group map" begin
    # Results come back in index order regardless of completion order, and
    # groups pack by their own charge (not the largest group's).
    res, peak = FP._budget_scheduled_map(x -> x * 10, 1:8, [10, 3, 3, 3, 1, 2, 2, 2], 12; max_tasks = 4)
    @test res == [10, 20, 30, 40, 50, 60, 70, 80]
    @test 1 <= peak <= 4

    # A group charged more than the whole budget is clamped so it still runs.
    res2, _ = FP._budget_scheduled_map(x -> x + 1, 1:3, [100, 1, 1], 10; max_tasks = 4)
    @test res2 == [2, 3, 4]

    # Empty input and worker-error propagation.
    res3, peak3 = FP._budget_scheduled_map(identity, Int[], Float64[], 10; max_tasks = 2)
    @test isempty(res3) && peak3 == 0
    @test_throws Exception FP._budget_scheduled_map(
        x -> x == 2 ? error("boom") : x, 1:3, [1, 1, 1], 10; max_tasks = 2,
    )
end

@testset "EHT-HOPS station flags: unconstrained station zero-weighted" begin
    # Station 4 participates (baselines with valid weights) but carries NO
    # fringe — pure weak noise — so after the closure-screened global solve no
    # strong detection constrains it: the EHT-HOPS flag criterion. Its gains
    # stay identity, and apply_calibration must zero-weight its baselines
    # instead of passing the uncalibrated data through at full weight.
    uvset, _ = _build_fringe_uvset(nant = 4, nbands = 2, nchan = 8, ntime = 12, feed_common = true)
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
    sol = FP.solve_fringes(
        uvset; ref_ant = 1, phase_bandpass = false, amp_bandpass = false,
        sbd = false, dispersion = false, search = FP.FringeSearch(algorithm = :full),
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
