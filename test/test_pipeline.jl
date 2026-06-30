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
    )
    UV = Gustavo.UVData
    rng = MersenneTwister(seed)
    npol = length(pol_labels)

    ants_v = [
        UV.Antenna(;
                name = "A$(i)",
                station_xyz = Float64[100.0 * i, 200.0 * i, 300.0 * i],
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
        chf = ref_freq + (b - 1) * band_sep .+ (0:(nchan - 1)) .* chan_bw
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
            ph = dφ + 2π * dτ * (f - f0) + 2π * dṙ * (tsec - t0_sec) + dscr + dbp
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
    return uvset, (; delay, rate, phi, screen, bandpass, amp_bandpass, f0, t0_sec, bl_pairs, pol_labels, feeds)
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
        adhoc = FP.AdhocPhasing(; window = 7, order = 2, snr_floor = 0.0),
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
    adhoc = FP.AdhocPhasing(; window = 7, order = 2, snr_floor = 0.0)

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
        adhoc = FP.AdhocPhasing(; window = 7, order = 2, snr_floor = 0.0),
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

@testset "Fringe pipeline: rounds > 1 accumulates (no corruption)" begin
    # Regression for the θ-overwrite bug: even rounds previously wiped the
    # round-1 solution (coherence collapsed). Accumulation keeps all rounds good.
    uvset, _ = _build_fringe_uvset()
    adhoc = FP.AdhocPhasing(; window = 7, order = 2, snr_floor = 0.0)
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
    adhoc = FP.AdhocPhasing(; window = 7, order = 2, snr_floor = 0.0)
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

    adhoc = FP.AdhocPhasing(; window = 7, order = 2, snr_floor = 0.0)
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
