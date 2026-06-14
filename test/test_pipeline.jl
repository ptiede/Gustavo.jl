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

    # Injected per-(station, feed) parameters. Reference antenna 1 = 0 (so the
    # recovered solution matches the gauge), others drawn small.
    delay = zeros(nant, 2)         # seconds
    rate = zeros(nant, 2)          # Hz
    phi = zeros(nant, 2)           # rad
    for a in 2:nant, f in 1:2
        delay[a, f] = (rand(rng) - 0.5) * 2.0e-9      # ±1 ns  (« 1/chan_bw)
        rate[a, f] = (rand(rng) - 0.5) * 2.0e-3       # ±1 mHz
        phi[a, f] = (rand(rng) - 0.5) * 2.0           # ±1 rad
    end

    # Per-(station, feed, AP) atmospheric screen (rad), reference held at 0. A
    # smooth (slowly-varying) phase track — the regime the Savitzky–Golay adhoc
    # smoother is designed for — plus a per-(station, feed) constant offset that
    # Stage-B's per-scan constant phase absorbs.
    screen = zeros(nant, 2, ntime)
    for a in 2:nant, f in 1:2
        base = (rand(rng) - 0.5) * 1.0
        for ti in 1:ntime
            screen[a, f, ti] = base + 0.2 * sin(0.5 * ti + a + f)
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
            ph = dφ + 2π * dτ * (f - f0) + 2π * dṙ * (tsec - t0_sec) + dscr
            vis_dense[c, ti, bl, p] = ComplexF32(A0 * cis(ph))
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
    return uvset, (; delay, rate, phi, screen, f0, t0_sec, bl_pairs, pol_labels, feeds)
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
