# Synthetic multi-band UVSet with KNOWN injected per-(station, feed) fringe
# parameters (delay/rate/phase + atmospheric screen, optional bandpass/dTEC) —
# THE shared generator for pipeline-level tests, plus the _coherence metric.
# Split out of test_pipeline.jl so fixture scripts and stage tests can build
# synthetic data without running that file's testsets.
using Gustavo
using Test
using Random
using LinearAlgebra: Diagonal
using StructArrays
using DimensionalData
using DimensionalData: DimArray, Ti, dims, lookup
using PolarizedTypes: RPol, LPol
using Gustavo.UVData: Pol, Frequency, UVW, Baseline, UVSet, pol_products, channel_freqs
using Gustavo.UVData: antennas, baselines, source_name, scan_name, frequencies, timestamps
using HDF5   # triggers GustavoHDF5Ext (solution save/load round-trip)
using FITSFiles   # triggers GustavoFITSFilesExt (write_uvfits/load_uvfits round-trip)

const CAL = Gustavo.Calibration
const FP = Gustavo.Fringe
const ST = Gustavo.Streaming
const UVP = Gustavo.UVData

# The default fringe term list. Dispersion (dTEC) and SBD are NOT part of
# `default_fringe_terms()` — they are fit by a separate `DispersionSBDFit`
# step, not by `FringeModel` — so `dispersion`/`sbd` are now no-op kwargs kept
# only so existing call sites (almost all `dispersion = false, sbd = false`,
# i.e. the now-default behavior) don't need touching. A test that wants
# dispersion/SBD actually fit should compose a `_dispersion_sbd_step(...)`
# into its pipeline instead.
_fringe_terms(; dispersion = true, sbd = true) = default_fringe_terms()

# `DispersionSBDFit`, or `nothing` when both halves are disabled — the
# DispersionSBDFit-step equivalent of the old `_fringe_terms(dispersion, sbd)`
# kwargs, for tests that want dTEC/SBD actually fit.
_dispersion_sbd_step(; dispersion = true, sbd = true) =
    !dispersion && !sbd ? nothing :
    DispersionSBDFit(;
        dispersion = dispersion ? DispersionModel() : nothing,
        sbd = sbd ? SingleBandDelay() : nothing,
    )

# ── Synthetic UVSet with injected station fringe parameters ──────────────────
#
# For baseline (a, b) and product p with feeds (fa, fb):
#   V[c, ti] = A · exp(i·[ Δφ + 2π·Δτ·(f_c − f0) + 2π·Δṙ·(t_sec − t0_sec)
#                          + Δscreen[ti] ])
# with Δx = x[a, fa] − x[b, fb]. Times in seconds use t0_sec; freqs in Hz.
function _build_fringe_uvset(;
        nant = 4, nspw = 2, nchan = 8, ntime = 12, nscans = 1,
        scan_gap = nothing,     # hours between scan starts (default: back-to-back)
        noise = nothing,        # σ per visibility sample (same units as the A0 = 2.5 signal);
                                #   without it every estimate is exact and no
                                #   uncertainty-driven effect is observable
        pol_labels = ["PP", "PQ", "QP", "QQ"],
        ref_freq = 230.0e9, chan_bw = 2.0e6, spw_sep = 1.0e8,
        seed = 1234,
        bandpass = nothing,    # optional (nant, 2, nspw*nchan) per-channel phase (rad)
        amp_bandpass = nothing, # optional (nant, 2, nspw*nchan) per-channel log-amp
        dtec = nothing,         # optional (nant,) station TEC (TECU, feed-common)
        feed_common = false,    # tie delay/phi across feeds (zero true inter-feed offset)
        rel_rate = nothing,      # optional (nant,) feed-2 − feed-1 rate offset (Hz) —
                                #   exercises the opt-in RL(rate = ...) solve

        spw_origins = nothing, # optional (nspw,) explicit band start freqs (Hz) — overrides spw_sep
        station_positions = nothing, # optional (nant,) xyz vectors (m) — for co-location tests
        omit_station = nothing, # optional station index present in the antenna table but
                                #   observing no baseline — a station that dropped out
    )
    UV = Gustavo.UVData
    rng = MersenneTwister(seed)
    npol = length(pol_labels)

    ants_v = [
        UV.Antenna(;
                name = "A$(i)",
                # VLBI-scale by default (tens of km apart): co-located grouping
                # rejects a table whose stations sit within a single site.
                station_xyz = station_positions === nothing ?
                Float64[1.0e4 * i, 2.0e4 * i, 3.0e4 * i] : Float64.(station_positions[i]),
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
    omit_station === nothing || filter!(p -> omit_station ∉ p, bl_pairs)
    nbl = length(bl_pairs)
    baselines = UV.BaselineIndex(bl_pairs, bl_pairs; antenna_names = collect(antennas.name))

    setups = UV.FrequencySetup[]
    for b in 1:nspw
        f_lo = spw_origins === nothing ? ref_freq + (b - 1) * spw_sep : Float64(spw_origins[b])
        chf = f_lo .+ (0:(nchan - 1)) .* chan_bw
        push!(
            setups, UV.FrequencySetup(;
                name = "band_$(b)",
                ref_freq = ref_freq,
                channel_freqs = collect(chf),
                ch_widths = fill(chan_bw, nchan),
                total_bandwidths = fill(chan_bw * nchan, nchan),
                sidebands = fill(1.0, nchan),
                extras = (; bandfreq = (b - 1) * spw_sep, band = b),
            ),
        )
    end

    array_obs = UV.ObsArrayMetadata(;
        telescope = "SYNTH", instrume = "SYNTH",
        date_obs = "2021-03-04", equinox = 2000.0f0, bunit = "JY",
        rdate = "2021-03-04", earth_rot_rate = 360.0f0,
        extras = (; correlat = "DiFX", obscode = "SY001"),
    )

    # Times (hours): `nscans` scans of `ntime` APs spaced 30 s, scans separated
    # by a 2-AP gap. Scan 1 keeps the historical single-scan time axis.
    ti_vals = collect((0:(ntime - 1)) .* (30.0 / 3600.0))
    scan_span = scan_gap === nothing ? (ntime + 2) * (30.0 / 3600.0) : Float64(scan_gap)

    # Reference geometry constants (must match what the solve derives):
    # f0 = mean of all channel freqs across both bands; t0 = first time (hours).
    allfreqs = Float64[]
    for fs in setups
        append!(allfreqs, collect(channel_freqs(fs)))
    end
    sort!(unique!(allfreqs))
    f0 = sum(allfreqs) / length(allfreqs)
    t0_sec = ti_vals[1] * 3600.0

    # Injected parameters. Reference antenna 1 = 0 (so the recovered solution matches
    # the gauge), others drawn small. `delay`/`phi` are PER-FEED (their constant inter-feed
    # offset is recovered by the model's global `FeedComponent(2)` delay/const terms);
    # `rate` is FEED-COMMON by default because the fringe model ties rate
    # `SharedFeeds` (the inter-feed rate tied ≡ 0); pass `rel_rate` to inject a feed-2
    # offset for the opt-in `RL(rate = ...)` solve.
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

    rel_rate === nothing || (rate[:, 2] .= rate[:, 1] .+ rel_rate)

    if feed_common
        delay[:, 2] .= delay[:, 1]
        phi[:, 2] .= phi[:, 1]
    end

    # Per-(station, AP) atmospheric screen (rad), reference held at 0, FEED-COMMON:
    # the atmosphere is non-birefringent, so both feeds see the same screen, and the
    # model's adhoc term is `SharedFeeds`. A smooth (slowly-varying) phase track — the
    # regime the Savitzky–Golay adhoc smoother is designed for — plus a per-station
    # constant offset that Stage-B's per-scan constant phase absorbs. Each scan
    # draws its own screen (scan 1's rng sequence matches the historical
    # single-scan builder exactly).
    screen = zeros(nant, 2, ntime, nscans)
    for s in 1:nscans, a in 2:nant
        base = (rand(rng) - 0.5) * 1.0
        for ti in 1:ntime
            v = base + 0.2 * sin(0.5 * ti + a)
            screen[a, 1, ti, s] = v
            screen[a, 2, ti, s] = v
        end
    end

    feeds = [CAL.correlation_feed_pair(p) for p in pol_labels]
    A0 = 2.5

    src_name = "SRC1"
    branches = DimensionalData.TreeDict()
    for s in 1:nscans, b in 1:nspw
        fs = setups[b]
        fch = collect(channel_freqs(fs))
        ti_scan = ti_vals .+ (s - 1) * scan_span
        vis_dense = Array{ComplexF32}(undef, nchan, ntime, nbl, npol)
        w_dense = fill(1.0f3, nchan, ntime, nbl, npol)   # high weight → high SNR
        for p in 1:npol, bl in 1:nbl, ti in 1:ntime, c in 1:nchan
            a, bb = bl_pairs[bl]
            fa, fb = feeds[p]
            f = fch[c]
            tsec = ti_scan[ti] * 3600.0
            dτ = delay[a, fa] - delay[bb, fb]
            dṙ = rate[a, fa] - rate[bb, fb]
            dφ = phi[a, fa] - phi[bb, fb]
            dscr = screen[a, fa, ti, s] - screen[bb, fb, ti, s]
            gc = (b - 1) * nchan + c          # global channel index (bands stacked by freq)
            dbp = bandpass === nothing ? 0.0 : (bandpass[a, fa, gc] - bandpass[bb, fb, gc])
            dla = amp_bandpass === nothing ? 0.0 : (amp_bandpass[a, fa, gc] + amp_bandpass[bb, fb, gc])  # log-amp SUMS
            ddt = dtec === nothing ? 0.0 :
                CAL.DISPERSION_K * (dtec[a] - dtec[bb]) * (1.0 / f0 - 1.0 / f)
            ph = dφ + 2π * dτ * (f - f0) + 2π * dṙ * (tsec - t0_sec) + dscr + dbp + ddt
            z = A0 * exp(dla) * cis(ph)
            noise === nothing ||
                (z += noise * complex(randn(rng), randn(rng)) / sqrt(2))
            vis_dense[c, ti, bl, p] = ComplexF32(z)
        end
        vis_part = DimArray(
            vis_dense,
            (
                Frequency(fch), Ti(ti_scan),
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
            (Ti(ti_scan), Baseline(baselines.labels), UVP.UVW(["U", "V", "W"])),
        )

        info = UV.PartitionInfo(;
            source_name = src_name,
            source_key = UV.sanitize_source(src_name),
            scan_name = "$(s)",
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
    return uvset, (;
        delay, rate, phi,
        screen = nscans == 1 ? reshape(screen, nant, 2, ntime) : screen,
        bandpass, amp_bandpass, dtec, f0, t0_sec, bl_pairs, pol_labels, feeds,
    )
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
