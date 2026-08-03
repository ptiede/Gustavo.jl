# FITS-IDI I/O round-trip tests. Builds a small synthetic multi-band UVSet in
# memory, writes it with `write_fitsidi`, reads it back with `load_fitsidi`, and
# asserts leaf-for-leaf equality. `using FITSFiles` triggers the
# `GustavoFITSFilesExt` extension that provides both functions.

using Gustavo
using Test
using FITSFiles
using Dates: DateTime, Date, Millisecond, Hour
using LinearAlgebra: Diagonal
using StructArrays
using DimensionalData
using DimensionalData: DimArray, Ti, dims, lookup
using Gustavo.UVData: Integration, Pol, Frequency, UVW, Baseline, UVSet, pol_products, channel_freqs
using PolarizedTypes: RPol, LPol

# ── Synthetic multi-band UVSet builder ───────────────────────────────────────
#
# Layout per leaf is (Frequency, Ti, Baseline, Pol). We build the leaves
# directly via `_build_leaf` + `PartitionInfo`, one leaf per (band, scan), so
# they exactly mirror what `load_fitsidi` produces.
#
# `vis_value(band, ti, bl, pol_index)` lets each test inject distinguishable
# values to catch permutation/band/stokes bugs.
function build_synth_idi_uvset(;
        nant = 3, nspw = 2, nchan = 4, nscan = 2, ntime = 3,
        pol_labels = ["PP", "PQ", "QP", "QQ"],
        rdate = "2021-03-04",
        ref_freq = 43.0e9, chan_bw = 2.0e6, spw_sep = 1.0e8,
        weight_fn = (band, ti, bl, p) -> 1.0f0,
        vis_fn = (band, ti, bl, p, c) -> ComplexF32(band + 0.1 * ti + 0.01 * bl + 0.001 * p + 0.0001 * c, -0.5),
    )
    UV = Gustavo.UVData
    npol = length(pol_labels)

    # Antennas (NOSTA = 1:nant).
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
        StructArray(ants_v), Float64[1.0, 2.0, 3.0], "SYNTH",
        (; NOSTA = Int32.(1:nant), DIAMETER = fill(25.0f0, nant)),
    )

    # Baselines: all (a, b) with a < b.
    bl_pairs = Tuple{Int, Int}[(a, b) for a in 1:nant for b in (a + 1):nant]
    nbl = length(bl_pairs)
    baselines = UV.BaselineIndex(bl_pairs, bl_pairs; antenna_names = collect(antennas.name))

    # Per-band frequency setups.
    setups = UV.FrequencySetup[]
    for b in 1:nspw
        chf = ref_freq + (b - 1) * spw_sep .+ (0:(nchan - 1)) .* chan_bw
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
        date_obs = rdate, equinox = 2000.0f0, bunit = "JY",
        rdate = rdate, earth_rot_rate = 360.0f0,
        extras = (; correlat = "DiFX", obscode = "SY001"),
    )

    src_name = "SRC1"
    ra = 1.234
    dec = -0.56

    branches = DimensionalData.TreeDict()
    for scan in 1:nscan
        # Distinct, well-separated time windows per scan so the reader's
        # gap-based segmentation splits them back out. Times in hours.
        t0 = (scan - 1) * 5.0
        ti_vals = collect(t0 .+ (0:(ntime - 1)) .* 0.01)   # 36 s spacing

        # uvw shared across bands.
        uvw_dense = zeros(Float32, ntime, nbl, 3)
        for ti in 1:ntime, bl in 1:nbl
            uvw_dense[ti, bl, 1] = Float32(10 * scan + ti + 0.1 * bl)
            uvw_dense[ti, bl, 2] = Float32(20 * scan + ti + 0.2 * bl)
            uvw_dense[ti, bl, 3] = Float32(30 * scan + ti + 0.3 * bl)
        end
        uvw_part = DimArray(
            uvw_dense,
            (Ti(ti_vals), Baseline(baselines.labels), UVW(["U", "V", "W"])),
        )

        for b in 1:nspw
            fs = setups[b]
            vis_dense = Array{ComplexF32}(undef, nchan, ntime, nbl, npol)
            w_dense = Array{Float32}(undef, nchan, ntime, nbl, npol)
            for p in 1:npol, bl in 1:nbl, ti in 1:ntime, c in 1:nchan
                vis_dense[c, ti, bl, p] = vis_fn(b, ti, bl, p, c)
                w_dense[c, ti, bl, p] = weight_fn(b, ti, bl, p)
            end
            vis_part = DimArray(
                vis_dense,
                (
                    Frequency(collect(channel_freqs(fs))), Ti(ti_vals),
                    Baseline(baselines.labels), Pol(pol_labels),
                ),
            )
            w_part = DimArray(w_dense, dims(vis_part))

            info = UV.PartitionInfo(;
                source_name = src_name,
                source_key = UV.sanitize_source(src_name),
                scan_name = string(scan),
                ra = ra, dec = dec,
                antennas = antennas,
                baselines = baselines,
                record_order = [(ti, bl) for bl in 1:nbl for ti in 1:ntime],
                freq_setup = fs,
                spw_name = "band_$(b)",
                ddi = b - 1,
                basename = "synth_idi",
            )
            leaf = UV._build_leaf(vis_part, w_part, uvw_part; partition_info = info)
            key = UV.partition_key(info)
            branches[key] = leaf
        end
    end

    return UVSet(; metadata = UV.UVMetadata(array_obs), branches = branches)
end

# Match a written leaf to the corresponding read-back leaf by (scan_name, band).
function _leaf_band(leaf)
    fs = DimensionalData.metadata(leaf).freq_setup
    return Int(get(fs.extras, :band, DimensionalData.metadata(leaf).ddi + 1))
end

function _index_leaves_by_scan_band(uvset)
    out = Dict{Tuple{String, Int}, Any}()
    for (_, leaf) in DimensionalData.branches(uvset)
        sn = DimensionalData.metadata(leaf).scan_name
        out[(sn, _leaf_band(leaf))] = leaf
    end
    return out
end

const _F32EPS = 1.0f-4

@testset "FITS-IDI I/O" begin
    UV = Gustavo.UVData

    @testset "round-trip (eager)" begin
        uvset = build_synth_idi_uvset()
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            # :validity — this is a raw write→read fidelity check, comparing the
            # read-back WEIGHT to the synthetic input verbatim. The default :auto would
            # apply the WEIGHTYP=CORRELAT radiometer conversion (f·2Δν·τ·η²), which is
            # correct for analysis but not what a byte round-trip compares against.
            rt = UV.load_fitsidi(path; lazy = false, weight_mode = :validity)

            @test length(DimensionalData.branches(rt)) ==
                length(DimensionalData.branches(uvset))

            orig = _index_leaves_by_scan_band(uvset)
            read_idx = _index_leaves_by_scan_band(rt)
            @test Set(keys(read_idx)) == Set(keys(orig))

            for (k, oleaf) in orig
                rleaf = read_idx[k]
                ovis = parent(oleaf[:vis])
                rvis = parent(rleaf[:vis])
                @test size(ovis) == size(rvis)
                @test maximum(abs.(ovis .- rvis)) < _F32EPS

                ow = parent(oleaf[:weights])
                rw = parent(rleaf[:weights])
                # weights collapse to one value per (stokes, band) on write; the
                # synthetic weights are channel-constant so they round-trip.
                @test maximum(abs.(ow .- rw)) < _F32EPS

                # Flag pattern (weight ≤ 0) round-trips.
                @test (ow .<= 0) == (rw .<= 0)

                # Axis labels.
                @test collect(lookup(oleaf[:vis], Pol)) == collect(lookup(rleaf[:vis], Pol))
                @test collect(lookup(oleaf[:vis], Ti)) ≈ collect(lookup(rleaf[:vis], Ti)) atol = 1.0e-6
                @test collect(lookup(oleaf[:vis], Baseline)) == collect(lookup(rleaf[:vis], Baseline))
                @test collect(lookup(oleaf[:vis], Frequency)) ≈ collect(lookup(rleaf[:vis], Frequency)) rtol = 1.0e-9
            end

            # Antenna names / positions.
            oants = UV.union_antennas(uvset)
            rants = UV.union_antennas(rt)
            @test collect(oants.name) == collect(rants.name)
            @test collect(oants.station_xyz) ≈ collect(rants.station_xyz)

            # Source ra/dec.
            ometa = DimensionalData.metadata(first(values(DimensionalData.branches(uvset))))
            rmeta = DimensionalData.metadata(first(values(DimensionalData.branches(rt))))
            @test ometa.ra ≈ rmeta.ra atol = 1.0e-9
            @test ometa.dec ≈ rmeta.dec atol = 1.0e-9
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "lazy vs eager" begin
        uvset = build_synth_idi_uvset()
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            lazy = UV.load_fitsidi(path; lazy = true)
            eager = UV.load_fitsidi(path; lazy = false)

            @test UV.is_lazy(lazy)
            @test !UV.is_lazy(eager)

            lidx = _index_leaves_by_scan_band(lazy)
            eidx = _index_leaves_by_scan_band(eager)
            @test Set(keys(lidx)) == Set(keys(eidx))
            for (k, lleaf) in lidx
                @test UV.is_lazy(lleaf)
                mleaf = UV.materialize_leaf(lleaf)
                eleaf = eidx[k]
                @test parent(mleaf[:vis]) == parent(eleaf[:vis])
                @test parent(mleaf[:weights]) == parent(eleaf[:weights])
            end
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "bulk group read ≡ per-leaf" begin
        # materialize_group reads a scan's whole contiguous row span in ONE read
        # and extracts every band — must be byte-identical to per-leaf reads.
        uvset = build_synth_idi_uvset(; nspw = 3, nchan = 4, nscan = 2, ntime = 3)
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            lazy = UV.load_fitsidi(path; lazy = true)
            # Group lazy leaves by scan (sibling spws share a row map → bulk path).
            byscan = Dict{String, Vector{Any}}()
            for (_, leaf) in DimensionalData.branches(lazy)
                push!(get!(byscan, DimensionalData.metadata(leaf).scan_name, Any[]), leaf)
            end
            @test !isempty(byscan)
            for (_, leaves) in byscan
                @test length(leaves) == 3                       # 3 bands per scan
                # Confirm the bulk path is actually taken (IDI-backed lazy leaf).
                @test UV._bulk_backend(parent(leaves[1][:vis])) !== nothing
                bulk = UV.materialize_group(leaves)
                perleaf = [UV.materialize_leaf(l) for l in leaves]
                for (b, p) in zip(bulk, perleaf)
                    @test parent(b[:vis]) == parent(p[:vis])
                    @test parent(b[:weights]) == parent(p[:weights])
                    @test parent(b[:uvw]) == parent(p[:uvw])
                end
                # Threaded decode (per-leaf baseline threading) must be identical to
                # the sequential path — tasks write disjoint baseline columns.
                old = UV._DECODE_NTASKS[]
                UV._DECODE_NTASKS[] = 4
                try
                    threaded = UV.materialize_group(leaves)
                    for (t, b) in zip(threaded, bulk)
                        @test parent(t[:vis]) == parent(b[:vis])
                        @test parent(t[:weights]) == parent(b[:weights])
                    end
                finally
                    UV._DECODE_NTASKS[] = old
                end
            end
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "stokes order" begin
        # Inject a distinct magnitude per pol: PP=1, PQ=2, QP=3, QQ=4. A
        # perm-inversion bug would scramble these in the read-back MSv4 order.
        pol_mag = Dict(1 => 1.0, 2 => 2.0, 3 => 3.0, 4 => 4.0)  # p index in MSv4 order
        uvset = build_synth_idi_uvset(;
            nspw = 1, nchan = 2, nscan = 1, ntime = 1,
            vis_fn = (band, ti, bl, p, c) -> ComplexF32(pol_mag[p], 0.0),
        )
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            rt = UV.load_fitsidi(path; lazy = false)
            leaf = first(values(DimensionalData.branches(rt)))
            pols = collect(lookup(leaf[:vis], Pol))
            @test pols == ["PP", "PQ", "QP", "QQ"]
            vis = parent(leaf[:vis])  # (Frequency, Ti, Baseline, Pol)
            for (p, lab) in enumerate(pols)
                expected = Dict("PP" => 1.0, "PQ" => 2.0, "QP" => 3.0, "QQ" => 4.0)[lab]
                @test real(vis[1, 1, 1, p]) ≈ expected atol = _F32EPS
            end
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "band / freq round-trip" begin
        uvset = build_synth_idi_uvset(; nspw = 2, spw_sep = 5.0e8)
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            rt = UV.load_fitsidi(path; lazy = false)
            orig = _index_leaves_by_scan_band(uvset)
            read_idx = _index_leaves_by_scan_band(rt)
            for b in 1:2
                ok = first(filter(k -> k[2] == b, collect(keys(orig))))
                ofreqs = collect(lookup(orig[ok][:vis], Frequency))
                rfreqs = collect(lookup(read_idx[ok][:vis], Frequency))
                @test ofreqs ≈ rfreqs rtol = 1.0e-9
            end
            # Bands must have distinct channel frequencies.
            k1 = first(filter(k -> k[2] == 1, collect(keys(read_idx))))
            k2 = first(filter(k -> k[2] == 2, collect(keys(read_idx))))
            f1 = collect(lookup(read_idx[k1][:vis], Frequency))
            f2 = collect(lookup(read_idx[k2][:vis], Frequency))
            @test !(f1 ≈ f2)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "weights / flags" begin
        # Flag pol index 2 (PQ) on every cell via a non-positive weight.
        uvset = build_synth_idi_uvset(;
            nspw = 1, nchan = 3, nscan = 1, ntime = 2,
            weight_fn = (band, ti, bl, p) -> p == 2 ? -1.0f0 : 2.0f0,
        )
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            rt = UV.load_fitsidi(path; lazy = false)
            leaf = first(values(DimensionalData.branches(rt)))
            pols = collect(lookup(leaf[:vis], Pol))
            w = parent(leaf[:weights])    # (Frequency, Ti, Baseline, Pol)
            pq = findfirst(==("PQ"), pols)
            pp = findfirst(==("PP"), pols)
            @test all(w[:, :, :, pq] .<= 0)        # PQ flagged everywhere
            @test all(w[:, :, :, pp] .> 0)         # PP not flagged
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "weight_mode = :radiometer" begin
        # Δν = chan_bw, τ = INTTIM = min Ti spacing (0.01 h = 36 s); a valid
        # validity weight w becomes w·2·Δν·τ·η². Flag sentinels (≤0) pass through.
        chan_bw = 2.0e6
        tau = 36.0                              # 0.01 h spacing → 36 s
        factor = 2 * chan_bw * tau              # η = 1
        uvset = build_synth_idi_uvset(;
            nspw = 1, nchan = 3, nscan = 1, ntime = 3, chan_bw = chan_bw,
            weight_fn = (band, ti, bl, p) -> p == 2 ? -1.0f0 : 2.0f0,
        )
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            valid = UV.load_fitsidi(path; lazy = false, weight_mode = :validity)
            radio = UV.load_fitsidi(path; lazy = false, weight_mode = :radiometer)
            radio_eta = UV.load_fitsidi(
                path; lazy = false, weight_mode = :radiometer, weight_efficiency = 0.5,
            )
            lv = first(values(DimensionalData.branches(valid)))
            lr = first(values(DimensionalData.branches(radio)))
            le = first(values(DimensionalData.branches(radio_eta)))
            pols = collect(lookup(lv[:vis], Pol))
            pp = findfirst(==("PP"), pols)
            pq = findfirst(==("PQ"), pols)
            wv = parent(lv[:weights]); wr = parent(lr[:weights]); we = parent(le[:weights])
            # Validity weights are the raw correlator values (≈2 here).
            @test all(wv[:, :, :, pp] .≈ 2.0f0)
            # Radiometer weights = validity · 2·Δν·τ (η=1), and ·0.25 for η=0.5.
            @test all(isapprox.(wr[:, :, :, pp], Float32(2 * factor); rtol = 1.0f-4))
            @test all(isapprox.(we[:, :, :, pp], Float32(2 * factor * 0.25); rtol = 1.0f-4))
            # Flag sentinels (≤0) are preserved, not scaled into valid weights.
            @test all(wr[:, :, :, pq] .<= 0)
        finally
            isfile(path) && rm(path)
        end
    end
end

# Per-band a-priori amplitude calibration: `apply_calibration(uvset, spw_cals)`
# selects a distinct AntabCalibration per band (info.ddi + 1). This is the
# format-neutral half of the FITS-IDI a-priori path (`load_fitsidi_apriori`
# builds the per-band cals from GAIN_CURVE + SYSTEM_TEMPERATURE).
@testset "a-priori per-band apply_calibration" begin
    UV = Gustavo.UVData
    BP = Gustavo.UVData

    # Two bands; all weights 1. Flat gain (POLY=[1.0]) + DPFU=1 ⇒ SEFD = Tsys,
    # so the per-baseline amplitude factor is √(Tsys_a·Tsys_b) = Tsys_band.
    uvset = build_synth_idi_uvset(; nant = 3, nspw = 2, nchan = 2, nscan = 1, ntime = 3)
    ant_names = UV.union_antennas(uvset).name

    rdate = DimensionalData.metadata(uvset).array_obs.rdate
    base_dt = DateTime(Date(rdate))
    ts_all = sort!(unique(reduce(vcat, [collect(UV.obs_time(l)) for l in values(UV.branches(uvset))])))
    times = [base_dt + Millisecond(round(Int, t * 3_600_000)) for t in ts_all]
    times = [times[1] - Hour(1); times; times[end] + Hour(1)]   # pad the window

    tsys_band = Dict(1 => 100.0, 2 => 400.0)
    spw_cals = Dict{Int, BP.AntabCalibration}()
    for (b, tsys) in tsys_band
        stns = Dict{String, BP.AntabStation}()
        for nm in ant_names
            gain = BP.AntabGainCurve((1.0, 1.0), [1.0])
            vals = repeat([tsys tsys], length(times), 1)
            series = BP.AntabTsysSeries(times, [(0, :R), (0, :L)], vals)
            stns[String(nm)] = BP.AntabStation(String(nm), gain, series, 0)
        end
        spw_cals[b] = BP.AntabCalibration("synthetic", "synth", 2000, stns)
    end

    # min_elevation_deg = -Inf: synthetic station_xyz aren't real ECEF coords
    # (elevation ill-defined) and the gain curve is flat, so disable the
    # below-horizon cutoff — this test only checks per-band SEFD scaling.
    corr = BP.apply_calibration(
        uvset, spw_cals; on_missing_station = :error, min_elevation_deg = -Inf,
    )

    seen_bands = Set{Int}()
    for (k, leaf_in) in UV.branches(uvset)
        leaf_out = UV.branches(corr)[k]
        band = Int(DimensionalData.metadata(leaf_in).ddi) + 1
        push!(seen_bands, band)
        factor = tsys_band[band]                       # √(SEFD_a·SEFD_b), equal SEFDs
        vin = parent(leaf_in[:vis]); vout = parent(leaf_out[:vis])
        win = parent(leaf_in[:weights]); wout = parent(leaf_out[:weights])
        idx = findfirst(i -> isfinite(vin[i]) && win[i] > 0, eachindex(vin))
        @test idx !== nothing
        @test abs(vout[idx]) ≈ abs(vin[idx]) * factor rtol = 1.0e-6
        @test wout[idx] ≈ win[idx] / factor^2 rtol = 1.0e-6
    end
    @test seen_bands == Set([1, 2])                    # both bands exercised, distinctly
end

# ── FLAG-table support ────────────────────────────────────────────────────────
#
# `write_fitsidi` does not emit a FLAG table, so these tests construct one by
# hand: write a base file with `write_fitsidi`, reopen it to collect its HDUs,
# append a FLAG bintable HDU (built directly with FITSFiles), and rewrite. Then
# `load_fitsidi` reads the FLAG table and applies it to the lazy weight/flag
# layers. We do NOT modify fitsidi_write.jl.

using FITSFiles: HDU, Bintable, Card, fits

# Build + write a FITS-IDI file that carries `flag_rows`. Each flag row is a
# NamedTuple with fields SOURCE_ID, ANTS (length-2 vector, NOSTA numbers, 0 =
# wildcard), FREQID, TIMERANG (length-2 vector, DAYS rel. RDATE), BANDS
# (length-no_band), CHANS (length-2, 1-based, (0,0)=all), PFLAGS (length-no_stkd,
# on-disk RR/LL/RL/LR order), SEVERITY. The base file uses ≥2 bands so the
# vector columns are length>1 (FITSFiles cannot serialize length-1 vector
# columns).
function write_idi_with_flags(path, uvset, flag_rows; no_band, no_stkd, no_chan)
    UV = Gustavo.UVData
    base = tempname() * ".idifits"
    UV.write_fitsidi(base, uvset)
    fid = fits(base)
    hdus = collect(fid)

    nrow = length(flag_rows)
    flagdata = (
        SOURCE_ID = Int32[Int32(r.SOURCE_ID) for r in flag_rows],
        ARRAY = fill(Int32(1), nrow),
        ANTS = [Int32.(collect(r.ANTS)) for r in flag_rows],
        FREQID = Int32[Int32(r.FREQID) for r in flag_rows],
        TIMERANG = [Float32.(collect(r.TIMERANG)) for r in flag_rows],
        BANDS = [Int32.(collect(r.BANDS)) for r in flag_rows],
        CHANS = [Int32.(collect(r.CHANS)) for r in flag_rows],
        PFLAGS = [Int32.(collect(r.PFLAGS)) for r in flag_rows],
        REASON = [rpad("TEST", 24) for _ in flag_rows],
        SEVERITY = Int32[Int32(r.SEVERITY) for r in flag_rows],
    )
    flagcards = Card[
        Card("EXTNAME", "FLAG"),
        Card("EXTVER", Int32(1)),
        Card("NO_STKD", Int32(no_stkd)),
        Card("NO_BAND", Int32(no_band)),
        Card("NO_CHAN", Int32(no_chan)),
    ]
    flag_hdu = HDU(Bintable, flagdata, flagcards)
    write(path, vcat(hdus, [flag_hdu]))
    isfile(base) && rm(base)
    return path
end

@testset "FITS-IDI FLAG table" begin
    UV = Gustavo.UVData

    @testset "synthetic FLAG selection" begin
        # 2 bands so the FLAG vector columns are length>1. Antennas NOSTA 1:3,
        # baselines (1,2),(1,3),(2,3). All weights positive so the only flags
        # come from the FLAG table.
        nspw = 2
        nchan = 4
        nant = 3
        ntime = 3
        uvset = build_synth_idi_uvset(;
            nant = nant, nspw = nspw, nchan = nchan, nscan = 1, ntime = ntime,
            weight_fn = (band, ti, bl, p) -> 2.0f0,
        )
        # The synthetic scan times are 0.0, 0.01, 0.02 hours (scan 1).
        # Flag: antenna 2, band 1, channels 2..3, pol RR(disk idx1 → MSv4 PP),
        # over time 0.005..0.025 hours → in days = /24.
        thi = 0.025 / 24.0
        tlo = 0.005 / 24.0
        flag_rows = [
            (;
                SOURCE_ID = 0, ANTS = [2, 0], FREQID = 0,
                TIMERANG = [tlo, thi],
                BANDS = [1, 0],                  # band 1 only
                CHANS = [2, 3],                  # channels 2..3
                PFLAGS = [1, 0, 0, 0],           # RR only (disk order)
                SEVERITY = -1,
            ),
        ]
        path = tempname() * ".idifits"
        try
            write_idi_with_flags(
                path, uvset, flag_rows;
                no_band = nspw, no_stkd = 4, no_chan = nchan,
            )
            rt = UV.load_fitsidi(path; lazy = true)
            idx = _index_leaves_by_scan_band(rt)

            # Band 1 leaf: flags should appear; band 2 leaf: none.
            leaf1 = idx[("1", 1)]
            leaf2 = idx[("1", 2)]
            m1 = UV.materialize_leaf(leaf1)
            m2 = UV.materialize_leaf(leaf2)

            # A cell is flagged iff its weight is ≤ 0: the reader bakes every
            # FLAG-table flag into the weights.
            w1 = parent(m1[:weights])
            f1 = w1 .<= 0               # (Frequency, Ti, Baseline, Pol)
            f2 = parent(m2[:weights]) .<= 0

            pols = collect(lookup(leaf1[:vis], Pol))   # MSv4 order
            pp = findfirst(==("PP"), pols)             # RR → PP
            qq = findfirst(==("QQ"), pols)
            bls = collect(lookup(leaf1[:vis], Baseline))
            # Baseline columns touching antenna 2: (1,2) and (2,3).
            touch2 = findall(b -> occursin("A2", string(b)), bls)
            notouch = setdiff(1:length(bls), touch2)
            # Time indices within the flag window 0.005..0.025h: ti 2,3 (0.01,0.02).
            tin = [2, 3]
            tout = [1]

            # Flagged cells: band 1, baselines touching A2, ti in window, chans
            # 2..3, pol PP.
            for bi in touch2, ti in tin, c in 2:3
                @test f1[c, ti, bi, pp]
                @test w1[c, ti, bi, pp] == 0
            end
            # NOT flagged: other pol (QQ).
            for bi in touch2, ti in tin, c in 2:3
                @test !f1[c, ti, bi, qq]
                @test w1[c, ti, bi, qq] > 0
            end
            # NOT flagged: channels outside 2..3.
            for bi in touch2, ti in tin, c in (1, 4)
                @test !f1[c, ti, bi, pp]
            end
            # NOT flagged: baselines not touching A2.
            for bi in notouch, ti in tin, c in 2:3
                @test !f1[c, ti, bi, pp]
            end
            # NOT flagged: time outside the window.
            for bi in touch2, ti in tout, c in 2:3
                @test !f1[c, ti, bi, pp]
            end
            # Band 2 leaf: completely unflagged (BANDS[2] == 0).
            @test !any(f2)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "wildcard antenna + all-channel + pol flag" begin
        nspw = 2
        nchan = 3
        nant = 3
        ntime = 2
        uvset = build_synth_idi_uvset(;
            nant = nant, nspw = nspw, nchan = nchan, nscan = 1, ntime = ntime,
            weight_fn = (band, ti, bl, p) -> 1.0f0,
        )
        # Wildcard antenna (ANTS=(0,0)), all channels (CHANS=(0,0)), all times
        # (TIMERANG=(0,0)), pol LL(disk idx2 → MSv4 QQ), band 2 only.
        flag_rows = [
            (;
                SOURCE_ID = 0, ANTS = [0, 0], FREQID = 0,
                TIMERANG = [0.0, 0.0],
                BANDS = [0, 1],
                CHANS = [0, 0],
                PFLAGS = [0, 1, 0, 0],   # LL only (disk order)
                SEVERITY = -1,
            ),
        ]
        path = tempname() * ".idifits"
        try
            write_idi_with_flags(
                path, uvset, flag_rows;
                no_band = nspw, no_stkd = 4, no_chan = nchan,
            )
            rt = UV.load_fitsidi(path; lazy = true)
            idx = _index_leaves_by_scan_band(rt)
            m2 = UV.materialize_leaf(idx[("1", 2)])
            m1 = UV.materialize_leaf(idx[("1", 1)])
            w2 = parent(m2[:weights])
            f2 = w2 .<= 0
            pols = collect(lookup(idx[("1", 2)][:vis], Pol))
            qq = findfirst(==("QQ"), pols)
            others = setdiff(1:length(pols), [qq])

            # Band 2, QQ pol: everything flagged (all ants, chans, times).
            @test all(f2[:, :, :, qq])
            @test all(w2[:, :, :, qq] .== 0)
            # Other pols on band 2: untouched.
            for p in others
                @test !any(f2[:, :, :, p])
                @test all(w2[:, :, :, p] .> 0)
            end
            # Band 1: untouched entirely.
            @test !any(parent(m1[:weights]) .<= 0)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "real-file FLAG smoke test" begin
        real_path = joinpath(
            @__DIR__, "..", "testdata", "BT164", "BT164A",
            "VLBA_BT164A_bt164aQband_BIN0_SRC0_0_260108T230453.idifits",
        )
        if !isfile(real_path)
            @test_skip "real FITS-IDI file not present"
        else
            # Build the UVSet lazily; this must NOT read the FLUX matrix.
            uvset = UV.load_fitsidi(real_path; lazy = true)
            leaves = collect(DimensionalData.branches(uvset))
            @test !isempty(leaves)

            # Find a leaf whose scan/band the FLAG table actually touches. The
            # real FLAG rows flag whole (antenna, band, time-range) selections,
            # so scan the leaves for one that picks up FLAG-table flags on cells
            # whose visibility is present (finite) — i.e. flags the pre-fix
            # reader (weight<=0 only) would have missed.
            found = false
            for (_, leaf) in leaves
                m = UV.materialize_leaf(leaf)
                flag = parent(m[:weights]) .<= 0
                vis = parent(m[:vis])
                # A flagged cell whose vis is finite cannot be a "missing row"
                # (those are NaN); on this file all on-disk weights are
                # positive, so such a flag can only come from the FLAG table.
                flagged_with_data = flag .& isfinite.(real.(vis)) .& isfinite.(imag.(vis))
                n = count(flagged_with_data)
                if n > 0
                    @info "real-file FLAG smoke" scan = DimensionalData.metadata(leaf).scan_name n_flag = count(flag) n_flag_with_data = n size(flag)
                    @test n > 0
                    found = true
                    break
                end
            end
            @test found
        end
    end
end

@testset "FITS-IDI Int16/TSCAL flux decode (E3)" begin
    # Exercises the reader's scaled integer FLUX decode path (DiFX Int16 variant),
    # which the real Float32 validation file does not cover. `_swap_into!` is the
    # per-row hot loop: big-endian on-disk integers → host order (ntoh) → Float32,
    # applying TSCAL/TZERO when `scale = true`.
    ext = Base.get_extension(Gustavo, :GustavoFITSFilesExt)
    @test ext !== nothing

    raw = Int16[2, 4, -6, 1000, -32768]
    rawbuf = UInt8[]
    for v in raw
        append!(rawbuf, reinterpret(UInt8, [hton(v)]))      # big-endian on disk
    end
    nb = length(raw)
    # Duck-typed field with active TSCAL/TZERO: decoded = zero + scale*raw.
    field = (; type = Int16, zero = 100.0, scale = 0.5)

    cube = zeros(Float32, nb)
    ext._swap_into!(cube, rawbuf, Int16, nb, field, true)
    @test cube ≈ Float32[100 + 0.5 * r for r in raw]

    # scale = false: byte-swap only, no TSCAL/TZERO applied.
    cube2 = zeros(Float32, nb)
    ext._swap_into!(cube2, rawbuf, Int16, nb, field, false)
    @test cube2 ≈ Float32.(raw)

    # scale_value agrees on a scalar (the primitive _swap_into! calls).
    @test FITSFiles.scale_value(Int16(8), field, true) ≈ 104.0
end
