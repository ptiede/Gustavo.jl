# FITS-IDI I/O round-trip tests. Builds a small synthetic multi-band UVSet in
# memory, writes it with `write_fitsidi`, reads it back with `load_fitsidi`, and
# asserts leaf-for-leaf equality. `using FITSFiles` triggers the
# `GustavoFITSFilesExt` extension that provides both functions.

using Gustavo
using Test
using FITSFiles
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
        nant = 3, nbands = 2, nchan = 4, nscan = 2, ntime = 3,
        pol_labels = ["PP", "PQ", "QP", "QQ"],
        rdate = "2021-03-04",
        ref_freq = 43.0e9, chan_bw = 2.0e6, band_sep = 1.0e8,
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

        for b in 1:nbands
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
            rt = UV.load_fitsidi(path; lazy = false)

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

                @test parent(oleaf[:flag]) == parent(rleaf[:flag])

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
                @test parent(mleaf[:flag]) == parent(eleaf[:flag])
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
            nbands = 1, nchan = 2, nscan = 1, ntime = 1,
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
        uvset = build_synth_idi_uvset(; nbands = 2, band_sep = 5.0e8)
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
            nbands = 1, nchan = 3, nscan = 1, ntime = 2,
            weight_fn = (band, ti, bl, p) -> p == 2 ? -1.0f0 : 2.0f0,
        )
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            rt = UV.load_fitsidi(path; lazy = false)
            leaf = first(values(DimensionalData.branches(rt)))
            pols = collect(lookup(leaf[:vis], Pol))
            flag = parent(leaf[:flag])    # (Frequency, Ti, Baseline, Pol)
            w = parent(leaf[:weights])
            pq = findfirst(==("PQ"), pols)
            pp = findfirst(==("PP"), pols)
            @test all(flag[:, :, :, pq])           # PQ flagged everywhere
            @test !any(flag[:, :, :, pp])          # PP not flagged
            @test all(w[:, :, :, pq] .<= 0)
            @test all(w[:, :, :, pp] .> 0)
        finally
            isfile(path) && rm(path)
        end
    end
end
