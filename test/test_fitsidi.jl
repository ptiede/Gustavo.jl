# FITS-IDI I/O round-trip tests. Builds a small synthetic multi-band UVSet in
# memory, writes it with `write_fitsidi`, reads it back with `load_fitsidi`, and
# asserts leaf-for-leaf equality. `using FITSFiles` triggers the
# `GustavoFITSFilesExt` extension that provides both functions.

using Gustavo
using Test
using FITSFiles
using Dates: DateTime, Date, Millisecond, Hour, unix2datetime, datetime2unix
using LinearAlgebra: Diagonal
using StructArrays
using DimensionalData
using DimensionalData: DimArray, Ti, dims, lookup
using Gustavo.UVData: Pol, Frequency, UVW, Baseline, UVSet, pol_products, channel_freqs
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
        include_autocorr = false,
        weight_fn = (band, ti, bl, p) -> 1.0f0,
        flag_fn = (band, ti, bl, p, c) -> false,
        vis_fn = (band, ti, bl, p, c) -> ComplexF32(band + 0.1 * ti + 0.01 * bl + 0.001 * p + 0.0001 * c, -0.5),
    )
    UV = Gustavo.UVData
    npol = length(pol_labels)

    # Antennas (NOSTA = 1:nant).
    ants_v = [
        UV.Antenna(;
            name = "A$(i)",
            station_xyz = Float64[100.0 * i, 200.0 * i, 300.0 * i],
            mount = (UV.MountAltAz(), UV.MountNasmythR((0.5, 0.0, 0.0)), UV.MountEquatorial())[mod1(i, 3)],
            nominal_basis = (RPol(), LPol()),
            pol_angles = (0.1f0 * i, 0.2f0 * i),
        )
            for i in 1:nant
    ]
    antennas = UV.AntennaTable(
        StructArray(ants_v), "SYNTH",
        (; NOSTA = Int32.(1:nant), DIAMETER = fill(25.0f0, nant)),
    )

    # Baselines: all (a, b) with a < b, plus the (a, a) autocorrelations when
    # `include_autocorr` (which the reader consumes as normalizers).
    bl_pairs = Tuple{Int, Int}[
        (a, b) for a in 1:nant for b in (include_autocorr ? a : a + 1):nant
    ]
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
        # gap-based segmentation splits them back out. Absolute seconds,
        # anchored at 0h UTC on RDATE.
        t0 = datetime2unix(DateTime(Date(rdate))) + (scan - 1) * 18000.0   # scans 5 h apart
        ti_vals = collect(t0 .+ (0:(ntime - 1)) .* 36.0)

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
            f_dense = Array{Bool}(undef, nchan, ntime, nbl, npol)
            for p in 1:npol, bl in 1:nbl, ti in 1:ntime, c in 1:nchan
                vis_dense[c, ti, bl, p] = vis_fn(b, ti, bl, p, c)
                w_dense[c, ti, bl, p] = weight_fn(b, ti, bl, p)
                f_dense[c, ti, bl, p] =
                    flag_fn(b, ti, bl, p, c) || !(w_dense[c, ti, bl, p] > 0)
            end
            vis_part = DimArray(
                vis_dense,
                (
                    Frequency(collect(channel_freqs(fs))), Ti(ti_vals),
                    Baseline(baselines.labels), Pol(pol_labels),
                ),
            )
            w_part = DimArray(w_dense, dims(vis_part))
            f_part = DimArray(f_dense, dims(vis_part))

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
            leaf = UV._build_leaf(vis_part, w_part, uvw_part, f_part; partition_info = info)
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

# The `TFORM` of the named UV_DATA column, as the header spells it.
function _tform_of(uv_hdu, name::AbstractString)
    i = 1
    while haskey(uv_hdu.cards, "TTYPE$i")
        strip(uv_hdu.cards["TTYPE$i"]) == name && return strip(uv_hdu.cards["TFORM$i"])
        i += 1
    end
    error("UV_DATA has no $name column")
end

# A copy of `uvset` whose visibility layers are `ComplexF64`.
function _widen_vis(uvset)
    branches = DimensionalData.TreeDict()
    for (k, leaf) in DimensionalData.branches(uvset)
        vis = leaf[:vis]
        branches[k] = Gustavo.UVData.rebuild_visibilities(
            leaf, DimArray(ComplexF64.(parent(vis)), dims(vis))
        )
    end
    return UVSet(; metadata = DimensionalData.metadata(uvset), branches)
end

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
            @test collect(oants.mount) == collect(rants.mount)
            @test [collect(p) for p in rants.pol_angles] ≈ [collect(p) for p in oants.pol_angles]

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
                threaded = UV.materialize_group(leaves; executor = DynamicScheduler(ntasks = 4))
                for (t, b) in zip(threaded, bulk)
                    @test parent(t[:vis]) == parent(b[:vis])
                    @test parent(t[:weights]) == parent(b[:weights])
                end
            end
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "merge_spws: bands → one Frequency axis" begin
        # `merge_spws=true` yields one leaf per scan whose Frequency axis is every
        # band's channels concatenated, and that leaf stays lazy: nothing is read
        # until it is materialized.
        uvset = build_synth_idi_uvset(; nspw = 3, nchan = 4, nscan = 2, ntime = 3)
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            perband = UV.load_fitsidi(path; lazy = true)
            merged = UV.load_fitsidi(path; lazy = true, merge_spws = true)

            # One leaf per scan instead of one per (scan, band).
            @test length(DimensionalData.branches(merged)) == 2
            @test length(DimensionalData.branches(perband)) == 6

            # Merged leaves are still disk-backed, and still take the bulk path.
            for (_, leaf) in DimensionalData.branches(merged)
                @test UV.is_lazy(leaf)
                @test DimensionalData.metadata(leaf).spw_name == "combined"
            end

            # Per-scan: the merged data equals the per-band leaves concatenated
            # along Frequency (bands ordered by ascending frequency).
            byscan = Dict{String, Vector{Any}}()
            for (_, leaf) in DimensionalData.branches(perband)
                push!(get!(byscan, DimensionalData.metadata(leaf).scan_name, Any[]), leaf)
            end
            for (_, mleaf) in DimensionalData.branches(merged)
                scan = DimensionalData.metadata(mleaf).scan_name
                bands = sort(
                    byscan[scan];
                    by = l -> minimum(channel_freqs(DimensionalData.metadata(l).freq_setup)),
                )
                mat = UV.materialize_leaf(mleaf)

                exp_vis = cat([parent(UV.materialize_leaf(l)[:vis]) for l in bands]...; dims = 1)
                exp_w = cat([parent(UV.materialize_leaf(l)[:weights]) for l in bands]...; dims = 1)
                @test parent(mat[:vis]) == exp_vis
                @test parent(mat[:weights]) == exp_w

                # Frequency axis is the concatenation, ascending; uvw is
                # frequency-independent and carries through unchanged.
                exp_freqs = reduce(
                    vcat, [collect(lookup(l[:vis], Frequency)) for l in bands],
                )
                @test collect(lookup(mat[:vis], Frequency)) ≈ exp_freqs
                @test issorted(exp_freqs)
                @test parent(mat[:uvw]) == parent(UV.materialize_leaf(bands[1])[:uvw])

                # Partial reads: a channel sub-range spanning a band boundary must
                # decode the same values as the full read (exercises the scalar
                # span path rather than the whole-band fast path).
                @test mleaf[:vis][3:9, :, :, :] == exp_vis[3:9, :, :, :]
                @test mleaf[:weights][3:9, :, :, :] == exp_w[3:9, :, :, :]
                @test mleaf[:vis][2:3, 1:2, :, 1:2] == exp_vis[2:3, 1:2, :, 1:2]
            end

            # Equivalent to applying combine_spw to the per-band set.
            viacombine = UV.combine_spw(UV.load_fitsidi(path; lazy = false))
            mkey(s) = DimensionalData.metadata(s).scan_name
            cidx = Dict(mkey(l) => l for (_, l) in DimensionalData.branches(viacombine))
            for (_, mleaf) in DimensionalData.branches(merged)
                cleaf = cidx[mkey(mleaf)]
                mat = UV.materialize_leaf(mleaf)
                @test parent(mat[:vis]) == parent(cleaf[:vis])
                @test parent(mat[:weights]) == parent(cleaf[:weights])
                @test collect(lookup(mat[:vis], Frequency)) ≈
                    collect(lookup(cleaf[:vis], Frequency))
            end
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "merge_spws: autocorr normalization" begin
        # `normalize_autocorr` (on by default) divides each cross by √(A_a·A_b),
        # which the merged reader must apply per band out of the shared row span.
        # Give each antenna a distinct autocorr amplitude so a mis-indexed
        # normalizer cannot coincidentally produce the right answer.
        uvset = build_synth_idi_uvset(;
            nant = 3, nspw = 2, nchan = 4, nscan = 1, ntime = 2,
            include_autocorr = true,
            vis_fn = (band, ti, bl, p, c) ->
            ComplexF32(band + 0.1 * ti + 0.37 * bl + 0.001 * p + 0.0001 * c, -0.5),
        )
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            perband = UV.load_fitsidi(path; lazy = true, normalize_autocorr = true)
            merged = UV.load_fitsidi(
                path; lazy = true, merge_spws = true, normalize_autocorr = true,
            )

            # Autocorrelations are consumed, so only cross baselines remain.
            for (_, leaf) in DimensionalData.branches(merged)
                @test all(a != b for (a, b) in DimensionalData.metadata(leaf).baselines.pairs)
            end

            bands = sort(
                collect(values(DimensionalData.branches(perband)));
                by = l -> minimum(channel_freqs(DimensionalData.metadata(l).freq_setup)),
            )
            exp_vis = cat([parent(UV.materialize_leaf(l)[:vis]) for l in bands]...; dims = 1)
            exp_w = cat([parent(UV.materialize_leaf(l)[:weights]) for l in bands]...; dims = 1)

            mleaf = only(values(DimensionalData.branches(merged)))
            mat = UV.materialize_leaf(mleaf)
            @test parent(mat[:vis]) == exp_vis
            @test parent(mat[:weights]) == exp_w
            # Partial read across the band boundary takes the scalar span path,
            # which recomputes the normalizer for its own sub-range.
            @test mleaf[:vis][3:6, :, :, :] == exp_vis[3:6, :, :, :]

            # Normalization actually changed the data (guards against the test
            # passing because both sides skipped it).
            plain = UV.load_fitsidi(
                path; lazy = true, merge_spws = true, normalize_autocorr = false,
                drop_autocorr = true,
            )
            @test parent(UV.materialize_leaf(only(values(DimensionalData.branches(plain))))[:vis]) !=
                parent(mat[:vis])
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "merge_spws: single band is a no-op" begin
        # With one band per scan there is nothing to concatenate; the leaf keeps
        # its own band identity rather than becoming a "combined" leaf.
        uvset = build_synth_idi_uvset(; nspw = 1, nchan = 4, nscan = 1, ntime = 2)
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            merged = UV.load_fitsidi(path; lazy = true, merge_spws = true)
            @test length(DimensionalData.branches(merged)) == 1
            for (_, leaf) in DimensionalData.branches(merged)
                @test DimensionalData.metadata(leaf).spw_name == "band_1"
            end
            plain = UV.load_fitsidi(path; lazy = true)
            for ((_, m), (_, p)) in zip(
                    DimensionalData.branches(merged), DimensionalData.branches(plain),
                )
                @test parent(UV.materialize_leaf(m)[:vis]) ==
                    parent(UV.materialize_leaf(p)[:vis])
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

    @testset "the file is written at the precision the set holds" begin
        ext = Base.get_extension(Gustavo, :GustavoFITSFilesExt)
        narrow = build_synth_idi_uvset(; nspw = 1)
        wide = _widen_vis(narrow)
        @test eltype(first(values(UV.branches(wide)))[:vis]) == ComplexF64

        # UVFITS random groups carry one type for the data and every group
        # parameter, so the set's precision is the file's BITPIX.
        for (uvset, bitpix, vis_eltype) in (
                (narrow, -32, ComplexF32), (wide, -64, ComplexF64),
            )
            path = tempname() * ".uvfits"
            try
                @test UV.write_uvfits(path, uvset) == path
                hdus = FITSFiles.fits(path)
                @test hdus[1].cards["BITPIX"] == bitpix
                rt = UV.load_uvfits(path)
                @test eltype(first(values(UV.branches(rt)))[:vis]) == vis_eltype
            finally
                isfile(path) && rm(path)
            end
        end

        # A FITS-IDI column carries its own TFORM, so FLUX widens to `D` on its
        # own — the weights beside it stay `E`, as their layer is Float32.
        path = tempname() * ".idifits"
        try
            @test UV.write_fitsidi(path, wide) == path
            uv = ext._idi_find_hdu(FITSFiles.fits(path), "UV_DATA")
            @test _tform_of(uv, "FLUX")[end] == 'D'
            @test _tform_of(uv, "WEIGHT")[end] == 'E'
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

    ts_all = sort!(unique(reduce(vcat, [collect(UV.obs_time(l)) for l in values(UV.branches(uvset))])))
    times = [unix2datetime(t) for t in ts_all]
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

    # The public verb is `calibrate(spw_cals, uvset)`; same correction.
    corr_pub = calibrate(
        spw_cals, uvset; on_missing_station = :error, min_elevation_deg = -Inf,
    )
    for (k, leaf) in UV.branches(corr_pub)
        @test isequal(parent(leaf[:vis]), parent(UV.branches(corr)[k][:vis]))
    end

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
# These tests construct a FLAG table by hand: write a base file with
# `write_fitsidi`, reopen it to collect its HDUs, append a FLAG bintable HDU
# (built directly with FITSFiles), and rewrite. `write_fitsidi` emits its own
# FLAG table, but only in the narrow form it needs — one band, one stokes and an
# explicit antenna pair, channel range and time range per row. Building rows by
# hand is what reaches the wildcard encodings a producer may use and the reader
# must honor: `ANTS = (0, 0)`, `CHANS = (0, 0)`, `TIMERANG = (0, 0)`.

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
        # The synthetic scan-1 times are 0, 36 and 72 s past RDATE 0h.
        # Flag: antenna 2, band 1, channels 2..3, pol RR (disk idx1 → MSv4 PP),
        # over 18..90 s, which covers the second and third samples. TIMERANG is
        # in days relative to RDATE.
        tlo = 18.0 / 86400.0
        thi = 90.0 / 86400.0
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

            # The FLAG table backs the `:flags` layer alone; the weights carry
            # the WEIGHT column unaltered.
            w1 = parent(m1[:weights])
            f1 = parent(m1[:flags])     # (Frequency, Ti, Baseline, Pol)
            f2 = parent(m2[:flags])

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
                # The flag does not touch the weight: the flagged pol's weight
                # still matches the unflagged pol's at the same cell.
                @test w1[c, ti, bi, pp] == w1[c, ti, bi, qq]
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
            f2 = parent(m2[:flags])
            pols = collect(lookup(idx[("1", 2)][:vis], Pol))
            qq = findfirst(==("QQ"), pols)
            others = setdiff(1:length(pols), [qq])

            # Band 2, QQ pol: everything flagged (all ants, chans, times).
            @test all(f2[:, :, :, qq])
            @test all(w2[:, :, :, qq] .> 0)
            # Other pols on band 2: untouched.
            for p in others
                @test !any(f2[:, :, :, p])
                @test all(w2[:, :, :, p] .> 0)
            end
            # Band 1: untouched entirely.
            @test !any(parent(m1[:flags]))
            @test all(parent(m1[:weights]) .> 0)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "write_fitsidi round-trips the flag layer" begin
        nspw = 2
        nchan = 4
        ntime = 3
        # A channel run on one baseline and pol of band 1, over the last two
        # integrations, with every weight positive — so the flag survives only
        # if the writer emits a FLAG table and the reader decodes it.
        uvset = build_synth_idi_uvset(;
            nant = 3, nspw = nspw, nchan = nchan, nscan = 1, ntime = ntime,
            weight_fn = (band, ti, bl, p) -> 3.0f0,
            flag_fn = (band, ti, bl, p, c) ->
                band == 1 && bl == 2 && p == 3 && ti >= 2 && 2 <= c <= 3,
        )
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            rt = UV.load_fitsidi(path; lazy = true, weight_mode = :validity)
            orig = _index_leaves_by_scan_band(uvset)
            read_idx = _index_leaves_by_scan_band(rt)
            @test Set(keys(read_idx)) == Set(keys(orig))
            for (k, oleaf) in orig
                m = UV.materialize_leaf(read_idx[k])
                @test parent(m[:flags]) == parent(UV.materialize_leaf(oleaf)[:flags])
                # The flags cost no weight on either side of the round trip.
                @test all(parent(m[:weights]) .> 0)
            end
            @test any(parent(UV.materialize_leaf(read_idx[("1", 1)])[:flags]))
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
            # whose visibility is present (finite).
            found = false
            for (_, leaf) in leaves
                m = UV.materialize_leaf(leaf)
                flag = parent(m[:flags])
                vis = parent(m[:vis])
                # A flagged cell whose vis is finite cannot be a "missing row"
                # (those are NaN), so it can only come from the FLAG table.
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

# ── Source identity survives select-out-of-merge ─────────────────────────────
#
# The primary-HDU stash carries a set's FITS layout through `rebuild`/`select_*`/
# `merge_uvsets`, and a merge inherits the FIRST input's cards. `OBJECT` is not
# layout — it is what the reader takes `source_name` from — so a source selected
# back out of a merge must be written under its OWN identity, not whichever
# source the merge happened to inherit.
@testset "write_uvfits: OBJECT follows the set, not the merge stash" begin
    UV = Gustavo.UVData
    one = build_synth_idi_uvset(; nant = 3, nspw = 1, nchan = 2, nscan = 1, ntime = 2)
    # A second set under a different source name, otherwise identical. The leaf
    # KEY encodes the source, so it has to be recomputed — merging keys by
    # (source, scan), and a stale key would collide with the original's.
    two_branches = DimensionalData.TreeDict()
    for (_, leaf) in UV.branches(one)
        info = UV.metadata(leaf)
        ni = UV.update(
            info; source_name = "OTHER", source_key = :src_OTHER,
            ra = info.ra + 0.01, dec = info.dec - 0.01,
        )
        two_branches[UV.partition_key(ni)] = DimensionalData.rebuild(leaf; metadata = ni)
    end
    two = DimensionalData.rebuild(one; branches = two_branches)
    merged = UV.merge_uvsets(one, two)
    @test Set(UV.sources(merged)) == Set([UV.metadata(first(values(UV.leaves(one)))).source_name, "OTHER"])

    for want in UV.sources(merged)
        sub = UV.select_source(merged, want)
        mktempdir() do dir
            path = joinpath(dir, "sel.uvfits")
            UV.write_uvfits(path, sub)
            back = UV.load_uvfits(path)
            @test unique([UV.metadata(l).source_name for l in values(UV.leaves(back))]) == [want]
            # The phase centre travels with the identity, not with the stash —
            # on EVERY card that carries it. OBSRA/OBSDEC and the RA/DEC axis
            # CRVALs are written from one source position and must agree, or a
            # reader taking the axis pair gets another source's coordinates.
            @test UV.metadata(first(values(UV.leaves(back)))).ra ≈
                UV.metadata(first(values(UV.leaves(sub)))).ra atol = 1.0e-9
            @test UV.metadata(first(values(UV.leaves(back)))).dec ≈
                UV.metadata(first(values(UV.leaves(sub)))).dec atol = 1.0e-9
            cards = UV.primary_cards(back)
            cv(k) = only([c.value for c in cards if rstrip(string(c.key)) == k])
            @test cv("CRVAL6") ≈ cv("OBSRA") atol = 1.0e-9
            @test cv("CRVAL7") ≈ cv("OBSDEC") atol = 1.0e-9
        end
    end
end

# ── Absolute visibility phase convention ─────────────────────────────────────
#
# Gustavo's internal convention is the AIPS/CASA/MSv4 one; FITS-IDI stores its
# complex conjugate. A write→read round trip is blind to that — the two
# conjugations cancel, and so would a uniformly wrong pair — so these tests read
# the on-disk arrays directly. Each boundary's absolute sign is then pinned
# against the file format rather than against Gustavo's other half.
@testset "absolute visibility phase convention" begin
    UV = Gustavo.UVData
    FR = Gustavo.Fring
    ext = Base.get_extension(Gustavo, :GustavoFITSFilesExt)

    τ = 10.0e-9                  # injected delay, in Gustavo's (CASA K-Jones) sense
    nchan, npol, nspw = 32, 2, 2
    ref_freq, chan_bw, spw_sep = 43.0e9, 4.0e6, 2.0e8
    pol_labels = ["PP", "QQ"]
    band_freqs(b) = ref_freq + (b - 1) * spw_sep .+ (0:(nchan - 1)) .* chan_bw
    # V = cis(+2πτ(f − f0)) is what `term_eval(::Delay)` and the matched filter
    # both call a delay of +τ. Constant over time, baseline and polarization, so
    # every on-disk cell of a band carries the same value.
    fringe(b, c) = cis(2π * τ * (band_freqs(b)[c] - band_freqs(b)[1]))

    uvset = build_synth_idi_uvset(;
        nant = 3, nspw, nchan, nscan = 1, ntime = 4, pol_labels,
        ref_freq, chan_bw, spw_sep,
        vis_fn = (b, ti, bl, p, c) -> ComplexF32(fringe(b, c)),
    )

    # The solver's own reading of that fringe: a positive delay.
    for leaf in values(UV.leaves(uvset))
        V = parent(leaf[:vis])[:, :, 1, 1]
        W = ones(size(V))
        freqs = collect(channel_freqs(DimensionalData.metadata(leaf).freq_setup))
        times = collect(lookup(leaf[:vis], Ti))
        det = FR.baseline_fringe_search(FR.fringe_plane(V, W, freqs, times), freqs[1], times[1])
        @test det.valid
        @test isapprox(det.delay, τ; atol = 1.0e-9)
    end

    mktempdir() do dir
        idipath = joinpath(dir, "conv.idifits")
        UV.write_fitsidi(idipath, uvset)

        # FITS-IDI on disk: V = ⟨E_a1 · conj(E_a2)⟩, the conjugate of the above.
        # Read straight out of the FLUX column, bypassing `load_fitsidi`.
        fid = FITSFiles.fits(idipath)
        uv_hdu = ext._idi_find_hdu(fid, "UV_DATA")
        @test uv_hdu !== nothing
        flux = getproperty(uv_hdu.data, :FLUX)
        nperband = 2 * npol * nchan
        for b in 1:nspw, c in 1:nchan, s in 1:npol
            off = (b - 1) * nperband + (c - 1) * 2 * npol + (s - 1) * 2
            v_disk = complex(flux[1, off + 1], flux[1, off + 2])
            @test isapprox(v_disk, conj(fringe(b, c)); atol = 1.0e-5)
        end

        # Reading that file back must undo the conjugation, so the fitted delay
        # keeps the sign it had before the write.
        rt = UV.load_fitsidi(idipath; lazy = false, weight_mode = :validity)
        for leaf in values(UV.leaves(rt))
            V = parent(leaf[:vis])[:, :, 1, 1]
            freqs = collect(channel_freqs(DimensionalData.metadata(leaf).freq_setup))
            times = collect(lookup(leaf[:vis], Ti))
            det = FR.baseline_fringe_search(FR.fringe_plane(V, ones(size(V)), freqs, times), freqs[1], times[1])
            @test det.valid
            @test isapprox(det.delay, τ; atol = 1.0e-9)
        end
    end

    # UVFITS shares Gustavo's convention, so its on-disk imaginary part is the
    # internal one — the opposite of what FITS-IDI holds for the same data.
    # Single-band: `load_uvfits` reads the AIPS layout with one IF.
    uvset1 = build_synth_idi_uvset(;
        nant = 3, nspw = 1, nchan, nscan = 1, ntime = 4, pol_labels,
        ref_freq, chan_bw,
        vis_fn = (b, ti, bl, p, c) -> ComplexF32(fringe(1, c)),
    )
    mktempdir() do dir
        uvpath = joinpath(dir, "conv.uvfits")
        UV.write_uvfits(uvpath, uvset1)

        fid = FITSFiles.fits(uvpath)
        dt = fid[1].data
        raw = dropdims(dt.data; dims = Tuple(findall(==(1), size(dt.data))))
        for c in 1:nchan, p in 1:npol
            v_disk = complex(raw[1, 1, p, c], raw[1, 2, p, c])
            @test isapprox(v_disk, fringe(1, c); atol = 1.0e-5)
        end

        back = UV.load_uvfits(uvpath)
        for leaf in values(UV.leaves(back))
            V = parent(leaf[:vis])[:, :, 1, 1]
            freqs = collect(channel_freqs(DimensionalData.metadata(leaf).freq_setup))
            times = collect(lookup(leaf[:vis], Ti))
            det = FR.baseline_fringe_search(FR.fringe_plane(V, ones(size(V)), freqs, times), freqs[1], times[1])
            @test det.valid
            @test isapprox(det.delay, τ; atol = 1.0e-9)
        end
    end
end

# ── The Ti axis is absolute ──────────────────────────────────────────────────
#
# `Ti` holds seconds since `UVData.JD_UNIX_EPOCH`, so a loaded epoch names a
# real instant. Nothing else in the suite pins this: a round trip through
# either FITS flavour is self-consistent under any origin, and the ANTAB
# matching compares the axis against timestamps derived from the same axis.
# These check it against a date the file itself declares.

@testset "the Ti axis carries absolute epochs" begin
    UV = Gustavo.UVData
    rdate = "2021-03-04"
    uvset = build_synth_idi_uvset(; nant = 3, nspw = 1, nchan = 2, nscan = 1, ntime = 3, rdate)

    leaf_times(set) = sort!(unique(reduce(
        vcat, [collect(UV.obs_time(l)) for l in values(DimensionalData.branches(set))],
    )))

    @testset "in memory" begin
        ts = leaf_times(uvset)
        @test Date(unix2datetime(first(ts))) == Date(rdate)
        # An hours-since-RDATE axis would put these three samples inside the
        # first minute of 1970.
        @test unix2datetime(first(ts)) > DateTime(2000)
    end

    @testset "through FITS-IDI" begin
        path = tempname() * ".idifits"
        try
            UV.write_fitsidi(path, uvset)
            round_ts = leaf_times(UV.load_fitsidi(path))
            @test Date(unix2datetime(first(round_ts))) == Date(rdate)
            @test maximum(abs.(round_ts .- leaf_times(uvset))) < 5.0e-3
        finally
            isfile(path) && rm(path; force = true)
        end
    end

    @testset "through UVFITS" begin
        path = tempname() * ".uvfits"
        try
            UV.write_uvfits(path, uvset)
            round_ts = leaf_times(UV.load_uvfits(path))
            @test Date(unix2datetime(first(round_ts))) == Date(rdate)
            @test maximum(abs.(round_ts .- leaf_times(uvset))) < 5.0e-3
        finally
            isfile(path) && rm(path; force = true)
        end
    end

    @testset "a real FITS-IDI file" begin
        real_path = joinpath(
            @__DIR__, "..", "testdata", "BT164", "BT164A",
            "VLBA_BT164A_bt164aQband_BIN0_SRC0_0_260108T230453.idifits",
        )
        if !isfile(real_path)
            @test_skip "real FITS-IDI file not present"
        else
            real_set = UV.load_fitsidi(real_path; lazy = true)
            file_rdate = DimensionalData.metadata(real_set).array_obs.rdate
            ts = leaf_times(real_set)
            @test !isempty(file_rdate)
            @test Date(unix2datetime(first(ts))) == Date(file_rdate)
        end
    end
end

@testset "storage precision follows the file" begin
    UV = Gustavo.UVData
    uvset = build_synth_idi_uvset(; nant = 3, nspw = 2, nchan = 4, nscan = 1, ntime = 3)

    @testset "load_uvfits takes the file's own precision" begin
        mktempdir() do dir
            path = joinpath(dir, "prec.uvfits")
            UV.write_uvfits(path, uvset)
            leaf = first(values(UV.leaves(UV.load_uvfits(path))))
            @test eltype(leaf[:vis]) === ComplexF32
            @test eltype(leaf[:weights]) === Float32

            wide = first(values(UV.leaves(UV.load_uvfits(path; element_type = Float64))))
            @test eltype(wide[:vis]) === ComplexF64
            @test eltype(wide[:weights]) === Float64
            @test parent(wide[:vis]) == ComplexF64.(parent(leaf[:vis]))
            @test parent(wide[:weights]) == Float64.(parent(leaf[:weights]))

            @test_throws "element_type must be a real float type" UV.load_uvfits(
                path; element_type = ComplexF64,
            )
        end
    end

    @testset "load_fitsidi refuses a double-precision FLUX column" begin
        mktempdir() do dir
            path = joinpath(dir, "prec.idi")
            UV.write_fitsidi(path, uvset)
            # The guard runs before any row is read, so retyping the FLUX
            # column's TFORM is enough to reach it.
            bytes = read(path)
            text = String(copy(bytes))
            n = only(m.captures[1] for m in eachmatch(r"TTYPE(\d+) = 'FLUX *'", text))
            tform = only(eachmatch(Regex("TFORM$(n) *= *'\\d+(E)'"), text))
            bytes[tform.offsets[1]] = UInt8('D')
            wide_path = joinpath(dir, "prec_d.idi")
            write(wide_path, bytes)

            @test_throws "double-precision ('D') FLUX column" UV.load_fitsidi(wide_path)
        end
    end
end
