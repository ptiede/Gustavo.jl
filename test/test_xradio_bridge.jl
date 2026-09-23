# ── UVSet → ProcessingSet ─────────────────────────────────────────────────────
#
# The bridge is checked two ways: against the standard, by handing the result to
# `XRadio.check`, and against the source, by reading every science value back
# through the transposed axes. A conforming store holding the wrong numbers
# passes the first and fails the second.

using XRadio: XRadio, ProcessingSet, MeasurementSet

@isdefined(_build_fringe_uvset) || include("synthetic_uvset.jl")

@testset "uvset_to_processingset" begin
    UV = Gustavo.UVData
    uvset, _ = _build_fringe_uvset(; nant = 4, nspw = 2, nchan = 8, ntime = 6, nscans = 2)
    ps = UV.uvset_to_processingset(uvset)

    @testset "one MeasurementSet per leaf, under the leaf's key" begin
        @test ps isa ProcessingSet
        @test collect(keys(ps)) == collect(keys(UV.branches(uvset)))
        @test all(ms -> ms isa MeasurementSet, values(ps))
    end

    @testset "the result conforms to the standard" begin
        # `check` throws on the first violation, so reaching the assertion is
        # the result.
        @test (XRadio.check(ps); true)
    end

    @testset "axes carry the leaf's own coordinates" begin
        for (key, leaf) in UV.branches(uvset)
            ms = ps[key]
            info = DimensionalData.metadata(leaf)
            @test collect(lookup(ms[:visibility], Ti)) == collect(UV.obs_time(leaf))
            @test collect(lookup(ms[:visibility], XRadio.Frequency)) ==
                collect(UV.channel_freqs(info.freq_setup))
            @test collect(lookup(ms[:visibility], XRadio.Polarization)) ==
                String.(collect(pol_products(leaf)))
            @test collect(lookup(ms[:visibility], XRadio.BaselineID)) ==
                collect(1:length(info.baselines.pairs))
            @test collect(ms[:baseline_antenna1_name]) == info.baselines.ant1_names
            @test collect(ms[:baseline_antenna2_name]) == info.baselines.ant2_names
        end
    end

    @testset "science values survive the transpose" begin
        # Gustavo holds (Frequency, Ti, Baseline, Pol); MSv4's Julia order is
        # (polarization, frequency, baseline_id, time).
        for (key, leaf) in UV.branches(uvset)
            ms = ps[key]
            vis_g = parent(leaf[:vis])
            w_g = parent(leaf[:weights])
            uvw_g = parent(leaf[:uvw])
            @test parent(ms[:visibility]) == permutedims(vis_g, (4, 1, 3, 2))
            @test parent(ms[:weight]) == permutedims(w_g, (4, 1, 3, 2))
            @test parent(ms[:uvw]) == permutedims(uvw_g, (3, 2, 1))
            # FLAG comes from the `:flags` layer, not from the weight sign.
            @test parent(ms[:flag]) == permutedims(parent(leaf[:flags]), (4, 1, 3, 2))
        end
    end

    @testset "scan, field and source travel with the partition" begin
        for (key, leaf) in UV.branches(uvset)
            ms = ps[key]
            info = DimensionalData.metadata(leaf)
            @test all(==(info.scan_name), collect(ms[:scan_name]))
            @test all(==(info.field_name), collect(ms[:field_name]))
            fas = DimensionalData.branches(ms)[:field_and_source_base]
            @test collect(fas[:source_name]) == [info.source_name]
            @test parent(fas[:field_phase_center_direction]) ≈ [info.ra; info.dec;;]
        end
    end

    @testset "the antenna table becomes an antenna sub-dataset" begin
        leaf = first(values(UV.branches(uvset)))
        info = DimensionalData.metadata(leaf)
        ant = DimensionalData.branches(ps[first(keys(ps))])[:antenna]
        names = String.(collect(info.antennas.name))
        @test collect(lookup(ant[:station_name], XRadio.AntennaName)) == names
        @test collect(ant[:station_name]) == names
        # Positions are (cartesian_pos_label, antenna_name) in Julia order.
        @test size(parent(ant[:antenna_position])) == (3, length(names))
        for (i, xyz) in enumerate(info.antennas.station_xyz)
            @test parent(ant[:antenna_position])[:, i] ≈ Float64.(xyz)
        end
        @test collect(XRadio.mounts(ant)) == collect(info.antennas.mount)
        @test [Tuple(c) for c in eachcol(parent(ant[:antenna_receptor_angle]))] ==
            [Float64.(p) for p in info.antennas.pol_angles]
        @test collect(ant[:antenna_dish_diameter]) == Float64.(UV.extras(info.antennas).DIAMETER)
    end

    @testset "a lazy set converts through materialization" begin
        # `is_lazy` leaves are materialized on the way through, so a lazy and an
        # eager set give the same store.
        @test UV.uvset_to_processingset(uvset)[first(keys(ps))][:visibility] ==
            ps[first(keys(ps))][:visibility]
    end
end

# ── Gustavo's MSv4 schema extension ───────────────────────────────────────────
#
# What MSv4 has no field for travels as attributes and as a coordinate over a
# dimension the standard already defines, so a store carrying it still conforms.
# The specification is what makes `check` examine it and `write` name it.

@testset "GUSTAVO_VISIBILITY_SCHEMA" begin
    UV = Gustavo.UVData
    uvset, _ = _build_fringe_uvset(; nant = 4, nspw = 2, nchan = 8, ntime = 6, nscans = 2)
    ps = UV.uvset_to_processingset(uvset)
    spec = UV.GUSTAVO_VISIBILITY_SCHEMA

    @testset "it extends the standard rather than replacing it" begin
        @test spec.type == XRadio.VISIBILITY_SCHEMA.type
        freq = only(filter(c -> c.name === :frequency, spec.coords))
        named = [a.name for a in freq.attrs]
        # The standard's own frequency attributes survive the merge.
        @test :channel_width in named
        @test :reference_frequency in named
        @test :sideband in named
        @test :total_bandwidth in named
        @test any(c -> c.name === :sub_scan_name, spec.coords)
        @test any(a -> a.name === :earth_orientation, spec.attrs)
    end

    @testset "every addition is optional" begin
        # A store written by anything else still passes, which is what lets
        # Gustavo check data it did not write.
        freq = only(filter(c -> c.name === :frequency, spec.coords))
        @test only(filter(a -> a.name === :sideband, freq.attrs)).optional
        @test only(filter(a -> a.name === :total_bandwidth, freq.attrs)).optional
        @test only(filter(c -> c.name === :sub_scan_name, spec.coords)).optional
        @test only(filter(a -> a.name === :earth_orientation, spec.attrs)).optional
    end

    @testset "the earth-orientation block is all-or-nothing" begin
        # A half-filled block is the case worth reporting: read as zeros it
        # would silently mis-state the rotation.
        block = only(filter(a -> a.name === :earth_orientation, spec.attrs)).nested
        @test !any(f -> f.optional, block.fields)
        @test sort([f.name for f in block.fields]) == sort([
            :gst_iat0, :earth_rot_rate, :ut1utc, :polarx,
            :polary, :datutc, :xyzhand, :poltype,
        ])
    end

    @testset "the bridge writes what the schema describes" begin
        obs = DimensionalData.metadata(uvset).array_obs
        for (key, leaf) in UV.branches(uvset)
            ms = ps[key]
            info = DimensionalData.metadata(leaf)
            fs = info.freq_setup
            fm = DimensionalData.metadata(lookup(ms[:visibility], XRadio.Frequency))

            # One scalar for the window, as `channel_width` already was.
            @test fm[:sideband] == Int(first(UV.sidebands(fs)))
            @test XRadio.value(fm[:total_bandwidth]) ==
                Float64(first(UV.total_bandwidths(fs)))

            # An unnamed sub-scan is "" and carries no coordinate at all.
            if isempty(info.sub_scan_name)
                @test !haskey(ms, :sub_scan_name)
            else
                @test collect(ms[:sub_scan_name]) ==
                    fill(String(info.sub_scan_name), length(UV.obs_time(leaf)))
            end

            eo = DimensionalData.metadata(ms)[:earth_orientation]
            @test eo[:gst_iat0] == Float64(obs.gst_iat0)
            @test eo[:earth_rot_rate] == Float64(obs.earth_rot_rate)
            @test eo[:ut1utc] == Float64(obs.ut1utc)
            @test eo[:polarx] == Float64(obs.polarx)
            @test eo[:polary] == Float64(obs.polary)
            @test eo[:datutc] == Float64(obs.datutc)
            @test eo[:xyzhand] == String(obs.xyzhand)
            @test eo[:poltype] == String(obs.poltype)
        end
    end

    @testset "a store carrying the extension conforms" begin
        @test isempty(XRadio.check(ps; schemas = [spec]))
        # And still conforms when nothing is told about the extension: the
        # additions are an attribute and a coordinate over `time`, both of
        # which MSv4 permits unconditionally.
        @test isempty(XRadio.check(ps))
    end

    @testset "the schema names `sub_scan_name` on disk" begin
        # Without the specification the layer is written `SUB_SCAN_NAME`, which
        # a Python reader sees as a data variable rather than a coordinate.
        # A set that does name its sub-scans, since an unnamed one writes no
        # coordinate at all.
        named = UV.uvset_to_processingset(uvset)
        for ms in values(named)
            t = dims(ms, Ti)
            ms[:sub_scan_name] = DimArray(fill("sub_0", length(t)), (t,))
        end
        path = joinpath(mktempdir(), "named.ps.zarr")
        write(path, named; schemas = [spec], validate = false)
        node = joinpath(path, String(first(keys(named))))
        @test isdir(joinpath(node, "sub_scan_name"))
        @test !isdir(joinpath(node, "SUB_SCAN_NAME"))
        # The variables it labels list it among their coordinates, which is how
        # xarray tells a coordinate from a data variable.
        @test occursin(
            "sub_scan_name", read(joinpath(node, "VISIBILITY", ".zattrs"), String)
        )

        back = read(XRadio.ProcessingSet, path)
        key = first(keys(named))
        @test collect(back[key][:sub_scan_name]) == collect(named[key][:sub_scan_name])
        @test DimensionalData.metadata(
                lookup(back[key][:visibility], XRadio.Frequency)
            )[:sideband] ==
            DimensionalData.metadata(
                lookup(named[key][:visibility], XRadio.Frequency)
            )[:sideband]
    end
end

# ── Partition accessors on a MeasurementSet ───────────────────────────────────
#
# Each accessor answers from the store what it answers from the leaf the store
# was built from.

@testset "partition accessors on a MeasurementSet" begin
    UV = Gustavo.UVData
    uvset, _ = _build_fringe_uvset(; nant = 4, nspw = 2, nchan = 8, ntime = 6, nscans = 2)
    root = DimensionalData.metadata(uvset)
    ps = UV.uvset_to_processingset(uvset)

    function agrees(ms, leaf)
        @test UV.scan_name(ms) == UV.scan_name(leaf)
        @test UV.primary_scan_name(ms) == UV.primary_scan_name(leaf)
        @test UV.source_name(ms) == UV.source_name(leaf)
        @test UV.sub_scan_name(ms) == UV.sub_scan_name(leaf)
        @test UV.scan_intents(ms) == UV.scan_intents(leaf)
        @test collect(UV.obs_time(ms)) == collect(UV.obs_time(leaf))
        @test UV.scan_window(ms) == UV.scan_window(leaf)
        @test UV.pol_products(ms) == UV.pol_products(leaf)
        @test UV.participating_antennas(ms) == UV.participating_antennas(leaf)

        bm, bl = UV.baselines(ms), UV.baselines(leaf)
        @test bm.pairs == bl.pairs
        @test bm.labels == bl.labels
        @test bm.ant1_names == bl.ant1_names
        @test bm.ant2_names == bl.ant2_names
        @test [UV.baseline_number(ms, p) for p in zip(bl.ant1_names, bl.ant2_names)] ==
            collect(eachindex(bl.pairs))

        fm, fl = UV.freq_setup(ms), UV.freq_setup(leaf)
        @test UV.setup_name(fm) == UV.setup_name(fl)
        @test UV.ref_freq(fm) == UV.ref_freq(fl)
        @test UV.channel_freqs(fm) == UV.channel_freqs(fl)
        @test UV.ch_widths(fm) == UV.ch_widths(fl)
        @test UV.total_bandwidths(fm) == UV.total_bandwidths(fl)
        @test UV.sidebands(fm) == UV.sidebands(fl)

        am, al = UV.antennas(ms), UV.antennas(leaf)
        @test am.name == al.name
        @test am.station_xyz == al.station_xyz
        @test am.mount == al.mount
        @test am.nominal_basis == al.nominal_basis
        @test am.pol_angles == al.pol_angles
        @test UV.array_name(am) == UV.array_name(al)
        return @test UV.extras(am).DIAMETER == UV.extras(al).DIAMETER
    end

    @testset "in memory" begin
        for (key, leaf) in UV.branches(uvset)
            agrees(ps[key], leaf)
        end
    end

    @testset "read back from disk" begin
        path = joinpath(mktempdir(), "accessors.ps.zarr")
        write(path, ps; schemas = [UV.GUSTAVO_VISIBILITY_SCHEMA])
        back = read(XRadio.ProcessingSet, path)
        for (key, leaf) in UV.branches(uvset)
            agrees(back[key], leaf)
        end
    end

    key, leaf = first(UV.branches(uvset))
    info = DimensionalData.metadata(leaf)
    rebuilt(; kw...) = UV._leaf_to_measurementset(
        DimensionalData.rebuild(leaf; metadata = UV.update(info; kw...)), root,
    )

    @testset "a named sub-scan and stated intents" begin
        intents = ["OBSERVE_TARGET#ON_SOURCE", "CALIBRATE_DELAY#ON_SOURCE"]
        ms = rebuilt(; sub_scan_name = "sub_1", scan_intents = intents)
        @test UV.sub_scan_name(ms) == "sub_1"
        @test UV.scan_intents(ms) == intents
    end

    @testset "sideband and bandwidth derived where the store states none" begin
        # A stated bandwidth wider than the channels span, so reading the stored
        # value and deriving one give different answers.
        fs = info.freq_setup
        setup(freqs, sideband) = UV.FrequencySetup(;
            name = fs.name, ref_freq = fs.ref_freq, channel_freqs = freqs,
            ch_widths = fs.ch_widths, total_bandwidths = fill(5.0e7, length(fs)),
            sidebands = fill(sideband, length(fs)),
        )
        for sideband in (1.0, -1.0)
            freqs = sideband > 0 ? fs.channel_freqs : reverse(fs.channel_freqs)
            ms = rebuilt(; freq_setup = setup(freqs, sideband))
            @test all(==(5.0e7), UV.total_bandwidths(UV.freq_setup(ms)))
            meta = DimensionalData.metadata(lookup(dims(ms, XRadio.Frequency)))
            delete!(meta, :sideband)
            delete!(meta, :total_bandwidth)
            derived = UV.freq_setup(ms)
            @test all(==(sideband), UV.sidebands(derived))
            @test all(==(length(fs) * first(fs.ch_widths)), UV.total_bandwidths(derived))
        end
        @test_throws "no direction to derive one from" UV._channel_direction([2.3e11])
    end

    @testset "a Measurement Set holding two scans has no scan name" begin
        ms = UV.uvset_to_processingset(uvset)[key]
        t = dims(ms, Ti)
        ms[:scan_name] = DimArray([i <= length(t) ÷ 2 ? "1" : "2" for i in eachindex(t)], (t,))
        @test_throws "holds 2 scan names (1, 2)" UV.scan_name(ms)
        @test_throws "holds 2 scan names" UV.primary_scan_name(ms)
    end
end

# ── Set-level functions on a ProcessingSet ────────────────────────────────────

@testset "set-level functions on a ProcessingSet" begin
    UV = Gustavo.UVData
    uvset, _ = _build_fringe_uvset(; nant = 4, nspw = 2, nchan = 8, ntime = 6, nscans = 2)
    ps = UV.uvset_to_processingset(uvset)

    @testset "leaves are the Measurement Sets, by name" begin
        @test collect(UV.leaves(ps)) == collect(pairs(ps))
    end

    @testset "frequency and polarization unions agree with the UVSet's" begin
        mine, theirs = UV.union_frequency_axis(ps), UV.union_frequency_axis(uvset)
        @test length(mine) == length(theirs) == 2
        @test UV.channel_freqs.(mine) == UV.channel_freqs.(theirs)
        @test UV.setup_name.(mine) == UV.setup_name.(theirs)
        @test UV.union_pol_products(ps) == UV.union_pol_products(uvset)
        @test_throws "2 distinct frequency setups" UV.freq_setup(ps)

        one_spw, _ = _build_fringe_uvset(; nant = 4, nspw = 1, nchan = 8, ntime = 6, nscans = 2)
        one_ps = UV.uvset_to_processingset(one_spw)
        @test UV.channel_freqs(UV.freq_setup(one_ps)) ==
            UV.channel_freqs(UV.freq_setup(one_spw))
        @test UV.nchannels(one_ps) == UV.nchannels(one_spw) == 8
    end

    @testset "the antenna union agrees with the UVSet's" begin
        mine, theirs = UV.union_antennas(ps), UV.union_antennas(uvset)
        @test mine.name == theirs.name
        @test mine.station_xyz == theirs.station_xyz
        @test mine.mount == theirs.mount
        @test UV.extras(mine).DIAMETER == UV.extras(theirs).DIAMETER
    end

    @testset "members that saw different sub-arrays" begin
        # The second scan drops the second antenna, so its antenna dataset
        # lists three stations and the union must restore the full order.
        names = String.(UV.union_antennas(uvset).name)
        keep = names[[1, 3, 4]]
        sub = UV.apply(uvset) do leaf, info, root
            info.scan_name == "2" || return leaf
            full = String.(info.antennas.name)
            rows = findall(in(keep), full)
            ext = map(c -> c[rows], UV.extras(info.antennas))
            tbl = UV.AntennaTable(
                StructArray(getfield(info.antennas, :antennas)[rows]),
                UV.array_name(info.antennas), ext,
            )
            local_of = Dict(n => i for (i, n) in pairs(keep))
            b = info.baselines
            idx = [i for (i, (a, c)) in pairs(b.pairs) if full[a] in keep && full[c] in keep]
            local_pairs = [(local_of[full[a]], local_of[full[c]]) for (a, c) in b.pairs[idx]]
            newb = UV.BaselineIndex(local_pairs, local_pairs; antenna_names = keep)
            return UV._build_leaf(
                leaf[:vis][Baseline = idx], leaf[:weights][Baseline = idx],
                leaf[:uvw][Baseline = idx], leaf[:flags][Baseline = idx];
                partition_info = UV.update(
                    info; antennas = tbl, baselines = newb, record_order = Tuple{Int, Int}[],
                ),
            )
        end
        sub_ps = UV.uvset_to_processingset(sub)
        @test length(unique(XRadio.antennas.(values(sub_ps)))) == 2

        tab = UV.union_antennas(sub_ps)
        @test tab.name == names
        @test UV.extras(tab).DIAMETER == UV.extras(UV.union_antennas(uvset)).DIAMETER
    end

    @testset "unstated receptor angles still union" begin
        bare = UV.uvset_to_processingset(uvset)
        for ms in values(bare)
            delete!(DimensionalData.branches(ms)[:antenna], :antenna_receptor_angle)
        end
        tab = UV.union_antennas(bare)
        @test tab.name == UV.union_antennas(uvset).name
        @test all(a -> all(isnan, a), tab.pol_angles)
    end

    @testset "one antenna stated two ways is refused" begin
        clash = UV.uvset_to_processingset(uvset)
        moved = DimensionalData.branches(last(collect(values(clash))))[:antenna]
        parent(moved[:antenna_position])[1, 1] += 1.0
        @test_throws "inconsistent metadata across partitions" UV.union_antennas(clash)
    end
end
