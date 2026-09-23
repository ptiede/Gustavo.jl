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
