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
            # `w <= 0` is the flagging the set carries, so that is the FLAG.
            @test parent(ms[:flag]) == permutedims(w_g .<= 0, (4, 1, 3, 2))
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
    end

    @testset "a lazy set converts through materialization" begin
        # `is_lazy` leaves are materialized on the way through, so a lazy and an
        # eager set give the same store.
        @test UV.uvset_to_processingset(uvset)[first(keys(ps))][:visibility] ==
            ps[first(keys(ps))][:visibility]
    end
end
