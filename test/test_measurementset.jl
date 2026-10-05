# ── Gustavo's MSv4 schema extension, and the accessors on a Measurement Set ───
#
# What MSv4 has no field for travels as attributes and as a coordinate over a
# dimension the standard already defines, so a store carrying it still conforms.
# The specification is what makes `check` examine it and `write` name it.

using XRadio: XRadio, ProcessingSet, MeasurementSet, Testing

@isdefined(uvfits_fixture) || include("synthetic_uvfits.jl")

@testset "GUSTAVO_VISIBILITY_SCHEMA" begin
    UV = Gustavo.UVData
    ps = load_uvfits(first(uvfits_fixture()))
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
        @test isempty(XRadio.check(Testing.processing_set(); schemas = [spec]))
    end

    @testset "the earth-orientation block is all-or-nothing" begin
        # A half-filled block is the case worth reporting: read as zeros it
        # would silently mis-state the rotation.
        block = only(filter(a -> a.name === :earth_orientation, spec.attrs)).nested
        @test !any(f -> f.optional, block.fields)
        @test sort([f.name for f in block.fields]) == sort(
            [
                :gst_iat0, :earth_rot_rate, :ut1utc, :polarx,
                :polary, :datutc, :xyzhand, :poltype,
            ]
        )
        half = load_uvfits(first(uvfits_fixture()))
        for ms in values(half)
            delete!(DimensionalData.metadata(ms)[:earth_orientation], :poltype)
        end
        @test !isempty(XRadio.check(half; schemas = [spec]))
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
        named = load_uvfits(first(uvfits_fixture()))
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
