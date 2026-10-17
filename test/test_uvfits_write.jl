# ── write_uvfits ──────────────────────────────────────────────────────────────
#
# A written file is checked by reading it back with `load_uvfits`; the raw
# random-group arrays are checked where the encoding itself is the point.

using Test
using DimensionalData
using DimensionalData: lookup, Ti
using FITSFiles
using Dates: DateTime
import XRadio

@isdefined(uvfits_fixture) || include("synthetic_uvfits.jl")

# Light-seconds stored at Float32.
const _UVW_RTOL = 1.0e-6

_layer(ms, key) = parent(ms[key])
_attrs(ms, dim) = DimensionalData.metadata(lookup(ms, dim))

function _measure_values(attrs)
    return Dict(k => (v isa XRadio.Measure ? XRadio.value(v) : v) for (k, v) in attrs)
end

# Every cell that is not a gap (flagged with weight zero) reads back verbatim;
# a gap reads back flagged with weight zero.
function _test_same_set(a, b; same_keys = true, time_atol = 0.0)
    same_keys && @test collect(keys(a)) == collect(keys(b))
    @test length(a) == length(b)
    for (m, n) in zip(a, b)
        @test XRadio.polarizations(m) == XRadio.polarizations(n)
        @test XRadio.frequencies(m) ≈ XRadio.frequencies(n)
        @test XRadio.times(m) ≈ XRadio.times(n) atol = time_atol
        @test XRadio.baselines(m) == XRadio.baselines(n)
        @test XRadio.scans(m) == XRadio.scans(n)
        @test eltype(m[:visibility]) === eltype(n[:visibility])
        gap = _layer(m, :flag) .& iszero.(_layer(m, :weight))
        @test isequal(_layer(m, :visibility)[.!gap], _layer(n, :visibility)[.!gap])
        @test _layer(m, :weight) == _layer(n, :weight)
        @test _layer(m, :flag) == _layer(n, :flag)
        written = [!all(view(gap, :, :, b, k)) for b in axes(gap, 3), k in axes(gap, 4)]
        uvw_m, uvw_n = _layer(m, :uvw), _layer(n, :uvw)
        @test all(
            isapprox(uvw_m[:, b, k], uvw_n[:, b, k]; rtol = _UVW_RTOL)
                for b in axes(written, 1), k in axes(written, 2) if written[b, k]
        )
        @test all(isnan, uvw_n[:, .!written])
        @test isequal(_FITS_EXT._antenna_union([m]), _FITS_EXT._antenna_union([n]))
        @test DimensionalData.metadata(m[:visibility])[:units] ==
            DimensionalData.metadata(n[:visibility])[:units]
    end
    return nothing
end

const _FITS_EXT = Base.get_extension(Gustavo, :GustavoFITSFilesExt)

_scan(ps, name) = filter(ms -> only(XRadio.scans(ms)) == name, ps)
_tmp(name = "out.uvfits") = joinpath(mktempdir(), name)
_quiet_write(path, ps) = Test.@test_logs (:warn, r"earth_orientation") write_uvfits(path, ps)

@testset "write_uvfits" begin
    @testset "round trip of a UVFITS file" begin
        # Weights: flagged (-3), zero (+0.0), flagged with zero weight (-0.0);
        # scan 1 has no record for AA–CC at its second time.
        weights = fill(2.0f0, 11, 2, 2)
        weights[1, 1, 2] = -3.0f0
        weights[2, 2, 1] = 0.0f0
        weights[3, 1, 1] = -0.0f0
        path, _ = uvfits_fixture(; weights)
        ps = load_uvfits(path)
        out = write_uvfits(_tmp(), ps)
        back = load_uvfits(out)
        _test_same_set(ps, back)

        for (m, n) in zip(ps, back)
            @test haskey(n, :effective_integration_time)
            @test _layer(m, :effective_integration_time) == _layer(n, :effective_integration_time)
            @test _measure_values(_attrs(m, XRadio.Frequency)) == _measure_values(_attrs(n, XRadio.Frequency))
            @test _measure_values(_attrs(m, Ti)) == _measure_values(_attrs(n, Ti))
            @test DimensionalData.metadata(m)[:earth_orientation] ==
                DimensionalData.metadata(n)[:earth_orientation]
            fs_m, fs_n = XRadio.field_and_source(m), XRadio.field_and_source(n)
            @test collect(fs_m[:field_phase_center_direction]) ≈ collect(fs_n[:field_phase_center_direction])
            @test collect(DimensionalData.branches(m)[:antenna][:antenna_dish_diameter]) ==
                collect(DimensionalData.branches(n)[:antenna][:antenna_dish_diameter])
        end

        scan1, back1 = ps[:src_3C273_spw_0_scan_1], back[:src_3C273_spw_0_scan_1]
        # The flagged zero-weight sample (BB–CC, LL, IF 1, first time) reads back
        # flagged with weight zero and visibility zero.
        cell = (2, 1, 3, 1)
        @test _layer(scan1, :flag)[cell...] && iszero(_layer(scan1, :weight)[cell...])
        @test _layer(back1, :flag)[cell...]
        @test iszero(_layer(back1, :weight)[cell...])
        @test iszero(_layer(back1, :visibility)[cell...])
        # The zero-weight sample stays unflagged.
        @test !_layer(back1, :flag)[1, 1, 2, 1] && iszero(_layer(back1, :weight)[1, 1, 2, 1])
        # The missing record is still missing.
        @test all(_layer(back1, :flag)[:, :, 2, 2])
        @test all(isnan, _layer(back1, :visibility)[:, :, 2, 2])
    end

    @testset "the random-group encoding" begin
        weights = fill(2.0f0, 11, 2, 2)
        weights[1, 1, 2] = -3.0f0
        weights[3, 1, 1] = -0.0f0
        path, args = uvfits_fixture(; weights)
        out = write_uvfits(_tmp(), load_uvfits(path))
        hdus = FITSFiles.fits(out)
        cards = hdus[1].cards
        card(key) = only(c.value for c in cards if rstrip(string(c.key)) == key)
        @test card("CTYPE3") == "STOKES"
        @test card("CRVAL3") == -1.0 && card("CDELT3") == -1.0
        @test strip(card("BUNIT")) == "JY"
        @test card("BITPIX") == -32
        data = Base.get_extension(Gustavo, :GustavoFITSFilesExt)._fast_random_read(
            getfield(hdus[1], :data)
        )
        # Eleven records: the missing slot is not written.
        @test size(data.data, 1) == 11
        # (time, baseline) order.
        bl = Int.(data.BASELINE)
        @test bl[1:3] == [258, 259, 515]
        @test data[Symbol("UU---SIN")] ≈ args.uvw[:, 1] rtol = _UVW_RTOL
        # On-disk STOKES order is RR, LL; record 1 is AA–BB at the first time,
        # whose LL weight in IF 2 is flagged.
        @test data.data[1, 3, 2, 1, 2, 1, 1] == -3.0f0
        w = data.data[3, 3, 2, 1, 1, 1, 1]
        @test iszero(w) && signbit(w)
        @test iszero(data.data[3, 1, 2, 1, 1, 1, 1])
        # DATE is the Julian Date of the day's 0h UT and the fraction of that day.
        @test all(d -> isinteger(d - 0.5), data.DATE[:, 1])
        @test all(0 .<= data.DATE[:, 2] .< 1)
    end

    @testset "element type" begin
        path, _ = uvfits_fixture()
        ps32 = load_uvfits(path)
        @test eltype(load_uvfits(write_uvfits(_tmp(), ps32))[1][:visibility]) === ComplexF32
        ps64 = load_uvfits(path; element_type = Float64)
        back64 = load_uvfits(write_uvfits(_tmp(), ps64))
        @test eltype(back64[1][:visibility]) === ComplexF64
        @test eltype(back64[1][:weight]) === Float64
        _test_same_set(ps64, back64)
    end

    @testset "windows of several channels" begin
        ps = XRadio.Testing.processing_set(; nscan = 2, nspw = 2, nantenna = 3)
        ms = Gustavo.materialize(ps)
        for (i, m) in pairs(collect(ms))
            m[:visibility] .= complex.(Float32.(i .+ reshape(1:length(m[:visibility]), size(m[:visibility]))), 1.0f0)
        end
        # A gap record: every sample of scan 1's A1–A2 at its second time.
        for m in _scan(ms, "1")
            m[:flag][:, :, 1, 2] .= true
            m[:weight][:, :, 1, 2] .= 0
        end
        # A gap sample in a written record.
        ms[1][:flag][1, 1, 2, 1] = true
        ms[1][:weight][1, 1, 2, 1] = 0
        out = _tmp()
        _quiet_write(out, ms)
        back = load_uvfits(out)
        @test collect(keys(back)) ==
            [:src_SRC_spw_0_scan_1, :src_SRC_spw_1_scan_1, :src_SRC_spw_0_scan_2, :src_SRC_spw_1_scan_2]
        _test_same_set(ms, back; same_keys = false, time_atol = 0.01)
        @test all(isnan, _layer(back[1], :visibility)[:, :, 1, 2])
        @test iszero(_layer(back[1], :visibility)[1, 1, 2, 1])
        hdus = FITSFiles.fits(out)
        @test getfield(hdus[1], :data).format.shape[3:4] == (8, 2)
    end

    idi = joinpath(@__DIR__, "..", "testdata", "references", "chain_two_scans.idifits")
    @testset "round trip of a fitsidi2msv4 store" begin
        if isfile(idi)
            dest = joinpath(mktempdir(), "chain.ps.zarr")
            Test.@test_logs (:warn,) (:warn,) match_mode = :any XRadio.fitsidi2msv4(
                idi, dest; mode = "w", release_date = "2000-01-01"
            )
            ps = Gustavo.materialize(open(XRadio.ProcessingSet, dest))
            for m in _scan(ps, "2")
                m[:flag][:, :, 3, 4] .= true
                m[:weight][:, :, 3, 4] .= 0
            end
            out = _tmp()
            _quiet_write(out, ps)
            back = load_uvfits(out)
            _test_same_set(ps, back; same_keys = false, time_atol = 0.01)
            for m in _scan(back, "2")
                @test all(_layer(m, :flag)[:, :, 3, 4])
                @test all(iszero, _layer(m, :weight)[:, :, 3, 4])
            end
        else
            @test_skip isfile(idi)
        end
    end

    @testset "FQ sideband and bandwidth, stated or derived" begin
        freqs = 2.3e11 .+ 2.0e6 .* (0:7)
        for sideband in (1, -1)
            m = XRadio.Testing.measurement_set(; frequencies = sideband > 0 ? freqs : reverse(freqs))
            meta = _attrs(m, XRadio.Frequency)
            meta[:sideband] = sideband
            meta[:total_bandwidth] = XRadio.Measure(
                5.0e7, Dict{Symbol, Any}(:units => "Hz", :type => "quantity")
            )
            # A stated bandwidth wider than the channels span, so reading the
            # stored value and deriving one give different answers.
            @test _FITS_EXT._window_signature(m).total_bandwidth == 5.0e7
            delete!(meta, :sideband)
            delete!(meta, :total_bandwidth)
            derived = _FITS_EXT._window_signature(m)
            @test derived.sideband == sideband
            @test derived.total_bandwidth == 8 * 2.0e6
        end
        @test_throws "no direction to derive one from" _FITS_EXT._channel_direction([2.3e11])
    end

    @testset "refusals" begin
        path, _ = uvfits_fixture()
        ps = load_uvfits(path)
        out = write_uvfits(_tmp(), ps)
        @test_throws "exists; pass `overwrite = true`" write_uvfits(out, ps)
        @test write_uvfits(out, ps; overwrite = true) == out

        other = load_uvfits(first(uvfits_fixture(; object = "OTHER")))
        @test_throws "Select one first" write_uvfits(_tmp(), merge(ps, other))

        shifted = load_uvfits(first(uvfits_fixture(; if_freqs = [0.0, 64.0e6])))
        mixed = merge(_scan(ps, "1"), _scan(shifted, "2"))
        @test_throws "differ from those of scan 1" write_uvfits(_tmp(), mixed)

        rr_rl = XRadio.Testing.processing_set(; polarizations = ["RR", "RL"])
        @test_throws "not a contiguous run" write_uvfits(_tmp(), rr_rl)

        two = Gustavo.materialize(XRadio.Testing.processing_set(; nspw = 2, nantenna = 3))
        _scan(two, "1")[2][:uvw] .+= 1
        @test_throws "of one scan state different uvw" write_uvfits(_tmp(), two)
    end
end
