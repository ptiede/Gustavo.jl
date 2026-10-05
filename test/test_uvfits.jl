# ── load_uvfits ───────────────────────────────────────────────────────────────
#
# Every assertion compares against the values written into the fixture file,
# so a reader that places a value in the wrong cell fails even when the result
# conforms to the standard.

using Test
using DimensionalData
using DimensionalData: dims, lookup, Ti
using FITSFiles
using StableRNGs: StableRNG
import XRadio

@isdefined(uvfits_fixture) || include("synthetic_uvfits.jl")

const _UVF_C = 299792458.0

# The (Measurement Set, time, baseline) cell that record `r` lands in.
function _record_cell(ps, args, r)
    key = args.times[r] < args.times[1] + 50 ? :src_3C273_spw_0_scan_1 : :src_3C273_spw_0_scan_2
    ms = ps[key]
    k = argmin(abs.(collect(lookup(ms[:visibility], Ti)) .- args.times[r]))
    names = ["AA", "BB", "CC"]
    a, b = names[args.baselines[r][1]], names[args.baselines[r][2]]
    bl = findfirst(
        i -> ms[:baseline_antenna1_name][i] == a && ms[:baseline_antenna2_name][i] == b,
        eachindex(parent(ms[:baseline_antenna1_name])),
    )
    return ms, k, bl
end

@testset "load_uvfits" begin
    UV = Gustavo.UVData
    schema = UV.GUSTAVO_VISIBILITY_SCHEMA
    path, args = uvfits_fixture()
    ps = load_uvfits(path)

    @testset "one Measurement Set per NX scan, keyed by source, window and scan" begin
        @test ps isa XRadio.ProcessingSet
        @test collect(keys(ps)) == [:src_3C273_spw_0_scan_1, :src_3C273_spw_0_scan_2]
        @test isempty(XRadio.check(ps; schemas = [schema]))
    end

    @testset "axes" begin
        ms = ps[:src_3C273_spw_0_scan_1]
        @test map(DimensionalData.name, dims(ms[:visibility])) ==
            (:Polarization, :Frequency, :BaselineID, :Ti)
        @test map(DimensionalData.name, dims(ms[:uvw])) == (:UVWLabel, :BaselineID, :Ti)
        @test collect(lookup(ms[:visibility], XRadio.Polarization)) == ["RR", "LL"]
        @test collect(lookup(ms[:visibility], XRadio.Frequency)) == 86.0e9 .+ [0.0, 32.0e6]
        @test collect(lookup(ms[:visibility], Ti)) ≈ args.times[1] .+ [0, 10] atol = 0.01
        @test collect(ms[:baseline_antenna1_name]) == ["AA", "AA", "BB"]
        @test collect(ms[:baseline_antenna2_name]) == ["BB", "CC", "CC"]
        @test eltype(ms[:visibility]) === ComplexF32
        @test eltype(ms[:uvw]) === Float64
        @test eltype(load_uvfits(path; element_type = Float64)[1][:visibility]) === ComplexF64
    end

    @testset "values land at their (polarization, channel, baseline, time)" begin
        # The file's STOKES axis is (LL, RR); the Measurement Set's is (RR, LL).
        disk = (2, 1)
        for r in eachindex(args.times)
            ms, k, bl = _record_cell(ps, args, r)
            for p in 1:2, c in 1:2
                @test parent(ms[:visibility])[p, c, bl, k] == args.vis[r, disk[p], c]
            end
            @test parent(ms[:uvw])[:, bl, k] == Float64.(Float32.(args.uvw[r, :])) .* _UVF_C
        end
    end

    @testset "weights and flags" begin
        ms, k, bl = _record_cell(ps, args, 1)
        @test parent(ms[:weight])[2, 2, bl, k] == 3.0f0        # w < 0
        @test parent(ms[:flag])[2, 2, bl, k]
        @test parent(ms[:weight])[1, 2, bl, k] == 2.0f0        # w > 0
        @test !parent(ms[:flag])[1, 2, bl, k]
        ms, k, bl = _record_cell(ps, args, 2)
        @test parent(ms[:weight])[1, 1, bl, k] == 0.0f0        # w == 0
        @test !parent(ms[:flag])[1, 1, bl, k]
        @test parent(ms[:visibility])[1, 1, bl, k] == args.vis[2, 2, 1]

        # No record for AA–CC at the second time of scan 1.
        ms = ps[:src_3C273_spw_0_scan_1]
        cell = (:, :, 2, 2)
        @test all(isnan, parent(ms[:visibility])[cell...])
        @test all(iszero, parent(ms[:weight])[cell...])
        @test all(parent(ms[:flag])[cell...])
        @test count(parent(ms[:flag])) == 1 + 4
        @test !any(parent(ps[:src_3C273_spw_0_scan_2][:flag]))
    end

    @testset "antennas" begin
        ant = DimensionalData.branches(ps[1])[:antenna]
        @test collect(lookup(ant[:antenna_position], XRadio.AntennaName)) == ["AA", "BB", "CC"]
        @test parent(ant[:antenna_position]) == args.positions
        @test collect(XRadio.mounts(ant)) == [
            XRadio.MountAltAz(), XRadio.MountEquatorial((2.5, 0.0, 0.0)), XRadio.MountNasmythR(),
        ]
        @test parent(ant[:antenna_receptor_angle]) ≈ deg2rad.([args.polaa'; args.polab'])
        @test collect(XRadio.polarization_types(ps[1])[:, 1]) == ["R", "L"]
        @test collect(ant[:antenna_dish_diameter]) == fill(25.0, 3)
        @test all(==([(1, 1), (2, 2)]), eachcol(UV.feed_pairs(ps[1])))
    end

    @testset "integration time" begin
        meta(ms) = DimensionalData.metadata(lookup(ms[:visibility], Ti))
        @test XRadio.value(meta(ps[1])[:integration_time]) == 8.0
        @test all(==(8.0), parent(ps[2][:effective_integration_time]))
        # Without INTTIM, the scan's time spacing and no per-sample time.
        bare = load_uvfits(first(uvfits_fixture(; inttim = nothing)))
        @test XRadio.value(meta(bare[1])[:integration_time]) ≈ 10.0 atol = 0.01
        @test !haskey(bare[1], :effective_integration_time)
    end

    @testset "INTTIM varying within a scan" begin
        inttim = Float64.(1:length(args.times))
        vpath, vargs = uvfits_fixture(; inttim)
        varying = load_uvfits(vpath)
        @test isempty(XRadio.check(varying; schemas = [schema]))
        for r in eachindex(vargs.times)
            ms, k, bl = _record_cell(varying, vargs, r)
            @test parent(ms[:effective_integration_time])[bl, k] == inttim[r]
        end
        eff = varying[:src_3C273_spw_0_scan_1][:effective_integration_time]
        @test map(DimensionalData.name, dims(eff)) == (:BaselineID, :Ti)
        @test parent(eff)[2, 2] == 0.0      # AA–CC, second time: no record
        @test DimensionalData.metadata(eff)[:units] == "s"
        meta(ms) = DimensionalData.metadata(lookup(ms[:visibility], Ti))
        @test XRadio.value(meta(varying[:src_3C273_spw_0_scan_1])[:integration_time]) == 5.0
        @test XRadio.value(meta(varying[:src_3C273_spw_0_scan_2])[:integration_time]) == 11.0

        store = joinpath(mktempdir(), "varying.ps.zarr")
        write(store, varying; schemas = [schema])
        back = read(XRadio.ProcessingSet, store)
        for k in keys(varying)
            @test parent(back[k][:effective_integration_time]) ==
                parent(varying[k][:effective_integration_time])
        end
    end

    @testset "visibility units" begin
        units(ps) = DimensionalData.metadata(ps[1][:visibility])[:units]
        @test units(ps) == "Jy"
        @test units(load_uvfits(first(uvfits_fixture(; bunit = "UNCALIB")))) == "uncalib"
        nounit = first(uvfits_fixture(; bunit = nothing))
        @test_logs (:warn, r"no BUNIT") match_mode = :any load_uvfits(nounit)
        @test units(@test_logs (:warn,) match_mode = :any load_uvfits(nounit)) == "uncalib"
    end

    @testset "what GUSTAVO_VISIBILITY_SCHEMA describes" begin
        ms = ps[1]
        fm = DimensionalData.metadata(lookup(ms[:visibility], XRadio.Frequency))
        @test fm[:sideband] == 1
        @test XRadio.value(fm[:total_bandwidth]) == 32.0e6
        @test XRadio.value(fm[:channel_width]) == 32.0e6
        @test !haskey(ms, :sub_scan_name)
        eo = DimensionalData.metadata(ms)[:earth_orientation]
        @test eo[:gst_iat0] == 196.74
        @test eo[:earth_rot_rate] == 360.9856
        @test eo[:ut1utc] == -0.02
        @test eo[:polarx] == 0.1
        @test eo[:polary] == 0.3
        @test eo[:datutc] == 37.0
        @test eo[:xyzhand] == "RIGHT"
        @test eo[:poltype] == "APPROX"
        fas = DimensionalData.branches(ms)[:field_and_source_base]
        @test collect(fas[:source_name]) == ["3C273"]
        @test parent(fas[:field_phase_center_direction]) ≈ deg2rad.([187.27791667; 2.05238889;;])
    end

    @testset "a Zarr round trip" begin
        store = joinpath(mktempdir(), "fixture.ps.zarr")
        write(store, ps; schemas = [schema])
        back = read(XRadio.ProcessingSet, store)
        @test collect(keys(back)) == collect(keys(ps))
        for k in keys(ps)
            for v in (:visibility, :weight, :flag, :uvw)
                @test isequal(parent(back[k][v]), parent(ps[k][v]))
            end
            @test DimensionalData.metadata(back[k])[:earth_orientation][:poltype] == "APPROX"
            @test DimensionalData.metadata(back[k][:visibility])[:units] == "Jy"
        end
    end

    @testset "a flag read from the file is a flag like the solver's" begin
        CALu = Gustavo.Calibration
        geom = CALu.DataGeometry(ps)
        model = CALu.GainModel(;
            phase = (;
                ph = CALu.GainComponent(
                    CALu.ConstantTerm(); Ti = CALu.PerScan(),
                    Frequency = CALu.PerSpectralWindow(), Feed = CALu.PerFeed(),
                ),
            ),
        )
        layout = CALu.plan_parameters(model, geom.stations, geom)
        sol = CALu.CalibrationSolution(
            model, layout, geom, 0.3 .* randn(StableRNG(3), layout.nθ), (;); name = :hand,
        )
        ms, k, bl = _record_cell(ps, args, 1)
        cell = (2, 2, bl, k)

        # A degenerate gain flags a sample and keeps its weight, as the file's
        # negative weight does.
        bad = deepcopy(sol)
        foreach(c -> parent(c.params) .= NaN, bad.components)
        gainflagged = calibrate(bad, ps; apply_flags = false)[1]
        @test parent(gainflagged[:flag])[1, 2, bl, k]
        @test parent(gainflagged[:weight])[1, 2, bl, k] == parent(ms[:weight])[1, 2, bl, k]
        @test parent(ms[:flag])[cell...] && parent(ms[:weight])[cell...] > 0

        # A phase-only correction leaves the flagged sample flagged at |w|.
        out = calibrate(sol, ps; apply_flags = false)[1]
        @test parent(out[:flag])[cell...]
        @test parent(out[:weight])[cell...] ≈ 3.0f0
    end
end
