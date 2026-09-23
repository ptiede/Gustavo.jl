# ── Antenna tables from the FITS readers ──────────────────────────────────────
#
# Both readers are fed hand-built tables rather than files: the writers put the
# array center at the geocenter, so only a hand-built table carries one that
# must be added back in.

using FITSFiles: Card

@testset "FITS antenna tables" begin
    UV = Gustavo.UVData
    ext = Base.get_extension(Gustavo, :GustavoFITSFilesExt)
    center = [1.0e6, -2.0e6, 3.0e6]
    stabxyz = [10.0 20.0 30.0; -5.0 0.0 5.0]
    cards = [
        Card("ARRAYX", center[1]), Card("ARRAYY", center[2]), Card("ARRAYZ", center[3]),
        Card("ARRNAM", "TEST"),
    ]
    angles(tab) = [collect(Float64.(p)) for p in tab.pol_angles]

    @testset "UVFITS AN table" begin
        an = (;
            ANNAME = ["AA", "BB"], STABXYZ = stabxyz, NOSTA = Int32[1, 2],
            MNTSTA = Int32[0, 4], STAXOF = Float32[0, 1.5],
            POLTYA = ["R", "R"], POLAA = Float32[0, 45],
            POLTYB = ["L", "L"], POLAB = Float32[90, 135],
        )
        tab = ext._build_antenna_table((; cards, data = an))
        @test collect(tab.station_xyz) == [center .+ stabxyz[i, :] for i in 1:2]
        @test collect(tab.mount) == [UV.MountAltAz(), UV.MountNasmythR((1.5, 0.0, 0.0))]
        @test angles(tab) ≈ [[0, π / 2], [π / 4, 3π / 4]] rtol = 1.0e-6
    end

    @testset "FITS-IDI ARRAY_GEOMETRY and ANTENNA tables" begin
        ag = (;
            ANNAME = ["AA", "BB"], STABXYZ = stabxyz, NOSTA = Int32[1, 2],
            MNTSTA = Int32[3, 5], STAXOF = Float32[0 0 0; 0.5 0 -1],
            DIAMETER = Float32[25, 12],
        )
        an = (;
            POLTYA = ["X", "X"], POLAA = Float32[0, 45],
            POLTYB = ["Y", "Y"], POLAB = Float32[90, 135],
        )
        tab = ext._build_idi_antenna_table((; cards, data = ag), (; cards = Card[], data = an))
        @test collect(tab.station_xyz) == [center .+ stabxyz[i, :] for i in 1:2]
        @test collect(tab.mount) == [UV.MountXY(), UV.MountNasmythL((0.5, 0.0, -1.0))]
        @test angles(tab) ≈ [[0, π / 2], [π / 4, 3π / 4]] rtol = 1.0e-6
        @test UV.extras(tab).DIAMETER == Float32[25, 12]

        # A diameter the file does not state is not invented.
        nodiam = Base.structdiff(ag, NamedTuple{(:DIAMETER,)})
        tab = ext._build_idi_antenna_table((; cards, data = nodiam), (; cards = Card[], data = an))
        @test !haskey(UV.extras(tab), :DIAMETER)

        scalar = merge(ag, (; STAXOF = Float32[0, 1.5]))
        @test_throws "three values per antenna" ext._build_idi_antenna_table(
            (; cards, data = scalar), (; cards = Card[], data = an)
        )
    end

    @testset "a UVFITS axis offset lies along the station's x" begin
        @test ext._uvfits_staxof(UV.MountAltAz((2.5, 0.0, 0.0))) == 2.5f0
        @test_throws "off the station's x axis" ext._uvfits_staxof(UV.MountAltAz((0.0, 1.0, 0.0)))
    end

    @testset "every AIPS mount code round-trips" begin
        for code in 0:6
            @test ext.mount_to_mntsta(ext.mnt_codes_to_type(code, (0.0, 0.0, 0.0))) == code
        end
        @test_throws "MNTSTA 7 names no mount" ext.mnt_codes_to_type(7, (0.0, 0.0, 0.0))
        @test_throws "has no AIPS MNTSTA code" ext.mount_to_mntsta(UV.Mount(0.5, 0.0))
    end
end
