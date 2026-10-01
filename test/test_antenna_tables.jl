# ── Antenna tables from the UVFITS reader ─────────────────────────────────────
#
# The reader is fed a hand-built table rather than a file, so the table can
# carry an array center off the geocenter that must be added back in.

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

    @testset "AIPS mount codes" begin
        @test ext.mnt_codes_to_type(6, (0.0, 0.0, 0.0)) == UV.MountBWGR()
        @test ext.mnt_codes_to_type(7, (0.0, 0.0, 0.0)) == UV.MountBWGL()
        @test_throws "MNTSTA 8 names no mount" ext.mnt_codes_to_type(8, (0.0, 0.0, 0.0))
    end
end
