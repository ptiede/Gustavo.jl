# A minimal AIPS random-groups UVFITS file built from known values.
#
# It is a fixture generator, not a UVFITS writer: one FQ row, one AN table, no
# scaled parameters. The caller states every record explicitly, so a test
# compares what `load_uvfits` returns against the values themselves.

using FITSFiles: FITSFiles, HDU, Bintable, Card
using Dates: Date, DateTime, datetime2julian

const _JD_UNIX_EPOCH = 2440587.5

"""
    write_synthetic_uvfits(path; times, baselines, uvw, vis, weights, kw...) -> path

Write a UVFITS file holding one record per entry of `times` (Unix seconds),
`baselines` (`(a1, a2)` antenna numbers), `uvw` (`nrec × 3`, light-seconds),
`vis` and `weights` (`nrec × npol × nif`, the STOKES axis in the order
`stokes` gives it).

# Keywords
- `inttim`: one integration time per record, or `nothing` to omit the parameter.
- `stokes = [-1, -2]`: the AIPS Stokes codes of the polarization axis.
- `ref_freq`, `if_freqs` (offsets from `ref_freq`), `ch_width`,
  `total_bandwidth`, `sideband`: the FQ row, one value per IF.
- `antennas`, `positions` (`3 × nant`, geocentric meters), `mntsta`, `staxof`,
  `poltya`, `poltyb`, `polaa`, `polab` (degrees), `diameter` (`nothing` omits
  the column): the AN table.
- `scans`: one `(start, stop)` window in Unix seconds per NX row.
- `bunit`: the `BUNIT` card, or `nothing` to omit it.
- `object`, `ra`, `dec` (degrees), `telescope`, `instrume`, `rdate`, and the
  AN table's Earth-orientation cards `gstia0`, `degpdy`, `ut1utc`, `polarx`,
  `polary`, `datutc`, `xyzhand`, `poltype`.
"""
function write_synthetic_uvfits(
        path; times, baselines, uvw, vis, weights,
        inttim = nothing,
        stokes = [-1, -2],
        ref_freq = 86.0e9, if_freqs = [0.0, 32.0e6], ch_width = fill(32.0e6, length(if_freqs)),
        total_bandwidth = ch_width, sideband = fill(1, length(if_freqs)),
        antennas = ["AA", "BB", "CC"],
        positions = [1.0e6 * c * i for c in 1:3, i in eachindex(antennas)],
        mntsta = fill(0, length(antennas)), staxof = zeros(length(antennas)),
        poltya = fill("R", length(antennas)), poltyb = fill("L", length(antennas)),
        polaa = zeros(length(antennas)), polab = zeros(length(antennas)),
        diameter = fill(25.0, length(antennas)),
        scans = [(minimum(times) - 1, maximum(times) + 1)],
        bunit = "JY",
        object = "3C273", ra = 187.27791667, dec = 2.05238889,
        telescope = "SYNTH", instrume = "SYNTHCORR", rdate = "2024-04-08",
        gstia0 = 196.74, degpdy = 360.9856, ut1utc = -0.02, polarx = 0.1, polary = 0.3,
        datutc = 37.0, xyzhand = "RIGHT", poltype = "APPROX",
    )
    T = Float32
    nrec, npol, nif = size(vis)
    raw = zeros(T, nrec, 3, npol, 1, nif, 1, 1)
    raw[:, 1, :, 1, :, 1, 1] .= real.(vis)
    raw[:, 2, :, 1, :, 1, 1] .= imag.(vis)
    raw[:, 3, :, 1, :, 1, 1] .= weights

    # The integer Julian Day and the fraction of it, as AIPS splits them.
    day = floor.(_JD_UNIX_EPOCH .+ times ./ 86400)
    frac = times ./ 86400 .- (day .- _JD_UNIX_EPOCH)
    params = Pair{Symbol, Any}[
        Symbol("UU---SIN") => T.(uvw[:, 1]),
        Symbol("VV---SIN") => T.(uvw[:, 2]),
        Symbol("WW---SIN") => T.(uvw[:, 3]),
        :BASELINE => T[256 * a + b for (a, b) in baselines],
    ]
    isnothing(inttim) || push!(params, :INTTIM => T.(inttim))
    # An `N × 2` column is written as two `DATE` parameters.
    push!(params, :DATE => T[day frac], :data => raw)

    cards = Card[
        Card("NAXIS", 7), Card("EXTEND", true),
        Card("OBJECT", object), Card("TELESCOP", telescope), Card("INSTRUME", instrume),
        Card("DATE-OBS", rdate), Card("EQUINOX", 2000.0),
        Card("CTYPE2", "COMPLEX"), Card("CRVAL2", 1.0), Card("CDELT2", 1.0), Card("CRPIX2", 1.0),
        Card("CTYPE3", "STOKES"), Card("CRVAL3", Float64(stokes[1])),
        Card("CDELT3", Float64(length(stokes) > 1 ? stokes[2] - stokes[1] : -1)),
        Card("CRPIX3", 1.0),
        Card("CTYPE4", "FREQ"), Card("CRVAL4", ref_freq), Card("CDELT4", first(ch_width)),
        Card("CRPIX4", 1.0),
        Card("CTYPE5", "IF"), Card("CRVAL5", 1.0), Card("CDELT5", 1.0), Card("CRPIX5", 1.0),
        Card("CTYPE6", "RA"), Card("CRVAL6", ra), Card("CDELT6", 1.0), Card("CRPIX6", 1.0),
        Card("CTYPE7", "DEC"), Card("CRVAL7", dec), Card("CDELT7", 1.0), Card("CRPIX7", 1.0),
        Card("OBSRA", ra), Card("OBSDEC", dec),
    ]
    isnothing(bunit) || push!(cards, Card("BUNIT", bunit))
    primary = HDU(FITSFiles.Random, (; params...), cards)

    nant = length(antennas)
    an_data = (;
        ANNAME = rpad.(antennas, 8),
        STABXYZ = [Float64.(positions[:, i]) for i in 1:nant],
        ORBPARM = [Float64[] for _ in 1:nant],
        NOSTA = Int32.(1:nant),
        MNTSTA = Int32.(mntsta),
        STAXOF = Float32.(staxof),
        POLTYA = poltya, POLAA = Float32.(polaa),
        POLTYB = poltyb, POLAB = Float32.(polab),
        (isnothing(diameter) ? (;) : (; DIAMETER = Float32.(diameter)))...,
    )
    an = HDU(
        Bintable, an_data, Card[
            Card("EXTNAME", "AIPS AN"), Card("EXTVER", Int32(1)),
            Card("ARRAYX", 0.0), Card("ARRAYY", 0.0), Card("ARRAYZ", 0.0),
            Card("ARRNAM", telescope), Card("FREQ", ref_freq), Card("RDATE", rdate),
            Card("GSTIA0", gstia0), Card("DEGPDY", degpdy), Card("UT1UTC", ut1utc),
            Card("POLARX", polarx), Card("POLARY", polary), Card("DATUTC", datutc),
            Card("TIMSYS", "UTC"), Card("FRAME", "ITRF"), Card("XYZHAND", xyzhand),
            Card("POLTYPE", poltype), Card("NUMORB", Int32(0)), Card("NO_IF", Int32(nif)),
            Card("NOPCAL", Int32(0)), Card("FREQID", Int32(1)),
        ]
    )

    fq = HDU(
        Bintable, (;
            FRQSEL = Int32[1],
            var"IF FREQ" = [Float64.(if_freqs)],
            var"CH WIDTH" = [Float32.(ch_width)],
            var"TOTAL BANDWIDTH" = [Float32.(total_bandwidth)],
            SIDEBAND = [Int32.(sideband)],
        ), Card[Card("EXTNAME", "AIPS FQ"), Card("NO_IF", Int32(nif))]
    )

    rdate_unix = (datetime2julian(DateTime(Date(rdate))) - _JD_UNIX_EPOCH) * 86400
    nx = HDU(
        Bintable, (;
            TIME = [((lo + hi) / 2 - rdate_unix) / 86400 for (lo, hi) in scans],
            var"TIME INTERVAL" = Float32[(hi - lo) / 86400 for (lo, hi) in scans],
            var"SOURCE ID" = fill(Int32(1), length(scans)),
            SUBARRAY = fill(Int32(1), length(scans)),
            var"FREQ ID" = fill(Int32(1), length(scans)),
        ), Card[Card("EXTNAME", "AIPS NX")]
    )

    write(path, HDU[primary, an, fq, nx])
    return path
end

"""
    uvfits_fixture(; kw...) -> (path, args)

A UVFITS file of three antennas, two NX scans and two IFs, and the keywords
`write_synthetic_uvfits` wrote it with; `kw` replaces any of them. The file's
STOKES axis runs LL, RR, so the reader reorders it. Scan 1 has no record for AA–CC at its second time.
"""
function uvfits_fixture(; kw...)
    t0 = 1.7125344e9
    recs = [
        (t0, (1, 2)), (t0, (1, 3)), (t0, (2, 3)),
        (t0 + 10, (1, 2)), (t0 + 10, (2, 3)),
        (t0 + 100, (1, 2)), (t0 + 100, (1, 3)), (t0 + 100, (2, 3)),
        (t0 + 110, (1, 2)), (t0 + 110, (1, 3)), (t0 + 110, (2, 3)),
    ]
    nrec = length(recs)
    times = first.(recs)
    baselines = last.(recs)
    uvw = [r * 1.0e-5 + k * 1.0e-7 for r in 1:nrec, k in 1:3]
    vis = ComplexF32[complex(r + p / 4, c / 8) for r in 1:nrec, p in 1:2, c in 1:2]
    weights = fill(2.0f0, nrec, 2, 2)
    weights[1, 1, 2] = -3.0f0     # flagged: AA–BB, LL, IF 2, first time
    weights[2, 2, 1] = 0.0f0      # zero weight: AA–CC, RR, IF 1, first time
    path = joinpath(mktempdir(), "fixture.uvfits")
    args = (;
        times, baselines, uvw, vis, weights,
        inttim = fill(8.0, nrec),
        stokes = [-2, -1],
        positions = [1.0e6 * c + 1.0e3 * i for c in 1:3, i in 1:3],
        mntsta = [0, 1, 4], staxof = [0.0, 2.5, 0.0],
        polaa = [0.0, 10.0, 20.0], polab = [90.0, 100.0, 110.0],
        scans = [(t0 - 5, t0 + 15), (t0 + 95, t0 + 115)],
    )
    args = merge(args, (; kw...))
    write_synthetic_uvfits(path; args...)
    return path, args
end
