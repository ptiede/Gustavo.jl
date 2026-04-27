"""
    Antenna

Single-antenna record drawn from the AIPS AN binary table (Memo 117 §4.1,
Table 10). Only the **mandatory** columns of Table 10 are stored as fields;
optional columns (`DIAMETER`, `BEAMFWHM`) live in `AntennaTable.extras`.

`station_xyz` is the (x, y, z) coordinate of the antenna *relative to the array
center* in the frame given by `ArrayConfig.frame` (typically ITRF), rotated to
the longitude of the array center. Mount type codes: 0=alt-az, 1=equatorial,
2=orbiting, 3=X-Y, 4=right-Naismith, 5=left-Naismith, 6=aperture/phased array.

In-memory eltypes mirror Memo 117 TFORM types (E→Float32, D→Float64).
"""
struct Antenna{TName, TXYZ, TMnt, TAxOff, TFa, TFb, TPola, TPolb, TPolcal}
    name::TName             # ANNAME  (8A)
    station_xyz::TXYZ       # STABXYZ (3D, Float64) — meters
    mount_type::TMnt        # MNTSTA  (1J, Int32)
    axis_offset::TAxOff     # STAXOF  (1E, Float32) — meters
    feed_a::TFa             # POLTYA  (1A) — 'R','L','X','Y'
    feed_b::TFb             # POLTYB  (1A)
    pola_angle::TPola       # POLAA   (1E, Float32) — degrees
    polb_angle::TPolb       # POLAB   (1E, Float32) — degrees
    polcala::TPolcal        # POLCALA (E(NOPCAL,NO_IF), Float32) — empty if NOPCAL==0
    polcalb::TPolcal        # POLCALB (E(NOPCAL,NO_IF), Float32)
end

"""
    AntennaTable

Array-of-structs antenna table from the AIPS AN HDU.

- `antennas`   : `StructArray{Antenna}` of the mandatory Table 10 columns
- `array_xyz`  : geocentric array center in meters (ARRAYX/Y/Z header cards)
- `array_name` : ARRNAM
- `extras`     : `NamedTuple` of optional Table 10 columns actually present in
  the file (e.g. `DIAMETER`, `BEAMFWHM`). Keys match the FITS column names.
"""
struct AntennaTable{TAnt, TXyz, TName, TExtras <: NamedTuple}
    antennas::TAnt
    array_xyz::TXyz
    array_name::TName
    extras::TExtras
end

# Backward-compatible no-extras constructor.
AntennaTable(antennas, array_xyz, array_name) =
    AntennaTable(antennas, array_xyz, array_name, NamedTuple())

"""
    ArrayConfig

Timing and geodetic metadata from the AIPS AN table header (Memo 117 §4.1,
Table 8 mandatory keywords). These are tied to the observation epoch rather
than the physical antenna array.
"""
struct ArrayConfig{TRdate, TGst, TDeg, TUt, TPol, TDut, TSys, TFrame, THand, TStr, TInt}
    rdate::TRdate  # RDATE   (D)
    gst_iat0::TGst    # GSTIA0  (E, Float32) — degrees
    earth_rot_rate::TDeg    # DEGPDY  (E, Float32) — degrees/day
    ut1utc::TUt     # UT1UTC  (E, Float32) — seconds
    polarx::TPol    # POLARX  (E, Float32) — arc seconds
    polary::TPol    # POLARY  (E, Float32) — arc seconds
    datutc::TDut    # DATUTC  (E, Float32) — seconds
    time_sys::TSys    # TIMSYS  (A) — 'IAT' or 'UTC'
    frame::TFrame  # FRAME   (A) — e.g. 'ITRF'
    xyzhand::THand   # XYZHAND (A) — 'RIGHT' or 'LEFT'
    poltype::TStr    # POLTYPE (A)
    extver::TInt    # EXTVER  (I) — subarray number
    numorb::TInt    # NUMORB  (I)
    no_if::TInt    # NO_IF   (I) — number of spectral windows
    nopcal::TInt    # NOPCAL  (I) — 0 or 2
    freqid::TInt    # FREQID  (I) — frequency setup number
end

Base.length(t::AntennaTable) = length(t.antennas)
Base.getindex(t::AntennaTable, i) = t.antennas[i]
Base.iterate(t::AntennaTable, args...) = iterate(t.antennas, args...)

function Base.getproperty(t::AntennaTable, s::Symbol)
    return hasfield(typeof(t), s) ? getfield(t, s) : getproperty(getfield(t, :antennas), s)
end

"""
    FrequencySetup

Frequency-axis description for a UVFITS dataset, drawn from the FREQ regular
axis (Memo 117 §3.1.1) and the AIPS FQ binary table (Table 22). Pulled out of
`ObsMetadata` because frequency setup is a self-contained concept used heavily
on its own (e.g. by the bandpass solver) and one FITS file may, in principle,
carry several setups identified by `FRQSEL`.

In-memory eltypes mirror Memo 117 TFORM types (E→Float32, D→Float64), with
two principled exceptions: `ref_freq` and `channel_freqs` stay Float64 for
arithmetic stability under subtractions of similar-magnitude frequencies.

- `freqid`            : FQ `FRQSEL` value identifying this setup (Int32)
- `ref_freq`          : FREQ-axis CRVAL — reference frequency (Hz, Float64)
- `channel_freqs`     : absolute IF center frequencies (Hz, Float64) =
                        `ref_freq` + `if_freqs`
- `if_freqs`          : FQ `IF FREQ` column (D, Float64) — Hz offsets
- `ch_widths`         : FQ `CH WIDTH` column (E, Float32) per IF — Hz
- `total_bandwidths`  : FQ `TOTAL BANDWIDTH` column (E, Float32) per IF — Hz
- `sidebands`         : FQ `SIDEBAND` column (J, Int32): +1=upper, -1=lower
- `extras`            : NamedTuple of optional FQ columns (e.g. `BANDCODE`)
"""
struct FrequencySetup{TFid, TFreq, TCfreqs, TIff, TCw, TBw, TSb, TExtras <: NamedTuple}
    freqid::TFid
    ref_freq::TFreq
    channel_freqs::TCfreqs
    if_freqs::TIff
    ch_widths::TCw
    total_bandwidths::TBw
    sidebands::TSb
    extras::TExtras
end

FrequencySetup(freqid, ref_freq, channel_freqs, if_freqs, ch_widths, total_bandwidths, sidebands) =
    FrequencySetup(freqid, ref_freq, channel_freqs, if_freqs, ch_widths, total_bandwidths, sidebands, NamedTuple())

function Base.show(io::IO, fs::FrequencySetup)
    flo = round(minimum(fs.channel_freqs) / 1.0e9; digits = 3)
    fhi = round(maximum(fs.channel_freqs) / 1.0e9; digits = 3)
    return print(io, "FrequencySetup(FRQSEL=$(fs.freqid), $(length(fs.channel_freqs)) IFs, $(flo)–$(fhi) GHz)")
end

"""
    ObsMetadata

Observation-level metadata from the primary HDU FITS cards (Memo 117 Table 5).
The frequency description is held in a separate `FrequencySetup` struct,
because it's a self-contained concept (one row of the AIPS FQ table) used
heavily on its own.

For convenience, the `FrequencySetup` fields (`channel_freqs`, `ref_freq`,
`if_freqs`, `ch_widths`, `total_bandwidths`, `sidebands`, `freqid`) are
forwarded through `getproperty`, so call sites can keep writing
`metadata.channel_freqs` rather than `metadata.freq_setup.channel_freqs`.

- `pol_codes` / `pol_labels` : AIPS Stokes codes/labels (mandatory)
- `freq_setup`               : see `FrequencySetup`
- `extras`                   : NamedTuple of optional primary-HDU cards
"""
struct ObsMetadata{
        TObj, TTel, TInst, TDate, TEq, TBunit, TRa, TDec,
        TFs <: FrequencySetup, TPcodes, TPlabs, TExtras <: NamedTuple,
    }
    object::TObj             # OBJECT
    telescope::TTel          # TELESCOP
    instrume::TInst          # INSTRUME
    date_obs::TDate          # DATE-OBS
    equinox::TEq             # EQUINOX (E, Float32) — year, e.g. 2000.0
    bunit::TBunit            # BUNIT — 'UNCALIB','JY',…
    ra::TRa                  # phase center RA (degrees)
    dec::TDec                # phase center Dec (degrees)
    freq_setup::TFs          # FrequencySetup
    pol_codes::TPcodes       # AIPS Stokes codes
    pol_labels::TPlabs       # 'RR','LL',… string labels
    extras::TExtras          # optional primary-HDU cards
end

# Forward FrequencySetup fields so existing call sites
# (`metadata.channel_freqs`, `metadata.ref_freq`, …) keep working without
# every consumer reaching through `metadata.freq_setup`.
const _FREQ_SETUP_FIELDS = (:freqid, :ref_freq, :channel_freqs, :if_freqs, :ch_widths, :total_bandwidths, :sidebands)

function Base.getproperty(m::ObsMetadata, s::Symbol)
    s in _FREQ_SETUP_FIELDS && return getproperty(getfield(m, :freq_setup), s)
    return getfield(m, s)
end

Base.propertynames(m::ObsMetadata) = (fieldnames(typeof(m))..., _FREQ_SETUP_FIELDS...)

function Base.show(io::IO, t::AntennaTable)
    return print(io, "AntennaTable($(t.array_name), $(length(t)) antennas)")
end

function Base.show(io::IO, m::ObsMetadata)
    freq_ghz = round(m.ref_freq / 1.0e9; digits = 3)
    return print(io, "ObsMetadata($(m.object), $(m.telescope), $(m.date_obs), ref=$(freq_ghz) GHz)")
end
