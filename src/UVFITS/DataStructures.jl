"""
    Antenna{TName,TXYZ,TMnt,TAxOff,TDiam,TFa,TFb,TPola,TPolb}

Single-antenna record drawn from the AIPS AN binary table (Memo 117 §4.1).
Intended to be collected into a `StructArray` to form an `AntennaTable`.

`station_xyz` is the (x,y,z) coordinate of the antenna *relative to the array
center*, in the frame given by `AntennaTable.frame` (typically ITRF), rotated
to the longitude of the array center.  Mount type codes: 0=alt-az, 1=equatorial,
2=orbiting, 3=X-Y, 4=right-Naismith, 5=left-Naismith, 6=aperture/phased array.
"""
struct Antenna{TName,TXYZ,TMnt,TAxOff,TDiam,TFa,TFb,TPola,TPolb}
    name::TName             # ANNAME — station code
    station_xyz::TXYZ       # STABXYZ — 3-element (x,y,z) in meters
    mount_type::TMnt        # MNTSTA
    axis_offset::TAxOff     # STAXOF — Y-axis offset (meters)
    diameter::TDiam         # DIAMETER (meters)
    feed_a::TFa             # POLTYA — 'R','L','X','Y'
    feed_b::TFb             # POLTYB
    pola_angle::TPola       # POLAA — feed A position angle (degrees)
    polb_angle::TPolb       # POLAB — feed B position angle (degrees)
end

"""
    AntennaTable{TAnt,TXyz,TName,TFreq,TRdate,TGst,TDeg,TUt,TSys,TFrame,THand}

Array-of-structs antenna table.  `antennas` is a `StructArray{Antenna}`;
the remaining fields carry array-level metadata from the AIPS AN table header.
"""
struct AntennaTable{TAnt,TXyz,TName,TFreq,TRdate,TGst,TDeg,TUt,TSys,TFrame,THand}
    antennas::TAnt          # StructArray{Antenna}
    array_xyz::TXyz         # ARRAYX/Y/Z — array center (meters)
    array_name::TName       # ARRNAM
    ref_freq::TFreq         # FREQ — reference frequency (Hz)
    rdate::TRdate           # RDATE — reference date
    gst_iat0::TGst          # GSTIA0 — GST at 0h on reference date (degrees)
    earth_rot_rate::TDeg    # DEGPDY — Earth rotation rate (degrees/day)
    ut1utc::TUt             # UT1UTC — UT1 minus UTC (seconds)
    time_sys::TSys          # TIMSYS — 'IAT' or 'UTC'
    frame::TFrame           # FRAME — coordinate frame (e.g. 'ITRF')
    xyzhand::THand          # XYZHAND — 'RIGHT' or 'LEFT'
end

Base.length(t::AntennaTable) = length(t.antennas)
Base.getindex(t::AntennaTable, i) = t.antennas[i]
Base.iterate(t::AntennaTable, args...) = iterate(t.antennas, args...)

"""
    ObsMetadata{TObj,TTel,TObs,TDate,TEq,TBunit,TRa,TDec,TFreq,TCfreqs,TBw,TCw,TSb,TPcodes,TPlabs}

Observation-level metadata from the primary HDU FITS cards and the AIPS FQ
frequency table (Memo 117 §4.7).

`channel_freqs` are absolute IF center frequencies in Hz (reference frequency
+ FQ `IF FREQ` offset).  `channel_bwidths` and `ch_width` come from `TOTAL
BANDWIDTH` and `CH WIDTH` columns of the FQ table, respectively.
"""
struct ObsMetadata{TObj,TTel,TObs,TDate,TEq,TBunit,TRa,TDec,TFreq,TCfreqs,TBw,TCw,TSb,TPcodes,TPlabs}
    object::TObj            # OBJECT — source name
    telescope::TTel         # TELESCOP
    observer::TObs          # OBSERVER
    date_obs::TDate         # DATE-OBS
    equinox::TEq            # EQUINOX (year, e.g. 2000.0)
    bunit::TBunit           # BUNIT — flux units ('UNCALIB','JY',…)
    ra::TRa                 # phase center RA (degrees)
    dec::TDec               # phase center Dec (degrees)
    ref_freq::TFreq         # reference frequency (Hz)
    channel_freqs::TCfreqs  # absolute IF center frequencies (Hz)
    channel_bwidths::TBw    # total IF bandwidth per spectral window (Hz)
    ch_width::TCw           # spectral channel separation (Hz)
    sidebands::TSb          # SIDEBAND: +1=upper, -1=lower
    pol_codes::TPcodes      # Stokes codes (-1=RR,-2=LL,-3=RL,-4=LR)
    pol_labels::TPlabs      # polarization labels ('11','22','12','21')
end
