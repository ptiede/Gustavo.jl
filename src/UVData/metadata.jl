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


"""
    ObsArrayMetadata(; telescope, instrume, date_obs, equinox, bunit,
                       freq_setup, extras = NamedTuple())

Array-wide observation metadata that is **shared across every source /
partition** in a UVSet: telescope, frequency setup, primary-HDU extras.
Source-specific fields (`object`, `ra`, `dec`) live on each per-leaf
`partition_info` NamedTuple. Polarization products are not stored here —
they live on the `Pol` dimension of the data cube and can be read with
`pol_products(uvset)` / `pol_products(leaf)` / `pol_products(data)`.

- `freq_setup`  : see `FrequencySetup`. Frequency-axis fields are reached
  via `m.freq_setup.channel_freqs` etc. (no field forwarding).
- `extras`      : `NamedTuple` of optional primary-HDU cards preserved
  verbatim for round-trip.
"""
Base.@kwdef struct ObsArrayMetadata{
        TTel, TInst, TDate, TEq, TBunit,
        TFs <: FrequencySetup, TExtras <: NamedTuple,
    }
    telescope::TTel
    instrume::TInst
    date_obs::TDate
    equinox::TEq
    bunit::TBunit
    freq_setup::TFs
    extras::TExtras = NamedTuple()
end

freqsetup(m::ObsArrayMetadata) = m.freq_setup

function Base.show(io::IO, m::ObsArrayMetadata)
    fs = freqsetup(m)
    freq_ghz = round(fs.ref_freq / 1.0e9; digits = 3)
    return print(io, "ObsArrayMetadata($(m.telescope), $(m.date_obs), ref=$(freq_ghz) GHz)")
end

"""
    UVMetadata

Bundle of array-wide observation globals shared across every leaf of a
`UVSet`: scan time bounds, antenna table, array config, the array-wide
`ObsArrayMetadata` (telescope / freq setup / pol codes), and the FITS
primary cards preserved verbatim for write-back. Source-specific fields
(`object`, `ra`, `dec`) live on each leaf's `partition_info` NamedTuple.
"""
struct UVMetadata{TScans, TAnt, TCfg, TObs <: ObsArrayMetadata, TCards}
    scans::TScans              # StructArray{lower, upper}
    antennas::TAnt             # AntennaTable
    array_config::TCfg         # ArrayConfig
    array_obs::TObs            # ObsArrayMetadata
    primary_cards::TCards      # Vector{Card}
end
