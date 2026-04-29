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
                       extras = NamedTuple())

Array-wide observation metadata that is **shared across every source /
partition** in a UVSet: telescope, observation epoch, primary-HDU extras.
Source-specific fields (`object`, `ra`, `dec`) and the per-partition
`freq_setup` live on each per-leaf `PartitionInfo`. Polarization products
are not stored here — they live on the `Pol` dimension of the data cube
and can be read with `pol_products(uvset)` / `pol_products(leaf)` /
`pol_products(data)`.

- `extras`      : `NamedTuple` of optional primary-HDU cards preserved
  verbatim for round-trip.
"""
Base.@kwdef struct ObsArrayMetadata{
        TTel, TInst, TDate, TEq, TBunit, TExtras <: NamedTuple,
    }
    telescope::TTel
    instrume::TInst
    date_obs::TDate
    equinox::TEq
    bunit::TBunit
    extras::TExtras = NamedTuple()
end

Base.:(==)(a::ObsArrayMetadata, b::ObsArrayMetadata) =
    a.telescope == b.telescope && a.instrume == b.instrume &&
    a.date_obs == b.date_obs && a.equinox == b.equinox &&
    a.bunit == b.bunit && a.extras == b.extras
Base.hash(a::ObsArrayMetadata, h::UInt) = hash(
    (a.telescope, a.instrume, a.date_obs, a.equinox, a.bunit, a.extras),
    hash(:ObsArrayMetadata, h),
)

Base.:(==)(a::ArrayConfig, b::ArrayConfig) =
    a.rdate == b.rdate && a.gst_iat0 == b.gst_iat0 &&
    a.earth_rot_rate == b.earth_rot_rate && a.ut1utc == b.ut1utc &&
    a.polarx == b.polarx && a.polary == b.polary &&
    a.datutc == b.datutc && a.time_sys == b.time_sys &&
    a.frame == b.frame && a.xyzhand == b.xyzhand &&
    a.poltype == b.poltype && a.extver == b.extver &&
    a.numorb == b.numorb && a.no_if == b.no_if &&
    a.nopcal == b.nopcal && a.freqid == b.freqid
Base.hash(a::ArrayConfig, h::UInt) = hash(
    (
        a.rdate, a.gst_iat0, a.earth_rot_rate, a.ut1utc,
        a.polarx, a.polary, a.datutc, a.time_sys, a.frame, a.xyzhand,
        a.poltype, a.extver, a.numorb, a.no_if, a.nopcal, a.freqid,
    ),
    hash(:ArrayConfig, h),
)

function Base.show(io::IO, m::ObsArrayMetadata)
    return print(io, "ObsArrayMetadata($(m.telescope), $(m.date_obs))")
end

"""
    UVMetadata

Bundle of array-wide observation globals shared across every leaf of a
`UVSet`: antenna table, array config, and the array-wide
`ObsArrayMetadata` (telescope / observation epoch).

Per-leaf metadata — including the leaf's frequency setup, scan handles
(`scan_name`, `scan_intents`, `sub_scan_name`), and source identification
(`source_name`/`ra`/`dec`) — lives on each leaf's `PartitionInfo`. There
is *no* root-level scan table; scan time bounds derive from each leaf's
`Ti` axis (mirrors xradio's `ScanArray` / `ProcessingSet` model).

FITS primary-HDU cards (write-back state) live in a FITS-extension-owned
`WeakKeyDict` keyed by `UVSet`; access via the extension's
`primary_cards(uvset)` accessor. They are not part of `UVMetadata`
because they are pure UVFITS-format state, not format-neutral
observation metadata.
"""
struct UVMetadata{TAnt, TCfg, TObs <: ObsArrayMetadata}
    antennas::TAnt             # AntennaTable
    array_config::TCfg         # ArrayConfig
    array_obs::TObs            # ObsArrayMetadata
end
