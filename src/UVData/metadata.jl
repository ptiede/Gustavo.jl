"""
    ObsArrayMetadata(; telescope, instrume, date_obs, equinox, bunit,
                       rdate = "", gst_iat0 = 0f0, earth_rot_rate = 360f0,
                       ut1utc = 0f0, polarx = 0f0, polary = 0f0, datutc = 0f0,
                       time_sys = "UTC", frame = "ITRF", xyzhand = "RIGHT",
                       poltype = "",
                       extras = NamedTuple())

Array-wide observation metadata that is **shared across every source /
partition** in a UVSet: telescope identity, observation epoch, time
system, Earth-orientation parameters, coord frame, polcal scheme.
Source-specific fields (`object`, `ra`, `dec`) and the per-partition
`freq_setup` live on each per-leaf `PartitionInfo`. Polarization products
are not stored here — they live on the `Pol` dimension of the data cube
and can be read with `pol_products(uvset)` / `pol_products(leaf)` /
`pol_products(data)`.

- `rdate`, `gst_iat0`, `earth_rot_rate`, `ut1utc`, `polarx`, `polary`,
  `datutc`: time-system / Earth-orientation parameters (sourced from
  AIPS AN HDU on UVFITS read).
- `time_sys`: "IAT" or "UTC".
- `frame`: coord frame, e.g. "ITRF".
- `xyzhand`: "RIGHT" / "LEFT" handedness convention.
- `poltype`: polcal scheme tag (e.g. "APPROX", "ORI-ELP").
- `extras`: `NamedTuple` of optional primary-HDU cards preserved
  verbatim for round-trip.
"""
Base.@kwdef struct ObsArrayMetadata{
        TTel, TInst, TDate, TEq, TBunit,
        TRdate, TGst, TDeg, TUt, TPol, TDut, TSys, TFrame, THand, TPType,
        TExtras <: NamedTuple,
    }
    telescope::TTel
    instrume::TInst
    date_obs::TDate
    equinox::TEq
    bunit::TBunit
    rdate::TRdate = ""
    gst_iat0::TGst = 0.0f0
    earth_rot_rate::TDeg = 360.0f0
    ut1utc::TUt = 0.0f0
    polarx::TPol = 0.0f0
    polary::TPol = 0.0f0
    datutc::TDut = 0.0f0
    time_sys::TSys = "UTC"
    frame::TFrame = "ITRF"
    xyzhand::THand = "RIGHT"
    poltype::TPType = ""
    extras::TExtras = NamedTuple()
end

Base.:(==)(a::ObsArrayMetadata, b::ObsArrayMetadata) =
    a.telescope == b.telescope && a.instrume == b.instrume &&
    a.date_obs == b.date_obs && a.equinox == b.equinox &&
    a.bunit == b.bunit &&
    a.rdate == b.rdate && a.gst_iat0 == b.gst_iat0 &&
    a.earth_rot_rate == b.earth_rot_rate && a.ut1utc == b.ut1utc &&
    a.polarx == b.polarx && a.polary == b.polary && a.datutc == b.datutc &&
    a.time_sys == b.time_sys && a.frame == b.frame &&
    a.xyzhand == b.xyzhand && a.poltype == b.poltype &&
    a.extras == b.extras
Base.hash(a::ObsArrayMetadata, h::UInt) = hash(
    (
        a.telescope, a.instrume, a.date_obs, a.equinox, a.bunit,
        a.rdate, a.gst_iat0, a.earth_rot_rate, a.ut1utc,
        a.polarx, a.polary, a.datutc,
        a.time_sys, a.frame, a.xyzhand, a.poltype, a.extras,
    ),
    hash(:ObsArrayMetadata, h),
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
struct UVMetadata{TObs <: ObsArrayMetadata}
    array_obs::TObs
end
