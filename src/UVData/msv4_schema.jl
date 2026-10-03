# What Gustavo carries that MSv4 has no field for.
#
# MSv4 permits extra attributes unconditionally, and extra coordinates provided
# they span dimensions the schema already defines. Everything here is one or the
# other, so a store carrying it still conforms; the specification is what makes
# `check` examine it rather than pass over it, and what gives `sub_scan_name`
# its lower-case name on disk.

"""
    EARTH_ORIENTATION

The time-system and Earth-orientation block, from the AIPS `AN` table.

MSv4 states no equivalent. Every field is required once the block is present,
so a half-filled one is reported rather than read as zeros.
"""
const EARTH_ORIENTATION = XRadio.DictSpec(
    :earth_orientation,
    [
        XRadio.AttrSpec(
            :gst_iat0, :float;
            doc = "Greenwich sidereal time at IAT 0 on the reference date, in degrees"
        ),
        XRadio.AttrSpec(
            :earth_rot_rate, :float;
            doc = "Earth rotation rate, in degrees per IAT day"
        ),
        XRadio.AttrSpec(:ut1utc, :float; doc = "UT1 − UTC, in seconds"),
        XRadio.AttrSpec(:polarx, :float; doc = "Polar motion x, in arcseconds"),
        XRadio.AttrSpec(:polary, :float; doc = "Polar motion y, in arcseconds"),
        XRadio.AttrSpec(:datutc, :float; doc = "IAT − UTC, in seconds"),
        XRadio.AttrSpec(
            :xyzhand, :str; values = ["RIGHT", "LEFT"],
            doc = "Handedness of the antenna coordinate system"
        ),
        XRadio.AttrSpec(
            :poltype, :str;
            doc = "Polarization calibration scheme, such as `APPROX` or `ORI-ELP`"
        ),
    ];
    doc = "Time-system and Earth-orientation parameters, from the AIPS `AN` table"
)

"""
    GUSTAVO_VISIBILITY_SCHEMA

`XRadio.VISIBILITY_SCHEMA` with the three things Gustavo needs and MSv4 does not
define: the [`EARTH_ORIENTATION`](@ref) block, the per-channel `sideband` and
`total_bandwidth` of a [`FrequencySetup`](@ref), and a `sub_scan_name`
coordinate.

Pass it to `XRadio.check` and `XRadio.write` as `schemas`:

```julia
XRadio.check(ps; schemas = [GUSTAVO_VISIBILITY_SCHEMA])
```

Each addition is optional, so a store written by anything else still passes.
What the specification buys is that a store which *does* carry them is examined
rather than waved through, and that `sub_scan_name` is written under that name
and listed among the coordinates of the variables it labels — without it the
layer is written `SUB_SCAN_NAME` and a Python reader sees a data variable.

`sideband` cannot be restricted to `±1`: a schema's permitted values are
strings, so a stray `0` passes here and is caught by whatever reads it.
"""
const GUSTAVO_VISIBILITY_SCHEMA = XRadio.extend(
    XRadio.VISIBILITY_SCHEMA;
    attrs = [
        XRadio.AttrSpec(
            :earth_orientation, :dict; optional = true, nested = EARTH_ORIENTATION,
            doc = "Time-system and Earth-orientation parameters"
        ),
    ],
    coords = [
        XRadio.ArraySpec(
            "frequency", (:frequency,), Float64;
            attrs = [
                XRadio.AttrSpec(
                    :sideband, :int; optional = true,
                    doc = "Sideband of each channel: +1 upper, −1 lower"
                ),
                XRadio.AttrSpec(
                    :total_bandwidth, :dataarray; optional = true,
                    nested = XRadio.ArraySpec(
                        "total_bandwidth", (), Float64;
                        attrs = [
                            XRadio.AttrSpec(
                                :type, :str; values = ["quantity"], default = "quantity"
                            ),
                            XRadio.AttrSpec(
                                :units, :str; values = ["Hz"], default = "Hz"
                            ),
                        ],
                    ),
                    doc = "Nominal total bandwidth of the setup, in hertz. MSv4's \
                           `EFFECTIVE_CHANNEL_WIDTH` is what survived flagging, which \
                           is a different quantity."
                ),
            ],
        ),
        XRadio.ArraySpec(
            "sub_scan_name", (:time,), String; optional = true,
            doc = "Sub-scan observed at each time step, within the scan `scan_name` names."
        ),
    ],
)
