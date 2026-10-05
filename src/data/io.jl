"""
    load_uvfits(path; element_type = nothing) -> XRadio.ProcessingSet

Load an AIPS random-groups UVFITS file as an `XRadio.ProcessingSet` holding one
Measurement Set per (NX scan, FQ row, subarray), keyed
`:<source>_spw_<n>[_sub_<n>]_scan_<n>` (e.g. `:src_3C273_spw_0_scan_1`); the
`sub_<n>` segment appears only when the file carries more than one AN table, and
each subarray's Measurement Set takes its antennas from its own.

Each Measurement Set conforms to [`GUSTAVO_VISIBILITY_SCHEMA`](@ref): dense
`(polarization, frequency, baseline_id, time)` visibility, flag and weight
arrays in Julia order, `uvw` in meters, the Earth-orientation block of the AN
table, and each window's sideband and total bandwidth. When the FREQ axis holds
one channel, the IFs of an FQ row are the channels of one spectral window;
when it holds several, each IF is a window of its own, numbered
`(FQ row - 1) * nIF + IF - 1`.
`VISIBILITY.units` is the file's `BUNIT` (`JY` written `Jy`, `UNCALIB` written
`uncalib`); a file with none
is written `uncalib`, with a warning. Where the file states `INTTIM`, each record's
value is written to `effective_integration_time` over `(baseline_id, time)`, 0
for a slot no record fills, and the time axis' nominal `integration_time` is the
largest of them; otherwise `effective_integration_time` is absent and
`integration_time` is the smallest positive spacing of the scan's times.

A record's weight `w` becomes:

| record           | `weight` | `flag`  |
|:-----------------|:---------|:--------|
| `w > 0`, `+0.0`  | `w`      | `false` |
| `w < 0`, `-0.0`  | `abs(w)` | `true`  |
| no record        | `0`      | `true`  |

with the visibility read as stored, or `NaN` for a (time, baseline) slot no
record fills.

`element_type` is the real float type the visibilities and weights are stored
at; `nothing` takes the precision the file itself holds, so a double-precision
random-groups file is read as `ComplexF64`/`Float64`. `uvw` is always `Float64`.

Visibilities are read verbatim: AIPS UVFITS shares MSv4's phase sense. See
[Conventions](@ref conventions).

Provided by the `GustavoFITSFilesExt` extension; load `FITSFiles` to enable.
"""
function load_uvfits end

"""
    write_uvfits(path, ps::XRadio.ProcessingSet; overwrite = false) -> path

Write `ps` as an AIPS random-groups UVFITS file (AIPS Memo 117) that
[`load_uvfits`](@ref) reads back as `ps`. An existing `path` throws unless
`overwrite = true`.

The set must describe one observation UVFITS can hold:

- one source and field; select one first, e.g. with `filter`;
- one frequency setup: every scan holds the same spectral windows (channel
  frequencies, widths, sidebands and total bandwidths), and the windows of one
  scan share their time and baseline axes;
- one polarization set, a contiguous run of AIPS Stokes codes
  (`RR, LL, RL, LR` = -1…-4, `XX, YY, XY, YX` = -5…-8);
- one subarray: an antenna name states one position, mount and receptor set
  across the set, and no two scans overlap in time. A set read from a file of
  several subarrays throws rather than being written as one.

Each scan's windows become the IFs of one FQ row, ordered by frequency, and the
records run in (time, baseline) order, one NX row per scan. A single window is
written one channel per IF, the layout `load_uvfits` reads as one window; several
windows are written one per IF with their channels along the FREQ axis, which
requires equal channel counts and channels evenly spaced by the channel width.
Several one-channel windows have the single-window layout and so read back as
one window whose channels are the windows; they must then share their channel
width, sideband and total bandwidth. The FREQ axis' reference value is the
lowest window's reference frequency, which every window reads back with.

Visibilities are written verbatim, at their own float type (`Float32` as
`BITPIX = -32`, `Float64` as `-64`), and `uvw` in light-seconds. A flagged
sample's weight is written negated, `-0.0` for a weight of zero. A (time,
baseline) record whose every sample is flagged with weight zero is not
written; such a sample in a written record has visibility 0, which it reads
back with. `BUNIT` is the visibilities' `units` (`Jy` as `JY`, `uncalib` as
`UNCALIB`), and `INTTIM` the `effective_integration_time`, else the time axis'
`integration_time`. `DATE` is the Julian Date of the day's 0h UT and the fraction of that day.

The AN table's Earth-orientation cards come from the `earth_orientation`
attribute ([`GUSTAVO_VISIBILITY_SCHEMA`](@ref)); a set without one, such as a
store from `XRadio.fitsidi2msv4`, is written with `load_uvfits`' defaults, with a
warning naming them.

Not written: the sub-datasets other than antenna and field
(`system_calibration`, `gain_curve`, `weather`, `phase_calibration`),
`sub_scan_name`, `scan_intents`, and every attribute UVFITS has no place for.
The file is not a byte copy of any file the set was read from.

Provided by the `GustavoFITSFilesExt` extension; load `FITSFiles` to enable.
"""
function write_uvfits end
