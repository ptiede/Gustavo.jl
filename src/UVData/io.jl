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
table, and each window's sideband and total bandwidth. Each IF is one channel.
`VISIBILITY.units` is the file's `BUNIT` (`JY` written `Jy`); a file with none
is written `uncalib`, with a warning. Where the file states `INTTIM`, each record's
value is written to `effective_integration_time` over `(baseline_id, time)`, 0
for a slot no record fills, and the time axis' nominal `integration_time` is the
largest of them; otherwise `effective_integration_time` is absent and
`integration_time` is the smallest positive spacing of the scan's times.

A record's weight `w` becomes:

| record           | `weight` | `flag`  |
|:-----------------|:---------|:--------|
| `w > 0`          | `w`      | `false` |
| `w < 0`          | `abs(w)` | `true`  |
| `w == 0`         | `0`      | `false` |
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
