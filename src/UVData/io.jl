"""
    load_uvfits(path) -> UVSet

Load a UVData file, returning a `UVSet` whose `branches` is a flat
`OrderedDict` of MSv4-shaped per-scan leaf `DimTree`s keyed by sanitized
`:<source>_scan_<n>` Symbols. Each leaf carries dense
`(Ti, Baseline, Pol, IF)` cubes for `vis`/`weights` and
`(Ti, Baseline, UVW)` for `uvw`, mirroring xradio's MSv4 visibility schema.

Provided by the `GustavoFITSFilesExt` extension; load `FITSFiles` to enable.
"""
function load_uvfits end

"""
    write_uvfits(output_path, uvset::UVSet)

Write a UVData file by walking the leaves of `uvset` directly and emitting
random-groups records in scan-insertion order, then assembling the AN, FQ,
and NX bintables from the root metadata.

Single-source UVSets only — multi-source UVSets must first be narrowed via
`select_source(uvset, name)`.

Provided by the `GustavoFITSFilesExt` extension; load `FITSFiles` to enable.
"""
function write_uvfits end
