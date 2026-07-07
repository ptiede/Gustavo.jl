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

"""
    load_fitsidi(path; lazy = true, scans = :, bands = :) -> UVSet

Load a FITS-IDI file (AIPS Memo 114) into a `UVSet`. Header tables
(ARRAY_GEOMETRY, FREQUENCY, SOURCE, ANTENNA, …) are read eagerly; the
`UV_DATA` payload is left lazy by default — each per-(scan, band) leaf's
`vis`/`weights`/`flag` layers are disk-backed and materialized on demand
(`materialize_leaf`). Pass `lazy = false` to materialize everything up front
(only for small files), or restrict `scans`/`bands` to a subset.

Provided by the `GustavoFITSFilesExt` extension; load `FITSFiles` to enable.
"""
function load_fitsidi end

"""
    write_fitsidi(output_path, uvset::UVSet)

Write a `UVSet` to a FITS-IDI file (AIPS Memo 114): a stub PRIMARY HDU plus
ARRAY_GEOMETRY, SOURCE, ANTENNA, FREQUENCY, and a time-ordered `UV_DATA`
binary table. Used primarily to build round-trip and fringe-injection test
fixtures.

Provided by the `GustavoFITSFilesExt` extension; load `FITSFiles` to enable.
"""
function write_fitsidi end

"""
    default_output_path(uvset::UVSet; dir = pwd(), ext = "uvfits") -> String

Default on-disk path for writing `uvset`, named after its (single) real
`source_name` — deliberately NOT the internal `source_key`, so a digit-leading
catalog name like `"3C273"` yields `3C273.uvfits`, not the identifier-safe
`src_3C273` tree key. Characters that are illegal or awkward in a filename are
replaced with `_`; `ext` (without a leading dot) is appended as the extension.

Multi-source UVSets must first be narrowed via `select_source(uvset, name)`.
"""
function default_output_path(uvset::UVSet; dir = pwd(), ext = "uvfits")
    srcs = sources(uvset)
    length(srcs) == 1 || error(
        "default_output_path: expected a single-source UVSet; got sources=$(srcs). " *
            "Use select_source(uvset, name) first.",
    )
    name = replace(strip(only(srcs)), r"[^A-Za-z0-9._+-]+" => "_")
    isempty(name) && (name = "uvdata")
    return joinpath(dir, string(name, ".", ext))
end

"""
    apply_calibration(uvset::UVSet, calibration; kwargs...) -> UVSet

Apply a calibration to `uvset`, returning a corrected `UVSet`. Generic entry
point with methods for different calibration objects (an ANTAB a-priori
calibration in `Gustavo.Bandpass`; a `Gustavo.Calibration.CalibrationSolution`
fringe/bandpass solution). Visibilities are divided by `g_a · conj(g_b)` and
weights scaled by `|g_a g_b|²`.
"""
function apply_calibration end

"""
    primary_cards(uvset::UVSet) -> Vector

Return the FITS primary-HDU cards registered for `uvset`. Provided by
the `GustavoFITSFilesExt` extension; cards are registered automatically
on `load_uvfits` and can be set explicitly via `register_primary_cards!`.
"""
function primary_cards end

"""
    register_primary_cards!(uvset::UVSet, cards)

Register a vector of FITS primary-HDU cards for `uvset`. Used by callers
that build a `UVSet` from scratch (e.g. test fixtures) before calling
`write_uvfits`. Provided by the `GustavoFITSFilesExt` extension.
"""
function register_primary_cards! end
