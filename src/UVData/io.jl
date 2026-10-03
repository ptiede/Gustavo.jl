"""
    load_uvfits(path; element_type = nothing) -> UVSet

Load an AIPS random-groups UVFITS file, returning a `UVSet` whose `branches` is
a flat `OrderedDict` of MSv4-shaped per-scan leaf `DimTree`s keyed by sanitized
`:<source>_<spw>_scan_<n>` Symbols. Each leaf carries dense
`(Frequency, Ti, BaselineID, Polarization)` cubes for `vis`/`weights`/`flags` and
`(Ti, BaselineID, UVW)` for `uvw`. [`uvset_to_processingset`](@ref) converts the
result to the `XRadio.ProcessingSet` the solver reads.

`element_type` is the real float type the `vis`/`weights` cubes are stored at;
`nothing` takes the precision the file itself holds, so a double-precision
random-groups file is read as `ComplexF64`/`Float64`. Name it explicitly to
store at a different precision — including to narrow a double file deliberately.

Visibilities are read verbatim: AIPS UVFITS shares Gustavo's internal phase
sense. A non-positive weight is read as a flag. See [Conventions](@ref conventions).

Provided by the `GustavoFITSFilesExt` extension; load `FITSFiles` to enable.
"""
function load_uvfits end

"""
    load_fitsidi_apriori(path; tsys_max = 1.0e4) -> Dict{Int, AntabCalibration}

Build per-band a-priori flux calibrations from a FITS-IDI file's `GAIN_CURVE`
(DPFU + elevation gain polynomial) and `SYSTEM_TEMPERATURE` (Tsys) tables, one
[`AntabCalibration`](@ref) per 1-based band index.
Tsys values that are non-positive, the `999` placeholder, or `> tsys_max` are
treated as missing and fall back to the other feed's value for the same
(antenna, band, time). Provided by the `GustavoFITSFilesExt` extension.
"""
function load_fitsidi_apriori end
