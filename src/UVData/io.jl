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
