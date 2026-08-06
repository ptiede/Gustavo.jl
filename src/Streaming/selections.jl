# ── Scan selections: fit-on-subset / apply-everywhere ────────────────────────
#
# An `AbstractScanSelection` names WHICH scans feed a solve stage's
# accumulation. Time-global model components (a track-wide bandpass, a global
# inter-feed delay) then apply to every scan even when fit from a few — the
# "fit on bright calibrator scans, apply across the board" pattern. Selections
# are resolved against a table of per-scan records `(index, source, scan, snr)`
# (snr = the stage-A per-scan max detection SNR; NaN before stage A has run).

"""
    AbstractScanSelection

Selects the scans a solve stage accumulates from. Resolve with
[`select_scans`](@ref). Built-ins: [`AllScans`](@ref), [`SourceScans`](@ref),
[`ScanIndices`](@ref), [`ScanWhere`](@ref).
"""
abstract type AbstractScanSelection end

"""
    select_scans(sel::AbstractScanSelection, scans) -> Vector{Int}

Resolve `sel` against per-scan records — an indexable collection of
`(; index, source, scan, snr)` NamedTuples — returning the selected `index`es
in ascending order. The extension point for custom selections.
"""
function select_scans end

"""
    AllScans()

Every scan (the default selection).
"""
struct AllScans <: AbstractScanSelection end
select_scans(::AllScans, scans) = Int[s.index for s in scans]

"""
    SourceScans(names...)

Scans of the named source(s) only.
"""
struct SourceScans <: AbstractScanSelection
    sources::Vector{String}
end
SourceScans(names::AbstractString...) = SourceScans(collect(String, names))
select_scans(sel::SourceScans, scans) =
    Int[s.index for s in scans if s.source in sel.sources]

"""
    ScanIndices(idx)

Explicit scan-group indices.
"""
struct ScanIndices <: AbstractScanSelection
    idx::Vector{Int}
end
ScanIndices(idx::Integer...) = ScanIndices(collect(Int, idx))
select_scans(sel::ScanIndices, scans) =
    Int[s.index for s in scans if s.index in sel.idx]

"""
    ScanWhere(pred)

Scans for which `pred(record)` is true; `record` is the per-scan NamedTuple
`(; index, source, scan, snr)`.
"""
struct ScanWhere{F} <: AbstractScanSelection
    pred::F
end
select_scans(sel::ScanWhere, scans) =
    Int[s.index for s in scans if sel.pred(s)]
