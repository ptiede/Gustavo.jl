# ── Scan selections: fit-on-subset / apply-everywhere ────────────────────────
#
# An `AbstractScanSelection` names WHICH scans feed a solve stage's
# accumulation. Time-global model components (a track-wide bandpass, a global
# R–L delay) then apply to every scan even when fit from a few — the
# "fit on bright calibrator scans, apply across the board" pattern. Selections
# are resolved against a table of per-scan records `(index, source, scan, snr)`
# (snr = the stage-A per-scan max detection SNR; NaN before stage A has run).

"""
    AbstractScanSelection

Selects the scans a solve stage accumulates from. Resolve with
[`select_scans`](@ref). Built-ins: [`AllScans`](@ref), [`SourceScans`](@ref),
[`BrightestCalibrator`](@ref), [`ScanIndices`](@ref), [`ScanWhere`](@ref).
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

"""
    BrightestCalibrator(; max_scans = 0)

The single source with the highest TOTAL stage-A detection SNR over its scans
(the "brightest calibrator"), optionally capped to its `max_scans` highest-SNR
scans (`0` = all of them). The bandpass stage's default: a time-stable solve
needs only a few strong scans, and skipping the rest avoids reading them.
Requires stage-A SNRs, so it resolves only after a fringe search has run.
"""
Base.@kwdef struct BrightestCalibrator <: AbstractScanSelection
    max_scans::Int = 0
end

function select_scans(sel::BrightestCalibrator, scans)
    isempty(scans) && return Int[]
    total = Dict{String, Float64}()
    for s in scans
        isfinite(s.snr) || continue
        total[s.source] = get(total, s.source, 0.0) + s.snr
    end
    isempty(total) &&
        error("BrightestCalibrator: no scan carries a finite SNR — this selection resolves only after a fringe search has run.")
    best = argmax(total)
    picked = [s for s in scans if s.source == best]
    if sel.max_scans > 0 && length(picked) > sel.max_scans
        sort!(picked; by = s -> -s.snr)
        picked = picked[1:sel.max_scans]
    end
    return sort!(Int[s.index for s in picked])
end
