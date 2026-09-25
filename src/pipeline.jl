# ── Calibration pipeline ──────────────────────────────────────────────────────
#
# A pipeline is an ordered tuple of solve steps and corrections. `fit` solves
# the steps in order, each on the data the corrections and steps before it
# produced, and records the sequence on the solution; `calibrate` replays it.

# The per-operation `UVSet -> UVSet` kernels backing the output reducers.
using .UVData: scan_average, time_bin_average, frequency_average, flag_spw_edges,
    combine_spw

"""
    AprioriAmplitude(spw_cals; min_elevation_deg = 0.0, on_missing_station = :warn)

A-priori amplitude calibration, applied to the output: `spw_cals` (from
`load_fitsidi_apriori(path)`) scales the corrected data after the solution's
gains and before `post`, wherever the element sits in the pipeline, so a
solve step never sees the scaled amplitudes. It is recorded in the solution's
sequence, so `calibrate(sol, uvset)` applies it without re-passing `spw_cals`.
"""
struct AprioriAmplitude{C}
    spw_cals::C
    min_elevation_deg::Float64
    on_missing_station::Symbol
end
AprioriAmplitude(spw_cals; min_elevation_deg::Real = 0.0, on_missing_station::Symbol = :warn) =
    AprioriAmplitude(spw_cals, Float64(min_elevation_deg), on_missing_station)

_apply_apriori(s::AprioriAmplitude, uv) = UVData.apply_calibration(
    uv, s.spw_cals; min_elevation_deg = s.min_elevation_deg, on_missing_station = s.on_missing_station,
)

# Run-wide resources: schedulers, the memory budget, progress.
include("pipeline/execution.jl")

# Corrections: Measurement Set → Measurement Set, the recorded ones as structs.
include("pipeline/corrections.jl")

# The step protocol: SolveStep, its hooks, and `|>` building a pipeline.
include("pipeline/protocol.jl")

# The built-in solve steps: BaselineFringeFit, DispersionSBDFit, Bandpass,
# AdhocPhase.
include("pipeline/steps.jl")


"""
    AverageFrequency(; nout = 1)

`uvset -> uvset`: inverse-variance-average each spw's `Frequency` axis into
`nout` channels. For `calibrate`'s `post`, e.g.
`post = CombineSpw() ∘ AverageFrequency(nout = 1)`.
"""
Base.@kwdef struct AverageFrequency
    nout::Int = 1
end
(s::AverageFrequency)(uv::UVSet) = frequency_average(uv; nout = s.nout)

"""
    CombineSpw()

`uvset -> uvset`: merge sibling spw leaves into one `Frequency` axis. Apply
after [`AverageFrequency`](@ref) so each spw is one channel.
"""
struct CombineSpw end
(::CombineSpw)(uv::UVSet) = combine_spw(uv)

"""
    AverageTime(; seconds = nothing)

`uvset -> uvset`: inverse-variance-average each leaf's `Ti` axis — into
`seconds`-wide bins, or, with no `seconds`, collapsing each scan to a single
sample.
"""
Base.@kwdef struct AverageTime
    seconds::Union{Nothing, Float64} = nothing
end
(s::AverageTime)(uv::UVSet) = s.seconds === nothing ? scan_average(uv) : time_bin_average(uv, s.seconds)

"""
    FlagSpwEdges(; mode = :flag_fraction, fraction = 0.0)

`uvset -> uvset`: handle polyphase-filterbank spectral-window edges.
`:flag_fraction` zeros the outer `fraction` of channels' weights at each edge;
`:trim` drops them.
"""
Base.@kwdef struct FlagSpwEdges
    mode::Symbol = :flag_fraction
    fraction::Float64 = 0.0
end
(s::FlagSpwEdges)(uv::UVSet) = flag_spw_edges(uv; mode = s.mode, fraction = s.fraction)

# ── Runner ────────────────────────────────────────────────────────────────────

# The pipeline verbs: fit and calibrate.
include("pipeline/verbs.jl")
