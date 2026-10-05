# Shared usings and aliases for the pipeline-level tests, the full component mix
# of the standard pipeline, and the `_coherence` metric.
using Gustavo
using Test
using Random
using StableRNGs: StableRNG
using LinearAlgebra: Diagonal
using StructArrays
using DimensionalData
using DimensionalData: DimArray, Ti, dims, lookup
using PolarizedTypes: RPol, LPol
using Gustavo.UVData: Polarization, Frequency, UVW, BaselineID, channel_freqs
using Gustavo.UVData: antennas, baselines, frequencies
using Dates: Date, DateTime, datetime2unix
using FITSFiles   # triggers GustavoFITSFilesExt

const CAL = Gustavo.Calibration
const FP = Gustavo.Fring
const UVP = Gustavo.UVData

include("synthetic_ps.jl")

# The union of the components the standard pipeline's steps solve (each on its
# own private θ in a real run), assembled as ONE GainModel: the fringe
# terms plus the phase/log-amplitude bandpass and the adhoc phase. Structural
# tests use it to exercise the plan routers and θ decoding on a realistic full
# component mix.
function _full_fringe_model(; rel_time = PerScan())
    return GainModel(
        phase = merge(
            default_fringe_terms(; rel_time).phase,
            (
                bandpass = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed()),
                adhoc = GainComponent(ConstantTerm(); Ti = PerIntegration(), Frequency = GlobalFrequency(), Feed = SharedFeeds()),
            ),
        ),
        logamp = (
            bandpass = GainComponent(ConstantTerm(); Ti = GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed()),
        ),
    )
end

# Coherence of a (baseline, product) block: |Σ w·V| / Σ (w·|V|). 1 ⇒ phase flat.
function _coherence(V, W)
    num = zero(ComplexF64)
    den = 0.0
    for i in eachindex(V)
        v = V[i]
        w = W[i]
        (isfinite(v) && isfinite(w) && w > 0) || continue
        num += w * v
        den += w * abs(v)
    end
    den == 0 && return 0.0
    return abs(num) / den
end
