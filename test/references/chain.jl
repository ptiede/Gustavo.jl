# The cases and the calibration chain `record.jl` records and `compare.jl`
# replays, so both run the same steps on the same data.

using Gustavo
using Gustavo: Calibration
using XRadio
using LinearAlgebra: BLAS

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const SYNTHETIC_DIR = joinpath(REPO, "testdata", "references")

const BT164A_DIR = expanduser("~/Research/M87Monitor/BT164/BT164A")
const BT164A_QBAND = joinpath(BT164A_DIR, "VLBA_BT164A_bt164aQband_BIN0_SRC0_0_260108T230453.idifits")
const BT164A_STORE = joinpath(BT164A_DIR, "VLBA_BT164A_bt164aQband.ps.zarr")
const BT164A_REFERENCE = joinpath(BT164A_DIR, "reference_Qband_scans_1-4.zarr")
# One cycle of the schedule: 3C273, M87, 3C279, M87.
const BT164A_SCANS = string.(1:4)

convert_bt164a() = fitsidi2msv4(BT164A_QBAND, BT164A_STORE; mode = "w-", release_date = "2000-01-01")

bt164a_scans() = filter(ms -> all(in(BT164A_SCANS), ms[:scan_name]), open(ProcessingSet, BT164A_STORE))

# Keyword arguments of `_build_fringe_ps` (test/synthetic_ps.jl).
const SYNTHETIC_CASES = [
    (name = "chain_two_scans", fixture = (; seed = 1234, nant = 4, nspw = 2, nchan = 8, ntime = 12, nscans = 2, noise = 0.05)),
    (
        name = "phase_bandpass",
        fixture = (;
            seed = 1234, nant = 4, nspw = 2, nchan = 8, noise = 0.05,
            bandpass = [a == 1 ? 0.0 : 0.4 * sin(2π * c / 16 + a + f) for a in 1:4, f in 1:2, c in 1:16],
        ),
    ),
    (name = "four_scans_gapped", fixture = (; seed = 1234, nant = 5, nspw = 2, nchan = 8, ntime = 10, nscans = 4, scan_gap = 600.0, noise = 0.05)),
]

synthetic_input(name) = joinpath(SYNTHETIC_DIR, "$name.ps.zarr")
synthetic_output(name) = joinpath(SYNTHETIC_DIR, "$name.zarr")

# A second-order random walk along frequency on both observables, with σ in
# radians (nepers) per channel.
function bandpass_model(ps; phase_σ = 0.05, logamp_σ = 0.02)
    f = Calibration.DataGeometry(ps).channel_freqs
    Δν = minimum(abs, diff(f))
    rw2(σ) = Calibration.RandomWalkPrior(; order = 2, σ = σ * sqrt(3 / (2 * Δν^3)))
    comp(prior) = GainComponent(ConstantTerm(); Ti = Calibration.GlobalTime(), Frequency = ChannelBlocks(1), Feed = PerFeed(), prior)
    return Calibration.GainModel(; phase = (; bandpass = comp(rw2(phase_σ))), logamp = (; bandpass = comp(rw2(logamp_σ))))
end

const CHAIN_TEXT = "fit(BaselineFringeFit()) → calibrate! → fit(Bandpass(; model = bandpass_model(ps))) → " *
    "calibrate! → fit(AdhocPhase()), each step's default gauge and calibrate!'s default flags"

# The gains of every step, composed; the steps share one grid.
composed_gains(sols) = reduce((a, b) -> a .* b, map(gains, values(sols)))

# Each step is fit on data the steps before it corrected in place.
function replay(ps)
    BLAS.set_num_threads(1)
    ps = Gustavo.materialize(ps)
    fringe = fit(BaselineFringeFit(), ps)
    calibrate!(fringe, ps)
    bandpass = fit(Bandpass(; model = bandpass_model(ps)), ps)
    calibrate!(bandpass, ps)
    adhoc = fit(AdhocPhase(), ps)
    return (; fringe, bandpass, adhoc)
end
