"""
    Fringe

VLBI fringe fitting on top of the unified `Gustavo.Calibration` model framework
and the `Gustavo.UVData` visibility model. Stage by stage (EHT-HOPS-inspired,
Blackburn et al. 2019, but with globally-closing per-feed solutions):

- `search.jl`     — per-baseline FFT delay/rate fringe search.
- (stationize / adhoc / pipeline — later phases.)
"""
module Fringe

using ..UVData
using ..Calibration
using FFTW: fft, fftfreq
using Statistics: median, mean
using LinearAlgebra

include("Fringe/search.jl")
include("Fringe/stationize.jl")
include("Fringe/adhoc.jl")
include("Fringe/pipeline.jl")
include("Fringe/diagnostics.jl")

# ── Plot stubs — implemented by `GustavoMakieExt`. Load Makie or CairoMakie
# to enable plotting.
"""
    plot_fringe_spectrum(sol; sites, feeds, ti)
    plot_fringe_spectrum(parent, sol; ...)

Per-station gain phase vs frequency at one time index (delay slope + constant).
Provided by `GustavoMakieExt`.
"""
function plot_fringe_spectrum end

"""
    plot_fringe_phases(sol; sites, feeds, ci)
    plot_fringe_phases(parent, sol; ...)

Per-station gain phase vs time at one channel (rate + adhoc evolution).
Provided by `GustavoMakieExt`.
"""
function plot_fringe_phases end

"""
    plot_fringe_snr(sol)
    plot_fringe_snr(parent, sol)

Per-scan max detection SNR (and χ) from the solver diagnostics. Provided by
`GustavoMakieExt`.
"""
function plot_fringe_snr end

export FringeSearch, FringeDetection, baseline_fringe_search
export Stationization, StationSolution, stationize_scan, station_closure_residuals
export AdhocPhasing, AdhocSolution, solve_adhoc_phasing
export solve_fringes
export fringe_snr_table, print_fringe_snr_table, fringe_solution_summary
export fringe_gain_spectrum, fringe_gain_time_series
export plot_fringe_spectrum, plot_fringe_phases, plot_fringe_snr

end
