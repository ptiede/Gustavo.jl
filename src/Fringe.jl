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
using FFTW: fft, fftfreq, plan_fft, MEASURE
import FFTW
using Statistics: median, mean
using LinearAlgebra
using Printf: @sprintf
using OhMyThreads: tmap, TaskLocalValue

include("Fringe/search.jl")
include("Fringe/stationize.jl")
include("Fringe/statespace.jl")
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

"""
    plot_baseline_fringes(uvset, sol; kind, pol, baselines, scan_index, show)
    plot_baseline_fringes(data::BaselineFringeData; ...)
    plot_baseline_fringes(parent, data; ...)

Per-baseline before/after fringe-fit check for one scan: a grid of panels (one per
baseline) overlaying the coherent visibility BEFORE and AFTER applying `sol`.
`kind = :freq` plots phase (or amplitude) vs frequency — a group delay shows as a
slope that flattens after a good fit; `kind = :time` plots vs time — a fringe rate
shows as a slope that flattens. `show = :phase` (default) or `:amp`. `pol` selects
the correlation product (default `:parallel`); `baselines` selects which to draw.
Provided by `GustavoMakieExt`.
"""
function plot_baseline_fringes end

export FringeSearch, FringeDetection, baseline_fringe_search
export Stationization, StationSolution, stationize_scan, station_closure_residuals
export AbstractAdhocSmoother, PerTrackAdhocSmoother, SavitzkyGolaySmoother, PenalizedSmoother
export OUSmoother, JointOUSmoother, NoSmoothing, AdhocSolution, solve_adhoc_phasing
export solve_fringes, solve_and_reduce_fringes
export AbstractBandpassSmoother, FreeBandpass, PolynomialBandpass, PenalizedBandpass
export fringe_snr_table, print_fringe_snr_table, fringe_solution_summary
export fringe_gain_spectrum, fringe_gain_time_series
export BaselineFringeData, baseline_fringe_data, baseline_pol_index, fringe_scan_groups
export delay_closure, print_delay_closure
export plot_fringe_spectrum, plot_fringe_phases, plot_fringe_snr, plot_baseline_fringes

end
