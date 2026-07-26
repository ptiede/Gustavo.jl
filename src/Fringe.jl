"""
    Fringe

VLBI fringe fitting on top of the unified `Gustavo.Calibration` model framework
and the `Gustavo.UVData` visibility model. Stage by stage (EHT-HOPS-inspired,
Blackburn et al. 2019, but with globally-closing per-feed solutions):

- `search.jl`     — per-baseline FFT delay/rate fringe search.
- (stationize / adhoc / pipeline — later phases.)
"""
module Fringe

using ..Executors
using ..Executors: exec_foreach
using ..UVData
using ..UVData: Frequency
using ..Calibration
using ..Calibration: ComponentPlan
using FFTW: fft, fftfreq, plan_fft, MEASURE
import FFTW
import DimensionalData
using DimensionalData: lookup, dims, Ti
using Statistics: median, mean
using LinearAlgebra
using Printf: @sprintf

include("Fringe/search.jl")
include("Fringe/stationize.jl")
include("Fringe/statespace.jl")
include("Fringe/adhoc.jl")
include("Fringe/pseudostokes.jl")
include("Fringe/phasecal.jl")
# The composable-pipeline engine: the materialization transform chain, scan
# selections (fit-on-subset), the pluggable fringe-estimator strategy, the
# scan-group streaming layer, the model/plan routers, and the three carved-out
# stage implementations the step visitors call into.
include("Fringe/transforms.jl")
include("Fringe/selections.jl")
include("Fringe/estimators.jl")
include("Fringe/stream.jl")
include("Fringe/search_stage.jl")
include("Fringe/model_plans.jl")
include("Fringe/bandpass_stage.jl")
include("Fringe/refine_stage.jl")
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
    plot_fringe_search(uvset, sol; scan_index, baseline, pol, search)
    plot_fringe_search(m::BaselineFringeMap)
    plot_fringe_search(parent, m)

HOPS-style fringe-search diagnostic for one baseline of one scan — THE plot for
judging a suspected false fringe. Draws the delay–rate matched-filter SNR
surface with delay/rate cross-sections through the peak, and annotates the
refined detection (delay, rate, SNR) and its false-alarm probability. A real
fringe is a single sharp peak far above the sidelobe forest with `pfa ≪ 1`; a
false fringe barely clears the forest (`pfa` not small) and shows several
comparable-height peaks. Selectors as [`fringe_search_map`](@ref). Provided by
`GustavoMakieExt`.
"""
function plot_fringe_search end

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
`band = k` restricts the view to the k-th band group ([`fringe_band_groups`](@ref)):
`:freq` panels show only that group's channels on the real frequency axis, `:time`
panels average over only that group — the readable view on wide multi-group data
(VGOS), where the all-band view hides which group misfits.
Provided by `GustavoMakieExt`.
"""
function plot_baseline_fringes end

export FringeSearch, FringeDetection, baseline_fringe_search
export AbstractSearchAlgorithm, FullGrid, HierarchicalMBD
export FringeSearchMap, baseline_fringe_map, fringe_pfa, fringe_snr_cut
export PhaseCalTable, load_fitsidi_phasecal, phasecal_solution, tone_channel_mask
export Stationization, StationSolution, stationize_scan, station_closure_residuals
export AbstractAdhocSmoother, PerTrackAdhocSmoother, SavitzkyGolaySmoother, PenalizedSmoother
export OUSmoother, JointOUSmoother, NoSmoothing, AdhocSolution, solve_adhoc_phasing
export station_weight_scale
export AbstractBandpassSmoother, FreeBandpass, PolynomialBandpass, PenalizedBandpass
export fringe_snr_table, print_fringe_snr_table, fringe_solution_summary, print_solve_timing
export fringe_gain_spectrum, fringe_bandpass_spectrum, fringe_gain_time_series, fringe_station_solutions
export BaselineFringeData, baseline_fringe_data, baseline_pol_index, fringe_scan_groups
export fringe_band_stats, fringe_band_groups
export BaselineFringeMap, fringe_search_map, suspect_fringes, fringe_station_flags
export delay_closure, print_delay_closure
export AbstractDataTransform, ScanDataView, apply_transform!, apply_transform
export ApplySolution, StationWeightScale, FlagChannels, CalFunction
export AbstractScanSelection, AllScans, SourceScans, BrightestCalibrator, ScanIndices, ScanWhere
export select_scans
export AbstractFringeEstimator, MatchedFilter, FringeModel, CrossFeed
export AbstractLeafGrouping, ByScan, ByBand, ByKey
export ScanStream, scan_stream, ScanGroupSpec, ScanGroup, scan_view, select_groups
export materialize_cube, materialize_leaves
export ScanSearchResult, search_scan
export map_groups, foreach_group
export plot_fringe_spectrum, plot_fringe_phases, plot_fringe_snr, plot_baseline_fringes
export plot_fringe_search

end
