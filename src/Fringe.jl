"""
    Fringe

VLBI fringe fitting on top of the unified `Gustavo.Calibration` model framework,
the `Gustavo.UVData` visibility model, and the `Gustavo.Streaming` scan-group
layer it runs its passes over (EHT-HOPS-inspired, Blackburn et al. 2019, but
with globally-closing per-feed solutions): per-baseline FFT delay/rate search,
stationization, bandpass and adhoc-phase stages, and the diagnostics over them.

The streaming vocabulary (`ScanStream`, the transform and
selection types) is re-exported from `Gustavo.Streaming`, so a caller driving
the fringe engine reaches it without a second `using`.
"""
module Fringe

using OhMyThreads: tforeach, DynamicScheduler, SerialScheduler, TaskLocalValue
using ..UVData
using ..UVData: Frequency, Baseline, Feed, Pol, Scan
using ..Calibration
using ..Calibration: ComponentPlan, GeometryWindow, _dispersion_enabled, _is_dispersion,
    _flatten_components, _component_leaf, _feed_node, _block_index, _composed_gains
# `SingleBandDelay` and the `FringeModel` term-list compilation are methods of
# the model layer's element-compilation generic.
import ..Calibration: model_components
using ..Streaming
# `CoverageTopup` (bandpass_stage.jl) is another `AbstractScanSelection`, so its
# resolver must be a METHOD of the streaming layer's generic — defining
# `select_scans` under a bare `using` would mint a second function of the same
# name and leave the two ambiguous wherever both modules are in scope.
import ..Streaming: select_scans
using FFTW: fft, fftfreq, plan_fft, ESTIMATE
import DimensionalData
using DimensionalData: lookup, dims, Ti, DimArray, AbstractDimStack
using Statistics: median, mean
using LinearAlgebra
using Printf: @sprintf

include("Fringe/search.jl")
include("Fringe/stationize.jl")
include("Fringe/statespace.jl")
# Per-observable frequency-shape specs and their per-track fit — pure functions
# over one (station, feed, spw) track, independent of any solver stage.
include("Fringe/shapes.jl")
include("Fringe/adhoc.jl")
include("Fringe/phasecal.jl")
# The composable-pipeline engine: the pluggable fringe-estimator strategy, the
# search over one streamed scan group, the model/plan routers, and the three
# carved-out stage implementations the step visitors call into.
include("Fringe/estimators.jl")
include("Fringe/scan_search.jl")
include("Fringe/search_stage.jl")
include("Fringe/model_plans.jl")
include("Fringe/bandpass_stage.jl")
include("Fringe/refine_search.jl")
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

Per-scan max detection SNR from the solver diagnostics. Provided by
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
`freqgroup = k` restricts the view to the k-th frequency group ([`fringe_freq_groups`](@ref)):
`:freq` panels show only that group's channels on the real frequency axis, `:time`
panels average over only that group — the readable view on wide multi-group data
(VGOS), where the all-group view hides which group misfits.
Provided by `GustavoMakieExt`.
"""
function plot_baseline_fringes end

export FringeSearch, baseline_fringe_search
export AbstractSearchAlgorithm, FullGrid, HierarchicalMBD
export FringeSearchMap, baseline_fringe_map, fringe_pfa, fringe_snr_cut
export PhaseCalTable, load_fitsidi_phasecal, phasecal_solution, tone_channel_mask
export Stationization, station_closure_residuals
export AbstractRobustLoss, LeastSquares, SoftL1, Huber, Cauchy
export AbstractAdhocSmoother, PerTrackAdhocSmoother, SavitzkyGolaySmoother, PenalizedSmoother
export OUSmoother, JointOUSmoother, NoSmoothing, solve_adhoc_phasing
export station_weight_scale
export AbstractShapeSpec, FreeShape, PolynomialShape, WhittakerShape, ARShape, fit_track
export fit_track_group
export BandpassModel
export AbstractBandpassSmoother, PerTrackSmoother, JointSmoother
export bandpass_derotate, validate_bandpass, solve_bandpass!, bandpass_track_report
export fringe_snr_table, print_fringe_snr_table, fringe_solution_summary, print_solve_timing
export fringe_gain_spectrum, fringe_bandpass_spectrum, fringe_gain_time_series, fringe_station_solutions
export BaselineFringeData, baseline_fringe_data, baseline_pol_index, fringe_scan_groups
export fringe_freq_group_stats, fringe_freq_groups
export BaselineFringeMap, fringe_search_map, suspect_fringes, fringe_station_flags
export delay_closure, print_delay_closure
export AbstractDataTransform, apply_transform!, apply_transform
export ApplySolution, StationWeightScale, FlagChannels, CalFunction
export AbstractScanSelection, AllScans, SourceScans, ScanIndices, ScanWhere
export select_scans
export AbstractFringeEstimator, estimate_scan!, finish_estimate!, estimator_info
export can_fit, validate_model
export MatchedFilter, FringeModel, SingleBandDelay, BandGroups, default_fringe_terms
# `DispersionModel` is `Calibration`'s (the propagation model beside the
# `Dispersion` term it configures); re-exported so a caller driving the fringe
# engine names it without a second `using`.
export DispersionModel
export AbstractLeafGrouping, ByScan, BySpw, ByKey
export ExecutionConfig, ProgressLogger, outer_executor, inner_executor
export ScanStream, scan_stream, ScanGroupSpec, select_groups
export materialize_cube, materialize_leaves
export search_scan
export map_groups, foreach_group
export plot_fringe_spectrum, plot_fringe_phases, plot_fringe_snr, plot_baseline_fringes
export plot_fringe_search

end
