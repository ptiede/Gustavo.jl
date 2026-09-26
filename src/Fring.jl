"""
    Fring

VLBI fringe fitting on top of the unified `Gustavo.Calibration` model framework
and the `Gustavo.UVData` visibility model (EHT-HOPS-inspired, Blackburn et al.
2019, but with globally-closing per-feed solutions): per-baseline FFT
delay/rate search, stationization, bandpass and adhoc-phase stages, and the
diagnostics over them.
"""
module Fring

using OhMyThreads: tforeach, tmap, DynamicScheduler, SerialScheduler, TaskLocalValue
using ..UVData
using ..UVData: Frequency, BaselineID, Feed, FeedNode, Polarization, Scan, StationPair, FeedPair
using ..Calibration
using ..Calibration: _epoch_atol
import XRadio
using ..Calibration: ComponentPlan, GeometryWindow,
    _flatten_components, _component_leaf, _feed_node, _block_index, component_is_per_scan,
    _segment_lookup, _frequency_segment_lookup, _time_segment_lookup,
    _freq_group_ranges
# Numeric kernels the model layer keeps off its public surface.
using ..Calibration: weighted_regularized_least_squares, ConstrainedWLS, FactoredWLS,
    unwrap_phase_track, phase_unwrap_ambiguity, connected_components, savitzky_golay_smooth
using FFTW: fft, fftfreq, plan_fft, ESTIMATE
import DimensionalData
using DimensionalData: lookup, dims, dimnum, Ti, At, DimArray, DimStack, AbstractDimStack
using Statistics: median, mean
using LinearAlgebra
using Printf: @sprintf

include("Fring/search.jl")
include("Fring/stationize.jl")
include("Fring/statespace.jl")
# Per-observable frequency-shape specs and their per-track fit — pure functions
# over one (station, feed, spw) track, independent of any solver stage.
include("Fring/shapes.jl")
include("Fring/weighted_sums.jl")
include("Fring/adhoc.jl")
include("Fring/phasecal.jl")
# The composable-pipeline engine: the solver capability checks, the
# search over one streamed scan group, the model/plan routers, and the three
# carved-out stage implementations the steps call into.
include("Fring/capability.jl")
include("Fring/scan_search.jl")
include("Fring/search_stage.jl")
include("Fring/model_plans.jl")
include("Fring/bandpass_stage.jl")
include("Fring/diagnostics.jl")

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
    plot_fringe_search(uvset, sol; scan_index, baseline, pol, search, zoom)
    plot_fringe_search(m::BaselineFringeMap; zoom)
    plot_fringe_search(parent, m; zoom)

HOPS-style fringe-search diagnostic for one baseline of one scan — the plot for
judging a suspected false fringe. Draws the delay–rate matched-filter SNR
surface with delay/rate cross-sections through the peak, and annotates the
refined detection (delay, rate, SNR) and its false-alarm probability. A real
fringe is a single sharp peak far above the sidelobe forest with `pfa ≪ 1`; a
false fringe barely clears the forest (`pfa` not small) and shows several
comparable-height peaks. Selectors as [`fringe_search_map`](@ref). Provided by
`GustavoMakieExt`.

`zoom` sets the view: `true` (default) centres both axes on the peak over a span
of a dozen main-lobe widths, a `Real` gives that span in main-lobe widths, and
`false` shows the whole searched window. The main lobe is a few grid cells wide
against a window sized for the clock search, so the unzoomed plane resolves the
alias structure but not the peak itself. `pfa` and the SNR normalization come
from the full plane either way.
"""
function plot_fringe_search end

"""
    plot_baseline_fringes(uvset, sol; kind, pol, baselines, scan_index, show)
    plot_baseline_fringes(data::BaselineFringeData; ...)
    plot_baseline_fringes(parent, data; ...)

Per-baseline before/after fringe-fit check for one scan: a grid of panels (one per
baseline) overlaying the coherent visibility before and after applying `sol`.
`kind = :freq` plots phase (or amplitude) vs frequency — a group delay shows as a
slope that flattens after a good fit; `kind = :time` plots vs time — a fringe rate
shows as a slope that flattens. `show = :phase` (default) or `:amp`. `pol` selects
the correlation product (required); `baselines` selects which to draw.
`freqgroup = k` restricts the view to the k-th frequency group ([`fringe_freq_groups`](@ref)):
`:freq` panels show only that group's channels on the real frequency axis, `:time`
panels average over only that group — the readable view on wide multi-group data
(VGOS), where the all-group view hides which group misfits.
Provided by `GustavoMakieExt`.
"""
function plot_baseline_fringes end

export FringeSearch, baseline_fringe_search, fringe_plane
export AbstractSearchAlgorithm, FullGrid, HierarchicalMBD
export FringeSearchMap, baseline_fringe_map, fringe_pfa, fringe_snr_cut
export PhaseCalTable, load_fitsidi_phasecal, phasecal_solution, tone_channel_mask
export Stationization, station_closure_residuals
export AbstractRobustLoss, LeastSquares, SoftL1, Huber, Cauchy
export AbstractAdhocSmoother, AdhocOptions, SavitzkyGolaySmoother, PenalizedSmoother
export OUSmoother, JointOUSmoother, NoSmoothing, solve_adhoc_phasing, default_adhoc_terms
export AbstractShapeSpec, FreeShape, PolynomialShape, WhittakerShape, ARShape, fit_track
export fit_track_group
export default_bandpass_terms
export AbstractBandpassSmoother, PerTrackSmoother, JointSmoother
export validate_bandpass_groups, solve_bandpass!, bandpass_track_report, bandpass_blocks
export fringe_snr_table, print_fringe_snr_table, fringe_solution_summary, print_solve_timing
export fringe_station_solutions
export BaselineFringeData, baseline_fringe_data, baseline_pol_index, fringe_scan_groups
export fringe_freq_group_stats, fringe_freq_groups
export BaselineFringeMap, fringe_search_map, suspect_fringes, fringe_station_flags
export delay_closure, print_delay_closure
export can_fit, validate_model
export BandGroups, default_fringe_terms
export search_scan
export plot_fringe_spectrum, plot_fringe_phases, plot_fringe_snr, plot_baseline_fringes
export plot_fringe_search

end
