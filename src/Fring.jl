"""
    Fring

VLBI fringe fitting on top of the unified `Gustavo.Calibration` model framework
and the Gustavo visibility model (EHT-HOPS-inspired, Blackburn et al.
2019, but with globally-closing per-feed solutions): per-baseline FFT
delay/rate search, stationization, bandpass and adhoc-phase stages, and the
diagnostics over them.
"""
module Fring

using OhMyThreads: tforeach, tmap, DynamicScheduler, SerialScheduler, TaskLocalValue
using ..Gustavo: Frequency, BaselineID, Feed, FeedNode, Polarization, Scan, AntennaPair, FeedPair, AntennaName,
    feed_pairs, check_layer_axes, _cell_plane
using ..Calibration
import XRadio
using OrderedCollections: OrderedDict
using DensityInterface: logdensityof
using Distributions: LogNormal
using ..Calibration: ComponentPlan, GeometryWindow,
    _flatten_components, _component_leaf, _feed_node, nfeed_blocks, _block_index, component_is_per_scan,
    _segment_lookup, _frequency_segment_lookup, _time_segment_lookup,
    _freq_group_ranges, _init_moments, _call_string, _gauge_system, _gauge_for, _leaf_paths
# Numeric kernels the model layer keeps off its public surface.
using ..Calibration: weighted_regularized_least_squares, ConstrainedWLS, FactoredWLS,
    unwrap_phase_track, phase_unwrap_ambiguity, connected_components
using FFTW: fft, fftfreq, plan_fft, ESTIMATE
import DimensionalData
using DimensionalData: lookup, dims, dimnum, Ti, At, DimArray, DimStack, AbstractDimArray, AbstractDimStack
using Statistics: median, mean
using LinearAlgebra
using StaticArrays: SMatrix, SVector

include("Fring/search.jl")
include("Fring/stationize.jl")
include("Fring/statespace.jl")
# Fitting one track's parameter blocks under a component prior, independent of
# any solver stage.
include("Fring/prior_fits.jl")
include("Fring/weighted_sums.jl")
include("Fring/adhoc.jl")
# The composable-pipeline engine: the solver capability checks, the
# search over one streamed scan group, the model/plan routers, and the three
# carved-out stage implementations the steps call into.
include("Fring/capability.jl")
include("Fring/scan_search.jl")
include("Fring/search_stage.jl")
include("Fring/model_plans.jl")
include("Fring/bandpass_stage.jl")
include("Fring/coherence.jl")
include("Fring/diagnostics.jl")

# ── Plot stubs — implemented by `GustavoMakieExt`. Load Makie or CairoMakie
# to enable plotting.
"""
    plot_fringe_search(m::FringeSearchMap; zoom)
    plot_fringe_search(parent, m; zoom)

HOPS-style fringe-search diagnostic for one baseline of one scan — the plot for
judging a suspected false fringe, from [`fringe_search_map`](@ref) or
[`baseline_fringe_map`](@ref). Draws the delay–rate matched-filter SNR
surface with delay/rate cross-sections through the peak, titled with the map's
`refdims` (scan, baseline, feed pair), the refined detection (delay, rate,
SNR) and its false-alarm probability. A real
fringe is a single sharp peak far above the sidelobe forest with `pfa ≪ 1`; a
false fringe barely clears the forest (`pfa` not small) and shows several
comparable-height peaks. Provided by `GustavoMakieExt`.

`zoom` sets the view: a number centers both axes on the peak over that many
main-lobe widths (default 24), and `false` shows the whole searched window. The
main lobe is a few grid cells wide against a window sized for the clock search,
so the unzoomed plane resolves the alias structure but not the peak itself. `pfa` and the SNR normalization come
from the full plane either way.
"""
function plot_fringe_search end

"""
    plot_coherence(coh; nlabel = 0)
    plot_coherence(parent, coh; nlabel = 0)

Coherence factor η against time-averaging interval and frequency-averaging
width for one scan and feed pair of a [`coherence_report`](@ref) result,
selected first (`coh[Scan = 1, FeedPair = At((1, 1))]`): log-scaled panels with
a faint trace per antenna pair and the pooled η bold. `nlabel` colors and
labels the `nlabel` antenna pairs with the lowest η at the longest time
interval. Provided by `GustavoMakieExt`.
"""
function plot_coherence end

export FringeSearch, baseline_fringe_search, fringe_plane
export AbstractSearchAlgorithm, FullGrid, HierarchicalMBD
export FringeSearchMap, FringeDelay, FringeRate, baseline_fringe_map, fringe_search_map, fringe_pfa, fringe_snr_cut
export Stationization, station_closure_residuals
export AbstractRobustLoss, LeastSquares, SoftL1, Huber, Cauchy
export AbstractAdhocSmoother, AdhocOptions, PerTrackAdhocSmoother, JointKalmanSmoother
export solve_adhoc_phasing, default_adhoc_terms, default_adhoc_prior
export default_bandpass_terms
export AbstractBandpassSmoother, PerTrackSmoother, JointSmoother
export validate_bandpass_groups, solve_bandpass!, bandpass_track_report, bandpass_blocks,
    bandpass_level_blocks
export fringe_snr_table, fringe_detections, cat_scans
export fringe_station_solutions
export baseline_spectra, freq_group_coherence, fringe_freq_groups, FreqGroup, Triangle
export fringe_station_flags
export baseline_delays, delay_closure
export can_fit, validate_model
export BandGroups, default_fringe_terms
export search_scan
export plot_fringe_search, plot_coherence
export coherence_report, AveragingTime, AveragingBandwidth

end
