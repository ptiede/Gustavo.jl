module Bandpass

using ..UVData
using ..UVData:
    Integration, Pol, Frequency, UVW, Baseline, Ant, UVSet,
    scan_key, partition_key, sanitize_source,
    TimeAverage, AbstractPartitionReducer,
    apply, leaves, sources, scan_ids,
    select_source, select_scan, select_partition, merge_uvsets,
    scan_idx, scan_id, baselines, record_order, extra_columns, obs_time,
    pol_products, PolTypes
using PolarizedTypes: RPol, LPol, XPol, YPol
using DimensionalData: Ti
using OrderedCollections: OrderedDict
using DimensionalData
using DimensionalData: AbstractDimArray, AbstractDimTree, DimStack, DimTree, hasdim, name2dim, At
using StructArrays
using Statistics
using LinearAlgebra
using LinearSolve
using Printf
using Random: AbstractRNG, default_rng, randn

include("Bandpass/DataIO.jl")
include("Bandpass/StationModels.jl")
include("Bandpass/Setup.jl")
include("Bandpass/Solver.jl")
include("Bandpass/Corrections.jl")
include("Bandpass/Diagnostics.jl")

# ── Plot stubs — implemented by `GustavoMakieExt`. Load Makie or CairoMakie
# to enable plotting.
"""
    plot_stability(data, corr, bl_plot; quantity, pol, relative, comparison_weights)

Per-leaf time-stability scatter for a baseline. Provided by `GustavoMakieExt`.
"""
function plot_stability end

"""
    plot_baseline_phases(data, corr, bl_plot; relative, comparison_weights)

Per-leaf phase-vs-channel scatter for a baseline. Provided by `GustavoMakieExt`.
"""
function plot_baseline_phases end

"""
    plot_gain_solutions(gains, data; quantity, pol, sites, relative)

Grid of solved gain tracks. Provided by `GustavoMakieExt`.
"""
function plot_gain_solutions end

"""
    plot_baseline_bandpass(setup, gains_or_state, bl_plot; pol, normalize_by_source)

Bandpass overlay diagnostics. Provided by `GustavoMakieExt`.
"""
function plot_baseline_bandpass end

"""
    plot_baseline_bandpass_residuals(setup, gains_or_state, bl_plot; pol)

Residual diagnostics. Provided by `GustavoMakieExt`.
"""
function plot_baseline_bandpass_residuals end

"""
    plot_noise_segments!(ax, series, noise, scan_index, scan_wheel, nscan; ...)

Internal Makie helper. Provided by `GustavoMakieExt`.
"""
function plot_noise_segments! end

"""
    annotate_coherence!(ax, stats; fontsize)

Internal Makie helper. Provided by `GustavoMakieExt`.
"""
function annotate_coherence! end

"""
    diagnostic_scan_colormap(nscan)

Categorical scan colormap for plot helpers. Provided by `GustavoMakieExt`.
"""
function diagnostic_scan_colormap end

export load_uvfits, write_uvfits, scan_average
export baseline_visibilities, baseline_weights, wrap_gain_solutions, wrap_xy_correction
export polarization_feeds, parallel_hand_indices, cross_hand_indices
export correlation_feed_pair, is_parallel_hand
export AbstractBandpassModel
export PerChannelBandpassModel, FlatBandpassModel, DelayBandpassModel, PolynomialBandpassModel
export SegmentedBandpassModel, CompositeBandpassModel
export AbstractTimeSegmentation, AbstractFrequencySegmentation
export GlobalTimeSegmentation, PerScanTimeSegmentation
export GlobalFrequencySegmentation, BlockFrequencySegmentation
export BandpassSegmentation, BandpassSpec, FeedBandpassModel, StationBandpassModel
export bandpass, parameter_count
export validate_station_bandpass_model, is_per_scan, phase_is_per_scan, amplitude_is_per_scan
export best_ref_channel, design_matrices, build_station_models, station_model_summary, choose_phase_reference
export BandpassSolverSetup, BandpassSolverState, BandpassSolverResult
export AbstractBandpassGauge, ZeroMeanBandpassGauge, ReferenceAntennaBandpassGauge
export AbstractBandpassInitializer, RatioBandpassInitializer, RandomBandpassInitializer
export AbstractBandpassRefinement, BandpassALS
export prepare_bandpass_solver, initialize_bandpass_state, refine_bandpass!, finalize_bandpass_state, bandpass_state_objective, bandpass_fit_stats
export bandpass_residual_stats, print_bandpass_residual_stats
export solve_bandpass
export apply_bandpass, default_output_path
export coherence_loss_table, print_coherence_loss_table, choose_diagnostic_baseline
export plot_baseline_phases, plot_stability, plot_gain_solutions, plot_baseline_bandpass, plot_baseline_bandpass_residuals
export parallel_hand_support_summary, site_parallel_hand_support, print_parallel_hand_support
export site_coherence_rows, print_site_coherence_rows

end
