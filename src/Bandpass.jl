module Bandpass

using FITSFiles
using CairoMakie
using DimensionalData
using StructArrays
using Statistics
using LinearAlgebra
using LinearSolve
using Printf
using Random: AbstractRNG, default_rng, randn
using ColorSchemes

include("Bandpass/DataIO.jl")
include("Bandpass/StationModels.jl")
include("Bandpass/Setup.jl")
include("Bandpass/Solver.jl")
include("Bandpass/Corrections.jl")
include("Bandpass/Diagnostics.jl")

export UVData, load_uvfits, decode_baseline, assign_scans, scan_average
export baseline_visibilities, baseline_weights, wrap_gain_solutions, wrap_xy_correction
export feed_pair_label, polarization_label, polarization_feeds, parallel_hand_indices, cross_hand_indices
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
export AbstractBandpassInitializer, RatioBandpassInitializer, RandomBandpassInitializer
export AbstractBandpassRefinement, BandpassALS
export prepare_bandpass_solver, initialize_bandpass_state, refine_bandpass!, finalize_bandpass_state, bandpass_state_objective, bandpass_fit_stats
export bandpass_residual_stats, print_bandpass_residual_stats
export solve_bandpass
export apply_bandpass, export_uvfits, default_output_path
export coherence_loss_table, print_coherence_loss_table, choose_diagnostic_baseline
export plot_baseline_phases, plot_stability, plot_gain_solutions, plot_baseline_bandpass_residuals
export parallel_hand_support_summary, site_parallel_hand_support, print_parallel_hand_support
export site_coherence_rows, print_site_coherence_rows

end
