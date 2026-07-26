module Bandpass

using ..UVData
using ..UVData:
    Integration, Pol, Frequency, UVW, Baseline, Ant, UVSet,
    scan_key, partition_key, sanitize_source,
    TimeAverage, AbstractPartitionReducer,
    apply, leaves, sources, scan_ids,
    select_source, select_scan, select_partition, merge_uvsets,
    baselines, record_order, extra_columns, obs_time,
    pol_products, PolTypes
using ..Calibration: correlation_feed_pair, _feed_index, is_parallel_hand,
    same_feed_label, parallel_hand_indices, cross_hand_indices,
    build_parallel_hand_mask, design_matrices,
    weighted_phase_mean, weighted_complex_correction,
    _row_scale, _promote_lsq,
    weighted_least_squares, weighted_regularized_least_squares,
    weighted_constrained_least_squares,
    unwrap_phase_track, phase_relative_to_ref
# Unified gain-model framework (the bandpass model layer is now built on this)
using ..Calibration:
    AbstractGainTerm, ConstantTerm, Delay, PolynomialFreq, PerChannel,
    AbstractTimeSegmentation, GlobalTime, PerScan,
    AbstractFrequencySegmentation, GlobalFrequency, PerSpectralWindow, ChannelBlocks,
    GainComponent, TiedComponent, StationGainModel,
    AbstractFeedTying, PerFeed, SharedFeeds, FeedComponent,
    phase_components, logamp_components,
    validate_station_gain_model, is_per_scan, component_is_per_scan,
    station_model_summary, component_label
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
include("Bandpass/Models.jl")
include("Bandpass/Setup.jl")
include("Bandpass/Solver.jl")
include("Bandpass/Corrections.jl")

export load_uvfits, write_uvfits, scan_average
export baseline_visibilities, baseline_weights, wrap_gain_solutions
export polarization_feeds, parallel_hand_indices, cross_hand_indices
export correlation_feed_pair, is_parallel_hand
# Bandpass model builders (produce Calibration types; see Bandpass/Models.jl)
export PerChannelBandpassModel, FlatBandpassModel, DelayBandpassModel, PolynomialBandpassModel
export SegmentedBandpassModel, CompositeBandpassModel
export GlobalTimeSegmentation, PerScanTimeSegmentation
export GlobalFrequencySegmentation, BlockFrequencySegmentation
export BandpassSpec, FeedBandpassModel, StationBandpassModel
export spec_components
export phase_is_per_scan, amplitude_is_per_scan
export design_matrices, build_station_models, station_model_summary, choose_phase_reference
export BandpassSolverSetup, BandpassSolverState
export AbstractBandpassGauge, ZeroMeanBandpassGauge, ReferenceAntennaBandpassGauge
export AbstractBandpassInitializer, RatioBandpassInitializer, RandomBandpassInitializer
export AbstractBandpassRefinement, BandpassALS
export prepare_bandpass_solver, initialize_bandpass_state, refine_bandpass!, finalize_bandpass_state, bandpass_state_objective, bandpass_fit_stats
export bandpass_residual_stats, print_bandpass_residual_stats
export solve_bandpass
export apply_bandpass, default_output_path
export apply_calibration

end
