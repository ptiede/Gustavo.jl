module UVData

using StructArrays
using DimensionalData
using DimensionalData:
    AbstractDimArray, AbstractDimStack, AbstractDimTree, DimArray, DimStack, DimTree,
    DataDict, TreeDict, TupleDict, At, hasdim, dims, lookup, name2dim
import DimensionalData: metadata, branches
using OrderedCollections: OrderedDict
using PolarizedTypes: CirBasis, LinBasis, XPol, YPol, RPol, LPol
using Statistics: mean, median
using Printf: @sprintf
using Dates
using AstroLib: ct2lst

include("dimensions.jl")
include("antenna.jl")
include("feeds.jl")
include("baselineidx.jl")
include("frequencyband.jl")
include("metadata.jl")
include("UVSet/UVSet.jl")
include("io.jl")
include("dataset.jl")
include("coherence.jl")
include("diagnostics.jl")
include("antab.jl")
include("apriori.jl")
include("utilities.jl")

export Antenna, AntennaTable, ObsArrayMetadata, FrequencySetup, AbstractFrequencySetup, UVMetadata
export antennas, union_antennas, union_pol_products
export freq_setup, union_frequency_axis, channel_freqs, ref_freq, ch_widths, total_bandwidths, sidebands, setup_name
export Mount, MountAltAz, MountEquatorial, MountNaismithR, MountNaismithL
export BaselineIndex, UVSet, with_visibilities, scan_key
export apply, mapleaves, flatmap, leaves, sources, scan_ids, partitions, pol_products
export set_bunit, with_bunit
export baseline, baselines_per_scan, baselines
export pol_index, pol_at, baseline_index
export obs_time
export scan_name, primary_scan_name, scan_intents, sub_scan_name, scan_window, participating_antennas
export select_source, select_scan, select_station, select_baseline, select_partition
export merge_uvsets, time_window
export load_uvfits, write_uvfits, default_output_path
export load_fitsidi, write_fitsidi
export is_lazy, materialize_leaf, materialize, materialize_group
export apply_calibration
export correlation_feed_pair, is_parallel_hand, same_feed_label
export parallel_hand_indices, cross_hand_indices, build_parallel_hand_mask
export AntabCalibration, AntabStation, AntabGainCurve, AntabTsysSeries
export load_antab, load_fitsidi_apriori, tsys_at, elevation_gain, stations
export AprioriFluxGains, apriori_flux_gains
export primary_cards, register_primary_cards!
export TimeAverage, scan_average
export FrequencyAverage, frequency_average, TimeBinAverage, time_bin_average
export BandEdgeFlag, flag_band_edges
export combine_spw
export CoherenceReport, CoherenceCurve, coherence_report, coherence_headline
export print_coherence_report, plot_coherence, plot_coherence_matrix
export phase_relative_to_ref
export plot_stability, plot_baseline_phases, plot_gain_solutions
export scan_time_centers, band_center_frequency, centered_channel_freqs
export baseline_sites, baseline_number
export antenna_names, nbaselines, nscans, nchannels, npols, nintegrations
export Pol, Frequency, Ant
export pol_products

end
