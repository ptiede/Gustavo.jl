module UVData

using StructArrays
using DimensionalData
using DimensionalData:
    AbstractDimArray, AbstractDimStack, AbstractDimTree, DimArray, DimStack, DimTree,
    DataDict, TreeDict, TupleDict, At, hasdim, dims, lookup, name2dim
import DimensionalData: metadata, branches
using OrderedCollections: OrderedDict
using OhMyThreads: SerialScheduler
using PolarizedTypes: CirBasis, LinBasis, XPol, YPol, RPol, LPol
using Statistics: mean, median
using Printf: @sprintf
using Dates
using AstroLib: ct2lst
import XRadio
using XRadio: AbstractMount, Mount, MountAltAz, MountEquatorial, MountNasmythR, MountNasmythL,
    MountBWGR, MountBWGL, MountXY, MountOrbiting, MountOther

include("dimensions.jl")
include("antenna.jl")
include("baselineidx.jl")
include("frequencyband.jl")
include("metadata.jl")
include("UVSet/UVSet.jl")
include("io.jl")
include("coherence.jl")
include("antab.jl")
include("apriori.jl")
include("utilities.jl")
include("msv4_schema.jl")
include("xradio_bridge.jl")
include("measurementset.jl")
include("processingset.jl")
include("autocorrelations.jl")

export Antenna, AntennaTable, ObsArrayMetadata, FrequencySetup, AbstractFrequencySetup, UVMetadata
export antennas, union_antennas, union_pol_products
export freq_setup, union_frequency_axis, channel_freqs, ref_freq, ch_widths, total_bandwidths, sidebands, setup_name
export AbstractMount, Mount, MountAltAz, MountEquatorial, MountNasmythR, MountNasmythL,
    MountBWGR, MountBWGL, MountXY, MountOrbiting, MountOther
export BaselineIndex, UVSet
export leaves, pol_products
export frequencies, timestamps
export baselines
export pol_index, pol_at, baseline_index
export obs_time
export uvset_to_processingset, GUSTAVO_VISIBILITY_SCHEMA
export source_name, scan_name, primary_scan_name, scan_intents, sub_scan_name
export load_uvfits
export check_layer_axes
export feed_pairs, normalize_by_autocorrelations, normalize_by_autocorrelations!
export AntabCalibration, AntabStation, AntabGainCurve, AntabTsysSeries
export load_antab, load_fitsidi_apriori, tsys_at, elevation_gain, stations
export AprioriFluxGains, apriori_gains
export CoherenceReport, CoherenceCurve, coherence_headline
export print_coherence_report, plot_coherence, plot_coherence_matrix
export spw_center_frequency, centered_channel_freqs
export nchannels

end
