module UVData

using StructArrays
using DimensionalData
using DimensionalData:
    AbstractDimArray, AbstractDimStack, AbstractDimTree, DimArray, DimStack, DimTree,
    DataDict, TreeDict, TupleDict, At, hasdim, dims, lookup, name2dim
import DimensionalData: metadata, branches
using OrderedCollections: OrderedDict
using PolarizedTypes: CirBasis, LinBasis, XPol, YPol, RPol, LPol

include("dimensions.jl")
include("antenna.jl")
include("baselineidx.jl")
include("frequencyband.jl")
include("metadata.jl")
include("UVSet/UVSet.jl")
include("io.jl")
include("utilities.jl")

export Antenna, AntennaTable, ArrayConfig, ObsArrayMetadata, FrequencySetup, AbstractFrequencySetup, UVMetadata
export freq_setup, union_frequency_axis, channel_freqs, ref_freq, ch_widths, total_bandwidths, sidebands, setup_name
export Mount, MountAltAz, MountEquatorial, MountNaismithR, MountNaismithL
export BaselineIndex, UVSet, with_visibilities, scan_key
export apply, leaves, sources, scan_ids, partitions, pol_products
export scan_name, primary_scan_name, scan_intents, sub_scan_name, scan_window, participating_antennas
export select_source, select_scan, select_station, select_baseline, select_partition
export merge_uvsets, time_window
export load_uvfits, write_uvfits, default_output_path
export primary_cards, register_primary_cards!
export TimeAverage, scan_average
export scan_time_centers, band_center_frequency, centered_channel_freqs
export baseline_sites, baseline_number
export antenna_names, nbaselines, nscans, nchannels, npols, nintegrations
export Pol, IF, Scan, Ant
export pol_products

end
