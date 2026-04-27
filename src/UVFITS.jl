module UVFITS

using FITSFiles
using FITSFiles: HDU, Random, Bintable, Card
using StructArrays
using DimensionalData
using DimensionalData: AbstractDimArray, DimArray, At, hasdim, dims, name2dim

include("UVFITS/DataStructures.jl")
include("UVFITS/DataIO.jl")

export Antenna, AntennaTable, ArrayConfig, ObsMetadata, FrequencySetup
export BaselineIndex, UVData, with_visibilities
export load_uvfits, write_uvfits, default_output_path
export decode_baseline, assign_scans, scan_average
export scan_time_centers, band_center_frequency, centered_channel_freqs
export baseline_sites, baseline_number
export stokes_feed_pair, polarization_label, feed_type, same_feed_type
export antenna_names, nbaselines, nscans, nchannels, npols, nintegrations
export Pol, IF, Scan, Ant

end
