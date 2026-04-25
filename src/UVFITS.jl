module UVFITS

using FITSFiles
using FITSFiles: HDU, Random, Bintable, Card
using StructArrays

include("UVFITS/DataStructures.jl")
include("UVFITS/DataIO.jl")

export Antenna, AntennaTable, ObsMetadata
export UVData, with_visibilities
export load_uvfits, write_uvfits, default_output_path
export decode_baseline, assign_scans, scan_average
export scan_time_centers, band_center_frequency, centered_channel_freqs
export baseline_sites, baseline_number
export stokes_feed_pair, feed_pair_label, polarization_label
export Pol, IF, Scan, Ant

end
