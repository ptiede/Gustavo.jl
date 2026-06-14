"""
    Fringe

VLBI fringe fitting on top of the unified `Gustavo.Calibration` model framework
and the `Gustavo.UVData` visibility model. Stage by stage (EHT-HOPS-inspired,
Blackburn et al. 2019, but with globally-closing per-feed solutions):

- `search.jl`     — per-baseline FFT delay/rate fringe search.
- (stationize / adhoc / pipeline — later phases.)
"""
module Fringe

using ..UVData
using ..Calibration
using FFTW: fft, fftfreq
using Statistics: median
using LinearAlgebra

include("Fringe/search.jl")

export FringeSearch, FringeDetection, baseline_fringe_search

end
