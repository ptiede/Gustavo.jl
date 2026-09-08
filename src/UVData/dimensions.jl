export Pol, Frequency, Ant, Baseline, Ti, UVW, Feed, Scan

using DimensionalData: @dim, TimeDim, Ti
@dim Pol "Polarization product"
@dim Frequency "Frequency channel (Hz)"
@dim Ant "Antenna Index"
@dim Baseline "Baseline"
@dim UVW "UVW "
@dim Feed "Feed (receptor index)"
@dim Scan "Scan index"
