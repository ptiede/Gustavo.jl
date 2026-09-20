export Pol, Frequency, Ant, Baseline, Ti, UVW, Feed, Scan

using DimensionalData: @dim, TimeDim, Ti
# The frequency axis is MSv4's, not one of Gustavo's own: `@dim` defines a
# method on `DimensionalData.name2dim`, so two packages declaring a dimension
# of the same name overwrite each other and neither can precompile.
using XRadio: Frequency
@dim Pol "Polarization product"
@dim Ant "Antenna Index"
@dim Baseline "Baseline"
@dim UVW "UVW "
@dim Feed "Feed (receptor index)"
@dim Scan "Scan index"
