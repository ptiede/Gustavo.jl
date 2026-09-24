export Polarization, Frequency, Ant, BaselineID, Ti, UVW, Feed, Scan

using DimensionalData: @dim, TimeDim, Ti
# The frequency, baseline and polarization axes are MSv4's, not Gustavo's own:
# `@dim` defines a method on `DimensionalData.name2dim`, so two packages
# declaring a dimension of the same name overwrite each other and neither can
# precompile.
using XRadio: BaselineID, Frequency, Polarization
@dim Ant "Antenna Index"
@dim UVW "UVW "
@dim Feed "Feed (receptor index)"
@dim Scan "Scan index"
