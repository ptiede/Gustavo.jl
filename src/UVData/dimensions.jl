export Polarization, Frequency, Ant, BaselineID, Ti, UVW, Feed, Scan, StationPair, FeedPair

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
@dim StationPair "Station pair (antenna names)"
@dim FeedPair "Feed pair (receptor indices)"

# The `(Frequency, Ti)` plane of layer `L` at baseline `bi` and product `p`, in
# that axis order whatever order `L` is stored in.
function _cell_plane(L, bi, p)
    v = view(L, BaselineID(bi), Polarization(p))
    return PermutedDimsArray(parent(v), DimensionalData.dimnum(v, (Frequency, Ti)))
end

# `A`, a plain array laid out `(Frequency, Ti, BaselineID, Polarization)`,
# viewed lazily in the axis order of the visibility array `ref`.
_in_axis_order(ref, A::AbstractArray{<:Any, 4}) = PermutedDimsArray(
    A, invperm(DimensionalData.dimnum(ref, (Frequency, Ti, BaselineID, Polarization))),
)
