export Polarization, Frequency, AntennaName, BaselineID, Ti, UVW, Feed, FeedNode, Scan, AntennaPair, FeedPair

using DimensionalData: @dim, TimeDim, Ti
# The frequency, baseline, polarization and antenna axes are MSv4's, not Gustavo's own:
# `@dim` defines a method on `DimensionalData.name2dim`, so two packages
# declaring a dimension of the same name overwrite each other and neither can
# precompile.
using XRadio: BaselineID, Frequency, Polarization, AntennaName
@dim UVW "UVW "
@dim Feed "Feed (receptor index)"
@dim FeedNode "Feed node (one phase unknown, shared by the feeds a tying maps to it)"
@dim Scan "Scan index"
@dim AntennaPair "Antenna pair (antenna names)"
@dim FeedPair "Feed pair (receptor indices)"

# The `(Frequency, Ti)` plane of layer `L` at baseline `bi` and product `p`, in
# that axis order whatever order `L` is stored in.
function _cell_plane(L, bi, p)
    v = view(L, BaselineID(bi), Polarization(p))
    return PermutedDimsArray(parent(v), DimensionalData.dimnum(v, (Frequency, Ti)))
end

# Layer `L` as a plain array in MSv4's storage order
# `(Polarization, Frequency, BaselineID, Ti)`, whatever order `L` is stored in.
_storage_order(L) = PermutedDimsArray(
    parent(L), DimensionalData.dimnum(L, (Polarization, Frequency, BaselineID, Ti)),
)

# `A`, a plain array laid out `(Frequency, Ti, BaselineID, Polarization)`,
# viewed lazily in the axis order of the visibility array `ref`.
_in_axis_order(ref, A::AbstractArray{<:Any, 4}) = PermutedDimsArray(
    A, invperm(DimensionalData.dimnum(ref, (Frequency, Ti, BaselineID, Polarization))),
)
