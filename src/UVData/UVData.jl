module UVData

using DimensionalData
using DimensionalData:
    AbstractDimArray, AbstractDimStack, AbstractDimTree, DimArray, DimStack, DimTree,
    DataDict, TreeDict, TupleDict, At, hasdim, dims, lookup, name2dim
import DimensionalData: metadata, branches
using OrderedCollections: OrderedDict
using OhMyThreads: SerialScheduler
using Statistics: mean, median
using Printf: @sprintf
using Dates
using AstroLib: ct2lst
import XRadio

include("dimensions.jl")
include("io.jl")
include("apriori.jl")
include("utilities.jl")
include("msv4_schema.jl")
include("measurementset.jl")
include("autocorrelations.jl")

export GUSTAVO_VISIBILITY_SCHEMA
export load_uvfits, write_uvfits
export check_layer_axes
export feed_pairs, normalize_by_autocorrelations!
export apriori_calibrate!, TsysPlacement, ScanMean, LinearInTime, NearestInTime

end
