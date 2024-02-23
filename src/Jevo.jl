module Jevo

using Random
using StableRNGs
using LRUCache
using StatsBase
# For checkpointing and handling exceptions
using Serialization
# For logging
using Dates
using HDF5
using Logging
using LoggingExtras
using FileWatching

import Base: show
# using Flux
# using LinearAlgebra
# using LowRankApprox  # for approximate SVD

include("./abstracts.jl")
include("./utils.jl")
include("./datatypes/datatypes.jl")
include("./logger.jl")
include("./creators/creator.jl")
include("./individuals/individual.jl")
include("./populations/populations.jl")
include("./state.jl")
include("./organisms/organisms.jl")
include("./environments/environments.jl")
include("./operators/operators.jl")

end
