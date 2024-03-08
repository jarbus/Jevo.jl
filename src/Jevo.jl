module Jevo

using Random
using StableRNGs
using LRUCache
using StatsBase

# Neural Nets
using CUDA
using Flux
# For checkpointing and handling exceptions
using Serialization
# For logging
using Dates
using HDF5
using Logging
using LoggingExtras
using FileWatching
# For MNIST
using MLDatasets
using OneHotArrays

import Base: show
# using LinearAlgebra
# using LowRankApprox  # for approximate SVD

include("./abstracts.jl")
include("./macro.jl")
include("./utils.jl")
include("./creators/creator.jl")
include("./datatypes/datatypes.jl")
include("./logger.jl")
include("./individuals/individual.jl")
include("./populations/populations.jl")
include("./state.jl")
include("./organisms/organisms.jl")
include("./environments/environments.jl")
include("./operators/operators.jl")

end
