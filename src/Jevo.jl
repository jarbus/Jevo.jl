module Jevo

using Random
using StableRNGs
using LRUCache
using StatsBase
using Distributed

# Neural Nets
using CUDA
using Flux
using Transformers
using Transformers.TextEncoders
# For checkpointing and handling exceptions
using Serialization
# For logging
using Dates
using HDF5
using Logging
using LoggingExtras
using FileWatching
using PhylogeneticTrees
# For MNIST
using OneHotArrays
using Reexport
using XPlot
using ClusterManagers
@reexport using XPlot

import Base: show

# using LinearAlgebra
# using LowRankApprox  # for approximate SVD

include("./abstracts.jl")
include("./utils.jl")
include("./creators/creator.jl")
include("./datatypes/datatypes.jl")
include("./logger.jl")
include("./individuals/individual.jl")
include("./populations/populations.jl")
include("./state.jl")
include("./operators/operators.jl")
include("./phylo.jl")
include("./organisms/organisms.jl")
include("./environments/environments.jl")
end
