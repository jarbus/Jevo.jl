module Jevo

using Flux
using Random
using LinearAlgebra
using LowRankApprox  # for approximate SVD
using StableRNGs
using LRUCache

include("abstracts.jl")
include("individuals/individual.jl")
include("populations/populations.jl")
include("state.jl")

include("genotypes/nn.jl")

include("operators/retrievers/retrievers.jl")
include("operators/operator.jl")
include("operators/mutators/mutators.jl")


include("run.jl")
end
