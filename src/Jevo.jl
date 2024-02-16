module Jevo

using Random
using StableRNGs
using LRUCache
# using Flux
# using LinearAlgebra
# using LowRankApprox  # for approximate SVD

include("./abstracts.jl")
include("./utils.jl")
include("./creators/creator.jl")
include("./individuals/individual.jl")
include("./populations/populations.jl")
include("./state.jl")

include("./datatypes/counters.jl")

include("./genotypes/numbersgame.jl")
include("./genotypes/nn.jl")

include("./phenotypes/phenotype.jl")
include("./phenotypes/numbersgame.jl")

include("./datatypes/match.jl")
include("./environments/numbersgame.jl")

include("./operators/matchmaker/all_vs_all.jl")
include("./operators/retrievers/retrievers.jl")
include("./operators/operator.jl")
include("./operators/mutators/mutators.jl")


include("run.jl")
end
