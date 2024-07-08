export Dense, NetworkGene, Network, Weights, WeightCache, GenotypeCache, Model
abstract type AbstractInitializer <: Function end
abstract type AbstractWeights end
abstract type AbstractLayer <: AbstractGenotype end
abstract type AbstractMutation end

include("./structs.jl")
include("./traverse.jl")
include("./utils.jl")
include("./constructors.jl")
include("./develop.jl")
include("./mutate.jl")
include("./distributed.jl")
