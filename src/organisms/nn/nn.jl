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

struct SVDInitializer <: AbstractInitializer
    """Initialize a weight matrix by sampling rows/columns from a singular value decomposition of an existing initializer `fn`.
    To be used as part of a mutation, so it does not need to hold an rng, mr, or dims"""
    fn::Function
end

function get_nearest_ancestor(ancestors::Vector{Int}) 
    """Return nearest cached ancestor from list"""
end
function send_delta_genotype(ancestor::Int, descendant::Int, genotype::Network)
    """Send diff of nearest cached ancestor and descendant and caches the result"""
end
