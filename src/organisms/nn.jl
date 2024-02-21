"""
    @enum Coupling Strict Loose None

# Values
- `Coupled`: Weights in all sub-layers are mutated using the same initial rng seed
- `SemiCoupled`: Weights in all sub-layers can be mutated independently or together
- `UnCoupled`: Weights in all sub-layers are mutated independently
"""
@enum Coupling Strict Loose None

# Weights
struct NetworkGene <: AbstractMutation
    id::Int
    seed::Int
    mr::Real
    init::AbstractInitializer
end

struct Weights{N} <: AbstractWeights where N <: Int
    """A weight matrix parameterized by a series of mutations"""
    dims::NTuple{N, Int}
    muts::Vector{NetworkGene}
end

struct WeightsCollection <: AbstractWeights
    """Concatenation of multiple weight blocks into a single weight tensor, to adjust subsets of weights independently"""
    weights::Array{AbstractWeights}
end

struct FactorWeight <: AbstractWeights
    """Low-rank factorization of a weight matrix"""
    A::AbstractWeights
    B::AbstractWeights
end

struct Network <: AbstractLayer
    """A collection and a coupling scheme."""
    layers::Vector{<:AbstractLayer}
    coupling::Coupling
end

function tensor(w::AbstractWeights)
    """Converts the genotype of a weight matrix
    (RNGs, MRs, factors) to the phenotype."""
end

function mutate(w::AbstractWeights, rng::AbstractRNG)
    """Adds additional mutations to the genotype of a weight matrix"""
end

function create_network(network::Network)
    """Creates flux network from a genotype"""
end

struct SVDInitializer <: AbstractInitializer
    """Initialize a weight matrix by sampling rows/columns from a singular value decomposition of an existing initializer `fn`.
    To be used as part of a mutation, so it does not need to hold an rng, mr, or dims"""
    fn::Function
end

struct DenseLayer <: AbstractLayer
    weights::AbstractWeights
    bias::AbstractWeights
    σ::Any
end

struct ResidualBlockLayer <: AbstractLayer
    """A block of layers with a skip connection"""
    layer::AbstractLayer
    muts::Vector{NetworkGene}
end

# TODO: specify interface for distributed cache construction
# so we only need to reconstruct deltas weights
WeightCache = LRU{AbstractWeights, Array}
# so we only need to transmit delta genotypes
GenotypeCache = LRU{Int, Network}

struct NetworkInstantiator
    """A function that creates a network from a genotype
    using caches"""
    weight_cache::WeightCache
    genotype_cache::GenotypeCache
end

function get_nearest_ancestor(ancestors::Vector{Int}) 
    """Return nearest cached ancestor from list"""
end
function send_delta_genotype(ancestor::Int, descendant::Int, genotype::Network)
    """Send diff of nearest cached ancestor and descendant and caches the result"""
end

function create_network!(instantiator::NetworkInstantiator, network::Network)
    """Creates flux network from a genotype using caches"""
end
