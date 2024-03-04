export Dense, NetworkGene, Network, Weights, StrictCoupling, LooseCoupling, NoCoupling
abstract type AbstractInitializer <: Function end
abstract type AbstractWeights end
abstract type AbstractLayer end
abstract type AbstractMutation end
"""
    @enum Coupling Strict Loose None

# Values
- `StrictCoupling`: Weights in all sub-layers are mutated using the same initial rng seed
- `LooseCoupling`: Weights in all sub-layers can be mutated independently or together
- `NoCoupling`: Weights in all sub-layers are mutated independently
"""
@enum Coupling StrictCoupling LooseCoupling NoCoupling

# Weights
struct NetworkGene <: AbstractMutation
    id::Int
    seed::UInt16
    mr::Float16
    init::Union{AbstractInitializer,Function}
end

# TODO: change init to svd/kaiming_normal where appropriate
NetworkGene(rng::AbstractRNG, counter::Counter, mr::Float16, init::Function=randn) = 
    NetworkGene(inc!(counter), rand(rng, UInt16), mr, init)

struct Weights{N} <: AbstractWeights where N <: Int
    dims::NTuple{N, Int}
    muts::Vector{NetworkGene}
end

function Weights(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}})
    Weights(dims, [NetworkGene(rng, counter, Float16(1.0))])
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
    coupling::Coupling
    layers::Vector{<:AbstractLayer}
end

function Network(rng::AbstractRNG, counter::AbstractCounter, coupling::Coupling, layers::Vector)
    """Create a network with a collection of layers and a coupling scheme"""
    for l in layers
        @assert length(l) == 3
        @assert l[1] <: AbstractLayer "Layer must be a subtype of AbstractLayer, got $(l[1])"
        @assert typeof(l[2]) <: Tuple{Vararg{Int}} "Dimensions must be a tuple of integers, got $(l[2])"
        @assert typeof(l[3]) <: Union{AbstractInitializer,Function} "Initializer must be a function or a subtype of AbstractInitializer, got $(l[3])"
    end
    Network(coupling, [l[1](l[2], l[3], counter, rng) for l in layers])
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

struct Dense <: AbstractLayer
    weights::AbstractWeights
    bias::AbstractWeights
    σ::Any
end

function Dense(dims::Tuple{Vararg{Int}}, σ::Function, counter::AbstractCounter, rng::AbstractRNG)
    """Create a dense layer with a weight matrix and a bias vector"""
    Dense(Weights(rng, counter, dims), Weights(rng, counter, (dims[end],)), σ)
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
