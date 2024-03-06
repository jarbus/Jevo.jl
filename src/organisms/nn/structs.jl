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
    mr::Float32
    init!::Union{AbstractInitializer,Function}
end

struct Weights{N} <: AbstractWeights where N <: Int
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
    coupling::Coupling
    layers::Vector{<:AbstractLayer}
end

struct Dense <: AbstractLayer
    weights::AbstractWeights
    bias::AbstractWeights
    σ::Any
end

struct Model <: AbstractPhenotype 
    chain::Chain
end

_WeightCache = Union{LRU{Int, <:Array{Float32}}, Nothing}
# so we only need to transmit delta genotypes
GenotypeCache = Union{LRU{Int, Network}, Nothing}


