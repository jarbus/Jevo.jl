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
    seed::UInt64
    mr::Float32
    init!::Union{AbstractInitializer,Function}
end

struct Weights <: AbstractWeights
    dims::Tuple{Vararg{Int}}
    muts::Vector{NetworkGene}
end

struct WeightsCollection{T} <: AbstractWeights where T <: AbstractWeights
    """Concatenation of multiple weight blocks into a single weight tensor, to adjust subsets of weights independently"""
    weights::Array{T}
end

struct FactorWeight{T} <: AbstractWeights where T <: AbstractWeights
    """Low-rank factorization of a weight matrix"""
    A::T
    B::T
end

struct Network <: AbstractLayer
    """A collection and a coupling scheme."""
    coupling::Coupling
    layers::Vector
end

struct Dense{T} <: AbstractLayer where T <: AbstractWeights
    weights::T
    bias::T
    σ::Function
end

# Transformer stuff
# ignore scale for embed, ignore bias for decoder
struct Embed{T} <: AbstractLayer where T <: AbstractWeights
    weights::T
end
struct EmbedDecoder{T} <: AbstractLayer where T <: AbstractWeights
    weights::T
end
struct TransformerDecoder <: AbstractLayer
    blocks::Tuple{Vararg{AbstractLayer}}
end
struct TransformerDecoderBlock <: AbstractLayer
    attention # postnorm residual
    ff # postnorm residual Chain Dense Dense
end
struct PostNormResidual <: AbstractLayer
    layer # ff or attention
    norm # layer norm
end
struct SelfAttention <: AbstractLayer
    n_heads::Int
    qkv::Dense
    out::Dense
end

struct Model <: AbstractPhenotype 
    chain
end

"""
We identify weights of a layer by their dimensions and the last two rng seeds used to generate them.
"""
struct WeightBinding 
    dims::Tuple{Vararg{Int}}
    last_seed::UInt64
    second_to_last_seed::Union{UInt64,Nothing}
end

_WeightCache = Union{LRU{WeightBinding, <:Array{Float32}}, Nothing}
# so we only need to transmit delta genotypes
GenotypeCache = Union{LRU{Int, Network}, Nothing}
