export TransformerPhenotype, Transformer, FactorWeight, CompositeWeight, WeightsCollection, Weights, Dense, SelfAttention, Chain, Network, Model, PostNormResidual, Embed, EmbedDecoder, LayerNorm, TransformerDecoderBlock
"""
    struct NetworkGene <: AbstractMutation
        id::Int
        seed::UInt64
        mr::Float32
        init!::Union{AbstractInitializer,Function}
    end

Gene for a [Weights](@ref) object, used to generate a tensor of `Float32`. A `StableRNG` object is created with seed `seed`, which is then passed into `init!` to generate the tensor. The tensor is then mutated by mutation rate `mr`. `id` is a unique identifier for the gene.
"""
struct NetworkGene <: AbstractMutation
    id::Int
    seed::UInt64
    mr::Float32
    init!::Union{AbstractInitializer,Function}
end

"""
    struct Weights <: AbstractWeights
        dims::Tuple{Vararg{Int}}
        muts::Vector{NetworkGene}
    end

A collection of genes which generate a tensor of `Float32` when developed. Each gene in `muts` is developed into a tensor and added together to form the final tensor. A [`_WeightCache`](@ref) can be used to cache intermediate tensors to avoid redundant computation.
"""
struct Weights <: AbstractWeights
    dims::Tuple{Vararg{Int}}
    muts::Vector{NetworkGene}
end

"""
    struct WeightsCollection{T<:AbstractWeights} <: AbstractWeights
        dims::Tuple{Vararg{Int}}
        weights::Array{T}
    end
Concatenation of multiple weight blocks into a single weight tensor, to adjust subsets of weights independently
"""
struct WeightsCollection{T<:AbstractWeights} <: AbstractWeights
    dims::Tuple{Vararg{Int}}
    weights::Array{T}
end

"""
    struct FactorWeight{F1<:AbstractWeights, F2<:AbstractWeights} <: AbstractWeights
        dims::Tuple{Vararg{Int}}
        A::F1
        B::F2
    end
Low-rank factorization of a weight matrix
"""
struct FactorWeight{F1<:AbstractWeights, F2<:AbstractWeights} <: AbstractWeights
    dims::Tuple{Vararg{Int}}
    A::F1
    B::F2
end

"""
    struct CompositeWeight{T<:AbstractWeights} <: AbstractWeights
        dims::Tuple{Vararg{Int}}
        weights::Vector{T}
    end

A collection of weights which are added together. Each element must develop to the same size
"""
struct CompositeWeight{T<:AbstractWeights} <: AbstractWeights
    dims::Tuple{Vararg{Int}}
    weights::Vector{T}
end

"""
    struct Network <: AbstractLayer
        layers::Vector
    end

A collection of sequential layers.
"""
struct Network <: AbstractLayer
    layers::Vector
end

"""
    struct Dense{W,B} <: AbstractLayer where {W <: AbstractWeights, B <: AbstractWeights}
        weights::W
        bias::B
        σ::Function
    end

Standard dense layer with weights, bias, and activation function
"""
struct Dense{W,B} <: AbstractLayer where {W <: AbstractWeights, B <: AbstractWeights}
    weights::W
    bias::B
    σ::Function
end

# Transformer stuff
# ignore scale for embed
"""
    struct Embed{T} <: AbstractLayer where T <: AbstractWeights
        weights::T
    end

Embedding Matrix for language models
"""
struct Embed{T} <: AbstractLayer where T <: AbstractWeights
    weights::T
end

struct EmbedDecoder{W, B} <: AbstractLayer where {W <: AbstractWeights, B <: AbstractWeights}
    embed::Embed{W}
    bias::Union{Nothing,B}
end

"""
    struct LayerNorm{T} <: AbstractLayer where T <: Union{Nothing, <:AbstractWeights}
        hidden_dim::Int
        scale::T
        bias::T
    end

Layer normalization layer. If `scale` and `bias` are initialized to `1` and `0` respectively.
"""
struct LayerNorm{T} <: AbstractLayer where T <: Union{Nothing, <:AbstractWeights}
    hidden_dim::Int
    scale::T
    bias::T
end

struct TransformerDecoderBlock <: AbstractLayer
    attention # postnorm residual
    ff # postnorm residual Chain Dense Dense
end

"""
    struct Transformer <: AbstractLayer
        embed::Embed
        blocks::Tuple{Vararg{TransformerDecoderBlock}}
        embeddecoder::EmbedDecoder
    end

Decoder-only transformer genotype
"""
struct Transformer <: AbstractLayer
    embed::Embed
    blocks::Tuple{Vararg{TransformerDecoderBlock}}
    embeddecoder::EmbedDecoder
end

struct PostNormResidual <: AbstractLayer
    layer # ff or attention
    norm # layer norm
end

"""
    struct SelfAttention <: AbstractLayer
        n_heads::Int
        qkv::Dense
        out::Dense
    end

A self-attention layer. Cross-attention is not supported.
"""
struct SelfAttention <: AbstractLayer
    n_heads::Int
    qkv::Dense
    out::Dense
end

"""
    struct Chain <: AbstractLayer
        layers::Tuple{Vararg{<:AbstractLayer}}
    end

A collection of sequential layers
"""
struct Chain <: AbstractLayer
    layers::Tuple{Vararg{<:AbstractLayer}}
end

struct Model <: AbstractPhenotype 
    chain
end

"""
    _WeightCache::LRU{Int, <:Array{Float32}}

Stores developed tensors of weights for genes. Keys are tensor dimensions and the last gene id used. For a weight of dimensions `(a, b)` containing gene ids `1, 2, 3`, `_WeightCache[3, (a,b)]` would map to a tensor equivalent to `tensor(gene_1) + tensor(gene_2) + tensor(gene_3)`.
"""
_WeightCache = Union{LRU{Int, <:Array{Float32}}, Nothing}
# so we only need to transmit delta genotypes
_GenotypeCache = Union{LRU{Int, Network}, Nothing}

struct TransformerPhenotype <: AbstractPhenotype
    textenc::Transformers.TextEncoders.TransformerTextEncoder
    posembed::Transformers.Layers.AbstractEmbedding
    embed::Transformers.Layers.Embed
    trf::Transformers.Layers.Transformer
    embeddecoder::Transformers.Layers.EmbedDecoder
end
