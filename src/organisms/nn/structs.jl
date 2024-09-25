export Transformer, FactorWeight, CompositeWeight, WeightsCollection, Weights, Dense, SelfAttention, JevoChain, PostNormResidual, Embed, EmbedDecoder, LayerNorm, TransformerDecoderBlock, RNN, TextModel, TextTransformer, TextRNN, CoupledWeights
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
mutable struct Weights <: AbstractWeights
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
mutable struct WeightsCollection{T<:AbstractWeights} <: AbstractWeights
    dims::Tuple{Vararg{Int}}
    weights::Array{T}
end

"""
    struct FactorWeight{F1<:AbstractWeights, F2<:AbstractWeights} <: AbstractWeights
        dims::Tuple{Vararg{Int}}
        A::F1
        B::F2
    end
Low-rank factorization of a weight matrix, final tensor is A * B.
"""
mutable struct FactorWeight{F1<:AbstractWeights, F2<:AbstractWeights} <: AbstractWeights
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
mutable struct CompositeWeight{T<:AbstractWeights} <: AbstractWeights
    dims::Tuple{Vararg{Int}}
    weights::Vector{T}
end

struct CoupledWeights
    weights::Vector{Weights}
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

struct RNN{Wi,Wh,B} <: AbstractLayer where {Wi <: AbstractWeights, Wh <: AbstractWeights, B <: AbstractWeights}
    input::Wi
    hidden::Wh
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

"""
    mutable struct SelfAttention <: AbstractLayer
        n_heads::Int
        qkv::Dense
        out::Dense
    end

A self-attention layer. Cross-attention is not supported.
"""
mutable struct SelfAttention <: AbstractLayer
    n_heads::Int
    qkv::Dense
    out::Dense
end

"""
    mutable struct JevoSelfAttention <: AbstractLayer
        n_heads::Int
        qkv::Dense
        out::Dense
    end

A self-attention layer. Cross-attention is not supported. This struct is mutable to allow for a dynamic number of heads.
"""
mutable struct JevoSelfAttention <: AbstractLayer
    n_heads::Int
    qkv::Dense
    out::Dense
end

"""
    struct JevoChain <: AbstractLayer
        layers::Vector
    end

A collection of sequential layers.
"""
struct JevoChain <: AbstractLayer
    layers::Vector
end


struct PostNormResidual <: AbstractLayer
    layer::Union{JevoSelfAttention,SelfAttention,Dense,JevoChain} # ff or attention
    norm::LayerNorm # layer norm
end

struct TransformerDecoderBlock <: AbstractLayer
    attention::PostNormResidual # postnorm residual
    ff::PostNormResidual # postnorm residual Chain Dense Dense
end

struct Transformer <: AbstractLayer
    blocks::Vector{TransformerDecoderBlock}
end


struct TextNetwork{N} <: AbstractLayer where {N <: AbstractLayer}
    embed::Embed
    network::N
    embeddecoder::EmbedDecoder
end

"""
    _WeightCache::LRU{Int, <:Array{Float32}}

Stores developed tensors of weights for genes. Keys are tensor dimensions and the last gene id used. For a weight of dimensions `(a, b)` containing gene ids `1, 2, 3`, `_WeightCache[3, (a,b)]` would map to a tensor equivalent to `tensor(gene_1) + tensor(gene_2) + tensor(gene_3)`.
"""
_WeightCache = Union{LRU{Int, <:Array{Float32}}, Nothing}
# Should be LRU{Int, <:AbstractLayer}, but abstract types slow down the code
_GenotypeCache = Union{LRU, Nothing}

struct TextTransformer end # for creators
struct TextRNN end
# TODO refactor to allow custom position embeds
# this is non trivial and may require us to pass anonymous functions
struct TextModel{TE, M} <: AbstractPhenotype where {TE <: Transformers.TextEncoders.AbstractTransformerTextEncoder,
                                                    M <: AbstractLayer}
    textenc::TE
    posembed::Transformers.Layers.AbstractEmbedding
    embed::Transformers.Layers.Embed
    model::M
    embeddecoder::Transformers.Layers.EmbedDecoder
end
