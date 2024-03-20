NetworkGene(rng::AbstractRNG, counter::Counter, mr::Float32, init!::Function=Jevo.apply_kaiming_normal_noise!) = 
    NetworkGene(inc!(counter), rand(rng, UInt64), mr, init!)

function Weights(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}})
    Weights(dims, [NetworkGene(rng, counter, Float32(1.0))])
end

function WeightCache(;maxsize::Int, by::Function=sizeof)
    LRU{WeightBinding, Array{Float32}}(maxsize=maxsize, by=by)
end

function Network(rng::AbstractRNG, counter::AbstractCounter, coupling::Coupling, layers::Vector)
    """Create a network with a collection of layers and a coupling scheme"""
    for l in layers
        @assert length(l) == 3
        @assert l[1] <: AbstractLayer "Layer must be a subtype of AbstractLayer, got $(l[1])"
        @assert typeof(l[2]) <: Tuple{Vararg{Int}} "Dimensions must be a tuple of integers, got $(l[2])"
        @assert typeof(l[3]) <: Union{AbstractInitializer,Function} "Initializer must be a function or a subtype of AbstractInitializer, got $(l[3])"
    end
    Network(coupling, [l[1](rng, counter, l[2:end]...) for l in layers])
end

function Dense(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}}, σ::Function)
    """Create a dense layer with a weight matrix and a bias vector"""
    @assert length(dims) == 2 "Dense layer must have 2 dimensions, got $(length(dims))"
    weights = Weights(rng, counter, (dims[2], dims[1]))
    bias = Weights(rng, counter, (dims[2],))
    Dense(weights, bias, σ)
end

function create_embeds(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}})
    """Create an embed layer with a (hidden_dim, vocab_dim) weight matrix"""
    @assert length(dims) == 2 "Embed layer must have 2 dimensions, got $(length(dims))"
    embeds = Weights(rng, counter, dims)
    bias = Weights(rng, counter, (dims[2],))
    Embed(embeds), EmbedDecoder(embeds, bias)
end

function SelfAttention(rng::AbstractRNG, counter::AbstractCounter;
        n_heads::Int, head_dim::Int, hidden_dim::Int,
        init!::Function=Jevo.apply_kaiming_normal_noise!,
    )
    """Create a self-attention layer with n_heads and head_dim"""
    
    qkv = Dense(rng, counter, (hidden_dim, n_heads*head_dim*3), identity)
    out = Dense(rng, counter, (n_heads*head_dim, hidden_dim), identity)
    SelfAttention(n_heads, qkv, out)
end

function TransformerDecoderBlock(rng::AbstractRNG, counter::AbstractCounter;
        n_heads::Int, head_dim::Int, ff_dim::Int)
    # attention
    # ff
end

function Transformer(rng::AbstractRNG, counter::AbstractCounter;
        n_blocks::Int,
        hidden_dim::Int, 
        n_heads::Int,
        head_dim, 
        ff_dim,
        vocab_size::Int)
    """Create a transformer with n_layers of attention and feedforward blocks"""
    embeds, decoder = create_embeds(rng, counter, (hidden_dim, vocab_size))
end
