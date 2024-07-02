import Base: +, ==

NetworkGene(rng::AbstractRNG, counter::Counter, mr::Float32, init::Function=Jevo.apply_kaiming_normal_noise!) = 
    NetworkGene(inc!(counter), rand(rng, UInt64), mr, init)
NetworkGene(counter::Counter, seed::UInt64, mr::Float32, init::Function=Jevo.apply_kaiming_normal_noise!) = 
    NetworkGene(inc!(counter), seed, mr, init)

function Weights(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}}; init::Function=Jevo.apply_kaiming_normal_noise!, rank=-1)
    rank == -1 && return Weights(dims, [NetworkGene(rng, counter, 1f0, init)])
    @assert length(dims) == 2 "Factorized weights must have 2 dimensions, got $(length(dims))"
    CompositeWeight(AbstractWeights[
        FactorWeight(
            dims,
            Weights(rng, counter, (dims[1], rank), init=apply_kaiming_normal_noise_factored!),
            Weights(rng, counter, (rank, dims[2]), init=apply_kaiming_normal_noise_factored!)
        ),
        Weights(rng, counter, dims, init=apply_zero!)])
end

function WeightsCollection(rng::AbstractRNG, counter::Counter; dims::Tuple{Vararg{Int}}, breakdown::Array{Tuple{Int, Int}}, init::Function=Jevo.apply_kaiming_normal_noise!)
    verify_weights_collection(dims, breakdown)
    WeightsCollection(dims, map(tup -> Weights(rng, counter, tup, init=init), breakdown))
end

"""Confirms that each row of weights has the same # of rows in each weight,
and each column has the same number of columns."""
function verify_weights_collection(dims::Tuple{Vararg{Int}}, breakdown::BD) where {BD <: Array{Tuple{Int, Int}}}
    @assert length(dims) <= 2 "WeightsCollection must have 2 dimensions, got $(length(dims))"

    total_rows = sum(dims[1][1] for dims in eachrow(breakdown))
    total_cols = sum(dims[1][2] for dims in eachcol(breakdown))
    @assert dims == (total_rows, total_cols) "WeightsCollection dimensions do not match breakdown"
    row_aligned, column_aligned = true, true

    for row in eachrow(breakdown)
        row_aligned = row_aligned && length(Set(el[1] for el in row)) == 1
    end
    for col in eachcol(breakdown)
        column_aligned = column_aligned && length(Set(el[2] for el in col)) == 1
    end
    @assert row_aligned && column_aligned "WeightsCollection with breakdown $(breakdown) must have aligned rows and columns"
end

WeightCache(;maxsize::Int, by::Function=Base.summarysize) =
    LRU{Int, Array{Float32}}(maxsize=maxsize, by=by)
GenotypeCache(;maxsize::Int, by::Function=Base.summarysize) =
    LRU{Int, Network}(maxsize=maxsize, by=by)


function Base.:+(a::Network, b::Delta) 
    # Likely a performance bottleneck, because we keep copying the network
    # for each delta application
    a = deepcopy(a)
    ws_a, ws_b = get_weights(a), get_weights(b.change)
    @assert length(ws_a) == length(ws_b) "Different number of weights in network and delta"
    for (wa, wb) in zip(ws_a, ws_b)
        wa.dims != wb.dims && @assert false "Different dimensions in network and delta"
        append!(wa.muts, wb.muts)
    end
    a
end

function Base.:(==)(a::Network, b::Network)
    ws_a, ws_b = get_weights(a), get_weights(b)
    length(ws_a) != length(ws_b) && return false
    for (wa, wb) in zip(ws_a, ws_b)
        wa.dims != wb.dims && return false
    end
    true
end

"""Recursively makes copy of network architecture without copying
the individual genes."""
copyarchitecture(x) = x
copyarchitecture(ws::Jevo.Weights) = Jevo.Weights(ws.dims, Vector{Jevo.NetworkGene}())
copyarchitecture(v::Vector) = copyarchitecture.(v)
copyarchitecture(tup::Tuple) = tup |> collect |> copyarchitecture |> Tuple
# embed decoder holds a reference to Embed. We assume this function is always called
# on the Transformer, and thus is called on the Embed layer first, implying the layer
# has already been copied without the weights. We just want to reference that.
copyarchitecture(ed::EmbedDecoder) = EmbedDecoder(ed.embed, copyarchitecture(ed.bias))
copyarchitecture(d::Delta) = Delta(copyarchitecture(d.change))
copyarchitecture(net::T) where {T <: Union{Jevo.AbstractLayer, Jevo.AbstractWeights}} =
    T((copyarchitecture(getfield(net, p)) for p in propertynames(net))...)

function Network(rng::AbstractRNG, counter::AbstractCounter, layers::Vector)
    """Create a network with a collection of layers"""
    for l in layers
        @assert l[1] <: AbstractLayer "Layer must be a subtype of AbstractLayer, got $(l[1])"
    end
    Network([l[1](rng, counter; l[2]...) for l in layers])
end

function Dense(rng::AbstractRNG, counter::AbstractCounter; dims::Tuple{Vararg{Int}}, σ::Function, rank::Int=-1)
    """Create a dense layer with a weight matrix and a bias vector"""
    @assert length(dims) == 2 "Dense layer must have 2 dimensions, got $(length(dims))"
    weights = Weights(rng, counter, (dims[2], dims[1]), rank=rank)
    bias = Weights(rng, counter, (dims[2],))
    Dense(weights, bias, σ)
end

function create_embeds(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}}; rank::Int=-1)
    """Create an embed layer with a (hidden_dim, vocab_dim) weight matrix"""
    @assert length(dims) == 2 "Embed layer must have 2 dimensions, got $(length(dims))"
    embeds = Weights(rng, counter, dims, init=apply_gaussian_normal_noise!, rank=rank)
    bias = Weights(rng, counter, (dims[2],), init=apply_gaussian_normal_noise!)
    embed = Embed(embeds)
    embed, EmbedDecoder(embed, bias)
end

function SelfAttention(rng::AbstractRNG, counter::AbstractCounter;
        n_heads::Int, head_dim::Int, hidden_dim::Int,
        qkv_rank::Int=-1, o_rank::Int=-1,
        init!::Function=Jevo.apply_kaiming_normal_noise!,
    )
    """Create a self-attention layer with n_heads and head_dim"""
    
    # qkv_weight = CompositeWeight([
    #                 FactorWeight(
    #                     Weights(rng, counter, (hidden_dim, qkv_rank), init=init!),
    #                     WeightsCollection(
    #                         reduce(hcat,
    #                                [Weights(rng, counter, (qkv_rank, 1), init=init!) for i in 1:n_heads*head_dim*3]))
    #                 ),
    #                 Weights(rng, counter, (hidden_dim, n_heads*head_dim*3), init=apply_zero!)
    # ])
    # qkv_bias = WeightsCollection(
    #                 reduce(hcat,
    #                     [Weights(rng, counter, (qkv_rank, 1), init=init!) for i in 1:n_heads*head_dim*3]))
    # qkv = Dense(qkv_weight, qkv_bias, σ=identity)
    qkv = Dense(rng, counter, dims=(hidden_dim, n_heads*head_dim*3), σ=identity, rank=qkv_rank)
    out = Dense(rng, counter, dims=(n_heads*head_dim, hidden_dim), σ=identity, rank=o_rank)
    SelfAttention(n_heads, qkv, out)
end
function LayerNorm(rng::AbstractRNG, counter::AbstractCounter; hidden_dim::Int)
    """Create a layer norm with scale and bias"""
    scale = Weights(rng, counter, (hidden_dim,), init=apply_one!)
    bias = Weights(rng, counter, (hidden_dim,), init=apply_zero!)
    LayerNorm(hidden_dim, scale, bias)
end

function PostNormResidual(rng::AbstractRNG, counter::AbstractCounter, layer::AbstractLayer; hidden_dim::Int)
    """Create a post-norm residual layer with a layer and a layer norm"""
    norm = LayerNorm(rng, counter, hidden_dim=hidden_dim)
    PostNormResidual(layer, norm)
end

function TransformerDecoderBlock(rng::AbstractRNG, counter::AbstractCounter;
        n_heads::Int, head_dim::Int, ff_dim::Int, hidden_dim::Int, qkv_rank::Int=-1, o_rank::Int=-1, ff_rank::Int=-1)
    sa = SelfAttention(rng, counter, n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim, qkv_rank=qkv_rank, o_rank=o_rank)
    ff = Jevo.Chain(
                (Dense(rng, counter, dims=(hidden_dim, ff_dim), σ=gelu, rank=ff_rank),
                Dense(rng, counter, dims=(ff_dim, hidden_dim), σ=identity, rank=ff_rank),)
        )
    # postnorm
    pnr_sa = Jevo.PostNormResidual(rng, counter, sa, hidden_dim=hidden_dim)
    pnr_ff = Jevo.PostNormResidual(rng, counter, ff, hidden_dim=hidden_dim)
    TransformerDecoderBlock(pnr_sa, pnr_ff)
end

function Transformer(rng::AbstractRNG, counter::AbstractCounter;
        n_blocks::Int,
        hidden_dim::Int, 
        n_heads::Int,
        head_dim, 
        ff_dim,
        qkv_rank::Int=-1,
        o_rank::Int=-1,
        ff_rank::Int=-1,
        embed_rank::Int=-1,
        vocab_size::Int)
    """Create a transformer with n_layers of attention and feedforward blocks"""
    embed, embeddecoder = create_embeds(rng, counter, (hidden_dim, vocab_size), rank=embed_rank)
    blocks = Tuple(TransformerDecoderBlock(rng, counter,
                                           n_heads=n_heads,
                                           head_dim=head_dim,
                                           ff_dim=ff_dim,
                                           hidden_dim=hidden_dim,
                                           qkv_rank=qkv_rank,
                                           o_rank=o_rank,
                                           ff_rank=ff_rank,
                                          ) for _ in 1:n_blocks)
    Transformer(embed, blocks, embeddecoder)
end
