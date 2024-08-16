import Base: +, ==

NetworkGene(rng::AbstractRNG, counter::Counter, mr::Float32, init::Function=Jevo.apply_kaiming_normal_noise!) = 
    NetworkGene(inc!(counter), rand(rng, UInt64), mr, init)
NetworkGene(counter::Counter, seed::UInt64, mr::Float32, init::Function=Jevo.apply_kaiming_normal_noise!) = 
    NetworkGene(inc!(counter), seed, mr, init)

function Weights(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}}; init::Function=Jevo.apply_kaiming_normal_noise!, rank=-1)
    rank == -1 && return Weights(dims, [NetworkGene(rng, counter, 1f0, init)])
    @assert length(dims) == 2 "Factorized weights must have 2 dimensions, got $(length(dims))"
    CompositeWeight(dims, AbstractWeights[
        FactorWeight(
            dims,
            Weights(rng, counter, (dims[1], rank), init=apply_kaiming_normal_noise_factored!),
            Weights(rng, counter, (rank, dims[2]), init=apply_kaiming_normal_noise_factored!)
        ),
        Weights(rng, counter, dims, init=apply_zero!)])
end

function WeightsCollection(rng::AbstractRNG, counter::Counter; dims::Tuple{Vararg{Int}}, breakdown::BD, init::Function=Jevo.apply_kaiming_normal_noise!) where {BD <: Array{<:Tuple{Vararg{Int}}}}
    verify_weights_collection(dims, breakdown)
    WeightsCollection(dims, map(tup -> Weights(rng, counter, tup, init=init), breakdown))
end

"""Confirms that each row of weights has the same # of rows in each weight,
and each column has the same number of columns."""
function verify_weights_collection(dims::Tuple{Vararg{Int}}, breakdown::BD) where {BD <: Matrix{Tuple{Int, Int}}}
    total_rows = sum(w_dims[1][1] for w_dims in eachrow(breakdown))
    total_cols = sum(w_dims[1][2] for w_dims in eachcol(breakdown))
    @assert dims == (total_rows, total_cols) "WeightsCollection dimensions do not match breakdown"
    row_aligned, column_aligned = true, true

    for row in eachrow(breakdown)
        row_aligned = row_aligned && length(Set(w_dims[1] for w_dims in row)) == 1
    end
    for col in eachcol(breakdown)
        column_aligned = column_aligned && length(Set(w_dims[2] for w_dims in col)) == 1
    end
    @assert row_aligned && column_aligned "WeightsCollection with breakdown $(breakdown) must have aligned rows and columns"
end
function verify_weights_collection(dims::Tuple{Vararg{Int}}, breakdown::BD) where {BD <: Vector{<:Tuple{Vararg{Int}}}}
    # convert vector to Nx1 matrix if we are targeting a matrix
    length(dims) == 2 && return verify_weights_collection(dims, reshape(breakdown, (length(breakdown), 1)))
    @assert length(dims) == 1
    @assert all(length(w_dims)==1 for w_dims in breakdown) "Weights in breakdown must be a vector if breakdown is a vector, got $(breakdown)"
    @assert dims[1] == sum(w_dims[1] for w_dims in breakdown) "WeightsCollection dimensions do not match breakdown, got $(dims) and $(sum(w_dims[1] for w_dims in breakdown))"
end

WeightCache(;maxsize::Int, by::Function=Base.summarysize) =
    LRU{Int, Array{Float32}}(maxsize=maxsize, by=by)
GenotypeCache(;maxsize::Int, by::Function=Base.summarysize) =
    LRU{Int, Network}(maxsize=maxsize, by=by)

"""
    Base.:+(a::Network b::Delta) -> Network

Adds delta to genome, returns a new, full genome.

Where `add_delta_to_genome(genome::Network delta::Delta; n_back::Float)` creates a compact copy of the tail genes for each weight on a worker, this function adds the delta to a full copy of the genome on master.

See also: [add_delta_to_genome(genome::Network delta::Delta)](@ref)
"""
function Base.:+(a::Network, b::Delta)
    # Likely a performance bottleneck, because we keep copying the network
    # for each delta application
    a = deepcopy(a)
    insert_new_decoder_blocks!(a, b)
    ws_a, ws_b = get_weights(a), get_weights(b.change)
    @assert length(ws_a) == length(ws_b) "Different number of weights in network and delta"
    for (wa, wb) in zip(ws_a, ws_b)
        wa.dims != wb.dims && @assert false "Different dimensions in network and delta"
        append!(wa.muts, wb.muts)
    end
    a
end

"""
    add_delta_to_genome(genome::Network delta::Delta; n_back=20) -> Network

Adds delta to genome, keeping track of the last `n_back` mutations from the full genome. Does not look at mutations before the last `n_back` mutations to save memory and time.

Where `Base.:+(a::Network, b::Delta)` adds a delta to the full genome on master, this function only adds the delta to a compact genome on workers for evaluation. This isn't great; we'd rather use Base.:+ for both purposes, but this is essential for performance.

See also: [Base.:+(a::Network, b::Delta)](@ref)
"""
add_delta_to_genome(genome::AbstractGenotype, delta::Delta) = genome + delta
function add_delta_to_genome(full_genome::Network, delta::Delta{Network}; n_back=20)
    # Likely a performance bottleneck, because we keep copying the network
    # for each delta application
    compact_genome = copyarchitecture(full_genome)
    insert_new_decoder_blocks!(compact_genome, delta)
    ws_compact, ws_delta, ws_full =
        get_weights(compact_genome), get_weights(delta.change), get_weights(full_genome)
    @assert length(ws_compact) == length(ws_delta) "Different number of weights in compact genome and delta"
    # full genome might have different number of weights than compact genome, so we need
    # to copy full genome for the existing weights at different indices. We explicitly
    # copy new layers and attention heads before this, so we can ignore fresh weights.
    full_idx, delta_idx = 1, 1
    while delta_idx <= length(ws_compact)
        if is_fresh(ws_delta[delta_idx])
            delta_idx += 1
            continue
        end
        wc, wd, wf = ws_compact[delta_idx], ws_delta[delta_idx], ws_full[full_idx]
        wc.dims != wd.dims && @assert false "Different dimensions in compact network and delta"
        @assert isempty(wc.muts) || wc.muts[1].id < 0 "wc with dims $(wc.dims) not empty for type $(typeof(compact_genome)) and is not a fresh weight"
        @assert !isempty(wf.muts) || wc.muts[1].id < 0 "Full genome has no mutations"
        start_idx = max(1, length(wf.muts)-n_back)
        append!(wc.muts, wf.muts[start_idx:end]) # add last n_back muts from full genome
        append!(wc.muts, wd.muts)                # then add delta muts
        full_idx += 1
        delta_idx += 1
    end
    compact_genome
end

function insert_new_attention_heads!(genome::Network, delta::Delta)
    tfr1, tfr2 = genome.layers[1], delta.change.layers[1]
    !(tfr1 isa Transformer && tfr2 isa Transformer) && return
end


function insert_new_decoder_blocks!(genome::Network, delta::Delta)
    tfr1, tfr2 = genome.layers[1], delta.change.layers[1]
    !(tfr1 isa Transformer && tfr2 isa Transformer) && return
    # check if transformer has a new block
    new_blocks = [is_fresh(block) for block in tfr2.blocks]
    !any(new_blocks) && length(tfr1.blocks) == length(tfr2.blocks) && return
    @assert length(tfr1.blocks) < length(tfr2.blocks) "New block must be added to delta"
    # deep copy new blocks from delta into genome at same index
    for i in 1:length(tfr2.blocks)
        !new_blocks[i] && continue
        insert!(tfr1.blocks, i, deepcopy(tfr2.blocks[i]))
    end
    @assert length(tfr1.blocks) == length(tfr2.blocks) "Should be same number of blocks, $(length(tfr1.blocks)) ≠ $(length(tfr2.blocks))"
    @info "Delta has new block, added to genome"
end

"""
Check if two networks have the same architecture, but *not* the same weights.
"""
function Base.:(==)(a::Network, b::Network)
    ws_a, ws_b = get_weights(a), get_weights(b)
    length(ws_a) != length(ws_b) && return false
    for (wa, wb) in zip(ws_a, ws_b)
        wa.dims != wb.dims && return false
    end
    true
end

"""
    copyarchitecture(x)
    copyarchitecture(layer::AbstractLayer)
    copyarchitecture(layer::Network)

Recursively makes copy of network architecture without copying
individual genes."""
copyarchitecture(x) = x
copyarchitecture(ws::Jevo.Weights) = Jevo.Weights(ws.dims, Vector{Jevo.NetworkGene}())
copyarchitecture(itr::Union{Array, Tuple}) = copyarchitecture.(itr)
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
    pnr_ff = nothing
    #= pnr_ff = Jevo.PostNormResidual(rng, counter, ff, hidden_dim=hidden_dim) =#
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
    blocks = [TransformerDecoderBlock(rng, counter,
                                           n_heads=n_heads,
                                           head_dim=head_dim,
                                           ff_dim=ff_dim,
                                           hidden_dim=hidden_dim,
                                           qkv_rank=qkv_rank,
                                           o_rank=o_rank,
                                           ff_rank=ff_rank,
                                           ) for _ in 1:n_blocks]
    Transformer(embed, blocks, embeddecoder)
end
