export TextRNN, TextTransformer
import Base: +, ==, hash

NetworkGene(rng::AbstractRNG, counter::Counter, mr::Float32, init::Function=Jevo.apply_kaiming_normal_noise!) = 
    NetworkGene(inc!(counter), rand(rng, UInt64), mr, init)
NetworkGene(counter::Counter, seed::UInt64, mr::Float32, init::Function=Jevo.apply_kaiming_normal_noise!) = 
    NetworkGene(inc!(counter), seed, mr, init)

function FreshWeights(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}}; init::Function=Jevo.apply_kaiming_normal_noise!, rank=-1)
    (length(dims) < 2 || rank < 0) && return Weights(dims, [NetworkGene(-inc!(counter), rand(rng, UInt64), 0.1f0, init)])
    CompositeWeight(dims, AbstractWeights[
        FactorWeight(dims,
            Weights((dims[1], rank), [NetworkGene(-inc!(counter), rand(rng, UInt64), 0.1f0, apply_kaiming_normal_noise_factored!)]),
            Weights((rank, dims[2]), [NetworkGene(-inc!(counter), rand(rng, UInt64), 0.1f0, apply_kaiming_normal_noise_factored!)]),
        ),
        Weights(dims, [NetworkGene(-inc!(counter), rand(rng, UInt64), 0.1f0, apply_zero!)])
    ])
end
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

WeightCache(;maxsize::Int) = LRU{Int, Array{Float32}}(maxsize=maxsize)
GenotypeCache(;maxsize::Int) = LRU{Int, Any}(maxsize=maxsize)

"""
    Base.:+(a::Layer b::Delta) -> Layer

Adds delta to genome, returns a new, full genome. Only valid for applying a child delta to a parent genome, behavior is undefined otherwise.

Where `add_delta_to_genome(genome::Layer delta::Delta; n_back::Float)` creates a compact copy of the tail genes for each weight on a worker, this function adds the delta to a full copy of the genome on master.

See also: [add_delta_to_genome(genome::Layer delta::Delta)](@ref)
"""
function Base.:+(a::AbstractLayer, b::Delta)
    # NOTE Likely a performance bottleneck, because we keep copying the network
    # for each delta application
    a = deepcopy(a)
    apply_transformer_architecture_changes!(a, b)
    ws_a, ws_b = get_weights(a), get_weights(b.change)
    @assert length(ws_a) == length(ws_b) "Different number of weights in network and delta"
    for (wa, wb) in zip(ws_a, ws_b)
        wa.dims != wb.dims && @assert false "Different dimensions in network and delta"
        is_fresh(wb) && continue  # fresh weights have already been added above
        append!(wa.muts, wb.muts)
    end

    a
end

"""
    add_delta_to_genome(genome::AbstractLayer delta::Delta; n_back=20) -> Layer

Adds delta to genome, keeping track of the last `n_back` mutations from the full genome. Does not look at mutations before the last `n_back` mutations to save memory and time. Only valid for applying a child delta to a parent genome, behavior is undefined otherwise.

Where `Base.:+(a::AbstractLayer, b::Delta)` adds a delta to the full genome on master, this function only adds the delta to a compact genome on workers for evaluation. This isn't great; we'd rather use Base.:+ for both purposes, but this is essential for performance.

See also: [Base.:+(a::AbstractLayer, b::Delta)](@ref)
"""
add_delta_to_genome(genome::AbstractGenotype, delta::Delta) = genome + delta
function add_delta_to_genome(full_genome::AbstractLayer, delta::Delta; n_back=20)
    # Likely a performance bottleneck, because we keep copying the network
    # for each delta application

    compact_genome = copyarchitecture(full_genome)
    apply_transformer_architecture_changes!(compact_genome, delta)
    ws_compact, ws_delta, ws_full =
        get_weights(compact_genome), get_weights(delta.change), get_weights(full_genome)
    @assert length(ws_compact) == length(ws_delta) "Different number of weights in compact genome and delta"
    # full genome might have different number of weights than compact genome, so we need
    # to copy full genome for the existing weights at different indices. We explicitly
    # copy new layers and attention heads before this, so we can ignore fresh weights.
    full_idx, delta_idx = 1, 1
    while delta_idx <= length(ws_compact)
        if is_fresh(ws_delta[delta_idx])  # fresh weights have been added above
            delta_idx += 1
            continue
        end
        wc, wd, wf = ws_compact[delta_idx], ws_delta[delta_idx], ws_full[full_idx]
        wc.dims != wd.dims && @assert false "Different dimensions in compact network and delta"
        @assert isempty(wc.muts) || wc.muts[1].id < 0 "wc with dims $(wc.dims) not empty for type $(typeof(compact_genome)) and is not a fresh weight"
        @assert !isempty(wf.muts) "Full genome has weight with no mutations: $(visualize(full_genome))"
        start_idx = max(1, length(wf.muts)-n_back)
        append!(wc.muts, wf.muts[start_idx:end]) # add last n_back muts from full genome
        append!(wc.muts, wd.muts)                # then add delta muts
        full_idx += 1
        delta_idx += 1
    end 
    compact_genome
end

function apply_transformer_architecture_changes!(a, b)
    isempty(map_get(a, Jevo.Transformer)) && return
    insert_new_decoder_blocks!(a, b)
    insert_new_attention_heads!(a, b)
    update_dimensions!(a)
end

update_dimensions!(x::Delta) = update_dimensions!(x.change)
update_dimensions!(x) = nothing
update_dimensions!(x::Weights) = nothing
update_dimensions!(x::FactorWeight) = x.dims = (x.A.dims[1], x.B.dims[2])
function update_dimensions!(x::CompositeWeight) 
    @assert 1 == length(unique(w.dims for w in x.weights)) "All weights should have the same dimension for a composite weight"
    x.dims = x.weights[1].dims
end
function update_dimensions!(x::WeightsCollection)
    @assert 1 <= length(x.dims) <= 2 "WeightsCollection must be a vector or matrix"
    n_rows = sum(r[1].dims[1] for r in eachrow(x.weights))
    if length(x.dims) == 1
        x.dims = (n_rows,)
        return
    end
    n_cols = sum(c[1].dims[2] for c in eachcol(x.weights))
    x.dims = (n_rows, n_cols)
end
update_dimensions!(x::AbstractLayer) = map!(x, postordering=true) do hierarchy
    x != hierarchy[end] && hierarchy[end] isa AbstractWeights && update_dimensions!(hierarchy[end])  # prevent stack overflow
end

"""
    insert_new_attention_heads!(genome::AbstractLayer, delta::Delta)

Goes through all weight collections in each self attention layer, and makes copies of fresh heads from the delta where appropriate. Only valid for a parent and a child network with at most one extra head.
"""
function insert_new_attention_heads!(genome::AbstractLayer, delta::Delta)
    genome_attn_layers, delta_attn_layers = map_get(genome, JevoSelfAttention), map_get(delta, JevoSelfAttention)
    @assert length(genome_attn_layers) == length(delta_attn_layers) "Different number of attention layers in genome and delta"
    # go over each attention layer
    for (g_attn, d_attn) in zip(genome_attn_layers, delta_attn_layers)
        g_wcs, d_wcs = map_get(g_attn, WeightsCollection), map_get(d_attn, WeightsCollection)
        isempty(d_wcs) && isempty(g_wcs) && continue  # we cant insert new heads for this layer
        @assert length(d_wcs) == 3 "Delta must have 3 weight collections for qkv and o, got $(length(d_wcs))"
        # enumerate qkv weight collection,, qkv bias collection,, out weight collection
        for (idx, (g_wc, d_wc)) in enumerate(zip(g_wcs, d_wcs))
            length(g_wc.weights) == length(d_wc.weights) && continue
            @assert length(d_wc.weights) > length(g_wc.weights) "New heads must be added to delta"
            concatenate = idx == 3 ? hcat : vcat
            g_idx, d_idx = 1, 1
            while d_idx <= length(d_wc.weights)
                if is_fresh(d_wc.weights[d_idx])
                    # If a weight is fresh, it might be part of a new layer,
                    # so we skip if it's already in the genome.
                    g_idx <= length(g_wc.weights) && d_wc.weights[d_idx] == g_wc.weights[g_idx] &&  continue
                    if g_idx <= length(g_wc.weights)
                        g_wc.weights = concatenate(g_wc.weights[1:g_idx-1], deepcopy(d_wc.weights[d_idx]), g_wc.weights[g_idx:end])
                    else
                        g_wc.weights =concatenate(g_wc.weights, deepcopy(d_wc.weights[d_idx]))
                    end
                end
                g_idx += 1
                d_idx += 1
            end
            @assert length(g_wc.weights) == length(d_wc.weights) "Should be same number of weights, $(length(g_wc.weights)) ≠ $(d_idx)"
        end
        g_attn.n_heads = length(d_wcs[3].weights)
    end
end


"""
    insert_new_decoder_blocks!(genome::AbstractLayer, delta::Delta)

If a delta was created that has new decoder blocks, this function inserts them into the genome at the same index. Used for creating new genomes by applying a delta.

See also: [Base.:+(a::AbstractLayer, b::Delta)](@ref), [add_delta_to_genome(genome::AbstractLayer delta::Delta)](@ref)
"""
function insert_new_decoder_blocks!(genome::AbstractLayer, delta::Delta)
    tfr1, tfr2 = map_get(genome, Transformer), map_get(delta, Transformer)
    @assert length(tfr1) == length(tfr2) == 1 "There should only be one transformer in genome and delta"
    tfr1, tfr2 = tfr1[1], tfr2[1]
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
end

function Base.:(==)(a::AbstractLayer, b::AbstractLayer)
    ws_a, ws_b = get_weights(a), get_weights(b)
    length(ws_a) != length(ws_b) && return false
    for (wa, wb) in zip(ws_a, ws_b)
        wa != wb && return false
    end
    true
end
Base.:(==)(a::Weights, b::Weights) = a.dims == b.dims && a.muts == b.muts
Base.:(==)(a::FactorWeight, b::FactorWeight) = a.dims == b.dims && a.A == b.A && a.B == b.B
Base.:(==)(a::CompositeWeight, b::CompositeWeight) = a.dims == b.dims && a.weights == b.weights
Base.:(==)(a::WeightsCollection, b::WeightsCollection) = a.dims == b.dims && a.weights == b.weights
hash(x::Weights) = hash((x.dims, x.muts))
hash(x::FactorWeight) = hash((x.dims, x.A, x.B))
hash(x::Union{CompositeWeight,WeightsCollection}) = hash((x.dims, x.weights))
hash(l::Union{AbstractLayer, Delta}) = hash(hash.(get_weights(l)))
hash(c::Flux.Chain) = c |> Transformers.tocpudevice |> Flux.params |> Iterators.flatten |> collect |> hash

function hash(tm::TextModel) 
    embed_decode_params = tm.embeddecoder |> Transformers.tocpudevice |> Flux.params |> Iterators.flatten |> collect
    tm_params = tm.model |> Transformers.tocpudevice |> Flux.params |> Iterators.flatten |> collect
    hash((embed_decode_params, tm_params))
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
copyarchitecture(ed::EmbedDecoder) = error("Invalid, we need to copy Embed and EmbedDecoder together")
function copyarchitecture(tn::TextNetwork) 
    embed = copyarchitecture(tn.embed)
    embeddecoder = EmbedDecoder(embed, copyarchitecture(tn.embeddecoder.bias))
    TextNetwork(embed , copyarchitecture(tn.network),  embeddecoder)
end
copyarchitecture(d::Delta) = Delta(copyarchitecture(d.change))
copyarchitecture(net::T) where {T <: Union{Jevo.AbstractLayer, Jevo.AbstractWeights}} =
    T((copyarchitecture(getfield(net, p)) for p in propertynames(net))...)

function JevoChain(rng::AbstractRNG, counter::AbstractCounter, layers::Vector)
    """Create a network with a collection of layers"""
    for l in layers
        @assert l[1] <: AbstractLayer "Layer must be a subtype of AbstractLayer, got $(l[1])"
    end
    JevoChain([l[1](rng, counter; l[2]...) for l in layers])
end

function Dense(rng::AbstractRNG, counter::AbstractCounter; dims::Tuple{Vararg{Int}}, σ::Function, rank::Int=-1)
    """Create a dense layer with a weight matrix and a bias vector"""
    @assert length(dims) == 2 "Dense layer must have 2 dimensions, got $(length(dims))"
    weights = Weights(rng, counter, (dims[2], dims[1]), rank=rank)
    bias = Weights(rng, counter, (dims[2],))
    Dense(weights, bias, σ)
end

function RNN(rng::AbstractRNG, counter::AbstractCounter; dims::Tuple{Int, Int}, σ::Function, input_rank::Int=-1, hidden_rank::Int=-1)
    input = Weights(rng, counter, (dims[2], dims[1]))
    hidden = Weights(rng, counter, (dims[2], dims[2]))
    bias = Weights(rng, counter, (dims[2],))
    RNN(input, hidden, bias, σ)
end

function create_embeds(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}}; rank::Int=-1)
    """Create an embed layer with a (hidden_dim, vocab_dim) weight matrix"""
    @assert length(dims) == 2 "Embed layer must have 2 dimensions, got $(length(dims))"
    embeds = Weights(rng, counter, dims, init=apply_gaussian_normal_noise!, rank=rank)
    bias = Weights(rng, counter, (dims[2],))
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

function JevoSelfAttention(rng::Jevo.AbstractRNG, counter::Jevo.AbstractCounter; n_heads::Int, head_dim::Int, hidden_dim::Int, qkv_rank::Int=-1, o_rank::Int=-1, init!::Function=Jevo.apply_kaiming_normal_noise!,)
    # NOTE: QKV weights are transposed, because we aren't going through our custom Dense constructor
    #       which automatically transposes for us.
    head_init! = qkv_rank < 1 ? init! : Jevo.apply_kaiming_normal_noise_factored!
    o_init! = o_rank < 1 ? init! : Jevo.apply_kaiming_normal_noise!
    qkv_weights = Jevo.WeightsCollection(
        (head_dim*n_heads*3, hidden_dim),
        [Jevo.Weights(rng, counter, (head_dim, hidden_dim), init=head_init!, rank=qkv_rank) for i in 1:n_heads*3])

    qkv_bias = Jevo.WeightsCollection(
        (head_dim*n_heads*3,),
        [Jevo.Weights(rng, counter, (head_dim,), init=init!) for i in 1:n_heads*3]
    )

    out_weights = Jevo.WeightsCollection(
        (hidden_dim, head_dim*n_heads),
        [Jevo.Weights(rng, counter, (hidden_dim, head_dim), init=o_init!, rank=o_rank) for _ in 1:1, h in 1:n_heads])

    out_bias = Jevo.Weights(rng, counter, (hidden_dim,))
    
    qkv = Jevo.Dense(qkv_weights, qkv_bias, identity)
    out = Jevo.Dense(out_weights, out_bias, identity)

    JevoSelfAttention(n_heads, qkv, out)
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
    sa = JevoSelfAttention(rng, counter, n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim, qkv_rank=qkv_rank, o_rank=o_rank)
    ff = JevoChain([
            Dense(rng, counter, dims=(hidden_dim, ff_dim), σ=gelu, rank=ff_rank),
            Dense(rng, counter, dims=(ff_dim, hidden_dim), σ=identity, rank=ff_rank)
    ])
    # postnorm
    pnr_sa = Jevo.PostNormResidual(rng, counter, sa, hidden_dim=hidden_dim)
    #pnr_ff = nothing
    pnr_ff = Jevo.PostNormResidual(rng, counter, ff, hidden_dim=hidden_dim)
    TransformerDecoderBlock(pnr_sa, pnr_ff)
end

function TextTransformer(rng::AbstractRNG, counter::AbstractCounter;
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
    tfr = Jevo.Transformer([TransformerDecoderBlock(rng, counter,
                                           n_heads=n_heads,
                                           head_dim=head_dim,
                                           ff_dim=ff_dim,
                                           hidden_dim=hidden_dim,
                                           qkv_rank=qkv_rank,
                                           o_rank=o_rank,
                                           ff_rank=ff_rank,
                                            ) for _ in 1:n_blocks])
    TextNetwork(embed, tfr, embeddecoder)
end
TextTransformer(rng::AbstractRNG, counter::AbstractCounter, nt::NamedTuple) = TextTransformer(rng, counter; nt...)

function TextRNN(rng::AbstractRNG, counter::AbstractCounter; hidden_dim::Int, vocab_size::Int, σ::Function, embed_rank::Int=-1, input_rank::Int=-1, hidden_rank::Int=-1)
    """Create a text RNN with hidden_dim and vocab_size"""
    embed, embeddecoder = create_embeds(rng, counter, (hidden_dim, vocab_size), rank=embed_rank)
    rnn = RNN(rng, counter, dims=(hidden_dim, hidden_dim), σ=σ, input_rank=input_rank, hidden_rank=hidden_rank)
    TextNetwork(embed, rnn, embeddecoder)
end
TextRNN(rng::AbstractRNG, counter::AbstractCounter, nt::NamedTuple) = TextRNN(rng, counter; nt...)
