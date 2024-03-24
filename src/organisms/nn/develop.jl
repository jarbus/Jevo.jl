function get_binding(dims::NTuple{N, Int}, genes::Vector{NetworkGene}, idx::Int=length(genes))::WeightBinding where {N}
    length(genes) == 0 && error("No genes found")
    idx == 1 && return WeightBinding(dims, genes[1].seed, nothing)
    WeightBinding(dims, genes[idx].seed, genes[idx-1].seed)
end

############################
# PERFORMANCE CRITICAL START  (suspected)
function get_earliest_cached_weight(dims::NTuple{N, Int}, genes::Vector{NetworkGene}, weight_cache::_WeightCache)::Tuple{Array{Float32}, Int} where {N}
    """Return the earliest cached weight in the gene list. If none are cached, return a zero tensor of the given dimensions. Allocates memory. Also returns the idx of the earliest cached gene."""
    isnothing(weight_cache) && return zeros(Float32, dims), 0
    @inbounds @fastmath @simd for i in length(genes):-1:1
        binding = get_binding(dims, genes, i)
        weights = @inline get(weight_cache, binding, nothing)
        if !isnothing(weights)
            return copy(weights), i
        end
    end
    zeros(Float32, dims), 0
end
function tensor(w::Weights; weight_cache::_WeightCache=nothing)::Array{Float32}
    # ADD MUTS that have have not been cached
    dims, genes = w.dims, w.muts
    n_genes = length(genes)
    # get earliest cached weight or zero tensor if none found
    arr, ancestor_idx = @inline get_earliest_cached_weight(dims, genes, weight_cache)
    yes_weight_cache = !isnothing(weight_cache)
    # iteratively apply remaining mutations
    @inbounds @fastmath @simd for i in ancestor_idx+1:n_genes
        gene = genes[i]
        gid = gene.id
        rng = StableRNG(gene.seed)
        gene.init!(rng, Float32, arr, gene.mr)
        # update cache if we are using one
        if yes_weight_cache && i != n_genes && gid ∉ keys(weight_cache)
            binding = get_binding(dims, genes, i)
            weight_cache[binding] = copy(arr)
        end
    end
    arr
end

function get_factors(rng::AbstractRNG, scratch::Array, rank::Int, gene::NetworkGene, idx::Int)
    # zero-out-array
    scratch .= 0f0
    gene.init!(rng, Float32, scratch, gene.mr)
    @assert any(scratch .!= 0) "Gene did not initialize any values"
    Random.seed!(gene.seed)
    U, S, V = psvd(scratch, rank=rank)
    S_diag = diagm(0=>S)
    if idx == 1
        return U * sqrt(S_diag)
    elseif idx == 2
        return sqrt(S_diag) * V'
    else
        error("Invalid index for factorized tensor: $idx")
    end
end
function factorized_tensor(;full_dims::Tuple{Vararg{Int}},
        rank::Int,
        lora_dims::Tuple{Vararg{Int}},
        genes::Vector{NetworkGene},
        weight_cache::_WeightCache=nothing)::Array{Float32}
    """Performs psvd on full-rank matrix to generate factors"""
    @assert length(full_dims) == 2 "Only 2D tensors are supported"
    @assert length(lora_dims) == 2 "Only 2D tensors are supported"
    @assert lora_dims[1] == rank || lora_dims[2] == rank "Rank must be in the dimensions of the factorized tensor"
    if lora_dims[1] == full_dims[1] && lora_dims[2] != full_dims[2]
        idx = 1
    elseif lora_dims[2] == full_dims[2]
        idx = 2
    else
        error("Invalid dimensions for factorized tensor: lora=$lora_dims, full=$full_dims")
    end
    # ADD MUTS that have have not been cached
    n_genes = length(genes)
    # get earliest cached weight or zero tensor if none found
    arr, ancestor_idx = @inline get_earliest_cached_weight(lora_dims, genes, weight_cache)
    yes_weight_cache = !isnothing(weight_cache)
    # iteratively apply remaining mutations
    scratch = zeros(Float32, full_dims) # we use this to generate random numbers for SVD
    @inbounds @fastmath @simd for i in ancestor_idx+1:n_genes
        gene = genes[i]
        gid = gene.id
        rng = StableRNG(gene.seed)
        mut = get_factors(rng, scratch, rank, gene, idx)
        @assert size(mut) == size(arr) "size(mut)=$(size(mut)) != size(arr)=$(size(arr)), size(scratch)=$(size(scratch)) size(full_dims)=$(full_dims), size(lora_dims)=$(lora_dims), idx=$idx, rank=$rank"
        arr .+= mut
        # update cache if we are using one
        if yes_weight_cache && i != n_genes && gid ∉ keys(weight_cache)
            binding = get_binding(lora_dims, genes, i)
            weight_cache[binding] = copy(arr)
        end
    end
    arr
end

function tensor(fw::FactorWeight; weight_cache::_WeightCache=nothing)::Array{Float32}
    full_dims = (fw.A.dims[1], fw.B.dims[2])
    A = factorized_tensor(full_dims=full_dims, rank=fw.A.dims[2], lora_dims=fw.A.dims,
                          genes=fw.A.muts, weight_cache=weight_cache)
    B = factorized_tensor(full_dims=full_dims, rank=fw.B.dims[1], lora_dims=fw.B.dims,
                          genes=fw.B.muts, weight_cache=weight_cache)
    A * B
end
# PERFORMANCE CRITICAL END
############################

function create_layer(layer::Jevo.Embed; weight_cache::_WeightCache)
    weights = @inline tensor(layer.weights, weight_cache=weight_cache)
    Transformers.Embed(weights; scale=nothing) |> todevice
end

function create_layer(layer::Jevo.EmbedDecoder; weight_cache::_WeightCache)
    embed = create_layer(layer.embed, weight_cache=weight_cache)
    bias = @inline tensor(layer.bias, weight_cache=weight_cache) |> todevice
    Transformers.EmbedDecoder(embed, bias)
end

function create_layer(layer::Jevo.TransformerDecoderBlock; weight_cache::_WeightCache)
    attn_layer = Threads.@spawn create_layer(layer.attention, weight_cache=weight_cache)
    ff_layer = Threads.@spawn create_layer(layer.ff, weight_cache=weight_cache)
    wait(attn_layer)
    wait(ff_layer)
    Transformers.Layers.TransformerDecoderBlock(
        attn_layer.result,
        ff_layer.result
    )
end

# PostNormResidual
function create_layer(layer::Jevo.PostNormResidual; weight_cache::_WeightCache)
    l = create_layer(layer.layer, weight_cache=weight_cache)
    norm = create_layer(layer.norm, weight_cache=weight_cache)
    Transformers.Layers.PostNormResidual(l, norm)
end

# LayerNorm
function create_layer(layer::Jevo.LayerNorm; weight_cache::_WeightCache)
    @assert isnothing(layer.scale) && isnothing(layer.bias) "LayerNorm scale and bias must be nothing, we aren't evolving these yet"
    Transformers.Layers.LayerNorm(layer.hidden_dim)
end

# SelfAttention
function create_layer(layer::Jevo.SelfAttention; weight_cache::_WeightCache)
    Transformers.Layers.SelfAttention(
        Transformers.NeuralAttentionlib.CausalMultiheadQKVAttenOp(layer.n_heads),
        Transformers.Layers.NSplit(3, create_layer(layer.qkv, weight_cache=weight_cache)),
        create_layer(layer.out, weight_cache=weight_cache)
    )
end

function create_layer(geno_blocks::Tuple{Vararg{TransformerDecoderBlock}}; weight_cache::_WeightCache)
    pheno_blocks = Vector{Transformers.Layers.TransformerDecoderBlock}(undef, length(geno_blocks))
    Threads.@threads for i in eachindex(geno_blocks)
        pheno_blocks[i] = create_layer(geno_blocks[i], weight_cache=weight_cache)
    end
    Transformers.Transformer(Tuple(pheno_blocks)) |> todevice
end

# Chain
create_layer(layer::Jevo.Chain; weight_cache::_WeightCache) =
    Flux.Chain((create_layer(l, weight_cache=weight_cache) for l in layer.layers)...)

function create_layer(layer::Jevo.Dense; weight_cache::_WeightCache)
    weights = @inline tensor(layer.weights, weight_cache=weight_cache)
    bias = @inline tensor(layer.bias, weight_cache=weight_cache)
    Flux.Dense(weights, bias, layer.σ)
end
create_layer(f::Function; kwargs...) = f

function get_weight_cache()
    # get global variable Main.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Main, :weight_cache)
        @warn "No weight cache found, looking specifically for a variable called `weight_cache` in the global scope. Main has the following variables: $(names(Main))"
        Main.weight_cache = nothing
    end
    Main.weight_cache
end

function develop(::Creator{Model}, network::Network)
    weight_cache = get_weight_cache()
    Flux.Chain((create_layer(l, weight_cache=weight_cache) for l in network.layers)...) |> Model
end

function develop(c::Creator{TransformerPhenotype}, net::Network)
    trf = net.layers[1]
    @assert trf isa Transformer
    weight_cache = get_weight_cache()
    embed = create_layer(trf.embed, weight_cache=weight_cache)
    hidden_dim = size(embed.embeddings, 1)
    TransformerPhenotype(
        c.kwargs.textenc,
        Transformers.Layers.SinCosPositionEmbed(hidden_dim),
        embed,
        create_layer(trf.blocks, weight_cache=weight_cache),
        create_layer(trf.embeddecoder, weight_cache=weight_cache),
    )
end

function (trf::TransformerPhenotype)(input)
    mask = get(input, :attention_mask, nothing)
    embeds = trf.embed(input.token)
    pos_embed = trf.posembed(embeds) |> todevice
    embeds = embeds .+ pos_embed
    logits = trf.trf(embeds, mask)
    trf.embeddecoder(logits.hidden_state)
end
