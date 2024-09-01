############################
# PERFORMANCE CRITICAL START  (suspected)
function get_earliest_cached_weight(dims::NTuple{N, Int}, genes::Vector{NetworkGene}, weight_cache::_WeightCache)::Tuple{Array{Float32}, Int} where {N}
    """Return the earliest cached weight in the gene list. If none are cached, return a zero tensor of the given dimensions. Allocates memory. Also returns the idx of the earliest cached gene."""
    isnothing(weight_cache) && return zeros(Float32, dims), 0
    @inbounds for i in length(genes):-1:1
        weights = get(weight_cache, genes[i].id, nothing)
        if !isnothing(weights)
            @assert size(weights) == dims "Cached weight for $(genes[i].id) has different dimensions than requested"
            return (copy(weights), i)
        end
    end
    zeros(Float32, dims), 0
end
"""
    tensor(w::Weights; weight_cache::_WeightCache=nothing)::Array{Float32}
    tensor(fw::FactorWeight; weight_cache::_WeightCache=nothing)::Array{Float32}
    tensor(cw::CompositeWeight; weight_cache::_WeightCache=nothing)::Array{Float32}
    tensor(w::WeightsCollection; weight_cache::_WeightCache=nothing)::Array{Float32}

Create a tensor from a Weights object. If `weight_cache` is provided, it will used cached weights during development, and update the cache accordingly. If the cache is not provided, it will not be used.
"""
function tensor(w::Weights; weight_cache::_WeightCache=nothing)::Array{Float32}
    # ADD MUTS that have have not been cached
    dims, genes = w.dims, w.muts
    n_genes = length(genes)
    # get earliest cached weight or zero tensor if none found
    arr, ancestor_idx = @inline get_earliest_cached_weight(dims, genes, weight_cache)
    yes_weight_cache = !isnothing(weight_cache)
    # iteratively apply remaining mutations
    @inbounds for i in ancestor_idx+1:n_genes
        gene = genes[i]
        gid = gene.id
        rng = StableRNG(gene.seed)
        @fastmath gene.init!(rng, Float32, arr, gene.mr)
        # update cache if we are using one
        if yes_weight_cache && i != n_genes && gid ∉ keys(weight_cache)
            gene_id = genes[i].id
            weight_cache[gene_id] = copy(arr)
        end
    end
    arr
end
function tensor(fw::FactorWeight; weight_cache::_WeightCache=nothing)::Array{Float32}
    A = @inline tensor(fw.A, weight_cache=weight_cache)
    B = @inline tensor(fw.B, weight_cache=weight_cache)
    Transformers.tocpudevice(gpu(A) * gpu(B))
end
function tensor(cw::CompositeWeight; weight_cache::_WeightCache=nothing)::Array{Float32}
    gpu_weights = [gpu(tensor(w, weight_cache=weight_cache)) for w in cw.weights]
    reduce(+, gpu_weights) |> Transformers.tocpudevice
end
function tensor(wc::WeightsCollection; weight_cache::_WeightCache=nothing)
    @assert ndims(wc.weights) <= 2 "WeightsCollection only supports 2 or fewer dimensions, got $(ndims(wc.weights))"
    developed_weights = map(w->tensor(w, weight_cache=weight_cache), wc.weights)
    mat = zeros(Float32, wc.dims...)
    row_idx, col_idx = 1, 1
    for row in eachrow(developed_weights)
        nrows = size(row[1], 1)
        for developed_weight in row
            ncols = size(developed_weight, 2)
            mat[row_idx:row_idx+nrows-1, col_idx:col_idx+ncols-1] = developed_weight
            col_idx += ncols
        end
        row_idx += nrows
        col_idx = 1
    end
    mat
end
# PERFORMANCE CRITICAL END
############################

function create_layer(layer::Jevo.Embed; weight_cache::_WeightCache)
    weights = @inline tensor(layer.weights, weight_cache=weight_cache)
    embed = Transformers.Embed(weights; scale=nothing) |> gpu
    embed
end

function create_layer(layer::Jevo.EmbedDecoder; weight_cache::_WeightCache)
    embed = create_layer(layer.embed, weight_cache=weight_cache)
    bias = @inline tensor(layer.bias, weight_cache=weight_cache) |> gpu
    Transformers.EmbedDecoder(embed, bias)
end

function create_layer(layer::Jevo.TransformerDecoderBlock; weight_cache::_WeightCache)
    attn_layer = create_layer(layer.attention, weight_cache=weight_cache)
    #ff_layer = create_layer(layer.ff, weight_cache=weight_cache)
    ff_layer = identity
    Transformers.Layers.TransformerDecoderBlock(
        attn_layer,
        ff_layer
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
    scale = @inline tensor(layer.scale, weight_cache=weight_cache)
    bias = @inline tensor(layer.bias, weight_cache=weight_cache)
    Transformers.Layers.LayerNorm(scale, bias, Float32(1e-7))
end

# SelfAttention
function create_layer(layer::Jevo.SelfAttention; weight_cache::_WeightCache)
    Transformers.Layers.SelfAttention(
        Transformers.NeuralAttentionlib.CausalMultiheadQKVAttenOp(layer.n_heads),
        Transformers.Layers.NSplit(3, create_layer(layer.qkv, weight_cache=weight_cache)),
        create_layer(layer.out, weight_cache=weight_cache)
    )
end

function create_layer(geno_blocks::Vector{TransformerDecoderBlock}; weight_cache::_WeightCache)
    pheno_blocks =[create_layer(geno_blocks[i], weight_cache=weight_cache) for i in eachindex(geno_blocks)]
    Transformers.Transformer(Tuple(pheno_blocks)) |> gpu
end

# Chain
create_layer(layer::Jevo.Chain; weight_cache::_WeightCache) =
    Transformers.Chain((create_layer(l, weight_cache=weight_cache) for l in layer.layers)...)

function create_layer(layer::Jevo.Dense; weight_cache::_WeightCache)
    weights = @inline tensor(layer.weights, weight_cache=weight_cache)
    bias = @inline tensor(layer.bias, weight_cache=weight_cache)
    Transformers.Dense(weights, bias, layer.σ)
end
"""
    create_layer(layer::Jevo.AbstractLayer; weight_cache::_WeightCache)

Creates a phenotype layer from a genotype, calls [tensor](@ref) on contained weights.
"""
create_layer(f::Function; kwargs...) = f

function develop(::Creator{Model}, network::Network)
    weight_cache = get_weight_cache()
    Transformers.Chain((create_layer(l, weight_cache=weight_cache) for l in network.layers)...) |> Model
end

function develop(c::Creator{TransformerPhenotype}, net::Network)
    trf = net.layers[1]
    @assert trf isa Transformer "Only TransformerPhenotype can be developed, got $(typeof(trf))"
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
    pos_embed = trf.posembed(embeds) |> gpu
    embeds = embeds .+ pos_embed
    logits = Transformers.ChainRulesCore.ignore_derivatives() do
        trf.trf(embeds, mask).hidden_state
    end
    trf.embeddecoder(logits)
end
