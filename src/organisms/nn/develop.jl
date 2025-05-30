############################
# PERFORMANCE CRITICAL START  (suspected)
function get_earliest_cached_weight(dims::NTuple{N, Int}, genes::Vector{NetworkGene}, weight_cache::_WeightCache) where {N}
    """Return the earliest cached weight in the gene list. If none are cached, return a zero tensor of the given dimensions. Allocates memory. Also returns the idx of the earliest cached gene."""
    isnothing(weight_cache) && return zeros(Float32, dims), 0
    @inbounds for i in length(genes):-1:1
        weights = get(weight_cache, genes[i].id, nothing)
        if !isnothing(weights)
            @assert size(weights) == dims "Cached weight for $(genes[i].id) has different dimensions than requested"
            return (copy(weights), i)
        end
    end
    if weight_cache.currentsize > weight_cache.maxsize / 2
        error("Failed to find cache entry, but cache is more than half full. This is likely an error; consider increasing cache size.")

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
function tensor(w::Weights; weight_cache::_WeightCache=nothing)
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
            weight_cache[gene_id] = arr |> deepcopy
        end
    end
    #CUDA.synchronize()
    #gpu(arr)
    arr
end
function tensor(fw::FactorWeight; weight_cache::_WeightCache=nothing)
    A = @inline tensor(fw.A, weight_cache=weight_cache)
    B = @inline tensor(fw.B, weight_cache=weight_cache)
    A * B
end
function n_mode_product(tensor, matrix, mode)
    # Bring the specified mode to the front, reshape, multiply, and restore original shape.
    dims = size(tensor)
    order = [mode; setdiff(1:length(dims), mode)]
    permuted_tensor = permutedims(tensor, order)
    reshaped_tensor = reshape(permuted_tensor, dims[mode], :)

    # Multiply the matrix with the reshaped tensor
    result = matrix * reshaped_tensor

    # Reshape and permute back to the original order
    new_dims = (size(matrix, 1), dims[1:mode-1]..., dims[mode+1:end]...)
    result = reshape(result, new_dims)
    return permutedims(result, invperm(order))
end


function tensor(cw::CompositeWeight; weight_cache::_WeightCache=nothing)
    reduce(+, [tensor(w, weight_cache=weight_cache) for w in cw.weights])
end
function tensor(wc::WeightsCollection; weight_cache::_WeightCache=nothing)
    @assert ndims(wc.weights) <= 2 "WeightsCollection only supports 2 or fewer dimensions, got $(ndims(wc.weights))"
    developed_weights = map(w->tensor(w, weight_cache=weight_cache), wc.weights)
    mat = CUDA.zeros(Float32, wc.dims...)
    row_idx, col_idx = 1, 1
    for row in eachrow(developed_weights)
        nrows = size(row[1], 1)
        for developed_weight in row
            ncols = size(developed_weight, 2)
            mat[row_idx:row_idx+nrows-1, col_idx:col_idx+ncols-1] .= developed_weight
            col_idx += ncols
        end
        row_idx += nrows
        col_idx = 1
    end
    mat
end
# PERFORMANCE CRITICAL END
############################
function create_layer(layer::Jevo.RNN; weight_cache::_WeightCache)
    wi = @inline tensor(layer.input, weight_cache=weight_cache)
    wh = @inline tensor(layer.hidden, weight_cache=weight_cache)
    b = @inline tensor(layer.bias, weight_cache=weight_cache)
    Flux.Recur(Flux.RNNCell(layer.σ, wi, wh, b, zeros(Float32, (size(wh, 1), 1))))
end

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
    ff_layer = create_layer(layer.ff, weight_cache=weight_cache)
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
function create_layer(layer::Union{SelfAttention,JevoSelfAttention}; weight_cache::_WeightCache)
    Transformers.Layers.SelfAttention(
        Transformers.Layers.CausalMultiheadQKVAttenOp(layer.n_heads),
        #Transformers.Layers.CausalFlashMultiheadQKVAttenOp(layer.n_heads),
        Transformers.Layers.NSplit(3, create_layer(layer.qkv, weight_cache=weight_cache)),
        create_layer(layer.out, weight_cache=weight_cache)
    )
end

function create_layer(geno_blocks::Vector{TransformerDecoderBlock}; weight_cache::_WeightCache)
    pheno_blocks =[create_layer(geno_blocks[i], weight_cache=weight_cache) for i in eachindex(geno_blocks)]
    Transformers.Transformer(Tuple(pheno_blocks)) |> gpu
end
create_layer(layers::Vector; weight_cache::_WeightCache) =
    Flux.Chain((create_layer(l, weight_cache=weight_cache) for l in layers)...)
create_layer(layer::Jevo.Transformer; weight_cache::_WeightCache) = create_layer(layer.blocks, weight_cache=weight_cache)
create_layer(layer::JevoChain; weight_cache::_WeightCache) = create_layer(layer.layers, weight_cache=weight_cache)

function create_layer(layer::Jevo.Conv; weight_cache::_WeightCache)
    weights = @inline tensor(layer.weights, weight_cache=weight_cache)
    if ndims(weights) == 2
        n_in_channels = Int(size(weights, 1)  / (layer.kernel[1] * layer.kernel[2]))
        weights = reshape(weights, (layer.kernel[1], layer.kernel[2], n_in_channels, size(weights,2))) ./ 2
    end
    bias = @inline tensor(layer.bias, weight_cache=weight_cache)
    Flux.Conv(layer.σ, weights, bias, layer.stride, layer.padding, layer.dilation, 1)
end

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


function develop(::Creator{Flux.Chain}, chain::JevoChain)
    weight_cache = get_weight_cache()
    Flux.Chain((create_layer(l, weight_cache=weight_cache) for l in chain.layers)...)
end

function develop(::Creator{Model}, chain::JevoChain)
    develop(Creator(Flux.Chain), chain) |> Model
end
function (m::Model)(x...)
    Transformers.ChainRulesCore.ignore_derivatives() do
        m.chain(x...)
    end
end

function develop(c::Creator{TextModel}, textnet::TextNetwork)
    weight_cache = get_weight_cache()
    TextModel(
        c.kwargs.textenc,
        #Transformers.Layers.SinCosPositionEmbed(textnet.embed.weights.dims[1]),
        Transformers.Layers.RotaryPositionEmbed(4),
        create_layer(textnet.embed, weight_cache=weight_cache),
        create_layer(textnet.network, weight_cache=weight_cache) |> gpu,
        create_layer(textnet.embeddecoder, weight_cache=weight_cache),
    )
end

process_text_embeds(tfr::Transformers.Transformer, embeds::AbstractArray, mask) = isempty(tfr.blocks) ? embeds : tfr(embeds, mask).hidden_state
function process_text_embeds(recur::Flux.Recur, embeds::AbstractArray, _) 
    Flux.reset!(recur)
    if ndims(embeds) == 2  # If we do inference on a single sample, then embeds are 2D
        embeds = reshape(embeds, (size(embeds)..., 1))
    end
    @assert ndims(embeds) == 3 "Expected 3D tensor, got $(ndims(embeds))"
    logits = stack([recur(x) for x in eachslice(embeds, dims=2)], dims=2)
    logits
end

function (tm::TextModel)(input)
    Transformers.ChainRulesCore.ignore_derivatives() do
        mask = get(input, :attention_mask, nothing)
        embeds = tm.embed(input.token) |> tm.posembed
        last_hidden = process_text_embeds(tm.model, embeds, mask)
        tm.embeddecoder(last_hidden)
    end
end
